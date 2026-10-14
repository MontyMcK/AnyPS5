#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/ShapeReuse.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::DriverDetail {

namespace {

// The walk's address width (SrtAddressArithmetic.hpp): a relocated read outside it would have
// failed where the remembered one succeeded.
constexpr std::uint64_t AddressMask = 0x0000ffffffffffffull;
// ShaderMemory's page: its regions never cross one, so relocated regions are split the same way.
constexpr std::uint64_t PageBytes = 4096;
// ShapeProgram::classes bits: the UserWordUse flags, and a copy into the snapshot.
constexpr std::uint8_t UserWordClassPointer = UserWordUse::PointerLow;
constexpr std::uint8_t UserWordClassOpaque = UserWordUse::Opaque;
constexpr std::uint8_t UserWordClassCopy = 4u;

constexpr std::array<const char*, static_cast<std::size_t>(ShapeRefusal::Count)> RefusalNames{"unlearned", "ineligible", "no entry", "key collision", "relocation", "conflict", "differing", "inaccessible", "pending/moved", "inconsistent", "source", "materialize"};

struct ShapeTotals {
    std::atomic<std::uint64_t> lookups{0}, hits{0}, dataRefreshes{0}, descriptorRefreshes{0}, remembered{0}, notRemembered{0}, verified{0}, mismatches{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShapeRefusal::Count)> refusals{};
    std::atomic<std::uint64_t> hitNanoseconds{0}, fullCaptures{0}, fullNanoseconds{0};
};

// The 10 s window of the [shape-reuse] / [draw-shape] line: per program (address) lookups, hits,
// refusals and the key words that kept a lookup from finding an entry; why captures were not
// remembered.
struct ShapeWindow {
    struct Program {
        std::uint64_t lookups = 0, hits = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(ShapeRefusal::Count)> refusals{};
        std::map<std::uint32_t, std::uint64_t> keyWordsChanged;
        std::vector<std::uint8_t> classes;
    };
    std::map<std::uint64_t, Program> programs;
    std::map<std::string, std::uint64_t> notRemembered;
    std::uint64_t lookups = 0, hits = 0, refreshes = 0, descriptorRefreshes = 0, remembered = 0, verified = 0, mismatches = 0, fullCaptures = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ShapeRefusal::Count)> refusals{};
    double hitUs = 0, fullUs = 0;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

// One kind's counters (ShapeCore::Kind).
struct ShapeStats {
    ShapeTotals totals;
    std::mutex mutex;
    ShapeWindow window;
};

ShapeStats& stats(ShapeCore::Kind kind) {
    static std::array<ShapeStats, 2> all;
    return all[static_cast<std::size_t>(kind)];
}

ShapeTotals& shapeTotals(ShapeCore::Kind kind = ShapeCore::Kind::Dispatch) {
    return stats(kind).totals;
}

std::mutex& windowMutex(ShapeCore::Kind kind = ShapeCore::Kind::Dispatch) {
    return stats(kind).mutex;
}

ShapeWindow& shapeWindow(ShapeCore::Kind kind = ShapeCore::Kind::Dispatch) {
    return stats(kind).window;
}

bool profiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

bool insideAny(std::span<const ShaderRecompiler::MemoryRegion> regions, std::uint64_t address, std::size_t bytes) {
    return std::any_of(regions.begin(), regions.end(), [&](const auto& region) { return address < region.guestAddress + region.bytes.size() && region.guestAddress < address + bytes; });
}

bool startsInside(std::span<const ShaderRecompiler::MemoryRegion> regions, std::uint64_t address) {
    return std::any_of(regions.begin(), regions.end(), [&](const auto& region) { return address >= region.guestAddress && address < region.guestAddress + region.bytes.size(); });
}

// The dword at `address` among sorted, non-overlapping regions.
bool wordAt(std::span<const ShaderRecompiler::MemoryRegion> regions, std::uint64_t address, std::uint32_t& value) {
    auto it = std::upper_bound(regions.begin(), regions.end(), address, [](std::uint64_t wanted, const ShaderRecompiler::MemoryRegion& region) { return wanted < region.guestAddress; });
    if (it == regions.begin()) return false;
    --it;
    if (address - it->guestAddress + sizeof(value) > it->bytes.size()) return false;
    std::memcpy(&value, it->bytes.data() + (address - it->guestAddress), sizeof(value));
    return true;
}

// materializeTables' check of every buffer descriptor: a full capture of a patched descriptor
// that fails it throws, so reuse leaves that to the full capture.
bool validBuffer(const ShaderRecompiler::DescriptorValue& value) {
    ShaderRecompiler::ShaderBufferResource decoded;
    if (value.dwordCount != decoded.fields.size()) return false;
    std::copy_n(value.dwords.begin(), decoded.fields.size(), decoded.fields.begin());
    return decoded.Type() == 0u;
}

// Live `dispatch_shape_refresh_descriptors` (env APS5_DISPATCH_SHAPE_REFRESH_DESCRIPTORS=1): a
// moved word that the walk only copies into buffer descriptors (a descriptor leaf) may differ too.
bool refreshDescriptors() {
    static const bool fallback = std::getenv("APS5_DISPATCH_SHAPE_REFRESH_DESCRIPTORS") != nullptr;
    return Graphics::LiveFlag("dispatch_shape_refresh_descriptors", fallback);
}

using IrOpcode = ShaderRecompiler::IrOpcode;
using IrResourcePlan = ShaderRecompiler::IrResourcePlan;
using IrType = ShaderRecompiler::IrType;
using IrValue = ShaderRecompiler::IrValue;

// EvaluateRawRead's raw reads (SrtInstructionPredicates.cpp IsRawRead).
bool rawRead(const IrResourcePlan& plan, const IrValue& value) {
    const auto op = value.Opcode();
    if (op != IrOpcode::LoadAddressU32 && op != IrOpcode::ReadConstBuffer) return false;
    const auto index = value.Flags<ShaderRecompiler::MemoryFlags>().index;
    if (index >= plan.memoryInfo.size()) return false;
    const auto kind = plan.memoryInfo[index].kind;
    return (op == IrOpcode::LoadAddressU32 && kind == ShaderRecompiler::ResourceKind::ScalarAddress) || (op == IrOpcode::ReadConstBuffer && kind == ShaderRecompiler::ResourceKind::ScalarBuffer);
}

// The user data word a GetUserData value reads (RegIndex is the register number itself), or
// NoWord when it is not one.
std::uint32_t userWordOf(const IrResourcePlan& plan, const IrValue& value) {
    if (value.Opcode() != IrOpcode::GetUserData || value.ArgumentCount() != 1 || value.Argument(0) == nullptr) return ShapeRead::NoWord;
    const auto reg = value.Argument(0)->Resolve()->Register().index;
    return reg >= plan.userDataBase ? reg - plan.userDataBase : ShapeRead::NoWord;
}

// The driver's analysis of a plan, with the raw read nodes that have a refresh target.
struct PlanAnalysis {
    const IrResourcePlan* plan = nullptr;
    std::shared_ptr<ShapePlan> shape;
    std::unordered_map<const IrValue*, std::uint32_t> nodeTargets;
};

// The plan's use-def edges, read the way the walk (SrtEvaluator.cpp, SrtDescriptorEvaluation.cpp,
// ResourceMaterializer.cpp materializeSnapshot) consumes values. A GetUserData value is PointerLow
// where it is argument 0 of a handle whose every use is argument 0 of a raw read (EvaluateRawRead
// takes it as the low half of the base and nothing evaluates the handle itself); a copy where it is
// itself a dword of a buffer-only descriptor source outside every control-flow block (always
// materialized into ResourceSnapshot::buffers, in info.buffers order) or the value of a flat slot
// no ReadConst refers to; Opaque everywhere else. A raw read that is a slot's value is a descriptor
// leaf when nothing evaluates it except as such dwords (directly or through ReadConst of its slots)
// and the flattening of its slots, and a pure leaf when its slots are pure (pureFlatSlots). Only
// these edges are accepted, so no value derived from a word reaches the walk by another route.
std::shared_ptr<PlanAnalysis> classifyPlan(const std::shared_ptr<const IrResourcePlan>& owner) {
    auto analysis = std::make_shared<PlanAnalysis>();
    analysis->plan = owner.get();
    analysis->shape = std::make_shared<ShapePlan>();
    const auto& plan = *owner;
    auto& shape = *analysis->shape;
    const auto fail = [&](const char* reason) {
        shape = ShapePlan{};
        shape.refusal = reason;
        analysis->nodeTargets.clear();
        return analysis;
    };
    if (!plan.denseValueIds || !plan.srtPlanComplete || !plan.resourceTrackingComplete) return fail("plan incomplete");
    if (plan.requiresSpecializationMemory || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const auto& source) { return source.indirectImage.has_value(); })) return fail("indirect images");
    if (plan.uniformFill.fill.kind != ShaderRecompiler::UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size()) return fail("uniform fill");
    struct Use {
        const IrValue* user;
        std::size_t operand;
    };
    std::unordered_map<const IrValue*, std::vector<Use>> users;
    std::unordered_map<std::uint32_t, std::vector<const IrValue*>> slotReaders;
    std::vector<const IrValue*> userData;
    for (const auto& owned : plan.valueStorage) {
        const IrValue* value = &owned;
        if (value->IsIdentity()) continue;
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
            if (value->Argument(i) != nullptr) users[value->Argument(i)->Resolve()].push_back({value, i});
        }
        if (value->Opcode() == IrOpcode::ReadConst) {
            if (value->ArgumentCount() < 2 || value->Argument(1) == nullptr) return fail("unknown ReadConst");
            const IrValue* slot = value->Argument(1)->Resolve();
            if (!slot->HasImmediate() || slot->Type() != IrType::U32) return fail("unknown ReadConst");
            slotReaders[slot->ImmediateU32()].push_back(value);
        }
        if (value->Opcode() == IrOpcode::GetUserData) userData.push_back(value);
        // A masked evaluator (ReadFirstLane) evaluates its cone again under another mask, where a
        // read may land elsewhere; the plan's values are all reachable from its roots.
        if (value->Opcode() == IrOpcode::ReadFirstLane) return fail("masked evaluation");
    }
    // The values the walk evaluates without an operand edge, by kind.
    std::unordered_map<const IrValue*, std::vector<std::pair<std::uint32_t, std::uint32_t>>> descriptorRoots;
    std::unordered_set<const IrValue*> otherRoots;
    for (std::uint32_t s = 0; s < plan.descriptorSources.size(); ++s) {
        const auto& source = plan.descriptorSources[s];
        for (std::uint32_t d = 0; d < source.dwordCount && d < source.dwords.size(); ++d) {
            if (source.dwords[d] != nullptr) descriptorRoots[source.dwords[d]->Resolve()].emplace_back(s, d);
        }
    }
    std::unordered_map<const IrValue*, std::vector<std::uint32_t>> slotValues;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr) slotValues[plan.srtReads[slot].value->Resolve()].push_back(slot);
    }
    for (const auto& block : plan.controlFlow) {
        if (block.condition != nullptr) otherRoots.insert(block.condition->Resolve());
    }
    for (const auto* value : plan.uniformFill.values) {
        if (value != nullptr) otherRoots.insert(value->Resolve());
    }
    std::vector<std::vector<std::uint32_t>> bufferIndices(plan.descriptorSources.size());
    std::vector<std::uint8_t> shared(plan.descriptorSources.size(), 0u);
    std::vector<std::uint8_t> materialized(plan.descriptorSources.size(), 0u);
    for (std::uint32_t b = 0; b < plan.info.buffers.size(); ++b) {
        if (plan.info.buffers[b].source >= plan.descriptorSources.size()) return fail("unknown buffer source");
        bufferIndices[plan.info.buffers[b].source].push_back(b);
    }
    for (const auto& image : plan.info.images) {
        if (image.source < shared.size()) shared[image.source] = 1u;
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.source < shared.size()) shared[sampler.source] = 1u;
    }
    for (const auto& block : plan.controlFlow) {
        for (const auto source : block.sources) {
            if (source < shared.size()) shared[source] = 1u;
        }
    }
    for (const auto source : plan.materializationSources) {
        if (source < materialized.size()) materialized[source] = 1u;
    }
    const auto isRoot = [&](const IrValue* value) { return descriptorRoots.contains(value) || slotValues.contains(value) || otherRoots.contains(value); };
    // The buffer dwords a descriptor-root value lands in; false when any of its sources is not
    // buffer-only and always evaluated. Unevaluated sources (never materialized, never shared) are skipped.
    const auto bufferDwordsOf = [&](const IrValue* value, std::vector<std::pair<std::uint32_t, std::uint32_t>>& out) {
        const auto found = descriptorRoots.find(value);
        if (found == descriptorRoots.end()) return true;
        for (const auto& [source, dword] : found->second) {
            if (!materialized[source] && !shared[source]) continue;
            if (shared[source] || bufferIndices[source].empty()) return false;
            for (const auto b : bufferIndices[source]) out.emplace_back(b, dword);
        }
        return true;
    };
    const auto rawReadHandle = [&](const IrValue* handle) {
        if (isRoot(handle)) return false;
        const auto found = users.find(handle);
        if (found == users.end() || found->second.empty()) return false;
        return std::all_of(found->second.begin(), found->second.end(), [&](const Use& use) { return use.operand == 0 && rawRead(plan, *use.user); });
    };
    for (const auto* value : userData) {
        const auto word = userWordOf(plan, *value);
        if (word == ShapeRead::NoWord) return fail("unknown user data register");
        if (shape.uses.size() <= word) shape.uses.resize(word + 1u);
        auto& use = shape.uses[word];
        if (const auto found = users.find(value); found != users.end()) {
            for (const auto& edge : found->second) {
                if (edge.operand == 0 && rawReadHandle(edge.user)) use.flags |= UserWordUse::PointerLow;
                else use.flags |= UserWordUse::Opaque;
            }
        }
        if (!bufferDwordsOf(value, use.bufferDwords)) use.flags |= UserWordUse::Opaque;
        if (const auto found = slotValues.find(value); found != slotValues.end()) {
            for (const auto slot : found->second) {
                if (slotReaders.contains(slot)) use.flags |= UserWordUse::Opaque;
                else use.flatOffsets.push_back(plan.srtReads[slot].flatOffset);
            }
        }
        if (otherRoots.contains(value)) use.flags |= UserWordUse::Opaque;
    }
    if (shape.uses.size() < plan.userDataCount) shape.uses.resize(plan.userDataCount);
    // Refresh targets of the raw reads that are slot values.
    for (const auto& [value, slots] : slotValues) {
        if (!rawRead(plan, *value) || otherRoots.contains(value)) continue;
        if (const auto found = users.find(value); found != users.end() && !found->second.empty()) continue;
        RefreshTarget target;
        const bool pure = std::all_of(slots.begin(), slots.end(), [&](std::uint32_t slot) { return plan.srtReads[slot].flatOffset < plan.pureFlatSlots.size() && plan.pureFlatSlots[plan.srtReads[slot].flatOffset] != 0u; });
        bool leaf = true;
        for (const auto slot : slots) {
            target.flatOffsets.push_back(plan.srtReads[slot].flatOffset);
            const auto readers = slotReaders.find(slot);
            if (readers == slotReaders.end()) continue;
            for (const auto* reader : readers->second) {
                const auto readerUses = users.find(reader);
                if ((readerUses != users.end() && !readerUses->second.empty()) || slotValues.contains(reader) || otherRoots.contains(reader) || !bufferDwordsOf(reader, target.bufferDwords)) leaf = false;
            }
        }
        if (!bufferDwordsOf(value, target.bufferDwords)) leaf = false;
        if (pure && slots.size() == 1u && target.bufferDwords.empty()) {
            target.descriptor = false;
        } else if (leaf && !pure) {
            target.descriptor = true;
        } else {
            continue;
        }
        analysis->nodeTargets.emplace(value, static_cast<std::uint32_t>(shape.targets.size()));
        shape.targets.push_back(std::move(target));
    }
    shape.known = true;
    return analysis;
}

// The analysis of a plan, made once per plan. The cache does not keep plans alive: an entry whose
// plan expired (its address may be reused) is made again, and expired entries are dropped as the
// cache grows.
std::shared_ptr<const PlanAnalysis> analysisFor(const std::shared_ptr<const IrResourcePlan>& plan) {
    struct Entry {
        std::weak_ptr<const IrResourcePlan> plan;
        std::shared_ptr<const PlanAnalysis> analysis;
    };
    static std::mutex mutex;
    static std::unordered_map<const IrResourcePlan*, Entry> analyses;
    std::lock_guard lock(mutex);
    if (analyses.size() >= 4096) std::erase_if(analyses, [](const auto& item) { return item.second.plan.expired(); });
    auto& entry = analyses[plan.get()];
    if (entry.analysis == nullptr || entry.plan.lock() != plan) entry = {plan, classifyPlan(plan)};
    return entry.analysis;
}

// The raw reads a successful walk of `plan` made, recomputed from the user data and the words the
// capture read (`regions`, the registered regions included), without the walk. Each raw read
// reachable from the plan's roots must be one this evaluation reaches through operations that
// evaluate every argument (SrtEvaluator.cpp EvaluateInst) from a flat slot value, which the
// flattening evaluates in every successful walk: then the reads are exactly the walk's, each read
// once per node, at an address that no mask or evaluator changes. Null `refusal` on success; `leaves`
// are (flat offset, read index) of the pure-slot leaves in slot order.
struct AttributedReads {
    std::vector<ShapeRead> reads;
    std::vector<std::pair<std::uint32_t, std::size_t>> leaves;
    const char* refusal = nullptr;
};

AttributedReads attributeReads(const PlanAnalysis& analysis, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, std::span<const ShaderRecompiler::MemoryRegion> regions) {
    const auto& plan = *analysis.plan;
    AttributedReads out;
    std::unordered_map<const IrValue*, std::uint64_t> values;
    std::unordered_map<const IrValue*, std::size_t> readIndex;
    std::vector<const IrValue*> visiting;
    const std::function<bool(const IrValue*, std::uint64_t&)> evaluate = [&](const IrValue* raw, std::uint64_t& result) -> bool {
        if (raw == nullptr) return false;
        const IrValue* value = raw->Resolve();
        if (value->HasImmediate()) {
            const auto bits = value->ImmediateU64();
            switch (value->Type()) {
                case IrType::Bool: result = bits != 0u ? 1u : 0u; return true;
                case IrType::U8: result = static_cast<std::uint8_t>(bits); return true;
                case IrType::U16: result = static_cast<std::uint16_t>(bits); return true;
                case IrType::U32: result = static_cast<std::uint32_t>(bits); return true;
                case IrType::U64: result = bits; return true;
                default: return false;
            }
        }
        if (const auto found = values.find(value); found != values.end()) {
            result = found->second;
            return true;
        }
        if (std::find(visiting.begin(), visiting.end(), value) != visiting.end()) return false;
        visiting.push_back(value);
        struct Pop {
            std::vector<const IrValue*>& stack;
            ~Pop() { stack.pop_back(); }
        } pop{visiting};
        std::uint64_t a = 0, b = 0, c = 0;
        const auto binary = [&] { return value->ArgumentCount() >= 2 && evaluate(value->Argument(0), a) && evaluate(value->Argument(1), b); };
        bool done = false;
        switch (value->Opcode()) {
            case IrOpcode::GetUserData: {
                const auto word = userWordOf(plan, *value);
                if (word == ShapeRead::NoWord || word >= userData.size()) return false;
                result = userData[word];
                done = true;
                break;
            }
            case IrOpcode::GetShaderBase: result = shaderBase; done = true; break;
            case IrOpcode::BitCastU32F32:
            case IrOpcode::BitCastF32U32: done = value->ArgumentCount() >= 1 && evaluate(value->Argument(0), result); break;
            case IrOpcode::CompositeConstructU64: done = binary(); result = static_cast<std::uint32_t>(a) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)) << 32u); break;
            case IrOpcode::IAdd32: done = binary(); result = static_cast<std::uint32_t>(a + b); break;
            case IrOpcode::IAdd64: done = binary(); result = a + b; break;
            case IrOpcode::ISub32: done = binary(); result = static_cast<std::uint32_t>(a - b); break;
            case IrOpcode::ISub64: done = binary(); result = a - b; break;
            case IrOpcode::IMul32: done = binary(); result = static_cast<std::uint32_t>(a * b); break;
            case IrOpcode::IMul64: done = binary(); result = a * b; break;
            case IrOpcode::BitwiseAnd32: done = binary(); result = static_cast<std::uint32_t>(a & b); break;
            case IrOpcode::BitwiseAnd64: done = binary(); result = a & b; break;
            case IrOpcode::BitwiseOr32: done = binary(); result = static_cast<std::uint32_t>(a | b); break;
            case IrOpcode::BitwiseXor32: done = binary(); result = static_cast<std::uint32_t>(a ^ b); break;
            case IrOpcode::ShiftLeftLogical32: done = binary(); result = static_cast<std::uint32_t>(a) << (b & 31u); break;
            case IrOpcode::ShiftLeftLogical64: done = binary(); result = a << (b & 63u); break;
            case IrOpcode::ShiftRightLogical32: done = binary(); result = static_cast<std::uint32_t>(a) >> (b & 31u); break;
            case IrOpcode::ShiftRightLogical64: done = binary(); result = a >> (b & 63u); break;
            case IrOpcode::SelectU32:
                done = value->ArgumentCount() >= 3 && evaluate(value->Argument(0), a) && evaluate(value->Argument(1), b) && evaluate(value->Argument(2), c);
                result = a != 0u ? b : c;
                break;
            case IrOpcode::ReadConst: {
                const IrValue* slot = value->ArgumentCount() >= 2 && value->Argument(1) != nullptr ? value->Argument(1)->Resolve() : nullptr;
                if (slot == nullptr || !slot->HasImmediate() || slot->Type() != IrType::U32 || slot->ImmediateU32() >= plan.srtReads.size()) return false;
                done = evaluate(plan.srtReads[slot->ImmediateU32()].value, result);
                break;
            }
            case IrOpcode::LoadAddressU32:
            case IrOpcode::ReadConstBuffer: {
                if (!rawRead(plan, *value) || value->ArgumentCount() < 2 || value->Argument(0) == nullptr) return false;
                // EvaluateRawRead's address arithmetic.
                const auto& memory = plan.memoryInfo[value->Flags<ShaderRecompiler::MemoryFlags>().index];
                const IrValue* handle = value->Argument(0)->Resolve();
                std::uint64_t low = 0, high = 0, offset = 0;
                if (handle->ArgumentCount() < 2 || !evaluate(handle->Argument(0), low) || !evaluate(handle->Argument(1), high) || !evaluate(value->Argument(1), offset)) return false;
                const auto base = ((high << 32u) | static_cast<std::uint32_t>(low)) & AddressMask;
                const auto immediate = static_cast<std::int64_t>(static_cast<std::int32_t>(memory.offset));
                std::uint64_t address = 0;
                if (value->Opcode() == IrOpcode::ReadConstBuffer) {
                    std::uint64_t records = 0, word3 = 0;
                    if (handle->ArgumentCount() != 4u || !evaluate(handle->Argument(2), records) || !evaluate(handle->Argument(3), word3) || immediate < 0) return false;
                    const auto byteOffset = static_cast<std::uint64_t>(immediate) + static_cast<std::uint32_t>(offset);
                    const auto aligned = byteOffset & ~std::uint64_t{3};
                    const auto stride = (static_cast<std::uint32_t>(high) >> 16u) & 0x3fffu;
                    const auto size = stride == 0u ? static_cast<std::uint64_t>(static_cast<std::uint32_t>(records)) : static_cast<std::uint64_t>(stride) * static_cast<std::uint32_t>(records);
                    if (aligned > size || size - aligned < sizeof(std::uint32_t)) return false;
                    address = ((base & ~std::uint64_t{3}) + byteOffset) & ~std::uint64_t{3};
                } else {
                    const auto relative = (immediate & ~std::int64_t{3}) + static_cast<std::int64_t>(static_cast<std::uint32_t>(offset) & ~3u);
                    const auto aligned = static_cast<std::int64_t>(base & ~std::uint64_t{3});
                    if (aligned + relative < 0 || static_cast<std::uint64_t>(aligned + relative) > AddressMask) return false;
                    address = static_cast<std::uint64_t>(aligned + relative);
                }
                std::uint32_t word = 0;
                if (!wordAt(regions, address, word)) return false;
                const IrValue* lowValue = handle->Argument(0) != nullptr ? handle->Argument(0)->Resolve() : nullptr;
                ShapeRead read;
                read.address = address;
                read.value = word;
                read.userWord = lowValue != nullptr ? userWordOf(plan, *lowValue) : ShapeRead::NoWord;
                if (const auto target = analysis.nodeTargets.find(value); target != analysis.nodeTargets.end()) read.target = target->second;
                readIndex.emplace(value, out.reads.size());
                out.reads.push_back(read);
                result = word;
                done = true;
                break;
            }
            default: return false;
        }
        if (!done) return false;
        values.emplace(value, result);
        return true;
    };
    // Every flat slot value is evaluated by a successful walk.
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        std::uint64_t ignored = 0;
        const bool evaluated = evaluate(plan.srtReads[slot].value, ignored);
        const IrValue* value = plan.srtReads[slot].value != nullptr ? plan.srtReads[slot].value->Resolve() : nullptr;
        const auto flatOffset = plan.srtReads[slot].flatOffset;
        if (evaluated && value != nullptr && readIndex.contains(value) && flatOffset < plan.pureFlatSlots.size() && plan.pureFlatSlots[flatOffset] != 0u) {
            if (std::any_of(out.leaves.begin(), out.leaves.end(), [&](const auto& leaf) { return leaf.second == readIndex[value]; })) {
                out.refusal = "a leaf shared by slots";
                return out;
            }
            out.reads[readIndex[value]].leafSlot = flatOffset;
            out.leaves.emplace_back(flatOffset, readIndex[value]);
        }
    }
    // Every raw read reachable from a root must be one of those.
    std::unordered_set<const IrValue*> seen;
    std::vector<const IrValue*> pending;
    for (const auto& source : plan.descriptorSources) {
        for (std::uint32_t d = 0; d < source.dwordCount && d < source.dwords.size(); ++d) pending.push_back(source.dwords[d]);
    }
    for (const auto& read : plan.srtReads) pending.push_back(read.value);
    for (const auto& block : plan.controlFlow) pending.push_back(block.condition);
    for (const auto* value : plan.uniformFill.values) pending.push_back(value);
    while (!pending.empty()) {
        const IrValue* raw = pending.back();
        pending.pop_back();
        if (raw == nullptr) continue;
        const IrValue* value = raw->Resolve();
        if (!seen.insert(value).second) continue;
        if (rawRead(plan, *value) && !readIndex.contains(value)) {
            out.refusal = "reads the driver cannot attribute";
            return out;
        }
        if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
            const IrValue* slot = value->Argument(1)->Resolve();
            if (slot->HasImmediate() && slot->ImmediateU32() < plan.srtReads.size()) pending.push_back(plan.srtReads[slot->ImmediateU32()].value);
        }
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) pending.push_back(value->Argument(i));
    }
    return out;
}

}

namespace {

bool sameBindings(const ShaderRecompiler::RecompileResult& a, const ShaderRecompiler::RecompileResult& b) {
    if (a.bindings.size() != b.bindings.size()) return false;
    for (std::size_t i = 0; i < a.bindings.size(); ++i) {
        const auto& x = a.bindings[i];
        const auto& y = b.bindings[i];
        if (x.kind != y.kind || x.role != y.role || x.descriptorSet != y.descriptorSet || x.binding != y.binding || x.count != y.count || x.guestDescriptor != y.guestDescriptor || x.readOnly != y.readOnly || x.imageShape != y.imageShape) return false;
        if (x.samplerDepthCompare != y.samplerDepthCompare || x.imageWritten != y.imageWritten || x.imageDepthCompare != y.imageDepthCompare || x.imageAtomic != y.imageAtomic || x.imageAtomic64 != y.imageAtomic64) return false;
        if (x.bufferAtomic != y.bufferAtomic || x.bufferWritten != y.bufferWritten || x.samplerUnnormalized != y.samplerUnnormalized || x.imageUnnormalized != y.imageUnnormalized || x.imageSamplers != y.imageSamplers || x.bufferRead != y.bufferRead || x.bufferCovers != y.bufferCovers) return false;
    }
    return true;
}

void countNotRemembered(ShapeCore::Kind kind, const char* reason) {
    ++shapeTotals(kind).notRemembered;
    if (!profiled()) return;
    std::lock_guard windowLock(windowMutex(kind));
    ++shapeWindow(kind).notRemembered[reason];
}

}

namespace ShapeCore {

bool Profiled() {
    return profiled();
}

Learned Learn(const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, bool capturedCalls, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, const std::function<std::vector<ShaderRecompiler::MemoryRegion>()>& readRegions, Regions registered) {
    Learned out;
    const auto analysis = analysisFor(capture->plan);
    const auto& plan = *analysis->shape;
    out.plan = analysis->shape;
    out.learnable = plan.known && !capturedCalls;
    out.refusal = !plan.known ? plan.refusal : capturedCalls ? "user-data calls" : nullptr;
    if (out.refusal != nullptr) return out;
    auto regions = readRegions();
    std::sort(regions.begin(), regions.end(), [](const auto& a, const auto& b) { return a.guestAddress < b.guestAddress; });
    auto attributed = attributeReads(*analysis, userData, shaderBase, regions);
    out.refusal = attributed.refusal;
    if (out.refusal == nullptr) {
        // Every word the capture read from guest memory must be one of the attributed reads (a
        // callee or table read the plan does not explain would not move with the pointers).
        std::vector<std::uint64_t> pageWords;
        for (const auto& region : regions) {
            if (startsInside(registered, region.guestAddress)) continue;
            for (std::size_t offset = 0; offset < region.bytes.size(); offset += sizeof(std::uint32_t)) pageWords.push_back(region.guestAddress + offset);
        }
        std::sort(pageWords.begin(), pageWords.end());
        std::vector<std::uint64_t> traced;
        traced.reserve(attributed.reads.size());
        for (const auto& read : attributed.reads) {
            if (!insideAny(registered, read.address, sizeof(std::uint32_t))) traced.push_back(read.address);
        }
        std::sort(traced.begin(), traced.end());
        traced.erase(std::unique(traced.begin(), traced.end()), traced.end());
        if (traced != pageWords) out.refusal = "reads outside the plan";
    }
    // The walk's own trace, where it kept one, must name the same reads.
    const auto& trace = capture->readTrace;
    if (out.refusal == nullptr && (!trace.leaves.empty() || !trace.otherReads.empty())) {
        std::vector<std::pair<std::uint32_t, std::uint64_t>> leaves;
        std::vector<std::uint64_t> other;
        std::vector<std::uint8_t> leaf(attributed.reads.size(), 0u);
        for (const auto& [slot, index] : attributed.leaves) {
            leaves.emplace_back(slot, attributed.reads[index].address);
            leaf[index] = 1u;
        }
        for (std::size_t r = 0; r < attributed.reads.size(); ++r) {
            if (leaf[r] == 0u) other.push_back(attributed.reads[r].address);
        }
        std::sort(other.begin(), other.end());
        other.erase(std::unique(other.begin(), other.end()), other.end());
        if (leaves != trace.leaves || other != trace.otherReads) out.refusal = "trace disagrees";
    }
    out.reads = std::move(attributed.reads);
    out.leaves = std::move(attributed.leaves);
    return out;
}

std::uint64_t Key(std::span<const std::uint8_t> excluded, std::uint64_t staticKey, std::span<const std::uint32_t> userData) {
    std::uint64_t key = staticKey;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(userData.size());
    for (std::size_t i = 0; i < userData.size(); ++i) {
        if (i < excluded.size() && excluded[i] != 0u) continue;
        mix(userData[i]);
    }
    return key;
}

ShapeRefusal Relocate(const ShapeCandidate& candidate, std::span<const std::uint32_t> userData, Regions registered, std::vector<std::uint8_t>& changed, std::vector<std::uint64_t>& addresses) {
    const auto& uses = candidate.plan->uses;
    if (candidate.userData.size() != userData.size() || !candidate.plan->known) return ShapeRefusal::KeyCollision;
    // The difference of each moved pointer word, as the walk applies it: the base is
    // (high << 32 | low) with the low two bits dropped, so the high half stays and only the
    // aligned low half moves.
    std::vector<std::int64_t> delta(userData.size(), 0);
    changed.assign(userData.size(), 0u);
    for (std::size_t i = 0; i < userData.size(); ++i) {
        const auto before = candidate.userData[i];
        if (before == userData[i]) continue;
        changed[i] = 1u;
        const auto* use = i < uses.size() ? &uses[i] : nullptr;
        if (use == nullptr) continue;
        if ((use->flags & UserWordUse::Opaque) != 0u) return ShapeRefusal::KeyCollision;
        if ((use->flags & UserWordUse::PointerLow) != 0u) delta[i] = static_cast<std::int64_t>(userData[i] & ~3u) - static_cast<std::int64_t>(before & ~3u);
    }
    addresses.assign(candidate.reads.size(), 0u);
    for (std::size_t r = 0; r < candidate.reads.size(); ++r) {
        const auto& read = candidate.reads[r];
        std::int64_t shift = 0;
        if (read.userWord != ShapeRead::NoWord) {
            if (read.userWord >= userData.size()) return ShapeRefusal::Inconsistent;
            if (changed[read.userWord] != 0u) {
                const auto* use = read.userWord < uses.size() ? &uses[read.userWord] : nullptr;
                if (use == nullptr || (use->flags & UserWordUse::PointerLow) == 0u) return ShapeRefusal::Inconsistent;
                shift = delta[read.userWord];
            }
        }
        const auto target = static_cast<std::int64_t>(read.address) + shift;
        if (target < 0 || static_cast<std::uint64_t>(target) > AddressMask - (sizeof(std::uint32_t) - 1u)) return ShapeRefusal::Relocation;
        addresses[r] = static_cast<std::uint64_t>(target);
        if (shift != 0 && insideAny(registered, addresses[r], sizeof(std::uint32_t))) return ShapeRefusal::Relocation;
    }
    return ShapeRefusal::Count;
}

bool Refreshable(const ShapeCandidate& candidate, const ShapeRead& read, bool descriptors) {
    const auto& targets = candidate.plan->targets;
    return read.target < targets.size() && (!targets[read.target].descriptor || descriptors);
}

ShapeRefusal Rebuild(const ShapeCandidate& candidate, std::span<const std::uint32_t> userData, std::span<const std::uint8_t> changed, std::span<const std::uint64_t> addresses, std::span<const std::pair<std::uint64_t, std::uint32_t>> refreshed, bool descriptors, std::shared_ptr<ShaderRecompiler::ResourceCapture>& out, bool& descriptorRefreshed) {
    descriptorRefreshed = false;
    const auto& plan = *candidate.capture->plan;
    const auto& uses = candidate.plan->uses;
    const auto& targets = candidate.plan->targets;
    if (changed.size() != userData.size() || addresses.size() != candidate.reads.size()) return ShapeRefusal::Inconsistent;
    // The snapshot a full walk would build: the remembered one with the new user data, the
    // copied words replaced and the refreshed leaves' targets updated.
    auto rebuilt = std::make_shared<ShaderRecompiler::ResourceCapture>(*candidate.capture);
    auto& resources = rebuilt->snapshot;
    if (resources.userData.size() > userData.size()) return ShapeRefusal::Inconsistent;
    resources.userData.assign(userData.begin(), userData.begin() + static_cast<std::ptrdiff_t>(resources.userData.size()));
    for (std::size_t i = 0; i < userData.size(); ++i) {
        if (changed[i] == 0u || i >= uses.size()) continue;
        for (const auto& [buffer, dword] : uses[i].bufferDwords) {
            if (buffer >= resources.buffers.size() || dword >= resources.buffers[buffer].dwordCount || resources.buffers[buffer].dwords[dword] != candidate.userData[i]) return ShapeRefusal::Inconsistent;
            resources.buffers[buffer].dwords[dword] = userData[i];
            if (!validBuffer(resources.buffers[buffer])) return ShapeRefusal::Inconsistent;
        }
        for (const auto offset : uses[i].flatOffsets) {
            if (offset >= resources.flattenedSrt.size() || resources.flattenedSrt[offset] != candidate.userData[i]) return ShapeRefusal::Inconsistent;
            resources.flattenedSrt[offset] = userData[i];
        }
    }
    // The refreshed words: every snapshot word each refreshable read at that address copies its
    // value into (a pure leaf's flat slot; a descriptor leaf's buffer dwords and slots).
    for (const auto& [address, value] : refreshed) {
        bool any = false;
        for (std::size_t r = 0; r < candidate.reads.size(); ++r) {
            const auto& read = candidate.reads[r];
            if (addresses[r] != address) continue;
            any = true;
            if (!Refreshable(candidate, read, descriptors)) return ShapeRefusal::Inconsistent;
            const auto& target = targets[read.target];
            for (const auto offset : target.flatOffsets) {
                if (offset >= resources.flattenedSrt.size() || resources.flattenedSrt[offset] != read.value) return ShapeRefusal::Inconsistent;
                resources.flattenedSrt[offset] = value;
            }
            for (const auto& [buffer, dword] : target.bufferDwords) {
                if (buffer >= resources.buffers.size() || dword >= resources.buffers[buffer].dwordCount || resources.buffers[buffer].dwords[dword] != read.value) return ShapeRefusal::Inconsistent;
                resources.buffers[buffer].dwords[dword] = value;
                if (!validBuffer(resources.buffers[buffer])) return ShapeRefusal::Inconsistent;
            }
            if (target.descriptor) descriptorRefreshed = true;
        }
        if (!any) return ShapeRefusal::Inconsistent;
    }
    // The read trace a full walk would record (materializeCapture traces only plans with pure
    // slots): the leaves in walk order, every other read sorted and unique.
    auto& readTrace = rebuilt->readTrace;
    if (std::any_of(plan.pureFlatSlots.begin(), plan.pureFlatSlots.end(), [](std::uint8_t pure) { return pure != 0u; })) {
        readTrace.leaves.clear();
        readTrace.otherReads.clear();
        std::vector<std::uint8_t> leaf(candidate.reads.size(), 0u);
        for (const auto& [slot, index] : candidate.leaves) {
            if (index >= candidate.reads.size()) return ShapeRefusal::Inconsistent;
            readTrace.leaves.emplace_back(slot, addresses[index]);
            leaf[index] = 1u;
        }
        for (std::size_t r = 0; r < candidate.reads.size(); ++r) {
            if (leaf[r] == 0u) readTrace.otherReads.push_back(addresses[r]);
        }
        std::sort(readTrace.otherReads.begin(), readTrace.otherReads.end());
        readTrace.otherReads.erase(std::unique(readTrace.otherReads.begin(), readTrace.otherReads.end()), readTrace.otherReads.end());
    } else if (!readTrace.leaves.empty() || !readTrace.otherReads.empty()) {
        return ShapeRefusal::Inconsistent;
    }
    out = std::move(rebuilt);
    return ShapeRefusal::Count;
}

const char* Difference(const ShaderRecompiler::RecompileResult& reused, const ShaderRecompiler::RecompileResult& full, const ShaderRecompiler::ResourceCapture& reusedCapture, const ShaderRecompiler::ResourceCapture& fullCapture, Regions reusedRegions, Regions fullRegions) {
    if (reusedCapture.plan != fullCapture.plan) return "the plan";
    const auto& x = reusedCapture.snapshot;
    const auto& y = fullCapture.snapshot;
    if (x.buffers != y.buffers) return "the buffer descriptors";
    if (x.images != y.images) return "the image descriptors";
    if (x.samplers != y.samplers) return "the sampler descriptors";
    if (x.flattenedSrt != y.flattenedSrt) return "the flattened SRT";
    if (x.userData != y.userData) return "the snapshot user data";
    if (!(x.uniformFill == y.uniformFill)) return "the uniform fill";
    if (reusedCapture.readTrace.leaves != fullCapture.readTrace.leaves || reusedCapture.readTrace.otherReads != fullCapture.readTrace.otherReads) return "the read trace";
    if (reusedRegions.size() != fullRegions.size()) return "the captured regions";
    for (std::size_t i = 0; i < reusedRegions.size(); ++i) {
        if (reusedRegions[i].guestAddress != fullRegions[i].guestAddress || reusedRegions[i].bytes.size() != fullRegions[i].bytes.size()) return "the captured regions";
        if (std::memcmp(reusedRegions[i].bytes.data(), fullRegions[i].bytes.data(), reusedRegions[i].bytes.size()) != 0) return "the captured bytes";
    }
    if (reused.variantId != full.variantId || reused.specializationId != full.specializationId || reused.specialization != full.specialization) return "the compiled variant";
    if (!(reused.spirv == full.spirv)) return "the SPIR-V";
    if (reused.pushConstants != full.pushConstants) return "the push constants";
    if (!sameBindings(reused, full)) return "the bindings";
    return nullptr;
}


void Table::touch(Entry& entry) {
    order.splice(order.begin(), order, entry.order);
}

// Called with the mutex held after an insert: the least recently used shape keys past maxKeys go,
// and every 256th remember drops the candidates whose shader expired (a replaced shader's capture
// is never reused, and would keep its plan and source alive) and the programs not used for a
// while past 4 * maxKeys.
void Table::trim() {
    while (entries.size() > maxKeys && !order.empty()) {
        entries.erase(order.back());
        order.pop_back();
    }
    if (++remembers % 256u != 0u) return;
    for (auto it = entries.begin(); it != entries.end();) {
        auto& list = it->second.candidates;
        std::erase_if(list, [](const std::shared_ptr<const ShapeCandidate>& candidate) { return candidate->shader.expired(); });
        if (!list.empty()) {
            ++it;
            continue;
        }
        order.erase(it->second.order);
        it = entries.erase(it);
    }
    const auto maxPrograms = 4u * maxKeys;
    if (programs.size() <= maxPrograms) return;
    // The older half of the clock's span goes (programs touched since stay).
    const auto keepFrom = clock > maxPrograms / 2u ? clock - maxPrograms / 2u : 0u;
    std::erase_if(programs, [&](const auto& item) { return item.second.touched < keepFrom; });
}

bool Table::ShouldLearn(std::uint64_t staticKey) {
    if (!learnBackoff) return true;
    std::lock_guard lock(mutex);
    const auto program = programs.find(staticKey);
    if (program == programs.end() || (program->second.refusedInARow < 4u && program->second.noEntryInARow < 8u)) return true;
    return ++program->second.skipped % 64u == 0u;
}

void Table::Remember(Kind kind, std::uint64_t address, std::uint64_t staticKey, std::span<const std::uint32_t> userData, Learned learned, std::shared_ptr<ShapeCandidate> candidate) {
    static const std::vector<UserWordUse> none;
    const auto& uses = learned.plan != nullptr ? learned.plan->uses : none;
    const bool learnable = learned.learnable;
    std::lock_guard lock(mutex);
    auto& program = programs[staticKey];
    program.touched = ++clock;
    program.address = address;
    program.lastUserData.assign(userData.begin(), userData.end());
    program.known = learnable;
    program.refusal = learnable ? nullptr : learned.refusal;
    program.excluded.assign(learnable ? userData.size() : 0u, 0u);
    program.classes.assign(learnable ? userData.size() : 0u, 0u);
    for (std::size_t i = 0; learnable && i < userData.size(); ++i) {
        const auto* use = i < uses.size() ? &uses[i] : nullptr;
        program.excluded[i] = use == nullptr || (use->flags & UserWordUse::Opaque) == 0u ? 1u : 0u;
        if (use != nullptr) program.classes[i] = static_cast<std::uint8_t>(use->flags | (!use->bufferDwords.empty() || !use->flatOffsets.empty() ? UserWordClassCopy : 0u));
    }
    if (learned.refusal != nullptr) {
        if (learnBackoff) ++program.refusedInARow;
        countNotRemembered(kind, learned.refusal);
        return;
    }
    program.refusedInARow = 0;
    program.skipped = 0;
    candidate->staticKey = staticKey;
    candidate->userData.assign(userData.begin(), userData.end());
    candidate->plan = learned.plan;
    candidate->reads = std::move(learned.reads);
    candidate->leaves = std::move(learned.leaves);
    const auto key = Key(program.excluded, staticKey, userData);
    auto found = entries.find(key);
    if (found == entries.end()) {
        order.push_front(key);
        found = entries.emplace(key, Entry{{}, order.begin()}).first;
    } else {
        touch(found->second);
    }
    auto& list = found->second.candidates;
    list.insert(list.begin(), std::move(candidate));
    if (list.size() > maxCandidates) list.pop_back();
    trim();
    ++shapeTotals(kind).remembered;
    if (profiled()) {
        std::lock_guard windowLock(windowMutex(kind));
        ++shapeWindow(kind).remembered;
    }
}

ShapeRefusal Table::Find(Kind kind, std::uint64_t address, std::uint64_t staticKey, std::span<const std::uint32_t> userData, std::vector<std::shared_ptr<const ShapeCandidate>>& candidates) {
    candidates.clear();
    std::lock_guard lock(mutex);
    const auto program = programs.find(staticKey);
    if (program == programs.end()) {
        CountRefusal(kind, address, ShapeRefusal::Unlearned);
        return ShapeRefusal::Unlearned;
    }
    program->second.touched = ++clock;
    if (!program->second.known) {
        CountRefusal(kind, address, ShapeRefusal::Ineligible);
        return ShapeRefusal::Ineligible;
    }
    const auto found = entries.find(Key(program->second.excluded, staticKey, userData));
    if (found == entries.end() || found->second.candidates.empty()) {
        if (learnBackoff) ++program->second.noEntryInARow;
        CountRefusal(kind, address, ShapeRefusal::NoEntry);
        if (profiled()) {
            std::lock_guard windowLock(windowMutex(kind));
            auto& entry = shapeWindow(kind).programs[address];
            entry.classes = program->second.classes;
            const auto& last = program->second.lastUserData;
            for (std::size_t i = 0; i < userData.size() && i < last.size(); ++i) {
                if (userData[i] != last[i] && (i >= program->second.excluded.size() || program->second.excluded[i] == 0u)) ++entry.keyWordsChanged[static_cast<std::uint32_t>(i)];
            }
        }
        return ShapeRefusal::NoEntry;
    }
    program->second.noEntryInARow = 0;
    touch(found->second);
    candidates = found->second.candidates;
    return ShapeRefusal::Count;
}

void CountLookup(Kind kind, std::uint64_t address) {
    ++shapeTotals(kind).lookups;
    if (!profiled()) return;
    std::lock_guard lock(windowMutex(kind));
    ++shapeWindow(kind).lookups;
    ++shapeWindow(kind).programs[address].lookups;
}

void CountRefusal(Kind kind, std::uint64_t address, ShapeRefusal reason) {
    ++shapeTotals(kind).refusals[static_cast<std::size_t>(reason)];
    if (!profiled()) return;
    std::lock_guard lock(windowMutex(kind));
    auto& window = shapeWindow(kind);
    ++window.refusals[static_cast<std::size_t>(reason)];
    ++window.programs[address].refusals[static_cast<std::size_t>(reason)];
}

void CountHitTotals(Kind kind, std::uint64_t nanoseconds, bool refreshed, bool descriptorRefreshed) {
    auto& totals = shapeTotals(kind);
    ++totals.hits;
    totals.hitNanoseconds += nanoseconds;
    if (refreshed) ++totals.dataRefreshes;
    if (descriptorRefreshed) ++totals.descriptorRefreshes;
}

void CountHitWindow(Kind kind, std::uint64_t address, std::uint64_t nanoseconds, bool refreshed, bool descriptorRefreshed) {
    if (!profiled()) return;
    std::lock_guard lock(windowMutex(kind));
    auto& window = shapeWindow(kind);
    ++window.hits;
    ++window.programs[address].hits;
    if (refreshed) ++window.refreshes;
    if (descriptorRefreshed) ++window.descriptorRefreshes;
    window.hitUs += static_cast<double>(nanoseconds) / 1000.0;
}

std::uint64_t CountVerified(Kind kind, bool mismatch) {
    auto& totals = shapeTotals(kind);
    ++totals.verified;
    const std::uint64_t mismatches = mismatch ? ++totals.mismatches : totals.mismatches.load();
    if (profiled()) {
        std::lock_guard lock(windowMutex(kind));
        ++shapeWindow(kind).verified;
        if (mismatch) ++shapeWindow(kind).mismatches;
    }
    return mismatches;
}

std::uint64_t CountMismatch(Kind kind) {
    return ++shapeTotals(kind).mismatches;
}

void CountFullCapture(Kind kind, double microseconds) {
    ++shapeTotals(kind).fullCaptures;
    shapeTotals(kind).fullNanoseconds += static_cast<std::uint64_t>(microseconds * 1000.0);
    if (!profiled()) return;
    std::lock_guard lock(windowMutex(kind));
    ++shapeWindow(kind).fullCaptures;
    shapeWindow(kind).fullUs += microseconds;
}

ShapeReuseCounts Counts(Kind kind) {
    auto& totals = shapeTotals(kind);
    ShapeReuseCounts counts;
    counts.lookups = totals.lookups.load();
    counts.hits = totals.hits.load();
    counts.dataRefreshes = totals.dataRefreshes.load();
    counts.descriptorRefreshes = totals.descriptorRefreshes.load();
    for (std::size_t i = 0; i < counts.refusals.size(); ++i) counts.refusals[i] = totals.refusals[i].load();
    counts.remembered = totals.remembered.load();
    counts.notRemembered = totals.notRemembered.load();
    counts.verified = totals.verified.load();
    counts.mismatches = totals.mismatches.load();
    counts.fullCaptures = totals.fullCaptures.load();
    counts.hitNanoseconds = totals.hitNanoseconds.load();
    counts.fullNanoseconds = totals.fullNanoseconds.load();
    return counts;
}

void Report(Kind kind) {
    if (!profiled()) return;
    const char* line = kind == Kind::Draw ? "[draw-shape]" : "[shape-reuse]";
    std::lock_guard lock(windowMutex(kind));
    auto& window = shapeWindow(kind);
    const auto now = std::chrono::steady_clock::now();
    if (now - window.last < std::chrono::seconds(10)) return;
    window.last = now;
    std::string refusals;
    for (std::size_t i = 0; i < window.refusals.size(); ++i) {
        if (window.refusals[i] == 0) continue;
        char part[64];
        std::snprintf(part, sizeof(part), "%s%s %llu", refusals.empty() ? "" : ", ", RefusalNames[i], static_cast<unsigned long long>(window.refusals[i]));
        refusals += part;
    }
    std::string skipped;
    for (const auto& [reason, count] : window.notRemembered) {
        char part[96];
        std::snprintf(part, sizeof(part), "%s%s %llu", skipped.empty() ? "" : ", ", reason.c_str(), static_cast<unsigned long long>(count));
        skipped += part;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s (10 s) %llu lookups, %llu hits (%llu with refreshed words, %llu of them descriptor words), refused: %s; %.1f us per hit vs %.1f us per full capture (%llu); %llu captures remembered, not remembered: %s; verify %llu compared, %llu mismatches (%llu in total)\n", line, static_cast<unsigned long long>(window.lookups), static_cast<unsigned long long>(window.hits), static_cast<unsigned long long>(window.refreshes), static_cast<unsigned long long>(window.descriptorRefreshes), refusals.empty() ? "none" : refusals.c_str(), window.hits != 0 ? window.hitUs / static_cast<double>(window.hits) : 0.0, window.fullCaptures != 0 ? window.fullUs / static_cast<double>(window.fullCaptures) : 0.0, static_cast<unsigned long long>(window.fullCaptures), static_cast<unsigned long long>(window.remembered), skipped.empty() ? "none" : skipped.c_str(), static_cast<unsigned long long>(window.verified), static_cast<unsigned long long>(window.mismatches), static_cast<unsigned long long>(shapeTotals(kind).mismatches.load()));
    std::vector<std::pair<std::uint64_t, const ShapeWindow::Program*>> order;
    for (const auto& [program, entry] : window.programs) order.emplace_back(program, &entry);
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second->lookups > b.second->lookups; });
    for (std::size_t rank = 0; rank < order.size() && rank < 8; ++rank) {
        const auto& entry = *order[rank].second;
        std::size_t top = 0;
        for (std::size_t i = 1; i < entry.refusals.size(); ++i) {
            if (entry.refusals[i] > entry.refusals[top]) top = i;
        }
        // Word classes: '-' not read by the walk, 'P' pointer, 'C' copied, 'X' pointer and copied, 'O' opaque.
        std::string classes;
        for (const auto value : entry.classes) classes += (value & UserWordClassOpaque) != 0u ? 'O' : (value & UserWordClassPointer) != 0u ? ((value & UserWordClassCopy) != 0u ? 'X' : 'P') : (value & UserWordClassCopy) != 0u ? 'C' : '-';
        std::string changed;
        for (const auto& [word, count] : entry.keyWordsChanged) {
            char part[32];
            std::snprintf(part, sizeof(part), " u%u:%llu", word, static_cast<unsigned long long>(count));
            changed += part;
        }
        AgcDriver::ProfilePrint_nid_no_patch("%s   program 0x%llx: %llu lookups, %llu hits, most refused: %s %llu; words %s; key words changed since the last capture:%s\n", line, static_cast<unsigned long long>(order[rank].first), static_cast<unsigned long long>(entry.lookups), static_cast<unsigned long long>(entry.hits), entry.refusals[top] != 0 ? RefusalNames[top] : "none", static_cast<unsigned long long>(entry.refusals[top]), classes.empty() ? "unknown" : classes.c_str(), changed.empty() ? " none" : changed.c_str());
    }
    const auto keepLast = window.last;
    window = ShapeWindow{};
    window.last = keepLast;
}

}

ShapeReuseCounts DispatchShapeReuseCounts() {
    return ShapeCore::Counts(ShapeCore::Kind::Dispatch);
}

ShapeReuseCounts DrawShapeReuseCounts() {
    return ShapeCore::Counts(ShapeCore::Kind::Draw);
}

void CountShapeFullCapture(double microseconds) {
    ShapeCore::CountFullCapture(ShapeCore::Kind::Dispatch, microseconds);
}

bool Driver::shapeReuse() {
    static const bool fallback = std::getenv("APS5_DISPATCH_SHAPE_REUSE") != nullptr;
    return Graphics::LiveFlag("dispatch_shape_reuse", fallback);
}

bool Driver::verifyShapeReuse() {
    static const bool verify = std::getenv("APS5_VERIFY_SHAPE_REUSE") != nullptr;
    return verify;
}

void Driver::rememberShape(std::uint64_t address, std::uint64_t staticKey, const std::shared_ptr<const ShaderSnapshot>& registeredShader, std::span<const std::uint32_t> userData, std::span<const ShaderRecompiler::MemoryRegion> memory, const ShaderMemory& shaderMemory, const ShaderRecompiler::RecompileRequest& invocationRequest, const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, const std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiled) {
    auto learned = ShapeCore::Learn(capture, !invocationRequest.shader.capturedCalls.empty(), userData, invocationRequest.shader.codeAddress, [&] { return shaderMemory.Regions(); }, memory);
    auto candidate = std::make_shared<ShapeCandidate>();
    candidate->shader = registeredShader;
    candidate->capture = capture;
    candidate->compiled = compiled;
    candidate->sourceCode = invocationRequest.shader.code.data();
    candidate->sourceWords = invocationRequest.shader.code.size();
    dispatchShapes.Remember(ShapeCore::Kind::Dispatch, address, staticKey, userData, std::move(learned), std::move(candidate));
}

bool Driver::lookupShape(const ShaderSnapshot& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t address, std::uint32_t queue, std::uint64_t staticKey, const std::shared_ptr<const ShaderSnapshot>& registeredShader, std::span<const std::uint32_t> userData, std::shared_ptr<const ShaderRecompiler::RecompileResult>& compiledResult, std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, std::vector<ShaderRecompiler::MemoryRegion>& captured, std::shared_ptr<ShaderMemory>& shaderMemory, std::shared_ptr<const std::vector<std::uint32_t>>& shapeWords) {
    constexpr auto kind = ShapeCore::Kind::Dispatch;
    const auto started = std::chrono::steady_clock::now();
    ShapeCore::CountLookup(kind, address);
    std::vector<std::shared_ptr<const ShapeCandidate>> candidates;
    if (dispatchShapes.Find(kind, address, staticKey, userData, candidates) != ShapeRefusal::Count) return false;

    const auto initial = [&] {
        std::vector<ShaderRecompiler::MemoryRegion> sorted(memory.begin(), memory.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.guestAddress < b.guestAddress; });
        return sorted;
    }();
    auto firstRefusal = ShapeRefusal::Count;
    bool refreshed = false;
    bool refreshedDescriptors = false;
    const bool descriptors = refreshDescriptors();
    const auto refuse = [&](ShapeRefusal reason) {
        if (firstRefusal == ShapeRefusal::Count) firstRefusal = reason;
        return false;
    };
    // One candidate: relocate its reads, validate the moved words, rebuild the snapshot and
    // materialize. On success the outputs are set.
    const auto attempt = [&](const ShapeCandidate& candidate) -> bool {
        refreshed = false;
        refreshedDescriptors = false;
        if (candidate.staticKey != staticKey || candidate.shader.lock() != registeredShader || candidate.userData.size() != userData.size() || !candidate.plan->known) return refuse(ShapeRefusal::KeyCollision);
        std::vector<std::uint8_t> changed;
        std::vector<std::uint64_t> newAddresses;
        if (const auto reason = ShapeCore::Relocate(candidate, userData, initial, changed, newAddresses); reason != ShapeRefusal::Count) return refuse(reason);
        // The moved reads: the address each dword is read from now and what it must hold.
        struct Moved {
            std::uint64_t address;
            std::uint32_t value;
            bool leafOnly;
        };
        std::vector<Moved> moved;
        moved.reserve(candidate.reads.size());
        for (std::size_t r = 0; r < candidate.reads.size(); ++r) {
            const auto& read = candidate.reads[r];
            const auto now = newAddresses[r];
            const bool registered = insideAny(initial, now, sizeof(std::uint32_t));
            // A read of the registered code or header (the shape key fixes the snapshot) needs no check.
            if (registered) continue;
            // A word may differ when this read only copies it into a refresh target.
            moved.push_back({now, read.value, ShapeCore::Refreshable(candidate, read, descriptors)});
        }
        std::sort(moved.begin(), moved.end(), [](const Moved& a, const Moved& b) { return a.address < b.address; });
        // One word per address; a word read by any read that is not refreshable must keep its value.
        std::vector<Moved> words;
        words.reserve(moved.size());
        for (const auto& entry : moved) {
            if (!words.empty() && words.back().address == entry.address) {
                if (words.back().value != entry.value) return refuse(ShapeRefusal::Conflict);
                words.back().leafOnly = words.back().leafOnly && entry.leafOnly;
                continue;
            }
            words.push_back(entry);
        }
        DispatchVariant probe;
        probe.compiled = candidate.compiled;
        probe.words.reserve(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            const auto begin = words[i].address;
            const bool extends = !probe.runs.empty() && probe.runs.back().second == begin && (begin % PageBytes) != 0;
            if (extends) probe.runs.back().second += sizeof(std::uint32_t);
            else probe.runs.emplace_back(begin, begin + sizeof(std::uint32_t));
            probe.words.push_back(words[i].value);
            if (words[i].leafOnly) probe.dataPositions.push_back(static_cast<std::uint32_t>(i));
        }
        std::vector<ShaderRecompiler::MemoryRegion> regions;
        appendEntryRegions(probe, regions);
        std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
        {
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DispatchCache);
            std::optional<SampledReadScope> sampling;
            std::uint64_t imagesFlushed = 0, runsSynced = 0;
            auto outcome = validateVariant(address, queue, probe, regions, imagesFlushed, runsSynced, sampling, &live);
            if (gateRetry() && (outcome == EntryOutcome::PublishMoved || outcome == EntryOutcome::PendingMoved)) outcome = validateVariant(address, queue, probe, regions, imagesFlushed, runsSynced, sampling, &live);
            switch (outcome) {
                case EntryOutcome::Equal:
                case EntryOutcome::EqualData: break;
                case EntryOutcome::Differing: return refuse(ShapeRefusal::Differing);
                case EntryOutcome::Inaccessible: return refuse(ShapeRefusal::Inaccessible);
                default: return refuse(ShapeRefusal::Pending);
            }
        }
        // The refreshed words, at their new addresses.
        std::vector<std::pair<std::uint64_t, std::uint32_t>> refreshedWords;
        refreshedWords.reserve(live.size());
        for (const auto& [position, value] : live) {
            if (position >= words.size() || !words[position].leafOnly) return refuse(ShapeRefusal::Inconsistent);
            refreshedWords.emplace_back(words[position].address, value);
            probe.words[position] = value;
        }
        std::sort(refreshedWords.begin(), refreshedWords.end());
        std::shared_ptr<ShaderRecompiler::ResourceCapture> rebuilt;
        if (const auto reason = ShapeCore::Rebuild(candidate, userData, changed, newAddresses, refreshedWords, descriptors, rebuilt, refreshedDescriptors); reason != ShapeRefusal::Count) return refuse(reason);
        std::shared_ptr<const ShaderRecompiler::RecompileResult> result;
        try {
            // A scratch memory: an invocation that would resolve user-data calls reads guest code,
            // which a remembered capture never explains.
            ShaderMemory scratch(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
            const auto invocation = InvocationFor(snapshot, codeOffset, request, &scratch);
            if (scratch.Regions().size() != memory.size() || !invocation.Request().shader.capturedCalls.empty()) return refuse(ShapeRefusal::Source);
            if (invocation.Request().shader.code.data() != candidate.sourceCode || invocation.Request().shader.code.size() != candidate.sourceWords) return refuse(ShapeRefusal::Source);
            result = invocation.Materialize(*rebuilt);
        } catch (const std::exception&) {
            return refuse(ShapeRefusal::Materialize);
        }
        // The regions ShaderMemory::Regions() would report: the registered regions merged in
        // address order with the page runs.
        auto owned = std::make_shared<std::vector<std::uint32_t>>(std::move(probe.words));
        std::vector<ShaderRecompiler::MemoryRegion> merged;
        merged.reserve(initial.size() + probe.runs.size());
        std::size_t next = 0, offset = 0;
        for (const auto& [begin, end] : probe.runs) {
            const auto base = begin & ~(PageBytes - 1u);
            while (next < initial.size() && initial[next].guestAddress < base) merged.push_back(initial[next++]);
            const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
            merged.push_back({begin, std::as_bytes(std::span<const std::uint32_t>(*owned).subspan(offset, count))});
            offset += count;
        }
        for (; next < initial.size(); ++next) merged.push_back(initial[next]);
        compiledResult = std::move(result);
        capture = std::move(rebuilt);
        captured = std::move(merged);
        shapeWords = std::move(owned);
        shaderMemory = nullptr;
        refreshed = !live.empty();
        return true;
    };
    for (const auto& candidate : candidates) {
        if (!attempt(*candidate)) continue;
        const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        ShapeCore::CountHitTotals(kind, elapsed, refreshed, refreshedDescriptors);
        if (verifyShapeReuse()) {
            // The full capture as well: on any difference its results are used.
            auto full = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
            try {
                const auto invocation = InvocationFor(snapshot, codeOffset, request, full.get());
                const auto fullCapture = full->Capture(invocation);
                const auto fullRegions = full->Regions();
                const auto fullResult = invocation.Materialize(*fullCapture);
                const char* what = ShapeCore::Difference(*compiledResult, *fullResult, *capture, *fullCapture, captured, fullRegions);
                const auto count = ShapeCore::CountVerified(kind, what != nullptr);
                if (what != nullptr) {
                    if (count <= 8) std::fprintf(stderr, "[shape-reuse] APS5_VERIFY_SHAPE_REUSE: program 0x%llx: %s differs from the full capture (mismatch %llu)\n", static_cast<unsigned long long>(address), what, static_cast<unsigned long long>(count));
                    compiledResult = fullResult;
                    capture = fullCapture;
                    captured = fullRegions;
                    shaderMemory = full;
                    shapeWords = nullptr;
                }
            } catch (const std::exception& error) {
                const auto count = ShapeCore::CountMismatch(kind);
                if (count <= 8) std::fprintf(stderr, "[shape-reuse] APS5_VERIFY_SHAPE_REUSE: program 0x%llx: the full capture failed where reuse succeeded: %s\n", static_cast<unsigned long long>(address), error.what());
                throw;
            }
        }
        ShapeCore::CountHitWindow(kind, address, elapsed, refreshed, refreshedDescriptors);
        ShapeCore::Report(kind);
        return true;
    }
    ShapeCore::CountRefusal(kind, address, firstRefusal == ShapeRefusal::Count ? ShapeRefusal::Inconsistent : firstRefusal);
    ShapeCore::Report(kind);
    return false;
}

void Driver::reportShapeReuse() {
    ShapeCore::Report(ShapeCore::Kind::Dispatch);
}

}
