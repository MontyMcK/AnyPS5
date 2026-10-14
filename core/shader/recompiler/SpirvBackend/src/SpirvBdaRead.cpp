#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include "SpirvBackend/SpirvEmitterInstructions.hpp"
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>
#include <stdexcept>
#include <string>

namespace ShaderRecompiler {

namespace {

void EmitBdaOverflowCheck(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction) {
    const auto overflow = Binary(state, spv::OpUGreaterThan, TypeBool(state), address, BdaConstant(state, std::numeric_limits<std::uint64_t>::max() - bytes));
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
}

bool BdaCoherent(const SpirvEmitterState& state, const IrValue& inst) {
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = state.program.Resources().memoryInfo;
    return index < memory.size() && memory[index].coherent;
}

std::uint32_t BdaAccessMask(bool coherent) {
    return spv::MemoryAccessAlignedMask | (coherent ? spv::MemoryAccessVolatileMask : 0u);
}

std::uint32_t BdaAccessMask(const SpirvEmitterState& state, const IrValue& inst) {
    return BdaAccessMask(BdaCoherent(state, inst));
}

std::uint32_t EmitBdaByte(SpirvEmitterState& state, std::uint32_t guest, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
    const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaPointerFunction, guest, ConstantU32(state, 1u), instruction);
    if (state.bdaStopsInvocations) {
        StopBdaInvocationIf(state, Binary(state, spv::OpIEqual, TypeBool(state), physical, BdaConstant(state, 0u)));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
        return Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    }
    // Unmapped bytes read as zero; the fault is recorded and every invocation reaches the
    // program's barriers.
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto before = state.currentLabel;
    const auto loadLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, mapped, loadLabel, merge);
    EmitLabel(state, loadLabel);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, physical);
    const auto loaded = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
    const auto widened = Unary(state, spv::OpUConvert, TypeU32(state), loaded);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, TypeU32(state), value, widened, loadLabel, ConstantU32(state, 0u), before);
    return value;
}

std::uint32_t EmitBdaBytes(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t accessMask) {
    auto result = ConstantU32(state, 0u);
    for (std::uint32_t byte = 0; byte < bytes; ++byte) {
        const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
        const auto value = EmitBdaByte(state, guest, instruction, accessMask);
        result = Binary(state, spv::OpBitwiseOr, TypeU32(state), result, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), value, ConstantU32(state, byte * 8u)));
    }
    return result;
}

// The dwords of a wide load that the program extracts (all of them when the composite is used
// whole). Dead-code elimination has dropped the unused extracts, as it dropped unused dword loads
// when every dword was its own instruction.
std::uint32_t UsedBdaDwords(const IrValue& inst, std::uint32_t dwords) {
    const auto all = (1u << dwords) - 1u;
    if (dwords == 1u) return all;
    std::uint32_t used = 0;
    for (const IrValue* user : inst.Uses()) {
        const auto opcode = user->Opcode();
        if (opcode != IrOpcode::CompositeExtractU32x2 && opcode != IrOpcode::CompositeExtractU32x3 && opcode != IrOpcode::CompositeExtractU32x4) return all;
        const IrValue* index = user->ArgumentCount() > 1u ? user->Argument(1) : nullptr;
        if (index == nullptr || !index->HasImmediate() || index->ImmediateU32() >= dwords) return all;
        used |= 1u << index->ImmediateU32();
    }
    return used;
}

void EmitBdaStoreAt(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t pointerType, std::uint32_t value, std::uint32_t instruction, std::uint32_t accessMask) {
    const auto physical = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
    EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), [&] {
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, pointerType, pointer, physical);
        state.module.AddFunction(spv::OpStore, pointer, value, accessMask, bytes);
        state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
    });
}

std::uint32_t BdaAddressUnaligned(SpirvEmitterState& state, std::uint32_t address) {
    return Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, 3u)), BdaConstant(state, 0u));
}

}

void DefineBdaDwordReadFunctions(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto pair = TypeU32Vector(state, 2);
    const auto type = state.module.Type(spv::OpTypeFunction, pair, u64, u32);
    const auto& memory = state.program.Resources().memoryInfo;
    const bool coherentAccesses = std::any_of(memory.begin(), memory.end(), [](const MemoryInfo& info) { return info.coherent; });
    const bool stops = state.bdaStopsInvocations;
    if (stops) state.bdaStopValue = state.module.Constant(spv::OpConstantComposite, pair, ConstantU32(state, 0u), ConstantU32(state, 1u));
    for (const bool stop : {false, true}) {
        for (const bool coherent : {false, true}) {
            if ((stop && !stops) || (coherent && !coherentAccesses)) continue;
            const auto function = state.module.AllocateId();
            state.module.AddName(function, std::string("read_bda_dword_bytes") + (stop ? "_stop" : "") + (coherent ? "_coherent" : ""));
            state.module.AddFunction(spv::OpFunction, pair, function, spv::FunctionControlDontInlineMask, type);
            const auto address = state.module.AllocateId();
            const auto instruction = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionParameter, u64, address);
            state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
            EmitLabel(state, state.module.AllocateId());
            state.bdaStopsInvocations = stop;
            EmitBdaOverflowCheck(state, address, 4u, instruction);
            const auto value = EmitBdaBytes(state, address, 4u, instruction, BdaAccessMask(coherent));
            const auto result = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeConstruct, pair, result, value, ConstantU32(state, 0u));
            state.module.AddFunction(spv::OpReturnValue, result);
            state.module.AddFunction(spv::OpFunctionEnd);
            state.bdaDwordReadFunctions[stop][coherent] = function;
        }
    }
    state.bdaStopsInvocations = stops;
    state.bdaStopValue = 0;
}

// APS5_BDA_BYTE_READS=1 restores one lookup and one byte load per byte for every read.
bool BdaByteReadsForced() {
    static const bool forced = std::getenv("APS5_BDA_BYTE_READS") != nullptr;
    return forced;
}

bool BdaHintFastPath() {
    static const bool enabled = std::getenv("APS5_BDA_HINT_FAST") != nullptr;
    return enabled;
}

// APS5_ELECT_SCALAR_BDA=1: a read a scalar memory instruction made (s_load, s_buffer_load: its address
// comes from scalar registers, one value for the whole wave) probes the table in one elected lane of
// the subgroup, and the device address it found is broadcast to the others, instead of every lane
// repeating the same lookup. Part of the shader disk cache version (ShaderDiskCache.cpp).
bool ElectScalarBdaReads() {
    static const bool enabled = std::getenv("APS5_ELECT_SCALAR_BDA") != nullptr;
    return enabled;
}

bool ScalarBdaAccess(const IrProgram& program, const IrValue& inst) {
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = program.Resources().memoryInfo;
    return index < memory.size() && (memory[index].kind == ResourceKind::ScalarAddress || memory[index].kind == ResourceKind::ScalarBuffer);
}

namespace {

// `probe` run by the subgroup's elected lane; every lane receives its 64-bit result.
std::uint32_t EmitElectedU64(SpirvEmitterState& state, const std::function<std::uint32_t()>& probe) {
    const auto u64 = TypeScalarU64(state);
    const auto pair = TypeU32Vector(state, 2);
    const auto elected = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected, ConstantU32(state, spv::ScopeSubgroup));
    const auto before = state.currentLabel;
    const auto electedLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, elected, electedLabel, merge);
    EmitLabel(state, electedLabel);
    const auto probed = probe();
    const auto electedExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u64, value, probed, electedExit, BdaConstant(state, 0u), before);
    const auto halves = state.module.AllocateId();
    state.module.AddFunction(spv::OpBitcast, pair, halves, value);
    const auto shared = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, pair, shared, ConstantU32(state, spv::ScopeSubgroup), halves);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpBitcast, u64, result, shared);
    return result;
}

}

std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract) {
    auto& state = ctx.state;
    const auto result = Binary(state, subtract ? spv::OpISub : spv::OpIAdd, TypeScalarU64(state), address, offset);
    const auto overflow = subtract ? Binary(state, spv::OpUGreaterThan, TypeBool(state), offset, address) : Binary(state, spv::OpULessThan, TypeBool(state), result, address);
    EmitIfCondition(state, overflow, [&] { RecordBdaFault(state, address, ConstantU32(state, 0u), ConstantU32(state, inst.Flags<MemoryFlags>().pc), BdaAbi::FaultReason::Overflow); });
    StopBdaInvocationIf(state, overflow);
    return result;
}

std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate) {
    if (immediate == 0) {
        return address;
    }
    const auto magnitude = immediate < 0 ? -static_cast<std::int64_t>(immediate) : static_cast<std::int64_t>(immediate);
    return AddBdaAddress(ctx, inst, address, BdaConstant(ctx.state, static_cast<std::uint64_t>(magnitude)), immediate < 0);
}

void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target) {
    if (!program.Info().usesDma) return;
    if (target.bdaAbiVersion != BdaAbi::Version) throw std::runtime_error("unsupported BDA ABI version");
    for (const auto capability : {spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess}) {
        if (std::find(target.supportedCapabilities.begin(), target.supportedCapabilities.end(), static_cast<std::uint32_t>(capability)) == target.supportedCapabilities.end()) throw std::runtime_error("BDA requires unsupported SPIR-V capability " + std::to_string(capability));
    }
    for (const auto extension : {"SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"}) {
        if (std::find(target.supportedExtensions.begin(), target.supportedExtensions.end(), extension) == target.supportedExtensions.end()) throw std::runtime_error(std::string("BDA requires unsupported extension ") + extension);
    }
    if (program.Resources().stage == IrShaderStage::TessellationControl) throw std::runtime_error("BDA fault termination requires a barrier-safe tessellation-control execution protocol");
}

bool BdaInvocationsMayStop(const IrProgram& program) {
    for (const auto* block : program.BlockOrder()) {
        for (const auto* instruction : block->Instructions()) {
            if (instruction->Opcode() == IrOpcode::Barrier) return false;
        }
    }
    return true;
}

// ---- anchors --------------------------------------------------------------------------------

namespace {

// APS5_NO_BDA_ANCHORS=1 (A/B measurement): every read takes the lookup path, as before.
bool BdaAnchorsEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BDA_ANCHORS") != nullptr;
    return !disabled;
}

std::optional<std::pair<std::uint32_t, std::uint32_t>> UserDataBase(const IrProgram& program, const IrValue& inst) {
    if (AddressOpcodeInfoOf(inst.Opcode()).access != AddressAccess::Read || inst.ArgumentCount() == 0) return std::nullopt;
    const auto index = inst.Flags<MemoryFlags>().index;
    const auto& memory = program.Resources().memoryInfo;
    if (index >= memory.size()) return std::nullopt;
    const auto& info = memory[index];
    if ((info.kind != ResourceKind::ScalarAddress && info.kind != ResourceKind::Flat && info.kind != ResourceKind::Global) || info.planningOnly || info.addressIsFull) return std::nullopt;
    const IrValue* handle = inst.Argument(0) != nullptr ? inst.Argument(0)->Resolve() : nullptr;
    if (handle == nullptr || handle->Opcode() != IrOpcode::GetAddressResource || handle->ArgumentCount() != 2u) return std::nullopt;
    const auto userData = [](const IrValue* value) -> std::optional<std::uint32_t> {
        value = value != nullptr ? value->Resolve() : nullptr;
        if (value == nullptr || value->Opcode() != IrOpcode::GetUserData || value->ArgumentCount() == 0 || value->Argument(0) == nullptr) return std::nullopt;
        return value->Argument(0)->Register().index;
    };
    const auto low = userData(handle->Argument(0));
    const auto high = userData(handle->Argument(1));
    if (!low || !high) return std::nullopt;
    return std::pair{*low, *high};
}

std::uint32_t AnchorAddressFromWords(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) {
    const auto u64 = TypeScalarU64(state);
    const auto high64 = Binary(state, spv::OpShiftLeftLogical, u64, Unary(state, spv::OpUConvert, u64, high), BdaConstant(state, 32u));
    return Binary(state, spv::OpBitwiseOr, u64, Unary(state, spv::OpUConvert, u64, low), high64);
}

// The u64 vector of 3 that `compute` makes in the subgroup's elected lane, in every lane.
std::uint32_t EmitElectedRange(SpirvEmitterState& state, std::uint32_t type, std::uint32_t zero, const std::function<std::uint32_t()>& compute) {
    const auto elected = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected, ConstantU32(state, spv::ScopeSubgroup));
    const auto before = state.currentLabel;
    const auto electedLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, elected, electedLabel, merge);
    EmitLabel(state, electedLabel);
    const auto computed = compute();
    const auto electedExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, type, value, computed, electedExit, zero, before);
    const auto u64 = TypeScalarU64(state);
    const auto pair = TypeU32Vector(state, 2);
    std::array<std::uint32_t, 3> parts{};
    for (std::uint32_t component = 0; component < 3u; ++component) {
        const auto part = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, part, value, component);
        const auto halves = state.module.AllocateId();
        state.module.AddFunction(spv::OpBitcast, pair, halves, part);
        const auto shared = state.module.AllocateId();
        state.module.AddFunction(spv::OpGroupNonUniformBroadcastFirst, pair, shared, ConstantU32(state, spv::ScopeSubgroup), halves);
        parts[component] = state.module.AllocateId();
        state.module.AddFunction(spv::OpBitcast, u64, parts[component], shared);
    }
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, type, result, parts[0], parts[1], parts[2]);
    return result;
}

}

void CollectBdaAnchors(SpirvEmitterState& state) {
    state.bdaAnchors.clear();
    if (!BdaAnchorsEnabled() || !state.program.Info().usesDma || BdaByteReadsForced()) return;
    // Each anchor costs one table search per subgroup at the entry, whether its reads run or not,
    // and a read the anchor saves costs at most one search: bases with two reads or more, at most
    // MaxBdaAnchors of them (those with the most reads).
    constexpr std::size_t MaxBdaAnchors = 4;
    std::vector<std::pair<std::pair<std::uint32_t, std::uint32_t>, std::size_t>> bases;
    for (const IrBlock* block : state.program.BlockOrder()) {
        for (const IrValue* inst : block->Instructions()) {
            const auto base = UserDataBase(state.program, *inst);
            if (!base) continue;
            const auto found = std::find_if(bases.begin(), bases.end(), [&](const auto& entry) { return entry.first == *base; });
            if (found != bases.end()) ++found->second;
            else bases.push_back({*base, 1u});
        }
    }
    std::stable_sort(bases.begin(), bases.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    for (const auto& [base, reads] : bases) {
        if (reads < 2u || state.bdaAnchors.size() == MaxBdaAnchors) break;
        state.bdaAnchors.push_back({base.first, base.second, 0u, 0u, 0u, 0u});
    }
}

void EmitBdaAnchorRange(SpirvEmitterState& state, SpirvEmitterState::BdaAnchor& anchor, std::uint32_t base) {
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto rangeType = state.module.Type(spv::OpTypeVector, u64, 3u);
    const auto zero = state.module.Constant(spv::OpConstantComposite, rangeType, BdaConstant(state, 0u), BdaConstant(state, 0u), BdaConstant(state, 0u));
    const auto search = [&] {
        const auto range = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, rangeType, range, state.bdaAnchorFunction, base, ConstantU32(state, 1u), ConstantU32(state, 0u));
        return range;
    };
    // The base is one value for the draw, so one lane searches for the subgroup.
    const auto range = state.requirements.subgroupBallot ? EmitElectedRange(state, rangeType, zero, search) : search();
    const auto extract = [&](std::uint32_t component) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeExtract, u64, value, range, component);
        return value;
    };
    anchor.begin = extract(0u);
    const auto end = extract(1u);
    anchor.device = extract(2u);
    anchor.offset = Binary(state, spv::OpISub, u64, base, anchor.begin);
    // The search returns zeros (size 0: no read uses the anchor) where the lookup would fail.
    // A range whose device addresses wrap, or that ends above 2^63 (see BdaAnchorHolds), is not
    // used either; its reads take the lookup path, which checks each translation.
    const auto span = Binary(state, spv::OpISub, u64, end, anchor.begin);
    const auto deviceEnd = Binary(state, spv::OpIAdd, u64, anchor.device, span);
    auto usable = Binary(state, spv::OpLogicalAnd, boolean, Binary(state, spv::OpINotEqual, boolean, anchor.device, BdaConstant(state, 0u)), Binary(state, spv::OpUGreaterThan, boolean, deviceEnd, anchor.device));
    usable = Binary(state, spv::OpLogicalAnd, boolean, usable, Binary(state, spv::OpULessThanEqual, boolean, end, BdaConstant(state, 0x8000000000000000ull)));
    anchor.size = Select(state, u64, usable, span, BdaConstant(state, 0u));
}

void EmitBdaAnchors(SpirvEmitterState& state) {
    if (state.bdaAnchors.empty() || state.bdaAnchorFunction == 0) return;
    for (auto& anchor : state.bdaAnchors) {
        const auto low = EmitGetUserData(state, static_cast<ScalarReg>(anchor.lowRegister));
        const auto high = EmitGetUserData(state, static_cast<ScalarReg>(anchor.highRegister));
        EmitBdaAnchorRange(state, anchor, AnchorAddressFromWords(state, low, high));
    }
}

const SpirvEmitterState::BdaAnchor* FindBdaAnchor(const SpirvEmitterState& state, const IrValue& inst) {
    if (state.bdaAnchors.empty()) return nullptr;
    const auto base = UserDataBase(state.program, inst);
    if (!base) return nullptr;
    for (const auto& anchor : state.bdaAnchors) {
        if (anchor.lowRegister == base->first && anchor.highRegister == base->second && anchor.size != 0) return &anchor;
    }
    return nullptr;
}

namespace {

// Whether the anchor holds [address, address + bytes) for address = base + zext(low) + immediate
// (`translated`: the device address of its first byte).
std::uint32_t BdaAnchorHolds(SpirvEmitterState& state, const SpirvEmitterState::BdaAnchor& anchor, std::uint32_t low, std::int32_t immediate, std::uint32_t bytes, std::uint32_t& translated) {
    // offset = (base - begin) + low + immediate, mod 2^64. The anchor's range ends at or below 2^63
    // (EmitBdaAnchorRange), so an offset at most size - bytes is the true offset of a read that
    // lies in the range, and base + low and its sum with the immediate wrap nowhere: the lookup
    // path's address checks pass and its lookups return device + offset. A true offset below 0 or
    // above size - bytes stays above size - bytes mod 2^64.
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    auto offset = Binary(state, spv::OpIAdd, u64, anchor.offset, Unary(state, spv::OpUConvert, u64, low));
    if (immediate != 0) offset = Binary(state, spv::OpIAdd, u64, offset, BdaConstant(state, static_cast<std::uint64_t>(static_cast<std::int64_t>(immediate))));
    const auto width = BdaConstant(state, bytes);
    const auto fits = Binary(state, spv::OpUGreaterThanEqual, boolean, anchor.size, width);
    const auto inside = Binary(state, spv::OpULessThanEqual, boolean, offset, Binary(state, spv::OpISub, u64, anchor.size, width));
    translated = Binary(state, spv::OpIAdd, u64, anchor.device, offset);
    return Binary(state, spv::OpLogicalAnd, boolean, fits, inside);
}

}

std::array<std::uint32_t, 4> EmitAnchoredBdaRead(SpirvEmitterState& state, const IrValue& inst, const SpirvEmitterState::BdaAnchor& anchor, std::uint32_t low, std::int32_t immediate, std::uint32_t bits, std::uint32_t dwords, std::uint32_t used, const std::function<std::array<std::uint32_t, 4>()>& lookup) {
    std::uint32_t first = 0;
    std::uint32_t last = 0;
    std::uint32_t bytes = bits / 8u;
    if (bits == 32u) {
        first = static_cast<std::uint32_t>(std::countr_zero(used));
        last = 31u - static_cast<std::uint32_t>(std::countl_zero(used));
        bytes = (last - first + 1u) * 4u;
    }
    std::uint32_t translated = 0;
    // The immediate of the first dword as the lookup path forms it (EmitBdaDwordReads: the sum in
    // 32 bits, then signed).
    auto fast = BdaAnchorHolds(state, anchor, low, static_cast<std::int32_t>(static_cast<std::uint32_t>(immediate) + first * 4u), bytes, translated);
    // Dword reads take the anchor's path where the lookup path takes its wide one: at a 4-aligned
    // device address; elsewhere the lookup path reads the bytes.
    if (bits == 32u) fast = Binary(state, spv::OpLogicalAnd, TypeBool(state), fast, Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), translated, BdaConstant(state, 3u)), BdaConstant(state, 0u)));
    const auto fastLabel = state.module.AllocateId();
    const auto slowLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, fast, fastLabel, slowLabel);
    EmitLabel(state, fastLabel);
    std::array<std::uint32_t, 4> fastValues{};
    const auto accessMask = BdaAccessMask(state, inst);
    if (bits == 32u) {
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if ((used & (1u << dword)) == 0u) continue;
            const auto element = dword == first ? translated : Binary(state, spv::OpIAdd, TypeScalarU64(state), translated, BdaConstant(state, (dword - first) * 4u));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
            fastValues[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, TypeU32(state), fastValues[dword], pointer, accessMask, 4u);
        }
    } else {
        const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
        const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
        auto value = ConstantU32(state, 0u);
        for (std::uint32_t byte = 0; byte < bytes; ++byte) {
            const auto element = byte == 0u ? translated : Binary(state, spv::OpIAdd, TypeScalarU64(state), translated, BdaConstant(state, byte));
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, bytePointer, pointer, element);
            const auto loaded = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, byteType, loaded, pointer, accessMask, 1u);
            value = Binary(state, spv::OpBitwiseOr, TypeU32(state), value, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), Unary(state, spv::OpUConvert, TypeU32(state), loaded), ConstantU32(state, byte * 8u)));
        }
        fastValues[0] = value;
    }
    const auto fastExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, slowLabel);
    const auto slowValues = lookup();
    const auto slowExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    std::array<std::uint32_t, 4> values{};
    for (std::uint32_t dword = 0; dword < (bits == 32u ? dwords : 1u); ++dword) {
        if (bits == 32u && (used & (1u << dword)) == 0u) {
            values[dword] = ConstantU32(state, 0u);
            continue;
        }
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, TypeU32(state), values[dword], fastValues[dword], fastExit, slowValues[dword], slowExit);
    }
    return values;
}

std::uint32_t UsedBdaReadDwords(const IrValue& inst, std::uint32_t dwords) {
    return UsedBdaDwords(inst, dwords);
}

std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    if (bits == 32u) return EmitBdaDwordReads(ctx, inst, address, 0u, 1u)[0];
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
    return EmitBdaBytes(state, address, bits / 8u, instruction, BdaAccessMask(state, inst));
}

void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA write width");
    if (bits != 32u) {
        EmitBdaOverflowCheck(state, address, bits / 8u, instruction);
        for (std::uint32_t byte = 0u; byte < bits / 8u; ++byte) {
            const auto guest = Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
            const auto physical = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, guest, ConstantU32(state, 1u), instruction);
            EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), [&] {
                const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
                const auto pointer = state.module.AllocateId();
                state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType), pointer, physical);
                const auto shifted = Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, ConstantU32(state, byte * 8u));
                state.module.AddFunction(spv::OpStore, pointer, Unary(state, spv::OpUConvert, byteType, shifted), BdaAccessMask(state, inst), 1u);
                state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, guest);
            });
        }
        return;
    }
    const auto unaligned = BdaAddressUnaligned(state, address);
    EmitIfCondition(state, unaligned, [&] { RecordBdaFault(state, address, ConstantU32(state, 4u), instruction, BdaAbi::FaultReason::Unaligned); });
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
        EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, BdaAccessMask(state, inst));
    });
}

void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits) {
    auto& state = ctx.state;
    if (bits != 8u && bits != 16u && bits != 32u) ctx.Fail(inst, "unsupported BDA store width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto accessMask = BdaAccessMask(state, inst);
    const auto storeBytes = [&] {
        const auto byteType = state.module.Type(spv::OpTypeInt, 8u, 0u);
        const auto bytePointer = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, byteType);
        for (std::uint32_t byte = 0; byte < bits / 8u; ++byte) {
            const auto guest = byte == 0u ? address : Binary(state, spv::OpIAdd, TypeScalarU64(state), address, BdaConstant(state, byte));
            const auto shifted = byte == 0u ? value : Binary(state, spv::OpShiftRightLogical, TypeU32(state), value, ConstantU32(state, byte * 8u));
            EmitBdaStoreAt(state, guest, 1u, bytePointer, Unary(state, spv::OpUConvert, byteType, shifted), instruction, accessMask);
        }
    };
    if (bits != 32u) {
        storeBytes();
        return;
    }
    const auto unaligned = BdaAddressUnaligned(state, address);
    EmitIfCondition(state, unaligned, storeBytes);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), [&] {
        EmitBdaStoreAt(state, address, 4u, TypePhysicalU32Pointer(state), value, instruction, accessMask);
    });
}

std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation) {
    auto& state = ctx.state;
    if (bytes != 4u && bytes != 8u) ctx.Fail(inst, "unsupported BDA atomic width");
    if (state.bdaWritePointerFunction == 0 || state.bdaNoteWriteFunction == 0) ctx.Fail(inst, "BDA write functions are missing");
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto type = bytes == 8u ? TypeScalarU64(state) : TypeU32(state);
    const auto zero = bytes == 8u ? BdaConstant(state, 0u) : ConstantU32(state, 0u);
    const auto unaligned = Binary(state, spv::OpINotEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, TypeScalarU64(state), address, BdaConstant(state, bytes - 1u)), BdaConstant(state, 0u));
    EmitIfCondition(state, unaligned, [&] { RecordBdaFault(state, address, ConstantU32(state, bytes), instruction, BdaAbi::FaultReason::Unaligned); });
    return EmitValueOrDefaultIfCondition(state, Unary(state, spv::OpLogicalNot, TypeBool(state), unaligned), type, zero, [&] {
        const auto physical = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, TypeScalarU64(state), physical, state.bdaWritePointerFunction, address, ConstantU32(state, bytes), instruction);
        return EmitValueOrDefaultIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u)), type, zero, [&] {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, type), pointer, physical);
            const auto old = operation(pointer);
            state.module.AddFunction(spv::OpFunctionCall, state.module.Type(spv::OpTypeVoid), state.module.AllocateId(), state.bdaNoteWriteFunction, address);
            return old;
        });
    });
}

// One probe of the span from the first to the last extracted dword replaces 4 lookups per dword (a
// Bink DC pass spent ~450 dependent table levels per iteration on them). The span is loaded
// dword-wise from the returned device address when it lies in one range and that address is
// 4-aligned; a span crossing ranges (the table is byte-granular, see tests/BdaExecution.cpp), an
// unaligned mirror or a wrapping span falls back to the byte path. That path repeats, per extracted
// dword in order, the address, overflow check and byte lookups of a separate dword load, so every
// fault is recorded exactly as one; the probe records none, and a span it accepts cannot fault.
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword) {
    auto& state = ctx.state;
    if (dwords == 0u || dwords > 4u) ctx.Fail(inst, "unsupported BDA read width");
    if (state.bdaPointerFunction == 0) ctx.Fail(inst, "BDA lookup function is missing");
    const auto u64 = TypeScalarU64(state);
    const auto instruction = ConstantU32(state, inst.Flags<MemoryFlags>().pc);
    const auto used = everyDword ? (1u << dwords) - 1u : UsedBdaDwords(inst, dwords);
    const auto isUsed = [&](std::uint32_t dword) { return (used & (1u << dword)) != 0u; };
    std::array<std::uint32_t, 4> values{};
    std::uint32_t first = dwords;
    std::uint32_t last = 0;
    for (std::uint32_t dword = 0; dword < dwords; ++dword) {
        if (!isUsed(dword)) {
            values[dword] = ConstantU32(state, 0u);
            continue;
        }
        first = std::min(first, dword);
        last = dword;
    }
    if (used == 0u) return values;
    std::array<std::uint32_t, 4> addresses{};
    const auto dwordAddress = [&](std::uint32_t dword) { return AddBdaImmediate(ctx, inst, address, static_cast<std::int32_t>(offset + dword * 4u)); };
    const auto readBytes = [&](std::array<std::uint32_t, 4>& into) {
        const auto function = state.bdaDwordReadFunctions[state.bdaStopsInvocations][BdaCoherent(state, inst)];
        if (function == 0) ctx.Fail(inst, "BDA dword read function is missing");
        for (std::uint32_t dword = first; dword <= last; ++dword) {
            if (!isUsed(dword)) continue;
            if (dword != first) addresses[dword] = dwordAddress(dword);
            const auto read = state.module.AllocateId();
            state.module.AddFunction(spv::OpFunctionCall, TypeU32Vector(state, 2), read, function, addresses[dword], instruction);
            into[dword] = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), into[dword], read, 0u);
            if (!state.bdaStopsInvocations) continue;
            const auto stopped = state.module.AllocateId();
            state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), stopped, read, 1u);
            StopBdaInvocationIf(state, Binary(state, spv::OpINotEqual, TypeBool(state), stopped, ConstantU32(state, 0u)));
        }
    };
    addresses[first] = dwordAddress(first);
    if (state.bdaProbeFunction == 0) {
        readBytes(values);
        return values;
    }
    const auto probeSpan = [&] {
        return EmitBdaLookupCall(state, state.bdaProbeFunction, addresses[first], ConstantU32(state, (last - first + 1u) * 4u), instruction, BdaAbi::Read);
    };
    // The probe records no fault, so the elected lane's answer is every lane's: a span it refuses
    // takes the byte path below in each lane, which records the faults as before.
    const bool elect = ElectScalarBdaReads() && state.requirements.subgroupBallot && ScalarBdaAccess(state.program, inst);
    const auto physical = elect ? EmitElectedU64(state, probeSpan) : probeSpan();
    const auto mapped = Binary(state, spv::OpINotEqual, TypeBool(state), physical, BdaConstant(state, 0u));
    const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), Binary(state, spv::OpBitwiseAnd, u64, physical, BdaConstant(state, 3u)), BdaConstant(state, 0u));
    const auto wide = Binary(state, spv::OpLogicalAnd, TypeBool(state), mapped, aligned);
    const auto wideLabel = state.module.AllocateId();
    const auto byteLabel = state.module.AllocateId();
    const auto byteExit = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, wide, wideLabel, byteLabel);
    EmitLabel(state, wideLabel);
    std::array<std::uint32_t, 4> wideValues{};
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        const auto element = dword == first ? physical : Binary(state, spv::OpIAdd, u64, physical, BdaConstant(state, (dword - first) * 4u));
        const auto pointer = state.module.AllocateId();
        state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, element);
        wideValues[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), wideValues[dword], pointer, BdaAccessMask(state, inst), 4u);
    }
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, byteLabel);
    std::array<std::uint32_t, 4> byteValues{};
    readBytes(byteValues);
    state.module.AddFunction(spv::OpBranch, byteExit);
    EmitLabel(state, byteExit);
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    for (std::uint32_t dword = first; dword <= last; ++dword) {
        if (!isUsed(dword)) continue;
        values[dword] = state.module.AllocateId();
        state.module.AddFunction(spv::OpPhi, TypeU32(state), values[dword], wideValues[dword], wideLabel, byteValues[dword], byteExit);
    }
    return values;
}

}
