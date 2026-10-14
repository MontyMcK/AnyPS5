#include "Optimization/BufferStoreCovers.hpp"
#include "IntermediateRepresentation/IrMetadata/StageIO.hpp"
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ShaderRecompiler {
namespace {

// Address chains are short (a shift, a multiply-add of the group id, the immediate); deeper ones
// are given up rather than walked.
constexpr unsigned MaxDepth = 48;
// Conditions one store may rely on; a chain needing more is given up.
constexpr std::size_t MaxBounds = 6;

InvocationAffine add(const InvocationAffine& left, const InvocationAffine& right) {
    InvocationAffine sum;
    sum.constant = left.constant + right.constant;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        sum.group[axis] = left.group[axis] + right.group[axis];
        sum.local[axis] = left.local[axis] + right.local[axis];
    }
    return sum;
}

InvocationAffine scale(const InvocationAffine& form, std::uint32_t factor) {
    InvocationAffine scaled;
    scaled.constant = form.constant * factor;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        scaled.group[axis] = form.group[axis] * factor;
        scaled.local[axis] = form.local[axis] * factor;
    }
    return scaled;
}

bool constantOnly(const InvocationAffine& form) {
    return form.group == std::array<std::uint32_t, 3>{} && form.local == std::array<std::uint32_t, 3>{};
}

bool uniform(const InvocationAffine& form) {
    return form.local == std::array<std::uint32_t, 3>{};
}

// Low bits every value of the form has clear (32 for the constant zero): the constant and every
// coefficient are multiples of 2^n, so is any sum of them modulo 2^32.
unsigned trailingZeros(const InvocationAffine& form) {
    unsigned zeros = 32;
    const auto take = [&](std::uint32_t value) {
        if (value != 0u) zeros = std::min(zeros, static_cast<unsigned>(std::countr_zero(value)));
    };
    take(form.constant);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        take(form.group[axis]);
        take(form.local[axis]);
    }
    return zeros;
}

bool appendBounds(std::vector<InvocationBound>& into, const std::vector<InvocationBound>& from) {
    for (const auto& bound : from) {
        if (std::find(into.begin(), into.end(), bound) != into.end()) continue;
        if (into.size() == MaxBounds) return false;
        into.push_back(bound);
    }
    return true;
}

bool appendBound(std::vector<InvocationBound>& into, const InvocationBound& bound) {
    return appendBounds(into, std::vector<InvocationBound>{bound});
}

std::uint32_t storeBytes(IrOpcode opcode) {
    switch (opcode) {
        case IrOpcode::StoreBufferU8: return 1u;
        case IrOpcode::StoreBufferU16: return 2u;
        case IrOpcode::StoreBufferU32: return 4u;
        case IrOpcode::StoreBufferU32x2: return 8u;
        case IrOpcode::StoreBufferU32x3: return 12u;
        case IrOpcode::StoreBufferU32x4: return 16u;
        default: return 0u;
    }
}

}

BufferStoreCoverAnalysis::BufferStoreCoverAnalysis(const IrProgram& program) {
    computeStage = program.Resources().stage == IrShaderStage::Compute;
    if (!computeStage) return;
    // Terminators live in the block metadata, paired with BlockOrder by position (as the SPIR-V
    // emitter pairs them); BlockOrder starts with the synthetic entry block.
    const auto& order = program.BlockOrder();
    const auto& info = program.Metadata().blockInfo;
    if (order.empty() || order.size() != info.size()) {
        computeStage = false;
        return;
    }
    const auto position = [&](const IrBlock* block) {
        return static_cast<std::size_t>(std::find(order.begin(), order.end(), block) - order.begin());
    };
    const IrBlock* block = order.front();
    while (block != nullptr && everyPath.insert(block).second) {
        const auto index = position(block);
        if (index >= info.size()) break;
        const auto& terminator = info[index].terminator;
        if (terminator.kind != TerminatorKind::Branch) break;
        const auto target = std::find_if(info.begin(), info.end(), [&](const BlockInfo& entry) { return entry.id == terminator.trueBlock; });
        if (target == info.end()) break;
        block = order[static_cast<std::size_t>(target - info.begin())];
    }
}

const BufferStoreCoverAnalysis::Form& BufferStoreCoverAnalysis::affine(const IrValue* value, unsigned depth) {
    value = value->Resolve();
    if (const auto found = memo.find(value); found != memo.end()) return found->second;
    auto form = compute(value, depth);
    return memo.emplace(value, std::move(form)).first->second;
}

BufferStoreCoverAnalysis::Form BufferStoreCoverAnalysis::compute(const IrValue* value, unsigned depth) {
    Form result;
    if (depth > MaxDepth) return result;
    if (value->HasImmediate()) {
        if (value->Type() != IrType::U32 && value->Type() != IrType::S32) return result;
        result.valid = true;
        result.value.constant = value->ImmediateU32();
        return result;
    }
    const auto argument = [&](std::size_t index) -> const Form& { return affine(value->Argument(index), depth + 1u); };
    const auto constantArgument = [&](std::size_t index, std::uint32_t& out) {
        const auto& form = argument(index);
        if (!form.valid || !constantOnly(form.value)) return false;
        out = form.value.constant;
        return true;
    };
    // A constant argument may itself rely on bounds (a fold of masked values): they go with the result.
    const auto withBoundsOf = [&](Form form, std::initializer_list<std::size_t> indices) {
        for (const auto index : indices) {
            if (form.valid && !appendBounds(form.bounds, argument(index).bounds)) form.valid = false;
        }
        return form;
    };
    // Taking `form` for a value known to fit in `limit` (a mask or extract it passes unchanged).
    const auto bounded = [&](const Form& form, std::uint32_t limit) {
        Form out;
        if (!form.valid) return out;
        if (constantOnly(form.value)) {
            out.valid = true;
            out.value.constant = form.value.constant & limit;
            out.bounds = form.bounds;
            return out;
        }
        out = form;
        if (!appendBound(out.bounds, {form.value, limit})) out.valid = false;
        return out;
    };
    switch (value->Opcode()) {
        case IrOpcode::GetBuiltin: {
            if (value->ArgumentCount() != 2u) return result;
            const IrValue* kind = value->Argument(0)->Resolve();
            const IrValue* component = value->Argument(1)->Resolve();
            if (!kind->HasImmediate() || !component->HasImmediate() || component->ImmediateU32() >= 3u) return result;
            const auto axis = component->ImmediateU32();
            switch (static_cast<StageInputKind>(kind->ImmediateU32())) {
                case StageInputKind::WorkgroupId: result.value.group[axis] = 1u; break;
                case StageInputKind::LocalInvocationId: result.value.local[axis] = 1u; break;
                default: return result;
            }
            result.valid = true;
            return result;
        }
        case IrOpcode::IAdd32:
        case IrOpcode::ISub32: {
            if (value->ArgumentCount() != 2u) return result;
            const auto& left = argument(0);
            const auto& right = argument(1);
            if (!left.valid || !right.valid) return result;
            result.value = add(left.value, value->Opcode() == IrOpcode::ISub32 ? scale(right.value, 0xffffffffu) : right.value);
            result.bounds = left.bounds;
            result.valid = appendBounds(result.bounds, right.bounds);
            return result;
        }
        case IrOpcode::IMul32: {
            if (value->ArgumentCount() != 2u) return result;
            std::uint32_t factor = 0;
            const bool rightConstant = constantArgument(1, factor);
            if (!rightConstant && !constantArgument(0, factor)) return result;
            const auto& other = argument(rightConstant ? 0 : 1);
            if (!other.valid) return result;
            result = other;
            result.value = scale(other.value, factor);
            return withBoundsOf(result, {rightConstant ? 1u : 0u});
        }
        case IrOpcode::ShiftLeftLogical32:
        case IrOpcode::IShiftLeft32: {
            if (value->ArgumentCount() != 2u) return result;
            std::uint32_t shift = 0;
            if (!constantArgument(1, shift) || shift >= 32u) return result;
            const auto& shifted = argument(0);
            if (!shifted.valid) return result;
            result = shifted;
            result.value = scale(shifted.value, 1u << shift);
            return withBoundsOf(result, {1u});
        }
        case IrOpcode::BitwiseOr32:
        case IrOpcode::IOr32: {
            if (value->ArgumentCount() != 2u) return result;
            const auto& left = argument(0);
            const auto& right = argument(1);
            if (!left.valid || !right.valid) return result;
            if (constantOnly(left.value) && constantOnly(right.value)) {
                result.valid = true;
                result.value.constant = left.value.constant | right.value.constant;
                result.bounds = left.bounds;
                result.valid = appendBounds(result.bounds, right.bounds);
                return result;
            }
            // An OR of a value below 2^n with one whose low n bits are clear is their sum.
            const bool leftLow = trailingZeros(left.value) <= trailingZeros(right.value);
            const auto& low = leftLow ? left : right;
            const auto& high = leftLow ? right : left;
            const auto zeros = trailingZeros(high.value);
            if (zeros == 0u) return result;
            result.bounds = left.bounds;
            if (!appendBounds(result.bounds, right.bounds)) return result;
            if (zeros < 32u) {
                const auto limit = (std::uint32_t{1} << zeros) - 1u;
                if (constantOnly(low.value)) {
                    if (low.value.constant > limit) return result;
                } else if (!appendBound(result.bounds, {low.value, limit})) {
                    return result;
                }
            }
            result.value = add(left.value, right.value);
            result.valid = true;
            return result;
        }
        case IrOpcode::BitwiseAnd32:
        case IrOpcode::IAnd32: {
            if (value->ArgumentCount() != 2u) return result;
            std::uint32_t mask = 0;
            const bool rightConstant = constantArgument(1, mask);
            if (!rightConstant && !constantArgument(0, mask)) return result;
            const auto& other = argument(rightConstant ? 0 : 1);
            if (!other.valid) return result;
            const auto maskIndex = rightConstant ? 1u : 0u;
            if (constantOnly(other.value)) {
                result = other;
                result.value.constant = other.value.constant & mask;
                return withBoundsOf(result, {maskIndex});
            }
            if (mask == 0xffffffffu) return withBoundsOf(other, {maskIndex});
            // A low mask passes a value that fits in it unchanged.
            if (((mask + 1u) & mask) != 0u) return result;
            return withBoundsOf(bounded(other, mask), {maskIndex});
        }
        case IrOpcode::BitFieldUExtract: {
            if (value->ArgumentCount() != 3u) return result;
            std::uint32_t offset = 0;
            std::uint32_t width = 0;
            if (!constantArgument(1, offset) || !constantArgument(2, width) || offset != 0u) return result;
            const auto& field = argument(0);
            if (!field.valid) return result;
            if (width == 0u) {
                result.valid = true;
                return withBoundsOf(result, {1u, 2u});
            }
            if (width >= 32u) return withBoundsOf(field, {1u, 2u});
            return withBoundsOf(bounded(field, (std::uint32_t{1} << width) - 1u), {1u, 2u});
        }
        case IrOpcode::ReadFirstLane:
        case IrOpcode::ReadLane: {
            if (value->ArgumentCount() == 0u) return result;
            const auto& read = argument(0);
            // A wave-uniform value reads the same from any lane.
            if (!read.valid || !uniform(read.value)) return result;
            return read;
        }
        default:
            return result;
    }
}

std::optional<BufferStoreCover> BufferStoreCoverAnalysis::Analyze(const IrValue& store, const IrBlock* block, const MemoryInfo& memory) {
    if (!computeStage || block == nullptr || !everyPath.contains(block)) return std::nullopt;
    const auto bytes = storeBytes(store.Opcode());
    if (bytes == 0u || store.ArgumentCount() != 6u) return std::nullopt;
    if (memory.kind != ResourceKind::Buffer || memory.formatted || memory.typed || memory.gpuDescriptor || memory.planningOnly) return std::nullopt;
    // Every launched lane stores: the exec operand is still the entry's constant true.
    const IrValue* exec = store.Argument(5)->Resolve();
    if (!exec->HasImmediate() || exec->Type() != IrType::Bool || !exec->ImmediateBool()) return std::nullopt;
    const auto& index = affine(store.Argument(1), 0u);
    const auto& offset = affine(store.Argument(2), 0u);
    const auto& scalar = affine(store.Argument(3), 0u);
    if (!index.valid || !offset.valid || !scalar.valid) return std::nullopt;
    BufferStoreCover cover;
    cover.bytes = bytes;
    // Naturally aligned stores only: the emulated stores address dwords (a 32-bit store at an
    // unaligned offset lands on the dword below it, a 16-bit one at offset 3 drops its high byte).
    const auto alignment = std::min(bytes, 4u);
    const auto aligned = [&](const InvocationAffine& form) {
        if (form.constant % alignment != 0u) return false;
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (form.group[axis] % alignment != 0u || form.local[axis] % alignment != 0u) return false;
        }
        return true;
    };
    if (memory.idxen) {
        // A whole record at the start of each indexed record: no byte offset at all.
        if (!constantOnly(offset.value) || !constantOnly(scalar.value)) return std::nullopt;
        if (offset.value.constant != 0u || scalar.value.constant != 0u || memory.offset != 0u) return std::nullopt;
        // Records start where the stride puts them; a driver takes only whole-stride records.
        cover.indexed = true;
        cover.offset = index.value;
        cover.bounds = index.bounds;
        if (!appendBounds(cover.bounds, offset.bounds) || !appendBounds(cover.bounds, scalar.bounds)) return std::nullopt;
        return cover;
    }
    if (!constantOnly(index.value) || index.value.constant != 0u) return std::nullopt;
    if (!aligned(offset.value) || !aligned(scalar.value) || memory.offset % alignment != 0u) return std::nullopt;
    cover.offset = offset.value;
    cover.scalarOffset = scalar.value;
    cover.immediate = memory.offset;
    cover.bounds = offset.bounds;
    if (!appendBounds(cover.bounds, scalar.bounds) || !appendBounds(cover.bounds, index.bounds)) return std::nullopt;
    return cover;
}

}
