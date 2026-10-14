#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <cstdlib>

namespace ShaderRecompiler {

namespace {

void ReturnBdaZeroIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

void ReturnBdaRangeZeroIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t zero) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    state.module.AddFunction(spv::OpReturnValue, zero);
    EmitLabel(state, next);
}

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
// or with `rangeResult` (the anchor search, see EmitBdaAnchors) the u64 vector (begin, end, device
// address) of that range, zero when the lookup would fail. The range search does not read or fill
// the lookup cache.
std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read, bool rangeResult = false) {
    const bool cached = state.bdaCacheBegin != 0 && !rangeResult;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto load = [&](std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, u32, value, pointer);
        return value;
    };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    // APS5_BDA_INLINE_ALL=1 (experiment, in the cache version): the driver may inline the lookups,
    // so a call's register saves and restores go away at the cost of code size.
    static const bool inlineAll = std::getenv("APS5_BDA_INLINE_ALL") != nullptr;
    const auto rangeType = rangeResult ? state.module.Type(spv::OpTypeVector, u64, 3u) : 0u;
    const auto resultType = rangeResult ? rangeType : u64;
    state.module.AddFunction(spv::OpFunction, resultType, function, inlineAll || rangeResult ? spv::FunctionControlMaskNone : spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, resultType, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto low = state.module.AllocateId();
    const auto high = state.module.AllocateId();
    const auto pointer = TypePointer(state, spv::StorageClassFunction, u32);
    state.module.AddFunction(spv::OpVariable, pointer, low, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpVariable, pointer, high, spv::StorageClassFunction);
    const bool hinted = !state.bdaNoLookupState;
    if (hinted && state.bdaHintVariable == 0) state.bdaHintVariable = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, u32), spv::StorageClassPrivate, constant(0));
    if (cached) {
        const auto load64 = [&](std::uint32_t pointer) {
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, u64, value, pointer);
            return value;
        };
        const auto cachedBegin = load64(state.bdaCacheBegin);
        const auto accessEnd = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
        auto hit = binary(spv::OpUGreaterThanEqual, boolean, address, cachedBegin);
        hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpULessThanEqual, boolean, accessEnd, load64(state.bdaCacheEnd)));
        hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpUGreaterThan, boolean, accessEnd, address));
        hit = binary(spv::OpLogicalAnd, boolean, hit, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, load(state.bdaCachePermissions), constant(permission)), constant(permission)));
        const auto found = state.module.AllocateId();
        const auto search = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, search, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, hit, found, search);
        EmitLabel(state, found);
        state.module.AddFunction(spv::OpReturnValue, binary(spv::OpIAdd, u64, load64(state.bdaCacheBase), binary(spv::OpISub, u64, address, cachedBegin)));
        EmitLabel(state, search);
        if (state.bdaVictimBegin != 0) {
            // The range before the current one: a shader alternating between two ranges finds the
            // other here and swaps the two, without searching the table.
            const auto victimBegin = load64(state.bdaVictimBegin);
            const auto victimEnd = load64(state.bdaVictimEnd);
            const auto victimPermissions = load(state.bdaVictimPermissions);
            auto victimHit = binary(spv::OpUGreaterThanEqual, boolean, address, victimBegin);
            victimHit = binary(spv::OpLogicalAnd, boolean, victimHit, binary(spv::OpULessThanEqual, boolean, accessEnd, victimEnd));
            victimHit = binary(spv::OpLogicalAnd, boolean, victimHit, binary(spv::OpUGreaterThan, boolean, accessEnd, address));
            victimHit = binary(spv::OpLogicalAnd, boolean, victimHit, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, victimPermissions, constant(permission)), constant(permission)));
            const auto swapped = state.module.AllocateId();
            const auto victimMiss = state.module.AllocateId();
            state.module.AddFunction(spv::OpSelectionMerge, victimMiss, spv::SelectionControlMaskNone);
            state.module.AddFunction(spv::OpBranchConditional, victimHit, swapped, victimMiss);
            EmitLabel(state, swapped);
            const auto victimBase = load64(state.bdaVictimBase);
            const auto mainEnd = load64(state.bdaCacheEnd);
            const auto mainBase = load64(state.bdaCacheBase);
            const auto mainPermissions = load(state.bdaCachePermissions);
            state.module.AddFunction(spv::OpStore, state.bdaVictimBegin, cachedBegin);
            state.module.AddFunction(spv::OpStore, state.bdaVictimEnd, mainEnd);
            state.module.AddFunction(spv::OpStore, state.bdaVictimBase, mainBase);
            state.module.AddFunction(spv::OpStore, state.bdaVictimPermissions, mainPermissions);
            state.module.AddFunction(spv::OpStore, state.bdaCacheBegin, victimBegin);
            state.module.AddFunction(spv::OpStore, state.bdaCacheEnd, victimEnd);
            state.module.AddFunction(spv::OpStore, state.bdaCacheBase, victimBase);
            state.module.AddFunction(spv::OpStore, state.bdaCachePermissions, victimPermissions);
            state.module.AddFunction(spv::OpReturnValue, binary(spv::OpIAdd, u64, victimBase, binary(spv::OpISub, u64, address, victimBegin)));
            EmitLabel(state, victimMiss);
        }
    }
    if (BdaHintFastPath() && !rangeResult && hinted) {
        const auto tableWords = state.module.AllocateId();
        state.module.AddFunction(spv::OpArrayLength, u32, tableWords, state.bdaPagetableVariable, 0u);
        const auto headerOk = binary(spv::OpUGreaterThanEqual, boolean, tableWords, constant(4));
        const auto headerBlock = state.module.AllocateId();
        const auto headerDone = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, headerDone, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, headerOk, headerBlock, headerDone);
        EmitLabel(state, headerBlock);
        const auto hinted = load(state.bdaHintVariable);
        const auto entries = BdaLoadWord(state, constant(1));
        auto valid = binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, hinted, constant(0)), binary(spv::OpULessThanEqual, boolean, hinted, entries));
        valid = binary(spv::OpLogicalAnd, boolean, valid, binary(spv::OpULessThanEqual, boolean, entries, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, tableWords, constant(4)), constant(3))));
        const auto entryBlock = state.module.AllocateId();
        const auto entryDone = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, entryDone, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, valid, entryBlock, entryDone);
        EmitLabel(state, entryBlock);
        const auto slot = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, hinted, constant(1)), constant(8)));
        const auto stop = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
        const auto first = BdaLoadAddress(state, slot);
        const auto last = BdaLoadAddress(state, binary(spv::OpIAdd, u32, slot, constant(2)));
        const auto origin = BdaLoadAddress(state, binary(spv::OpIAdd, u32, slot, constant(4)));
        const auto allowed = BdaLoadWord(state, binary(spv::OpIAdd, u32, slot, constant(6)));
        const auto reserved = BdaLoadWord(state, binary(spv::OpIAdd, u32, slot, constant(7)));
        const auto translated = binary(spv::OpIAdd, u64, origin, binary(spv::OpISub, u64, address, first));
        auto quick = binary(spv::OpULessThanEqual, boolean, first, address);
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpULessThan, boolean, address, stop));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpULessThanEqual, boolean, stop, last));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, allowed, constant(permission)), constant(permission)));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpINotEqual, boolean, origin, BdaConstant(state, 0)));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpIEqual, boolean, reserved, constant(0)));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpULessThan, boolean, first, last));
        quick = binary(spv::OpLogicalAnd, boolean, quick, binary(spv::OpUGreaterThanEqual, boolean, translated, origin));
        const auto taken = state.module.AllocateId();
        const auto quickDone = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, quickDone, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, quick, taken, quickDone);
        EmitLabel(state, taken);
        state.module.AddFunction(spv::OpReturnValue, translated);
        EmitLabel(state, quickDone);
        state.module.AddFunction(spv::OpBranch, entryDone);
        EmitLabel(state, entryDone);
        state.module.AddFunction(spv::OpBranch, headerDone);
        EmitLabel(state, headerDone);
    }
    const auto rangeZero = rangeResult ? state.module.Constant(spv::OpConstantComposite, rangeType, BdaConstant(state, 0u), BdaConstant(state, 0u), BdaConstant(state, 0u)) : 0u;
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        if (rangeResult) {
            ReturnBdaRangeZeroIf(state, condition, rangeZero);
        } else if (recordFaults) {
            ReturnBdaFailureIf(state, condition, address, bytes, instruction, reason);
        } else {
            ReturnBdaZeroIf(state, condition);
        }
    };
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    const auto buckets = BdaLoadWord(state, constant(3));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    fail(binary(spv::OpUGreaterThan, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), BdaAbi::FaultReason::InvalidTable);
    const auto rest = binary(spv::OpISub, u32, available, binary(spv::OpShiftLeftLogical, u32, count, constant(3)));
    fail(binary(spv::OpUGreaterThan, boolean, buckets, rest), BdaAbi::FaultReason::InvalidTable);
    const auto hasBuckets = binary(spv::OpINotEqual, boolean, buckets, constant(0));
    const auto indexWords = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, indexWords, hasBuckets, binary(spv::OpIAdd, u32, constant(4), binary(spv::OpShiftLeftLogical, u32, buckets, constant(1))), constant(0));
    fail(binary(spv::OpINotEqual, boolean, indexWords, rest), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow);
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow);
    state.module.AddFunction(spv::OpStore, low, constant(0));
    state.module.AddFunction(spv::OpStore, high, count);
    if (hinted) {
        const auto hint = load(state.bdaHintVariable);
        const auto inHint = binary(spv::OpLogicalAnd, boolean, binary(spv::OpINotEqual, boolean, hint, constant(0)), binary(spv::OpULessThanEqual, boolean, hint, count));
        EmitIfCondition(state, inHint, [&] {
            const auto candidate = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, hint, constant(1)), constant(8)));
            const auto inside = binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThanEqual, boolean, BdaLoadAddress(state, candidate), address), binary(spv::OpULessThanEqual, boolean, end, BdaLoadAddress(state, binary(spv::OpIAdd, u32, candidate, constant(2)))));
            EmitIfCondition(state, inside, [&] {
                state.module.AddFunction(spv::OpStore, low, hint);
                state.module.AddFunction(spv::OpStore, high, hint);
            });
        });
    }
    EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, hasBuckets, binary(spv::OpULessThan, boolean, load(low), load(high))), [&] {
        const auto index = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpShiftLeftLogical, u32, count, constant(3)));
        const auto first = BdaLoadAddress(state, index);
        const auto shift = BdaLoadWord(state, binary(spv::OpIAdd, u32, index, constant(2)));
        EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, binary(spv::OpUGreaterThanEqual, boolean, address, first), binary(spv::OpULessThan, boolean, shift, constant(64))), [&] {
            const auto bucket = binary(spv::OpShiftRightLogical, u64, binary(spv::OpISub, u64, address, first), Unary(state, spv::OpUConvert, u64, shift));
            EmitIfCondition(state, binary(spv::OpULessThan, boolean, bucket, Unary(state, spv::OpUConvert, u64, buckets)), [&] {
                const auto entry = binary(spv::OpIAdd, u32, binary(spv::OpIAdd, u32, index, constant(4)), binary(spv::OpShiftLeftLogical, u32, Unary(state, spv::OpUConvert, u32, bucket), constant(1)));
                const auto bucketLow = BdaLoadWord(state, entry);
                const auto bucketHigh = BdaLoadWord(state, binary(spv::OpIAdd, u32, entry, constant(1)));
                EmitIfCondition(state, binary(spv::OpLogicalAnd, boolean, binary(spv::OpULessThanEqual, boolean, bucketLow, bucketHigh), binary(spv::OpULessThanEqual, boolean, bucketHigh, count)), [&] {
                    state.module.AddFunction(spv::OpStore, low, bucketLow);
                    state.module.AddFunction(spv::OpStore, high, bucketHigh);
                });
            });
        });
    });
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = load(low);
    const auto upper = load(high);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpStore, low, newLow);
    state.module.AddFunction(spv::OpStore, high, newHigh);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = load(low);
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable);
    fail(binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped);
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission);
    fail(binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable);
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    fail(binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow);
    if (rangeResult) {
        const auto range = state.module.AllocateId();
        state.module.AddFunction(spv::OpCompositeConstruct, rangeType, range, begin, finish, base);
        state.module.AddFunction(spv::OpReturnValue, range);
        state.module.AddFunction(spv::OpFunctionEnd);
        return function;
    }
    if (hinted) state.module.AddFunction(spv::OpStore, state.bdaHintVariable, index);
    if (cached) {
        if (state.bdaVictimBegin != 0) {
            const auto demote = [&](std::uint32_t from, std::uint32_t to, std::uint32_t type) {
                const auto value = state.module.AllocateId();
                state.module.AddFunction(spv::OpLoad, type, value, from);
                state.module.AddFunction(spv::OpStore, to, value);
            };
            demote(state.bdaCacheBegin, state.bdaVictimBegin, u64);
            demote(state.bdaCacheEnd, state.bdaVictimEnd, u64);
            demote(state.bdaCacheBase, state.bdaVictimBase, u64);
            demote(state.bdaCachePermissions, state.bdaVictimPermissions, u32);
        }
        state.module.AddFunction(spv::OpStore, state.bdaCacheBegin, begin);
        state.module.AddFunction(spv::OpStore, state.bdaCacheEnd, finish);
        state.module.AddFunction(spv::OpStore, state.bdaCacheBase, base);
        state.module.AddFunction(spv::OpStore, state.bdaCachePermissions, permissions);
    }
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    const auto page = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, TypeScalarU64(state), address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(1));
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, page, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    for (std::uint32_t probe = 0; probe < BdaAbi::WrittenPageProbes; probe++) {
        const auto pending = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, boolean, pending, done);
        EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, pending), [&] {
            const auto slot = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, constant(probe)), constant(BdaAbi::WrittenPageSlots - 1u)));
            const auto previous = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, slot), scope, relaxed, relaxed, page, constant(0u));
            const auto taken = binary(spv::OpLogicalOr, boolean, binary(spv::OpIEqual, boolean, previous, constant(0u)), binary(spv::OpIEqual, boolean, previous, page));
            state.module.AddFunction(spv::OpStore, done, taken);
        });
    }
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

static bool BdaLookupCached() {
    static const bool disabled = std::getenv("APS5_NO_BDA_LOOKUP_CACHE") != nullptr;
    return !disabled;
}

static bool BdaVictimCache() {
    static const bool enabled = std::getenv("APS5_BDA_VICTIM") != nullptr;
    return enabled;
}

bool BdaInlineCacheHit() {
    static const bool enabled = std::getenv("APS5_BDA_INLINE_HIT") != nullptr;
    return enabled;
}

static std::uint32_t EmitBdaLookupCallInner(SpirvEmitterState& state, std::uint32_t function, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t permission);

bool BdaClockEnabled() {
    static const bool enabled = std::getenv("APS5_BDA_CLOCK") != nullptr;
    return enabled;
}

std::uint32_t EmitBdaClock(SpirvEmitterState& state) {
    state.module.EmitCapability(spv::CapabilityShaderClockKHR);
    state.module.EmitExtension("SPV_KHR_shader_clock");
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpReadClockKHR, TypeScalarU64(state), value, ConstantU32(state, static_cast<std::uint32_t>(spv::ScopeSubgroup)));
    return value;
}

std::uint32_t EmitBdaLookupCall(SpirvEmitterState& state, std::uint32_t function, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t permission) {
    if (state.bdaClockLookup == 0) return EmitBdaLookupCallInner(state, function, address, bytes, instruction, permission);
    const auto u64 = TypeScalarU64(state);
    const auto u32 = TypeU32(state);
    const auto before = EmitBdaClock(state);
    const auto result = EmitBdaLookupCallInner(state, function, address, bytes, instruction, permission);
    const auto after = EmitBdaClock(state);
    const auto spent = Binary(state, spv::OpISub, u64, after, before);
    const auto sum = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u64, sum, state.bdaClockLookup);
    state.module.AddFunction(spv::OpStore, state.bdaClockLookup, Binary(state, spv::OpIAdd, u64, sum, spent));
    const auto calls = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, calls, state.bdaClockCalls);
    state.module.AddFunction(spv::OpStore, state.bdaClockCalls, Binary(state, spv::OpIAdd, u32, calls, ConstantU32(state, 1u)));
    return result;
}

static std::uint32_t EmitBdaLookupCallInner(SpirvEmitterState& state, std::uint32_t function, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t permission) {
    const auto u64 = TypeScalarU64(state);
    const auto call = [&] {
        const auto result = state.module.AllocateId();
        state.module.AddFunction(spv::OpFunctionCall, u64, result, function, address, bytes, instruction);
        return result;
    };
    if (!BdaInlineCacheHit() || state.bdaCacheBegin == 0) return call();
    // The lookup's own cache test (DefineBdaLookup), made here so a hit costs no call: NVIDIA keeps
    // DontInline functions as calls. A miss calls the lookup, which searches and refills the cache.
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto begin = load(u64, state.bdaCacheBegin);
    const auto accessEnd = Binary(state, spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    auto hit = Binary(state, spv::OpUGreaterThanEqual, boolean, address, begin);
    hit = Binary(state, spv::OpLogicalAnd, boolean, hit, Binary(state, spv::OpULessThanEqual, boolean, accessEnd, load(u64, state.bdaCacheEnd)));
    hit = Binary(state, spv::OpLogicalAnd, boolean, hit, Binary(state, spv::OpUGreaterThan, boolean, accessEnd, address));
    hit = Binary(state, spv::OpLogicalAnd, boolean, hit, Binary(state, spv::OpIEqual, boolean, Binary(state, spv::OpBitwiseAnd, u32, load(u32, state.bdaCachePermissions), ConstantU32(state, permission)), ConstantU32(state, permission)));
    const auto hitLabel = state.module.AllocateId();
    const auto missLabel = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, hit, hitLabel, missLabel);
    EmitLabel(state, hitLabel);
    const auto translated = Binary(state, spv::OpIAdd, u64, load(u64, state.bdaCacheBase), Binary(state, spv::OpISub, u64, address, begin));
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, missLabel);
    const auto searched = call();
    const auto missExit = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, merge);
    EmitLabel(state, merge);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u64, result, translated, hitLabel, searched, missExit);
    return result;
}

static void DefineBdaLookupCache(SpirvEmitterState& state) {
    const auto address = TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state));
    const auto zero = BdaConstant(state, 0u);
    state.bdaCacheBegin = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
    state.bdaCacheEnd = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
    state.bdaCacheBase = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
    state.bdaCachePermissions = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
    if (BdaVictimCache()) {
        state.bdaVictimBegin = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaVictimEnd = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaVictimBase = state.module.DefineInitializedGlobalVariable(address, spv::StorageClassPrivate, zero);
        state.bdaVictimPermissions = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
    }
}

void EmitBdaClockEntry(SpirvEmitterState& state) {
    if (state.bdaClockEntry == 0) return;
    state.module.AddFunction(spv::OpStore, state.bdaClockEntry, EmitBdaClock(state));
}

void EmitBdaClockReport(SpirvEmitterState& state) {
    if (state.bdaClockEntry == 0 || state.faultBufferVariable == 0) return;
    const auto u64 = TypeScalarU64(state);
    const auto u32 = TypeU32(state);
    const auto load = [&](std::uint32_t type, std::uint32_t pointer) {
        const auto value = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, type, value, pointer);
        return value;
    };
    const auto total = Binary(state, spv::OpISub, u64, EmitBdaClock(state), load(u64, state.bdaClockEntry));
    const auto lookup = load(u64, state.bdaClockLookup);
    const auto calls = load(u32, state.bdaClockCalls);
    state.module.EmitCapability(spv::CapabilityGroupNonUniform);
    const auto elected = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected, ConstantU32(state, spv::ScopeSubgroup));
    EmitIfCondition(state, elected, [&] {
        const auto add = [&](std::uint32_t word, std::uint32_t value) {
            const auto previous = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, previous, BdaWord(state, state.faultBufferVariable, ConstantU32(state, word)), ConstantU32(state, spv::ScopeDevice), ConstantU32(state, spv::MemorySemanticsMaskNone), value);
        };
        const auto units = [&](std::uint32_t cycles) { return Unary(state, spv::OpUConvert, u32, Binary(state, spv::OpShiftRightLogical, u64, cycles, BdaConstant(state, 6u))); };
        add(BdaAbi::ClockLookupWord, units(lookup));
        add(BdaAbi::ClockTotalWord, units(total));
        add(BdaAbi::ClockCallsWord, calls);
    });
}

void DefineBdaAnchorFunction(SpirvEmitterState& state) {
    if (state.bdaAnchorFunction == 0) state.bdaAnchorFunction = DefineBdaLookup(state, "bda_anchor_range", false, BdaAbi::Read, true);
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (!state.program.Info().usesDma) return;
    if (BdaClockEnabled()) {
        const auto u64Pointer = TypePointer(state, spv::StorageClassPrivate, TypeScalarU64(state));
        state.bdaClockEntry = state.module.DefineInitializedGlobalVariable(u64Pointer, spv::StorageClassPrivate, BdaConstant(state, 0u));
        state.bdaClockLookup = state.module.DefineInitializedGlobalVariable(u64Pointer, spv::StorageClassPrivate, BdaConstant(state, 0u));
        state.bdaClockCalls = state.module.DefineInitializedGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate, ConstantU32(state, 0u));
    }
    CollectBdaAnchors(state);
    // A program with anchors keeps no lookup state (the cache and the hint) across its lookups:
    // Private variables that the lookup functions and the program share hold registers through
    // the whole program on NVIDIA (anchored 4K passes of PPSA01317 ran 3.2 ms with them, 3.0
    // without), and the anchors leave the lookups to the rare reads they miss. Neither changes
    // what a lookup returns.
    state.bdaNoLookupState = state.bdaNoLookupState || !state.bdaAnchors.empty();
    if (BdaLookupCached() && !state.bdaNoLookupState) DefineBdaLookupCache(state);
    DefineBdaFaultFunction(state);
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    if (!state.bdaAnchors.empty()) DefineBdaAnchorFunction(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
    }
}

}
