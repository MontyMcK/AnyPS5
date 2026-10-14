#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBDA_HPP

#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "SpirvBackend/SpirvEmitter.hpp"
#include "BdaAbi.hpp"
#include <array>
#include <functional>

namespace ShaderRecompiler {

std::uint32_t BdaConstant(SpirvEmitterState& state, std::uint64_t value);
std::uint32_t BdaWord(SpirvEmitterState& state, std::uint32_t variable, std::uint32_t index);
std::uint32_t BdaLoadWord(SpirvEmitterState& state, std::uint32_t index);
std::uint32_t BdaLoadAddress(SpirvEmitterState& state, std::uint32_t index);
void DefineBdaFaultFunction(SpirvEmitterState& state);
void DefineBdaDwordReadFunctions(SpirvEmitterState& state);
void RecordBdaFault(SpirvEmitterState& state, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ReturnBdaFailureIf(SpirvEmitterState& state, std::uint32_t condition, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, BdaAbi::FaultReason reason);
void ValidateBdaTarget(const IrProgram& program, const SpirvTargetOptions& target);
// Whether a faulting BDA access may end its invocation. Programs with workgroup barriers must keep
// every invocation running, so their faulting reads return zero instead.
bool BdaInvocationsMayStop(const IrProgram& program);
void StopBdaInvocationIf(SpirvEmitterState& state, std::uint32_t condition);
std::uint32_t EmitBdaRead(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bits);
void EmitBdaWrite(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits = 32u);
void EmitBdaStore(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t value, std::uint32_t bits);
std::uint32_t EmitBdaAtomic(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t bytes, const std::function<std::uint32_t(std::uint32_t)>& operation);
// Reads the dwords of a 1-4 dword load at address + offset that the program extracts, with one
// table lookup for the whole span; the per-byte lookups of EmitBdaRead remain the fallback (and
// the only path under APS5_BDA_BYTE_READS=1). Dwords the program never extracts are neither read
std::array<std::uint32_t, 4> EmitBdaDwordReads(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, std::uint32_t dwords, bool everyDword = false);
bool BdaByteReadsForced();
bool BdaHintFastPath();
bool ElectScalarBdaReads();
// APS5_BDA_INLINE_HIT=1: a lookup call (get/probe) tests the lookup cache inline and calls only on a
// miss. Part of the shader disk cache version.
bool BdaInlineCacheHit();
std::uint32_t EmitBdaLookupCall(SpirvEmitterState& state, std::uint32_t function, std::uint32_t address, std::uint32_t bytes, std::uint32_t instruction, std::uint32_t permission);
// APS5_BDA_CLOCK=1 (measurement): the entry clock at the program's start and, at each return, one
// lane per subgroup adds the lookup cycles, the invocation's cycles and its lookups to the fault
// buffer's clock words (see BdaAbi::ClockLookupWord).
void EmitBdaClockEntry(SpirvEmitterState& state);
void EmitBdaClockReport(SpirvEmitterState& state);
// Whether `inst` reads through a scalar memory instruction (its address is wave-uniform).
bool ScalarBdaAccess(const IrProgram& program, const IrValue& inst);
std::uint32_t AddBdaAddress(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::uint32_t offset, bool subtract);
// Anchors (SpirvEmitterState::BdaAnchor): CollectBdaAnchors lists the user data register pairs that
// reads use as their base (after the lookups are defined), EmitBdaAnchors finds their ranges at the
// program's entry, FindBdaAnchor returns the anchor of a read (null when it has none or the
// anchors are off).
void CollectBdaAnchors(SpirvEmitterState& state);
void DefineBdaAnchorFunction(SpirvEmitterState& state);
void EmitBdaAnchors(SpirvEmitterState& state);
// The range holding `base` (u64) into `anchor`. `base` must be the same in every invocation of the
// subgroup: with subgroup ballots the subgroup's elected lane searches for all.
void EmitBdaAnchorRange(SpirvEmitterState& state, SpirvEmitterState::BdaAnchor& anchor, std::uint32_t base);
const SpirvEmitterState::BdaAnchor* FindBdaAnchor(const SpirvEmitterState& state, const IrValue& inst);
// A read at base + zext(low) + immediate, base being the anchor's: of `bits` 8 or 16 (the value
// in [0]), or 32: the `used` dwords of `dwords` (0 for the others). Where the anchor holds the
// read (and, for dwords, its device address is 4-aligned) one branch leads to the loads; elsewhere
// `lookup` emits the read through the lookups, which checks the address arithmetic and records the
// faults, exactly as without the anchor.
std::array<std::uint32_t, 4> EmitAnchoredBdaRead(SpirvEmitterState& state, const IrValue& inst, const SpirvEmitterState::BdaAnchor& anchor, std::uint32_t low, std::int32_t immediate, std::uint32_t bits, std::uint32_t dwords, std::uint32_t used, const std::function<std::array<std::uint32_t, 4>()>& lookup);
// The dwords of a wide read the program extracts (see EmitBdaDwordReads).
std::uint32_t UsedBdaReadDwords(const IrValue& inst, std::uint32_t dwords);
std::uint32_t AddBdaImmediate(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t address, std::int32_t immediate);

}

#endif
