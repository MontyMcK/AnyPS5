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
// After DefineBdaDwordReadFunctions, for programs with anchors: the lookup path of a dword read
// (EmitBdaDwordReads) as one function per [stops invocations][coherent], so the rarely taken
// fallback of an anchored read is one call instead of its probe, byte reads and fault checks
// inline (which grew NVIDIA's native code by ~40% and spilled registers in PPSA01317's 4K passes).
void DefineBdaSlowReadFunctions(SpirvEmitterState& state);
// The dwords one call of the outlined function reads (a mask of them; a group of reads within
// 64 bytes shares one call, see EmitAnchoredBdaGroupRead).
inline constexpr std::uint32_t BdaSlowReadDwords = 16;
// The outlined function for `inst`'s reads (its coherence and the program's stop mode), 0 if none.
std::uint32_t BdaSlowReadFunction(const SpirvEmitterState& state, const IrValue& inst);
// Adjacent anchored dword reads of one guest instruction with the same base, lane offset `low`
// (anchor form) and `baseLow` (GuestAddressBase form) whose dwords lie in 64 bytes: the dwords of
// `used` (bit d: the dword at offset + 4 d) through the anchor where it holds them all, else
// through one call of the outlined function. Each member, read alone, takes the anchor's path
// exactly when the anchor holds it, and a held member's lookup path reads the same value without
// a fault; the members' faults keep their order (dword order is program order). Needs
// BdaSlowReadFunction(state, inst).
std::array<std::uint32_t, BdaSlowReadDwords> EmitAnchoredBdaGroupRead(SpirvEmitterState& state, const IrValue& inst, const SpirvEmitterState::BdaAnchor& anchor, std::uint32_t low, std::uint32_t offset, std::uint32_t used, std::uint32_t base, std::uint32_t baseLow);
// EmitBdaDwordReads(address(), offset, dwords) through the outlined function where it is defined,
// with the address base + zext(low) formed inside it as GuestAddressBase forms it (`address`
// emits that inline when there is no function). The same reads, faults and stops in the same
// order; for the fallback of an anchored read.
std::array<std::uint32_t, 4> EmitBdaDwordReadsOutlined(SpirvValueEmitContext& ctx, const IrValue& inst, std::uint32_t base, std::uint32_t low, std::uint32_t offset, std::uint32_t dwords, const std::function<std::uint32_t()>& address);
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
