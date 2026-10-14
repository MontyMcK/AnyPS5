#include "BdaShader.hpp"
#include "SpirvBackend/SpirvBda.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"

namespace {

template<typename TEmit>
std::vector<std::uint32_t> MakeShader(std::uint64_t address, bool coherent, bool stops, const TEmit& emit) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    if (coherent) {
        MemoryInfo memory;
        memory.coherent = true;
        program.Resources().memoryInfo = {memory};
    }
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    const auto output = define(2);
    std::vector<std::uint32_t> interfaceVariables;
    if (!stops) {
        interfaceVariables.push_back(state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32(state)), spv::StorageClassInput));
        state.module.AddAnnotation(spv::OpDecorate, interfaceVariables[0], spv::DecorationBuiltIn, spv::BuiltInLocalInvocationIndex);
    }
    state.bdaStopsInvocations = stops;
    DefineGetBdaPointer(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    SpirvValueEmitContext ctx(state);
    auto& instruction = program.CreateValue(IrOpcode::LoadAddressU32, IrType::U32);
    MemoryFlags flags{};
    flags.pc = 0x1234;
    instruction.SetFlags(flags);
    auto base = BdaConstant(state, address);
    if (!stops) {
        const auto index = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), index, interfaceVariables[0]);
        base = Binary(state, spv::OpIAdd, TypeScalarU64(state), base, Unary(state, spv::OpUConvert, TypeScalarU64(state), index));
    }
    const auto values = emit(ctx, instruction, base);
    for (std::uint32_t index = 0; index < values.size(); ++index) state.module.AddFunction(spv::OpStore, BdaWord(state, output, ConstantU32(state, index)), values[index]);
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, 1u, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", interfaceVariables);
    return state.module.Finalize();
}

}

std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset) {
    using namespace ShaderRecompiler;
    return MakeShader(address, false, true, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        if (offset != 0) base = AddBdaAddress(ctx, instruction, base, BdaConstant(ctx.state, offset < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(offset) : static_cast<std::uint64_t>(offset)), offset < 0);
        return std::vector<std::uint32_t>{EmitBdaRead(ctx, instruction, base, bits)};
    });
}

std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops) {
    using namespace ShaderRecompiler;
    return MakeShader(address, coherent, stops, [&](SpirvValueEmitContext& ctx, const IrValue& instruction, std::uint32_t base) {
        const auto values = EmitBdaDwordReads(ctx, instruction, base, 0u, dwords, true);
        return std::vector<std::uint32_t>(values.begin(), values.begin() + dwords);
    });
}

// A read at anchor + zext(low + lane * stride) + immediate by each of `lanes` invocations (lane:
// the local invocation index), through EmitAnchoredBdaRead with the anchor of `anchor` when
// `anchored` (and, as in a program with anchors, no lookup cache or hint), else through the lookups
// alone with the same address arithmetic. A 32-bit read reads `dwords` dwords and the program
// extracts those of `used`; lane L stores its dwords at output words 4L to 4L + 3.
std::vector<std::uint32_t> MakeBdaAnchoredReadTestShader(const BdaAnchoredRead& read, bool anchored) {
    using namespace ShaderRecompiler;
    IrProgram program;
    program.Resources().stage = IrShaderStage::Compute;
    program.Info().usesDma = true;
    SpirvEmitterState state(program, {});
    EmitBaseHeader(state.module, program);
    if (read.elect) {
        state.module.EmitCapability(spv::CapabilityGroupNonUniform);
        state.module.EmitCapability(spv::CapabilityGroupNonUniformBallot);
        state.requirements.subgroupBallot = true;
    }
    const auto define = [&](std::uint32_t binding) {
        const auto variable = state.module.DefineGlobalVariable(TypeStorageBufferPointer(state), spv::StorageClassStorageBuffer);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 0u);
        state.module.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
        return variable;
    };
    state.bdaPagetableVariable = define(0);
    state.faultBufferVariable = define(1);
    const auto output = define(2);
    std::vector<std::uint32_t> interfaceVariables{state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassInput, TypeU32(state)), spv::StorageClassInput)};
    state.module.AddAnnotation(spv::OpDecorate, interfaceVariables[0], spv::DecorationBuiltIn, spv::BuiltInLocalInvocationIndex);
    state.bdaStopsInvocations = read.stops;
    state.bdaNoLookupState = anchored;
    DefineGetBdaPointer(state);
    if (anchored) DefineBdaAnchorFunction(state);
    const auto main = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, TypeVoid(state), main, spv::FunctionControlMaskNone, TypeFunction(state));
    EmitLabel(state, state.module.AllocateId());
    SpirvValueEmitContext ctx(state);
    SpirvEmitterState::BdaAnchor range{};
    const auto base = BdaConstant(state, read.anchor);
    if (anchored) EmitBdaAnchorRange(state, range, base);
    auto& instruction = program.CreateValue(IrOpcode::LoadAddressU32, IrType::U32);
    MemoryFlags flags{};
    flags.pc = 0x1234;
    instruction.SetFlags(flags);
    // The dwords the program extracts: one extract per used dword (UsedBdaReadDwords reads them).
    if (read.bits == 32u && read.dwords > 1u) {
        const auto extract = read.dwords == 2u ? IrOpcode::CompositeExtractU32x2 : read.dwords == 3u ? IrOpcode::CompositeExtractU32x3 : IrOpcode::CompositeExtractU32x4;
        for (std::uint32_t dword = 0; dword < read.dwords; ++dword) {
            if ((read.used & (1u << dword)) == 0u) continue;
            auto& index = program.CreateValue(IrOpcode::Void, IrType::U32);
            index.SetImmediateU32(dword);
            auto& user = program.CreateValue(extract, IrType::U32);
            user.AddArgument(&instruction);
            user.AddArgument(&index);
        }
    }
    const auto laneIndex = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), laneIndex, interfaceVariables[0]);
    const auto lane = Binary(state, spv::OpIAdd, TypeU32(state), ConstantU32(state, read.low), Binary(state, spv::OpIMul, TypeU32(state), laneIndex, ConstantU32(state, read.stride)));
    const auto lookup = [&] {
        const auto address = AddBdaAddress(ctx, instruction, base, Unary(state, spv::OpUConvert, TypeScalarU64(state), lane), false);
        if (read.bits == 32u) return EmitBdaDwordReads(ctx, instruction, address, static_cast<std::uint32_t>(read.immediate), read.dwords);
        return std::array<std::uint32_t, 4>{EmitBdaRead(ctx, instruction, AddBdaImmediate(ctx, instruction, address, read.immediate), read.bits)};
    };
    const auto used = read.bits == 32u ? UsedBdaReadDwords(instruction, read.dwords) : 1u;
    const auto values = anchored ? EmitAnchoredBdaRead(state, instruction, range, lane, read.immediate, read.bits, read.dwords, used, lookup) : lookup();
    const auto slot = Binary(state, spv::OpIMul, TypeU32(state), laneIndex, ConstantU32(state, 4u));
    for (std::uint32_t index = 0; index < (read.bits == 32u ? read.dwords : 1u); ++index) state.module.AddFunction(spv::OpStore, BdaWord(state, output, Binary(state, spv::OpIAdd, TypeU32(state), slot, ConstantU32(state, index))), values[index]);
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    state.module.AddExecutionMode(main, spv::ExecutionModeLocalSize, read.lanes, 1u, 1u);
    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, main, "main", interfaceVariables);
    return state.module.Finalize();
}
