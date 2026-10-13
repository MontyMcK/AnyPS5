#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_DESCRIPTORBINDINGBUILDER_HPP

#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BindingAllocator.hpp"
#include "Recompiler.hpp"
#include <cstdint>
#include <string>

namespace ShaderRecompiler {

// depthReduction: every image the sampler reads is a depth-bits view (sampled through the float depth
// view), so a min or max reduction keeps its filtering there. integerOnly: every image it reads is a
// signed-integer image, which the texture unit never filters, so a reduction sampler point-samples.
std::uint32_t PointFilteredSamplerWord(std::uint32_t word0, std::uint32_t filter, bool depthReduction = false, bool integerOnly = false);
struct PreparedDescriptorBinding {
    DescriptorBinding descriptor;
    std::vector<std::uint32_t> resources;
    std::vector<bool> identityImageSwizzle;
    std::vector<std::uint32_t> samplerFilterElements;
    std::vector<bool> samplerDepthReduction;
    std::vector<bool> samplerIntegerOnly;
    std::vector<std::string> samplerPointWhy;
    std::size_t writtenBuffers = 0;
    std::size_t readOnlyBuffers = 0;
};

struct PreparedImageMetadata {
    std::uint32_t resource;
    std::uint32_t offset;
    RuntimeAbi::ResourceMetadata metadata;
};

struct DescriptorBindingPlan {
    std::vector<PreparedDescriptorBinding> bindings;
    std::vector<PreparedImageMetadata> imageMetadata;
    std::vector<PipelineSpecializationConstant> specialization;
    bool storageData = false;
};

class DescriptorBindingBuilder {
public:
    DescriptorBindingPlan Prepare(const IrBindingLayout& layout, const ShaderInfo& info, IrShaderStage stage, const ResourceSnapshot& snapshot, std::span<const std::uint8_t> exportMappings = {}) const;
    DescriptorBindingPlan Select(const DescriptorBindingPlan& plan, std::span<const std::uint32_t> bindings) const;
    void Populate(BindingAllocationResult& allocation, const CompiledBindingLayout& compiled, const DescriptorBindingPlan& plan, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads) const;
    void Populate(BindingAllocationResult& allocation, const IrProgram& program, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
    void Populate(BindingAllocationResult& allocation, const ShaderInfo& info, IrShaderStage stage, std::uint32_t userDataBase, const ResourceSnapshot& snapshot, const std::array<std::uint32_t, 3>& partialThreads, std::span<const std::uint8_t> exportMappings = {}) const;
};

}

#endif
