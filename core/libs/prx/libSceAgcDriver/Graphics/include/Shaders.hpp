#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERS_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "Recompiler.hpp"
#include <array>
#include <cstddef>
#include <span>

namespace AgcDriver::Graphics {

inline constexpr std::uint32_t PipelinePushConstantBytes = 128;

struct CompiledShader {
    ShaderRecompiler::ShaderStage stage;
    const ShaderRecompiler::RecompileResult* program;
    std::uint32_t pushConstantOffset;
};

inline VkShaderStageFlagBits VulkanStage(ShaderRecompiler::ShaderStage stage) {
    switch (stage) {
        case ShaderRecompiler::ShaderStage::Vertex:
        case ShaderRecompiler::ShaderStage::Local: return VK_SHADER_STAGE_VERTEX_BIT;
        case ShaderRecompiler::ShaderStage::TessellationControl: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        case ShaderRecompiler::ShaderStage::TessellationEvaluation: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        case ShaderRecompiler::ShaderStage::Mesh: return VK_SHADER_STAGE_MESH_BIT_EXT;
        case ShaderRecompiler::ShaderStage::Fragment: return VK_SHADER_STAGE_FRAGMENT_BIT;
        case ShaderRecompiler::ShaderStage::Compute: return VK_SHADER_STAGE_COMPUTE_BIT;
        default: throw std::runtime_error("AGC graphics: unsupported compiled shader stage");
    }
}

inline VkPipelineStageFlags PipelineStages(std::span<const CompiledShader> shaders) {
    VkPipelineStageFlags result = 0;
    for (const auto& shader : shaders) {
        switch (VulkanStage(shader.stage)) {
            case VK_SHADER_STAGE_VERTEX_BIT: result |= VK_PIPELINE_STAGE_VERTEX_SHADER_BIT; break;
            case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: result |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT; break;
            case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: result |= VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT; break;
            case VK_SHADER_STAGE_MESH_BIT_EXT: result |= VK_PIPELINE_STAGE_MESH_SHADER_BIT_EXT; break;
            case VK_SHADER_STAGE_FRAGMENT_BIT: result |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; break;
            default: throw std::runtime_error("AGC graphics: invalid graphics pipeline stage");
        }
    }
    return result;
}

// Whether a vertex-pipeline stage writes gl_Layer (decorated BuiltIn Layer): it then reads the
// framebuffer's last layer from push dword LayerLimitPushDword to clamp its layer as the hardware does.
inline bool WritesLayer(const ShaderRecompiler::RecompileResult& program) {
    const auto& words = program.spirv;
    for (std::size_t cursor = 5; cursor + 3 < words.size();) {
        const auto count = words[cursor] >> 16u;
        if (count == 0) break;
        if ((words[cursor] & 0xffffu) == 71u && count >= 4 && words[cursor + 2] == 11u && words[cursor + 3] == 9u) return true;
        cursor += count;
    }
    return false;
}

inline bool LayerLimited(const CompiledShader& shader) {
    return shader.stage != ShaderRecompiler::ShaderStage::Mesh && shader.stage != ShaderRecompiler::ShaderStage::Fragment && shader.stage != ShaderRecompiler::ShaderStage::Compute && shader.program != nullptr && WritesLayer(*shader.program);
}

inline VkShaderStageFlags PushConstantStages(std::span<const CompiledShader> shaders) {
    VkShaderStageFlags result = 0;
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        if (!shader.program->pushConstants.empty() || shader.stage == ShaderRecompiler::ShaderStage::Mesh || LayerLimited(shader)) result |= VulkanStage(shader.stage);
    }
    return result;
}

inline std::array<std::byte, PipelinePushConstantBytes> AssemblePushConstants(std::span<const CompiledShader> shaders) {
    std::array<std::byte, PipelinePushConstantBytes> result{};
    std::array<bool, PipelinePushConstantBytes> occupied{};
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const auto& bytes = shader.program->pushConstants;
        if (bytes.empty()) continue;
        Require(bytes.size() % 4 == 0 && shader.pushConstantOffset % 4 == 0, "shader push constant range is not DWORD aligned");
        Require(shader.pushConstantOffset < PipelinePushConstantBytes && bytes.size() <= PipelinePushConstantBytes - shader.pushConstantOffset, "shader push constant range lies outside the pipeline push constant block");
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            const auto position = shader.pushConstantOffset + i;
            Require(!occupied[position], "shader push constant ranges of different stages overlap");
            occupied[position] = true;
            result[position] = bytes[i];
        }
    }
    return result;
}

}

#endif
