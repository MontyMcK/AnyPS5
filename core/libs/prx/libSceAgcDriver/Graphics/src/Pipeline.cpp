#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineSpecialization.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include <cstring>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <list>
#include <mutex>
#include <pthread.h>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace AgcDriver::Graphics {

void LogPipelineStatistics_nid_no_patch(const Context& context, VkPipeline pipeline, std::string_view header) {
    if (!context.pipelineExecutableInfo) return;
    const auto getProperties = context.Function<PFN_vkGetPipelineExecutablePropertiesKHR>("vkGetPipelineExecutablePropertiesKHR");
    const auto getStatistics = context.Function<PFN_vkGetPipelineExecutableStatisticsKHR>("vkGetPipelineExecutableStatisticsKHR");
    VkPipelineInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
    pipelineInfo.pipeline = pipeline;
    std::uint32_t count = 0;
    Check(getProperties(context.device, &pipelineInfo, &count, nullptr), "vkGetPipelineExecutablePropertiesKHR count");
    std::vector<VkPipelineExecutablePropertiesKHR> properties(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR});
    Check(getProperties(context.device, &pipelineInfo, &count, properties.data()), "vkGetPipelineExecutablePropertiesKHR");
    properties.resize(count);
    // Pipelines compile on several threads, so each pipeline's report is one line written at once.
    std::string line = "[pipeline-stats]";
    if (!header.empty()) {
        line += ' ';
        line += header;
    }
    char item[160];
    for (std::uint32_t index = 0; index < properties.size(); ++index) {
        VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        executable.pipeline = pipeline;
        executable.executableIndex = index;
        count = 0;
        Check(getStatistics(context.device, &executable, &count, nullptr), "vkGetPipelineExecutableStatisticsKHR count");
        std::vector<VkPipelineExecutableStatisticKHR> statistics(count, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        Check(getStatistics(context.device, &executable, &count, statistics.data()), "vkGetPipelineExecutableStatisticsKHR");
        statistics.resize(count);
        const auto& property = properties[index];
        std::snprintf(item, sizeof(item), " | %s stages=0x%x subgroup=%u", property.name, property.stages, property.subgroupSize);
        line += item;
        for (const auto& statistic : statistics) {
            std::string name = statistic.name;
            for (auto& c : name) {
                if (c == ' ') c = '_';
            }
            switch (statistic.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                std::snprintf(item, sizeof(item), " %s=%s", name.c_str(), statistic.value.b32 ? "true" : "false");
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                std::snprintf(item, sizeof(item), " %s=%lld", name.c_str(), static_cast<long long>(statistic.value.i64));
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                std::snprintf(item, sizeof(item), " %s=%llu", name.c_str(), static_cast<unsigned long long>(statistic.value.u64));
                break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                std::snprintf(item, sizeof(item), " %s=%.17g", name.c_str(), statistic.value.f64);
                break;
            default:
                throw std::runtime_error("unsupported pipeline statistic format");
            }
            line += item;
        }
    }
    line += '\n';
    std::fwrite(line.data(), 1, line.size(), stderr);
}

Framebuffer::Framebuffer(const Context& context, VkRenderPass renderPass, std::span<const VkImageView> targets, VkExtent2D extent, std::uint32_t layers) : context(context) {
    // Cached objects outlive their device's teardown; they must not keep its buffer pool alive past it.
    this->context.bufferPool.reset();
    Require(extent.width != 0 && extent.height != 0 && extent.width <= context.limits.maxFramebufferWidth && extent.height <= context.limits.maxFramebufferHeight, "framebuffer extent exceeds device limits");
    VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebufferInfo.renderPass = renderPass;
    framebufferInfo.attachmentCount = static_cast<std::uint32_t>(targets.size());
    framebufferInfo.pAttachments = targets.empty() ? nullptr : targets.data();
    framebufferInfo.width = extent.width;
    framebufferInfo.height = extent.height;
    Require(layers != 0 && layers <= context.limits.maxFramebufferLayers, "framebuffer layers exceed device limits");
    framebufferInfo.layers = layers;
    Check(context.Function<PFN_vkCreateFramebuffer>("vkCreateFramebuffer")(context.device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer");
}

Framebuffer::~Framebuffer() {
    if (framebuffer) context.Function<PFN_vkDestroyFramebuffer>("vkDestroyFramebuffer")(context.device, framebuffer, nullptr);
}

void ValidateDepthBounds(const Context& context, const State& state) {
    Require(!state.depthBoundsTest || context.depthRangeUnrestricted || (state.minDepthBounds >= 0.0f && state.minDepthBounds <= 1.0f && state.maxDepthBounds >= 0.0f && state.maxDepthBounds <= 1.0f), "depth bounds outside [0, 1] require VK_EXT_depth_range_unrestricted");
}

void ValidateViewport(const Context& context, const VkViewport& viewport) {
    Require(std::isfinite(viewport.minDepth) && std::isfinite(viewport.maxDepth), "non-finite viewport depth range");
    Require(context.depthRangeUnrestricted || (viewport.minDepth >= 0 && viewport.minDepth <= 1 && viewport.maxDepth >= 0 && viewport.maxDepth <= 1), "viewport depth outside [0, 1] requires VK_EXT_depth_range_unrestricted");
    Require(std::isfinite(viewport.x) && std::isfinite(viewport.y) && std::isfinite(viewport.width) && std::isfinite(viewport.height), "viewport arithmetic overflow");
    Require(viewport.width <= context.limits.maxViewportDimensions[0] && std::abs(viewport.height) <= context.limits.maxViewportDimensions[1], "viewport dimensions exceed device limits");
    Require(viewport.x >= context.limits.viewportBoundsRange[0] && viewport.x + viewport.width <= context.limits.viewportBoundsRange[1], "viewport X exceeds device bounds");
    Require(std::min(viewport.y, viewport.y + viewport.height) >= context.limits.viewportBoundsRange[0] && std::max(viewport.y, viewport.y + viewport.height) <= context.limits.viewportBoundsRange[1], "viewport Y exceeds device bounds");
}

Pipeline::Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) : Pipeline(context, state, vertexInput, resources.Layout(), shaders, attachmentLayout) {}

Pipeline::Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, VkDescriptorSetLayout setLayout, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) : context(context), _modules(shaders.size()), attachments(state.colors.size() + (state.depth ? 1u : 0u)), colorAttachments(state.colors.size()), depthBounds(state.depth.has_value() && state.depthBoundsTest), depthBias(state.depth.has_value() && state.depthBias) {
    PerformanceTimer timing("Vulkan.GraphicsPipeline");
    // A cached pipeline may outlive its device's teardown (see ClearCachedPipelines); it must not keep
    // the buffer pool, which is reset with the device, alive past it.
    this->context.bufferPool.reset();
    Require(state.blends.size() == (state.colors.empty() ? 0u : state.colors.back().exportIndex + 1u) && state.colors.size() <= state.blends.size(), "blend states do not match decoded color state");
    Require(state.blends.size() <= context.limits.maxColorAttachments, "color targets exceed device attachment limits");
    Require(state.reversedBlend == 0 || context.dualSrcBlend, "reversed blending requires dual-source blending support");
    Require(state.reversedBlend == 0 || state.blends.size() <= context.limits.maxFragmentDualSrcAttachments, "reversed blending exceeds dual-source attachment limits");
    Require(state.hasColorTarget || (context.limits.framebufferNoAttachmentsSampleCounts & VK_SAMPLE_COUNT_1_BIT) != 0, "device does not support single-sample rendering without attachments");
    Require(!depthBounds || context.depthBounds, "device does not support the depth bounds test");
    Require(!depthBias || state.depthBiasClamp == 0.0f || context.depthBiasClamp, "device does not support depth bias clamping");
    Require(!state.negativeOneToOne || context.depthClipControl, "negative-one-to-one depth clipping requires VK_EXT_depth_clip_control with depthClipControl enabled");
    Require(state.conservativeRasterization == VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT || context.conservativeRasterization, "conservative rasterization requires VK_EXT_conservative_rasterization with at most 1/256 pixel of overestimation and degenerate triangles rasterized");
    if (state.rectList) Require(context.tessellationShader && context.limits.maxTessellationPatchSize >= 4, "rect-list requires tessellation with four output control points");
    if (state.stages.tessellation) {
        Require(context.tessellationShader, "device does not support tessellation shaders");
        Require(state.stages.tessellation->inputControlPoints <= context.limits.maxTessellationPatchSize && state.stages.tessellation->outputControlPoints <= context.limits.maxTessellationPatchSize, "tessellation patch exceeds device limits");
    }
    if (state.stages.mesh) {
        Require(context.meshShader, "device does not support VK_EXT_mesh_shader");
        const auto& mesh = *state.stages.mesh;
        const auto invocations = state.stages.vertexWaveSize == 64u && context.subgroup.subgroupSize == 32u ? mesh.threadsPerGroup / 2u : mesh.threadsPerGroup;
        Require(invocations <= context.meshLimits.maxMeshWorkGroupInvocations && invocations <= context.meshLimits.maxMeshWorkGroupSize[0], "mesh workgroup exceeds device limits");
        Require(mesh.maxVertices <= context.meshLimits.maxMeshOutputVertices && mesh.maxPrimitives <= context.meshLimits.maxMeshOutputPrimitives && static_cast<std::uint64_t>(mesh.ldsSizeDwords) * 4 <= context.meshLimits.maxMeshSharedMemorySize, "mesh output or LDS exceeds device limits");
    }
    const auto pushStages = PushConstantStages(shaders);
    Require(pushStages == 0 || context.limits.maxPushConstantsSize >= PipelinePushConstantBytes, "graphics push constant range exceeds device limit");
    try {
        std::vector<VkPipelineShaderStageCreateInfo> stages(shaders.size());
        std::vector<PipelineSpecialization> specializations;
        specializations.reserve(shaders.size());
        for (std::uint32_t i = 0; i < shaders.size(); ++i) {
            const auto& shader = *shaders[i].program;
            VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            module.codeSize = shader.spirv.size() * sizeof(std::uint32_t);
            module.pCode = shader.spirv.data();
            // APS5_SPV_OVERRIDE=<dir>: a module whose SPIR-V bytes hash (FNV-1a 64) to H is replaced by
            // <dir>/<H as 16 hex digits>.spv when that file exists, for trying offline-transformed shaders.
            std::vector<std::uint32_t> overrideCode;
            if (static const char* overrideDir = std::getenv("APS5_SPV_OVERRIDE"); overrideDir != nullptr && *overrideDir != '\0') {
                std::uint64_t hash = 14695981039346656037ull;
                const auto* bytes = reinterpret_cast<const unsigned char*>(shader.spirv.data());
                for (std::size_t b = 0; b < module.codeSize; ++b) hash = (hash ^ bytes[b]) * 1099511628211ull;
                char path[512];
                std::snprintf(path, sizeof(path), "%s/%016llx.spv", overrideDir, static_cast<unsigned long long>(hash));
                if (auto* file = std::fopen(path, "rb")) {
                    std::fseek(file, 0, SEEK_END);
                    const auto size = std::ftell(file);
                    std::fseek(file, 0, SEEK_SET);
                    overrideCode.resize(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
                    const auto read = std::fread(overrideCode.data(), sizeof(std::uint32_t), overrideCode.size(), file);
                    std::fclose(file);
                    if (read == overrideCode.size() && !overrideCode.empty()) {
                        module.codeSize = overrideCode.size() * sizeof(std::uint32_t);
                        module.pCode = overrideCode.data();
                        std::fprintf(stderr, "[spv-override] variant 0x%llx replaced by %s (%zu -> %zu bytes)\n", static_cast<unsigned long long>(shader.PipelineVariantId()), path, shader.spirv.size() * sizeof(std::uint32_t), static_cast<std::size_t>(module.codeSize));
                    }
                }
            }
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &module, nullptr, &_modules[i]), "vkCreateShaderModule graphics");
            const auto stage = VulkanStage(shaders[i].stage);
            stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].stage = stage;
            stages[i].module = _modules[i];
            stages[i].pName = "main";
            specializations.emplace_back(shader);
            stages[i].pSpecializationInfo = specializations.back().Info();
        }
        // A descriptor set layout with the same bindings as this one is compatible with the pipeline
        // layout, so later draws bind their own ShaderResources' set under it.
        const VkPushConstantRange push{pushStages, 0, PipelinePushConstantBytes};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &setLayout;
        layoutInfo.pushConstantRangeCount = pushStages != 0 ? 1 : 0;
        layoutInfo.pPushConstantRanges = pushStages != 0 ? &push : nullptr;
        Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout graphics");
        std::vector<VkAttachmentDescription> colors;
        std::vector<VkAttachmentReference> references(state.blends.size(), VkAttachmentReference{VK_ATTACHMENT_UNUSED, attachmentLayout});
        for (std::uint32_t index = 0; index < state.colors.size(); ++index) {
            VkAttachmentDescription color{};
            color.format = state.colors[index].format;
            color.samples = VK_SAMPLE_COUNT_1_BIT;
            color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.initialLayout = attachmentLayout;
            color.finalLayout = attachmentLayout;
            colors.push_back(color);
            references.at(state.colors[index].exportIndex) = {index, attachmentLayout};
        }
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = static_cast<std::uint32_t>(references.size());
        subpass.pColorAttachments = references.empty() ? nullptr : references.data();
        const VkAttachmentReference depthReference{static_cast<std::uint32_t>(colors.size()), VK_IMAGE_LAYOUT_GENERAL};
        if (state.depth) {
            VkAttachmentDescription depth{};
            depth.format = state.depth->format;
            depth.samples = VK_SAMPLE_COUNT_1_BIT;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            depth.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
            depth.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
            colors.push_back(depth);
            subpass.pDepthStencilAttachment = &depthReference;
        }
        VkRenderPassCreateInfo passInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        passInfo.attachmentCount = static_cast<std::uint32_t>(colors.size());
        passInfo.pAttachments = colors.empty() ? nullptr : colors.data();
        passInfo.subpassCount = 1;
        passInfo.pSubpasses = &subpass;
        Check(context.Function<PFN_vkCreateRenderPass>("vkCreateRenderPass")(context.device, &passInfo, nullptr, &renderPass), "vkCreateRenderPass");
        VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        input.vertexBindingDescriptionCount = static_cast<std::uint32_t>(vertexInput.bindings.size());
        input.pVertexBindingDescriptions = vertexInput.bindings.data();
        input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(vertexInput.attributes.size());
        input.pVertexAttributeDescriptions = vertexInput.attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = state.topology;
        const bool listTopology = state.topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || state.topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        assembly.primitiveRestartEnable = state.primitiveRestart && (!listTopology || context.primitiveListRestart) ? VK_TRUE : VK_FALSE;
        // Viewport and scissor are set per draw (Begin), so they do not multiply pipelines; the depth
        // clip control stays baked in.
        VkPipelineViewportStateCreateInfo viewports{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        VkPipelineViewportDepthClipControlCreateInfoEXT depthClip{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
        depthClip.negativeOneToOne = state.negativeOneToOne;
        if (state.negativeOneToOne) viewports.pNext = &depthClip;
        viewports.viewportCount = 1;
        viewports.scissorCount = 1;
        std::vector<VkDynamicState> dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        if (depthBounds) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        if (depthBias) dynamicStates.push_back(VK_DYNAMIC_STATE_DEPTH_BIAS);
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size());
        dynamic.pDynamicStates = dynamicStates.data();
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.depthClampEnable = state.depthClamp && context.depthClamp ? VK_TRUE : VK_FALSE;
        raster.cullMode = state.cullMode;
        raster.frontFace = state.frontFace;
        raster.depthBiasEnable = depthBias;
        raster.lineWidth = 1;
        VkPipelineRasterizationConservativeStateCreateInfoEXT conservative{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
        conservative.conservativeRasterizationMode = state.conservativeRasterization;
        if (state.conservativeRasterization != VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT) raster.pNext = &conservative;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        samples.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = state.depthTest;
        depthStencil.depthWriteEnable = state.depthWrite;
        depthStencil.depthCompareOp = state.depthCompare;
        depthStencil.depthBoundsTestEnable = depthBounds;
        depthStencil.stencilTestEnable = state.stencilTest;
        depthStencil.front = state.stencilFront;
        depthStencil.back = state.stencilBack;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = static_cast<std::uint32_t>(state.blends.size());
        blend.pAttachments = state.blends.empty() ? nullptr : state.blends.data();
        std::copy(state.blendConstants.begin(), state.blendConstants.end(), blend.blendConstants);
        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.flags = context.pipelineExecutableInfo ? VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR : 0;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        VkPipelineTessellationStateCreateInfo tessellation{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        if (state.rectList || state.stages.tessellation) {
            tessellation.patchControlPoints = state.rectList ? 3u : state.stages.tessellation->inputControlPoints;
            pipelineInfo.pTessellationState = &tessellation;
        }
        pipelineInfo.pVertexInputState = state.stages.mesh ? nullptr : &input;
        pipelineInfo.pInputAssemblyState = state.stages.mesh ? nullptr : &assembly;
        pipelineInfo.pViewportState = &viewports;
        pipelineInfo.pRasterizationState = &raster;
        pipelineInfo.pMultisampleState = &samples;
        pipelineInfo.pDepthStencilState = state.depth ? &depthStencil : nullptr;
        pipelineInfo.pColorBlendState = &blend;
        pipelineInfo.pDynamicState = &dynamic;
        pipelineInfo.layout = layout;
        pipelineInfo.renderPass = renderPass;
        timing.Mark("modules_and_state");
        const auto createStarted = std::chrono::steady_clock::now();
        Check(context.Function<PFN_vkCreateGraphicsPipelines>("vkCreateGraphicsPipelines")(context.device, context.pipelineCache, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateGraphicsPipelines");
        const auto createSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - createStarted).count();
        if (createSeconds > 0.25) std::fprintf(stderr, "[draw] graphics pipeline took %.2f s\n", createSeconds);
        timing.Mark("create");
        if (context.pipelineExecutableInfo) {
            // Names the pipeline's stages (their variant ids) in its statistics line.
            std::string header = "graphics";
            char item[48];
            for (const auto& shader : shaders) {
                std::snprintf(item, sizeof(item), " stage%u=0x%llx", static_cast<unsigned>(shader.stage), static_cast<unsigned long long>(shader.program != nullptr ? shader.program->PipelineVariantId() : 0));
                header += item;
            }
            std::snprintf(item, sizeof(item), " extent %ux%u", state.renderExtent.width, state.renderExtent.height);
            header += item;
            LogPipelineStatistics_nid_no_patch(context, pipeline, header);
        }
    } catch (...) {
        release();
        throw;
    }
}

Pipeline::~Pipeline() {
    release();
}

void Pipeline::release() noexcept {
    framebuffers.clear();
    if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
    if (renderPass) context.Function<PFN_vkDestroyRenderPass>("vkDestroyRenderPass")(context.device, renderPass, nullptr);
    if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
    for (auto module : _modules) {
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
    }
    pipeline = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    _modules.clear();
}

void Pipeline::Abandon() noexcept {
    for (auto& entry : framebuffers) entry.framebuffer->Abandon();
    framebuffers.clear();
    pipeline = VK_NULL_HANDLE;
    renderPass = VK_NULL_HANDLE;
    layout = VK_NULL_HANDLE;
    _modules.clear();
}

void Framebuffer::Abandon() noexcept {
    framebuffer = VK_NULL_HANDLE;
}

VkPipelineLayout Pipeline::Layout() const {
    return layout;
}

std::shared_ptr<Framebuffer> Pipeline::AcquireFramebuffer(std::span<const VkImageView> targets, std::span<const std::shared_ptr<StorageTexture>> owners, VkExtent2D extent, std::uint32_t layers) {
    Require(targets.size() == attachments && owners.size() == colorAttachments, "render targets do not match the pipeline's attachments");
    const bool resident = std::all_of(owners.begin(), owners.end(), [](const auto& owner) { return owner != nullptr; });
    if (!resident && layers > 1) {
        static std::atomic<int> told{0};
        if (told.fetch_add(1) < 8) std::fprintf(stderr, "[framebuffer] layered framebuffer (%u layers, %ux%u) over a non-resident single-layer render target\n", layers, extent.width, extent.height);
    }
    if (resident) {
        const auto matches = [&](const CachedFramebuffer& entry) {
            if (entry.extent.width != extent.width || entry.extent.height != extent.height || entry.layers != layers || !std::equal(entry.views.begin(), entry.views.end(), targets.begin(), targets.end())) return false;
            for (std::size_t i = 0; i < owners.size(); ++i) {
                if (entry.owners[i].lock().get() != owners[i].get()) return false;
            }
            return true;
        };
        if (!framebuffers.empty() && matches(framebuffers.back())) return framebuffers.back().framebuffer;
        // Entries whose views are gone can never match again and go as soon as no recorded draw holds
        // them (Kept keeps its framebuffer until the batch completes).
        std::erase_if(framebuffers, [](const CachedFramebuffer& entry) {
            return entry.framebuffer.use_count() == 1 && std::any_of(entry.owners.begin(), entry.owners.end(), [](const auto& owner) { return owner.expired(); });
        });
        for (auto it = framebuffers.begin(); it != framebuffers.end(); ++it) {
            if (!matches(*it)) continue;
            std::rotate(it, std::next(it), framebuffers.end());
            return framebuffers.back().framebuffer;
        }
    }
    auto framebuffer = std::make_shared<Framebuffer>(context, renderPass, targets, extent, layers);
    if (!resident) return framebuffer;
    // Beyond the bound the least recently used unreferenced entry goes.
    constexpr std::size_t bound = 8;
    while (framebuffers.size() >= bound) {
        const auto victim = std::find_if(framebuffers.begin(), framebuffers.end(), [](const CachedFramebuffer& entry) { return entry.framebuffer.use_count() == 1; });
        if (victim == framebuffers.end()) break;
        framebuffers.erase(victim);
    }
    CachedFramebuffer entry;
    entry.views.assign(targets.begin(), targets.end());
    entry.owners.assign(owners.begin(), owners.end());
    entry.extent = extent;
    entry.layers = layers;
    entry.framebuffer = framebuffer;
    framebuffers.push_back(std::move(entry));
    return framebuffer;
}

namespace {

// The graphics pipeline, viewport and scissor last recorded into a command buffer inside its open
// render pass. Continue is the only graphics bind, and every pipeline declares viewport and scissor
// dynamic, so they persist across pipeline binds; Begin forgets them (a recycled command buffer
// always opens a pass first). APS5_NO_BIND_FILTER=1 records every bind.
//
// With live `bind_elide` the record also tracks, for the same command buffer and pass, the
// descriptor set bound as set 0 with its pipeline layout, the push constant block with its layout
// and stages, the vertex buffers, the index buffer, and the depth bias and depth bounds values; a
// bind equal to the tracked one is skipped. Each is exact Vulkan state persistence: a set and push
// constants bound with layout L stay valid for later pipelines of layout L (the same handle is
// compatible with itself), vertex and index buffer bindings are untouched by pipeline binds, and
// dynamic depth bias and bounds persist while every pipeline bound since declares them dynamic (a
// pipeline bind without them forgets the tracked value). Begin forgets everything (a new pass, or a
// recycled command buffer), as does a Continue into another command buffer. Every graphics bind a
// recorded draw makes inside its pass goes through here (BindSet, PushConstants, BindDrawBuffers);
// a draw that binds outside these (the synchronous path, after its own Begin) never continues a
// pass, and its Begin reset the tracking first.
enum BindKind : std::size_t { BindSetKind, BindPushKind, BindVertexKind, BindIndexKind, BindBiasKind, BindBoundsKind, BindKindCount };

struct BoundGraphicsState {
    std::mutex mutex;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkViewport viewport{};
    VkRect2D scissor{};
    bool viewportSet = false;
    // bind_elide's tracking (valid only while `commands` is the buffer recorded into).
    bool setValid = false;
    VkPipelineLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    bool pushValid = false;
    VkPipelineLayout pushLayout = VK_NULL_HANDLE;
    VkShaderStageFlags pushStages = 0;
    std::array<std::byte, PipelinePushConstantBytes> push{};
    bool vertexValid = false;
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceSize> vertexOffsets;
    bool indexValid = false;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    bool biasValid = false;
    std::array<float, 3> bias{};
    bool boundsValid = false;
    std::array<float, 2> bounds{};
    std::array<std::uint64_t, BindKindCount> recorded{};
    std::array<std::uint64_t, BindKindCount> skipped{};
    void forgetTracked() {
        setValid = pushValid = vertexValid = indexValid = biasValid = boundsValid = false;
    }
};

BoundGraphicsState& boundGraphicsState() {
    static BoundGraphicsState state;
    return state;
}

bool bindFilter() {
    static const bool enabled = std::getenv("APS5_NO_BIND_FILTER") == nullptr;
    return enabled;
}

bool bindElide() {
    static const bool requested = std::getenv("APS5_BIND_ELIDE") != nullptr;
    return bindFilter() && LiveFlag("bind_elide", requested);
}

// A bind recorded while bind_elide was off went untracked (no lock is taken then): the tracked
// state is forgotten at the next tracked bind, so a switch turned back on mid-pass never skips a
// bind against what an untracked one replaced.
std::atomic<bool> untrackedBind{false};

bool trackBinds(BoundGraphicsState& bound) {
    if (untrackedBind.exchange(false, std::memory_order_relaxed)) bound.forgetTracked();
    return true;
}

bool sameViewport(const VkViewport& a, const VkViewport& b) {
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height && a.minDepth == b.minDepth && a.maxDepth == b.maxDepth;
}

bool sameScissor(const VkRect2D& a, const VkRect2D& b) {
    return a.offset.x == b.offset.x && a.offset.y == b.offset.y && a.extent.width == b.extent.width && a.extent.height == b.extent.height;
}

}

void Pipeline::Begin(VkCommandBuffer commands, const Framebuffer& framebuffer, VkExtent2D extent, const State& state) const {
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = renderPass;
    begin.framebuffer = framebuffer.Handle();
    begin.renderArea = {{0, 0}, extent};
    context.Resolved(&DeviceFunctions::cmdBeginRenderPass, "vkCmdBeginRenderPass")(commands, &begin, VK_SUBPASS_CONTENTS_INLINE);
    if (bindFilter()) {
        auto& bound = boundGraphicsState();
        std::lock_guard lock(bound.mutex);
        bound.commands = VK_NULL_HANDLE;
        bound.forgetTracked();
    }
    Continue(commands, state);
}

void Pipeline::Continue(VkCommandBuffer commands, const State& state) const {
    bool bindPipeline = true, setViewport = true, setScissor = true;
    bool setBias = depthBias, setBounds = depthBounds;
    if (bindFilter()) {
        auto& bound = boundGraphicsState();
        std::lock_guard lock(bound.mutex);
        if (bound.commands == commands) {
            bindPipeline = bound.pipeline != pipeline;
            setViewport = !bound.viewportSet || !sameViewport(bound.viewport, state.viewport);
            setScissor = !bound.viewportSet || !sameScissor(bound.scissor, state.scissor);
        } else {
            bound.forgetTracked();
        }
        bound.commands = commands;
        bound.pipeline = pipeline;
        bound.viewport = state.viewport;
        bound.scissor = state.scissor;
        bound.viewportSet = true;
        // A pipeline bound without dynamic depth bias (bounds) applies its static value: the
        // tracked dynamic value no longer stands.
        if (bindPipeline && !depthBias) bound.biasValid = false;
        if (bindPipeline && !depthBounds) bound.boundsValid = false;
        const bool elide = bindElide();
        if (depthBias) {
            const std::array<float, 3> bias{state.depthBiasConstant, state.depthBiasClamp, state.depthBiasSlope};
            setBias = !(elide && bound.biasValid && std::memcmp(bound.bias.data(), bias.data(), sizeof(bias)) == 0);
            bound.bias = bias;
            bound.biasValid = elide;
            ++(setBias ? bound.recorded : bound.skipped)[BindBiasKind];
        }
        if (depthBounds) {
            const std::array<float, 2> bounds{state.minDepthBounds, state.maxDepthBounds};
            setBounds = !(elide && bound.boundsValid && std::memcmp(bound.bounds.data(), bounds.data(), sizeof(bounds)) == 0);
            bound.bounds = bounds;
            bound.boundsValid = elide;
            ++(setBounds ? bound.recorded : bound.skipped)[BindBoundsKind];
        }
    }
    if (bindPipeline) context.Resolved(&DeviceFunctions::cmdBindPipeline, "vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    if (setViewport) context.Resolved(&DeviceFunctions::cmdSetViewport, "vkCmdSetViewport")(commands, 0, 1, &state.viewport);
    if (setScissor) context.Resolved(&DeviceFunctions::cmdSetScissor, "vkCmdSetScissor")(commands, 0, 1, &state.scissor);
    if (setBias) context.Resolved(&DeviceFunctions::cmdSetDepthBias, "vkCmdSetDepthBias")(commands, state.depthBiasConstant, state.depthBiasClamp, state.depthBiasSlope);
    if (!setBounds) return;
    context.Resolved(&DeviceFunctions::cmdSetDepthBounds, "vkCmdSetDepthBounds")(commands, state.minDepthBounds, state.maxDepthBounds);
}

void Pipeline::PushConstants(VkCommandBuffer commands, VkShaderStageFlags stages, std::span<const std::byte, PipelinePushConstantBytes> bytes) const {
    if (stages == 0) return;
    // Switch off: no tracking (and no lock) at all, as before.
    if (!bindElide()) {
        untrackedBind.store(true, std::memory_order_relaxed);
    } else {
        auto& bound = boundGraphicsState();
        std::lock_guard lock(bound.mutex);
        if (trackBinds(bound) && bound.commands == commands) {
            const bool elide = true;
            // A push of other stages updates only theirs: the tracked block then no longer
            // describes every stage's values, so it is replaced rather than merged.
            if (elide && bound.pushValid && bound.pushLayout == layout && bound.pushStages == stages && std::memcmp(bound.push.data(), bytes.data(), PipelinePushConstantBytes) == 0) {
                ++bound.skipped[BindPushKind];
                return;
            }
            std::memcpy(bound.push.data(), bytes.data(), PipelinePushConstantBytes);
            bound.pushLayout = layout;
            bound.pushStages = stages;
            bound.pushValid = elide;
            ++bound.recorded[BindPushKind];
        }
    }
    context.Resolved(&DeviceFunctions::cmdPushConstants, "vkCmdPushConstants")(commands, layout, stages, 0, PipelinePushConstantBytes, bytes.data());
}

void Pipeline::BindSet(VkCommandBuffer commands, VkDescriptorSet set) const {
    if (!bindElide()) {
        untrackedBind.store(true, std::memory_order_relaxed);
    } else {
        auto& bound = boundGraphicsState();
        std::lock_guard lock(bound.mutex);
        if (trackBinds(bound) && bound.commands == commands) {
            const bool elide = true;
            if (elide && bound.setValid && bound.setLayout == layout && bound.set == set) {
                ++bound.skipped[BindSetKind];
                return;
            }
            bound.setLayout = layout;
            bound.set = set;
            bound.setValid = elide;
            ++bound.recorded[BindSetKind];
        }
    }
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
}

void BindDrawBuffers(const Context& context, VkCommandBuffer commands, std::span<const VkBuffer> vertexBuffers, std::span<const VkDeviceSize> vertexOffsets, bool indexed, VkBuffer indexBuffer, VkDeviceSize indexOffset, VkIndexType indexType) {
    bool bindVertex = !vertexBuffers.empty(), bindIndex = indexed;
    if (!bindVertex && !bindIndex) {
    } else if (!bindElide()) {
        untrackedBind.store(true, std::memory_order_relaxed);
    } else {
        auto& bound = boundGraphicsState();
        std::lock_guard lock(bound.mutex);
        if (trackBinds(bound) && bound.commands == commands) {
            const bool elide = true;
            if (bindVertex) {
                bindVertex = !(elide && bound.vertexValid && std::equal(vertexBuffers.begin(), vertexBuffers.end(), bound.vertexBuffers.begin(), bound.vertexBuffers.end()) && std::equal(vertexOffsets.begin(), vertexOffsets.end(), bound.vertexOffsets.begin(), bound.vertexOffsets.end()));
                if (bindVertex) {
                    bound.vertexBuffers.assign(vertexBuffers.begin(), vertexBuffers.end());
                    bound.vertexOffsets.assign(vertexOffsets.begin(), vertexOffsets.end());
                }
                bound.vertexValid = elide;
                ++(bindVertex ? bound.recorded : bound.skipped)[BindVertexKind];
            }
            if (bindIndex) {
                bindIndex = !(elide && bound.indexValid && bound.indexBuffer == indexBuffer && bound.indexOffset == indexOffset && bound.indexType == indexType);
                bound.indexBuffer = indexBuffer;
                bound.indexOffset = indexOffset;
                bound.indexType = indexType;
                bound.indexValid = elide;
                ++(bindIndex ? bound.recorded : bound.skipped)[BindIndexKind];
            }
        }
    }
    if (bindVertex) context.Resolved(&DeviceFunctions::cmdBindVertexBuffers, "vkCmdBindVertexBuffers")(commands, 0, static_cast<std::uint32_t>(vertexBuffers.size()), vertexBuffers.data(), vertexOffsets.data());
    if (bindIndex) context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, indexBuffer, indexOffset, indexType);
}

bool BindElision() {
    return bindElide();
}

BindCounts TakeBindCounts() {
    auto& bound = boundGraphicsState();
    std::lock_guard lock(bound.mutex);
    BindCounts counts;
    counts.recorded = std::exchange(bound.recorded, {});
    counts.skipped = std::exchange(bound.skipped, {});
    return counts;
}

namespace {

template<typename TValue>
void append(std::vector<std::byte>& key, const TValue& value) {
    static_assert(std::is_trivially_copyable_v<TValue>);
    const auto bytes = std::as_bytes(std::span(&value, 1));
    key.insert(key.end(), bytes.begin(), bytes.end());
}

// Everything the Pipeline objects are built from, or empty when a stage's result has no variant id
// (the recompiler could not identify it, so nothing else may share its pipeline). The rect-list
// control and evaluation stages are generated from the vertex and fragment results, which the key
// already names, so they carry no id of their own.
void pipelineKeyInto(std::vector<std::byte>& key, const Context& context, const State& state, const VertexInputLayout& input, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    using Stage = ShaderRecompiler::ShaderStage;
    key.clear();
    append(key, context.device);
    append(key, attachmentLayout);
    append(key, shaders.size());
    for (const auto& shader : shaders) {
        Require(shader.program != nullptr, "missing compiled shader");
        const bool generated = state.rectList && (shader.stage == Stage::TessellationControl || shader.stage == Stage::TessellationEvaluation);
        if (!generated && shader.program->PipelineVariantId() == 0) {
            key.clear();
            return;
        }
        append(key, shader.stage);
        append(key, generated ? std::uint64_t{0} : shader.program->PipelineVariantId());
        // Where the stage's push constants sit in the block (AssemblePushConstants).
        append(key, shader.pushConstantOffset);
    }
    append(key, PushConstantStages(shaders));
    append(key, input.bindings.size());
    for (const auto& binding : input.bindings) {
        append(key, binding.binding);
        append(key, binding.stride);
        append(key, binding.inputRate);
    }
    append(key, input.attributes.size());
    for (const auto& attribute : input.attributes) {
        append(key, attribute.location);
        append(key, attribute.binding);
        append(key, attribute.format);
        append(key, attribute.offset);
    }
    append(key, resources.LayoutKey().size());
    for (const auto word : resources.LayoutKey()) append(key, word);
    append(key, state.hasColorTarget);
    append(key, state.rectList);
    append(key, state.topology);
    append(key, state.primitiveRestart);
    append(key, state.cullMode);
    append(key, state.frontFace);
    append(key, state.negativeOneToOne);
    append(key, state.depthClamp && context.depthClamp);
    append(key, state.conservativeRasterization);
    append(key, state.blends.size());
    for (const auto& blend : state.blends) append(key, blend);
    for (const auto value : state.blendConstants) append(key, value);
    append(key, state.colors.size());
    for (const auto& color : state.colors) append(key, color.format);
    if (state.blends.size() != state.colors.size()) {
        for (const auto& color : state.colors) append(key, color.exportIndex);
    }
    append(key, state.depth.has_value());
    if (state.depth) {
        append(key, state.depth->format);
        append(key, state.depthTest);
        append(key, state.depthWrite);
        append(key, state.depthCompare);
        append(key, state.depthBoundsTest);
        append(key, state.depthBias);
        append(key, state.stencilTest);
        append(key, state.stencilFront);
        append(key, state.stencilBack);
    }
    append(key, state.stages.mesh.has_value());
    if (state.stages.mesh) {
        const auto& mesh = *state.stages.mesh;
        append(key, mesh.inputPrimitive);
        append(key, mesh.primitivesPerGroup);
        append(key, mesh.verticesPerGroup);
        append(key, mesh.maxVertices);
        append(key, mesh.maxPrimitives);
        append(key, mesh.threadsPerGroup);
        append(key, mesh.ldsSizeDwords);
        append(key, mesh.provokingVertex);
    }
    append(key, state.stages.tessellation.has_value());
    if (state.stages.tessellation) {
        const auto& tessellation = *state.stages.tessellation;
        append(key, tessellation.inputControlPoints);
        append(key, tessellation.outputControlPoints);
        append(key, tessellation.domain);
        append(key, tessellation.partitioning);
        append(key, tessellation.outputTopology);
    }
}

std::vector<std::byte> pipelineKey(const Context& context, const State& state, const VertexInputLayout& input, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    std::vector<std::byte> key;
    pipelineKeyInto(key, context, state, input, resources, shaders, attachmentLayout);
    return key;
}

std::uint64_t hashKey(const std::vector<std::byte>& key) {
    return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(key.data()), key.size()));
}

struct PipelineStore {
    struct Entry {
        VkDevice device;
        // The device's buffer pool at insertion: it is made and reset with the device, so it tells
        // the device instance apart from a later one the loader gave the same handle value.
        std::weak_ptr<BufferPool> pool;
        std::uint64_t hash;
        std::vector<std::byte> key;
        std::shared_ptr<Pipeline> pipeline;
    };
    std::mutex mutex;
    std::vector<std::byte> scratchKey;
    // Least recently used first.
    std::list<Entry> entries;
    std::unordered_map<std::uint64_t, std::list<Entry>::iterator> index;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t uncached = 0;
    std::uint64_t evicted = 0;
    std::uint64_t generation = 1;
    std::uint64_t identityHits = 0;
    std::unordered_map<std::uint64_t, std::vector<std::byte>> pending;
    std::unordered_set<std::uint64_t> failed;
    std::size_t inflight = 0;
    std::condition_variable idle;
    std::uint64_t asyncStarted = 0;
    std::uint64_t asyncSkips = 0;
    std::chrono::steady_clock::time_point waitWindow{};
    std::chrono::steady_clock::duration waitedInWindow{};
    double asyncSeconds = 0.0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

bool pipelineIdentity() {
    static const bool enabled = [] { const char* text = std::getenv("APS5_PIPELINE_IDENTITY"); return text == nullptr || std::strcmp(text, "0") != 0; }();
    return enabled;
}

struct LastPipeline {
    std::vector<std::byte> key;
    std::vector<std::byte> scratch;
    std::list<PipelineStore::Entry>::iterator entry;
    std::uint64_t generation = 0;
};

LastPipeline& ThreadLastPipeline() {
    thread_local LastPipeline last;
    return last;
}

// Never destroyed: the pipelines belong to a device that may already be gone when statics die, and
// the device's teardown (ClearCachedPipelines) is the place to destroy them.
PipelineStore& Pipelines() {
    static auto* store = new PipelineStore();
    return *store;
}

// Whether the entry's objects belong to the device the context names. A context without a buffer
// pool (tests) is identified by the handle alone.
bool alive(const PipelineStore::Entry& entry, const Context& context) {
    if (entry.device != context.device) return false;
    return context.bufferPool == nullptr || entry.pool.lock() == context.bufferPool;
}

// Drops an entry of a device that is gone: its objects went with the device, so they are forgotten,
// not destroyed.
std::list<PipelineStore::Entry>::iterator abandon(PipelineStore& store, std::list<PipelineStore::Entry>::iterator it) {
    it->pipeline->Abandon();
    ++store.generation;
    store.index.erase(it->hash);
    return store.entries.erase(it);
}

void reportPipelines(PipelineStore& store) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - store.lastReport < std::chrono::seconds(10)) return;
    store.lastReport = now;
    const auto lookups = store.hits + store.misses + store.uncached;
    AgcDriver::ProfilePrint_nid_no_patch( "[pipecache] %llu lookups over 10 s: %llu hits (%.0f%%; %llu repeats of the thread's previous key), %llu misses, %llu private (no variant id), %llu evicted, %zu cached; async: %llu compiles started (%.1f s compiling), %llu draws skipped while compiling, %zu in flight\n", static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(store.hits), lookups != 0 ? 100.0 * static_cast<double>(store.hits) / static_cast<double>(lookups) : 0.0, static_cast<unsigned long long>(store.identityHits), static_cast<unsigned long long>(store.misses), static_cast<unsigned long long>(store.uncached), static_cast<unsigned long long>(store.evicted), store.entries.size(), static_cast<unsigned long long>(store.asyncStarted), store.asyncSeconds, static_cast<unsigned long long>(store.asyncSkips), store.inflight);
    store.hits = store.misses = store.uncached = store.evicted = store.identityHits = 0;
    store.asyncStarted = store.asyncSkips = 0;
    store.asyncSeconds = 0.0;
}

bool asyncPipelines() {
    static const bool enabled = [] { const char* text = std::getenv("APS5_ASYNC_PIPELINES"); return text != nullptr && std::strcmp(text, "0") != 0; }();
    return enabled;
}

class PipelineCompilePool {
public:
    static PipelineCompilePool& Instance() {
        static auto* pool = new PipelineCompilePool();
        return *pool;
    }

    void Submit(std::function<void()> task) {
        {
            std::lock_guard lock(mutex);
            tasks.push_back(std::move(task));
        }
        ready.notify_one();
    }

private:
    PipelineCompilePool() {
        const char* text = std::getenv("APS5_ASYNC_PIPELINE_THREADS");
        const auto requested = text != nullptr ? static_cast<unsigned>(std::strtoul(text, nullptr, 10)) : 0u;
        const auto count = requested != 0 ? requested : std::max(2u, std::thread::hardware_concurrency() / 2u);
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setstacksize(&attributes, 64u << 20u);
        for (unsigned index = 0; index < count; ++index) {
            pthread_t thread;
            if (pthread_create(&thread, &attributes, &PipelineCompilePool::Main, this) == 0) pthread_detach(thread);
        }
        pthread_attr_destroy(&attributes);
    }

    static void* Main(void* argument) {
        auto& pool = *static_cast<PipelineCompilePool*>(argument);
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock lock(pool.mutex);
                pool.ready.wait(lock, [&] { return !pool.tasks.empty(); });
                task = std::move(pool.tasks.front());
                pool.tasks.pop_front();
            }
            task();
        }
        return nullptr;
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::function<void()>> tasks;
};

void insertPipeline(PipelineStore& store, VkDevice device, const std::shared_ptr<BufferPool>& pool, std::uint64_t hash, const std::vector<std::byte>& key, const std::shared_ptr<Pipeline>& pipeline) {
    store.entries.push_back({device, pool, hash, key, pipeline});
    store.index[hash] = std::prev(store.entries.end());
    static const std::size_t bound = [] { const char* text = std::getenv("APS5_PIPELINE_CACHE_ENTRIES"); const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 0ull; return value != 0 ? static_cast<std::size_t>(value) : std::size_t{4096}; }();
    while (store.entries.size() > bound) {
        const auto victim = std::find_if(store.entries.begin(), store.entries.end(), [](const PipelineStore::Entry& entry) { return entry.pipeline.use_count() == 1; });
        if (victim == store.entries.end()) break;
        store.index.erase(victim->hash);
        store.entries.erase(victim);
        ++store.generation;
        ++store.evicted;
    }
}

void compileInBackground(PipelineStore& store, const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, std::uint64_t hash, const std::vector<std::byte>& key) {
    struct Job {
        Context context;
        State state;
        VertexInputLayout vertexInput;
        VkDescriptorSetLayout setLayout;
        std::vector<ShaderRecompiler::RecompileResult> programs;
        std::vector<CompiledShader> shaders;
        VkImageLayout attachmentLayout;
        std::uint64_t hash;
    };
    auto job = std::make_shared<Job>(Job{context, state, vertexInput, resources.Layout(), {}, {}, attachmentLayout, hash});
    job->programs.reserve(shaders.size());
    for (const auto& shader : shaders) job->programs.push_back(*shader.program);
    for (std::size_t index = 0; index < shaders.size(); ++index) job->shaders.push_back({shaders[index].stage, &job->programs[index], shaders[index].pushConstantOffset});
    store.pending.emplace(hash, key);
    ++store.inflight;
    ++store.asyncStarted;
    PipelineCompilePool::Instance().Submit([job, &store] {
        const auto started = std::chrono::steady_clock::now();
        std::shared_ptr<Pipeline> pipeline;
        try {
            pipeline = std::make_shared<Pipeline>(job->context, job->state, job->vertexInput, job->setLayout, job->shaders, job->attachmentLayout);
        } catch (const std::exception&) {
            pipeline = nullptr;
        }
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::lock_guard lock(store.mutex);
        store.asyncSeconds += seconds;
        const auto found = store.pending.find(job->hash);
        if (found != store.pending.end()) {
            if (pipeline == nullptr) store.failed.insert(job->hash);
            else if (!store.index.contains(job->hash)) insertPipeline(store, job->context.device, job->context.bufferPool, job->hash, found->second, pipeline);
            store.pending.erase(found);
        }
        --store.inflight;
        store.idle.notify_all();
    });
}

}

std::shared_ptr<Pipeline> CachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout, bool mayDefer) {
    static const bool disabled = std::getenv("APS5_NO_PIPELINE_CACHE") != nullptr;
    if (disabled) ForkTrap("pipeline uncached");
    if (disabled) return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    auto& store = Pipelines();
    std::unique_lock lock(store.mutex);
    reportPipelines(store);
    const bool identity = pipelineIdentity();
    auto* last = identity ? &ThreadLastPipeline() : nullptr;
    if (identity) {
        pipelineKeyInto(last->scratch, context, state, vertexInput, resources, shaders, attachmentLayout);
        if (!last->scratch.empty() && last->generation == store.generation && last->scratch == last->key && alive(*last->entry, context)) {
            ++store.hits;
            ++store.identityHits;
            store.entries.splice(store.entries.end(), store.entries, last->entry);
            return last->entry->pipeline;
        }
        last->generation = 0;
    } else {
        pipelineKeyInto(store.scratchKey, context, state, vertexInput, resources, shaders, attachmentLayout);
    }
    const auto& key = identity ? last->scratch : store.scratchKey;
    const auto remember = [&](std::list<PipelineStore::Entry>::iterator entry) {
        if (!identity) return;
        last->key = key;
        last->entry = entry;
        last->generation = store.generation;
    };
    if (key.empty()) {
        ForkTrap("pipeline uncached");
        ++store.uncached;
        return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    }
    const auto hash = hashKey(key);
    if (const auto found = store.index.find(hash); found != store.index.end()) {
        const auto it = found->second;
        if (it->key == key) {
            // The key names the device handle, which the loader may reuse for a device created after
            // this one was destroyed without ClearCachedPipelines: such an entry is a miss.
            if (alive(*it, context)) {
                ++store.hits;
                store.entries.splice(store.entries.end(), store.entries, it);
                remember(it);
                return it->pipeline;
            }
            ForkTrap("pipeline abandoned");
            abandon(store, it);
        } else {
            // A different configuration with the same hash keeps the resident entry; this one stays private.
            ForkTrap("pipeline collision");
            ++store.uncached;
            return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
        }
    }
    ForkTrap("pipeline miss");
    if (const auto found = store.pending.find(hash); found != store.pending.end() && (!mayDefer || found->second != key)) {
        ++store.uncached;
        return std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    }
    if (asyncPipelines() && mayDefer && !store.failed.contains(hash)) {
        static const auto waitMs = [] {
            const char* text = std::getenv("APS5_ASYNC_PIPELINE_WAIT_MS");
            return text != nullptr ? std::strtoull(text, nullptr, 10) : 10ull;
        }();
        const auto waitForCompile = [&]() -> std::shared_ptr<Pipeline> {
            const auto now = std::chrono::steady_clock::now();
            if (now - store.waitWindow > std::chrono::seconds(1)) {
                store.waitWindow = now;
                store.waitedInWindow = std::chrono::steady_clock::duration::zero();
            }
            if (waitMs == 0 || store.waitedInWindow > std::chrono::milliseconds(100)) return nullptr;
            const std::vector<std::byte> wanted = key;
            store.idle.wait_for(lock, std::chrono::milliseconds(waitMs), [&] { return !store.pending.contains(hash); });
            store.waitedInWindow += std::chrono::steady_clock::now() - now;
            const auto found = store.index.find(hash);
            if (found == store.index.end() || found->second->key != wanted || !alive(*found->second, context)) return nullptr;
            store.entries.splice(store.entries.end(), store.entries, found->second);
            return found->second->pipeline;
        };
        if (store.pending.contains(hash)) {
            if (auto ready = waitForCompile()) return ready;
            ++store.asyncSkips;
            return nullptr;
        }
        ++store.misses;
        for (auto it = store.entries.begin(); it != store.entries.end();) {
            it = alive(*it, context) ? std::next(it) : abandon(store, it);
        }
        compileInBackground(store, context, state, vertexInput, resources, shaders, attachmentLayout, hash, key);
        if (auto ready = waitForCompile()) return ready;
        ++store.asyncSkips;
        return nullptr;
    }
    ++store.misses;
    // Entries of another device belong to one the driver replaced (it does so under the GpuMutex
    // before any draw reaches the new device), whose objects went with it: forget them.
    for (auto it = store.entries.begin(); it != store.entries.end();) {
        it = alive(*it, context) ? std::next(it) : abandon(store, it);
    }
    auto pipeline = std::make_shared<Pipeline>(context, state, vertexInput, resources, shaders, attachmentLayout);
    insertPipeline(store, context.device, context.bufferPool, hash, key, pipeline);
    if (const auto found = store.index.find(hash); found != store.index.end() && found->second->pipeline == pipeline) remember(found->second);
    return pipeline;
}

PipelinePeek PeekCachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout) {
    static const bool disabled = std::getenv("APS5_NO_PIPELINE_CACHE") != nullptr;
    if (disabled) return {};
    // The key is built outside the store mutex, into this thread's own scratch.
    thread_local std::vector<std::byte> key;
    pipelineKeyInto(key, context, state, vertexInput, resources, shaders, attachmentLayout);
    if (key.empty()) return {};
    const auto hash = hashKey(key);
    auto& store = Pipelines();
    std::lock_guard lock(store.mutex);
    const auto found = store.index.find(hash);
    if (found == store.index.end() || found->second->key != key || !alive(*found->second, context)) return {};
    return {found->second->pipeline, hash, &resources, context.device};
}

std::shared_ptr<Pipeline> TakePeekedPipeline(const Context& context, const PipelinePeek& peek, const ShaderResources& resources) {
    if (peek.device != context.device || peek.resources != &resources) return nullptr;
    const auto peeked = peek.pipeline.lock();
    if (peeked == nullptr) return nullptr;
    auto& store = Pipelines();
    std::lock_guard lock(store.mutex);
    reportPipelines(store);
    // An entry's pipeline object is made for that entry alone, and the locked peek names that very
    // object (a weak pointer locks only while the object it watched lives), so an entry holding it
    // is the one inserted under the peeked key.
    const auto found = store.index.find(peek.hash);
    if (found == store.index.end() || found->second->pipeline != peeked || !alive(*found->second, context)) return nullptr;
    ++store.hits;
    store.entries.splice(store.entries.end(), store.entries, found->second);
    return found->second->pipeline;
}

void ClearCachedPipelines(VkDevice device) {
    auto& store = Pipelines();
    std::unique_lock lock(store.mutex);
    store.idle.wait(lock, [&] { return store.inflight == 0; });
    store.failed.clear();
    for (auto it = store.entries.begin(); it != store.entries.end();) {
        if (it->device != device) {
            ++it;
            continue;
        }
        store.index.erase(it->hash);
        it = store.entries.erase(it);
        ++store.generation;
    }
}

}
