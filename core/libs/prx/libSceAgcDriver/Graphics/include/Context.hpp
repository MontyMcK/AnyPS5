#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include "prx/libSceAgcDriver/Graphics/include/DeferredVulkan.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

class TextureDetiler;
class GpuColorTransfer;
class BufferPool;
class TextureCache;
class RenderCache;
class DrawQueue;
class GraphicsPipelineCache;
class Recorder;
class DescriptorCache;
class SamplerCache;
class ShaderResources;

inline void Require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error("AGC graphics: " + reason);
}

inline void (*&DeviceLostReporter())() {
    static void (*reporter)() = nullptr;
    return reporter;
}

inline void Check(VkResult result, const char* operation) {
    if (result == VK_ERROR_DEVICE_LOST && DeviceLostReporter() != nullptr) DeviceLostReporter()();
    if (result != VK_SUCCESS) throw std::runtime_error(std::string("AGC graphics: ") + operation + ": Vulkan result " + std::to_string(result));
}

inline void Require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(std::string("AGC graphics: ") + reason);
}

// Device entry points resolved once per device (VulkanDevice's State fills it after setup): the
// loader's vkGetDeviceProcAddr is a name lookup under its global mutex per call, paid at every
// record site otherwise. Null members (tests, APS5_NO_PROC_TABLE=1) resolve per call (Resolved).
struct DeviceFunctions {
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkCmdFillBuffer cmdFillBuffer = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants cmdPushConstants = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdDispatchIndirect cmdDispatchIndirect = nullptr;
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass cmdEndRenderPass = nullptr;
    PFN_vkCmdSetViewport cmdSetViewport = nullptr;
    PFN_vkCmdSetScissor cmdSetScissor = nullptr;
    PFN_vkCmdSetDepthBounds cmdSetDepthBounds = nullptr;
    PFN_vkCmdSetDepthBias cmdSetDepthBias = nullptr;
    PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer = nullptr;
    PFN_vkCmdDraw cmdDraw = nullptr;
    PFN_vkCmdDrawIndexed cmdDrawIndexed = nullptr;
    PFN_vkCmdDrawIndirect cmdDrawIndirect = nullptr;
    PFN_vkCmdDrawIndexedIndirect cmdDrawIndexedIndirect = nullptr;
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer = nullptr;
    PFN_vkCmdClearColorImage cmdClearColorImage = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkGetFenceStatus getFenceStatus = nullptr;
};

// Per-thread count of vkGetDeviceProcAddr lookups made through Context::Function (the [vk] line).
inline std::uint64_t& DeviceProcLookups() {
    thread_local std::uint64_t count = 0;
    return count;
}

// vkGetDeviceProcAddr is a string lookup in the loader and the driver, and the record path asks it for
// an entry point on every Vulkan call. Entry points are remembered here per (resolver, device, name), keyed by
// the name's address: callers pass string literals. APS5_NO_PROC_CACHE=1 asks every time.
inline PFN_vkVoidFunction CachedDeviceProc(PFN_vkGetDeviceProcAddr deviceProc, VkDevice device, const char* name) {
    struct Entry {
        std::atomic<const char*> name{nullptr};
        std::atomic<VkDevice> device{VK_NULL_HANDLE};
        std::atomic<PFN_vkGetDeviceProcAddr> resolver{nullptr};
        std::atomic<PFN_vkVoidFunction> function{nullptr};
    };
    constexpr std::size_t Slots = 2048;
    constexpr std::size_t Probes = 16;
    static Entry table[Slots];
    static const bool disabled = std::getenv("APS5_NO_PROC_CACHE") != nullptr;
    if (disabled) return DeferredVulkanEntry(name, deviceProc(device, name));
    const auto hash = static_cast<std::size_t>((reinterpret_cast<std::uintptr_t>(name) * 0x9e3779b97f4a7c15ull) >> 53);
    for (std::size_t probe = 0; probe < Probes; ++probe) {
        auto& entry = table[(hash + probe) & (Slots - 1)];
        const char* key = entry.name.load(std::memory_order_acquire);
        if (key == name) {
            const auto function = entry.function.load(std::memory_order_acquire);
            if (function != nullptr && entry.device.load(std::memory_order_relaxed) == device && entry.resolver.load(std::memory_order_relaxed) == deviceProc) return function;
            continue;
        }
        if (key != nullptr) continue;
        const auto function = DeferredVulkanEntry(name, deviceProc(device, name));
        if (function == nullptr) return nullptr;
        const char* empty = nullptr;
        if (entry.name.compare_exchange_strong(empty, name, std::memory_order_acq_rel)) {
            entry.device.store(device, std::memory_order_relaxed);
            entry.resolver.store(deviceProc, std::memory_order_relaxed);
            entry.function.store(function, std::memory_order_release);
        }
        return function;
    }
    return DeferredVulkanEntry(name, deviceProc(device, name));
}

inline constexpr std::size_t EmptyBufferBytes = 16;

struct Context {
    VkDevice device;
    VkPhysicalDevice physical;
    VkQueue queue;
    VkCommandPool pool;
    PFN_vkGetDeviceProcAddr deviceProc;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties imageFormatProperties;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceLimits limits;
    bool tessellationShader = false;
    bool meshShader = false;
    VkPhysicalDeviceMeshShaderPropertiesEXT meshLimits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    bool depthClipControl = false;
    bool depthRangeUnrestricted = false;
    bool bufferDeviceAddress = false;
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    bool fragmentShaderBarycentric = false;
    bool samplerAnisotropy = false;
    bool textureCompressionBC = false;
    TextureDetiler* detiler = nullptr;
    GpuColorTransfer* colorTransfer = nullptr;
    mutable std::shared_ptr<BufferPool> bufferPool;
    TextureCache* textureCache = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    bool depthClamp = false;
    // Nonzero when VK_EXT_external_memory_host is enabled: the required host pointer alignment.
    VkDeviceSize hostImportAlignment = 0;
    bool dmaBufImport = false;
    RenderCache* renderCache = nullptr;
    DrawQueue* drawQueue = nullptr;
    GraphicsPipelineCache* graphicsPipelines = nullptr;
    // Batches GPU work across guest commands (see Recorder); null before the device finished setup.
    Recorder* recorder = nullptr;
    // Per-device caches of the immutable descriptor objects a ShaderResources build needs (set
    // layouts, descriptor pools, samplers); null (tests) means every build makes and destroys its own.
    DescriptorCache* descriptorCache = nullptr;
    SamplerCache* samplerCache = nullptr;
    // Indirect draw features of the device: records with a non-zero first instance, several records
    // per call, and a GPU-side draw count (VK_KHR_draw_indirect_count).
    bool drawIndirectFirstInstance = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool occlusionQueryPrecise = false;
    bool depthBounds = false;
    bool dualSrcBlend = false;
    bool depthBiasClamp = false;
    bool samplerFilterMinmax = false;
    bool nullDescriptor = false;
    bool conservativeRasterization = false;
    VkBuffer emptyBuffer = VK_NULL_HANDLE;
    // The device's list of recorded dispatches whose copied written buffers await a CPU write-back
    // (VulkanDevice's State::copiedWriters; the draw counterpart is DrawCopiedWriters): an indirect
    // draw whose records one of them produces reads them on the CPU. Null in tests.
    const std::vector<std::shared_ptr<ShaderResources>>* copiedWriters = nullptr;
    // The device's resolved entry points (see DeviceFunctions); null until the device set up.
    const DeviceFunctions* functions = nullptr;
    // VK_EXT_descriptor_indexing with non-uniform sampled/storage image array indexing enabled
    // (bindless image tables in graphics stages).
    bool descriptorIndexing = false;
    bool imageInt64Atomics = false;
    bool geometryShader = false;
    bool sampleRateShading = false;
    bool viewportIndexLayer = false;
    bool nullDescriptors = false;
    bool primitiveListRestart = false;
    bool imageViewMinLod = false;
    bool pipelineExecutableInfo = false;
    // VK_KHR_push_descriptor is enabled (APS5_NO_PUSH_DESCRIPTORS=1 leaves it off).
    bool pushDescriptors = false;
    std::uint32_t srgbDecodeFormats = 0;

    template<typename TFunction>
    TFunction Function(const char* name) const {
        Require(deviceProc != nullptr, "missing Vulkan device function resolver");
        ++DeviceProcLookups();
        const auto function = reinterpret_cast<TFunction>(CachedDeviceProc(deviceProc, device, name));
        if (function == nullptr) throw std::runtime_error(std::string("AGC graphics: missing Vulkan function: ") + name);
        return function;
    }

    // The table's entry point, or a per-call lookup while the table is absent or lacks it.
    template<typename TFunction>
    TFunction Resolved(TFunction DeviceFunctions::*member, const char* name) const {
        if (functions != nullptr && functions->*member != nullptr) return functions->*member;
        return Function<TFunction>(name);
    }

    std::uint32_t MemoryType(std::uint32_t mask, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((mask & (1u << i)) != 0 && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("AGC graphics: required Vulkan memory type is unavailable");
    }
};

}

#endif
