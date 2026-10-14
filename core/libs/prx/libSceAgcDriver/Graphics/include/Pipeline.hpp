#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINE_HPP

#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <set>
#include <string_view>

namespace AgcDriver::Graphics {

struct VertexInputLayout;

void LogPipelineStatistics_nid_no_patch(const Context& context, VkPipeline pipeline, std::string_view header = {});

// The attachments of one render pass instance. Work recorded with it references the handle until
// the batch completes, so a recorded draw keeps the object (Recorder::Keep) like its pipeline.
class Framebuffer {
public:
    Framebuffer(const Context& context, VkRenderPass renderPass, std::span<const VkImageView> targets, VkExtent2D extent, std::uint32_t layers = 1);
    ~Framebuffer();
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    VkFramebuffer Handle() const { return framebuffer; }
    // Forgets the handle without destroying it (the device it belongs to is already gone).
    void Abandon() noexcept;

private:
    Context context;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

// The shader modules, layout, render pass and VkPipeline of one draw configuration. Viewport,
// scissor, depth bounds and depth bias are dynamic state set at Begin, so pipelines are shared by draws that differ only there
// (see CachedPipeline).
class Pipeline {
public:
    // `state` carries the blend states with the outputs the pixel shader lacks already masked and
    // `vertexInput` the layout of the draw's vertex descriptors; the shaders were validated by the
    // caller (ValidateShaders). `attachmentLayout` is the layout the color attachments are in
    // before, during and after the pass (GENERAL for resident targets, which then need no
    // transitions).
    Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    Pipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, VkDescriptorSetLayout setLayout, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout);
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    VkPipelineLayout Layout() const;
    std::shared_ptr<Framebuffer> AcquireFramebuffer(std::span<const VkImageView> targets, std::span<const std::shared_ptr<StorageTexture>> owners, VkExtent2D extent, std::uint32_t layers = 1);
    // Begins the render pass on the framebuffer, binds the pipeline and sets its dynamic state.
    void Begin(VkCommandBuffer commands, const Framebuffer& framebuffer, VkExtent2D extent, const State& state) const;
    // The same inside a render pass another pipeline of the same attachments began (compatible by
    // construction: the attachment formats alone decide).
    void Continue(VkCommandBuffer commands, const State& state) const;
    void PushConstants(VkCommandBuffer commands, VkShaderStageFlags stages, std::span<const std::byte, PipelinePushConstantBytes> bytes) const;
    // Binds `set` as set 0 at the graphics bind point with this pipeline's layout (a draw inside a
    // pass this pipeline began or continued).
    void BindSet(VkCommandBuffer commands, VkDescriptorSet set) const;
    // Forgets the Vulkan objects without destroying them: for entries of a device that is already gone.
    void Abandon() noexcept;

private:
    struct CachedFramebuffer {
        std::vector<VkImageView> views;
        std::vector<std::weak_ptr<StorageTexture>> owners;
        VkExtent2D extent;
        std::uint32_t layers = 1;
        std::shared_ptr<Framebuffer> framebuffer;
    };
    void release() noexcept;
    Context context;
    std::vector<VkShaderModule> _modules;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::size_t attachments = 0;
    std::size_t colorAttachments = 0;
    bool depthBounds = false;
    bool depthBias = false;
    std::vector<CachedFramebuffer> framebuffers;
};

// The pipeline for this configuration, shared across draws and frames: keyed by the stages' variant
// ids (the rect-list tessellation pair by the vertex and fragment ids that generate it), the vertex
// input layout, the descriptor set layout key, push constant stages and every pipeline state field.
// A configuration whose stage has no variant id gets a private pipeline. Entries are LRU-bounded and
// only evicted once no recorded draw holds them. Debug aid: APS5_NO_PIPELINE_CACHE=1 builds one per
// draw as before.
std::shared_ptr<Pipeline> CachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, bool mayDefer = false);
// A pipeline store hit found off the committer (a draw speculation's pipeline, live
// `spec_pipeline`): the entry's key hash, its pipeline (weakly: a peek must not pin the entry against
// eviction or outlive the device; while it locks, its identity is the entry's) and the resources
// object whose layout the key named.
struct PipelinePeek {
    std::weak_ptr<Pipeline> pipeline;
    std::uint64_t hash = 0;
    const ShaderResources* resources = nullptr;
    VkDevice device = VK_NULL_HANDLE;
};
// CachedPipeline's hit without its side effects: no compile, insert, wait, LRU move or counter;
// an empty peek when the configuration is not cached (or private).
PipelinePeek PeekCachedPipeline(const Context& context, const State& state, const VertexInputLayout& vertexInput, const ShaderResources& resources, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout);
// The committer's use of a peek for a draw of the same configuration over `resources`: when the
// store still maps the peek's key hash to the peeked pipeline of the context's device (the entry,
// the only one holding that pipeline, carries the peeked key), counts the hit and moves the entry
// as CachedPipeline's hit does and returns the pipeline, which is then exactly CachedPipeline's
// answer; null otherwise (the caller asks CachedPipeline).
std::shared_ptr<Pipeline> TakePeekedPipeline(const Context& context, const PipelinePeek& peek, const ShaderResources& resources);
// Destroys the cached pipelines of a device; to be called before the device goes away. Without it,
// entries of a gone device are recognised by their buffer pool (made and reset with the device, so
// it tells device instances apart when the loader reuses a VkDevice handle) and forgotten unused.
void ClearCachedPipelines(VkDevice device);
// The vertex and index buffer binds of a draw inside a pass a Pipeline began or continued
// (`indexed`: the index buffer too).
void BindDrawBuffers(const Context& context, VkCommandBuffer commands, std::span<const VkBuffer> vertexBuffers, std::span<const VkDeviceSize> vertexOffsets, bool indexed, VkBuffer indexBuffer, VkDeviceSize indexOffset, VkIndexType indexType);
// Live `bind_elide` (env APS5_BIND_ELIDE): inside one render pass of one command buffer, the
// descriptor set, push constant, vertex and index buffer binds and the depth bias and depth bounds
// a draw would record identical to what the pass already holds are skipped (see Pipeline.cpp).
bool BindElision();
// The binds recorded and skipped so far by kind (set, push constants, vertex buffers, index
// buffer, depth bias, depth bounds), for the [binds] line; taking resets them.
struct BindCounts {
    std::array<std::uint64_t, 6> recorded{};
    std::array<std::uint64_t, 6> skipped{};
};
BindCounts TakeBindCounts();
// Device limit checks of the viewport, which is dynamic state and so no longer checked by Pipeline.
void ValidateViewport(const Context& context, const VkViewport& viewport);
void ValidateDepthBounds(const Context& context, const State& state);

void ValidateShaderPair(const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment);
// Returns the color attachment locations the pixel shader writes.
std::set<std::uint32_t> ValidateShaders(std::span<const CompiledShader> shaders, const State& state, const VkPhysicalDeviceSubgroupProperties& subgroup, bool fragmentShaderBarycentric, bool descriptorIndexing = false, bool imageInt64Atomics = false, bool geometryShader = false, bool sampleRateShading = false, bool viewportIndexLayer = false);

}

#endif
