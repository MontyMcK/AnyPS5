#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEJOURNAL_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEJOURNAL_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineIdentity.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "Recompiler.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::Graphics {

// What a graphics pipeline is built from besides the device: the stage results (named by their
// run-stable identity), the vertex input layout, the descriptor set layout key (the binding, type,
// count and stage flags per binding that ShaderResources::LayoutKey lists, and from which the
// DescriptorCache builds the layout) and the state fields the Pipeline constructor reads, with every
// other State field cleared (CanonicalPipelineState). The pipeline store keys on exactly this, and
// the journal records it so the next run builds the same VkPipeline before the title asks for it.
// A generated stage (the rect-list control and evaluation pair) is built from the vertex and
// fragment results and the device's limits; its identity is in the key like any other stage's, so a
// replayed pipeline built from another device's generated module is missed, not used.
struct GraphicsPipelineRecipe {
    struct Stage {
        ShaderRecompiler::ShaderStage stage;
        PipelineIdentity identity;
        std::uint32_t pushConstantOffset;
        bool generated;
    };
    VkImageLayout attachmentLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    std::vector<Stage> stages;
    VertexInputLayout vertexInput;
    std::vector<std::uint32_t> layoutKey;
    State state;
};

[[nodiscard]] State CanonicalPipelineState(const State& state);
[[nodiscard]] bool GeneratedPipelineStage(const State& state, ShaderRecompiler::ShaderStage stage);
[[nodiscard]] GraphicsPipelineRecipe MakeGraphicsPipelineRecipe(const State& state, const VertexInputLayout& vertexInput, std::span<const std::uint32_t> layoutKey, std::span<const CompiledShader> shaders, VkImageLayout attachmentLayout);
[[nodiscard]] std::vector<std::byte> EncodeGraphicsPipelineRecipe(const GraphicsPipelineRecipe& recipe);
[[nodiscard]] bool DecodeGraphicsPipelineRecipe(std::span<const std::byte> bytes, GraphicsPipelineRecipe& recipe);
// The stage fields a pipeline is built from (the module, specialization constants, subgroup width,
// push constant use, the bindings' shape): what the journal stores of a result, and what a result
// decoded from it holds (variant ids zero, everything else default). Decode fails on a truncated or
// malformed record or one whose identity differs from the result it decodes to.
void EncodePipelineModule(const ShaderRecompiler::RecompileResult& result, std::vector<std::byte>& out);
[[nodiscard]] bool DecodePipelineModule(std::span<const std::byte> bytes, ShaderRecompiler::RecompileResult& result);
// The VkDescriptorSetLayoutBinding list a layout key describes (ShaderResources builds the key from
// exactly these four fields), for DescriptorCache::Layout; false for a malformed key.
[[nodiscard]] bool DescribeLayoutKey(std::span<const std::uint32_t> key, std::vector<VkDescriptorSetLayoutBinding>& bindings);

// The title first: a pipeline build on behalf of a draw or dispatch (or a wait for one) is held in
// a GamePipelineBuild for its duration. While one is held the replay starts no build (a worker
// waits until none has been held for a short quiet window), and a replay build already running is
// raised from its low priority to normal until it ends, so nothing the title then waits on (the
// build itself, a driver lock it holds) is starved by the title's own threads. A no-op on a replay
// thread. Process-wide: one device replays at a time. Lock-free: it may be held under the store's
// or the compute map's mutex.
class GamePipelineBuild {
public:
    GamePipelineBuild();
    ~GamePipelineBuild();
    GamePipelineBuild(const GamePipelineBuild&) = delete;
    GamePipelineBuild& operator=(const GamePipelineBuild&) = delete;

private:
    bool counted = false;
};
// Around a driver pipeline compile (vkCreate*Pipelines) on a replay thread: the thread runs at
// THREAD_PRIORITY_LOWEST for the compile unless a title build is held, and is raised back to
// normal by a title build that starts meanwhile and when the scope ends. Replay threads run at
// normal priority everywhere else, so the store, descriptor cache and compute map locks they take
// briefly (which every draw and dispatch also takes) are never held by a starved thread. A no-op
// on other threads.
class ReplayCompileScope {
public:
    ReplayCompileScope();
    ~ReplayCompileScope();
    ReplayCompileScope(const ReplayCompileScope&) = delete;
    ReplayCompileScope& operator=(const ReplayCompileScope&) = delete;

private:
    bool active = false;
};
// Whether a title pipeline build is held now (the replay defers a build it was about to start).
[[nodiscard]] bool GamePipelineBuildPending();
// A title build that found its pipeline being built by the replay and waited for that build
// instead of building it again (APS5_PROFILE_DRAW counters).
void CountGameWaitOnReplay(double milliseconds);

// What one replay entry came to: built and inserted, present (resident, or already being built by
// the title or another replay thread), rejected (unusable), or deferred (a title build started; the
// entry is tried again once the title is quiet).
enum class PipelineReplayOutcome { Built, Present, Rejected, Deferred };

struct PipelineJournalCounters {
    std::uint64_t loadedRecipes = 0;
    std::uint64_t loadedModules = 0;
    std::uint64_t loadedBytes = 0;
    std::uint64_t droppedRecords = 0;
    double loadMs = 0;
    std::uint64_t replayedGraphics = 0;
    std::uint64_t replayedCompute = 0;
    std::uint64_t replayPresent = 0;
    std::uint64_t replaySkipped = 0;
    std::uint64_t replayRejected = 0;
    std::uint64_t replayFailed = 0;
    double replayMs = 0;
    std::uint32_t replayThreads = 0;
    std::uint64_t recordedGraphics = 0;
    std::uint64_t recordedCompute = 0;
    std::uint64_t recordedModules = 0;
    std::uint64_t writtenBytes = 0;
    std::uint64_t writeFailures = 0;
    // The game-first gate: worker pauses for title builds (and their total), entries deferred
    // because a title build started, title builds that started while a replay build ran (each
    // raised the running replay builds' priority), title waits on a replay build of the same
    // pipeline (and their total), and the replay's own build time (total and longest).
    std::uint64_t replayPauses = 0;
    double replayPausedMs = 0;
    std::uint64_t replayDeferred = 0;
    std::uint64_t gameBuildsOverlapped = 0;
    std::uint64_t gameWaitsOnReplay = 0;
    double gameWaitOnReplayMs = 0;
    double replayBuildMs = 0;
    double replayLongestBuildMs = 0;
};

// A decimal switch of the pipeline store or journal, or its default when unset or malformed (reported once).
[[nodiscard]] std::uint64_t PipelineSwitch(const char* name, std::uint64_t fallback);
[[nodiscard]] bool PipelineJournalEnabled();
// True on a replay thread while it builds an entry: what it builds is already in the file.
[[nodiscard]] bool PipelineJournalReplaying();
[[nodiscard]] std::filesystem::path PipelineJournalPath();
[[nodiscard]] PipelineJournalCounters PipelineJournalTotals();
void RecordGraphicsPipeline(const GraphicsPipelineRecipe& recipe, std::span<const CompiledShader> shaders);
// A compute pipeline over the set layout `layoutKey` describes (its dispatch's ShaderResources).
void RecordComputePipeline(const ShaderRecompiler::RecompileResult& shader, bool pushConstants, std::span<const std::uint32_t> layoutKey);
void FlushPipelineJournal();
void ReportPipelineJournal();

// The replay runs on a few low-priority threads (APS5_PIPELINE_JOURNAL_THREADS, 2 by default) and
// yields to the title (GamePipelineBuild). Cancel stops the threads taking further entries and
// wakes a paused one (the pipeline each is building completes); Wait joins them. A device cancels
// its replay before it is replaced or destroyed.
class PipelineJournalReplay {
public:
    virtual ~PipelineJournalReplay() = default;
    virtual void Cancel() = 0;
    virtual void Wait() = 0;
    [[nodiscard]] virtual bool Done() const = 0;
};

struct PipelineJournalHooks {
    std::function<PipelineReplayOutcome(const ShaderRecompiler::RecompileResult& shader, bool pushConstants, std::span<const std::uint32_t> layoutKey)> compute;
};

// The journal belongs to the device that wrote it, named by the same properties as the driver's
// pipeline cache file; a replay for another device starts the file over.
[[nodiscard]] std::uint64_t PipelineJournalDeviceFingerprint(const VkPhysicalDeviceProperties& properties);
[[nodiscard]] std::shared_ptr<PipelineJournalReplay> StartPipelineJournalReplay(const Context& context, std::uint64_t deviceFingerprint, PipelineJournalHooks hooks);

}

#endif
