#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSPECULATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSPECULATION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

struct DrawSpeculation {
    enum class Kind : std::uint8_t { None, Address, Template };
    using Miss = ShaderResources::SpeculationMiss;
    Kind kind = Kind::None;
    VkDevice device = VK_NULL_HANDLE;
    ResourceCache::Key key;
    std::shared_ptr<ShaderResources> source;
    bool aliasesOk = false;
    std::vector<ShaderResources::MovedBuffer> moved;
    std::size_t movedGuest = 0;
    std::vector<ShaderResources::SpeculatedImage> movedImages;
    ShaderResources::ProofSpeculation proof;
    ShaderResources::MovedProof movedProof;
    std::uint64_t frontier = 0;
    bool windowed = false;
    std::weak_ptr<const GuestBufferMemory::AddressSpace> space;
    std::shared_ptr<ShaderResources> use;
    // The address use the committer's UseAddressSpeculation refused (live `defer_drops`): kept here
    // for the committer to hand to the open batch rather than destroyed inside the lookup. Never
    // used again.
    std::shared_ptr<ShaderResources> refusedUse;
    Miss miss = Miss::None;
    // ResourceCache::HashKey(key), computed on the prep thread (live `spec_key_hash`).
    std::size_t keyHash = 0;
    bool hashed = false;
    // The pipeline store hit of the draw's configuration over `source` (template) or `use`
    // (address), peeked on the prep thread with `pipelineWith`'s masked state and vertex input
    // (live `spec_pipeline`, SpeculateDrawPipeline). The committer takes it only for that prebuild.
    PipelinePeek pipeline;
    const DrawPrebuild* pipelineWith = nullptr;
    // The cache entry the prep thread peeked `source` in (live `spec_entry_stamp`): the committer
    // proves it still the key's entry by its stamp (ResourceCache::TouchStamped) instead of
    // hashing and comparing the key under the cache mutex.
    ResourceCache::Handle entry;
    // The ranges ScreenTokenWindow screens (the proof's surfaces, then the moved buffers' ranges),
    // gathered on the prep thread once both are final (live `spec_window_reads`).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> windowReads;
    bool windowReadsMade = false;
};

// Live `spec_entry_stamp` (env APS5_SPEC_ENTRY_STAMP): see DrawSpeculation::entry.
bool SpeculatedEntryStamps();
// Cache entries the committer proved by their stamp so far (tests).
std::uint64_t SpeculatedEntryStampsTaken();
// Live `defer_drops` (env APS5_DEFER_DROPS): see Draw.cpp deferDrop.
bool DeferDropsEnabled();
// Live `spec_window_reads` (env APS5_SPEC_WINDOW_READS): see DrawSpeculation::windowReads.
bool SpeculatedWindowReads();

// Live `spec_key_hash` (env APS5_SPEC_KEY_HASH): the prep thread hashes a speculation's (and a
// prebuild's) resource key, and the committer's resource cache lookup reuses the hash.
bool SpeculatedKeyHash();
// Live `spec_pipeline` (env APS5_SPEC_PIPELINE): the prep thread peeks the pipeline of a
// speculated draw (SpeculateDrawPipeline), and the committer takes it (TakePeekedPipeline)
// instead of building and comparing the pipeline key.
bool SpeculatedPipelines();
// Speculated pipelines the committer took so far (tests; the [spec] line reports per 10 s).
std::uint64_t SpeculatedPipelinesTaken();
// Peeks `speculation`'s pipeline with `prebuilt`'s masked state and vertex input (prep thread,
// after PrebuildDraw); nothing when either is unusable or the store has no such entry.
void SpeculateDrawPipeline(const Context& context, const State& state, std::span<const CompiledShader> shaders, DrawSpeculation& speculation, const std::shared_ptr<DrawPrebuild>& prebuilt);
// The refusal of this thread's last SpeculateDrawResources call (None: it speculated or was not
// called since the last take), for the committer's [commit] profile.
DrawSpeculation::Miss TakeSpeculationRefusal();
const char* SpeculationMissName(DrawSpeculation::Miss miss);
// What became of the speculation the committer's last draw on this thread brought: 0 none, 1
// accepted, 2 refused (the [commit] profile).
std::uint8_t TakeCommitSpeculationOutcome();

enum class WindowVerdict : std::uint8_t { Clear, Overlap, Unknown, Overflow };
struct AppliedWindow {
    std::uint64_t (*frontier)() = nullptr;
    WindowVerdict (*check)(std::uint64_t fromExclusive, std::span<const std::pair<std::uint64_t, std::uint64_t>> reads) = nullptr;
};
void SetAppliedWindow(AppliedWindow window);
bool DrawTokensEnabled();
bool TargetTokensEnabled();
bool InputTokensEnabled();
bool VerifyDrawTokens();
enum class TokenFamily : std::uint8_t { Proof, Moved };
void CountTokenOutcome(TokenFamily family, ShaderResources::TokenFallback outcome);
void ScreenTokenWindow(DrawSpeculation& speculation);
std::uint64_t TakeCrossEpochTargetProofs();

bool DrawSpeculationEnabled();
std::shared_ptr<ForkedDraw> ForkDrawSpeculated(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const std::shared_ptr<DrawPrebuild>& prebuilt, std::shared_ptr<const DrawRecipe>* recipe, const ForkSeed* seed, bool seedCurrent, std::size_t slot, std::shared_ptr<DrawSpeculation> speculation);
bool VerifyDrawSpeculation();
std::shared_ptr<DrawSpeculation> SpeculateDrawResources(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots);
void CountSpeculationMiss(DrawSpeculation::Miss miss, bool committer);

class SpeculationScope {
public:
    explicit SpeculationScope(std::shared_ptr<DrawSpeculation> speculation);
    ~SpeculationScope();
    SpeculationScope(const SpeculationScope&) = delete;
    SpeculationScope& operator=(const SpeculationScope&) = delete;
};

}

#endif
