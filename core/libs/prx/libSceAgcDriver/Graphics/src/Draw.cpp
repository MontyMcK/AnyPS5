#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuCheckpoints.hpp"
#include <immintrin.h>
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawSpeculation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuColorTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DebugControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/NvPerfRanges.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include <algorithm>
#include <unordered_set>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

namespace AgcDriver::Graphics {

// The attachment proxy format of a color target: B10G11R11 for a blended COLOR_11_11_10 target
// (its packed image is repacked around the pass), otherwise the format's own proxy, if any.
// An attachment proxy a draw renders through, with the subresource it loads and stores.
struct ProxyTarget {
    std::shared_ptr<StorageTexture> texture;
    std::uint32_t mip = 0;
    std::uint32_t layer = 0;
    std::uint32_t layers = 1;
    bool operator==(const ProxyTarget&) const = default;
};

static VkFormat ColorProxyFormat(const Context& context, const ColorTarget& color) {
    if (color.blendProxy111110) return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    return AttachmentProxyFormat(context, color.format);
}

namespace {



}

namespace {

std::size_t colorKeyCount(const ColorTarget& color, std::size_t bytes) {
    if (color.mipCount != 1 || color.depth != 1 || !color.dccPipeAligned) return DccKeyBytes(bytes);
    return DccKeyCount(ColorTextureTileMode(color.tileMode), color.elementBytes, color.extent.width, color.extent.height, bytes);
}

std::array<std::byte, 16> clearTexel(const ColorTarget& color, DccKeys keys) {
    std::array<std::byte, 16> texel{};
    const auto elementBytes = static_cast<std::size_t>(color.elementBytes);
    Require(elementBytes != 0 && elementBytes <= texel.size(), "unexpected color element size");
    if (keys == DccKeys::ClearRegister) {
        Require(elementBytes <= sizeof(color.clearWords), "the DCC register clear of a texel over 64 bits is not modeled");
        std::memcpy(texel.data(), color.clearWords.data(), elementBytes);
    } else {
        Require(FillDccClear(color.format, keys, color.dccAlphaOnMsb, std::span(texel.data(), elementBytes)), "the target's format has no encoding of its DCC clear code");
    }
    return texel;
}

bool clearToTexel(StorageTexture& image, const std::array<std::byte, 16>& texel, std::uint32_t elementBytes, const char*& refusal) {
    if (image.Descriptor().dimension == TextureDimension::k2DArray && image.ImageLayers() > 1) {
        refusal = "layered resident";
        return false;
    }
    std::array<std::byte, 16> repeated{};
    for (std::size_t offset = 0; offset + elementBytes <= repeated.size(); offset += elementBytes) std::memcpy(repeated.data() + offset, texel.data(), elementBytes);
    std::array<std::uint32_t, 4> pattern{};
    std::memcpy(pattern.data(), repeated.data(), repeated.size());
    return image.FillClear(std::span<const std::uint32_t, 4>(pattern), StorageTexture::WholeImage, refusal);
}

void writeTexels(const ColorTarget& color, const std::array<std::byte, 16>& texel) {
    const auto elementBytes = static_cast<std::size_t>(color.elementBytes);
    std::vector<std::byte> texels(color.bytes);
    for (std::size_t offset = 0; offset + elementBytes <= texels.size(); offset += elementBytes) std::memcpy(texels.data() + offset, texel.data(), elementBytes);
    GuestMemory::Write(color.address, texels);
}

void storeClearTexels(const Context& context, const ColorTarget& color, const std::array<std::byte, 16>& texel) {
    StorageTexture::FlushPending(color.address, color.bytes, nullptr, "fast-clear materialization");
    const auto keys = ReadDccKeys(color.dccAddress, color.bytes);
    if (!IsDccClear(keys)) return;
    writeTexels(color, keys == DccKeys::ClearRegister ? texel : clearTexel(color, keys));
    MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
}

void materializeRegisterClear(const Context& context, const ColorTarget& color, StorageTexture& resident) {
    if (color.dccAddress == 0 || resident.Descriptor().dccAddress != color.dccAddress) return;
    if (CurrentDccKeys(color.dccAddress, color.bytes) != DccKeys::ClearRegister) return;
    const auto texel = clearTexel(color, DccKeys::ClearRegister);
    const char* refusal = nullptr;
    bool cleared = clearToTexel(resident, texel, color.elementBytes, refusal);
    const auto keyBytes = static_cast<std::size_t>(color.bytes / 256);
    if (!cleared && keyBytes != 0) {
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->PendingWriteOverlaps(color.dccAddress, keyBytes)) {
            Recorder::CountSync(2);
            recorder->SyncThrough(color.dccAddress, keyBytes);
            if (CurrentDccKeys(color.dccAddress, color.bytes) != DccKeys::ClearRegister) return;
            cleared = clearToTexel(resident, texel, color.elementBytes, refusal);
        }
    }
    if (cleared) {
        MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
        return;
    }
    storeClearTexels(context, color, texel);
    resident.Refresh();
}

bool materializeCmaskClear(const Context& context, const ColorTarget& color, StorageTexture* resident) {
    if (color.cmaskAddress == 0) return false;
    const auto metadataBytes = static_cast<std::uint64_t>(color.cmaskBytes) * 256u;
    const auto state = CurrentDccKeys(color.cmaskAddress, metadataBytes);
    if (state == DccKeys::Uncompressed) return false;
    Require(color.dccAddress == 0, std::string("CMASK of a DCC color target that is not all expanded is not modeled (") + DccKeysName(state) + ")");
    Require(state == DccKeys::Clear0000, std::string("CMASK whose tiles are not all fast-cleared or all expanded is not modeled (") + DccKeysName(state) + ")");
    const auto texel = clearTexel(color, DccKeys::ClearRegister);
    const char* refusal = nullptr;
    if (resident == nullptr || !clearToTexel(*resident, texel, color.elementBytes, refusal)) {
        StorageTexture::FlushPending(color.address, color.bytes, nullptr, "fast-clear materialization");
        writeTexels(color, texel);
        if (resident != nullptr) resident->Refresh();
    }
    MarkDccUncompressed(context, color.cmaskAddress, metadataBytes);
    return true;
}

void imageBarrier(const Context& context, VkCommandBuffer commands, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = sourceAccess;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void memoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The color surface as one untailed SW_64KB_R_X mip for the GPU detiler, with tightly packed linear rows.
TileMipLayout ColorTargetMip(const ColorTarget& color, const ColorTargetLayout& layout) {
    TileMipLayout mip{};
    mip.width = color.extent.width;
    mip.height = color.extent.height;
    mip.blocksPerRow = layout.BlocksPerRow();
    mip.pitchBytes = color.extent.width * color.elementBytes;
    mip.tiledSize = layout.Bytes();
    mip.linearSize = layout.LinearBytes();
    return mip;
}

// APS5_PROFILE_DRAW: per-draw phase timers in microseconds (a recorded draw's phases are far below
// the millisecond the old print rounded to), totalled over 10 s in the [draws] line.
enum DrawPhase : std::size_t { PhaseValidate, PhaseVertex, PhaseSetup, PhaseReadTarget, PhasePrepare, PhaseLookup, PhaseResources, PhasePipeline, PhaseRecord, PhaseKeep, PhaseSync, PhaseWriteBack, PhaseDescribe, PhaseCount };
constexpr std::array<const char*, PhaseCount> DrawPhaseNames{"validate", "vertex", "setup", "readTarget", "prepare", "lookup", "resources", "pipeline", "record", "keep", "sync", "writeBack", "describe"};

// Why a draw did not go into the recorder without a wait (counted in the [draws] line): its targets
// are not all resident, no recorder is active, a switch (APS5_SYNC_DRAWS, APS5_DUMP_TARGETS,
// APS5_SYNC_COMPLETION_DRAWS) forced it, or its completion work writes copied guest buffers or
// releases an address-based build's lease, which the CPU must not outrun (see `recorded` in Draw).
enum SyncReason : std::size_t { SyncNone, SyncNotResident, SyncNoRecorder, SyncDisabled, SyncCopiedWrites, SyncLease, SyncCount };
constexpr std::array<const char*, SyncCount> SyncReasonNames{"none", "non-resident target", "no recorder", "disabled", "copied writes", "lease"};
constexpr std::size_t IndirectPathCount = static_cast<std::size_t>(IndirectDrawPath::Count);
constexpr std::array<const char*, IndirectPathCount> IndirectDrawPathNames{"gpu-side", "patched SGPR not folded", "fetch offset unknown", "non-vertex path", "GE_INDX_OFFSET", "draw index", "vertex range too large", "feature gap", "pending image results", "pending label or copied write", "not imported", "disabled"};

// What became of one draw, for the totals.
struct DrawOutcome {
    // In the recorder; `waited` when the recorder was synced right after (copied writes or a lease).
    bool recorded = false;
    bool waited = false;
    bool completion = false;
    // A recorded draw began a render pass of its own, or continued the previous draw's.
    bool passBegun = false;
    bool passContinued = false;
    SyncReason reason = SyncNone;
    // Shader validation memo (see CachedFragmentOutputs).
    bool validateMemoized = false;
    bool validateHit = false;
    // A build that acquired the allocation registry lease (BDA), whose cost sits outside the
    // build's own sub-phases.
    bool addressBased = false;
    // Resident target lookups, and those that took a millisecond or more. Whether a slow one
    // re-uploaded the image or just walked its pages under contention is not visible from here
    // (StorageTexture::Version advances on every MarkDirty as well as on an upload, and Generation
    // is stamped afresh by every write-watch walk): the [texture] line's "reused" and "direct
    // uploads" counters and APS5_TRACE_UPLOAD name the actual uploads.
    std::uint64_t targetLookups = 0;
    std::uint64_t slowLookups = 0;
    double slowLookupUs = 0;
    // The lookups by whether the draw's scissor covers the whole target (an area-scoped proof
    // would change nothing for those), the pages their write-watch collects walked and their time.
    std::uint64_t fullScissorLookups = 0;
    std::uint64_t partialLookups = 0;
    std::uint64_t pagesWalked = 0;
    double lookupUs = 0;
    // How the draw's resources came about (DrawKind), for the per-kind averages.
    std::size_t kind = 0;
    // The flush hook's fence waits made inside this draw (index and vertex reads, the target
    // lookups' flushes, the build's texture lookups): nested in the draw's GpuMutex hold, so they
    // are the part of the hold that is waiting rather than working. The draw's own syncs (a
    // recorded-then-waited draw's recorder Sync, a synchronous draw's SubmitAndWait) are kept apart
    // in ownSyncUs: those wait by design and would otherwise count as nested hook waits.
    double hookWaitUs = 0;
    double ownSyncUs = 0;
};

// The resources of a draw: a recipe hit (reserved for the draw recipe step), a resource-cache
// template hit, a build, an address-based (BDA) build, a BDA template hit, or a hit on a stream
// template (ShaderResources::StreamRing; printed only once one happened).
enum DrawKind : std::size_t { KindRecipeHit, KindTemplateHit, KindBuild, KindBda, KindBdaHit, KindStreamHit, KindCount };
constexpr std::array<const char*, KindCount> DrawKindNames{"recipe hit", "template hit", "build", "BDA", "BDA template hit", "stream template hit"};

struct DrawProfile {
    std::mutex mutex;
    std::array<double, KindCount> kindUs{};
    std::array<std::uint64_t, KindCount> kindCounts{};
    std::uint64_t fullScissorLookups = 0;
    std::uint64_t partialLookups = 0;
    std::uint64_t pagesWalked = 0;
    double lookupUs = 0;
    std::array<double, PhaseCount> totalsUs{};
    // The longest single draw's time per phase, and the longest draw: the [lock] line's 'draw'
    // hold max (tens of ms against an average well under a millisecond) needs a phase name.
    std::array<double, PhaseCount> maxUs{};
    double maxDrawUs = 0;
    double hookWaitUs = 0;
    double maxHookWaitUs = 0;
    double ownSyncUs = 0;
    // The resources build's sub-phases (ShaderResources::Timing) of draws only; dispatches report
    // theirs in the [resources] line. `other` is the rest of the build (prepareAddressBindings:
    // the registry lease and snapshots of an address-based build, the BDA table, the layout
    // lookup), and how much of it address-based builds account for.
    double bindingsUs = 0;
    double uploadUs = 0;
    double descriptorsUs = 0;
    double otherUs = 0;
    double addressOtherUs = 0;
    std::uint64_t addressBuilds = 0;
    std::uint64_t draws = 0;
    std::uint64_t recorded = 0;
    std::uint64_t waited = 0;
    // Recorded draws whose write-back (BDA fault check) runs as a completion action instead of
    // making the draw synchronous (see APS5_SYNC_COMPLETION_DRAWS).
    std::uint64_t completion = 0;
    // Render passes recorded draws began, and draws that continued the previous draw's pass.
    std::uint64_t passesBegun = 0;
    std::uint64_t passesContinued = 0;
    std::array<std::uint64_t, SyncCount> reasons{};
    // The CPU inside Graphics::Draw, split by outcome: a synchronous draw costs ten times a recorded
    // one, so one average would only show the synchronous population.
    double recordedUs = 0;
    double waitedUs = 0;
    double synchronousUs = 0;
    std::uint64_t targetLookups = 0;
    std::uint64_t slowLookups = 0;
    double slowLookupUs = 0;
    // Resource cache outcomes: hits, misses (built and inserted when reusable), entries that failed
    // Revalidate, and draws that could not use the cache (synchronous, no variant id, or disabled).
    // Which key words the misses differ in is the cache's own "[rescache] miss churn" line
    // (ResourceCache::noteMiss compares each miss with the last key of the same variants).
    std::uint64_t cacheHits = 0;
    std::uint64_t cacheMisses = 0;
    std::uint64_t cacheInvalidated = 0;
    std::uint64_t uncacheable = 0;
    // Stream templates (the [reuse] line): valid ones whose moved check refused the hit (the draw
    // built), and ones dropped when found with the switch off.
    std::uint64_t streamRefused = 0;
    std::uint64_t streamDropped = 0;
    // Shader validation memo (see CachedFragmentOutputs).
    std::uint64_t validateHits = 0;
    std::uint64_t validateMisses = 0;
    // Draw recipe outcomes (DrawWithRecipe): hits, and misses by DrawRecipeMiss.
    std::uint64_t recipeHits = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawRecipeMiss::Count)> recipeMisses{};
    // Indirect draws by path (IndirectDrawPath), the GPU-side ones whose records were rewritten
    // with a constant, and the time of the CPU record reads (their syncs).
    std::array<std::uint64_t, IndirectPathCount> indirect{};
    std::uint64_t indirectRewritten = 0;
    double indirectReadUs = 0;
    // Draw packets that drew nothing, by DrawSkip, and their time.
    std::array<std::uint64_t, static_cast<std::size_t>(DrawSkip::Count)> skips{};
    std::array<double, static_cast<std::size_t>(DrawSkip::Count)> skipUs{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};
constexpr std::array<const char*, static_cast<std::size_t>(DrawSkip::Count)> DrawSkipNames{"nothing to draw", "prechecked", "thrown", "pipeline compiling"};

DrawProfile& Profile() {
    static DrawProfile profile;
    return profile;
}

constexpr std::size_t SpeculationMissCount = static_cast<std::size_t>(ShaderResources::SpeculationMiss::Count);
constexpr std::array<const char*, SpeculationMissCount> SpeculationMissNames{"none", "uncacheable", "no entry", "image words", "record", "collect", "depth", "keys", "evicted", "pending", "changed", "moved", "space", "outside", "set", "thrown", "device", "identity", "version", "moved recheck", "shared", "finish", "tail", "no recorder", "moved image"};

struct SpeculationStats {
    std::atomic<std::uint64_t> address{0};
    std::atomic<std::uint64_t> templates{0};
    std::atomic<std::uint64_t> acceptedAddress{0};
    std::atomic<std::uint64_t> acceptedTemplate{0};
    std::array<std::atomic<std::uint64_t>, SpeculationMissCount> prep{};
    std::array<std::atomic<std::uint64_t>, SpeculationMissCount> commit{};
};

SpeculationStats& Speculations() {
    static SpeculationStats stats;
    return stats;
}

void reportSpeculation() {
    auto& stats = Speculations();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    std::string line = "[spec] speculated address " + std::to_string(take(stats.address)) + " template " + std::to_string(take(stats.templates)) + "; accepted address " + std::to_string(take(stats.acceptedAddress)) + " template " + std::to_string(take(stats.acceptedTemplate)) + "; refused on prep:";
    for (std::size_t i = 1; i < SpeculationMissCount; ++i) {
        if (const auto count = take(stats.prep[i]); count != 0) line += " " + std::string(SpeculationMissNames[i]) + " " + std::to_string(count);
    }
    line += "; refused on commit:";
    for (std::size_t i = 1; i < SpeculationMissCount; ++i) {
        if (const auto count = take(stats.commit[i]); count != 0) line += " " + std::string(SpeculationMissNames[i]) + " " + std::to_string(count);
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

constexpr std::size_t TokenFallbackCount = static_cast<std::size_t>(ShaderResources::TokenFallback::Count);
constexpr std::array<const char*, TokenFallbackCount> TokenFallbackNames{"fast", "no tokens", "epoch", "unwatch serial", "allocations", "pending serial", "publish generation", "window overlap", "window unknown", "window overflow"};

struct TokenStats {
    std::array<std::array<std::atomic<std::uint64_t>, TokenFallbackCount>, 2> outcomes{};
    std::array<std::atomic<std::uint64_t>, 3> draws{};
    std::array<std::atomic<std::uint64_t>, 3> drawNs{};
    std::atomic<std::uint64_t> inPlaceChecks{0};
    std::atomic<std::uint64_t> inPlaceNs{0};
    std::atomic<std::uint64_t> indexScans{0};
    std::atomic<std::uint64_t> indexScanNs{0};
};

TokenStats& Tokens() {
    static TokenStats stats;
    return stats;
}

thread_local std::uint8_t drawTokenState = 0;

void reportTokens() {
    auto& stats = Tokens();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto perCount = [](unsigned long long ns, unsigned long long count) { return count != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(count) : 0.0; };
    std::string line = "[tokens] (10 s)";
    constexpr std::array<const char*, 2> families{" proof:", "; moved:"};
    for (std::size_t family = 0; family < families.size(); ++family) {
        line += families[family];
        for (std::size_t i = 0; i < TokenFallbackCount; ++i) {
            if (const auto count = take(stats.outcomes[family][i]); count != 0 || i == 0) line += " " + std::string(TokenFallbackNames[i]) + " " + std::to_string(count);
        }
    }
    char tail[384];
    const auto accepted = take(stats.draws[1]);
    const auto fallback = take(stats.draws[2]);
    const auto untried = take(stats.draws[0]);
    const auto checks = take(stats.inPlaceChecks);
    const auto scans = take(stats.indexScans);
    std::snprintf(tail, sizeof(tail), "; committer us/draw: accepted %.1f (%llu), fallback %.1f (%llu), untried %.1f (%llu); inputs: in-place checks %llu at %.2f us, index scans %llu at %.2f us; targets: cross-epoch proofs %llu", perCount(take(stats.drawNs[1]), accepted), accepted, perCount(take(stats.drawNs[2]), fallback), fallback, perCount(take(stats.drawNs[0]), untried), untried, checks, perCount(take(stats.inPlaceNs), checks), scans, perCount(take(stats.indexScanNs), scans), static_cast<unsigned long long>(TakeCrossEpochTargetProofs()));
    line += tail;
    std::fprintf(stderr, "%s\n", line.c_str());
}

bool tokenWindowEnabled() {
    static const bool enabled = std::getenv("APS5_NO_TOKEN_WINDOW") == nullptr;
    return LiveFlag("token_window", enabled);
}

AppliedWindow& appliedWindow() {
    static AppliedWindow window;
    return window;
}

// Adds one draw's phases to the totals and prints the [draws] and [rescache] lines every 10 s.
void reportDraw(const std::array<double, PhaseCount>& us, const ShaderResources::BuildTiming* built, const DrawOutcome& outcome) {
    auto& profile = Profile();
    std::lock_guard lock(profile.mutex);
    double drawUs = 0;
    for (std::size_t i = 0; i < PhaseCount; ++i) {
        profile.totalsUs[i] += us[i];
        profile.maxUs[i] = std::max(profile.maxUs[i], us[i]);
        drawUs += us[i];
    }
    profile.maxDrawUs = std::max(profile.maxDrawUs, drawUs);
    profile.hookWaitUs += outcome.hookWaitUs;
    profile.maxHookWaitUs = std::max(profile.maxHookWaitUs, outcome.hookWaitUs);
    profile.ownSyncUs += outcome.ownSyncUs;
    if (built != nullptr) {
        profile.bindingsUs += built->bindingsMs * 1000.0;
        profile.uploadUs += built->uploadMs * 1000.0;
        profile.descriptorsUs += built->descriptorsMs * 1000.0;
        const auto other = std::max(0.0, us[PhaseResources] - (built->bindingsMs + built->uploadMs + built->descriptorsMs) * 1000.0);
        profile.otherUs += other;
        if (outcome.addressBased) {
            profile.addressOtherUs += other;
            ++profile.addressBuilds;
        }
    }
    ++profile.draws;
    if (outcome.recorded) ++(outcome.waited ? profile.waited : profile.recorded);
    if (outcome.completion) ++profile.completion;
    if (outcome.passBegun) ++profile.passesBegun;
    if (outcome.passContinued) ++profile.passesContinued;
    ++profile.reasons[outcome.reason];
    (outcome.recorded ? (outcome.waited ? profile.waitedUs : profile.recordedUs) : profile.synchronousUs) += drawUs;
    profile.targetLookups += outcome.targetLookups;
    profile.slowLookups += outcome.slowLookups;
    profile.slowLookupUs += outcome.slowLookupUs;
    profile.fullScissorLookups += outcome.fullScissorLookups;
    profile.partialLookups += outcome.partialLookups;
    profile.pagesWalked += outcome.pagesWalked;
    profile.lookupUs += outcome.lookupUs;
    if (outcome.kind < KindCount) {
        ++profile.kindCounts[outcome.kind];
        profile.kindUs[outcome.kind] += drawUs;
    }
    if (outcome.validateMemoized) ++(outcome.validateHit ? profile.validateHits : profile.validateMisses);
    const auto now = std::chrono::steady_clock::now();
    if (now - profile.lastReport < std::chrono::seconds(10)) return;
    profile.lastReport = now;
    const auto synchronous = profile.draws - profile.recorded - profile.waited;
    const auto average = [](double total, std::uint64_t count) { return count != 0 ? total / static_cast<double>(count) : 0.0; };
    char line[2048];
    int n = std::snprintf(line, sizeof(line), "[draws] %llu draws over 10 s (%llu recorded avg %.0f us, of them %llu with completion; %llu recorded then waited avg %.0f us; %llu synchronous avg %.0f us; render passes %llu begun, %llu draws continued one; waited or synchronous because:", static_cast<unsigned long long>(profile.draws), static_cast<unsigned long long>(profile.recorded), average(profile.recordedUs, profile.recorded), static_cast<unsigned long long>(profile.completion), static_cast<unsigned long long>(profile.waited), average(profile.waitedUs, profile.waited), static_cast<unsigned long long>(synchronous), average(profile.synchronousUs, synchronous), static_cast<unsigned long long>(profile.passesBegun), static_cast<unsigned long long>(profile.passesContinued));
    const auto room = [&] { return n > 0 && static_cast<std::size_t>(n) < sizeof(line); };
    for (std::size_t i = SyncNone + 1; i < SyncCount && room(); ++i) {
        if (profile.reasons[i] != 0) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", SyncReasonNames[i], static_cast<unsigned long long>(profile.reasons[i]));
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "):");
    for (std::size_t i = 0; i < PhaseCount && room(); ++i) {
        if (profile.totalsUs[i] <= 0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s=%.1fms", DrawPhaseNames[i], profile.totalsUs[i] / 1000.0);
        if (i == PhaseResources && room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " (bindings %.1f, upload %.1f, descriptors %.1f, other %.1f of which %.1f in %llu address-based builds)", profile.bindingsUs / 1000.0, profile.uploadUs / 1000.0, profile.descriptorsUs / 1000.0, profile.otherUs / 1000.0, profile.addressOtherUs / 1000.0, static_cast<unsigned long long>(profile.addressBuilds));
        if (i == PhaseReadTarget && room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " (%llu resident lookups, %llu of them >= 1 ms = %.1f; target lookups: full-scissor %llu / partial %llu / pages walked %llu / %.0f us)", static_cast<unsigned long long>(profile.targetLookups), static_cast<unsigned long long>(profile.slowLookups), profile.slowLookupUs / 1000.0, static_cast<unsigned long long>(profile.fullScissorLookups), static_cast<unsigned long long>(profile.partialLookups), static_cast<unsigned long long>(profile.pagesWalked), profile.lookupUs);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; per kind (avg us, count):");
    for (std::size_t kind = 0; kind < KindCount && room(); ++kind) {
        // Without a stream hit the line stays as it was before stream templates existed.
        if (kind == KindStreamHit && profile.kindCounts[kind] == 0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %.0f (%llu)", DrawKindNames[kind], average(profile.kindUs[kind], profile.kindCounts[kind]), static_cast<unsigned long long>(profile.kindCounts[kind]));
    }
    // The longest draw and the longest single phase of any draw (which phase a long 'draw' hold
    // was), plus the hook waits nested inside the draws.
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; longest draw %.1f ms, longest phase of any draw:", profile.maxDrawUs / 1000.0);
    for (std::size_t i = 0; i < PhaseCount && room(); ++i) {
        if (profile.maxUs[i] < 500.0) continue;
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %.1f", DrawPhaseNames[i], profile.maxUs[i] / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; hook waits inside draws %.1f ms (max %.1f; the draws' own syncs, %.1f ms, not counted)", profile.hookWaitUs / 1000.0, profile.maxHookWaitUs / 1000.0, profile.ownSyncUs / 1000.0);
    std::uint64_t indirectCpu = 0;
    for (std::size_t i = 1; i < IndirectPathCount; ++i) indirectCpu += profile.indirect[i];
    if ((profile.indirect[0] != 0 || indirectCpu != 0) && room()) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; indirect: gpu-side %llu (%llu rewritten), cpu-side %llu (", static_cast<unsigned long long>(profile.indirect[0]), static_cast<unsigned long long>(profile.indirectRewritten), static_cast<unsigned long long>(indirectCpu));
        for (std::size_t i = 1; i < IndirectPathCount && room(); ++i) {
            if (profile.indirect[i] != 0) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", IndirectDrawPathNames[i], static_cast<unsigned long long>(profile.indirect[i]));
        }
        if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "), argument reads %.1f ms", profile.indirectReadUs / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; skipped packets:");
    for (std::size_t i = 0; i < profile.skips.size() && room(); ++i) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu in %.1f ms", DrawSkipNames[i], static_cast<unsigned long long>(profile.skips[i]), profile.skipUs[i] / 1000.0);
    }
    if (room()) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), "; recipe hits %llu, misses by reason:", static_cast<unsigned long long>(profile.recipeHits));
    for (std::size_t i = 1; i < profile.recipeMisses.size() && room(); ++i) {
        n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", DrawRecipeMissName(static_cast<DrawRecipeMiss>(i)), static_cast<unsigned long long>(profile.recipeMisses[i]));
    }
    std::fprintf(stderr, "%s\n", line);
    const auto inputMemo = DrawInputMemoCounters();
    if (inputMemo.hits != 0 || inputMemo.misses != 0)
        std::fprintf(stderr, "[draw-input] cumulative memo hits %llu, fallbacks %llu\n", static_cast<unsigned long long>(inputMemo.hits), static_cast<unsigned long long>(inputMemo.misses));
    std::fprintf(stderr, "[rescache] draws: %llu hits, %llu misses, %llu invalidated, %llu uncacheable; validation memo %llu hits / %llu misses (which key words the misses differ in: the miss churn line)\n", static_cast<unsigned long long>(profile.cacheHits), static_cast<unsigned long long>(profile.cacheMisses), static_cast<unsigned long long>(profile.cacheInvalidated), static_cast<unsigned long long>(profile.uncacheable), static_cast<unsigned long long>(profile.validateHits), static_cast<unsigned long long>(profile.validateMisses));
    if (ShaderResources::AddressTemplates()) {
        const auto address = ShaderResources::AddressTemplateCounters();
        using Miss = ShaderResources::AddressMiss;
        const auto missed = [&](Miss reason) { return static_cast<unsigned long long>(address.misses[static_cast<std::size_t>(reason)]); };
        const auto perHit = [&](double us) { return address.timedHits != 0 ? us / static_cast<double>(address.timedHits) : 0.0; };
        std::fprintf(stderr, "[rescache] address templates (cumulative): %llu hits (%llu moved guest buffers rebound), %llu inserted; misses: space replaced %llu, imports %llu, proof %llu, moved buffers %llu, moved outside the space %llu, snapshot outside %llu; us per hit: moved check %.1f, space %.1f, proof %.1f, finish %.1f, descriptor set %.1f\n", static_cast<unsigned long long>(address.hits), static_cast<unsigned long long>(address.movedBuffers), static_cast<unsigned long long>(address.detached), missed(Miss::Replaced), missed(Miss::Imports), missed(Miss::Proof), missed(Miss::Moved), missed(Miss::MovedOutside), missed(Miss::Outside), perHit(address.movedUs), perHit(address.spaceUs), perHit(address.proofUs), perHit(address.finishUs), perHit(address.setUs));
    }
    {
        // Draw builds by noteReusable's verdict over the last 10 s (see ShaderResources::ReuseStats):
        // with the stream_ring switch off, 'streamable' over 'draw builds' is the share of builds
        // stream templates would serve; with it on, the stream hits and their copies.
        static ShaderResources::ReuseStats last;
        const auto now = ShaderResources::ReuseCounters();
        const auto delta = [](std::uint64_t current, std::uint64_t previous) { return static_cast<unsigned long long>(current - previous); };
        constexpr std::array<const char*, static_cast<std::size_t>(GuestBufferMemory::StreamRefusal::Count)> refusalNames{"other", "written", "over limit", "sparse", "writable mirror", "snapshot", "staged", "address space"};
        std::string refusals;
        for (std::size_t index = 0; index < refusalNames.size(); ++index) {
            if (const auto count = delta(now.copyRefused[index], last.copyRefused[index]); count != 0) refusals += " " + std::string(refusalNames[index]) + " " + std::to_string(count);
        }
        const auto builds = delta(now.builds, last.builds);
        const auto streamable = delta(now.streamableOff, last.streamableOff) + delta(now.streamMade, last.streamMade);
        std::fprintf(stderr, "[reuse] (10 s) %llu draw builds: in place %llu, stream templates made %llu, streamable with the switch off %llu (%.0f%% of builds streamable); refused: completion or lease %llu, per-use texture %llu, data over 64 KiB %llu, copied region:%s%s, straddled %llu, no serial %llu; copied %llu ranges (%.1f KiB); stream hits %llu, refused at hit %llu, dropped by the switch %llu; streamed %llu ranges (%.1f KiB), verified %llu\n",
            builds, delta(now.inPlace, last.inPlace), delta(now.streamMade, last.streamMade), delta(now.streamableOff, last.streamableOff), builds != 0 ? 100.0 * static_cast<double>(streamable) / static_cast<double>(builds) : 0.0,
            delta(now.completion, last.completion), delta(now.perUseTexture, last.perUseTexture), delta(now.largeData, last.largeData), refusals.empty() ? " none" : "", refusals.c_str(), delta(now.straddled, last.straddled), delta(now.noSerial, last.noSerial),
            delta(now.copiedRanges, last.copiedRanges), static_cast<double>(now.copiedBytes - last.copiedBytes) / 1024.0, static_cast<unsigned long long>(profile.kindCounts[KindStreamHit]), static_cast<unsigned long long>(profile.streamRefused), static_cast<unsigned long long>(profile.streamDropped),
            delta(now.streamedRanges, last.streamedRanges), static_cast<double>(now.streamedBytes - last.streamedBytes) / 1024.0, delta(now.verified, last.verified));
        last = now;
    }
    if (DrawSpeculationEnabled()) reportSpeculation();
    if (DrawTokensEnabled() || InputTokensEnabled() || TargetTokensEnabled()) reportTokens();
    profile.totalsUs.fill(0);
    profile.maxUs.fill(0);
    profile.maxDrawUs = profile.hookWaitUs = profile.maxHookWaitUs = profile.ownSyncUs = 0;
    profile.bindingsUs = profile.uploadUs = profile.descriptorsUs = profile.otherUs = profile.addressOtherUs = 0;
    profile.addressBuilds = 0;
    profile.draws = profile.recorded = profile.waited = profile.completion = 0;
    profile.passesBegun = profile.passesContinued = 0;
    profile.reasons.fill(0);
    profile.recordedUs = profile.waitedUs = profile.synchronousUs = 0;
    profile.targetLookups = profile.slowLookups = 0;
    profile.slowLookupUs = 0;
    profile.fullScissorLookups = profile.partialLookups = profile.pagesWalked = 0;
    profile.lookupUs = 0;
    profile.kindUs.fill(0);
    profile.kindCounts.fill(0);
    profile.cacheHits = profile.cacheMisses = profile.cacheInvalidated = profile.uncacheable = 0;
    profile.streamRefused = profile.streamDropped = 0;
    profile.validateHits = profile.validateMisses = 0;
    profile.recipeHits = 0;
    profile.recipeMisses.fill(0);
    profile.indirect.fill(0);
    profile.indirectRewritten = 0;
    profile.indirectReadUs = 0;
    profile.skips.fill(0);
    profile.skipUs.fill(0);
}

void countCache(std::uint64_t DrawProfile::*counter) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    if (auto* log = ForkLogged()) {
        log->Defer(ForkLog::Effect::Counter, 0, 0, [counter] { countCache(counter); });
        return;
    }
    if (auto* witness = ForkWitnessed()) witness->Note(ForkLog::Effect::Counter, 0, 0);
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++(stats.*counter);
}

// ValidateShaders decodes every SPIR-V instruction of every stage on every draw. Its outcome depends
// only on the stages' compiled variants (a variant id names identical SPIR-V and binding layout),
// their push constant placement and vertex attribute shapes, and the state fields it checks, so the
// fragment output set is remembered by exactly those. Nothing here is keyed by pointer: the results
// a draw hands in live in a per-draw vector. A stage without a variant id is validated as before,
// except the rect-list control and evaluation stages, which are generated from the vertex and
// fragment results the key already names (the pipeline key treats them the same way). `memoized`
// says whether the memo applied, `hit` whether it answered. APS5_NO_VALIDATE_CACHE=1 validates
// every draw.
bool ValidationKey(const Context& context, std::span<const CompiledShader> shaders, const State& state, std::vector<std::uint64_t>& key) {
    using Stage = ShaderRecompiler::ShaderStage;
    static const bool disabled = std::getenv("APS5_NO_VALIDATE_CACHE") != nullptr;
    const auto add = [&](auto value) { key.push_back(static_cast<std::uint64_t>(value)); };
    bool keyed = !disabled;
    if (keyed) {
        key.reserve(40 + shaders.size() * 12);
        add(shaders.size());
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            const auto& program = *shader.program;
            const bool generated = state.rectList && (shader.stage == Stage::TessellationControl || shader.stage == Stage::TessellationEvaluation);
            if (!generated && program.PipelineVariantId() == 0) {
                keyed = false;
                break;
            }
            add(shader.stage);
            add(generated ? std::uint64_t{0} : program.PipelineVariantId());
            add(shader.pushConstantOffset);
            add(program.pushConstants.size());
            add(program.bdaAbiVersion);
            add(program.vertexAttributes.size());
            for (const auto& attribute : program.vertexAttributes) {
                add(attribute.location);
                add(attribute.components);
                add(attribute.fetchIndex);
                add(attribute.formatComponents);
                // The data format bits of the V# decide the attribute's signature.
                add((attribute.resource.fields[3] >> 12u) & 0x7fu);
            }
        }
    }
    if (keyed) {
        add(state.stages.path);
        add(state.rectList);
        add(state.topology);
        add(state.cullMode);
        add(state.blends.size());
        add(context.subgroup.subgroupSize);
        add(context.subgroup.supportedStages);
        add(context.subgroup.supportedOperations);
        add(context.fragmentShaderBarycentric);
        add(state.stages.mesh.has_value());
        if (state.stages.mesh) {
            const auto& mesh = *state.stages.mesh;
            add(mesh.inputPrimitive);
            add(mesh.primitivesPerGroup);
            add(mesh.verticesPerGroup);
            add(mesh.maxVertices);
            add(mesh.maxPrimitives);
            add(mesh.threadsPerGroup);
            add(mesh.ldsSizeDwords);
            add(mesh.provokingVertex);
        }
        add(state.stages.tessellation.has_value());
        if (state.stages.tessellation) {
            const auto& tessellation = *state.stages.tessellation;
            add(tessellation.inputControlPoints);
            add(tessellation.outputControlPoints);
            add(tessellation.domain);
            add(tessellation.partitioning);
            add(tessellation.outputTopology);
        }
    }
    return keyed;
}

std::mutex& validationMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::vector<std::uint64_t>, std::string>& validationFailures() {
    static std::map<std::vector<std::uint64_t>, std::string> failures;
    return failures;
}

std::set<std::uint32_t> CachedFragmentOutputs(const Context& context, std::span<const CompiledShader> shaders, const State& state, bool& memoized, bool& hit) {
    memoized = false;
    hit = false;
    std::vector<std::uint64_t> key;
    const bool keyed = ValidationKey(context, shaders, state, key);
    static std::map<std::vector<std::uint64_t>, std::set<std::uint32_t>> memo;
    if (keyed) {
        memoized = true;
        std::lock_guard lock(validationMutex());
        if (const auto found = memo.find(key); found != memo.end()) {
            hit = true;
            return found->second;
        }
    }
    std::set<std::uint32_t> outputs;
    try {
        outputs = ValidateShaders(shaders, state, context.subgroup, context.fragmentShaderBarycentric, context.descriptorIndexing, context.imageInt64Atomics, context.geometryShader, context.sampleRateShading, context.viewportIndexLayer);
    } catch (const std::exception& error) {
        if (keyed) {
            std::lock_guard lock(validationMutex());
            auto& failures = validationFailures();
            if (failures.size() >= 1024) failures.clear();
            failures.emplace(std::move(key), error.what());
        }
        throw;
    }
    if (keyed) {
        std::lock_guard lock(validationMutex());
        // A handful of configurations recur; a runaway key space is dropped wholesale.
        if (memo.size() >= 1024) memo.clear();
        memo.emplace(std::move(key), outputs);
    }
    return outputs;
}

// The resource cache key of a recorded draw: a marker no compute key starts with (those begin with
// the stage), the device handle (the cache is process-wide and the driver replaces the headless
// device with the windowed one while workers may still use the old one: an entry's descriptor set
// and pooled buffers belong to the device that built it) and every stage's content key
// (ResourceCache::noteMiss walks this layout to attribute misses, so it changes together with it).
// With `ranges`, also the render target and the index buffer the build's alias checks compared the
// guest buffers against; without it the checks are repeated for a hit (CheckBufferAliases), since
// neither range is part of the descriptor set (the target is attached, the index buffer copied per
// draw), so a target or index ring that moves per frame does not miss on every draw. That guards a
// workload with such rings; in the profiled menu stage every draw was DRAW_INDEX_AUTO (index range
// 0) onto one fixed target, so its misses come from the stages' descriptor words themselves.
bool MovableBuffers() {
    static const bool enabled = std::getenv("APS5_NO_MOVED_BUFFER_TEMPLATES") == nullptr;
    return enabled;
}

ResourceCache::Key DrawResourceKey(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::uint64_t indexBytes, bool ranges) {
    ResourceCache::Key key{0xffffffffu};
    const auto append64 = [&](std::uint64_t value) {
        key.push_back(static_cast<std::uint32_t>(value));
        key.push_back(static_cast<std::uint32_t>(value >> 32u));
    };
    append64(reinterpret_cast<std::uint64_t>(context.device));
    key.push_back(static_cast<std::uint32_t>(shaders.size()));
    const bool movableImages = ShaderResources::MovableImages() && std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaAbiVersion != 0; });
    key.push_back(movableImages ? 1u : 0u);
    for (const auto& shader : shaders) {
        const auto part = ShaderResources::ContentKey(shader, true, MovableBuffers(), movableImages);
        key.push_back(static_cast<std::uint32_t>(part.size()));
        key.insert(key.end(), part.begin(), part.end());
    }
    if (ranges) {
        append64(target.address);
        append64(target.bytes);
        append64(indexAddress);
        append64(indexBytes);
    }
    return key;
}

// The alias checks the build makes for every guest buffer descriptor (ShaderResources::
// addGuestBuffer), repeated for a cached build whose own checks compared the buffers against
// another draw's render target and index buffer. The same conditions and messages, so a draw that
// would have failed its build fails its hit.
void CheckBufferAliases(std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::uint64_t indexBytes) {
    const auto overlap = [](std::uint64_t first, std::uint64_t firstSize, std::uint64_t second, std::uint64_t secondSize) { return first < second + secondSize && second < first + firstSize; };
    for (const auto& shader : shaders) {
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto& words = binding.guestDescriptor;
            for (std::size_t offset = 0; offset + 4 <= words.size(); offset += 4) {
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[offset], words[offset + 1], words[offset + 2], words[offset + 3]}};
                const auto address = descriptor.Base48();
                const auto size = descriptor.GetSize();
                if (size == 0 || address == 0) continue;
                const auto element = offset / 4;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                Require(!overlap(address, size, target.address, target.bytes), "shader buffer aliases the render target");
                Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
            }
        }
    }
}

}

bool DrawSpeculationEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_DRAW_SPECULATION");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("spec", enabled);
}

bool VerifyDrawSpeculation() {
    static const bool enabled = std::getenv("APS5_VERIFY_DRAW_SPECULATION") != nullptr;
    return enabled;
}

void CountSpeculationMiss(DrawSpeculation::Miss miss, bool committer) {
    const auto index = static_cast<std::size_t>(miss);
    if (index == 0 || index >= SpeculationMissCount) return;
    (committer ? Speculations().commit : Speculations().prep)[index].fetch_add(1, std::memory_order_relaxed);
}

bool DrawTokensEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_DRAW_TOKENS");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("tokens", enabled);
}

bool TargetTokensEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_TARGET_TOKENS");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("target_tokens", enabled);
}

bool InputTokensEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_INPUT_TOKENS");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("input_tokens", enabled);
}

bool VerifyDrawTokens() {
    static const bool enabled = std::getenv("APS5_VERIFY_DRAW_TOKENS") != nullptr;
    return enabled;
}

void SetAppliedWindow(AppliedWindow window) {
    appliedWindow() = window;
}

void CountTokenOutcome(TokenFamily family, ShaderResources::TokenFallback outcome) {
    const auto index = static_cast<std::size_t>(outcome);
    if (index >= TokenFallbackCount) return;
    Tokens().outcomes[static_cast<std::size_t>(family)][index].fetch_add(1, std::memory_order_relaxed);
    if (outcome != ShaderResources::TokenFallback::None) drawTokenState = 2;
    else if (drawTokenState == 0) drawTokenState = 1;
}

void ScreenTokenWindow(DrawSpeculation& speculation) {
    using Fallback = ShaderResources::TokenFallback;
    auto verdict = Fallback::None;
    const auto& window = appliedWindow();
    if (tokenWindowEnabled()) {
        if (!speculation.windowed || window.check == nullptr) {
            verdict = Fallback::WindowOverflow;
        } else {
            std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;
            reads.reserve(speculation.proof.surfaces.size() + speculation.movedProof.ranges.size());
            for (const auto& surface : speculation.proof.surfaces) reads.emplace_back(surface.address, surface.address + surface.bytes);
            reads.insert(reads.end(), speculation.movedProof.ranges.begin(), speculation.movedProof.ranges.end());
            switch (window.check(speculation.frontier, reads)) {
                case WindowVerdict::Clear: break;
                case WindowVerdict::Overlap: verdict = Fallback::WindowOverlap; break;
                case WindowVerdict::Unknown: verdict = Fallback::WindowUnknown; break;
                case WindowVerdict::Overflow: verdict = Fallback::WindowOverflow; break;
            }
        }
    }
    speculation.proof.window = verdict;
    speculation.movedProof.window = verdict;
}

namespace {

thread_local std::shared_ptr<DrawSpeculation> currentSpeculation;

}

SpeculationScope::SpeculationScope(std::shared_ptr<DrawSpeculation> speculation) {
    currentSpeculation = std::move(speculation);
}

SpeculationScope::~SpeculationScope() {
    currentSpeculation.reset();
}

std::shared_ptr<DrawSpeculation> SpeculateDrawResources(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots) {
    using Miss = DrawSpeculation::Miss;
    static const bool noDrawResourceCache = std::getenv("APS5_NO_DRAW_RESOURCE_CACHE") != nullptr;
    static const bool noTextureCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    const auto refuse = [](Miss miss) {
        CountSpeculationMiss(miss, false);
        return std::shared_ptr<DrawSpeculation>{};
    };
    if (noDrawResourceCache || noTextureCache || !std::all_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->PipelineVariantId() != 0; })) return refuse(Miss::Uncacheable);
    try {
        auto speculation = std::make_shared<DrawSpeculation>();
        speculation->device = context.device;
        if (const auto& window = appliedWindow(); DrawTokensEnabled() && window.frontier != nullptr) {
            speculation->frontier = window.frontier();
            speculation->windowed = true;
        }
        const auto indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
        speculation->key = DrawResourceKey(context, shaders, state.color, draw.indexAddress, indexBytes, !trimKey);
        auto cached = SharedResourceCache().Peek(speculation->key);
        if (cached == nullptr) return refuse(Miss::NoEntry);
        speculation->aliasesOk = true;
        if (trimKey) {
            try {
                CheckBufferAliases(shaders, state.color, draw.indexAddress, indexBytes);
            } catch (const std::exception&) {
                speculation->aliasesOk = false;
            }
        }
        const bool made = cached->AddressReusable() ? ShaderResources::SpeculateAddressUse(cached, shaders, snapshots, *speculation) : cached->SpeculateTemplate(shaders, *speculation);
        if (!made) return refuse(speculation->miss);
        speculation->source = std::move(cached);
        (speculation->kind == DrawSpeculation::Kind::Address ? Speculations().address : Speculations().templates).fetch_add(1, std::memory_order_relaxed);
        return speculation;
    } catch (const std::exception&) {
        return refuse(Miss::Thrown);
    }
}

std::array<std::uint32_t, 4> MeshIndexBufferDescriptor(const Pm4::DrawParameters& draw) {
    constexpr std::uint32_t RawWord3 = 0x31016facu;
    if (!draw.indexed) return {0u, 0u, 0u, RawWord3};
    const auto address = draw.indexAddress;
    const auto bytes = (static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize + 3u) & ~std::uint64_t{3};
    Require(address != 0 && bytes != 0 && bytes <= 0xffffffffu && (address >> 48u) == 0, "invalid mesh index buffer range");
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, static_cast<std::uint32_t>(bytes), RawWord3};
}

MeshArgumentRules MeshArgumentRulesFor(const Context& context, const ShaderRecompiler::MeshConfiguration& mesh, std::uint32_t indexCount) {
    const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
    const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
    Require(mesh.primitivesPerGroup != 0, "mesh draw contains no complete primitive");
    return {indexCount, inputSize, step, mesh.primitivesPerGroup, context.meshLimits.maxMeshWorkGroupCount[0], context.meshLimits.maxMeshWorkGroupCount[1], context.meshLimits.maxMeshWorkGroupTotalCount};
}

MeshArguments ResolveMeshArguments(const Pm4::DrawArguments& record, const MeshArgumentRules& rules) {
    const auto first = record.firstVertexOrIndex;
    const auto effective = first < rules.indexCount ? std::min(record.count, rules.indexCount - first) : 0u;
    std::uint32_t groups = 0;
    if (effective >= rules.inputSize && record.instances != 0) {
        groups = (effective - rules.inputSize) / rules.step / rules.primitivesPerGroup + 1u;
        if (groups > rules.maxGroups || record.instances > rules.maxInstances || groups > rules.maxTotal / record.instances) groups = 0;
    }
    return {groups, groups != 0 ? record.instances : 0u, groups != 0 ? 1u : 0u, effective, first};
}

std::shared_ptr<std::vector<std::shared_ptr<ShaderResources>>> DrawCopiedWriters() {
    // Never destroyed: a completion action of a batch still in flight at static teardown may erase from it.
    static auto* const writers = new std::shared_ptr<std::vector<std::shared_ptr<ShaderResources>>>(std::make_shared<std::vector<std::shared_ptr<ShaderResources>>>());
    return *writers;
}

const char* IndirectDrawPathName(IndirectDrawPath path) {
    return IndirectDrawPathNames[static_cast<std::size_t>(path)];
}

void CountIndirectDraw(IndirectDrawPath path, double readMs, bool rewritten) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++stats.indirect[static_cast<std::size_t>(path)];
    if (rewritten) ++stats.indirectRewritten;
    stats.indirectReadUs += readMs * 1000.0;
}

void CountDrawSkip(DrawSkip kind, double us) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = Profile();
    std::lock_guard lock(stats.mutex);
    ++stats.skips[static_cast<std::size_t>(kind)];
    stats.skipUs[static_cast<std::size_t>(kind)] += us;
}


bool DrawRecipes() {
    static const bool noDrawRecipe = std::getenv("APS5_NO_DRAW_RECIPE") != nullptr;
    return !noDrawRecipe;
}

const char* DrawRecipeMissName(DrawRecipeMiss miss) {
    constexpr std::array<const char*, static_cast<std::size_t>(DrawRecipeMiss::Count)> names{"none", "not recordable", "target gone", "template gone", "objects gone", "proof", "depth gone", "moved refused"};
    return names[static_cast<std::size_t>(miss)];
}

// Depth recipes: a draw with a depth target gets a recipe that stores the depth attachment view;
// a hit calls DepthSurfaceView as the ordinary path does (the per-draw depth work: the pending
// fast clear, the storage-image writes taken back, the generation bump) and misses unless it hands
// back the stored view. APS5_NO_DEPTH_RECIPE=1 builds no recipe for such draws, as before.
bool DepthDrawRecipes() {
    static const bool noDepthRecipe = std::getenv("APS5_NO_DEPTH_RECIPE") != nullptr;
    return !noDepthRecipe;
}

// Moved-buffer recipes: a draw whose template hit rebound moved read-only buffers gets a recipe
// whose hits recompute them (ShaderResources::MovedReadOnlyBuffers, as the template hit does) and
// miss when that refuses. APS5_NO_MOVED_RECIPE=1 builds no recipe for such draws, as before.
bool MovedDrawRecipes() {
    static const bool noMovedRecipe = std::getenv("APS5_NO_MOVED_RECIPE") != nullptr;
    return !noMovedRecipe;
}

namespace {

// APS5_VERIFY_DRAW_RECIPE=1 (or APS5_VERIFY_RECIPE=1): every recipe hit also asks the ordinary
// path's lookups: DepthSurfaceView again for the depth view, the moved buffers of a recipe built
// without any, the resource cache for the template and the pipeline store for the pipeline. A
// different depth view or moved-buffer answer aborts; another live template or pipeline is counted
// (the [recipe] line's "object replaced") and its first occurrences printed.
bool VerifyDrawRecipeObjects() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_RECIPE") != nullptr || std::getenv("APS5_VERIFY_RECIPE") != nullptr;
    return verify;
}

// Whether a recorded draw of this state and moved-buffer list may get a recipe.
bool RecipeEligibleTargets(const State& state, std::span<const ShaderResources::MovedBuffer> moved) {
    return (!state.depth || DepthDrawRecipes()) && (moved.empty() || MovedDrawRecipes());
}

}

namespace {

// APS5_PROFILE_DRAW: the per-phase timers of one draw (DrawPhase), shared by Draw's parts and
// DrawWithRecipe.
struct DrawTimer {
    bool profile;
    std::chrono::steady_clock::time_point phaseStart;
    std::array<double, PhaseCount> us{};
    explicit DrawTimer(bool profile) : profile(profile), phaseStart(std::chrono::steady_clock::now()) {}
    void phase(DrawPhase which) {
        if (!profile) return;
        const auto now = std::chrono::steady_clock::now();
        us[which] += std::chrono::duration<double, std::micro>(now - phaseStart).count();
        phaseStart = now;
    }
};

// One line per draw with the phases that took time (APS5_TRACE_DRAWS), and the 10 s totals.
// `waitedBefore` is the thread's GPU waits at the draw's start and `ownWaitedMs` the draw's own
// syncs (see DrawOutcome::hookWaitUs).
void reportDrawEnd(const State& state, const DrawTimer& timer, const ShaderResources::BuildTiming* built, DrawOutcome& outcome, double waitedBefore, double ownWaitedMs, const char* suffix) {
    if (!timer.profile) return;
    static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    outcome.hookWaitUs = std::max(0.0, Recorder::ThreadWaitedMs() - waitedBefore - ownWaitedMs) * 1000.0;
    outcome.ownSyncUs = ownWaitedMs * 1000.0;
    if (traceDraws) {
        char line[512];
        int n = std::snprintf(line, sizeof(line), "[draw] %ux%u %zu targets%s:", state.renderExtent.width, state.renderExtent.height, state.colors.size(), suffix);
        for (std::size_t i = 0; i < PhaseCount && n > 0 && static_cast<std::size_t>(n) < sizeof(line); ++i) {
            if (timer.us[i] <= 0) continue;
            n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s=%.0fus", DrawPhaseNames[i], timer.us[i]);
        }
        std::fprintf(stderr, "%s\n", line);
    }
    reportDraw(timer.us, built, outcome);
}

}

namespace {

bool drawInputMemo() {
    static const bool enabled = std::getenv("APS5_DRAW_INPUT_MEMO") != nullptr;
    return enabled;
}

struct DrawInputMemoEntry {
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    Recorder::SnapshotUse use = Recorder::SnapshotUse::Vertex;
    std::uint64_t epoch = 0;
    std::uint64_t unwatched = 0;
    std::uint64_t pendingSerial = 0;
    std::uint64_t registryGeneration = 0;
    std::uint64_t generation = 0;
    std::uint32_t derived = 0;
    const Recorder* recorder = nullptr;
    std::weak_ptr<Buffer> buffer;
};

constexpr std::size_t DrawInputMemoSlots = 64;

std::array<DrawInputMemoEntry, DrawInputMemoSlots>& DrawInputMemoTable() {
    thread_local std::array<DrawInputMemoEntry, DrawInputMemoSlots> table;
    return table;
}

DrawInputMemoEntry& drawInputMemoSlot(std::uint64_t address, std::size_t bytes, Recorder::SnapshotUse use) {
    const auto hash = (address >> 4u) ^ (static_cast<std::uint64_t>(bytes) * 0x9e3779b97f4a7c15ull) ^ static_cast<std::uint64_t>(use);
    return DrawInputMemoTable()[static_cast<std::size_t>((hash ^ (hash >> 29u)) % DrawInputMemoSlots)];
}

std::atomic<std::uint64_t> drawInputMemoHits{0};
std::atomic<std::uint64_t> drawInputMemoMisses{0};

}

DrawInputMemoCounts DrawInputMemoCounters() {
    return {drawInputMemoHits.load(std::memory_order_relaxed), drawInputMemoMisses.load(std::memory_order_relaxed)};
}

// APS5_VRAM_DRAW_INPUTS=1: vertex and index ranges are copied (once while unchanged, through the
// draw snapshots) into device-local, host-visible memory instead of being read in place from the
// host import on every draw, so the GPU fetches geometry from VRAM rather than across PCIe.
bool VramDrawInputs() {
    static const bool enabled = std::getenv("APS5_VRAM_DRAW_INPUTS") != nullptr;
    return enabled;
}

std::shared_ptr<Buffer> drawInputBuffer(const Context& context, std::size_t bytes, Recorder::SnapshotUse use) {
    const VkBufferUsageFlags usage = use == Recorder::SnapshotUse::Vertex ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (VramDrawInputs()) {
        static std::atomic<bool> unavailable{false};
        if (!unavailable.load(std::memory_order_relaxed)) {
            try {
                return std::make_shared<Buffer>(context, bytes, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            } catch (const std::runtime_error& error) {
                if (!unavailable.exchange(true)) std::fprintf(stderr, "[vertex-input] device-local host-visible draw inputs unavailable (%s); using host memory\n", error.what());
            }
        }
    }
    return std::make_shared<Buffer>(context, bytes, usage);
}

DrawInputCopy CopyDrawInput(const Context& context, Recorder* recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment, Recorder::SnapshotUse use) {
    Require(use != Recorder::SnapshotUse::Storage, "a draw input is a vertex or index buffer");
    DrawInputCopy copy;
    if (recorder != nullptr && bytes != 0) {
        const bool memo = drawInputMemo();
        const auto epoch = memo ? GuestMemory::ThreadCollectEpoch() : 0;
        if (epoch != 0) {
            auto& slot = drawInputMemoSlot(address, bytes, use);
            if (slot.epoch == epoch && slot.address == address && slot.bytes == bytes && slot.use == use && slot.recorder == recorder && slot.unwatched == GuestMemory::UnwatchSerial() && slot.pendingSerial == StorageTexture::PendingSerial() && slot.registryGeneration == GuestAllocations::GuestAllocationsGeneration_nid_postfix() && GuestMemory::UnchangedSince(address, bytes, slot.generation) && !(InputTokensEnabled() ? recorder->PendingWriteOverlapsScreened(address, bytes) : recorder->PendingWriteOverlaps(address, bytes)) && !Recorder::QueuedLabelsOverlap(address, bytes)) {
                if (auto buffer = slot.buffer.lock()) {
                    GuestMemory::NoteProbeEvent("memo draw input");
                    drawInputMemoHits.fetch_add(1, std::memory_order_relaxed);
                    copy.buffer = std::move(buffer);
                    copy.reused = true;
                    copy.derived = slot.derived;
                    copy.generation = slot.generation;
                    copy.registryGeneration = slot.registryGeneration;
                    return copy;
                }
            }
            drawInputMemoMisses.fetch_add(1, std::memory_order_relaxed);
        }
        GuestMemory::FlushGpuWrites(address, bytes);
        const auto pendingSerial = epoch != 0 ? StorageTexture::PendingSerial() : 0;
        copy.registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        copy.generation = GuestMemory::CollectWrites(address, bytes);
        if (copy.generation != 0) copy.buffer = recorder->ReusableDrawSnapshot(address, bytes, use, &copy.derived);
        if (copy.buffer != nullptr) {
            copy.reused = true;
            if (epoch != 0 && GuestMemory::ThreadCollectEpoch() == epoch && StorageTexture::PendingSerial() == pendingSerial) {
                drawInputMemoSlot(address, bytes, use) = {address, bytes, use, epoch, GuestMemory::UnwatchSerial(), pendingSerial, copy.registryGeneration, copy.generation, copy.derived, recorder, copy.buffer};
            }
            return copy;
        }
    }
    copy.buffer = drawInputBuffer(context, bytes, use);
    GuestMemory::Read(address, copy.buffer->Bytes(), alignment);
    return copy;
}

void KeepDrawInput(Recorder* recorder, std::uint64_t address, const DrawInputCopy& copy, Recorder::SnapshotUse use, std::uint32_t derived) {
    if (recorder == nullptr || copy.reused || copy.generation == 0 || copy.buffer == nullptr) return;
    recorder->KeepDrawSnapshot(address, copy.buffer->Bytes().size(), copy.generation, copy.registryGeneration, copy.buffer, use, derived);
}

namespace {

enum class DrawInputOutcome : std::size_t { InPlace, Disabled, PendingWrite, CopiedWriter, NotImported, Misaligned, Count };

struct DrawInputProfile {
    std::mutex mutex;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawInputOutcome::Count)> counts{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrawInputOutcome::Count)> bytes{};
    std::uint64_t copies = 0;
    std::uint64_t copiedBytes = 0;
    double copyUs = 0.0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

DrawInputProfile& DrawInputStats() {
    static auto* profile = new DrawInputProfile();
    return *profile;
}

bool drawInputProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

void reportDrawInputs(DrawInputProfile& profile) {
    const auto now = std::chrono::steady_clock::now();
    if (now - profile.lastReport < std::chrono::seconds(10)) return;
    profile.lastReport = now;
    constexpr std::array<const char*, static_cast<std::size_t>(DrawInputOutcome::Count)> names{"in place", "disabled", "pending GPU write or label", "copied writer", "not imported", "misaligned"};
    std::string line = "[vertex-input] in-place checks (10 s):";
    for (std::size_t index = 0; index < names.size(); ++index) {
        char item[96];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f MiB)", names[index], static_cast<unsigned long long>(profile.counts[index]), static_cast<double>(profile.bytes[index]) / (1024.0 * 1024.0));
        line += item;
    }
    char tail[160];
    std::snprintf(tail, sizeof(tail), "; vertex range copies %llu (%.1f MiB) in %.1f ms\n", static_cast<unsigned long long>(profile.copies), static_cast<double>(profile.copiedBytes) / (1024.0 * 1024.0), profile.copyUs / 1000.0);
    line += tail;
    std::fputs(line.c_str(), stderr);
    profile.counts = {};
    profile.bytes = {};
    profile.copies = profile.copiedBytes = 0;
    profile.copyUs = 0.0;
}

const HostImport* noteDrawInput(DrawInputOutcome outcome, std::size_t bytes, const HostImport* import) {
    if (!drawInputProfiled()) return import;
    auto& profile = DrawInputStats();
    std::lock_guard lock(profile.mutex);
    ++profile.counts[static_cast<std::size_t>(outcome)];
    profile.bytes[static_cast<std::size_t>(outcome)] += bytes;
    reportDrawInputs(profile);
    return import;
}

void noteDrawInputCopy(std::size_t bytes, double us) {
    if (!drawInputProfiled()) return;
    auto& profile = DrawInputStats();
    std::lock_guard lock(profile.mutex);
    ++profile.copies;
    profile.copiedBytes += bytes;
    profile.copyUs += us;
}

// Whether a draw input range is unchanged since its previous use (by its write generation then).
bool stableDrawInput(std::uint64_t address, std::size_t bytes) {
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, std::uint64_t> seen;
    const auto key = address * 0x9e3779b97f4a7c15ull ^ static_cast<std::uint64_t>(bytes);
    std::lock_guard lock(mutex);
    if (seen.size() > 262144) seen.clear();
    auto& generation = seen[key];
    if (generation != 0 && GuestMemory::UnchangedSince(address, bytes, generation)) return true;
    generation = GuestMemory::CollectWrites(address, bytes);
    return false;
}

const HostImport* checkInPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment) {
    const bool screened = InputTokensEnabled();
    struct CheckTiming {
        bool on;
        std::chrono::steady_clock::time_point start;
        ~CheckTiming() {
            if (!on) return;
            Tokens().inPlaceChecks.fetch_add(1, std::memory_order_relaxed);
            Tokens().inPlaceNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
        }
    } timing{drawInputProfiled() && (screened || DrawTokensEnabled()), {}};
    if (timing.on) timing.start = std::chrono::steady_clock::now();
    if ((LivePerformanceControls().Get() & PerformanceControls::InPlaceDrawInputs) == 0) return noteDrawInput(DrawInputOutcome::Disabled, bytes, nullptr);
    auto* recorder = context.recorder;
    if (recorder == nullptr || bytes == 0) return noteDrawInput(DrawInputOutcome::Disabled, bytes, nullptr);
    GuestMemory::FlushGpuWrites(address, bytes);
    if ((screened ? recorder->PendingWriteOverlapsScreened(address, bytes) : recorder->PendingWriteOverlaps(address, bytes)) || recorder->PendingLabelIn(address, bytes) || recorder->QueuedStoreOverlaps(address, bytes)) return noteDrawInput(DrawInputOutcome::PendingWrite, bytes, nullptr);
    const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(address, bytes); };
    if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return noteDrawInput(DrawInputOutcome::CopiedWriter, bytes, nullptr);
    const auto drawWriters = DrawCopiedWriters();
    if (std::any_of(drawWriters->begin(), drawWriters->end(), overlaps)) return noteDrawInput(DrawInputOutcome::CopiedWriter, bytes, nullptr);
    const auto* import = HostImportFor(context, address, bytes);
    if (import == nullptr) return noteDrawInput(DrawInputOutcome::NotImported, bytes, nullptr);
    if ((address - import->base) % alignment != 0) return noteDrawInput(DrawInputOutcome::Misaligned, bytes, nullptr);
    // APS5_VRAM_DRAW_INPUTS: a range unchanged since its previous use is copied to VRAM (the copy
    // path, reused while unchanged); one the guest rewrites between uses stays in place.
    if (VramDrawInputs() && stableDrawInput(address, bytes)) return noteDrawInput(DrawInputOutcome::Disabled, bytes, nullptr);
    return noteDrawInput(DrawInputOutcome::InPlace, bytes, import);
}

}

const HostImport* InPlaceDrawInput(const Context& context, std::uint64_t address, std::size_t bytes, std::size_t alignment) {
    const auto* import = checkInPlaceDrawInput(context, address, bytes, alignment);
    if (import != nullptr) context.recorder->NotePendingRead(address, bytes, Recorder::ReadKind::DrawInput);
    return import;
}

namespace {

// The draw's inputs before its resources (prepareDrawInputs): the validated parameters, the index
// buffer copy with its highest index, the vertex buffer copies and their layout, the fragment
// outputs and the pipeline stages.
struct DrawInputs {
    // An empty auto draw: nothing to record.
    bool nothing = false;
    std::uint64_t indexBytes = 0;
    std::shared_ptr<Buffer> indices;
    VkBuffer indexHandle = VK_NULL_HANDLE;
    VkDeviceSize indexOffset = 0;
    std::uint32_t maxIndex = 0;
    VertexInputLayout vertexInput;
    std::vector<std::shared_ptr<Buffer>> vertexBuffers;
    std::vector<VkBuffer> vertexHandles;
    std::vector<VkDeviceSize> vertexOffsets;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inPlaceReads;
    std::set<std::uint32_t> fragmentOutputs;
    VkPipelineStageFlags shaderStages = 0;
    std::uint32_t meshGroups = 0;
};

void validateDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe* recipe, bool checkRanges, DrawInputs& inputs, bool& validateMemoized, bool& validateHit) {
    APS5_LOG_OUT_DEBUG("Draw indices=%u instances=%u indexSize=%u flags=%u indexAddress=0x%llx", draw.indexCount, draw.instanceCount, draw.indexSize, draw.flags, static_cast<unsigned long long>(draw.indexAddress));
    APS5_LOG_OUT_DEBUG("State colorTarget=%u render=%ux%u colorAddress=0x%llx colorBytes=%llu colorExtent=%ux%u", state.hasColorTarget ? 1u : 0u, state.renderExtent.width, state.renderExtent.height, static_cast<unsigned long long>(state.color.address), static_cast<unsigned long long>(state.color.bytes), state.color.extent.width, state.color.extent.height);
    APS5_LOG_OUT_DEBUG("Viewport x=%f y=%f w=%f h=%f minDepth=%f maxDepth=%f", state.viewport.x, state.viewport.y, state.viewport.width, state.viewport.height, state.viewport.minDepth, state.viewport.maxDepth);
    APS5_LOG_OUT_DEBUG("Scissor x=%d y=%d w=%u h=%u topology=%u cullMode=0x%x frontFace=%u", state.scissor.offset.x, state.scissor.offset.y, state.scissor.extent.width, state.scissor.extent.height, static_cast<unsigned>(state.topology), static_cast<unsigned>(state.cullMode), static_cast<unsigned>(state.frontFace));
    // An indirect draw: the counts live in guest memory records (Pm4::DrawParameters::IndirectDraw),
    // read by the GPU from the host import or, when the GPU could not see their current bytes, by
    // the CPU (Draw's `records`). The driver resolved every non-vertex-path draw before this.
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    Require(draw.indexed ? draw.flags == 0 : (draw.flags & ~0x20u) == 0, "draw modifiers are unsupported");
    if (draw.indexed) {
        Require(draw.indexSize == 2 || draw.indexSize == 4, "only uint16 and uint32 index buffers are supported");
    } else {
        Require(draw.indexAddress == 0 && draw.indexSize == 0, "auto draw must not reference an index buffer");
        if (args == nullptr) {
            if (draw.indexCount == 0 || draw.instanceCount == 0) {
                inputs.nothing = true;
                return;
            }
            Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - (draw.indexCount - 1u), "auto draw vertex range overflow");
            Require(draw.firstInstance <= std::numeric_limits<std::uint32_t>::max() - (draw.instanceCount - 1u), "auto draw instance range overflow");
        }
    }
    if (args == nullptr) Require(draw.indexCount != 0 && draw.instanceCount != 0, "zero-count indexed draws are unsupported");
    else Require((!state.stages.mesh || draw.indexed) && !state.stages.tessellation && !state.rectList, "indirect draw on a non-vertex path must be resolved by the driver");
    inputs.indexBytes = static_cast<std::uint64_t>(draw.indexCount) * draw.indexSize;
    const auto indexBytes = inputs.indexBytes;
    APS5_LOG_OUT_DEBUG("Index buffer bytes=%llu", static_cast<unsigned long long>(indexBytes));
    Require(indexBytes <= std::numeric_limits<std::size_t>::max(), "index buffer size overflow");
    if (draw.indexed && checkRanges) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes), draw.indexSize);
    APS5_LOG_CHARS_OUT_DEBUG("Index buffer range OK");
    Require(!draw.indexed || !state.hasColorTarget || draw.indexAddress + indexBytes <= state.color.address || state.color.address + state.color.bytes <= draw.indexAddress, "index buffer aliases the render target");
    if (state.rectList) Require(draw.indexCount % 3 == 0, "incomplete rect-list primitive");
    APS5_LOG_CHARS_OUT_DEBUG("ValidateShaders");
    if (recipe != nullptr) {
        inputs.fragmentOutputs = recipe->fragmentOutputs;
        inputs.shaderStages = recipe->shaderStages;
    } else {
        inputs.fragmentOutputs = CachedFragmentOutputs(context, shaders, state, validateMemoized, validateHit);
        inputs.shaderStages = PipelineStages(shaders);
    }
    APS5_LOG_CHARS_OUT_DEBUG("ValidateShaders OK");
    APS5_LOG_OUT_DEBUG("PipelineStages=0x%x", static_cast<unsigned>(inputs.shaderStages));
    if (state.stages.mesh) {
        APS5_LOG_CHARS_OUT_DEBUG("Mesh path");
        Require(context.meshShader, "device does not support mesh shaders");
        const auto& mesh = *state.stages.mesh;
        const auto inputSize = mesh.inputPrimitive == 1 ? 1u : mesh.inputPrimitive == 2 ? 2u : 3u;
        Require(draw.indexCount >= inputSize && mesh.primitivesPerGroup != 0, "mesh draw contains no complete primitive");
        if (args == nullptr) {
            const auto step = mesh.inputPrimitive == 5 || mesh.inputPrimitive == 6 ? 1u : inputSize;
            const auto primitives = (draw.indexCount - inputSize) / step + 1u;
            inputs.meshGroups = (primitives - 1u) / mesh.primitivesPerGroup + 1u;
            APS5_LOG_OUT_DEBUG("Mesh primitives=%u groups=%u", primitives, inputs.meshGroups);
            Require(inputs.meshGroups <= context.meshLimits.maxMeshWorkGroupCount[0] && draw.instanceCount <= context.meshLimits.maxMeshWorkGroupCount[1] && static_cast<std::uint64_t>(inputs.meshGroups) * draw.instanceCount <= context.meshLimits.maxMeshWorkGroupTotalCount, "mesh draw exceeds workgroup count limits");
        }
    }
    if (state.stages.tessellation) Require(draw.indexCount % state.stages.tessellation->inputControlPoints == 0, "incomplete tessellation patch");
    // Viewport and scissor are dynamic pipeline state, so their limits are checked here per draw.
    ValidateViewport(context, state.viewport);
    ValidateDepthBounds(context, state);
}

struct VertexFetchPlan {
    std::vector<std::optional<std::size_t>> shortBytes;
    std::vector<std::size_t> fetchOf;
    VertexCopyPlan plan;
};

struct AssembledPush {
    std::array<std::byte, PipelinePushConstantBytes> bytes{};
    VkShaderStageFlags stages = 0;
};

struct DrawInputEffect {
    bool keep = false;
    std::uint64_t address = 0;
    std::size_t bytes = 0;
    DrawInputCopy copy;
    Recorder::SnapshotUse use = Recorder::SnapshotUse::Vertex;
    std::uint32_t derived = 0;
};

}

struct DrawPrebuild {
    VkDevice device = VK_NULL_HANDLE;
    bool refused = false;
    bool nothing = false;
    std::uint64_t indexBytes = 0;
    std::set<std::uint32_t> fragmentOutputs;
    VkPipelineStageFlags shaderStages = 0;
    std::uint32_t meshGroups = 0;
    bool validateMemoized = false;
    bool validateHit = false;
    // An indexed draw's highest index, scanned on the prep worker at the index range's collect
    // generation (0: not scanned). The committer takes it while the range is unchanged since.
    std::uint32_t highestIndex = 0;
    std::uint64_t indexGeneration = 0;
    VertexInputLayout vertexInput;
    std::optional<VertexFetchPlan> vertexPlan;
    std::vector<bool> proxied;
    std::optional<State> masked;
    bool keyed = false;
    ResourceCache::Key key;
    bool aliasesOk = false;
    AssembledPush push;
    bool storesThroughAddresses = false;
};

namespace {

// The highest index of an index buffer's bytes (`indexSize` 2 or 4).
namespace {

// The committer scans every unmemoized index range (about 5 KiB a draw at the menu); 32 bytes at a time.
__attribute__((target("avx2"))) std::uint32_t ScanHighestIndexAvx2(const std::byte* data, std::size_t count, std::uint32_t indexSize) {
    std::size_t index = 0;
    std::uint32_t highest = 0;
    if (indexSize == 2) {
        __m256i lanes = _mm256_setzero_si256();
        for (; index + 16 <= count; index += 16) lanes = _mm256_max_epu16(lanes, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index * 2)));
        alignas(32) std::uint16_t values[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(values), lanes);
        for (const auto value : values) highest = std::max<std::uint32_t>(highest, value);
        for (; index < count; ++index) {
            std::uint16_t value;
            std::memcpy(&value, data + index * 2, sizeof(value));
            highest = std::max<std::uint32_t>(highest, value);
        }
    } else {
        __m256i lanes = _mm256_setzero_si256();
        for (; index + 8 <= count; index += 8) lanes = _mm256_max_epu32(lanes, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + index * 4)));
        alignas(32) std::uint32_t values[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(values), lanes);
        for (const auto value : values) highest = std::max(highest, value);
        for (; index < count; ++index) {
            std::uint32_t value;
            std::memcpy(&value, data + index * 4, sizeof(value));
            highest = std::max(highest, value);
        }
    }
    return highest;
}

}

std::uint32_t ScanHighestIndex(std::span<const std::byte> bytes, std::uint32_t indexSize) {
    static const bool avx2 = __builtin_cpu_supports("avx2") && std::getenv("APS5_NO_AVX2_INDEX_SCAN") == nullptr;
    if (avx2 && (indexSize == 2 || indexSize == 4)) return ScanHighestIndexAvx2(bytes.data(), bytes.size() / indexSize, indexSize);
    std::uint32_t highest = 0;
    for (std::size_t offset = 0; offset + indexSize <= bytes.size(); offset += indexSize) {
        std::uint32_t index = 0;
        if (indexSize == 2) {
            std::uint16_t value = 0;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            index = value;
        } else {
            std::memcpy(&index, bytes.data() + offset, sizeof(index));
        }
        highest = std::max(highest, index);
    }
    return highest;
}

// Whether a prebuild scans the index range on the prep worker (live `prep_index`, env
// APS5_PREP_INDEX_SCAN), so the committer's scan becomes an unchanged-range check.
bool VerifyPrepIndexScan() {
    static const bool enabled = std::getenv("APS5_VERIFY_PREP_INDEX") != nullptr;
    return enabled;
}

bool PrepIndexScan() {
    static const bool enabled = std::getenv("APS5_PREP_INDEX_SCAN") != nullptr;
    return LiveFlag("prep_index", enabled);
}

VertexFetchPlan planVertexFetches(const State& state, const Pm4::DrawParameters& draw, std::span<const ShaderRecompiler::VertexAttribute> attributes, std::uint32_t maxIndex, bool indirect) {
    VertexFetchPlan result;
    result.shortBytes.resize(attributes.size());
    result.fetchOf.assign(attributes.size(), attributes.size());
    std::vector<VertexFetch> fetches;
    fetches.reserve(attributes.size());
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        const auto& attribute = attributes[index];
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        if (const auto inRange = ShortRawVertexBufferBytes(attribute)) {
            result.shortBytes[index] = *inRange;
            continue;
        }
        // An indirect draw's counts are unknown here: the descriptor's whole range is copied.
        const auto bytes = indirect ? VertexBufferExtent(attribute) : VertexBufferReadSize(attribute, maxIndex, draw.instanceCount, draw.firstInstance);
        result.fetchOf[index] = fetches.size();
        Require(!state.hasColorTarget || address + bytes <= state.color.address || state.color.address + state.color.bytes <= address, "vertex buffer aliases the render target");
        fetches.push_back({address, address + bytes, (fields[1] >> 16u) & 0x3fffu, attribute.fetchIndex, DecodeVertexFormat(attribute).alignment});
    }
    result.plan = PlanVertexCopies(fetches);
    return result;
}

// Draw's validation, index and vertex phases. With `recipe` the fragment outputs, the pipeline
// stages and the vertex input layout are the recipe's (derived from the same compiled stages)
// instead of computed.
DrawInputs prepareDrawInputs(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawOutcome& outcome, DrawTimer& timer, const DrawRecipe* recipe, DrawPrebuild* prebuilt = nullptr, std::vector<DrawInputEffect>* effects = nullptr) {
    DrawInputs inputs;
    if (prebuilt != nullptr) {
        inputs.nothing = prebuilt->nothing;
        inputs.indexBytes = prebuilt->indexBytes;
        if (!inputs.nothing && draw.indexed) GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(inputs.indexBytes), draw.indexSize);
        inputs.fragmentOutputs = std::move(prebuilt->fragmentOutputs);
        inputs.shaderStages = prebuilt->shaderStages;
        inputs.meshGroups = prebuilt->meshGroups;
        outcome.validateMemoized = prebuilt->validateMemoized;
        outcome.validateHit = prebuilt->validateHit;
    } else {
        validateDraw(context, state, draw, shaders, recipe, true, inputs, outcome.validateMemoized, outcome.validateHit);
    }
    if (inputs.nothing) return inputs;
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    const auto indexBytes = inputs.indexBytes;
    timer.phase(PhaseValidate);
    inputs.maxIndex = draw.indexed ? 0u : draw.firstVertex + draw.indexCount - 1u;
    // APS5_PROFILE_LOOKUP: [vertex-split] times the index block (in-place check, copy or memo, max
    // scan) apart from the vertex ranges.
    static const bool vertexSplit = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    struct VertexSplit {
        std::atomic<std::uint64_t> draws{0}, indexed{0}, indexBytes{0}, scanned{0}, prescanned{0}, indexNs{0}, rangesNs{0};
        std::atomic<std::int64_t> reported{0};
    };
    static VertexSplit vertexStats;
    auto vertexLap = vertexSplit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto vertexMark = [&](std::atomic<std::uint64_t>& into) {
        if (!vertexSplit) return;
        const auto now = std::chrono::steady_clock::now();
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - vertexLap).count()), std::memory_order_relaxed);
        vertexLap = now;
    };
    struct VertexReport {
        VertexSplit& stats;
        bool enabled;
        ~VertexReport() {
            if (!enabled) return;
            const auto draws = stats.draws.fetch_add(1, std::memory_order_relaxed) + 1;
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            auto last = stats.reported.load(std::memory_order_relaxed);
            if (now - last <= 10'000'000'000ll || !stats.reported.compare_exchange_strong(last, now)) return;
            stats.draws.store(0);
            const auto indexed = stats.indexed.exchange(0);
            const auto per = [&](std::atomic<std::uint64_t>& value) { return static_cast<double>(value.exchange(0)) / 1000.0 / static_cast<double>(draws); };
            const auto bytes = stats.indexBytes.exchange(0);
            const auto scanned = stats.scanned.exchange(0);
            const auto prescanned = stats.prescanned.exchange(0);
            std::fprintf(stderr, "[vertex-split] %llu draws (%llu indexed, %.0f index bytes each, %llu unmemoized, %llu of them prescanned), us per draw: index %.2f ranges %.2f\n", static_cast<unsigned long long>(draws), static_cast<unsigned long long>(indexed), indexed != 0 ? static_cast<double>(bytes) / static_cast<double>(indexed) : 0.0, static_cast<unsigned long long>(scanned), static_cast<unsigned long long>(prescanned), per(stats.indexNs), per(stats.rangesNs));
        }
    } vertexReport{vertexStats, vertexSplit};
    if (draw.indexed) {
        if (vertexSplit) {
            vertexStats.indexed.fetch_add(1, std::memory_order_relaxed);
            vertexStats.indexBytes.fetch_add(indexBytes, std::memory_order_relaxed);
        }
        const auto use = draw.indexSize == 2 ? Recorder::SnapshotUse::Index16 : Recorder::SnapshotUse::Index32;
        const auto* import = effects != nullptr ? checkInPlaceDrawInput(context, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize) : InPlaceDrawInput(context, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize);
        if (import != nullptr && effects != nullptr) effects->push_back({false, draw.indexAddress, static_cast<std::size_t>(indexBytes)});
        DrawInputCopy copy;
        if (import == nullptr) copy = CopyDrawInput(context, context.recorder, draw.indexAddress, static_cast<std::size_t>(indexBytes), draw.indexSize, use);
        else GuestMemory::CheckRange(reinterpret_cast<const void*>(draw.indexAddress), static_cast<std::size_t>(indexBytes), draw.indexSize);
        std::uint32_t highest = copy.derived;
        if (!copy.reused) {
            if (vertexSplit) vertexStats.scanned.fetch_add(1, std::memory_order_relaxed);
            const bool timeScan = drawInputProfiled() && (InputTokensEnabled() || DrawTokensEnabled());
            const auto scanStart = timeScan ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            highest = 0;
            if (import != nullptr) GuestMemory::NoteBaked(draw.indexAddress, static_cast<std::size_t>(indexBytes));
            // A copy in VRAM is slow to read back: its bytes are the guest's, just read.
            const bool guestBytes = import != nullptr || VramDrawInputs();
            const auto bytes = guestBytes ? std::span<const std::byte>(reinterpret_cast<const std::byte*>(draw.indexAddress), static_cast<std::size_t>(indexBytes)) : std::span<const std::byte>(copy.buffer->Bytes());
            // The prep worker's scan holds while the range is unchanged since its collect: a CPU
            // write is stamped by this collect, a stored GPU result when it lands.
            const bool prescanned = prebuilt != nullptr && prebuilt->indexGeneration != 0 && prebuilt->indexBytes == indexBytes && GuestMemory::CollectWrites(draw.indexAddress, static_cast<std::size_t>(indexBytes)) != 0 && GuestMemory::UnchangedSince(draw.indexAddress, static_cast<std::size_t>(indexBytes), prebuilt->indexGeneration);
            if (prescanned) {
                highest = prebuilt->highestIndex;
                if (VerifyPrepIndexScan() && highest != ScanHighestIndex(bytes, draw.indexSize)) {
                    std::fprintf(stderr, "[vertex] APS5_VERIFY_PREP_INDEX: the prep worker's highest index %u differs from the committer's scan at 0x%llx\n", highest, static_cast<unsigned long long>(draw.indexAddress));
                    std::fflush(stderr);
                    std::abort();
                }
                if (vertexSplit) vertexStats.prescanned.fetch_add(1, std::memory_order_relaxed);
            } else {
                highest = ScanHighestIndex(bytes, draw.indexSize);
            }
            if (timeScan) {
                Tokens().indexScans.fetch_add(1, std::memory_order_relaxed);
                Tokens().indexScanNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - scanStart).count()), std::memory_order_relaxed);
            }
            if (import == nullptr && effects != nullptr) effects->push_back({true, draw.indexAddress, 0, copy, use, highest});
            else if (import == nullptr) KeepDrawInput(context.recorder, draw.indexAddress, copy, use, highest);
        }
        Require(highest <= context.limits.maxDrawIndexedIndexValue, "index exceeds the device's indexed draw limit");
        Require(!state.stages.mesh || state.stages.mesh->inputPrimitive != 5 || !state.primitiveRestart || highest != (draw.indexSize == 2 ? 0xffffu : 0xffffffffu), "primitive restart in a triangle fan geometry draw is unsupported");
        inputs.maxIndex = highest;
        if (import != nullptr) {
            inputs.indexHandle = import->buffer;
            inputs.indexOffset = draw.indexAddress - import->base;
            inputs.inPlaceReads.emplace_back(draw.indexAddress, draw.indexAddress + indexBytes);
        } else {
            inputs.indices = std::move(copy.buffer);
            inputs.indexHandle = inputs.indices->Handle();
        }
    }
    vertexMark(vertexStats.indexNs);
    APS5_LOG_CHARS_OUT_DEBUG("Index validation OK");
    const auto& attributes = shaders.front().program->vertexAttributes;
    // Validates the vertex descriptors; the layout also keys and builds the pipeline.
    if (recipe != nullptr) inputs.vertexInput = recipe->vertexInput;
    else if (prebuilt != nullptr) inputs.vertexInput = std::move(prebuilt->vertexInput);
    else inputs.vertexInput = BuildVertexInputLayout(context, attributes);
    inputs.vertexOffsets.assign(attributes.size(), 0);
    // An indexed draw's vertex offset moves every fetch: the copy must reach the last one.
    if (draw.indexed) {
        Require(draw.firstVertex <= std::numeric_limits<std::uint32_t>::max() - inputs.maxIndex, "indexed draw vertex range overflow");
        inputs.maxIndex += draw.firstVertex;
    }
    auto fetchPlan = prebuilt != nullptr && prebuilt->vertexPlan.has_value() ? std::move(*prebuilt->vertexPlan) : planVertexFetches(state, draw, attributes, inputs.maxIndex, args != nullptr);
    const auto& fetchOf = fetchPlan.fetchOf;
    const auto& plan = fetchPlan.plan;
    std::vector<std::shared_ptr<Buffer>> shortBuffers(attributes.size());
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        if (!fetchPlan.shortBytes[index].has_value()) continue;
        const auto& attribute = attributes[index];
        const auto& fields = attribute.resource.fields;
        const auto address = fields[0] | (static_cast<std::uint64_t>(fields[1] & 0xffffu) << 32u);
        const auto inRange = *fetchPlan.shortBytes[index];
        auto buffer = std::make_shared<Buffer>(context, DecodeVertexFormat(attribute).bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        std::fill(buffer->Bytes().begin(), buffer->Bytes().end(), std::byte{0});
        if (inRange != 0) GuestMemory::Read(address, buffer->Bytes().first(inRange), 4);
        shortBuffers[index] = std::move(buffer);
    }
    inputs.vertexBuffers.reserve(plan.copies.size() + std::count_if(shortBuffers.begin(), shortBuffers.end(), [](const auto& buffer) { return buffer != nullptr; }));
    inputs.vertexHandles.reserve(attributes.size());
    std::vector<VkBuffer> rangeHandles;
    std::vector<VkDeviceSize> rangeOffsets;
    rangeHandles.reserve(plan.copies.size());
    rangeOffsets.reserve(plan.copies.size());
    for (std::size_t range = 0; range < plan.copies.size(); ++range) {
        const auto [begin, end] = plan.copies[range];
        const auto bytes = static_cast<std::size_t>(end - begin);
        GuestMemory::CheckRange(reinterpret_cast<const void*>(begin), bytes, 1);
        if (const auto* import = effects != nullptr ? checkInPlaceDrawInput(context, begin, bytes, plan.alignments[range]) : InPlaceDrawInput(context, begin, bytes, plan.alignments[range])) {
            if (effects != nullptr) effects->push_back({false, begin, bytes});
            rangeHandles.push_back(import->buffer);
            rangeOffsets.push_back(begin - import->base);
            inputs.inPlaceReads.emplace_back(begin, end);
            continue;
        }
        const auto copyStarted = std::chrono::steady_clock::now();
        auto copy = CopyDrawInput(context, context.recorder, begin, bytes, 1, Recorder::SnapshotUse::Vertex);
        if (effects != nullptr) effects->push_back({true, begin, 0, copy, Recorder::SnapshotUse::Vertex, 0});
        else KeepDrawInput(context.recorder, begin, copy, Recorder::SnapshotUse::Vertex, 0);
        noteDrawInputCopy(bytes, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - copyStarted).count());
        rangeHandles.push_back(copy.buffer->Handle());
        rangeOffsets.push_back(0);
        inputs.vertexBuffers.push_back(std::move(copy.buffer));
    }
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        if (shortBuffers[i] != nullptr) {
            inputs.vertexHandles.push_back(shortBuffers[i]->Handle());
            inputs.vertexOffsets[i] = 0;
            inputs.vertexBuffers.push_back(std::move(shortBuffers[i]));
            continue;
        }
        inputs.vertexHandles.push_back(rangeHandles[plan.copyOf[fetchOf[i]]]);
        inputs.vertexOffsets[i] = rangeOffsets[plan.copyOf[fetchOf[i]]] + plan.offsets[fetchOf[i]];
    }
    vertexMark(vertexStats.rangesNs);
    timer.phase(PhaseVertex);
    return inputs;
}

// The resident-target proof of one attachment (StorageTexture::Refresh: FlushPending, CollectWrites
// over the target's pages, the DCC key scan of TextureClearKeys, then UnchangedSince), with the
// [draws] target-lookup accounting. `lookup` makes (or finds) the image; DrawWithRecipe refreshes
// the stored object instead.
template<typename Lookup>
std::shared_ptr<StorageTexture> refreshResidentTarget(const Context& context, const State& state, const ColorTarget& color, DrawOutcome& outcome, bool profile, Lookup&& lookup) {
    const auto lookupStart = std::chrono::steady_clock::now();
    const auto walkedBefore = profile ? GuestMemory::ThreadCollectedBytes() : 0;
    std::shared_ptr<StorageTexture> resident;
    try {
        resident = lookup();
        if (resident != nullptr) materializeRegisterClear(context, color, *resident);
    } catch (const std::exception& error) {
        static std::mutex reportMutex;
        static std::set<std::uint64_t> reported;
        std::lock_guard lock(reportMutex);
        if (reported.insert(color.address).second) std::fprintf(stderr, "[gpu] color target 0x%llx stays non-resident: %s\n", static_cast<unsigned long long>(color.address), error.what());
    }
    if (resident != nullptr) materializeCmaskClear(context, color, resident.get());
    if (profile) {
        const auto lookupUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - lookupStart).count();
        ++outcome.targetLookups;
        if (lookupUs >= 1000.0) {
            ++outcome.slowLookups;
            outcome.slowLookupUs += lookupUs;
        }
        outcome.lookupUs += lookupUs;
        outcome.pagesWalked += (GuestMemory::ThreadCollectedBytes() - walkedBefore) / 4096;
        const bool fullScissor = state.scissor.offset.x <= 0 && state.scissor.offset.y <= 0 && state.scissor.extent.width >= color.extent.width && state.scissor.extent.height >= color.extent.height;
        ++(fullScissor ? outcome.fullScissorLookups : outcome.partialLookups);
    }
    return resident;
}

// A draw's resources (resolveDrawResources): the resource-cache template a recordable draw's
// stages repeat, or a fresh build.
thread_local std::shared_ptr<DrawSpeculation> forkSpeculation;
std::atomic<std::uint64_t> forkSpeculationReuses{0};

void countForkReuse() {
    forkSpeculationReuses.fetch_add(1, std::memory_order_relaxed);
}

struct ResolvedResources {
    std::shared_ptr<ShaderResources> resources;
    std::vector<ShaderResources::MovedBuffer> moved;
    ResourceCache::Key contentKey;
    bool cacheable = false;
    const ShaderResources::BuildTiming* built = nullptr;
    std::shared_ptr<DrawSpeculation> speculated;
    std::shared_ptr<ShaderResources> speculatedSource;
};

const char* addressSpeculationRefusal(DrawSpeculation::Miss miss) {
    static const auto names = [] {
        std::array<std::string, SpeculationMissCount> list;
        for (std::size_t index = 0; index < SpeculationMissCount; ++index) list[index] = std::string("address ") + SpeculationMissNames[index];
        return list;
    }();
    const auto index = static_cast<std::size_t>(miss);
    return names[index < SpeculationMissCount ? index : 0].c_str();
}

struct LookupSplit {
    std::atomic<std::uint64_t> draws{0}, keyNs{0}, findNs{0}, proofNs{0}, aliasNs{0}, templateNs{0}, templates{0};
    std::atomic<std::int64_t> reported{0};
};

LookupSplit& LookupSplitStats() {
    static LookupSplit stats;
    return stats;
}

ResolvedResources resolveDrawResources(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::uint64_t indexBytes, bool recordable, DrawOutcome& outcome, DrawTimer& timer, DrawPrebuild* prebuilt = nullptr) {
    ResolvedResources resolved;
    static const bool split = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    auto& splitStats = LookupSplitStats();
    auto splitAt = split ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto splitLap = [&](std::atomic<std::uint64_t>& into) {
        if (!split) return;
        const auto now = std::chrono::steady_clock::now();
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - splitAt).count()), std::memory_order_relaxed);
        splitAt = now;
    };
    APS5_LOG_CHARS_OUT_DEBUG("Creating ShaderResources");
    // A recordable draw whose stages' compiled content repeats an earlier one binds that build's
    // descriptor set when it is still valid (see ResourceCache; the dispatch path does the same).
    // Only recordable draws take part: a synchronous draw writes its resources back, which a shared
    // object must never do. Revalidate proves a hit by texture identity, which needs the texture
    // caches. Push constants still come from this draw's stages. APS5_NO_DRAW_RESOURCE_CACHE=1
    // builds every draw's resources as before.
    static const bool noDrawResourceCache = std::getenv("APS5_NO_DRAW_RESOURCE_CACHE") != nullptr;
    static const bool noTextureCache = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    // The render target and index ranges stay out of the key (see DrawResourceKey); a hit repeats
    // the alias checks instead. Debug aid: APS5_NO_DRAW_KEY_TRIM=1 keys them as before.
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    resolved.cacheable = recordable && !noDrawResourceCache && !noTextureCache && std::all_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->PipelineVariantId() != 0; });
    auto speculation = ForkLogged() != nullptr ? std::shared_ptr<DrawSpeculation>{} : std::exchange(currentSpeculation, nullptr);
    if (speculation != nullptr && speculation->kind != DrawSpeculation::Kind::None && (!resolved.cacheable || speculation->device != context.device)) {
        CountSpeculationMiss(resolved.cacheable ? DrawSpeculation::Miss::Device : DrawSpeculation::Miss::Uncacheable, true);
        speculation = nullptr;
    }
    if (resolved.cacheable) {
        std::shared_ptr<ShaderResources> cached;
        std::shared_ptr<ShaderResources> speculatedUse;
        bool speculatedProof = false;
        std::optional<std::vector<ShaderResources::MovedBuffer>> verifyMoved;
        if (speculation != nullptr && speculation->kind != DrawSpeculation::Kind::None) {
            if (DrawTokensEnabled()) ScreenTokenWindow(*speculation);
            resolved.contentKey = std::move(speculation->key);
            if (VerifyDrawSpeculation() && resolved.contentKey != DrawResourceKey(context, shaders, state.color, draw.indexAddress, indexBytes, !trimKey)) {
                std::fprintf(stderr, "[spec] APS5_VERIFY_DRAW_SPECULATION: a speculation's resource key differs from the committer's\n");
                std::fflush(stderr);
                std::abort();
            }
            splitLap(splitStats.keyNs);
            cached = SharedResourceCache().Find(resolved.contentKey);
            splitLap(splitStats.findNs);
            if (cached == nullptr || cached != speculation->source) {
                CountSpeculationMiss(DrawSpeculation::Miss::Identity, true);
            } else if (speculation->kind == DrawSpeculation::Kind::Address) {
                if (VerifyDrawSpeculation()) verifyMoved = cached->MovedBuffersNow(shaders, Recorder::Active());
                speculatedUse = ShaderResources::UseAddressSpeculation(cached, *speculation);
                if (speculatedUse == nullptr) CountSpeculationMiss(speculation->miss, true);
            } else if (Recorder::Active() == nullptr) {
                CountSpeculationMiss(DrawSpeculation::Miss::NoRecorder, true);
            } else if (const auto miss = cached->ProveSpeculated(speculation->proof); miss != DrawSpeculation::Miss::None) {
                CountSpeculationMiss(miss, true);
            } else {
                speculatedProof = true;
            }
        } else {
            resolved.contentKey = prebuilt != nullptr && prebuilt->keyed ? std::move(prebuilt->key) : DrawResourceKey(context, shaders, state.color, draw.indexAddress, indexBytes, !trimKey);
            splitLap(splitStats.keyNs);
            cached = SharedResourceCache().Find(resolved.contentKey);
            splitLap(splitStats.findNs);
        }
        const auto sameMoved = [](std::span<const ShaderResources::MovedBuffer> left, std::span<const ShaderResources::MovedBuffer> right) {
            return std::equal(left.begin(), left.end(), right.begin(), right.end(), [](const ShaderResources::MovedBuffer& a, const ShaderResources::MovedBuffer& b) { return a.allocation == b.allocation && a.address == b.address && a.size == b.size && a.words == b.words; });
        };
        // A stream template made while the stream_ring switch was on is dropped once the switch is
        // off, so an A/B toggle compares against today's behaviour: the draw builds as before.
        if (cached != nullptr && cached->Streamed() && !ShaderResources::StreamRing()) {
            ForkTrap("resource invalidated");
            SharedResourceCache().Remove(resolved.contentKey, cached.get());
            countCache(&DrawProfile::cacheInvalidated);
            countCache(&DrawProfile::streamDropped);
            cached = nullptr;
            speculatedProof = false;
            speculatedUse = nullptr;
        }
        if (cached != nullptr && cached->AddressReusable() && ForkLogged() != nullptr) {
            auto speculated = forkSpeculation;
            if (speculated == nullptr || speculated->kind != DrawSpeculation::Kind::Address || speculated->device != context.device || speculated->source != cached || speculated->use == nullptr || speculated->key != resolved.contentKey) {
                speculated = std::make_shared<DrawSpeculation>();
                speculated->device = context.device;
                if (!ShaderResources::SpeculateAddressUse(cached, shaders, snapshots, *speculated)) ForkRefuse(addressSpeculationRefusal(speculated->miss));
            } else {
                countForkReuse();
            }
            if (trimKey && !speculated->aliasesOk && !(prebuilt != nullptr && prebuilt->aliasesOk)) CheckBufferAliases(shaders, state.color, draw.indexAddress, indexBytes);
            resolved.resources = speculated->use;
            resolved.speculated = std::move(speculated);
            resolved.speculatedSource = std::move(cached);
            outcome.kind = KindBdaHit;
            outcome.addressBased = true;
            countCache(&DrawProfile::cacheHits);
        } else if (cached != nullptr && cached->AddressReusable()) {
            auto miss = ShaderResources::AddressMiss::None;
            const bool speculatedHit = speculatedUse != nullptr;
            if (speculatedHit && VerifyDrawSpeculation()) {
                if (!verifyMoved.has_value() || !sameMoved(*verifyMoved, speculation->moved) || (speculation->movedImages.empty() && !cached->SameImageWords(shaders))) {
                    std::fprintf(stderr, "[spec] APS5_VERIFY_DRAW_SPECULATION: an accepted address speculation disagrees with the committer's moved buffers or image words\n");
                    std::fflush(stderr);
                    std::abort();
                }
            }
            auto use = speculatedHit ? std::move(speculatedUse) : ShaderResources::AddressUse(cached, shaders, snapshots, miss);
            splitLap(splitStats.proofNs);
            if (use != nullptr) {
                if (speculatedHit) Speculations().acceptedAddress.fetch_add(1, std::memory_order_relaxed);
                if (trimKey && !(speculatedHit && speculation->aliasesOk) && !(prebuilt != nullptr && prebuilt->aliasesOk)) CheckBufferAliases(shaders, state.color, draw.indexAddress, indexBytes);
                splitLap(splitStats.aliasNs);
                resolved.resources = std::move(use);
                outcome.kind = KindBdaHit;
                outcome.addressBased = true;
                countCache(&DrawProfile::cacheHits);
            } else if (miss == ShaderResources::AddressMiss::Moved || miss == ShaderResources::AddressMiss::MovedOutside || miss == ShaderResources::AddressMiss::Outside) {
                countCache(&DrawProfile::cacheMisses);
            } else {
                ForkTrap("resource invalidated");
                SharedResourceCache().Remove(resolved.contentKey, cached.get());
                countCache(&DrawProfile::cacheInvalidated);
            }
        } else if (cached != nullptr) {
            const bool valid = speculatedProof || (cached->SameImageWords(shaders) && cached->Revalidate(shaders));
            auto* recorder = Recorder::Active();
            std::optional<std::vector<ShaderResources::MovedBuffer>> moved;
            if (valid && recorder != nullptr && speculatedProof) {
                if (cached->MovedStillFree(speculation->moved, recorder, &speculation->movedProof)) {
                    if (VerifyDrawSpeculation()) {
                        const auto now = cached->MovedReadOnlyBuffers(shaders, *recorder);
                        if (!now.has_value() || !sameMoved(*now, speculation->moved) || !cached->SameImageWords(shaders)) {
                            std::fprintf(stderr, "[spec] APS5_VERIFY_DRAW_SPECULATION: an accepted template speculation disagrees with the committer's moved buffers or image words\n");
                            std::fflush(stderr);
                            std::abort();
                        }
                    }
                    moved = std::move(speculation->moved);
                    Speculations().acceptedTemplate.fetch_add(1, std::memory_order_relaxed);
                } else {
                    CountSpeculationMiss(DrawSpeculation::Miss::MovedRecheck, true);
                }
            } else if (valid && recorder != nullptr) {
                moved = cached->MovedReadOnlyBuffers(shaders, *recorder);
            }
            if (split) {
                splitStats.templates.fetch_add(1, std::memory_order_relaxed);
                const auto before = splitAt;
                splitLap(splitStats.proofNs);
                splitStats.templateNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(splitAt - before).count()), std::memory_order_relaxed);
            }
            if (moved.has_value()) {
                if (trimKey && !(speculatedProof && speculation->aliasesOk) && !(prebuilt != nullptr && prebuilt->aliasesOk)) CheckBufferAliases(shaders, state.color, draw.indexAddress, indexBytes);
                splitLap(splitStats.aliasNs);
                resolved.resources = std::move(cached);
                resolved.moved = std::move(*moved);
                outcome.kind = resolved.resources->Streamed() ? KindStreamHit : KindTemplateHit;
                countCache(&DrawProfile::cacheHits);
            } else if (valid) {
                countCache(&DrawProfile::cacheMisses);
                if (cached->Streamed()) countCache(&DrawProfile::streamRefused);
            } else {
                ForkTrap("resource invalidated");
                SharedResourceCache().Remove(resolved.contentKey);
                countCache(&DrawProfile::cacheInvalidated);
            }
        }
    } else {
        countCache(&DrawProfile::uncacheable);
    }
    timer.phase(PhaseLookup);
    if (split) {
        const auto count = splitStats.draws.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = splitStats.reported.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && splitStats.reported.compare_exchange_strong(last, now)) {
            const auto us = [&](std::atomic<std::uint64_t>& value) { return static_cast<double>(value.exchange(0)) / 1000.0 / static_cast<double>(std::max<std::uint64_t>(count, 1)); };
            splitStats.draws.store(0);
            const auto templates = splitStats.templates.exchange(0);
            const auto templateUs = templates != 0 ? static_cast<double>(splitStats.templateNs.exchange(0)) / 1000.0 / static_cast<double>(templates) : 0.0;
            std::fprintf(stderr, "[lookup-split] %llu templates at %.1f us (revalidate + moved buffers); ", static_cast<unsigned long long>(templates), templateUs);
            std::fprintf(stderr, "[lookup-split] %llu draws, us per draw: key %.1f find %.1f proof %.1f alias %.1f\n", static_cast<unsigned long long>(count), us(splitStats.keyNs), us(splitStats.findNs), us(splitStats.proofNs), us(splitStats.aliasNs));
        }
    }
    if (resolved.resources == nullptr) {
        ForkTrap("resource build");
        resolved.resources = std::make_shared<ShaderResources>(context, shaders, state.color, draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
        resolved.built = &resolved.resources->Timing();
        outcome.addressBased = resolved.resources->HoldsLease();
        outcome.kind = outcome.addressBased ? KindBda : KindBuild;
        if (resolved.cacheable) countCache(&DrawProfile::cacheMisses);
    }
    timer.phase(PhaseResources);
    APS5_LOG_CHARS_OUT_DEBUG("ShaderResources created");
    return resolved;
}

// A GPU-side indirect draw's records (Draw's recorded path; a recipe draw has none) as recordDraw
// takes them: the arguments, how they are read, their host imports and, on the CPU path, the
// records read.
struct IndirectRecord {
    const Pm4::DrawParameters::IndirectDraw* args = nullptr;
    IndirectDrawPath path = IndirectDrawPath::Gpu;
    const HostImport* argumentImport = nullptr;
    const HostImport* countImport = nullptr;
    std::span<const Pm4::DrawArguments> records;
    double readMs = 0;
};

// The draw commands of one draw: the vertex and index buffer binds, then the direct draw, the
// GPU-side indirect draw from `argumentBuffer` or the CPU-read records with the driver's rules.
void recordDrawCommands(const Context& context, VkCommandBuffer commands, const State& state, const Pm4::DrawParameters& draw, const DrawInputs& inputs, const IndirectRecord* indirect, VkBuffer argumentBuffer, VkDeviceSize argumentOffset, std::span<const CompiledShader> shaders = {}) {
    if (context.recorder != nullptr) context.recorder->NoteSampledDraw();
    if (GpuCheckpoints::Active().load(std::memory_order_relaxed)) {
        std::uint64_t vertex = 0, fragment = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) continue;
            if (shader.stage == ShaderRecompiler::ShaderStage::Fragment) fragment = shader.program->variantId;
            else vertex = shader.program->variantId;
        }
        char text[160] = {};
        int n = 0;
        if (!state.colors.empty()) {
            const auto& color = state.colors.front();
            n = std::snprintf(text, sizeof(text), "fb layers %u; color 0x%llx %ux%u fmt %d mip %u/%u slice %u layers %u array %u surface 0x%llx %ux%u", state.renderLayers, static_cast<unsigned long long>(color.address), color.extent.width, color.extent.height, static_cast<int>(color.format), color.mip, color.mipCount, color.depthSlice, color.layers, color.arraySize, static_cast<unsigned long long>(color.surfaceAddress), color.surfaceExtent.width, color.surfaceExtent.height);
        }
        if (state.depth && n >= 0 && static_cast<std::size_t>(n) < sizeof(text)) std::snprintf(text + n, sizeof(text) - static_cast<std::size_t>(n), "; depth 0x%llx %ux%u layers %u", static_cast<unsigned long long>(state.depth->address), state.depth->extent.width, state.depth->extent.height, state.depth->layers);
        GpuCheckpoints::Set(context, commands, "draw (vs, fs, extent)", vertex, fragment, (static_cast<std::uint64_t>(state.renderExtent.width) << 32) | state.renderExtent.height, text);
    }
    const auto* args = indirect != nullptr ? indirect->args : nullptr;
    if (state.stages.mesh && args != nullptr) {
        context.Function<PFN_vkCmdDrawMeshTasksIndirectEXT>("vkCmdDrawMeshTasksIndirectEXT")(commands, argumentBuffer, argumentOffset, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
        return;
    }
    if (state.stages.mesh) {
        APS5_LOG_OUT_DEBUG("vkCmdDrawMeshTasksEXT groups=%u instances=%u", inputs.meshGroups, draw.instanceCount);
        context.Function<PFN_vkCmdDrawMeshTasksEXT>("vkCmdDrawMeshTasksEXT")(commands, inputs.meshGroups, draw.instanceCount, 1);
        return;
    }
    if (!inputs.vertexHandles.empty()) context.Resolved(&DeviceFunctions::cmdBindVertexBuffers, "vkCmdBindVertexBuffers")(commands, 0, static_cast<std::uint32_t>(inputs.vertexHandles.size()), inputs.vertexHandles.data(), inputs.vertexOffsets.data());
    if (draw.indexed) context.Resolved(&DeviceFunctions::cmdBindIndexBuffer, "vkCmdBindIndexBuffer")(commands, inputs.indexHandle, inputs.indexOffset, draw.indexSize == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    if (args == nullptr) {
        if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, draw.indexCount, draw.instanceCount, 0, static_cast<std::int32_t>(draw.firstVertex), draw.firstInstance);
        else context.Resolved(&DeviceFunctions::cmdDraw, "vkCmdDraw")(commands, draw.indexCount, draw.instanceCount, draw.firstVertex, draw.firstInstance);
    } else if (indirect->path == IndirectDrawPath::Gpu) {
        if (args->countIndirect) {
            if (draw.indexed) context.Function<PFN_vkCmdDrawIndexedIndirectCountKHR>("vkCmdDrawIndexedIndirectCountKHR")(commands, argumentBuffer, argumentOffset, indirect->countImport->buffer, args->countAddress - indirect->countImport->base, args->count, args->stride);
            else context.Function<PFN_vkCmdDrawIndirectCountKHR>("vkCmdDrawIndirectCountKHR")(commands, argumentBuffer, argumentOffset, indirect->countImport->buffer, args->countAddress - indirect->countImport->base, args->count, args->stride);
        } else if (args->count <= 1 || context.multiDrawIndirect) {
            if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, argumentBuffer, argumentOffset, args->count, args->stride);
            else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, argumentBuffer, argumentOffset, args->count, args->stride);
        } else {
            for (std::uint32_t record = 0; record < args->count; ++record) {
                const auto offset = argumentOffset + static_cast<VkDeviceSize>(record) * args->stride;
                if (draw.indexed) context.Resolved(&DeviceFunctions::cmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect")(commands, argumentBuffer, offset, 1, args->stride);
                else context.Resolved(&DeviceFunctions::cmdDrawIndirect, "vkCmdDrawIndirect")(commands, argumentBuffer, offset, 1, args->stride);
            }
        }
    } else {
        // The CPU-read records with the driver's rules: an InPlace dimension takes the record's
        // value, a Constant one (the CP writes nowhere) the constant.
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        for (const auto& record : indirect->records) {
            if (record.count == 0 || record.instances == 0) continue;
            const auto firstInstance = args->instanceRule == Rule::InPlace ? record.firstInstance : args->instanceConstant;
            if (draw.indexed) {
                // The CP clamps the index range to INDEX_BUFFER_SIZE (the bound copy).
                if (record.firstVertexOrIndex >= draw.indexCount) continue;
                const auto vertexOffset = args->vertexRule == Rule::InPlace ? record.vertexOffset : args->vertexConstant;
                context.Resolved(&DeviceFunctions::cmdDrawIndexed, "vkCmdDrawIndexed")(commands, std::min(record.count, draw.indexCount - record.firstVertexOrIndex), record.instances, record.firstVertexOrIndex, static_cast<std::int32_t>(vertexOffset), firstInstance);
            } else {
                const auto firstVertex = args->vertexRule == Rule::InPlace ? record.firstVertexOrIndex : args->vertexConstant;
                context.Resolved(&DeviceFunctions::cmdDraw, "vkCmdDraw")(commands, record.count, record.instances, firstVertex, firstInstance);
            }
        }
    }
}

// The GPU-side records of an indirect draw, recorded outside the render pass: the stores that
// produced them (a dispatch in place, a fill, the host) precede the indirect read, as for
// DISPATCH_INDIRECT; a Constant dimension is rewritten in a scratch copy (vkCmdDrawIndirect reads
// the record's dword as the first vertex / instance, and the constant is what the fixed-function
// fetch must start at). The scratch is kept until the batch completed. Returns whether a copy was
// rewritten; `argumentBuffer`/`argumentOffset` receive where the draw reads the records.
bool recordIndirectArguments(const Context& context, VkCommandBuffer commands, Recorder* recorder, bool recorded, const IndirectRecord& indirect, std::unique_ptr<DeviceBuffer>& scratch, VkBuffer& argumentBuffer, VkDeviceSize& argumentOffset, const std::function<void(std::uint32_t)>& countBarrier) {
    using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
    const auto* args = indirect.args;
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT);
    countBarrier(1);
    if (recorded) {
        // Read in place from the import when the batch runs (a synchronous draw waits for its own).
        recorder->NotePendingRead(args->arguments, static_cast<std::size_t>(args->RangeBytes()), Recorder::ReadKind::Indirect);
        if (args->countIndirect) recorder->NotePendingRead(args->countAddress, 4, Recorder::ReadKind::Indirect);
    }
    argumentBuffer = indirect.argumentImport->buffer;
    argumentOffset = args->arguments - indirect.argumentImport->base;
    const bool rewritten = args->vertexRule == Rule::Constant || args->instanceRule == Rule::Constant;
    if (!rewritten) return false;
    const auto bytes = static_cast<std::size_t>(args->RangeBytes());
    scratch = std::make_unique<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    CopyBuffer(context, commands, argumentBuffer, argumentOffset, scratch->Handle(), 0, bytes);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    countBarrier(1);
    const auto update = context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer");
    for (std::uint32_t record = 0; record < args->count; ++record) {
        const VkDeviceSize base = static_cast<VkDeviceSize>(record) * args->stride;
        if (args->vertexRule == Rule::Constant) update(commands, scratch->Handle(), base + args->VertexDwordOffset(), 4, &args->vertexConstant);
        if (args->instanceRule == Rule::Constant) update(commands, scratch->Handle(), base + args->InstanceDwordOffset(), 4, &args->instanceConstant);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
    countBarrier(1);
    argumentBuffer = scratch->Handle();
    argumentOffset = 0;
    return true;
}

std::shared_ptr<Buffer> recordMeshArguments(const Context& context, VkCommandBuffer commands, Recorder* recorder, bool recorded, const State& state, const Pm4::DrawParameters& draw, const IndirectRecord& indirect, const std::function<void(std::uint32_t)>& countBarrier) {
    const auto* args = indirect.args;
    const auto rules = MeshArgumentRulesFor(context, *state.stages.mesh, draw.indexCount);
    auto arguments = std::make_shared<Buffer>(context, ShaderRecompiler::MeshArgumentBytes, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    if (indirect.path == IndirectDrawPath::Gpu && recorder != nullptr && indirect.argumentImport->address != 0) {
        const std::array<std::uint32_t, 7> words{rules.indexCount, rules.inputSize, rules.step, rules.primitivesPerGroup, rules.maxGroups, rules.maxInstances, rules.maxTotal};
        if (recorder->RecordMeshArguments(commands, indirect.argumentImport->address + (args->arguments - indirect.argumentImport->base), arguments->DeviceAddress(), words)) {
            countBarrier(2);
            if (recorded) recorder->NotePendingRead(args->arguments, static_cast<std::size_t>(args->RangeBytes()), Recorder::ReadKind::Indirect);
            return arguments;
        }
    }
    const auto record = !indirect.records.empty() ? indirect.records.front() : Pm4::ReadDrawArguments(*args, 0);
    const auto resolved = ResolveMeshArguments(record, rules);
    std::memcpy(arguments->Bytes().data(), &resolved, sizeof(resolved));
    return arguments;
}

// Debug aid: APS5_CHECK_INDIRECT_ARGS=1 reads the records of a GPU-side draw back on the CPU once
// its batch completed (the GPU has finished writing them by then) and prints the first 16, counting
// records whose Constant dimension holds a value the hardware would have ignored. Empty otherwise.
std::function<void()> indirectRecordCheck(const IndirectRecord* indirect) {
    static const bool checkArguments = std::getenv("APS5_CHECK_INDIRECT_ARGS") != nullptr;
    if (indirect == nullptr || indirect->args == nullptr || indirect->path != IndirectDrawPath::Gpu || !checkArguments) return {};
    return [args = *indirect->args] {
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        static std::atomic<std::uint64_t> printed{0};
        static std::atomic<std::uint64_t> ignoredValues{0};
        static std::atomic<std::uint64_t> checked{0};
        try {
            for (std::uint32_t record = 0; record < args.count; ++record) {
                const auto arguments = Pm4::ReadDrawArguments(args, record);
                const bool ignoredVertex = args.vertexRule == Rule::Constant && (args.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex) != 0;
                const bool ignoredInstance = args.instanceRule == Rule::Constant && arguments.firstInstance != 0;
                ++checked;
                if (ignoredVertex || ignoredInstance) ++ignoredValues;
                if (printed.fetch_add(1) < 16) std::fprintf(stderr, "[draw] indirect record 0x%llx: count %u instances %u first %u vertexOffset %u startInstance %u (vertex %s, instance %s)\n", static_cast<unsigned long long>(args.arguments + static_cast<std::uint64_t>(record) * args.stride), arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance, args.vertexRule == Rule::Constant ? "const: record value ignored" : "in-place", args.instanceRule == Rule::Constant ? "const: record value ignored" : "in-place");
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[draw] indirect record read-back failed: %s\n", error.what());
        }
        if (checked % 64 == 0) std::fprintf(stderr, "[draw] indirect records checked %llu, with a value in a Constant dimension %llu\n", static_cast<unsigned long long>(checked.load()), static_cast<unsigned long long>(ignoredValues.load()));
    };
}

// Everything a recorded draw uses lives until its batch completes.
struct Kept {
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Framebuffer> framebuffer;
    std::shared_ptr<Buffer> indices;
    std::vector<std::shared_ptr<Buffer>> vertexBuffers;
    std::vector<std::shared_ptr<StorageTexture>> targets;
    std::unique_ptr<DeviceBuffer> scratch;
};

// The completion side of a recorded draw: the kept objects, the record check, the lease outcome,
// the GPU write notes, the copied-buffer write-back (listed in DrawCopiedWriters or run as a
// completion action) and the targets marked dirty.
void keepRecordedDraw(Recorder& recorder, const std::shared_ptr<ShaderResources>& resources, std::shared_ptr<Pipeline> pipeline, std::shared_ptr<Framebuffer> framebuffer, DrawInputs& inputs, std::vector<std::shared_ptr<StorageTexture>> targets, std::unique_ptr<DeviceBuffer> scratch, std::function<void()> checkRecords, bool listed, bool completion, const DrawOutcome& outcome) {
    auto kept = std::make_shared<Kept>();
    kept->resources = resources;
    kept->pipeline = std::move(pipeline);
    kept->framebuffer = std::move(framebuffer);
    kept->indices = std::move(inputs.indices);
    kept->vertexBuffers = std::move(inputs.vertexBuffers);
    kept->scratch = std::move(scratch);
    kept->targets = std::move(targets);
    const auto& residents = kept->targets;
    recorder.Keep(kept);
    if (checkRecords) recorder.OnComplete(std::move(checkRecords));
    // Counted for the [address-sync] line: a lease released by the completion (the pin waiter
    // finishes the recorder up to the open batch, which holds the kept resources and gets the
    // next serial), or waited for by the caller (any wait releases it at once).
    if (resources->HoldsLease()) CountLeaseOutcome(outcome.waited, outcome.waited ? 0 : recorder.Submissions() + 1);
    // Every use of a (possibly shared) resources object registers its GPU writes anew.
    Recorder::SetNoteProgram(1);
    resources->MarkGpuWrites(recorder);
    Recorder::SetNoteProgram(0);
    // The write-back (fault check, copied buffers) runs when the batch completed; a fault is
    // reported by the recorder ("deferred write-back failed") instead of thrown out of the draw.
    if (listed) {
        ForkQuiesce("copied writers");
        // As VulkanDevice::dispatch lists its copied writers: delisted before the write-back
        // (one that fails must not keep indirect dispatches on the CPU), listed after the
        // registration (a throw there leaves nothing behind).
        auto writers = DrawCopiedWriters();
        recorder.OnComplete([resources, writers] {
            writers->erase(std::remove(writers->begin(), writers->end(), resources), writers->end());
            resources->WriteBackBuffers();
        });
        writers->push_back(resources);
    } else if (completion) {
        recorder.OnComplete([resources] { resources->WriteBackBuffers(); });
    }
    for (const auto& resident : residents) {
        if (resident != nullptr) resident->MarkDirty();
    }
}

// What recordDraw records from: the recorder, the resources, the pipeline and framebuffer, the
// resident targets with their views, the indirect records (null for a direct draw), how the
// completion work is handled (`listed`, `completion`, `waited`: see Draw) and, from a recipe, the
// push constant block (else assembled from the stages).
struct RecordedDraw {
    Recorder* recorder = nullptr;
    std::shared_ptr<ShaderResources> resources;
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Framebuffer> framebuffer;
    std::span<const VkImageView> targetViews;
    std::vector<std::shared_ptr<StorageTexture>> targets;
    std::vector<ProxyTarget> proxies;
    const IndirectRecord* indirect = nullptr;
    std::span<const ShaderResources::MovedBuffer> moved;
    bool listed = false;
    bool completion = false;
    bool waited = false;
    const std::array<std::byte, PipelinePushConstantBytes>* pushBytes = nullptr;
    VkShaderStageFlags pushStages = 0;
    const AssembledPush* assembled = nullptr;
};

void pushDrawConstants(const Pipeline& pipeline, VkCommandBuffer commands, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const ShaderResources& resources, const std::array<std::byte, PipelinePushConstantBytes>* bytes, VkShaderStageFlags stages, VkDeviceAddress meshArguments = 0, const AssembledPush* assembled = nullptr) {
    auto block = bytes != nullptr ? *bytes : assembled != nullptr ? assembled->bytes : AssemblePushConstants(shaders);
    if (bytes == nullptr) {
        resources.PatchPushConstants(block);
        stages = assembled != nullptr ? assembled->stages : PushConstantStages(shaders);
    }
    if (!state.stages.mesh) {
        if (std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return LayerLimited(shader); })) {
            const std::uint32_t layerLimit = state.renderLayers > 0 ? state.renderLayers - 1u : 0u;
            std::memcpy(block.data() + ShaderRecompiler::LayerLimitPushBytes, &layerLimit, sizeof(layerLimit));
            stages |= PushConstantStages(shaders);
        }
        pipeline.PushConstants(commands, stages, block);
        return;
    }
    const auto firstVertex = draw.indirect ? draw.indirect->vertexConstant : draw.firstVertex;
    const auto firstInstance = draw.indirect ? draw.indirect->instanceConstant : draw.firstInstance;
    const auto layerLimit = state.renderLayers > 0 ? state.renderLayers - 1u : 0u;
    const std::array<std::uint32_t, ShaderRecompiler::MeshDrawPushBytes / 4> words{draw.indexCount, firstVertex, firstInstance, draw.indexed ? draw.indexSize : 0u, static_cast<std::uint32_t>(meshArguments), static_cast<std::uint32_t>(meshArguments >> 32u), layerLimit};
    static_assert(ShaderRecompiler::MeshLayerLimitDword == 6 && ShaderRecompiler::MeshDrawPushBytes / 4 == 7);
    static_assert(ShaderRecompiler::MeshDrawPushOffsetBytes + ShaderRecompiler::MeshDrawPushBytes == PipelinePushConstantBytes);
    std::memcpy(block.data() + ShaderRecompiler::MeshDrawPushOffsetBytes, words.data(), sizeof(words));
    pipeline.PushConstants(commands, stages | VK_SHADER_STAGE_MESH_BIT_EXT, block);
}

// The record of a draw whose targets are all resident (`lean`: recorded into the device's open
// batch, rendering in the general layout): one memory barrier before the pass (earlier writes to
// the attachments and the draw's inputs visible to it) and one after it, recorded by the recorder
// when the pass ends (Recorder::LeaveRenderPassOpen), so consecutive draws into the same
// attachments with nothing recorded between them share one pass; the host-write and host-read
// barriers are implicit in the submission and the fence. Then the kept objects, the completion
// work and, for a waited-for draw, the recorder sync.
bool CaptureInputsEnabled() {
    static const bool enabled = std::getenv("APS5_CAPTURE_INPUTS") != nullptr;
    Require(!enabled || CaptureTrace::Enabled(), "APS5_CAPTURE_INPUTS requires APS5_CAPTURE_TRACE");
    return enabled;
}

void captureInputs(const Context& context, Recorder& recorder, VkCommandBuffer commands, const ShaderResources& resources, const std::shared_ptr<ShaderResources::DrawBindings>& bindings, std::uint64_t target) {
    struct Sample {
        std::uint64_t address;
        std::size_t offset;
        std::vector<std::byte> expected;
        VkBuffer source;
        VkDeviceSize sourceOffset;
    };
    static unsigned long long nextDraw = 0;
    const auto draw = ++nextDraw;
    const auto batch = static_cast<unsigned long long>(recorder.Submissions() + 1);
    std::vector<Sample> samples;
    std::size_t total = 0;
    const auto addSample = [&](std::uint64_t address, std::size_t bytes, VkBuffer source, VkDeviceSize offset, const std::byte* expected) {
        if (bytes == 0 || bytes > 512) return;
        Require(offset % 4 == 0 && bytes % 4 == 0, "capture input is not aligned for a Vulkan buffer copy");
        Require(total + bytes <= 65536, "capture inputs exceed 64 KiB per draw");
        Sample sample{address, total, std::vector<std::byte>(bytes), source, offset};
        std::memcpy(sample.expected.data(), expected, bytes);
        total += bytes;
        samples.push_back(std::move(sample));
    };
    if (bindings != nullptr) {
        for (const auto& snapshot : bindings->snapshots) {
            const auto bytes = snapshot.Bytes();
            addSample(snapshot.address, bytes.size(), snapshot.buffer->Handle(), snapshot.offset, bytes.data());
        }
    }
    for (const auto& [begin, end] : resources.InPlaceReads()) {
        Require(end >= begin, "invalid capture input range");
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (bytes == 0 || bytes > 512) continue;
        if (bindings != nullptr && std::any_of(bindings->snapshots.begin(), bindings->snapshots.end(), [&](const auto& snapshot) { return snapshot.address < end && begin < snapshot.address + snapshot.bytes; })) continue;
        if (resources.WritesOverlap(begin, bytes) || recorder.PendingWriteOverlaps(begin, bytes)) {
            CaptureTrace::Log("input-skip draw=%llu batch=%llu address=%llx bytes=%zu reason=gpu-writer", draw, batch, static_cast<unsigned long long>(begin), bytes);
            continue;
        }
        const auto* imported = HostImportFor(context, begin, bytes);
        Require(imported != nullptr, "capture input has no host import");
        addSample(begin, bytes, imported->buffer, begin - imported->base, reinterpret_cast<const std::byte*>(begin));
    }
    CaptureTrace::Log("input-capture draw=%llu batch=%llu target=%llx ranges=%zu bytes=%zu", draw, batch, static_cast<unsigned long long>(target), samples.size(), total);
    if (samples.empty()) return;
    auto readback = std::make_shared<Buffer>(context, total, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    const auto copy = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    for (const auto& sample : samples) {
        const VkBufferCopy region{sample.sourceOffset, sample.offset, sample.expected.size()};
        copy(commands, sample.source, readback->Handle(), 1, &region);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
    recorder.OnComplete([draw, batch, target, readback, samples = std::move(samples)] {
        const auto actual = readback->Bytes();
        std::size_t differences = 0;
        for (const auto& sample : samples) {
            std::size_t changed = 0;
            for (std::size_t index = 0; index < sample.expected.size(); ++index) {
                const auto gpu = actual[sample.offset + index];
                if (sample.expected[index] == gpu) continue;
                if (changed < 16) CaptureTrace::Log("input-byte draw=%llu batch=%llu address=%llx offset=%zu cpu=%02x gpu=%02x", draw, batch, static_cast<unsigned long long>(sample.address), index, std::to_integer<unsigned>(sample.expected[index]), std::to_integer<unsigned>(gpu));
                ++changed;
            }
            if (changed != 0) {
                ++differences;
                CaptureTrace::Log("input-mismatch draw=%llu batch=%llu target=%llx address=%llx bytes=%zu changed=%zu", draw, batch, static_cast<unsigned long long>(target), static_cast<unsigned long long>(sample.address), sample.expected.size(), changed);
            }
        }
        CaptureTrace::Log("input-result draw=%llu batch=%llu ranges=%zu mismatched=%zu", draw, batch, samples.size(), differences);
    });
}

struct TargetMemo {
    Recorder* recorder = nullptr;
    std::uint64_t passKey = 0;
    std::uint64_t epoch = 0;
    std::vector<ColorTarget> colors;
    std::vector<std::weak_ptr<StorageTexture>> targets;
};

TargetMemo& ThreadTargetMemo() {
    static thread_local TargetMemo memo;
    return memo;
}

bool SameColorTarget(const ColorTarget& a, const ColorTarget& b) {
    return a.address == b.address && a.extent.width == b.extent.width && a.extent.height == b.extent.height && a.format == b.format && a.bytes == b.bytes && a.componentMapping == b.componentMapping && a.tileMode == b.tileMode && a.elementBytes == b.elementBytes && a.packing == b.packing && a.dccAddress == b.dccAddress && a.dccAlphaOnMsb == b.dccAlphaOnMsb && a.dccPipeAligned == b.dccPipeAligned && a.cmaskAddress == b.cmaskAddress && a.cmaskBytes == b.cmaskBytes && a.surfaceAddress == b.surfaceAddress && a.surfaceExtent.width == b.surfaceExtent.width && a.surfaceExtent.height == b.surfaceExtent.height && a.mipCount == b.mipCount && a.mip == b.mip && a.mipTail == b.mipTail && a.clearWords == b.clearWords && a.depth == b.depth && a.depthSlice == b.depthSlice && a.layers == b.layers && a.arraySize == b.arraySize;
}

std::shared_ptr<StorageTexture> MemoizedTarget(std::size_t index, const State& state) {
    static const bool enabled = std::getenv("APS5_NO_TARGET_MEMO") == nullptr;
    auto& memo = ThreadTargetMemo();
    auto* recorder = Recorder::Active();
    if (!enabled || recorder == nullptr || memo.recorder != recorder || memo.epoch == 0 || memo.epoch != GuestMemory::ThreadCollectEpoch()) return nullptr;
    if (memo.colors.size() != state.colors.size() || index >= memo.targets.size() || (ForkLogged() == nullptr && !recorder->ContinuesRenderPass(memo.passKey))) return nullptr;
    if (!SameColorTarget(memo.colors[index], state.colors[index])) return nullptr;
    return memo.targets[index].lock();
}

void RememberTargets(Recorder* recorder, std::uint64_t passKey, const State& state, const std::vector<std::shared_ptr<StorageTexture>>& targets) {
    auto& memo = ThreadTargetMemo();
    memo.recorder = recorder;
    memo.passKey = passKey;
    memo.epoch = GuestMemory::ThreadCollectEpoch();
    // Consecutive draws of a pass name the same targets: skip the copies when nothing changed.
    const bool sameColors = memo.colors.size() == state.colors.size() && std::equal(memo.colors.begin(), memo.colors.end(), state.colors.begin(), SameColorTarget);
    if (!sameColors) memo.colors = state.colors;
    bool sameTargets = sameColors && memo.targets.size() == targets.size();
    for (std::size_t index = 0; sameTargets && index < targets.size(); ++index) sameTargets = !memo.targets[index].owner_before(targets[index]) && !targets[index].owner_before(memo.targets[index]) && !memo.targets[index].expired();
    if (!sameTargets) memo.targets.assign(targets.begin(), targets.end());
}

// APS5_NO_RECIPE_TARGET_MEMO=1: recipe draws refresh their targets on every draw.
bool RecipeTargetMemo() {
    static const bool enabled = std::getenv("APS5_NO_RECIPE_TARGET_MEMO") == nullptr;
    return enabled;
}

struct RecordPlan {
    std::uint64_t passKey = 0;
    bool readsTarget = false;
    std::shared_ptr<ShaderResources::DrawBindings> drawBindings;
};

RecordPlan planRecord(const State& state, const RecordedDraw& record) {
    RecordPlan plan;
    // The pass is named by its attachment views (stable while the kept targets live, so unique
    // within the open batch) and the extent. The previous draw's pass is continued only when this
    // draw neither reads its attachments (a barrier would be owed, which no pass allows) nor
    // records anything outside a pass (an indirect draw's argument barrier and scratch copies).
    std::uint64_t passKey = 14695981039346656037ull;
    const auto mix = [&](std::uint64_t value) {
        passKey ^= value;
        passKey *= 1099511628211ull;
    };
    for (const auto view : record.targetViews) mix(reinterpret_cast<std::uint64_t>(view));
    if (state.blends.size() != state.colors.size()) {
        mix(state.blends.size());
        for (const auto& color : state.colors) mix(color.exportIndex);
    }
    mix(state.renderExtent.width);
    mix(state.renderExtent.height);
    plan.passKey = passKey;
    const auto& resources = *record.resources;
    plan.readsTarget = std::any_of(record.targets.begin(), record.targets.end(), [&](const std::shared_ptr<StorageTexture>& target) { return resources.ReadsImage(target.get()); });
    return plan;
}

std::function<bool(std::uint64_t, std::uint64_t)> recordTouches(const RecordedDraw& record) {
    const auto& resources = *record.resources;
    const auto* args = record.indirect != nullptr ? record.indirect->args : nullptr;
    const bool gpuIndirect = args != nullptr && record.indirect->path == IndirectDrawPath::Gpu;
    // A queued DCC key store over memory the draw writes or reads in place (unknown for an
    // address-based build), or over its GPU-side records, lands before it, as before a dispatch
    // (VulkanDevice::dispatch); the flush ends an open pass, so it comes before the decision.
    return [&resources, args, gpuIndirect](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (resources.WritesOverlap(begin, bytes) || resources.ReadsOverlap(begin, bytes)) return true;
        if (!gpuIndirect) return false;
        return (begin < args->arguments + args->RangeBytes() && args->arguments < end) || (args->countIndirect && begin < args->countAddress + 4 && args->countAddress < end);
    };
}

bool recordStoresDue(const RecordedDraw& record) {
    auto* recorder = record.recorder;
    const auto& resources = *record.resources;
    const auto touches = recordTouches(record);
    return (recorder->HasQueuedKeyStores() && (resources.HoldsLease() || recorder->AnyQueuedKeyStore(touches))) || (recorder->HasQueuedStores() && (resources.HoldsLease() || recorder->AnyQueuedStore(touches)));
}

void flushRecordStores(const RecordedDraw& record) {
    auto* recorder = record.recorder;
    auto& resources = *record.resources;
    const auto touches = recordTouches(record);
    if (recorder->HasQueuedKeyStores() && (resources.HoldsLease() || recorder->AnyQueuedKeyStore(touches))) recorder->FlushKeyStores();
    // A queued label store over such memory likewise (Recorder::RecordStore).
    if (recorder->HasQueuedStores() && (resources.HoldsLease() || recorder->AnyQueuedStore(touches))) recorder->FlushStores();
}

void emitRecord(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawInputs& inputs, RecordedDraw& record, const RecordPlan& plan, DrawOutcome& outcome, DrawTimer& timer, double& ownWaitedMs) {
    auto* recorder = record.recorder;
    auto& resources = *record.resources;
    const auto* args = record.indirect != nullptr ? record.indirect->args : nullptr;
    const bool gpuIndirect = args != nullptr && record.indirect->path == IndirectDrawPath::Gpu;
    using CommandClass = Recorder::CommandClass;
    const auto countBarrier = [&](std::uint32_t count) { Recorder::CountBarriers(CommandClass::Draw, count); };
    const auto passKey = plan.passKey;
    const bool readsTarget = plan.readsTarget;
    const auto& drawBindings = plan.drawBindings;
    const bool capture = CaptureInputsEnabled();
    const bool meshIndirect = state.stages.mesh && args != nullptr;
    const bool continued = !capture && !readsTarget && !gpuIndirect && !meshIndirect && recorder->ContinuesRenderPass(passKey);
    outcome.passContinued = continued;
    outcome.passBegun = !continued;
    VkAccessFlags covered = 0;
    bool elided = false;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (continued) {
        commands = recorder->CommandsInRenderPass();
    } else if (Recorder::BarrierTracking()) {
        auto reads = resources.TrackedReads();
        reads.insert(reads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
        if (gpuIndirect) {
            reads.emplace_back(args->arguments, args->arguments + args->RangeBytes());
            if (args->countIndirect) reads.emplace_back(args->countAddress, args->countAddress + 4);
        }
        auto images = resources.StorageImages();
        for (const auto& target : record.targets) images.emplace_back(target->Image(), true);
        const bool attachmentsKnown = !state.depth && record.targets.size() == state.colors.size();
        const bool keepBarriers = capture || meshIndirect || !record.proxies.empty();
        // An address-based draw that only reads through addresses is tracked as reading all memory
        // (live `lease_reads`); one storing through them stays conservative.
        const bool addressStores = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr || shader.program->bdaWrites; });
        const bool leaseReads = resources.HoldsLease() && !addressStores && LiveFlag("lease_reads", false);
        Recorder::Access access{reads, resources.GpuWrites(), images, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, (resources.HoldsLease() && !leaseReads) || !attachmentsKnown, false, keepBarriers};
        access.readsAnyMemory = leaseReads;
        commands = recorder->Commands(CommandClass::Draw, access, elided, &covered);
    } else {
        commands = recorder->Commands(&covered);
    }
    if (capture) captureInputs(context, *recorder, commands, resources, drawBindings, record.targets.empty() || record.targets.front() == nullptr ? 0 : record.targets.front()->Descriptor().baseAddress);
    // The draw's [gputime] class range: from its first barrier to the pass's trailing barrier (a
    // continued draw lies inside its pass's range).
    static const bool timePasses = std::getenv("APS5_PROFILE_GPU_PASSES") != nullptr;
    std::uint64_t passTimingKey = Recorder::ClassKey(CommandClass::Draw);
    if (!timePasses && NvPerf::Ranging()) passTimingKey = (static_cast<std::uint64_t>(state.renderExtent.width & 0xffffu) << 48u) | (static_cast<std::uint64_t>(state.renderExtent.height & 0xffffu) << 32u) | 0x100;
    if (timePasses) {
        std::uint64_t fragment = 0;
        for (const auto& shader : shaders) {
            if (shader.stage == ShaderRecompiler::ShaderStage::Fragment && shader.program != nullptr) fragment = shader.program->PipelineVariantId();
        }
        passTimingKey = (static_cast<std::uint64_t>(state.renderExtent.width & 0xffffu) << 48u) | (static_cast<std::uint64_t>(state.renderExtent.height & 0xffffu) << 32u) | (fragment & 0xffffffffu) | 0x100;
        static std::mutex passMutex;
        static std::unordered_map<std::uint64_t, std::array<std::uint64_t, 3>> passDraws;
        static thread_local std::uint64_t openPass = 0;
        static auto passReport = std::chrono::steady_clock::now();
        std::lock_guard passLock(passMutex);
        if (!continued) openPass = passTimingKey;
        static std::unordered_set<std::uint64_t> dumpedPrograms;
        if (state.renderExtent.width >= 3840 && resources.HoldsLease()) {
            for (const auto& shader : shaders) {
                if (shader.program == nullptr || shader.program->spirv.empty() || !dumpedPrograms.insert(shader.program->PipelineVariantId()).second) continue;
                char path[160];
                std::snprintf(path, sizeof(path), "D:/ps5/work/spv/%s-%llx.spv", shader.stage == ShaderRecompiler::ShaderStage::Fragment ? "frag" : "vert", static_cast<unsigned long long>(shader.program->PipelineVariantId()));
                if (auto* file = std::fopen(path, "wb")) {
                    std::fwrite(shader.program->spirv.data(), sizeof(std::uint32_t), shader.program->spirv.size(), file);
                    std::fclose(file);
                }
            }
        }
        auto& counts = passDraws[openPass];
        counts[0] += continued ? 0 : 1;
        counts[1] += 1;
        counts[2] += resources.HoldsLease() ? 1 : 0;
        if (std::chrono::steady_clock::now() - passReport > std::chrono::seconds(10)) {
            passReport = std::chrono::steady_clock::now();
            std::vector<std::pair<std::uint64_t, std::array<std::uint64_t, 3>>> sorted(passDraws.begin(), passDraws.end());
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second[1] > b.second[1]; });
            std::string line = "[passes] draws per pass key (10 s; key passes draws address-based):";
            char item[96];
            for (std::size_t i = 0; i < sorted.size() && i < 30; ++i) {
                std::snprintf(item, sizeof(item), " 0x%llx %llu %llu %llu", static_cast<unsigned long long>(sorted[i].first), static_cast<unsigned long long>(sorted[i].second[0]), static_cast<unsigned long long>(sorted[i].second[1]), static_cast<unsigned long long>(sorted[i].second[2]));
                line += item;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
            passDraws.clear();
        }
    }
    // APS5_PROFILE_LOOKUP: [record-split] the record phase of continued draws by step.
    static const bool recordSplit = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    struct RecordSplit {
        std::array<std::atomic<std::uint64_t>, 6> ns{};
        std::atomic<std::uint64_t> draws{0};
        std::atomic<std::int64_t> reported{0};
    };
    static RecordSplit recordStats;
    auto recordLap = recordSplit && continued ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto recordMark = [&](std::size_t step) {
        if (!recordSplit || !continued) return;
        const auto now = std::chrono::steady_clock::now();
        recordStats.ns[step].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - recordLap).count()), std::memory_order_relaxed);
        recordLap = now;
    };
    const auto drawTiming = !continued ? recorder->BeginGpuTiming(passTimingKey) : Recorder::NoTiming;
    std::unique_ptr<DeviceBuffer> scratch;
    std::shared_ptr<Buffer> meshArguments;
    VkBuffer argumentBuffer = VK_NULL_HANDLE;
    VkDeviceSize argumentOffset = 0;
    bool rewritten = false;
    if (continued) {
        record.pipeline->Continue(commands, state);
    } else {
        for (const auto& proxy : record.proxies) {
            proxy.texture->RecordAttachmentProxyLoad(commands, VK_IMAGE_LAYOUT_GENERAL, proxy.mip, proxy.layer, proxy.layers);
            countBarrier(2);
        }
        if (elided) {
            Recorder::CountSkipped(CommandClass::Draw);
        } else if (Recorder::MergeDrawBarriers() && !capture && record.proxies.empty() && (covered & Recorder::DrawBarrierAccess) == Recorder::DrawBarrierAccess) {
            Recorder::CountMerged(CommandClass::Draw);
        } else {
            const VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, Recorder::DrawBarrierAccess};
            Recorder::DebugLabelBegin(context, commands, "draw-barrier");
            context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, 0, 1, &before, 0, nullptr, 0, nullptr);
            Recorder::DebugLabelEnd(context, commands);
            countBarrier(1);
        }
        APS5_LOG_CHARS_OUT_DEBUG("Upload barrier recorded");
        if (meshIndirect) {
            meshArguments = recordMeshArguments(context, commands, recorder, true, state, draw, *record.indirect, countBarrier);
            argumentBuffer = meshArguments->Handle();
        } else if (gpuIndirect) {
            rewritten = recordIndirectArguments(context, commands, recorder, true, *record.indirect, scratch, argumentBuffer, argumentOffset, countBarrier);
        }
        // Earlier recorded work (dispatches, the previous draw) wrote the images in the general
        // layout, which a lean draw renders in: no transitions.
        APS5_LOG_OUT_DEBUG("Beginning pipeline renderExtent=%ux%u", state.renderExtent.width, state.renderExtent.height);
        record.pipeline->Begin(commands, *record.framebuffer, state.renderExtent, state);
    }
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline Begin OK");
    recordMark(0);
    if (Recorder::FlatDebugLabels()) {
        // Name the stages so a GPU trace can tie each draw to its pipeline (variant ids match the
        // [pipeline-stats] lines of the same run).
        std::uint64_t fs = 0, vs = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) continue;
            if (shader.stage == ShaderRecompiler::ShaderStage::Fragment) fs = shader.program->PipelineVariantId();
            else if (vs == 0) vs = shader.program->PipelineVariantId();
        }
        char name[128];
        std::snprintf(name, sizeof(name), "%s fs=%llx vs=%llx n=%u i=%u rl=%d", continued ? "draw-cmds-continued" : "draw-cmds-new-pass", static_cast<unsigned long long>(fs), static_cast<unsigned long long>(vs), draw.indexCount, draw.instanceCount, state.rectList ? 1 : 0);
        Recorder::DebugLabelBegin(context, commands, name);
    }
    if (drawBindings != nullptr) {
        const auto set = drawBindings->allocation.set;
        context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, record.pipeline->Layout(), 0, 1, &set, 0, nullptr);
    } else {
        resources.Bind(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, record.pipeline->Layout());
    }
    APS5_LOG_CHARS_OUT_DEBUG("Resources bound");
    recordMark(1);
    pushDrawConstants(*record.pipeline, commands, state, draw, shaders, resources, record.pushBytes, record.pushStages, meshArguments != nullptr ? meshArguments->DeviceAddress() : 0, record.assembled);
    APS5_LOG_CHARS_OUT_DEBUG("Push constants recorded");
    recordMark(2);
    recordDrawCommands(context, commands, state, draw, inputs, record.indirect, argumentBuffer, argumentOffset, shaders);
    Recorder::DebugLabelEnd(context, commands);
    recordMark(3);
    if (meshArguments != nullptr) recorder->Keep(meshArguments);
    if (args != nullptr) CountIndirectDraw(record.indirect->path, record.indirect->readMs, rewritten);
    auto checkRecords = indirectRecordCheck(record.indirect);
    APS5_LOG_CHARS_OUT_DEBUG("Draw recorded");
    // The pass stays open for the next draw of these attachments; the recorder ends it (and the
    // draw class range) before anything else is recorded. A draw that wrote memory owes the next
    // one a barrier, so its pass cannot be continued.
    std::function<void(VkCommandBuffer)> storeProxies;
    if (!continued && !record.proxies.empty()) {
        storeProxies = [proxies = record.proxies](VkCommandBuffer passCommands) {
            for (const auto& proxy : proxies) {
                proxy.texture->RecordAttachmentProxyStore(passCommands, VK_IMAGE_LAYOUT_GENERAL, proxy.mip, proxy.layer, proxy.layers);
                Recorder::CountBarriers(Recorder::CommandClass::Draw, 2);
            }
        };
    }
    static const bool leaseWrites = std::getenv("APS5_LEASE_ENDS_PASS") != nullptr;
    bool writesMemory = resources.WritesMemory();
    if (writesMemory && !leaseWrites && resources.HoldsLease()) {
        const auto storageImages = resources.StorageImages();
        const bool storesThroughAddresses = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program == nullptr || shader.program->bdaWrites; });
        writesMemory = storesThroughAddresses || resources.HasCopiedWrites() || !resources.GpuWrites().empty() || std::any_of(storageImages.begin(), storageImages.end(), [](const auto& image) { return image.second; });
    }
    recordMark(4);
    recorder->LeaveRenderPassOpen(passKey, drawTiming, !writesMemory, std::move(storeProxies));
    if (state.colors.size() == record.targets.size()) RememberTargets(recorder, passKey, state, record.targets);
    recordMark(5);
    if (recordSplit && continued) {
        const auto count = recordStats.draws.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = recordStats.reported.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && recordStats.reported.compare_exchange_strong(last, now)) {
            recordStats.draws.store(0);
            const auto us = [&](std::size_t step) { return static_cast<double>(recordStats.ns[step].exchange(0)) / 1000.0 / static_cast<double>(count); };
            std::fprintf(stderr, "[record-split] %llu continued draws, us per draw: pipeline+dynamic state %.2f descriptor set %.2f push constants %.2f vertex/index+draw %.2f writes check %.2f leave+remember %.2f\n", static_cast<unsigned long long>(count), us(0), us(1), us(2), us(3), us(4), us(5));
        }
    }
    timer.phase(PhaseRecord);
    keepRecordedDraw(*recorder, record.resources, std::move(record.pipeline), std::move(record.framebuffer), inputs, std::move(record.targets), std::move(scratch), std::move(checkRecords), record.listed, record.completion, outcome);
    timer.phase(PhaseKeep);
    if (record.waited) {
        // The wait the dispatch path makes for such work (source 3, "address-based"): the
        // write-back runs before the next packet, so the CPU never reads copied results or
        // touches a leased allocation before they landed.
        Recorder::CountSync(3);
        const auto ownBefore = timer.profile ? Recorder::ThreadWaitedMs() : 0.0;
        recorder->Sync();
        if (timer.profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
        timer.phase(PhaseSync);
    }
}

void recordDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, DrawInputs& inputs, RecordedDraw& record, DrawOutcome& outcome, DrawTimer& timer, double& ownWaitedMs) {
    auto plan = planRecord(state, record);
    flushRecordStores(record);
    plan.drawBindings = record.resources->PrepareDrawBindings(*record.recorder, record.moved);
    emitRecord(context, state, draw, shaders, inputs, record, plan, outcome, timer, ownWaitedMs);
}

// The state with the outputs the pixel shader lacks masked: Vulkan leaves attachments a pixel
// shader has no output for undefined, so their writes are masked (shaders that only store to
// images or buffers keep their targets as they were). Empty when no mask must change.
std::optional<State> maskedState(const State& state, const std::set<std::uint32_t>& fragmentOutputs) {
    std::optional<State> masked;
    for (std::size_t index = 0; index < state.blends.size(); ++index) {
        if (fragmentOutputs.contains(static_cast<std::uint32_t>(index)) || state.blends[index].colorWriteMask == 0) continue;
        if (!masked.has_value()) masked = state;
        masked->blends[index].colorWriteMask = 0;
    }
    return masked;
}

// Debug aid: APS5_DUMP_TARGETS=<n> saves the first n renders of every color target as a raw file
// (u32 width, u32 height, u32 VkFormat, then tightly packed rows).
int DumpTargetLimit() {
    static const int dumpLimit = [] { const char* text = std::getenv("APS5_DUMP_TARGETS"); return text ? std::atoi(text) : 0; }();
    return dumpLimit;
}

// A draw whose targets are all resident is recorded into the device's open batch like a dispatch:
// the CPU does not wait for it, and the drain before it goes away (queue order and the barriers
// keep it behind earlier recorded work); see `recorded` in Draw for the buffers that still need a
// synchronous draw. Other draws keep their own synchronous batch. Debug aid: APS5_SYNC_DRAWS=1
// makes every draw synchronous.
bool RecordDraws() {
    static const bool recordDraws = std::getenv("APS5_SYNC_DRAWS") == nullptr;
    return recordDraws;
}

std::shared_ptr<DrawRecipe> makeDrawRecipe(const Context& context, const State& state, std::span<const CompiledShader> shaders, const std::shared_ptr<ShaderResources>& resources, const ResourceCache::Key& contentKey, const std::shared_ptr<Pipeline>& pipeline, const std::shared_ptr<Framebuffer>& framebuffer, const std::vector<std::shared_ptr<StorageTexture>>& owners, const std::vector<VkImageView>& targetViews, const DrawInputs& inputs, const std::optional<State>& masked, bool movedBuffers) {
    // The attachment views are the colors' then the depth view (Draw and BuildDraw push it last).
    if (targetViews.size() != owners.size() + (state.depth ? 1u : 0u) || owners.size() != state.colors.size()) return nullptr;
    auto recipe = std::make_shared<DrawRecipe>();
    recipe->depth = static_cast<bool>(state.depth);
    recipe->depthView = state.depth ? targetViews.back() : VK_NULL_HANDLE;
    if (recipe->depth && recipe->depthView == VK_NULL_HANDLE) return nullptr;
    recipe->movedBuffers = movedBuffers;
    recipe->device = context.device;
    recipe->templateRef = resources;
    recipe->key = contentKey;
    recipe->pipeline = pipeline;
    recipe->framebuffer = framebuffer;
    for (const auto& owner : owners) recipe->targets.emplace_back(owner);
    recipe->targetViews = targetViews;
    std::uint64_t passKey = 14695981039346656037ull;
    for (const auto view : targetViews) passKey = (passKey ^ reinterpret_cast<std::uint64_t>(view)) * 1099511628211ull;
    passKey = (passKey ^ state.renderExtent.width) * 1099511628211ull;
    passKey = (passKey ^ state.renderExtent.height) * 1099511628211ull;
    recipe->passKey = passKey;
    recipe->vertexInput = inputs.vertexInput;
    recipe->pushStages = PushConstantStages(shaders);
    if (recipe->pushStages != 0) {
        recipe->pushBytes = AssemblePushConstants(shaders);
        resources->PatchPushConstants(recipe->pushBytes);
    }
    recipe->masked = masked;
    recipe->fragmentOutputs = inputs.fragmentOutputs;
    recipe->shaderStages = inputs.shaderStages;
    return recipe;
}

void drawPrepared(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, DrawInputs& inputs, PerformanceTimer& timing, DrawTimer& timer, DrawOutcome& outcome, double waitedBefore) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // APS5_TRACE_DRAWS (together with APS5_PROFILE_DRAW; alone it only enables Driver.cpp's own
    // [draw] lines) additionally prints one line per draw with its phases and, for synchronous
    // draws, their inputs: per-draw string building at ~1400 draws per 10 s is itself a cost the
    // totals must not carry.
    static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
    const ShaderResources::BuildTiming* built = nullptr;
    double ownWaitedMs = 0;
    const auto report = [&](const char* suffix) { reportDrawEnd(state, timer, built, outcome, waitedBefore, ownWaitedMs, suffix); };
    const auto* args = draw.indirect ? &*draw.indirect : nullptr;
    const auto indexBytes = inputs.indexBytes;
    // One binding per color attachment. Tiled targets are detiled and retiled on the GPU in video memory;
    // only changed guest blocks are stored back.
    struct TargetBinding {
        ColorTarget color;
        bool gpuTiling = false;
        // Resident target: the surface's cached storage image is attached directly and marked dirty
        // afterwards, so nothing is copied in or out per draw.
        std::shared_ptr<StorageTexture> resident;
        bool proxied = false;
        // Linear pixels converted on the CPU, for targets the GPU detiler does not handle.
        std::unique_ptr<Buffer> transfer;
        // Guest bytes in host memory, and their device-local copies for the detiler.
        std::unique_ptr<Buffer> tiled;
        std::unique_ptr<DeviceBuffer> tiledDevice;
        std::unique_ptr<DeviceBuffer> linearDevice;
        std::vector<std::byte> original;
        TileMipLayout mip{};
        std::unique_ptr<RenderTarget> target;
        // APS5_DUMP_TARGETS: the rendered linear pixels, read back for target_<address>_<n>.raw.
        std::unique_ptr<Buffer> dump;
    };
    const int dumpLimit = DumpTargetLimit();
    static std::mutex dumpMutex;
    static std::map<std::uint64_t, int> dumped;
    std::vector<TargetBinding> targets(state.colors.size());
    std::vector<VkImageView> targetViews;
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        auto& binding = targets[index];
        binding.color = state.colors[index];
        const auto& color = binding.color;
        binding.gpuTiling = color.tileMode == ColorTileMode::RenderTarget && context.detiler != nullptr;
        APS5_LOG_OUT_DEBUG("Creating color target %zu address=0x%llx bytes=%llu extent=%ux%u", index, static_cast<unsigned long long>(color.address), static_cast<unsigned long long>(color.bytes), color.extent.width, color.extent.height);
        const ColorTargetLayout colorLayout(color.extent.width, color.extent.height, color.tileMode, color.elementBytes);
        constexpr VkBufferUsageFlags copies = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        timer.phase(PhaseSetup);
        // Debug aid: APS5_NO_RESIDENT_TARGETS=1 copies every target in and out again.
        static const bool residentTargets = std::getenv("APS5_NO_RESIDENT_TARGETS") == nullptr;
        static const bool residentLinear = std::getenv("APS5_NO_RESIDENT_LINEAR_TARGETS") == nullptr;
        const bool linearResident = residentLinear && color.tileMode == ColorTileMode::Linear && RegisteredReadableCovers(color.address, static_cast<std::size_t>(color.bytes));
        if ((binding.gpuTiling || ((color.tileMode == ColorTileMode::Standard4KB || linearResident) && context.detiler != nullptr)) && residentTargets) {
            // The lookup refreshes the image on every draw (StorageTexture::Refresh: FlushPending,
            // CollectWrites over the target's pages, the DCC key scan of TextureClearKeys, then
            // UnchangedSince). The page walk is skipped while the worker's collect epoch lasts
            // (GuestMemory::BumpCollectEpoch: ordering points of the queue, or every packet under
            // APS5_PACKET_EPOCH=1); the key scan runs on every lookup regardless. Lookups of a
            // millisecond or more are counted apart; the [texture] line says how many uploaded.
            const RefreshSiteScope site("render-target");
            binding.resident = MemoizedTarget(index, state);
            if (binding.resident == nullptr) binding.resident = refreshResidentTarget(context, state, color, outcome, profile, [&] {
                auto resident = CachedStorageSurface(context, SurfaceForTarget(color));
                Require(resident->Attachable(), "storage format cannot be a color attachment");
                Require(color.mipCount > 1 || color.depth > 1 || resident->GuestBytes() == colorLayout.Bytes() || (color.arraySize > 1 && resident->GuestBytes() == colorLayout.Bytes() * color.arraySize), "resident image layout differs from the color layout");
                return resident;
            });
        }
        if (binding.resident != nullptr) {
            timer.phase(PhaseReadTarget);
            binding.proxied = ColorProxyFormat(context, color) != VK_FORMAT_UNDEFINED;
            targetViews.push_back(binding.proxied ? binding.resident->AttachmentProxyView(color.blendProxy111110, color.mip, color.depthSlice, color.layers) : binding.resident->AttachmentView(color.format, color.mip, color.depthSlice, color.layers));
            continue;
        }
        materializeCmaskClear(context, color, nullptr);
        Require(!color.mipTail, "rendering into a packed mip tail needs the resident image of its surface");
        Require(color.depth == 1, "rendering into a 3D color target needs the resident image of its surface");
        if (color.layers > 1 || state.renderLayers > 1) {
            // The copied render target has one layer: a layered framebuffer over it makes the device
            // write the other layers outside the image (a device fault).
            char text[200];
            std::snprintf(text, sizeof(text), "AGC graphics: layered rendering (%u layers from slice %u) into color target 0x%llx needs its resident image (tile mode %d, gpu tiling %d)", std::max(color.layers, state.renderLayers), color.depthSlice, static_cast<unsigned long long>(color.address), static_cast<int>(color.tileMode), binding.gpuTiling ? 1 : 0);
            throw std::runtime_error(text);
        }
        if (binding.gpuTiling) {
            binding.mip = ColorTargetMip(color, colorLayout);
            binding.tiled = std::make_unique<Buffer>(context, colorLayout.Bytes(), copies);
            binding.tiledDevice = std::make_unique<DeviceBuffer>(context, colorLayout.Bytes(), copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            binding.linearDevice = std::make_unique<DeviceBuffer>(context, colorLayout.LinearBytes(), copies | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            binding.original.resize(colorLayout.Bytes());
            GuestMemory::Read(color.address, binding.original, colorLayout.Alignment());
            std::memcpy(binding.tiled->Bytes().data(), binding.original.data(), binding.original.size());
        } else {
            binding.transfer = std::make_unique<Buffer>(context, colorLayout.LinearBytes(), copies);
            ReadColorTarget(color, binding.transfer->Bytes());
        }
        // A fast-cleared DCC target holds its clear value wherever the draw does not write.
        if (color.dccAddress != 0) {
            const auto keys = CurrentDccKeys(color.dccAddress, colorLayout.Bytes());
            if (IsDccClear(keys)) {
                const auto pixels = binding.gpuTiling ? binding.tiled->Bytes() : binding.transfer->Bytes();
                if (keys == DccKeys::ClearRegister) {
                    const auto texel = clearTexel(color, keys);
                    for (std::size_t offset = 0; offset + color.elementBytes <= pixels.size(); offset += color.elementBytes) std::memcpy(pixels.data() + offset, texel.data(), color.elementBytes);
                } else if (!FillDccClear(color.format, keys, color.dccAlphaOnMsb, pixels)) {
                    static std::set<std::pair<std::uint64_t, int>> reported;
                    if (reported.size() < 32 && reported.insert({color.address, static_cast<int>(keys)}).second) std::fprintf(stderr, "[gpu] color target 0x%llx (VkFormat %d) has %s DCC keys; its stored texels are used\n", static_cast<unsigned long long>(color.address), static_cast<int>(color.format), DccKeysName(keys));
                    if (binding.gpuTiling) std::memcpy(pixels.data(), binding.original.data(), binding.original.size());
                    else ReadColorTarget(color, pixels);
                }
            }
        }
        timer.phase(PhaseReadTarget);
        binding.target = std::make_unique<RenderTarget>(context, color, state.blends.at(color.exportIndex).blendEnable != 0);
        targetViews.push_back(binding.target->View());
    }
    if (state.depth) targetViews.push_back(DepthSurfaceView(context, *state.depth));
    timer.phase(PhasePrepare);
    const bool recordDraws = RecordDraws();
    auto* recorder = Recorder::Active();
    const bool recordable = recordDraws && recorder != nullptr && dumpLimit == 0 && std::all_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.resident != nullptr; });
    auto resolved = resolveDrawResources(context, state, draw, shaders, snapshots, indexBytes, recordable, outcome, timer);
    auto& resources = resolved.resources;
    const auto& contentKey = resolved.contentKey;
    const bool cacheable = resolved.cacheable;
    built = resolved.built;
    // The memory-state decision of an indirect draw, made here after the resource build (which may
    // retire imports and wait for recorded work) and before anything is recorded, exactly as
    // VulkanDevice::dispatch decides for DISPATCH_INDIRECT: the GPU reads the records in place from
    // the host import unless storage-image results are pending over them (flushed, then synced), a
    // label or a copied buffer's write-back still has to land on them, or they are not imported; in
    // those cases (and with APS5_NO_GPU_INDIRECT_DRAW=1) the CPU reads them through the flush hook
    // and records plain draws with the same per-dimension rules. The import pointer stays valid:
    // refreshImports runs only under GuestMemory::GpuMutex, held here, and a retired import's
    // buffer is kept while the recorder is busy.
    IndirectRecord indirect;
    indirect.args = args;
    std::vector<Pm4::DrawArguments> records;
    if (args != nullptr) {
        static const bool gpuIndirectDraws = std::getenv("APS5_NO_GPU_INDIRECT_DRAW") == nullptr;
        const auto decide = [&](std::uint64_t address, std::size_t bytes, const HostImport*& import) {
            if (StorageTexture::FlushPending(address, bytes, nullptr, "indirect draw arguments")) {
                if (recorder != nullptr) {
                    Recorder::CountSync(2);
                    recorder->Sync();
                }
                return IndirectDrawPath::PendingImage;
            }
            if (recorder != nullptr && recorder->PendingLabelIn(address, bytes)) return IndirectDrawPath::PendingLabelOrCopy;
            const auto overlaps = [&](const auto& writer) { return writer->WritesOverlap(address, bytes); };
            if (context.copiedWriters != nullptr && std::any_of(context.copiedWriters->begin(), context.copiedWriters->end(), overlaps)) return IndirectDrawPath::PendingLabelOrCopy;
            if (std::any_of(DrawCopiedWriters()->begin(), DrawCopiedWriters()->end(), overlaps)) return IndirectDrawPath::PendingLabelOrCopy;
            if ((import = HostImportFor(context, address, bytes)) == nullptr) return IndirectDrawPath::NotImported;
            return IndirectDrawPath::Gpu;
        };
        const auto rangeBytes = args->RangeBytes();
        if (rangeBytes == 0) {
            report(" indirect draw without records");
            return;
        }
        Require(rangeBytes <= std::numeric_limits<std::size_t>::max(), "indirect draw record range overflow");
        if (!gpuIndirectDraws) indirect.path = IndirectDrawPath::Disabled;
        else indirect.path = decide(args->arguments, static_cast<std::size_t>(rangeBytes), indirect.argumentImport);
        if (indirect.path == IndirectDrawPath::Gpu && args->countIndirect) {
            Require(context.drawIndirectCount, "indirect draw count without VK_KHR_draw_indirect_count must be resolved by the driver");
            indirect.path = decide(args->countAddress, 4, indirect.countImport);
        }
        if (indirect.path != IndirectDrawPath::Gpu) {
            // The reads wait for the producing batch through the flush hook, as the driver's
            // resolve did for every indirect dispatch before its GPU path.
            const auto readStart = std::chrono::steady_clock::now();
            const auto count = std::min(args->countIndirect ? Pm4::ReadDrawCount(*args) : args->count, args->count);
            records.reserve(count);
            for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(*args, record));
            indirect.readMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
            indirect.records = records;
            if (std::none_of(records.begin(), records.end(), [](const Pm4::DrawArguments& record) { return record.count != 0 && record.instances != 0; })) {
                CountIndirectDraw(indirect.path, indirect.readMs);
                report(" indirect draw with empty records");
                return;
            }
        }
    }
    // A draw whose only completion work is a BDA fault check (the rect-list stages carry a fault
    // buffer, so every rect-list draw needed one) is recorded too, with the check as a completion
    // action like a dispatch's write-back: its own batch would first wait for everything recorded
    // before it. A draw that writes a copied buffer (at the menu stage: 0x48-byte constant buffers
    // inside an import but not at its storage offset alignment, so GuestBufferMemory copies them
    // and every guest buffer counts as writable) is recorded and then waited for at once: the
    // write-back runs before the next packet, without the separate command batch and fence a
    // synchronous draw takes. The copied write-back is listed in DrawCopiedWriters, which lets such
    // draws run without the wait; both consumers of the device's list in VulkanDevice.cpp
    // (DispatchIndirect's argument check and the fill HLE's range check, see Draw.hpp) consult this
    // list too. A draw holding an address-based build's lease (which pins guest allocations until
    // the write-back releases it) runs free like the dispatch path's: the completion releases the
    // lease, and a guest thread that needs a leased allocation syncs the recorder through the
    // registry's pin waiter (see SyncLeaseWork); APS5_SYNC_LEASE_DISPATCH=1 waits for such draws
    // at once as before. Debug aids:
    // APS5_SYNC_COMPLETION_DRAWS=1 keeps every draw with completion work synchronous;
    // APS5_NO_RECORDER_SYNC_DRAWS=1 gives the waited-for draws their own batch as before;
    // APS5_NO_RECORD_COPIED_DRAWS=1 waits for copied-write draws at once instead of listing them.
    static const bool syncCompletionDraws = std::getenv("APS5_SYNC_COMPLETION_DRAWS") != nullptr;
    static const bool recorderSyncDraws = std::getenv("APS5_NO_RECORDER_SYNC_DRAWS") == nullptr;
    static const bool recordCopiedDraws = std::getenv("APS5_NO_RECORD_COPIED_DRAWS") == nullptr;
    const bool completion = resources->NeedsCompletion();
    const bool copiedWrites = completion && resources->HasCopiedWrites();
    // A lease only forces the wait when deferred release is off.
    const bool lease = resources->HoldsLease() && SyncLeaseWork();
    outcome.completion = completion;
    outcome.recorded = recordable;
    // Whether the copied write-back is listed in DrawCopiedWriters instead of waited for.
    bool listed = false;
    if (!recordable) {
        outcome.reason = !recordDraws || dumpLimit != 0 ? SyncDisabled : recorder == nullptr ? SyncNoRecorder : SyncNotResident;
    } else if (completion && (syncCompletionDraws || copiedWrites || lease)) {
        outcome.reason = syncCompletionDraws ? SyncDisabled : lease ? SyncLease : SyncCopiedWrites;
        if (syncCompletionDraws) outcome.recorded = false;
        else if (copiedWrites && !lease && recordCopiedDraws) listed = true;
        else if (recorderSyncDraws) outcome.waited = true;
        else outcome.recorded = false;
    }
    const bool recorded = outcome.recorded;
    // A build this recorded draw can share with later identical ones goes into the cache (a cache
    // hit is reusable by construction, so `recorded` holds for it; one with completion work is
    // never reusable).
    if (cacheable && built != nullptr && recorded && resources->Reusable()) SharedResourceCache().Insert(contentKey, resources);
    else if (cacheable && built != nullptr && recorded && resources->AddressReusable()) {
        auto use = ShaderResources::DetachAddressUse(resources);
        SharedResourceCache().Insert(contentKey, resources);
        resources = std::move(use);
    }
    // A recorded draw renders into its resident targets in the general layout (recordDraw). Debug
    // aid: APS5_DRAW_TRANSITIONS=1 keeps the per-draw upload and download barriers, the layout
    // transitions of every target and one pass per draw, as before.
    static const bool drawTransitions = std::getenv("APS5_DRAW_TRANSITIONS") != nullptr;
    const bool lean = recorded && !drawTransitions;
    if (!lean && !resolved.moved.empty()) {
        resolved.resources = std::make_shared<ShaderResources>(context, shaders, state.color, draw.indexAddress, static_cast<std::size_t>(indexBytes), snapshots);
        resolved.moved.clear();
    }
    APS5_LOG_CHARS_OUT_DEBUG("Creating Pipeline");
    // The state is copied only when a mask must change.
    auto masked = maskedState(state, inputs.fragmentOutputs);
    if (std::any_of(targets.begin(), targets.end(), [](const TargetBinding& binding) { return binding.proxied; })) recipeOut = nullptr;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (!targets[index].proxied) continue;
        if (!masked.has_value()) masked = state;
        masked->colors[index].format = ColorProxyFormat(context, state.colors[index]);
        if (!state.colors[index].blendProxy111110) masked->blends.at(state.colors[index].exportIndex).colorWriteMask &= VK_COLOR_COMPONENT_R_BIT;
    }
    const State& pipelineState = masked.has_value() ? *masked : state;
    const auto storageImages = resources->StorageImages();
    const bool storesThroughAddresses = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaWrites; });
    const bool mayDefer = !resources->HasCopiedWrites() && resources->GpuWrites().empty() && !storesThroughAddresses && std::none_of(storageImages.begin(), storageImages.end(), [](const auto& image) { return image.second; });
    auto pipeline = CachedPipeline(context, pipelineState, inputs.vertexInput, *resources, shaders, lean ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, mayDefer);
    if (pipeline == nullptr) {
        CountDrawSkip(DrawSkip::PipelinePending, 0.0);
        report(" skipped while its pipeline compiles");
        return;
    }
    // Resident targets' views are stable while their storage image lives, so the framebuffer is
    // reused with the pipeline; a per-draw RenderTarget gets a framebuffer of its own.
    std::vector<std::shared_ptr<StorageTexture>> owners;
    owners.reserve(targets.size());
    for (const auto& binding : targets) owners.push_back(binding.resident);
    auto framebuffer = pipeline->AcquireFramebuffer(targetViews, owners, state.renderExtent, state.renderLayers);
    timer.phase(PhasePipeline);
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline created");
    if (lean) {
        RecordedDraw record;
        record.recorder = recorder;
        record.resources = resources;
        record.moved = resolved.moved;
        record.pipeline = pipeline;
        record.framebuffer = framebuffer;
        record.targetViews = targetViews;
        record.targets = owners;
        for (const auto& binding : targets) {
            if (binding.proxied) record.proxies.push_back({binding.resident, binding.color.mip, binding.color.depthSlice, binding.color.layers});
        }
        record.indirect = args != nullptr ? &indirect : nullptr;
        record.listed = listed;
        record.completion = completion;
        record.waited = outcome.waited;
        recordDraw(context, state, draw, shaders, inputs, record, outcome, timer, ownWaitedMs);
        // The recipe for the caller's draw-cache entry (design_cpu_final M8, rule R3 for draws):
        // only a template the resource cache serves under this content key (reusable: no lease,
        // no copied writes, every direct region import- or mirror-served) of a direct draw, so a
        // hit's proof is the template's ProveCurrent and nothing needs completion work.
        // A stream template's recipe always recomputes moved buffers on its hits (its streamed
        // elements are moved on every use, also when this draw is the build itself).
        if (recipeOut != nullptr && DrawRecipes() && cacheable && !outcome.waited && args == nullptr && resources->Reusable() && RecipeEligibleTargets(state, resolved.moved) && (!resources->Streamed() || MovedDrawRecipes())) *recipeOut = makeDrawRecipe(context, state, shaders, resources, contentKey, pipeline, framebuffer, owners, targetViews, inputs, masked, !resolved.moved.empty() || resources->Streamed());
        timing.Mark("draw_and_resource_release");
        report(outcome.waited ? (outcome.reason == SyncLease ? " recorded then waited (lease)" : " recorded then waited (copied writes)") : completion ? " recorded with completion" : " recorded");
        return;
    }
    APS5_LOG_CHARS_OUT_DEBUG("Creating CommandBatch");
    std::optional<CommandBatch> batch;
    if (!recorded) {
        // The color-target detiles below take their descriptor sets from a fresh batch.
        if (context.detiler != nullptr) context.detiler->BeginBatch();
        batch.emplace(context);
    }
    using CommandClass = Recorder::CommandClass;
    const auto countBarrier = [&](std::uint32_t count = 1) { if (recorded) Recorder::CountBarriers(CommandClass::Draw, count); };
    const bool gpuIndirect = args != nullptr && indirect.path == IndirectDrawPath::Gpu;
    // A queued DCC key store over memory the draw writes or reads in place (unknown for an
    // address-based build), or over its GPU-side records, lands before it, as before a dispatch
    // (VulkanDevice::dispatch).
    const auto touches = [&](std::uint64_t begin, std::uint64_t end) {
        const auto bytes = static_cast<std::size_t>(end - begin);
        if (resources->WritesOverlap(begin, bytes) || resources->ReadsOverlap(begin, bytes)) return true;
        if (!gpuIndirect) return false;
        return (begin < args->arguments + args->RangeBytes() && args->arguments < end) || (args->countIndirect && begin < args->countAddress + 4 && args->countAddress < end);
    };
    if (recorded && recorder->HasQueuedKeyStores() && (resources->HoldsLease() || recorder->AnyQueuedKeyStore(touches))) recorder->FlushKeyStores();
    // A queued label store over such memory likewise (Recorder::RecordStore).
    if (recorded && recorder->HasQueuedStores() && (resources->HoldsLease() || recorder->AnyQueuedStore(touches))) recorder->FlushStores();
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (!recorded) {
        commands = batch->Handle();
    } else if (Recorder::BarrierTracking()) {
        auto reads = resources->TrackedReads();
        reads.insert(reads.end(), inputs.inPlaceReads.begin(), inputs.inPlaceReads.end());
        if (gpuIndirect) {
            reads.emplace_back(args->arguments, args->arguments + args->RangeBytes());
            if (args->countIndirect) reads.emplace_back(args->countAddress, args->countAddress + 4);
        }
        auto images = resources->StorageImages();
        bool allResident = true;
        for (const auto& binding : targets) {
            if (binding.resident != nullptr) images.emplace_back(binding.resident->Image(), true);
            else allResident = false;
        }
        bool elided = false;
        commands = recorder->Commands(CommandClass::Draw, Recorder::Access{reads, resources->GpuWrites(), images, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, resources->HoldsLease() || !allResident || static_cast<bool>(state.depth), false, true}, elided);
    } else {
        commands = recorder->Commands();
    }
    APS5_LOG_CHARS_OUT_DEBUG("CommandBatch created");
    // The draw's [gputime] class range: from its first barrier to the download barrier.
    const auto drawTiming = recorded ? recorder->BeginGpuTiming(CommandClass::Draw) : Recorder::NoTiming;
    std::unique_ptr<DeviceBuffer> scratch;
    VkBuffer argumentBuffer = VK_NULL_HANDLE;
    VkDeviceSize argumentOffset = 0;
    bool rewritten = false;
    VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    upload.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | inputs.shaderStages, 0, 1, &upload, 0, nullptr, 0, nullptr);
    countBarrier();
    if (state.depth) {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        countBarrier();
    }
    APS5_LOG_CHARS_OUT_DEBUG("Upload barrier recorded");
    std::shared_ptr<Buffer> meshArguments;
    if (state.stages.mesh && args != nullptr) {
        meshArguments = recordMeshArguments(context, commands, recorder, recorded, state, draw, indirect, countBarrier);
        argumentBuffer = meshArguments->Handle();
    } else if (gpuIndirect) {
        rewritten = recordIndirectArguments(context, commands, recorder, recorded, indirect, scratch, argumentBuffer, argumentOffset, countBarrier);
    }
    for (auto& binding : targets) {
        if (binding.proxied) {
            binding.resident->RecordAttachmentProxyLoad(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, binding.color.mip, binding.color.depthSlice, binding.color.layers);
            countBarrier(2);
            continue;
        }
        if (binding.resident != nullptr) {
            imageBarrier(context, commands, binding.resident->Image(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
            countBarrier();
            continue;
        }
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {binding.color.extent.width, binding.color.extent.height, 1};
        const VkBuffer linear = binding.gpuTiling ? binding.linearDevice->Handle() : binding.transfer->Handle();
        countBarrier(binding.gpuTiling ? 4 : 2);
        if (binding.gpuTiling) {
            CopyBuffer(context, commands, binding.tiled->Handle(), 0, binding.tiledDevice->Handle(), 0, binding.original.size());
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            context.detiler->Dispatch(commands, TextureTileMode::kR64KBX, binding.color.elementBytes, binding.tiledDevice->Handle(), 0, binding.linearDevice->Handle(), 0, binding.mip);
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        }
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        context.Resolved(&DeviceFunctions::cmdCopyBufferToImage, "vkCmdCopyBufferToImage")(commands, linear, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
    }
    APS5_LOG_OUT_DEBUG("Beginning pipeline renderExtent=%ux%u", state.renderExtent.width, state.renderExtent.height);
    pipeline->Begin(commands, *framebuffer, state.renderExtent, state);
    APS5_LOG_CHARS_OUT_DEBUG("Pipeline Begin OK");
    resources->Bind(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->Layout());
    APS5_LOG_CHARS_OUT_DEBUG("Resources bound");
    pushDrawConstants(*pipeline, commands, state, draw, shaders, *resources, nullptr, 0, meshArguments != nullptr ? meshArguments->DeviceAddress() : 0);
    APS5_LOG_CHARS_OUT_DEBUG("Push constants recorded");
    recordDrawCommands(context, commands, state, draw, inputs, args != nullptr ? &indirect : nullptr, argumentBuffer, argumentOffset, shaders);
    if (meshArguments != nullptr && recorded) recorder->Keep(meshArguments);
    if (args != nullptr) CountIndirectDraw(indirect.path, indirect.readMs, rewritten);
    auto checkRecords = indirectRecordCheck(args != nullptr ? &indirect : nullptr);
    APS5_LOG_CHARS_OUT_DEBUG("Draw recorded");
    context.Resolved(&DeviceFunctions::cmdEndRenderPass, "vkCmdEndRenderPass")(commands);
    APS5_LOG_CHARS_OUT_DEBUG("Render pass ended");
    for (auto& binding : targets) {
        if (binding.proxied) {
            binding.resident->RecordAttachmentProxyStore(commands, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, binding.color.mip, binding.color.depthSlice, binding.color.layers);
            countBarrier(2);
            continue;
        }
        if (binding.resident != nullptr) {
            imageBarrier(context, commands, binding.resident->Image(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
            countBarrier();
            continue;
        }
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {binding.color.extent.width, binding.color.extent.height, 1};
        countBarrier(binding.gpuTiling ? 4 : 2);
        imageBarrier(context, commands, binding.target->Image(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBuffer linear = binding.gpuTiling ? binding.linearDevice->Handle() : binding.transfer->Handle();
        VkBufferMemoryBarrier reuse{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        reuse.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        reuse.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        reuse.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        reuse.buffer = linear;
        reuse.size = VK_WHOLE_SIZE;
        context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &reuse, 0, nullptr);
        context.Resolved(&DeviceFunctions::cmdCopyImageToBuffer, "vkCmdCopyImageToBuffer")(commands, binding.target->Image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, linear, 1, &copy);
        if (binding.gpuTiling) {
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT);
            if (dumpLimit > 0) {
                std::lock_guard lock(dumpMutex);
                if (dumped[binding.color.address] < dumpLimit) {
                    binding.dump = std::make_unique<Buffer>(context, binding.linearDevice->Size(), VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                    CopyBuffer(context, commands, binding.linearDevice->Handle(), 0, binding.dump->Handle(), 0, binding.linearDevice->Size());
                }
            }
            context.detiler->Dispatch(commands, TextureTileMode::kR64KBX, binding.color.elementBytes, binding.linearDevice->Handle(), 0, binding.tiledDevice->Handle(), 0, binding.mip, true);
            memoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            CopyBuffer(context, commands, binding.tiledDevice->Handle(), 0, binding.tiled->Handle(), 0, binding.original.size());
        }
    }
    VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    download.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    download.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | inputs.shaderStages, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
    countBarrier();
    if (recorded) recorder->EndGpuTiming(drawTiming);
    APS5_LOG_CHARS_OUT_DEBUG("Download barrier recorded");
    APS5_LOG_CHARS_OUT_DEBUG("SubmitAndWait");
    timer.phase(PhaseRecord);
    if (recorded) {
        keepRecordedDraw(*recorder, resources, pipeline, framebuffer, inputs, owners, std::move(scratch), std::move(checkRecords), listed, completion, outcome);
        timer.phase(PhaseKeep);
        if (outcome.waited) {
            // The wait the dispatch path makes for such work (source 3, "address-based"): the
            // write-back runs before the next packet, so the CPU never reads copied results or
            // touches a leased allocation before they landed.
            Recorder::CountSync(3);
            const auto ownBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
            recorder->Sync();
            if (profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
            timer.phase(PhaseSync);
        }
        report(outcome.waited ? (outcome.reason == SyncLease ? " recorded then waited (lease)" : " recorded then waited (copied writes)") : completion ? " recorded with completion" : " recorded");
        return;
    }
    {
        // The recorder sync SubmitAndWait makes is the draw's own wait, not a nested hook wait.
        const auto ownBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
        batch->SubmitAndWait();
        if (profile) ownWaitedMs += Recorder::ThreadWaitedMs() - ownBefore;
    }
    if (checkRecords) checkRecords();
    timer.phase(PhaseSync);
    APS5_LOG_CHARS_OUT_DEBUG("SubmitAndWait OK");
    for (const auto& binding : targets) GuestMemory::CheckRange(reinterpret_cast<const void*>(binding.color.address), binding.color.bytes, 256, true);
    for (auto& binding : targets) {
        if (!binding.dump) continue;
        std::lock_guard lock(dumpMutex);
        char name[64];
        std::snprintf(name, sizeof(name), "target_%llx_%d.raw", static_cast<unsigned long long>(binding.color.address), dumped[binding.color.address]++);
        if (std::FILE* file = std::fopen(name, "wb")) {
            const std::uint32_t header[3] = {binding.color.extent.width, binding.color.extent.height, static_cast<std::uint32_t>(binding.color.format)};
            std::fwrite(header, sizeof(header), 1, file);
            std::fwrite(binding.dump->Bytes().data(), 1, binding.dump->Bytes().size(), file);
            std::fclose(file);
        }
    }
    APS5_LOG_CHARS_OUT_DEBUG("Shader resources WriteBack");
    resources->WriteBack();
    APS5_LOG_CHARS_OUT_DEBUG("Shader resources WriteBack OK");
    static const bool skipTargetWrite = std::getenv("APS5_NO_TARGET_WRITEBACK") != nullptr;
    for (const auto& binding : targets) {
        if (skipTargetWrite) break;
        if (binding.resident != nullptr) {
            // The results stay on the GPU until something reads the target's memory.
            binding.resident->MarkDirty();
            continue;
        }
        if (binding.gpuTiling) GuestMemory::WriteChanged(binding.color.address, binding.tiled->Bytes(), binding.original);
        else WriteColorTarget(binding.color, binding.transfer->Bytes());
        // The stored texels are the whole target now, so later reads must see them rather than a fast clear.
        const auto targetBytes = ColorTargetLayout(binding.color.extent.width, binding.color.extent.height, binding.color.tileMode, binding.color.elementBytes).Bytes();
        MarkDccUncompressed(binding.color.dccAddress, targetBytes, colorKeyCount(binding.color, targetBytes));
    }
    timer.phase(PhaseWriteBack);
    if (profile && traceDraws) {
        // A synchronous draw's inputs and target sample are described as a rendering debug aid;
        // the ~0.5 ms that takes (Describe samples every bound range and reads its DCC keys) is
        // shown as its own phase and kept out of the plain profile.
        std::size_t nonzero = 0;
        std::size_t sampled = 0;
        if (!targets.empty() && targets.front().resident == nullptr) {
            const auto bytes = targets.front().gpuTiling ? targets.front().tiled->Bytes() : targets.front().transfer->Bytes();
            sampled = bytes.size() / 64;
            for (std::size_t i = 0; i < bytes.size(); i += 64) nonzero += bytes[i] != std::byte{0};
        }
        std::fprintf(stderr, "[draw]   inputs:%s\n", resources->Describe().c_str());
        if (targets.size() > 1) {
            std::string list;
            for (const auto& binding : targets) {
                char entry[64];
                std::snprintf(entry, sizeof(entry), " 0x%llx(%d,%ux%u)", static_cast<unsigned long long>(binding.color.address), static_cast<int>(binding.color.format), binding.color.extent.width, binding.color.extent.height);
                list += entry;
            }
            std::fprintf(stderr, "[draw]   targets:%s\n", list.c_str());
        }
        timer.phase(PhaseDescribe);
        char suffix[160];
        std::snprintf(suffix, sizeof(suffix), " synchronous (%s), first 0x%llx format %d (%zu of %zu sampled bytes nonzero)", SyncReasonNames[outcome.reason], static_cast<unsigned long long>(state.color.address), static_cast<int>(state.color.format), nonzero, sampled);
        report(suffix);
    } else {
        report(" synchronous");
    }
    APS5_LOG_CHARS_OUT_DEBUG("Draw finished");
}


enum class PlanFallback : std::size_t { Indirect, NotLean, TargetLost, NoPrebuild, Device, PrepRefused, PlanOff, Count };
constexpr std::array<const char*, static_cast<std::size_t>(PlanFallback::Count)> PlanFallbackNames{"indirect", "not lean", "target lost", "no prep build", "device replaced", "prep refused", "plan off"};

struct PlanStats {
    std::atomic<std::uint64_t> planned{0};
    std::atomic<std::uint64_t> buildNs{0};
    std::atomic<std::uint64_t> applyNs{0};
    std::atomic<std::uint64_t> used{0};
    std::atomic<std::uint64_t> prebuilt{0};
    std::atomic<std::uint64_t> prepNs{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(PlanFallback::Count)> fallbacks{};
    std::atomic<std::int64_t> lastReport{0};
};

PlanStats& PlanStatistics() {
    static PlanStats stats;
    return stats;
}

bool planProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

bool verifyDrawPlan() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_VERIFY_DRAW_PLAN");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since, std::chrono::steady_clock::time_point until = std::chrono::steady_clock::now()) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(until - since).count());
}

void reportPlan() {
    if (!planProfiled()) return;
    auto& stats = PlanStatistics();
    const auto now = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    auto last = stats.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        stats.lastReport.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 10'000'000'000ll || !stats.lastReport.compare_exchange_strong(last, now)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto perDraw = [](unsigned long long ns, unsigned long long count) { return count != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(count) : 0.0; };
    const auto planned = take(stats.planned);
    const auto buildNs = take(stats.buildNs);
    const auto applyNs = take(stats.applyNs);
    const auto used = take(stats.used);
    const auto prebuilt = take(stats.prebuilt);
    const auto prepNs = take(stats.prepNs);
    char line[512];
    int n = std::snprintf(line, sizeof(line), "[plan] %llu planned draws over 10 s: committer %.1f us/draw (build %.1f, apply %.1f), %llu with a prep build; prep build %.1f us/draw over %llu; fallbacks:", planned, perDraw(buildNs + applyNs, planned), perDraw(buildNs, planned), perDraw(applyNs, planned), used, perDraw(prepNs, prebuilt), prebuilt);
    for (std::size_t i = 0; i < stats.fallbacks.size() && n > 0 && static_cast<std::size_t>(n) < sizeof(line); ++i) n += std::snprintf(line + n, sizeof(line) - static_cast<std::size_t>(n), " %s %llu", PlanFallbackNames[i], take(stats.fallbacks[i]));
    std::fprintf(stderr, "%s\n", line);
}

void countPlanFallback(PlanFallback reason) {
    PlanStatistics().fallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    reportPlan();
}

thread_local std::shared_ptr<DrawPrebuild> currentPrebuild;

std::shared_ptr<DrawPrebuild> buildPrebuild(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, bool keyed) {
    static const bool trimKey = std::getenv("APS5_NO_DRAW_KEY_TRIM") == nullptr;
    auto prebuilt = std::make_shared<DrawPrebuild>();
    prebuilt->device = context.device;
    DrawInputs inputs;
    validateDraw(context, state, draw, shaders, nullptr, false, inputs, prebuilt->validateMemoized, prebuilt->validateHit);
    prebuilt->nothing = inputs.nothing;
    if (inputs.nothing) return prebuilt;
    prebuilt->indexBytes = inputs.indexBytes;
    if (draw.indexed && inputs.indexBytes != 0 && PrepIndexScan()) {
        const auto bytes = static_cast<std::size_t>(inputs.indexBytes);
        if (GuestMemory::Accessible(reinterpret_cast<const void*>(draw.indexAddress), bytes)) {
            const auto generation = GuestMemory::CollectWrites(draw.indexAddress, bytes);
            prebuilt->highestIndex = ScanHighestIndex(std::span(reinterpret_cast<const std::byte*>(draw.indexAddress), bytes), draw.indexSize);
            prebuilt->indexGeneration = generation;
        }
    }
    prebuilt->fragmentOutputs = std::move(inputs.fragmentOutputs);
    prebuilt->shaderStages = inputs.shaderStages;
    prebuilt->meshGroups = inputs.meshGroups;
    const auto& attributes = shaders.front().program->vertexAttributes;
    prebuilt->vertexInput = BuildVertexInputLayout(context, attributes);
    if (!draw.indexed && !draw.indirect) prebuilt->vertexPlan = planVertexFetches(state, draw, attributes, draw.firstVertex + draw.indexCount - 1u, false);
    prebuilt->masked = maskedState(state, prebuilt->fragmentOutputs);
    prebuilt->proxied.assign(state.colors.size(), false);
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        const auto proxy = ColorProxyFormat(context, state.colors[index]);
        if (proxy == VK_FORMAT_UNDEFINED) continue;
        prebuilt->proxied[index] = true;
        if (!prebuilt->masked.has_value()) prebuilt->masked = state;
        prebuilt->masked->colors[index].format = proxy;
        if (!state.colors[index].blendProxy111110) prebuilt->masked->blends.at(state.colors[index].exportIndex).colorWriteMask &= VK_COLOR_COMPONENT_R_BIT;
    }
    const bool variants = std::all_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->PipelineVariantId() != 0; });
    if (keyed && variants) {
        prebuilt->key = DrawResourceKey(context, shaders, state.color, draw.indexAddress, inputs.indexBytes, !trimKey);
        prebuilt->keyed = true;
    }
    if (trimKey) {
        try {
            CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
            prebuilt->aliasesOk = true;
        } catch (const std::exception&) {
            prebuilt->aliasesOk = false;
        }
    }
    prebuilt->push.bytes = AssemblePushConstants(shaders);
    prebuilt->push.stages = PushConstantStages(shaders);
    prebuilt->storesThroughAddresses = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaWrites; });
    return prebuilt;
}

const char* prebuildDifference(const DrawPrebuild& prep, const DrawPrebuild& committer) {
    const auto sameLayout = [](const VertexInputLayout& a, const VertexInputLayout& b) {
        const bool bindings = std::equal(a.bindings.begin(), a.bindings.end(), b.bindings.begin(), b.bindings.end(), [](const VkVertexInputBindingDescription& x, const VkVertexInputBindingDescription& y) { return x.binding == y.binding && x.stride == y.stride && x.inputRate == y.inputRate; });
        return bindings && std::equal(a.attributes.begin(), a.attributes.end(), b.attributes.begin(), b.attributes.end(), [](const VkVertexInputAttributeDescription& x, const VkVertexInputAttributeDescription& y) { return x.location == y.location && x.binding == y.binding && x.format == y.format && x.offset == y.offset; });
    };
    const auto samePlan = [](const std::optional<VertexFetchPlan>& a, const std::optional<VertexFetchPlan>& b) {
        if (a.has_value() != b.has_value()) return false;
        if (!a.has_value()) return true;
        return a->shortBytes == b->shortBytes && a->fetchOf == b->fetchOf && a->plan.copies == b->plan.copies && a->plan.copyOf == b->plan.copyOf && a->plan.offsets == b->plan.offsets && a->plan.alignments == b->plan.alignments;
    };
    const auto sameMasked = [](const std::optional<State>& a, const std::optional<State>& b) {
        if (a.has_value() != b.has_value()) return false;
        if (!a.has_value()) return true;
        if (a->blends.size() != b->blends.size() || a->colors.size() != b->colors.size()) return false;
        for (std::size_t index = 0; index < a->blends.size(); ++index) {
            if (a->blends[index].colorWriteMask != b->blends[index].colorWriteMask) return false;
        }
        for (std::size_t index = 0; index < a->colors.size(); ++index) {
            if (a->colors[index].format != b->colors[index].format) return false;
        }
        return true;
    };
    if (prep.nothing != committer.nothing) return "empty-draw decision";
    if (prep.indexBytes != committer.indexBytes) return "index bytes";
    if (prep.fragmentOutputs != committer.fragmentOutputs) return "fragment outputs";
    if (prep.shaderStages != committer.shaderStages) return "pipeline stages";
    if (prep.meshGroups != committer.meshGroups) return "mesh groups";
    if (!sameLayout(prep.vertexInput, committer.vertexInput)) return "vertex input layout";
    if (!samePlan(prep.vertexPlan, committer.vertexPlan)) return "vertex fetch plan";
    if (prep.proxied != committer.proxied) return "attachment proxies";
    if (!sameMasked(prep.masked, committer.masked)) return "masked state";
    if (prep.keyed != committer.keyed || prep.key != committer.key) return "resource key";
    if (prep.aliasesOk != committer.aliasesOk) return "alias checks";
    if (prep.push.bytes != committer.push.bytes || prep.push.stages != committer.push.stages) return "push constants";
    if (prep.storesThroughAddresses != committer.storesThroughAddresses) return "address stores";
    return nullptr;
}

void verifyPrebuild(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawPrebuild& prebuilt) {
    const char* what = nullptr;
    try {
        const auto again = buildPrebuild(context, state, draw, shaders, prebuilt.keyed);
        what = prebuildDifference(prebuilt, *again);
    } catch (const std::exception&) {
        what = "validation";
    }
    if (what == nullptr) return;
    std::fprintf(stderr, "[plan] APS5_VERIFY_DRAW_PLAN: the prep-built %s differs from the committer's\n", what);
    std::fflush(stderr);
    std::abort();
}

struct DrawPlan {
    explicit DrawPlan(bool profile) : timer(profile) {}
    const State* state = nullptr;
    const Pm4::DrawParameters* draw = nullptr;
    std::span<const CompiledShader> shaders;
    std::shared_ptr<const DrawRecipe>* recipeOut = nullptr;
    DrawTimer timer;
    DrawOutcome outcome;
    double waitedBefore = 0;
    double ownWaitedMs = 0;
    bool nothing = false;
    bool skipped = false;
    bool targetLost = false;
    bool recipeAllowed = false;
    std::shared_ptr<DrawPrebuild> prebuilt;
    std::vector<DrawInputEffect> effects;
    DrawInputs inputs;
    std::vector<VkImageView> targetViews;
    std::vector<std::shared_ptr<StorageTexture>> owners;
    ResolvedResources resolved;
    std::optional<State> masked;
    std::shared_ptr<Pipeline> pipeline;
    std::shared_ptr<Framebuffer> framebuffer;
    RecordedDraw record;
    RecordPlan recordPlan;
};

void applyDrawInputs(const Context& context, std::vector<DrawInputEffect>& effects) {
    for (const auto& effect : effects) {
        if (effect.keep) KeepDrawInput(context.recorder, effect.address, effect.copy, effect.use, effect.derived);
        else context.recorder->NotePendingRead(effect.address, effect.bytes, Recorder::ReadKind::DrawInput);
    }
    effects.clear();
}

DrawPlan BuildDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, std::shared_ptr<DrawPrebuild> prebuilt) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool recordCopiedDraws = std::getenv("APS5_NO_RECORD_COPIED_DRAWS") == nullptr;
    DrawPlan plan(profile);
    plan.state = &state;
    plan.draw = &draw;
    plan.shaders = shaders;
    plan.recipeOut = recipeOut;
    plan.prebuilt = std::move(prebuilt);
    auto* pre = plan.prebuilt.get();
    plan.waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    plan.inputs = prepareDrawInputs(context, state, draw, shaders, plan.outcome, plan.timer, nullptr, pre, &plan.effects);
    if (plan.inputs.nothing) {
        plan.nothing = true;
        return plan;
    }
    bool anyProxied = false;
    plan.targetViews.reserve(state.colors.size() + 1);
    plan.owners.reserve(state.colors.size());
    for (std::size_t index = 0; index < state.colors.size(); ++index) {
        const auto& color = state.colors[index];
        const ColorTargetLayout colorLayout(color.extent.width, color.extent.height, color.tileMode, color.elementBytes);
        plan.timer.phase(PhaseSetup);
        std::shared_ptr<StorageTexture> resident;
        {
            const RefreshSiteScope site("render-target");
            resident = MemoizedTarget(index, state);
            if (resident == nullptr) {
                ForkTrap("target memo");
                resident = refreshResidentTarget(context, state, color, plan.outcome, profile, [&] {
                    auto found = CachedStorageSurface(context, SurfaceForTarget(color));
                    Require(found->Attachable(), "storage format cannot be a color attachment");
                    Require(color.mipCount > 1 || color.depth > 1 || found->GuestBytes() == colorLayout.Bytes() || (color.arraySize > 1 && found->GuestBytes() == colorLayout.Bytes() * color.arraySize), "resident image layout differs from the color layout");
                    return found;
                });
            }
        }
        if (resident == nullptr) {
            plan.targetLost = true;
            return plan;
        }
        plan.timer.phase(PhaseReadTarget);
        const bool proxied = pre != nullptr ? static_cast<bool>(pre->proxied[index]) : ColorProxyFormat(context, color) != VK_FORMAT_UNDEFINED;
        plan.targetViews.push_back(proxied ? resident->AttachmentProxyView(color.blendProxy111110, color.mip, color.depthSlice, color.layers) : resident->AttachmentView(color.format, color.mip, color.depthSlice, color.layers));
        if (proxied) plan.record.proxies.push_back({resident, color.mip, color.depthSlice, color.layers});
        anyProxied = anyProxied || proxied;
        plan.owners.push_back(std::move(resident));
    }
    if (state.depth) plan.targetViews.push_back(DepthSurfaceView(context, *state.depth));
    plan.timer.phase(PhasePrepare);
    auto* recorder = Recorder::Active();
    plan.resolved = resolveDrawResources(context, state, draw, shaders, snapshots, plan.inputs.indexBytes, true, plan.outcome, plan.timer, pre);
    auto& resources = plan.resolved.resources;
    const bool completion = resources->NeedsCompletion();
    const bool copiedWrites = completion && resources->HasCopiedWrites();
    const bool lease = resources->HoldsLease() && SyncLeaseWork();
    plan.outcome.completion = completion;
    plan.outcome.recorded = true;
    bool listed = false;
    if (completion && (copiedWrites || lease)) {
        plan.outcome.reason = lease ? SyncLease : SyncCopiedWrites;
        if (copiedWrites && !lease && recordCopiedDraws) listed = true;
        else plan.outcome.waited = true;
    }
    if (plan.resolved.cacheable && plan.resolved.built != nullptr && resources->Reusable()) SharedResourceCache().Insert(plan.resolved.contentKey, resources);
    else if (plan.resolved.cacheable && plan.resolved.built != nullptr && resources->AddressReusable()) {
        auto use = ShaderResources::DetachAddressUse(resources);
        SharedResourceCache().Insert(plan.resolved.contentKey, resources);
        resources = std::move(use);
    }
    if (pre != nullptr) {
        plan.masked = std::move(pre->masked);
    } else {
        plan.masked = maskedState(state, plan.inputs.fragmentOutputs);
        for (std::size_t index = 0; index < state.colors.size(); ++index) {
            const auto proxy = ColorProxyFormat(context, state.colors[index]);
            if (proxy == VK_FORMAT_UNDEFINED) continue;
            if (!plan.masked.has_value()) plan.masked = state;
            plan.masked->colors[index].format = proxy;
            if (!state.colors[index].blendProxy111110) plan.masked->blends.at(state.colors[index].exportIndex).colorWriteMask &= VK_COLOR_COMPONENT_R_BIT;
        }
    }
    plan.recipeAllowed = recipeOut != nullptr && !anyProxied;
    const State& pipelineState = plan.masked.has_value() ? *plan.masked : state;
    const auto storageImages = resources->StorageImages();
    const bool storesThroughAddresses = pre != nullptr ? pre->storesThroughAddresses : std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaWrites; });
    const bool mayDefer = !resources->HasCopiedWrites() && resources->GpuWrites().empty() && !storesThroughAddresses && std::none_of(storageImages.begin(), storageImages.end(), [](const auto& image) { return image.second; });
    plan.pipeline = CachedPipeline(context, pipelineState, plan.inputs.vertexInput, *resources, shaders, VK_IMAGE_LAYOUT_GENERAL, mayDefer);
    if (plan.pipeline == nullptr) {
        plan.skipped = true;
        return plan;
    }
    if (ForkLogged() == nullptr) plan.framebuffer = plan.pipeline->AcquireFramebuffer(plan.targetViews, plan.owners, state.renderExtent, state.renderLayers);
    plan.timer.phase(PhasePipeline);
    auto& record = plan.record;
    record.recorder = recorder;
    record.resources = resources;
    record.moved = plan.resolved.moved;
    record.pipeline = plan.pipeline;
    record.framebuffer = plan.framebuffer;
    record.targetViews = plan.targetViews;
    record.targets = plan.owners;
    record.listed = listed;
    record.completion = completion;
    record.waited = plan.outcome.waited;
    record.assembled = pre != nullptr ? &pre->push : nullptr;
    plan.recordPlan = planRecord(state, record);
    // APS5_PROFILE_LOOKUP: [bindings-split] the per-draw descriptor set PrepareDrawBindings makes.
    static const bool bindingsSplit = std::getenv("APS5_PROFILE_LOOKUP") != nullptr;
    const auto bindingsStart = bindingsSplit ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    plan.recordPlan.drawBindings = resources->PrepareDrawBindings(*recorder, record.moved);
    if (bindingsSplit) {
        static std::atomic<std::uint64_t> calls{0}, made{0}, snapshots{0}, ns{0};
        static std::atomic<std::int64_t> reported{0};
        ns.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - bindingsStart).count()), std::memory_order_relaxed);
        if (plan.recordPlan.drawBindings != nullptr) {
            made.fetch_add(1, std::memory_order_relaxed);
            snapshots.fetch_add(plan.recordPlan.drawBindings->snapshots.size(), std::memory_order_relaxed);
        }
        const auto count = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = reported.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && reported.compare_exchange_strong(last, now)) {
            calls.store(0);
            const auto sets = made.exchange(0);
            const auto total = ns.exchange(0);
            std::fprintf(stderr, "[bindings-split] %llu planned draws: %llu got a per-draw descriptor set (%.1f snapshot buffers each); %.2f us per draw, %.2f us per set\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(sets), sets != 0 ? static_cast<double>(snapshots.exchange(0)) / static_cast<double>(sets) : 0.0, static_cast<double>(total) / 1000.0 / static_cast<double>(count), sets != 0 ? static_cast<double>(total) / 1000.0 / static_cast<double>(sets) : 0.0);
        }
    }
    return plan;
}

void ApplyDraw(const Context& context, DrawPlan& plan) {
    const auto& state = *plan.state;
    const auto& draw = *plan.draw;
    const auto shaders = plan.shaders;
    const auto report = [&](const char* suffix) { reportDrawEnd(state, plan.timer, plan.resolved.built, plan.outcome, plan.waitedBefore, plan.ownWaitedMs, suffix); };
    applyDrawInputs(context, plan.effects);
    if (plan.skipped) {
        CountDrawSkip(DrawSkip::PipelinePending, 0.0);
        report(" skipped while its pipeline compiles");
        return;
    }
    flushRecordStores(plan.record);
    emitRecord(context, state, draw, shaders, plan.inputs, plan.record, plan.recordPlan, plan.outcome, plan.timer, plan.ownWaitedMs);
    const auto& resources = plan.resolved.resources;
    // As in drawPrepared: a stream template's recipe always recomputes moved buffers.
    if (plan.recipeAllowed && DrawRecipes() && plan.resolved.cacheable && !plan.outcome.waited && resources->Reusable() && RecipeEligibleTargets(state, plan.resolved.moved) && (!resources->Streamed() || MovedDrawRecipes())) {
        if (plan.framebuffer == nullptr) plan.framebuffer = plan.pipeline->AcquireFramebuffer(plan.targetViews, plan.owners, state.renderExtent, state.renderLayers);
        *plan.recipeOut = makeDrawRecipe(context, state, shaders, resources, plan.resolved.contentKey, plan.pipeline, plan.framebuffer, plan.owners, plan.targetViews, plan.inputs, plan.masked, !plan.resolved.moved.empty() || resources->Streamed());
    }
    report(plan.outcome.waited ? (plan.outcome.reason == SyncLease ? " recorded then waited (lease)" : " recorded then waited (copied writes)") : plan.outcome.completion ? " recorded with completion" : " recorded");
}

bool planLean(const Context& context, const State& state) {
    static const bool residentTargets = std::getenv("APS5_NO_RESIDENT_TARGETS") == nullptr;
    static const bool residentLinear = std::getenv("APS5_NO_RESIDENT_LINEAR_TARGETS") == nullptr;
    static const bool drawTransitions = std::getenv("APS5_DRAW_TRANSITIONS") != nullptr;
    static const bool syncCompletionDraws = std::getenv("APS5_SYNC_COMPLETION_DRAWS") != nullptr;
    static const bool recorderSyncDraws = std::getenv("APS5_NO_RECORDER_SYNC_DRAWS") == nullptr;
    const auto residentEligible = [&](const ColorTarget& color) {
        const bool gpuTiling = color.tileMode == ColorTileMode::RenderTarget && context.detiler != nullptr;
        const bool linearResident = residentLinear && color.tileMode == ColorTileMode::Linear && RegisteredReadableCovers(color.address, static_cast<std::size_t>(color.bytes));
        return gpuTiling || ((color.tileMode == ColorTileMode::Standard4KB || linearResident) && context.detiler != nullptr);
    };
    return RecordDraws() && Recorder::Active() != nullptr && DumpTargetLimit() == 0 && !drawTransitions && !syncCompletionDraws && recorderSyncDraws && residentTargets && std::all_of(state.colors.begin(), state.colors.end(), residentEligible);
}

bool finishPlanned(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, DrawPlan& plan, PerformanceTimer& timing, std::chrono::steady_clock::time_point started, bool used) {
    if (plan.nothing) return true;
    if (plan.targetLost) {
        countPlanFallback(PlanFallback::TargetLost);
        applyDrawInputs(context, plan.effects);
        timing.Mark("validate");
        timing.Mark("index_upload");
        drawPrepared(context, state, draw, shaders, snapshots, recipeOut, plan.inputs, timing, plan.timer, plan.outcome, plan.waitedBefore);
        return true;
    }
    const auto built = planProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ApplyDraw(context, plan);
    timing.Mark("draw_and_resource_release");
    if (planProfiled()) {
        auto& stats = PlanStatistics();
        stats.planned.fetch_add(1, std::memory_order_relaxed);
        stats.buildNs.fetch_add(elapsedNs(started, built), std::memory_order_relaxed);
        stats.applyNs.fetch_add(elapsedNs(built), std::memory_order_relaxed);
        if (used) stats.used.fetch_add(1, std::memory_order_relaxed);
        reportPlan();
    }
    return true;
}

}

struct ForkSeed {
    TargetMemo memo;
    bool current = false;
};

struct ForkedDraw {
    explicit ForkedDraw(std::size_t slot) : log(false, slot) {}
    ForkLog log;
    std::shared_ptr<DrawPrebuild> prebuilt;
    std::unique_ptr<DrawPlan> plan;
    const char* refused = nullptr;
    const char* cut = nullptr;
    std::uint64_t buildNs = 0;
};

struct ForkStamp {
    Recorder* recorder = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t pendingSerial = 0;
    std::uint64_t unwatched = 0;
    std::uint64_t allocations = 0;
    std::uint64_t submissions = 0;
    std::uint64_t writeNotes = 0;
    std::uint64_t finished = 0;
    std::uint64_t passGeneration = 0;
    std::size_t keyStores = 0;
    std::size_t stores = 0;
    std::size_t copiedWriters = 0;
    std::size_t drawWriters = 0;
    HostImportsProof imports;
};

namespace {

struct ForkStatistics {
    std::atomic<std::uint64_t> builds{0};
    std::atomic<std::uint64_t> refused{0};
    std::atomic<std::uint64_t> buildNs{0};
    std::atomic<std::uint64_t> applied{0};
    std::atomic<std::uint64_t> applyNs{0};
    std::atomic<std::uint64_t> replayed{0};
    std::atomic<std::uint64_t> verified{0};
    std::atomic<std::uint64_t> mismatches{0};
    std::mutex mutex;
    std::map<std::string, std::uint64_t> cuts;
};

ForkStatistics& ForkCounters() {
    static auto* stats = new ForkStatistics();
    return *stats;
}

bool sameStamp(const ForkStamp& a, const ForkStamp& b) {
    return a.recorder == b.recorder && a.device == b.device && a.pendingSerial == b.pendingSerial && a.unwatched == b.unwatched && a.allocations == b.allocations && a.submissions == b.submissions && a.writeNotes == b.writeNotes && a.finished == b.finished && a.passGeneration == b.passGeneration && a.keyStores == b.keyStores && a.stores == b.stores && a.copiedWriters == b.copiedWriters && a.drawWriters == b.drawWriters && a.imports.device == b.imports.device && a.imports.epoch == b.imports.epoch && a.imports.refreshedGeneration == b.imports.refreshedGeneration;
}

ForkStamp stampNow(const Context& context) {
    ForkStamp stamp;
    stamp.recorder = Recorder::Active();
    stamp.device = context.device;
    stamp.pendingSerial = StorageTexture::PendingSerial();
    stamp.unwatched = GuestMemory::UnwatchSerial();
    stamp.allocations = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (stamp.recorder != nullptr) {
        stamp.submissions = stamp.recorder->Submissions();
        stamp.writeNotes = stamp.recorder->WriteNotes();
        stamp.finished = stamp.recorder->FinishedBatches();
        stamp.passGeneration = stamp.recorder->PassGeneration();
        stamp.keyStores = stamp.recorder->QueuedKeyStoreCount();
        stamp.stores = stamp.recorder->QueuedStoreCount();
    }
    stamp.copiedWriters = context.copiedWriters != nullptr ? context.copiedWriters->size() : 0;
    stamp.drawWriters = DrawCopiedWriters()->size();
    stamp.imports = HostImportsIdentity(context);
    return stamp;
}

const char* forkIneligible(const Context& context, const State& state, const Pm4::DrawParameters& draw, const DrawPrebuild* prebuilt) {
    static const bool traceAllDraws = std::getenv("APS5_TRACE_ALL_DRAWS") != nullptr;
    if (draw.indirect) return "indirect";
    if (prebuilt == nullptr) return "no prep build";
    if (prebuilt->refused || prebuilt->device != context.device) return "prep build refused";
    if (CaptureTrace::Enabled()) return "capture";
    if (traceAllDraws || DebugValue(DebugKey::TraceTarget) != 0 || DebugValue(DebugKey::TraceDepth) != 0) return "trace";
    if (verifyDrawPlan()) return "plan verify";
    if (!planLean(context, state)) return "not lean";
    return nullptr;
}

void redecideCompletion(DrawPlan& plan) {
    static const bool recordCopiedDraws = std::getenv("APS5_NO_RECORD_COPIED_DRAWS") == nullptr;
    const auto& resources = plan.resolved.resources;
    const bool completion = resources->NeedsCompletion();
    const bool copiedWrites = completion && resources->HasCopiedWrites();
    const bool lease = resources->HoldsLease() && SyncLeaseWork();
    plan.outcome.completion = completion;
    plan.outcome.reason = SyncNone;
    plan.outcome.waited = false;
    bool listed = false;
    if (completion && (copiedWrites || lease)) {
        plan.outcome.reason = lease ? SyncLease : SyncCopiedWrites;
        if (copiedWrites && !lease && recordCopiedDraws) listed = true;
        else plan.outcome.waited = true;
    }
    plan.record.listed = listed;
    plan.record.completion = completion;
    plan.record.waited = plan.outcome.waited;
}

const char* forkFinish(DrawPlan& plan) {
    auto speculated = std::move(plan.resolved.speculated);
    if (speculated == nullptr || plan.nothing || plan.skipped) return nullptr;
    if (DrawTokensEnabled()) ScreenTokenWindow(*speculated);
    const auto use = ShaderResources::UseAddressSpeculation(plan.resolved.speculatedSource, *speculated);
    if (use == nullptr || use != plan.resolved.resources) return addressSpeculationRefusal(speculated->miss);
    redecideCompletion(plan);
    plan.recordPlan.readsTarget = std::any_of(plan.owners.begin(), plan.owners.end(), [&](const std::shared_ptr<StorageTexture>& target) { return use->ReadsImage(target.get()); });
    return nullptr;
}

bool forkQuiet(const DrawPlan& plan) {
    if (plan.nothing || plan.skipped) return true;
    if (plan.resolved.resources == nullptr || plan.outcome.waited || plan.record.listed) return false;
    const auto& resources = *plan.resolved.resources;
    if (resources.HasCopiedWrites() || !resources.GpuWrites().empty()) return false;
    const auto images = resources.StorageImages();
    if (std::any_of(images.begin(), images.end(), [](const auto& image) { return image.second; })) return false;
    return std::all_of(plan.owners.begin(), plan.owners.end(), [](const std::shared_ptr<StorageTexture>& target) { return target != nullptr && target->DirtyQuiet(); });
}

const char* forkInapplicable(const DrawPlan& plan, const ForkLog& log) {
    if (plan.nothing || plan.skipped) return log.Holds() ? nullptr : "expectation";
    if (plan.targetLost) return "target lost";
    auto* recorder = plan.record.recorder;
    if (recorder == nullptr || recorder != Recorder::Active()) return "recorder";
    if (CaptureInputsEnabled()) return "capture";
    if (plan.recordPlan.readsTarget) return "reads target";
    if (!recorder->ContinuesRenderPass(plan.recordPlan.passKey)) return "pass ended";
    if (recordStoresDue(plan.record)) return "store flush";
    if (!log.Holds()) return "expectation";
    return nullptr;
}

void countForkCut(const char* reason) {
    auto& stats = ForkCounters();
    std::lock_guard lock(stats.mutex);
    ++stats.cuts[reason];
}

void forkBuild(ForkedDraw& forked, const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, const DrawPrebuild& prebuilt) {
    auto& stats = ForkCounters();
    stats.builds.fetch_add(1, std::memory_order_relaxed);
    const auto started = std::chrono::steady_clock::now();
    try {
        forked.prebuilt = std::make_shared<DrawPrebuild>(prebuilt);
        const ForkScope scope(forked.log);
        forked.plan = std::make_unique<DrawPlan>(BuildDraw(context, state, draw, shaders, snapshots, recipeOut, forked.prebuilt));
    } catch (const ForkRefusal& refusal) {
        forked.refused = refusal.site;
    } catch (const std::exception&) {
        CountForkRefusal("thrown");
        forked.refused = "thrown";
    } catch (...) {
        CountForkRefusal("thrown");
        forked.refused = "thrown";
    }
    if (forked.refused != nullptr) {
        forked.plan.reset();
        stats.refused.fetch_add(1, std::memory_order_relaxed);
    }
    forked.buildNs = elapsedNs(started);
    stats.buildNs.fetch_add(forked.buildNs, std::memory_order_relaxed);
}

void forkApply(const Context& context, ForkedDraw& forked) {
    auto& plan = *forked.plan;
    const auto started = std::chrono::steady_clock::now();
    PerformanceTimer timing("Graphics.Draw");
    forked.log.Replay();
    if (!plan.nothing) {
        plan.waitedBefore = planProfiled() ? Recorder::ThreadWaitedMs() : 0.0;
        ApplyDraw(context, plan);
        timing.Mark("draw_and_resource_release");
    }
    const auto applyNs = elapsedNs(started);
    auto& stats = ForkCounters();
    stats.applied.fetch_add(1, std::memory_order_relaxed);
    stats.applyNs.fetch_add(applyNs, std::memory_order_relaxed);
    if (planProfiled() && !plan.nothing) {
        auto& planStats = PlanStatistics();
        planStats.planned.fetch_add(1, std::memory_order_relaxed);
        planStats.buildNs.fetch_add(forked.buildNs, std::memory_order_relaxed);
        planStats.applyNs.fetch_add(applyNs, std::memory_order_relaxed);
        planStats.used.fetch_add(1, std::memory_order_relaxed);
        reportPlan();
    }
}

bool sameBytes(const std::shared_ptr<Buffer>& a, VkDeviceSize aOffset, const std::shared_ptr<Buffer>& b, VkDeviceSize bOffset, std::size_t bytes) {
    if (a == b && aOffset == bOffset) return true;
    if (a == nullptr || b == nullptr) return false;
    const auto left = a->Bytes();
    const auto right = b->Bytes();
    return aOffset + bytes <= left.size() && bOffset + bytes <= right.size() && std::memcmp(left.data() + aOffset, right.data() + bOffset, bytes) == 0;
}

const char* forkDifference(const DrawPlan& fork, const ForkLog& forkLog, const DrawPlan& plan, const ForkLog& witness) {
    if (fork.nothing != plan.nothing || fork.skipped != plan.skipped || fork.targetLost != plan.targetLost) return "outcome";
    if (plan.nothing) return nullptr;
    const auto& a = fork.inputs;
    const auto& b = plan.inputs;
    if (a.indexBytes != b.indexBytes || a.maxIndex != b.maxIndex || a.inPlaceReads != b.inPlaceReads || a.vertexOffsets != b.vertexOffsets || a.indexOffset != b.indexOffset || a.vertexHandles.size() != b.vertexHandles.size() || a.vertexBuffers.size() != b.vertexBuffers.size() || a.fragmentOutputs != b.fragmentOutputs || a.shaderStages != b.shaderStages || a.meshGroups != b.meshGroups) return "draw inputs";
    if ((a.indices == nullptr) != (b.indices == nullptr)) return "index copy";
    if (a.indices != nullptr && (a.indices->Bytes().size() != b.indices->Bytes().size() || !sameBytes(a.indices, 0, b.indices, 0, a.indices->Bytes().size()))) return "index copy";
    for (std::size_t index = 0; index < a.vertexBuffers.size(); ++index) {
        const auto& left = a.vertexBuffers[index];
        const auto& right = b.vertexBuffers[index];
        if ((left == nullptr) != (right == nullptr) || (left != nullptr && (left->Bytes().size() != right->Bytes().size() || !sameBytes(left, 0, right, 0, left->Bytes().size())))) return "vertex copies";
    }
    if (fork.effects.size() != plan.effects.size()) return "input effects";
    for (std::size_t index = 0; index < fork.effects.size(); ++index) {
        const auto& left = fork.effects[index];
        const auto& right = plan.effects[index];
        if (left.keep != right.keep || left.address != right.address || left.bytes != right.bytes || left.use != right.use || left.derived != right.derived || left.copy.reused != right.copy.reused) return "input effects";
    }
    if (fork.targetViews != plan.targetViews || fork.owners != plan.owners) return "targets";
    if (plan.targetLost) return nullptr;
    const bool addressUse = plan.outcome.kind == KindBdaHit;
    if ((fork.resolved.resources != plan.resolved.resources && !addressUse) || fork.resolved.cacheable != plan.resolved.cacheable || fork.resolved.contentKey != plan.resolved.contentKey || fork.outcome.kind != plan.outcome.kind || fork.outcome.addressBased != plan.outcome.addressBased) return "resources";
    const auto sameMoved = [](const ShaderResources::MovedBuffer& x, const ShaderResources::MovedBuffer& y) { return x.allocation == y.allocation && x.address == y.address && x.size == y.size && x.words == y.words; };
    if (!std::equal(fork.resolved.moved.begin(), fork.resolved.moved.end(), plan.resolved.moved.begin(), plan.resolved.moved.end(), sameMoved)) return "moved buffers";
    if (fork.masked.has_value() != plan.masked.has_value()) return "masked state";
    if (fork.pipeline != plan.pipeline) return "pipeline";
    if (plan.skipped) return nullptr;
    if (fork.recordPlan.passKey != plan.recordPlan.passKey || (!addressUse && fork.recordPlan.readsTarget != plan.recordPlan.readsTarget)) return "render pass";
    if (fork.record.proxies != plan.record.proxies) return "completion";
    if (!addressUse && (fork.record.listed != plan.record.listed || fork.record.completion != plan.record.completion || fork.record.waited != plan.record.waited || fork.outcome.reason != plan.outcome.reason)) return "completion";
    const auto& x = fork.recordPlan.drawBindings;
    const auto& y = plan.recordPlan.drawBindings;
    if ((x == nullptr) != (y == nullptr)) return "draw bindings";
    if (x != nullptr) {
        if (x->snapshots.size() != y->snapshots.size()) return "draw bindings";
        for (std::size_t index = 0; index < x->snapshots.size(); ++index) {
            const auto& left = x->snapshots[index];
            const auto& right = y->snapshots[index];
            if (left.address != right.address || left.bytes != right.bytes || !sameBytes(left.buffer, left.offset, right.buffer, right.offset, left.bytes)) return "draw snapshots";
        }
    }
    const auto essential = [](const ForkLog& log) {
        std::vector<ForkLog::Entry> kept;
        for (const auto& entry : log.Entries()) {
            if (entry.kind != ForkLog::Effect::Touch && entry.kind != ForkLog::Effect::Erase && entry.kind != ForkLog::Effect::Counter) kept.push_back(entry);
        }
        return kept;
    };
    const auto left = essential(forkLog);
    const auto right = essential(witness);
    if (!std::equal(left.begin(), left.end(), right.begin(), right.end(), [](const ForkLog::Entry& p, const ForkLog::Entry& q) { return p.kind == q.kind && p.address == q.address && p.bytes == q.bytes; })) return "deferred effects";
    return nullptr;
}

bool forkLogged(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, std::shared_ptr<DrawPrebuild>& prebuilt, PerformanceTimer& timing, std::chrono::steady_clock::time_point started) {
    if (const char* why = forkIneligible(context, state, draw, prebuilt.get())) {
        CountForkRefusal(why);
        return false;
    }
    auto& stats = ForkCounters();
    if (auto* recorder = Recorder::Active(); !VerifyForks() && (recorder == nullptr || !recorder->PassContinuable())) {
        countForkCut("no open pass");
        return false;
    }
    ForkedDraw forked(0);
    forkSpeculation = currentSpeculation;
    forkBuild(forked, context, state, draw, shaders, snapshots, recipeOut, *prebuilt);
    forkSpeculation.reset();
    if (VerifyForks()) {
        ForkLog witness(true, 0);
        std::unique_ptr<DrawPlan> plan;
        {
            const ForkScope scope(witness);
            plan = std::make_unique<DrawPlan>(BuildDraw(context, state, draw, shaders, snapshots, recipeOut, std::move(prebuilt)));
        }
        if (forked.plan != nullptr) {
            stats.verified.fetch_add(1, std::memory_order_relaxed);
            if (const char* what = forkDifference(*forked.plan, forked.log, *plan, witness)) {
                stats.mismatches.fetch_add(1, std::memory_order_relaxed);
                std::fprintf(stderr, "[fork] APS5_VERIFY_FORK: the forked build's %s differs from the committer's\n", what);
                std::fflush(stderr);
                std::abort();
            }
        }
        forked.plan.reset();
        return finishPlanned(context, state, draw, shaders, snapshots, recipeOut, *plan, timing, started, true);
    }
    if (forked.plan == nullptr) return false;
    if (const char* why = forkInapplicable(*forked.plan, forked.log)) {
        countForkCut(why);
        return false;
    }
    const bool speculated = forked.plan->resolved.speculated != nullptr;
    if (const char* why = forkFinish(*forked.plan)) {
        countForkCut(why);
        return false;
    }
    if (const char* why = speculated ? forkInapplicable(*forked.plan, forked.log) : nullptr) {
        countForkCut(why);
        return false;
    }
    stats.replayed.fetch_add(1, std::memory_order_relaxed);
    forkApply(context, forked);
    return true;
}

bool drawPlanned(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut, std::shared_ptr<DrawPrebuild> prebuilt) {
    if (draw.indirect) {
        countPlanFallback(PlanFallback::Indirect);
        return false;
    }
    if (!planLean(context, state)) {
        countPlanFallback(PlanFallback::NotLean);
        return false;
    }
    if (prebuilt != nullptr && (prebuilt->refused || prebuilt->device != context.device)) {
        countPlanFallback(prebuilt->refused ? PlanFallback::PrepRefused : PlanFallback::Device);
        prebuilt = nullptr;
    } else if (prebuilt == nullptr && ParallelDrawBuild()) {
        countPlanFallback(PlanFallback::NoPrebuild);
    }
    if (prebuilt != nullptr && verifyDrawPlan()) verifyPrebuild(context, state, draw, shaders, *prebuilt);
    const bool used = prebuilt != nullptr;
    PerformanceTimer timing("Graphics.Draw");
    const auto started = planProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (prebuilt != nullptr && ForkLogEnabled() && forkLogged(context, state, draw, shaders, snapshots, recipeOut, prebuilt, timing, started)) return true;
    auto plan = BuildDraw(context, state, draw, shaders, snapshots, recipeOut, std::move(prebuilt));
    return finishPlanned(context, state, draw, shaders, snapshots, recipeOut, plan, timing, started, used);
}

}

bool DrawPlansEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_DRAW_PLAN");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("plan", enabled);
}

bool ParallelDrawBuild() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_PARALLEL_DRAW_BUILD");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return LiveFlag("parallel_build", enabled);
}

std::shared_ptr<const ForkSeed> TakeForkSeed() {
    auto seed = std::make_shared<ForkSeed>();
    seed->memo = ThreadTargetMemo();
    seed->current = seed->memo.epoch != 0 && seed->memo.epoch == GuestMemory::ThreadCollectEpoch();
    return seed;
}

std::shared_ptr<ForkedDraw> ForkDrawSpeculated(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const std::shared_ptr<DrawPrebuild>& prebuilt, std::shared_ptr<const DrawRecipe>* recipeOut, const ForkSeed* seed, bool seedCurrent, std::size_t slot, std::shared_ptr<DrawSpeculation> speculation) {
    forkSpeculation = std::move(speculation);
    struct Reset {
        ~Reset() { forkSpeculation.reset(); }
    } reset;
    return ForkDraw(context, state, draw, shaders, snapshots, prebuilt, recipeOut, seed, seedCurrent, slot);
}

std::shared_ptr<ForkedDraw> ForkDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const std::shared_ptr<DrawPrebuild>& prebuilt, std::shared_ptr<const DrawRecipe>* recipeOut, const ForkSeed* seed, bool seedCurrent, std::size_t slot) {
    auto forked = std::make_shared<ForkedDraw>(slot);
    if (const char* why = forkIneligible(context, state, draw, prebuilt.get())) {
        CountForkRefusal(why);
        forked->refused = why;
        return forked;
    }
    if (seed != nullptr) {
        auto& memo = ThreadTargetMemo();
        memo = seed->memo;
        memo.epoch = seed->current && seedCurrent ? GuestMemory::ThreadCollectEpoch() : 0;
    }
    forkBuild(*forked, context, state, draw, shaders, snapshots, recipeOut, *prebuilt);
    return forked;
}

std::shared_ptr<const ForkStamp> TakeForkStamp(const Context& context) {
    return std::make_shared<ForkStamp>(stampNow(context));
}

const char* ForkedDrawRefusal(const ForkedDraw& forked) {
    return forked.refused;
}

const char* ForkedDrawCut(const ForkedDraw& forked) {
    return forked.cut;
}

bool ForkedDrawNeedsFinish(const ForkedDraw& forked) {
    return forked.plan != nullptr && forked.plan->resolved.speculated != nullptr && !forked.plan->nothing && !forked.plan->skipped;
}

bool PrepareForkedDraw(ForkedDraw& forked) {
    if (!ForkedDrawNeedsFinish(forked)) return forked.plan != nullptr;
    if (const char* why = forkFinish(*forked.plan)) {
        forked.cut = why;
        countForkCut(why);
        forked.plan.reset();
        return false;
    }
    return true;
}

bool ApplyForkedDraw(const Context& context, ForkedDraw& forked, const ForkStamp& stamp, const std::function<void()>* quiesce) {
    const auto cut = [&](const char* why) {
        forked.cut = why;
        countForkCut(why);
        return false;
    };
    if (forked.plan == nullptr) return forked.cut != nullptr ? false : cut("refused");
    if (!sameStamp(stampNow(context), stamp)) return cut("state moved");
    if (const char* why = forkInapplicable(*forked.plan, forked.log)) return cut(why);
    const bool speculated = ForkedDrawNeedsFinish(forked);
    if (const char* why = forkFinish(*forked.plan)) return cut(why);
    if (const char* why = speculated ? forkInapplicable(*forked.plan, forked.log) : nullptr) return cut(why);
    if ((quiesce != nullptr || ForkQuiescing()) && !forkQuiet(*forked.plan)) {
        if (quiesce != nullptr) (*quiesce)();
        else ForkQuiesce("apply not quiet");
    }
    forkApply(context, forked);
    return true;
}

void CountForkCut(const char* reason) {
    countForkCut(reason);
}

std::string TakeForkReport() {
    auto& stats = ForkCounters();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto perCount = [](unsigned long long ns, unsigned long long count) { return count != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(count) : 0.0; };
    const auto builds = take(stats.builds);
    const auto refused = take(stats.refused);
    const auto buildNs = take(stats.buildNs);
    const auto applied = take(stats.applied);
    const auto applyNs = take(stats.applyNs);
    const auto replayed = take(stats.replayed);
    const auto verified = take(stats.verified);
    const auto mismatches = take(stats.mismatches);
    const auto reuses = static_cast<unsigned long long>(forkSpeculationReuses.exchange(0, std::memory_order_relaxed));
    char head[448];
    std::snprintf(head, sizeof(head), "builds %llu (refused %llu = %.0f%%, %.1f us each, %llu reused a prep speculation), applied %llu at %.1f us each (%llu serial replays), verified %llu with %llu mismatches; refusals:", builds, refused, builds != 0 ? 100.0 * static_cast<double>(refused) / static_cast<double>(builds) : 0.0, perCount(buildNs, builds), reuses, applied, perCount(applyNs, applied), replayed, verified, mismatches);
    std::string line = head;
    const auto refusals = TakeForkRefusals();
    for (std::size_t index = 0; index < refusals.sites.size() && index < 10; ++index) line += " " + refusals.sites[index].first + " " + std::to_string(refusals.sites[index].second);
    if (refusals.backstops != 0) line += " (backstop traps " + std::to_string(refusals.backstops) + ")";
    line += "; cuts:";
    std::lock_guard lock(stats.mutex);
    for (const auto& [reason, count] : stats.cuts) line += " " + reason + " " + std::to_string(count);
    stats.cuts.clear();
    return line;
}

std::shared_ptr<DrawPrebuild> PrebuildDraw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, bool keyed) {
    const auto started = planProfiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::shared_ptr<DrawPrebuild> prebuilt;
    if (!shaders.empty() && std::all_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr; })) {
        try {
            prebuilt = buildPrebuild(context, state, draw, shaders, keyed);
        } catch (...) {
            prebuilt = nullptr;
        }
    }
    if (prebuilt == nullptr) {
        prebuilt = std::make_shared<DrawPrebuild>();
        prebuilt->refused = true;
    }
    if (planProfiled()) {
        auto& stats = PlanStatistics();
        stats.prebuilt.fetch_add(1, std::memory_order_relaxed);
        stats.prepNs.fetch_add(elapsedNs(started), std::memory_order_relaxed);
    }
    return prebuilt;
}

DrawPrebuildScope::DrawPrebuildScope(std::shared_ptr<DrawPrebuild> prebuilt) {
    currentPrebuild = std::move(prebuilt);
}

DrawPrebuildScope::~DrawPrebuildScope() {
    currentPrebuild.reset();
}

std::optional<std::string> KnownValidationFailure(const Context& context, std::span<const CompiledShader> shaders, const State& state) {
    std::vector<std::uint64_t> key;
    if (!ValidationKey(context, shaders, state, key)) return std::nullopt;
    std::lock_guard lock(validationMutex());
    const auto& failures = validationFailures();
    if (const auto found = failures.find(key); found != failures.end()) return found->second;
    return std::nullopt;
}

void Draw(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, std::shared_ptr<const DrawRecipe>* recipeOut) {
    static const bool profileTokens = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct TokenTiming {
        bool on;
        std::chrono::steady_clock::time_point start;
        ~TokenTiming() {
            if (!on) return;
            const auto slot = static_cast<std::size_t>(std::min<std::uint8_t>(drawTokenState, 2));
            Tokens().draws[slot].fetch_add(1, std::memory_order_relaxed);
            Tokens().drawNs[slot].fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
        }
    } tokenTiming{profileTokens && DrawTokensEnabled(), {}};
    if (tokenTiming.on) {
        tokenTiming.start = std::chrono::steady_clock::now();
        drawTokenState = 0;
    }
    auto prebuilt = std::exchange(currentPrebuild, nullptr);
    const auto traceTarget = DebugValue(DebugKey::TraceTarget);
    const auto traceSpan = DebugValue(DebugKey::TraceTargetSpan);
    const auto traceDepth = DebugValue(DebugKey::TraceDepth);
    if (traceDepth != 0 && state.depth && state.depth->address == traceDepth) {
        static std::atomic<std::uint64_t> depthHits{0};
        const auto count = depthHits.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 300 || count % 2000 == 0) {
            std::fprintf(stderr, "[trace-depth] draw %llu colors %zu first 0x%llx indices %u instances %u ztest %d zwrite %d zcmp %d stencil %d front(cmp %d ref %u mask 0x%x write 0x%x pass %d) bias %d htile 0x%llx clear %g vp depth (%g,%g) neg11 %d clamp %d vp %gx%g array 0x%llx slice %u of %u layers %u\n", static_cast<unsigned long long>(count), state.colors.size(), state.colors.empty() ? 0ull : static_cast<unsigned long long>(state.colors.front().address), draw.indexCount, draw.instanceCount, state.depthTest ? 1 : 0, state.depthWrite ? 1 : 0, static_cast<int>(state.depthCompare), state.stencilTest ? 1 : 0, static_cast<int>(state.stencilFront.compareOp), state.stencilFront.reference, state.stencilFront.compareMask, state.stencilFront.writeMask, static_cast<int>(state.stencilFront.passOp), state.depthBias ? 1 : 0, static_cast<unsigned long long>(state.depth->htileAddress), state.depth->clearDepth, state.viewport.minDepth, state.viewport.maxDepth, state.negativeOneToOne ? 1 : 0, state.depthClamp ? 1 : 0, state.viewport.width, state.viewport.height, static_cast<unsigned long long>(state.depth->arrayAddress), state.depth->arraySlice, state.depth->arraySlices, state.depth->layers);
        }
    }
    static const bool traceAllDraws = std::getenv("APS5_TRACE_ALL_DRAWS") != nullptr;
    if (traceAllDraws) {
        static std::atomic<std::uint64_t> drawIndex{0};
        const auto index = drawIndex.fetch_add(1, std::memory_order_relaxed) + 1;
        if (index <= 8000) {
            std::string colors;
            for (const auto& color : state.colors) {
                char text[64];
                std::snprintf(text, sizeof(text), " 0x%llx/vk%d/%ux%u", static_cast<unsigned long long>(color.address), static_cast<int>(color.format), color.extent.width, color.extent.height);
                colors += text;
            }
            std::fprintf(stderr, "[draw-all] %llu indices %u instances %u depth 0x%llx ztest %d zcmp %d colors%s\n", static_cast<unsigned long long>(index), draw.indexCount, draw.instanceCount, state.depth ? static_cast<unsigned long long>(state.depth->address) : 0ull, state.depthTest ? 1 : 0, static_cast<int>(state.depthCompare), colors.c_str());
        }
    }
    const bool traceNoDepth = DebugValue(DebugKey::TraceTargetNoDepth) != 0;
    if (traceNoDepth && traceTarget != 0 && (state.depthTest || state.stencilTest) && std::ranges::any_of(state.colors, [traceTarget, traceSpan](const ColorTarget& color) { return color.address == traceTarget || (traceSpan != 0 && color.address >= traceTarget && color.address < traceTarget + traceSpan); })) {
        auto relaxed = state;
        relaxed.depthTest = false;
        relaxed.depthWrite = false;
        relaxed.stencilTest = false;
        Draw(context, relaxed, draw, shaders, snapshots, recipeOut);
        return;
    }
    if (const auto traceReads = DebugValue(DebugKey::TraceReads); traceReads != 0) {
        // trace_reads: the draws sampling a texture at the address, with the texture's extent and format
        // and the guest code of every stage (debug aid; bounded output).
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) continue;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.guestDescriptor.size() < 8) continue;
                for (std::size_t element = 0; (element + 1) * 8 <= binding.guestDescriptor.size(); ++element) {
                    const auto* t = binding.guestDescriptor.data() + element * 8;
                    const std::uint64_t base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
                    if (base != traceReads) continue;
                    static std::atomic<std::uint64_t> reads{0};
                    const auto count = reads.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (count > 600) continue;
                    const auto width = ((t[1] >> 30u) | ((t[2] & 0xfffu) << 2u)) + 1u;
                    const auto height = ((t[2] >> 14u) & 0x3fffu) + 1u;
                    std::string codes;
                    for (const auto& stage : shaders) {
                        if (stage.program == nullptr) continue;
                        char item[48];
                        std::snprintf(item, sizeof(item), " stage%d 0x%llx", static_cast<int>(stage.stage), static_cast<unsigned long long>(GpuCheckpoints::VariantAddress(stage.program->variantId)));
                        codes += item;
                    }
                    std::fprintf(stderr, "[trace-read] %llu 0x%llx %ux%u format %u tile %u by stage %d binding %u element %zu; draw extent %ux%u color 0x%llx vk %d;%s\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(base), width, height, (t[1] >> 20u) & 0x1ffu, (t[3] >> 20u) & 0x1fu, static_cast<int>(shader.stage), binding.binding, element, state.renderExtent.width, state.renderExtent.height, state.colors.empty() ? 0ull : static_cast<unsigned long long>(state.colors.front().address), state.colors.empty() ? 0 : static_cast<int>(state.colors.front().format), codes.c_str());
                }
            }
        }
    }
    if (const auto traceInto = DebugValue(DebugKey::TraceReadsInto); traceInto != 0 && std::ranges::any_of(state.colors, [traceInto](const ColorTarget& color) { return color.address == traceInto; })) {
        // trace_reads_into: each distinct texture sampled by draws into that color target, with the
        // draw's scissor and the guest code of its stages (debug aid; bounded output).
        static std::mutex seenMutex;
        static std::set<std::tuple<std::uint64_t, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t, std::int32_t>> seen;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) continue;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.guestDescriptor.size() < 8) continue;
                for (std::size_t element = 0; (element + 1) * 8 <= binding.guestDescriptor.size(); ++element) {
                    const auto* t = binding.guestDescriptor.data() + element * 8;
                    const std::uint64_t base = (static_cast<std::uint64_t>(t[0]) | (static_cast<std::uint64_t>(t[1] & 0xffu) << 32u)) << 8u;
                    if (base == 0) continue;
                    const auto width = ((t[1] >> 30u) | ((t[2] & 0xfffu) << 2u)) + 1u;
                    const auto height = ((t[2] >> 14u) & 0x3fffu) + 1u;
                    const auto format = (t[1] >> 20u) & 0x1ffu;
                    {
                        std::lock_guard lock(seenMutex);
                        if (seen.size() > 2000 || !seen.emplace(base, width, height, format, state.scissor.offset.x, state.scissor.offset.y).second) continue;
                    }
                    std::string codes;
                    for (const auto& stage : shaders) {
                        if (stage.program == nullptr) continue;
                        char item[48];
                        std::snprintf(item, sizeof(item), " stage%d 0x%llx", static_cast<int>(stage.stage), static_cast<unsigned long long>(GpuCheckpoints::VariantAddress(stage.program->variantId)));
                        codes += item;
                    }
                    std::fprintf(stderr, "[trace-into] 0x%llx %ux%u format %u tile %u by stage %d binding %u element %zu; scissor %d,%d %ux%u extent %ux%u;%s\n", static_cast<unsigned long long>(base), width, height, format, (t[3] >> 20u) & 0x1fu, static_cast<int>(shader.stage), binding.binding, element, state.scissor.offset.x, state.scissor.offset.y, state.scissor.extent.width, state.scissor.extent.height, state.renderExtent.width, state.renderExtent.height, codes.c_str());
                }
            }
        }
    }
    const auto traceMinWidth = DebugValue(DebugKey::TraceTargetMinWidth);
    const auto traceMaxWidth = DebugValue(DebugKey::TraceTargetMaxWidth);
    const bool traceWidthOk = state.renderExtent.width >= traceMinWidth && (traceMaxWidth == 0 || state.renderExtent.width <= traceMaxWidth);
    const auto traceFormat = DebugValue(DebugKey::TraceTargetFormat);
    if (traceTarget != 0 && traceWidthOk && std::ranges::any_of(state.colors, [traceTarget, traceSpan, traceFormat](const ColorTarget& color) { return (color.address == traceTarget || (traceSpan != 0 && color.address >= traceTarget && color.address < traceTarget + traceSpan)) && (traceFormat <= 1 || static_cast<std::uint64_t>(color.format) == traceFormat); })) {
        auto& hits = g_traceTargetDraws;
        static std::atomic<std::int64_t> lastReport{0};
        const auto count = hits.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = lastReport.load(std::memory_order_relaxed);
        const bool periodic = now - last > std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(10)).count() && lastReport.compare_exchange_strong(last, now);
        static std::atomic<std::int64_t> formatHits{0};
        // The budget restarts whenever the trace settings change.
        static std::atomic<std::uint64_t> formatKey{0};
        const auto key = traceTarget ^ (traceFormat << 48u) ^ (traceMinWidth << 32u) ^ (traceMaxWidth << 16u);
        if (formatKey.exchange(key, std::memory_order_relaxed) != key) formatHits.store(0, std::memory_order_relaxed);
        if (count <= 40 || periodic || (traceFormat != 0 && formatHits.fetch_add(1, std::memory_order_relaxed) < 400)) {
            std::string line = "[trace-target] draw " + std::to_string(count) + " indices " + std::to_string(draw.indexCount) + " instances " + std::to_string(draw.instanceCount) + " extent " + std::to_string(state.renderExtent.width) + "x" + std::to_string(state.renderExtent.height) + " shaders " + std::to_string(shaders.size());
            if (state.depth) {
                char depth[160];
                std::snprintf(depth, sizeof(depth), " depth 0x%llx %ux%u vk %d", static_cast<unsigned long long>(state.depth->address), state.depth->extent.width, state.depth->extent.height, static_cast<int>(state.depth->format));
                line += depth;
                char tests[240];
                std::snprintf(tests, sizeof(tests), " ztest %d zwrite %d zcmp %d bounds %d (%g,%g) bias %d stencil %d front(cmp %d ref %u mask 0x%x write 0x%x pass %d fail %d zfail %d) htile 0x%llx clear %g", state.depthTest ? 1 : 0, state.depthWrite ? 1 : 0, static_cast<int>(state.depthCompare), state.depthBoundsTest ? 1 : 0, state.minDepthBounds, state.maxDepthBounds, state.depthBias ? 1 : 0, state.stencilTest ? 1 : 0, static_cast<int>(state.stencilFront.compareOp), state.stencilFront.reference, state.stencilFront.compareMask, state.stencilFront.writeMask, static_cast<int>(state.stencilFront.passOp), static_cast<int>(state.stencilFront.failOp), static_cast<int>(state.stencilFront.depthFailOp), static_cast<unsigned long long>(state.depth->htileAddress), state.depth->clearDepth);
                line += tests;
            } else {
                line += " no depth";
            }
            for (const auto& color : state.colors) {
                char item[400];
                const auto& blend = color.exportIndex < state.blends.size() ? state.blends[color.exportIndex] : VkPipelineColorBlendAttachmentState{};
                std::snprintf(item, sizeof(item), " | slot %u export %u addr 0x%llx vk %d map 0x%x pack %u mask 0x%x blend %u (%d,%d,%d) surface 0x%llx mip %u/%u slice %u layers %u array %u layerExports 0x%x", color.slot, color.exportIndex, static_cast<unsigned long long>(color.address), static_cast<int>(color.format), color.componentMapping, static_cast<unsigned>(color.packing), blend.colorWriteMask, blend.blendEnable, static_cast<int>(blend.srcColorBlendFactor), static_cast<int>(blend.dstColorBlendFactor), static_cast<int>(blend.colorBlendOp), static_cast<unsigned long long>(color.surfaceAddress), color.mip, color.mipCount, color.depthSlice, color.layers, color.arraySize, state.layerExports);
                line += item;
            }
            std::fprintf(stderr, "%s\n", line.c_str());
        }
        static std::atomic<std::uint64_t> seenDumpSetting{~0ull};
        static std::atomic<std::uint64_t> dumpBase{0};
        const auto dumpSetting = DebugValue(DebugKey::TraceTargetDumpAt);
        if (const auto previous = seenDumpSetting.exchange(dumpSetting); previous != dumpSetting) dumpBase = previous == ~0ull ? 0ull : count;
        const auto dumpAt = dumpSetting != 0 ? dumpBase.load() + dumpSetting : 0ull;
        if (dumpAt != 0 &&(count == dumpAt || count == dumpAt + 100 || count == dumpAt + 1000)) {
            std::vector<std::uint64_t> addresses;
            for (const auto& color : state.colors) addresses.push_back(color.address);
            if (const auto extra = DebugValue(DebugKey::DumpExtra); extra != 0) addresses.push_back(extra);
            std::fprintf(stderr, "[trace-target] dumping targets before draw %llu\n", static_cast<unsigned long long>(count));
            DumpCachedStorageImages(context.device, "target_dump_" + std::to_string(count), addresses);
            if (state.depth) DumpDepthSurface(state.depth->address, "target_dump_" + std::to_string(count) + "/depth.raw");
        }
        // Local debug aid: every 500th traced draw, print each stage's push constants and the
        // first 64 dwords of every buffer (V#) it binds, read from guest memory.
        static std::atomic<std::int64_t> lastSkyReport{0};
        const auto skyNow = std::chrono::steady_clock::now().time_since_epoch().count();
        auto skyLast = lastSkyReport.load(std::memory_order_relaxed);
        const bool skyDue = draw.indexCount == 4 && skyNow - skyLast > std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::milliseconds(400)).count() && lastSkyReport.compare_exchange_strong(skyLast, skyNow);
        if (skyDue) {
            for (const auto& shader : shaders) {
                if (shader.program == nullptr) continue;
                std::string push;
                for (std::size_t offset = 0; offset + 4 <= shader.program->pushConstants.size(); offset += 4) {
                    std::uint32_t word = 0;
                    std::memcpy(&word, shader.program->pushConstants.data() + offset, 4);
                    char item[12];
                    std::snprintf(item, sizeof(item), " %08x", word);
                    push += item;
                }
                std::fprintf(stderr, "[trace-cb] draw %llu stage %d push%s\n", static_cast<unsigned long long>(count), static_cast<int>(shader.stage), push.c_str());
                for (const auto& binding : shader.program->bindings) {
                    for (std::size_t element = 0; element * 4 + 4 <= binding.guestDescriptor.size() && element < 8; ++element) {
                        const auto* v = binding.guestDescriptor.data() + element * 4;
                        const std::uint64_t base = static_cast<std::uint64_t>(v[0]) | (static_cast<std::uint64_t>(v[1] & 0xffffu) << 32u);
                        if (base < 0x100000000ull) continue;
                        MEMORY_BASIC_INFORMATION mbi{};
                        if (VirtualQuery(reinterpret_cast<const void*>(base), &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) continue;
                        const auto available = reinterpret_cast<std::uintptr_t>(mbi.BaseAddress) + mbi.RegionSize - base;
                        const auto words = std::min<std::uint64_t>(64, available / 4);
                        std::string data;
                        for (std::uint64_t i = 0; i < words; ++i) {
                            char item[12];
                            std::snprintf(item, sizeof(item), " %08x", reinterpret_cast<const std::uint32_t*>(base)[i]);
                            data += item;
                        }
                        std::fprintf(stderr, "[trace-cb] draw %llu stage %d binding %u element %zu base 0x%llx:%s\n", static_cast<unsigned long long>(count), static_cast<int>(shader.stage), binding.binding, element, static_cast<unsigned long long>(base), data.c_str());
                    }
                }
            }
        }
        if (count <= 3) {
            for (const auto& shader : shaders) {
                if (shader.program == nullptr) continue;
                for (const auto& binding : shader.program->bindings) {
                    if (binding.guestDescriptor.size() < 8) continue;
                    for (std::size_t element = 0; element * 8 < binding.guestDescriptor.size() && element < 16; ++element) {
                        std::string words;
                        for (std::size_t word = element * 8; word < binding.guestDescriptor.size() && word < element * 8 + 8; ++word) {
                            char item[12];
                            std::snprintf(item, sizeof(item), " %08x", binding.guestDescriptor[word]);
                            words += item;
                        }
                        std::fprintf(stderr, "[trace-target] stage %d binding %u element %zu of %u kind %u role %u:%s\n", static_cast<int>(shader.stage), binding.binding, element, binding.count, static_cast<unsigned>(binding.kind), static_cast<unsigned>(binding.role), words.c_str());
                    }
                }
                for (const auto& attribute : shader.program->vertexAttributes) {
                    const auto& fields = attribute.resource.fields;
                    std::fprintf(stderr, "[trace-target] attribute location %u components %u format 0x%x stride %u fetch %u words %08x %08x %08x %08x\n", attribute.location, attribute.components, (fields[3] >> 12u) & 0x7fu, (fields[1] >> 16u) & 0x3fffu, attribute.fetchIndex, fields[0], fields[1], fields[2], fields[3]);
                }
            }
        }
        if (count == 1) {
            for (std::size_t index = 0; index < shaders.size(); ++index) {
                if (shaders[index].program == nullptr) continue;
                const auto& words = shaders[index].program->spirv.Words();
                const auto name = "trace_target_" + std::to_string(index) + "_stage" + std::to_string(static_cast<int>(shaders[index].stage)) + ".spv";
                if (std::FILE* file = std::fopen(name.c_str(), "wb")) {
                    std::fwrite(words.data(), sizeof(std::uint32_t), words.size(), file);
                    std::fclose(file);
                }
            }
        }
    }
    if (DrawPlansEnabled()) {
        if (drawPlanned(context, state, draw, shaders, snapshots, recipeOut, std::move(prebuilt))) return;
    } else if (prebuilt != nullptr) {
        countPlanFallback(PlanFallback::PlanOff);
    }
    PerformanceTimer timing("Graphics.Draw");
    // APS5_PROFILE_DRAW prints the time of each phase of the draw (microseconds) and the [draws] totals.
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    DrawTimer timer(profile);
    DrawOutcome outcome;
    // The thread's GPU waits so far: the difference at the report, less the draw's own syncs
    // (`ownWaitedMs`, sampled around them), is what this draw waited for through the flush hook
    // (nested in the draw's hold, see DrawOutcome::hookWaitUs).
    const auto waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    auto inputs = prepareDrawInputs(context, state, draw, shaders, outcome, timer, nullptr);
    if (inputs.nothing) return;
    timing.Mark("validate");
    timing.Mark("index_upload");
    drawPrepared(context, state, draw, shaders, snapshots, recipeOut, inputs, timing, timer, outcome, waitedBefore);
}

DrawRecipeOutcome DrawWithRecipe(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const DrawRecipe& recipe) {
    static_cast<void>(snapshots);
    PerformanceTimer timing("Graphics.DrawWithRecipe");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    DrawTimer timer(profile);
    DrawRecipeOutcome result;
    DrawOutcome outcome;
    outcome.kind = KindRecipeHit;
    const auto waitedBefore = profile ? Recorder::ThreadWaitedMs() : 0.0;
    double ownWaitedMs = 0;
    const auto miss = [&](DrawRecipeMiss reason) {
        result.miss = reason;
        if (profile) {
            auto& stats = Profile();
            std::lock_guard lock(stats.mutex);
            ++stats.recipeMisses[static_cast<std::size_t>(reason)];
        }
        return result;
    };
    Require(!draw.indirect, "a draw recipe covers direct draws only");
    auto* recorder = Recorder::Active();
    if (!RecordDraws() || recorder == nullptr || DumpTargetLimit() != 0) return miss(DrawRecipeMiss::NotRecordable);
    Require(recipe.targets.size() == state.colors.size() && recipe.targetViews.size() == state.colors.size() + (recipe.depth ? 1u : 0u), "draw recipe targets do not match the state");
    // The draw key names the depth registers, so a hit has a depth target iff the recipe's draw
    // had one; a disagreement is a miss, never a record against the wrong attachments.
    if (recipe.depth != static_cast<bool>(state.depth)) return miss(DrawRecipeMiss::DepthGone);
    auto inputs = prepareDrawInputs(context, state, draw, shaders, outcome, timer, &recipe);
    if (inputs.nothing) {
        result.recorded = true;
        return result;
    }
    // The resident-target proof on the stored images (design_cpu_final 3.7): while an image is
    // still the storage cache's for its surface (StorageImageCached: the cache entry, removed under
    // the cache mutex before an eviction's flush, and the LRU touch a lookup would make), it is
    // what the lookup would return, and Refresh brings it up to date exactly as the lookup's does
    // (FlushPending, the whole-surface collect and UnchangedSince, the key scan). An image gone
    // from the cache is a miss: the ordinary path looks up anew.
    std::vector<std::shared_ptr<StorageTexture>> targets;
    targets.reserve(recipe.targets.size());
    for (std::size_t index = 0; index < recipe.targets.size(); ++index) {
        timer.phase(PhaseSetup);
        auto stored = recipe.targets[index].lock();
        // The previous draw's target in the same collect epoch and continued pass (MemoizedTarget)
        // was refreshed then; nothing collectable has changed since, as on the build paths.
        if (stored != nullptr && RecipeTargetMemo()) {
            if (auto memoized = MemoizedTarget(index, state); memoized == stored) {
                targets.push_back(std::move(memoized));
                continue;
            }
        }
        if (stored == nullptr || !StorageImageCached(context, stored.get()) || !StorageImageServesKeys(*stored, state.colors[index].dccAddress)) return miss(DrawRecipeMiss::TargetGone);
        auto resident = refreshResidentTarget(context, state, state.colors[index], outcome, profile, [&] {
            stored->Refresh();
            return stored;
        });
        timer.phase(PhaseReadTarget);
        if (resident == nullptr) return miss(DrawRecipeMiss::TargetGone);
        targets.push_back(std::move(resident));
    }
    // The depth proof: the ordinary path's whole per-draw depth work is DepthSurfaceView (Draw and
    // BuildDraw, after the color targets and before the template lookup, as here): it finds the
    // surface for the state's depth target (address, stencil plane, extent, format, layers; the
    // array slice's layer view for a sliced target), applies its pending fast clear (an HTILE fill
    // noted by NoteDepthMetadataFill, or the per-frame debug clear), takes back the writes of a
    // storage image seeded from it and bumps its generation (what sampled plane copies compare).
    // The draw records into that view; anything else (another surface, a new one) is a miss. A
    // template sampling the depth surface is proved by ProveCurrent below exactly as on a template
    // hit; one holding a per-use plane copy is never reusable and so never gets a recipe.
    if (state.depth) {
        timer.phase(PhaseSetup);
        const auto depthView = DepthSurfaceView(context, *state.depth);
        timer.phase(PhaseReadTarget);
        if (depthView == VK_NULL_HANDLE || depthView != recipe.depthView || recipe.targetViews.back() != depthView) return miss(DrawRecipeMiss::DepthGone);
        result.depth = true;
    }
    timer.phase(PhasePrepare);
    // The template's proof (rules R6/R7): ProveCurrent, T1 included, the alias checks the trimmed
    // key leaves to a hit repeated; a failure removes the template from the cache (the batch keeps
    // it) and the caller rebuilds.
    auto resources = recipe.templateRef.lock();
    if (resources == nullptr) return miss(DrawRecipeMiss::TemplateGone);
    Require(resources->Reusable(), "draw recipe over a non-reusable template");
    // A recipe over a stream template never reaches resolveDrawResources' drop, so drop it here too
    // once the stream_ring switch is off (the A/B compares against today's behaviour).
    if (resources->Streamed() && !ShaderResources::StreamRing()) {
        SharedResourceCache().Remove(recipe.key, resources.get());
        countCache(&DrawProfile::cacheInvalidated);
        countCache(&DrawProfile::streamDropped);
        recorder->Keep(std::move(resources));
        timer.phase(PhaseLookup);
        return miss(DrawRecipeMiss::Proof);
    }
    const auto proofStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool proved = resources->ProveCurrent(shaders, &result.proof);
    if (profile) result.proofUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - proofStart).count();
    if (!proved) {
        SharedResourceCache().Remove(recipe.key, resources.get());
        countCache(&DrawProfile::cacheInvalidated);
        recorder->Keep(std::move(resources));
        timer.phase(PhaseLookup);
        return miss(DrawRecipeMiss::Proof);
    }
    // A moved-buffer recipe recomputes the moved read-only buffers as the template hit does
    // (resolveDrawResources: after the proof, before the alias checks); a refusal is a miss and the
    // ordinary path decides again. A recipe built without moved buffers has none on a hit (the
    // draw-cache hit repeats the stage words the template was compared with; verified below).
    std::vector<ShaderResources::MovedBuffer> moved;
    if (recipe.movedBuffers) {
        auto found = resources->MovedReadOnlyBuffers(shaders, *recorder);
        if (!found.has_value()) {
            timer.phase(PhaseLookup);
            return miss(DrawRecipeMiss::MovedRefused);
        }
        moved = std::move(*found);
        result.moved = true;
    }
    CheckBufferAliases(shaders, state.color, draw.indexAddress, inputs.indexBytes);
    SharedResourceCache().Touch(recipe.key);
    countCache(&DrawProfile::cacheHits);
    timer.phase(PhaseLookup);
    outcome.completion = false;
    outcome.recorded = true;
    // The pipeline is the recipe's while the pipeline store still holds it (its device's teardown
    // or its eviction bound drops it: a miss). The framebuffer is the recipe's while the pipeline's
    // list holds it: the views are the stored targets' (stable while they live, and they are the
    // same objects), so AcquireFramebuffer only runs when the list evicted it.
    auto pipeline = recipe.pipeline.lock();
    if (pipeline == nullptr) return miss(DrawRecipeMiss::ObjectsGone);
    if (VerifyDrawRecipeObjects()) {
        const auto fail = [&](const char* what) {
            std::fprintf(stderr, "[recipe] APS5_VERIFY_DRAW_RECIPE: %s (depth %d, moved %d, colors %zu)\n", what, recipe.depth ? 1 : 0, recipe.movedBuffers ? 1 : 0, state.colors.size());
            std::fflush(stderr);
            std::abort();
        };
        if (state.depth && DepthSurfaceView(context, *state.depth) != recipe.depthView) fail("the depth view of the recipe disagrees with DepthSurfaceView");
        if (!recipe.movedBuffers) {
            const auto now = resources->MovedReadOnlyBuffers(shaders, *recorder);
            if (!now.has_value() || !now->empty()) fail("a recipe built without moved buffers has moved buffers on its hit");
        }
        static std::atomic<int> reported{0};
        const auto note = [&](const char* what) {
            result.replaced = true;
            if (reported.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[recipe] APS5_VERIFY_DRAW_RECIPE: %s (depth %d, moved %d)\n", what, recipe.depth ? 1 : 0, recipe.movedBuffers ? 1 : 0);
        };
        if (const auto cached = SharedResourceCache().Find(recipe.key); cached != nullptr && cached != resources) note("the resource cache serves another template under the recipe's key");
        const State& pipelineState = recipe.masked.has_value() ? *recipe.masked : state;
        const auto storageImages = resources->StorageImages();
        const bool storesThroughAddresses = std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->bdaWrites; });
        const bool mayDefer = !resources->HasCopiedWrites() && resources->GpuWrites().empty() && !storesThroughAddresses && std::none_of(storageImages.begin(), storageImages.end(), [](const auto& image) { return image.second; });
        if (const auto current = CachedPipeline(context, pipelineState, recipe.vertexInput, *resources, shaders, VK_IMAGE_LAYOUT_GENERAL, mayDefer); current != nullptr && current != pipeline) note("the pipeline store serves another pipeline for the recipe's state");
        result.verified = true;
    }
    auto framebuffer = recipe.framebuffer.lock();
    if (framebuffer == nullptr) framebuffer = pipeline->AcquireFramebuffer(recipe.targetViews, targets, state.renderExtent, state.renderLayers);
    timer.phase(PhasePipeline);
    const auto recordStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    RecordedDraw record;
    record.recorder = recorder;
    record.resources = std::move(resources);
    record.pipeline = std::move(pipeline);
    record.framebuffer = std::move(framebuffer);
    record.targetViews = recipe.targetViews;
    record.targets = std::move(targets);
    record.moved = moved;
    record.pushBytes = &recipe.pushBytes;
    record.pushStages = recipe.pushStages;
    recordDraw(context, state, draw, shaders, inputs, record, outcome, timer, ownWaitedMs);
    if (profile) {
        result.recordUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - recordStart).count();
        auto& stats = Profile();
        std::lock_guard lock(stats.mutex);
        ++stats.recipeHits;
    }
    timing.Mark("draw_and_resource_release");
    reportDrawEnd(state, timer, nullptr, outcome, waitedBefore, ownWaitedMs, " recorded from recipe");
    result.recorded = true;
    return result;
}

namespace {

std::shared_ptr<StorageTexture> metadataPassResident(const Context& context, const ColorTarget& color) {
    if (color.tileMode == ColorTileMode::Linear || context.detiler == nullptr) return nullptr;
    std::shared_ptr<StorageTexture> resident;
    try {
        resident = CachedStorageSurface(context, SurfaceForTarget(color));
    } catch (const std::exception&) {
        return nullptr;
    }
    return resident != nullptr && resident->GuestBytes() == color.bytes ? resident : nullptr;
}

// Whether the draw path keeps a color target in its surface's storage image (planLean's rule).
bool resolveTargetEligible(const Context& context, const ColorTarget& color) {
    static const bool residentLinear = std::getenv("APS5_NO_RESIDENT_LINEAR_TARGETS") == nullptr;
    if (context.detiler == nullptr) return false;
    return color.tileMode == ColorTileMode::RenderTarget || color.tileMode == ColorTileMode::Standard4KB || (residentLinear && color.tileMode == ColorTileMode::Linear && RegisteredReadableCovers(color.address, static_cast<std::size_t>(color.bytes)));
}

// One target of a CB resolve as a draw attaches it: its surface's cached storage image, refreshed
// (a surface never rendered on the GPU is uploaded from its guest bytes), with a pending register
// or CMASK fast clear of the surface materialized into it first.
std::shared_ptr<StorageTexture> resolveTarget(const Context& context, const ColorTarget& color, const char* role) {
    std::shared_ptr<StorageTexture> resident;
    {
        const RefreshSiteScope site("render-target");
        resident = CachedStorageSurface(context, SurfaceForTarget(color));
    }
    const ColorTargetLayout layout(color.extent.width, color.extent.height, color.tileMode, color.elementBytes);
    if (color.mipCount == 1 && resident->GuestBytes() != layout.Bytes()) throw std::runtime_error(std::string("AGC graphics: the storage image of the CB resolve ") + role + " differs from its color layout");
    materializeRegisterClear(context, color, *resident);
    if (materializeCmaskClear(context, color, resident.get())) resident->MarkDirty();
    return resident;
}

// CB resolve (see DecodeColorMetadataPass): the region of the source's storage image is copied on
// the device into the destination's, which is then pending like a draw's target (its guest bytes
// follow through the deferred write-back); the source stays as it was.
void runColorResolve(const Context& context, const ColorMetadataPass& pass) {
    Require(pass.targets.size() == 2, "a CB resolve needs its source and destination targets");
    const auto& from = pass.targets[0];
    const auto& to = pass.targets[1];
    const auto& region = pass.region;
    if (region.extent.width == 0 || region.extent.height == 0) return;
    for (const auto* color : {&from, &to}) {
        if (resolveTargetEligible(context, *color)) continue;
        char text[128];
        std::snprintf(text, sizeof(text), "AGC graphics: CB resolve %s in tile mode %u has no storage image", color == &from ? "source" : "destination", static_cast<unsigned>(color->tileMode));
        throw std::runtime_error(text);
    }
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "a CB resolve needs the active recorder");
    const auto source = resolveTarget(context, from, "source");
    const auto destination = resolveTarget(context, to, "destination");
    Require(source != destination || from.mip != to.mip, "CB resolve onto its own source is unsupported");
    // Same guest format (DecodeColorMetadataPass). An R11G11B10 surface can be stored as
    // B10G11R11_UFLOAT on one side and packed R32_UINT on the other: both are 32-bit texels,
    // which vkCmdCopyImage copies bit for bit (size-compatible formats).
    const auto float11 = [](VkFormat format) { return format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || format == VK_FORMAT_R32_UINT; };
    Require(source->StorageFormat() == destination->StorageFormat() || (from.elementBytes == 4 && to.elementBytes == 4 && float11(source->StorageFormat()) && float11(destination->StorageFormat())), "the CB resolve source and destination storage formats differ");
    const std::array<std::pair<VkImage, bool>, 2> images{{{source->Image(), false}, {destination->Image(), true}}};
    bool elided = false;
    const auto commands = Recorder::BarrierTracking() ? recorder->Commands(Recorder::CommandClass::Copy, Recorder::Access{{}, {}, images, VK_PIPELINE_STAGE_TRANSFER_BIT, false, false, true}, elided) : recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 2);
    recorder->Keep(source);
    recorder->Keep(destination);
    VkImageMemoryBarrier barriers[2]{};
    for (auto& barrier : barriers) {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    }
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barriers[0].image = source->Image();
    barriers[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].image = destination->Image();
    const auto pipelineBarrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    pipelineBarrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, from.mip, 0, 1};
    copy.srcOffset = {region.offset.x, region.offset.y, 0};
    copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, to.mip, 0, 1};
    copy.dstOffset = copy.srcOffset;
    copy.extent = {region.extent.width, region.extent.height, 1};
    context.Function<PFN_vkCmdCopyImage>("vkCmdCopyImage")(commands, source->Image(), VK_IMAGE_LAYOUT_GENERAL, destination->Image(), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    VkImageMemoryBarrier copied = barriers[1];
    copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &copied);
    recorder->EndGpuTiming(timing, static_cast<std::uint64_t>(region.extent.width) * region.extent.height * from.elementBytes);
    destination->MarkDirty();
    static std::atomic<int> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) std::fprintf(stderr, "[gpu] CB resolve 0x%llx -> 0x%llx: %ux%u at (%d, %d), VkFormat %d\n", static_cast<unsigned long long>(from.address), static_cast<unsigned long long>(to.address), region.extent.width, region.extent.height, region.offset.x, region.offset.y, static_cast<int>(from.format));
}

}

void RunColorMetadataPass(const Context& context, const ColorMetadataPass& pass) {
    if (pass.mode == ColorMetadataPass::Mode::Resolve) {
        runColorResolve(context, pass);
        return;
    }
    for (const auto& color : pass.targets) {
        if (color.cmaskAddress != 0) {
            const auto resident = metadataPassResident(context, color);
            if (materializeCmaskClear(context, color, resident.get()) && resident != nullptr) resident->MarkDirty();
        }
        if (color.dccAddress == 0) continue;
        auto keys = CurrentDccKeys(color.dccAddress, color.bytes);
        if (keys == DccKeys::Uncompressed) continue;
        Require(IsDccClear(keys), std::string("CB metadata pass over DCC keys that are ") + DccKeysName(keys) + " (per-block metadata is not modeled)");
        const auto texel = clearTexel(color, keys);
        const auto resident = metadataPassResident(context, color);
        if (resident != nullptr) {
            const char* refusal = nullptr;
            const bool current = keys == DccKeys::ClearRegister ? clearToTexel(*resident, texel, color.elementBytes, refusal) : StorageTexture::FindPending(color.address, color.bytes) == resident || resident->UploadedKeys() == keys;
            if (current) {
                resident->MarkDirty();
                MarkDccUncompressed(context, color.dccAddress, color.bytes, colorKeyCount(color, color.bytes));
                continue;
            }
        }
        storeClearTexels(context, color, texel);
    }
}

}
