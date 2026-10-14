#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/NvPerfRanges.hpp"
#include <cstdlib>
#include <map>
#include <string>

namespace AgcDriver::DriverDetail {

void Driver::RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    require(output != nullptr, "null video output");
    std::lock_guard lock(mutex);
    rethrowFailure();
    checkStopping();
    require(outputs.emplace(handle, output).second, "video output already registered");
}

void Driver::UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    std::lock_guard lock(mutex);
    const auto it = outputs.find(handle);
    require(it != outputs.end() && it->second == output, "video output registration mismatch");
    outputs.erase(it);
}

void Driver::Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context) {

    PerformanceContext timingContext(window.timing.get());
    PerformanceTimer timing("Driver.Present");
    static thread_local bool pinned = false;
    if (!pinned) {
        pinned = true;
        PinWorkerThread("presenter");
        GuestMemory::MarkPresenterThread();
    }
    CheckFailure();
    require(gpuReady != nullptr && context != nullptr, "missing GPU completion callback");
    require(window.getDrawableSize != nullptr, "missing window drawable size query");

    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    static const std::size_t inFlight = VulkanDevice::FlipInFlight();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::shared_ptr<VulkanDevice> presenting;
    timing.Mark("validate");
    try {
        bool submitted = false;
        bool presentable = false;
        bool trailing = false;
        double waitedMs = 0;
        {
            std::unique_lock replacing(deviceReplacement, std::defer_lock);
            if (const auto current = device.Load(); current == nullptr || current->Window() == nullptr) replacing.lock();
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            timing.Mark("gpu_mutex_wait");
            if (device == nullptr || device->Window() == nullptr) {
                // The headless device that served the draws before the first flip takes the window
                // when it can (one device, its caches kept); otherwise a windowed device replaces it.
                if (device == nullptr || !device->AttachWindow(window)) {
                    if (device) {
                        device->PrepareForReplacement();
                        replacedDevices.push_back(device);
                    }
                    device = std::make_shared<VulkanDevice>(&window);
                }
            }
            require(device->Window() == window.context, "presentation window does not match device surface");
            presenting = device;
            timing.Mark("device_setup");
            std::uint32_t drawableWidth = 0;
            std::uint32_t drawableHeight = 0;
            window.getDrawableSize(window.context, &drawableWidth, &drawableHeight);
            presenting->Resize(drawableWidth, drawableHeight);
            presenting->SelectDynamicRange(buffer);
            timing.Mark("resize");
            presentable = presenting->Presentable();
            if (buffer != nullptr) require(buffer->width == window.width && buffer->height == window.height, "display buffer extent differs from output");
        }

        if (presentable && !syncFlip) {
            if (inFlight != 0) {

                waitedMs = presenting->RetirePresents(presenting->PresentWaitsForSlots(buffer) ? 0 : inFlight);
                timing.Mark("inflight_wait");
            }
            presentable = presenting->AcquireImage();
            timing.Mark("acquire_image");
        }
        if (presentable) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            timing.Mark("gpu_mutex_wait");
            Graphics::StorageTexture::StoreSmallPending();
            timing.Mark("small_results_store");
            if (buffer != nullptr) {
                if (syncFlip) {
                    presenting->WaitIdle();
                    timing.Mark("device_idle_wait");
                }
                submitted = presenting->PresentDisplayBuffer(*buffer);
                timing.Mark("present_display_buffer");
            } else {
                submitted = presenting->PresentClear(window.width, window.height, opaque);
                timing.Mark("present_clear");
            }
            // The range profiler's frame ends with the flip's submit (APS5_NVPERF).
            if (submitted) Graphics::NvPerf::FrameBoundary();
            if (submitted && syncFlip) {
                waitedMs = presenting->FinishPresent();
                timing.Mark("render_fence_wait");
            }
            if (submitted && (syncFlip || inFlight != 0)) {

                presenting->QueuePresent();
                timing.Mark("queue_present");
                trailing = !syncFlip;
                submitted = false;
            }
        }
        if (trailing) {
            // APS5_FLIP_EARLY=1: the flip completes once its present is queued; the GPU orders the blit
            // before any later work, and only presentations past the in-flight allowance are waited for.
            static const bool earlyFlip = std::getenv("APS5_FLIP_EARLY") != nullptr;
            waitedMs += earlyFlip ? presenting->RetirePresents(inFlight) : presenting->FinishPresent();
            timing.Mark("render_fence_wait");
        }
        if (submitted) {
            waitedMs = presenting->FinishPresent();
            timing.Mark("render_fence_wait");

            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Present);
            std::lock_guard lock(GuestMemory::GpuMutex());
            presenting->QueuePresent();
            timing.Mark("queue_present");
        }
        gpuReady(context);
        timing.Mark("release_and_callback");
        if (profile) reportPresents(waitedMs, inFlight);
        CheckFailure();
    } catch (const ProcessShutdown&) {
        throw;
    } catch (...) {
        ReportFailure(std::current_exception());
        throw;
    }
}

void Driver::reportPresents(double waitedMs, std::size_t inFlight) {
    static std::vector<double> waits;
    static VulkanDevice::PresentStatistics previous{};
    static std::uint64_t previousFlips = 0, windowSerial = 0, previousUnsignaled = 0;
    static auto lastReport = std::chrono::steady_clock::now();
    waits.push_back(waitedMs);
    if (waits.size() == 1) windowSerial = flipSerial.load();
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::sort(waits.begin(), waits.end());
    const auto percentile = [&](double fraction) { return waits.empty() ? 0.0 : waits[std::min(waits.size() - 1, static_cast<std::size_t>(fraction * static_cast<double>(waits.size())))]; };
    const auto counts = VulkanDevice::PresentCounts();
    const auto presents = counts.presents - previous.presents;
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    const auto flips = flipsCounted.load() - previousFlips;
    const double perFlip = flips != 0 ? 1.0 / static_cast<double>(flips) : 0.0;
    const auto serial = flipSerial.load();
    AgcDriver::ProfilePrint_nid_no_patch( "[present] %llu presents over 10 s (%zu may trail on the GPU); %s p50 %.2f ms, p90 %.2f ms, max %.2f ms over %zu; per present: GPU busy %.1f ms, idle gaps %.1f ms (%llu batches without a completion record); per flip: %.1f batches recorded (%.1f unsignaled at the flip); per present: %.1f batches ahead of the blit, %.1f after it; after the flip packet: last submit %.1f ms, blit submit %.1f ms; in-place reads overwritten before execution: %llu of %llu checked\n", static_cast<unsigned long long>(presents), inFlight, inFlight != 0 ? "inflight_wait" : "render_fence_wait", percentile(0.5), percentile(0.9), waits.empty() ? 0.0 : waits.back(), waits.size(), (counts.gpuBusyMs - previous.gpuBusyMs) * perPresent, (counts.gpuGapMs - previous.gpuGapMs) * perPresent, static_cast<unsigned long long>(counts.gpuUnread - previous.gpuUnread), static_cast<double>(serial - windowSerial) * perFlip, static_cast<double>(flipBatchesUnsignaled.load() - previousUnsignaled) * perFlip, static_cast<double>(counts.batchesAheadOfBlit - previous.batchesAheadOfBlit) * perPresent, static_cast<double>(counts.batchesAfterFlip - previous.batchesAfterFlip) * perPresent, (counts.lastSubmitAfterFlipMs - previous.lastSubmitAfterFlipMs) * perPresent, (counts.blitSubmitAfterFlipMs - previous.blitSubmitAfterFlipMs) * perPresent, static_cast<unsigned long long>(counts.readsOverwritten - previous.readsOverwritten), static_cast<unsigned long long>(counts.readsChecked - previous.readsChecked));
    reportSubmits(counts, previous, presents);
    previous = counts;
    previousFlips = flipsCounted.load();
    previousUnsignaled = flipBatchesUnsignaled.load();
    waits.clear();
}

// [submits] (APS5_PROFILE_DRAW, with [present]): batches per present by submit reason (and how many
// of them found the GPU with nothing queued), the GPU idle per present by the reason of the batch
// that ended it, and the long (>= 5 ms) gaps by that reason.
void Driver::reportSubmits(const VulkanDevice::PresentStatistics& counts, const VulkanDevice::PresentStatistics& previous, std::uint64_t presents) {
    using Recorder = Graphics::Recorder;
    static Recorder::SubmitStatistics before{};
    static std::uint64_t holdsBefore = 0, resetsBefore = 0;
    const auto now = Recorder::SubmitCounts();
    const double perPresent = presents != 0 ? 1.0 / static_cast<double>(presents) : 0.0;
    std::uint64_t total = 0, starved = 0;
    for (std::size_t i = 0; i < Recorder::SubmitReasonCount; ++i) {
        total += now.batches[i] - before.batches[i];
        starved += now.starved[i] - before.starved[i];
    }
    std::string text;
    char part[160];
    std::snprintf(part, sizeof(part), "[submits] per present (%llu presents): %.1f batches (%.1f found the GPU with nothing queued):", static_cast<unsigned long long>(presents), static_cast<double>(total) * perPresent, static_cast<double>(starved) * perPresent);
    text += part;
    for (std::size_t i = 0; i < Recorder::SubmitReasonCount; ++i) {
        const auto count = now.batches[i] - before.batches[i];
        if (count == 0) continue;
        std::snprintf(part, sizeof(part), " %s %.1f (%.1f)", Recorder::SubmitReasonName(static_cast<Recorder::SubmitReason>(i)), static_cast<double>(count) * perPresent, static_cast<double>(now.starved[i] - before.starved[i]) * perPresent);
        text += part;
    }
    double idle = 0, longIdle = 0;
    std::uint64_t longGaps = 0;
    for (std::size_t i = 0; i < Recorder::SubmitReasonCount; ++i) {
        idle += counts.idleMsByReason[i] - previous.idleMsByReason[i];
        longIdle += counts.longIdleMsByReason[i] - previous.longIdleMsByReason[i];
        longGaps += counts.longGapsByReason[i] - previous.longGapsByReason[i];
    }
    std::snprintf(part, sizeof(part), "; GPU idle %.1f ms per present (nothing queued at the next submit %.1f, at a boundary with work queued %.1f) before batches for:", idle * perPresent, (counts.starvedIdleMs - previous.starvedIdleMs) * perPresent, (counts.queuedIdleMs - previous.queuedIdleMs) * perPresent);
    text += part;
    for (std::size_t i = 0; i < Recorder::SubmitReasonCount; ++i) {
        const auto ms = counts.idleMsByReason[i] - previous.idleMsByReason[i];
        if (ms * perPresent < 0.05) continue;
        std::snprintf(part, sizeof(part), " %s %.1f", Recorder::SubmitReasonName(static_cast<Recorder::SubmitReason>(i)), ms * perPresent);
        text += part;
    }
    std::snprintf(part, sizeof(part), "; gaps >= 5 ms: %llu (%.1f ms per present) before:", static_cast<unsigned long long>(longGaps), longIdle * perPresent);
    text += part;
    for (std::size_t i = 0; i < Recorder::SubmitReasonCount; ++i) {
        const auto gaps = counts.longGapsByReason[i] - previous.longGapsByReason[i];
        if (gaps == 0) continue;
        std::snprintf(part, sizeof(part), " %s %llu/%.0f ms", Recorder::SubmitReasonName(static_cast<Recorder::SubmitReason>(i)), static_cast<unsigned long long>(gaps), counts.longIdleMsByReason[i] - previous.longIdleMsByReason[i]);
        text += part;
    }
    const auto holds = Recorder::StarveAgeHolds();
    const auto resets = Recorder::UndrainedFenceResets();
    std::snprintf(part, sizeof(part), "; starve submits held for age %llu, fence resets without a stream drain %llu; unannounced submit sites:", static_cast<unsigned long long>(holds - holdsBefore), static_cast<unsigned long long>(resets - resetsBefore));
    text += part;
    std::map<std::uint64_t, std::uint64_t> sitesBefore(before.otherSites.begin(), before.otherSites.end());
    std::size_t shown = 0;
    for (const auto& [site, count] : now.otherSites) {
        const auto delta = count - sitesBefore[site];
        if (delta == 0 || shown++ >= 6) continue;
        std::snprintf(part, sizeof(part), " +0x%llx x%llu", static_cast<unsigned long long>(site), static_cast<unsigned long long>(delta));
        text += part;
    }
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", text.c_str());
    before = now;
    holdsBefore = holds;
    resetsBefore = resets;
}

void Driver::ReleaseWindow(void* window) {
    std::lock_guard lock(GuestMemory::GpuMutex());
    if (device && device->Window() == window) device.Reset();
}

}
