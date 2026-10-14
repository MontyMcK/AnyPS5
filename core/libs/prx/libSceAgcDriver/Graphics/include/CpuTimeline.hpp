#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CPUTIMELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CPUTIMELINE_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <utility>

namespace AgcDriver::Graphics::CpuTimeline {

// APS5_GPU_TIMELINE=<start s>,<seconds> opens one window of the run, counted from the first use of
// either timeline (the recorder's creation with the device; before this change the GPU file's
// window counted from its first timed batch, a little later). With APS5_PROFILE_GPU the recorder writes the GPU's timed ranges into
// gpu-timeline.txt over it (the batch rows here carry the batches' stamps without it); this
// writes what the CPU did over the same window into cpu-timeline.txt (the game directory): batch
// submits with their reason, the batches with their GPU stamps and serials, the waits for the GPU,
// GPU mutex holds of a millisecond or more (with APS5_PROFILE_DRAW), EOP interrupt deliveries,
// flips and the queue workers' submissions. Lines are
//   <begin ns> <end ns> <event> <thread> <tid> <a> <b> [text]
// with times in the GPU timestamp domain of gpu-timeline.txt when the device has calibrated
// timestamps (the '# calibration' lines say so), else on the CPU's steady clock.
using Clock = std::chrono::steady_clock;

// The window as steady-clock times, {0, 0} without the switch. Fixed at the first call.
std::pair<std::int64_t, std::int64_t> Window();
// Whether the window is open now, or the file is still open after it so the next event closes it
// (one clock read; false at once without the switch).
bool Active();
// The calibration source (VulkanDevice::CalibrateClock): the GPU time of a CPU time point.
// `owner` identifies the setter: ClearCalibration(owner) removes the source only while it is still
// that owner's (a replaced device's teardown must not drop its successor's source).
void SetCalibration(std::function<bool(double& gpuNs, Clock::time_point& cpu)> calibrate, const void* owner = nullptr);
void ClearCalibration(const void* owner);
// One event. `text` may be null. Takes a mutex of its own; never the GPU mutex.
void Event(const char* event, Clock::time_point begin, Clock::time_point end, std::uint64_t a = 0, std::uint64_t b = 0, const char* text = nullptr);
inline void Instant(const char* event, std::uint64_t a = 0, std::uint64_t b = 0, const char* text = nullptr) {
    if (!Active()) return;
    const auto now = Clock::now();
    Event(event, now, now, a, b, text);
}
// A finished batch: its GPU stamps (already in the GPU domain), when it was submitted, its serial,
// submit reason and whether the GPU had nothing queued at the submit.
void Batch(double gpuStartNs, double gpuEndNs, Clock::time_point submittedAt, std::uint64_t serial, const char* reason, bool starved, std::uint32_t queue);
// Writes buffered lines out (the file is also flushed when the window closes).
void Flush();
// The time a steady-clock point has in the file (GPU domain when calibrated). For tests.
double FileTime(Clock::time_point point);

}

#endif
