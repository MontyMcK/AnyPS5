#include "prx/libSceAgcDriver/Graphics/include/CpuTimeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace AgcDriver::Graphics::CpuTimeline {

namespace {

struct State {
    std::mutex mutex;
    std::FILE* file = nullptr;
    bool closed = false;
    std::function<bool(double&, Clock::time_point&)> calibrate;
    const void* owner = nullptr;
    bool calibrated = false;
    double offsetNs = 0;
    Clock::time_point calibratedAt{};
};

State& Shared() {
    static State state;
    return state;
}

// The file is open: Active() stays true past the window until an event closes it.
std::atomic<bool> fileOpen{false};

std::uint64_t ThreadId() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return 0;
#endif
}

// Under the state mutex: the GPU-minus-CPU offset, refreshed every 250 ms (the clocks drift apart
// by microseconds per second; a line per refresh lets a reader see by how much).
void Recalibrate(State& state, Clock::time_point now) {
    if (state.file == nullptr || !state.calibrate) return;
    if (state.calibratedAt != Clock::time_point{} && now - state.calibratedAt < std::chrono::milliseconds(250)) return;
    state.calibratedAt = now;
    // Lines reach the file at least as often, so a run that ends abruptly loses little.
    std::fflush(state.file);
    double gpuNs = 0;
    Clock::time_point cpu{};
    if (!state.calibrate(gpuNs, cpu)) return;
    const auto cpuNs = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu.time_since_epoch()).count());
    const double offset = gpuNs - cpuNs;
    std::fprintf(state.file, "# calibration gpu_ns %.0f cpu_ns %.0f offset_ns %.0f drift_ns %.0f\n", gpuNs, cpuNs, offset, state.calibrated ? offset - state.offsetNs : 0.0);
    state.offsetNs = offset;
    state.calibrated = true;
}

// Under the state mutex: whether lines go to the file now (opened at the window's first event,
// closed one second after its end, so batches finishing just after the window still land).
bool Writable(State& state, Clock::time_point now) {
    if (state.closed) return false;
    const auto [first, last] = Window();
    const auto at = now.time_since_epoch().count();
    if (state.file == nullptr) {
        if (at < first || at >= last) return false;
        state.file = std::fopen("cpu-timeline.txt", "w");
        if (state.file == nullptr) {
            state.closed = true;
            return false;
        }
        fileOpen.store(true, std::memory_order_release);
        std::fprintf(state.file, "# cpu-timeline: <begin ns> <end ns> <event> <thread> <tid> <a> <b> [text]\n");
        std::fprintf(state.file, "# events: submit (a serial, b 1 when the GPU had nothing queued, text reason), batch (GPU stamps; a serial, b opening thread tag, text reason/starved), wait-fence|wait-timeline|wait-hook (a target serial, b newest submitted), hold (GPU mutex >= 1 ms; a site), eop (a queue, b us since recorded), flip, dequeue (a queue, b submission serial), commit (>= 1 ms; a sequence), draw-slow (>= 1 ms in Graphics::Draw with APS5_PROFILE_DRAW; a us, b draw kind, text phases)\n");
        Recalibrate(state, now);
        if (!state.calibrated) std::fprintf(state.file, "# calibration none: times are steady-clock ns, not the GPU domain\n");
        return true;
    }
    if (at >= last + 1'000'000'000ll) {
        std::fclose(state.file);
        state.file = nullptr;
        state.closed = true;
        fileOpen.store(false, std::memory_order_release);
        return false;
    }
    Recalibrate(state, now);
    return true;
}

double ToFile(const State& state, Clock::time_point point) {
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count()) + (state.calibrated ? state.offsetNs : 0.0);
}

void ThreadName(char* text, std::size_t size) {
    const auto tag = GuestMemory::GpuLockThreadTag();
    if (tag == 0xffffffffu) std::snprintf(text, size, "untagged");
    else std::snprintf(text, size, "q0x%x", tag);
}

}

std::pair<std::int64_t, std::int64_t> Window() {
    static const auto window = []() -> std::pair<std::int64_t, std::int64_t> {
        const char* text = std::getenv("APS5_GPU_TIMELINE");
        if (text == nullptr) return {0, 0};
        double start = 0, seconds = 2;
        std::sscanf(text, "%lf,%lf", &start, &seconds);
        const auto now = Clock::now().time_since_epoch().count();
        return {now + static_cast<std::int64_t>(start * 1e9), now + static_cast<std::int64_t>((start + seconds) * 1e9)};
    }();
    return window;
}

bool Active() {
    const auto [first, last] = Window();
    if (last == 0) return false;
    const auto now = Clock::now().time_since_epoch().count();
    return (now >= first && now < last + 1'000'000'000ll) || (now >= last && fileOpen.load(std::memory_order_acquire));
}

void SetCalibration(std::function<bool(double& gpuNs, Clock::time_point& cpu)> calibrate, const void* owner) {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    state.calibrate = std::move(calibrate);
    state.owner = owner;
    // A new source is taken at the next event.
    state.calibratedAt = Clock::time_point{};
}

void ClearCalibration(const void* owner) {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    if (state.owner != owner) return;
    state.calibrate = nullptr;
    state.owner = nullptr;
}

void Event(const char* event, Clock::time_point begin, Clock::time_point end, std::uint64_t a, std::uint64_t b, const char* text) {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    if (!Writable(state, Clock::now())) return;
    char thread[24];
    ThreadName(thread, sizeof(thread));
    std::fprintf(state.file, "%.0f %.0f %s %s %llu %llu %llu%s%s\n", ToFile(state, begin), ToFile(state, end), event, thread, static_cast<unsigned long long>(ThreadId()), static_cast<unsigned long long>(a), static_cast<unsigned long long>(b), text != nullptr ? " " : "", text != nullptr ? text : "");
}

void Batch(double gpuStartNs, double gpuEndNs, Clock::time_point submittedAt, std::uint64_t serial, const char* reason, bool starved, std::uint32_t queue) {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    if (!Writable(state, Clock::now())) return;
    char thread[24];
    ThreadName(thread, sizeof(thread));
    std::fprintf(state.file, "%.0f %.0f batch %s %llu %llu %llu %s/%d submit %.0f\n", gpuStartNs, gpuEndNs, thread, static_cast<unsigned long long>(ThreadId()), static_cast<unsigned long long>(serial), static_cast<unsigned long long>(queue), reason, starved ? 1 : 0, ToFile(state, submittedAt));
}

void Flush() {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    if (state.file != nullptr) std::fflush(state.file);
}

double FileTime(Clock::time_point point) {
    auto& state = Shared();
    std::lock_guard lock(state.mutex);
    return ToFile(state, point);
}

}
