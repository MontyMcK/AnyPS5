#include "prx/libSceAgcDriver/Execution/include/HostHdr.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mutex>

namespace AgcDriver {
namespace {

struct HostHdrState {
    std::mutex mutex;
    HostHdrStatus status;
};

HostHdrState& State() {
    static HostHdrState state;
    return state;
}

bool SameDisplay(const HostDisplayHdr& a, const HostDisplayHdr& b) {
    return a.hdr == b.hdr && a.maxLuminance == b.maxLuminance && a.minLuminance == b.minLuminance && a.maxFullFrameLuminance == b.maxFullFrameLuminance;
}

void Publish(HostHdrStatus& status, const HostDisplayHdr& display, bool surfaceHdr10) {
    const bool displayChanged = !SameDisplay(status.display, display);
    const bool available = display.hdr && surfaceHdr10;
    const bool availabilityChanged = available != status.available;
    status.display = display;
    status.surfaceHdr10 = surfaceHdr10;
    status.available = available;
    if (displayChanged) ++status.displayGeneration;
    if (availabilityChanged || (available && displayChanged)) {
        ++status.generation;
        if (available) std::fprintf(stderr, "[hdr] HDR output available: the monitor runs HDR10 (peak %.1f nits, full-frame %.1f nits, black %.4f nits) and the surface offers an HDR10 swapchain\n", display.maxLuminance, display.maxFullFrameLuminance, display.minLuminance);
        else std::fprintf(stderr, "[hdr] HDR output unavailable (monitor in HDR10: %s, surface offers HDR10: %s)\n", display.hdr ? "yes" : "no", surfaceHdr10 ? "yes" : "no");
    }
}

}

}

extern "C" void AgcDriverSetHostDisplayHdr_nid_postfix(const AgcDriver::HostDisplayHdr& display) {
    auto& state = AgcDriver::State();
    std::lock_guard lock(state.mutex);
    if (!AgcDriver::SameDisplay(state.status.display, display)) {
        if (display.hdr) std::fprintf(stderr, "[hdr] the monitor showing the window runs HDR10 (ST 2084, BT.2020): peak %.1f nits, full-frame %.1f nits, black %.4f nits\n", display.maxLuminance, display.maxFullFrameLuminance, display.minLuminance);
        else std::fprintf(stderr, "[hdr] the monitor showing the window runs SDR\n");
    }
    AgcDriver::Publish(state.status, display, state.status.surfaceHdr10);
}

extern "C" void AgcDriverSetSurfaceHdr10_nid_postfix(bool offered) {
    auto& state = AgcDriver::State();
    std::lock_guard lock(state.mutex);
    AgcDriver::Publish(state.status, state.status.display, offered);
}

namespace {
std::atomic<bool> presentingHdr10{false};
}

extern "C" void AgcDriverSetPresentingHdr10_nid_postfix(bool hdr10) {
    presentingHdr10.store(hdr10, std::memory_order_release);
}

extern "C" bool AgcDriverPresentingHdr10_nid_postfix() {
    return presentingHdr10.load(std::memory_order_acquire);
}

extern "C" AgcDriver::HostHdrStatus AgcDriverHostHdr_nid_postfix() {
    auto& state = AgcDriver::State();
    std::lock_guard lock(state.mutex);
    return state.status;
}
