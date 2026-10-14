#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTHDRSTATUS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTHDRSTATUS_HPP

#include <cmath>
#include <cstdint>

namespace AgcDriver {

// The monitor that shows the game window, as the host OS reports it.
struct HostDisplayHdr {
    bool hdr = false;
    float maxLuminance = 0.0f;
    float minLuminance = 0.0f;
    float maxFullFrameLuminance = 0.0f;
};

// HDR output is available when the monitor runs in HDR10 and the window's Vulkan surface offers an
// HDR10 (ST 2084) swapchain in a 10-bit format. `generation` changes whenever `available` or the
// reported luminance changes; `displayGeneration` whenever the monitor's report changes.
struct HostHdrStatus {
    bool available = false;
    bool surfaceHdr10 = false;
    HostDisplayHdr display{};
    std::uint64_t generation = 0;
    std::uint64_t displayGeneration = 0;
};

// The monitor's state from its OS description: whether it scans out ST 2084 with BT.2020 primaries,
// and its luminance range in nits. A monitor that reports no usable peak luminance is not treated
// as HDR; an unusable minimum reads as 0 and an unusable full-frame peak as the peak.
inline HostDisplayHdr DisplayHdrFromOutput(bool hdr10ColorSpace, float maxLuminance, float minLuminance, float maxFullFrameLuminance) {
    if (!hdr10ColorSpace) return {};
    if (!std::isfinite(maxLuminance) || maxLuminance <= 0.0f) return {};
    HostDisplayHdr display;
    display.hdr = true;
    display.maxLuminance = maxLuminance;
    display.minLuminance = std::isfinite(minLuminance) && minLuminance >= 0.0f && minLuminance < maxLuminance ? minLuminance : 0.0f;
    display.maxFullFrameLuminance = std::isfinite(maxFullFrameLuminance) && maxFullFrameLuminance > 0.0f && maxFullFrameLuminance <= maxLuminance ? maxFullFrameLuminance : maxLuminance;
    return display;
}

}

extern "C" void AgcDriverSetHostDisplayHdr_nid_postfix(const AgcDriver::HostDisplayHdr& display);
extern "C" void AgcDriverSetSurfaceHdr10_nid_postfix(bool offered);
extern "C" AgcDriver::HostHdrStatus AgcDriverHostHdr_nid_postfix();
// Whether the presenter's swapchain is HDR10 (set by the presenter, read by tests and diagnostics).
extern "C" void AgcDriverSetPresentingHdr10_nid_postfix(bool hdr10);
extern "C" bool AgcDriverPresentingHdr10_nid_postfix();

#endif
