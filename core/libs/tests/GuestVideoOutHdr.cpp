#include "SceTypes.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdrStatus.hpp"
#include "prx/libSceSystemService/SystemService.hpp"
#include <chrono>
#include <cstdint>
#include <thread>
#include <cstdio>
#include <cstdlib>

extern "C" {
int APS5_VABI sceVideoOutOpen(int userId, int busType, int index, const void* param);
int APS5_VABI sceVideoOutClose(int handle);
int APS5_VABI sceVideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status);
int APS5_VABI sceVideoOutAddOutputModeEvent(KernelEqueue eq, int handle, void* udata);
int APS5_VABI sceVideoOutGetEventId(const KernelEvent* ev);
int APS5_VABI sceVideoOutGetEventData(const KernelEvent* ev, int64_t* data);
int APS5_VABI sceKernelCreateEqueue(KernelEqueue* eq, const char* name);
int APS5_VABI sceKernelDeleteEqueue(KernelEqueue eq);
int APS5_VABI sceKernelWaitEqueue(KernelEqueue eq, KernelEvent* ev, int num, int* out, const KernelUseconds* timo);
int APS5_VABI sceSystemServiceGetHdrToneMapLuminance(SystemServiceHdrToneMapLuminance* luminance);
}

namespace {

constexpr int SystemUser = 255;
constexpr int MainBus = 0;
constexpr int OutputModeEvent = 8;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    std::abort();
}

VideoOutOutputStatus Status(int handle) {
    VideoOutOutputStatus status{};
    Require(sceVideoOutGetOutputStatus(handle, &status) == 0, "sceVideoOutGetOutputStatus failed");
    return status;
}

SystemServiceHdrToneMapLuminance ToneMap() {
    SystemServiceHdrToneMapLuminance luminance{-1.0f, -1.0f, -1.0f};
    Require(sceSystemServiceGetHdrToneMapLuminance(&luminance) == SYSTEM_SERVICE_OK, "sceSystemServiceGetHdrToneMapLuminance failed");
    return luminance;
}

bool OutputModeEventArrives(KernelEqueue queue) {
    KernelEvent event{};
    int count = 0;
    KernelUseconds timeout = 2'000'000;
    if (sceKernelWaitEqueue(queue, &event, 1, &count, &timeout) != 0 || count != 1) return false;
    int64_t mode = -1;
    return sceVideoOutGetEventId(&event) == OutputModeEvent && sceVideoOutGetEventData(&event, &mode) == 0 && mode == 1;
}

void DrainInitialEvent(KernelEqueue queue) {
    KernelEvent event{};
    int count = 0;
    KernelUseconds timeout = 0;
    static_cast<void>(sceKernelWaitEqueue(queue, &event, 1, &count, &timeout));
}

}

int main() {
    const int handle = sceVideoOutOpen(SystemUser, MainBus, 0, nullptr);
    Require(handle > 0, "sceVideoOutOpen failed");
    // Opening VideoOut probed the real host; the rest of the test drives the host state itself,
    // starting from an SDR monitor and a surface without HDR10.
    AgcDriverSetHostDisplayHdr_nid_postfix({});
    AgcDriverSetSurfaceHdr10_nid_postfix(false);
    // The presenter announces that change within one 10 ms loop, before the subscription below.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto status = Status(handle);
    Require(status.dynamicRange == 1 && status.flags == 0, "an SDR host reports HDR output");
    auto luminance = ToneMap();
    Require(luminance.max_tone_map_luminance == 100.0f && luminance.max_full_frame_tone_map_luminance == 100.0f && luminance.min_tone_map_luminance == 0.0f, "an SDR host does not report SDR reference white");

    KernelEqueue queue = 0;
    Require(sceKernelCreateEqueue(&queue, "HDR test") == 0, "event queue creation failed");
    Require(sceVideoOutAddOutputModeEvent(queue, handle, nullptr) == 0, "output mode subscription failed");
    DrainInitialEvent(queue);

    // An HDR10 monitor alone (no HDR10 swapchain on the window's surface) is not HDR output.
    AgcDriverSetHostDisplayHdr_nid_postfix(AgcDriver::DisplayHdrFromOutput(true, 1015.0f, 0.005f, 640.0f));
    status = Status(handle);
    Require(status.dynamicRange == 1 && status.flags == 0, "an HDR10 monitor without an HDR10 surface reports HDR output");
    Require(ToneMap().max_tone_map_luminance == 100.0f, "an HDR10 monitor without an HDR10 surface reports its luminance");

    AgcDriverSetSurfaceHdr10_nid_postfix(true);
    Require(OutputModeEventArrives(queue), "HDR output becoming available does not fire the output mode event");
    status = Status(handle);
    Require(status.dynamicRange == 2 && status.flags == 1, "HDR output is not reported");
    Require(status.resolution != 0 && status.refreshRate == 3 && status.reserved[0] == 0 && status.reserved[1] == 0 && status.reserved[2] == 0, "the rest of the output status changed with HDR");
    luminance = ToneMap();
    Require(luminance.max_tone_map_luminance == 1015.0f && luminance.max_full_frame_tone_map_luminance == 640.0f && luminance.min_tone_map_luminance == 0.005f, "the monitor's luminance is not the tone-map luminance");

    AgcDriverSetHostDisplayHdr_nid_postfix(AgcDriver::DisplayHdrFromOutput(true, 400.0f, 0.05f, 300.0f));
    Require(OutputModeEventArrives(queue), "a new monitor luminance does not fire the output mode event");
    Require(ToneMap().max_tone_map_luminance == 400.0f && Status(handle).dynamicRange == 2, "the new monitor luminance is not reported");

    AgcDriverSetHostDisplayHdr_nid_postfix({});
    Require(OutputModeEventArrives(queue), "HDR output going away does not fire the output mode event");
    status = Status(handle);
    Require(status.dynamicRange == 1 && status.flags == 0, "HDR output is reported after the monitor left HDR mode");
    Require(ToneMap().max_tone_map_luminance == 100.0f, "SDR reference white is not reported after the monitor left HDR mode");

    Require(sceVideoOutClose(handle) == 0, "sceVideoOutClose failed");
    Require(sceKernelDeleteEqueue(queue) == 0, "event queue deletion failed");
    std::printf("guest VideoOut HDR tests passed\n");
    LibcRunShutdown_nid_postfix();
}
