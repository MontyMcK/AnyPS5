#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstddef>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <cmath>

#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceVideoOut/include/VideoOutDriver.hpp"
#include "prx/libkernel/Automation/include/Automation.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdrStatus.hpp"

static int validateOutputConfig(int handle, uint64_t mode, const VideoOutOutputOptions* options, void* reservedPtr, uint64_t reserved) {
    if (!VideoOutDriver::Get().IsOpen(handle)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    if (reservedPtr != nullptr || reserved != 0) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    if (options != nullptr) {
        for (auto v : options->internalData) {
            if (v != 0) {
                throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_OPTION");
            }
        }
    }
    return 0;
}

static void validateOpenParam(const void* param) {
    if (param == nullptr) return;
    VideoOutOpenParam openParam{};
    std::memcpy(&openParam, param, offsetof(VideoOutOpenParam, affinity));
    if (openParam.firstWord != VIDEO_OUT_OPEN_PARAM_FIRST_WORD) {
        throw std::runtime_error(std::string(__func__) + ": unsupported first word");
    }
    if (openParam.setPriority > 1 || openParam.setAffinity > 1) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    if (openParam.setPriority == 1 && (openParam.priority < VIDEO_OUT_SERVICE_THREAD_PRIORITY_HIGHEST || openParam.priority > VIDEO_OUT_SERVICE_THREAD_PRIORITY_LOWEST)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    if (openParam.setAffinity == 1) {
        std::memcpy(&openParam.affinity, static_cast<const std::byte*>(param) + offsetof(VideoOutOpenParam, affinity), sizeof(openParam.affinity));
        if (openParam.affinity == 0 || (openParam.affinity & ~VIDEO_OUT_SERVICE_THREAD_AFFINITY_ALL) != 0) {
            throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
        }
    }
}

extern "C" {

int APS5_VABI sceVideoOutOpen(int userId, int busType, int index, const void* param) try {
    validateOpenParam(param);
    if (userId != 255 && userId != 0) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    if (busType != VIDEO_OUT_BUS_TYPE_MAIN && busType != VIDEO_OUT_BUS_TYPE_OVERLAY && busType != VIDEO_OUT_BUS_TYPE_SUB) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    if (index != 0) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    const int handle = VideoOutDriver::Get().Open(busType);
    if (handle < 0) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_RESOURCE_BUSY");
    }
    return handle;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutClose(int handle) try {
    return VideoOutDriver::Get().Close(handle) ? 0 : VIDEO_OUT_ERROR_INVALID_HANDLE;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutAllowOutputResolutionWqhdDetection(int handle) try {
    if (!VideoOutDriver::Get().IsOpen(handle)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutSetFlipRate(int handle, int rate) try {
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    if (rate < 0 || rate > 2) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    std::lock_guard lock(cfg->mutex);
    cfg->Check();
    cfg->flipRate = rate;
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status) try {
    if (status == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    *status = cfg->flipStatus;
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status) try {
    if (status == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    *status = cfg->vblankStatus;
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status) try {
    if (status == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    // APS5_VIDEO_OUT_1080P=1 (experiment): report a 1080p output whatever the buffers are.
    static const bool force1080 = std::getenv("APS5_VIDEO_OUT_1080P") != nullptr;
    status->resolution = !force1080 && (cfg->width >= 3840 || cfg->height >= 2160) ? 2u : 1u;
    // The output is HDR while the host can show HDR10 (the monitor in HDR mode, an HDR10 swapchain on
    // the window's surface), as a console's output follows its HDR display.
    const bool hdr = AgcDriverHostHdr_nid_postfix().available;
    status->dynamicRange = hdr ? VIDEO_OUT_DYNAMIC_RANGE_HDR : VIDEO_OUT_DYNAMIC_RANGE_SDR;
    status->refreshRate = (cfg->outputMode == VIDEO_OUT_OUTPUT_MODE_119_88HZ) ? VIDEO_OUT_REFRESH_RATE_119_88HZ : VIDEO_OUT_REFRESH_RATE_59_94HZ;
    status->flags = hdr ? VIDEO_OUT_OUTPUT_STATUS_FLAG_HDR : 0;
    status->reserved[0] = 0;
    status->reserved[1] = 0;
    status->reserved[2] = 0;
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutIsFlipPending(int handle) try {
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    return cfg->flipStatus.flipPendingNum;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutWaitVblank(int handle) try {
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    const uint64_t count = cfg->vblankStatus.count;
    cfg->vblankCond.wait(lock, cfg->shutdownToken, [&] { return !cfg->opened || cfg->closing || cfg->failure || cfg->vblankStatus.count != count; });
    cfg->Check();
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutInitializeOutputOptions(VideoOutOutputOptions* options) {
    if (options == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    std::memset(options, 0, sizeof(VideoOutOutputOptions));
    return 0;
}

int APS5_VABI sceVideoOutIsOutputSupported(int handle, uint64_t mode, const VideoOutOutputOptions* options, void* reservedPtr, uint64_t reserved) try {
    const int result = validateOutputConfig(handle, mode, options, reservedPtr, reserved);
    if (result != 0) {
        return result;
    }
    // APS5_VIDEO_OUT_120HZ=1 (experiment): report a 119.88 Hz capable display too, so titles offer
    // their 120 Hz modes (which may render at a lower internal resolution).
    static const bool allow120 = std::getenv("APS5_VIDEO_OUT_120HZ") != nullptr;
    const int supported = mode == VIDEO_OUT_OUTPUT_MODE_DEFAULT || (allow120 && mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ) ? 1 : 0;
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < 16) std::fprintf(stderr, "[videoout] IsOutputSupported mode 0x%llx -> %d\n", static_cast<unsigned long long>(mode), supported);
    return supported;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutConfigureOutput(int handle, uint64_t mode, const VideoOutOutputOptions* options, void* reservedPtr, uint64_t reserved) try {
    const int supported = sceVideoOutIsOutputSupported(handle, mode, options, reservedPtr, reserved);
    if (supported < 0) {
        return supported;
    }
    if (supported == 0) {
        return VIDEO_OUT_ERROR_UNAVAILABLE_OUTPUT_MODE;
    }
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    std::unique_lock lock(cfg->mutex);
    cfg->Check();
    cfg->outputMode = mode;
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutSetWindowModeMargins(int handle, int top, int bottom) try {
    (void)top;
    (void)bottom;
    if (!VideoOutDriver::Get().IsOpen(handle)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutLatencyControlWaitBeforeInput(int handle) try {
    if (!VideoOutDriver::Get().IsOpen(handle)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point) try {
    (void)point;
    if (!VideoOutDriver::Get().IsOpen(handle)) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma) {
    if (settings == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    if (!std::isfinite(gamma) || gamma < 0.1f || gamma > 2.0f) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_VALUE");
    }
    settings->gamma = gamma;
    return 0;
}

int APS5_VABI sceVideoOutAdjustColor(int handle, const VideoOutColorSettings* settings) try {
    if (settings == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_ADDRESS");
    }
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    if (cfg == nullptr) {
        throw std::runtime_error(std::string(__func__) + ": VIDEO_OUT_ERROR_INVALID_HANDLE");
    }
    // Output gamma is accepted but not yet applied to presentation.
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutVrrUnpegFromFixedRate(int handle) try {
    // The host presents through Vulkan without variable refresh, so pegging to (or releasing) a
    // fixed rate changes nothing.
    static std::atomic<int> told{0};
    if (told.fetch_add(1) < 4) std::fprintf(stderr, "[videoout] %s(handle %d): no variable refresh on the host, accepted\n", __func__, handle);
    if (Automation::Enabled()) Automation::Fields().Str("op", "unpeg").Num("handle", handle).Emit("vrr");
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

int APS5_VABI sceVideoOutVrrPegToFixedRate(int handle, uint64_t arg1, uint64_t arg2) try {
    // The host presents through Vulkan without variable refresh, so pegging to (or releasing) a
    // fixed rate changes nothing.
    static std::atomic<int> told{0};
    if (told.fetch_add(1) < 4) std::fprintf(stderr, "[videoout] %s(handle %d, 0x%llx, 0x%llx): no variable refresh on the host, accepted\n", __func__, handle, static_cast<unsigned long long>(arg1), static_cast<unsigned long long>(arg2));
    if (Automation::Enabled()) Automation::Fields().Str("op", "peg").Num("handle", handle).Hex("arg1", arg1).Hex("arg2", arg2).Emit("vrr");
    return 0;
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

APS5_EXPORT("kP2L8t3j-aM", sceVideoOutUnknown00);
int APS5_VABI sceVideoOutUnknown00() {
    return 0;
}

}
