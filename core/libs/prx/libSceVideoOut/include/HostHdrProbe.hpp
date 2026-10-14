#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_HOSTHDRPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_INCLUDE_HOSTHDRPROBE_HPP

#include <chrono>

// Reads whether the monitor that shows the game window runs in HDR10 and its luminance range, and
// publishes it to the driver (AgcDriverSetHostDisplayHdr_nid_postfix). On Windows this is DXGI's
// description of the output (IDXGIOutput6::GetDesc1); the monitor is read again when the window moves
// to another monitor or the display configuration changes (an HDR toggle makes the DXGI factory
// stale). Other hosts report SDR.
class HostHdrProbe {
public:
    HostHdrProbe() = default;
    ~HostHdrProbe();
    HostHdrProbe(const HostHdrProbe&) = delete;
    HostHdrProbe& operator=(const HostHdrProbe&) = delete;

    // At most every 250 ms. `nativeWindow` is the window's HWND on Windows.
    void Poll(void* nativeWindow);
    // Before the game window exists: reads the monitor a new window opens on and whether a Vulkan
    // surface there offers an HDR10 swapchain, through a hidden window of the same placement. The
    // presenter's own surface replaces the surface answer once it exists.
    void ProbeBeforeWindow();
    // Drops the DXGI factory (on the presenter thread, before the process tears down).
    void Release() noexcept;

private:
    std::chrono::steady_clock::time_point next{};
    void* library = nullptr;
    void* factory = nullptr;
    void* monitor = nullptr;
    bool failureReported = false;
};

#endif
