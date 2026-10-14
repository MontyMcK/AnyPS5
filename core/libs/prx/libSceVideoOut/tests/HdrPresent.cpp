#include "prx/libSceVideoOut/include/VideoOutDriver.hpp"
#include "prx/libSceVideoOut/include/Buffer.hpp"
#include "prx/libSceVideoOut/include/Output.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdrStatus.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <span>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#undef main

// Presents 10-bit BT.2020 ST 2084 (HDR10) display buffers through VideoOut. With HDR output on the
// host (Windows HDR on for the monitor that shows the window, an HDR10 swapchain on its surface) they
// must reach an HDR10 swapchain, and an SDR buffer must bring the SDR swapchain back. Without HDR
// output the HDR10 buffers must still be presented (through the SDR swapchain) and the test reports
// itself skipped.

namespace {

constexpr int SkipReturnCode = 77;
constexpr std::uint64_t Bgr10Pq = 0x8100070400000000ull;
constexpr std::uint64_t Rgb10Pq = 0x8100070422000000ull;
constexpr std::uint64_t Bgra8 = 0x8000000000000000ull;
constexpr std::uint32_t Width = 259;
constexpr std::uint32_t Height = 137;

void check(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::span<std::byte> alignedBuffer(std::vector<std::byte>& allocation, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(allocation.data());
    const auto aligned = (address + 65535u) & ~static_cast<std::uintptr_t>(65535u);
    return {allocation.data() + (aligned - address), bytes};
}

// A PQ ramp: 10-bit code values from black to the 10 000 nit code across the buffer.
void fillPq(std::span<std::byte> storage) {
    for (std::size_t i = 0; i + 4 <= storage.size(); i += 4) {
        const auto code = static_cast<std::uint32_t>((i / 4) % 1024u);
        const std::uint32_t word = code | (code << 10u) | ((1023u - code) << 20u) | (3u << 30u);
        std::memcpy(storage.data() + i, &word, 4);
    }
}

void flip(int handle, const std::shared_ptr<VideoOutConfig>& cfg, int index) {
    std::uint64_t target = 0;
    {
        std::lock_guard lock(cfg->mutex);
        target = cfg->flipStatus.count + 1;
    }
    check(sceVideoOutSubmitFlip(handle, index, 1, index) == 0, "flip submission failed");
    std::unique_lock lock(cfg->mutex);
    check(cfg->vblankCond.wait_for(lock, std::chrono::seconds(30), [&] { return cfg->failure != nullptr || cfg->flipStatus.count == target; }), "presentation did not complete");
    if (cfg->failure) std::rethrow_exception(cfg->failure);
    check(cfg->flipStatus.currentBuffer == index && cfg->flipStatus.flipPendingNum == 0, "presentation status is wrong");
}

// The window title comes from the title's param.json.
std::filesystem::path fixtureDirectory;

void writeFixture() {
    const auto directory = std::filesystem::temp_directory_path() / ("anyps5-hdr-present-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory / "app0" / "sce_sys");
    std::filesystem::current_path(directory);
    fixtureDirectory = directory;
    std::ofstream param("app0/sce_sys/param.json", std::ios::binary);
    param << R"({"titleId":"PPSA00000","localizedParameters":{"en-US":{"titleName":"HDR present test"}},"downloadDataSize":0})";
    check(static_cast<bool>(param), "the param.json fixture could not be written");
}

// A host without a display or a Vulkan loader cannot present at all.
bool canPresent() {
#ifdef _WIN32
    if (GetSystemMetrics(SM_CMONITORS) == 0) return false;
    const HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
    if (vulkan == nullptr) return false;
    FreeLibrary(vulkan);
    return true;
#else
    return false;
#endif
}

int run() {
    if (!canPresent()) {
        std::printf("no display or Vulkan loader: HDR presentation not exercised\n");
        return SkipReturnCode;
    }
    writeFixture();
    const int handle = sceVideoOutOpen(255, 0, 0, nullptr);
    check(handle > 0, "sceVideoOutOpen failed");
    VideoOutOutputStatus early{};
    check(sceVideoOutGetOutputStatus(handle, &early) == 0 && early.dynamicRange == (AgcDriverHostHdr_nid_postfix().available ? 2u : 1u), "the output status before the first flip does not follow the host's HDR output");
    auto cfg = VideoOutDriver::Get().GetConfig(handle);
    const std::size_t tiledBytes = static_cast<std::size_t>((Width + 127u) / 128u) * ((Height + 127u) / 128u) * 65536u;
    const std::size_t linearBytes = static_cast<std::size_t>(Width) * Height * 4u;
    std::vector<std::byte> pqAllocation(tiledBytes + 65535u);
    std::vector<std::byte> linearAllocation(linearBytes + 65535u);
    std::vector<std::byte> sdrAllocation(tiledBytes + 65535u);
    const auto pq = alignedBuffer(pqAllocation, tiledBytes);
    const auto linear = alignedBuffer(linearAllocation, linearBytes);
    const auto sdr = alignedBuffer(sdrAllocation, tiledBytes);
    fillPq(pq);
    fillPq(linear);
    std::memset(sdr.data(), 0x40, sdr.size());

    VideoOutBufferAttribute2 attribute{};
    VideoOutBuffers buffer{pq.data(), nullptr, {nullptr, nullptr}};
    sceVideoOutSetBufferAttribute2(&attribute, Bgr10Pq, 0, Width, Height, 0, 0, 0);
    check(sceVideoOutRegisterBuffers2(handle, 0, 0, &buffer, 1, &attribute, 0, nullptr) == 0, "the tiled HDR10 buffer was not registered");
    buffer = {linear.data(), nullptr, {nullptr, nullptr}};
    sceVideoOutSetBufferAttribute2(&attribute, Rgb10Pq, 1, Width, Height, 0, 0, 0);
    check(sceVideoOutRegisterBuffers2(handle, 1, 1, &buffer, 1, &attribute, 0, nullptr) == 0, "the linear HDR10 buffer was not registered");
    buffer = {sdr.data(), nullptr, {nullptr, nullptr}};
    sceVideoOutSetBufferAttribute2(&attribute, Bgra8, 0, Width, Height, 0, 0, 0);
    check(sceVideoOutRegisterBuffers2(handle, 2, 2, &buffer, 1, &attribute, 0, nullptr) == 0, "the SDR buffer was not registered");

    flip(handle, cfg, 0);
    const auto host = AgcDriverHostHdr_nid_postfix();
    VideoOutOutputStatus status{};
    check(sceVideoOutGetOutputStatus(handle, &status) == 0, "sceVideoOutGetOutputStatus failed");
    check(status.dynamicRange == (host.available ? 2u : 1u), "the output status does not follow the host's HDR output");
    int result = 0;
    if (!host.available) {
        check(!AgcDriverPresentingHdr10_nid_postfix(), "an HDR10 swapchain was made without HDR output");
        flip(handle, cfg, 1);
        flip(handle, cfg, 2);
        check(!AgcDriverPresentingHdr10_nid_postfix(), "an HDR10 swapchain was made without HDR output");
        std::printf("HDR output is unavailable on this host (monitor in HDR10: %s, surface offers HDR10: %s): the HDR10 buffers were presented through the SDR swapchain; HDR passthrough not exercised\n", host.display.hdr ? "yes" : "no", host.surfaceHdr10 ? "yes" : "no");
        result = SkipReturnCode;
    } else {
        check(AgcDriverPresentingHdr10_nid_postfix(), "the tiled HDR10 buffer did not reach an HDR10 swapchain");
        flip(handle, cfg, 1);
        check(AgcDriverPresentingHdr10_nid_postfix(), "the linear HDR10 buffer did not reach an HDR10 swapchain");
        flip(handle, cfg, -1);
        check(AgcDriverPresentingHdr10_nid_postfix(), "a blank flip changed the swapchain");
        flip(handle, cfg, 2);
        check(!AgcDriverPresentingHdr10_nid_postfix(), "the SDR buffer did not bring the SDR swapchain back");
        flip(handle, cfg, 0);
        check(AgcDriverPresentingHdr10_nid_postfix(), "the HDR10 swapchain did not come back for an HDR10 buffer");
        std::printf("HDR10 presentation passed (peak %.1f nits)\n", host.display.maxLuminance);
    }
    sceVideoOutUnregisterBuffers(handle, 0);
    sceVideoOutUnregisterBuffers(handle, 1);
    sceVideoOutUnregisterBuffers(handle, 2);
    sceVideoOutClose(handle);
    LibcRunShutdown_nid_postfix();
    std::error_code ignored;
    std::filesystem::current_path(fixtureDirectory.parent_path(), ignored);
    std::filesystem::remove_all(fixtureDirectory, ignored);
    return result;
}

}

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        try { LibcRunShutdown_nid_postfix(); }
        catch (const std::exception& shutdown) { std::fprintf(stderr, "shutdown: %s\n", shutdown.what()); }
        return 1;
    }
}
