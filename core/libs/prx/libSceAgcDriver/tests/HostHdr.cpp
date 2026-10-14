#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostHdr.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::uint64_t Bgra8 = 0x8000000000000000ull;
constexpr std::uint64_t Rgba8 = 0x8000000022000000ull;
constexpr std::uint64_t Bgr10 = 0x8100000000000000ull;
constexpr std::uint64_t Rgb10 = 0x8100000022000000ull;
constexpr std::uint64_t Bgr10Pq = 0x8100070400000000ull;
constexpr std::uint64_t Rgb10Pq = 0x8100070422000000ull;
constexpr std::uint64_t Bgr10Linear = 0x8100000600000000ull;
constexpr std::uint64_t Bgra16FloatPq = 0xc001070700000000ull;
constexpr std::uint64_t Bgra8Pq = 0x8000070400000000ull;

bool sizeRejects(std::uint64_t pixelFormat) {
    try {
        static_cast<void>(AgcDriver::DisplayBufferSize({0x10000, pixelFormat, 1920, 1080}));
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void formatDecoding() {
    using namespace AgcDriver;
    for (const auto format : {Bgra8, Rgba8, Bgr10, Rgb10, Bgr10Pq, Rgb10Pq}) Expect(DisplayFormatSupported(format) && !sizeRejects(format), "a presented display format is refused");
    for (const auto format : {Bgr10Linear, Bgra16FloatPq, Bgra8Pq, Bgr10Pq | 0x1ull, 0x8100070000000000ull, 0x8100000400000000ull}) Expect(!DisplayFormatSupported(format) && sizeRejects(format), "an unpresented display format is accepted");
    Expect(DisplayPq(Bgr10Pq) && DisplayPq(Rgb10Pq), "the 10-bit BT.2020 ST 2084 formats are not HDR10");
    for (const auto format : {Bgra8, Rgba8, Bgr10, Rgb10, Bgr10Linear, Bgra16FloatPq, Bgra8Pq}) Expect(!DisplayPq(format), "a format without the 10-bit BT.2020 ST 2084 encoding is HDR10");
    Expect(DisplayTenBit(Bgr10Pq) && DisplayTenBit(Rgb10Pq), "the HDR10 formats are not 10-bit");
    Expect(DisplayTexelFormat(Bgr10Pq) == VK_FORMAT_A2R10G10B10_UNORM_PACK32, "the BGR HDR10 format does not hold blue in the low bits");
    Expect(DisplayTexelFormat(Rgb10Pq) == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "the RGB HDR10 format does not hold red in the low bits");
    Expect(DisplayTexelFormat(Bgra8) == VK_FORMAT_B8G8R8A8_UNORM && DisplayTexelFormat(Rgba8) == VK_FORMAT_R8G8B8A8_UNORM && DisplayTexelFormat(Bgr10) == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && DisplayTexelFormat(Rgb10) == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "an SDR display texel format changed");
}

void residentPaths() {
    using AgcDriver::ResidentPresent;
    using AgcDriver::ResidentPresentPath;
    Expect(ResidentPresentPath(VK_FORMAT_A2R10G10B10_UNORM_PACK32, Bgr10Pq, true, true) == ResidentPresent::Blit, "an HDR10 image in its display's order is not blitted to the HDR10 swapchain");
    Expect(ResidentPresentPath(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Rgb10Pq, true, true) == ResidentPresent::Blit, "an RGB HDR10 image in its display's order is not blitted to the HDR10 swapchain");
    Expect(ResidentPresentPath(VK_FORMAT_A2B10G10R10_UNORM_PACK32, Bgr10Pq, true, true) == ResidentPresent::Convert, "an HDR10 image in the other order is not copied texel for texel");
    Expect(ResidentPresentPath(VK_FORMAT_A2R10G10B10_UNORM_PACK32, Bgr10Pq, false, true) == ResidentPresent::Convert, "an HDR10 image that cannot be a blit source is not copied texel for texel");
    Expect(ResidentPresentPath(VK_FORMAT_B8G8R8A8_UNORM, Bgr10Pq, true, true) == ResidentPresent::None, "an 8-bit image presents an HDR10 display buffer");
    Expect(ResidentPresentPath(VK_FORMAT_A2R10G10B10_UNORM_PACK32, Bgr10, true, true) == ResidentPresent::None, "an SDR display buffer is passed through to the HDR10 swapchain");
    for (const auto display : {Bgr10Pq, Rgb10Pq}) {
        for (const auto storage : {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32}) Expect(ResidentPresentPath(storage, display, true) == ResidentPresent::Convert, "an HDR10 image without HDR output is not reduced to 8-bit like any 10-bit display buffer");
    }
}

void texels() {
    constexpr std::uint32_t width = 3;
    constexpr std::uint32_t height = 2;
    constexpr std::uint32_t pitch = 4;
    std::vector<std::byte> source(static_cast<std::size_t>(pitch) * height * 4);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<std::byte>(i * 7 + 3);
    const AgcDriver::DisplayBuffer buffer{0x10000, Bgr10Pq, width, height, 1, pitch};
    const auto rows = AgcDriver::DisplayBufferTexels(buffer, source);
    Expect(rows.size() == static_cast<std::size_t>(width) * height * 4, "linear texels have the wrong size");
    bool same = true;
    for (std::uint32_t y = 0; y < height; ++y) same = same && std::memcmp(rows.data() + y * width * 4, source.data() + y * pitch * 4, width * 4) == 0;
    Expect(same, "linear HDR10 texels are not copied unchanged");
    std::vector<std::byte> tiled(65536);
    for (std::size_t i = 0; i < tiled.size(); i += 4) {
        const std::uint32_t word = 0x80000000u | static_cast<std::uint32_t>(i / 4);
        std::memcpy(tiled.data() + i, &word, 4);
    }
    const AgcDriver::DisplayBuffer tiledBuffer{0x10000, Rgb10Pq, 128, 128};
    const auto detiled = AgcDriver::DisplayBufferTexels(tiledBuffer, tiled);
    std::vector<bool> seen(16384);
    bool permutation = detiled.size() == tiled.size();
    for (std::size_t i = 0; permutation && i < detiled.size(); i += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, detiled.data() + i, 4);
        const auto index = word & 0x3fffu;
        permutation = (word & 0xffffc000u) == 0x80000000u && !seen[index];
        if (permutation) seen[index] = true;
    }
    Expect(permutation, "tiled HDR10 texels are not each copied once, unchanged");
}

void luminance() {
    using AgcDriver::DisplayHdrFromOutput;
    const auto hdr = DisplayHdrFromOutput(true, 1015.0f, 0.005f, 640.0f);
    Expect(hdr.hdr && hdr.maxLuminance == 1015.0f && hdr.minLuminance == 0.005f && hdr.maxFullFrameLuminance == 640.0f, "an HDR10 monitor's luminance is not kept");
    Expect(!DisplayHdrFromOutput(false, 1015.0f, 0.005f, 640.0f).hdr, "a monitor in SDR is reported as HDR");
    Expect(!DisplayHdrFromOutput(true, 0.0f, 0.0f, 0.0f).hdr && !DisplayHdrFromOutput(true, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f).hdr, "an HDR10 monitor without a peak luminance is reported as HDR");
    const auto partial = DisplayHdrFromOutput(true, 600.0f, -1.0f, 0.0f);
    Expect(partial.hdr && partial.minLuminance == 0.0f && partial.maxFullFrameLuminance == 600.0f, "an unusable black level or full-frame peak is not replaced");
    const auto inverted = DisplayHdrFromOutput(true, 400.0f, 500.0f, 900.0f);
    Expect(inverted.hdr && inverted.minLuminance == 0.0f && inverted.maxFullFrameLuminance == 400.0f, "a black level above the peak or a full-frame peak above the peak is kept");
}

void formatSelection() {
    using AgcDriver::SelectSwapchainFormat;
    using AgcDriver::SurfaceOffersHdr10;
    const std::array<VkSurfaceFormatKHR, 2> sdrOnly{{{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, {VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}}};
    Expect(!SurfaceOffersHdr10(sdrOnly) && !SelectSwapchainFormat(sdrOnly, true, VK_FORMAT_A2B10G10R10_UNORM_PACK32), "an SDR surface offers HDR10");
    const auto sdr = SelectSwapchainFormat(sdrOnly, false, VK_FORMAT_UNDEFINED);
    Expect(sdr && sdr->format == VK_FORMAT_B8G8R8A8_UNORM && sdr->colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, "the SDR swapchain format changed");
    const std::array<VkSurfaceFormatKHR, 4> hdr{{{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}, {VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_HDR10_ST2084_EXT}, {VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT}}};
    Expect(SurfaceOffersHdr10(hdr), "a surface with a 10-bit HDR10 format does not offer HDR10");
    const auto blueLow = SelectSwapchainFormat(hdr, true, VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    Expect(blueLow && blueLow->format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && blueLow->colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT, "the only 10-bit HDR10 format is not chosen");
    const std::array<VkSurfaceFormatKHR, 3> both{{{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT}, {VK_FORMAT_A2R10G10B10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT}, {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}}};
    Expect(SelectSwapchainFormat(both, true, VK_FORMAT_A2R10G10B10_UNORM_PACK32)->format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 && SelectSwapchainFormat(both, true, VK_FORMAT_A2B10G10R10_UNORM_PACK32)->format == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "the display buffer's own 10-bit order is not preferred");
    Expect(SelectSwapchainFormat(both, true, VK_FORMAT_B8G8R8A8_UNORM)->format == VK_FORMAT_A2B10G10R10_UNORM_PACK32, "a preferred format that is not 10-bit HDR10 is chosen");
    const std::array<VkSurfaceFormatKHR, 1> floatOnly{{{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_HDR10_ST2084_EXT}}};
    Expect(!SurfaceOffersHdr10(floatOnly) && !SelectSwapchainFormat(floatOnly, false, VK_FORMAT_UNDEFINED), "a float HDR10 surface offers HDR10 or SDR");
}

void hostState() {
    const auto initial = AgcDriverHostHdr_nid_postfix();
    Expect(!initial.available && !initial.display.hdr && !initial.surfaceHdr10, "HDR output is available before the host was probed");
    AgcDriverSetSurfaceHdr10_nid_postfix(true);
    auto status = AgcDriverHostHdr_nid_postfix();
    Expect(!status.available && status.generation == initial.generation, "a surface that offers HDR10 makes HDR output available on an SDR monitor");
    AgcDriverSetHostDisplayHdr_nid_postfix(AgcDriver::DisplayHdrFromOutput(true, 800.0f, 0.01f, 400.0f));
    status = AgcDriverHostHdr_nid_postfix();
    Expect(status.available && status.generation != initial.generation && status.displayGeneration != initial.displayGeneration, "an HDR10 monitor and surface do not make HDR output available");
    Expect(status.display.maxLuminance == 800.0f && status.display.minLuminance == 0.01f && status.display.maxFullFrameLuminance == 400.0f, "the monitor's luminance is not published");
    const auto availableGeneration = status.generation;
    AgcDriverSetHostDisplayHdr_nid_postfix(AgcDriver::DisplayHdrFromOutput(true, 800.0f, 0.01f, 400.0f));
    Expect(AgcDriverHostHdr_nid_postfix().generation == availableGeneration, "an unchanged monitor report changes the generation");
    AgcDriverSetHostDisplayHdr_nid_postfix(AgcDriver::DisplayHdrFromOutput(true, 1000.0f, 0.01f, 400.0f));
    status = AgcDriverHostHdr_nid_postfix();
    Expect(status.available && status.generation != availableGeneration && status.display.maxLuminance == 1000.0f, "a new peak luminance (the window moved to another HDR monitor) is not announced");
    const auto movedGeneration = status.generation;
    AgcDriverSetSurfaceHdr10_nid_postfix(false);
    status = AgcDriverHostHdr_nid_postfix();
    Expect(!status.available && status.generation != movedGeneration, "a surface without HDR10 leaves HDR output available");
    AgcDriverSetSurfaceHdr10_nid_postfix(true);
    AgcDriverSetHostDisplayHdr_nid_postfix({});
    status = AgcDriverHostHdr_nid_postfix();
    Expect(!status.available && !status.display.hdr, "a monitor switched to SDR leaves HDR output available");
}

}

int main() {
    try {
        formatDecoding();
        residentPaths();
        texels();
        luminance();
        formatSelection();
        hostState();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::printf("host HDR tests passed\n");
    return 0;
}
