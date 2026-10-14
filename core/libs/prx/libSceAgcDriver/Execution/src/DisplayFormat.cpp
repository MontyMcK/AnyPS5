#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"

namespace AgcDriver {
namespace {

// A display pixel format is a 64-bit token: bits 63-56 the texel layout (0x80 8:8:8:8, 0x81
// 10:10:10:2), bits 47-40 the primaries (0 BT.709, 7 BT.2020), bits 39-32 the transfer function
// (0 sRGB, 4 SMPTE ST 2084) and bits 31-24 the component order (0x22: red in the low bits).
constexpr std::uint64_t PixelFormatUnormBit = 0x0100000000000000ull;
constexpr std::uint64_t PixelFormatBt2020Pq = 0x0000070400000000ull;
constexpr std::uint64_t PixelFormatB8G8R8A8 = 0x8000000000000000ull;
constexpr std::uint64_t PixelFormatR8G8B8A8 = 0x8000000022000000ull;

std::uint64_t baseFormat(std::uint64_t pixelFormat) {
    const bool pq = (pixelFormat & PixelFormatUnormBit) != 0 && (pixelFormat & PixelFormatBt2020Pq) == PixelFormatBt2020Pq;
    return pixelFormat & ~(PixelFormatUnormBit | (pq ? PixelFormatBt2020Pq : 0ull));
}

}

bool DisplayFormatSupported(std::uint64_t pixelFormat) {
    const auto base = baseFormat(pixelFormat);
    return base == PixelFormatB8G8R8A8 || base == PixelFormatR8G8B8A8;
}

bool DisplayTenBit(std::uint64_t pixelFormat) {
    return (pixelFormat & PixelFormatUnormBit) != 0;
}

bool DisplayRedLow(std::uint64_t pixelFormat) {
    return baseFormat(pixelFormat) == PixelFormatR8G8B8A8;
}

bool DisplayPq(std::uint64_t pixelFormat) {
    return DisplayFormatSupported(pixelFormat) && DisplayTenBit(pixelFormat) && (pixelFormat & PixelFormatBt2020Pq) == PixelFormatBt2020Pq;
}

VkFormat DisplayTexelFormat(std::uint64_t pixelFormat) {
    const bool redLow = DisplayRedLow(pixelFormat);
    if (DisplayTenBit(pixelFormat)) return redLow ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_A2R10G10B10_UNORM_PACK32;
    return redLow ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
}

ResidentPresent ResidentPresentPath(VkFormat storage, std::uint64_t pixelFormat, bool blitSource, bool passthrough) {
    const bool tenBit = DisplayTenBit(pixelFormat);
    if (passthrough) {
        if (!DisplayPq(pixelFormat)) return ResidentPresent::None;
        if (blitSource && storage == DisplayTexelFormat(pixelFormat)) return ResidentPresent::Blit;
        return storage == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || storage == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ? ResidentPresent::Convert : ResidentPresent::None;
    }
    if (!tenBit && blitSource && storage == DisplayTexelFormat(pixelFormat)) return ResidentPresent::Blit;
    switch (storage) {
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
            return tenBit ? ResidentPresent::Convert : ResidentPresent::None;
        case VK_FORMAT_R8G8B8A8_UNORM:
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB:
            return tenBit ? ResidentPresent::None : ResidentPresent::Convert;
        default:
            return ResidentPresent::None;
    }
}

}
