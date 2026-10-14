#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DISPLAYFORMAT_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <cstdint>

namespace AgcDriver {

// The display formats VideoOut presents: B8G8R8A8 and R8G8B8A8 (sRGB), their 10:10:10:2 variants,
// and the 10:10:10:2 BT.2020 / SMPTE ST 2084 (HDR10) variants.
bool DisplayFormatSupported(std::uint64_t pixelFormat);
VkFormat DisplayTexelFormat(std::uint64_t pixelFormat);
bool DisplayTenBit(std::uint64_t pixelFormat);
bool DisplayRedLow(std::uint64_t pixelFormat);
bool DisplayPq(std::uint64_t pixelFormat);

enum class ResidentPresent : std::uint8_t { None, Blit, Convert };
// `passthrough`: the display buffer's texels reach an HDR10 swapchain unchanged (an HDR10 display
// format presented to an HDR10 swapchain), instead of being reduced to 8-bit BGRA.
ResidentPresent ResidentPresentPath(VkFormat storage, std::uint64_t pixelFormat, bool blitSource, bool passthrough = false);

}

#endif
