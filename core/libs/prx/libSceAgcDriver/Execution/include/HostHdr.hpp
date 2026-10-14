#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTHDR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTHDR_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <algorithm>
#include <optional>
#include <span>
#include "prx/libSceAgcDriver/Execution/include/HostHdrStatus.hpp"

namespace AgcDriver {

inline bool Hdr10SwapchainFormat(VkFormat format) {
    return format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}
// The swapchain format for SDR (BGRA8 sRGB-nonlinear) or HDR10 (a 10-bit format in
// VK_COLOR_SPACE_HDR10_ST2084_EXT, `preferred` first when it is offered) output, if offered.
inline std::optional<VkSurfaceFormatKHR> SelectSwapchainFormat(std::span<const VkSurfaceFormatKHR> formats, bool hdr, VkFormat preferred) {
    const auto offered = [&](VkFormat format, VkColorSpaceKHR colorSpace) {
        return std::any_of(formats.begin(), formats.end(), [&](const VkSurfaceFormatKHR& item) { return item.format == format && item.colorSpace == colorSpace; });
    };
    if (!hdr) {
        if (offered(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) return VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        return std::nullopt;
    }
    if (Hdr10SwapchainFormat(preferred) && offered(preferred, VK_COLOR_SPACE_HDR10_ST2084_EXT)) return VkSurfaceFormatKHR{preferred, VK_COLOR_SPACE_HDR10_ST2084_EXT};
    for (const auto format : {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2R10G10B10_UNORM_PACK32}) {
        if (offered(format, VK_COLOR_SPACE_HDR10_ST2084_EXT)) return VkSurfaceFormatKHR{format, VK_COLOR_SPACE_HDR10_ST2084_EXT};
    }
    return std::nullopt;
}

inline bool SurfaceOffersHdr10(std::span<const VkSurfaceFormatKHR> formats) {
    return SelectSwapchainFormat(formats, true, VK_FORMAT_UNDEFINED).has_value();
}

}

#endif
