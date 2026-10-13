#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

// Local profiling aid (not for upstream): the NVIDIA Nsight Perf SDK's in-app range profiler.
// APS5_NVPERF=1 (with a build configured with APS5_NVPERF_SDK=<sdk dir>) enables the profiler's
// Vulkan extensions, starts a report APS5_NVPERF_DELAY seconds (default 150) after the device is
// made, and names a range per render pass ("pass WxH") and per command class. The HTML and CSV
// report goes to APS5_NVPERF_DIR (default "nvperf" in the working directory).
namespace AgcDriver::Graphics::NvPerf {

bool Enabled();
void AppendInstanceExtensions(std::vector<const char*>& names, std::uint32_t apiVersion);
void AppendDeviceExtensions(VkInstance instance, VkPhysicalDevice physical, PFN_vkGetInstanceProcAddr instanceProc, std::vector<const char*>& names);
void Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device, VkQueue queue, std::uint32_t family, PFN_vkGetInstanceProcAddr instanceProc, PFN_vkGetDeviceProcAddr deviceProc);
// The end of one frame and the start of the next, under the GPU mutex after the flip's submit.
void FrameBoundary();
bool Ranging();
void PushRange(VkCommandBuffer commands, const char* name);
void PopRange(VkCommandBuffer commands);

}
