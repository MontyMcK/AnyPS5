#ifndef AGCDRIVER_GRAPHICS_DEFERREDVULKAN_HPP
#define AGCDRIVER_GRAPHICS_DEFERREDVULKAN_HPP

#include <vulkan/vulkan.h>

namespace AgcDriver::Graphics {

// APS5_RECORDER_THREAD=1: command recording is handed to a recorder thread. A thread holding the GPU
// mutex that records (vkCmd*, command buffer begin/end/reset, vkUpdateDescriptorSets, vkQueueSubmit)
// only copies the call into an ordered stream; the recorder thread makes the driver calls. Calls that
// observe or retire what the stream does (waits, fence resets, frees, present) first wait for the
// stream to empty. Status queries do not wait: they can only answer "not yet" early.
//
// Returns the entry point the device resolver should hand out for `name`: a thunk when the stream is
// enabled and the call takes part in it, `real` otherwise.
PFN_vkVoidFunction DeferredVulkanEntry(const char* name, PFN_vkVoidFunction real);
// Waits until the recorder thread made every call streamed so far (no-op when disabled).
void DrainDeferredVulkan();
bool DeferredVulkanEnabled();

}

#endif
