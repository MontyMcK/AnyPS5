#ifndef AGCDRIVER_GRAPHICS_DEFERREDVULKAN_HPP
#define AGCDRIVER_GRAPHICS_DEFERREDVULKAN_HPP

#include <vulkan/vulkan.h>
#include <cstdint>

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
// The stream position of the newest streamed call (0 when disabled). Read by the thread that just
// streamed a call (producers hold the GPU mutex, one at a time), it is that call's ticket:
// DeferredVulkanExecuted(ticket) then says the recorder thread made the call and returned from it
// (Recorder::FenceNoDrain: a batch's vkQueueSubmit no longer touches its fence or command buffer).
std::uint64_t DeferredVulkanTicket();
bool DeferredVulkanExecuted(std::uint64_t ticket);
bool DeferredVulkanEnabled();

}

#endif
