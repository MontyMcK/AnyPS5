// The recorder thread (APS5_RECORDER_THREAD=1, see DeferredVulkan.hpp). The committer spends about a
// fifth of its time inside the Vulkan driver recording; streamed, that time moves to this thread.
//
// Producers are the threads holding the GPU mutex, one at a time, so the stream's order is the order
// the original calls would have been made in. Each call's pointer arguments are copied when it is
// streamed (the caller's structures are on its stack). A deferred call that returns a VkResult reports
// success; the recorder thread checks the real result and aborts on a failure, which the original
// call would have thrown on.
#include "prx/libSceAgcDriver/Graphics/include/DeferredVulkan.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <windows.h>
#include <immintrin.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

struct Item {
    std::atomic<Item*> next{nullptr};
    void (*run)(Item*) = nullptr;
};

template<typename F>
struct ItemOf final : Item {
    explicit ItemOf(F&& f) : function(std::move(f)) {
        run = [](Item* item) {
            auto* self = static_cast<ItemOf*>(item);
            self->function();
            delete self;
        };
    }
    F function;
};

// Vyukov's intrusive multi-producer single-consumer queue.
struct Stream {
    Item stub;
    std::atomic<Item*> head{&stub};
    Item* tail = &stub;
    std::atomic<std::uint64_t> pushed{0};
    std::atomic<std::uint64_t> executed{0};
    std::atomic<int> sleeping{0};
    SRWLOCK lock = SRWLOCK_INIT;
    CONDITION_VARIABLE wake = CONDITION_VARIABLE_INIT;
    DWORD recorderThread = 0;

    void push(Item* item) {
        item->next.store(nullptr, std::memory_order_relaxed);
        Item* previous = head.exchange(item, std::memory_order_acq_rel);
        previous->next.store(item, std::memory_order_release);
        pushed.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping.load(std::memory_order_seq_cst) != 0) {
            AcquireSRWLockExclusive(&lock);
            WakeConditionVariable(&wake);
            ReleaseSRWLockExclusive(&lock);
        }
    }

    // Consumer only. Null when nothing is ready (or a push is half way).
    Item* pop() {
        Item* current = tail;
        Item* next = current->next.load(std::memory_order_acquire);
        if (current == &stub) {
            if (next == nullptr) return nullptr;
            tail = next;
            current = next;
            next = next->next.load(std::memory_order_acquire);
        }
        if (next != nullptr) {
            tail = next;
            return current;
        }
        if (current != head.load(std::memory_order_acquire)) return nullptr;
        // Re-insert the stub behind the last item so the last item can be taken.
        stub.next.store(nullptr, std::memory_order_relaxed);
        Item* previous = head.exchange(&stub, std::memory_order_acq_rel);
        previous->next.store(&stub, std::memory_order_release);
        next = current->next.load(std::memory_order_acquire);
        if (next != nullptr) {
            tail = next;
            return current;
        }
        return nullptr;
    }
};

Stream& TheStream() {
    static Stream stream;
    return stream;
}

DWORD WINAPI RecorderMain(void*) {
    auto& stream = TheStream();
    SetThreadDescription(GetCurrentThread(), L"AgcRecorder");
    for (;;) {
        Item* item = stream.pop();
        if (item != nullptr) {
            item->run(item);
            stream.executed.fetch_add(1, std::memory_order_release);
            continue;
        }
        // Spin briefly (the committer streams a call every few microseconds), then sleep.
        bool found = false;
        for (int spin = 0; spin < 4000 && !found; ++spin) {
            _mm_pause();
            found = stream.executed.load(std::memory_order_relaxed) != stream.pushed.load(std::memory_order_acquire);
        }
        if (found) continue;
        AcquireSRWLockExclusive(&stream.lock);
        stream.sleeping.store(1, std::memory_order_seq_cst);
        if (stream.executed.load(std::memory_order_relaxed) == stream.pushed.load(std::memory_order_seq_cst)) SleepConditionVariableSRW(&stream.wake, &stream.lock, 1, 0);
        stream.sleeping.store(0, std::memory_order_seq_cst);
        ReleaseSRWLockExclusive(&stream.lock);
    }
}

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("APS5_RECORDER_THREAD");
        if (value == nullptr || value[0] != '1') return false;
        DWORD id = 0;
        if (HANDLE thread = CreateThread(nullptr, 0, RecorderMain, nullptr, 0, &id)) {
            CloseHandle(thread);
            TheStream().recorderThread = id;
            std::fprintf(stderr, "[recorder] command recording runs on a recorder thread (APS5_RECORDER_THREAD=1)\n");
            return true;
        }
        return false;
    }();
    return enabled;
}

bool OnRecorder() {
    return GetCurrentThreadId() == TheStream().recorderThread;
}

// Whether this call goes into the stream: the caller holds the GPU mutex (every recording does).
bool Streaming() {
    return Enabled() && !OnRecorder() && GuestMemory::GpuMutex().HeldByThisThread();
}

template<typename F>
void Defer(F&& function) {
    TheStream().push(new ItemOf<std::decay_t<F>>(std::forward<F>(function)));
}

void Drain() {
    if (!Enabled() || OnRecorder()) return;
    auto& stream = TheStream();
    const auto target = stream.pushed.load(std::memory_order_acquire);
    for (int spin = 0; stream.executed.load(std::memory_order_acquire) < target; ++spin) {
        if (spin < 2000) _mm_pause();
        else SwitchToThread();
    }
}

void CheckDeferred(VkResult result, const char* call) {
    if (result == VK_SUCCESS) return;
    std::fprintf(stderr, "[recorder] FATAL: deferred %s returned Vulkan result %d\n", call, static_cast<int>(result));
    std::fflush(stderr);
    // The device fault report (APS5_DEVICE_FAULT) for a loss found on the recorder thread.
    if (result == VK_ERROR_DEVICE_LOST && DeviceLostReporter() != nullptr) DeviceLostReporter()();
    std::fflush(stderr);
    std::abort();
}

template<typename T>
std::vector<T> Copy(const T* values, std::uint32_t count) {
    return values != nullptr && count != 0 ? std::vector<T>(values, values + count) : std::vector<T>{};
}

// Real entry points, recorded when the thunks are handed out.
struct Real {
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier;
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage;
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer;
    PFN_vkCmdClearColorImage cmdClearColorImage;
    PFN_vkCmdClearDepthStencilImage cmdClearDepthStencilImage;
    PFN_vkCmdCopyBuffer cmdCopyBuffer;
    PFN_vkCmdCopyImage cmdCopyImage;
    PFN_vkCmdBlitImage cmdBlitImage;
    PFN_vkCmdWriteTimestamp cmdWriteTimestamp;
    PFN_vkCmdPushConstants cmdPushConstants;
    PFN_vkCmdDispatch cmdDispatch;
    PFN_vkCmdDispatchIndirect cmdDispatchIndirect;
    PFN_vkCmdBindPipeline cmdBindPipeline;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer;
    PFN_vkCmdFillBuffer cmdFillBuffer;
    PFN_vkCmdResetQueryPool cmdResetQueryPool;
    PFN_vkCmdBeginQuery cmdBeginQuery;
    PFN_vkCmdEndQuery cmdEndQuery;
    PFN_vkCmdCopyQueryPoolResults cmdCopyQueryPoolResults;
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass;
    PFN_vkCmdEndRenderPass cmdEndRenderPass;
    PFN_vkCmdDraw cmdDraw;
    PFN_vkCmdDrawIndexed cmdDrawIndexed;
    PFN_vkCmdDrawIndirect cmdDrawIndirect;
    PFN_vkCmdDrawIndexedIndirect cmdDrawIndexedIndirect;
    PFN_vkCmdDrawIndirectCount cmdDrawIndirectCount;
    PFN_vkCmdDrawIndexedIndirectCount cmdDrawIndexedIndirectCount;
    PFN_vkCmdDrawMeshTasksEXT cmdDrawMeshTasks;
    PFN_vkCmdDrawMeshTasksIndirectEXT cmdDrawMeshTasksIndirect;
    PFN_vkCmdSetViewport cmdSetViewport;
    PFN_vkCmdSetScissor cmdSetScissor;
    PFN_vkCmdSetDepthBias cmdSetDepthBias;
    PFN_vkCmdSetDepthBounds cmdSetDepthBounds;
    PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers;
    PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer;
    PFN_vkCmdPushDescriptorSetKHR cmdPushDescriptorSet;
    PFN_vkBeginCommandBuffer beginCommandBuffer;
    PFN_vkEndCommandBuffer endCommandBuffer;
    PFN_vkResetCommandBuffer resetCommandBuffer;
    PFN_vkQueueSubmit queueSubmit;
    PFN_vkUpdateDescriptorSets updateDescriptorSets;
    PFN_vkCmdSetCheckpointNV cmdSetCheckpoint;
    PFN_vkCmdWriteBufferMarkerAMD cmdWriteBufferMarker;
} real{};

// A copied descriptor write: the arrays its type reads.
struct WriteCopy {
    VkWriteDescriptorSet write;
    std::vector<VkDescriptorImageInfo> images;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkBufferView> views;
};

std::optional<std::vector<WriteCopy>> CopyWrites(std::uint32_t count, const VkWriteDescriptorSet* writes) {
    std::vector<WriteCopy> copies(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        if (writes[i].pNext != nullptr) return std::nullopt;
        copies[i].write = writes[i];
        copies[i].images = Copy(writes[i].pImageInfo, writes[i].pImageInfo != nullptr ? writes[i].descriptorCount : 0);
        copies[i].buffers = Copy(writes[i].pBufferInfo, writes[i].pBufferInfo != nullptr ? writes[i].descriptorCount : 0);
        copies[i].views = Copy(writes[i].pTexelBufferView, writes[i].pTexelBufferView != nullptr ? writes[i].descriptorCount : 0);
    }
    return copies;
}

std::vector<VkWriteDescriptorSet> Rebuild(std::vector<WriteCopy>& copies) {
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(copies.size());
    for (auto& copy : copies) {
        auto write = copy.write;
        write.pImageInfo = copy.images.empty() ? nullptr : copy.images.data();
        write.pBufferInfo = copy.buffers.empty() ? nullptr : copy.buffers.data();
        write.pTexelBufferView = copy.views.empty() ? nullptr : copy.views.data();
        writes.push_back(write);
    }
    return writes;
}

// --- Streamed calls ---

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t memoryCount, const VkMemoryBarrier* memory, uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, uint32_t imageCount, const VkImageMemoryBarrier* images) {
    if (!Streaming()) return real.cmdPipelineBarrier(cb, src, dst, flags, memoryCount, memory, bufferCount, buffers, imageCount, images);
    Defer([=, m = Copy(memory, memoryCount), b = Copy(buffers, bufferCount), i = Copy(images, imageCount)] {
        real.cmdPipelineBarrier(cb, src, dst, flags, static_cast<uint32_t>(m.size()), m.data(), static_cast<uint32_t>(b.size()), b.data(), static_cast<uint32_t>(i.size()), i.data());
    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer buffer, VkImage image, VkImageLayout layout, uint32_t count, const VkBufferImageCopy* regions) {
    if (!Streaming()) return real.cmdCopyBufferToImage(cb, buffer, image, layout, count, regions);
    Defer([=, r = Copy(regions, count)] { real.cmdCopyBufferToImage(cb, buffer, image, layout, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer cb, VkImage image, VkImageLayout layout, VkBuffer buffer, uint32_t count, const VkBufferImageCopy* regions) {
    if (!Streaming()) return real.cmdCopyImageToBuffer(cb, image, layout, buffer, count, regions);
    Defer([=, r = Copy(regions, count)] { real.cmdCopyImageToBuffer(cb, image, layout, buffer, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearColorValue* color, uint32_t count, const VkImageSubresourceRange* ranges) {
    if (!Streaming()) return real.cmdClearColorImage(cb, image, layout, color, count, ranges);
    Defer([=, c = *color, r = Copy(ranges, count)] { real.cmdClearColorImage(cb, image, layout, &c, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, uint32_t count, const VkImageSubresourceRange* ranges) {
    if (!Streaming()) return real.cmdClearDepthStencilImage(cb, image, layout, value, count, ranges);
    Defer([=, v = *value, r = Copy(ranges, count)] { real.cmdClearDepthStencilImage(cb, image, layout, &v, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer source, VkBuffer destination, uint32_t count, const VkBufferCopy* regions) {
    if (!Streaming()) return real.cmdCopyBuffer(cb, source, destination, count, regions);
    Defer([=, r = Copy(regions, count)] { real.cmdCopyBuffer(cb, source, destination, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer cb, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, uint32_t count, const VkImageCopy* regions) {
    if (!Streaming()) return real.cmdCopyImage(cb, source, sourceLayout, destination, destinationLayout, count, regions);
    Defer([=, r = Copy(regions, count)] { real.cmdCopyImage(cb, source, sourceLayout, destination, destinationLayout, static_cast<uint32_t>(r.size()), r.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer cb, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, uint32_t count, const VkImageBlit* regions, VkFilter filter) {
    if (!Streaming()) return real.cmdBlitImage(cb, source, sourceLayout, destination, destinationLayout, count, regions, filter);
    Defer([=, r = Copy(regions, count)] { real.cmdBlitImage(cb, source, sourceLayout, destination, destinationLayout, static_cast<uint32_t>(r.size()), r.data(), filter); });
}

VKAPI_ATTR void VKAPI_CALL CmdWriteTimestamp(VkCommandBuffer cb, VkPipelineStageFlagBits stage, VkQueryPool pool, uint32_t query) {
    if (!Streaming()) return real.cmdWriteTimestamp(cb, stage, pool, query);
    Defer([=] { real.cmdWriteTimestamp(cb, stage, pool, query); });
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* values) {
    if (!Streaming()) return real.cmdPushConstants(cb, layout, stages, offset, size, values);
    Defer([=, v = Copy(static_cast<const std::uint8_t*>(values), size)] { real.cmdPushConstants(cb, layout, stages, offset, size, v.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    if (!Streaming()) return real.cmdDispatch(cb, x, y, z);
    Defer([=] { real.cmdDispatch(cb, x, y, z); });
}

VKAPI_ATTR void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset) {
    if (!Streaming()) return real.cmdDispatchIndirect(cb, buffer, offset);
    Defer([=] { real.cmdDispatchIndirect(cb, buffer, offset); });
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipeline pipeline) {
    if (!Streaming()) return real.cmdBindPipeline(cb, point, pipeline);
    Defer([=] { real.cmdBindPipeline(cb, point, pipeline); });
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSetsThunk(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t first, uint32_t count, const VkDescriptorSet* sets, uint32_t dynamicCount, const uint32_t* dynamicOffsets) {
    if (!Streaming()) return real.cmdBindDescriptorSets(cb, point, layout, first, count, sets, dynamicCount, dynamicOffsets);
    Defer([=, s = Copy(sets, count), d = Copy(dynamicOffsets, dynamicCount)] { real.cmdBindDescriptorSets(cb, point, layout, first, static_cast<uint32_t>(s.size()), s.data(), static_cast<uint32_t>(d.size()), d.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
    if (!Streaming()) return real.cmdUpdateBuffer(cb, buffer, offset, size, data);
    Defer([=, v = Copy(static_cast<const std::uint8_t*>(data), static_cast<std::uint32_t>(size))] { real.cmdUpdateBuffer(cb, buffer, offset, size, v.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t value) {
    if (!Streaming()) return real.cmdFillBuffer(cb, buffer, offset, size, value);
    Defer([=] { real.cmdFillBuffer(cb, buffer, offset, size, value); });
}

VKAPI_ATTR void VKAPI_CALL CmdResetQueryPool(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count) {
    if (!Streaming()) return real.cmdResetQueryPool(cb, pool, first, count);
    Defer([=] { real.cmdResetQueryPool(cb, pool, first, count); });
}

VKAPI_ATTR void VKAPI_CALL CmdBeginQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query, VkQueryControlFlags flags) {
    if (!Streaming()) return real.cmdBeginQuery(cb, pool, query, flags);
    Defer([=] { real.cmdBeginQuery(cb, pool, query, flags); });
}

VKAPI_ATTR void VKAPI_CALL CmdEndQuery(VkCommandBuffer cb, VkQueryPool pool, uint32_t query) {
    if (!Streaming()) return real.cmdEndQuery(cb, pool, query);
    Defer([=] { real.cmdEndQuery(cb, pool, query); });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyQueryPoolResults(VkCommandBuffer cb, VkQueryPool pool, uint32_t first, uint32_t count, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize stride, VkQueryResultFlags flags) {
    if (!Streaming()) return real.cmdCopyQueryPoolResults(cb, pool, first, count, buffer, offset, stride, flags);
    Defer([=] { real.cmdCopyQueryPoolResults(cb, pool, first, count, buffer, offset, stride, flags); });
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* begin, VkSubpassContents contents) {
    if (!Streaming() || begin->pNext != nullptr) {
        if (Streaming()) Drain();
        return real.cmdBeginRenderPass(cb, begin, contents);
    }
    Defer([=, info = *begin, clears = Copy(begin->pClearValues, begin->clearValueCount)]() mutable {
        info.pClearValues = clears.empty() ? nullptr : clears.data();
        real.cmdBeginRenderPass(cb, &info, contents);
    });
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer cb) {
    if (!Streaming()) return real.cmdEndRenderPass(cb);
    Defer([=] { real.cmdEndRenderPass(cb); });
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer cb, uint32_t vertices, uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) {
    if (!Streaming()) return real.cmdDraw(cb, vertices, instances, firstVertex, firstInstance);
    Defer([=] { real.cmdDraw(cb, vertices, instances, firstVertex, firstInstance); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(VkCommandBuffer cb, uint32_t indices, uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
    if (!Streaming()) return real.cmdDrawIndexed(cb, indices, instances, firstIndex, vertexOffset, firstInstance);
    Defer([=] { real.cmdDrawIndexed(cb, indices, instances, firstIndex, vertexOffset, firstInstance); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride) {
    if (!Streaming()) return real.cmdDrawIndirect(cb, buffer, offset, count, stride);
    Defer([=] { real.cmdDrawIndirect(cb, buffer, offset, count, stride); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride) {
    if (!Streaming()) return real.cmdDrawIndexedIndirect(cb, buffer, offset, count, stride);
    Defer([=] { real.cmdDrawIndexedIndirect(cb, buffer, offset, count, stride); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
    if (!Streaming()) return real.cmdDrawIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    Defer([=] { real.cmdDrawIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexedIndirectCount(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
    if (!Streaming()) return real.cmdDrawIndexedIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    Defer([=] { real.cmdDrawIndexedIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasks(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    if (!Streaming()) return real.cmdDrawMeshTasks(cb, x, y, z);
    Defer([=] { real.cmdDrawMeshTasks(cb, x, y, z); });
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasksIndirect(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride) {
    if (!Streaming()) return real.cmdDrawMeshTasksIndirect(cb, buffer, offset, count, stride);
    Defer([=] { real.cmdDrawMeshTasksIndirect(cb, buffer, offset, count, stride); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkViewport* viewports) {
    if (!Streaming()) return real.cmdSetViewport(cb, first, count, viewports);
    Defer([=, v = Copy(viewports, count)] { real.cmdSetViewport(cb, first, static_cast<uint32_t>(v.size()), v.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkRect2D* scissors) {
    if (!Streaming()) return real.cmdSetScissor(cb, first, count, scissors);
    Defer([=, s = Copy(scissors, count)] { real.cmdSetScissor(cb, first, static_cast<uint32_t>(s.size()), s.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(VkCommandBuffer cb, float constant, float clamp, float slope) {
    if (!Streaming()) return real.cmdSetDepthBias(cb, constant, clamp, slope);
    Defer([=] { real.cmdSetDepthBias(cb, constant, clamp, slope); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(VkCommandBuffer cb, float minimum, float maximum) {
    if (!Streaming()) return real.cmdSetDepthBounds(cb, minimum, maximum);
    Defer([=] { real.cmdSetDepthBounds(cb, minimum, maximum); });
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (!Streaming()) return real.cmdBindVertexBuffers(cb, first, count, buffers, offsets);
    Defer([=, b = Copy(buffers, count), o = Copy(offsets, count)] { real.cmdBindVertexBuffers(cb, first, static_cast<uint32_t>(b.size()), b.data(), o.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    if (!Streaming()) return real.cmdBindIndexBuffer(cb, buffer, offset, type);
    Defer([=] { real.cmdBindIndexBuffer(cb, buffer, offset, type); });
}

// Debug breadcrumbs (APS5_GPU_CHECKPOINTS): streamed so they land between the commands they mark.
VKAPI_ATTR void VKAPI_CALL CmdSetCheckpoint(VkCommandBuffer cb, const void* marker) {
    if (!Streaming()) return real.cmdSetCheckpoint(cb, marker);
    Defer([=] { real.cmdSetCheckpoint(cb, marker); });
}

VKAPI_ATTR void VKAPI_CALL CmdWriteBufferMarker(VkCommandBuffer cb, VkPipelineStageFlagBits stage, VkBuffer buffer, VkDeviceSize offset, uint32_t marker) {
    if (!Streaming()) return real.cmdWriteBufferMarker(cb, stage, buffer, offset, marker);
    Defer([=] { real.cmdWriteBufferMarker(cb, stage, buffer, offset, marker); });
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSet(VkCommandBuffer cb, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes) {
    if (!Streaming()) return real.cmdPushDescriptorSet(cb, point, layout, set, count, writes);
    auto copies = CopyWrites(count, writes);
    if (!copies.has_value()) {
        Drain();
        return real.cmdPushDescriptorSet(cb, point, layout, set, count, writes);
    }
    Defer([=, c = std::move(*copies)]() mutable {
        const auto rebuilt = Rebuild(c);
        real.cmdPushDescriptorSet(cb, point, layout, set, static_cast<uint32_t>(rebuilt.size()), rebuilt.data());
    });
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
    if (!Streaming() || info->pNext != nullptr || info->pInheritanceInfo != nullptr) {
        if (Streaming()) Drain();
        return real.beginCommandBuffer(cb, info);
    }
    Defer([=, copy = *info] { CheckDeferred(real.beginCommandBuffer(cb, &copy), "vkBeginCommandBuffer"); });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer cb) {
    if (!Streaming()) return real.endCommandBuffer(cb);
    Defer([=] { CheckDeferred(real.endCommandBuffer(cb), "vkEndCommandBuffer"); });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags flags) {
    if (!Streaming()) {
        Drain();
        return real.resetCommandBuffer(cb, flags);
    }
    Defer([=] { CheckDeferred(real.resetCommandBuffer(cb, flags), "vkResetCommandBuffer"); });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    if (!Streaming()) {
        Drain();
        return real.queueSubmit(queue, count, submits, fence);
    }
    struct SubmitCopy {
        VkSubmitInfo info;
        std::optional<VkTimelineSemaphoreSubmitInfo> timeline;
        std::vector<std::uint64_t> waitValues, signalValues;
        std::vector<VkSemaphore> waits, signals;
        std::vector<VkPipelineStageFlags> stages;
        std::vector<VkCommandBuffer> buffers;
    };
    std::vector<SubmitCopy> copies(count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& submit = submits[i];
        auto& copy = copies[i];
        copy.info = submit;
        if (submit.pNext != nullptr) {
            const auto* chained = static_cast<const VkBaseInStructure*>(submit.pNext);
            if (chained->sType != VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO || chained->pNext != nullptr) {
                Drain();
                return real.queueSubmit(queue, count, submits, fence);
            }
            copy.timeline = *reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(chained);
            copy.waitValues = Copy(copy.timeline->pWaitSemaphoreValues, copy.timeline->waitSemaphoreValueCount);
            copy.signalValues = Copy(copy.timeline->pSignalSemaphoreValues, copy.timeline->signalSemaphoreValueCount);
        }
        copy.waits = Copy(submit.pWaitSemaphores, submit.waitSemaphoreCount);
        copy.stages = Copy(submit.pWaitDstStageMask, submit.waitSemaphoreCount);
        copy.buffers = Copy(submit.pCommandBuffers, submit.commandBufferCount);
        copy.signals = Copy(submit.pSignalSemaphores, submit.signalSemaphoreCount);
    }
    Defer([=, c = std::move(copies)]() mutable {
        std::vector<VkSubmitInfo> infos;
        infos.reserve(c.size());
        for (auto& copy : c) {
            auto info = copy.info;
            if (copy.timeline.has_value()) {
                copy.timeline->pWaitSemaphoreValues = copy.waitValues.empty() ? nullptr : copy.waitValues.data();
                copy.timeline->pSignalSemaphoreValues = copy.signalValues.empty() ? nullptr : copy.signalValues.data();
                info.pNext = &*copy.timeline;
            }
            info.pWaitSemaphores = copy.waits.empty() ? nullptr : copy.waits.data();
            info.pWaitDstStageMask = copy.stages.empty() ? nullptr : copy.stages.data();
            info.pCommandBuffers = copy.buffers.empty() ? nullptr : copy.buffers.data();
            info.pSignalSemaphores = copy.signals.empty() ? nullptr : copy.signals.data();
            infos.push_back(info);
        }
        CheckDeferred(real.queueSubmit(queue, static_cast<uint32_t>(infos.size()), infos.data(), fence), "vkQueueSubmit");
    });
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL UpdateDescriptorSets(VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* writes, uint32_t copyCount, const VkCopyDescriptorSet* copies) {
    // Off the GPU mutex (a resource build's stage A) the update is the caller's own business.
    if (!Streaming()) return real.updateDescriptorSets(device, writeCount, writes, copyCount, copies);
    auto writeCopies = CopyWrites(writeCount, writes);
    if (!writeCopies.has_value()) {
        Drain();
        return real.updateDescriptorSets(device, writeCount, writes, copyCount, copies);
    }
    Defer([=, w = std::move(*writeCopies), c = Copy(copies, copyCount)]() mutable {
        const auto rebuilt = Rebuild(w);
        real.updateDescriptorSets(device, static_cast<uint32_t>(rebuilt.size()), rebuilt.data(), static_cast<uint32_t>(c.size()), c.data());
    });
}

// --- Calls that wait for the stream first ---

template<typename T>
struct DrainSlot {
    static inline T real = nullptr;
};

template<typename Pfn, typename Signature>
struct DrainThunk;

template<typename Pfn, typename R, typename... Args>
struct DrainThunk<Pfn, R (VKAPI_PTR*)(Args...)> {
    static VKAPI_ATTR R VKAPI_CALL call(Args... args) {
        Drain();
        return DrainSlot<Pfn>::real(args...);
    }
};

template<typename Pfn>
PFN_vkVoidFunction Draining(PFN_vkVoidFunction function) {
    DrainSlot<Pfn>::real = reinterpret_cast<Pfn>(function);
    return reinterpret_cast<PFN_vkVoidFunction>(&DrainThunk<Pfn, Pfn>::call);
}

template<typename Pfn, typename Thunk>
PFN_vkVoidFunction Streamed(Pfn& slot, PFN_vkVoidFunction function, Thunk thunk) {
    slot = reinterpret_cast<Pfn>(function);
    return reinterpret_cast<PFN_vkVoidFunction>(thunk);
}

}

bool DeferredVulkanEnabled() {
    return Enabled();
}

void DrainDeferredVulkan() {
    Drain();
}

PFN_vkVoidFunction DeferredVulkanEntry(const char* name, PFN_vkVoidFunction function) {
    if (function == nullptr || !Enabled()) return function;
    const std::string_view n(name);
    // Streamed.
    if (n == "vkCmdPipelineBarrier") return Streamed(real.cmdPipelineBarrier, function, &CmdPipelineBarrier);
    if (n == "vkCmdCopyBufferToImage") return Streamed(real.cmdCopyBufferToImage, function, &CmdCopyBufferToImage);
    if (n == "vkCmdCopyImageToBuffer") return Streamed(real.cmdCopyImageToBuffer, function, &CmdCopyImageToBuffer);
    if (n == "vkCmdClearColorImage") return Streamed(real.cmdClearColorImage, function, &CmdClearColorImage);
    if (n == "vkCmdClearDepthStencilImage") return Streamed(real.cmdClearDepthStencilImage, function, &CmdClearDepthStencilImage);
    if (n == "vkCmdCopyBuffer") return Streamed(real.cmdCopyBuffer, function, &CmdCopyBuffer);
    if (n == "vkCmdCopyImage") return Streamed(real.cmdCopyImage, function, &CmdCopyImage);
    if (n == "vkCmdBlitImage") return Streamed(real.cmdBlitImage, function, &CmdBlitImage);
    if (n == "vkCmdWriteTimestamp") return Streamed(real.cmdWriteTimestamp, function, &CmdWriteTimestamp);
    if (n == "vkCmdPushConstants") return Streamed(real.cmdPushConstants, function, &CmdPushConstants);
    if (n == "vkCmdDispatch") return Streamed(real.cmdDispatch, function, &CmdDispatch);
    if (n == "vkCmdDispatchIndirect") return Streamed(real.cmdDispatchIndirect, function, &CmdDispatchIndirect);
    if (n == "vkCmdBindPipeline") return Streamed(real.cmdBindPipeline, function, &CmdBindPipeline);
    if (n == "vkCmdBindDescriptorSets") return Streamed(real.cmdBindDescriptorSets, function, &CmdBindDescriptorSetsThunk);
    if (n == "vkCmdUpdateBuffer") return Streamed(real.cmdUpdateBuffer, function, &CmdUpdateBuffer);
    if (n == "vkCmdFillBuffer") return Streamed(real.cmdFillBuffer, function, &CmdFillBuffer);
    if (n == "vkCmdResetQueryPool") return Streamed(real.cmdResetQueryPool, function, &CmdResetQueryPool);
    if (n == "vkCmdBeginQuery") return Streamed(real.cmdBeginQuery, function, &CmdBeginQuery);
    if (n == "vkCmdEndQuery") return Streamed(real.cmdEndQuery, function, &CmdEndQuery);
    if (n == "vkCmdCopyQueryPoolResults") return Streamed(real.cmdCopyQueryPoolResults, function, &CmdCopyQueryPoolResults);
    if (n == "vkCmdBeginRenderPass") return Streamed(real.cmdBeginRenderPass, function, &CmdBeginRenderPass);
    if (n == "vkCmdEndRenderPass") return Streamed(real.cmdEndRenderPass, function, &CmdEndRenderPass);
    if (n == "vkCmdDraw") return Streamed(real.cmdDraw, function, &CmdDraw);
    if (n == "vkCmdDrawIndexed") return Streamed(real.cmdDrawIndexed, function, &CmdDrawIndexed);
    if (n == "vkCmdDrawIndirect") return Streamed(real.cmdDrawIndirect, function, &CmdDrawIndirect);
    if (n == "vkCmdDrawIndexedIndirect") return Streamed(real.cmdDrawIndexedIndirect, function, &CmdDrawIndexedIndirect);
    if (n == "vkCmdDrawIndirectCount" || n == "vkCmdDrawIndirectCountKHR") return Streamed(real.cmdDrawIndirectCount, function, &CmdDrawIndirectCount);
    if (n == "vkCmdDrawIndexedIndirectCount" || n == "vkCmdDrawIndexedIndirectCountKHR") return Streamed(real.cmdDrawIndexedIndirectCount, function, &CmdDrawIndexedIndirectCount);
    if (n == "vkCmdDrawMeshTasksEXT") return Streamed(real.cmdDrawMeshTasks, function, &CmdDrawMeshTasks);
    if (n == "vkCmdDrawMeshTasksIndirectEXT") return Streamed(real.cmdDrawMeshTasksIndirect, function, &CmdDrawMeshTasksIndirect);
    if (n == "vkCmdSetViewport") return Streamed(real.cmdSetViewport, function, &CmdSetViewport);
    if (n == "vkCmdSetScissor") return Streamed(real.cmdSetScissor, function, &CmdSetScissor);
    if (n == "vkCmdSetDepthBias") return Streamed(real.cmdSetDepthBias, function, &CmdSetDepthBias);
    if (n == "vkCmdSetDepthBounds") return Streamed(real.cmdSetDepthBounds, function, &CmdSetDepthBounds);
    if (n == "vkCmdBindVertexBuffers") return Streamed(real.cmdBindVertexBuffers, function, &CmdBindVertexBuffers);
    if (n == "vkCmdBindIndexBuffer") return Streamed(real.cmdBindIndexBuffer, function, &CmdBindIndexBuffer);
    if (n == "vkCmdPushDescriptorSetKHR") return Streamed(real.cmdPushDescriptorSet, function, &CmdPushDescriptorSet);
    if (n == "vkBeginCommandBuffer") return Streamed(real.beginCommandBuffer, function, &BeginCommandBuffer);
    if (n == "vkEndCommandBuffer") return Streamed(real.endCommandBuffer, function, &EndCommandBuffer);
    if (n == "vkResetCommandBuffer") return Streamed(real.resetCommandBuffer, function, &ResetCommandBuffer);
    if (n == "vkQueueSubmit") return Streamed(real.queueSubmit, function, &QueueSubmit);
    if (n == "vkUpdateDescriptorSets") return Streamed(real.updateDescriptorSets, function, &UpdateDescriptorSets);
    if (n == "vkCmdSetCheckpointNV") return Streamed(real.cmdSetCheckpoint, function, &CmdSetCheckpoint);
    if (n == "vkCmdWriteBufferMarkerAMD") return Streamed(real.cmdWriteBufferMarker, function, &CmdWriteBufferMarker);
    // Waiting for the stream first.
    if (n == "vkWaitForFences") return Draining<PFN_vkWaitForFences>(function);
    if (n == "vkResetFences") return Draining<PFN_vkResetFences>(function);
    if (n == "vkQueueWaitIdle") return Draining<PFN_vkQueueWaitIdle>(function);
    if (n == "vkDeviceWaitIdle") return Draining<PFN_vkDeviceWaitIdle>(function);
    if (n == "vkWaitSemaphores" || n == "vkWaitSemaphoresKHR") return Draining<PFN_vkWaitSemaphores>(function);
    if (n == "vkQueuePresentKHR") return Draining<PFN_vkQueuePresentKHR>(function);
    if (n == "vkGetQueryPoolResults") return Draining<PFN_vkGetQueryPoolResults>(function);
    if (n == "vkAllocateCommandBuffers") return Draining<PFN_vkAllocateCommandBuffers>(function);
    if (n == "vkFreeCommandBuffers") return Draining<PFN_vkFreeCommandBuffers>(function);
    if (n == "vkResetCommandPool") return Draining<PFN_vkResetCommandPool>(function);
    if (n == "vkResetDescriptorPool") return Draining<PFN_vkResetDescriptorPool>(function);
    if (n == "vkFreeDescriptorSets") return Draining<PFN_vkFreeDescriptorSets>(function);
    if (n == "vkFreeMemory") return Draining<PFN_vkFreeMemory>(function);
    if (n == "vkGetDeviceFaultInfoEXT") return Draining<PFN_vkGetDeviceFaultInfoEXT>(function);
    if (n == "vkDestroyBuffer") return Draining<PFN_vkDestroyBuffer>(function);
    if (n == "vkDestroyImage") return Draining<PFN_vkDestroyImage>(function);
    if (n == "vkDestroyImageView") return Draining<PFN_vkDestroyImageView>(function);
    if (n == "vkDestroyFramebuffer") return Draining<PFN_vkDestroyFramebuffer>(function);
    if (n == "vkDestroyRenderPass") return Draining<PFN_vkDestroyRenderPass>(function);
    if (n == "vkDestroyPipeline") return Draining<PFN_vkDestroyPipeline>(function);
    if (n == "vkDestroyPipelineLayout") return Draining<PFN_vkDestroyPipelineLayout>(function);
    if (n == "vkDestroyDescriptorPool") return Draining<PFN_vkDestroyDescriptorPool>(function);
    if (n == "vkDestroyDescriptorSetLayout") return Draining<PFN_vkDestroyDescriptorSetLayout>(function);
    if (n == "vkDestroySampler") return Draining<PFN_vkDestroySampler>(function);
    if (n == "vkDestroyQueryPool") return Draining<PFN_vkDestroyQueryPool>(function);
    if (n == "vkDestroyFence") return Draining<PFN_vkDestroyFence>(function);
    if (n == "vkDestroySemaphore") return Draining<PFN_vkDestroySemaphore>(function);
    if (n == "vkDestroyCommandPool") return Draining<PFN_vkDestroyCommandPool>(function);
    if (n == "vkDestroyShaderModule") return Draining<PFN_vkDestroyShaderModule>(function);
    if (n == "vkDestroySwapchainKHR") return Draining<PFN_vkDestroySwapchainKHR>(function);
    return function;
}

}
