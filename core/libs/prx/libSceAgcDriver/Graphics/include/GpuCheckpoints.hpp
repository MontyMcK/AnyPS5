#pragma once

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DebugControls.hpp"
#include <memory>
#include <mutex>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string_view>
#include <unordered_map>

// Debug aid: APS5_GPU_CHECKPOINTS=1 (the recorder thread streams the markers in order with the
// commands they mark) marks every dispatch and draw with VK_NV_device_diagnostic_checkpoints;
// on a device loss the last markers each pipeline stage reached are printed with what they mark.
namespace AgcDriver::Graphics::GpuCheckpoints {

struct Mark {
    const char* kind = "";
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::uint64_t c = 0;
    char text[160] = {};
};

inline bool Requested() {
    static const bool requested = std::getenv("APS5_GPU_CHECKPOINTS") != nullptr;
    return requested;
}

inline std::atomic<bool>& Active() {
    static std::atomic<bool> active{false};
    return active;
}

inline std::array<Mark, 1u << 16>& Ring() {
    static auto* ring = new std::array<Mark, 1u << 16>();
    return *ring;
}

// Shader variant -> guest code address, noted by the draw path while checkpoints are on.
struct Variants {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::uint64_t> address;
};

inline Variants& VariantMap() {
    static auto* map = new Variants();
    return *map;
}

inline void NoteVariant(std::uint64_t variant, std::uint64_t codeAddress) {
    if ((!Active().load(std::memory_order_relaxed) && DebugValue(DebugKey::TraceReads) == 0) || variant == 0) return;
    auto& map = VariantMap();
    std::lock_guard lock(map.mutex);
    map.address[variant] = codeAddress;
}

inline std::uint64_t VariantAddress(std::uint64_t variant) {
    auto& map = VariantMap();
    std::lock_guard lock(map.mutex);
    const auto found = map.address.find(variant);
    return found != map.address.end() ? found->second : 0;
}

inline std::atomic<std::uint64_t>& Serial() {
    static std::atomic<std::uint64_t> serial{0};
    return serial;
}

// VK_AMD_buffer_marker breadcrumbs: dword 0 takes each marker's serial at the top of the pipe (the
// command was reached), dword 1 at the bottom (everything before it completed). Host memory, so it
// stays readable after a device loss.
inline std::atomic<bool>& MarkersActive() {
    static std::atomic<bool> active{false};
    return active;
}

inline std::shared_ptr<Buffer>& MarkerBuffer() {
    static std::shared_ptr<Buffer> buffer;
    return buffer;
}

inline void Set(const Context& context, VkCommandBuffer commands, const char* kind, std::uint64_t a, std::uint64_t b = 0, std::uint64_t c = 0, const char* text = nullptr) {
    if (!Active().load(std::memory_order_relaxed)) return;
    static const auto set = reinterpret_cast<PFN_vkCmdSetCheckpointNV>(CachedDeviceProc(context.deviceProc, context.device, "vkCmdSetCheckpointNV"));
    const auto serial = Serial().fetch_add(1, std::memory_order_relaxed) + 1;
    auto& slot = Ring()[serial & (Ring().size() - 1)];
    slot.kind = kind;
    slot.a = a;
    slot.b = b;
    slot.c = c;
    std::snprintf(slot.text, sizeof(slot.text), "%s", text != nullptr ? text : "");
    if (set != nullptr) set(commands, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(serial)));
    if (!MarkersActive().load(std::memory_order_relaxed)) return;
    static const auto marker = reinterpret_cast<PFN_vkCmdWriteBufferMarkerAMD>(CachedDeviceProc(context.deviceProc, context.device, "vkCmdWriteBufferMarkerAMD"));
    if (marker == nullptr) return;
    static std::once_flag made;
    std::call_once(made, [&] {
        MarkerBuffer() = std::make_shared<Buffer>(context, 64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        std::fill(MarkerBuffer()->Bytes().begin(), MarkerBuffer()->Bytes().end(), std::byte{0});
    });
    const auto handle = MarkerBuffer()->Handle();
    marker(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, handle, 0, static_cast<std::uint32_t>(serial));
    marker(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, handle, 4, static_cast<std::uint32_t>(serial));
}

inline void ReportMarkers() {
    if (!MarkersActive().load(std::memory_order_relaxed) || MarkerBuffer() == nullptr) return;
    const auto bytes = MarkerBuffer()->Bytes();
    std::uint32_t top = 0, bottom = 0;
    std::memcpy(&top, bytes.data(), 4);
    std::memcpy(&bottom, bytes.data() + 4, 4);
    const auto newest = Serial().load();
    std::fprintf(stderr, "[markers] last reached (top of pipe) %u, last with all earlier work done (bottom) %u, newest recorded %llu\n", top, bottom, static_cast<unsigned long long>(newest));
    // Everything between `bottom` and `top` was in flight; the fault is in one of these.
    for (std::uint64_t serial = bottom; serial != 0 && serial <= static_cast<std::uint64_t>(top) + 2 && serial <= newest && serial < static_cast<std::uint64_t>(bottom) + 24; ++serial) {
        const auto& mark = Ring()[serial & (Ring().size() - 1)];
        std::fprintf(stderr, "[markers] %s %llu: %s 0x%llx 0x%llx 0x%llx\n", serial <= bottom ? "done" : serial <= top ? "IN FLIGHT" : "queued", static_cast<unsigned long long>(serial), mark.kind, static_cast<unsigned long long>(mark.a), static_cast<unsigned long long>(mark.b), static_cast<unsigned long long>(mark.c));
        if (mark.text[0] != 0) std::fprintf(stderr, "[markers]   %s\n", mark.text);
        if (std::string_view(mark.kind).rfind("draw", 0) == 0) std::fprintf(stderr, "[markers]   draw shaders: vertex code 0x%llx, pixel code 0x%llx\n", static_cast<unsigned long long>(VariantAddress(mark.a)), static_cast<unsigned long long>(VariantAddress(mark.b)));
    }
    std::fflush(stderr);
}

inline void Report(VkDevice device, VkQueue queue, PFN_vkGetDeviceProcAddr deviceProc) {
    ReportMarkers();
    if (!Active().load(std::memory_order_relaxed) || deviceProc == nullptr) return;
    const auto get = reinterpret_cast<PFN_vkGetQueueCheckpointDataNV>(deviceProc(device, "vkGetQueueCheckpointDataNV"));
    if (get == nullptr) return;
    std::uint32_t count = 0;
    get(queue, &count, nullptr);
    std::vector<VkCheckpointDataNV> data(count, VkCheckpointDataNV{VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
    get(queue, &count, data.data());
    const auto newest = Serial().load();
    std::fprintf(stderr, "[checkpoints] %u stage records (newest marker recorded %llu)\n", count, static_cast<unsigned long long>(newest));
    for (const auto& item : data) {
        const auto serial = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(item.pCheckpointMarker));
        const auto& mark = Ring()[serial & (Ring().size() - 1)];
        std::fprintf(stderr, "[checkpoints] stage 0x%x reached marker %llu: %s 0x%llx 0x%llx 0x%llx\n", static_cast<unsigned>(item.stage), static_cast<unsigned long long>(serial), mark.kind, static_cast<unsigned long long>(mark.a), static_cast<unsigned long long>(mark.b), static_cast<unsigned long long>(mark.c));
    }
    // The markers recorded just after the furthest one reached: the work that was in flight.
    std::uint64_t furthest = 0;
    for (const auto& item : data) furthest = std::max<std::uint64_t>(furthest, reinterpret_cast<std::uintptr_t>(item.pCheckpointMarker));
    for (std::uint64_t serial = furthest + 1; serial <= newest && serial <= furthest + 6; ++serial) {
        const auto& mark = Ring()[serial & (Ring().size() - 1)];
        std::fprintf(stderr, "[checkpoints] next recorded %llu: %s 0x%llx 0x%llx 0x%llx\n", static_cast<unsigned long long>(serial), mark.kind, static_cast<unsigned long long>(mark.a), static_cast<unsigned long long>(mark.b), static_cast<unsigned long long>(mark.c));
    }
    std::fflush(stderr);
}

}
