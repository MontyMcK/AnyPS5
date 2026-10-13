#pragma once

#include <vulkan/vulkan.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <string>

// Debug aid (APS5_DEVICE_FAULT): every buffer device address range handed out, live and recently
// freed, so a device fault's reported address can be named (which object, whether it was freed and
// how long before). Keyed by the VkBuffer at destruction.
namespace AgcDriver::Graphics::AddressLedger {

struct Entry {
    VkDeviceAddress begin = 0;
    VkDeviceSize bytes = 0;
    const char* tag = "";
    std::uint64_t guest = 0;
    std::chrono::steady_clock::time_point made{};
    std::chrono::steady_clock::time_point freed{};
    bool live = true;
    char name[112] = {};
};

struct State {
    std::mutex mutex;
    std::unordered_map<VkBuffer, Entry> live;
    std::deque<Entry> freed;
    // VK_EXT_device_address_binding_report: every object's virtual address binding, by base.
    std::map<VkDeviceAddress, Entry> bound;
    // Labels given to objects at creation (NameObject), copied into their binding entries.
    std::unordered_map<std::uint64_t, std::string> names;
};

inline bool Enabled();
inline State& Ledger();

// A binding report (VK_EXT_device_address_binding_report): `object` is the bound object's handle,
// `type` names its object type.
inline void Bound(VkDeviceAddress begin, VkDeviceSize bytes, const char* type, std::uint64_t object, bool unbind) {
    if (!Enabled() || begin == 0) return;
    auto& state = Ledger();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (!unbind) {
        Entry entry{begin, bytes, type, object, now, {}, true};
        if (const auto named = state.names.find(object); named != state.names.end()) std::snprintf(entry.name, sizeof(entry.name), "%s", named->second.c_str());
        state.bound[begin] = entry;
        return;
    }
    auto found = state.bound.find(begin);
    Entry entry = found != state.bound.end() ? found->second : Entry{begin, bytes, type, object, now, {}, true};
    if (found != state.bound.end()) state.bound.erase(found);
    entry.live = false;
    entry.freed = now;
    state.freed.push_back(entry);
    if (state.freed.size() > 65536) state.freed.pop_front();
}

inline bool Enabled() {
    static const bool enabled = std::getenv("APS5_DEVICE_FAULT") != nullptr;
    return enabled;
}

inline State& Ledger() {
    static auto* state = new State();
    return *state;
}

inline void NameObject(std::uint64_t object, const char* name) {
    if (!Enabled() || object == 0) return;
    auto& state = Ledger();
    std::lock_guard lock(state.mutex);
    state.names[object] = name;
}

inline void Made(VkBuffer buffer, VkDeviceAddress begin, VkDeviceSize bytes, const char* tag, std::uint64_t guest = 0) {
    if (!Enabled() || buffer == VK_NULL_HANDLE || begin == 0) return;
    auto& state = Ledger();
    std::lock_guard lock(state.mutex);
    state.live[buffer] = Entry{begin, bytes, tag, guest, std::chrono::steady_clock::now(), {}, true};
}

inline void Destroyed(VkBuffer buffer) {
    if (!Enabled() || buffer == VK_NULL_HANDLE) return;
    auto& state = Ledger();
    std::lock_guard lock(state.mutex);
    const auto found = state.live.find(buffer);
    if (found == state.live.end()) return;
    auto entry = found->second;
    state.live.erase(found);
    entry.live = false;
    entry.freed = std::chrono::steady_clock::now();
    state.freed.push_back(entry);
    if (state.freed.size() > 65536) state.freed.pop_front();
}

inline void Report(VkDeviceAddress address, VkDeviceSize precision) {
    if (!Enabled()) return;
    auto& state = Ledger();
    std::lock_guard lock(state.mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto lo = precision > 1 ? address & ~(precision - 1) : address;
    const auto hi = lo + (precision > 1 ? precision : 1);
    const auto ms = [&](std::chrono::steady_clock::time_point t) { return std::chrono::duration<double, std::milli>(now - t).count(); };
    int shown = 0;
    VkDeviceAddress nearestBelow = 0, nearestAbove = ~0ull;
    const Entry* below = nullptr;
    const Entry* above = nullptr;
    for (const auto& [buffer, entry] : state.live) {
        if (entry.begin < hi && lo < entry.begin + entry.bytes) {
            std::fprintf(stderr, "[address-ledger] LIVE %s 0x%llx+0x%llx (guest 0x%llx) made %.0f ms ago contains the fault\n", entry.tag, static_cast<unsigned long long>(entry.begin), static_cast<unsigned long long>(entry.bytes), static_cast<unsigned long long>(entry.guest), ms(entry.made));
            ++shown;
        }
        if (entry.begin + entry.bytes <= lo && entry.begin + entry.bytes > nearestBelow) { nearestBelow = entry.begin + entry.bytes; below = &entry; }
        if (entry.begin >= hi && entry.begin < nearestAbove) { nearestAbove = entry.begin; above = &entry; }
    }
    for (const auto& [begin, entry] : state.bound) {
        if (entry.begin < hi && lo < entry.begin + entry.bytes) {
            std::fprintf(stderr, "[address-ledger] BOUND %s 0x%llx+0x%llx (object 0x%llx) bound %.0f ms ago contains the fault [%s]\n", entry.tag, static_cast<unsigned long long>(entry.begin), static_cast<unsigned long long>(entry.bytes), static_cast<unsigned long long>(entry.guest), ms(entry.made), entry.name);
            ++shown;
        }
    }
    for (auto it = state.freed.rbegin(); it != state.freed.rend() && shown < 12; ++it) {
        if (it->begin < hi && lo < it->begin + it->bytes) {
            std::fprintf(stderr, "[address-ledger] FREED %s 0x%llx+0x%llx (guest/object 0x%llx) made %.0f ms ago, freed %.0f ms ago contains the fault [%s]\n", it->tag, static_cast<unsigned long long>(it->begin), static_cast<unsigned long long>(it->bytes), static_cast<unsigned long long>(it->guest), ms(it->made), ms(it->freed), it->name);
            ++shown;
        }
    }
    if (shown == 0) std::fprintf(stderr, "[address-ledger] no live or recently freed range contains 0x%llx (%zu live, %zu freed tracked)\n", static_cast<unsigned long long>(address), state.live.size(), state.freed.size());
    if (below) std::fprintf(stderr, "[address-ledger] nearest live below: %s 0x%llx+0x%llx\n", below->tag, static_cast<unsigned long long>(below->begin), static_cast<unsigned long long>(below->bytes));
    if (above) std::fprintf(stderr, "[address-ledger] nearest live above: %s 0x%llx+0x%llx\n", above->tag, static_cast<unsigned long long>(above->begin), static_cast<unsigned long long>(above->bytes));
    std::fflush(stderr);
}

}
