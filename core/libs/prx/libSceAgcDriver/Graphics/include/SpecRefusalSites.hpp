#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SPECREFUSALSITES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SPECREFUSALSITES_HPP

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {

// What blocked a prep-thread speculation refused for pending results (Miss::Pending): the kind,
// the draw's surface the scan found overlapped, and the pending image over it. Logging only.
struct SpeculationRefusalDetail {
    enum Kind : std::uint8_t { None, Identity, OwnSource, ForeignView, Storage, Snapshot, Count };
    std::uint8_t kind = None;
    std::uint64_t surface = 0;
    std::uint64_t surfaceBytes = 0;
    std::uint64_t blocker = 0;
    std::uint64_t blockerBytes = 0;
    std::uint32_t blockerFormat = 0;
    std::uint32_t blockerWidth = 0;
    std::uint32_t blockerHeight = 0;
};

inline const char* PendingKindName(std::uint8_t kind) {
    constexpr const char* names[] = {"-", "identity", "own-source view", "foreign over a view", "over a storage image", "over a snapshot"};
    return kind < SpeculationRefusalDetail::Count ? names[kind] : "?";
}

// APS5_PROFILE_DRAW's [spec-refusals] table: draw commits by what became of their speculation
// (class and sub-kind) and draw site (programs, target, and for pending refusals the blocker).
// Not thread-safe: the caller serializes Add and Report.
class RefusalSites {
public:
    struct Key {
        std::uint8_t cls = 0;        // the [commit] line's class
        std::uint8_t sub = 0;        // a refusal on commit: the committer's miss
        std::uint64_t vertex = 0;    // the first non-fragment stage's code address
        std::uint64_t pixel = 0;     // the fragment stage's code address
        std::uint64_t target = 0;    // the colour target's address and bytes
        std::uint64_t targetBytes = 0;
        SpeculationRefusalDetail detail;
        auto tie() const { return std::tie(cls, sub, vertex, pixel, target, targetBytes, detail.kind, detail.surface, detail.surfaceBytes, detail.blocker, detail.blockerBytes, detail.blockerFormat); }
        bool operator<(const Key& other) const { return tie() < other.tie(); }
    };

    void Add(const Key& key, std::uint64_t ns) {
        auto& entry = sites[key];
        ++entry.count;
        entry.ns += ns;
    }

    std::size_t Size() const { return sites.size(); }

    // The report lines (each starting "[spec-refusals]"), heaviest classes first; clears the table.
    // `className(cls, sub)` names a class. Classes in `skip` (e.g. accepted) are left out.
    std::vector<std::string> Report(const std::function<std::string(std::uint8_t, std::uint8_t)>& className, const std::vector<std::uint8_t>& skip, std::size_t classes = 8, std::size_t perClass = 6) {
        struct Group {
            std::uint64_t count = 0, ns = 0;
            std::vector<std::pair<Key, Entry>> sites;
        };
        std::map<std::pair<std::uint8_t, std::uint8_t>, Group> groups;
        for (const auto& [key, entry] : sites) {
            if (std::find(skip.begin(), skip.end(), key.cls) != skip.end()) continue;
            auto& group = groups[{key.cls, key.sub}];
            group.count += entry.count;
            group.ns += entry.ns;
            group.sites.emplace_back(key, entry);
        }
        std::vector<std::pair<std::pair<std::uint8_t, std::uint8_t>, Group*>> order;
        for (auto& [id, group] : groups) order.emplace_back(id, &group);
        std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second->ns > b.second->ns; });
        std::vector<std::string> lines;
        char text[512];
        for (std::size_t g = 0; g < std::min(order.size(), classes); ++g) {
            auto& group = *order[g].second;
            std::snprintf(text, sizeof(text), "[spec-refusals] %s: %llu commits at %.1f us (%.1f ms), %zu sites", className(order[g].first.first, order[g].first.second).c_str(), static_cast<unsigned long long>(group.count), static_cast<double>(group.ns) / 1000.0 / static_cast<double>(group.count), static_cast<double>(group.ns) / 1e6, group.sites.size());
            lines.emplace_back(text);
            std::sort(group.sites.begin(), group.sites.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
            for (std::size_t s = 0; s < std::min(group.sites.size(), perClass); ++s) {
                const auto& [key, entry] = group.sites[s];
                int used = std::snprintf(text, sizeof(text), "[spec-refusals]   %llu at %.1f us: vs 0x%llx ps 0x%llx target 0x%llx+0x%llx", static_cast<unsigned long long>(entry.count), static_cast<double>(entry.ns) / 1000.0 / static_cast<double>(entry.count), static_cast<unsigned long long>(key.vertex), static_cast<unsigned long long>(key.pixel), static_cast<unsigned long long>(key.target), static_cast<unsigned long long>(key.targetBytes));
                if (key.detail.kind != SpeculationRefusalDetail::None && used > 0 && static_cast<std::size_t>(used) < sizeof(text)) {
                    std::snprintf(text + used, sizeof(text) - static_cast<std::size_t>(used), "; pending %s: surface 0x%llx+0x%llx, blocker 0x%llx+0x%llx fmt %u %ux%u", PendingKindName(key.detail.kind), static_cast<unsigned long long>(key.detail.surface), static_cast<unsigned long long>(key.detail.surfaceBytes), static_cast<unsigned long long>(key.detail.blocker), static_cast<unsigned long long>(key.detail.blockerBytes), key.detail.blockerFormat, key.detail.blockerWidth, key.detail.blockerHeight);
                }
                lines.emplace_back(text);
            }
        }
        sites.clear();
        return lines;
    }

private:
    struct Entry {
        std::uint64_t count = 0;
        std::uint64_t ns = 0;
    };
    std::map<Key, Entry> sites;
};

}

#endif
