#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INDEXSCANMEMO_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INDEXSCANMEMO_HPP

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace AgcDriver::Graphics {

// The highest index of in-place index ranges a thread scanned (live `index_scan_memo`, Draw.cpp):
// per range (address, bytes, index size), the highest index and the write-tracker generation
// collected before the scan. A lookup answers from the entry while the range is unchanged since
// that generation, and scans (and remembers) otherwise.
class IndexScanMemoTable {
public:
    static constexpr std::size_t Capacity = 16384;

    // `collect()`: collects the range's writes and returns the tracker generation (0: untracked).
    // `unchanged(generation)`: whether the range is unchanged since `generation`.
    // `scan()`: the highest index of the range's current bytes.
    // `hit` (optional) says whether the entry answered.
    template<typename TCollect, typename TUnchanged, typename TScan>
    std::uint32_t Highest(std::uint64_t address, std::uint64_t bytes, std::uint32_t indexSize, TCollect&& collect, TUnchanged&& unchanged, TScan&& scan, bool* hit = nullptr) {
        const Key key{address, bytes, indexSize};
        if (hit != nullptr) *hit = false;
        if (const auto found = entries.find(key); found != entries.end()) {
            if (collect() != 0 && unchanged(found->second.generation)) {
                if (hit != nullptr) *hit = true;
                return found->second.highest;
            }
            entries.erase(found);
        }
        // Collected before the bytes are read: a write after it stamps a newer generation.
        const std::uint64_t generation = collect();
        const std::uint32_t highest = scan();
        if (generation != 0) {
            if (entries.size() >= Capacity) entries.clear();
            entries.emplace(key, Entry{generation, highest});
        }
        return highest;
    }

    std::size_t Size() const { return entries.size(); }

private:
    struct Key {
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint32_t indexSize;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            auto hash = key.address * 0x9e3779b97f4a7c15ull;
            hash ^= (key.bytes * 4u + key.indexSize) * 0xc2b2ae3d27d4eb4full;
            return static_cast<std::size_t>(hash ^ (hash >> 29u));
        }
    };
    struct Entry {
        std::uint64_t generation = 0;
        std::uint32_t highest = 0;
    };
    std::unordered_map<Key, Entry, KeyHash> entries;
};

}

#endif
