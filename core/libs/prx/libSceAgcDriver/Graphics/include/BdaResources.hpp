#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"

namespace AgcDriver::Graphics {

bool LoopGuardTripped();

// The shape of a table's lookup index (BdaAbi::BucketHeader): `count` buckets of 1 << shift bytes
// from base, 0 for no index. A lookup outside [base, base + (count << shift)) searches every range,
// so any window is exact; only its depth differs. `windowed`: the dense window below was chosen.
struct BdaBucketWindow {
    std::uint64_t base = 0;
    std::uint32_t shift = 0;
    std::size_t count = 0;
    bool windowed = false;
};

// Over the joined, sorted table ranges, at most `limit` buckets. dense=false: one window over the
// whole span (the shape every table had before). That span runs from the exe image below
// 0x10000000 to the image mirrors at 0x7ff6..., so with 4096 buckets each is 32 GiB and the whole
// guest heap shares one: a cold lookup there binary-searches all of its joined entries. dense=true:
// the window covers only the cluster (ranges split at gaps over 4 GiB, mirrors excluded) holding
// the most entries, for the race the 16 GiB heap in 4 MiB buckets; the low image and the mirrors
// then take the full search. No cluster of 16 or more entries: the whole-span shape.
BdaBucketWindow BdaBucketShape(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges, std::size_t limit, bool dense);
// APS5_BDA_BUCKETS (default 4096), and the live flag bda_dense_buckets (APS5_BDA_DENSE_BUCKETS set
// is its value while the live-flags file does not name it) as the table builder reads them.
std::size_t BdaBucketLimit();
bool BdaDenseBuckets();

class BdaResources {
public:
    explicit BdaResources(const Context& context);
    ~BdaResources();
    BdaResources(const BdaResources&) = delete;
    BdaResources& operator=(const BdaResources&) = delete;
    BdaResources(const Context& context, const GuestBufferMemory& memory);
    VkDescriptorBufferInfo Table() const;
    VkDescriptorBufferInfo Fault() const;
    void CheckFault() const;
    std::unique_ptr<BdaResources> ShareTable(const Context& context) const;
    void SetMayWrite(bool value) { mayWrite = value; }
    bool MayWrite() const { return mayWrite; }
    // APS5_BDA_CLOCK=1: whether the clock words this buffer collects belong to a dispatch (Phase A,
    // where the lookups sit in the BVH refits' waterfall loops) or to a draw, reported apart.
    void SetCompute(bool value) { compute = value; }
    // APS5_PROFILE_DRAW, once bda_dense_buckets has been on: the lookup index of the tables built
    // (cache misses) since the last call, which resets them ([bda-buckets] next to [address]).
    struct BucketShapeStats {
        std::uint64_t buckets = 0;
        std::uint64_t entries = 0;
        std::uint64_t maxEntries = 0;
    };
    struct BucketStats {
        bool requested = false;
        std::uint64_t windowed = 0;
        std::uint64_t flagOff = 0;
        std::uint64_t noCluster = 0;
        std::uint64_t noIndex = 0;
        // Over the windowed tables: the window spans, and the entries that intersect the window
        // and those outside it (the latter take the full search).
        std::uint64_t windowBytes = 0;
        std::uint64_t inside = 0;
        std::uint64_t outside = 0;
        // Non-empty buckets and the entries they hold, for windowed and whole-span tables.
        BucketShapeStats window;
        BucketShapeStats whole;
    };
    static BucketStats TakeBucketCounters();
    // APS5_PROFILE_DRAW: page tables served from the per-device cache and built anew (cumulative),
    // and how many recent tables the cache holds right now.
    struct TableCacheStats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::size_t held = 0;
        // Hits served by the cached address space's serial alone (no hash, no compare; see
        // GuestBufferMemory::CachedAddressTable), counted in `hits` too.
        std::uint64_t spaceTables = 0;
        // Why the most recently used table did not serve a build ([bda-table] first-differing entry
        // class): its owners were gone (expired weak reference), or its word hash differed, split by
        // whether the first differing range begins below 0x10000000 (a low mapping: the exe image,
        // the system heaps) or in the guest heap above; `sameHash` is a hash match whose ranges
        // still differed, `empty` a lookup with no entry held.
        std::uint64_t firstExpired = 0;
        std::uint64_t firstDiffersLow = 0;
        std::uint64_t firstDiffersHeap = 0;
        std::uint64_t firstSameHash = 0;
        std::uint64_t firstEmpty = 0;
    };
    static TableCacheStats TableCacheCounters();

private:
    // The page table is read-only to the shader, so consecutive builds mapping the same ranges to the
    // same device addresses share one buffer while one of them is alive (see the table cache in
    // BdaResources.cpp, which refers to it weakly).
    void markWrittenPages() const;
    std::shared_ptr<Buffer> table;
    std::unique_ptr<Buffer> fault;
    std::size_t tableBytes = 0;
    bool mayWrite = true;
    bool compute = false;
    mutable bool checked = false;
    VkDevice device = VK_NULL_HANDLE;
};

}

#endif
