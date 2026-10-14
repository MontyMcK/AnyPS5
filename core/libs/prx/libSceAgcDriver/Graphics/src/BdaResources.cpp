#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <sstream>
#include <string_view>

namespace AgcDriver::Graphics {

namespace {

// The page tables built per device, reused by a build whose ranges (guest ranges, device
// addresses, permissions) equal one of them: consecutive address-based builds map the same registered
// set through the same imports and mirrors, so the ~1200-entry table repeats. The shader only reads
// the table (its lookup keeps its state in function variables), and a batch in flight keeps its
// BdaResources, so sharing the buffer is safe. The cache holds the tables weakly: they never outlive
// the builds that own them (a static owner would destroy the buffer after its device, at exit or on
// device replacement, and would fail the tests' leak check), and with deferred lease release the
// previous builds are still in flight when the next one asks, which is what makes the hit rate.
// One table was enough while every build repeated the last one exactly; now the misaligned
// descriptor sub-ranges the GPU copies out of imports sit in pool staging buffers whose device
// addresses cycle through the few buffers the batches in flight leave free, and the three
// address-based programs of one queue interleave with the lease draws, so a table differs from the
// previous one and comes back a few builds later (69% misses in the wave-8 run, 4% before the GPU
// copies). Hence several recent tables, most recently used first, matched by a word hash before
// the byte compare. APS5_BDA_TABLE_CACHE_ENTRIES sets how many (default 8; 1 is the previous
// behaviour); APS5_NO_BDA_TABLE_CACHE=1 builds every table.
// A table built for (or found equal to) the cached address space's ranges carries the space's
// serial: a build served by that space alone binds it by the serial, with no hash and no compare.
struct TableEntry {
    std::uint64_t hash;
    std::vector<ShaderRecompiler::BdaAbi::Range> ranges;
    std::weak_ptr<Buffer> buffer;
    std::uint64_t spaceSerial = 0;
    bool deviceLocal = false;
    // Built with the dense window (BdaBucketWindow::windowed): the index layout is part of the key,
    // or a build after a live toggle would keep binding the other layout's tables by the serial
    // (exact either way, but the A/B would measure the wrong one). Keyed on the layout built, not
    // the flag: with no cluster the flag-on table is the whole-span one and shares its entry.
    bool dense = false;
};

struct TableCache {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::list<TableEntry> entries;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    BdaResources::TableCacheStats classes;
};

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right);

std::shared_ptr<Buffer> makeTableBuffer(const Context& context, std::size_t bytes, bool local) {
    if (local) {
        try {
            auto buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            static std::once_flag reported;
            std::call_once(reported, [] { std::fprintf(stderr, "[bda-table] using mapped device-local memory for read-only address tables\n"); });
            return buffer;
        } catch (const std::runtime_error& error) {
            if (std::string_view(error.what()) != "AGC graphics: required Vulkan memory type is unavailable") throw;
            static std::once_flag reported;
            std::call_once(reported, [] { std::fprintf(stderr, "[bda-table] mapped device-local memory unavailable; using host memory\n"); });
        }
    }
    return std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

// APS5_PROFILE_DRAW: why the most recently used entry did not serve this build (see
// TableCacheStats). Under the cache mutex.
void classifyFirstEntry(TableCache& cache, std::uint64_t hash, const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& classes = cache.classes;
    if (cache.entries.empty()) {
        ++classes.firstEmpty;
        return;
    }
    const auto& first = cache.entries.front();
    if (first.hash == hash) {
        if (first.buffer.expired()) ++classes.firstExpired;
        else if (!sameRanges(first.ranges, ranges)) ++classes.firstSameHash;
        return;
    }
    if (first.buffer.expired()) {
        ++classes.firstExpired;
        return;
    }
    const auto common = std::min(first.ranges.size(), ranges.size());
    std::size_t at = 0;
    while (at < common && std::memcmp(&first.ranges[at], &ranges[at], sizeof(ranges[at])) == 0) ++at;
    const auto begin = at < ranges.size() ? ranges[at].begin : at < first.ranges.size() ? first.ranges[at].begin : 0;
    ++(begin < 0x10000000ull ? classes.firstDiffersLow : classes.firstDiffersHeap);
}

TableCache& Tables() {
    static TableCache cache;
    return cache;
}

bool tableCacheEnabled() {
    static const bool disabled = std::getenv("APS5_NO_BDA_TABLE_CACHE") != nullptr;
    return !disabled;
}

std::size_t tableCacheEntries() {
    static const std::size_t entries = [] {
        const char* value = std::getenv("APS5_BDA_TABLE_CACHE_ENTRIES");
        const auto parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 8ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(parsed, 1, 64));
    }();
    return entries;
}

bool sameRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& left, const std::vector<ShaderRecompiler::BdaAbi::Range>& right) {
    return left.size() == right.size() && (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(left.front())) == 0);
}

// A 64-bit mix of the table's words: a table is ~1200 ranges of four words, and comparing the bytes
// against every held table would cost as much as building it.
std::uint64_t hashRanges(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    static_assert(sizeof(ShaderRecompiler::BdaAbi::Range) % sizeof(std::uint64_t) == 0);
    std::uint64_t hash = 0x9e3779b97f4a7c15ull ^ ranges.size();
    for (const auto& range : ranges) {
        std::uint64_t words[sizeof(range) / sizeof(std::uint64_t)];
        std::memcpy(words, &range, sizeof(range));
        for (const auto word : words) {
            hash ^= word;
            hash *= 0xff51afd7ed558ccdull;
            hash ^= hash >> 33u;
        }
    }
    return hash;
}

}

namespace {

struct FaultBufferPool {
    std::mutex mutex;
    VkDevice device = VK_NULL_HANDLE;
    std::vector<std::unique_ptr<Buffer>> free;
};

FaultBufferPool& FaultBuffers() {
    static FaultBufferPool* pool = new FaultBufferPool();
    return *pool;
}

bool FaultBufferPoolEnabled() {
    static const bool enabled = std::getenv("APS5_NO_FAULT_BUFFER_POOL") == nullptr;
    return enabled;
}

constexpr std::size_t FaultHeaderBytes = (ShaderRecompiler::BdaAbi::WrittenOverflowWord + 1) * sizeof(std::uint32_t);

struct BucketIndex {
    ShaderRecompiler::BdaAbi::BucketHeader header{};
    std::vector<std::uint32_t> bounds;
};

std::size_t bucketBytes(const BdaBucketWindow& window) {
    return window.count == 0 ? 0 : sizeof(ShaderRecompiler::BdaAbi::BucketHeader) + window.count * 2u * sizeof(std::uint32_t);
}

// Per bucket, the ranges [low, high) that intersect it. The sweep works for a window starting past
// the first ranges (the dense window): `low` skips the ranges that end below the bucket.
BucketIndex buildBuckets(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges, const BdaBucketWindow& window) {
    BucketIndex index;
    if (window.count == 0) return index;
    index.header.base = window.base;
    index.header.shift = window.shift;
    index.bounds.resize(window.count * 2u);
    std::size_t low = 0;
    std::size_t high = 0;
    for (std::size_t bucket = 0; bucket < window.count; ++bucket) {
        const auto start = index.header.base + (static_cast<std::uint64_t>(bucket) << window.shift);
        const auto stop = start + (std::uint64_t{1} << window.shift);
        while (low < ranges.size() && ranges[low].end <= start) ++low;
        while (high < ranges.size() && ranges[high].begin < stop) ++high;
        index.bounds[bucket * 2u] = static_cast<std::uint32_t>(low);
        index.bounds[bucket * 2u + 1u] = static_cast<std::uint32_t>(high);
    }
    return index;
}

// [bda-buckets] (see BdaResources::TakeBucketCounters): gathered only under APS5_PROFILE_DRAW, on
// table builds (cache misses), and reported once the dense window has been asked for, so a run that
// never sets bda_dense_buckets prints what it printed before.
struct BucketCounters {
    std::mutex mutex;
    BdaResources::BucketStats stats;
};

BucketCounters& Buckets() {
    static BucketCounters counters;
    return counters;
}

std::atomic<bool> denseRequested{false};

void countBuckets(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges, const BdaBucketWindow& window, const BucketIndex& index, bool dense) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // Nothing is printed until the window has been asked for, so do not time the walk before then.
    if (!profile || !denseRequested.load(std::memory_order_relaxed)) return;
    std::uint64_t inside = 0;
    if (window.windowed) {
        const auto stop = window.base + (static_cast<std::uint64_t>(window.count) << window.shift);
        for (const auto& range : ranges) inside += range.end > window.base && range.begin < stop ? 1u : 0u;
    }
    BdaResources::BucketShapeStats shape;
    for (std::size_t bucket = 0; bucket < window.count; ++bucket) {
        const std::uint64_t entries = index.bounds[bucket * 2u + 1u] - index.bounds[bucket * 2u];
        if (entries == 0) continue;
        ++shape.buckets;
        shape.entries += entries;
        shape.maxEntries = std::max(shape.maxEntries, entries);
    }
    auto& counters = Buckets();
    std::lock_guard lock(counters.mutex);
    auto& stats = counters.stats;
    if (window.count == 0) ++stats.noIndex;
    else if (window.windowed) ++stats.windowed;
    else if (dense) ++stats.noCluster;
    else ++stats.flagOff;
    if (window.windowed) {
        stats.windowBytes += static_cast<std::uint64_t>(window.count) << window.shift;
        stats.inside += inside;
        stats.outside += ranges.size() - inside;
    }
    auto& into = window.windowed ? stats.window : stats.whole;
    into.buckets += shape.buckets;
    into.entries += shape.entries;
    into.maxEntries = std::max(into.maxEntries, shape.maxEntries);
}

}

std::size_t BdaBucketLimit() {
    // APS5_BDA_BUCKETS=N caps the lookup index at N buckets (0: no index, every lookup searches all ranges).
    static const std::size_t limit = [] {
        const char* text = std::getenv("APS5_BDA_BUCKETS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{4096};
    }();
    return limit;
}

bool BdaDenseBuckets() {
    static const bool requested = std::getenv("APS5_BDA_DENSE_BUCKETS") != nullptr;
    const bool dense = LiveFlag("bda_dense_buckets", requested);
    if (dense && !denseRequested.load(std::memory_order_relaxed)) denseRequested.store(true, std::memory_order_relaxed);
    return dense;
}

BdaBucketWindow BdaBucketShape(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges, std::size_t limit, bool dense) {
    if (limit == 0 || ranges.size() < 16) return {};
    // [first, last) in at most `limit` buckets of the smallest power of two from 4 KiB.
    const auto fit = [limit](std::uint64_t first, std::uint64_t last) -> BdaBucketWindow {
        const auto bucketsFor = [&](std::uint32_t bits) { return (last - (first & ~((std::uint64_t{1} << bits) - 1u)) + (std::uint64_t{1} << bits) - 1u) >> bits; };
        std::uint32_t shift = 12;
        while (shift < 48 && bucketsFor(shift) > limit) ++shift;
        const auto count = bucketsFor(shift);
        if (count == 0 || count > limit) return {};
        return {first & ~((std::uint64_t{1} << shift) - 1u), shift, static_cast<std::size_t>(count), false};
    };
    if (dense) {
        // Clusters split where the next range starts more than 4 GiB past the previous one's end:
        // the exe image and system heaps below 0x10000000, the guest heap at 0x10_0000_0000 and
        // up, the image mirrors at 0x7ff6.... The mirrors are left out: they are a few large
        // ranges, and the lookups that serialize (the BVH refits' waterfall V#s) read guest memory.
        constexpr std::uint64_t ClusterGap = std::uint64_t{4} << 30u;
        constexpr std::uint64_t MirrorFloor = 0x7000'0000'0000ull;
        std::size_t bestBegin = 0, bestEnd = 0, begin = 0;
        for (std::size_t at = 1; at <= ranges.size(); ++at) {
            if (at < ranges.size() && !(ranges[at].begin > ranges[at - 1u].end && ranges[at].begin - ranges[at - 1u].end > ClusterGap)) continue;
            if (ranges[begin].begin < MirrorFloor) {
                const auto entries = at - begin, best = bestEnd - bestBegin;
                const auto span = ranges[at - 1u].end - ranges[begin].begin;
                if (entries > best || (entries == best && span < ranges[bestEnd - 1u].end - ranges[bestBegin].begin)) {
                    bestBegin = begin;
                    bestEnd = at;
                }
            }
            begin = at;
        }
        if (bestEnd - bestBegin >= 16) {
            auto window = fit(ranges[bestBegin].begin, ranges[bestEnd - 1u].end);
            if (window.count != 0) {
                window.windowed = true;
                return window;
            }
        }
    }
    return fit(ranges.front().begin, ranges.back().end);
}

BdaResources::BucketStats BdaResources::TakeBucketCounters() {
    auto& counters = Buckets();
    std::lock_guard lock(counters.mutex);
    auto stats = counters.stats;
    counters.stats = {};
    stats.requested = denseRequested.load(std::memory_order_relaxed);
    return stats;
}

BdaResources::BdaResources(const Context& context) : device(context.device) {
    static_assert(std::endian::native == std::endian::little);
    Require(ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA fault buffer exceeds storage buffer range limit");
    if (FaultBufferPoolEnabled()) {
        auto& pool = FaultBuffers();
        std::lock_guard lock(pool.mutex);
        if (pool.device == context.device && !pool.free.empty()) {
            fault = std::move(pool.free.back());
            pool.free.pop_back();
        }
    }
    if (fault != nullptr) {
        std::memset(fault->Bytes().data(), 0, FaultHeaderBytes);
        return;
    }
    fault = std::make_unique<Buffer>(context, ShaderRecompiler::BdaAbi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memset(fault->Bytes().data(), 0, fault->Bytes().size());
}

BdaResources::~BdaResources() {
    // APS5_BDA_CLOCK=1: the shaders' lookup and invocation cycles (BdaAbi::ClockLookupWord), summed
    // as the buffers are released (after the work that used them) and reported every 5 s.
    static const bool clock = std::getenv("APS5_BDA_CLOCK") != nullptr;
    if (clock && fault != nullptr) {
        namespace Abi = ShaderRecompiler::BdaAbi;
        // Split by dispatch and draw (SetCompute): only the dispatches' (Phase A, GPU-bound) share
        // can turn into frame time, since the draws' phase is bound by the committer on the CPU.
        struct Sums {
            std::atomic<std::uint64_t> lookup{0}, total{0}, calls{0};
        };
        static Sums sums[2];
        static std::atomic<std::int64_t> reportAt{0};
        auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
        auto& into = sums[compute ? 1 : 0];
        into.lookup.fetch_add(words[Abi::ClockLookupWord], std::memory_order_relaxed);
        into.total.fetch_add(words[Abi::ClockTotalWord], std::memory_order_relaxed);
        into.calls.fetch_add(words[Abi::ClockCallsWord], std::memory_order_relaxed);
        words[Abi::ClockLookupWord] = words[Abi::ClockTotalWord] = words[Abi::ClockCallsWord] = 0;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto due = reportAt.load(std::memory_order_relaxed);
        if (now >= due && reportAt.compare_exchange_strong(due, now + 5'000'000'000ll)) {
            std::uint64_t l[2], t[2], c[2];
            for (int kind = 0; kind < 2; ++kind) {
                l[kind] = sums[kind].lookup.exchange(0);
                t[kind] = sums[kind].total.exchange(0);
                c[kind] = sums[kind].calls.exchange(0);
            }
            const auto share = [](std::uint64_t part, std::uint64_t whole) { return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0; };
            std::fprintf(stderr, "[bda-clock] lookup %.1f%% of subgroup cycles (%llu of %llu x64), %llu lookups; compute %.1f%% (%llu lookups) / graphics %.1f%% (%llu lookups)\n", share(l[0] + l[1], t[0] + t[1]), static_cast<unsigned long long>(l[0] + l[1]), static_cast<unsigned long long>(t[0] + t[1]), static_cast<unsigned long long>(c[0] + c[1]), share(l[1], t[1]), static_cast<unsigned long long>(c[1]), share(l[0], t[0]), static_cast<unsigned long long>(c[0]));
        }
    }
    if (fault == nullptr || !FaultBufferPoolEnabled() || (mayWrite && !checked)) return;
    const auto* words = reinterpret_cast<const std::uint32_t*>(fault->Bytes().data());
    if (words[0] != 0 || words[ShaderRecompiler::BdaAbi::WrittenOverflowWord] != 0) return;
    auto& pool = FaultBuffers();
    std::lock_guard lock(pool.mutex);
    if (pool.device != device) {
        for (auto& stale : pool.free) static_cast<void>(stale.release());
        pool.free.clear();
        pool.device = device;
    }
    if (pool.free.size() < 8192) pool.free.push_back(std::move(fault));
}

BdaResources::BdaResources(const Context& context, const GuestBufferMemory& memory) : BdaResources(context) {
    const bool deviceLocal = (LivePerformanceControls().Get() & PerformanceControls::BdaTableDeviceLocal) != 0;
    // Read once: the table's size and its index must come from the same value, and a table whose
    // index words disagree with its size fails every lookup (InvalidTable).
    const bool dense = BdaDenseBuckets();
    const auto cached = memory.CachedAddressTable();
    std::vector<ShaderRecompiler::BdaAbi::Range> built;
    if (!cached.has_value()) built = memory.AddressRanges();
    const auto& ranges = cached.has_value() ? *cached->ranges : built;
    const auto serial = cached.has_value() ? cached->serial : 0;
    // The GPU table joins neighbours that continue each other on the device as well (same delta,
    // permissions and flags): one entry then covers what the shader's one-range lookup cache would
    // otherwise miss on at each boundary. The table caches still compare the unjoined ranges.
    std::vector<ShaderRecompiler::BdaAbi::Range> joined;
    static const bool joining = std::getenv("APS5_NO_BDA_MERGE") == nullptr;
    if (joining) {
        joined.reserve(ranges.size());
        for (const auto& range : ranges) {
            if (!joined.empty()) {
                auto& last = joined.back();
                if (last.end == range.begin && last.permissions == range.permissions && last.reserved == range.reserved && range.deviceAddress - last.deviceAddress == range.begin - last.begin) {
                    last.end = range.end;
                    continue;
                }
            }
            joined.push_back(range);
        }
    }
    const auto& tableRanges = joining ? joined : ranges;
    Require(ShaderRecompiler::BdaAbi::RangesOrdered(tableRanges.data(), tableRanges.size()), "BDA table ranges are not ascending and disjoint");
    Require(ranges.size() <= std::numeric_limits<std::uint32_t>::max(), "BDA table range count overflow");
    Require(ranges.size() <= (std::numeric_limits<std::size_t>::max() - sizeof(ShaderRecompiler::BdaAbi::Header)) / sizeof(ShaderRecompiler::BdaAbi::Range), "BDA table size overflow");
    const auto window = BdaBucketShape(tableRanges, BdaBucketLimit(), dense);
    tableBytes = sizeof(ShaderRecompiler::BdaAbi::Header) + tableRanges.size() * sizeof(ShaderRecompiler::BdaAbi::Range) + bucketBytes(window);
    Require(tableBytes <= context.limits.maxStorageBufferRange && ShaderRecompiler::BdaAbi::FaultBufferBytes <= context.limits.maxStorageBufferRange, "BDA descriptors exceed storage buffer range limit");
    auto& cache = Tables();
    std::uint64_t hash = 0;
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        // Another device's tables are not this one's (their buffers are dead or foreign).
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        if (serial != 0) {
            for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
                if (it->spaceSerial != serial || it->deviceLocal != deviceLocal || it->dense != window.windowed) continue;
                if (auto shared = it->buffer.lock(); shared != nullptr) {
                    ++cache.hits;
                    ++cache.classes.spaceTables;
                    table = std::move(shared);
                    cache.entries.splice(cache.entries.begin(), cache.entries, it);
                    return;
                }
                cache.entries.erase(it);
                break;
            }
        }
        hash = hashRanges(ranges);
        classifyFirstEntry(cache, hash, ranges);
        for (auto it = cache.entries.begin(); it != cache.entries.end();) {
            // Compare the hash before taking a strong reference: locking a non-matching entry's
            // buffer could make this thread its last owner (a concurrent reap releasing it) and
            // run the pool release under the cache mutex; expired() takes no reference.
            if (it->hash != hash || it->deviceLocal != deviceLocal || it->dense != window.windowed) {
                if (it->buffer.expired()) {
                    it = cache.entries.erase(it);
                } else {
                    ++it;
                }
                continue;
            }
            auto shared = it->buffer.lock();
            if (shared == nullptr) {
                // Every build that owned the table is gone, and its buffer with it.
                it = cache.entries.erase(it);
                continue;
            }
            if (sameRanges(it->ranges, ranges)) {
                ++cache.hits;
                // An equal table of an earlier space (a rebuild that mapped the same ranges the
                // same way): the next build of this space binds it by the serial.
                if (serial != 0) it->spaceSerial = serial;
                table = std::move(shared);
                cache.entries.splice(cache.entries.begin(), cache.entries, it);
                return;
            }
            ++it;
        }
    }
    const auto buckets = buildBuckets(tableRanges, window);
    countBuckets(tableRanges, window, buckets, dense);
    static std::atomic<int> tablesLogged{0};
    if (std::getenv("APS5_TRACE_BDA_TABLES") != nullptr && tablesLogged.fetch_add(1, std::memory_order_relaxed) < 40) {
        std::uint64_t largest = 0, total = 0;
        for (const auto& range : ranges) {
            largest = std::max(largest, range.end - range.begin);
            total += range.end - range.begin;
        }
        std::size_t contiguous = 0, sameDeltaGaps = 0;
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            const auto& a = ranges[i - 1];
            const auto& b = ranges[i];
            const bool consistent = b.deviceAddress - a.deviceAddress == b.begin - a.begin && a.permissions == b.permissions && a.reserved == b.reserved;
            if (consistent && a.end == b.begin) ++contiguous;
            else if (consistent && a.end < b.begin) ++sameDeltaGaps;
        }
        std::fprintf(stderr, "[bda-table] %zu joined entries from %zu ranges, %llu MiB in total, largest %llu MiB; %zu adjacent pairs contiguous on the device, %zu more with a gap but the same delta\n", tableRanges.size(), ranges.size(), static_cast<unsigned long long>(total >> 20u), static_cast<unsigned long long>(largest >> 20u), contiguous, sameDeltaGaps);
    }
    table = makeTableBuffer(context, tableBytes, deviceLocal);
    const ShaderRecompiler::BdaAbi::Header header{ShaderRecompiler::BdaAbi::Version, static_cast<std::uint32_t>(tableRanges.size()), sizeof(ShaderRecompiler::BdaAbi::Range), static_cast<std::uint32_t>(buckets.bounds.size() / 2u)};
    std::memcpy(table->Bytes().data(), &header, sizeof(header));
    if (!tableRanges.empty()) std::memcpy(table->Bytes().data() + sizeof(header), tableRanges.data(), tableRanges.size() * sizeof(tableRanges.front()));
    if (!buckets.bounds.empty()) {
        auto* index = table->Bytes().data() + sizeof(header) + tableRanges.size() * sizeof(tableRanges.front());
        std::memcpy(index, &buckets.header, sizeof(buckets.header));
        std::memcpy(index + sizeof(buckets.header), buckets.bounds.data(), buckets.bounds.size() * sizeof(std::uint32_t));
    }
    if (tableCacheEnabled()) {
        std::lock_guard lock(cache.mutex);
        ++cache.misses;
        if (cache.device != context.device) {
            cache.entries.clear();
            cache.device = context.device;
        }
        cache.entries.push_front({hash, cached.has_value() ? ranges : std::move(built), table, serial, deviceLocal, window.windowed});
        while (cache.entries.size() > tableCacheEntries()) cache.entries.pop_back();
    }
}

BdaResources::TableCacheStats BdaResources::TableCacheCounters() {
    auto& cache = Tables();
    std::lock_guard lock(cache.mutex);
    auto stats = cache.classes;
    stats.hits = cache.hits;
    stats.misses = cache.misses;
    stats.held = cache.entries.size();
    return stats;
}

VkDescriptorBufferInfo BdaResources::Table() const {
    Require(table != nullptr, "BDA page table was not requested");
    return {table->Handle(), 0, tableBytes};
}

std::unique_ptr<BdaResources> BdaResources::ShareTable(const Context& context) const {
    auto shared = std::make_unique<BdaResources>(context);
    shared->table = table;
    shared->tableBytes = tableBytes;
    shared->mayWrite = mayWrite;
    shared->compute = compute;
    return shared;
}

VkDescriptorBufferInfo BdaResources::Fault() const {
    return {fault->Handle(), 0, ShaderRecompiler::BdaAbi::FaultBufferBytes};
}

namespace {
std::atomic<bool> loopGuardTripped{false};
}

bool LoopGuardTripped() {
    return loopGuardTripped.load(std::memory_order_relaxed);
}

void BdaResources::CheckFault() const {
    checked = true;
    markWrittenPages();
    ShaderRecompiler::BdaAbi::Fault report{};
    std::memcpy(&report, fault->Bytes().data(), sizeof(report));
    if (report.state == ShaderRecompiler::BdaAbi::FaultState::Empty) {
        Require(static_cast<std::uint32_t>(report.reason) == 0 && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.reserved == 0, "BDA fault record has data without publication");
        return;
    }
    Require(report.state == ShaderRecompiler::BdaAbi::FaultState::Ready && report.reserved == 0, "incomplete or invalid BDA fault record");
    Require(report.reason != ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, "rect-list requires finite nondegenerate axis-aligned positions with equal positive W");
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::LoopLimit) {
        // APS5_LOOP_GUARD: the shader left a loop that ran past the guard; the dispatch result is kept.
        loopGuardTripped.store(true, std::memory_order_relaxed);
        std::fprintf(stderr, "[gpu] loop guard: the loop exit at pc 0x%x of shader 0x%llx ran past %u evaluations\n", report.instruction, static_cast<unsigned long long>(report.address), report.bytes);
        std::memset(fault->Bytes().data(), 0, fault->Bytes().size());
        return;
    }
    std::ostringstream message;
    message << "BDA access failed: address=0x" << std::hex << report.address << " instruction=0x" << report.instruction << std::dec << " bytes=" << report.bytes << " stage=" << report.stage << " reason=" << static_cast<std::uint32_t>(report.reason);
    if (report.reason == ShaderRecompiler::BdaAbi::FaultReason::Permission && table != nullptr) {
        const auto bytes = table->Bytes();
        ShaderRecompiler::BdaAbi::Header header{};
        std::memcpy(&header, bytes.data(), sizeof(header));
        for (std::uint32_t index = 0; index < header.count; ++index) {
            ShaderRecompiler::BdaAbi::Range range{};
            std::memcpy(&range, bytes.data() + sizeof(header) + index * sizeof(range), sizeof(range));
            if (report.address < range.begin || report.address >= range.end) continue;
            message << std::hex << "; the store hit 0x" << range.begin << "+0x" << range.end - range.begin << ", read-only in the BDA table: stores through GPU-selected descriptors and FLAT/GLOBAL addresses reach only writable ranges imported in place, not ones served by a mirror or a copy (past APS5_HOST_IMPORT_MIB, or refused by the driver)";
            break;
        }
    }
    throw std::runtime_error(message.str());
}

}

namespace AgcDriver::Graphics {

void BdaResources::markWrittenPages() const {
    static const bool alwaysScan = std::getenv("APS5_SCAN_BDA_WRITES") != nullptr;
    if (!mayWrite && !alwaysScan) return;
    namespace Abi = ShaderRecompiler::BdaAbi;
    auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
    Require(words[Abi::WrittenOverflowWord] == 0, "more than " + std::to_string(Abi::WrittenPageSlots) + " pages stored to through the BDA table in one use are not implemented");
    bool any = false;
    for (std::uint32_t slot = 0; slot < Abi::WrittenPageSlots; ++slot) {
        const auto page = words[Abi::WrittenSlotsWord + slot];
        if (page == 0) continue;
        GuestMemory::MarkWritten(static_cast<std::uint64_t>(page - 1u) << Abi::WrittenPageShift, std::size_t{1} << Abi::WrittenPageShift);
        any = true;
    }
    if (any) std::memset(words + Abi::WrittenSlotsWord, 0, Abi::WrittenPageSlots * sizeof(std::uint32_t));
}

}
