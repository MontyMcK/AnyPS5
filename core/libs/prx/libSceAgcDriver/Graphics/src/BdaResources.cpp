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

// APS5_BDA_BUCKETS=N caps the lookup index at N buckets (0: no index, every lookup searches all ranges).
std::size_t bucketLimit() {
    static const std::size_t limit = [] {
        const char* text = std::getenv("APS5_BDA_BUCKETS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{4096};
    }();
    return limit;
}

struct BucketIndex {
    ShaderRecompiler::BdaAbi::BucketHeader header{};
    std::vector<std::uint32_t> bounds;
};

std::pair<std::uint32_t, std::size_t> bucketShape(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    const auto limit = bucketLimit();
    if (limit == 0 || ranges.size() < 16) return {0, 0};
    const auto first = ranges.front().begin;
    const auto last = ranges.back().end;
    std::uint32_t shift = 12;
    const auto bucketsFor = [&](std::uint32_t bits) { return (last - (first & ~((std::uint64_t{1} << bits) - 1u)) + (std::uint64_t{1} << bits) - 1u) >> bits; };
    while (shift < 48 && bucketsFor(shift) > limit) ++shift;
    const auto count = bucketsFor(shift);
    if (count == 0 || count > limit) return {0, 0};
    return {shift, static_cast<std::size_t>(count)};
}

std::size_t bucketBytes(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    const auto count = bucketShape(ranges).second;
    return count == 0 ? 0 : sizeof(ShaderRecompiler::BdaAbi::BucketHeader) + count * 2u * sizeof(std::uint32_t);
}

BucketIndex buildBuckets(const std::vector<ShaderRecompiler::BdaAbi::Range>& ranges) {
    BucketIndex index;
    const auto [shift, count] = bucketShape(ranges);
    if (count == 0) return index;
    index.header.base = ranges.front().begin & ~((std::uint64_t{1} << shift) - 1u);
    index.header.shift = shift;
    index.bounds.resize(count * 2u);
    std::size_t low = 0;
    std::size_t high = 0;
    for (std::size_t bucket = 0; bucket < count; ++bucket) {
        const auto start = index.header.base + (static_cast<std::uint64_t>(bucket) << shift);
        const auto stop = start + (std::uint64_t{1} << shift);
        while (low < ranges.size() && ranges[low].end <= start) ++low;
        while (high < ranges.size() && ranges[high].begin < stop) ++high;
        index.bounds[bucket * 2u] = static_cast<std::uint32_t>(low);
        index.bounds[bucket * 2u + 1u] = static_cast<std::uint32_t>(high);
    }
    return index;
}

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
        static std::atomic<std::uint64_t> lookup{0}, total{0}, calls{0};
        static std::atomic<std::int64_t> reportAt{0};
        auto* words = reinterpret_cast<std::uint32_t*>(fault->Bytes().data());
        lookup.fetch_add(words[Abi::ClockLookupWord], std::memory_order_relaxed);
        total.fetch_add(words[Abi::ClockTotalWord], std::memory_order_relaxed);
        calls.fetch_add(words[Abi::ClockCallsWord], std::memory_order_relaxed);
        words[Abi::ClockLookupWord] = words[Abi::ClockTotalWord] = words[Abi::ClockCallsWord] = 0;
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto due = reportAt.load(std::memory_order_relaxed);
        if (now >= due && reportAt.compare_exchange_strong(due, now + 5'000'000'000ll)) {
            const auto l = lookup.exchange(0), t = total.exchange(0), c = calls.exchange(0);
            std::fprintf(stderr, "[bda-clock] lookup %.1f%% of subgroup cycles (%llu of %llu x64), %llu lookups\n", t != 0 ? 100.0 * static_cast<double>(l) / static_cast<double>(t) : 0.0, static_cast<unsigned long long>(l), static_cast<unsigned long long>(t), static_cast<unsigned long long>(c));
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
    Require(ranges.size() <= std::numeric_limits<std::uint32_t>::max(), "BDA table range count overflow");
    Require(ranges.size() <= (std::numeric_limits<std::size_t>::max() - sizeof(ShaderRecompiler::BdaAbi::Header)) / sizeof(ShaderRecompiler::BdaAbi::Range), "BDA table size overflow");
    tableBytes = sizeof(ShaderRecompiler::BdaAbi::Header) + tableRanges.size() * sizeof(ShaderRecompiler::BdaAbi::Range) + bucketBytes(tableRanges);
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
                if (it->spaceSerial != serial || it->deviceLocal != deviceLocal) continue;
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
            if (it->hash != hash || it->deviceLocal != deviceLocal) {
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
    const auto buckets = buildBuckets(tableRanges);
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
        cache.entries.push_front({hash, cached.has_value() ? ranges : std::move(built), table, serial, deviceLocal});
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
