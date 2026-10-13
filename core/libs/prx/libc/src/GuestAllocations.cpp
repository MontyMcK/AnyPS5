#include "prx/libc/include/GuestAllocations.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <limits>
#include <iterator>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>
#include <vector>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace GuestAllocations {
namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::uint64_t, std::shared_ptr<const Range>> ranges;
    bool mainImageRegistered = false;
    // The readable ranges as a lease snapshot (see Lease), built by the first Acquire after a change
    // and dropped when a mutation begins, before it checks pins.
    std::shared_ptr<const Lease::Items> snapshot;
};

Registry& registry() {
    static Registry value;
    return value;
}

std::atomic<std::uint64_t> generation{1};

// [guest-allocations] (APS5_PROFILE_COLLECT or APS5_PROFILE_DRAW): registry mutations per 10 s by
// kind, to tell which ones move the generation (every End and Invalidate does) during a scene.
struct MutationProfile {
    std::atomic<std::uint64_t> ends{0}, emptyEnds{0}, addReadable{0}, addHidden{0}, removeReadable{0}, removeHidden{0}, protectReadable{0}, protectHidden{0}, unmapReadable{0}, unmapHidden{0}, invalidates{0}, invalidateBytes{0};
    // Protects whose every registered piece already had the requested readable/writable flags.
    std::atomic<std::uint64_t> protectSameFlags{0}, protectSameFlagsPinned{0};
    std::atomic<std::int64_t> reported{0};
};

bool mutationProfiled() {
    static const bool enabled = std::getenv("APS5_PROFILE_COLLECT") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return enabled;
}

MutationProfile& mutationProfile() {
    static MutationProfile profile;
    return profile;
}

void countMutation(std::atomic<std::uint64_t> MutationProfile::*counter) {
    if (mutationProfiled()) (mutationProfile().*counter).fetch_add(1, std::memory_order_relaxed);
}

void reportMutations() {
    if (!mutationProfiled()) return;
    auto& profile = mutationProfile();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = profile.reported.load(std::memory_order_relaxed);
    if (last == 0) {
        profile.reported.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 10'000'000'000ll || !profile.reported.compare_exchange_strong(last, now)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto ends = take(profile.ends);
    const auto empty = take(profile.emptyEnds);
    const auto addR = take(profile.addReadable);
    const auto addH = take(profile.addHidden);
    const auto remR = take(profile.removeReadable);
    const auto remH = take(profile.removeHidden);
    const auto proR = take(profile.protectReadable);
    const auto proH = take(profile.protectHidden);
    const auto unR = take(profile.unmapReadable);
    const auto unH = take(profile.unmapHidden);
    const auto inv = take(profile.invalidates);
    const auto invBytes = take(profile.invalidateBytes);
    const auto same = take(profile.protectSameFlags);
    const auto samePinned = take(profile.protectSameFlagsPinned);
    std::fprintf(stderr, "[guest-allocations] (10 s) mutations %llu (%llu changed nothing); add readable %llu / not %llu, remove readable %llu / not %llu, protect to readable %llu / not %llu, unmap readable %llu / not %llu; invalidates %llu (%.1f MiB); protects that keep every flag %llu (%llu of them over a range GPU work held)\n", ends, empty, addR, addH, remR, remH, proR, proH, unR, unH, inv, static_cast<double>(invBytes) / 1048576.0, same, samePinned);
}
std::atomic<void (*)(std::uintptr_t, std::size_t)> invalidator{nullptr};
std::atomic<bool (*)()> pinWaiter{nullptr};
// Whether same-flag protections keep the registry as it is (APS5_KEEP_SAME_PROTECT, or the driver's
// live flag through GuestAllocationsSetKeepSameQuery).
std::atomic<bool (*)()> keepSameQuery{nullptr};

// Whether leases share the registry's snapshot (see Lease); on unless the driver's live flag
// (GuestAllocationsSetSnapshotQuery) or APS5_NO_LEASE_SNAPSHOT turns it off, for A/B runs.
std::atomic<bool (*)()> snapshotQuery{nullptr};

bool leaseSnapshots() {
    static const bool environment = std::getenv("APS5_NO_LEASE_SNAPSHOT") == nullptr;
    if (const auto query = snapshotQuery.load(std::memory_order_acquire)) return query();
    return environment;
}

bool keepSameProtect() {
    static const bool environment = std::getenv("APS5_KEEP_SAME_PROTECT") != nullptr;
    if (const auto query = keepSameQuery.load(std::memory_order_acquire)) return query();
    return environment;
}

std::chrono::milliseconds pinWait() {
    static const std::chrono::milliseconds value{[] {
        const char* text = std::getenv("APS5_PIN_WAIT_MS");
        return text != nullptr ? std::strtoull(text, nullptr, 10) : 60000ull;
    }()};
    return value;
}

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

// A mutation holds the registry lock and remembers the ranges it changed, which are reported to the
// invalidator once the generation has moved on, so a cache that observed the old state meanwhile
// notices either way.
struct MutationState {
    std::unique_lock<std::mutex> lock;
    std::vector<std::pair<std::uintptr_t, std::size_t>> changed;
    // The registry was modified (an entry added, erased or split). A mutation that only applied
    // protections the registry already records (APS5_KEEP_SAME_PROTECT) leaves it false, and its
    // End moves no generation: nothing a lease, an address space or an import was built from changed.
    bool modified = false;
};

void recordChange(void* mutation, const void* pointer, std::size_t bytes) {
    if (mutation == nullptr) return;
    static_cast<MutationState*>(mutation)->modified = true;
    if (bytes != 0) static_cast<MutationState*>(mutation)->changed.emplace_back(reinterpret_cast<std::uintptr_t>(pointer), bytes);
}

#ifndef _WIN32
std::vector<std::pair<std::uintptr_t, std::size_t>> fileBackedWritableImage() {
    const auto image = std::filesystem::read_symlink("/proc/self/exe").string();
    std::ifstream maps("/proc/self/maps");
    std::vector<std::pair<std::uintptr_t, std::size_t>> mappings;
    for (std::string line; std::getline(maps, line);) {
        std::istringstream fields(line);
        std::string span, permissions, offset, device, inode, path;
        fields >> span >> permissions >> offset >> device >> inode >> std::ws;
        std::getline(fields, path);
        if (permissions != "rw-p" || path != image) continue;
        const auto dash = span.find('-');
        const auto begin = std::stoull(span.substr(0, dash), nullptr, 16);
        mappings.emplace_back(begin, std::stoull(span.substr(dash + 1), nullptr, 16) - begin);
    }
    return mappings;
}

bool backWritableImageAnonymously() {
    for (const auto& [address, bytes] : fileBackedWritableImage()) {
        auto* image = reinterpret_cast<void*>(address);
        auto* copy = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (copy == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "mmap of an anonymous main image copy failed");
        std::memcpy(copy, image, bytes);
        if (mremap(copy, bytes, bytes, MREMAP_MAYMOVE | MREMAP_FIXED, image) == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "mremap of an anonymous main image copy failed");
    }
    return true;
}

[[maybe_unused]] const bool writableImageAnonymous = backWritableImageAnonymously();
#endif

}

void* GuestAllocationsBegin_nid_postfix() {
    auto* state = new MutationState{std::unique_lock<std::mutex>(registry().mutex), {}};
    // The registry's own snapshot reference would pin every range it holds.
    registry().snapshot.reset();
    return state;
}

void GuestAllocationsEnd_nid_postfix(void* mutation) noexcept {
    auto* state = static_cast<MutationState*>(mutation);
    countMutation(&MutationProfile::ends);
    if (state->changed.empty()) countMutation(&MutationProfile::emptyEnds);
    reportMutations();
    if (!state->modified && keepSameProtect()) {
        delete state;
        return;
    }
    generation.fetch_add(1, std::memory_order_release);
    if (const auto callback = invalidator.load(std::memory_order_acquire)) {
        for (const auto& [address, bytes] : state->changed) callback(address, bytes);
    }
    delete state;
}

std::uint64_t GuestAllocationsGeneration_nid_postfix() {
    return generation.load(std::memory_order_acquire);
}

void GuestAllocationsSetInvalidator_nid_postfix(void (*callback)(std::uintptr_t, std::size_t)) {
    invalidator.store(callback, std::memory_order_release);
}

void GuestAllocationsInvalidate_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    countMutation(&MutationProfile::invalidates);
    if (mutationProfiled()) mutationProfile().invalidateBytes.fetch_add(bytes, std::memory_order_relaxed);
    generation.fetch_add(1, std::memory_order_release);
    if (const auto callback = invalidator.load(std::memory_order_acquire)) callback(address, bytes);
}

void GuestAllocationsSetPinWaiter_nid_postfix(bool (*callback)()) {
    pinWaiter.store(callback, std::memory_order_release);
}

#ifdef _WIN32
void GuestAllocationsRegisterMainImage_nid_postfix(void* mutation) {
    auto& state = registry();
    if (state.mainImageRegistered) return;
    const auto image = GetModuleHandleW(nullptr);
    require(image != nullptr, "cannot locate the main guest image");
    auto replacement = state.ranges;
    auto cursor = reinterpret_cast<std::uintptr_t>(image);
    bool registered = false;
    for (;;) {
        MEMORY_BASIC_INFORMATION memory{};
        require(VirtualQuery(reinterpret_cast<const void*>(cursor), &memory, sizeof(memory)) == sizeof(memory), "cannot query the main guest image");
        if (memory.AllocationBase != image) break;
        require(memory.Type == MEM_IMAGE && (memory.State == MEM_COMMIT || memory.State == MEM_RESERVE), "unsupported guest image mapping");
        require(reinterpret_cast<std::uintptr_t>(memory.BaseAddress) == cursor && memory.RegionSize != 0 && memory.RegionSize <= std::numeric_limits<std::uintptr_t>::max() - cursor, "invalid guest image range");
        if (memory.State == MEM_COMMIT) {
            require((memory.Protect & PAGE_GUARD) == 0, "guarded guest image pages are not supported");
            const auto protection = memory.Protect & 0xffu;
            const bool writable = protection == PAGE_READWRITE || protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
            const bool readable = writable || protection == PAGE_READONLY || protection == PAGE_EXECUTE_READ;
            require(readable || protection == PAGE_NOACCESS || protection == PAGE_EXECUTE, "unsupported guest image protection");
            const auto next = replacement.lower_bound(cursor);
            require(next == replacement.end() || cursor + memory.RegionSize <= next->first, "guest image overlaps a registered allocation");
            if (next != replacement.begin()) {
                const auto& previous = *std::prev(next)->second;
                require(previous.address + previous.bytes <= cursor, "guest image overlaps a registered allocation");
            }
            replacement.emplace(cursor, std::make_shared<const Range>(Range{cursor, memory.RegionSize, readable, writable, cursor, memory.RegionSize, false}));
            registered = true;
        }
        cursor += memory.RegionSize;
    }
    require(registered, "main guest image has no committed pages");
    state.ranges.swap(replacement);
    state.mainImageRegistered = true;
    if (mutation != nullptr) static_cast<MutationState*>(mutation)->modified = true;
}
#else
void GuestAllocationsRegisterMainImage_nid_postfix(void* mutation) {
    auto& state = registry();
    if (state.mainImageRegistered) return;
    struct Page {
        bool readable = false;
        bool writable = false;
    };
    std::map<std::uint64_t, Page> pages;
    const auto pageSize = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    std::pair<std::map<std::uint64_t, Page>*, std::uint64_t> collection{&pages, pageSize};
    dl_iterate_phdr([](dl_phdr_info* image, std::size_t, void* data) {
        auto& collected = *static_cast<std::pair<std::map<std::uint64_t, Page>*, std::uint64_t>*>(data);
        for (int index = 0; index < image->dlpi_phnum; ++index) {
            const auto& header = image->dlpi_phdr[index];
            if (header.p_type != PT_LOAD || header.p_memsz == 0) continue;
            const auto start = (image->dlpi_addr + header.p_vaddr) & ~(collected.second - 1);
            const auto end = (image->dlpi_addr + header.p_vaddr + header.p_memsz + collected.second - 1) & ~(collected.second - 1);
            for (auto page = start; page < end; page += collected.second) {
                auto& entry = (*collected.first)[page];
                entry.readable = entry.readable || (header.p_flags & (PF_R | PF_W)) != 0;
                entry.writable = entry.writable || (header.p_flags & PF_W) != 0;
            }
        }
        return 1;
    }, &collection);
    require(!pages.empty(), "main guest image has no loadable segments");
    auto replacement = state.ranges;
    for (auto page = pages.begin(); page != pages.end();) {
        auto last = page;
        while (std::next(last) != pages.end() && std::next(last)->first == last->first + pageSize && std::next(last)->second.readable == page->second.readable && std::next(last)->second.writable == page->second.writable) ++last;
        const auto address = page->first;
        const auto bytes = static_cast<std::size_t>(last->first + pageSize - address);
        const auto next = replacement.lower_bound(address);
        require(next == replacement.end() || address + bytes <= next->first, "guest image overlaps a registered allocation");
        if (next != replacement.begin()) {
            const auto& previous = *std::prev(next)->second;
            require(previous.address + previous.bytes <= address, "guest image overlaps a registered allocation");
        }
        replacement.emplace(address, std::make_shared<const Range>(Range{address, bytes, page->second.readable, page->second.writable, address, bytes, false}));
        page = std::next(last);
    }
    state.ranges.swap(replacement);
    state.mainImageRegistered = true;
    if (mutation != nullptr) static_cast<MutationState*>(mutation)->modified = true;
}
#endif

void GuestAllocationsAdd_nid_postfix(void* mutation, void* pointer, std::size_t bytes, bool readable, bool writable) {
    countMutation(readable ? &MutationProfile::addReadable : &MutationProfile::addHidden);
    recordChange(mutation, pointer, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(address != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest allocation range");
    require(!writable || readable, "writable guest allocation must be readable");
    auto& ranges = registry().ranges;
    const auto next = ranges.lower_bound(address);
    require(next == ranges.end() || (next->first != address && address + bytes <= next->first), "overlapping guest allocation");
    if (next != ranges.begin()) {
        const auto& previous = *std::prev(next)->second;
        require(previous.address + previous.bytes <= address, "overlapping guest allocation");
    }
    ranges.emplace(address, std::make_shared<const Range>(Range{address, bytes, readable, writable, address, bytes}));
}

[[noreturn]] void PinnedFailure(std::uintptr_t address, std::size_t bytes, const char* why) {
    char message[160];
    std::snprintf(message, sizeof(message), "guest allocation 0x%llx+0x%llx is owned by an active GPU command (%s)", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), why);
    throw std::runtime_error(message);
}

void GuestAllocationsRequireUnpinned_nid_postfix(void* mutation, const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(bytes <= std::numeric_limits<std::uint64_t>::max() - address, "guest allocation range overflow");
    const auto end = address + bytes;
    auto& ranges = registry().ranges;
    // GPU work leases allocations (a lease copies the shared_ptr) until the work completed, so wait
    // for it rather than failing the guest. The wait releases the registry lock: the driver's waiter
    // finishes that work under the device lock, under which the driver's workers acquire leases from
    // this registry, so waiting with the lock held would deadlock them. The scan restarts after the
    // lock is retaken, since another mutation may have run meanwhile. Without a waiter the lease is
    // dropped by another thread (a synchronous draw or dispatch), so yielding suffices. A waiter
    auto* state = static_cast<MutationState*>(mutation);
    const auto deadline = std::chrono::steady_clock::now() + pinWait();
    int syncedRounds = 0;
    for (;;) {
        // A lease taken while the lock was released below rebuilt the registry's snapshot, whose own
        // reference is no pin.
        registry().snapshot.reset();
        bool pinned = false;
        auto it = ranges.upper_bound(address);
        if (it != ranges.begin()) --it;
        for (; it != ranges.end(); ++it) {
            const auto& [base, range] = *it;
            if (base >= end && base != address) break;
            if (((address < base + range->bytes && base < end) || base == address) && range.use_count() != 1) {
                pinned = true;
                break;
            }
        }
        if (!pinned) return;
        if (std::chrono::steady_clock::now() >= deadline) PinnedFailure(address, bytes, ("waited " + std::to_string(pinWait().count()) + " ms for the lease (APS5_PIN_WAIT_MS)").c_str());
        const auto waiter = pinWaiter.load(std::memory_order_acquire);
        bool progressed = false;
        if (state != nullptr && state->lock.owns_lock()) {
            state->lock.unlock();
            if (waiter != nullptr) progressed = waiter();
            else std::this_thread::yield();
            state->lock.lock();
        } else {
            std::this_thread::yield();
        }
        if (progressed) {
            // The lease's work finished, yet the range is pinned again: between the waiter's return
            // and the re-lock a driver worker took a new lease (every address-based build pins every
            // registered range). Rare per round, but each round is a GPU wait, so a repeat is reported.
            if (++syncedRounds == 8) std::fprintf(stderr, "[gpu] leased allocation 0x%llx+0x%llx still pinned after %d recorder syncs\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), syncedRounds);
            continue;
        }
    }
}

void GuestAllocationsRequireAvailable_nid_postfix(void*, const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(address != 0 && bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid fixed guest mapping");
    for (const auto& [base, range] : registry().ranges) {
        if (base >= address + bytes) break;
        if (base + range->bytes > address) {
            char message[160];
            std::snprintf(message, sizeof(message), "fixed mapping 0x%llx+0x%llx overlaps a registered guest allocation 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(base), static_cast<unsigned long long>(range->bytes));
            throw std::runtime_error(message);
        }
    }
}

bool GuestAllocationsOverlaps_nid_postfix(void*, const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(address != 0 && bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest allocation range");
    for (const auto& [base, range] : registry().ranges) {
        if (base >= address + bytes) break;
        if (base + range->bytes > address) return true;
    }
    return false;
}

bool GuestAllocationsCovers_nid_postfix(void*, const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(address != 0 && bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest allocation range");
    const auto end = address + bytes;
    auto cursor = address;
    for (const auto& [base, range] : registry().ranges) {
        const auto finish = base + range->bytes;
        if (finish <= cursor) continue;
        if (base > cursor || !range->releasable) return false;
        cursor = finish;
        if (cursor >= end) return true;
    }
    return false;
}

Range GuestAllocationsFind_nid_postfix(void*, const void* pointer) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto exact = registry().ranges.find(address);
    if (exact != registry().ranges.end() && exact->second->allocationAddress == address && exact->second->allocationBytes == exact->second->bytes) {
        const auto& range = *exact->second;
        require(range.releasable, "guest image memory is not a releasable allocation");
        return {address, range.allocationBytes, range.readable, range.writable, address, range.allocationBytes, range.releasable};
    }
    for (const auto& [base, range] : registry().ranges) {
        if (range->allocationAddress == address) {
            require(range->releasable, "guest image memory is not a releasable allocation");
            return {address, range->allocationBytes, range->readable, range->writable, address, range->allocationBytes, range->releasable};
        }
    }
    throw std::runtime_error("guest allocation is not registered");
}

void GuestAllocationsRemove_nid_postfix(void* mutation, const void* pointer) {
    const auto range = GuestAllocationsFind_nid_postfix(mutation, pointer);
    countMutation(range.readable ? &MutationProfile::removeReadable : &MutationProfile::removeHidden);
    GuestAllocationsRequireUnpinned_nid_postfix(mutation, pointer, range.bytes);
    recordChange(mutation, pointer, range.bytes);
    auto& ranges = registry().ranges;
    const auto exact = ranges.find(range.address);
    if (exact != ranges.end() && exact->second->allocationBytes == exact->second->bytes) {
        ranges.erase(exact);
        return;
    }
    std::erase_if(ranges, [&](const auto& entry) { return entry.second->allocationAddress == range.address; });
}

namespace {

std::map<std::uint64_t, std::shared_ptr<const Range>> replaceRange(const void* pointer, std::size_t bytes, bool remove, bool readable, bool writable) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(bytes != 0 && bytes <= std::numeric_limits<std::uint64_t>::max() - address, "invalid guest protection or unmap range");
    require(!writable || readable, "writable guest allocation must be readable");
    const auto end = address + bytes;
    auto replacement = registry().ranges;
    auto cursor = address;
    for (const auto& [base, entry] : registry().ranges) {
        const auto& range = *entry;
        const auto finish = base + range.bytes;
        if (finish <= address) continue;
        if (base >= end) break;
        require(base <= cursor, "guest protection or unmap range has a hole");
        replacement.erase(base);
        const auto insert = [&](std::uint64_t first, std::uint64_t last, bool canRead, bool canWrite) {
            if (first < last) replacement.emplace(first, std::make_shared<const Range>(Range{first, static_cast<std::size_t>(last - first), canRead, canWrite, range.allocationAddress, range.allocationBytes, range.releasable}));
        };
        insert(base, std::max(base, address), range.readable, range.writable);
        if (!remove) insert(std::max(base, address), std::min(finish, end), readable, writable);
        insert(std::min(finish, end), finish, range.readable, range.writable);
        cursor = std::min(finish, end);
    }
    require(cursor == end, "guest protection or unmap range is not registered");
    return replacement;
}

}

namespace {

// Whether registered entries cover all of [pointer, pointer + bytes) without a hole and every one
// of them already has these flags.
bool rangesCarry(const void* pointer, std::size_t bytes, bool readable, bool writable) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address) return false;
    const auto end = address + bytes;
    auto& ranges = registry().ranges;
    auto it = ranges.upper_bound(address);
    if (it != ranges.begin()) --it;
    auto cursor = address;
    for (; it != ranges.end() && it->first < end; ++it) {
        const auto& range = *it->second;
        if (range.address + range.bytes <= cursor) continue;
        if (range.address > cursor || range.readable != readable || range.writable != writable) return false;
        cursor = range.address + range.bytes;
        if (cursor >= end) return true;
    }
    return false;
}

}

bool GuestAllocationsCarries_nid_postfix(void*, const void* pointer, std::size_t bytes, bool readable, bool writable) {
    return keepSameProtect() && rangesCarry(pointer, bytes, readable, writable);
}

void GuestAllocationsSetKeepSameQuery_nid_postfix(bool (*query)()) {
    keepSameQuery.store(query, std::memory_order_release);
}

void GuestAllocationsSetSnapshotQuery_nid_postfix(bool (*query)()) {
    snapshotQuery.store(query, std::memory_order_release);
}

void GuestAllocationsProtect_nid_postfix(void* mutation, const void* pointer, std::size_t bytes, bool readable, bool writable, const std::function<void()>& apply) {
    countMutation(readable ? &MutationProfile::protectReadable : &MutationProfile::protectHidden);
    if (mutationProfiled()) {
        const auto address = reinterpret_cast<std::uintptr_t>(pointer);
        const auto end = address + bytes;
        bool same = bytes != 0;
        bool pinned = false;
        auto cursor = address;
        auto& ranges = registry().ranges;
        auto it = ranges.upper_bound(address);
        if (it != ranges.begin()) --it;
        for (; it != ranges.end() && it->first < end; ++it) {
            const auto& range = *it->second;
            if (range.address + range.bytes <= address) continue;
            if (range.address > cursor || range.readable != readable || range.writable != writable) same = false;
            if (it->second.use_count() != 1) pinned = true;
            cursor = std::max<std::uintptr_t>(cursor, range.address + range.bytes);
        }
        if (cursor < end) same = false;
        if (same) {
            countMutation(&MutationProfile::protectSameFlags);
            if (pinned) countMutation(&MutationProfile::protectSameFlagsPinned);
        }
    }
    GuestAllocationsRequireUnpinned_nid_postfix(mutation, pointer, bytes);
    recordChange(mutation, pointer, bytes);
    auto replacement = replaceRange(pointer, bytes, false, readable, writable);
    apply();
    registry().ranges.swap(replacement);
}

void GuestAllocationsUnmap_nid_postfix(void* mutation, const void* pointer, std::size_t bytes, const std::function<void(const void*, std::size_t, const void*, bool)>& apply) {
    GuestAllocationsRequireUnpinned_nid_postfix(mutation, pointer, bytes);
    recordChange(mutation, pointer, bytes);
    const auto end = reinterpret_cast<std::uintptr_t>(pointer) + bytes;
    auto cursor = reinterpret_cast<std::uintptr_t>(pointer);
    bool any = false;
    while (cursor < end) {
        const auto found = registry().ranges.upper_bound(cursor);
        const Range* containing = nullptr;
        if (found != registry().ranges.begin()) {
            const auto& candidate = *std::prev(found)->second;
            if (cursor < candidate.allocationAddress + candidate.allocationBytes) containing = &candidate;
        }
        if (containing == nullptr) {
            if (found == registry().ranges.end() || found->first >= end) break;
            cursor = found->first;
            continue;
        }
        const auto range = *containing;
        countMutation(range.readable ? &MutationProfile::unmapReadable : &MutationProfile::unmapHidden);
        require(range.releasable, "guest image memory cannot be unmapped");
        const auto pieceEnd = std::min<std::uint64_t>(end, range.allocationAddress + range.allocationBytes);
        auto replacement = replaceRange(reinterpret_cast<const void*>(cursor), pieceEnd - cursor, true, false, false);
        bool last = true;
        for (const auto& [base, entry] : replacement) {
            if (entry->allocationAddress == range.allocationAddress) last = false;
        }
        apply(reinterpret_cast<const void*>(cursor), pieceEnd - cursor, reinterpret_cast<const void*>(range.allocationAddress), last);
        registry().ranges.swap(replacement);
        any = true;
        cursor = pieceEnd;
    }
    require(any, "unmap address is not registered");
}

Lease GuestAllocationsAcquire_nid_postfix() {
    auto& state = registry();
    std::lock_guard lock(state.mutex);
    if (!leaseSnapshots()) {
        // A private copy per lease, as before snapshots.
        state.snapshot.reset();
        auto items = std::make_shared<Lease::Items>();
        for (const auto& [address, range] : state.ranges) {
            if (range->readable && range->bytes != 0) items->push_back(range);
        }
        return Lease(std::move(items));
    }
    if (state.snapshot == nullptr) {
        auto items = std::make_shared<Lease::Items>();
        for (const auto& [address, range] : state.ranges) {
            if (range->readable && range->bytes != 0) items->push_back(range);
        }
        state.snapshot = std::move(items);
    }
    return Lease(state.snapshot);
}

}
