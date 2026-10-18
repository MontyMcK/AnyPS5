#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// The tracker's live flags, answered by the test: `unchanged_lockfree` and `collect_local_copy`
// from the switches below (-1: the fallback), every other name from its fallback.
namespace AgcDriver::Graphics {

std::atomic<int> testLockFree{-1};
std::atomic<int> testLocalCopy{-1};

bool LiveFlag(std::string_view name, bool fallback) {
    const auto pick = [&](const std::atomic<int>& value) { return value.load() < 0 ? fallback : value.load() != 0; };
    if (name == "unchanged_lockfree") return pick(testLockFree);
    if (name == "collect_local_copy") return pick(testLocalCopy);
    return fallback;
}

bool LockSpinEnabled() {
    return true;
}

bool LiveSwitch::Get(bool fallback) const {
    return LiveFlag(name, fallback);
}

}

namespace {

using namespace AgcDriver::GuestMemory;

constexpr std::size_t Block = 65536;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void* AllocateWatched(std::size_t bytes, bool crossLeaf = false) {
#ifdef _WIN32
    static_cast<void>(crossLeaf);
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    const std::size_t alignment = crossLeaf ? std::size_t{1} << 32 : Block;
    const auto prefix = crossLeaf ? Block : 0;
    void* raw = mmap(nullptr, bytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = ((begin + prefix + alignment - 1) & ~static_cast<std::uintptr_t>(alignment - 1)) - prefix;
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + alignment) munmap(reinterpret_cast<void*>(aligned + bytes), begin + alignment - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    Require(Watched(reinterpret_cast<std::uint64_t>(block), bytes), "the test block is not watched");
    return block;
}

void CheckSharedBlock() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto boundary = base + 2 * Block - 0xb800;
    const auto firstBegin = base;
    const auto secondEnd = base + 3 * Block;
    const auto sharedFirst = boundary - (base + Block);
    const auto sharedSecond = base + 2 * Block - boundary;
    std::memset(memory, 0x11, 3 * Block);

    const auto synced = CollectWrites(base, 3 * Block);
    Require(synced != 0, "the test block is not collected");

    Require(MarkWritten(firstBegin, static_cast<std::size_t>(boundary - firstBegin)) != 0, "a driver store into watched memory is not stamped");
    std::array<std::uint64_t, 2> generations{synced, synced};
    std::array<std::uint8_t, 2> changed{};
    Require(ChangedBlocks(base + Block, 2 * Block, generations, changed) && changed[0] == BlockWritten && changed[1] == BlockUnchanged, "the shared block does not read as written");
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), synced), "a store of the neighbour's bytes counts for the second surface's part of the shared block");
    Require(StoredOver(base + Block, static_cast<std::size_t>(sharedFirst), synced), "the first surface's own store is not seen over its part of the shared block");
    Require(!StoredOver(boundary, static_cast<std::size_t>(secondEnd - boundary), synced), "the second surface reads as stored over");

    const auto touched = MarkWritten(boundary + 0x100, 4);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), synced), "a store over the second surface's bytes is not seen");
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), touched), "a store older than the generation is seen");

    const auto whole = TrackerGeneration();
    MarkWritten(base + Block, Block);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), whole), "a store over the whole block is not seen");
    const auto afterWhole = TrackerGeneration();
    MarkWritten(base + Block, static_cast<std::size_t>(sharedFirst));
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), afterWhole), "a partial store after a whole-block store is taken for the whole block");

    const auto many = TrackerGeneration();
    for (int store = 0; store < 6; ++store) MarkWritten(base + Block + static_cast<std::uint64_t>(store) * 64, 4);
    Require(StoredOver(boundary, static_cast<std::size_t>(sharedSecond), many), "forgotten driver stores are taken as missing the range");

    const auto beforeCpu = CollectWrites(base, 3 * Block);
    static_cast<volatile std::uint8_t*>(memory)[Block + 8] = 0x22;
    CollectWritesUncached(base, 3 * Block);
    Require(!StoredOver(boundary, static_cast<std::size_t>(sharedSecond), beforeCpu), "a CPU store outside the shared range counts for it");

    std::array<std::uint8_t, 64> unwatched{};
    const auto outside = reinterpret_cast<std::uint64_t>(unwatched.data());
    if (!Watched(outside, unwatched.size())) Require(StoredOver(outside, unwatched.size(), TrackerGeneration()), "an unwatched range reads as not stored over");
}

void CheckOwnStore() {
    void* memory = AllocateWatched(2 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto second = base + Block;
    std::memset(memory, 0x11, 2 * Block);
    const auto synced = CollectWrites(base, 2 * Block);
    Require(synced != 0, "the own-store block is not collected");

    std::array<std::byte, 8192> bytes{};
    bytes.fill(std::byte{0x44});
    Write(second, bytes);
    const auto stored = TrackerGeneration();
    Require(!UnchangedSince(second, bytes.size(), synced), "the driver store is not stamped");
    CollectWritesUncached(base, 2 * Block);
    Require(UnchangedSince(second, Block, stored), "a walk after a driver store reports the store again as a newer write");
    Require(UnchangedSinceCollected(second, bytes.size(), synced), "the driver store's own page faults are stamped as a CPU write");

    static_cast<volatile std::uint8_t*>(memory)[Block + 3 * 4096 + 8] = 0x22;
    const std::array<std::byte, 64> label{};
    Write(second + 3 * 4096 + 64, label);
    Require(!UnchangedSinceCollected(second + 3 * 4096, 4096, stored), "a CPU write before a driver store in its page is not stamped as one");

    const auto beforeCpu = TrackerGeneration();
    static_cast<volatile std::uint8_t*>(memory)[16] = 0x33;
    CollectWritesUncached(base, 2 * Block);
    Require(!UnchangedSince(base, 64, beforeCpu), "a CPU write after the driver store is not seen");
}

void CheckUnwatch() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    const auto imported = base + Block;
    std::memset(memory, 0x11, 3 * Block);
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 3 * Block);
    Require(before != 0 && UnchangedSince(imported, Block, before), "the import range was not initially watched");
    Unwatch(imported, Block);
    Require(!Watched(imported, Block) && !UnchangedSince(imported, Block, before), "unwatch kept trusting import stamps");
    Require(CollectWrites(base, 3 * Block) == 0 && CollectWritesUncached(imported, Block) == 0, "unwatch did not invalidate a cached collect");
    Require(Watched(base, Block) && Watched(base + 2 * Block, Block), "unwatch disabled unrelated memory");
    Require(MarkWritten(imported, Block) == 0, "an unwatched import still produces trusted stamps");
    std::array<std::uint64_t, 1> generations{before};
    std::array<std::uint8_t, 1> changed{};
    Require(!ChangedBlocks(imported, Block, generations, changed), "an unwatched import still uses block stamps");
    Unwatch(imported, Block);
    const auto adjacent = CollectWrites(base, Block);
    Require(adjacent != 0, "unwatch broke the adjacent block's collect");
    static_cast<volatile std::uint8_t*>(memory)[0] = 0x33;
    CollectWritesUncached(base, Block);
    Require(!UnchangedSince(base, Block, adjacent), "unwatch hid a CPU edit in unrelated memory");
}

// Live `unchanged_lockfree`: the lock-free UnchangedSince and UnchangedSinceAll answer exactly as
// the locked queries for every range shape (inside one block, across blocks, partly unwatched,
// outside the arena, empty, generation 0), before and after stamps, unwatches and fresh backing.
void CheckLockFreeAgrees(const char* stage, std::uint64_t base, std::size_t bytes, std::span<const std::uint64_t> generations) {
    struct Query {
        std::uint64_t address;
        std::size_t bytes;
    };
    std::array<std::uint64_t, 64> outsideBytes{};
    const auto outside = reinterpret_cast<std::uint64_t>(outsideBytes.data());
    const std::vector<Query> queries{{base, bytes}, {base, 64}, {base + 4096, 8192}, {base + Block - 16, 32}, {base + Block, Block}, {base + 2 * Block + 100, Block}, {base, 0}, {outside, sizeof(outsideBytes)}, {base + bytes - 8, 8}, {base + bytes - 8, 16}};
    for (const auto generation : generations) {
        for (const auto& query : queries) {
            const std::array<UnchangedQuery, 2> both{UnchangedQuery{query.address, query.bytes, generation}, UnchangedQuery{base, 64, generation}};
            AgcDriver::Graphics::testLockFree.store(0);
            const bool locked = UnchangedSince(query.address, query.bytes, generation);
            const bool lockedAll = UnchangedSinceAll(both);
            AgcDriver::Graphics::testLockFree.store(1);
            const bool lockFree = UnchangedSince(query.address, query.bytes, generation);
            const bool lockFreeAll = UnchangedSinceAll(both);
            Require(locked == lockFree, std::string(stage) + ": the lock-free UnchangedSince differs from the locked query");
            Require(lockedAll == lockFreeAll, std::string(stage) + ": the lock-free UnchangedSinceAll differs from the locked query");
        }
    }
    AgcDriver::Graphics::testLockFree.store(-1);
}

void CheckLockFreeUnchanged() {
    void* memory = AllocateWatched(4 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 4 * Block);
    BumpCollectEpoch();
    const auto first = CollectWrites(base, 4 * Block);
    Require(first != 0, "the lock-free test block is not collected");
    std::vector<std::uint64_t> generations{0, 1, first};
    CheckLockFreeAgrees("collected", base, 4 * Block, generations);
    const auto marked = MarkWritten(base + Block + 64, 4);
    generations.push_back(marked);
    CheckLockFreeAgrees("after a driver store", base, 4 * Block, generations);
    static_cast<volatile std::uint8_t*>(memory)[2 * Block + 8] = 0x22;
    CollectWritesUncached(base, 4 * Block);
    generations.push_back(TrackerGeneration());
    CheckLockFreeAgrees("after a CPU store", base, 4 * Block, generations);
    Unwatch(base + 2 * Block, Block);
    CheckLockFreeAgrees("after an unwatch", base, 4 * Block, generations);
    AgcDriver::Graphics::testLockFree.store(1);
    Require(!UnchangedSince(base + 2 * Block, Block, TrackerGeneration()), "an unwatched block reads as unchanged without the lock");
    Require(UnchangedSince(base, Block, TrackerGeneration()), "a watched, unwritten block reads as changed without the lock");
    AgcDriver::Graphics::testLockFree.store(-1);
}

// Live `unchanged_lockfree` under concurrency: a writer stamps blocks (MarkWritten) while readers
// ask about them without the lock. A store whose MarkWritten returned before a reader's question
// (published through `published`) must be seen: the reader never answers "unchanged" for a
// generation older than a stamp it was told about.
void CheckLockFreeConcurrent() {
    void* memory = AllocateWatched(8 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 8 * Block);
    BumpCollectEpoch();
    Require(CollectWrites(base, 8 * Block) != 0, "the concurrent test block is not collected");
    AgcDriver::Graphics::testLockFree.store(1);
    std::atomic<std::uint64_t> published{0};
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};
    std::atomic<std::uint64_t> answered{0};
    std::vector<std::thread> readers;
    for (int reader = 0; reader < 3; ++reader) {
        readers.emplace_back([&, reader] {
            while (!stop.load(std::memory_order_acquire)) {
                const auto seen = published.load(std::memory_order_acquire);
                if (seen <= 1) continue;
                // The newest published store is newer than seen - 1: its block must read changed.
                const auto block = static_cast<std::uint64_t>(reader) % 8u;
                static_cast<void>(block);
                if (UnchangedSince(base, 8 * Block, seen - 1)) failures.fetch_add(1);
                const std::array<UnchangedQuery, 1> all{UnchangedQuery{base, 8 * Block, seen - 1}};
                if (UnchangedSinceAll(all)) failures.fetch_add(1);
                answered.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    // At least 20000 stores, and on until the readers answered a few hundred times (a loaded
    // machine may not schedule them during the first stores).
    for (int store = 0; store < 20000 || (answered.load(std::memory_order_relaxed) < 300 && store < 50000000); ++store) {
        const auto generation = MarkWritten(base + static_cast<std::uint64_t>(store % 8) * Block + 16, 4);
        Require(generation != 0, "a concurrent driver store was not stamped");
        published.store(generation, std::memory_order_release);
    }
    stop.store(true, std::memory_order_release);
    for (auto& reader : readers) reader.join();
    AgcDriver::Graphics::testLockFree.store(-1);
    Require(failures.load() == 0, "a lock-free reader missed a published store");
    Require(answered.load() != 0, "the lock-free readers asked nothing");
}

// Live `collect_local_copy`: a collect answered by another thread's walk of the same followed epoch
// answers the same with the copy on and off, a later collect of the range answers from the copy,
// and an unwatch makes the copy answer no more.
void CheckLocalCopy() {
    if (!SharedEpochMemo()) {
        std::cout << "APS5_SHARED_EPOCH_MEMO unset: collect_local_copy check skipped\n";
        return;
    }
    static std::uint64_t token = 1000;
    for (const int copy : {0, 1}) {
        void* memory = AllocateWatched(2 * Block);
        const auto base = reinterpret_cast<std::uint64_t>(memory);
        std::memset(memory, 0x11, 2 * Block);
        AgcDriver::Graphics::testLocalCopy.store(copy);
        const auto followed = ++token;
        std::uint64_t walked = 0;
        std::thread walker([&] {
            FollowCollectEpoch(followed);
            walked = CollectWrites(base, 2 * Block);
        });
        walker.join();
        Require(walked != 0, "the walker's collect failed");
        FollowCollectEpoch(followed);
        const auto first = CollectWrites(base, Block);
        const auto second = CollectWrites(base, Block);
        Require(first != 0 && first == TrackerGeneration() && second == first, "a collect answered by the walker's entry does not return the current generation");
        Require(UnchangedSince(base, Block, first), "the range reads as changed after the shared answer");
        Unwatch(base + 4096, 4096);
        Require(CollectWrites(base, Block) == 0, "a collect answered after an unwatch of the range");
        BumpCollectEpoch();
    }
    AgcDriver::Graphics::testLocalCopy.store(-1);
}

#ifdef _WIN32
void CheckPrivateMappingReuse() {
    void* memory = AllocateWatched(3 * Block);
    const auto base = reinterpret_cast<std::uint64_t>(memory);
    std::memset(memory, 0x11, 3 * Block);
    BumpCollectEpoch();
    const auto before = CollectWrites(base, 3 * Block);
    Require(before != 0, "the original backing was not collected");
    Unwatch(base + Block, Block);
    Require(!Watched(base + Block, Block), "the imported block stayed watched");
    GuestArena::GuestArenaCommit_nid_postfix(memory, 3 * Block, PAGE_READWRITE, 3 * Block);
    Require(!Watched(base + Block, Block), "a protection-only commit restored an imported block");
    const auto neighbour = CollectWrites(base, Block);
    Require(neighbour != 0, "the neighbouring block was not collected before remapping");
    GuestArena::GuestArenaReset_nid_postfix(memory, 3 * Block);
    GuestArena::GuestArenaRelease_nid_postfix(memory, 3 * Block);
    auto* replacement = GuestArena::GuestArenaAllocateAtOrAbove_nid_postfix(base, 3 * Block, Block);
    Require(replacement == memory, "the released arena address was not reused");
    GuestArena::GuestArenaCommit_nid_postfix(replacement, 3 * Block, PAGE_READWRITE, 3 * Block);
    Require(Watched(base, 3 * Block), "fresh backing at an excluded address is not watched");
    Require(!UnchangedSince(base + Block, Block, before), "fresh backing reused an old import generation");
    Require(!UnchangedSince(base, Block, neighbour), "fresh backing reused an adjacent block's old generation");
    const auto after = CollectWrites(base, 3 * Block);
    Require(after > before && UnchangedSince(base, 3 * Block, after), "fresh backing was not collected");
    static_cast<volatile std::uint8_t*>(replacement)[Block + 8] = 0x22;
    CollectWritesUncached(base, 3 * Block);
    Require(!UnchangedSince(base + Block, Block, after), "a CPU edit in fresh backing was missed");
    Unwatch(base + Block, Block);
    Require(!Watched(base + Block, Block) && CollectWrites(base + Block, Block) == 0, "a second import of fresh backing stayed watched");
    GuestArena::GuestArenaReset_nid_postfix(replacement, 3 * Block);
    GuestArena::GuestArenaRelease_nid_postfix(replacement, 3 * Block);
}
#endif
}

int main() {
    try {
        if (!WriteWatched()) {
            std::cout << "no write watching: skipped\n";
            return 77;
        }
        // Every check runs with the lock-free queries off, then on (live `unchanged_lockfree`).
        for (const int lockFree : {0, 1}) {
            AgcDriver::Graphics::testLockFree.store(lockFree);
            CheckSharedBlock();
            CheckOwnStore();
            CheckUnwatch();
#ifdef _WIN32
            CheckPrivateMappingReuse();
#endif
        }
        AgcDriver::Graphics::testLockFree.store(-1);
#ifdef _WIN32
        CheckLockFreeUnchanged();
        CheckLockFreeConcurrent();
        CheckLocalCopy();
#endif
    } catch (const std::exception& error) {
        std::cerr << "write tracking test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "write tracking tests passed\n";
    return 0;
}
