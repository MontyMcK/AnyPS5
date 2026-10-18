// Live `fast_heap_prefetch` (Execution/src/FastHeap.cpp): the allocator hands out the same blocks
// with the prefetch on and off, keeps them intact across threads, and follows the live flag.
// Runs with APS5_FAST_HEAP=1 (set by the test registration), so malloc and free are redirected.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

bool FastHeapTestInstalled();
bool FastHeapTestOwned(const void* block);
bool FastHeapTestPrefetch();

namespace AgcDriver::Graphics {

std::atomic<int> testPrefetchFlag{-1};

// The allocator's live flags, answered by the test: `fast_heap_prefetch` from testPrefetchFlag
// (-1: the fallback), every other name from its fallback.
bool LiveFlag(std::string_view name, bool fallback) {
    if (name == "fast_heap_prefetch") {
        const int value = testPrefetchFlag.load();
        return value < 0 ? fallback : value != 0;
    }
    return fallback;
}

}

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void SetPrefetch(int value) {
    AgcDriver::Graphics::testPrefetchFlag.store(value);
    // The allocator re-reads its live flags twice a second.
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    const bool wanted = value != 0;
    while (FastHeapTestPrefetch() != wanted) {
        Require(std::chrono::steady_clock::now() < until, "the allocator did not follow fast_heap_prefetch=" + std::to_string(value));
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

// Blocks of one class keep their contents while held; a few freed blocks (fewer than a batch, so
// they stay in the thread's list) come back last-in first-out, the same set either way.
void CheckReuse(const char* mode) {
    for (const std::size_t size : {24u, 48u, 200u, 1000u, 4000u, 30000u}) {
        constexpr int Count = 300;
        constexpr int Few = 2;
        std::vector<void*> first;
        for (int i = 0; i < Count; ++i) {
            void* block = std::malloc(size);
            Require(block != nullptr, std::string(mode) + ": malloc failed");
            Require(FastHeapTestOwned(block), std::string(mode) + ": a small block is not from the arena");
            std::memset(block, i & 0xff, size);
            first.push_back(block);
        }
        for (int i = 0; i < Count; ++i) {
            const auto* bytes = static_cast<const unsigned char*>(first[i]);
            Require(bytes[0] == static_cast<unsigned char>(i & 0xff) && bytes[size - 1] == static_cast<unsigned char>(i & 0xff), std::string(mode) + ": a block's contents changed while it was held");
        }
        for (int i = Count - 1; i >= Few; --i) std::free(first[i]);
        std::free(first[1]);
        std::free(first[0]);
        void* again0 = std::malloc(size);
        void* again1 = std::malloc(size);
        Require(again0 == first[0] && again1 == first[1], std::string(mode) + ": the blocks freed last were not taken first");
        std::free(again1);
        std::free(again0);
    }
}

// Blocks made on one thread and freed on another keep their contents and are reused.
void CheckCrossThread(const char* mode) {
    constexpr int Rounds = 50;
    constexpr int Count = 2000;
    for (int round = 0; round < Rounds; ++round) {
        std::vector<std::uint64_t*> blocks(Count);
        std::thread maker([&] {
            for (int i = 0; i < Count; ++i) {
                blocks[i] = static_cast<std::uint64_t*>(std::malloc(64));
                for (int word = 0; word < 8; ++word) blocks[i][word] = static_cast<std::uint64_t>(round) * 1000003u + static_cast<std::uint64_t>(i) * 8u + static_cast<std::uint64_t>(word);
            }
        });
        maker.join();
        std::thread freer([&] {
            for (int i = 0; i < Count; ++i) {
                for (int word = 0; word < 8; ++word) {
                    if (blocks[i][word] != static_cast<std::uint64_t>(round) * 1000003u + static_cast<std::uint64_t>(i) * 8u + static_cast<std::uint64_t>(word)) throw std::runtime_error(std::string(mode) + ": a block made on another thread lost its contents");
                }
                std::free(blocks[i]);
            }
        });
        freer.join();
    }
}

}

int main() {
    try {
        Require(FastHeapTestInstalled(), "the fast heap did not redirect malloc (APS5_FAST_HEAP=1 expected)");
        Require(FastHeapTestPrefetch(), "fast_heap_prefetch is not on by default");
        CheckReuse("prefetch on");
        CheckCrossThread("prefetch on");
        SetPrefetch(0);
        CheckReuse("prefetch off");
        CheckCrossThread("prefetch off");
        SetPrefetch(1);
        CheckReuse("prefetch on again");
        std::cout << "fast heap prefetch tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
