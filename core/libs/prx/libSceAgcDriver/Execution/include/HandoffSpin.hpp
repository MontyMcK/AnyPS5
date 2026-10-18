#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HANDOFFSPIN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HANDOFFSPIN_HPP

#include "prx/libc/include/CpuTopology.hpp"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace AgcDriver {

inline void SpinPause() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

inline unsigned UsableProcessors() {
    const auto count = static_cast<unsigned>(std::bitset<64>(CpuTopology::Get().process).count());
    return count != 0 ? count : std::thread::hardware_concurrency();
}

inline std::chrono::nanoseconds HandoffSpinWindowFor(const char* configured, unsigned usableProcessors) {
    constexpr std::chrono::microseconds automaticWindow{150};
    constexpr unsigned automaticProcessorsAbove = 8;
    constexpr long long longestWindowUs = 10000;
    if (configured != nullptr) return std::chrono::microseconds(std::clamp<long long>(std::strtoll(configured, nullptr, 10), 0, longestWindowUs));
    return usableProcessors > automaticProcessorsAbove ? std::chrono::nanoseconds(automaticWindow) : std::chrono::nanoseconds::zero();
}

inline std::chrono::nanoseconds HandoffSpinWindow() {
    static const std::chrono::nanoseconds window = HandoffSpinWindowFor(std::getenv("APS5_HANDOFF_SPIN_US"), UsableProcessors());
    return window;
}

template<typename TReady>
bool SpinUntil(std::chrono::nanoseconds window, TReady&& ready) {
    if (ready()) return true;
    if (window.count() <= 0) return false;
    const auto until = std::chrono::steady_clock::now() + window;
    do {
        SpinPause();
        if (ready()) return true;
    } while (std::chrono::steady_clock::now() < until);
    return false;
}

template<typename TMutex>
void LockSpinning(TMutex& mutex, std::chrono::nanoseconds window) {
    if (window.count() > 0 && SpinUntil(window, [&] { return mutex.try_lock(); })) return;
    mutex.lock();
}

// Live `lock_spin` (env APS5_LOCK_SPIN=0 turns it off; APS5_LOCK_SPIN_US sets the window, default
// 5 us with more than 4 usable processors, else no spin): the driver's short critical sections (the
// write tracker, the resource cache, the fork-guarded caches) are std::mutex, which blocks in the
// kernel at the first contended acquisition. Sampling the World Map committer found about 6% of
// Graphics::Draw in such waits (ZwWaitForSingleObject under lockTracker and ResourceCache::Find),
// each a sleep and a wake for a section held a microsecond or two. A contended acquisition now
// retries try_lock for the window first. Which thread gets the mutex, and when, is still decided by
// the mutex: only the wait before blocking changes.
inline std::chrono::nanoseconds LockSpinWindowFor(const char* configured, unsigned usableProcessors) {
    constexpr std::chrono::microseconds automaticWindow{5};
    constexpr unsigned automaticProcessorsAbove = 4;
    constexpr long long longestWindowUs = 1000;
    if (configured != nullptr) return std::chrono::microseconds(std::clamp<long long>(std::strtoll(configured, nullptr, 10), 0, longestWindowUs));
    return usableProcessors > automaticProcessorsAbove ? std::chrono::nanoseconds(automaticWindow) : std::chrono::nanoseconds::zero();
}

inline std::chrono::nanoseconds LockSpinWindow() {
    static const std::chrono::nanoseconds window = LockSpinWindowFor(std::getenv("APS5_LOCK_SPIN_US"), UsableProcessors());
    return window;
}

// Live `room_spin` (env APS5_ROOM_SPIN=0 turns it off; APS5_ROOM_SPIN_US sets the window, default
// 100 us with more than 8 usable processors, as the handoff spin, else no spin): how long a producer
// finding the draw queue full spins for room before sleeping on the condition variable
// (DrawPipeline::spinForRoom). On fewer processors the spin would compete with the committer.
inline std::chrono::nanoseconds RoomSpinWindowFor(const char* configured, unsigned usableProcessors) {
    constexpr std::chrono::microseconds automaticWindow{100};
    constexpr unsigned automaticProcessorsAbove = 8;
    constexpr long long longestWindowUs = 10000;
    if (configured != nullptr) return std::chrono::microseconds(std::clamp<long long>(std::strtoll(configured, nullptr, 10), 0, longestWindowUs));
    return usableProcessors > automaticProcessorsAbove ? std::chrono::nanoseconds(automaticWindow) : std::chrono::nanoseconds::zero();
}

inline std::chrono::nanoseconds RoomSpinWindow() {
    static const std::chrono::nanoseconds window = RoomSpinWindowFor(std::getenv("APS5_ROOM_SPIN_US"), UsableProcessors());
    return window;
}

// Locks `mutex`, spinning for LockSpinWindow() first when `spin`.
template<typename TMutex>
void LockMaybeSpinning(TMutex& mutex, bool spin) {
    if (spin) LockSpinning(mutex, LockSpinWindow());
    else mutex.lock();
}

}

#endif
