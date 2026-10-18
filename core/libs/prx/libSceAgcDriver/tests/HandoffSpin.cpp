#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using AgcDriver::HandoffSpinWindowFor;
using AgcDriver::LockMaybeSpinning;
using AgcDriver::LockSpinWindowFor;
using AgcDriver::RoomSpinWindowFor;
using AgcDriver::LockSpinning;
using AgcDriver::SpinUntil;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void CheckWindowPolicy() {
    Require(HandoffSpinWindowFor(nullptr, 20) == 150us, "20 usable processors do not spin for 150 us");
    Require(HandoffSpinWindowFor(nullptr, 9) == 150us, "9 usable processors do not spin for 150 us");
    Require(HandoffSpinWindowFor(nullptr, 8) == 0ns, "8 usable processors spin");
    Require(HandoffSpinWindowFor(nullptr, 4) == 0ns, "4 usable processors spin");
    Require(HandoffSpinWindowFor(nullptr, 0) == 0ns, "an unknown processor count spins");
    Require(HandoffSpinWindowFor("0", 20) == 0ns, "APS5_HANDOFF_SPIN_US=0 still spins");
    Require(HandoffSpinWindowFor("400", 2) == 400us, "APS5_HANDOFF_SPIN_US=400 does not set 400 us on 2 logical processors");
    Require(HandoffSpinWindowFor("-5", 20) == 0ns, "a negative window spins");
    Require(HandoffSpinWindowFor("text", 20) == 0ns, "a window that is no number spins");
    Require(HandoffSpinWindowFor("99999999999", 20) == 10000us, "a huge window is not limited to 10000 us");
}

// Live `lock_spin`: 5 us with more than 4 usable processors, none otherwise; the env value wins.
void CheckLockSpinPolicy() {
    Require(LockSpinWindowFor(nullptr, 20) == 5us, "20 usable processors do not spin a contended lock for 5 us");
    Require(LockSpinWindowFor(nullptr, 5) == 5us, "5 usable processors do not spin a contended lock for 5 us");
    Require(LockSpinWindowFor(nullptr, 4) == 0ns, "4 usable processors spin a contended lock");
    Require(LockSpinWindowFor(nullptr, 0) == 0ns, "an unknown processor count spins a contended lock");
    Require(LockSpinWindowFor("0", 20) == 0ns, "APS5_LOCK_SPIN_US=0 still spins");
    Require(LockSpinWindowFor("12", 2) == 12us, "APS5_LOCK_SPIN_US=12 does not set 12 us on 2 logical processors");
    Require(LockSpinWindowFor("-3", 20) == 0ns, "a negative lock spin window spins");
    Require(LockSpinWindowFor("99999999", 20) == 1000us, "a huge lock spin window is not limited to 1000 us");
}

// Live `room_spin`: 100 us with more than 8 usable processors, none otherwise; the env value wins.
void CheckRoomSpinPolicy() {
    Require(RoomSpinWindowFor(nullptr, 20) == 100us, "20 usable processors do not spin for queue room for 100 us");
    Require(RoomSpinWindowFor(nullptr, 9) == 100us, "9 usable processors do not spin for queue room for 100 us");
    Require(RoomSpinWindowFor(nullptr, 8) == 0ns, "8 usable processors spin for queue room");
    Require(RoomSpinWindowFor("0", 20) == 0ns, "APS5_ROOM_SPIN_US=0 still spins");
    Require(RoomSpinWindowFor("40", 2) == 40us, "APS5_ROOM_SPIN_US=40 does not set 40 us on 2 logical processors");
    Require(RoomSpinWindowFor("99999999", 20) == 10000us, "a huge room spin window is not limited to 10000 us");
}

// The producer side of `room_spin` as DrawPipeline runs it: a bounded queue whose producer spins
// outside the mutex on an atomic copy of the size, then sleeps on the condition variable; the
// consumer notifies only when a producer sleeps. Nothing is lost, the bound holds, and the order
// of items is kept, with the spin and without.
void CheckRoomSpinQueue() {
    for (const auto window : {0us, 100us}) {
        constexpr std::size_t Depth = 4;
        constexpr int Items = 20000;
        std::mutex mutex;
        std::condition_variable idle, wake;
        std::deque<int> items;
        std::atomic<std::size_t> outstanding{0};
        int idleWaiters = 0;
        bool consumerWaiting = false;
        bool overfull = false;
        // Two producers, each with its own increasing sequence (item * 2 + producer): every item
        // arrives once and each producer's items stay in order.
        std::atomic<bool> outOfOrder{false};
        std::thread consumer([&] {
            int next[2] = {0, 1};
            for (int count = 0; count < Items * 2; ++count) {
                std::unique_lock lock(mutex);
                if (items.empty()) {
                    consumerWaiting = true;
                    wake.wait(lock, [&] { return !items.empty(); });
                    consumerWaiting = false;
                }
                const int value = items.front();
                if (value != next[value & 1]) outOfOrder.store(true);
                next[value & 1] = value + 2;
                items.pop_front();
                outstanding.fetch_sub(1, std::memory_order_release);
                if (idleWaiters != 0) idle.notify_all();
            }
        });
        auto produce = [&](int producer) {
          for (int sequence = 0; sequence < Items; ++sequence) {
            const int item = sequence * 2 + producer;
            std::unique_lock lock(mutex);
            if (items.size() >= Depth) {
                bool room = false;
                if (window.count() > 0) {
                    lock.unlock();
                    SpinUntil(window, [&] { return outstanding.load(std::memory_order_acquire) < Depth; });
                    lock.lock();
                    room = items.size() < Depth;
                }
                if (!room) {
                    ++idleWaiters;
                    idle.wait(lock, [&] { return items.size() < Depth; });
                    --idleWaiters;
                }
            }
            if (items.size() >= Depth) overfull = true;
            items.push_back(item);
            outstanding.fetch_add(1, std::memory_order_release);
            if (consumerWaiting) wake.notify_one();
          }
        };
        std::thread second(produce, 1);
        produce(0);
        second.join();
        consumer.join();
        Require(!outOfOrder.load(), "room spin: a producer's items came out of order or one was lost");
        Require(!overfull, "room spin: an item was pushed into a full queue");
        Require(items.empty() && outstanding.load() == 0, "room spin: items were left behind");
    }
}

// Both modes of LockMaybeSpinning keep mutual exclusion and lose no update under contention, and
// a spin that outlasts its window still waits for the holder.
void CheckLockMaybeSpinning() {
    for (const bool spin : {false, true}) {
        constexpr int Workers = 4;
        constexpr int Rounds = 20000;
        std::mutex mutex;
        std::atomic<int> inside{0};
        std::atomic<bool> overlapped{false};
        std::uint64_t counter = 0;
        std::vector<std::thread> workers;
        // The spin case uses an explicit window: LockSpinWindow() is 0 on small machines.
        auto lockIt = [&] {
            if (spin) LockSpinning(mutex, 5us);
            else LockMaybeSpinning(mutex, false);
        };
        for (int worker = 0; worker < Workers; ++worker) {
            workers.emplace_back([&] {
                for (int round = 0; round < Rounds; ++round) {
                    lockIt();
                    if (inside.fetch_add(1) != 0) overlapped.store(true);
                    ++counter;
                    inside.fetch_sub(1);
                    mutex.unlock();
                }
            });
        }
        for (auto& worker : workers) worker.join();
        const std::string what = spin ? "lock_spin on" : "lock_spin off";
        Require(!overlapped.load(), what + ": two threads held the mutex at once");
        Require(counter == static_cast<std::uint64_t>(Workers) * Rounds, what + ": increments were lost");
        std::atomic<bool> held{false}, released{false};
        std::thread holder([&] {
            mutex.lock();
            held.store(true);
            std::this_thread::sleep_for(20ms);
            released.store(true);
            mutex.unlock();
        });
        while (!held.load()) std::this_thread::yield();
        lockIt();
        const bool afterRelease = released.load();
        mutex.unlock();
        holder.join();
        Require(afterRelease, what + ": the lock was taken while another thread held it");
    }
}

void CheckUsableProcessors() {
    const unsigned usable = AgcDriver::UsableProcessors();
    Require(usable >= 1, "no usable processor was counted");
    Require(usable <= (std::max)(std::thread::hardware_concurrency(), 64u), "more usable processors than the machine has");
}

void CheckSpinUntil() {
    int calls = 0;
    Require(SpinUntil(0ns, [&] { ++calls; return true; }) && calls == 1, "a ready condition was evaluated more than once");
    calls = 0;
    Require(!SpinUntil(0ns, [&] { ++calls; return false; }) && calls == 1, "a zero window evaluated the condition more than once");
    calls = 0;
    Require(SpinUntil(1s, [&] { return ++calls == 5; }) && calls == 5, "a condition that holds on its fifth evaluation was not seen");
    const auto start = std::chrono::steady_clock::now();
    Require(!SpinUntil(3ms, [] { return false; }), "a condition that never holds was reported ready");
    Require(std::chrono::steady_clock::now() - start >= 3ms, "the spin returned before its window ended");
    std::atomic<bool> flag{false};
    std::thread setter([&] {
        std::this_thread::sleep_for(500us);
        flag.store(true, std::memory_order_release);
    });
    const bool seen = SpinUntil(5s, [&] { return flag.load(std::memory_order_acquire); });
    setter.join();
    Require(seen, "a flag set by another thread was not seen within the window");
}

template<typename TMutex>
void Contend(std::chrono::nanoseconds window, const char* what) {
    constexpr int Workers = 4;
    constexpr int Rounds = 20000;
    TMutex mutex;
    std::atomic<int> inside{0};
    std::atomic<bool> overlapped{false};
    std::uint64_t counter = 0;
    std::vector<std::thread> workers;
    for (int worker = 0; worker < Workers; ++worker) {
        workers.emplace_back([&] {
            for (int round = 0; round < Rounds; ++round) {
                LockSpinning(mutex, window);
                if (inside.fetch_add(1) != 0) overlapped.store(true);
                ++counter;
                inside.fetch_sub(1);
                mutex.unlock();
            }
        });
    }
    for (auto& worker : workers) worker.join();
    Require(!overlapped.load(), std::string(what) + ": two threads held the mutex at once");
    Require(counter == static_cast<std::uint64_t>(Workers) * Rounds, std::string(what) + ": increments were lost");
}

void CheckBlockingFallback() {
    std::mutex mutex;
    std::atomic<bool> held{false};
    std::atomic<bool> released{false};
    std::thread holder([&] {
        mutex.lock();
        held.store(true);
        std::this_thread::sleep_for(20ms);
        released.store(true);
        mutex.unlock();
    });
    while (!held.load()) std::this_thread::yield();
    LockSpinning(mutex, 100us);
    const bool afterRelease = released.load();
    mutex.unlock();
    holder.join();
    Require(afterRelease, "a spin that outlasted its window did not wait for the holder");
}

void CheckRecursion() {
    std::recursive_mutex mutex;
    LockSpinning(mutex, 100us);
    LockSpinning(mutex, 100us);
    mutex.unlock();
    mutex.unlock();
}

}

int main() {
    try {
        CheckWindowPolicy();
        CheckLockSpinPolicy();
        CheckLockMaybeSpinning();
        CheckRoomSpinPolicy();
        CheckRoomSpinQueue();
        CheckUsableProcessors();
        CheckSpinUntil();
        Contend<std::mutex>(0ns, "plain lock");
        Contend<std::mutex>(50us, "spin 50 us");
        Contend<std::mutex>(1ms, "spin 1 ms");
        Contend<std::recursive_mutex>(150us, "recursive mutex");
        CheckBlockingFallback();
        CheckRecursion();
        std::cout << "handoff spin tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
