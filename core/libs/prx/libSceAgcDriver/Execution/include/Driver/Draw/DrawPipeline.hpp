#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    // What the queue thread does while the committer waits for work ([draw] pipelined line).
    enum class Producer : std::uint8_t { Working, MemoryWait, PrepRoom, EnqueueRoom, Draining, Polling, Count };
    static std::atomic<std::uint8_t>& ProducerState();
    // APS5_TRACE_PIPE=<seconds>: a timeline of the queue thread, the preparation workers and the
    // committer from that many seconds after the first event, written to pipe-trace.txt.
    enum class Trace : std::uint16_t { Producer = 1, QueueDraw, QueueDrawEnd, QueueDispatch, QueueDispatchEnd, QueueFlip, Prep, PrepEnd, PrepTurn, PrepTurnEnd, Commit, CommitEnd, CommitIdle, CommitIdleEnd, QueuePacket, QueueFlush, QueueFlushEnd, QueueDeviceLock, QueueDeviceLockEnd };
    static void PipeTrace(Trace kind, std::uint32_t argument = 0);
    struct TraceScope {
        Trace end;
        TraceScope(Trace begin, Trace end, std::uint32_t argument = 0) : end(end) { PipeTrace(begin, argument); }
        ~TraceScope() { PipeTrace(end); }
    };
    struct ProducerScope {
        std::uint8_t previous;
        bool active;
        explicit ProducerScope(Producer state) : previous(0), active(Active() && PrepSequence() == 0) {
            if (active) previous = ProducerState().exchange(static_cast<std::uint8_t>(state), std::memory_order_relaxed);
            if (active) PipeTrace(Trace::Producer, static_cast<std::uint32_t>(state));
        }
        ~ProducerScope() {
            if (active) ProducerState().store(previous, std::memory_order_relaxed);
            if (active) PipeTrace(Trace::Producer, previous);
        }
    };

    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {});
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    bool Busy() const { return outstanding.load(std::memory_order_acquire) != 0 || prepOutstanding.load(std::memory_order_acquire) != 0; }
    static bool SplitDraws();
    void Prepare(Commit job, std::vector<Range> writes = {}, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {});
    // A job of the queue thread itself: in order behind the outstanding preparations, without
    // waiting for them (its writes are visible to Overlaps from here on).
    void EnqueueOrdered(Commit commit, std::vector<Range> writes);
    // A preparation reading memory an earlier job writes: every earlier job is enqueued first.
    void WaitEarlierPreparations();
    void DrainPrep() { drainPrep(); }
    bool OnPrepThread() const { return PrepSequence() != 0; }
    static std::size_t PrepWorkers();
    static std::uint64_t& PrepSequence();
    static bool& PrepEnqueued();
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    std::pair<std::uint64_t, std::uint64_t> OverlapRange(std::uint64_t address, std::size_t bytes);
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);

private:
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
    };
    DrawPipeline() = default;
    void run();
    void prepRun();
    void drainPrep();
    void waitPrepTurn(std::uint64_t sequence);
    void finishPrepTurn(std::uint64_t sequence);
    void rethrowFailure();
    void report(std::chrono::steady_clock::time_point now);

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Item> items;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::uint32_t idleWaiters = 0;
    bool committerWaiting = false;
    std::thread thread;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::uint64_t depthSum = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Producer::Count)> idleNs{};
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::mutex prepMutex;
    std::condition_variable prepWake;
    std::condition_variable prepIdle;
    std::deque<std::pair<std::uint64_t, Commit>> prepItems;
    struct PrepWrites {
        std::uint64_t sequence;
        std::vector<Range> writes;
        std::uint64_t labelAddress;
        std::vector<std::byte> labelBytes;
    };
    std::deque<PrepWrites> prepWrites;
    std::atomic<std::size_t> prepOutstanding{0};
    std::vector<std::thread> prepThreads;
    std::uint64_t prepNextSequence = 1;
    std::uint64_t prepTurn = 1;
    std::condition_variable prepTurnChanged;
    std::uint64_t prepJobs = 0;
    std::uint64_t prepNs = 0;
    std::uint64_t prepWaitNs = 0;
    std::uint64_t prepFailures = 0;
};

}

#endif
