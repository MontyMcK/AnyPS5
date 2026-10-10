#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

}

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        if (text == nullptr) return std::size_t{8};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

bool& DrawPipeline::Active() {
    static thread_local bool active = false;
    return active;
}

namespace {

struct PipeEvent {
    std::int64_t ns;
    std::uint32_t thread;
    std::uint16_t kind;
    std::uint16_t argument;
};

struct PipeTraceState {
    std::int64_t from = 0;
    std::int64_t until = 0;
    std::vector<PipeEvent> events;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> dumped{false};
    std::atomic<std::uint32_t> threads{0};
};

PipeTraceState* PipeTracing() {
    static PipeTraceState* state = [] () -> PipeTraceState* {
        const char* text = std::getenv("APS5_TRACE_PIPE");
        if (text == nullptr) return nullptr;
        auto* created = new PipeTraceState();
        created->events.resize(std::size_t{1} << 21u);
        const auto start = std::chrono::steady_clock::now().time_since_epoch().count();
        created->from = start + static_cast<std::int64_t>(std::strtod(text, nullptr) * 1e9);
        created->until = created->from + 2'000'000'000ll;
        return created;
    }();
    return state;
}

void dumpPipeTrace(PipeTraceState& state) {
    if (state.dumped.exchange(true)) return;
    const auto count = std::min(state.next.load(), state.events.size());
    if (FILE* file = std::fopen("pipe-trace.txt", "w")) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto& event = state.events[i];
            std::fprintf(file, "%.3f %u %u %u%c", static_cast<double>(event.ns - state.from) / 1e3, event.thread, event.kind, event.argument, 10);
        }
        std::fclose(file);
    }
    std::fprintf(stderr, "[pipe-trace] wrote %zu events to pipe-trace.txt%c", count, 10);
}

}

void DrawPipeline::PipeTrace(Trace kind, std::uint32_t argument) {
    auto* state = PipeTracing();
    if (state == nullptr || state->dumped.load(std::memory_order_relaxed)) return;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    if (now < state->from) return;
    if (now > state->until) {
        dumpPipeTrace(*state);
        return;
    }
    static thread_local std::uint32_t thread = 0;
    if (thread == 0) thread = state->threads.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto index = state->next.fetch_add(1, std::memory_order_relaxed);
    if (index >= state->events.size()) {
        dumpPipeTrace(*state);
        return;
    }
    state->events[index] = {now, thread, static_cast<std::uint16_t>(kind), static_cast<std::uint16_t>(argument)};
}

std::atomic<std::uint8_t>& DrawPipeline::ProducerState() {
    static std::atomic<std::uint8_t> state{0};
    return state;
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    static thread_local std::uint64_t followed = 0;
    if (followed == token) return;
    followed = token;
    if (GuestMemory::SharedEpochMemo()) GuestMemory::FollowCollectEpoch(token);
    else GuestMemory::BumpCollectEpoch();
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

bool DrawPipeline::SplitDraws() {
    static const bool enabled = std::getenv("APS5_SPLIT_DRAWS") != nullptr && Depth() != 0;
    return enabled;
}

std::size_t DrawPipeline::PrepWorkers() {
    static const std::size_t workers = [] {
        const char* text = std::getenv("APS5_SPLIT_DRAWS");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 1ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1, 16));
    }();
    return workers;
}

std::uint64_t& DrawPipeline::PrepSequence() {
    static thread_local std::uint64_t sequence = 0;
    return sequence;
}

bool& DrawPipeline::PrepEnqueued() {
    static thread_local bool enqueued = false;
    return enqueued;
}

void DrawPipeline::Prepare(Commit job, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes) {
    std::unique_lock lock(prepMutex);
    const auto room = Depth() * PrepWorkers();
    if (prepOutstanding.load(std::memory_order_relaxed) >= room) {
        const ProducerScope producer(Producer::PrepRoom);
        const auto start = std::chrono::steady_clock::now();
        prepIdle.wait(lock, [&] { return prepOutstanding.load(std::memory_order_relaxed) < room; });
        prepWaitNs += elapsedNs(start);
    }
    if (!writes.empty()) prepWrites.push_back({prepNextSequence, std::move(writes), labelAddress, std::move(labelBytes)});
    prepItems.emplace_back(prepNextSequence++, std::move(job));
    prepOutstanding.fetch_add(1, std::memory_order_release);
    while (prepThreads.size() < PrepWorkers()) prepThreads.emplace_back([this] { prepRun(); });
    prepWake.notify_one();
}

void DrawPipeline::waitPrepTurn(std::uint64_t sequence) {
    const TraceScope trace(Trace::PrepTurn, Trace::PrepTurnEnd);
    std::unique_lock lock(prepMutex);
    prepTurnChanged.wait(lock, [&] { return prepTurn == sequence; });
}

void DrawPipeline::finishPrepTurn(std::uint64_t sequence) {
    std::lock_guard lock(prepMutex);
    if (prepTurn == sequence) ++prepTurn;
    prepTurnChanged.notify_all();
}

void DrawPipeline::prepRun() {
    Active() = true;
    for (;;) {
        std::pair<std::uint64_t, Commit> job;
        {
            std::unique_lock lock(prepMutex);
            prepWake.wait(lock, [&] { return !prepItems.empty(); });
            job = std::move(prepItems.front());
            prepItems.pop_front();
        }
        PrepSequence() = job.first;
        PrepEnqueued() = false;
        const auto start = std::chrono::steady_clock::now();
        bool failed = false;
        PipeTrace(Trace::Prep, static_cast<std::uint32_t>(job.first & 0xffffu));
        try {
            job.second();
        } catch (const std::exception& error) {
            failed = true;
            static std::atomic<std::uint64_t> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[draw] split draw preparation failed: %.200s\n", error.what());
        } catch (...) {
            failed = true;
        }
        if (!PrepEnqueued()) {
            waitPrepTurn(job.first);
            finishPrepTurn(job.first);
        }
        PipeTrace(Trace::PrepEnd);
        PrepSequence() = 0;
        const auto spent = elapsedNs(start);
        std::lock_guard lock(prepMutex);
        std::erase_if(prepWrites, [&](const PrepWrites& entry) { return entry.sequence == job.first; });
        ++prepJobs;
        prepNs += spent;
        if (failed) ++prepFailures;
        prepOutstanding.fetch_sub(1, std::memory_order_release);
        prepIdle.notify_all();
    }
}

void DrawPipeline::EnqueueOrdered(Commit commit, std::vector<Range> writes) {
    if (!SplitDraws() || OnPrepThread() || prepOutstanding.load(std::memory_order_acquire) == 0) {
        Enqueue(std::move(commit), std::move(writes));
        return;
    }
    auto held = std::make_shared<std::pair<Commit, std::vector<Range>>>(std::move(commit), writes);
    Prepare([this, held] { Enqueue(std::move(held->first), std::move(held->second)); }, std::move(writes));
}

void DrawPipeline::WaitEarlierPreparations() {
    if (OnPrepThread() && !PrepEnqueued()) waitPrepTurn(PrepSequence());
}

void DrawPipeline::drainPrep() {
    if (prepOutstanding.load(std::memory_order_acquire) == 0 || OnPrepThread()) return;
    std::unique_lock lock(prepMutex);
    prepIdle.wait(lock, [&] { return prepOutstanding.load(std::memory_order_relaxed) == 0; });
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool orderedHandoff = std::getenv("APS5_NO_ORDERED_HANDOFF") == nullptr;
    if (orderedHandoff && SplitDraws() && !OnPrepThread() && prepOutstanding.load(std::memory_order_acquire) != 0) {
        // Behind the outstanding preparations, in their order, instead of waiting for them.
        struct Held {
            Commit commit;
            std::vector<Range> writes;
            std::uint64_t labelAddress;
            std::vector<std::byte> labelBytes;
        };
        auto held = std::make_shared<Held>(Held{std::move(commit), writes, labelAddress, labelBytes});
        Prepare([this, held] { Enqueue(std::move(held->commit), std::move(held->writes), held->labelAddress, std::move(held->labelBytes)); }, std::move(writes), labelAddress, std::move(labelBytes));
        return;
    }
    // Finished commits are destroyed here, off the committer (their captures are a draw's whole
    // front end), after the mutex and the preparation turn are released (declared before both).
    std::vector<Commit> finished;
    drainPrep();
    const auto sequence = PrepSequence();
    if (sequence != 0 && !PrepEnqueued()) waitPrepTurn(sequence);
    struct TurnRelease {
        DrawPipeline& pipeline;
        std::uint64_t sequence;
        ~TurnRelease() {
            if (sequence == 0 || PrepEnqueued()) return;
            PrepEnqueued() = true;
            pipeline.finishPrepTurn(sequence);
        }
    } release{*this, sequence};
    std::unique_lock lock(mutex);
    rethrowFailure();
    finished.swap(retired);
    if (items.size() >= Depth()) {
        const ProducerScope producer(Producer::EnqueueRoom);
        const auto start = std::chrono::steady_clock::now();
        ++idleWaiters;
        idle.wait(lock, [&] { return items.size() < Depth(); });
        --idleWaiters;
        fullWaitNs += elapsedNs(start);
        rethrowFailure();
    }
    depthSum += items.size();
    ++enqueued;
    items.push_back({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes)});
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) thread = std::thread([this] { run(); });
    if (committerWaiting) wake.notify_one();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    std::fprintf(stderr, "[draw] committer idle by queue-thread state (10 s): working %.0f ms, memory wait %.0f ms, prep room %.0f ms, enqueue room %.0f ms, draining %.0f ms, polling for submissions %.0f ms\n", idleNs[0] / 1e6, idleNs[1] / 1e6, idleNs[2] / 1e6, idleNs[3] / 1e6, idleNs[4] / 1e6, idleNs[5] / 1e6);
    idleNs = {};
    std::fprintf(stderr, "[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, drainText.c_str());
    if (SplitDraws()) {
        std::lock_guard prepLock(prepMutex);
        std::fprintf(stderr, "[draw] split preparation (10 s): %llu draws at %.1f us each, %llu failed, %.1f ms waiting for room\n", static_cast<unsigned long long>(prepJobs), prepJobs == 0 ? 0.0 : static_cast<double>(prepNs) / 1e3 / static_cast<double>(prepJobs), static_cast<unsigned long long>(prepFailures), static_cast<double>(prepWaitNs) / 1e6);
        prepJobs = prepNs = prepWaitNs = prepFailures = 0;
    }
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = 0;
    drains = {};
    drainWaitNs = {};
}

void DrawPipeline::run() {
    for (;;) {
        Commit* commit = nullptr;
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            if (items.empty()) {
                const auto state = std::min<std::size_t>(ProducerState().load(std::memory_order_relaxed), idleNs.size() - 1);
                const auto idleStart = std::chrono::steady_clock::now();
                PipeTrace(Trace::CommitIdle);
                wake.wait(lock, [&] { return !items.empty(); });
                PipeTrace(Trace::CommitIdleEnd);
                idleNs[state] += elapsedNs(idleStart);
            }
            committerWaiting = false;
            commit = &items.front().commit;
        }
        const auto start = std::chrono::steady_clock::now();
        std::exception_ptr fatal;
        bool failed = false;
        PipeTrace(Trace::Commit);
        try {
            (*commit)();
        } catch (const std::exception& error) {
            failed = true;
            static std::atomic<std::uint64_t> reported{0};
            if (reported.fetch_add(1, std::memory_order_relaxed) < 20) std::fprintf(stderr, "[draw] pipelined commit failed: %.200s\n", error.what());
        } catch (...) {
            fatal = std::current_exception();
        }
        PipeTrace(Trace::CommitEnd);
        const auto spent = elapsedNs(start);
        std::lock_guard lock(mutex);
        static const bool retireOffCommitter = std::getenv("APS5_NO_RETIRED_COMMITS") == nullptr;
        if (retireOffCommitter && retired.size() < 4096) retired.push_back(std::move(items.front().commit));
        items.pop_front();
        ++commits;
        commitNs += spent;
        if (failed) ++commitErrors;
        if (fatal != nullptr && failure == nullptr) failure = fatal;
        outstanding.fetch_sub(1, std::memory_order_release);
        if (idleWaiters != 0) idle.notify_all();
    }
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    if (!Busy()) return;
    const ProducerScope producer(Producer::Draining);
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    const auto start = std::chrono::steady_clock::now();
    drainPrep();
    std::unique_lock lock(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    drainWaitNs[index] += elapsedNs(start);
    rethrowFailure();
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    const auto end = address + bytes;
    if (prepOutstanding.load(std::memory_order_acquire) != 0 && !OnPrepThread()) {
        // The newest writer decides: jobs still held by the preparation come after every queued item.
        std::lock_guard prepLock(prepMutex);
        for (auto entry = prepWrites.rbegin(); entry != prepWrites.rend(); ++entry) {
            const bool overlaps = std::any_of(entry->writes.begin(), entry->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
            if (!overlaps) continue;
            if (entry->labelBytes.empty() || address < entry->labelAddress || end > entry->labelAddress + entry->labelBytes.size()) return std::nullopt;
            std::uint64_t value = 0;
            std::memcpy(&value, entry->labelBytes.data() + (address - entry->labelAddress), bytes);
            return value;
        }
    }
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

std::pair<std::uint64_t, std::uint64_t> DrawPipeline::OverlapRange(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return {begin, limit};
        }
    }
    return {0, 0};
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    if (prepOutstanding.load(std::memory_order_acquire) != 0) {
        std::lock_guard prepLock(prepMutex);
        const auto own = PrepSequence();
        for (const auto& entry : prepWrites) {
            if (own != 0 && entry.sequence >= own) continue;
            for (const auto& [begin, limit] : entry.writes) {
                if (address < limit && begin < end) return true;
            }
        }
    }
    std::lock_guard lock(mutex);
    for (const auto& item : items) {
        for (const auto& [begin, limit] : item.writes) {
            if (address < limit && begin < end) return true;
        }
    }
    return false;
}

}
