#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PASSRUN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PASSRUN_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace AgcDriver::DriverDetail {

// Live `parallel_commit`, stage P1 ("pass runs"). A pass run is a sequence of queued draw commits
// that continue one render pass: the draw that begins it (its head) and the queued draws behind it
// with the same pass hint (forkPassHint: the same attachments and extent). The committer keeps the
// GPU mutex across the whole run, so no other thread records between its draws (an async queue's
// dispatch, a starved-GPU submit or a label run would end the render pass and make the next draw a
// new head: another target refresh, another pass begin and another full barrier). The run ends at
// the first queued item that is not such a draw, when the recorder's pass is no longer continuable,
// or at the run's budget, so the waiters behind it wait a bounded time.
//
// Nothing a run does changes what is recorded for a given order of commits: only which other
// threads' work may interleave between them (an order the guest leaves open: other queues are
// ordered against queue 0 only by labels and waits, which are commits of their own and end runs)
// and when the batch is submitted (starve submits are timing, not semantics). Pass runs are the
// unit the parallel body builders (fork_build, fork_stream) work on; see parcommit-design.md.
//
// This header is the run's state machine, kept free of the driver so it is unit tested
// (tests/PassRun.cpp); DrawPipeline::run drives it.

struct PassRunItem {
    bool draw = false;
    std::uint64_t hint = 0;
};

struct PassRunLimits {
    std::chrono::nanoseconds budget{std::chrono::microseconds(300)};
    std::size_t maxDraws = 64;
};

enum class PassRunEnd : std::uint8_t { NotDraw, OtherPass, PassClosed, Budget, Cap, Stream, FlagOff, Count };

inline const char* PassRunEndName(PassRunEnd end) {
    constexpr std::array<const char*, static_cast<std::size_t>(PassRunEnd::Count)> names{"next not a draw", "next in another pass", "pass not continuable", "budget", "draw cap", "stream window", "flag off"};
    const auto index = static_cast<std::size_t>(end);
    return index < names.size() ? names[index] : "?";
}

struct PassRunStatistics {
    std::uint64_t runs = 0;
    std::uint64_t draws = 0;
    std::uint64_t contendedContinuations = 0;
    std::uint64_t deferredStarveChecks = 0;
    std::uint64_t cooldowns = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(PassRunEnd::Count)> ends{};
};

class PassRunTracker {
public:
    using Clock = std::chrono::steady_clock;

    // Before committing `front`: whether the commit belongs to a run (starting one when `front` is
    // a draw and the item queued right behind it is a draw of the same pass). While a run is active
    // the caller holds the GPU mutex across commits and defers its starve check.
    bool BeforeCommit(const PassRunItem& front, const std::optional<PassRunItem>& next, Clock::time_point now) {
        if (active && (!front.draw || front.hint != hint)) {
            // The caller only commits what AfterCommit let continue; a mismatch here means the queue
            // changed under the run (it never does: items are appended, the committer pops).
            end(front.draw ? PassRunEnd::OtherPass : PassRunEnd::NotDraw);
        }
        if (active) return true;
        if (!front.draw || !next.has_value() || !next->draw || next->hint != front.hint) {
            cooling = false;
            return false;
        }
        // Right after a run, one commit goes outside any run (the caller releases the mutex
        // around it), so waiters the contender count misses (try-lock pollers: WaitMemory's poll,
        // boundary reaps, build-ahead) get a window before the next run holds it again.
        if (cooling) {
            cooling = false;
            ++stats.cooldowns;
            return false;
        }
        active = true;
        hint = front.hint;
        started = now;
        committed = 0;
        ++stats.runs;
        return true;
    }

    // After `count` commits of the run (a forked run commits several at once): whether the run goes
    // on with `next`, the item now at the front of the queue. `passContinuable` is the recorder's
    // state read under the GPU mutex the run holds; `contended` whether other threads wait for it.
    // Returns the reason the run ended, or nothing while it continues.
    std::optional<PassRunEnd> AfterCommit(std::size_t count, const std::optional<PassRunItem>& next, bool passContinuable, bool contended, Clock::time_point now) {
        if (!active) return std::nullopt;
        committed += count;
        stats.draws += count;
        std::optional<PassRunEnd> why;
        if (!next.has_value() || !next->draw) why = PassRunEnd::NotDraw;
        else if (next->hint != hint) why = PassRunEnd::OtherPass;
        else if (!passContinuable) why = PassRunEnd::PassClosed;
        else if (committed >= limits.maxDraws) why = PassRunEnd::Cap;
        else if (now - started >= limits.budget) why = PassRunEnd::Budget;
        if (why.has_value()) {
            end(*why);
            return why;
        }
        if (contended) ++stats.contendedContinuations;
        ++stats.deferredStarveChecks;
        return std::nullopt;
    }

    // Ends an active run from outside (the flag was switched off, a stream window committed past
    // the pass); the caller releases the mutex at its next loop top.
    void Abandon(PassRunEnd why) {
        if (active) end(why);
    }

    bool Active() const { return active; }
    std::uint64_t Hint() const { return hint; }
    std::size_t Committed() const { return committed; }
    Clock::time_point Started() const { return started; }
    void SetLimits(const PassRunLimits& value) { limits = value; }
    const PassRunLimits& Limits() const { return limits; }
    PassRunStatistics TakeStatistics() {
        auto taken = stats;
        stats = {};
        return taken;
    }

private:
    void end(PassRunEnd why) {
        active = false;
        cooling = true;
        ++stats.ends[static_cast<std::size_t>(why)];
    }

    bool active = false;
    bool cooling = false;
    std::uint64_t hint = 0;
    std::size_t committed = 0;
    Clock::time_point started{};
    PassRunLimits limits{};
    PassRunStatistics stats{};
};

}

#endif
