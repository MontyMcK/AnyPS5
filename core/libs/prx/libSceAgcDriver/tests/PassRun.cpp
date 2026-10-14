#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/PassRun.hpp"
#include <chrono>
#include <cstddef>
#include <deque>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// The pass run state machine of live `parallel_commit` (PassRun.hpp), driven the way
// DrawPipeline::run drives it: BeforeCommit with the queue's front and the item behind it, then
// AfterCommit with the new front, the recorder's pass state and whether others wait.

namespace {

using AgcDriver::DriverDetail::PassRunEnd;
using AgcDriver::DriverDetail::PassRunItem;
using AgcDriver::DriverDetail::PassRunLimits;
using AgcDriver::DriverDetail::PassRunTracker;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("pass run check failed: " + what);
}

PassRunItem draw(std::uint64_t hint) {
    return {true, hint};
}

PassRunItem other() {
    return {false, 0};
}

std::size_t ended(PassRunTracker& tracker, PassRunEnd why) {
    return static_cast<std::size_t>(tracker.TakeStatistics().ends[static_cast<std::size_t>(why)]);
}

// Commits a queue the way the committer does (one item per commit) and returns, per commit,
// whether it was inside a run. `continuable(i)`: the recorder's pass after commit i.
template<typename Continuable>
std::vector<bool> commitQueue(PassRunTracker& tracker, std::deque<PassRunItem> queue, Continuable continuable, Clock::time_point now = Clock::time_point{}) {
    std::vector<bool> inRun;
    std::size_t index = 0;
    while (!queue.empty()) {
        const auto front = queue.front();
        const auto next = queue.size() > 1 ? std::optional<PassRunItem>(queue[1]) : std::nullopt;
        inRun.push_back(tracker.BeforeCommit(front, next, now));
        queue.pop_front();
        const auto queued = queue.empty() ? std::nullopt : std::optional<PassRunItem>(queue.front());
        static_cast<void>(tracker.AfterCommit(1, queued, continuable(index), false, now));
        ++index;
    }
    return inRun;
}

void noRunWithoutAContinuingDraw() {
    PassRunTracker tracker;
    const auto now = Clock::time_point{};
    require(!tracker.BeforeCommit(other(), draw(1), now), "a non-draw starts no run");
    require(!tracker.BeforeCommit(draw(1), std::nullopt, now), "a draw with nothing queued behind it starts no run");
    require(!tracker.BeforeCommit(draw(1), other(), now), "a draw followed by a non-draw starts no run");
    require(!tracker.BeforeCommit(draw(1), draw(2), now), "a draw followed by another pass starts no run");
    require(!tracker.Active() && tracker.TakeStatistics().runs == 0, "no run counted");
    require(!tracker.AfterCommit(1, draw(1), true, false, now).has_value(), "AfterCommit outside a run does nothing");
}

void runStartsAtAHeadAndContinues() {
    PassRunTracker tracker;
    const auto now = Clock::time_point{};
    require(tracker.BeforeCommit(draw(7), draw(7), now), "a draw followed by its pass starts a run");
    require(tracker.Active() && tracker.Hint() == 7 && tracker.Committed() == 0, "the run holds the head's hint");
    require(!tracker.AfterCommit(1, draw(7), true, true, now).has_value(), "the next draw of the pass continues the run");
    require(tracker.BeforeCommit(draw(7), draw(7), now), "inside the run the next commit belongs to it");
    require(tracker.Committed() == 1, "one commit counted");
    const auto why = tracker.AfterCommit(1, draw(7), true, false, now);
    require(!why.has_value(), "still continuing");
    const auto stats = tracker.TakeStatistics();
    require(stats.runs == 1 && stats.draws == 2, "one run of two draws so far");
    require(stats.contendedContinuations == 1, "the contended continuation is counted");
    require(stats.deferredStarveChecks == 2, "every continuation defers the starve check");
}

void runEndsForEachReason() {
    const auto now = Clock::time_point{};
    {
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(tracker.AfterCommit(1, other(), true, false, now) == PassRunEnd::NotDraw && !tracker.Active(), "a non-draw ends the run");
        require(ended(tracker, PassRunEnd::NotDraw) == 1, "counted as not a draw");
    }
    {
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(tracker.AfterCommit(1, std::nullopt, true, false, now) == PassRunEnd::NotDraw, "an empty queue ends the run");
    }
    {
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(tracker.AfterCommit(1, draw(2), true, false, now) == PassRunEnd::OtherPass, "another pass ends the run");
        require(ended(tracker, PassRunEnd::OtherPass) == 1, "counted as another pass");
    }
    {
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(tracker.AfterCommit(1, draw(1), false, false, now) == PassRunEnd::PassClosed, "a pass left not continuable ends the run");
        require(ended(tracker, PassRunEnd::PassClosed) == 1, "counted as pass closed");
    }
    {
        PassRunTracker tracker;
        tracker.SetLimits({std::chrono::microseconds(300), 3});
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(!tracker.AfterCommit(1, draw(1), true, false, now).has_value(), "first of three");
        require(!tracker.AfterCommit(1, draw(1), true, false, now).has_value(), "second of three");
        require(tracker.AfterCommit(1, draw(1), true, false, now) == PassRunEnd::Cap, "the draw cap ends the run");
        require(ended(tracker, PassRunEnd::Cap) == 1, "counted as cap");
    }
    {
        PassRunTracker tracker;
        tracker.SetLimits({std::chrono::microseconds(300), 64});
        tracker.BeforeCommit(draw(1), draw(1), now);
        require(!tracker.AfterCommit(1, draw(1), true, false, now + std::chrono::microseconds(299)).has_value(), "inside the budget");
        require(tracker.AfterCommit(1, draw(1), true, false, now + std::chrono::microseconds(300)) == PassRunEnd::Budget, "the budget ends the run");
        require(ended(tracker, PassRunEnd::Budget) == 1, "counted as budget");
    }
    {
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        tracker.Abandon(PassRunEnd::NotDraw);
        require(!tracker.Active(), "an abandoned run is over");
        tracker.Abandon(PassRunEnd::NotDraw);
        require(ended(tracker, PassRunEnd::NotDraw) == 1, "abandoning twice counts once");
    }
    {
        // A cooldown is spent by any commit outside a run, so a non-qualifying commit after a run
        // leaves the next qualifying head free to start one.
        PassRunTracker tracker;
        tracker.BeforeCommit(draw(1), draw(1), now);
        static_cast<void>(tracker.AfterCommit(1, other(), true, false, now));
        require(!tracker.BeforeCommit(other(), draw(2), now), "a non-draw after the run");
        require(tracker.BeforeCommit(draw(2), draw(2), now), "the next head starts a run");
        require(tracker.TakeStatistics().cooldowns == 0, "no cooldown was needed");
    }
}

void reasonsAreCheckedInOrder() {
    // The queue's next item decides before the recorder's state, which decides before the limits:
    // a run never continues into another pass because it still had budget, and a closed pass ends
    // a run whatever the limits say.
    const auto now = Clock::time_point{};
    PassRunTracker tracker;
    tracker.SetLimits({std::chrono::microseconds(1), 1});
    tracker.BeforeCommit(draw(1), draw(1), now);
    require(tracker.AfterCommit(1, draw(2), false, false, now + std::chrono::seconds(1)) == PassRunEnd::OtherPass, "another pass is reported first");
    require(!tracker.BeforeCommit(draw(1), draw(1), now), "the commit right after a run stays outside one");
    require(tracker.BeforeCommit(draw(1), draw(1), now), "the next one may start a run again");
    require(tracker.AfterCommit(1, draw(1), false, false, now + std::chrono::seconds(1)) == PassRunEnd::PassClosed, "a closed pass before the limits");
}

void forkedCommitsCountTowardTheCap() {
    const auto now = Clock::time_point{};
    PassRunTracker tracker;
    tracker.SetLimits({std::chrono::milliseconds(10), 8});
    tracker.BeforeCommit(draw(3), draw(3), now);
    require(!tracker.AfterCommit(5, draw(3), true, false, now).has_value(), "a forked run of five continues");
    require(tracker.Committed() == 5, "five commits counted");
    require(tracker.AfterCommit(3, draw(3), true, false, now) == PassRunEnd::Cap, "three more reach the cap of eight");
    const auto stats = tracker.TakeStatistics();
    require(stats.draws == 8 && stats.runs == 1, "eight draws in one run");
}

void queuesSplitIntoRuns() {
    PassRunTracker tracker;
    // Two passes, a non-draw, a lone draw, then a pass whose second draw leaves it not
    // continuable (it writes memory), then that pass again.
    const std::deque<PassRunItem> queue{draw(1), draw(1), draw(1), draw(2), draw(2), other(), draw(3), draw(4), draw(4), draw(4), draw(4)};
    const auto inRun = commitQueue(tracker, queue, [](std::size_t index) { return index != 8; });
    // Runs: {1,1,1}; pass 2's head is the cooldown commit after it and its last draw has no
    // successor in the pass; {4,4} closed after its second draw; the next draw of pass 4 is the
    // cooldown commit and the last one has nothing behind it.
    const std::vector<bool> expected{true, true, true, false, false, false, false, true, true, false, false};
    require(inRun == expected, "commits inside runs");
    const auto stats = tracker.TakeStatistics();
    require(stats.runs == 2, "two runs, got " + std::to_string(stats.runs));
    require(stats.draws == 5, "five draws inside runs, got " + std::to_string(stats.draws));
    require(stats.cooldowns == 2, "two cooldown commits, got " + std::to_string(stats.cooldowns));
    require(stats.ends[static_cast<std::size_t>(PassRunEnd::OtherPass)] == 1, "one run ended by another pass");
    require(stats.ends[static_cast<std::size_t>(PassRunEnd::NotDraw)] == 0, "no run ended by a non-draw");
    require(stats.ends[static_cast<std::size_t>(PassRunEnd::PassClosed)] == 1, "one run ended by a closed pass");
    require(!tracker.Active(), "the queue's end leaves no run active");
}

void limitsDefaults() {
    const PassRunLimits limits;
    require(limits.budget == std::chrono::microseconds(300) && limits.maxDraws == 64, "default limits");
    require(std::string(AgcDriver::DriverDetail::PassRunEndName(PassRunEnd::Budget)) == "budget", "end names");
    require(std::string(AgcDriver::DriverDetail::PassRunEndName(PassRunEnd::FlagOff)) == "flag off", "flag off end name");
    require(std::string(AgcDriver::DriverDetail::PassRunEndName(PassRunEnd::Count)) == "?", "out of range end name");
}

}

int main() {
    try {
        noRunWithoutAContinuingDraw();
        runStartsAtAHeadAndContinues();
        runEndsForEachReason();
        reasonsAreCheckedInOrder();
        forkedCommitsCountTowardTheCap();
        queuesSplitIntoRuns();
        limitsDefaults();
        std::cout << "Pass runs: starts, continuations, every end reason, forked counts and queue splits passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
