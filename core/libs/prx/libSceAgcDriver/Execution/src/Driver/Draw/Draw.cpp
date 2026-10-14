#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GpuCheckpoints.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawSpeculation.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

std::vector<std::shared_ptr<DispatchVariant>>& RebasedProbes();
bool verifyRebasedHits();

struct DrawForkJob final : DrawPipeline::ForkJob {
    std::shared_ptr<VulkanDevice> device;
    const Graphics::State* state = nullptr;
    const Pm4::DrawParameters* draw = nullptr;
    std::span<const Graphics::CompiledShader> stages;
    std::span<const Graphics::GuestMemorySnapshot> snapshots;
    std::shared_ptr<Graphics::DrawPrebuild> prebuilt;
    const std::vector<std::shared_ptr<DispatchVariant>>* recipeStages = nullptr;
    std::uint64_t drawKey = 0;
    std::uint32_t queue = 0;
    std::uint32_t packet = 0;
    std::uint64_t epoch = 0;
    std::shared_ptr<Graphics::ForkedDraw> ahead;
    std::shared_ptr<const Graphics::ForkStamp> aheadStamp;
    std::shared_ptr<const Graphics::Context> aheadContext;
    std::shared_ptr<const DrawRecipe> aheadRecipe;
    std::weak_ptr<void> owner;
    std::shared_ptr<Graphics::DrawSpeculation> speculation;
    int streamState = 0;
};

namespace {

struct CommitTimes {
    std::atomic<std::uint64_t> commits{0};
    std::atomic<std::uint64_t> lambdaNs{0};
    std::atomic<std::uint64_t> lockNs{0};
    std::atomic<std::uint64_t> labelsNs{0};
    std::atomic<std::uint64_t> writersNs{0};
    std::atomic<std::uint64_t> drawNs{0};
    std::atomic<std::uint64_t> submitNs{0};
    std::atomic<std::uint64_t> forked{0};
    std::atomic<std::uint64_t> forkedNs{0};
    std::atomic<std::uint64_t> runs{0};
    std::atomic<std::uint64_t> runBuildNs{0};
    std::atomic<std::int64_t> lastReport{0};
};

CommitTimes& Commits() {
    static auto* times = new CommitTimes();
    return *times;
}

struct AheadStatistics {
    std::atomic<std::uint64_t> attempted{0};
    std::atomic<std::uint64_t> ready{0};
    std::atomic<std::uint64_t> refused{0};
    std::atomic<std::uint64_t> buildNs{0};
    std::atomic<std::uint64_t> used{0};
    std::atomic<std::uint64_t> usedNs{0};
    std::atomic<std::uint64_t> stale{0};
    std::mutex mutex;
    std::map<std::string, std::uint64_t> skipped;
    std::map<std::string, std::uint64_t> staleReasons;
};

AheadStatistics& Ahead() {
    static auto* stats = new AheadStatistics();
    return *stats;
}

void countAhead(std::map<std::string, std::uint64_t> AheadStatistics::*table, const char* reason) {
    auto& stats = Ahead();
    std::lock_guard lock(stats.mutex);
    ++(stats.*table)[reason];
}

struct PublishedSeed {
    std::shared_ptr<const Graphics::ForkSeed> seed;
    std::uint64_t token = 0;
    std::uint64_t hint = 0;
};

PublishedSeed& Published() {
    static auto* published = new PublishedSeed();
    return *published;
}

void publishSeed(std::uint64_t token, std::uint64_t hint) {
    auto& published = Published();
    published.seed = Graphics::TakeForkSeed();
    published.token = token;
    published.hint = hint;
}

struct StreamStatistics {
    std::atomic<std::uint64_t> windows{0};
    std::atomic<std::uint64_t> scheduled{0};
    std::atomic<std::uint64_t> started{0};
    std::atomic<std::uint64_t> completed{0};
    std::atomic<std::uint64_t> refused{0};
    std::atomic<std::uint64_t> cancelled{0};
    std::atomic<std::uint64_t> used{0};
    std::atomic<std::uint64_t> usedNs{0};
    std::atomic<std::uint64_t> normal{0};
    std::atomic<std::uint64_t> normalNs{0};
    std::atomic<std::uint64_t> stale{0};
    std::atomic<std::uint64_t> discarded{0};
    std::atomic<std::uint64_t> waitNs{0};
    std::atomic<std::uint64_t> pauses{0};
    std::atomic<std::uint64_t> pauseNs{0};
    std::mutex mutex;
    std::map<std::string, std::uint64_t> staleReasons;
    std::map<std::string, std::uint64_t> pauseReasons;
};

StreamStatistics& Streams() {
    static auto* stats = new StreamStatistics();
    return *stats;
}

std::size_t StreamAhead() {
    static const std::size_t ahead = [] {
        const char* text = std::getenv("APS5_FORK_STREAM_AHEAD");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 12ull;
        return static_cast<std::size_t>(std::clamp<unsigned long long>(value, 1, 64));
    }();
    return ahead;
}

std::int64_t StreamBudgetNs() {
    static const std::int64_t budget = [] {
        const char* text = std::getenv("APS5_FORK_STREAM_US");
        const auto value = text != nullptr ? std::strtoull(text, nullptr, 10) : 1000ull;
        return static_cast<std::int64_t>(std::clamp<unsigned long long>(value, 50, 20000)) * 1000;
    }();
    return budget;
}

class DrawStream {
public:
    enum State : int { None = 0, Scheduled, Running, Done, Cancelled };
    struct Entry {
        std::shared_ptr<DrawForkJob> job;
        std::shared_ptr<const Graphics::ForkSeed> seed;
    };

    static DrawStream& Get() {
        static auto* stream = new DrawStream();
        return *stream;
    }

    void Begin(std::shared_ptr<const Graphics::Context> windowContext) {
        std::lock_guard lock(mutex);
        while (threads.size() < DrawPipeline::ForkWorkers()) threads.emplace_back([this, slot = threads.size() + 1] { work(slot); });
        context = std::move(windowContext);
        entries.clear();
        next = 0;
        open = true;
        wake.notify_all();
    }

    bool Schedule(const std::shared_ptr<DrawForkJob>& job, const std::shared_ptr<const Graphics::ForkSeed>& seed) {
        std::lock_guard lock(mutex);
        if (job->streamState != None || job->ahead != nullptr) return false;
        job->streamState = Scheduled;
        entries.push_back({job, seed});
        wake.notify_all();
        return true;
    }

    int Await(DrawForkJob& job, bool waitScheduled) {
        std::unique_lock lock(mutex);
        if (job.streamState == Scheduled && !waitScheduled) {
            job.streamState = Cancelled;
            return Cancelled;
        }
        waitWatched(lock, "await", [&] { return job.streamState != Scheduled && job.streamState != Running; });
        return job.streamState;
    }

    std::shared_ptr<Graphics::ForkedDraw> Take(DrawForkJob& job) {
        std::lock_guard lock(mutex);
        return std::move(job.ahead);
    }

    bool Pause(const char* reason) {
        std::unique_lock lock(mutex);
        if (!open) return false;
        open = false;
        Graphics::SetForkPausePending(true);
        waitWatched(lock, reason, [&] { return running == 0; });
        Graphics::SetForkPausePending(false);
        return true;
    }

    void Resume() {
        std::lock_guard lock(mutex);
        if (open) return;
        open = true;
        wake.notify_all();
    }

    void End() {
        std::unique_lock lock(mutex);
        open = false;
        Graphics::SetForkPausePending(true);
        waitWatched(lock, "window end", [&] { return running == 0; });
        Graphics::SetForkPausePending(false);
        for (auto index = next; index < entries.size(); ++index) {
            if (entries[index].job->streamState == Scheduled || entries[index].job->streamState == Cancelled) entries[index].job->streamState = None;
        }
        for (const auto& entry : entries) {
            if (entry.job->streamState == Done && entry.job->ahead != nullptr) Streams().discarded.fetch_add(1, std::memory_order_relaxed);
        }
        entries.clear();
        next = 0;
        context.reset();
    }

private:
    template<typename Predicate>
    void waitWatched(std::unique_lock<std::mutex>& lock, const char* reason, Predicate&& ready) {
        const auto started = std::chrono::steady_clock::now();
        while (!done.wait_for(lock, std::chrono::milliseconds(50), ready)) {
            static std::atomic<std::uint64_t> reports{0};
            const auto count = reports.fetch_add(1, std::memory_order_relaxed);
            if (count >= 20 && count % 100 != 0) continue;
            const auto now = std::chrono::steady_clock::now();
            std::string slots;
            for (std::size_t slot = 0; slot < slotStarted.size(); ++slot) {
                const auto at = slotStarted[slot].load(std::memory_order_relaxed);
                if (at == 0) continue;
                slots += " slot " + std::to_string(slot) + " running " + std::to_string((now.time_since_epoch().count() - at) / 1000000) + " ms";
            }
            std::fprintf(stderr, "[fork] committer waiting %lld ms for stream workers (%s): running %zu, open %d, scheduled %zu, next %zu, pause pending %d;%s\n", static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - started).count()), reason, running, open ? 1 : 0, entries.size(), next, Graphics::ForkPausePending() ? 1 : 0, slots.c_str());
        }
    }

    void work(std::size_t slot) {
        PinWorkerThread("fork stream");
        for (;;) {
            Entry entry;
            std::shared_ptr<const Graphics::Context> windowContext;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return open && next < entries.size(); });
                entry = entries[next++];
                if (entry.job->streamState != Scheduled) continue;
                entry.job->streamState = Running;
                windowContext = context;
                ++running;
                if (slot < slotStarted.size()) slotStarted[slot].store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed);
            }
            auto& stats = Streams();
            stats.started.fetch_add(1, std::memory_order_relaxed);
            std::shared_ptr<Graphics::ForkedDraw> forked;
            std::shared_ptr<const Graphics::ForkStamp> stamp;
            try {
                if (const auto owner = entry.job->owner.lock(); owner != nullptr && windowContext != nullptr) {
                    auto& job = *entry.job;
                    DrawPipeline::FollowEpoch(job.epoch);
                    GuestMemory::SetCurrentPacket(job.packet, job.queue);
                    stamp = Graphics::TakeForkStamp(*windowContext);
                    forked = Graphics::ForkDrawSpeculated(*windowContext, *job.state, *job.draw, job.stages, job.snapshots, job.prebuilt, job.recipeStages->empty() ? nullptr : &job.aheadRecipe, entry.seed.get(), true, slot, job.speculation);
                }
            } catch (...) {
                forked.reset();
            }
            std::lock_guard lock(mutex);
            if (forked != nullptr && Graphics::ForkedDrawRefusal(*forked) == nullptr) {
                entry.job->ahead = std::move(forked);
                entry.job->aheadStamp = std::move(stamp);
                entry.job->aheadContext = windowContext;
                stats.completed.fetch_add(1, std::memory_order_relaxed);
            } else {
                stats.refused.fetch_add(1, std::memory_order_relaxed);
            }
            entry.job->streamState = Done;
            --running;
            if (slot < slotStarted.size()) slotStarted[slot].store(0, std::memory_order_relaxed);
            done.notify_all();
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    bool open = false;
    std::size_t running = 0;
    std::vector<Entry> entries;
    std::size_t next = 0;
    std::shared_ptr<const Graphics::Context> context;
    std::vector<std::thread> threads;
    std::array<std::atomic<std::int64_t>, 16> slotStarted{};
};

std::string takeStreamReport() {
    auto& stats = Streams();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto perCount = [](unsigned long long ns, unsigned long long count) { return count != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(count) : 0.0; };
    const auto windows = take(stats.windows);
    const auto scheduled = take(stats.scheduled);
    const auto started = take(stats.started);
    const auto completed = take(stats.completed);
    const auto refused = take(stats.refused);
    const auto cancelled = take(stats.cancelled);
    const auto used = take(stats.used);
    const auto usedNs = take(stats.usedNs);
    const auto normal = take(stats.normal);
    const auto normalNs = take(stats.normalNs);
    const auto stale = take(stats.stale);
    const auto discarded = take(stats.discarded);
    const auto waitNs = take(stats.waitNs);
    const auto pauses = take(stats.pauses);
    const auto pauseNs = take(stats.pauseNs);
    char head[512];
    std::snprintf(head, sizeof(head), "stream: windows %llu, scheduled %llu, started %llu, completed %llu, refused %llu, cancelled %llu, used %llu at %.1f us/commit, normal %llu at %.1f us/commit, carried %llu, waits %.1f ms, pauses %llu (%.1f ms), stale %llu (", windows, scheduled, started, completed, refused, cancelled, used, perCount(usedNs, used), normal, perCount(normalNs, normal), discarded, static_cast<double>(waitNs) / 1e6, pauses, static_cast<double>(pauseNs) / 1e6, stale);
    std::string line = head;
    std::lock_guard lock(stats.mutex);
    for (const auto& [reason, count] : stats.staleReasons) line += " " + reason + " " + std::to_string(count);
    line += "), pauses by reason:";
    for (const auto& [reason, count] : stats.pauseReasons) line += " " + reason + " " + std::to_string(count);
    stats.staleReasons.clear();
    stats.pauseReasons.clear();
    return line;
}

std::string takeAheadReport() {
    auto& stats = Ahead();
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto attempted = take(stats.attempted);
    const auto ready = take(stats.ready);
    const auto refused = take(stats.refused);
    const auto buildNs = take(stats.buildNs);
    const auto used = take(stats.used);
    const auto usedNs = take(stats.usedNs);
    const auto stale = take(stats.stale);
    char head[256];
    std::snprintf(head, sizeof(head), "build-ahead: attempted %llu, built %llu at %.1f us, refused %llu, used %llu at %.1f us/commit, stale %llu (", attempted, ready + refused, ready + refused != 0 ? static_cast<double>(buildNs) / 1000.0 / static_cast<double>(ready + refused) : 0.0, refused, used, used != 0 ? static_cast<double>(usedNs) / 1000.0 / static_cast<double>(used) : 0.0, stale);
    std::string line = head;
    std::lock_guard lock(stats.mutex);
    for (const auto& [reason, count] : stats.staleReasons) line += " " + reason + " " + std::to_string(count);
    line += "), not built:";
    for (const auto& [reason, count] : stats.skipped) line += " " + reason + " " + std::to_string(count);
    stats.staleReasons.clear();
    stats.skipped.clear();
    return line;
}

bool forkTimed() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile && Graphics::ForkStatsEnabled();
}

std::uint64_t nanosSince(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

void reportForks() {
    if (!forkTimed()) return;
    auto& times = Commits();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = times.lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        times.lastReport.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 10'000'000'000ll || !times.lastReport.compare_exchange_strong(last, now)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
    const auto commits = take(times.commits);
    const auto lambdaNs = take(times.lambdaNs);
    const auto lockNs = take(times.lockNs);
    const auto labelsNs = take(times.labelsNs);
    const auto writersNs = take(times.writersNs);
    const auto drawNs = take(times.drawNs);
    const auto submitNs = take(times.submitNs);
    const auto forked = take(times.forked);
    const auto forkedNs = take(times.forkedNs);
    const auto runs = take(times.runs);
    const auto runBuildNs = take(times.runBuildNs);
    const auto perCommit = [&](std::uint64_t ns) { return commits != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(commits) : 0.0; };
    const auto rest = lambdaNs > lockNs + labelsNs + writersNs + drawNs + submitNs ? lambdaNs - lockNs - labelsNs - writersNs - drawNs - submitNs : 0;
    const auto pipelineRuns = DrawPipeline::Queue0().TakeForkRuns();
    const auto graphics = Graphics::TakeForkReport();
    const auto ahead = takeAheadReport() + "; " + takeStreamReport();
    std::fprintf(stderr, "[fork] (10 s) %s; %s; parallel builds %llu runs at %.1f us each; %llu forked commits at %.1f us each; %s; normal commits %llu at %.1f us, us per commit outside Graphics::Draw: lock %.2f, labels %.2f, writers %.2f, submit check %.2f, rest %.2f (Graphics::Draw %.1f)\n", pipelineRuns.c_str(), ahead.c_str(), static_cast<unsigned long long>(runs), runs != 0 ? static_cast<double>(runBuildNs) / 1000.0 / static_cast<double>(runs) : 0.0, static_cast<unsigned long long>(forked), forked != 0 ? static_cast<double>(forkedNs) / 1000.0 / static_cast<double>(forked) : 0.0, graphics.c_str(), static_cast<unsigned long long>(commits), perCommit(lambdaNs), perCommit(lockNs), perCommit(labelsNs), perCommit(writersNs), perCommit(submitNs), perCommit(rest), perCommit(drawNs));
}

std::uint64_t forkPassHint(const Graphics::State& graphics) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto mix = [&](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    for (const auto& color : graphics.colors) {
        mix(color.address);
        mix(static_cast<std::uint64_t>(color.format));
        mix(color.extent.width);
        mix(color.extent.height);
        mix(color.mip);
        mix(color.depthSlice);
        mix(color.layers);
        mix(color.exportIndex);
    }
    if (graphics.depth) {
        mix(graphics.depth->address);
        mix(static_cast<std::uint64_t>(graphics.depth->format));
        mix(graphics.depth->extent.width);
        mix(graphics.depth->extent.height);
        mix(graphics.depth->arraySlice);
        mix(graphics.depth->layers);
    }
    mix(graphics.blends.size());
    mix(graphics.renderExtent.width);
    mix(graphics.renderExtent.height);
    return hash;
}

}

class DrawForkRunner final : public DrawPipeline::ForkRunner {
public:
    explicit DrawForkRunner(Driver& driver) : driver(driver) {}

    std::size_t Run(std::span<DrawPipeline::ForkJob* const> jobs, const std::function<bool(std::size_t, const DrawPipeline::Commit*)>& commit) override {
        std::vector<DrawForkJob*> draws;
        draws.reserve(jobs.size());
        for (auto* job : jobs) {
            auto* draw = static_cast<DrawForkJob*>(job);
            if ((!draws.empty() && draw->epoch != draws.front()->epoch) || draw->ahead != nullptr) break;
            draws.push_back(draw);
        }
        if (draws.size() < 2) {
            Graphics::CountForkCut("epoch boundary");
            return 0;
        }
        const auto hint = draws.front()->passHint;
        if (const auto found = backoff.find(hint); found != backoff.end() && found->second != 0) {
            --found->second;
            Graphics::CountForkCut("refusal backoff");
            return 0;
        }
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::unique_lock gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = driver.device.Load();
        auto* recorder = Graphics::Recorder::Active();
        if (localDevice == nullptr || recorder == nullptr || !recorder->PassContinuable()) {
            Graphics::CountForkCut("no open pass");
            return 0;
        }
        if (std::any_of(draws.begin(), draws.end(), [&](const DrawForkJob* job) { return job->device != localDevice; })) {
            Graphics::CountForkCut("device replaced");
            return 0;
        }
        const auto runEpoch = draws.front()->epoch;
        const bool timed = forkTimed();
        std::vector<std::shared_ptr<Graphics::ForkedDraw>> forked(draws.size());
        std::vector<std::shared_ptr<const DrawRecipe>> built(draws.size());
        std::optional<Graphics::Context> context;
        std::shared_ptr<const Graphics::ForkStamp> stamp;
        try {
            DrawPipeline::FollowEpoch(runEpoch);
            const auto seed = Graphics::TakeForkSeed();
            context.emplace(localDevice->DrawContext());
            const auto buildStart = std::chrono::steady_clock::now();
            std::atomic<std::size_t> firstRefused{draws.size()};
            DrawPipeline::RunForkJobs(draws.size(), [&](std::size_t index, std::size_t slot) {
                if (index > firstRefused.load(std::memory_order_relaxed)) return;
                const auto& job = *draws[index];
                DrawPipeline::FollowEpoch(job.epoch);
                GuestMemory::SetCurrentPacket(job.packet, job.queue);
                forked[index] = Graphics::ForkDrawSpeculated(*context, *job.state, *job.draw, job.stages, job.snapshots, job.prebuilt, job.recipeStages->empty() ? nullptr : &built[index], seed.get(), job.epoch == runEpoch, slot, job.speculation);
                if (Graphics::ForkedDrawRefusal(*forked[index]) == nullptr) return;
                auto current = firstRefused.load(std::memory_order_relaxed);
                while (index < current && !firstRefused.compare_exchange_weak(current, index, std::memory_order_relaxed)) {
                }
            });
            if (firstRefused.load(std::memory_order_relaxed) == 0) {
                if (backoff.size() > 256) backoff.clear();
                backoff[hint] = 16;
            }
            if (timed) {
                Commits().runs.fetch_add(1, std::memory_order_relaxed);
                Commits().runBuildNs.fetch_add(nanosSince(buildStart), std::memory_order_relaxed);
            }
            stamp = Graphics::TakeForkStamp(*context);
        } catch (...) {
            Graphics::CountForkCut("runner setup");
            return 0;
        }
        std::size_t handled = 0;
        for (std::size_t index = 0; index < draws.size(); ++index) {
            auto* job = draws[index];
            bool cut = false;
            const auto commitStart = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            const DrawPipeline::Commit body = [&] { driver.commitForked(*job, *context, forked[index].get(), *stamp, built[index], cut); };
            const bool committed = commit(index, &body);
            forked[index].reset();
            built[index].reset();
            ++handled;
            if (timed && committed && !cut) {
                Commits().forked.fetch_add(1, std::memory_order_relaxed);
                Commits().forkedNs.fetch_add(nanosSince(commitStart), std::memory_order_relaxed);
            }
            if (!committed || cut) break;
        }
        reportForks();
        return handled;
    }

    std::size_t Stream(const std::function<std::vector<std::shared_ptr<DrawPipeline::ForkJob>>(std::size_t)>& peek, const std::function<bool(const DrawPipeline::Commit*)>& commit) override {
        auto queued = peek(1);
        if (queued.empty() || !queued.front()->eligible) return 0;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::unique_lock gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = driver.device.Load();
        if (localDevice == nullptr || Graphics::Recorder::Active() == nullptr) return 0;
        std::shared_ptr<const Graphics::Context> context;
        try {
            context = std::make_shared<const Graphics::Context>(localDevice->DrawContext());
        } catch (...) {
            return 0;
        }
        auto& stream = DrawStream::Get();
        auto& stats = Streams();
        const bool timed = forkTimed();
        stats.windows.fetch_add(1, std::memory_order_relaxed);
        stream.Begin(context);
        const auto windowStart = std::chrono::steady_clock::now();
        const auto ahead = StreamAhead();
        const bool waitScheduled = Graphics::ForkStreamWaitEnabled();
        const std::function<void(const char*)> hook = [&](const char* reason) {
            const auto pauseStart = std::chrono::steady_clock::now();
            if (!stream.Pause(reason)) return;
            stats.pauses.fetch_add(1, std::memory_order_relaxed);
            stats.pauseNs.fetch_add(nanosSince(pauseStart), std::memory_order_relaxed);
            std::lock_guard lock(stats.mutex);
            ++stats.pauseReasons[reason];
        };
        const Graphics::ForkQuiesceScope quiesceScope(&hook);
        const bool lazy = Graphics::ForkStreamLazyEnabled();
        std::shared_ptr<const Graphics::ForkSeed> seed;
        std::uint64_t seedHint = 0;
        std::uint64_t seedEpoch = 0;
        bool committedOne = false;
        std::uint64_t lastHint = 0;
        std::uint64_t lastEpoch = 0;
        std::size_t handled = 0;
        for (;;) {
            queued = peek(ahead + 1);
            if (queued.empty()) break;
            auto job = std::static_pointer_cast<DrawForkJob>(queued.front());
            if (!job->eligible || job->device != localDevice) break;
            if (handled != 0 && static_cast<std::int64_t>(nanosSince(windowStart)) > StreamBudgetNs()) break;
            auto* recorder = Graphics::Recorder::Active();
            if (committedOne && recorder != nullptr && recorder->PassContinuable()) {
                if (seed == nullptr || seedHint != lastHint || seedEpoch != lastEpoch) {
                    seed = Graphics::TakeForkSeed();
                    seedHint = lastHint;
                    seedEpoch = lastEpoch;
                }
                for (const auto& candidate : queued) {
                    auto draw = std::static_pointer_cast<DrawForkJob>(candidate);
                    if (!draw->eligible || draw->passHint != lastHint || draw->epoch != lastEpoch) break;
                    if (stream.Schedule(draw, seed)) stats.scheduled.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const auto waitStart = std::chrono::steady_clock::now();
            const auto state = stream.Await(*job, waitScheduled);
            stats.waitNs.fetch_add(nanosSince(waitStart), std::memory_order_relaxed);
            if (state == DrawStream::Cancelled) stats.cancelled.fetch_add(1, std::memory_order_relaxed);
            auto forked = stream.Take(*job);
            const auto commitStart = std::chrono::steady_clock::now();
            bool committed = false;
            if (forked != nullptr) {
                bool cut = false;
                auto stamp = job->aheadStamp;
                auto forkContext = job->aheadContext != nullptr ? job->aheadContext : context;
                const DrawPipeline::Commit body = [&] { driver.commitForked(*job, *forkContext, forked.get(), *stamp, job->aheadRecipe, cut); };
                committed = commit(&body);
                if (cut) {
                    stats.stale.fetch_add(1, std::memory_order_relaxed);
                    const char* why = Graphics::ForkedDrawCut(*forked);
                    std::lock_guard lock(stats.mutex);
                    ++stats.staleReasons[why != nullptr ? why : "device"];
                } else {
                    stats.used.fetch_add(1, std::memory_order_relaxed);
                    stats.usedNs.fetch_add(nanosSince(commitStart), std::memory_order_relaxed);
                }
            } else {
                if (!lazy) hook("normal commit");
                committed = commit(nullptr);
                stats.normal.fetch_add(1, std::memory_order_relaxed);
                stats.normalNs.fetch_add(nanosSince(commitStart), std::memory_order_relaxed);
            }
            stream.Resume();
            static_cast<void>(committed);
            ++handled;
            committedOne = true;
            lastHint = job->passHint;
            lastEpoch = job->epoch;
        }
        stream.End();
        if (timed) reportForks();
        return handled;
    }

private:
    Driver& driver;
    std::unordered_map<std::uint64_t, std::uint32_t> backoff;
};

DrawVerdict Driver::draw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::string& rejected) {
    const DrawPipeline::TraceScope pipeTrace(DrawPipeline::Trace::QueueDraw, DrawPipeline::Trace::QueueDrawEnd);
    PerformanceTimer timing("Driver.Draw");
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    std::uint64_t captures = 0;
    auto phaseLap = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[DrawRowPrologue] = std::chrono::duration<double, std::milli>(phaseLap - packetStartedAt()).count();

    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    const bool pipelined = DrawPipeline::Active();
    // APS5_SPLIT_DRAW_FRONT=1: the whole draw runs on a preparation worker from a snapshot of the
    // queue's registers; the queue thread only takes the snapshot.
    static const bool frontSplit = std::getenv("APS5_SPLIT_DRAW_FRONT") != nullptr;
    if (frontSplit && pipelined && submission.queue == 0 && DrawPipeline::SplitDraws() && !DrawPipeline::Queue0().OnPrepThread() && !drawParameters.indirect && deferredLabels().labels.empty()) {
        auto snapshot = std::make_shared<QueueState>(queue, QueueState::DrawSnapshot{});
        auto stable = std::make_shared<Submission>();
        stable->serial = submission.serial;
        stable->queue = submission.queue;
        stable->shaders = submission.shaders;
        stable->received = submission.received;
        auto words = std::make_shared<const std::vector<std::uint32_t>>(packet.begin(), packet.end());
        DrawPipeline::Queue0().Prepare([this, snapshot, stable, words] {
            std::string frontRejected;
            static_cast<void>(draw(*snapshot, std::span<const std::uint32_t>(*words), *stable, frontRejected));
        });
        return drawn();
    }
    if (pipelined && drawParameters.indirect) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Indirect);
    bool traceIndirect = false;
    if (const auto verdict = precheckDraw(queue, submission, packet, drawParameters, rejected, traceIndirect)) return *verdict;
    phaseTiming.Phase(DrawRowPrecheck);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;

    static const std::uint64_t dumpTarget = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SHADERS"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const std::uint64_t dumpSlot1 = [] { const char* text = std::getenv("APS5_DUMP_DRAW_SLOT1"); return text ? std::strtoull(text, nullptr, 16) : 0ull; }();

    static const bool lockedPrepare = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr;
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::defer_lock);
    std::shared_ptr<VulkanDevice> localDevice;
    if (lockedPrepare) {

        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    } else if ((localDevice = device.Load()) == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard createLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    timing.Mark("device_setup");
    phaseTiming.Phase(DrawRowVectors);

    const bool useDrawEntries = drawEntries() && !ShaderRecompiler::DebugProbeActive() && dumpTarget == 0 && dumpSlot1 == 0;
    const bool registerKey = useDrawEntries && registerKeyEnabled();
    std::uint64_t drawKey = 0;
    std::shared_ptr<DrawEntry> entry;
    std::shared_ptr<const DrawDecode> decode;
    std::shared_ptr<DrawEntry> dataEntry;
    std::uint64_t shapeKey = 0;
    if (registerKey) {
        const auto keyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        drawKey = drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), &shapeKey);
        std::lock_guard cacheLock(drawCacheMutex);
        ++drawEntryCounters.lookups;
        ++drawEntryCounters.registerKeyLookups;
        if (profile) drawEntryCounters.keyUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - keyStart).count();
        const auto found = drawCache.find(drawKey);
        if (found != drawCache.end()) {
            entry = found->second;
            decode = entry->decode;
        } else {
            ++drawEntryCounters.absent;
            if (traceDrawCache()) {
                auto& counters = drawEntryCounters;
                if (traceKeysEvicted.contains(drawKey)) ++counters.absentEvicted;
                else if (traceKeysSeen.contains(drawKey)) ++counters.absentNeverInserted;
                else if (traceShapes.contains(shapeKey)) ++counters.absentUserWords;
                else ++counters.absentNewRegisters;
            }
            if (drawDataHits() && !drawParameters.indirect) {
                if (const auto shape = drawShapes.find(shapeKey); shape != drawShapes.end()) {
                    const auto kept = drawCache.find(shape->second);
                    if (kept != drawCache.end() && kept->second->decode != nullptr && kept->second->shape == shapeKey) dataEntry = kept->second;
                }
                if (dataEntry != nullptr) ++drawEntryCounters.dataHits;
            }
        }
        if (traceDrawCache()) {
            traceInsert(traceKeysSeen, drawKey);
            traceInsert(traceShapes, shapeKey);
        }
        maybeReportDrawCache(profile);
    }
    phaseTiming.Phase(DrawRowKeyLookupValidate);

    if (dataEntry != nullptr) {
        auto refreshed = std::make_shared<DrawDecode>(*dataEntry->decode);
        for (auto& program : refreshed->programs) {
            readUserWords(queue, program);
            if (program.merged) initializeMerged(queue, program, program.mergedPointer, program.mergedPointerRequired);
        }
        if (traceDrawCache()) {
            static std::atomic<int> printed{0};
            for (std::size_t i = 0; i < refreshed->programs.size() && i < dataEntry->decode->programs.size(); ++i) {
                const auto& before = dataEntry->decode->programs[i].userData;
                const auto& after = refreshed->programs[i].userData;
                if (before == after) continue;
                std::string line = "[draw-cache] user words differ: stage " + std::to_string(i) + " role " + std::to_string(static_cast<int>(dataEntry->decode->roles[i])) + " code 0x";
                char item[96];
                std::snprintf(item, sizeof(item), "%llx words %zu:", static_cast<unsigned long long>(refreshed->programs[i].binary.codeAddress), after.size());
                line += item;
                for (std::size_t w = 0; w < after.size() && w < before.size(); ++w) {
                    if (before[w] == after[w]) continue;
                    std::snprintf(item, sizeof(item), " [%zu] %08x->%08x", w, before[w], after[w]);
                    line += item;
                }
                if (printed.fetch_add(1, std::memory_order_relaxed) < 60) std::fprintf(stderr, "%s\n", line.c_str());
                if (i >= dataEntry->stages.size() || dataEntry->stages[i].empty()) continue;
                const auto& variant = *dataEntry->stages[i].front();
                std::size_t runs = 0, rebasedRuns = 0, equalRuns = 0, wordsCompared = 0, wordsDiffering = 0, inaccessible = 0, shiftedWords = 0;
                std::size_t offset = 0;
                std::uint32_t commonDelta = 0;
                bool oneDelta = true;
                for (std::size_t w = 0; w < after.size() && w < before.size(); ++w) {
                    if (before[w] == after[w]) continue;
                    if (commonDelta == 0) commonDelta = after[w] - before[w];
                    else if (after[w] - before[w] != commonDelta) oneDelta = false;
                }
                for (const auto& [begin, end] : variant.runs) {
                    const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
                    ++runs;
                    auto address = begin;
                    const auto low = static_cast<std::uint32_t>(begin);
                    for (std::size_t w = 0; w < after.size() && w < before.size(); ++w) {
                        if (before[w] == after[w] || low < before[w] || low - before[w] >= 0x100000u) continue;
                        address = begin + static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(after[w] - before[w])));
                        ++rebasedRuns;
                        break;
                    }
                    if (offset + count <= variant.words.size() && GuestMemory::Accessible(reinterpret_cast<const void*>(address), count * 4)) {
                        std::size_t differing = 0;
                        const auto* live = reinterpret_cast<const std::uint32_t*>(address);
                        for (std::size_t k = 0; k < count; ++k) {
                            if (live[k] == variant.words[offset + k]) continue;
                            if (oneDelta && live[k] == variant.words[offset + k] + commonDelta) ++shiftedWords;
                            ++differing;
                            static std::atomic<int> wordsPrinted{0};
                            if (wordsPrinted.fetch_add(1, std::memory_order_relaxed) < 60) std::fprintf(stderr, "[draw-cache] rebase word: run 0x%llx -> 0x%llx word %zu/%zu old %08x new %08x delta %08x (user delta %08x)\n", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(address), k, count, variant.words[offset + k], live[k], live[k] - variant.words[offset + k], commonDelta);
                        }
                        wordsCompared += count;
                        wordsDiffering += differing;
                        equalRuns += differing == 0 ? 1 : 0;
                    } else {
                        ++inaccessible;
                    }
                    offset += count;
                }
                static std::atomic<int> rebasedPrinted{0};
                if (rebasedPrinted.fetch_add(1, std::memory_order_relaxed) < 80) std::fprintf(stderr, "[draw-cache] rebase probe: stage %zu runs %zu rebased %zu equal %zu inaccessible %zu words %zu differing %zu shifted %zu one delta %d\n", i, runs, rebasedRuns, equalRuns, inaccessible, wordsCompared, wordsDiffering, shiftedWords, oneDelta ? 1 : 0);
            }
        }
        if (verifyDrawDataHits()) {
            const auto full = decodeDraw(queue, submission);
            if (!sameDecode(*refreshed, *full)) {
                std::size_t differing = 0;
                while (differing < full->programs.size() && differing < refreshed->programs.size() && full->programs[differing].userData == refreshed->programs[differing].userData) ++differing;
                char message[320];
                std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: a data-only hit's decode differs from the full decode (key 0x%llx, shape 0x%llx, %zu / %zu programs, first with differing user words %zu, path %u / %u, target 0x%llx / 0x%llx)", static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey), refreshed->programs.size(), full->programs.size(), differing, static_cast<unsigned>(refreshed->state.stages.path), static_cast<unsigned>(full->state.stages.path), static_cast<unsigned long long>(refreshed->state.color.address), static_cast<unsigned long long>(full->state.color.address));
                throw std::runtime_error(message);
            }
        }
        decode = std::move(refreshed);
    }
    resolveDrawDecode(queue, submission, decode, registerKey, drawKey, profile);
    const auto& graphics = decode->state;
    const auto& pixel = decode->pixel;
    std::vector<DrawProgram> programs = decode->programs;
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = programs.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    if (!drawParameters.indirect) setMeshIndexBuffer(drawParameters);
    else if (graphics.stages.mesh) setMeshIndexBuffer(Pm4::DrawParameters{drawParameters.indexAddress, std::max(drawParameters.indexCount, 1u), drawParameters.indexSize, 1, 0, drawParameters.indexed});
    const std::vector<Role>& roles = decode->roles;
    phaseTiming.Phase(DrawRowDecode);

    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    if (drawParameters.indirect) {
        auto& indirect = *drawParameters.indirect;
        const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
            const auto word = locate(location);
            if (!word || word->first != 0) return -1;
            return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
        };
        indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
        indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
        indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
    }
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    timing.Mark("prepare");
    phaseTiming.Phase(DrawRowProgramPrepare);

    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos(programs.size());
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads(programs.size());
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i]);
        if (vertexInfos[i]) vertexInfos[i]->paClVsOutCntl = (readRegister(queue.context, 0x207) & (0xffffu | (1u << 21u) | (1u << 22u) | (1u << 23u))) | graphics.layerExports;
    };
    if (!registerKey) {
        for (std::size_t i = 0; i < programs.size(); ++i) decodeVertexInfo(i);
        phaseTiming.Phase(DrawRowDecode);
    }
    std::vector<ShaderRecompiler::RecompileResult> results;
    std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> compiledStages;
    compiledStages.reserve(programs.size());
    std::vector<Graphics::CompiledShader> stages;
    results.reserve(programs.size() + (graphics.rectList ? 2u : 0u));
    stages.reserve(programs.size());
    std::uint32_t pushCursorBytes = 0;

    std::vector<const ShaderRecompiler::RecompileResult*> programResults(programs.size(), nullptr);

    std::vector<StageCapture> stageCaptures(programs.size());
    std::vector<std::shared_ptr<DispatchVariant>> matched(programs.size());
    std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions(programs.size());

    std::vector<std::shared_ptr<DispatchVariant>> fresh(programs.size());

    std::vector<bool> recompiled(programs.size(), false);
    std::vector<std::vector<std::pair<std::shared_ptr<DispatchVariant>, std::vector<ShaderRecompiler::MemoryRegion>>>> dataCandidates(programs.size());
    std::vector<bool> reused(programs.size(), false);
    std::uint64_t dataStagesReused = 0, dataStagesCompiled = 0, dataVerified = 0;
    bool drawHit = false;
    bool verifyHit = false;
    if (verifyRebasedHits()) RebasedProbes().clear();
    lookupDraw(submission, localDevice, graphics, pixel, programs, roles, vertexInfos, useDrawEntries, registerKey, profile, drawKey, entry, matched, matchedRegions, drawHit, verifyHit, phaseTiming, phaseMs, dataEntry, dataCandidates);

    if (registerKey) {

        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (matched[i] != nullptr && drawHit && !verifyDrawRecipe()) {
                if (matched[i]->vertexInfo != nullptr) vertexInfos[i] = *matched[i]->vertexInfo;
                continue;
            }
            if (!dataCandidates[i].empty() && !verifyDrawDataHits()) {
                const auto& info = dataCandidates[i].front().first->vertexInfo;
                if (info != nullptr) vertexInfos[i] = *info;
                continue;
            }
            decodeVertexInfo(i);
            if (matched[i] != nullptr && verifyDrawRecipe() && (vertexInfos[i].has_value() != (matched[i]->vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *matched[i]->vertexInfo)))) {
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) std::fprintf(stderr, "[draw-cache] verify: stage %zu (program 0x%llx) of a hit has a vertex stage info unlike its variant's\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress));
                std::lock_guard cacheLock(drawCacheMutex);
                ++drawEntryCounters.verifyDecodeMismatches;
            }
        }
        phaseTiming.Phase(DrawRowDecode);
    }

    static const bool splitDataHits = std::getenv("APS5_NO_SPLIT_DATA_HITS") == nullptr;
    const bool splitDraw = DrawPipeline::SplitDraws() && pipelined && !drawParameters.indirect && !lockedPrepare && deferredLabels().labels.empty() && dumpTarget == 0 && dumpSlot1 == 0 && !verifyHit && !verifyDrawDataHits() && !verifyRebasedHits() && (splitDataHits || std::all_of(dataCandidates.begin(), dataCandidates.end(), [](const auto& candidates) { return candidates.empty(); }));
    const auto clipControl = readRegister(queue.context, 0x207);
    if (DrawPipeline::SplitDraws() && profile) {
        static std::array<std::atomic<std::uint64_t>, 8> refusals{};
        static std::atomic<std::int64_t> refusalReport{0};
        const bool candidates = !std::all_of(dataCandidates.begin(), dataCandidates.end(), [](const auto& entries) { return entries.empty(); });
        const std::size_t reason = splitDraw ? 0 : !pipelined ? 1 : drawParameters.indirect ? 2 : lockedPrepare ? 3 : !deferredLabels().labels.empty() ? 4 : candidates ? 5 : 6;
        refusals[reason].fetch_add(1, std::memory_order_relaxed);
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        auto last = refusalReport.load(std::memory_order_relaxed);
        if (now - last > 10'000'000'000ll && refusalReport.compare_exchange_strong(last, now)) std::fprintf(stderr, "[draw] split decisions (10 s): split %llu, not pipelined %llu, indirect %llu, locked prepare %llu, deferred labels %llu, data candidates %llu, other %llu%c", static_cast<unsigned long long>(refusals[0].exchange(0)), static_cast<unsigned long long>(refusals[1].exchange(0)), static_cast<unsigned long long>(refusals[2].exchange(0)), static_cast<unsigned long long>(refusals[3].exchange(0)), static_cast<unsigned long long>(refusals[4].exchange(0)), static_cast<unsigned long long>(refusals[5].exchange(0)), static_cast<unsigned long long>(refusals[6].exchange(0)), 10);
    }
    auto splitRejected = std::make_shared<std::string>();
    std::string* rejectedOut = splitDraw ? splitRejected.get() : &rejected;
    const auto splitEpoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
    // A split tail outlives the submission it came from (the queue thread moves on): it keeps what it
    // reads of it.
    auto stableSubmission = std::make_shared<Submission>();
    stableSubmission->serial = submission.serial;
    stableSubmission->queue = submission.queue;
    stableSubmission->shaders = submission.shaders;
    stableSubmission->received = submission.received;
    auto packetWords = std::make_shared<const std::vector<std::uint32_t>>(packet.begin(), packet.end());
    auto tail = [this, &queue, stableSubmission, packetWords, clipControl, rejectedOut, splitRejected, splitDraw, splitEpoch, phaseMsIn = phaseMs, phaseLapIn = phaseLap, capturesIn = captures, drawParameters, pipelined, traceIndirect, gpuLock = std::move(gpuLock), localDevice, useDrawEntries, registerKey, drawKey, entry = std::move(entry), decode = std::move(decode), dataEntry = std::move(dataEntry), shapeKey, programs = std::move(programs), memory = std::move(memory), linked = std::move(linked), vertexInfos = std::move(vertexInfos), decodeReads = std::move(decodeReads), results = std::move(results), compiledStages = std::move(compiledStages), stages = std::move(stages), pushCursorBytes, programResults = std::move(programResults), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), recompiled = std::move(recompiled), dataCandidates = std::move(dataCandidates), reused = std::move(reused), dataStagesReused, dataStagesCompiled, dataVerified, drawHit, verifyHit]() mutable -> DrawVerdict {
    if (splitDraw) DrawPipeline::FollowEpoch(splitEpoch);
    const Submission& submission = *stableSubmission;
    const std::span<const std::uint32_t> packet(*packetWords);
    using Stage = ShaderRecompiler::ShaderStage;
    using Role = ShaderRecompiler::ProgramRole;
    auto phaseMs = phaseMsIn;
    auto phaseLap = phaseLapIn;
    auto captures = capturesIn;
    DrawPhaseTiming phaseTiming{profile, phaseMs, phaseLap};
    PerformanceTimer timing("Driver.DrawTail");
    std::string& rejected = *rejectedOut;
    const auto drawn = [&] {
        phaseTiming.Phase(DrawRowVectors);
        if (!profile) return DrawVerdict::Drawn;
        auto& pending = pendingDrawPhases();
        pending.phases = true;
        pending.captures = captures;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
        return DrawVerdict::Drawn;
    };
    const auto& graphics = decode->state;
    const auto& pixel = decode->pixel;
    const std::vector<Role>& roles = decode->roles;
    const auto setMeshIndexBuffer = [&](const Pm4::DrawParameters& parameters) {
        if (!graphics.stages.mesh) return;
        auto& words = programs.front().userData;
        require(programs.front().firstUserSgpr == 0 && words.size() >= ShaderRecompiler::MeshIndexBufferUserWord + 4, "mesh program lacks the hidden user words");
        const auto descriptor = Graphics::MeshIndexBufferDescriptor(parameters);
        std::copy(descriptor.begin(), descriptor.end(), words.begin() + ShaderRecompiler::MeshIndexBufferUserWord);
    };
    const auto locate = [&](std::uint32_t location) -> std::optional<std::pair<std::size_t, std::size_t>> {
        if (location == 0x280u) return std::nullopt;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
            const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
            if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
        }
        return std::nullopt;
    };
    const auto decodeVertexInfo = [&](std::size_t i) {
        const auto& program = programs[i];
        if (program.binary.stage == Stage::Fragment || roles[i] == Role::GeometryBack) return;
        decodeReads[i].clear();
        vertexInfos[i] = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, program.userData, &decodeReads[i]);
        if (vertexInfos[i]) vertexInfos[i]->paClVsOutCntl = (clipControl & (0xffffu | (1u << 21u) | (1u << 22u) | (1u << 23u))) | graphics.layerExports;
    };
    // Draw shape reuse (DrawShapeReuse.cpp) for the stages this draw captures: not for indirect
    // draws (their stages are captured again per record), debug dumps and probes, or the modes that
    // verify the draw cache against full captures. `registered` are the program regions the stages'
    // ShaderMemory serves from the snapshots.
    const bool shapeReuse = drawShapeReuse() && !drawParameters.indirect && dumpTarget == 0 && dumpSlot1 == 0 && !ShaderRecompiler::DebugProbeActive() && !stampValidate() && !verifyHit && !verifyDrawDataHits() && !verifyRebasedHits();
    const std::vector<ShaderRecompiler::MemoryRegion> registered(shapeReuse ? memory.begin() : memory.end(), memory.end());
    ShaderMemory shaderMemory(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
    // APS5_VERIFY_DRAW_SHAPE_REUSE: a second memory over the same regions on which every captured
    // stage is captured in full again and compared.
    std::optional<ShaderMemory> shapeShadow;
    if (shapeReuse && verifyDrawShapeReuse()) shapeShadow.emplace(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
    static_cast<void>(setMeshIndexBuffer);
    static_cast<void>(locate);

    static const bool indxOffsetSkipFold = std::getenv("APS5_INDX_OFFSET_SKIP_FOLD") != nullptr;
    static const bool indexedOffsetFold = std::getenv("APS5_NO_INDEXED_OFFSET_FOLD") == nullptr;
    const auto fold = [&](const ShaderRecompiler::RecompileResult& main, Pm4::DrawParameters& parameters) {
        if (parameters.indexed && !indexedOffsetFold) return;
        if (main.vertexOffsetSgpr >= 0 && (parameters.firstVertex == 0 || !indxOffsetSkipFold)) {
            const auto offset = drawUserWord(programs.front(), main.vertexOffsetSgpr);
            require(offset <= std::numeric_limits<std::uint32_t>::max() - parameters.firstVertex, "draw vertex offset overflow");
            parameters.firstVertex += offset;
        }
        if (main.instanceOffsetSgpr >= 0) parameters.firstInstance = drawUserWord(programs.front(), main.instanceOffsetSgpr);
    };

    const auto verifyDataStage = [&](std::size_t i, const DispatchVariant& kept) {
        std::vector<ShaderRecompiler::MemoryRegion> regions(stageCaptures[i].regions.begin(), stageCaptures[i].regions.end());
        for (const auto& read : decodeReads[i]) regions.push_back({read.address, std::as_bytes(std::span(read.bytes))});
        std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
        std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
        std::vector<std::uint32_t> words;
        for (const auto& region : regions) {
            runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
            const auto count = region.bytes.size() / sizeof(std::uint32_t);
            const auto offset = words.size();
            words.resize(offset + count);
            std::memcpy(words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
        }
        const auto& compiled = *stageCaptures[i].compiled;
        const char* what = runs != kept.runs ? "runs" : words != kept.words ? "words" : vertexInfos[i].has_value() != (kept.vertexInfo != nullptr) || (vertexInfos[i] && !sameVertexInfo(*vertexInfos[i], *kept.vertexInfo)) ? "vertex stage info" : compiled.variantId != kept.compiled->variantId ? "variant" : compiled.pushConstants != kept.compiled->pushConstants ? "push constants" : nullptr;
        ++dataVerified;
        if (what == nullptr) return;
        char message[320];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_DRAW_DATA_HITS: stage %zu (program 0x%llx) of a data-only hit differs from its full build in its %s (%zu runs / %zu words kept, %zu / %zu fresh; variant 0x%llx / 0x%llx; key 0x%llx, shape 0x%llx)", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), what, kept.runs.size(), kept.words.size(), runs.size(), words.size(), static_cast<unsigned long long>(kept.compiled->variantId), static_cast<unsigned long long>(compiled.variantId), static_cast<unsigned long long>(drawKey), static_cast<unsigned long long>(shapeKey));
        throw std::runtime_error(message);
    };
    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    std::vector<std::uint32_t> pushOffsets(programs.size(), 0);
    std::vector<std::size_t> resultIndex(programs.size(), 0);
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        const auto& program = programs[i];
        pushOffsets[i] = pushCursorBytes;
        if (drawHit) {

            programResults[i] = matched[i]->compiled.get();
            if (programResults[i] != nullptr) Graphics::GpuCheckpoints::NoteVariant(programResults[i]->variantId, programs[i].binary.codeAddress);
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            const auto candidate = std::find_if(dataCandidates[i].begin(), dataCandidates[i].end(), [&](const auto& kept) { return kept.first->pushOffset == pushCursorBytes; });
            if (candidate != dataCandidates[i].end() && !verifyDrawDataHits()) {
                matched[i] = candidate->first;
                matchedRegions[i] = std::move(candidate->second);
                reused[i] = true;
                programResults[i] = matched[i]->compiled.get();
                if (programResults[i] != nullptr) Graphics::GpuCheckpoints::NoteVariant(programResults[i]->variantId, programs[i].binary.codeAddress);
                memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
            } else {
                if (!dataCandidates[i].empty() && !verifyDrawDataHits()) decodeVertexInfo(i);
                compiledStages.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected, shapeReuse, registered, shapeShadow ? &*shapeShadow : nullptr));
                if (!rejected.empty()) return DrawVerdict::Rejected;
                programResults[i] = compiledStages.back().get();
                if (programResults[i] != nullptr) Graphics::GpuCheckpoints::NoteVariant(programResults[i]->variantId, programs[i].binary.codeAddress);
                if (candidate != dataCandidates[i].end()) verifyDataStage(i, *candidate->first);
                if (verifyRebasedHits() && i < RebasedProbes().size() && RebasedProbes()[i] != nullptr) {
                    const auto& probe = *RebasedProbes()[i];
                    const auto& fresh = *programResults[i];
                    const auto& kept = *probe.compiled;
                    std::vector<ShaderRecompiler::MemoryRegion> regions(stageCaptures[i].regions.begin(), stageCaptures[i].regions.end());
                    for (const auto& read : decodeReads[i]) regions.push_back({read.address, std::as_bytes(std::span(read.bytes))});
                    std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
                    std::vector<std::pair<std::uint64_t, std::uint64_t>> runs;
                    std::vector<std::uint32_t> words;
                    for (const auto& region : regions) {
                        runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
                        const auto count = region.bytes.size() / sizeof(std::uint32_t);
                        const auto offset = words.size();
                        words.resize(offset + count);
                        std::memcpy(words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
                    }
                    std::string what;
                    const auto merged = [](const std::vector<std::pair<std::uint64_t, std::uint64_t>>& list) {
                        std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
                        for (const auto& run : list) {
                            if (!out.empty() && out.back().second == run.first) out.back().second = run.second;
                            else out.push_back(run);
                        }
                        return out;
                    };
                    if (merged(runs) != merged(probe.runs)) {
                        what = "runs";
                        char item[96];
                        std::snprintf(item, sizeof(item), " fresh %zu kept %zu:", runs.size(), probe.runs.size());
                        what += item;
                        for (std::size_t r = 0; r < runs.size() || r < probe.runs.size(); ++r) {
                            std::snprintf(item, sizeof(item), " %llx/%llx", r < runs.size() ? static_cast<unsigned long long>(runs[r].first) : 0ull, r < probe.runs.size() ? static_cast<unsigned long long>(probe.runs[r].first) : 0ull);
                            what += item;
                        }
                    }
                    else if (words != probe.words) what = "words";
                    else if (fresh.variantId != kept.variantId) what = "variant";
                    else if (fresh.specializationId != kept.specializationId) what = "specialization";
                    else if (fresh.pushConstants != kept.pushConstants) {
                        what = "push constants";
                        const auto wordsOf = [](const std::vector<std::byte>& bytes) {
                            std::vector<std::uint32_t> out(bytes.size() / sizeof(std::uint32_t));
                            std::memcpy(out.data(), bytes.data(), out.size() * sizeof(std::uint32_t));
                            return out;
                        };
                        const auto x = wordsOf(fresh.pushConstants);
                        const auto y = wordsOf(kept.pushConstants);
                        char item[96];
                        for (std::size_t w = 0; w < x.size() && w < y.size(); ++w) {
                            if (x[w] == y[w]) continue;
                            std::snprintf(item, sizeof(item), " [%zu] fresh %08x kept %08x", w, x[w], y[w]);
                            what += item;
                        }
                        what += " user words now:";
                        for (const auto word : program.userData) {
                            std::snprintf(item, sizeof(item), " %08x", word);
                            what += item;
                        }
                    }
                    else if (fresh.bindings.size() != kept.bindings.size()) what = "binding count";
                    else if (fresh.vertexAttributes.size() != kept.vertexAttributes.size()) what = "attribute count";
                    if (what.empty()) {
                        for (std::size_t b = 0; b < fresh.bindings.size() && what.empty(); ++b) {
                            const auto& x = fresh.bindings[b].guestDescriptor;
                            const auto& y = kept.bindings[b].guestDescriptor;
                            if (x == y) continue;
                            std::size_t at = 0;
                            while (at < x.size() && at < y.size() && x[at] == y[at]) ++at;
                            char item[160];
                            std::snprintf(item, sizeof(item), "binding %zu role %u word %zu/%zu fresh %08x kept %08x", b, static_cast<unsigned>(fresh.bindings[b].role), at, x.size(), at < x.size() ? x[at] : 0u, at < y.size() ? y[at] : 0u);
                            what = item;
                        }
                        for (std::size_t a = 0; a < fresh.vertexAttributes.size() && what.empty(); ++a) {
                            if (fresh.vertexAttributes[a].resource.fields != kept.vertexAttributes[a].resource.fields) {
                                const auto& x = fresh.vertexAttributes[a].resource.fields;
                                const auto& y = kept.vertexAttributes[a].resource.fields;
                                char item[320];
                                std::size_t match = probe.words.size();
                                for (std::size_t w = 0; w < probe.words.size(); ++w) {
                                    if (probe.words[w] == x[0]) {
                                        match = w;
                                        break;
                                    }
                                }
                                std::snprintf(item, sizeof(item), "attribute %zu fresh %08x %08x %08x %08x kept %08x %08x %08x %08x; probe word %zu: %08x %08x %08x %08x", a, x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], match, match + 0 < probe.words.size() ? probe.words[match] : 0u, match + 1 < probe.words.size() ? probe.words[match + 1] : 0u, match + 2 < probe.words.size() ? probe.words[match + 2] : 0u, match + 3 < probe.words.size() ? probe.words[match + 3] : 0u);
                                what = item;
                            }
                        }
                    }
                    static std::atomic<std::uint64_t> agreed{0}, differed{0};
                    if (what.empty()) agreed.fetch_add(1, std::memory_order_relaxed);
                    else differed.fetch_add(1, std::memory_order_relaxed);
                    static std::atomic<int> printed{0};
                    if (!what.empty() && printed.fetch_add(1, std::memory_order_relaxed) < 40) std::fprintf(stderr, "[draw-cache] rebased verify: stage %zu program 0x%llx differs in %s\n", i, static_cast<unsigned long long>(program.binary.codeAddress), what.c_str());
                    static std::atomic<std::uint64_t> reported{0};
                    if ((agreed.load(std::memory_order_relaxed) + differed.load(std::memory_order_relaxed)) / 2000 != reported.load(std::memory_order_relaxed)) {
                        reported.store((agreed.load(std::memory_order_relaxed) + differed.load(std::memory_order_relaxed)) / 2000, std::memory_order_relaxed);
                        std::fprintf(stderr, "[draw-cache] rebased verify: %llu agreed, %llu differed\n", static_cast<unsigned long long>(agreed.load(std::memory_order_relaxed)), static_cast<unsigned long long>(differed.load(std::memory_order_relaxed)));
                    }
                }
            }
            if (dataEntry != nullptr && reused[i]) ++dataStagesReused;
            else if (dataEntry != nullptr) ++dataStagesCompiled;
        }
        const auto& result = *programResults[i];
        if (i == 0 && drawParameters.indirect) {
            indirectCpu = classifyIndirectDraw(result, graphics, programs.front(), localDevice, drawParameters, traceIndirect);
        } else if (i == 0) {
            fold(result, drawParameters);
        }
        require(result.pushConstants.size() <= Graphics::PipelinePushConstantBytes - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        stages.push_back({program.binary.stage, &result, result.pushConstants.empty() ? 0u : pushCursorBytes});
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }

    if (dataEntry != nullptr) {
        std::lock_guard cacheLock(drawCacheMutex);
        drawEntryCounters.dataStagesReused += dataStagesReused;
        drawEntryCounters.dataStagesCompiled += dataStagesCompiled;
        drawEntryCounters.dataVerified += dataVerified;
    }
    cacheDrawStages(useDrawEntries, drawHit, drawParameters, indirectCpu, programs, stageCaptures, vertexInfos, decodeReads, verifyHit, matched, fresh, drawKey, registerKey, decode, phaseTiming, reused, entry, shapeKey, dataEntry);
    timing.Mark("shader_compile_and_link");

    for (const auto& reads : decodeReads) {
        for (const auto& read : reads) memory.push_back({read.address, std::as_bytes(std::span(read.bytes))});
    }

    if (pipelined && !deferredLabels().labels.empty()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Labels);
    if (recordQueuedLabelsAfterCapture(submission.queue, memory)) return draw(queue, packet, submission, rejected);

    bool rectListBuilt = false;

    std::size_t rectIndex = 0;
    const auto buildRectList = [&] {
        phaseTiming.Phase(DrawRowVectors);
        require(programs.size() == 2 && programResults[0] != nullptr && programResults[1] != nullptr, "rect-list requires vertex and fragment programs");
        auto rectangle = PreparedRectangle(*programs[0].snapshot, programResults[0]->variantId, programResults[1]->variantId);
        if (rectListBuilt) {
            results[rectIndex] = std::move(rectangle.control);
            results[rectIndex + 1] = std::move(rectangle.evaluation);
            phaseTiming.Phase(DrawRowRectList);
            return;
        }
        require(stages.size() == 2, "rect-list requires vertex and fragment programs");
        rectIndex = results.size();
        results.push_back(std::move(rectangle.control));
        results.push_back(std::move(rectangle.evaluation));
        stages.insert(stages.begin() + 1, {{Stage::TessellationControl, &results[rectIndex], 0}, {Stage::TessellationEvaluation, &results[rectIndex + 1], 0}});
        rectListBuilt = true;
        phaseTiming.Phase(DrawRowRectList);
    };
    if (graphics.rectList) buildRectList();
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    const auto snapshot = [&] {
        snapshots.clear();
        for (const auto& region : memory) snapshots.push_back({region.guestAddress, region.bytes});
    };
    snapshot();
    timing.Mark("post_compile_prepare");
    const auto lockForDraw = [&] {
        if (gpuLock.owns_lock()) return;
        phaseTiming.Phase(DrawRowVectors);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        gpuLock.lock();
        timing.Mark("gpu_mutex_wait");
        phaseTiming.Phase(DrawRowLockWait);

        if (auto current = device.Load(); current != nullptr && current != localDevice) {
            static std::atomic<std::uint64_t> replaced{0};
            std::fprintf(stderr, "[draw] device replaced during unlocked preparation (%llu)\n", static_cast<unsigned long long>(++replaced));
            localDevice = std::move(current);
        }

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(DrawRowLabels);
    };
    if (drawParameters.indirect && indirectCpu) {

        const auto indirect = *drawParameters.indirect;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            if (programResults[i] == nullptr) continue;
            resultIndex[i] = results.size();
            results.push_back(ShaderRecompiler::RecompileResult(*programResults[i]));
            for (auto& stage : stages) {
                if (stage.program == programResults[i]) stage.program = &results[resultIndex[i]];
            }
            programResults[i] = &results[resultIndex[i]];
        }
        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        const auto count = std::min(indirect.countIndirect ? Pm4::ReadDrawCount(indirect) : indirect.count, indirect.count);
        std::vector<Pm4::DrawArguments> records;
        for (std::uint32_t record = 0; record < count; ++record) records.push_back(Pm4::ReadDrawArguments(indirect, record));
        Graphics::CountIndirectDraw(*indirectCpu, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        const auto baseVertexWord = locate(indirect.baseVertexLocation);
        const auto startInstanceWord = locate(indirect.startInstanceLocation);
        const auto drawIndexWord = indirect.drawIndexEnabled ? locate(indirect.drawIndexLocation) : std::nullopt;
        for (std::uint32_t record = 0; record < records.size(); ++record) {
            const auto& arguments = records[record];
            if (traceIndirect) std::fprintf(stderr, "[draw]   record %u: count %u instances %u first %u vertexOffset %u startInstance %u\n", record, arguments.count, arguments.instances, arguments.firstVertexOrIndex, arguments.vertexOffset, arguments.firstInstance);
            if (arguments.count == 0 || arguments.instances == 0) continue;
            std::set<std::size_t> patched;
            const auto patch = [&](const std::optional<std::pair<std::size_t, std::size_t>>& word, std::uint32_t value) {
                if (!word) return;
                programs[word->first].userData[word->second] = value;
                patched.insert(word->first);
            };
            patch(baseVertexWord, indirect.recordBytes == 20 ? arguments.vertexOffset : arguments.firstVertexOrIndex);
            patch(startInstanceWord, arguments.firstInstance);
            patch(drawIndexWord, record);
            Pm4::DrawParameters direct{0, arguments.count, 0, arguments.instances, drawParameters.flags, drawParameters.indexed, 0, 0};
            if (drawParameters.indexed) {

                if (arguments.firstVertexOrIndex >= drawParameters.indexCount) continue;
                direct.indexAddress = drawParameters.indexAddress + static_cast<std::uint64_t>(arguments.firstVertexOrIndex) * drawParameters.indexSize;
                direct.indexCount = std::min(arguments.count, drawParameters.indexCount - arguments.firstVertexOrIndex);
                direct.indexSize = drawParameters.indexSize;
            } else {
                direct.firstVertex = indirect.indxOffset;
            }
            if (graphics.stages.mesh) {
                setMeshIndexBuffer(direct);
                patched.insert(0);
            }
            for (const auto programIndex : patched) {
                auto& result = results[resultIndex[programIndex]];
                const auto pushBytes = result.pushConstants.size();
                decodeVertexInfo(programIndex);
                const auto patchedResult = materializeDrawStage(programIndex, pushOffsets[programIndex], queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected);
                if (!rejected.empty()) return DrawVerdict::Rejected;
                result = *patchedResult;
                require(result.pushConstants.size() == pushBytes, "patched program changed its push constant layout");
            }
            fold(*programResults[0], direct);
            if (graphics.rectList && patched.contains(0)) buildRectList();
            snapshot();
            if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
                rejected = std::move(*known);
                return DrawVerdict::Rejected;
            }
            lockForDraw();
            noteDrawWriters(stages, submission.queue);
            phaseTiming.Phase(DrawRowVectors);
            localDevice->Draw(graphics, direct, stages, snapshots);
            phaseTiming.Phase(DrawRowGraphics);
        }
        timing.Mark("draw_and_resource_release");
        return drawn();
    }

    std::vector<std::shared_ptr<DispatchVariant>> recipeStages;
    if (registerKey && !drawParameters.indirect && Graphics::DrawRecipes() && Graphics::DrawRecipeWanted(drawHit)) {
        recipeStages.reserve(programs.size());
        for (std::size_t i = 0; i < programs.size(); ++i) recipeStages.push_back(drawHit ? matched[i] : fresh[i]);
        if (std::all_of(recipeStages.begin(), recipeStages.end(), [](const std::shared_ptr<DispatchVariant>& variant) { return variant == nullptr; })) recipeStages.clear();
    }
    std::shared_ptr<const DrawRecipe> recipe;
    if (drawHit && !recipeStages.empty()) {
        recipe = findDrawRecipe(drawKey, recipeStages);
        if (recipe == nullptr) VulkanDevice::NoteDrawRecipeMiss(VulkanDevice::DrawRecipePrecheck::NoRecipe);
    }
    if (recipe == nullptr) {
        if (auto known = localDevice->KnownDrawRejection(graphics, stages)) {
            rejected = std::move(*known);
            return DrawVerdict::Rejected;
        }
    }
    if (pipelined && !drawParameters.indirect && deferredLabels().labels.empty()) {
        static const bool windowRegistered = [] {
            Graphics::SetAppliedWindow({[] { return DrawPipeline::Queue0().AppliedFrontier(); }, [](std::uint64_t fromExclusive, std::span<const std::pair<std::uint64_t, std::uint64_t>> reads) {
                return static_cast<Graphics::WindowVerdict>(DrawPipeline::Queue0().CheckWindow(fromExclusive, DrawPipeline::CommittingSequence(), reads));
            }});
            GuestMemory::SetCollectHistoryFlag(&Graphics::TargetTokensEnabled);
            return true;
        }();
        static_cast<void>(windowRegistered);
        phaseTiming.Phase(DrawRowVectors);
        auto writes = drawWriteRanges(graphics, stages);
        const bool unknownWrites = std::any_of(stages.begin(), stages.end(), [](const Graphics::CompiledShader& stage) { return stage.program == nullptr || stage.program->bdaWrites; });
        const auto commitQueue = submission.queue;
        const auto commitPacket = (packet[0] >> 8u) & 0xffu;
        const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        if (GuestMemory::SharedEpochMemo()) {
            // The committer's collects of this draw's surfaces become memo hits of the shared epoch.
            DrawPipeline::FollowEpoch(epoch);
            for (const auto& stage : stages) {
                if (stage.program == nullptr) continue;
                for (const auto& binding : stage.program->bindings) {
                    if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0 || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 8u) continue;
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                        if (words[0] == 0u && (words[1] & 0xffu) == 0u) continue;
                        try {
                            const auto resource = Graphics::DecodeTextureResource(words);
                            if (resource.baseAddress == 0) continue;
                            const auto bytes = Graphics::DescribeSurface(resource).guestBytes;
                            if (bytes != 0) static_cast<void>(GuestMemory::CollectWrites(resource.baseAddress, static_cast<std::size_t>(bytes)));
                        } catch (const std::exception&) {
                        }
                    }
                }
            }
            const auto prefetchTarget = [](const Graphics::ColorTarget& color) {
                if (color.address == 0 || color.bytes == 0) return;
                try {
                    const auto surface = Graphics::SurfaceForTarget(color);
                    const auto bytes = Graphics::DescribeSurface(surface).guestBytes;
                    if (surface.baseAddress != 0 && bytes != 0) static_cast<void>(GuestMemory::CollectWrites(surface.baseAddress, static_cast<std::size_t>(bytes)));
                } catch (const std::exception&) {
                }
            };
            if (graphics.hasColorTarget) prefetchTarget(graphics.color);
            for (const auto& color : graphics.colors) prefetchTarget(color);
        }
        std::shared_ptr<Graphics::DrawSpeculation> speculation;
        // Why the prep thread did not speculate (the [commit] line): Count when it did not try.
        auto prepRefusal = Graphics::DrawSpeculation::Miss::Count;
        if (recipe == nullptr && Graphics::DrawSpeculationEnabled() && DrawPipeline::Queue0().OnPrepThread()) {
            try {
                speculation = localDevice->SpeculateDraw(graphics, drawParameters, stages, snapshots);
                prepRefusal = Graphics::TakeSpeculationRefusal();
            } catch (...) {
                speculation = nullptr;
                prepRefusal = Graphics::DrawSpeculation::Miss::Thrown;
            }
        }
        std::shared_ptr<Graphics::DrawPrebuild> prebuilt;
        if (recipe == nullptr && Graphics::ParallelDrawBuild() && Graphics::DrawPlansEnabled() && DrawPipeline::Queue0().OnPrepThread()) prebuilt = localDevice->PrebuildDraw(graphics, drawParameters, stages, speculation == nullptr);
        if (speculation != nullptr && prebuilt != nullptr) localDevice->SpeculatePipeline(graphics, stages, *speculation, prebuilt);
        static const bool runnerRegistered = [this] {
            DrawPipeline::SetForkRunner(new DrawForkRunner(*this));
            return true;
        }();
        static_cast<void>(runnerRegistered);
        std::shared_ptr<DrawForkJob> forkJob;
        // The job's pass hint also tells the committer which queued draws continue a pass (live
        // `parallel_commit` pass runs, DrawPipeline::run); it changes nothing else there.
        const bool forkFlags = Graphics::ForkBuildEnabled() || Graphics::ForkStatsEnabled() || Graphics::ForkAheadEnabled();
        if (forkFlags || Graphics::ParallelCommitEnabled()) {
            forkJob = std::make_shared<DrawForkJob>();
            forkJob->forFork = forkFlags;
        }
        auto commit = [this, forkJob, localDevice = std::move(localDevice), decode = std::move(decode), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), drawKey, commitQueue, commitPacket, epoch, prepRefusal, speculation = std::move(speculation), prebuilt = std::move(prebuilt), programs = std::move(programs), results = std::move(results), compiledStages = std::move(compiledStages), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry)](DrawForkJob* fork) {
            if (fork != nullptr) {
                fork->device = localDevice;
                fork->state = &decode->state;
                fork->draw = &drawParameters;
                fork->stages = stages;
                fork->snapshots = snapshots;
                fork->prebuilt = prebuilt;
                fork->recipeStages = &recipeStages;
                fork->drawKey = drawKey;
                fork->queue = commitQueue;
                fork->packet = commitPacket;
                fork->epoch = epoch;
                fork->speculation = speculation;
                fork->eligible = recipe == nullptr && prebuilt != nullptr && !drawParameters.indirect;
                fork->passHint = forkPassHint(decode->state);
                return;
            }
            const bool timed = forkTimed();
            const auto lambdaStart = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            if (forkJob != nullptr && forkJob->ahead != nullptr) {
                auto forked = std::move(forkJob->ahead);
                bool cut = false;
                commitForked(*forkJob, *forkJob->aheadContext, forked.get(), *forkJob->aheadStamp, forkJob->aheadRecipe, cut);
                auto& ahead = Ahead();
                if (cut) {
                    ahead.stale.fetch_add(1, std::memory_order_relaxed);
                    const char* why = Graphics::ForkedDrawCut(*forked);
                    countAhead(&AheadStatistics::staleReasons, why != nullptr ? why : "device");
                } else {
                    ahead.used.fetch_add(1, std::memory_order_relaxed);
                    if (timed) ahead.usedNs.fetch_add(nanosSince(lambdaStart), std::memory_order_relaxed);
                }
                reportForks();
                return;
            }
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
            commitDraw(localDevice, commitQueue, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey, std::move(speculation), std::move(prebuilt), static_cast<std::uint8_t>(prepRefusal));
            if (forkJob != nullptr && Graphics::ForkAheadEnabled()) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
                std::lock_guard publishLock(GuestMemory::GpuMutex());
                publishSeed(epoch, forkJob->passHint);
            }
            if (!timed) return;
            Commits().commits.fetch_add(1, std::memory_order_relaxed);
            Commits().lambdaNs.fetch_add(nanosSince(lambdaStart), std::memory_order_relaxed);
            reportForks();
        };
        auto held = std::make_shared<decltype(commit)>(std::move(commit));
        if (forkJob != nullptr) {
            (*held)(forkJob.get());
            forkJob->owner = std::shared_ptr<void>(held);
            if (Graphics::ForkAheadEnabled() && DrawPipeline::Queue0().OnPrepThread()) buildAhead(*forkJob);
        }
        DrawPipeline::Queue0().Enqueue([held] { (*held)(nullptr); }, std::move(writes), 0, {}, unknownWrites, std::move(forkJob));
        phaseTiming.Phase(DrawRowLockWait);
        timing.Mark("draw_enqueued");
        return drawn();
    }
    lockForDraw();
    noteDrawWriters(stages, submission.queue);
    phaseTiming.Phase(DrawRowVectors);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe) == RecipeOutcome::Recorded) {
            phaseTiming.Phase(DrawRowGraphics);
            timing.Mark("draw_and_resource_release");
            return drawn();
        }

        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built);
    phaseTiming.Phase(DrawRowGraphics);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
    timing.Mark("draw_and_resource_release");
    return drawn();
    };
    if (!splitDraw || DrawPipeline::Queue0().OnPrepThread()) return tail();
    auto held = std::make_shared<decltype(tail)>(std::move(tail));
    DrawPipeline::Queue0().Prepare([held] { static_cast<void>((*held)()); });
    return drawn();
}

namespace {

// APS5_PROFILE_DRAW: the committer's draw commits (Driver::commitDraw) every 10 s, the [commit]
// line: the GPU mutex wait, the hold and its parts, per draw, and the hold by what became of the
// draw's speculation (accepted, refused on commit, or why the prep thread made none).
struct CommitProfile {
    // Classes: recipe draws, accepted, refused on commit, then the prep refusals by
    // SpeculationMiss (index Count: the prep thread did not try; Count + 1: unknown).
    static constexpr std::size_t MissCount = static_cast<std::size_t>(Graphics::DrawSpeculation::Miss::Count);
    static constexpr std::size_t Recipe = 0, Accepted = 1, Refused = 2, FirstPrep = 3, ClassCount = FirstPrep + MissCount + 2;
    std::array<std::atomic<std::uint64_t>, ClassCount> counts{};
    std::array<std::atomic<std::uint64_t>, ClassCount> heldNs{};
    std::atomic<std::uint64_t> commits{0}, waitNs{0}, labelsNs{0}, writersNs{0}, drawNs{0}, attachNs{0}, submitNs{0}, held{0}, attached{0};
    std::atomic<std::int64_t> reported{0};
};

CommitProfile& CommitStats() {
    static CommitProfile stats;
    return stats;
}

bool commitProfiled() {
    static const bool profiled = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profiled;
}

std::size_t commitClass(bool recipeDrawn, std::uint8_t outcome, std::uint8_t prepRefusal) {
    if (recipeDrawn) return CommitProfile::Recipe;
    if (outcome == 1) return CommitProfile::Accepted;
    if (outcome == 2) return CommitProfile::Refused;
    if (prepRefusal == 0 || prepRefusal > CommitProfile::MissCount) return CommitProfile::FirstPrep + CommitProfile::MissCount + 1;
    return CommitProfile::FirstPrep + prepRefusal;
}

void reportCommits() {
    auto& stats = CommitStats();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = stats.reported.load(std::memory_order_relaxed);
    if (last == 0) {
        stats.reported.compare_exchange_strong(last, now);
        return;
    }
    if (now - last < 10'000'000'000ll || !stats.reported.compare_exchange_strong(last, now)) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
    const auto commits = take(stats.commits);
    const auto per = [&](std::uint64_t ns) { return commits != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(commits) : 0.0; };
    const auto wait = take(stats.waitNs), held = take(stats.held), labels = take(stats.labelsNs), writers = take(stats.writersNs), drawn = take(stats.drawNs), attach = take(stats.attachNs), submit = take(stats.submitNs);
    const auto rest = held > labels + writers + drawn + attach + submit ? held - labels - writers - drawn - attach - submit : 0;
    char head[640];
    std::snprintf(head, sizeof(head), "[commit] (10 s) %llu draw commits, us per commit: GPU mutex wait %.2f, held %.2f (labels %.2f, writers %.2f, Graphics::Draw %.2f, recipe attach %.2f (%llu attached, %llu skipped by recipe_on_hit), submit check %.2f, other %.2f); held %.1f ms in total; by speculation (commits at us held each):", static_cast<unsigned long long>(commits), per(wait), per(held), per(labels), per(writers), per(drawn), per(attach), static_cast<unsigned long long>(take(stats.attached)), static_cast<unsigned long long>(Graphics::TakeDrawRecipeSkips()), per(submit), per(rest), static_cast<double>(held) / 1e6);
    std::string line = head;
    for (std::size_t index = 0; index < CommitProfile::ClassCount; ++index) {
        const auto count = take(stats.counts[index]);
        const auto ns = take(stats.heldNs[index]);
        if (count == 0) continue;
        std::string name = index == CommitProfile::Recipe ? "recipe" : index == CommitProfile::Accepted ? "accepted" : index == CommitProfile::Refused ? "refused on commit" : index == CommitProfile::FirstPrep + CommitProfile::MissCount ? "not tried" : index == CommitProfile::FirstPrep + CommitProfile::MissCount + 1 ? "unknown" : std::string("prep refused ") + Graphics::SpeculationMissName(static_cast<Graphics::DrawSpeculation::Miss>(index - CommitProfile::FirstPrep));
        char item[160];
        std::snprintf(item, sizeof(item), " %s %llu at %.1f;", name.c_str(), static_cast<unsigned long long>(count), static_cast<double>(ns) / 1000.0 / static_cast<double>(count));
        line += item;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

}

void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey, std::shared_ptr<Graphics::DrawSpeculation> speculation, std::shared_ptr<Graphics::DrawPrebuild> prebuilt, std::uint8_t prepRefusal) {
    const bool timed = forkTimed();
    auto lap = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto mark = [&](std::atomic<std::uint64_t>& into) {
        if (!timed) return;
        const auto now = std::chrono::steady_clock::now();
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lap).count()), std::memory_order_relaxed);
        lap = now;
    };
    // The [commit] profile's clock: `stamp` laps into a CommitProfile counter; the hold ends when
    // `holdEnd` (declared after the lock, so destroyed before it) runs, after the submit check.
    const bool profiled = commitProfiled();
    auto& commitStats = CommitStats();
    auto stampAt = profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto stamp = [&](std::atomic<std::uint64_t>& into) {
        if (!profiled) return;
        const auto now = std::chrono::steady_clock::now();
        into.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - stampAt).count()), std::memory_order_relaxed);
        stampAt = now;
    };
    // The 10 s report runs once the GPU mutex is released (declared before the lock).
    struct ReportAfterUnlock {
        bool profiled;
        ~ReportAfterUnlock() {
            if (!profiled) return;
            try {
                reportCommits();
            } catch (...) {
            }
        }
    } reportAfterUnlock{profiled};
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    stamp(commitStats.waitNs);
    bool recipeDrawn = false;
    struct HoldEnd {
        bool profiled;
        CommitProfile& stats;
        std::chrono::steady_clock::time_point locked;
        const bool& recipeDrawn;
        std::uint8_t prepRefusal;
        ~HoldEnd() {
            if (!profiled) return;
            const auto held = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - locked).count());
            const auto index = commitClass(recipeDrawn, Graphics::TakeCommitSpeculationOutcome(), prepRefusal);
            stats.held.fetch_add(held, std::memory_order_relaxed);
            stats.counts[index].fetch_add(1, std::memory_order_relaxed);
            stats.heldNs[index].fetch_add(held, std::memory_order_relaxed);
            stats.commits.fetch_add(1, std::memory_order_relaxed);
        }
    } holdEnd{profiled, commitStats, stampAt, recipeDrawn, prepRefusal};
    if (profiled) static_cast<void>(Graphics::TakeCommitSpeculationOutcome());
    if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
    mark(Commits().lockNs);
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        bool timed;
        bool profiled;
        CommitProfile& stats;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() != 0) return;
            const auto start = timed || profiled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            driver.submitDueAfterCommit(device.get());
            if (timed || profiled) {
                const auto ns = nanosSince(start);
                if (timed) Commits().submitNs.fetch_add(ns, std::memory_order_relaxed);
                if (profiled) stats.submitNs.fetch_add(ns, std::memory_order_relaxed);
            }
        }
    } submitDue{*this, localDevice, timed, profiled, commitStats};
    recordLabelsForPacket(localDevice.get(), queue);
    mark(Commits().labelsNs);
    stamp(commitStats.labelsNs);
    noteDrawWriters(stages, queue);
    mark(Commits().writersNs);
    stamp(commitStats.writersNs);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe) == RecipeOutcome::Recorded) {
            mark(Commits().drawNs);
            stamp(commitStats.drawNs);
            recipeDrawn = true;
            return;
        }
        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    {
        const Graphics::SpeculationScope speculationScope(std::move(speculation));
        const Graphics::DrawPrebuildScope prebuildScope(std::move(prebuilt));
        localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built);
    }
    stamp(commitStats.drawNs);
    if (built != nullptr) {
        attachDrawRecipe(drawKey, recipeStages, std::move(built));
        if (profiled) commitStats.attached.fetch_add(1, std::memory_order_relaxed);
    }
    stamp(commitStats.attachNs);
    mark(Commits().drawNs);
}

void Driver::commitForked(const DrawForkJob& job, const Graphics::Context& context, Graphics::ForkedDraw* forked, const Graphics::ForkStamp& stamp, std::shared_ptr<const DrawRecipe>& built, bool& cut, const std::function<void()>* quiesce) {
    DrawPipeline::FollowEpoch(job.epoch);
    GuestMemory::SetCurrentPacket(job.packet, job.queue);
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    auto localDevice = job.device;
    if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        const std::function<void()>* quiesce;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() != 0) return;
            if ((quiesce != nullptr || Graphics::ForkQuiescing()) && driver.submitPending(device.get())) {
                if (quiesce != nullptr) (*quiesce)();
                else Graphics::ForkQuiesce("submit due");
            }
            driver.submitDueAfterCommit(device.get());
        }
    } submitDue{*this, localDevice, quiesce};
    if ((quiesce != nullptr || Graphics::ForkQuiescing()) && labelsPending(localDevice.get())) {
        if (quiesce != nullptr) (*quiesce)();
        else Graphics::ForkQuiesce("labels");
    }
    recordLabelsForPacket(localDevice.get(), job.queue);
    noteDrawWriters(job.stages, job.queue);
    const bool sameDevice = localDevice == job.device;
    if (!sameDevice) Graphics::CountForkCut("device replaced");
    else if (forked == nullptr) Graphics::CountForkCut("build failed");
    if (!sameDevice || forked == nullptr || !localDevice->ApplyForkedDraw(context, job.stages, *forked, stamp, quiesce)) {
        cut = true;
        if (quiesce != nullptr) (*quiesce)();
        else Graphics::ForkQuiesce("cut fallback");
        built.reset();
        const Graphics::DrawPrebuildScope prebuildScope(job.prebuilt);
        localDevice->Draw(*job.state, *job.draw, job.stages, job.snapshots, job.recipeStages->empty() ? nullptr : &built);
    }
    if (built != nullptr) attachDrawRecipe(job.drawKey, *job.recipeStages, std::move(built));
    if (Graphics::ForkAheadEnabled()) publishSeed(job.epoch, job.passHint);
}

void Driver::buildAhead(DrawForkJob& job) {
    auto& stats = Ahead();
    stats.attempted.fetch_add(1, std::memory_order_relaxed);
    if (!job.eligible) {
        countAhead(&AheadStatistics::skipped, "ineligible");
        return;
    }
    if (!DrawPipeline::Queue0().CommitterIdle()) {
        countAhead(&AheadStatistics::skipped, "committer busy");
        return;
    }
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::try_to_lock);
    if (!gpuLock.owns_lock()) {
        countAhead(&AheadStatistics::skipped, "lock busy");
        return;
    }
    auto* recorder = Graphics::Recorder::Active();
    const auto& published = Published();
    if (device.Load() != job.device) {
        countAhead(&AheadStatistics::skipped, "device");
        return;
    }
    if (recorder == nullptr || !recorder->PassContinuable()) {
        countAhead(&AheadStatistics::skipped, "no open pass");
        return;
    }
    if (published.seed == nullptr || published.hint != job.passHint) {
        countAhead(&AheadStatistics::skipped, "other pass");
        return;
    }
    const auto started = std::chrono::steady_clock::now();
    DrawPipeline::FollowEpoch(job.epoch);
    auto context = std::make_shared<const Graphics::Context>(job.device->DrawContext());
    auto forked = Graphics::ForkDrawSpeculated(*context, *job.state, *job.draw, job.stages, job.snapshots, job.prebuilt, job.recipeStages->empty() ? nullptr : &job.aheadRecipe, published.seed.get(), job.epoch == published.token, DrawPipeline::AheadSlot, job.speculation);
    stats.buildNs.fetch_add(nanosSince(started), std::memory_order_relaxed);
    if (Graphics::ForkedDrawRefusal(*forked) != nullptr) {
        stats.refused.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    job.aheadStamp = Graphics::TakeForkStamp(*context);
    job.aheadContext = std::move(context);
    job.ahead = std::move(forked);
    stats.ready.fetch_add(1, std::memory_order_relaxed);
}

}
