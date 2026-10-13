#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ReplayProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "ThreadOwned.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

struct ReplaySegment : std::enable_shared_from_this<ReplaySegment> {
    struct Step {
        std::uint64_t full;
        std::uint64_t user;
        std::uint64_t other;
        std::uint64_t item;
    };
    struct Range {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t generation;
    };
    std::uint64_t key = 0;
    std::uint64_t keyUser = 0;
    std::uint64_t keyOther = 0;
    std::uint64_t device = 0;
    std::uint64_t registry = 0;
    std::uint64_t content = 0;
    std::uint64_t startGeneration = 0;
    std::size_t ordinal = 0;
    std::vector<Step> steps;
    bool hitPrevious = false;
    bool hitAny = false;
    bool validated = false;
    bool passed = false;
    bool recordingClean = false;
    std::uint8_t failure = 0;
    std::uint8_t miss = 0;
    std::atomic<std::uint64_t> commitNs{0};
    std::atomic<std::uint32_t> commits{0};
    std::atomic<std::uint32_t> pending{0};
    std::atomic<std::uint64_t> reasons{0};
    std::mutex mutex;
    std::vector<Range> baked;
    bool overflow = false;
};

namespace ReplayProbe {

namespace {

constexpr std::uint64_t Seed = 0xcbf29ce484222325ull;
constexpr std::size_t SegmentLength = 128;
constexpr std::size_t FramesKept = 4;
constexpr std::size_t FrameCapacity = 4096;
constexpr std::size_t BakedCapacity = 16384;
constexpr std::uint64_t BlockBytes = 0x10000;
constexpr std::size_t ReasonSlots = 64;

enum Failure : std::uint8_t { FailIncomplete, FailOverflow, FailUncovered, FailUntracked, FailCpu, FailStamped, FailCount };
constexpr std::array<const char*, FailCount> FailureNames{"incomplete", "overflow", "uncovered", "untracked", "cpu", "stamped"};
enum Miss : std::uint8_t { MissNew, MissStartRegisters, MissStartUserData, MissDevice, MissLength, MissDraw, MissRegisters, MissUserData, MissCount };
constexpr std::array<const char*, MissCount> MissNames{"new", "start registers", "start user data", "device or shaders", "length", "draw packet", "registers", "user data"};

constexpr std::array<const char*, 27> Disqualifying{
    "key store", "key stores", "label store", "label stores", "label", "completion label", "recorded work action",
    "finish", "sync", "sync through", "storage flush", "storage results flush", "dcc key fill memo", "dcc keys store",
    "shadow publish", "shadow retire", "import refresh", "storage upload", "storage write-back", "alias borrow",
    "write-back", "on complete", "completions run", "mesh arguments", "commit error", "snapshot ring", "other"};

std::uint64_t mix(std::uint64_t hash, std::uint64_t value) {
    hash = (hash ^ value) * 0x9e3779b97f4a7c15ull;
    return hash ^ (hash >> 32u);
}

bool userDataRegister(std::uint32_t offset) {
    if (offset >= 0x240 && offset < 0x250) return true;
    if (offset == 0x82 || offset == 0x83 || offset == 0x102 || offset == 0x103) return true;
    return offset < 0x200 && (offset & 0x3fu) >= 0x0cu && (offset & 0x3fu) < 0x2cu;
}

struct ReasonTable {
    std::mutex mutex;
    std::vector<std::string> names;
    std::uint64_t disqualifying = 0;
    std::uint64_t memo = 0;
};

ReasonTable& Reasons() {
    static auto* table = new ReasonTable();
    return *table;
}

std::size_t internReason(const char* reason) {
    struct Cache {
        std::array<std::pair<const char*, std::uint8_t>, 64> entries{};
        std::size_t count = 0;
    };
    thread_local Cache* cacheSlot = nullptr;
    auto& cache = ShaderRecompiler::ThreadOwned(cacheSlot);
    for (std::size_t i = 0; i < cache.count; ++i) {
        if (cache.entries[i].first == reason) return cache.entries[i].second;
    }
    auto& table = Reasons();
    std::size_t index = ReasonSlots - 1;
    {
        std::lock_guard lock(table.mutex);
        const auto found = std::find(table.names.begin(), table.names.end(), reason);
        if (found != table.names.end()) {
            index = static_cast<std::size_t>(found - table.names.begin());
        } else if (table.names.size() + 1 < ReasonSlots) {
            index = table.names.size();
            table.names.emplace_back(reason);
            const std::string_view name(reason);
            if (std::find(Disqualifying.begin(), Disqualifying.end(), name) != Disqualifying.end()) table.disqualifying |= std::uint64_t{1} << index;
            if (name.starts_with("memo ")) table.memo |= std::uint64_t{1} << index;
        } else {
            table.disqualifying |= std::uint64_t{1} << index;
        }
    }
    if (cache.count < cache.entries.size()) cache.entries[cache.count++] = {reason, static_cast<std::uint8_t>(index)};
    return index;
}

std::pair<std::uint64_t, std::uint64_t> reasonMasks() {
    auto& table = Reasons();
    std::lock_guard lock(table.mutex);
    return {table.disqualifying, table.memo};
}

std::string reasonName(std::size_t index) {
    auto& table = Reasons();
    std::lock_guard lock(table.mutex);
    return index < table.names.size() ? table.names[index] : std::string("other");
}

void compact(std::vector<ReplaySegment::Range>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const ReplaySegment::Range& a, const ReplaySegment::Range& b) { return a.begin < b.begin; });
    std::size_t kept = 0;
    for (const auto& range : ranges) {
        if (kept != 0) {
            auto& last = ranges[kept - 1];
            const auto reach = (last.end + BlockBytes - 1) & ~(BlockBytes - 1);
            if (range.begin <= reach) {
                last.end = std::max(last.end, range.end);
                last.generation = std::min(last.generation, range.generation);
                continue;
            }
        }
        ranges[kept++] = range;
    }
    ranges.resize(kept);
}

class ThreadSink final : public GuestMemory::ProbeSink {
public:
    void Baked(std::uint64_t address, std::size_t bytes, std::uint64_t generation) override {
        if (segment == nullptr || bytes == 0) return;
        if (address == lastAddress && bytes == lastBytes && generation == lastGeneration) return;
        lastAddress = address;
        lastBytes = bytes;
        lastGeneration = generation;
        const auto stamp = generation != 0 ? generation : segment->startGeneration;
        std::lock_guard lock(segment->mutex);
        if (segment->overflow) return;
        if (segment->baked.size() >= BakedCapacity) {
            compact(segment->baked);
            if (segment->baked.size() >= BakedCapacity / 2) {
                segment->overflow = true;
                return;
            }
        }
        segment->baked.push_back({address, address + bytes, stamp});
    }

    void Event(const char* reason) override {
        if (segment == nullptr || reason == nullptr) return;
        segment->reasons.fetch_or(std::uint64_t{1} << internReason(reason), std::memory_order_relaxed);
    }

    ReplaySegment* segment = nullptr;
    std::uint64_t lastAddress = 0;
    std::uint64_t lastBytes = 0;
    std::uint64_t lastGeneration = 0;
};

thread_local ReplaySegment* currentSegment = nullptr;

ThreadSink& Sink() {
    thread_local ThreadSink* sinkSlot = nullptr;
    return ShaderRecompiler::ThreadOwned(sinkSlot);
}

struct Frame {
    std::vector<std::shared_ptr<ReplaySegment>> segments;
    std::unordered_map<std::uint64_t, ReplaySegment*> byContent;
    std::unordered_map<std::uint64_t, ReplaySegment*> byKey;
};

struct Window {
    std::uint64_t frames = 0;
    std::uint64_t forcedFrames = 0;
    std::uint64_t segments = 0;
    std::uint64_t minSegments = ~std::uint64_t{0};
    std::uint64_t maxSegments = 0;
    std::uint64_t steps = 0;
    std::uint64_t commits = 0;
    std::uint64_t segmentNs = 0;
    std::uint64_t hitPrevious = 0;
    std::uint64_t hitPreviousNs = 0;
    std::uint64_t hitAny = 0;
    std::uint64_t hitAnyNs = 0;
    std::uint64_t passed = 0;
    std::uint64_t passedNs = 0;
    std::uint64_t emptyBaked = 0;
    std::array<std::uint64_t, FailCount> failures{};
    std::uint64_t validations = 0;
    std::uint64_t validateNs = 0;
    std::uint64_t clean = 0;
    std::uint64_t cleanNs = 0;
    std::uint64_t memo = 0;
    std::uint64_t memoNs = 0;
    std::uint64_t replayable = 0;
    std::uint64_t replayableNs = 0;
    std::uint64_t bakedRanges = 0;
    std::uint64_t bakedOverflows = 0;
    std::uint64_t emptySegments = 0;
    std::array<std::uint64_t, MissCount> misses{};
    std::array<std::uint64_t, MissCount> missNs{};
    std::array<std::uint64_t, ReasonSlots> reasonSegments{};
    std::array<std::uint64_t, ReasonSlots> reasonNs{};
    std::array<std::uint64_t, CutCount> cuts{};
};

struct Stream {
    std::shared_ptr<ReplaySegment> open;
    std::uint64_t pendingUser = Seed;
    std::uint64_t pendingOther = Seed;
    Frame current;
    std::deque<Frame> history;
    Window window;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::uint64_t lastCommitNs = 0;
    bool started = false;
};

Stream& Queue0() {
    static auto* stream = new Stream();
    return *stream;
}

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

double percent(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

std::uint8_t classifyFailure(const std::vector<ReplaySegment::Range>& spans) {
    for (const auto& span : spans) {
        const auto bytes = static_cast<std::size_t>(span.end - span.begin);
        if (GuestMemory::UnchangedSince(span.begin, bytes, span.generation)) continue;
        if (span.generation == 0) return FailUntracked;
        if (!GuestMemory::Watched(span.begin, bytes)) return FailUncovered;
        const auto blocks = static_cast<std::size_t>(((span.end - 1) / BlockBytes) - (span.begin / BlockBytes) + 2);
        std::vector<std::uint64_t> generations(blocks, span.generation);
        std::vector<std::uint8_t> changed(blocks), cpu(blocks);
        if (!GuestMemory::ChangedBlocks(span.begin, bytes, generations, changed, cpu)) return FailUncovered;
        for (std::size_t k = 0; k < blocks; ++k) {
            if (changed[k] != GuestMemory::BlockUnchanged && cpu[k] != 0) return FailCpu;
        }
        return FailStamped;
    }
    return FailStamped;
}

void validate(Stream& stream, ReplaySegment& recorded, ReplaySegment& segment, std::uint64_t disqualifying) {
    const auto start = std::chrono::steady_clock::now();
    auto* previousSink = GuestMemory::SetProbeSink(nullptr);
    segment.validated = true;
    segment.recordingClean = (recorded.reasons.load(std::memory_order_relaxed) & disqualifying) == 0;
    std::vector<ReplaySegment::Range> spans;
    bool overflow = false;
    const bool incomplete = recorded.pending.load(std::memory_order_acquire) != 0;
    if (!incomplete) {
        std::lock_guard lock(recorded.mutex);
        spans = recorded.baked;
        overflow = recorded.overflow;
    }
    if (incomplete) {
        segment.failure = FailIncomplete;
    } else if (overflow) {
        segment.failure = FailOverflow;
    } else if (spans.empty()) {
        segment.passed = true;
        ++stream.window.emptyBaked;
    } else {
        compact(spans);
        std::vector<GuestMemory::UnchangedQuery> queries;
        queries.reserve(spans.size());
        for (const auto& span : spans) {
            const auto bytes = static_cast<std::size_t>(span.end - span.begin);
            static_cast<void>(GuestMemory::CollectWrites(span.begin, bytes));
            queries.push_back({span.begin, bytes, span.generation});
        }
        segment.passed = GuestMemory::UnchangedSinceAll(queries);
        if (!segment.passed) segment.failure = classifyFailure(spans);
    }
    GuestMemory::SetProbeSink(previousSink);
    ++stream.window.validations;
    stream.window.validateNs += elapsedNs(start);
}

std::uint8_t classifyMiss(const Stream& stream, const ReplaySegment& segment) {
    if (stream.history.empty()) return MissNew;
    const auto& previous = stream.history.front();
    const ReplaySegment* counterpart = nullptr;
    if (const auto found = previous.byKey.find(segment.key); found != previous.byKey.end()) counterpart = found->second;
    if (counterpart == nullptr) {
        if (segment.ordinal >= previous.segments.size()) return MissNew;
        const auto& positional = *previous.segments[segment.ordinal];
        if (positional.keyOther != segment.keyOther) return MissStartRegisters;
        if (positional.keyUser != segment.keyUser) return MissStartUserData;
        if (positional.device != segment.device || positional.registry != segment.registry) return MissDevice;
        counterpart = &positional;
    }
    if (counterpart->steps.size() != segment.steps.size()) return MissLength;
    for (std::size_t i = 0; i < segment.steps.size(); ++i) {
        const auto& mine = segment.steps[i];
        const auto& theirs = counterpart->steps[i];
        if (mine.full == theirs.full) continue;
        if (mine.item != theirs.item) return MissDraw;
        if (mine.other != theirs.other) return MissRegisters;
        return MissUserData;
    }
    return MissStartRegisters;
}

void account(Stream& stream, const Frame& frame) {
    auto& window = stream.window;
    const auto [disqualifying, memo] = reasonMasks();
    const auto count = static_cast<std::uint64_t>(frame.segments.size());
    ++window.frames;
    window.segments += count;
    window.minSegments = std::min(window.minSegments, count);
    window.maxSegments = std::max(window.maxSegments, count);
    for (const auto& segment : frame.segments) {
        const auto ns = segment->commitNs.load(std::memory_order_relaxed);
        const auto reasons = segment->reasons.load(std::memory_order_relaxed);
        const bool clean = (reasons & disqualifying) == 0;
        window.segmentNs += ns;
        window.steps += segment->steps.size();
        window.commits += segment->commits.load(std::memory_order_relaxed);
        if (segment->hitPrevious) {
            ++window.hitPrevious;
            window.hitPreviousNs += ns;
        }
        if (segment->hitAny) {
            ++window.hitAny;
            window.hitAnyNs += ns;
            if (segment->passed) {
                ++window.passed;
                window.passedNs += ns;
            } else if (segment->validated) {
                ++window.failures[segment->failure];
            }
        } else {
            ++window.misses[segment->miss];
            window.missNs[segment->miss] += ns;
        }
        if (clean) {
            ++window.clean;
            window.cleanNs += ns;
        }
        if ((reasons & memo) != 0) {
            ++window.memo;
            window.memoNs += ns;
        }
        if (segment->hitAny && segment->passed && clean && segment->recordingClean) {
            ++window.replayable;
            window.replayableNs += ns;
        }
        for (auto bits = reasons; bits != 0; bits &= bits - 1) {
            const auto index = static_cast<std::size_t>(std::countr_zero(bits));
            ++window.reasonSegments[index];
            window.reasonNs[index] += ns;
        }
        std::lock_guard lock(segment->mutex);
        window.bakedRanges += segment->baked.size();
        if (segment->overflow) ++window.bakedOverflows;
    }
}

std::string cutName(std::size_t cut) {
    switch (cut) {
        case CutFlip: return "flip";
        case CutRenderingWait: return "rendering wait";
        case CutPredicated: return "predicated";
        case CutIndirect: return "indirect draw";
        case CutLength: return "length cap";
        case CutSubmission: return "submission";
        default: {
            char name[16];
            std::snprintf(name, sizeof(name), "0x%02zx", cut);
            return name;
        }
    }
}

void report(Stream& stream, std::chrono::steady_clock::time_point now) {
    auto& window = stream.window;
    const auto seconds = std::chrono::duration<double>(now - stream.lastReport).count();
    stream.lastReport = now;
    const auto commitTotal = DrawPipeline::Queue0().CommitNsTotal();
    const auto committerNs = commitTotal - std::min(commitTotal, stream.lastCommitNs);
    stream.lastCommitNs = commitTotal;
    const auto frames = std::max<std::uint64_t>(window.frames, 1);
    const auto segments = std::max<std::uint64_t>(window.segments, 1);
    std::uint64_t failed = 0;
    for (const auto value : window.failures) failed += value;
    std::fprintf(stderr, "[replay-probe] %.0f s: %llu frames (%llu forced), segments/frame %.1f (min %llu, max %llu), steps/segment %.1f, commits/segment %.1f; hash hit prev %.1f%% / last-%zu %.1f%% (time %.1f%% / %.1f%%); baked unchanged %.1f%% of %llu hits (empty %llu; fail:",
        seconds, static_cast<unsigned long long>(window.frames), static_cast<unsigned long long>(window.forcedFrames), static_cast<double>(window.segments) / static_cast<double>(frames), static_cast<unsigned long long>(window.segments == 0 ? 0 : window.minSegments), static_cast<unsigned long long>(window.maxSegments),
        static_cast<double>(window.steps) / static_cast<double>(segments), static_cast<double>(window.commits) / static_cast<double>(segments),
        percent(window.hitPrevious, window.segments), FramesKept, percent(window.hitAny, window.segments), percent(window.hitPreviousNs, window.segmentNs), percent(window.hitAnyNs, window.segmentNs),
        percent(window.passed, window.hitAny), static_cast<unsigned long long>(window.hitAny), static_cast<unsigned long long>(window.emptyBaked));
    for (std::size_t i = 0; i < FailCount; ++i) {
        if (window.failures[i] != 0) std::fprintf(stderr, " %s %llu", FailureNames[i], static_cast<unsigned long long>(window.failures[i]));
    }
    std::fprintf(stderr, "%s), validation %.1f us/frame; clean %.1f%% (time %.1f%%); memo-path segments %.1f%% (time %.1f%%); baked ranges/segment %.1f (%llu overflowed); REPLAYABLE %.1f%% of committer time (%llu segments, segment time %.1f%% of %.1f ms committer)\n",
        failed == 0 ? " none" : "", static_cast<double>(window.validateNs) / 1e3 / static_cast<double>(frames),
        percent(window.clean, window.segments), percent(window.cleanNs, window.segmentNs), percent(window.memo, window.segments), percent(window.memoNs, window.segmentNs),
        static_cast<double>(window.bakedRanges) / static_cast<double>(segments), static_cast<unsigned long long>(window.bakedOverflows),
        percent(window.replayableNs, committerNs), static_cast<unsigned long long>(window.replayable), percent(window.segmentNs, committerNs), static_cast<double>(committerNs) / 1e6);
    const auto disqualifying = reasonMasks().first;
    std::string disqualifyText, otherText;
    auto reasonNs = window.reasonNs;
    auto reasonSegments = window.reasonSegments;
    for (int shown = 0, others = 0; shown < 8 || others < 6;) {
        std::size_t best = ReasonSlots;
        for (std::size_t i = 0; i < ReasonSlots; ++i) {
            if (reasonSegments[i] == 0) continue;
            if (best == ReasonSlots || reasonNs[i] > reasonNs[best] || (reasonNs[i] == reasonNs[best] && reasonSegments[i] > reasonSegments[best])) best = i;
        }
        if (best == ReasonSlots) break;
        char item[96];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f%%)", reasonName(best).c_str(), static_cast<unsigned long long>(reasonSegments[best]), percent(reasonNs[best], window.segmentNs));
        if (((disqualifying >> best) & 1u) != 0) {
            if (shown++ < 8) disqualifyText += item;
        } else if (others++ < 6) {
            otherText += item;
        }
        reasonSegments[best] = 0;
        reasonNs[best] = 0;
    }
    std::fprintf(stderr, "[replay-probe] not clean by reason (segments, time%%):%s; other events:%s\n", disqualifyText.empty() ? " none" : disqualifyText.c_str(), otherText.empty() ? " none" : otherText.c_str());
    std::string missText;
    for (std::size_t i = 0; i < MissCount; ++i) {
        if (window.misses[i] == 0) continue;
        char item[80];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f%%)", MissNames[i], static_cast<unsigned long long>(window.misses[i]), percent(window.missNs[i], window.segmentNs));
        missText += item;
    }
    std::string cutText;
    auto cuts = window.cuts;
    for (int shown = 0; shown < 8; ++shown) {
        const auto top = std::max_element(cuts.begin(), cuts.end());
        if (*top == 0) break;
        cutText += " " + cutName(static_cast<std::size_t>(top - cuts.begin())) + " " + std::to_string(*top);
        *top = 0;
    }
    std::fprintf(stderr, "[replay-probe] misses (segments, time%%):%s; cuts:%s; state-only segments dropped %llu\n", missText.empty() ? " none" : missText.c_str(), cutText.empty() ? " none" : cutText.c_str(), static_cast<unsigned long long>(window.emptySegments));
    window = {};
}

void flip(Stream& stream, bool forced) {
    stream.history.push_front(std::move(stream.current));
    stream.current = {};
    if (stream.history.size() >= 2) account(stream, stream.history[1]);
    while (stream.history.size() > FramesKept) stream.history.pop_back();
    if (forced) ++stream.window.forcedFrames;
    const auto now = std::chrono::steady_clock::now();
    if (!stream.started) {
        stream.started = true;
        stream.lastReport = now;
        stream.lastCommitNs = DrawPipeline::Queue0().CommitNsTotal();
        stream.window = {};
        return;
    }
    if (now - stream.lastReport >= std::chrono::seconds(10)) report(stream, now);
}

void close(Stream& stream, std::uint32_t cut) {
    if (stream.open == nullptr) return;
    auto segment = std::move(stream.open);
    stream.open = nullptr;
    ++stream.window.cuts[std::min<std::uint32_t>(cut, CutCount - 1)];
    if (segment->steps.empty()) {
        ++stream.window.emptySegments;
        return;
    }
    auto content = segment->key;
    for (const auto& step : segment->steps) content = mix(content, step.full);
    segment->content = content;
    ReplaySegment* matched = nullptr;
    for (std::size_t k = 0; k < stream.history.size() && k < FramesKept; ++k) {
        const auto found = stream.history[k].byContent.find(content);
        if (found == stream.history[k].byContent.end()) continue;
        matched = found->second;
        segment->hitPrevious = k == 0;
        segment->hitAny = true;
        break;
    }
    if (matched != nullptr) validate(stream, *matched, *segment, reasonMasks().first);
    else segment->miss = classifyMiss(stream, *segment);
    auto* raw = segment.get();
    stream.current.byContent.emplace(content, raw);
    stream.current.byKey.emplace(segment->key, raw);
    stream.current.segments.push_back(std::move(segment));
}

void openSegment(Stream& stream, const QueueState& queue, std::uint64_t device, const void* registry) {
    if (stream.current.segments.size() >= FrameCapacity) flip(stream, true);
    auto segment = std::make_shared<ReplaySegment>();
    std::uint64_t user = Seed, other = Seed;
    for (const auto& entry : queue.shader) {
        const auto word = (static_cast<std::uint64_t>(entry.first) << 32u) | entry.second;
        if (userDataRegister(entry.first)) user = mix(user, word);
        else other = mix(other, word);
    }
    other = mix(other, 0x1000000000000ull);
    for (const auto& entry : queue.context) other = mix(other, (static_cast<std::uint64_t>(entry.first) << 32u) | entry.second);
    other = mix(other, 0x2000000000000ull);
    for (const auto& entry : queue.userConfig) other = mix(other, (static_cast<std::uint64_t>(entry.first) << 32u) | entry.second);
    other = mix(mix(mix(other, queue.indexBase), queue.drawIndirectBase), queue.dispatchIndirectBase);
    other = mix(mix(mix(other, queue.indexBufferSize), queue.indexType), queue.instanceCount);
    other = mix(mix(mix(other, queue.predication.address), queue.predication.operation), (queue.predication.executeWhenSet ? 1u : 0u) | (queue.savedContext.has_value() ? 2u : 0u));
    segment->keyUser = user;
    segment->keyOther = other;
    segment->device = device;
    segment->registry = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(registry));
    segment->key = mix(mix(mix(mix(Seed, user), other), segment->device), segment->registry);
    segment->startGeneration = GuestMemory::TrackerGeneration();
    segment->ordinal = stream.current.segments.size();
    stream.pendingUser = Seed;
    stream.pendingOther = Seed;
    stream.open = std::move(segment);
}

void hashState(Stream& stream, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> registers, std::uint32_t opcode) {
    if (opcode == 0x76 && packet.size() >= 3) {
        const auto base = packet[1] & ~0x70000000u;
        stream.pendingOther = mix(mix(stream.pendingOther, packet[0]), packet[1]);
        for (std::size_t i = 2; i < packet.size(); ++i) {
            const auto offset = base + static_cast<std::uint32_t>(i - 2);
            const auto word = (static_cast<std::uint64_t>(offset) << 32u) | packet[i];
            if (userDataRegister(offset)) stream.pendingUser = mix(stream.pendingUser, word);
            else stream.pendingOther = mix(stream.pendingOther, word);
        }
        return;
    }
    for (const auto word : packet) stream.pendingOther = mix(stream.pendingOther, word);
    for (const auto word : registers) stream.pendingOther = mix(stream.pendingOther, word);
}

}

bool Enabled() {
    return Graphics::ReplayProbeEnabled();
}

std::shared_ptr<ReplaySegment> Hold() {
    return currentSegment != nullptr ? currentSegment->shared_from_this() : nullptr;
}

std::shared_ptr<ReplaySegment> Tag() {
    if (currentSegment == nullptr) return nullptr;
    currentSegment->pending.fetch_add(1, std::memory_order_relaxed);
    return currentSegment->shared_from_this();
}

void Committed(ReplaySegment& segment, std::uint64_t ns, bool failed) {
    segment.commitNs.fetch_add(ns, std::memory_order_relaxed);
    segment.commits.fetch_add(1, std::memory_order_relaxed);
    if (failed) segment.reasons.fetch_or(std::uint64_t{1} << internReason("commit error"), std::memory_order_relaxed);
    segment.pending.fetch_sub(1, std::memory_order_release);
}

std::shared_ptr<ReplaySegment> Observe(const QueueState& queue, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> registers, std::uint32_t cut, const std::function<std::uint64_t()>& device, const void* registry) {
    auto& stream = Queue0();
    if (cut != NoCut) {
        close(stream, cut);
        if (cut == CutFlip) flip(stream, false);
        return nullptr;
    }
    if (stream.open != nullptr && stream.open->steps.size() >= SegmentLength) close(stream, CutLength);
    if (stream.open == nullptr) openSegment(stream, queue, device(), registry);
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    if (!Pm4::DrawOpcode(opcode) && opcode != 0x15) {
        hashState(stream, packet, registers, opcode);
        return stream.open;
    }
    auto item = Seed;
    for (const auto word : packet) item = mix(item, word);
    const auto full = mix(mix(mix(Seed, stream.pendingUser), stream.pendingOther), item);
    stream.open->steps.push_back({full, stream.pendingUser, stream.pendingOther, item});
    stream.pendingUser = Seed;
    stream.pendingOther = Seed;
    return stream.open;
}

void Cut(std::uint32_t reason) {
    close(Queue0(), reason);
}

Scope::Scope(std::shared_ptr<ReplaySegment> segment) : held(std::move(segment)) {
    if (held == nullptr) return;
    auto& sink = Sink();
    previous = currentSegment;
    currentSegment = held.get();
    sink.segment = held.get();
    sink.lastBytes = 0;
    previousSink = GuestMemory::SetProbeSink(&sink);
}

Scope::~Scope() {
    if (held == nullptr) return;
    auto& sink = Sink();
    GuestMemory::SetProbeSink(previousSink);
    sink.segment = previous;
    sink.lastBytes = 0;
    currentSegment = previous;
}

}

}
