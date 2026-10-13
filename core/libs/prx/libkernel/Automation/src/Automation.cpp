// The automation channel (see Automation.hpp): a named-pipe command server, the timed pad inputs
// it injects, snapshot requests for the presenter, and the event log.
//
// Protocol: one command per line, one JSON reply line per command, on \\.\pipe\aps5-<pid>
// (APS5_AUTOMATION=1) or \\.\pipe\<name> (APS5_AUTOMATION=<name>). Several clients may connect at
// once. Events also go to APS5_AUTOMATION_EVENTS (default automation-events.jsonl in the working
// directory), flushed every 100 ms. Commands:
//   ping | status | help
//   press <buttons> [ms=100] [polls=2] [frames=0] [timeout=30000] [nowait]   e.g. press l1+cross 300 polls=3
//   hold <buttons> [polls=N]        release <buttons|left-stick|right-stick|l2-trigger|r2-trigger|all>
//   stick <left|right> <x 0..255> <y 0..255> [ms] [polls=1] [frames=0]   (no ms: held until changed)
//   trigger <l2|r2> <0..255> [ms] [polls=1] [frames=0]                    (no ms: held; 0 releases)
//   wait [id=N] [polls=N] [presents=N] [timeout=ms]
//   snapshot <path.bmp|path.png> [after=1] [timeout=30000]   the after-th frame presented from now
//   events [since=0] [timeout=0] [max=2000]                  long-polls the event ring
//   mark <text>                                              writes a "mark" event

#include "prx/libkernel/Automation/include/Automation.hpp"
#include "prx/libkernel/Automation/include/AutomationEngine.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using Automation::Arguments;
using Automation::InputAction;
using Automation::InputState;
using Automation::Resource;

constexpr std::size_t EventRingCapacity = 50000;
constexpr std::size_t ResultCapacity = 4096;

struct SnapshotJob {
    enum class State { Waiting, Capturing, Done } state = State::Waiting;
    std::uint64_t id = 0;
    std::string path;
    std::uint64_t duePresent = 0;
    bool ok = false;
    std::string fields;
};

class ArgumentError : public std::exception {
public:
    explicit ArgumentError(std::string text) : message(std::move(text)) {}
    const char* what() const noexcept override { return message.c_str(); }
private:
    std::string message;
};

std::string Reply(bool ok, const std::string& fields) {
    std::string out = ok ? "{\"ok\":true" : "{\"ok\":false";
    if (!fields.empty()) {
        out.push_back(',');
        out += fields;
    }
    out.push_back('}');
    return out;
}

std::string Error(const std::string& message) {
    return Reply(false, Automation::Fields().Str("error", message).Text());
}

long long IntegerArgument(const std::string& text, const char* what, long long low, long long high) {
    const auto value = Automation::ParseInteger(text);
    if (!value || *value < low || *value > high) throw ArgumentError(std::string(what) + " must be an integer in [" + std::to_string(low) + ", " + std::to_string(high) + "], got '" + text + "'");
    return *value;
}

long long OptionArgument(const Arguments& arguments, const char* key, long long fallback, long long low, long long high) {
    const auto* text = arguments.Option(key);
    return text ? IntegerArgument(*text, key, low, high) : fallback;
}

class Channel {
public:
    explicit Channel(std::string pipe, std::string events) : pipeName(std::move(pipe)), eventsPath(std::move(events)), started(Clock::now()) {}

    void Start() {
        eventsFile = std::fopen(eventsPath.c_str(), "wb");
        if (eventsFile == nullptr) std::fprintf(stderr, "[automation] cannot open the event log %s; events stay on the pipe only\n", eventsPath.c_str());
        std::fprintf(stderr, "[automation] commands on %s, events in %s\n", pipeName.c_str(), eventsPath.c_str());
        std::fflush(stderr);
#ifdef _WIN32
        Event("automation", Automation::Fields().Str("pipe", pipeName).Str("events", eventsPath).Unsigned("pid", GetCurrentProcessId()).Text());
        std::thread([this] { Listen(); }).detach();
#endif
        std::thread([this] { Tick(); }).detach();
    }

    std::uint64_t NowUs() const {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count());
    }

    void Event(const char* type, const std::string& fields) {
        std::lock_guard lock(mutex);
        EventLocked(type, fields);
    }

    AutomationPadOverlay PadPoll() {
        std::lock_guard lock(mutex);
        const auto overlay = pad.Poll(NowUs());
        CollectFinished();
        changed.notify_all();
        return overlay;
    }

    bool Presented(std::uint32_t width, std::uint32_t height, AutomationSnapshotRequest* request) {
        std::lock_guard lock(mutex);
        ++presents;
        lastWidth = width;
        lastHeight = height;
        pad.Present();
        bool due = false;
        for (auto& job : snapshots) {
            if (job.state != SnapshotJob::State::Waiting || presents < job.duePresent) continue;
            job.state = SnapshotJob::State::Capturing;
            request->id = job.id;
            std::snprintf(request->path, sizeof(request->path), "%s", job.path.c_str());
            due = true;
            break;
        }
        changed.notify_all();
        return due;
    }

    void SnapshotDone(std::uint64_t id, bool ok, const char* fields) {
        std::lock_guard lock(mutex);
        for (auto& job : snapshots) {
            if (job.id != id) continue;
            job.state = SnapshotJob::State::Done;
            job.ok = ok;
            job.fields = fields != nullptr ? fields : "";
            auto event = Automation::Fields().Unsigned("id", id).Bool("ok", ok).Unsigned("present", presents).Text();
            if (!job.fields.empty()) event += "," + job.fields;
            EventLocked("snapshot", event);
        }
        changed.notify_all();
    }

    std::string Execute(const std::string& line) {
        std::vector<std::string> tokens;
        std::string error;
        if (!Automation::Tokenize(line, tokens, error)) return Error(error);
        if (tokens.empty()) return Error("empty command");
        const auto verb = Automation::Lower(tokens[0]);
        const auto arguments = Arguments::From(tokens, 1);
        try {
            if (verb == "ping") return Reply(true, Automation::Fields().Real("t", NowUs() / 1e6).Unsigned("pid", Pid()).Text());
            if (verb == "status") return Status();
            if (verb == "help") return Reply(true, Automation::Fields().Str("commands", "ping status press hold release stick trigger wait snapshot events mark").Text());
            if (verb == "press") return Press(arguments);
            if (verb == "hold") return Hold(arguments);
            if (verb == "release") return ReleaseCommand(arguments);
            if (verb == "stick") return Stick(arguments);
            if (verb == "trigger") return Trigger(arguments);
            if (verb == "wait") return Wait(arguments);
            if (verb == "snapshot") return Snapshot(arguments);
            if (verb == "events") return Events(arguments);
            if (verb == "mark") {
                const auto text = line.size() > tokens[0].size() ? line.substr(line.find(tokens[0]) + tokens[0].size() + 1) : std::string{};
                Event("mark", Automation::Fields().Str("text", text).Text());
                return Reply(true, {});
            }
            return Error("unknown command '" + tokens[0] + "' (try help)");
        } catch (const ArgumentError& failure) {
            return Error(failure.what());
        } catch (const std::exception& failure) {
            return Error(std::string("internal error: ") + failure.what());
        }
    }

private:
    static unsigned long Pid() {
#ifdef _WIN32
        return GetCurrentProcessId();
#else
        return 0;
#endif
    }

    void EventLocked(const char* type, const std::string& fields) {
        const auto wall = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        char head[96];
        std::snprintf(head, sizeof(head), "{\"seq\":%llu,\"t\":%.3f,\"wall\":%lld,\"type\":", static_cast<unsigned long long>(nextSeq), NowUs() / 1e6, static_cast<long long>(wall));
        std::string line = head;
        Automation::AppendJsonString(line, type);
        if (!fields.empty()) {
            line.push_back(',');
            line += fields;
        }
        line.push_back('}');
        pendingFile += line;
        pendingFile.push_back('\n');
        ring.push_back(std::move(line));
        ++nextSeq;
        if (ring.size() > EventRingCapacity) ring.pop_front();
        changed.notify_all();
    }

    std::uint64_t FirstSeq() const { return nextSeq - ring.size(); }

    static std::string ActionFields(const InputAction& action) {
        Automation::Fields fields;
        fields.Unsigned("id", action.id).Str("state", Automation::InputStateName(action.state)).Str("input", Automation::ResourceName(action.resource));
        if (action.resource == Resource::Buttons) fields.Str("buttons", Automation::ButtonsName(action.buttons));
        if (action.resource == Resource::LeftStick || action.resource == Resource::RightStick) fields.Num("x", action.x).Num("y", action.y);
        if (action.resource == Resource::L2 || action.resource == Resource::R2) fields.Num("value", action.value);
        fields.Bool("hold", action.hold).Unsigned("polls", action.polls).Unsigned("frames", action.frames);
        if (action.state == InputState::Done || action.state == InputState::Cancelled) {
            const auto start = action.startUs != 0 ? action.startUs : action.endUs;
            fields.Real("held_ms", (action.endUs - start) / 1000.0).Real("queued_ms", (start - action.queuedUs) / 1000.0);
        }
        if (action.polls != 0) fields.Unsigned("first_poll", action.firstPoll).Unsigned("last_poll", action.lastPoll);
        return fields.Text();
    }

    // Moves finished inputs to the results (acked on the pipe and in the log). Under the mutex.
    void CollectFinished() {
        for (auto& action : pad.TakeFinished()) {
            EventLocked("input", ActionFields(action));
            results[action.id] = action;
            if (results.size() > ResultCapacity) results.erase(results.begin());
        }
    }

    // Waits for input `id` to finish; on timeout it is cancelled (released).
    std::string AwaitInput(std::uint64_t id, long long timeoutMs) {
        std::unique_lock lock(mutex);
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
        const bool finished = changed.wait_until(lock, deadline, [&] { return results.count(id) != 0; });
        if (!finished) {
            const auto* active = pad.Find(id);
            const auto progress = active != nullptr ? ActionFields(*active) : std::string{};
            pad.Cancel(id, NowUs());
            CollectFinished();
            return Reply(false, Automation::Fields().Str("error", "timeout: the input did not complete (the title stopped reading the pad?)").Text() + (progress.empty() ? "" : "," + progress));
        }
        const auto& action = results[id];
        return Reply(action.state == InputState::Done, ActionFields(action));
    }

    std::string Status() {
        std::lock_guard lock(mutex);
        Automation::Fields fields;
        fields.Unsigned("pid", Pid()).Real("t", NowUs() / 1e6).Unsigned("polls", pad.Polls()).Unsigned("presents", presents)
            .Real("fps", fps).Real("polls_per_s", pollRate).Unsigned("width", lastWidth).Unsigned("height", lastHeight)
            .Unsigned("next_seq", nextSeq).Str("pipe", pipeName).Str("events", eventsPath);
        std::string active = "[";
        for (const auto& action : pad.Actions()) {
            if (active.size() > 1) active.push_back(',');
            active += "{" + ActionFields(action) + "}";
        }
        active.push_back(']');
        return Reply(true, fields.Text() + ",\"inputs\":" + active);
    }

    InputAction TimedFrom(const Arguments& arguments, std::size_t msIndex, long long defaultMs, long long defaultPolls) {
        InputAction action;
        const bool hasMs = arguments.positional.size() > msIndex && !Automation::Lower(arguments.positional[msIndex]).starts_with("nowait");
        const auto ms = arguments.Option("ms") ? OptionArgument(arguments, "ms", defaultMs, 0, 600000) : hasMs ? IntegerArgument(arguments.positional[msIndex], "ms", 0, 600000) : defaultMs;
        action.hold = ms < 0;
        action.durationUs = ms < 0 ? 0 : static_cast<std::uint64_t>(ms) * 1000;
        action.minPolls = static_cast<std::uint32_t>(OptionArgument(arguments, "polls", defaultPolls, 0, 100000));
        action.minFrames = static_cast<std::uint32_t>(OptionArgument(arguments, "frames", 0, 0, 100000));
        return action;
    }

    std::string Submit(const InputAction& action, const Arguments& arguments) {
        const auto timeout = OptionArgument(arguments, "timeout", 30000, 1, 3600000);
        std::uint64_t id = 0;
        {
            std::lock_guard lock(mutex);
            id = pad.Add(action, NowUs());
            CollectFinished();
            if (action.hold || arguments.Has("nowait")) {
                const auto* added = pad.Find(id);
                return Reply(true, added != nullptr ? ActionFields(*added) : Automation::Fields().Unsigned("id", id).Text());
            }
        }
        return AwaitInput(id, timeout);
    }

    std::string Press(const Arguments& arguments) {
        if (arguments.positional.empty()) throw ArgumentError("usage: press <buttons> [ms] [polls=N] [frames=N] [timeout=ms] [nowait]");
        auto action = TimedFrom(arguments, 1, 100, 2);
        if (!Automation::ParseButtons(arguments.positional[0], action.buttons)) throw ArgumentError("unknown button in '" + arguments.positional[0] + "'");
        action.resource = Resource::Buttons;
        action.hold = false;
        return Submit(action, arguments);
    }

    std::string Hold(const Arguments& arguments) {
        if (arguments.positional.empty()) throw ArgumentError("usage: hold <buttons> [polls=N]");
        InputAction action;
        if (!Automation::ParseButtons(arguments.positional[0], action.buttons)) throw ArgumentError("unknown button in '" + arguments.positional[0] + "'");
        action.hold = true;
        std::uint64_t id = 0;
        {
            std::lock_guard lock(mutex);
            id = pad.Add(action, NowUs());
        }
        return AwaitSeen(id, arguments);
    }

    // For holds: optionally waits until `polls=N` guest reads reported the input.
    std::string AwaitSeen(std::uint64_t id, const Arguments& arguments) {
        const auto polls = OptionArgument(arguments, "polls", 0, 0, 100000);
        const auto timeout = OptionArgument(arguments, "timeout", 30000, 1, 3600000);
        std::unique_lock lock(mutex);
        const bool seen = changed.wait_for(lock, std::chrono::milliseconds(timeout), [&] {
            const auto* action = pad.Find(id);
            return action == nullptr || action->polls >= polls;
        });
        const auto* action = pad.Find(id);
        if (action == nullptr) return results.count(id) ? Reply(false, ActionFields(results[id])) : Error("input vanished");
        return Reply(seen, ActionFields(*action) + (seen ? "" : ",\"error\":\"timeout before the requested polls\""));
    }

    std::string ReleaseCommand(const Arguments& arguments) {
        if (arguments.positional.empty()) throw ArgumentError("usage: release <buttons|left-stick|right-stick|l2-trigger|r2-trigger|all>");
        const auto target = Automation::Lower(arguments.positional[0]);
        std::lock_guard lock(mutex);
        std::size_t released = 0;
        const auto now = NowUs();
        if (target == "all") released = pad.ReleaseAll(now);
        else if (target == "left-stick" || target == "lstick") released = pad.Release(Resource::LeftStick, 0, now);
        else if (target == "right-stick" || target == "rstick") released = pad.Release(Resource::RightStick, 0, now);
        else if (target == "l2-trigger") released = pad.Release(Resource::L2, 0, now);
        else if (target == "r2-trigger") released = pad.Release(Resource::R2, 0, now);
        else {
            std::uint32_t buttons = 0;
            if (!Automation::ParseButtons(target, buttons)) throw ArgumentError("unknown release target '" + target + "'");
            released = pad.Release(Resource::Buttons, buttons, now);
        }
        CollectFinished();
        return Reply(true, Automation::Fields().Unsigned("released", released).Text());
    }

    std::string Stick(const Arguments& arguments) {
        if (arguments.positional.size() < 3) throw ArgumentError("usage: stick <left|right> <x 0..255> <y 0..255> [ms] [polls=N] [frames=N]");
        const auto which = Automation::Lower(arguments.positional[0]);
        if (which != "left" && which != "right") throw ArgumentError("stick must be left or right");
        auto action = TimedFrom(arguments, 3, -1, 1);
        action.resource = which == "left" ? Resource::LeftStick : Resource::RightStick;
        action.x = static_cast<std::uint8_t>(IntegerArgument(arguments.positional[1], "x", 0, 255));
        action.y = static_cast<std::uint8_t>(IntegerArgument(arguments.positional[2], "y", 0, 255));
        if (action.hold && action.x == 128 && action.y == 128) {
            std::lock_guard lock(mutex);
            const auto released = pad.Release(action.resource, 0, NowUs());
            CollectFinished();
            return Reply(true, Automation::Fields().Unsigned("released", released).Text());
        }
        return Submit(action, arguments);
    }

    std::string Trigger(const Arguments& arguments) {
        if (arguments.positional.size() < 2) throw ArgumentError("usage: trigger <l2|r2> <0..255> [ms] [polls=N] [frames=N]");
        const auto which = Automation::Lower(arguments.positional[0]);
        if (which != "l2" && which != "r2") throw ArgumentError("trigger must be l2 or r2");
        auto action = TimedFrom(arguments, 2, -1, 1);
        action.resource = which == "l2" ? Resource::L2 : Resource::R2;
        action.value = static_cast<std::uint8_t>(IntegerArgument(arguments.positional[1], "value", 0, 255));
        if (action.hold && action.value == 0) {
            std::lock_guard lock(mutex);
            const auto released = pad.Release(action.resource, 0, NowUs());
            CollectFinished();
            return Reply(true, Automation::Fields().Unsigned("released", released).Text());
        }
        return Submit(action, arguments);
    }

    std::string Wait(const Arguments& arguments) {
        const auto timeout = OptionArgument(arguments, "timeout", 30000, 1, 3600000);
        if (const auto* id = arguments.Option("id")) return AwaitInput(static_cast<std::uint64_t>(IntegerArgument(*id, "id", 1, 1LL << 62)), timeout);
        const auto polls = OptionArgument(arguments, "polls", 0, 0, 1LL << 40);
        const auto presentCount = OptionArgument(arguments, "presents", 0, 0, 1LL << 40);
        std::unique_lock lock(mutex);
        const auto pollTarget = pad.Polls() + static_cast<std::uint64_t>(polls);
        const auto presentTarget = presents + static_cast<std::uint64_t>(presentCount);
        const bool reached = changed.wait_for(lock, std::chrono::milliseconds(timeout), [&] { return pad.Polls() >= pollTarget && presents >= presentTarget; });
        const auto fields = Automation::Fields().Unsigned("polls", pad.Polls()).Unsigned("presents", presents).Text();
        return Reply(reached, reached ? fields : fields + ",\"error\":\"timeout\"");
    }

    std::string Snapshot(const Arguments& arguments) {
        if (arguments.positional.empty()) throw ArgumentError("usage: snapshot <path.bmp|path.png> [after=1] [timeout=ms]");
        const auto after = OptionArgument(arguments, "after", 1, 1, 1000);
        const auto timeout = OptionArgument(arguments, "timeout", 30000, 1, 3600000);
        std::unique_lock lock(mutex);
        SnapshotJob job;
        job.id = nextSnapshot++;
        job.path = arguments.positional[0];
        job.duePresent = presents + static_cast<std::uint64_t>(after);
        snapshots.push_back(job);
        const auto id = job.id;
        const auto find = [&]() -> SnapshotJob* {
            for (auto& entry : snapshots) if (entry.id == id) return &entry;
            return nullptr;
        };
        const bool done = changed.wait_for(lock, std::chrono::milliseconds(timeout), [&] { return find()->state == SnapshotJob::State::Done; });
        auto* entry = find();
        std::string reply;
        if (done) {
            reply = Reply(entry->ok, Automation::Fields().Unsigned("id", id).Str("path", entry->path).Unsigned("present", presents).Text() + (entry->fields.empty() ? "" : "," + entry->fields));
        } else {
            const bool capturing = entry->state == SnapshotJob::State::Capturing;
            reply = Error(capturing ? "timeout while the presenter captured the frame" : "timeout: no frame was presented");
        }
        // A capture still in progress reports to a job that no longer exists, which is ignored.
        std::erase_if(snapshots, [&](const SnapshotJob& candidate) { return candidate.id == id; });
        return reply;
    }

    std::string Events(const Arguments& arguments) {
        auto since = static_cast<std::uint64_t>(arguments.Option("since") ? OptionArgument(arguments, "since", 0, 0, 1LL << 62) : !arguments.positional.empty() ? IntegerArgument(arguments.positional[0], "since", 0, 1LL << 62) : 0);
        const auto timeout = OptionArgument(arguments, "timeout", 0, 0, 3600000);
        const auto max = static_cast<std::uint64_t>(OptionArgument(arguments, "max", 2000, 1, EventRingCapacity));
        std::unique_lock lock(mutex);
        if (timeout > 0) changed.wait_for(lock, std::chrono::milliseconds(timeout), [&] { return nextSeq > since; });
        const auto first = FirstSeq();
        const auto dropped = since < first ? first - since : 0;
        if (since < first) since = first;
        std::string list = "[";
        std::uint64_t seq = since;
        for (; seq < nextSeq && seq - since < max; ++seq) {
            if (list.size() > 1) list.push_back(',');
            list += ring[static_cast<std::size_t>(seq - first)];
        }
        list.push_back(']');
        return Reply(true, Automation::Fields().Unsigned("next", seq).Unsigned("dropped", dropped).Text() + ",\"events\":" + list);
    }

    void Tick() {
        auto lastRate = Clock::now();
        std::uint64_t lastPresents = 0;
        std::uint64_t lastPolls = 0;
        bool reportedIdle = false;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            std::string chunk;
            {
                std::lock_guard lock(mutex);
                const auto now = Clock::now();
                const double seconds = std::chrono::duration<double>(now - lastRate).count();
                if (seconds >= 1.0) {
                    const auto polls = pad.Polls();
                    fps = static_cast<double>(presents - lastPresents) / seconds;
                    pollRate = static_cast<double>(polls - lastPolls) / seconds;
                    const bool moving = presents != lastPresents || polls != lastPolls;
                    if (moving || !reportedIdle) {
                        EventLocked("fps", Automation::Fields().Real("fps", fps).Real("polls_per_s", pollRate).Unsigned("presents", presents).Unsigned("polls", polls).Text());
                        reportedIdle = !moving;
                    }
                    lastPresents = presents;
                    lastPolls = polls;
                    lastRate = now;
                }
                chunk.swap(pendingFile);
            }
            if (eventsFile != nullptr && !chunk.empty()) {
                std::fwrite(chunk.data(), 1, chunk.size(), eventsFile);
                std::fflush(eventsFile);
            }
        }
    }

#ifdef _WIN32
    void Listen() {
        bool first = true;
        for (;;) {
            const DWORD openMode = PIPE_ACCESS_DUPLEX | (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0);
            HANDLE pipe = CreateNamedPipeA(pipeName.c_str(), openMode, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, PIPE_UNLIMITED_INSTANCES, 1 << 16, 1 << 16, 0, nullptr);
            if (pipe == INVALID_HANDLE_VALUE) {
                std::fprintf(stderr, "[automation] CreateNamedPipe(%s) failed: error %lu\n", pipeName.c_str(), GetLastError());
                std::fflush(stderr);
                if (first) return;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            first = false;
            const bool connected = ConnectNamedPipe(pipe, nullptr) ? true : GetLastError() == ERROR_PIPE_CONNECTED;
            if (!connected) {
                CloseHandle(pipe);
                continue;
            }
            std::thread([this, pipe] { Serve(pipe); }).detach();
        }
    }

    void Serve(HANDLE pipe) {
        std::string buffer;
        char chunk[4096];
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(pipe, chunk, sizeof(chunk), &got, nullptr) || got == 0) break;
            buffer.append(chunk, got);
            bool alive = true;
            for (auto newline = buffer.find('\n'); newline != std::string::npos && alive; newline = buffer.find('\n')) {
                std::string line = buffer.substr(0, newline);
                buffer.erase(0, newline + 1);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.find_first_not_of(" \t") == std::string::npos) continue;
                std::string reply = Execute(line);
                reply.push_back('\n');
                std::size_t written = 0;
                while (written < reply.size()) {
                    DWORD wrote = 0;
                    if (!WriteFile(pipe, reply.data() + written, static_cast<DWORD>(reply.size() - written), &wrote, nullptr) || wrote == 0) {
                        alive = false;
                        break;
                    }
                    written += wrote;
                }
            }
            if (!alive) break;
        }
        FlushFileBuffers(pipe);
        DisconnectNamedPipe(pipe);
        CloseHandle(pipe);
    }
#endif

    const std::string pipeName;
    const std::string eventsPath;
    const Clock::time_point started;
    std::FILE* eventsFile = nullptr;

    std::mutex mutex;
    std::condition_variable changed;
    Automation::PadEngine pad;
    std::map<std::uint64_t, InputAction> results;
    std::deque<std::string> ring;
    std::uint64_t nextSeq = 0;
    std::string pendingFile;
    std::uint64_t presents = 0;
    std::uint32_t lastWidth = 0;
    std::uint32_t lastHeight = 0;
    double fps = 0.0;
    double pollRate = 0.0;
    std::vector<SnapshotJob> snapshots;
    std::uint64_t nextSnapshot = 1;
};

// The channel, or null when APS5_AUTOMATION is unset. Never destroyed: its threads outlive static
// destruction at exit.
Channel* Instance() {
    static Channel* channel = [] () -> Channel* {
        const char* value = std::getenv("APS5_AUTOMATION");
        if (value == nullptr) return nullptr;
        const auto setting = Automation::Lower(value);
        if (setting.empty() || setting == "0" || setting == "off" || setting == "false" || setting == "no") return nullptr;
#ifdef _WIN32
        std::string pipe;
        if (setting == "1" || setting == "on" || setting == "true" || setting == "yes" || setting == "pipe") pipe = "\\\\.\\pipe\\aps5-" + std::to_string(GetCurrentProcessId());
        else if (std::strncmp(value, "\\\\.\\pipe\\", 9) == 0) pipe = value;
        else pipe = std::string("\\\\.\\pipe\\") + value;
#else
        std::fprintf(stderr, "[automation] APS5_AUTOMATION needs Windows named pipes; ignored\n");
        return nullptr;
        std::string pipe;
#endif
        const char* events = std::getenv("APS5_AUTOMATION_EVENTS");
        std::error_code error;
        const std::filesystem::path eventsPath = events != nullptr && *events != '\0' ? events : "automation-events.jsonl";
        const auto absolute = std::filesystem::absolute(eventsPath, error);
        auto* created = new Channel(pipe, (error ? eventsPath : absolute).string());
        created->Start();
        return created;
    }();
    return channel;
}

}

extern "C" {

bool Aps5AutomationEnabled_nid_postfix() {
    try {
        return Instance() != nullptr;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[automation] disabled: %s\n", error.what());
        return false;
    }
}

void Aps5AutomationEvent_nid_postfix(const char* type, const char* fields) {
    auto* channel = Instance();
    if (channel == nullptr || type == nullptr) return;
    try {
        channel->Event(type, fields != nullptr ? fields : "");
    } catch (...) {
    }
}

void Aps5AutomationPadPoll_nid_postfix(AutomationPadOverlay* overlay) {
    auto* channel = Instance();
    if (channel == nullptr || overlay == nullptr) return;
    try {
        *overlay = channel->PadPoll();
    } catch (...) {
        *overlay = AutomationPadOverlay{};
    }
}

bool Aps5AutomationPresented_nid_postfix(std::uint32_t width, std::uint32_t height, AutomationSnapshotRequest* request) {
    auto* channel = Instance();
    if (channel == nullptr || request == nullptr) return false;
    try {
        return channel->Presented(width, height, request);
    } catch (...) {
        return false;
    }
}

void Aps5AutomationSnapshotDone_nid_postfix(std::uint64_t id, bool ok, const char* fields) {
    auto* channel = Instance();
    if (channel == nullptr) return;
    try {
        channel->SnapshotDone(id, ok, fields);
    } catch (...) {
    }
}

}
