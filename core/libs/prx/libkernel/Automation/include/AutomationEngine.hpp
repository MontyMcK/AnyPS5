#ifndef CORE_LIBS_PRX_LIBKERNEL_AUTOMATION_AUTOMATIONENGINE_HPP
#define CORE_LIBS_PRX_LIBKERNEL_AUTOMATION_AUTOMATIONENGINE_HPP

// The host-independent part of the automation channel: command tokens, button names and the
// timed-input engine that the guest's pad reads advance. Header-only so tests drive it directly.

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "prx/libkernel/Automation/include/Automation.hpp"

namespace Automation {

// libScePad's PadButton bits.
inline constexpr std::array<std::pair<std::string_view, std::uint32_t>, 22> ButtonNames{{
    {"l3", 0x0002}, {"r3", 0x0004}, {"options", 0x0008}, {"start", 0x0008},
    {"up", 0x0010}, {"right", 0x0020}, {"down", 0x0040}, {"left", 0x0080},
    {"l2", 0x0100}, {"r2", 0x0200}, {"l1", 0x0400}, {"r1", 0x0800},
    {"triangle", 0x1000}, {"circle", 0x2000}, {"cross", 0x4000}, {"square", 0x8000},
    {"touchpad", 0x100000}, {"touch", 0x100000},
    {"x", 0x4000}, {"o", 0x2000}, {"tri", 0x1000}, {"sq", 0x8000},
}};

inline std::string Lower(std::string_view text) {
    std::string out(text);
    for (auto& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// "cross" or "l1+cross": false on an unknown name.
inline bool ParseButtons(std::string_view spec, std::uint32_t& mask) {
    mask = 0;
    if (spec.empty()) return false;
    std::size_t start = 0;
    while (start <= spec.size()) {
        const auto end = std::min(spec.find('+', start), spec.size());
        const auto name = Lower(spec.substr(start, end - start));
        bool found = false;
        for (const auto& [known, bit] : ButtonNames) {
            if (known == name) {
                mask |= bit;
                found = true;
                break;
            }
        }
        if (!found) return false;
        start = end + 1;
    }
    return mask != 0;
}

inline std::string ButtonsName(std::uint32_t mask) {
    std::string out;
    static constexpr std::array<std::pair<std::string_view, std::uint32_t>, 17> canonical{{
        {"l3", 0x0002}, {"r3", 0x0004}, {"options", 0x0008}, {"up", 0x0010}, {"right", 0x0020},
        {"down", 0x0040}, {"left", 0x0080}, {"l2", 0x0100}, {"r2", 0x0200}, {"l1", 0x0400},
        {"r1", 0x0800}, {"triangle", 0x1000}, {"circle", 0x2000}, {"cross", 0x4000},
        {"square", 0x8000}, {"touchpad", 0x100000}, {"", 0},
    }};
    for (const auto& [name, bit] : canonical) {
        if (bit == 0 || (mask & bit) == 0) continue;
        if (!out.empty()) out.push_back('+');
        out += name;
    }
    return out;
}

// Splits a command line on blanks; "double quoted" tokens may hold blanks (\" and \\ escape).
inline bool Tokenize(std::string_view line, std::vector<std::string>& tokens, std::string& error) {
    tokens.clear();
    std::size_t i = 0;
    while (i < line.size()) {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
        if (i >= line.size()) break;
        std::string token;
        if (line[i] == '"') {
            ++i;
            bool closed = false;
            while (i < line.size()) {
                const char c = line[i++];
                if (c == '"') { closed = true; break; }
                if (c == '\\' && i < line.size() && (line[i] == '"' || line[i] == '\\')) token.push_back(line[i++]);
                else token.push_back(c);
            }
            if (!closed) { error = "unterminated quote"; return false; }
        } else {
            while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) token.push_back(line[i++]);
        }
        tokens.push_back(std::move(token));
    }
    return true;
}

inline std::optional<long long> ParseInteger(std::string_view text) {
    long long value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}

// Positional arguments and key=value options of one command (after the verb).
struct Arguments {
    std::vector<std::string> positional;
    std::vector<std::pair<std::string, std::string>> options;
    std::vector<std::string> flags;

    static Arguments From(const std::vector<std::string>& tokens, std::size_t first) {
        Arguments arguments;
        for (std::size_t i = first; i < tokens.size(); ++i) {
            const auto& token = tokens[i];
            const auto equals = token.find('=');
            if (equals != std::string::npos && equals > 0) arguments.options.emplace_back(Lower(token.substr(0, equals)), token.substr(equals + 1));
            else arguments.positional.push_back(token);
        }
        return arguments;
    }
    const std::string* Option(std::string_view key) const {
        for (const auto& [name, value] : options) if (name == key) return &value;
        return nullptr;
    }
    bool Has(std::string_view word) const {
        for (const auto& token : positional) if (Lower(token) == word) return true;
        return false;
    }
};

enum class Resource : std::uint8_t { Buttons, LeftStick, RightStick, L2, R2 };

inline const char* ResourceName(Resource resource) {
    switch (resource) {
    case Resource::Buttons: return "buttons";
    case Resource::LeftStick: return "left";
    case Resource::RightStick: return "right";
    case Resource::L2: return "l2";
    case Resource::R2: return "r2";
    }
    return "?";
}

enum class InputState : std::uint8_t { Pending, Active, Done, Cancelled };

inline const char* InputStateName(InputState state) {
    switch (state) {
    case InputState::Pending: return "queued";
    case InputState::Active: return "active";
    case InputState::Done: return "done";
    case InputState::Cancelled: return "cancelled";
    }
    return "?";
}

// One injected input. Timed inputs are released at the first guest pad read at which their
// wall-clock duration has elapsed AND they were reported in at least minPolls reads (and while at
// least minFrames presents happened); that read already reports them released. Holds last until
// released.
struct InputAction {
    std::uint64_t id = 0;
    Resource resource = Resource::Buttons;
    std::uint32_t buttons = 0;
    std::uint8_t x = 128;
    std::uint8_t y = 128;
    std::uint8_t value = 0;
    bool hold = false;
    std::uint64_t durationUs = 0;
    std::uint32_t minPolls = 0;
    std::uint32_t minFrames = 0;

    InputState state = InputState::Pending;
    std::uint64_t queuedUs = 0;
    std::uint64_t startUs = 0;
    std::uint64_t endUs = 0;
    std::uint32_t polls = 0;
    std::uint32_t frames = 0;
    std::uint64_t firstPoll = 0;
    std::uint64_t lastPoll = 0;

    bool Overlaps(const InputAction& other) const {
        if (resource != other.resource) return false;
        return resource != Resource::Buttons || (buttons & other.buttons) != 0;
    }
};

class PadEngine {
public:
    // Queues an input. It starts now unless an earlier input on an overlapping resource is still
    // queued or held, or one was released after the last guest read (each release is visible to
    // at least one read, so two presses of a button never merge). A new stick or trigger input
    // replaces a hold on the same stick or trigger; holding a held button again is a no-op that
    // returns the existing hold.
    std::uint64_t Add(InputAction action, std::uint64_t nowUs) {
        if (action.resource != Resource::Buttons) {
            for (auto& existing : actions) {
                if (existing.hold && existing.resource == action.resource) Finish(existing, InputState::Cancelled, nowUs, pollIndex + 1);
            }
            Prune();
        } else if (action.hold) {
            for (const auto& existing : actions) {
                if (existing.hold && existing.buttons == action.buttons && existing.state == InputState::Active) return existing.id;
            }
        }
        action.id = nextId++;
        action.state = InputState::Pending;
        action.queuedUs = nowUs;
        action.polls = 0;
        action.frames = 0;
        if (!Blocked(action, actions.size(), pollIndex + 1)) Start(action, nowUs);
        actions.push_back(action);
        return action.id;
    }

    // A guest pad read at nowUs.
    AutomationPadOverlay Poll(std::uint64_t nowUs) {
        ++pollIndex;
        for (auto& action : actions) {
            if (action.state != InputState::Active || action.hold) continue;
            if (action.polls >= action.minPolls && action.frames >= action.minFrames && nowUs - action.startUs >= action.durationUs) Finish(action, InputState::Done, nowUs, pollIndex);
        }
        Prune();
        for (std::size_t i = 0; i < actions.size(); ++i) {
            auto& action = actions[i];
            if (action.state == InputState::Pending && !Blocked(action, i, pollIndex)) Start(action, nowUs);
        }
        for (auto& action : actions) {
            if (action.state != InputState::Active) continue;
            if (action.polls == 0) action.firstPoll = pollIndex;
            ++action.polls;
            action.lastPoll = pollIndex;
        }
        std::erase_if(releases, [&](const ReleaseRecord& release) { return release.visiblePoll < pollIndex; });
        return Current();
    }

    // A presented frame: counted for every held input.
    void Present() {
        for (auto& action : actions) if (action.state == InputState::Active) ++action.frames;
    }

    // The overrides of the inputs held now, without a read.
    AutomationPadOverlay Current() const {
        AutomationPadOverlay overlay;
        for (const auto& action : actions) {
            if (action.state != InputState::Active) continue;
            switch (action.resource) {
            case Resource::Buttons: overlay.buttons |= action.buttons; break;
            case Resource::LeftStick: overlay.leftStick = true; overlay.sticks[0] = action.x; overlay.sticks[1] = action.y; break;
            case Resource::RightStick: overlay.rightStick = true; overlay.sticks[2] = action.x; overlay.sticks[3] = action.y; break;
            case Resource::L2: overlay.l2 = std::max(overlay.l2, action.value); break;
            case Resource::R2: overlay.r2 = std::max(overlay.r2, action.value); break;
            }
        }
        return overlay;
    }

    // Releases (cancels) the queued and held inputs on these buttons or this stick/trigger.
    std::size_t Release(Resource resource, std::uint32_t buttons, std::uint64_t nowUs) {
        InputAction probe;
        probe.resource = resource;
        probe.buttons = buttons;
        std::size_t released = 0;
        for (auto& action : actions) {
            if (!action.Overlaps(probe)) continue;
            Finish(action, InputState::Cancelled, nowUs, pollIndex + 1);
            ++released;
        }
        Prune();
        return released;
    }

    std::size_t ReleaseAll(std::uint64_t nowUs) {
        const auto count = actions.size();
        for (auto& action : actions) Finish(action, InputState::Cancelled, nowUs, pollIndex + 1);
        Prune();
        return count;
    }

    bool Cancel(std::uint64_t id, std::uint64_t nowUs) {
        for (auto& action : actions) {
            if (action.id != id) continue;
            Finish(action, InputState::Cancelled, nowUs, pollIndex + 1);
            Prune();
            return true;
        }
        return false;
    }

    const InputAction* Find(std::uint64_t id) const {
        for (const auto& action : actions) if (action.id == id) return &action;
        return nullptr;
    }

    // Inputs finished or cancelled since the last call.
    std::vector<InputAction> TakeFinished() {
        std::vector<InputAction> out;
        out.swap(finished);
        return out;
    }

    const std::deque<InputAction>& Actions() const { return actions; }
    std::uint64_t Polls() const { return pollIndex; }

private:
    struct ReleaseRecord {
        InputAction action;
        std::uint64_t visiblePoll;
    };

    // Whether the input at queue position `position` may not start in read `poll`.
    bool Blocked(const InputAction& action, std::size_t position, std::uint64_t poll) const {
        for (std::size_t i = 0; i < position && i < actions.size(); ++i) {
            const auto& earlier = actions[i];
            if ((earlier.state == InputState::Pending || earlier.state == InputState::Active) && earlier.Overlaps(action)) return true;
        }
        // A button's release must be seen by one read before it is pressed again; sticks and
        // triggers take new values at once.
        if (action.resource != Resource::Buttons) return false;
        for (const auto& release : releases) {
            if (release.visiblePoll >= poll && release.action.Overlaps(action)) return true;
        }
        return false;
    }

    static void Start(InputAction& action, std::uint64_t nowUs) {
        action.state = InputState::Active;
        action.startUs = nowUs;
    }

    void Finish(InputAction& action, InputState state, std::uint64_t nowUs, std::uint64_t visiblePoll) {
        if (action.state == InputState::Done || action.state == InputState::Cancelled) return;
        const bool wasActive = action.state == InputState::Active;
        action.state = state;
        action.endUs = nowUs;
        if (wasActive) releases.push_back({action, visiblePoll});
    }

    void Prune() {
        for (auto it = actions.begin(); it != actions.end();) {
            if (it->state == InputState::Done || it->state == InputState::Cancelled) {
                finished.push_back(*it);
                it = actions.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::deque<InputAction> actions;
    std::vector<InputAction> finished;
    std::vector<ReleaseRecord> releases;
    std::uint64_t nextId = 1;
    std::uint64_t pollIndex = 0;
};

}

#endif
