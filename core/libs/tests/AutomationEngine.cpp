// The automation channel's host-independent parts: command tokens, button names and the timed
// pad-input engine driven by simulated guest pad reads.
#include "prx/libkernel/Automation/include/AutomationEngine.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

void Check(bool condition, const char* what, int line) {
    if (condition) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
    ++failures;
}
#define CHECK(condition) Check((condition), #condition, __LINE__)

constexpr std::uint32_t Cross = 0x4000;
constexpr std::uint32_t Circle = 0x2000;
constexpr std::uint32_t L1 = 0x0400;

constexpr std::uint64_t Ms = 1000;

Automation::InputAction Press(std::uint32_t buttons, std::uint64_t ms, std::uint32_t polls, std::uint32_t frames = 0) {
    Automation::InputAction action;
    action.resource = Automation::Resource::Buttons;
    action.buttons = buttons;
    action.durationUs = ms * Ms;
    action.minPolls = polls;
    action.minFrames = frames;
    return action;
}

// Polls every `periodMs` from `startMs` and returns the per-read Cross state until it is released.
std::vector<bool> PollUntilReleased(Automation::PadEngine& engine, std::uint32_t mask, std::uint64_t startMs, std::uint64_t periodMs, int maxPolls = 1000) {
    std::vector<bool> states;
    for (int i = 0; i < maxPolls; ++i) {
        const auto overlay = engine.Poll((startMs + periodMs * static_cast<std::uint64_t>(i)) * Ms);
        const bool down = (overlay.buttons & mask) != 0;
        states.push_back(down);
        if (!down && engine.Actions().empty()) break;
    }
    return states;
}

void Tokens() {
    std::vector<std::string> tokens;
    std::string error;
    CHECK(Automation::Tokenize("press  cross 300 polls=3", tokens, error));
    CHECK((tokens == std::vector<std::string>{"press", "cross", "300", "polls=3"}));
    CHECK(Automation::Tokenize("snapshot \"D:\\ps5\\a b\\shot.bmp\" after=2", tokens, error));
    CHECK(tokens.size() == 3 && tokens[1] == "D:\\ps5\\a b\\shot.bmp");
    CHECK(Automation::Tokenize("mark \"say \\\"hi\\\"\"", tokens, error) && tokens[1] == "say \"hi\"");
    CHECK(!Automation::Tokenize("snapshot \"open", tokens, error) && error == "unterminated quote");
    CHECK(Automation::Tokenize("   ", tokens, error) && tokens.empty());

    const auto arguments = Automation::Arguments::From({"press", "cross", "300", "Polls=3", "nowait"}, 1);
    CHECK(arguments.positional.size() == 3 && arguments.positional[1] == "300");
    CHECK(arguments.Option("polls") != nullptr && *arguments.Option("polls") == "3");
    CHECK(arguments.Has("nowait") && !arguments.Has("hold"));
    CHECK(Automation::ParseInteger("42") == 42 && !Automation::ParseInteger("4x") && !Automation::ParseInteger(""));
}

void Buttons() {
    std::uint32_t mask = 0;
    CHECK(Automation::ParseButtons("cross", mask) && mask == Cross);
    CHECK(Automation::ParseButtons("L1+Cross", mask) && mask == (L1 | Cross));
    CHECK(Automation::ParseButtons("options", mask) && mask == 0x0008);
    CHECK(Automation::ParseButtons("touchpad", mask) && mask == 0x100000);
    CHECK(Automation::ParseButtons("up+down+left+right", mask) && mask == 0xF0);
    CHECK(Automation::ParseButtons("l2+r2+l3+r3", mask) && mask == 0x306);
    CHECK(Automation::ParseButtons("triangle+circle+square", mask) && mask == 0xB000);
    CHECK(!Automation::ParseButtons("bogus", mask));
    CHECK(!Automation::ParseButtons("cross+", mask));
    CHECK(!Automation::ParseButtons("", mask));
    CHECK(Automation::ButtonsName(L1 | Cross) == "l1+cross");
}

// At ~1 FPS (one read per second) a 300 ms press with polls=3 stays down for three reads.
void LowRatePressSpansPolls() {
    Automation::PadEngine engine;
    const auto id = engine.Add(Press(Cross, 300, 3), 0);
    CHECK(engine.Find(id) != nullptr && engine.Find(id)->state == Automation::InputState::Active);
    const auto states = PollUntilReleased(engine, Cross, 1000, 1000);
    CHECK((states == std::vector<bool>{true, true, true, false}));
    const auto finished = engine.TakeFinished();
    CHECK(finished.size() == 1 && finished[0].id == id && finished[0].polls == 3);
    CHECK(finished[0].state == Automation::InputState::Done);
    CHECK(finished[0].endUs == 4000 * Ms);
}

// At a high read rate the wall-clock duration decides: released at the first read at >= 300 ms.
void HighRatePressHonoursDuration() {
    Automation::PadEngine engine;
    engine.Add(Press(Cross, 300, 3), 0);
    const auto states = PollUntilReleased(engine, Cross, 5, 10);
    // Reads at 5, 15, ..., 295 ms are down (30 reads); the read at 305 ms is up.
    CHECK(states.size() == 31);
    CHECK(states.back() == false);
    CHECK(std::count(states.begin(), states.end(), true) == 30);
    const auto finished = engine.TakeFinished();
    CHECK(finished.size() == 1 && finished[0].polls == 30 && finished[0].endUs - finished[0].startUs >= 300 * Ms);
}

// polls=0 and 0 ms: released at the next read (that read reports it up), so a zero-length press
// is never seen; polls=1 guarantees one read.
void MinimumPolls() {
    Automation::PadEngine engine;
    engine.Add(Press(Cross, 0, 0), 0);
    CHECK((engine.Poll(1 * Ms).buttons & Cross) == 0);
    engine.Add(Press(Circle, 0, 1), 2 * Ms);
    CHECK((engine.Poll(3 * Ms).buttons & Circle) != 0);
    CHECK((engine.Poll(4 * Ms).buttons & Circle) == 0);
}

// Back-to-back presses of one button are separated by at least one read that reports it up.
void RepeatedPressesNeverMerge() {
    Automation::PadEngine engine;
    engine.Add(Press(Cross, 0, 2), 0);
    engine.Add(Press(Cross, 0, 2), 0);
    std::vector<bool> states;
    for (int i = 1; i <= 7; ++i) states.push_back((engine.Poll(static_cast<std::uint64_t>(i) * Ms).buttons & Cross) != 0);
    CHECK((states == std::vector<bool>{true, true, false, true, true, false, false}));
    CHECK(engine.TakeFinished().size() == 2);
}

// A press queued right after another finished between reads still waits for one up read.
void PressAfterCancelWaitsOneRead() {
    Automation::PadEngine engine;
    const auto first = engine.Add(Press(Cross, 10000, 1), 0);
    CHECK((engine.Poll(1 * Ms).buttons & Cross) != 0);
    CHECK(engine.Cancel(first, 2 * Ms));
    const auto second = engine.Add(Press(Cross, 0, 1), 2 * Ms);
    CHECK(engine.Find(second)->state == Automation::InputState::Pending);
    CHECK((engine.Poll(3 * Ms).buttons & Cross) == 0);
    CHECK((engine.Poll(4 * Ms).buttons & Cross) != 0);
    CHECK((engine.Poll(5 * Ms).buttons & Cross) == 0);
}

// Different buttons are independent; a combination press holds both.
void IndependentButtons() {
    Automation::PadEngine engine;
    engine.Add(Press(Cross, 0, 3), 0);
    engine.Add(Press(L1 | Circle, 0, 1), 0);
    auto overlay = engine.Poll(1 * Ms);
    CHECK(overlay.buttons == (Cross | L1 | Circle));
    overlay = engine.Poll(2 * Ms);
    CHECK(overlay.buttons == Cross);
    // Circle was released by the read before, so a new Circle press starts at once.
    engine.Add(Press(Circle, 0, 1), 2 * Ms);
    overlay = engine.Poll(3 * Ms);
    CHECK((overlay.buttons & Cross) != 0 && (overlay.buttons & Circle) != 0);
}

// minFrames: held until that many presents happened while it was down.
void MinimumFrames() {
    Automation::PadEngine engine;
    engine.Add(Press(Cross, 0, 1, 2), 0);
    CHECK((engine.Poll(1 * Ms).buttons & Cross) != 0);
    CHECK((engine.Poll(2 * Ms).buttons & Cross) != 0);
    engine.Present();
    CHECK((engine.Poll(3 * Ms).buttons & Cross) != 0);
    engine.Present();
    CHECK((engine.Poll(4 * Ms).buttons & Cross) == 0);
    CHECK(engine.TakeFinished()[0].frames == 2);
}

void HoldAndRelease() {
    Automation::PadEngine engine;
    Automation::InputAction hold;
    hold.buttons = Cross;
    hold.hold = true;
    const auto id = engine.Add(hold, 0);
    CHECK(engine.Add(hold, 0) == id);
    for (int i = 1; i <= 5; ++i) CHECK((engine.Poll(static_cast<std::uint64_t>(i) * 100000 * Ms).buttons & Cross) != 0);
    // A press of a held button waits for the release.
    const auto press = engine.Add(Press(Cross, 0, 1), 0);
    CHECK(engine.Find(press)->state == Automation::InputState::Pending);
    CHECK(engine.Release(Automation::Resource::Buttons, Cross, 0) == 2);
    CHECK((engine.Poll(600000 * Ms).buttons & Cross) == 0);
    const auto finished = engine.TakeFinished();
    CHECK(finished.size() == 2 && finished[0].state == Automation::InputState::Cancelled && finished[0].polls == 5);
}

void SticksAndTriggers() {
    Automation::PadEngine engine;
    Automation::InputAction stick;
    stick.resource = Automation::Resource::LeftStick;
    stick.x = 255;
    stick.y = 128;
    stick.hold = true;
    engine.Add(stick, 0);
    auto overlay = engine.Poll(1 * Ms);
    CHECK(overlay.leftStick && !overlay.rightStick && overlay.sticks[0] == 255 && overlay.sticks[1] == 128);
    // A new value replaces the hold at once (no release read for sticks).
    stick.x = 0;
    engine.Add(stick, 1 * Ms);
    overlay = engine.Poll(2 * Ms);
    CHECK(overlay.leftStick && overlay.sticks[0] == 0);
    CHECK(engine.TakeFinished().size() == 1);
    // Timed right stick and a trigger.
    Automation::InputAction right;
    right.resource = Automation::Resource::RightStick;
    right.x = 10;
    right.y = 20;
    right.durationUs = 0;
    right.minPolls = 1;
    engine.Add(right, 2 * Ms);
    Automation::InputAction trigger;
    trigger.resource = Automation::Resource::R2;
    trigger.value = 200;
    trigger.durationUs = 0;
    trigger.minPolls = 2;
    engine.Add(trigger, 2 * Ms);
    overlay = engine.Poll(3 * Ms);
    CHECK(overlay.rightStick && overlay.sticks[2] == 10 && overlay.sticks[3] == 20 && overlay.r2 == 200 && overlay.l2 == 0);
    overlay = engine.Poll(4 * Ms);
    CHECK(!overlay.rightStick && overlay.r2 == 200 && overlay.leftStick);
    overlay = engine.Poll(5 * Ms);
    CHECK(overlay.r2 == 0 && overlay.leftStick);
    CHECK(engine.ReleaseAll(6 * Ms) == 1);
    overlay = engine.Poll(7 * Ms);
    CHECK(!overlay.leftStick && overlay.buttons == 0);
}

void PollCounter() {
    Automation::PadEngine engine;
    for (int i = 0; i < 17; ++i) engine.Poll(static_cast<std::uint64_t>(i));
    CHECK(engine.Polls() == 17);
    const auto id = engine.Add(Press(Cross, 0, 2), 100);
    engine.Poll(200);
    engine.Poll(300);
    engine.Poll(400);
    const auto finished = engine.TakeFinished();
    CHECK(finished.size() == 1 && finished[0].id == id && finished[0].firstPoll == 18 && finished[0].lastPoll == 19);
}

void JsonFields() {
    std::string text;
    Automation::AppendJsonString(text, "a\"b\\c\n\x01");
    CHECK(text == "\"a\\\"b\\\\c\\n\\u0001\"");
}

}

int main() {
    Tokens();
    Buttons();
    LowRatePressSpansPolls();
    HighRatePressHonoursDuration();
    MinimumPolls();
    RepeatedPressesNeverMerge();
    PressAfterCancelWaitsOneRead();
    IndependentButtons();
    MinimumFrames();
    HoldAndRelease();
    SticksAndTriggers();
    PollCounter();
    JsonFields();
    if (failures != 0) {
        std::fprintf(stderr, "%d automation engine check(s) failed\n", failures);
        return 1;
    }
    std::printf("automation engine: all checks passed\n");
    return 0;
}
