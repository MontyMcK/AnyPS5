// The automation channel end to end: a client on the named pipe presses buttons that the guest's
// pad reads (Pad::ReadState) report for the requested number of reads, takes a snapshot through a
// fake presenter, and reads the event ring and the event log.
#include "prx/libScePad/include/PadState.hpp"
#include "prx/libkernel/Automation/include/Automation.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include <windows.h>

namespace {

int failures = 0;

void Check(bool condition, const char* what, int line) {
    if (condition) return;
    std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
    ++failures;
}
#define CHECK(condition) Check((condition), #condition, __LINE__)

class Client {
public:
    explicit Client(const std::string& name) {
        for (int attempt = 0; attempt < 100 && handle == INVALID_HANDLE_VALUE; ++attempt) {
            handle = CreateFileA(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
            if (handle == INVALID_HANDLE_VALUE) std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    ~Client() {
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    }
    bool Connected() const { return handle != INVALID_HANDLE_VALUE; }
    std::string Send(const std::string& command) {
        const std::string line = command + "\n";
        DWORD wrote = 0;
        if (!WriteFile(handle, line.data(), static_cast<DWORD>(line.size()), &wrote, nullptr)) return "<write failed>";
        std::string reply;
        char c = 0;
        DWORD got = 0;
        while (ReadFile(handle, &c, 1, &got, nullptr) && got == 1 && c != '\n') reply.push_back(c);
        return reply;
    }
private:
    HANDLE handle = INVALID_HANDLE_VALUE;
};

bool Contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// 64x36 gradient BMP (the fake presenter's "frame"); the left half turns white while Cross is held.
bool WriteFakeFrame(const char* path, bool crossHeld) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) return false;
    const std::uint32_t width = 64, height = 36, stride = width * 3, bytes = stride * height;
    std::uint8_t header[54] = {'B', 'M'};
    const auto put32 = [&](int offset, std::uint32_t value) { for (int i = 0; i < 4; ++i) header[offset + i] = static_cast<std::uint8_t>(value >> (8 * i)); };
    put32(2, 54 + bytes); put32(10, 54); put32(14, 40); put32(18, width); put32(22, height);
    header[26] = 1; header[28] = 24; put32(34, bytes);
    std::fwrite(header, 1, sizeof(header), file);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint8_t level = crossHeld && x < width / 2 ? 255 : static_cast<std::uint8_t>(x * 4);
            const std::uint8_t pixel[3] = {level, level, level};
            std::fwrite(pixel, 1, 3, file);
        }
    }
    return std::fclose(file) == 0;
}

// `automation_channel_tests serve <pipe> <seconds>`: a fake game for client tests. The guest reads the
// pad every 50 ms (each new Cross press emits a "vrr" peg event); the presenter flips every 40 ms and
// answers snapshots with a 64x36 BMP.
int Serve(const char* pipe, int seconds) {
    _putenv((std::string("APS5_AUTOMATION=") + pipe).c_str());
    Pad::Initialize();
    if (!Aps5AutomationEnabled_nid_postfix()) return 2;
    std::atomic<bool> stop{false};
    std::atomic<bool> crossHeld{false};
    std::thread guest([&] {
        bool previous = false;
        while (!stop) {
            const bool cross = (Pad::ReadState().buttons & 0x4000) != 0;
            crossHeld = cross;
            if (cross && !previous) Automation::Fields().Str("op", "peg").Num("handle", 1).Emit("vrr");
            previous = cross;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    std::thread presenter([&] {
        while (!stop) {
            AutomationSnapshotRequest request;
            if (Aps5AutomationPresented_nid_postfix(64, 36, &request)) {
                const bool ok = WriteFakeFrame(request.path, crossHeld);
                Aps5AutomationSnapshotDone_nid_postfix(request.id, ok, ok ? "\"width\":64,\"height\":36,\"method\":\"fake\",\"black\":false" : "\"error\":\"cannot write\"");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    });
    std::printf("serving %s for %d s\n", pipe, seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    stop = true;
    guest.join();
    presenter.join();
    std::_Exit(0);
}

long long Field(const std::string& reply, const std::string& key) {
    const auto at = reply.find("\"" + key + "\":");
    if (at == std::string::npos) return -1;
    return std::atoll(reply.c_str() + at + key.size() + 3);
}

}

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "serve") return Serve(argv[2], std::atoi(argv[3]));
    const auto pid = std::to_string(GetCurrentProcessId());
    const std::string pipeName = "aps5-automation-test-" + pid;
    char temp[MAX_PATH] = {};
    GetTempPathA(sizeof(temp), temp);
    const std::string eventsPath = std::string(temp) + "aps5-automation-test-" + pid + ".jsonl";
    _putenv(("APS5_AUTOMATION=" + pipeName).c_str());
    _putenv(("APS5_AUTOMATION_EVENTS=" + eventsPath).c_str());

    Pad::Initialize();
    CHECK(Aps5AutomationEnabled_nid_postfix());

    // The guest: reads the pad every 40 ms and counts the reads that report Cross.
    std::atomic<bool> stop{false};
    std::atomic<int> crossReads{0};
    std::atomic<int> leftStickReads{0};
    std::thread guest([&] {
        while (!stop) {
            const PadData data = Pad::ReadState();
            if ((data.buttons & 0x4000) != 0) ++crossReads;
            if (data.left_stick_x == 0 && data.left_stick_y == 255) ++leftStickReads;
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
    });
    // The presenter: one flip every 30 ms; snapshots are "captured" as a 1x1 image.
    std::thread presenter([&] {
        while (!stop) {
            AutomationSnapshotRequest request;
            if (Aps5AutomationPresented_nid_postfix(64, 32, &request)) {
                std::string fields = "\"width\":1,\"height\":1,\"method\":\"fake\",\"path_seen\":";
                Automation::AppendJsonString(fields, request.path);
                Aps5AutomationSnapshotDone_nid_postfix(request.id, true, fields.c_str());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }
    });

    Automation::Fields().Str("path", "/app0/test.bin").Num("fd", 3).Emit("open");

    {
        Client client("\\\\.\\pipe\\" + pipeName);
        CHECK(client.Connected());
        auto reply = client.Send("ping");
        CHECK(Contains(reply, "\"ok\":true") && Field(reply, "pid") == std::atoll(pid.c_str()));

        const auto before = crossReads.load();
        reply = client.Send("press cross 120 polls=3");
        CHECK(Contains(reply, "\"ok\":true") && Contains(reply, "\"state\":\"done\""));
        CHECK(Field(reply, "polls") >= 3);
        CHECK(crossReads.load() - before >= 3);
        CHECK(Contains(reply, "\"held_ms\":"));

        reply = client.Send("press bogus");
        CHECK(Contains(reply, "\"ok\":false") && Contains(reply, "unknown button"));
        reply = client.Send("frobnicate");
        CHECK(Contains(reply, "\"ok\":false"));

        reply = client.Send("stick left 0 255 50 polls=2");
        CHECK(Contains(reply, "\"ok\":true") && Field(reply, "polls") >= 2);
        CHECK(leftStickReads.load() >= 2);

        reply = client.Send("hold circle polls=1");
        CHECK(Contains(reply, "\"ok\":true") && Contains(reply, "\"state\":\"active\""));
        reply = client.Send("release circle");
        CHECK(Contains(reply, "\"released\":1"));

        reply = client.Send("snapshot \"C:\\shots\\frame one.bmp\" after=2");
        CHECK(Contains(reply, "\"ok\":true") && Contains(reply, "\"method\":\"fake\"") && Contains(reply, "frame one.bmp"));

        reply = client.Send("wait polls=2 presents=2 timeout=5000");
        CHECK(Contains(reply, "\"ok\":true"));

        reply = client.Send("mark checkpoint A");
        CHECK(Contains(reply, "\"ok\":true"));

        reply = client.Send("events since=0 max=50000");
        CHECK(Contains(reply, "\"type\":\"automation\""));
        CHECK(Contains(reply, "\"type\":\"open\"") && Contains(reply, "/app0/test.bin"));
        CHECK(Contains(reply, "\"type\":\"input\"") && Contains(reply, "\"buttons\":\"cross\""));
        CHECK(Contains(reply, "\"type\":\"snapshot\""));
        CHECK(Contains(reply, "\"text\":\"checkpoint A\""));
        const auto next = Field(reply, "next");
        CHECK(next > 0);

        // Long poll: nothing new for a moment, then the mark from a second client arrives.
        std::thread marker([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            Client other("\\\\.\\pipe\\" + pipeName);
            other.Send("mark late");
        });
        reply = client.Send("events since=" + std::to_string(next) + " timeout=5000");
        marker.join();
        CHECK(Contains(reply, "\"ok\":true") && Field(reply, "next") > next);

        reply = client.Send("status");
        CHECK(Contains(reply, "\"ok\":true") && Field(reply, "presents") > 0 && Field(reply, "polls") > 0 && Contains(reply, "\"inputs\":[]"));

        reply = client.Send("press cross 10 polls=100000 timeout=300");
        CHECK(Contains(reply, "\"ok\":false") && Contains(reply, "timeout"));
    }

    stop = true;
    guest.join();
    presenter.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    std::ifstream log(eventsPath);
    std::stringstream text;
    text << log.rdbuf();
    CHECK(Contains(text.str(), "\"type\":\"automation\""));
    CHECK(Contains(text.str(), "\"type\":\"input\""));

    if (failures != 0) {
        std::fprintf(stderr, "%d automation channel check(s) failed\n", failures);
        return 1;
    }
    std::printf("automation channel: all checks passed\n");
    std::_Exit(0);
}
