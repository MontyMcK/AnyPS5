#ifndef CORE_LIBS_PRX_LIBKERNEL_AUTOMATION_AUTOMATION_HPP
#define CORE_LIBS_PRX_LIBKERNEL_AUTOMATION_AUTOMATION_HPP

// Script automation (APS5_AUTOMATION=1 or a pipe name): a command channel on a named pipe that
// drives the emulated pad and takes snapshots, and a JSON-lines event log of what the title does
// (file opens, module loads, dialogs, VRR peg, save data, presents per second). The server lives in
// libkernel; other libraries report events and poll pad overrides through these exports. Off by
// default: every hook first checks Automation::Enabled(), a cached flag. Client and protocol:
// profiling/aps5auto.py.

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

// Pad state the automation channel imposes on the next guest pad read (merged over host input).
struct AutomationPadOverlay {
    std::uint32_t buttons = 0;
    std::uint8_t sticks[4] = {128, 128, 128, 128};
    bool leftStick = false;
    bool rightStick = false;
    std::uint8_t l2 = 0;
    std::uint8_t r2 = 0;
};

// A snapshot the presenter must take of the frame it just presented.
struct AutomationSnapshotRequest {
    std::uint64_t id = 0;
    char path[1024] = {};
};

extern "C" {
// Whether APS5_AUTOMATION is set; the first call starts the channel.
bool Aps5AutomationEnabled_nid_postfix();
// Appends {"type": type, <fields>} to the event log; `fields` is a JSON member list or null.
void Aps5AutomationEvent_nid_postfix(const char* type, const char* fields);
// One guest pad read: advances timed inputs and returns the overrides for this read.
void Aps5AutomationPadPoll_nid_postfix(AutomationPadOverlay* overlay);
// One presented flip; true when a snapshot of it is due (the presenter then captures it and
// reports with Aps5AutomationSnapshotDone).
bool Aps5AutomationPresented_nid_postfix(std::uint32_t width, std::uint32_t height, AutomationSnapshotRequest* request);
void Aps5AutomationSnapshotDone_nid_postfix(std::uint64_t id, bool ok, const char* fields);
}

namespace Automation {

inline bool Enabled() {
    static const bool enabled = Aps5AutomationEnabled_nid_postfix();
    return enabled;
}

inline void AppendJsonString(std::string& out, std::string_view text) {
    out.push_back('"');
    for (const char c : text) {
        const auto u = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == '\t') {
            out += "\\t";
        } else if (u < 0x20 || u == 0x7f) {
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", u);
            out += escaped;
        } else {
            out.push_back(c);
        }
    }
    out.push_back('"');
}

// Builds the member list of an event: Fields().Str("path", p).Num("fd", fd).Emit("open").
class Fields {
public:
    Fields& Str(const char* key, std::string_view value) {
        Key(key);
        AppendJsonString(text, value);
        return *this;
    }
    Fields& Num(const char* key, long long value) {
        Key(key);
        text += std::to_string(value);
        return *this;
    }
    Fields& Unsigned(const char* key, unsigned long long value) {
        Key(key);
        text += std::to_string(value);
        return *this;
    }
    Fields& Real(const char* key, double value) {
        Key(key);
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.3f", value);
        text += buffer;
        return *this;
    }
    Fields& Bool(const char* key, bool value) {
        Key(key);
        text += value ? "true" : "false";
        return *this;
    }
    Fields& Hex(const char* key, unsigned long long value) {
        char buffer[24];
        std::snprintf(buffer, sizeof(buffer), "0x%llx", value);
        return Str(key, buffer);
    }
    const std::string& Text() const { return text; }
    void Emit(const char* type) const { Aps5AutomationEvent_nid_postfix(type, text.c_str()); }

private:
    void Key(const char* key) {
        if (!text.empty()) text.push_back(',');
        AppendJsonString(text, key);
        text.push_back(':');
    }
    std::string text;
};

}

#endif
