// APS5_SAMPLE=<delay s>,<duration s>[,<period ms>]: an in-process sampling profiler for host threads.
// After the delay, every thread that used CPU since the last tick is suspended, its context read and
// its stack walked with the Windows unwinder (RtlVirtualUnwind over each module's .pdata), then
// resumed. Guest code has no unwind data, so a walk stops at the first guest frame. At the end the
// distinct stacks are written with their counts to APS5_SAMPLE_OUT (default agc-stacks.txt):
//   <tid>\t<thread name>\t<count>\t<module>+<offset> <module>+<offset> ...   (leaf first)
// Nothing is allocated while a thread is suspended.
#include <windows.h>
#include <tlhelp32.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr std::size_t MaxFrames = 32;

struct Sample {
    DWORD tid;
    std::uint32_t frames;
    std::uint64_t pc[MaxFrames];
};

struct Tracked {
    HANDLE handle = nullptr;
    ULONG64 cycles = 0;
    bool alive = true;
};

// This module's own .pdata, searched directly: the emulator maps it outside the system's function tables.
std::uint8_t* moduleBase;
std::uint64_t moduleEnd;
const RUNTIME_FUNCTION* moduleFunctions;
std::size_t moduleFunctionCount;

PRUNTIME_FUNCTION LookupFunction(DWORD64 pc, DWORD64* imageBase) {
    const auto base = reinterpret_cast<std::uint64_t>(moduleBase);
    if (moduleFunctions == nullptr || pc < base || pc >= moduleEnd) return RtlLookupFunctionEntry(pc, imageBase, nullptr);
    const auto rva = static_cast<DWORD>(pc - base);
    std::size_t low = 0, high = moduleFunctionCount;
    while (low < high) {
        const auto middle = (low + high) / 2;
        if (moduleFunctions[middle].EndAddress <= rva) low = middle + 1; else high = middle;
    }
    if (low == moduleFunctionCount || moduleFunctions[low].BeginAddress > rva) return nullptr;
    *imageBase = base;
    return const_cast<PRUNTIME_FUNCTION>(&moduleFunctions[low]);
}

std::uint32_t Walk(HANDLE thread, std::uint64_t* pc) {
    CONTEXT context{};
    context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (!GetThreadContext(thread, &context)) return 0;
    std::uint32_t count = 0;
    pc[count++] = context.Rip;
    while (count < MaxFrames) {
        DWORD64 imageBase = 0;
        auto* function = LookupFunction(context.Rip, &imageBase);
        if (function == nullptr) {
            // A leaf without unwind data: take [rsp] only when it looks like a return into code that has some.
            if (count != 1) break;
            const auto candidate = *reinterpret_cast<const DWORD64*>(context.Rsp);
            DWORD64 candidateBase = 0;
            if (LookupFunction(candidate, &candidateBase) == nullptr) break;
            context.Rip = candidate;
            context.Rsp += 8;
        } else {
            PVOID handlerData = nullptr;
            DWORD64 establisher = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, imageBase, context.Rip, function, &context, &handlerData, &establisher, nullptr);
        }
        if (context.Rip == 0) break;
        pc[count++] = context.Rip;
    }
    return count;
}

std::string ThreadName(HANDLE thread) {
    using GetDescription = HRESULT(WINAPI*)(HANDLE, PWSTR*);
    static const auto get = reinterpret_cast<GetDescription>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription"));
    if (get == nullptr) return "";
    PWSTR wide = nullptr;
    if (FAILED(get(thread, &wide)) || wide == nullptr) return "";
    char text[256]{};
    WideCharToMultiByte(CP_UTF8, 0, wide, -1, text, sizeof(text) - 1, nullptr, nullptr);
    LocalFree(wide);
    for (auto* c = text; *c != '\0'; ++c) {
        if (*c == '\t' || *c == ' ') *c = '_';
    }
    return text;
}

std::string Symbolize(std::uint64_t address) {
    HMODULE module = nullptr;
    char text[320];
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(address), &module) && module != nullptr) {
        wchar_t path[MAX_PATH]{};
        GetModuleFileNameW(module, path, MAX_PATH);
        const wchar_t* base = path;
        for (const wchar_t* c = path; *c != 0; ++c) {
            if (*c == L'\\' || *c == L'/') base = c + 1;
        }
        char name[MAX_PATH]{};
        WideCharToMultiByte(CP_UTF8, 0, base, -1, name, sizeof(name) - 1, nullptr, nullptr);
        std::snprintf(text, sizeof(text), "%s+%llx", name, static_cast<unsigned long long>(address - reinterpret_cast<std::uint64_t>(module)));
    } else {
        std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(address));
    }
    return text;
}

DWORD WINAPI SamplerMain(void*) {
    const char* spec = std::getenv("APS5_SAMPLE");
    double delay = 0, duration = 10;
    unsigned period = 1;
    std::sscanf(spec, "%lf,%lf,%u", &delay, &duration, &period);
    if (period == 0) period = 1;
    const char* outEnv = std::getenv("APS5_SAMPLE_OUT");
    const std::string outPath = outEnv != nullptr && *outEnv != '\0' ? outEnv : "agc-stacks.txt";
    {
        HMODULE module = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&SamplerMain), &module) && module != nullptr) {
            moduleBase = reinterpret_cast<std::uint8_t*>(module);
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(moduleBase + reinterpret_cast<const IMAGE_DOS_HEADER*>(moduleBase)->e_lfanew);
            const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
            moduleEnd = reinterpret_cast<std::uint64_t>(moduleBase) + nt->OptionalHeader.SizeOfImage;
            moduleFunctions = reinterpret_cast<const RUNTIME_FUNCTION*>(moduleBase + directory.VirtualAddress);
            moduleFunctionCount = directory.Size / sizeof(RUNTIME_FUNCTION);
        }
    }
    Sleep(static_cast<DWORD>(delay * 1000));

    const DWORD self = GetCurrentThreadId();
    const DWORD process = GetCurrentProcessId();
    std::unordered_map<DWORD, Tracked> threads;
    std::unordered_map<DWORD, std::string> names;
    std::vector<Sample> samples;
    samples.reserve(1u << 20);
    ULONGLONG start = GetTickCount64(), lastScan = 0;
    std::uint64_t ticks = 0, suspended = 0;
    std::fprintf(stderr, "[sampler] sampling for %.1f s every %u ms -> %s\n", duration, period, outPath.c_str());
    while (GetTickCount64() - start < static_cast<ULONGLONG>(duration * 1000)) {
        if (GetTickCount64() - lastScan > 500) {
            lastScan = GetTickCount64();
            for (auto& [tid, tracked] : threads) tracked.alive = false;
            HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            THREADENTRY32 entry{sizeof(entry)};
            for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
                if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
                auto& tracked = threads[entry.th32ThreadID];
                tracked.alive = true;
                if (tracked.handle == nullptr) {
                    tracked.handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
                    if (tracked.handle != nullptr) names[entry.th32ThreadID] = ThreadName(tracked.handle);
                }
            }
            CloseHandle(snapshot);
            for (auto it = threads.begin(); it != threads.end();) {
                if (!it->second.alive) {
                    if (it->second.handle != nullptr) CloseHandle(it->second.handle);
                    it = threads.erase(it);
                } else {
                    ++it;
                }
            }
        }
        ++ticks;
        for (auto& [tid, tracked] : threads) {
            if (tracked.handle == nullptr) continue;
            ULONG64 cycles = 0;
            if (!QueryThreadCycleTime(tracked.handle, &cycles) || cycles == tracked.cycles) continue;
            tracked.cycles = cycles;
            Sample sample;
            sample.tid = tid;
            if (SuspendThread(tracked.handle) == static_cast<DWORD>(-1)) continue;
            sample.frames = Walk(tracked.handle, sample.pc);
            ResumeThread(tracked.handle);
            ++suspended;
            if (sample.frames != 0) samples.push_back(sample);
        }
        Sleep(period);
    }

    std::map<std::pair<DWORD, std::vector<std::uint64_t>>, std::uint64_t> stacks;
    for (const auto& sample : samples) ++stacks[{sample.tid, std::vector<std::uint64_t>(sample.pc, sample.pc + sample.frames)}];
    std::unordered_map<std::uint64_t, std::string> symbols;
    if (auto* file = std::fopen(outPath.c_str(), "w")) {
        for (const auto& [key, count] : stacks) {
            std::fprintf(file, "%lu\t%s\t%llu\t", key.first, names[key.first].c_str(), static_cast<unsigned long long>(count));
            for (const auto pc : key.second) {
                auto [it, inserted] = symbols.try_emplace(pc);
                if (inserted) it->second = Symbolize(pc);
                std::fprintf(file, "%s ", it->second.c_str());
            }
            std::fputc('\n', file);
        }
        std::fclose(file);
    }
    std::fprintf(stderr, "[sampler] %llu ticks, %llu thread samples, %zu stacks written to %s\n", static_cast<unsigned long long>(ticks), static_cast<unsigned long long>(suspended), stacks.size(), outPath.c_str());
    for (auto& [tid, tracked] : threads) {
        if (tracked.handle != nullptr) CloseHandle(tracked.handle);
    }
    return 0;
}

[[maybe_unused]] const bool started = [] {
    if (std::getenv("APS5_SAMPLE") == nullptr) return false;
    if (HANDLE thread = CreateThread(nullptr, 0, SamplerMain, nullptr, 0, nullptr)) CloseHandle(thread);
    return true;
}();

}
