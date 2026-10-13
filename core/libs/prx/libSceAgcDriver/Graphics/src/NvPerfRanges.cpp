#define VK_NO_PROTOTYPES
#include "prx/libSceAgcDriver/Graphics/include/NvPerfRanges.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(APS5_HAVE_NVPERF)
#include "nvperf_host_impl.h"
#include "NvPerfReportGeneratorVulkan.h"
#endif

namespace AgcDriver::Graphics::NvPerf {

bool Enabled() {
#if defined(APS5_HAVE_NVPERF)
    static const bool enabled = std::getenv("APS5_NVPERF") != nullptr;
    return enabled;
#else
    return false;
#endif
}

#if defined(APS5_HAVE_NVPERF)

namespace {

struct Profiler {
    nv::perf::profiler::ReportGeneratorVulkan generator;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    std::chrono::steady_clock::time_point start;
    bool requested = false;
    bool reported = false;
    std::uint64_t frames = 0;
};

Profiler*& profiler() {
    static Profiler* instance = nullptr;
    return instance;
}

std::atomic<bool> ranging{false};

bool hostReady() {
    static const bool ready = [] {
        nv::perf::UserLogEnableStderr(true);
        const bool initialized = nv::perf::InitializeNvPerf();
        std::fprintf(stderr, "[nvperf] host %s\n", initialized ? "initialized" : "failed to initialize (is nvperf_grfx_host.dll next to app.exe?)");
        return initialized;
    }();
    return ready;
}

double delaySeconds() {
    const char* value = std::getenv("APS5_NVPERF_DELAY");
    return value != nullptr ? std::atof(value) : 150.0;
}

}

void AppendInstanceExtensions(std::vector<const char*>& names, std::uint32_t apiVersion) {
    if (!Enabled() || !hostReady()) return;
    const auto before = names.size();
    if (!nv::perf::VulkanAppendInstanceRequiredExtensions(names, apiVersion)) std::fprintf(stderr, "[nvperf] instance extensions unavailable\n");
    for (auto i = before; i < names.size(); ++i) std::fprintf(stderr, "[nvperf] instance extension %s\n", names[i]);
}

void AppendDeviceExtensions(VkInstance instance, VkPhysicalDevice physical, PFN_vkGetInstanceProcAddr instanceProc, std::vector<const char*>& names) {
    if (!Enabled() || !hostReady()) return;
    const auto before = names.size();
    if (!nv::perf::VulkanAppendDeviceRequiredExtensions(instance, physical, reinterpret_cast<void*>(instanceProc), names)) std::fprintf(stderr, "[nvperf] device extensions unavailable\n");
    for (auto i = before; i < names.size(); ++i) std::fprintf(stderr, "[nvperf] device extension %s\n", names[i]);
}

void Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device, VkQueue queue, std::uint32_t family, PFN_vkGetInstanceProcAddr instanceProc, PFN_vkGetDeviceProcAddr deviceProc) {
    if (!Enabled() || !hostReady()) return;
    ranging = false;
    if (profiler() != nullptr) {
        // A replaced device: the report follows the newest one.
        profiler()->generator.Reset();
        delete profiler();
        profiler() = nullptr;
    }
    auto* created = new Profiler;
    created->queue = queue;
    created->family = family;
    created->instanceProc = instanceProc;
    created->deviceProc = deviceProc;
    created->start = std::chrono::steady_clock::now();
    if (!created->generator.InitializeReportGenerator(instance, physical, device, instanceProc, deviceProc)) {
        std::fprintf(stderr, "[nvperf] report generator failed to initialize\n");
        delete created;
        return;
    }
    const char* directory = std::getenv("APS5_NVPERF_DIR");
    created->generator.outputOptions.directoryName = directory != nullptr ? directory : "nvperf";
    created->generator.SetFrameLevelRangeName("frame");
    created->generator.SetNumNestingLevels(2);
    created->generator.SetMaxNumRanges(4096);
    profiler() = created;
    ranging = true;
    std::fprintf(stderr, "[nvperf] report generator ready; collection starts %.0f s from now\n", delaySeconds());
}

void FrameBoundary() {
    auto* active = profiler();
    if (active == nullptr) return;
    ++active->frames;
    if (active->frames > 1) active->generator.OnFrameEnd();
    if (!active->requested && std::chrono::steady_clock::now() - active->start >= std::chrono::duration<double>(delaySeconds())) {
        active->requested = true;
        std::fprintf(stderr, "[nvperf] collecting a report from frame %llu\n", static_cast<unsigned long long>(active->frames));
        active->generator.StartCollectionOnNextFrame();
    }
    if (active->requested && !active->reported && !active->generator.IsCollectingReport() && active->frames > 2) {
        active->reported = true;
        std::fprintf(stderr, "[nvperf] report written to %s\n", active->generator.GetLastReportDirectoryName().c_str());
    }
    active->generator.OnFrameStart(active->queue, active->family, active->instanceProc, active->deviceProc);
}

bool Ranging() {
    return ranging.load(std::memory_order_relaxed);
}

void PushRange(VkCommandBuffer commands, const char* name) {
    if (auto* active = profiler()) active->generator.rangeCommands.PushRange(commands, name);
}

void PopRange(VkCommandBuffer commands) {
    if (auto* active = profiler()) active->generator.rangeCommands.PopRange(commands);
}

#else

void AppendInstanceExtensions(std::vector<const char*>&, std::uint32_t) {}
void AppendDeviceExtensions(VkInstance, VkPhysicalDevice, PFN_vkGetInstanceProcAddr, std::vector<const char*>&) {}
void Initialize(VkInstance, VkPhysicalDevice, VkDevice, VkQueue, std::uint32_t, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr) {}
void FrameBoundary() {}
bool Ranging() { return false; }
void PushRange(VkCommandBuffer, const char*) {}
void PopRange(VkCommandBuffer) {}

#endif

}
