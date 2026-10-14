#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <algorithm>
#include <chrono>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <array>
#include <optional>
#include <cstdlib>
#include <stdexcept>

namespace AgcDriver::DriverDetail {

namespace {

// APS5_DISPATCH_KEY_PROFILE=<queue hex> (default 22): why the dispatch cache misses on that queue.
// Every 10 s: dispatches, distinct keys, how many keys were already seen in the previous window (an
// unbounded cache would hit those), and for the busiest programs which user data words change
// between dispatches and how many distinct values each takes.
void profileDispatchKeys(std::uint64_t address, const std::vector<std::uint32_t>& userData, std::uint64_t key) {
    static const std::uint32_t wanted = [] {
        const char* text = std::getenv("APS5_DISPATCH_KEY_PROFILE");
        if (text == nullptr) return 0xffffffffu;
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(text, nullptr, 16));
        return parsed == 0 || parsed == 1 ? 0x22u : parsed;
    }();
    if (wanted == 0xffffffffu || GuestMemory::GpuLockThreadTag() != wanted) return;
    struct Word { std::uint32_t last = 0; std::uint64_t changes = 0; std::unordered_set<std::uint32_t> values; };
    struct Program { std::uint64_t dispatches = 0; std::unordered_set<std::uint64_t> keys; std::vector<Word> words; };
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, Program> programs;
    static std::unordered_set<std::uint64_t> window, previous;
    static std::uint64_t dispatches = 0, seenBefore = 0, seenThisWindow = 0;
    static auto last = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex);
    ++dispatches;
    if (previous.contains(key)) ++seenBefore;
    else if (window.contains(key)) ++seenThisWindow;
    window.insert(key);
    auto& program = programs[address];
    ++program.dispatches;
    if (program.keys.size() < 65536) program.keys.insert(key);
    if (program.words.size() < userData.size()) program.words.resize(userData.size());
    for (std::size_t i = 0; i < userData.size(); ++i) {
        auto& word = program.words[i];
        if (program.dispatches > 1 && word.last != userData[i]) ++word.changes;
        word.last = userData[i];
        if (word.values.size() < 4096) word.values.insert(userData[i]);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    std::fprintf(stderr, "[key-churn] queue 0x%x (10 s): %llu dispatches, %zu distinct keys; keys seen in the previous window %llu (%.0f%%), first seen earlier this window %llu (%.0f%%), %zu programs\n", wanted, static_cast<unsigned long long>(dispatches), window.size(), static_cast<unsigned long long>(seenBefore), dispatches != 0 ? 100.0 * static_cast<double>(seenBefore) / static_cast<double>(dispatches) : 0.0, static_cast<unsigned long long>(seenThisWindow), dispatches != 0 ? 100.0 * static_cast<double>(seenThisWindow) / static_cast<double>(dispatches) : 0.0, programs.size());
    std::vector<std::pair<std::uint64_t, const Program*>> order;
    for (const auto& [programAddress, entry] : programs) order.emplace_back(programAddress, &entry);
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second->dispatches > b.second->dispatches; });
    for (std::size_t rank = 0; rank < order.size() && rank < 8; ++rank) {
        const auto& entry = *order[rank].second;
        std::string line;
        char part[96];
        for (std::size_t i = 0; i < entry.words.size(); ++i) {
            if (entry.words[i].values.size() <= 1) continue;
            std::snprintf(part, sizeof(part), " u%zu:%llu/%zu", i, static_cast<unsigned long long>(entry.words[i].changes), entry.words[i].values.size());
            line += part;
        }
        std::fprintf(stderr, "[key-churn]   program 0x%llx: %llu dispatches, %zu keys, %zu user words; changing words (changes/distinct values):%s\n", static_cast<unsigned long long>(order[rank].first), static_cast<unsigned long long>(entry.dispatches), entry.keys.size(), entry.words.size(), line.c_str());
    }
    previous = std::move(window);
    window.clear();
    programs.clear();
    dispatches = seenBefore = seenThisWindow = 0;
}

void noteDwordFillKernel(std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, std::span<const std::uint32_t> packet, std::uint64_t indirectArguments, const ShaderRecompiler::ShaderComputeStageInfo& compute) {
    static const bool disabled = std::getenv("APS5_NO_DWORD_FILL_CLEAR") != nullptr;
    if (disabled) return;
    static constexpr std::array<std::uint32_t, 23> dwordFillKernel{0xbfa00001u, 0xd7460004u, 0x04010c09u, 0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u, 0x7da80808u, 0xbf88000du, 0x360a0883u, 0xbe84047eu, 0xbe86047eu, 0x7e100505u, 0x7da40a08u, 0xbefc0308u, 0x7e0c8700u, 0x8a867e06u, 0xbefe0406u, 0xbf85fff9u, 0xbefe0404u, 0xe0102000u, 0x80000604u, 0xbf810000u};
    if (indirectArguments != 0 || userData.size() < 9 || packet.size() < 5 || compute.numThreads[0] != 64 || compute.numThreads[1] != 1 || compute.numThreads[2] != 1) return;
    if (code.size() < dwordFillKernel.size() || !std::equal(dwordFillKernel.begin(), dwordFillKernel.end(), code.begin())) return;
    if (userData[4] != userData[5] || userData[5] != userData[6] || userData[6] != userData[7]) return;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    if ((userData[3] >> 30u) != 0 || stride != 4 || swizzled || addTid || ((packet[4] & 0x20u) != 0) || packet[2] != 1 || packet[3] != 1) return;
    const auto records = std::min<std::uint64_t>({static_cast<std::uint64_t>(packet[1]) * 64u, userData[8], userData[2]});
    const auto base = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    Graphics::TraceDepthMetadataFill(base, static_cast<std::size_t>(records * 4u), userData[4], true, "dword-fill dispatch");
    Graphics::NoteDepthMetadataFill(base, static_cast<std::size_t>(records * 4u), userData[4]);
    // Live `fill_kernel_keys` (env APS5_FILL_KERNEL_KEYS): a fill of one repeated byte that is a DCC
    // key code (the title fast-clears render-target keys with this kernel every frame) is noted
    // in the key-store memo, as a buffer fill of the keys is (BufferFill.cpp): a key read of the
    // range then answers the fill's keys instead of waiting for the GPU. The memo keeps the entry
    // only while the dispatch's own write note is the newest over exactly this range
    // (Recorder::LastWriteNote, PendingStoreKeys), so any other writer still makes the read wait.
    static const bool keysDefault = std::getenv("APS5_FILL_KERNEL_KEYS") != nullptr;
    const auto pattern = userData[4];
    if (pattern != (pattern & 0xffu) * 0x01010101u || !Graphics::LiveFlag("fill_kernel_keys", keysDefault)) return;
    Graphics::DccKeys keys = Graphics::DccKeys::Mixed;
    switch (pattern & 0xffu) {
        case 0x00: keys = Graphics::DccKeys::Clear0000; break;
        case 0x40: keys = Graphics::DccKeys::Clear0001; break;
        case 0x80: keys = Graphics::DccKeys::Clear1110; break;
        case 0xc0: keys = Graphics::DccKeys::Clear1111; break;
        case 0x20: keys = Graphics::DccKeys::ClearRegister; break;
        case 0xff: keys = Graphics::DccKeys::Uncompressed; break;
        default: return;
    }
    // The pipelined queue-0 path calls this after its device hold ended: take the mutex for the note
    // (recursive; the stage-B path still holds it).
    std::unique_lock<GuestMemory::GpuMutexType> hold(GuestMemory::GpuMutex(), std::defer_lock);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        hold.lock();
    }
    auto* recorder = Graphics::Recorder::Active();
    if (recorder == nullptr) return;
    // The dispatch notes its whole written buffer, which may extend past the filled records: the
    // range is the dispatch's only while the newest note over it is one the dispatch itself made.
    const auto bytes = static_cast<std::size_t>(records * 4u);
    const auto note = recorder->NewestWriteNote(base, bytes);
    const auto [first, last] = Graphics::Recorder::DispatchNotes();
    if (note <= first || note > last) return;
    Graphics::NoteKeysFillOnGpuAt(base, bytes, keys, note);
}

}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments) {
    const DrawPipeline::TraceScope pipeTrace(DrawPipeline::Trace::QueueDispatch, DrawPipeline::Trace::QueueDispatchEnd, submission.queue);
    PerformanceTimer timing("Driver.Dispatch");
    // Live `skip_queue_dispatches` (experiment only, wrong rendering): drops every dispatch of the
    // queue named by APS5_SKIP_DISPATCH_QUEUE (hex, default 0x22), to measure what they cost a frame.
    static const std::uint32_t skipQueue = [] {
        const char* text = std::getenv("APS5_SKIP_DISPATCH_QUEUE");
        return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 16)) : 0x22u;
    }();
    if (submission.queue == skipQueue && Graphics::LiveFlag("skip_queue_dispatches", false)) return;
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    auto it = submission.shaders->upper_bound(address);
    std::shared_ptr<const ShaderSnapshot> registeredShader;
    if (it != submission.shaders->begin()) {
        --it;
        if (address - it->second->codeAddress < it->second->code.size() * sizeof(std::uint32_t)) registeredShader = it->second;
    }
    if (!registeredShader) registeredShader = ReadRawComputeShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    static const std::pair<std::uint64_t, std::uint64_t> traceRange = [] {
        const char* text = std::getenv("APS5_TRACE_DISPATCH_RANGE");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto begin = std::strtoull(text, &end, 0);
        const auto last = end != nullptr && *end == '-' ? std::strtoull(end + 1, nullptr, 0) : begin + 1;
        return std::pair<std::uint64_t, std::uint64_t>{begin, last};
    }();
    if (traceRange.second != 0 && address >= traceRange.first && address < traceRange.second) {
        static std::atomic<std::uint64_t> traced{0};
        const auto count = traced.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 2000 || count % 1000 == 0) std::fprintf(stderr, "[trace-dispatch] #%llu shader 0x%llx queue 0x%x present %llu %s groups %u %u %u args 0x%llx registered %d words %zu\n", static_cast<unsigned long long>(count), static_cast<unsigned long long>(address), submission.queue, static_cast<unsigned long long>(Graphics::Recorder::Presents()), indirectArguments != 0 ? "indirect" : "direct", packet.size() > 3 ? packet[1] : 0u, packet.size() > 3 ? packet[2] : 0u, packet.size() > 3 ? packet[3] : 0u, static_cast<unsigned long long>(indirectArguments), snapshot.header.empty() ? 0 : 1, snapshot.code.size());
    }
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    static const bool unlockedDevice = std::getenv("APS5_NO_UNLOCKED_DEVICE") == nullptr;
    std::shared_ptr<VulkanDevice> localDevice = unlockedDevice ? device.Load() : nullptr;
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    std::array<std::uint32_t, 5> resolved{};
    if (indirectArguments != 0 && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {

        recordQueuedLabelsBeforeRead(submission.queue);
        const auto readStart = std::chrono::steady_clock::now();
        resolved = Pm4::ReadDispatchArguments(indirectArguments, packet[4]);
        countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    if (fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
        pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
        return;
    }
    // APS5_PIPELINED_DISPATCHES=1: a direct dispatch on the draw pipeline's queue keeps the pipeline
    // running; its device work becomes the committer's next job instead of a drain.
    static const bool pipelinedDispatches = std::getenv("APS5_PIPELINED_DISPATCHES") != nullptr;
    bool deferDrain = pipelinedDispatches && DrawPipeline::Active() && indirectArguments == 0 && !matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute);
    if (DrawPipeline::Active() && !deferDrain) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    if (indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
        pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
        return;
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header, SnapshotCodeHash(snapshot, codeOffset)},
        {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory},
        localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
        {0, 0, 0, 128}
    };
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static double captureMs = 0, keyMs = 0, recompileMs = 0, deviceMs = 0;
    static std::uint64_t cacheHits = 0;
    static std::uint64_t dispatches = 0;
    auto lap = std::chrono::steady_clock::now();

    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    static const bool noDispatchCacheEnv = std::getenv("APS5_NO_DISPATCH_CACHE") != nullptr;

    static const std::pair<std::uint64_t, std::uint64_t> probeDispatch = [] {
        const char* text = std::getenv("APS5_PROBE_DISPATCH");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto probeAddress = std::strtoull(text, &end, 16);
        const auto index = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0ull;
        return std::pair<std::uint64_t, std::uint64_t>{probeAddress, index};
    }();
    bool probeThis = false;

    if (probeDispatch.first != 0 && (address & 0xfffffffffull) == (probeDispatch.first & 0xfffffffffull)) {
        static std::atomic<std::uint64_t> dispatchesSeen{0};
        probeThis = dispatchesSeen.fetch_add(1) == probeDispatch.second;
        if (probeThis) std::fprintf(stderr, "[gpu] probing dispatch %llu of 0x%llx\n", static_cast<unsigned long long>(probeDispatch.second), static_cast<unsigned long long>(address));
    }

    struct ProbeScope {
        bool active;
        explicit ProbeScope(bool active) : active(active) { if (active) ShaderRecompiler::SetDebugProbeActive(true); }
        ~ProbeScope() { if (active) ShaderRecompiler::SetDebugProbeActive(false); }
    } probeScope{probeThis};
    const bool noDispatchCache = noDispatchCacheEnv || probeThis;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    for (const auto word : userData) mix(word);

    static const bool keyHygiene = std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    if (keyHygiene) {
        for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u}) {
            const auto found = queue.shader.find(offset);

            mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
        }
    } else {
        for (const auto& [offset, value] : queue.shader) {
            mix(offset);
            mix(value);
        }
    }
    profileDispatchKeys(address, userData, key);
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiledResult;

    std::shared_ptr<DispatchVariant> keepVariant;

    std::shared_ptr<DispatchVariant> attachVariant;

    std::shared_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;

    std::vector<std::uint32_t> liveWords;
    bool dataHit = false;
    bool cached = false;
    bool validated = false;

    std::shared_ptr<DispatchEntry> missedEntry;
    bool missedDiffering = false;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;

    static const bool traceCache = std::getenv("APS5_TRACE_DISPATCH_CACHE") != nullptr;
    if (traceCache) {
        std::lock_guard traceLock(dispatchCacheMutex);
        struct Last { std::vector<std::uint32_t> userData; std::map<std::uint32_t, std::uint32_t> shader; std::uint64_t key; };
        static std::map<std::uint64_t, Last> last;
        static int reports = 0;
        auto& previous = last[address];
        if (previous.key != 0 && previous.key != key && reports < 200) {
            std::string what;
            for (std::size_t i = 0; i < userData.size(); ++i) {
                if (i >= previous.userData.size() || previous.userData[i] != userData[i]) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " user[%zu] %08x->%08x", i, i < previous.userData.size() ? previous.userData[i] : 0u, userData[i]);
                    what += text;
                }
            }
            for (const auto& [offset, value] : queue.shader) {
                const auto old = previous.shader.find(offset);
                if (old == previous.shader.end() || old->second != value) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " sh[%x] %08x->%08x", offset, old == previous.shader.end() ? 0u : old->second, value);
                    what += text;
                }
            }
            ++reports;
            std::fprintf(stderr, "[dispatch-cache] 0x%llx key changed:%s\n", static_cast<unsigned long long>(address), what.c_str());
        }
        previous.userData = userData;
        previous.shader = std::map<std::uint32_t, std::uint32_t>(queue.shader.begin(), queue.shader.end());
        previous.key = key;
    }

    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    // Live `dispatch_shape_reuse` (env APS5_DISPATCH_SHAPE_REUSE=1): a miss with no entry may reuse
    // an earlier capture of the same program whose user words differ only where the walk takes
    // them as pointers or copies them (ShapeReuse.cpp). The static key is the key above without the
    // user words: the shape key adds the words the program's plan does not let move.
    const bool shapeOn = !noDispatchCache && !stampValidate() && !probeThis && shapeReuse();
    std::uint64_t staticKey = 0;
    if (shapeOn) {
        staticKey = 0xcbf29ce484222325ull;
        const auto mixStatic = [&](std::uint64_t value) {
            staticKey ^= value;
            staticKey *= 0x100000001b3ull;
        };
        mixStatic(address);
        mixStatic(packet[4] & 0x8000u);
        for (const auto threads : compute.partialThreads) mixStatic(threads);
        mixStatic(userData.size());
        if (keyHygiene) {
            for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u}) {
                const auto found = queue.shader.find(offset);
                mixStatic(found == queue.shader.end() ? (1ull << 32u) : found->second);
            }
        } else {
            for (const auto& [offset, value] : queue.shader) {
                if (offset >= 0x240u && offset < 0x260u) continue;
                mixStatic(offset);
                mixStatic(value);
            }
        }
        mixStatic(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    }
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    lookupDispatch(address, submission, key, noDispatchCache, traceCache, profile, memory, phaseTiming, phaseMs, compiledResult, keepVariant, captured, liveWords, dataHit, cached, validated, missedEntry, missedDiffering);
    timing.Mark("lookup_cache");
    std::shared_ptr<const std::vector<std::uint32_t>> shapeWords;
    bool shapeHit = false;
    const auto forgetAtShape = GuestMemory::ForgetSerial();
    if (!cached && shapeOn && missedEntry == nullptr) {
        shapeHit = lookupShape(snapshot, codeOffset, request, memory, address, submission.queue, staticKey, registeredShader, userData, compiledResult, capture, captured, shaderMemory, shapeWords);
        timing.Mark("shape_reuse");
    }
    if (cached) {
        require(keepVariant != nullptr && keepVariant->shader == registeredShader, "dispatch cache belongs to another registered shader");
        captureMs += phaseTiming.Elapsed();
    } else if (shapeHit) {
        captureMs += phaseTiming.Elapsed();
        phaseTiming.Phase(PhaseCapture);
        insertDispatch(address, key, noDispatchCache, profile, registeredShader, forgetAtShape, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming);
    } else {
        shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
        std::uint64_t forgetAtCapture = 0;

        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        static const bool dumpFailures = std::getenv("APS5_DUMP_SHADER_FAILURES") != nullptr;
        const auto shapeStarted = std::chrono::steady_clock::now();
        try {
            const auto waitedBefore = traceCapSync() ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            forgetAtCapture = GuestMemory::ForgetSerial();
            const auto invocation = InvocationFor(snapshot, codeOffset, request, shaderMemory.get());
            request.shader = invocation.Request().shader;
            timing.Mark("prepared_invocation");

            capture = [&] {
                const SampledReadScope sampling(evidenceReads);
                return shaderMemory->Capture(invocation);
            }();
            captured = shaderMemory->Regions();
            request.context.memory = captured;
            if (traceCapSync()) traceCapture("dispatch-capture", address, submission.queue, captured, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
            captureMs += phaseTiming.Elapsed();
            phaseTiming.Phase(PhaseCapture);
            if (dumpShaders) static_cast<void>(dumpRequest(address, request));
            const auto started = std::chrono::steady_clock::now();


            timing.Mark("capture_resources");
            compiledResult = invocation.Materialize(*capture);
            timing.Mark("materialize");
            if (shapeOn) {
                CountShapeFullCapture(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - shapeStarted).count());
                rememberShape(address, staticKey, registeredShader, userData, memory, *shaderMemory, invocation.Request(), capture, compiledResult);
            }
            if (compiledResult->cacheHit) ++cacheHits;
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            static double totalMs = 0;
            totalMs += elapsed;
            if (profile && elapsed > 200) std::fprintf(stderr, "[gpu] compute shader 0x%llx recompile took %.0f ms (%zu SPIR-V words, %zu captured regions, total %.1f s)\n", static_cast<unsigned long long>(address), elapsed, compiledResult->spirv.size(), captured.size(), totalMs / 1000);
        } catch (const std::exception& error) {
            // Capture may fail during descriptor materialization, before the normal
            // assignment above. Preserve the reads completed up to that failure.
            if (dumpShaders || dumpFailures) {
                captured = shaderMemory->Regions();
                request.context.memory = captured;
            }
            const auto dump = (dumpShaders || dumpFailures) ? dumpRequest(address, request) : std::string{};

            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            char where[96];
            if (dump.empty()) std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
            else std::snprintf(where, sizeof(where), "compute shader 0x%llx (%s): ", static_cast<unsigned long long>(address), dump.c_str());
            throw std::runtime_error(where + reason);
        }
        recompileMs += phaseTiming.Elapsed();
        phaseTiming.Phase(PhaseRecompile);
        insertDispatch(address, key, noDispatchCache, profile, registeredShader, forgetAtCapture, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming);
    }
    if (verifyDataHits() && dataHit) verifyDataHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);

    if (recordQueuedLabelsAfterCapture(submission.queue, captured)) {
        dispatch(queue, packet, submission, indirectArguments);
        return;
    }
    phaseTiming.Phase(PhaseQueuedLabels);
    const auto& compiled = *compiledResult;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {

        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = (groups[axis] + threads - 1) / threads;
        }
    }
    static const bool traceIo = std::getenv("APS5_TRACE_DISPATCH_IO") != nullptr;
    if (traceIo) {

        std::string words;
        if (std::getenv("APS5_TRACE_DISPATCH_IO")[0] == '2') {
            for (const auto word : userData) {
                char text[12];
                std::snprintf(text, sizeof(text), " %08x", word);
                words += text;
            }
        }
        std::fprintf(stderr, "[dispatch-io] shader 0x%llx%s\n", static_cast<unsigned long long>(address), words.c_str());
    }

    const auto rethrow = [&](const std::exception& error) {
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + std::string(error.what()));
    };
    phaseTiming.Phase(PhaseSnapshots);

    if (deferDrain) {
        auto writes = ComputeWriteRanges(compiled);
        if (writes.has_value() && !stampValidate() && deferredLabels().labels.empty() && !probeThis) {
            const auto commitQueue = submission.queue;
            const auto commitPacket = (packet[0] >> 8u) & 0xffu;
            const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
            const std::array<std::uint32_t, 5> packetWords{packet[0], packet[1], packet[2], packet[3], packet[4]};
            const auto runToken = Graphics::Recorder::DispatchRunToken();
            auto commit = [this, localDevice, compiledResult, keepVariant, attachVariant, shaderMemory, shapeWords, captured, registeredShader, groups, address, dataHit, cached, commitQueue, commitPacket, epoch, packetWords, codeOffset, userData, compute, runToken]() mutable {
                DrawPipeline::FollowEpoch(epoch);
                GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
                // The dispatch's grid, for the store covers of its buffer elements (a direct dispatch).
                const Graphics::DispatchGridScope gridScope(ShaderRecompiler::DispatchGrid{groups, compute.numThreads});
                // The dispatch run token of the packet, for the committer's recordDispatch.
                struct RunToken {
                    explicit RunToken(std::uint64_t token) { Graphics::Recorder::SetDispatchRunToken(token); }
                    ~RunToken() { Graphics::Recorder::SetDispatchRunToken(0); }
                } runTokenScope{runToken};
                const auto& compiled = *compiledResult;
                std::vector<Graphics::GuestMemorySnapshot> snapshots;
                snapshots.reserve(captured.size());
                for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
                std::shared_ptr<RecipeHit> recipeHit;
                if (cached && keepVariant != nullptr) recipeHit = localDevice->PrepareRecipe(keepVariant->recipe.load(std::memory_order_acquire), false);
                std::shared_ptr<const Recipe> builtRecipe;
                auto* const attachTo = keepVariant != nullptr ? keepVariant.get() : attachVariant.get();
                const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
                bool writersNoted = false;
                for (;;) {
                    std::shared_ptr<PreparedDispatch> prepared;
                    if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) prepared = localDevice->PrepareDispatch(compiled, snapshots);
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
                    struct SubmitDue {
                        Driver& driver;
                        const std::shared_ptr<VulkanDevice>& device;
                        ~SubmitDue() noexcept(false) {
                            if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
                        }
                    } submitDue{*this, localDevice};
                    recordLabelsForPacket(localDevice.get(), commitQueue);
                    if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                        noteWrittenBuffers(address, commitQueue, compiled);
                        writersNoted = true;
                    }
                    if (recipeHit != nullptr) {
                        VulkanDevice::IndirectOutcome outcome{0, 0};
                        if (localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], 0, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit) == RecipeOutcome::Rebuild) {
                            recipeHit = nullptr;
                            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                            continue;
                        }
                    } else {
                        localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr);
                    }
                    if (builtRecipe != nullptr) {
                        if (dataHit) {
                            auto own = std::make_shared<Recipe>(*builtRecipe);
                            own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                            builtRecipe = std::move(own);
                        }
                        attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
                        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                    }
                    break;
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, commitQueue, compiled);
                noteDwordFillKernel(std::span(registeredShader->code).subspan(codeOffset), userData, packetWords, 0, compute);
            };
            auto held = std::make_shared<decltype(commit)>(std::move(commit));
            DrawPipeline::Queue0().EnqueueOrdered([held] { (*held)(); }, std::move(*writes));
            if (profile) {
                auto& pending = pendingDispatchPhases();
                pending.outcome = DispatchOutcome::Real;
                pending.phases = false;
            }
            return;
        }
        DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    std::shared_ptr<RecipeHit> recipeHit;
    if (cached && keepVariant != nullptr && !stampValidate()) {
        recipeHit = localDevice->PrepareRecipe(keepVariant->recipe.load(std::memory_order_acquire), indirectArguments != 0);
        phaseTiming.Phase(PhaseRecipePrecheck);
    }

    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : keepVariant != nullptr ? keepVariant.get() : attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    // A direct dispatch's grid, for the store covers of its buffer elements (GuestBufferMemory's
    // write-back elision); an indirect one's counts are not known here.
    std::optional<Graphics::DispatchGridScope> gridScope;
    if (indirectArguments == 0) gridScope.emplace(ShaderRecompiler::DispatchGrid{groups, compute.numThreads});

    for (;;) {

        std::shared_ptr<PreparedDispatch> prepared;
        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) {
            try {
                prepared = localDevice->PrepareDispatch(compiled, snapshots);
            } catch (const std::exception& error) {
                rethrow(error);
            }
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
                phaseLap = now;
                double parts = 0;
                if (prepared != nullptr) {
                    const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
                    phaseMs[PhasePrepareKey] += phases[0];
                    phaseMs[PhasePrepareFind] += phases[1];
                    phaseMs[PhasePreparePrecollect] += phases[2];
                    phaseMs[PhasePreparePresync] += phases[3];
                    phaseMs[PhasePrepareStageA] += phases[4];
                    for (const auto part : phases) parts += part;
                }
                phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
            }
        }
        GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(address, submission.queue, compiled);
            writersNoted = true;
        }
        phaseTiming.Phase(PhaseNoteWriters);
        try {
            if (recipeHit != nullptr) {
                VulkanDevice::IndirectOutcome outcome{0, 0};
                const auto result = localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit);
                if (result == RecipeOutcome::Rebuild) {

                    recipeHit = nullptr;
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, indirectArguments != 0);
                    continue;
                }
                if (indirectArguments != 0) countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr);
            }
        } catch (const std::exception& error) {
            rethrow(error);
        }
        if (builtRecipe != nullptr) {
            if (dataHit) {

                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, indirectArguments != 0);
        }
        break;
    }
    phaseTiming.Phase(PhaseDevice);
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, submission.queue, compiled);
    noteDwordFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, packet, indirectArguments, compute);
    static const std::uint64_t traceWrite = [] { const char* text = std::getenv("APS5_TRACE_WRITE_ADDRESS"); return text != nullptr ? std::strtoull(text, nullptr, 0) : 0ull; }();
    if (traceWrite != 0) {
        for (const auto& binding : compiled.bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            for (std::uint32_t element = 0; element < binding.count && binding.guestDescriptor.size() >= (static_cast<std::size_t>(element) + 1) * 4; ++element) {
                const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4, 4);
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto begin = descriptor.Base48();
                const auto end = begin + descriptor.GetSize();
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (begin <= traceWrite && traceWrite < end) std::fprintf(stderr, "[trace-write] dispatch shader 0x%llx queue 0x%x %s buffer element %u 0x%llx-0x%llx over 0x%llx\n", static_cast<unsigned long long>(address), submission.queue, written ? "writes" : "reads", element, static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end), static_cast<unsigned long long>(traceWrite));
            }
        }
    }
    deviceMs += phaseTiming.Elapsed();
    phaseTiming.Phase(PhaseTail);
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = cached;
        pending.validated = validated;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && ++dispatches % 100 == 0) AgcDriver::ProfilePrint_nid_no_patch( "[gpu] %llu dispatches (%llu dispatch cache hits, %llu evictions, %llu recompile cache hits): capture %.1f s, cache key %.1f s, recompile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches), static_cast<unsigned long long>(dispatchCacheHits), static_cast<unsigned long long>(dispatchCacheEvictions), static_cast<unsigned long long>(cacheHits), captureMs / 1000, keyMs / 1000, recompileMs / 1000, deviceMs / 1000);
}

}
