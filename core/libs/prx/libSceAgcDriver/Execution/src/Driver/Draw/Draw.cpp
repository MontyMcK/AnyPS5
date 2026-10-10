#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
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
    auto tail = [this, &queue, &submission, clipControl, packet, rejectedOut, splitRejected, splitDraw, splitEpoch, phaseMsIn = phaseMs, phaseLapIn = phaseLap, capturesIn = captures, drawParameters, pipelined, traceIndirect, gpuLock = std::move(gpuLock), localDevice, useDrawEntries, registerKey, drawKey, entry = std::move(entry), decode = std::move(decode), dataEntry = std::move(dataEntry), shapeKey, programs = std::move(programs), memory = std::move(memory), linked = std::move(linked), vertexInfos = std::move(vertexInfos), decodeReads = std::move(decodeReads), results = std::move(results), compiledStages = std::move(compiledStages), stages = std::move(stages), pushCursorBytes, programResults = std::move(programResults), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), recompiled = std::move(recompiled), dataCandidates = std::move(dataCandidates), reused = std::move(reused), dataStagesReused, dataStagesCompiled, dataVerified, drawHit, verifyHit]() mutable -> DrawVerdict {
    if (splitDraw) DrawPipeline::FollowEpoch(splitEpoch);
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
    ShaderMemory shaderMemory(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter());
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
            memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
        } else {
            const auto candidate = std::find_if(dataCandidates[i].begin(), dataCandidates[i].end(), [&](const auto& kept) { return kept.first->pushOffset == pushCursorBytes; });
            if (candidate != dataCandidates[i].end() && !verifyDrawDataHits()) {
                matched[i] = candidate->first;
                matchedRegions[i] = std::move(candidate->second);
                reused[i] = true;
                programResults[i] = matched[i]->compiled.get();
                memory.insert(memory.end(), matchedRegions[i].begin(), matchedRegions[i].end());
            } else {
                if (!dataCandidates[i].empty() && !verifyDrawDataHits()) decodeVertexInfo(i);
                compiledStages.push_back(materializeDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, vertexInfos, memory, linked, drawParameters, localDevice, shaderMemory, stageCaptures, recompiled, drawHit, matched, matchedRegions, profile, dumpTarget, dumpSlot1, captures, phaseTiming, phaseMs, rejected));
                if (!rejected.empty()) return DrawVerdict::Rejected;
                programResults[i] = compiledStages.back().get();
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
    if (registerKey && !drawParameters.indirect && Graphics::DrawRecipes()) {
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
        phaseTiming.Phase(DrawRowVectors);
        auto writes = drawWriteRanges(graphics, stages);
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
        auto commit = [this, localDevice = std::move(localDevice), decode = std::move(decode), drawParameters, stages = std::move(stages), snapshots = std::move(snapshots), recipe = std::move(recipe), recipeStages = std::move(recipeStages), drawKey, commitQueue, commitPacket, epoch, programs = std::move(programs), results = std::move(results), compiledStages = std::move(compiledStages), memory = std::move(memory), linked = std::move(linked), stageCaptures = std::move(stageCaptures), matched = std::move(matched), matchedRegions = std::move(matchedRegions), fresh = std::move(fresh), decodeReads = std::move(decodeReads), shaderMemory = std::move(shaderMemory), entry = std::move(entry), dataEntry = std::move(dataEntry)] {
            DrawPipeline::FollowEpoch(epoch);
            GuestMemory::SetCurrentPacket(commitPacket, commitQueue);
            commitDraw(localDevice, commitQueue, decode->state, drawParameters, stages, snapshots, recipe, recipeStages, drawKey);
        };
        auto held = std::make_shared<decltype(commit)>(std::move(commit));
        DrawPipeline::Queue0().Enqueue([held] { (*held)(); }, std::move(writes));
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
    if (!splitDraw) return tail();
    auto held = std::make_shared<decltype(tail)>(std::move(tail));
    DrawPipeline::Queue0().Prepare([held] { static_cast<void>((*held)()); });
    return drawn();
}

void Driver::commitDraw(std::shared_ptr<VulkanDevice> localDevice, std::uint32_t queue, const Graphics::State& graphics, const Pm4::DrawParameters& drawParameters, std::span<const Graphics::CompiledShader> stages, std::span<const Graphics::GuestMemorySnapshot> snapshots, const std::shared_ptr<const DrawRecipe>& recipe, const std::vector<std::shared_ptr<DispatchVariant>>& recipeStages, std::uint64_t drawKey) {
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    if (auto current = device.Load(); current != nullptr && current != localDevice) localDevice = std::move(current);
    struct SubmitDue {
        Driver& driver;
        const std::shared_ptr<VulkanDevice>& device;
        ~SubmitDue() noexcept(false) {
            if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
        }
    } submitDue{*this, localDevice};
    recordLabelsForPacket(localDevice.get(), queue);
    noteDrawWriters(stages, queue);
    if (recipe != nullptr) {
        if (localDevice->DrawFromRecipe(graphics, drawParameters, stages, snapshots, recipe) == RecipeOutcome::Recorded) return;
        VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
    }
    std::shared_ptr<const DrawRecipe> built;
    localDevice->Draw(graphics, drawParameters, stages, snapshots, recipeStages.empty() ? nullptr : &built);
    if (built != nullptr) attachDrawRecipe(drawKey, recipeStages, std::move(built));
}

}
