#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <algorithm>
#include <optional>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <array>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

std::vector<std::shared_ptr<DispatchVariant>>& RebasedProbes() {
    static thread_local std::vector<std::shared_ptr<DispatchVariant>> probes;
    return probes;
}

bool rebasedDrawHits() {
    static const bool enabled = std::getenv("APS5_REBASED_DRAW_HITS") != nullptr || std::getenv("APS5_VERIFY_REBASED_HITS") != nullptr;
    return enabled;
}

struct RebaseCounters {
    std::atomic<std::uint64_t> stages{0}, sizes{0}, validated{0}, unusable{0}, used{0}, noEntry{0}, notVertex{0}, unlearned{0}, ambiguous{0};
    std::atomic<std::int64_t> last{0};
};

RebaseCounters& rebaseCounters() {
    static RebaseCounters counters;
    return counters;
}

void reportRebase() {
    auto& counters = rebaseCounters();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = counters.last.load(std::memory_order_relaxed);
    if (now - last < 10'000'000'000ll || !counters.last.compare_exchange_strong(last, now)) return;
    std::fprintf(stderr, "[draw-cache] rebased (10 s): stages tried %llu, size mismatch %llu, failed validation %llu, unusable patch %llu, used %llu; misses without a shape entry %llu, shape entry not vertex path %llu, unlearned %llu, ambiguous %llu\n", static_cast<unsigned long long>(counters.stages.exchange(0)), static_cast<unsigned long long>(counters.sizes.exchange(0)), static_cast<unsigned long long>(counters.validated.exchange(0)), static_cast<unsigned long long>(counters.unusable.exchange(0)), static_cast<unsigned long long>(counters.used.exchange(0)), static_cast<unsigned long long>(counters.noEntry.exchange(0)), static_cast<unsigned long long>(counters.notVertex.exchange(0)), static_cast<unsigned long long>(counters.unlearned.exchange(0)), static_cast<unsigned long long>(counters.ambiguous.exchange(0)));
}

bool verifyRebasedHits() {
    static const bool enabled = std::getenv("APS5_VERIFY_REBASED_HITS") != nullptr;
    return enabled;
}

void Driver::lookupDraw(const Submission& submission, const std::shared_ptr<VulkanDevice>& localDevice, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, bool useDrawEntries, bool registerKey, bool profile, std::uint64_t& drawKey, std::shared_ptr<DrawEntry>& entry, std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool& drawHit, bool& verifyHit, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, const std::shared_ptr<DrawEntry>& dataEntry, DrawDataCandidates& dataCandidates) {
    using Role = ShaderRecompiler::ProgramRole;
    if (useDrawEntries) {
        if (!registerKey) {
            drawKey = 0xcbf29ce484222325ull;
            const auto mix = [&](std::uint64_t value) {
                drawKey ^= value;
                drawKey *= 0x100000001b3ull;
            };
            mix(localDevice->Serial());
            mix(static_cast<std::uint64_t>(graphics.stages.path));
            mix(graphics.stages.registerValue);
            mix(graphics.stages.vertexWaveSize);
            mix(graphics.stages.fragmentWaveSize);
            mix(graphics.stages.mesh.has_value());
            if (graphics.stages.mesh) {
                const auto& mesh = *graphics.stages.mesh;
                for (const auto value : {mesh.inputPrimitive, mesh.primitivesPerGroup, mesh.verticesPerGroup, mesh.maxVertices, mesh.maxPrimitives, mesh.threadsPerGroup, mesh.ldsSizeDwords, mesh.provokingVertex, mesh.esgsItemSize}) mix(value);
            }
            mix(graphics.stages.tessellation.has_value());
            if (graphics.stages.tessellation) {
                const auto& tess = *graphics.stages.tessellation;
                for (const auto value : {tess.inputControlPoints, tess.outputControlPoints, tess.domain, tess.partitioning, tess.outputTopology}) mix(value);
            }
            mix(graphics.rectList);
            mix(programs.size());
            for (std::size_t i = 0; i < programs.size(); ++i) {
                const auto& program = programs[i];
                mix(reinterpret_cast<std::uintptr_t>(program.snapshot.get()));
                mix(program.codeOffset);
                mix(static_cast<std::uint64_t>(roles[i]));
                mix(static_cast<std::uint64_t>(program.binary.stage));
                mix(program.userDataBase);
                mix(program.firstUserSgpr);
                mix(program.userData.size());
                for (const auto word : program.userData) mix(word);
                mix(vertexInfos[i].has_value());
                if (!vertexInfos[i]) continue;
                const auto& vertex = *vertexInfos[i];
                require(vertex.resourcesNum <= vertex.resources.size(), "vertex stage info resource count exceeds its table");
                mix(vertex.resourcesNum);
                mix(vertex.fetchAttribReg);
                mix(vertex.fetchBufferReg);
                mix(vertex.fetchEmbedded);
                mix(vertex.paClVsOutCntl);
                for (std::uint32_t r = 0; r < vertex.resourcesNum; ++r) {
                    for (const auto field : vertex.resources[r].fields) mix(field);
                    const auto& destination = vertex.resourcesDst[r];
                    mix(static_cast<std::uint32_t>(destination.registerStart));
                    mix(static_cast<std::uint32_t>(destination.registersNum));
                    mix(static_cast<std::uint32_t>(destination.attrId));
                    mix(destination.fetchIndex);
                }
            }
            require(pixel.interpolatorCount <= pixel.interpolatorSettings.size(), "pixel stage info interpolator count exceeds its table");
            mix(pixel.interpolatorCount);
            for (std::uint32_t i = 0; i < pixel.interpolatorCount; ++i) mix(pixel.interpolatorSettings[i]);
            mix(pixel.inputAddr);
            for (const bool flag : {pixel.wave32, pixel.hasPerspectiveCenterVgpr, pixel.perspectiveCentroid, pixel.posX, pixel.posY, pixel.posZ, pixel.posW, pixel.frontFace, pixel.ancillary, pixel.sampleShading, pixel.noPerspective, pixel.linearCentroid, pixel.pixelKillEnable, pixel.depthExportEnable, pixel.sampleMaskExportEnable, pixel.earlyZ, pixel.executeOnNoop}) mix(flag);
            mix(static_cast<std::uint64_t>(pixel.conservativeZExport));
            mix(pixel.orderedPixelShader);
            for (const auto value : pixel.targetOutputMode) mix(value);
            for (const auto value : pixel.targetExportMapping) mix(value);
            for (const auto value : pixel.targetExportPacking) mix(static_cast<std::uint64_t>(value));
            std::lock_guard cacheLock(drawCacheMutex);
            ++drawEntryCounters.lookups;
            const auto found = drawCache.find(drawKey);
            if (found != drawCache.end()) entry = found->second;
            else ++drawEntryCounters.absent;
            maybeReportDrawCache(profile);
        }
        if (entry != nullptr && DrawPipeline::SplitDraws()) DrawPipeline::Queue0().DrainPrep();
        if (entry != nullptr) {
            const auto waitedBeforeValidate = profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            std::optional<DrawMiss> miss;
            std::vector<std::size_t> ranks(programs.size(), 0);
            std::uint64_t stageValidations = 0, stageEqual = 0, compared = 0, imagesFlushed = 0, runsSynced = 0;
            if (entry->stages.size() != programs.size()) miss = DrawMiss::Stages;

            std::uint32_t cursor = 0;
            {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);

                std::optional<SampledReadScope> sampling;
                for (std::size_t i = 0; !miss && i < programs.size(); ++i) {
                    if (roles[i] == Role::GeometryBack) continue;
                    ++stageValidations;
                    const auto& variants = entry->stages[i];
                    auto outcome = EntryOutcome::Differing;
                    bool anyLayout = false;
                    for (std::size_t rank = 0; rank < variants.size(); ++rank) {
                        const auto& variant = variants[rank];
                        if (variant->pushOffset != cursor) continue;
                        auto& regions = matchedRegions[i];
                        regions.clear();
                        appendEntryRegions(*variant, regions);
                        ++compared;
                        auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                        if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                        if (!anyLayout) outcome = result;
                        anyLayout = true;
                        if (result != EntryOutcome::Equal) continue;
                        matched[i] = variant;
                        ranks[i] = rank;
                        ++stageEqual;
                        cursor += static_cast<std::uint32_t>(variant->compiled->pushConstants.size());
                        break;
                    }
                    if (matched[i] != nullptr) continue;
                    if (!anyLayout) miss = DrawMiss::Layout;
                    else if (outcome != EntryOutcome::Differing) miss = DrawMiss::Gate;
                    else miss = roles[i] == Role::Fragment ? DrawMiss::FragmentDiffering : i == 0 ? DrawMiss::FrontDiffering : DrawMiss::OtherDiffering;
                }
            }
            drawHit = !miss;
            std::lock_guard cacheLock(drawCacheMutex);
            auto& counters = drawEntryCounters;
            counters.stageValidations += stageValidations;
            counters.stageEqual += stageEqual;
            counters.variantsCompared += compared;
            if (drawHit) {
                ++counters.hits;
                if (registerKey) ++counters.registerKeyHits;
                ++drawCacheHits;
                ++drawCacheClock;
                bool rotate = false;
                for (std::size_t i = 0; i < programs.size(); ++i) {
                    if (matched[i] == nullptr) continue;
                    ++counters.variantHitsByRank[ranks[i]];
                    if (ranks[i] != 0) rotate = true;
                }

                const auto again = drawCache.find(drawKey);
                if (again != drawCache.end() && again->second == entry) {
                    if (rotate) {
                        auto rotated = std::make_shared<DrawEntry>();
                        rotated->decode = entry->decode;
                        rotated->stages = entry->stages;
                        rotated->shape = entry->shape;
                        rotated->recipes.store(entry->recipes.load());
                        for (std::size_t i = 0; i < programs.size(); ++i) {
                            if (ranks[i] == 0) continue;
                            auto& variants = rotated->stages[i];
                            variants.erase(variants.begin() + static_cast<std::ptrdiff_t>(ranks[i]));
                            variants.insert(variants.begin(), matched[i]);
                        }
                        rotated->touched = entry->touched;
                        rotated->order = entry->order;
                        again->second = std::move(rotated);
                    }
                    if (drawCacheClock - again->second->touched > drawCacheEntries() / 8) {
                        drawOrder.splice(drawOrder.begin(), drawOrder, again->second->order);
                        again->second->touched = drawCacheClock;
                        ++counters.touches;
                    }
                }
                if (verifyDrawEntries()) {
                    ++counters.verifyHits;
                    verifyHit = true;
                    drawHit = false;
                }
            } else {
                ++counters.misses[static_cast<std::size_t>(*miss)];
                if (registerKey && entry->decode != nullptr) ++counters.decodePartial;
            }
            if (profile) {

                phaseTiming.Phase(DrawRowKeyLookupValidate);
                const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBeforeValidate, phaseMs[DrawRowKeyLookupValidate]);
                phaseMs[DrawRowKeyLookupValidate] -= waited;
                phaseMs[DrawRowValidateWait] += waited;
                counters.validateUs += phaseMs[DrawRowKeyLookupValidate] * 1000;
            }
        } else if (dataEntry != nullptr && graphics.stages.path == Graphics::ShaderPath::Vertex && dataEntry->stages.size() == programs.size() && dataEntry->decode->programs.size() == programs.size()) {
            std::uint64_t imagesFlushed = 0, runsSynced = 0;
            const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
            std::optional<SampledReadScope> sampling;
            auto& probes = RebasedProbes();
            probes.assign(programs.size(), nullptr);
            for (std::size_t i = 0; i < programs.size(); ++i) {
                if (roles[i] == Role::GeometryBack) continue;
                if (programs[i].userData != dataEntry->decode->programs[i].userData) {
                    if (!rebasedDrawHits() || i >= dataEntry->stages.size()) continue;
                    const auto& before = dataEntry->decode->programs[i].userData;
                    const auto& after = programs[i].userData;
                    rebaseCounters().stages.fetch_add(1, std::memory_order_relaxed);
                    if (before.size() != after.size()) {
                        rebaseCounters().sizes.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    auto& candidates = dataCandidates[i];
                    for (const auto& variant : dataEntry->stages[i]) {
                        if ((i == 0 && variant->pushOffset != 0) || std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate.first->pushOffset == variant->pushOffset; })) continue;
                        auto rebased = std::make_shared<DispatchVariant>();
                        rebased->shader = variant->shader;
                        rebased->forgetSerial = GuestMemory::ForgetSerial();
                        rebased->pushOffset = variant->pushOffset;
                        rebased->vertexInfo = variant->vertexInfo;
                        rebased->compiled = variant->compiled;
                        struct ShiftedRun {
                            std::uint64_t begin;
                            std::uint64_t end;
                            std::size_t words;
                            std::size_t count;
                        };
                        std::vector<ShiftedRun> shiftedRuns;
                        shiftedRuns.reserve(variant->runs.size());
                        if (variant->readRuns.empty() != variant->runs.empty() || variant->readSources.size() != variant->readRuns.size()) {
                            rebaseCounters().unlearned.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        std::vector<std::pair<std::uint64_t, std::uint64_t>> shiftedRead;
                        shiftedRead.reserve(variant->readRuns.size());
                        bool located = before.size() <= 63;
                        for (std::size_t r = 0; r < variant->readRuns.size() && located; ++r) {
                            const auto candidates = variant->readSources[r];
                            std::optional<std::int64_t> delta;
                            if ((candidates >> 63u) != 0) delta = 0;
                            for (std::size_t w = 0; w < before.size() && located; ++w) {
                                if (((candidates >> w) & 1u) == 0) continue;
                                const auto shift = static_cast<std::int64_t>(static_cast<std::int32_t>(after[w] - before[w]));
                                if (delta.has_value() && *delta != shift) located = false;
                                delta = shift;
                            }
                            if (!delta.has_value()) located = false;
                            if (!located) break;
                            const auto shifted = static_cast<std::uint64_t>(static_cast<std::int64_t>(variant->readRuns[r].first) + *delta);
                            shiftedRead.emplace_back(shifted, shifted + (variant->readRuns[r].second - variant->readRuns[r].first));
                        }
                        std::size_t source = 0;
                        for (const auto& [begin, end] : variant->runs) {
                            const auto count = static_cast<std::size_t>((end - begin) / sizeof(std::uint32_t));
                            std::optional<std::uint64_t> chosen;
                            for (std::size_t r = 0; r < variant->readRuns.size(); ++r) {
                                if (variant->readRuns[r].first != begin || variant->readRuns[r].second != end) continue;
                                if (chosen.has_value() && *chosen != shiftedRead[r].first) {
                                    located = false;
                                    break;
                                }
                                chosen = shiftedRead[r].first;
                            }
                            if (!located || !chosen.has_value() || source + count > variant->words.size()) {
                                located = false;
                                break;
                            }
                            shiftedRuns.push_back({*chosen, *chosen + (end - begin), source, count});
                            source += count;
                        }
                        if (!located) {
                            rebaseCounters().ambiguous.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        rebased->readRuns = std::move(shiftedRead);
                        rebased->readSources = variant->readSources;
                        std::stable_sort(shiftedRuns.begin(), shiftedRuns.end(), [](const ShiftedRun& a, const ShiftedRun& b) { return a.begin < b.begin; });
                        std::vector<std::uint32_t> oldWords;
                        oldWords.reserve(variant->words.size());
                        rebased->runs.reserve(shiftedRuns.size());
                        for (const auto& run : shiftedRuns) {
                            if (run.words + run.count > variant->words.size()) break;
                            rebased->runs.emplace_back(run.begin, run.end);
                            const auto position = static_cast<std::uint32_t>(oldWords.size());
                            if (run.count == 4) rebased->dataPositions.push_back(position);
                            oldWords.insert(oldWords.end(), variant->words.begin() + static_cast<std::ptrdiff_t>(run.words), variant->words.begin() + static_cast<std::ptrdiff_t>(run.words + run.count));
                        }
                        if (rebased->runs.size() != shiftedRuns.size()) continue;
                        rebased->words = oldWords;
                        std::vector<ShaderRecompiler::MemoryRegion> regions;
                        appendEntryRegions(*rebased, regions);
                        std::vector<std::pair<std::uint32_t, std::uint32_t>> live;
                        auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *rebased, regions, imagesFlushed, runsSynced, sampling, &live);
                        if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *rebased, regions, imagesFlushed, runsSynced, sampling, &live);
                        if (result != EntryOutcome::Equal && result != EntryOutcome::EqualData) {
                            rebaseCounters().validated.fetch_add(1, std::memory_order_relaxed);
                            static std::array<std::atomic<std::uint64_t>, 9> outcomes{};
                            static std::atomic<std::uint64_t> outcomeTotal{0};
                            outcomes[static_cast<std::size_t>(result)].fetch_add(1, std::memory_order_relaxed);
                            if (outcomeTotal.fetch_add(1, std::memory_order_relaxed) % 50000 == 49999) std::fprintf(stderr, "[draw-cache] rebased failure outcomes: differing %llu inaccessible %llu queued label %llu flushing image %llu publish moved %llu pending moved %llu forget moved %llu\n", static_cast<unsigned long long>(outcomes[2].load()), static_cast<unsigned long long>(outcomes[3].load()), static_cast<unsigned long long>(outcomes[4].load()), static_cast<unsigned long long>(outcomes[5].load()), static_cast<unsigned long long>(outcomes[6].load()), static_cast<unsigned long long>(outcomes[7].load()), static_cast<unsigned long long>(outcomes[8].load()));
                            static std::atomic<int> failurePrints{0};
                            if (result == EntryOutcome::Differing && failurePrints.load(std::memory_order_relaxed) < 60) {
                                std::size_t offset = 0;
                                for (const auto& [runBegin, runEnd] : rebased->runs) {
                                    const auto count = static_cast<std::size_t>((runEnd - runBegin) / sizeof(std::uint32_t));
                                    if (GuestMemory::Accessible(reinterpret_cast<const void*>(runBegin), count * 4)) {
                                        const auto* current = reinterpret_cast<const std::uint32_t*>(runBegin);
                                        for (std::size_t k = 0; k < count; ++k) {
                                            if (current[k] == rebased->words[offset + k] || (count == 4 && k == 0)) continue;
                                            if (failurePrints.fetch_add(1, std::memory_order_relaxed) < 60) std::fprintf(stderr, "[draw-cache] rebased failure: stage %zu run 0x%llx words %zu index %zu old %08x new %08x\n", i, static_cast<unsigned long long>(runBegin), count, k, rebased->words[offset + k], current[k]);
                                            break;
                                        }
                                    }
                                    offset += count;
                                }
                            }
                            continue;
                        }
                        std::vector<std::pair<std::array<std::uint32_t, 4>, std::array<std::uint32_t, 4>>> quads;
                        bool usable = true;
                        for (const auto& [at, value] : live) {
                            if (at + 4 > oldWords.size()) {
                                usable = false;
                                break;
                            }
                            std::array<std::uint32_t, 4> old{oldWords[at], oldWords[at + 1], oldWords[at + 2], oldWords[at + 3]};
                            auto fresh = old;
                            fresh[0] = value;
                            quads.emplace_back(old, fresh);
                            rebased->words[at] = value;
                        }
                        if (!usable) continue;
                        auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(*variant->compiled);
                        const auto memoryQuads = quads.size();
                        for (std::size_t start = 0; start + 4 <= before.size(); ++start) {
                            if (std::equal(before.begin() + static_cast<std::ptrdiff_t>(start), before.begin() + static_cast<std::ptrdiff_t>(start + 4), after.begin() + static_cast<std::ptrdiff_t>(start))) continue;
                            quads.emplace_back(std::array<std::uint32_t, 4>{before[start], before[start + 1], before[start + 2], before[start + 3]}, std::array<std::uint32_t, 4>{after[start], after[start + 1], after[start + 2], after[start + 3]});
                        }
                        std::vector<bool> quadUsed(quads.size(), false);
                        const auto patchQuads = [&](std::uint32_t* words, std::size_t count) {
                            for (std::size_t at = 0; at + 4 <= count; ++at) {
                                for (std::size_t q = 0; q < quads.size(); ++q) {
                                    const auto& [old, fresh] = quads[q];
                                    if (std::equal(old.begin(), old.end(), words + at)) {
                                        std::copy(fresh.begin(), fresh.end(), words + at);
                                        quadUsed[q] = true;
                                        break;
                                    }
                                }
                            }
                        };
                        const auto patchPrefix = [&](std::uint32_t* words, std::size_t count) {
                            if (count < before.size() || !std::equal(before.begin(), before.end(), words)) return false;
                            std::copy(after.begin(), after.end(), words);
                            return true;
                        };
                        const auto stale = [&](const std::uint32_t* words, std::size_t count, bool prefixed) {
                            const auto first = prefixed ? before.size() : 0;
                            for (std::size_t w = 0; w < before.size(); ++w) {
                                if (before[w] == after[w] || (prefixed && before[w] < 0x10000u)) continue;
                                if (first < count && std::find(words + first, words + count, before[w]) != words + count) return true;
                            }
                            return false;
                        };
                        for (auto& binding : patched->bindings) {
                            auto& words = binding.guestDescriptor;
                            const bool prefixed = (binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt) && patchPrefix(words.data(), words.size());
                            patchQuads(words.data(), words.size());
                            if (stale(words.data(), words.size(), prefixed)) usable = false;
                        }
                        for (auto& attribute : patched->vertexAttributes) {
                            auto& fields = attribute.resource.fields;
                            for (std::size_t q = 0; q < memoryQuads; ++q) {
                                const auto& [old, fresh] = quads[q];
                                if (fields[0] == old[0] && fields[1] == old[1] && fields[2] == old[2]) {
                                    fields[0] = fresh[0];
                                    quadUsed[q] = true;
                                    break;
                                }
                            }
                            if (stale(fields.data(), 1, false)) usable = false;
                        }
                        if (patched->pushConstants.size() % sizeof(std::uint32_t) != 0) usable = false;
                        if (usable && !patched->pushConstants.empty()) {
                            std::vector<std::uint32_t> push(patched->pushConstants.size() / sizeof(std::uint32_t));
                            std::memcpy(push.data(), patched->pushConstants.data(), patched->pushConstants.size());
                            const bool prefixed = patchPrefix(push.data(), push.size());
                            patchQuads(push.data(), push.size());
                            if (stale(push.data(), push.size(), prefixed)) usable = false;
                            std::memcpy(patched->pushConstants.data(), push.data(), patched->pushConstants.size());
                        }
                        for (std::size_t q = 0; q < memoryQuads; ++q) {
                            if (!quadUsed[q]) usable = false;
                        }
                        if (!usable) {
                            rebaseCounters().unusable.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        rebaseCounters().used.fetch_add(1, std::memory_order_relaxed);
                        rebased->compiled = std::move(patched);
                        rebased->dataPositions.clear();
                        regions.clear();
                        appendEntryRegions(*rebased, regions);
                        if (verifyRebasedHits()) {
                            probes[i] = rebased;
                            break;
                        }
                        candidates.emplace_back(std::move(rebased), std::move(regions));
                    }
                    continue;
                }
                auto& candidates = dataCandidates[i];
                for (const auto& variant : dataEntry->stages[i]) {
                    if ((i == 0 && variant->pushOffset != 0) || std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) { return candidate.first->pushOffset == variant->pushOffset; })) continue;
                    std::vector<ShaderRecompiler::MemoryRegion> regions;
                    appendEntryRegions(*variant, regions);
                    auto result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                    if (gateRetry() && (result == EntryOutcome::PublishMoved || result == EntryOutcome::PendingMoved)) result = validateVariant(programs[i].binary.codeAddress, submission.queue, *variant, regions, imagesFlushed, runsSynced, sampling);
                    if (result == EntryOutcome::Equal) candidates.emplace_back(variant, std::move(regions));
                }
            }
        } else if (rebasedDrawHits()) {
            if (dataEntry == nullptr) rebaseCounters().noEntry.fetch_add(1, std::memory_order_relaxed);
            else rebaseCounters().notVertex.fetch_add(1, std::memory_order_relaxed);
        }
        if (rebasedDrawHits()) reportRebase();
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
