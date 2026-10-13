#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

SampledReadScope::SampledReadScope(std::atomic<std::uint64_t>& reads) : previous(Driver::sampledRead()) {
    Driver::sampledRead() = Driver::writeEvidenceEnabled() && reads.fetch_add(1, std::memory_order_relaxed) % Driver::writeEvidenceSampleEvery() == 0;
}

SampledReadScope::~SampledReadScope() { Driver::sampledRead() = previous; }

void PendingView::Load() {
    if (Driver::validateLegacy()) return;
    const auto current = Graphics::Recorder::PublishGeneration();
    if (loaded && current == generation) return;
    snapshot = Graphics::Recorder::PendingWriteSnapshot();
    generation = current;
    loaded = true;
}

bool PendingView::Overlaps(std::uint64_t address, std::size_t bytes) const {
    return Driver::validateLegacy() ? Graphics::Recorder::SnapshotWriteOverlaps(address, bytes) : Graphics::Recorder::SnapshotOverlaps(snapshot.get(), address, bytes);
}

bool& Driver::sampledRead() {
    static thread_local bool sampled = false;
    return sampled;
}

bool Driver::writeEvidenceEnabled() {
    static const bool enabled = std::getenv("APS5_NO_WRITE_EVIDENCE") == nullptr;
    return enabled;
}

bool Driver::writerKeyedEvidence() {
    static const bool keyed = std::getenv("APS5_NO_WRITER_KEYED_EVIDENCE") == nullptr;
    return keyed;
}

bool Driver::foreignWriters() {
    static const bool noted = std::getenv("APS5_NO_FOREIGN_WRITERS") == nullptr;
    return noted;
}

bool Driver::observationGuard() {
    static const bool guard = std::getenv("APS5_NO_OBSERVATION_GUARD") == nullptr;
    return guard;
}

bool Driver::siteSampling() {
    static const bool perSite = std::getenv("APS5_NO_SITE_SAMPLING") == nullptr;
    return perSite;
}

std::uint64_t Driver::hookWaits() {
    return Graphics::Recorder::ThreadHookWaits();
}

ShaderMemory::HookWaitCounter Driver::hookWaitCounter() {
    return observationGuard() ? &hookWaits : nullptr;
}

bool Driver::validateSkipEnabled() {
    static const bool enabled = std::getenv("APS5_NO_VALIDATE_SKIP") == nullptr;
    return enabled;
}

bool Driver::validateLegacy() {
    static const bool legacy = std::getenv("APS5_VALIDATE_LEGACY") != nullptr;
    return legacy;
}

bool Driver::gateRetry() {
    static const bool retry = std::getenv("APS5_NO_GATE_RETRY") == nullptr && !validateLegacy();
    return retry;
}

bool Driver::validateSkipVerify() {
    static const bool verify = std::getenv("APS5_VALIDATE_SYNC_SKIP") != nullptr;
    return verify;
}

std::uint32_t Driver::writeEvidenceAfter() {
    static const std::uint32_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_AFTER"); return text ? static_cast<std::uint32_t>(std::atoi(text)) : 4u; }();
    return value;
}

std::uint32_t Driver::writeEvidenceSampleEvery() {
    static const std::uint32_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_SAMPLE"); const auto parsed = text ? std::atoi(text) : 16; return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 16u; }();
    return value;
}

std::uint64_t Driver::writeEvidenceMaxBytes() {
    static const std::uint64_t value = [] { const char* text = std::getenv("APS5_WRITE_EVIDENCE_MAX_KIB"); return (text ? std::strtoull(text, nullptr, 10) : 16ull) << 10u; }();
    return value;
}

bool Driver::traceCapSync() {
    static const bool trace = std::getenv("APS5_TRACE_CAPSYNC") != nullptr;
    return trace;
}

void Driver::observeDword(std::uint64_t address, bool unchanged) {
    if (!writeEvidenceEnabled()) return;
    (unchanged ? observedUnchanged : observedChanged).fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(writtenBuffersMutex);
    if (dwordEvidence.size() >= DwordEvidenceEntries && !dwordEvidence.contains(address)) dwordEvidence.clear();
    auto& evidence = dwordEvidence[address];
    if (writerKeyedEvidence()) {
        const auto writer = newestWriterLocked(address, address + 4);
        const std::uint64_t program = writer ? writer->program : 0, begin = writer ? writer->begin : 0, end = writer ? writer->end : 0;
        if (program != evidence.program || begin != evidence.begin || end != evidence.end) evidence = DwordEvidence{0, 0, program, begin, end};
    }
    if (!unchanged) {
        evidence.streak = 0;
        ++evidence.changed;
    } else if (evidence.streak < (1u << 30u)) {
        ++evidence.streak;
    }
}

void Driver::observeRange(std::uint64_t address, std::span<const std::byte> before) {
    if (before.empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), before.size())) return;
    const auto* now = reinterpret_cast<const std::byte*>(address);
    for (std::size_t offset = 0; offset + 4 <= before.size(); offset += 4) {
        observeDword(address + offset, std::memcmp(before.data() + offset, now + offset, 4) == 0);
    }
}

void Driver::observePendingWrite(std::uint64_t address, bool unchanged) {
    Get().observeDword(address, unchanged);
}

bool Driver::knownValueCurrent(const WrittenBuffer& writer) {
    const auto bytes = static_cast<std::size_t>(writer.end - writer.begin);
    GuestMemory::CollectWrites(writer.begin, bytes);
    return GuestMemory::UnchangedSince(writer.begin, bytes, writer.generation);
}

ShaderMemory::PendingWrite Driver::classifyPendingWrite(std::uint64_t address, std::size_t bytes, std::uint64_t ValidateCounters::*& reason, const PendingView& pending, std::span<std::byte> known) {
    using Policy = ShaderMemory::PendingWrite;
    if (!pending.Overlaps(address, bytes)) return Policy::None;

    if (Graphics::AnyShadowedOverlaps(address, bytes)) {
        reason = &ValidateCounters::syncedShadow;
        return Policy::Sync;
    }
    // Live `label_known` (env APS5_LABEL_KNOWN): a dword whose pending writer is a recorded label
    // store (the title builds V#s and BVH headers with WRITE_DATA right before the dispatches that
    // read them) takes the store's value instead of a GPU round trip (Recorder::CurrentLabelDword).
    // Live `label_known_verify` waits anyway and counts mismatches ([capture] known-value words).
    static const bool labelKnownDefault = std::getenv("APS5_LABEL_KNOWN") != nullptr;
    if (bytes == sizeof(std::uint32_t) && known.size() == bytes && Graphics::LiveFlag("label_known", labelKnownDefault)) {
        std::uint32_t value = 0;
        if (Graphics::Recorder::CurrentLabelDword(address, value)) {
            std::memcpy(known.data(), &value, sizeof(value));
            knownValueReads.fetch_add(1, std::memory_order_relaxed);
            return Graphics::LiveFlag("label_known_verify", false) ? Policy::VerifyKnownValue : Policy::KnownValue;
        }
    }
    reason = &ValidateCounters::syncedOff;
    if (!writeEvidenceEnabled()) return Policy::Sync;
    const auto writer = newestWriter(address, address + bytes);
    if (!writer) {
        reason = &ValidateCounters::syncedNoWriter;
        return Policy::Sync;
    }
    if (writer->program == 0) {
        reason = &ValidateCounters::syncedForeign;
        return Policy::Sync;
    }
    if (writer->end - writer->begin > writeEvidenceMaxBytes() || writer->begin < 4 || pending.Overlaps(writer->begin - 4, 4) || pending.Overlaps(writer->end, 4)) {
        reason = &ValidateCounters::syncedLargeRange;
        return Policy::Sync;
    }
    const auto first = address & ~3ull;
    const auto limit = address + bytes;
    if (Graphics::Recorder::WideLabelIn(first, static_cast<std::size_t>(limit - first))) {
        reason = &ValidateCounters::syncedLabel;
        return Policy::Sync;
    }
    for (std::uint64_t dword = first; dword < limit; dword += 4) {
        if (Graphics::Recorder::LookupLabelValue(dword, 4, 0).has_value()) {
            reason = &ValidateCounters::syncedLabel;
            return Policy::Sync;
        }
    }
    if (writer->value && writer->begin <= address && limit <= writer->end && (known.empty() || known.size() == bytes) && knownValueCurrent(*writer)) {
        if (!known.empty()) {
            std::memcpy(known.data(), writer->value->data() + (address - writer->begin), bytes);
            knownValueReads.fetch_add(1, std::memory_order_relaxed);
        }
        return knownValueVerify() ? Policy::VerifyKnownValue : Policy::KnownValue;
    }
    {
        std::lock_guard lock(writtenBuffersMutex);
        for (std::uint64_t dword = first; dword < limit; dword += 4) {
            const auto found = dwordEvidence.find(dword);
            if (found == dwordEvidence.end() || found->second.streak < writeEvidenceAfter()) {
                reason = &ValidateCounters::syncedEvidence;
                return Policy::Sync;
            }
            if (writerKeyedEvidence() && (found->second.program != writer->program || found->second.begin != writer->begin || found->second.end != writer->end)) {
                reason = &ValidateCounters::syncedWriterChanged;
                return Policy::Sync;
            }
        }
    }
    if (sampledRead()) {
        reason = &ValidateCounters::syncedSample;
        return Policy::Sync;
    }
    return validateSkipVerify() ? Policy::VerifyRaw : Policy::Raw;
}

ShaderMemory::PendingWrite Driver::queryPendingWrite(std::uint64_t address, std::size_t bytes, std::span<std::byte> known) {
    if (DrawPipeline::Active()) {
        auto& pipeline = DrawPipeline::Queue0();
        if (pipeline.Busy() && pipeline.Overlaps(address, bytes)) {
            // A page probe only picks word-by-word reads: the words that meet a queued write drain.
            static const bool probeDrains = std::getenv("APS5_PROBE_DRAINS") != nullptr;
            if (!probeDrains && known.empty() && bytes > sizeof(std::uint32_t)) return ShaderMemory::PendingWrite::Sync;
            static const bool trace = std::getenv("APS5_TRACE_CAPTURE_DRAINS") != nullptr;
            static std::atomic<int> traced{0};
            if (trace && traced.fetch_add(1, std::memory_order_relaxed) < 200) {
                const auto [begin, limit] = pipeline.OverlapRange(address, bytes);
                std::fprintf(stderr, "[capture-drain] read 0x%llx+0x%zx overlaps queued write 0x%llx+0x%llx\n", static_cast<unsigned long long>(address), bytes, static_cast<unsigned long long>(begin), static_cast<unsigned long long>(limit - begin));
            }
            pipeline.WaitEarlierPreparations();
            pipeline.Drain(DrawPipeline::DrainReason::Capture);
        }
    }
    std::uint64_t ValidateCounters::*reason = nullptr;
    PendingView pending;
    pending.Load();
    const auto policy = Get().classifyPendingWrite(address, bytes, reason, pending, known);
    countCaptureQuery(policy, reason, bytes);
    return policy;
}

void Driver::countCaptureQuery(ShaderMemory::PendingWrite policy, std::uint64_t ValidateCounters::*reason, std::size_t bytes) {
    // APS5_PROFILE_DRAW: why capture reads over pending GPU writes waited ([capsync], 10 s): by the
    // first classifyPendingWrite rule that refused a skip, next to the reads it answered without a
    // wait (known values, raw on evidence). Word-sized reads and larger ones are counted apart.
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || policy == ShaderMemory::PendingWrite::None) return;
    enum Slot : std::size_t { Shadow, Off, NoWriter, Foreign, LargeRange, Label, Evidence, WriterChanged, Sample, Image, Other, Known, Raw, Count };
    static std::array<std::atomic<std::uint64_t>, Count * 2> counts{};
    static std::atomic<std::int64_t> last{0};
    std::size_t slot = Other;
    using Policy = ShaderMemory::PendingWrite;
    if (policy == Policy::KnownValue || policy == Policy::VerifyKnownValue) slot = Known;
    else if (policy == Policy::Raw || policy == Policy::VerifyRaw) slot = Raw;
    else if (reason == &ValidateCounters::syncedShadow) slot = Shadow;
    else if (reason == &ValidateCounters::syncedOff) slot = Off;
    else if (reason == &ValidateCounters::syncedNoWriter) slot = NoWriter;
    else if (reason == &ValidateCounters::syncedForeign) slot = Foreign;
    else if (reason == &ValidateCounters::syncedLargeRange) slot = LargeRange;
    else if (reason == &ValidateCounters::syncedLabel) slot = Label;
    else if (reason == &ValidateCounters::syncedEvidence) slot = Evidence;
    else if (reason == &ValidateCounters::syncedWriterChanged) slot = WriterChanged;
    else if (reason == &ValidateCounters::syncedSample) slot = Sample;
    else if (reason == &ValidateCounters::syncedImage) slot = Image;
    counts[slot + (bytes <= sizeof(std::uint32_t) ? 0 : Count)].fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (now - previous < 10 || !last.compare_exchange_strong(previous, now, std::memory_order_relaxed)) return;
    std::array<unsigned long long, Count * 2> taken{};
    for (std::size_t i = 0; i < taken.size(); ++i) taken[i] = counts[i].exchange(0, std::memory_order_relaxed);
    std::fprintf(stderr, "[capsync] capture reads over pending GPU writes (10 s; dword/larger): sync by reason: shadow %llu/%llu, label_known off %llu/%llu, no writer %llu/%llu, foreign writer %llu/%llu, large range %llu/%llu, label %llu/%llu, evidence short %llu/%llu, writer changed %llu/%llu, sampled %llu/%llu, image %llu/%llu, other %llu/%llu; no wait: known value %llu/%llu, raw %llu/%llu\n",
                 taken[Shadow], taken[Count + Shadow], taken[Off], taken[Count + Off], taken[NoWriter], taken[Count + NoWriter], taken[Foreign], taken[Count + Foreign], taken[LargeRange], taken[Count + LargeRange], taken[Label], taken[Count + Label], taken[Evidence], taken[Count + Evidence], taken[WriterChanged], taken[Count + WriterChanged], taken[Sample], taken[Count + Sample], taken[Image], taken[Count + Image], taken[Other], taken[Count + Other], taken[Known], taken[Count + Known], taken[Raw], taken[Count + Raw]);
}

}
