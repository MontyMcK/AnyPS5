#include "prx/libSceAgcDriver/Graphics/include/CpuTimeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Capture/FrameCapture.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/ReplayProbe.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <functional>
#include <shared_mutex>

namespace AgcDriver::DriverDetail {

namespace {

bool orderedFills() {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_FILLS") == nullptr;
    return ordered;
}

bool keepsDrawPipeline(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x11: case 0x12: case 0x13: case 0x26: case 0x2a: case 0x2f: case 0x42: case 0x58: case 0x59: case 0x69: case 0x76: case 0x79: case 0x7a: case 0x81: return true;
        case 0x46: return packet.size() >= 2 && (packet[1] & 0x3fu) != 0x39u;
        case 0x15: return orderedFills();
        case 0x63: case 0x64: case 0x9f:
            return packet.size() >= 5 && !DrawPipeline::Queue0().Overlaps(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u), static_cast<std::size_t>(packet[4]) * 8);
        case 0x10: {
            const auto operation = (header >> 2u) & 0x3fu;
            return operation == 0 || operation == 0x09 || operation == 0x0b || operation == 0x0c || operation == 0x1a;
        }
        default: return false;
    }
}

std::uint32_t replayCut(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, bool predicated) {
    if (header == FlipPacketHeader) return ReplayProbe::CutFlip;
    if (header == RenderingWaitPacketHeader) return ReplayProbe::CutRenderingWait;
    if (predicated) return ReplayProbe::CutPredicated;
    if (Pm4::IndirectDrawOpcode(opcode)) return ReplayProbe::CutIndirect;
    return keepsDrawPipeline(packet, header, opcode) ? ReplayProbe::NoCut : opcode;
}

}

template <typename TWork>
void Driver::timed(double WorkerProfile::*bucket, TWork&& work) {
    static thread_local WorkerProfile profile;
    const auto begin = std::chrono::steady_clock::now();
    work();
    const auto end = std::chrono::steady_clock::now();
    profile.*bucket += std::chrono::duration<double, std::milli>(end - begin).count();
    static const bool report = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (report && end - profile.reported > std::chrono::seconds(10)) {
        profile.reported = end;
        AgcDriver::ProfilePrint_nid_no_patch( "[gpu] worker at %.0f s: dispatch %.1f s, draw %.1f s, wait %.1f s\n", std::chrono::duration<double>(end - profile.start).count(), profile.dispatchMs / 1000, profile.drawMs / 1000, profile.waitMs / 1000);
    }
}

void Driver::execute(const Submission& submission) {
    if (submission.queue == 0) DrawPipeline::PipeTrace(DrawPipeline::Trace::QueuePacket, submission.suspend ? 0x201u : 0x200u);
    const bool pipelined = submission.queue == 0 && DrawPipeline::Depth() != 0;
    static const bool orderedWaits = std::getenv("APS5_PIPELINE_ORDERED_LABELS") != nullptr;
    DrawPipeline::Active() = pipelined;
    struct PipelineDrain {
        bool pipelined;
        ~PipelineDrain() {
            if (!pipelined || std::uncaught_exceptions() == 0) return;
            try {
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
            } catch (...) {
            }
        }
    } pipelineDrain{pipelined};
    static const bool orderedCompletion = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && orderedWaits;
    if (pipelined && (submission.suspend || !orderedCompletion)) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    auto* submissionTiming = includeTimingSubmission(submission, true);
    if (submission.suspend) {
        PerformanceContext timingContext(submissionTiming);
        PerformanceTimer timing("Driver.Suspend");

        static const bool suspendDrain = std::getenv("APS5_SUSPEND_DRAIN") != nullptr;
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        ++suspendPoints;
        auto& costs = submissionCosts(submission.queue);

        if (suspendDrain || !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
            const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (const auto localDevice = device.Load()) {
                recordDeferredLabels(localDevice.get(), submission.queue);
                if (suspendDrain) localDevice->WaitIdle();
                else localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::Suspend);
            }
            ++costs.suspends;
            if (profile) costs.suspendNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        } else {
            ++costs.suspendsSkipped;
        }
        resetGraphics = true;
        return;
    }
    if (submission.queue == 0) DrawPipeline::PipeTrace(DrawPipeline::Trace::QueuePacket, 0x202u);
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (submission.queue == 0 && resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[submission.queue];
    }
    auto& queue = *state;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) AgcDriver::ProfilePrint_nid_no_patch("[gpu] %.1f execute serial=%llu queue=0x%x dwords=%zu\n", TraceMs(), static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size());

    static const long dumpQueue = [] { const char* text = std::getenv("APS5_DUMP_QUEUE"); return text ? std::strtol(text, nullptr, 16) : -1L; }();
    // APS5_DUMP_QUEUE_LIVE=1: dump only while the live flag `dump_queue` is set, at most 3 submissions
    // (to look at a scene the boot never reaches).
    static const bool dumpLive = std::getenv("APS5_DUMP_QUEUE_LIVE") != nullptr;
    if (static_cast<long>(submission.queue) == dumpQueue && (!dumpLive || Graphics::LiveFlag("dump_queue", false))) {

        static int dumped = 0;
        if (dumped++ < (dumpLive ? 3 : 40)) {
            std::string text = "[queue] submission " + std::to_string(submission.serial) + ":\n";
            for (std::size_t cursor = 0; cursor < submission.commands.size();) {
                const auto header = submission.commands[cursor];
                const auto count = Pm4::PacketWords(header);
                char line[200];
                int length = std::snprintf(line, sizeof(line), "[queue]   %s", Pm4::Name(header).c_str());
                for (std::size_t i = 1; i < count && i < 10 && length < 180; ++i) length += std::snprintf(line + length, sizeof(line) - length, " %08x", submission.commands[cursor + i]);
                text += line;
                text += "\n";
                cursor += count;
            }
            std::fputs(text.c_str(), stderr);
        }
    }
    PacketHistory recent{submission.commands};
    const bool probing = submission.queue == 0 && pipelined && ReplayProbe::Enabled();
    std::function<std::uint64_t()> probeDevice;
    if (probing) {
        probeDevice = [this] {
            const auto localDevice = device.Load();
            return localDevice != nullptr ? localDevice->Serial() : std::uint64_t{0};
        };
        ReplayProbe::Cut(ReplayProbe::CutSubmission);
    }

    static const bool profilePackets = std::getenv("APS5_PROFILE_DRAW") != nullptr;

    thread_local PacketProfile* packetProfileSlot = nullptr;
    auto& packetProfile = ShaderRecompiler::ThreadOwned(packetProfileSlot);
    ++packetProfile.submissions;

    static const bool epochEach = std::getenv("APS5_SUBMISSION_EPOCH_EACH") != nullptr;
    if (epochEach || submission.enqueuedAt == std::chrono::steady_clock::time_point{} || submission.enqueuedAt > lastEpochBump()) {
        if (!epochEach) noteReleasedTails(submission);
        bumpEpoch(&EpochBumps::submissions);
    }
    // Dispatch runs (Graphics::Recorder::DispatchRunsEnabled): a queue's run lasts while only
    // dispatches and SET_SH_REG writes follow each other in its stream. Any other packet (a wait, a
    // flush or cache action, a release, a data write, a NOP-wrapped custom packet) or a new
    // submission starts the next one.
    static std::array<std::atomic<std::uint32_t>, 256> dispatchRunEpochs{};
    auto& dispatchRunEpoch = dispatchRunEpochs[submission.queue & 0xffu];
    // Debug aid: APS5_DISPATCH_RUN_QUEUE=<hex queue> gives runs to that one queue only.
    static const long dispatchRunQueue = [] { const char* text = std::getenv("APS5_DISPATCH_RUN_QUEUE"); return text != nullptr ? std::strtol(text, nullptr, 16) : -1L; }();
    const bool dispatchRunsHere = dispatchRunQueue < 0 || static_cast<long>(submission.queue) == dispatchRunQueue;
    dispatchRunEpoch.fetch_add(1, std::memory_order_relaxed);
    for (std::size_t cursor = 0; cursor < submission.commands.size();) {
        if (packetEpoch()) bumpEpoch(&EpochBumps::packets);
        CheckFailure();
        const auto header = submission.commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(submission.commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (submission.queue == 0) DrawPipeline::PipeTrace(DrawPipeline::Trace::QueuePacket, header == FlipPacketHeader ? 0x1ffu : opcode);
        auto nextCursor = cursor + count;
        const bool predicated = Pm4::Predicated(header) && queue.predication.operation != 0;
        if (opcode == 0x37 && !predicated && dispatchRunsHere && Graphics::Recorder::DispatchRunStoresEnabled()) {
            // A WRITE_DATA keeps the run (Recorder::DispatchRunStoresEnabled): its store goes in
            // place inside it.
            Graphics::Recorder::SetDispatchRunToken(((static_cast<std::uint64_t>(submission.queue) + 1u) << 32u) | dispatchRunEpoch.load(std::memory_order_relaxed));
        } else if (opcode != 0x15 || predicated) {
            if (opcode != 0x76 || predicated) dispatchRunEpoch.fetch_add(1, std::memory_order_relaxed);
            Graphics::Recorder::SetDispatchRunToken(0);
        } else {
            Graphics::Recorder::SetDispatchRunToken(dispatchRunsHere ? ((static_cast<std::uint64_t>(submission.queue) + 1u) << 32u) | dispatchRunEpoch.load(std::memory_order_relaxed) : 0);
        }
        const ReplayProbe::Scope probeScope(probing ? ReplayProbe::Observe(queue, packet, Pm4::IndirectRegisterOpcode(opcode) && submission.registerLists.contains(cursor) ? std::span<const std::uint32_t>(submission.registerLists.at(cursor)) : std::span<const std::uint32_t>{}, replayCut(packet, header, opcode, predicated), probeDevice, submission.shaders.get()) : nullptr);
        if (pipelined && !predicated && header != FlipPacketHeader && header != RenderingWaitPacketHeader && (enqueueLabelPacket(packet, opcode, submission.queue) || enqueueDmaPacket(packet, opcode, submission.queue, queue))) {
            traceLabel(packet, submission.queue);
            recent.Record(cursor);
            cursor = nextCursor;
            continue;
        }
        if (pipelined && !predicated && (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && orderedWaits && DrawPipeline::Queue0().Busy()) {
            const std::uint64_t awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
            if (const auto value = DrawPipeline::Queue0().PendingLabel(awaited, Pm4::WaitAwaitedBytes(packet)); value.has_value() && Pm4::WaitComparesValue(packet, *value)) {
                ++waitOutcomes().fromRecorderSameQueue;
                recent.Record(cursor);
                cursor = nextCursor;
                continue;
            }
        }
        if (header == FlipPacketHeader) DrawPipeline::PipeTrace(DrawPipeline::Trace::QueueFlip, submission.queue);
        if (pipelined && DrawPipeline::Queue0().Busy() && !keepsDrawPipeline(packet, header, opcode)) {
            // A memory wait another queue satisfies (live `wait_overlap`, env APS5_WAIT_OVERLAP) needs
            // no empty pipeline: the queued work commits in order ahead of what follows the wait, so
            // the drain ends as soon as the awaited value is in memory or in the label table, as the
            // wait's own entry checks would take it. Otherwise the drain runs to the end as before.
            static const bool overlapDefault = std::getenv("APS5_WAIT_OVERLAP") != nullptr;
            const bool waitPacket = (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && !predicated && packet.size() >= 7;
            if (waitPacket && Graphics::LiveFlag("wait_overlap", overlapDefault)) {
                const std::uint64_t awaited = packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u);
                const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
                const auto received = submission.received;
                DrawPipeline::Queue0().DrainUnless(DrawPipeline::DrainReason::Packet, opcode, [&] {
                    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(awaited), awaitedBytes)) return false;
                    if (Pm4::WaitSatisfiedUnchecked(packet)) return true;
                    const auto hit = Graphics::Recorder::LookupLabel(awaited, awaitedBytes, received);
                    return hit.has_value() && !hit->late && Pm4::WaitComparesValue(packet, hit->value);
                });
            } else {
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, header == FlipPacketHeader ? 0xffu : opcode);
            }
        }
        if (APS5_ENABLE_TIMING_LOG && submission.queue == 0 && pendingFrameTiming == nullptr) includeTimingSubmission(submission, false);
        PerformanceContext timingContext(submission.queue == 0 ? pendingFrameTiming.get() : FrameTiming::Async());
        PerformanceTimer timing("Driver.Packet");
        std::shared_lock deviceUse(deviceReplacement, std::defer_lock);
        if (opcode != 0x3c && opcode != 0x93 && header != RenderingWaitPacketHeader && header != FlipPacketHeader) {
            const DrawPipeline::TraceScope lockTrace(DrawPipeline::Trace::QueueDeviceLock, DrawPipeline::Trace::QueueDeviceLockEnd);
            deviceUse.lock();
        }

        timing.Mark("device_lock");
        GuestMemory::SetCurrentPacket(header == FlipPacketHeader ? 0xffffu : opcode, submission.queue);
        CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
        if (predicated) {
            recordQueuedLabelsBeforeRead(submission.queue);
            if (!Pm4::PredicationPasses(queue)) {
                // Debug aid: APS5_TRACE_PREDICATION=1 names the first skipped predicated packets.
                static const bool tracePredication = std::getenv("APS5_TRACE_PREDICATION") != nullptr;
                static std::atomic<int> predicationSkips{0};
                if (tracePredication && predicationSkips.fetch_add(1) < 200) std::fprintf(stderr, "[predication] queue 0x%x skips opcode 0x%x at DWORD %zu: predicate 0x%llx operation %u execute-when-set %d\n", submission.queue, opcode, cursor, static_cast<unsigned long long>(queue.predication.address), queue.predication.operation, queue.predication.executeWhenSet ? 1 : 0);
                cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
                continue;
            }
        }
        if (opcode == 0x3f) {
            cursor = nextCursor;
            continue;
        }

        const auto flushStart = profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        {
            const DrawPipeline::TraceScope flushTrace(DrawPipeline::Trace::QueueFlush, DrawPipeline::Trace::QueueFlushEnd);
            flushBetweenPackets(submission.queue, header, opcode == 0x49 || opcode == 0x37);
        }
        timing.Mark("flush_between_packets");
        PacketTimer packetTimer{profilePackets, header == FlipPacketHeader ? 0xffffu : opcode, submission.queue, packetProfile, std::chrono::steady_clock::now()};

        if (profilePackets) {
            packetStartedAt() = packetTimer.start;
            pendingDispatchPhases() = {};
            pendingDrawPhases() = {};
        }
        const auto finishDispatchPacket = [&](bool indirect) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDispatchPhases();
            if (pending.phases) {
                pending.ms[PhaseEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
                addDriverPhases(submission.queue != 0 ? OtherQueues : indirect ? Queue0Indirect : Queue0Direct, pending.ms, pending.hit, pending.validated);
            }
            if (indirect) return;
            auto& row = packetProfile.dispatchOutcomes[static_cast<std::size_t>(pending.outcome)];
            ++row.first;
            row.second += std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
        };
        const auto finishDrawPacket = [&](bool drawn) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDrawPhases();
            std::array<double, DrawDriverPhaseCount> ms{};
            if (drawn && pending.phases) {
                ms = pending.ms;
                ms[DrawRowEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
            } else {
                ms[DrawRowSkipped] = std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
            }
            addDrawPhases(ms, drawn && pending.phases, pending.captures);
        };
        if (profilePackets) packetProfile.flushMs += std::chrono::duration<double, std::milli>(packetTimer.start - flushStart).count();

        bool wroteOnGpu = false, endOfPipeInterrupt = false, interruptDeferred = false, drawPacket = false, sampleDump = false;
        const bool drains = preparePacketMemory(submission, queue, packet, header, opcode, wroteOnGpu, endOfPipeInterrupt, interruptDeferred, drawPacket, sampleDump);
        traceLabel(packet, submission.queue);
        timing.Mark("prepare_memory");

        const bool waitPacket = opcode == 0x3c || opcode == 0x93 || header == RenderingWaitPacketHeader;
        struct Progress {
            Driver& driver;
            bool counted;
            ~Progress() {
                if (!counted) return;
                --driver.packetsInFlight;
                ++driver.packetsDone;
            }
        } progress{*this, !waitPacket};
        if (!waitPacket) ++packetsInFlight;
        recent.Record(cursor);
        if (header == RenderingWaitPacketHeader) {
            timed(&WorkerProfile::waitMs, [&] { submission.renderingWaits.at(cursor)->Wait(); });
        } else if (header == FlipPacketHeader) {
            CheckFailure();
            // APS5_PROFILE_FLIP: [flip-split] the queue thread's steps of a flip packet.
            static const bool flipSplit = std::getenv("APS5_PROFILE_FLIP") != nullptr;
            const auto flipStart = std::chrono::steady_clock::now();
            auto flipLap = flipStart;
            std::array<double, 5> flipMs{};
            const auto flipMark = [&](std::size_t step) {
                if (!flipSplit) return;
                const auto now = std::chrono::steady_clock::now();
                flipMs[step] = std::chrono::duration<double, std::milli>(now - flipLap).count();
                flipLap = now;
            };
            std::uint64_t batchesAtFlip = 0, unsignaledAtFlip = 0;
            if (!drains) {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                flipMark(0);
                const auto localDevice = device.Load();

                if (!recordLabelsForPacket(localDevice.get(), submission.queue) && localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::Flip);
                flipMark(1);
                if (localDevice != nullptr) localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
                flipMark(2);
            }
            struct FlipReport {
                bool enabled;
                std::array<double, 5>& ms;
                std::chrono::steady_clock::time_point start;
                ~FlipReport() {
                    if (!enabled) return;
                    static std::array<double, 6> sums{};
                    static int count = 0;
                    for (std::size_t i = 0; i < 5; ++i) sums[i] += ms[i];
                    sums[5] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                    if (++count < 100) return;
                    std::fprintf(stderr, "[flip-split] %d flips, ms each: gpu lock %.2f labels+submit %.2f flip batches %.2f timing %.2f gpu ready %.2f, total %.2f\n", count, sums[0] / count, sums[1] / count, sums[2] / count, sums[3] / count, sums[4] / count, sums[5] / count);
                    sums = {};
                    count = 0;
                }
            } flipReport{flipSplit, flipMs, flipStart};
            ++flipsCounted;
            if (Graphics::CpuTimeline::Active()) Graphics::CpuTimeline::Event("flip", flipStart, std::chrono::steady_clock::now(), submission.queue, batchesAtFlip);
            if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
            flipBatchesUnsignaled += unsignaledAtFlip;

            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            auto frame = APS5_ENABLE_TIMING_LOG ? std::move(pendingFrameTiming) : std::make_shared<FrameTiming>(++frameSerial);
            require(frame != nullptr, "flip has no frame timing");
            const auto now = FrameTiming::Clock::now();
            if (!APS5_ENABLE_TIMING_LOG) frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, APS5_ENABLE_TIMING_LOG ? submission.receivedAt : now, now);
            if (APS5_ENABLE_TIMING_LOG) frame->CollectBackground();
            frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
            CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(frameSerial), static_cast<unsigned long long>(submission.serial), cursor, static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
            flipMark(3);
            submission.flips.at(cursor)->GpuReady(frame);
            flipMark(4);
        } else if (opcode == 0x15) {
            timed(&WorkerProfile::dispatchMs, [&] { try { dispatch(queue, packet, submission); } catch (const std::exception& error) { reportSkip("dispatch", error.what()); } });
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(false);
            if (FrameReplay::WorkObserved()) FrameReplay::NoteWork(false);
        } else if (opcode == 0x16) {
            timed(&WorkerProfile::dispatchMs, [&] { try { dispatchIndirect(queue, packet, submission); } catch (const std::exception& error) { reportSkip("dispatch", error.what()); } });
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(true);
            if (FrameReplay::WorkObserved()) FrameReplay::NoteWork(false);
        } else if (opcode == 0x3c || opcode == 0x93) {
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            const auto waitStart = std::chrono::steady_clock::now();
            timed(&WorkerProfile::waitMs, [&] { waitMemory(packet, submission.queue, recent, submission.received, submission.heldAtSubmit.contains(cursor)); });
            if (traceGpu && std::chrono::steady_clock::now() - waitStart > std::chrono::milliseconds(200)) {

                for (std::size_t next = cursor + count, shown = 0; next < submission.commands.size() && shown < 48; ++shown) {
                    const auto nextHeader = submission.commands[next];
                    const auto nextCount = Pm4::PacketWords(nextHeader);
                    const auto nextPacket = std::span(submission.commands).subspan(next, nextCount);
                    std::fprintf(stderr, "[gpu]   then %s", Pm4::Name(nextHeader).c_str());
                    for (std::size_t i = 1; i < nextPacket.size() && i < 7; ++i) std::fprintf(stderr, " %08x", nextPacket[i]);
                    std::fprintf(stderr, "\n");
                    next += nextCount;
                }
            }
        } else if (drawPacket) {
            bool drawn = false;
            timed(&WorkerProfile::drawMs, [&] {
                static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
                static const bool profileDraws = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                const auto color = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
                const auto started = profileDraws ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto countSkip = [&](Graphics::DrawSkip kind) {
                    if (profileDraws) Graphics::CountDrawSkip(kind, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
                };

                try {
                    std::string rejected;
                    const auto verdict = draw(queue, packet, submission, rejected);
                    drawn = verdict == DrawVerdict::Drawn;
                    CaptureTrace::Log("draw submission=%llu queue=%x offset=%zu target=%llx mask=%x verdict=%d reason=%.256s", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e), static_cast<int>(verdict), rejected.c_str());
                    if (verdict == DrawVerdict::Rejected) {
                        countSkip(Graphics::DrawSkip::Prechecked);
                        reportSkip("draw", rejected);
                    } else if (verdict == DrawVerdict::Nothing) {
                        countSkip(Graphics::DrawSkip::Nothing);
                    } else if (traceDraws) {
                        std::fprintf(stderr, "[draw] target 0x%llx mask 0x%x ok\n",static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e));
                    }
                } catch (const std::exception& error) {
                    CaptureTrace::Log("draw-error submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, error.what());
                    countSkip(Graphics::DrawSkip::Thrown);
                    reportSkip("draw", error.what());
                }
            });
            finishDrawPacket(drawn);
            if (FrameReplay::WorkObserved()) {
                if (pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
                FrameReplay::NoteWork(true);
            }
        } else if (opcode == 0x22) {
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto condition = Pm4::ReadCondition(packet);
            if (condition == 0) nextCursor = submission.conditionalEnds.at(cursor);
            static const bool traceConditions = std::getenv("APS5_TRACE_PREDICATION") != nullptr;
            static std::atomic<int> conditionSkips{0};
            if (traceConditions && condition == 0 && conditionSkips.fetch_add(1) < 200) std::fprintf(stderr, "[cond-exec] queue 0x%x skips %zu dwords at DWORD %zu: condition at 0x%llx is 0\n", submission.queue, submission.conditionalEnds.at(cursor) - cursor - count, cursor, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)));
            if (traceGpu) AgcDriver::ProfilePrint_nid_no_patch("[gpu] %.1f queue 0x%x COND_EXEC at DWORD %zu reads 0x%x at 0x%llx: %s %zu dwords\n", TraceMs(), submission.queue, cursor, condition, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)), condition == 0 ? "skips" : "executes", submission.conditionalEnds.at(cursor) - cursor - count);
        } else if (sampleDump && !wroteOnGpu) {
            dumpSampleCounters(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u));
        } else if (opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (!wroteOnGpu) {
                if (Pm4::IndirectRegisterOpcode(opcode)) Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(cursor), queue);
                else Pm4::Execute(packet, queue);
                if (opcode == 0x49 || opcode == 0x37) {
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial);
                }
            }
            if (endOfPipeInterrupt && !interruptDeferred) AgcDriverDeliverEopInterrupt(submission.queue);
        }
        if (drawPacket || (sampleDump && wroteOnGpu)) Graphics::Recorder::CountRecordedWork();
        timing.Mark(header == RenderingWaitPacketHeader ? "rendering_wait" : opcode == 0x3c || opcode == 0x93 ? "memory_wait" : opcode == 0x15 || opcode == 0x16 ? "dispatch" : drawPacket ? "draw" : "execute");
        if (submission.queue == 0) timing.FinishPacket(submission.serial, cursor, opcode, count);
        else timing.Finish();
        cursor = nextCursor;
    }
    if (probing) ReplayProbe::Cut(ReplayProbe::CutSubmission);

    PerformanceContext endContext(submission.queue == 0 ? frameTiming() : FrameTiming::Async());
    PerformanceTimer endTiming("Driver.EndSubmission");
    static const bool submitAtEnd = std::getenv("APS5_SUBMIT_AT_END") != nullptr;
    if (pipelined && orderedCompletion && deferredLabels().labels.empty()) {
        if (submission.rewindTail == nullptr) {
            deferCompletion(submission);
            return;
        }
        if ((std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(submission.rewindTail - 1)).load(std::memory_order_acquire) & 0x80000000u) != 0) {
            DrawPipeline::Queue0().Enqueue([this] { submitOpenWork(); }, {}, 0, {}, true);
            executeRewindTail(submission);
            return;
        }
    }
    if (pipelined) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    if (!deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
        auto& costs = submissionCosts(submission.queue);
        if (!submitAtEnd && submission.rewindTail == nullptr && submission.queue == 0 && workerQueued() != nullptr && workerQueued()->load(std::memory_order_acquire) != 0) {
            ++costs.endSkipped;
            return;
        }
        ++costs.endSubmits;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = device.Load();

        recordDeferredLabels(localDevice.get(), submission.queue);
        if (localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0, Graphics::Recorder::SubmitReason::SubmissionEnd);
    }
    endTiming.Finish();
    if (submission.rewindTail != nullptr) executeRewindTail(submission);
}

}
