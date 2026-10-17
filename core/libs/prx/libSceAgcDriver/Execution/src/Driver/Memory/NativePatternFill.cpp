#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/NativePatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/NativeFillCounters.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace AgcDriver::DriverDetail {

namespace {

// v_lshl_add_u32 v4, s9, 6, v0 (tid = group * 64 + lane); v0..v3 = s4..s7 (the pattern);
// v_cmpx_gt_u32 s8, v4 (tid < count); a readfirstlane waterfall picking v6 = v[tid & 3];
// buffer_store_format_x v6, v4, s[0:3] idxen; s_endpgm. The same words noteDwordFillKernel matches.
constexpr std::array<std::uint32_t, 23> DwordFillKernel{0xbfa00001u, 0xd7460004u, 0x04010c09u, 0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u, 0x7da80808u, 0xbf88000du, 0x360a0883u, 0xbe84047eu, 0xbe86047eu, 0x7e100505u, 0x7da40a08u, 0xbefc0308u, 0x7e0c8700u, 0x8a867e06u, 0xbefe0406u, 0xbf85fff9u, 0xbefe0404u, 0xe0102000u, 0x80000604u, 0xbf810000u};

// The V# formats whose X component buffer_store_format_x stores as the raw 32-bit lane value
// (the recompiler's EmitFormatStoreComponent passes 32-bit UINT, SINT and FLOAT data through
// unchanged, as the hardware does): 32_UINT, 32_SINT, 32_FLOAT.
bool RawDwordFormat(std::uint32_t format) {
    return format == 20u || format == 21u || format == 22u;
}

bool KeysOf(std::uint32_t pattern, Graphics::DccKeys& keys) {
    if (pattern != (pattern & 0xffu) * 0x01010101u) return false;
    switch (pattern & 0xffu) {
        case 0x00: keys = Graphics::DccKeys::Clear0000; return true;
        case 0x40: keys = Graphics::DccKeys::Clear0001; return true;
        case 0x80: keys = Graphics::DccKeys::Clear1110; return true;
        case 0xc0: keys = Graphics::DccKeys::Clear1111; return true;
        case 0x20: keys = Graphics::DccKeys::ClearRegister; return true;
        case 0xff: keys = Graphics::DccKeys::Uncompressed; return true;
        default: return false;
    }
}

bool ProfileDraw() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

// The [draw] line of the counters, every 10 s (APS5_PROFILE_DRAW).
void ReportDrawLine() {
    if (!ProfileDraw()) return;
    static std::atomic<std::int64_t> last{0};
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto previous = last.load(std::memory_order_relaxed);
    if (previous == 0) {
        last.compare_exchange_strong(previous, now);
        return;
    }
    if (now - previous < 10000 || !last.compare_exchange_strong(previous, now)) return;
    const auto summary = Graphics::NativeFillSummary(1);
    if (!summary.empty()) AgcDriver::ProfilePrint_nid_no_patch("[draw] %s (10 s)\n", summary.c_str());
}

// Live `native_pattern_fill` (env APS5_NATIVE_PATTERN_FILL, default on; =0 starts it off).
bool NativeFillEnabled() {
    static const bool requested = [] {
        const char* text = std::getenv("APS5_NATIVE_PATTERN_FILL");
        return text == nullptr || std::strcmp(text, "0") != 0;
    }();
    return Graphics::LiveFlag("native_pattern_fill", requested);
}

bool VerifySample() {
    static const bool verify = [] {
        const char* text = std::getenv("APS5_VERIFY_NATIVE_FILL");
        return text != nullptr && std::strcmp(text, "0") != 0;
    }();
    if (!verify) return false;
    static std::atomic<std::uint64_t> seen{0};
    const auto index = seen.fetch_add(1, std::memory_order_relaxed);
    return index < 32 || index % 500 == 0;
}

std::vector<std::byte> PatternBytes(std::size_t bytes, std::uint32_t word) {
    std::vector<std::byte> block(bytes);
    for (std::size_t at = 0; at + 4 <= bytes; at += 4) std::memcpy(block.data() + at, &word, 4);
    return block;
}

}

const char* NativeFillRefusalName(NativeFillRefusal refusal) {
    static constexpr const char* names[static_cast<std::size_t>(NativeFillRefusal::Count)] = {"none", "not the kernel", "indirect", "user data", "group id", "group shape", "partial groups", "pattern", "descriptor", "bounds mode", "format", "unaligned", "empty", "too large", "uncommitted", "disabled"};
    const auto index = static_cast<std::size_t>(refusal);
    return index < static_cast<std::size_t>(NativeFillRefusal::Count) ? names[index] : "?";
}

bool IsDwordFillKernel(std::span<const std::uint32_t> code) {
    return code.size() >= DwordFillKernel.size() && std::equal(DwordFillKernel.begin(), DwordFillKernel.end(), code.begin());
}

NativeFillRefusal MatchDwordPatternFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, std::span<const std::uint32_t> packet, std::uint32_t pgmRsrc2, std::array<std::uint32_t, 3> groupSize, bool indirect, DwordPatternFill& fill) {
    if (!IsDwordFillKernel(code)) return NativeFillRefusal::NotKernel;
    // An indirect dispatch's group count is not known here.
    if (indirect) return NativeFillRefusal::Indirect;
    // s0..s8 are user data and s9 the group id: exactly nine user SGPRs, TGID_X_EN.
    if (userData.size() != 9) return NativeFillRefusal::UserData;
    if (((pgmRsrc2 >> 1u) & 0x1fu) != 9u || ((pgmRsrc2 >> 7u) & 1u) == 0) return NativeFillRefusal::GroupId;
    // tid = group * 64 + lane is the store index only for a 64x1x1 group in a 1-D grid.
    if (groupSize != std::array<std::uint32_t, 3>{64, 1, 1} || packet.size() < 5 || packet[2] != 1 || packet[3] != 1) return NativeFillRefusal::GroupShape;
    if ((packet[4] & 0x20u) != 0) return NativeFillRefusal::PartialGroups;
    // Element i holds pattern word i & 3: one dword for every element only when all four agree.
    if (userData[4] != userData[5] || userData[5] != userData[6] || userData[6] != userData[7]) return NativeFillRefusal::Pattern;
    const auto stride = (userData[1] >> 16u) & 0x3fffu;
    const bool swizzled = ((userData[1] >> 31u) & 1u) != 0;
    const bool addTid = ((userData[3] >> 23u) & 1u) != 0;
    const auto type = userData[3] >> 30u;
    const auto base = userData[0] | (static_cast<std::uint64_t>(userData[1] & 0xffffu) << 32u);
    if (stride != 4 || swizzled || addTid || type != 0 || base == 0) return NativeFillRefusal::Descriptor;
    // OOB_SELECT 0 (structured) and 1 (index only) store index i only while i < num_records; the
    // raw modes (2, 3) bound bytes, not indices.
    if (((userData[3] >> 28u) & 3u) > 1u) return NativeFillRefusal::Bounds;
    if (!RawDwordFormat((userData[3] >> 12u) & 0x7fu)) return NativeFillRefusal::Format;
    // vkCmdFillBuffer stores whole dwords at dword offsets; a misaligned V# base splits each element.
    if (base % 4u != 0) return NativeFillRefusal::Unaligned;
    const auto threads = static_cast<std::uint64_t>(packet[1]) * 64u;
    // The kernel's 32-bit tid wraps past 2^32 threads.
    if (threads > (1ull << 32u)) return NativeFillRefusal::GroupShape;
    const auto records = std::min<std::uint64_t>({threads, userData[8], userData[2]});
    if (records == 0) return NativeFillRefusal::Empty;
    // A fill of 4 GiB or more is not one this kernel makes in practice; the kernel keeps it.
    if (records > (1ull << 30u)) return NativeFillRefusal::TooLarge;
    fill.base = base;
    fill.bytes = static_cast<std::size_t>(records * 4u);
    fill.pattern = userData[4];
    fill.descriptorEnd = base + static_cast<std::uint64_t>(userData[2]) * 4u;
    return NativeFillRefusal::None;
}

bool StoreDwordPatternFill(VulkanDevice& device, const DwordPatternFill& fill, bool noteDepth, bool noteKeys) {
    // Pending image results over the range reach memory before the fill lands over them, as they do
    // for the range a dispatch binds as a written buffer (GuestBufferMemory's bound-region flush);
    // unit shadows publish their partial units over it, as for the fill HLE (BufferFill.cpp).
    Graphics::StorageTexture::FlushPending(fill.base, fill.bytes, nullptr, "buffer fill", Graphics::PublishScope::PartialUnits);
    const std::array<std::uint32_t, 4> words{fill.pattern, fill.pattern, fill.pattern, fill.pattern};
    // Ordered behind every recorded store over the range, noted as a pending write (a byte-uniform
    // one as a pending fill) and marked written: the notes a dispatch's direct write makes.
    const bool gpu = device.FillBuffer(fill.base, fill.bytes, words);
    if (!gpu) {
        // No host import over the range: the CPU stores it after the recorded work, as the fill
        // HLE does (the dispatch's copied buffer would have been written back the same way).
        device.WaitIdle();
        constexpr std::size_t Chunk = 1u << 20u;
        const auto block = PatternBytes(std::min(fill.bytes, Chunk), fill.pattern);
        for (std::size_t done = 0; done < fill.bytes; done += Chunk) GuestMemory::Write(fill.base + done, std::span<const std::byte>(block).first(std::min(Chunk, fill.bytes - done)), 4);
    }
    // The notes noteDwordFillKernel makes after the dispatch of this kernel.
    if (noteDepth) {
        Graphics::TraceDepthMetadataFill(fill.base, fill.bytes, fill.pattern, true, gpu ? "native dword fill" : "native dword fill (cpu)");
        Graphics::NoteDepthMetadataFill(fill.base, fill.bytes, fill.pattern);
    }
    Graphics::DccKeys keys = Graphics::DccKeys::Mixed;
    // The fill's own note covers exactly the range (FillBuffer's NotePendingFill), so the memo
    // entry takes the newest note over it.
    if (noteKeys && gpu && KeysOf(fill.pattern, keys)) Graphics::NoteKeysFillOnGpu(fill.base, fill.bytes, keys);
    return gpu;
}

bool Driver::nativePatternFill(QueueState& queue, std::uint32_t queueId, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> code, const std::vector<std::uint32_t>& userData, const ShaderRecompiler::ShaderComputeStageInfo& compute, bool indirect, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t programAddress, NativeFillVerify& verify) {
    if (!IsDwordFillKernel(code)) return false;
    DwordPatternFill fill;
    auto refusal = MatchDwordPatternFill(code, userData, packet, readRegister(queue.shader, 0x213), {compute.numThreads[0], compute.numThreads[1], compute.numThreads[2]}, indirect, fill);
    // Sparse or read-only pages inside the range: the kernel path stores only the committed ones.
    if (refusal == NativeFillRefusal::None && !GuestMemory::Accessible(reinterpret_cast<const void*>(fill.base), fill.bytes, true)) refusal = NativeFillRefusal::Uncommitted;
    if (refusal == NativeFillRefusal::None && !NativeFillEnabled()) refusal = NativeFillRefusal::Disabled;
    if (refusal != NativeFillRefusal::None) {
        Graphics::CountNativeFillRefusal(NativeFillRefusalName(refusal));
        ReportDrawLine();
        return false;
    }
    // The same switches gate the notes as on the dispatch path (noteDwordFillKernel).
    static const bool noteDepth = std::getenv("APS5_NO_DWORD_FILL_CLEAR") == nullptr;
    static const bool keysDefault = std::getenv("APS5_FILL_KERNEL_KEYS") != nullptr;
    const bool noteKeys = noteDepth && Graphics::LiveFlag("fill_kernel_keys", keysDefault);
    if (VerifySample()) {
        // Both paths, in order, on this thread: the pipeline drains first so nothing is in between.
        if (DrawPipeline::Active()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        auto begin = fill.base >= 64u ? fill.base - 64u : fill.base;
        auto end = fill.base + fill.bytes + 64u;
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), static_cast<std::size_t>(end - begin))) {
            begin = fill.base;
            end = fill.base + fill.bytes;
        }
        verify.fill = fill;
        verify.window = begin;
        verify.before.resize(static_cast<std::size_t>(end - begin));
        GuestMemory::Read(begin, verify.before);
        const auto poison = PatternBytes(fill.bytes, fill.pattern ^ 0xa5a5a5a5u);
        GuestMemory::Write(fill.base, poison, 4);
        recordLabelsForPacket(localDevice.get(), queueId);
        static_cast<void>(StoreDwordPatternFill(*localDevice, fill, noteDepth, noteKeys));
        verify.native.resize(verify.before.size());
        GuestMemory::Read(begin, verify.native);
        GuestMemory::Write(fill.base, poison, 4);
        verify.active = true;
        return false;
    }
    const auto store = [this, queueId, fill, localDevice, programAddress, noteKeys] {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        // The packet's queued labels go first, as the dispatch records them under its lock.
        recordLabelsForPacket(localDevice.get(), queueId);
        // The write evidence noteWrittenBuffers keeps for the dispatch: its program over the V#'s range.
        if (writeEvidenceEnabled() || traceCapSync()) {
            std::lock_guard lock(writtenBuffersMutex);
            writtenBuffers.push_back({programAddress, fill.base, fill.descriptorEnd, ++writtenBufferSerial, queueId, false});
            while (writtenBuffers.size() > WrittenBufferRing) writtenBuffers.pop_front();
        }
        const bool gpu = StoreDwordPatternFill(*localDevice, fill, noteDepth, noteKeys);
        Graphics::CountNativeFill(fill.bytes, !gpu);
    };
    // Ordered on the draw pipeline like the fill HLE (BufferFill.cpp): a queue-0 fill while the
    // pipeline runs becomes its next job, behind the draws enqueued before it.
    static const bool orderedFills = std::getenv("APS5_PIPELINE_DRAIN_FILLS") == nullptr;
    if (orderedFills && queueId == 0 && DrawPipeline::Active()) {
        DrawPipeline::Queue0().Enqueue([store, queueId] {
            GuestMemory::SetCurrentPacket(0x15, queueId);
            store();
        }, {{fill.base, fill.base + fill.bytes}});
    } else {
        if (DrawPipeline::Active()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        store();
    }
    ReportDrawLine();
    return true;
}

void Driver::finishNativeFillVerify(NativeFillVerify& verify) {
    if (!verify.active) return;
    verify.active = false;
    std::vector<std::byte> kernel(verify.before.size());
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        GuestMemory::Read(verify.window, kernel);
    }
    const auto& fill = verify.fill;
    const auto offset = static_cast<std::size_t>(fill.base - verify.window);
    const auto expected = PatternBytes(fill.bytes, fill.pattern);
    const auto outsideSame = [&](const std::vector<std::byte>& bytes) {
        return std::equal(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(offset), verify.before.begin()) && std::equal(bytes.begin() + static_cast<std::ptrdiff_t>(offset + fill.bytes), bytes.end(), verify.before.begin() + static_cast<std::ptrdiff_t>(offset + fill.bytes));
    };
    // The fill range: the two paths' bytes, and the native bytes against the pattern. Around it
    // (up to 64 bytes each side): neither path may change a byte (another writer of those bytes
    // in between would show here too, so a failure names which check failed).
    const bool nativeRange = std::memcmp(verify.native.data() + offset, expected.data(), fill.bytes) == 0;
    const bool same = std::memcmp(verify.native.data() + offset, kernel.data() + offset, fill.bytes) == 0;
    const bool nativeOutside = outsideSame(verify.native);
    const bool kernelOutside = outsideSame(kernel);
    const bool matched = same && nativeRange && nativeOutside && kernelOutside;
    Graphics::CountNativeFillVerify(matched);
    static std::atomic<int> reported{0};
    if (!matched || reported.fetch_add(1, std::memory_order_relaxed) < 32) {
        std::size_t first = fill.bytes;
        for (std::size_t i = 0; i < fill.bytes; ++i) {
            if (verify.native[offset + i] != kernel[offset + i]) {
                first = i;
                break;
            }
        }
        std::fprintf(stderr, "[native-fill] verify 0x%llx+0x%zx pattern %08x: native range %s, kernel range %s, outside the range native %s kernel %s%s\n", static_cast<unsigned long long>(fill.base), fill.bytes, fill.pattern, nativeRange ? "ok" : "WRONG", same ? "identical" : "DIFFERS", nativeOutside ? "untouched" : "CHANGED", kernelOutside ? "untouched" : "CHANGED", first < fill.bytes ? (" (first difference at range byte " + std::to_string(first) + ")").c_str() : "");
    }
}

}
