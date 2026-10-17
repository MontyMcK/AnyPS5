#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace AgcDriver {
class VulkanDevice;
}

namespace AgcDriver::DriverDetail {

// The title's dword pattern-fill kernel (a 64-thread group storing pattern[tid & 3] at index tid of
// a V# with stride 4 through buffer_store_format_x, for tid < count): when every pattern word is
// the same and the stored range is fully known, the dispatch is executed as a native GPU fill
// (VulkanDevice::FillBuffer: vkCmdFillBuffer into the range's host import) instead of the recompiled
// kernel. The same bytes land in the same range, with the same pending-write note and write mark a
// buffer-writing dispatch makes; see NativePatternFill.cpp.
struct DwordPatternFill {
    std::uint64_t base = 0;
    std::size_t bytes = 0;
    std::uint32_t pattern = 0;
    // The end of the V#'s range (base + num_records * 4), the range the dispatch binds and notes as
    // its written buffer (write evidence only).
    std::uint64_t descriptorEnd = 0;
};

// Why a dispatch of the kernel keeps the shader. NotKernel: the code is not the kernel (never
// counted). Pattern: the four pattern words differ (a native fill takes one dword). TooLarge: more
// than 2^30 dwords. Uncommitted: part of the range is not committed writable guest memory (the
// kernel path handles sparse ranges; Driver::nativePatternFill checks it, the pure match does not).
enum class NativeFillRefusal : std::uint8_t { None, NotKernel, Indirect, UserData, GroupId, GroupShape, PartialGroups, Pattern, Descriptor, Bounds, Format, Unaligned, Empty, TooLarge, Uncommitted, Disabled, Count };
const char* NativeFillRefusalName(NativeFillRefusal refusal);

// Whether `code` (the program from its entry) is the dword fill kernel.
bool IsDwordFillKernel(std::span<const std::uint32_t> code);

// The recognition, pure: `packet` is the DISPATCH_DIRECT packet (header and four words), `pgmRsrc2`
// COMPUTE_PGM_RSRC2, `groupSize` the program's COMPUTE_NUM_THREAD_X/Y/Z. Fills `fill` when it
// answers None.
NativeFillRefusal MatchDwordPatternFill(std::span<const std::uint32_t> code, std::span<const std::uint32_t> userData, std::span<const std::uint32_t> packet, std::uint32_t pgmRsrc2, std::array<std::uint32_t, 3> groupSize, bool indirect, DwordPatternFill& fill);

// The device half, under GuestMemory::GpuMutex, after the packet's labels were recorded: pending
// image results over the range reach memory first (as for a dispatch's bound buffer), then the
// fill is recorded on the GPU (or stored by the CPU when the range has no host import), then the
// notes the dispatch path makes for this kernel (the depth metadata fill when `noteDepth`, the DCC key memo when
// `noteKeys`). True when the GPU stores the fill, false when the CPU stored it.
bool StoreDwordPatternFill(VulkanDevice& device, const DwordPatternFill& fill, bool noteDepth, bool noteKeys);

// APS5_VERIFY_NATIVE_FILL=1: a sampled fill runs both paths. The native fill runs first over a
// poisoned range and its bytes are kept (a window around the range); the range is poisoned again
// and the dispatch then runs the kernel; Driver::finishNativeFillVerify compares the fill range and
// checks neither path changed the bytes around it. The poison is transient: the range holds it
// from the second CPU store until this dispatch's kernel stores the range (the queue's draw pipeline
// drained first); another queue reading the range in that window would see it (verify mode only).
struct NativeFillVerify {
    bool active = false;
    std::uint64_t window = 0;
    DwordPatternFill fill;
    std::vector<std::byte> before;
    std::vector<std::byte> native;
};

}
