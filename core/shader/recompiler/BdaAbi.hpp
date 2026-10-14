#ifndef CORE_SHADER_RECOMPILER_BDAABI_HPP
#define CORE_SHADER_RECOMPILER_BDAABI_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ShaderRecompiler::BdaAbi {

inline constexpr std::uint32_t Version = 2;
inline constexpr std::uint32_t Read = 1;
inline constexpr std::uint32_t Write = 2;

// buckets: entries of the index after the ranges, or 0. The index is a BucketHeader and, per
// bucket of 1 << shift bytes from base, the range indices [low, high) a lookup in it searches.
struct Header {
    std::uint32_t version;
    std::uint32_t count;
    std::uint32_t entryBytes;
    std::uint32_t buckets;
};

struct BucketHeader {
    std::uint64_t base;
    std::uint32_t shift;
    std::uint32_t reserved;
};

struct Range {
    std::uint64_t begin;
    std::uint64_t end;
    std::uint64_t deviceAddress;
    std::uint32_t permissions;
    std::uint32_t reserved;
};

// The table's ranges ascend by begin and do not overlap (each begins at or after the previous
// one's end, and none ends before it begins), and the bucket index is built from them. The shader
// lookups rely on it: the hint, the lookup cache and the anchors (SpirvBdaRead.cpp) reuse a range
// for every address inside it, which is the range a search would select only in such a table.
inline bool RangesOrdered(const Range* ranges, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        if (ranges[index].end < ranges[index].begin) return false;
        if (index != 0 && ranges[index].begin < ranges[index - 1].end) return false;
    }
    return true;
}

enum class FaultState : std::uint32_t { Empty, Writing, Ready };
enum class FaultReason : std::uint32_t { Unmapped = 1, Permission, Overflow, InvalidTable, InvalidRectangle, LoopLimit, Unaligned, InvalidDescriptor };

struct Fault {
    FaultState state;
    FaultReason reason;
    std::uint64_t address;
    std::uint32_t bytes;
    std::uint32_t stage;
    std::uint32_t instruction;
    std::uint32_t reserved;
};

static_assert(std::is_standard_layout_v<Header> && std::is_trivially_copyable_v<Header> && sizeof(Header) == 16);
static_assert(std::is_standard_layout_v<Range> && std::is_trivially_copyable_v<Range> && sizeof(Range) == 32);
static_assert(std::is_standard_layout_v<BucketHeader> && sizeof(BucketHeader) == 16);
static_assert(offsetof(Range, deviceAddress) == 16 && offsetof(Range, permissions) == 24);
static_assert(std::is_standard_layout_v<Fault> && std::is_trivially_copyable_v<Fault> && sizeof(Fault) == 32);
static_assert(offsetof(Fault, address) == 8 && offsetof(Fault, instruction) == 24);

inline constexpr std::uint32_t WrittenPageShift = 12;
inline constexpr std::uint32_t WrittenPageSlots = 4096;
inline constexpr std::uint32_t WrittenPageProbes = 8;
inline constexpr std::uint32_t WrittenOverflowWord = sizeof(Fault) / sizeof(std::uint32_t);
inline constexpr std::uint32_t WrittenSlotsWord = WrittenOverflowWord + 1;
// APS5_BDA_CLOCK=1 (measurement): per subgroup, the shader-clock cycles spent in address lookups,
// the cycles of the whole invocation and the lookups made, each added in units of 64 cycles (calls
// as counted), summed by the driver as the buffers are recycled.
inline constexpr std::uint32_t ClockLookupWord = WrittenSlotsWord + WrittenPageSlots;
inline constexpr std::uint32_t ClockTotalWord = ClockLookupWord + 1;
inline constexpr std::uint32_t ClockCallsWord = ClockLookupWord + 2;
inline constexpr std::size_t FaultBufferBytes = (ClockLookupWord + 3) * sizeof(std::uint32_t);
static_assert((WrittenPageSlots & (WrittenPageSlots - 1)) == 0);

}

#endif
