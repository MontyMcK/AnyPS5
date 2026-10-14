#ifndef CORE_SHADER_RECOMPILIER_BUFFERSTORECOVER_HPP
#define CORE_SHADER_RECOMPILIER_BUFFERSTORECOVER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ShaderRecompiler {

// An affine function of a compute invocation's ids as a program computes it in 32-bit integers:
// constant + sum over the axes of group[a] * WorkgroupId.a + local[a] * LocalInvocationId.a, the
// constant and every coefficient taken modulo 2^32. The program's 32-bit value is the form's value
// modulo 2^32; when the form's largest value over a dispatch (AffineMaximum, with every coefficient
// read as an unsigned number) stays below 2^32, it is the exact integer.
struct InvocationAffine {
    std::uint32_t constant = 0;
    std::array<std::uint32_t, 3> group{};
    std::array<std::uint32_t, 3> local{};

    bool operator==(const InvocationAffine& other) const = default;
};

// A condition the affine form of a store's address relied on (a mask, a bit-field extract or an OR
// taken as an addition): the form `value` never exceeds `limit` for any invocation of the dispatch.
struct InvocationBound {
    InvocationAffine value;
    std::uint32_t limit = 0;

    bool operator==(const InvocationBound& other) const = default;
};

// A raw buffer store (no format conversion) that every invocation of a compute dispatch makes
// through one buffer resource, as the recompiler's resource tracker proves it: the store's block
// runs on every path from the entry, under the launch exec mask, and its address is affine in the
// invocation's ids. Raw (`indexed` false): the store writes `bytes` bytes at byte offset
// offset + scalarOffset + immediate from the V#'s base (the VGPR offset, the SGPR offset and the
// instruction's immediate). Indexed: the store writes the whole record `offset` (its `bytes` equal
// the V#'s stride at evaluation; scalarOffset and immediate are zero).
struct BufferStoreCover {
    InvocationAffine offset;
    InvocationAffine scalarOffset;
    std::uint32_t immediate = 0;
    std::uint32_t bytes = 0;
    bool indexed = false;
    std::vector<InvocationBound> bounds;

    bool operator==(const BufferStoreCover& other) const = default;
};

// The shape of a compute dispatch a cover is evaluated for: workgroup counts and size per axis.
struct DispatchGrid {
    std::array<std::uint32_t, 3> groups{};
    std::array<std::uint32_t, 3> groupSize{};

    bool operator==(const DispatchGrid& other) const = default;
};

namespace Detail {

inline std::uint64_t SaturatingAdd(std::uint64_t left, std::uint64_t right) {
    return left > std::numeric_limits<std::uint64_t>::max() - right ? std::numeric_limits<std::uint64_t>::max() : left + right;
}

}

// The largest value of `form` over the invocations of `grid` as an unbounded integer (every
// coefficient and the constant read as unsigned), saturated at 2^64 - 1.
inline std::uint64_t AffineMaximum(const InvocationAffine& form, const DispatchGrid& grid) {
    std::uint64_t total = form.constant;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (grid.groups[axis] > 1u) total = Detail::SaturatingAdd(total, static_cast<std::uint64_t>(form.group[axis]) * (grid.groups[axis] - 1u));
        if (grid.groupSize[axis] > 1u) total = Detail::SaturatingAdd(total, static_cast<std::uint64_t>(form.local[axis]) * (grid.groupSize[axis] - 1u));
    }
    return total;
}

// The bytes, relative to the V#'s base address, that every dispatch of `grid` writes through the
// V# `words` by the store `cover` describes: one interval, a subset of what the dispatch writes
// with the emulated bounds checks applied (SpirvRuntimeBuffer.cpp), or nothing when no byte is
// proved (a bound does not hold, a value could wrap, an empty dispatch, a V# mode not handled: a
// typed or invalid descriptor, swizzling, ADD_TID, a zero format).
inline std::optional<std::pair<std::uint64_t, std::uint64_t>> CoveredBufferBytes(const BufferStoreCover& cover, const DispatchGrid& grid, std::span<const std::uint32_t, 4> words) {
    constexpr std::uint64_t Wrap = std::uint64_t{1} << 32u;
    if (cover.bytes == 0u) return std::nullopt;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (grid.groups[axis] == 0u || grid.groupSize[axis] == 0u) return std::nullopt;
    }
    for (const auto& bound : cover.bounds) {
        if (AffineMaximum(bound.value, grid) > bound.limit) return std::nullopt;
    }
    const std::uint32_t stride = (words[1] >> 16u) & 0x3fffu;
    const bool swizzle = ((words[1] >> 31u) & 1u) != 0u;
    const std::uint32_t records = words[2];
    const std::uint32_t format = (words[3] >> 12u) & 0x7fu;
    const bool addTid = ((words[3] >> 23u) & 1u) != 0u;
    const std::uint32_t mode = (words[3] >> 28u) & 3u;
    const std::uint32_t type = (words[3] >> 30u) & 3u;
    if (type != 0u || format == 0u || addTid || swizzle || records == 0u) return std::nullopt;
    // The bytes a driver binds for the V# (ShaderBufferResource::GetSize); nothing past them is written.
    const std::uint64_t bound = stride == 0u ? records : static_cast<std::uint64_t>(stride) * records;
    const auto offsetMaximum = AffineMaximum(cover.offset, grid);
    const auto scalarMaximum = AffineMaximum(cover.scalarOffset, grid);
    // The terms of offset + scalarOffset, each id counted once (64-bit coefficients: the 32-bit
    // sum of two coefficients would hide a wrap), ascending by coefficient.
    std::array<std::pair<std::uint64_t, std::uint32_t>, 6> terms{};
    std::size_t count = 0;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const std::uint64_t group = static_cast<std::uint64_t>(cover.offset.group[axis]) + cover.scalarOffset.group[axis];
        const std::uint64_t local = static_cast<std::uint64_t>(cover.offset.local[axis]) + cover.scalarOffset.local[axis];
        if (group != 0u && grid.groups[axis] > 1u) terms[count++] = {group, grid.groups[axis]};
        if (local != 0u && grid.groupSize[axis] > 1u) terms[count++] = {local, grid.groupSize[axis]};
    }
    std::sort(terms.begin(), terms.begin() + static_cast<std::ptrdiff_t>(count));
    // The invocations' stores form one run while each next term's step is at most the run so far
    // (copies of the run shifted by the step overlap or abut); past a gap, the run with the
    // remaining ids at zero is still written.
    std::uint64_t run = cover.indexed ? 1u : cover.bytes;
    for (std::size_t index = 0; index < count; ++index) {
        if (terms[index].first > run) break;
        run = Detail::SaturatingAdd(run, terms[index].first * (terms[index].second - 1u));
    }
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    if (!cover.indexed) {
        // Raw bounds (OOB_SELECT 3, records in bytes): a store is written when its last byte lies
        // below the records, with no 32-bit wrap of the offset or of the address.
        if (mode != 3u) return std::nullopt;
        if (offsetMaximum >= Wrap || scalarMaximum >= Wrap) return std::nullopt;
        const auto last = Detail::SaturatingAdd(Detail::SaturatingAdd(Detail::SaturatingAdd(offsetMaximum, scalarMaximum), cover.immediate), cover.bytes);
        if (last > Wrap) return std::nullopt;
        const std::uint64_t limit = std::min<std::uint64_t>(records, bound);
        if (limit < cover.bytes) return std::nullopt;
        begin = static_cast<std::uint64_t>(cover.offset.constant) + cover.scalarOffset.constant + cover.immediate;
        if (begin >= limit) return std::nullopt;
        // Every byte at most `limit - bytes` lies in a store wholly below the limit (the store
        // covering it starts at or before it). When the start and every step are multiples of the
        // store's size, the stores sit on that lattice with no gap, so the run reaches the end of
        // the last whole store below the limit.
        bool lattice = begin % cover.bytes == 0u;
        for (std::size_t index = 0; index < count; ++index) lattice = lattice && terms[index].first % cover.bytes == 0u;
        const auto inBounds = lattice ? begin + (limit - begin) / cover.bytes * cover.bytes : limit - cover.bytes + 1u;
        end = std::min(Detail::SaturatingAdd(begin, run), inBounds);
    } else {
        // Index bounds (OOB_SELECT 0 or 1): record i is written when i is below the records; the
        // whole record when the store is the stride wide at its start.
        if (mode != 0u && mode != 1u) return std::nullopt;
        if (stride == 0u || cover.bytes != stride || cover.immediate != 0u || cover.scalarOffset != InvocationAffine{}) return std::nullopt;
        if (offsetMaximum >= Wrap || bound > Wrap) return std::nullopt;
        const std::uint64_t first = cover.offset.constant;
        const std::uint64_t last = std::min<std::uint64_t>(Detail::SaturatingAdd(first, run), records);
        begin = first * stride;
        end = last * stride;
    }
    if (begin >= end) return std::nullopt;
    return std::pair<std::uint64_t, std::uint64_t>{begin, end};
}

}

#endif
