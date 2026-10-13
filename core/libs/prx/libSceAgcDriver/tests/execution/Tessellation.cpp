#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "Recompiler.hpp"
#include "SceShaders.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 256;
constexpr std::uint32_t Height = 80;
constexpr std::array<std::uint8_t, 4> Background{16, 24, 40, 255};
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

// The local program writes each control point's (x, y) from its index bits and the patch level
// twice into 32 bytes of LDS, then jumps into the hull half (s_swappc_b64 null, s[6:7]).
alignas(256) constexpr std::array<std::uint32_t, 29> LocalCode{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x161406a0, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000070a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

alignas(256) constexpr std::array<std::uint32_t, 49> HullCode{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xbefe0414,
    0xe0781000, 0x0403180a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 51> HullRestoredExecSaved{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xbefe0414,
    0xbe94047e, 0xbefe0414, 0xe0781000, 0x0403180a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000,
    0xe0741000, 0x0202070b, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 30> DomainCode{
    0xf8000941, 0x00000000, 0x34020e84, 0x16020283, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01,
    0xe0341020, 0x04030e01, 0xbf8c3f70, 0x08040af2, 0x08040d02, 0x10201505, 0x3e201906, 0x3e201d02,
    0x10221705, 0x3e221b06, 0x3e221f02, 0xd5510012, 0x040a0d05, 0x102424ff, 0x40400000, 0x7e260280,
    0x7e2802f2, 0xf80008cf, 0x14131110, 0xf800020f, 0x14131214, 0xbf810000,
};

// The quad hull: the triangle hull with 4 input and output control points (LDS patch 0x80, ring
// patch 0x40) and 24 bytes of factors a patch (buffer_store_dwordx4 + dwordx2 at R * 24).
alignas(256) constexpr std::array<std::uint32_t, 51> HullQuad{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000080, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0080, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x16142498, 0xbefe0414,
    0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000,
    0xe0741000, 0x0202070b, 0xbf810000,
};

// The quad hull reading its control point at an index computed from data (v_min_u32 v0, 3, v20;
// v_lshlrev_b32 v0, 5, v0), not affine in the invocation.
alignas(256) constexpr std::array<std::uint32_t, 52> HullQuadDynamicIndex{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x26002883, 0x34000085, 0xd5430015, 0x040224ff, 0x00000080,
    0xdbfc0000, 0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000,
    0x02020216, 0x7da82881, 0xbe94047e, 0xb06a0080, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013,
    0xd8dc1008, 0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x16142498,
    0xbefe0414, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690, 0x4c1616ff,
    0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// HullQuadDynamicIndex scaling its computed index by 16 (v_lshlrev_b32 v0, 4, v0), half the 32-byte
// LDS slot: the computed term is not a whole number of control points.
alignas(256) constexpr std::array<std::uint32_t, 52> HullQuadHalfSlotIndex{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x26002883, 0x34000084, 0xd5430015, 0x040224ff, 0x00000080,
    0xdbfc0000, 0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000,
    0x02020216, 0x7da82881, 0xbe94047e, 0xb06a0080, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013,
    0xd8dc1008, 0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x16142498,
    0xbefe0414, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690, 0x4c1616ff,
    0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// The quad hull taking the patch level from control point 2's ring record (written by invocation 2:
// v_lshlrev_b32 v29, 6, v18; buffer_load_dword v6, v29, s[8:11], soff=s2 offset=40 offen).
alignas(256) constexpr std::array<std::uint32_t, 52> HullQuadReadsControlPoints{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000080, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a0080, 0x162602f9, 0x0086066a, 0x343a2486, 0xe0301028, 0x0202061d,
    0xd8dc1008, 0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x16142498,
    0xbefe0414, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690, 0x4c1616ff,
    0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// The quad hull shaped like a title's: the lane guard branches (s_cbranch_execz); v1 is overwritten
// under that exec with a list of input control points, (1 << invocation) | 1; a loop takes its lowest
// bit (v_ffbl_b32), clears it and reads that control point's LDS at R * 0x80 + index * 32 until the
// list is empty (v_cmpx_ne_u32, s_cbranch_execnz, exec saved and restored through s[22:23]), so the
// last read is the invocation's own control point; the invocation-0 region branches too and takes
// the patch level from control point 2's ring record, as HullQuadReadsControlPoints.
alignas(256) constexpr std::array<std::uint32_t, 61> HullQuadPatchList{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xbf880034,
    0xd5480014, 0x02151101, 0x362402ff, 0x000000ff, 0x7e020281, 0x34020314, 0x38020281, 0xbe96047e,
    0x7e2e7501, 0x4a3002c1, 0x36020318, 0x34322e85, 0xd5430015, 0x046624ff, 0x00000080, 0xdbfc0000,
    0x02000015, 0xbf8cc07f, 0x7daa0280, 0xbf89fff4, 0xbefe0416, 0x162c2890, 0xd5430016, 0x045a24ff,
    0x00000040, 0xe0781000, 0x02020216, 0x7da82881, 0xbf880017, 0x162624ff, 0x00000080, 0x343a2486,
    0xe0301028, 0x0202061d, 0xd8dc1008, 0x07000013, 0xbf8c0070, 0x7e300306, 0x7e320306, 0x7e340306,
    0x7e360306, 0x16142498, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690,
    0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// The local program with a 24-byte LDS slot per control point (v_mul_u32_u24 v10, 24, v3): its
// second vec4 is half used, so a byte offset and its vec4 component differ past the first slot.
alignas(256) constexpr std::array<std::uint32_t, 29> LocalCodeStride24{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x16140698, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000070a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

// The quad hull over 8 input control points of 24 bytes (LDS patch 0xc0): control point i reads
// input i + 1 (ds_read_b128 v[2:5], v21 offset:24 at R * 0xc0 + i * 24), a constant past its own
// slot, and takes the patch level from input 0.
alignas(256) constexpr std::array<std::uint32_t, 51> HullQuadNextInput{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x16002898, 0xd5430015, 0x040224ff, 0x000000c0, 0xdbfc0018,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000040, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xbe94047e, 0xb06a00c0, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008,
    0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x16142498, 0xbefe0414,
    0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000,
    0xe0741000, 0x0202070b, 0xbf810000,
};

// HullQuadPatchList with its control-point store moved into the invocation-0 region, ahead of the
// read-back: no position outside that region's branch on a lane mask is free of a control-point
// write, so the read-back barrier has no place in uniform control flow.
alignas(256) constexpr std::array<std::uint32_t, 61> HullReadBackInBranch{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xbf880034,
    0xd5480014, 0x02151101, 0x362402ff, 0x000000ff, 0x7e020281, 0x34020314, 0x38020281, 0xbe96047e,
    0x7e2e7501, 0x4a3002c1, 0x36020318, 0x34322e85, 0xd5430015, 0x046624ff, 0x00000080, 0xdbfc0000,
    0x02000015, 0xbf8cc07f, 0x7daa0280, 0xbf89fff4, 0xbefe0416, 0x162c2890, 0xd5430016, 0x045a24ff,
    0x00000040, 0x7da82881, 0xbf880019, 0xe0781000, 0x02020216, 0x162624ff, 0x00000080, 0x343a2486,
    0xe0301028, 0x0202061d, 0xd8dc1008, 0x07000013, 0xbf8c0070, 0x7e300306, 0x7e320306, 0x7e340306,
    0x7e360306, 0x16142498, 0xe0781000, 0x0403180a, 0xe0741010, 0x04031a0a, 0x4a162481, 0x16161690,
    0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// The quad domain: bilinear in the four control points' (x, y) at gl_TessCoord (v5, v6); green is
// 4 * min(u, v, 1 - u, 1 - v), 0 on the patch edges.
alignas(256) constexpr std::array<std::uint32_t, 37> DomainQuad{
    0xf8000941, 0x00000000, 0x34020e86, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01, 0xe0341020,
    0x04030e01, 0xe0341030, 0x04031001, 0xbf8c3f70, 0x08040af2, 0x08060cf2, 0x10240702, 0x10260705,
    0x10280d05, 0x102a0d02, 0x102c1512, 0x3e2c1913, 0x3e2c1d14, 0x3e2c2115, 0x102e1712, 0x3e2e1b13,
    0x3e2e1f14, 0x3e2e2315, 0xd5510018, 0x040a0d05, 0x1e303103, 0x103030f6, 0x7e320280, 0x7e3402f2,
    0xf80008cf, 0x1a191716, 0xf800020f, 0x1a19181a, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 29> LocalIndexAsData{
    0xd5480004, 0x02090502, 0xd5480005, 0x02090102, 0xd5480006, 0x02110902, 0xd5480007, 0x02111102,
    0x7e080d04, 0x7e0a0d05, 0x7e0c0d06, 0x7e0e0d07, 0x10100cff, 0x3f19999a, 0x061010ff, 0xbf666666,
    0x3e1008ff, 0x3e800000, 0xd54b0009, 0x03c5e105, 0x161406a0, 0xd9340000, 0x0000080a, 0xd8340008,
    0x0000030a, 0xd8340010, 0x0000070a, 0xbf8cc07f, 0xbefd2106,
};

alignas(256) constexpr std::array<std::uint32_t, 46> HullUnguardedFactors{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008, 0x07000013, 0xbf8cc07f,
    0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xe0781000, 0x0403180a, 0x4a162481,
    0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

// The off-chip control-point address squares the invocation (v_mul_u32_u24 v22, v20, v20): ring
// addresses must stay affine; only hull LDS reads take a dynamic index.
alignas(256) constexpr std::array<std::uint32_t, 47> HullSquaredRingAddress{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2914, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0xe0781000, 0x02020216,
    0x7da82881, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013, 0xd8dc1008, 0x07000013,
    0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484, 0xe0781000, 0x0403180a,
    0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 49> HullAndSaveexec{
    0xd7650000, 0x000100c1, 0xd7660000, 0x000200c1, 0xbf8a0000, 0x7da206f9, 0x81060000, 0xd5480014,
    0x02151101, 0x362402ff, 0x000000ff, 0x160028a0, 0xd5430015, 0x040224ff, 0x00000060, 0xdbfc0000,
    0x02000015, 0x162c2890, 0xd5430016, 0x045a24ff, 0x00000030, 0xbf8cc07f, 0x7d8a2887, 0xbe94246a,
    0xe0781000, 0x02020216, 0x7da82881, 0xb06a0060, 0x162602f9, 0x0086066a, 0xd8d80008, 0x06000013,
    0xd8dc1008, 0x07000013, 0xbf8cc07f, 0x7e300306, 0x7e320306, 0x7e340306, 0x7e360306, 0x34142484,
    0xe0781000, 0x0403180a, 0x4a162481, 0x16161690, 0x4c1616ff, 0x00008000, 0xe0741000, 0x0202070b,
    0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 36> DomainPatchRead{
    0xf8000941, 0x00000000, 0x34020e84, 0x16020283, 0xe0341000, 0x04030a01, 0xe0341010, 0x04030c01,
    0xe0341020, 0x04030e01, 0x4a060e81, 0x16060690, 0x4c0606ff, 0x00008000, 0xe0341000, 0x04030c03,
    0xbf8c3f70, 0x08040af2, 0x08040d02, 0x10201505, 0x3e201906, 0x3e201d02, 0x10221705, 0x3e221b06,
    0x3e221f02, 0xd5510012, 0x040a0d05, 0x102424ff, 0x40400000, 0x7e260280, 0x7e2802f2, 0xf80008cf,
    0x14131110, 0xf800020f, 0x14131214, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 32> DomainCulledPrimitive{
    0x380000ff, 0x80000000, 0xf8000941, 0x00000000, 0x34020e84, 0x16020283, 0xe0341000, 0x04030a01,
    0xe0341010, 0x04030c01, 0xe0341020, 0x04030e01, 0xbf8c3f70, 0x08040af2, 0x08040d02, 0x10201505,
    0x3e201906, 0x3e201d02, 0x10221705, 0x3e221b06, 0x3e221f02, 0xd5510012, 0x040a0d05, 0x102424ff,
    0x40400000, 0x7e260280, 0x7e2802f2, 0xf80008cf, 0x14131110, 0xf800020f, 0x14131214, 0xbf810000,
};

alignas(256) constexpr std::array<std::uint32_t, 7> PixelCode{
    0xc8020002, 0xc8060102, 0xc80a0202, 0xc80e0302, 0xf800180f, 0x03020100, 0xbf810000,
};

constexpr ShaderRecompiler::TessellationConfiguration Patches{3u, 3u, 1u, 2u, 2u};
constexpr ShaderRecompiler::TessellationConfiguration QuadPatches{4u, 4u, 2u, 2u, 2u};
constexpr ShaderRecompiler::TessellationConfiguration QuadNextInputPatches{8u, 4u, 2u, 2u, 2u};
constexpr std::array<std::uint32_t, 3> PatchLevels{1u, 0u, 5u};

constexpr std::uint16_t Corner(std::uint32_t patch, std::uint32_t x, std::uint32_t y) {
    return static_cast<std::uint16_t>((PatchLevels[patch] << 8u) | (patch << 4u) | (x << 2u) | y);
}

alignas(256) constexpr std::array<std::uint16_t, 9> Indices{
    Corner(0, 0, 0), Corner(0, 2, 0), Corner(0, 1, 2),
    Corner(1, 0, 0), Corner(1, 2, 0), Corner(1, 1, 2),
    Corner(2, 0, 0), Corner(2, 2, 0), Corner(2, 1, 2),
};

alignas(256) constexpr std::array<std::uint16_t, 12> QuadIndices{
    Corner(0, 0, 0), Corner(0, 2, 0), Corner(0, 2, 2), Corner(0, 0, 2),
    Corner(1, 0, 0), Corner(1, 2, 0), Corner(1, 2, 2), Corner(1, 0, 2),
    Corner(2, 0, 0), Corner(2, 2, 0), Corner(2, 2, 2), Corner(2, 0, 2),
};

// HullQuadNextInput's patches: input 0 gives the level, inputs 1 to 4 are the corners, 5 to 7 unused.
alignas(256) constexpr std::array<std::uint16_t, 24> QuadNextInputIndices{
    Corner(0, 0, 0), Corner(0, 0, 0), Corner(0, 2, 0), Corner(0, 2, 2), Corner(0, 0, 2), Corner(0, 0, 0), Corner(0, 0, 0), Corner(0, 0, 0),
    Corner(1, 0, 0), Corner(1, 0, 0), Corner(1, 2, 0), Corner(1, 2, 2), Corner(1, 0, 2), Corner(1, 0, 0), Corner(1, 0, 0), Corner(1, 0, 0),
    Corner(2, 0, 0), Corner(2, 0, 0), Corner(2, 2, 0), Corner(2, 2, 2), Corner(2, 0, 2), Corner(2, 0, 0), Corner(2, 0, 0), Corner(2, 0, 0),
};

struct Topology {
    const ShaderRecompiler::TessellationConfiguration& patches;
    std::span<const std::uint16_t> indices;
};

const Topology Triangles{Patches, Indices};
const Topology Quads{QuadPatches, QuadIndices};
const Topology QuadsNextInput{QuadNextInputPatches, QuadNextInputIndices};

struct Stages {
    ShaderRecompiler::RecompileResult local;
    ShaderRecompiler::RecompileResult hull;
    ShaderRecompiler::RecompileResult domain;
    ShaderRecompiler::RecompileResult fragment;
};

ShaderRecompiler::RecompileResult Compile(const ShaderRecompiler::SpirvTarget& target, ShaderStage stage, std::span<const std::uint32_t> code, std::uint32_t& pushCursor, const Topology& topology = Triangles) {
    static const std::array<std::uint32_t, 16> hullUserData{0x1000u};
    static const std::array<std::uint32_t, 8> domainUserData{};
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const bool hull = stage == ShaderStage::TessellationControl;
    const auto userData = hull ? std::span<const std::uint32_t>(hullUserData) : stage == ShaderStage::TessellationEvaluation ? std::span<const std::uint32_t>(domainUserData) : std::span<const std::uint32_t>();
    std::optional<ShaderRecompiler::ShaderPixelStageInfo> pixel;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::optional<ShaderRecompiler::GraphicsCompileContext> graphics;
    if (stage == ShaderStage::Fragment) {
        ShaderRecompiler::ShaderPixelStageInfo info{};
        info.interpolatorCount = 1;
        info.interpolatorSettings[0] = 0x400u;
        info.targetOutputMode[0] = 9;
        info.targetExportMapping.fill(0xe4u);
        pixel = info;
    } else {
        vertex = ShaderRecompiler::ShaderVertexStageInfo{};
        graphics = ShaderRecompiler::GraphicsCompileContext{hull ? 0u : 8u, {}, std::nullopt, topology.patches, {reinterpret_cast<std::uintptr_t>(topology.indices.data()), static_cast<std::uint32_t>(topology.indices.size()), 2, 1}};
    }
    ShaderRecompiler::RecompileRequest request{
        {stage, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, hull ? 0u : 8u, userData, std::nullopt, pixel, vertex, memory},
        target,
        {0, 0, pushCursor, AgcDriver::Graphics::PipelinePushConstantBytes - pushCursor},
        graphics
    };
    request.useCache = false;
    auto result = ShaderRecompiler::Recompile(request);
    pushCursor += static_cast<std::uint32_t>(result.pushConstants.size());
    return result;
}

void ExpectRefusal(const ShaderRecompiler::SpirvTarget& target, ShaderStage stage, std::span<const std::uint32_t> code, const std::string& reason, const Topology& topology = Triangles) {
    std::uint32_t cursor = 0;
    try {
        static_cast<void>(Compile(target, stage, code, cursor, topology));
    } catch (const std::exception& error) {
        Require(std::string(error.what()).find(reason) != std::string::npos, "the refusal named another reason: " + std::string(error.what()));
        return;
    }
    Require(false, "a tessellation program was accepted although " + reason);
}

std::array<std::uint8_t, 4> PixelAt(std::uint32_t x, std::uint32_t y) {
    const auto offset = (static_cast<std::size_t>(y) * Width + x) * 4u;
    return {std::to_integer<std::uint8_t>(Pixels[offset]), std::to_integer<std::uint8_t>(Pixels[offset + 1]), std::to_integer<std::uint8_t>(Pixels[offset + 2]), std::to_integer<std::uint8_t>(Pixels[offset + 3])};
}

std::string PixelText(const std::array<std::uint8_t, 4>& pixel) {
    char text[64];
    std::snprintf(text, sizeof(text), "(%u, %u, %u, %u)", pixel[0], pixel[1], pixel[2], pixel[3]);
    return text;
}

std::uint32_t Column(std::uint32_t patch, float offset) {
    return static_cast<std::uint32_t>((-0.9f + 0.6f * static_cast<float>(patch) + offset + 1.0f) * 0.5f * Width);
}

std::uint32_t Row(float y) {
    return static_cast<std::uint32_t>((1.0f - y) * 0.5f * Height);
}

void Draw(AgcDriver::VulkanDevice& device, const Stages& stages, VkCullModeFlags cull, VkFrontFace front, const Topology& topology = Triangles) {
    for (std::size_t i = 0; i < Pixels.size(); i += 4) {
        for (std::size_t c = 0; c < 4; ++c) Pixels[i + c] = std::byte{Background[c]};
    }
    std::uint32_t cursor = 0;
    const auto offset = [&](const ShaderRecompiler::RecompileResult& result) {
        const auto at = result.pushConstants.empty() ? 0u : cursor;
        cursor += static_cast<std::uint32_t>(result.pushConstants.size());
        return at;
    };
    const std::array<AgcDriver::Graphics::CompiledShader, 4> shaders{{
        {ShaderStage::Local, &stages.local, offset(stages.local)},
        {ShaderStage::TessellationControl, &stages.hull, offset(stages.hull)},
        {ShaderStage::TessellationEvaluation, &stages.domain, offset(stages.domain)},
        {ShaderStage::Fragment, &stages.fragment, offset(stages.fragment)},
    }};
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Tessellation, 0x0200210du, 64, 64, std::nullopt, topology.patches};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = cull;
    state.frontFace = front;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    device.Draw(state, {reinterpret_cast<std::uintptr_t>(topology.indices.data()), static_cast<std::uint32_t>(topology.indices.size()), 2, 1, 0, true}, shaders);
    device.WaitIdle();
}

std::uint32_t CoveredPixels(std::uint32_t patch, bool interior) {
    std::uint32_t count = 0;
    for (std::uint32_t y = Row(0.5f); y <= Row(-0.5f) && y < Height; ++y) {
        for (std::uint32_t x = Column(patch, 0.0f); x <= Column(patch, 0.5f) && x < Width; ++x) {
            const auto pixel = PixelAt(x, y);
            if (pixel == Background) continue;
            Require(pixel[0] == 255 && pixel[2] == 0 && pixel[3] == 255, "patch " + std::to_string(patch) + " pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") is " + PixelText(pixel));
            if (!interior || pixel[1] > 32) ++count;
        }
    }
    return count;
}

void RequirePatches(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> hull, const std::string& program) {
    std::uint32_t cursor = 0;
    const Stages stages{
        Compile(target, ShaderStage::Local, LocalCode, cursor),
        Compile(target, ShaderStage::TessellationControl, hull, cursor),
        Compile(target, ShaderStage::TessellationEvaluation, DomainCode, cursor),
        Compile(target, ShaderStage::Fragment, PixelCode, cursor),
    };
    const auto center = Row(-0.5f + 1.0f / 3.0f);
    for (const auto cull : {VK_CULL_MODE_NONE, VK_CULL_MODE_BACK_BIT}) {
        Draw(device, stages, cull, VK_FRONT_FACE_COUNTER_CLOCKWISE);
        const std::string name = program + (cull == VK_CULL_MODE_NONE ? ", unculled" : ", back-face culled");
        Require(CoveredPixels(0, false) > 100 && CoveredPixels(0, true) == 0, name + ": the level-1 patch was not drawn as one flat triangle, centre " + PixelText(PixelAt(Column(0, 0.25f), center)));
        Require(CoveredPixels(1, false) == 0, name + ": the patch with level 0 was not culled");
        Require(CoveredPixels(2, false) > 100 && PixelAt(Column(2, 0.25f), center)[1] > 32, name + ": the level-5 patch has no interior domain points at its centre, " + PixelText(PixelAt(Column(2, 0.25f), center)));
        Require(PixelAt(0, 0) == Background && PixelAt(Width - 1, Height - 1) == Background, name + ": the corners changed");
    }
    Draw(device, stages, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE);
    Require(CoveredPixels(0, false) == 0 && CoveredPixels(2, false) == 0, program + ": tessellated triangles did not keep the control points' winding");
}

void RequireQuadPatches(AgcDriver::VulkanDevice& device, const ShaderRecompiler::SpirvTarget& target, std::span<const std::uint32_t> hull, const std::string& program, std::span<const std::uint32_t> local = LocalCode, const Topology& topology = Quads) {
    std::uint32_t cursor = 0;
    const Stages stages{
        Compile(target, ShaderStage::Local, local, cursor, topology),
        Compile(target, ShaderStage::TessellationControl, hull, cursor, topology),
        Compile(target, ShaderStage::TessellationEvaluation, DomainQuad, cursor, topology),
        Compile(target, ShaderStage::Fragment, PixelCode, cursor, topology),
    };
    const auto center = Row(0.0f);
    Draw(device, stages, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, topology);
    const auto square = (Column(0, 0.5f) - Column(0, 0.0f) + 1u) * (Row(-0.5f) - Row(0.5f) + 1u);
    Require(CoveredPixels(0, false) * 10u >= square * 9u && CoveredPixels(0, true) == 0, program + ": the level-1 quad patch was not drawn as one flat quad, centre " + PixelText(PixelAt(Column(0, 0.25f), center)));
    Require(CoveredPixels(1, false) == 0, program + ": the quad patch with level 0 was not culled");
    Require(CoveredPixels(2, false) * 10u >= square * 9u && PixelAt(Column(2, 0.25f), center)[1] > 32, program + ": the level-5 quad patch has no interior domain points at its centre, " + PixelText(PixelAt(Column(2, 0.25f), center)));
    Require(PixelAt(0, 0) == Background && PixelAt(Width - 1, Height - 1) == Background, program + ": the corners changed");
    // VGT_TF_PARAM topology 2 (cw) is VertexOrderCw in Vulkan's default upper-left domain origin
    // (RADV's mapping, not measured on the console). gl_TessCoord u runs right and v up on screen
    // here, so every triangle of a quad patch is counter-clockwise in framebuffer space.
    Draw(device, stages, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE, topology);
    const auto counterClockwise = CoveredPixels(0, false) + CoveredPixels(2, false);
    Draw(device, stages, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, topology);
    const auto clockwise = CoveredPixels(0, false) + CoveredPixels(2, false);
    Require(clockwise == 0 && counterClockwise * 10u >= 2u * square * 9u, program + ": tessellated quads are not all counter-clockwise (" + std::to_string(counterClockwise) + " pixels kept with a counter-clockwise front face, " + std::to_string(clockwise) + " with a clockwise one)");
}

}

int main() {
    try {
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const auto target = device->Target();
        if (!target.tessellation.has_value()) {
            std::puts("tessellation: skipped, the device has no tessellation shaders");
            return VulkanTestSkipped;
        }
        ExpectRefusal(target, ShaderStage::TessellationControl, HullUnguardedFactors, "not written by control point 0 alone");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullSquaredRingAddress, "not affine");
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainPatchRead, "patch constants");
        ExpectRefusal(target, ShaderStage::TessellationEvaluation, DomainCulledPrimitive, "passthrough primitive");
        ExpectRefusal(target, ShaderStage::Local, LocalIndexAsData, "outside a tessellation ring or LDS address");
        ExpectRefusal(target, ShaderStage::TessellationControl, HullReadBackInBranch, "read-back barrier in uniform control flow", Quads);
        ExpectRefusal(target, ShaderStage::TessellationControl, HullQuadHalfSlotIndex, "not a whole number of control points", Quads);

        RequirePatches(*device, target, HullCode, "hull program");
        RequirePatches(*device, target, HullRestoredExecSaved, "hull program saving its restored exec");
        RequirePatches(*device, target, HullAndSaveexec, "hull program masking its control-point store with s_and_saveexec");
        RequireQuadPatches(*device, target, HullQuad, "quad hull program");
        RequireQuadPatches(*device, target, HullQuadDynamicIndex, "quad hull program reading a control point at a computed index");
        RequireQuadPatches(*device, target, HullQuadReadsControlPoints, "quad hull program reading back another control point");
        RequireQuadPatches(*device, target, HullQuadPatchList, "quad hull program looping over a list of input control points");
        RequireQuadPatches(*device, target, HullQuadNextInput, "quad hull program reading the next of 8 input control points of 24 bytes", LocalCodeStride24, QuadsNextInput);
        std::puts("Tessellation tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
