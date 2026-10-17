#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/NativePatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

// The native pattern fill of the title's dword fill kernel (NativePatternFill.cpp): the recognition
// accepts exactly the shapes a single-dword fill expresses and refuses the others, and the native
// fill leaves the same bytes and the same pending state as running the kernel through the
// recompiler, including over a storage image whose results are still pending over the range.

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DwordPatternFill;
using AgcDriver::DriverDetail::NativeFillRefusal;
using AgcDriver::DriverDetail::MatchDwordPatternFill;

alignas(256) constexpr std::array<std::uint32_t, 23> FillKernel{0xbfa00001u, 0xd7460004u, 0x04010c09u, 0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u, 0x7da80808u, 0xbf88000du, 0x360a0883u, 0xbe84047eu, 0xbe86047eu, 0x7e100505u, 0x7da40a08u, 0xbefc0308u, 0x7e0c8700u, 0x8a867e06u, 0xbefe0406u, 0xbf85fff9u, 0xbefe0404u, 0xe0102000u, 0x80000604u, 0xbf810000u};

// image_store of 1.0 into texel (lane, 0) of the image in s[4:11] (DccKeysFirstWrite's WriteCode).
alignas(256) constexpr std::array<std::uint32_t, 8> WriteImageCode{0x7e020280, 0x7e0402f2, 0x7e0602f2, 0x7e0802f2, 0x7e0a02f2, 0xf0201f08, 0x00010200, 0xbf810000};

constexpr std::uint32_t Format32Uint = 20;
constexpr std::uint32_t Format8888Unorm = 56;
constexpr std::uint32_t TileRenderTarget64KB = 0x1b;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t Side = 128;
constexpr std::size_t SurfaceBytes = 65536;
constexpr std::size_t BlockBytes = 8 * 65536;
// COMPUTE_PGM_RSRC2: nine user SGPRs, TGID_X_EN.
constexpr std::uint32_t PgmRsrc2 = (9u << 1u) | (1u << 7u);

std::uint64_t AddressOf(const void* data) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(65536, BlockBytes));
#endif
        Require(block != nullptr, "native pattern fill: cannot allocate the guest block");
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    }
    ~GuestBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        VirtualFree(block, 0, MEM_RELEASE);
#else
        std::free(block);
#endif
    }
    GuestBlock(const GuestBlock&) = delete;
    GuestBlock& operator=(const GuestBlock&) = delete;
    std::uint8_t* Data() { return block; }

private:
    std::uint8_t* block = nullptr;
};

struct Shape {
    std::uint64_t base = 0;
    std::uint32_t records = 0;
    std::uint32_t count = 0;
    std::uint32_t groups = 0;
    std::array<std::uint32_t, 4> pattern{};
    std::uint32_t format = Format32Uint;
    std::uint32_t oob = 0;
    std::uint32_t stride = 4;
};

std::vector<std::uint32_t> UserData(const Shape& shape) {
    return {
        static_cast<std::uint32_t>(shape.base),
        static_cast<std::uint32_t>((shape.base >> 32u) & 0xffffu) | (shape.stride << 16u),
        shape.records,
        0xfacu | (shape.format << 12u) | (shape.oob << 28u),
        shape.pattern[0], shape.pattern[1], shape.pattern[2], shape.pattern[3],
        shape.count,
    };
}

std::array<std::uint32_t, 5> Packet(std::uint32_t groups, std::uint32_t initiator = 1u) {
    return {0u, groups, 1u, 1u, initiator};
}

NativeFillRefusal Match(const Shape& shape, DwordPatternFill& fill, std::array<std::uint32_t, 5> packet = {}, std::uint32_t rsrc2 = PgmRsrc2, std::span<const std::uint32_t> code = FillKernel) {
    if (packet[1] == 0) packet = Packet(shape.groups);
    const auto userData = UserData(shape);
    return MatchDwordPatternFill(code, userData, packet, rsrc2, {64, 1, 1}, false, fill);
}

void RequireRefusal(NativeFillRefusal got, NativeFillRefusal expected, const char* what) {
    char text[256];
    std::snprintf(text, sizeof(text), "%s: the recognition answered '%s', expected '%s'", what, AgcDriver::DriverDetail::NativeFillRefusalName(got), AgcDriver::DriverDetail::NativeFillRefusalName(expected));
    Require(got == expected, text);
}

void TestRecognition() {
    const std::uint64_t base = 0x123456780ull;
    const Shape accepted{base, 2000, 1000, 20, {0x40404040u, 0x40404040u, 0x40404040u, 0x40404040u}};
    DwordPatternFill fill;
    RequireRefusal(Match(accepted, fill), NativeFillRefusal::None, "a uniform dword fill");
    Require(fill.base == base && fill.bytes == 4000 && fill.pattern == 0x40404040u && fill.descriptorEnd == base + 8000, "the fill's range is not the kernel's: count bounds it to 1000 dwords");
    auto shape = accepted;
    shape.count = 5000;
    shape.records = 300;
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, "a fill bound by num_records");
    Require(fill.bytes == 1200, "num_records does not bound the fill");
    shape.records = 5000;
    shape.groups = 3;
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, "a fill bound by the grid");
    Require(fill.bytes == 192u * 4u, "the grid does not bound the fill");
    shape = accepted;
    shape.oob = 1;
    shape.format = 22;
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, "a 32_FLOAT V# with index bounds");
    shape.format = 21;
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, "a 32_SINT V#");

    const auto refused = [&](Shape changed, NativeFillRefusal expected, const char* what, std::array<std::uint32_t, 5> packet = {}, std::uint32_t rsrc2 = PgmRsrc2) {
        DwordPatternFill unused;
        RequireRefusal(Match(changed, unused, packet, rsrc2), expected, what);
    };
    shape = accepted;
    shape.pattern[2] = 0x40404041u;
    refused(shape, NativeFillRefusal::Pattern, "a pattern whose dwords differ");
    shape = accepted;
    shape.stride = 16;
    refused(shape, NativeFillRefusal::Descriptor, "a strided V#");
    shape = accepted;
    shape.format = Format8888Unorm;
    refused(shape, NativeFillRefusal::Format, "a format that converts the stored dword");
    shape.format = 75;
    refused(shape, NativeFillRefusal::Format, "a four-component format");
    shape = accepted;
    shape.oob = 3;
    refused(shape, NativeFillRefusal::Bounds, "raw bounds (num_records in bytes)");
    shape.oob = 2;
    refused(shape, NativeFillRefusal::Bounds, "bounds checking only num_records != 0");
    shape = accepted;
    shape.base = base + 2;
    refused(shape, NativeFillRefusal::Unaligned, "a V# base that splits dwords");
    shape = accepted;
    shape.count = 0;
    refused(shape, NativeFillRefusal::Empty, "a fill of no dwords");
    shape = accepted;
    shape.count = 0xffffffffu;
    shape.records = 0xffffffffu;
    shape.groups = (1u << 24u) + 1u;
    refused(shape, NativeFillRefusal::TooLarge, "a fill of more than 2^30 dwords");
    refused(accepted, NativeFillRefusal::PartialGroups, "a partial-group dispatch", Packet(20, 0x21u));
    refused(accepted, NativeFillRefusal::GroupShape, "a 2-D grid", {0u, 20u, 2u, 1u, 1u});
    refused(accepted, NativeFillRefusal::GroupId, "a program without the group id in s9", {}, 9u << 1u);
    refused(accepted, NativeFillRefusal::GroupId, "a program with ten user SGPRs", {}, (10u << 1u) | (1u << 7u));
    {
        auto userData = UserData(accepted);
        userData.push_back(0);
        const auto packet = Packet(20);
        DwordPatternFill unused;
        RequireRefusal(MatchDwordPatternFill(FillKernel, userData, packet, PgmRsrc2, {64, 1, 1}, false, unused), NativeFillRefusal::UserData, "ten user data words");
        userData.pop_back();
        RequireRefusal(MatchDwordPatternFill(FillKernel, userData, packet, PgmRsrc2, {64, 1, 1}, true, unused), NativeFillRefusal::Indirect, "an indirect dispatch");
        RequireRefusal(MatchDwordPatternFill(FillKernel, userData, packet, PgmRsrc2, {32, 1, 1}, false, unused), NativeFillRefusal::GroupShape, "a 32-thread group");
        auto code = std::vector<std::uint32_t>(FillKernel.begin(), FillKernel.end());
        code[20] = 0xe01c2000u;
        RequireRefusal(MatchDwordPatternFill(code, userData, packet, PgmRsrc2, {64, 1, 1}, false, unused), NativeFillRefusal::NotKernel, "another store in the kernel's place");
    }
    std::puts("recognition: accepted and refused shapes as expected");
}

void DispatchKernel(AgcDriver::VulkanDevice& device, const std::vector<std::uint32_t>& userData, std::uint32_t groups) {
    const std::span<const std::uint32_t> code(FillKernel);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{64, 1, 1}, 0u, {true, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.ComputeTarget(64),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

void DispatchImageWrite(AgcDriver::VulkanDevice& device, std::uint8_t* texels) {
    const auto address = AddressOf(texels);
    const std::array<std::uint32_t, 8> texture{
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (Format8888Unorm << 20u) | (((Side - 1u) & 3u) << 30u),
        ((Side - 1u) >> 2u) | ((Side - 1u) << 14u),
        0xfacu | (TileRenderTarget64KB << 20u) | (Type2D << 28u),
        0u, 0u, 0u, 0u,
    };
    std::vector<std::uint32_t> userData(16, 0u);
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> code(WriteImageCode);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{32, 1, 1}, 0u, {false, false, false}, false, 1, {}};
    ShaderRecompiler::RecompileRequest request{
        {ShaderRecompiler::ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

// One shape, both ways: the kernel over region `kernel`, the native fill over the same offsets in
// region `native`; the two regions must hold the same bytes after, and the pending-write state
// right after each must agree.
void Compare(AgcDriver::VulkanDevice& device, std::uint8_t* kernel, std::uint8_t* native, std::size_t windowBytes, std::size_t offset, Shape shape, const char* what) {
    std::memset(kernel, 0x11, windowBytes);
    std::memset(native, 0x11, windowBytes);
    shape.base = AddressOf(kernel + offset);
    auto* recorder = AgcDriver::Graphics::Recorder::Active();
    Require(recorder != nullptr, "native pattern fill: no active recorder");
    DwordPatternFill fill;
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, what);
    DispatchKernel(device, UserData(shape), shape.groups);
    const bool kernelPending = recorder->PendingWriteOverlaps(fill.base, fill.bytes);
    device.WaitIdle();
    shape.base = AddressOf(native + offset);
    RequireRefusal(Match(shape, fill), NativeFillRefusal::None, what);
    const bool gpu = AgcDriver::DriverDetail::StoreDwordPatternFill(device, fill, true, false);
    const bool nativePending = recorder->PendingWriteOverlaps(fill.base, fill.bytes);
    device.WaitIdle();
    char text[320];
    std::snprintf(text, sizeof(text), "%s: the kernel's write is %spending after its record, the native fill's %s", what, kernelPending ? "" : "not ", nativePending ? "is" : "is not");
    Require(!gpu || kernelPending == nativePending, text);
    const auto mismatch = std::mismatch(kernel, kernel + windowBytes, native);
    if (mismatch.first != kernel + windowBytes) {
        const auto at = static_cast<std::size_t>(mismatch.first - kernel);
        std::string map;
        for (std::size_t dword = 0; dword < (offset + fill.bytes) / 4 + 4 && dword < 200; ++dword) map += kernel[dword * 4] == 0x11 ? '.' : 'X';
        std::printf("kernel dwords written (first 200): %s\n", map.c_str());
        std::snprintf(text, sizeof(text), "%s: byte %zu differs (kernel 0x%02x, native 0x%02x; the fill covers %zu..%zu)", what, at, *mismatch.first, *mismatch.second, offset, offset + fill.bytes);
        Require(false, text);
    }
    std::uint32_t word = 0;
    std::memcpy(&word, native + offset, 4);
    std::snprintf(text, sizeof(text), "%s: the filled range does not hold the pattern", what);
    Require(word == shape.pattern[0] && native[offset + fill.bytes] == 0x11 && (offset == 0 || native[offset - 1] == 0x11), text);
    std::printf("%s: identical (%zu bytes, %s)\n", what, fill.bytes, gpu ? "GPU fill" : "CPU store");
}

void TestDevice(AgcDriver::VulkanDevice& device, std::uint8_t* block) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    auto* kernel = block;
    auto* native = block + 65536;
    constexpr std::size_t Window = 32768;
    const std::uint32_t word = 0xdeadbeefu;
    Compare(device, kernel, native, Window, 12, {0, 2000, 1000, 20, {word, word, word, word}}, "a fill bound by count at a dword (not 16-byte) offset");
    Compare(device, kernel, native, Window, 64, {0, 300, 5000, 10, {word, word, word, word}}, "a fill bound by num_records");
    Compare(device, kernel, native, Window, 4, {0, 5000, 5000, 3, {0, 0, 0, 0}, Format32Uint, 1}, "a zero fill bound by the grid");
    Compare(device, kernel, native, Window, 256, {0, 777, 777, 13, {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, 22}, "an all-ones fill through a 32_FLOAT V#");
    // A fill that is not dword-uniform keeps a 16-byte granule: FillBuffer refuses it at a dword offset.
    const std::array<std::uint32_t, 4> mixed{1u, 2u, 3u, 4u};
    Require(!device.FillBuffer(AddressOf(native + 4), 64, mixed), "FillBuffer accepted a 16-byte pattern at a dword offset");

    // A storage image's results still pending over the range: both ways publish them first, so
    // the fill wins inside the range and the image's bytes stay outside it.
    auto* surfaceKernel = block + 2 * 65536;
    auto* surfaceNative = block + 3 * 65536;
    std::memset(surfaceKernel, 0x55, SurfaceBytes);
    std::memset(surfaceNative, 0x55, SurfaceBytes);
    const auto overImage = [&](std::uint8_t* surface, bool useKernel) {
        DispatchImageWrite(device, surface);
        Require(AgcDriver::Graphics::StorageTexture::FindPending(AddressOf(surface), SurfaceBytes) != nullptr, "the image write left no pending image");
        Shape shape{AddressOf(surface + 128), 64, 64, 1, {0x01020304u, 0x01020304u, 0x01020304u, 0x01020304u}};
        if (useKernel) {
            DispatchKernel(device, UserData(shape), 1);
        } else {
            DwordPatternFill fill;
            RequireRefusal(Match(shape, fill), NativeFillRefusal::None, "the fill over a pending image");
            AgcDriver::DriverDetail::StoreDwordPatternFill(device, fill, true, false);
        }
        AgcDriver::Graphics::StorageTexture::FlushPending(AddressOf(surface), SurfaceBytes, nullptr, "test");
        device.WaitIdle();
    };
    overImage(surfaceKernel, true);
    overImage(surfaceNative, false);
    const auto written = std::count(surfaceNative, surfaceNative + SurfaceBytes, std::uint8_t{0xff});
    Require(std::equal(surfaceKernel, surfaceKernel + SurfaceBytes, surfaceNative), "a fill over a pending image: the kernel and the native fill leave different surfaces");
    std::uint32_t first = 0;
    std::memcpy(&first, surfaceNative + 128, 4);
    Require(first == 0x01020304u && written != 0, "a fill over a pending image: the fill or the image's write is lost");
    std::printf("a fill over a pending storage image: identical surfaces (%lld image bytes kept)\n", static_cast<long long>(written));

    // The DCC key memo: a 0001 key fill noted with the fill answers a key read without a wait.
    auto* keys = block + 4 * 65536;
    std::memset(keys, 0xff, 256);
    device.WaitIdle();
    Shape keyShape{AddressOf(keys), 64, 64, 1, {0x40404040u, 0x40404040u, 0x40404040u, 0x40404040u}};
    DwordPatternFill keyFill;
    RequireRefusal(Match(keyShape, keyFill), NativeFillRefusal::None, "the key fill");
    if (AgcDriver::DriverDetail::StoreDwordPatternFill(device, keyFill, false, true)) {
        auto* recorder = AgcDriver::Graphics::Recorder::Active();
        const auto read = AgcDriver::Graphics::CurrentDccKeys(AddressOf(keys), 256u * 256u);
        const bool waited = !recorder->PendingWriteOverlaps(AddressOf(keys), 256);
        char text[256];
        std::snprintf(text, sizeof(text), "a native 0001 key fill: the keys read %s and the read %s", AgcDriver::Graphics::DccKeysName(read), waited ? "waited for the GPU" : "did not wait");
        Require(read == AgcDriver::Graphics::DccKeys::Clear0001, text);
        std::printf("%s\n", text);
    }
    device.WaitIdle();
    Require(std::count(keys, keys + 256, std::uint8_t{0x40}) == 256, "the key fill did not store 0x40 over the keys");
}

}

int main() {
    try {
        TestRecognition();
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestBlock block;
        TestDevice(*device, block.Data());
        std::puts("native pattern fill tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
