#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The recompiler's buffer usage metadata (DescriptorBinding::bufferRead and bufferCovers) for
// hand-assembled compute programs, and, for the programs it proves a store cover for, a real
// dispatch: every byte CoveredBufferBytes claims for the dispatch's grid must hold what the
// program stored (the soundness the write-back elision relies on).

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t GroupSize = 64;
constexpr std::uint32_t Groups = 4;
constexpr std::uint32_t Sentinel = 0xdeadbeefu;
alignas(256) std::array<std::uint32_t, 64> Input{};
alignas(256) std::array<std::uint32_t, GroupSize * Groups * 2> Output{};

// GFX10 encodings.
constexpr std::uint32_t SLshlB32S9S8By6 = 0x8f098608u;   // s_lshl_b32 s9, s8, 6        (WorkgroupId.x * 64)
constexpr std::uint32_t VAddNcU32V1S9V0 = 0x4a020009u;   // v_add_nc_u32 v1, s9, v0     (global id)
constexpr std::array<std::uint32_t, 2> VMadU32U24 = {0xd5430001u, 0x04018008u};  // v_mad_u32_u24 v1, s8, 64, v0
constexpr std::array<std::uint32_t, 2> VLshlOrB32 = {0xd76f0001u, 0x04010c08u};   // v_lshl_or_b32 v1, s8, 6, v0
constexpr std::uint32_t VLshlrevB32V1By2 = 0x34020282u;  // v_lshlrev_b32 v1, 2, v1     (dword offset)
constexpr std::uint32_t VLshlrevB32V1By3 = 0x34020283u;  // v_lshlrev_b32 v1, 3, v1     (every other dword)
constexpr std::uint32_t VCmpGtU32Vcc32V0 = 0x7d8800a0u;  // v_cmp_gt_u32 vcc, 32, v0
constexpr std::uint32_t SAndSaveexecB64 = 0xbe8a246au;   // s_and_saveexec_b64 s[10:11], vcc
constexpr std::array<std::uint32_t, 2> BufferLoadV2 = {0xe0301000u, 0x80010201u};   // buffer_load_dword v2, v1, s[4:7], 0 offen
constexpr std::array<std::uint32_t, 2> BufferStoreV0 = {0xe0701000u, 0x80010001u};  // buffer_store_dword v0, v1, s[4:7], 0 offen
constexpr std::array<std::uint32_t, 2> BufferStoreV0Plus2 = {0xe0701002u, 0x80010001u};  // buffer_store_dword v0, v1, s[4:7], 0 offen offset:2
constexpr std::array<std::uint32_t, 2> BufferAtomicAdd = {0xe0c81000u, 0x80010001u}; // buffer_atomic_add v0, v1, s[4:7], 0 offen
constexpr std::array<std::uint32_t, 2> BufferStoreV2 = {0xe0701000u, 0x80010201u};  // buffer_store_dword v2, v1, s[4:7], 0 offen
constexpr std::uint32_t VAddNcU32V2By1 = 0x4a040481u;    // v_add_nc_u32 v2, 1, v2
constexpr std::uint32_t SWaitcnt = 0xbf8c3f70u;
constexpr std::uint32_t SEndpgm = 0xbf810000u;

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

struct Program {
    const char* name;
    std::vector<std::uint32_t> code;
};

std::vector<std::uint32_t> Code(std::initializer_list<std::uint32_t> words) {
    return std::vector<std::uint32_t>(words);
}

std::vector<std::uint32_t> Join(std::initializer_list<std::vector<std::uint32_t>> parts) {
    std::vector<std::uint32_t> code;
    for (const auto& part : parts) code.insert(code.end(), part.begin(), part.end());
    return code;
}

std::vector<std::uint32_t> Pair(const std::array<std::uint32_t, 2>& words) {
    return {words[0], words[1]};
}

struct Compiled {
    ShaderRecompiler::RecompileResult result;
    std::vector<std::uint32_t> code;
};

// The output element of the program's GuestBuffers binding (its V# is the output's), with what the
// binding says about it.
struct OutputElement {
    bool written = true;
    bool read = true;
    bool atomic = false;
    std::vector<ShaderRecompiler::BufferStoreCover> covers;
    std::array<std::uint32_t, 4> words{};
};

Compiled Compile(AgcDriver::VulkanDevice& device, std::vector<std::uint32_t> code) {
    Compiled compiled{{}, std::move(code)};
    std::vector<std::uint32_t> userData(8, 0u);
    const auto input = BufferDescriptor(Input.data(), static_cast<std::uint32_t>(Input.size() * 4u));
    const auto output = BufferDescriptor(Output.data(), static_cast<std::uint32_t>(Output.size() * 4u));
    std::copy(input.begin(), input.end(), userData.begin());
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const std::span<const std::uint32_t> span(compiled.code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(span.data()), std::as_bytes(span)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{GroupSize, 1, 1}, 0u, {true, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(span.data()), span, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    compiled.result = ShaderRecompiler::Recompile(request);
    return compiled;
}

OutputElement Output0(const ShaderRecompiler::RecompileResult& result) {
    const auto low = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(Output.data()));
    for (const auto& binding : result.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            if (binding.guestDescriptor.at(static_cast<std::size_t>(element) * 4u) != low) continue;
            OutputElement out;
            out.written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
            out.read = element >= binding.bufferRead.size() || binding.bufferRead[element];
            out.atomic = element < binding.bufferAtomic.size() && binding.bufferAtomic[element];
            if (element < binding.bufferCovers.size()) out.covers = binding.bufferCovers[element];
            std::copy_n(binding.guestDescriptor.begin() + static_cast<std::ptrdiff_t>(element) * 4, 4, out.words.begin());
            return out;
        }
    }
    throw std::runtime_error("buffer store covers: the output element is not bound");
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> Covered(const OutputElement& element, std::uint32_t groups) {
    std::optional<std::pair<std::uint64_t, std::uint64_t>> covered;
    const ShaderRecompiler::DispatchGrid grid{{groups, 1, 1}, {GroupSize, 1, 1}};
    for (const auto& cover : element.covers) {
        const auto range = ShaderRecompiler::CoveredBufferBytes(cover, grid, std::span<const std::uint32_t, 4>(element.words));
        if (!range.has_value()) continue;
        if (!covered.has_value() || range->second - range->first > covered->second - covered->first) covered = range;
    }
    return covered;
}

std::string Range(const std::optional<std::pair<std::uint64_t, std::uint64_t>>& range) {
    if (!range.has_value()) return "none";
    return "[" + std::to_string(range->first) + ", " + std::to_string(range->second) + ")";
}

// Runs the program over `groups` workgroups and checks that every dword of `covered` was stored.
void RunAndCheck(AgcDriver::VulkanDevice& device, const Compiled& compiled, std::uint32_t groups, std::pair<std::uint64_t, std::uint64_t> covered, const char* name) {
    Output.fill(Sentinel);
    device.Dispatch(compiled.result, groups, 1, 1, {}, reinterpret_cast<std::uintptr_t>(compiled.code.data()));
    device.WaitIdle();
    for (auto byte = covered.first; byte < covered.second; byte += 4) {
        const auto word = Output.at(static_cast<std::size_t>(byte / 4));
        Require(word != Sentinel, std::string("buffer store covers: ") + name + " claims byte " + std::to_string(byte) + " written, the dispatch left it");
    }
}

void Run(AgcDriver::VulkanDevice& device) {
    const auto prologue = Code({SLshlB32S9S8By6, VAddNcU32V1S9V0});
    {
        // Every invocation stores its global id at gid * 4: write-only, the whole grid's dwords.
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By2}), Pair(BufferStoreV0), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && !element.read && !element.atomic, "buffer store covers: a write-only element is not marked written and unread");
        Require(element.covers.size() == 1, "buffer store covers: the dense store has no cover");
        const auto covered = Covered(element, Groups);
        Require(covered == std::pair<std::uint64_t, std::uint64_t>{0, Groups * GroupSize * 4u}, "buffer store covers: the dense store covers " + Range(covered) + ", expected the grid's dwords");
        // One group fewer, one dword per invocation fewer.
        Require(Covered(element, Groups - 1) == std::pair<std::uint64_t, std::uint64_t>{0, (Groups - 1) * GroupSize * 4u}, "buffer store covers: the cover does not follow the group count");
        // Past the records: clipped to the last whole dword below them.
        Require(Covered(element, 1000) == std::pair<std::uint64_t, std::uint64_t>{0, Output.size() * 4u}, "buffer store covers: the cover is not clipped to the records");
        RunAndCheck(device, compiled, Groups, *covered, "the dense store");
        for (std::uint32_t id = 0; id < Groups * GroupSize; ++id) Require(Output[id] == id % GroupSize, "buffer store covers: the dense store wrote the wrong value");
    }
    {
        // The global id as v_mad_u32_u24 and as v_lshl_or_b32 compute it: the 24-bit operands and
        // the OR of disjoint bits are taken as plain arithmetic under the bounds they need.
        for (const auto& [name, idCode] : {std::pair{"v_mad_u32_u24", Pair(VMadU32U24)}, std::pair{"v_lshl_or_b32", Pair(VLshlOrB32)}}) {
            const auto compiled = Compile(device, Join({idCode, Code({VLshlrevB32V1By2}), Pair(BufferStoreV0), Code({SEndpgm})}));
            const auto element = Output0(compiled.result);
            const auto covered = Covered(element, Groups);
            Require(covered == std::pair<std::uint64_t, std::uint64_t>{0, Groups * GroupSize * 4u}, std::string("buffer store covers: the global id from ") + name + " covers " + Range(covered) + ", expected the grid's dwords");
            RunAndCheck(device, compiled, Groups, *covered, name);
            if (std::string_view(name) == "v_lshl_or_b32") {
                // The OR stays an addition only while the local id fits below the shift: a
                // workgroup of 128 would carry into the group bits.
                const ShaderRecompiler::DispatchGrid wide{{Groups, 1, 1}, {128, 1, 1}};
                bool any = false;
                for (const auto& cover : element.covers) any = any || ShaderRecompiler::CoveredBufferBytes(cover, wide, std::span<const std::uint32_t, 4>(element.words)).has_value();
                Require(!any, "buffer store covers: the OR's bound did not refuse a workgroup wider than its shift");
            }
        }
    }
    {
        // A load through the same V# whose value the store writes back, plus one: read and
        // written, still a cover.
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By2}), Pair(BufferLoadV2), Code({SWaitcnt, VAddNcU32V2By1}), Pair(BufferStoreV2), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && element.read, "buffer store covers: a read-write element is not marked read");
        const auto covered = Covered(element, Groups);
        Require(covered.has_value(), "buffer store covers: the read-write store lost its cover");
        RunAndCheck(device, compiled, Groups, *covered, "the read-write store");
    }
    {
        // Every other dword: only the first store's bytes form a proved run.
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By3}), Pair(BufferStoreV0), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && !element.read, "buffer store covers: the strided element is misclassified");
        const auto covered = Covered(element, Groups);
        Require(covered == std::pair<std::uint64_t, std::uint64_t>{0, 4}, "buffer store covers: the strided store covers " + Range(covered) + ", expected its first dword only");
        RunAndCheck(device, compiled, Groups, *covered, "the strided store");
        Require(Output[1] == Sentinel, "buffer store covers: the strided store wrote between its stores");
    }
    {
        // A store under a lane mask (s_and_saveexec): not every invocation stores, no cover.
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By2, VCmpGtU32Vcc32V0, SAndSaveexecB64}), Pair(BufferStoreV0), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && !element.read, "buffer store covers: the masked element is misclassified");
        Require(element.covers.empty(), "buffer store covers: a store under a lane mask has a cover");
    }
    {
        // A dword store two bytes off its natural alignment: the emulated store lands on the dword
        // below, so no cover is claimed.
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By2}), Pair(BufferStoreV0Plus2), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && !element.read, "buffer store covers: the unaligned element is misclassified");
        Require(element.covers.empty(), "buffer store covers: an unaligned dword store has a cover");
    }
    {
        // An atomic: read and written, no cover (atomics are not raw stores).
        const auto compiled = Compile(device, Join({prologue, Code({VLshlrevB32V1By2}), Pair(BufferAtomicAdd), Code({SEndpgm})}));
        const auto element = Output0(compiled.result);
        Require(element.written && element.read && element.atomic, "buffer store covers: an atomic element is not marked read, written and atomic");
        Require(element.covers.empty(), "buffer store covers: an atomic has a cover");
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        std::puts("buffer store covers tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
