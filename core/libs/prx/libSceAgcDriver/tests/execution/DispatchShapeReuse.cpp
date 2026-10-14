#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/ShapeReuse.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "VulkanTestDevice.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

// Dispatch shape reuse (APS5_DISPATCH_SHAPE_REUSE): a program reads an output V# and a data word
// through a user pointer. A dispatch whose pointer moved to a copy of the same table reuses the
// earlier capture; one whose moved table holds only another data word (a pure flat-slot leaf)
// reuses it with the word refreshed; one whose moved table holds another V# falls back to a full
// capture. A program whose output V# is the user data itself reuses a capture across a moved
// output (the V# words are copied into the snapshot); one that adds to its pointer before loading
// keeps the pointer word in the shape key. Every dispatch must store what its own data says. With
// the argument `verify` (APS5_VERIFY_SHAPE_REUSE) every reuse is also compared with a full capture;
// with `descriptors` (APS5_DISPATCH_SHAPE_REFRESH_DESCRIPTORS) the moved table with another V#
// reuses the capture with the descriptor words refreshed instead of falling back.

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DispatchShapeReuseCounts;
using AgcDriver::DriverDetail::ShapeRefusal;

void SetEnvironment(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "1");
#else
    setenv(name, "1", 1);
#endif
}

alignas(256) std::array<std::uint32_t, 16> OutputA;
alignas(256) std::array<std::uint32_t, 16> OutputB;

struct alignas(256) Table {
    std::array<std::uint32_t, 4> output{};
    std::uint32_t value = 0;
    std::array<std::uint32_t, 59> padding{};
};
std::array<Table, 6> Tables;

// The table pointer in u[0:1].
alignas(256) constexpr std::array<std::uint32_t, 16> TableCode{
    0xf4080100u, 0xfa000000u, // s_load_dwordx4 s[4:7], s[0:1], 0x0
    0xf4000200u, 0xfa000010u, // s_load_dword s8, s[0:1], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e020208u,              // v_mov_b32 v1, s8
    0xe0700000u, 0x80010100u, // buffer_store_dword v1, off, s[4:7], 0
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// The output V# in u[0:3] and the data pointer in u[4:5].
alignas(256) constexpr std::array<std::uint32_t, 16> DirectCode{
    0xf4000202u, 0xfa000000u, // s_load_dword s8, s[4:5], 0x0
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e020208u,              // v_mov_b32 v1, s8
    0xe0700000u, 0x80000100u, // buffer_store_dword v1, off, s[0:3], 0
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// The table pointer moved by 16 bytes before the loads: arithmetic on the pointer word.
alignas(256) constexpr std::array<std::uint32_t, 16> OffsetCode{
    0x80009000u,              // s_add_u32 s0, s0, 16
    0x82018001u,              // s_addc_u32 s1, s1, 0
    0xf4080100u, 0xfa000000u, // s_load_dwordx4 s[4:7], s[0:1], 0x0
    0xf4000200u, 0xfa000010u, // s_load_dword s8, s[0:1], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e020208u,              // v_mov_b32 v1, s8
    0xe0700000u, 0x80010100u, // buffer_store_dword v1, off, s[4:7], 0
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

std::array<std::uint32_t, 4> Buffer(const void* pointer) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, sizeof(OutputA), 0x31016facu};
}

std::array<std::uint32_t, 2> Pointer(const void* pointer) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
}

struct Header {
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    ShaderSpecialRegs specials{};
};

std::uint64_t Refused(ShapeRefusal reason) {
    return DispatchShapeReuseCounts().refusals[static_cast<std::size_t>(reason)];
}

void Register(Header& header, std::span<const std::uint32_t> code, std::uint32_t userWords) {
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = const_cast<std::uint32_t*>(code.data());
    header.shader.sh_registers = header.registers.data();
    header.shader.num_sh_registers = header.registers.size();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000u;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)},
        {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {0x213, userWords << 1u}}};
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    AgcDriverResolveShaderAbi_nid_postfix(&header.shader, {}, {});
}

void Submit(const Header& header, std::span<const std::uint32_t> users, std::array<std::uint32_t, 16>& output, std::uint32_t expected, const std::string& what) {
    OutputA.fill(0xdeadbeefu);
    OutputB.fill(0xdeadbeefu);
    std::vector<std::uint32_t> commands;
    for (const auto reg : header.registers) commands.insert(commands.end(), {0xc0017600u, reg.offset, reg.value});
    for (std::size_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x240u + static_cast<std::uint32_t>(index), users[index]});
    commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x41u | 0x8000u});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    sceAgcDriverSubmitAcb(0x20, &packet);
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(output.data()), sizeof(output));
    char text[160];
    std::snprintf(text, sizeof(text), "%s: stored 0x%08x, expected 0x%08x", what.c_str(), output[0], expected);
    Require(output[0] == expected, text);
}

// The table pointer in u[0:1] and two words the program never reads.
void RunTable(bool descriptors) {
    Header header;
    Register(header, TableCode, 4);
    const auto dispatch = [&](const Table& table, std::uint32_t unused, std::array<std::uint32_t, 16>& output, std::uint32_t expected, const std::string& what) {
        const auto pointer = Pointer(&table);
        const std::array<std::uint32_t, 4> users{pointer[0], pointer[1], unused, ~unused};
        Submit(header, users, output, expected, what);
    };
    Tables[0].output = Buffer(OutputA.data());
    Tables[0].value = 0x1111u;
    Tables[1] = Tables[0];
    Tables[2] = Tables[0];
    Tables[2].value = 0x2222u;
    Tables[3].output = Buffer(OutputB.data());
    Tables[3].value = 0x3333u;
    Tables[4] = Tables[3];

    const auto before = DispatchShapeReuseCounts();
    dispatch(Tables[0], 7, OutputA, 0x1111u, "first capture");
    auto counts = DispatchShapeReuseCounts();
    Require(counts.hits == before.hits && counts.remembered == before.remembered + 1, "the first capture was not remembered");

    dispatch(Tables[1], 9, OutputA, 0x1111u, "pointer moved to an identical table");
    counts = DispatchShapeReuseCounts();
    Require(counts.hits == before.hits + 1, "a pointer moved to identical data did not reuse the capture");

    dispatch(Tables[2], 11, OutputA, 0x2222u, "pointer moved to a table with another data word");
    counts = DispatchShapeReuseCounts();
    Require(counts.hits == before.hits + 2 && counts.dataRefreshes == before.dataRefreshes + 1, "another data word (a pure leaf) did not refresh a reuse");

    auto hitsBefore = counts.hits;
    const auto rememberedBefore = counts.remembered;
    const auto differing = Refused(ShapeRefusal::Differing);
    const auto descriptorRefreshes = counts.descriptorRefreshes;
    dispatch(Tables[3], 13, OutputB, 0x3333u, "pointer moved to a table with another descriptor");
    counts = DispatchShapeReuseCounts();
    if (descriptors) {
        Require(counts.hits == hitsBefore + 1 && counts.descriptorRefreshes == descriptorRefreshes + 1, "another descriptor word was not refreshed");
        hitsBefore = counts.hits;
    } else {
        Require(counts.hits == hitsBefore, "a pointer moved to different data reused the capture");
        Require(Refused(ShapeRefusal::Differing) == differing + 1, "different data was not refused as differing");
        Require(counts.remembered == rememberedBefore + 1, "the fallback capture was not remembered");
    }

    dispatch(Tables[4], 15, OutputB, 0x3333u, "pointer moved to a copy of the second table");
    counts = DispatchShapeReuseCounts();
    Require(counts.hits == hitsBefore + 1, "the second table's copy was not reused");

    // The table the last reuse read changes in place: an exact key repeat must not hit stale data.
    Tables[4].value = 0x4444u;
    dispatch(Tables[4], 15, OutputB, 0x4444u, "same pointer, table changed in place");
}

// The output V# in u[0:3] (its words are copied into the snapshot) and the data pointer in u[4:5].
void RunDirect() {
    Header header;
    Register(header, DirectCode, 6);
    const auto dispatch = [&](std::array<std::uint32_t, 16>& output, const Table& table, std::uint32_t expected, const std::string& what) {
        const auto descriptor = Buffer(output.data());
        const auto pointer = Pointer(&table.value);
        const std::array<std::uint32_t, 6> users{descriptor[0], descriptor[1], descriptor[2], descriptor[3], pointer[0], pointer[1]};
        Submit(header, users, output, expected, what);
    };
    Tables[5] = {};
    Tables[5].value = 0x5555u;
    Tables[4].value = 0x5555u;
    const auto before = DispatchShapeReuseCounts();
    dispatch(OutputA, Tables[5], 0x5555u, "direct V#: first capture");
    dispatch(OutputB, Tables[5], 0x5555u, "direct V#: output moved");
    auto counts = DispatchShapeReuseCounts();
    Require(counts.hits == before.hits + 1, "a moved output V# in user data did not reuse the capture");
    dispatch(OutputA, Tables[4], 0x5555u, "direct V#: output and pointer moved");
    counts = DispatchShapeReuseCounts();
    Require(counts.hits == before.hits + 2, "a moved output V# and data pointer did not reuse the capture");
}

// Arithmetic on the pointer word keeps it in the shape key: a moved pointer finds no entry.
void RunOffset() {
    Header header;
    Register(header, OffsetCode, 2);
    const auto dispatch = [&](const Table& table, std::array<std::uint32_t, 16>& output, std::uint32_t expected, const std::string& what) {
        // The program reads 16 bytes past its pointer.
        const auto pointer = Pointer(reinterpret_cast<const std::byte*>(&table) - 16);
        Submit(header, pointer, output, expected, what);
    };
    Tables[0].output = Buffer(OutputA.data());
    Tables[0].value = 0x6666u;
    Tables[1] = Tables[0];
    const auto noEntry = Refused(ShapeRefusal::NoEntry);
    const auto hits = DispatchShapeReuseCounts().hits;
    dispatch(Tables[0], OutputA, 0x6666u, "offset pointer: first capture");
    dispatch(Tables[1], OutputA, 0x6666u, "offset pointer: moved");
    Require(DispatchShapeReuseCounts().hits == hits, "a pointer word used in arithmetic was relocated");
    Require(Refused(ShapeRefusal::NoEntry) == noEntry + 1, "a pointer word used in arithmetic was left out of the shape key");
}

void Run(bool verify, bool descriptors) {
    RunTable(descriptors);
    RunDirect();
    RunOffset();
    const auto counts = DispatchShapeReuseCounts();
    if (verify) {
        Require(counts.verified == counts.hits && counts.verified != 0, "verify mode did not compare every reuse");
        Require(counts.mismatches == 0, "a reuse differs from the full capture");
    }
    std::printf("shape reuse: %llu lookups, %llu hits (%llu refreshed, %llu descriptors), %llu remembered, %llu verified, %llu mismatches\n", static_cast<unsigned long long>(counts.lookups), static_cast<unsigned long long>(counts.hits), static_cast<unsigned long long>(counts.dataRefreshes), static_cast<unsigned long long>(counts.descriptorRefreshes), static_cast<unsigned long long>(counts.remembered), static_cast<unsigned long long>(counts.verified), static_cast<unsigned long long>(counts.mismatches));
}

}

int main(int argc, char** argv) {
    bool verify = false, descriptors = false;
    for (int i = 1; i < argc; ++i) {
        verify = verify || std::strcmp(argv[i], "verify") == 0;
        descriptors = descriptors || std::strcmp(argv[i], "descriptors") == 0;
    }
    SetEnvironment("APS5_DISPATCH_SHAPE_REUSE");
    if (verify) SetEnvironment("APS5_VERIFY_SHAPE_REUSE");
    if (descriptors) SetEnvironment("APS5_DISPATCH_SHAPE_REFRESH_DESCRIPTORS");
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(verify, descriptors);
        AgcDriverShutdown_nid_postfix();
        std::puts(verify ? "dispatch shape reuse verify tests passed" : "dispatch shape reuse tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { AgcDriverShutdown_nid_postfix(); } catch (const std::exception&) {}
        return 1;
    }
}
