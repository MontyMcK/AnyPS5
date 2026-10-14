#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/ShapeReuse.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Draw shape reuse (APS5_DRAW_SHAPE_REUSE). A vertex program loads its position V# through a user
// pointer and pixel programs load their colors through others; every draw must render what its
// own tables say, and the counters say which stages reused an earlier capture:
// - pointers moved to copies of the same tables reuse both stages; another color (a pure flat-slot
//   leaf) refreshes the pixel stage; another V# (a descriptor leaf) refreshes the vertex stage, or
//   with `keep` (APS5_DRAW_SHAPE_KEEP_DESCRIPTORS) falls back to a full capture;
// - one table word read by both stages, and twice by one stage;
// - two pointers moved onto one table whose earlier reads held different words (a conflict);
// - a nested pointer that differs refuses the newest remembered capture and reuses an older one;
// - a pointer moved into a registered region (the pixel header) and out again;
// - a color table a dispatch in the same submission writes (pending GPU writes);
// - a pointer word used in arithmetic stays in the shape key (the pixel stage finds no entry), and
//   a program whose captures keep missing stops being learned (Table::ShouldLearn).
// With `verify` (APS5_VERIFY_DRAW_SHAPE_REUSE) every captured stage is also captured in full on a
// shadow memory and compared, reused or not.

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::DriverDetail::DrawShapeReuseCounts;
using AgcDriver::DriverDetail::ShapeRefusal;
using AgcDriver::DriverDetail::ShapeReuseCounts;

void SetEnvironment(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "1");
#else
    setenv(name, "1", 1);
#endif
}

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 4;
alignas(256) std::array<std::uint32_t, Width * Height> ColorTarget{};

// Full-screen triangle positions (x, y, z, w), each a copy for a moved buffer.
struct alignas(256) Positions {
    std::array<std::array<float, 4>, 3> vertices{{{-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}}};
};
std::array<Positions, 2> PositionBuffers;

// The vertex program's table: the position V# at words 4..7 (words 0..3 lie before it so a pixel
// pointer can point 16 bytes ahead of a V# word).
struct alignas(256) VertexTable {
    std::array<std::uint32_t, 4> before{};
    std::array<std::uint32_t, 4> positions{};
    std::array<std::uint32_t, 56> padding{};
};
std::array<VertexTable, 8> VertexTables;

// A pixel program's table: a color at 0x10, or a nested pointer at 0x0.
struct alignas(256) ColorTable {
    std::array<std::uint32_t, 2> pointer{};
    std::array<std::uint32_t, 2> unused{};
    float color = 0.0f;
    std::array<std::uint32_t, 59> padding{};
};
std::array<ColorTable, 40> ColorTables;

// The table pointer in u[0:1] (s[8:9]): the V# at 0x10 of it, the vertex index in v5.
alignas(256) constexpr std::array<std::uint32_t, 16> VertexCode{
    0xf4080304u, 0xfa000010u, // s_load_dwordx4 s[12:15], s[8:9], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0xe0382000u, 0x80030005u, // buffer_load_format_xyzw v[0:3], v5, s[12:15], 0 idxen
    0xbf8c3f70u,              // s_waitcnt vmcnt(0)
    0xf80008cfu, 0x03020100u, // exp pos0 v0, v1, v2, v3 done
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// VertexCode that also exports its table's first word as a parameter (a pure leaf the pixel program
// reads as well).
alignas(256) constexpr std::array<std::uint32_t, 16> SharedVertexCode{
    0xf4080304u, 0xfa000010u, // s_load_dwordx4 s[12:15], s[8:9], 0x10
    0xf4000404u, 0xfa000000u, // s_load_dword s16, s[8:9], 0x0
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0xe0382000u, 0x80030005u, // buffer_load_format_xyzw v[0:3], v5, s[12:15], 0 idxen
    0x7e080210u,              // v_mov_b32 v4, s16
    0xbf8c3f70u,              // s_waitcnt vmcnt(0)
    0xf800020fu, 0x04040404u, // exp param0 v4, v4, v4, v4
    0xf80008cfu, 0x03020100u, // exp pos0 v0, v1, v2, v3 done
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u,
};

// The table pointer in u[0:1] (s[0:1]); u[2:3] are never read.
alignas(256) constexpr std::array<std::uint32_t, 16> PixelCode{
    0xf4000100u, 0xfa000010u, // s_load_dword s4, s[0:1], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e080204u,              // v_mov_b32 v4, s4
    0x7e0a0204u,              // v_mov_b32 v5, s4
    0x7e0c0204u,              // v_mov_b32 v6, s4
    0x7e0e0204u,              // v_mov_b32 v7, s4
    0xf800180fu, 0x07060504u, // exp mrt0 v4, v5, v6, v7 done vm
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// The color word loaded twice: red, green and blue from the first load, alpha from the second.
alignas(256) constexpr std::array<std::uint32_t, 16> TwicePixelCode{
    0xf4000100u, 0xfa000010u, // s_load_dword s4, s[0:1], 0x10
    0xf4000140u, 0xfa000010u, // s_load_dword s5, s[0:1], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e080204u,              // v_mov_b32 v4, s4
    0x7e0a0204u,              // v_mov_b32 v5, s4
    0x7e0c0204u,              // v_mov_b32 v6, s4
    0x7e0e0205u,              // v_mov_b32 v7, s5
    0xf800180fu, 0x07060504u, // exp mrt0 v4, v5, v6, v7 done vm
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// Two table pointers: red, green and blue from u[0:1], alpha from u[2:3].
alignas(256) constexpr std::array<std::uint32_t, 16> TwoPointerPixelCode{
    0xf4000100u, 0xfa000010u, // s_load_dword s4, s[0:1], 0x10
    0xf4000141u, 0xfa000010u, // s_load_dword s5, s[2:3], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e080204u,              // v_mov_b32 v4, s4
    0x7e0a0204u,              // v_mov_b32 v5, s4
    0x7e0c0204u,              // v_mov_b32 v6, s4
    0x7e0e0205u,              // v_mov_b32 v7, s5
    0xf800180fu, 0x07060504u, // exp mrt0 v4, v5, v6, v7 done vm
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// A nested pointer: the color at 0x10 of the table u[0:1]'s first two words point at.
alignas(256) constexpr std::array<std::uint32_t, 16> NestedPixelCode{
    0xf4040100u, 0xfa000000u, // s_load_dwordx2 s[4:5], s[0:1], 0x0
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0xf4000182u, 0xfa000010u, // s_load_dword s6, s[4:5], 0x10
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e080206u,              // v_mov_b32 v4, s6
    0x7e0a0206u,              // v_mov_b32 v5, s6
    0x7e0c0206u,              // v_mov_b32 v6, s6
    0x7e0e0206u,              // v_mov_b32 v7, s6
    0xf800180fu, 0x07060504u, // exp mrt0 v4, v5, v6, v7 done vm
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u,
};

// The pointer moved by 16 bytes before the load: arithmetic on the pointer word.
alignas(256) constexpr std::array<std::uint32_t, 16> OffsetPixelCode{
    0x80009000u,              // s_add_u32 s0, s0, 16
    0x82018001u,              // s_addc_u32 s1, s1, 0
    0xf4000100u, 0xfa000000u, // s_load_dword s4, s[0:1], 0x0
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e080204u,              // v_mov_b32 v4, s4
    0x7e0a0204u,              // v_mov_b32 v5, s4
    0x7e0c0204u,              // v_mov_b32 v6, s4
    0x7e0e0204u,              // v_mov_b32 v7, s4
    0xf800180fu, 0x07060504u, // exp mrt0 v4, v5, v6, v7 done vm
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u,
};

// A compute program storing the word at u[4:5] through the buffer V# in u[0:3].
alignas(256) constexpr std::array<std::uint32_t, 16> StoreCode{
    0xf4000202u, 0xfa000000u, // s_load_dword s8, s[4:5], 0x0
    0xbf8cc07fu,              // s_waitcnt lgkmcnt(0)
    0x7e020208u,              // v_mov_b32 v1, s8
    0xe0700000u, 0x80000100u, // buffer_store_dword v1, off, s[0:3], 0
    0xbf810000u,              // s_endpgm
    0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

std::array<std::uint32_t, 4> PositionDescriptor(const Positions& positions, std::uint32_t records = 3u) {
    const auto address = reinterpret_cast<std::uintptr_t>(positions.vertices.data());
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (16u << 16u), records, 0x01016facu};
}

std::array<std::uint32_t, 2> Pointer(const void* pointer) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
}

std::vector<std::uint32_t> Words(std::initializer_list<const void*> pointers) {
    std::vector<std::uint32_t> words;
    for (const auto* pointer : pointers) {
        const auto pair = Pointer(pointer);
        words.insert(words.end(), pair.begin(), pair.end());
    }
    return words;
}

// The pixel byte of a color.
constexpr std::uint32_t Gray(std::uint8_t value) {
    return value * 0x01010101u;
}

struct Header {
    Shader shader{};
    std::array<ShaderRegister, 7> registers{};
    std::uint32_t registerCount = 0;
    ShaderUserData userData{};
    ShaderSpecialRegs specials{};
    // Registered with the header: a color table inside a registered region (color at data[4]).
    alignas(16) std::array<std::uint32_t, 8> data{};
};

// Registers a vertex (type 2) or pixel (type 1) program with `userWords` user SGPRs.
void Register(Header& header, std::span<const std::uint32_t> code, std::uint8_t type, std::uint32_t userWords) {
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    const bool vertex = type == 2;
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = code.data();
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    header.shader.sh_registers = header.registers.data();
    const std::uint32_t program = vertex ? 0xc8u : 0x8u;
    const std::uint32_t resources = vertex ? 0x8bu : 0xbu;
    header.registers[0] = {program, static_cast<std::uint32_t>(address >> 8u)};
    header.registers[1] = {program + 1u, static_cast<std::uint32_t>(address >> 40u)};
    header.registers[2] = {resources, userWords << 1u};
    header.registerCount = 3;
    header.shader.num_sh_registers = static_cast<std::uint8_t>(header.registerCount);
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void RegisterCompute(Header& header, std::span<const std::uint32_t> code, std::uint32_t userWords) {
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(header);
    header.shader.shader_size = static_cast<std::uint32_t>(code.size_bytes());
    header.shader.code = code.data();
    header.shader.sh_registers = header.registers.data();
    header.shader.specials = &header.specials;
    header.specials.dispatch_modifier = 0x8000u;
    header.registers = {{{0x20c, static_cast<std::uint32_t>(address >> 8u)}, {0x20d, static_cast<std::uint32_t>(address >> 40u)},
        {0x207, 1}, {0x208, 1}, {0x209, 1}, {0x212, 0}, {userWords == 0 ? 0x213u : 0x213u, userWords << 1u}}};
    header.registerCount = 7;
    header.shader.num_sh_registers = 7;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
    AgcDriverResolveShaderAbi_nid_postfix(&header.shader, {}, {});
}

// The context state of a 64x4 RGBA8 linear color target with a full viewport (Graphics.cpp's
// makeState).
std::vector<std::pair<std::uint32_t, std::uint32_t>> ContextState() {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> state{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, ((Width - 1u) << 14u) | (Height - 1u)},
        {0x3b8, 0x9000000}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
    };
    const auto address = reinterpret_cast<std::uintptr_t>(ColorTarget.data());
    state.emplace_back(0x318, static_cast<std::uint32_t>(address >> 8u));
    state.emplace_back(0x390, static_cast<std::uint32_t>(address >> 40u));
    return state;
}

std::uint64_t Refused(const ShapeReuseCounts& counts, ShapeRefusal reason) {
    return counts.refusals[static_cast<std::size_t>(reason)];
}

// A dispatch of StoreCode run before a draw in the same submission: stores `*value` at `target`.
struct Store {
    const Header* header = nullptr;
    void* target = nullptr;
    const std::uint32_t* value = nullptr;
};

// One draw of a full-screen triangle with the vertex table `vertex` (a pointer to its `before`
// words: the V# is read 16 bytes on) and the pixel user words `pixelWords`; every pixel must hold
// `expected`. Returns the change in the draw shape counters.
struct Delta {
    ShapeReuseCounts before, after;
    std::uint64_t Hits() const { return after.hits - before.hits; }
    std::uint64_t Lookups() const { return after.lookups - before.lookups; }
    std::uint64_t Refreshes() const { return after.dataRefreshes - before.dataRefreshes; }
    std::uint64_t Descriptors() const { return after.descriptorRefreshes - before.descriptorRefreshes; }
    std::uint64_t Refusals(ShapeRefusal reason) const { return Refused(after, reason) - Refused(before, reason); }
    std::uint64_t Remembered() const { return after.remembered - before.remembered; }
    std::uint64_t Learned() const { return after.remembered + after.notRemembered - before.remembered - before.notRemembered; }
};

Delta Draw(const Header& vertexHeader, const Header& pixelHeader, const void* vertex, const std::vector<std::uint32_t>& pixelWords, std::uint32_t expected, const std::string& what, const Store& store = {}) {
    Delta delta{DrawShapeReuseCounts(), {}};
    ColorTarget.fill(0x5a5a5a5au);
    std::vector<std::uint32_t> commands;
    if (store.header != nullptr) {
        const auto& compute = *store.header;
        for (std::uint32_t r = 0; r < compute.registerCount; ++r) commands.insert(commands.end(), {0xc0017600u, compute.registers[r].offset, compute.registers[r].value});
        const auto target = reinterpret_cast<std::uintptr_t>(store.target);
        const auto value = Pointer(store.value);
        const std::array<std::uint32_t, 6> users{static_cast<std::uint32_t>(target), static_cast<std::uint32_t>(target >> 32u) & 0xffffu, 4u, 0x31016facu, value[0], value[1]};
        for (std::size_t index = 0; index < users.size(); ++index) commands.insert(commands.end(), {0xc0017600u, 0x240u + static_cast<std::uint32_t>(index), users[index]});
        commands.insert(commands.end(), {0xc0031500u, 1, 1, 1, 0x41u | 0x8000u});
    }
    for (const auto& [offset, value] : ContextState()) commands.insert(commands.end(), {0xc0016900u, offset, value});
    commands.insert(commands.end(), {0xc0017900u, 0x242u, 4u});
    for (const auto* header : {&vertexHeader, &pixelHeader}) {
        for (std::uint32_t r = 0; r < header->registerCount; ++r) commands.insert(commands.end(), {0xc0017600u, header->registers[r].offset, header->registers[r].value});
    }
    const auto vertexPointer = Pointer(vertex);
    commands.insert(commands.end(), {0xc0027600u, 0x8cu, vertexPointer[0], vertexPointer[1]});
    commands.push_back(0xc0007600u | (static_cast<std::uint32_t>(pixelWords.size()) << 16u));
    commands.push_back(0xcu);
    commands.insert(commands.end(), pixelWords.begin(), pixelWords.end());
    commands.insert(commands.end(), {0xc0012d00u, 3u, 2u});
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, what + ": submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(ColorTarget.data()), sizeof(ColorTarget));
    for (std::size_t pixel = 0; pixel < ColorTarget.size(); ++pixel) {
        if (ColorTarget[pixel] == expected) continue;
        char text[192];
        std::snprintf(text, sizeof(text), "%s: pixel %zu holds 0x%08x, expected 0x%08x", what.c_str(), pixel, ColorTarget[pixel], expected);
        Require(false, text);
    }
    delta.after = DrawShapeReuseCounts();
    return delta;
}

void Expect(bool condition, const std::string& what, const Delta& delta) {
    if (condition) return;
    char text[320];
    std::snprintf(text, sizeof(text), "%s (this draw: %llu lookups, %llu hits, %llu refreshed, %llu descriptors, %llu remembered; refused: no entry %llu, differing %llu, conflict %llu)", what.c_str(), static_cast<unsigned long long>(delta.Lookups()), static_cast<unsigned long long>(delta.Hits()), static_cast<unsigned long long>(delta.Refreshes()), static_cast<unsigned long long>(delta.Descriptors()), static_cast<unsigned long long>(delta.Remembered()), static_cast<unsigned long long>(delta.Refusals(ShapeRefusal::NoEntry)), static_cast<unsigned long long>(delta.Refusals(ShapeRefusal::Differing)), static_cast<unsigned long long>(delta.Refusals(ShapeRefusal::Conflict)));
    Require(false, text);
}

Header VertexHeader;

void SetVertexTable(VertexTable& table, const Positions& positions, std::uint32_t records = 3u) {
    table.positions = PositionDescriptor(positions, records);
}

void RunTables(bool descriptors) {
    Header pixelHeader;
    Register(pixelHeader, PixelCode, 1, 4);
    SetVertexTable(VertexTables[0], PositionBuffers[0]);
    VertexTables[1] = VertexTables[0];
    SetVertexTable(VertexTables[2], PositionBuffers[1]);
    VertexTables[3] = VertexTables[2];
    ColorTables[0].color = 1.0f;
    ColorTables[1] = ColorTables[0];
    ColorTables[2].color = 0.2f;
    ColorTables[3].color = 0.6f;
    ColorTables[4] = ColorTables[3];
    ColorTables[5].color = 0.2f;
    const auto words = [](const ColorTable& table, std::uint32_t unused) { auto w = Words({&table}); w.insert(w.end(), {unused, ~unused}); return w; };

    auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], words(ColorTables[0], 7), Gray(255), "first draw");
    Expect(d.Hits() == 0 && d.Remembered() == 2, "the first draw's stages were not remembered", d);
    d = Draw(VertexHeader, pixelHeader, &VertexTables[1], words(ColorTables[1], 9), Gray(255), "both pointers moved to identical tables");
    Expect(d.Hits() == 2 && d.Refreshes() == 0, "pointers moved to identical tables did not reuse both stages", d);
    d = Draw(VertexHeader, pixelHeader, &VertexTables[0], words(ColorTables[2], 11), Gray(51), "color pointer moved to another color");
    Expect(d.Hits() == 2 && d.Refreshes() == 1, "another color (a pure leaf) did not refresh a reuse", d);
    d = Draw(VertexHeader, pixelHeader, &VertexTables[2], words(ColorTables[3], 13), Gray(153), "vertex pointer moved to another V#");
    if (descriptors) {
        Expect(d.Hits() == 2 && d.Descriptors() == 1, "another V# (a descriptor leaf) was not refreshed", d);
    } else {
        Expect(d.Hits() == 1 && d.Refusals(ShapeRefusal::Differing) == 1 && d.Remembered() == 1, "a vertex table with another V# was not refused as differing and captured again", d);
    }
    d = Draw(VertexHeader, pixelHeader, &VertexTables[3], words(ColorTables[4], 15), Gray(153), "pointers moved to copies of the last tables");
    Expect(d.Hits() == 2, "copies of the last tables were not reused", d);
    // The table the last reuse read changes in place, then a moved copy of the old contents: both
    // draws must render their own color, the second by reuse.
    ColorTables[4].color = 0.2f;
    Draw(VertexHeader, pixelHeader, &VertexTables[3], words(ColorTables[4], 15), Gray(51), "same pointers, color changed in place");
    d = Draw(VertexHeader, pixelHeader, &VertexTables[2], words(ColorTables[5], 17), Gray(51), "pointers moved again after an in-place change");
    Expect(d.Hits() == 2, "pointers moved after an in-place change did not reuse both stages", d);
}

// One table word read by both stages (the vertex program exports it, the pixel program renders it)
// and twice by the pixel program.
void RunShared() {
    Header vertexHeader, pixelHeader;
    Register(vertexHeader, SharedVertexCode, 2, 2);
    Register(pixelHeader, TwicePixelCode, 1, 2);
    SetVertexTable(VertexTables[4], PositionBuffers[0]);
    VertexTables[4].before[0] = std::bit_cast<std::uint32_t>(1.0f);
    VertexTables[5] = VertexTables[4];
    VertexTables[6] = VertexTables[4];
    VertexTables[6].before[0] = std::bit_cast<std::uint32_t>(0.2f);
    // The pixel program reads 0x10 past its pointer: the table's first word.
    const auto pixel = [](const VertexTable& table) { return Words({reinterpret_cast<const std::byte*>(&table) - 16}); };
    auto d = Draw(vertexHeader, pixelHeader, &VertexTables[4], pixel(VertexTables[4]), Gray(255), "shared word: first draw");
    Expect(d.Hits() == 0 && d.Remembered() == 2, "the shared word's stages were not remembered", d);
    d = Draw(vertexHeader, pixelHeader, &VertexTables[5], pixel(VertexTables[5]), Gray(255), "shared word: moved to a copy");
    Expect(d.Hits() == 2 && d.Refreshes() == 0, "a table word read by both stages was not reused by both", d);
    d = Draw(vertexHeader, pixelHeader, &VertexTables[6], pixel(VertexTables[6]), Gray(51), "shared word: moved to another word");
    Expect(d.Hits() == 2 && d.Refreshes() == 2, "a shared word that changed was not refreshed in both stages", d);
}

// Two pointers whose remembered reads held different words, moved onto one table: the reads now
// read one word, so the remembered capture cannot be the walk's.
void RunConflict() {
    Header pixelHeader;
    Register(pixelHeader, TwoPointerPixelCode, 1, 4);
    ColorTables[6].color = 1.0f;
    ColorTables[7].color = 0.2f;
    ColorTables[8].color = 0.6f;
    Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[6], &ColorTables[7]}), 0x33ffffffu, "conflict: first draw");
    const auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[8], &ColorTables[8]}), Gray(153), "conflict: both pointers on one table");
    Expect(d.Lookups() == d.Hits() + 1 && d.Refusals(ShapeRefusal::Conflict) == 1, "two reads moved onto one word were not refused as a conflict", d);
}

// A nested pointer that differs: the newest remembered capture is refused (differing), the older one
// it falls through to is reused.
void RunFallThrough() {
    Header pixelHeader;
    Register(pixelHeader, NestedPixelCode, 1, 2);
    ColorTables[9].color = 1.0f;
    ColorTables[10].color = 0.2f;
    const auto nest = [](ColorTable& table, const ColorTable& target) { table.pointer = Pointer(&target); };
    nest(ColorTables[11], ColorTables[9]);
    nest(ColorTables[12], ColorTables[10]);
    nest(ColorTables[13], ColorTables[9]);
    Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[11]}), Gray(255), "nested: first draw");
    auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[12]}), Gray(51), "nested: another nested pointer");
    Expect(d.Lookups() == d.Hits() + 1 && d.Refusals(ShapeRefusal::Differing) == 1 && d.Remembered() == 1, "a nested pointer that differs was not refused and captured again", d);
    d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[13]}), Gray(255), "nested: the first nested pointer again");
    Expect(d.Lookups() >= 1 && d.Hits() == d.Lookups() && d.Refusals(ShapeRefusal::Differing) == 0, "the older remembered capture was not reused after the newest was refused", d);
}

// The pointer moved into a registered region (the pixel header's data, served from its snapshot)
// and out again.
void RunRegistered() {
    static Header pixelHeader;
    pixelHeader.data[4] = std::bit_cast<std::uint32_t>(1.0f);
    Register(pixelHeader, PixelCode, 1, 2);
    ColorTables[14].color = 0.2f;
    ColorTables[15].color = 0.6f;
    Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[14]}), Gray(51), "registered: first draw");
    auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({pixelHeader.data.data()}), Gray(255), "registered: pointer into the registered header");
    Expect(d.Lookups() >= 1 && d.Hits() == d.Lookups() && d.Refreshes() == 1, "a read moved into a registered region was not reused with its word refreshed", d);
    d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[15]}), Gray(153), "registered: pointer out of the registered header");
    Expect(d.Lookups() >= 1 && d.Hits() == d.Lookups(), "a read moved out of a registered region was not reused", d);
}

// A color table a dispatch earlier in the submission writes: the reuse must read the stored word.
void RunPending() {
    Header pixelHeader, computeHeader;
    Register(pixelHeader, PixelCode, 1, 2);
    RegisterCompute(computeHeader, StoreCode, 6);
    static std::uint32_t stored = std::bit_cast<std::uint32_t>(1.0f);
    ColorTables[16].color = 0.2f;
    ColorTables[17].color = 0.6f;
    Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[16]}), Gray(51), "pending: first draw");
    const auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[17]}), Gray(255), "pending: color stored by a dispatch", {&computeHeader, &ColorTables[17].color, &stored});
    Expect(d.Lookups() >= 1 && d.Hits() == d.Lookups() && d.Refreshes() == 1, "a table with a pending GPU write was not reused with the stored word", d);
}

// Arithmetic on the pixel program's pointer word keeps it in the shape key: a moved pointer finds
// no entry for the pixel stage (the vertex stage, whose user data repeats, reuses a capture or the
// draw cache's data entry), and a program that keeps finding no
// entry stops being learned.
void RunOffset() {
    Header pixelHeader;
    Register(pixelHeader, OffsetPixelCode, 1, 2);
    for (std::size_t i = 18; i < ColorTables.size(); ++i) ColorTables[i].color = i % 2 == 0 ? 1.0f : 0.2f;
    Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[18]}), Gray(255), "offset pointer: first draw");
    auto d = Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[19]}), Gray(51), "offset pointer: moved");
    Expect(d.Lookups() == d.Hits() + 1 && d.Refusals(ShapeRefusal::NoEntry) == 1, "a pointer word used in arithmetic was left out of the shape key (or the pixel stage hit)", d);
    // Draws 3 to 14 of this program: after 8 lookups in a row without an entry, its captures are
    // learned only every 64th time.
    std::uint64_t learned = 0;
    for (std::size_t table = 20; table < 32; ++table) learned += Draw(VertexHeader, pixelHeader, &VertexTables[0], Words({&ColorTables[table]}), table % 2 == 0 ? Gray(255) : Gray(51), "offset pointer: repeated").Learned();
    char text[96];
    std::snprintf(text, sizeof(text), "a program that keeps finding no entry was learned %llu times in 12 draws", static_cast<unsigned long long>(learned));
    Require(learned <= 7, text);
}

template<typename TAction>
void ExpectFailure(TAction action, const char* expected, const std::string& message) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(expected) != std::string::npos, message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message);
}

// The reads a reuse makes (ShaderMemory::ReadWord): a read is marked in Regions() and
// TakeRecentRegions() like a walk's, a registered word is served from the initial regions without a
// mark, and EndTentative(false) removes exactly the marks the tentative reads added (a word read
// before keeps its mark, a failed read adds none).
void RunTentativeReads() {
    alignas(4096) static std::array<std::uint32_t, 2048> words{};
    for (std::size_t i = 0; i < words.size(); ++i) words[i] = 0x1000u + static_cast<std::uint32_t>(i);
    alignas(256) static const std::array<std::uint32_t, 4> code{0xbf810000u, 0x12345678u, 0xbf800000u, 0xbf800000u};
    const auto base = reinterpret_cast<std::uint64_t>(words.data());
    const auto codeAddress = reinterpret_cast<std::uint64_t>(code.data());
    const std::array<ShaderRecompiler::MemoryRegion, 1> initial{{{codeAddress, std::as_bytes(std::span(code))}}};
    using Runs = std::vector<std::pair<std::uint64_t, std::size_t>>;
    const auto runs = [](const std::vector<ShaderRecompiler::MemoryRegion>& regions) {
        Runs out;
        for (const auto& region : regions) out.emplace_back(region.guestAddress, region.bytes.size());
        return out;
    };
    AgcDriver::ShaderMemory memory(initial);
    Require(memory.ReadWord(base + 8) == words[2], "tentative reads: a read returned another word");
    Require(runs(memory.TakeRecentRegions()) == Runs{{base + 8, 4}}, "tentative reads: a read was not a recent region");
    const auto before = runs(memory.Regions());

    memory.BeginTentative();
    Require(memory.ReadWord(base + 8) == words[2] && memory.ReadWord(base + 12) == words[3] && memory.ReadWord(base + 4096) == words[1024], "tentative reads: a tentative read returned another word");
    Require(memory.ReadWord(codeAddress + 4) == code[1], "tentative reads: a registered word was not served from the initial region");
    ExpectFailure([&] { static_cast<void>(memory.ReadWord(base + 2)); }, "invalid shader memory read address", "tentative reads: a misaligned read was accepted");
    memory.EndTentative(false);
    Require(memory.TakeRecentRegions().empty(), "tentative reads: undone reads stayed recent");
    Require(runs(memory.Regions()) == before, "tentative reads: undone reads stayed in the regions");

    memory.BeginTentative();
    Require(memory.ReadWord(base + 12) == words[3], "tentative reads: a kept read returned another word");
    memory.EndTentative(true);
    Require(runs(memory.TakeRecentRegions()) == Runs{{base + 12, 4}}, "tentative reads: a kept read was not recent");
    const auto kept = runs(memory.Regions());
    Require(kept.size() == 2 && std::find(kept.begin(), kept.end(), std::pair<std::uint64_t, std::size_t>{base + 8, 8}) != kept.end() && std::find(kept.begin(), kept.end(), std::pair<std::uint64_t, std::size_t>{codeAddress, sizeof(code)}) != kept.end(), "tentative reads: a kept read did not join the regions");
}


void Run(bool verify, bool descriptors) {
    RunTentativeReads();
    Register(VertexHeader, VertexCode, 2, 2);
    RunTables(descriptors);
    RunShared();
    RunConflict();
    RunFallThrough();
    RunRegistered();
    RunPending();
    RunOffset();
    const auto counts = DrawShapeReuseCounts();
    if (verify) {
        Require(counts.verified == counts.lookups && counts.verified != 0, "verify mode did not compare every looked-up stage");
        Require(counts.mismatches == 0, "a stage differs from its full capture");
    } else {
        Require(counts.verified == 0, "stages were compared without verify mode");
    }
    std::printf("draw shape reuse: %llu lookups, %llu hits (%llu refreshed, %llu descriptors), %llu remembered, %llu not remembered, %llu full captures, %llu verified, %llu mismatches; %.1f us per hit vs %.1f us per full capture\n", static_cast<unsigned long long>(counts.lookups), static_cast<unsigned long long>(counts.hits), static_cast<unsigned long long>(counts.dataRefreshes), static_cast<unsigned long long>(counts.descriptorRefreshes), static_cast<unsigned long long>(counts.remembered), static_cast<unsigned long long>(counts.notRemembered), static_cast<unsigned long long>(counts.fullCaptures), static_cast<unsigned long long>(counts.verified), static_cast<unsigned long long>(counts.mismatches), counts.hits != 0 ? static_cast<double>(counts.hitNanoseconds) / 1000.0 / static_cast<double>(counts.hits) : 0.0, counts.fullCaptures != 0 ? static_cast<double>(counts.fullNanoseconds) / 1000.0 / static_cast<double>(counts.fullCaptures) : 0.0);
}

}

int main(int argc, char** argv) {
    bool verify = false, keep = false;
    for (int i = 1; i < argc; ++i) {
        verify = verify || std::strcmp(argv[i], "verify") == 0;
        keep = keep || std::strcmp(argv[i], "keep") == 0;
    }
    SetEnvironment("APS5_DRAW_SHAPE_REUSE");
    if (verify) SetEnvironment("APS5_VERIFY_DRAW_SHAPE_REUSE");
    if (keep) SetEnvironment("APS5_DRAW_SHAPE_KEEP_DESCRIPTORS");
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(verify, !keep);
        AgcDriverShutdown_nid_postfix();
        std::puts(verify ? "draw shape reuse verify tests passed" : "draw shape reuse tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        try { AgcDriverShutdown_nid_postfix(); } catch (const std::exception&) {}
        return 1;
    }
}
