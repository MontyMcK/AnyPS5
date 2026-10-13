#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <iostream>

// A T# loaded by s_load through a 64-bit pointer from user data at `(key & 3) * 32 + 0x40`, the
// key a runtime user data dword: an image table with no V# around it. The entries are read when
// the dispatch is recorded; a null entry samples zeros and a repeated descriptor shares a slot.
// Changing the key or the table contents must not change the compiled artifact.

namespace {

using AgcDriver::Graphics::Require;
using namespace ShaderRecompiler;

struct alignas(4096) GuestData {
    std::array<std::uint32_t, 4> unused{};
    std::array<std::uint32_t, 4> sampler{};
    std::array<std::uint32_t, 4> unused2{};
    std::array<std::uint32_t, 4> output{};
    std::array<std::array<std::uint32_t, 8>, 4> table{};
    std::array<std::uint32_t, 64> values{};
};

alignas(256) std::array<std::array<std::uint32_t, 64>, 2> textures{};
std::array<GuestData, 8> guest;
constexpr std::array<std::uint32_t, 16> code{
    0xf4080700u, 0xfa000030u, // s_load_dwordx4 s[28:31], s[0:1], 0x30
    0xf4080200u, 0xfa000010u, // s_load_dwordx4 s[8:11], s[0:1], 0x10
    0x87038302u,              // s_and_b32 s3, s2, 3
    0x8f038503u,              // s_lshl_b32 s3, s3, 5
    0xf40c0500u, 0x06000040u, // s_load_dwordx8 s[20:27], s[0:1], s3 offset:0x40
    0xf09c0f08u, 0x00450000u, // image_sample_lz v[0:3], v[0:1], s[20:27], s[8:11] dmask:0xf dim:2D
    0xe0700000u, 0x80070000u, // buffer_store_dword v0, off, s[28:31]
    0xbf810000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

std::array<std::uint32_t, 4> Buffer(const void* pointer, std::uint32_t stride, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

std::array<std::uint32_t, 8> Texture(std::uint32_t index) {
    const auto address = reinterpret_cast<std::uintptr_t>(textures[index].data());
    return {static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u) | ((index == 0u ? 20u : 22u) << 20u) | (3u << 30u), 7u, 0x90000facu, 0u, 0u, 0u, 0u};
}

void Run(AgcDriver::VulkanDevice& device) {
    textures[0].fill(0x40000000u);
    textures[1].fill(0x40400000u);
    CompiledShaderArtifact artifact;
    for (std::uint32_t iteration = 0u; iteration < guest.size(); ++iteration) {
        auto& data = guest[iteration];
        // Entry 2 is null; entry 3 repeats entry 0. Odd iterations swap the textures.
        const auto first = iteration % 2u;
        data.table[0] = Texture(first);
        data.table[1] = Texture(1u - first);
        data.table[2] = {};
        data.table[3] = Texture(first);
        data.values.fill(0xdeadbeefu);
        data.output = Buffer(data.values.data(), 0u, 256u);
        const auto pointer = reinterpret_cast<std::uintptr_t>(&data);
        // Bits above the two the shader keeps must not matter.
        const auto key = (iteration / 2u) % 4u + (iteration << 8u);
        const std::array<std::uint32_t, 3> userData{static_cast<std::uint32_t>(pointer), static_cast<std::uint32_t>(pointer >> 32u), key};
        const std::array regions{MemoryRegion{pointer, std::as_bytes(std::span(&data, 1u))}};
        const ShaderComputeStageInfo compute{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0u, {}}, {32u, 0u, userData, compute, std::nullopt, std::nullopt, regions}, device.Target(), {0u, 0u, 0u, 128u}};
        const auto shader = Recompile(request);
        if (artifact.variantId == 0u) artifact = shader;
        else Require(shader.cacheHit && shader.variantId == artifact.variantId, "address image table key or contents changed the compiled artifact");
        device.Dispatch(shader, 1u, 1u, 1u);
        device.SubmitRecorded(false);
    }
    device.WaitIdle();
    for (std::uint32_t iteration = 0u; iteration < guest.size(); ++iteration) {
        const auto first = iteration % 2u;
        const std::array<std::uint32_t, 4> expected{textures[first][0], textures[1u - first][0], 0u, textures[first][0]};
        const auto want = expected[(iteration / 2u) % 4u];
        Require(guest[iteration].values[0] == want, "address image table selected the wrong entry at " + std::to_string(iteration) + " value=" + std::to_string(guest[iteration].values[0]) + " expected=" + std::to_string(want));
    }
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        std::cout << "address image table tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
