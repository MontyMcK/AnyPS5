#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

namespace {

using namespace ShaderRecompiler;
using AgcDriver::Graphics::Require;

struct alignas(4096) Data {
    std::array<std::array<std::uint32_t, 8>, 4> heap{};
    std::array<std::uint32_t, 132> material{};
    std::array<std::uint32_t, 16> srt{};
};

alignas(4096) std::array<std::array<std::uint32_t, 1024>, 2> textures{};
Data data;
alignas(256) constexpr std::array<std::uint32_t, 25> code{
    0xbf068002u, 0xbf850013u,
    0xf4080100u, 0xfa000000u, 0xf4080200u, 0xfa000010u,
    0xf4080300u, 0xfa000020u,
    0x7e200500u, 0x93109010u, 0xf4200406u, 0x20000004u,
    0x8f108510u, 0xf42c0502u, 0x20000000u,
    0xbf8cc07fu, 0x7e000280u, 0x7e020280u, 0x7e280203u,
    0xf0200108u, 0x00051400u,
    0xbf810000u, 0xbf800000u, 0xbf800000u, 0xbf800000u,
};

std::array<std::uint32_t, 4> Buffer(const void* pointer, std::uint32_t stride, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) | (stride << 16u), records, stride == 0u ? 0x31016facu : 0x01016facu};
}

void Run(AgcDriver::VulkanDevice& device) {
    GuestAllocations::Mutation().Add(textures.data(), sizeof(textures), true, true);
    const auto table = Buffer(data.heap.data(), 32u, 4u);
    const auto material = Buffer(data.material.data(), 16u, 33u);
    std::copy(table.begin(), table.end(), data.srt.begin());
    std::copy(material.begin(), material.end(), data.srt.begin() + 8u);
    const auto address = reinterpret_cast<std::uintptr_t>(data.srt.data());
    std::array<std::uint32_t, 4> users{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 1u, 0u};
    for (std::uint32_t i = 0; i < textures.size(); ++i) {
        const auto image = reinterpret_cast<std::uintptr_t>(textures[i].data());
        data.heap[i] = {static_cast<std::uint32_t>(image >> 8u), static_cast<std::uint32_t>(image >> 40u) | (20u << 20u) | (3u << 30u), 7u, 0x90000facu, 0u, 0u, 0u, 0u};
        textures[i].fill(0x3f800000u);
    }
    const std::array regions{MemoryRegion{reinterpret_cast<std::uintptr_t>(&data), std::as_bytes(std::span(&data, 1u))}};
    const ShaderComputeStageInfo compute{{1u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    const std::array<std::uint32_t, 5> keys{0u, 1u, 2u, 0xffffffffu, 0u};
    CompiledShaderArtifact artifact;
    for (std::uint32_t iteration = 0; iteration < keys.size(); ++iteration) {
        data.material[1] = keys[iteration];
        users[2] = iteration == 4u ? 0u : 1u;
        users[3] = 0x40000000u + iteration * 0x00800000u;
        RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0u, {}}, {32u, 0u, users, compute, std::nullopt, std::nullopt, regions}, device.Target(), {0u, 0u, 0u, 128u}};
        const auto shader = Recompile(request);
        if (artifact.variantId == 0u) artifact = shader;
        else Require(shader.cacheHit && shader.variantId == artifact.variantId, "bindless store keys changed the compiled artifact");
        device.Dispatch(shader, 1u, 1u, 1u);
        device.WaitIdle();
        AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(textures.data()), sizeof(textures), nullptr, "test");
        device.WaitIdle();
        for (std::uint32_t image = 0; image < textures.size(); ++image) {
            Require(textures[image][0] == (iteration >= image ? 0x40000000u + image * 0x00800000u : 0x3f800000u), "bindless image store selected the wrong texture");
            Require(std::all_of(textures[image].begin() + 1, textures[image].end(), [](auto value) { return value == 0x3f800000u; }), "bindless image store changed untouched texels");
        }
    }
    GuestAllocations::Mutation().Remove(textures.data());
}

void RunDivergent(AgcDriver::VulkanDevice& device, std::uint32_t waveSize) {
    GuestAllocations::Mutation().Add(textures.data(), sizeof(textures), true, true);
    data.material.fill(0u);
    data.material[129] = 1u;
    const auto address = reinterpret_cast<std::uintptr_t>(data.srt.data());
    const std::array<std::uint32_t, 4> users{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), 1u, 0x40800000u};
    for (std::uint32_t image = 0; image < textures.size(); ++image) {
        textures[image].fill(0x3f800000u);
        data.heap[image][2] = 63u;
    }
    auto divergentCode = code;
    divergentCode[16] = 0xbf800000u;
    const std::array regions{MemoryRegion{reinterpret_cast<std::uintptr_t>(&data), std::as_bytes(std::span(&data, 1u))}};
    const ShaderComputeStageInfo compute{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    RecompileRequest request{{ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(divergentCode.data()), divergentCode, 0u, {}}, {waveSize, 0u, users, compute, std::nullopt, std::nullopt, regions}, device.Target(), {0u, 0u, 0u, 128u}};
    device.Dispatch(Recompile(request), 1u, 1u, 1u);
    device.WaitIdle();
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(textures.data()), sizeof(textures), nullptr, "test");
    device.WaitIdle();
    for (std::uint32_t image = 0; image < textures.size(); ++image) {
        for (std::uint32_t texel = 0; texel < textures[image].size(); ++texel) {
            const bool selected = texel < 64u && texel / waveSize == image;
            Require(textures[image][texel] == (selected ? 0x40800000u : 0x3f800000u), "divergent bindless store selected the wrong texture or texel");
        }
    }
    GuestAllocations::Mutation().Remove(textures.data());
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        Run(*device);
        RunDivergent(*device, 32u);
        RunDivergent(*device, 64u);
        std::cout << "bindless image stores passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
