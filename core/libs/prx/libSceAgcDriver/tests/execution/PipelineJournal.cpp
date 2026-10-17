#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Pipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PipelineJournal.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "../NarrowConstantStoreFixture.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

// The pipeline journal on a device: a first device records the graphics pipeline of a draw and the
// compute pipeline of a dispatch; a second device replays them before its first draw and dispatch,
// which then find them resident (the store hit, no compute pipeline built) and render the same.
// `--benchmark [count]` times `count` distinct compute programs' first dispatches on a cold device,
// a device that replayed the journal, and a warm device without the journal.

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Width = 64;
constexpr std::uint32_t Height = 16;
constexpr std::uint32_t WaveSize = 64;
constexpr std::size_t BlockBytes = 65536;
constexpr std::byte Kept{0x40};
alignas(256) std::array<std::byte, Width * Height * 4> Pixels{};

alignas(256) constexpr std::array<std::uint32_t, 6> VertexCode{
    0xe0382000, 0x80000005, 0xbf8c3f70, 0xf80008cf, 0x03020100, 0xbf810000,
};

// v7 = 1.0, exported to every channel of MRT0.
alignas(256) constexpr std::array<std::uint32_t, 4> PixelCode{
    0x7e0e02f2, 0xf800180f, 0x07070707, 0xbf810000,
};

alignas(256) constexpr std::array<std::array<float, 4>, 3> Triangle{{
    {-1.0f, -1.0f, 0.5f, 1.0f}, {3.0f, -1.0f, 0.5f, 1.0f}, {-1.0f, 3.0f, 0.5f, 1.0f}
}};

class GuestBlock {
public:
    GuestBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        block = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        Require(block != nullptr, "pipeline journal: cannot allocate the guest block");
        std::memset(block, NarrowConstantStoreFixture::Fill, BlockBytes);
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

    void Clear() const { std::memset(block, NarrowConstantStoreFixture::Fill, BlockBytes); }
    const std::uint8_t* Data() const { return block; }

private:
    std::uint8_t* block = nullptr;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t stride, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), count, 0x01016facu};
}

void Draw(AgcDriver::VulkanDevice& device) {
    Pixels.fill(Kept);
    const auto target = device.Target();
    std::vector<std::uint32_t> vertexUserData(4, 0u);
    const auto vertexBuffer = BufferDescriptor(Triangle.data(), 16u, static_cast<std::uint32_t>(Triangle.size()));
    std::copy(vertexBuffer.begin(), vertexBuffer.end(), vertexUserData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> vertexMemory{{{reinterpret_cast<std::uintptr_t>(VertexCode.data()), std::as_bytes(std::span(VertexCode))}}};
    ShaderRecompiler::RecompileRequest vertex{
        {ShaderStage::Vertex, reinterpret_cast<std::uintptr_t>(VertexCode.data()), VertexCode, 0, {}},
        {WaveSize, 0, vertexUserData, std::nullopt, std::nullopt, ShaderRecompiler::ShaderVertexStageInfo{}, vertexMemory},
        target,
        {0, 0, 0, 64}
    };
    vertex.useCache = false;
    const auto vertexResult = ShaderRecompiler::Recompile(vertex);
    const auto vertexPush = static_cast<std::uint32_t>(vertexResult.pushConstants.size());
    ShaderRecompiler::ShaderPixelStageInfo pixel{};
    pixel.targetOutputMode[0] = 9;
    pixel.targetExportMapping.fill(0xe4u);
    const std::vector<std::uint32_t> pixelUserData(8, 0u);
    const std::array<ShaderRecompiler::MemoryRegion, 1> pixelMemory{{{reinterpret_cast<std::uintptr_t>(PixelCode.data()), std::as_bytes(std::span(PixelCode))}}};
    ShaderRecompiler::RecompileRequest fragment{
        {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(PixelCode.data()), PixelCode, 0, {}},
        {WaveSize, 0, pixelUserData, std::nullopt, pixel, std::nullopt, pixelMemory},
        target,
        {0, 0, vertexPush, 128 - vertexPush}
    };
    fragment.useCache = false;
    const auto pixelResult = ShaderRecompiler::Recompile(fragment);
    const std::array<AgcDriver::Graphics::CompiledShader, 2> shaders{{
        {ShaderStage::Vertex, &vertexResult, 0},
        {ShaderStage::Fragment, &pixelResult, vertexPush}
    }};
    AgcDriver::Graphics::State state{};
    state.stages = {AgcDriver::Graphics::ShaderPath::Vertex, 0u, WaveSize, WaveSize, std::nullopt, std::nullopt};
    state.color = {reinterpret_cast<std::uintptr_t>(Pixels.data()), {Width, Height}, VK_FORMAT_R8G8B8A8_UNORM, Pixels.size(), 0xe4u};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {Width, Height};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, static_cast<float>(Height), static_cast<float>(Width), -static_cast<float>(Height), 0, 1};
    state.negativeOneToOne = false;
    state.scissor = {{0, 0}, {Width, Height}};
    state.cullMode = VK_CULL_MODE_NONE;
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    state.blendConstants = {};
    const AgcDriver::Pm4::DrawParameters draw{0, static_cast<std::uint32_t>(Triangle.size()), 0, 1, 0, false};
    device.Draw(state, draw, shaders);
    device.WaitIdle();
}

void ExpectDrawn(const char* what) {
    for (std::size_t index = 0; index < Pixels.size(); ++index) {
        Require(Pixels[index] == std::byte{255}, std::string(what) + ": byte " + std::to_string(index) + " is " + std::to_string(static_cast<unsigned>(Pixels[index])) + ", expected 255");
    }
}

// The constant-store compute program with its two stored literals replaced: `salt` 0 is the
// fixture's program, every other value a program of its own with its own module.
std::vector<std::uint32_t> ComputeCode(std::uint32_t salt) {
    std::vector<std::uint32_t> code(NarrowConstantStoreFixture::Code.begin(), NarrowConstantStoreFixture::Code.end());
    if (salt != 0) {
        code[3] = 0x00a10200u ^ (salt * 0x01010101u);
        code[5] = 0x0000fec3u ^ (salt * 0x00010001u);
    }
    return code;
}

ShaderRecompiler::RecompileResult CompileCompute(AgcDriver::VulkanDevice& device, const std::vector<std::uint32_t>& code, const GuestBlock& guest) {
    using namespace ShaderRecompiler;
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(guest.Data()));
    const std::array<std::uint32_t, 2> userData{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u)};
    const std::span<const std::uint32_t> words(code);
    const std::array<MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(words.data()), std::as_bytes(words)}}};
    const ShaderComputeStageInfo compute{{NarrowConstantStoreFixture::Threads, 1, 1}, 0, {false, false, false}, false, 1, {}};
    RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(words.data()), words, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(), {0, 0, 0, 128}, std::nullopt, false,
    };
    return Recompile(request);
}

void ExpectConstantStore(const GuestBlock& guest, const char* what) {
    using namespace NarrowConstantStoreFixture;
    for (std::size_t offset = 0; offset < Threads * LaneBytes; ++offset) {
        const auto expected = ExpectedLane[offset % LaneBytes];
        Require(guest.Data()[offset] == expected, std::string(what) + ": byte " + std::to_string(offset) + " is " + std::to_string(guest.Data()[offset]) + ", expected " + std::to_string(expected));
    }
}

void SetEnvironment(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

struct Totals {
    AgcDriver::Graphics::PipelineJournalCounters journal;
    AgcDriver::Graphics::PipelineStoreCounters store;
};

Totals Snapshot() {
    return {AgcDriver::Graphics::PipelineJournalTotals(), AgcDriver::Graphics::PipelineStoreTotals()};
}

void JournalTests() {
    Require(AgcDriver::Graphics::PipelineJournalEnabled(), "the pipeline journal is disabled although the shader cache directory is set");
    const auto journalPath = AgcDriver::Graphics::PipelineJournalPath();
    const GuestBlock guest;
    const auto computeCode = ComputeCode(0);
    {
        const auto device = OpenVulkanTestDevice();
        Require(device != nullptr, "the first device is required once the probe opened");
        device->WaitForPipelineJournalReplay();
        const auto before = Snapshot();
        Require(before.journal.loadedRecipes == 0 && before.journal.replayedGraphics == 0 && before.journal.replayedCompute == 0, "an empty journal replayed pipelines");
        guest.Clear();
        const auto result = CompileCompute(*device, computeCode, guest);
        device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(computeCode.data()));
        device->WaitIdle();
        ExpectConstantStore(guest, "first run: the constant store dispatch");
        Draw(*device);
        ExpectDrawn("first run: the draw");
        Require(device->ComputePipelinesCreated() == 1, "the first run did not build exactly one compute pipeline");
        AgcDriver::Graphics::FlushPipelineJournal();
        const auto after = Snapshot();
        Require(after.store.misses == before.store.misses + 1 && after.store.hits == before.store.hits, "the first draw did not miss the pipeline store exactly once");
        Require(after.journal.recordedGraphics == 1 && after.journal.recordedCompute == 1 && after.journal.recordedModules == 3 && after.journal.writeFailures == 0, "the first run did not record one graphics and one compute pipeline with three modules");
        Draw(*device);
        ExpectDrawn("first run: the repeated draw");
        AgcDriver::Graphics::FlushPipelineJournal();
        const auto repeated = Snapshot();
        Require(repeated.store.hits == after.store.hits + 1 && repeated.store.misses == after.store.misses, "a repeated draw did not hit the pipeline store");
        Require(repeated.journal.recordedGraphics == 1 && repeated.journal.recordedModules == 3, "a repeated draw recorded its pipeline again");
    }
    Require(std::filesystem::exists(journalPath) && std::filesystem::file_size(journalPath) > 0, "the journal file was not written");
    {
        const auto device = OpenVulkanTestDevice();
        Require(device != nullptr, "the second device could not be opened");
        device->WaitForPipelineJournalReplay();
        const auto replayed = Snapshot();
        Require(replayed.journal.loadedRecipes == 2 && replayed.journal.loadedModules == 3 && replayed.journal.droppedRecords == 0, "the second run did not load the two recorded pipelines and their three modules");
        Require(replayed.journal.replayedGraphics == 1 && replayed.journal.replayedCompute == 1 && replayed.journal.replayRejected == 0 && replayed.journal.replayFailed == 0 && replayed.journal.replayPresent == 0, "the replay did not build one graphics and one compute pipeline");
        Require(device->ComputePipelinesCreated() == 1, "the replay did not build the compute pipeline on the new device");
        guest.Clear();
        const auto result = CompileCompute(*device, computeCode, guest);
        device->Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(computeCode.data()));
        device->WaitIdle();
        ExpectConstantStore(guest, "second run: the constant store dispatch through the replayed pipeline");
        Require(device->ComputePipelinesCreated() == 1, "the second run's dispatch built a compute pipeline the replay had built");
        Draw(*device);
        ExpectDrawn("second run: the draw through the replayed pipeline");
        const auto drawn = Snapshot();
        Require(drawn.store.hits == replayed.store.hits + 1 && drawn.store.misses == replayed.store.misses, "the second run's draw did not find the replayed pipeline");
        AgcDriver::Graphics::FlushPipelineJournal();
        const auto final = Snapshot();
        Require(final.journal.recordedGraphics == 1 && final.journal.recordedCompute == 1 && final.journal.recordedModules == 3, "the second run recorded pipelines the journal already held");
    }
}

void Benchmark(std::uint32_t count) {
    Require(AgcDriver::Graphics::PipelineJournalEnabled(), "the pipeline journal is disabled");
    const GuestBlock guest;
    std::vector<std::vector<std::uint32_t>> codes;
    for (std::uint32_t salt = 1; salt <= count; ++salt) codes.push_back(ComputeCode(salt));
    struct Run {
        double openMs = 0;
        double dispatchMs = 0;
        std::uint64_t created = 0;
        AgcDriver::Graphics::PipelineJournalCounters journal;
    };
    const auto run = [&](const char* what) {
        Run measured;
        const auto openedAt = std::chrono::steady_clock::now();
        const auto device = OpenVulkanTestDevice();
        Require(device != nullptr, std::string("no device for the benchmark's ") + what);
        device->WaitForPipelineJournalReplay();
        measured.openMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - openedAt).count();
        measured.journal = AgcDriver::Graphics::PipelineJournalTotals();
        std::vector<ShaderRecompiler::RecompileResult> results;
        for (const auto& code : codes) results.push_back(CompileCompute(*device, code, guest));
        const auto createdBefore = device->ComputePipelinesCreated();
        const auto started = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < results.size(); ++index) device->Dispatch(results[index], 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(codes[index].data()));
        device->WaitIdle();
        measured.dispatchMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        measured.created = device->ComputePipelinesCreated() - createdBefore;
        AgcDriver::Graphics::FlushPipelineJournal();
        return measured;
    };
    const auto cold = run("cold run");
    const auto replayed = run("replayed run");
    std::error_code error;
    std::filesystem::remove(AgcDriver::Graphics::PipelineJournalPath(), error);
    Require(!error, "cannot remove the journal for the benchmark's last run");
    const auto warm = run("warm run without the journal");
    std::printf("pipeline journal benchmark: %u compute programs, %u replay threads\n", count, replayed.journal.replayThreads);
    std::printf("  cold driver cache, no journal:  %llu pipelines built at their dispatches, %u dispatches in %.1f ms\n", static_cast<unsigned long long>(cold.created), count, cold.dispatchMs);
    std::printf("  warm driver cache, no journal:  %llu pipelines built at their dispatches, %u dispatches in %.1f ms\n", static_cast<unsigned long long>(warm.created), count, warm.dispatchMs);
    std::printf("  warm driver cache, journal:     %llu built by the replay in %.1f ms (%.1f ms from the device opening), then %llu built at their dispatches, %u dispatches in %.1f ms\n", static_cast<unsigned long long>(replayed.journal.replayedCompute), replayed.journal.replayMs, replayed.openMs, static_cast<unsigned long long>(replayed.created), count, replayed.dispatchMs);
}

}

int main(int argc, char** argv) {
    try {
        const auto directory = std::filesystem::temp_directory_path() / ("aps5-pipeline-journal-test-" + std::to_string(std::random_device{}()));
        std::filesystem::create_directories(directory);
        SetEnvironment("ANYPS5_NO_SHADER_CACHE", "0");
        SetEnvironment("ANYPS5_SHADER_CACHE_DIR", directory.string());
        SetEnvironment("APS5_NO_PIPELINE_JOURNAL", "0");
        SetEnvironment("APS5_VERIFY_SET_LAYOUTS", "1");
        SetEnvironment("APS5_PIPELINE_JOURNAL_THREADS", "2");
        struct Cleanup {
            std::filesystem::path directory;
            ~Cleanup() {
                std::error_code error;
                std::filesystem::remove_all(directory, error);
            }
        } cleanup{directory};
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        if (argc >= 2 && std::string(argv[1]) == "--benchmark") {
            Benchmark(argc >= 3 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 24u);
            return 0;
        }
        JournalTests();
        std::puts("pipeline journal record and replay passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
