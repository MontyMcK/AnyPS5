#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Queue 0 reaps the recorder's completions while it polls, and once dormant only when woken. A
// completion another thread leaves behind after queue 0 went dormant (here: a write-back
// completion of a submitted batch, then a compute queue finishing a submission) must wake it, or
// the completion runs only when queue 0 gets its next submission.

namespace {

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::size_t BlockBytes = 65536;

alignas(256) constexpr std::array<std::uint32_t, 8> Code{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("queue zero late completion: " + what);
}

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    check(block != nullptr, "cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true);
    return block;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

template <typename Done>
bool WaitFor(const Done& done, std::chrono::milliseconds limit = std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    for (;;) {
        if (done()) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_NO_OPPORTUNISTIC_REAP", "1");
#else
    setenv("APS5_NO_OPPORTUNISTIC_REAP", "1", 1);
#endif
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        struct ShutdownGuard {
            ~ShutdownGuard() { try { AgcDriverShutdown_nid_postfix(); } catch (...) {} }
        } shutdown;

        Shader shader{};
        shader.file_header = 0x34333231;
        shader.version = 0x18;
        shader.header_size = sizeof(Shader);
        shader.shader_size = sizeof(Code);
        shader.code = Code.data();
        AgcDriverRegisterShader_nid_postfix(&shader);

        auto* input = NewBlock();
        auto* output = NewBlock();
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) input[lane] = 0x5b000000u | lane;

        const auto codeAddress = reinterpret_cast<std::uintptr_t>(Code.data());
        const auto in = BufferDescriptor(input, BufferBytes);
        const auto out = BufferDescriptor(output, BufferBytes);
        std::vector<std::uint32_t> dispatch{
            0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
            0xc0037600u, 0x207u, Lanes, 1u, 1u,
            0xc0017600u, 0x213u, 8u << 1u,
            0xc0087600u, 0x240u, in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3],
            0xc0031500u, 1u, 1u, 1u, 0x8041u,
        };
        Packet packet{dispatch.data(), static_cast<std::uint32_t>(dispatch.size()), 0, {}};
        check(sceAgcDriverSubmitDcb(&packet) == 0, "the dispatch submission failed");

        const auto dormant = [] { return AgcDriver::DriverDetail::queue0Dormant.load(std::memory_order_relaxed); };
        const auto& pipeline = AgcDriver::DriverDetail::DrawPipeline::Queue0();
        check(WaitFor([&] { return dormant() && !pipeline.Busy(); }), "queue 0 did not wait for work after the dispatch");

        // A write-back completion behind a batch submitted while queue 0 is dormant.
        std::atomic<bool> ran{false};
        bool registered = false;
        const bool submitted = WaitFor([&] {
            std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
            auto* recorder = AgcDriver::Graphics::Recorder::Active();
            if (recorder == nullptr) return false;
            if (!registered) {
                recorder->OnComplete([&ran] { ran.store(true, std::memory_order_release); });
                registered = true;
            }
            return recorder->SubmitIfStarved(0, true);
        });
        check(registered, "no active recorder after the dispatch");
        check(submitted, "the batch holding the write-back completion was not submitted");
        check(dormant(), "queue 0 left its wait without work");
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // A compute queue finishes a submission while the completion is pending: queue 0 reaps it.
        std::vector<std::uint32_t> compute{0xc0017600u, 0x213u, 0u};
        Packet computePacket{compute.data(), static_cast<std::uint32_t>(compute.size()), 0, {}};
        check(sceAgcDriverSubmitAcb(0x20, &computePacket) == 0, "the compute submission failed");
        check(WaitFor([&] { return ran.load(std::memory_order_acquire); }), "a write-back completion left behind while queue 0 waited was not reaped after a compute queue finished its submission");

        AgcDriverWaitIdle_nid_postfix();
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) check(output[lane] == input[lane] + 1u, "the dispatch stored a wrong value");
        AgcDriverShutdown_nid_postfix();
        std::puts("queue zero late completion tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
