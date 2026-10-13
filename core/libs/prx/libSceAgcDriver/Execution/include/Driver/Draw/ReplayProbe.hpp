#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_REPLAYPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_REPLAYPROBE_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace AgcDriver::DriverDetail {

struct ReplaySegment;

namespace ReplayProbe {

constexpr std::uint32_t NoCut = 0xffffffffu;
constexpr std::uint32_t CutFlip = 0x100;
constexpr std::uint32_t CutRenderingWait = 0x101;
constexpr std::uint32_t CutPredicated = 0x102;
constexpr std::uint32_t CutIndirect = 0x103;
constexpr std::uint32_t CutLength = 0x104;
constexpr std::uint32_t CutSubmission = 0x105;
constexpr std::uint32_t CutCount = 0x106;

bool Enabled();
std::shared_ptr<ReplaySegment> Hold();
std::shared_ptr<ReplaySegment> Tag();
void Committed(ReplaySegment& segment, std::uint64_t ns, bool failed);
std::shared_ptr<ReplaySegment> Observe(const QueueState& queue, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> registers, std::uint32_t cut, const std::function<std::uint64_t()>& device, const void* registry);
void Cut(std::uint32_t reason);

class Scope {
public:
    explicit Scope(std::shared_ptr<ReplaySegment> segment);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    std::shared_ptr<ReplaySegment> held;
    ReplaySegment* previous = nullptr;
    GuestMemory::ProbeSink* previousSink = nullptr;
};

}

}

#endif
