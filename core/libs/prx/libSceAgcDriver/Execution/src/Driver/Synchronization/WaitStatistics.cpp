#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstdlib>

namespace AgcDriver::DriverDetail {

PollStats& Driver::pollStats() {
    static thread_local PollStats stats;
    return stats;
}

WaitOutcomes& Driver::waitOutcomes() {
    static thread_local WaitOutcomes outcomes;
    return outcomes;
}

Graphics::Recorder::LateStatistics& Driver::lateCountsSeen() {
    static thread_local Graphics::Recorder::LateStatistics seen{};
    return seen;
}

EpochBumps& Driver::epochBumps() {
    static thread_local EpochBumps bumps;
    return bumps;
}

std::chrono::steady_clock::time_point& Driver::lastEpochBump() {
    static thread_local std::chrono::steady_clock::time_point at{};
    return at;
}

std::uint64_t& Driver::lastBumpGeneration() {
    static thread_local std::uint64_t generation = 0;
    return generation;
}

void Driver::bumpEpoch(std::uint64_t EpochBumps::*counter) {
    lastEpochBump() = std::chrono::steady_clock::now();
    static const bool entrySkip = std::getenv("APS5_ENTRY_EPOCH_SKIP") != nullptr || std::getenv("APS5_PROFILE_COLLECT") != nullptr;
    // Before the bump: a CPU store collected from here on stamps newer than this.
    if (entrySkip) lastBumpGeneration() = GuestMemory::TrackerGeneration();
    GuestMemory::BumpCollectEpoch();
    if (DrawPipeline::Active()) GuestMemory::NoteEpochToken(DrawPipeline::EpochToken().fetch_add(1, std::memory_order_relaxed) + 1);
    ++(epochBumps().*counter);
}

bool Driver::packetEpoch() {
    static const bool packet = std::getenv("APS5_PACKET_EPOCH") != nullptr;
    return packet;
}

}
