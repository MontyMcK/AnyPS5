#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSPECULATION_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DRAWSPECULATION_HPP

#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

struct DrawSpeculation {
    enum class Kind : std::uint8_t { None, Address, Template };
    using Miss = ShaderResources::SpeculationMiss;
    Kind kind = Kind::None;
    VkDevice device = VK_NULL_HANDLE;
    ResourceCache::Key key;
    std::shared_ptr<ShaderResources> source;
    bool aliasesOk = false;
    std::vector<ShaderResources::MovedBuffer> moved;
    std::size_t movedGuest = 0;
    std::vector<ShaderResources::SpeculatedImage> movedImages;
    ShaderResources::ProofSpeculation proof;
    ShaderResources::MovedProof movedProof;
    std::uint64_t frontier = 0;
    bool windowed = false;
    std::weak_ptr<const GuestBufferMemory::AddressSpace> space;
    std::shared_ptr<ShaderResources> use;
    Miss miss = Miss::None;
};

enum class WindowVerdict : std::uint8_t { Clear, Overlap, Unknown, Overflow };
struct AppliedWindow {
    std::uint64_t (*frontier)() = nullptr;
    WindowVerdict (*check)(std::uint64_t fromExclusive, std::span<const std::pair<std::uint64_t, std::uint64_t>> reads) = nullptr;
};
void SetAppliedWindow(AppliedWindow window);
bool DrawTokensEnabled();
bool TargetTokensEnabled();
bool InputTokensEnabled();
bool VerifyDrawTokens();
enum class TokenFamily : std::uint8_t { Proof, Moved };
void CountTokenOutcome(TokenFamily family, ShaderResources::TokenFallback outcome);
void ScreenTokenWindow(DrawSpeculation& speculation);
std::uint64_t TakeCrossEpochTargetProofs();

bool DrawSpeculationEnabled();
std::shared_ptr<ForkedDraw> ForkDrawSpeculated(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots, const std::shared_ptr<DrawPrebuild>& prebuilt, std::shared_ptr<const DrawRecipe>* recipe, const ForkSeed* seed, bool seedCurrent, std::size_t slot, std::shared_ptr<DrawSpeculation> speculation);
bool VerifyDrawSpeculation();
std::shared_ptr<DrawSpeculation> SpeculateDrawResources(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots);
void CountSpeculationMiss(DrawSpeculation::Miss miss, bool committer);

class SpeculationScope {
public:
    explicit SpeculationScope(std::shared_ptr<DrawSpeculation> speculation);
    ~SpeculationScope();
    SpeculationScope(const SpeculationScope&) = delete;
    SpeculationScope& operator=(const SpeculationScope&) = delete;
};

}

#endif
