#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEIDENTITY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINEIDENTITY_HPP

#include "Recompiler.hpp"
#include <compare>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Graphics {

// A 128-bit hash of a module's words, zero for an empty module. The mixing is pure integer
// arithmetic over the words in order, so the value is the same on every run, platform and compiler:
// the pipeline journal names modules by it across runs.
struct SpirvContentHash {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    bool operator==(const SpirvContentHash&) const = default;
};

inline std::uint64_t HashSpirvWords(std::span<const std::uint32_t> words, std::uint64_t seed) {
    constexpr std::uint64_t multiplier = 0x9e3779b97f4a7c15ull;
    std::uint64_t hash = seed ^ (static_cast<std::uint64_t>(words.size()) * multiplier);
    const auto mix = [&](std::uint64_t chunk) {
        chunk *= 0xff51afd7ed558ccdull;
        chunk ^= chunk >> 32u;
        hash = (hash ^ chunk) * multiplier;
        hash = (hash << 27u) | (hash >> 37u);
    };
    std::size_t index = 0;
    for (; index + 2 <= words.size(); index += 2) mix(words[index] | (static_cast<std::uint64_t>(words[index + 1]) << 32u));
    if (index < words.size()) mix(words[index] ^ 0x8000000000000000ull);
    hash ^= hash >> 33u;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33u;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33u;
    return hash;
}

// HashSpirvWords with two seeds in one pass over the words (the module hash reads each module once).
inline SpirvContentHash HashSpirvWordsTwice(std::span<const std::uint32_t> words, std::uint64_t seedLow, std::uint64_t seedHigh) {
    constexpr std::uint64_t multiplier = 0x9e3779b97f4a7c15ull;
    std::uint64_t low = seedLow ^ (static_cast<std::uint64_t>(words.size()) * multiplier);
    std::uint64_t high = seedHigh ^ (static_cast<std::uint64_t>(words.size()) * multiplier);
    const auto mix = [&](std::uint64_t chunk) {
        chunk *= 0xff51afd7ed558ccdull;
        chunk ^= chunk >> 32u;
        low = (low ^ chunk) * multiplier;
        low = (low << 27u) | (low >> 37u);
        high = (high ^ chunk) * multiplier;
        high = (high << 27u) | (high >> 37u);
    };
    std::size_t index = 0;
    for (; index + 2 <= words.size(); index += 2) mix(words[index] | (static_cast<std::uint64_t>(words[index + 1]) << 32u));
    if (index < words.size()) mix(words[index] ^ 0x8000000000000000ull);
    const auto finish = [](std::uint64_t hash) {
        hash ^= hash >> 33u;
        hash *= 0xff51afd7ed558ccdull;
        hash ^= hash >> 33u;
        hash *= 0xc4ceb9fe1a85ec53ull;
        hash ^= hash >> 33u;
        return hash;
    };
    return {finish(low), finish(high)};
}

// What a Vulkan pipeline built from a result depends on besides the fixed-function state and the
// descriptor set layout: the module words, the specialization constants handed to the driver, the
// subgroup width the module was laid out for and the bindings' shape. Two results with equal
// identities build equal pipelines, in this run or another: unlike the process-lifetime variant ids
// (which a specialized module without constants shares with its unspecialized artifact), the
// identity follows the module, so the pipeline stores key on it and the pipeline journal names a
// stage across runs. Empty for a result without a module.
//
// The module hash is kept per thread for the module's shared word buffer (SharedSpirv shares one
// buffer between a cached variant and every result materialized from it): each entry holds a copy
// of the SharedSpirv, so the buffer it names stays alive, and its address cannot be reused by
// another module, for as long as the entry is remembered. A holder that writes to shared words gets
// a buffer of its own (copy on write), hence another address and its own hash.
struct PipelineIdentity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    [[nodiscard]] bool Empty() const { return low == 0 && high == 0; }
    bool operator==(const PipelineIdentity&) const = default;
    auto operator<=>(const PipelineIdentity&) const = default;
};

[[nodiscard]] SpirvContentHash ModuleContentHash(const ShaderRecompiler::SharedSpirv& spirv);
[[nodiscard]] PipelineIdentity StageIdentity(const ShaderRecompiler::RecompileResult& result);

}

#endif
