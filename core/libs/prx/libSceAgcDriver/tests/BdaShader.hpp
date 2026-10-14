#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_BDASHADER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <vector>

std::vector<std::uint32_t> MakeBdaTestShader(std::uint64_t address, std::uint32_t bits, std::int64_t offset = 0);
std::vector<std::uint32_t> MakeBdaDwordReadTestShader(std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops);
// An anchored BDA read test program (MakeBdaAnchoredReadTestShader): a read of `bits` (8, 16, or
// 32: `dwords` dwords of which the program extracts `used`) at anchor + zext(low + lane * stride)
// + immediate in each of `lanes` invocations, the anchor searched by the subgroup's elected lane
// when `elect`.
struct BdaAnchoredRead {
    std::uint64_t anchor = 0;
    std::uint32_t low = 0;
    std::int32_t immediate = 0;
    std::uint32_t bits = 32;
    std::uint32_t dwords = 1;
    std::uint32_t used = 1;
    bool stops = true;
    bool elect = false;
    std::uint32_t lanes = 1;
    std::uint32_t stride = 0;
};
std::vector<std::uint32_t> MakeBdaAnchoredReadTestShader(const BdaAnchoredRead& read, bool anchored);
void RunBdaExecutionTests(const AgcDriver::Graphics::Context& context);
void RunBdaContractTests();

#endif
