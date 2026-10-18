#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGUNITS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PENDINGUNITS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

namespace AgcDriver::Graphics {

// Sets units [first, first + count) of a storage image's pending bits (StorageTexture::
// markLayersPending). With `fill` (live `pending_fill`) one std::fill, which libstdc++ runs over
// whole words of the bit vector; without, one bit at a time as before. The bits set are the same.
inline void MarkUnitsPending(std::vector<bool>& units, std::uint32_t first, std::uint32_t count, bool fill) {
    if (count == 0) return;
    if (fill) {
        std::fill(units.begin() + first, units.begin() + first + count, true);
        return;
    }
    for (std::uint32_t unit = first; unit < first + count; ++unit) units[unit] = true;
}

}

#endif
