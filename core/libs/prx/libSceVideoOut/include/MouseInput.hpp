#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_MOUSEINPUT_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_MOUSEINPUT_HPP

#include "SDL_events.h"

#include <cstdint>

class MouseInput {
public:
    void HandleEvent(const SDL_Event& event, unsigned windowId);
    // Motion is reported in window pixels; the title moves its pointer in pixels of the image it
    // presents, which the window shows scaled to fit. Deltas are converted with this geometry.
    void SetGeometry(std::uint32_t sourceWidth, std::uint32_t sourceHeight, int windowWidth, int windowHeight);

private:
    bool focused = true;
    double scale = 1.0;
    double remainderX = 0.0;
    double remainderY = 0.0;
};

#endif
