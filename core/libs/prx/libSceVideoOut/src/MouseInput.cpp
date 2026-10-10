#include "prx/libSceVideoOut/include/MouseInput.hpp"
#include "prx/libSceMouse/include/MouseState.hpp"

#include "SDL_mouse.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

void MouseInput::SetGeometry(std::uint32_t sourceWidth, std::uint32_t sourceHeight, int windowWidth, int windowHeight) {
    static const auto space = [] {
        std::pair<std::uint32_t, std::uint32_t> size{0, 0};
        if (const char* text = std::getenv("APS5_MOUSE_SPACE"); text != nullptr) std::sscanf(text, "%ux%u", &size.first, &size.second);
        return size;
    }();
    if (space.first != 0 && space.second != 0) {
        sourceWidth = space.first;
        sourceHeight = space.second;
    }
    if (sourceWidth == 0 || sourceHeight == 0 || windowWidth <= 0 || windowHeight <= 0) return;
    // The image is fitted inside the window, so one window pixel covers this many source pixels.
    scale = std::max(static_cast<double>(sourceWidth) / windowWidth, static_cast<double>(sourceHeight) / windowHeight);
}

void MouseInput::HandleEvent(const SDL_Event& event, unsigned windowId) {
    MouseInputEvent input{};
    switch (event.type) {
        case SDL_MOUSEMOTION:
            if (event.motion.windowID != windowId || !focused) return;
            remainderX += event.motion.xrel * scale;
            remainderY += event.motion.yrel * scale;
            input.x = static_cast<std::int32_t>(std::trunc(remainderX));
            input.y = static_cast<std::int32_t>(std::trunc(remainderY));
            remainderX -= input.x;
            remainderY -= input.y;
            if (input.x == 0 && input.y == 0) return;
            break;
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            if (event.button.windowID != windowId || !focused) return;
            switch (event.button.button) {
                case SDL_BUTTON_LEFT: input.button = 1; break;
                case SDL_BUTTON_RIGHT: input.button = 2; break;
                case SDL_BUTTON_MIDDLE: input.button = 4; break;
                default: return;
            }
            input.pressed = event.type == SDL_MOUSEBUTTONDOWN;
            break;
        case SDL_MOUSEWHEEL:
            if (event.wheel.windowID != windowId || !focused) return;
            input.wheel = event.wheel.y;
            input.tilt = event.wheel.x;
            if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
                input.wheel = -input.wheel;
                input.tilt = -input.tilt;
            }
            if (input.wheel == 0 && input.tilt == 0) return;
            break;
        case SDL_WINDOWEVENT:
            if (event.window.windowID != windowId) return;
            if (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST) {
                focused = false;
                input.resetButtons = true;
            } else if (event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
                focused = true;
                input.connectionChange = true;
            } else if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                focused = false;
                input.connectionChange = true;
                input.connected = false;
            } else {
                return;
            }
            break;
        default:
            return;
    }
    MousePublishInput_nid_postfix(input);
}
