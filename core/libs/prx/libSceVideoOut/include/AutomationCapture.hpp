#ifndef CORE_LIBS_PRX_LIBSCEVIDEOOUT_AUTOMATIONCAPTURE_HPP
#define CORE_LIBS_PRX_LIBSCEVIDEOOUT_AUTOMATIONCAPTURE_HPP

#include "SDL.h"
#include "prx/libkernel/Automation/include/Automation.hpp"

// APS5_AUTOMATION snapshot: captures the client area of the output window as presented (the
// window's own content through PrintWindow, so it works unfocused and behind other windows; the
// visible screen pixels when that comes back black), writes it to request.path (.png, otherwise
// .bmp) and reports the result to the automation channel. Runs on the presentation thread.
void AutomationCaptureWindow(SDL_Window* window, const AutomationSnapshotRequest& request);

#endif
