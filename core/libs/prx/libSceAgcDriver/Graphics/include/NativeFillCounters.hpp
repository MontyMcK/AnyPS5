#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace AgcDriver::Graphics {

// The native pattern fill's counters (Driver::nativePatternFill): fills that replaced a dispatch of
// the title's dword fill kernel (stored on the GPU or, without a host import, by the CPU), and the
// dispatches of the kernel that kept the shader, by reason (a string literal). Thread-safe.
void CountNativeFill(std::size_t bytes, bool cpu);
void CountNativeFillRefusal(const char* reason);
void CountNativeFillVerify(bool matched);
// The counts since this consumer's previous call (0: the [gputime] line, 1: the [draw] line), as
// "native pattern fills: ..." text; empty while no kernel dispatch was seen at all.
std::string NativeFillSummary(int consumer);

}
