#include "prx/libSceVideoOut/include/AutomationCapture.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "SDL_syswm.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

// Top-down BGRA pixels.
struct Image {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> bgra;
};

double MeanLuma(const Image& image, bool& black) {
    double sum = 0.0;
    std::uint64_t count = 0;
    black = true;
    for (std::uint32_t y = 0; y < image.height; y += 4) {
        for (std::uint32_t x = 0; x < image.width; x += 4) {
            const auto* p = image.bgra.data() + (static_cast<std::size_t>(y) * image.width + x) * 4;
            if (p[0] != 0 || p[1] != 0 || p[2] != 0) black = false;
            sum += 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
            ++count;
        }
    }
    return count != 0 ? sum / static_cast<double>(count) : 0.0;
}

void WriteAll(std::FILE* file, const void* data, std::size_t size) {
    if (size != 0 && std::fwrite(data, 1, size, file) != size) throw std::runtime_error("short write");
}

void PutU32BE(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value >> 24));
    out.push_back(static_cast<std::uint8_t>(value >> 16));
    out.push_back(static_cast<std::uint8_t>(value >> 8));
    out.push_back(static_cast<std::uint8_t>(value));
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size, std::uint32_t crc = 0) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> entries{};
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            entries[n] = c;
        }
        return entries;
    }();
    crc = ~crc;
    for (std::size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void WriteChunk(std::FILE* file, const char* type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> chunk;
    chunk.reserve(payload.size() + 12);
    PutU32BE(chunk, static_cast<std::uint32_t>(payload.size()));
    chunk.insert(chunk.end(), type, type + 4);
    chunk.insert(chunk.end(), payload.begin(), payload.end());
    PutU32BE(chunk, Crc32(chunk.data() + 4, chunk.size() - 4));
    WriteAll(file, chunk.data(), chunk.size());
}

// RGB8 PNG with stored (uncompressed) deflate blocks: no zlib dependency, any viewer reads it.
void WritePng(std::FILE* file, const Image& image) {
    static constexpr std::uint8_t signature[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    WriteAll(file, signature, sizeof(signature));
    std::vector<std::uint8_t> header;
    PutU32BE(header, image.width);
    PutU32BE(header, image.height);
    header.insert(header.end(), {8, 2, 0, 0, 0});
    WriteChunk(file, "IHDR", header);
    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(image.height) * (image.width * 3 + 1));
    for (std::uint32_t y = 0; y < image.height; ++y) {
        raw.push_back(0);
        const auto* row = image.bgra.data() + static_cast<std::size_t>(y) * image.width * 4;
        for (std::uint32_t x = 0; x < image.width; ++x) {
            raw.push_back(row[x * 4 + 2]);
            raw.push_back(row[x * 4 + 1]);
            raw.push_back(row[x * 4 + 0]);
        }
    }
    std::vector<std::uint8_t> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    zlib.push_back(0x78);
    zlib.push_back(0x01);
    std::uint32_t a = 1, b = 0;
    for (std::size_t offset = 0; offset < raw.size() || offset == 0; ) {
        const auto length = static_cast<std::uint16_t>(std::min<std::size_t>(65535, raw.size() - offset));
        const bool final = offset + length >= raw.size();
        zlib.push_back(final ? 1 : 0);
        zlib.push_back(static_cast<std::uint8_t>(length));
        zlib.push_back(static_cast<std::uint8_t>(length >> 8));
        zlib.push_back(static_cast<std::uint8_t>(~length));
        zlib.push_back(static_cast<std::uint8_t>(static_cast<std::uint16_t>(~length) >> 8));
        for (std::size_t i = offset; i < offset + length; ++i) {
            a = (a + raw[i]) % 65521;
            b = (b + a) % 65521;
        }
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset), raw.begin() + static_cast<std::ptrdiff_t>(offset + length));
        offset += length;
        if (final) break;
    }
    PutU32BE(zlib, (b << 16) | a);
    WriteChunk(file, "IDAT", zlib);
    WriteChunk(file, "IEND", {});
}

// 24-bit bottom-up BMP.
void WriteBmp(std::FILE* file, const Image& image) {
    const std::uint32_t stride = (image.width * 3 + 3) & ~3u;
    const std::uint32_t imageBytes = stride * image.height;
    std::uint8_t header[54] = {'B', 'M'};
    const auto put32 = [&](int offset, std::uint32_t value) {
        for (int i = 0; i < 4; ++i) header[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
    };
    put32(2, 54 + imageBytes);
    put32(10, 54);
    put32(14, 40);
    put32(18, image.width);
    put32(22, image.height);
    header[26] = 1;
    header[28] = 24;
    put32(34, imageBytes);
    put32(38, 2835);
    put32(42, 2835);
    WriteAll(file, header, sizeof(header));
    std::vector<std::uint8_t> row(stride, 0);
    for (std::uint32_t y = image.height; y-- > 0;) {
        const auto* source = image.bgra.data() + static_cast<std::size_t>(y) * image.width * 4;
        for (std::uint32_t x = 0; x < image.width; ++x) {
            row[x * 3 + 0] = source[x * 4 + 0];
            row[x * 3 + 1] = source[x * 4 + 1];
            row[x * 3 + 2] = source[x * 4 + 2];
        }
        WriteAll(file, row.data(), row.size());
    }
}

void WriteImage(const std::string& path, const Image& image) {
    const auto dot = path.find_last_of('.');
    std::string extension = dot == std::string::npos ? "" : path.substr(dot);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // Written next to the target and renamed, so a reader never sees a partial image.
    const auto temporary = path + ".partial";
    std::FILE* file = std::fopen(temporary.c_str(), "wb");
    if (file == nullptr) throw std::runtime_error("cannot create " + temporary);
    try {
        if (extension == ".png") WritePng(file, image);
        else WriteBmp(file, image);
    } catch (...) {
        std::fclose(file);
        std::remove(temporary.c_str());
        throw;
    }
    if (std::fclose(file) != 0) throw std::runtime_error("cannot finish " + temporary);
#ifdef _WIN32
    if (!MoveFileExA(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) throw std::runtime_error("cannot rename " + temporary + " to " + path);
#else
    if (std::rename(temporary.c_str(), path.c_str()) != 0) throw std::runtime_error("cannot rename " + temporary + " to " + path);
#endif
}

#ifdef _WIN32
#ifndef PW_CLIENTONLY
#define PW_CLIENTONLY 0x1
#endif
#ifndef PW_RENDERFULLCONTENT
#define PW_RENDERFULLCONTENT 0x2
#endif

// Captures the client area of `window` with PrintWindow, or by copying the screen.
Image CaptureClient(HWND window, bool fromScreen) {
    RECT client{};
    if (!GetClientRect(window, &client)) throw std::runtime_error("GetClientRect failed");
    Image image;
    image.width = static_cast<std::uint32_t>(client.right - client.left);
    image.height = static_cast<std::uint32_t>(client.bottom - client.top);
    if (image.width == 0 || image.height == 0) throw std::runtime_error("the window has no client area (minimized?)");
    HDC windowDc = GetDC(window);
    if (windowDc == nullptr) throw std::runtime_error("GetDC failed");
    HDC memory = CreateCompatibleDC(windowDc);
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(info.bmiHeader);
    info.bmiHeader.biWidth = static_cast<LONG>(image.width);
    info.bmiHeader.biHeight = -static_cast<LONG>(image.height);
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(windowDc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    bool captured = false;
    if (memory != nullptr && bitmap != nullptr) {
        HGDIOBJ previous = SelectObject(memory, bitmap);
        if (fromScreen) {
            POINT origin{0, 0};
            ClientToScreen(window, &origin);
            HDC screen = GetDC(nullptr);
            captured = screen != nullptr && BitBlt(memory, 0, 0, static_cast<int>(image.width), static_cast<int>(image.height), screen, origin.x, origin.y, SRCCOPY | CAPTUREBLT);
            if (screen != nullptr) ReleaseDC(nullptr, screen);
        } else {
            captured = PrintWindow(window, memory, PW_CLIENTONLY | PW_RENDERFULLCONTENT) != FALSE;
        }
        GdiFlush();
        if (captured) {
            const auto* source = static_cast<const std::uint8_t*>(bits);
            image.bgra.assign(source, source + static_cast<std::size_t>(image.width) * image.height * 4);
        }
        SelectObject(memory, previous);
    }
    if (bitmap != nullptr) DeleteObject(bitmap);
    if (memory != nullptr) DeleteDC(memory);
    ReleaseDC(window, windowDc);
    if (!captured) throw std::runtime_error(fromScreen ? "screen copy failed" : "PrintWindow failed");
    return image;
}
#endif

}

void AutomationCaptureWindow(SDL_Window* window, const AutomationSnapshotRequest& request) {
    Automation::Fields fields;
    bool ok = false;
    try {
#ifdef _WIN32
        if (window == nullptr) throw std::runtime_error("no output window");
        if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0) throw std::runtime_error("the output window is minimized");
        SDL_SysWMinfo wm;
        SDL_VERSION(&wm.version);
        if (!SDL_GetWindowWMInfo(window, &wm) || wm.subsystem != SDL_SYSWM_WINDOWS) throw std::runtime_error("no native window handle");
        HWND handle = wm.info.win.window;
        std::string method = "printwindow";
        std::string printError;
        Image image;
        bool black = true;
        double mean = 0.0;
        try {
            image = CaptureClient(handle, false);
            mean = MeanLuma(image, black);
        } catch (const std::exception& error) {
            printError = error.what();
        }
        // Flip-model swapchains may not reach PrintWindow; the screen holds them when visible.
        if (image.bgra.empty() || black) {
            try {
                auto screen = CaptureClient(handle, true);
                bool screenBlack = true;
                const double screenMean = MeanLuma(screen, screenBlack);
                if (image.bgra.empty() || !screenBlack) {
                    image = std::move(screen);
                    black = screenBlack;
                    mean = screenMean;
                    method = "screen";
                    // The screen copy shows whatever covers the window.
                    RECT client{};
                    GetClientRect(handle, &client);
                    POINT center{(client.right - client.left) / 2, (client.bottom - client.top) / 2};
                    ClientToScreen(handle, &center);
                    const HWND top = WindowFromPoint(center);
                    fields.Bool("occluded", top != handle && GetAncestor(top, GA_ROOT) != handle);
                }
            } catch (const std::exception& error) {
                if (image.bgra.empty()) throw std::runtime_error(printError + "; " + error.what());
            }
        }
        WriteImage(request.path, image);
        fields.Unsigned("width", image.width).Unsigned("height", image.height).Str("method", method).Real("mean", mean).Bool("black", black);
        ok = true;
#else
        (void)window;
        throw std::runtime_error("snapshots need Windows");
#endif
    } catch (const std::exception& error) {
        fields.Str("error", error.what());
    }
    Aps5AutomationSnapshotDone_nid_postfix(request.id, ok, fields.Text().c_str());
}
