#include "prx/libSceAgcDriver/Graphics/include/PipelineCache.hpp"
#include "ShaderCacheDirectory.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

constexpr std::uint32_t FileMagic = 0x50565041u;
constexpr std::uint32_t FileFormat = 1;
constexpr std::size_t HeaderBytes = 24;

// Every save of the process runs under this lock: the objects of one process (the headless device's
// and the windowed one's) share the file, and each compares the header on disk with its own.
std::mutex& filesMutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}

struct Registry {
    std::mutex mutex;
    std::vector<PipelineCache*> objects;
};

Registry& registry() {
    static auto* objects = new Registry;
    return *objects;
}

const char* reasonName(PipelineCache::SaveReason reason) {
    switch (reason) {
        case PipelineCache::SaveReason::Periodic: return "periodic";
        case PipelineCache::SaveReason::Exit: return "exit";
        case PipelineCache::SaveReason::Teardown: return "teardown";
    }
    return "?";
}

}

std::chrono::seconds PipelineCache::PeriodicInterval() {
    static const std::chrono::seconds interval = [] {
        const char* value = std::getenv("APS5_PIPELINE_CACHE_SAVE_SECONDS");
        if (value == nullptr || *value == '\0') return std::chrono::seconds(PeriodicSaveSeconds);
        const long seconds = std::strtol(value, nullptr, 10);
        return std::chrono::seconds(std::max<long>(seconds, SaveIntervalSeconds));
    }();
    return interval;
}

bool PipelineCache::WriteFileDurably(const std::filesystem::path& path, std::span<const std::byte> bytes) {
#ifdef _WIN32
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    static std::atomic<std::uint64_t> serial{0};
    auto temporary = path;
    temporary += ".tmp." + std::to_string(GetCurrentProcessId()) + "." + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
    const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    bool written = true;
    constexpr std::size_t chunk = std::size_t{64} << 20u;
    for (std::size_t offset = 0; offset < bytes.size() && written; offset += chunk) {
        const auto count = static_cast<DWORD>(std::min(chunk, bytes.size() - offset));
        DWORD done = 0;
        written = WriteFile(file, bytes.data() + offset, count, &done, nullptr) && done == count;
    }
    written = written && FlushFileBuffers(file);
    CloseHandle(file);
    if (written && MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
    DeleteFileW(temporary.c_str());
    return false;
#else
    return ShaderRecompiler::WriteFileAtomically(path, bytes);
#endif
}

PipelineCache::PipelineCache(const Context& context, const VkPhysicalDeviceProperties& properties) : context(context), properties(properties) {
    const auto directory = ShaderRecompiler::ShaderCacheDirectory();
    if (!directory.empty()) {
        std::string uuid;
        for (const auto byte : properties.pipelineCacheUUID) {
            char digits[3];
            std::snprintf(digits, sizeof(digits), "%02x", static_cast<unsigned>(byte));
            uuid += digits;
        }
        char name[160];
        std::snprintf(name, sizeof(name), "vk-pipelines-%04x-%04x-%08x-%s.bin", properties.vendorID, properties.deviceID, properties.driverVersion, uuid.c_str());
        path = directory / name;
    }
    std::vector<std::byte> initialData;
    if (!path.empty()) {
        std::lock_guard fileLock(filesMutex());
        // Temporaries of a save that never finished. A temporary named for another live process
        // (".tmp.<pid>.<serial>") may be a save in progress and stays.
        std::error_code error;
        const auto prefix = path.filename().string() + ".tmp.";
        for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), error)) {
            const auto name = entry.path().filename().string();
            if (name.rfind(prefix, 0) != 0) continue;
#ifdef _WIN32
            const auto writer = std::strtoul(name.c_str() + prefix.size(), nullptr, 10);
            if (writer != 0 && writer != GetCurrentProcessId()) {
                if (const HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, static_cast<DWORD>(writer))) {
                    const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
                    CloseHandle(process);
                    if (alive) continue;
                }
            }
#endif
            std::error_code removeError;
            std::filesystem::remove(entry.path(), removeError);
        }
        load(initialData);
    }
    VkPipelineCacheCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    info.initialDataSize = initialData.size();
    info.pInitialData = initialData.empty() ? nullptr : initialData.data();
    const auto create = context.Function<PFN_vkCreatePipelineCache>("vkCreatePipelineCache");
    VkResult result = create(context.device, &info, nullptr, &cache);
    if (result != VK_SUCCESS && !initialData.empty()) {
        std::fprintf(stderr, "[pipeline-cache] the driver refused %s (Vulkan result %d); starting empty\n", path.string().c_str(), static_cast<int>(result));
        info.initialDataSize = 0;
        info.pInitialData = nullptr;
        savedBytes = 0;
        rejectedHeader = ownHeader;
        ownHeader.reset();
        result = create(context.device, &info, nullptr, &cache);
    }
    Check(result, "vkCreatePipelineCache");
    // The driver's own size for what it holds now (the loaded data re-serialized, or an empty
    // cache's header) is the baseline a save compares with.
    if (std::size_t live = 0; context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData")(context.device, cache, &live, nullptr) == VK_SUCCESS) savedBytes = live;
    if (!path.empty()) {
        saver = std::thread([this] { run(); });
        auto& objects = registry();
        std::lock_guard lock(objects.mutex);
        objects.objects.push_back(this);
    }
}

PipelineCache::~PipelineCache() {
    if (saver.joinable()) {
        {
            auto& objects = registry();
            std::lock_guard lock(objects.mutex);
            std::erase(objects.objects, this);
        }
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_all();
        saver.join();
        try {
            save(SaveReason::Teardown, Clock::now());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[pipeline-cache] teardown save failed: %s\n", error.what());
        }
    }
    context.Function<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache")(context.device, cache, nullptr);
}

void PipelineCache::load(std::vector<std::byte>& initialData) {
    std::vector<std::byte> file;
    if (!ShaderRecompiler::ReadWholeFile(path, file)) return;
    const auto reject = [&](const char* why) {
        std::fprintf(stderr, "[pipeline-cache] ignoring %s: %s\n", path.string().c_str(), why);
    };
    FileHeader header{};
    static_assert(sizeof(FileHeader) == HeaderBytes);
    if (file.size() < HeaderBytes) return reject("truncated");
    std::memcpy(&header, file.data(), HeaderBytes);
    if (header.magic != FileMagic || header.format != FileFormat) return reject("another format");
    rejectedHeader = header;
    if (header.dataBytes != file.size() - HeaderBytes) return reject("truncated");
    const auto data = std::span(file).subspan(HeaderBytes);
    if (ShaderRecompiler::HashBytes(data) != header.dataHash) return reject("checksum mismatch");
    struct VulkanHeader {
        std::uint32_t headerSize;
        std::uint32_t headerVersion;
        std::uint32_t vendorID;
        std::uint32_t deviceID;
        std::uint8_t uuid[VK_UUID_SIZE];
    } vulkan{};
    static_assert(sizeof(VulkanHeader) == 32);
    if (data.size() < sizeof(vulkan)) return reject("no Vulkan header");
    std::memcpy(&vulkan, data.data(), sizeof(vulkan));
    if (vulkan.headerSize < sizeof(vulkan) || vulkan.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || vulkan.vendorID != properties.vendorID || vulkan.deviceID != properties.deviceID || std::memcmp(vulkan.uuid, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) return reject("made for another device or driver");
    initialData.assign(data.begin(), data.end());
    savedBytes = initialData.size();
    ownHeader = header;
    rejectedHeader.reset();
    std::fprintf(stderr, "[pipeline-cache] loaded %.1f KiB from %s\n", static_cast<double>(initialData.size()) / 1024.0, path.string().c_str());
}

std::optional<PipelineCache::FileHeader> PipelineCache::readHeader() const {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return std::nullopt;
    const auto size = static_cast<std::streamoff>(file.tellg());
    if (size < static_cast<std::streamoff>(HeaderBytes)) return std::nullopt;
    file.seekg(0, std::ios::beg);
    FileHeader header{};
    if (!file.read(reinterpret_cast<char*>(&header), HeaderBytes)) return std::nullopt;
    if (header.magic != FileMagic || header.format != FileFormat || header.dataBytes != static_cast<std::uint64_t>(size) - HeaderBytes) return std::nullopt;
    return header;
}

bool PipelineCache::SaveIfDue(Clock::time_point now) {
    return save(SaveReason::Periodic, now);
}

bool PipelineCache::save(SaveReason reason, Clock::time_point now) {
    std::lock_guard fileLock(filesMutex());
    const auto getData = context.Function<PFN_vkGetPipelineCacheData>("vkGetPipelineCacheData");
    std::size_t size = 0;
    if (getData(context.device, cache, &size, nullptr) != VK_SUCCESS || size == 0) return false;
    if (size == savedBytes) {
        if (reason == SaveReason::Exit) std::fprintf(stderr, "[pipeline-cache] %s holds this run's pipelines already (%.1f KiB; exit)\n", path.string().c_str(), static_cast<double>(size) / 1024.0);
        return false;
    }
    if (reason == SaveReason::Periodic) {
        // Each save rewrites the whole file: between the interval's saves, only growth of a quarter
        // (at least 16 MiB) pays for another copy, hash and write.
        const auto growth = size - std::min(size, savedBytes);
        if (now - savedAt < PeriodicInterval() && growth < std::max<std::size_t>(savedBytes / 4u, std::size_t{16} << 20u)) return false;
    }
    if (const auto disk = readHeader(); disk && disk != ownHeader && disk != rejectedHeader && disk->dataBytes >= size) {
        if (!reportedForeign) {
            std::fprintf(stderr, "[pipeline-cache] not replacing %s (%.1f KiB, written by another cache object or process) with %.1f KiB (%s)\n", path.string().c_str(), static_cast<double>(disk->dataBytes) / 1024.0, static_cast<double>(size) / 1024.0, reasonName(reason));
            reportedForeign = true;
        }
        return false;
    }
    std::unique_ptr<std::byte[]> file;
    VkResult result = VK_INCOMPLETE;
    for (int attempt = 0; attempt < 3 && result == VK_INCOMPLETE; ++attempt) {
        if (attempt != 0 && (getData(context.device, cache, &size, nullptr) != VK_SUCCESS || size == 0)) return false;
        file = std::make_unique_for_overwrite<std::byte[]>(HeaderBytes + size);
        result = getData(context.device, cache, &size, file.get() + HeaderBytes);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) return false;
    }
    if (result != VK_SUCCESS) return false;
    const auto started = Clock::now();
    const FileHeader header{FileMagic, FileFormat, size, ShaderRecompiler::HashBytes(std::span(file.get() + HeaderBytes, size))};
    std::memcpy(file.get(), &header, HeaderBytes);
    if (!WriteFileDurably(path, std::span(file.get(), HeaderBytes + size))) {
        std::fprintf(stderr, "[pipeline-cache] cannot write %s (%s)\n", path.string().c_str(), reasonName(reason));
        return false;
    }
    savedBytes = size;
    ownHeader = header;
    reportedForeign = false;
    savedAt = Clock::now();
    std::fprintf(stderr, "[pipeline-cache] saved %.1f KiB to %s in %.1f ms (%s)\n", static_cast<double>(size) / 1024.0, path.string().c_str(), std::chrono::duration<double, std::milli>(savedAt - started).count(), reasonName(reason));
    return true;
}

void PipelineCache::SaveAllForExit(std::chrono::milliseconds bound) {
    auto& objects = registry();
    std::lock_guard registryLock(objects.mutex);
    const auto deadline = Clock::now() + bound;
    std::vector<std::pair<PipelineCache*, std::uint64_t>> pending;
    for (auto* object : objects.objects) {
        std::lock_guard lock(object->mutex);
        pending.emplace_back(object, ++object->exitRequests);
        object->wake.notify_all();
    }
    for (const auto& [object, ticket] : pending) {
        std::unique_lock lock(object->mutex);
        if (!object->exitSaved.wait_until(lock, deadline, [&] { return object->exitSaves >= ticket; })) {
            std::fprintf(stderr, "[pipeline-cache] the exit save of %s is still running after %lld ms; the exit goes on\n", object->path.string().c_str(), static_cast<long long>(bound.count()));
        }
    }
}

void PipelineCache::run() {
    std::unique_lock lock(mutex);
    for (;;) {
        wake.wait_for(lock, std::chrono::seconds(SaveIntervalSeconds), [&] { return stopping || exitSaves != exitRequests; });
        if (stopping) return;
        const auto requested = exitRequests;
        const bool exiting = exitSaves != requested;
        lock.unlock();
        try {
            save(exiting ? SaveReason::Exit : SaveReason::Periodic, Clock::now());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[pipeline-cache] %s save failed: %s\n", exiting ? "exit" : "periodic", error.what());
        }
        lock.lock();
        if (exiting) {
            exitSaves = requested;
            exitSaved.notify_all();
        }
    }
}

}
