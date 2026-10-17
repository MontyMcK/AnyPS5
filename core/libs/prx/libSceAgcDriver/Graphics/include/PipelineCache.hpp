#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace AgcDriver::Graphics {

class PipelineCache {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr int SaveIntervalSeconds = 10;
    static constexpr int PeriodicSaveSeconds = 150;
    static constexpr int ExitSaveSeconds = 30;

    enum class SaveReason { Periodic, Exit, Teardown };

    PipelineCache(const Context& context, const VkPhysicalDeviceProperties& properties);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    VkPipelineCache Handle() const { return cache; }
    const std::filesystem::path& Path() const { return path; }

    // The saver thread's step: writes the file when the driver data changed since this object last
    // loaded or wrote it and the periodic rule allows it at `now`. True when it wrote the file.
    bool SaveIfDue(Clock::time_point now);
    // Has every live object's saver thread write its changed data, waiting at most `bound` in all.
    // Runs before the exit tears anything down; the teardown save stays as a second chance.
    static void SaveAllForExit(std::chrono::milliseconds bound);
    // The periodic interval: APS5_PIPELINE_CACHE_SAVE_SECONDS (at least SaveIntervalSeconds), else
    // PeriodicSaveSeconds.
    static std::chrono::seconds PeriodicInterval();
    // Writes `bytes` to a temporary next to `path`, flushes it to the disk and renames it over `path`.
    static bool WriteFileDurably(const std::filesystem::path& path, std::span<const std::byte> bytes);

private:
    struct FileHeader {
        std::uint32_t magic;
        std::uint32_t format;
        std::uint64_t dataBytes;
        std::uint64_t dataHash;
        bool operator==(const FileHeader&) const = default;
    };

    void load(std::vector<std::byte>& initialData);
    std::optional<FileHeader> readHeader() const;
    bool save(SaveReason reason, Clock::time_point now);
    void run();

    Context context;
    VkPhysicalDeviceProperties properties{};
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::filesystem::path path;
    // Driver data size when this object last loaded or wrote the file.
    std::size_t savedBytes = 0;
    // Header of the file this object loaded or last wrote; another header on disk means another
    // object or process wrote it since, and this object then only replaces it with more data.
    std::optional<FileHeader> ownHeader;
    // Header of a file this object's load rejected (damaged, or refused by the driver): replaceable.
    std::optional<FileHeader> rejectedHeader;
    bool reportedForeign = false;
    Clock::time_point savedAt = Clock::now();
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable exitSaved;
    bool stopping = false;
    std::uint64_t exitRequests = 0;
    std::uint64_t exitSaves = 0;
    std::thread saver;
};

}

#endif
