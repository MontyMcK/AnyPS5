#include "ShaderWarmList.hpp"
#include "ShaderCacheDirectory.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "Optimization/ResourceProgram.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ShaderRecompiler::WarmList {

namespace {

constexpr std::uint32_t RecordMagic = 0x4c575041u;
constexpr std::uint32_t RecordFormat = 1u;
// Records not written or claimed by a first use for this long are removed at Start.
constexpr auto RecordLifetime = std::chrono::hours(24 * 30);
constexpr std::size_t MaxRequestBytes = 16u << 20u;
constexpr std::uint32_t MaxSpecializationValues = 4096u;

struct RecordHeader {
    std::uint32_t magic;
    std::uint32_t format;
    std::uint32_t module;
    std::uint32_t reserved;
    std::uint64_t identityLow;
    std::uint64_t identityHigh;
    std::uint64_t elapsedMs;
    std::uint64_t target;
    std::uint64_t payloadBytes;
    std::uint64_t payloadHash;
};
static_assert(sizeof(RecordHeader) == 64 && std::is_trivially_copyable_v<RecordHeader>);

const auto processStart = std::chrono::steady_clock::now();

thread_local bool registrationThread = false;
thread_local bool replayThread = false;

struct IdentityHash {
    std::size_t operator()(const Identity& identity) const {
        return static_cast<std::size_t>(identity.low ^ (identity.high * 0x9e3779b97f4a7c15ull));
    }
};

std::uint64_t mixHash(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    return hash * 0xff51afd7ed558ccdull;
}

// The parts of a target that tell one device and driver from another (the subgroup width and the
// per-request flags vary between the targets of one device).
std::uint64_t targetFingerprint(const SpirvTarget& target) {
    std::uint64_t hash = 0x5eed0005ull;
    hash = mixHash(hash, target.vulkanVersion);
    hash = mixHash(hash, target.spirvVersion);
    hash = mixHash(hash, target.bdaAbiVersion);
    hash = mixHash(hash, target.fragmentShaderBarycentricEnabled);
    hash = mixHash(hash, target.supportedCapabilities.size());
    for (const auto capability : target.supportedCapabilities) hash = mixHash(hash, capability);
    hash = mixHash(hash, target.supportedExtensions.size());
    for (const auto extension : target.supportedExtensions) hash = mixHash(hash, HashBytes(std::as_bytes(std::span(extension.data(), extension.size()))));
    return hash;
}

std::string hex(std::uint64_t value, int digits) {
    char text[17];
    std::snprintf(text, sizeof(text), "%0*llx", digits, static_cast<unsigned long long>(value));
    return text;
}

std::string recordName(const Identity& identity, bool module, std::uint64_t elapsedMs) {
    return std::string(module ? "m-" : "v-") + hex(identity.low, 16) + hex(identity.high, 16) + "-" + hex(std::min<std::uint64_t>(elapsedMs, 0xffffffffull), 8) + ".wl";
}

bool parseName(const std::string& name, Identity& identity, bool& module, std::uint64_t& elapsedMs) {
    // "v-" + 32 hex + "-" + 8 hex + ".wl"
    if (name.size() != 2 + 32 + 1 + 8 + 3 || (name[0] != 'v' && name[0] != 'm') || name[1] != '-' || name[34] != '-' || name.compare(43, 3, ".wl") != 0) return false;
    const auto parse = [&](std::size_t offset, std::size_t count, std::uint64_t& value) {
        value = 0;
        for (std::size_t index = offset; index < offset + count; ++index) {
            const char c = name[index];
            const int digit = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
            if (digit < 0) return false;
            value = (value << 4u) | static_cast<std::uint64_t>(digit);
        }
        return true;
    };
    module = name[0] == 'm';
    return parse(2, 16, identity.low) && parse(18, 16, identity.high) && parse(35, 8, elapsedMs);
}

template<typename TValue>
void put(std::vector<std::byte>& out, TValue value) requires std::is_trivially_copyable_v<TValue> {
    const auto offset = out.size();
    out.resize(offset + sizeof(value));
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

class Cursor {
public:
    explicit Cursor(std::span<const std::byte> bytes) : bytes(bytes) {}

    template<typename TValue>
    bool Get(TValue& value) requires std::is_trivially_copyable_v<TValue> {
        if (bytes.size() - position < sizeof(value)) return false;
        std::memcpy(&value, bytes.data() + position, sizeof(value));
        position += sizeof(value);
        return true;
    }

    bool Take(std::size_t count, std::span<const std::byte>& out) {
        if (bytes.size() - position < count) return false;
        out = bytes.subspan(position, count);
        position += count;
        return true;
    }

    [[nodiscard]] bool Done() const { return position == bytes.size(); }

private:
    std::span<const std::byte> bytes;
    std::size_t position = 0;
};

struct Listed {
    Identity identity;
    bool module = false;
    std::uint64_t elapsedMs = 0;
};

struct Job {
    enum class Kind { Write, Touch, Remove } kind;
    Identity identity;
    bool module = false;
    std::vector<std::byte> file;
    std::filesystem::path path;
};

struct AtomicCounters {
    std::atomic<std::uint64_t> replayedVariants{0}, replayedModules{0}, warmedVariants{0}, warmedModules{0}, replayFailures{0};
    std::atomic<std::uint64_t> claimedVariants{0}, claimedModules{0}, lateVariants{0}, lateModules{0}, lateQueuedVariants{0}, lateQueuedModules{0};
    std::atomic<std::uint64_t> recordedVariants{0}, recordedModules{0};
};

class State {
public:
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::condition_variable finished;
    bool started = false;
    std::atomic<bool> active{false};
    std::filesystem::path directory;
    // Identities on disk or recorded this session, per kind (0 variants, 1 modules).
    std::array<std::unordered_set<Identity, IdentityHash>, 2> known;
    // Records loaded at Start that the replay has not reached yet, per kind.
    std::array<std::unordered_set<Identity, IdentityHash>, 2> queued;
    std::array<std::unordered_map<Identity, std::filesystem::path, IdentityHash>, 2> paths;
    std::deque<Job> jobs;
    bool writing = false;
    bool writerRunning = false;
    std::vector<Listed> records;
    std::atomic<std::size_t> next{0};
    unsigned workers = 0;
    bool replayDone = true;
    ReplayOptions options;
    std::optional<std::uint64_t> target;
    std::uint64_t loadedVariants = 0;
    std::uint64_t loadedModules = 0;
    std::uint64_t rejected = 0;
    std::atomic<std::uint64_t> otherDevice{0};
    std::chrono::steady_clock::time_point replayStarted{};
    std::chrono::steady_clock::time_point replayFinished{};
    AtomicCounters counters;
    std::atomic<std::int64_t> lastReport{0};
    std::mutex reportMutex;
    Counters reported;
};

State& state() {
    static auto* instance = new State();
    return *instance;
}

void report(bool force);

void writerMain() {
    auto& self = state();
    std::unique_lock lock(self.mutex);
    for (;;) {
        self.wake.wait_for(lock, std::chrono::seconds(10), [&] { return !self.jobs.empty(); });
        while (!self.jobs.empty()) {
            auto job = std::move(self.jobs.front());
            self.jobs.pop_front();
            self.writing = true;
            lock.unlock();
            std::error_code error;
            if (job.kind == Job::Kind::Write) {
                const auto path = job.path;
                if (WriteFileAtomically(path, job.file)) {
                    (job.module ? self.counters.recordedModules : self.counters.recordedVariants).fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard pathLock(self.mutex);
                    self.paths[job.module ? 1 : 0].emplace(job.identity, path);
                }
            } else if (job.kind == Job::Kind::Touch) {
                std::filesystem::last_write_time(job.path, std::filesystem::file_time_type::clock::now(), error);
            } else {
                std::filesystem::remove(job.path, error);
            }
            job = {};
            lock.lock();
            self.writing = false;
        }
        self.idle.notify_all();
        lock.unlock();
        report(false);
        lock.lock();
    }
}

// Under self.mutex.
void enqueue(State& self, Job job) {
    self.jobs.push_back(std::move(job));
    if (!self.writerRunning) {
        self.writerRunning = true;
        std::thread(writerMain).detach();
    }
    self.wake.notify_one();
}

std::vector<std::byte> encodeRecord(const RecompileRequest& request, const Identity& identity, bool module, std::uint64_t elapsedMs, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants) {
    // A static preparation never reads guest memory (it only enters runtime materialization), so
    // the record leaves it out: it would only add the resources the first use happened to read.
    RecompileRequest stripped = request;
    stripped.context.memory = {};
    const auto text = RequestSerializer{}.Serialize(stripped);
    if (text.size() > MaxRequestBytes) return {};
    std::vector<std::byte> payload;
    payload.reserve(text.size() + 64 + classes.size() * 4 + constants.size() * 8);
    put<std::uint64_t>(payload, text.size());
    const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
    payload.insert(payload.end(), bytes.begin(), bytes.end());
    if (module) {
        put<std::uint64_t>(payload, classes.size());
        for (const auto value : classes) put(payload, value);
        put<std::uint64_t>(payload, constants.size());
        for (const auto& constant : constants) {
            put(payload, constant.id);
            put(payload, constant.value);
        }
    }
    const RecordHeader header{RecordMagic, RecordFormat, module ? 1u : 0u, 0u, identity.low, identity.high, elapsedMs, targetFingerprint(request.target), payload.size(), HashBytes(payload, 0x5eed0006ull)};
    std::vector<std::byte> file(sizeof(header) + payload.size());
    std::memcpy(file.data(), &header, sizeof(header));
    std::memcpy(file.data() + sizeof(header), payload.data(), payload.size());
    return file;
}

void record(const RecompileRequest& request, const Identity& identity, bool module, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants) {
    auto& self = state();
    const auto kind = module ? 1 : 0;
    {
        std::lock_guard lock(self.mutex);
        if (!self.known[kind].insert(identity).second) return;
    }
    const auto elapsedMs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - processStart).count());
    std::vector<std::byte> file;
    try {
        file = encodeRecord(request, identity, module, elapsedMs, classes, constants);
    } catch (const std::exception&) {
        file.clear();
    }
    if (file.empty()) return;
    std::lock_guard lock(self.mutex);
    enqueue(self, {Job::Kind::Write, identity, module, std::move(file), self.directory / recordName(identity, module, elapsedMs)});
}

void replayOne(State& self, const Listed& listed) {
    std::filesystem::path path;
    {
        std::lock_guard lock(self.mutex);
        const auto found = self.paths[listed.module ? 1 : 0].find(listed.identity);
        if (found == self.paths[listed.module ? 1 : 0].end()) return;
        path = found->second;
    }
    std::vector<std::byte> file;
    bool valid = ReadWholeFile(path, file) && file.size() >= sizeof(RecordHeader);
    RecordHeader header{};
    if (valid) {
        std::memcpy(&header, file.data(), sizeof(header));
        const auto payload = std::span<const std::byte>(file).subspan(sizeof(header));
        valid = header.magic == RecordMagic && header.format == RecordFormat && (header.module != 0) == listed.module && header.identityLow == listed.identity.low && header.identityHigh == listed.identity.high && header.payloadBytes == payload.size() && HashBytes(payload, 0x5eed0006ull) == header.payloadHash;
    }
    const auto discard = [&] {
        self.counters.replayFailures.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(self.mutex);
        enqueue(self, {Job::Kind::Remove, listed.identity, listed.module, {}, path});
    };
    if (!valid) {
        discard();
        return;
    }
    if (self.target && *self.target != header.target) {
        self.otherDevice.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    Cursor cursor(std::span<const std::byte>(file).subspan(sizeof(header)));
    std::uint64_t textBytes = 0;
    std::span<const std::byte> text;
    std::vector<std::uint32_t> classes;
    std::vector<PipelineSpecializationConstant> constants;
    bool parsed = cursor.Get(textBytes) && textBytes <= MaxRequestBytes && cursor.Take(static_cast<std::size_t>(textBytes), text);
    if (parsed && listed.module) {
        std::uint64_t count = 0;
        parsed = cursor.Get(count) && count <= MaxSpecializationValues;
        for (std::uint64_t index = 0; parsed && index < count; ++index) {
            std::uint32_t value = 0;
            parsed = cursor.Get(value);
            classes.push_back(value);
        }
        parsed = parsed && cursor.Get(count) && count <= MaxSpecializationValues;
        for (std::uint64_t index = 0; parsed && index < count; ++index) {
            PipelineSpecializationConstant constant{};
            parsed = cursor.Get(constant.id) && cursor.Get(constant.value);
            constants.push_back(constant);
        }
    }
    if (!parsed || !cursor.Done()) {
        discard();
        return;
    }
    try {
        const auto deserialized = RequestSerializer{}.Deserialize(std::string_view(reinterpret_cast<const char*>(text.data()), text.size()));
        const auto handle = PrepareShader(deserialized.request);
        if (listed.module) PrewarmSpecializedModule(deserialized.request, *handle, classes, constants);
    } catch (const std::exception& error) {
        const auto failures = self.counters.replayFailures.load(std::memory_order_relaxed);
        if (failures < 8) {
            const std::string message = error.what();
            std::fprintf(stderr, "[shader-warm] dropped %s: %.200s\n", path.filename().string().c_str(), message.substr(0, message.find("\nRecompileRequest:")).c_str());
        }
        discard();
        return;
    }
    (listed.module ? self.counters.replayedModules : self.counters.replayedVariants).fetch_add(1, std::memory_order_relaxed);
}

void* replayMain(void*) {
    auto& self = state();
    replayThread = true;
#ifdef _WIN32
    if (self.options.lowPriority) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
    for (;;) {
        const auto index = self.next.load(std::memory_order_relaxed);
        if (index >= self.records.size()) break;
        if (self.options.waitTurn && !self.options.waitTurn()) break;
        const auto claimed = self.next.fetch_add(1, std::memory_order_relaxed);
        if (claimed >= self.records.size()) break;
        const auto& listed = self.records[claimed];
        replayOne(self, listed);
        {
            std::lock_guard lock(self.mutex);
            self.queued[listed.module ? 1 : 0].erase(listed.identity);
        }
        report(false);
    }
    bool last = false;
    {
        std::lock_guard lock(self.mutex);
        last = --self.workers == 0;
        if (last) {
            self.replayDone = true;
            self.replayFinished = std::chrono::steady_clock::now();
            self.queued[0].clear();
            self.queued[1].clear();
        }
    }
    if (last) {
        report(true);
        self.finished.notify_all();
    }
    return nullptr;
}

void report(bool force) {
    auto& self = state();
    if (!self.active.load(std::memory_order_relaxed)) return;
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = self.lastReport.load(std::memory_order_relaxed);
    const auto interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(10)).count();
    if (!force && (now - last < interval || !self.lastReport.compare_exchange_strong(last, now, std::memory_order_relaxed))) return;
    if (force) self.lastReport.store(now, std::memory_order_relaxed);
    std::lock_guard lock(self.reportMutex);
    const auto totals = Totals();
    const auto same = [](const Counters& a, const Counters& b) {
        return a.replayedVariants == b.replayedVariants && a.replayedModules == b.replayedModules && a.warmedVariants == b.warmedVariants && a.warmedModules == b.warmedModules && a.replayFailures == b.replayFailures && a.claimedVariants == b.claimedVariants && a.claimedModules == b.claimedModules && a.lateVariants == b.lateVariants && a.lateModules == b.lateModules && a.recordedVariants == b.recordedVariants && a.recordedModules == b.recordedModules && a.replayDone == b.replayDone && a.otherDevice == b.otherDevice;
    };
    if (!force && same(totals, self.reported)) return;
    std::fprintf(stderr, "[shader-warm] replay %s: variants %llu/%llu (%llu prepared ahead), modules %llu/%llu (%llu built ahead), %llu dropped, %llu other device, %.1f s; first uses: variants %llu served warm, %llu prepared late (%llu still queued); modules %llu served warm, %llu built late (%llu still queued); recorded %llu variants, %llu modules\n",
                 totals.replayDone ? "done" : "running",
                 static_cast<unsigned long long>(totals.replayedVariants), static_cast<unsigned long long>(totals.loadedVariants), static_cast<unsigned long long>(totals.warmedVariants),
                 static_cast<unsigned long long>(totals.replayedModules), static_cast<unsigned long long>(totals.loadedModules), static_cast<unsigned long long>(totals.warmedModules),
                 static_cast<unsigned long long>(totals.replayFailures), static_cast<unsigned long long>(totals.otherDevice), totals.replaySeconds,
                 static_cast<unsigned long long>(totals.claimedVariants), static_cast<unsigned long long>(totals.lateVariants), static_cast<unsigned long long>(totals.lateQueuedVariants),
                 static_cast<unsigned long long>(totals.claimedModules), static_cast<unsigned long long>(totals.lateModules), static_cast<unsigned long long>(totals.lateQueuedModules),
                 static_cast<unsigned long long>(totals.recordedVariants), static_cast<unsigned long long>(totals.recordedModules));
    self.reported = totals;
}

}

void Start(ReplayOptions options) {
    auto& self = state();
    std::unique_lock lock(self.mutex);
    if (self.started) return;
    self.started = true;
    const auto root = ShaderCacheDirectory();
    if (root.empty()) return;
    self.directory = root / "warm";
    std::error_code error;
    std::filesystem::create_directories(self.directory, error);
    const auto started = std::chrono::steady_clock::now();
    const auto now = std::filesystem::file_time_type::clock::now();
    for (std::filesystem::directory_iterator it(self.directory, error), end; !error && it != end; it.increment(error)) {
        const auto name = it->path().filename().string();
        Identity identity;
        bool module = false;
        std::uint64_t elapsedMs = 0;
        std::error_code entryError;
        if (!parseName(name, identity, module, elapsedMs)) {
            // Interrupted writes leave temporaries; anything else is not ours to remove.
            if (name.find(".wl.tmp.") != std::string::npos) {
                const auto written = it->last_write_time(entryError);
                if (!entryError && now - written > std::chrono::hours(1)) std::filesystem::remove(it->path(), entryError);
            }
            continue;
        }
        const auto written = it->last_write_time(entryError);
        if (!entryError && now - written > RecordLifetime) {
            std::filesystem::remove(it->path(), entryError);
            ++self.rejected;
            continue;
        }
        const auto kind = module ? 1 : 0;
        if (!self.known[kind].insert(identity).second) {
            // The same record under another first-use time: keep the earlier one.
            ++self.rejected;
            continue;
        }
        self.paths[kind].emplace(identity, it->path());
        self.queued[kind].insert(identity);
        self.records.push_back({identity, module, elapsedMs});
        ++(module ? self.loadedModules : self.loadedVariants);
    }
    // In first-use order: what a session needed first (the boot and the menus) is replayed first.
    std::stable_sort(self.records.begin(), self.records.end(), [](const Listed& left, const Listed& right) { return left.elapsedMs < right.elapsedMs; });
    self.options = std::move(options);
    if (self.options.target) self.target = targetFingerprint(*self.options.target);
    // The target's spans belong to the caller's device: only the fingerprint is kept.
    self.options.target.reset();
    self.active.store(true, std::memory_order_relaxed);
    const auto listMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    const auto threads = std::max(1u, self.options.threads);
    std::fprintf(stderr, "[shader-warm] %s: %llu variant and %llu module records (%llu removed) listed in %.1f ms; replaying on %u low-priority threads once shader registration is idle\n", self.directory.string().c_str(), static_cast<unsigned long long>(self.loadedVariants), static_cast<unsigned long long>(self.loadedModules), static_cast<unsigned long long>(self.rejected), listMs, threads);
    if (self.records.empty()) return;
    self.replayDone = false;
    self.replayStarted = std::chrono::steady_clock::now();
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    // A recompile walks deep recursive passes: the same stack as the driver's preparation pools.
    pthread_attr_setstacksize(&attributes, 64u << 20u);
    for (unsigned index = 0; index < threads; ++index) {
        pthread_t thread;
        if (pthread_create(&thread, &attributes, &replayMain, nullptr) == 0) {
            pthread_detach(thread);
            ++self.workers;
        }
    }
    pthread_attr_destroy(&attributes);
    if (self.workers == 0) {
        self.replayDone = true;
        self.queued[0].clear();
        self.queued[1].clear();
    }
}

bool Active() {
    return state().active.load(std::memory_order_relaxed);
}

void WaitForReplay() {
    auto& self = state();
    std::unique_lock lock(self.mutex);
    self.finished.wait(lock, [&] { return self.replayDone; });
}

void Flush() {
    auto& self = state();
    std::unique_lock lock(self.mutex);
    self.idle.wait(lock, [&] { return self.jobs.empty() && !self.writing; });
}

Counters Totals() {
    auto& self = state();
    Counters totals;
    {
        std::lock_guard lock(self.mutex);
        totals.loadedVariants = self.loadedVariants;
        totals.loadedModules = self.loadedModules;
        totals.rejected = self.rejected;
        totals.started = self.active.load(std::memory_order_relaxed);
        totals.replayDone = self.replayDone;
        if (self.replayStarted != std::chrono::steady_clock::time_point{}) totals.replaySeconds = std::chrono::duration<double>((self.replayDone ? self.replayFinished : std::chrono::steady_clock::now()) - self.replayStarted).count();
    }
    const auto& counters = self.counters;
    totals.otherDevice = self.otherDevice.load(std::memory_order_relaxed);
    totals.replayedVariants = counters.replayedVariants.load(std::memory_order_relaxed);
    totals.replayedModules = counters.replayedModules.load(std::memory_order_relaxed);
    totals.warmedVariants = counters.warmedVariants.load(std::memory_order_relaxed);
    totals.warmedModules = counters.warmedModules.load(std::memory_order_relaxed);
    totals.replayFailures = counters.replayFailures.load(std::memory_order_relaxed);
    totals.claimedVariants = counters.claimedVariants.load(std::memory_order_relaxed);
    totals.claimedModules = counters.claimedModules.load(std::memory_order_relaxed);
    totals.lateVariants = counters.lateVariants.load(std::memory_order_relaxed);
    totals.lateModules = counters.lateModules.load(std::memory_order_relaxed);
    totals.lateQueuedVariants = counters.lateQueuedVariants.load(std::memory_order_relaxed);
    totals.lateQueuedModules = counters.lateQueuedModules.load(std::memory_order_relaxed);
    totals.recordedVariants = counters.recordedVariants.load(std::memory_order_relaxed);
    totals.recordedModules = counters.recordedModules.load(std::memory_order_relaxed);
    return totals;
}

std::filesystem::path Directory() {
    auto& self = state();
    std::lock_guard lock(self.mutex);
    return self.directory;
}

RegistrationScope::RegistrationScope() : previous(registrationThread) {
    registrationThread = true;
}

RegistrationScope::~RegistrationScope() {
    registrationThread = previous;
}

namespace Detail {

bool Replaying() {
    return replayThread;
}

bool FirstUse() {
    return !registrationThread && !replayThread && state().active.load(std::memory_order_relaxed);
}

Identity ModuleIdentity(const Identity& variant, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants) {
    std::vector<std::uint64_t> words;
    words.reserve(4 + classes.size() + constants.size());
    words.push_back(variant.low);
    words.push_back(variant.high);
    words.push_back(classes.size());
    for (const auto value : classes) words.push_back(value);
    words.push_back(constants.size());
    for (const auto& constant : constants) words.push_back((static_cast<std::uint64_t>(constant.id) << 32u) | constant.value);
    const auto bytes = std::as_bytes(std::span<const std::uint64_t>(words));
    return {HashBytes(bytes, 0x5eed0007ull), HashBytes(bytes, 0x5eed0008ull)};
}

bool Queued(const Identity& identity, bool module) {
    auto& self = state();
    std::lock_guard lock(self.mutex);
    return self.queued[module ? 1 : 0].contains(identity);
}

void RecordVariant(const RecompileRequest& request, const Identity& identity) {
    record(request, identity, false, {}, {});
}

void RecordModule(const RecompileRequest& request, const Identity& identity, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants) {
    record(request, ModuleIdentity(identity, classes, constants), true, classes, constants);
}

void NoteWarmed(bool module) {
    (module ? state().counters.warmedModules : state().counters.warmedVariants).fetch_add(1, std::memory_order_relaxed);
}

void NoteClaimed(const Identity& identity, bool module) {
    auto& self = state();
    (module ? self.counters.claimedModules : self.counters.claimedVariants).fetch_add(1, std::memory_order_relaxed);
    // A claimed record stays: its age restarts.
    std::lock_guard lock(self.mutex);
    const auto& paths = self.paths[module ? 1 : 0];
    if (const auto found = paths.find(identity); found != paths.end()) enqueue(self, {Job::Kind::Touch, identity, module, {}, found->second});
}

void NoteLate(const Identity& identity, bool module) {
    auto& self = state();
    (module ? self.counters.lateModules : self.counters.lateVariants).fetch_add(1, std::memory_order_relaxed);
    if (Queued(identity, module)) (module ? self.counters.lateQueuedModules : self.counters.lateQueuedVariants).fetch_add(1, std::memory_order_relaxed);
    report(false);
}

}

}
