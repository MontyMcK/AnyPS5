#ifndef CORE_SHADER_RECOMPILER_SHADERWARMLIST_HPP
#define CORE_SHADER_RECOMPILER_SHADERWARMLIST_HPP

#include "Recompiler.hpp"
#include "PipelineSpecialization.hpp"
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>

// The shader warm list: what a session prepared outside shader registration (variants a draw,
// dispatch or ABI resolution needed for its own state, and the specialized modules its pipelines
// used), kept beside the shader disk cache in a form that outlives a recompiler change (the
// serialized request, without guest memory, which a static preparation never reads). A later
// session replays the list on low-priority threads once registration is idle, so the recompiler's
// source/variant/module caches (and the disk cache of the current version) hold those artifacts
// before their first use. A replay only ever adds entries under the exact keys a draw-time lookup
// uses (PrepareShader / the specialized-module key); a record that no longer matches anything is
// compiled to no effect, and the draw-time path still prepares whatever it actually needs.
namespace ShaderRecompiler {

struct SourceHandle;

namespace WarmList {

struct Identity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;

    bool operator==(const Identity&) const = default;
};

struct ReplayOptions {
    unsigned threads = 2;
    // Called before each record; blocks while the replay should yield (e.g. registration is busy)
    // and returns false to stop the replay.
    std::function<bool()> waitTurn;
    // Records prepared for another device (Vulkan/SPIR-V version, BDA ABI, capabilities or
    // extensions differ from this target's) are skipped. Unset: every record is replayed.
    std::optional<SpirvTarget> target;
    bool lowPriority = true;
};

struct Counters {
    // Records found at Start (after age and format checks) and records rejected or skipped.
    std::uint64_t loadedVariants = 0;
    std::uint64_t loadedModules = 0;
    std::uint64_t rejected = 0;
    std::uint64_t otherDevice = 0;
    // Replay progress: records done, records that prepared something new (not already in memory),
    // and records that failed (their files are removed).
    std::uint64_t replayedVariants = 0;
    std::uint64_t replayedModules = 0;
    std::uint64_t warmedVariants = 0;
    std::uint64_t warmedModules = 0;
    std::uint64_t replayFailures = 0;
    double replaySeconds = 0.0;
    // Preparations outside registration and replay: first uses served by a replayed artifact
    // (claimed), and first uses that prepared their artifact themselves (late; queued: the replay
    // had the record but had not reached it yet).
    std::uint64_t claimedVariants = 0;
    std::uint64_t claimedModules = 0;
    std::uint64_t lateVariants = 0;
    std::uint64_t lateModules = 0;
    std::uint64_t lateQueuedVariants = 0;
    std::uint64_t lateQueuedModules = 0;
    // Records written this session.
    std::uint64_t recordedVariants = 0;
    std::uint64_t recordedModules = 0;
    bool started = false;
    bool replayDone = false;
};

// Enables recording and starts the replay of the records under Directory() (once per process;
// later calls do nothing). Without a shader cache directory the warm list stays inert.
void Start(ReplayOptions options);
// Whether Start enabled the warm list.
[[nodiscard]] bool Active();
// Blocks until the replay has finished (tests).
void WaitForReplay();
// Blocks until queued record writes are on disk (tests).
void Flush();
[[nodiscard]] Counters Totals();
[[nodiscard]] std::filesystem::path Directory();

// Preparations in scope belong to shader registration: registration prepares them at every boot,
// so they are neither recorded nor counted as first uses.
class RegistrationScope {
public:
    RegistrationScope();
    ~RegistrationScope();
    RegistrationScope(const RegistrationScope&) = delete;
    RegistrationScope& operator=(const RegistrationScope&) = delete;

private:
    bool previous;
};

// Hooks for the recompiler's caches (Recompiler.cpp).
namespace Detail {

// Whether this thread is replaying records.
[[nodiscard]] bool Replaying();
// Whether a preparation on this thread is a first use (warm list active, not registration, not replay).
[[nodiscard]] bool FirstUse();
[[nodiscard]] Identity ModuleIdentity(const Identity& variant, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants);
[[nodiscard]] bool Queued(const Identity& identity, bool module);
void RecordVariant(const RecompileRequest& request, const Identity& identity);
void RecordModule(const RecompileRequest& request, const Identity& identity, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants);
void NoteWarmed(bool module);
void NoteClaimed(const Identity& identity, bool module);
void NoteLate(const Identity& identity, bool module);

}

// The specialized module a draw would build for the prepared variant with these vertex classes
// and constants, made (or loaded from the disk cache) ahead of that draw. Defined in Recompiler.cpp.
// `request` is the record's (its layout is the variant's; its target is the module's).
void PrewarmSpecializedModule(const RecompileRequest& request, const SourceHandle& handle, std::span<const std::uint32_t> classes, std::span<const PipelineSpecializationConstant> constants);

}

}

#endif
