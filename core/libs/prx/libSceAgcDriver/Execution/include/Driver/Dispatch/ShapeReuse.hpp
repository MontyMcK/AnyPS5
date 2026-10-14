#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_SHAPEREUSE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_SHAPEREUSE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

// Dispatch shape reuse (live `dispatch_shape_reuse`, env APS5_DISPATCH_SHAPE_REUSE=1; off by
// default). A dispatch whose key differs from every cached one only in user words that the walk
// takes as raw read base pointers (or copies unchanged into a buffer descriptor or a flat slot, or
// never reads: ShapePlan::uses) reuses an earlier capture of the same program: each read the
// earlier walk made moves by its pointer word's difference, the moved words must hold what the
// earlier walk read (pure flat-slot leaves may differ: refreshed as data hits do; with live
// `dispatch_shape_refresh_descriptors`, words the walk only copies into buffer descriptors too),
// and the snapshot is the earlier one with the user data, the copied words and the refreshed words
// replaced; Materialize then builds the same result a full capture would. Anything that cannot be
// proved falls back to the full capture (ShapeReuse.cpp). Everything here is the driver's own
// reading of the resource plan: nothing is added to the plan or the recompiler, so the shader disk
// cache key is unchanged.

// How the walk consumes one user data word. PointerLow: the word is the low argument of raw read
// handles and nothing else, so a new value moves those reads by the difference; bufferDwords
// (buffer index, dword) and flatOffsets: the word is copied unchanged into those snapshot words.
// Opaque: any other use, so a capture is only valid for the same value. A word with no flag and no
// copy is not read by the walk.
struct UserWordUse {
    static constexpr std::uint8_t PointerLow = 1u;
    static constexpr std::uint8_t Opaque = 2u;
    std::uint8_t flags = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bufferDwords;
    std::vector<std::uint32_t> flatOffsets;
};

// The snapshot words a refreshable raw read's value is copied into, and nowhere else: a pure
// flat-slot leaf (one flat offset) or a descriptor leaf (buffer dwords, and the flat offsets of the
// slots whose value it is).
struct RefreshTarget {
    bool descriptor = false;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bufferDwords;
    std::vector<std::uint32_t> flatOffsets;
};

// The driver's analysis of one resource plan (ShapeReuse.cpp classifyPlan), made once per plan.
struct ShapePlan {
    bool known = false;
    const char* refusal = nullptr;
    std::vector<UserWordUse> uses;
    std::vector<RefreshTarget> targets;
};

// One raw read of the remembered walk: its address, the dword it read, the user word that is the
// low half of its base (NoWord if none), the flat offset of a pure-slot leaf (NoSlot if none) and
// its refresh target in ShapePlan::targets (NoTarget if the word must keep its value).
struct ShapeRead {
    static constexpr std::uint32_t NoWord = 0xffffffffu;
    static constexpr std::uint32_t NoSlot = 0xffffffffu;
    static constexpr std::uint32_t NoTarget = 0xffffffffu;
    std::uint64_t address = 0;
    std::uint32_t value = 0;
    std::uint32_t userWord = NoWord;
    std::uint32_t leafSlot = NoSlot;
    std::uint32_t target = NoTarget;
};

// A remembered full capture: what a later dispatch of the same shape relocates.
struct ShapeCandidate {
    std::uint64_t staticKey = 0;
    // Weak: a candidate does not keep a replaced shader's snapshot alive, and Table drops the
    // candidates whose shader expired.
    std::weak_ptr<const ShaderSnapshot> shader;
    // The walk's shader base (a draw stage's entry point; dispatches leave it 0: their static key
    // and registered shader fix it).
    std::uint64_t shaderBase = 0;
    std::vector<std::uint32_t> userData;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::shared_ptr<const ShapePlan> plan;
    // The prepared source the capture belongs to (PreparedShaderInvocation::Request().shader.code).
    const std::uint32_t* sourceCode = nullptr;
    std::size_t sourceWords = 0;
    std::vector<ShapeRead> reads;
    // The pure-slot leaves in walk order: (flat offset, index into reads).
    std::vector<std::pair<std::uint32_t, std::size_t>> leaves;
};

// What the driver learned about a program under one static key (the dispatch key without the user
// words; a draw stage's program, stage and layout): which words the shape key leaves out
// (UserWordUse not Opaque), or why none can be.
struct ShapeProgram {
    std::uint64_t address = 0;
    bool known = false;
    const char* refusal = nullptr;
    std::vector<std::uint8_t> excluded;
    std::vector<std::uint8_t> classes;
    std::vector<std::uint32_t> lastUserData;
    // Table::ShouldLearn: captures in a row that could not be remembered, lookups in a row that
    // found no entry, and captures not learned since; Table's clock at the last use.
    std::uint32_t refusedInARow = 0;
    std::uint32_t noEntryInARow = 0;
    std::uint32_t skipped = 0;
    std::uint64_t touched = 0;
};

enum class ShapeRefusal : std::uint32_t {
    Unlearned,
    Ineligible,
    NoEntry,
    KeyCollision,
    Relocation,
    Conflict,
    Differing,
    Inaccessible,
    Pending,
    Inconsistent,
    Source,
    Materialize,
    Count
};

// For tests and the [shape-reuse] / [draw-shape] lines: totals since start.
struct ShapeReuseCounts {
    std::uint64_t lookups = 0;
    std::uint64_t hits = 0;
    std::uint64_t dataRefreshes = 0;
    std::uint64_t descriptorRefreshes = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(ShapeRefusal::Count)> refusals{};
    std::uint64_t remembered = 0;
    std::uint64_t notRemembered = 0;
    std::uint64_t verified = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t fullCaptures = 0;
    std::uint64_t hitNanoseconds = 0;
    std::uint64_t fullNanoseconds = 0;
};
ShapeReuseCounts DispatchShapeReuseCounts();
// Draw shape reuse (Draw/DrawShapeReuse.cpp): counted per draw stage.
ShapeReuseCounts DrawShapeReuseCounts();
// A full capture made while shape reuse is on (Dispatch.cpp), for the [shape-reuse] line.
void CountShapeFullCapture(double microseconds);

// What dispatch and draw shape reuse share (ShapeReuse.cpp): the plan analysis, the remembered
// reads, the relocation and the rebuilt capture, the tables and the counters.
namespace ShapeCore {

using Regions = std::span<const ShaderRecompiler::MemoryRegion>;

enum class Kind : std::uint8_t { Dispatch, Draw };

// What one full capture teaches: `learnable` when its plan is known and it resolved no user-data
// calls (the program's excluded words can be learned), `refusal` when the capture itself cannot be
// remembered (null: `reads` and `leaves` are its attributed reads).
struct Learned {
    bool learnable = false;
    const char* refusal = nullptr;
    std::shared_ptr<const ShapePlan> plan;
    std::vector<ShapeRead> reads;
    std::vector<std::pair<std::uint32_t, std::size_t>> leaves;
};
// `readRegions` (called only for a known plan): every region the capture read from (the registered
// regions and its own page reads, nothing else); `registered`: the registered regions among them
// (the walk serves those from the registered snapshots, not from guest memory).
Learned Learn(const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture, bool capturedCalls, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, const std::function<std::vector<ShaderRecompiler::MemoryRegion>()>& readRegions, Regions registered);

// The shape key: the static key and every user word not excluded.
std::uint64_t Key(std::span<const std::uint8_t> excluded, std::uint64_t staticKey, std::span<const std::uint32_t> userData);

// Where a candidate's reads move for `userData`: the user words that differ (`changed`) and each
// read's new address. ShapeRefusal::Count when every changed word may move, every moved read stays
// inside the walk's address width and none moves into `registered` (the dispatch path's registered
// regions, which it does not validate; empty for draws, which read every word again).
ShapeRefusal Relocate(const ShapeCandidate& candidate, std::span<const std::uint32_t> userData, Regions registered, std::vector<std::uint8_t>& changed, std::vector<std::uint64_t>& addresses);

// Whether `read` may return another word: it only copies its value into a refresh target (a pure
// leaf's flat slot, or with `descriptors` a descriptor leaf's buffer dwords and slots).
bool Refreshable(const ShapeCandidate& candidate, const ShapeRead& read, bool descriptors);

// The capture a full walk builds for `userData` when the moved reads return the remembered words
// except the `refreshed` ones ((new address, word), sorted by address, each address read only by
// refreshable reads under `descriptors`): the remembered capture with the new user data, the copied
// user words, the refreshed targets and the moved read trace. ShapeRefusal::Count on success.
ShapeRefusal Rebuild(const ShapeCandidate& candidate, std::span<const std::uint32_t> userData, std::span<const std::uint8_t> changed, std::span<const std::uint64_t> addresses, std::span<const std::pair<std::uint64_t, std::uint32_t>> refreshed, bool descriptors, std::shared_ptr<ShaderRecompiler::ResourceCapture>& rebuilt, bool& descriptorRefreshed);

// What differs between a reused and a full capture's results, or null when nothing does.
const char* Difference(const ShaderRecompiler::RecompileResult& reused, const ShaderRecompiler::RecompileResult& full, const ShaderRecompiler::ResourceCapture& reusedCapture, const ShaderRecompiler::ResourceCapture& fullCapture, Regions reusedRegions, Regions fullRegions);

// The learned programs and the remembered captures of one kind of reuse.
class Table {
public:
    // `maxKeys` shape keys, least recently used evicted first, of `maxCandidates` captures each;
    // at most 4 * `maxKeys` programs; with `learnBackoff`, ShouldLearn backs off.
    Table(std::size_t maxKeys, std::size_t maxCandidates, bool learnBackoff) : maxKeys(maxKeys), maxCandidates(maxCandidates), learnBackoff(learnBackoff) {}
    // Whether a full capture under `staticKey` is worth learning: after four captures in a row that
    // could not be remembered, or eight lookups in a row that found no entry (a key word that never
    // repeats), only every 64th is (Learn's attribution is not free).
    bool ShouldLearn(std::uint64_t staticKey);
    // Records what a capture under `staticKey` taught and remembers `candidate` (its staticKey,
    // userData and plan set here) unless `learned.refusal`.
    void Remember(Kind kind, std::uint64_t address, std::uint64_t staticKey, std::span<const std::uint32_t> userData, Learned learned, std::shared_ptr<ShapeCandidate> candidate);
    // The remembered captures of `userData`'s shape under `staticKey`, newest first; on a refusal
    // (Unlearned, Ineligible, NoEntry) `candidates` is empty and the refusal is counted.
    ShapeRefusal Find(Kind kind, std::uint64_t address, std::uint64_t staticKey, std::span<const std::uint32_t> userData, std::vector<std::shared_ptr<const ShapeCandidate>>& candidates);

private:
    struct Entry {
        std::vector<std::shared_ptr<const ShapeCandidate>> candidates;
        std::list<std::uint64_t>::iterator order;
    };
    void touch(Entry& entry);
    void trim();
    std::size_t maxKeys;
    std::size_t maxCandidates;
    bool learnBackoff;
    std::mutex mutex;
    std::uint64_t clock = 0;
    std::uint64_t remembers = 0;
    std::unordered_map<std::uint64_t, ShapeProgram> programs;
    std::unordered_map<std::uint64_t, Entry> entries;
    std::list<std::uint64_t> order;
};

// Counters of one kind of reuse: totals and, with APS5_PROFILE_DRAW, the 10 s window reported by
// Report as the [shape-reuse] or [draw-shape] line.
bool Profiled();
void CountLookup(Kind kind, std::uint64_t address);
void CountRefusal(Kind kind, std::uint64_t address, ShapeRefusal reason);
// A hit's totals and its 10 s window entry (the dispatch path counts the window after its
// verification, as it did before draws shared this).
void CountHitTotals(Kind kind, std::uint64_t nanoseconds, bool refreshed, bool descriptorRefreshed);
void CountHitWindow(Kind kind, std::uint64_t address, std::uint64_t nanoseconds, bool refreshed, bool descriptorRefreshed);
// A comparison with the full capture; the mismatches in total after it.
std::uint64_t CountVerified(Kind kind, bool mismatch);
// A mismatch without a comparison (the full capture failed); the mismatches in total after it.
std::uint64_t CountMismatch(Kind kind);
void CountFullCapture(Kind kind, double microseconds);
void Report(Kind kind);
ShapeReuseCounts Counts(Kind kind);

}

}

#endif
