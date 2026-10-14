#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/ShapeReuse.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PerformanceControls.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <numeric>

namespace AgcDriver::DriverDetail {

// Draw shape reuse (live `draw_shape_reuse`, env APS5_DRAW_SHAPE_REUSE=1; off by default): the
// dispatch mechanism (ShapeReuse.cpp) for each stage of a draw. A stage whose user words differ
// from a remembered capture of the same program, stage and layout only where its plan takes them
// as read base pointers, copies them or never reads them, reuses that capture: each remembered
// read moves by its pointer word's difference and is read again through the draw's own
// ShaderMemory (the reader the walk uses: the same registered regions, pages, pending-write policy
// and evidence, and the same marks in the stage's regions); a read must return the remembered word
// unless it only copies its word into a refresh target (a pure flat-slot leaf, or with live
// `draw_shape_refresh_descriptors`, on by default, a descriptor leaf). The snapshot is the
// remembered one with the new user data, copies and refreshed words; Materialize runs on the
// stage's own request, so its vertex inputs (decoded from the current V# table as before), pixel
// inputs, layout and linked programs are the current ones. Anything not proved falls back to the
// full capture with the stage's marks undone; the attempt reads in the remembered walk's order and
// stops at the first word that may not differ, so it reads no address the full capture would not.
// APS5_VERIFY_DRAW_SHAPE_REUSE=1 keeps the production path (reuse used, or refused then captured)
// and captures every looked-up stage again in full on a shadow ShaderMemory of the draw, comparing
// the capture, the result, the stage regions and the memory's regions; differences are counted and
// printed (the production result is still the one used).

namespace {

constexpr auto Kind = ShapeCore::Kind::Draw;

}

bool Driver::drawShapeReuse() {
    static const bool fallback = std::getenv("APS5_DRAW_SHAPE_REUSE") != nullptr;
    return Graphics::LiveFlag("draw_shape_reuse", fallback);
}

bool Driver::verifyDrawShapeReuse() {
    static const bool verify = std::getenv("APS5_VERIFY_DRAW_SHAPE_REUSE") != nullptr;
    return verify;
}

bool Driver::drawShapeRefreshDescriptors() {
    static const bool fallback = std::getenv("APS5_DRAW_SHAPE_KEEP_DESCRIPTORS") == nullptr;
    return Graphics::LiveFlag("draw_shape_refresh_descriptors", fallback);
}

std::uint64_t Driver::drawShapeStaticKey(const ShaderRecompiler::RecompileRequest& request, const DrawProgram& program) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(reinterpret_cast<std::uintptr_t>(program.snapshot.get()));
    mix(program.codeOffset);
    mix(request.shader.codeAddress);
    mix(static_cast<std::uint64_t>(request.shader.stage));
    mix(request.context.waveSize);
    mix(request.context.userDataBaseRegister);
    mix(request.layout.pushConstantOffsetBytes);
    mix(request.layout.pushConstantSizeBytes);
    mix(request.context.userData.size());
    return key;
}

bool Driver::lookupDrawShape(const DrawProgram& program, std::uint64_t staticKey, const ShaderRecompiler::PreparedShaderInvocation& invocation, ShaderMemory& shaderMemory, DrawShapeReuse& reuse) {
    const auto started = std::chrono::steady_clock::now();
    const auto& request = invocation.Request();
    const auto address = request.shader.codeAddress;
    const std::span<const std::uint32_t> userData = request.context.userData;
    ShapeCore::CountLookup(Kind, address);
    std::vector<std::shared_ptr<const ShapeCandidate>> candidates;
    if (drawStageShapes.Find(Kind, address, staticKey, userData, candidates) != ShapeRefusal::Count) {
        ShapeCore::Report(Kind);
        return false;
    }
    const bool descriptors = drawShapeRefreshDescriptors();
    auto firstRefusal = ShapeRefusal::Count;
    const auto refuse = [&](ShapeRefusal reason) {
        if (firstRefusal == ShapeRefusal::Count) firstRefusal = reason;
        return false;
    };
    const auto attempt = [&](const ShapeCandidate& candidate) -> bool {
        if (candidate.staticKey != staticKey || candidate.shader.lock() != program.snapshot || candidate.shaderBase != address || candidate.userData.size() != userData.size() || !candidate.plan->known) return refuse(ShapeRefusal::KeyCollision);
        // The prepared source the walk would use, so the remembered plan is this stage's plan.
        if (!request.shader.capturedCalls.empty() || request.shader.code.data() != candidate.sourceCode || request.shader.code.size() != candidate.sourceWords) return refuse(ShapeRefusal::Source);
        std::vector<std::uint8_t> changed;
        std::vector<std::uint64_t> addresses;
        if (const auto reason = ShapeCore::Relocate(candidate, userData, {}, changed, addresses); reason != ShapeRefusal::Count) return refuse(reason);
        // Reads sharing an address read one word: the remembered walk read the same word there.
        std::vector<std::size_t> order(candidate.reads.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return addresses[a] < addresses[b]; });
        for (std::size_t k = 1; k < order.size(); ++k) {
            if (addresses[order[k]] == addresses[order[k - 1]] && candidate.reads[order[k]].value != candidate.reads[order[k - 1]].value) return refuse(ShapeRefusal::Conflict);
        }
        // Every moved read through the stage's memory, as the walk reads it, in the remembered
        // walk's order: attributeReads records a read after the reads its address depends on, so a
        // read whose base came from another read is made only once that read returned the
        // remembered word, and the first word that may not differ ends the attempt. Every address
        // read is one the full capture reads too (its pages, hook waits and evidence are the
        // walk's own); the marks stay only for a reuse that is used.
        struct Tentative {
            ShaderMemory& memory;
            bool keep = false;
            explicit Tentative(ShaderMemory& memory) : memory(memory) { memory.BeginTentative(); }
            ~Tentative() { memory.EndTentative(keep); }
            Tentative(const Tentative&) = delete;
            Tentative& operator=(const Tentative&) = delete;
        } tentative(shaderMemory);
        std::vector<std::pair<std::uint64_t, std::uint32_t>> refreshed;
        for (std::size_t r = 0; r < candidate.reads.size(); ++r) {
            const auto& read = candidate.reads[r];
            std::uint32_t word = 0;
            try {
                word = shaderMemory.ReadWord(addresses[r]);
            } catch (const std::exception&) {
                return refuse(ShapeRefusal::Inaccessible);
            }
            if (word == read.value) continue;
            // A word may differ only where every read of it copies it into a refresh target (such a
            // read has no users, so no later address depends on it).
            if (!ShapeCore::Refreshable(candidate, read, descriptors)) return refuse(ShapeRefusal::Differing);
            refreshed.emplace_back(addresses[r], word);
        }
        std::sort(refreshed.begin(), refreshed.end());
        refreshed.erase(std::unique(refreshed.begin(), refreshed.end()), refreshed.end());
        std::shared_ptr<ShaderRecompiler::ResourceCapture> rebuilt;
        bool descriptorRefreshed = false;
        if (const auto reason = ShapeCore::Rebuild(candidate, userData, changed, addresses, refreshed, descriptors, rebuilt, descriptorRefreshed); reason != ShapeRefusal::Count) return refuse(reason);
        std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
        try {
            compiled = invocation.Materialize(*rebuilt);
        } catch (const std::exception&) {
            return refuse(ShapeRefusal::Materialize);
        }
        tentative.keep = true;
        reuse.capture = std::move(rebuilt);
        reuse.compiled = std::move(compiled);
        reuse.refreshed = !refreshed.empty();
        reuse.descriptorRefreshed = descriptorRefreshed;
        return true;
    };
    for (const auto& candidate : candidates) {
        if (!attempt(*candidate)) continue;
        const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
        ShapeCore::CountHitTotals(Kind, elapsed, reuse.refreshed, reuse.descriptorRefreshed);
        ShapeCore::CountHitWindow(Kind, address, elapsed, reuse.refreshed, reuse.descriptorRefreshed);
        ShapeCore::Report(Kind);
        return true;
    }
    ShapeCore::CountRefusal(Kind, address, firstRefusal == ShapeRefusal::Count ? ShapeRefusal::Inconsistent : firstRefusal);
    ShapeCore::Report(Kind);
    return false;
}

void Driver::rememberDrawShape(const DrawProgram& program, std::uint64_t staticKey, std::span<const ShaderRecompiler::MemoryRegion> registered, std::span<const ShaderRecompiler::MemoryRegion> stageRegions, const ShaderRecompiler::PreparedShaderInvocation& invocation, const std::shared_ptr<const ShaderRecompiler::ResourceCapture>& capture) {
    if (!drawStageShapes.ShouldLearn(staticKey)) return;
    const auto& request = invocation.Request();
    // What the walk read from: the registered regions (served from the program snapshots) and the
    // stage's own page reads (a word an earlier stage read too included).
    const auto readRegions = [&] {
        std::vector<ShaderRecompiler::MemoryRegion> regions(registered.begin(), registered.end());
        regions.insert(regions.end(), stageRegions.begin(), stageRegions.end());
        return regions;
    };
    auto learned = ShapeCore::Learn(capture, !request.shader.capturedCalls.empty(), request.context.userData, request.shader.codeAddress, readRegions, registered);
    auto candidate = std::make_shared<ShapeCandidate>();
    candidate->shader = program.snapshot;
    candidate->shaderBase = request.shader.codeAddress;
    candidate->capture = capture;
    candidate->sourceCode = request.shader.code.data();
    candidate->sourceWords = request.shader.code.size();
    drawStageShapes.Remember(Kind, request.shader.codeAddress, staticKey, request.context.userData, std::move(learned), std::move(candidate));
}

void Driver::verifyDrawStage(std::uint64_t address, const ShaderRecompiler::PreparedShaderInvocation& invocation, const ShaderRecompiler::ResourceCapture& capture, const ShaderRecompiler::RecompileResult& compiled, std::span<const ShaderRecompiler::MemoryRegion> stageRegions, std::span<const ShaderRecompiler::MemoryRegion> memoryRegions, ShaderMemory& shadow) {
    // The stage again on the shadow memory, which only ever makes full captures: what this stage
    // produced (reused, refused then captured, or captured) and the memory it leaves must be the
    // full captures'.
    const char* what = nullptr;
    std::string failure;
    try {
        const auto full = shadow.Capture(invocation);
        const auto fullRegions = shadow.TakeRecentRegions();
        const auto fullResult = invocation.Materialize(*full);
        what = ShapeCore::Difference(compiled, *fullResult, capture, *full, stageRegions, fullRegions);
        if (what == nullptr) {
            const auto fullMemory = shadow.Regions();
            const auto same = [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress == b.guestAddress && a.bytes.size() == b.bytes.size() && std::memcmp(a.bytes.data(), b.bytes.data(), a.bytes.size()) == 0; };
            if (memoryRegions.size() != fullMemory.size() || !std::equal(memoryRegions.begin(), memoryRegions.end(), fullMemory.begin(), same)) what = "the stage memory's regions";
        }
    } catch (const std::exception& error) {
        failure = error.what();
        what = "the full capture (it failed)";
    }
    const auto count = ShapeCore::CountVerified(Kind, what != nullptr);
    if (what != nullptr && count <= 8) std::fprintf(stderr, "[draw-shape] APS5_VERIFY_DRAW_SHAPE_REUSE: program 0x%llx: %s differs from the full capture (mismatch %llu)%s%s\n", static_cast<unsigned long long>(address), what, static_cast<unsigned long long>(count), failure.empty() ? "" : ": ", failure.c_str());
}

}
