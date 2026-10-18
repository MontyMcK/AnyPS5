// The committer cuts that are pure functions of their inputs:
//   window_early_out  DrawPipeline::ScanWindow gives the same verdict with and without the early out
//   pending_fill      MarkUnitsPending sets the same bits with std::fill as one bit at a time
//   index_scan_memo   IndexScanMemoTable answers from an entry only while the range is unchanged
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Graphics/include/IndexScanMemo.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PendingUnits.hpp"
#include "prx/libSceAgcDriver/Graphics/include/SpecRefusalSites.hpp"
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawPipeline;
using AgcDriver::Graphics::IndexScanMemoTable;
using AgcDriver::Graphics::MarkUnitsPending;
using AgcDriver::Graphics::RefusalSites;
using AgcDriver::Graphics::SpeculationRefusalDetail;
using Range = DrawPipeline::Range;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Slot {
    std::uint64_t sequence = 0;
    bool unknown = false;
    std::vector<Range> writes;
};

const char* Name(DrawPipeline::Window window) {
    switch (window) {
        case DrawPipeline::Window::Clear: return "Clear";
        case DrawPipeline::Window::Overlap: return "Overlap";
        case DrawPipeline::Window::Unknown: return "Unknown";
        case DrawPipeline::Window::Overflow: return "Overflow";
    }
    return "?";
}

DrawPipeline::Window Scan(const std::vector<Slot>& slots, std::uint64_t from, std::uint64_t to, const std::vector<Range>& reads, bool earlyOut) {
    return DrawPipeline::ScanWindow(from, to, [&](std::uint64_t sequence) -> const Slot& { return slots[sequence % slots.size()]; }, reads, earlyOut);
}

// Hand-made windows for each verdict, including the orders the early out must keep: an unknown slot
// after an overlap (Unknown), an overlap after an unknown slot (Unknown), an overflow after either.
void CheckWindowCases() {
    std::vector<Slot> slots(8);
    for (std::uint64_t sequence = 1; sequence <= 8; ++sequence) slots[sequence % 8] = {sequence, false, {{sequence * 100, sequence * 100 + 10}}};
    const std::vector<Range> readNone{{5000, 5010}};
    const std::vector<Range> readThird{{305, 306}};
    for (const bool earlyOut : {false, true}) {
        const std::string mode = earlyOut ? " (early out)" : " (full scan)";
        Require(Scan(slots, 0, 9, readNone, earlyOut) == DrawPipeline::Window::Clear, "no overlap is not Clear" + mode);
        Require(Scan(slots, 0, 9, readThird, earlyOut) == DrawPipeline::Window::Overlap, "an overlap is not Overlap" + mode);
        Require(Scan(slots, 3, 4, readThird, earlyOut) == DrawPipeline::Window::Clear, "an empty window is not Clear" + mode);
        auto withUnknown = slots;
        withUnknown[6].unknown = true;
        Require(Scan(withUnknown, 0, 9, readThird, earlyOut) == DrawPipeline::Window::Unknown, "an unknown slot after an overlap is not Unknown" + mode);
        auto unknownFirst = slots;
        unknownFirst[2].unknown = true;
        Require(Scan(unknownFirst, 0, 9, readThird, earlyOut) == DrawPipeline::Window::Unknown, "an overlap after an unknown slot is not Unknown" + mode);
        auto overwritten = slots;
        overwritten[7].sequence = 15;
        Require(Scan(overwritten, 0, 9, readThird, earlyOut) == DrawPipeline::Window::Overflow, "an overwritten slot after an overlap is not Overflow" + mode);
        auto overwrittenAfterUnknown = unknownFirst;
        overwrittenAfterUnknown[7].sequence = 15;
        Require(Scan(overwrittenAfterUnknown, 0, 9, readNone, earlyOut) == DrawPipeline::Window::Overflow, "an overwritten slot after an unknown slot is not Overflow" + mode);
    }
}

// Random windows: the early out never changes the verdict.
void CheckWindowRandom() {
    std::mt19937_64 random(12345);
    for (int trial = 0; trial < 20000; ++trial) {
        const std::size_t capacity = 1 + random() % 32;
        std::vector<Slot> slots(capacity);
        const std::uint64_t base = random() % 1000;
        for (std::size_t i = 0; i < capacity; ++i) {
            auto& slot = slots[(base + i) % capacity];
            slot.sequence = base + i;
            if (random() % 50 == 0) slot.sequence += capacity;  // overwritten
            slot.unknown = random() % 12 == 0;
            const auto writes = random() % 4;
            for (std::uint64_t w = 0; w < writes; ++w) {
                const auto begin = random() % 4096;
                slot.writes.push_back({begin, begin + 1 + random() % 64});
            }
        }
        std::vector<Range> reads;
        const auto readCount = random() % 4;
        for (std::uint64_t r = 0; r < readCount; ++r) {
            const auto begin = random() % 4096;
            reads.push_back({begin, begin + 1 + random() % 128});
        }
        const std::uint64_t from = base == 0 ? 0 : base - 1 + random() % 2;
        const std::uint64_t to = from + 1 + random() % (capacity + 1);
        const auto full = Scan(slots, from, to, reads, false);
        const auto early = Scan(slots, from, to, reads, true);
        Require(full == early, std::string("the early out changed a verdict from ") + Name(full) + " to " + Name(early) + " (trial " + std::to_string(trial) + ")");
    }
}

// std::fill over the bit vector sets exactly the bits the loop sets, at every alignment.
void CheckPendingFill() {
    std::mt19937_64 random(777);
    for (int trial = 0; trial < 20000; ++trial) {
        const std::uint32_t size = 1 + static_cast<std::uint32_t>(random() % 700);
        std::vector<bool> start(size);
        for (std::uint32_t i = 0; i < size; ++i) start[i] = random() % 3 == 0;
        const std::uint32_t first = static_cast<std::uint32_t>(random() % (size + 1));
        const std::uint32_t count = static_cast<std::uint32_t>(random() % (size - first + 1));
        auto looped = start, filled = start;
        MarkUnitsPending(looped, first, count, false);
        MarkUnitsPending(filled, first, count, true);
        Require(looped == filled, "std::fill set other bits than the loop (size " + std::to_string(size) + ", first " + std::to_string(first) + ", count " + std::to_string(count) + ")");
        for (std::uint32_t i = 0; i < size; ++i) Require(filled[i] == (start[i] || (i >= first && i < first + count)), "a bit outside the range changed, or one inside was not set");
    }
}

// The memo against a fake write tracker: an entry answers only while the range is unchanged since
// its generation, an untracked range (generation 0) is never remembered, and the key includes the
// byte count and the index size.
void CheckIndexMemo() {
    IndexScanMemoTable memo;
    std::uint64_t generation = 10;          // the tracker's current generation
    std::uint64_t changedAt = 0;            // the generation of the range's last write
    std::uint32_t contents = 7;             // the highest index in the range's bytes
    int scans = 0;
    auto lookup = [&](std::uint64_t address, std::uint64_t bytes, std::uint32_t size, bool* hit) {
        return memo.Highest(address, bytes, size, [&] { return generation; }, [&](std::uint64_t since) { return changedAt <= since; }, [&] { ++scans; return contents; }, hit);
    };
    bool hit = true;
    Require(lookup(0x1000, 96, 2, &hit) == 7 && !hit && scans == 1, "a first lookup did not scan");
    Require(lookup(0x1000, 96, 2, &hit) == 7 && hit && scans == 1, "an unchanged range was scanned again");
    ++generation;
    Require(lookup(0x1000, 96, 2, &hit) == 7 && hit && scans == 1, "a newer generation without a write of the range refused the entry");
    changedAt = ++generation;
    contents = 9;
    Require(lookup(0x1000, 96, 2, &hit) == 9 && !hit && scans == 2, "a written range answered from its old entry");
    Require(lookup(0x1000, 96, 2, &hit) == 9 && hit && scans == 2, "the rescan was not remembered");
    Require(lookup(0x1000, 96, 4, &hit) == 9 && !hit && scans == 3, "another index size took the entry");
    Require(lookup(0x1000, 64, 2, &hit) == 9 && !hit && scans == 4, "another byte count took the entry");
    Require(lookup(0x2000, 96, 2, &hit) == 9 && !hit && scans == 5, "another address took the entry");
    generation = 0;
    Require(lookup(0x3000, 96, 2, &hit) == 9 && !hit && scans == 6, "an untracked range was not scanned");
    Require(lookup(0x3000, 96, 2, &hit) == 9 && !hit && scans == 7, "an untracked range was remembered");
    // A write while the range is scanned: the generation was collected before the scan, so the
    // entry it leaves is refused on the next lookup.
    generation = 50;
    changedAt = 0;
    contents = 3;
    Require(memo.Highest(0x4000, 96, 2, [&] { return generation; }, [&](std::uint64_t since) { return changedAt <= since; }, [&] { ++scans; changedAt = ++generation; return contents; }, &hit) == 3 && !hit, "a first lookup did not scan");
    contents = 4;
    Require(lookup(0x4000, 96, 2, &hit) == 4 && !hit, "a write during the scan did not refuse the entry it left");
    // Capacity: the table empties instead of growing without bound.
    generation = 100;
    for (std::uint64_t i = 0; i < IndexScanMemoTable::Capacity + 10; ++i) lookup(0x100000 + i * 64, 64, 2, nullptr);
    Require(memo.Size() <= IndexScanMemoTable::Capacity, "the memo grew past its capacity");
}

// The [spec-refusals] table: classes by held time, sites by held time within a class, skipped
// classes left out, the pending detail printed for pending sites, and the table cleared.
void CheckRefusalSites() {
    RefusalSites sites;
    RefusalSites::Key pending;
    pending.cls = 12;
    pending.vertex = 0x1000;
    pending.pixel = 0x2000;
    pending.target = 0x80000000;
    pending.targetBytes = 0x400000;
    pending.detail = {SpeculationRefusalDetail::Storage, 0x90000000, 0x10000, 0x90000000, 0x20000, 71, 256, 128};
    for (int i = 0; i < 3; ++i) sites.Add(pending, 40'000);
    auto lighter = pending;
    lighter.pixel = 0x3000;
    lighter.detail = {};
    sites.Add(lighter, 10'000);
    RefusalSites::Key refused;
    refused.cls = 2;
    refused.sub = 9;
    sites.Add(refused, 500'000);
    RefusalSites::Key accepted;
    accepted.cls = 1;
    sites.Add(accepted, 9'000'000);
    Require(sites.Size() == 4, "the table did not keep one entry per site");
    const auto lines = sites.Report([](std::uint8_t cls, std::uint8_t sub) { return "class" + std::to_string(cls) + "/" + std::to_string(sub); }, {1});
    Require(sites.Size() == 0, "the report did not clear the table");
    Require(lines.size() == 5, "the report has " + std::to_string(lines.size()) + " lines, not 5");
    Require(lines[0].find("class2/9: 1 commits at 500.0 us") != std::string::npos, "the heaviest class is not first: " + lines[0]);
    Require(lines[2].find("class12/0: 4 commits at 32.5 us") != std::string::npos, "the pending class line is wrong: " + lines[2]);
    Require(lines[3].find("3 at 40.0 us: vs 0x1000 ps 0x2000 target 0x80000000+0x400000; pending over a storage image: surface 0x90000000+0x10000, blocker 0x90000000+0x20000 fmt 71 256x128") != std::string::npos, "the heaviest pending site is wrong: " + lines[3]);
    Require(lines[4].find("1 at 10.0 us: vs 0x1000 ps 0x3000") != std::string::npos && lines[4].find("pending") == std::string::npos, "a site without a pending detail printed one: " + lines[4]);
    for (const auto& line : lines) Require(line.rfind("[spec-refusals]", 0) == 0 && line.find("class1/") == std::string::npos, "a line is unprefixed or names a skipped class: " + line);
}

}

int main() {
    try {
        CheckWindowCases();
        CheckWindowRandom();
        CheckPendingFill();
        CheckIndexMemo();
        CheckRefusalSites();
        std::cout << "committer cut tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
