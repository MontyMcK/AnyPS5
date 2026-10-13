#include "BdaTests.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestHeap.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <string>

namespace {

using namespace AgcDriver::Graphics;
using Role = ShaderRecompiler::DescriptorRole;

template<typename TAction>
void reject(TAction action, const char* reason) {
    try { action(); }
    catch (const std::runtime_error& error) {
        Require(std::string(error.what()).find(reason) != std::string::npos, std::string("unexpected BDA test error: ") + error.what());
        return;
    }
    throw std::runtime_error(std::string("expected BDA rejection: ") + reason);
}

ShaderRecompiler::DescriptorBinding binding(Role role, std::uint32_t slot) {
    return {ShaderRecompiler::DescriptorKind::StorageBuffer, role, 0, slot, 1, {}, false};
}

void heapMirrorTests(const Context& context, const BdaTestAccess& access) {
    if (!GuestArena::GuestArenaAvailable_nid_postfix() || !GuestArena::GuestArenaWriteWatched_nid_postfix()) {
        std::cout << "guest arena unavailable or not write-watched: heap mirrors not tested\n";
        return;
    }
    constexpr std::size_t bytes = 2 * 65536;
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536);
#ifdef _WIN32
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#endif
    auto* guest = static_cast<std::uint8_t*>(block);
    std::memset(block, 0x11, bytes);
    const auto address = reinterpret_cast<std::uintptr_t>(block);
    const auto registry = [&](bool add, bool writable) {
        auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
        if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, block, bytes, true, writable);
        else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, block);
        GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
    };
    const auto build = [&](std::uint64_t written, const std::function<void(GuestBufferMemory&)>& gpu) {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        if (written != 0) leased.AddWritable(written, 32);
        leased.Upload(true);
        const auto ranges = leased.AddressRanges();
        const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == address; });
        Require(found != ranges.end() && found->deviceAddress != 0u, "the heap range is missing from the BDA table");
        const auto device = found->deviceAddress;
        auto cursor = address;
        std::uint64_t writableBytes = 0;
        for (auto part = found; cursor < address + bytes && part != ranges.end(); ++part) {
            Require(part->begin == cursor && part->end > cursor && part->end <= address + bytes, "heap BDA ranges have gaps or overlap");
            Require(part->deviceAddress == device + cursor - address, "heap BDA ranges are not contiguous on the device");
            const bool writable = written != 0u && part->begin >= written && part->end <= written + 32u;
            const auto permissions = ShaderRecompiler::BdaAbi::Read | (writable ? ShaderRecompiler::BdaAbi::Write : 0u);
            Require(part->permissions == permissions, "heap BDA range has incorrect permissions");
            if (writable) writableBytes += part->end - part->begin;
            cursor = part->end;
        }
        Require(cursor == address + bytes && writableBytes == (written != 0u ? 32u : 0u), "heap BDA coverage or writable extent is incorrect");
        if (gpu) gpu(leased);
        leased.WriteBack();
        return device;
    };
    const auto sweep = [&] {
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.Upload(true);
        leased.WriteBack();
    };
    sweep();
    registry(true, false);
    const auto before = MirrorCounters();
    const auto first = build(0, {});
    const auto made = MirrorCounters();
    Require(made.heapMirrors == before.heapMirrors + 1 && made.heapBytes == before.heapBytes + bytes, "a read-only heap range was not mirrored");
    Require(access.addressBytes(first)[65536 + 3] == std::byte{0x11}, "the heap mirror was not filled");
    Require(build(0, {}) == first && MirrorCounters().blocksCopied == made.blocksCopied && MirrorCounters().heapRefills == made.heapRefills, "an unchanged heap range was read into its mirror again");
    guest[65536 + 3] = 0x22;
    Require(build(0, {}) == first && MirrorCounters().heapRefills == made.heapRefills + 1 && MirrorCounters().blocksCopied == made.blocksCopied + 1, "a written heap block was not read again alone");
    Require(access.addressBytes(first)[65536 + 3] == std::byte{0x22}, "the heap mirror missed the CPU write");
    const auto one = MirrorCounters();
    guest[5] = 0x33;
    guest[65536 + 7] = 0x44;
    Require(build(0, {}) == first && MirrorCounters().blocksCopied == one.blocksCopied + 2, "two written heap blocks were not read again");
    Require(access.addressBytes(first)[5] == std::byte{0x33} && access.addressBytes(first)[65536 + 7] == std::byte{0x44}, "the heap mirror missed the CPU writes");
    registry(false, false);
    sweep();
    const auto swept = MirrorCounters();
    Require(swept.heapMirrors == before.heapMirrors && swept.heapBytes == before.heapBytes, "the heap mirror outlived its range");

    registry(true, true);
    const auto device = build(address + 16, [&](GuestBufferMemory& leased) {
        std::uint32_t adjustment = 0;
        const auto view = leased.Descriptor(address + 16, 32, adjustment);
        access.bytes(view.buffer)[view.offset + adjustment] = std::byte{0x77};
        guest[16 + 8] = 0x55;
    });
    Require(guest[16] == 0x77 && guest[16 + 8] == 0x55, "a writable heap mirror's write-back lost the GPU's store or rolled back the CPU's");
    Require(build(0, {}) == device && access.addressBytes(device)[16] == std::byte{0x77} && access.addressBytes(device)[16 + 8] == std::byte{0x55}, "the writable heap mirror missed the stores");
    registry(false, true);
    sweep();
    Require(MirrorCounters().heapMirrors == before.heapMirrors, "the writable heap mirror outlived its range");

#ifdef _WIN32
    {
        constexpr std::size_t half = 2 * 65536;
        void* pair = GuestArena::GuestArenaAllocate_nid_postfix(2 * half, 65536);
        GuestArena::GuestArenaCommit_nid_postfix(pair, 2 * half, PAGE_READWRITE, 2 * half);
        std::memset(pair, 0x11, 2 * half);
        auto* first = static_cast<std::uint8_t*>(pair);
        auto* second = first + half;
        const auto registerPair = [&](bool add) {
            auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
            for (auto* range : {first, second}) {
                if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, range, half, true, false);
                else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, range);
            }
            GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        };
        const auto firstDevice = [&] {
            GuestBufferMemory leased(context);
            leased.AcquireRegistered();
            leased.Upload(true);
            const auto ranges = leased.AddressRanges();
            const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == reinterpret_cast<std::uintptr_t>(first); });
            Require(found != ranges.end(), "the first heap range is missing from the BDA table");
            const auto device = found->deviceAddress;
            leased.WriteBack();
            return device;
        };
        registerPair(true);
        const auto device = firstDevice();
        first[5] = 0x66;
        DWORD previous = 0;
        Require(VirtualProtect(second, 65536, PAGE_NOACCESS, &previous) != 0, "cannot protect the second heap range");
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(reinterpret_cast<std::uintptr_t>(second), 65536);
        bool threw = false;
        try {
            GuestBufferMemory leased(context);
            leased.AcquireRegistered();
        } catch (const std::runtime_error&) {
            threw = true;
        }
        Require(VirtualProtect(second, 65536, PAGE_READWRITE, &previous) != 0, "cannot unprotect the second heap range");
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(reinterpret_cast<std::uintptr_t>(second), 65536);
        Require(threw, "a build over an inaccessible heap mirror range did not fail");
        Require(firstDevice() == device && access.addressBytes(device)[5] == std::byte{0x66}, "an interrupted build left a changed heap block marked current");
        registerPair(false);
        sweep();
        GuestArena::GuestArenaReset_nid_postfix(pair, 2 * half);
        GuestArena::GuestArenaRelease_nid_postfix(pair, 2 * half);
    }
#endif
#ifdef _WIN32
    {
        constexpr std::size_t size = 2 * 65536;
        void* raw = GuestArena::GuestArenaAllocate_nid_postfix(size, 65536);
        GuestArena::GuestArenaCommit_nid_postfix(raw, size, PAGE_READWRITE, size);
        auto* bytes8 = static_cast<std::uint8_t*>(raw);
        std::memset(raw, 0x11, size);
        const auto base = reinterpret_cast<std::uintptr_t>(raw);
        const auto page = base + 65536;
        DWORD previous = 0;
        Require(VirtualProtect(reinterpret_cast<void*>(page), 4096, PAGE_READONLY, &previous) != 0, "cannot make the aliased page read-only");
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(page, 4096);
        const auto registerRange = [&](bool add) {
            auto* mutation = GuestAllocations::GuestAllocationsBegin_nid_postfix();
            if (add) GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, raw, size, true, true);
            else GuestAllocations::GuestAllocationsRemove_nid_postfix(mutation, raw);
            GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation);
        };
        registerRange(true);
        const auto storeAt = [&](std::uint64_t at) {
            GuestBufferMemory leased(context);
            leased.AcquireRegistered();
            leased.AddWritable(page - 16, 32);
            leased.Upload(true);
            const auto ranges = leased.AddressRanges();
            const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == base; });
            Require(found != ranges.end() && access.addressBytes(found->deviceAddress)[65536 + 8] == std::byte{0x11}, "a writable range with a read-only page was not mirrored with its bytes");
            std::uint32_t adjustment = 0;
            const auto view = leased.Descriptor(page - 16, 32, adjustment);
            access.bytes(view.buffer)[view.offset + adjustment + static_cast<std::size_t>(at - (page - 16))] = std::byte{0x77};
            leased.WriteBack();
        };
        const auto made = MirrorCounters().heapMirrors;
        storeAt(page - 8);
        Require(MirrorCounters().heapMirrors == made + 1 && bytes8[65536 - 8] == 0x77, "a store next to a read-only page was not written back");
        bool refused = false;
        try {
            storeAt(page + 4);
        } catch (const std::runtime_error& error) {
            refused = std::string(error.what()).find("aliased writes are not implemented") != std::string::npos;
        }
        Require(refused && bytes8[65536 + 4] == 0x11, "a GPU change of a read-only page was not refused");
        {
            GuestBufferMemory plain(context);
            plain.AddReadable(page, 16);
            plain.Upload(false);
            std::uint32_t adjustment = 0;
            const auto view = plain.Descriptor(page, 16, adjustment);
            Require(access.bytes(view.buffer)[view.offset + adjustment] == std::byte{0x11}, "a descriptor over a read-only page of a writable range read zeros");
            plain.WriteBack();
        }
        registerRange(false);
        sweep();
        Require(VirtualProtect(reinterpret_cast<void*>(page), 4096, PAGE_READWRITE, &previous) != 0, "cannot restore the aliased page");
        GuestAllocations::GuestAllocationsInvalidate_nid_postfix(page, 4096);
        GuestArena::GuestArenaReset_nid_postfix(raw, size);
        GuestArena::GuestArenaRelease_nid_postfix(raw, size);
    }
#endif
#ifdef _WIN32
    GuestArena::GuestArenaReset_nid_postfix(block, bytes);
#endif
    GuestArena::GuestArenaRelease_nid_postfix(block, bytes);
}

void addressTemplateTests(const Context& context, const BdaTestAccess& access) {
    void* block = GuestHeap::GuestHeapAllocate_nid_postfix(64);
    const auto blockAddress = reinterpret_cast<std::uintptr_t>(block);
    const auto registered = [&] {
        const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
        return std::any_of(lease.begin(), lease.end(), [&](const auto& range) { return range->address == blockAddress; });
    };
    const auto release = [&] {
        GuestHeap::GuestHeapFree_nid_postfix(block);
        GuestBufferMemory leased(context);
        leased.AcquireRegistered();
        leased.Upload(true);
        leased.WriteBack();
    };
    ShaderRecompiler::RecompileResult vertex;
    vertex.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    vertex.shaderDataDwords = 4;
    const auto vsharp = [](std::uint64_t at, std::uint32_t bytes) { return std::vector<std::uint32_t>{static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(at >> 32) & 0xffffu, bytes, 0x31000000u}; };
    vertex.bindings = {binding(Role::BdaPagetable, 4), binding(Role::FaultBuffer, 5), binding(Role::ShaderData, 7), binding(Role::GuestBuffers, 6)};
    vertex.bindings[2].guestDescriptor = {1, 2, 3, 4};
    vertex.bindings[3].guestDescriptor = vsharp(blockAddress, 32);
    vertex.bindings[3].bufferWritten = {false};
    ShaderRecompiler::RecompileResult fragment;
    const std::array<CompiledShader, 2> stages{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}}};
    auto first = std::make_shared<ShaderResources>(context, stages, ColorTarget{}, 0, 0);
    Require(first->HoldsLease() && first->NeedsCompletion() && first->WritesMemory(), "an address-based draw build holds no lease");
    if (!ShaderResources::AddressTemplates()) {
        Require(!first->AddressReusable(), "an address-based build became an address template with APS5_ADDRESS_TEMPLATES unset");
        first->WriteBackBuffers();
        Require(!first->HoldsLease(), "write-back kept the lease");
        first.reset();
        release();
        std::cout << "APS5_ADDRESS_TEMPLATES unset: address template reuse not tested\n";
        return;
    }
    if (!first->AddressReusable()) {
        first->WriteBackBuffers();
        first.reset();
        release();
        std::cout << "registered memory outside the address space's imports and mirrors: address template reuse not tested\n";
        return;
    }
    const auto table = access.descriptor(4).buffer;
    const auto templateFault = access.descriptor(5).buffer;
    const auto templateData = access.descriptor(7).buffer;
    const auto templateView = access.descriptor(6);
    auto own = ShaderResources::DetachAddressUse(first);
    Require(!first->HoldsLease() && own->HoldsLease() && own->NeedsCompletion() && own->WritesMemory() && !own->HasCopiedWrites(), "the detached use does not answer as the build it came from");
    Require(access.descriptor(4).buffer == table && access.descriptor(5).buffer != templateFault && access.descriptor(7).buffer == templateData, "the detached use did not get a fault buffer of its own");
    own->WriteBackBuffers();
    Require(!own->HoldsLease(), "the detached use's write-back kept the lease");
    own.reset();
    auto miss = ShaderResources::AddressMiss::Count;
    auto use = ShaderResources::AddressUse(first, stages, {}, miss);
    Require(use != nullptr && miss == ShaderResources::AddressMiss::None, "an unchanged address-based draw was not served by its template");
    Require(use->HoldsLease() && use->NeedsCompletion() && use->WritesMemory() && !first->HoldsLease() && !use->AddressReusable(), "a template use does not hold the lease as a build would");
    const auto useFault = access.descriptor(5).buffer;
    Require(access.descriptor(4).buffer == table && useFault != templateFault && access.descriptor(7).buffer == templateData, "a template use shares the template's fault buffer or lost its table");
    vertex.bindings[2].guestDescriptor = {5, 6, 7, 8};
    auto second = ShaderResources::AddressUse(first, stages, {}, miss);
    Require(second != nullptr && miss == ShaderResources::AddressMiss::None, "an address-based draw with other constants was not served by its template");
    const auto secondData = access.descriptor(7);
    Require(access.descriptor(5).buffer != useFault && secondData.buffer != templateData, "two template uses in flight share a fault or data buffer");
    std::array<std::uint32_t, 4> words{};
    std::memcpy(words.data(), access.bytes(secondData.buffer).data() + secondData.offset, sizeof(words));
    Require(words == std::array<std::uint32_t, 4>{5, 6, 7, 8}, "a template use does not bind its own constants");
    vertex.bindings[3].guestDescriptor = vsharp(blockAddress + 32, 32);
    auto moved = ShaderResources::AddressUse(first, stages, {}, miss);
    Require(moved != nullptr && miss == ShaderResources::AddressMiss::None, "a draw whose read-only buffer moved inside the address space was not served by its template");
    const auto movedView = access.descriptor(6);
    Require(movedView.buffer == templateView.buffer && movedView.offset == templateView.offset + 32 && movedView.range == templateView.range && moved->HoldsLease(), "a template use did not bind its moved read-only buffer at the new address");
    moved->WriteBackBuffers();
    moved.reset();
    alignas(16) std::array<std::uint32_t, 8> unregistered{};
    vertex.bindings[3].guestDescriptor = vsharp(reinterpret_cast<std::uintptr_t>(unregistered.data()), 32);
    Require(ShaderResources::AddressUse(first, stages, {}, miss) == nullptr && miss == ShaderResources::AddressMiss::MovedOutside, "a read-only buffer moved outside the address space was bound from the template");
    vertex.bindings[3].guestDescriptor = vsharp(blockAddress, 32);
    const auto watched = GuestArena::GuestArenaAvailable_nid_postfix() && GuestArena::GuestArenaWriteWatched_nid_postfix();
    void* pages = watched ? GuestArena::GuestArenaAllocate_nid_postfix(65536, 65536) : nullptr;
    if (pages != nullptr) {
#ifdef _WIN32
        GuestArena::GuestArenaCommit_nid_postfix(pages, 65536, PAGE_READWRITE, 65536);
#endif
        const auto page = reinterpret_cast<std::uintptr_t>(pages);
        const auto generation = AgcDriver::GuestMemory::CollectWrites(page, 4);
        Require(AgcDriver::GuestMemory::UnchangedSince(page, 4, generation), "an untouched page is written");
        const auto slot = static_cast<std::uint32_t>(page >> ShaderRecompiler::BdaAbi::WrittenPageShift) + 1u;
        auto fault = access.bytes(useFault);
        std::memcpy(fault.data() + ShaderRecompiler::BdaAbi::WrittenSlotsWord * sizeof(std::uint32_t), &slot, sizeof(slot));
        use->WriteBackBuffers();
        Require(!AgcDriver::GuestMemory::UnchangedSince(page, 4, generation), "a template use's written page was not marked");
        std::uint32_t cleared = 1;
        std::memcpy(&cleared, fault.data() + ShaderRecompiler::BdaAbi::WrittenSlotsWord * sizeof(std::uint32_t), sizeof(cleared));
        Require(cleared == 0, "a template use's written page slot was not cleared");
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(pages, 65536);
#endif
        GuestArena::GuestArenaRelease_nid_postfix(pages, 65536);
    } else {
        std::cout << "guest arena unavailable or not write-watched: template use written pages not tested\n";
        use->WriteBackBuffers();
    }
    Require(!use->HoldsLease() && second->HoldsLease(), "one template use's write-back released another's lease");
    second->WriteBackBuffers();
    Require(!second->HoldsLease(), "a template use's write-back kept the lease");
    use.reset();
    second.reset();
    Require(registered(), "guest heap block is not registered");
    release();
    Require(!registered(), "a guest free waited on an idle address template");
    Require(ShaderResources::AddressUse(first, stages, {}, miss) == nullptr && miss == ShaderResources::AddressMiss::Replaced, "a template outlived its address space");
    const auto counts = ShaderResources::AddressTemplateCounters();
    Require(counts.hits >= 3 && counts.detached >= 1 && counts.movedBuffers >= 1 && counts.misses[static_cast<std::size_t>(ShaderResources::AddressMiss::Replaced)] >= 1 && counts.misses[static_cast<std::size_t>(ShaderResources::AddressMiss::MovedOutside)] >= 1, "address template counters missed the uses");
}

// The lookup index over a low cluster (the exe image), a heap cluster and an image mirror, as the
// race's tables are laid out: the dense window covers the heap alone, the whole-span shape is the
// one every table had before, and a table built here carries the shape its builder chose with
// every range inside the [low, high) of each bucket it intersects, which is what the lookup's
// narrowing relies on (a lookup outside the window searches every range; BdaExecution pins that).
void bucketShapeTests(const Context& context, const BdaTestAccess& access) {
    using Range = ShaderRecompiler::BdaAbi::Range;
    constexpr std::uint64_t heap = 0x10'0000'0000ull;
    constexpr std::uint64_t mirror = 0x7ff6'b000'0000ull;
    const auto make = [](std::size_t low, std::size_t heaped, std::size_t mirrors) {
        std::vector<Range> ranges;
        const auto add = [&](std::uint64_t begin, std::uint64_t bytes) { ranges.push_back({begin, begin + bytes, 0x100000000000ull + ranges.size() * 0x10000u, ShaderRecompiler::BdaAbi::Read, 0}); };
        for (std::size_t i = 0; i < low; ++i) add(0x400000u + i * 0x20000u, 0x10000);
        for (std::size_t i = 0; i < heaped; ++i) add(heap + i * 0x100000u, 0x8000);
        for (std::size_t i = 0; i < mirrors; ++i) add(mirror + i * 0x1000000u, 0x400000);
        return ranges;
    };
    // Written out apart from the builder: the smallest shift from 12 whose buckets over
    // [first & ~mask, last) number at most `limit`.
    const auto fitted = [](std::uint64_t first, std::uint64_t last, std::size_t limit) {
        for (std::uint32_t shift = 12; shift <= 48; ++shift) {
            const auto mask = (std::uint64_t{1} << shift) - 1u;
            const auto count = (last - (first & ~mask) + mask) >> shift;
            if (count <= limit) return BdaBucketWindow{first & ~mask, shift, static_cast<std::size_t>(count), false};
        }
        return BdaBucketWindow{};
    };
    const auto same = [](const BdaBucketWindow& left, const BdaBucketWindow& right) { return left.base == right.base && left.shift == right.shift && left.count == right.count && left.windowed == right.windowed; };
    const auto ranges = make(6, 20, 1);
    const auto whole = BdaBucketShape(ranges, 4096, false);
    Require(same(whole, fitted(ranges.front().begin, ranges.back().end, 4096)) && whole.shift == 35, "the whole-span bucket shape changed");
    const auto dense = BdaBucketShape(ranges, 4096, true);
    const auto heapEnd = ranges[25].end;
    Require(dense.windowed && same(dense, BdaBucketWindow{heap, 13, (heapEnd - heap + 0x1fffu) >> 13u, true}), "the dense window does not fit the heap cluster");
    Require(dense.base <= ranges[6].begin && dense.base + (static_cast<std::uint64_t>(dense.count) << dense.shift) >= heapEnd && dense.base >= ranges[5].end, "the dense window does not cover exactly the heap cluster");
    Require(same(BdaBucketShape(ranges, 1024, true), BdaBucketWindow{heap, 15, (heapEnd - heap + 0x7fffu) >> 15u, true}), "the dense window ignores the bucket limit");
    // The mirrors never take the window, even when they hold the most entries.
    const auto mirrored = make(0, 17, 20);
    const auto skipped = BdaBucketShape(mirrored, 4096, true);
    Require(skipped.windowed && skipped.base == heap && skipped.base + (static_cast<std::uint64_t>(skipped.count) << skipped.shift) < mirror, "the dense window covered the image mirrors");
    // Equal entries: the smaller span, here the later cluster (16 low ranges over ~15 MiB, 16 heap
    // ranges over ~1 MiB).
    std::vector<Range> tie;
    for (std::uint64_t i = 0; i < 16; ++i) tie.push_back({0x400000u + i * 0x100000u, 0x400000u + i * 0x100000u + 0x1000u, 0x100000000000ull, ShaderRecompiler::BdaAbi::Read, 0});
    for (std::uint64_t i = 0; i < 16; ++i) tie.push_back({heap + i * 0x10000u, heap + i * 0x10000u + 0x1000u, 0x100000000000ull, ShaderRecompiler::BdaAbi::Read, 0});
    const auto tied = BdaBucketShape(tie, 4096, true);
    Require(tied.windowed && tied.base == heap && tied.shift == 12 && tied.count == 0xf1u, "a tie between clusters did not take the smaller span");
    // No cluster of 16: the whole-span shape, not a window over a handful of entries.
    const auto scattered = make(8, 8, 1);
    Require(same(BdaBucketShape(scattered, 4096, true), BdaBucketShape(scattered, 4096, false)) && !BdaBucketShape(scattered, 4096, true).windowed, "a table with no 16-entry cluster got a dense window");
    Require(BdaBucketShape(make(4, 11, 0), 4096, true).count == 0 && BdaBucketShape(make(4, 11, 0), 4096, false).count == 0 && BdaBucketShape(ranges, 0, true).count == 0, "a small table or APS5_BDA_BUCKETS=0 got an index");

    // A table built from snapshots at those addresses (snapshots map apart on the device, so no
    // two join) carries the shape for the flag as the builder read it.
    const std::array<std::byte, 8> bytes{};
    std::vector<GuestMemorySnapshot> snapshots;
    for (const auto& range : ranges) snapshots.push_back({range.begin, bytes});
    ShaderRecompiler::RecompileResult shader;
    shader.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    shader.bindings = {binding(Role::BdaPagetable, 4), binding(Role::FaultBuffer, 5)};
    const CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &shader, 0};
    const bool denseOn = BdaDenseBuckets();
    {
        ShaderResources resources(context, compiled, snapshots);
        const auto table = access.bytes(access.descriptor(4).buffer);
        ShaderRecompiler::BdaAbi::Header header{};
        std::memcpy(&header, table.data(), sizeof(header));
        Require(header.version == ShaderRecompiler::BdaAbi::Version && header.count >= ranges.size(), "the snapshot table lost or joined ranges");
        std::vector<Range> built(header.count);
        std::memcpy(built.data(), table.data() + sizeof(header), built.size() * sizeof(Range));
        for (const auto& range : ranges) Require(std::any_of(built.begin(), built.end(), [&](const Range& entry) { return entry.begin == range.begin && entry.end == range.begin + bytes.size(); }), "a snapshot range is missing from the table");
        const auto expected = BdaBucketShape(built, BdaBucketLimit(), denseOn);
        Require(header.buckets == expected.count && expected.count != 0 && expected.windowed == denseOn, "the table's index does not have the builder's shape");
        ShaderRecompiler::BdaAbi::BucketHeader index{};
        const auto at = sizeof(header) + built.size() * sizeof(Range);
        Require(table.size() >= at + sizeof(index) + header.buckets * 2u * sizeof(std::uint32_t), "the table is shorter than its index");
        std::memcpy(&index, table.data() + at, sizeof(index));
        Require(index.base == expected.base && index.shift == expected.shift && index.reserved == 0, "the table's index window differs from its shape");
        std::vector<std::uint32_t> bounds(header.buckets * 2u);
        std::memcpy(bounds.data(), table.data() + at + sizeof(index), bounds.size() * sizeof(std::uint32_t));
        for (std::uint32_t bucket = 0; bucket < header.buckets; ++bucket) {
            const auto start = index.base + (static_cast<std::uint64_t>(bucket) << index.shift);
            const auto stop = start + (std::uint64_t{1} << index.shift);
            const auto low = bounds[bucket * 2u], high = bounds[bucket * 2u + 1u];
            Require(low <= high && high <= header.count, "a bucket's bounds are out of order");
            for (std::uint32_t entry = 0; entry < header.count; ++entry) {
                const bool intersects = built[entry].end > start && built[entry].begin < stop;
                Require(!intersects || (low <= entry && entry < high), "a range is missing from a bucket it intersects");
            }
        }
        resources.WriteBack();
    }
    std::cout << "BDA bucket index tested with bda_dense_buckets " << (denseOn ? "on" : "off") << '\n';
}

}

void RunBdaResourceTests(const Context& context, const BdaTestAccess& access) {
    alignas(64) std::array<std::uint32_t, 16> guest{};
    guest[0] = 123;
    const auto address = reinterpret_cast<std::uintptr_t>(guest.data());
    GuestBufferMemory memory(context);
    memory.AddWritable(address, sizeof(guest));
    memory.AddWritable(address + 16, 16);
    memory.Upload(true);
    std::uint32_t adjustment = 0;
    const auto first = memory.Descriptor(address, sizeof(guest), adjustment);
    Require(adjustment == 0, "a view at its owner's start is bound off it");
    const auto alias = memory.Descriptor(address + 16, 16, adjustment);
    Require(first.buffer == alias.buffer && alias.offset + adjustment == 16 && alias.range == 16 + adjustment, "aliased guest buffers have different owners");
    const auto ranges = memory.AddressRanges();
    Require(ranges.size() == 1 && ranges[0].begin == address && ranges[0].end == address + sizeof(guest), "incorrect BDA range bounds");
    Require(ranges[0].deviceAddress != 0 && ranges[0].permissions == (ShaderRecompiler::BdaAbi::Read | ShaderRecompiler::BdaAbi::Write), "incorrect BDA address or permissions");
    std::uint32_t changed = 321;
    std::memcpy(access.bytes(alias.buffer).data() + alias.offset, &changed, sizeof(changed));
    memory.WriteBack();
    Require(guest[4] == changed, "aliased GPU write was not published");
    reject([&] { memory.WriteBack(); }, "cannot be committed twice");
    reject([&] { memory.AddWritable(address, sizeof(guest)); }, "frozen");
    GuestBufferMemory overflow(context);
    const std::array<std::byte, 8> source{};
    reject([&] { overflow.AddSnapshot({std::numeric_limits<std::uint64_t>::max() - 3, source}); }, "overflow");

    const std::array<GuestMemorySnapshot, 1> snapshots{{{0x7fff12340000ULL, source}}};
    ShaderRecompiler::RecompileResult shader;
    shader.bindings = {binding(Role::BdaPagetable, 4), binding(Role::FaultBuffer, 5)};
    CompiledShader compiled{ShaderRecompiler::ShaderStage::Compute, &shader, 0};
    reject([&] { ShaderResources resources(context, compiled, snapshots); }, "ABI version");
    shader.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
    auto disabled = context;
    disabled.bufferDeviceAddress = false;
    reject([&] { ShaderResources resources(disabled, compiled, snapshots); }, "not enabled");
    {
        // A rect-list fault buffer must not require BDA or consume guest snapshots.
        ShaderRecompiler::RecompileResult control;
        control.bdaAbiVersion = ShaderRecompiler::BdaAbi::Version;
        control.bindings = {binding(Role::FaultBuffer, 5)};
        auto writable = binding(Role::GuestBuffers, 6);
        writable.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32) & 0xffffu, sizeof(guest), 0x31000000u};
        control.bindings.push_back(writable);
        const std::array<CompiledShader, 1> stages{{{ShaderRecompiler::ShaderStage::TessellationControl, &control, 0}}};
        const std::array<GuestMemorySnapshot, 1> unusedSnapshots{{{0, source}}};
        ShaderResources resources(disabled, stages, ColorTarget{}, 0, 0, unusedSnapshots);
        const auto fault = access.bytes(access.descriptor(5).buffer);
        for (const auto byte : fault) Require(byte == std::byte{}, "rect-list fault buffer was not initialized");
        const ShaderRecompiler::BdaAbi::Fault report{ShaderRecompiler::BdaAbi::FaultState::Ready, ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle, 0, 0, 0, 0, 0};
        std::memcpy(fault.data(), &report, sizeof(report));
        reject([&] { resources.WriteBack(); }, "rect-list requires");
        std::memset(fault.data(), 0, fault.size());
        changed = 456;
        std::memcpy(access.bytes(access.descriptor(6).buffer).data() + sizeof(std::uint32_t), &changed, sizeof(changed));
        resources.WriteBack();
        Require(guest[1] == changed, "rect-list fault-only path lost guest buffer writes");
    }
    {
        ShaderResources resources(context, compiled, snapshots);
        const auto table = access.bytes(access.descriptor(4).buffer);
        ShaderRecompiler::BdaAbi::Header header{};
        ShaderRecompiler::BdaAbi::Range range{};
        std::memcpy(&header, table.data(), sizeof(header));
        Require(header.version == ShaderRecompiler::BdaAbi::Version && header.count == 1 && header.entryBytes == sizeof(range), "BDA header layout mismatch");
        std::memcpy(&range, table.data() + sizeof(header), sizeof(range));
        Require(range.begin == snapshots[0].address && range.end == range.begin + source.size(), "64-bit guest address was truncated");
        const auto fault = access.bytes(access.descriptor(5).buffer);
        for (const auto byte : fault) Require(byte == std::byte{}, "fault buffer was not initialized");
        const ShaderRecompiler::BdaAbi::Fault denied{ShaderRecompiler::BdaAbi::FaultState::Ready, ShaderRecompiler::BdaAbi::FaultReason::Permission, snapshots[0].address + 4, 4, 0, 0x88, 0};
        std::memcpy(fault.data(), &denied, sizeof(denied));
        reject([&] { resources.WriteBack(); }, "read-only in the BDA table");
        std::memset(fault.data(), 0, fault.size());
        resources.WriteBack();
    }
    bucketShapeTests(context, access);
    {
        auto writable = binding(Role::GuestBuffers, 6);
        writable.guestDescriptor = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32) & 0xffffu, sizeof(guest), 0x31000000u};
        shader.bindings.push_back(writable);
        ShaderResources resources(context, compiled);
        changed = 999;
        std::memcpy(access.bytes(access.descriptor(6).buffer).data(), &changed, sizeof(changed));
        const ShaderRecompiler::BdaAbi::Fault report{ShaderRecompiler::BdaAbi::FaultState::Ready, ShaderRecompiler::BdaAbi::FaultReason::Unmapped, 0x7fff99880000ULL, 4, 0, 0x44, 0};
        std::memcpy(access.bytes(access.descriptor(5).buffer).data(), &report, sizeof(report));
        reject([&] { resources.WriteBack(); }, "BDA access failed");
        auto invalidRectangle = report;
        invalidRectangle.reason = ShaderRecompiler::BdaAbi::FaultReason::InvalidRectangle;
        std::memcpy(access.bytes(access.descriptor(5).buffer).data(), &invalidRectangle, sizeof(invalidRectangle));
        reject([&] { resources.WriteBack(); }, "rect-list requires");
        Require(guest[0] == 123, "failed GPU command published writes");
    }
    {
        auto aligned = context;
        aligned.limits.minStorageBufferOffsetAlignment = 16;
        GuestBufferMemory unaligned(aligned);
        unaligned.AddWritable(address, sizeof(guest));
        unaligned.Upload(true);
        std::uint32_t adjustment = 0;
        const auto view = unaligned.Descriptor(address + 4, 4, adjustment);
        Require(adjustment == 4 && view.offset == 0 && view.range == 8, "a view off the offset alignment binds from below it");
        reject([&] { unaligned.Descriptor(address + sizeof(guest), 4, adjustment); }, "exceeds its GPU owner");
        const auto odd = unaligned.Descriptor(address + 2, 6, adjustment);
        Require(adjustment == 2 && odd.offset == 0 && odd.range == 8, "a view off a DWORD boundary does not bind the DWORDs around it");
        const auto late = unaligned.Descriptor(address + 0x13, 8, adjustment);
        Require(adjustment == 3 && late.offset == 16 && late.range == 12, "a view off a DWORD boundary does not bind from the offset alignment below it");
        GuestBufferMemory lone(aligned);
        lone.AddReadable(address + 6, 5);
        lone.Upload(true);
        const auto copied = lone.Descriptor(address + 6, 5, adjustment);
        Require(adjustment == 2 && copied.offset == 0 && copied.range == 8, "a lone view off a DWORD boundary is not copied from the DWORD below it");
        Require(std::memcmp(access.bytes(copied.buffer).data(), reinterpret_cast<const void*>(address + 4), 8) == 0, "a lone view off a DWORD boundary copied other bytes");
        GuestBufferMemory tail(aligned);
        tail.AddWritable(address, 62);
        tail.Upload(true);
        reject([&] { tail.Descriptor(address + 58, 4, adjustment); }, "exceeds its GPU owner");
    }
    {
        // The cached address space: a second build in an unchanged registry takes the first one's
        // space, and a guest free of a range pinned only by the cache goes through the pin waiter's
        // drop (no GPU work to wait for) and empties the registry of it.
        void* block = GuestHeap::GuestHeapAllocate_nid_postfix(64);
        const auto blockAddress = reinterpret_cast<std::uintptr_t>(block);
        std::memset(block, 0x5a, 64);
        const auto registered = [&] {
            const auto lease = GuestAllocations::GuestAllocationsAcquire_nid_postfix();
            return std::any_of(lease.begin(), lease.end(), [&](const auto& range) { return range->address == blockAddress; });
        };
        Require(registered(), "guest heap block is not registered");
        const auto before = AddressSpaceCounters();
        for (int build = 0; build < 2; ++build) {
            GuestBufferMemory leased(context);
            leased.AcquireRegistered();
            Require(leased.HoldsLease(), "address-based build holds no lease");
            leased.Upload(true);
            std::uint32_t adjustment = 0;
            const auto view = leased.Descriptor(blockAddress, 64, adjustment);
            Require(view.range == 64 + adjustment, "leased block has no descriptor");
            const auto ranges = leased.AddressRanges();
            Require(std::any_of(ranges.begin(), ranges.end(), [&](const auto& range) { return range.begin == blockAddress && range.end == blockAddress + 64; }), "leased block is missing from the BDA table");
            leased.WriteBack();
            Require(!leased.HoldsLease(), "write-back kept the lease");
        }
        const auto after = AddressSpaceCounters();
        if (after.enabled) Require(after.hits == before.hits + 1 && after.rebuiltFirst + after.rebuiltGeneration + after.rebuiltWaiterDrop + after.rebuiltEpoch + after.rebuiltDevice == before.rebuiltFirst + before.rebuiltGeneration + before.rebuiltWaiterDrop + before.rebuiltEpoch + before.rebuiltDevice + 1, "second build did not take the cached address space");
        GuestHeap::GuestHeapFree_nid_postfix(block);
        Require(!registered(), "freed guest heap block remains registered");
        const auto dropped = AddressSpaceCounters();
        if (dropped.enabled) Require(dropped.waiterDrops == after.waiterDrops + 1 && LeaseCounters().cacheDrops == dropped.waiterDrops, "the free did not drop the cached address space");
    }
    heapMirrorTests(context, access);
    addressTemplateTests(context, access);
    Require(AddressCopyOverflow({{0x1000, 0x3000, 0x2000, "uncommitted pages"}}, 0x2000).empty(), "copies within the limit were refused");
    const auto copies = AddressCopyOverflow({{0x1000, 0x2000, 0x1000, "not mirrored"}, {0x10000, 0x30000, 0x18000, "uncommitted pages"}}, 0x2000);
    Require(!copies.empty() && copies.find("0x10000+0x20000 (0.1 MiB committed, uncommitted pages)") < copies.find("0x1000+0x1000"), "the copy limit does not name the largest copy first");
}
