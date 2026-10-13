#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTALLOCATIONS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTALLOCATIONS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <functional>
#include <mutex>
#include <vector>

namespace GuestAllocations {

struct Range {
    std::uint64_t address;
    std::size_t bytes;
    bool readable;
    bool writable;
    std::uint64_t allocationAddress;
    std::size_t allocationBytes;
    bool releasable = true;
};

// The registered readable ranges at one moment. Holding a lease pins every range in it (a
// mutation of a pinned range waits, RequireUnpinned). Copies share one immutable snapshot that the
// registry keeps until its next mutation, so taking a lease is a shared_ptr copy rather than a copy of
// every range's shared_ptr under the registry lock (that copy cost the committer tens of ms per
// frame in a race); the pins are the same, since a range is pinned while any snapshot holding it is
// alive and the registry drops its own reference before a mutation checks them.
class Lease {
public:
    using Items = std::vector<std::shared_ptr<const Range>>;
    Lease() = default;
    explicit Lease(std::shared_ptr<const Items> snapshot) : items(std::move(snapshot)) {}
    Items::const_iterator begin() const { return view().begin(); }
    Items::const_iterator end() const { return view().end(); }
    std::size_t size() const { return view().size(); }
    bool empty() const { return view().empty(); }
    const std::shared_ptr<const Range>& operator[](std::size_t index) const { return view()[index]; }
    const std::shared_ptr<const Range>& front() const { return view().front(); }
    void clear() { items.reset(); }
    // Keeps (and so pins) only the ranges the predicate accepts, in a private copy.
    template <class Predicate>
    void RetainIf(Predicate keep) {
        auto kept = std::make_shared<Items>();
        for (const auto& range : view()) {
            if (keep(range)) kept->push_back(range);
        }
        items = std::move(kept);
    }

private:
    const Items& view() const {
        static const Items none;
        return items != nullptr ? *items : none;
    }
    std::shared_ptr<const Items> items;
};

extern "C" {
void* GuestAllocationsBegin_nid_postfix();
void GuestAllocationsRegisterMainImage_nid_postfix(void* mutation);
void GuestAllocationsEnd_nid_postfix(void* mutation) noexcept;
void GuestAllocationsAdd_nid_postfix(void* mutation, void* pointer, std::size_t bytes, bool readable, bool writable);
void GuestAllocationsRequireUnpinned_nid_postfix(void* mutation, const void* pointer, std::size_t bytes);
void GuestAllocationsRequireAvailable_nid_postfix(void* mutation, const void* pointer, std::size_t bytes);
bool GuestAllocationsCovers_nid_postfix(void* mutation, const void* pointer, std::size_t bytes);
bool GuestAllocationsOverlaps_nid_postfix(void* mutation, const void* pointer, std::size_t bytes);
Range GuestAllocationsFind_nid_postfix(void* mutation, const void* pointer);
void GuestAllocationsRemove_nid_postfix(void* mutation, const void* pointer);
void GuestAllocationsProtect_nid_postfix(void* mutation, const void* pointer, std::size_t bytes, bool readable, bool writable, const std::function<void()>& apply);
// Whether registered entries cover the whole range and each already has these flags (the
// readable/writable pair the registry keeps), so a protection with them changes nothing it holds.
bool GuestAllocationsCarries_nid_postfix(void* mutation, const void* pointer, std::size_t bytes, bool readable, bool writable);
// The driver's switch for keeping same-flag protections out of the registry (see Carries); without
// it APS5_KEEP_SAME_PROTECT decides. Carries answers false while the switch is off.
void GuestAllocationsSetKeepSameQuery_nid_postfix(bool (*query)());
// The driver's switch for shared lease snapshots (see Lease); without it APS5_NO_LEASE_SNAPSHOT decides.
void GuestAllocationsSetSnapshotQuery_nid_postfix(bool (*query)());
void GuestAllocationsUnmap_nid_postfix(void* mutation, const void* pointer, std::size_t bytes, const std::function<void(const void*, std::size_t, const void*, bool)>& apply);
Lease GuestAllocationsAcquire_nid_postfix();
// Changes whenever a mutation ends, so callers can cache facts about guest mappings between changes.
std::uint64_t GuestAllocationsGeneration_nid_postfix();
// Precise invalidation for such caches: the callback runs, after the generation changed, for every
// range whose mapping or protection changed (registry mutations and guest heap decommits).
void GuestAllocationsSetInvalidator_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes));
void GuestAllocationsInvalidate_nid_postfix(std::uintptr_t address, std::size_t bytes);
// A mutation of a range that GPU work still leases waits for the lease (RequireUnpinned). The
// waiter, set by the GPU driver, is called on the mutating thread with the registry lock released
// and finishes the GPU work that holds leases; it returns whether it finished any (a round that found
// nothing to wait for counts against the same bound as a plain spin). Without one the mutation spins
// until the lease is dropped by another thread.
void GuestAllocationsSetPinWaiter_nid_postfix(bool (*callback)());
}

class Mutation {
public:
    Mutation() : handle(GuestAllocationsBegin_nid_postfix()) {}
    ~Mutation() { GuestAllocationsEnd_nid_postfix(handle); }
    Mutation(const Mutation&) = delete;
    Mutation& operator=(const Mutation&) = delete;
    void RegisterMainImage() { GuestAllocationsRegisterMainImage_nid_postfix(handle); }
    void Add(void* pointer, std::size_t bytes, bool readable, bool writable) { GuestAllocationsAdd_nid_postfix(handle, pointer, bytes, readable, writable); }
    void RequireUnpinned(const void* pointer, std::size_t bytes) const { GuestAllocationsRequireUnpinned_nid_postfix(handle, pointer, bytes); }
    void RequireAvailable(const void* pointer, std::size_t bytes) const { GuestAllocationsRequireAvailable_nid_postfix(handle, pointer, bytes); }
    bool Covers(const void* pointer, std::size_t bytes) const { return GuestAllocationsCovers_nid_postfix(handle, pointer, bytes); }
    bool Overlaps(const void* pointer, std::size_t bytes) const { return GuestAllocationsOverlaps_nid_postfix(handle, pointer, bytes); }
    Range Find(const void* pointer) const { return GuestAllocationsFind_nid_postfix(handle, pointer); }
    void Remove(const void* pointer) { GuestAllocationsRemove_nid_postfix(handle, pointer); }
    void Unmap(const void* pointer, std::size_t bytes, const std::function<void(const void*, std::size_t, const void*, bool)>& apply) { GuestAllocationsUnmap_nid_postfix(handle, pointer, bytes, apply); }
    void Protect(const void* pointer, std::size_t bytes, bool readable, bool writable, const std::function<void()>& apply) { GuestAllocationsProtect_nid_postfix(handle, pointer, bytes, readable, writable, apply); }
    bool Carries(const void* pointer, std::size_t bytes, bool readable, bool writable) const { return GuestAllocationsCarries_nid_postfix(handle, pointer, bytes, readable, writable); }

private:
    void* handle;
};

}

#endif
