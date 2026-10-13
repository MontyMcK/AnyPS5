// MinGW GCC compiles thread_local as emulated TLS: every access calls __emutls_get_address, which
// libgcc_s answers through pthread_getspecific (TlsGetValue between GetLastError/SetLastError), about
// 13 ns an access. The draw path touches thread_locals (scratch vectors, the refresh site, the
// performance context, flag caches) dozens of times per draw. This definition takes the module's
// calls: a variable's first access on a thread goes to libgcc_s (so objects shared with libstdc++
// keep one address per thread), and the address it returns, stable for the thread's life, is kept in
// a per-thread table found through one Windows TLS slot. APS5_NO_FAST_EMUTLS=1 always asks libgcc_s.
#include <windows.h>
#include <intrin.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

struct EmutlsObject {
    std::uintptr_t size;
    std::uintptr_t align;
    union {
        std::uintptr_t offset;
        void* pointer;
    } loc;
    void* templ;
};

using EmutlsGet = void* (*)(EmutlsObject*);

// 0: not resolved; the libgcc_s function otherwise.
std::atomic<EmutlsGet> original{nullptr};
// TLS_OUT_OF_INDEXES - 1: not allocated yet (TLS_OUT_OF_INDEXES: the allocation failed).
std::atomic<DWORD> tableSlot{TLS_OUT_OF_INDEXES - 1u};
// 1: fast path on, 2: off.
std::atomic<int> mode{0};

EmutlsGet resolveOriginal() {
    auto function = original.load(std::memory_order_acquire);
    if (function != nullptr) return function;
    const auto module = GetModuleHandleW(L"libgcc_s_seh-1.dll");
    function = module != nullptr ? reinterpret_cast<EmutlsGet>(reinterpret_cast<void*>(GetProcAddress(module, "__emutls_get_address"))) : nullptr;
    if (function == nullptr) std::abort();
    original.store(function, std::memory_order_release);
    return function;
}

DWORD resolveSlot() {
    auto slot = tableSlot.load(std::memory_order_acquire);
    if (slot != TLS_OUT_OF_INDEXES - 1u) return slot;
    const auto allocated = TlsAlloc();
    auto expected = TLS_OUT_OF_INDEXES - 1u;
    if (!tableSlot.compare_exchange_strong(expected, allocated, std::memory_order_acq_rel)) {
        if (allocated != TLS_OUT_OF_INDEXES) TlsFree(allocated);
        return expected;
    }
    return allocated;
}

bool fastEnabled() {
    auto current = mode.load(std::memory_order_relaxed);
    if (current == 0) {
        char value[8];
        current = GetEnvironmentVariableA("APS5_NO_FAST_EMUTLS", value, sizeof(value)) != 0 ? 2 : 1;
        mode.store(current, std::memory_order_relaxed);
    }
    return current == 1;
}

// The thread's table: word 0 holds the capacity, word k the address of the variable at offset k.
inline void** threadTable(DWORD slot) {
    if (slot < 64u) return static_cast<void**>(reinterpret_cast<void*>(__readgsqword(0x1480u + slot * 8u)));
    return static_cast<void**>(TlsGetValue(slot));
}

}

extern "C" void* __emutls_get_address(EmutlsObject* object) {
    const auto offset = __atomic_load_n(&object->loc.offset, __ATOMIC_ACQUIRE);
    const auto slot = tableSlot.load(std::memory_order_relaxed);
    if (offset != 0 && slot < TLS_OUT_OF_INDEXES - 1u) {
        void** table = threadTable(slot);
        if (table != nullptr && offset < reinterpret_cast<std::uintptr_t>(table[0]) && table[offset] != nullptr) return table[offset];
    }
    void* address = resolveOriginal()(object);
    if (!fastEnabled()) return address;
    const auto known = __atomic_load_n(&object->loc.offset, __ATOMIC_ACQUIRE);
    const auto tls = resolveSlot();
    if (known == 0 || tls == TLS_OUT_OF_INDEXES) return address;
    void** table = threadTable(tls);
    const auto capacity = table != nullptr ? reinterpret_cast<std::uintptr_t>(table[0]) : 0u;
    if (known >= capacity) {
        auto grown = known * 2u + 32u;
        auto* replacement = static_cast<void**>(std::calloc(grown, sizeof(void*)));
        if (replacement == nullptr) return address;
        if (table != nullptr) std::memcpy(replacement + 1, table + 1, (capacity - 1u) * sizeof(void*));
        replacement[0] = reinterpret_cast<void*>(grown);
        if (!TlsSetValue(tls, replacement)) {
            std::free(replacement);
            return address;
        }
        std::free(table);
        table = replacement;
    }
    table[known] = address;
    return address;
}
