// APS5_FAST_HEAP=1: a thread-caching small-object allocator behind the C runtime's malloc family.
// The driver's per-draw work makes and frees many small objects (shared_ptr control blocks, vectors,
// hash nodes, std::function state) on several threads; through ucrtbase they cost a process-heap
// RtlAllocateHeap/RtlFreeHeap each, which sampling put at 20-40% of the queue, prep and commit
// threads' busy time.
//
// ucrtbase's exported malloc, free, realloc, calloc, _msize, _recalloc and _expand get a 5-byte jump
// to a trampoline near the module (written as one aligned 8-byte store, so a thread already inside
// one of them keeps running the original bytes). The C runtime's internal callers use the *_base
// functions directly and are left alone. Requests up to 32 KiB are served from size-class spans in a
// reserved arena; larger ones, and every block that is not in the arena (made before the redirect or
// by the runtime's internal callers), go to the original functions. Freed blocks go to a per-thread
// cache, which trades batches with a per-class global pool, so a block freed on another thread is
// reused there. Thread caches are returned to the pools when a thread exits.
#include <windows.h>
#include <intrin.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {

using Malloc = void* (*)(std::size_t);
using Free = void (*)(void*);
using Realloc = void* (*)(void*, std::size_t);
using Calloc = void* (*)(std::size_t, std::size_t);
using Msize = std::size_t (*)(void*);
using Recalloc = void* (*)(void*, std::size_t, std::size_t);
using Expand = void* (*)(void*, std::size_t);

Malloc realMalloc;
Free realFree;
Realloc realRealloc;
Calloc realCalloc;
Msize realMsize;
Recalloc realRecalloc;
Expand realExpand;

constexpr std::size_t ArenaBytes = 64ull << 30;
constexpr std::size_t SpanShift = 16;
constexpr std::size_t SpanBytes = std::size_t{1} << SpanShift;
constexpr std::size_t CommitStep = 4u << 20;
constexpr std::size_t MaxSmall = 32u << 10;
constexpr std::size_t MaxClasses = 64;

std::uint8_t* arenaBase;
std::uint8_t* arenaEnd;
std::atomic<std::size_t> arenaUsed{0};
std::size_t arenaCommitted;
SRWLOCK commitLock = SRWLOCK_INIT;
std::uint8_t* spanClass;

std::uint32_t classCount;
std::uint32_t classSize[MaxClasses];
std::uint32_t classBatch[MaxClasses];
std::uint8_t smallClass[(1024 >> 4) + 1];

struct Pool {
    SRWLOCK lock = SRWLOCK_INIT;
    void* batches = nullptr;
};
Pool pools[MaxClasses];

// head is the list blocks are taken from and freed to; when it reaches a batch it becomes the spare
// (one full batch held back), and an older spare goes to the pool. Nothing walks a list.
struct Cache {
    void* head[MaxClasses];
    std::uint32_t count[MaxClasses];
    void* spare[MaxClasses];
};

// Slow-path counters for the 10 s report.
std::atomic<std::uint64_t> carves{0}, refills{0}, returns{0}, large{0}, foreignFrees{0};

DWORD cacheSlot = TLS_OUT_OF_INDEXES;
DWORD exitSlot = FLS_OUT_OF_INDEXES;

inline bool Owned(const void* pointer) {
    return pointer >= arenaBase && pointer < arenaEnd;
}

inline void*& Next(void* block) {
    return *static_cast<void**>(block);
}

// A pooled batch's head block holds the next batch in its second word, with the batch's block count
// in the top 16 bits.
inline void SetNextBatch(void* block, void* next, std::uint32_t count) {
    static_cast<std::uintptr_t*>(block)[1] = reinterpret_cast<std::uintptr_t>(next) | (static_cast<std::uintptr_t>(count) << 48);
}

inline void* NextBatch(void* block) {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t*>(block)[1] & ((std::uintptr_t{1} << 48) - 1));
}

inline std::uint32_t BatchCount(void* block) {
    return static_cast<std::uint32_t>(static_cast<std::uintptr_t*>(block)[1] >> 48);
}

inline std::uint32_t ClassOf(std::size_t bytes) {
    if (bytes <= 1024) return smallClass[(bytes + 15) >> 4];
    std::uint32_t low = 0, high = classCount;
    while (low < high) {
        const auto middle = (low + high) / 2;
        if (classSize[middle] < bytes) low = middle + 1; else high = middle;
    }
    return low;
}

inline Cache* ThreadCache() {
    // TlsGetValue would clear the thread's last error; read the TEB slot directly (slots < 64).
    return reinterpret_cast<Cache*>(__readgsqword(0x1480 + cacheSlot * 8));
}

void ReturnBatch(std::uint32_t sizeClass, void* head, std::uint32_t count) {
    returns.fetch_add(1, std::memory_order_relaxed);
    auto& pool = pools[sizeClass];
    AcquireSRWLockExclusive(&pool.lock);
    SetNextBatch(head, pool.batches, count);
    pool.batches = head;
    ReleaseSRWLockExclusive(&pool.lock);
}

void WINAPI FlushThread(void* value) {
    auto* cache = static_cast<Cache*>(value);
    if (cache == nullptr) return;
    for (std::uint32_t c = 0; c < classCount; ++c) {
        if (cache->head[c] != nullptr) ReturnBatch(c, cache->head[c], cache->count[c]);
        if (cache->spare[c] != nullptr) ReturnBatch(c, cache->spare[c], classBatch[c]);
        cache->head[c] = cache->spare[c] = nullptr;
        cache->count[c] = 0;
    }
    TlsSetValue(cacheSlot, nullptr);
    VirtualFree(cache, 0, MEM_RELEASE);
}

Cache* MakeCache() {
    auto* cache = static_cast<Cache*>(VirtualAlloc(nullptr, sizeof(Cache), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (cache == nullptr) return nullptr;
    const DWORD error = GetLastError();
    TlsSetValue(cacheSlot, cache);
    FlsSetValue(exitSlot, cache);
    SetLastError(error);
    return cache;
}

bool CarveSpan(std::uint32_t sizeClass, Cache* cache) {
    carves.fetch_add(1, std::memory_order_relaxed);
    const auto offset = arenaUsed.fetch_add(SpanBytes, std::memory_order_relaxed);
    if (offset + SpanBytes > ArenaBytes) return false;
    if (offset + SpanBytes > arenaCommitted) {
        AcquireSRWLockExclusive(&commitLock);
        while (offset + SpanBytes > arenaCommitted) {
            const DWORD error = GetLastError();
            if (VirtualAlloc(arenaBase + arenaCommitted, CommitStep, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
                SetLastError(error);
                ReleaseSRWLockExclusive(&commitLock);
                return false;
            }
            SetLastError(error);
            arenaCommitted += CommitStep;
        }
        ReleaseSRWLockExclusive(&commitLock);
    }
    spanClass[offset >> SpanShift] = static_cast<std::uint8_t>(sizeClass);
    // Link the span into batches; the first refills this thread (whose head is empty), the rest pool.
    auto* span = arenaBase + offset;
    const auto size = classSize[sizeClass];
    const auto blocks = static_cast<std::uint32_t>(SpanBytes / size);
    const auto batch = classBatch[sizeClass];
    for (std::uint32_t first = 0; first < blocks; first += batch) {
        const auto count = blocks - first < batch ? blocks - first : batch;
        for (std::uint32_t i = 0; i < count; ++i) Next(span + (first + i) * size) = i + 1 < count ? span + (first + i + 1) * size : nullptr;
        if (first == 0) {
            cache->head[sizeClass] = span;
            cache->count[sizeClass] = count;
        } else {
            ReturnBatch(sizeClass, span + first * size, count);
        }
    }
    return true;
}

bool Refill(std::uint32_t sizeClass, Cache* cache) {
    if (cache->spare[sizeClass] != nullptr) {
        cache->head[sizeClass] = cache->spare[sizeClass];
        cache->count[sizeClass] = classBatch[sizeClass];
        cache->spare[sizeClass] = nullptr;
        return true;
    }
    auto& pool = pools[sizeClass];
    void* batch = nullptr;
    std::uint32_t count = 0;
    AcquireSRWLockExclusive(&pool.lock);
    batch = pool.batches;
    if (batch != nullptr) {
        pool.batches = NextBatch(batch);
        count = BatchCount(batch);
    }
    ReleaseSRWLockExclusive(&pool.lock);
    if (batch == nullptr) return CarveSpan(sizeClass, cache);
    refills.fetch_add(1, std::memory_order_relaxed);
    cache->head[sizeClass] = batch;
    cache->count[sizeClass] = count;
    return true;
}

void* FastMalloc(std::size_t bytes) {
    if (bytes > MaxSmall) {
        large.fetch_add(1, std::memory_order_relaxed);
        return realMalloc(bytes);
    }
    const auto sizeClass = ClassOf(bytes == 0 ? 1 : bytes);
    Cache* cache = ThreadCache();
    if (cache == nullptr && (cache = MakeCache()) == nullptr) return realMalloc(bytes);
    void* block = cache->head[sizeClass];
    if (block == nullptr) {
        if (!Refill(sizeClass, cache)) return realMalloc(bytes);
        block = cache->head[sizeClass];
    }
    cache->head[sizeClass] = Next(block);
    --cache->count[sizeClass];
    return block;
}

void FastFree(void* block) {
    if (block == nullptr) return;
    if (!Owned(block)) {
        foreignFrees.fetch_add(1, std::memory_order_relaxed);
        realFree(block);
        return;
    }
    const std::uint32_t sizeClass = spanClass[(static_cast<std::uint8_t*>(block) - arenaBase) >> SpanShift];
    Cache* cache = ThreadCache();
    if (cache == nullptr && (cache = MakeCache()) == nullptr) {
        Next(block) = nullptr;
        ReturnBatch(sizeClass, block, 1);
        return;
    }
    Next(block) = cache->head[sizeClass];
    cache->head[sizeClass] = block;
    if (++cache->count[sizeClass] >= classBatch[sizeClass]) {
        if (cache->spare[sizeClass] != nullptr) ReturnBatch(sizeClass, cache->spare[sizeClass], classBatch[sizeClass]);
        cache->spare[sizeClass] = cache->head[sizeClass];
        cache->head[sizeClass] = nullptr;
        cache->count[sizeClass] = 0;
    }
}

inline std::size_t OwnedSize(void* block) {
    return classSize[spanClass[(static_cast<std::uint8_t*>(block) - arenaBase) >> SpanShift]];
}

void* FastRealloc(void* block, std::size_t bytes) {
    if (block == nullptr) return FastMalloc(bytes);
    if (!Owned(block)) return realRealloc(block, bytes);
    if (bytes == 0) {
        FastFree(block);
        return nullptr;
    }
    const auto old = OwnedSize(block);
    if (bytes <= old && (bytes > old / 2 || old <= 16)) return block;
    void* moved = FastMalloc(bytes);
    if (moved == nullptr) return nullptr;
    std::memcpy(moved, block, bytes < old ? bytes : old);
    FastFree(block);
    return moved;
}

void* FastCalloc(std::size_t count, std::size_t size) {
    if (size != 0 && count > SIZE_MAX / size) return realCalloc(count, size);
    const auto bytes = count * size;
    if (bytes > MaxSmall) return realCalloc(count, size);
    void* block = FastMalloc(bytes);
    if (block != nullptr) std::memset(block, 0, bytes);
    return block;
}

std::size_t FastMsize(void* block) {
    if (block != nullptr && Owned(block)) return OwnedSize(block);
    return realMsize(block);
}

void* FastRecalloc(void* block, std::size_t count, std::size_t size) {
    if (block == nullptr || !Owned(block)) return realRecalloc(block, count, size);
    if (size != 0 && count > SIZE_MAX / size) return nullptr;
    const auto bytes = count * size;
    const auto old = OwnedSize(block);
    void* moved = FastRealloc(block, bytes);
    if (moved != nullptr && bytes > old) std::memset(static_cast<std::uint8_t*>(moved) + old, 0, bytes - old);
    return moved;
}

void* FastExpand(void* block, std::size_t bytes) {
    if (block != nullptr && Owned(block)) return bytes <= OwnedSize(block) ? block : nullptr;
    return realExpand(block, bytes);
}

void BuildClasses() {
    std::uint32_t count = 0;
    for (std::uint32_t size = 16; size <= 128; size += 16) classSize[count++] = size;
    for (std::uint32_t base = 128; base < MaxSmall; base *= 2) {
        for (std::uint32_t step = 1; step <= 4; ++step) classSize[count++] = base + base / 4 * step;
    }
    classCount = count;
    for (std::uint32_t c = 0; c < count; ++c) {
        const auto perBatch = (32u << 10) / classSize[c];
        classBatch[c] = perBatch < 2 ? 2 : (perBatch > 64 ? 64 : perBatch);
    }
    for (std::uint32_t index = 0; index <= (1024 >> 4); ++index) {
        const std::size_t bytes = index == 0 ? 1 : index * 16;
        std::uint32_t c = 0;
        while (classSize[c] < bytes) ++c;
        smallClass[index] = static_cast<std::uint8_t>(c);
    }
}

std::uint8_t* NearPage(std::uint8_t* module) {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const auto granularity = static_cast<std::uintptr_t>(info.dwAllocationGranularity);
    const auto origin = reinterpret_cast<std::uintptr_t>(module) & ~(granularity - 1);
    for (std::uintptr_t distance = granularity; distance < (1ull << 31) - (1ull << 24); distance += granularity) {
        for (const auto candidate : {origin - distance, origin + distance}) {
            if (auto* page = static_cast<std::uint8_t*>(VirtualAlloc(reinterpret_cast<void*>(candidate), 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE))) return page;
        }
    }
    return nullptr;
}

std::uint8_t* JumpTarget(std::uint8_t* function) {
    if (function[0] != 0xe9) return nullptr;
    std::int32_t displacement;
    std::memcpy(&displacement, function + 1, 4);
    return function + 5 + displacement;
}

bool Redirect(std::uint8_t* function, void* target, std::uint8_t*& trampolines) {
    if ((reinterpret_cast<std::uintptr_t>(function) & 7) != 0) return false;
    // mov rax, imm64; jmp rax
    std::uint8_t* stub = trampolines;
    stub[0] = 0x48;
    stub[1] = 0xb8;
    std::memcpy(stub + 2, &target, 8);
    stub[10] = 0xff;
    stub[11] = 0xe0;
    trampolines += 16;
    const std::int64_t displacement = stub - (function + 5);
    if (displacement < INT32_MIN || displacement > INT32_MAX) return false;
    DWORD oldProtect;
    if (!VirtualProtect(function, 8, PAGE_EXECUTE_READWRITE, &oldProtect)) return false;
    std::uint8_t bytes[8];
    std::memcpy(bytes, function, 8);
    bytes[0] = 0xe9;
    const auto rel = static_cast<std::int32_t>(displacement);
    std::memcpy(bytes + 1, &rel, 4);
    std::int64_t word;
    std::memcpy(&word, bytes, 8);
    InterlockedExchange64(reinterpret_cast<volatile LONG64*>(function), word);
    VirtualProtect(function, 8, oldProtect, &oldProtect);
    FlushInstructionCache(GetCurrentProcess(), function, 8);
    return true;
}

DWORD WINAPI ReportMain(void*) {
    for (;;) {
        Sleep(10000);
        std::fprintf(stderr, "[fast-heap] arena %.1f MiB, 10 s: span carves %llu, pool refills %llu, batches returned %llu, large (real) allocations %llu, foreign frees %llu\n", arenaUsed.load(std::memory_order_relaxed) / 1048576.0, static_cast<unsigned long long>(carves.exchange(0)), static_cast<unsigned long long>(refills.exchange(0)), static_cast<unsigned long long>(returns.exchange(0)), static_cast<unsigned long long>(large.exchange(0)), static_cast<unsigned long long>(foreignFrees.exchange(0)));
    }
}

bool Install() {
    HMODULE ucrt = GetModuleHandleW(L"ucrtbase.dll");
    if (ucrt == nullptr) return false;
    auto symbol = [&](const char* name) { return reinterpret_cast<std::uint8_t*>(GetProcAddress(ucrt, name)); };
    auto* malloc_ = symbol("malloc");
    auto* free_ = symbol("free");
    auto* realloc_ = symbol("realloc");
    auto* calloc_ = symbol("calloc");
    auto* msize_ = symbol("_msize");
    auto* recalloc_ = symbol("_recalloc");
    auto* expand_ = symbol("_expand");
    realMalloc = reinterpret_cast<Malloc>(symbol("_malloc_base"));
    realFree = reinterpret_cast<Free>(symbol("_free_base"));
    realRealloc = reinterpret_cast<Realloc>(symbol("_realloc_base"));
    realCalloc = reinterpret_cast<Calloc>(symbol("_calloc_base"));
    realMsize = reinterpret_cast<Msize>(JumpTarget(msize_));
    realRecalloc = reinterpret_cast<Recalloc>(JumpTarget(recalloc_));
    realExpand = reinterpret_cast<Expand>(JumpTarget(expand_));
    if (!malloc_ || !free_ || !realloc_ || !calloc_ || !msize_ || !recalloc_ || !expand_ || !realMalloc || !realFree || !realRealloc || !realCalloc || !realMsize || !realRecalloc || !realExpand) {
        std::fprintf(stderr, "[fast-heap] ucrtbase exports not as expected; not redirected\n");
        return false;
    }
    cacheSlot = TlsAlloc();
    exitSlot = FlsAlloc(FlushThread);
    if (cacheSlot == TLS_OUT_OF_INDEXES || cacheSlot >= 64 || exitSlot == FLS_OUT_OF_INDEXES) {
        std::fprintf(stderr, "[fast-heap] no TLS slot below 64; not redirected\n");
        return false;
    }
    BuildClasses();
    arenaBase = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, ArenaBytes, MEM_RESERVE, PAGE_NOACCESS));
    spanClass = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, ArenaBytes >> SpanShift, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    if (arenaBase == nullptr || spanClass == nullptr) return false;
    arenaEnd = arenaBase + ArenaBytes;
    std::uint8_t* trampolines = NearPage(reinterpret_cast<std::uint8_t*>(ucrt));
    if (trampolines == nullptr) return false;
    // free first: anything allocated by the new malloc must already be freed by the new free.
    const bool ok = Redirect(free_, reinterpret_cast<void*>(&FastFree), trampolines) &&
                    Redirect(msize_, reinterpret_cast<void*>(&FastMsize), trampolines) &&
                    Redirect(expand_, reinterpret_cast<void*>(&FastExpand), trampolines) &&
                    Redirect(recalloc_, reinterpret_cast<void*>(&FastRecalloc), trampolines) &&
                    Redirect(realloc_, reinterpret_cast<void*>(&FastRealloc), trampolines) &&
                    Redirect(calloc_, reinterpret_cast<void*>(&FastCalloc), trampolines) &&
                    Redirect(malloc_, reinterpret_cast<void*>(&FastMalloc), trampolines);
    if (ok) {
        if (HANDLE thread = CreateThread(nullptr, 0, ReportMain, nullptr, 0, nullptr)) CloseHandle(thread);
    }
    std::fprintf(stderr, "[fast-heap] %s: %u size classes up to %zu bytes, arena %p\n", ok ? "malloc family redirected" : "redirect FAILED part way", classCount, MaxSmall, static_cast<void*>(arenaBase));
    return ok;
}

[[maybe_unused]] const bool installed = [] {
    const char* value = std::getenv("APS5_FAST_HEAP");
    return value != nullptr && value[0] == '1' && Install();
}();

}
