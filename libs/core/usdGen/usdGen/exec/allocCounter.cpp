// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — engine heap-allocation counter. See allocCounter.h.
#include "usdGen/exec/allocCounter.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace usdGen {
namespace {

// Constant-initialized: valid before any dynamic initializer allocates.
std::atomic<bool> g_counting{false};
std::atomic<bool> g_tracing{false};
std::atomic<uint64_t> g_count{0};

struct AllocTraceSlot
{
    void *frames[kUsdGenAllocTraceFrames];
    int count;
    std::size_t bytes;
};
AllocTraceSlot g_trace[kUsdGenAllocTraceSlots];

inline void Note(std::size_t bytes)
{
    if (!g_counting.load(std::memory_order_relaxed)) return;
    uint64_t const n = g_count.fetch_add(1, std::memory_order_relaxed);
    if (!g_tracing.load(std::memory_order_relaxed) || n >= kUsdGenAllocTraceSlots) return;
    AllocTraceSlot &s = g_trace[n];
    s.bytes = bytes;
#ifdef _WIN32
    s.count = RtlCaptureStackBackTrace(2, kUsdGenAllocTraceFrames, s.frames, nullptr);
#else
    s.frames[0] = __builtin_return_address(0);
    s.count = 1;
#endif
}

#ifdef _WIN32
// Every module links its own operator new, and the pxr DLLs instantiate
// VtArray storage and UsdGenToken/SdfPath tables inside themselves, so no single
// operator new sees the process. All of them end in the CRT heap imports;
// counting wraps those import slots in every loaded module (the ELF
// equivalent of an exe-level operator new, which interposes everywhere).
using MallocFn = void *(__cdecl *)(std::size_t);
using CallocFn = void *(__cdecl *)(std::size_t, std::size_t);
using ReallocFn = void *(__cdecl *)(void *, std::size_t);
using AlignedMallocFn = void *(__cdecl *)(std::size_t, std::size_t);
using AlignedReallocFn = void *(__cdecl *)(void *, std::size_t, std::size_t);
MallocFn g_malloc = nullptr;
CallocFn g_calloc = nullptr;
ReallocFn g_realloc = nullptr;
AlignedMallocFn g_alignedMalloc = nullptr;
AlignedReallocFn g_alignedRealloc = nullptr;

void *__cdecl CountedMalloc(std::size_t n) { Note(n); return g_malloc(n); }
void *__cdecl CountedCalloc(std::size_t c, std::size_t n) { Note(c * n); return g_calloc(c, n); }
void *__cdecl CountedRealloc(void *p, std::size_t n) { Note(n); return g_realloc(p, n); }
void *__cdecl CountedAlignedMalloc(std::size_t n, std::size_t a)
{
    Note(n);
    return g_alignedMalloc(n, a);
}
void *__cdecl CountedAlignedRealloc(void *p, std::size_t n, std::size_t a)
{
    Note(n);
    return g_alignedRealloc(p, n, a);
}

struct ImportHook
{
    char const *name;
    void *hook;
    void *original;
};

void PatchModule(HMODULE module, ImportHook const *hooks, int hookCount)
{
    auto *base = reinterpret_cast<BYTE *>(module);
    auto *dos = reinterpret_cast<IMAGE_DOS_HEADER *>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto *nt = reinterpret_cast<IMAGE_NT_HEADERS *>(base + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY const &dir =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return;
    for (auto *imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress);
         imp->Name; ++imp) {
        if (!imp->OriginalFirstThunk) continue;
        auto *names = reinterpret_cast<IMAGE_THUNK_DATA *>(base + imp->OriginalFirstThunk);
        auto *slots = reinterpret_cast<IMAGE_THUNK_DATA *>(base + imp->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) continue;
            auto *byName = reinterpret_cast<IMAGE_IMPORT_BY_NAME *>(base + names->u1.AddressOfData);
            for (int h = 0; h < hookCount; ++h) {
                // Only slots bound to the release CRT; patched ones are skipped.
                if (std::strcmp(reinterpret_cast<char const *>(byName->Name), hooks[h].name) ||
                    slots->u1.Function != reinterpret_cast<ULONG_PTR>(hooks[h].original))
                    continue;
                DWORD old = 0;
                if (VirtualProtect(&slots->u1.Function, sizeof(ULONG_PTR), PAGE_READWRITE, &old)) {
                    reinterpret_cast<std::atomic<ULONG_PTR> *>(&slots->u1.Function)
                        ->store(reinterpret_cast<ULONG_PTR>(hooks[h].hook));
                    VirtualProtect(&slots->u1.Function, sizeof(ULONG_PTR), old, &old);
                }
            }
        }
    }
}

// Idempotent; picks up modules loaded since the last call. Test/bench only:
// it runs when counting starts, never on an engine path.
void PatchLoadedModules()
{
    HMODULE crt = GetModuleHandleW(L"ucrtbase.dll");
    if (!crt) return;
    g_malloc = reinterpret_cast<MallocFn>(GetProcAddress(crt, "malloc"));
    g_calloc = reinterpret_cast<CallocFn>(GetProcAddress(crt, "calloc"));
    g_realloc = reinterpret_cast<ReallocFn>(GetProcAddress(crt, "realloc"));
    g_alignedMalloc = reinterpret_cast<AlignedMallocFn>(GetProcAddress(crt, "_aligned_malloc"));
    g_alignedRealloc = reinterpret_cast<AlignedReallocFn>(GetProcAddress(crt, "_aligned_realloc"));
    if (!g_malloc || !g_calloc || !g_realloc || !g_alignedMalloc || !g_alignedRealloc) return;
    ImportHook const hooks[] = {
        {"malloc", reinterpret_cast<void *>(&CountedMalloc), reinterpret_cast<void *>(g_malloc)},
        {"calloc", reinterpret_cast<void *>(&CountedCalloc), reinterpret_cast<void *>(g_calloc)},
        {"realloc", reinterpret_cast<void *>(&CountedRealloc), reinterpret_cast<void *>(g_realloc)},
        {"_aligned_malloc", reinterpret_cast<void *>(&CountedAlignedMalloc),
         reinterpret_cast<void *>(g_alignedMalloc)},
        {"_aligned_realloc", reinterpret_cast<void *>(&CountedAlignedRealloc),
         reinterpret_cast<void *>(g_alignedRealloc)},
    };
    HMODULE modules[2048];
    DWORD needed = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed)) return;
    DWORD const count = needed / sizeof(HMODULE) < 2048 ? needed / sizeof(HMODULE) : 2048;
    for (DWORD i = 0; i < count; ++i)
        if (modules[i] != crt) PatchModule(modules[i], hooks, int(sizeof(hooks) / sizeof(hooks[0])));
}
#endif

}  // namespace

void UsdGenEngineAllocCounting(bool on)
{
    if (on) {
#ifdef _WIN32
        PatchLoadedModules();
#endif
        g_count.store(0, std::memory_order_relaxed);
        for (AllocTraceSlot &s : g_trace) s.count = 0;
    }
    g_counting.store(on, std::memory_order_seq_cst);
}

uint64_t UsdGenEngineAllocCount()
{
    return g_count.load(std::memory_order_relaxed);
}

void UsdGenEngineAllocNote(std::size_t bytes) { Note(bytes); }

void UsdGenEngineAllocTracing(bool on)
{
    g_tracing.store(on, std::memory_order_relaxed);
}

int UsdGenEngineAllocTrace(int slot, void **frames, int maxFrames, std::size_t *bytes)
{
    if (slot < 0 || slot >= kUsdGenAllocTraceSlots ||
        uint64_t(slot) >= g_count.load(std::memory_order_relaxed))
        return 0;
    AllocTraceSlot const &s = g_trace[slot];
    int const n = s.count < maxFrames ? s.count : maxFrames;
    for (int i = 0; i < n; ++i) frames[i] = s.frames[i];
    if (bytes) *bytes = s.bytes;
    return n;
}

}  // namespace usdGen

