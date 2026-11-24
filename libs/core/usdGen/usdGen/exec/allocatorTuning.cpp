// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — glibc allocator policy for capture/bake churn.
//
// Fresh capture, bake and draw buffers churn ~70-150MB of mmap/munmap plus
// fresh-page faults per call (the caller owns those buffers, so the
// thread-local pools cannot retain them). Keeping sub-64MB blocks on the
// brk heap (M_MMAP_THRESHOLD) with trimming disabled (M_TRIM_THRESHOLD)
// turns every rep after the first into dirty-page reuse: freed chunks stay
// mapped and the next same-sized allocation reuses them warm, skipping the
// munmap/mmap round-trip, the page-table teardown and the fresh faults.
// Bit-identical: allocation placement only, no value impact. Retention is
// about one churn cycle (~100-200MB at bench scale), the same class as the
// thread-local pools; the fully-pooled paths (CUDA convert, ValidateRoots
// temps) are unaffected either way.
//
// Precedence: an explicitly-set MALLOC_MMAP_THRESHOLD_ /
// MALLOC_TRIM_THRESHOLD_ wins over the corresponding default below, and
// USDGEN_ALLOCATOR_DEFAULTS=1 keeps stock glibc behavior entirely (both
// read once, here, at load). Non-glibc platforms compile to a no-op.
#include <cstdlib>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace usdGen {
namespace {

struct AllocatorPolicy {
    AllocatorPolicy()
    {
#if defined(__GLIBC__)
        if (std::getenv("USDGEN_ALLOCATOR_DEFAULTS"))
            return;
        // 64MB covers churn planes to ~4M roots/CVs (16MB at bench scale);
        // larger blocks keep the mmap path (spread VMAs, released on free).
        if (!std::getenv("MALLOC_MMAP_THRESHOLD_"))
            ::mallopt(M_MMAP_THRESHOLD, 64 << 20);
        if (!std::getenv("MALLOC_TRIM_THRESHOLD_"))
            ::mallopt(M_TRIM_THRESHOLD, -1);
#endif
    }
};

// Load-time: libusdGen is linked by every usdGen consumer (engine sessions,
// CUDA/Vulkan input builders, usdGenImaging, benches, tests), so one static
// initializer covers all entries. libc-only (B-1 clean): getenv + mallopt.
AllocatorPolicy const g_allocatorPolicy;

}  // namespace
}  // namespace usdGen
