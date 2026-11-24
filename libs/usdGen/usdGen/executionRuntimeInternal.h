#ifndef USDGEN_EXECUTION_RUNTIME_INTERNAL_H
#define USDGEN_EXECUTION_RUNTIME_INTERNAL_H

#include "usdGen/executionPipeline.h"
#include "usdGen/taskScheduler.h"
#include <algorithm>
#include <tbb/task_arena.h>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace usdGen {
// Shared callback-scope guard across pipeline and ready-task graphs.  It is
// intentionally internal: synchronous waits from either framework are invalid.
extern thread_local bool usdGenExecutionGraphActive;

// Pin the process-wide TBB worker market before any 1-slot cleanup arena is
// created. oneTBB sizes the market from the first initialized arena; the
// retirement drain arena is concurrency-one, and if it wins that race the
// imaging runtime's later request for 8 workers is ignored. Pipeline
// wait_for_all then deadlocks (observed in CI as TBB "workers limited to 1"
// plus 60s ctest timeouts after the test body had already printed PASS).
// The market arena is leaked: tearing it down at DSO unload waits for
// workers that process shutdown may already have stopped.
inline void EnsureUsdGenTbbMarket() {
    static tbb::task_arena* const market = [] {
        int n = 8;
        if (unsigned const hw = std::thread::hardware_concurrency()) {
            n = std::max(n, static_cast<int>(hw));
        }
        auto* arena = new tbb::task_arena(n, 0);
        arena->initialize();
        return arena;
    }();
    (void)market;
}

// Inline so both libusdGen (default visibility) and usdGenExecutionResources
// (hidden) get a copy; a single out-of-line definition cannot be referenced
// across that DSO boundary.
inline bool UsdGenProcessShutdownInProgress() noexcept {
#if defined(_WIN32)
    using Fn = BOOLEAN(NTAPI*)();
    static Fn const fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(
        ::GetProcAddress(::GetModuleHandleW(L"ntdll.dll"), "RtlDllShutdownInProgress")));
    return fn && fn();
#else
    return false;
#endif
}

// Private shared definition: pipelines run their serial lanes on this
// runtime's scheduler and retain it past wrapper destruction.
struct UsdGenExecutionRuntime::Impl {
    explicit Impl(int concurrency) : scheduler(concurrency) {}
    UsdGenTaskScheduler scheduler;
};
}
#endif
