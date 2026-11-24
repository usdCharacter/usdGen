#ifndef USDGEN_EXECUTION_RUNTIME_INTERNAL_H
#define USDGEN_EXECUTION_RUNTIME_INTERNAL_H

#include "usdGen/executionPipeline.h"
#include "usdGen/taskScheduler.h"
#include <algorithm>
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

// Inline so both libusdGen (default visibility) and usdGenExecutionResources
// (hidden) get a copy; a single out-of-line definition cannot be referenced
// across that DSO boundary.
inline bool UsdGenProcessShutdownInProgress() noexcept {
#if defined(_WIN32)
    using Fn = BOOLEAN(NTAPI*)();
    // Looked up per call (shutdown paths only): a function-local static
    // would add a guarded initialisation.
    Fn const fn = reinterpret_cast<Fn>(reinterpret_cast<void*>(
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
