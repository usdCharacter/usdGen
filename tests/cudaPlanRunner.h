// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Test driver for the staged CUDA executor. The synchronous executor was
// deleted in 8.7a; tests that need a plan run on a chosen workspace drive the
// same stage API a Session uses (source, operators in index order,
// publication) and wait for each completion. Test hook waits only.
#ifndef USDGEN_TEST_CUDA_PLAN_RUNNER_H
#define USDGEN_TEST_CUDA_PLAN_RUNNER_H

#include "usdGenCurves/cuda/cudaExecution.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <thread>
#include <utility>

namespace usdGen {
namespace cudaPlanRunner {

constexpr auto kStageTimeout = std::chrono::seconds(60);

// Waits briefly until relays dropped their job references, so borrowed
// workspace and diagnostics may die after the call returns. A quarantined
// relay keeps its job for good (and has detached the workspace): stop early.
inline void AwaitJobRelease(std::shared_ptr<UsdGenCudaExecutionJob> const& job) {
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (job.use_count() > 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
}

inline void FailIfSilent(UsdGenDiagnostics* diagnostics, char const* message) {
    if (diagnostics && !diagnostics->HasErrors()) diagnostics->Error(message);
}

} // namespace cudaPlanRunner

// Source stage; false when admission is rejected or the stage fails.
inline bool RunCudaJobSourceForTesting(std::shared_ptr<UsdGenCudaExecutionJob> const& job) {
    auto done = std::make_shared<std::promise<bool>>();
    auto result = done->get_future();
    if (!ExecuteCudaJobSourceAsync(job, [done](bool ok) { done->set_value(ok); }))
        return false;
    return result.wait_for(cudaPlanRunner::kStageTimeout) == std::future_status::ready &&
        result.get();
}

// One operator stage; false when admission is rejected or the stage fails.
inline bool RunCudaJobOperatorForTesting(std::shared_ptr<UsdGenCudaExecutionJob> const& job,
                                         size_t index) {
    auto done = std::make_shared<std::promise<bool>>();
    auto result = done->get_future();
    if (!ExecuteCudaJobOperatorAsync(job, index, [done](bool ok) { done->set_value(ok); }))
        return false;
    return result.wait_for(cudaPlanRunner::kStageTimeout) == std::future_status::ready &&
        result.get();
}

// Publication stage; null when admission is rejected or the stage fails.
inline std::shared_ptr<const UsdGenDeviceGeneration> FinalizeCudaJobForTesting(
    std::shared_ptr<UsdGenCudaExecutionJob> const& job) {
    using Generation = std::shared_ptr<const UsdGenDeviceGeneration>;
    auto done = std::make_shared<std::promise<Generation>>();
    auto result = done->get_future();
    if (!FinalizeCudaExecutionJobAsync(job,
            [done](Generation generation) { done->set_value(std::move(generation)); }))
        return {};
    if (result.wait_for(cudaPlanRunner::kStageTimeout) != std::future_status::ready) return {};
    return result.get();
}

// Whole plan on one workspace: the staged replacement for ExecuteCudaGraph.
inline std::shared_ptr<const UsdGenDeviceGeneration> ExecuteCudaPlanForTesting(
    std::shared_ptr<const UsdGenCudaExecutionPlan> const& plan,
    UsdGenCudaExecutionWorkspace& workspace, double frame, uint64_t generation,
    UsdGenDiagnostics* diagnostics,
    std::shared_ptr<const UsdGenDeviceGeneration> const& previous = {}) {
    auto job = CreateCudaExecutionJob(plan, workspace, frame, generation, diagnostics,
                                      previous);
    if (!job) return {};
    std::shared_ptr<const UsdGenDeviceGeneration> result;
    bool ok = RunCudaJobSourceForTesting(job);
    for (size_t i = 0; ok && i != CudaExecutionJobOperatorCount(*job); ++i)
        ok = RunCudaJobOperatorForTesting(job, i);
    if (ok) result = FinalizeCudaJobForTesting(job);
    cudaPlanRunner::AwaitJobRelease(job);
    if (!result) cudaPlanRunner::FailIfSilent(diagnostics, "CUDA execution request failed");
    return result;
}

} // namespace usdGen

#endif // USDGEN_TEST_CUDA_PLAN_RUNNER_H
