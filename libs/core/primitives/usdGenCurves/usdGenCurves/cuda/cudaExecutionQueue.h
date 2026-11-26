#ifndef USDGEN_CUDA_EXECUTION_QUEUE_H
#define USDGEN_CUDA_EXECUTION_QUEUE_H

#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/sharedCell.h"
#include "usdGen/executionPipeline.h"
#include "usdGen/executionTaskGraph.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen {

// Immutable once published; readers hold it by shared reference.
struct UsdGenCudaQueueSnapshot {
    uint64_t epoch = 0;
    double frame = 0.0;
    std::shared_ptr<const UsdGenDeviceGeneration> generation;
    std::vector<UsdGenCudaBindingStats> bindings;
};

// Per-description asynchronous CUDA execution boundary.  The queue owns
// mutable device resources while the plan remains immutable and shareable.
// Drain/destruction are external boundaries after stopping/joining submitters,
// never callback operations. Callbacks enqueue follow-up requests, not waits.
class UsdGenCudaExecutionQueue {
public:
    using Pipeline = UsdGenExecutionPipeline;
    using Completion = std::function<void(uint64_t, Pipeline::Outcome,
                                          UsdGenDiagnostics const&)>;

    UsdGenCudaExecutionQueue(UsdGenExecutionRuntime& runtime, int device,
                             UsdGenDiagnostics* diagnostics = nullptr);
    ~UsdGenCudaExecutionQueue();
    UsdGenCudaExecutionQueue(UsdGenCudaExecutionQueue const&) = delete;
    UsdGenCudaExecutionQueue& operator=(UsdGenCudaExecutionQueue const&) = delete;

    bool Valid() const noexcept;
    // Null plans, invalid queues and nonfinite frames return zero without a
    // callback. Accepted requests report their eventual pipeline outcome.
    uint64_t Submit(std::shared_ptr<const UsdGenCudaExecutionPlan> plan,
                    double frame, Completion completion = {});
    // Selects shared-dispatcher scheduling priority; the three CUDA stages
    // still serialize through this description's workspace.
    uint64_t Submit(std::shared_ptr<const UsdGenCudaExecutionPlan> plan,
                    double frame, Completion,
                    UsdGenExecutionTaskGraph::RequestClass);
    uint64_t CancelPending();
    std::shared_ptr<const UsdGenCudaQueueSnapshot> Snapshot() const noexcept;
    void Drain();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace usdGen

#endif // USDGEN_CUDA_EXECUTION_QUEUE_H
