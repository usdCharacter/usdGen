// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Test fixture: per-description staged CUDA execution queue. It drives the
// public stage API (source, operators, publication) on the shared execution
// task graph. Moved out of usdGenCurvesCuda in 8.7a: only tests used it.
#ifndef USDGEN_TEST_CUDA_EXECUTION_QUEUE_FIXTURE_H
#define USDGEN_TEST_CUDA_EXECUTION_QUEUE_FIXTURE_H

#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/api/sharedCell.h"
#include "usdGen/api/pipeline.h"
#include "usdGen/api/ext/device/executionTaskGraph.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

struct UsdGenCudaExecutionQueue::Impl {
    // Declaration order is intentional: the pipeline drains before the
    // workspace is destroyed, and the workspace owns all mutable CUDA state.
    std::unique_ptr<UsdGenCudaExecutionWorkspace> workspace;
    // The displaced snapshot (and the generation it holds) is released by
    // the thread that displaces it, and the last one when Impl destroys the
    // cell, before the workspace: never later on some epoch drainer.
    UsdGenSharedCell<const UsdGenCudaQueueSnapshot> snapshot;
    std::shared_ptr<UsdGenExecutionTaskGraph> dispatcher;
    std::unique_ptr<Pipeline> pipeline;

    Impl(UsdGenExecutionRuntime& runtime, int device,
         UsdGenDiagnostics* diagnostics)
        : workspace(CreateCudaExecutionWorkspace(device, diagnostics)),
          pipeline(std::make_unique<Pipeline>(runtime)) {
        if (workspace)
            dispatcher = UsdGenExecutionTaskGraph::GetOrCreate(
                runtime, "cuda", workspace->DeviceIndex());
    }
};

inline UsdGenCudaExecutionQueue::UsdGenCudaExecutionQueue(
    UsdGenExecutionRuntime& runtime, int device, UsdGenDiagnostics* diagnostics)
    : impl_(std::make_unique<Impl>(runtime, device, diagnostics)) {}

inline UsdGenCudaExecutionQueue::~UsdGenCudaExecutionQueue() = default;

inline bool UsdGenCudaExecutionQueue::Valid() const noexcept {
    return impl_ && impl_->workspace && impl_->pipeline && impl_->dispatcher;
}

inline uint64_t UsdGenCudaExecutionQueue::Submit(
    std::shared_ptr<const UsdGenCudaExecutionPlan> plan, double frame,
    Completion completion) {
    return Submit(std::move(plan), frame, std::move(completion),
        UsdGenExecutionTaskGraph::RequestClass::Interactive);
}

inline uint64_t UsdGenCudaExecutionQueue::Submit(
    std::shared_ptr<const UsdGenCudaExecutionPlan> plan, double frame,
    Completion completion, UsdGenExecutionTaskGraph::RequestClass requestClass) {
    if (!Valid() || !plan || !std::isfinite(frame) ||
        (requestClass != UsdGenExecutionTaskGraph::RequestClass::Interactive &&
         requestClass != UsdGenExecutionTaskGraph::RequestClass::Background)) return 0;

    auto diagnostics = std::make_shared<UsdGenDiagnostics>();
    auto metadata = GetCudaExecutionPlanMetadata(*plan);
    if (!metadata || (metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
                      metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
                      metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedValueDag) ||
        metadata->Operators().empty() ||
        // Authored capabilities are not executable task cardinality: the
        // Scatter/Grow native producer fuses two authored operators.
        metadata->Tasks().size() < 2 ||
        !metadata->FindTask(metadata->TerminalTask())) {
        diagnostics->Error("CUDA execution plan metadata is missing or incompatible");
        return 0;
    }
    for (uint32_t i = 0; i != metadata->Tasks().size(); ++i) {
        if (!metadata->FindTask(i)) {
            diagnostics->Error("CUDA execution plan task ids are not dense");
            return 0;
        }
    }
    std::string dependencyReason;
    if (!UsdGenExecutionDependencyCompiler::Validate(
            metadata->Tasks(), metadata->Values(), &dependencyReason)) {
        diagnostics->Error("CUDA execution plan dependencies are invalid: " +
            dependencyReason);
        return 0;
    }
    auto* impl = impl_.get();
    auto work = [impl, plan = std::move(plan), metadata = std::move(metadata),
                 frame, requestClass, diagnostics]
                (Pipeline::Cancellation const& cancel,
                 Pipeline::AsyncCompletion done) mutable {
        // Cancellation is checked by the pipeline before publication.  Once
        // device work starts, execute it to completion so workspace buffers
        // and CUDA events remain valid; a superseding epoch discards the
        // resulting publication action in the owner node.
        auto previousSnapshot = impl->snapshot.Load();
        auto previous = previousSnapshot ? previousSnapshot->generation
                                         : std::shared_ptr<const UsdGenDeviceGeneration>{};
        auto completion = std::make_shared<Pipeline::AsyncCompletion>(std::move(done));
        auto jobState = CreateCudaExecutionJob(std::move(plan), *impl->workspace,
            frame, cancel.epoch, diagnostics.get(), std::move(previous));
        if (!jobState) {
            // Preserve the precise admission/allocation diagnostic produced
            // by job creation instead of appending a misleading generic one.
            if (!diagnostics->HasErrors())
                diagnostics->Error("CUDA execution job allocation failed");
            (*completion)({}, std::make_exception_ptr(
                std::runtime_error("CUDA execution job allocation failed")));
            return;
        }
        if (metadata->Tasks().size() != CudaExecutionJobOperatorCount(*jobState) + 2) {
            (*completion)({}, std::make_exception_ptr(
                std::runtime_error("CUDA execution plan task count differs from native stages")));
            return;
        }
        UsdGenExecutionTaskGraph::Job job;
        job.cancellation = cancel;
        job.requestClass = requestClass;
        // Source -> each authored operator -> publication.  The dependency
        // chain serializes mutable workspace state while still allowing the
        // shared dispatcher to interleave independent descriptions.
        auto const* sourceMetadata = metadata->FindTask(0);
        if (!sourceMetadata || sourceMetadata->kind != UsdGenExecutionTaskKind::Source ||
            sourceMetadata->id != job.tasks.size()) {
            (*completion)({}, std::make_exception_ptr(
                std::runtime_error("CUDA source task metadata is missing")));
            return;
        }
        job.tasks.push_back({sourceMetadata->dependencies,
                            [jobState](Pipeline::Cancellation const& taskCancel,
                                            UsdGenExecutionTaskGraph::TaskCompletion finish) {
            auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                std::move(finish));
            try {
                if (taskCancel.Superseded()) {
                    (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA source stage failed")));
                    return;
                }
                if (!ExecuteCudaJobSourceAsync(jobState,
                        [completion](bool ok) {
                    if (ok) (*completion)({}, {});
                    else (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA source stage failed")));
                })) {
                    (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA source stage failed")));
                }
            } catch (...) { (*completion)({}, std::current_exception()); }
        }});
        auto const operatorCount = CudaExecutionJobOperatorCount(*jobState);
        for (size_t index = 0; index != operatorCount; ++index) {
            auto const* operatorMetadata = metadata->FindTask(
                static_cast<uint32_t>(index + 1));
            if (!operatorMetadata || operatorMetadata->kind != UsdGenExecutionTaskKind::Operator ||
                operatorMetadata->id != job.tasks.size()) {
                (*completion)({}, std::make_exception_ptr(
                    std::runtime_error("CUDA operator task metadata is missing")));
                return;
            }
            job.tasks.push_back({operatorMetadata->dependencies,
                [jobState, index](Pipeline::Cancellation const& taskCancel,
                                  UsdGenExecutionTaskGraph::TaskCompletion finish) {
                    auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                        std::move(finish));
                    try {
                        if (taskCancel.Superseded()) {
                            (*completion)({}, std::make_exception_ptr(
                                std::runtime_error("CUDA operator stage " + std::to_string(index) + " superseded")));
                            return;
                        }
                        if (!ExecuteCudaJobOperatorAsync(jobState, index,
                                [completion, index](bool ok) {
                            if (ok) (*completion)({}, {});
                            else (*completion)({}, std::make_exception_ptr(
                                std::runtime_error("CUDA operator stage " + std::to_string(index) + " completion failed")));
                        })) {
                            (*completion)({}, std::make_exception_ptr(
                                std::runtime_error("CUDA operator stage " + std::to_string(index) + " submission failed")));
                        }
                    } catch (...) { (*completion)({}, std::current_exception()); }
                }});
        }
        auto const* terminalMetadata = metadata->FindTask(metadata->TerminalTask());
        if (!terminalMetadata || terminalMetadata->kind != UsdGenExecutionTaskKind::Publication ||
            terminalMetadata->id != job.tasks.size()) {
            (*completion)({}, std::make_exception_ptr(
                std::runtime_error("CUDA publication task metadata is missing")));
            return;
        }
        job.tasks.push_back({terminalMetadata->dependencies,
                            [impl, jobState,
                                  frame, diagnostics]
                                 (Pipeline::Cancellation const& taskCancel,
                                  UsdGenExecutionTaskGraph::TaskCompletion finish) mutable {
            auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                std::move(finish));
            try {
                if (taskCancel.Superseded()) { (*completion)({}, {}); return; }
                if (!FinalizeCudaExecutionJobAsync(jobState,
                        [impl, completion, frame, diagnostics,
                         epoch = taskCancel.epoch](std::shared_ptr<const UsdGenDeviceGeneration> generation) mutable {
                    try {
                        if (!generation) {
                            if (!diagnostics->HasErrors())
                                diagnostics->Error("CUDA execution produced no generation");
                            (*completion)({}, std::make_exception_ptr(
                                std::runtime_error("CUDA execution failed")));
                            return;
                        }
                        auto next = std::make_shared<UsdGenCudaQueueSnapshot>();
                        next->epoch = epoch;
                        next->frame = frame;
                        next->generation = std::move(generation);
                        next->bindings = GetCudaBindingStats(*impl->workspace);
                        (*completion)([impl, next = std::shared_ptr<const UsdGenCudaQueueSnapshot>(
                                          std::move(next))] {
                            impl->snapshot.Store(next);
                        }, {});
                    } catch (...) { (*completion)({}, std::current_exception()); }
                })) {
                    (*completion)({}, std::make_exception_ptr(
                        std::runtime_error("CUDA final stage failed")));
                }
            } catch (...) { (*completion)({}, std::current_exception()); }
        }});
        job.finalTask = terminalMetadata->id;
        job.completion = [completion](Pipeline::Publish publish,
                                      std::exception_ptr error) mutable {
            (*completion)(std::move(publish), error);
        };
        if (!impl->dispatcher->Submit(std::move(job))) {
            diagnostics->Error("CUDA execution dispatcher rejected request");
            (*completion)({}, std::make_exception_ptr(
                std::runtime_error("CUDA execution dispatcher rejected request")));
        }
    };
    auto callback = [diagnostics, completion = std::move(completion)](
                        uint64_t epoch, Pipeline::Outcome outcome,
                        std::exception_ptr error) {
        if (!diagnostics->HasErrors() && error &&
            outcome == Pipeline::Outcome::Failed)
            diagnostics->Error("CUDA execution request failed");
        if (completion) completion(epoch, outcome, *diagnostics);
    };
    return impl_->pipeline->SubmitAsync(std::move(work), std::move(callback));
}

inline uint64_t UsdGenCudaExecutionQueue::CancelPending() {
    return Valid() ? impl_->pipeline->CancelPending() : 0;
}

inline std::shared_ptr<const UsdGenCudaQueueSnapshot>
UsdGenCudaExecutionQueue::Snapshot() const noexcept {
    if (!impl_) return {};
    return impl_->snapshot.Load();
}

inline void UsdGenCudaExecutionQueue::Drain() {
    if (!Valid()) return;
    impl_->pipeline->Drain();
}

} // namespace usdGen

#endif // USDGEN_TEST_CUDA_EXECUTION_QUEUE_FIXTURE_H
