// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CUDA Session provider: the staged source -> operator* -> final task
// graph that ran inside the cooker until step 8.3, behind the provider
// boundary. The cooker owns compilation, the cache and publication.
#include "usdGenCuda/cudaSessionProvider.h"
#include "usdGenCuda/backendId.h"
#include "usdGenCuda/bindingReport.h"
#include "usdGenCuda/cudaExecution.h"
#include "usdGen/exec/executionTaskGraph.h"
#include "usdGen/registries/backend.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {
namespace {

std::exception_ptr Failure(std::string const &what)
{
    return std::make_exception_ptr(std::runtime_error(what));
}

class CudaSessionProvider final : public UsdGenSessionDeviceProvider {
public:
    // One cook's state, shared by its task closures until the graph completes.
    struct Stages {
        std::shared_ptr<UsdGenCudaExecutionJob> job;
        UsdGenSessionDeviceReturn transport;
        UsdGenSessionDeviceIdentity identity;
        std::shared_ptr<const UsdGenDeviceGeneration> generation;
        UsdGenRef<const UsdGenPublication> report;
        std::atomic<bool> returned{false};
    };

    ~CudaSessionProvider() override = default;

    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const &desc,
                                          UsdGenDiagnostics *diagnostics) const override
    {
        UsdGenDiagnostics local;
        UsdGenDiagnostics *const diag = diagnostics ? diagnostics : &local;
        size_t const knownErrors = diag->errors.size();
        auto plan = CompileCudaGraph(desc, diag);
        if (!plan) {
            if (diag->errors.size() == knownErrors) diag->Error("CUDA plan compilation failed");
            return {};
        }
        // A plan without metadata still compiles; Submit rejects it.
        auto metadata = GetCudaExecutionPlanMetadata(*plan);
        if (!metadata)
            metadata = std::make_shared<const UsdGenExecutionPlanMetadata>(
                "cuda", kUsdGenCudaCapabilityVersion,
                UsdGenExecutionPlanShape::LinearAuthoredChain,
                std::vector<UsdGenCompiledOperatorCapability>{},
                std::vector<UsdGenExecutionTaskMetadata>{},
                std::vector<UsdGenExecutionValueMetadata>{}, 0);
        std::string reason;
        auto handle = UsdGenExecutionPlanHandle::Create(
            UsdGenCudaBackendId(), std::move(metadata), std::move(plan), &reason);
        if (!handle) diag->Error("CUDA plan handle: " + reason);
        return handle;
    }

    UsdGenSessionDeviceIdentity Identity() const noexcept override
    {
        UsdGenSessionDeviceIdentity identity;
        identity.backend = UsdGenCudaBackendId();
        identity.deviceIndex = _workspace ? _workspace->DeviceIndex() : -1;
        identity.capabilityVersion = kUsdGenCudaCapabilityVersion;
        return identity;
    }

    // Quarantines the workspace of the lost device; the cooker invalidates
    // the cache domain. Called on the cook lane between cooks.
    bool NotifyContextLost(UsdGenSessionDeviceIdentity const &lost) noexcept override
    {
        if (!_workspace || lost.backend != UsdGenCudaBackendId() ||
            (lost.deviceIndex >= 0 && lost.deviceIndex != _workspace->DeviceIndex()))
            return false;
        _workspace->MarkContextLost();
        _Drop();
        return true;
    }

    bool Select(int callerDevice, UsdGenNodeKey description,
                UsdGenSessionDeviceIdentity *identity, UsdGenDiagnostics *diagnostics) override
    {
        _fresh = !_workspace || _description != description ||
            (callerDevice >= 0 && _workspace->DeviceIndex() != callerDevice);
        if (_fresh) {
            auto replacement = CreateCudaExecutionWorkspace(callerDevice, diagnostics);
            if (!replacement) return false;
            _workspace = std::move(replacement);
            _description = description;
        }
        if (identity) *identity = Identity();
        return true;
    }

    bool Submit(UsdGenSessionDeviceRequest request) override
    {
        UsdGenDiagnostics *const diagnostics = request.diagnostics;
        auto const plan = request.plan
            ? std::static_pointer_cast<const UsdGenCudaExecutionPlan>(request.plan->Payload())
            : std::shared_ptr<const UsdGenCudaExecutionPlan>{};
        if (!plan || !_workspace || !request.runtime || !request.returnTransport) return false;
        auto stages = std::make_shared<Stages>();
        stages->transport = std::move(request.returnTransport);
        stages->identity = Identity();
        // A fresh workspace has no incremental baseline.
        stages->job = CreateCudaExecutionJob(plan, *_workspace,
            request.evaluationFrame ? *request.evaluationFrame : 0.0,
            request.publicationGeneration, diagnostics,
            _fresh ? nullptr : request.previous);
        if (!stages->job) {
            if (diagnostics) diagnostics->Error("CUDA execution job creation failed");
            return false;
        }
        auto metadata = GetCudaExecutionPlanMetadata(*plan);
        if (!metadata ||
            (metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
             metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
             metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedValueDag) ||
            metadata->Operators().empty() ||
            metadata->Tasks().size() != CudaExecutionJobOperatorCount(*stages->job) + 2 ||
            !metadata->FindTask(metadata->TerminalTask()))
            return _Fail(stages, "CUDA execution plan metadata is missing or incompatible");
        for (uint32_t i = 0; i != metadata->Tasks().size(); ++i)
            if (!metadata->FindTask(i))
                return _Fail(stages, "CUDA execution plan task ids are not dense");
        std::string dependencyReason;
        if (!UsdGenExecutionDependencyCompiler::Validate(
                metadata->Tasks(), metadata->Values(), &dependencyReason))
            return _Fail(stages, "CUDA execution plan dependencies are invalid: " + dependencyReason);
        auto dispatcher = UsdGenExecutionTaskGraph::GetOrCreate(
            *request.runtime, "cuda", _workspace->DeviceIndex());
        if (!dispatcher) return _Fail(stages, "CUDA dispatcher unavailable");
        UsdGenExecutionTaskGraph::Job graph;
        graph.cancellation = request.cancellation;
        auto const *sourceMetadata = metadata->FindTask(0);
        if (!sourceMetadata || sourceMetadata->kind != UsdGenExecutionTaskKind::Source ||
            sourceMetadata->id != graph.tasks.size())
            return _Fail(stages, "CUDA source task metadata is missing");
        graph.tasks.push_back({sourceMetadata->dependencies, [stages](auto const &taskCancel, auto done) {
            auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                std::move(done));
            try {
                if (taskCancel.Superseded()) {
                    (*completion)({}, Failure("CUDA source stage failed"));
                    return;
                }
                if (!ExecuteCudaJobSourceAsync(stages->job, [completion](bool ok) {
                        if (ok) (*completion)({}, {});
                        else (*completion)({}, Failure("CUDA source stage failed"));
                    }))
                    (*completion)({}, Failure("CUDA source stage failed"));
            } catch (...) { (*completion)({}, std::current_exception()); }
        }});
        size_t const operators = CudaExecutionJobOperatorCount(*stages->job);
        for (size_t i = 0; i < operators; ++i) {
            auto const *operatorMetadata = metadata->FindTask(static_cast<uint32_t>(i + 1));
            if (!operatorMetadata || operatorMetadata->kind != UsdGenExecutionTaskKind::Operator ||
                operatorMetadata->id != graph.tasks.size())
                return _Fail(stages, "CUDA operator task metadata is missing");
            graph.tasks.push_back({operatorMetadata->dependencies,
                                   [stages, i](auto const &taskCancel, auto done) {
                auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                    std::move(done));
                try {
                    if (taskCancel.Superseded()) {
                        (*completion)({}, Failure("CUDA operator stage " + std::to_string(i) + " superseded"));
                        return;
                    }
                    if (!ExecuteCudaJobOperatorAsync(stages->job, i, [completion, i](bool ok) {
                            if (ok) (*completion)({}, {});
                            else (*completion)({}, Failure("CUDA operator stage " +
                                                           std::to_string(i) + " completion failed"));
                        }))
                        (*completion)({}, Failure("CUDA operator stage " + std::to_string(i) +
                                                  " submission failed"));
                } catch (...) { (*completion)({}, std::current_exception()); }
            }});
        }
        auto const *terminalMetadata = metadata->FindTask(metadata->TerminalTask());
        if (!terminalMetadata || terminalMetadata->kind != UsdGenExecutionTaskKind::Publication ||
            terminalMetadata->id != graph.tasks.size())
            return _Fail(stages, "CUDA publication task metadata is missing");
        UsdGenCudaExecutionWorkspace const *const workspace = _workspace.get();
        graph.tasks.push_back({terminalMetadata->dependencies,
                               [stages, workspace](auto const &, auto done) {
            auto completion = std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(
                std::move(done));
            try {
                if (!FinalizeCudaExecutionJobAsync(stages->job,
                        [stages, workspace, completion](
                            std::shared_ptr<const UsdGenDeviceGeneration> device) {
                    try {
                        if (!device) {
                            (*completion)({}, Failure("CUDA final stage failed"));
                            return;
                        }
                        // Native diagnostic snapshot, before the result leaves.
                        auto report = UsdGenMakeRef<UsdGenCudaBindingReport>();
                        report->stats = GetCudaBindingStats(*workspace);
                        stages->report = std::move(report);
                        stages->generation = std::move(device);
                        (*completion)([] {}, {});
                    } catch (...) { (*completion)({}, std::current_exception()); }
                }))
                    (*completion)({}, Failure("CUDA final stage failed"));
            } catch (...) { (*completion)({}, std::current_exception()); }
        }});
        graph.finalTask = terminalMetadata->id;
        graph.completion = [this, stages](auto, std::exception_ptr error) {
            // Task completion proves no dispatcher closure can still borrow
            // the workspace. Destroy the job first: its destructor can be the
            // operation that marks an unprovable transaction poisoned.
            stages->job.reset();
            bool const poisoned = _workspace && _workspace->IsPoisoned();
            if (poisoned) _Drop();
            _Return(stages, error, poisoned);
        };
        if (!dispatcher->Submit(std::move(graph)))
            _Return(stages, Failure("CUDA dispatcher rejected job"), false);
        return true;
    }

    void Shutdown() noexcept override { _Drop(); }

private:
    void _Drop() noexcept
    {
        _workspace.reset();
        _description = UsdGenNodeKey();
    }

    // Sends a synchronous admission failure as the cook's error.
    bool _Fail(std::shared_ptr<Stages> const &stages, std::string const &what)
    {
        _Return(stages, Failure(what), false);
        return true;
    }

    static void _Return(std::shared_ptr<Stages> const &stages, std::exception_ptr error,
                        bool poisoned) noexcept
    {
        if (stages->returned.exchange(true, std::memory_order_acq_rel)) return;
        try {
            auto result = std::make_shared<UsdGenSessionDeviceResult>();
            result->identity = stages->identity;
            result->error = error;
            result->poisoned = poisoned;
            if (!error) {
                result->generation = std::move(stages->generation);
                result->report = std::move(stages->report);
            }
            auto transport = std::move(stages->transport);
            (void)transport(std::move(result));
        } catch (...) {}
    }

    std::unique_ptr<UsdGenCudaExecutionWorkspace> _workspace;
    UsdGenNodeKey _description;
    bool _fresh = true;   // the selected workspace was created for this cook
};

}  // namespace

std::shared_ptr<UsdGenSessionDeviceProvider> CreateCudaSessionProvider(
    UsdGenBackendProviderDesc const &, std::string *reason)
{
    try {
        return std::make_shared<CudaSessionProvider>();
    } catch (...) {
        if (reason) *reason = "CUDA Session provider allocation failed";
        return nullptr;
    }
}

int CaptureCudaCallerDevice() noexcept
{
    int device = -2;
    if (cudaGetDevice(&device) != cudaSuccess) device = -2;
    return device;
}

}  // namespace usdGen
