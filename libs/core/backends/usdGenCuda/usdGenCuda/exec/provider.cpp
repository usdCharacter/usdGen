// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CUDA Session provider: the staged source -> operator* -> publication
// task graph of a registered device program, behind the provider boundary.
// The cooker owns compilation, the cache and publication.
#include "usdGenCuda/exec/provider.h"
#include "usdGenCuda/backendId.h"
#include "usdGenCuda/compile/program.h"
#include "usdGenCuda/exec/workspace.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/device/executionTaskGraph.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/typeRegistry.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace usdGen {
namespace {

std::exception_ptr Failure(std::string const &what)
{
    return std::make_exception_ptr(std::runtime_error(what));
}

// The payload of the handles this provider hands out: the program and its plan.
struct CompiledProgram {
    cuda::ProgramDef const *program = nullptr;
    std::shared_ptr<const UsdGenExecutionPlanHandle> plan;
};

class CudaSessionProvider final : public UsdGenSessionDeviceProvider {
public:
    // One cook's state, shared by its task closures until the graph completes.
    struct Stages {
        UsdGenRef<cuda::Job> job;
        UsdGenSessionDeviceReturn transport;
        UsdGenSessionDeviceIdentity identity;
        std::shared_ptr<const UsdGenDeviceGeneration> generation;
        UsdGenRef<const UsdGenPublication> report;
        std::atomic<bool> returned{false};
    };

    // One started task's completion; owned by the TaskDone it is handed to.
    struct TaskRelay {
        UsdGenExecutionTaskGraph::TaskCompletion completion;
        std::shared_ptr<Stages> stages;   // set on the publication task only
        std::string failure;              // what a failed task reports

        static void Done(void *user, bool ok) noexcept
        {
            std::unique_ptr<TaskRelay> self(static_cast<TaskRelay *>(user));
            try {
                if (ok && self->stages) {
                    // Native diagnostic snapshot, before the result leaves.
                    auto device = self->stages->job->Result();
                    if (device) {
                        self->stages->report = self->stages->job->Report();
                        self->stages->generation = std::move(device);
                        self->completion([] {}, {});
                        return;
                    }
                    ok = false;
                }
                if (ok) self->completion({}, {});
                else self->completion({}, Failure(self->failure));
            } catch (...) {
                try { self->completion({}, std::current_exception()); } catch (...) {}
            }
        }
    };

    ~CudaSessionProvider() override = default;

    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const &desc,
                                          UsdGenDiagnostics *diagnostics) const override
    {
        UsdGenDiagnostics local;
        UsdGenDiagnostics *const diag = diagnostics ? diagnostics : &local;
        cuda::ProgramDef const *const program = cuda::ProgramFor(desc);
        if (!program || !program->compile || !program->createJob) {
            diag->Error("CUDA: no device program is registered for the graph's primitive");
            return {};
        }
        auto plan = program->compile(desc, diag);
        if (!plan) return {};
        std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata = plan->Metadata();
        UsdGenBackendId const backend = plan->Backend();
        std::string reason;
        auto handle = UsdGenExecutionPlanHandle::Create(backend, std::move(metadata),
            std::make_shared<const CompiledProgram>(CompiledProgram{program, std::move(plan)}),
            &reason);
        if (!handle) diag->Error("CUDA plan handle: " + reason);
        return handle;
    }

    UsdGenSessionDeviceIdentity Identity() const noexcept override
    {
        UsdGenSessionDeviceIdentity identity;
        identity.backend = UsdGenCudaBackendId();
        identity.deviceIndex = _workspace ? _workspace->DeviceIndex() : -1;
        UsdGenBackendDef const *const def = UsdGenGetBackend(identity.backend);
        identity.capabilityVersion = def ? def->capabilityVersion : 0;
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
            auto replacement = cuda::Workspace::Create(callerDevice, diagnostics);
            if (!replacement || !_MakeStates(*replacement, diagnostics)) return false;
            _workspace = std::move(replacement);
            _description = description;
        }
        if (identity) *identity = Identity();
        return true;
    }

    bool Submit(UsdGenSessionDeviceRequest request) override
    {
        UsdGenDiagnostics *const diagnostics = request.diagnostics;
        auto const *const compiled = request.plan
            ? static_cast<CompiledProgram const *>(request.plan->Payload().get()) : nullptr;
        if (!compiled || !compiled->program || !compiled->plan || !_workspace ||
            !request.runtime || !request.returnTransport)
            return false;
        cuda::ProgramState *const state = _workspace->State(*compiled->program, diagnostics);
        if (!state) return false;
        auto stages = std::make_shared<Stages>();
        stages->transport = std::move(request.returnTransport);
        stages->identity = Identity();
        // A fresh workspace has no incremental baseline.
        std::shared_ptr<const UsdGenDeviceGeneration> const none;
        stages->job = compiled->program->createJob(*compiled->plan, *_workspace, *state, request,
                                                   _fresh ? none : request.previous, diagnostics);
        if (!stages->job) {
            if (diagnostics) diagnostics->Error("CUDA execution job creation failed");
            return false;
        }
        UsdGenExecutionPlanMetadata const *const metadata = stages->job->Metadata();
        size_t const operators = stages->job->OperatorCount();
        if (!metadata ||
            (metadata->Shape() != UsdGenExecutionPlanShape::LinearAuthoredChain &&
             metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedUnaryDag &&
             metadata->Shape() != UsdGenExecutionPlanShape::SourceRootedValueDag) ||
            metadata->Operators().empty() ||
            metadata->Tasks().size() != operators + 2 ||
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
            if (taskCancel.Superseded()) {
                done({}, Failure("CUDA source stage failed"));
                return;
            }
            CudaSessionProvider::_Start(stages, 0, std::move(done), "CUDA source stage failed",
                   "CUDA source stage failed", false);
        }});
        for (size_t i = 0; i < operators; ++i) {
            auto const *operatorMetadata = metadata->FindTask(static_cast<uint32_t>(i + 1));
            if (!operatorMetadata || operatorMetadata->kind != UsdGenExecutionTaskKind::Operator ||
                operatorMetadata->id != graph.tasks.size())
                return _Fail(stages, "CUDA operator task metadata is missing");
            graph.tasks.push_back({operatorMetadata->dependencies,
                                   [stages, i](auto const &taskCancel, auto done) {
                std::string const stage = "CUDA operator stage " + std::to_string(i);
                if (taskCancel.Superseded()) {
                    done({}, Failure(stage + " superseded"));
                    return;
                }
                CudaSessionProvider::_Start(stages, static_cast<uint32_t>(i + 1), std::move(done),
                       stage + " completion failed", stage + " submission failed", false);
            }});
        }
        auto const *terminalMetadata = metadata->FindTask(metadata->TerminalTask());
        if (!terminalMetadata || terminalMetadata->kind != UsdGenExecutionTaskKind::Publication ||
            terminalMetadata->id != graph.tasks.size())
            return _Fail(stages, "CUDA publication task metadata is missing");
        uint32_t const terminal = terminalMetadata->id;
        graph.tasks.push_back({terminalMetadata->dependencies,
                               [stages, terminal](auto const &, auto done) {
            CudaSessionProvider::_Start(stages, terminal, std::move(done), "CUDA final stage failed",
                   "CUDA final stage failed", true);
        }});
        graph.finalTask = terminalMetadata->id;
        graph.completion = [this, stages](auto, std::exception_ptr error) {
            // Task completion proves no dispatcher closure can still borrow
            // the workspace. Release the job first: its destructor can be the
            // operation that marks an unprovable transaction poisoned.
            stages->job.Reset();
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
    // Starts one task; its relay reports to 'done' exactly once.
    static void _Start(std::shared_ptr<Stages> const &stages, uint32_t task,
                       UsdGenExecutionTaskGraph::TaskCompletion done, std::string failed,
                       std::string const &rejected, bool publication)
    {
        std::unique_ptr<TaskRelay> relay;
        try {
            relay.reset(new TaskRelay{std::move(done), publication ? stages : nullptr,
                                      std::move(failed)});
        } catch (...) {
            done({}, std::current_exception());
            return;
        }
        cuda::TaskDone handoff;
        handoff.fn = &TaskRelay::Done;
        handoff.user = relay.get();
        if (stages->job->Start(task, handoff)) {
            (void)relay.release();
            return;
        }
        relay->completion({}, Failure(rejected));
    }

    // One state per registered program, made with the workspace so a
    // device that cannot host them fails the selection, not the cook.
    static bool _MakeStates(cuda::Workspace &workspace, UsdGenDiagnostics *diagnostics)
    {
        UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
        for (uint32_t t = 0; t < types.Count(); ++t) {
            cuda::ProgramDef const *const program = cuda::ProgramOf(UsdGenTypeId(t));
            if (program && !workspace.State(*program, diagnostics)) return false;
        }
        return true;
    }

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

    std::unique_ptr<cuda::Workspace> _workspace;
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
