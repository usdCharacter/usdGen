// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' CUDA device program over the staged executor (cudaExecution,
// still in usdGenCuda until its kernels move here).
#include "usdGenCurves/cuda/program.h"
#include "usdGenCuda/backendId.h"
#include "usdGenCurves/cuda/bindingReport.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGenCuda/exec/workspace.h"
#include "usdGen/api/ext/op.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {
namespace {

// The executor workspace: streams, binding and graph caches of one device.
class CurvesState final : public cuda::ProgramState {
public:
    explicit CurvesState(std::unique_ptr<UsdGenCudaExecutionWorkspace> workspace) noexcept
        : workspace(std::move(workspace)) {}
    bool Poisoned() const noexcept override { return workspace->IsPoisoned(); }
    void MarkContextLost() noexcept override { workspace->MarkContextLost(); }
    std::unique_ptr<UsdGenCudaExecutionWorkspace> const workspace;
};

class CurvesJob final : public cuda::Job {
public:
    CurvesJob(std::shared_ptr<UsdGenCudaExecutionJob> job,
              std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata,
              CurvesState const &state) noexcept
        : _job(std::move(job)), _metadata(std::move(metadata)), _state(state) {}

    UsdGenExecutionPlanMetadata const *Metadata() const noexcept override
    {
        return _metadata.get();
    }
    size_t OperatorCount() const noexcept override
    {
        return CudaExecutionJobOperatorCount(*_job);
    }
    bool Start(uint32_t task, cuda::TaskDone done) noexcept override
    {
        try {
            auto relay = [done](bool ok) { done.fn(done.user, ok); };
            size_t const operators = OperatorCount();
            if (task == 0) return ExecuteCudaJobSourceAsync(_job, relay);
            if (task <= operators) return ExecuteCudaJobOperatorAsync(_job, task - 1, relay);
            if (task != operators + 1) return false;
            // The job outlives this callback: the runtime releases it only
            // after the publication task completed. Nothing here runs after done.
            return FinalizeCudaExecutionJobAsync(_job,
                [this, done](std::shared_ptr<const UsdGenDeviceGeneration> device) {
                    bool ok = bool(device);
                    if (ok) {
                        try {
                            auto report = UsdGenMakeRef<UsdGenCudaBindingReport>();
                            report->stats = GetCudaBindingStats(*_state.workspace);
                            _report = std::move(report);
                            _result = std::move(device);
                        } catch (...) { ok = false; }
                    }
                    done.fn(done.user, ok);
                });
        } catch (...) {
            return false;
        }
    }
    std::shared_ptr<const UsdGenDeviceGeneration> Result() const noexcept override
    {
        return _result;
    }
    UsdGenRef<const UsdGenPublication> Report() const noexcept override { return _report; }

private:
    std::shared_ptr<UsdGenCudaExecutionJob> _job;
    std::shared_ptr<const UsdGenExecutionPlanMetadata> _metadata;
    CurvesState const &_state;
    std::shared_ptr<const UsdGenDeviceGeneration> _result;
    UsdGenRef<const UsdGenPublication> _report;
};

std::shared_ptr<const UsdGenExecutionPlanHandle> Compile(UsdGenGraphSpec const &desc,
                                                          UsdGenDiagnostics *diagnostics)
{
    UsdGenDiagnostics local;
    UsdGenDiagnostics *const diag = diagnostics ? diagnostics : &local;
    size_t const knownErrors = diag->errors.size();
    auto plan = CompileCudaGraph(desc, diag);
    if (!plan) {
        if (diag->errors.size() == knownErrors) diag->Error("CUDA plan compilation failed");
        return {};
    }
    // A plan without metadata still compiles; the runtime rejects it at Submit.
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

std::unique_ptr<cuda::ProgramState> CreateState(cuda::Workspace &workspace,
                                                UsdGenDiagnostics *diagnostics)
{
#if defined(USDGEN_ENABLE_CUDA)
    auto executor = CreateCudaExecutionWorkspace(workspace.DeviceIndex(), diagnostics);
    if (!executor) return {};
    return std::unique_ptr<cuda::ProgramState>(new CurvesState(std::move(executor)));
#else
    // Without the toolkit usdGenCuda builds no workspaces (exec/workspace.cpp
    // is CUDA-only), so no provider reaches this; it must not name
    // Workspace's members either: MSVC imports the inline DeviceIndex from a
    // usdGenCuda.dll that never instantiated it.
    (void)workspace;
    if (diagnostics) diagnostics->Error("CUDA: usdGen was built without CUDA");
    return {};
#endif
}

UsdGenRef<cuda::Job> CreateJob(UsdGenExecutionPlanHandle const &handle, cuda::Workspace &,
                               cuda::ProgramState &state, UsdGenSessionDeviceRequest const &request,
                               std::shared_ptr<const UsdGenDeviceGeneration> const &previous,
                               UsdGenDiagnostics *diagnostics)
{
    auto plan = std::static_pointer_cast<const UsdGenCudaExecutionPlan>(handle.Payload());
    if (!plan) return {};
    // The runtime hands back the state this program's createState made.
    auto &curves = static_cast<CurvesState &>(state);
    auto job = CreateCudaExecutionJob(plan, *curves.workspace,
        request.evaluationFrame ? *request.evaluationFrame : 0.0,
        request.publicationGeneration, diagnostics, previous);
    if (!job) return {};
    return UsdGenMakeRef<CurvesJob>(std::move(job), GetCudaExecutionPlanMetadata(*plan), curves);
}

// Constant-initialized: registered by address, read after the freeze.
cuda::ProgramDef const gProgram{&Compile, &ValidateCudaGraph, &CreateState, &CreateJob};

}  // namespace

cuda::ProgramDef const &UsdGenCurvesCudaProgram() noexcept
{
    return gProgram;
}

}  // namespace usdGen
