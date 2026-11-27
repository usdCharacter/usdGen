// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The staged program a primitive registers for the CUDA runtime through
// RegisterDeviceProgram(cuda, type, &def). The runtime owns the device,
// the workspace, the task graph and the completion transport; the program
// owns lowering and kernels.
#pragma once
#include "usdGenCuda/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/backend.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/ext/device/sessionDeviceProvider.h"
#include "usdGen/api/ext/publisher.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace usdGen {
struct UsdGenDiagnostics;
struct UsdGenGraphSpec;
}  // namespace usdGen

namespace usdGen::cuda {

class Workspace;   // usdGenCuda/exec/workspace.h

/// Runs once per started task, off the native callback thread, after the
/// task's device work. ok false: the task failed.
struct TaskDone {
    void (*fn)(void *user, bool ok) noexcept = nullptr;
    void *user = nullptr;
};

/// One cook of a compiled plan on one workspace.
class USDGEN_CUDA_API Job : public UsdGenRefCounted, public UsdGenPooled {
public:
    ~Job() override;
    /// Dense task ids: 0 source, 1..OperatorCount() operators, then
    /// publication. Null when the plan carries none.
    virtual UsdGenExecutionPlanMetadata const *Metadata() const noexcept = 0;
    virtual size_t OperatorCount() const noexcept = 0;
    /// Enqueues the task and returns at once. False: nothing enqueued and
    /// done never runs.
    virtual bool Start(uint32_t task, TaskDone done) noexcept = 0;
    /// After the publication task completed; null on failure.
    virtual std::shared_ptr<const UsdGenDeviceGeneration> Result() const noexcept = 0;
    /// Diagnostics for tools and tests, taken when publication completed.
    virtual UsdGenRef<const UsdGenPublication> Report() const noexcept;
};

/// Per-workspace program state (caches, native streams, statistics). Dies
/// with its workspace.
class USDGEN_CUDA_API ProgramState : public UsdGenPooled {
public:
    virtual ~ProgramState();
    /// The last cook left the state unprovable; the workspace is dropped.
    virtual bool Poisoned() const noexcept { return false; }
    /// The device context was reported lost; later cooks fail closed.
    virtual void MarkContextLost() noexcept {}
};

struct ProgramDef;

/// The program registered for (cuda, type); null if none.
USDGEN_CUDA_API ProgramDef const *ProgramOf(UsdGenTypeId type) noexcept;
/// The program of the graph's terminal type; the default primitive's when
/// the terminal names none or its type has no program.
USDGEN_CUDA_API ProgramDef const *ProgramFor(UsdGenGraphSpec const &spec) noexcept;

struct ProgramDef {
    /// Null with diagnostics on refusal. The handle's backend is "cuda".
    std::shared_ptr<const UsdGenExecutionPlanHandle> (*compile)(
        UsdGenGraphSpec const &, UsdGenDiagnostics *) = nullptr;
    /// Compile-time admission of a graph.
    bool (*validate)(UsdGenGraphSpec const &, UsdGenDiagnostics *) = nullptr;
    /// Null with diagnostics on failure. Runs on the cook lane.
    std::unique_ptr<ProgramState> (*createState)(Workspace &, UsdGenDiagnostics *) = nullptr;
    /// previous is null on a fresh workspace. Null with diagnostics on failure.
    UsdGenRef<Job> (*createJob)(UsdGenExecutionPlanHandle const &, Workspace &, ProgramState &,
                                UsdGenSessionDeviceRequest const &,
                                std::shared_ptr<const UsdGenDeviceGeneration> const &previous,
                                UsdGenDiagnostics *) = nullptr;
};

}  // namespace usdGen::cuda
