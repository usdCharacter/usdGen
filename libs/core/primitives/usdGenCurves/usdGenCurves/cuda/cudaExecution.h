#ifndef USDGEN_CUDA_EXECUTION_H
#define USDGEN_CUDA_EXECUTION_H
#include "usdGenCuda/backendId.h"
#include "usdGen/api/ext/device/deviceGeneration.h"
#include "usdGen/api/ext/device/executionPlan.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/op.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include "usdGenCurves/cuda/export.h"

namespace usdGen {
// CUDA backend admission is explicit and shared by compilation/execution.
// Unsupported operators/configurations are errors, never CPU fallbacks.
USDGEN_CURVES_CUDA_INTERNAL_API bool ValidateCudaGraph(UsdGenGraphSpec const&, UsdGenDiagnostics*);
class UsdGenCudaExecutionPlan;
class UsdGenCudaExecutionJob;
struct ExecutionState;
struct OperatorRelayService;
struct FinalizationRelayService;
// Immutable CPU programs, LUTs and authored input snapshot; no device state.
USDGEN_CURVES_CUDA_INTERNAL_API std::shared_ptr<const UsdGenCudaExecutionPlan> CompileCudaGraph(
    UsdGenGraphSpec const&, UsdGenDiagnostics*);
// Versioned backend capability declaration and the exact immutable metadata
// snapshot captured by a successfully compiled CUDA plan.
USDGEN_CURVES_CUDA_INTERNAL_API UsdGenExecutionCapabilityMatrix const& GetCudaExecutionCapabilityMatrix() noexcept;
USDGEN_CURVES_CUDA_INTERNAL_API std::shared_ptr<const UsdGenExecutionPlanMetadata> GetCudaExecutionPlanMetadata(
    UsdGenCudaExecutionPlan const&) noexcept;
struct UsdGenCudaBindingStats {
    UsdGenNodeKey key;
    uint64_t identity = 0, bindCount = 0, solveCount = 0;
    size_t sampleCount = 0;
};
// CUDA-free result contract for the test-only Width-DAG device-overlap probe.
// maxActive >= 2 proves that two bounded probes submitted on legal branch
// streams executed simultaneously. It does not claim arbitrary task fan-in,
// or that every Width kernel itself overlapped on every device.
enum class UsdGenExecutionOverlapWitnessStatus : uint8_t {
    Disabled, Armed, Submitted, Observed, Incomplete, Error
};
struct UsdGenExecutionOverlapWitnessSnapshot {
    static constexpr size_t kMaxTasks = 8;
    UsdGenExecutionOverlapWitnessStatus status =
        UsdGenExecutionOverlapWitnessStatus::Disabled;
    uint32_t expected = 0, claims = 0, arrivals = 0, maxActive = 0;
    std::array<uint32_t, kMaxTasks> taskIds{};
    std::array<uint32_t, kMaxTasks> laneIds{};
};
// Test/diagnostic observation for the narrow workspace-private WidthBlend
// CUDA Graph cache. It names no CUDA handle and does not make the cache part
// of the portable execution-plan ABI.
struct UsdGenCudaWidthBlendGraphStats {
    uint64_t captures = 0, replays = 0, misses = 0, evictions = 0, quarantines = 0;
    size_t cacheBytes = 0;
    bool inFlight = false, quarantined = false;
};
// One description's mutable execution resources. Calls must belong to its
// serial work node; different descriptions never share a workspace. Plans may
// be shared freely. The workspace owns an explicit nonblocking device stream.
class USDGEN_CURVES_CUDA_INTERNAL_API UsdGenCudaExecutionWorkspace {
public:
    ~UsdGenCudaExecutionWorkspace();
    UsdGenCudaExecutionWorkspace(UsdGenCudaExecutionWorkspace const&) = delete;
    UsdGenCudaExecutionWorkspace& operator=(UsdGenCudaExecutionWorkspace const&) = delete;
    int DeviceIndex() const noexcept;
    /// True after native completion or device-selection proof is lost. A
    /// poisoned workspace is permanently ineligible for execution or cache
    /// compatibility and may only be retired conservatively.
    bool IsPoisoned() const noexcept;
    /// Serialized backend-adapter boundary for an externally reported context
    /// loss. It performs no CUDA call and only makes future use fail closed.
    void MarkContextLost() noexcept;
private:
    UsdGenCudaExecutionWorkspace();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend USDGEN_CURVES_CUDA_INTERNAL_API std::unique_ptr<UsdGenCudaExecutionWorkspace> CreateCudaExecutionWorkspace(int, UsdGenDiagnostics*);
    friend USDGEN_CURVES_CUDA_INTERNAL_API std::vector<UsdGenCudaBindingStats> GetCudaBindingStats(UsdGenCudaExecutionWorkspace const&);
    friend USDGEN_CURVES_CUDA_INTERNAL_API UsdGenCudaWidthBlendGraphStats GetCudaWidthBlendGraphStats(
        UsdGenCudaExecutionWorkspace const&) noexcept;
    friend USDGEN_CURVES_CUDA_INTERNAL_API uintptr_t getCudaRestSurfaceBindingCacheIdentityForTesting(
        UsdGenCudaExecutionWorkspace const&) noexcept;
    friend struct ExecutionState;
    friend USDGEN_CURVES_CUDA_INTERNAL_API std::shared_ptr<UsdGenCudaExecutionJob> CreateCudaExecutionJob(
        std::shared_ptr<const UsdGenCudaExecutionPlan>, UsdGenCudaExecutionWorkspace&,
        double, uint64_t, UsdGenDiagnostics*, std::shared_ptr<const UsdGenDeviceGeneration>);
    friend USDGEN_CURVES_CUDA_INTERNAL_API bool ExecuteCudaJobSourceAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
                                          std::function<void(bool)>);
    friend USDGEN_CURVES_CUDA_INTERNAL_API bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob>, size_t,
                                            std::function<void(bool)>);
    friend USDGEN_CURVES_CUDA_INTERNAL_API bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
        std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)>);
    friend class UsdGenCudaExecutionJob;
    friend struct SourceRelayService;
    friend struct OperatorRelayService;
    friend struct FinalizationRelayService;
};
// device=-1 captures the caller's current CUDA device before dispatch. Every
// execution explicitly selects that device and restores the worker's previous
// device. Destruction follows draining, never overlaps execution.
USDGEN_CURVES_CUDA_INTERNAL_API std::unique_ptr<UsdGenCudaExecutionWorkspace> CreateCudaExecutionWorkspace(
    int device, UsdGenDiagnostics*);
// Immutable last-completed diagnostics; safe to query while the owner works.
USDGEN_CURVES_CUDA_INTERNAL_API std::vector<UsdGenCudaBindingStats> GetCudaBindingStats(UsdGenCudaExecutionWorkspace const&);
USDGEN_CURVES_CUDA_INTERNAL_API UsdGenCudaWidthBlendGraphStats GetCudaWidthBlendGraphStats(
    UsdGenCudaExecutionWorkspace const&) noexcept;
// Test-only identity observation for the workspace-local immutable COW binding
// cache.  It exposes no cache contents or mutable device state; zero means no
// cache is currently published in this workspace. The caller must keep the
// workspace quiescent; this observation is not synchronized with execution.
USDGEN_CURVES_CUDA_INTERNAL_API uintptr_t getCudaRestSurfaceBindingCacheIdentityForTesting(
    UsdGenCudaExecutionWorkspace const&) noexcept;
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaWidthBlendGraphCaptureForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API std::shared_ptr<UsdGenCudaExecutionJob> CreateCudaExecutionJob(
    std::shared_ptr<const UsdGenCudaExecutionPlan>, UsdGenCudaExecutionWorkspace&,
    double, uint64_t, UsdGenDiagnostics*, std::shared_ptr<const UsdGenDeviceGeneration> = {});
// A job retains its plan and previous publication, but borrows the workspace
// and diagnostics.  Drain all stage tasks before either borrowed object dies.
USDGEN_CURVES_CUDA_INTERNAL_API size_t CudaExecutionJobOperatorCount(UsdGenCudaExecutionJob const&) noexcept;
// Shared asynchronous CurveSource stage. The completion is invoked exactly
// once off the native CUDA callback after its retained source payload reaches
// a terminal success/failure state. A process-lifetime relay owns a fixed
// 1024 admission slots and uses up to eight dedicated worker threads; unsafe
// terminal failures permanently charge their slot rather than freeing CUDA
// staging without proof. Callers must keep their task completion admission in
// the supplied closure; false means no callback was accepted.
USDGEN_CURVES_CUDA_INTERNAL_API bool ExecuteCudaJobSourceAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
                               std::function<void(bool)> completion);
// Asynchronous Width/Length operator continuation.  Other operators run
// inline but obey the same accepted-completion contract.
USDGEN_CURVES_CUDA_INTERNAL_API bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob>, size_t,
                                 std::function<void(bool)> completion);
// Asynchronous terminal publication.  A `true` return retains the callback
// and invokes it exactly once; a clean pre-submission setup failure may report
// a null generation inline on the ordinary launcher.  A successful generation
// is delivered only after native tile/topology proof, launcher return, and
// relay-worker construction.  User completion never runs on a native CUDA
// callback thread.  `false` means admission was rejected and no callback was
// retained.
USDGEN_CURVES_CUDA_INTERNAL_API bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> completion);
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncCallbackGateForTesting();
// Test-only relay controls. Capacity is capped at the production 1024 slots;
// injected callback failures deliberately quarantine an admission.
USDGEN_CURVES_CUDA_INTERNAL_API void setCudaSourceRelayCapacityForTesting(size_t);
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaSourceRelayOccupiedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaSourceRelayQuarantinedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayNativeCallbackForTesting();
// Source-control proof phases precede the existing upload/resample phases.
// They are test-only controls; the production relay never waits on them.
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncScalarCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncScalarCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncScalarCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncScalarLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncScalarLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncScalarLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayContextCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayProgramCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayScalarCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayContextNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayProgramNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayScalarNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayControlsAllocationForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncResampleCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncResampleCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncResampleCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayResampleCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayResampleNativeCallbackForTesting();
void failNextCudaSourceRelayNamedCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaSourceRelayNamedNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaSourceAsyncResampleLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaSourceAsyncResampleLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaSourceAsyncResampleLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void setCudaOperatorRelayCapacityForTesting(size_t);
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaOperatorRelayOccupiedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaOperatorRelayQuarantinedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayContextCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayProgramCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayWidthCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayLengthCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayCountsCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayScatterCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayNamedTopologyCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayNonWidthCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayScatterAllocationForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayContextNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayProgramNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayWidthNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayLengthNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayCountsNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayScatterNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayNamedTopologyNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaOperatorRelayNonWidthNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncContextCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncProgramCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncWidthCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncWidthCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncWidthCallbackGateForTesting();
// Multi-claim Width-DAG launcher rendezvous. It is deliberately test-only:
// independently ready branch tasks select their streams and arrive before
// release, without relying on CUDA callback concurrency or wall-clock timing.
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncWidthBranchGateForTesting(size_t expected);
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncWidthBranchGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncWidthBranchGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting();
// Arms a bounded device-side overlap probe for up to eight legal Width-DAG
// branches. `dwellCycles` is clamped to a finite device-clock duration. The
// dormant production path submits no probe, waits, or extra CUDA work.
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncWidthDeviceOverlapWitnessForTesting(
    size_t expected, uint64_t dwellCycles = 50000000);
// Test observation only: performs a synchronous D2H counter read, so call it
// after the queue/job has drained. It is never invoked by a CUDA callback.
USDGEN_CURVES_CUDA_INTERNAL_API UsdGenExecutionOverlapWitnessSnapshot
cudaOperatorAsyncWidthDeviceOverlapWitnessSnapshotForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncLengthCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncLengthCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncLengthCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncCountsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncCountsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncCountsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncScatterCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncScatterCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncScatterCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaOperatorAsyncWidthLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaOperatorAsyncWidthLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaOperatorAsyncWidthLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void setCudaFinalizationRelayCapacityForTesting(size_t);
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaFinalizationRelayOccupiedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API size_t cudaFinalizationRelayQuarantinedCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayTileCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayBoundsCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayMetadataCallbackInstallForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayAllocationForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayTopologyAllocationForTesting();
// RBF transaction seams are routed through usdGen so an integrated test
// reaches the same statically linked GPU implementation as the executor.
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfFieldResolvePreflightForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfFieldResolveCommitForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfSolveAfterInputSubmitForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfSurfaceResolvePreflightForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfSurfaceResolveCommitForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API uint64_t cudaRbfFieldAcceptAttemptCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API uint64_t cudaRbfFieldRollbackAttemptCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API uint64_t cudaRbfSurfaceAcceptAttemptCountForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API uint64_t cudaRbfSurfaceRollbackAttemptCountForTesting();
// Fails the next finalization that holds a pending RBF transaction.
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaRbfFinalizationForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayTileNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayBoundsNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void failNextCudaFinalizationRelayMetadataNativeCallbackForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaFinalizationAsyncTileCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaFinalizationAsyncTileCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaFinalizationAsyncTileCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaFinalizationAsyncBoundsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaFinalizationAsyncBoundsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaFinalizationAsyncBoundsCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaFinalizationAsyncMetadataCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaFinalizationAsyncMetadataCallbackGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaFinalizationAsyncMetadataCallbackGateForTesting();
// Defers only finalization phase-2's Return signal; it never blocks a relay
// worker while a test observes native-terminal-before-return ordering.
USDGEN_CURVES_CUDA_INTERNAL_API void armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
USDGEN_CURVES_CUDA_INTERNAL_API void releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();

}
#endif
