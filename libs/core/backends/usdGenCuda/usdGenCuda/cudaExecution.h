#ifndef USDGEN_CUDA_EXECUTION_H
#define USDGEN_CUDA_EXECUTION_H
#include "usdGen/deviceGeneration.h"
#include "usdGen/executionPlan.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>

namespace usdGen {
// CUDA backend admission is explicit and shared by compilation/execution.
// Unsupported operators/configurations are errors, never CPU fallbacks.
bool ValidateCudaGraph(UsdGenGraphDesc const&, UsdGenDiagnostics*);
class UsdGenCudaExecutionPlan;
class UsdGenCudaExecutionJob;
struct ExecutionState;
struct OperatorRelayService;
struct FinalizationRelayService;
// Immutable CPU programs, LUTs and authored input snapshot; no device state.
std::shared_ptr<const UsdGenCudaExecutionPlan> CompileCudaGraph(
    UsdGenGraphDesc const&, UsdGenDiagnostics*);
// Versioned backend capability declaration and the exact immutable metadata
// snapshot captured by a successfully compiled CUDA plan.
UsdGenExecutionCapabilityMatrix const& GetCudaExecutionCapabilityMatrix() noexcept;
std::shared_ptr<const UsdGenExecutionPlanMetadata> GetCudaExecutionPlanMetadata(
    UsdGenCudaExecutionPlan const&) noexcept;
struct UsdGenCudaBindingStats {
    SdfPath path;
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
class UsdGenCudaExecutionWorkspace {
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
    friend std::unique_ptr<UsdGenCudaExecutionWorkspace> CreateCudaExecutionWorkspace(int, UsdGenDiagnostics*);
    friend std::vector<UsdGenCudaBindingStats> GetCudaBindingStats(UsdGenCudaExecutionWorkspace const&);
    friend UsdGenCudaWidthBlendGraphStats GetCudaWidthBlendGraphStats(
        UsdGenCudaExecutionWorkspace const&) noexcept;
    friend uintptr_t getCudaRestSurfaceBindingCacheIdentityForTesting(
        UsdGenCudaExecutionWorkspace const&) noexcept;
    friend std::shared_ptr<const UsdGenDeviceGeneration> ExecuteCudaGraph(
        UsdGenCudaExecutionPlan const&, UsdGenCudaExecutionWorkspace&, double,
        uint64_t, UsdGenDiagnostics*, std::shared_ptr<const UsdGenDeviceGeneration> const&);
    friend struct ExecutionState;
    friend std::shared_ptr<UsdGenCudaExecutionJob> CreateCudaExecutionJob(
        std::shared_ptr<const UsdGenCudaExecutionPlan>, UsdGenCudaExecutionWorkspace&,
        double, uint64_t, UsdGenDiagnostics*, std::shared_ptr<const UsdGenDeviceGeneration>);
    friend bool ExecuteCudaJobSource(UsdGenCudaExecutionJob&);
    friend bool ExecuteCudaJobSourceAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
                                          std::function<void(bool)>);
    friend bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob>, size_t,
                                            std::function<void(bool)>);
    friend bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
        std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)>);
    friend std::shared_ptr<const UsdGenDeviceGeneration> FinalizeCudaExecutionJob(
        UsdGenCudaExecutionJob&);
    friend class UsdGenCudaExecutionJob;
    friend struct SourceRelayService;
    friend struct OperatorRelayService;
    friend struct FinalizationRelayService;
};
// device=-1 captures the caller's current CUDA device before dispatch. Every
// execution explicitly selects that device and restores the worker's previous
// device. Destruction follows draining, never overlaps execution.
std::unique_ptr<UsdGenCudaExecutionWorkspace> CreateCudaExecutionWorkspace(
    int device, UsdGenDiagnostics*);
// Immutable last-completed diagnostics; safe to query while the owner works.
std::vector<UsdGenCudaBindingStats> GetCudaBindingStats(UsdGenCudaExecutionWorkspace const&);
UsdGenCudaWidthBlendGraphStats GetCudaWidthBlendGraphStats(
    UsdGenCudaExecutionWorkspace const&) noexcept;
// Test-only identity observation for the workspace-local immutable COW binding
// cache.  It exposes no cache contents or mutable device state; zero means no
// cache is currently published in this workspace. The caller must keep the
// workspace quiescent; this observation is not synchronized with execution.
uintptr_t getCudaRestSurfaceBindingCacheIdentityForTesting(
    UsdGenCudaExecutionWorkspace const&) noexcept;
void failNextCudaWidthBlendGraphCaptureForTesting();
std::shared_ptr<const UsdGenDeviceGeneration> ExecuteCudaGraph(
    UsdGenCudaExecutionPlan const&, UsdGenCudaExecutionWorkspace&, double frame,
    uint64_t generation, UsdGenDiagnostics*,
    std::shared_ptr<const UsdGenDeviceGeneration> const& previous = {});
std::shared_ptr<UsdGenCudaExecutionJob> CreateCudaExecutionJob(
    std::shared_ptr<const UsdGenCudaExecutionPlan>, UsdGenCudaExecutionWorkspace&,
    double, uint64_t, UsdGenDiagnostics*, std::shared_ptr<const UsdGenDeviceGeneration> = {});
// A job retains its plan and previous publication, but borrows the workspace
// and diagnostics.  Drain all stage tasks before either borrowed object dies.
size_t CudaExecutionJobOperatorCount(UsdGenCudaExecutionJob const&) noexcept;
bool ExecuteCudaJobSource(UsdGenCudaExecutionJob&);
// Shared asynchronous CurveSource stage. The completion is invoked exactly
// once off the native CUDA callback after its retained source payload reaches
// a terminal success/failure state. A process-lifetime relay owns a fixed
// 1024 admission slots and uses up to eight dedicated worker threads; unsafe
// terminal failures permanently charge their slot rather than freeing CUDA
// staging without proof. Callers must keep their task completion admission in
// the supplied closure; false means no callback was accepted.
bool ExecuteCudaJobSourceAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
                               std::function<void(bool)> completion);
// Asynchronous Width/Length operator continuation.  Other operators retain
// the synchronous implementation but obey the same accepted-completion contract.
bool ExecuteCudaJobOperatorAsync(std::shared_ptr<UsdGenCudaExecutionJob>, size_t,
                                 std::function<void(bool)> completion);
// Asynchronous terminal publication.  A `true` return retains the callback
// and invokes it exactly once; a clean pre-submission setup failure may report
// a null generation inline on the ordinary launcher.  A successful generation
// is delivered only after native tile/topology proof, launcher return, and
// relay-worker construction.  User completion never runs on a native CUDA
// callback thread.  `false` means admission was rejected and no callback was
// retained.
bool FinalizeCudaExecutionJobAsync(std::shared_ptr<UsdGenCudaExecutionJob>,
    std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>)> completion);
void armCudaSourceAsyncCallbackGateForTesting();
void waitCudaSourceAsyncCallbackGateForTesting();
void releaseCudaSourceAsyncCallbackGateForTesting();
// Test-only relay controls. Capacity is capped at the production 1024 slots;
// injected callback failures deliberately quarantine an admission.
void setCudaSourceRelayCapacityForTesting(size_t);
size_t cudaSourceRelayOccupiedCountForTesting();
size_t cudaSourceRelayQuarantinedCountForTesting();
void failNextCudaSourceRelayCallbackInstallForTesting();
void failNextCudaSourceRelayNativeCallbackForTesting();
// Source-control proof phases precede the existing upload/resample phases.
// They are test-only controls; the production relay never waits on them.
void armCudaSourceAsyncContextCallbackGateForTesting();
void waitCudaSourceAsyncContextCallbackGateForTesting();
void releaseCudaSourceAsyncContextCallbackGateForTesting();
void armCudaSourceAsyncProgramCallbackGateForTesting();
void waitCudaSourceAsyncProgramCallbackGateForTesting();
void releaseCudaSourceAsyncProgramCallbackGateForTesting();
void armCudaSourceAsyncScalarCallbackGateForTesting();
void waitCudaSourceAsyncScalarCallbackGateForTesting();
void releaseCudaSourceAsyncScalarCallbackGateForTesting();
void armCudaSourceAsyncScalarLauncherReturnGateForTesting();
void waitCudaSourceAsyncScalarLauncherReturnGateForTesting();
void releaseCudaSourceAsyncScalarLauncherReturnGateForTesting();
void failNextCudaSourceRelayContextCallbackInstallForTesting();
void failNextCudaSourceRelayProgramCallbackInstallForTesting();
void failNextCudaSourceRelayScalarCallbackInstallForTesting();
void failNextCudaSourceRelayContextNativeCallbackForTesting();
void failNextCudaSourceRelayProgramNativeCallbackForTesting();
void failNextCudaSourceRelayScalarNativeCallbackForTesting();
void failNextCudaSourceRelayControlsAllocationForTesting();
void armCudaSourceAsyncResampleCallbackGateForTesting();
void waitCudaSourceAsyncResampleCallbackGateForTesting();
void releaseCudaSourceAsyncResampleCallbackGateForTesting();
void failNextCudaSourceRelayResampleCallbackInstallForTesting();
void failNextCudaSourceRelayResampleNativeCallbackForTesting();
void failNextCudaSourceRelayNamedCallbackInstallForTesting();
void failNextCudaSourceRelayNamedNativeCallbackForTesting();
void armCudaSourceAsyncResampleLauncherReturnGateForTesting();
void waitCudaSourceAsyncResampleLauncherReturnGateForTesting();
void releaseCudaSourceAsyncResampleLauncherReturnGateForTesting();
void setCudaOperatorRelayCapacityForTesting(size_t);
size_t cudaOperatorRelayOccupiedCountForTesting();
size_t cudaOperatorRelayQuarantinedCountForTesting();
void failNextCudaOperatorRelayContextCallbackInstallForTesting();
void failNextCudaOperatorRelayProgramCallbackInstallForTesting();
void failNextCudaOperatorRelayWidthCallbackInstallForTesting();
void failNextCudaOperatorRelayLengthCallbackInstallForTesting();
void failNextCudaOperatorRelayCountsCallbackInstallForTesting();
void failNextCudaOperatorRelayScatterCallbackInstallForTesting();
void failNextCudaOperatorRelayNamedTopologyCallbackInstallForTesting();
void failNextCudaOperatorRelayNonWidthCallbackInstallForTesting();
void failNextCudaOperatorRelayScatterAllocationForTesting();
void failNextCudaOperatorRelayContextNativeCallbackForTesting();
void failNextCudaOperatorRelayProgramNativeCallbackForTesting();
void failNextCudaOperatorRelayWidthNativeCallbackForTesting();
void failNextCudaOperatorRelayLengthNativeCallbackForTesting();
void failNextCudaOperatorRelayCountsNativeCallbackForTesting();
void failNextCudaOperatorRelayScatterNativeCallbackForTesting();
void failNextCudaOperatorRelayNamedTopologyNativeCallbackForTesting();
void failNextCudaOperatorRelayNonWidthNativeCallbackForTesting();
void armCudaOperatorAsyncContextCallbackGateForTesting();
void waitCudaOperatorAsyncContextCallbackGateForTesting();
void releaseCudaOperatorAsyncContextCallbackGateForTesting();
void armCudaOperatorAsyncProgramCallbackGateForTesting();
void waitCudaOperatorAsyncProgramCallbackGateForTesting();
void releaseCudaOperatorAsyncProgramCallbackGateForTesting();
void armCudaOperatorAsyncWidthCallbackGateForTesting();
void waitCudaOperatorAsyncWidthCallbackGateForTesting();
void releaseCudaOperatorAsyncWidthCallbackGateForTesting();
// Multi-claim Width-DAG launcher rendezvous. It is deliberately test-only:
// independently ready branch tasks select their streams and arrive before
// release, without relying on CUDA callback concurrency or wall-clock timing.
void armCudaOperatorAsyncWidthBranchGateForTesting(size_t expected);
void waitCudaOperatorAsyncWidthBranchGateForTesting();
void releaseCudaOperatorAsyncWidthBranchGateForTesting();
size_t cudaOperatorAsyncWidthDistinctBranchStreamCountForTesting();
// Arms a bounded device-side overlap probe for up to eight legal Width-DAG
// branches. `dwellCycles` is clamped to a finite device-clock duration. The
// dormant production path submits no probe, waits, or extra CUDA work.
void armCudaOperatorAsyncWidthDeviceOverlapWitnessForTesting(
    size_t expected, uint64_t dwellCycles = 50000000);
// Test observation only: performs a synchronous D2H counter read, so call it
// after the queue/job has drained. It is never invoked by a CUDA callback.
UsdGenExecutionOverlapWitnessSnapshot
cudaOperatorAsyncWidthDeviceOverlapWitnessSnapshotForTesting();
void armCudaOperatorAsyncLengthCallbackGateForTesting();
void waitCudaOperatorAsyncLengthCallbackGateForTesting();
void releaseCudaOperatorAsyncLengthCallbackGateForTesting();
void armCudaOperatorAsyncCountsCallbackGateForTesting();
void waitCudaOperatorAsyncCountsCallbackGateForTesting();
void releaseCudaOperatorAsyncCountsCallbackGateForTesting();
void armCudaOperatorAsyncScatterCallbackGateForTesting();
void waitCudaOperatorAsyncScatterCallbackGateForTesting();
void releaseCudaOperatorAsyncScatterCallbackGateForTesting();
void armCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
void waitCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
void releaseCudaOperatorAsyncNamedTopologyCallbackGateForTesting();
void armCudaOperatorAsyncWidthLauncherReturnGateForTesting();
void waitCudaOperatorAsyncWidthLauncherReturnGateForTesting();
void releaseCudaOperatorAsyncWidthLauncherReturnGateForTesting();
void setCudaFinalizationRelayCapacityForTesting(size_t);
size_t cudaFinalizationRelayOccupiedCountForTesting();
size_t cudaFinalizationRelayQuarantinedCountForTesting();
void failNextCudaFinalizationRelayTileCallbackInstallForTesting();
void failNextCudaFinalizationRelayBoundsCallbackInstallForTesting();
void failNextCudaFinalizationRelayMetadataCallbackInstallForTesting();
void failNextCudaFinalizationRelayAllocationForTesting();
void failNextCudaFinalizationRelayTopologyAllocationForTesting();
// RBF transaction seams are routed through usdGen so an integrated test
// reaches the same statically linked GPU implementation as the executor.
void failNextCudaRbfFieldResolvePreflightForTesting();
void failNextCudaRbfFieldResolveCommitForTesting();
void failNextCudaRbfSolveAfterInputSubmitForTesting();
void failNextCudaRbfSurfaceResolvePreflightForTesting();
void failNextCudaRbfSurfaceResolveCommitForTesting();
uint64_t cudaRbfFieldAcceptAttemptCountForTesting();
uint64_t cudaRbfFieldRollbackAttemptCountForTesting();
uint64_t cudaRbfSurfaceAcceptAttemptCountForTesting();
uint64_t cudaRbfSurfaceRollbackAttemptCountForTesting();
void failNextCudaSynchronousFinalizationForTesting();
void failNextCudaFinalizationRelayTileNativeCallbackForTesting();
void failNextCudaFinalizationRelayBoundsNativeCallbackForTesting();
void failNextCudaFinalizationRelayMetadataNativeCallbackForTesting();
void armCudaFinalizationAsyncTileCallbackGateForTesting();
void waitCudaFinalizationAsyncTileCallbackGateForTesting();
void releaseCudaFinalizationAsyncTileCallbackGateForTesting();
void armCudaFinalizationAsyncBoundsCallbackGateForTesting();
void waitCudaFinalizationAsyncBoundsCallbackGateForTesting();
void releaseCudaFinalizationAsyncBoundsCallbackGateForTesting();
void armCudaFinalizationAsyncMetadataCallbackGateForTesting();
void waitCudaFinalizationAsyncMetadataCallbackGateForTesting();
void releaseCudaFinalizationAsyncMetadataCallbackGateForTesting();
// Defers only finalization phase-2's Return signal; it never blocks a relay
// worker while a test observes native-terminal-before-return ordering.
void armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
void waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
void releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
bool ExecuteCudaJobOperator(UsdGenCudaExecutionJob&, size_t);
std::shared_ptr<const UsdGenDeviceGeneration> FinalizeCudaExecutionJob(UsdGenCudaExecutionJob&);

}
#endif
