#ifndef USDGEN_SESSION_H
#define USDGEN_SESSION_H

#include "usdGen/exec/executionPipeline.h"
#include "usdGen/exec/device/executionBackend.h"
#include "usdGen/exec/generationStore.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/stats.h"

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>


namespace usdGen {

class UsdGenSessionDeviceProvider;
struct UsdGenSessionDeviceObservers;

/// Value-only dirty input delivered to the command owner. Structural dirt
/// replaces all finer-grained entries; all other entries are OR-merged.
struct UsdGenPendingDirty
{
    bool structural = false;
    bool surfaceTopology = false;
    std::map<UsdGenNodeId, uint32_t> nodeBits;
    std::map<UsdGenSurfaceId, uint32_t> surfaceBits;

    bool Any() const {
        return structural || surfaceTopology || !nodeBits.empty() ||
               !surfaceBits.empty();
    }
};

/// Backend-neutral context-loss fence recorded by the Session command owner
/// and consumed by its serialized cooker lane. Native handles never cross
/// this boundary; device adapters identify only backend and
/// logical device, while `serial` prevents a stale cook from acknowledging a
/// newer loss event.
struct UsdGenDeviceContextLoss
{
    UsdGenBackendId backend = UsdGenBackendId::Invalid;
    int32_t deviceIndex = -1; // -1 invalidates every device of this backend.
    uint64_t serial = 0;

    bool IsValid() const noexcept {
        return backend != UsdGenBackendId::Invalid && deviceIndex >= -1 && serial != 0;
    }
};

/// Detached public graph information. It deliberately contains no compiled
/// graph/operator alias: the cooker may mutate or replace its graph safely.
struct UsdGenSessionGraphSnapshot
{
    int nodeCount = 0;
    UsdGenGraphSpec spec;
    // The device plan compiled with the graph (kUsdGenBackendReusesPlan).
    std::shared_ptr<const UsdGenExecutionPlanHandle> devicePlan;
    std::unordered_map<UsdGenNodeKey, UsdGenNodeId, UsdGenNodeKeyHash> nodeIds;

    int NodeCount() const noexcept { return nodeCount; }
    UsdGenNodeId NodeIdForKey(UsdGenNodeKey key) const {
        auto found = nodeIds.find(key);
        return found == nodeIds.end() ? kUsdGenInvalidNode : found->second;
    }
    UsdGenGraphSpec const& Spec() const noexcept { return spec; }
    std::shared_ptr<const UsdGenExecutionPlanHandle> const& DevicePlan() const noexcept {
        return devicePlan;
    }
};

/// Atomically published immutable result bundle. Readers never enter either
/// command or cook ownership domain.
/// A failed current attempt publishes new diagnostics/compiled graph metadata
/// alongside the retained last-good generation and an empty dirty report.
struct UsdGenSessionSnapshot
{
    UsdGenGenerationConstPtr generation;
    // The publisher's diff against the previous generation; null when empty.
    UsdGenRef<const UsdGenPublication> report;
    UsdGenDiagnostics diagnostics;
    UsdGenStats stats;
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> routing;
    UsdGenSessionGraphSnapshot graphInfo;
    std::unordered_map<UsdGenNodeId, UsdGenNodeStats> nodeStats;
    // The device backend's report of the last executed cook (provider
    // defined, e.g. binding statistics); null on the CPU lane.
    UsdGenRef<const UsdGenPublication> deviceReport;
};

/// One atomic command-owner commit request. Optional input changes are applied
/// immediately before capturing the publication baseline and submitting work,
/// so they cannot interleave with a separately posted descriptor/context
/// command.
struct UsdGenSessionCommitRequest
{
    double frame = 0.0;
    UsdGenCommitReason reason = UsdGenCommitReason::NoticeBatchEnd;
    std::shared_ptr<const UsdGenGraphSpec> spec;
    std::optional<UsdGenContext> context;
    // Captured by an upstream owner when it must relay the original caller's
    // device (a registered backend's captureCallerDevice). Absent means
    // capture at this Session API boundary.
    std::optional<int> callerDevice;
    // Optional renderer admission selection.  Applied with desc/context in
    // this one owner command before its publication baseline is captured;
    // absent preserves the previously selected device-publication mode.
    std::optional<bool> devicePublication;
};

class UsdGenSession
{
public:
    using SnapshotPtr = std::shared_ptr<const UsdGenSessionSnapshot>;
    /// Provisional CPU tile from an in-flight cook. `epoch` identifies the
    /// pipeline attempt; a terminal failure or supersession discards every
    /// tile from that attempt. The final Session snapshot remains atomic.
    struct TileProgress {
        uint64_t epoch = 0;
        uint64_t sequence = 0;
        double frame = 0.0;
        UsdGenRef<const UsdGenPublication> tile;   // one group
        std::shared_ptr<const void> host;   // the cooked spec's host data
    };
    using TileProgressPtr = std::shared_ptr<const TileProgress>;
    using TileProgressCallback = std::function<void(TileProgressPtr)>;
    using Completion = std::function<void(
        SnapshotPtr, UsdGenExecutionPipeline::Outcome)>;
    using CommitRequest = UsdGenSessionCommitRequest;

    // commandCapacity is primarily a deterministic admission-test seam; it
    // bounds owner mutations independently from retained cook requests.
    explicit UsdGenSession(
        int threadLimit = 0, uint64_t commandCapacity = 4096);
    UsdGenSession(
        int threadLimit, uint64_t commandCapacity,
        std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain);
    // External ownership boundary: stop/join submitters first. Accepted work
    // receives terminal callbacks before Session state is destroyed.
    ~UsdGenSession();
    UsdGenSession(UsdGenSession const&) = delete;
    UsdGenSession& operator=(UsdGenSession const&) = delete;

    // External synchronous adapters. They cooperatively invoke the short
    // command owner and must not be called from a pipeline callback.
    void SetGraphSpec(UsdGenGraphSpec const&);
    void SetContext(UsdGenContext);
    void SetDevicePublicationEnabled(bool);

    // Primary callback-safe mutation API: enqueue a command and return.
    // False means bounded command admission rejected the mutation; callers
    // must retain/retry their authoritative input at a later boundary.
    bool PostGraphSpec(UsdGenGraphSpec);
    bool PostContext(UsdGenContext);
    bool PostDevicePublicationEnabled(bool);
    bool PostDirty(UsdGenPendingDirty);
    /// Callback-safe backend context-loss notification. Accepted notification
    /// cancels stale publication and schedules serialized cache/workspace
    /// invalidation. False means command admission rejected it.
    bool PostDeviceContextLost(UsdGenBackendId, int32_t deviceIndex = -1);

    /// External synchronous adapter for the same owner command. It returns
    /// after the loss fence is recorded, not after native retirement drains.
    void NotifyDeviceContextLost(UsdGenBackendId, int32_t deviceIndex = -1);

    bool BeginDeviceEdit(UsdGenGenerationConstPtr const&, uint64_t*,
                         std::string* = nullptr);
    bool StageDeviceRevision(UsdGenGenerationConstPtr const&,
                             std::shared_ptr<const UsdGenDeviceGeneration>,
                             uint64_t, std::string* = nullptr);
    bool EndDeviceEdit(uint64_t, UsdGenGenerationConstPtr const&, bool = true);
    void AccumulateDirty(UsdGenPendingDirty&&);
    bool NeedsCommit() const noexcept;
    /// The first registered backend's captureCallerDevice (a thread-local
    /// current device), or -2 when none captures or capture fails.
    static int CaptureCallerDevice() noexcept;

    /// Asynchronously accept a command snapshot and schedule private cooking.
    /// The completion runs on the command owner and may enqueue follow-up
    /// commands, but must not make synchronous Session calls. Optional tile
    /// progress is CPU-only, delivered serially from the work lane as terminal
    /// tiles complete. It is provisional until completion reports Published;
    /// callers discard the attempt's progress by `epoch` on Failed or Superseded.
    bool CommitAsync(CommitRequest, Completion = {}, TileProgressCallback = {});
    bool CommitAsync(double, UsdGenCommitReason, Completion = {});
    /// Single external quiescent shutdown/test boundary; never called by
    /// render reads or other command owners.
    void Drain();
    /// Compatibility boundary that cooperatively waits for one CommitAsync.
    SnapshotPtr CommitSnapshot(CommitRequest);
    SnapshotPtr CommitSnapshot(double, UsdGenCommitReason);
    UsdGenGenerationConstPtr Commit(double frame, UsdGenCommitReason reason) {
        auto snapshot = CommitSnapshot(frame, reason);
        return snapshot ? snapshot->generation : Generation();
    }

    SnapshotPtr Snapshot() const noexcept;
    // Returned by value; this copy can allocate and is not noexcept.
    UsdGenSessionGraphSnapshot Graph() const {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->graphInfo : UsdGenSessionGraphSnapshot{};
    }
    UsdGenGenerationConstPtr Generation() const noexcept {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->generation : nullptr;
    }
    UsdGenRef<const UsdGenPublication> LastReport() const {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->report : UsdGenRef<const UsdGenPublication>();
    }
    UsdGenDiagnostics LastDiagnostics() const {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->diagnostics : UsdGenDiagnostics{};
    }
    UsdGenStats Stats() const {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->stats : UsdGenStats{};
    }
    std::optional<UsdGenNodeStats> NodeStats(UsdGenNodeId) const;
    /// The routing snapshot of the last published graph; null before one.
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> Routing() const noexcept {
        auto snapshot = Snapshot();
        return snapshot ? snapshot->routing : nullptr;
    }

    void InvalidateAllValues();
    void BeginDensityDrag();
    void EndDensityDrag();

private:
    friend std::unique_ptr<UsdGenSession> CreateUsdGenDeviceSession(
        int, uint64_t, std::shared_ptr<UsdGenExecutionCacheDomain>,
        std::shared_ptr<UsdGenSessionDeviceProvider>,
        UsdGenSessionDeviceObservers const&);
    UsdGenSession(int, uint64_t, std::shared_ptr<UsdGenExecutionCacheDomain>,
                  std::shared_ptr<UsdGenSessionDeviceProvider>,
                  UsdGenSessionDeviceObservers const&);
    struct State;
    std::unique_ptr<State> _state;
    UsdGenSharedCell<UsdGenSessionSnapshot const> _snapshot;
};

} // namespace usdGen

#endif // USDGEN_SESSION_H
