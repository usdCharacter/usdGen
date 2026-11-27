#ifndef USDGEN_SESSION_COOKER_H
#define USDGEN_SESSION_COOKER_H

#include "usdGen/session.h"
#include "usdGen/compiler.h"
#include "usdGen/epoch.h"
#include "usdGen/executionCache.h"
#include "usdGen/sessionDeviceProvider.h"
#include "usdGen/scheduler.h"
#include "usdGen/registries/publisher.h"

#include <memory>
#include <functional>
#include <cstddef>
#include <optional>
#include <string>

namespace usdGen {

// Private serial-work-owner state.  It is deliberately not a publication
// object: all values it returns are copied into a candidate for Session's
// command owner to accept or discard.
class UsdGenSessionCooker {
public:
    static constexpr size_t kDefaultExecutionCacheBytes = 64u * 1024u * 1024u;
    using CookCompletion = std::function<void(UsdGenGenerationConstPtr, std::exception_ptr)>;
    using DeviceResultHandler = std::function<void(
        std::shared_ptr<const UsdGenSessionDeviceResult>)>;
    // Session supplies a pre-reserved command-owner return route. The
    // provider may invoke it from its device owner or a thread-neutral loss
    // route; the handler therefore never directly enters cooker state.
    using DeviceReturnBinder = std::function<UsdGenSessionDeviceReturn(
        DeviceResultHandler)>;
    explicit UsdGenSessionCooker(
        int threadLimit = 0,
        size_t executionCacheBytes = kDefaultExecutionCacheBytes);
    UsdGenSessionCooker(
        int threadLimit, size_t executionCacheBytes,
        std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain);
    ~UsdGenSessionCooker();
    UsdGenSessionCooker(UsdGenSessionCooker const&) = delete;
    UsdGenSessionCooker& operator=(UsdGenSessionCooker const&) = delete;

    struct ExecutionPublicationFence {
        std::shared_ptr<UsdGenExecutionCacheDomain> domain;
        UsdGenExecutionCacheDomain::PublicationFence fence;
        explicit operator bool() const noexcept {
            return domain && bool(fence);
        }
    };
    using CoalescedLease = UsdGenExecutionCacheDomain::Lease;
    enum class CoalescedRole { None, Leader, Follower, Resident };
    struct CoalescedHooks {
        using Callback = std::function<void(
            UsdGenExecutionCacheDomain::CoalescedStatus,
            std::optional<CoalescedLease>, ExecutionPublicationFence)>;
        using Registered = std::function<void(
            std::shared_ptr<UsdGenExecutionCacheDomain>,
            UsdGenExecutionCacheDomain::CoalescedRegistration)>;
        Callback callback;
        Registered registered;
    };

    /// Called solely by the serial work owner. `desc` is the exact immutable
    /// command snapshot, never an alias to command-owner mutable input.
    UsdGenGenerationConstPtr Cook(std::shared_ptr<const UsdGenGraphSpec> desc,
        UsdGenContext context, bool devicePublicationEnabled,
        UsdGenPendingDirty pending, double frame, UsdGenCommitReason reason,
        UsdGenGenerationConstPtr previous, UsdGenStats publishedStats,
        bool invalidateValues, uint64_t previousPublishedWorkerEpoch,
        uint64_t workEpoch, int callerDevice, CoalescedHooks hooks = {},
        UsdGenSession::TileProgressCallback progress = {},
        std::function<bool()> superseded = {});
    // One device cook through the backend's Session provider. The caller must
    // retain this cooker through completion (the session pipeline does).
    void CookDeviceAsync(UsdGenExecutionRuntime&, UsdGenExecutionPipeline::Cancellation const&,
        std::shared_ptr<const UsdGenGraphSpec>, UsdGenContext, bool,
        UsdGenPendingDirty, double, UsdGenCommitReason, UsdGenGenerationConstPtr,
        UsdGenStats, bool, uint64_t, int,
        std::shared_ptr<UsdGenSessionDeviceProvider>, DeviceReturnBinder,
        CookCompletion, CoalescedHooks hooks = {});
    UsdGenRef<const UsdGenPublication> Report() const { return _lastReport; }
    UsdGenDiagnostics Diagnostics() const { return _lastDiagnostics; }
    UsdGenStats Stats() const { return _stats; }
    /// The last executed device cook's backend report (provider-defined).
    UsdGenRef<const UsdGenPublication> DeviceReport() const { return _deviceReport; }
    std::unordered_map<UsdGenNodeId, UsdGenNodeStats> NodeStats() const;
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> Routing() const;
    UsdGenGraphSpec GraphSpec() const;
    /// The device plan compiled with the graph (kUsdGenBackendReusesPlan).
    std::shared_ptr<const UsdGenExecutionPlanHandle> DevicePlan() const { return _devicePlan; }
    struct ExecutionCacheCandidate {
        UsdGenExecutionCacheKey key;
        UsdGenGenerationConstPtr generation;
        size_t bytes = 0;
    };
    // Moves with a completed cooker generation into Session's command-owner
    // publication action.  The domain stays alive until that action either
    // publishes the snapshot or observes matching invalidation.
    CoalescedRole GetCoalescedRole() const noexcept { return _coalescedRole; }
    bool AdoptCoalesced(ExecutionPublicationFence const&, CoalescedLease const&,
                        double frame, UsdGenGenerationConstPtr* generation);
    void AbandonCoalesced(bool notify = false);
    /// Candidate ownership crosses into Session's accepted publication
    /// action. Taking it does not seed the cache; only CommitCacheCandidate
    /// after that action is accepted may do so.
    std::optional<ExecutionCacheCandidate> TakeCacheCandidate();
    bool CommitCacheCandidate(ExecutionCacheCandidate const&);
    std::optional<ExecutionPublicationFence> TakePublicationFence();
    // This allocation-free exchange executes with the shared domain serialized
    // against invalidation. It returns the displaced immutable owner so that
    // destruction happens only after the domain lock has been released.
    bool PublishIfCurrent(ExecutionPublicationFence const&,
                          std::function<std::shared_ptr<const void>()>);
    /// Work-lane-only consumption of a command-owner context-loss fence.
    bool ApplyDeviceContextLoss(UsdGenDeviceContextLoss const&);
    // Installed at Session construction, before the serial loss queue is
    // replayed, so a first context-loss fence cannot race provider attachment.
    void SetDeviceProvider(std::shared_ptr<UsdGenSessionDeviceProvider>);
    // Work-lane (owner) reads: the lane is the only writer of the domain.
    size_t ExecutionCacheBytes() const noexcept {
        return _executionCacheDomain ? _executionCacheDomain->Bytes() : 0;
    }
    size_t ExecutionCacheSize() const noexcept {
        return _executionCacheDomain ? _executionCacheDomain->Size() : 0;
    }
    int NodeCount() const noexcept { return _graph.NodeCount(); }
    UsdGenNodeId NodeIdForKey(UsdGenNodeKey key) const { return _graph.NodeIdForKey(key); }
private:
    bool _Prepare(std::shared_ptr<const UsdGenGraphSpec>, UsdGenContext,
                  bool devicePublicationEnabled, UsdGenPendingDirty&,
                  UsdGenGenerationConstPtr previous, UsdGenStats publishedStats,
                  bool invalidateValues, uint64_t previousPublishedWorkerEpoch,
                  UsdGenCompiler::DevicePlanCompiler const& deviceCompiler = {},
                  std::shared_ptr<const UsdGenExecutionPlanHandle>* devicePlan = nullptr,
                  bool reusesPlan = false);
    // Retries a pending poisoned-workspace invalidation of the cache domain.
    bool _InvalidatePoisonedDevice();
    // Shared device finalization: prepares the private immutable candidate and
    // cache fence only. Session's existing command publication still owns final
    // acceptance and cache insertion. Call solely from the retained cook lane.
    bool _FinalizeDevicePublication(
        std::shared_ptr<const UsdGenDeviceGeneration>,
        std::shared_ptr<UsdGenExecutionCacheDomain> const&,
        UsdGenExecutionCacheKey const&, double frame);
    bool _SelectExecutionCacheDomain(UsdGenBackendId, int32_t,
                                    uint64_t contextIdentity = 0);
    bool _ObserveExecutionCacheDomainEpoch();
    void _BeginCoalesced(std::shared_ptr<UsdGenExecutionCacheDomain> const&,
                         UsdGenExecutionCacheKey const&, CoalescedHooks const&);
    UsdGenCompiler _compiler;
    UsdGenRef<const UsdGenPublication> _lastReport;
    UsdGenDiagnostics _lastDiagnostics;
    UsdGenGraphSpec _desc;
    std::shared_ptr<const UsdGenGraphSpec> _descIdentity;
    UsdGenGraph _graph;
    std::shared_ptr<UsdGenSessionDeviceProvider> _deviceProvider;
    // The device backend last cooked: its def flags and diagnostics label.
    uint32_t _deviceFlags = 0;
    char const *_deviceLabel = "device";
    // A poisoned workspace whose cache-domain invalidation is still owed.
    bool _devicePoisoned = false;
    std::shared_ptr<const UsdGenExecutionPlanHandle> _devicePlan;
    UsdGenRef<const UsdGenPublication> _deviceReport;
    // Captured at the accepted device publication boundary. Cache admission
    // must reject if the provider's full identity changes before Session's
    // later command-owner acceptance action.
    std::optional<UsdGenSessionDeviceIdentity> _acceptedDeviceIdentity;
    std::optional<UsdGenSessionDeviceIdentity> _activeDeviceIdentity;
    uint64_t _lastDeviceContextLossSerial = 0;
    UsdGenScheduler _scheduler;
    // The terminal type's publisher and the state it keeps across cooks.
    UsdGenPublisherDef const *_publisher = nullptr;
    std::unique_ptr<UsdGenPublisherState> _publisherState;
    // The terminal type asks to be published and no publisher is registered.
    bool _warnedNoPublisher = false;
    // USDGEN_COMMIT: why and how long the current cook compiled ("" = no compile).
    std::string _traceCompile;
    // A cook failed part-way: the next compile starts from scratch.
    bool _graphUntrusted = false;
    // The published baseline is not this graph's last run: rebuild every group.
    bool _rebuildAllTiles = false;
    // The published publication carries every channel but the points of the
    // graph's current terminal: it was built from this graph, and no node
    // that writes more than points has run since. A cook whose nodes again
    // wrote only points (a deform over an unchanged upstream) then lets the
    // publisher refresh just the points (UsdGenPublishContext::pointsOnly).
    bool _tileChannelsSettled = false;
    UsdGenGenerationStore _store;
    size_t _executionCacheBytes = kDefaultExecutionCacheBytes;
    bool _cacheDomainExplicit = false;
    // Owned by the work lane: written and read only there.
    std::shared_ptr<UsdGenExecutionCacheDomain> _executionCacheDomain;
    uint64_t _observedCacheDomainEpoch = 0;
    // Streamed tile publications of the cook in progress, one slot per
    // completed tile (claimed by fetch_add); kept across cooks.
    struct _ProgressSlot;
    std::unique_ptr<_ProgressSlot[]> _progressSlots;
    size_t _progressSlotCapacity = 0;
    std::optional<ExecutionCacheCandidate> _cacheCandidate;
    std::optional<ExecutionPublicationFence> _publicationFence;
    CoalescedRole _coalescedRole = CoalescedRole::None;
    std::shared_ptr<UsdGenExecutionCacheDomain> _coalescedDomain;
    UsdGenExecutionCacheKey _coalescedKey;
    bool _hasCoalescedKey = false;
    std::optional<UsdGenExecutionCacheDomain::CoalescedRegistration>
        _coalescedRegistration;
    // A cache hit publishes an immutable generation without evaluating the
    // mutable graph buffers.  Until a real execution successfully publishes,
    // those buffers must not be treated as the incremental baseline for the
    // cached generation.
    bool _graphBaselineDetached = false;
    UsdGenStats _stats;
    UsdGenContext _context = UsdGenContext::Interactive;
    bool _devicePublicationEnabled = false;
    std::unordered_map<UsdGenNodeId, UsdGenNodeRunStats> _lastNodeStats;
    // The graph has mutable skip/capture state. If a completed worker did not
    // become the publication baseline, its state must not suppress a later
    // cook relative to that baseline.
    uint64_t _lastCookedEpoch = 0;
    /// The publisher of `terminal`'s type, with its state made on first use.
    UsdGenPublisherDef const *_SelectPublisher(UsdGenElementSet const &terminal);
};
} // namespace usdGen
#endif
