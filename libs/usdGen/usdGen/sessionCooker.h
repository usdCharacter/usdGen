#ifndef USDGEN_SESSION_COOKER_H
#define USDGEN_SESSION_COOKER_H

#include "usdGen/session.h"
#include "usdGen/compiler.h"
#include "usdGen/executionCache.h"
#include "usdGen/furOcclusion.h"
#include "usdGen/sessionDeviceProvider.h"
#include "usdGen/scheduler.h"
#include "usdGen/valuePreview.h"

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
    // CUDA stage state is private mutable cooker ownership carried by one
    // pipeline epoch. It is never shared between sessions.
    struct CudaStages;
    using CudaCompletion = std::function<void(UsdGenGenerationConstPtr, std::exception_ptr)>;
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
    UsdGenGenerationConstPtr Cook(std::shared_ptr<const UsdGenGraphDesc> desc,
        UsdGenContext context, bool devicePublicationEnabled,
        UsdGenPendingDirty pending, double frame, UsdGenCommitReason reason,
        UsdGenGenerationConstPtr previous, UsdGenStats publishedStats,
        bool invalidateValues, uint64_t previousPublishedWorkerEpoch,
        uint64_t workEpoch, int callerDevice, CoalescedHooks hooks = {},
        UsdGenSession::TileProgressCallback progress = {},
        std::function<bool()> superseded = {});
    // Begins a real source -> operator* -> final CUDA task graph. The caller
    // must retain this cooker through completion (the session pipeline does).
    void CookCudaAsync(UsdGenExecutionRuntime&, UsdGenExecutionPipeline::Cancellation const&,
        std::shared_ptr<const UsdGenGraphDesc>, UsdGenContext, bool,
        UsdGenPendingDirty, double, UsdGenCommitReason, UsdGenGenerationConstPtr,
        UsdGenStats, bool, uint64_t, int, CudaCompletion,
        CoalescedHooks hooks = {});
    void CookDeviceAsync(UsdGenExecutionRuntime&, UsdGenExecutionPipeline::Cancellation const&,
        std::shared_ptr<const UsdGenGraphDesc>, UsdGenContext, bool,
        UsdGenPendingDirty, double, UsdGenCommitReason, UsdGenGenerationConstPtr,
        UsdGenStats, bool, uint64_t, int,
        std::shared_ptr<UsdGenSessionDeviceProvider>, DeviceReturnBinder,
        CudaCompletion, CoalescedHooks hooks = {});
    UsdGenDirtyReport Report() const { return _lastReport; }
    UsdGenDiagnostics Diagnostics() const { return _lastDiagnostics; }
    UsdGenStats Stats() const { return _stats; }
    std::vector<UsdGenCudaBindingStats> CudaBindingStats() const;
    std::unordered_map<UsdGenNodeId, UsdGenNodeStats> NodeStats() const;
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> Routing() const;
    UsdGenGraphDesc GraphDesc() const;
    std::shared_ptr<const UsdGenCudaExecutionPlan> CudaPlan() const;
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
    size_t ExecutionCacheBytes() const noexcept {
        auto domain = std::atomic_load(&_executionCacheDomain);
        return domain ? domain->Bytes() : 0;
    }
    size_t ExecutionCacheSize() const noexcept {
        auto domain = std::atomic_load(&_executionCacheDomain);
        return domain ? domain->Size() : 0;
    }
    int NodeCount() const noexcept { return _graph.NodeCount(); }
    UsdGenNodeId NodeIdForPath(SdfPath const& path) const { return _graph.NodeIdForPath(path); }
private:
    bool _Prepare(std::shared_ptr<const UsdGenGraphDesc>, UsdGenContext,
                  bool devicePublicationEnabled, UsdGenPendingDirty&,
                  UsdGenGenerationConstPtr previous, UsdGenStats publishedStats,
                  bool invalidateValues, uint64_t previousPublishedWorkerEpoch,
                  UsdGenCompiler::DevicePlanCompiler const& deviceCompiler = {},
                  std::shared_ptr<const UsdGenExecutionPlanHandle>* devicePlan = nullptr);
    bool _InvalidatePoisonedCudaWorkspace();
    // Shared device finalization: prepares the private immutable candidate and
    // cache fence only. Session's existing command publication still owns final
    // acceptance and cache insertion. Call solely from the retained cook lane.
    bool _FinalizeDevicePublication(
        std::shared_ptr<const UsdGenDeviceGeneration>,
        std::shared_ptr<UsdGenExecutionCacheDomain> const&,
        UsdGenExecutionCacheKey const&, double frame);
    bool _SelectExecutionCacheDomain(UsdGenDeviceBackend, int32_t,
                                    uint64_t contextIdentity = 0);
    bool _ObserveExecutionCacheDomainEpoch();
    void _BeginCoalesced(std::shared_ptr<UsdGenExecutionCacheDomain> const&,
                         UsdGenExecutionCacheKey const&, CoalescedHooks const&);
    UsdGenCompiler _compiler;
    UsdGenDirtyReport _lastReport;
    UsdGenDiagnostics _lastDiagnostics;
    UsdGenGraphDesc _desc;
    std::shared_ptr<const UsdGenGraphDesc> _descIdentity;
    UsdGenGraph _graph;
    std::unique_ptr<UsdGenCudaExecutionWorkspace> _cudaWorkspace;
    std::shared_ptr<UsdGenSessionDeviceProvider> _deviceProvider;
    // Captured at the accepted Vulkan publication boundary. Cache admission
    // must reject if the provider's full identity changes before Session's
    // later command-owner acceptance action.
    std::optional<UsdGenSessionDeviceIdentity> _acceptedDeviceIdentity;
    std::optional<UsdGenSessionDeviceIdentity> _activeDeviceIdentity;
    SdfPath _cudaWorkspaceDescription;
    uint64_t _lastDeviceContextLossSerial = 0;
    std::shared_ptr<const std::vector<UsdGenCudaBindingStats>> _cudaBindingStats =
        std::make_shared<const std::vector<UsdGenCudaBindingStats>>();
    UsdGenScheduler _scheduler;
    // usdGen:preview:* colours of the cook in progress (CPU lane only).
    UsdGenValuePreview _preview;
    UsdGenPreviewColors _previewColors;
    bool _cudaPreviewWarned = false;
    // USDGEN_COMMIT: why and how long the current cook compiled ("" = no compile).
    std::string _traceCompile;
    // A cook failed part-way: the next compile starts from scratch.
    bool _graphUntrusted = false;
    // Identity of the fur density volume's non-tile inputs (occluder meshes,
    // grid parameters): the tile COW check alone cannot see them change.
    uint64_t _furVolumeKey = 0;
    // The world-space occluder mesh, carried across cooks while the emitting
    // surfaces sit still (see UsdGenFurOcclusionParams::occluderCache).
    UsdGenFurOccluderBuild _furOccluderBuild;
    // The cap-build chunk outputs, carried across cooks for their capacity
    // (see UsdGenFurOcclusionParams::scalpScratch).
    UsdGenScalpShadowScratch _furScalpScratch;
    // The volume bake's temporaries, carried across cooks for their capacity
    // (see UsdGenFurOcclusionParams::bakeScratch).
    UsdGenFurBakeScratch _furBakeScratch;
    // The published baseline is not this graph's last run: rebuild every tile.
    bool _rebuildAllTiles = false;
    UsdGenGenerationStore _store;
    size_t _executionCacheBytes = kDefaultExecutionCacheBytes;
    bool _cacheDomainExplicit = false;
    std::shared_ptr<UsdGenExecutionCacheDomain> _executionCacheDomain;
    uint64_t _observedCacheDomainEpoch = 0;
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
    // Per-array tile reuse: when every node that ran this cook is a
    // points-only deform and the partition did not move, each rebuilt tile
    // carries its previous non-point arrays (widths, hairT, counts, extra
    // planes) instead of re-gathering them, and its displayColor too when
    // the color inputs are unchanged. Points are always re-gathered.
    struct TileReuse {
        UsdGenTilePublication const* prevTile = nullptr;
        bool arraysStable = false;
        bool colorsStable = false;
    };
    UsdGenTilePublication _BuildTilePublication(UsdGenTileView const&, UsdGenRunResult const&,
                                                UsdGenGenerationConstPtr const&, TileReuse);
    // Step-6 tile-build fan-out: the worker pool takes a plain function
    // pointer, so the per-tile work travels in this payload. Each body call
    // builds exactly one tile publication from the immutable run result; the
    // cook collects the slots in tile order afterwards.
    struct TileBuildWork {
        UsdGenSessionCooker* cooker = nullptr;
        UsdGenRunResult const* result = nullptr;
        UsdGenGenerationConstPtr const* prev = nullptr;
        size_t const* tileIndex = nullptr;      // result->tiles subscript per slot
        UsdGenTilePublication* built = nullptr; // one slot per ParallelFor index
        TileReuse const* reuse = nullptr;       // one entry per result tile
    };
    static void _BuildTileWork(size_t slot, void* payload);
};
} // namespace usdGen
#endif
