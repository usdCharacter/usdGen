#include "usdGen/exec/session.h"
#include "usdGen/exec/sessionBackendExecutor.h"
#include "usdGen/exec/sessionCooker.h"
#include "usdGen/exec/sessionDeviceIntegration.h"
#include "usdGen/registries/backend.h"

#include <limits>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace usdGen {
namespace {
// Leaked on purpose: the runtime owns a TBB arena, and a static destructor in
// this library runs at DLL_PROCESS_DETACH on Windows, after ExitProcess has
// terminated every worker thread. Arena teardown waits for workers that can
// never answer. The OS releases the mapping at process death regardless.
// Published once by CAS; a losing candidate is destroyed by its creator.
std::atomic<UsdGenExecutionRuntime*> gDefaultRuntime{nullptr};
UsdGenExecutionRuntime& DefaultRuntime() {
    UsdGenExecutionRuntime* runtime = gDefaultRuntime.load(std::memory_order_acquire);
    if (runtime) return *runtime;
    auto candidate = std::make_unique<UsdGenExecutionRuntime>(8);
    if (gDefaultRuntime.compare_exchange_strong(runtime, candidate.get(),
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire))
        return *candidate.release();
    return *runtime;
}

void Merge(UsdGenPendingDirty& into, UsdGenPendingDirty const& pending) {
    if (pending.structural) {
        into = UsdGenPendingDirty{};
        into.structural = true;
        return;
    }
    into.surfaceTopology |= pending.surfaceTopology;
    for (auto const& entry : pending.nodeBits) into.nodeBits[entry.first] |= entry.second;
    for (auto const& entry : pending.surfaceBits) into.surfaceBits[entry.first] |= entry.second;
}

UsdGenSessionGraphSnapshot GraphInfo(UsdGenSessionCooker const& cooker) {
    UsdGenSessionGraphSnapshot info;
    info.nodeCount = cooker.NodeCount();
    info.spec = cooker.GraphSpec();
    info.devicePlan = cooker.DevicePlan();
    for (auto const& node : info.spec.nodes)
        info.nodeIds[node.key] = cooker.NodeIdForKey(node.key);
    return info;
}

struct CommitReply {
    UsdGenSession::Completion completion;
    bool completed = false; // command owner only
    void Finish(UsdGenSession::SnapshotPtr snapshot, UsdGenExecutionPipeline::Outcome outcome) {
        if (completed) return;
        completed = true;
        if (completion) completion(std::move(snapshot), outcome);
    }
};

// A parked cache follower has no worker callback to observe supersession.
// Keep its registration in a tiny cross-thread control object; cancellation
// only touches the shared domain and the pipeline AsyncCompletion, never the
// owner-confined cooker. One atomic slot holds the registration or CLOSED:
// whoever takes a registration out of the slot owns it, and a Register that
// finds CLOSED cancels its own token.
struct CoalescedControl {
    struct Registered {
        std::shared_ptr<UsdGenExecutionCacheDomain> domain;
        UsdGenExecutionCacheDomain::CoalescedRegistration token;
    };
    static Registered* Closed() noexcept {
        return reinterpret_cast<Registered*>(uintptr_t(1));
    }
    std::atomic<Registered*> slot{nullptr};

    ~CoalescedControl() {
        Registered* left = slot.load(std::memory_order_acquire);
        if (left != Closed()) delete left;
    }
    void Register(std::shared_ptr<UsdGenExecutionCacheDomain> value,
                  UsdGenExecutionCacheDomain::CoalescedRegistration token) {
        Registered* current = slot.load(std::memory_order_acquire);
        if (current != Closed()) {
            auto* next = new Registered{value, token};
            while (current != Closed()) {
                if (slot.compare_exchange_weak(current, next, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
                    delete current;  // a displaced earlier registration
                    return;
                }
            }
            delete next;
        }
        if (value) value->CancelCoalescedAndNotify(token);
    }
    void Cancel() {
        Registered* taken = slot.exchange(Closed(), std::memory_order_acq_rel);
        if (!taken || taken == Closed()) return;
        std::unique_ptr<Registered> owned(taken);
        if (owned->domain) owned->domain->CancelCoalescedAndNotify(owned->token);
    }
};

// One explicitly injected device cook keeps the Session work gate occupied
// until its immutable result reaches the original command owner. The raw
// pipeline pointer is protected by that gate; it is never accessed after a
// thread-neutral failure releases the gate.
struct DeviceReturnRelay : std::enable_shared_from_this<DeviceReturnRelay> {
    UsdGenExecutionPipeline* owner = nullptr;
    UsdGenExecutionPipeline::CommandTicket ticket;
    UsdGenExecutionPipeline::AsyncCompletion failWork;
    UsdGenSessionCooker::DeviceResultHandler handler;
    std::shared_ptr<UsdGenSessionBackendExecutor> executor;
    std::shared_ptr<const UsdGenSessionDeviceResult> result;
    std::exception_ptr lost;
    std::atomic<bool> received{false}, settled{false};
    std::shared_ptr<DeviceReturnRelay> quarantine;

    void Lost() noexcept {
        if (settled.exchange(true, std::memory_order_acq_rel)) return;
        // Accepted native-bearing results and mutable handler captures may
        // have no safe destruction owner after transport loss. Retain them
        // before releasing the gate that protects the Session lifetime.
        quarantine = shared_from_this();
        try { if (failWork) failWork({}, lost); } catch (...) {}
    }
    bool Return(std::shared_ptr<const UsdGenSessionDeviceResult> value) noexcept {
        if (received.exchange(true, std::memory_order_acq_rel)) return false;
        result = std::move(value);
        auto self = shared_from_this();
        auto returnTicket = std::move(ticket);
        bool posted = false;
        try {
            posted = owner->PostCommand(std::move(returnTicket), [self] {
                if (self->settled.exchange(true, std::memory_order_acq_rel)) return;
                try { self->handler(self->result); }
                catch (...) {
                    try { self->failWork({}, std::current_exception()); } catch (...) {}
                }
                // Native producer closures may still retain the relay after
                // this command. Clear Session/cooker edges on this owner.
                self->handler = {};
                self->result.reset();
                self->executor.reset();
                self->failWork = {};
                self->owner = nullptr;
            }, [self] { self->Lost(); });
        } catch (...) {}
        if (!posted) Lost();
        return posted;
    }
};
}

struct UsdGenSession::State {
    // Command-owner state. The executor/cooker below is accessed only by the
    // serialized work node and closures retained by that node.
    std::shared_ptr<const UsdGenGraphSpec> desc = std::make_shared<const UsdGenGraphSpec>();
    UsdGenContext context = UsdGenContext::Interactive;
    UsdGenPendingDirty pending;
    std::atomic<bool> dirty{false};
    bool devicePublication = false, densityDrag = false, invalidateValues = false;
    // Losses are coalesced only for an identical target.  Different devices
    // (and a backend-wide target versus a device target) remain distinct so a
    // rapid sequence cannot overwrite an unconsumed invalidation fence.
    std::vector<UsdGenDeviceContextLoss> deviceContextLosses;
    uint64_t nextDeviceContextLossSerial = 0;
    bool deviceContextLossWorkQueued = false;
    uint64_t deviceContextLossWorkToken = 0;
    std::shared_ptr<const UsdGenDeviceGeneration> staged;
    UsdGenGenerationConstPtr stagedBase;
    uint64_t activeEdit = 0, nextEdit = 0;
    uint64_t publishedWorkerEpoch = 0, activeWorkEpoch = 0;
    // Each work closure retains this one per-Session executor. It routes each
    // immutable descriptor without replacing its cooker/workspace.
    std::shared_ptr<UsdGenSessionBackendExecutor> executor;
    std::unique_ptr<UsdGenExecutionRuntime> privateRuntime;
    UsdGenSessionDeviceObservers deviceObservers;
    std::vector<std::shared_ptr<CoalescedControl>> coalescedControls;
    // Last member: shutdown drains/cancels before any captured owner state dies.
    UsdGenExecutionPipeline pipeline;

    explicit State(int limit, uint64_t commandCapacity,
                   std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain,
                   std::shared_ptr<UsdGenSessionDeviceProvider> deviceProvider,
                   UsdGenSessionDeviceObservers const& observers)
        : executor(CreateUsdGenSessionBackendExecutor(
              limit,
              UsdGenSessionCooker::kDefaultExecutionCacheBytes,
              std::move(executionCacheDomain), std::move(deviceProvider))),
          privateRuntime(limit > 0 ? new UsdGenExecutionRuntime(limit) : nullptr),
          deviceObservers(observers),
          pipeline(privateRuntime ? *privateRuntime : DefaultRuntime(), 4096, commandCapacity) {}

    void Invalidate() {
        staged.reset();
        stagedBase.reset();
        dirty.store(true, std::memory_order_release);
        auto controls = coalescedControls;
        for (auto const& control : controls) if (control) control->Cancel();
        // Called on this pipeline's owner: invalidation precedes any queued
        // completion, even while the work node is occupied by an old cook.
        pipeline.CancelPending();
    }
    void TrackCoalesced(std::shared_ptr<CoalescedControl> control) {
        if (control) coalescedControls.push_back(std::move(control));
    }
    void DropCoalesced(std::shared_ptr<CoalescedControl> const& control) {
        coalescedControls.erase(std::remove(coalescedControls.begin(),
                                            coalescedControls.end(), control),
                                coalescedControls.end());
    }
    void SetDesc(std::shared_ptr<const UsdGenGraphSpec> value) {
        Invalidate();
        desc = std::move(value);
        pending = UsdGenPendingDirty{};
        pending.structural = true;
    }
    void SetContext(UsdGenContext value) {
        if (context == value) return;
        context = value;
        Invalidate();
    }
    void SetDevicePublication(bool value) {
        if (devicePublication == value) return;
        devicePublication = value;
        Invalidate();
    }
    std::vector<UsdGenDeviceContextLoss> DeviceContextLosses() const {
        auto result = deviceContextLosses;
        std::sort(result.begin(), result.end(),
                  [](UsdGenDeviceContextLoss const& a,
                     UsdGenDeviceContextLoss const& b) {
                      return a.serial < b.serial;
                  });
        return result;
    }
    void ScheduleDeviceContextLossWork() {
        if (deviceContextLossWorkQueued || deviceContextLosses.empty()) return;
        auto workExecutor = executor;
        if (!workExecutor) throw std::runtime_error("session has no backend executor");
        auto losses = DeviceContextLosses();
        uint64_t const scheduledThrough = losses.back().serial;
        // Set this before Submit: owner reentry may accept a synchronous
        // successor, and it must observe that this batch is already owned.
        deviceContextLossWorkQueued = true;
        uint64_t const workToken = ++deviceContextLossWorkToken;
        uint64_t submitted = 0;
        try {
            submitted = pipeline.Submit([workExecutor, losses](auto const&) {
                for (UsdGenDeviceContextLoss const& loss : losses) {
                    if (!workExecutor->Cooker().ApplyDeviceContextLoss(loss))
                        throw std::runtime_error(
                            "device context-loss invalidation failed");
                }
                return UsdGenExecutionPipeline::Publish([] {});
            }, [this, scheduledThrough, workToken](uint64_t,
                                        UsdGenExecutionPipeline::Outcome outcome,
                                        std::exception_ptr) {
                // RecordDeviceContextLost may have already installed a newer
                // replacement batch while this old one was being cancelled.
                if (workToken != deviceContextLossWorkToken) return;
                deviceContextLossWorkQueued = false;
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                    deviceContextLosses.erase(
                        std::remove_if(deviceContextLosses.begin(),
                                       deviceContextLosses.end(),
                            [scheduledThrough](UsdGenDeviceContextLoss const& loss) {
                                return loss.serial <= scheduledThrough;
                            }),
                        deviceContextLosses.end());
                }
                // A completed batch may have raced a newer target record;
                // dispatch that remaining suffix. A superseded batch already
                // has a newer commit or replacement batch, and a native
                // failure stays retained for a later explicit retry rather
                // than spinning a failing work item on the owner lane.
                if (outcome == UsdGenExecutionPipeline::Outcome::Published)
                    ScheduleDeviceContextLossWork();
            });
        } catch (...) {
            if (workToken == deviceContextLossWorkToken)
                deviceContextLossWorkQueued = false;
            throw;
        }
        if (!submitted) {
            // Do not drop the loss when the bounded request ingress is full.
            // A terminal completion retries this retained batch after its
            // request credit is released; a later Commit also consumes it.
            if (workToken == deviceContextLossWorkToken)
                deviceContextLossWorkQueued = false;
        }
    }
    void RecordDeviceContextLost(UsdGenBackendId backend, int32_t deviceIndex) {
        if (backend == UsdGenBackendId::Invalid || deviceIndex < -1)
            throw std::invalid_argument("invalid device context-loss target");
        if (nextDeviceContextLossSerial == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("device context-loss serial exhausted");
        UsdGenDeviceContextLoss loss;
        loss.backend = backend;
        loss.deviceIndex = deviceIndex;
        loss.serial = ++nextDeviceContextLossSerial;
        auto existing = std::find_if(deviceContextLosses.begin(),
                                     deviceContextLosses.end(),
            [backend, deviceIndex](UsdGenDeviceContextLoss const& value) {
                return value.backend == backend && value.deviceIndex == deviceIndex;
            });
        if (existing != deviceContextLosses.end()) *existing = loss;
        else deviceContextLosses.push_back(loss);
        activeEdit = 0;
        Invalidate();
        // CancelPending can supersede an already-running loss batch. Replace
        // it explicitly so the new target is not hidden behind the old
        // completion callback's queued flag.
        if (deviceContextLossWorkQueued) {
            deviceContextLossWorkQueued = false;
            ++deviceContextLossWorkToken;
        }
        ScheduleDeviceContextLossWork();
    }
};

UsdGenSession::UsdGenSession(int limit, uint64_t commandCapacity)
    : UsdGenSession(limit, commandCapacity, {}) {}

UsdGenSession::UsdGenSession(
    int limit, uint64_t commandCapacity,
    std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain)
    : UsdGenSession(limit, commandCapacity, std::move(executionCacheDomain), {}, {}) {}

UsdGenSession::UsdGenSession(int limit, uint64_t commandCapacity,
    std::shared_ptr<UsdGenExecutionCacheDomain> executionCacheDomain,
    std::shared_ptr<UsdGenSessionDeviceProvider> deviceProvider,
    UsdGenSessionDeviceObservers const& observers)
    : _state(new State(limit, commandCapacity, std::move(executionCacheDomain),
                       std::move(deviceProvider), observers)) {}

std::unique_ptr<UsdGenSession> CreateUsdGenDeviceSession(int limit,
    uint64_t commandCapacity, std::shared_ptr<UsdGenExecutionCacheDomain> domain,
    std::shared_ptr<UsdGenSessionDeviceProvider> provider,
    UsdGenSessionDeviceObservers const& observers)
{
    if (!provider || !domain) return {};
    auto const identity = provider->Identity();
    auto const& key = domain->Key();
    // Only a backend that owns its logical contexts and revisions shares one.
    UsdGenBackendDef const *def = UsdGenGetBackend(identity.backend);
    if (!def || !(def->flags & kUsdGenBackendAuthoritativeRevisions) ||
        !identity.logicalContextIdentity || key.backend != identity.backend ||
        key.deviceIndex != identity.deviceIndex ||
        key.contextIdentity != identity.logicalContextIdentity) return {};
    // All Sessions on this provider context must share one invalidation and
    // publication fence, not merely compare equal context labels.
    if (domain != UsdGenExecutionCacheDomain::AcquireShared(key, domain->MaxBytes()))
        return {};
    return std::unique_ptr<UsdGenSession>(new UsdGenSession(limit, commandCapacity,
        std::move(domain), std::move(provider), observers));
}
void UsdGenSession::Drain() { _state->pipeline.Drain(); }

UsdGenSession::~UsdGenSession() {
    // External submitters must already have stopped. Keep _snapshot alive while
    // closing invokes terminal callbacks for accepted work and commands.
    if (UsdGenExecutionPipeline::IsExecuting()) std::terminate();
    for (auto const& control : _state->coalescedControls)
        if (control) control->Cancel();
    _state->pipeline.Shutdown(_state->deviceObservers.admissionClosed);
    // Shutdown happens only after the pipeline has terminally drained every
    // captured executor reference. Device completions can no longer enter
    // the cooker before it releases a remaining silent coalesced registration.
    if (_state->executor) _state->executor->Shutdown();
    _state.reset();
}
UsdGenSession::SnapshotPtr UsdGenSession::Snapshot() const noexcept {
    return _snapshot.Load();
}
int UsdGenSession::CaptureCallerDevice() noexcept {
    for (uint32_t i = 1; i < kUsdGenMaxBackends; ++i) {
        UsdGenBackendDef const *def = UsdGenGetBackend(UsdGenBackendId(i));
        if (def && def->captureCallerDevice) return def->captureCallerDevice();
    }
    return -2;
}
bool UsdGenSession::NeedsCommit() const noexcept {
    return _state->dirty.load(std::memory_order_acquire);
}
bool UsdGenSession::PostGraphSpec(UsdGenGraphSpec spec) {
    auto value = std::make_shared<const UsdGenGraphSpec>(std::move(spec));
    return _state->pipeline.PostCommand([this, value] { _state->SetDesc(value); });
}
bool UsdGenSession::PostContext(UsdGenContext context) {
    return _state->pipeline.PostCommand([this, context] { _state->SetContext(context); });
}
bool UsdGenSession::PostDevicePublicationEnabled(bool value) {
    return _state->pipeline.PostCommand([this, value] { _state->SetDevicePublication(value); });
}
bool UsdGenSession::PostDirty(UsdGenPendingDirty pending) {
    if (!pending.Any()) return true;
    return _state->pipeline.PostCommand([this, pending=std::move(pending)] {
        Merge(_state->pending, pending);
        _state->Invalidate();
    });
}
bool UsdGenSession::PostDeviceContextLost(
    UsdGenBackendId backend, int32_t deviceIndex)
{
    if (backend == UsdGenBackendId::Invalid || deviceIndex < -1) return false;
    return _state->pipeline.PostCommand([this, backend, deviceIndex] {
        _state->RecordDeviceContextLost(backend, deviceIndex);
    });
}
void UsdGenSession::SetGraphSpec(UsdGenGraphSpec const& spec) {
    auto value = std::make_shared<const UsdGenGraphSpec>(spec);
    _state->pipeline.InvokeOwner([this, value] { _state->SetDesc(value); });
}
void UsdGenSession::SetContext(UsdGenContext context) {
    _state->pipeline.InvokeOwner([this, context] { _state->SetContext(context); });
}
void UsdGenSession::SetDevicePublicationEnabled(bool value) {
    _state->pipeline.InvokeOwner([this, value] { _state->SetDevicePublication(value); });
}
void UsdGenSession::NotifyDeviceContextLost(
    UsdGenBackendId backend, int32_t deviceIndex)
{
    if (backend == UsdGenBackendId::Invalid || deviceIndex < -1)
        throw std::invalid_argument("invalid device context-loss target");
    _state->pipeline.InvokeOwner([this, backend, deviceIndex] {
        _state->RecordDeviceContextLost(backend, deviceIndex);
    });
}
void UsdGenSession::AccumulateDirty(UsdGenPendingDirty&& pending) {
    if (!pending.Any()) return;
    _state->pipeline.InvokeOwner([this, pending=std::move(pending)] {
        Merge(_state->pending, pending);
        _state->Invalidate();
    });
}

bool UsdGenSession::BeginDeviceEdit(UsdGenGenerationConstPtr const& expected,
                                    uint64_t* token, std::string* reason) {
    bool ok = false;
    _state->pipeline.InvokeOwner([&] {
        auto& state = *_state;
        if (!token || state.activeEdit || state.nextEdit == UINT64_MAX ||
            !state.devicePublication || !expected || !expected->device ||
            expected != Generation() || state.dirty.load() || state.pending.Any() ||
            state.activeWorkEpoch) {
            if (reason) *reason = "device stroke is already reserved or its base is not clean/current";
            return;
        }
        state.activeEdit = ++state.nextEdit;
        *token = state.activeEdit;
        ok = true;
    });
    return ok;
}

bool UsdGenSession::StageDeviceRevision(UsdGenGenerationConstPtr const& expected,
    std::shared_ptr<const UsdGenDeviceGeneration> revision, uint64_t token, std::string* reason) {
    bool ok = false;
    _state->pipeline.InvokeOwner([&] {
        auto& state = *_state;
        if (!token || token != state.activeEdit || !state.devicePublication ||
            !expected || !expected->device || expected != Generation() ||
            state.pending.Any() || state.activeWorkEpoch || (state.dirty.load() && !state.staged)) {
            if (reason) *reason = "device edit base is stale or the graph has pending changes";
            return;
        }
        if (!revision || !revision->Owner()->ProducerReady() ||
            revision->Identity().backend != expected->device->Identity().backend ||
            revision->Identity().deviceIndex != expected->device->Identity().deviceIndex ||
            revision->Identity().generation != static_cast<uint64_t>(expected->id + 1) ||
            revision->Geometry().topologyVersion != expected->device->Geometry().topologyVersion ||
            revision->Geometry().elementCount != expected->device->Geometry().elementCount ||
            revision->Geometry().pointCount != expected->device->Geometry().pointCount ||
            revision->Geometry().alreadyDeformed != expected->device->Geometry().alreadyDeformed) {
            if (reason) *reason = "device edit revision has incompatible identity or topology";
            return;
        }
        state.staged = std::move(revision);
        state.stagedBase = expected;
        state.dirty.store(true, std::memory_order_release);
        ok = true;
    });
    return ok;
}

bool UsdGenSession::EndDeviceEdit(uint64_t token,
    UsdGenGenerationConstPtr const& expected, bool discard) {
    bool ok = false;
    _state->pipeline.InvokeOwner([&] {
        auto& state = *_state;
        if (!token || token != state.activeEdit) return;
        ok = expected && expected == Generation() && !state.pending.Any() &&
            !state.activeWorkEpoch && (!state.dirty.load() || state.staged);
        // A matching reservation must be released even when a cook has made
        // its base stale. A wrong token returned above cannot release it.
        state.activeEdit = 0;
        if (discard && state.staged) {
            state.staged.reset();
            state.stagedBase.reset();
            state.dirty.store(state.pending.Any(), std::memory_order_release);
        }
    });
    return ok;
}

bool UsdGenSession::CommitAsync(CommitRequest request, Completion completion,
                                TileProgressCallback tileProgress) {
    // A backend's current device is thread-local. Capture it at the caller boundary,
    // never infer it from whichever framework worker picks up this request.
    const int callerDevice = request.callerDevice ? *request.callerDevice : CaptureCallerDevice();
    auto reply = std::make_shared<CommitReply>();
    reply->completion = std::move(completion);
    return _state->pipeline.PostCommand([this, request=std::move(request), callerDevice,
                                         reply, tileProgress=std::move(tileProgress)] {
        auto& state = *_state;
        try {
            // These optional mutations and the baseline capture are one owner
            // command.  Do not route them through PostGraphSpec/PostContext:
            // another command could otherwise split this request in two.
            if (request.spec) state.SetDesc(request.spec);
            if (request.context) state.SetContext(*request.context);
            if (request.devicePublication)
                state.SetDevicePublication(*request.devicePublication);
            const double frame = request.frame;
            const UsdGenCommitReason reason = request.reason;
            auto const contextLosses = state.DeviceContextLosses();
            auto baseline = Snapshot();
            auto previous = baseline ? baseline->generation : nullptr;
            if (state.staged && reason == UsdGenCommitReason::LiveOverride &&
                state.devicePublication && !state.pending.Any() && !state.activeWorkEpoch &&
                previous && previous->frame == frame && state.stagedBase == previous) {
                auto next = std::make_shared<UsdGenGeneration>();
                next->id = previous->id + 1;
                next->frame = frame;
                next->device = std::move(state.staged);
                // A point/tool revision replaces only the device payload;
                // retain the host data (paths, presentation) of its base.
                next->host = previous->host;
                auto snapshot = std::make_shared<UsdGenSessionSnapshot>(*baseline);
                snapshot->generation = std::move(next);
                snapshot->report.Reset();
                snapshot->diagnostics = UsdGenDiagnostics{};
                ++snapshot->stats.commits;
                _snapshot.Store(SnapshotPtr(snapshot));
                state.stagedBase.reset();
                state.dirty.store(false, std::memory_order_release);
                reply->Finish(snapshot, UsdGenExecutionPipeline::Outcome::Published);
                return;
            }
            state.staged.reset();
            state.stagedBase.reset();
            state.dirty.store(true, std::memory_order_release);
            auto executor = state.executor;
            if (!executor) throw std::runtime_error("session has no backend executor");
            // Both lanes turn a cooker result into exactly the same candidate.
            // In particular, a rejected cook republishes the last good
            // generation with diagnostics, but never consumes pending dirt.
            auto makePublication = [this, baseline, previous, executor](
                UsdGenGenerationConstPtr generation, std::exception_ptr workError,
                uint64_t epoch) {
                auto& cooker = executor->Cooker();
                auto snapshot = std::make_shared<UsdGenSessionSnapshot>();
                snapshot->generation = std::move(generation);
                snapshot->report = cooker.Report();
                snapshot->diagnostics = cooker.Diagnostics();
                snapshot->stats = cooker.Stats();
                snapshot->routing = cooker.Routing();
                snapshot->graphInfo = GraphInfo(cooker);
                snapshot->nodeStats = cooker.NodeStats();
                snapshot->deviceReport = cooker.DeviceReport();
                bool failed = workError || snapshot->diagnostics.HasErrors() ||
                    !snapshot->generation || snapshot->generation == previous;
                std::optional<UsdGenSessionCooker::ExecutionPublicationFence>
                    publicationFence;
                if (!failed) {
                    publicationFence = cooker.TakePublicationFence();
                    if (!publicationFence) {
                        failed = true;
                        workError = std::make_exception_ptr(std::runtime_error(
                            "execution completed without a current domain publication fence"));
                    }
                }
                if (failed) {
                    snapshot->generation = previous;
                    snapshot->report.Reset();
                    snapshot->stats.commits = baseline ? baseline->stats.commits : 0;
                    snapshot->nodeStats = baseline ? baseline->nodeStats
                        : std::unordered_map<UsdGenNodeId, UsdGenNodeStats>{};
                    snapshot->deviceReport = baseline ? baseline->deviceReport
                        : UsdGenRef<const UsdGenPublication>{};
                    if (!snapshot->diagnostics.HasErrors()) {
                        if (workError) {
                            try {
                                std::rethrow_exception(workError);
                            } catch (std::exception const& e) {
                                snapshot->diagnostics.Error(e.what());
                            } catch (...) {
                                snapshot->diagnostics.Error(
                                    "unknown session execution failure");
                            }
                        } else {
                            snapshot->diagnostics.Error(
                                "execution returned no new generation");
                        }
                    }
                } else if (!snapshot->generation->device) {
                    snapshot->deviceReport.Reset();
                }
                std::optional<UsdGenSessionCooker::ExecutionCacheCandidate>
                    cacheCandidate;
                if (!failed)
                    cacheCandidate = cooker.TakeCacheCandidate();
                return UsdGenExecutionPipeline::Publish(
                    [this, executor, snapshot, failed, workError, epoch,
                     cacheCandidate=std::move(cacheCandidate),
                     publicationFence=std::move(publicationFence)]() mutable {
                        // The pipeline executes this only for its current ticket.
                        if (!failed && cacheCandidate) {
                            executor->Cooker().CommitCacheCandidate(*cacheCandidate);
                            // Admission is part of this publication action, so
                            // expose its result in the same immutable snapshot.
                            snapshot->stats = executor->Cooker().Stats();
                        }
                        // Allocate the holder before the fence; publishing
                        // inside it is one exchange.
                        auto prepared = decltype(_snapshot)::Prepare(SnapshotPtr(snapshot));
                        if (!failed && !executor->Cooker().PublishIfCurrent(
                                *publicationFence, [this, &prepared] {
                                    return std::shared_ptr<const void>(
                                        _snapshot.Exchange(std::move(prepared)));
                                })) {
                            throw std::runtime_error(
                                "execution domain was invalidated before session publication");
                        }
                        if (failed)
                            _snapshot.Exchange(std::move(prepared));
                        if (failed) {
                            if (workError) std::rethrow_exception(workError);
                            throw std::runtime_error(
                                "session cook failed; previous generation retained");
                        }
                        // Reaching this action proves the pipeline ticket is
                        // still current: every newer input mutation calls
                        // Invalidate/CancelPending and suppresses the action.
                        // `pending` still contains the dirt captured by this
                        // accepted cook until the lines below clear it, so it
                        // must not be mistaken for a newer mutation.
                        _state->pending = UsdGenPendingDirty{};
                        _state->invalidateValues = false;
                        _state->publishedWorkerEpoch = epoch;
                        _state->dirty.store(false, std::memory_order_release);
                    });
            };
            auto coalescedControl = std::make_shared<CoalescedControl>();
            auto makeCoalescedHooks = [this, frame, executor,
                                       makePublication, coalescedControl,
                                       observer=state.deviceObservers.coalescedRegistered](
                UsdGenExecutionPipeline::AsyncCompletion done,
                uint64_t epoch) mutable {
                UsdGenSessionCooker::CoalescedHooks hooks;
                hooks.registered = [coalescedControl, observer, epoch](
                    std::shared_ptr<UsdGenExecutionCacheDomain> domain,
                    UsdGenExecutionCacheDomain::CoalescedRegistration registration) {
                    coalescedControl->Register(std::move(domain), registration);
                    try { if (observer) observer(epoch, registration.leader); } catch (...) {}
                };
                hooks.callback = [this, frame, epoch, executor,
                                  makePublication, done=std::move(done)](
                    UsdGenExecutionCacheDomain::CoalescedStatus status,
                    std::optional<UsdGenExecutionCacheDomain::Lease> lease,
                    UsdGenSessionCooker::ExecutionPublicationFence fence) mutable {
                    if ((status != UsdGenExecutionCacheDomain::CoalescedStatus::Success &&
                         status != UsdGenExecutionCacheDomain::CoalescedStatus::Resident) ||
                        !lease) {
                        if (status == UsdGenExecutionCacheDomain::CoalescedStatus::Cancelled ||
                            status == UsdGenExecutionCacheDomain::CoalescedStatus::Stale)
                            done({}, {});
                        else
                            done({}, std::make_exception_ptr(std::runtime_error(
                                "coalesced execution did not produce a cache result")));
                        return;
                    }
                    done([this, executor, lease=std::move(*lease), fence, frame,
                          makePublication, epoch]() mutable {
                        UsdGenGenerationConstPtr generation;
                        if (!executor->Cooker().AdoptCoalesced(
                                fence, lease, frame, &generation))
                            throw std::runtime_error(
                                "coalesced cache result could not be adopted");
                        auto publication = makePublication(
                            std::move(generation), {}, epoch);
                        publication();
                    }, {});
                };
                return hooks;
            };
            auto finishReply = [this, reply, executor, coalescedControl](uint64_t epoch,
                UsdGenExecutionPipeline::Outcome outcome, std::exception_ptr error) {
                if (_state->activeWorkEpoch == epoch) _state->activeWorkEpoch = 0;
                executor->Cooker().AbandonCoalesced(true);
                coalescedControl->Cancel();
                _state->DropCoalesced(coalescedControl);
                // A context-loss batch which could not reserve request
                // ingress is retried here after this terminal request has
                // released its credit.  The State owns/coalesces the fences;
                // this callback never captures a stale copy.
                _state->ScheduleDeviceContextLossWork();
                // A candidate publication normally preserves its own diagnostic
                // before throwing. This fallback covers a failure while building
                // that candidate, so no terminal exception is silently lost.
                // Diagnostics are best-effort here: allocation/string handling
                // must not prevent the mandatory external terminal reply.
                if (outcome == UsdGenExecutionPipeline::Outcome::Failed && error) {
                    try {
                        auto current = Snapshot();
                        if (!current || !current->diagnostics.HasErrors()) {
                            auto failed = current ? std::make_shared<UsdGenSessionSnapshot>(*current)
                                                  : std::make_shared<UsdGenSessionSnapshot>();
                            failed->report.Reset();
                            try {
                                std::rethrow_exception(error);
                            } catch (std::exception const& e) {
                                failed->diagnostics.Error(e.what());
                            } catch (...) {
                                failed->diagnostics.Error("unknown session execution failure");
                            }
                            _snapshot.Store(SnapshotPtr(failed));
                        }
                    } catch (...) {
                        // Keep the existing snapshot when recovery diagnostics
                        // cannot be allocated or populated.
                    }
                }
                reply->Finish(Snapshot(), outcome);
            };
            std::shared_ptr<DeviceReturnRelay> deviceReturn;
            if (state.desc->executionBackend != kUsdGenBackendCpu &&
                executor->CanRoute(state.desc->executionBackend)) {
                deviceReturn = std::make_shared<DeviceReturnRelay>();
                deviceReturn->owner = &state.pipeline;
                deviceReturn->executor = executor;
                deviceReturn->lost = std::make_exception_ptr(
                    std::runtime_error("device result lost its Session return owner"));
                deviceReturn->ticket = state.pipeline.ReserveCommandTicket();
                if (!deviceReturn->ticket)
                    throw std::runtime_error("Session device return admission exhausted");
            }
            state.TrackCoalesced(coalescedControl);
            // CPU and device cooks are selected by the immutable request at
            // the private executor boundary. A device cook completes through
            // its provider's return route; CPU completes on this work
            // callback. Either way the pipeline retains `executor` until the
            // terminal completion.
            auto& runtime = state.privateRuntime ? *state.privateRuntime : DefaultRuntime();
            auto work = [executor, &runtime, desc=state.desc, context=state.context,
                         device=state.devicePublication, pending=state.pending,
                         invalidate=state.invalidateValues, publishedEpoch=state.publishedWorkerEpoch,
                         baseline, previous, frame, reason, callerDevice, contextLosses,
                         makePublication, makeCoalescedHooks, deviceReturn,
                         tileProgress=std::move(tileProgress)]
                        (UsdGenExecutionPipeline::Cancellation const& cancellation,
                         UsdGenExecutionPipeline::AsyncCompletion done) mutable {
                auto coalescedHooks = makeCoalescedHooks(done, cancellation.epoch);
                if (deviceReturn) deviceReturn->failWork = done;
                auto finish = [done=std::move(done), makePublication,
                               epoch=cancellation.epoch](UsdGenGenerationConstPtr generation,
                                                         std::exception_ptr stageError) mutable {
                    try {
                        done(makePublication(std::move(generation), stageError, epoch), {});
                    } catch (...) { done({}, std::current_exception()); }
                };
                UsdGenSessionBackendRequest request;
                request.runtime = &runtime;
                request.cancellation = cancellation;
                request.desc = std::move(desc);
                request.context = context;
                request.devicePublicationEnabled = device;
                request.pending = std::move(pending);
                request.frame = frame;
                request.reason = reason;
                request.previous = previous;
                request.publishedStats = baseline ? baseline->stats : UsdGenStats{};
                request.invalidateValues = invalidate;
                request.previousPublishedWorkerEpoch = publishedEpoch;
                request.callerDevice = callerDevice;
                request.contextLosses = std::move(contextLosses);
                request.completion = std::move(finish);
                request.coalescedHooks = std::move(coalescedHooks);
                request.tileProgress = std::move(tileProgress);
                if (deviceReturn) {
                    request.deviceReturnBinder = [deviceReturn](
                        UsdGenSessionCooker::DeviceResultHandler handler) {
                        if (!handler || deviceReturn->handler)
                            throw std::runtime_error("Session device return handler bound twice or empty");
                        deviceReturn->handler = std::move(handler);
                        return UsdGenSessionDeviceReturn([deviceReturn](auto result) {
                            return deviceReturn->Return(std::move(result));
                        });
                    };
                }
                executor->Submit(std::move(request));
            };
            state.activeWorkEpoch = state.pipeline.SubmitAsync(std::move(work), finishReply);
            if (!state.activeWorkEpoch) {
                state.DropCoalesced(coalescedControl);
                coalescedControl->Cancel();
                reply->Finish(Snapshot(), UsdGenExecutionPipeline::Outcome::Superseded);
            }
        } catch (...) {
            reply->Finish(Snapshot(), UsdGenExecutionPipeline::Outcome::Failed);
        }
    }, [this, reply] { reply->Finish(Snapshot(), UsdGenExecutionPipeline::Outcome::Superseded); });
}

bool UsdGenSession::CommitAsync(double frame, UsdGenCommitReason reason,
                                Completion completion) {
    CommitRequest request;
    request.frame = frame;
    request.reason = reason;
    return CommitAsync(std::move(request), std::move(completion));
}

UsdGenSession::SnapshotPtr UsdGenSession::CommitSnapshot(CommitRequest request) {
    SnapshotPtr result;
    _state->pipeline.Await([&](std::function<void()> release) {
        if (!CommitAsync(std::move(request), [&, release](SnapshotPtr snapshot, UsdGenExecutionPipeline::Outcome) {
                result = std::move(snapshot);
                release();
            })) throw std::runtime_error("session rejected commit command");
    });
    return result;
}

UsdGenSession::SnapshotPtr UsdGenSession::CommitSnapshot(
    double frame, UsdGenCommitReason reason) {
    CommitRequest request;
    request.frame = frame;
    request.reason = reason;
    return CommitSnapshot(std::move(request));
}
std::optional<UsdGenNodeStats> UsdGenSession::NodeStats(UsdGenNodeId id) const {
    auto snapshot = Snapshot();
    if (!snapshot) return {};
    auto found = snapshot->nodeStats.find(id);
    return found == snapshot->nodeStats.end() ? std::optional<UsdGenNodeStats>{} : found->second;
}
void UsdGenSession::InvalidateAllValues() {
    _state->pipeline.InvokeOwner([this] {
        _state->invalidateValues = true;
        _state->Invalidate();
    });
}
void UsdGenSession::BeginDensityDrag() {
    _state->pipeline.InvokeOwner([this] { _state->densityDrag = true; });
}
void UsdGenSession::EndDensityDrag() {
    _state->pipeline.InvokeOwner([this] {
        _state->densityDrag = false;
        _state->Invalidate();
    });
}
} // namespace usdGen
