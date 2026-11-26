// usdGen imaging — session store implementation (06-imaging.md §3.7, §3.9).
//
// The wiring contract (06 §3.9): the renderer-level UsdGenGroomSceneIndex
// attaches a session in its constructor, routes observer notices through
// UsdGenDirtyRouter into UsdGenPendingDirty, and commits on triggers
// (a)/(b)/(c). GetPrim paths never commit — they atomic_load the latest
// generation (I7). Superseded commits leave the session dirty (03 §5.6).
#include "usdGenImaging/usdGenImagingSession.h"
#include "usdGen/sharedCell.h"
#include "usdGenImaging/cleanupQueue.h"
#include "usdGenUsd/resolve/imageMapCache.h"

#include "usdGen/epoch.h"
#include "usdGen/generationStore.h"
#include "usdGen/maps/ptexMap.h"
#include "usdGen/session.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/warning.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <limits>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

namespace {
using Pipeline = usdGen::UsdGenExecutionPipeline;
// Process-lifetime objects that own worker threads or outstanding tasks are
// leaked on purpose throughout this file (runtime, retirement queue, session store). A
// static destructor in this library runs at DLL_PROCESS_DETACH on Windows,
// which is after ExitProcess has terminated every other thread: a graph wait
// or arena teardown issued there can never be serviced by a worker, so it
// spins for as long as the process is allowed to live (observed for the store
// destructor's close Await and for ~Retirements). On Linux the same
// destructors ran during exit() with the workers alive, so the leak changes
// nothing that was observable there beyond the mapping's release, which the OS
// performs anyway.
std::atomic<usdGen::UsdGenExecutionRuntime*> g_imagingRuntime{nullptr};
usdGen::UsdGenExecutionRuntime& ImagingRuntime() {
    return usdGen::UsdGenLazyLeaked(g_imagingRuntime,
        [] { return new usdGen::UsdGenExecutionRuntime(8); });
}

// Public TfRefPtr handles may disappear during a notice callback. Retirement
// runs as scheduler tasks, never an inline wait/destructor in that callback.
// Each task owns its retired state until all relayed replies finish.
class Retirements {
    UsdGenCleanupQueue cleanup;
public:
    void Post(std::function<void()> action) { cleanup.Post(std::move(action)); }
    void Drain() {
        if (Pipeline::IsExecuting()) throw std::logic_error("callback cannot drain retired imaging states");
        cleanup.Wait();
    }
};
std::atomic<Retirements*> g_retirementQueue{nullptr};
Retirements& RetirementQueue() { return usdGen::UsdGenLazyLeaked(g_retirementQueue); }
}

struct UsdGenImagingSession::State {
    std::shared_ptr<usdGen::UsdGenSession> engine;
    std::atomic<bool> accepting{true}, stopped{false}, closeClaimed{false};
    std::atomic<bool> appDriver{false}, needsDesc{false};
    std::atomic<int> attached{0};
    std::atomic<int64_t> generation{-1};
    std::atomic<uint64_t> nextToken{0};
    // Everything below is imaging-command-owner-only.
    double frame;
    usdGen::UsdGenContext context;
    std::shared_ptr<const usdGen::UsdGenGraphDesc> staged;
    // Last accepted descriptor remains immutable and available for explicit
    // map reload even after `staged` transfers to the core owner.
    std::shared_ptr<const usdGen::UsdGenGraphDesc> currentDesc;
    // Node keys of every descriptor this session lowered: stable across
    // descriptor edits, so the engine's keyed caches survive them.
    usdGenUsd::NodeKeyMap keys;
    struct Subscription {
        int token;
        std::function<void(CommitPayload const&)> terminal;
        ProgressCallback progress;
    };
    std::vector<Subscription> callbacks;
    unsigned outstanding = 0;
    bool closing = false;
    std::function<void()> closeDone;
    Pipeline owner;
    // This is acquired while the owner is open, before any external producer
    // can make close depend on a terminal owner relay.
    Pipeline::CommandTicket closeTicket;
    // A global context update is required session state, not a best-effort
    // diagnostic. Keep one bounded latest-wins lane permanently reserved so
    // ordinary callback traffic cannot drop it.
    Pipeline::CommandMailbox contextMailbox;

    State(std::shared_ptr<usdGen::UsdGenSession> e, double f, usdGen::UsdGenContext c,
          uint64_t commandCapacity)
        : engine(std::move(e)), frame(f), context(c),
          owner(ImagingRuntime(), 4096, commandCapacity),
          closeTicket(owner.ReserveCommandTicket()),
          contextMailbox(owner.ReserveCommandMailbox()) {
        if (!closeTicket || !contextMailbox)
            throw std::runtime_error("imaging lifecycle admission unavailable");
    }

    // Monotonic cook cancellation token (plan/18 §7 G7). Atomic because
    // CancelCooks is callable from the gesture thread while the owner frame
    // reads it.
    std::atomic<uint64_t> cookToken{0};

    void CompleteClose() {
        if (closing && outstanding == 0 && closeDone) {
            auto done = std::move(closeDone);
            done();
        }
    }
    void Close() {
        if (Pipeline::IsExecuting()) throw std::logic_error("callback cannot shut down imaging synchronously");
        if (stopped.load()) return;
        // Lifecycle admission, not a waiting lock: only one external boundary
        // may drain a graph. Reject competing close callers immediately.
        if (closeClaimed.exchange(true))
            throw std::logic_error("concurrent imaging shutdown is not supported");
        accepting.store(false, std::memory_order_release);
        owner.Await([this](std::function<void()> done) {
            if (!owner.PostCommand(std::move(closeTicket), [this, done] {
                closing = true;
                closeDone = done;
                CompleteClose();
            })) throw std::runtime_error("imaging close command rejected");
        });
        owner.Shutdown();
        stopped.store(true, std::memory_order_release);
    }

    struct ProgressRelay {
        // Producers push tiles from cook threads; the owner takes them all at
        // once (newest first). The mailbox coalesces the drain requests.
        struct Waiting {
            TileProgressPtr tile;
            Waiting* next = nullptr;
        };
        usdGen::UsdGenTreiberStack<Waiting, &Waiting::next> waiting;
        ~ProgressRelay() {
            for (Waiting* w = waiting.PopAll(); w;) {
                Waiting* next = w->next;
                delete w;
                w = next;
            }
        }
        std::vector<TileProgressPtr> cumulative;
        uint64_t epoch = 0;
        uint64_t sequence = 0;
        double frame = 0.0;
        Pipeline::CommandMailbox mailbox;
        ProgressCallback direct;
        uint64_t cookToken = 0;
    };

    void DrainProgress(std::shared_ptr<ProgressRelay> const& relay) {
        if (!relay || !accepting.load(std::memory_order_acquire) ||
            relay->cookToken != cookToken.load(std::memory_order_acquire)) return;
        std::vector<TileProgressPtr> waiting;
        for (ProgressRelay::Waiting* w = relay->waiting.PopAll(); w;) {
            ProgressRelay::Waiting* next = w->next;
            waiting.push_back(std::move(w->tile));
            delete w;
            w = next;
        }
        std::reverse(waiting.begin(), waiting.end());  // newest first -> arrival order
        if (waiting.empty()) return;
        for (auto const& tile : waiting) {
            if (!tile) continue;
            if (relay->epoch != tile->epoch) {
                relay->cumulative.clear();
                relay->epoch = tile->epoch;
            }
            relay->sequence = tile->sequence;
            relay->frame = tile->frame;
            relay->cumulative.push_back(tile);
        }
        ProgressPayload payload;
        payload.epoch = relay->epoch;
        payload.sequence = relay->sequence;
        payload.frame = relay->frame;
        payload.tiles = std::make_shared<const std::vector<TileProgressPtr>>(
            relay->cumulative);
        for (auto const& entry : callbacks) if (entry.progress) {
            try { entry.progress(payload); }
            catch (...) { TF_WARN("usdGen progress callback threw"); }
        }
        if (relay->direct) {
            try { relay->direct(payload); }
            catch (...) { TF_WARN("usdGen imaging progress completion threw"); }
        }
    }

    void Finish(usdGen::UsdGenSession::SnapshotPtr snapshot,
                Pipeline::Outcome outcome, Completion const& done,
                std::shared_ptr<ProgressRelay> const& relay) {
        DrainProgress(relay);
        CommitPayload payload;
        if (relay) payload.progressEpoch = relay->epoch;
        if (!accepting.load()) outcome = Pipeline::Outcome::Superseded;
        if (snapshot) {
            payload.generation = snapshot->generation;
            payload.report = snapshot->report;
            payload.routing = snapshot->routing;
            payload.diagnostics = snapshot->diagnostics;
            payload.stats = snapshot->stats;
        }
        if (outcome == Pipeline::Outcome::Published && payload.generation &&
            payload.generation->id > generation.load()) {
            payload.published = true;
            generation.store(payload.generation->id);
            for (auto const& warning : payload.diagnostics.warnings)
                TF_WARN("usdGen commit: %s", warning.c_str());
            // All mutations enqueue commands; callbacks cannot modify this
            // collection recursively while this publication is being sent.
            for (auto const& entry : callbacks) {
                if (!accepting.load()) break;
                try { if (entry.terminal) entry.terminal(payload); }
                catch (...) { TF_WARN("usdGen republish callback threw"); }
            }
        } else if (outcome == Pipeline::Outcome::Failed) {
            for (auto const& error : payload.diagnostics.errors)
                TF_WARN("usdGen commit rejected: %s", error.c_str());
        }
        if (!payload.published && payload.progressEpoch) {
            for (auto const& entry : callbacks) {
                if (!accepting.load()) break;
                try { if (entry.terminal) entry.terminal(payload); }
                catch (...) { TF_WARN("usdGen republish callback threw"); }
            }
        }
        if (done) {
            try { done(payload, outcome); }
            catch (...) { TF_WARN("usdGen imaging completion threw"); }
        }
        --outstanding;
        CompleteClose();
    }

    void Start(CommitRequest request, int callerDevice, Completion done,
               ProgressCallback progress) {
        if (!accepting.load() || !engine) {
            if (done) done({}, Pipeline::Outcome::Superseded);
            return;
        }
        // Cook cancellation token (plan/17 §3.2 rule 2): a request stamped
        // before the last CancelCooks() describes a state the caller has
        // already replaced. Drop it here, on the owner, before the engine
        // sees it — the cheapest possible abandonment.
        if (request.cookToken && *request.cookToken < cookToken.load()) {
            if (done) done({}, Pipeline::Outcome::Superseded);
            return;
        }
        // Reserve before changing command-owner state or handing anything to
        // the engine. A full terminal lane is a clean rejected request, not a
        // partial frame/context/staged-descriptor edit.
        auto finishTicket = std::make_shared<Pipeline::CommandTicket>(
            owner.ReserveCommandTicket());
        if (!*finishTicket) {
            if (done) {
                try { done({}, Pipeline::Outcome::Failed); }
                catch (...) { TF_WARN("usdGen imaging completion threw"); }
            }
            return;
        }
        bool wantsProgress = request.progressive && bool(progress);
        if (request.progressive) for (auto const& entry : callbacks)
            wantsProgress = wantsProgress || bool(entry.progress);
        std::shared_ptr<ProgressRelay> relay;
        if (wantsProgress) {
            auto candidate = std::make_shared<ProgressRelay>();
            candidate->mailbox = owner.ReserveCommandMailbox();
            // Pressure may suppress preview without suppressing the complete
            // commit. No callback runs unless its relay has reserved ingress.
            if (candidate->mailbox) {
                candidate->direct = std::move(progress);
                candidate->cookToken = *request.cookToken;
                relay = std::move(candidate);
            }
        }
        if (request.frame) frame = *request.frame;
        if (request.context) context = *request.context;
        if (request.desc) {
            auto resolved = std::make_shared<usdGen::UsdGenGraphDesc>(*request.desc);
            ResolveUsdGenImageMaps(resolved.get());
            currentDesc = resolved;
            staged = std::move(resolved);
        }
        usdGen::UsdGenSession::CommitRequest input;
        input.frame = frame;
        input.context = context;
        input.reason = request.reason;
        // Retain our staged descriptor until the core owner accepts it. This
        // lets a public retry preserve the exact graph after core admission
        // rejects a saturated request.
        input.spec = staged ? usdGenUsd::LowerShared(*staged, &keys) : nullptr;
        input.devicePublication = request.devicePublication;
        input.callerDevice = callerDevice;
        ++outstanding;
        try {
            usdGen::UsdGenSession::TileProgressCallback progressFromCore;
            if (relay) progressFromCore = [this, relay](TileProgressPtr tile) {
                if (!tile || relay->cookToken !=
                    cookToken.load(std::memory_order_acquire)) return;
                relay->waiting.Push(new ProgressRelay::Waiting{std::move(tile)});
                (void)owner.PostLatestCommand(relay->mailbox, [this, relay] {
                    DrainProgress(relay);
                });
            };
            if (engine->CommitAsync(std::move(input),
                [this, done, finishTicket, relay](
                    usdGen::UsdGenSession::SnapshotPtr snapshot, Pipeline::Outcome outcome) mutable {
                    // Close leaves the owner alive until every relay arrives.
                    // A valid reserved ticket cannot be rejected for normal
                    // pressure. Rejection here is a framework/lifecycle
                    // breach; never mutate this owner from the producer.
                    if (!owner.PostCommand(std::move(*finishTicket),
                        [this, snapshot, outcome, done, relay] {
                            Finish(snapshot, outcome, done, relay);
                        },
                        [this, done, relay] {
                            Finish({}, Pipeline::Outcome::Superseded, done, relay);
                        }))
                        std::terminate();
                }, std::move(progressFromCore))) {
                // The core owner has retained its input snapshot. Only now
                // consume this staging slot: keeping it after acceptance
                // would resend a structural descriptor on every later tool
                // commit and invalidate an active device edit.
                staged.reset();
                return;
            }
        } catch (...) {
            Finish({}, Pipeline::Outcome::Failed, done, relay);
            return;
        }
        Finish({}, Pipeline::Outcome::Superseded, done, relay);
    }
};

UsdGenImagingSession::UsdGenImagingSession(UsdGenSessionKey const& key,
    std::shared_ptr<usdGen::UsdGenSession> engine, double frame, usdGen::UsdGenContext context,
    uint64_t commandCapacity)
    : _key(key), _engine(std::move(engine)),
      _state(new State(_engine, frame, context, commandCapacity)) {
    (void)RetirementQueue();
}
UsdGenImagingSession::~UsdGenImagingSession() {
    State* state = _state.release();
    state->accepting.store(false, std::memory_order_release);
    // The engine reference leaves with the state. Released here it could be
    // the last one (the retirement task may already have deleted the state),
    // and ~UsdGenSession must never run inline, on an owner lane.
    RetirementQueue().Post([state, engine = std::move(_engine)] {
        state->Close();
        delete state;
    });
}
void UsdGenImagingSession::Shutdown() { _state->Close(); }
void UsdGenImagingSession::DrainRetired() { RetirementQueue().Drain(); }
bool UsdGenImagingSession::SetTime(double frame) {
    auto* state = _state.get();
    return state->owner.PostCommand([state, frame] { state->frame = frame; });
}
bool UsdGenImagingSession::SetContext(usdGen::UsdGenContext context) {
    auto* state = _state.get();
    return state->owner.PostLatestCommand(state->contextMailbox,
                                          [state, context] { state->context = context; });
}
bool UsdGenImagingSession::StageDesc(usdGen::UsdGenGraphDesc const& desc) {
    auto* state = _state.get();
    auto value = std::make_shared<const usdGen::UsdGenGraphDesc>(desc);
    return state->owner.PostCommand([state, value] { state->staged = value; });
}
uint64_t UsdGenImagingSession::CancelCooks() noexcept {
    return _state->cookToken.fetch_add(1, std::memory_order_acq_rel) + 1;
}
uint64_t UsdGenImagingSession::CookToken() const noexcept {
    return _state->cookToken.load(std::memory_order_acquire);
}
bool UsdGenImagingSession::CommitAsync(CommitRequest request, Completion done,
                                       ProgressCallback progress) {
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    // Stamp the request with the token it was accepted under, so a
    // CancelCooks between here and the owner frame drops it (plan/18 G7).
    if (!request.cookToken) {
        request.cookToken = state->cookToken.load(std::memory_order_acquire);
    }
    const int device = request.callerDevice ? *request.callerDevice :
        usdGen::UsdGenSession::CaptureCallerDevice();
    return state->owner.PostCommand([state, request=std::move(request), device,
                                     done, progress=std::move(progress)] {
        state->Start(request, device, done, progress);
    }, [done] { if (done) done({}, Pipeline::Outcome::Superseded); });
}
void UsdGenImagingSession::_Commit(CommitRequest request) {
    _state->owner.Await([this, request=std::move(request)](std::function<void()> release) mutable {
        if (!CommitAsync(std::move(request), [release](CommitPayload const&, Pipeline::Outcome) {
            release();
        })) throw std::runtime_error("imaging session rejected commit");
    });
}
void UsdGenImagingSession::Commit(usdGen::UsdGenCommitReason reason) {
    CommitRequest request; request.reason = reason; _Commit(std::move(request));
}
void UsdGenImagingSession::CommitAtTime(double frame, usdGen::UsdGenCommitReason reason) {
    CommitRequest request; request.reason = reason; request.frame = frame; _Commit(std::move(request));
}
void UsdGenImagingSession::StageAndCommit(usdGen::UsdGenGraphDesc const& desc, usdGen::UsdGenCommitReason reason) {
    CommitRequest request; request.reason = reason;
    request.desc = std::make_shared<const usdGen::UsdGenGraphDesc>(desc);
    _Commit(std::move(request));
}
int UsdGenImagingSession::RegisterRepublishCallback(
    std::function<void(CommitPayload const&)> callback, ProgressCallback progress) {
    auto* state = _state.get();
    const auto token = state->nextToken.fetch_add(1);
    if (token > static_cast<uint64_t>(std::numeric_limits<int>::max()) || !state->accepting.load()) return -1;
    if (!state->owner.PostCommand([state, token, callback=std::move(callback),
                                   progress=std::move(progress)] {
        state->callbacks.push_back({static_cast<int>(token), callback, progress});
    })) return -1;
    return static_cast<int>(token);
}
int UsdGenImagingSession::RegisterRepublishCallback(
    Pipeline::CommandTicket&& ticket,
    std::function<void(CommitPayload const&)> callback,
    std::function<void()> registered, ProgressCallback progress) {
    auto* state = _state.get();
    const auto token = state->nextToken.fetch_add(1);
    if (token > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        !state->accepting.load(std::memory_order_acquire)) return -1;
    auto cancelled = registered;
    if (!state->owner.PostCommand(std::move(ticket),
        [state, token, callback=std::move(callback), registered=std::move(registered),
         progress=std::move(progress)] {
            state->callbacks.push_back({static_cast<int>(token), callback, progress});
            if (registered) registered();
        }, [cancelled=std::move(cancelled)] {
            if (cancelled) cancelled();
        })) return -1;
    return static_cast<int>(token);
}
bool UsdGenImagingSession::UnregisterRepublishCallbackAsync(
    int token, std::function<void()> completion) {
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    auto cancelled = completion;
    return state->owner.PostCommand([state, token, completion=std::move(completion)] {
        state->callbacks.erase(std::remove_if(state->callbacks.begin(), state->callbacks.end(),
            [token](auto const& entry) { return entry.token == token; }), state->callbacks.end());
        if (completion) completion();
    }, [completion=std::move(cancelled)] {
        if (completion) completion();
    });
}
Pipeline::CommandTicket UsdGenImagingSession::ReserveLifecycleCommand()
{
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return {};
    return state->owner.ReserveCommandTicket();
}
bool UsdGenImagingSession::UnregisterRepublishCallbackAsync(
    Pipeline::CommandTicket&& ticket, int token, std::function<void()> completion) {
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return false;
    auto cancelled = completion;
    return state->owner.PostCommand(std::move(ticket),
        [state, token, completion=std::move(completion)] {
            state->callbacks.erase(std::remove_if(state->callbacks.begin(), state->callbacks.end(),
                [token](auto const& entry) { return entry.token == token; }), state->callbacks.end());
            if (completion) completion();
        }, [completion=std::move(cancelled)] {
            if (completion) completion();
        });
}
bool UsdGenImagingSession::UnregisterRepublishCallback(int token) {
    return UnregisterRepublishCallbackAsync(token);
}
int64_t UsdGenImagingSession::Generation() const noexcept { return _state->generation.load(); }
usdGen::UsdGenGenerationConstPtr UsdGenImagingSession::LatestGeneration() const noexcept {
    return _engine ? _engine->Generation() : usdGen::UsdGenGenerationConstPtr();
}
int UsdGenImagingSession::AttachedIndices() const noexcept { return _state->attached.load(); }
void UsdGenImagingSession::NoteAttach() { _state->attached.fetch_add(1); }
void UsdGenImagingSession::NoteDetach() { _state->attached.fetch_sub(1); }
void UsdGenImagingSession::MarkAppDriver() noexcept { _state->appDriver.store(true); }
bool UsdGenImagingSession::HasAppDriver() const noexcept { return _state->appDriver.load(); }
void UsdGenImagingSession::MarkNeedsDesc() noexcept { _state->needsDesc.store(true); }
bool UsdGenImagingSession::NeedsDesc() const noexcept { return _state->needsDesc.load(); }
bool UsdGenImagingSession::ConsumeNeedsDesc() noexcept { return _state->needsDesc.exchange(false); }
bool UsdGenImagingSession::ReloadMaps() {
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return false;
    return state->owner.PostCommand([state] {
        if (!state->currentDesc) {
            state->needsDesc.store(true, std::memory_order_release);
            return;
        }
        try {
            auto resolved = std::make_shared<usdGen::UsdGenGraphDesc>(
                *state->currentDesc);
            ResolveUsdGenImageMaps(resolved.get());
            state->currentDesc = resolved;
            state->staged = std::move(resolved);
            state->needsDesc.store(false, std::memory_order_release);
        } catch (...) {
            state->needsDesc.store(true, std::memory_order_release);
            TF_WARN("usdGen image-map payload reload could not be staged");
        }
    });
}

// ---------------------------------------------------------------------------

namespace {
void StartCommitBatch(std::vector<UsdGenSessionHandle> sessions,
    usdGen::UsdGenCommitReason reason, std::optional<double> frame,
    std::optional<usdGen::UsdGenContext> context, int device,
    std::function<void()> completion, std::function<void()> retired)
{
    if (sessions.empty()) {
        try { if (completion) completion(); }
        catch (...) { TF_WARN("usdGen store batch completion threw"); }
        retired();
        return;
    }
    struct Batch {
        std::vector<UsdGenSessionHandle> sessions;
        std::atomic<size_t> remaining;
        std::function<void()> done;
        std::function<void()> retired;
        Batch(std::vector<UsdGenSessionHandle> input, std::function<void()> callback,
              std::function<void()> release)
            : sessions(std::move(input)), remaining(sessions.size()),
              done(std::move(callback)), retired(std::move(release)) {}
        ~Batch() {
            // Teardown waits for ownership release, not merely for the user
            // completion. Last public handles must enqueue their retirement
            // before the process-global store can finish shutting down.
            done = {};
            sessions.clear();
            retired();
        }
        void Complete() {
            if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1 && done) {
                try { done(); }
                catch (...) { TF_WARN("usdGen store batch completion threw"); }
            }
        }
    };
    auto batch = std::make_shared<Batch>(std::move(sessions), std::move(completion), std::move(retired));
    // Hold every handle through its reply, including after registry removal.
    // This command submits all descriptions without waiting for any cook.
    for (auto const& session : batch->sessions) {
        UsdGenImagingSession::CommitRequest request;
        request.frame = frame;
        request.context = context;
        request.reason = reason;
        request.callerDevice = device;
        try {
            if (session->CommitAsync(std::move(request),
                [batch](UsdGenImagingSession::CommitPayload const&, Pipeline::Outcome) {
                    batch->Complete();
                })) continue;
        } catch (...) { TF_WARN("usdGen imaging batch request could not be dispatched"); }
        batch->Complete();
    }
}
}

struct UsdGenSessionStore::State {
    using Members = std::unordered_map<UsdGenSessionKey, UsdGenSessionHandle,
                                       UsdGenSessionKeyHash>;
    usdGen::UsdGenSharedCell<const Members> published{std::make_shared<const Members>()};
    std::optional<double> frame;
    std::optional<usdGen::UsdGenContext> context;
    std::atomic<bool> accepting{true};
    size_t outstandingBatches = 0;
    std::function<void()> closeDone;
    Pipeline owner; // destroyed/drained before membership handles are released
    Pipeline::CommandTicket closeTicket;

    State() : owner(ImagingRuntime()), closeTicket(owner.ReserveCommandTicket()) {
        if (!closeTicket) throw std::runtime_error("store close admission unavailable");
    }
    ~State() {
        accepting.store(false);
        owner.Await([this](std::function<void()> done) {
            if (!owner.PostCommand(std::move(closeTicket),
                [this, done] { closeDone = done; CompleteClose(); }))
                throw std::runtime_error("store close command rejected");
        });
        owner.Shutdown();
    }
    void CompleteClose() {
        if (outstandingBatches == 0 && closeDone) {
            auto done = std::move(closeDone);
            done();
        }
    }
    void StartBatch(std::vector<UsdGenSessionHandle> sessions,
                    usdGen::UsdGenCommitReason reason, int device,
                    std::function<void()> completion) {
        auto retiredTicket = std::make_shared<Pipeline::CommandTicket>(
            owner.ReserveCommandTicket());
        if (!*retiredTicket) {
            TF_WARN("usdGen store batch completion admission is full");
            if (completion) completion();
            return;
        }
        ++outstandingBatches;
        try {
            StartCommitBatch(std::move(sessions), reason, frame, context, device, completion,
                [this, retiredTicket]() mutable {
                    if (!owner.PostCommand(std::move(*retiredTicket), [this] {
                        --outstandingBatches; CompleteClose();
                    }, [this] { --outstandingBatches; CompleteClose(); }))
                        std::terminate();
                });
        } catch (...) {
            --outstandingBatches;
            CompleteClose();
            throw; // preparation failed before accepting session work
        }
    }
    std::shared_ptr<const Members> Snapshot() const { return published.Load(); }
    UsdGenSessionHandle Attach(UsdGenSessionKey const& key) {
        auto current = Snapshot();
        auto found = current->find(key);
        if (found != current->end()) {
            found->second->NoteAttach();
            return found->second;
        }
        auto next = std::make_shared<Members>(*current);
        auto session = TfCreateRefPtr(new UsdGenImagingSession(
            key, std::make_shared<usdGen::UsdGenSession>(), frame.value_or(0.0),
            context.value_or(usdGen::UsdGenContext::Interactive)));
        if (frame) session->MarkAppDriver();
        next->emplace(key, session);
        session->NoteAttach();
        published.Store(std::move(next));
        return session;
    }
    void Detach(UsdGenSessionKey const& key, UsdGenSessionHandle const& expected) {
        // An empty identity is never a wildcard. Key-only adapters resolve
        // identity when the request is made, before it can be delayed.
        if (!expected) return;
        auto current = Snapshot();
        auto found = current->find(key);
        if (found == current->end() || found->second != expected) return;
        if (expected->AttachedIndices() > 1) {
            expected->NoteDetach();
            return;
        }
        // Allocate before changing counts; a failed copy leaves membership
        // and its attachment count unchanged.
        auto next = std::make_shared<Members>(*current);
        next->erase(key);
        expected->NoteDetach();
        published.Store(std::move(next));
    }
};

UsdGenSessionStore::UsdGenSessionStore() : _state(new State) {}
UsdGenSessionStore::~UsdGenSessionStore() = default;

namespace {
std::atomic<UsdGenSessionStore*> s_storeInstance{nullptr};
}

UsdGenSessionStore &UsdGenSessionStore::GetInstance()
{
    // The store can own the last public handles until process teardown.
    // Construct retirement first so its graph outlives the store. Both are
    // leaked (see ImagingRuntime): the store destructor's close Await cannot
    // complete during DLL_PROCESS_DETACH, where no worker thread remains.
    (void)RetirementQueue();
    return usdGen::UsdGenLazyLeaked(s_storeInstance, [] { return new UsdGenSessionStore; });
}

UsdGenSessionHandle UsdGenSessionStore::Attach(UsdGenSessionKey const &key)
{
    UsdGenSessionHandle session;
    _state->owner.InvokeOwner([this, key, &session] { session = _state->Attach(key); });
    return session;
}

bool UsdGenSessionStore::AttachAsync(UsdGenSessionKey key,
    std::function<void(UsdGenSessionHandle)> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    return state->owner.PostCommand([state, key=std::move(key), completion] {
        UsdGenSessionHandle session;
        try { session = state->Attach(key); }
        catch (...) { TF_WARN("usdGen session attachment failed"); }
        if (completion) completion(std::move(session));
    }, [completion] { if (completion) completion({}); });
}
bool UsdGenSessionStore::AttachAsync(Pipeline::CommandTicket&& ticket,
    UsdGenSessionKey key, std::function<void(UsdGenSessionHandle)> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return false;
    return state->owner.PostCommand(std::move(ticket),
        [state, key=std::move(key), completion] {
            UsdGenSessionHandle session;
            try { session = state->Attach(key); }
            catch (...) { TF_WARN("usdGen session attachment failed"); }
            if (completion) completion(std::move(session));
        }, [completion] { if (completion) completion({}); });
}

void UsdGenSessionStore::Detach(UsdGenSessionKey const &key)
{
    Detach(key, Find(key));
}

void UsdGenSessionStore::Detach(UsdGenSessionKey const& key,
                               UsdGenSessionHandle const& expected)
{
    _state->owner.InvokeOwner([this, key, expected] { _state->Detach(key, expected); });
}

bool UsdGenSessionStore::DetachAsync(UsdGenSessionKey key, std::function<void()> completion)
{
    auto expected = Find(key);
    return DetachAsync(std::move(key), std::move(expected), std::move(completion));
}

bool UsdGenSessionStore::DetachAsync(UsdGenSessionKey key, UsdGenSessionHandle expected,
                                    std::function<void()> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    return state->owner.PostCommand([state, key=std::move(key), expected, completion] {
        try { state->Detach(key, expected); }
        catch (...) { TF_WARN("usdGen session detachment failed"); }
        if (completion) completion();
    }, [completion] { if (completion) completion(); });
}
Pipeline::CommandTicket UsdGenSessionStore::ReserveLifecycleCommand()
{
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return {};
    return state->owner.ReserveCommandTicket();
}
bool UsdGenSessionStore::DetachAsync(Pipeline::CommandTicket&& ticket,
    UsdGenSessionKey key, UsdGenSessionHandle expected, std::function<void()> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load(std::memory_order_acquire)) return false;
    auto cancelled = completion;
    return state->owner.PostCommand(std::move(ticket),
        [state, key=std::move(key), expected, completion=std::move(completion)] {
            try { state->Detach(key, expected); }
            catch (...) { TF_WARN("usdGen session detachment failed"); }
            if (completion) completion();
        }, [completion=std::move(cancelled)] {
            if (completion) completion();
        });
}

UsdGenSessionHandle UsdGenSessionStore::Find(UsdGenSessionKey const &key) const
{
    auto snapshot = _state->Snapshot();
    auto it = snapshot->find(key);
    if (it != snapshot->end()) return it->second;
    return UsdGenSessionHandle();
}

void UsdGenSessionStore::Drain() { _state->owner.Drain(); }

std::vector<UsdGenImagingSessionRefPtr> UsdGenSessionStore::LiveSessions() const
{
    std::vector<UsdGenImagingSessionRefPtr> out;
    auto snapshot = _state->Snapshot();
    out.reserve(snapshot->size());
    for (auto const &entry : *snapshot) {
        out.push_back(entry.second);
    }
    return out;
}

size_t UsdGenSessionStore::CancelCooks(SdfPath const &groomRoot)
{
    // No owner command and no wait: a gesture's press handler calls this,
    // and the token is an atomic the owner frame reads (plan/18 §7 G7).
    size_t moved = 0;
    for (auto const &session : LiveSessions()) {
        if (!groomRoot.IsEmpty() && session->Key().groomRoot != groomRoot) {
            continue;
        }
        (void)session->CancelCooks();
        ++moved;
    }
    return moved;
}

void UsdGenSessionStore::SetTime(double frame)
{
    _state->owner.Await([&](std::function<void()> done) {
        if (!SetTimeAsync(frame, std::move(done)))
            throw std::runtime_error("store rejected frame request");
    });
}

bool UsdGenSessionStore::SetTimeAsync(double frame, std::function<void()> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    const int device = usdGen::UsdGenSession::CaptureCallerDevice();
    return state->owner.PostCommand([this, state, frame, device, completion] {
        try {
            state->frame = frame;
            auto sessions = LiveSessions();
            for (auto const& session : sessions) session->MarkAppDriver();
            state->StartBatch(std::move(sessions), usdGen::UsdGenCommitReason::SetTime,
                              device, completion);
        } catch (...) {
            TF_WARN("usdGen store frame batch could not be prepared");
            if (completion) completion();
        }
    }, completion);
}

bool UsdGenSessionStore::SetContext(usdGen::UsdGenContext context)
{
    auto* state = _state.get();
    return state->owner.PostCommand([state, context] {
        state->context = context;
        for (auto const& entry : *state->Snapshot()) {
            // Each session has a permanent latest-wins context mailbox. A
            // false return therefore means only shutdown/framework failure,
            // never ordinary command pressure; do not silently claim that
            // the registry-wide current context was applied.
            if (!entry.second->SetContext(context))
                throw std::runtime_error("session context mailbox rejected");
        }
    });
}

void UsdGenSessionStore::Commit(usdGen::UsdGenCommitReason reason)
{
    _state->owner.Await([&](std::function<void()> done) {
        if (!CommitAsync(reason, std::move(done)))
            throw std::runtime_error("store rejected commit request");
    });
}

bool UsdGenSessionStore::CommitAsync(usdGen::UsdGenCommitReason reason,
                                   std::function<void()> completion)
{
    auto* state = _state.get();
    if (!state->accepting.load()) return false;
    const int device = usdGen::UsdGenSession::CaptureCallerDevice();
    return state->owner.PostCommand([this, state, reason, device, completion] {
        try { state->StartBatch(LiveSessions(), reason, device, completion); }
        catch (...) {
            TF_WARN("usdGen store commit batch could not be prepared");
            if (completion) completion();
        }
    }, completion);
}

int64_t UsdGenSessionStore::Generation() const noexcept
{
    int64_t gen = -1;
    for (auto const &session : LiveSessions()) {
        gen = std::max(gen, session->Generation());
    }
    return gen;
}

bool UsdGenSessionStore::ReloadMaps()
{
    auto* state = _state.get();
    auto sessions = LiveSessions();
    return state->owner.PostCommand([sessions=std::move(sessions)] {
        (void)InvalidateUsdGenImageMapCache();
        // Ptex files are read through their own process cache; a reload must
        // re-read them too (the new textureGeneration reopens every map).
        usdGen::UsdGenPtexTexture::PurgeCache();
        for (auto const& session : sessions) {
            if (!session->ReloadMaps()) session->MarkNeedsDesc();
        }
    });
}

}  // namespace usdGenImaging
