// usdGen imaging — session key + session + process-global session store
// (06-imaging.md §3.7; ADR §4.5 "sessions across chains").
//
// Stage-free key (plan 13 §7): a nonempty usdGen:sessionId explicitly
// shares a CPU groom session across renderer chains. Device/GL publication
// sets rendererLocal, retaining the renderer instance in the key because its
// CUDA/GL resources cannot be shared across registries. Without an id, every
// groom is scoped to its render instance. No stage registry is consulted.
//
// The store holds the session alive (strong refs); scene indices hold weak
// handles and re-attach on construction (S15). A renderer switch therefore
// re-attaches to the same session and continues its generation counter.
#ifndef USDGEN_IMAGING_SESSION_H
#define USDGEN_IMAGING_SESSION_H

#include "usdGenImaging/api.h"
#include "usdGenImaging/export.h"
#include "usdGen/api/session.h"
#include "usdGenUsd/lower.h"
#include "usdGen/api/types.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/refBase.h"
#include "pxr/base/tf/weakPtr.h"
#include "pxr/usd/sdf/path.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

struct UsdGenSessionKey
{
    std::string     sessionId;  // explicit cross-renderer sharing identity
    SdfPath         groomRoot;  // the UsdGenGroom prim path this session serves
    uint64_t        renderInstanceId = 0; // used only without sessionId
    // CUDA/GL device generations own renderer-registry resources and must
    // never cross a renderer boundary, even when an authored sessionId would
    // otherwise request CPU session sharing. Appended for aggregate callers.
    bool            rendererLocal = false;

    bool operator==(UsdGenSessionKey const &rhs) const
    {
        if (sessionId != rhs.sessionId || groomRoot != rhs.groomRoot ||
            rendererLocal != rhs.rendererLocal) return false;
        // An authored CPU session is deliberately shareable.  Anonymous and
        // renderer-local sessions both include their renderer identity.
        return (!sessionId.empty() && !rendererLocal) ||
            renderInstanceId == rhs.renderInstanceId;
    }
};

struct UsdGenSessionKeyHash
{
    std::size_t operator()(UsdGenSessionKey const &k) const
    {
        const std::size_t h0 = (k.sessionId.empty() || k.rendererLocal)
            ? std::hash<uint64_t>()(k.renderInstanceId) : 0;
        const std::size_t h1 = std::hash<std::string>()(k.sessionId);
        const std::size_t h2 = std::hash<std::string>()(k.groomRoot.GetText());
        const std::size_t h3 = std::hash<bool>()(k.rendererLocal);
        return h0 ^ (h1 * 0x9E3779B97F4A7C15ull) ^ (h2 << 1 | h2 >> 31) ^
            (h3 * 0x85EBCA6Bull);
    }
};

/// One usdGen session: owns the engine session, the current frame/context,
/// and the publication generation counter. Several scene indices may attach
/// to one session (renderer switch, Storm + hdPrman preflight in one
/// process); the session owns ONE current frame, ONE context, ONE
/// generation — each attached index republishes the same generation. The
/// complete-generation prim set is a session property and never differs per
/// index. Opted-in progressive viewports can show provisional tiles while a
/// complete generation is still cooking.
class USDGEN_HYDRA_API UsdGenImagingSession : public TfRefBase, public TfWeakBase
{
public:
    UsdGenImagingSession(
        UsdGenSessionKey const &key,
        std::shared_ptr<usdGen::UsdGenSession> engineSession,
        double frame = 0.0,
        usdGen::UsdGenContext context = usdGen::UsdGenContext::Interactive,
        uint64_t commandCapacity = 4096);
    ~UsdGenImagingSession() override;

    UsdGenSessionKey const &Key() const { return _key; }
    usdGen::UsdGenSession *Engine() const { return _engine.get(); }

    /// Trigger (a)/(b) time source (06 §3.9). Frame-only: never commits.
    bool SetTime(double frame);
    /// ADR §2.3 explicit context (session property, shared by all indices).
    bool SetContext(usdGen::UsdGenContext context);
    /// Trigger (a) explicit commit (after a live-override change) and
    /// trigger (c) end-of-batch commit (06 §3.9, R32: unconditional).
    /// Publishes a generation only when the session is dirty or a graph desc
    /// was staged; the publication generation advances in lockstep with the
    /// engine's, so an app-driven commit reaches the app through the
    /// republish callbacks.
    void Commit(usdGen::UsdGenCommitReason reason);
    /// Install a freshly pulled descriptor and evaluate it as one command.
    /// Unlike separate StageDesc/Commit calls, no store-driven commit can
    /// consume a different descriptor between those operations.
    void StageAndCommit(usdGen::UsdGenGraphDesc const &desc,
                        usdGen::UsdGenCommitReason reason);
    void CommitAtTime(double frame, usdGen::UsdGenCommitReason reason);

    /// 06 §3.9 rule (b): once an external SetTime has driven this session,
    /// frame dirties never trigger commits (the app owns time). Set by the
    /// store-level SetTime only; scene indices never set it.
    void MarkAppDriver() noexcept;
    bool HasAppDriver() const noexcept;

    /// The staged UsdGenGraphDesc (built by the owning scene index) is to be
    /// consumed by the next Commit. Set on structural changes and map
    /// reloads (S13); never on frame-only dirties (S14: one desc pull per
    /// topology generation, not per frame).
    void MarkNeedsDesc() noexcept;
    bool NeedsDesc() const noexcept;
    /// Rebuilds the retained descriptor's immutable map payloads against the
    /// current imaging decode generation and stages it for the next commit.
    bool ReloadMaps();

    /// Immutable per-commit payload for republish callbacks (P0 races):
    /// the generation published by THIS commit plus a COPY of its dirty
    /// report. Callbacks must never reread live Engine()->Generation() /
    /// LastReport() — a concurrent commit can replace them mid-iteration
    /// (report/generation pairing + data race). Empty published==false
    /// normally means "nothing published, no republish"; a nonzero
    /// progressEpoch is the terminal rollback exception for provisional tiles.
    struct CommitPayload {
        bool published = false;
        // Nonzero when this attempt exposed provisional tiles. Terminal
        // callbacks also run for failed/superseded attempts in that case so
        // subscribers can restore their last complete scene snapshot.
        uint64_t progressEpoch = 0;
        usdGen::UsdGenGenerationConstPtr generation;
        usdGen::UsdGenRef<const usdGen::UsdGenPublication> report;   // the publisher's diff
        std::shared_ptr<const usdGen::UsdGenGraphRoutingSnapshot> routing;
        usdGen::UsdGenDiagnostics diagnostics;
        usdGen::UsdGenStats stats;
    };

    struct CommitRequest {
        usdGen::UsdGenCommitReason reason = usdGen::UsdGenCommitReason::NoticeBatchEnd;
        std::optional<double> frame;
        std::optional<usdGen::UsdGenContext> context;
        std::shared_ptr<const usdGen::UsdGenGraphDesc> desc;
        // Value-only dirt a host router resolved (UsdGenDirtyRouter::Route);
        // posted to the engine ahead of the commit.
        usdGen::UsdGenPendingDirty dirty;
        // Upstream owner relays preserve the original CUDA caller affinity.
        std::optional<int> callerDevice;
        // Relayed atomically to the core commit owner.  Absent preserves the
        // core session's existing renderer-admission selection.
        std::optional<bool> devicePublication;
        // Cook cancellation token (plan/17 §3.2 rule 2, plan/18 §7 G7).
        // Stamped by CommitAsync from the session's token when absent; the
        // owner frame drops the request as Superseded, without touching the
        // engine, if CancelCooks() has moved the token since. This is how an
        // authoring tool abandons a cook its own newer edit has invalidated.
        std::optional<uint64_t> cookToken;
        // Scene indices set this from Hydra's asyncAllow handshake. Offline
        // and app-driven callers retain the default for interested viewers.
        bool progressive = true;
    };
    using Completion = std::function<void(CommitPayload const&,
        usdGen::UsdGenExecutionPipeline::Outcome)>;
    using TileProgress = usdGen::UsdGenSession::TileProgress;
    using TileProgressPtr = std::shared_ptr<const TileProgress>;
    struct ProgressPayload {
        uint64_t epoch = 0;
        uint64_t sequence = 0;
        double frame = 0.0;
        // Cumulative pointer set makes latest-wins owner mailboxes lossless
        // without copying tile geometry for each completed chunk.
        std::shared_ptr<const std::vector<TileProgressPtr>> tiles;
    };
    using ProgressCallback = std::function<void(ProgressPayload const&)>;
    bool CommitAsync(CommitRequest request, Completion completion = {},
                     ProgressCallback progress = {});
    // Single external boundary: finish accepted requests/callbacks, then stop.
    // Concurrent Shutdown callers are rejected, not made to wait on each other. Never
    // call from a pipeline callback. Destruction otherwise retires state on
    // the framework, so accepted completion captures must own their lifetimes.
    void Shutdown();
    // Single external draining caller; no callback may wait for retirement.
    static void DrainRetired();

    /// Atomic desc-dirty consume for staging (S14): returns true and clears
    /// the flag when a desc pull is owed. Marks arriving after consumption
    /// stay set for the next commit (no lost ReloadMaps/structural marks).
    bool ConsumeNeedsDesc() noexcept;

    /// Install a descriptor without committing (offline/test clients).
    /// Production staging uses StageAndCommit for an atomic pair.
    bool StageDesc(usdGen::UsdGenGraphDesc const &desc);

    /// Monotonic publication generation (UsdGenImaging_GetGeneration).
    int64_t Generation() const noexcept;
    /// Latest published engine generation (lock-free atomic load, I7).
    usdGen::UsdGenGenerationConstPtr LatestGeneration() const noexcept;

    /// Invalidate every cook accepted before now (plan/17 §3.2 rule 2).
    /// Requests already handed to the engine keep the engine's own
    /// supersede behaviour; requests still queued on this session's owner
    /// are dropped before they reach it. Returns the new token. Callable
    /// from any thread, including a gesture's press handler.
    uint64_t CancelCooks() noexcept;
    /// The current token: a request stamped with a smaller one is stale.
    uint64_t CookToken() const noexcept;

    int AttachedIndices() const noexcept;
    void NoteAttach();
    void NoteDetach();
    /// Republish hooks (06 §3.7 "each index republishes the same
    /// generation"): invoked after a Commit that published, on the scheduled
    /// command owner, with THIS commit's immutable payload. Callbacks must not
    /// block on GetPrim and must not reread live engine state.
    int RegisterRepublishCallback(
        std::function<void(CommitPayload const &)> cb,
        ProgressCallback progress = {});
    int RegisterRepublishCallback(
        usdGen::UsdGenExecutionPipeline::CommandTicket&& ticket,
        std::function<void(CommitPayload const &)> cb,
        std::function<void()> registered = {},
        ProgressCallback progress = {});
    /// Enqueues callback removal and acknowledges after the owner has erased
    /// it. If the accepted command is cancelled during shutdown, completion
    /// is still invoked; false means it was never accepted and completion is
    /// not invoked. Completion must not synchronously wait on an owner.
    bool UnregisterRepublishCallbackAsync(
        int token, std::function<void()> completion = {});
    // A lifetime relay reserves this session owner's cleanup admission before
    // accepting an external source callback. The ticket overload consumes
    // that exact credit; false means cleanup was never accepted.
    usdGen::UsdGenExecutionPipeline::CommandTicket ReserveLifecycleCommand();
    bool UnregisterRepublishCallbackAsync(
        usdGen::UsdGenExecutionPipeline::CommandTicket&& ticket, int token,
        std::function<void()> completion = {});
    // False means the ordinary owner lane was full/closed and the callback
    // remains registered. Lifetime owners must reserve cleanup admission
    // before accepting a callback source and use the ticket overload above.
    bool UnregisterRepublishCallback(int token);
    // Registration/removal enqueue nonblocking owner commands. A currently
    // executing callback may finish; later publications observe the removal.
private:
    UsdGenSessionKey        _key;
    std::shared_ptr<usdGen::UsdGenSession> _engine;
    struct State;
    std::unique_ptr<State> _state;
    void _Commit(CommitRequest request);
};

using UsdGenImagingSessionRefPtr = TfRefPtr<UsdGenImagingSession>;
using UsdGenSessionHandle = UsdGenImagingSessionRefPtr;

/// Process-global session store (06 §3.7).
class USDGEN_HYDRA_API UsdGenSessionStore
{
public:
    static UsdGenSessionStore &GetInstance();
    ~UsdGenSessionStore();

    /// Primary mutation APIs. Completion runs on the store command owner;
    /// it may enqueue follow-up commands, but must not synchronously wait.
    /// Attach reports an empty handle on failure/cancellation. False means
    /// the request was not accepted and no completion will be delivered.
    bool AttachAsync(UsdGenSessionKey key,
                     std::function<void(UsdGenSessionHandle)> completion);
    bool AttachAsync(usdGen::UsdGenExecutionPipeline::CommandTicket&& ticket,
                     UsdGenSessionKey key,
                     std::function<void(UsdGenSessionHandle)> completion);
    /// Key-only compatibility adapter resolves the currently published
    /// identity, not a pending AttachAsync. Prefer the exact-handle overload.
    bool DetachAsync(UsdGenSessionKey key, std::function<void()> completion = {});
    /// Identity-aware release: a delayed release cannot touch a same-key
    /// replacement. Each successful Attach still requires exactly one release.
    bool DetachAsync(UsdGenSessionKey key, UsdGenSessionHandle expected,
                     std::function<void()> completion = {});
    // Reserve store-owner cleanup admission before a source lifetime is
    // accepted. Use the ticket overload for the matching terminal detach.
    usdGen::UsdGenExecutionPipeline::CommandTicket ReserveLifecycleCommand();
    bool DetachAsync(usdGen::UsdGenExecutionPipeline::CommandTicket&& ticket,
                     UsdGenSessionKey key, UsdGenSessionHandle expected,
                     std::function<void()> completion = {});
    /// Batch dispatch is one store command; independent sessions execute in
    /// parallel. Completion runs as final imaging reply owner work (or store
    /// owner work for an empty/rejected batch), with no OS-thread affinity.
    /// External cooperative waits may execute framework work on their caller.
    bool SetTimeAsync(double frame, std::function<void()> completion = {});
    bool CommitAsync(usdGen::UsdGenCommitReason reason,
                     std::function<void()> completion = {});

    /// Find-or-create the session for \p key; increments its attach count.
    /// The engine session is created with the thread limit resolved by the
    /// engine (USDGEN_THREAD_LIMIT / one-shot calibration).
    UsdGenSessionHandle Attach(UsdGenSessionKey const &key);
    /// Decrements the attach count; drops the store's strong reference at
    /// zero so the session is destroyed once every index detaches.
    void Detach(UsdGenSessionKey const &key);
    void Detach(UsdGenSessionKey const &key, UsdGenSessionHandle const &expected);
    UsdGenSessionHandle Find(UsdGenSessionKey const &key) const;

    /// CancelCooks on every live session whose groomRoot is \p groomRoot,
    /// or on all of them when it is empty (plan/17 §3.2 rule 2). Returns
    /// the number of sessions whose token moved.
    size_t CancelCooks(SdfPath const &groomRoot = SdfPath());

    /// Registry-level forwarding (06 §3.7): applied to every live session.
    /// SetTime additionally marks every session app-driven (06 §3.9 rule b)
    /// and commits it at the new frame — the future UsdGenImaging_SetTime
    /// C ABI (06 §6.1) forwards here.
    void SetTime(double frame);
    bool SetContext(usdGen::UsdGenContext context);
    void Commit(usdGen::UsdGenCommitReason reason);
    int64_t Generation() const noexcept;  // max over live sessions
    /// S13: invalidate the shared decoded-image COW cache and stage a newly
    /// resolved payload generation in every live session.
    bool ReloadMaps();

    /// Immutable membership snapshot (strong refs). These reads never enter
    /// or await the command owner; returned handles survive later detachment.
    std::vector<UsdGenImagingSessionRefPtr> LiveSessions() const;
    /// Single external shutdown/test boundary after admission has quiesced.
    /// Drains store command frames, not independent session cook work.
    void Drain();

private:
    UsdGenSessionStore();
    UsdGenSessionStore(const UsdGenSessionStore &) = delete;
    UsdGenSessionStore &operator=(const UsdGenSessionStore &) = delete;

    struct State;
    std::unique_ptr<State> _state;
};

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_SESSION_H