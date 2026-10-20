// testUsdGenAsyncSession — scheduled session ownership and immutable
// publication snapshots.  The held Capture operator makes owner/work
// separation observable without a timing-dependent sleep.

#include "usdGen/opRegistry.h"
#include "usdGen/session.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

struct HoldCaptureState {
    static std::atomic<bool> hold;
    static std::atomic<bool> entered;
    static std::atomic<bool> release;
    // The operator's capture digest. A new descriptor recompiles
    // incrementally and keeps the capture of an operator it did not change,
    // so a test that wants this one held again moves its digest.
    static std::atomic<uint64_t> epoch;
};
std::atomic<bool> HoldCaptureState::hold{false};
std::atomic<bool> HoldCaptureState::entered{false};
std::atomic<bool> HoldCaptureState::release{true};
std::atomic<uint64_t> HoldCaptureState::epoch{1};

class HoldCaptureOp final : public UsdGenOp {
public:
    TfToken Type() const override { return TfToken("UsdGenAsyncHold"); }
    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    bool Bind(UsdGenParamView const &, UsdGenDiagnostics *) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const &) const override {
        return {1, HoldCaptureState::epoch.load(std::memory_order_acquire)};
    }
    bool Capture(UsdGenCaptureContext const &, UsdGenCurveBuffer const &,
                 UsdGenCapture *, UsdGenDiagnostics *diagnostics) override {
        if (HoldCaptureState::hold.load(std::memory_order_acquire)) {
            HoldCaptureState::entered.store(true, std::memory_order_release);
            auto const deadline = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(10);
            while (!HoldCaptureState::release.load(std::memory_order_acquire) &&
                   std::chrono::steady_clock::now() < deadline)
                std::this_thread::yield();
            if (!HoldCaptureState::release.load(std::memory_order_acquire)) {
                if (diagnostics) diagnostics->Error("async Capture gate timed out");
                return false;
            }
        }
        return true;
    }
    void Evaluate(UsdGenEvalContext const &, UsdGenCapture const &,
                  UsdGenChunkView *) const override {}
};

struct HoldTileState {
    static std::atomic<bool> entered;
    static std::atomic<bool> release;
    static std::atomic<bool> fail;
};
std::atomic<bool> HoldTileState::entered{false};
std::atomic<bool> HoldTileState::release{true};
std::atomic<bool> HoldTileState::fail{false};

class HoldTileOp final : public UsdGenOp {
public:
    TfToken Type() const override { return TfToken("UsdGenAsyncTileHold"); }
    TfSpan<const TfToken> TopologyParameters() const override { return {}; }
    TfSpan<const TfToken> ValueParameters() const override { return {}; }
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {1, 1}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenCurveBuffer const&,
                 UsdGenCapture*, UsdGenDiagnostics*) override { return true; }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&,
                  UsdGenChunkView* view) const override {
        if (view->desc->tile != 1) return;
        HoldTileState::entered.store(true, std::memory_order_release);
        auto const deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(10);
        while (!HoldTileState::release.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (HoldTileState::fail.load(std::memory_order_acquire))
            throw std::runtime_error("held terminal tile failed");
    }
};

bool WaitFor(std::atomic<bool> const &value, int milliseconds = 10000)
{
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(milliseconds);
    while (!value.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    return value.load(std::memory_order_acquire);
}

UsdGenGraphDesc MakeDesc(float width)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/async");
    d.terminal = SdfPath("/async/width");

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/async/surface");
    surface.id = 0;
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
        GfVec3f(0, 1, 0), GfVec3f(1, 1, 0)};
    surface.points = surface.restPoints;
    surface.uv = VtVec2fArray{
        GfVec2f(0, 0), GfVec2f(1, 0), GfVec2f(0, 1), GfVec2f(1, 1)};
    surface.faceVertexCounts = VtIntArray{4};
    surface.faceVertexIndices = VtIntArray{0, 1, 3, 2};
    d.surfaces.push_back(std::move(surface));

    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/async/scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {SdfPath("/async/surface")};
    d.nodes.push_back(std::move(scatter));

    UsdGenNodeDesc grow;
    grow.path = SdfPath("/async/grow");
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {SdfPath("/async/scatter")};
    grow.params.push_back({TfToken("segments"), VtValue(4), false});
    grow.params.push_back({TfToken("length"), VtValue(1.0), false});
    d.nodes.push_back(std::move(grow));

    UsdGenNodeDesc hold;
    hold.path = SdfPath("/async/hold");
    hold.type = TfToken("UsdGenAsyncHold");
    hold.inputs = {SdfPath("/async/grow")};
    d.nodes.push_back(std::move(hold));

    UsdGenNodeDesc widthNode;
    widthNode.path = SdfPath("/async/width");
    widthNode.type = TfToken("UsdGenWidth");
    widthNode.inputs = {SdfPath("/async/hold")};
    widthNode.params.push_back({TfToken("width"), VtValue(width), false});
    d.nodes.push_back(std::move(widthNode));
    return d;
}

UsdGenGraphDesc MakeProgressDesc()
{
    UsdGenGraphDesc desc = MakeDesc(0.025f);
    // Six or more chunks on this unit-area surface make tile 1 independently
    // holdable while tile 0 can complete on another arena worker.
    desc.nodes.front().params.push_back(
        {TfToken("density"), VtValue(3000.0), false});
    desc.tileTarget = 32;
    UsdGenNodeDesc tail;
    tail.path = SdfPath("/async/tileHold");
    tail.type = TfToken("UsdGenAsyncTileHold");
    tail.inputs = {desc.terminal};
    desc.terminal = tail.path;
    desc.nodes.push_back(std::move(tail));
    return desc;
}

} // namespace

int main()
{
    // Exercise cold registry initialization via actual simultaneous compiles,
    // before explicit initialization/custom startup registration can hide it.
    std::atomic<unsigned> coldSuccess{0};
    std::vector<std::thread> coldStarts;
    auto coldDesc = MakeDesc(0.01f);
    coldDesc.nodes.erase(coldDesc.nodes.begin() + 2); // no custom Hold kernel yet
    coldDesc.nodes.back().inputs = {SdfPath("/async/grow")};
    for (unsigned i = 0; i != 8; ++i) {
        coldStarts.emplace_back([&, i] {
            UsdGenSession cold;
            cold.SetGraphDesc(coldDesc);
            auto snapshot = cold.CommitSnapshot(i, UsdGenCommitReason::NoticeBatchEnd);
            if (snapshot && snapshot->generation && !snapshot->diagnostics.HasErrors())
                coldSuccess.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& thread : coldStarts) thread.join();
    Check(coldSuccess.load() == 8, "simultaneous cold sessions read complete built-in registry");
    usdGenRegisterM1Operators();
    UsdGenOpRegistry::Get().Register(TfToken("UsdGenAsyncHold"),
                                      [] { return std::make_unique<HoldCaptureOp>(); });
    UsdGenOpRegistry::Get().Register(TfToken("UsdGenAsyncTileHold"),
                                      [] { return std::make_unique<HoldTileOp>(); });

    // A terminal tile publishes before a later tile finishes evaluation.
    // The public full-generation snapshot remains untouched until the cook
    // succeeds, even though a worker delivered an immutable provisional tile.
    {
        UsdGenSession progressive(4);
        progressive.SetGraphDesc(MakeProgressDesc());
        HoldTileState::entered.store(false, std::memory_order_release);
        HoldTileState::release.store(false, std::memory_order_release);
        HoldTileState::fail.store(false, std::memory_order_release);
        std::atomic<bool> firstTile{false}, terminal{false};
        std::atomic<bool> workerCallback{false}, finalStillEmpty{false};
        std::atomic<bool> terminalPublished{false};
        auto const submitter = std::this_thread::get_id();
        UsdGenSession::CommitRequest request;
        request.frame = 1.0;
        bool const accepted = progressive.CommitAsync(std::move(request),
            [&](UsdGenSession::SnapshotPtr snapshot,
                UsdGenExecutionPipeline::Outcome outcome) {
                terminalPublished.store(outcome == UsdGenExecutionPipeline::Outcome::Published &&
                    snapshot && snapshot->generation && snapshot->generation->tiles.size() > 1,
                    std::memory_order_release);
                terminal.store(true, std::memory_order_release);
            },
            [&](UsdGenSession::TileProgressPtr tile) {
                if (!tile || tile->tile.tile != 0) return;
                workerCallback.store(std::this_thread::get_id() != submitter,
                                     std::memory_order_release);
                auto snapshot = progressive.Snapshot();
                finalStillEmpty.store(!snapshot || !snapshot->generation,
                                      std::memory_order_release);
                firstTile.store(true, std::memory_order_release);
            });
        Check(accepted, "progressive CPU cook accepted");
        bool const held = WaitFor(HoldTileState::entered);
        bool const early = WaitFor(firstTile);
        Check(held && early && !terminal.load(std::memory_order_acquire),
              "first tile arrives while a later terminal tile is still evaluating");
        Check(workerCallback.load(std::memory_order_acquire) &&
              finalStillEmpty.load(std::memory_order_acquire),
              "tile callback is off submitter and full snapshot remains atomic");
        HoldTileState::release.store(true, std::memory_order_release);
        Check(WaitFor(terminal) && terminalPublished.load(std::memory_order_acquire),
              "full generation publishes after held tile completes");
        progressive.Drain();
        auto const baseline = progressive.Generation();
        auto animated = MakeProgressDesc();
        for (auto& node : animated.nodes)
            if (node.type == TfToken("UsdGenWidth"))
                node.params.front().value = VtValue(0.04f);
        progressive.SetGraphDesc(animated);
        std::atomic<unsigned> playbackTiles{0};
        terminal.store(false, std::memory_order_release);
        UsdGenSession::CommitRequest playback;
        playback.frame = 2.0;
        Check(progressive.CommitAsync(std::move(playback),
            [&](UsdGenSession::SnapshotPtr, UsdGenExecutionPipeline::Outcome) {
                terminal.store(true, std::memory_order_release);
            }, [&](UsdGenSession::TileProgressPtr) { ++playbackTiles; }),
            "animated value update accepted with progress subscriber");
        Check(WaitFor(terminal), "playback update completes");
        progressive.Drain();
        auto const next = progressive.Generation();
        Check(next && baseline && next->id > baseline->id && next->frame == 2.0 &&
                  next->tiles.size() == baseline->tiles.size() && playbackTiles == 0,
              "frame change publishes existing tiles atomically without progressive redraw");
    }
    {
        UsdGenSession progressive(4);
        progressive.SetGraphDesc(MakeProgressDesc());
        HoldTileState::entered.store(false, std::memory_order_release);
        HoldTileState::release.store(false, std::memory_order_release);
        HoldTileState::fail.store(true, std::memory_order_release);
        std::atomic<bool> firstTile{false}, terminal{false}, failed{false};
        UsdGenSession::CommitRequest request;
        request.frame = 2.0;
        Check(progressive.CommitAsync(std::move(request),
            [&](UsdGenSession::SnapshotPtr snapshot,
                UsdGenExecutionPipeline::Outcome outcome) {
                failed.store(outcome == UsdGenExecutionPipeline::Outcome::Failed &&
                    snapshot && !snapshot->generation,
                    std::memory_order_release);
                terminal.store(true, std::memory_order_release);
            }, [&](UsdGenSession::TileProgressPtr tile) {
                if (tile && tile->tile.tile == 0)
                    firstTile.store(true, std::memory_order_release);
            }), "failing progressive cook accepted");
        bool const early = WaitFor(firstTile);
        bool const held = WaitFor(HoldTileState::entered);
        Check(early && held && !terminal.load(std::memory_order_acquire),
              "provisional tile can precede a later tile failure");
        HoldTileState::release.store(true, std::memory_order_release);
        Check(WaitFor(terminal) && failed.load(std::memory_order_acquire),
              "terminal failure identifies provisional tiles for rollback");
        progressive.Drain();
        HoldTileState::fail.store(false, std::memory_order_release);
    }
    {
        UsdGenSession progressive(4);
        progressive.SetGraphDesc(MakeProgressDesc());
        HoldTileState::entered.store(false, std::memory_order_release);
        HoldTileState::release.store(false, std::memory_order_release);
        std::atomic<bool> firstTile{false}, oldTerminal{false}, newTerminal{false};
        std::atomic<bool> oldSuperseded{false}, newPublished{false};
        UsdGenSession::CommitRequest oldRequest;
        oldRequest.frame = 3.0;
        Check(progressive.CommitAsync(std::move(oldRequest),
            [&](UsdGenSession::SnapshotPtr,
                UsdGenExecutionPipeline::Outcome outcome) {
                oldSuperseded.store(outcome == UsdGenExecutionPipeline::Outcome::Superseded,
                                    std::memory_order_release);
                oldTerminal.store(true, std::memory_order_release);
            }, [&](UsdGenSession::TileProgressPtr tile) {
                if (tile && tile->tile.tile == 0)
                    firstTile.store(true, std::memory_order_release);
            }), "superseded progressive cook accepted");
        bool const early = WaitFor(firstTile);
        bool const held = WaitFor(HoldTileState::entered);
        Check(early && held && !oldTerminal.load(std::memory_order_acquire),
              "superseded attempt emitted provisional tile before replacement");
        auto replacement = MakeProgressDesc();
        replacement.nodes[3].params[0] =
            {TfToken("width"), VtValue(0.03f), false};
        progressive.SetGraphDesc(replacement);
        UsdGenSession::CommitRequest newRequest;
        newRequest.frame = 4.0;
        Check(progressive.CommitAsync(std::move(newRequest),
            [&](UsdGenSession::SnapshotPtr snapshot,
                UsdGenExecutionPipeline::Outcome outcome) {
                newPublished.store(outcome == UsdGenExecutionPipeline::Outcome::Published &&
                    snapshot && snapshot->generation && snapshot->generation->frame == 4.0,
                    std::memory_order_release);
                newTerminal.store(true, std::memory_order_release);
            }), "replacement progressive cook accepted");
        HoldTileState::release.store(true, std::memory_order_release);
        Check(WaitFor(oldTerminal) && oldSuperseded.load(std::memory_order_acquire),
              "superseded outcome identifies old provisional tiles for rollback");
        Check(WaitFor(newTerminal) && newPublished.load(std::memory_order_acquire),
              "replacement alone publishes the full generation");
        progressive.Drain();
    }

    // Seed an actual good publication before exercising supersession and
    // failure retention.  The held operator is disabled for this cook.
    HoldCaptureState::hold.store(false);
    HoldCaptureState::release.store(true);
    // Callback storage must outlive session shutdown, including early-failure
    // exits where accepted work may still be awaiting its terminal callback.
    std::atomic<bool> oldCallback{false}, latestCallback{false};
    std::atomic<bool> callbackDone{false}, callbackRejected{false}, followupAccepted{false};
    std::atomic<bool> failureDone{false}, failureOutcome{false};
    UsdGenSession::SnapshotPtr latestSnapshot;
    UsdGenSession session(4);
    UsdGenGraphDesc baselineDesc = MakeDesc(0.02f);
    session.SetGraphDesc(baselineDesc);
    auto baseline = session.CommitSnapshot(0.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(baseline && baseline->generation && baseline->generation->id == 0,
          "initial scheduled snapshot publishes generation zero");
    if (!baseline || !baseline->generation) {
        std::printf("testUsdGenAsyncSession: FAILED (no baseline)\n");
        return 1;
    }
    Check(baseline && baseline->routing && baseline->graphInfo.NodeCount() == 4,
          "initial snapshot pairs routing and graph metadata");
    Check(baseline && baseline->stats.commits == 1,
          "initial snapshot carries paired commit statistics");
    Check(!session.NeedsCommit(), "accepted initial publication clears dirty summary");

    // A worker is blocked in Capture.  Owner commands must still accept a
    // same-path descriptor replacement and a newer commit request.  The old
    // work is superseded; only the newest descriptor may publish.
    HoldCaptureState::hold.store(true, std::memory_order_release);
    HoldCaptureState::entered.store(false, std::memory_order_release);
    HoldCaptureState::release.store(false, std::memory_order_release);
    HoldCaptureState::epoch.fetch_add(1, std::memory_order_acq_rel);
    UsdGenGraphDesc firstEdit = MakeDesc(0.03f);
    UsdGenGraphDesc latestEdit = MakeDesc(0.07f);
    session.SetGraphDesc(firstEdit);
    Check(true, "first edit owner command accepted");
    bool firstSubmitted = session.CommitAsync(1.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr, UsdGenExecutionPipeline::Outcome outcome) {
            oldCallback.store(outcome == UsdGenExecutionPipeline::Outcome::Superseded,
                              std::memory_order_release);
        });
    Check(firstSubmitted, "blocked edit commit submitted");
    if (!firstSubmitted) {
        HoldCaptureState::release.store(true, std::memory_order_release);
        return 1;
    }
    Check(WaitFor(HoldCaptureState::entered),
          "worker reached bounded Capture gate");
    if (!HoldCaptureState::entered.load(std::memory_order_acquire)) {
        HoldCaptureState::release.store(true, std::memory_order_release);
        return 1;
    }
    // SetGraphDesc is the short owner operation and must complete while the
    // worker remains held in Capture.
    session.SetGraphDesc(latestEdit);
    Check(true, "latest same-path descriptor accepted while worker is blocked");
    bool latestSubmitted = session.CommitAsync(2.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr snapshot,
            UsdGenExecutionPipeline::Outcome outcome) {
            latestSnapshot = std::move(snapshot);
            latestCallback.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
        });
    Check(latestSubmitted, "latest commit submitted before releasing old worker");
    if (!latestSubmitted) {
        HoldCaptureState::release.store(true, std::memory_order_release);
        return 1;
    }
    HoldCaptureState::release.store(true, std::memory_order_release);
    Check(WaitFor(latestCallback), "latest worker publishes after supersession");
    if (!latestCallback.load(std::memory_order_acquire) || !latestSnapshot ||
        !latestSnapshot->generation) {
        return 1;
    }
    Check(oldCallback.load(std::memory_order_acquire),
          "older in-flight cook reports superseded");
    Check(latestSnapshot && latestSnapshot->generation &&
              latestSnapshot->generation->id == baseline->generation->id + 1 &&
              latestSnapshot->generation->frame == 2.0,
          "latest descriptor publishes exactly next generation and frame");
    Check(latestSnapshot && latestSnapshot->graphInfo.Desc().nodes.back().params.front().value ==
              VtValue(0.07f),
          "published snapshot contains newest same-path descriptor");
    bool hasWidthGeometry = false;
    for (auto const &tile : latestSnapshot->generation->tiles) {
        for (float value : tile.widths) {
            hasWidthGeometry = hasWidthGeometry ||
                (std::isfinite(value) && std::fabs(value - 0.07f) < 1e-4f);
            if (hasWidthGeometry) break;
        }
        if (hasWidthGeometry) break;
    }
    Check(hasWidthGeometry,
          "latest publication contains evaluated Width geometry");
    Check(latestSnapshot->stats.commits == 2 && !session.NeedsCommit(),
          "superseded work consumes no public commit count or pending dirt");
    Check(baseline->graphInfo.Desc().nodes.back().params.front().value == VtValue(0.02f) &&
          baseline->generation->id == 0,
          "retained initial snapshot is unchanged by replacement");

    // A malformed replacement returns the last good generation paired with
    // diagnostics; it must not replace the published generation.
    auto badDesc = latestEdit;
    badDesc.nodes.back().type = TfToken("UsdGenMissingAsyncKernel");
    session.SetGraphDesc(badDesc);
    auto rejected = session.CommitSnapshot(3.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(rejected && rejected->generation == latestSnapshot->generation,
          "failed cook retains the last good generation");
    Check(rejected && rejected->diagnostics.HasErrors(),
          "failed cook returns paired diagnostics");
    Check(session.Snapshot() && session.Snapshot()->generation == latestSnapshot->generation,
          "failed cook does not publish replacement generation");
    Check(session.NeedsCommit(), "failed current cook retains pending dirty inputs");
    Check(session.CommitAsync(3.5, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr snapshot, UsdGenExecutionPipeline::Outcome outcome) {
            failureOutcome.store(outcome == UsdGenExecutionPipeline::Outcome::Failed &&
                snapshot && snapshot->diagnostics.HasErrors() &&
                snapshot->generation == latestSnapshot->generation, std::memory_order_release);
            failureDone.store(true, std::memory_order_release);
        }), "failed graph can be retried asynchronously");
    Check(WaitFor(failureDone) && failureOutcome.load(std::memory_order_acquire),
          "rejected async cook reports Failed and retains last good generation");

    // Concurrent synchronous compatibility calls use independent await
    // boundaries and both complete without sharing an application lock.
    session.SetGraphDesc(latestEdit);
    std::atomic<bool> syncStarted{false};
    UsdGenSession::SnapshotPtr syncA, syncB;
    std::thread a([&] {
        syncStarted.store(true, std::memory_order_release);
        syncA = session.CommitSnapshot(4.0, UsdGenCommitReason::NoticeBatchEnd);
    });
    std::thread b([&] {
        while (!syncStarted.load(std::memory_order_acquire)) std::this_thread::yield();
        syncB = session.CommitSnapshot(5.0, UsdGenCommitReason::NoticeBatchEnd);
    });
    a.join();
    b.join();
    auto paired = [](UsdGenSession::SnapshotPtr const &snapshot) {
        return snapshot && snapshot->generation &&
            snapshot->stats.commits ==
                static_cast<uint64_t>(snapshot->generation->id + 1);
    };
    Check(paired(syncA) && paired(syncB),
          "concurrent synchronous CommitSnapshot calls return paired snapshots");
    auto beforeDefinitive = session.Snapshot();
    auto definitive = session.CommitSnapshot(5.5, UsdGenCommitReason::NoticeBatchEnd);
    Check(paired(definitive) && beforeDefinitive && beforeDefinitive->generation &&
              definitive->generation &&
              definitive->generation->id == beforeDefinitive->generation->id + 1,
          "a definitive synchronous commit advances exactly one generation");

    // Callback re-entry may enqueue asynchronous work, but a synchronous wait
    // from pipeline execution is rejected rather than self-deadlocking.
    session.CommitAsync(6.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr,
            UsdGenExecutionPipeline::Outcome outcome) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                followupAccepted.store(
                    session.CommitAsync(7.0, UsdGenCommitReason::NoticeBatchEnd),
                    std::memory_order_release);
                try {
                    (void)session.CommitSnapshot(8.0, UsdGenCommitReason::NoticeBatchEnd);
                } catch (std::logic_error const &) {
                    callbackRejected.store(true, std::memory_order_release);
                }
            }
            callbackDone.store(true, std::memory_order_release);
        });
    Check(WaitFor(callbackDone), "commit callback completes");
    Check(followupAccepted.load(std::memory_order_acquire),
          "callback can enqueue asynchronous follow-up");
    Check(callbackRejected.load(std::memory_order_acquire),
          "synchronous CommitSnapshot from callback is rejected");

    // A bundled request carries descriptor/context/frame as one owner command.
    // Either concurrent request may be superseded, but a published callback
    // must never observe the other request's descriptor or frame.
    auto requestA = std::make_shared<const UsdGenGraphDesc>(MakeDesc(0.11f));
    auto requestB = std::make_shared<const UsdGenGraphDesc>(MakeDesc(0.17f));
    std::atomic<unsigned> requestCallbacks{0}, requestPublications{0};
    std::atomic<bool> requestMismatch{false}, requestAccepted{true};
    auto submitRequest = [&](std::shared_ptr<const UsdGenGraphDesc> desc,
                             double frame, UsdGenContext context, float width) {
        UsdGenSession::CommitRequest request;
        request.frame = frame;
        request.reason = UsdGenCommitReason::NoticeBatchEnd;
        request.desc = std::move(desc);
        request.context = context;
        // This explicit value is important for an imaging owner relaying a
        // caller's device. CPU builds use -2 and intentionally ignore it.
        request.callerDevice = UsdGenSession::CaptureCallerDevice();
        if (!session.CommitAsync(std::move(request),
            [&, frame, width](UsdGenSession::SnapshotPtr snapshot,
                              UsdGenExecutionPipeline::Outcome outcome) {
                if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                    ++requestPublications;
                    const bool matches = snapshot && snapshot->generation &&
                        snapshot->generation->frame == frame &&
                        !snapshot->graphInfo.Desc().nodes.empty() &&
                        snapshot->graphInfo.Desc().nodes.back().params.front().value ==
                            VtValue(width);
                    if (!matches) requestMismatch.store(true, std::memory_order_release);
                }
                ++requestCallbacks;
            })) requestAccepted.store(false, std::memory_order_release);
    };
    std::thread requestThreadA(submitRequest, requestA, 10.0,
                               UsdGenContext::Interactive, 0.11f);
    std::thread requestThreadB(submitRequest, requestB, 11.0,
                               UsdGenContext::Render, 0.17f);
    requestThreadA.join();
    requestThreadB.join();
    // Both terminal callbacks are required.
    auto callbackDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (requestCallbacks.load(std::memory_order_acquire) != 2 &&
           std::chrono::steady_clock::now() < callbackDeadline)
        std::this_thread::yield();
    Check(requestAccepted.load(std::memory_order_acquire) &&
              requestCallbacks.load(std::memory_order_acquire) == 2,
          "both concurrent bundled request commands accepted");
    Check(requestPublications.load(std::memory_order_acquire) <= 2 &&
              !requestMismatch.load(std::memory_order_acquire),
          "each published bundled outcome is paired with its own descriptor and frame");

    // PostDirty is callback-safe: it queues the owner mutation and the later
    // external commit observes it without entering a synchronous owner wait.
    std::atomic<bool> dirtyCallback{false};
    std::atomic<bool> dirtyPostAccepted{false};
    auto node = session.Graph().NodeIdForPath(SdfPath("/async/width"));
    Check(session.CommitAsync(12.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr,
            UsdGenExecutionPipeline::Outcome outcome) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                UsdGenPendingDirty dirty;
                dirty.nodeBits[node] = UsdGenDirtyParameter;
                dirtyPostAccepted.store(session.PostDirty(std::move(dirty)),
                                        std::memory_order_release);
            }
            dirtyCallback.store(true, std::memory_order_release);
        }), "callback-safe dirty source commit accepted");
    Check(WaitFor(dirtyCallback), "callback-safe dirty post callback completes");
    Check(dirtyPostAccepted.load(std::memory_order_acquire),
          "callback-safe dirty source is admitted");
    auto postDirtySnapshot = session.CommitSnapshot(13.0, UsdGenCommitReason::NoticeBatchEnd);
    Check(postDirtySnapshot && postDirtySnapshot->generation &&
              !postDirtySnapshot->diagnostics.HasErrors(),
          "PostDirty queued from callback is consumed by a later owner commit");

    // Command admission is independent from retained cook requests. While an
    // owner completion is executing, a cap-two command lane admits exactly
    // two callback-safe edits and exposes the third rejection; the rejected
    // descriptor must not replace the dirty baseline before a later retry.
    UsdGenSession capped(1, 2);
    capped.SetGraphDesc(MakeDesc(.01f));
    auto cappedInitial = capped.CommitSnapshot(0.0, UsdGenCommitReason::NoticeBatchEnd);
    std::atomic<bool> cappedCallback{false}, cappedFirst{false}, cappedSecond{false},
        cappedRejected{false};
    Check(capped.CommitAsync(1.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr, UsdGenExecutionPipeline::Outcome outcome) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                cappedFirst.store(capped.PostContext(UsdGenContext::Render),
                                  std::memory_order_release);
                cappedSecond.store(capped.PostDevicePublicationEnabled(true),
                                   std::memory_order_release);
                cappedRejected.store(!capped.PostGraphDesc(MakeDesc(.77f)),
                                     std::memory_order_release);
            }
            cappedCallback.store(true, std::memory_order_release);
        }), "small-cap source commit accepted");
    Check(WaitFor(cappedCallback), "small-cap callback completes");
    Check(cappedFirst.load(std::memory_order_acquire) &&
              cappedSecond.load(std::memory_order_acquire) &&
              cappedRejected.load(std::memory_order_acquire),
          "small command lane exposes the third public mutation rejection");
    capped.Drain();
    Check(capped.NeedsCommit(), "rejected descriptor leaves the accepted dirty baseline pending");
    auto cappedRetry = capped.CommitSnapshot(2.0, UsdGenCommitReason::NoticeBatchEnd);
    const bool retainedDescriptor = cappedInitial && cappedRetry &&
        cappedRetry->graphInfo.desc.nodes.size() == 4 &&
        cappedRetry->graphInfo.desc.nodes.back().params.front().value.Get<float>() == .01f;
    Check(retainedDescriptor,
          "retry retains the pre-rejection descriptor rather than a dropped edit");

    // Destruction drains accepted work and gives its callback a terminal
    // outcome; this scope intentionally ends without an explicit Drain().
    std::atomic<unsigned> terminalCallbacks{0};
    UsdGenSession::SnapshotPtr retainedAfterShutdown;
    {
        UsdGenSession closing(2);
        closing.SetGraphDesc(MakeDesc(0.01f));
        retainedAfterShutdown = closing.CommitSnapshot(0.0, UsdGenCommitReason::NoticeBatchEnd);
        for (unsigned i = 0; i != 32; ++i) {
            Check(closing.CommitAsync(i + 1.0, UsdGenCommitReason::NoticeBatchEnd,
                [&](UsdGenSession::SnapshotPtr, UsdGenExecutionPipeline::Outcome) {
                    terminalCallbacks.fetch_add(1, std::memory_order_release);
                }), "closing session accepts asynchronous commit");
        }
    }
    Check(terminalCallbacks.load(std::memory_order_acquire) == 32,
          "session destruction completes every accepted callback exactly once");
    Check(retainedAfterShutdown && retainedAfterShutdown->generation &&
          retainedAfterShutdown->generation->id == 0 && retainedAfterShutdown->routing &&
          retainedAfterShutdown->graphInfo.NodeCount() == 4,
          "immutable geometry and routing survive session destruction");

    // Independent sessions sharing one CPU domain join an in-flight exact-key
    // miss. The leader is held before key registration; the follower parks
    // without entering the held capture and receives its own COW generation.
    auto coalescedDomain = std::make_shared<UsdGenExecutionCacheDomain>(
        UsdGenExecutionCacheDomainKey{UsdGenDeviceBackend::CpuReference, -1, 0},
        64u * 1024u * 1024u);
    UsdGenSession coalescedLeader(2, 4096, coalescedDomain);
    UsdGenSession coalescedFollower(2, 4096, coalescedDomain);
    auto coalescedDesc = std::make_shared<const UsdGenGraphDesc>(MakeDesc(.031f));
    coalescedLeader.SetGraphDesc(*coalescedDesc);
    coalescedFollower.SetGraphDesc(*coalescedDesc);
    HoldCaptureState::entered.store(false, std::memory_order_release);
    HoldCaptureState::release.store(false, std::memory_order_release);
    HoldCaptureState::hold.store(true, std::memory_order_release);
    std::atomic<bool> coalescedLeaderDone{false}, coalescedFollowerDone{false};
    UsdGenSession::SnapshotPtr coalescedLeaderSnapshot, coalescedFollowerSnapshot;
    Check(coalescedLeader.CommitAsync(1.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr snapshot, UsdGenExecutionPipeline::Outcome) {
            coalescedLeaderSnapshot = std::move(snapshot);
            coalescedLeaderDone.store(true, std::memory_order_release);
        }), "CPU coalesced leader accepted");
    Check(WaitFor(HoldCaptureState::entered), "CPU coalesced leader reached hold");
    HoldCaptureState::hold.store(false, std::memory_order_release);
    Check(coalescedFollower.CommitAsync(1.0, UsdGenCommitReason::NoticeBatchEnd,
        [&](UsdGenSession::SnapshotPtr snapshot, UsdGenExecutionPipeline::Outcome) {
            coalescedFollowerSnapshot = std::move(snapshot);
            coalescedFollowerDone.store(true, std::memory_order_release);
        }), "CPU coalesced follower accepted");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Check(!coalescedFollowerDone.load(std::memory_order_acquire),
          "CPU coalesced follower remains nonblocking while leader is held");
    HoldCaptureState::release.store(true, std::memory_order_release);
    Check(WaitFor(coalescedLeaderDone) && WaitFor(coalescedFollowerDone),
          "CPU coalesced leader and follower complete");
    Check(coalescedLeaderSnapshot && coalescedFollowerSnapshot &&
              coalescedLeaderSnapshot->generation && coalescedFollowerSnapshot->generation &&
              coalescedLeaderSnapshot->generation != coalescedFollowerSnapshot->generation &&
              coalescedLeaderSnapshot->generation->tiles.size() ==
                  coalescedFollowerSnapshot->generation->tiles.size() &&
              !coalescedLeaderSnapshot->generation->tiles.empty() &&
              coalescedLeaderSnapshot->generation->tiles.front().points.cdata() ==
                  coalescedFollowerSnapshot->generation->tiles.front().points.cdata() &&
              coalescedLeaderSnapshot->generation->tiles.front().widths.cdata() ==
                  coalescedFollowerSnapshot->generation->tiles.front().widths.cdata() &&
              coalescedFollowerSnapshot->stats.executionCacheCoalesced == 1,
          "CPU coalesced follower publishes a distinct generation with COW payload");
    auto const retainedPoint = coalescedFollowerSnapshot &&
        !coalescedFollowerSnapshot->generation->tiles.empty() &&
        !coalescedFollowerSnapshot->generation->tiles.front().points.empty()
            ? coalescedFollowerSnapshot->generation->tiles.front().points.front()
            : GfVec3f(-1);
    coalescedLeaderSnapshot.reset();
    Check(coalescedFollowerSnapshot &&
              !coalescedFollowerSnapshot->generation->tiles.empty() &&
              !coalescedFollowerSnapshot->generation->tiles.front().points.empty() &&
              coalescedFollowerSnapshot->generation->tiles.front().points.front() ==
                  retainedPoint,
          "CPU follower COW payload remains readable after leader release");

    // Superseding a parked follower withdraws only that waiter. Its pipeline
    // settles as Superseded while the leader remains allowed to finish.
    auto supersedeDomain = std::make_shared<UsdGenExecutionCacheDomain>(
        UsdGenExecutionCacheDomainKey{UsdGenDeviceBackend::CpuReference, -1, 0},
        64u * 1024u * 1024u);
    UsdGenSession supersedeLeader(2, 4096, supersedeDomain);
    UsdGenSession supersedeFollower(2, 4096, supersedeDomain);
    supersedeLeader.SetGraphDesc(*coalescedDesc);
    supersedeFollower.SetGraphDesc(*coalescedDesc);
    HoldCaptureState::entered.store(false, std::memory_order_release);
    HoldCaptureState::release.store(false, std::memory_order_release);
    HoldCaptureState::hold.store(true, std::memory_order_release);
    auto supersedeLeaderDone = std::make_shared<std::atomic<bool>>(false);
    auto supersedeFollowerDone = std::make_shared<std::atomic<bool>>(false);
    auto supersedeFollowerOutcome = std::make_shared<UsdGenExecutionPipeline::Outcome>(
        UsdGenExecutionPipeline::Outcome::Published);
    Check(supersedeLeader.CommitAsync(2.0, UsdGenCommitReason::NoticeBatchEnd,
        [supersedeLeaderDone](UsdGenSession::SnapshotPtr,
                              UsdGenExecutionPipeline::Outcome) {
            supersedeLeaderDone->store(true, std::memory_order_release);
        }), "CPU supersession leader accepted");
    Check(WaitFor(HoldCaptureState::entered), "CPU supersession leader reached hold");
    HoldCaptureState::hold.store(false, std::memory_order_release);
    Check(supersedeFollower.CommitAsync(2.0, UsdGenCommitReason::NoticeBatchEnd,
        [supersedeFollowerDone, supersedeFollowerOutcome](
            UsdGenSession::SnapshotPtr,
            UsdGenExecutionPipeline::Outcome outcome) {
            *supersedeFollowerOutcome = outcome;
            supersedeFollowerDone->store(true, std::memory_order_release);
        }), "CPU supersession follower accepted");
    UsdGenPendingDirty supersedingDirty;
    supersedingDirty.structural = true;
    supersedeFollower.PostDirty(std::move(supersedingDirty));
    Check(WaitFor(*supersedeFollowerDone) &&
              *supersedeFollowerOutcome == UsdGenExecutionPipeline::Outcome::Superseded,
          "CPU superseded follower settles without publication");
    HoldCaptureState::release.store(true, std::memory_order_release);
    Check(WaitFor(*supersedeLeaderDone), "CPU supersession leader still completes");

    std::printf("testUsdGenAsyncSession: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
