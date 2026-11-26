// testUsdGenAsyncImagingSession -- imaging-owner request pairing, callback
// re-entry, retirement, and terminal shutdown behavior.

#include "usdGenCurves/publish/curvePublication.h"
#include "testBackend.h"
#include "usdGen/op.h"
#include "usdGenUsd/lower.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGenImaging/usdGenImagingSession.h"
#include "usdGenUsd/resolve/imageMapCache.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/pxr.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/systemMessages.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGenImaging;

namespace {

int failures = 0;

void Check(bool value, std::string const &what)
{
    if (!value) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

template <class Predicate>
bool WaitFor(Predicate predicate, int milliseconds = 10000)
{
    auto const deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(milliseconds);
    while (!predicate() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    return predicate();
}

UsdGenGraphDesc MakeDesc(float width)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/asyncImaging");
    desc.terminal = SdfPath("/asyncImaging/width");

    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/asyncImaging/surface");
    surface.id = 0;
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
        GfVec3f(0, 1, 0), GfVec3f(1, 1, 0)};
    surface.points = surface.restPoints;
    surface.uv = VtVec2fArray{
        GfVec2f(0, 0), GfVec2f(1, 0),
        GfVec2f(0, 1), GfVec2f(1, 1)};
    surface.faceVertexCounts = VtIntArray{4};
    surface.faceVertexIndices = VtIntArray{0, 1, 3, 2};
    desc.surfaces.push_back(std::move(surface));

    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/asyncImaging/scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {SdfPath("/asyncImaging/surface")};
    desc.nodes.push_back(std::move(scatter));

    UsdGenNodeDesc grow;
    grow.path = SdfPath("/asyncImaging/grow");
    grow.type = TfToken("UsdGenGrow");
    grow.inputs = {SdfPath("/asyncImaging/scatter")};
    grow.params.push_back({TfToken("segments"), VtValue(4), false});
    grow.params.push_back({TfToken("length"), VtValue(1.0), false});
    desc.nodes.push_back(std::move(grow));

    UsdGenNodeDesc widthNode;
    widthNode.path = SdfPath("/asyncImaging/width");
    widthNode.type = TfToken("UsdGenWidth");
    widthNode.inputs = {SdfPath("/asyncImaging/grow")};
    widthNode.params.push_back({TfToken("width"), VtValue(width), false});
    desc.nodes.push_back(std::move(widthNode));
    return desc;
}

// Keep this CUDA descriptor intentionally scalar and source-only.  The
// request-relay test must reach the cooker admission check; the ordinary
// Scatter/Grow fixture is not admitted by the CUDA graph compiler.
UsdGenGraphDesc MakeCudaSourceDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/asyncImaging/cudaRequest");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    desc.defaultWidth = .025f;

    UsdGenNodeDesc source;
    source.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("Source"));
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/asyncImaging/cudaRequest/curves")};
    source.surfaces = {SdfPath("/asyncImaging/cudaRequest/surface")};
    desc.terminal = source.path;
    desc.nodes.push_back(source);

    UsdGenSurfaceDesc surface;
    surface.path = source.surfaces.front();
    surface.faceVertexCounts = VtIntArray{3};
    surface.faceVertexIndices = VtIntArray{0, 1, 2};
    surface.restPoints = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0)};
    surface.points = surface.restPoints;
    desc.surfaces.push_back(std::move(surface));

    UsdGenCurveSetDesc curves;
    curves.path = source.curves.front();
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = VtIntArray{2, 3};
    curves.points = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(0, 1, 0), GfVec3f(1, 0, 0),
        GfVec3f(1, .5f, 0), GfVec3f(1, 1, 0)};
    curves.rest = curves.points;
    curves.curveId = {37, 91};
    curves.skinPrim = VtIntArray{0, 0};
    curves.skinPrimUv = VtVec2fArray{GfVec2f(0, 0), GfVec2f(1, 0)};
    desc.curveSets.push_back(std::move(curves));
    return desc;
}

UsdGenSessionHandle NewSession(uint64_t commandCapacity = 4096)
{
    UsdGenSessionKey key;
    key.groomRoot = SdfPath("/asyncImaging");
    key.renderInstanceId = 991;
    return TfCreateRefPtr(new UsdGenImagingSession(
        key, std::make_shared<UsdGenSession>(2), 0.0,
        UsdGenContext::Interactive, commandCapacity));
}

UsdGenImagingSession::CommitRequest Request(
    std::shared_ptr<const UsdGenGraphDesc> desc, double frame)
{
    UsdGenImagingSession::CommitRequest request;
    request.reason = UsdGenCommitReason::NoticeBatchEnd;
    request.frame = frame;
    request.desc = std::move(desc);
    request.context = UsdGenContext::Interactive;
    return request;
}

bool HasWidth(UsdGenImagingSession::CommitPayload const &payload,
             double frame, float width)
{
    if (!payload.published || !payload.generation ||
        payload.generation->frame != frame)
        return false;
    bool found = false;
    for (auto const &tile : usdGen::UsdGenCurvePublicationOf(*payload.generation).tiles) {
        for (float value : tile.widths) {
            if (!std::isfinite(value) || std::fabs(value - width) > 1e-4f)
                return false;
            found = true;
        }
    }
    return found;
}

std::atomic<unsigned> batchEntered{0};
std::atomic<bool> batchRelease{false}, batchTimedOut{false};
class BatchHoldOp final : public UsdGenOp {
public:
    explicit BatchHoldOp(UsdGenOpRecord const &r) : UsdGenOp(r) {}
    bool Bind(UsdGenParamView const&, UsdGenDiagnostics*) override { return true; }
    UsdGenEpoch CaptureDigest(UsdGenCaptureContext const&) const override { return {1, 1}; }
    bool Capture(UsdGenCaptureContext const&, UsdGenOpInputs const &,
                 UsdGenCapture*, UsdGenDiagnostics* diagnostics) override {
        batchEntered.fetch_add(1, std::memory_order_release);
        if (WaitFor([] { return batchRelease.load(std::memory_order_acquire); })) return true;
        batchTimedOut.store(true);
        diagnostics->Error("store batch capture gate timed out");
        return false;
    }
    void Evaluate(UsdGenEvalContext const&, UsdGenCapture const&, UsdGenChunkView const &) const override {}
};

void RegisterBatchHoldOp(UsdGenRegistrar &r)
{
    {
        UsdGenOpDef d;
        d.name = "UsdGenImagingBatchHold";
        d.displayName = "ImagingBatchHold";
        d.create = [](UsdGenOpRecord const &rec, void *) -> std::unique_ptr<UsdGenOp> {
            return std::make_unique<BatchHoldOp>(rec);
        };
        r.RegisterOp(d);
    }
}

void TestPairedConcurrentRequests()
{
    auto session = NewSession();
    auto descA = std::make_shared<const UsdGenGraphDesc>(MakeDesc(.031f));
    auto descB = std::make_shared<const UsdGenGraphDesc>(MakeDesc(.047f));
    std::atomic<unsigned> callbacks{0};
    std::atomic<unsigned> publications{0};
    std::atomic<unsigned> mismatches{0};
    std::atomic<unsigned> accepted{0};

    auto submit = [&](std::shared_ptr<const UsdGenGraphDesc> desc,
                      double frame, float width) {
        auto request = Request(std::move(desc), frame);
        if (session->CommitAsync(std::move(request),
                [&, frame, width](UsdGenImagingSession::CommitPayload const &payload,
                                  UsdGenExecutionPipeline::Outcome outcome) {
                    if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                        publications.fetch_add(1, std::memory_order_relaxed);
                        if (!HasWidth(payload, frame, width))
                            mismatches.fetch_add(1, std::memory_order_relaxed);
                    }
                    callbacks.fetch_add(1, std::memory_order_release);
                }))
            accepted.fetch_add(1, std::memory_order_relaxed);
    };

    std::thread a(submit, descA, 10.0, .031f);
    std::thread b(submit, descB, 11.0, .047f);
    a.join();
    b.join();

    Check(WaitFor([&] { return callbacks.load(std::memory_order_acquire) ==
                               accepted.load(std::memory_order_acquire); }),
          "concurrent imaging requests reach one terminal callback each");
    Check(accepted.load() == 2, "both concurrent imaging requests are accepted");
    Check(publications.load() > 0 && publications.load() <= accepted.load() && mismatches.load() == 0,
          "published imaging payload keeps descriptor geometry paired with frame");
    session->Shutdown();
    session.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestDevicePublicationRequestRelay()
{
    auto session = NewSession();
    // A CUDA-enabled build can exercise false admission without allocating a
    // CUDA workspace.  CPU-off builds reject graph compilation before this
    // admission point, which is reported as unavailable rather than treated
    // as relay evidence.
    auto cudaDesc = std::make_shared<const UsdGenGraphDesc>(MakeCudaSourceDesc());
    UsdGenDiagnostics preflight;

    struct Result {
        std::atomic<bool> done{false};
        bool accepted = false;
        UsdGenImagingSession::CommitPayload payload;
        UsdGenExecutionPipeline::Outcome outcome =
            UsdGenExecutionPipeline::Outcome::Superseded;
    };
    auto submit = [&](UsdGenImagingSession::CommitRequest request) {
        auto result = std::make_shared<Result>();
        result->accepted = session->CommitAsync(std::move(request), [result](
            UsdGenImagingSession::CommitPayload const& payload,
            UsdGenExecutionPipeline::Outcome outcome) {
                result->payload = payload;
                result->outcome = outcome;
                result->done.store(true, std::memory_order_release);
            });
        return result;
    };
    auto disabledDiagnostic = [](UsdGenImagingSession::CommitPayload const& payload) {
        for (std::string const& error : payload.diagnostics.errors)
            if (error.find("CUDA publication requires a device-aware consumer") !=
                std::string::npos)
                return true;
        return false;
    };
    auto backendUnavailable = [](UsdGenDiagnostics const& diagnostics) {
        for (std::string const& error : diagnostics.errors)
            if (error.find("backend is not built") != std::string::npos)
                return true;
        return false;
    };
    if (!ValidateCudaGraph(usdGenUsd::Lower(*cudaDesc), &preflight)) {
        Check(backendUnavailable(preflight),
            "CPU-disabled build reports CUDA relay admission as unavailable");
        session->Shutdown();
        session.Reset();
        UsdGenImagingSession::DrainRetired();
        return;
    }
    Check(preflight.errors.empty(), "scalar CUDA source descriptor passes preflight");

    // The core starts true so the first imaging request must actively relay
    // false.  callerDevice=0 avoids the CPU caller-capture sentinel; false
    // exits at the admission guard before any CUDA workspace is allocated.
    session->Engine()->SetDevicePublicationEnabled(true);

    UsdGenImagingSession::CommitRequest disable =
        Request(cudaDesc, 31.0);
    disable.callerDevice = 0;
    disable.devicePublication = false;
    auto first = submit(std::move(disable));
    Check(first->accepted && WaitFor([&] {
              return first->done.load(std::memory_order_acquire);
          }), "imaging device-disable request reaches completion");
    Check(first->outcome == UsdGenExecutionPipeline::Outcome::Failed &&
              disabledDiagnostic(first->payload),
          "imaging request relays false device publication before CUDA admission");

    UsdGenImagingSession::CommitRequest absent;
    absent.reason = UsdGenCommitReason::SetTime;
    absent.frame = 32.0;
    absent.callerDevice = 0;
    auto second = submit(std::move(absent));
    Check(second->accepted && WaitFor([&] {
              return second->done.load(std::memory_order_acquire);
          }), "imaging absent-selection request reaches completion");
    Check(second->outcome == UsdGenExecutionPipeline::Outcome::Failed &&
              disabledDiagnostic(second->payload),
          "imaging absent device publication preserves the prior false selection");

    session->Shutdown();
    session.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestCallbackUnregisterAndReentry()
{
    auto session = NewSession();
    auto desc = std::make_shared<const UsdGenGraphDesc>(MakeDesc(.055f));
    auto token = std::make_shared<std::atomic<int>>(-1);
    std::atomic<unsigned> callbackCount{0};
    std::atomic<bool> followupAccepted{false};
    std::atomic<bool> followupDone{false};
    std::atomic<bool> synchronousRejected{false};
    std::atomic<bool> unregisterAccepted{false};

    // The callback intentionally owns a handle while it is registered.  It
    // unregisters before queuing the follow-up, so the follow-up cannot call
    // this callback after the owner processes those commands in order.
    UsdGenSessionHandle callbackOwner = session;
    int callbackToken = session->RegisterRepublishCallback(
        [callbackOwner, token, desc, &callbackCount, &followupAccepted,
         &followupDone, &synchronousRejected, &unregisterAccepted](
            UsdGenImagingSession::CommitPayload const &payload) mutable {
            if (!payload.published) return;
            unsigned const call = callbackCount.fetch_add(
                1, std::memory_order_relaxed) + 1;
            if (call != 1) return;
            unregisterAccepted.store(callbackOwner->UnregisterRepublishCallback(token->load()),
                                     std::memory_order_release);
            UsdGenImagingSession::CommitRequest followup =
                Request(desc, 2.0);
            followupAccepted.store(callbackOwner->CommitAsync(
                std::move(followup),
                [&followupDone](UsdGenImagingSession::CommitPayload const &,
                                UsdGenExecutionPipeline::Outcome) {
                    followupDone.store(true, std::memory_order_release);
                }), std::memory_order_release);
            try {
                callbackOwner->Shutdown();
            } catch (std::logic_error const &) {
                synchronousRejected.store(true, std::memory_order_release);
            }
        });
    token->store(callbackToken, std::memory_order_release);
    Check(callbackToken >= 0, "republish callback registration returns a token");

    Check(session->CommitAsync(Request(desc, 1.0)),
          "initial imaging request for callback test is accepted");
    Check(WaitFor([&] { return followupDone.load(std::memory_order_acquire); }),
          "callback-enqueued follow-up reaches a terminal completion");
    Check(callbackCount.load(std::memory_order_acquire) == 1,
          "owner-ordered unregister prevents callback on follow-up publication");
    Check(followupAccepted.load(std::memory_order_acquire),
          "republish callback can enqueue asynchronous follow-up");
    Check(unregisterAccepted.load(std::memory_order_acquire),
          "ordinary callback unregister reports owner admission");
    Check(synchronousRejected.load(std::memory_order_acquire),
          "synchronous shutdown from callback is rejected");

    session->Shutdown();
    session.Reset();
    callbackOwner.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestLastHandleReleaseInCompletion()
{
    UsdGenSessionHandle session = NewSession();
    auto weakSession = TfCreateWeakPtr(session.operator->());
    UsdGenSessionHandle completionOwner = session;
    session.Reset();
    std::atomic<bool> completed{false};
    std::atomic<bool> published{false};
    std::shared_ptr<const UsdGenGeneration> retainedGeneration;
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> retainedRouting;

    // Keep a separate caller handle while moving the final handle into the
    // completion; argument evaluation order must not make the call through
    // an already-moved TfRefPtr.
    UsdGenSessionHandle submitter = completionOwner;
    bool accepted = submitter->CommitAsync(
        Request(std::make_shared<const UsdGenGraphDesc>(MakeDesc(.065f)), 3.0),
        [completionOwner = std::move(completionOwner), &completed, &published,
         &retainedGeneration, &retainedRouting](
            UsdGenImagingSession::CommitPayload const &payload,
            UsdGenExecutionPipeline::Outcome outcome) mutable {
            retainedGeneration = payload.generation;
            retainedRouting = payload.routing;
            published.store(outcome == UsdGenExecutionPipeline::Outcome::Published &&
                                payload.published,
                            std::memory_order_release);
            // This is deliberately the last TfRefPtr release.  The imaging
            // state must retire asynchronously instead of deleting itself on
            // the owner callback stack.
            completionOwner.Reset();
            completed.store(true, std::memory_order_release);
        });
    submitter.Reset();
    Check(accepted, "last-handle completion request is accepted");
    Check(WaitFor([&] { return completed.load(std::memory_order_acquire); }),
          "last-handle completion runs");
    Check(published.load(std::memory_order_acquire) && retainedGeneration &&
              retainedRouting,
          "completion retains immutable geometry and routing payloads");
    Check(WaitFor([&] { return !weakSession; }),
          "all callback-owned public handles are released before retirement drain");
    UsdGenImagingSession::DrainRetired();
}

void TestShutdownCompletesAcceptedQueue()
{
    auto session = NewSession();
    auto desc = std::make_shared<const UsdGenGraphDesc>(MakeDesc(.075f));
    std::atomic<bool> baselineDone{false};
    std::shared_ptr<const UsdGenGeneration> retainedGeneration;
    std::shared_ptr<const UsdGenGraphRoutingSnapshot> retainedRouting;
    Check(session->CommitAsync(Request(desc, 4.0),
        [&](UsdGenImagingSession::CommitPayload const &payload,
            UsdGenExecutionPipeline::Outcome outcome) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                retainedGeneration = payload.generation;
                retainedRouting = payload.routing;
            }
            baselineDone.store(true, std::memory_order_release);
        }), "shutdown test baseline is accepted");
    Check(WaitFor([&] { return baselineDone.load(std::memory_order_acquire); }),
          "shutdown test baseline completes");

    constexpr unsigned requestCount = 48;
    std::vector<std::atomic<unsigned>> perRequest(requestCount);
    for (auto &count : perRequest) count.store(0);
    unsigned accepted = 0;
    for (unsigned i = 0; i != requestCount; ++i) {
        if (session->CommitAsync(
                Request(desc, 5.0 + static_cast<double>(i)),
                [&, i](UsdGenImagingSession::CommitPayload const &,
                        UsdGenExecutionPipeline::Outcome) {
                    perRequest[i].fetch_add(1, std::memory_order_relaxed);
                }))
            ++accepted;
    }
    session->Shutdown();
    unsigned terminal = 0;
    bool duplicate = false;
    for (auto const &count : perRequest) {
        unsigned const value = count.load(std::memory_order_acquire);
        terminal += value;
        duplicate = duplicate || value > 1;
    }
    Check(accepted == requestCount, "shutdown queue accepts all bounded requests");
    Check(terminal == accepted && !duplicate,
          "shutdown gives every accepted request exactly one terminal callback");
    Check(retainedGeneration && retainedRouting,
          "shutdown preserves retained immutable geometry and routing");
    session.Reset();
    UsdGenImagingSession::DrainRetired();
}

class AdoptionInput final : public HdSceneIndexBase {
public:
    SdfPath const root{"/adoptionRace"};
    std::atomic<bool> present{true}, gateArmed{false}, release{false}, timedOut{false};
    mutable std::atomic<bool> entered{false};
    HdSceneIndexPrim GetPrim(SdfPath const& path) const override {
        if (path != root) return {};
        const bool exists = present.load();
        // Constructor-time population may read the root. Arm the gate only
        // after that initial owner synchronization, then block one later
        // candidate query to exercise cancellation ordering.
        if (gateArmed.load(std::memory_order_acquire) &&
            !entered.exchange(true, std::memory_order_acq_rel)) {
            auto* self = const_cast<AdoptionInput*>(this);
            if (!WaitFor([&] { return release.load(std::memory_order_acquire); }))
                self->timedOut.store(true);
        }
        return exists ? HdSceneIndexPrim{TfToken("UsdGenGroom"), HdRetainedContainerDataSource::New()}
                      : HdSceneIndexPrim{};
    }
    SdfPathVector GetChildPrimPaths(SdfPath const& path) const override {
        return path == SdfPath::AbsoluteRootPath() && present.load()
            ? SdfPathVector{root} : SdfPathVector{};
    }
    void Remove() { present.store(false); _SendPrimsRemoved({{root}}); }
    void Add() { present.store(true); _SendPrimsAdded({{root, TfToken("UsdGenGroom")}}); }
};

void TestCancelledAdoptionCannotReplaceNewAttachment()
{
    auto input = TfCreateRefPtr(new AdoptionInput);
    auto index = UsdGenGroomSceneIndex::New(input, 7321);
    auto& store = UsdGenSessionStore::GetInstance();
    UsdGenSessionKey key{"", input->root, 7321};
    auto *groomIndex = dynamic_cast<UsdGenGroomSceneIndex *>(index.operator->());
    Check(groomIndex != nullptr, "adoption test obtains scene owner boundary");
    if (groomIndex) groomIndex->Synchronize();
    index->SystemMessage(HdSystemMessageTokens->asyncAllow, nullptr);
    input->gateArmed.store(true, std::memory_order_release);
    input->Remove();
    std::thread discover([&] { input->Add(); });
    const bool entered = WaitFor([&] { return input->entered.load(std::memory_order_acquire); });
    Check(entered, "candidate adoption reaches bounded input gate");
    if (!entered) {
        input->release.store(true);
        discover.join();
        return;
    }
    input->Remove();
    Check(!input->timedOut.load(), "removal is accepted while candidate construction is held");
    input->gateArmed.store(false, std::memory_order_release);
    input->Add();
    input->release.store(true, std::memory_order_release);
    discover.join();
    if (groomIndex) groomIndex->Synchronize();
    auto replacement = store.Find(key);
    Check(replacement && replacement->AttachedIndices() == 1,
          "same-path replacement installs a fresh adoption ticket");
    Check(replacement && store.Find(key) == replacement && replacement->AttachedIndices() == 1,
          "late cancelled candidate cannot replace or leak an attachment");
    input->Remove();
    if (groomIndex) groomIndex->Synchronize();
    Check(!store.Find(key), "replacement detach removes the final store attachment");
    index.Reset();
    replacement.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestStoreDispatchesIndependentSessionsTogether()
{
    auto& store = UsdGenSessionStore::GetInstance();
    UsdGenSessionKey keyA{"", SdfPath("/asyncImaging"), 8301};
    UsdGenSessionKey keyB{"", SdfPath("/asyncImaging"), 8302};
    auto a = store.Attach(keyA), b = store.Attach(keyB);
    auto desc = MakeDesc(.03f);
    UsdGenNodeDesc hold;
    hold.path = SdfPath("/asyncImaging/hold");
    hold.type = TfToken("UsdGenImagingBatchHold");
    hold.inputs = {desc.terminal};
    desc.nodes.push_back(hold);
    desc.terminal = hold.path;
    Check(a->StageDesc(desc), "first staged descriptor is admitted");
    Check(b->StageDesc(desc), "second staged descriptor is admitted");
    std::exception_ptr failure;
    std::thread caller([&] {
        try { store.SetTime(42.0); }
        catch (...) { failure = std::current_exception(); }
    });
    const bool overlap = WaitFor([] { return batchEntered.load(std::memory_order_acquire) == 2; });
    Check(overlap && !batchTimedOut.load(),
          "store frame update dispatches both descriptions before either capture is released");
    batchRelease.store(true, std::memory_order_release);
    caller.join();
    Check(!failure && a->LatestGeneration() && b->LatestGeneration() &&
          a->LatestGeneration()->frame == 42.0 && b->LatestGeneration()->frame == 42.0,
          "parallel store batch waits for both exact-frame publications");
    store.Detach(keyA);
    store.Detach(keyB);
    a.Reset();
    b.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestContextMailboxSurvivesOrdinaryCommandPressure()
{
    // While the completion callback runs, capacity four is fully accounted
    // for by the close ticket, durable context mailbox, and two ordinary
    // commands. The context post must still be accepted by its mailbox.
    auto session = NewSession(4);
    std::atomic<bool> callback{false}, first{false}, second{false}, context{false};
    Check(session->CommitAsync(Request(std::make_shared<UsdGenGraphDesc>(MakeDesc(.02f)), 1.0),
        [&](UsdGenImagingSession::CommitPayload const&, UsdGenExecutionPipeline::Outcome outcome) {
            if (outcome == UsdGenExecutionPipeline::Outcome::Published) {
                first.store(session->SetTime(2.0), std::memory_order_release);
                second.store(session->StageDesc(MakeDesc(.03f)), std::memory_order_release);
                context.store(session->SetContext(UsdGenContext::Render),
                              std::memory_order_release);
            }
            callback.store(true, std::memory_order_release);
        }), "small-cap imaging request reserves its terminal relay");
    Check(WaitFor([&] { return callback.load(std::memory_order_acquire); }),
          "small-cap imaging terminal relay completes");
    Check(first.load(std::memory_order_acquire) && second.load(std::memory_order_acquire) &&
              context.load(std::memory_order_acquire),
          "durable context mailbox is admitted while ordinary queue is full");
    // Close owns its ticket from construction, so queued ordinary/mailbox
    // traffic cannot consume lifecycle admission.
    session->Shutdown();
    Check(!session->SetContext(UsdGenContext::Interactive),
          "context mailbox rejects only after lifecycle shutdown");
}

void TestReservedCallbackCleanupSurvivesSourceOwnerPressure()
{
    // cap=5: close ticket, context mailbox, register ticket, unregister
    // ticket, and one held filler saturate the source owner. Registration and
    // cleanup still consume their pre-reserved admissions in order.
    auto session = NewSession(5);
    auto registerTicket = session->ReserveLifecycleCommand();
    auto unregisterTicket = session->ReserveLifecycleCommand();
    std::vector<UsdGenExecutionPipeline::CommandTicket> fillers;
    for (;;) {
        auto ticket = session->ReserveLifecycleCommand();
        if (!ticket) break;
        fillers.emplace_back(std::move(ticket));
    }
    Check(registerTicket && unregisterTicket && !fillers.empty(),
          "session cleanup tickets reserve before source-owner saturation");
    Check(!session->StageDesc(MakeDesc(.08f)),
          "ordinary imaging mutation rejects at full source owner");
    std::atomic<unsigned> calls{0};
    std::atomic<bool> registered{false};
    const int token = session->RegisterRepublishCallback(std::move(registerTicket),
        [&calls](UsdGenImagingSession::CommitPayload const&) {
            calls.fetch_add(1, std::memory_order_release);
        }, [&] { registered.store(true, std::memory_order_release); });
    Check(token >= 0, "reserved callback registration is accepted at full source owner");
    if (!WaitFor([&] { return registered.load(std::memory_order_acquire); })) std::abort();
    auto replacementFiller = session->ReserveLifecycleCommand();
    Check(static_cast<bool>(replacementFiller),
          "session source credit released after reserved registration");
    if (replacementFiller) fillers.emplace_back(std::move(replacementFiller));
    Check(!session->UnregisterRepublishCallback(token),
          "ordinary callback cleanup rejects at full source owner");
    std::atomic<bool> unregistered{false}, commitDone{false};
    Check(session->UnregisterRepublishCallbackAsync(std::move(unregisterTicket), token,
        [&] { unregistered.store(true, std::memory_order_release); }),
          "reserved callback cleanup is accepted at full source owner");
    if (!WaitFor([&] { return unregistered.load(std::memory_order_acquire); })) {
        Check(false, "reserved callback cleanup completion runs");
        std::abort(); // callbacks retain this stack state
    }
    std::atomic<bool> commitPublished{false};
    Check(session->CommitAsync(Request(std::make_shared<UsdGenGraphDesc>(MakeDesc(.08f)), 1.0),
        [&](UsdGenImagingSession::CommitPayload const&, UsdGenExecutionPipeline::Outcome outcome) {
            commitPublished.store(outcome == UsdGenExecutionPipeline::Outcome::Published,
                                  std::memory_order_release);
            commitDone.store(true, std::memory_order_release);
        }), "post-cleanup imaging commit is accepted");
    if (!WaitFor([&] { return commitDone.load(std::memory_order_acquire); })) {
        Check(false, "post-cleanup imaging commit completes");
        std::abort(); // callbacks retain this stack state
    }
    Check(unregistered.load(std::memory_order_acquire) &&
              commitDone.load(std::memory_order_acquire) &&
              commitPublished.load(std::memory_order_acquire) &&
              calls.load(std::memory_order_acquire) == 0,
          "pre-admitted callback cleanup prevents later republish delivery");
    fillers.clear(); // release held source credits before Shutdown/retirement
    session->Shutdown();
    session.Reset();
    UsdGenImagingSession::DrainRetired();
}

void TestImageMapReloadStagesNewCowGeneration()
{
    auto session = NewSession();
    auto desc = MakeDesc(.08f);
    UsdGenMapDesc map;
    map.path = SdfPath("/asyncImaging/imageMap");
    map.type = TfToken("UsdGenImageMap");
    map.resolvedAssetPath = std::string(USDGEN_TEST_SOURCE_DIR) +
        "/tests/golden/stormLook_A.png";
    desc.maps.push_back(map);
    desc.nodes.back().mapBindings.push_back({map.path, TfToken("usdGen:map")});

    std::atomic<bool> firstDone{false};
    std::atomic<bool> firstPublished{false};
    Check(session->CommitAsync(Request(
        std::make_shared<const UsdGenGraphDesc>(desc), 1.0),
        [&](UsdGenImagingSession::CommitPayload const&,
            UsdGenExecutionPipeline::Outcome outcome) {
            firstPublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            firstDone.store(true, std::memory_order_release);
        }), "image-map imaging commit is accepted");
    Check(WaitFor([&] { return firstDone.load(std::memory_order_acquire); }) &&
              firstPublished.load(std::memory_order_acquire),
          "image-map imaging commit publishes");
    auto firstGraph = session->Engine()->Graph();
    Check(UsdGenMapInputs(firstGraph.Spec()).size() == 1 &&
              UsdGenMapInputView(UsdGenMapInputs(firstGraph.Spec())[0]).imagePayload,
          "imaging session publishes a decoded immutable map payload");
    auto const oldPayload = UsdGenMapInputView(UsdGenMapInputs(firstGraph.Spec())[0]).imagePayload;
    uint64_t const oldGeneration = UsdGenMapInputView(UsdGenMapInputs(firstGraph.Spec())[0]).textureGeneration;
    float const oldFirst = oldPayload ? oldPayload->Data()[0] : 0.0f;

    uint64_t const nextGeneration = InvalidateUsdGenImageMapCache();
    Check(session->ReloadMaps(),
          "imaging session accepts an explicit map payload reload");
    std::atomic<bool> secondDone{false};
    std::atomic<bool> secondPublished{false};
    UsdGenImagingSession::CommitRequest request;
    request.reason = UsdGenCommitReason::NoticeBatchEnd;
    request.frame = 1.0;
    Check(session->CommitAsync(std::move(request),
        [&](UsdGenImagingSession::CommitPayload const&,
            UsdGenExecutionPipeline::Outcome outcome) {
            secondPublished.store(
                outcome == UsdGenExecutionPipeline::Outcome::Published,
                std::memory_order_release);
            secondDone.store(true, std::memory_order_release);
        }), "reloaded image-map commit is accepted");
    Check(WaitFor([&] { return secondDone.load(std::memory_order_acquire); }) &&
              secondPublished.load(std::memory_order_acquire),
          "reloaded image-map generation publishes");
    auto replacementGraph = session->Engine()->Graph();
    Check(UsdGenMapInputs(replacementGraph.Spec()).size() == 1 &&
              UsdGenMapInputView(UsdGenMapInputs(replacementGraph.Spec())[0]).imagePayload &&
              UsdGenMapInputView(UsdGenMapInputs(replacementGraph.Spec())[0]).textureGeneration == nextGeneration &&
              UsdGenMapInputView(UsdGenMapInputs(replacementGraph.Spec())[0]).textureGeneration > oldGeneration &&
              UsdGenMapInputView(UsdGenMapInputs(replacementGraph.Spec())[0]).imagePayload != oldPayload &&
              oldPayload && oldPayload->Data()[0] == oldFirst,
          "reload replaces the map COW generation while retained pixels stay immutable");
    session->Shutdown();
    session.Reset();
    UsdGenImagingSession::DrainRetired();
}

} // namespace

int main()
{
    UsdGenRegisterStatic(&RegisterBatchHoldOp);
    TestPairedConcurrentRequests();
    TestDevicePublicationRequestRelay();
    TestCallbackUnregisterAndReentry();
    TestLastHandleReleaseInCompletion();
    TestShutdownCompletesAcceptedQueue();
    TestCancelledAdoptionCannotReplaceNewAttachment();
    TestStoreDispatchesIndependentSessionsTogether();
    TestContextMailboxSurvivesOrdinaryCommandPressure();
    TestReservedCallbackCleanupSurvivesSourceOwnerPressure();
    TestImageMapReloadStagesNewCowGeneration();
    std::printf("testUsdGenAsyncImagingSession: %s\n",
                failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
