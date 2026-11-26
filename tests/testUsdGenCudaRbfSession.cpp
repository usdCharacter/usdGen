#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "usdGen/session.h"
#include "usdGenUsd/lower.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/gpu/generation.h"
#include <cmath>
#include <cstdio>
#include <limits>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

static expr::ValueShape Scalar(expr::ScalarType t) { return {t, 1, 1, 1, 1, false}; }

static UsdGenGraphDesc MakeDesc() {
    UsdGenGraphDesc d; d.description = SdfPath("/Groom/Description");
    d.executionBackend = UsdGenExecutionBackend::Cuda; d.defaultWidth = .025f;
    UsdGenSurfaceDesc s; s.path = SdfPath("/Scalp"); s.worldMatrix.SetIdentity();
    s.restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    s.points = s.restPoints;
    s.faceVertexCounts = {3,3,3}; s.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4};
    d.surfaces.push_back(s);
    UsdGenCurveSetDesc c; c.path = SdfPath("/Hair"); c.role = UsdGenLane::Chunked;
    c.curveRole = TfToken("hair"); c.type = TfToken("cubic"); c.basis = TfToken("bspline"); c.wrap = TfToken("pinned");
    // Roots lie on their bound triangles.  The IDs deliberately arrive out
    // of order; the source preparation sorts every curve payload by ID.
    c.curveVertexCounts = {2,3};
    c.points = {{.2f,0,.2f},{.2f,.5f,.2f}, {.8f,.4f,.2f},{.8f,.5f,.2f},{.8f,.5f,.5f}};
    c.rest = c.points; c.curveId = {7,3}; c.skinPrim = {1,2}; c.skinPrimUv = {{.2f,.2f},{.2f,.2f}};
    d.curveSets.push_back(c);
    UsdGenNodeDesc source; source.path = SdfPath("/Ops/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {c.path}; source.surfaces = {s.path};
    UsdGenNodeDesc first; first.path = SdfPath("/Ops/WidthBefore"); first.type = TfToken("UsdGenWidth");
    first.inputs = {source.path}; first.params.push_back({TfToken("width"), VtValue(.2f), false});
    UsdGenNodeDesc deform; deform.path = SdfPath("/Ops/Deform"); deform.type = TfToken("UsdGenDeform");
    deform.inputs = {first.path}; deform.surfaces = {s.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false});
    deform.params.push_back({TfToken("lockRoots"), VtValue(true), false});
    UsdGenNodeDesc last; last.path = SdfPath("/Ops/WidthAfter"); last.type = TfToken("UsdGenWidth");
    last.inputs = {deform.path}; last.params.push_back({TfToken("width"), VtValue(.5f), false});
    d.nodes = {source, first, deform, last}; d.terminal = last.path;
    return d;
}

static bool Read(gpu::CudaGeometryLease const& lease, std::vector<float3>* points,
                 std::vector<float>* widths, std::vector<uint64_t>* ids, cudaStream_t stream) {
    auto v = lease.Geometry();
    points->resize(v.pointCount); widths->resize(v.pointCount); ids->resize(v.curveCount);
    if (cudaMemcpyAsync(points->data(), v.points.data, points->size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(widths->data(), v.widths.data, widths->size()*sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(ids->data(), v.stableIds.data, ids->size()*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

int main() {
    cudaStream_t stream = nullptr; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    UsdGenSession session; session.SetDevicePublicationEnabled(true);
    auto desc = MakeDesc(); usdGenUsd::SetGraphDesc(session, desc);
    auto first = session.Commit(1, UsdGenCommitReason::SetTime);
    if (!first) for (auto const& e : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", e.c_str());
    CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
    auto lease = gpu::AcquireGeometry(first->device, stream); CHECK(lease);
    std::vector<float3> points; std::vector<float> widths; std::vector<uint64_t> ids;
    CHECK(Read(lease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({3,7}));
    for (float width : widths) CHECK(std::fabs(width - .5f) < 1e-5f);
    auto stats1 = session.CudaBindingStats();
    CHECK(stats1.size() == 1 && stats1[0].bindCount == 1 && stats1[0].solveCount == 1 && stats1[0].sampleCount == 5);
    uint64_t identity = stats1[0].identity;
    CHECK(std::fabs(points[0].x - .8f) < 2e-3f && std::fabs(points[0].y - .4f) < 2e-3f &&
          std::fabs(points[0].z - .2f) < 2e-3f);

    // Effective groom useRest, rather than its false authored fallback,
    // controls RBF admission and retains the last good generation on false.
    {
        UsdGenSession useRestSession; useRestSession.SetDevicePublicationEnabled(true);
        auto useRestDesc = desc;
        useRestDesc.nodes[0].params.push_back({TfToken("useRest"), VtValue(false), false});
        UsdGenExpressionDesc useRestExpr;
        useRestExpr.path = SdfPath("/UseRestExpr"); useRestExpr.source = "$frame < 2";
        useRestExpr.outputs.push_back({TfToken("result"), TfToken("bool"), Scalar(expr::ScalarType::Bool)});
        useRestDesc.expressions.push_back(useRestExpr);
        UsdGenExpressionBinding useRestBinding;
        useRestBinding.expression = useRestExpr.path; useRestBinding.destination = TfToken("useRest");
        useRestBinding.nativeType = TfToken("bool"); useRestBinding.destinationShape = Scalar(expr::ScalarType::Bool);
        useRestBinding.domain = expr::Domain::Groom; useRestBinding.literal = VtValue(false);
        useRestDesc.nodes[0].expressionBindings.push_back(useRestBinding);
        usdGenUsd::SetGraphDesc(useRestSession, useRestDesc);
        auto enabledRest = useRestSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(enabledRest && enabledRest->device && !useRestSession.LastDiagnostics().HasErrors());
        usdGenUsd::SetGraphDesc(useRestSession, useRestDesc);
        CHECK(useRestSession.Commit(2, UsdGenCommitReason::SetTime) == enabledRest &&
              useRestSession.LastDiagnostics().HasErrors());
    }

    // Resampling precedes the RBF chain.  Each published topology owns its
    // own device buffers, so the outstanding first-generation lease remains
    // readable while target counts change.
    {
        UsdGenSession resampleSession;
        resampleSession.SetDevicePublicationEnabled(true);
        usdGenUsd::SetGraphDesc(resampleSession, desc);
        auto resampleBase = resampleSession.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(resampleBase && resampleBase->device && !resampleSession.LastDiagnostics().HasErrors());
        auto retainedResampleLease = gpu::AcquireGeometry(resampleBase->device, stream);
        CHECK(retainedResampleLease);
        auto baseStats = resampleSession.CudaBindingStats();
        CHECK(baseStats.size() == 1 && baseStats[0].bindCount == 1);
        auto resampleFour = desc;
        resampleFour.nodes[0].params.push_back({TfToken("resampleTo"), VtValue(4), false});
        usdGenUsd::SetGraphDesc(resampleSession, resampleFour);
        auto four = resampleSession.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(four && four != resampleBase && four->device && !resampleSession.LastDiagnostics().HasErrors());
        auto fourLease = gpu::AcquireGeometry(four->device, stream); CHECK(fourLease);
        CHECK(Read(fourLease, &points, &widths, &ids, stream));
        CHECK(points.size() == 8 && ids == std::vector<uint64_t>({3,7}) &&
          four->device->Geometry().topologyVersion != resampleBase->device->Geometry().topologyVersion);
        for (size_t i = 0; i < points.size(); ++i) {
            CHECK(std::isfinite(points[i].x) && std::isfinite(points[i].y) && std::isfinite(points[i].z));
            CHECK(std::fabs(widths[i] - .5f) < 1e-5f);
        }
        auto fourStats = resampleSession.CudaBindingStats();
        CHECK(fourStats.size() == 1 && fourStats[0].identity == baseStats[0].identity &&
          fourStats[0].bindCount == baseStats[0].bindCount);
        CHECK(std::fabs(points[0].x - .8f) < 2e-3f && std::fabs(points[0].y - .4f) < 2e-3f &&
          std::fabs(points[0].z - .2f) < 2e-3f);
        CHECK(Read(retainedResampleLease, &points, &widths, &ids, stream));
        CHECK(points.size() == 5 && std::fabs(points[0].x - .8f) < 2e-3f &&
          ids == std::vector<uint64_t>({3,7}));
        auto resampleThree = resampleFour;
        resampleThree.nodes[0].params.front().value = VtValue(3);
        usdGenUsd::SetGraphDesc(resampleSession, resampleThree);
        auto three = resampleSession.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(three && three != four && three->device && !resampleSession.LastDiagnostics().HasErrors());
        auto threeLease = gpu::AcquireGeometry(three->device, stream); CHECK(threeLease);
        CHECK(Read(threeLease, &points, &widths, &ids, stream));
        CHECK(points.size() == 6 && ids == std::vector<uint64_t>({3,7}) &&
          three->device->Geometry().topologyVersion != four->device->Geometry().topologyVersion);
        auto threeStats = resampleSession.CudaBindingStats();
        CHECK(threeStats.size() == 1 && threeStats[0].identity == baseStats[0].identity &&
          threeStats[0].bindCount == baseStats[0].bindCount);
    } // resample leases/session complete before the shared stream is destroyed

    auto animated = desc; for (auto& p : animated.surfaces[0].points) p[0] += 1.f;
    usdGenUsd::SetGraphDesc(session, animated); auto second = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(second && second->device && second != first && !session.LastDiagnostics().HasErrors());
    auto stats2 = session.CudaBindingStats();
    CHECK(stats2.size() == 1 && stats2[0].identity == identity && stats2[0].bindCount == 1 && stats2[0].solveCount == 2);
    auto lease2 = gpu::AcquireGeometry(second->device, stream); CHECK(lease2);
    CHECK(Read(lease2, &points, &widths, &ids, stream));
    CHECK(std::fabs(points[0].x - 1.8f) < 2e-3f && std::fabs(points[0].y - .4f) < 2e-3f &&
          std::fabs(points[0].z - .2f) < 2e-3f && ids == std::vector<uint64_t>({3,7}));

    // Scrubbing back changes the solve inputs but reuses the same rest bind.
    usdGenUsd::SetGraphDesc(session, desc); auto scrubbed = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(scrubbed && scrubbed != second && !session.LastDiagnostics().HasErrors());
    auto scrubStats = session.CudaBindingStats();
    CHECK(scrubStats.size() == 1 && scrubStats[0].identity == identity && scrubStats[0].bindCount == 1);
    auto affine = desc;
    for (auto& p : affine.surfaces[0].points) { p[0] *= 2.f; p[1] *= .5f; }
    usdGenUsd::SetGraphDesc(session, affine); auto affineGeneration = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(affineGeneration && affineGeneration != scrubbed && !session.LastDiagnostics().HasErrors());
    auto affineLease = gpu::AcquireGeometry(affineGeneration->device, stream); CHECK(affineLease);
    CHECK(Read(affineLease, &points, &widths, &ids, stream));
    const std::vector<float3> affineExpected{{1.6f,.2f,.2f},{1.6f,.25f,.2f},{1.6f,.25f,.5f},
                                              {.4f,0,.2f},{.4f,.25f,.2f}};
    CHECK(points.size() == affineExpected.size());
    for (size_t i = 0; i < points.size(); ++i)
        CHECK(std::fabs(points[i].x - affineExpected[i].x) < 2e-3f &&
              std::fabs(points[i].y - affineExpected[i].y) < 2e-3f &&
              std::fabs(points[i].z - affineExpected[i].z) < 2e-3f);

    auto restChanged = animated; restChanged.surfaces[0].restPoints[0][0] += .25f;
    usdGenUsd::SetGraphDesc(session, restChanged); auto third = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(third && third->device && third != scrubbed && !session.LastDiagnostics().HasErrors());
    auto stats3 = session.CudaBindingStats();
    CHECK(stats3.size() == 1 && stats3[0].identity != identity && stats3[0].bindCount == 1);
    // Exercise execution-time controls at all supported rates: the groom
    // expression changes the structural sample budget, primitive lockRoots
    // pins only primitive 0 (the sorted ID-3 curve), and point blend halves motion.
    auto controlled = restChanged;
    UsdGenExpressionDesc samplesExpr;
    samplesExpr.path = SdfPath("/RbfSamplesExpr"); samplesExpr.source = "$frame < 2 ? 4 : 5";
    samplesExpr.outputs.push_back({TfToken("result"), TfToken("int"), Scalar(expr::ScalarType::Int32)});
    controlled.expressions.push_back(samplesExpr);
    UsdGenExpressionDesc lockExpr;
    lockExpr.path = SdfPath("/LockRootsExpr"); lockExpr.source = "$primIndex == 0";
    lockExpr.outputs.push_back({TfToken("result"), TfToken("bool"), Scalar(expr::ScalarType::Bool)});
    controlled.expressions.push_back(lockExpr);
    UsdGenExpressionDesc blendExpr;
    blendExpr.path = SdfPath("/PointMaskExpr"); blendExpr.source = "$P[0] * 0 + 0.5";
    blendExpr.outputs.push_back({TfToken("result"), TfToken("float"), Scalar(expr::ScalarType::Float32)});
    controlled.expressions.push_back(blendExpr);
    auto bindScalar = [](char const* path, char const* name, expr::Domain domain,
                         expr::ScalarType type, char const* native, VtValue literal) {
        UsdGenExpressionBinding b; b.expression = SdfPath(path); b.destination = TfToken(name);
        b.domain = domain; b.nativeType = TfToken(native); b.destinationShape = Scalar(type);
        b.literal = std::move(literal); return b;
    };
    controlled.nodes[2].expressionBindings.push_back(bindScalar(
        "/RbfSamplesExpr", "rbfSamples", expr::Domain::Groom, expr::ScalarType::Int32, "int", VtValue(4)));
    controlled.nodes[2].expressionBindings.push_back(bindScalar(
        "/LockRootsExpr", "lockRoots", expr::Domain::Primitive, expr::ScalarType::Bool, "bool", VtValue(false)));
    controlled.nodes[2].expressionBindings.push_back(bindScalar(
        "/PointMaskExpr", "mask", expr::Domain::Point, expr::ScalarType::Float32, "float", VtValue(.5f)));
    usdGenUsd::SetGraphDesc(session, controlled); auto controlledFirst = session.Commit(1, UsdGenCommitReason::SetTime);
    CHECK(controlledFirst && controlledFirst != third && !session.LastDiagnostics().HasErrors());
    auto controlledStats1 = session.CudaBindingStats();
    CHECK(controlledStats1.size() == 1 && controlledStats1[0].sampleCount == 4);
    auto controlledLease = gpu::AcquireGeometry(controlledFirst->device, stream); CHECK(controlledLease);
    CHECK(Read(controlledLease, &points, &widths, &ids, stream));
    CHECK(std::fabs(points[0].x - 1.3f) < 2e-3f && ids == std::vector<uint64_t>({3,7}));
    auto controlledSecond = session.Commit(2, UsdGenCommitReason::SetTime);
    CHECK(controlledSecond && controlledSecond != controlledFirst && !session.LastDiagnostics().HasErrors());
    auto controlledStats2 = session.CudaBindingStats();
    CHECK(controlledStats2.size() == 1 && controlledStats2[0].sampleCount == 5 &&
          controlledStats2[0].identity != controlledStats1[0].identity &&
          controlledStats2[0].bindCount == controlledStats1[0].bindCount + 1);
    auto nonAffine = controlled;
    nonAffine.surfaces[0].points[4][2] += .7f;
    usdGenUsd::SetGraphDesc(session, nonAffine); auto nonAffineGeneration = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(nonAffineGeneration && nonAffineGeneration != controlledSecond && !session.LastDiagnostics().HasErrors());
    auto nonAffineLease = gpu::AcquireGeometry(nonAffineGeneration->device, stream); CHECK(nonAffineLease);
    CHECK(Read(nonAffineLease, &points, &widths, &ids, stream));
    // ID 3 is primitive 0 and is explicitly root-locked by the expression;
    // its root follows the moved face vertex, then point blend halves motion.
    CHECK(std::fabs(points[0].z - .27f) < 2e-3f);
    const auto lockedPoints = points;
    // Compare with an independently cooked all-unlocked field. This catches
    // ignoring the primitive boolean: the first curve must change, the second
    // (whose expression already evaluates false) must remain identical.
    auto unlocked = nonAffine;
    unlocked.expressions[1].source = "$primIndex < 0";
    UsdGenSession unlockedSession;
    unlockedSession.SetDevicePublicationEnabled(true);
    usdGenUsd::SetGraphDesc(unlockedSession, unlocked);
    auto unlockedGeneration = unlockedSession.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(unlockedGeneration && unlockedGeneration->device && !unlockedSession.LastDiagnostics().HasErrors());
    auto unlockedLease = gpu::AcquireGeometry(unlockedGeneration->device, stream); CHECK(unlockedLease);
    CHECK(Read(unlockedLease, &points, &widths, &ids, stream));
    CHECK(std::fabs(points[0].z - lockedPoints[0].z) > 3e-3f);
    for (size_t i = 3; i < points.size(); ++i) {
        CHECK(std::fabs(points[i].x - lockedPoints[i].x) < 1e-5f);
        CHECK(std::fabs(points[i].y - lockedPoints[i].y) < 1e-5f);
        CHECK(std::fabs(points[i].z - lockedPoints[i].z) < 1e-5f);
    }
    unlockedLease = {};
    auto lastGood = nonAffineGeneration;
    auto bad = restChanged; bad.surfaces[0].restPoints[0][1] = std::numeric_limits<float>::quiet_NaN();
    usdGenUsd::SetGraphDesc(session, bad); CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    auto mismatch = restChanged;
    mismatch.curveSets[0].worldMatrix.SetTranslate(GfVec3d(1,0,0));
    usdGenUsd::SetGraphDesc(session, mismatch); CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    mismatch = restChanged;
    mismatch.nodes[0].surfaces[0] = SdfPath("/WrongSurface");
    usdGenUsd::SetGraphDesc(session, mismatch); CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    auto alreadyDeformed = restChanged;
    alreadyDeformed.nodes[0].params.push_back({TfToken("useRest"), VtValue(false), false});
    usdGenUsd::SetGraphDesc(session, alreadyDeformed); CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    CHECK(Read(lease, &points, &widths, &ids, stream));
    CHECK(std::fabs(points[0].x - .8f) < 2e-3f && ids == std::vector<uint64_t>({3,7}));
    bad = restChanged; bad.surfaces[0].restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,0},{1,1,0}};
    usdGenUsd::SetGraphDesc(session, bad); CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    auto doubleDeform = restChanged; auto extra = doubleDeform.nodes[2]; extra.path = SdfPath("/Ops/Deform2"); extra.inputs = {doubleDeform.nodes[3].path};
    doubleDeform.nodes.push_back(extra); doubleDeform.terminal = extra.path;
    usdGenUsd::SetGraphDesc(session, doubleDeform); CHECK(session.Commit(5, UsdGenCommitReason::SetTime) == lastGood);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    controlledLease = {}; affineLease = {}; nonAffineLease = {};
    lease2 = {}; lease = {}; CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaRbfSession: PASS"); return 0;
}
