#include "utils/pxrInterop.h"
#include "pxr/base/gf/vec2f.h"
#include "api/testBackend.h"
#include "pxr/base/gf/vec3f.h"
#include "usdGen/api/session.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/generation.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

static UsdGenGraphDesc MakeDesc() {
    UsdGenGraphDesc d; d.description = SdfPath("/Groom/Description");
    d.executionBackend = usdGenTestUtils::Backend("cuda"); d.defaultWidth = .025f;
    UsdGenSurfaceDesc surface; surface.path = SdfPath("/Scalp"); surface.worldMatrix.SetIdentity();
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    surface.points = surface.restPoints; surface.faceVertexCounts = {3,3,3};
    surface.faceVertexIndices = {0,1,2, 0,1,3, 1,2,4}; d.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves; curves.path = SdfPath("/Hair"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.type = TfToken("cubic"); curves.basis = TfToken("bspline"); curves.wrap = TfToken("pinned");
    curves.curveVertexCounts = {2,3,4}; curves.curveId = {30,10,20};
    curves.points = {{.2f,0,.2f},{.2f,.4f,.2f}, {.8f,.4f,.2f},{.8f,.6f,.2f},{.8f,.6f,.5f},
                     {.2f,.2f,0},{.3f,.3f,0},{.3f,.3f,.3f},{.3f,.3f,.6f}};
    curves.rest = curves.points; curves.skinPrim = {1,2,0}; curves.skinPrimUv = {{.2f,.2f},{.2f,.2f},{.2f,.2f}};
    d.curveSets.push_back(curves);
    UsdGenNodeDesc source; source.path = SdfPath("/Ops/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    UsdGenNodeDesc length; length.path = SdfPath("/Ops/Length"); length.type = TfToken("UsdGenLength"); length.inputs = {source.path};
    length.params.push_back({TfToken("length:mode"), VtValue(TfToken("cull")), false});
    length.params.push_back({TfToken("cullThreshold"), VtValue(.45f), false});
    UsdGenNodeDesc deform; deform.path = SdfPath("/Ops/Deform"); deform.type = TfToken("UsdGenDeform");
    deform.inputs = {length.path}; deform.surfaces = {surface.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false}); deform.params.push_back({TfToken("lockRoots"), VtValue(true), false});
    UsdGenNodeDesc width; width.path = SdfPath("/Ops/Width"); width.type = TfToken("UsdGenWidth"); width.inputs = {deform.path};
    width.params.push_back({TfToken("width"), VtValue(.2f), false});
    d.nodes = {source, length, deform, width}; d.terminal = width.path;
    UsdGenExpressionDesc valueExpr; valueExpr.path = SdfPath("/LengthValueExpr"); valueExpr.source = "$P[0] * 0 + 1";
    valueExpr.outputs.push_back({TfToken("result"), TfToken("float"), {expr::ScalarType::Float32,1,1,1,1,false}});
    d.expressions.push_back(valueExpr);
    UsdGenExpressionDesc randomExpr; randomExpr.path = SdfPath("/LengthRandomExpr"); randomExpr.source = "[$primIndex * 0 + 1, $primIndex * 0 + 1]";
    randomExpr.outputs.push_back({TfToken("result"), TfToken("float2"), {expr::ScalarType::Float32,1,2,1,1,false}});
    d.expressions.push_back(randomExpr);
    UsdGenExpressionDesc widthExpr; widthExpr.path = SdfPath("/WidthAfterExpr"); widthExpr.source = "$P[0] * 0 + 0.3";
    widthExpr.outputs.push_back({TfToken("result"), TfToken("float"), {expr::ScalarType::Float32,1,1,1,1,false}});
    d.expressions.push_back(widthExpr);
    auto bind = [](char const* path, char const* destination, expr::Domain domain,
                   expr::ValueShape shape, char const* native, VtValue literal) {
        UsdGenExpressionBinding b; b.expression = SdfPath(path); b.destination = TfToken(destination);
        b.domain = domain; b.destinationShape = shape; b.nativeType = TfToken(native); b.literal = std::move(literal); return b;
    };
    d.nodes[1].expressionBindings.push_back(bind("/LengthValueExpr", "length:value", expr::Domain::Point,
        {expr::ScalarType::Float32,1,1,1,1,false}, "float", VtValue(1.0f)));
    d.nodes[1].expressionBindings.push_back(bind("/LengthRandomExpr", "length:random", expr::Domain::Primitive,
        {expr::ScalarType::Float32,1,2,1,1,false}, "float2", VtValue(GfVec2f(1,1))));
    d.nodes[3].expressionBindings.push_back(bind("/WidthAfterExpr", "width", expr::Domain::Point,
        {expr::ScalarType::Float32,1,1,1,1,false}, "float", VtValue(.3f)));
    return d;
}

static bool Read(gpu::CudaGeometryLease const& lease, std::vector<float3>* points,
                 std::vector<float>* widths, std::vector<uint64_t>* ids, cudaStream_t stream) {
    auto v = lease.Geometry(); points->resize(v.pointCount); widths->resize(v.pointCount); ids->resize(v.curveCount);
    if (cudaMemcpyAsync(points->data(), v.points.data, points->size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(widths->data(), v.widths.data, widths->size()*sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(ids->data(), v.stableIds.data, ids->size()*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

template <class T>
static bool ReadNamed(gpu::CudaNamedChannelLease const& lease,
                      std::vector<T>* values, cudaStream_t stream) {
    if (!lease || !values || lease.Bytes().size % sizeof(T)) return false;
    values->resize(lease.Bytes().size / sizeof(T));
    return cudaMemcpyAsync(values->data(), lease.Bytes().data, lease.Bytes().size,
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

int main() {
    cudaStream_t stream = nullptr; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    UsdGenSession session; session.SetDevicePublicationEnabled(true); auto desc = MakeDesc();
    UsdGenAuthoredPlaneDesc pointPlane;
    pointPlane.name = TfToken("pointPayload");
    pointPlane.type = UsdGenAuthoredPlaneType::Float32;
    pointPlane.domain = UsdGenAuthoredPlaneDomain::Point;
    pointPlane.floatValues = {0,1,2,3,4,5,6,7,8};
    UsdGenAuthoredPlaneDesc primitivePlane;
    primitivePlane.name = TfToken("primitivePayload");
    primitivePlane.type = UsdGenAuthoredPlaneType::Int32;
    primitivePlane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primitivePlane.arity = 2;
    primitivePlane.intValues = {300,301,100,101,200,201};
    UsdGenAuthoredPlaneDesc groomPlane;
    groomPlane.name = TfToken("groomPayload");
    groomPlane.type = UsdGenAuthoredPlaneType::Float32;
    groomPlane.domain = UsdGenAuthoredPlaneDomain::Groom;
    groomPlane.floatValues = {.5f};
    desc.curveSets[0].authoredPlanes = {pointPlane, primitivePlane, groomPlane};
    usdGenUsd::SetGraphDesc(session, desc);
    auto first = session.Commit(1, UsdGenCommitReason::SetTime);
    if (!first) for (auto const& e : session.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", e.c_str());
    CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
    auto firstLease = gpu::AcquireGeometry(first->device, stream); CHECK(firstLease);
    std::vector<float3> points; std::vector<float> widths; std::vector<uint64_t> ids;
    CHECK(Read(firstLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20}) && points.size() == 7 && widths.size() == 7);
    for (float width : widths) CHECK(std::fabs(width - .3f) < 1e-5f);
    CHECK(firstLease.Geometry().curveOffsets.size == 3);
    uint32_t offsets[3]{};
    CHECK(cudaMemcpy(offsets, firstLease.Geometry().curveOffsets.data, sizeof(offsets), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(offsets[0] == 0 && offsets[1] == 3 && offsets[2] == 7);
    auto pointPayload = gpu::AcquireNamedChannel(first->device, TfToken("pointPayload"), stream);
    auto primitivePayload = gpu::AcquireNamedChannel(first->device, TfToken("primitivePayload"), stream);
    auto groomPayload = gpu::AcquireNamedChannel(first->device, TfToken("groomPayload"), stream);
    std::vector<float> pointValues, groomValues;
    std::vector<int32_t> primitiveValues;
    CHECK(ReadNamed(pointPayload, &pointValues, stream) &&
          pointValues == std::vector<float>({2,3,4,5,6,7,8}));
    CHECK(ReadNamed(primitivePayload, &primitiveValues, stream) &&
          primitiveValues == std::vector<int32_t>({100,101,200,201}));
    CHECK(ReadNamed(groomPayload, &groomValues, stream) &&
          groomValues == std::vector<float>({.5f}));
    std::vector<float> hairT(7); std::vector<int32_t> rootPrim(2); std::vector<float2> rootUV(2);
    CHECK(cudaMemcpyAsync(hairT.data(), firstLease.HairT().data, hairT.size()*sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaMemcpyAsync(rootPrim.data(), firstLease.RootPrim().data, rootPrim.size()*sizeof(int32_t), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaMemcpyAsync(rootUV.data(), firstLease.RootUV().data, rootUV.size()*sizeof(float2), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(hairT.size() == 7 && hairT == std::vector<float>({0.f,.5f,1.f,0.f,.33333334f,.6666667f,1.f}) &&
          rootPrim == std::vector<int32_t>({2,0}));
    CHECK(std::fabs(rootUV[0].x-.2f)<1e-6f && std::fabs(rootUV[0].y-.2f)<1e-6f &&
          std::fabs(rootUV[1].x-.2f)<1e-6f && std::fabs(rootUV[1].y-.2f)<1e-6f);
    std::vector<float3> rest(7);
    CHECK(cudaMemcpyAsync(rest.data(), firstLease.Geometry().restPoints.data,
                          rest.size()*sizeof(float3), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
          cudaStreamSynchronize(stream) == cudaSuccess);
    for (size_t i = 0; i < rest.size(); ++i) {
        auto expected = desc.curveSets[0].rest[i+2];
        CHECK(std::fabs(rest[i].x-expected[0])<1e-6f && std::fabs(rest[i].y-expected[1])<1e-6f &&
              std::fabs(rest[i].z-expected[2])<1e-6f);
    }

    auto malformed = desc; malformed.nodes[1].params[0].value = VtValue(TfToken("unknown"));
    usdGenUsd::SetGraphDesc(session, malformed); CHECK(session.Commit(2, UsdGenCommitReason::SetTime) == first);
    CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
    CHECK(Read(firstLease, &points, &widths, &ids, stream) && ids == std::vector<uint64_t>({10,20}));

    auto translated = desc;
    for (auto& p : translated.surfaces[0].points) p[0] += 1.f;
    usdGenUsd::SetGraphDesc(session, translated); auto translatedGeneration = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(translatedGeneration && translatedGeneration != first && !session.LastDiagnostics().HasErrors());
    auto translatedLease = gpu::AcquireGeometry(translatedGeneration->device, stream); CHECK(translatedLease);
    CHECK(Read(translatedLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20}) && std::fabs(points[0].x - 1.8f) < 2e-3f &&
          std::fabs(points[3].x - 1.2f) < 2e-3f);

    auto allRemoved = desc; allRemoved.nodes[1].params[1].value = VtValue(1000.0f);
    usdGenUsd::SetGraphDesc(session, allRemoved); auto empty = session.Commit(3, UsdGenCommitReason::SetTime);
    CHECK(empty && empty != translatedGeneration && !session.LastDiagnostics().HasErrors());
    auto emptyLease = gpu::AcquireGeometry(empty->device, stream); CHECK(emptyLease);
    CHECK(Read(emptyLease, &points, &widths, &ids, stream));
    CHECK(ids.empty() && points.empty() && widths.empty() && emptyLease.Geometry().curveOffsets.size == 1);
    auto emptyPoint = gpu::AcquireNamedChannel(empty->device, TfToken("pointPayload"), stream);
    auto emptyPrimitive = gpu::AcquireNamedChannel(empty->device, TfToken("primitivePayload"), stream);
    auto emptyGroom = gpu::AcquireNamedChannel(empty->device, TfToken("groomPayload"), stream);
    CHECK(ReadNamed(emptyPoint, &pointValues, stream) && pointValues.empty());
    CHECK(ReadNamed(emptyPrimitive, &primitiveValues, stream) && primitiveValues.empty());
    CHECK(ReadNamed(emptyGroom, &groomValues, stream) &&
          groomValues == std::vector<float>({.5f}));
    CHECK(cudaMemcpy(offsets, emptyLease.Geometry().curveOffsets.data, sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(offsets[0] == 0);
    CHECK(Read(firstLease, &points, &widths, &ids, stream) && ids == std::vector<uint64_t>({10,20}) && points.size() == 7);
    CHECK(ReadNamed(pointPayload, &pointValues, stream) &&
          pointValues == std::vector<float>({2,3,4,5,6,7,8}));

    // In non-cull modes length:value is evaluated, while cull ignores it.
    // Use a separate cook so the distinction is observable in the points.
    auto scaled = desc;
    scaled.nodes[1].expressionBindings.clear();
    scaled.nodes[1].params[0].value = VtValue(TfToken("scale"));
    scaled.nodes[1].params[1].value = VtValue(0.0f);
    scaled.nodes[1].params.push_back({TfToken("length:value"), VtValue(.5f), false});
    UsdGenExpressionDesc scaleExpr; scaleExpr.path = SdfPath("/ScaleLengthExpr"); scaleExpr.source = "$frame * 0 + 0.5";
    scaleExpr.outputs.push_back({TfToken("result"), TfToken("float"), {expr::ScalarType::Float32,1,1,1,1,false}});
    scaled.expressions.push_back(scaleExpr);
    UsdGenExpressionBinding scaleBinding; scaleBinding.expression = scaleExpr.path; scaleBinding.destination = TfToken("length:value");
    scaleBinding.nativeType = TfToken("float"); scaleBinding.destinationShape = {expr::ScalarType::Float32,1,1,1,1,false};
    scaleBinding.domain = expr::Domain::Point; scaleBinding.literal = VtValue(.5f);
    scaled.nodes[1].expressionBindings.push_back(scaleBinding);
    usdGenUsd::SetGraphDesc(session, scaled); auto scaledGeneration = session.Commit(4, UsdGenCommitReason::SetTime);
    CHECK(scaledGeneration && scaledGeneration != empty && !session.LastDiagnostics().HasErrors());
    auto scaledLease = gpu::AcquireGeometry(scaledGeneration->device, stream); CHECK(scaledLease);
    CHECK(Read(scaledLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20,30}) && points.size() == 9);
    // Sorted ID 30 is the final two-CV curve; scaling .4 units about its root
    // produces a .2-unit tip displacement.
    CHECK(std::fabs(points[8].y - .2f) < 2e-3f);

    auto keepParam = scaled;
    keepParam.nodes[1].expressionBindings.clear();
    keepParam.nodes[1].params[0].value = VtValue(TfToken("set"));
    keepParam.nodes[1].params[2].value = VtValue(.3f);
    keepParam.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    keepParam.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("keepParam")), false});
    usdGenUsd::SetGraphDesc(session, keepParam); auto cut = session.Commit(5, UsdGenCommitReason::SetTime);
    CHECK(cut && cut != scaledGeneration && !session.LastDiagnostics().HasErrors());
    auto cutLease = gpu::AcquireGeometry(cut->device, stream); CHECK(cutLease);
    CHECK(Read(cutLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20,30}) && points.size() == 9);
    CHECK(std::fabs(points[8].y - .3f) < 2e-3f);
    CHECK(std::fabs(points[1].y - .6f) < 2e-3f && std::fabs(points[2].z - .3f) < 2e-3f);
    auto reparam = keepParam;
    reparam.nodes[1].params.back().value = VtValue(TfToken("reparam"));
    usdGenUsd::SetGraphDesc(session, reparam); auto redistributed = session.Commit(6, UsdGenCommitReason::SetTime);
    CHECK(redistributed && redistributed != cut && !session.LastDiagnostics().HasErrors());
    auto redistributedLease = gpu::AcquireGeometry(redistributed->device, stream); CHECK(redistributedLease);
    CHECK(Read(redistributedLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20,30}) && points.size() == 9);
    CHECK(std::fabs(points[8].y - .3f) < 2e-3f);
    CHECK(std::fabs(points[1].y - .55f) < 2e-3f && std::fabs(points[2].z - .3f) < 2e-3f);

    // A second topology revision consumes the first one's reordered channels.
    auto twice = scaled;
    auto secondLength = twice.nodes[1];
    secondLength.path = SdfPath("/Ops/LengthAgain");
    secondLength.expressionBindings.clear();
    secondLength.params = {{TfToken("length:mode"), VtValue(TfToken("cull")), false},
                          {TfToken("cullThreshold"), VtValue(.3f), false}};
    secondLength.inputs = {twice.nodes[1].path};
    twice.nodes.insert(twice.nodes.begin() + 2, secondLength);
    twice.nodes[3].inputs = {secondLength.path};
    usdGenUsd::SetGraphDesc(session, twice);
    auto twiceGeneration = session.Commit(7, UsdGenCommitReason::SetTime);
    CHECK(twiceGeneration && twiceGeneration != redistributed && !session.LastDiagnostics().HasErrors());
    auto twiceLease = gpu::AcquireGeometry(twiceGeneration->device, stream); CHECK(twiceLease);
    CHECK(Read(twiceLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>{20} && points.size() == 4);
    int32_t twiceRoot = -1;
    CHECK(cudaMemcpy(&twiceRoot, twiceLease.RootPrim().data, sizeof(twiceRoot), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(twiceRoot == 0);
    CHECK(Read(firstLease, &points, &widths, &ids, stream) && ids == std::vector<uint64_t>({10,20}));
    twiceLease = {};

    auto timed = desc;
    UsdGenExpressionDesc thresholdExpr;
    thresholdExpr.path = SdfPath("/ThresholdExpr"); thresholdExpr.source = "$frame < 10 ? 0.45 : 0.9";
    thresholdExpr.outputs.push_back({TfToken("result"), TfToken("float"),
        {expr::ScalarType::Float32,1,1,1,1,false}});
    timed.expressions.push_back(thresholdExpr);
    auto thresholdBinding = scaleBinding;
    thresholdBinding.expression = thresholdExpr.path;
    thresholdBinding.destination = TfToken("cullThreshold");
    thresholdBinding.domain = expr::Domain::Groom;
    thresholdBinding.literal = VtValue(.45f);
    timed.nodes[1].expressionBindings.push_back(thresholdBinding);
    usdGenUsd::SetGraphDesc(session, timed);
    auto timedFirst = session.Commit(9, UsdGenCommitReason::SetTime);
    CHECK(timedFirst && timedFirst != twiceGeneration && !session.LastDiagnostics().HasErrors());
    auto timedLease = gpu::AcquireGeometry(timedFirst->device, stream); CHECK(timedLease);
    CHECK(timedLease.Geometry().curveCount == 2);
    auto timedSecond = session.Commit(10, UsdGenCommitReason::SetTime);
    CHECK(timedSecond && timedSecond != timedFirst && !session.LastDiagnostics().HasErrors());
    auto timedEmptyLease = gpu::AcquireGeometry(timedSecond->device, stream); CHECK(timedEmptyLease);
    CHECK(timedEmptyLease.Geometry().curveCount == 0 && timedEmptyLease.Geometry().pointCount == 0);
    timedLease = {}; timedEmptyLease = {};

    auto randomField = scaled;
    randomField.expressions[1].source = "[2,2]";
    randomField.nodes[1].expressionBindings.push_back(desc.nodes[1].expressionBindings[1]);
    usdGenUsd::SetGraphDesc(session, randomField);
    auto randomGeneration = session.Commit(11, UsdGenCommitReason::SetTime);
    CHECK(randomGeneration && randomGeneration != timedSecond && !session.LastDiagnostics().HasErrors());
    auto randomLease = gpu::AcquireGeometry(randomGeneration->device, stream); CHECK(randomLease);
    CHECK(Read(randomLease, &points, &widths, &ids, stream));
    CHECK(ids == std::vector<uint64_t>({10,20,30}) && points.size() == 9);
    CHECK(std::fabs(points[8].y - .4f) < 2e-3f); // .5 scale * connected [2,2]
    randomLease = {};

    pointPayload = {}; primitivePayload = {}; groomPayload = {};
    emptyPoint = {}; emptyPrimitive = {}; emptyGroom = {};
    translatedLease = {}; scaledLease = {}; cutLease = {}; redistributedLease = {};
    firstLease = {}; emptyLease = {}; CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaLengthSession: PASS"); return 0;
}
