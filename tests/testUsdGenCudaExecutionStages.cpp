#include "utils/pxrInterop.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "cudaPlanRunner.h"
#include "usdGenCurves/cuda/cudaParameters.h"
#include "usdGen/api/ext/device/executionResources.h"
#include "usdGenCurves/cuda/generation.h"

#include <cmath>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static size_t ResourceKindBytes(UsdGenExecutionResourceSnapshot const& snapshot,
                                UsdGenExecutionResourceKind kind) {
    return snapshot.byKind[static_cast<size_t>(kind)];
}

static UsdGenGraphDesc Desc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Stages/Hair");
    d.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc s; s.path = SdfPath("/Scalp");
    s.restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    s.points = s.restPoints; s.points[0][0] = 1;
    s.faceVertexCounts = {3,3,3}; s.faceVertexIndices = {0,1,2,0,1,3,1,2,4};
    d.surfaces.push_back(s);
    UsdGenCurveSetDesc c; c.path = SdfPath("/Curves"); c.role = UsdGenLane::Chunked;
    c.curveRole = TfToken("hair"); c.curveVertexCounts = {2};
    c.points = {{.2f,.2f,0},{.2f,.4f,0}}; c.rest = c.points;
    c.curveId = {7}; c.skinPrim = {0}; c.skinPrimUv = {{.2f,.2f}};
    d.curveSets.push_back(c);
    UsdGenNodeDesc source; source.path=SdfPath("/Stages/Hair/Source");
    source.type=TfToken("UsdGenCurveSource"); source.curves={c.path}; source.surfaces={s.path};
    UsdGenNodeDesc length; length.path=SdfPath("/Stages/Hair/Length");
    length.type=TfToken("UsdGenLength"); length.inputs={source.path};
    length.params.push_back({TfToken("length:value"), VtValue(.5f), false});
    UsdGenNodeDesc width; width.path=SdfPath("/Stages/Hair/Width");
    width.type=TfToken("UsdGenWidth"); width.inputs={length.path};
    width.params.push_back({TfToken("width"), VtValue(.03f), false});
    UsdGenNodeDesc deform; deform.path=SdfPath("/Stages/Hair/Deform");
    deform.type=TfToken("UsdGenDeform"); deform.inputs={width.path}; deform.surfaces={s.path};
    deform.params.push_back({TfToken("rbfSamples"), VtValue(5), false});
    d.nodes={source,length,width,deform}; d.terminal=deform.path;
    return d;
}

// Source controls are intentionally connected expressions: the authored
// fallbacks below are invalid at runtime and therefore prove the evaluated
// Bool/Int32 values, rather than the literals, reach source preparation.
static UsdGenGraphDesc SourceControlDesc(bool empty = false, bool withRbf = false) {
    auto d = Desc();
    if (withRbf) {
        auto deform = d.nodes.back();
        deform.inputs = {d.nodes.front().path};
        d.nodes = {d.nodes.front(), std::move(deform)};
    } else d.nodes.resize(1);
    d.terminal = d.nodes.back().path;
    auto& curves = d.curveSets.front();
    curves.points = {{2.2f,.2f,0},{2.2f,.4f,0}};
    curves.rest = {{.2f,.2f,0},{.2f,.4f,0}};
    if (empty) {
        curves.curveVertexCounts.clear(); curves.points.clear(); curves.rest.clear();
        curves.widths.clear(); curves.skinPrim.clear(); curves.curveId.clear();
        curves.skinPrimUv.clear(); curves.rootFrame.clear();
    }
    auto& source = d.nodes.front();
    source.params.push_back({TfToken("useRest"), VtValue(false), false});
    source.params.push_back({TfToken("resampleTo"), VtValue(-1), false});
    auto add = [&](char const* path, char const* expression, char const* destination,
                   TfToken native, expr::ScalarType scalar, VtValue literal) {
        expr::ValueShape shape{scalar, 1, 1, 1, 1, false};
        UsdGenExpressionDesc e; e.path = SdfPath(path); e.source = expression;
        e.outputs.push_back({TfToken("result"), native, shape}); d.expressions.push_back(std::move(e));
        UsdGenExpressionBinding binding; binding.expression = SdfPath(path);
        binding.destination = TfToken(destination); binding.domain = expr::Domain::Groom;
        binding.nativeType = native; binding.destinationShape = shape; binding.literal = std::move(literal);
        source.expressionBindings.push_back(std::move(binding));
    };
    add("/SourceUseRest", "$frame < 3", "useRest", TfToken("bool"), expr::ScalarType::Bool, VtValue(true));
    add("/SourceResample", "$frame < 2 ? 0 : ($frame < 3 ? 2 : 4)", "resampleTo", TfToken("int"), expr::ScalarType::Int32, VtValue(0));
    return d;
}

static UsdGenGraphDesc WidthExpressionDesc(bool invalid = false) {
    auto d = Desc();
    auto& width = d.nodes[2];
    auto add = [&](char const* path, char const* source, char const* destination,
                   expr::Domain domain, TfToken native, expr::ScalarType scalar,
                   VtValue literal) {
        UsdGenExpressionDesc expression;
        expression.path = SdfPath(path); expression.source = source;
        expr::ValueShape shape{scalar, 1, 1, 1, 1, false};
        expression.outputs.push_back({TfToken("result"), native, shape});
        d.expressions.push_back(std::move(expression));
        UsdGenExpressionBinding binding;
        binding.expression = SdfPath(path); binding.destination = TfToken(destination);
        binding.domain = domain; binding.nativeType = native;
        binding.destinationShape = shape; binding.literal = std::move(literal);
        width.expressionBindings.push_back(std::move(binding));
    };
    add("/WidthPoint", invalid ? "$frame / 0" : "$value * (0.5 + 0.5 * $t)",
        "width", expr::Domain::Point, TfToken("float"), expr::ScalarType::Float32, VtValue(.03f));
    add("/WidthPrimitive", "$value * (1 - 0.5 * $primIndex)", "mask",
        expr::Domain::Primitive, TfToken("float"), expr::ScalarType::Float32,
        VtValue(1.f));
    add("/WidthEnabled", "$frame >= 0", "enabled", expr::Domain::Groom,
        TfToken("bool"), expr::ScalarType::Bool, VtValue(true));
    add("/WidthReplace", "$primIndex == 0", "replace", expr::Domain::Primitive,
        TfToken("bool"), expr::ScalarType::Bool, VtValue(true));
    return d;
}

static UsdGenGraphDesc LengthExpressionDesc(bool cull = false, bool invalid = false) {
    auto d=Desc();
    auto& curves=d.curveSets.front();
    curves.curveVertexCounts={2,2};
    curves.points={{.2f,.2f,0},{.2f,.4f,0},{.7f,.2f,0},{.7f,2.2f,0}};
    curves.rest=curves.points; curves.curveId={7,8}; curves.skinPrim={0,0};
    curves.skinPrimUv={{.2f,.2f},{.7f,.2f}};
    UsdGenAuthoredPlaneDesc density;
    density.name=TfToken("density");
    density.type=UsdGenAuthoredPlaneType::Float32;
    density.domain=UsdGenAuthoredPlaneDomain::Point;
    density.arity=1;
    density.floatValues={1.f,2.f,3.f,4.f};
    curves.authoredPlanes={std::move(density)};
    auto& length=d.nodes[1];
    if (cull) {
        length.params.clear();
        length.params.push_back({TfToken("length:mode"),VtValue(TfToken("cull")),false});
        length.params.push_back({TfToken("cullThreshold"),VtValue(1.f),false});
    }
    auto add=[&](char const* path, char const* source, char const* destination,
                 expr::Domain domain, TfToken native, expr::ScalarType scalar, VtValue literal) {
        UsdGenExpressionDesc expression;
        expression.path=SdfPath(path); expression.source=source;
        expr::ValueShape shape{scalar,1,1,1,1,false};
        expression.outputs.push_back({TfToken("result"),native,shape});
        d.expressions.push_back(std::move(expression));
        UsdGenExpressionBinding binding;
        binding.expression=SdfPath(path); binding.destination=TfToken(destination);
        binding.domain=domain; binding.nativeType=native; binding.destinationShape=shape;
        binding.literal=std::move(literal);
        length.expressionBindings.push_back(std::move(binding));
    };
    add("/LengthPoint",invalid ? "$value / 0" : "$value * (0.5 + 0.5 * $t)",
        "length:value",expr::Domain::Point,TfToken("float"),expr::ScalarType::Float32,VtValue(.5f));
    // Scale mode still evaluates cullThreshold.  Keep it below both scaled
    // lengths so this non-cull fixture actually retains its two curves;
    // cull fixtures retain the threshold progression they are testing.
    add("/LengthPrimitive",cull ? "$value + $primIndex" : "$value + 0.01 * $primIndex",
        "cullThreshold",expr::Domain::Primitive,TfToken("float"),expr::ScalarType::Float32,
        VtValue(cull ? 1.f : 0.f));
    add("/LengthMask","$value","mask",expr::Domain::Primitive,
        TfToken("float"),expr::ScalarType::Float32,VtValue(1.f));
    add("/LengthEnabled","$frame >= 0","enabled",expr::Domain::Groom,
        TfToken("bool"),expr::ScalarType::Bool,VtValue(true));
    return d;
}

// A literal Length chain is runtime-refined for admission: the plan metadata
// remains conservative-unavailable, while the executor can query CUB and
// reserve a worst-case survivor bound before any stage submission.
static UsdGenGraphDesc LiteralLengthDesc(unsigned cullMode = 0) {
    auto d = Desc();
    auto& curves = d.curveSets.front();
    curves.curveVertexCounts = {2, 2};
    curves.points = {{.2f,.2f,0}, {.2f,.4f,0}, {.7f,.2f,0}, {.7f,2.2f,0}};
    curves.rest = curves.points;
    curves.curveId = {7, 8};
    curves.skinPrim = {0, 0};
    curves.skinPrimUv = {{.2f,.2f}, {.7f,.2f}};
    d.nodes.resize(3);
    d.terminal = d.nodes.back().path;
    d.expressions.clear();
    auto& length = d.nodes[1];
    length.expressionBindings.clear();
    length.params.clear();
    if (!cullMode) {
        length.params.push_back({TfToken("length:value"), VtValue(.5f), false});
    } else {
        length.params.push_back({TfToken("length:mode"),
                                 VtValue(TfToken("cull")), false});
        length.params.push_back({TfToken("cullThreshold"),
                                 VtValue(cullMode == 2 ? 100.f : 1.f), false});
    }
    return d;
}

static bool ParseLiteralLengthPeak(UsdGenDiagnostics const& diagnostics,
                                   uint64_t* peak) {
    if (!peak || diagnostics.errors.size() != 1 || !diagnostics.warnings.empty())
        return false;
    std::string const marker = "literal-Length memory reservation of ";
    auto const begin = diagnostics.errors.front().find(marker);
    if (begin == std::string::npos) return false;
    auto const number = begin + marker.size();
    auto const end = diagnostics.errors.front().find(" bytes", number);
    if (end == std::string::npos || end == number) return false;
    try {
        size_t consumed = 0;
        auto const value = std::stoull(diagnostics.errors.front().substr(
            number, end - number), &consumed);
        if (consumed != end - number || value == 0) return false;
        *peak = static_cast<uint64_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

static bool Same(std::shared_ptr<const UsdGenDeviceGeneration> const& a,
                 std::shared_ptr<const UsdGenDeviceGeneration> const& b,
                 cudaStream_t stream) {
    auto aa=gpu::AcquireGeometry(a,stream), bb=gpu::AcquireGeometry(b,stream);
    if (!aa || !bb || aa.Geometry().pointCount != bb.Geometry().pointCount ||
        aa.Geometry().curveCount != bb.Geometry().curveCount ||
        a->Geometry().tiles.size()!=b->Geometry().tiles.size()) return false;
    std::vector<float3> x(aa.Geometry().pointCount), y(bb.Geometry().pointCount);
    std::vector<float> wx(aa.Geometry().pointCount), wy(bb.Geometry().pointCount);
    auto bytes=x.size()*sizeof(float3);
    if (bytes && (cudaMemcpyAsync(x.data(),aa.Geometry().points.data,bytes,cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
        cudaMemcpyAsync(y.data(),bb.Geometry().points.data,bytes,cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
        cudaMemcpyAsync(wx.data(),aa.Geometry().widths.data,wx.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
        cudaMemcpyAsync(wy.data(),bb.Geometry().widths.data,wy.size()*sizeof(float),cudaMemcpyDeviceToHost,stream)!=cudaSuccess)) return false;
    if (cudaStreamSynchronize(stream)!=cudaSuccess) return false;
    for (size_t i=0;i<x.size();++i)
        if (!std::isfinite(x[i].x) || !std::isfinite(x[i].y) || !std::isfinite(x[i].z) ||
            !std::isfinite(y[i].x) || !std::isfinite(y[i].y) || !std::isfinite(y[i].z) ||
            std::fabs(x[i].x-y[i].x)>1e-5f || std::fabs(x[i].y-y[i].y)>1e-5f ||
            std::fabs(x[i].z-y[i].z)>1e-5f) return false;
    for (size_t i=0;i<wx.size();++i) if (!std::isfinite(wx[i]) || !std::isfinite(wy[i]) ||
        std::fabs(wx[i]-wy[i])>1e-5f || wx[i] <= 0) return false;
    return true;
}

static bool SameRetainedCurveChannels(std::shared_ptr<const UsdGenDeviceGeneration> const& a,
                                      std::shared_ptr<const UsdGenDeviceGeneration> const& b,
                                      cudaStream_t stream) {
    auto aa=gpu::AcquireGeometry(a,stream), bb=gpu::AcquireGeometry(b,stream);
    if (!aa || !bb || aa.Geometry().stableIds.size!=bb.Geometry().stableIds.size ||
        aa.HairT().size!=bb.HairT().size || aa.RootPrim().size!=bb.RootPrim().size ||
        aa.RootUV().size!=bb.RootUV().size) return false;
    std::vector<uint64_t> idsA(aa.Geometry().stableIds.size), idsB(bb.Geometry().stableIds.size);
    std::vector<float> hairTA(aa.HairT().size), hairTB(bb.HairT().size);
    std::vector<int32_t> rootsA(aa.RootPrim().size), rootsB(bb.RootPrim().size);
    std::vector<float2> uvA(aa.RootUV().size), uvB(bb.RootUV().size);
    auto copy=[&](auto& hostA, auto const& deviceA, auto& hostB, auto const& deviceB) {
        using T=typename std::decay_t<decltype(hostA)>::value_type;
        return (!hostA.empty() && (cudaMemcpyAsync(hostA.data(),deviceA.data,hostA.size()*sizeof(T),cudaMemcpyDeviceToHost,stream)!=cudaSuccess ||
                                   cudaMemcpyAsync(hostB.data(),deviceB.data,hostB.size()*sizeof(T),cudaMemcpyDeviceToHost,stream)!=cudaSuccess));
    };
    if (copy(idsA,aa.Geometry().stableIds,idsB,bb.Geometry().stableIds) ||
        copy(hairTA,aa.HairT(),hairTB,bb.HairT()) || copy(rootsA,aa.RootPrim(),rootsB,bb.RootPrim()) ||
        copy(uvA,aa.RootUV(),uvB,bb.RootUV()) || cudaStreamSynchronize(stream)!=cudaSuccess) return false;
    if (idsA!=idsB || rootsA!=rootsB) return false;
    for (size_t i=0;i<hairTA.size();++i) if (std::fabs(hairTA[i]-hairTB[i])>1e-6f) return false;
    for (size_t i=0;i<uvA.size();++i)
        if (std::fabs(uvA[i].x-uvB[i].x)>1e-6f || std::fabs(uvA[i].y-uvB[i].y)>1e-6f) return false;
    return true;
}

static bool SameFinalMetadata(std::shared_ptr<const UsdGenDeviceGeneration> const& a,
                              std::shared_ptr<const UsdGenDeviceGeneration> const& b) {
    if (!a || !b) return false;
    auto const& ag = a->Geometry(); auto const& bg = b->Geometry();
    if (ag.elementCount != bg.elementCount || ag.pointCount != bg.pointCount ||
        ag.topologyVersion != bg.topologyVersion || ag.valueVersion != bg.valueVersion ||
        ag.alreadyDeformed != bg.alreadyDeformed ||
        ag.layoutMetadata != bg.layoutMetadata || ag.tiles.size() != bg.tiles.size() ||
        a->Channels().size() != b->Channels().size()) return false;
    for (size_t i = 0; i != ag.tiles.size(); ++i) {
        auto const& x = ag.tiles[i]; auto const& y = bg.tiles[i];
        if (x.tile != y.tile || x.firstElement != y.firstElement || x.elementCount != y.elementCount ||
            x.firstPoint != y.firstPoint || x.pointCount != y.pointCount ||
            x.boundsValid != y.boundsValid || x.extentMin != y.extentMin || x.extentMax != y.extentMax)
            return false;
    }
    for (size_t i = 0; i != a->Channels().size(); ++i) {
        auto const& x = a->Channels()[i]; auto const& y = b->Channels()[i];
        if (x.name != y.name || x.type != y.type || x.domain != y.domain ||
            x.elementCount != y.elementCount || x.arity != y.arity ||
            x.strideBytes != y.strideBytes || x.readOnly != y.readOnly || x.semantic != y.semantic)
            return false;
    }
    return true;
}

static bool Wait(std::atomic<bool> const& done) {
    auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<deadline)
        std::this_thread::yield();
    return done.load(std::memory_order_acquire);
}

struct ReleaseSourceGate {
    ~ReleaseSourceGate() { releaseCudaSourceAsyncCallbackGateForTesting(); }
};
struct ReleaseSourceControlGate {
    void (*release)() = nullptr;
    ~ReleaseSourceControlGate() { if (release) release(); }
};
struct ReleaseResampleGate {
    ~ReleaseResampleGate() { releaseCudaSourceAsyncResampleCallbackGateForTesting(); }
};
struct ReleaseResampleLauncherGate {
    ~ReleaseResampleLauncherGate() {
        releaseCudaSourceAsyncResampleLauncherReturnGateForTesting();
    }
};
struct ResetRelayCapacity {
    ~ResetRelayCapacity() { setCudaSourceRelayCapacityForTesting(1024); }
};
struct ReleaseOperatorContextGate {
    ~ReleaseOperatorContextGate() { releaseCudaOperatorAsyncContextCallbackGateForTesting(); }
};
struct ReleaseOperatorProgramGate {
    ~ReleaseOperatorProgramGate() { releaseCudaOperatorAsyncProgramCallbackGateForTesting(); }
};
struct ReleaseOperatorWidthGate {
    ~ReleaseOperatorWidthGate() { releaseCudaOperatorAsyncWidthCallbackGateForTesting(); }
};
struct ReleaseOperatorWidthLauncherGate {
    ~ReleaseOperatorWidthLauncherGate() {
        releaseCudaOperatorAsyncWidthLauncherReturnGateForTesting();
    }
};
struct ResetOperatorRelayCapacity {
    ~ResetOperatorRelayCapacity() { setCudaOperatorRelayCapacityForTesting(1024); }
};
struct ReleaseOperatorLengthGate {
    ~ReleaseOperatorLengthGate() { releaseCudaOperatorAsyncLengthCallbackGateForTesting(); }
};
struct ReleaseOperatorCountsGate {
    ~ReleaseOperatorCountsGate() { releaseCudaOperatorAsyncCountsCallbackGateForTesting(); }
};
struct ReleaseOperatorScatterGate {
    ~ReleaseOperatorScatterGate() { releaseCudaOperatorAsyncScatterCallbackGateForTesting(); }
};
struct ReleaseOperatorGate {
    void (*release)() = nullptr;
    ~ReleaseOperatorGate() { if (release) release(); }
};
struct ResetFinalizationRelayCapacity {
    ~ResetFinalizationRelayCapacity() { setCudaFinalizationRelayCapacityForTesting(1024); }
};
struct ReleaseFinalizationGate {
    void (*release)() = nullptr;
    ~ReleaseFinalizationGate() { if (release) release(); }
};

int main() {
    UsdGenDiagnostics diagnostics;
    auto plan=CompileCudaGraph(usdGenUsd::Lower(Desc()),&diagnostics); CHECK(plan && !diagnostics.HasErrors());
    auto planMetadata = GetCudaExecutionPlanMetadata(*plan);
    // Length/RBF estimates intentionally retain their known components but
    // are not conservative admission bounds (scan/cuSOLVER cardinalities are
    // runtime-dependent), so the direct path must not precharge them.
    CHECK(planMetadata && !planMetadata->MemoryEstimate().memoryAvailable &&
          !planMetadata->MemoryEstimate().conservativeUpperBound);
    auto stagedWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto sequentialWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(stagedWorkspace && sequentialWorkspace);
    auto job=CreateCudaExecutionJob(plan,*stagedWorkspace,2,2,&diagnostics);
    CHECK(job && CudaExecutionJobOperatorCount(*job)==3);
    // Each stage runs on a freshly joined host thread: stage-local device
    // binding/restoration must not depend on a persistent worker thread.
    bool ok=false; std::thread source([&] { ok=RunCudaJobSourceForTesting(job); }); source.join(); CHECK(ok);
    for (size_t i=0;i<CudaExecutionJobOperatorCount(*job);++i) {
        std::thread op([&] { ok=RunCudaJobOperatorForTesting(job,i); }); op.join(); CHECK(ok);
    }
    std::shared_ptr<const UsdGenDeviceGeneration> staged;
    std::thread final([&] { staged=FinalizeCudaJobForTesting(job); }); final.join(); CHECK(staged);
    int initialResourceDevice = -1;
    CHECK(cudaGetDevice(&initialResourceDevice) == cudaSuccess);
    auto initialResourcePool = FindUsdGenExecutionResourcePool(
        {usdGenTestUtils::Backend("cuda"), initialResourceDevice});
    CHECK(initialResourcePool);
    auto const unknownBefore = initialResourcePool->Snapshot();
    auto sequential=ExecuteCudaPlanForTesting(plan,*sequentialWorkspace,2,2,&diagnostics); CHECK(sequential);
    auto const unknownAfter = initialResourcePool->Snapshot();
    CHECK(ResourceKindBytes(unknownAfter, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(unknownBefore, UsdGenExecutionResourceKind::Pending) &&
          ResourceKindBytes(unknownAfter, UsdGenExecutionResourceKind::Pending) == 0);
    // The synchronous RBF finalization seam is deterministic and exercises
    // failure cleanup on the unknown-estimate direct path.  Destroy the fresh
    // workspace before observing the ledger so its cache/program state cannot
    // obscure a leaked Pending balance.
    auto syncFailureWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(syncFailureWorkspace);
    auto const syncFailureBefore = initialResourcePool->Snapshot();
    failNextCudaRbfFinalizationForTesting();
    diagnostics = {};
    auto syncFailure=ExecuteCudaPlanForTesting(plan,*syncFailureWorkspace,2,3,&diagnostics);
    CHECK(!syncFailure && diagnostics.HasErrors());
    syncFailureWorkspace.reset();
    auto const syncFailureAfter = initialResourcePool->Snapshot();
    CHECK(syncFailureAfter.usedBytes == syncFailureBefore.usedBytes &&
          ResourceKindBytes(syncFailureAfter, UsdGenExecutionResourceKind::Pending) == 0);

    // Literal Length has no truthful compile-time estimate because CUB's
    // selected-device scan size is runtime data, but the executor can refine
    // a complete linear Source/Length/Width plan before it submits work.
    auto literalLengthDesc = LiteralLengthDesc();
    diagnostics = {};
    auto literalLengthPlan = CompileCudaGraph(usdGenUsd::Lower(literalLengthDesc), &diagnostics);
    CHECK(literalLengthPlan && !diagnostics.HasErrors());
    auto literalLengthMetadata = GetCudaExecutionPlanMetadata(*literalLengthPlan);
    CHECK(literalLengthMetadata &&
          !literalLengthMetadata->MemoryEstimate().memoryAvailable &&
          !literalLengthMetadata->MemoryEstimate().conservativeUpperBound &&
          literalLengthMetadata->MemoryEstimate().runtimeRefinementAvailable);
    auto literalAdmissionWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(literalAdmissionWorkspace);
    diagnostics = {};
    auto literalFirst = ExecuteCudaPlanForTesting(literalLengthPlan, *literalAdmissionWorkspace,
                                         2, 30, &diagnostics);
    CHECK(literalFirst && !diagnostics.HasErrors() &&
          literalFirst->Geometry().elementCount == 2 &&
          literalFirst->Geometry().pointCount == 4);
    int literalDevice = -1;
    CHECK(cudaGetDevice(&literalDevice) == cudaSuccess);
    auto literalResources = FindUsdGenExecutionResourcePool(
        {usdGenTestUtils::Backend("cuda"), literalDevice});
    CHECK(literalResources);
    auto const literalBaseline = literalResources->Snapshot();
    CHECK(literalBaseline.usableBytes >= literalBaseline.usedBytes);
    auto const literalAvailable = literalBaseline.usableBytes -
        literalBaseline.usedBytes;

    // Saturating the pool once reveals the exact runtime-refined peak in the
    // deterministic admission diagnostic, without observing physical device
    // free-memory deltas. The failed call must not replace the last good
    // generation or mutate the resource ledger.
    auto fullFiller = literalResources->TryReserve(
        literalAvailable, UsdGenExecutionResourceKind::Active);
    CHECK(fullFiller);
    auto const fullySaturated = literalResources->Snapshot();
    diagnostics = {};
    auto fullyRejected = ExecuteCudaPlanForTesting(literalLengthPlan, *literalAdmissionWorkspace,
                                          2, 31, &diagnostics, literalFirst);
    uint64_t literalPeak = 0;
    CHECK(!fullyRejected && ParseLiteralLengthPeak(diagnostics, &literalPeak));
    auto const afterFullReject = literalResources->Snapshot();
    CHECK(afterFullReject.usedBytes == fullySaturated.usedBytes &&
          ResourceKindBytes(afterFullReject, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(fullySaturated, UsdGenExecutionResourceKind::Pending));

    // A resampler remains owned after Length publication. With a second
    // Length, source + resample + current compacted + candidate geometries
    // overlap; the runtime bound must grow rather than treating the current
    // compacted value as a replacement for the retained resample owner.
    auto oneResampledLengthDesc = LiteralLengthDesc();
    oneResampledLengthDesc.nodes.resize(2);
    oneResampledLengthDesc.terminal = oneResampledLengthDesc.nodes.back().path;
    oneResampledLengthDesc.nodes.front().params.push_back(
        {TfToken("resampleTo"), VtValue(3), false});
    auto twoResampledLengthsDesc = oneResampledLengthDesc;
    auto secondLength = twoResampledLengthsDesc.nodes.back();
    secondLength.path = SdfPath("/Stages/Hair/LengthSecond");
    secondLength.inputs = {twoResampledLengthsDesc.nodes.back().path};
    twoResampledLengthsDesc.nodes.push_back(std::move(secondLength));
    twoResampledLengthsDesc.terminal = twoResampledLengthsDesc.nodes.back().path;
    diagnostics = {};
    auto oneResampledLengthPlan = CompileCudaGraph(usdGenUsd::Lower(oneResampledLengthDesc), &diagnostics);
    auto twoResampledLengthsPlan = CompileCudaGraph(usdGenUsd::Lower(twoResampledLengthsDesc), &diagnostics);
    CHECK(oneResampledLengthPlan && twoResampledLengthsPlan && !diagnostics.HasErrors());
    diagnostics = {};
    auto rejectedOneResampledLength = ExecuteCudaPlanForTesting(
        oneResampledLengthPlan, *literalAdmissionWorkspace, 2, 311,
        &diagnostics, literalFirst);
    uint64_t oneResampledLengthPeak = 0;
    CHECK(!rejectedOneResampledLength &&
          ParseLiteralLengthPeak(diagnostics, &oneResampledLengthPeak));
    diagnostics = {};
    auto rejectedTwoResampledLengths = ExecuteCudaPlanForTesting(
        twoResampledLengthsPlan, *literalAdmissionWorkspace, 2, 312,
        &diagnostics, literalFirst);
    uint64_t twoResampledLengthsPeak = 0;
    CHECK(!rejectedTwoResampledLengths &&
          ParseLiteralLengthPeak(diagnostics, &twoResampledLengthsPeak) &&
          twoResampledLengthsPeak > oneResampledLengthPeak);
    auto const afterResampleRejects = literalResources->Snapshot();
    CHECK(afterResampleRejects.usedBytes == fullySaturated.usedBytes &&
          ResourceKindBytes(afterResampleRejects,
              UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(fullySaturated,
                  UsdGenExecutionResourceKind::Pending));
    fullFiller->Release();
    CHECK(literalResources->Snapshot().usedBytes == literalBaseline.usedBytes);
    CHECK(literalPeak <= literalAvailable);

    auto const peakFillerBytes = literalAvailable - static_cast<size_t>(literalPeak) + 1;
    auto peakFiller = literalResources->TryReserve(
        peakFillerBytes, UsdGenExecutionResourceKind::Active);
    CHECK(peakFiller);
    auto const peakSaturated = literalResources->Snapshot();
    diagnostics = {};
    auto peakRejected = ExecuteCudaPlanForTesting(literalLengthPlan, *literalAdmissionWorkspace,
                                         2, 32, &diagnostics, literalFirst);
    uint64_t repeatedPeak = 0;
    CHECK(!peakRejected && ParseLiteralLengthPeak(diagnostics, &repeatedPeak) &&
          repeatedPeak == literalPeak);
    auto const afterPeakReject = literalResources->Snapshot();
    CHECK(afterPeakReject.usedBytes == peakSaturated.usedBytes &&
          ResourceKindBytes(afterPeakReject, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(peakSaturated, UsdGenExecutionResourceKind::Pending));
    peakFiller->Release();
    CHECK(literalResources->Snapshot().usedBytes == literalBaseline.usedBytes);

    diagnostics = {};
    auto literalSecond = ExecuteCudaPlanForTesting(literalLengthPlan, *literalAdmissionWorkspace,
                                          2, 33, &diagnostics, literalFirst);
    CHECK(literalSecond && !diagnostics.HasErrors() && literalSecond->Owner() &&
          literalSecond->Owner() != literalFirst->Owner());
    auto const afterLiteralDirect = literalResources->Snapshot();
    CHECK(ResourceKindBytes(afterLiteralDirect, UsdGenExecutionResourceKind::Pending) == 0 &&
          afterLiteralDirect.usedBytes > literalBaseline.usedBytes &&
          afterLiteralDirect.usedBytes <= literalBaseline.usedBytes + literalPeak);

    // Runtime admission is shared by the staged asynchronous boundary. A
    // held CreateCudaExecutionJob owns Pending before its first callback; the
    // same job then publishes the async result with Pending fully retired.
    auto asyncLiteralWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    CHECK(asyncLiteralWorkspace);
    auto const asyncRejectBaseline = literalResources->Snapshot();
    auto const asyncAvailable = asyncRejectBaseline.usableBytes -
        asyncRejectBaseline.usedBytes;
    CHECK(asyncAvailable >= literalPeak);
    auto asyncFiller = literalResources->TryReserve(
        asyncAvailable - static_cast<size_t>(literalPeak) + 1,
        UsdGenExecutionResourceKind::Active);
    CHECK(asyncFiller);
    auto const asyncSaturated = literalResources->Snapshot();
    diagnostics = {};
    auto asyncRejected = CreateCudaExecutionJob(
        literalLengthPlan, *asyncLiteralWorkspace, 2, 34, &diagnostics, literalSecond);
    uint64_t asyncPeak = 0;
    CHECK(!asyncRejected && ParseLiteralLengthPeak(diagnostics, &asyncPeak) &&
          asyncPeak == literalPeak);
    auto const afterAsyncReject = literalResources->Snapshot();
    CHECK(afterAsyncReject.usedBytes == asyncSaturated.usedBytes &&
          ResourceKindBytes(afterAsyncReject, UsdGenExecutionResourceKind::Pending) ==
              ResourceKindBytes(asyncSaturated, UsdGenExecutionResourceKind::Pending));
    asyncFiller->Release();
    CHECK(literalResources->Snapshot().usedBytes == asyncRejectBaseline.usedBytes);

    diagnostics = {};
    auto asyncLiteralJob = CreateCudaExecutionJob(
        literalLengthPlan, *asyncLiteralWorkspace, 2, 35, &diagnostics, literalSecond);
    CHECK(asyncLiteralJob);
    auto const asyncHeld = literalResources->Snapshot();
    CHECK(ResourceKindBytes(asyncHeld, UsdGenExecutionResourceKind::Pending) >
          ResourceKindBytes(asyncRejectBaseline, UsdGenExecutionResourceKind::Pending));
    std::atomic<bool> literalSourceDone{false};
    std::atomic<bool> literalSourceOk{false};
    CHECK(ExecuteCudaJobSourceAsync(asyncLiteralJob, [&](bool result) {
        literalSourceOk.store(result, std::memory_order_release);
        literalSourceDone.store(true, std::memory_order_release);
    }) && Wait(literalSourceDone) && literalSourceOk.load(std::memory_order_acquire));
    for (size_t i = 0; i != CudaExecutionJobOperatorCount(*asyncLiteralJob); ++i) {
        std::atomic<bool> operatorDone{false};
        std::atomic<bool> operatorOk{false};
        CHECK(ExecuteCudaJobOperatorAsync(asyncLiteralJob, i, [&](bool result) {
            operatorOk.store(result, std::memory_order_release);
            operatorDone.store(true, std::memory_order_release);
        }) && Wait(operatorDone) && operatorOk.load(std::memory_order_acquire));
    }
    auto asyncLiteralGeneration = FinalizeCudaJobForTesting(asyncLiteralJob);
    CHECK(asyncLiteralGeneration && asyncLiteralGeneration->Geometry().elementCount == 2 &&
          asyncLiteralGeneration->Geometry().pointCount == 4);
    auto const afterLiteralAsync = literalResources->Snapshot();
    CHECK(ResourceKindBytes(afterLiteralAsync, UsdGenExecutionResourceKind::Pending) == 0);

    // Literal culling exercises the exact survivor cardinality while keeping
    // the same conservative reservation bound. The retained child owner is
    // private to each publication, including the all-cull zero-cardinality
    // result, and no Pending balance survives the synchronous boundary.
    auto partialLiteralDesc = LiteralLengthDesc(1);
    auto partialLiteralPlan = CompileCudaGraph(usdGenUsd::Lower(partialLiteralDesc), &diagnostics);
    CHECK(partialLiteralPlan);
    diagnostics = {};
    auto partialLiteral = ExecuteCudaPlanForTesting(partialLiteralPlan, *literalAdmissionWorkspace,
                                           2, 36, &diagnostics, asyncLiteralGeneration);
    CHECK(partialLiteral && !diagnostics.HasErrors() &&
          partialLiteral->Geometry().elementCount == 1 &&
          partialLiteral->Geometry().pointCount == 2 && partialLiteral->Owner() &&
          partialLiteral->Owner() != asyncLiteralGeneration->Owner() &&
          ResourceKindBytes(literalResources->Snapshot(),
              UsdGenExecutionResourceKind::Pending) == 0);
    auto allCullLiteralDesc = LiteralLengthDesc(2);
    auto allCullLiteralPlan = CompileCudaGraph(usdGenUsd::Lower(allCullLiteralDesc), &diagnostics);
    CHECK(allCullLiteralPlan);
    diagnostics = {};
    auto allCullLiteral = ExecuteCudaPlanForTesting(allCullLiteralPlan, *literalAdmissionWorkspace,
                                           2, 37, &diagnostics, partialLiteral);
    CHECK(allCullLiteral && !diagnostics.HasErrors() &&
          allCullLiteral->Geometry().elementCount == 0 &&
          allCullLiteral->Geometry().pointCount == 0 && allCullLiteral->Owner() &&
          allCullLiteral->Owner() != partialLiteral->Owner() &&
          ResourceKindBytes(literalResources->Snapshot(),
              UsdGenExecutionResourceKind::Pending) == 0);
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(staged,sequential,stream)); CHECK(cudaStreamDestroy(stream)==cudaSuccess);

    // The shared source relay returns after upload submission, then commits
    // exactly once only after the stream callback passes its post-Set gate.
    auto asyncJob=CreateCudaExecutionJob(plan,*stagedWorkspace,2,22,&diagnostics); CHECK(asyncJob);
    std::atomic<int> asyncCalls{0}; std::atomic<bool> asyncDone{false}, asyncOk{false};
    armCudaSourceAsyncCallbackGateForTesting();
    ReleaseSourceGate releaseGate;
    CHECK(ExecuteCudaJobSourceAsync(asyncJob,[&](bool ok) {
        asyncOk.store(ok,std::memory_order_release); asyncCalls.fetch_add(1,std::memory_order_relaxed);
        asyncDone.store(true,std::memory_order_release);
    }));
    waitCudaSourceAsyncCallbackGateForTesting();
    CHECK(!asyncDone.load(std::memory_order_acquire) && asyncCalls == 0);
    releaseCudaSourceAsyncCallbackGateForTesting();
    CHECK(Wait(asyncDone) && asyncOk.load(std::memory_order_acquire) && asyncCalls == 1);
    for (size_t i=0;i<CudaExecutionJobOperatorCount(*asyncJob);++i)
        CHECK(RunCudaJobOperatorForTesting(asyncJob,i));
    auto asyncGeneration=FinalizeCudaJobForTesting(asyncJob); CHECK(asyncGeneration);
    cudaStream_t asyncStream=nullptr; CHECK(cudaStreamCreateWithFlags(&asyncStream,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(asyncGeneration,sequential,asyncStream));
    CHECK(cudaStreamDestroy(asyncStream)==cudaSuccess);

    // Deform follows the same relay contract as Width/Length, but spans the
    // rest bind (when cold), pose solve, RBF evaluation, apply and private
    // copy proofs.  The candidate points and coefficients are not installed
    // until the final callback has completed.
    auto rbfAsyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto rbfSyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(rbfAsyncWorkspace && rbfSyncWorkspace);
    auto rbfJob=CreateCudaExecutionJob(plan,*rbfAsyncWorkspace,2,23,&diagnostics);
    CHECK(rbfJob && RunCudaJobSourceForTesting(rbfJob) && RunCudaJobOperatorForTesting(rbfJob,0) &&
          RunCudaJobOperatorForTesting(rbfJob,1));
    std::atomic<int> rbfCalls{0}; std::atomic<bool> rbfDone{false}, rbfOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(rbfJob,2,[&](bool value) {
        rbfOk.store(value,std::memory_order_release); rbfCalls.fetch_add(1,std::memory_order_relaxed);
        rbfDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(rbfDone) && rbfOk.load(std::memory_order_acquire) && rbfCalls==1);
    std::shared_ptr<const UsdGenDeviceGeneration> rbfAsyncGeneration;
    std::atomic<bool> rbfFinalDone{false}; std::atomic<int> rbfFinalCalls{0};
    CHECK(FinalizeCudaExecutionJobAsync(rbfJob,[&](auto generation) {
        rbfAsyncGeneration=std::move(generation); ++rbfFinalCalls;
        rbfFinalDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(rbfFinalDone) && rbfAsyncGeneration && rbfFinalCalls==1);
    auto rbfSyncGeneration=ExecuteCudaPlanForTesting(plan,*rbfSyncWorkspace,2,24,&diagnostics); CHECK(rbfSyncGeneration);
    cudaStream_t rbfReader=nullptr; CHECK(cudaStreamCreateWithFlags(&rbfReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(rbfAsyncGeneration,rbfSyncGeneration,rbfReader));
    CHECK(cudaStreamDestroy(rbfReader)==cudaSuccess);

    // Source Bool/Int32 controls are evaluated and proved before scalar
    // readback/upload.  useRest declares the source space; it never replaces
    // loaded posed points with canonical rest.  Both buffers remain visible,
    // while alreadyDeformed records the selected space.
    auto sourceControlPlan=CompileCudaGraph(usdGenUsd::Lower(SourceControlDesc()),&diagnostics); CHECK(sourceControlPlan);
    auto sourceControlMetadata = GetCudaExecutionPlanMetadata(*sourceControlPlan);
    CHECK(sourceControlMetadata &&
          !sourceControlMetadata->MemoryEstimate().memoryAvailable &&
          !sourceControlMetadata->MemoryEstimate().conservativeUpperBound);
    auto controlRun=[&](double frame, uint64_t generation, size_t points, float posedFirstX,
                        float restFirstX, bool expectedAlreadyDeformed) {
        auto asyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        auto legacyWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!asyncWorkspace || !legacyWorkspace) return false;
        auto controlJob=CreateCudaExecutionJob(sourceControlPlan,*asyncWorkspace,frame,generation,&diagnostics);
        if (!controlJob) return false;
        std::atomic<bool> done{false}, ok{false}; std::atomic<int> calls{0};
        bool const accepted=ExecuteCudaJobSourceAsync(controlJob,[&](bool value) {
            ok.store(value,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
            done.store(true,std::memory_order_release);
        });
        bool const completed=accepted && Wait(done);
        bool const sourceOk=completed && ok.load(std::memory_order_acquire);
        if (!sourceOk) {
            std::fprintf(stderr,
                "source-control frame=%g generation=%llu accepted=%d done=%d calls=%d ok=%d\n",
                frame,static_cast<unsigned long long>(generation),accepted,
                done.load(std::memory_order_acquire),calls.load(std::memory_order_relaxed),
                ok.load(std::memory_order_acquire));
            return false;
        }
        auto stagedControl=FinalizeCudaJobForTesting(controlJob);
        int controlDevice = -1;
        if (cudaGetDevice(&controlDevice) != cudaSuccess) return false;
        auto controlPool = FindUsdGenExecutionResourcePool(
            {usdGenTestUtils::Backend("cuda"), controlDevice});
        if (!controlPool) return false;
        auto const controlBefore = controlPool->Snapshot();
        auto legacyControl=ExecuteCudaPlanForTesting(sourceControlPlan,*legacyWorkspace,frame,generation,&diagnostics);
        auto const controlAfter = controlPool->Snapshot();
        if (ResourceKindBytes(controlAfter, UsdGenExecutionResourceKind::Pending) !=
                ResourceKindBytes(controlBefore, UsdGenExecutionResourceKind::Pending) ||
            ResourceKindBytes(controlAfter, UsdGenExecutionResourceKind::Pending) != 0)
            return false;
        cudaStream_t controlStream=nullptr;
        cudaError_t const streamStatus=cudaStreamCreateWithFlags(&controlStream,cudaStreamNonBlocking);
        if (!stagedControl || !legacyControl || streamStatus!=cudaSuccess) {
            std::fprintf(stderr,
                "source-control frame=%g generation=%llu staged=%d legacy=%d stream=%d calls=%d\n",
                frame,static_cast<unsigned long long>(generation),static_cast<bool>(stagedControl),
                static_cast<bool>(legacyControl),static_cast<int>(streamStatus),
                calls.load(std::memory_order_relaxed));
            if (controlStream) cudaStreamDestroy(controlStream);
            return false;
        }
        auto lease=gpu::AcquireGeometry(stagedControl,controlStream);
        auto legacyLease=gpu::AcquireGeometry(legacyControl,controlStream);
        float3 point{}, restPoint{}, legacyRestPoint{};
        size_t const stagedCurves=lease ? lease.Geometry().curveCount : 0;
        size_t const stagedPoints=lease ? lease.Geometry().pointCount : 0;
        size_t const legacyCurves=legacyLease ? legacyLease.Geometry().curveCount : 0;
        size_t const legacyPoints=legacyLease ? legacyLease.Geometry().pointCount : 0;
        bool const counts=lease && legacyLease && stagedCurves==1 && stagedPoints==points &&
            legacyCurves==1 && legacyPoints==points;
        bool const sameGeometry=lease && Same(stagedControl,legacyControl,controlStream);
        bool const sameMetadata=SameFinalMetadata(stagedControl,legacyControl);
        cudaError_t copyStatus=cudaErrorUnknown, syncStatus=cudaErrorUnknown;
        if (lease && legacyLease && lease.Geometry().points.data && lease.Geometry().restPoints.data &&
            legacyLease.Geometry().restPoints.data && stagedPoints!=0) {
            copyStatus=cudaMemcpyAsync(&point,lease.Geometry().points.data,sizeof(point),cudaMemcpyDeviceToHost,controlStream);
            if (copyStatus==cudaSuccess)
                copyStatus=cudaMemcpyAsync(&restPoint,lease.Geometry().restPoints.data,sizeof(restPoint),cudaMemcpyDeviceToHost,controlStream);
            if (copyStatus==cudaSuccess)
                copyStatus=cudaMemcpyAsync(&legacyRestPoint,legacyLease.Geometry().restPoints.data,sizeof(legacyRestPoint),cudaMemcpyDeviceToHost,controlStream);
            if (copyStatus==cudaSuccess) syncStatus=cudaStreamSynchronize(controlStream);
        }
        bool const pointsExpected=copyStatus==cudaSuccess && syncStatus==cudaSuccess &&
            std::fabs(point.x-posedFirstX)<1e-5f && std::fabs(restPoint.x-restFirstX)<1e-5f &&
            std::fabs(legacyRestPoint.x-restFirstX)<1e-5f;
        bool const deformed=stagedControl->Geometry().alreadyDeformed==expectedAlreadyDeformed &&
            legacyControl->Geometry().alreadyDeformed==expectedAlreadyDeformed;
        bool const same=counts && sameGeometry && sameMetadata && pointsExpected && deformed &&
            calls.load(std::memory_order_relaxed)==1;
        if (!same) {
            std::fprintf(stderr,
                "source-control frame=%g generation=%llu calls=%d leases=(%d,%d) staged=(%zu,%zu) legacy=(%zu,%zu) expected=(1,%zu,posed=%g,rest=%g,deformed=%d) geometry=%d metadata=%d copy=%d sync=%d posedX=%g restX=(%g,%g) deformed=(%d,%d)\n",
                frame,static_cast<unsigned long long>(generation),calls.load(std::memory_order_relaxed),
                static_cast<bool>(lease),static_cast<bool>(legacyLease),stagedCurves,stagedPoints,
                legacyCurves,legacyPoints,points,posedFirstX,restFirstX,expectedAlreadyDeformed,
                sameGeometry,sameMetadata,static_cast<int>(copyStatus),static_cast<int>(syncStatus),
                point.x,restPoint.x,legacyRestPoint.x,stagedControl->Geometry().alreadyDeformed,
                legacyControl->Geometry().alreadyDeformed);
        }
        legacyLease={}; lease={}; cudaStreamDestroy(controlStream); return same;
    };
    CHECK(controlRun(1,201,2,2.2f,.2f,false)); // useRest=true, resampleTo=0
    CHECK(controlRun(2,202,2,2.2f,.2f,false)); // useRest=true, resampleTo=2
    CHECK(controlRun(3,203,4,2.2f,.2f,true));  // useRest=false, resampleTo=4
    struct SourceControlGate { char const* name; void (*arm)(); void (*wait)(); void (*release)(); };
    SourceControlGate const sourceControlGates[] = {
        {"context",armCudaSourceAsyncContextCallbackGateForTesting, waitCudaSourceAsyncContextCallbackGateForTesting,
         releaseCudaSourceAsyncContextCallbackGateForTesting},
        {"program",armCudaSourceAsyncProgramCallbackGateForTesting, waitCudaSourceAsyncProgramCallbackGateForTesting,
         releaseCudaSourceAsyncProgramCallbackGateForTesting},
        {"scalar",armCudaSourceAsyncScalarCallbackGateForTesting, waitCudaSourceAsyncScalarCallbackGateForTesting,
         releaseCudaSourceAsyncScalarCallbackGateForTesting},
    };
    for (size_t gateIndex=0; gateIndex<sizeof(sourceControlGates)/sizeof(sourceControlGates[0]); ++gateIndex) {
        auto const& gate=sourceControlGates[gateIndex];
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
        auto held=CreateCudaExecutionJob(sourceControlPlan,*workspace,1,210,&diagnostics); CHECK(held);
        std::atomic<bool> heldDone{false}, heldOk{false}; std::atomic<int> heldCalls{0};
        gate.arm(); ReleaseSourceControlGate release{gate.release};
        auto const sourceExecuteStart=std::chrono::steady_clock::now();
        bool const sourceAccepted=ExecuteCudaJobSourceAsync(held,[&](bool ok) { heldOk.store(ok,std::memory_order_release); heldCalls.fetch_add(1,std::memory_order_relaxed); heldDone.store(true,std::memory_order_release); });
        auto const sourceExecuteElapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-sourceExecuteStart).count();
        CHECK(sourceAccepted);
        auto const sourceGateWaitStart=std::chrono::steady_clock::now();
        gate.wait();
        auto const sourceGateWaitElapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-sourceGateWaitStart).count();
        CHECK(!heldDone.load(std::memory_order_acquire) && cudaSourceRelayOccupiedCountForTesting()!=0);
        gate.release(); release.release=nullptr;
        bool const heldWait=Wait(heldDone);
        bool const heldDoneAfter=heldDone.load(std::memory_order_acquire);
        bool const heldOkAfter=heldOk.load(std::memory_order_acquire);
        int const heldCallsAfter=heldCalls.load(std::memory_order_acquire);
        size_t const sourceOccupiedAfter=cudaSourceRelayOccupiedCountForTesting();
        size_t const sourceQuarantinedAfter=cudaSourceRelayQuarantinedCountForTesting();
        bool const diagnosticsErrorsAfter=heldDoneAfter && diagnostics.HasErrors();
        if (!heldWait || !heldOkAfter || sourceOccupiedAfter!=0) {
            std::fprintf(stderr,
                "source-control gate[%zu]=%s accepted=%d executeMs=%lld gateWaitMs=%lld wait=%d done=%d ok=%d calls=%d sourceOccupied=%zu sourceQuarantined=%zu diagnosticsErrors=%d\n",
                gateIndex,gate.name,sourceAccepted,static_cast<long long>(sourceExecuteElapsed),
                static_cast<long long>(sourceGateWaitElapsed),heldWait,heldDoneAfter,heldOkAfter,heldCallsAfter,
                sourceOccupiedAfter,sourceQuarantinedAfter,diagnosticsErrorsAfter);
        }
        CHECK(heldWait && heldOkAfter && sourceOccupiedAfter==0);
    }
    auto controlAllocationWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(controlAllocationWorkspace);
    auto controlAllocationFailure=CreateCudaExecutionJob(sourceControlPlan,*controlAllocationWorkspace,1,212,&diagnostics);
    CHECK(controlAllocationFailure);
    size_t const controlAllocationQuarantine=cudaSourceRelayQuarantinedCountForTesting();
    std::atomic<int> controlAllocationCalls{0}; std::atomic<bool> controlAllocationDone{false}, controlAllocationOk{true};
    failNextCudaSourceRelayControlsAllocationForTesting();
    CHECK(ExecuteCudaJobSourceAsync(controlAllocationFailure,[&](bool ok) {
        controlAllocationOk.store(ok,std::memory_order_release); controlAllocationCalls.fetch_add(1,std::memory_order_relaxed);
        controlAllocationDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(controlAllocationDone) && !controlAllocationOk.load(std::memory_order_acquire) && controlAllocationCalls==1 &&
          cudaSourceRelayQuarantinedCountForTesting()==controlAllocationQuarantine);
    auto controlAllocationRetry=CreateCudaExecutionJob(sourceControlPlan,*controlAllocationWorkspace,1,213,&diagnostics);
    CHECK(controlAllocationRetry);
    std::atomic<bool> controlRetryDone{false}, controlRetryOk{false};
    CHECK(ExecuteCudaJobSourceAsync(controlAllocationRetry,[&](bool ok) {
        controlRetryOk.store(ok,std::memory_order_release); controlRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(controlRetryDone) && controlRetryOk.load(std::memory_order_acquire));
    auto emptyControlPlan=CompileCudaGraph(usdGenUsd::Lower(SourceControlDesc(true)),&diagnostics); CHECK(emptyControlPlan);
    auto emptyAsyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto emptyLegacyWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(emptyAsyncWorkspace && emptyLegacyWorkspace);
    auto emptyControl=CreateCudaExecutionJob(emptyControlPlan,*emptyAsyncWorkspace,1,204,&diagnostics); CHECK(emptyControl);
    std::atomic<bool> emptyDone{false}, emptyOk{false};
    CHECK(ExecuteCudaJobSourceAsync(emptyControl,[&](bool ok) { emptyOk.store(ok,std::memory_order_release); emptyDone.store(true,std::memory_order_release); }));
    CHECK(Wait(emptyDone) && emptyOk.load(std::memory_order_acquire));
    auto emptyAsync=FinalizeCudaJobForTesting(emptyControl);
    auto emptyLegacy=ExecuteCudaPlanForTesting(emptyControlPlan,*emptyLegacyWorkspace,1,204,&diagnostics);
    cudaStream_t emptyStream=nullptr; CHECK(cudaStreamCreateWithFlags(&emptyStream,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(emptyAsync && emptyLegacy && emptyAsync->Geometry().elementCount==0 && emptyAsync->Geometry().pointCount==0 &&
          emptyAsync->Geometry().tiles.empty() && Same(emptyAsync,emptyLegacy,emptyStream) &&
          SameFinalMetadata(emptyAsync,emptyLegacy));
    CHECK(cudaStreamDestroy(emptyStream)==cudaSuccess);

    // A dynamic useRest=false value is legal for source-only output but is
    // rejected once it would feed the rest-to-animated RBF stage.
    auto invalidRestPlan=CompileCudaGraph(usdGenUsd::Lower(SourceControlDesc(false,true)),&diagnostics); CHECK(invalidRestPlan);
    auto invalidRestWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(invalidRestWorkspace);
    auto invalidRestJob=CreateCudaExecutionJob(invalidRestPlan,*invalidRestWorkspace,3,205,&diagnostics); CHECK(invalidRestJob);
    CHECK(!RunCudaJobSourceForTesting(invalidRestJob));
    auto asyncSemanticFailure=[&](std::shared_ptr<const UsdGenCudaExecutionPlan> const& failingPlan,
                                  double frame, uint64_t generation) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); if (!workspace) return false;
        auto failing=CreateCudaExecutionJob(failingPlan,*workspace,frame,generation,&diagnostics); if (!failing) return false;
        size_t const before=cudaSourceRelayQuarantinedCountForTesting();
        std::atomic<int> calls{0}; std::atomic<bool> done{false}, ok{true};
        if (!ExecuteCudaJobSourceAsync(failing,[&](bool value) {
                ok.store(value,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
                done.store(true,std::memory_order_release);
            }) || !Wait(done) || ok.load(std::memory_order_acquire) || calls!=1 ||
            cudaSourceRelayQuarantinedCountForTesting()!=before) return false;
        auto retry=CreateCudaExecutionJob(sourceControlPlan,*workspace,1,generation+1,&diagnostics); if (!retry) return false;
        std::atomic<bool> retryDone{false}, retryOk{false};
        if (!ExecuteCudaJobSourceAsync(retry,[&](bool value) {
                retryOk.store(value,std::memory_order_release); retryDone.store(true,std::memory_order_release);
            })) return false;
        return Wait(retryDone) && retryOk.load(std::memory_order_acquire) &&
            cudaSourceRelayQuarantinedCountForTesting()==before;
    };
    CHECK(asyncSemanticFailure(invalidRestPlan,3,206));
    auto invalidNegative=SourceControlDesc();
    for (auto& expression : invalidNegative.expressions)
        if (expression.path==SdfPath("/SourceResample")) expression.source="-1";
    auto invalidNegativePlan=CompileCudaGraph(usdGenUsd::Lower(invalidNegative),&diagnostics); CHECK(invalidNegativePlan);
    CHECK(asyncSemanticFailure(invalidNegativePlan,1,207));
    auto invalidOne=invalidNegative;
    for (auto& expression : invalidOne.expressions)
        if (expression.path==SdfPath("/SourceResample")) expression.source="1";
    auto invalidOnePlan=CompileCudaGraph(usdGenUsd::Lower(invalidOne),&diagnostics); CHECK(invalidOnePlan);
    CHECK(asyncSemanticFailure(invalidOnePlan,1,208));
    auto invalidExpression=SourceControlDesc();
    for (auto& expression : invalidExpression.expressions)
        if (expression.path==SdfPath("/SourceResample")) expression.source="$frame / 0";
    auto invalidExpressionPlan=CompileCudaGraph(usdGenUsd::Lower(invalidExpression),&diagnostics); CHECK(invalidExpressionPlan);
    CHECK(asyncSemanticFailure(invalidExpressionPlan,1,209));

    auto resampleDesc=Desc();
    resampleDesc.nodes.front().params.push_back({TfToken("resampleTo"),VtValue(4),false});
    auto resamplePlan=CompileCudaGraph(usdGenUsd::Lower(resampleDesc),&diagnostics); CHECK(resamplePlan);
    auto resampleWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(resampleWorkspace);
    auto resampleJob=CreateCudaExecutionJob(resamplePlan,*resampleWorkspace,2,24,&diagnostics); CHECK(resampleJob);
    std::atomic<int> resampleCalls{0}; std::atomic<bool> resampleDone{false}, resampleOk{false};
    armCudaSourceAsyncResampleCallbackGateForTesting();
    ReleaseResampleGate releaseResampleGate;
    CHECK(ExecuteCudaJobSourceAsync(resampleJob,[&](bool ok) {
        resampleOk.store(ok,std::memory_order_release); resampleCalls.fetch_add(1,std::memory_order_relaxed);
        resampleDone.store(true,std::memory_order_release);
    }));
    waitCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(!resampleDone.load(std::memory_order_acquire));
    releaseCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(Wait(resampleDone) && resampleOk.load(std::memory_order_acquire) && resampleCalls == 1);
    for (size_t i=0;i<CudaExecutionJobOperatorCount(*resampleJob);++i)
        CHECK(RunCudaJobOperatorForTesting(resampleJob,i));
    auto asyncResample=FinalizeCudaJobForTesting(resampleJob); CHECK(asyncResample);
    auto syncResample=ExecuteCudaPlanForTesting(resamplePlan,*resampleWorkspace,2,25,&diagnostics); CHECK(syncResample);
    cudaStream_t resampleStream=nullptr; CHECK(cudaStreamCreateWithFlags(&resampleStream,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(asyncResample,syncResample,resampleStream));
    CHECK(cudaStreamDestroy(resampleStream)==cudaSuccess);

    // A held relay occupies the only test slot.  The second source must be
    // rejected before its upload/callback can begin; releasing the first
    // returns the slot for a later fresh source.
    setCudaSourceRelayCapacityForTesting(1); ResetRelayCapacity resetRelayCapacity;
    auto capFirst=CreateCudaExecutionJob(resamplePlan,*stagedWorkspace,2,30,&diagnostics); CHECK(capFirst);
    auto capSecond=CreateCudaExecutionJob(resamplePlan,*stagedWorkspace,2,31,&diagnostics); CHECK(capSecond);
    std::atomic<bool> capFirstDone{false}, capFirstOk{false};
    armCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(ExecuteCudaJobSourceAsync(capFirst,[&](bool ok) {
        capFirstOk.store(ok,std::memory_order_release); capFirstDone.store(true,std::memory_order_release);
    }));
    waitCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(cudaSourceRelayOccupiedCountForTesting() == 1 &&
          !ExecuteCudaJobSourceAsync(capSecond,[](bool) {}));
    releaseCudaSourceAsyncResampleCallbackGateForTesting();
    CHECK(Wait(capFirstDone) && capFirstOk.load(std::memory_order_acquire) &&
          cudaSourceRelayOccupiedCountForTesting() == 0);
    std::atomic<bool> capReuseDone{false}, capReuseOk{false};
    CHECK(ExecuteCudaJobSourceAsync(capSecond,[&](bool ok) {
        capReuseOk.store(ok,std::memory_order_release); capReuseDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(capReuseDone) && capReuseOk.load(std::memory_order_acquire) &&
          cudaSourceRelayOccupiedCountForTesting() == 0);

    // Complement the held-stream ordering above: here CUDA has reached the
    // real resample terminal callback, but the launcher has not returned to
    // release the slot.  Capacity remains charged until both phases arrive.
    auto terminalFirst=CreateCudaExecutionJob(resamplePlan,*stagedWorkspace,2,32,&diagnostics);
    auto terminalSecond=CreateCudaExecutionJob(resamplePlan,*stagedWorkspace,2,33,&diagnostics);
    CHECK(terminalFirst && terminalSecond);
    std::atomic<int> terminalFirstCalls{0}; std::atomic<bool> terminalFirstDone{false}, terminalFirstOk{false};
    armCudaSourceAsyncResampleLauncherReturnGateForTesting();
    ReleaseResampleLauncherGate releaseLauncherGate;
    CHECK(ExecuteCudaJobSourceAsync(terminalFirst,[&](bool ok) {
        terminalFirstOk.store(ok,std::memory_order_release);
        terminalFirstCalls.fetch_add(1,std::memory_order_relaxed);
        terminalFirstDone.store(true,std::memory_order_release);
    }));
    waitCudaSourceAsyncResampleLauncherReturnGateForTesting();
    CHECK(!terminalFirstDone.load(std::memory_order_acquire) &&
          cudaSourceRelayOccupiedCountForTesting() == 1 &&
          !ExecuteCudaJobSourceAsync(terminalSecond,[](bool) {}));
    releaseCudaSourceAsyncResampleLauncherReturnGateForTesting();
    CHECK(Wait(terminalFirstDone) && terminalFirstOk.load(std::memory_order_acquire) &&
          terminalFirstCalls == 1 && cudaSourceRelayOccupiedCountForTesting() == 0);
    std::atomic<bool> terminalReuseDone{false}, terminalReuseOk{false};
    CHECK(ExecuteCudaJobSourceAsync(terminalSecond,[&](bool ok) {
        terminalReuseOk.store(ok,std::memory_order_release);
        terminalReuseDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(terminalReuseDone) && terminalReuseOk.load(std::memory_order_acquire) &&
          cudaSourceRelayOccupiedCountForTesting() == 0);

    // The scalar terminal can arrive before its launcher returns.  That
    // nonblocking handoff still holds the one relay admission, rejects a
    // reentrant second source, and returns the slot only after launcher
    // return is released.
    setCudaSourceRelayCapacityForTesting(1);
    auto scalarFirst=CreateCudaExecutionJob(sourceControlPlan,*stagedWorkspace,1,34,&diagnostics);
    auto scalarSecond=CreateCudaExecutionJob(sourceControlPlan,*stagedWorkspace,1,35,&diagnostics);
    CHECK(scalarFirst && scalarSecond);
    std::atomic<int> scalarFirstCalls{0}; std::atomic<bool> scalarFirstDone{false}, scalarFirstOk{false};
    armCudaSourceAsyncScalarLauncherReturnGateForTesting();
    ReleaseSourceControlGate releaseScalarReturn{releaseCudaSourceAsyncScalarLauncherReturnGateForTesting};
    CHECK(ExecuteCudaJobSourceAsync(scalarFirst,[&](bool ok) {
        scalarFirstOk.store(ok,std::memory_order_release); scalarFirstCalls.fetch_add(1,std::memory_order_relaxed);
        scalarFirstDone.store(true,std::memory_order_release);
    }));
    waitCudaSourceAsyncScalarLauncherReturnGateForTesting();
    CHECK(!scalarFirstDone.load(std::memory_order_acquire) && cudaSourceRelayOccupiedCountForTesting()==1 &&
          !ExecuteCudaJobSourceAsync(scalarSecond,[](bool) {}));
    std::atomic<int> reentryCallbacks{0};
    CHECK(!RunCudaJobSourceForTesting(scalarFirst) &&
          !ExecuteCudaJobSourceAsync(scalarFirst,[&](bool) { ++reentryCallbacks; }) &&
          !RunCudaJobOperatorForTesting(scalarFirst,0) &&
          !ExecuteCudaJobOperatorAsync(scalarFirst,0,[&](bool) { ++reentryCallbacks; }) &&
          !FinalizeCudaJobForTesting(scalarFirst) &&
          !FinalizeCudaExecutionJobAsync(scalarFirst,[&](auto) { ++reentryCallbacks; }) &&
          reentryCallbacks.load(std::memory_order_acquire)==0);
    releaseCudaSourceAsyncScalarLauncherReturnGateForTesting(); releaseScalarReturn.release=nullptr;
    CHECK(Wait(scalarFirstDone) && scalarFirstOk.load(std::memory_order_acquire) && scalarFirstCalls==1 &&
          cudaSourceRelayOccupiedCountForTesting()==0);
    std::atomic<bool> scalarReuseDone{false}, scalarReuseOk{false};
    CHECK(ExecuteCudaJobSourceAsync(scalarSecond,[&](bool ok) {
        scalarReuseOk.store(ok,std::memory_order_release); scalarReuseDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(scalarReuseDone) && scalarReuseOk.load(std::memory_order_acquire) && cudaSourceRelayOccupiedCountForTesting()==0);
    setCudaSourceRelayCapacityForTesting(1024);

    // Installation failure is an accepted terminal failure: its relay slot
    // remains charged/quarantined and the workspace rejects future sources.
    auto poisonedWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(poisonedWorkspace);
    auto poisoned=CreateCudaExecutionJob(plan,*poisonedWorkspace,2,40,&diagnostics); CHECK(poisoned);
    auto queuedBeforePoison=CreateCudaExecutionJob(plan,*poisonedWorkspace,2,41,&diagnostics);
    CHECK(queuedBeforePoison);
    cudaStream_t poisonedReader=nullptr; CHECK(cudaStreamCreateWithFlags(&poisonedReader,cudaStreamNonBlocking)==cudaSuccess);
    auto poisonedPriorLease=gpu::AcquireGeometry(asyncGeneration,poisonedReader); float3 poisonedPriorPoint{};
    size_t const quarantinedBefore=cudaSourceRelayQuarantinedCountForTesting();
    std::atomic<int> poisonedCalls{0}; std::atomic<bool> poisonedDone{false}, poisonedOk{true};
    failNextCudaSourceRelayCallbackInstallForTesting();
    CHECK(ExecuteCudaJobSourceAsync(poisoned,[&](bool ok) {
        poisonedOk.store(ok,std::memory_order_release); poisonedCalls.fetch_add(1,std::memory_order_relaxed);
        poisonedDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(poisonedDone) && !poisonedOk.load(std::memory_order_acquire) && poisonedCalls == 1 &&
          poisonedWorkspace->IsPoisoned() &&
          cudaSourceRelayQuarantinedCountForTesting() == quarantinedBefore + 1);
    CHECK(poisonedPriorLease && cudaMemcpyAsync(&poisonedPriorPoint,poisonedPriorLease.Geometry().points.data,
          sizeof(poisonedPriorPoint),cudaMemcpyDeviceToHost,poisonedReader)==cudaSuccess &&
          cudaStreamSynchronize(poisonedReader)==cudaSuccess && std::isfinite(poisonedPriorPoint.x));
    poisonedPriorLease={}; CHECK(cudaStreamDestroy(poisonedReader)==cudaSuccess);
    CHECK(!CreateCudaExecutionJob(plan,*poisonedWorkspace,2,42,&diagnostics));
    CHECK(!ExecuteCudaJobSourceAsync(queuedBeforePoison,[](bool) {}));
    queuedBeforePoison.reset();

    // A real native callback status error follows the same terminal relay
    // path, rather than silently committing a source after CUDA failure.
    setCudaSourceRelayCapacityForTesting(1024);
    auto callbackWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(callbackWorkspace);
    auto callbackFailure=CreateCudaExecutionJob(plan,*callbackWorkspace,2,42,&diagnostics); CHECK(callbackFailure);
    cudaStream_t nativeReader=nullptr; CHECK(cudaStreamCreateWithFlags(&nativeReader,cudaStreamNonBlocking)==cudaSuccess);
    auto nativePriorLease=gpu::AcquireGeometry(asyncGeneration,nativeReader); float3 nativePriorPoint{};
    size_t const nativeQuarantinedBefore=cudaSourceRelayQuarantinedCountForTesting();
    std::atomic<int> nativeCalls{0}; std::atomic<bool> nativeDone{false}, nativeOk{true};
    failNextCudaSourceRelayNativeCallbackForTesting();
    CHECK(ExecuteCudaJobSourceAsync(callbackFailure,[&](bool ok) {
        nativeOk.store(ok,std::memory_order_release); nativeCalls.fetch_add(1,std::memory_order_relaxed);
        nativeDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(nativeDone) && !nativeOk.load(std::memory_order_acquire) && nativeCalls == 1 &&
          cudaSourceRelayQuarantinedCountForTesting() == nativeQuarantinedBefore + 1);
    CHECK(nativePriorLease && cudaMemcpyAsync(&nativePriorPoint,nativePriorLease.Geometry().points.data,
          sizeof(nativePriorPoint),cudaMemcpyDeviceToHost,nativeReader)==cudaSuccess &&
          cudaStreamSynchronize(nativeReader)==cudaSuccess && std::isfinite(nativePriorPoint.x));
    nativePriorLease={}; CHECK(cudaStreamDestroy(nativeReader)==cudaSuccess);

    // These control-phase failures occur after CUDA work is admitted.  Their
    // permanent relay quarantine must not invalidate an already-published
    // generation; retain and explicitly release a named lease before its
    // reader stream is destroyed.
    auto sourceControlFailure=[&](void (*inject)()) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!workspace) return false;
        auto failing=CreateCudaExecutionJob(sourceControlPlan,*workspace,1,211,&diagnostics);
        if (!failing) return false;
        cudaStream_t reader=nullptr;
        if (cudaStreamCreateWithFlags(&reader,cudaStreamNonBlocking)!=cudaSuccess) return false;
        auto priorLease=gpu::AcquireGeometry(asyncGeneration,reader); float3 priorPoint{};
        size_t const before=cudaSourceRelayQuarantinedCountForTesting();
        std::atomic<int> calls{0}; std::atomic<bool> done{false}, ok{true}; inject();
        bool const accepted=ExecuteCudaJobSourceAsync(failing,[&](bool value) {
            ok.store(value,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
            done.store(true,std::memory_order_release);
        });
        bool const retained=accepted && Wait(done) && priorLease &&
            cudaMemcpyAsync(&priorPoint,priorLease.Geometry().points.data,sizeof(priorPoint),cudaMemcpyDeviceToHost,reader)==cudaSuccess &&
            cudaStreamSynchronize(reader)==cudaSuccess && std::isfinite(priorPoint.x);
        priorLease={}; cudaStreamDestroy(reader);
        return retained && !ok.load(std::memory_order_acquire) && calls==1 &&
            cudaSourceRelayQuarantinedCountForTesting()==before+1;
    };
    CHECK(sourceControlFailure(failNextCudaSourceRelayContextCallbackInstallForTesting));
    CHECK(sourceControlFailure(failNextCudaSourceRelayProgramCallbackInstallForTesting));
    CHECK(sourceControlFailure(failNextCudaSourceRelayScalarCallbackInstallForTesting));
    CHECK(sourceControlFailure(failNextCudaSourceRelayContextNativeCallbackForTesting));
    CHECK(sourceControlFailure(failNextCudaSourceRelayProgramNativeCallbackForTesting));
    CHECK(sourceControlFailure(failNextCudaSourceRelayScalarNativeCallbackForTesting));

    // Width's asynchronous operator path first commits all expression-context
    // domains, then all expression programs, then Width itself.  The direct
    // staged result must be bit-for-bit equivalent to the synchronous graph.
    auto widthPlan=CompileCudaGraph(usdGenUsd::Lower(WidthExpressionDesc()),&diagnostics); CHECK(widthPlan);
    auto widthMetadata = GetCudaExecutionPlanMetadata(*widthPlan);
    CHECK(widthMetadata && !widthMetadata->MemoryEstimate().memoryAvailable &&
          !widthMetadata->MemoryEstimate().conservativeUpperBound);
    auto widthAsyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto widthSyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(widthAsyncWorkspace && widthSyncWorkspace);
    auto prepareWidth = [&](std::unique_ptr<UsdGenCudaExecutionWorkspace>& workspace,
                            uint64_t generation,
                            std::shared_ptr<const UsdGenDeviceGeneration> previous = {}) {
        auto prepared=CreateCudaExecutionJob(widthPlan,*workspace,2,generation,&diagnostics,previous);
        if (!prepared || !RunCudaJobSourceForTesting(prepared) || !RunCudaJobOperatorForTesting(prepared,0)) return std::shared_ptr<UsdGenCudaExecutionJob>{};
        return prepared;
    };
    auto widthAsync=prepareWidth(widthAsyncWorkspace,60); CHECK(widthAsync);
    std::atomic<int> widthCalls{0}; std::atomic<bool> widthDone{false}, widthOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(widthAsync,1,[&](bool ok) {
        widthOk.store(ok,std::memory_order_release); widthCalls.fetch_add(1,std::memory_order_relaxed);
        widthDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(widthDone) && widthOk.load(std::memory_order_acquire) && widthCalls == 1);
    CHECK(RunCudaJobOperatorForTesting(widthAsync,2));
    auto widthAsyncGeneration=FinalizeCudaJobForTesting(widthAsync); CHECK(widthAsyncGeneration);
    auto widthSyncGeneration=ExecuteCudaPlanForTesting(widthPlan,*widthSyncWorkspace,2,61,&diagnostics); CHECK(widthSyncGeneration);
    cudaStream_t widthReader=nullptr; CHECK(cudaStreamCreateWithFlags(&widthReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(widthAsyncGeneration,widthSyncGeneration,widthReader));
    CHECK(cudaStreamDestroy(widthReader)==cudaSuccess);

    // One bounded admission stays occupied through each proof-gated phase;
    // a second otherwise-valid Width job is rejected before its callback/GPU
    // acceptance and can be retried after the first terminal completion.
    setCudaOperatorRelayCapacityForTesting(1); ResetOperatorRelayCapacity resetOperatorCapacity;
    auto contextFirst=prepareWidth(widthAsyncWorkspace,62); auto contextSecond=prepareWidth(widthAsyncWorkspace,63);
    CHECK(contextFirst && contextSecond);
    std::atomic<int> contextCalls{0}, contextRejectedCalls{0}; std::atomic<bool> contextDone{false}, contextOk{false};
    armCudaOperatorAsyncContextCallbackGateForTesting(); ReleaseOperatorContextGate releaseContextGate;
    CHECK(ExecuteCudaJobOperatorAsync(contextFirst,1,[&](bool ok) {
        contextOk.store(ok,std::memory_order_release); contextCalls.fetch_add(1,std::memory_order_relaxed);
        contextDone.store(true,std::memory_order_release);
    }));
    waitCudaOperatorAsyncContextCallbackGateForTesting();
    CHECK(cudaOperatorRelayOccupiedCountForTesting()==1 && !contextDone.load(std::memory_order_acquire) &&
          !ExecuteCudaJobOperatorAsync(contextSecond,1,[&](bool) { contextRejectedCalls.fetch_add(1); }) &&
          contextRejectedCalls==0);
    releaseCudaOperatorAsyncContextCallbackGateForTesting();
    CHECK(Wait(contextDone) && contextOk.load(std::memory_order_acquire) && contextCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0);

    auto programFirst=prepareWidth(widthAsyncWorkspace,64); auto programSecond=prepareWidth(widthAsyncWorkspace,65);
    CHECK(programFirst && programSecond);
    std::atomic<int> programCalls{0}; std::atomic<bool> programDone{false}, programOk{false};
    armCudaOperatorAsyncProgramCallbackGateForTesting(); ReleaseOperatorProgramGate releaseProgramGate;
    CHECK(ExecuteCudaJobOperatorAsync(programFirst,1,[&](bool ok) {
        programOk.store(ok,std::memory_order_release); programCalls.fetch_add(1,std::memory_order_relaxed);
        programDone.store(true,std::memory_order_release);
    }));
    waitCudaOperatorAsyncProgramCallbackGateForTesting();
    CHECK(cudaOperatorRelayOccupiedCountForTesting()==1 &&
          !ExecuteCudaJobOperatorAsync(programSecond,1,[](bool) {}));
    releaseCudaOperatorAsyncProgramCallbackGateForTesting();
    CHECK(Wait(programDone) && programOk.load(std::memory_order_acquire) && programCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0);

    auto widthFirst=prepareWidth(widthAsyncWorkspace,66); auto widthSecond=prepareWidth(widthAsyncWorkspace,67);
    CHECK(widthFirst && widthSecond);
    std::atomic<int> widthPhaseCalls{0}; std::atomic<bool> widthPhaseDone{false}, widthPhaseOk{false};
    armCudaOperatorAsyncWidthCallbackGateForTesting(); ReleaseOperatorWidthGate releaseWidthGate;
    CHECK(ExecuteCudaJobOperatorAsync(widthFirst,1,[&](bool ok) {
        widthPhaseOk.store(ok,std::memory_order_release); widthPhaseCalls.fetch_add(1,std::memory_order_relaxed);
        widthPhaseDone.store(true,std::memory_order_release);
    }));
    waitCudaOperatorAsyncWidthCallbackGateForTesting();
    CHECK(cudaOperatorRelayOccupiedCountForTesting()==1 &&
          !ExecuteCudaJobOperatorAsync(widthSecond,1,[](bool) {}));
    releaseCudaOperatorAsyncWidthCallbackGateForTesting();
    CHECK(Wait(widthPhaseDone) && widthPhaseOk.load(std::memory_order_acquire) && widthPhaseCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0);

    // CUDA may publish Width's native terminal before the launcher returns.
    // The slot and user completion remain withheld until that second proof.
    auto terminalWidth=prepareWidth(widthAsyncWorkspace,68); auto terminalWidthSecond=prepareWidth(widthAsyncWorkspace,69);
    CHECK(terminalWidth && terminalWidthSecond);
    std::atomic<int> terminalWidthCalls{0}; std::atomic<bool> terminalWidthDone{false}, terminalWidthOk{false};
    armCudaOperatorAsyncWidthLauncherReturnGateForTesting(); ReleaseOperatorWidthLauncherGate releaseWidthLauncherGate;
    CHECK(ExecuteCudaJobOperatorAsync(terminalWidth,1,[&](bool ok) {
        terminalWidthOk.store(ok,std::memory_order_release); terminalWidthCalls.fetch_add(1,std::memory_order_relaxed);
        terminalWidthDone.store(true,std::memory_order_release);
    }));
    waitCudaOperatorAsyncWidthLauncherReturnGateForTesting();
    CHECK(!terminalWidthDone.load(std::memory_order_acquire) && cudaOperatorRelayOccupiedCountForTesting()==1 &&
          !ExecuteCudaJobOperatorAsync(terminalWidthSecond,1,[](bool) {}));
    releaseCudaOperatorAsyncWidthLauncherReturnGateForTesting();
    CHECK(Wait(terminalWidthDone) && terminalWidthOk.load(std::memory_order_acquire) && terminalWidthCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0);

    // The following injected unsafe terminal paths permanently spend their
    // own slots, so restore production-sized admission before exercising all
    // three phase-specific install/native failure contracts.
    setCudaOperatorRelayCapacityForTesting(1024);

    // A terminally proven semantic expression error releases its slot and does
    // not poison the workspace; a subsequent valid Width is admitted.
    auto invalidWidthPlan=CompileCudaGraph(usdGenUsd::Lower(WidthExpressionDesc(true)),&diagnostics); CHECK(invalidWidthPlan);
    auto semanticWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(semanticWorkspace);
    auto semantic=CreateCudaExecutionJob(invalidWidthPlan,*semanticWorkspace,2,70,&diagnostics,widthAsyncGeneration);
    CHECK(semantic && RunCudaJobSourceForTesting(semantic) && RunCudaJobOperatorForTesting(semantic,0));
    size_t const semanticQuarantined=cudaOperatorRelayQuarantinedCountForTesting();
    std::atomic<int> semanticCalls{0}; std::atomic<bool> semanticDone{false}, semanticOk{true};
    CHECK(ExecuteCudaJobOperatorAsync(semantic,1,[&](bool ok) {
        semanticOk.store(ok,std::memory_order_release); semanticCalls.fetch_add(1,std::memory_order_relaxed);
        semanticDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(semanticDone) && !semanticOk.load(std::memory_order_acquire) && semanticCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0 &&
          cudaOperatorRelayQuarantinedCountForTesting()==semanticQuarantined);
    cudaStream_t lastGoodReader=nullptr; CHECK(cudaStreamCreateWithFlags(&lastGoodReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(gpu::AcquireGeometry(widthAsyncGeneration,lastGoodReader));
    CHECK(cudaStreamDestroy(lastGoodReader)==cudaSuccess);
    auto semanticRetry=prepareWidth(semanticWorkspace,71,widthAsyncGeneration); CHECK(semanticRetry);
    std::atomic<bool> semanticRetryDone{false}, semanticRetryOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(semanticRetry,1,[&](bool ok) {
        semanticRetryOk.store(ok,std::memory_order_release); semanticRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(semanticRetryDone) && semanticRetryOk.load(std::memory_order_acquire));

    // A malformed literal reaches Width only after the profile transfers have
    // been queued.  It is nevertheless a terminally proven semantic
    // rejection, not an unproven-upload quarantine: the old generation stays
    // readable and a valid new job can use the same workspace.
    auto literalInvalidDesc=Desc();
    literalInvalidDesc.nodes[2].params.clear();
    literalInvalidDesc.nodes[2].params.push_back({TfToken("width"), VtValue(-1.f), false});
    auto literalInvalidPlan=CompileCudaGraph(usdGenUsd::Lower(literalInvalidDesc),&diagnostics); CHECK(literalInvalidPlan);
    auto literalWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(literalWorkspace);
    auto literalInvalid=CreateCudaExecutionJob(literalInvalidPlan,*literalWorkspace,2,72,&diagnostics,widthAsyncGeneration);
    CHECK(literalInvalid && RunCudaJobSourceForTesting(literalInvalid) && RunCudaJobOperatorForTesting(literalInvalid,0));
    size_t const literalQuarantined=cudaOperatorRelayQuarantinedCountForTesting();
    std::atomic<int> literalCalls{0}; std::atomic<bool> literalDone{false}, literalOk{true};
    CHECK(ExecuteCudaJobOperatorAsync(literalInvalid,1,[&](bool ok) {
        literalOk.store(ok,std::memory_order_release); literalCalls.fetch_add(1,std::memory_order_relaxed);
        literalDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(literalDone) && !literalOk.load(std::memory_order_acquire) && literalCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0 &&
          cudaOperatorRelayQuarantinedCountForTesting()==literalQuarantined);
    cudaStream_t literalLastGoodReader=nullptr; CHECK(cudaStreamCreateWithFlags(&literalLastGoodReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(gpu::AcquireGeometry(widthAsyncGeneration,literalLastGoodReader));
    CHECK(cudaStreamDestroy(literalLastGoodReader)==cudaSuccess);
    auto literalRetry=CreateCudaExecutionJob(widthPlan,*literalWorkspace,2,73,&diagnostics,widthAsyncGeneration);
    CHECK(literalRetry && RunCudaJobSourceForTesting(literalRetry) && RunCudaJobOperatorForTesting(literalRetry,0));
    std::atomic<int> literalRetryCalls{0}; std::atomic<bool> literalRetryDone{false}, literalRetryOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(literalRetry,1,[&](bool ok) {
        literalRetryOk.store(ok,std::memory_order_release); literalRetryCalls.fetch_add(1,std::memory_order_relaxed);
        literalRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(literalRetryDone) && literalRetryOk.load(std::memory_order_acquire) && literalRetryCalls==1);

    // Allocation pressure before the first program literal H2D has no
    // unproven CUDA work.  The proved context candidate must be discarded so
    // a later job can begin fresh programs on this same workspace.
    auto allocationWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(allocationWorkspace);
    auto allocationFailure=prepareWidth(allocationWorkspace,74,widthAsyncGeneration); CHECK(allocationFailure);
    size_t const allocationQuarantined=cudaOperatorRelayQuarantinedCountForTesting();
    std::atomic<int> allocationCalls{0}; std::atomic<bool> allocationDone{false}, allocationOk{true};
    failNextCudaParameterFreshProgramAllocationForTesting();
    CHECK(ExecuteCudaJobOperatorAsync(allocationFailure,1,[&](bool ok) {
        allocationOk.store(ok,std::memory_order_release); allocationCalls.fetch_add(1,std::memory_order_relaxed);
        allocationDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(allocationDone) && !allocationOk.load(std::memory_order_acquire) && allocationCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0 &&
          cudaOperatorRelayQuarantinedCountForTesting()==allocationQuarantined);
    cudaStream_t allocationLastGoodReader=nullptr; CHECK(cudaStreamCreateWithFlags(&allocationLastGoodReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(gpu::AcquireGeometry(widthAsyncGeneration,allocationLastGoodReader));
    CHECK(cudaStreamDestroy(allocationLastGoodReader)==cudaSuccess);
    auto allocationRetry=prepareWidth(allocationWorkspace,75,widthAsyncGeneration); CHECK(allocationRetry);
    std::atomic<int> allocationRetryCalls{0}; std::atomic<bool> allocationRetryDone{false}, allocationRetryOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(allocationRetry,1,[&](bool ok) {
        allocationRetryOk.store(ok,std::memory_order_release); allocationRetryCalls.fetch_add(1,std::memory_order_relaxed);
        allocationRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(allocationRetryDone) && allocationRetryOk.load(std::memory_order_acquire) && allocationRetryCalls==1);

    // Length carries mixed groom/primitive/point controls through context and
    // program proof, then conditionally publishes points+keep before the
    // scalar-count and exact-scatter compaction phases.  Its final generation
    // matches the synchronous graph without a host geometry readback.
    auto lengthPlan=CompileCudaGraph(usdGenUsd::Lower(LengthExpressionDesc()),&diagnostics); CHECK(lengthPlan);
    auto lengthMetadata = GetCudaExecutionPlanMetadata(*lengthPlan);
    CHECK(lengthMetadata && !lengthMetadata->MemoryEstimate().memoryAvailable &&
          !lengthMetadata->MemoryEstimate().conservativeUpperBound);
    auto lengthAsyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto lengthSyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(lengthAsyncWorkspace && lengthSyncWorkspace);
    auto prepareLength=[&](std::shared_ptr<const UsdGenCudaExecutionPlan> const& plan,
                           std::unique_ptr<UsdGenCudaExecutionWorkspace>& workspace,
                           uint64_t generation,
                           std::shared_ptr<const UsdGenDeviceGeneration> previous={}) {
        auto prepared=CreateCudaExecutionJob(plan,*workspace,2,generation,&diagnostics,previous);
        if (!prepared || !RunCudaJobSourceForTesting(prepared)) return std::shared_ptr<UsdGenCudaExecutionJob>{};
        return prepared;
    };
    auto lengthAsync=prepareLength(lengthPlan,lengthAsyncWorkspace,80,widthAsyncGeneration); CHECK(lengthAsync);
    std::atomic<int> lengthCalls{0}; std::atomic<bool> lengthDone{false}, lengthOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(lengthAsync,0,[&](bool ok) {
        lengthOk.store(ok,std::memory_order_release); lengthCalls.fetch_add(1,std::memory_order_relaxed);
        lengthDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(lengthDone) && lengthOk.load(std::memory_order_acquire) && lengthCalls==1);
    for (size_t i=1;i<CudaExecutionJobOperatorCount(*lengthAsync);++i)
        CHECK(RunCudaJobOperatorForTesting(lengthAsync,i));
    auto lengthAsyncGeneration=FinalizeCudaJobForTesting(lengthAsync); CHECK(lengthAsyncGeneration);
    // This is deliberately a non-cull fixture.  Keep the cardinality
    // assertion ahead of all value/topology comparisons so an all-culled
    // generation cannot make their changed-value oracle vacuous.
    CHECK(lengthAsyncGeneration->Geometry().elementCount==2 &&
          lengthAsyncGeneration->Geometry().pointCount==4);
    auto lengthSyncGeneration=ExecuteCudaPlanForTesting(lengthPlan,*lengthSyncWorkspace,2,81,&diagnostics,widthAsyncGeneration);
    CHECK(lengthSyncGeneration);
    cudaStream_t lengthReader=nullptr; CHECK(cudaStreamCreateWithFlags(&lengthReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(Same(lengthAsyncGeneration,lengthSyncGeneration,lengthReader));
    CHECK(cudaStreamDestroy(lengthReader)==cudaSuccess);

    // Culling is topology-changing: one short curve is removed while every
    // surviving channel keeps its matching curve cardinality and root binding.
    auto cullPlan=CompileCudaGraph(usdGenUsd::Lower(LengthExpressionDesc(true)),&diagnostics); CHECK(cullPlan);
    auto cullWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(cullWorkspace);
    auto cullSyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(cullSyncWorkspace);
    auto cullJob=prepareLength(cullPlan,cullWorkspace,82,widthAsyncGeneration); CHECK(cullJob);
    std::atomic<bool> cullDone{false}, cullOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(cullJob,0,[&](bool ok) {
        cullOk.store(ok,std::memory_order_release); cullDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(cullDone) && cullOk.load(std::memory_order_acquire));
    for (size_t i=1;i<CudaExecutionJobOperatorCount(*cullJob);++i) CHECK(RunCudaJobOperatorForTesting(cullJob,i));
    auto cullGeneration=FinalizeCudaJobForTesting(cullJob); CHECK(cullGeneration);
    auto cullSyncGeneration=ExecuteCudaPlanForTesting(cullPlan,*cullSyncWorkspace,2,821,&diagnostics,widthAsyncGeneration);
    CHECK(cullSyncGeneration);
    cudaStream_t cullReader=nullptr; CHECK(cudaStreamCreateWithFlags(&cullReader,cudaStreamNonBlocking)==cudaSuccess);
    {
        auto cullLease=gpu::AcquireGeometry(cullGeneration,cullReader);
        CHECK(cullLease && cullLease.Geometry().curveCount==1 && cullLease.Geometry().pointCount==2 &&
              cullLease.Geometry().stableIds.size==1 && cullLease.HairT().size==2 &&
              cullLease.RootPrim().size==1 && cullLease.RootUV().size==1);
    }
    CHECK(Same(cullGeneration,cullSyncGeneration,cullReader) &&
          SameRetainedCurveChannels(cullGeneration,cullSyncGeneration,cullReader));
    CHECK(cudaStreamDestroy(cullReader)==cudaSuccess);

    auto allCullDesc=LengthExpressionDesc(true);
    for (auto& expression : allCullDesc.expressions)
        if (expression.path==SdfPath("/LengthPrimitive")) expression.source="$value + 100";
    auto allCullPlan=CompileCudaGraph(usdGenUsd::Lower(allCullDesc),&diagnostics); CHECK(allCullPlan);
    auto allCullWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(allCullWorkspace);
    auto allCullJob=prepareLength(allCullPlan,allCullWorkspace,83,widthAsyncGeneration); CHECK(allCullJob);
    std::atomic<bool> allCullDone{false}, allCullOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(allCullJob,0,[&](bool ok) {
        allCullOk.store(ok,std::memory_order_release); allCullDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(allCullDone) && allCullOk.load(std::memory_order_acquire));
    for (size_t i=1;i<CudaExecutionJobOperatorCount(*allCullJob);++i) CHECK(RunCudaJobOperatorForTesting(allCullJob,i));
    auto allCullGeneration=FinalizeCudaJobForTesting(allCullJob); CHECK(allCullGeneration);
    cudaStream_t allCullReader=nullptr; CHECK(cudaStreamCreateWithFlags(&allCullReader,cudaStreamNonBlocking)==cudaSuccess);
    {
        auto allCullLease=gpu::AcquireGeometry(allCullGeneration,allCullReader);
        CHECK(allCullLease && allCullLease.Geometry().curveCount==0 && allCullLease.Geometry().pointCount==0 &&
              allCullLease.Geometry().stableIds.size==0 && allCullLease.HairT().size==0 &&
              allCullLease.RootPrim().size==0 && allCullLease.RootUV().size==0);
    }
    CHECK(cudaStreamDestroy(allCullReader)==cudaSuccess);

    // The single operator admission spans every Length/count/scatter proof.
    // No subsequent operator is admitted until the held first operator has
    // committed its topology revision.
    setCudaOperatorRelayCapacityForTesting(1); ResetOperatorRelayCapacity resetLengthCapacity;
    auto heldLength=[&](auto arm, auto wait, auto release, uint64_t generation) {
        auto first=prepareLength(lengthPlan,lengthAsyncWorkspace,generation,widthAsyncGeneration);
        auto second=prepareLength(lengthPlan,lengthAsyncWorkspace,generation+1,widthAsyncGeneration);
        if (!first || !second) return false;
        std::atomic<int> calls{0}; std::atomic<bool> done{false}, ok{false};
        arm();
        ReleaseOperatorGate releaseGate{release};
        if (!ExecuteCudaJobOperatorAsync(first,0,[&](bool result) {
                ok.store(result,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
                done.store(true,std::memory_order_release);
            })) return false;
        wait();
        bool const blocked=cudaOperatorRelayOccupiedCountForTesting()==1 &&
            !done.load(std::memory_order_acquire) &&
            !ExecuteCudaJobOperatorAsync(first,0,[](bool) {}) &&
            !ExecuteCudaJobOperatorAsync(second,0,[](bool) {});
        release(); releaseGate.release=nullptr;
        if (!blocked || !Wait(done) || !ok.load(std::memory_order_acquire) || calls!=1 ||
            cudaOperatorRelayOccupiedCountForTesting()!=0) return false;
        return RunCudaJobOperatorForTesting(first,1);
    };
    CHECK(heldLength(armCudaOperatorAsyncLengthCallbackGateForTesting,
                     waitCudaOperatorAsyncLengthCallbackGateForTesting,
                     releaseCudaOperatorAsyncLengthCallbackGateForTesting,84));
    CHECK(heldLength(armCudaOperatorAsyncCountsCallbackGateForTesting,
                     waitCudaOperatorAsyncCountsCallbackGateForTesting,
                     releaseCudaOperatorAsyncCountsCallbackGateForTesting,86));
    CHECK(heldLength(armCudaOperatorAsyncScatterCallbackGateForTesting,
                     waitCudaOperatorAsyncScatterCallbackGateForTesting,
                     releaseCudaOperatorAsyncScatterCallbackGateForTesting,88));
    CHECK(heldLength(armCudaOperatorAsyncNamedTopologyCallbackGateForTesting,
                     waitCudaOperatorAsyncNamedTopologyCallbackGateForTesting,
                     releaseCudaOperatorAsyncNamedTopologyCallbackGateForTesting,881));
    setCudaOperatorRelayCapacityForTesting(1024);

    // A device-semantic Length failure after admission does not poison the
    // workspace or retain a relay slot; a valid new job is still accepted.
    auto badLengthDesc=LengthExpressionDesc();
    for (auto& expression : badLengthDesc.expressions)
        if (expression.path==SdfPath("/LengthPoint")) expression.source="$value - 2";
    auto badLengthPlan=CompileCudaGraph(usdGenUsd::Lower(badLengthDesc),&diagnostics); CHECK(badLengthPlan);
    auto badLengthWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(badLengthWorkspace);
    auto badLength=prepareLength(badLengthPlan,badLengthWorkspace,89,widthAsyncGeneration); CHECK(badLength);
    size_t const lengthSemanticQuarantined=cudaOperatorRelayQuarantinedCountForTesting();
    std::atomic<int> badLengthCalls{0}; std::atomic<bool> badLengthDone{false}, badLengthOk{true};
    CHECK(ExecuteCudaJobOperatorAsync(badLength,0,[&](bool ok) {
        badLengthOk.store(ok,std::memory_order_release); badLengthCalls.fetch_add(1,std::memory_order_relaxed);
        badLengthDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(badLengthDone) && !badLengthOk.load(std::memory_order_acquire) && badLengthCalls==1 &&
          cudaOperatorRelayOccupiedCountForTesting()==0 &&
          cudaOperatorRelayQuarantinedCountForTesting()==lengthSemanticQuarantined);
    auto badLengthRetry=prepareLength(lengthPlan,badLengthWorkspace,90,widthAsyncGeneration); CHECK(badLengthRetry);
    std::atomic<bool> badLengthRetryDone{false}, badLengthRetryOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(badLengthRetry,0,[&](bool ok) {
        badLengthRetryOk.store(ok,std::memory_order_release); badLengthRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(badLengthRetryDone) && badLengthRetryOk.load(std::memory_order_acquire));

    // Exact survivor storage allocation occurs only after the proved count
    // phase.  A host-only pre-scatter allocation rejection is not poison and
    // leaves the same workspace able to accept a complete new Length job.
    auto scatterAllocationWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(scatterAllocationWorkspace);
    auto scatterAllocationFailure=prepareLength(lengthPlan,scatterAllocationWorkspace,91,widthAsyncGeneration);
    CHECK(scatterAllocationFailure);
    size_t const scatterAllocationQuarantined=cudaOperatorRelayQuarantinedCountForTesting();
    std::atomic<int> scatterAllocationCalls{0}; std::atomic<bool> scatterAllocationDone{false}, scatterAllocationOk{true};
    failNextCudaOperatorRelayScatterAllocationForTesting();
    CHECK(ExecuteCudaJobOperatorAsync(scatterAllocationFailure,0,[&](bool ok) {
        scatterAllocationOk.store(ok,std::memory_order_release); scatterAllocationCalls.fetch_add(1,std::memory_order_relaxed);
        scatterAllocationDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(scatterAllocationDone) && !scatterAllocationOk.load(std::memory_order_acquire) &&
          scatterAllocationCalls==1 && cudaOperatorRelayOccupiedCountForTesting()==0 &&
          cudaOperatorRelayQuarantinedCountForTesting()==scatterAllocationQuarantined);
    cudaStream_t scatterLastGoodReader=nullptr; CHECK(cudaStreamCreateWithFlags(&scatterLastGoodReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(gpu::AcquireGeometry(widthAsyncGeneration,scatterLastGoodReader));
    CHECK(cudaStreamDestroy(scatterLastGoodReader)==cudaSuccess);
    auto scatterAllocationRetry=prepareLength(lengthPlan,scatterAllocationWorkspace,92,widthAsyncGeneration);
    CHECK(scatterAllocationRetry);
    std::atomic<bool> scatterAllocationRetryDone{false}, scatterAllocationRetryOk{false};
    CHECK(ExecuteCudaJobOperatorAsync(scatterAllocationRetry,0,[&](bool ok) {
        scatterAllocationRetryOk.store(ok,std::memory_order_release);
        scatterAllocationRetryDone.store(true,std::memory_order_release);
    }));
    CHECK(Wait(scatterAllocationRetryDone) && scatterAllocationRetryOk.load(std::memory_order_acquire));

    auto unsafeLengthFailure=[&](void (*inject)()) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!workspace) return false;
        auto failing=prepareLength(lengthPlan,workspace,93,widthAsyncGeneration);
        if (!failing) return false;
        size_t const before=cudaOperatorRelayQuarantinedCountForTesting();
        std::atomic<int> calls{0}; std::atomic<bool> done{false}, ok{true};
        inject();
        if (!ExecuteCudaJobOperatorAsync(failing,0,[&](bool result) {
                ok.store(result,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
                done.store(true,std::memory_order_release);
            }) || !Wait(done) || ok.load(std::memory_order_acquire) || calls!=1 ||
            cudaOperatorRelayQuarantinedCountForTesting()!=before+1)
            return false;
        auto retry=CreateCudaExecutionJob(lengthPlan,*workspace,2,94,&diagnostics,widthAsyncGeneration);
        return workspace->IsPoisoned() && !retry;
    };
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayLengthCallbackInstallForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayCountsCallbackInstallForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayScatterCallbackInstallForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayNamedTopologyCallbackInstallForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayLengthNativeCallbackForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayCountsNativeCallbackForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayScatterNativeCallbackForTesting));
    CHECK(unsafeLengthFailure(failNextCudaOperatorRelayNamedTopologyNativeCallbackForTesting));

    auto unsafeOperatorFailure = [&](void (*inject)()) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!workspace) return false;
        auto failing=prepareWidth(workspace,72,widthAsyncGeneration);
        if (!failing) return false;
        size_t const before=cudaOperatorRelayQuarantinedCountForTesting();
        std::atomic<int> calls{0}; std::atomic<bool> done{false}, ok{true};
        inject();
        if (!ExecuteCudaJobOperatorAsync(failing,1,[&](bool result) {
                ok.store(result,std::memory_order_release); calls.fetch_add(1,std::memory_order_relaxed);
                done.store(true,std::memory_order_release);
            }) || !Wait(done) || ok.load(std::memory_order_acquire) || calls!=1 ||
            cudaOperatorRelayQuarantinedCountForTesting()!=before+1)
            return false;
        // The callback's unsafe path poisons this workspace, so a later
        // source admission cannot silently reuse its retained CUDA state.
        auto retry=CreateCudaExecutionJob(widthPlan,*workspace,2,73,&diagnostics,widthAsyncGeneration);
        return workspace->IsPoisoned() && !retry;
    };
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayContextCallbackInstallForTesting));
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayProgramCallbackInstallForTesting));
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayWidthCallbackInstallForTesting));
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayContextNativeCallbackForTesting));
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayProgramNativeCallbackForTesting));
    CHECK(unsafeOperatorFailure(failNextCudaOperatorRelayWidthNativeCallbackForTesting));

    // Finalization moves tile/status/bounds/topology proof off the calling
    // worker.  The direct path must preserve every published metadata field
    // and device channel relative to the legacy synchronous finalizer.
    auto prepareFinal = [&](std::shared_ptr<const UsdGenCudaExecutionPlan> const& finalPlan,
                            std::unique_ptr<UsdGenCudaExecutionWorkspace>& workspace,
                            uint64_t generation,
                            std::shared_ptr<const UsdGenDeviceGeneration> previous = {}) {
        auto finalJob=CreateCudaExecutionJob(finalPlan,*workspace,2,generation,&diagnostics,std::move(previous));
        if (!finalJob || !RunCudaJobSourceForTesting(finalJob)) return std::shared_ptr<UsdGenCudaExecutionJob>{};
        for (size_t i=0;i<CudaExecutionJobOperatorCount(*finalJob);++i)
            if (!RunCudaJobOperatorForTesting(finalJob,i)) return std::shared_ptr<UsdGenCudaExecutionJob>{};
        return finalJob;
    };
    auto finishAsync = [&](std::shared_ptr<UsdGenCudaExecutionJob> const& finalJob,
                           std::shared_ptr<const UsdGenDeviceGeneration>* result,
                           std::atomic<int>* calls = nullptr) {
        std::atomic<bool> done{false};
        bool const accepted=FinalizeCudaExecutionJobAsync(finalJob,[&](auto generation) {
            *result=std::move(generation); if (calls) calls->fetch_add(1,std::memory_order_relaxed);
            done.store(true,std::memory_order_release);
        });
        return accepted && Wait(done);
    };

    // An async Deform pose is provisional until terminal generation
    // publication.  Clean finalizer rejection and job abandonment both
    // restore the last accepted surface targets, coefficients, key, identity,
    // and counters; a later retry remains usable and deterministic.
    auto sameBindingStats = [](std::vector<UsdGenCudaBindingStats> const& a,
                               std::vector<UsdGenCudaBindingStats> const& b) {
        if (a.size()!=b.size()) return false;
        for (size_t i=0;i<a.size();++i)
            if (a[i].key!=b[i].key || a[i].identity!=b[i].identity ||
                a[i].bindCount!=b[i].bindCount || a[i].solveCount!=b[i].solveCount ||
                a[i].sampleCount!=b[i].sampleCount) return false;
        return true;
    };
    auto prepareAsyncRbf = [&](std::shared_ptr<const UsdGenCudaExecutionPlan> const& rbfPlan,
                               std::unique_ptr<UsdGenCudaExecutionWorkspace>& workspace,
                               uint64_t generation,
                               std::shared_ptr<const UsdGenDeviceGeneration> previous = {}) {
        auto candidate=CreateCudaExecutionJob(rbfPlan,*workspace,2,generation,&diagnostics,
                                              std::move(previous));
        if (!candidate || !RunCudaJobSourceForTesting(candidate))
            return std::shared_ptr<UsdGenCudaExecutionJob>{};
        auto const count=CudaExecutionJobOperatorCount(*candidate);
        if (!count) return std::shared_ptr<UsdGenCudaExecutionJob>{};
        for (size_t i=0;i+1<count;++i)
            if (!RunCudaJobOperatorForTesting(candidate,i))
                return std::shared_ptr<UsdGenCudaExecutionJob>{};
        std::atomic<bool> done{false}, ok{false};
        if (!ExecuteCudaJobOperatorAsync(candidate,count-1,[&](bool value) {
                ok.store(value,std::memory_order_release);
                done.store(true,std::memory_order_release);
            }) || !Wait(done) || !ok.load(std::memory_order_acquire))
            return std::shared_ptr<UsdGenCudaExecutionJob>{};
        return candidate;
    };
    auto const acceptedRbfStats=GetCudaBindingStats(*rbfAsyncWorkspace);
    CHECK(acceptedRbfStats.size()==1 && acceptedRbfStats[0].identity!=0 &&
          acceptedRbfStats[0].bindCount==1 && acceptedRbfStats[0].solveCount==1);

    auto rejectedRbf=prepareAsyncRbf(plan,rbfAsyncWorkspace,111,rbfAsyncGeneration);
    CHECK(rejectedRbf && sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),acceptedRbfStats));
    failNextCudaFinalizationRelayAllocationForTesting();
    std::shared_ptr<const UsdGenDeviceGeneration> rejectedRbfResult;
    CHECK(finishAsync(rejectedRbf,&rejectedRbfResult) && !rejectedRbfResult &&
          sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),acceptedRbfStats));

    auto abandonedRbf=prepareAsyncRbf(plan,rbfAsyncWorkspace,112,rbfAsyncGeneration);
    CHECK(abandonedRbf && sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),acceptedRbfStats));
    abandonedRbf.reset();
    CHECK(sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),acceptedRbfStats));

    auto retriedRbf=prepareAsyncRbf(plan,rbfAsyncWorkspace,113,rbfAsyncGeneration);
    CHECK(retriedRbf);
    std::shared_ptr<const UsdGenDeviceGeneration> retriedRbfResult;
    CHECK(finishAsync(retriedRbf,&retriedRbfResult) && retriedRbfResult);
    auto const retriedRbfStats=GetCudaBindingStats(*rbfAsyncWorkspace);
    CHECK(retriedRbfStats.size()==1 &&
          retriedRbfStats[0].identity==acceptedRbfStats[0].identity &&
          retriedRbfStats[0].bindCount==acceptedRbfStats[0].bindCount &&
          retriedRbfStats[0].solveCount==acceptedRbfStats[0].solveCount+1);

    auto changedRestDesc=Desc();
    changedRestDesc.surfaces.front().restPoints.back()[2]+=.25f;
    changedRestDesc.surfaces.front().points.back()[2]+=.25f;
    auto changedRestPlan=CompileCudaGraph(usdGenUsd::Lower(changedRestDesc),&diagnostics); CHECK(changedRestPlan);
    auto rejectedRebind=prepareAsyncRbf(changedRestPlan,rbfAsyncWorkspace,114,retriedRbfResult);
    CHECK(rejectedRebind && sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),retriedRbfStats));
    failNextCudaFinalizationRelayTopologyAllocationForTesting();
    std::shared_ptr<const UsdGenDeviceGeneration> rejectedRebindResult;
    CHECK(finishAsync(rejectedRebind,&rejectedRebindResult) && !rejectedRebindResult &&
          sameBindingStats(GetCudaBindingStats(*rbfAsyncWorkspace),retriedRbfStats));
    auto retriedRebind=prepareAsyncRbf(changedRestPlan,rbfAsyncWorkspace,115,retriedRbfResult);
    CHECK(retriedRebind);
    std::shared_ptr<const UsdGenDeviceGeneration> retriedRebindResult;
    CHECK(finishAsync(retriedRebind,&retriedRebindResult) && retriedRebindResult);
    auto const retriedRebindStats=GetCudaBindingStats(*rbfAsyncWorkspace);
    CHECK(retriedRebindStats.size()==1 &&
          retriedRebindStats[0].identity!=retriedRbfStats[0].identity &&
          retriedRebindStats[0].bindCount==1 && retriedRebindStats[0].solveCount==1);

    auto emptyRbfPlan=CompileCudaGraph(usdGenUsd::Lower(SourceControlDesc(true,true)),&diagnostics); CHECK(emptyRbfPlan);
    auto emptyRbfWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(emptyRbfWorkspace);
    auto emptyRbfJob=prepareAsyncRbf(emptyRbfPlan,emptyRbfWorkspace,116); CHECK(emptyRbfJob);
    std::shared_ptr<const UsdGenDeviceGeneration> emptyRbfResult;
    CHECK(finishAsync(emptyRbfJob,&emptyRbfResult) && emptyRbfResult &&
          emptyRbfResult->Geometry().pointCount==0);
    auto const emptyRbfStats=GetCudaBindingStats(*emptyRbfWorkspace);
    CHECK(emptyRbfStats.size()==1 && emptyRbfStats.front().identity==0 &&
          emptyRbfStats.front().bindCount==0 && emptyRbfStats.front().solveCount==0);

    // The compatibility synchronous finalizer must resolve the same pending
    // transaction if its Deform stage was submitted through the relay.
    auto syncHandoffWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(syncHandoffWorkspace);
    auto syncHandoffJob=prepareAsyncRbf(plan,syncHandoffWorkspace,117); CHECK(syncHandoffJob);
    auto syncHandoffResult=FinalizeCudaJobForTesting(syncHandoffJob); CHECK(syncHandoffResult);
    auto const syncHandoffStats=GetCudaBindingStats(*syncHandoffWorkspace);
    CHECK(syncHandoffStats.size()==1 && syncHandoffStats.front().identity!=0 &&
          syncHandoffStats.front().bindCount==1 && syncHandoffStats.front().solveCount==1);

    auto finalAsyncWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto finalLegacyWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    CHECK(finalAsyncWorkspace && finalLegacyWorkspace);
    auto sameTopologyDesc=LengthExpressionDesc();
    // Change an observable value channel while preserving source/Length tile
    // topology.
    sameTopologyDesc.nodes[2].params.front().value=VtValue(.08f);
    CHECK(sameTopologyDesc.nodes[2].type==TfToken("UsdGenWidth") &&
          sameTopologyDesc.nodes[2].inputs.size()==1 &&
          sameTopologyDesc.nodes[2].inputs.front()==SdfPath("/Stages/Hair/Length") &&
          sameTopologyDesc.nodes[2].params.front().name==TfToken("width") &&
          sameTopologyDesc.nodes[2].params.front().value==VtValue(.08f));
    auto sameTopologyPlan=CompileCudaGraph(usdGenUsd::Lower(sameTopologyDesc),&diagnostics); CHECK(sameTopologyPlan);
    auto finalAsyncJob=prepareFinal(sameTopologyPlan,finalAsyncWorkspace,100,lengthAsyncGeneration); CHECK(finalAsyncJob);
    std::shared_ptr<const UsdGenDeviceGeneration> finalAsyncGeneration;
    std::atomic<int> finalCalls{0};
    CHECK(finishAsync(finalAsyncJob,&finalAsyncGeneration,&finalCalls) && finalAsyncGeneration && finalCalls==1);
    auto finalLegacy=ExecuteCudaPlanForTesting(sameTopologyPlan,*finalLegacyWorkspace,2,100,&diagnostics,lengthAsyncGeneration);
    cudaStream_t finalReader=nullptr; CHECK(cudaStreamCreateWithFlags(&finalReader,cudaStreamNonBlocking)==cudaSuccess);
    // Keep the async-final publication oracle factored: a topology-version
    // disagreement must not obscure a geometry/value regression (or vice
    // versa) behind one combined CHECK.
    CHECK(finalLegacy);
    CHECK(Same(finalAsyncGeneration,finalLegacy,finalReader));
    CHECK(SameRetainedCurveChannels(finalAsyncGeneration,finalLegacy,finalReader));
    CHECK(SameFinalMetadata(finalAsyncGeneration,finalLegacy));
    CHECK(!Same(finalAsyncGeneration,lengthAsyncGeneration,finalReader));
    CHECK(finalAsyncGeneration->Geometry().topologyVersion == lengthAsyncGeneration->Geometry().topologyVersion &&
          finalAsyncGeneration->Geometry().valueVersion != lengthAsyncGeneration->Geometry().valueVersion);
    CHECK(cudaStreamDestroy(finalReader)==cudaSuccess);

    // A changed cull layout advances the topology version while the async
    // result remains fully readable, including the all-cull empty case.
    auto finalCullWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto finalCullJob=prepareFinal(cullPlan,finalCullWorkspace,101,lengthAsyncGeneration); CHECK(finalCullJob);
    std::shared_ptr<const UsdGenDeviceGeneration> finalCull;
    CHECK(finishAsync(finalCullJob,&finalCull) && finalCull &&
          finalCull->Geometry().topologyVersion != lengthAsyncGeneration->Geometry().topologyVersion);
    auto finalEmptyWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto finalEmptyJob=prepareFinal(allCullPlan,finalEmptyWorkspace,102,lengthAsyncGeneration); CHECK(finalEmptyJob);
    std::shared_ptr<const UsdGenDeviceGeneration> finalEmpty;
    CHECK(finishAsync(finalEmptyJob,&finalEmpty) && finalEmpty &&
          finalEmpty->Geometry().elementCount==0 && finalEmpty->Geometry().pointCount==0 &&
          !finalEmpty->Geometry().tiles.empty());

    struct FinalGate { void (*arm)(); void (*wait)(); void (*release)(); };
    FinalGate const finalGates[] = {
        {armCudaFinalizationAsyncTileCallbackGateForTesting,
         waitCudaFinalizationAsyncTileCallbackGateForTesting,
         releaseCudaFinalizationAsyncTileCallbackGateForTesting},
        {armCudaFinalizationAsyncBoundsCallbackGateForTesting,
         waitCudaFinalizationAsyncBoundsCallbackGateForTesting,
         releaseCudaFinalizationAsyncBoundsCallbackGateForTesting},
        {armCudaFinalizationAsyncMetadataCallbackGateForTesting,
         waitCudaFinalizationAsyncMetadataCallbackGateForTesting,
         releaseCudaFinalizationAsyncMetadataCallbackGateForTesting},
    };
    for (auto const& gate : finalGates) {
        setCudaFinalizationRelayCapacityForTesting(1); ResetFinalizationRelayCapacity resetPhaseCapacity;
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(workspace);
        auto held=prepareFinal(lengthPlan,workspace,103,lengthAsyncGeneration); CHECK(held);
        auto blockedWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(blockedWorkspace);
        auto blocked=prepareFinal(lengthPlan,blockedWorkspace,104,lengthAsyncGeneration); CHECK(blocked);
        std::shared_ptr<const UsdGenDeviceGeneration> result;
        std::atomic<int> calls{0}, blockedCalls{0};
        gate.arm(); ReleaseFinalizationGate releaseFinalGate{gate.release};
        CHECK(FinalizeCudaExecutionJobAsync(held,[&](auto generation) {
            result=std::move(generation); ++calls;
        }));
        gate.wait();
        CHECK(calls==0 && !result && cudaFinalizationRelayOccupiedCountForTesting()!=0 &&
              !FinalizeCudaExecutionJobAsync(blocked,[&](auto) { ++blockedCalls; }) && blockedCalls==0 &&
              !FinalizeCudaExecutionJobAsync(held,[](auto) {}) &&
              !FinalizeCudaJobForTesting(held) && !RunCudaJobOperatorForTesting(held,0));
        gate.release(); releaseFinalGate.release=nullptr;
        auto const deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        while (calls.load(std::memory_order_acquire)==0 && std::chrono::steady_clock::now()<deadline) std::this_thread::yield();
        CHECK(calls==1 && result && cudaFinalizationRelayOccupiedCountForTesting()==0);
    }

    // The metadata launcher-return gate pauses after the native terminal is
    // installed; a retained previous lease remains readable until return.
    auto returnWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(returnWorkspace);
    auto returnJob=prepareFinal(lengthPlan,returnWorkspace,104,lengthAsyncGeneration); CHECK(returnJob);
    std::shared_ptr<const UsdGenDeviceGeneration> returnResult;
    std::atomic<int> returnCalls{0};
    armCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
    ReleaseFinalizationGate releaseReturn{releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting};
    CHECK(FinalizeCudaExecutionJobAsync(returnJob,[&](auto generation) {
        returnResult=std::move(generation); ++returnCalls;
    }));
    waitCudaFinalizationAsyncMetadataLauncherReturnGateForTesting();
    cudaStream_t previousReader=nullptr; CHECK(cudaStreamCreateWithFlags(&previousReader,cudaStreamNonBlocking)==cudaSuccess);
    auto previousLease=gpu::AcquireGeometry(lengthAsyncGeneration,previousReader);
    float3 previousPoint{};
    CHECK(previousLease && previousLease.Geometry().points.size &&
          cudaMemcpyAsync(&previousPoint,previousLease.Geometry().points.data,sizeof(previousPoint),
                          cudaMemcpyDeviceToHost,previousReader)==cudaSuccess &&
          cudaStreamSynchronize(previousReader)==cudaSuccess && std::isfinite(previousPoint.x));
    previousLease={};
    CHECK(cudaStreamDestroy(previousReader)==cudaSuccess && returnCalls==0);
    releaseCudaFinalizationAsyncMetadataLauncherReturnGateForTesting(); releaseReturn.release=nullptr;
    auto returnDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (returnCalls.load(std::memory_order_acquire)==0 && std::chrono::steady_clock::now()<returnDeadline) std::this_thread::yield();
    CHECK(returnCalls==1 && returnResult);

    // A shared cap rejects the second request without retaining its callback;
    // after the first terminal proof, that normal slot is reusable.
    {
    setCudaFinalizationRelayCapacityForTesting(1); ResetFinalizationRelayCapacity resetFinalCapacity;
    auto capAWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
    auto capBWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(capAWorkspace && capBWorkspace);
    auto capA=prepareFinal(lengthPlan,capAWorkspace,105,lengthAsyncGeneration);
    auto capB=prepareFinal(lengthPlan,capBWorkspace,106,lengthAsyncGeneration); CHECK(capA && capB);
    std::atomic<int> capACalls{0}, capBCalls{0}; std::shared_ptr<const UsdGenDeviceGeneration> capAResult, capBResult;
    armCudaFinalizationAsyncTileCallbackGateForTesting();
    ReleaseFinalizationGate releaseCap{releaseCudaFinalizationAsyncTileCallbackGateForTesting};
    CHECK(FinalizeCudaExecutionJobAsync(capA,[&](auto generation) { capAResult=std::move(generation); ++capACalls; }));
    waitCudaFinalizationAsyncTileCallbackGateForTesting();
    CHECK(!FinalizeCudaExecutionJobAsync(capB,[&](auto generation) { capBResult=std::move(generation); ++capBCalls; }) && capBCalls==0);
    releaseCudaFinalizationAsyncTileCallbackGateForTesting(); releaseCap.release=nullptr;
    auto capDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while (capACalls.load(std::memory_order_acquire)==0 && std::chrono::steady_clock::now()<capDeadline) std::this_thread::yield();
    CHECK(capACalls==1 && capAResult && cudaFinalizationRelayOccupiedCountForTesting()==0 &&
          finishAsync(capB,&capBResult,&capBCalls) && capBCalls==1 && capBResult);
    }

    // Allocation rejection before the first tile submission is a proven,
    // retryable failure: it delivers one null result but neither poisons the
    // workspace nor consumes a permanent finalization slot.
    auto finalAllocationWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(finalAllocationWorkspace);
    auto finalAllocationRejected=prepareFinal(lengthPlan,finalAllocationWorkspace,107,lengthAsyncGeneration); CHECK(finalAllocationRejected);
    size_t const finalAllocationQuarantined=cudaFinalizationRelayQuarantinedCountForTesting();
    std::shared_ptr<const UsdGenDeviceGeneration> finalAllocationResult;
    std::atomic<int> finalAllocationCalls{0};
    failNextCudaFinalizationRelayAllocationForTesting();
    CHECK(finishAsync(finalAllocationRejected,&finalAllocationResult,&finalAllocationCalls) && !finalAllocationResult &&
          finalAllocationCalls==1 && cudaFinalizationRelayOccupiedCountForTesting()==0 &&
          cudaFinalizationRelayQuarantinedCountForTesting()==finalAllocationQuarantined);
    auto finalAllocationRetry=prepareFinal(lengthPlan,finalAllocationWorkspace,108,lengthAsyncGeneration); CHECK(finalAllocationRetry);
    std::shared_ptr<const UsdGenDeviceGeneration> finalAllocationRetryResult;
    CHECK(finishAsync(finalAllocationRetry,&finalAllocationRetryResult) && finalAllocationRetryResult);

    // The topology comparator has a later clean allocation boundary, after
    // tile/bounds proof but before metadata D2H. It is likewise retryable
    // when a previous generation requires that comparison.
    auto topologyAllocationWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(topologyAllocationWorkspace);
    auto topologyAllocationRejected=prepareFinal(lengthPlan,topologyAllocationWorkspace,109,lengthAsyncGeneration);
    CHECK(topologyAllocationRejected);
    size_t const topologyAllocationQuarantined=cudaFinalizationRelayQuarantinedCountForTesting();
    std::shared_ptr<const UsdGenDeviceGeneration> topologyAllocationResult;
    std::atomic<int> topologyAllocationCalls{0};
    failNextCudaFinalizationRelayTopologyAllocationForTesting();
    CHECK(finishAsync(topologyAllocationRejected,&topologyAllocationResult,&topologyAllocationCalls) &&
          !topologyAllocationResult && topologyAllocationCalls==1 &&
          cudaFinalizationRelayOccupiedCountForTesting()==0 &&
          cudaFinalizationRelayQuarantinedCountForTesting()==topologyAllocationQuarantined);
    auto topologyAllocationRetry=prepareFinal(lengthPlan,topologyAllocationWorkspace,110,lengthAsyncGeneration);
    CHECK(topologyAllocationRetry);
    std::shared_ptr<const UsdGenDeviceGeneration> topologyAllocationRetryResult;
    CHECK(finishAsync(topologyAllocationRetry,&topologyAllocationRetryResult) && topologyAllocationRetryResult);

    auto unsafeFinalFailure = [&](void (*inject)()) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!workspace) return false;
        auto failing=prepareFinal(lengthPlan,workspace,107,lengthAsyncGeneration);
        if (!failing) return false;
        size_t const before=cudaFinalizationRelayQuarantinedCountForTesting();
        std::atomic<int> calls{0}; std::shared_ptr<const UsdGenDeviceGeneration> result;
        inject();
        if (!finishAsync(failing,&result,&calls) || result || calls!=1 ||
            cudaFinalizationRelayQuarantinedCountForTesting()!=before+1)
            return false;
        cudaStream_t reader=nullptr;
        if (cudaStreamCreateWithFlags(&reader,cudaStreamNonBlocking)!=cudaSuccess) return false;
        bool retained=false;
        {
            // The lease's completion event is recorded on reader at scope
            // exit, so it must be released before destroying that stream.
            auto lease=gpu::AcquireGeometry(lengthAsyncGeneration,reader);
            float3 point{};
            retained=lease && lease.Geometry().pointCount &&
                cudaMemcpyAsync(&point,lease.Geometry().points.data,sizeof(point),
                                cudaMemcpyDeviceToHost,reader)==cudaSuccess &&
                cudaStreamSynchronize(reader)==cudaSuccess && std::isfinite(point.x);
        }
        bool const destroyed=cudaStreamDestroy(reader)==cudaSuccess;
        return retained && destroyed;
    };
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayTileCallbackInstallForTesting));
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayBoundsCallbackInstallForTesting));
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayMetadataCallbackInstallForTesting));
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayTileNativeCallbackForTesting));
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayBoundsNativeCallbackForTesting));
    CHECK(unsafeFinalFailure(failNextCudaFinalizationRelayMetadataNativeCallbackForTesting));

    // Illegal order is rejected at admission; source-only jobs publish.
    auto bad=CreateCudaExecutionJob(plan,*stagedWorkspace,3,3,&diagnostics); CHECK(bad);
    CHECK(!RunCudaJobOperatorForTesting(bad,0));
    auto sourceDesc=Desc(); sourceDesc.nodes.resize(1); sourceDesc.terminal=sourceDesc.nodes.front().path;
    auto sourcePlan=CompileCudaGraph(usdGenUsd::Lower(sourceDesc),&diagnostics); CHECK(sourcePlan);
    auto sourceOnly=CreateCudaExecutionJob(sourcePlan,*stagedWorkspace,3,3,&diagnostics); CHECK(sourceOnly);
    CHECK(CudaExecutionJobOperatorCount(*sourceOnly)==0 && RunCudaJobSourceForTesting(sourceOnly));
    auto sourceGeneration=FinalizeCudaJobForTesting(sourceOnly); CHECK(sourceGeneration);
    // A canonical empty C3 source still reaches the direct asynchronous
    // finalizer: no placeholder curve/point channels or tile ranges may be
    // manufactured for it.
    auto zeroSourceDesc=sourceDesc;
    auto& zeroSource=zeroSourceDesc.curveSets.front();
    zeroSource.curveVertexCounts.clear(); zeroSource.points.clear(); zeroSource.rest.clear();
    zeroSource.curveId.clear(); zeroSource.skinPrim.clear(); zeroSource.skinPrimUv.clear();
    auto zeroSourcePlan=CompileCudaGraph(usdGenUsd::Lower(zeroSourceDesc),&diagnostics); CHECK(zeroSourcePlan);
    auto zeroSourceWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(zeroSourceWorkspace);
    auto zeroSourceJob=CreateCudaExecutionJob(zeroSourcePlan,*zeroSourceWorkspace,3,109,&diagnostics); CHECK(zeroSourceJob);
    CHECK(RunCudaJobSourceForTesting(zeroSourceJob));
    std::shared_ptr<const UsdGenDeviceGeneration> zeroSourceGeneration;
    CHECK(finishAsync(zeroSourceJob,&zeroSourceGeneration) && zeroSourceGeneration &&
          zeroSourceGeneration->Geometry().elementCount==0 && zeroSourceGeneration->Geometry().pointCount==0 &&
          zeroSourceGeneration->Geometry().tiles.empty());
    // The initial publication has no prior lease/topology comparison but uses
    // the same asynchronous tile/bounds/metadata pipeline.
    auto initialFinalWorkspace=CreateCudaExecutionWorkspace(-1,&diagnostics); CHECK(initialFinalWorkspace);
    auto initialFinal=CreateCudaExecutionJob(sourcePlan,*initialFinalWorkspace,3,108,&diagnostics); CHECK(initialFinal);
    CHECK(RunCudaJobSourceForTesting(initialFinal));
    std::shared_ptr<const UsdGenDeviceGeneration> initialAsync;
    CHECK(finishAsync(initialFinal,&initialAsync) && initialAsync &&
          initialAsync->Geometry().topologyVersion==108);
    auto unsafeInitialMetadataFailure = [&](void (*inject)()) {
        auto workspace=CreateCudaExecutionWorkspace(-1,&diagnostics);
        if (!workspace) return false;
        auto failing=CreateCudaExecutionJob(sourcePlan,*workspace,3,111,&diagnostics);
        if (!failing || !RunCudaJobSourceForTesting(failing)) return false;
        size_t const before=cudaFinalizationRelayQuarantinedCountForTesting();
        std::shared_ptr<const UsdGenDeviceGeneration> result; std::atomic<int> calls{0};
        inject();
        return finishAsync(failing,&result,&calls) && !result && calls==1 &&
            cudaFinalizationRelayOccupiedCountForTesting()!=0 &&
            cudaFinalizationRelayQuarantinedCountForTesting()==before+1;
    };
    CHECK(unsafeInitialMetadataFailure(failNextCudaFinalizationRelayMetadataCallbackInstallForTesting));
    CHECK(unsafeInitialMetadataFailure(failNextCudaFinalizationRelayMetadataNativeCallbackForTesting));
    CHECK(!FinalizeCudaJobForTesting(sourceOnly)); // duplicate final is latched
    cudaStream_t retainedReader=nullptr; CHECK(cudaStreamCreateWithFlags(&retainedReader,cudaStreamNonBlocking)==cudaSuccess);
    CHECK(gpu::AcquireGeometry(sourceGeneration,retainedReader)); // retained publication remains readable
    CHECK(cudaStreamDestroy(retainedReader)==cudaSuccess);
    CHECK(!RunCudaJobOperatorForTesting(sourceOnly,0));

    // Automatic root capture must reserve the same source/output peak as an
    // equivalent authored binding.  In particular, an absent or partial
    // skinPrim/skinPrimUv pair can add root binding planes at
    // runtime; admission must not size from the authored arrays alone.  Each
    // variant is isolated in a workspace, and the retained lease proves that
    // a cap rejection leaves the previous COW publication untouched.
    auto runAutoRootVariant = [&](UsdGenGraphDesc variant, uint64_t generation,
                                  uint64_t* peakOut) {
        diagnostics = {};
        auto variantPlan = CompileCudaGraph(usdGenUsd::Lower(variant), &diagnostics);
        if (!variantPlan || diagnostics.HasErrors()) return false;
        auto variantWorkspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
        if (!variantWorkspace) return false;
        diagnostics = {};
        auto first = ExecuteCudaPlanForTesting(variantPlan, *variantWorkspace, 2,
                                      generation, &diagnostics);
        if (!first || diagnostics.HasErrors() ||
            first->Geometry().elementCount != 2 ||
            first->Geometry().pointCount != 4 || !first->Owner()) return false;
        auto const priorOwner = first->Owner();
        auto const baseline = literalResources->Snapshot();
        if (baseline.usableBytes < baseline.usedBytes) return false;
        auto const available = baseline.usableBytes - baseline.usedBytes;
        auto filler = literalResources->TryReserve(
            available, UsdGenExecutionResourceKind::Active);
        if (!filler) return false;
        auto const saturated = literalResources->Snapshot();
        bool rejectOk = false;
        {
            cudaStream_t reader = nullptr;
            if (cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking) !=
                cudaSuccess) {
                filler->Release();
                return false;
            }
            {
                auto priorLease = gpu::AcquireGeometry(first, reader);
                diagnostics = {};
                auto rejected = ExecuteCudaPlanForTesting(variantPlan, *variantWorkspace,
                                                 2, generation + 1, &diagnostics,
                                                 first);
                uint64_t peak = 0;
                // The failed call must retain the old owner and leave the
                // complete resource ledger unchanged. `rejected` is
                // intentionally expected to be null.
                auto const afterReject = literalResources->Snapshot();
                rejectOk = priorLease && !rejected &&
                    ParseLiteralLengthPeak(diagnostics, &peak) &&
                    afterReject.usedBytes == saturated.usedBytes &&
                    ResourceKindBytes(afterReject,
                        UsdGenExecutionResourceKind::Pending) ==
                        ResourceKindBytes(saturated,
                            UsdGenExecutionResourceKind::Pending) &&
                    first->Owner() == priorOwner;
                if (peakOut) *peakOut = peak;
                if (!rejectOk) {
                    std::fprintf(stderr, "auto-root reject: peak=%llu used=%zu/%zu pending=%zu/%zu\n",
                        static_cast<unsigned long long>(peak), afterReject.usedBytes,
                        saturated.usedBytes,
                        ResourceKindBytes(afterReject, UsdGenExecutionResourceKind::Pending),
                        ResourceKindBytes(saturated, UsdGenExecutionResourceKind::Pending));
                    for (auto const& error : diagnostics.errors)
                        std::fprintf(stderr, "%s\n", error.c_str());
                }
            }
            cudaStreamDestroy(reader);
        }
        filler->Release();
        if (!rejectOk) return false;
        diagnostics = {};
        auto direct = ExecuteCudaPlanForTesting(variantPlan, *variantWorkspace, 2,
                                       generation + 2, &diagnostics, first);
        if (!direct || diagnostics.HasErrors() || !direct->Owner() ||
            direct->Owner() == priorOwner ||
            direct->Geometry().elementCount != 2 ||
            direct->Geometry().pointCount != 4) return false;

        // Exercise the staged asynchronous publication with the same
        // auto-bound source.  Synchronous stage calls are intentionally not
        // used here: the callback relays prove the async ownership boundary.
        diagnostics = {};
        auto asyncJob = CreateCudaExecutionJob(
            variantPlan, *variantWorkspace, 2, generation + 3, &diagnostics,
            direct);
        if (!asyncJob) return false;
        std::atomic<bool> sourceDone{false}, sourceOk{false};
        if (!ExecuteCudaJobSourceAsync(asyncJob, [&](bool ok) {
                sourceOk.store(ok, std::memory_order_release);
                sourceDone.store(true, std::memory_order_release);
            }) || !Wait(sourceDone) ||
            !sourceOk.load(std::memory_order_acquire)) return false;
        for (size_t i = 0; i != CudaExecutionJobOperatorCount(*asyncJob); ++i) {
            std::atomic<bool> opDone{false}, opOk{false};
            if (!ExecuteCudaJobOperatorAsync(asyncJob, i, [&](bool ok) {
                    opOk.store(ok, std::memory_order_release);
                    opDone.store(true, std::memory_order_release);
                }) || !Wait(opDone) || !opOk.load(std::memory_order_acquire))
                return false;
        }
        std::shared_ptr<const UsdGenDeviceGeneration> asyncResult;
        std::atomic<bool> finalDone{false};
        if (!FinalizeCudaExecutionJobAsync(asyncJob, [&](auto result) {
                asyncResult = std::move(result);
                finalDone.store(true, std::memory_order_release);
            }) || !Wait(finalDone) || !asyncResult ||
            asyncResult->Geometry().elementCount != 2 ||
            asyncResult->Geometry().pointCount != 4) return false;

        cudaStream_t compareStream = nullptr;
        if (cudaStreamCreateWithFlags(&compareStream, cudaStreamNonBlocking) !=
            cudaSuccess) return false;
        bool parity = Same(first, direct, compareStream) &&
            Same(direct, asyncResult, compareStream) &&
            SameRetainedCurveChannels(first, direct, compareStream) &&
            SameRetainedCurveChannels(direct, asyncResult, compareStream);
        {
            auto retained = gpu::AcquireGeometry(asyncResult, compareStream);
            // This Source->Length->Width route publishes binding planes;
            // unlike ScatterGrow it does not publish device T/B/N planes.
            parity = parity && retained && retained.RootPrim().size == 2 &&
                retained.RootUV().size == 2;
        }
        parity = parity && cudaStreamDestroy(compareStream) == cudaSuccess;
        return parity;
    };
    auto absentRoots = literalLengthDesc;
    absentRoots.curveSets.front().skinPrim.clear();
    absentRoots.curveSets.front().skinPrimUv.clear();
    uint64_t absentRootPeak = 0;
    CHECK(runAutoRootVariant(absentRoots, 38, &absentRootPeak) &&
          absentRootPeak == literalPeak);
    auto partialRoots = literalLengthDesc;
    partialRoots.curveSets.front().skinPrim = {0};
    partialRoots.curveSets.front().skinPrimUv = {{.2f, .2f}};
    uint64_t partialRootPeak = 0;
    CHECK(runAutoRootVariant(partialRoots, 39, &partialRootPeak) &&
          partialRootPeak == literalPeak);
    return 0;
}
