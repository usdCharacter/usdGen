// testUsdGenReferenceSource — UsdGenReferenceSource pulls one already-baked
// in-stage BasisCurves prim into the operation stack through rel
// usdGen:reference, as a generator alternative to Scatter/Instance.
//
// T1 headless: an in-memory stage drives BOTH graph-desc builders, then the
// real compiler + scheduler cook the stack. Asserted: the rel routes to the
// node's reference lane with a Reference-role curve set, both builders agree,
// the cooked points equal the baked points verbatim, a downstream operator
// consumes the pulled geometry, and an empty reference fails closed.

#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/compile/graph.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/exec/scheduler.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/xform.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"

#include <cmath>
#include <cstdio>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using namespace usdGenImaging;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
    else     { std::printf("ok:   %s\n", what.c_str()); }
}

SdfPath const kDescription("/World/Groom/Hair");
SdfPath const kBaked("/World/Baked");

// Baked strands: 2 curves x 3 CVs with authored ids, widths and rest.
usdGen::UsdGenVec3fArray BakedPoints()
{
    return usdGen::UsdGenVec3fArray{
        usdGen::UsdGenVec3f(0.0f, 0.0f, 0.0f), usdGen::UsdGenVec3f(0.0f, 1.0f, 0.1f), usdGen::UsdGenVec3f(0.0f, 2.0f, 0.2f),
        usdGen::UsdGenVec3f(1.0f, 0.0f, 0.0f), usdGen::UsdGenVec3f(1.0f, 1.0f, 0.1f), usdGen::UsdGenVec3f(1.0f, 2.0f, 0.4f)};
}

// A description whose stack is ReferenceSource -> Width. |withTarget| false
// leaves usdGen:reference unauthored (the fail-closed case).
UsdStageRefPtr MakeStage(bool withTarget, bool malformedClump = false)
{
    UsdStageRefPtr stage = UsdStage::CreateInMemory("referenceSource");
    UsdGeomXform::Define(stage, SdfPath("/World"));

    usdGen::UsdGenVec3fArray const points = BakedPoints();
    UsdGeomBasisCurves baked =
        UsdGeomBasisCurves::Define(stage, kBaked);
    baked.CreateTypeAttr().Set(UsdGeomTokens->cubic);
    baked.CreateBasisAttr().Set(UsdGeomTokens->bspline);
    baked.CreateWrapAttr().Set(UsdGeomTokens->pinned);
    baked.CreateCurveVertexCountsAttr().Set(VtIntArray{3, 3});
    baked.CreatePointsAttr().Set(ToHost(points));
    baked.CreateWidthsAttr().Set(VtFloatArray{0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f});
    baked.SetWidthsInterpolation(UsdGeomTokens->vertex);
    baked.GetPrim().AddAppliedSchema(TfToken("UsdGenCurveAPI"));
    UsdGeomPrimvarsAPI bakedPv(baked.GetPrim());
    bakedPv.CreatePrimvar(TfToken("usdGen:role"), SdfValueTypeNames->Token,
                          UsdGeomTokens->constant)
        .Set(TfToken("hair"));
    bakedPv.CreatePrimvar(TfToken("rest"), SdfValueTypeNames->Point3fArray,
                          UsdGeomTokens->vertex)
        .Set(ToHost(points));
    bakedPv.CreatePrimvar(TfToken("usdGen:curveId"), SdfValueTypeNames->UInt64Array,
                          UsdGeomTokens->uniform)
        .Set(VtArray<uint64_t>{7, 9});
    bakedPv.CreatePrimvar(TfToken("tubeId"), SdfValueTypeNames->IntArray,
                          UsdGeomTokens->uniform)
        .Set(VtIntArray{11, 12});
    bakedPv.CreatePrimvar(TfToken("regionId"), SdfValueTypeNames->IntArray,
                          UsdGeomTokens->uniform)
        .Set(VtIntArray{101, 102});
    bakedPv.CreatePrimvar(TfToken("hierarchyLevel"), SdfValueTypeNames->IntArray,
                          UsdGeomTokens->uniform)
        .Set(VtIntArray{1, 2});
    bakedPv.CreatePrimvar(TfToken("clumpId_0"), SdfValueTypeNames->IntArray,
                          UsdGeomTokens->uniform)
        .Set(VtIntArray{0, 0});
    bakedPv.CreatePrimvar(TfToken("clumpCenter_0"), SdfValueTypeNames->Float3Array,
                          UsdGeomTokens->uniform)
        .Set(VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(0, 0, 0)});
    if (malformedClump) {
        bakedPv.CreatePrimvar(TfToken("clumpCenterId_0"), SdfValueTypeNames->Float2Array,
                              UsdGeomTokens->uniform)
            .Set(VtVec2fArray{GfVec2f(7, 0), GfVec2f(7, 0)});
    } else {
        bakedPv.CreatePrimvar(TfToken("clumpCenterId_0"), SdfValueTypeNames->Int2Array,
                              UsdGeomTokens->uniform)
            .Set(VtVec2iArray{GfVec2i(-1, 0x76543210),
                               GfVec2i(-1, 0x76543210)});
    }
    bakedPv.CreatePrimvar(TfToken("clumpWeight_0"), SdfValueTypeNames->FloatArray,
                          UsdGeomTokens->vertex)
        .Set(VtFloatArray{0, .2f, .6f, 0, .3f, .8f});

    UsdGeomXform::Define(stage, SdfPath("/World/Groom"));
    UsdPrim description =
        stage->DefinePrim(kDescription, TfToken("UsdGenDescription"));
    stage->DefinePrim(kDescription.AppendChild(TfToken("Ops")), TfToken("Scope"));
    // Definition order is the reverse of execution order (ScenePublication
    // fixture): Width first, the generator second.
    UsdPrim width = stage->DefinePrim(
        kDescription.AppendPath(SdfPath("Ops/width")), TfToken("UsdGenWidth"));
    width.CreateAttribute(TfToken("usdGen:width"), SdfValueTypeNames->Float)
        .Set(0.05f);
    width.CreateAttribute(TfToken("usdGen:replace"), SdfValueTypeNames->Bool)
        .Set(true);
    UsdPrim source = stage->DefinePrim(
        kDescription.AppendPath(SdfPath("Ops/source")),
        TfToken("UsdGenReferenceSource"));
    if (withTarget) {
        source.CreateRelationship(TfToken("usdGen:reference"))
            .SetTargets({kBaked});
    }
    return stage;
}

UsdGenGraphDesc BuildFromStage(UsdStageRefPtr const &stage)
{
    UsdGenGraphDescBuildOptions opts;
    opts.time = 0.0;
    return BuildGraphDescFromStage(stage, kDescription, opts);
}

UsdGenGraphDesc BuildFromHydra(UsdStageRefPtr const &stage)
{
    UsdGenGraphDescBuildOptions opts;
    opts.time = 0.0;
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices const sis = UsdImagingCreateSceneIndices(info);
    return BuildGraphDescFromHydra(*sis.finalSceneIndex, kDescription, opts);
}

UsdGenNodeDesc const *FindNode(UsdGenGraphDesc const &desc, TfToken const &type)
{
    for (UsdGenNodeDesc const &node : desc.nodes)
        if (node.type == type) return &node;
    return nullptr;
}

UsdGenCurveSetDesc const *FindCurveSet(UsdGenGraphDesc const &desc,
                                       SdfPath const &path)
{
    for (UsdGenCurveSetDesc const &set : desc.curveSets)
        if (set.path == path) return &set;
    return nullptr;
}

UsdGenAuthoredPlaneDesc const *FindAuthoredPlane(UsdGenCurveSetDesc const &set,
                                                 TfToken const &name)
{
    for (UsdGenAuthoredPlaneDesc const &plane : set.authoredPlanes)
        if (plane.name == name) return &plane;
    return nullptr;
}

void CheckOwnershipPlane(char const *which, UsdGenCurveSetDesc const &set,
                         TfToken const &name, VtIntArray const &values)
{
    UsdGenAuthoredPlaneDesc const *plane = FindAuthoredPlane(set, name);
    std::string const what = std::string(which) + ": " + name.GetString();
    Check(plane != nullptr &&
              plane->type == UsdGenAuthoredPlaneType::Int32 &&
              plane->domain == UsdGenAuthoredPlaneDomain::Primitive &&
              plane->arity == 1 && plane->intValues == values,
          what + " forwards as one uniform integer per curve");
}

void CheckBuilderRouting(char const *which, UsdGenGraphDesc const &desc)
{
    std::string const what(which);
    UsdGenNodeDesc const *source = FindNode(desc, TfToken("UsdGenReferenceSource"));
    Check(source != nullptr, what + ": builder emits a UsdGenReferenceSource node");
    if (!source) return;
    Check(source->references.size() == 1 && source->references.front() == kBaked,
          what + ": usdGen:reference routes to the node's reference lane");
    Check(source->curves.empty() && source->inputs.empty(),
          what + ": source takes no curve or geometry inputs");
    UsdGenCurveSetDesc const *set = FindCurveSet(desc, kBaked);
    Check(set != nullptr, what + ": target becomes a captured curve set");
    if (!set) return;
    Check(set->role == UsdGenLane::Reference,
          what + ": captured set rides the Reference lane");
    Check(set->points == BakedPoints(),
          what + ": captured points equal the baked points");
    Check(set->curveId == VtArray<uint64_t>{7, 9},
          what + ": captured ids equal the baked ids");
    CheckOwnershipPlane(which, *set, TfToken("tubeId"), VtIntArray{11, 12});
    CheckOwnershipPlane(which, *set, TfToken("regionId"), VtIntArray{101, 102});
    CheckOwnershipPlane(which, *set, TfToken("hierarchyLevel"), VtIntArray{1, 2});
}

void CheckNativeClumpImport(UsdGenGraphDesc const &desc)
{
    UsdGenCurveSetDesc const *set = FindCurveSet(desc, kBaked);
    Check(set != nullptr, "stage builder retains baked native Clump planes");
    if (!set) return;
    auto check = [&](char const *name, UsdGenAuthoredPlaneType type,
                     UsdGenAuthoredPlaneDomain domain, uint8_t arity) {
        UsdGenAuthoredPlaneDesc const *plane = FindAuthoredPlane(*set, TfToken(name));
        Check(plane && plane->type == type && plane->domain == domain &&
                  plane->arity == arity,
              std::string("stage builder preserves typed ") + name);
        return plane;
    };
    auto const *id = check("clumpId_0", UsdGenAuthoredPlaneType::Int32,
                           UsdGenAuthoredPlaneDomain::Primitive, 1);
    auto const *anchor = check("clumpCenter_0", UsdGenAuthoredPlaneType::Float32,
                               UsdGenAuthoredPlaneDomain::Primitive, 3);
    auto const *centerId = check("clumpCenterId_0", UsdGenAuthoredPlaneType::Int32,
                                 UsdGenAuthoredPlaneDomain::Primitive, 2);
    auto const *weight = check("clumpWeight_0", UsdGenAuthoredPlaneType::Float32,
                               UsdGenAuthoredPlaneDomain::Point, 1);
    Check(id && id->intValues == VtIntArray{0, 0},
          "Clump membership values survive Stage import");
    Check(anchor && anchor->floatValues == VtFloatArray{0, 0, 0, 0, 0, 0},
          "Clump rest anchors survive Stage import");
    Check(centerId && centerId->intValues ==
          VtIntArray{-1, 0x76543210, -1, 0x76543210},
          "int2 center ID words survive Stage import bit-exactly");
    Check(weight && weight->floatValues ==
          VtFloatArray{0, .2f, .6f, 0, .3f, .8f},
          "vertex cohesion values survive Stage import");
}

void CheckCook(UsdGenGraphDesc const &desc, bool checkNative = false)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        for (std::string const &e : compiled.errors)
            std::printf("compile error: %s\n", e.c_str());
    }
    Check(compiled.ok, "reference stack compiles");
    if (!compiled.ok) return;

    UsdGenScheduler scheduler(1);
    UsdGenEvalContext context;
    context.desc = &graph.Spec();
    UsdGenRunResult const run = scheduler.Run(graph, context, 1);
    Check(!run.diagnostics.HasErrors(), "reference stack executes");
    if (run.diagnostics.HasErrors()) return;

    UsdGenNodeId const sourceId =
        usdGenUsd::NodeIdForPath(graph, kDescription.AppendPath(SdfPath("Ops/source")));
    UsdGenNodeId const widthId =
        usdGenUsd::NodeIdForPath(graph, kDescription.AppendPath(SdfPath("Ops/width")));
    Check(sourceId != kUsdGenInvalidNode && widthId != kUsdGenInvalidNode,
          "both stack nodes compiled");
    if (sourceId == kUsdGenInvalidNode || widthId == kUsdGenInvalidNode) return;
    UsdGenCurveBuffer const source = NodeBuffer(graph, sourceId);
    UsdGenCurveBuffer const output = NodeBuffer(graph, widthId);
    usdGen::UsdGenVec3fArray const baked = BakedPoints();
    bool pointsVerbatim = source.totalCurves == 2 && source.totalCvs == 6;
    for (size_t i = 0; pointsVerbatim && i < baked.size(); ++i)
        pointsVerbatim = source.px[i] == baked[i][0] &&
                         source.py[i] == baked[i][1] &&
                         source.pz[i] == baked[i][2];
    Check(pointsVerbatim, "pulled points equal the baked points verbatim");
    Check(source.curveId == VtArray<uint64_t>{7, 9},
          "pulled ids equal the baked ids");
    if (checkNative) {
        auto find = [&](std::vector<UsdGenPlane> const &planes, char const *name) {
            for (UsdGenPlane const &plane : planes)
                if (plane.name == TfToken(name)) return &plane;
            return static_cast<UsdGenPlane const *>(nullptr);
        };
        UsdGenPlane const *centerId = find(source.extraCurve, "clumpCenterId_0");
        UsdGenPlane const *weight = find(source.extraCv, "clumpWeight_0");
        Check(centerId && centerId->type == TfToken("int") && centerId->arity == 2 &&
                  centerId->i == VtIntArray{-1, 0x76543210, -1, 0x76543210},
              "CurveSource preserves baked int2 center identity exactly");
        Check(weight && weight->type == TfToken("float") && weight->arity == 1 &&
                  weight->f == VtFloatArray{0, .2f, .6f, 0, .3f, .8f},
              "CurveSource preserves baked vertex cohesion");
    }
    // Epsilon, not ==: the width ramp LUT multiplies by ~1.0, not exactly 1.0.
    bool widthApplied = output.totalCvs == 6;
    for (size_t i = 0; widthApplied && i < output.width.size(); ++i)
        widthApplied = std::fabs(output.width[i] - 0.05f) <= 1e-6f;
    if (!widthApplied) {
        std::printf("width plane: size=%zu totalCvs=%u values=",
                    output.width.size(), output.totalCvs);
        for (size_t i = 0; i < output.width.size(); ++i)
            std::printf("%g ", double(output.width[i]));
        std::printf("\n");
    }
    Check(widthApplied, "downstream Width consumes the pulled geometry");
    bool restFollows = source.rest.size() == 6;
    for (size_t i = 0; restFollows && i < baked.size(); ++i)
        restFollows = source.rest[i] == baked[i];
    Check(restFollows, "pulled rest follows the baked rest");
}

}  // namespace

int main()
{

    UsdStageRefPtr stage = MakeStage(/*withTarget=*/true);
    UsdGenGraphDesc const fromStage = BuildFromStage(stage);
    UsdGenGraphDesc const fromHydra = BuildFromHydra(stage);
    CheckBuilderRouting("stage builder", fromStage);
    CheckBuilderRouting("Hydra builder", fromHydra);
    CheckNativeClumpImport(fromStage);
    Check(fromStage.validationErrors.empty() && fromHydra.validationErrors.empty(),
          "no builder validation errors");
    CheckCook(fromHydra);
    CheckCook(fromStage, /*checkNative=*/true);

    UsdGenGraphDesc const malformed = BuildFromStage(
        MakeStage(/*withTarget=*/true, /*malformedClump=*/true));
    Check(!malformed.validationErrors.empty(),
          "malformed baked Clump int2 metadata reports a builder error");
    UsdGenCompiler malformedCompiler;
    UsdGenGraph malformedGraph;
    Check(!usdGenUsd::Compile(malformedCompiler, malformed, &malformedGraph).ok,
          "malformed baked Clump metadata fails closed at compile");

    // Fail closed: no reference target, no silent empty stack.
    UsdGenGraphDesc const missing = BuildFromStage(MakeStage(/*withTarget=*/false));
    UsdGenNodeDesc const *source =
        FindNode(missing, TfToken("UsdGenReferenceSource"));
    Check(source != nullptr && source->references.empty(),
          "an unauthored reference leaves the lane empty");
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const rejected = usdGenUsd::Compile(compiler, missing, &graph);
    Check(!rejected.ok && !rejected.errors.empty() &&
              rejected.errors.front().find("0 resolved reference values") !=
                  std::string::npos,
          "an empty reference fails closed at compile");

    if (g_failures != 0) {
        std::printf("%d reference-source FAILURES\n", g_failures);
        return 1;
    }
    std::printf("reference-source contract holds\n");
    return 0;
}
