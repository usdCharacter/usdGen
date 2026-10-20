// UsdGenCollide operator contract and CPU cooks.
//
// Covers iterative push-out from collider meshes:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/
//       OutputPrimvars plus the Topology/Value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): identity at zero effect, the flexible push-out formula,
//       pushAmount scale, the iterations loop, the mask envelope, stiff
//       rigid motion (segment lengths preserved), multi-surface and
//       GeomSubset colliders, and a primitive-domain pushAmount expression;
//   (3) every edge fails closed (resolveType, iterations, finiteness,
//       missing/degenerate collider meshes);
//   (4) determinism (two cooks bit-identical) and CUDA-backend explicit
//       rejection (no capability-matrix row, like Clump).
#include "usdGen/compiler.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGen/limitSurface.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/base/js/json.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace usdGen;

static int failures = 0;
static void Check(bool v, char const *s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s); }
    else std::printf("ok: %s\n", s);
}
static expr::ValueShape FloatShape(uint32_t n = 1)
{
    return {expr::ScalarType::Float32, 1, n, 1, 1, false};
}

constexpr float kTol = 1e-4f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}

// Straight strands along +Y from the given roots. Mirrors the CurveSource
// fixture (surface-free, rebind never): valid skin bindings plus identity
// root frames, so rootT=(1,0,0), rootB=(0,1,0), rootN=(0,0,1).
static UsdGenCurveSetDesc StraightStrands(SdfPath const &path,
                                          std::vector<GfVec3f> const &roots,
                                          int cvs, float step)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    size_t const n = roots.size();
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * size_t(cvs));
    for (size_t c = 0; c < n; ++c)
        for (int i = 0; i < cvs; ++i)
            set.points[c * size_t(cvs) + size_t(i)] =
                roots[c] + GfVec3f(0.0f, float(i) * step, 0.0f);
    set.rest = set.points;
    set.widths.assign(n * size_t(cvs), 0.1f);
    set.curveId.resize(n);
    for (size_t c = 0; c < n; ++c) set.curveId[c] = uint64_t(c);
    set.skinPrim.resize(n);
    for (size_t c = 0; c < n; ++c) set.skinPrim[c] = int(c);
    set.skinPrimUv.assign(n, GfVec2f(0.2f, 0.3f));
    set.rootFrame.assign(n, GfMatrix4d(1.0));
    return set;
}

static UsdGenNodeDesc SourceNode(SdfPath const &path, SdfPath const &curves)
{
    UsdGenNodeDesc node;
    node.path = path;
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {curves};
    node.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                   {TfToken("useRest"), VtValue(true), false}};
    return node;
}

static UsdGenNodeDesc OpNode(char const *path, char const *type,
                             SdfPath const &input, int seed)
{
    UsdGenNodeDesc node;
    node.path = SdfPath(path);
    node.type = TfToken(type);
    node.seed = seed;
    node.inputs = {input};
    return node;
}

static UsdGenSurfaceDesc TriangleSurface(SdfPath const &path,
                                         GfVec3f a, GfVec3f b, GfVec3f c)
{
    UsdGenSurfaceDesc s;
    s.path = path;
    s.faceVertexCounts = {3};
    s.faceVertexIndices = {0, 1, 2};
    s.points = {a, b, c};
    s.restPoints = s.points;
    return s;
}

static UsdGenSurfaceDesc ClosedCube(SdfPath const &path)
{
    UsdGenSurfaceDesc s;
    s.path = path;
    s.points = {{0.3f, 0.12f, 0.3f}, {0.7f, 0.12f, 0.3f},
                {0.7f, 0.48f, 0.3f}, {0.3f, 0.48f, 0.3f},
                {0.3f, 0.12f, 0.7f}, {0.7f, 0.12f, 0.7f},
                {0.7f, 0.48f, 0.7f}, {0.3f, 0.48f, 0.7f}};
    s.restPoints = s.points;
    s.faceVertexCounts = {4, 4, 4, 4, 4, 4};
    s.faceVertexIndices = {0, 3, 2, 1, 4, 5, 6, 7,
                           0, 1, 5, 4, 3, 7, 6, 2,
                           0, 4, 7, 3, 1, 2, 6, 5};
    return s;
}

static void AddBinding(UsdGenGraphDesc &desc, SdfPath const &nodePath,
                       SdfPath const &exprPath, std::string const &source,
                       TfToken const &nativeType, expr::ValueShape const &shape,
                       TfToken const &destination, expr::Domain domain,
                       VtValue const &literal)
{
    desc.expressions.push_back(
        {exprPath, source, {{TfToken("result"), nativeType, shape}}});
    for (UsdGenNodeDesc &node : desc.nodes) {
        if (node.path != nodePath) continue;
        UsdGenExpressionBinding binding;
        binding.expression = exprPath;
        binding.output = TfToken("result");
        binding.nativeType = nativeType;
        binding.destination = destination;
        binding.destinationShape = shape;
        binding.domain = domain;
        binding.literal = literal;
        node.expressionBindings.push_back(binding);
    }
}

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
                 std::vector<std::string> *warnings = nullptr,
                 std::vector<std::string> *errors = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors)
            std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (warnings) *warnings = run.diagnostics.warnings;
    if (errors) *errors = run.diagnostics.errors;
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors)
            std::printf("  run: %s\n", e.c_str());
        return false;
    }
    *out = graph.Output();
    if (std::getenv("USDGEN_TEST_COLLIDE_FINGERPRINT")) {
        uint64_t hash = 14695981039346656037ull;
        for (auto const* plane : {&out->px, &out->py, &out->pz, &out->width}) {
            auto const* bytes =
                reinterpret_cast<unsigned char const*>(plane->cdata());
            for (size_t i=0; i<plane->size()*sizeof(float); ++i)
                hash = (hash ^ bytes[i]) * 1099511628211ull;
        }
        std::printf("fingerprint: %u %u %llu\n", out->totalCurves,
                    out->totalCvs, static_cast<unsigned long long>(hash));
    }
    return true;
}

static bool CookSourceOnly(UsdGenGraphDesc const &desc, SdfPath const &source,
                           UsdGenCurveBuffer *out)
{
    UsdGenGraphDesc plain = desc;
    plain.nodes.pop_back();
    plain.expressions.clear();
    plain.terminal = source;
    return Cook(plain, out);
}

// --- fixtures ----------------------------------------------------------
// Main fixture: two 6-CV strands at x = 0 and x = 3 under a large triangle
// at z = -0.25, so every CV projects inside it at distance 0.25. With
// offset 0.5 the penetration is 0.25 straight up (+Z).
static UsdGenGraphDesc CollideDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/collide");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/collide/hair"), roots, 6, 0.25f));
    desc.surfaces.push_back(TriangleSurface(
        SdfPath("/collide/collider"),
        GfVec3f(-10, -10, -0.25f), GfVec3f(10, -10, -0.25f),
        GfVec3f(0, 10, -0.25f)));
    desc.nodes.push_back(
        SourceNode(SdfPath("/collide/source"), SdfPath("/collide/hair")));
    UsdGenNodeDesc collide = OpNode("/collide/op", "UsdGenCollide",
                                    SdfPath("/collide/source"), 7);
    collide.params = params;
    collide.surfaces = {SdfPath("/collide/collider")};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/collide/op");
    return desc;
}

// Stiff fixture: a small triangle under the strand tips only. Strand 0 CVs
// (0, y, 0) hit vertex C = (0, 1, -0.1) at y = 0.75/1 (dist 0.269/0.1) and
// the interior at y = 1.25 (dist 0.1); y = 0/0.25/0.5 stay clean (nearest
// C at dist >= 0.51 > offset 0.5), so the hinge is CV 3. Strand 1 (x = 3)
// is ~2.8 away and never penetrates.
static UsdGenGraphDesc StiffDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc = CollideDesc(params);
    desc.surfaces.front() = TriangleSurface(
        SdfPath("/collide/collider"),
        GfVec3f(-2, 3, -0.1f), GfVec3f(2, 3, -0.1f), GfVec3f(0, 1, -0.1f));
    return desc;
}

static UsdGenGraphDesc CubeDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc = CollideDesc(params);
    desc.curveSets.front() = StraightStrands(
        SdfPath("/collide/hair"),
        {GfVec3f(0.5f, 0, 0.5f), GfVec3f(2, 0, 2)}, 6, 0.1f);
    desc.surfaces.front() = ClosedCube(SdfPath("/collide/collider"));
    return desc;
}

static bool SamePoints(UsdGenCurveBuffer const &a,
                       UsdGenCurveBuffer const &b)
{
    return a.totalCvs == b.totalCvs && a.px == b.px &&
           a.py == b.py && a.pz == b.pz;
}

static bool NearPoints(UsdGenCurveBuffer const &a,
                       UsdGenCurveBuffer const &b, float tolerance = 2e-5f)
{
    if (a.totalCvs != b.totalCvs) return false;
    for (uint32_t cv = 0; cv < a.totalCvs; ++cv)
        if (!Near(a.px[cv], b.px[cv], tolerance) ||
            !Near(a.py[cv], b.py[cv], tolerance) ||
            !Near(a.pz[cv], b.pz[cv], tolerance))
            return false;
    return true;
}

static void CheckColliderWorldTransforms()
{
    auto parameters = std::vector<UsdGenParamValue>{
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(2), false}};
    UsdGenGraphDesc const base = CubeDesc(parameters);
    UsdGenCurveBuffer plain;
    Check(CookSourceOnly(base, SdfPath("/collide/source"), &plain),
          "Transformed Collide source cooks");

    // Local points sit near the hair. A world-only translation clears it;
    // this fails if the collision kernel silently treats local points as
    // groom-space points (the editable xformOp case).
    UsdGenGraphDesc raised = base;
    raised.surfaces.front().worldMatrix.SetTranslate(GfVec3d(0, 2, 0));
    UsdGenCurveBuffer raisedOut;
    Check(Cook(raised, &raisedOut), "Translated collider cooks");
    Check(SamePoints(raisedOut, plain),
          "World-raised collider leaves local-origin groom unchanged");

    UsdGenCompiler compiler;
    UsdGenGraph liveGraph;
    Check(compiler.Compile(raised, &liveGraph).ok,
          "Animated collider graph compiles");
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext liveContext;
    liveContext.desc = &liveGraph.Desc();
    Check(!scheduler.Run(liveGraph, liveContext, 1).diagnostics.HasErrors(),
          "Raised collider live cook runs");
    VtFloatArray const raisedPx(liveGraph.Output().px.begin(),
                               liveGraph.Output().px.end());
    raised.surfaces.front().worldMatrix = GfMatrix4d(1.0);
    Check(compiler.Recompile(raised, &liveGraph).ok,
          "Animated collider world transform recompiles");
    liveContext.desc = &liveGraph.Desc();
    Check(!scheduler.Run(liveGraph, liveContext, 2).diagnostics.HasErrors(),
          "Moved collider live recook runs");
    Check(liveGraph.Output().px != raisedPx ||
          !SamePoints(liveGraph.Output(), plain),
          "Animated collider matrix invalidates capture and changes points");

    // An identical parent transform on description and collider cancels.
    UsdGenGraphDesc sharedParent = base;
    sharedParent.xformMatrix.SetTranslate(GfVec3d(4, 5, 6));
    sharedParent.surfaces.front().worldMatrix = sharedParent.xformMatrix;
    UsdGenCurveBuffer baseOut, parentOut;
    Check(Cook(base, &baseOut) && Cook(sharedParent, &parentOut),
          "Shared-parent collider and description cook");
    Check(SamePoints(liveGraph.Output(), baseOut),
          "Animated collider recook matches fresh local collider cook");
    Check(SamePoints(baseOut, parentOut),
          "Shared parent transform cancels from collider-to-groom frame");
    GfMatrix4d scale(1.0), rotation(1.0), translationParent(1.0);
    scale.SetScale(GfVec3d(2.0, 0.7, 1.3));
    rotation.SetRotate(GfRotation(GfVec3d(0, 0, 1), 37.0));
    translationParent.SetTranslate(GfVec3d(4, 5, 6));
    GfMatrix4d const complexParent = scale * rotation * translationParent;
    sharedParent.xformMatrix = complexParent;
    sharedParent.surfaces.front().worldMatrix = complexParent;
    Check(Cook(sharedParent, &parentOut),
          "Shared scaled and rotated parent collider cooks");
    Check(NearPoints(baseOut, parentOut),
          "Shared scaled and rotated parent cancels from collider frame");

    // The description itself can move while the collider stays fixed.
    // Its inverse changes the relative collider frame and must recapture.
    UsdGenGraph descLiveGraph;
    Check(compiler.Compile(base, &descLiveGraph).ok,
          "Description-motion graph compiles");
    liveContext.desc = &descLiveGraph.Desc();
    Check(!scheduler.Run(descLiveGraph, liveContext, 1).diagnostics.HasErrors(),
          "Description-motion initial cook runs");
    UsdGenGraphDesc movedDescription = base;
    movedDescription.xformMatrix.SetTranslate(GfVec3d(0, 2, 0));
    Check(compiler.Recompile(movedDescription, &descLiveGraph).ok,
          "Description world transform recompiles");
    liveContext.desc = &descLiveGraph.Desc();
    Check(!scheduler.Run(descLiveGraph, liveContext, 2).diagnostics.HasErrors(),
          "Moved-description live recook runs");
    Check(SamePoints(descLiveGraph.Output(), plain),
          "Description-only transform invalidates collider capture");

    // Nonuniform scale and reflection change the triangle geometry. An
    // explicit transformed-points oracle must agree with the authored
    // matrix, including the closed-solid winding and derived face normals.
    UsdGenGraphDesc scaled = base;
    GfMatrix4d relative(1.0);
    relative.SetScale(GfVec3d(-1.5, 0.7, 2.0));
    GfMatrix4d translation(1.0);
    translation.SetTranslate(GfVec3d(1.2, 0.08, -0.35));
    relative *= translation;
    scaled.xformMatrix.SetTranslate(GfVec3d(4, 5, 6));
    scaled.surfaces.front().worldMatrix = relative * scaled.xformMatrix;
    UsdGenGraphDesc oracle = base;
    for (GfVec3f &point : oracle.surfaces.front().points)
        point = GfVec3f(relative.Transform(GfVec3d(point)));
    UsdGenCurveBuffer scaledOut, oracleOut;
    Check(Cook(scaled, &scaledOut) && Cook(oracle, &oracleOut),
          "Scaled reflected collider and baked-point oracle cook");
    bool same = scaledOut.totalCvs == oracleOut.totalCvs;
    for (uint32_t cv = 0; same && cv < scaledOut.totalCvs; ++cv)
        same = Near(scaledOut.px[cv], oracleOut.px[cv], 2e-5f) &&
               Near(scaledOut.py[cv], oracleOut.py[cv], 2e-5f) &&
               Near(scaledOut.pz[cv], oracleOut.pz[cv], 2e-5f);
    Check(same, "Nonuniform reflected xform matches transformed geometry oracle");
}

static float SegLen(UsdGenCurveBuffer const &b, size_t a, size_t c);

static void CheckClosedCubeCook()
{
    UsdGenGraphDesc desc = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out) &&
          CookSourceOnly(desc, SdfPath("/collide/source"), &plain),
          "Collide closed-cube cook runs");
    if (out.totalCvs != 12 || plain.totalCvs != 12) return;
    Check(Near(out.py[0], plain.py[0]) && Near(out.py[1], plain.py[1]),
          "Collide at zero offset anchors exterior roots and clean CVs");
    Check(Near(out.py[2], 0.12f) && Near(out.py[4], 0.12f),
          "Collide at zero offset ejects interior CVs through entry face");
    Check(Near(out.py[5], 0.12f),
          "Collide keeps exterior tip from spanning closed cube interior");
    bool farExact = true;
    for (size_t i = 6; i < 12; ++i)
        farExact &= out.px[i] == plain.px[i] &&
                    out.py[i] == plain.py[i] && out.pz[i] == plain.pz[i];
    Check(farExact, "Collide leaves exterior strand bit-for-bit unchanged");

    UsdGenGraphDesc reversed = desc;
    for (size_t f = 0; f < 6; ++f) {
        auto begin = reversed.surfaces.front().faceVertexIndices.begin() +
                     std::ptrdiff_t(4 * f);
        std::reverse(begin, begin + 4);
    }
    UsdGenCurveBuffer reversedOut;
    Check(Cook(reversed, &reversedOut), "Collide reversed-winding cube cooks");
    if (reversedOut.totalCvs == 12)
        Check(Near(reversedOut.py[2], 0.12f) &&
              Near(reversedOut.py[4], 0.12f),
              "Collide closed-volume test is winding independent");

    // Cut-flex uses the actual current segment entry rather than projecting
    // an outside endpoint to a clamped source facet.  A threshold above every
    // possible depth selects that collision strategy without shortening.
    auto cutFlex=desc;
    cutFlex.nodes.back().params.push_back({TfToken("deepPenetrationMode"),
        VtValue(TfToken("cutThenCollide")),false});
    cutFlex.nodes.back().params.push_back({TfToken("cutDepthThreshold"),
        VtValue(10.0f),false});
    cutFlex.nodes.back().params.push_back({TfToken("cutBlendDepth"),
        VtValue(0.02f),false});
    cutFlex.nodes.back().params.push_back({TfToken("iterations"),
        VtValue(32),false});
    UsdGenCurveBuffer cutFlexOut;
    Check(Cook(cutFlex,&cutFlexOut),
          "Collide cut-flex polygon entry cooks without shortening");
    if (cutFlexOut.totalCvs==12) {
        Check(Near(cutFlexOut.py[2],0.12f) &&
              Near(cutFlexOut.py[4],0.12f),
              "Collide cut-flex uses the oriented polygon entry normal");
        Check(Near(cutFlexOut.py[5],0.12f),
              "Collide cut-flex grazing exit tail stays on the entry side");
    }
    auto cutFlexReversed=cutFlex;
    cutFlexReversed.surfaces.front()=reversed.surfaces.front();
    UsdGenCurveBuffer cutFlexReversedOut;
    Check(Cook(cutFlexReversed,&cutFlexReversedOut),
          "Collide cut-flex reversed polygon entry cooks");
    if (cutFlexReversedOut.totalCvs==12) {
        Check(Near(cutFlexReversedOut.py[2],0.12f),
              "Collide cut-flex polygon entry is winding independent");
    }

    auto cutFlexPartial=cutFlex;
    for (UsdGenParamValue &p : cutFlexPartial.nodes.back().params)
        if (p.name==TfToken("pushAmount")) p.value=VtValue(0.5f);
        else if (p.name==TfToken("iterations")) p.value=VtValue(1);
    UsdGenCurveBuffer cutFlexPartialOut;
    Check(Cook(cutFlexPartial,&cutFlexPartialOut),
          "Collide cut-flex partial response cooks");
    if (cutFlexPartialOut.totalCvs==12)
        Check(cutFlexPartialOut.py[2]<plain.py[2] &&
              cutFlexPartialOut.py[2]>cutFlexOut.py[2],
              "Collide cut-flex preserves authored partial response");
    UsdGenGraphDesc reversedFace = reversed;
    reversedFace.curveSets.front() = StraightStrands(
        SdfPath("/collide/hair"), {GfVec3f(0.5f, 0.12f, 0.5f)}, 2, 0);
    reversedFace.nodes.back().params.front().value = VtValue(0.02f);
    UsdGenCurveBuffer reversedFaceOut;
    Check(Cook(reversedFace, &reversedFaceOut),
          "Collide reversed-winding face offset cook runs");
    if (reversedFaceOut.totalCvs == 2) {
        if (!Near(reversedFaceOut.py[0], 0.10f))
            std::printf("  reversed face y=%.9f\n", reversedFaceOut.py[0]);
        Check(Near(reversedFaceOut.py[0], 0.10f),
              "Collide face offset points out of reversed solid");
    }

    UsdGenGraphDesc withSkin = desc;
    withSkin.surfaces.insert(withSkin.surfaces.begin(), TriangleSurface(
        SdfPath("/collide/skin"), GfVec3f(-2, 0.18f, -2),
        GfVec3f(4, 0.18f, -2), GfVec3f(-2, 0.18f, 4)));
    withSkin.nodes.back().surfaces = {SdfPath("/collide/skin"),
                                       SdfPath("/collide/collider")};
    UsdGenCurveBuffer nearSkin;
    Check(Cook(withSkin, &nearSkin), "Collide cube with nearer open skin cooks");
    if (nearSkin.totalCvs == 12)
        Check(Near(nearSkin.py[2], 0.12f),
              "Collide ejects interior cube CV despite nearer open skin");

    desc.nodes.back().params.front().value = VtValue(0.02f);
    UsdGenCurveBuffer gap;
    Check(Cook(desc, &gap), "Collide closed-cube offset cook runs");
    if (gap.totalCvs == 12)
        Check(Near(gap.py[2], 0.10f) && Near(gap.py[4], 0.10f),
              "Collide places interior CVs offset outside entered face");

    // A single-face GeomSubset has open boundaries, even though its parent
    // geometry is a closed cube. It must keep the established shell behavior.
    desc.nodes.back().params.front().value = VtValue(0.0f);
    desc.surfaces.front().subsetFaces = {2};
    UsdGenCurveBuffer open;
    Check(Cook(desc, &open), "Collide open cube-face subset cooks");
    if (open.totalCvs == 12)
        Check(open.py[2] == plain.py[2] && open.py[4] == plain.py[4],
              "Collide open subset does not classify enclosed cube interior");

    UsdGenGraphDesc sparse = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false}});
    sparse.curveSets.front() = StraightStrands(
        SdfPath("/collide/hair"), {GfVec3f(0.5f, 0, 0.5f)}, 2, 0.4f);
    UsdGenCurveBuffer sparseOut;
    Check(Cook(sparse, &sparseOut), "Collide sparse entry cook runs");
    if (sparseOut.totalCvs == 2)
        Check(Near(sparseOut.py[0], 0.0f) &&
              Near(sparseOut.py[1], 0.12f),
              "Collide enters bottom face even when first interior CV is near top");
    UsdGenGraphDesc boundaryStart = sparse;
    boundaryStart.curveSets.front() = StraightStrands(
        SdfPath("/collide/hair"),
        {GfVec3f(0.5f, 0.12f, 0.5f)}, 2, 0.28f);
    UsdGenCurveBuffer boundaryOut;
    Check(Cook(boundaryStart, &boundaryOut),
          "Collide boundary-start sparse strand cooks");
    if (boundaryOut.totalCvs == 2)
        Check(Near(boundaryOut.py[0], 0.12f) &&
              Near(boundaryOut.py[1], 0.12f),
              "Collide boundary-start segment selects entry face");
    UsdGenGraphDesc cutBoundary=boundaryStart;
    cutBoundary.nodes.back().params.push_back(
        {TfToken("deepPenetrationMode"),
         VtValue(TfToken("cutThenCollide")),false});
    cutBoundary.nodes.back().params.push_back(
        {TfToken("cutDepthThreshold"),VtValue(10.0f),false});
    cutBoundary.nodes.back().params.push_back(
        {TfToken("cutBlendDepth"),VtValue(0.02f),false});
    cutBoundary.nodes.back().params.push_back(
        {TfToken("iterations"),VtValue(32),false});
    UsdGenCurveBuffer cutBoundaryOut;
    Check(Cook(cutBoundary,&cutBoundaryOut),
          "Collide cut-flex boundary-start strand cooks");
    if (cutBoundaryOut.totalCvs==2)
        Check(Near(cutBoundaryOut.py[0],0.12f) &&
              Near(cutBoundaryOut.py[1],0.12f),
              "Collide cut-flex boundary contact reaches an exact fixed point");
    UsdGenGraphDesc stiffBoundary = boundaryStart;
    stiffBoundary.nodes.back().params.push_back(
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false});
    UsdGenCurveBuffer stiffBoundaryOut;
    Check(Cook(stiffBoundary, &stiffBoundaryOut),
          "Collide stiff boundary-start strand cooks");
    if (stiffBoundaryOut.totalCvs == 2)
        Check(Near(stiffBoundaryOut.py[0], 0.12f) &&
              Near(SegLen(stiffBoundaryOut, 0, 1), 0.28f, 3e-4f) &&
              stiffBoundaryOut.py[1] <= 0.1201f,
              "Collide stiff boundary-start swivel preserves length and root");

    UsdGenGraphDesc tiny = sparse;
    for (GfVec3f &point : tiny.surfaces.front().points) point *= 1e-5f;
    tiny.surfaces.front().restPoints = tiny.surfaces.front().points;
    for (GfVec3f &point : tiny.curveSets.front().points) point *= 1e-5f;
    tiny.curveSets.front().rest = tiny.curveSets.front().points;
    UsdGenCurveBuffer tinyOut;
    Check(Cook(tiny, &tinyOut), "Collide small-scale closed cube cooks");
    if (tinyOut.totalCvs == 2)
        Check(Near(tinyOut.py[1], 0.12e-5f, 1e-9f),
              "Collide small-scale segment selects entry face");

    UsdGenGraphDesc crossing = sparse;
    auto &hair = crossing.curveSets.front();
    hair.points = {GfVec3f(0.1f, 0.2f, 0.5f),
                   GfVec3f(1.5f, 0.2f, 0.5f)};
    hair.rest = hair.points;
    UsdGenCurveBuffer crossingOut;
    Check(Cook(crossing, &crossingOut),
          "Collide outside-to-outside segment crossing cooks");
    if (crossingOut.totalCvs == 2)
        Check(Near(crossingOut.px[0], 0.1f) &&
              Near(crossingOut.px[1], 0.3f),
              "Collide catches off-center segment crossing with exterior endpoints");
    auto cutCrossing=crossing;
    cutCrossing.nodes.back().params.push_back(
        {TfToken("deepPenetrationMode"),
         VtValue(TfToken("cutThenCollide")),false});
    cutCrossing.nodes.back().params.push_back(
        {TfToken("cutDepthThreshold"),VtValue(10.0f),false});
    cutCrossing.nodes.back().params.push_back(
        {TfToken("cutBlendDepth"),VtValue(0.02f),false});
    cutCrossing.nodes.back().params.push_back(
        {TfToken("iterations"),VtValue(32),false});
    auto reversedCutCrossing=cutCrossing;
    for (size_t f=0;f<6;++f) {
        auto begin=reversedCutCrossing.surfaces.front().faceVertexIndices.begin()+
            std::ptrdiff_t(4*f);
        std::reverse(begin,begin+4);
    }
    UsdGenCurveBuffer cutCrossingOut,reversedCutCrossingOut;
    Check(Cook(cutCrossing,&cutCrossingOut) &&
          Cook(reversedCutCrossing,&reversedCutCrossingOut),
          "Collide cut-flex outside crossing cooks in both windings");
    if (cutCrossingOut.totalCvs==2 && reversedCutCrossingOut.totalCvs==2)
    {
        if (!(cutCrossingOut.px[1]<=0.3001f &&
              reversedCutCrossingOut.px[1]<=0.3001f))
            std::printf("  cut crossing x=%.9f reversed=%.9f\n",
                cutCrossingOut.px[1],reversedCutCrossingOut.px[1]);
        Check(cutCrossingOut.px[1]<=0.3001f &&
              reversedCutCrossingOut.px[1]<=0.3001f,
              "Collide cut-flex outside crossing uses geometric outward entry");
    }

    // Bilinear exercises refined limit entry normals without a curved cage
    // changing this outside-to-outside witness's contact coordinate.
    auto limitCutCrossing=cutCrossing;
    auto reversedLimitCutCrossing=reversedCutCrossing;
    limitCutCrossing.surfaces.front().subdivisionScheme=TfToken("bilinear");
    reversedLimitCutCrossing.surfaces.front().subdivisionScheme=TfToken("bilinear");
    UsdGenCurveBuffer limitCrossingOut,reversedLimitCrossingOut;
    Check(Cook(limitCutCrossing,&limitCrossingOut) &&
          Cook(reversedLimitCutCrossing,&reversedLimitCrossingOut),
          "Collide cut-flex limit outside crossing cooks in both windings");
    if (limitCrossingOut.totalCvs==2 && reversedLimitCrossingOut.totalCvs==2)
        Check(Near(limitCrossingOut.px[1],0.3f,1e-4f) &&
              Near(reversedLimitCrossingOut.px[1],0.3f,1e-4f),
              "Collide cut-flex refined limit entry is winding independent");
    UsdGenGraphDesc twoSolids = sparse;
    UsdGenSurfaceDesc second = ClosedCube(SdfPath("/collide/second"));
    for (GfVec3f &point : second.points) point[0] += 1.5f;
    second.restPoints = second.points;
    twoSolids.surfaces.push_back(second);
    twoSolids.nodes.back().surfaces.push_back(SdfPath("/collide/second"));
    auto &four = twoSolids.curveSets.front();
    four = StraightStrands(SdfPath("/collide/hair"),
                           {GfVec3f(0.5f, 0, 0.5f)}, 4, 0.1f);
    four.points = {GfVec3f(0.5f, 0, 0.5f),
                   GfVec3f(0.5f, 0.2f, 0.5f),
                   GfVec3f(1.2f, 0.2f, 0.5f),
                   GfVec3f(2.0f, 0.2f, 0.5f)};
    four.rest = four.points;
    UsdGenCurveBuffer twoOut;
    Check(Cook(twoSolids, &twoOut), "Collide two closed solids cook");
    if (twoOut.totalCvs == 4)
        Check(Near(twoOut.px[3], 1.8f) && Near(twoOut.py[3], 0.2f),
              "Collide switches entry anchor to second closed solid");
    auto cutTwo=twoSolids;
    cutTwo.nodes.back().params.push_back({TfToken("deepPenetrationMode"),
        VtValue(TfToken("cutThenCollide")),false});
    cutTwo.nodes.back().params.push_back({TfToken("cutDepthThreshold"),
        VtValue(10.0f),false});
    cutTwo.nodes.back().params.push_back({TfToken("cutBlendDepth"),
        VtValue(0.02f),false});
    cutTwo.nodes.back().params.push_back({TfToken("iterations"),
        VtValue(32),false});
    UsdGenCurveBuffer cutTwoOut;
    Check(Cook(cutTwo,&cutTwoOut),
          "Collide cut-flex resolves two closed solids");
    if (cutTwoOut.totalCvs==4)
    {
        if (!(cutTwoOut.px[3]<=1.8001f && Near(cutTwoOut.py[1],0.12f)))
            std::printf("  cut two cv3=(%.9f,%.9f,%.9f)\n",
                cutTwoOut.px[3],cutTwoOut.py[3],cutTwoOut.pz[3]);
        Check(cutTwoOut.px[3]<=1.8001f && Near(cutTwoOut.py[1],0.12f),
              "Collide cut-flex iterates across two closed collider entry sides");
    }

    auto cutTwoFloor=cutTwo;
    auto floor=ClosedCube(SdfPath("/collide/floor"));
    for (auto &point:floor.points) {
        point[0]=(point[0]-0.3f)*10.0f-1.0f;
        point[1]=(point[1]-0.12f)-0.41f;
    }
    floor.restPoints=floor.points;
    cutTwoFloor.surfaces.push_back(floor);
    cutTwoFloor.nodes.back().surfaces.push_back(floor.path);
    auto &floorHair=cutTwoFloor.curveSets.front();
    floorHair.points.push_back(GfVec3f(2.0f,-0.2f,0.5f));
    floorHair.rest=floorHair.points;
    floorHair.curveVertexCounts={5}; floorHair.widths.push_back(0.1f);
    UsdGenCurveBuffer floorOut,floorRepeat;
    Check(Cook(cutTwoFloor,&floorOut) && Cook(cutTwoFloor,&floorRepeat),
          "Collide cut-flex two solids and floor cook repeatedly");
    if (floorOut.totalCvs==5)
        Check(Near(floorOut.py[1],0.12f) && floorOut.px[3]<=1.8001f &&
              floorOut.py[4]>=-0.0501f && SamePoints(floorOut,floorRepeat),
              "Collide cut-flex clears two entry sides and floor deterministically");
    UsdGenGraphDesc stiffCube = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false}});
    UsdGenCurveBuffer rigid;
    Check(Cook(stiffCube, &rigid), "Collide stiff closed-cube cook runs");
    UsdGenCurveBuffer rigidAgain;
    Check(Cook(stiffCube, &rigidAgain),
          "Collide stiff closed-cube repeat cook runs");
    if (rigid.totalCvs == 12 && rigidAgain.totalCvs == 12)
        Check(rigid.px == rigidAgain.px && rigid.py == rigidAgain.py &&
              rigid.pz == rigidAgain.pz,
              "Collide stiff closed-cube cook is deterministic");
    if (rigid.totalCvs == 12) {
        bool lengths = true, exterior = true;
        for (size_t i = 1; i < 6; ++i) {
            lengths &= Near(SegLen(rigid, i - 1, i), 0.1f, 3e-4f);
            exterior &= rigid.py[i] <= 0.1201f;
        }
        Check(Near(rigid.px[0], 0.5f) && Near(rigid.py[0], 0.0f) &&
              Near(rigid.pz[0], 0.5f),
              "Collide stiff closed cube keeps exterior root fixed");
        Check(lengths, "Collide stiff closed cube preserves segment lengths");
        bool const fan = exterior &&
              (rigid.px[5] > 0.71f || rigid.px[5] < 0.29f ||
               rigid.pz[5] > 0.71f || rigid.pz[5] < 0.29f);
        if (!fan)
            for (size_t i = 0; i < 6; ++i)
                std::printf("  stiff cube cv%zu=(%.6f, %.6f, %.6f)\n",
                            i, rigid.px[i], rigid.py[i], rigid.pz[i]);
        Check(fan,
              "Collide stiff swivels strand under cube and fans tip outward");
    }
    UsdGenGraphDesc separated = stiffCube;
    separated.curveSets.front() = StraightStrands(
        SdfPath("/collide/hair"), {GfVec3f(2, 0, 2)}, 6, 0.1f);
    separated.nodes.back().params.push_back(
        {TfToken("iterations"), VtValue(1), false});
    UsdGenCurveBuffer once, repeated;
    Check(Cook(separated, &once), "Collide separated stiff one sweep cooks");
    separated.nodes.back().params.back().value = VtValue(32);
    Check(Cook(separated, &repeated),
          "Collide separated stiff 32 sweeps cook");
    if (once.totalCvs == 6 && repeated.totalCvs == 6)
        Check(once.px == repeated.px && once.py == repeated.py &&
              once.pz == repeated.pz,
              "Collide no-contact early exit is bit-identical at 1/32 sweeps");

    UsdGenGraphDesc settled = stiffCube;
    settled.nodes.back().params.push_back(
        {TfToken("iterations"), VtValue(8), false});
    UsdGenCurveBuffer settledEight, settledThirtyTwo;
    Check(Cook(settled, &settledEight),
          "Collide settled stiff eight sweeps cook");
    settled.nodes.back().params.back().value = VtValue(32);
    Check(Cook(settled, &settledThirtyTwo),
          "Collide settled stiff 32 sweeps cook");
    if (settledEight.totalCvs == 12 && settledThirtyTwo.totalCvs == 12)
        Check(settledEight.px == settledThirtyTwo.px &&
              settledEight.py == settledThirtyTwo.py &&
              settledEight.pz == settledThirtyTwo.pz,
              "Collide converged early exit is bit-identical at 8/32 sweeps");
    UsdGenGraphDesc stiffCrossing = crossing;
    stiffCrossing.nodes.back().params.push_back(
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false});
    UsdGenCurveBuffer stiffThrough;
    Check(Cook(stiffCrossing, &stiffThrough),
          "Collide stiff outside-to-outside segment cook runs");
    if (stiffThrough.totalCvs == 2) {
        bool const clear = Near(stiffThrough.px[0], 0.1f) &&
            Near(SegLen(stiffThrough, 0, 1), 1.4f, 3e-4f) &&
            stiffThrough.px[1] < 0.3f;
        if (!clear)
            std::printf("  stiff crossing root=(%.6f,%.6f,%.6f) "
                        "tip=(%.6f,%.6f,%.6f) len=%.6f\n",
                        stiffThrough.px[0], stiffThrough.py[0],
                        stiffThrough.pz[0], stiffThrough.px[1],
                        stiffThrough.py[1], stiffThrough.pz[1],
                        SegLen(stiffThrough, 0, 1));
        Check(clear,
              "Collide stiff catches whole-segment crossing without interior CV");
    }
    UsdGenGraphDesc nearSide = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(6), false},
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false}});
    auto &sideHair = nearSide.curveSets.front();
    sideHair = StraightStrands(SdfPath("/collide/hair"),
                               {GfVec3f(0.28f, 0, 0.5f)}, 6, 0.1f);
    sideHair.points = {{0.28f, 0, 0.5f}, {0.285f, 0.1f, 0.5f},
                       {0.31f, 0.2f, 0.5f}, {0.335f, 0.3f, 0.5f},
                       {0.36f, 0.4f, 0.5f}, {0.385f, 0.5f, 0.5f}};
    sideHair.rest = sideHair.points;
    UsdGenCurveBuffer sideOut;
    Check(Cook(nearSide, &sideOut), "Collide stiff side-entry cook runs");
    if (sideOut.totalCvs == 6) {
        bool lengths = true, aboveScalp = true, outsideSide = true;
        for (size_t i = 0; i < 6; ++i) {
            aboveScalp &= sideOut.py[i] >= -1e-4f;
            if (i >= 2 && sideOut.py[i] > 0.12f)
                outsideSide &= sideOut.px[i] <= 0.3001f;
            if (i > 0) {
                GfVec3f const d = sideHair.points[i] - sideHair.points[i - 1];
                lengths &= Near(SegLen(sideOut, i - 1, i),
                                d.GetLength(), 4e-4f);
            }
        }
        Check(Near(sideOut.px[0], 0.28f) && Near(sideOut.py[0], 0.0f) &&
              lengths && aboveScalp && outsideSide,
              "Collide stiff side-entry keeps root and lengths without inversion");
    }

    // A strand grazing a cube corner can alternate just inside/outside the
    // side wall between samples. Its corrected segments must not splice a
    // bottom-face CV directly to a side-face CV through the solid.
    UsdGenGraphDesc corner = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(6), false}});
    auto &box = corner.surfaces.front();
    for (GfVec3f &point : box.points) {
        point[0] = point[0] < 0.5f ? 0.29f : 0.71f;
        point[1] = point[1] < 0.3f ? 0.04f : 0.40f;
        point[2] = point[2] < 0.5f ? 0.29f : 0.71f;
    }
    box.restPoints = box.points;
    auto &strand = corner.curveSets.front();
    strand = StraightStrands(SdfPath("/collide/hair"),
                              {GfVec3f(0.287946f, 0, 0.296759f)}, 9, 0);
    strand.points = {
        {0.287946f, 0, 0.296759f},
        {0.291095f, 0.019052f, 0.296402f},
        {0.294530f, 0.038038f, 0.295527f},
        {0.298302f, 0.056922f, 0.294045f},
        {0.302462f, 0.075655f, 0.291861f},
        {0.307061f, 0.094175f, 0.288877f},
        {0.309055f, 0.113350f, 0.290037f},
        {0.312208f, 0.132390f, 0.289269f},
        {0.315197f, 0.151463f, 0.288707f}};
    strand.rest = strand.points;
    UsdGenCurveBuffer cornerOut;
    Check(Cook(corner, &cornerOut), "Collide corner-grazing strand cooks");
    if (cornerOut.totalCvs == 9) {
        auto crosses = [&](size_t a, size_t b) {
            float enter = 0.0f, leave = 1.0f;
            for (int axis = 0; axis < 3; ++axis) {
                float const lo = axis == 1 ? 0.0401f : 0.2901f;
                float const hi = axis == 1 ? 0.3999f : 0.7099f;
                float const av = axis == 0 ? cornerOut.px[a] :
                                 axis == 1 ? cornerOut.py[a] : cornerOut.pz[a];
                float const bv = axis == 0 ? cornerOut.px[b] :
                                 axis == 1 ? cornerOut.py[b] : cornerOut.pz[b];
                float const delta = bv - av;
                if (std::abs(delta) < 1e-12f) {
                    if (!(lo < av && av < hi)) return false;
                    continue;
                }
                float const t0 = (lo - av) / delta;
                float const t1 = (hi - av) / delta;
                enter = std::max(enter, std::min(t0, t1));
                leave = std::min(leave, std::max(t0, t1));
                if (leave <= enter) return false;
            }
            return leave - enter > 1e-5f;
        };
        bool clear = true;
        for (size_t i = 0; i + 1 < 9; ++i)
            clear &= !crosses(i, i + 1);
        Check(clear, "Collide corner-grazing segments avoid cube interior");
    }
}

static float SegLen(UsdGenCurveBuffer const &b, size_t a, size_t c)
{
    float const dx = b.px[c] - b.px[a];
    float const dy = b.py[c] - b.py[a];
    float const dz = b.pz[c] - b.pz[a];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// An icosphere gives adjacent entry/exit planes with a grazing chord. A
// closest-point swivel can alternate between an interior endpoint and an
// exterior endpoint whose connecting segment still crosses the solid.
static UsdGenSurfaceDesc ClosedIcosphere(SdfPath const &path)
{
    UsdGenSurfaceDesc s;
    s.path = path;
    double const phi = (1.0 + std::sqrt(5.0)) * 0.5;
    double const radius = 0.5;
    GfVec3d const center(0.5, 0.60247475, 0.5);
    std::array<GfVec3d, 12> const base = {{
        {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
        {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
        {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}}};
    for (GfVec3d point : base) {
        point.Normalize();
        s.points.emplace_back(center + point * radius);
    }
    std::vector<std::array<int, 3>> faces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};
    for (auto &f : faces) {
        GfVec3d const a(s.points[size_t(f[0])]);
        GfVec3d const b(s.points[size_t(f[1])]);
        GfVec3d const c(s.points[size_t(f[2])]);
        if (GfDot(GfCross(b - a, c - a), a - center) < 0.0)
            std::swap(f[1], f[2]);
    }
    for (int level = 0; level < 2; ++level) {
        std::map<std::pair<int, int>, int> middle;
        auto midpoint = [&](int i, int j) {
            auto const key = std::minmax(i, j);
            auto const found = middle.find(key);
            if (found != middle.end()) return found->second;
            GfVec3d point = GfVec3d(s.points[size_t(i)]) +
                            GfVec3d(s.points[size_t(j)]) - center * 2.0;
            point.Normalize();
            int const index = int(s.points.size());
            s.points.emplace_back(center + point * radius);
            middle.emplace(key, index);
            return index;
        };
        std::vector<std::array<int, 3>> refined;
        for (auto const &f : faces) {
            int const ab = midpoint(f[0], f[1]);
            int const bc = midpoint(f[1], f[2]);
            int const ca = midpoint(f[2], f[0]);
            refined.insert(refined.end(), {{f[0], ab, ca}, {f[1], bc, ab},
                                           {f[2], ca, bc}, {ab, bc, ca}});
        }
        faces = std::move(refined);
    }
    for (auto const &f : faces) {
        s.faceVertexCounts.push_back(3);
        s.faceVertexIndices.insert(s.faceVertexIndices.end(),
                                   {f[0], f[1], f[2]});
    }
    s.restPoints = s.points;
    return s;
}

static bool ClearOfConvexSolid(UsdGenSurfaceDesc const &solid,
                               UsdGenCurveBuffer const &strand)
{
    for (size_t i = 0; i < strand.totalCvs; ++i) {
        GfVec3d const p(strand.px[i], strand.py[i], strand.pz[i]);
        GfVec3d const q = i ? GfVec3d(strand.px[i - 1], strand.py[i - 1],
                                       strand.pz[i - 1]) : p;
        double enter = 0.0, leave = 1.0, furthest = -1e9;
        for (size_t f = 0; f < solid.faceVertexCounts.size(); ++f) {
            size_t const o = f * 3;
            GfVec3d const a(solid.points[size_t(solid.faceVertexIndices[o])]);
            GfVec3d const b(solid.points[size_t(solid.faceVertexIndices[o + 1])]);
            GfVec3d const c(solid.points[size_t(solid.faceVertexIndices[o + 2])]);
            GfVec3d normal = GfCross(b - a, c - a);
            normal.Normalize();
            double const atEnd = GfDot(normal, p - a);
            furthest = std::max(furthest, atEnd);
            if (i == 0) continue;
            // Clip against a slightly inset convex hull: mere surface
            // contact is legal; a positive interval inside is not.
            double const atStart = GfDot(normal, q - a) + 1e-5;
            double const inward = atEnd + 1e-5 - atStart;
            if (std::abs(inward) < 1e-14) {
                if (atStart > 0.0) { enter = 1.0; leave = 0.0; }
            } else if (inward < 0.0) {
                enter = std::max(enter, -atStart / inward);
            } else {
                leave = std::min(leave, -atStart / inward);
            }
        }
        if (furthest < -1e-5 || (i && leave > enter + 1e-5)) return false;
    }
    return true;
}


static bool RayTriangleRadius(GfVec3d const &direction, GfVec3d const &a,
                              GfVec3d const &b, GfVec3d const &c,
                              double *radius)
{
    GfVec3d const e1=b-a,e2=c-a,p=GfCross(direction,e2);
    double const det=GfDot(e1,p);
    if (std::abs(det)<1e-12) return false;
    double const inv=1.0/det;
    GfVec3d const t=-a;
    double const u=GfDot(t,p)*inv;
    if (u < -1e-9 || u > 1.0+1e-9) return false;
    GfVec3d const q=GfCross(t,e1);
    double const v=GfDot(direction,q)*inv;
    if (v < -1e-9 || u+v > 1.0+1e-9) return false;
    double const hit=GfDot(e2,q)*inv;
    if (!(hit>0.0)) return false;
    *radius=hit;
    return true;
}

static void CheckStiffSubdivisionHingeCycle()
{
    // Exact 18-CV input and posed colliders from canonical frame 37's
    // curve 4967.  This is intentionally small: it retains the alternating
    // subdivision-limit hinge state without depending on the 8k-curve bake.
    UsdGenGraphDesc desc = CollideDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(128), false},
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false}});
    UsdGenSurfaceDesc shield;
    shield.path = SdfPath("/collide/shield");
    shield.subdivisionScheme = TfToken("catmullClark");
    shield.points = {
        {-0.305995643f, -0.305995643f, -0.305995643f},
        {-0.353333324f, -0.176666662f, -0.353333324f},
        {-0.432743192f, -0.216371596f, -0.216371596f},
        {-0.353333324f, -0.353333324f, -0.176666662f},
        {-0.474046409f, -0.237023205f, 0.0f},
        {-0.374766588f, -0.374766588f, 0.0f},
        {-0.432743192f, -0.216371596f, 0.216371596f},
        {-0.353333324f, -0.353333324f, 0.176666662f},
        {-0.353333324f, -0.176666662f, 0.353333324f},
        {-0.305995643f, -0.305995643f, 0.305995643f},
        {-0.374766588f, 0.0f, -0.374766588f},
        {-0.474046409f, 0.0f, -0.237023205f},
        {-0.529999971f, 0.0f, 0.0f},
        {-0.474046409f, 0.0f, 0.237023205f},
        {-0.374766588f, 0.0f, 0.374766588f},
        {-0.353333324f, 0.176666662f, -0.353333324f},
        {-0.432743192f, 0.216371596f, -0.216371596f},
        {-0.474046409f, 0.237023205f, 0.0f},
        {-0.432743192f, 0.216371596f, 0.216371596f},
        {-0.353333324f, 0.176666662f, 0.353333324f},
        {-0.305995643f, 0.305995643f, -0.305995643f},
        {-0.353333324f, 0.353333324f, -0.176666662f},
        {-0.374766588f, 0.374766588f, 0.0f},
        {-0.353333324f, 0.353333324f, 0.176666662f},
        {-0.305995643f, 0.305995643f, 0.305995643f},
        {0.305995643f, -0.305995643f, -0.305995643f},
        {0.353333324f, -0.176666662f, -0.353333324f},
        {0.432743192f, -0.216371596f, -0.216371596f},
        {0.353333324f, -0.353333324f, -0.176666662f},
        {0.474046409f, -0.237023205f, 0.0f},
        {0.374766588f, -0.374766588f, 0.0f},
        {0.432743192f, -0.216371596f, 0.216371596f},
        {0.353333324f, -0.353333324f, 0.176666662f},
        {0.353333324f, -0.176666662f, 0.353333324f},
        {0.305995643f, -0.305995643f, 0.305995643f},
        {0.374766588f, 0.0f, -0.374766588f},
        {0.474046409f, 0.0f, -0.237023205f},
        {0.529999971f, 0.0f, 0.0f},
        {0.474046409f, 0.0f, 0.237023205f},
        {0.374766588f, 0.0f, 0.374766588f},
        {0.353333324f, 0.176666662f, -0.353333324f},
        {0.432743192f, 0.216371596f, -0.216371596f},
        {0.474046409f, 0.237023205f, 0.0f},
        {0.432743192f, 0.216371596f, 0.216371596f},
        {0.353333324f, 0.176666662f, 0.353333324f},
        {0.305995643f, 0.305995643f, -0.305995643f},
        {0.353333324f, 0.353333324f, -0.176666662f},
        {0.374766588f, 0.374766588f, 0.0f},
        {0.353333324f, 0.353333324f, 0.176666662f},
        {0.305995643f, 0.305995643f, 0.305995643f},
        {-0.176666662f, -0.353333324f, -0.353333324f},
        {-0.216371596f, -0.432743192f, -0.216371596f},
        {-0.237023205f, -0.474046409f, 0.0f},
        {-0.216371596f, -0.432743192f, 0.216371596f},
        {-0.176666662f, -0.353333324f, 0.353333324f},
        {0.0f, -0.374766588f, -0.374766588f},
        {0.0f, -0.474046409f, -0.237023205f},
        {0.0f, -0.529999971f, 0.0f},
        {0.0f, -0.474046409f, 0.237023205f},
        {0.0f, -0.374766588f, 0.374766588f},
        {0.176666662f, -0.353333324f, -0.353333324f},
        {0.216371596f, -0.432743192f, -0.216371596f},
        {0.237023205f, -0.474046409f, 0.0f},
        {0.216371596f, -0.432743192f, 0.216371596f},
        {0.176666662f, -0.353333324f, 0.353333324f},
        {-0.176666662f, 0.353333324f, -0.353333324f},
        {-0.216371596f, 0.432743192f, -0.216371596f},
        {-0.237023205f, 0.474046409f, 0.0f},
        {-0.216371596f, 0.432743192f, 0.216371596f},
        {-0.176666662f, 0.353333324f, 0.353333324f},
        {0.0f, 0.374766588f, -0.374766588f},
        {0.0f, 0.474046409f, -0.237023205f},
        {0.0f, 0.529999971f, 0.0f},
        {0.0f, 0.474046409f, 0.237023205f},
        {0.0f, 0.374766588f, 0.374766588f},
        {0.176666662f, 0.353333324f, -0.353333324f},
        {0.216371596f, 0.432743192f, -0.216371596f},
        {0.237023205f, 0.474046409f, 0.0f},
        {0.216371596f, 0.432743192f, 0.216371596f},
        {0.176666662f, 0.353333324f, 0.353333324f},
        {-0.216371596f, -0.216371596f, -0.432743192f},
        {-0.237023205f, 0.0f, -0.474046409f},
        {-0.216371596f, 0.216371596f, -0.432743192f},
        {0.0f, -0.237023205f, -0.474046409f},
        {0.0f, 0.0f, -0.529999971f},
        {0.0f, 0.237023205f, -0.474046409f},
        {0.216371596f, -0.216371596f, -0.432743192f},
        {0.237023205f, 0.0f, -0.474046409f},
        {0.216371596f, 0.216371596f, -0.432743192f},
        {-0.216371596f, -0.216371596f, 0.432743192f},
        {-0.237023205f, 0.0f, 0.474046409f},
        {-0.216371596f, 0.216371596f, 0.432743192f},
        {0.0f, -0.237023205f, 0.474046409f},
        {0.0f, 0.0f, 0.529999971f},
        {0.0f, 0.237023205f, 0.474046409f},
        {0.216371596f, -0.216371596f, 0.432743192f},
        {0.237023205f, 0.0f, 0.474046409f},
        {0.216371596f, 0.216371596f, 0.432743192f},
    };
    shield.restPoints = shield.points;
    shield.faceVertexCounts = {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4};
    shield.faceVertexIndices = {3,2,1,0,5,4,2,3,7,6,4,5,9,8,6,7,2,11,10,1,4,12,11,2,6,13,12,4,8,14,13,6,11,16,15,10,12,17,16,11,13,18,17,12,14,19,18,13,16,21,20,15,17,22,21,16,18,23,22,17,19,24,23,18,25,26,27,28,28,27,29,30,30,29,31,32,32,31,33,34,26,35,36,27,27,36,37,29,29,37,38,31,31,38,39,33,35,40,41,36,36,41,42,37,37,42,43,38,38,43,44,39,40,45,46,41,41,46,47,42,42,47,48,43,43,48,49,44,0,50,51,3,3,51,52,5,5,52,53,7,7,53,54,9,50,55,56,51,51,56,57,52,52,57,58,53,53,58,59,54,55,60,61,56,56,61,62,57,57,62,63,58,58,63,64,59,60,25,28,61,61,28,30,62,62,30,32,63,63,32,34,64,21,66,65,20,22,67,66,21,23,68,67,22,24,69,68,23,66,71,70,65,67,72,71,66,68,73,72,67,69,74,73,68,71,76,75,70,72,77,76,71,73,78,77,72,74,79,78,73,76,46,45,75,77,47,46,76,78,48,47,77,79,49,48,78,1,80,50,0,10,81,80,1,15,82,81,10,20,65,82,15,80,83,55,50,81,84,83,80,82,85,84,81,65,70,85,82,83,86,60,55,84,87,86,83,85,88,87,84,70,75,88,85,86,26,25,60,87,35,26,86,88,40,35,87,75,45,40,88,9,54,89,8,8,89,90,14,14,90,91,19,19,91,69,24,54,59,92,89,89,92,93,90,90,93,94,91,91,94,74,69,59,64,95,92,92,95,96,93,93,96,97,94,94,97,79,74,64,34,33,95,95,33,39,96,96,39,44,97,97,44,49,79};
    shield.worldMatrix = GfMatrix4d(1.0).SetTranslate(
        GfVec3d(0.5, 0.525, 0.5486111111111112));
    UsdGenSurfaceDesc floor;
    floor.path = SdfPath("/collide/floor");
    floor.subdivisionScheme = TfToken("none");
    floor.points = {
        {-0.5f, -0.100000001f, -0.5f},
        {1.5f, -0.100000001f, -0.5f},
        {1.5f, -0.100000001f, 1.5f},
        {-0.5f, -0.100000001f, 1.5f},
        {-0.5f, 0.0f, -0.5f},
        {1.5f, 0.0f, -0.5f},
        {1.5f, 0.0f, 1.5f},
        {-0.5f, 0.0f, 1.5f},
    };
    floor.restPoints = floor.points;
    floor.faceVertexCounts = {3,3,3,3,3,3,3,3,3,3,3,3};
    floor.faceVertexIndices = {0,1,2,0,2,3,4,6,5,4,7,6,0,5,1,0,4,5,3,2,6,3,6,7,0,3,7,0,7,4,1,5,6,1,6,2};
    UsdGenSurfaceDesc patch;
    patch.path = SdfPath("/collide/patch");
    patch.subdivisionScheme = TfToken("none");
    patch.points = {{0,0,0},{1,0,0},{1,0,1},{0,0,1}};
    patch.restPoints = patch.points;
    patch.faceVertexCounts = {4};
    patch.faceVertexIndices = {0,3,2,1};
    desc.surfaces = {patch,shield,floor};
    desc.nodes.back().surfaces = {patch.path,shield.path,floor.path};
    auto &hair=desc.curveSets.front();
    hair=StraightStrands(SdfPath("/collide/hair"),{{0.444828033f,0,0.653074145f}},18,0);
    hair.points = {
        {0.444828033f,0.000000000f,0.653074145f},{0.447049230f,0.022468261f,0.657169759f},
        {0.449030489f,0.044745393f,0.662301481f},{0.450745672f,0.066763520f,0.668529809f},
        {0.452182144f,0.088459648f,0.675861001f},{0.453341573f,0.109784640f,0.684253454f},
        {0.455477029f,0.132398367f,0.687507987f},{0.457377970f,0.154883400f,0.691672444f},
        {0.459174633f,0.177343503f,0.696014166f},{0.460885972f,0.199829042f,0.700258195f},
        {0.462569416f,0.222362131f,0.704253197f},{0.464254051f,0.244958520f,0.707872391f},
        {0.465967208f,0.267626792f,0.710994601f},{0.467737019f,0.290367991f,0.713491857f},
        {0.469592243f,0.313173831f,0.715219319f},{0.471563727f,0.336021751f,0.716006637f},
        {0.473691761f,0.358866334f,0.715650320f},{0.473723024f,0.378744304f,0.727112472f}};
    hair.rest=hair.points;
    UsdGenCurveBuffer out;
    bool const cooked=Cook(desc,&out);
    Check(cooked,"stiff Catmull-Clark hinge-cycle fixture cooks");
    if (!cooked || out.totalCvs!=18) return;

    UsdGenLimitSurface limit;
    std::string why;
    bool const oracleBuilt=limit.Build(shield,6,&why);
    Check(oracleBuilt,"stiff hinge-cycle independent Far oracle builds");
    if (!oracleBuilt) return;
    struct Tri { GfVec3d a,b,c; };
    std::vector<Tri> tris;
    int constexpr resolution=32;
    for (int face=0;face<int(shield.faceVertexCounts.size());++face) {
        std::vector<GfVec3d> samples(size_t(resolution+1)*size_t(resolution+1));
        for (int y=0;y<=resolution;++y) for (int x=0;x<=resolution;++x) {
            GfVec3f p,du,dv;
            bool const evaluated=limit.Evaluate(
                face,float(x)/resolution,float(y)/resolution,&p,&du,&dv);
            if (!evaluated) {
                Check(false,"stiff hinge-cycle Far oracle evaluates");
                return;
            }
            samples[size_t(y)*(resolution+1)+x]=GfVec3d(p);
        }
        for (int y=0;y<resolution;++y) for (int x=0;x<resolution;++x) {
            auto const p00=samples[size_t(y)*(resolution+1)+x];
            auto const p10=samples[size_t(y)*(resolution+1)+x+1];
            auto const p01=samples[size_t(y+1)*(resolution+1)+x];
            auto const p11=samples[size_t(y+1)*(resolution+1)+x+1];
            tris.push_back({p00,p10,p11}); tris.push_back({p00,p11,p01});
        }
    }
    auto clearance = [&](GfVec3d world) {
        GfVec3d const local=world-GfVec3d(0.5,0.525,0.5486111111111112);
        double const length=local.GetLength();
        if (!(length>0.0)) return -1e9;
        GfVec3d const direction=local/length;
        double radius=-1e9,hit=0;
        for (auto const &t:tris)
            if (RayTriangleRadius(direction,t.a,t.b,t.c,&hit)) radius=std::max(radius,hit);
        if (!(radius>0.0)) return -1e9;
        return length-radius;
    };
    bool clear=true,lengths=true,root=true;
    for (size_t i=0;i<18;++i) {
        GfVec3d const p(out.px[i],out.py[i],out.pz[i]);
        clear &= clearance(p)>=-2e-4 && p[1]>=-2e-4;
        if (i) {
            GfVec3d const a(out.px[i-1],out.py[i-1],out.pz[i-1]);
            for (int s=1;s<128;++s)
                clear &= clearance(a+(p-a)*(double(s)/128.0))>=-2e-4;
            lengths &= Near(float((p-a).GetLength()),
                (hair.points[i]-hair.points[i-1]).GetLength(),2e-5f);
        }
    }
    root = out.px[0]==hair.points[0][0] && out.py[0]==hair.points[0][1] &&
           out.pz[0]==hair.points[0][2];
    Check(clear,"stiff Catmull-Clark hinge cycle clears CVs, segments, and floor");
    Check(root && lengths,"stiff Catmull-Clark hinge cycle preserves root and lengths");
}
static void CheckStiffSphereCycle()
{
    UsdGenGraphDesc desc = CubeDesc({
        {TfToken("offset"), VtValue(0.0f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(1), false},
        {TfToken("resolveType"), VtValue(TfToken("stiff")), false}});
    desc.surfaces.front() = ClosedIcosphere(SdfPath("/collide/collider"));
    auto &hair = desc.curveSets.front();
    hair = StraightStrands(SdfPath("/collide/hair"),
                            {GfVec3f(0.610770583f, 0, 0.216808140f)}, 18, 0);
    // Two successive states of a strand that previously alternated forever
    // against a curved closed collider. Both must clear in one more sweep.
    hair.points = {
        {0.610770583f, 0.000000000f, 0.216808140f},
        {0.615075409f, 0.025729142f, 0.218050480f},
        {0.619807363f, 0.051382691f, 0.219302699f},
        {0.625050366f, 0.076936014f, 0.220568925f},
        {0.630901098f, 0.102356195f, 0.221852347f},
        {0.637483239f, 0.127595797f, 0.223157331f},
        {0.639887750f, 0.153551236f, 0.224769041f},
        {0.644559264f, 0.179197669f, 0.226350605f},
        {0.648657560f, 0.204933390f, 0.228065833f},
        {0.661742151f, 0.223561883f, 0.215265751f},
        {0.674583077f, 0.242407918f, 0.202537596f},
        {0.687096298f, 0.261512071f, 0.189868212f},
        {0.699245334f, 0.280888259f, 0.177257627f},
        {0.710963786f, 0.300542921f, 0.164670527f},
        {0.697096586f, 0.321389824f, 0.172098219f},
        {0.682703614f, 0.341888189f, 0.179495171f},
        {0.668247759f, 0.362293124f, 0.187027603f},
        {0.692901731f, 0.357067734f, 0.180178076f}};
    for (int state = 0; state < 2; ++state) {
        if (state) {
            hair.points[14] = {0.726743221f, 0.319196522f, 0.155444667f};
            hair.points[15] = {0.742023289f, 0.338232398f, 0.146159768f};
            hair.points[16] = {0.757247269f, 0.357375413f, 0.137003005f};
            hair.points[17] = {0.759046793f, 0.332116187f, 0.143388405f};
        }
        hair.rest = hair.points;
        UsdGenCurveBuffer out;
        bool const cooked = Cook(desc, &out);
        Check(cooked, state ? "Collide exterior chord sphere cook runs" :
                              "Collide interior tip sphere cook runs");
        if (!cooked || out.totalCvs != 18) continue;
        bool lengths = true;
        for (size_t i = 1; i < 18; ++i)
            lengths &= Near(SegLen(out, i - 1, i),
                            (hair.points[i] - hair.points[i - 1]).GetLength(),
                            2e-5f);
        Check(out.px[0] == hair.points[0][0] &&
              out.py[0] == hair.points[0][1] &&
              out.pz[0] == hair.points[0][2] && lengths &&
              ClearOfConvexSolid(desc.surfaces.front(), out),
              state ? "Collide sphere exterior chord clears with fixed root and lengths" :
                      "Collide sphere interior tail clears with fixed root and lengths");
    }
}

static double CurveLength(UsdGenCurveBuffer const &b, size_t first, size_t count)
{
    double length=0.0;
    for (size_t i=1;i<count;++i) {
        GfVec3d const a(b.px[first+i-1],b.py[first+i-1],b.pz[first+i-1]);
        GfVec3d const c(b.px[first+i],b.py[first+i],b.pz[first+i]);
        length+=(c-a).GetLength();
    }
    return length;
}

static void CheckDeepPenetrationCut()
{
    auto make=[&](char const *mode,float threshold) {
        auto desc=CubeDesc({
            {TfToken("offset"),VtValue(0.0f),false},
            {TfToken("pushAmount"),VtValue(1.0f),false},
            {TfToken("iterations"),VtValue(32),false},
            {TfToken("resolveType"),VtValue(TfToken("stiff")),false},
            {TfToken("deepPenetrationMode"),VtValue(TfToken(mode)),false},
            {TfToken("cutDepthThreshold"),VtValue(threshold),false}});
        desc.curveSets.front()=StraightStrands(SdfPath("/collide/hair"),
            {GfVec3f(0.5f,0,0.5f)},6,0.1f);
        return desc;
    };
    UsdGenCurveBuffer legacy,explicitLegacy,below,equal,above,repeat;
    Check(Cook(make("collide",0),&legacy),"legacy deep-penetration mode cooks");
    auto omitted=CubeDesc({{TfToken("offset"),VtValue(0.0f),false},
        {TfToken("pushAmount"),VtValue(1.0f),false},
        {TfToken("iterations"),VtValue(32),false},
        {TfToken("resolveType"),VtValue(TfToken("stiff")),false}});
    omitted.curveSets.front()=StraightStrands(SdfPath("/collide/hair"),
        {GfVec3f(0.5f,0,0.5f)},6,0.1f);
    Check(Cook(omitted,&explicitLegacy) && SamePoints(legacy,explicitLegacy),
          "default Collide remains bit-identical to explicit legacy mode");
    Check(Cook(make("cutThenCollide",0.3f),&below) &&
          Near(float(CurveLength(below,0,6)),0.5f,2e-4f),
          "cut threshold above penetration preserves incoming length");
    Check(Cook(make("cutThenCollide",0.18f),&equal) &&
          Near(float(CurveLength(equal,0,6)),0.5f,3e-3f),
          "cut threshold at penetration is continuous");
    Check(Cook(make("cutThenCollide",0.1f),&above) &&
          Near(float(CurveLength(above,0,6)),0.24f,2e-3f),
          "excess deep penetration shortens the control polygon before collision");
    Check(Cook(make("cutThenCollide",0.1f),&repeat) && SamePoints(above,repeat),
          "cut-then-collide is stateless and deterministic");
    Check(above.px[0]==0.5f && above.py[0]==0.0f && above.pz[0]==0.5f,
          "cut-then-collide preserves the root exactly");

    auto crossing=make("cutThenCollide",0.1f);
    auto &hair=crossing.curveSets.front();
    hair=StraightStrands(SdfPath("/collide/hair"),{GfVec3f(0.5f,0,0.5f)},2,0);
    hair.points[1]=GfVec3f(0.5f,0.6f,0.5f); hair.rest=hair.points;
    UsdGenCurveBuffer crossed;
    Check(Cook(crossing,&crossed) && CurveLength(crossed,0,2)<0.59,
          "outside-endpoint segment interior contributes deep cut depth");

    auto zero=make("cutThenCollide",0.0f);
    zero.nodes.back().params.push_back({TfToken("mask"),VtValue(0.0f),false});
    UsdGenCurveBuffer zeroOut,zeroIn;
    Check(Cook(zero,&zeroOut) && CookSourceOnly(zero,SdfPath("/collide/source"),&zeroIn) &&
          SamePoints(zeroOut,zeroIn),"all-zero mask disables cut and collision exactly");

    auto partial=make("cutThenCollide",0.1f);
    partial.nodes.back().params.push_back(
        {TfToken("mask"),VtValue(0.5f),false});
    UsdGenCurveBuffer partialOut;
    Check(Cook(partial,&partialOut) &&
          CurveLength(partialOut,0,6)>CurveLength(above,0,6) &&
          CurveLength(partialOut,0,6)<CurveLength(zeroOut,0,6),
          "partial mask continuously weights one whole-strand cut");

    auto flexible=make("cutThenCollide",0.1f);
    flexible.nodes.back().params[3].value=VtValue(TfToken("flexible"));
    UsdGenCurveBuffer flexibleOut;
    Check(Cook(flexible,&flexibleOut) &&
          CurveLength(flexibleOut,0,6)<0.48 &&
          flexibleOut.px[0]==0.5f && flexibleOut.py[0]==0.0f,
          "flexible cut shortens before its ordinary collision solve");

    auto longCurve=make("cutThenCollide",0.1f);
    longCurve.curveSets.front()=StraightStrands(SdfPath("/collide/hair"),
        {GfVec3f(0.5f,0,0.5f)},65,0.01f);
    UsdGenCurveBuffer longOut;
    Check(Cook(longCurve,&longOut) && longOut.totalCvs==65 &&
          CurveLength(longOut,0,65)<0.63,
          "cut supports imported strands longer than 64 CVs");

    auto toggle=make("collide",0.1f);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    bool toggleOk=compiler.Compile(toggle,&graph).ok;
    context.desc=&graph.Desc();
    toggleOk &= !scheduler.Run(graph,context,1).diagnostics.HasErrors();
    UsdGenCurveBuffer first=graph.Output();
    toggle.nodes.back().params[4].value=VtValue(TfToken("cutThenCollide"));
    toggleOk &= compiler.Recompile(toggle,&graph).ok;
    context.desc=&graph.Desc();
    toggleOk &= !scheduler.Run(graph,context,2).diagnostics.HasErrors();
    UsdGenCurveBuffer cutFresh;
    toggleOk &= Cook(toggle,&cutFresh) && SamePoints(graph.Output(),cutFresh) &&
                CurveLength(graph.Output(),0,6)<CurveLength(first,0,6);
    toggle.nodes.back().params[4].value=VtValue(TfToken("collide"));
    toggleOk &= compiler.Recompile(toggle,&graph).ok;
    context.desc=&graph.Desc();
    toggleOk &= !scheduler.Run(graph,context,3).diagnostics.HasErrors();
    UsdGenCurveBuffer fresh;
    toggleOk &= Cook(toggle,&fresh) && SamePoints(graph.Output(),fresh) &&
                SamePoints(first,fresh);
    Check(toggleOk,"collide-cut-collide structural toggles match fresh graphs");

    // Make every point of a 0.5-unit strand exactly 0.2 units inside the
    // nearest side wall.  The maximum remaining-arc objective is then at the
    // root, so these lengths are the analytic smoothstep values rather than
    // an implementation-specific sample of a varying depth field.
    auto activationFixture=[&](double activation) {
        auto desc=make("cutThenCollide",float(0.2-0.02*activation));
        desc.nodes.back().params.push_back(
            {TfToken("cutBlendDepth"),VtValue(0.02),false});
        auto &surface=desc.surfaces.front();
        for (GfVec3f &p:surface.points)
            p[1]=p[1]<0.3f ? -10.0f : 10.0f;
        surface.restPoints=surface.points;
        auto &curves=desc.curveSets.front();
        curves=StraightStrands(SdfPath("/collide/hair"),
            {GfVec3f(0.5f,0.0f,0.5f)},6,0.1f);
        return desc;
    };
    for (double const x:{0.0,0.25,0.5,0.75,1.0}) {
        UsdGenCurveBuffer actual;
        double const smooth=x*x*(3.0-2.0*x);
        double const expected=0.5*(1.0-smooth);
        bool const cooked=Cook(activationFixture(x),&actual);
        Check(cooked && Near(float(CurveLength(actual,0,6)),float(expected),2e-3f),
              "cut remaining-arc law matches analytic smoothstep activation");
    }
    std::array<UsdGenCurveBuffer,3> boundaryGeometry;
    for (size_t sample=0;sample<boundaryGeometry.size();++sample) {
        auto desc=activationFixture(0.0);
        desc.nodes.back().params[5].value=VtValue(
            sample==0 ? 0.200001f : sample==1 ? 0.2f : 0.199999f);
        Check(Cook(desc,&boundaryGeometry[sample]),
              "cut activation-boundary geometry cooks");
    }
    double boundaryMotion=0.0;
    for(size_t sample=1;sample<boundaryGeometry.size();++sample)
        for(size_t cv=0;cv<boundaryGeometry[sample].totalCvs;++cv) {
            GfVec3d const a(boundaryGeometry[sample-1].px[cv],
                boundaryGeometry[sample-1].py[cv],boundaryGeometry[sample-1].pz[cv]);
            GfVec3d const b(boundaryGeometry[sample].px[cv],
                boundaryGeometry[sample].py[cv],boundaryGeometry[sample].pz[cv]);
            boundaryMotion=std::max(boundaryMotion,(b-a).GetLength());
        }
    Check(boundaryMotion<1e-3 &&
          boundaryGeometry.front().px[0]==boundaryGeometry.back().px[0] &&
          boundaryGeometry.front().py[0]==boundaryGeometry.back().py[0] &&
          boundaryGeometry.front().pz[0]==boundaryGeometry.back().pz[0],
          "cut-mode geometry stays bounded at sampled shortening thresholds");

    auto tinyBoundary=activationFixture(0.0);
    tinyBoundary.nodes.back().params.back().value=VtValue(1e-12);
    auto tinySaturated=tinyBoundary;
    tinySaturated.nodes.back().params[5].value=VtValue(0.199999);
    UsdGenCurveBuffer tinyBoundaryOut,tinySaturatedOut;
    std::vector<std::string> tinyErrors;
    auto finite=[](UsdGenCurveBuffer const &b) {
        return std::all_of(b.px.begin(),b.px.end(),[](float v){return std::isfinite(v);}) &&
               std::all_of(b.py.begin(),b.py.end(),[](float v){return std::isfinite(v);}) &&
               std::all_of(b.pz.begin(),b.pz.end(),[](float v){return std::isfinite(v);});
    };
    bool const boundaryCooked=Cook(tinyBoundary,&tinyBoundaryOut,nullptr,&tinyErrors);
    bool const explicitBudget=!boundaryCooked && std::any_of(
        tinyErrors.begin(),tinyErrors.end(),[](std::string const &error) {
            return error.find("deterministic resource budget")!=std::string::npos;
        });
    bool const saturatedCooked=Cook(tinySaturated,&tinySaturatedOut);
    Check((explicitBudget || (boundaryCooked && finite(tinyBoundaryOut) &&
          Near(float(CurveLength(tinyBoundaryOut,0,6)),0.5f,2e-3f))) &&
          saturatedCooked && finite(tinySaturatedOut) &&
          CurveLength(tinySaturatedOut,0,6)<2e-3,
          "tiny positive cutBlendDepth is finite or fails its budget explicitly");
}

static void CheckCutVertexPlaneChunks()
{
    auto desc=CubeDesc({
        {TfToken("offset"),VtValue(0.0f),false},
        {TfToken("pushAmount"),VtValue(1.0f),false},
        {TfToken("iterations"),VtValue(1),false},
        {TfToken("resolveType"),VtValue(TfToken("stiff")),false},
        {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
        {TfToken("cutDepthThreshold"),VtValue(0.1f),false}});
    UsdGenCurveSetDesc hair;
    hair.path=SdfPath("/collide/hair"); hair.role=UsdGenRole::Curves;
    hair.curveRole=TfToken("hair");
    int constexpr curves=513;
    for (int c=0;c<curves;++c) {
        int const count=c==curves-1 ? 3 : 2;
        hair.curveVertexCounts.push_back(count);
        for (int i=0;i<count;++i) {
            float const y=count==2 ? float(i)*0.6f : float(i)*0.3f;
            hair.points.push_back(GfVec3f(0.5f,y,0.5f));
            hair.widths.push_back(count==2 ? 1.0f+2.0f*float(i)
                                            : 1.0f+float(i));
        }
        hair.curveId.push_back(uint64_t(1000+c)); hair.skinPrim.push_back(c);
        hair.skinPrimUv.push_back(GfVec2f(0.2f,0.3f));
        hair.rootFrame.push_back(GfMatrix4d(1.0));
    }
    hair.rest=hair.points;
    UsdGenAuthoredPlaneDesc fp;
    fp.name=TfToken("cutFloat"); fp.type=UsdGenAuthoredPlaneType::Float32;
    fp.domain=UsdGenAuthoredPlaneDomain::Point;
    UsdGenAuthoredPlaneDesc ip;
    ip.name=TfToken("cutInt"); ip.type=UsdGenAuthoredPlaneType::Int32;
    ip.domain=UsdGenAuthoredPlaneDomain::Point;
    for (int c=0;c<curves;++c) {
        int const count=c==curves-1 ? 3 : 2;
        for (int i=0;i<count;++i) {
            fp.floatValues.push_back(count==2 ? float(i)*10.0f : float(i)*5.0f);
            ip.intValues.push_back(count==2 ? i*10 : i*5);
        }
    }
    VtFloatArray const originalFloat=fp.floatValues;
    VtIntArray const originalInt=ip.intValues;
    hair.authoredPlanes={fp,ip};
    desc.curveSets.front()=hair;
    UsdGenCurveBuffer out;
    bool const cooked=Cook(desc,&out);
    Check(cooked,"cut remaps a >512-curve ragged input");
    if (!cooked) return;
    UsdGenPlane const *floatPlane=nullptr,*intPlane=nullptr;
    for (auto const &plane:out.extraCv) {
        if (plane.name==TfToken("cutFloat")) floatPlane=&plane;
        if (plane.name==TfToken("cutInt")) intPlane=&plane;
    }
    size_t const tip=out.totalCvs-1;
    Check(floatPlane && intPlane && tip<floatPlane->f.size() && tip<intPlane->i.size() &&
          Near(floatPlane->f[tip],4.0f,2e-3f) && intPlane->i[tip]==5 &&
          Near(out.width[tip],1.8f,4e-4f),
          "cut remaps float/int vertex planes and widths across chunk/ragged boundaries");
    Check(desc.curveSets.front().authoredPlanes[0].floatValues==originalFloat &&
          desc.curveSets.front().authoredPlanes[1].intValues==originalInt,
          "cut leaves upstream authored vertex planes immutable");
    Check(out.curveId.back()==uint64_t(1512) && out.rootPrim.back()==512 &&
          out.rest==hair.rest && out.hairT.back()==1.0f,
          "cut preserves uniform metadata, rest, and generator hairT");
}

static void CheckCutTemporalCurve7000()
{
#ifdef USDGEN_TEST_SOURCE_DIR
    std::ifstream stream(std::string(USDGEN_TEST_SOURCE_DIR)+
        "/tests/fixtures/collide/curve7000-89-90-input.json");
    std::stringstream text; text<<stream.rdbuf();
    JsParseError error;
    JsValue root=JsParseString(text.str(),&error);
    bool ok=stream.good() || stream.eof();
    JsArray const *records=nullptr;
    if (ok && root.IsObject()) {
        auto const &object=root.GetJsObject();
        auto it=object.find("records");
        if (it!=object.end() && it->second.IsArray()) records=&it->second.GetJsArray();
    }
    ok &= records && records->size()==5;
    std::vector<UsdGenCurveBuffer> outputs;
    std::vector<UsdGenGraphDesc> fixtures;
    double incomingLength=0.0;
    auto number=[](JsValue const &v) { return v.IsReal()?v.GetReal():double(v.GetInt64()); };
    if (ok) for (JsValue const &value:*records) {
        auto const &r=value.GetJsObject();
        UsdGenGraphDesc desc;
        desc.description=SdfPath("/temporal"); desc.defaultWidth=0.01f;
        UsdGenCurveSetDesc hair; hair.path=SdfPath("/temporal/hair");
        hair.role=UsdGenRole::Curves; hair.curveRole=TfToken("hair");
        auto const &points=r.at("upstreamPoints").GetJsArray();
        hair.curveVertexCounts={int(points.size())};
        for (auto const &point:points) { auto const &p=point.GetJsArray();
            hair.points.emplace_back(float(number(p[0])),float(number(p[1])),float(number(p[2]))); }
        hair.rest=hair.points; hair.widths.assign(points.size(),0.01f);
        hair.curveId={7000}; hair.skinPrim={0}; hair.skinPrimUv={GfVec2f(0)};
        hair.rootFrame={GfMatrix4d(1)}; desc.curveSets={hair};
        std::vector<SdfPath> colliderPaths;
        for (auto const &targetValue:r.at("targets").GetJsArray()) {
            auto const &target=targetValue.GetJsObject(); UsdGenSurfaceDesc surface;
            surface.path=SdfPath(target.at("path").GetString());
            for (auto const &point:target.at("pointsDescriptionSpace").GetJsArray()) {
                auto const &p=point.GetJsArray(); surface.points.emplace_back(
                    float(number(p[0])),float(number(p[1])),float(number(p[2]))); }
            surface.restPoints=surface.points;
            for(auto const &v:target.at("faceVertexCounts").GetJsArray()) surface.faceVertexCounts.push_back(v.GetInt());
            for(auto const &v:target.at("faceVertexIndices").GetJsArray()) surface.faceVertexIndices.push_back(v.GetInt());
            for(auto const &v:target.at("holeIndices").GetJsArray()) surface.holeIndices.push_back(v.GetInt());
            surface.subdivisionScheme=TfToken(target.at("subdivisionScheme").GetString());
            surface.orientation=TfToken(target.at("orientation").GetString());
            colliderPaths.push_back(surface.path); desc.surfaces.push_back(std::move(surface));
        }
        desc.nodes={SourceNode(SdfPath("/temporal/source"),hair.path)};
        auto collide=OpNode("/temporal/collide","UsdGenCollide",SdfPath("/temporal/source"),7);
        collide.surfaces=colliderPaths; collide.params={
            {TfToken("offset"),VtValue(0.0f),false},{TfToken("pushAmount"),VtValue(1.0f),false},
            {TfToken("iterations"),VtValue(128),false},{TfToken("resolveType"),VtValue(TfToken("stiff")),false},
            {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
            {TfToken("cutDepthThreshold"),VtValue(0.0f),false}};
        desc.nodes.push_back(collide); desc.terminal=collide.path;
        fixtures.push_back(desc);
        UsdGenCurveBuffer out; ok &= Cook(desc,&out); outputs.push_back(std::move(out));
        if (!incomingLength) { UsdGenCurveBuffer in; in.px.resize(points.size());in.py.resize(points.size());in.pz.resize(points.size());
            for(size_t i=0;i<points.size();++i){in.px[i]=hair.points[i][0];in.py[i]=hair.points[i][1];in.pz[i]=hair.points[i][2];}
            incomingLength=CurveLength(in,0,points.size()); }
    }
    double worst=0.0; size_t worstFrame=0,worstCv=0;
    for(size_t f=1;ok && f<outputs.size();++f) for(size_t i=0;i<outputs[f].totalCvs;++i) {
        GfVec3d a(outputs[f-1].px[i],outputs[f-1].py[i],outputs[f-1].pz[i]);
        GfVec3d b(outputs[f].px[i],outputs[f].py[i],outputs[f].pz[i]);
        double const motion=(b-a).GetLength()/incomingLength;
        if (motion>worst) { worst=motion; worstFrame=f; worstCv=i; }
    }
    ok &= worst<0.01 && outputs.front().px[0]==outputs.back().px[0] &&
          CurveLength(outputs.front(),0,18)<incomingLength*0.2;
    if (!(worst<0.01)) std::printf(
        "curve7000 sampled worst %.17g transition=%zu cv=%zu\n",
        worst,worstFrame,worstCv);
    UsdGenCurveBuffer legacyA,legacyB;
    if (fixtures.size()==5) {
        for (size_t f=3;f<5;++f)
            fixtures[f].nodes.back().params[4].value=VtValue(TfToken("collide"));
        ok &= Cook(fixtures[3],&legacyA) && Cook(fixtures[4],&legacyB);
        double legacyJump=0.0;
        for(size_t i=0;i<legacyA.totalCvs;++i) {
            GfVec3d a(legacyA.px[i],legacyA.py[i],legacyA.pz[i]);
            GfVec3d b(legacyB.px[i],legacyB.py[i],legacyB.pz[i]);
            legacyJump=std::max(legacyJump,(b-a).GetLength()/incomingLength);
        }
        ok &= legacyJump>0.5;
    }
    Check(ok,"curve7000 quarter-frame cut is stateless, shortened, and branch-continuous");
#endif
}

static void CheckCutProjectedCurve1870()
{
#ifdef USDGEN_TEST_SOURCE_DIR
    std::ifstream stream(std::string(USDGEN_TEST_SOURCE_DIR)+
        "/tests/fixtures/collide/curve1870-cut-local-progress.json");
    std::stringstream text; text<<stream.rdbuf();
    JsParseError error; JsValue root=JsParseString(text.str(),&error);
    bool ok=(stream.good() || stream.eof()) && root.IsObject();
    if (!ok) { Check(false,"curve1870 cut local-progress fixture parses"); return; }
    auto const &document=root.GetJsObject();
    auto const &curve=document.at("curve").GetJsObject();
    auto const &poses=document.at("poses").GetJsArray();
    auto const &staticTargets=document.at("staticTargets").GetJsArray();
    auto number=[](JsValue const &v) {
        return v.IsReal()?v.GetReal():double(v.GetInt64());
    };
    auto surface=[&](JsObject const &target) {
        UsdGenSurfaceDesc s; s.path=SdfPath(target.at("path").GetString());
        for (auto const &value:target.at("pointsDescriptionSpace").GetJsArray()) {
            auto const &p=value.GetJsArray(); s.points.emplace_back(
                float(number(p[0])),float(number(p[1])),float(number(p[2])));
        }
        s.restPoints=s.points;
        for (auto const &v:target.at("faceVertexCounts").GetJsArray())
            s.faceVertexCounts.push_back(v.GetInt());
        for (auto const &v:target.at("faceVertexIndices").GetJsArray())
            s.faceVertexIndices.push_back(v.GetInt());
        for (auto const &v:target.at("holeIndices").GetJsArray())
            s.holeIndices.push_back(v.GetInt());
        s.subdivisionScheme=TfToken(target.at("subdivisionScheme").GetString());
        s.orientation=TfToken(target.at("orientation").GetString());
        return s;
    };
    UsdGenCurveSetDesc hair; hair.path=SdfPath("/cutLocal/hair");
    hair.role=UsdGenRole::Curves; hair.curveRole=TfToken("hair");
    for (auto const &value:curve.at("upstreamPoints").GetJsArray()) {
        auto const &p=value.GetJsArray(); hair.points.emplace_back(
            float(number(p[0])),float(number(p[1])),float(number(p[2])));
    }
    hair.curveVertexCounts={int(hair.points.size())}; hair.rest=hair.points;
    for (auto const &v:curve.at("widths").GetJsObject().at("value").GetJsArray())
        hair.widths.push_back(float(number(v)));
    hair.curveId={1870}; hair.skinPrim={0}; hair.skinPrimUv={GfVec2f(0)};
    hair.rootFrame={GfMatrix4d(1)};
    UsdGenCurveBuffer incoming; incoming.totalCvs=uint32_t(hair.points.size());
    incoming.px.resize(hair.points.size()); incoming.py.resize(hair.points.size());
    incoming.pz.resize(hair.points.size());
    for(size_t i=0;i<hair.points.size();++i) {
        incoming.px[i]=hair.points[i][0]; incoming.py[i]=hair.points[i][1];
        incoming.pz[i]=hair.points[i][2];
    }
    double const incomingLength=CurveLength(incoming,0,hair.points.size());
    std::vector<UsdGenCurveBuffer> outputs;
    std::vector<double> lengths;
    for (auto const &poseValue:poses) {
        auto const &pose=poseValue.GetJsObject(); UsdGenGraphDesc desc;
        desc.description=SdfPath("/cutLocal"); desc.defaultWidth=0.01f;
        desc.curveSets={hair}; std::vector<SdfPath> paths;
        for (auto const &targetValue:staticTargets) {
            desc.surfaces.push_back(surface(targetValue.GetJsObject()));
            paths.push_back(desc.surfaces.back().path);
        }
        auto shield=surface(pose.at("shield").GetJsObject());
        desc.surfaces.insert(desc.surfaces.begin()+1,std::move(shield));
        paths.insert(paths.begin()+1,desc.surfaces[1].path);
        desc.nodes={SourceNode(SdfPath("/cutLocal/source"),hair.path)};
        auto collide=OpNode("/cutLocal/collide","UsdGenCollide",
                            SdfPath("/cutLocal/source"),7);
        collide.surfaces=paths; collide.params={
            {TfToken("offset"),VtValue(0.0f),false},
            {TfToken("pushAmount"),VtValue(1.0f),false},
            {TfToken("iterations"),VtValue(128),false},
            {TfToken("resolveType"),VtValue(TfToken("flexible")),false},
            {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
            {TfToken("cutDepthThreshold"),VtValue(0.01f),false},
            {TfToken("cutBlendDepth"),VtValue(0.02f),false}};
        desc.nodes.push_back(collide); desc.terminal=collide.path;
        UsdGenCurveBuffer out; ok &= Cook(desc,&out);
        if (!out.totalCvs) continue;
        ok &= out.px[0]==hair.points[0][0] && out.py[0]==hair.points[0][1] &&
              out.pz[0]==hair.points[0][2];
        lengths.push_back(CurveLength(out,0,out.totalCvs));
        outputs.push_back(std::move(out));
    }
    double worst=0.0;
    for(size_t f=1;f<outputs.size();++f)
        for(size_t i=0;i<outputs[f].totalCvs;++i) {
            GfVec3d const a(outputs[f-1].px[i],outputs[f-1].py[i],outputs[f-1].pz[i]);
            GfVec3d const b(outputs[f].px[i],outputs[f].py[i],outputs[f].pz[i]);
            worst=std::max(worst,(b-a).GetLength()/incomingLength);
        }
    ok &= outputs.size()==poses.size() && std::isfinite(worst) &&
          std::all_of(lengths.begin(),lengths.end(),[&](double length) {
              return std::isfinite(length) && length>0.0;
          });
    Check(ok,"cut projected-flex curve1870 preserves finite geometry and roots");
#endif
}


// --- (1) registry contract ----------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenCollide");
    Check(registry.HasKernel(type), "HasKernel(UsdGenCollide)");
    std::unique_ptr<UsdGenOp> op = registry.Create(type);
    Check(op != nullptr, "Create(UsdGenCollide)");
    if (!op) return;
    Check(op->Type() == type, "Collide reports its own type");
    Check(op->TopologyEffect() == UsdGenTopoFx::None,
          "Collide has TopologyEffect None");
    Check(op->ReferenceInputs().empty(),
          "Collide declares no reference inputs");
    Check(op->OutputPrimvars().empty(), "Collide emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "Collide touches the points plane only");
    std::set<std::string> topo, value;
    for (TfToken const &t : op->TopologyParameters()) topo.insert(t.GetString());
    for (TfToken const &t : op->ValueParameters()) value.insert(t.GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed",
                                        "colliders", "resolveType",
                                        "deepPenetrationMode"},
          "Collide topology parameters include deep-penetration mode");
    Check(value == std::set<std::string>{"enabled", "mask", "offset",
                                         "pushAmount", "iterations",
                                         "cutDepthThreshold", "cutBlendDepth"},
          "Collide value parameters include cut-depth threshold");
}

// --- (2) CPU cooks -------------------------------------------------------

static void CheckIdentityCook()
{
    UsdGenGraphDesc desc = CollideDesc({{TfToken("pushAmount"), VtValue(0.0f), false},
                                        {TfToken("offset"), VtValue(0.5f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Collide cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/collide/source"), &plain),
          "Collide fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "Collide preserves the strand topology");
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "Collide with pushAmount 0 passes points through bit-for-bit");

    UsdGenGraphDesc noGap = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false},
                                         {TfToken("offset"), VtValue(0.0f), false}});
    UsdGenCurveBuffer flat;
    Check(Cook(noGap, &flat), "Collide with offset 0 cooks");
    if (flat.totalCvs == 0) return;
    Check(flat.px == plain.px && flat.py == plain.py && flat.pz == plain.pz,
          "Collide with offset 0 passes points through bit-for-bit");
}

static void CheckFlexiblePush()
{
    // pen = 0.5 - 0.25 = 0.25 straight up; pushAmount 1 resolves it in one
    // iteration, so every CV lands at z = 0.25 with x/y undisturbed.
    UsdGenGraphDesc desc = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Collide flexible push cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    bool sides = true, depth = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            sides = sides && Near(out.px[o], rootX) &&
                Near(out.py[o], float(i) * 0.25f);
            depth = depth && Near(out.pz[o], 0.25f);
        }
    }
    Check(sides, "Collide pushes along the surface normal only");
    Check(depth, "Collide resolves offset - dist with pushAmount 1");

    // pushAmount scales the push: 0.5 leaves z = 0.125 after one iteration.
    UsdGenGraphDesc half = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(0.5f), false}});
    UsdGenCurveBuffer halved;
    Check(Cook(half, &halved), "Collide pushAmount scale cook runs");
    if (halved.totalCvs == 0) return;
    bool scaled = true;
    for (size_t o = 0; o < 12; ++o) scaled = scaled && Near(halved.pz[o], 0.125f);
    Check(scaled, "Collide scales the push by pushAmount");

    // A second iteration keeps pushing from the new position:
    // 0.125 + (0.5 - 0.375) * 0.5 = 0.1875.
    UsdGenGraphDesc twice = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                         {TfToken("pushAmount"), VtValue(0.5f), false},
                                         {TfToken("iterations"), VtValue(2), false}});
    UsdGenCurveBuffer iterated;
    Check(Cook(twice, &iterated), "Collide iterations cook runs");
    if (iterated.totalCvs == 0) return;
    bool looped = true;
    for (size_t o = 0; o < 12; ++o) looped = looped && Near(iterated.pz[o], 0.1875f);
    Check(looped, "Collide iterates the push-out loop");

    UsdGenGraphDesc settled = CollideDesc({
        {TfToken("offset"), VtValue(0.5f), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(128), false}});
    UsdGenCurveBuffer settledOut;
    Check(Cook(settled,&settledOut) && SamePoints(out,settledOut),
          "Collide flexible exact fixed point matches one completed sweep");
    auto negativeIterations=[&](int iterations) {
        return CollideDesc({{TfToken("offset"),VtValue(0.5f),false},
            {TfToken("pushAmount"),VtValue(-1.0f),false},
            {TfToken("iterations"),VtValue(iterations),false}});
    };
    UsdGenCurveBuffer negativeTwo,negativeThree;
    Check(Cook(negativeIterations(2),&negativeTwo) &&
          Cook(negativeIterations(3),&negativeThree) &&
          !SamePoints(negativeTwo,negativeThree),
          "Collide flexible changing negative pushes do not exit early");
}

static void CheckMaskCook()
{
    // The mask envelopes the push: half mask halves the 0.25 push.
    UsdGenGraphDesc half = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false},
                                        {TfToken("mask"), VtValue(0.5f), false}});
    UsdGenCurveBuffer halved, plain;
    Check(Cook(half, &halved), "Collide mask envelope cook runs");
    Check(CookSourceOnly(half, SdfPath("/collide/source"), &plain),
          "Collide mask-cook source cooks on its own");
    if (halved.totalCvs == 0 || plain.totalCvs == 0) return;
    bool enveloped = true;
    for (size_t o = 0; o < 12; ++o)
        enveloped = enveloped && Near(halved.pz[o], 0.125f);
    Check(enveloped, "Collide scales the push by the mask");

    UsdGenGraphDesc muted = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                         {TfToken("pushAmount"), VtValue(1.0f), false},
                                         {TfToken("mask"), VtValue(0.0f), false}});
    UsdGenCurveBuffer zero;
    Check(Cook(muted, &zero), "Collide with mask 0 cooks");
    if (zero.totalCvs == 0) return;
    Check(zero.px == plain.px && zero.py == plain.py && zero.pz == plain.pz,
          "Collide with mask 0 passes points through bit-for-bit");
}

static void CheckStiffCook()
{
    // Hinge at CV 3: CVs 0-2 stay bit-for-bit, the hinge CV stays exactly,
    // and the rigid rotation preserves every segment length (0.25).
    UsdGenGraphDesc desc = StiffDesc({{TfToken("offset"), VtValue(0.5f), false},
                                      {TfToken("pushAmount"), VtValue(1.0f), false},
                                      {TfToken("resolveType"), VtValue(TfToken("stiff")), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Collide stiff cook runs on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/collide/source"), &plain),
          "Collide stiff-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool cleanExact = true;
    for (int i = 0; i < 3; ++i)
        cleanExact = cleanExact && out.px[size_t(i)] == plain.px[size_t(i)] &&
            out.py[size_t(i)] == plain.py[size_t(i)] &&
            out.pz[size_t(i)] == plain.pz[size_t(i)];
    Check(cleanExact, "Collide stiff leaves pre-hinge CVs bit-for-bit");
    bool hingeExact = out.px[2] == plain.px[2] && out.py[2] == plain.py[2] &&
        out.pz[2] == plain.pz[2];
    Check(hingeExact, "Collide stiff keeps the hinge CV exactly");
    bool rigid = true;
    for (int i = 0; i < 5; ++i)
        rigid = rigid && Near(SegLen(out, size_t(i), size_t(i) + 1), 0.25f);
    Check(rigid, "Collide stiff preserves every segment length");
    bool tipMoved = !Near(out.px[5], plain.px[5], 1e-6f) ||
        !Near(out.py[5], plain.py[5], 1e-6f) ||
        !Near(out.pz[5], plain.pz[5], 1e-6f);
    Check(tipMoved, "Collide stiff rotates the downstream CVs");
    bool farExact = true;
    for (int i = 0; i < 6; ++i) {
        size_t const o = 6 + size_t(i);
        farExact = farExact && out.px[o] == plain.px[o] &&
            out.py[o] == plain.py[o] && out.pz[o] == plain.pz[o];
    }
    Check(farExact, "Collide stiff leaves the clean strand bit-for-bit");

    // Flexible on the same fixture pushes CVs 3-5 independently, so the
    // hinge segment stretches instead of rotating.
    UsdGenGraphDesc flex = StiffDesc({{TfToken("offset"), VtValue(0.5f), false},
                                      {TfToken("pushAmount"), VtValue(1.0f), false}});
    UsdGenCurveBuffer bent;
    Check(Cook(flex, &bent), "Collide flexible tip fixture cooks");
    if (bent.totalCvs == 0) return;
    bool flexClean = true;
    for (int i = 0; i < 3; ++i)
        flexClean = flexClean && bent.px[size_t(i)] == plain.px[size_t(i)] &&
            bent.py[size_t(i)] == plain.py[size_t(i)] &&
            bent.pz[size_t(i)] == plain.pz[size_t(i)];
    Check(flexClean, "Collide flexible leaves clean CVs bit-for-bit");
    Check(!Near(SegLen(bent, 2, 3), 0.25f, 1e-3f),
          "Collide flexible stretches the hinge segment");
}

static void CheckTwoSurfaceCook()
{
    // A distant bound skin in front plus the collider appended: the skin
    // never wins the closest-point search, so the cook is bit-identical to
    // the collider-only run. This is the production routing shape (bound
    // surface first, colliders after).
    UsdGenGraphDesc desc = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
    desc.surfaces.push_back(TriangleSurface(
        SdfPath("/collide/skin"),
        GfVec3f(-10, -10, -100), GfVec3f(10, -10, -100),
        GfVec3f(0, 10, -100)));
    desc.nodes.back().surfaces = {SdfPath("/collide/skin"),
                                  SdfPath("/collide/collider")};
    UsdGenCurveBuffer out, solo;
    Check(Cook(desc, &out), "Collide two-surface cook runs");
    UsdGenGraphDesc single = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                          {TfToken("pushAmount"), VtValue(1.0f), false}});
    Check(Cook(single, &solo), "Collide single-surface cook runs");
    if (out.totalCvs == 0 || solo.totalCvs == 0) return;
    Check(out.px == solo.px && out.py == solo.py && out.pz == solo.pz,
          "Collide ignores a distant skin and matches the collider-only run");
}

static void CheckSubsetColliderCook()
{
    // A GeomSubset stub (subsetFaces only) over a two-face parent mesh
    // collides against the restricted face exactly like a single-triangle
    // collider of that face.
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/collide");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/collide/hair"), roots, 6, 0.25f));
    UsdGenSurfaceDesc parent;
    parent.path = SdfPath("/collide/parent");
    parent.faceVertexCounts = {3, 3};
    parent.faceVertexIndices = {0, 1, 2, 3, 4, 5};
    parent.points = {GfVec3f(-10, -10, -0.25f), GfVec3f(10, -10, -0.25f),
                     GfVec3f(0, 10, -0.25f),
                     GfVec3f(0, 0, -100), GfVec3f(1, 0, -100),
                     GfVec3f(0, 1, -100)};
    parent.restPoints = parent.points;
    desc.surfaces.push_back(parent);
    UsdGenSurfaceDesc subset;
    subset.path = SdfPath("/collide/parent/sub");
    subset.subsetFaces = {0};
    desc.surfaces.push_back(subset);
    desc.nodes.push_back(
        SourceNode(SdfPath("/collide/source"), SdfPath("/collide/hair")));
    UsdGenNodeDesc collide = OpNode("/collide/op", "UsdGenCollide",
                                    SdfPath("/collide/source"), 7);
    collide.params = {{TfToken("offset"), VtValue(0.5f), false},
                      {TfToken("pushAmount"), VtValue(1.0f), false}};
    collide.surfaces = {SdfPath("/collide/parent/sub")};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/collide/op");
    UsdGenCurveBuffer out, solo;
    Check(Cook(desc, &out), "Collide subset collider cook runs");
    UsdGenGraphDesc single = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                          {TfToken("pushAmount"), VtValue(1.0f), false}});
    Check(Cook(single, &solo), "Collide subset baseline cook runs");
    if (out.totalCvs == 0 || solo.totalCvs == 0) return;
    Check(out.px == solo.px && out.py == solo.py && out.pz == solo.pz,
          "Collide subset collider matches the restricted-face run bit-for-bit");
    UsdGenCurveBuffer plain, moved;
    Check(CookSourceOnly(desc, SdfPath("/collide/source"), &plain),
          "Subset transform source cooks");
    desc.surfaces.front().worldMatrix.SetTranslate(GfVec3d(0, 0, -10));
    Check(Cook(desc, &moved), "Transformed subset parent mesh cooks");
    Check(SamePoints(moved, plain),
          "Subset collider reads its parent mesh world matrix");
}

static void CheckPushAmountExpressionCook()
{
    // Per-strand pushAmount 1 - primIndex: strand 0 resolves to z = 0.25,
    // strand 1 is pushed by 0 and stays bit-for-bit.
    UsdGenGraphDesc desc = CollideDesc({{TfToken("offset"), VtValue(0.5f), false}});
    AddBinding(desc, SdfPath("/collide/op"),
               SdfPath("/collide/Expressions/push"), "1.0 - $primIndex",
               TfToken("float"), FloatShape(), TfToken("pushAmount"),
               expr::Domain::Primitive, VtValue(0.0f));
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Collide with a pushAmount expression cooks on the CPU");
    Check(CookSourceOnly(desc, SdfPath("/collide/source"), &plain),
          "Collide expression-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool pushed = true, held = true;
    for (int i = 0; i < 6; ++i) {
        pushed = pushed && Near(out.pz[size_t(i)], 0.25f);
        size_t const o = 6 + size_t(i);
        held = held && out.px[o] == plain.px[o] && out.py[o] == plain.py[o] &&
            out.pz[o] == plain.pz[o];
    }
    Check(pushed, "Collide reads pushAmount per strand root");
    Check(held, "Collide leaves the zero-push strand bit-for-bit");
}

// --- (3) fail-closed edges ------------------------------------------------

static void CheckEdgesFailClosed()
{
    UsdGenCurveBuffer out;
    auto badMode=CollideDesc({{TfToken("deepPenetrationMode"),
        VtValue(TfToken("bogus")),false}});
    Check(!Cook(badMode,&out),"Collide rejects unknown deep-penetration mode");
    for (double invalid : {-0.01,std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        auto bad=CollideDesc({{TfToken("deepPenetrationMode"),
            VtValue(TfToken("cutThenCollide")),false},
            {TfToken("cutDepthThreshold"),VtValue(invalid),false}});
        Check(!Cook(bad,&out),"Collide rejects invalid literal cut threshold");
    }
    for (char const *expression : {"-0.01","1.0 / 0.0","sqrt(-1.0)"}) {
        auto bad=CollideDesc({{TfToken("deepPenetrationMode"),
            VtValue(TfToken("cutThenCollide")),false}});
        AddBinding(bad,SdfPath("/collide/op"),SdfPath("/collide/threshold"),
            expression,TfToken("float"),FloatShape(),TfToken("cutDepthThreshold"),
            expr::Domain::Groom,VtValue(0.0f));
        Check(!Cook(bad,&out),"Collide rejects invalid connected cut threshold");
    }
    UsdGenGraphDesc bogus = CollideDesc(
        {{TfToken("resolveType"), VtValue(TfToken("bogus")), false}});
    Check(!Cook(bogus, &out), "Collide with an unknown resolveType fails closed");
    UsdGenGraphDesc zeroIt = CollideDesc({{TfToken("iterations"), VtValue(0), false}});
    Check(!Cook(zeroIt, &out), "Collide with iterations 0 fails closed");
    UsdGenGraphDesc negIt = CollideDesc({{TfToken("iterations"), VtValue(-2), false}});
    Check(!Cook(negIt, &out), "Collide with negative iterations fails closed");
    UsdGenGraphDesc infOffset = CollideDesc(
        {{TfToken("offset"), VtValue(std::numeric_limits<double>::infinity()), false}});
    Check(!Cook(infOffset, &out), "Collide with infinite offset fails closed");
    UsdGenGraphDesc nanPush = CollideDesc(
        {{TfToken("pushAmount"),
          VtValue(std::numeric_limits<double>::quiet_NaN()), false}});
    Check(!Cook(nanPush, &out), "Collide with NaN pushAmount fails closed");
    for (double invalid : {0.0,-0.01,std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
        UsdGenGraphDesc badBlend=CollideDesc({
            {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
            {TfToken("cutBlendDepth"),VtValue(invalid),false}});
        Check(!Cook(badBlend,&out),"Collide with invalid cutBlendDepth fails closed");
    }
    for (char const *source : {"0.0", "-0.01", "1.0 / 0.0", "sqrt(-1.0)"}) {
        UsdGenGraphDesc connected=CollideDesc({
            {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
            {TfToken("cutBlendDepth"),VtValue(0.02),false}});
        AddBinding(connected,SdfPath("/collide/op"),
                   SdfPath("/collide/Expressions/blend"),source,
                   TfToken("float"),FloatShape(),TfToken("cutBlendDepth"),
                   expr::Domain::Groom,VtValue(0.02f));
        Check(!Cook(connected,&out),
              "Collide with invalid connected cutBlendDepth fails closed");
    }
    UsdGenGraphDesc varyingBlend=CubeDesc({
        {TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false},
        {TfToken("cutDepthThreshold"),VtValue(0.1),false},
        {TfToken("cutBlendDepth"),VtValue(0.02),false}});
    varyingBlend.curveSets.front()=StraightStrands(SdfPath("/collide/hair"),
        {GfVec3f(0.5f,0,0.5f),GfVec3f(0.5f,0,0.5f)},6,0.1f);
    AddBinding(varyingBlend,SdfPath("/collide/op"),
               SdfPath("/collide/Expressions/blend"),"0.01 + 0.01 * $primIndex",
               TfToken("float"),FloatShape(),TfToken("cutBlendDepth"),
               expr::Domain::Primitive,VtValue(0.02f));
    Check(Cook(varyingBlend,&out) && out.totalCurves==2,
          "Collide accepts finite positive per-primitive cutBlendDepth");
    UsdGenGraphDesc bare = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    bare.nodes.back().surfaces.clear();
    Check(!Cook(bare, &out), "Collide with no collider surfaces fails closed");
    UsdGenGraphDesc missing = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    missing.nodes.back().surfaces = {SdfPath("/collide/absent")};
    Check(!Cook(missing, &out), "Collide with an unresolved collider fails closed");
    UsdGenGraphDesc noPoints = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    noPoints.surfaces.front().points.clear();
    Check(!Cook(noPoints, &out), "Collide with a point-free collider fails closed");
    UsdGenGraphDesc noFaces = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    noFaces.surfaces.front().faceVertexCounts.clear();
    noFaces.surfaces.front().faceVertexIndices.clear();
    Check(!Cook(noFaces, &out), "Collide with a topology-free collider fails closed");
    UsdGenGraphDesc ragged = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    ragged.surfaces.front().faceVertexIndices = {0, 1};
    Check(!Cook(ragged, &out), "Collide with mismatched collider topology fails closed");
    UsdGenGraphDesc wild = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    wild.surfaces.front().faceVertexIndices = {0, 1, 9};
    Check(!Cook(wild, &out), "Collide with an out-of-range collider index fails closed");
    UsdGenGraphDesc orphan = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    orphan.surfaces.front().faceVertexCounts.clear();
    orphan.surfaces.front().faceVertexIndices.clear();
    orphan.surfaces.front().subsetFaces = {0};
    Check(!Cook(orphan, &out), "Collide with a parentless subset collider fails closed");
    UsdGenGraphDesc wildFace = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    wildFace.surfaces.front().subsetFaces = {7};
    Check(!Cook(wildFace, &out), "Collide with an out-of-range subset face fails closed");
    UsdGenGraphDesc flat = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
    flat.surfaces.front().points = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0),
                                    GfVec3f(2, 0, 0)};
    Check(!Cook(flat, &out), "Collide with an all-degenerate collider fails closed");
}

// --- (4) determinism + CUDA rejection -------------------------------------

static void CheckDeterminism()
{
    UsdGenGraphDesc flex = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(0.7f), false},
                                        {TfToken("iterations"), VtValue(3), false}});
    UsdGenCurveBuffer first, second;
    Check(Cook(flex, &first), "Collide determinism first cook runs");
    Check(Cook(flex, &second), "Collide determinism second cook runs");
    if (first.totalCvs == 0 || second.totalCvs == 0) return;
    Check(first.px == second.px && first.py == second.py && first.pz == second.pz,
          "Collide flexible cooks twice bit-identical");
    UsdGenGraphDesc stiff = StiffDesc({{TfToken("offset"), VtValue(0.5f), false},
                                       {TfToken("pushAmount"), VtValue(1.0f), false},
                                       {TfToken("resolveType"), VtValue(TfToken("stiff")), false},
                                       {TfToken("iterations"), VtValue(2), false}});
    UsdGenCurveBuffer third, fourth;
    Check(Cook(stiff, &third), "Collide stiff determinism first cook runs");
    Check(Cook(stiff, &fourth), "Collide stiff determinism second cook runs");
    if (third.totalCvs == 0 || fourth.totalCvs == 0) return;
    Check(third.px == fourth.px && third.py == fourth.py && third.pz == fourth.pz,
          "Collide stiff cooks twice bit-identical");
}

static void CheckCudaRejectsCollide()
{
    // CPU-only, like Clump: the capability matrix admits only Scatter, Grow,
    // CurveSource, Width, Length, Noise, Deform, ReferenceSource and
    // WidthBlend, so the planner fails the graph before any layout work.
    Check(GetCudaExecutionCapabilityMatrix().Find(TfToken("UsdGenCollide")) == nullptr,
          "CUDA capability matrix has no UsdGenCollide row");
    UsdGenGraphDesc desc = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
    desc.executionBackend = UsdGenExecutionBackend::Cuda;
    UsdGenDiagnostics diag;
    auto plan = CompileCudaGraph(desc, &diag);
    Check(plan == nullptr, "CUDA planner rejects a Collide graph");
    Check(diag.HasErrors(), "CUDA planner reports the Collide rejection");
    if (diag.HasErrors()) {
        std::string const &message = diag.errors.front();
        bool const noRow = message ==
            "CUDA: unsupported operator: CUDA capability matrix has no "
            "implementation for UsdGenCollide";
        bool const noBuild = message == "CUDA: backend is not built";
        Check(noRow || noBuild,
              "CUDA rejection names the missing Collide row (or no CUDA build)");
    }
}

static UsdGenSurfaceDesc RaisedLimitGrid()
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/collide/collider");
    s.subdivisionScheme = TfToken("catmullClark");
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
            s.points.push_back(GfVec3f(float(x), float(y),
                                       x == 1 && y == 1 ? 1.0f : 0.0f));
    s.restPoints = s.points;
    s.faceVertexCounts = {4,4,4,4};
    s.faceVertexIndices = {0,1,4,3, 1,2,5,4,
                           3,4,7,6, 4,5,8,7};
    return s;
}

static UsdGenGraphDesc OneLimitStrand(UsdGenSurfaceDesc const &surface,
                                      GfVec3f const &root, float offset)
{
    UsdGenGraphDesc desc = CollideDesc({
        {TfToken("offset"), VtValue(offset), false},
        {TfToken("pushAmount"), VtValue(1.0f), false},
        {TfToken("iterations"), VtValue(1), false}});
    desc.surfaces[0] = surface;
    desc.curveSets[0] = StraightStrands(SdfPath("/collide/hair"),
                                       {root}, 2, 0.0f);
    return desc;
}

static void CheckAnalyticLimitContact(double scale)
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/collide/collider");
    s.subdivisionScheme = TfToken("catmullClark");
    double const curvature = 0.1;
    for (int y=0; y<4; ++y)
        for (int x=0; x<4; ++x)
            s.points.push_back(GfVec3f(float(scale*x),float(scale*y),
                float(scale*curvature*(x*x+y*y))));
    s.restPoints = s.points;
    for (int y=0; y<3; ++y)
        for (int x=0; x<3; ++x) {
            int const i = y*4+x;
            s.faceVertexCounts.push_back(4);
            for (int v : {i,i+1,i+5,i+4})
                s.faceVertexIndices.push_back(v);
        }
    UsdGenCollisionLimitSurface limit;
    std::string why;
    bool const built = limit.Build(s,&why);
    Check(built,"regular curved Catmull-Clark patch builds");
    if (!built) return;
    double const u=.37,v=.41;
    GfVec3d actual,du,dv;
    bool const evaluated = limit.Evaluate(4,u,v,&actual,&du,&dv);
    Check(evaluated,"regular curved Catmull-Clark patch evaluates");
    if (!evaluated) return;
    GfVec3d const analytic(scale*(1+u),scale*(1+v),
        scale*curvature*((1+u)*(1+u)+(1+v)*(1+v)+2.0/3.0));
    GfVec3d const analyticDu(scale,0,scale*2*curvature*(1+u));
    GfVec3d const analyticDv(0,scale,scale*2*curvature*(1+v));
    double const tolerance = std::max(1e-12,scale*1e-5);
    Check((actual-analytic).GetLength() < tolerance &&
          (du-analyticDu).GetLength() < tolerance &&
          (dv-analyticDv).GetLength() < tolerance,
          "regular Bfr limit matches independent bicubic quadratic oracle");
    GfVec3d lower, upper;
    bool const bounded = limit.FaceControlBounds(4,&lower,&upper);
    Check(bounded,"regular curved limit provides a face bound");
    if (bounded) {
        bool encloses = true;
        for (int iu=0; iu<=16; ++iu)
            for (int iv=0; iv<=16; ++iv) {
                GfVec3d sample,su,sv;
                if (!limit.Evaluate(4,double(iu)/16.0,double(iv)/16.0,
                                    &sample,&su,&sv)) {
                    encloses = false;
                    continue;
                }
                for (int axis=0; axis<3; ++axis)
                    encloses = encloses &&
                        sample[axis] >= lower[axis] &&
                        sample[axis] <= upper[axis];
            }
        Check(encloses,"regular Bernstein bound encloses dense limit samples");
        Check(upper[2] < scale*1.2,
              "regular Bernstein bound is tighter than the support cage");
    }
    GfVec3d const cross = GfCross(analyticDu,analyticDv);
    GfVec3d const normal = cross/cross.GetLength();
    GfVec3d const initial = analytic + normal*(scale*0.01);
    UsdGenCurveBuffer out;
    auto desc = OneLimitStrand(s,GfVec3f(initial),float(scale*0.02));
    bool cooked = Cook(desc,&out);
    Check(cooked,"regular curved limit contact cooks");
    if (cooked && out.totalCvs) {
        GfVec3d const contact(out.px[0],out.py[0],out.pz[0]);
        double const error =
            (contact-(analytic+normal*(scale*0.02))).GetLength();
        if (error >= tolerance)
            std::printf("  analytic contact scale %.3g error %.9g tol %.9g in=(%.12g,%.12g,%.12g) out=(%.12g,%.12g,%.12g)\n",
                        scale,error,tolerance,initial[0],initial[1],initial[2],
                        contact[0],contact[1],contact[2]);
        Check(error < tolerance,
              "contact follows analytic curved limit at groom-scale tolerance");
    }
}

static void CheckClosedLimitBoundary(double scale)
{
    UsdGenSurfaceDesc s = ClosedCube(SdfPath("/collide/collider"));
    s.subdivisionScheme = TfToken("catmullClark");
    for (GfVec3f &p : s.points) p *= float(scale);
    s.restPoints = s.points;
    UsdGenCollisionLimitSurface limit;
    std::string why;
    bool const built = limit.Build(s,&why);
    Check(built,"closed Catmull-Clark cube limit builds");
    if (!built) return;
    GfVec3d q,du,dv;
    bool const evaluated = limit.Evaluate(0,0.5,0.5,&q,&du,&dv);
    Check(evaluated,"closed Catmull-Clark face evaluates");
    if (!evaluated) return;
    GfVec3d const cross = GfCross(du,dv);
    GfVec3d const outward = cross/cross.GetLength();
    UsdGenCurveBuffer out;
    auto desc = OneLimitStrand(s,GfVec3f(q-outward*(scale*1e-4)),
                               float(scale*1e-3));
    bool cooked = Cook(desc,&out);
    Check(cooked,"near-shell closed limit inside cook runs");
    if (cooked && out.totalCvs) {
        GfVec3d const actual(out.px[0],out.py[0],out.pz[0]);
        Check((actual-(q+outward*(scale*1e-3))).GetLength() <
              std::max(1e-12,scale*2e-5),
              "near-shell inside point exits the true limit at scale");
    }
    desc = OneLimitStrand(s,GfVec3f(q+outward*(scale*3e-3)),
                          float(scale*1e-3));
    GfVec3f const outside = desc.curveSets[0].points[0];
    cooked = Cook(desc,&out);
    Check(cooked,"near-shell closed limit outside cook runs");
    if (cooked && out.totalCvs)
        Check(GfVec3f(out.px[0],out.py[0],out.pz[0]) == outside,
              "near-shell outside point remains unchanged");
}

static void CheckCurvedLimitSegment()
{
    UsdGenSurfaceDesc s = ClosedCube(SdfPath("/collide/collider"));
    s.subdivisionScheme = TfToken("catmullClark");
    UsdGenCollisionLimitSurface limit;
    std::string why;
    if (!limit.Build(s,&why)) {
        Check(false,"curved closed segment fixture builds");
        return;
    }
    GfVec3d q,du,dv;
    if (!limit.Evaluate(0,0.5,0.5,&q,&du,&dv)) {
        Check(false,"curved closed segment fixture evaluates");
        return;
    }
    GfVec3d const cross = GfCross(du,dv);
    GfVec3d const outward = cross/cross.GetLength();
    GfVec3d const tangent = du/du.GetLength();
    GfVec3f const start(q+tangent*0.11-outward*0.0005);
    GfVec3f const end(q-tangent*0.11-outward*0.0005);
    UsdGenCurveBuffer out;
    auto endpoint = OneLimitStrand(s,start,0.0f);
    bool const startCooked = Cook(endpoint,&out);
    bool const startOutside = startCooked && out.totalCvs &&
        GfVec3f(out.px[0],out.py[0],out.pz[0]) == start;
    endpoint = OneLimitStrand(s,end,0.0f);
    bool const endCooked = Cook(endpoint,&out);
    bool const endOutside = endCooked && out.totalCvs &&
        GfVec3f(out.px[0],out.py[0],out.pz[0]) == end;
    Check(startOutside && endOutside,
          "curved crossing segment endpoints are outside limit");
    auto desc = OneLimitStrand(s,start,0.0f);
    desc.curveSets[0].points[1] = end;
    desc.curveSets[0].rest = desc.curveSets[0].points;
    desc.nodes[1].params.push_back(
        {TfToken("resolveType"),VtValue(TfToken("stiff")),false});
    bool const cooked = Cook(desc,&out);
    Check(cooked,"curved closed segment cooks");
    if (cooked && out.totalCvs >= 2)
        Check(GfVec3f(out.px[0],out.py[0],out.pz[0]) == start &&
              (GfVec3f(out.px[1],out.py[1],out.pz[1])-end).GetLength() > 1e-4,
              "closed curved surface catches an exterior-to-exterior strand");
}

static void CheckSubdivisionLimit()
{
    CheckAnalyticLimitContact(1.0);
    CheckAnalyticLimitContact(1e-8);
    CheckClosedLimitBoundary(1.0);
    CheckClosedLimitBoundary(1e-8);
    CheckCurvedLimitSegment();
    UsdGenSurfaceDesc grid = RaisedLimitGrid();
    UsdGenCollisionLimitSurface limit;
    std::string why;
    Check(limit.Build(grid,&why), "Catmull-Clark posed limit builds");
    GfVec3d q, du, dv;
    Check(limit.Evaluate(0,0.5,0.5,&q,&du,&dv),
          "Catmull-Clark limit position and derivatives evaluate");
    if (limit.HasFace(0)) {
        GfVec3d normal = GfCross(du,dv).GetNormalized();
        GfVec3d const p = q + normal*0.01;
        UsdGenCurveBuffer out;
        auto desc = OneLimitStrand(grid,GfVec3f(p),0.03f);
        Check(Cook(desc,&out),"Catmull-Clark limit contact cooks");
        if (out.totalCvs >= 1) {
            GfVec3d const actual(out.px[0],out.py[0],out.pz[0]);
            Check((actual-(q+normal*0.03)).GetLength() < 0.002,
                  "contact offset follows posed limit normal");
        }
        desc = OneLimitStrand(grid,GfVec3f(1,1,1),0.03f);
        Check(Cook(desc,&out),"coarse-cage point cook runs");
        if (out.totalCvs >= 1)
            Check(GfVec3d(out.px[0],out.py[0],out.pz[0]) ==
                  GfVec3d(1,1,1),
                  "point on coarse cage but clear of limit stays unchanged");
    }

    UsdGenSurfaceDesc nonquad;
    nonquad.subdivisionScheme = TfToken("catmullClark");
    nonquad.faceVertexCounts = {5};
    nonquad.faceVertexIndices = {0,1,2,3,4};
    nonquad.points = {{0,0,0},{1,0,0},{1.5f,1,0},
                      {0.5f,1.5f,0},{-0.5f,1,0}};
    Check(limit.Build(nonquad,&why),"Catmull-Clark non-quad builds");
    std::vector<GfVec2d> uv;
    std::vector<std::array<int,3>> facets;
    Check(limit.TessellateFace(0,2,&uv,&facets) && !facets.empty(),
          "non-quad Bfr domain tessellates");
    if (!facets.empty()) {
        GfVec2d const middle = (uv[size_t(facets[0][0])] +
            uv[size_t(facets[0][1])] + uv[size_t(facets[0][2])])/3.0;
        Check(limit.Evaluate(0,middle[0],middle[1],&q,&du,&dv),
              "non-quad subface evaluates in its authored domain");
    }
    nonquad.path = SdfPath("/collide/collider");
    nonquad.restPoints = nonquad.points;
    std::vector<std::string> nonquadWarnings;
    UsdGenCurveBuffer nonquadOut;
    auto nonquadCook = OneLimitStrand(nonquad,GfVec3f(0.5f,0.5f,0.01f),
                                      0.03f);
    Check(Cook(nonquadCook,&nonquadOut,&nonquadWarnings),
          "Catmull-Clark non-quad full Collide cook runs");
    Check(std::none_of(nonquadWarnings.begin(),nonquadWarnings.end(),
        [](auto const &w) { return w.find("using polygon collision") !=
                                   std::string::npos; }),
          "Catmull-Clark non-quad full cook uses its limit surface");

    UsdGenSurfaceDesc ccTriangle = TriangleSurface(
        SdfPath("/collide/collider"),{-2,-2,0},{2,-2,0},{0,2,0});
    ccTriangle.subdivisionScheme = TfToken("catmullClark");
    std::vector<std::string> triangleWarnings;
    UsdGenCurveBuffer triangleOut;
    auto triangleCook = OneLimitStrand(ccTriangle,GfVec3f(0,0,0.01f),0.03f);
    Check(Cook(triangleCook,&triangleOut,&triangleWarnings),
          "Catmull-Clark triangle full Collide cook runs");
    bool const triangleFallback = std::any_of(
        triangleWarnings.begin(),triangleWarnings.end(),[](auto const &w) {
            return w.find("using polygon collision") != std::string::npos;
        });
    if (triangleFallback)
        Check(std::any_of(triangleWarnings.begin(),triangleWarnings.end(),
            [](auto const &w) {
                return w.find("OpenSubdiv") != std::string::npos &&
                       w.find("using polygon collision") != std::string::npos;
            }),"rejected Catmull-Clark triangle reports explicit polygon fallback");
    else
        Check(true,"Catmull-Clark triangle full cook uses its limit surface");
    nonquad.subdivisionScheme = TfToken("loop");
    nonquad.faceVertexCounts = {3};
    nonquad.faceVertexIndices = {0,1,2};
    Check(limit.Build(nonquad,&why) &&
          limit.Evaluate(0,0.2,0.2,&q,&du,&dv),
          "Loop triangle limit evaluates");
    nonquad.subdivisionScheme = TfToken("bilinear");
    nonquad.faceVertexCounts = {4};
    nonquad.faceVertexIndices = {0,1,2,3};
    Check(limit.Build(nonquad,&why) &&
          limit.Evaluate(0,0.5,0.5,&q,&du,&dv),
          "bilinear quad limit evaluates");
    nonquad.holeIndices = {0};
    Check(limit.Build(nonquad,&why) && !limit.HasFace(0),
          "all-hole limit is empty without restoring cage face");

    UsdGenCurveBuffer polygon;
    std::vector<std::string> warnings;
    auto none = OneLimitStrand(TriangleSurface(SdfPath("/collide/collider"),
        {-2,-2,0},{2,-2,0},{0,2,0}),GfVec3f(0,0,0.01f),0.03f);
    Check(Cook(none,&polygon,&warnings),"scheme none polygon cook runs");
    Check(std::any_of(warnings.begin(),warnings.end(),[](auto const &w) {
        return w.find("subdivisionScheme=none") != std::string::npos &&
               w.find("/collide/collider") != std::string::npos;
    }),"scheme none warning names path and polygon fallback");
    auto holedFallbackSurface = TriangleSurface(SdfPath("/collide/collider"),
        {-2,-2,0},{2,-2,0},{0,2,0});
    holedFallbackSurface.subdivisionScheme = TfToken("unsupportedScheme");
    holedFallbackSurface.holeIndices = {0};
    auto holedFallback = OneLimitStrand(holedFallbackSurface,
                                        GfVec3f(0,0,0.01f),0.03f);
    UsdGenCurveBuffer holedOut;
    Check(Cook(holedFallback,&holedOut),
          "all-hole unsupported subdivision fallback cooks as empty");
    if (holedOut.totalCvs)
        Check(Near(holedOut.pz[0],0.01f),
              "polygon fallback does not restore an authored hole face");
    grid.points[4][2] = std::numeric_limits<float>::quiet_NaN();
    auto invalid = OneLimitStrand(grid,GfVec3f(0,0,0),0.03f);
    Check(!Cook(invalid,&polygon),"non-finite posed limit fails closed");
}

static void CheckCutGrazingEntry()
{
#ifdef USDGEN_TEST_SOURCE_DIR
    std::ifstream stream(std::string(USDGEN_TEST_SOURCE_DIR)+
        "/tests/fixtures/collide/cut-grazing-entry-89_5.json");
    std::stringstream text; text<<stream.rdbuf();
    JsParseError error; JsValue document=JsParseString(text.str(),&error);
    Check(document.IsObject(),"exact grazing-entry fixture parses");
    if (!document.IsObject()) return;
    auto const &data=document.GetJsObject();
    auto number=[](JsValue const &v) {
        return v.IsReal()?v.GetReal():double(v.GetInt64());
    };
    auto point=[&](JsValue const &v) {
        auto const &p=v.GetJsArray();
        return GfVec3f(float(number(p[0])),float(number(p[1])),float(number(p[2])));
    };
    UsdGenSurfaceDesc original; original.path=SdfPath("/collide/collider");
    original.subdivisionScheme=TfToken("catmullClark");
    for (auto const &p:data.at("points").GetJsArray()) original.points.push_back(point(p));
    for (auto const &v:data.at("faceVertexCounts").GetJsArray()) original.faceVertexCounts.push_back(v.GetInt());
    for (auto const &v:data.at("faceVertexIndices").GetJsArray()) original.faceVertexIndices.push_back(v.GetInt());
    original.restPoints=original.points;
    for (int variant=0;variant<4;++variant) {
        auto surface=original;
        if (variant==1) for (size_t face=0;face<surface.faceVertexCounts.size();++face)
            std::reverse(surface.faceVertexIndices.begin()+std::ptrdiff_t(4*face),
                         surface.faceVertexIndices.begin()+std::ptrdiff_t(4*face+4));
        if (variant==2) surface.orientation=TfToken("leftHanded");
        GfMatrix4d affine(1);
        if (variant==3) {
            affine.SetScale(GfVec3d(-1.3,0.8,1.1));
            affine.SetTranslateOnly(GfVec3d(2,-1,0.5));
            surface.worldMatrix=affine;
        }
        auto oracleSurface=surface;
        GfVec3d center(0);
        for (auto &p:oracleSurface.points) {
            p=GfVec3f(affine.Transform(GfVec3d(p))); center+=GfVec3d(p);
        }
        center/=double(oracleSurface.points.size());
        oracleSurface.worldMatrix=GfMatrix4d(1);
        UsdGenCollisionLimitSurface limit; std::string why;
        bool ok=limit.Build(oracleSurface,&why);
        std::vector<std::pair<GfVec3d,GfVec3d>> planes;
        if (ok) for (size_t face=0;face<surface.faceVertexCounts.size();++face)
            for (int u=0;u<=32;++u) for (int v=0;v<=32;++v) {
                GfVec3d q,du,dv;
                if (!limit.Evaluate(int(face),double(u)/32,double(v)/32,&q,&du,&dv)) {
                    ok=false; continue;
                }
                GfVec3d n=GfCross(du,dv); double length=n.GetLength();
                if (!(length>0)) continue;
                n/=length; if (GfDot(n,q-center)<0) n=-n;
                planes.emplace_back(q,n);
            }
        for (auto const &w:data.at("witnesses").GetJsArray()) {
            auto const &witness=w.GetJsObject();
            GfVec3f const a(affine.Transform(GfVec3d(point(witness.at("a")))));
            GfVec3f const b(affine.Transform(GfVec3d(point(witness.at("b")))));
            auto desc=OneLimitStrand(surface,a,0.0f);
            desc.curveSets.front().points={a,b}; desc.curveSets.front().rest={a,b};
            desc.nodes.back().params.push_back({TfToken("deepPenetrationMode"),VtValue(TfToken("cutThenCollide")),false});
            desc.nodes.back().params.push_back({TfToken("cutDepthThreshold"),VtValue(10.0f),false});
            desc.nodes.back().params.push_back({TfToken("iterations"),VtValue(32),false});
            // This isolated chord starts at a contact CV rather than the
            // original strand root. Pin that CV with the supported envelope.
            AddBinding(desc,SdfPath("/collide/op"),SdfPath("/collide/pin"),
                "$pointIndex > 0",TfToken("float"),FloatShape(),TfToken("mask"),
                expr::Domain::Point,VtValue(1.0f));
            UsdGenCurveBuffer output; bool cooked=Cook(desc,&output);
            ok &= cooked && output.totalCvs==2;
            if (!cooked || output.totalCvs!=2) continue;
            GfVec3d const result(output.px[1],output.py[1],output.pz[1]);
            if (!(output.px[0]==a[0] && output.py[0]==a[1] && output.pz[0]==a[2]))
                std::printf("  grazing root curve=%d variant=%d delta=%g\n",witness.at("curve").GetInt(),variant,
                    (GfVec3d(output.px[0],output.py[0],output.pz[0])-GfVec3d(a)).GetLength());
            ok &= output.px[0]==a[0] && output.py[0]==a[1] && output.pz[0]==a[2];
            double minimum=1;
            for (int i=0;i<=32;++i) {
                GfVec3d const p=GfVec3d(a)+(result-GfVec3d(a))*(double(i)/32);
                double side=-1e30;
                for (auto const &plane:planes) side=std::max(side,GfDot(p-plane.first,plane.second));
                minimum=std::min(minimum,side);
            }
            if (minimum < -2e-5) std::printf("  grazing curve=%d variant=%d min=%g\n",witness.at("curve").GetInt(),variant,minimum);
            ok &= minimum>=-2e-5;
        }
        Check(ok,"cut-flex exact exterior-endpoint curved chords clear in winding/affine variants");
    }
#endif
}

// Coincident strands deliberately share query coordinates but not offsets.
// Cross a chunk boundary and exceed the bounded query cache's capacity.
static void CheckCoincidentStrandOffsets()
{
    auto desc = CollideDesc({{TfToken("pushAmount"), VtValue(1.0f), false}});
    desc.curveSets.front() = StraightStrands(SdfPath("/collide/hair"),
        std::vector<GfVec3f>(600, GfVec3f(0.0f)), 6, 0.25f);
    AddBinding(desc, SdfPath("/collide/op"), SdfPath("/collide/offset"),
        "0.1 + 0.001 * $primIndex", TfToken("float"), FloatShape(),
        TfToken("offset"), expr::Domain::Primitive, VtValue(0.0f));
    UsdGenCurveBuffer output;
    bool ok = Cook(desc, &output) && output.totalCvs == 3600;
    if (ok) for (size_t c = 0; c < 600; ++c) {
        float const z = std::max(0.0f, float(0.1 + 0.001 * c) - 0.25f);
        for (size_t i = 0; i < 6; ++i)
            ok = ok && Near(output.pz[c * 6 + i], z, 1e-6f) &&
                Near(output.px[c * 6 + i], 0.0f, 1e-6f) &&
                Near(output.py[c * 6 + i], float(i) * 0.25f, 1e-6f);
    }
    Check(ok, "coincident strands preserve distinct offsets across chunks and cache eviction");
}

int main()
{
    usdGenRegisterM1Operators();
    if (std::getenv("USDGEN_TEST_GRAZING_ONLY")) {
        CheckCutGrazingEntry(); return failures ? 1 : 0;
    }
    if (std::getenv("USDGEN_TEST_CURVE7000_ONLY")) {
        CheckCutTemporalCurve7000();
        return failures ? 1 : 0;
    }
    // Opt-in repeatable performance fixture; run with ctest -V for timing.
    if (std::getenv("USDGEN_TEST_COLLIDE_BENCHMARK")) {
        auto surface = ClosedCube(SdfPath("/collide/collider"));
        surface.subdivisionScheme = TfToken("catmullClark");
        auto desc = OneLimitStrand(surface, GfVec3f(0.5f, -0.1f, 0.5f), 0.01f);
        std::vector<GfVec3f> roots;
        for (int z=0; z<16; ++z) for (int x=0; x<16; ++x)
            roots.emplace_back(0.28f + x*0.03f, -0.1f, 0.28f + z*0.03f);
        desc.curveSets.front() = StraightStrands(
            SdfPath("/collide/hair"), roots, 16, 0.04f);
        desc.nodes.back().params[2].value = VtValue(128);
        desc.nodes.back().params.push_back({TfToken("deepPenetrationMode"),
            VtValue(TfToken("cutThenCollide")), false});
        UsdGenCurveBuffer output;
        Check(Cook(desc, &output), "dense cut collision benchmark");
        return failures ? 1 : 0;
    }
    CheckRegistryContracts();
    CheckColliderWorldTransforms();
    CheckClosedCubeCook();
    CheckStiffSphereCycle();
    CheckStiffSubdivisionHingeCycle();
    CheckDeepPenetrationCut();
    CheckCutVertexPlaneChunks();
    CheckCutTemporalCurve7000();
    CheckCutProjectedCurve1870();
    CheckIdentityCook();
    CheckFlexiblePush();
    CheckCoincidentStrandOffsets();
    CheckMaskCook();
    CheckStiffCook();
    CheckTwoSurfaceCook();
    CheckSubsetColliderCook();
    CheckPushAmountExpressionCook();
    CheckEdgesFailClosed();
    CheckDeterminism();
    CheckCudaRejectsCollide();
    CheckSubdivisionLimit();
    CheckCutGrazingEntry();
    std::printf("testUsdGenCollide: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
