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
//       rejection (no capability-matrix row, like Clump);
//   (5) BVH-vs-brute-force bit-identity over a mixed randomized collider
//       (tris/quads/ngons, degenerates, exact vertex/edge ties, subset
//       restriction, mixed BVH+brute colliders, tiny-BVH threshold edge).
#include "usdGen/compiler.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/collide.h"
#include "usdGen/scheduler.h"

#include <cmath>
#include <cstdio>
#include <limits>
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

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out)
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
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors)
            std::printf("  run: %s\n", e.c_str());
        return false;
    }
    *out = graph.Output();
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

static float SegLen(UsdGenCurveBuffer const &b, size_t a, size_t c)
{
    float const dx = b.px[c] - b.px[a];
    float const dy = b.py[c] - b.py[a];
    float const dz = b.pz[c] - b.pz[a];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
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
                                        "colliders", "resolveType"},
          "Collide topology parameters are input/surface/seed/colliders/resolveType");
    Check(value == std::set<std::string>{"enabled", "mask", "offset",
                                         "pushAmount", "iterations"},
          "Collide value parameters are enabled/mask/offset/pushAmount/iterations");
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

// --- (5) BVH-vs-brute-force bit-identity -----------------------------------

static uint64_t BvhHash(uint64_t k)
{
    k += 0x9e3779b97f4a7c15ULL;
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    return (k ^ (k >> 31)) >> 11;
}

static double BvhHash01(uint64_t k)
{
    return double(BvhHash(k)) / double(1ull << 53);
}

// 10x10 bumped grid (200 tris) plus a pentagon and a collinear degenerate:
// mixed face arities, fan ordinals past k=1, and a skipped degenerate.
static UsdGenSurfaceDesc BvhCollider(SdfPath const &path)
{
    int const n = 10;
    UsdGenSurfaceDesc s;
    s.path = path;
    s.points = VtVec3fArray(size_t(n + 1) * size_t(n + 1) + 8);
    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= n; ++i) {
            float const x = float(i), y = float(j);
            float const z =
                float(std::sin(0.5 * x) * std::cos(0.4 * y));
            s.points[size_t(j) * size_t(n + 1) + size_t(i)] =
                GfVec3f(x, y, z);
        }
    }
    size_t const base = size_t(n + 1) * size_t(n + 1);
    // Pentagon fan at z = 0.5 over (2..4, 2..4); collinear degenerate aside.
    s.points[base + 0] = GfVec3f(2, 2, 0.5f);
    s.points[base + 1] = GfVec3f(4, 2, 0.5f);
    s.points[base + 2] = GfVec3f(4.5f, 3, 0.5f);
    s.points[base + 3] = GfVec3f(3, 4, 0.5f);
    s.points[base + 4] = GfVec3f(2, 3, 0.5f);
    s.points[base + 5] = GfVec3f(0, 0, 5);
    s.points[base + 6] = GfVec3f(1, 0, 5);
    s.points[base + 7] = GfVec3f(2, 0, 5);
    s.restPoints = s.points;
    s.faceVertexCounts.assign(size_t(n) * size_t(n) + 2, 4);
    s.faceVertexCounts[size_t(n) * size_t(n)] = 5;
    s.faceVertexCounts[size_t(n) * size_t(n) + 1] = 3;
    s.faceVertexIndices = VtIntArray(size_t(n) * size_t(n) * 4 + 8);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            size_t const q = (size_t(j) * size_t(n) + size_t(i)) * 4;
            int const v00 = j * (n + 1) + i;
            s.faceVertexIndices[q + 0] = v00;
            s.faceVertexIndices[q + 1] = v00 + 1;
            s.faceVertexIndices[q + 2] = v00 + n + 2;
            s.faceVertexIndices[q + 3] = v00 + n + 1;
        }
    }
    size_t const extra = size_t(n) * size_t(n) * 4;
    for (int k = 0; k < 5; ++k)
        s.faceVertexIndices[extra + size_t(k)] = int(base) + k;
    for (int k = 0; k < 3; ++k)
        s.faceVertexIndices[extra + 5 + size_t(k)] = int(base) + 5 + k;
    return s;
}

static float BvhSurfaceZ(float x, float y)
{
    return float(std::sin(0.5 * x) * std::cos(0.4 * y));
}

static UsdGenGraphDesc BvhDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/bvh");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> roots;
    // 30 scattered roots spanning penetrating/clean heights.
    for (int c = 0; c < 30; ++c) {
        float const rx = float(10.0 * BvhHash01(uint64_t(c) * 3));
        float const ry = float(10.0 * BvhHash01(uint64_t(c) * 3 + 1));
        float const rz = BvhSurfaceZ(rx, ry) +
            float(-0.4 + 1.2 * BvhHash01(uint64_t(c) * 3 + 2));
        roots.push_back(GfVec3f(rx, ry, rz));
    }
    // Exact ties: grid vertices (up to 8-way, zero distance), quad centers
    // (4-way edge ties at height), and pentagon-interior points.
    roots.push_back(GfVec3f(3, 4, BvhSurfaceZ(3, 4)));
    roots.push_back(GfVec3f(7, 2, BvhSurfaceZ(7, 2)));
    roots.push_back(GfVec3f(5, 5, BvhSurfaceZ(5, 5)));
    roots.push_back(GfVec3f(3.5f, 4.5f, BvhSurfaceZ(3.5f, 4.5f) + 0.1f));
    roots.push_back(GfVec3f(6.5f, 1.5f, BvhSurfaceZ(6.5f, 1.5f) + 0.1f));
    roots.push_back(GfVec3f(3, 3, 0.6f));
    roots.push_back(GfVec3f(3, 3, 0.4f));
    roots.push_back(GfVec3f(10.5f, 10.5f, 3.0f));  // far corner, clean
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/bvh/hair"), roots, 6, 0.2f));
    desc.surfaces.push_back(BvhCollider(SdfPath("/bvh/collider")));
    desc.nodes.push_back(
        SourceNode(SdfPath("/bvh/source"), SdfPath("/bvh/hair")));
    UsdGenNodeDesc collide =
        OpNode("/bvh/op", "UsdGenCollide", SdfPath("/bvh/source"), 7);
    collide.params = params;
    collide.surfaces = {SdfPath("/bvh/collider")};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/bvh/op");
    return desc;
}

static bool CookWithPaths(UsdGenGraphDesc const &desc, bool forceBrute,
                          size_t threshold, UsdGenCurveBuffer *out)
{
    UsdGenCollideOp::SetForceBruteForce(forceBrute);
    UsdGenCollideOp::SetBruteForceThreshold(threshold);
    return Cook(desc, out);
}

static void CheckBvhBitIdentical()
{
    // (face, iterations, stiff) matrix over the mixed collider: BVH vs
    // forced brute force, bit for bit.
    std::vector<std::vector<UsdGenParamValue>> cases = {
        {{TfToken("offset"), VtValue(0.5f), false},
         {TfToken("pushAmount"), VtValue(1.0f), false}},
        {{TfToken("offset"), VtValue(0.5f), false},
         {TfToken("pushAmount"), VtValue(0.7f), false},
         {TfToken("iterations"), VtValue(3), false}},
        {{TfToken("offset"), VtValue(0.5f), false},
         {TfToken("pushAmount"), VtValue(1.0f), false},
         {TfToken("iterations"), VtValue(2), false},
         {TfToken("resolveType"), VtValue(TfToken("stiff")), false}},
    };
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        UsdGenGraphDesc desc = BvhDesc(cases[ci]);
        UsdGenCurveBuffer bvh, brute;
        bool const okBvh = CookWithPaths(desc, false, 16, &bvh);
        bool const okBrute = CookWithPaths(desc, true, 16, &brute);
        Check(okBvh && okBrute, "BVH differential case cooks both paths");
        if (!okBvh || !okBrute) continue;
        char label[128];
        std::snprintf(label, sizeof(label),
                      "BVH matches brute force bit-for-bit (case %u)",
                      unsigned(ci));
        Check(bvh.px == brute.px && bvh.py == brute.py &&
                  bvh.pz == brute.pz,
              label);
    }
    // Subset restriction through the BVH: the first 75 grid faces only
    // (150 tris, past the threshold so the BVH path runs).
    {
        UsdGenGraphDesc desc = BvhDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
        UsdGenSurfaceDesc subset;
        subset.path = SdfPath("/bvh/collider/sub");
        subset.subsetFaces.resize(75);
        for (int f = 0; f < 75; ++f) subset.subsetFaces[f] = f;
        desc.surfaces.push_back(subset);
        desc.nodes.back().surfaces = {SdfPath("/bvh/collider/sub")};
        UsdGenCurveBuffer bvh, brute;
        bool const okBvh = CookWithPaths(desc, false, 16, &bvh);
        bool const okBrute = CookWithPaths(desc, true, 16, &brute);
        Check(okBvh && okBrute, "BVH subset differential cooks both paths");
        if (okBvh && okBrute)
            Check(bvh.px == brute.px && bvh.py == brute.py &&
                      bvh.pz == brute.pz,
                  "BVH subset collider matches brute force bit-for-bit");
    }
    // Mixed node: a 1-tri brute collider plus the 200-tri BVH collider in
    // one node; the ordinal tiebreak composes them exactly.
    {
        UsdGenGraphDesc desc = BvhDesc({{TfToken("offset"), VtValue(0.5f), false},
                                        {TfToken("pushAmount"), VtValue(1.0f), false}});
        desc.surfaces.push_back(TriangleSurface(
            SdfPath("/bvh/skin"),
            GfVec3f(-10, -10, -100), GfVec3f(10, -10, -100),
            GfVec3f(0, 10, -100)));
        desc.nodes.back().surfaces = {SdfPath("/bvh/skin"),
                                      SdfPath("/bvh/collider")};
        UsdGenCurveBuffer bvh, brute;
        bool const okBvh = CookWithPaths(desc, false, 16, &bvh);
        bool const okBrute = CookWithPaths(desc, true, 16, &brute);
        Check(okBvh && okBrute, "BVH mixed-collider differential cooks both paths");
        if (okBvh && okBrute)
            Check(bvh.px == brute.px && bvh.py == brute.py &&
                      bvh.pz == brute.pz,
                  "BVH mixed colliders match brute force bit-for-bit");
    }
    // Threshold edges on the 1-tri standard fixture: threshold 0 forces
    // even one triangle through the BVH (single-leaf tree); a huge
    // threshold keeps the standard brute path. Both match forced brute.
    {
        UsdGenGraphDesc desc = CollideDesc({{TfToken("offset"), VtValue(0.5f), false},
                                            {TfToken("pushAmount"), VtValue(1.0f), false}});
        UsdGenCurveBuffer tiny, brute, big;
        bool const okTiny = CookWithPaths(desc, false, 0, &tiny);
        bool const okBrute = CookWithPaths(desc, true, 16, &brute);
        bool const okBig = CookWithPaths(desc, false, 1000000, &big);
        Check(okTiny && okBrute && okBig,
              "BVH threshold-edge cooks run on all three settings");
        if (okTiny && okBrute)
            Check(tiny.px == brute.px && tiny.py == brute.py &&
                      tiny.pz == brute.pz,
                  "BVH single-triangle tree matches brute force bit-for-bit");
        if (okBig && okBrute)
            Check(big.px == brute.px && big.py == brute.py &&
                      big.pz == brute.pz,
                  "BVH huge threshold matches brute force bit-for-bit");
    }
    UsdGenCollideOp::SetForceBruteForce(false);
    UsdGenCollideOp::SetBruteForceThreshold(16);
}

int main()
{
    usdGenRegisterM1Operators();
    CheckRegistryContracts();
    CheckIdentityCook();
    CheckFlexiblePush();
    CheckMaskCook();
    CheckStiffCook();
    CheckTwoSurfaceCook();
    CheckSubsetColliderCook();
    CheckPushAmountExpressionCook();
    CheckEdgesFailClosed();
    CheckDeterminism();
    CheckCudaRejectsCollide();
    CheckBvhBitIdentical();
    std::printf("testUsdGenCollide: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
