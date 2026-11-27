// UsdGenStraighten operator contract and CPU cooks.
//
// Covers the UsdGenStraighten styler:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/OutputPrimvars;
//   (2) deterministic CPU cooks over small hand-built bent topologies (fixed
//       seeds): the per-plane blend toward the root->tip line, exact roots and
//       tips, identity at zero effect, and one connected-expression cook.
#include "utils/pxrInterop.h"
#include "usdGen/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

constexpr int kCvs = 6;
constexpr float kStep = 0.25f;
constexpr float kBumpT = 0.3f;
constexpr float kBumpN = 0.2f;

// Bent strands along +Y from the given roots, bowed in +X (the root tangent)
// and +Z (the root normal) by a half-sine so roots and tips stay on axis.
// Mirrors the CurveSource fixture (surface-free, rebind never): valid skin
// bindings plus identity root frames, so rootT=(1,0,0), rootB=(0,1,0),
// rootN=(0,0,1).
static UsdGenCurveSetDesc BentStrands(SdfPath const &path,
                                      std::vector<GfVec3f> const &roots,
                                      int cvs, float step)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = UsdGenLane::Chunked;
    set.curveRole = TfToken("hair");
    size_t const n = roots.size();
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * size_t(cvs));
    constexpr double kPi = 3.14159265358979323846;
    for (size_t c = 0; c < n; ++c)
        for (int i = 0; i < cvs; ++i) {
            // sin(pi) is ~1e-16, not 0: pin the endpoints so roots and tips
            // sit on axis exactly, as the fixture contract states.
            float const bow = (i == 0 || i + 1 == cvs)
                ? 0.0f
                : float(std::sin(kPi * double(i) / double(cvs - 1)));
            set.points[c * size_t(cvs) + size_t(i)] =
                roots[c] + GfVec3f(kBumpT * bow, float(i) * step, kBumpN * bow);
        }
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
    UsdGenCompileResult compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors)
            std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Spec();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors)
            std::printf("  run: %s\n", e.c_str());
        return false;
    }
    *out = OutputBuffer(graph);
    return true;
}

static std::vector<GfVec3f> Roots()
{
    return {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0), GfVec3f(6, 0, 0)};
}

static float Bow(int i)
{
    if (i == 0 || i + 1 == kCvs) return 0.0f;
    constexpr double kPi = 3.14159265358979323846;
    return float(std::sin(kPi * double(i) / double(kCvs - 1)));
}

static UsdGenGraphDesc StraightenDesc(float tangent, float normal)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/straighten");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(
        BentStrands(SdfPath("/straighten/hair"), Roots(), kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/straighten/source"), SdfPath("/straighten/hair")));
    UsdGenNodeDesc op = OpNode("/straighten/op", "UsdGenStraighten",
                               SdfPath("/straighten/source"), 7);
    op.params = {{TfToken("tangentStraightness"), VtValue(tangent), false},
                 {TfToken("normalStraightness"), VtValue(normal), false}};
    desc.nodes.push_back(op);
    desc.terminal = SdfPath("/straighten/op");
    return desc;
}

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContract()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenStraighten");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenStraighten)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenStraighten)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "UsdGenStraighten reports its own type");
    Check(op->Record().Def().countChanges == 0,
          "UsdGenStraighten has TopologyEffect None");
    Check(op->Record().ReferenceInputs().empty(),
          "UsdGenStraighten declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(),
          "UsdGenStraighten emits no output primvars");
}

// --- (2) CPU cooks ------------------------------------------------------
// The chord of every fixture strand is (0, 5*step, 0): tips sit on axis, so
// full straightening in a plane collapses that plane's coordinate onto the
// root's, leaves the growth (binormal, +Y) coordinate alone, and fixes roots
// and tips exactly.

static void CheckFullTangentCook()
{
    UsdGenCurveBuffer out;
    Check(Cook(StraightenDesc(1.0f, 0.0f), &out),
          "Straighten cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 3 && out.totalCvs == 3 * size_t(kCvs),
          "Straighten preserves the strand topology");
    std::vector<GfVec3f> const roots = Roots();
    bool endsFixed = true, tangentCollapsed = true, restKept = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const base = c * size_t(kCvs);
        endsFixed = endsFixed &&
            out.px[base] == roots[c][0] && out.py[base] == 0.0f &&
            out.pz[base] == 0.0f &&
            out.px[base + size_t(kCvs - 1)] == roots[c][0] &&
            out.py[base + size_t(kCvs - 1)] == float(kCvs - 1) * kStep &&
            out.pz[base + size_t(kCvs - 1)] == 0.0f;
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = base + size_t(i);
            tangentCollapsed = tangentCollapsed &&
                Near(out.px[o], roots[c][0], 1e-5f);
            restKept = restKept &&
                Near(out.py[o], float(i) * kStep, 1e-5f) &&
                Near(out.pz[o], kBumpN * Bow(i), 1e-5f);
        }
    }
    Check(endsFixed, "Straighten keeps roots and tips exactly fixed");
    Check(tangentCollapsed,
          "Straighten with tangent 1 collapses every CV onto the root x");
    Check(restKept,
          "Straighten with normal 0 keeps the normal and growth coordinates");
}

static void CheckFullBothCook()
{
    UsdGenCurveBuffer out;
    Check(Cook(StraightenDesc(1.0f, 1.0f), &out),
          "Straighten cooks with both planes at full weight");
    if (out.totalCvs == 0) return;
    std::vector<GfVec3f> const roots = Roots();
    bool collapsed = true, growthKept = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const base = c * size_t(kCvs);
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = base + size_t(i);
            collapsed = collapsed && Near(out.px[o], roots[c][0], 1e-5f) &&
                Near(out.pz[o], 0.0f, 1e-5f);
            growthKept = growthKept && Near(out.py[o], float(i) * kStep, 1e-5f);
        }
    }
    Check(collapsed,
          "Straighten with both planes at 1 puts every CV on the chord");
    Check(growthKept,
          "Straighten never moves the binormal (growth) coordinate");
}

static void CheckHalfCook()
{
    UsdGenCurveBuffer out;
    Check(Cook(StraightenDesc(0.5f, 0.5f), &out),
          "Straighten cooks at half weight");
    if (out.totalCvs == 0) return;
    std::vector<GfVec3f> const roots = Roots();
    bool halves = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const base = c * size_t(kCvs);
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = base + size_t(i);
            halves = halves &&
                Near(out.px[o], roots[c][0] + 0.5f * kBumpT * Bow(i), 1e-5f) &&
                Near(out.pz[o], 0.5f * kBumpN * Bow(i), 1e-5f) &&
                Near(out.py[o], float(i) * kStep, 1e-5f);
        }
    }
    Check(halves, "Straighten at half weight blends each plane halfway");
}

static void CheckIdentityCook()
{
    UsdGenGraphDesc sourceOnly = StraightenDesc(0.0f, 0.0f);
    sourceOnly.nodes.pop_back();
    sourceOnly.terminal = SdfPath("/straighten/source");
    UsdGenCurveBuffer plain;
    Check(Cook(sourceOnly, &plain), "Straighten fixture source cooks on its own");
    if (plain.totalCvs == 0) return;
    UsdGenCurveBuffer zero;
    Check(Cook(StraightenDesc(0.0f, 0.0f), &zero),
          "Straighten cooks at zero effect");
    if (zero.totalCvs == 0) return;
    Check(zero.px == plain.px && zero.py == plain.py && zero.pz == plain.pz,
          "Straighten at zero effect leaves points bitwise unchanged");
    UsdGenGraphDesc masked = StraightenDesc(1.0f, 1.0f);
    masked.nodes.back().params.push_back(
        {TfToken("mask"), VtValue(0.0f), false});
    UsdGenCurveBuffer maskedOut;
    Check(Cook(masked, &maskedOut), "Straighten cooks with a zero mask");
    if (maskedOut.totalCvs == 0) return;
    Check(maskedOut.px == plain.px && maskedOut.py == plain.py &&
              maskedOut.pz == plain.pz,
          "Straighten with a zero mask leaves points bitwise unchanged");
}

static void CheckExpressionCook()
{
    UsdGenGraphDesc desc = StraightenDesc(0.0f, 0.0f);
    desc.nodes.back().params.clear();
    AddBinding(desc, SdfPath("/straighten/op"),
               SdfPath("/straighten/Expressions/tangent"), "$value",
               TfToken("float"), FloatShape(), TfToken("tangentStraightness"),
               expr::Domain::Groom, VtValue(1.0f));
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out),
          "Straighten cooks with a groom tangentStraightness expression");
    if (out.totalCvs == 0) return;
    UsdGenCurveBuffer literal;
    Check(Cook(StraightenDesc(1.0f, 0.0f), &literal),
          "Straighten literal cook for the expression comparison");
    if (literal.totalCvs == 0) return;
    bool matches = out.totalCvs == literal.totalCvs;
    for (size_t o = 0; matches && o < size_t(out.totalCvs); ++o)
        matches = Near(out.px[o], literal.px[o], 1e-6f) &&
            Near(out.py[o], literal.py[o], 1e-6f) &&
            Near(out.pz[o], literal.pz[o], 1e-6f);
    Check(matches,
          "Straighten groom expression matches the literal cook");
}

int main()
{
    CheckRegistryContract();
    CheckFullTangentCook();
    CheckFullBothCook();
    CheckHalfCook();
    CheckIdentityCook();
    CheckExpressionCook();
    std::printf("testUsdGenStraighten: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
