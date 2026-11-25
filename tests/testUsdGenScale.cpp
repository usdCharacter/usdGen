// UsdGenScale operator contract and CPU cooks.
//
// Covers the UsdGenScale global length multiplier:
//   (1) registry HasKernel/Create/TopologyEffect/PlanesTouched plus the
//       topology/value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): the geometric contract, roots behavior, widthToo, the
//       scale:knots ramp, the per-curve scaleRandom draw, identity at zero
//       effect, one connected-expression cook, and fail-closed validation.
#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/vt/array.h"

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

constexpr float kTol = 1e-5f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}

// Straight strands along +Y from the given roots. Mirrors the CurveSource
// fixture (surface-free, rebind never); widths are a flat 0.1.
static UsdGenCurveSetDesc StraightStrands(SdfPath const &path,
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
    *out = OutputBuffer(graph);
    return true;
}

// The source-only half of a Scale fixture, for bitwise-identity and
// pass-through comparisons.
static bool CookSourceOnly(UsdGenGraphDesc const &desc, SdfPath const &source,
                           UsdGenCurveBuffer *out)
{
    UsdGenGraphDesc plain = desc;
    plain.nodes.pop_back();
    plain.expressions.clear();
    plain.terminal = source;
    return Cook(plain, out);
}

static bool HasParam(TfSpan<const TfToken> const &span, char const *name)
{
    TfToken const want(name);
    for (TfToken const &t : span)
        if (t == want) return true;
    return false;
}

// --- (1) registry contract -----------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenScale");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenScale)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenScale)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "Scale reports its own type");
    Check(op->Record().Def().countChanges == 0,
          "Scale has TopologyEffect None");
    Check(op->PlanesTouched() ==
              (UsdGenOp::kPlanePoints | UsdGenOp::kPlaneWidths),
          "Scale touches the points and widths planes");
    Check(op->Record().ReferenceInputs().empty(), "Scale declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(), "Scale emits no output primvars");
    UsdGenOpRecord const &rec = registry.Record(registry.Find(type));
    std::vector<TfToken> topoV, valueV;   // routed decls, as the routing snapshot splits them
    for (UsdGenParamSlot s = 0; s < rec.ParamCount(); ++s)
        (rec.ParamClass(s) == UsdGenParamClass::Value ? valueV : topoV).push_back(rec.ParamName(s));
    TfSpan<const TfToken> const topo(topoV.data(), topoV.size());
    TfSpan<const TfToken> const value(valueV.data(), valueV.size());
    Check(HasParam(topo, "widthToo"), "Scale routes widthToo structurally");
    Check(HasParam(value, "scale") && HasParam(value, "scale:knots") &&
              HasParam(value, "scale:interpolation") &&
              HasParam(value, "scaleRandom") && HasParam(value, "mask") &&
              HasParam(value, "enabled"),
          "Scale routes scale/knots/interpolation/random/mask/enabled by value");
}

// --- (2) CPU cooks --------------------------------------------------------

static void CheckScaleCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scale");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0),
                                        GfVec3f(0, 0, 5)};
    constexpr int kCvs = 6;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/scale/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/scale/source"), SdfPath("/scale/hair")));
    UsdGenNodeDesc scale = OpNode("/scale/op", "UsdGenScale",
                                 SdfPath("/scale/source"), 7);
    scale.params = {{TfToken("scale"), VtValue(2.0f), false}};
    desc.nodes.push_back(scale);
    desc.terminal = SdfPath("/scale/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Scale cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 3 && out.totalCvs == 18,
          "Scale preserves the strand topology");
    // out = root + (in - root) * 2 with (in - root) = (0, i*step, 0).
    bool rootsFixed = true, profile = true;
    for (size_t c = 0; c < 3; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            float const lift = float(i) * kStep * 2.0f;
            if (i == 0)
                rootsFixed = rootsFixed && out.px[o] == roots[c][0] &&
                    out.py[o] == 0.0f && out.pz[o] == roots[c][2];
            profile = profile && Near(out.px[o], roots[c][0]) &&
                Near(out.py[o], lift) && Near(out.pz[o], roots[c][2]);
        }
    }
    Check(rootsFixed, "Scale leaves every root exactly unmoved");
    Check(profile, "Scale doubles every root-relative offset about its root");

    UsdGenCurveBuffer plain;
    Check(CookSourceOnly(desc, SdfPath("/scale/source"), &plain),
          "Scale fixture source cooks on its own");
    if (plain.totalCvs == 0) return;
    Check(out.width == plain.width,
          "Scale without widthToo leaves widths bitwise unchanged");
}

static void CheckWidthTooCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scalew");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 4;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/scalew/hair"), roots, kCvs, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/scalew/source"), SdfPath("/scalew/hair")));
    UsdGenNodeDesc scale = OpNode("/scalew/op", "UsdGenScale",
                                 SdfPath("/scalew/source"), 7);
    scale.params = {{TfToken("scale"), VtValue(2.0f), false},
                    {TfToken("widthToo"), VtValue(true), false}};
    desc.nodes.push_back(scale);
    desc.terminal = SdfPath("/scalew/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Scale with widthToo cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    bool widths = out.width.size() == 8;
    if (widths)
        for (float w : out.width) widths = widths && Near(w, 0.2f);
    Check(widths, "widthToo scales every width by the same factor (0.1 -> 0.2)");
    bool tips = true;
    for (size_t c = 0; c < 2; ++c) {
        size_t const tip = c * size_t(kCvs) + size_t(kCvs - 1);
        tips = tips && Near(out.px[tip], roots[c][0]) &&
            Near(out.py[tip], 3.0f * 0.25f * 2.0f) && Near(out.pz[tip], 0.0f);
    }
    Check(tips, "widthToo still scales points about each root");
}

static void CheckIdentityCook()
{
    auto cookWith = [](std::vector<UsdGenParamValue> params,
                       UsdGenCurveBuffer *out) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/scaleid");
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
        desc.curveSets.push_back(
            StraightStrands(SdfPath("/scaleid/hair"), roots, 5, 0.2f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/scaleid/source"), SdfPath("/scaleid/hair")));
        UsdGenNodeDesc scale = OpNode("/scaleid/op", "UsdGenScale",
                                     SdfPath("/scaleid/source"), 7);
        scale.params = std::move(params);
        desc.nodes.push_back(scale);
        desc.terminal = SdfPath("/scaleid/op");
        UsdGenCurveBuffer plain;
        if (!Cook(desc, out)) return false;
        if (!CookSourceOnly(desc, SdfPath("/scaleid/source"), &plain))
            return false;
        return out->px == plain.px && out->py == plain.py &&
            out->pz == plain.pz && out->width == plain.width;
    };
    UsdGenCurveBuffer masked;
    bool const identical = cookWith({{TfToken("scale"), VtValue(1.0f), false}},
                                    &masked);
    Check(masked.totalCvs != 0, "Scale cooks at the identity factor");
    if (masked.totalCvs == 0) return;
    Check(identical, "scale=1 is a bitwise pass-through of points and widths");
    UsdGenCurveBuffer zero;
    bool const zeroMask = cookWith({{TfToken("scale"), VtValue(3.0f), false},
                                    {TfToken("mask"), VtValue(0.0f), false}},
                                   &zero);
    Check(zeroMask, "mask=0 is a bitwise pass-through of points and widths");
}

static void CheckScaleRandomCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scalerand");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0),
                                        GfVec3f(0, 0, 5)};
    constexpr int kCvs = 4;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/scalerand/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/scalerand/source"), SdfPath("/scalerand/hair")));
    UsdGenNodeDesc scale = OpNode("/scalerand/op", "UsdGenScale",
                                 SdfPath("/scalerand/source"), 7);
    scale.params = {{TfToken("scale"), VtValue(1.0f), false},
                    {TfToken("scaleRandom"), VtValue(GfVec2f(0.5f, 1.5f)),
                     false}};
    desc.nodes.push_back(scale);
    desc.terminal = SdfPath("/scalerand/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Scale with scaleRandom cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // The per-curve draw over [0.5, 1.5). The expected salt is pinned
    // by value (not referenced): renumbering kSaltScale is an R12 look
    // change and must fail this test.
    constexpr uint32_t kExpectedSaltScale = 0x5363616Cu;
    float mult[3];
    for (size_t c = 0; c < 3; ++c)
        mult[c] = 0.5f + 1.0f * UsdGenDraw01(7, uint64_t(c), kExpectedSaltScale);
    Check(mult[0] != mult[1] && mult[1] != mult[2] && mult[0] != mult[2],
          "the scaleRandom draws vary per curve at seed 7");
    bool tips = true, rootsFixed = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const base = c * size_t(kCvs);
        rootsFixed = rootsFixed && out.px[base] == roots[c][0] &&
            out.py[base] == 0.0f && out.pz[base] == roots[c][2];
        size_t const tip = base + size_t(kCvs - 1);
        float const lift = 3.0f * kStep * mult[c];
        tips = tips && Near(out.px[tip], roots[c][0]) &&
            Near(out.py[tip], lift) && Near(out.pz[tip], roots[c][2]);
    }
    Check(rootsFixed, "scaleRandom keeps roots exactly fixed");
    Check(tips, "scaleRandom scales each strand by its own draw");
}

static void CheckRampCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scaleramp");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 5;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/scaleramp/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/scaleramp/source"), SdfPath("/scaleramp/hair")));
    VtVec2fArray knots(2);
    knots[0] = GfVec2f(0.0f, 0.0f);
    knots[1] = GfVec2f(1.0f, 1.0f);
    UsdGenNodeDesc scale = OpNode("/scaleramp/op", "UsdGenScale",
                                 SdfPath("/scaleramp/source"), 7);
    scale.params = {{TfToken("scale"), VtValue(1.0f), false},
                    {TfToken("scale:knots"), VtValue(knots), false},
                    {TfToken("scale:interpolation"),
                     VtValue(TfToken("linear")), false}};
    desc.nodes.push_back(scale);
    desc.terminal = SdfPath("/scaleramp/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Scale with a ramp cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    UsdGenCurveBuffer plain;
    Check(CookSourceOnly(desc, SdfPath("/scaleramp/source"), &plain),
          "Scale ramp fixture source cooks on its own");
    if (plain.totalCvs == 0 || plain.hairT.size() != out.totalCvs) return;
    // Expected factors from the same LUT builder over the source hairT.
    std::vector<float> lut(kUsdGenRampLutSize, 1.0f);
    UsdGenBuildRampLut(knots, TfToken("linear"), lut.data(),
                       kUsdGenRampLutSize);
    bool profile = true, rootsFixed = true, gathered = true;
    for (size_t c = 0; c < 2; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            float const L = UsdGenEvalLut257(lut.data(), plain.hairT[o]);
            float const ex = roots[c][0] + (plain.px[o] - roots[c][0]) * L;
            float const ey = 0.0f + (plain.py[o] - 0.0f) * L;
            float const ez = roots[c][2] + (plain.pz[o] - roots[c][2]) * L;
            profile = profile && Near(out.px[o], ex, 1e-4f) &&
                Near(out.py[o], ey, 1e-4f) && Near(out.pz[o], ez, 1e-4f);
            if (i == 0)
                rootsFixed = rootsFixed && out.px[o] == roots[c][0] &&
                    out.py[o] == 0.0f && out.pz[o] == roots[c][2];
            // The tip (t = 1, L = 1) is unmoved by a 0->1 ramp; every other
            // CV must gather strictly toward its root.
            if (i > 0 && i + 1 < kCvs) {
                float const inD = std::fabs(plain.py[o] - 0.0f);
                float const outD = std::fabs(out.py[o] - 0.0f);
                gathered = gathered && outD < inD;
            }
        }
    }
    Check(rootsFixed, "Scale ramp keeps roots exactly fixed");
    Check(profile, "Scale ramp follows the LUT over hairT per CV");
    Check(gathered, "a 0->1 ramp gathers every mid-CV toward its root");
}

static void CheckConnectedExpressionCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scaleexpr");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0),
                                        GfVec3f(0, 0, 5)};
    constexpr int kCvs = 4;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/scaleexpr/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/scaleexpr/source"), SdfPath("/scaleexpr/hair")));
    UsdGenNodeDesc scale = OpNode("/scaleexpr/op", "UsdGenScale",
                                 SdfPath("/scaleexpr/source"), 7);
    desc.nodes.push_back(scale);
    desc.terminal = SdfPath("/scaleexpr/op");
    AddBinding(desc, SdfPath("/scaleexpr/op"),
               SdfPath("/scaleexpr/Expressions/perCurve"), "$primIndex + 1",
               TfToken("float"), FloatShape(), TfToken("scale"),
               expr::Domain::Primitive, VtValue(1.0f));

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Scale with a scale expression cooks on the CPU");
    if (out.totalCvs == 0) return;
    UsdGenCurveBuffer plain;
    Check(CookSourceOnly(desc, SdfPath("/scaleexpr/source"), &plain),
          "Scale expression fixture source cooks on its own");
    if (plain.totalCvs == 0) return;
    // Strand c scales by (c + 1): strand 0 is untouched, the rest fan out.
    bool first = true;
    for (int i = 0; i < kCvs; ++i) {
        size_t const o = size_t(i);
        first = first && out.px[o] == plain.px[o] &&
            out.py[o] == plain.py[o] && out.pz[o] == plain.pz[o];
    }
    Check(first, "the unscaled expression strand is bitwise unchanged");
    bool fanned = true;
    for (size_t c = 1; c < 3; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            float const L = float(c + 1);
            fanned = fanned &&
                Near(out.px[o], roots[c][0] + (plain.px[o] - roots[c][0]) * L) &&
                Near(out.py[o], 0.0f + (plain.py[o] - 0.0f) * L) &&
                Near(out.pz[o], roots[c][2] + (plain.pz[o] - roots[c][2]) * L);
        }
    }
    Check(fanned, "a primitive scale expression fans strands per curve");
}

static void CheckFailClosed()
{
    auto cookScale = [](UsdGenParamValue const &param) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/scalefail");
        desc.defaultWidth = 0.01f;
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/scalefail/hair"), {GfVec3f(0, 0, 0)}, 4, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/scalefail/source"), SdfPath("/scalefail/hair")));
        UsdGenNodeDesc scale = OpNode("/scalefail/op", "UsdGenScale",
                                     SdfPath("/scalefail/source"), 7);
        scale.params = {param};
        desc.nodes.push_back(scale);
        desc.terminal = SdfPath("/scalefail/op");
        UsdGenCurveBuffer out;
        return Cook(desc, &out);
    };
    Check(!cookScale({TfToken("scale"), VtValue(-1.0f), false}),
          "a negative scale fails closed");
    Check(!cookScale({TfToken("scaleRandom"), VtValue(GfVec2f(-1.0f, 1.0f)),
                      false}),
          "a negative scaleRandom bound fails closed");
}

int main()
{
    CheckRegistryContracts();
    CheckScaleCook();
    CheckWidthTooCook();
    CheckIdentityCook();
    CheckScaleRandomCook();
    CheckRampCook();
    CheckConnectedExpressionCook();
    CheckFailClosed();
    std::printf("testUsdGenScale: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
