// UsdGenResample contracts, expression targets, and CPU cooks: uniform and
// keepParam resampling, mask hold, length restore, and cvCount validation.
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
static expr::ValueShape IntShape()
{
    return {expr::ScalarType::Int32, 1, 1, 1, 1, false};
}

constexpr float kTol = 1e-4f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}

static UsdGenCurveSetDesc Strands(SdfPath const &path,
                                  std::vector<std::vector<GfVec3f>> const &curves)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = UsdGenLane::Chunked;
    set.curveRole = TfToken("hair");
    size_t const n = curves.size();
    set.curveVertexCounts.assign(n, 0);
    for (size_t c = 0; c < n; ++c) {
        set.curveVertexCounts[c] = int(curves[c].size());
        for (GfVec3f const &p : curves[c]) set.points.push_back(p);
    }
    set.rest = set.points;
    set.widths.assign(set.points.size(), 0.1f);
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

static UsdGenGraphDesc ResampleDesc(
    std::vector<std::vector<GfVec3f>> const &curves,
    std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/resample");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(Strands(SdfPath("/resample/hair"), curves));
    desc.nodes.push_back(
        SourceNode(SdfPath("/resample/source"), SdfPath("/resample/hair")));
    UsdGenNodeDesc op;
    op.path = SdfPath("/resample/op");
    op.type = TfToken("UsdGenResample");
    op.seed = 7;
    op.inputs = {SdfPath("/resample/source")};
    op.params = params;
    desc.nodes.push_back(op);
    desc.terminal = SdfPath("/resample/op");
    return desc;
}

static float TotalLength(UsdGenCurveBuffer const &buf, size_t base, size_t n)
{
    float len = 0.0f;
    for (size_t i = 1; i < n; ++i) {
        float const dx = buf.px[base + i] - buf.px[base + i - 1];
        float const dy = buf.py[base + i] - buf.py[base + i - 1];
        float const dz = buf.pz[base + i] - buf.pz[base + i - 1];
        len += std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    return len;
}

static void CheckRegistry()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenResample");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenResample)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenResample)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "UsdGenResample reports its own type");
    Check(op->Record().Def().countChanges == UsdGenCountsAtLevel(2),
          "UsdGenResample has TopologyEffect CvCount");
    Check(op->Record().ReferenceInputs().empty(),
          "UsdGenResample declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(),
          "UsdGenResample emits no output primvars");
}

static void AddBinding(UsdGenGraphDesc &desc, std::string const &source,
                       TfToken const &nativeType,
                       expr::ValueShape const &shape,
                       TfToken const &destination, expr::Domain domain,
                       VtValue const &literal)
{
    SdfPath const exprPath("/resample/Expressions/ctl");
    desc.expressions.push_back(
        {exprPath, source, {{TfToken("result"), nativeType, shape}}});
    for (UsdGenNodeDesc &node : desc.nodes) {
        if (node.path != SdfPath("/resample/op")) continue;
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

static void CheckExpressionTargets()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    std::vector<std::vector<GfVec3f>> const curves = {
        {GfVec3f(0, 0, 0), GfVec3f(0, 1, 0)}};
    std::vector<UsdGenParamValue> const params = {
        {TfToken("cvCount"), VtValue(4), false}};
    UsdGenGraphDesc valid = ResampleDesc(curves, params);
    AddBinding(valid, "$value", TfToken("float"), FloatShape(),
               TfToken("mask"), expr::Domain::Point, VtValue(1.0f));
    Check(usdGenUsd::Compile(compiler, valid, &graph).ok,
          "a point-domain mask connection compiles on Resample");
    UsdGenGraphDesc groomCount = ResampleDesc(curves, params);
    AddBinding(groomCount, "4", TfToken("int"), IntShape(),
               TfToken("cvCount"), expr::Domain::Groom, VtValue(4));
    Check(usdGenUsd::Compile(compiler, groomCount, &graph).ok,
          "a groom-domain cvCount connection compiles on Resample");
    UsdGenGraphDesc pointCount = ResampleDesc(curves, params);
    AddBinding(pointCount, "$index", TfToken("int"), IntShape(),
               TfToken("cvCount"), expr::Domain::Point, VtValue(4));
    UsdGenCompileResult r = usdGenUsd::Compile(compiler, pointCount, &graph);
    Check(!r.ok, "a point-domain Resample cvCount fails closed");
    UsdGenGraphDesc foreign = ResampleDesc(curves, params);
    AddBinding(foreign, "$value", TfToken("float"), FloatShape(),
               TfToken("width"), expr::Domain::Point, VtValue(0.1f));
    UsdGenCompileResult rf = usdGenUsd::Compile(compiler, foreign, &graph);
    bool const specific = std::any_of(
        rf.errors.begin(), rf.errors.end(), [](std::string const &message) {
            return message.find("unsupported or incorrectly typed Resample") !=
                std::string::npos;
        });
    Check(!rf.ok && specific, "a destination Resample does not own fails closed");
}

// Non-uniform L: segments of length 2 and 1, total 3.
static std::vector<GfVec3f> LCurve()
{
    return {GfVec3f(0, 0, 0), GfVec3f(2, 0, 0), GfVec3f(2, 1, 0)};
}

static void CheckUniformCook()
{
    std::vector<UsdGenParamValue> const params = {
        {TfToken("cvCount"), VtValue(4), false}};
    UsdGenGraphDesc desc = ResampleDesc({LCurve()}, params);
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Resample cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 1 && out.totalCvs == 4,
          "Resample emits cvCount CVs per strand");
    // Uniform by arc length: s = 0, 1, 2, 3.
    float const ex[4] = {0.0f, 1.0f, 2.0f, 2.0f};
    float const ey[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    bool exact = true, widths = out.width.size() == 4;
    for (int i = 0; i < 4; ++i) {
        exact = exact && Near(out.px[size_t(i)], ex[i], 1e-5f) &&
            Near(out.py[size_t(i)], ey[i], 1e-5f) && out.pz[size_t(i)] == 0.0f;
        widths = widths && out.width[size_t(i)] == 0.1f;
    }
    Check(exact, "Resample uniform places CVs at equal arc steps");
    Check(widths, "Resample carries constant widths through");
    Check(out.px[0] == 0.0f && out.py[0] == 0.0f,
          "Resample keeps the root exactly fixed");
    Check(Near(TotalLength(out, 0, 4), 3.0f, 1e-4f),
          "Resample uniform preserves total length without restore");
}

static void CheckKeepParamCook()
{
    std::vector<UsdGenParamValue> const params = {
        {TfToken("cvCount"), VtValue(4), false},
        {TfToken("distribution"), VtValue(TfToken("keepParam")), false}};
    UsdGenGraphDesc desc = ResampleDesc({LCurve()}, params);
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Resample keepParam cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // Index space: f = 0, 2/3, 4/3, 2.
    float const ex[4] = {0.0f, 4.0f / 3.0f, 2.0f, 2.0f};
    float const ey[4] = {0.0f, 0.0f, 1.0f / 3.0f, 1.0f};
    bool exact = true;
    for (int i = 0; i < 4; ++i)
        exact = exact && Near(out.px[size_t(i)], ex[i], 1e-5f) &&
            Near(out.py[size_t(i)], ey[i], 1e-5f);
    Check(exact, "Resample keepParam interpolates in input-index space");
}

static void CheckMaskHoldCook()
{
    std::vector<UsdGenParamValue> const params = {
        {TfToken("cvCount"), VtValue(4), false},
        {TfToken("mask"), VtValue(0.0f), false}};
    UsdGenGraphDesc desc = ResampleDesc({LCurve()}, params);
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Resample mask hold cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // mask = 0 holds the nearest input CV: hold(j) = round(j*2/3).
    float const ex[4] = {0.0f, 2.0f, 2.0f, 2.0f};
    float const ey[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    bool exact = true;
    for (int i = 0; i < 4; ++i)
        exact = exact && out.px[size_t(i)] == ex[i] &&
            out.py[size_t(i)] == ey[i];
    Check(exact, "Resample mask 0 holds the nearest input CV");
}

static void CheckRestoreCook()
{
    std::vector<UsdGenParamValue> const params = {
        {TfToken("cvCount"), VtValue(3), false},
        {TfToken("restoreSegmentLengths"), VtValue(true), false}};
    UsdGenGraphDesc desc = ResampleDesc({LCurve()}, params);
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Resample restore cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(Near(TotalLength(out, 0, 3), 3.0f, 1e-4f),
          "Resample restore rescales to the input total length");
    Check(out.px[0] == 0.0f && out.py[0] == 0.0f,
          "Resample restore keeps the root exactly fixed");
}

static void CheckValidation()
{
    std::vector<UsdGenParamValue> const bad = {
        {TfToken("cvCount"), VtValue(0), false}};
    UsdGenCurveBuffer out;
    Check(!Cook(ResampleDesc({LCurve()}, bad), &out),
          "Resample cvCount 0 fails closed");
    std::vector<UsdGenParamValue> const badMode = {
        {TfToken("cvCount"), VtValue(4), false},
        {TfToken("distribution"), VtValue(TfToken("arc")), false}};
    Check(!Cook(ResampleDesc({LCurve()}, badMode), &out),
          "Resample unknown distribution fails closed");
}

int main()
{
    CheckRegistry();
    CheckExpressionTargets();
    CheckUniformCook();
    CheckKeepParamCook();
    CheckMaskHoldCook();
    CheckRestoreCook();
    CheckValidation();
    std::printf("testUsdGenResample: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
