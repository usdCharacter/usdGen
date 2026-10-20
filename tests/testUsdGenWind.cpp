// UsdGenWind operator contract and CPU cooks.
//
// Covers the time-dependent force field along the normalized direction:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/
//       OutputPrimvars plus the Topology/Value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): identity at zero effect, the constant-deflection formula,
//       root locking, direction normalization, stiffness resistance and its
//       ramp, gust determinism per time+seed, gust animation across time,
//       time-independence at gustStrength 0, and a connected-expression cook
//       for the per-strand constStrength;
//   (3) every edge fails closed (zero/non-finite direction, non-finite
//       strengths/stiffness/mask inputs, mistyped or non-finite
//       stiffness:knots, unknown stiffness:interpolation);
//   (4) CUDA-backend explicit rejection: Wind is CPU-only (no capability
//       matrix row, like Clump), so ValidateCudaGraph refuses a Wind graph.
#include "usdGen/compiler.h"
#include "usdGen/clumpMotion.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
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

static bool CookAt(UsdGenGraphDesc const &desc, double time, UsdGenCurveBuffer *out)
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
    ctx.time = time;
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

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out)
{
    return CookAt(desc, 0.0, out);
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

static bool PointsEqual(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCurves == b.totalCurves && a.totalCvs == b.totalCvs &&
        a.px == b.px && a.py == b.py && a.pz == b.pz;
}

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenWind");
    Check(registry.HasKernel(type), "HasKernel(UsdGenWind)");
    std::unique_ptr<UsdGenOp> op = registry.Create(type);
    Check(op != nullptr, "Create(UsdGenWind)");
    if (!op) return;
    Check(op->Type() == type, "Wind reports its own type");
    Check(op->TopologyEffect() == UsdGenTopoFx::None,
          "Wind has TopologyEffect None");
    Check(op->ReferenceInputs().empty(),
          "Wind declares no reference inputs");
    Check(op->OutputPrimvars().empty(), "Wind emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "Wind touches the points plane only");
    Check(op->ReadsTime() && registry.ReadsTime(type) &&
              !registry.ReadsTime(TfToken("UsdGenCurveSource")),
          "Wind declares a native time dependency; static source does not");
    std::set<std::string> topo, value;
    for (TfToken const &t : op->TopologyParameters()) topo.insert(t.GetString());
    for (TfToken const &t : op->ValueParameters()) value.insert(t.GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed"},
          "Wind topology parameters are input/surface/seed");
    Check(value == std::set<std::string>{"enabled", "mask", "direction",
                                         "constStrength", "gustStrength",
                                         "billowLowStrength", "billowLowFrequency",
                                         "billowLowRate", "billowHighStrength",
                                         "billowHighFrequency", "billowHighRate",
                                         "stiffness", "stiffness:knots",
                                         "stiffness:interpolation"},
          "Wind value parameters include both native billow bands");
}

// --- (2) CPU cooks -----------------------------------------------------
// With direction (0,0,1), gust 0, stiffness 0 and mask 1 the field is
// d = constStrength * hairT along +Z; roots (t = 0) never move.

static UsdGenGraphDesc WindDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/wind");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/wind/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/wind/source"), SdfPath("/wind/hair")));
    UsdGenNodeDesc wind = OpNode("/wind/op", "UsdGenWind",
                                 SdfPath("/wind/source"), 7);
    wind.params = params;
    desc.nodes.push_back(wind);
    desc.terminal = SdfPath("/wind/op");
    return desc;
}

static void CheckIdentityCook()
{
    UsdGenGraphDesc desc = WindDesc({{TfToken("constStrength"), VtValue(0.0f), false},
                                     {TfToken("gustStrength"), VtValue(0.0f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Wind cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/wind/source"), &plain),
          "Wind fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "Wind preserves the strand topology");
    Check(PointsEqual(out, plain),
          "Wind with zero strengths passes points through bit-for-bit");

    // Full stiffness with no knots (the empty-knots identity ramp, exactly 1.0)
    // resists everything.
    UsdGenGraphDesc stiff = WindDesc({{TfToken("constStrength"), VtValue(2.0f), false},
                                      {TfToken("stiffness"), VtValue(1.0f), false}});
    UsdGenCurveBuffer held;
    Check(Cook(stiff, &held), "Wind with full stiffness cooks");
    if (held.totalCvs == 0) return;
    Check(PointsEqual(held, plain),
          "Wind with stiffness 1 holds every CV bit-for-bit");

    UsdGenGraphDesc masked = WindDesc({{TfToken("constStrength"), VtValue(2.0f), false},
                                       {TfToken("mask"), VtValue(0.0f), false}});
    UsdGenCurveBuffer muted;
    Check(Cook(masked, &muted), "Wind with mask 0 cooks");
    if (muted.totalCvs == 0) return;
    Check(PointsEqual(muted, plain),
          "Wind with mask 0 passes points through bit-for-bit");
}

static void CheckConstantCook()
{
    // d = 2 * hairT along (0,0,1); the source strands lie in z = 0.
    UsdGenGraphDesc desc = WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                                     {TfToken("constStrength"), VtValue(2.0f), false},
                                     {TfToken("gustStrength"), VtValue(0.0f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Wind constant deflection cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/wind/source"), &plain),
          "Wind constant-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool hairTLinear = plain.hairT.size() == 12;
    for (size_t o = 0; o < 12 && hairTLinear; ++o)
        hairTLinear = Near(plain.hairT[o], float(o % 6) / 5.0f, 1e-6f);
    Check(hairTLinear, "Wind fixture hairT runs 0..1 linearly per strand");
    bool profile = true, sidesExact = true, rootsExact = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            float const t = plain.hairT[o];
            profile = profile && Near(out.pz[o], plain.pz[o] + 2.0f * t, 1e-5f);
            sidesExact = sidesExact && out.px[o] == plain.px[o] &&
                out.py[o] == plain.py[o];
        }
        rootsExact = rootsExact && out.px[c * 6] == plain.px[c * 6] &&
            out.py[c * 6] == plain.py[c * 6] && out.pz[c * 6] == plain.pz[c * 6];
    }
    Check(profile, "Wind deflects every CV by constStrength * hairT");
    Check(sidesExact, "Wind moves nothing off the direction axis");
    Check(rootsExact, "Wind locks roots bit-for-bit");

    // The direction normalizes: (0,0,5) cooks exactly like (0,0,1).
    UsdGenGraphDesc scaled = WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 5)), false},
                                       {TfToken("constStrength"), VtValue(2.0f), false},
                                       {TfToken("gustStrength"), VtValue(0.0f), false}});
    UsdGenCurveBuffer rescaled;
    Check(Cook(scaled, &rescaled), "Wind with a scaled direction cooks");
    if (rescaled.totalCvs == 0) return;
    Check(PointsEqual(out, rescaled),
          "Wind normalizes its direction before deflecting");

    // gustStrength 0 is pure constant deflection: the cook is time-independent.
    UsdGenCurveBuffer late;
    Check(CookAt(desc, 240.0, &late), "Wind constant cook runs at a late time");
    if (late.totalCvs == 0) return;
    Check(PointsEqual(out, late),
          "Wind with gustStrength 0 is identical at every time");
}

static void CheckStiffnessRampCook()
{
    // knots [(0,0),(1,1)] linear with stiffness 1: flex = 1 - t, so
    // d = 2 * t * (1 - t): the tip holds and the middle bends most.
    UsdGenGraphDesc desc = WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                                     {TfToken("constStrength"), VtValue(2.0f), false},
                                     {TfToken("gustStrength"), VtValue(0.0f), false},
                                     {TfToken("stiffness"), VtValue(1.0f), false},
                                     {TfToken("stiffness:interpolation"),
                                      VtValue(TfToken("linear")), false}});
    {
        VtVec2fArray knots(2);
        knots[0] = GfVec2f(0.0f, 0.0f);
        knots[1] = GfVec2f(1.0f, 1.0f);
        desc.nodes.back().params.push_back(
            {TfToken("stiffness:knots"), VtValue(knots), false});
    }
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Wind stiffness-ramp cook runs");
    Check(CookSourceOnly(desc, SdfPath("/wind/source"), &plain),
          "Wind ramp-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool profile = true, tipExact = true;
    for (size_t c = 0; c < 2; ++c) {
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            float const t = plain.hairT[o];
            profile = profile && Near(out.pz[o], plain.pz[o] + 2.0f * t * (1.0f - t), 1e-5f);
        }
        size_t const tip = c * 6 + 5;
        tipExact = tipExact && out.px[tip] == plain.px[tip] &&
            out.py[tip] == plain.py[tip] &&
            Near(out.pz[tip], plain.pz[tip], 1e-6f);
    }
    Check(profile, "Wind shapes deflection by 1 - stiffness * ramp(t)");
    Check(tipExact, "Wind holds the tip at full tip stiffness");
}

static void CheckGustDeterminismAndAnimation()
{
    UsdGenGraphDesc desc = WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                                     {TfToken("constStrength"), VtValue(0.0f), false},
                                     {TfToken("gustStrength"), VtValue(2.0f), false}});
    UsdGenCurveBuffer early, again, late, plain;
    Check(CookAt(desc, 0.0, &early), "Wind gust cook runs at time 0");
    Check(CookAt(desc, 0.0, &again), "Wind gust cook reruns at time 0");
    Check(CookAt(desc, 240.0, &late), "Wind gust cook runs at a late time");
    Check(CookSourceOnly(desc, SdfPath("/wind/source"), &plain),
          "Wind gust-cook source cooks on its own");
    if (early.totalCvs == 0 || again.totalCvs == 0 || late.totalCvs == 0 ||
        plain.totalCvs == 0) return;
    Check(PointsEqual(early, again),
          "Wind gusts are bit-for-bit deterministic per time+seed");
    bool moved = false;
    for (size_t o = 0; o < 12; ++o)
        moved = moved || early.pz[o] != late.pz[o];
    Check(moved, "Wind gusts animate across time");
    bool rootsExact = true;
    for (size_t c = 0; c < 2; ++c)
        rootsExact = rootsExact && early.pz[c * 6] == plain.pz[c * 6] &&
            late.pz[c * 6] == plain.pz[c * 6];
    Check(rootsExact, "Wind gusts never move roots");
}

static void CheckLiveTimeOnOneGraph()
{
    UsdGenGraphDesc desc = WindDesc({
        {TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
        {TfToken("gustStrength"), VtValue(0.0f), false},
        {TfToken("billowLowStrength"), VtValue(0.1f), false},
        {TfToken("billowLowRate"), VtValue(0.24f), false}});
    // An unchanged downstream node must also copy newly animated points.
    desc.nodes.push_back(OpNode("/wind/width", "UsdGenWidth",
                                SdfPath("/wind/op"), 0));
    desc.terminal = SdfPath("/wind/width");
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(desc, &graph);
    Check(compiled.ok, "Live Wind graph compiles");
    if (!compiled.ok) return;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    auto run = [&](double frame) {
        ctx.time = frame;
        return scheduler.Run(graph, ctx, uint64_t(frame) + 1);
    };
    auto first = run(0.0);
    Check(!first.diagnostics.HasErrors(), "Live Wind first frame cooks");
    std::vector<float> const firstZ(graph.Output().pz.begin(), graph.Output().pz.end());
    auto same = run(0.0);
    Check(!same.diagnostics.HasErrors(), "Live Wind same frame reuses output");
    bool noWork = true;
    for (auto const& stat : same.nodeStats) noWork &= stat.chunksEvaluated == 0;
    Check(noWork, "Unchanged time skips all Wind graph chunks");
    auto later = run(24.0);
    Check(!later.diagnostics.HasErrors(), "Live Wind later frame cooks");
    bool animated = false;
    for (size_t i = 0; i < firstZ.size(); ++i)
        animated |= firstZ[i] != graph.Output().pz[i];
    Check(animated, "Live Wind billow changes on the same compiled graph");
    uint64_t sourceChunks = 0, windChunks = 0, widthChunks = 0;
    for (auto const& stat : later.nodeStats) {
        auto const& type = graph.Node(stat.id).desc->type;
        if (type == TfToken("UsdGenCurveSource")) sourceChunks += stat.chunksEvaluated;
        if (type == TfToken("UsdGenWind")) windChunks += stat.chunksEvaluated;
        if (type == TfToken("UsdGenWidth")) widthChunks += stat.chunksEvaluated;
    }
    Check(sourceChunks == 0 && windChunks > 0 && widthChunks > 0,
          "Time dirt evaluates Wind and downstream, reusing static source");
    auto repeated = run(24.0);
    noWork = true;
    for (auto const& stat : repeated.nodeStats) noWork &= stat.chunksEvaluated == 0;
    Check(noWork, "Repeated frame reuses Wind and downstream chunks");
    std::vector<float> const priorRateZ(graph.Output().pz.begin(), graph.Output().pz.end());
    desc.timeCodesPerSecond = 12.0;
    Check(compiler.Recompile(desc, &graph).ok,
          "Live Wind timeCodesPerSecond edit recompiles");
    ctx.desc = &graph.Desc();
    auto rateChanged = run(24.0);
    animated = false;
    for (size_t i = 0; i < priorRateZ.size(); ++i)
        animated |= priorRateZ[i] != graph.Output().pz[i];
    Check(!rateChanged.diagnostics.HasErrors() && animated,
          "Wind reevaluates when seconds-per-frame changes at fixed frame");
    UsdGenCurveBuffer freshRate;
    Check(CookAt(desc, 24.0, &freshRate) &&
              PointsEqual(graph.Output(), freshRate),
          "Same-graph rate edit matches a fresh cook at the same frame");
}

static UsdGenGraphDesc BillowDesc(float frequency, float rate, bool high)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/wind");
    desc.timeCodesPerSecond = 1.0;
    std::vector<GfVec3f> roots;
    for (int i = 0; i <= 40; ++i)
        roots.emplace_back(0.0f, 0.0f, float(i) * 0.025f);
    desc.curveSets.push_back(StraightStrands(SdfPath("/wind/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(SourceNode(SdfPath("/wind/source"), SdfPath("/wind/hair")));
    UsdGenNodeDesc wind = OpNode("/wind/op", "UsdGenWind", SdfPath("/wind/source"), 7);
    wind.params = {{TfToken(high ? "billowHighStrength" : "billowLowStrength"),
                    VtValue(1.0f), false},
                   {TfToken(high ? "billowHighFrequency" : "billowLowFrequency"),
                    VtValue(frequency), false},
                   {TfToken(high ? "billowHighRate" : "billowLowRate"),
                    VtValue(rate), false}};
    desc.nodes.push_back(wind);
    desc.terminal = SdfPath("/wind/op");
    return desc;
}

static int TipSignChanges(UsdGenCurveBuffer const &out)
{
    int changes = 0;
    float previous = out.pz[5] - out.rest[5][2];
    for (uint32_t c = 1; c < out.totalCurves; ++c) {
        float const current = out.pz[size_t(c) * 6 + 5] - out.rest[size_t(c) * 6 + 5][2];
        if (previous * current < 0.0f) ++changes;
        previous = current;
    }
    return changes;
}

static void CheckBillowBands()
{
    UsdGenGraphDesc low = BillowDesc(0.5f, 0.25f, false);
    UsdGenGraphDesc high = BillowDesc(4.0f, 1.0f, true);
    UsdGenGraphDesc combined = low;
    combined.nodes.back().params.push_back(
        {TfToken("billowHighStrength"), VtValue(1.0f), false});
    combined.nodes.back().params.push_back(
        {TfToken("billowHighFrequency"), VtValue(4.0f), false});
    combined.nodes.back().params.push_back(
        {TfToken("billowHighRate"), VtValue(1.0f), false});
    UsdGenCurveBuffer lo, loAgain, loQuarter, loCycle, hi, hiQuarter, both, plain;
    Check(CookAt(low, 0.0, &lo) && CookAt(low, 0.0, &loAgain) &&
          CookAt(low, 1.0, &loQuarter) && CookAt(low, 4.0, &loCycle),
          "Low billow cooks deterministically across time");
    Check(CookAt(high, 0.0, &hi) && CookAt(high, 0.25, &hiQuarter),
          "High billow cooks at its faster temporal rate");
    Check(CookAt(combined, 0.0, &both), "Low and high billow cook together");
    Check(CookSourceOnly(low, SdfPath("/wind/source"), &plain),
          "Billow fixture source cooks");
    if (lo.totalCvs == 0 || hi.totalCvs == 0 || both.totalCvs == 0 ||
        plain.totalCvs == 0) return;
    Check(PointsEqual(lo, loAgain), "Billow repeats bit-for-bit at fixed seed/time");
    Check(lo.totalCurves == plain.totalCurves && lo.totalCvs == plain.totalCvs &&
          hi.totalCurves == plain.totalCurves && hi.totalCvs == plain.totalCvs,
          "Both billow bands preserve topology");
    bool rootsExact = true, curved = false, coherent = true, lowTimeMoved = false;
    for (uint32_t c = 0; c < lo.totalCurves; ++c) {
        size_t const root = size_t(c) * 6;
        rootsExact = rootsExact && lo.pz[root] == plain.pz[root] &&
            hi.pz[root] == plain.pz[root] && loQuarter.pz[root] == plain.pz[root] &&
            hiQuarter.pz[root] == plain.pz[root];
        float const d1 = lo.pz[root + 1] - plain.pz[root + 1];
        float const d4 = lo.pz[root + 4] - plain.pz[root + 4];
        if (std::fabs(d4) > 1e-3f) curved = curved || Near(d4, 16.0f * d1, 2e-4f);
        lowTimeMoved = lowTimeMoved || !Near(lo.pz[root + 5], loQuarter.pz[root + 5]);
        if (c > 0) {
            float const previous = lo.pz[root - 1] - plain.pz[root - 1];
            float const current = lo.pz[root + 5] - plain.pz[root + 5];
            coherent = coherent && std::fabs(current - previous) < 0.1f;
        }
    }
    Check(rootsExact, "Both billow bands anchor every root at every sampled time");
    Check(curved, "Low billow has a nonlinear shaft profile");
    Check(coherent, "Neighboring strands share a smooth rest-space wave");
    Check(TipSignChanges(hi) >= TipSignChanges(lo) + 4,
          "High spatial frequency yields more waves across the patch");
    bool additive = true;
    for (size_t i = 0; i < lo.totalCvs; ++i)
        additive = additive && Near(both.pz[i], lo.pz[i] + hi.pz[i] - plain.pz[i], 2e-5f);
    Check(additive, "Low and high billows add on the same curved strands");
    Check(lowTimeMoved && !PointsEqual(hi, hiQuarter),
          "Both bands animate at their authored rates");
    bool cycle = true;
    for (size_t i = 0; i < lo.totalCvs; ++i)
        cycle = cycle && Near(lo.pz[i], loCycle.pz[i], 2e-5f);
    Check(cycle, "Low band returns after one authored cycle");
}

static void CheckBillowExpressions()
{
    UsdGenGraphDesc connected = BillowDesc(0.5f, 0.0f, false);
    AddBinding(connected, SdfPath("/wind/op"), SdfPath("/wind/Expressions/strength"),
               "$primIndex + 1.0", TfToken("float"), FloatShape(),
               TfToken("billowLowStrength"), expr::Domain::Primitive, VtValue(1.0f));
    AddBinding(connected, SdfPath("/wind/op"), SdfPath("/wind/Expressions/frequency"),
               "2.0", TfToken("float"), FloatShape(),
               TfToken("billowLowFrequency"), expr::Domain::Groom, VtValue(0.5f));
    UsdGenGraphDesc reference = BillowDesc(2.0f, 0.0f, false);
    UsdGenCurveBuffer out, expected, plain;
    Check(Cook(connected, &out), "Billow accepts native strength and frequency expressions");
    Check(Cook(reference, &expected) &&
          CookSourceOnly(reference, SdfPath("/wind/source"), &plain),
          "Billow expression reference cooks");
    if (out.totalCvs == 0 || expected.totalCvs == 0 || plain.totalCvs == 0) return;
    bool matched = true;
    for (uint32_t c = 0; c < out.totalCurves; ++c) {
        size_t const tip = size_t(c) * 6 + 5;
        float const delta = expected.pz[tip] - plain.pz[tip];
        matched = matched && Near(out.pz[tip] - plain.pz[tip],
                                  delta * float(c + 1), 2e-4f);
    }
    Check(matched, "Billow reads connected groom frequency and per-root strength");
}

static UsdGenGraphDesc ClumpWindDesc(float amount, float mask, int levels = 1)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/wind");
    desc.timeCodesPerSecond = 24.0;
    std::vector<GfVec3f> roots;
    for (int z = 0; z < 8; ++z)
        for (int x = 0; x < 8; ++x)
            roots.emplace_back(float(x) / 7.0f, 0.0f, float(z) / 7.0f);
    desc.curveSets.push_back(StraightStrands(SdfPath("/wind/hair"), roots, 6, 0.2f));
    desc.nodes.push_back(SourceNode(SdfPath("/wind/source"), SdfPath("/wind/hair")));
    UsdGenNodeDesc clump = OpNode("/wind/clump", "UsdGenClump", SdfPath("/wind/source"), 19);
    clump.params = {{TfToken("clump:levels"), VtValue(levels), false},
                    {TfToken("clump:density"), VtValue(1.0f), false},
                    {TfToken("clump:size"), VtValue(10.0f), false},
                    {TfToken("clump:amount"), VtValue(amount), false},
                    {TfToken("clump:tightnessReduction"), VtValue(1.0f), false},
                    {TfToken("preserveLength"), VtValue(0.0f), false},
                    {TfToken("mask"), VtValue(mask), false}};
    desc.nodes.push_back(clump);
    UsdGenNodeDesc wind = OpNode("/wind/op", "UsdGenWind", SdfPath("/wind/clump"), 7);
    wind.params = {{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                   {TfToken("gustStrength"), VtValue(1.0f), false}};
    desc.nodes.push_back(wind);
    desc.terminal = SdfPath("/wind/op");
    return desc;
}

static UsdGenGraphDesc WithoutClump(UsdGenGraphDesc desc)
{
    desc.nodes.erase(desc.nodes.begin() + 1);
    desc.nodes.back().inputs = {SdfPath("/wind/source")};
    return desc;
}

static UsdGenGraphDesc OnlyClump(UsdGenGraphDesc desc)
{
    desc.nodes.pop_back();
    desc.terminal = SdfPath("/wind/clump");
    return desc;
}

static UsdGenPlane const *Extra(UsdGenCurveBuffer const &buffer, char const *name)
{
    for (UsdGenPlane const &plane : buffer.extraCurve)
        if (plane.name == TfToken(name)) return &plane;
    return nullptr;
}

static void CheckClumpMotion()
{
    UsdGenGraphDesc zero = ClumpWindDesc(0.0f, 1.0f);
    UsdGenCurveBuffer legacy, noAmount, noMask;
    Check(CookAt(WithoutClump(zero), 12.0, &legacy) &&
          CookAt(zero, 12.0, &noAmount),
          "Wind zero-clump-weight and legacy cooks run");
    zero.nodes[1].params[3].value = VtValue(1.0f);
    zero.nodes[1].params[6].value = VtValue(0.0f);
    Check(CookAt(zero, 12.0, &noMask), "Wind muted-Clump cook runs");
    if (legacy.totalCvs == 0 || noAmount.totalCvs == 0 || noMask.totalCvs == 0) return;
    Check(PointsEqual(legacy, noAmount) && PointsEqual(legacy, noMask),
          "Zero amount or mask preserves exact independent Wind output");

    UsdGenGraphDesc full = ClumpWindDesc(1.0f, 1.0f);
    UsdGenCurveBuffer clumped, base, repeated, later;
    Check(CookAt(full, 12.0, &clumped) && CookAt(OnlyClump(full), 12.0, &base) &&
          CookAt(full, 12.0, &repeated) && CookAt(full, 24.0, &later),
          "Native Clump-to-Wind cooks at two times");
    if (clumped.totalCvs == 0 || base.totalCvs == 0) return;
    Check(PointsEqual(clumped, repeated) && !PointsEqual(clumped, later),
          "Clump-coherent Wind is deterministic and animated");
    UsdGenPlane const *id = Extra(clumped, "clumpId_0");
    bool shared = id && id->i.size() == clumped.totalCurves;
    bool foundPair = false, rootsLocked = true;
    for (uint32_t c = 0; c < clumped.totalCurves; ++c) {
        size_t const root = size_t(c) * 6;
        rootsLocked = rootsLocked && clumped.pz[root] == base.pz[root];
        for (uint32_t other = 0; other < c && shared; ++other) {
            if (id->i[c] < 0 || id->i[c] != id->i[other]) continue;
            float const a = clumped.pz[root + 5] - base.pz[root + 5];
            size_t const otherTip = size_t(other) * 6 + 5;
            float const b = clumped.pz[otherTip] - base.pz[otherTip];
            shared = shared && Near(a, b, 2e-5f);
            foundPair = true;
        }
    }
    Check(foundPair && shared, "Clump members share the native gust field");
    Check(rootsLocked, "Clump-informed Wind still locks every root");

    // Keep the same compiled graph and scheduler alive across a value edit.
    // A zero-cohesion capture retains its binding for reuse, then Wind must
    // switch to group motion when the Clump amount becomes positive.
    UsdGenGraphDesc live = ClumpWindDesc(0.0f, 1.0f);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    bool const compiled = compiler.Compile(live, &graph).ok;
    Check(compiled, "Live Clump-to-Wind graph compiles");
    if (!compiled) return;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.time = 12.0;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 1).diagnostics.HasErrors(),
          "Live zero-cohesion Wind cook runs");
    UsdGenCurveBuffer const &zeroLive = graph.Output();
    VtFloatArray const zeroPx(zeroLive.px.begin(), zeroLive.px.end());
    VtFloatArray const zeroPy(zeroLive.py.begin(), zeroLive.py.end());
    VtFloatArray const zeroPz(zeroLive.pz.begin(), zeroLive.pz.end());
    Check(zeroPx == legacy.px && zeroPy == legacy.py && zeroPz == legacy.pz,
          "Live zero-cohesion cook matches exact legacy output");
    live.nodes[1].params[3].value = VtValue(1.0f);
    bool const recompiled = compiler.Recompile(live, &graph).ok;
    Check(recompiled, "Clump amount edit recompiles live graph");
    if (!recompiled) return;
    context.desc = &graph.Desc();
    Check(!scheduler.Run(graph, context, 2).diagnostics.HasErrors(),
          "Live positive-cohesion Wind recook runs");
    UsdGenCurveBuffer positiveLive = graph.Output();
    Check(positiveLive.px != zeroPx || positiveLive.py != zeroPy ||
          positiveLive.pz != zeroPz,
          "Zero-to-positive live edit switches to clump motion");
    Check(PointsEqual(positiveLive, clumped),
          "Live positive-cohesion recook matches fresh positive cook");
    UsdGenNodeId const clumpId = graph.NodeIdForPath(SdfPath("/wind/clump"));
    UsdGenPlane const *liveWeight = nullptr;
    for (UsdGenPlane const &plane : graph.Node(clumpId).buffer.extraCv)
        if (plane.name == TfToken("clumpWeight_0")) liveWeight = &plane;
    bool positiveWeight = liveWeight && liveWeight->f.size() == graph.Node(clumpId).buffer.totalCvs;
    if (positiveWeight)
        for (float w : liveWeight->f) positiveWeight = positiveWeight && w > 0.0f;
    Check(positiveWeight, "Live amount edit refreshes native Clump cohesion weights");
}

static void CheckClumpHierarchyMotion()
{
    UsdGenGraphDesc desc = ClumpWindDesc(1.0f, 1.0f, 2);
    // Leave some fine-level influence unclaimed so the composed capture
    // retains the coarse parent field as well as a finer flutter component.
    desc.nodes[1].params[4].value = VtValue(0.5f);
    desc.nodes.back().params = {
        {TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
        {TfToken("billowLowStrength"), VtValue(1.0f), false},
        {TfToken("billowLowFrequency"), VtValue(0.7f), false},
        {TfToken("billowHighStrength"), VtValue(1.0f), false},
        {TfToken("billowHighFrequency"), VtValue(2.3f), false}};
    UsdGenCurveBuffer both, base, low, high;
    Check(Cook(desc, &both) && Cook(OnlyClump(desc), &base),
          "Two-level native Clump-to-Wind graph cooks");
    if (both.totalCvs == 0 || base.totalCvs == 0) return;
    UsdGenPlane const *coarse = Extra(both, "clumpId_0");
    UsdGenPlane const *fine = Extra(both, "clumpId_1");
    Check(coarse && fine && coarse->i.size() == both.totalCurves &&
          fine->i.size() == both.totalCurves,
          "Both numeric clump levels survive through Wind");
    if (!coarse || !fine || fine->i.size() != both.totalCurves) return;
    desc.nodes.back().params[3].value = VtValue(0.0f);
    Check(Cook(desc, &low), "Coarse low-billow component cooks");
    desc.nodes.back().params[1].value = VtValue(0.0f);
    desc.nodes.back().params[3].value = VtValue(1.0f);
    Check(Cook(desc, &high), "Fine high-billow component cooks");
    if (low.totalCvs == 0 || high.totalCvs == 0) return;
    bool sameCoarse = true, sameFine = true;
    bool foundFinePair = false, foundDistinctFine = false;
    for (uint32_t c = 0; c < both.totalCurves; ++c) {
        size_t const tip = size_t(c) * 6 + 5;
        for (uint32_t other = 0; other < c; ++other) {
            size_t const otherTip = size_t(other) * 6 + 5;
            if (coarse->i[c] >= 0 && coarse->i[c] == coarse->i[other]) {
                float const a = low.pz[tip] - base.pz[tip];
                float const b = low.pz[otherTip] - base.pz[otherTip];
                sameCoarse = sameCoarse && Near(a, b, 3e-5f);
            }
            if (fine->i[c] >= 0 && fine->i[c] == fine->i[other]) {
                foundFinePair = true;
                float const a = high.pz[tip] - base.pz[tip];
                float const b = high.pz[otherTip] - base.pz[otherTip];
                sameFine = sameFine && Near(a, b, 3e-5f);
            } else if (fine->i[c] >= 0 && fine->i[other] >= 0) {
                float const a = high.pz[tip] - base.pz[tip];
                float const b = high.pz[otherTip] - base.pz[otherTip];
                foundDistinctFine = foundDistinctFine || std::fabs(a - b) > 1e-3f;
            }
        }
    }
    Check(sameCoarse, "Siblings retain their broad parent motion");
    Check(foundFinePair && sameFine && foundDistinctFine,
          "Fine groups flutter together and differ from neighboring groups");
}

static void CheckCurvedRaggedClumpMotion()
{
    UsdGenGraphDesc desc = ClumpWindDesc(1.0f, 1.0f);
    UsdGenCurveSetDesc &hair = desc.curveSets.front();
    hair = StraightStrands(SdfPath("/wind/hair"),
        {GfVec3f(0.0f, 0.0f, 0.0f), GfVec3f(0.1f, 0.0f, 0.05f),
         GfVec3f(0.2f, 0.0f, 0.1f)}, 6, 0.2f);
    hair.curveVertexCounts = {4, 6, 5};
    hair.points.clear();
    for (int c = 0; c < 3; ++c) {
        GfVec3f const root(float(c) * 0.1f, 0.0f, float(c) * 0.05f);
        for (int i = 0; i < hair.curveVertexCounts[size_t(c)]; ++i) {
            float const t = float(i) / float(hair.curveVertexCounts[size_t(c)] - 1);
            hair.points.push_back(root + GfVec3f(float(c + 1) * t * t,
                                                   t * float(c + 1) * 0.4f,
                                                   float(c + 2) * t * t * 0.1f));
        }
    }
    hair.rest = hair.points;
    hair.widths.assign(hair.points.size(), 0.1f);
    desc.nodes[1].params[1].value = VtValue(0.0f); // one parent center
    desc.nodes.back().params = {
        {TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
        {TfToken("billowHighStrength"), VtValue(1.0f), false},
        {TfToken("billowHighFrequency"), VtValue(1.7f), false}};
    UsdGenCurveBuffer moved, base;
    Check(Cook(desc, &moved) && Cook(OnlyClump(desc), &base),
          "Ragged curved Clump-to-Wind graph cooks");
    if (moved.totalCvs == 0 || base.totalCvs == 0) return;
    UsdGenPlane const *id = Extra(moved, "clumpId_0");
    Check(id && id->i.size() == 3 && id->i[0] >= 0 &&
          id->i[0] == id->i[1] && id->i[1] == id->i[2],
          "Ragged curved strands share one clump");
    if (!id || id->i.size() != 3) return;
    bool sharedTip = true;
    for (int c = 1; c < 3; ++c) {
        size_t const tip = size_t(moved.cvOffsets[size_t(c + 1)] - 1);
        size_t const firstTip = size_t(moved.cvOffsets[1] - 1);
        sharedTip = sharedTip && Near(moved.pz[tip] - base.pz[tip],
            moved.pz[firstTip] - base.pz[firstTip], 3e-5f);
    }
    Check(sharedTip,
          "Full cohesion shares high-billow phase despite ragged, curved rest shapes");
}

static void CheckGuideClumpInheritance()
{
    // Guide references are C3 curve sets in today's compiler. First cook a
    // real Clump, then author that result as the guide set consumed below.
    UsdGenGraphDesc clumpDesc = OnlyClump(ClumpWindDesc(1.0f, 1.0f, 2));
    clumpDesc.nodes[1].params.push_back(
        {TfToken("clump:method"), VtValue(TfToken("extrudeAndBlend")), false});
    UsdGenCurveBuffer clumpedGuides;
    Check(Cook(clumpDesc, &clumpedGuides), "Clump produces guide source metadata");
    if (clumpedGuides.totalCvs == 0) return;
    UsdGenClumpMotion sourceMotion;
    std::string sourceError;
    Check(UsdGenBuildClumpMotion(clumpedGuides, &sourceMotion, &sourceError),
          "Clump guide source has valid native groups");
    UsdGenGraphDesc desc = clumpDesc;
    desc.nodes.pop_back();
    UsdGenCurveSetDesc guides = desc.curveSets.front();
    guides.path = SdfPath("/wind/guides");
    guides.role = UsdGenRole::Reference;
    guides.curveRole = TfToken("guide");
    guides.points.resize(clumpedGuides.totalCvs);
    for (uint32_t cv = 0; cv < clumpedGuides.totalCvs; ++cv)
        guides.points[cv] = GfVec3f(clumpedGuides.px[cv], clumpedGuides.py[cv],
                                    clumpedGuides.pz[cv]);
    guides.rest = clumpedGuides.rest;
    guides.widths = clumpedGuides.width;
    for (UsdGenPlane const &plane : clumpedGuides.extraCurve) {
        UsdGenAuthoredPlaneDesc authored;
        authored.name = plane.name;
        authored.type = plane.type == TfToken("int")
            ? UsdGenAuthoredPlaneType::Int32 : UsdGenAuthoredPlaneType::Float32;
        authored.domain = UsdGenAuthoredPlaneDomain::Primitive;
        authored.arity = plane.arity;
        authored.intValues = plane.i;
        authored.floatValues = plane.f;
        guides.authoredPlanes.push_back(std::move(authored));
    }
    for (UsdGenPlane const &plane : clumpedGuides.extraCv) {
        UsdGenAuthoredPlaneDesc authored;
        authored.name = plane.name;
        authored.type = plane.type == TfToken("int")
            ? UsdGenAuthoredPlaneType::Int32 : UsdGenAuthoredPlaneType::Float32;
        authored.domain = UsdGenAuthoredPlaneDomain::Point;
        authored.arity = plane.arity;
        authored.intValues = plane.i;
        authored.floatValues = plane.f;
        guides.authoredPlanes.push_back(std::move(authored));
    }
    desc.curveSets.push_back(std::move(guides));
    UsdGenNodeDesc interpolate = OpNode("/wind/interpolate", "UsdGenGuideInterpolate",
                                        SdfPath("/wind/source"), 31);
    interpolate.references = {SdfPath("/wind/guides")};
    interpolate.params = {{TfToken("cvCount"), VtValue(6), false},
                          {TfToken("maxGuides"), VtValue(8), false},
                          {TfToken("influenceRadius"), VtValue(0.4f), false},
                          {TfToken("influenceDecay"), VtValue(1.5f), false},
                          {TfToken("maxGuideAngle"), VtValue(180.0f), false}};
    desc.nodes.push_back(interpolate);
    desc.terminal = SdfPath("/wind/interpolate");
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "GuideInterpolate consumes a clumped C3 guide reference");
    if (out.totalCvs == 0) return;
    UsdGenClumpMotion motion;
    std::string error;
    Check(UsdGenBuildClumpMotion(out, &motion, &error),
          "GuideInterpolate retains valid native clump quartets");
    if (!error.empty() || motion.levels.size() < 2) return;
    bool coarsePresent = false, finePresent = false, attenuated = false;
    for (UsdGenClumpMotionLevel const &level : motion.levels) {
        if (level.level == 0) coarsePresent = !level.groups.empty();
        if (level.level == 1) {
            finePresent = level.groups.size() > 1;
            for (float w : level.weight)
                attenuated = attenuated || (w > 0.0f && w < 0.99f);
        }
    }
    Check(coarsePresent && finePresent,
          "GuideInterpolate carries both coarse and fine guide groups");
    Check(attenuated,
          "Mixed guide influences attenuate selected-group cohesion");
    UsdGenGraphDesc driven = desc;
    UsdGenNodeDesc wind = OpNode("/wind/op", "UsdGenWind",
                                 SdfPath("/wind/interpolate"), 7);
    wind.params = {{TfToken("gustStrength"), VtValue(1.0f), false}};
    driven.nodes.push_back(wind);
    driven.terminal = SdfPath("/wind/op");
    UsdGenCurveBuffer windy;
    Check(CookAt(driven, 12.0, &windy) && !PointsEqual(windy, out) &&
          Extra(windy, "clumpId_1") != nullptr,
          "Wind consumes clump motion retained through GuideInterpolate");

    // A complete guide quartet with no assigned members is an explicit
    // independent result for that numeric level. It must replace an active
    // same-level root quartet, while other guide levels remain available.
    UsdGenGraphDesc unassigned = desc;
    std::set<std::string> const levelOneNames = {
        "clumpId_1", "clumpCenter_1", "clumpCenterId_1", "clumpWeight_1"};
    for (UsdGenAuthoredPlaneDesc &plane : unassigned.curveSets[1].authoredPlanes) {
        if (!levelOneNames.count(plane.name.GetString())) continue;
        unassigned.curveSets[0].authoredPlanes.push_back(plane);
        if (plane.name == TfToken("clumpId_1"))
            plane.intValues = VtIntArray(clumpedGuides.totalCurves, -1);
        else if (plane.name == TfToken("clumpCenter_1"))
            plane.floatValues = VtFloatArray(size_t(clumpedGuides.totalCurves) * 3, 0.0f);
        else if (plane.name == TfToken("clumpCenterId_1"))
            plane.intValues = VtIntArray(size_t(clumpedGuides.totalCurves) * 2, 0);
        else
            plane.floatValues = VtFloatArray(clumpedGuides.totalCvs, 0.0f);
    }
    UsdGenCurveBuffer independent;
    Check(Cook(unassigned, &independent),
          "Unassigned guide level overrides same-level root metadata");
    UsdGenPlane const *independentIds = Extra(independent, "clumpId_1");
    bool allUnassigned = independentIds &&
        independentIds->i.size() == independent.totalCurves;
    if (allUnassigned)
        for (int id : independentIds->i) allUnassigned = allUnassigned && id == -1;
    bool noCohesion = false;
    for (UsdGenPlane const &plane : independent.extraCv)
        if (plane.name == TfToken("clumpWeight_1")) {
            noCohesion = plane.f.size() == independent.totalCvs;
            for (float w : plane.f) noCohesion = noCohesion && w == 0.0f;
        }
    Check(allUnassigned && noCohesion,
          "Unassigned guide level clears stale root membership and cohesion");
}

static void CheckConstExpressionCook()
{
    // Per-strand constStrength 1 and 2 with no gust: strand tips rest at
    // inPz + 1 and inPz + 2 along the direction.
    UsdGenGraphDesc desc = WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                                     {TfToken("gustStrength"), VtValue(0.0f), false}});
    AddBinding(desc, SdfPath("/wind/op"),
               SdfPath("/wind/Expressions/const"), "$primIndex + 1.0",
               TfToken("float"), FloatShape(), TfToken("constStrength"),
               expr::Domain::Primitive, VtValue(0.0f));
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Wind with a constStrength expression cooks on the CPU");
    Check(CookSourceOnly(desc, SdfPath("/wind/source"), &plain),
          "Wind expression-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool profile = true;
    for (size_t c = 0; c < 2; ++c) {
        float const expected = float(c) + 1.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            profile = profile &&
                Near(out.pz[o], plain.pz[o] + expected * plain.hairT[o], 1e-5f);
        }
    }
    Check(profile, "Wind samples constStrength per strand root");
}

// --- (3) fail-closed edges ------------------------------------------------

static void CheckEdgesFailClosed()
{
    float const nan = std::numeric_limits<float>::quiet_NaN();
    float const inf = std::numeric_limits<float>::infinity();
    UsdGenCurveBuffer out;
    Check(!Cook(WindDesc({{TfToken("direction"), VtValue(GfVec3f(0, 0, 0)), false}}), &out),
          "Wind with a zero direction fails closed");
    Check(!Cook(WindDesc({{TfToken("direction"), VtValue(GfVec3f(nan, 0, 1)), false}}), &out),
          "Wind with a non-finite direction fails closed");
    Check(!Cook(WindDesc({{TfToken("constStrength"), VtValue(inf), false}}), &out),
          "Wind with a non-finite constStrength fails closed");
    Check(!Cook(WindDesc({{TfToken("gustStrength"), VtValue(nan), false}}), &out),
          "Wind with a non-finite gustStrength fails closed");
    Check(!Cook(WindDesc({{TfToken("billowLowStrength"), VtValue(inf), false}}), &out),
          "Wind with a non-finite billow strength fails closed");
    Check(!Cook(WindDesc({{TfToken("billowHighFrequency"), VtValue(-1.0f), false}}), &out),
          "Wind with a negative billow frequency fails closed");
    Check(!Cook(WindDesc({{TfToken("billowLowRate"), VtValue(nan), false}}), &out),
          "Wind with a non-finite billow rate fails closed");
    Check(!Cook(WindDesc({{TfToken("stiffness"), VtValue(nan), false}}), &out),
          "Wind with a non-finite stiffness fails closed");
    Check(!Cook(WindDesc({{TfToken("stiffness:knots"),
                           VtValue(VtFloatArray{0.0f, 1.0f}), false}}), &out),
          "Wind with mistyped stiffness:knots fails closed");
    {
        VtVec2fArray knots(2);
        knots[0] = GfVec2f(0.0f, 1.0f);
        knots[1] = GfVec2f(1.0f, nan);
        Check(!Cook(WindDesc({{TfToken("stiffness:knots"), VtValue(knots), false}}), &out),
              "Wind with non-finite stiffness:knots fails closed");
    }
    Check(!Cook(WindDesc({{TfToken("stiffness:interpolation"),
                           VtValue(TfToken("bogus")), false}}), &out),
          "Wind with an unknown stiffness:interpolation fails closed");
}

// --- (4) CUDA-backend explicit rejection ----------------------------------

static void CheckCudaRejectsWind()
{
    UsdGenGraphDesc desc = WindDesc({{TfToken("constStrength"), VtValue(2.0f), false}});
    desc.executionBackend = UsdGenExecutionBackend::Cuda;
    Check(GetCudaExecutionCapabilityMatrix().Find(TfToken("UsdGenWind")) == nullptr,
          "Wind has no CUDA capability row (CPU-only, like Clump)");
    UsdGenDiagnostics cudaDiag;
    bool const admitted = ValidateCudaGraph(desc, &cudaDiag);
    bool namesWind = false, backendUnbuilt = false;
    for (auto const &e : cudaDiag.errors) {
        if (e.find("UsdGenWind") != std::string::npos) namesWind = true;
        if (e.find("not built") != std::string::npos) backendUnbuilt = true;
    }
    Check(!admitted && cudaDiag.HasErrors(),
          "CUDA planner rejects a Wind graph");
    // CUDA builds name the missing operator; CPU-only builds report the
    // backend itself as unbuilt. Either way the graph never runs on CUDA.
    Check(namesWind || backendUnbuilt,
          "CUDA rejection names UsdGenWind (or the unbuilt backend)");
}

static void CheckIncrementalPlayback()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/playback");
    desc.curveSets.push_back(StraightStrands(SdfPath("/curves"),
        {GfVec3f(0), GfVec3f(1, 0, 0)}, 4, 1.0f));
    desc.nodes.push_back(SourceNode(SdfPath("/source"), SdfPath("/curves")));
    auto wind = OpNode("/wind", "UsdGenWind", SdfPath("/source"), 17);
    wind.params.push_back({TfToken("gustStrength"), VtValue(1.0f), false});
    desc.nodes.push_back(wind);
    desc.nodes.push_back(OpNode("/width", "UsdGenWidth", wind.path, 0));
    desc.terminal = SdfPath("/width");
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "playback graph compiles");
    if (!graph.NodeCount()) return;
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    ctx.time = 0;
    Check(!scheduler.Run(graph, ctx, 1).diagnostics.HasErrors(), "initial playback pose cooks");
    // Graph buffers are mutable work storage; detach the saved point planes.
    auto initial = graph.Output();
    initial.px.data(); initial.py.data(); initial.pz.data();
    auto const source = graph.NodeIdForPath(SdfPath("/source"));
    auto const windId = graph.NodeIdForPath(wind.path);
    auto const sourceVersion = graph.Node(source).buffer.valueVersion;
    auto const capture = graph.Node(windId).capture.get();
    ctx.time = 240;
    auto run = scheduler.Run(graph, ctx, 2);
    Check(!run.diagnostics.HasErrors() && !PointsEqual(initial, graph.Output()),
          "cached Wind animates on a new frame");
    Check(graph.Node(source).buffer.valueVersion == sourceVersion &&
              graph.Node(windId).capture.get() == capture,
          "playback reuses static source and Wind capture");
    auto animated = graph.Output();
    animated.px.data(); animated.py.data(); animated.pz.data();
    auto const windVersion = graph.Node(windId).buffer.valueVersion;
    run = scheduler.Run(graph, ctx, 3);
    Check(!run.diagnostics.HasErrors() && PointsEqual(animated, graph.Output()) &&
              graph.Node(windId).buffer.valueVersion == windVersion,
          "unchanged frame skips Wind evaluation");
    ctx.time = 0;
    run = scheduler.Run(graph, ctx, 4);
    Check(!run.diagnostics.HasErrors() && PointsEqual(initial, graph.Output()),
          "backward playback restores deterministic points through downstream Width");
}

int main()
{
    usdGenRegisterM1Operators();
    CheckIncrementalPlayback();
    CheckRegistryContracts();
    CheckIdentityCook();
    CheckConstantCook();
    CheckStiffnessRampCook();
    CheckGustDeterminismAndAnimation();
    CheckLiveTimeOnOneGraph();
    CheckBillowBands();
    CheckBillowExpressions();
    CheckClumpMotion();
    CheckClumpHierarchyMotion();
    CheckCurvedRaggedClumpMotion();
    CheckGuideClumpInheritance();
    CheckConstExpressionCook();
    CheckEdgesFailClosed();
    CheckCudaRejectsWind();
    std::printf("testUsdGenWind: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
