// UsdGenDirection operator contract and CPU cooks.
//
// Covers the UsdGenDirection styler:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/OutputPrimvars
//       plus the topology/value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed seeds):
//       the rigid and perSegment geometric contracts, roots behavior, the
//       direction:knots ramp, tilt/lift/followSkinContour shaping, identity at
//       zero effect, direction:source tangents, connected-expression and mask
//       cooks, and fail-closed validation.
#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGenExpr/expressions/context.h"
#include "usdGen/compile/graph.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/exec/scheduler.h"

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
static void Check(bool v, std::string const &s) { Check(v, s.c_str()); }

static expr::ValueShape FloatShape(uint32_t n = 1)
{
    return {expr::ScalarType::Float32, 1, n, 1, 1, false};
}

constexpr float kTol = 1e-4f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}

static bool HasParam(TfSpan<const TfToken> const &span, char const *name)
{
    TfToken const want(name);
    for (TfToken const &t : span)
        if (t == want) return true;
    return false;
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

static bool CookSourceOnly(UsdGenGraphDesc desc, SdfPath const &source,
                           UsdGenCurveBuffer *out)
{
    desc.nodes.pop_back();
    desc.expressions.clear();
    desc.terminal = source;
    return Cook(desc, out);
}

// --- (1) registry contracts ----------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenDirection");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenDirection)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenDirection)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "Direction reports its own type");
    UsdGenOpRecord const &rec = registry.Record(registry.Find(type));
    Check(rec.Def().countChanges == 0,
          "Direction has TopologyEffect None");
    std::vector<TfToken> topoV, valueV;   // routed decls, as the routing snapshot splits them
    for (UsdGenParamSlot s = 0; s < rec.ParamCount(); ++s)
        (rec.ParamClass(s) == UsdGenParamClass::Value ? valueV : topoV).push_back(rec.ParamName(s));
    UsdGenSpan<const UsdGenToken> const refs = rec.ReferenceInputs();
    Check(refs.size() == 1 && refs[0] == TfToken("direction:source"),
          "Direction declares the direction:source reference input");
    Check(usdGenTest::OutputPlaneNames(rec).empty(), "Direction emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "Direction touches the points plane only");
    TfSpan<const TfToken> const topo(topoV.data(), topoV.size());
    Check(HasParam(topo, "input") && HasParam(topo, "surface") &&
              HasParam(topo, "seed") && HasParam(topo, "mode") &&
              HasParam(topo, "direction:source"),
          "Direction routes input/surface/seed/mode/direction:source structurally");
    TfSpan<const TfToken> const value(valueV.data(), valueV.size());
    Check(HasParam(value, "enabled") && HasParam(value, "direction") &&
              HasParam(value, "amount") && HasParam(value, "lift") &&
              HasParam(value, "tiltU") && HasParam(value, "tiltV") &&
              HasParam(value, "tiltN") && HasParam(value, "aroundN") &&
              HasParam(value, "followSkinContour") &&
              HasParam(value, "direction:knots") &&
              HasParam(value, "direction:interpolation") &&
              HasParam(value, "mask"),
          "Direction routes every shaping control by value");
}

// --- (2) CPU cooks --------------------------------------------------------

static void CheckRigidCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/direction");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 5;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/direction/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/direction/source"), SdfPath("/direction/hair")));
    UsdGenNodeDesc dir = OpNode("/direction/op", "UsdGenDirection",
                                SdfPath("/direction/source"), 7);
    dir.params = {{TfToken("mode"), VtValue(TfToken("rigid")), false},
                  {TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false},
                  {TfToken("amount"), VtValue(1.0f), false}};
    desc.nodes.push_back(dir);
    desc.terminal = SdfPath("/direction/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Direction rigid cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 10,
          "Direction preserves the strand topology");
    // The full minimal rotation takes the +Y root segment onto +X, so every
    // strand lies along +X from its root with lengths preserved.
    bool rootsFixed = true, lengths = true, rigid = true, tips = true;
    for (size_t c = 0; c < 2; ++c) {
        size_t const base = c * size_t(kCvs);
        rootsFixed = rootsFixed && out.px[base] == roots[c][0] &&
            out.py[base] == 0.0f && out.pz[base] == 0.0f;
        for (int i = 1; i < kCvs; ++i) {
            float const dx = out.px[base + size_t(i)] - out.px[base + size_t(i - 1)];
            float const dy = out.py[base + size_t(i)] - out.py[base + size_t(i - 1)];
            float const dz = out.pz[base + size_t(i)] - out.pz[base + size_t(i - 1)];
            lengths = lengths &&
                Near(std::sqrt(dx * dx + dy * dy + dz * dz), kStep, 1e-5f);
        }
        for (int i = 0; i < kCvs; ++i) {
            float const dx = out.px[base + size_t(i)] - roots[c][0];
            float const dy = out.py[base + size_t(i)] - 0.0f;
            float const dz = out.pz[base + size_t(i)] - 0.0f;
            rigid = rigid &&
                Near(std::sqrt(dx * dx + dy * dy + dz * dz), float(i) * kStep);
        }
        size_t const tip = base + size_t(kCvs - 1);
        tips = tips && Near(out.px[tip], roots[c][0] + 1.0f, 1e-5f) &&
            Near(out.py[tip], 0.0f, 1e-5f) && Near(out.pz[tip], 0.0f, 1e-5f);
    }
    Check(rootsFixed, "Direction rigid keeps roots exactly fixed");
    Check(lengths, "Direction rigid preserves every segment length");
    Check(rigid, "Direction rigid rotates about the root");
    Check(tips, "Direction rigid combs the strand onto the target");
}

static void CheckPerSegmentCooks()
{
    auto cookMode = [](char const *label, char const *mode, bool withRamp,
                       UsdGenCurveBuffer *out) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath(label);
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
        constexpr int kCvs = 5;
        desc.curveSets.push_back(StraightStrands(
            SdfPath(std::string(label) + "/hair"), roots, kCvs, 0.25f));
        desc.nodes.push_back(SourceNode(SdfPath(std::string(label) + "/source"),
                                        SdfPath(std::string(label) + "/hair")));
        UsdGenNodeDesc dir = OpNode((std::string(label) + "/op").c_str(),
                                    "UsdGenDirection",
                                    SdfPath(std::string(label) + "/source"), 7);
        dir.params = {{TfToken("mode"), VtValue(TfToken(mode)), false},
                      {TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false},
                      {TfToken("amount"), VtValue(1.0f), false}};
        if (withRamp) {
            VtVec2fArray knots(2);
            knots[0] = GfVec2f(0.0f, 0.0f);
            knots[1] = GfVec2f(1.0f, 1.0f);
            dir.params.push_back(
                {TfToken("direction:knots"), VtValue(knots), false});
            dir.params.push_back({TfToken("direction:interpolation"),
                                  VtValue(TfToken("linear")), false});
        }
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath(std::string(label) + "/op");
        return Cook(desc, out);
    };
    UsdGenCurveBuffer flat, rigid;
    Check(cookMode("/dirsegflat", "perSegment", false, &flat),
          "Direction perSegment cooks on the CPU lane");
    Check(cookMode("/dirsegrigid", "rigid", false, &rigid),
          "Direction rigid reference cooks for the perSegment comparison");
    if (flat.totalCvs == 0 || rigid.totalCvs == 0) return;
    // A flat ramp turns once at the root, exactly like rigid.
    bool matches = flat.totalCvs == rigid.totalCvs;
    for (size_t o = 0; matches && o < flat.totalCvs; ++o)
        matches = Near(flat.px[o], rigid.px[o], 1e-4f) &&
            Near(flat.py[o], rigid.py[o], 1e-4f) &&
            Near(flat.pz[o], rigid.pz[o], 1e-4f);
    Check(matches, "Direction perSegment with a flat ramp matches rigid");

    UsdGenCurveBuffer ramped;
    Check(cookMode("/dirsegramp", "perSegment", true, &ramped),
          "Direction perSegment with a 0->1 ramp cooks");
    if (ramped.totalCvs == 0) return;
    // The 0->1 ramp puts no turn into the root segment (R(t_0) - R(t_{-1}) =
    // 0), then progresses down the strand: CV 1 keeps its input position,
    // lengths hold, roots stay fixed, and the tip turns only partway to +X.
    bool rootsFixed = true, lengths = true, firstSeg = true, partial = true;
    constexpr int kCvs = 5;
    constexpr float kStep = 0.25f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    for (size_t c = 0; c < 2; ++c) {
        size_t const base = c * size_t(kCvs);
        rootsFixed = rootsFixed && ramped.px[base] == roots[c][0] &&
            ramped.py[base] == 0.0f && ramped.pz[base] == 0.0f;
        firstSeg = firstSeg &&
            Near(ramped.px[base + 1], roots[c][0], 1e-5f) &&
            Near(ramped.py[base + 1], kStep, 1e-5f) &&
            Near(ramped.pz[base + 1], 0.0f, 1e-5f);
        for (int i = 1; i < kCvs; ++i) {
            float const dx = ramped.px[base + size_t(i)] - ramped.px[base + size_t(i - 1)];
            float const dy = ramped.py[base + size_t(i)] - ramped.py[base + size_t(i - 1)];
            float const dz = ramped.pz[base + size_t(i)] - ramped.pz[base + size_t(i - 1)];
            lengths = lengths &&
                Near(std::sqrt(dx * dx + dy * dy + dz * dz), kStep, 1e-5f);
        }
        float const tipX = ramped.px[base + size_t(kCvs - 1)] - roots[c][0];
        float const tipY = ramped.py[base + size_t(kCvs - 1)];
        partial = partial && tipX > 0.2f && tipX < 0.99f && tipY > 0.0f &&
            tipY < 1.0f;
    }
    Check(rootsFixed, "Direction perSegment keeps roots exactly fixed");
    Check(firstSeg, "Direction perSegment leaves the root segment unturned on a 0->1 ramp");
    Check(lengths, "Direction perSegment preserves every segment length");
    Check(partial, "Direction perSegment progresses the turn down the strand");
}

static void CheckIdentityCook()
{
    for (char const *mode : {"rigid", "perSegment"}) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dirid");
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0),
                                            GfVec3f(0, 0, 5)};
        desc.curveSets.push_back(
            StraightStrands(SdfPath("/dirid/hair"), roots, 4, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/dirid/source"), SdfPath("/dirid/hair")));
        UsdGenNodeDesc dir = OpNode("/dirid/op", "UsdGenDirection",
                                    SdfPath("/dirid/source"), 7);
        VtVec2fArray knots(2);
        knots[0] = GfVec2f(0.0f, 0.0f);
        knots[1] = GfVec2f(1.0f, 1.0f);
        dir.params = {{TfToken("mode"), VtValue(TfToken(mode)), false},
                      {TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false},
                      {TfToken("amount"), VtValue(0.0f), false},
                      {TfToken("lift"), VtValue(45.0f), false},
                      {TfToken("tiltU"), VtValue(30.0f), false},
                      {TfToken("followSkinContour"), VtValue(1.0f), false},
                      {TfToken("direction:knots"), VtValue(knots), false},
                      {TfToken("direction:interpolation"),
                       VtValue(TfToken("linear")), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dirid/op");
        UsdGenCurveBuffer out, plain;
        std::string const label = std::string("Direction ") + mode +
            " at zero amount cooks on the CPU";
        Check(Cook(desc, &out), label);
        if (out.totalCvs == 0) return;
        Check(CookSourceOnly(desc, SdfPath("/dirid/source"), &plain),
              "Direction identity fixture source cooks on its own");
        if (plain.totalCvs == 0) return;
        Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
              std::string("Direction ") + mode +
                  " at zero amount is bitwise identity");
    }
}

static void CheckShapingCooks()
{
    // lift = 90 about root B (0,1,0) turns the +Z target onto +X, so the
    // strand combs exactly as if direction had been authored as +X.
    {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dirlift");
        desc.defaultWidth = 0.01f;
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/dirlift/hair"), {GfVec3f(0, 0, 0)}, 5, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/dirlift/source"), SdfPath("/dirlift/hair")));
        UsdGenNodeDesc dir = OpNode("/dirlift/op", "UsdGenDirection",
                                    SdfPath("/dirlift/source"), 7);
        dir.params = {{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                      {TfToken("amount"), VtValue(1.0f), false},
                      {TfToken("lift"), VtValue(90.0f), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dirlift/op");
        UsdGenCurveBuffer out;
        Check(Cook(desc, &out), "Direction with lift cooks on the CPU lane");
        if (out.totalCvs == 0) return;
        Check(Near(out.px[4], 1.0f, 1e-5f) && Near(out.py[4], 0.0f, 1e-5f) &&
                  Near(out.pz[4], 0.0f, 1e-5f),
              "Direction lift=90 elevates the target like Grow lift");
    }
    // tiltN = 90 about root N (0,0,1) turns the +X target onto +Y, the
    // strand's own direction, so the cook is a no-op to tolerance.
    {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dirtilt");
        desc.defaultWidth = 0.01f;
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/dirtilt/hair"), {GfVec3f(0, 0, 0)}, 5, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/dirtilt/source"), SdfPath("/dirtilt/hair")));
        UsdGenNodeDesc dir = OpNode("/dirtilt/op", "UsdGenDirection",
                                    SdfPath("/dirtilt/source"), 7);
        dir.params = {{TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false},
                      {TfToken("amount"), VtValue(1.0f), false},
                      {TfToken("tiltN"), VtValue(90.0f), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dirtilt/op");
        UsdGenCurveBuffer out, plain;
        Check(Cook(desc, &out), "Direction with tiltN cooks on the CPU lane");
        if (out.totalCvs == 0) return;
        Check(CookSourceOnly(desc, SdfPath("/dirtilt/source"), &plain),
              "Direction tilt fixture source cooks on its own");
        if (plain.totalCvs == 0) return;
        bool same = out.totalCvs == plain.totalCvs;
        for (size_t o = 0; same && o < out.totalCvs; ++o)
            same = Near(out.px[o], plain.px[o], 1e-5f) &&
                Near(out.py[o], plain.py[o], 1e-5f) &&
                Near(out.pz[o], plain.pz[o], 1e-5f);
        Check(same, "Direction tiltN=90 pre-rotates the target in the root frame");
    }
    // followSkinContour = 1 projects the (1,0,1) target onto the root tangent
    // plane, leaving +X, so the strand combs onto +X.
    {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dircontour");
        desc.defaultWidth = 0.01f;
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/dircontour/hair"), {GfVec3f(0, 0, 0)}, 5, 0.25f));
        desc.nodes.push_back(SourceNode(SdfPath("/dircontour/source"),
                                        SdfPath("/dircontour/hair")));
        UsdGenNodeDesc dir = OpNode("/dircontour/op", "UsdGenDirection",
                                    SdfPath("/dircontour/source"), 7);
        dir.params = {{TfToken("direction"), VtValue(GfVec3f(1, 0, 1)), false},
                      {TfToken("amount"), VtValue(1.0f), false},
                      {TfToken("followSkinContour"), VtValue(1.0f), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dircontour/op");
        UsdGenCurveBuffer out;
        Check(Cook(desc, &out),
              "Direction with followSkinContour cooks on the CPU lane");
        if (out.totalCvs == 0) return;
        Check(Near(out.px[4], 1.0f, 1e-5f) && Near(out.py[4], 0.0f, 1e-5f) &&
                  Near(out.pz[4], 0.0f, 1e-5f),
              "Direction followSkinContour=1 blends the target onto the skin tangent plane");
    }
}

static void CheckConnectedCooks()
{
    // A primitive-domain amount expression combs per strand: strand 0 keeps
    // amount 0 (bitwise identity) while strand 1 combs fully onto +X.
    {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/direxpr");
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
        constexpr int kCvs = 5;
        desc.curveSets.push_back(
            StraightStrands(SdfPath("/direxpr/hair"), roots, kCvs, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/direxpr/source"), SdfPath("/direxpr/hair")));
        UsdGenNodeDesc dir = OpNode("/direxpr/op", "UsdGenDirection",
                                    SdfPath("/direxpr/source"), 7);
        dir.params = {{TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/direxpr/op");
        AddBinding(desc, SdfPath("/direxpr/op"),
                   SdfPath("/direxpr/Expressions/amount"), "$primIndex",
                   TfToken("float"), FloatShape(), TfToken("amount"),
                   expr::Domain::Primitive, VtValue(0.0f));
        UsdGenCurveBuffer out, plain;
        Check(Cook(desc, &out),
              "Direction with an amount expression cooks on the CPU");
        if (out.totalCvs == 0) return;
        Check(CookSourceOnly(desc, SdfPath("/direxpr/source"), &plain),
              "Direction expression fixture source cooks on its own");
        if (plain.totalCvs == 0) return;
        bool first = true;
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = size_t(i);
            first = first && out.px[o] == plain.px[o] &&
                out.py[o] == plain.py[o] && out.pz[o] == plain.pz[o];
        }
        Check(first, "the zero-amount expression strand is bitwise unchanged");
        size_t const tip = size_t(kCvs - 1) + size_t(kCvs);
        Check(Near(out.px[tip], 4.0f, 1e-5f) && Near(out.py[tip], 0.0f, 1e-5f) &&
                  Near(out.pz[tip], 0.0f, 1e-5f),
              "a primitive amount expression combs strands per curve");
    }
    // A primitive-domain mask expression gates the envelope per strand.
    {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dirmask");
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
        constexpr int kCvs = 5;
        desc.curveSets.push_back(
            StraightStrands(SdfPath("/dirmask/hair"), roots, kCvs, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/dirmask/source"), SdfPath("/dirmask/hair")));
        UsdGenNodeDesc dir = OpNode("/dirmask/op", "UsdGenDirection",
                                    SdfPath("/dirmask/source"), 7);
        dir.params = {{TfToken("direction"), VtValue(GfVec3f(1, 0, 0)), false},
                      {TfToken("amount"), VtValue(1.0f), false}};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dirmask/op");
        AddBinding(desc, SdfPath("/dirmask/op"),
                   SdfPath("/dirmask/Expressions/mask"), "$primIndex == 1 ? 0 : 1",
                   TfToken("float"), FloatShape(), TfToken("mask"),
                   expr::Domain::Primitive, VtValue(1.0f));
        UsdGenCurveBuffer out, plain;
        Check(Cook(desc, &out),
              "Direction with a mask expression cooks on the CPU");
        if (out.totalCvs == 0) return;
        Check(CookSourceOnly(desc, SdfPath("/dirmask/source"), &plain),
              "Direction mask fixture source cooks on its own");
        if (plain.totalCvs == 0) return;
        bool masked = true;
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = size_t(kCvs) + size_t(i);
            masked = masked && out.px[o] == plain.px[o] &&
                out.py[o] == plain.py[o] && out.pz[o] == plain.pz[o];
        }
        Check(masked, "the masked strand is bitwise unchanged");
        Check(Near(out.px[size_t(kCvs - 1)], 1.0f, 1e-5f),
              "the unmasked strand combs fully");
    }
}

static void CheckSourceCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/dirsource");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0.02f, 0),
                                        GfVec3f(0, -0.02f, 0)};
    constexpr int kCvs = 4;
    constexpr float kLen = 3 * 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/dirsource/hair"), roots, kCvs, 0.25f));
    UsdGenCurveSetDesc guides;
    guides.path = SdfPath("/dirsource/guides");
    guides.role = UsdGenLane::Reference;
    guides.curveRole = TfToken("guide");
    guides.curveVertexCounts = {3};
    guides.points = {GfVec3f(-1, 0, 0), GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
    guides.rest = guides.points;
    desc.curveSets.push_back(guides);
    desc.nodes.push_back(
        SourceNode(SdfPath("/dirsource/source"), SdfPath("/dirsource/hair")));
    UsdGenNodeDesc dir = OpNode("/dirsource/op", "UsdGenDirection",
                                SdfPath("/dirsource/source"), 7);
    dir.references = {guides.path};
    // The bound source overrides the authored direction (+Z here): both
    // strands comb onto the +X guide tangent from their roots.
    dir.params = {{TfToken("direction"), VtValue(GfVec3f(0, 0, 1)), false},
                  {TfToken("amount"), VtValue(1.0f), false}};
    desc.nodes.push_back(dir);
    desc.terminal = SdfPath("/dirsource/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Direction with direction:source cooks on the CPU");
    if (out.totalCvs == 0) return;
    bool rootsFixed = true, tips = true;
    for (size_t c = 0; c < 2; ++c) {
        size_t const base = c * size_t(kCvs);
        rootsFixed = rootsFixed && out.px[base] == roots[c][0] &&
            out.py[base] == roots[c][1] && out.pz[base] == roots[c][2];
        size_t const tip = base + size_t(kCvs - 1);
        tips = tips && Near(out.px[tip], roots[c][0] + kLen, 1e-4f) &&
            Near(out.py[tip], roots[c][1], 1e-4f) &&
            Near(out.pz[tip], roots[c][2], 1e-4f);
    }
    Check(rootsFixed, "Direction with direction:source keeps roots exactly fixed");
    Check(tips, "Direction combs strands onto the source curve tangents");
}

static void CheckFailClosed()
{
    auto cookDirection = [](UsdGenParamValue const &param) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/dirfail");
        desc.defaultWidth = 0.01f;
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/dirfail/hair"), {GfVec3f(0, 0, 0)}, 4, 0.25f));
        desc.nodes.push_back(
            SourceNode(SdfPath("/dirfail/source"), SdfPath("/dirfail/hair")));
        UsdGenNodeDesc dir = OpNode("/dirfail/op", "UsdGenDirection",
                                    SdfPath("/dirfail/source"), 7);
        dir.params = {param};
        desc.nodes.push_back(dir);
        desc.terminal = SdfPath("/dirfail/op");
        UsdGenCurveBuffer out;
        return Cook(desc, &out);
    };
    Check(!cookDirection({TfToken("mode"), VtValue(TfToken("bogus")), false}),
          "an unknown Direction mode fails closed");
    Check(!cookDirection({TfToken("amount"), VtValue(2.0f), false}),
          "an out-of-range Direction amount fails closed");
    Check(!cookDirection({TfToken("direction:interpolation"),
                          VtValue(TfToken("bogus")), false}),
          "an unknown Direction ramp interpolation fails closed");
}

int main()
{
    CheckRegistryContracts();
    CheckRigidCook();
    CheckPerSegmentCooks();
    CheckIdentityCook();
    CheckShapingCooks();
    CheckConnectedCooks();
    CheckSourceCook();
    CheckFailClosed();
    std::printf("testUsdGenDirection: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
