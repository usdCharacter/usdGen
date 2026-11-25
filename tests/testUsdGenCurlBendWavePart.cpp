// Curl/Bend/Wave/Part operator contracts, expression targets and CPU cooks.
//
// Covers the four UsdGenCurl/UsdGenBend/UsdGenWave/UsdGenPart stylers plus the
// primitive-domain Grow lift used to comb strands per curve:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/OutputPrimvars;
//   (2) expression-target validation through UsdGenCompiler (one valid binding
//       compiles, one wrong-destination and one wrong-domain binding fail);
//   (3) deterministic CPU cooks over small hand-built topologies (fixed seeds).
#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGen/cudaExecution.h"
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
static void Check(bool v, std::string const &s) { Check(v, s.c_str()); }

static expr::ValueShape FloatShape(uint32_t n = 1)
{
    return {expr::ScalarType::Float32, 1, n, 1, 1, false};
}
static expr::ValueShape BoolShape()
{
    return {expr::ScalarType::Bool, 1, 1, 1, 1, false};
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

static UsdGenPlane const *FindCurvePlane(UsdGenCurveBuffer const &buffer,
                                         char const *name)
{
    for (UsdGenPlane const &plane : buffer.extraCurve)
        if (plane.name == TfToken(name)) return &plane;
    return nullptr;
}

// --- (1) registry contracts ----------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    struct Expect {
        char const *type;
        char const *reference;
        char const *primvar;
    };
    Expect const kOps[] = {
        {"UsdGenCurl", nullptr, nullptr},
        {"UsdGenBend", nullptr, nullptr},
        {"UsdGenWave", nullptr, nullptr},
        {"UsdGenPart", "part:curves", "partId"},
    };
    for (Expect const &expect : kOps) {
        TfToken const type(expect.type);
        std::string const name = type.GetString();
        Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(" + name + ")");
        std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
        Check(op != nullptr, "Create(" + name + ")");
        if (!op) continue;
        Check(registry.Record(registry.Find(type)).Name() == type, name + " reports its own type");
        UsdGenOpRecord const &rec = op->Record();
        Check(rec.Def().topoFx == UsdGenTopoFx::None,
              name + " has TopologyEffect None");
        TfSpan<const TfToken> const refs = rec.ReferenceInputs();
        std::vector<TfToken> const primvars = usdGenTest::OutputPlaneNames(rec);
        if (expect.reference) {
            Check(refs.size() == 1 && refs[0] == TfToken(expect.reference),
                  name + " declares the part:curves reference input");
            Check(primvars.size() == 1 && primvars[0] == TfToken(expect.primvar),
                  name + " emits the partId output primvar");
        } else {
            Check(refs.empty(), name + " declares no reference inputs");
            Check(primvars.empty(), name + " emits no output primvars");
        }
    }
}

// --- (2) expression-target validation ------------------------------------
// One CurveSource-fed op node; Part additionally names its parting set.

static UsdGenGraphDesc OpDesc(char const *opType, bool withPartRef)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(StraightStrands(
        SdfPath("/groom/hair"), {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)}, 4, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/groom/source"), SdfPath("/groom/hair")));
    UsdGenNodeDesc op = OpNode("/groom/op", opType, SdfPath("/groom/source"), 7);
    if (withPartRef) {
        UsdGenCurveSetDesc parting;
        parting.path = SdfPath("/groom/parting");
        parting.role = UsdGenRole::Reference;
        parting.curveRole = TfToken("guide");
        parting.curveVertexCounts = {3};
        parting.points = {GfVec3f(-1, 0, 0), GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
        parting.rest = parting.points;
        desc.curveSets.push_back(parting);
        op.references = {parting.path};
    }
    desc.nodes.push_back(op);
    desc.terminal = SdfPath("/groom/op");
    return desc;
}

static void ExpectFail(UsdGenGraphDesc const &desc, char const *label,
                       char const *reason)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult r = compiler.Compile(desc, &graph);
    bool const specific =
        std::any_of(r.errors.begin(), r.errors.end(),
                    [&](std::string const &message) {
                        return message.find(reason) != std::string::npos;
                    });
    Check(!r.ok && specific, label);
    if (r.ok || !specific)
        for (auto const &e : r.errors)
            std::printf("  error: %s\n", e.c_str());
}

static void CheckExpressionTargets()
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    SdfPath const opPath("/groom/op");
    SdfPath const exprPath("/groom/Expressions/ctl");

    // Curl: radius varies per CV; taper/clockwise are groom-wide toggles.
    {
        UsdGenGraphDesc valid = OpDesc("UsdGenCurl", false);
        AddBinding(valid, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("radius"), expr::Domain::Point,
                   VtValue(0.1f));
        Check(compiler.Compile(valid, &graph).ok,
              "a point-domain radius connection compiles on Curl");
        UsdGenGraphDesc foreign = OpDesc("UsdGenCurl", false);
        AddBinding(foreign, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("width"), expr::Domain::Point,
                   VtValue(0.1f));
        ExpectFail(foreign, "a destination Curl does not own fails closed",
                   "unsupported or incorrectly typed Curl expression");
        UsdGenGraphDesc groomToggle = OpDesc("UsdGenCurl", false);
        AddBinding(groomToggle, opPath, exprPath, "$value", TfToken("bool"),
                   BoolShape(), TfToken("taper"), expr::Domain::Primitive,
                   VtValue(true));
        ExpectFail(groomToggle, "a primitive-domain Curl taper fails closed",
                   "Curl taper requires groom evaluation");
    }
    // Bend: the angle is per CV; angleRandom/axis are per strand root.
    {
        UsdGenGraphDesc valid = OpDesc("UsdGenBend", false);
        AddBinding(valid, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("angle"), expr::Domain::Point,
                   VtValue(0.0f));
        Check(compiler.Compile(valid, &graph).ok,
              "a point-domain angle connection compiles on Bend");
        UsdGenGraphDesc foreign = OpDesc("UsdGenBend", false);
        AddBinding(foreign, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("width"), expr::Domain::Point,
                   VtValue(0.1f));
        ExpectFail(foreign, "a destination Bend does not own fails closed",
                   "unsupported or incorrectly typed Bend expression");
        UsdGenGraphDesc pointRandom = OpDesc("UsdGenBend", false);
        AddBinding(pointRandom, opPath, exprPath,
                   "[$primIndex * 0 + 1, $primIndex * 0 + 1]", TfToken("float2"),
                   FloatShape(2), TfToken("angleRandom"), expr::Domain::Point,
                   VtValue(GfVec2f(1, 1)));
        ExpectFail(pointRandom, "a point-domain Bend angleRandom fails closed",
                   "requires groom/primitive evaluation");
    }
    // Wave: amplitudes vary per CV; enabled is groom-wide.
    {
        UsdGenGraphDesc valid = OpDesc("UsdGenWave", false);
        AddBinding(valid, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("amplitudeU"), expr::Domain::Point,
                   VtValue(0.0f));
        Check(compiler.Compile(valid, &graph).ok,
              "a point-domain amplitudeU connection compiles on Wave");
        UsdGenGraphDesc foreign = OpDesc("UsdGenWave", false);
        AddBinding(foreign, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("width"), expr::Domain::Point,
                   VtValue(0.1f));
        ExpectFail(foreign, "a destination Wave does not own fails closed",
                   "unsupported or incorrectly typed Wave expression");
        UsdGenGraphDesc pointEnabled = OpDesc("UsdGenWave", false);
        AddBinding(pointEnabled, opPath, exprPath, "$value", TfToken("bool"),
                   BoolShape(), TfToken("enabled"), expr::Domain::Primitive,
                   VtValue(true));
        ExpectFail(pointEnabled, "a primitive-domain Wave enabled fails closed",
                   "Wave enabled requires groom evaluation");
    }
    // Part: radius/strength are per strand root; the mask envelopes per CV.
    {
        UsdGenGraphDesc valid = OpDesc("UsdGenPart", true);
        AddBinding(valid, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("part:strength"),
                   expr::Domain::Primitive, VtValue(1.0f));
        Check(compiler.Compile(valid, &graph).ok,
              "a primitive-domain part:strength connection compiles on Part");
        UsdGenGraphDesc foreign = OpDesc("UsdGenPart", true);
        AddBinding(foreign, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("width"), expr::Domain::Point,
                   VtValue(0.1f));
        ExpectFail(foreign, "a destination Part does not own fails closed",
                   "unsupported or incorrectly typed Part expression");
        UsdGenGraphDesc pointStrength = OpDesc("UsdGenPart", true);
        AddBinding(pointStrength, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("part:strength"), expr::Domain::Point,
                   VtValue(1.0f));
        ExpectFail(pointStrength,
                   "a point-domain Part part:strength fails closed",
                   "Part part:strength requires groom/primitive evaluation");
    }
    // Grow bakes per-strand lift at capture: primitive compiles, point fails.
    {
        UsdGenGraphDesc valid = OpDesc("UsdGenGrow", false);
        AddBinding(valid, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("lift"), expr::Domain::Primitive,
                   VtValue(0.0f));
        Check(compiler.Compile(valid, &graph).ok,
              "a primitive-domain lift connection compiles on Grow");
        UsdGenGraphDesc pointLift = OpDesc("UsdGenGrow", false);
        AddBinding(pointLift, opPath, exprPath, "$value", TfToken("float"),
                   FloatShape(), TfToken("lift"), expr::Domain::Point,
                   VtValue(0.0f));
        ExpectFail(pointLift, "a point-domain Grow lift fails closed",
                   "requires groom/primitive evaluation");
    }
}

// --- (3) CPU cooks -------------------------------------------------------

static void CheckWaveCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/wave");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 5;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/wave/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/wave/source"), SdfPath("/wave/hair")));
    UsdGenNodeDesc wave = OpNode("/wave/op", "UsdGenWave",
                                 SdfPath("/wave/source"), 7);
    wave.params = {{TfToken("amplitudeU"), VtValue(0.5f), false},
                   {TfToken("frequencyU"), VtValue(1.0f), false},
                   {TfToken("amplitudeN"), VtValue(0.0f), false},
                   {TfToken("frequencyN"), VtValue(1.0f), false}};
    desc.nodes.push_back(wave);
    desc.terminal = SdfPath("/wave/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Wave cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 10,
          "Wave preserves the strand topology");
    // P[i] += A sin(2 pi f s_i) T with T = (1,0,0); roots (s = 0) never move.
    bool rootsFixed = true, profile = true, sidesExact = true;
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    for (size_t c = 0; c < 2; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            float const s = float(i) * kStep;
            float const expected =
                roots[c][0] + 0.5f * float(std::sin(kTwoPi * 1.0 * s));
            if (i == 0)
                rootsFixed = rootsFixed && out.px[o] == roots[c][0] &&
                    out.py[o] == 0.0f && out.pz[o] == 0.0f;
            profile = profile && Near(out.px[o], expected, 1e-5f);
            sidesExact = sidesExact && out.py[o] == float(i) * kStep &&
                out.pz[o] == 0.0f;
        }
    }
    Check(rootsFixed, "Wave leaves every root exactly unmoved");
    Check(profile, "Wave displaces mids by A sin(2 pi f s)");
    Check(sidesExact, "Wave moves nothing off the root tangent axis");
}

static void CheckCurlCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/curl");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 6;
    constexpr float kStep = 0.2f;
    constexpr float kRadius = 0.1f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/curl/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/curl/source"), SdfPath("/curl/hair")));
    UsdGenNodeDesc curl = OpNode("/curl/op", "UsdGenCurl",
                                 SdfPath("/curl/source"), 7);
    curl.params = {{TfToken("radius"), VtValue(kRadius), false},
                   {TfToken("frequency"), VtValue(2.0f), false},
                   {TfToken("phase"), VtValue(0.0f), false},
                   {TfToken("phaseRandom"), VtValue(0.0f), false},
                   {TfToken("taper"), VtValue(true), false}};
    desc.nodes.push_back(curl);
    desc.terminal = SdfPath("/curl/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Curl cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // With taper the coil radius is r(t) = radius * t, so every CV sits
    // radius*t from its input position and roots (t = 0) stay fixed.
    bool rootsFixed = true, envelope = true;
    for (size_t c = 0; c < 2; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            float const dx = out.px[o] - roots[c][0];
            float const dy = out.py[o] - float(i) * kStep;
            float const dz = out.pz[o] - 0.0f;
            float const dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            float const t = float(i) / float(kCvs - 1);
            if (i == 0)
                rootsFixed =
                    rootsFixed && dist == 0.0f && out.px[o] == roots[c][0];
            envelope = envelope && Near(dist, kRadius * t, 1e-5f);
        }
    }
    Check(rootsFixed, "Curl with taper keeps roots exactly fixed");
    Check(envelope, "Curl offsets mids by radius times the taper ramp");
}

static void CheckBendCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/bend");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    constexpr int kCvs = 5;
    constexpr float kStep = 0.25f;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/bend/hair"), roots, kCvs, kStep));
    desc.nodes.push_back(
        SourceNode(SdfPath("/bend/source"), SdfPath("/bend/hair")));
    UsdGenNodeDesc bend = OpNode("/bend/op", "UsdGenBend",
                                 SdfPath("/bend/source"), 7);
    bend.params = {{TfToken("angle"), VtValue(90.0f), false},
                   {TfToken("axisMode"), VtValue(TfToken("uniform")), false},
                   {TfToken("axis"), VtValue(GfVec3f(0, 0, 1)), false}};
    desc.nodes.push_back(bend);
    desc.terminal = SdfPath("/bend/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Bend cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // A flat ramp bends rigidly about the root: rotating +Y about +Z by +90
    // degrees lays the strand along -X with segment lengths preserved.
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
                Near(std::sqrt(dx * dx + dy * dy + dz * dz),
                     float(i) * kStep);
        }
        size_t const tip = base + size_t(kCvs - 1);
        tips = tips && Near(out.px[tip], roots[c][0] - 1.0f) &&
            Near(out.py[tip], 0.0f) && Near(out.pz[tip], 0.0f);
    }
    Check(rootsFixed, "Bend keeps roots exactly fixed");
    Check(lengths, "Bend preserves every segment length");
    Check(rigid, "Bend 90deg with a flat ramp rotates rigidly about the root");
    Check(tips, "Bend lays the strand along -X");
}

static void CheckClumpMotionConsumers()
{
    auto cookStyle = [](char const *type, int motionMode, float weight,
                        bool differentFrame, size_t curveCount,
                        UsdGenCurveBuffer *out) {
        UsdGenGraphDesc desc;
        desc.description = SdfPath("/motion");
        desc.defaultWidth = 0.01f;
        std::vector<GfVec3f> roots;
        roots.reserve(curveCount);
        for (size_t c = 0; c < curveCount; ++c)
            roots.emplace_back(float(c) * 2.0f, 0, 0);
        auto curves = StraightStrands(SdfPath("/motion/hair"), roots, 6, 0.2f);
        if (differentFrame) {
            curves.rootFrame[1].SetRow3(0, GfVec3d(0, 0, 1));
            curves.rootFrame[1].SetRow3(2, GfVec3d(-1, 0, 0));
        }
        if (motionMode) {
            UsdGenAuthoredPlaneDesc id;
            id.name = TfToken("clumpId_2");
            id.type = UsdGenAuthoredPlaneType::Int32;
            id.domain = UsdGenAuthoredPlaneDomain::Primitive;
            id.intValues.assign(curveCount, motionMode == 3 ? -1 : 0);
            curves.authoredPlanes.push_back(id);
        }
        if (motionMode == 1 || motionMode == 3) {
            UsdGenAuthoredPlaneDesc center;
            center.name = TfToken("clumpCenter_2");
            center.domain = UsdGenAuthoredPlaneDomain::Primitive;
            center.arity = 3;
            center.floatValues.resize(curveCount * 3);
            for (size_t c = 0; c < curveCount; ++c)
                center.floatValues[c * 3] = 1.0f;
            curves.authoredPlanes.push_back(center);
            UsdGenAuthoredPlaneDesc centerId;
            centerId.name = TfToken("clumpCenterId_2");
            centerId.type = UsdGenAuthoredPlaneType::Int32;
            centerId.domain = UsdGenAuthoredPlaneDomain::Primitive;
            centerId.arity = 2;
            centerId.intValues.resize(curveCount * 2);
            for (size_t c = 0; c < curveCount; ++c)
                centerId.intValues[c * 2] = 123;
            curves.authoredPlanes.push_back(centerId);
            UsdGenAuthoredPlaneDesc weights;
            weights.name = TfToken("clumpWeight_2");
            weights.domain = UsdGenAuthoredPlaneDomain::Point;
            weights.floatValues.assign(curveCount * 6, weight);
            for (size_t c = 0; c < curveCount; ++c)
                weights.floatValues[c * 6] = 0.0f;
            curves.authoredPlanes.push_back(weights);
        }
        desc.curveSets.push_back(std::move(curves));
        desc.nodes.push_back(SourceNode(SdfPath("/motion/source"), SdfPath("/motion/hair")));
        auto style = OpNode("/motion/style", type, SdfPath("/motion/source"), 19);
        if (std::string(type) == "UsdGenCurl") {
            style.params = {{TfToken("radius"), VtValue(0.2f), false},
                            {TfToken("frequency"), VtValue(0.5f), false},
                            {TfToken("phaseRandom"), VtValue(1.0f), false},
                            {TfToken("axisMode"), VtValue(TfToken("guide")), false}};
        } else if (std::string(type) == "UsdGenBend") {
            style.params = {{TfToken("angle"), VtValue(70.0f), false},
                            {TfToken("angleRandom"), VtValue(GfVec2f(0.2f, 1.8f)), false},
                            {TfToken("axisMode"), VtValue(TfToken("rootDirection")), false},
                            {TfToken("axis"), VtValue(GfVec3f(1, 0, 0)), false}};
        } else if (std::string(type) == "UsdGenNoise") {
            style.params = {{TfToken("noise:magnitude"), VtValue(0.2f), false},
                            {TfToken("preserveLength"), VtValue(false), false},
                            {TfToken("noise:correlation"), VtValue(0.6f), false}};
        } else {
            style.params = {{TfToken("amplitudeU"), VtValue(0.2f), false},
                            {TfToken("frequencyU"), VtValue(0.7f), false}};
        }
        desc.nodes.push_back(style);
        desc.terminal = style.path;
        return Cook(desc, out);
    };
    for (char const *type : {"UsdGenCurl", "UsdGenBend", "UsdGenNoise", "UsdGenWave"}) {
        UsdGenCurveBuffer legacy, zero, idOnly, unassigned, clumped, variedFrame;
        bool const cooked = cookStyle(type, 0, 0, false, 2, &legacy) &&
            cookStyle(type, 1, 0, false, 2, &zero) &&
            cookStyle(type, 2, 1, false, 2, &idOnly) &&
            cookStyle(type, 3, 1, false, 2, &unassigned) &&
            cookStyle(type, 1, 1, false, 2, &clumped) &&
            cookStyle(type, 1, 1, true, 2, &variedFrame);
        Check(cooked, std::string(type) + " clump motion fixtures cook");
        if (!cooked) continue;
        bool zeroExact = true, idOnlyExact = true, unassignedExact = true;
        for (size_t i = 0; i < legacy.totalCvs; ++i)
        {
            zeroExact = zeroExact && zero.px[i] == legacy.px[i] &&
                zero.py[i] == legacy.py[i] && zero.pz[i] == legacy.pz[i];
            idOnlyExact = idOnlyExact && idOnly.px[i] == legacy.px[i] &&
                idOnly.py[i] == legacy.py[i] && idOnly.pz[i] == legacy.pz[i];
            unassignedExact = unassignedExact && unassigned.px[i] == legacy.px[i] &&
                unassigned.py[i] == legacy.py[i] && unassigned.pz[i] == legacy.pz[i];
        }
        Check(zeroExact, std::string(type) + " zero cohesion preserves legacy points");
        Check(idOnlyExact, std::string(type) + " ID-only preserves legacy points");
        Check(unassignedExact, std::string(type) + " unassigned ID preserves legacy points");
        auto offset = [](UsdGenCurveBuffer const &b, size_t c, size_t i) {
            size_t const o = c * 6 + i;
            return GfVec3f(b.px[o] - float(c) * 2.0f,
                           b.py[o] - float(i) * 0.2f, b.pz[o]);
        };
        bool rootsPreserved = true, coherent = true;
        for (size_t c = 0; c < 2; ++c)
            rootsPreserved = rootsPreserved &&
                offset(clumped, c, 0) == offset(legacy, c, 0);
        for (size_t i = 1; i < 6; ++i)
            coherent = coherent && (offset(clumped, 0, i) -
                offset(clumped, 1, i)).GetLength() < 1e-4f;
        Check(rootsPreserved, std::string(type) + " root lock preserves legacy root");
        Check(coherent, std::string(type) + " same-group world offsets agree");
        if (std::string(type) != "UsdGenWave")
            Check((offset(legacy, 0, 3) - offset(legacy, 1, 3)).GetLength() >
                      1e-4f,
                  std::string(type) + " fixture starts with distinct strand motion");
        Check((offset(variedFrame, 0, 3) - offset(variedFrame, 1, 3)).GetLength() >
                  1e-4f,
              std::string(type) + " keeps distinct authored root frames");
    }
    UsdGenCurveBuffer acrossChunks;
    bool const acrossCooked = cookStyle("UsdGenCurl", 1, 1, false, 513,
                                       &acrossChunks);
    Check(acrossCooked, "Curl group across two chunks cooks");
    if (acrossCooked) {
        auto displacement = [&](size_t c) {
            size_t const o = c * 6 + 4;
            return GfVec3f(acrossChunks.px[o] - float(c) * 2.0f,
                           acrossChunks.py[o] - 0.8f, acrossChunks.pz[o]);
        };
        Check((displacement(0) - displacement(512)).GetLength() < 1e-4f,
              "Curl shares a clump draw across chunk boundaries");
    }
}

static void CheckCudaImportedClumpAdmission()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/cudaClump");
    auto curves = StraightStrands(SdfPath("/cudaClump/hair"),
                                  {GfVec3f(0, 0, 0)}, 3, 0.5f);
    UsdGenAuthoredPlaneDesc weight;
    weight.name = TfToken("clumpWeight_2");
    weight.domain = UsdGenAuthoredPlaneDomain::Point;
    weight.floatValues = {0.0f, 0.5f, 1.0f};
    curves.authoredPlanes.push_back(weight);
    desc.curveSets.push_back(curves);
    desc.nodes.push_back(SourceNode(SdfPath("/cudaClump/source"), curves.path));
    desc.nodes.push_back(OpNode("/cudaClump/noise", "UsdGenNoise",
                                SdfPath("/cudaClump/source"), 7));
    desc.terminal = SdfPath("/cudaClump/noise");
    UsdGenDiagnostics diagnostics;
    Check(!ValidateCudaGraph(desc, &diagnostics),
          "CUDA rejects imported clump-weighted Noise at admission");
    auto hasClumpError = [](UsdGenDiagnostics const &d) {
        return std::any_of(d.errors.begin(), d.errors.end(),
            [](std::string const &error) {
                return error.find("CUDA Noise does not support imported clump motion planes") !=
                    std::string::npos;
            });
    };
    Check(hasClumpError(diagnostics),
          "CUDA reports the unsupported imported clump motion plane");
    auto &idOnly = desc.curveSets.front().authoredPlanes.front();
    idOnly.name = TfToken("clumpId_2");
    idOnly.type = UsdGenAuthoredPlaneType::Int32;
    idOnly.domain = UsdGenAuthoredPlaneDomain::Primitive;
    idOnly.floatValues.clear();
    idOnly.intValues = {0};
    diagnostics = {};
    ValidateCudaGraph(desc, &diagnostics);
    Check(!hasClumpError(diagnostics), "legacy clump IDs do not trigger motion rejection");
}

static void CheckPartCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/part");
    desc.defaultWidth = 0.01f;
    // A/B straddle the parting line, C is far away, D twins A but is masked.
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0.02f, 0),
                                        GfVec3f(0, -0.02f, 0),
                                        GfVec3f(0, 5, 0),
                                        GfVec3f(0, 0.02f, 0)};
    constexpr int kCvs = 4;
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/part/hair"), roots, kCvs, 0.25f));
    UsdGenCurveSetDesc parting;
    parting.path = SdfPath("/part/parting");
    parting.role = UsdGenRole::Reference;
    parting.curveRole = TfToken("guide");
    parting.curveVertexCounts = {3};
    parting.points = {GfVec3f(-1, 0, 0), GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
    parting.rest = parting.points;
    desc.curveSets.push_back(parting);
    desc.nodes.push_back(
        SourceNode(SdfPath("/part/source"), SdfPath("/part/hair")));
    UsdGenNodeDesc part = OpNode("/part/op", "UsdGenPart",
                                 SdfPath("/part/source"), 7);
    part.references = {parting.path};
    part.params = {{TfToken("part:radius"), VtValue(0.05f), false},
                   {TfToken("part:strength"), VtValue(1.0f), false}};
    desc.nodes.push_back(part);
    desc.terminal = SdfPath("/part/op");
    AddBinding(desc, SdfPath("/part/op"), SdfPath("/part/Expressions/mask"),
               "$primIndex == 3 ? 0 : 1", TfToken("float"), FloatShape(),
               TfToken("mask"), expr::Domain::Primitive, VtValue(1.0f));

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Part cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    UsdGenPlane const *partId = FindCurvePlane(out, "partId");
    Check(partId != nullptr, "Part publishes the partId plane");
    if (!partId) return;
    Check(partId->interpolation == TfToken("uniform") &&
              partId->type == TfToken("int") && partId->arity == 1 &&
              partId->i.size() == 4,
          "partId is a uniform int plane with one entry per strand");
    if (partId->i.size() != 4) return;
    Check(partId->i[0] == 0 && partId->i[1] == 1,
          "Part assigns differing sides across the parting line");
    Check(partId->i[2] == -1, "Part leaves the far strand unparted (-1)");
    Check(partId->i[3] == -1, "Part leaves the masked strand unparted (-1)");

    UsdGenGraphDesc sourceOnly = desc;
    sourceOnly.nodes.pop_back();
    sourceOnly.expressions.clear();
    sourceOnly.terminal = SdfPath("/part/source");
    UsdGenCurveBuffer plain;
    Check(Cook(sourceOnly, &plain), "Part fixture source cooks on its own");
    if (plain.totalCvs == 0) return;
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "Part leaves points bitwise unchanged");
}

static void CheckGrowLiftCook()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/grow");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(5, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/grow/roots"), roots, 2, 0.1f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/grow/source"), SdfPath("/grow/roots")));
    UsdGenNodeDesc grow = OpNode("/grow/op", "UsdGenGrow",
                                 SdfPath("/grow/source"), 11);
    grow.params = {{TfToken("segments"), VtValue(4), false},
                   {TfToken("length"), VtValue(1.0f), false},
                   {TfToken("direction"),
                    VtValue(TfToken("surfaceNormal")), false}};
    desc.nodes.push_back(grow);
    desc.terminal = SdfPath("/grow/op");
    AddBinding(desc, SdfPath("/grow/op"), SdfPath("/grow/Expressions/lift"),
               "$primIndex * 90", TfToken("float"), FloatShape(),
               TfToken("lift"), expr::Domain::Primitive, VtValue(0.0f));

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Grow with a lift expression cooks on the CPU");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 8,
          "Grow emits four CVs per strand");
    // Strand 0 lifts 0 degrees off the surface normal (0,0,1); strand 1 lifts
    // 90 degrees about root-frame B (0,1,0) onto (1,0,0).
    bool rootsFixed = out.px[0] == 0.0f && out.py[0] == 0.0f &&
        out.pz[0] == 0.0f && out.px[4] == 5.0f && out.py[4] == 0.0f &&
        out.pz[4] == 0.0f;
    Check(rootsFixed, "Grow keeps both strand roots exactly fixed");
    Check(Near(out.px[3], 0.0f, 1e-5f) && Near(out.py[3], 0.0f, 1e-5f) &&
              Near(out.pz[3], 1.0f, 1e-5f),
          "Grow combs the unlifted strand along the surface normal");
    Check(Near(out.px[7], 6.0f, 1e-5f) && Near(out.py[7], 0.0f, 1e-5f) &&
              Near(out.pz[7], 0.0f, 1e-5f),
          "Grow combs the lifted strand 90 degrees about root B");
    float const dot = (out.px[3] - out.px[0]) * (out.px[7] - out.px[4]) +
        (out.py[3] - out.py[0]) * (out.py[7] - out.py[4]) +
        (out.pz[3] - out.pz[0]) * (out.pz[7] - out.pz[4]);
    Check(Near(dot, 0.0f, 1e-5f),
          "Grow lift expressions comb strands to different directions");
}

int main()
{
    CheckRegistryContracts();
    CheckExpressionTargets();
    CheckWaveCook();
    CheckCurlCook();
    CheckBendCook();
    CheckClumpMotionConsumers();
    CheckCudaImportedClumpAdmission();
    CheckPartCook();
    CheckGrowLiftCook();
    std::printf("testUsdGenCurlBendWavePart: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
