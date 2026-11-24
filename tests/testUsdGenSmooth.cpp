// UsdGenSmooth operator contract and deterministic CPU cooks.
//
// Covers the alongCurve Laplacian and the neighbours cross-strand blend:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/OutputPrimvars;
//   (2) the alongCurve geometric contract (Jacobi pass, clamped endpoints,
//       lockRoot on/off, negative-strength sharpening);
//   (3) identity at zero effect (strength 0, clamped mask, default neighbours);
//   (4) connected-expression cooks (per-strand strength and mask);
//   (5) the neighbours contract (in-radius averaging, far strands unchanged,
//       the iterations closed form).
#include "usdGen/compiler.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

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
    *out = graph.Output();
    return true;
}

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenSmooth");
    Check(registry.HasKernel(type), "HasKernel(UsdGenSmooth)");
    std::unique_ptr<UsdGenOp> op = registry.Create(type);
    Check(op != nullptr, "Create(UsdGenSmooth)");
    if (!op) return;
    Check(op->Type() == type, "UsdGenSmooth reports its own type");
    Check(op->TopologyEffect() == UsdGenTopoFx::None,
          "UsdGenSmooth has TopologyEffect None");
    Check(op->ReferenceInputs().empty(),
          "UsdGenSmooth declares no reference inputs");
    Check(op->OutputPrimvars().empty(),
          "UsdGenSmooth emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "UsdGenSmooth touches the points plane only");
}

// --- alongCurve fixtures ------------------------------------------------
// Three 6-CV strands with a kinked CV2 (kinked in both points and rest so
// the cook input is the kink whatever channel the source publishes).

constexpr int kCvs = 6;
constexpr float kStep = 0.25f;

static UsdGenGraphDesc KinkedDesc(char const *root)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath(root);
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0),
                                        GfVec3f(6, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath(std::string(root) + "/hair"), roots, kCvs, kStep));
    VtVec3fArray &points = desc.curveSets.back().points;
    VtVec3fArray &rest = desc.curveSets.back().rest;
    for (size_t c = 0; c < 3; ++c) {
        GfVec3f kink(0.5f + 0.25f * float(c), 0.0f, 0.0f);
        points[c * size_t(kCvs) + 2] += kink;
        rest[c * size_t(kCvs) + 2] += kink;
    }
    desc.nodes.push_back(SourceNode(SdfPath(std::string(root) + "/source"),
                                   SdfPath(std::string(root) + "/hair")));
    return desc;
}

static UsdGenNodeDesc SmoothNode(char const *root, int seed)
{
    return OpNode((std::string(root) + "/op").c_str(), "UsdGenSmooth",
                  SdfPath(std::string(root) + "/source"), seed);
}

static GfVec3f InputAt(VtVec3fArray const &points, size_t c, int i)
{
    return points[c * size_t(kCvs) + size_t(i)];
}

// --- (2) the alongCurve geometric contract ------------------------------

static void CheckAlongCurveCook()
{
    UsdGenGraphDesc desc = KinkedDesc("/smooth");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/smooth", 7);
    smooth.params = {{TfToken("strength"), VtValue(1.0f), false},
                     {TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(true), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/smooth/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth alongCurve cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    Check(out.totalCurves == 3 && out.totalCvs == 18,
          "Smooth preserves the strand topology");
    // One Jacobi pass at strength 1: interiors become exact neighbour
    // averages, the tip takes CV n-2, locked roots stay put.
    bool rootsExact = true, profile = true;
    for (size_t c = 0; c < 3; ++c) {
        for (int i = 0; i < kCvs; ++i) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            GfVec3f expected;
            if (i == 0) expected = InputAt(input, c, 0);
            else if (i == kCvs - 1) expected = InputAt(input, c, kCvs - 2);
            else
                expected = 0.5f * (InputAt(input, c, i - 1) + InputAt(input, c, i + 1));
            profile = profile && Near(out.px[o], expected[0], 1e-5f) &&
                Near(out.py[o], expected[1], 1e-5f) &&
                Near(out.pz[o], expected[2], 1e-5f);
            if (i == 0)
                rootsExact = rootsExact && out.px[o] == expected[0] &&
                    out.py[o] == expected[1] && out.pz[o] == expected[2];
        }
    }
    Check(rootsExact, "Smooth lockRoot keeps every root exactly fixed");
    Check(profile, "Smooth strength 1 replaces interiors by neighbour averages");
}

static void CheckLockRootOffCook()
{
    UsdGenGraphDesc desc = KinkedDesc("/unlocked");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/unlocked", 7);
    smooth.params = {{TfToken("strength"), VtValue(1.0f), false},
                     {TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(false), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/unlocked/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth lockRoot=false cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // Unlocked roots relax onto CV1 (the clamped single neighbour).
    bool rootsRelaxed = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const o = c * size_t(kCvs);
        GfVec3f const expected = InputAt(input, c, 1);
        rootsRelaxed = rootsRelaxed && Near(out.px[o], expected[0], 1e-5f) &&
            Near(out.py[o], expected[1], 1e-5f) &&
            Near(out.pz[o], expected[2], 1e-5f);
    }
    Check(rootsRelaxed, "Smooth lockRoot=false relaxes roots onto CV1");
}

static void CheckSharpenCook()
{
    UsdGenGraphDesc desc = KinkedDesc("/sharpen");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/sharpen", 7);
    smooth.params = {{TfToken("strength"), VtValue(-1.0f), false},
                     {TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(true), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/sharpen/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth negative strength cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // strength -1 extrapolates: out = 2*in - avg, so the kink grows.
    bool formula = true, amplified = true;
    for (size_t c = 0; c < 3; ++c) {
        size_t const o = c * size_t(kCvs) + 2;
        GfVec3f const avg =
            0.5f * (InputAt(input, c, 1) + InputAt(input, c, 3));
        GfVec3f const expected = 2.0f * InputAt(input, c, 2) - avg;
        formula = formula && Near(out.px[o], expected[0], 1e-5f) &&
            Near(out.py[o], expected[1], 1e-5f) &&
            Near(out.pz[o], expected[2], 1e-5f);
        amplified = amplified &&
            std::fabs(out.px[o] - avg[0]) > std::fabs(InputAt(input, c, 2)[0] - avg[0]);
    }
    Check(formula, "Smooth strength -1 extrapolates away from the average");
    Check(amplified, "Smooth negative strength amplifies the kink");
}

// --- (3) identity at zero effect ------------------------------------------

static void CheckIdentityCooks()
{
    // strength 0 is a bitwise pass-through.
    UsdGenGraphDesc desc = KinkedDesc("/identity");
    UsdGenNodeDesc smooth = SmoothNode("/identity", 7);
    smooth.params = {{TfToken("strength"), VtValue(0.0f), false},
                     {TfToken("iterations"), VtValue(3), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/identity/op");
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth strength 0 cooks on the CPU lane");
    UsdGenGraphDesc sourceOnly = KinkedDesc("/identity");
    sourceOnly.terminal = SdfPath("/identity/source");
    UsdGenCurveBuffer plain;
    Check(Cook(sourceOnly, &plain), "Smooth identity fixture source cooks");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "Smooth strength 0 leaves points bitwise unchanged");

    // A negative mask clamps to 0: identity again.
    UsdGenGraphDesc negMask = KinkedDesc("/negmask");
    UsdGenNodeDesc negOp = SmoothNode("/negmask", 7);
    negOp.params = {{TfToken("strength"), VtValue(1.0f), false},
                    {TfToken("mask"), VtValue(-1.0f), false}};
    negMask.nodes.push_back(negOp);
    negMask.terminal = SdfPath("/negmask/op");
    UsdGenCurveBuffer negOut;
    Check(Cook(negMask, &negOut), "Smooth negative-mask cook runs");
    UsdGenGraphDesc negSource = KinkedDesc("/negmask");
    negSource.terminal = SdfPath("/negmask/source");
    UsdGenCurveBuffer negPlain;
    Check(Cook(negSource, &negPlain), "Smooth negative-mask source cooks");
    if (negOut.totalCvs == 0 || negPlain.totalCvs == 0) return;
    Check(negOut.px == negPlain.px && negOut.py == negPlain.py &&
              negOut.pz == negPlain.pz,
          "Smooth clamps a negative mask to a bitwise identity");

    // A mask above 1 clamps to 1: identical to the unmasked cook.
    UsdGenGraphDesc hiMask = KinkedDesc("/himask");
    UsdGenNodeDesc hiOp = SmoothNode("/himask", 7);
    hiOp.params = {{TfToken("strength"), VtValue(1.0f), false},
                   {TfToken("iterations"), VtValue(1), false},
                   {TfToken("mask"), VtValue(2.0f), false}};
    hiMask.nodes.push_back(hiOp);
    hiMask.terminal = SdfPath("/himask/op");
    UsdGenCurveBuffer hiOut;
    Check(Cook(hiMask, &hiOut), "Smooth over-range-mask cook runs");
    UsdGenGraphDesc loMask = KinkedDesc("/himask");
    loMask.nodes.push_back(SmoothNode("/himask", 7));
    loMask.nodes.back().params = {{TfToken("strength"), VtValue(1.0f), false},
                                  {TfToken("iterations"), VtValue(1), false}};
    loMask.terminal = SdfPath("/himask/op");
    UsdGenCurveBuffer loOut;
    Check(Cook(loMask, &loOut), "Smooth default-mask cook runs");
    if (hiOut.totalCvs == 0 || loOut.totalCvs == 0) return;
    Check(hiOut.px == loOut.px && hiOut.py == loOut.py && hiOut.pz == loOut.pz,
          "Smooth clamps an over-range mask to 1");

    // neighbours mode at the default radius 0 queries nothing: identity.
    UsdGenGraphDesc def = KinkedDesc("/defnb");
    UsdGenNodeDesc defOp = SmoothNode("/defnb", 7);
    defOp.params = {{TfToken("mode"), VtValue(TfToken("neighbours")), false},
                    {TfToken("strength"), VtValue(1.0f), false}};
    def.nodes.push_back(defOp);
    def.terminal = SdfPath("/defnb/op");
    UsdGenCurveBuffer defOut;
    Check(Cook(def, &defOut), "Smooth default-neighbours cook runs");
    UsdGenGraphDesc defSource = KinkedDesc("/defnb");
    defSource.terminal = SdfPath("/defnb/source");
    UsdGenCurveBuffer defPlain;
    Check(Cook(defSource, &defPlain), "Smooth default-neighbours source cooks");
    if (defOut.totalCvs == 0 || defPlain.totalCvs == 0) return;
    Check(defOut.px == defPlain.px && defOut.py == defPlain.py &&
              defOut.pz == defPlain.pz,
          "Smooth neighbours at radius 0 leaves points bitwise unchanged");
}

// --- (4) connected-expression cooks -----------------------------------------

static void CheckStrengthExpressionCook()
{
    UsdGenGraphDesc desc = KinkedDesc("/expr");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/expr", 7);
    smooth.params = {{TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(true), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/expr/op");
    AddBinding(desc, SdfPath("/expr/op"), SdfPath("/expr/Expressions/strength"),
               "$primIndex * 0.5", TfToken("float"), FloatShape(),
               TfToken("strength"), expr::Domain::Primitive, VtValue(0.5f));

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth strength-expression cook runs");
    if (out.totalCvs == 0) return;
    // Per-strand strengths 0, 0.5, 1: strand 0 bitwise fixed, strand 2
    // fully averaged, strand 1 halfway there.
    bool fixed = true, half = true, full = true;
    for (int i = 0; i < kCvs; ++i) {
        size_t const o0 = size_t(i);
        fixed = fixed && out.px[o0] == InputAt(input, 0, i)[0] &&
            out.py[o0] == InputAt(input, 0, i)[1] &&
            out.pz[o0] == InputAt(input, 0, i)[2];
        for (size_t c = 1; c <= 2; ++c) {
            size_t const o = c * size_t(kCvs) + size_t(i);
            GfVec3f avg = InputAt(input, c, i);
            if (i > 0 && i + 1 < kCvs)
                avg = 0.5f * (InputAt(input, c, i - 1) + InputAt(input, c, i + 1));
            else if (i == kCvs - 1)
                avg = InputAt(input, c, kCvs - 2);
            float const s = c == 1 ? 0.5f : 1.0f;
            GfVec3f const in = InputAt(input, c, i);
            GfVec3f const expected = in + s * (avg - in);
            bool const match = Near(out.px[o], expected[0], 1e-5f) &&
                Near(out.py[o], expected[1], 1e-5f) &&
                Near(out.pz[o], expected[2], 1e-5f);
            if (c == 1) half = half && match;
            else full = full && match;
        }
    }
    Check(fixed, "Smooth strength 0 strand stays bitwise fixed");
    Check(half, "Smooth strength 0.5 strand lerps halfway to the average");
    Check(full, "Smooth strength 1 strand reaches the neighbour average");
}

static void CheckMaskExpressionCook()
{
    UsdGenGraphDesc desc = KinkedDesc("/maskexpr");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/maskexpr", 7);
    smooth.params = {{TfToken("strength"), VtValue(1.0f), false},
                     {TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(true), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/maskexpr/op");
    AddBinding(desc, SdfPath("/maskexpr/op"),
               SdfPath("/maskexpr/Expressions/mask"), "$primIndex == 1 ? 0 : 1",
               TfToken("float"), FloatShape(), TfToken("mask"),
               expr::Domain::Primitive, VtValue(1.0f));

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth mask-expression cook runs");
    if (out.totalCvs == 0) return;
    bool maskedFixed = true, othersSmooth = true;
    for (int i = 0; i < kCvs; ++i) {
        size_t const o = size_t(kCvs) + size_t(i);
        maskedFixed = maskedFixed && out.px[o] == InputAt(input, 1, i)[0] &&
            out.py[o] == InputAt(input, 1, i)[1] &&
            out.pz[o] == InputAt(input, 1, i)[2];
        for (size_t c : {size_t(0), size_t(2)}) {
            size_t const q = c * size_t(kCvs) + size_t(i);
            GfVec3f expected = InputAt(input, c, i);
            if (i > 0 && i + 1 < kCvs)
                expected = 0.5f * (InputAt(input, c, i - 1) + InputAt(input, c, i + 1));
            else if (i == kCvs - 1)
                expected = InputAt(input, c, kCvs - 2);
            othersSmooth = othersSmooth && Near(out.px[q], expected[0], 1e-5f) &&
                Near(out.py[q], expected[1], 1e-5f) &&
                Near(out.pz[q], expected[2], 1e-5f);
        }
    }
    Check(maskedFixed, "Smooth masked strand stays bitwise fixed");
    Check(othersSmooth, "Smooth unmasked strands reach the neighbour average");
}

// --- (5) the neighbours contract ---------------------------------------------

constexpr int kNbCvs = 4;

static UsdGenGraphDesc NeighboursDesc(char const *root)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath(root);
    desc.defaultWidth = 0.01f;
    // A close pair (roots 0.1 apart, strand 1 lifted in +Z past its root)
    // plus a far strand no radius reaches.
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0),
                                        GfVec3f(5, 0, 0)};
    desc.curveSets.push_back(StraightStrands(
        SdfPath(std::string(root) + "/hair"), roots, kNbCvs, kStep));
    VtVec3fArray &points = desc.curveSets.back().points;
    VtVec3fArray &rest = desc.curveSets.back().rest;
    for (int i = 1; i < kNbCvs; ++i) {
        points[size_t(kNbCvs) + size_t(i)] += GfVec3f(0, 0, 1.0f);
        rest[size_t(kNbCvs) + size_t(i)] += GfVec3f(0, 0, 1.0f);
    }
    desc.nodes.push_back(SourceNode(SdfPath(std::string(root) + "/source"),
                                   SdfPath(std::string(root) + "/hair")));
    return desc;
}

static GfVec3f NbInputAt(VtVec3fArray const &points, size_t c, int i)
{
    return points[c * size_t(kNbCvs) + size_t(i)];
}

static void CheckNeighboursCook()
{
    UsdGenGraphDesc desc = NeighboursDesc("/neighbours");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/neighbours", 7);
    smooth.params = {{TfToken("mode"), VtValue(TfToken("neighbours")), false},
                     {TfToken("strength"), VtValue(1.0f), false},
                     {TfToken("iterations"), VtValue(1), false},
                     {TfToken("lockRoot"), VtValue(true), false},
                     {TfToken("searchRadius"), VtValue(1.0f), false},
                     {TfToken("numNeighbors"), VtValue(4), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/neighbours/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth neighbours cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    // Each strand blends toward its neighbours' CVs, self excluded (the same
    // self-excluding Laplacian as alongCurve): at strength 1 the pair swaps
    // CV by CV past the locked roots. The far strand is outside every
    // radius and stays bitwise fixed.
    bool pairAveraged = true, rootsExact = true, farFixed = true;
    for (int i = 0; i < kNbCvs; ++i) {
        for (size_t c = 0; c < 2; ++c) {
            size_t const o = c * size_t(kNbCvs) + size_t(i);
            GfVec3f const expected =
                i == 0 ? NbInputAt(input, c, 0) : NbInputAt(input, 1 - c, i);
            pairAveraged = pairAveraged && Near(out.px[o], expected[0], 1e-5f) &&
                Near(out.py[o], expected[1], 1e-5f) &&
                Near(out.pz[o], expected[2], 1e-5f);
            if (i == 0)
                rootsExact = rootsExact && out.px[o] == expected[0] &&
                    out.py[o] == expected[1] && out.pz[o] == expected[2];
        }
        size_t const f = 2 * size_t(kNbCvs) + size_t(i);
        farFixed = farFixed && out.px[f] == NbInputAt(input, 2, i)[0] &&
            out.py[f] == NbInputAt(input, 2, i)[1] &&
            out.pz[f] == NbInputAt(input, 2, i)[2];
    }
    Check(pairAveraged, "Smooth neighbours blends toward the in-radius neighbour");
    Check(rootsExact, "Smooth neighbours keeps locked roots exactly fixed");
    Check(farFixed, "Smooth leaves the out-of-radius strand bitwise fixed");
}

static void CheckNeighboursIterationsCook()
{
    UsdGenGraphDesc desc = NeighboursDesc("/nbiter");
    VtVec3fArray const input = desc.curveSets.back().points;
    UsdGenNodeDesc smooth = SmoothNode("/nbiter", 7);
    smooth.params = {{TfToken("mode"), VtValue(TfToken("neighbours")), false},
                     {TfToken("strength"), VtValue(0.5f), false},
                     {TfToken("iterations"), VtValue(2), false},
                     {TfToken("lockRoot"), VtValue(true), false},
                     {TfToken("searchRadius"), VtValue(1.0f), false},
                     {TfToken("numNeighbors"), VtValue(4), false}};
    desc.nodes.push_back(smooth);
    desc.terminal = SdfPath("/nbiter/op");

    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Smooth neighbours-iterations cook runs");
    if (out.totalCvs == 0) return;
    // Two passes at alpha 0.5 toward the fixed self-excluded target collapse
    // to one lerp with 1 - 0.5^2 = 0.75 toward the other strand's input CV.
    bool closed = true;
    for (size_t c = 0; c < 2; ++c) {
        for (int i = 1; i < kNbCvs; ++i) {
            size_t const o = c * size_t(kNbCvs) + size_t(i);
            GfVec3f const target = NbInputAt(input, 1 - c, i);
            GfVec3f const in = NbInputAt(input, c, i);
            GfVec3f const expected = in + 0.75f * (target - in);
            closed = closed && Near(out.px[o], expected[0], 1e-5f) &&
                Near(out.py[o], expected[1], 1e-5f) &&
                Near(out.pz[o], expected[2], 1e-5f);
        }
    }
    Check(closed, "Smooth neighbours iterations follow 1 - (1 - a)^k");
}

int main()
{
    usdGenRegisterM1Operators();
    CheckRegistryContracts();
    CheckAlongCurveCook();
    CheckLockRootOffCook();
    CheckSharpenCook();
    CheckIdentityCooks();
    CheckStrengthExpressionCook();
    CheckMaskExpressionCook();
    CheckNeighboursCook();
    CheckNeighboursIterationsCook();
    std::printf("testUsdGenSmooth: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
