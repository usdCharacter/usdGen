// UsdGenExprOp operator contract and CPU cooks.
//
// Covers the capture-time-only expression styler (02-schema.md §2.7,
// 04-operators.md §3, 07 §7.8):
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/
//       OutputPrimvars/InputPrimvars/PlanesTouched plus the Topology/Value
//       parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): displacement and width at cv and curve granularity, the
//       mask envelope (literal and connected), and the negative-width clamp;
//   (3) every edge fails closed: empty/blank/uncompilable source,
//       unknown mode/returnType tokens, returnType=color (no vec3 color
//       plane in the emitted-primvar transport), $frame/$value reads,
//       sampler calls with and without expr:maps targets, and a wrongly
//       typed source;
//   (4) expr:maps transport: unused targets warn and are ignored, an
//       undescribed target fails at compile;
//   (5) determinism (two cooks are bit-identical);
//   (6) the CUDA backend explicitly rejects the CPU-only graph, like Clump.
#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/exec/device/executionBackend.h"
#include "usdGenExpr/expressions/context.h"
#include "usdGen/compile/graph.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/exec/scheduler.h"

#include <cmath>
#include <cstdio>
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
static bool Contains(std::vector<std::string> const &messages, char const *needle)
{
    for (auto const &m : messages)
        if (m.find(needle) != std::string::npos) return true;
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

static bool CookDiag(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
                     std::vector<std::string> *errors,
                     std::vector<std::string> *warnings)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        if (errors) *errors = compiled.errors;
        for (auto const &e : compiled.errors)
            std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Spec();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (errors) *errors = run.diagnostics.errors;
    if (warnings) *warnings = run.diagnostics.warnings;
    for (auto const &e : run.diagnostics.errors)
        std::printf("  run: %s\n", e.c_str());
    if (run.diagnostics.HasErrors()) return false;
    *out = OutputBuffer(graph);
    return true;
}

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out)
{
    return CookDiag(desc, out, nullptr, nullptr);
}

static bool CookSourceOnly(UsdGenGraphDesc const &desc, SdfPath const &source,
                           UsdGenCurveBuffer *out)
{
    UsdGenGraphDesc plain = desc;
    plain.nodes.pop_back();
    plain.expressions.clear();
    plain.maps.clear();
    plain.terminal = source;
    return Cook(plain, out);
}

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenExprOp");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenExprOp)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenExprOp)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "ExprOp reports its own type");
    Check(op->Record().Def().countChanges == 0,
          "ExprOp has TopologyEffect None");
    Check(op->Record().ReferenceInputs().empty(),
          "ExprOp declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(), "ExprOp emits no output primvars");
    Check(op->Record().InputPlanes().empty(), "ExprOp reads no input primvars");
    Check(op->PlanesTouched() == (UsdGenOp::kPlanePoints | UsdGenOp::kPlaneWidths),
          "ExprOp touches the points and widths planes");
    std::set<std::string> topo, value;
    for (TfToken const &t : usdGenTest::RoutedParams(op->Record(), false)) topo.insert(t.GetString());
    for (TfToken const &t : usdGenTest::RoutedParams(op->Record(), true)) value.insert(t.GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed", "mode",
                                        "expr:returnType", "expr:source",
                                        "expr:maps"},
          "ExprOp topology parameters are input/surface/seed/mode/returnType/source/maps");
    Check(value == std::set<std::string>{"enabled", "mask"},
          "ExprOp value parameters are enabled/mask");
}

// --- (2) CPU cooks -----------------------------------------------------

static UsdGenGraphDesc ExprDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/exprop");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/exprop/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/exprop/source"), SdfPath("/exprop/hair")));
    UsdGenNodeDesc op = OpNode("/exprop/op", "UsdGenExprOp",
                               SdfPath("/exprop/source"), 7);
    op.params = params;
    desc.nodes.push_back(op);
    desc.terminal = SdfPath("/exprop/op");
    return desc;
}

static void CheckDisplacementCvCook()
{
    // Schema defaults (mode=cv, returnType=displacement) apply: the
    // per-CV vec3 [$t*2, 0, -$t] is added to every point. On straight
    // strands $t is i/5 whether it reads hairT or the arc-length fallback.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t * 2, 0, -$t]")), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "ExprOp displacement/cv cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/exprop/source"), &plain),
          "ExprOp displacement fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "ExprOp preserves the strand topology");
    bool profile = true, sidesExact = true, rootsExact = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            float const t = float(i) / 5.0f;
            profile = profile && Near(out.px[o], rootX + 2.0f * t, 1e-6f) &&
                Near(out.pz[o], -t, 1e-6f);
            sidesExact = sidesExact && out.py[o] == float(i) * 0.25f;
            if (i == 0)
                rootsExact = rootsExact && out.px[o] == plain.px[o] &&
                    out.py[o] == plain.py[o] && out.pz[o] == plain.pz[o];
        }
    }
    Check(profile, "ExprOp adds the per-CV expression vec3 to the points");
    Check(sidesExact, "ExprOp displacement moves nothing off its axes");
    Check(rootsExact, "ExprOp copies zero-displacement roots bit-for-bit");
    Check(out.width == plain.width,
          "ExprOp displacement passes widths through bit-for-bit");
}

static void CheckDisplacementCurveCook()
{
    // Per-root broadcast: strand 0 rises by (0,1,0), strand 1 by (1,1,0),
    // uniformly across all six CVs of each strand.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$primIndex, 1, 0]")), false},
         {TfToken("mode"), VtValue(TfToken("curve")), false}});
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "ExprOp displacement/curve cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    bool exact = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            exact = exact && out.px[o] == rootX + float(c) &&
                out.py[o] == float(i) * 0.25f + 1.0f && out.pz[o] == 0.0f;
        }
    }
    Check(exact, "ExprOp broadcasts the per-root vec3 to every CV of the strand");
}

static void CheckWidthCvCook()
{
    // The expression result SETS the widths; the points pass through.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$t * 0.1 + 0.05")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "ExprOp width/cv cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/exprop/source"), &plain),
          "ExprOp width fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool profile = true;
    for (size_t o = 0; o < 12; ++o) {
        float const t = float(o % 6) / 5.0f;
        profile = profile && Near(out.width[o], t * 0.1f + 0.05f, 1e-6f);
    }
    Check(profile, "ExprOp sets per-CV widths from the expression");
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "ExprOp width passes points through bit-for-bit");
}

static void CheckWidthCurveCook()
{
    // Per-root broadcast: one width per strand, bitwise constant within it.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$primIndex * 0.25 + 0.1")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false},
         {TfToken("mode"), VtValue(TfToken("curve")), false}});
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "ExprOp width/curve cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    bool broadcast = true, values = true;
    for (size_t c = 0; c < 2; ++c) {
        float const first = out.width[c * 6];
        for (int i = 0; i < 6; ++i)
            broadcast = broadcast && out.width[c * 6 + size_t(i)] == first;
        values = values && Near(first, float(c) * 0.25f + 0.1f, 1e-6f);
    }
    Check(broadcast, "ExprOp broadcasts the per-root width within each strand");
    Check(values, "ExprOp per-strand widths match the expression");
}

static void CheckMaskCooks()
{
    // A zero literal mask is a bit-for-bit pass-through of the target plane.
    UsdGenGraphDesc muted = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[2, 0, 0]")), false},
         {TfToken("mask"), VtValue(0.0f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(muted, &out), "ExprOp with mask 0 cooks");
    Check(CookSourceOnly(muted, SdfPath("/exprop/source"), &plain),
          "ExprOp mask fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "ExprOp with mask 0 passes points through bit-for-bit");

    // A half mask halves the add: [2,0,0] pushes every CV by exactly +1 in x.
    UsdGenGraphDesc half = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[2, 0, 0]")), false},
         {TfToken("mask"), VtValue(0.5f), false}});
    UsdGenCurveBuffer halved;
    Check(Cook(half, &halved), "ExprOp mask envelope cook runs");
    if (halved.totalCvs == 0) return;
    bool halvedOk = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            halvedOk = halvedOk && halved.px[o] == rootX + 1.0f &&
                halved.py[o] == float(i) * 0.25f && halved.pz[o] == 0.0f;
        }
    }
    Check(halvedOk, "ExprOp scales the displacement add by the mask");

    // Width blends toward upstream: mask 0 keeps the authored 0.1 widths.
    UsdGenGraphDesc widthMuted = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("0.5")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false},
         {TfToken("mask"), VtValue(0.0f), false}});
    UsdGenCurveBuffer kept;
    Check(Cook(widthMuted, &kept), "ExprOp width with mask 0 cooks");
    if (kept.totalCvs == 0) return;
    Check(kept.width == plain.width,
          "ExprOp width with mask 0 keeps the upstream widths bit-for-bit");

    // A connected per-CV mask envelopes the add per CV: the root (mask 0)
    // stays bitwise, the tip (mask 1) moves the full +2.
    UsdGenGraphDesc connected = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[2, 0, 0]")), false}});
    AddBinding(connected, SdfPath("/exprop/op"),
               SdfPath("/exprop/Expressions/mask"), "$t",
               TfToken("float"), FloatShape(), TfToken("mask"),
               expr::Domain::Point, VtValue(1.0f));
    UsdGenCurveBuffer enveloped;
    Check(Cook(connected, &enveloped), "ExprOp with a connected mask cooks");
    if (enveloped.totalCvs == 0) return;
    bool envelope = true, rootExact = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            envelope = envelope &&
                Near(enveloped.px[o], rootX + 2.0f * float(i) / 5.0f, 1e-6f);
            if (i == 0)
                rootExact = rootExact && enveloped.px[o] == plain.px[o] &&
                    enveloped.py[o] == plain.py[o] && enveloped.pz[o] == plain.pz[o];
        }
    }
    Check(envelope, "ExprOp varies the add per CV from the connected mask");
    Check(rootExact, "ExprOp keeps zero-mask CVs bit-for-bit under a connected mask");
}

static void CheckNegativeWidthClamp()
{
    // Negative widths clamp to zero (Width's replace-path rule), they do
    // not fail the cook.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("-1")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "ExprOp negative-width cook runs");
    if (out.totalCvs == 0) return;
    bool clamped = true;
    for (size_t o = 0; o < 12; ++o) clamped = clamped && out.width[o] == 0.0f;
    Check(clamped, "ExprOp clamps negative expression widths to zero");
}

// --- (3) fail-closed edges ------------------------------------------------

static void CheckEdge(UsdGenGraphDesc const &desc, char const *needle, char const *what)
{
    UsdGenCurveBuffer out;
    std::vector<std::string> errors;
    bool const ok = CookDiag(desc, &out, &errors, nullptr);
    Check(!ok, what);
    if (needle) Check(Contains(errors, needle), std::string(what) + " (message)");
}

static void CheckFailClosedEdges()
{
    UsdGenGraphDesc empty = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("")), false}});
    CheckEdge(empty, "expr:source is empty", "ExprOp with an empty source fails closed");
    UsdGenGraphDesc blank = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("  \n ")), false}});
    CheckEdge(blank, "expr:source is empty", "ExprOp with a blank source fails closed");
    UsdGenGraphDesc syntax = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$t *")), false}});
    CheckEdge(syntax, "UsdGenExprOp", "ExprOp with a syntax error fails closed");
    UsdGenGraphDesc nonFinite = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("1/0")), false}});
    CheckEdge(nonFinite, "UsdGenExprOp", "ExprOp with a non-finite result fails closed");
    UsdGenGraphDesc bogusMode = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$t")), false},
         {TfToken("mode"), VtValue(TfToken("bogus")), false}});
    CheckEdge(bogusMode, "usdGen:mode", "ExprOp with an unknown mode fails closed");
    UsdGenGraphDesc bogusReturn = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$t")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("bogus")), false}});
    CheckEdge(bogusReturn, "expr:returnType",
              "ExprOp with an unknown returnType fails closed");
    UsdGenGraphDesc color = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t, 0, 1]")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("color")), false}});
    CheckEdge(color, "'color' is not supported",
              "ExprOp with returnType=color fails closed");
    UsdGenGraphDesc frame = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$frame")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    CheckEdge(frame, "$frame", "ExprOp reading $frame fails closed");
    UsdGenGraphDesc value = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("$value * 2")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    CheckEdge(value, "$value", "ExprOp reading $value fails closed");
    UsdGenGraphDesc ptex = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("ptex(\"a\")")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    CheckEdge(ptex, "input:<name>",
              "ExprOp calling ptex() without maps fails closed");
    UsdGenGraphDesc sampler = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("geoSampler(\"g\", \"$Q\")")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    CheckEdge(sampler, "UsdGenExprOp", "ExprOp calling geoSampler() fails closed");
    UsdGenGraphDesc mistyped = ExprDesc({{TfToken("expr:source"), VtValue(1.0f), false}});
    CheckEdge(mistyped, "must be a string",
              "ExprOp with a non-string source fails closed");
}

// --- (4) expr:maps transport ------------------------------------------------

static void CheckMapsTransport()
{
    // A sampler-free expression ignores rel targets with a warning; the cook
    // succeeds. The map descriptor needs no asset: nothing samples it.
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t, 0, 0]")), false}});
    UsdGenMapDesc map;
    map.path = SdfPath("/Maps/a");
    map.type = TfToken("UsdGenPtexMap");
    desc.maps.push_back(map);
    desc.nodes.back().maps = {SdfPath("/Maps/a")};
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(CookDiag(desc, &out, &errors, &warnings),
          "ExprOp with unused expr:maps cooks");
    Check(Contains(warnings, "expr:maps"),
          "ExprOp warns that unused expr:maps targets are ignored");

    // A sampler call names the expr:maps targets that do exist.
    UsdGenGraphDesc ptex = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("ptex(\"a\")")), false},
         {TfToken("expr:returnType"), VtValue(TfToken("width")), false}});
    ptex.maps.push_back(map);
    ptex.nodes.back().maps = {SdfPath("/Maps/a")};
    CheckEdge(ptex, "/Maps/a",
              "ExprOp ptex() failure names the expr:maps targets");

    // A rel target with no descriptor fails at compile (compiler diagnostic).
    UsdGenGraphDesc missing = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t, 0, 0]")), false}});
    missing.nodes.back().maps = {SdfPath("/Maps/missing")};
    CheckEdge(missing, "unresolved map value",
              "ExprOp with an undescribed expr:maps target fails at compile");
}

// --- (5) determinism ------------------------------------------------------

static void CheckDeterminism()
{
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t * 2, noise($P * 3), $id]")), false}});
    UsdGenCurveBuffer first, second;
    Check(Cook(desc, &first), "ExprOp determinism first cook runs");
    Check(Cook(desc, &second), "ExprOp determinism second cook runs");
    if (first.totalCvs == 0 || second.totalCvs == 0) return;
    Check(first.px == second.px && first.py == second.py && first.pz == second.pz &&
              first.width == second.width,
          "ExprOp cooks bit-identical results twice");
}

// --- (6) CUDA-backend explicit rejection ------------------------------------

static void CheckCudaRejection()
{
    // The capability matrix (~cudaExecution.cpp:1475) admits only
    // Scatter/Grow/CurveSource/Width/Length/Noise/Deform/ReferenceSource/
    // WidthBlend: ExprOp has no row, exactly like Clump.
    Check(GetCudaExecutionCapabilityMatrix().Find(TfToken("UsdGenExprOp")) == nullptr,
          "ExprOp has no CUDA capability row (CPU-only, like Clump)");
    UsdGenGraphDesc desc = ExprDesc(
        {{TfToken("expr:source"), VtValue(std::string("[$t, 0, 0]")), false}});
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = usdGenUsd::Compile(compiler, desc, &graph);
    Check(!compiled.ok, "ExprOp graph is refused for the CUDA backend");
    // A CUDA-enabled build names the operator in the capability-matrix
    // rejection; a CPU-only build refuses the backend itself before ever
    // reaching the planner.
    Check(Contains(compiled.errors, "UsdGenExprOp") ||
              Contains(compiled.errors, "was not enabled in this build"),
          "CUDA refusal names the operator or the missing backend");
}

int main()
{
    CheckRegistryContracts();
    CheckDisplacementCvCook();
    CheckDisplacementCurveCook();
    CheckWidthCvCook();
    CheckWidthCurveCook();
    CheckMaskCooks();
    CheckNegativeWidthClamp();
    CheckFailClosedEdges();
    CheckMapsTransport();
    CheckDeterminism();
    CheckCudaRejection();
    std::printf("testUsdGenExprOp: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
