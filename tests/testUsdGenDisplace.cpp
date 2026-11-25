// UsdGenDisplace operator contract and CPU cooks.
//
// Covers the height-field displacement along the rest root normal:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/
//       OutputPrimvars plus the Topology/Value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): identity at zero effect, the height formula, roots behavior,
//       the mask envelope, and connected-expression cooks for the map
//       (per strand root) and the amount (per CV);
//   (3) mode=vector and unknown modes fail closed.
#include "usdGen/compiler.h"
#include "nodeSets.h"
#include "usdGenOpFixture.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/scheduler.h"

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
    *out = OutputBuffer(graph);
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

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenDisplace");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenDisplace)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenDisplace)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "Displace reports its own type");
    Check(op->Record().Def().topoFx == UsdGenTopoFx::None,
          "Displace has TopologyEffect None");
    Check(op->Record().ReferenceInputs().empty(),
          "Displace declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(), "Displace emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "Displace touches the points plane only");
    std::set<std::string> topo, value;
    UsdGenOpRecord const &rec = registry.Record(registry.Find(type));
    for (UsdGenParamSlot s = 0; s < rec.ParamCount(); ++s)   // routed decls, as the routing snapshot splits them
        (rec.ParamClass(s) == UsdGenParamClass::Value ? value : topo).insert(rec.ParamName(s).GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed", "mode",
                                        "displace:map"},
          "Displace topology parameters are input/surface/seed/mode/map");
    Check(value == std::set<std::string>{"enabled", "mask", "displace:amount",
                                         "displace:base", "displace:scale",
                                         "displace:offset"},
          "Displace value parameters are enabled/mask/amount/base/scale/offset");
}

// --- (2) CPU cooks -----------------------------------------------------
// Unconnected sample is 0 and rootN is (0,0,1), so with literals the height
// field is d = amount * ((0 - base) * scale + offset) * mask along +Z.

static UsdGenGraphDesc HeightDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/displace");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/displace/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/displace/source"), SdfPath("/displace/hair")));
    UsdGenNodeDesc displace = OpNode("/displace/op", "UsdGenDisplace",
                                     SdfPath("/displace/source"), 7);
    displace.params = params;
    desc.nodes.push_back(displace);
    desc.terminal = SdfPath("/displace/op");
    return desc;
}

static void CheckIdentityCook()
{
    UsdGenGraphDesc desc = HeightDesc({{TfToken("displace:amount"), VtValue(0.0f), false}});
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Displace cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/displace/source"), &plain),
          "Displace fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "Displace preserves the strand topology");
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "Displace with amount 0 passes points through bit-for-bit");

    UsdGenGraphDesc masked = HeightDesc({{TfToken("displace:amount"), VtValue(2.0f), false},
                                         {TfToken("mask"), VtValue(0.0f), false}});
    UsdGenCurveBuffer muted;
    Check(Cook(masked, &muted), "Displace with mask 0 cooks");
    if (muted.totalCvs == 0) return;
    Check(muted.px == plain.px && muted.py == plain.py && muted.pz == plain.pz,
          "Displace with mask 0 passes points through bit-for-bit");
}

static void CheckHeightCook()
{
    // d = 2 * ((0 - 0.5) * 1 + 0) * 1 = -1 along rootN (0,0,1).
    UsdGenGraphDesc desc = HeightDesc({{TfToken("displace:amount"), VtValue(2.0f), false},
                                       {TfToken("displace:base"), VtValue(0.5f), false},
                                       {TfToken("displace:scale"), VtValue(1.0f), false},
                                       {TfToken("displace:offset"), VtValue(0.0f), false}});
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Displace height field cooks on the CPU lane");
    if (out.totalCvs == 0) return;
    bool sidesExact = true, depth = true, rootsMoved = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            sidesExact = sidesExact && out.px[o] == rootX &&
                out.py[o] == float(i) * 0.25f;
            depth = depth && Near(out.pz[o], -1.0f, 1e-6f);
        }
        rootsMoved = rootsMoved && Near(out.pz[c * 6], -1.0f, 1e-6f);
    }
    Check(sidesExact, "Displace moves nothing off the root normal axis");
    Check(depth, "Displace offsets every CV by amount*((0-base)*scale+offset)");
    Check(rootsMoved, "Displace moves roots with the height field");

    // d = 4 * ((0 - 0) * 2 + 0.25) * 1 = +1: scale applies to the
    // sample-minus-base term, offset after it.
    UsdGenGraphDesc offset = HeightDesc({{TfToken("displace:amount"), VtValue(4.0f), false},
                                         {TfToken("displace:base"), VtValue(0.0f), false},
                                         {TfToken("displace:scale"), VtValue(2.0f), false},
                                         {TfToken("displace:offset"), VtValue(0.25f), false}});
    UsdGenCurveBuffer raised;
    Check(Cook(offset, &raised), "Displace scale/offset cook runs");
    if (raised.totalCvs == 0) return;
    bool raisedOk = true;
    for (size_t o = 0; o < 12; ++o) raisedOk = raisedOk && Near(raised.pz[o], 1.0f, 1e-6f);
    Check(raisedOk, "Displace applies scale to (sample-base), offset after it");

    // The mask envelopes the field: half mask halves the -1 push.
    UsdGenGraphDesc half = HeightDesc({{TfToken("displace:amount"), VtValue(2.0f), false},
                                       {TfToken("displace:base"), VtValue(0.5f), false},
                                       {TfToken("mask"), VtValue(0.5f), false}});
    UsdGenCurveBuffer halved;
    Check(Cook(half, &halved), "Displace mask envelope cook runs");
    if (halved.totalCvs == 0) return;
    bool halvedOk = true;
    for (size_t o = 0; o < 12; ++o) halvedOk = halvedOk && Near(halved.pz[o], -0.5f, 1e-6f);
    Check(halvedOk, "Displace scales the push by the mask");
}

static void CheckMapExpressionCook()
{
    // Per-strand samples 0.5 and 0.75 against base 0.5: strand 0 sits
    // exactly on the base (bitwise unchanged), strand 1 rises by
    // 2 * (0.75 - 0.5) = 0.5.
    UsdGenGraphDesc desc = HeightDesc({{TfToken("displace:amount"), VtValue(2.0f), false},
                                       {TfToken("displace:base"), VtValue(0.5f), false}});
    AddBinding(desc, SdfPath("/displace/op"),
               SdfPath("/displace/Expressions/map"), "$primIndex * 0.25 + 0.5",
               TfToken("float"), FloatShape(), TfToken("displace:map"),
               expr::Domain::Primitive, VtValue(0.0f));
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "Displace with a map expression cooks on the CPU");
    Check(CookSourceOnly(desc, SdfPath("/displace/source"), &plain),
          "Displace map-cook source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool baseExact = true, raised = true;
    for (int i = 0; i < 6; ++i) {
        baseExact = baseExact && out.px[size_t(i)] == plain.px[size_t(i)] &&
            out.py[size_t(i)] == plain.py[size_t(i)] &&
            out.pz[size_t(i)] == plain.pz[size_t(i)];
        size_t const o = 6 + size_t(i);
        raised = raised && Near(out.pz[o], plain.pz[o] + 0.5f, 1e-6f) &&
            out.px[o] == plain.px[o] && out.py[o] == plain.py[o];
    }
    Check(baseExact, "Displace leaves the strand on the base bit-for-bit");
    Check(raised, "Displace samples the map per strand root");
}

static void CheckAmountExpressionCook()
{
    // Per-CV amount 2t with base 0, scale 1, offset 1 and no map:
    // d = 2t * ((0 - 0) * 1 + 1) = 2t, so the root stays and the tip rises 2.
    UsdGenGraphDesc desc = HeightDesc({{TfToken("displace:base"), VtValue(0.0f), false},
                                       {TfToken("displace:scale"), VtValue(1.0f), false},
                                       {TfToken("displace:offset"), VtValue(1.0f), false}});
    AddBinding(desc, SdfPath("/displace/op"),
               SdfPath("/displace/Expressions/amount"), "$t * 2.0",
               TfToken("float"), FloatShape(), TfToken("displace:amount"),
               expr::Domain::Point, VtValue(0.0f));
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "Displace with a per-CV amount cooks on the CPU");
    if (out.totalCvs == 0) return;
    bool profile = true, sidesExact = true;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 0.0f : 3.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            profile = profile && Near(out.pz[o], 2.0f * float(i) / 5.0f);
            sidesExact = sidesExact && out.px[o] == rootX &&
                out.py[o] == float(i) * 0.25f;
        }
    }
    Check(profile, "Displace varies the push per CV from the amount field");
    Check(sidesExact, "Displace amount cook moves nothing off the normal");
}

// --- (3) fail-closed modes ------------------------------------------------

static void CheckBadModeFailsClosed()
{
    UsdGenGraphDesc vector = HeightDesc({{TfToken("mode"), VtValue(TfToken("vector")), false}});
    UsdGenCurveBuffer out;
    Check(!Cook(vector, &out), "Displace mode=vector fails closed");
    UsdGenGraphDesc bogus = HeightDesc({{TfToken("mode"), VtValue(TfToken("bogus")), false}});
    Check(!Cook(bogus, &out), "Displace with an unknown mode fails closed");
}

int main()
{
    CheckRegistryContracts();
    CheckIdentityCook();
    CheckHeightCook();
    CheckMapExpressionCook();
    CheckAmountExpressionCook();
    CheckBadModeFailsClosed();
    std::printf("testUsdGenDisplace: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
