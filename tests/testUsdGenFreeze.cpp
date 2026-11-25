// UsdGenFreeze operator contract and CPU cooks.
//
// Covers the session-tier freeze (02 §2.9):
//   (1) registry Find/Record/Create, topoFx/role/arity/ReferenceInputs/
//       planes, PlanesTouched plus the Topology/Value partition;
//   (2) happy-path numerics: frozen chain-input snapshot, frozen explicit
//       snapshot (points, widths, curve ids), live passthrough, the
//       no-widths defaultWidth fill;
//   (3) frozen-mode immunity to upstream edits (same-graph mutate + rerun,
//       plus kernel-level digest/ValidForTopology proofs), deep-copy
//       aliasing, determinism (fresh cook twice, same graph twice);
//   (4) fail-closed edges: bad mode/tier/epoch, snapshot layout mismatch,
//       empty snapshot, missing input arity, connected expressions;
//   (5) explicit CUDA-backend rejection (capability matrix has no row);
//   (6) tier lowering: sublayer/payload cook as session with a warning.
//
// Freeze is a built-in op (ops/registerCurves.cpp). It keeps the scheduler's
// rest/named-plane ownership (kUsdGenOpCaptureOwnsRestAndPlanes), so frozen
// output carries the snapshot's rest and named planes instead of the
// upstream ones (asserted by the immunity checks below). The default
// expression Reject covers Freeze (asserted below).
#include "usdGen/compiler.h"
#include "bufferOpCalls.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/expressions/context.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/ops/freeze.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
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

// Views the kernel can read slots from, as the compiler resolves them.
static UsdGenOpRecord const &FreezeRecord()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    return registry.Record(registry.Find(TfToken("UsdGenFreeze")));
}
static expr::ValueShape FloatShape(uint32_t n = 1)
{
    return {expr::ScalarType::Float32, 1, n, 1, 1, false};
}

// Straight strands along +Y from the given roots. Mirrors the CurveSource
// fixture (surface-free, rebind never): valid skin bindings plus identity
// root frames, so rootT=(1,0,0), rootB=(0,1,0), rootN=(0,0,1).
static UsdGenCurveSetDesc StraightStrands(SdfPath const &path,
                                          std::vector<GfVec3f> const &roots,
                                          int cvs, float step,
                                          UsdGenRole role = UsdGenRole::Curves,
                                          float width = 0.1f,
                                          uint64_t idBase = 0,
                                          bool withWidths = true)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = role;
    set.curveRole = TfToken(role == UsdGenRole::Reference ? "guide" : "hair");
    size_t const n = roots.size();
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * size_t(cvs));
    for (size_t c = 0; c < n; ++c)
        for (int i = 0; i < cvs; ++i)
            set.points[c * size_t(cvs) + size_t(i)] =
                roots[c] + GfVec3f(0.0f, float(i) * step, 0.0f);
    set.rest = set.points;
    if (withWidths) set.widths.assign(n * size_t(cvs), width);
    set.curveId.resize(n);
    for (size_t c = 0; c < n; ++c) set.curveId[c] = idBase + uint64_t(c);
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

static bool CookFull(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
                     std::vector<std::string> *errors,
                     std::vector<std::string> *warnings)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        *errors = compiled.errors;
        for (auto const &e : compiled.errors)
            std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    *errors = run.diagnostics.errors;
    *warnings = run.diagnostics.warnings;
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
    std::vector<std::string> errors, warnings;
    return CookFull(desc, out, &errors, &warnings);
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

static bool AnyContains(std::vector<std::string> const &messages, char const *needle)
{
    for (auto const &m : messages)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// Private array copies: VtArray assignment is CoW, so a plain copy would
// still alias the graph's storage across a second Run.
template <typename Array>
static Array PrivateCopy(Array const &src)
{
    Array dst;
    dst.resize(src.size());
    if (!src.empty()) std::copy(src.begin(), src.end(), dst.begin());
    return dst;
}
struct FrozenCopy {
    VtFloatArray px, py, pz, width, hairT;
    VtArray<uint64_t> curveId;
    uint32_t totalCurves = 0, totalCvs = 0;
};
static FrozenCopy SnapshotOf(UsdGenCurveBuffer const &b)
{
    FrozenCopy s;
    s.px = PrivateCopy(b.px); s.py = PrivateCopy(b.py); s.pz = PrivateCopy(b.pz);
    s.width = PrivateCopy(b.width); s.hairT = PrivateCopy(b.hairT);
    s.curveId = PrivateCopy(b.curveId);
    s.totalCurves = b.totalCurves; s.totalCvs = b.totalCvs;
    return s;
}
static bool SameAs(FrozenCopy const &s, UsdGenCurveBuffer const &b)
{
    return s.totalCurves == b.totalCurves && s.totalCvs == b.totalCvs &&
        s.px == b.px && s.py == b.py && s.pz == b.pz && s.width == b.width &&
        s.hairT == b.hairT && s.curveId == b.curveId;
}

// --- (1) registry contract ---------------------------------------------

static void CheckRegistryContracts()
{
    UsdGenOpRegistry const &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenFreeze");
    UsdGenOpId const id = registry.Find(type);
    Check(id != UsdGenOpId::Invalid && registry.Record(id).Def().create,
          "Find(UsdGenFreeze) has a kernel");
    if (id == UsdGenOpId::Invalid) return;
    UsdGenOpRecord const &record = registry.Record(id);
    std::unique_ptr<UsdGenOp> op = registry.Create(id);
    Check(op != nullptr, "Create(UsdGenFreeze)");
    if (!op) return;
    Check(record.Name() == type, "Freeze reports its own type");
    Check(record.Def().topoFx == UsdGenTopoFx::CurveCount,
          "Freeze has TopologyEffect CurveCount (static worst case)");
    Check(record.Def().role == UsdGenRole::Curves, "Freeze is a Curves-role chain node");
    Check(record.Def().geometryInputArity == 1, "Freeze consumes one chain input");
    Check(!record.Has(kUsdGenOpGenerator), "Freeze is not a generator");
    TfSpan<const TfToken> const refs = record.ReferenceInputs();
    Check(refs.size() == 1 && refs[0] == TfToken("frozen:curves"),
          "Freeze declares the frozen:curves reference input");
    Check(record.OutputPlanes().empty(), "Freeze emits no output primvars");
    Check(record.InputPlanes().empty(), "Freeze reads no input primvars");
    Check(op->PlanesTouched() == (UsdGenOp::kPlanePoints | UsdGenOp::kPlaneWidths),
          "Freeze touches the points and widths planes only");
    std::set<std::string> topo, value;
    for (UsdGenParamSlot s = 0; s < record.ParamCount(); ++s)
        (record.ParamClass(s) == UsdGenParamClass::Value ? value : topo)
            .insert(record.ParamName(s).GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed", "enabled",
                                        "frozen:curves", "frozen:mode",
                                        "frozen:epoch"},
          "Freeze topology parameters are input/surface/seed/enabled/curves/mode/epoch");
    Check(value == std::set<std::string>{"frozen:tier"},
          "Freeze value parameters are the advisory tier only");
    std::unique_ptr<UsdGenCapture> cap = op->CreateCapture();
    Check(cap != nullptr, "Freeze creates a capture payload");
}

// --- (2) happy-path numerics --------------------------------------------

static UsdGenGraphDesc FreezeDesc(std::vector<UsdGenParamValue> const &params,
                                  bool withSnapshot, int snapshotCvs = 6,
                                  bool snapshotWidths = true)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/freeze");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        StraightStrands(SdfPath("/freeze/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/freeze/source"), SdfPath("/freeze/hair")));
    UsdGenNodeDesc freeze =
        OpNode("/freeze/op", "UsdGenFreeze", SdfPath("/freeze/source"), 7);
    freeze.params = params;
    if (withSnapshot) {
        std::vector<GfVec3f> const snapRoots = {GfVec3f(100, 0, 0),
                                                GfVec3f(103, 0, 0)};
        desc.curveSets.push_back(StraightStrands(
            SdfPath("/freeze/snap"), snapRoots, snapshotCvs, 0.5f,
            UsdGenRole::Reference, 0.5f, 7, snapshotWidths));
        freeze.references = {SdfPath("/freeze/snap")};
    }
    desc.nodes.push_back(freeze);
    desc.terminal = SdfPath("/freeze/op");
    return desc;
}

static void CheckFrozenChainInput()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "frozen chain-input snapshot cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/freeze/source"), &plain),
          "freeze fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "frozen snapshot preserves the strand topology");
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "frozen chain-input points match upstream bit-for-bit");
    Check(out.width == plain.width,
          "frozen chain-input widths match upstream bit-for-bit");
    Check(out.hairT == plain.hairT,
          "frozen chain-input hairT matches upstream bit-for-bit");
    Check(out.curveId == plain.curveId,
          "frozen chain-input curve ids match upstream bit-for-bit");
}

static void CheckSchemaDefaults()
{
    // No params at all: mode defaults to frozen, epoch to "", tier to session.
    UsdGenGraphDesc desc = FreezeDesc({}, false);
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "freeze with schema defaults cooks");
    Check(CookSourceOnly(desc, SdfPath("/freeze/source"), &plain),
          "defaults fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "schema defaults freeze the chain input");
}

static void CheckFrozenExplicitSnapshot()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false},
         {TfToken("frozen:epoch"), VtValue(std::string("usdgen1:sha1:abc")), false}},
        true);
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "frozen explicit snapshot cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/freeze/source"), &plain),
          "explicit-snapshot fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool pointsExact = out.totalCurves == 2 && out.totalCvs == 12;
    bool widthsExact = out.width.size() == 12;
    bool hairTExact = out.hairT.size() == 12;
    for (size_t c = 0; c < 2; ++c) {
        float const rootX = c == 0 ? 100.0f : 103.0f;
        for (int i = 0; i < 6; ++i) {
            size_t const o = c * 6 + size_t(i);
            pointsExact = pointsExact && out.px[o] == rootX &&
                out.py[o] == float(i) * 0.5f && out.pz[o] == 0.0f;
            widthsExact = widthsExact && out.width[o] == 0.5f;
            hairTExact = hairTExact &&
                std::fabs(out.hairT[o] - float(i) / 5.0f) <= 1e-6f;
        }
    }
    Check(pointsExact, "frozen output carries the snapshot points, not upstream");
    Check(widthsExact, "frozen output carries the snapshot widths");
    Check(hairTExact, "frozen output carries canonical root->tip hairT");
    Check(out.curveId.size() == 2 && out.curveId[0] == 7 && out.curveId[1] == 8,
          "frozen output carries the snapshot curve ids");
    Check(!(out.px == plain.px), "frozen output differs from live upstream values");
}

static void CheckSnapshotWithoutWidths()
{
    // A snapshot with no width plane materializes the description default.
    UsdGenGraphDesc desc =
        FreezeDesc({{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}},
                   true, 6, false);
    UsdGenCurveBuffer out;
    Check(Cook(desc, &out), "widthless snapshot cooks");
    if (out.totalCvs == 0) return;
    bool filled = out.width.size() == 12;
    for (size_t o = 0; o < out.width.size(); ++o)
        filled = filled && out.width[o] == 0.01f;
    Check(filled, "widthless snapshot fills usdGen:width:default");
}

static void CheckLivePassthrough()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("live")), false}}, false);
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "live freeze cooks on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/freeze/source"), &plain),
          "live fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
          "live freeze passes points through bit-for-bit");
    Check(out.width == plain.width && out.hairT == plain.hairT,
          "live freeze passes widths and hairT through bit-for-bit");

    // Live with an explicit snapshot bound: the snapshot is retained but
    // stale per 02 §2.9 and must not leak into the output.
    UsdGenGraphDesc stale = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("live")), false}}, true);
    UsdGenCurveBuffer live;
    Check(Cook(stale, &live), "live freeze with a bound snapshot cooks");
    if (live.totalCvs == 0) return;
    Check(live.px == plain.px && live.py == plain.py && live.pz == plain.pz,
          "live freeze ignores the stale bound snapshot");
}

// --- (3) immunity, aliasing, determinism ---------------------------------

static void CheckImmunitySameGraph()
{
    // Frozen chain-input snapshot: mutate the upstream node's committed
    // buffer (values + both versions, through const-cast exactly as the
    // scheduler's SweepChunk writes), dirty only the freeze node, rerun.
    // A recapture would copy the mutated upstream; immunity keeps run 1.
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "immunity fixture compiles");
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run1 = scheduler.Run(graph, ctx, 1);
    Check(!run1.diagnostics.HasErrors(), "immunity fixture first run cooks");
    if (run1.diagnostics.HasErrors()) return;
    FrozenCopy const first = SnapshotOf(graph.Output());

    UsdGenNodeId const sourceId = graph.NodeIdForPath(SdfPath("/freeze/source"));
    UsdGenNodeId const freezeId = graph.NodeIdForPath(SdfPath("/freeze/op"));
    Check(sourceId != UsdGenGraph::InvalidNode &&
              freezeId != UsdGenGraph::InvalidNode,
          "immunity fixture node ids resolve");
    if (sourceId == UsdGenGraph::InvalidNode ||
        freezeId == UsdGenGraph::InvalidNode)
        return;
    UsdGenCompiledNode &source = graph.Node(sourceId);
    Check(!source.buffer.rest.empty(),
          "immunity fixture source carries rest");
    // Private value copies: prep rebinds rest handles, so a handle copy of
    // run-1 output would alias the live array and pass vacuously.
    VtVec3fArray const run1UpstreamRest = PrivateCopy(source.buffer.rest);
    VtVec3fArray const firstRest = PrivateCopy(graph.Output().rest);
    Check(firstRest == run1UpstreamRest,
          "frozen output carries the snapshot rest, not a live alias");
    float *mutate = const_cast<float *>(source.buffer.px.cdata());
    for (size_t o = 0; o < source.buffer.px.size(); ++o) mutate[o] += 10.0f;
    // Rest mutates in place; a brand-new extra plane appears upstream. Both
    // must leave the frozen output untouched (captureAuthorsRest /
    // captureOwnsTransformedPlanes keep the snapshot's planes).
    GfVec3f *rr = const_cast<GfVec3f *>(source.buffer.rest.cdata());
    for (size_t o = 0; o < source.buffer.rest.size(); ++o)
        rr[o] += GfVec3f(1.0f, 2.0f, 3.0f);
    UsdGenPlane late;
    late.name = TfToken("latePlane");
    late.interpolation = TfToken("vertex");
    late.type = TfToken("float");
    late.arity = 1;
    late.f.assign(source.buffer.px.size(), 0.5f);
    source.buffer.extraCv.push_back(late);
    source.buffer.topologyVersion += 100;
    source.buffer.valueVersion += 100;
    graph.MarkNode(freezeId, UsdGenDirtyParameter);
    UsdGenRunResult run2 = scheduler.Run(graph, ctx, 2);
    Check(!run2.diagnostics.HasErrors(), "immunity fixture second run cooks");
    if (run2.diagnostics.HasErrors()) return;
    Check(SameAs(first, graph.Output()),
          "frozen output is bit-identical after an upstream edit (no recapture)");
    bool upstreamStillMutated = !source.buffer.px.empty();
    for (size_t o = 0; o < source.buffer.px.size(); ++o)
        upstreamStillMutated = upstreamStillMutated &&
            source.buffer.px[o] == first.px[o] + 10.0f;
    Check(upstreamStillMutated,
          "the upstream mutation persisted, so the freeze really ignored it");
    Check(graph.Output().rest == firstRest,
          "frozen rest is bit-identical after the upstream rest edit");
    Check(graph.Output().extraCv.empty() && graph.Output().extraCurve.empty(),
          "frozen output gains no upstream extra planes after the snapshot");
}

static UsdGenEpoch DigestFor(UsdGenFreezeOp &op, UsdGenGraphDesc const &desc,
                             size_t nodeIndex, uint32_t seed,
                             uint64_t upstreamGeneration)
{
    UsdGenStackParams stack(desc, desc.nodes[nodeIndex], FreezeRecord());
    UsdGenParamView const &view = stack.View();
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &view;
    ctx.seed = seed;
    ctx.upstreamGeneration = upstreamGeneration;
    return op.CaptureDigest(ctx);
}

static void CheckDigestImmunity()
{
    // Kernel-level proof: the capture identity ignores the upstream
    // generation (and the cosmetic tier) but moves on mode/epoch/seed edits.
    UsdGenFreezeOp op(FreezeRecord());
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false},
         {TfToken("frozen:epoch"), VtValue(std::string("e1")), false}},
        false);
    UsdGenEpoch const base = DigestFor(op, desc, 1, 7, 1);
    Check(DigestFor(op, desc, 1, 7, 999) == base,
          "capture digest ignores the upstream generation (immune)");
    UsdGenGraphDesc live = desc;
    live.nodes[1].params[0].value = VtValue(TfToken("live"));
    Check(DigestFor(op, live, 1, 7, 1) != base,
          "capture digest moves on a mode edit (recapture)");
    UsdGenGraphDesc epoch2 = desc;
    epoch2.nodes[1].params[1].value = VtValue(std::string("e2"));
    Check(DigestFor(op, epoch2, 1, 7, 1) != base,
          "capture digest moves on an epoch edit (recapture)");
    Check(DigestFor(op, desc, 1, 8, 1) != base,
          "capture digest moves on a seed edit");
    UsdGenGraphDesc tiered = desc;
    tiered.nodes[1].params.push_back(
        {TfToken("frozen:tier"), VtValue(TfToken("sublayer")), false});
    Check(DigestFor(op, tiered, 1, 7, 1) == base,
          "capture digest ignores the advisory tier (no recapture)");

    // ValidForTopology never invalidates on upstream revisions.
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    UsdGenStackParams stack(desc, desc.nodes[1], FreezeRecord());
    UsdGenParamView const &view = stack.View();
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &view;
    UsdGenCurveBuffer upstream;
    upstream.totalCurves = 2;
    upstream.totalCvs = 12;
    upstream.px.assign(12, 0.0f);
    upstream.py.assign(12, 0.0f);
    upstream.pz.assign(12, 0.0f);
    upstream.topologyVersion = 3;
    upstream.valueVersion = 4;
    UsdGenDiagnostics diag;
    Check(op.Capture(ctx, UsdGenBufferInputs(upstream), cap.get(), &diag) && !diag.HasErrors(),
          "kernel-level frozen capture succeeds");
    upstream.topologyVersion = 300;
    upstream.valueVersion = 400;
    upstream.totalCurves = 5;
    Check(cap->ValidForTopology(
              static_cast<UsdGenOpInputs const &>(UsdGenBufferInputs(upstream)).Primary()),
          "ValidForTopology ignores upstream revisions (immune)");
}

static void CheckAliasingKernel()
{
    // The snapshot is a private deep copy: mutating the source buffer after
    // Capture (through const-cast, the scheduler's write discipline) must
    // not move the captured planes.
    UsdGenFreezeOp op(FreezeRecord());
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    UsdGenStackParams stack(desc, desc.nodes[1], FreezeRecord());
    UsdGenParamView const &view = stack.View();
    UsdGenCaptureContext ctx;
    ctx.desc = &desc;
    ctx.params = &view;
    UsdGenCurveBuffer upstream;
    upstream.totalCurves = 2;
    upstream.totalCvs = 6;
    upstream.px = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    upstream.py = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f};
    upstream.pz.assign(6, 0.25f);
    upstream.width.assign(6, 0.2f);
    upstream.rest.assign(6, GfVec3f(0.1f, 0.2f, 0.3f));
    UsdGenPlane guideWeight;
    guideWeight.name = TfToken("guideWeight");
    guideWeight.interpolation = TfToken("vertex");
    guideWeight.type = TfToken("float");
    guideWeight.arity = 1;
    guideWeight.f = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    upstream.extraCv.push_back(guideWeight);
    UsdGenPlane clumpId;
    clumpId.name = TfToken("clumpId_0");
    clumpId.interpolation = TfToken("uniform");
    clumpId.type = TfToken("int");
    clumpId.arity = 1;
    clumpId.i = {3, 7};
    upstream.extraCurve.push_back(clumpId);
    upstream.topologyVersion = 11;
    upstream.valueVersion = 12;
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    UsdGenDiagnostics diag;
    Check(op.Capture(ctx, UsdGenBufferInputs(upstream), cap.get(), &diag) && !diag.HasErrors(),
          "aliasing fixture capture succeeds");
    if (diag.HasErrors()) return;
    Check((cap->OwnedBuffer() != nullptr), "frozen capture owns its buffer");
    FrozenCopy const snap = SnapshotOf(UsdGenCaptureBuffer(*cap));
    VtVec3fArray const restSnap = PrivateCopy(UsdGenCaptureBuffer(*cap).rest);
    VtFloatArray const cvSnap = UsdGenCaptureBuffer(*cap).extraCv.empty()
        ? VtFloatArray() : PrivateCopy(UsdGenCaptureBuffer(*cap).extraCv[0].f);
    VtIntArray const curveSnap = UsdGenCaptureBuffer(*cap).extraCurve.empty()
        ? VtIntArray() : PrivateCopy(UsdGenCaptureBuffer(*cap).extraCurve[0].i);
    float *wx = const_cast<float *>(upstream.px.cdata());
    float *wy = const_cast<float *>(upstream.py.cdata());
    float *wz = const_cast<float *>(upstream.pz.cdata());
    float *ww = const_cast<float *>(upstream.width.cdata());
    for (size_t o = 0; o < 6; ++o) {
        wx[o] = -wx[o];
        wy[o] = -wy[o];
        wz[o] = -wz[o];
        ww[o] = -ww[o];
    }
    GfVec3f *rx = const_cast<GfVec3f *>(upstream.rest.cdata());
    float *gx = const_cast<float *>(upstream.extraCv[0].f.cdata());
    for (size_t o = 0; o < 6; ++o) {
        rx[o] = -rx[o];
        gx[o] = -gx[o];
    }
    int *cx = const_cast<int *>(upstream.extraCurve[0].i.cdata());
    cx[0] = -cx[0];
    cx[1] = -cx[1];
    Check(SameAs(snap, UsdGenCaptureBuffer(*cap)),
          "captured snapshot is a deep copy (source mutation does not alias)");
    Check(UsdGenCaptureBuffer(*cap).rest == restSnap,
          "captured rest is a deep copy (source rest mutation does not alias)");
    Check(UsdGenCaptureBuffer(*cap).extraCv.size() == 1 &&
              UsdGenCaptureBuffer(*cap).extraCv[0].f == cvSnap,
          "captured per-CV planes are deep copies");
    Check(UsdGenCaptureBuffer(*cap).extraCurve.size() == 1 &&
              UsdGenCaptureBuffer(*cap).extraCurve[0].i == curveSnap,
          "captured per-curve planes are deep copies");

    // Live captures own nothing.
    UsdGenGraphDesc liveDesc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("live")), false}}, false);
    UsdGenStackParams liveStack(liveDesc, liveDesc.nodes[1], FreezeRecord());
    UsdGenParamView const &liveView = liveStack.View();
    ctx.desc = &liveDesc;
    ctx.params = &liveView;
    std::unique_ptr<UsdGenCapture> liveCap = op.CreateCapture();
    UsdGenDiagnostics liveDiag;
    Check(op.Capture(ctx, UsdGenBufferInputs(upstream), liveCap.get(), &liveDiag) &&
              !liveDiag.HasErrors(),
          "live capture succeeds");
    Check(!(liveCap->OwnedBuffer() != nullptr), "live capture owns no buffer");
}

static void CheckDeterminism()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false},
         {TfToken("frozen:epoch"), VtValue(std::string("e9")), false}},
        true);
    UsdGenCurveBuffer first, second;
    Check(Cook(desc, &first), "determinism first cook runs");
    Check(Cook(desc, &second), "determinism second cook runs");
    if (first.totalCvs == 0 || second.totalCvs == 0) return;
    FrozenCopy const snap = SnapshotOf(first);
    Check(SameAs(snap, second), "two fresh cooks are bit-identical");

    UsdGenCompiler compiler;
    UsdGenGraph graph;
    Check(compiler.Compile(desc, &graph).ok, "determinism fixture compiles");
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run1 = scheduler.Run(graph, ctx, 1);
    FrozenCopy const retained = SnapshotOf(graph.Output());
    UsdGenRunResult run2 = scheduler.Run(graph, ctx, 2);
    Check(!run1.diagnostics.HasErrors() && !run2.diagnostics.HasErrors(),
          "determinism retained runs cook");
    Check(SameAs(retained, graph.Output()),
          "two runs on one graph are bit-identical");
}

// --- (4) fail-closed edges ------------------------------------------------

static void CheckBadModeFailsClosed()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("thaw")), false}}, false);
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "unknown frozen:mode fails closed");
    Check(AnyContains(errors, "frozen:mode"),
          "unknown frozen:mode names the parameter");
}

static void CheckBadTierFailsClosed()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:tier"), VtValue(TfToken("cloud")), false}}, false);
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "unknown frozen:tier fails closed");
    Check(AnyContains(errors, "frozen:tier"),
          "unknown frozen:tier names the parameter");
}

static void CheckBadEpochFailsClosed()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:epoch"), VtValue(5), false}}, false);
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "non-string frozen:epoch fails closed");
    Check(AnyContains(errors, "frozen:epoch"),
          "non-string frozen:epoch names the parameter");

    // A token epoch is tolerated exactly as GetToken tolerates strings.
    UsdGenGraphDesc tokenEpoch = FreezeDesc(
        {{TfToken("frozen:epoch"), VtValue(TfToken("e1")), false}}, false);
    Check(Cook(tokenEpoch, &out), "token-spelled epoch cooks");
}

static void CheckSnapshotMismatchFailsClosed()
{
    // 2x4 snapshot against a 2x6 chain: the session tier cannot install it.
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, true, 4);
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "layout-mismatched snapshot fails closed");
    Check(AnyContains(errors, "must match the chain input layout"),
          "layout mismatch names the session-tier rule");
}

static void CheckEmptySnapshotFailsClosed()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    UsdGenCurveSetDesc empty;
    empty.path = SdfPath("/freeze/empty");
    empty.role = UsdGenRole::Reference;
    empty.curveRole = TfToken("guide");
    desc.curveSets.push_back(empty);
    desc.nodes.back().references = {SdfPath("/freeze/empty")};
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "empty snapshot fails closed");
    Check(AnyContains(errors, "has no curves"), "empty snapshot is named");
}

static void CheckMissingInputFailsClosed()
{
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    desc.nodes.back().inputs.clear();
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult r = compiler.Compile(desc, &graph);
    bool const specific = AnyContains(r.errors, "requires exactly 1 geometry input");
    Check(!r.ok && specific, "freeze with no chain input fails closed at compile");
    if (r.ok || !specific)
        for (auto const &e : r.errors) std::printf("  error: %s\n", e.c_str());
}

static void CheckExpressionRejected()
{
    // Freeze owns no connectable parameter: any binding fails at compile via
    // the shared table's default rejection (no per-op row needed).
    UsdGenGraphDesc desc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    AddBinding(desc, SdfPath("/freeze/op"), SdfPath("/freeze/Expressions/ctl"),
               "$value", TfToken("float"), FloatShape(), TfToken("mask"),
               expr::Domain::Point, VtValue(1.0f));
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult r = compiler.Compile(desc, &graph);
    bool const specific =
        AnyContains(r.errors, "does not accept connected (expression) parameters");
    Check(!r.ok && specific, "connected expression on Freeze fails closed");
    if (r.ok || !specific)
        for (auto const &e : r.errors) std::printf("  error: %s\n", e.c_str());
}

static void CheckBadDefaultWidthFailsClosed()
{
    UsdGenGraphDesc desc =
        FreezeDesc({{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}},
                   true, 6, false);
    desc.defaultWidth = std::numeric_limits<float>::quiet_NaN();
    UsdGenCurveBuffer out;
    std::vector<std::string> errors, warnings;
    Check(!CookFull(desc, &out, &errors, &warnings),
          "NaN defaultWidth with a widthless snapshot fails closed");
    // The graph fails in the source node first (it validates defaultWidth
    // unconditionally), so Freeze's own message is pinned at kernel level.
    UsdGenFreezeOp op(FreezeRecord());
    UsdGenGraphDesc kernelDesc = FreezeDesc(
        {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}}, false);
    kernelDesc.defaultWidth = std::numeric_limits<float>::quiet_NaN();
    UsdGenStackParams stack(kernelDesc, kernelDesc.nodes[1], FreezeRecord());
    UsdGenParamView const &view = stack.View();
    UsdGenCaptureContext ctx;
    ctx.desc = &kernelDesc;
    ctx.params = &view;
    UsdGenCurveBuffer widthless;
    widthless.totalCurves = 2;
    widthless.totalCvs = 6;
    widthless.px.assign(6, 0.0f);
    widthless.py.assign(6, 0.0f);
    widthless.pz.assign(6, 0.0f);
    std::unique_ptr<UsdGenCapture> cap = op.CreateCapture();
    UsdGenDiagnostics diag;
    Check(!op.Capture(ctx, UsdGenBufferInputs(widthless), cap.get(), &diag),
          "Freeze rejects a widthless snapshot under NaN defaultWidth");
    Check(AnyContains(diag.errors, "defaultWidth"),
          "NaN defaultWidth names the parameter");
}

static void CheckTierLowering()
{
    // sublayer/payload lower to session LOUDLY: the cook succeeds and carries
    // one warning quoting the requested tier.
    for (char const *tier : {"sublayer", "payload"}) {
        UsdGenGraphDesc desc = FreezeDesc(
            {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false},
             {TfToken("frozen:tier"), VtValue(TfToken(tier)), false}},
            false);
        UsdGenCurveBuffer out, plain;
        std::vector<std::string> errors, warnings;
        Check(CookFull(desc, &out, &errors, &warnings),
              std::string("frozen:tier ") + tier + " cooks (lowered to session)");
        Check(AnyContains(warnings, "lowering to 'session'"),
              std::string("frozen:tier ") + tier + " warns loudly");
        Check(AnyContains(warnings, tier),
              std::string("lowering warning quotes '") + tier + "'");
        Check(CookSourceOnly(desc, SdfPath("/freeze/source"), &plain),
              "lowering fixture source cooks on its own");
        if (out.totalCvs == 0 || plain.totalCvs == 0) continue;
        Check(out.px == plain.px && out.py == plain.py && out.pz == plain.pz,
              std::string("lowered tier '") + tier + "' still freezes");
    }
}

// --- (5) CUDA-backend explicit rejection ----------------------------------
// Mirrors testUsdGenCudaExecutionPlan's unsupported-operator case: the
// capability matrix admits only Scatter/Grow/CurveSource/Width/Length/Noise/
// Deform/ReferenceSource/WidthBlend, so a Freeze graph is rejected with the
// exact planner diagnostic (never a silent CPU fallback).

static UsdGenGraphDesc CudaFreezeDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Hair");
    desc.executionBackend = UsdGenExecutionBackend::Cuda;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3, 3, 3};
    surface.faceVertexIndices = {0, 1, 2, 0, 1, 3, 1, 2, 4};
    desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{.2f, .2f, 0}, {.2f, .4f, 0}};
    curves.rest = curves.points;
    curves.curveId = {42};
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f, .2f}};
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Hair/Ops/ZSource");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.surfaces = {surface.path};
    UsdGenNodeDesc freeze;
    freeze.path = SdfPath("/Groom/Hair/Ops/AFreeze");
    freeze.type = TfToken("UsdGenFreeze");
    freeze.inputs = {source.path};
    freeze.params = {{TfToken("frozen:mode"), VtValue(TfToken("frozen")), false}};
    desc.nodes = {source, freeze};
    desc.terminal = freeze.path;
    return desc;
}

static void CheckCudaRejection()
{
    UsdGenExecutionCapabilityMatrix const &matrix =
        GetCudaExecutionCapabilityMatrix();
    Check(matrix.Find(TfToken("UsdGenFreeze")) == nullptr,
          "CUDA capability matrix has no UsdGenFreeze row");
    UsdGenGraphDesc desc = CudaFreezeDesc();
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(desc, &diagnostics);
    Check(!plan, "CUDA planner refuses a Freeze graph");
    if (matrix.Available()) {
        bool const exact = diagnostics.errors.size() == 1 &&
            diagnostics.warnings.empty() &&
            diagnostics.errors.front() ==
                "CUDA: unsupported operator: CUDA capability matrix has no "
                "implementation for UsdGenFreeze";
        Check(exact, "CUDA refusal names the missing Freeze implementation");
        if (!exact)
            for (auto const &e : diagnostics.errors)
                std::printf("  cuda: %s\n", e.c_str());
    } else {
        bool const exact = diagnostics.errors.size() == 1 &&
            diagnostics.errors.front() == "CUDA: backend is not built";
        Check(exact, "CUDA-disabled build refuses explicitly (not built)");
    }
}

int main()
{
    CheckRegistryContracts();
    CheckFrozenChainInput();
    CheckSchemaDefaults();
    CheckFrozenExplicitSnapshot();
    CheckSnapshotWithoutWidths();
    CheckLivePassthrough();
    CheckImmunitySameGraph();
    CheckDigestImmunity();
    CheckAliasingKernel();
    CheckDeterminism();
    CheckBadModeFailsClosed();
    CheckBadTierFailsClosed();
    CheckBadEpochFailsClosed();
    CheckSnapshotMismatchFailsClosed();
    CheckEmptySnapshotFailsClosed();
    CheckMissingInputFailsClosed();
    CheckExpressionRejected();
    CheckBadDefaultWidthFailsClosed();
    CheckTierLowering();
    CheckCudaRejection();
    std::printf("testUsdGenFreeze: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
