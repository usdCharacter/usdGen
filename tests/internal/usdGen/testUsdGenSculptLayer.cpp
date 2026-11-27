// UsdGenSculptLayer operator contract and CPU cooks.
//
// Covers hand-authored per-CV deltas keyed by stable curveId:
//   (1) registry HasKernel/Create/TopologyEffect/ReferenceInputs/
//       OutputPrimvars plus the Topology/Value parameter partition;
//   (2) deterministic CPU cooks over small hand-built topologies (fixed
//       seeds): the root-frame and object-space formulae (a rotated rest
//       frame tells the spaces apart), the weight/mask envelope, id-keyed
//       matching (unknown ids ignored, unmatched strands bit-for-bit),
//       locks/rebase/epoch metadata, and the default empty layer;
//   (3) every malformed layer fails closed;
//   (4) run-twice determinism;
//   (5) the CUDA backend explicitly rejects a sculpt graph.
#include "utils/pxrInterop.h"
#include "usdGen/compile/compiler.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "hosts/usdGenUsd/usdGenOpFixture.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/compile/graph.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/exec/scheduler.h"

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

constexpr float kTol = 1e-4f;
static bool Near(float a, float b, float tol = kTol)
{
    return std::fabs(a - b) <= tol;
}

// Straight strands along +Y from the given roots. Mirrors the CurveSource
// fixture (surface-free, rebind never): valid skin bindings plus authored
// root frames. Strand 0 carries the identity frame
// (T=(1,0,0), B=(0,1,0), N=(0,0,1)); strand 1 a +90-degree Z rotation
// (T=(0,1,0), B=(-1,0,0), N=(0,0,1)), which is exactly orthonormal so the
// authored-frame contract accepts it verbatim.
static UsdGenCurveSetDesc SculptStrands(SdfPath const &path,
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
    uint64_t const ids[] = {41, 88};
    for (size_t c = 0; c < n; ++c) set.curveId[c] = ids[c];
    set.skinPrim.resize(n);
    for (size_t c = 0; c < n; ++c) set.skinPrim[c] = int(c);
    set.skinPrimUv.assign(n, GfVec2f(0.2f, 0.3f));
    set.rootFrame.assign(n, GfMatrix4d(1.0));
    if (n > 1) {
        set.rootFrame[1].SetRow3(0, GfVec3d(0, 1, 0));
        set.rootFrame[1].SetRow3(1, GfVec3d(-1, 0, 0));
        set.rootFrame[1].SetRow3(2, GfVec3d(0, 0, 1));
    }
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
    TfToken const type("UsdGenSculptLayer");
    Check(registry.Find(type) != UsdGenOpId::Invalid, "HasKernel(UsdGenSculptLayer)");
    std::unique_ptr<UsdGenOp> op = registry.Create(registry.Find(type));
    Check(op != nullptr, "Create(UsdGenSculptLayer)");
    if (!op) return;
    Check(registry.Record(registry.Find(type)).Name() == type, "SculptLayer reports its own type");
    Check(op->Record().Def().countChanges == 0,
          "SculptLayer has TopologyEffect None");
    Check(op->Record().ReferenceInputs().empty(),
          "SculptLayer declares no reference inputs");
    Check(usdGenTest::OutputPlaneNames(op->Record()).empty(), "SculptLayer emits no output primvars");
    Check(op->PlanesTouched() == UsdGenOp::kPlanePoints,
          "SculptLayer touches the points plane only");
    std::set<std::string> topo, value;
    for (TfToken const &t : usdGenTest::RoutedParams(op->Record(), false)) topo.insert(t.GetString());
    for (TfToken const &t : usdGenTest::RoutedParams(op->Record(), true)) value.insert(t.GetString());
    Check(topo == std::set<std::string>{"input", "surface", "seed",
                                        "sculpt:space", "sculpt:curveIds",
                                        "sculpt:cvOffsets", "sculpt:epoch",
                                        "sculpt:lockedCurves", "sculpt:rootPrims",
                                        "sculpt:rootUVs"},
          "SculptLayer topology parameters are input/surface/seed/space/ids/offsets/epoch/locks/rebase");
    Check(value == std::set<std::string>{"enabled", "mask", "sculpt:weight",
                                         "sculpt:deltas"},
          "SculptLayer value parameters are enabled/mask/weight/deltas");
}

// --- (2) CPU cooks -----------------------------------------------------

static UsdGenGraphDesc SculptDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/sculpt");
    desc.defaultWidth = 0.01f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(3, 0, 0)};
    desc.curveSets.push_back(
        SculptStrands(SdfPath("/sculpt/hair"), roots, 6, 0.25f));
    desc.nodes.push_back(
        SourceNode(SdfPath("/sculpt/source"), SdfPath("/sculpt/hair")));
    UsdGenNodeDesc sculpt = OpNode("/sculpt/op", "UsdGenSculptLayer",
                                   SdfPath("/sculpt/source"), 7);
    sculpt.params = params;
    desc.nodes.push_back(sculpt);
    desc.terminal = SdfPath("/sculpt/op");
    return desc;
}

static std::vector<UsdGenParamValue> LayerParams(
    std::vector<uint64_t> const &ids, std::vector<int> const &offsets,
    std::vector<GfVec3f> const &deltas)
{
    VtArray<uint64_t> idArray(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) idArray[i] = ids[i];
    VtIntArray offsetArray(offsets.size());
    for (size_t i = 0; i < offsets.size(); ++i) offsetArray[i] = offsets[i];
    VtVec3fArray deltaArray(deltas.size());
    for (size_t i = 0; i < deltas.size(); ++i) deltaArray[i] = deltas[i];
    return {{TfToken("sculpt:curveIds"), VtValue(idArray), false},
            {TfToken("sculpt:cvOffsets"), VtValue(offsetArray), false},
            {TfToken("sculpt:deltas"), VtValue(deltaArray), false}};
}

static void CheckRootFrameCook()
{
    // Strand 41 (identity frame): delta (0.1*j, 0.2, -0.3) lands as-is.
    // Strand 88 (rotated frame): delta (1, 0, 0) lands along T = (0, 1, 0).
    std::vector<GfVec3f> deltas;
    for (int j = 0; j < 6; ++j)
        deltas.push_back(GfVec3f(0.1f * float(j), 0.2f, -0.3f));
    for (int j = 0; j < 6; ++j) deltas.push_back(GfVec3f(1.0f, 0.0f, 0.0f));
    UsdGenGraphDesc desc =
        SculptDesc(LayerParams({41, 88}, {0, 6, 12}, deltas));
    UsdGenCurveBuffer out, plain;
    Check(Cook(desc, &out), "SculptLayer root-frame cook runs on the CPU lane");
    Check(CookSourceOnly(desc, SdfPath("/sculpt/source"), &plain),
          "SculptLayer fixture source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    Check(out.totalCurves == 2 && out.totalCvs == 12,
          "SculptLayer preserves the strand topology");
    bool identity = true, rotated = true;
    for (int j = 0; j < 6; ++j) {
        size_t const o = size_t(j);
        identity = identity && Near(out.px[o], plain.px[o] + 0.1f * float(j), 1e-6f) &&
            Near(out.py[o], plain.py[o] + 0.2f, 1e-6f) &&
            Near(out.pz[o], plain.pz[o] - 0.3f, 1e-6f);
        size_t const r = 6 + size_t(j);
        rotated = rotated && Near(out.px[r], plain.px[r], 1e-6f) &&
            Near(out.py[r], plain.py[r] + 1.0f, 1e-6f) &&
            Near(out.pz[r], plain.pz[r], 1e-6f);
    }
    Check(identity, "SculptLayer adds deltas through the identity root frame");
    Check(rotated, "SculptLayer maps deltas through a rotated root frame");

    // Half weight halves every offset on both strands.
    std::vector<UsdGenParamValue> half = LayerParams({41, 88}, {0, 6, 12}, deltas);
    half.push_back({TfToken("sculpt:weight"), VtValue(0.5f), false});
    UsdGenCurveBuffer halved;
    Check(Cook(SculptDesc(half), &halved), "SculptLayer weight cook runs");
    if (halved.totalCvs == 0) return;
    bool halvedOk = true;
    for (int j = 0; j < 6; ++j) {
        size_t const o = size_t(j), r = 6 + size_t(j);
        halvedOk = halvedOk &&
            Near(halved.px[o], plain.px[o] + 0.05f * float(j)) &&
            Near(halved.py[o], plain.py[o] + 0.1f) &&
            Near(halved.pz[o], plain.pz[o] - 0.15f) &&
            Near(halved.py[r], plain.py[r] + 0.5f);
    }
    Check(halvedOk, "SculptLayer scales offsets by the layer weight");

    // The mask envelopes the layer: half mask halves the push, zero mask is
    // a bit-for-bit pass-through.
    std::vector<UsdGenParamValue> masked = LayerParams({41, 88}, {0, 6, 12}, deltas);
    masked.push_back({TfToken("mask"), VtValue(0.5f), false});
    UsdGenCurveBuffer soft;
    Check(Cook(SculptDesc(masked), &soft), "SculptLayer mask cook runs");
    if (soft.totalCvs == 0) return;
    Check(soft.px == halved.px && soft.py == halved.py && soft.pz == halved.pz,
          "SculptLayer scales offsets by the mask exactly like the weight");
    std::vector<UsdGenParamValue> muted = LayerParams({41, 88}, {0, 6, 12}, deltas);
    muted.push_back({TfToken("mask"), VtValue(0.0f), false});
    UsdGenCurveBuffer off;
    Check(Cook(SculptDesc(muted), &off), "SculptLayer zero-mask cook runs");
    if (off.totalCvs == 0) return;
    Check(off.px == plain.px && off.py == plain.py && off.pz == plain.pz,
          "SculptLayer with mask 0 passes points through bit-for-bit");
}

static void CheckObjectSpaceCook()
{
    // Object space ignores the rest frame: delta (1, 2, 3) on the rotated
    // strand lands as (1, 2, 3).
    std::vector<GfVec3f> deltas(6, GfVec3f(1.0f, 2.0f, 3.0f));
    std::vector<UsdGenParamValue> params = LayerParams({88}, {0, 6}, deltas);
    params.push_back({TfToken("sculpt:space"), VtValue(TfToken("object")), false});
    UsdGenCurveBuffer out, plain;
    Check(Cook(SculptDesc(params), &out), "SculptLayer object-space cook runs");
    UsdGenGraphDesc desc = SculptDesc(params);
    Check(CookSourceOnly(desc, SdfPath("/sculpt/source"), &plain),
          "SculptLayer object-space source cooks on its own");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool moved = true, untouched = true;
    for (int j = 0; j < 6; ++j) {
        size_t const o = 6 + size_t(j);
        moved = moved && Near(out.px[o], plain.px[o] + 1.0f, 1e-6f) &&
            Near(out.py[o], plain.py[o] + 2.0f, 1e-6f) &&
            Near(out.pz[o], plain.pz[o] + 3.0f, 1e-6f);
        size_t const u = size_t(j);
        untouched = untouched && out.px[u] == plain.px[u] &&
            out.py[u] == plain.py[u] && out.pz[u] == plain.pz[u];
    }
    Check(moved, "SculptLayer applies object-space deltas as-is");
    Check(untouched, "SculptLayer leaves strands without an entry bit-for-bit");
}

static void CheckMatchingAndMetadata()
{
    std::vector<GfVec3f> deltas(6, GfVec3f(0.0f, 0.0f, 0.5f));
    UsdGenCurveBuffer plain;
    {
        UsdGenGraphDesc desc = SculptDesc(LayerParams({41}, {0, 6}, deltas));
        Check(CookSourceOnly(desc, SdfPath("/sculpt/source"), &plain),
              "SculptLayer matching source cooks on its own");
    }
    // Unknown ids are kept and ignored: the extra span cooks and changes
    // nothing.
    std::vector<GfVec3f> padded = deltas;
    for (int j = 0; j < 6; ++j) padded.push_back(GfVec3f(9.0f, 9.0f, 9.0f));
    UsdGenGraphDesc extra = SculptDesc(LayerParams({41, 999}, {0, 6, 12}, padded));
    UsdGenCurveBuffer out;
    Check(Cook(extra, &out), "SculptLayer ignores entries for unknown ids");
    if (out.totalCvs == 0 || plain.totalCvs == 0) return;
    bool onlyMatched = true;
    for (int j = 0; j < 6; ++j) {
        size_t const o = size_t(j), u = 6 + size_t(j);
        onlyMatched = onlyMatched && Near(out.pz[o], plain.pz[o] + 0.5f, 1e-6f) &&
            out.px[u] == plain.px[u] && out.py[u] == plain.py[u] &&
            out.pz[u] == plain.pz[u];
    }
    Check(onlyMatched, "SculptLayer sculpts only the matched strand");

    // Locks name downstream Freeze-brush curves: this layer still sculpts
    // them, bit-identical to the unlocked cook.
    std::vector<UsdGenParamValue> locked = LayerParams({41}, {0, 6}, deltas);
    VtArray<uint64_t> locks(1);
    locks[0] = 41;
    locked.push_back({TfToken("sculpt:lockedCurves"), VtValue(locks), false});
    UsdGenCurveBuffer frozen;
    Check(Cook(SculptDesc(locked), &frozen), "SculptLayer locked-curves cook runs");
    UsdGenCurveBuffer unlocked;
    Check(Cook(SculptDesc(LayerParams({41}, {0, 6}, deltas)), &unlocked),
          "SculptLayer unlocked cook runs");
    if (frozen.totalCvs == 0 || unlocked.totalCvs == 0) return;
    Check(frozen.px == unlocked.px && frozen.py == unlocked.py &&
              frozen.pz == unlocked.pz,
          "SculptLayer still sculpts its locked curves");

    // Rebase metadata and the epoch are accepted and change nothing.
    std::vector<UsdGenParamValue> meta = LayerParams({41}, {0, 6}, deltas);
    VtIntArray prims(1);
    prims[0] = 0;
    VtVec2fArray uvs(1);
    uvs[0] = GfVec2f(0.2f, 0.3f);
    meta.push_back({TfToken("sculpt:rootPrims"), VtValue(prims), false});
    meta.push_back({TfToken("sculpt:rootUVs"), VtValue(uvs), false});
    meta.push_back({TfToken("sculpt:epoch"),
                    VtValue(std::string("usdgen1:sha1:abc")), false});
    UsdGenCurveBuffer rebased;
    Check(Cook(SculptDesc(meta), &rebased), "SculptLayer rebase/epoch cook runs");
    if (rebased.totalCvs == 0) return;
    Check(rebased.px == unlocked.px && rebased.py == unlocked.py &&
              rebased.pz == unlocked.pz,
          "SculptLayer rebase metadata and epoch change nothing");

    // The schema-default layer (no sculpt params) and the explicit empty
    // layer both pass through bit-for-bit.
    UsdGenCurveBuffer bare, explicitEmpty;
    Check(Cook(SculptDesc({}), &bare), "SculptLayer default layer cooks");
    VtArray<uint64_t> noIds;
    VtIntArray zeroOff(1);
    zeroOff[0] = 0;
    VtVec3fArray noDeltas;
    std::vector<UsdGenParamValue> empty = {
        {TfToken("sculpt:curveIds"), VtValue(noIds), false},
        {TfToken("sculpt:cvOffsets"), VtValue(zeroOff), false},
        {TfToken("sculpt:deltas"), VtValue(noDeltas), false}};
    Check(Cook(SculptDesc(empty), &explicitEmpty),
          "SculptLayer explicit empty layer cooks");
    if (bare.totalCvs == 0 || explicitEmpty.totalCvs == 0) return;
    Check(bare.px == plain.px && bare.py == plain.py && bare.pz == plain.pz,
          "SculptLayer default layer passes points through bit-for-bit");
    Check(explicitEmpty.px == plain.px && explicitEmpty.py == plain.py &&
              explicitEmpty.pz == plain.pz,
          "SculptLayer explicit empty layer passes points through bit-for-bit");

    // Zero weight is a bit-for-bit pass-through.
    std::vector<UsdGenParamValue> zero = LayerParams({41}, {0, 6}, deltas);
    zero.push_back({TfToken("sculpt:weight"), VtValue(0.0f), false});
    UsdGenCurveBuffer muted;
    Check(Cook(SculptDesc(zero), &muted), "SculptLayer zero-weight cook runs");
    if (muted.totalCvs == 0) return;
    Check(muted.px == plain.px && muted.py == plain.py && muted.pz == plain.pz,
          "SculptLayer with weight 0 passes points through bit-for-bit");
}

// --- (3) fail-closed layers --------------------------------------------

static void CheckFailsClosed(char const *name,
                             std::vector<UsdGenParamValue> const &params)
{
    UsdGenCurveBuffer out;
    std::string label = std::string("SculptLayer ") + name + " fails closed";
    Check(!Cook(SculptDesc(params), &out), label.c_str());
}

static void CheckMalformedLayersFailClosed()
{
    std::vector<GfVec3f> twelve;
    for (int j = 0; j < 12; ++j) twelve.push_back(GfVec3f(0, 0, 0.1f));
    std::vector<GfVec3f> six(twelve.begin(), twelve.begin() + 6);

    std::vector<UsdGenParamValue> space = LayerParams({41}, {0, 6}, six);
    space.push_back({TfToken("sculpt:space"), VtValue(TfToken("bogus")), false});
    CheckFailsClosed("with an unknown space", space);

    for (float weight : {2.0f, -0.5f}) {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        params.push_back({TfToken("sculpt:weight"), VtValue(weight), false});
        CheckFailsClosed("with an out-of-range weight", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        params.push_back({TfToken("sculpt:weight"),
                          VtValue(std::numeric_limits<float>::quiet_NaN()), false});
        CheckFailsClosed("with a NaN weight", params);
    }

    CheckFailsClosed("with unsorted curveIds",
                     LayerParams({88, 41}, {0, 6, 12}, twelve));
    CheckFailsClosed("with duplicate curveIds",
                     LayerParams({41, 41}, {0, 6, 12}, twelve));
    CheckFailsClosed("with a short cvOffsets",
                     LayerParams({41, 88}, {0, 12}, twelve));
    CheckFailsClosed("with cvOffsets not starting at 0",
                     LayerParams({41, 88}, {1, 6, 12}, twelve));
    CheckFailsClosed("with decreasing cvOffsets",
                     LayerParams({41, 88}, {0, 12, 6}, twelve));
    CheckFailsClosed("with cvOffsets not ending at deltas.size()",
                     LayerParams({41, 88}, {0, 6, 10}, twelve));
    {
        std::vector<GfVec3f> nan = six;
        nan[2] = GfVec3f(0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f);
        CheckFailsClosed("with a non-finite delta",
                         LayerParams({41}, {0, 6}, nan));
    }
    {
        std::vector<GfVec3f> five(six.begin(), six.begin() + 5);
        CheckFailsClosed("with a span shorter than its strand",
                         LayerParams({41}, {0, 5}, five));
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41, 88}, {0, 6, 12}, twelve);
        VtIntArray prims(1);
        prims[0] = 0;
        params.push_back({TfToken("sculpt:rootPrims"), VtValue(prims), false});
        CheckFailsClosed("with short rootPrims", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        VtIntArray prims(1);
        prims[0] = -1;
        params.push_back({TfToken("sculpt:rootPrims"), VtValue(prims), false});
        CheckFailsClosed("with a negative rootPrim", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41, 88}, {0, 6, 12}, twelve);
        VtVec2fArray uvs(1);
        uvs[0] = GfVec2f(0.2f, 0.3f);
        params.push_back({TfToken("sculpt:rootUVs"), VtValue(uvs), false});
        CheckFailsClosed("with short rootUVs", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        VtVec2fArray uvs(1);
        uvs[0] = GfVec2f(std::numeric_limits<float>::quiet_NaN(), 0.0f);
        params.push_back({TfToken("sculpt:rootUVs"), VtValue(uvs), false});
        CheckFailsClosed("with a non-finite rootUV", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        VtArray<uint64_t> locks(2);
        locks[0] = 41;
        locks[1] = 41;
        params.push_back({TfToken("sculpt:lockedCurves"), VtValue(locks), false});
        CheckFailsClosed("with duplicate lockedCurves", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        params[0] = {TfToken("sculpt:curveIds"), VtValue(VtIntArray(1, 41)), false};
        CheckFailsClosed("with a mistyped curveIds", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        params[2] = {TfToken("sculpt:deltas"), VtValue(VtIntArray(6, 0)), false};
        CheckFailsClosed("with mistyped deltas", params);
    }
    {
        std::vector<UsdGenParamValue> params = LayerParams({41}, {0, 6}, six);
        params.push_back({TfToken("sculpt:epoch"), VtValue(7), false});
        CheckFailsClosed("with a mistyped epoch", params);
    }
    {
        VtVec3fArray orphaned(6);
        for (size_t i = 0; i < 6; ++i) orphaned[i] = GfVec3f(0, 0, 0.1f);
        std::vector<UsdGenParamValue> params = {
            {TfToken("sculpt:deltas"), VtValue(orphaned), false}};
        CheckFailsClosed("with deltas but no curveIds", params);
    }
}

// --- (4) determinism ----------------------------------------------------

static void CheckDeterminism()
{
    std::vector<GfVec3f> deltas;
    for (int j = 0; j < 6; ++j)
        deltas.push_back(GfVec3f(0.1f * float(j), -0.2f, 0.3f));
    for (int j = 0; j < 6; ++j)
        deltas.push_back(GfVec3f(0.0f, 0.4f, -0.1f * float(j)));
    UsdGenGraphDesc desc = SculptDesc(LayerParams({41, 88}, {0, 6, 12}, deltas));
    UsdGenCurveBuffer first, second;
    Check(Cook(desc, &first), "SculptLayer determinism first cook runs");
    Check(Cook(desc, &second), "SculptLayer determinism second cook runs");
    if (first.totalCvs == 0 || second.totalCvs == 0) return;
    Check(first.px == second.px && first.py == second.py && first.pz == second.pz,
          "SculptLayer cooks bit-identical points twice");
}

// --- (5) CUDA-backend explicit rejection --------------------------------
// SculptLayer is CPU-only: the capability matrix admits no row for it, so
// ValidateCudaGraph refuses a sculpt graph exactly as it refuses Clump.

static void CheckCudaRejection()
{
    std::vector<GfVec3f> six(6, GfVec3f(0, 0, 0.1f));
    UsdGenGraphDesc desc = SculptDesc(LayerParams({41}, {0, 6}, six));
    UsdGenDiagnostics errors;
    Check(!ValidateCudaGraph(usdGenUsd::Lower(desc), &errors) && errors.HasErrors(),
          "SculptLayer graph is explicitly rejected by the CUDA backend");
    Check(GetCudaExecutionCapabilityMatrix().Find(TfToken("UsdGenSculptLayer")) == nullptr,
          "SculptLayer has no CUDA capability-matrix row");
}

int main()
{
    CheckRegistryContracts();
    CheckRootFrameCook();
    CheckObjectSpaceCook();
    CheckMatchingAndMetadata();
    CheckMalformedLayersFailClosed();
    CheckDeterminism();
    CheckCudaRejection();
    std::printf("testUsdGenSculptLayer: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
