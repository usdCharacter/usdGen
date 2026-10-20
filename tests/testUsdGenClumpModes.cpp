// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenClumpModes: the reference clump modes of UsdGenClump and the
// Bezier thickness profile of UsdGenWidth.
//
//   (1) contracts: Clump declares the optional "guides" reference input;
//   (2) primitives: mt19937(1337) draws, the documented flyaway hash vector,
//       Bezier profile weights (expected values from an independent
//       bisection solve of the quadratic Bezier);
//   (3) cooks over hand-built strands (CurveSource -> Clump), expected
//       values derived by hand / an independent transcription of the spec:
//       count mode (farthest-point seeds in strandIndex order, nearest-seed
//       membership past capacity, parting buckets, the K > 255 shuffle
//       path), guides mode
//       (guide table arc-length rule, nearestGuide plane, flyaways),
//       guideSeeded mode, saved overrides, sub-clumps projected onto the
//       bound surface, determinism;
//   (4) Width bezier profile (set and multiply).
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/clumpModes.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/ops/width.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace usdGen;

namespace {

int failures = 0;

void Check(bool v, std::string const &s)
{
    if (!v) {
        ++failures;
        std::printf("FAIL: %s\n", s.c_str());
    } else {
        std::printf("ok: %s\n", s.c_str());
    }
}

bool Near(float a, float b, float tol = 2e-5f) { return std::fabs(a - b) <= tol; }
bool Near3(GfVec3f const &a, GfVec3f const &b, float tol = 2e-5f)
{
    return Near(a[0], b[0], tol) && Near(a[1], b[1], tol) && Near(a[2], b[2], tol);
}
std::string Str(GfVec3f const &v)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(%.6f, %.6f, %.6f)", v[0], v[1], v[2]);
    return buf;
}

// --- fixtures ---------------------------------------------------------------

/// Straight strands along +Y: strand c starts at roots[c] with step steps[c].
UsdGenCurveSetDesc Strands(std::vector<GfVec3f> const &roots, std::vector<float> steps,
                           int cvs = 5)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/groom/hair");
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    size_t const n = roots.size();
    if (steps.size() == 1) steps.assign(n, steps.front());
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * size_t(cvs));
    for (size_t c = 0; c < n; ++c)
        for (int i = 0; i < cvs; ++i)
            set.points[c * size_t(cvs) + size_t(i)] =
                roots[c] + GfVec3f(0.0f, float(i) * steps[c], 0.0f);
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

void AddIntPlane(UsdGenCurveSetDesc &set, char const *name, std::vector<int> const &values)
{
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name);
    plane.type = UsdGenAuthoredPlaneType::Int32;
    plane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    plane.intValues = VtIntArray(values.begin(), values.end());
    set.authoredPlanes.push_back(plane);
}

UsdGenCurveSetDesc GuideSet(std::vector<std::vector<GfVec3f>> const &curves)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/groom/guides");
    set.role = UsdGenRole::Reference;
    set.curveRole = TfToken("guide");
    for (auto const &curve : curves) {
        set.curveVertexCounts.push_back(int(curve.size()));
        for (GfVec3f const &p : curve) set.points.push_back(p);
    }
    set.rest = set.points;
    return set;
}

/// The two guides of the guide-mode scenarios: an L (length 2) and a
/// straight one (length 2).
UsdGenCurveSetDesc TwoGuides()
{
    return GuideSet({{GfVec3f(0, 0, 0), GfVec3f(0, 1, 0), GfVec3f(1, 1, 0)},
                     {GfVec3f(5, 0, 0), GfVec3f(5, 2, 0)}});
}

/// A large quad in the plane y = height.
UsdGenSurfaceDesc Floor(float height)
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/skin");
    s.faceVertexCounts = {4};
    s.faceVertexIndices = {0, 1, 2, 3};
    s.points = {GfVec3f(-10, height, -10), GfVec3f(-10, height, 10), GfVec3f(10, height, 10),
                GfVec3f(10, height, -10)};
    s.restPoints = s.points;
    return s;
}

UsdGenParamValue Param(char const *name, VtValue value)
{
    UsdGenParamValue p;
    p.name = TfToken(name);
    p.value = std::move(value);
    return p;
}

/// The reference-mode defaults the importer authors: Bezier profile
/// (0, 0.5 @ 0.5, 1) -> w(u) = u, amount 1, no length preservation.
std::vector<UsdGenParamValue> ReferenceParams(char const *centers)
{
    return {Param("clump:centers", VtValue(TfToken(centers))),
            Param("clump:profile:mode", VtValue(TfToken("bezier"))),
            Param("clump:root", VtValue(0.0f)), Param("clump:mid", VtValue(0.5f)),
            Param("clump:midShift", VtValue(0.5f)), Param("clump:tip", VtValue(1.0f)),
            Param("clump:amount", VtValue(1.0f)), Param("preserveLength", VtValue(0.0f))};
}

UsdGenGraphDesc ClumpDesc(UsdGenCurveSetDesc const &strands,
                          std::vector<UsdGenParamValue> const &params,
                          UsdGenCurveSetDesc const *guides = nullptr,
                          UsdGenSurfaceDesc const *surface = nullptr, int seed = 1337)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(strands);
    UsdGenNodeDesc source;
    source.path = SdfPath("/groom/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {strands.path};
    source.params = {Param("rebind", VtValue(TfToken("never"))), Param("useRest", VtValue(true))};
    desc.nodes.push_back(source);
    UsdGenNodeDesc clump;
    clump.path = SdfPath("/groom/clump");
    clump.type = TfToken("UsdGenClump");
    clump.seed = seed;
    clump.inputs = {source.path};
    clump.params = params;
    if (guides) {
        desc.curveSets.push_back(*guides);
        clump.curves = {guides->path};
    }
    if (surface) {
        desc.surfaces.push_back(*surface);
        clump.surfaces = {surface->path};
    }
    desc.nodes.push_back(clump);
    desc.terminal = clump.path;
    return desc;
}

bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
          std::vector<std::string> *warnings = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors) std::printf("  run: %s\n", e.c_str());
        return false;
    }
    if (warnings) *warnings = run.diagnostics.warnings;
    *out = graph.Output();
    return true;
}

GfVec3f Point(UsdGenCurveBuffer const &b, size_t curve, size_t i)
{
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(b, &spans, &error) || curve + 1 >= spans.size()) return GfVec3f(NAN);
    size_t const cv = spans[curve] + i;
    return GfVec3f(b.px[cv], b.py[cv], b.pz[cv]);
}

std::vector<int> ClumpIds(UsdGenCurveBuffer const &b, char const *name = "clumpId_0")
{
    for (UsdGenPlane const &plane : b.extraCurve)
        if (plane.name == TfToken(name)) return std::vector<int>(plane.i.begin(), plane.i.end());
    return {};
}

std::string IdsStr(std::vector<int> const &ids)
{
    std::string s = "[";
    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
    return s + "]";
}

void ExpectIds(UsdGenCurveBuffer const &out, std::vector<int> const &expected,
               std::string const &label)
{
    std::vector<int> const ids = ClumpIds(out);
    Check(ids == expected, label + " clumpId_0 " + IdsStr(ids) + " == " + IdsStr(expected));
}

void ExpectPoint(UsdGenCurveBuffer const &out, size_t curve, size_t i, GfVec3f const &expected,
                 std::string const &label)
{
    GfVec3f const got = Point(out, curve, i);
    Check(Near3(got, expected), label + " curve " + std::to_string(curve) + " cv " +
                                    std::to_string(i) + " " + Str(got) + " == " + Str(expected));
}

bool SamePoints(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    return a.totalCvs == b.totalCvs &&
           std::memcmp(a.px.cdata(), b.px.cdata(), a.totalCvs * sizeof(float)) == 0 &&
           std::memcmp(a.py.cdata(), b.py.cdata(), a.totalCvs * sizeof(float)) == 0 &&
           std::memcmp(a.pz.cdata(), b.pz.cdata(), a.totalCvs * sizeof(float)) == 0 &&
           ClumpIds(a) == ClumpIds(b);
}

// --- (1) contracts ------------------------------------------------------------

void TestContracts()
{
    std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenClump"));
    Check(op != nullptr, "registry creates UsdGenClump");
    if (!op) return;
    TfSpan<const TfToken> const refs = op->ReferenceInputs();
    Check(refs.size() == 1 && refs[0] == TfToken("guides"),
          "UsdGenClump declares the guides reference input");
    TfSpan<const TfToken> const outs = op->OutputPrimvars();
    Check(outs.size() == 4 && outs[0] == TfToken("clumpId_0") &&
              outs[1] == TfToken("clumpCenter_0") &&
              outs[2] == TfToken("clumpCenterId_0") &&
              outs[3] == TfToken("clumpWeight_0"),
          "UsdGenClump emits clumpId_0 plus its centre/weight planes");
    TfSpan<const TfToken> const topo = op->TopologyParameters();
    for (char const *name : {"clump:centers", "clump:count", "clump:profile:mode", "clump:root",
                             "clump:flyaway:fraction", "clump:sub:enabled", "clump:overrides",
                             "clump:overrideCount"})
        Check(std::find(topo.begin(), topo.end(), TfToken(name)) != topo.end(),
              std::string("UsdGenClump ") + name + " is capture-class");
    // An unknown centre mode fails closed.
    UsdGenCurveBuffer out;
    std::vector<UsdGenParamValue> bad = ReferenceParams("bogus");
    Check(!Cook(ClumpDesc(Strands({GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)}, {0.25f}), bad), &out),
          "an unknown usdGen:clump:centers fails the cook");
}

// --- (2) primitives -------------------------------------------------------------

void TestPrimitives()
{
    std::mt19937 rng(1337);
    uint32_t const a = rng(), b = rng(), c = rng();
    Check(a == 1125387415u && b == 2407456957u && c == 681542492u,
          "mt19937(1337) draws 1125387415, 2407456957, 681542492");

    // Documented flyaway hash vector u01(s), s = 0..7 (6 decimals).
    double const u[8] = {0.883311, 0.566562, 0.591190, 0.113450,
                         0.431456, 0.386768, 0.739817, 0.389830};
    bool vector = true;
    for (int s = 0; s < 8; ++s) {
        vector = vector && clumpRef::IsFlyaway(uint64_t(s), float(u[s] + 2e-6)) &&
                 !clumpRef::IsFlyaway(uint64_t(s), float(u[s] - 2e-6));
    }
    Check(vector, "flyaway test matches the documented hash vector u01(0..7)");
    Check(!clumpRef::IsFlyaway(3, 0.0f) && clumpRef::IsFlyaway(0, 1.0f) &&
              clumpRef::IsFlyaway(0, 7.0f),
          "flyaway fraction is clamped to [0, 1]");

    // Bezier weights: {u, L, amount, root, mid, midShift, tip, expected}.
    struct Row { float u, L, amount, r, m, c, t, w; };
    Row const rows[] = {
        {0.00f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.1600000f},
        {0.10f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.3075528f},
        {0.25f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.4296953f},
        {0.50f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.4865055f},
        {0.75f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.4367987f},
        {1.00f, 1.0f, 0.80f, 0.20f, 0.90f, 0.30f, 0.40f, 0.3200000f},
        // midShift 0.05 clamps to 0.1; 0.95 to 0.9.
        {0.30f, 1.0f, 1.00f, 0.00f, 1.00f, 0.05f, 0.00f, 0.5000000f},
        {0.60f, 1.0f, 1.00f, 1.00f, 0.00f, 0.95f, 1.00f, 0.5173242f},
        // A zero-length strand takes mid; amount clamps to 1.
        {0.40f, 0.0f, 0.50f, 0.20f, 0.90f, 0.30f, 0.40f, 0.4500000f},
        {0.40f, 1.0f, 1.70f, 0.20f, 0.90f, 0.30f, 0.40f, 0.6000000f},
    };
    for (Row const &row : rows) {
        clumpRef::BezierControls k;
        k.root = row.r; k.mid = row.m; k.midShift = row.c; k.tip = row.t;
        float const w = clumpRef::ClumpWeight(row.u, row.L, row.amount, k);
        char label[160];
        std::snprintf(label, sizeof(label),
                      "Bezier weight u=%.2f L=%.1f amount=%.2f (%.2f,%.2f@%.2f,%.2f) %.7f == %.7f",
                      row.u, row.L, row.amount, row.r, row.m, row.c, row.t, w, row.w);
        Check(Near(w, row.w, 1e-6f), label);
    }
    clumpRef::BezierControls linear;
    bool identity = true;
    for (int i = 0; i <= 10; ++i)
        identity = identity && Near(clumpRef::ClumpWeight(i / 10.0f, 1.0f, 1.0f, linear),
                                    i / 10.0f, 1e-6f);
    Check(identity, "root 0 / mid 0.5 @ 0.5 / tip 1 is the linear profile w(u) = u");
}

// --- (3) cooks --------------------------------------------------------------------

// Seven strands, strandIndex permuted: strand order A = c1 c3 c5 c0 c6 c4 c2.
// K = 2: first seed A[1125387415 % 7 = 5] = c4; farthest from c4 = c0.
// c0's clump takes c0 c1 c2 c3 c6: five members, over the reference's
// capacity ceil(7/2) = 4, but its capacity pass only moves a strand to a
// strictly nearer centre, so c6 stays with c0 (a far reassignment would
// pull its tip 2 units sideways).
std::vector<GfVec3f> const kCountRoots = {
    GfVec3f(0, 0, 0), GfVec3f(0.1f, 0, 0), GfVec3f(0.2f, 0, 0.05f), GfVec3f(0.3f, 0, 0),
    GfVec3f(2.0f, 0, 0), GfVec3f(2.1f, 0, 0.1f), GfVec3f(0.15f, 0, 0.3f)};

void TestCountMode()
{
    UsdGenCurveSetDesc strands = Strands(kCountRoots, {0.25f});
    AddIntPlane(strands, "strandIndex", {3, 0, 6, 1, 5, 2, 4});
    std::vector<UsdGenParamValue> params = ReferenceParams("count");
    params.push_back(Param("clump:count", VtValue(2)));
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(strands, params), &out)) {
        Check(false, "count mode cooks");
        return;
    }
    ExpectIds(out, {1, 1, 1, 1, 0, 0, 1}, "count: seeds [c4, c0], nearest seed");
    ExpectPoint(out, 4, 4, GfVec3f(2.0f, 1.0f, 0.0f), "count: seed c4 keeps its tip");
    ExpectPoint(out, 0, 4, GfVec3f(0.0f, 1.0f, 0.0f), "count: seed c0 keeps its tip");
    ExpectPoint(out, 5, 2, GfVec3f(2.05f, 0.5f, 0.05f), "count: c5 lerps half way at u=.5");
    ExpectPoint(out, 5, 4, GfVec3f(2.0f, 1.0f, 0.0f), "count: c5 tip on seed c4");
    ExpectPoint(out, 6, 2, GfVec3f(0.075f, 0.5f, 0.15f), "count: c6 over capacity stays with c0");
    ExpectPoint(out, 6, 4, GfVec3f(0.0f, 1.0f, 0.0f), "count: c6 tip on its nearest seed c0");
    ExpectPoint(out, 2, 2, GfVec3f(0.1f, 0.5f, 0.025f), "count: c2 lerps toward c0");
    ExpectPoint(out, 3, 0, GfVec3f(0.3f, 0.0f, 0.0f), "count: roots stay (root weight 0)");

    UsdGenCurveBuffer again;
    Check(Cook(ClumpDesc(strands, params), &again) && SamePoints(out, again),
          "count: two cooks are bitwise identical");

    // Without the strandIndex plane the curve order is the strand order: the
    // first seed becomes c5, so c4 now lerps toward c5.
    UsdGenCurveBuffer curveOrder;
    if (Cook(ClumpDesc(Strands(kCountRoots, {0.25f}), params), &curveOrder))
        ExpectPoint(curveOrder, 4, 4, GfVec3f(2.1f, 1.0f, 0.1f),
                    "count: strand order follows strandIndex, not curve order");

    // Parting keys: key 0 = {c0 c1 c2 c3 c6}, key 1 = {c4 c5}. Group 1 gets
    // max(1, round(5*2/7)) = 1 seed grp[1125387415 % 5 = 0] = c1, group 2 the
    // remaining seed grp[2407456957 % 2 = 1] = c4; candidates stay inside the
    // key bucket, so key 0 keeps all five members despite cap 4.
    UsdGenCurveSetDesc parted = strands;
    AddIntPlane(parted, "partId", {0, 0, 0, 0, 1, 1, 0});
    UsdGenCurveBuffer part;
    if (Cook(ClumpDesc(parted, params), &part)) {
        ExpectIds(part, {0, 0, 0, 0, 1, 1, 0}, "count + partId: seeds [c1, c4] per key");
        ExpectPoint(part, 6, 4, GfVec3f(0.1f, 1.0f, 0.0f), "count + partId: c6 tip on c1");
        ExpectPoint(part, 5, 4, GfVec3f(2.0f, 1.0f, 0.0f), "count + partId: c5 tip on c4");
    } else {
        Check(false, "count + partId cooks");
    }

    // K = 0 clumps nothing.
    std::vector<UsdGenParamValue> none = ReferenceParams("count");
    none.push_back(Param("clump:count", VtValue(0)));
    UsdGenCurveBuffer zero;
    Check(Cook(ClumpDesc(strands, none), &zero) &&
              ClumpIds(zero) == std::vector<int>(7, -1) &&
              Near3(Point(zero, 5, 4), GfVec3f(2.1f, 1.0f, 0.1f)),
          "count: clump:count 0 leaves every strand unclumped");
}

// K > 255: seeds are the first K of an MSVC-style shuffle of the strand
// order, every strand joins its nearest seed (no balancing).
void TestCountShuffle()
{
    std::vector<GfVec3f> roots;
    for (int i = 0; i < 300; ++i) {
        float const jx = strand::Hash01(uint64_t(i) * 2u) * 0.4f;
        float const jz = strand::Hash01(uint64_t(i) * 2u + 1u) * 0.4f;
        roots.emplace_back(float(i % 20) + jx, 0.0f, float(i / 20) + jz);
    }
    size_t const K = 260;
    std::vector<UsdGenParamValue> params = ReferenceParams("count");
    params.push_back(Param("clump:count", VtValue(int(K))));
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(Strands(roots, {0.25f}), params), &out)) {
        Check(false, "count K > 255 cooks");
        return;
    }
    std::mt19937 rng(1337);
    std::vector<uint32_t> order(roots.size());
    std::iota(order.begin(), order.end(), 0u);
    strand::MsvcShuffle(&order, rng);
    std::vector<int> expected(roots.size(), -1);
    for (size_t c = 0; c < roots.size(); ++c) {
        float best = 0.0f;
        for (size_t k = 0; k < K; ++k) {
            GfVec3f const d = roots[c] - roots[order[k]];
            float const d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (expected[c] < 0 || d2 < best) {
                best = d2;
                expected[c] = int(k);
            }
        }
    }
    Check(ClumpIds(out) == expected, "count K=260: shuffled seeds, plain nearest assignment");
    bool seedsKept = true;
    for (size_t k = 0; k < K; ++k)
        seedsKept = seedsKept && Near3(Point(out, order[k], 4), roots[order[k]] + GfVec3f(0, 1, 0));
    Check(seedsKept, "count K=260: seed strands are unchanged");
}

// Guides: G0 = L of length 2 (corner at (0,1,0)), G1 straight at x = 5.
// Guide table ng = clamp(5, 8, 1000) = 8 uniform-arc samples, so a = 1 hits
// the midpoint of samples 3 and 4 (1/14, 13/14, 0), cutting the corner.
void TestGuidesMode()
{
    UsdGenCurveSetDesc const guides = TwoGuides();
    std::vector<GfVec3f> const roots = {GfVec3f(0.2f, 0, 0), GfVec3f(-0.1f, 0, 0.1f),
                                        GfVec3f(5.3f, 0, 0), GfVec3f(4.8f, 0, 0)};
    UsdGenCurveSetDesc const strands = Strands(roots, {0.25f, 0.75f, 0.25f, 0.25f});
    std::vector<UsdGenParamValue> params = ReferenceParams("guides");
    // u01(3) = 0.113450 < 0.12: strand 3 is a flyaway, half weight.
    params.push_back(Param("clump:flyaway:fraction", VtValue(0.12f)));
    params.push_back(Param("clump:flyaway:amount", VtValue(0.5f)));
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(strands, params, &guides), &out)) {
        Check(false, "guides mode cooks");
        return;
    }
    ExpectIds(out, {0, 0, 1, -1}, "guides: id = guide, flyaway -1");
    // s0 (L = 1 <= 2): follows the guide by arc length a = u.
    ExpectPoint(out, 0, 2, GfVec3f(0.1f, 0.5f, 0.0f), "guides: s0 u=.5");
    ExpectPoint(out, 0, 3, GfVec3f(0.05f, 0.75f, 0.0f), "guides: s0 u=.75");
    ExpectPoint(out, 0, 4, GfVec3f(0.0714286f, 0.9285714f, 0.0f),
                "guides: s0 tip on the resampled guide (corner cut)");
    // s1 (L = 3 > 2): the guide is stretched, a = 2u.
    ExpectPoint(out, 1, 2, GfVec3f(-0.0142857f, 1.2142857f, 0.05f), "guides: s1 u=.5");
    ExpectPoint(out, 1, 3, GfVec3f(0.35f, 1.3125f, 0.025f), "guides: s1 u=.75");
    ExpectPoint(out, 1, 4, GfVec3f(1.0f, 1.0f, 0.0f), "guides: s1 tip on the guide tip");
    ExpectPoint(out, 2, 4, GfVec3f(5.0f, 1.0f, 0.0f), "guides: s2 tip on G1");
    ExpectPoint(out, 3, 4, GfVec3f(4.9f, 1.0f, 0.0f), "guides: flyaway s3 pulled half way");
    ExpectPoint(out, 3, 3, GfVec3f(4.875f, 0.75f, 0.0f), "guides: flyaway s3 u=.75");

    // The upstream nearestGuide plane overrides the root-distance guide.
    UsdGenCurveSetDesc planed = strands;
    AddIntPlane(planed, "nearestGuide", {1, 0, 1, 1});
    UsdGenCurveBuffer viaPlane;
    if (Cook(ClumpDesc(planed, params, &guides), &viaPlane)) {
        ExpectIds(viaPlane, {1, 0, 1, -1}, "guides + nearestGuide plane");
        ExpectPoint(viaPlane, 0, 2, GfVec3f(2.6f, 0.5f, 0.0f), "guides + plane: s0 toward G1");
        ExpectPoint(viaPlane, 0, 4, GfVec3f(5.0f, 1.0f, 0.0f), "guides + plane: s0 tip on G1");
    } else {
        Check(false, "guides + nearestGuide plane cooks");
    }

    // Without a guide set the mode warns and clumps nothing.
    UsdGenCurveBuffer bare;
    std::vector<std::string> warnings;
    Check(Cook(ClumpDesc(strands, params), &bare, &warnings) &&
              ClumpIds(bare) == std::vector<int>(4, -1) &&
              Near3(Point(bare, 0, 4), GfVec3f(0.2f, 1.0f, 0.0f)) &&
              std::any_of(warnings.begin(), warnings.end(), [](std::string const &w) {
                  return w.find("usdGen:guides") != std::string::npos;
              }),
          "guides mode without usdGen:guides warns and leaves strands alone");
}

// Guide-seeded: buckets {s0 s1 s2} (G0) and {s3 s4 s5} (G1); two non-empty
// guides == K, so one seed each: bucket[1125387415 % 3 = 1] = s1 and
// bucket[2407456957 % 3 = 1] = s4. Centre curves grow from the seed roots
// along the guide by arc length (s1 follows the L's corner).
void TestGuideSeeded()
{
    UsdGenCurveSetDesc const guides = TwoGuides();
    std::vector<GfVec3f> const roots = {GfVec3f(0.1f, 0, 0),  GfVec3f(-0.1f, 0, 0.05f),
                                        GfVec3f(0, 0, -0.1f), GfVec3f(5.1f, 0, 0),
                                        GfVec3f(4.9f, 0, 0.05f), GfVec3f(5, 0, -0.1f)};
    std::vector<UsdGenParamValue> params = ReferenceParams("guideSeeded");
    params.push_back(Param("clump:count", VtValue(2)));
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(Strands(roots, {0.25f}), params, &guides), &out)) {
        Check(false, "guideSeeded mode cooks");
        return;
    }
    ExpectIds(out, {0, 0, 0, 1, 1, 1}, "guideSeeded: one seed per guide bucket");
    ExpectPoint(out, 1, 4, GfVec3f(-0.0285714f, 0.9285714f, 0.05f),
                "guideSeeded: seed s1 takes its synthesised (guide-shaped) curve");
    ExpectPoint(out, 0, 4, GfVec3f(-0.0285714f, 0.9285714f, 0.05f),
                "guideSeeded: s0 tip on the synthesised centre curve");
    ExpectPoint(out, 2, 2, GfVec3f(-0.05f, 0.5f, -0.025f), "guideSeeded: s2 u=.5");
    ExpectPoint(out, 3, 2, GfVec3f(5.0f, 0.5f, 0.025f), "guideSeeded: s3 u=.5 toward s4");
    ExpectPoint(out, 5, 4, GfVec3f(4.9f, 1.0f, 0.05f), "guideSeeded: s5 tip on s4's curve");
}

// One random clump (K = 1, seed A[1125387415 % 6 = 1] = c1) split into two
// sub-clumps around the mean root; sector roots land on the floor y = -0.2.
void TestSubClumps()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0),       GfVec3f(0.10f, 0, 0.02f),
                                        GfVec3f(0.12f, 0, 0.11f), GfVec3f(-0.05f, 0, 0.09f),
                                        GfVec3f(-0.11f, 0, -0.03f), GfVec3f(0.03f, 0, -0.12f)};
    std::vector<UsdGenParamValue> params = ReferenceParams("count");
    params.push_back(Param("clump:count", VtValue(1)));
    params.push_back(Param("clump:sub:enabled", VtValue(true)));
    params.push_back(Param("clump:sub:count", VtValue(2)));
    params.push_back(Param("clump:sub:spread", VtValue(0.5f)));
    params.push_back(Param("clump:sub:amount", VtValue(1.0f)));
    params.push_back(Param("clump:sub:ids", VtValue(true)));
    UsdGenSurfaceDesc const floor = Floor(-0.2f);
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(Strands(roots, {0.25f}), params, nullptr, &floor), &out)) {
        Check(false, "sub-clumps cook");
        return;
    }
    // Angular order from the largest gap: {c5 c0 c4} then {c3 c2 c1}.
    ExpectIds(out, {0, 1, 1, 1, 0, 0}, "sub-clumps: ids key*count + sector");
    ExpectPoint(out, 0, 2, GfVec3f(0.0136166f, 0.4531305f, -0.0319623f), "sub-clumps: c0 u=.5");
    ExpectPoint(out, 0, 4, GfVec3f(0.0405665f, 1.0125218f, -0.0389245f),
                "sub-clumps: sector 0 tip = mean tip + (T - projected root)");
    ExpectPoint(out, 4, 2, GfVec3f(-0.0138834f, 0.4531305f, -0.0394623f), "sub-clumps: c4 u=.5");
    ExpectPoint(out, 2, 2, GfVec3f(0.1182920f, 0.4471059f, 0.0896165f), "sub-clumps: c2 u=.5");
    ExpectPoint(out, 1, 4, GfVec3f(0.1482506f, 0.9884236f, 0.0875663f), "sub-clumps: sector 1 tip");
    ExpectPoint(out, 3, 0, GfVec3f(-0.05f, 0.0f, 0.09f), "sub-clumps: roots stay");

    // Without sub-clump ids the clump id stays the parent's.
    params.push_back(Param("clump:sub:ids", VtValue(false)));
    params.erase(std::find_if(params.begin(), params.end(), [](UsdGenParamValue const &p) {
        return p.name == TfToken("clump:sub:ids");
    }));
    UsdGenCurveBuffer plain;
    Check(Cook(ClumpDesc(Strands(roots, {0.25f}), params, nullptr, &floor), &plain) &&
              ClumpIds(plain) == std::vector<int>(6, 0) &&
              Near3(Point(plain, 0, 4), GfVec3f(0.0405665f, 1.0125218f, -0.0389245f)),
          "sub-clumps without ids keep the parent clump id");
}

// Saved overrides on a K = 1 random clump (seed c1): strand 0 is unclumped
// toward its snapshot by the flyaway amount 0.25, strands 2..4 form custom
// clump 7 centred on c2 (root nearest their mean), INT_MIN keeps the rest.
void TestOverrides()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0),         GfVec3f(0.2f, 0, 0),
                                        GfVec3f(0.4f, 0, 0.1f),    GfVec3f(0.5f, 0, 0.3f),
                                        GfVec3f(0.65f, 0, 0.05f),  GfVec3f(1.0f, 0, 0)};
    std::vector<UsdGenParamValue> params = ReferenceParams("count");
    params.push_back(Param("clump:count", VtValue(1)));
    params.push_back(Param("clump:flyaway:amount", VtValue(0.25f)));
    params.push_back(Param("clump:overrides",
                           VtValue(VtIntArray{-1, INT_MIN, 7, 7, 7, INT_MIN})));
    params.push_back(Param("clump:overrideCount", VtValue(6)));
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(Strands(roots, {0.25f}), params), &out)) {
        Check(false, "overrides cook");
        return;
    }
    ExpectIds(out, {-1, 0, 7, 7, 7, 0}, "overrides: -1 / keep / custom 7");
    ExpectPoint(out, 0, 2, GfVec3f(0.025f, 0.5f, 0.0f), "overrides: -1 lerps back to 0.25");
    ExpectPoint(out, 0, 4, GfVec3f(0.05f, 1.0f, 0.0f), "overrides: -1 tip");
    ExpectPoint(out, 5, 4, GfVec3f(0.2f, 1.0f, 0.0f), "overrides: INT_MIN keeps the clump");
    ExpectPoint(out, 2, 4, GfVec3f(0.4f, 1.0f, 0.1f), "overrides: custom centre c2 restored");
    ExpectPoint(out, 3, 2, GfVec3f(0.45f, 0.5f, 0.2f), "overrides: c3 toward c2's snapshot");
    ExpectPoint(out, 4, 4, GfVec3f(0.4f, 1.0f, 0.1f), "overrides: c4 tip on c2");

    // A saved count that does not match the strand count is ignored.
    params.back() = Param("clump:overrideCount", VtValue(5));
    UsdGenCurveBuffer ignored;
    std::vector<std::string> warnings;
    Check(Cook(ClumpDesc(Strands(roots, {0.25f}), params), &ignored, &warnings) &&
              ClumpIds(ignored) == std::vector<int>(6, 0) &&
              std::any_of(warnings.begin(), warnings.end(), [](std::string const &w) {
                  return w.find("overrides ignored") != std::string::npos;
              }),
          "overrides with a mismatched saved count are ignored with a warning");
}

// Flyaways also apply to the density mode: amount 0 frees them entirely.
void TestDensityFlyaway()
{
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(0.01f, 0, 0),
                                        GfVec3f(0.02f, 0, 0), GfVec3f(0.03f, 0, 0)};
    std::vector<UsdGenParamValue> params = {
        Param("clump:density", VtValue(0.0f)), Param("clump:size", VtValue(10.0f)),
        Param("clump:amount", VtValue(1.0f)),
        Param("clump:flyaway:fraction", VtValue(0.12f))};
    UsdGenCurveBuffer out;
    if (!Cook(ClumpDesc(Strands(roots, {0.25f}), params), &out)) {
        Check(false, "density + flyaway cooks");
        return;
    }
    std::vector<int> const ids = ClumpIds(out);
    Check(ids.size() == 4 && ids[3] == -1 && ids[0] >= 0 && ids[1] >= 0 && ids[2] >= 0,
          "density: the flyaway strand (index 3) has no clump id");
    Check(Near3(Point(out, 3, 4), GfVec3f(0.03f, 1.0f, 0.0f)),
          "density: flyaway amount 0 leaves the flyaway unclumped");
}

// --- (4) width --------------------------------------------------------------------

UsdGenGraphDesc WidthDesc(std::vector<UsdGenParamValue> const &params)
{
    UsdGenGraphDesc desc = ClumpDesc(Strands({GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)}, {0.25f}), {});
    desc.nodes.back().type = TfToken("UsdGenWidth");
    desc.nodes.back().path = SdfPath("/groom/width");
    desc.nodes.back().params = params;
    desc.terminal = SdfPath("/groom/width");
    return desc;
}

void TestWidth()
{
    struct Case {
        float r, m, c, t;
        float expected[5];
        char const *label;
    };
    Case const cases[] = {
        {1.0f, 0.5f, 0.5f, 0.2f, {0.02f, 0.01525f, 0.011f, 0.00725f, 0.004f},
         "width bezier root 1 / mid .5 / tip .2"},
        {0.0f, 0.0f, 0.5f, 0.0f, {0.02f, 0.0075f, 0.01f, 0.0075f, 0.02f},
         "width bezier all-zero controls -> mid 1, f <= 1e-6 -> 1"},
        {0.3f, 1.0f, 0.25f, 0.0f, {0.006f, 0.0116936f, 0.0103181f, 0.0060183f, 0.02f},
         "width bezier midShift .25, zero tip -> full width"},
    };
    for (Case const &k : cases) {
        UsdGenCurveBuffer out;
        bool ok = Cook(WidthDesc({Param("width", VtValue(0.02f)),
                                  Param("width:profile:mode", VtValue(TfToken("bezier"))),
                                  Param("width:root", VtValue(k.r)),
                                  Param("width:mid", VtValue(k.m)),
                                  Param("width:midShift", VtValue(k.c)),
                                  Param("width:tip", VtValue(k.t))}),
                       &out) &&
                  out.width.size() == 10;
        for (int curve = 0; ok && curve < 2; ++curve)
            for (int i = 0; i < 5; ++i)
                ok = ok && Near(out.width[size_t(curve * 5 + i)], k.expected[i], 1e-7f);
        Check(ok, k.label);
    }
    // Multiply mode: the incoming 0.1 widths scaled by the profile.
    UsdGenCurveBuffer mult;
    bool ok = Cook(WidthDesc({Param("width", VtValue(1.0f)), Param("replace", VtValue(false)),
                              Param("width:profile:mode", VtValue(TfToken("bezier"))),
                              Param("width:root", VtValue(1.0f)), Param("width:mid", VtValue(0.5f)),
                              Param("width:midShift", VtValue(0.5f)),
                              Param("width:tip", VtValue(0.2f))}),
                   &mult) &&
              mult.width.size() == 10;
    float const f[5] = {1.0f, 0.7625f, 0.55f, 0.3625f, 0.2f};
    for (int i = 0; ok && i < 5; ++i) ok = Near(mult.width[size_t(i)], 0.1f * f[i], 1e-7f);
    Check(ok, "width bezier multiplies the incoming widths with usdGen:replace = false");
    // The default ramp mode is unchanged (flat knots -> the base width).
    UsdGenCurveBuffer ramp;
    ok = Cook(WidthDesc({Param("width", VtValue(0.02f)), Param("width:root", VtValue(0.0f))}),
              &ramp) &&
         ramp.width.size() == 10;
    for (size_t i = 0; ok && i < 10; ++i) ok = Near(ramp.width[i], 0.02f, 1e-7f);
    Check(ok, "width ramp mode ignores the Bezier controls");

    // The LUT the GPU lanes sample holds the same factors at k/(size-1).
    UsdGenNodeDesc node;
    node.type = TfToken("UsdGenWidth");
    node.params = {Param("width:profile:mode", VtValue(TfToken("bezier"))),
                   Param("width:root", VtValue(1.0f)), Param("width:mid", VtValue(0.5f)),
                   Param("width:midShift", VtValue(0.5f)), Param("width:tip", VtValue(0.2f))};
    UsdGenParamView view;
    view.node = &node;
    float lut[5] = {-1, -1, -1, -1, -1};
    ok = UsdGenWidthBezierLut(view, lut, 5);
    for (int i = 0; ok && i < 5; ++i) ok = Near(lut[i], f[i], 1e-7f);
    UsdGenNodeDesc rampNode;
    UsdGenParamView rampView;
    rampView.node = &rampNode;
    Check(ok && !UsdGenWidthBezierLut(rampView, lut, 5) && lut[0] == 1.0f,
          "UsdGenWidthBezierLut bakes the Bezier factors (and leaves ramp mode alone)");
}

}  // namespace

int main()
{
    TestContracts();
    TestPrimitives();
    TestCountMode();
    TestCountShuffle();
    TestGuidesMode();
    TestGuideSeeded();
    TestSubClumps();
    TestOverrides();
    TestDensityFlyaway();
    TestWidth();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::puts("all clump-mode checks passed");
    return 0;
}
