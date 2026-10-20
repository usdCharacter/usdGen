// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenTieCollide: UsdGenTieClump (hair ties) and UsdGenCollide's
// `relax` resolve type (the imported groom's push-out), plus the exactness
// of Collide's flexible closest-point index.
//
//   (1) TieClump: registry contract; the tie pull at hand-computed weights
//       (amount / spread falloff), the tie arc gate, multi-entry ties,
//       sequential ties against an independent transcription of the spec,
//       strands of untied guides bit-for-bit, determinism, fail-closed edges.
//   (2) Collide relax: lifts penetrating CVs to the root/tip clearance,
//       keeps every segment length, root fixed, equals a direct
//       transcription of the push-out algorithm over an analytic plane
//       (window 1 and window 2), the unresolved flag plane, mask envelope,
//       non-penetrating identity, determinism, fail-closed edges.
//   (3) Collide flexible over a bumpy many-triangle mesh is bit-identical to
//       the brute-force first-win closest-point scan it replaced.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"

#include <algorithm>
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
static void Check(bool v, std::string const &s) { Check(v, s.c_str()); }

static bool Near(double a, double b, double tol)
{
    return std::fabs(a - b) <= tol;
}

using Polyline = std::vector<GfVec3f>;

// Hair curve set from explicit polylines (CurveSource fixture shape:
// surface-free, rebind never, identity root frames).
static UsdGenCurveSetDesc Strands(SdfPath const &path, std::vector<Polyline> const &curves)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    for (Polyline const &c : curves) {
        set.curveVertexCounts.push_back(int(c.size()));
        for (GfVec3f const &p : c) set.points.push_back(p);
    }
    size_t const n = curves.size();
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

static Polyline Straight(GfVec3f root, GfVec3f step, int cvs)
{
    Polyline p;
    for (int i = 0; i < cvs; ++i) p.push_back(root + step * float(i));
    return p;
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

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out,
                 bool quiet = false)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        if (!quiet)
            for (auto const &e : compiled.errors) std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        if (!quiet)
            for (auto const &e : run.diagnostics.errors) std::printf("  run: %s\n", e.c_str());
        return false;
    }
    *out = graph.Output();
    return true;
}

static GfVec3f Point(UsdGenCurveBuffer const &b, size_t cv)
{
    return GfVec3f(b.px[cv], b.py[cv], b.pz[cv]);
}

// Output curve whose root equals `root` (curve order is the engine's), or -1.
static int FindCurve(UsdGenCurveBuffer const &b, GfVec3f const &root, size_t cvs)
{
    for (size_t c = 0; c < b.totalCurves; ++c)
        if (Point(b, c * cvs) == root) return int(c);
    return -1;
}

static Polyline CurvePoints(UsdGenCurveBuffer const &b, int curve, size_t cvs)
{
    Polyline p;
    if (curve < 0) return p;
    for (size_t i = 0; i < cvs; ++i) p.push_back(Point(b, size_t(curve) * cvs + i));
    return p;
}

// Output curve carrying curveId `id` (Strands() numbers curves 0..n-1).
static int FindCurveById(UsdGenCurveBuffer const &b, uint64_t id)
{
    for (size_t c = 0; c < b.curveId.size(); ++c)
        if (b.curveId[c] == id) return int(c);
    return -1;
}

static UsdGenPlane const *FindCurvePlane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCurve)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

// ===========================================================================
// (1) TieClump
// ===========================================================================

constexpr int kHairCvs = 11;           // 10 segments of 0.1: L = 1
const GfVec3f kRootA(0, 0, 0), kRootB(3, 0, 0), kRootC(6, 0, 0), kRootD(5, 0, 0);

struct TieSpec
{
    std::vector<GfVec3f> positions;
    std::vector<int> counts, guides, vertices;
};

// Guides: 0 = (0, 0.5 j, 0) j<5 (length 2), 1 = (5, {0, 0.5, 1, 2}, 0), 2 = (9, j, 0)
// j<2 (never tied). Strands (11 CVs along +Y, step 0.1): A at x=0 on guide
// 0, B at x=3 on guide 2, C at x=6 with no guide (-1), D at x=5 on guide 1.
static UsdGenGraphDesc TieDesc(TieSpec const &ties,
                               std::vector<UsdGenParamValue> params = {},
                               bool withGuides = true, bool withPlane = true)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/tie");
    desc.defaultWidth = 0.01f;
    GfVec3f const step(0, 0.1f, 0);
    UsdGenCurveSetDesc hair = Strands(SdfPath("/tie/hair"),
        {Straight(kRootA, step, kHairCvs), Straight(kRootB, step, kHairCvs),
         Straight(kRootC, step, kHairCvs), Straight(kRootD, step, kHairCvs)});
    if (withPlane) {
        UsdGenAuthoredPlaneDesc primary;
        primary.name = TfToken("primaryGuide");
        primary.type = UsdGenAuthoredPlaneType::Int32;
        primary.domain = UsdGenAuthoredPlaneDomain::Primitive;
        primary.arity = 1;
        primary.intValues = {0, 2, -1, 1};
        hair.authoredPlanes.push_back(primary);
    }
    desc.curveSets.push_back(hair);
    UsdGenCurveSetDesc guides;
    guides.path = SdfPath("/tie/guides");
    guides.role = UsdGenRole::Reference;
    guides.curveRole = TfToken("guide");
    guides.curveVertexCounts = {5, 4, 2};
    guides.points = {GfVec3f(0, 0, 0), GfVec3f(0, 0.5f, 0), GfVec3f(0, 1.0f, 0),
                     GfVec3f(0, 1.5f, 0), GfVec3f(0, 2.0f, 0),
                     GfVec3f(5, 0, 0), GfVec3f(5, 0.5f, 0), GfVec3f(5, 1, 0), GfVec3f(5, 2, 0),
                     GfVec3f(9, 0, 0), GfVec3f(9, 1, 0)};
    guides.rest = guides.points;
    desc.curveSets.push_back(guides);
    desc.nodes.push_back(SourceNode(SdfPath("/tie/source"), SdfPath("/tie/hair")));
    UsdGenNodeDesc tie = OpNode("/tie/op", "UsdGenTieClump", SdfPath("/tie/source"), 3);
    if (withGuides) tie.references = {guides.path};
    VtVec3fArray positions(ties.positions.begin(), ties.positions.end());
    tie.params = std::move(params);
    tie.params.push_back({TfToken("tie:positions"), VtValue(positions), false});
    tie.params.push_back({TfToken("tie:entryCounts"),
                          VtValue(VtIntArray(ties.counts.begin(), ties.counts.end())), false});
    tie.params.push_back({TfToken("tie:guideIndices"),
                          VtValue(VtIntArray(ties.guides.begin(), ties.guides.end())), false});
    tie.params.push_back({TfToken("tie:vertexIndices"),
                          VtValue(VtIntArray(ties.vertices.begin(), ties.vertices.end())), false});
    desc.nodes.push_back(tie);
    desc.terminal = SdfPath("/tie/op");
    return desc;
}

// Independent transcription of the tie-clump spec (double precision).
struct RefTie { GfVec3d position; double arc; };
static std::vector<GfVec3d> RefTiePull(Polyline const &in, std::vector<RefTie> const &ties,
                                       double amount, double spread)
{
    std::vector<GfVec3d> q(in.begin(), in.end());
    size_t const n = q.size();
    auto arcs = [&](std::vector<double> *a) {
        a->assign(n, 0.0);
        for (size_t k = 1; k < n; ++k) (*a)[k] = (*a)[k - 1] + (q[k] - q[k - 1]).GetLength();
        return (*a)[n - 1];
    };
    auto clamp01 = [](double x) { return std::min(1.0, std::max(0.0, x)); };
    std::vector<double> arc;
    double L = arcs(&arc);
    double const Lf = L;
    double const amt = clamp01(amount), spr = clamp01(spread);
    double const m = double(std::max<size_t>(1, n - 1));
    for (RefTie const &t : ties) {
        if (!(t.arc < Lf)) continue;
        double const inner = std::max(1e-4, std::max(0.04 * L, 1.5 * L / m));
        double const outer = std::max(0.4 * L, 8.0 * L / m);
        double const W = std::max(inner, outer);
        double best = 1e300, sStar = 0.0;
        for (size_t k = 0; k + 1 < n; ++k) {
            GfVec3d const ab = q[k + 1] - q[k];
            double const len2 = GfDot(ab, ab);
            double const u = len2 > 0 ? clamp01(GfDot(t.position - q[k], ab) / len2) : 0.0;
            double const d2 = (t.position - (q[k] + ab * u)).GetLengthSq();
            if (d2 < best) { best = d2; sStar = arc[k] + std::sqrt(len2) * u; }
        }
        for (size_t k = 1; k < n; ++k) {
            double const x = clamp01(std::fabs(arc[k] - sStar) / (W * spr + (1 - spr) * inner));
            double const f = 1.0 - (3.0 - 2.0 * x) * x * x;
            double const w = clamp01((1 - spr) * f + spr * std::sqrt(clamp01(f))) * amt;
            if (w > 1e-6) q[k] = q[k] + (t.position - q[k]) * w;
        }
        L = arcs(&arc);
        if (L <= 1e-6) break;
    }
    return q;
}

static bool MatchesRef(Polyline const &got, std::vector<GfVec3d> const &want, double tol)
{
    if (got.size() != want.size()) return false;
    for (size_t k = 0; k < got.size(); ++k)
        for (int d = 0; d < 3; ++d)
            if (!Near(got[k][d], want[k][d], tol)) {
                std::printf("  cv %zu axis %d: got %.7f want %.7f\n", k, d, got[k][d], want[k][d]);
                return false;
            }
    return true;
}

static void CheckTieRegistry()
{
    UsdGenOpRegistry &registry = UsdGenOpRegistry::Get();
    TfToken const type("UsdGenTieClump");
    Check(registry.HasKernel(type), "HasKernel(UsdGenTieClump)");
    std::unique_ptr<UsdGenOp> op = registry.Create(type);
    Check(op != nullptr, "Create(UsdGenTieClump)");
    if (!op) return;
    Check(op->TopologyEffect() == UsdGenTopoFx::None, "TieClump has TopologyEffect None");
    Check(op->ReferenceInputs().size() == 1 && op->ReferenceInputs()[0] == TfToken("guides"),
          "TieClump reads the single reference slot 'guides'");
    Check(op->OutputPrimvars().empty(), "TieClump emits no planes");
    std::set<std::string> topo;
    for (TfToken const &t : op->TopologyParameters()) topo.insert(t.GetString());
    bool all = true;
    for (char const *name : {"guides", "tie:amount", "tie:spread", "tie:positions",
                             "tie:entryCounts", "tie:guideIndices", "tie:vertexIndices"})
        all = all && topo.count(name);
    Check(all, "TieClump declares every tie attribute as capture-class");
}

static void CheckTiePull()
{
    // One tie at T = (0.2, 0.5, 0) on guide 0's point 1 (tieArc 0.5 < L = 1).
    // L = 1, m = 10: inner = max(0.04, 0.15) = 0.15, outer = max(0.4, 0.8)
    // = 0.8 = W; s* = 0.5 (closest strand point (0, 0.5, 0)).
    // spread 0.25: radius = 0.8*0.25 + 0.75*0.15 = 0.3125.
    //   k=5: x=0 -> w=1 -> P5 = T exactly.
    //   k=4: x=0.32, sstep=0.241664, f=0.758336,
    //        w = 0.75 f + 0.25 sqrt(f) = 0.786458 -> P4.x = 0.2 w = 0.157292
    //   k=3: x=0.64 -> f=0.295488 -> w=0.357513 -> P3.x = 0.071503
    //   k=1, k=9, k=10: x >= 1 -> untouched.
    TieSpec ties{{GfVec3f(0.2f, 0.5f, 0)}, {1}, {0}, {1}};
    UsdGenCurveBuffer out, again;
    Check(Cook(TieDesc(ties), &out), "TieClump cooks on the CPU lane");
    if (out.totalCvs != 4 * kHairCvs) { Check(false, "TieClump keeps the topology"); return; }
    int const a = FindCurve(out, kRootA, kHairCvs);
    Polyline const A = CurvePoints(out, a, kHairCvs);
    if (A.empty()) { Check(false, "TieClump strand A found by root"); return; }
    Check(A[0] == kRootA, "TieClump keeps the root exactly");
    Check(Near(A[5][0], 0.2, 1e-6) && Near(A[5][1], 0.5, 1e-6) && Near(A[5][2], 0.0, 1e-7),
          "TieClump weight 1 at s*: the tied point lands on the tie");
    Check(Near(A[4][0], 0.157292, 2e-6) && Near(A[6][0], 0.157292, 2e-6),
          "TieClump spread 0.25 weight at |s - s*| = 0.1 is 0.786458 (hand-computed)");
    Check(Near(A[4][1], 0.4 + 0.1 * 0.786458, 2e-6) && Near(A[6][1], 0.6 - 0.1 * 0.786458, 2e-6),
          "TieClump lerps every axis toward the tie");
    Check(Near(A[3][0], 0.071503, 2e-6) && Near(A[7][0], 0.071503, 2e-6),
          "TieClump weight at |s - s*| = 0.2 is 0.357513 (hand-computed)");
    Check(Near(A[2][0], 0.2 * 0.020592, 2e-6), "TieClump weight at |s - s*| = 0.3 is 0.020592");
    Polyline const input = Straight(kRootA, GfVec3f(0, 0.1f, 0), kHairCvs);
    Check(A[1] == input[1] && A[9] == input[9] && A[10] == input[10],
          "TieClump leaves CVs beyond the falloff radius bit-for-bit");
    Check(MatchesRef(A, RefTiePull(Straight(kRootA, GfVec3f(0, 0.1f, 0), kHairCvs),
                                   {{GfVec3d(0.2, 0.5, 0), 0.5}}, 1.0, 0.25), 1e-5),
          "TieClump matches the spec transcription");
    bool untouched = true;
    for (GfVec3f const &root : {kRootB, kRootC, kRootD}) {
        Polyline const s = CurvePoints(out, FindCurve(out, root, kHairCvs), kHairCvs);
        untouched = untouched && s == Straight(root, GfVec3f(0, 0.1f, 0), kHairCvs);
    }
    Check(untouched, "TieClump leaves strands of untied guides (and no guide) bit-for-bit");
    Check(Cook(TieDesc(ties), &again) && again.px == out.px && again.py == out.py &&
              again.pz == out.pz,
          "TieClump cooks twice bit-identical");
}

static void CheckTieAmountSpread()
{
    TieSpec ties{{GfVec3f(0.2f, 0.5f, 0)}, {1}, {0}, {1}};
    Polyline const straight = Straight(kRootA, GfVec3f(0, 0.1f, 0), kHairCvs);
    struct Case { double amount, spread, p4x; char const *label; };
    // spread 0: radius = inner = 0.15, x = 2/3, f = 0.259259, w = f.
    // spread 1: radius = W = 0.8, x = 0.125, f = 0.957031, w = sqrt(f) = 0.978280.
    // amount 0.5 halves the spread-0.25 weight 0.786458.
    Case const cases[] = {
        {1.0, 0.0, 0.2 * 0.259259, "TieClump spread 0 weight is 1 - smoothstep (0.259259)"},
        {1.0, 1.0, 0.2 * 0.978280, "TieClump spread 1 weight is sqrt(1 - smoothstep) (0.978280)"},
        {0.5, 0.25, 0.2 * 0.5 * 0.786458, "TieClump amount 0.5 halves the weight"},
        {3.0, 0.25, 0.2 * 0.786458, "TieClump clamps amount to 1"},
    };
    for (Case const &c : cases) {
        UsdGenCurveBuffer out;
        bool const ok = Cook(TieDesc(ties, {{TfToken("tie:amount"), VtValue(float(c.amount)), false},
                                            {TfToken("tie:spread"), VtValue(float(c.spread)), false}}),
                             &out);
        Polyline const A = CurvePoints(out, ok ? FindCurve(out, kRootA, kHairCvs) : -1, kHairCvs);
        Check(!A.empty() && Near(A[4][0], c.p4x, 2e-6) &&
                  MatchesRef(A, RefTiePull(straight, {{GfVec3d(0.2, 0.5, 0), 0.5}},
                                           c.amount, c.spread), 1e-5),
              c.label);
    }
    UsdGenCurveBuffer off;
    bool const ok = Cook(TieDesc(ties, {{TfToken("tie:amount"), VtValue(0.0f), false}}), &off);
    Check(ok && CurvePoints(off, FindCurve(off, kRootA, kHairCvs), kHairCvs) == straight,
          "TieClump with amount 0 is the identity");
}

static void CheckTieArcGate()
{
    Polyline const straight = Straight(kRootA, GfVec3f(0, 0.1f, 0), kHairCvs);
    // Guide 0's point 4 sits at arc 2.0 >= L = 1: the tie is skipped. An
    // out-of-range vertex index clamps to the last point (same skip); index 0
    // clamps to 1 (arc 0.5) and pulls.
    for (int vertex : {4, 9}) {
        UsdGenCurveBuffer out;
        bool const ok = Cook(TieDesc({{GfVec3f(0.2f, 0.5f, 0)}, {1}, {0}, {vertex}}), &out);
        Check(ok && CurvePoints(out, FindCurve(out, kRootA, kHairCvs), kHairCvs) == straight,
              "TieClump skips a tie whose guide arc is not shorter than the strand (vertex " +
                  std::to_string(vertex) + ")");
    }
    UsdGenCurveBuffer zero;
    bool const ok = Cook(TieDesc({{GfVec3f(0.2f, 0.5f, 0)}, {1}, {0}, {0}}), &zero);
    Polyline const A = CurvePoints(zero, ok ? FindCurve(zero, kRootA, kHairCvs) : -1, kHairCvs);
    Check(!A.empty() && Near(A[5][0], 0.2, 1e-6), "TieClump clamps vertex index 0 to 1");
}

static void CheckTieMultiEntryAndSequence()
{
    // One tie on guides 0 and 1: both strands' s* = 0.5, so their CV 5
    // lands on the tie.
    UsdGenCurveBuffer out;
    bool ok = Cook(TieDesc({{GfVec3f(5.3f, 0.5f, 0)}, {2}, {0, 1}, {1, 1}}), &out);
    Polyline const A = CurvePoints(out, ok ? FindCurve(out, kRootA, kHairCvs) : -1, kHairCvs);
    Polyline const D = CurvePoints(out, ok ? FindCurve(out, kRootD, kHairCvs) : -1, kHairCvs);
    Check(!A.empty() && !D.empty() && Near(A[5][0], 5.3, 1e-5) && Near(D[5][0], 5.3, 1e-5) &&
              Near(D[5][1], 0.5, 1e-6),
          "TieClump applies one tie to every guide it lists");
    Check(!A.empty() && A[0] == kRootA && !D.empty() && D[0] == kRootD,
          "TieClump never moves roots");

    // Two ties on guide 0 run in order with arc/L recomputed in between; an
    // entry naming a guide outside the set is skipped (warning only).
    TieSpec two{{GfVec3f(0.1f, 0.3f, 0.05f), GfVec3f(-0.15f, 0.8f, 0.0f), GfVec3f(1, 1, 1)},
                {1, 1, 1}, {0, 0, 7}, {1, 1, 1}};
    UsdGenCurveBuffer seq;
    ok = Cook(TieDesc(two, {{TfToken("tie:spread"), VtValue(0.6f), false},
                            {TfToken("tie:amount"), VtValue(0.8f), false}}), &seq);
    Polyline const S = CurvePoints(seq, ok ? FindCurve(seq, kRootA, kHairCvs) : -1, kHairCvs);
    Check(ok && MatchesRef(S, RefTiePull(Straight(kRootA, GfVec3f(0, 0.1f, 0), kHairCvs),
                                         {{GfVec3d(0.1, 0.3, 0.05), 0.5},
                                          {GfVec3d(-0.15, 0.8, 0.0), 0.5}}, 0.8, 0.6), 1e-5),
          "TieClump applies sequential ties like the spec (arc/L recomputed per tie)");
}

static void CheckTieFailClosed()
{
    UsdGenCurveBuffer out;
    Check(!Cook(TieDesc({{GfVec3f(0, 0.5f, 0)}, {1, 1}, {0}, {1}}), &out, true),
          "TieClump fails closed when entryCounts and positions differ in size");
    Check(!Cook(TieDesc({{GfVec3f(0, 0.5f, 0)}, {2}, {0}, {1}}), &out, true),
          "TieClump fails closed when the entry arrays do not sum up");
    Check(!Cook(TieDesc({{GfVec3f(0, 0.5f, 0)}, {1}, {0}, {1}}, {}, /*guides*/ false), &out, true),
          "TieClump fails closed when ties are authored without a guide set");
    Check(!Cook(TieDesc({{GfVec3f(0, 0.5f, 0)}, {1}, {0}, {1}}, {}, true, /*plane*/ false), &out,
                true),
          "TieClump fails closed without an upstream primaryGuide plane");
    UsdGenCurveBuffer none;
    Check(Cook(TieDesc({}, {}, /*guides*/ false), &none),
          "TieClump without ties (and without guides) is a pass-through");
}

// ===========================================================================
// (2) Collide relax
// ===========================================================================

// A 20 x 20 quad in z = 0 facing +Z (two fan triangles), so every query in
// |x|,|y| < 10 has the analytic signed distance z and gradient +Z.
static UsdGenSurfaceDesc PlaneSurface(SdfPath const &path, float z = 0.0f)
{
    UsdGenSurfaceDesc s;
    s.path = path;
    s.faceVertexCounts = {4};
    s.faceVertexIndices = {0, 1, 2, 3};
    s.points = {GfVec3f(-10, -10, z), GfVec3f(10, -10, z), GfVec3f(10, 10, z),
                GfVec3f(-10, 10, z)};
    s.restPoints = s.points;
    return s;
}

static UsdGenGraphDesc RelaxDesc(std::vector<Polyline> const &curves,
                                 std::vector<UsdGenParamValue> params)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/relax");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(Strands(SdfPath("/relax/hair"), curves));
    desc.surfaces.push_back(PlaneSurface(SdfPath("/relax/collider")));
    desc.nodes.push_back(SourceNode(SdfPath("/relax/source"), SdfPath("/relax/hair")));
    UsdGenNodeDesc collide = OpNode("/relax/op", "UsdGenCollide", SdfPath("/relax/source"), 7);
    params.push_back({TfToken("resolveType"), VtValue(TfToken("relax")), false});
    collide.params = params;
    collide.surfaces = {SdfPath("/relax/collider")};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/relax/op");
    return desc;
}

// Direct transcription of the push-out algorithm (separate A / Disp / S /
// Wk arrays, float like the reference) over the analytic plane z = 0.
static Polyline RefRelax(Polyline const &P, float rootClear, float tipClear, int iterations,
                         bool *unresolved, int forceWindow = 0)
{
    size_t const n = P.size();
    *unresolved = false;
    float const cR = std::max(1e-4f, rootClear), cT = std::max(1e-4f, tipClear);
    auto clear = [&](size_t k) { return cR + (cT - cR) * (float(k) / float(n - 1)); };
    bool any = false;
    for (size_t k = 1; k < n; ++k) any = any || P[k][2] < clear(k);
    if (!any) return P;
    Polyline const A = P;
    std::vector<float> seg(n, 0.0f);
    for (size_t k = 1; k < n; ++k) seg[k] = (A[k] - A[k - 1]).GetLength();
    std::vector<GfVec3f> disp(n, GfVec3f(0.0f)), S(n), W = A;
    float const r = float(std::min<size_t>(std::max<size_t>(n, 3), 1000)) /
                    float(std::min<size_t>(std::max<size_t>(n, 3), 100));
    size_t const w = forceWindow ? size_t(forceWindow) : size_t(std::max(1L, std::lroundf(r)));
    auto mul = [](GfVec3f const &v, float s) { return GfVec3f(v[0] * s, v[1] * s, v[2] * s); };
    for (int it = 0; it < iterations; ++it) {
        for (size_t k = 1; k < n; ++k) {
            // Plane z = 0: d = z; gradient +Z, formed as (p - q) * sign / |p - q|
            // like a closest-point gradient (the same float rounding).
            float const d = W[k][2];
            float const len = std::fabs(d);
            float const scale = 1.0f + std::max(std::fabs(W[k][0]),
                                                std::max(std::fabs(W[k][1]), len));
            float const gz = len > 1e-6f * scale ? d * ((d < 0 ? -1.0f : 1.0f) / len) : 1.0f;
            if (d < clear(k)) disp[k] += mul(GfVec3f(0, 0, gz), 0.8f * (clear(k) - d));
        }
        S[0] = disp[0];
        for (size_t k = 1; k + 1 < n; ++k)
            S[k] = mul(disp[k >= w ? k - w : 0], 0.25f) + mul(disp[k], 0.5f) +
                   mul(disp[std::min(n - 1, k + w)], 0.25f);
        S[n - 1] = mul(disp[n - 1], 0.5f) + mul(disp[n - 1 >= w ? n - 1 - w : 0], 0.5f);
        disp = S;
        for (size_t k = 1; k < n; ++k) W[k] = A[k] + disp[k];
        for (size_t k = 1; k < n; ++k) {
            GfVec3f const dir = W[k] - W[k - 1];
            float const len = dir.GetLength();
            GfVec3f const u = len > 1e-6f ? GfVec3f(dir[0] / len, dir[1] / len, dir[2] / len)
                                          : GfVec3f(0, 1, 0);
            W[k] = W[k - 1] + mul(u, seg[k]);
            disp[k] = W[k] - A[k];
        }
    }
    for (size_t k = 1; k < n; ++k) *unresolved = *unresolved || W[k][2] <= 1e-4f;
    return W;
}

static bool Same(Polyline const &a, Polyline const &b, double tol)
{
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k)
        for (int d = 0; d < 3; ++d)
            if (!Near(a[k][d], b[k][d], tol)) {
                std::printf("  cv %zu axis %d: %.7f vs %.7f\n", k, d, a[k][d], b[k][d]);
                return false;
            }
    return true;
}

// Root 0.12 above the plane, diving at 30 degrees below +Y: CVs 3..10 are
// inside (z < 0) — the tip 0.38 deep.
static Polyline DivingStrand(GfVec3f root, int cvs, float step)
{
    Polyline p;
    for (int i = 0; i < cvs; ++i)
        p.push_back(root + GfVec3f(0.0f, 0.866f * step * float(i), -0.5f * step * float(i)));
    return p;
}

static void CheckRelaxContract()
{
    std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenCollide"));
    if (!op) { Check(false, "Create(UsdGenCollide)"); return; }
    std::set<std::string> value;
    for (TfToken const &t : op->ValueParameters()) value.insert(t.GetString());
    Check(value.count("clearance:root") && value.count("clearance:tip") &&
              value.count("relax:iterations"),
          "Collide declares clearance:root/tip and relax:iterations as value parameters");
    Check(op->OutputPrimvars().empty(), "Collide emits no plane until configured for relax");
    UsdGenNodeDesc node;
    node.params = {{TfToken("resolveType"), VtValue(TfToken("relax")), false}};
    UsdGenParamView view;
    view.node = &node;
    op->Configure(view);
    Check(op->OutputPrimvars().size() == 1 &&
              op->OutputPrimvars()[0] == TfToken("collideUnresolved"),
          "Collide configured for relax emits collideUnresolved");
    node.params = {{TfToken("resolveType"), VtValue(TfToken("stiff")), false}};
    op->Configure(view);
    Check(op->OutputPrimvars().empty(), "Collide configured for stiff emits no plane");
}

static void CheckRelaxPushOut()
{
    constexpr int kCvs = 11;
    float const rootClear = 0.01f, tipClear = 0.03f;
    Polyline const diving = DivingStrand(GfVec3f(0.37f, 0.0f, 0.12f), kCvs, 0.08f);
    Polyline const above = Straight(GfVec3f(2.37f, 0, 0.5f), GfVec3f(0, 0.1f, 0), kCvs);
    std::vector<UsdGenParamValue> const params = {
        {TfToken("clearance:root"), VtValue(rootClear), false},
        {TfToken("clearance:tip"), VtValue(tipClear), false}};
    UsdGenCurveBuffer out, again;
    bool const ok = Cook(RelaxDesc({diving, above}, params), &out);
    Check(ok, "Collide relax cooks on the CPU lane");
    if (!ok || out.totalCvs != 2 * kCvs) return;
    Polyline const got = CurvePoints(out, FindCurve(out, diving[0], kCvs), kCvs);
    Polyline const clean = CurvePoints(out, FindCurve(out, above[0], kCvs), kCvs);
    if (got.empty() || clean.empty()) { Check(false, "Collide relax strands found by root"); return; }
    Check(got[0] == diving[0], "Collide relax keeps the root bit-for-bit");
    bool lengths = true;
    for (size_t k = 1; k < got.size(); ++k)
        lengths = lengths && Near((got[k] - got[k - 1]).GetLength(),
                                  (diving[k] - diving[k - 1]).GetLength(), 2e-6);
    Check(lengths, "Collide relax preserves every segment length");
    bool outside = true;
    double worst = 1e9;
    for (size_t k = 1; k < got.size(); ++k) {
        double const clear = rootClear + (tipClear - rootClear) * double(k) / (kCvs - 1);
        worst = std::min(worst, double(got[k][2]) - clear);
        outside = outside && got[k][2] > 0.0f;
    }
    std::printf("  relax (0.38 deep): min(z - clear) = %.6g, tip z = %.6f (clear %.3f)\n",
                worst, got.back()[2], tipClear);
    Check(outside, "Collide relax brings a 0.38-deep strand fully out of the collider");

    // A shallow dip (CVs 4..10 inside clear(k), the tip 0.035 deep) lands on
    // the clearance ramp: every penetrating interior CV ends at or above
    // clear(k) (below 2 clear(k): the rest-length chain lifts the first ones
    // a little higher); the tip, whose 0.5/0.5 tap converges slowest, at
    // >= 80% of clear(tip) after the 24 sweeps.
    Polyline dip;
    for (int i = 0; i < kCvs; ++i)
        dip.push_back(GfVec3f(0.37f, 0.1f * float(i), 0.05f - 0.0085f * float(i)));
    UsdGenCurveBuffer dipOut;
    bool const dipOk = Cook(RelaxDesc({dip}, params), &dipOut);
    Polyline const dipped = CurvePoints(dipOut, dipOk ? 0 : -1, kCvs);
    bool onRamp = !dipped.empty();
    double lo = 1e9, hi = -1e9;
    size_t penetrating = 0;
    for (size_t k = 1; onRamp && k < dipped.size(); ++k) {
        double const clear = rootClear + (tipClear - rootClear) * double(k) / (kCvs - 1);
        if (!(dip[k][2] < clear)) continue;
        ++penetrating;
        double const rel = double(dipped[k][2]) / clear;
        lo = std::min(lo, rel);
        hi = std::max(hi, rel);
        onRamp = rel >= (k + 1 == dipped.size() ? 0.8 : 1.0) && rel < 2.0;
    }
    std::printf("  relax (shallow): %zu penetrating CVs end at %.4f..%.4f x clear(k)\n",
                penetrating, lo, hi);
    bool dipRef = false;
    Check(onRamp && penetrating == 7 &&
              Same(dipped, RefRelax(dip, rootClear, tipClear, 24, &dipRef), 2e-5),
          "Collide relax lifts penetrating CVs onto their root/tip clearance");
    bool unresolved = false;
    Polyline const want = RefRelax(diving, rootClear, tipClear, 24, &unresolved);
    Check(Same(got, want, 2e-5), "Collide relax equals the push-out algorithm transcription");
    Check(!unresolved, "the transcription resolves the diving strand too");
    Check(clean == above, "Collide relax copies a strand clear of the collider bit-for-bit");
    UsdGenPlane const *flag = FindCurvePlane(out, "collideUnresolved");
    Check(flag && flag->type == TfToken("float") && flag->f.size() == 2 &&
              flag->f[0] == 0.0f && flag->f[1] == 0.0f,
          "Collide relax publishes collideUnresolved = 0 for resolved strands");
    Check(Cook(RelaxDesc({diving, above}, params), &again) && again.px == out.px &&
              again.py == out.py && again.pz == out.pz,
          "Collide relax cooks twice bit-identical");

    // Clearance lerps by point index: a uniform 0.03 clearance lifts the
    // CVs near the root higher than the 0.01..0.03 ramp does.
    UsdGenCurveBuffer flat;
    bool const flatOk = Cook(RelaxDesc({diving}, {{TfToken("clearance:root"), VtValue(0.03f), false},
                                                  {TfToken("clearance:tip"), VtValue(0.03f), false}}),
                             &flat);
    Polyline const uniform = CurvePoints(flat, flatOk ? 0 : -1, kCvs);
    bool uniformRef = false;
    Check(!uniform.empty() &&
              Same(uniform, RefRelax(diving, 0.03f, 0.03f, 24, &uniformRef), 2e-5) &&
              !Same(uniform, got, 1e-4),
          "Collide relax clearance follows the root/tip lerp");

    // relax:iterations is honoured (1 iteration differs from 24 and equals
    // the transcription's single iteration).
    UsdGenCurveBuffer once;
    std::vector<UsdGenParamValue> oneParams = params;
    oneParams.push_back({TfToken("relax:iterations"), VtValue(1), false});
    bool const onceOk = Cook(RelaxDesc({diving}, oneParams), &once);
    bool onceRef = false;
    Polyline const one = CurvePoints(once, onceOk ? 0 : -1, kCvs);
    Check(!one.empty() && Same(one, RefRelax(diving, rootClear, tipClear, 1, &onceRef), 2e-5) &&
              !Same(one, got, 1e-4),
          "Collide relax runs relax:iterations sweeps");

    // offset/pushAmount do not apply to relax.
    UsdGenCurveBuffer pushed;
    std::vector<UsdGenParamValue> withPush = params;
    withPush.push_back({TfToken("pushAmount"), VtValue(0.3f), false});
    withPush.push_back({TfToken("offset"), VtValue(0.7f), false});
    Check(Cook(RelaxDesc({diving, above}, withPush), &pushed) && pushed.px == out.px &&
              pushed.py == out.py && pushed.pz == out.pz,
          "Collide relax ignores offset/pushAmount");
}

static void CheckRelaxWindow()
{
    // 150 CVs: r = 150/100 = 1.5 -> window 2.
    constexpr int kCvs = 150;
    Polyline const diving = DivingStrand(GfVec3f(0.37f, 0.0f, 0.05f), kCvs, 0.004f);
    std::vector<UsdGenParamValue> const params = {
        {TfToken("clearance:root"), VtValue(0.002f), false},
        {TfToken("clearance:tip"), VtValue(0.01f), false}};
    UsdGenCurveBuffer out;
    bool const ok = Cook(RelaxDesc({diving}, params), &out);
    Polyline const got = CurvePoints(out, ok ? 0 : -1, kCvs);
    bool u2 = false, u1 = false;
    Polyline const w2 = RefRelax(diving, 0.002f, 0.01f, 24, &u2);
    Polyline const w1 = RefRelax(diving, 0.002f, 0.01f, 24, &u1, /*forceWindow*/ 1);
    auto maxDiff = [](Polyline const &a, Polyline const &b) {
        double m = a.size() == b.size() ? 0.0 : 1e9;
        for (size_t k = 0; k < a.size() && k < b.size(); ++k)
            for (int d = 0; d < 3; ++d) m = std::max(m, std::fabs(double(a[k][d]) - b[k][d]));
        return m;
    };
    double const d2 = got.empty() ? 1e9 : maxDiff(got, w2);
    double const d1 = got.empty() ? 1e9 : maxDiff(got, w1);
    std::printf("  window: max |engine - ref(w=2)| = %.3g, |engine - ref(w=1)| = %.3g\n", d2, d1);
    Check(d2 < 1e-5 && d1 > 1e-3, "Collide relax uses window 2 for 150 CVs");
    bool lengths = !got.empty();
    for (size_t k = 1; lengths && k < got.size(); ++k)
        lengths = Near((got[k] - got[k - 1]).GetLength(), (diving[k] - diving[k - 1]).GetLength(),
                       2e-6);
    Check(lengths, "Collide relax (150 CVs) preserves every segment length");
}

static void CheckRelaxUnresolvedMaskAndEdges()
{
    // Root 0.5 deep with a 0.2 strand: no rotation can bring it out.
    Polyline const buried = Straight(GfVec3f(0.37f, 0, -0.5f), GfVec3f(0, 0.1f, 0), 3);
    Polyline const fine = DivingStrand(GfVec3f(4.37f, 0.0f, 0.12f), 3, 0.08f);
    UsdGenCurveBuffer out;
    bool const ok = Cook(RelaxDesc({buried, fine}, {}), &out);
    UsdGenPlane const *flag = ok ? FindCurvePlane(out, "collideUnresolved") : nullptr;
    int const b = ok ? FindCurve(out, buried[0], 3) : -1;
    int const f = ok ? FindCurve(out, fine[0], 3) : -1;
    Check(flag && b >= 0 && f >= 0 && flag->f.size() == 2 && flag->f[size_t(b)] == 1.0f &&
              flag->f[size_t(f)] == 0.0f,
          "Collide relax flags the strand that stays inside (collideUnresolved = 1)");

    // Mask envelope: 0 copies, 0.5 lerps halfway per CV.
    Polyline const diving = DivingStrand(GfVec3f(0.37f, 0.0f, 0.12f), 11, 0.08f);
    UsdGenCurveBuffer full, half, none;
    bool const fullOk = Cook(RelaxDesc({diving}, {}), &full);
    bool const halfOk = Cook(RelaxDesc({diving}, {{TfToken("mask"), VtValue(0.5f), false}}), &half);
    bool const noneOk = Cook(RelaxDesc({diving}, {{TfToken("mask"), VtValue(0.0f), false}}), &none);
    Polyline const R = CurvePoints(full, fullOk ? 0 : -1, 11);
    Polyline const H = CurvePoints(half, halfOk ? 0 : -1, 11);
    Polyline const Z = CurvePoints(none, noneOk ? 0 : -1, 11);
    bool lerp = !R.empty() && !H.empty();
    for (size_t k = 0; lerp && k < R.size(); ++k)
        for (int d = 0; d < 3; ++d)
            lerp = lerp && Near(H[k][d], diving[k][d] + (R[k][d] - diving[k][d]) * 0.5, 1e-6);
    Check(lerp, "Collide relax mask 0.5 lerps halfway to the relaxed strand");
    Check(Z == diving, "Collide relax mask 0 copies the input bit-for-bit");

    // A distant second collider (skin first) changes nothing.
    UsdGenGraphDesc two = RelaxDesc({diving}, {});
    two.surfaces.push_back(PlaneSurface(SdfPath("/relax/skin"), -50.0f));
    two.nodes.back().surfaces = {SdfPath("/relax/skin"), SdfPath("/relax/collider")};
    UsdGenCurveBuffer twoOut;
    Check(Cook(two, &twoOut) && twoOut.px == full.px && twoOut.py == full.py &&
              twoOut.pz == full.pz,
          "Collide relax takes the union of colliders (a distant skin changes nothing)");

    UsdGenCurveBuffer bad;
    Check(!Cook(RelaxDesc({diving}, {{TfToken("relax:iterations"), VtValue(0), false}}), &bad, true),
          "Collide relax with relax:iterations 0 fails closed");
    Check(!Cook(RelaxDesc({diving}, {{TfToken("clearance:tip"),
                                      VtValue(std::numeric_limits<double>::infinity()), false}}),
                &bad, true),
          "Collide relax with an infinite clearance fails closed");
}

// ===========================================================================
// (3) Collide flexible: exact index == brute-force scan
// ===========================================================================

// Verbatim copy of the scan's per-triangle routine (Ericson 5.1.5).
static bool ScanClosestOnTriangle(GfVec3d const &p, GfVec3d const &a, GfVec3d const &b,
                                  GfVec3d const &c, GfVec3d *closest, double *dist2)
{
    GfVec3d const n = GfCross(b - a, c - a);
    double const len = n.GetLength();
    if (!(len > 0.0) || !std::isfinite(len)) return false;
    GfVec3d const ab = b - a, ac = c - a, ap = p - a;
    double const d1 = GfDot(ab, ap), d2 = GfDot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        *closest = a;
    } else {
        GfVec3d const bp = p - b;
        double const d3 = GfDot(ab, bp), d4 = GfDot(ac, bp);
        if (d3 >= 0.0 && d4 <= d3) {
            *closest = b;
        } else {
            double const vc = d1 * d4 - d3 * d2;
            if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
                *closest = a + ab * (d1 / (d1 - d3));
            } else {
                GfVec3d const cp = p - c;
                double const d5 = GfDot(ab, cp), d6 = GfDot(ac, cp);
                if (d6 >= 0.0 && d5 <= d6) {
                    *closest = c;
                } else {
                    double const vb = d5 * d2 - d1 * d6;
                    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
                        *closest = a + ac * (d2 / (d2 - d6));
                    } else {
                        double const va = d3 * d6 - d5 * d4;
                        double const d43 = d4 - d3, d56 = d5 - d6;
                        if (va <= 0.0 && d43 >= 0.0 && d56 >= 0.0) {
                            *closest = b + (c - b) * (d43 / (d43 + d56));
                        } else {
                            double const denom = 1.0 / (va + vb + vc);
                            *closest = a + ab * (vb * denom) + ac * (vc * denom);
                        }
                    }
                }
            }
        }
    }
    GfVec3d const d = p - *closest;
    *dist2 = GfDot(d, d);
    return true;
}

// The former flexible kernel: linear first-win scan over fan triangles.
static Polyline ScanFlexible(Polyline P, UsdGenSurfaceDesc const &mesh, double off,
                             double push, int iterations)
{
    std::vector<GfVec3d> tri;
    size_t o = 0;
    for (int corners : mesh.faceVertexCounts) {
        for (int k = 1; k + 1 < corners; ++k) {
            tri.push_back(GfVec3d(mesh.points[size_t(mesh.faceVertexIndices[o])]));
            tri.push_back(GfVec3d(mesh.points[size_t(mesh.faceVertexIndices[o + size_t(k)])]));
            tri.push_back(GfVec3d(mesh.points[size_t(mesh.faceVertexIndices[o + size_t(k) + 1])]));
        }
        o += size_t(corners);
    }
    for (int it = 0; it < iterations; ++it) {
        for (GfVec3f &cv : P) {
            GfVec3d const pos(cv);
            bool found = false;
            double best = 0.0;
            GfVec3d q, a, b, c;
            for (size_t t = 0; t < tri.size(); t += 3) {
                GfVec3d cq;
                double d2 = 0.0;
                if (!ScanClosestOnTriangle(pos, tri[t], tri[t + 1], tri[t + 2], &cq, &d2)) continue;
                if (!(d2 >= 0.0)) continue;
                if (!found || d2 < best) {
                    found = true; best = d2; q = cq;
                    a = tri[t]; b = tri[t + 1]; c = tri[t + 2];
                }
            }
            if (!found) continue;
            double const pen = off - std::sqrt(best);
            if (!(pen > 0.0)) continue;
            double const s = pen * push;
            GfVec3d dir;
            if (best == 0.0) {
                GfVec3d const nrm = GfCross(b - a, c - a);
                dir = nrm / nrm.GetLength();
            } else {
                dir = (pos - q) / std::sqrt(best);
            }
            GfVec3d const moved = pos + dir * s;
            cv = GfVec3f(float(moved[0]), float(moved[1]), float(moved[2]));
        }
    }
    return P;
}

static void CheckFlexibleMatchesScan()
{
    // 16 x 16 bumpy quads (4-corner faces: shared edges and vertices give
    // exact distance ties), strands crossing exactly over grid vertices and
    // between them.
    UsdGenSurfaceDesc mesh;
    mesh.path = SdfPath("/scan/collider");
    constexpr int kGrid = 16;
    for (int j = 0; j <= kGrid; ++j)
        for (int i = 0; i <= kGrid; ++i)
            mesh.points.push_back(GfVec3f(-2.0f + 0.25f * float(i), -2.0f + 0.25f * float(j),
                                          0.05f * float((i * 7 + j * 3) % 5) - 0.1f));
    for (int j = 0; j < kGrid; ++j)
        for (int i = 0; i < kGrid; ++i) {
            int const v = j * (kGrid + 1) + i;
            mesh.faceVertexCounts.push_back(4);
            for (int idx : {v, v + 1, v + kGrid + 2, v + kGrid + 1})
                mesh.faceVertexIndices.push_back(idx);
        }
    mesh.restPoints = mesh.points;
    std::vector<Polyline> strands;
    for (int i = 1; i < kGrid; i += 2)
        strands.push_back(Straight(GfVec3f(-2.0f + 0.25f * float(i), -1.75f, 0.02f),
                                   GfVec3f(0, 0.25f, 0), 14));
    for (int i = 0; i < 6; ++i)
        strands.push_back(Straight(GfVec3f(-1.9f + 0.61f * float(i), -1.8f, -0.03f + 0.02f * float(i)),
                                   GfVec3f(0.01f, 0.23f, 0.004f), 14));
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/scan");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(Strands(SdfPath("/scan/hair"), strands));
    desc.surfaces.push_back(mesh);
    desc.nodes.push_back(SourceNode(SdfPath("/scan/source"), SdfPath("/scan/hair")));
    UsdGenNodeDesc collide = OpNode("/scan/op", "UsdGenCollide", SdfPath("/scan/source"), 7);
    collide.params = {{TfToken("offset"), VtValue(0.3f), false},
                      {TfToken("pushAmount"), VtValue(0.7f), false},
                      {TfToken("iterations"), VtValue(3), false}};
    collide.surfaces = {mesh.path};
    desc.nodes.push_back(collide);
    desc.terminal = SdfPath("/scan/op");
    UsdGenCurveBuffer out;
    bool const ok = Cook(desc, &out);
    Check(ok, "Collide flexible over a bumpy grid cooks");
    if (!ok) return;
    bool exact = true;
    size_t moved = 0;
    for (size_t i = 0; i < strands.size(); ++i) {
        Polyline const &s = strands[i];
        Polyline const got = CurvePoints(out, FindCurveById(out, uint64_t(i)), 14);
        Polyline const want = ScanFlexible(s, mesh, double(0.3f), double(0.7f), 3);
        exact = exact && got == want;
        for (size_t k = 0; k < s.size(); ++k) moved += got.size() == s.size() && got[k] != s[k];
    }
    std::printf("  flexible scan comparison: %zu CVs pushed\n", moved);
    Check(exact && moved > 100,
          "Collide flexible through the exact BVH is bit-identical to the brute-force scan");
}

int main()
{
    usdGenRegisterM1Operators();
    CheckTieRegistry();
    CheckTiePull();
    CheckTieAmountSpread();
    CheckTieArcGate();
    CheckTieMultiEntryAndSequence();
    CheckTieFailClosed();
    CheckRelaxContract();
    CheckRelaxPushOut();
    CheckRelaxWindow();
    CheckRelaxUnresolvedMaskAndEdges();
    CheckFlexibleMatchesScan();
    std::printf("testUsdGenTieCollide: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
