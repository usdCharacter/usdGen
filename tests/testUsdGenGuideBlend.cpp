// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenGuideBlend: UsdGenGuideInterpolate's imported-groom weightings
// (usdGen:weighting = nearest / inverseDistance / gaussian), bend-follow
// frames, parting (rotation about the root and guide suppression), the
// emitted planes (guideIndex/guideWeight, primaryGuide, nearestGuide,
// partId, mirrorSign, pass-through strandIndex/strandSeed) and determinism.
//
// Graphs are hand-built: a CurveSource of short root strands (roots on the
// XZ plane, root normal +Y) -> GuideInterpolate with a reference guide set.
// Expected values are computed by hand from the reference rules.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/ops/guideBlend.h"
#include "usdGen/scheduler.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace usdGen;

static int failures = 0;
static void Check(bool v, std::string const &s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s.c_str()); }
    else std::printf("ok: %s\n", s.c_str());
}
static bool Near(double a, double b, double tol = 2e-5) { return std::fabs(a - b) <= tol; }
static bool NearV(GfVec3f const &a, GfVec3d const &b, double tol = 2e-5)
{
    return Near(a[0], b[0], tol) && Near(a[1], b[1], tol) && Near(a[2], b[2], tol);
}
static std::string Str(GfVec3f const &v)
{
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(%.6f, %.6f, %.6f)", v[0], v[1], v[2]);
    return buf;
}

// --- fixture ------------------------------------------------------------------

using Polyline = std::vector<GfVec3f>;

struct Fixture
{
    std::vector<GfVec3f> roots;
    std::vector<Polyline> guides;
    std::vector<int> mirrorSign;          // guide set plane (optional)
    std::vector<int> strandIndex, strandSeed;   // root planes (optional)
    std::vector<std::pair<char const *, VtValue>> params;
    int cvCount = 5;
    bool withGuides = true;
};

static UsdGenGraphDesc Build(Fixture const &f)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;

    // Roots: 2-CV strands along +Y with a root frame whose normal is +Y
    // (rows tangent, binormal, normal; tangent x binormal = normal).
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/groom/roots");
    hair.role = UsdGenRole::Curves;
    hair.curveRole = TfToken("hair");
    size_t const R = f.roots.size();
    hair.curveVertexCounts.assign(R, 2);
    for (GfVec3f const &r : f.roots) {
        hair.points.push_back(r);
        hair.points.push_back(r + GfVec3f(0.0f, 0.01f, 0.0f));
    }
    hair.rest = hair.points;
    hair.widths.assign(2 * R, 0.02f);
    hair.curveId.resize(R);
    hair.skinPrim.resize(R);
    for (size_t c = 0; c < R; ++c) {
        hair.curveId[c] = uint64_t(c);
        hair.skinPrim[c] = int(c);
    }
    hair.skinPrimUv.assign(R, GfVec2f(0.2f, 0.3f));
    for (GfVec3f const &r : f.roots) {
        GfMatrix4d frame(1.0);
        frame.SetRow(0, GfVec4d(1, 0, 0, 0));
        frame.SetRow(1, GfVec4d(0, 0, -1, 0));
        frame.SetRow(2, GfVec4d(0, 1, 0, 0));
        frame.SetRow(3, GfVec4d(r[0], r[1], r[2], 1));
        hair.rootFrame.push_back(frame);
    }
    auto addPlane = [&](char const *name, std::vector<int> const &values) {
        UsdGenAuthoredPlaneDesc plane;
        plane.name = TfToken(name);
        plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.domain = UsdGenAuthoredPlaneDomain::Primitive;
        plane.arity = 1;
        plane.intValues = VtIntArray(values.begin(), values.end());
        return plane;
    };
    if (!f.strandIndex.empty()) hair.authoredPlanes.push_back(addPlane("strandIndex", f.strandIndex));
    if (!f.strandSeed.empty()) hair.authoredPlanes.push_back(addPlane("strandSeed", f.strandSeed));
    desc.curveSets.push_back(hair);

    UsdGenNodeDesc source;
    source.path = SdfPath("/groom/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                     {TfToken("useRest"), VtValue(true), false}};
    desc.nodes.push_back(source);

    UsdGenNodeDesc op;
    op.path = SdfPath("/groom/interp");
    op.type = TfToken("UsdGenGuideInterpolate");
    op.seed = 3;
    op.inputs = {source.path};
    op.params.push_back({TfToken("cvCount"), VtValue(f.cvCount), false});
    for (auto const &p : f.params) op.params.push_back({TfToken(p.first), p.second, false});

    if (f.withGuides) {
        UsdGenCurveSetDesc guides;
        guides.path = SdfPath("/groom/guides");
        guides.role = UsdGenRole::Reference;
        guides.curveRole = TfToken("guide");
        for (Polyline const &g : f.guides) {
            guides.curveVertexCounts.push_back(int(g.size()));
            guides.points.insert(guides.points.end(), g.begin(), g.end());
        }
        guides.rest = guides.points;
        if (!f.mirrorSign.empty())
            guides.authoredPlanes.push_back(addPlane("mirrorSign", f.mirrorSign));
        desc.curveSets.push_back(guides);
        op.references = {guides.path};
    }
    desc.nodes.push_back(op);
    desc.terminal = op.path;
    return desc;
}

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out, bool quiet = false)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        if (!quiet)
            for (auto const &e : compiled.errors) std::printf("  compile: %s\n", e.c_str());
        return false;
    }
    UsdGenScheduler scheduler(4);
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

static bool Cook(Fixture const &f, UsdGenCurveBuffer *out) { return Cook(Build(f), out); }

static GfVec3f Point(UsdGenCurveBuffer const &b, size_t c, size_t i, size_t n)
{
    size_t const o = c * n + i;
    return GfVec3f(b.px[o], b.py[o], b.pz[o]);
}

static UsdGenPlane const *Plane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCurve)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

static std::vector<int> Ints(UsdGenCurveBuffer const &b, char const *name, uint8_t arity = 1)
{
    UsdGenPlane const *p = Plane(b, name);
    if (!p || p->type != TfToken("int") || p->arity != arity ||
        p->interpolation != TfToken("uniform") || p->i.size() != size_t(b.totalCurves) * arity)
        return {};
    return std::vector<int>(p->i.begin(), p->i.end());
}

static std::vector<float> Floats(UsdGenCurveBuffer const &b, char const *name, uint8_t arity)
{
    UsdGenPlane const *p = Plane(b, name);
    if (!p || p->type != TfToken("float") || p->arity != arity ||
        p->f.size() != size_t(b.totalCurves) * arity)
        return {};
    return std::vector<float>(p->f.begin(), p->f.end());
}

static Polyline Line(GfVec3f const &root, GfVec3f const &dir, float length)
{
    return {root, root + dir * length};
}

// --- helper units ---------------------------------------------------------------

static void CheckHelpers()
{
    using namespace guideBlend;
    std::vector<V3> poly = {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}};
    V3 a = SampleRaw(poly.data(), poly.size(), 1.5f);
    Check(a.x == 0.5f && a.y == 1.0f && a.z == 0.0f, "raw polyline sample at arc length 1.5");
    V3 b = SampleRaw(poly.data(), poly.size(), 2.5f);
    Check(Near(b.x, 1.5) && b.y == 1.0f, "raw polyline sample extrapolates past the tip");
    Check(PolylineLength(poly.data(), poly.size()) == 2.0f, "polyline length");

    Check(Delaunay2D({0, 0, 1, 0, 0, 1}).size() == 3, "Delaunay of 3 points: 1 triangle");
    Check(Delaunay2D({0, 0, 1, 0, 0, 1, 1, 1}).size() == 6, "Delaunay of a square: 2 triangles");
    Check(Delaunay2D({0, 0, 1, 0, 2, 0}).empty(), "collinear points: no triangle");
    // 4 points with one inside: 3 triangles fanning to the inner point.
    std::vector<int> const inner = Delaunay2D({0, 0, 4, 0, 0, 4, 1, 1});
    int innerUses = 0;
    for (int v : inner) innerUses += v == 3;
    Check(inner.size() == 9 && innerUses == 3, "Delaunay with an inner point: 3 triangles");

    Parting parting;
    parting.Set({{0, 0, -5}, {0, 0, 5}}, {2}, 1.0f, 45.0f, 0.1f, 1.0f);
    Check(parting.active && parting.lengths.size() == 1 && parting.lengths[0] == 10.0f,
          "parting curve length and activation");
    PartInfo const right = QueryParting(parting, {0.5f, 0, 0}, {0, 1, 0});
    Check(right.valid && right.curve == 0 && right.side == -1.0f && right.offLine == 1.0f &&
              Near(right.weight, 0.5, 1e-6) && Near(right.away.x, 1.0, 1e-6),
          "parting query: side, away direction and smoothstep weight (R * curve length)");
    Check(PartKey(right) == 1, "parting key curve*2 + (side <= 0)");
    PartInfo const left = QueryParting(parting, {-0.5f, 0, 0}, {0, 1, 0});
    Check(left.side == 1.0f && PartKey(left) == 0 && Near(left.away.x, -1.0, 1e-6),
          "parting query on the other side");
    Parting off;
    off.Set({{0, 0, -5}, {0, 0, 5}}, {2}, 0.0f, 45.0f, 0.1f, 1.0f);
    Check(!off.active && !QueryParting(off, {0.5f, 0, 0}, {0, 1, 0}).valid,
          "parting inactive at amount 0");
}

// --- nearest ----------------------------------------------------------------------

static void CheckNearest()
{
    Fixture f;
    f.guides = {{GfVec3f(0, 0, 0), GfVec3f(0, 0.5f, 0), GfVec3f(0, 1, 0)},
                {GfVec3f(2, 0, 0), GfVec3f(2, 1, 0), GfVec3f(3, 1, 0)}};
    f.mirrorSign = {1, -1};
    f.roots = {GfVec3f(0.3f, 0, 0.1f), GfVec3f(1.8f, 0, 0), GfVec3f(2, 0, 0)};
    f.strandIndex = {7, 3, 5};
    f.strandSeed = {11, 13, 17};
    f.params = {{"weighting", VtValue(TfToken("nearest"))}};
    UsdGenCurveBuffer b;
    if (!Cook(f, &b)) { Check(false, "nearest: cook"); return; }
    size_t const n = 5;
    Check(b.totalCurves == 3 && b.totalCvs == 15, "nearest: 3 strands of cvCount 5");
    bool straight = true;
    for (size_t i = 0; i < n; ++i)
        straight = straight && NearV(Point(b, 0, i, n), GfVec3d(0.3, 0.25 * double(i), 0.1));
    Check(straight, "nearest: rigid copy of the nearest guide at the root offset");
    // Strand 1 follows guide 1 (length 2) resampled to ng = 8 uniform
    // samples: at s = 1 (the corner) the linear table cuts the corner
    // halfway between the samples at s = 6/7 and s = 8/7.
    GfVec3d const corner(2.0 + 1.0 / 14.0 - 0.2, 0.5 * (12.0 / 14.0 + 1.0), 0.0);
    Check(NearV(Point(b, 1, 0, n), GfVec3d(1.8, 0, 0)) &&
              NearV(Point(b, 1, 1, n), GfVec3d(1.8, 0.5, 0)) &&
              NearV(Point(b, 1, 2, n), corner) &&
              NearV(Point(b, 1, 3, n), GfVec3d(2.3, 1, 0)) &&
              NearV(Point(b, 1, 4, n), GfVec3d(2.8, 1, 0)),
          "nearest: guide sampled from its uniform arc-length table (corner " +
              Str(Point(b, 1, 2, n)) + ")");
    // Strand 2 sits on guide 1's root: the guide's own strand copies the raw
    // polyline (the corner point is exact).
    Check(NearV(Point(b, 2, 2, n), GfVec3d(2, 1, 0), 1e-6) &&
              NearV(Point(b, 2, 3, n), GfVec3d(2.5, 1, 0), 1e-6),
          "nearest: a root on a guide root copies the raw guide " + Str(Point(b, 2, 2, n)));

    std::vector<int> const index = Ints(b, "guideIndex", 3);
    std::vector<float> const weight = Floats(b, "guideWeight", 3);
    Check(index == std::vector<int>({0, -1, -1, 1, -1, -1, 1, -1, -1}) &&
              weight.size() == 9 && weight[0] == 1.0f && weight[3] == 1.0f && weight[1] == 0.0f,
          "nearest: guideIndex/guideWeight name the single guide");
    Check(Ints(b, "primaryGuide") == std::vector<int>({0, 1, 1}), "nearest: primaryGuide");
    Check(Ints(b, "nearestGuide") == std::vector<int>({0, 1, 1}), "nearest: nearestGuide");
    Check(Ints(b, "partId") == std::vector<int>({-1, -1, -1}), "nearest: partId -1 without parting");
    Check(Ints(b, "mirrorSign") == std::vector<int>({1, -1, -1}),
          "nearest: mirrorSign forwarded from the primary guide");
    Check(Ints(b, "strandIndex") == f.strandIndex && Ints(b, "strandSeed") == f.strandSeed,
          "upstream strandIndex/strandSeed pass through the generator");
    bool hairT = b.hairT.size() == 15;
    for (size_t c = 0; c < 3 && hairT; ++c)
        for (size_t i = 0; i < n; ++i) hairT = hairT && b.hairT[c * n + i] == float(i) / 4.0f;
    Check(hairT, "nearest: hairT root to tip");
}

// --- inverse distance -----------------------------------------------------------------

static void CheckInverseDistance()
{
    GfVec3f const dirA(0, 1, 0), dirB(0, 0.6f, 0.8f), dirC(0.6f, 0.8f, 0);
    Fixture f;
    f.guides = {Line(GfVec3f(0, 0, 0), dirA, 1.0f), Line(GfVec3f(1, 0, 0), dirB, 2.0f),
                Line(GfVec3f(0, 0, 1), dirC, 1.5f), Line(GfVec3f(5, 0, 5), dirA, 1.0f)};
    f.roots = {GfVec3f(0.2f, 0, 0.1f)};
    f.params = {{"weighting", VtValue(TfToken("inverseDistance"))}};
    UsdGenCurveBuffer b;
    if (!Cook(f, &b)) { Check(false, "inverseDistance: cook"); return; }
    double const d[3] = {std::sqrt(0.05), std::sqrt(0.65), std::sqrt(0.85)};
    double w[3], sum = 0.0;
    for (int k = 0; k < 3; ++k) { w[k] = 1.0 / (d[k] + 1e-6); sum += w[k]; }
    for (double &v : w) v /= sum;
    double const L = w[0] * 1.0 + w[1] * 2.0 + w[2] * 1.5;
    GfVec3d dir = GfVec3d(dirA) * w[0] + GfVec3d(dirB) * w[1] + GfVec3d(dirC) * w[2];
    std::vector<int> const index = Ints(b, "guideIndex", 3);
    std::vector<float> const weight = Floats(b, "guideWeight", 3);
    Check(index == std::vector<int>({0, 1, 2}), "inverseDistance: the 3 nearest guides");
    Check(weight.size() == 3 && Near(weight[0], w[0], 1e-6) && Near(weight[1], w[1], 1e-6) &&
              Near(weight[2], w[2], 1e-6),
          "inverseDistance: weights 1/(d + 1e-6), normalised");
    // Every guide sampled at the same absolute arc length t * L (L the
    // blended length; the shorter guide extrapolates straight).
    bool ok = true;
    for (size_t i = 0; i < 5; ++i)
        ok = ok && NearV(Point(b, 0, i, 5), GfVec3d(0.2, 0, 0.1) + dir * (L * double(i) / 4.0));
    Check(ok, "inverseDistance: affine blend of guide shapes at the blended length (tip " +
                  Str(Point(b, 0, 4, 5)) + ")");
    Check(Ints(b, "primaryGuide") == std::vector<int>({0}), "inverseDistance: primary = nearest");
}

// --- gaussian -------------------------------------------------------------------------------

static void CheckGaussian()
{
    GfVec3f const dirA(0, 1, 0), dirB(0, 0.6f, 0.8f), dirC(0.6f, 0.8f, 0);
    Fixture f;
    f.guides = {Line(GfVec3f(0, 0, 0), dirA, 1.0f), Line(GfVec3f(1, 0, 0), dirB, 2.0f),
                Line(GfVec3f(0, 0, 1), dirC, 1.5f)};
    f.roots = {GfVec3f(0.3f, 0, 0.2f)};
    f.params = {{"weighting", VtValue(TfToken("gaussian"))}, {"tightness", VtValue(3.0f)}};
    UsdGenCurveBuffer b;
    if (!Cook(f, &b)) { Check(false, "gaussian: cook"); return; }
    // sigma = 0.5 * mean edge of the one Delaunay triangle (1, 1, sqrt 2).
    double const sigma = 0.5 * (2.0 + std::sqrt(2.0)) / 3.0;
    double const sigma2 = sigma * sigma;
    double const d2[3] = {0.09 + 0.04, 0.49 + 0.04, 0.09 + 0.64};
    double const L[3] = {1.0, 2.0, 1.5};
    GfVec3d const dirs[3] = {GfVec3d(dirA), GfVec3d(dirB), GfVec3d(dirC)};
    double w[3], sum = 0.0;
    for (int k = 0; k < 3; ++k) { w[k] = std::pow(std::exp(-d2[k] / sigma2), 3.0); sum += w[k]; }
    GfVec3d tip(0.3, 0, 0.2);
    for (int k = 0; k < 3; ++k) tip += dirs[k] * (w[k] / sum * L[k]);
    std::vector<int> const index = Ints(b, "guideIndex", 3);
    std::vector<float> const weight = Floats(b, "guideWeight", 3);
    Check(index == std::vector<int>({0, 1, 2}) && weight.size() == 3 &&
              Near(weight[0], w[0] / sum, 1e-5) && Near(weight[1], w[1] / sum, 1e-5) &&
              Near(weight[2], w[2] / sum, 1e-5),
          "gaussian: exp(-d^2/sigma^2)^tightness over all guides, sigma from the Delaunay "
          "triangulation in the PCA plane");
    // Each guide is sampled at its own fraction t * L_g.
    Check(NearV(Point(b, 0, 4, 5), tip) &&
              NearV(Point(b, 0, 2, 5), (tip + GfVec3d(0.3, 0, 0.2)) * 0.5),
          "gaussian: blended shapes at each guide's own length (tip " + Str(Point(b, 0, 4, 5)) +
              ")");
    Check(Ints(b, "primaryGuide") == std::vector<int>({0}), "gaussian: primary = argmax weight");

    // tightness clamps to [0.25, 8].
    f.params = {{"weighting", VtValue(TfToken("gaussian"))}, {"tightness", VtValue(100.0f)}};
    UsdGenCurveBuffer b8;
    if (!Cook(f, &b8)) { Check(false, "gaussian: cook tightness 100"); return; }
    double w8[3], sum8 = 0.0;
    for (int k = 0; k < 3; ++k) { w8[k] = std::pow(std::exp(-d2[k] / sigma2), 8.0); sum8 += w8[k]; }
    std::vector<float> const weight8 = Floats(b8, "guideWeight", 3);
    Check(weight8.size() == 3 && Near(weight8[0], w8[0] / sum8, 1e-5) &&
              Near(weight8[1], w8[1] / sum8, 1e-5),
          "gaussian: tightness clamps to 8");
}

// --- bend follow ------------------------------------------------------------------------------

static void CheckBendFollow()
{
    Fixture f;
    f.guides = {{GfVec3f(0, 0, 0), GfVec3f(0, 1, 0), GfVec3f(1, 1, 0)}};
    f.roots = {GfVec3f(-0.1f, 0, 0)};
    f.params = {{"weighting", VtValue(TfToken("nearest"))}};
    UsdGenCurveBuffer rigid, follow;
    if (!Cook(f, &rigid)) { Check(false, "bendFollow: cook off"); return; }
    f.params.push_back({"bendFollow", VtValue(true)});
    if (!Cook(f, &follow)) { Check(false, "bendFollow: cook on"); return; }
    Check(NearV(Point(rigid, 0, 4, 5), GfVec3d(0.9, 1, 0)),
          "bendFollow off: the root offset is a rigid translation");
    // The offset (-0.1, 0, 0) is -0.1 along the start binormal (-1, 0, 0);
    // after the bend the binormal is (0, 1, 0).
    Check(NearV(Point(follow, 0, 0, 5), GfVec3d(-0.1, 0, 0)) &&
              NearV(Point(follow, 0, 4, 5), GfVec3d(1, 1.1, 0)),
          "bendFollow on: the offset turns with the guide's rotation-minimising frame (tip " +
              Str(Point(follow, 0, 4, 5)) + ")");
    // Frames come from Catmull-Rom tangents of the raw guide, which already
    // lean toward the corner on the first segment: the offset turns but keeps
    // its length, in the bend plane.
    GfVec3f const mid = Point(follow, 0, 1, 5);
    double const offset = (GfVec3d(mid) - GfVec3d(0, 0.5, 0)).GetLength();
    Check(Near(offset, 0.1, 1e-5) && Near(mid[2], 0.0, 1e-7) && mid[1] < 0.5f,
          "bendFollow on: the carried offset keeps its length " + Str(mid));
}

// --- parting ------------------------------------------------------------------------------------

static Fixture PartingFixture(char const *weighting)
{
    Fixture f;
    GfVec3f const up(0, 1, 0);
    // g0 across the parting (closest by distance), g1..g3 on the root's side.
    f.guides = {Line(GfVec3f(-0.2f, 0, 0), up, 1.0f), Line(GfVec3f(1.5f, 0, 0), up, 1.0f),
                Line(GfVec3f(0.5f, 0, 1), up, 1.0f), Line(GfVec3f(0.5f, 0, -1.2f), up, 1.0f)};
    f.roots = {GfVec3f(0.5f, 0, 0), GfVec3f(-0.5f, 0, 0), GfVec3f(3.0f, 0, 0)};
    VtVec3fArray const points{GfVec3f(0, 0, -5), GfVec3f(0, 0, 5)};
    f.params = {{"weighting", VtValue(TfToken(weighting))},
                {"part:points", VtValue(points)},
                {"part:counts", VtValue(VtIntArray{2})},
                {"part:amount", VtValue(1.0f)},
                {"part:angle", VtValue(45.0f)},
                {"part:radius", VtValue(0.1f)},
                {"part:distance", VtValue(1.0f)}};
    return f;
}

static void CheckParting()
{
    double const half = 0.5 * std::atan(1.0);   // theta = 45 deg, weight 0.5
    {
        Fixture f = PartingFixture("nearest");
        UsdGenCurveBuffer b;
        if (!Cook(f, &b)) { Check(false, "parting nearest: cook"); return; }
        Check(Ints(b, "partId") == std::vector<int>({1, 0, -1}),
              "parting: partId = curve*2 + (side <= 0), -1 outside the radius");
        // Root (0.5,0,0): g0 (distance 0.7) is across the parting and
        // skipped; g2 (distance 1, both inside the parting radius on the same
        // side) is favoured by the factor 1 + 2 w = 2 over g1 (distance 1).
        std::vector<int> const nearest = Ints(b, "nearestGuide");
        Check(nearest.size() == 3 && nearest[0] == 2 && nearest[1] == 0 && nearest[2] == 1,
              "parting: the nearest guide across the parting is suppressed");
        Check(Ints(b, "primaryGuide").size() == 3 && Ints(b, "primaryGuide")[0] == 2,
              "parting: the strand follows a guide on its own side");
        // Rigid rotation (distance 1) of the straight strand about the root,
        // by theta * weight, away from the parting (+x).
        bool ok = true;
        for (size_t i = 0; i < 5; ++i) {
            double const t = double(i) / 4.0;
            ok = ok && NearV(Point(b, 0, i, 5),
                             GfVec3d(0.5 + t * std::sin(half), t * std::cos(half), 0));
        }
        Check(ok, "parting: strand rotated about its root by angle * weight (tip " +
                      Str(Point(b, 0, 4, 5)) + ")");
        Check(NearV(Point(b, 2, 4, 5), GfVec3d(3.0 - 1.5 + 1.5, 1, 0)),
              "parting: no bend outside the parting radius");
    }
    {
        // partingDistance 0: the rotation fades to 0 at the tip.
        Fixture f = PartingFixture("nearest");
        f.params.back().second = VtValue(0.0f);
        UsdGenCurveBuffer b;
        if (!Cook(f, &b)) { Check(false, "parting distance 0: cook"); return; }
        double const a = half * (1.0 - 0.5);   // at t = 0.5
        Check(NearV(Point(b, 0, 2, 5), GfVec3d(0.5 + 0.5 * std::sin(a), 0.5 * std::cos(a), 0)) &&
                  NearV(Point(b, 0, 4, 5), GfVec3d(0.5, 1, 0)),
              "parting distance 0: the bend fades linearly to the tip");
    }
    {
        // inverseDistance: guides scaled by the parting factor
        // (1 + 2 w dot(away_root, away_guide)) and bent per guide.
        Fixture f = PartingFixture("inverseDistance");
        UsdGenCurveBuffer b;
        if (!Cook(f, &b)) { Check(false, "parting inverseDistance: cook"); return; }
        std::vector<int> const index = Ints(b, "guideIndex", 3);
        std::vector<float> const weight = Floats(b, "guideWeight", 3);
        double const w[3] = {1.0 / (0.5 + 1e-6), 1.0 / (0.6 + 1e-6), 1.0 / (1.0 + 1e-6)};
        double const sum = w[0] + w[1] + w[2];
        Check(index.size() == 9 && index[0] == 2 && index[1] == 3 && index[2] == 1 &&
                  weight.size() == 9 && Near(weight[0], w[0] / sum, 1e-5) &&
                  Near(weight[1], w[1] / sum, 1e-5) && Near(weight[2], w[2] / sum, 1e-5),
              "parting inverseDistance: same-side guides favoured, across-guide skipped");
        Check(NearV(Point(b, 0, 4, 5), GfVec3d(0.5 + std::sin(half), std::cos(half), 0)),
              "parting inverseDistance: each guide bent about the root (tip " +
                  Str(Point(b, 0, 4, 5)) + ")");
    }
}

// --- no guides, falloff, limits ----------------------------------------------------------------

static void CheckMisc()
{
    {
        Fixture f;
        f.withGuides = false;
        f.roots = {GfVec3f(1, 0, 2)};
        f.params = {{"weighting", VtValue(TfToken("gaussian"))},
                    {"defaultLength", VtValue(0.4f)}};
        UsdGenCurveBuffer b;
        Check(Cook(f, &b) && NearV(Point(b, 0, 4, 5), GfVec3d(1, 0.4, 2)) &&
                  NearV(Point(b, 0, 2, 5), GfVec3d(1, 0.2, 2)),
              "no guides: straight strand of defaultLength along the root normal");
        Check(Ints(b, "primaryGuide") == std::vector<int>({-1}) &&
                  Ints(b, "mirrorSign") == std::vector<int>({1}),
              "no guides: primaryGuide -1, mirrorSign +1");
        f.params = {{"weighting", VtValue(TfToken("nearest"))}, {"defaultLength", VtValue(0.0f)}};
        Check(Cook(f, &b) && NearV(Point(b, 0, 4, 5), GfVec3d(1, 0, 2)),
              "a strand of length <= 1e-5 is zero-length at its root");
    }
    {
        Fixture f;
        f.guides = {{GfVec3f(0, 0, 0), GfVec3f(0, 1, 0)}, {GfVec3f(2, 0, 0), GfVec3f(2, 1, 0)}};
        f.roots = {GfVec3f(0, 0, 0), GfVec3f(1.9f, 0, 0)};
        f.mirrorSign = {-1, 1};
        UsdGenCurveBuffer b;
        Check(Cook(f, &b), "falloff (default weighting) still cooks");
        Check(Ints(b, "primaryGuide") == std::vector<int>({0, 1}) &&
                  Ints(b, "nearestGuide") == std::vector<int>({0, 1}) &&
                  Ints(b, "mirrorSign") == std::vector<int>({-1, 1}) &&
                  Ints(b, "partId") == std::vector<int>({-1, -1}),
              "falloff emits primaryGuide, nearestGuide, mirrorSign, partId");
    }
    {
        Fixture f;
        f.guides = {{GfVec3f(0, 0, 0), GfVec3f(0, 1, 0)}};
        f.roots = {GfVec3f(0.1f, 0, 0)};
        f.cvCount = 100;
        f.params = {{"weighting", VtValue(TfToken("nearest"))}};
        UsdGenCurveBuffer b;
        Check(Cook(f, &b) && b.totalCvs == 100 &&
                  NearV(Point(b, 0, 99, 100), GfVec3d(0.1, 1, 0)),
              "cvCount above 64 is accepted");
        f.cvCount = int(kUsdGenMaxCvsPerCurve) + 1;
        Check(!Cook(Build(f), &b, true), "cvCount above kUsdGenMaxCvsPerCurve is rejected");
        f.cvCount = 5;
        f.params = {{"weighting", VtValue(TfToken("bogus"))}};
        Check(!Cook(Build(f), &b, true), "an unknown weighting is rejected");
        f.params = {{"weighting", VtValue(TfToken("nearest"))},
                    {"part:points", VtValue(VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)})},
                    {"part:counts", VtValue(VtIntArray{3})}};
        Check(!Cook(Build(f), &b, true), "part:counts must sum to part:points");
    }
}

// --- determinism ------------------------------------------------------------------------------------

static bool SameBuffer(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    auto same = [](VtFloatArray const &x, VtFloatArray const &y) {
        return x.size() == y.size() &&
               (x.empty() || std::memcmp(x.cdata(), y.cdata(), x.size() * sizeof(float)) == 0);
    };
    if (!same(a.px, b.px) || !same(a.py, b.py) || !same(a.pz, b.pz) || !same(a.width, b.width))
        return false;
    if (a.extraCurve.size() != b.extraCurve.size()) return false;
    for (size_t k = 0; k < a.extraCurve.size(); ++k) {
        UsdGenPlane const &x = a.extraCurve[k], &y = b.extraCurve[k];
        if (x.name != y.name || x.i != y.i || !same(x.f, y.f)) return false;
    }
    return true;
}

static void CheckDeterminism()
{
    Fixture f;
    // Six bent guides around the parting; 400 roots (the parallel path).
    for (int g = 0; g < 6; ++g) {
        float const x = -1.0f + 0.4f * float(g), z = (g % 2) ? 0.7f : -0.6f;
        f.guides.push_back({GfVec3f(x, 0, z), GfVec3f(x, 0.6f, z + 0.1f),
                            GfVec3f(x + 0.3f, 1.0f, z + 0.3f), GfVec3f(x + 0.7f, 1.2f, z + 0.2f)});
    }
    uint32_t state = 12345u;
    auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return float(state >> 8) / 16777216.0f;
    };
    for (int r = 0; r < 400; ++r)
        f.roots.push_back(GfVec3f(next() * 2.4f - 1.2f, 0, next() * 2.0f - 1.0f));
    VtVec3fArray const points{GfVec3f(0, 0, -3), GfVec3f(0.1f, 0, 0), GfVec3f(0, 0, 3)};
    for (char const *weighting : {"nearest", "inverseDistance", "gaussian"}) {
        f.params = {{"weighting", VtValue(TfToken(weighting))},
                    {"bendFollow", VtValue(true)},
                    {"tightness", VtValue(2.0f)},
                    {"part:points", VtValue(points)},
                    {"part:counts", VtValue(VtIntArray{3})},
                    {"part:amount", VtValue(0.8f)},
                    {"part:angle", VtValue(60.0f)},
                    {"part:radius", VtValue(0.2f)},
                    {"part:distance", VtValue(0.5f)}};
        f.cvCount = 12;
        UsdGenCurveBuffer a, b;
        bool const cooked = Cook(f, &a) && Cook(f, &b);
        bool finite = cooked;
        for (size_t o = 0; finite && o < a.px.size(); ++o)
            finite = std::isfinite(a.px[o]) && std::isfinite(a.py[o]) && std::isfinite(a.pz[o]);
        Check(cooked && finite && a.totalCurves == 400 && SameBuffer(a, b),
              std::string("determinism: two cooks are bit-identical (") + weighting + ")");
        std::vector<int> const part = Ints(a, "partId");
        int sides[2] = {0, 0};
        for (int v : part) if (v == 0 || v == 1) ++sides[v];
        Check(sides[0] > 0 && sides[1] > 0, std::string("determinism fixture parts both sides (") +
                                                 weighting + ")");
    }
}

int main()
{
    CheckHelpers();
    CheckNearest();
    CheckInverseDistance();
    CheckGaussian();
    CheckBendFollow();
    CheckParting();
    CheckMisc();
    CheckDeterminism();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all guide-blend checks passed\n");
    return 0;
}
