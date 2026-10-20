// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenScatterModes: UsdGenScatter generation-order modes
// (usdGen:distribution uvGrid / vertex / even / perGuide, countMode surface,
// guide proximity distance / coverage, usdGen:densityMap thinning, the
// strandIndex / strandSeed planes). Expected values are hand-computed from
// the reference root generator (surface target, mt19937 draw order, UV grid
// order, MSVC shuffle, coverage quantile) or are invariants that fail when
// the math is wrong (counts, spacing, proximity radius, thinning fraction,
// stable strand identity, determinism).
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/ops/scatterModes.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace usdGen;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int failures = 0;
void Check(bool v, std::string const &what)
{
    if (!v) {
        ++failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok: %s\n", what.c_str());
    }
}

UsdGenParamValue P(char const *name, VtValue v) { return {TfToken(name), std::move(v), false}; }
VtValue Tok(char const *t) { return VtValue(TfToken(t)); }

// nx x ny quads over [0,sx] x [0,sy] in the z = 0 plane, normal +Z, vertex
// UVs (x/sx, y/sy) when `uvs`.
UsdGenSurfaceDesc Grid(int nx, int ny, float sx, float sy, bool uvs = true)
{
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            s.restPoints.push_back(GfVec3f(sx * float(i) / float(nx), sy * float(j) / float(ny), 0.0f));
            if (uvs) s.uv.push_back(GfVec2f(float(i) / float(nx), float(j) / float(ny)));
        }
    auto v = [nx](int i, int j) { return j * (nx + 1) + i; };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            s.faceVertexCounts.push_back(4);
            for (int c : {v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)})
                s.faceVertexIndices.push_back(c);
        }
    s.points = s.restPoints;
    s.restNormals = VtVec3fArray(s.restPoints.size(), GfVec3f(0.0f, 0.0f, 1.0f));
    s.restNormalDomain = UsdGenSurfaceNormalDomain::Vertex;
    return s;
}

UsdGenGraphDesc Desc(UsdGenSurfaceDesc const &surface, std::vector<UsdGenParamValue> params,
                     int seed = 1337)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom");
    d.surfaces.push_back(surface);
    UsdGenNodeDesc node;
    node.path = SdfPath("/Groom/Scatter");
    node.type = TfToken("UsdGenScatter");
    node.surfaces = {surface.path};
    node.seed = seed;
    node.params = std::move(params);
    d.nodes.push_back(node);
    d.terminal = node.path;
    return d;
}

// Linear guide set; curves with < 2 points or zero length are kept in the
// set (the operator must skip them).
UsdGenCurveSetDesc GuideSet(std::vector<std::vector<GfVec3f>> const &curves,
                            std::vector<uint64_t> const &ids)
{
    UsdGenCurveSetDesc g;
    g.path = SdfPath("/Groom/Guides");
    g.role = UsdGenRole::Reference;
    g.curveRole = TfToken("guide");
    g.type = TfToken("linear");
    for (auto const &c : curves) {
        g.curveVertexCounts.push_back(int(c.size()));
        for (GfVec3f const &p : c) g.points.push_back(p);
    }
    g.rest = g.points;
    g.curveId = VtArray<uint64_t>(ids.begin(), ids.end());
    return g;
}

void AddGuides(UsdGenGraphDesc *d, UsdGenCurveSetDesc const &g)
{
    d->curveSets.push_back(g);
    d->nodes[0].curves.push_back(g.path);
}

void AddDensityMap(UsdGenGraphDesc *d, char const *path, VtFloatArray values)
{
    UsdGenMapDesc map;
    map.path = SdfPath(path);
    map.type = TfToken("UsdGenPaintMap");
    map.paintSurface = SdfPath("/Scalp");
    map.paintPrimvar = TfToken("usdGen:paint:density");
    map.paintInterpolation = TfToken("faceVarying");
    map.paintValues = std::move(values);
    d->maps.push_back(map);
    d->nodes[0].maps.push_back(map.path);
    d->nodes[0].mapBindings.push_back({map.path, TfToken("usdGen:densityMap")});
}

bool Cook(UsdGenGraphDesc const &d, UsdGenCurveBuffer *out, std::string *error = nullptr)
{
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult const compiled = compiler.Compile(d, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) {
            std::printf("  compile: %s\n", e.c_str());
            if (error) *error += e;
        }
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    auto run = scheduler.Run(graph, ctx, 1);
    for (auto const &e : run.diagnostics.errors) {
        std::printf("  run: %s\n", e.c_str());
        if (error) *error += e;
    }
    if (run.diagnostics.HasErrors()) return false;
    *out = graph.Output();
    return true;
}

// The compiler's reference value for a guide set (positions + rest + ids).
UsdGenReferenceSet ReferenceOf(UsdGenCurveSetDesc const &g)
{
    UsdGenReferenceSet r;
    UsdGenCurveBuffer &b = r.buffer;
    b.totalCurves = uint32_t(g.curveVertexCounts.size());
    b.totalCvs = uint32_t(g.points.size());
    b.cvOffsets = VtIntArray(b.totalCurves + 1, 0);
    for (size_t c = 0; c < g.curveVertexCounts.size(); ++c)
        b.cvOffsets[c + 1] = b.cvOffsets[c] + g.curveVertexCounts[c];
    for (GfVec3f const &p : g.points) {
        b.px.push_back(p[0]);
        b.py.push_back(p[1]);
        b.pz.push_back(p[2]);
    }
    b.rest = g.rest;
    b.curveId = g.curveId;
    return r;
}

// Direct generator call (same inputs the scheduler hands Capture), for the
// internal statistics.
bool Direct(UsdGenGraphDesc const &d, UsdGenCurveBuffer *out, scatter::ModeStats *st)
{
    UsdGenParamView view;
    view.desc = &d;
    view.node = &d.nodes[0];
    scatter::ModeConfig cfg;
    std::string error;
    if (!scatter::ReadModeConfig(&view, &cfg, &error)) {
        std::printf("  config: %s\n", error.c_str());
        return false;
    }
    UsdGenCaptureContext ctx;
    ctx.desc = &d;
    ctx.params = &view;
    ctx.surface = 0;
    ctx.seed = uint32_t(d.nodes[0].seed);
    ctx.mapBindings = d.nodes[0].mapBindings.data();
    ctx.mapBindingCount = uint32_t(d.nodes[0].mapBindings.size());
    UsdGenReferenceSet ref;
    UsdGenReferenceSet const *refs[1] = {&ref};
    if (!d.curveSets.empty()) {
        ref = ReferenceOf(d.curveSets[0]);
        ctx.references = refs;
        ctx.referenceCount = 1;
    }
    UsdGenDiagnostics diag;
    bool const ok = scatter::CaptureModes(ctx, cfg, d.surfaces[0], out, &diag, st);
    for (auto const &e : diag.errors) std::printf("  direct: %s\n", e.c_str());
    return ok && !diag.HasErrors();
}

UsdGenPlane const *FindPlane(UsdGenCurveBuffer const &b, char const *name)
{
    for (UsdGenPlane const &p : b.extraCurve)
        if (p.name == TfToken(name)) return &p;
    return nullptr;
}

std::vector<int> PlaneValues(UsdGenCurveBuffer const &b, char const *name)
{
    UsdGenPlane const *p = FindPlane(b, name);
    if (!p || p->type != TfToken("int") || p->i.size() != b.totalCurves) return {};
    return std::vector<int>(p->i.begin(), p->i.end());
}

GfVec3f Root(UsdGenCurveBuffer const &b, size_t c) { return GfVec3f(b.px[c], b.py[c], b.pz[c]); }

// Roots indexed by strandIndex (generation order). Empty when the plane is
// missing or not a permutation-free index.
std::map<int, size_t> ByStrandIndex(UsdGenCurveBuffer const &b)
{
    std::map<int, size_t> m;
    std::vector<int> const idx = PlaneValues(b, "strandIndex");
    for (size_t c = 0; c < idx.size(); ++c) m[idx[c]] = c;
    return m;
}

bool SameBuffer(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    if (a.totalCurves != b.totalCurves || a.px != b.px || a.py != b.py || a.pz != b.pz ||
        a.curveId != b.curveId || a.rootPrim != b.rootPrim || a.rootUV != b.rootUV ||
        a.rootN != b.rootN || a.rootT != b.rootT || a.extraCurve.size() != b.extraCurve.size())
        return false;
    for (size_t i = 0; i < a.extraCurve.size(); ++i)
        if (a.extraCurve[i].name != b.extraCurve[i].name || a.extraCurve[i].i != b.extraCurve[i].i)
            return false;
    return true;
}

float MinSpacing(UsdGenCurveBuffer const &b)
{
    // O(n log n)-ish: sort by x and sweep.
    std::vector<GfVec3f> p(b.totalCurves);
    for (size_t c = 0; c < p.size(); ++c) p[c] = Root(b, c);
    std::sort(p.begin(), p.end(), [](GfVec3f const &x, GfVec3f const &y) { return x[0] < y[0]; });
    float best = 1e30f;
    for (size_t i = 0; i < p.size(); ++i)
        for (size_t j = i + 1; j < p.size() && p[j][0] - p[i][0] < best; ++j)
            best = std::min(best, (p[j] - p[i]).GetLength());
    return best;
}

float Mt01(std::mt19937 &rng) { return float(rng() >> 8) * (1.0f / 16777216.0f); }

bool Near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Spec §5 root hash, restated independently of the operator.
float SpecRootHash(uint64_t i, int tri, int vertex, GfVec3f const &pos, GfVec3f const &bary)
{
    auto bits = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint64_t(u); };
    uint64_t const vals[9] = {i, uint64_t(tri + 1), uint64_t(vertex + 1), bits(pos[0]), bits(pos[1]),
                              bits(pos[2]), bits(bary[0]), bits(bary[1]), bits(bary[2])};
    uint64_t h = 0x14650fb0739d0383ull;
    for (uint64_t v : vals) h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    return float(uint32_t((h >> 33) ^ h) & 0xffffffu) / 16777216.0f;
}

// ---------------------------------------------------------------------------

void TestUvGrid()
{
    // Unit quad, density 100 -> target 100, n = 10: the grid point of row r,
    // column c is generation index 10r + c at ((c+.5)/10, (r+.5)/10).
    UsdGenGraphDesc d = Desc(Grid(1, 1, 1.0f, 1.0f), {P("distribution", Tok("uvGrid")),
                                                      P("density", VtValue(100.0))});
    UsdGenCurveBuffer out;
    Check(Cook(d, &out), "uvGrid cooks");
    Check(out.totalCurves == 100, "uvGrid: unit quad, density 100 -> 100 roots (got " +
                                      std::to_string(out.totalCurves) + ")");
    std::map<int, size_t> const byIndex = ByStrandIndex(out);
    bool grid = byIndex.size() == 100, uvOk = true, frames = true, seeds = true;
    std::vector<int> const seed = PlaneValues(out, "strandSeed");
    for (auto const &[k, c] : byIndex) {
        float const x = (float(k % 10) + 0.5f) / 10.0f, y = (float(k / 10) + 0.5f) / 10.0f;
        GfVec3f const p = Root(out, c);
        grid = grid && Near(p[0], x, 1e-6f) && Near(p[1], y, 1e-6f) && p[2] == 0.0f;
        uvOk = uvOk && Near(out.rootUV[c][0], x, 1e-6f) && Near(out.rootUV[c][1], y, 1e-6f) &&
               out.rootPrim[c] == 0;
        frames = frames && Near(out.rootN[c][2], 1.0f, 1e-6f) &&
                 std::fabs(GfDot(out.rootT[c], out.rootN[c])) < 1e-6f &&
                 Near(out.rootB[c].GetLength(), 1.0f, 1e-5f);
        seeds = seeds && seed.size() == out.totalCurves && seed[c] == k;
    }
    Check(grid, "uvGrid: strandIndex k sits at grid point ((k%10+.5)/10, (k/10+.5)/10)");
    Check(uvOk, "uvGrid: rootUV = grid uv, rootPrim = face 0");
    Check(frames, "uvGrid: orthonormal rest frames with N = +Z");
    Check(seeds, "uvGrid: strandSeed == strandIndex for scalp roots");

    // Density 50 -> target 50, n = ceil(sqrt(50)) = 8: rows 0..5 full plus
    // two points of row 6, then the walk stops at the target.
    d = Desc(Grid(1, 1, 1.0f, 1.0f), {P("distribution", Tok("uvGrid")), P("density", VtValue(50.0))});
    Check(Cook(d, &out) && out.totalCurves == 50, "uvGrid: density 50 -> 50 roots");
    bool grid8 = true;
    for (auto const &[k, c] : ByStrandIndex(out)) {
        float const x = (float(k % 8) + 0.5f) / 8.0f, y = (float(k / 8) + 0.5f) / 8.0f;
        grid8 = grid8 && Near(out.px[c], x, 1e-6f) && Near(out.py[c], y, 1e-6f);
    }
    std::map<int, size_t> const by8 = ByStrandIndex(out);
    Check(grid8 && !by8.empty() && by8.rbegin()->first == 49,
          "uvGrid: n = 8 row-major walk stops at the 50th point");

    // Density is uniform in UV, the count comes from the surface area: a
    // 2 x 1 quad with [0,1]^2 UVs, density 32 -> 64 roots on an 8 x 8 UV grid.
    d = Desc(Grid(2, 1, 2.0f, 1.0f), {P("distribution", Tok("uvGrid")), P("density", VtValue(32.0))});
    Check(Cook(d, &out) && out.totalCurves == 64, "uvGrid: 2x1 quad, density 32 -> 64 roots");
    bool wide = true;
    for (auto const &[k, c] : ByStrandIndex(out)) {
        float const x = 2.0f * (float(k % 8) + 0.5f) / 8.0f, y = (float(k / 8) + 0.5f) / 8.0f;
        wide = wide && Near(out.px[c], x, 1e-5f) && Near(out.py[c], y, 1e-6f) &&
               out.rootPrim[c] == (x < 1.0f ? 0 : 1);
    }
    Check(wide, "uvGrid: UV grid maps to x = 2u, faces by half");

    // No UVs: mt19937(seed) area darts, draw order u, a, b; the first root
    // is hand-computed from the spec (quad fan (0,1,2),(0,2,3), cdf .5/1).
    d = Desc(Grid(1, 1, 1.0f, 1.0f, false),
             {P("distribution", Tok("uvGrid")), P("density", VtValue(37.0))});
    UsdGenCurveBuffer again;
    Check(Cook(d, &out) && out.totalCurves == 37, "uvGrid without UVs: darts reach the target");
    Check(Cook(d, &again) && SameBuffer(out, again), "uvGrid darts: deterministic");
    {
        std::mt19937 rng(1337u);
        float const u = Mt01(rng);
        float a = Mt01(rng), b = Mt01(rng);
        int const tri = u <= 0.5f ? 0 : 1;
        if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
        float const w0 = 1.0f - a - b;
        GfVec3f const P0(0, 0, 0), P1 = tri == 0 ? GfVec3f(1, 0, 0) : GfVec3f(1, 1, 0),
                      P2 = tri == 0 ? GfVec3f(1, 1, 0) : GfVec3f(0, 1, 0);
        GfVec3f const want(P0[0] * w0 + P1[0] * a + P2[0] * b, P0[1] * w0 + P1[1] * a + P2[1] * b, 0.0f);
        auto const m = ByStrandIndex(out);
        auto const it = m.find(0);
        Check(it != m.end() && Root(out, it->second) == want,
              "uvGrid darts: strand 0 equals the hand-drawn first dart");
    }
    UsdGenGraphDesc d2 = d;
    d2.nodes[0].seed = 7;
    Check(Cook(d2, &again) && again.totalCurves == 37 && again.px != out.px,
          "uvGrid darts: the seed changes the roots, not the count");
}

void TestCountMode()
{
    // random + surface: exactly roundf(A * density) roots.
    UsdGenGraphDesc d = Desc(Grid(3, 2, 1.0f, 1.0f), {P("countMode", Tok("surface")),
                                                      P("density", VtValue(123.4))});
    UsdGenCurveBuffer out, again;
    Check(Cook(d, &out) && out.totalCurves == 123, "random + surface: round(1 * 123.4) = 123 roots");
    Check(Cook(d, &again) && SameBuffer(out, again), "random + surface: deterministic");
    Check(FindPlane(out, "strandIndex") && FindPlane(out, "strandSeed"),
          "random + surface: emits strandIndex / strandSeed");
    std::set<uint64_t> ids(out.curveId.begin(), out.curveId.end());
    Check(ids.size() == out.totalCurves, "random + surface: unique curve ids");
    // The default path stays the per-face stochastic scatter without planes.
    d = Desc(Grid(3, 2, 1.0f, 1.0f), {P("density", VtValue(123.4))});
    Check(Cook(d, &out) && out.extraCurve.empty() && out.totalCurves > 0,
          "random + perFace (default): no strand planes");
    // Bind rejects unknown tokens.
    d = Desc(Grid(1, 1, 1.0f, 1.0f), {P("distribution", Tok("hexagonal"))});
    Check(!Cook(d, &out), "unknown distribution fails closed");
}

void TestEven()
{
    // Even on a unit plane: target 400, r0 = sqrt(1/400) = 0.05.
    UsdGenGraphDesc d = Desc(Grid(16, 16, 1.0f, 1.0f),
                             {P("distribution", Tok("even")), P("density", VtValue(400.0))});
    UsdGenCurveBuffer out, direct;
    scatter::ModeStats st;
    Check(Cook(d, &out) && out.totalCurves == 400, "even: 400 roots");
    Check(Direct(d, &direct, &st) && SameBuffer(out, direct), "even: direct capture == scheduled cook");
    Check(st.target == 400 && Near(st.spacing, 0.05f, 1e-7f), "even: target 400, r0 = 0.05");
    Check(st.candidates == 4000 && st.accepted == 4000, "even: candidates = max(2*target, 4000), all valid");
    float const minSp = MinSpacing(out);
    std::printf("  even: firstPass=%u relax=%u lastResort=%d r=%g minSpacing=%g\n", st.firstPass,
                st.relaxations, int(st.lastResort), double(st.finalRadius), double(minSp));
    Check(!st.lastResort && minSp >= st.finalRadius * (1.0f - 1e-6f),
          "even: min spacing >= the final Poisson radius");
    Check(st.finalRadius >= 0.75f * 0.05f * 0.85f, "even: radius relaxed at most one step past 0.75 r0");
    Check(st.firstPass > 0 && st.firstPass < 400, "even: first pass at r0 cannot saturate (RSA limit)");
    bool inside = true;
    for (size_t c = 0; c < out.totalCurves; ++c)
        inside = inside && out.px[c] >= 0.0f && out.px[c] <= 1.0f && out.py[c] >= 0.0f &&
                 out.py[c] <= 1.0f && out.pz[c] == 0.0f &&
                 out.rootPrim[c] == int(std::min(15.0f, std::floor(out.py[c] * 16.0f))) * 16 +
                                        int(std::min(15.0f, std::floor(out.px[c] * 16.0f)));
    Check(inside, "even: roots on the plane, rootPrim = containing face");

    // Autumn-equivalent targets: A_mesh = 0.1905421 m^2 (0.5 x 0.3810842).
    for (auto const &[density, want] : {std::pair<double, uint32_t>{30000.0, 5716u},
                                        std::pair<double, uint32_t>{12000.0, 2287u}}) {
        UsdGenGraphDesc a = Desc(Grid(10, 8, 0.5f, 0.3810842f),
                                 {P("distribution", Tok("even")), P("density", VtValue(density))});
        Check(Direct(a, &direct, &st) && st.target == want && direct.totalCurves == want &&
                  Near(st.areaMesh, 0.1905421f, 1e-6f),
              "even: Autumn-equivalent target " + std::to_string(want));
        Check(Near(st.spacing, std::sqrt(st.areaMesh / float(want)), 1e-9f),
              "even: r0 = sqrt(A_mesh / target) without proximity");
    }
}

void TestProximityDistance()
{
    // Two guides, cutoff 0.1: the target stays round(A * density) and every
    // root lands within 0.1 of a guide root.
    UsdGenCurveSetDesc const guides = GuideSet(
        {{GfVec3f(0.25f, 0.25f, 0.0f), GfVec3f(0.25f, 0.25f, 0.1f)},
         {GfVec3f(0.75f, 0.6f, 0.0f), GfVec3f(0.75f, 0.6f, 0.1f)}},
        {11, 12});
    UsdGenGraphDesc d = Desc(Grid(32, 32, 1.0f, 1.0f),
                             {P("distribution", Tok("even")), P("density", VtValue(2000.0)),
                              P("proximity:mode", Tok("distance")),
                              P("proximity:distance", VtValue(0.1))});
    AddGuides(&d, guides);
    UsdGenCurveBuffer out, direct;
    scatter::ModeStats st;
    Check(Cook(d, &out), "proximity distance: compiles with one usdGen:guides reference");
    Check(Direct(d, &direct, &st) && SameBuffer(out, direct), "proximity distance: direct == cook");
    Check(out.totalCurves == 2000 && st.target == 2000,
          "proximity distance: target not reduced (2000 roots in the guide disks)");
    bool within = true;
    for (size_t c = 0; c < out.totalCurves; ++c) {
        GfVec3f const p = Root(out, c);
        within = within && (std::min((p - guides.points[0]).GetLengthSq(),
                                     (p - guides.points[2]).GetLengthSq()) <= 0.01f + 1e-7f);
    }
    Check(within, "proximity distance: every root within 0.1 of a guide root");
    float const disks = 2.0f * 3.14159265f * 0.01f;
    std::printf("  proximity: A_dom=%g hits=%u A_eff=%g (disks %g) spacing=%g firstPass=%u relax=%u\n",
                double(st.areaDomain), st.calibrationHits, double(st.areaEffective), double(disks),
                double(st.spacing), st.firstPass, st.relaxations);
    Check(st.proximity && Near(st.cutoff2, 0.01f, 1e-9f), "proximity distance: cutoff2 = d^2");
    Check(st.areaDomain > disks && st.areaDomain < 4.0f * disks,
          "proximity distance: triangle domain hugs the disks");
    Check(st.areaEffective == float(st.calibrationHits) / 4096.0f * st.areaDomain,
          "proximity distance: A_eff = hits / 4096 * A_dom");
    Check(std::fabs(st.areaEffective - disks) < 0.08f * disks,
          "proximity distance: calibrated area ~ 2 pi d^2");
    Check(Near(st.spacing, std::sqrt(st.areaEffective / 2000.0f), 1e-9f),
          "proximity distance: r0 = sqrt(A_eff / target)");

    // uvGrid with proximity: grid points outside the region are lost.
    d.nodes[0].params[0] = P("distribution", Tok("uvGrid"));
    Check(Cook(d, &out), "proximity uvGrid cooks");
    bool gridWithin = out.totalCurves > 0 && out.totalCurves < 2000;
    for (size_t c = 0; c < out.totalCurves; ++c) {
        GfVec3f const p = Root(out, c);
        gridWithin = gridWithin && (std::min((p - guides.points[0]).GetLengthSq(),
                                             (p - guides.points[2]).GetLengthSq()) <= 0.01f + 1e-7f);
    }
    // 45 x 45 grid (2025 >= 2000 points) ~ 2 pi 0.01 * 2025 = 127 inside.
    Check(gridWithin && out.totalCurves > 100 && out.totalCurves < 160,
          "proximity uvGrid: only grid points inside the disks (" + std::to_string(out.totalCurves) + ")");

    // random + perFace + proximity: roots outside are dropped (thinning).
    d.nodes[0].params[0] = P("distribution", Tok("random"));
    Check(Cook(d, &out) && out.totalCurves > 0 && out.totalCurves < 400 &&
              FindPlane(out, "strandIndex") != nullptr,
          "proximity random/perFace: thins to the disks and emits planes");
}

void TestProximityCoverage()
{
    // One guide at the centre, coverage 0.3: cutoff2 is the 0.3 quantile of
    // the squared nearest-guide distance over 2000 mt19937(42) samples; the
    // disk of area 0.3 has rho^2 = 0.3 / pi.
    UsdGenSurfaceDesc const grid = Grid(8, 8, 1.0f, 1.0f);
    UsdGenCurveSetDesc const guides =
        GuideSet({{GfVec3f(0.5f, 0.5f, 0.0f), GfVec3f(0.5f, 0.5f, 0.2f)}}, {5});
    UsdGenGraphDesc d = Desc(grid, {P("distribution", Tok("even")), P("density", VtValue(600.0)),
                                    P("proximity:mode", Tok("coverage")),
                                    P("proximity:coverage", VtValue(0.3))});
    AddGuides(&d, guides);
    UsdGenCurveBuffer out;
    scatter::ModeStats st;
    Check(Direct(d, &out, &st), "proximity coverage: generates");
    Check(Near(st.cutoff2, 0.3f / 3.14159265f, 0.012f),
          "proximity coverage: cutoff2 ~ 0.3 / pi (got " + std::to_string(st.cutoff2) + ")");
    // Independent restatement of the calibration (spec §2.3).
    {
        std::vector<GfVec3f> tri;  // 3 per fan triangle, face order
        for (size_t f = 0; f < grid.faceVertexCounts.size(); ++f) {
            int const *c = &grid.faceVertexIndices[4 * f];
            for (int j = 1; j <= 2; ++j)
                for (int k : {c[0], c[j], c[j + 1]}) tri.push_back(grid.restPoints[size_t(k)]);
        }
        size_t const nt = tri.size() / 3;
        std::vector<float> area(nt), cdf(nt);
        float total = 0.0f;
        for (size_t t = 0; t < nt; ++t) {
            GfVec3f const x = GfCross(tri[3 * t + 1] - tri[3 * t], tri[3 * t + 2] - tri[3 * t]);
            area[t] = 0.5f * std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            total += area[t];
        }
        float run = 0.0f;
        for (size_t t = 0; t < nt; ++t) { run += area[t]; cdf[t] = run / total; }
        std::mt19937 rng(42u);
        std::vector<float> d2;
        for (int s = 0; s < 2000; ++s) {
            float const u = Mt01(rng);
            size_t t = size_t(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
            t = std::min(t, nt - 1);
            float a = Mt01(rng), b = Mt01(rng);
            if (a + b > 1.0f) { a = 1.0f - a; b = 1.0f - b; }
            float const w0 = 1.0f - a - b;
            GfVec3f const &A = tri[3 * t], &B = tri[3 * t + 1], &C = tri[3 * t + 2];
            GfVec3f const p(A[0] * w0 + B[0] * a + C[0] * b, A[1] * w0 + B[1] * a + C[1] * b,
                            A[2] * w0 + B[2] * a + C[2] * b);
            GfVec3f const g = guides.points[0];
            float const dx = p[0] - g[0], dy = p[1] - g[1], dz = p[2] - g[2];
            d2.push_back(dx * dx + dy * dy + dz * dz);
        }
        std::sort(d2.begin(), d2.end());
        float const x = 0.3f * 1999.0f;
        int const i = int(std::floor(x));
        float const f = x - float(i);
        float const want = d2[size_t(i)] * (1.0f - f) + d2[size_t(i + 1)] * f;
        Check(st.cutoff2 == want, "proximity coverage: cutoff2 bit-equals the spec quantile");
    }
    bool within = true;
    for (size_t c = 0; c < out.totalCurves; ++c)
        within = within && (Root(out, c) - guides.points[0]).GetLengthSq() <= st.cutoff2;
    Check(out.totalCurves == 600 && within,
          "proximity coverage: target kept, all roots inside the coverage radius");
    Check(st.areaDomain == 0.0f && st.areaEffective == st.areaMesh,
          "proximity coverage: no domain, A_eff = A_mesh");
    // coverage >= 0.9999 switches proximity off.
    d.nodes[0].params[3] = P("proximity:coverage", VtValue(1.0));
    Check(Direct(d, &out, &st) && !st.proximity && st.cutoff2 < 0.0f,
          "proximity coverage 1.0: proximity off");
}

void TestVertex()
{
    // Two unit quads side by side WITHOUT shared vertices: 8 points, 6
    // unique positions (x = 1 duplicated).
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/Scalp");
    s.restPoints = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {1, 1, 0}};
    s.points = s.restPoints;
    s.faceVertexCounts = {4, 4};
    s.faceVertexIndices = {0, 1, 2, 3, 4, 5, 6, 7};
    for (GfVec3f const &p : s.restPoints) s.uv.push_back(GfVec2f(p[0] * 0.5f, p[1]));
    s.restNormals = VtVec3fArray(8, GfVec3f(0, 0, 1));
    s.restNormalDomain = UsdGenSurfaceNormalDomain::Vertex;

    UsdGenGraphDesc d = Desc(s, {P("distribution", Tok("vertex")), P("density", VtValue(100.0))});
    UsdGenCurveBuffer out;
    Check(Cook(d, &out) && out.totalCurves == 6, "vertex: duplicate seam positions emitted once (6)");
    std::set<std::pair<float, float>> seen;
    bool atVertex = true;
    for (size_t c = 0; c < out.totalCurves; ++c) {
        seen.insert({out.px[c], out.py[c]});
        bool found = false;
        for (size_t v = 0; v < 8; ++v)
            if (Root(out, c) == s.restPoints[v]) {
                found = true;
                atVertex = atVertex && out.rootUV[c] == s.uv[v];
            }
        atVertex = atVertex && found && out.rootN[c] == GfVec3f(0, 0, 1);
    }
    Check(atVertex && seen.size() == 6, "vertex: unique positions, exactly at vertices, vertex uv / normal");

    // Target 3 (area 2, density 1.5): the first three unique positions of
    // the MSVC-shuffled vertex order with mt19937(1337).
    d = Desc(s, {P("distribution", Tok("vertex")), P("density", VtValue(1.5))});
    Check(Cook(d, &out) && out.totalCurves == 3, "vertex: target 3");
    std::vector<uint32_t> order(8);
    for (uint32_t i = 0; i < 8; ++i) order[i] = i;
    std::mt19937 rng(1337u);
    strand::MsvcShuffle(&order, rng);
    std::vector<GfVec3f> want;
    for (uint32_t v : order) {
        if (want.size() == 3) break;
        if (std::find(want.begin(), want.end(), s.restPoints[v]) == want.end())
            want.push_back(s.restPoints[v]);
    }
    bool shuffled = true;
    for (auto const &[k, c] : ByStrandIndex(out))
        shuffled = shuffled && k < 3 && Root(out, c) == want[size_t(k)];
    Check(shuffled && ByStrandIndex(out).size() == 3,
          "vertex: generation order = MSVC shuffle of the vertices");

    // Proximity: only vertices within 0.6 of the guide at (0,0).
    d = Desc(s, {P("distribution", Tok("vertex")), P("density", VtValue(100.0)),
                 P("proximity:mode", Tok("distance")), P("proximity:distance", VtValue(1.05))});
    AddGuides(&d, GuideSet({{GfVec3f(0, 0, 0), GfVec3f(0, 0, 1)}}, {1}));
    Check(Cook(d, &out) && out.totalCurves == 3, "vertex + proximity 1.05: (0,0), (1,0), (0,1) only");
}

void TestPerGuide()
{
    // 2 x 2 grid over [0,1]^2 (faces 0..3); guides 0 and 3 are usable, 1 has
    // a single point and 2 has zero length.
    UsdGenCurveSetDesc const guides = GuideSet(
        {{GfVec3f(0.2f, 0.3f, 0.0f), GfVec3f(0.2f, 0.3f, 0.5f), GfVec3f(0.2f, 0.4f, 0.9f)},
         {GfVec3f(0.6f, 0.6f, 0.0f)},
         {GfVec3f(0.4f, 0.4f, 0.0f), GfVec3f(0.4f, 0.4f, 0.0f)},
         {GfVec3f(0.7f, 0.8f, 0.0f), GfVec3f(0.7f, 0.8f, 0.3f)}},
        {101, 102, 103, 104});
    UsdGenGraphDesc d = Desc(Grid(2, 2, 1.0f, 1.0f),
                             {P("distribution", Tok("perGuide")), P("density", VtValue(1000.0)),
                              P("proximity:mode", Tok("distance")),
                              P("proximity:distance", VtValue(0.01))});
    AddGuides(&d, guides);
    UsdGenCurveBuffer out;
    Check(Cook(d, &out) && out.totalCurves == 2, "perGuide: one root per usable guide (2)");
    std::map<int, size_t> const m = ByStrandIndex(out);
    std::vector<int> const seed = PlaneValues(out, "strandSeed");
    bool ok = m.size() == 2 && m.count(0) && m.count(1) && seed.size() == 2;
    if (ok) {
        size_t const a = m.at(0), b = m.at(1);
        ok = Root(out, a) == guides.points[0] && Root(out, b) == guides.points[6] &&
             out.curveId[a] == 101 && out.curveId[b] == 104 && out.rootPrim[a] == 0 &&
             out.rootPrim[b] == 3 && Near(out.rootUV[a][0], 0.2f, 1e-6f) &&
             Near(out.rootUV[a][1], 0.3f, 1e-6f) && Near(out.rootUV[b][0], 0.7f, 1e-6f) &&
             seed[a] == int(uint32_t(strand::SplitMix64(101)) & 0x7fffffffu) &&
             seed[b] == int(uint32_t(strand::SplitMix64(104)) & 0x7fffffffu);
    }
    Check(ok, "perGuide: roots = guide roots in guide order; ids, faces, uv, seed = SM(id)&0x7fffffff");
}

void TestDensityMap()
{
    UsdGenSurfaceDesc const grid = Grid(8, 8, 1.0f, 1.0f);
    size_t const corners = grid.faceVertexIndices.size();
    std::vector<UsdGenParamValue> const params = {P("distribution", Tok("even")),
                                                  P("density", VtValue(2000.0))};
    UsdGenCurveBuffer all, out;
    scatter::ModeStats st;
    UsdGenGraphDesc base = Desc(grid, params);
    Check(Cook(base, &all) && all.totalCurves == 2000, "density map: unthinned reference cook");
    std::map<int, size_t> const allByIndex = ByStrandIndex(all);

    auto thinned = [&](VtFloatArray values, UsdGenCurveBuffer *result) {
        UsdGenGraphDesc d = Desc(grid, params);
        AddDensityMap(&d, "/Groom/Maps/density", std::move(values));
        return Cook(d, result);
    };
    // Survivors keep their strandIndex and their exact root.
    auto stable = [&](UsdGenCurveBuffer const &b) {
        for (auto const &[k, c] : ByStrandIndex(b)) {
            auto const it = allByIndex.find(k);
            if (it == allByIndex.end() || Root(b, c) != Root(all, it->second) ||
                b.curveId[c] != all.curveId[it->second])
                return false;
        }
        return true;
    };
    Check(thinned(VtFloatArray(corners, 0.5f), &out), "density map 0.5: compiled cook binds usdGen:densityMap");
    double const frac = double(out.totalCurves) / 2000.0;
    Check(frac > 0.45 && frac < 0.55, "density map 0.5: ~half kept (" + std::to_string(out.totalCurves) + ")");
    Check(stable(out), "density map 0.5: survivors keep strandIndex, position and id");
    // Exact: survivors are the roots whose spec hash is <= 0.5; uvGrid roots
    // expose tri / barycentrics analytically (unit quad, UV = xy).
    {
        UsdGenSurfaceDesc const quad = Grid(1, 1, 1.0f, 1.0f);
        std::vector<UsdGenParamValue> const gp = {P("distribution", Tok("uvGrid")),
                                                  P("density", VtValue(400.0))};
        UsdGenGraphDesc d = Desc(quad, gp);
        AddDensityMap(&d, "/Groom/Maps/density", VtFloatArray(4, 0.5f));
        Check(Cook(d, &out), "density map on uvGrid cooks");
        std::set<int> kept;
        for (auto const &kv : ByStrandIndex(out)) kept.insert(kv.first);
        bool exact = true;
        for (int k = 0; k < 400; ++k) {
            float const x = (float(k % 20) + 0.5f) / 20.0f, y = (float(k / 20) + 0.5f) / 20.0f;
            // Fan (0,1,2) holds y <= x, (0,2,3) the rest; UV barycentrics.
            GfVec3f bary;
            int tri;
            if (y <= x) { tri = 0; bary = GfVec3f(1.0f - x, x - y, y); }
            else { tri = 1; bary = GfVec3f(1.0f - y, x, y - x); }
            // The operator's dot-product barycentrics, recomputed the same way.
            GfVec2f const a(0, 0), b = tri == 0 ? GfVec2f(1, 0) : GfVec2f(1, 1),
                          c = tri == 0 ? GfVec2f(1, 1) : GfVec2f(0, 1), q(x, y);
            GfVec2f const v0 = b - a, v1 = c - a, v2 = q - a;
            float const d00 = v0[0] * v0[0] + v0[1] * v0[1], d01 = v0[0] * v1[0] + v0[1] * v1[1];
            float const d11 = v1[0] * v1[0] + v1[1] * v1[1], d20 = v2[0] * v0[0] + v2[1] * v0[1];
            float const d21 = v2[0] * v1[0] + v2[1] * v1[1], den = d00 * d11 - d01 * d01;
            float const w1 = (d11 * d20 - d01 * d21) / den, w2 = (d00 * d21 - d01 * d20) / den;
            bary = GfVec3f(1.0f - w1 - w2, w1, w2);
            GfVec3f const P0(0, 0, 0), P1 = tri == 0 ? GfVec3f(1, 0, 0) : GfVec3f(1, 1, 0),
                          P2 = tri == 0 ? GfVec3f(1, 1, 0) : GfVec3f(0, 1, 0);
            GfVec3f const pos(P0[0] * bary[0] + P1[0] * bary[1] + P2[0] * bary[2],
                              P0[1] * bary[0] + P1[1] * bary[1] + P2[1] * bary[2], 0.0f);
            float const h = SpecRootHash(uint64_t(k), tri, -1, pos, bary);
            exact = exact && (h <= 0.5f) == (kept.count(k) == 1) &&
                    scatter::RootHash01(uint64_t(k), tri, -1, pos, bary) == h;
        }
        Check(exact, "density map: kept iff rootHash(i, tri, bary, pos) <= m (spec §5)");
    }
    Check(thinned(VtFloatArray(corners, 1.0f), &out) && out.totalCurves == 2000,
          "density map 1.0: all kept");
    Check(thinned(VtFloatArray(corners, 0.9995f), &out) && out.totalCurves == 2000,
          "density map 0.9995 (>= 0.999): all kept");
    Check(thinned(VtFloatArray(corners, 0.0f), &out) && out.totalCurves == 0, "density map 0: none kept");
    // Left half painted 1, right half 0 (per face, all corners).
    VtFloatArray half(corners, 0.0f);
    for (size_t f = 0; f < 64; ++f)
        if (f % 8 < 4)
            for (int k = 0; k < 4; ++k) half[4 * f + size_t(k)] = 1.0f;
    Check(thinned(half, &out), "density map half: cooks");
    bool left = out.totalCurves > 0;
    size_t wantLeft = 0;
    for (size_t c = 0; c < all.totalCurves; ++c) wantLeft += all.rootPrim[c] % 8 < 4 ? 1 : 0;
    for (size_t c = 0; c < out.totalCurves; ++c) left = left && out.rootPrim[c] % 8 < 4;
    Check(left && out.totalCurves == wantLeft && stable(out),
          "density map half: exactly the roots of painted faces survive");
    // Two maps multiply: 0.5 * 0.5.
    {
        UsdGenGraphDesc d = Desc(grid, params);
        AddDensityMap(&d, "/Groom/Maps/a", VtFloatArray(corners, 0.5f));
        AddDensityMap(&d, "/Groom/Maps/b", VtFloatArray(corners, 0.5f));
        Check(Cook(d, &out), "two density maps cook");
        double const q = double(out.totalCurves) / 2000.0;
        Check(q > 0.21 && q < 0.29 && stable(out),
              "two density maps multiply (0.25 kept: " + std::to_string(out.totalCurves) + ")");
        Check(Direct(d, &out, &st) && st.placed == 2000 && st.emitted == out.totalCurves,
              "density maps thin after generation: target unchanged");
    }
    // random + perFace: the map thins the per-face roots.
    {
        UsdGenGraphDesc d = Desc(grid, {P("density", VtValue(2000.0))});
        UsdGenCurveBuffer plain;
        Check(Cook(d, &plain) && plain.extraCurve.empty(), "random/perFace reference cook");
        AddDensityMap(&d, "/Groom/Maps/density", half);
        Check(Cook(d, &out), "random/perFace + density map cooks");
        bool leftOnly = out.totalCurves > 0;
        size_t plainLeft = 0;
        for (size_t c = 0; c < plain.totalCurves; ++c) plainLeft += plain.rootPrim[c] % 8 < 4 ? 1 : 0;
        for (size_t c = 0; c < out.totalCurves; ++c) leftOnly = leftOnly && out.rootPrim[c] % 8 < 4;
        std::vector<int> const idx = PlaneValues(out, "strandIndex");
        std::set<int> unique(idx.begin(), idx.end());
        Check(leftOnly && out.totalCurves == plainLeft && unique.size() == out.totalCurves,
              "random/perFace + density map: painted faces only, unique strandIndex");
    }
}

void TestDeterminism()
{
    UsdGenCurveSetDesc const guides =
        GuideSet({{GfVec3f(0.3f, 0.3f, 0.0f), GfVec3f(0.3f, 0.3f, 0.1f)}}, {9});
    for (char const *mode : {"uvGrid", "vertex", "even", "perGuide", "random"}) {
        UsdGenGraphDesc d = Desc(Grid(6, 5, 1.0f, 0.8f),
                                 {P("distribution", Tok(mode)), P("density", VtValue(300.0)),
                                  P("countMode", Tok("surface")),
                                  P("proximity:mode", Tok("distance")),
                                  P("proximity:distance", VtValue(0.4))});
        AddGuides(&d, guides);
        AddDensityMap(&d, "/Groom/Maps/density", VtFloatArray(d.surfaces[0].faceVertexIndices.size(), 0.7f));
        UsdGenCurveBuffer a, b;
        bool const same = Cook(d, &a) && Cook(d, &b) && SameBuffer(a, b) && a.totalCurves > 0;
        Check(same, std::string("determinism: ") + mode + " cooks twice identically (" +
                        std::to_string(a.totalCurves) + " roots)");
    }
}

}  // namespace

int main()
{
    usdGenRegisterM1Operators();
    TestUvGrid();
    TestCountMode();
    TestEven();
    TestProximityDistance();
    TestProximityCoverage();
    TestVertex();
    TestPerGuide();
    TestDensityMap();
    TestDeterminism();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
