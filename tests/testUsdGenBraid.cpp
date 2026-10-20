// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenBraid: UsdGenBraid three-lane weave around each guide.
//
// Hand-built graphs (CurveSource of straight strands -> UsdGenBraid with a
// guide reference set) cooked through UsdGenCompiler + UsdGenScheduler.
// Expected values are computed here from the braid formulas (lane shape
// f(phi) = 2.2 cos / (1.2 |cos| + 1), knot rate f/R * 2pi/6, half width
// 0.045 R w / 2, depth (0.35 flat + 1 - flat) W, lane scale sqrt(3)/1.6875,
// separation lerp(2.4, 1.6, kt) / S0, fades, tail relax/cinch, knot mask),
// independently of the operator's code. Covered: registry contract, lane
// assignment by projection rank and clump ids (nearest root and the
// nearestGuide plane), weave positions at known phases, fade envelope,
// shorter strands sampling the braid at u * Lf / Lg, tail relax + cinch +
// bundle slot, knot-mask zeros at crossings, flyaways keyed by strandIndex,
// no-guide pass-through, determinism (repeat cooks, 1 vs 4 threads).
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace usdGen;

static int failures = 0;
static void Check(bool v, std::string const &s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s.c_str()); }
    else std::printf("ok: %s\n", s.c_str());
}

static bool Near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

constexpr double kPi = 3.14159265358979323846;

// --- spec formulas (double precision, independent of the operator) ---------

static double LaneF(double phi)
{
    double const c = std::cos(phi);
    return 2.2 * c / (1.2 * std::fabs(c) + 1.0);
}
static double LaneK() { return std::sqrt(3.0) / (LaneF(0.0) - LaneF(2.0 * kPi / 3.0)); }
static double MinSep()
{
    double best = 1e30;
    for (int i = 0; i < 1024; ++i) {
        double const base = double(i) / 1024.0 * 2.0 * kPi;
        double x[3], y[3];
        for (int l = 0; l < 3; ++l) {
            double const phi = base + double((l + 1) % 3) * 2.0 * kPi / 3.0;
            x[l] = LaneF(phi) * LaneK();
            y[l] = std::sin(2.0 * phi) * 2.0 / std::sqrt(3.0);
        }
        for (int a = 0; a < 3; ++a)
            for (int b = a + 1; b < 3; ++b)
                best = std::min(best, std::hypot(x[a] - x[b], y[a] - y[b]));
    }
    return best;
}
static double Sstep(double x)
{
    x = std::min(std::max(x, 0.0), 1.0);
    return x * x * (3.0 - 2.0 * x);
}

struct LaneSpec
{
    double R = 1.0, width = 1.0, flat = 0.55, kt = 0.5, freq = 2.0, Lg = 1.0;
    double W() const { return width * 0.045 * R * 0.5; }
    double D() const { return (flat * 0.35 + (1.0 - flat)) * W(); }
    double Rate() const { return freq / R * 2.0 * kPi / 6.0; }
    double Scale() const { return (2.4 + (1.6 - 2.4) * kt) / MinSep(); }
    // lane l (rank order) offset at strand parameter t: x along the wide
    // axis, y along the thin axis
    double Phi(int l, double t) const
    {
        static int const perm[3] = {1, 2, 0};
        return double(perm[l]) * 2.0 * kPi / 3.0 + t * Lg * Rate() + kPi / 6.0;
    }
    double X(int l, double t) const { return LaneF(Phi(l, t)) * W() * LaneK() * Scale(); }
    double Y(int l, double t) const
    {
        return std::sin(2.0 * Phi(l, t)) * D() * 2.0 / std::sqrt(3.0) * Scale();
    }
};

// --- graph fixtures -------------------------------------------------------

static UsdGenCurveSetDesc StraightSet(SdfPath const &path, UsdGenRole role,
                                      std::vector<GfVec3f> const &roots, int cvs,
                                      std::vector<float> const &lengths)
{
    UsdGenCurveSetDesc set;
    set.path = path;
    set.role = role;
    set.curveRole = TfToken(role == UsdGenRole::Reference ? "guide" : "hair");
    size_t const n = roots.size();
    set.curveVertexCounts.assign(n, cvs);
    set.points.resize(n * size_t(cvs));
    for (size_t c = 0; c < n; ++c) {
        float const step = lengths[c] / float(cvs - 1);
        for (int i = 0; i < cvs; ++i)
            set.points[c * size_t(cvs) + size_t(i)] = roots[c] + GfVec3f(0.0f, float(i) * step, 0.0f);
    }
    set.rest = set.points;
    set.widths.assign(n * size_t(cvs), 0.01f);
    set.curveId.resize(n);
    for (size_t c = 0; c < n; ++c) set.curveId[c] = uint64_t(c);
    set.skinPrim.resize(n);
    for (size_t c = 0; c < n; ++c) set.skinPrim[c] = int(c);
    set.skinPrimUv.assign(n, GfVec2f(0.2f, 0.3f));
    set.rootFrame.assign(n, GfMatrix4d(1.0));
    return set;
}

static void AddUniformInt(UsdGenCurveSetDesc *set, char const *name, std::vector<int> const &v)
{
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name);
    plane.type = UsdGenAuthoredPlaneType::Int32;
    plane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    plane.arity = 1;
    plane.intValues = VtIntArray(v.begin(), v.end());
    set->authoredPlanes.push_back(plane);
}

using Params = std::vector<std::pair<char const *, VtValue>>;

struct Scene
{
    std::vector<GfVec3f> strandRoots;
    std::vector<float> strandLengths;
    int strandCvs = 5;
    std::vector<GfVec3f> guideRoots;
    int guideCvs = 5;
    float guideLength = 1.0f;
    bool withGuides = true;
    std::vector<GfVec3f> surfaceRest;   // bound surface rest points, empty = none
    std::vector<int> strandIndex;   // authored strandIndex plane, empty = none
    std::vector<int> nearestGuide;  // authored nearestGuide plane, empty = none
    Params params;
};

static UsdGenGraphDesc MakeDesc(Scene const &scene)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;
    std::vector<float> lengths = scene.strandLengths;
    if (lengths.empty()) lengths.assign(scene.strandRoots.size(), 1.0f);
    UsdGenCurveSetDesc hair = StraightSet(SdfPath("/groom/hair"), UsdGenRole::Curves,
                                          scene.strandRoots, scene.strandCvs, lengths);
    if (!scene.strandIndex.empty()) AddUniformInt(&hair, "strandIndex", scene.strandIndex);
    if (!scene.nearestGuide.empty()) AddUniformInt(&hair, "nearestGuide", scene.nearestGuide);
    desc.curveSets.push_back(hair);
    if (scene.withGuides)
        desc.curveSets.push_back(StraightSet(
            SdfPath("/groom/guides"), UsdGenRole::Reference, scene.guideRoots, scene.guideCvs,
            std::vector<float>(scene.guideRoots.size(), scene.guideLength)));

    UsdGenNodeDesc source;
    source.path = SdfPath("/groom/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/groom/hair")};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                     {TfToken("useRest"), VtValue(true), false}};
    desc.nodes.push_back(source);

    UsdGenNodeDesc braid;
    braid.path = SdfPath("/groom/braid");
    braid.type = TfToken("UsdGenBraid");
    braid.seed = 3;
    braid.inputs = {SdfPath("/groom/source")};
    if (scene.withGuides) braid.curves = {SdfPath("/groom/guides")};
    if (!scene.surfaceRest.empty()) {
        UsdGenSurfaceDesc mesh;
        mesh.path = SdfPath("/groom/mesh");
        mesh.id = 0;
        mesh.faceVertexCounts = {int(scene.surfaceRest.size())};
        for (int i = 0; i < int(scene.surfaceRest.size()); ++i) mesh.faceVertexIndices.push_back(i);
        mesh.restPoints = VtVec3fArray(scene.surfaceRest.begin(), scene.surfaceRest.end());
        // deformed points differ: the reference length must come from rest
        mesh.points = mesh.restPoints;
        for (GfVec3f &q : mesh.points) q *= 5.0f;
        desc.surfaces.push_back(mesh);
        braid.surfaces = {mesh.path};
    }
    for (auto const &p : scene.params) braid.params.push_back({TfToken(p.first), p.second, false});
    desc.nodes.push_back(braid);
    desc.terminal = braid.path;
    return desc;
}

struct Result
{
    bool ok = false;
    std::vector<std::vector<GfVec3f>> points;   // by curveId
    std::vector<int> clumpId;                    // by curveId
    std::vector<std::vector<float>> mask;        // by curveId
    std::vector<std::string> warnings;
    UsdGenCurveBuffer buffer;
};

static Result Cook(Scene const &scene, int threads = 2)
{
    Result r;
    UsdGenGraphDesc const desc = MakeDesc(scene);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    UsdGenCompileResult compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const &e : compiled.errors) std::printf("  compile: %s\n", e.c_str());
        return r;
    }
    UsdGenScheduler scheduler(threads);
    UsdGenEvalContext ctx;
    ctx.desc = &graph.Desc();
    UsdGenRunResult run = scheduler.Run(graph, ctx, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const &e : run.diagnostics.errors) std::printf("  run: %s\n", e.c_str());
        return r;
    }
    r.warnings = run.diagnostics.warnings;
    UsdGenCurveBuffer const &out = graph.Output();
    r.buffer = out;
    size_t const R = out.totalCurves;
    if (R == 0 || out.curveId.size() != R || out.totalCvs % R != 0) return r;
    size_t const n = out.totalCvs / R;
    UsdGenPlane const *clump = nullptr, *mask = nullptr;
    for (UsdGenPlane const &p : out.extraCurve)
        if (p.name == TfToken("clumpId_0")) clump = &p;
    for (UsdGenPlane const &p : out.extraCv)
        if (p.name == TfToken("braidKnotMask")) mask = &p;
    if (!clump || clump->i.size() != R || !mask || mask->f.size() != out.totalCvs) {
        std::printf("  missing clumpId_0 / braidKnotMask planes\n");
        return r;
    }
    r.points.assign(R, {});
    r.clumpId.assign(R, -2);
    r.mask.assign(R, {});
    for (size_t c = 0; c < R; ++c) {
        size_t const id = size_t(out.curveId[c]);
        if (id >= R) return r;
        r.clumpId[id] = clump->i[c];
        for (size_t k = 0; k < n; ++k) {
            size_t const cv = c * n + k;
            r.points[id].push_back(GfVec3f(out.px[cv], out.py[cv], out.pz[cv]));
            r.mask[id].push_back(mask->f[cv]);
        }
    }
    r.ok = true;
    return r;
}

// Profile weight 1 everywhere, no fades: every CV lands on its target.
static Params FullPull(double freq = 2.0)
{
    return {{"braid:frequency", VtValue(freq)}, {"braid:referenceLength", VtValue(1.0)},
            {"braid:root", VtValue(1.0)},       {"braid:mid", VtValue(1.0)},
            {"braid:tip", VtValue(1.0)},        {"braid:fadeIn", VtValue(0.0)},
            {"braid:fadeOut", VtValue(0.0)},    {"braid:knotMask", VtValue(true)}};
}

static void Set(Params *p, char const *name, VtValue v)
{
    for (auto &q : *p)
        if (std::strcmp(q.first, name) == 0) { q.second = v; return; }
    p->push_back({name, v});
}

// --- checks -------------------------------------------------------------------

static void CheckRegistry()
{
    std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenBraid"));
    Check(op != nullptr, "UsdGenBraid is registered");
    if (!op) return;
    TfSpan<const TfToken> refs = op->ReferenceInputs();
    Check(refs.size() == 1 && refs[0] == TfToken("guides"), "Braid declares the guides reference");
    TfSpan<const TfToken> out = op->OutputPrimvars();
    bool clump = false, mask = false;
    for (TfToken const &t : out) {
        clump = clump || t == TfToken("clumpId_0");
        mask = mask || t == TfToken("braidKnotMask");
    }
    Check(out.size() == 2 && clump && mask, "Braid emits clumpId_0 and braidKnotMask");
    TfSpan<const TfToken> topo = op->TopologyParameters();
    bool hasFreq = false;
    for (TfToken const &t : topo) hasFreq = hasFreq || t == TfToken("braid:frequency");
    Check(hasFreq, "braid:frequency is a structural (capture) parameter");
}

// Three strands rooted on the x axis around one straight guide along +Y:
// guide frames N = (0,0,-1), B = (-1,0,0); the root PCA puts the wide axis on
// -X (theta = pi/2), so the projection rank runs from +x to -x, one strand per
// lane, each sitting exactly on its lane curve (slot offset 0).
static Scene ThreeStrands()
{
    Scene s;
    s.strandRoots = {GfVec3f(0.02f, 0, 0), GfVec3f(0, 0, 0), GfVec3f(-0.02f, 0, 0)};
    s.guideRoots = {GfVec3f(0, 0, 0)};
    s.params = FullPull();
    return s;
}

static void CheckWeave()
{
    Scene const scene = ThreeStrands();
    Result const r = Cook(scene);
    Check(r.ok, "three-strand braid cooks");
    if (!r.ok) return;
    LaneSpec const spec;
    Check(r.clumpId[0] == 0 && r.clumpId[1] == 1 && r.clumpId[2] == 2,
          "lanes follow the projection rank (+x -> lane 0, -x -> lane 2)");
    bool all = true;
    double worst = 0.0;
    for (int l = 0; l < 3; ++l)
        for (int j = 0; j < 5; ++j) {
            double const t = j / 4.0;
            GfVec3f const p = r.points[size_t(l)][size_t(j)];
            // P = C + A * x + X * y with A = (-1,0,0), X = (0,0,1)
            double const ex = -spec.X(l, t), ey = t, ez = spec.Y(l, t);
            double const e = std::max({std::fabs(p[0] - ex), std::fabs(p[1] - ey), std::fabs(p[2] - ez)});
            worst = std::max(worst, e);
            all = all && e < 2e-6;
        }
    Check(all, "every CV sits on its lane curve C + A f(phi) W K s + X sin(2phi) D 2/sqrt3 s (max err " +
                   std::to_string(worst) + ")");
    // Known phases: lane 2 (perm 0) at t = 0.5 has phi = pi/2 -> on the spine.
    GfVec3f const on = r.points[2][2];
    Check(Near(on[0], 0.0, 1e-6) && Near(on[1], 0.5, 1e-6) && Near(on[2], 0.0, 1e-6),
          "lane 2 crosses the guide at phi = pi/2 (t = 0.5)");
    // Lane 0 at the root: phi = 5pi/6, f = -0.9343, sin(5pi/3) = -0.866.
    double const s = spec.Scale();
    double const x0 = 2.2 * -0.8660254037844386 / (1.2 * 0.8660254037844386 + 1.0) * 0.0225 *
                      LaneK() * s;
    double const y0 = -0.8660254037844386 * 0.6425 * 0.0225 * 2.0 / std::sqrt(3.0) * s;
    GfVec3f const r0 = r.points[0][0];
    Check(Near(r0[0], -x0, 1e-6) && Near(r0[2], y0, 1e-6),
          "lane 0 root offset at phi = 5pi/6 matches the hand value");
    // Lanes 1 and 2 cross in x at t = 0.25 (phi = pi/3 + {4pi/3, 0}).
    Check(Near(r.points[1][1][0], r.points[2][1][0], 1e-6) &&
              !Near(r.points[1][1][2], r.points[2][1][2], 1e-4),
          "lanes 1 and 2 share x at the t = 0.25 crossing and separate in depth");
    // Knot mask: knot = sstep(((cos 6(u L rate + pi/6) + 1)/2 - 0.21)/0.44):
    // u = 0.25, 0.75 are crossings (mask 0); u = 0, 0.5, 1 are between knots.
    bool maskOk = true;
    for (int l = 0; l < 3; ++l)
        for (int j = 0; j < 5; ++j) {
            double const expect = (j == 1 || j == 3) ? 0.0 : 1.0;
            maskOk = maskOk && Near(r.mask[size_t(l)][size_t(j)], expect, 1e-5);
        }
    Check(maskOk, "braidKnotMask is 0 at the crossings and 1 between knots");

    Scene noMask = scene;
    Set(&noMask.params, "braid:knotMask", VtValue(false));
    Result const m = Cook(noMask);
    bool ones = m.ok;
    for (auto const &v : m.mask)
        for (float f : v) ones = ones && f == 1.0f;
    Check(ones, "braidKnotMask is all 1 when braid:knotMask is off");
    Check(m.ok && m.points == r.points, "the knot mask switch does not move points");

    // A finer sampling of the same braid: offsets at shared parameters agree.
    Scene fine = scene;
    fine.strandCvs = 9;
    fine.guideCvs = 9;
    Result const f9 = Cook(fine);
    bool fineOk = f9.ok;
    for (int l = 0; l < 3 && fineOk; ++l)
        for (int j = 0; j < 9; ++j) {
            double const t = j / 8.0;
            GfVec3f const p = f9.points[size_t(l)][size_t(j)];
            fineOk = fineOk && Near(p[0], -spec.X(l, t), 2e-6) && Near(p[2], spec.Y(l, t), 2e-6);
        }
    Check(fineOk, "9-CV braid follows the same lane formula");
}

static void CheckProfileAndReferenceLength()
{
    // Default profile (root 0, mid .5, tip 1, midShift .5) is w(u) = u:
    // P = lerp(P_in, target, u) with target = the full-pull position.
    Scene full = ThreeStrands();
    Result const target = Cook(full);
    Scene profiled = full;
    Set(&profiled.params, "braid:root", VtValue(0.0));
    Set(&profiled.params, "braid:mid", VtValue(0.5));
    Set(&profiled.params, "braid:tip", VtValue(1.0));
    Set(&profiled.params, "braid:amount", VtValue(0.8));
    Result const r = Cook(profiled);
    Check(target.ok && r.ok, "profiled braid cooks");
    if (!target.ok || !r.ok) return;
    bool ok = true;
    for (int l = 0; l < 3; ++l)
        for (int j = 0; j < 5; ++j) {
            float const w = 0.8f * float(j) / 4.0f;
            GfVec3f const in = full.strandRoots[size_t(l)] + GfVec3f(0, 0.25f * float(j), 0);
            GfVec3f const e = in * (1.0f - w) + target.points[size_t(l)][size_t(j)] * w;
            ok = ok && (r.points[size_t(l)][size_t(j)] - e).GetLength() < 1e-6f;
        }
    Check(ok, "pull weight = amount * Bezier(root 0, mid .5, tip 1) = 0.8 u");

    // referenceLength 0 -> diagonal of the bound surface's REST bounding box
    // (here |(2, 0, 1)| = sqrt 5), scaling the braid width and knot rate.
    Scene byValue = full;
    Set(&byValue.params, "braid:referenceLength", VtValue(std::sqrt(5.0)));
    Scene bySurface = full;
    Set(&bySurface.params, "braid:referenceLength", VtValue(0.0));
    bySurface.surfaceRest = {GfVec3f(-1, 0, 0), GfVec3f(1, 0, 0), GfVec3f(1, 0, 1),
                             GfVec3f(-1, 0, 1)};
    Result const a = Cook(byValue), b = Cook(bySurface);
    Check(a.ok && b.ok, "surface-bound braid cooks");
    bool same = a.ok && b.ok;
    for (int l = 0; l < 3 && same; ++l)
        for (int j = 0; j < 5; ++j)
            same = same && (a.points[size_t(l)][size_t(j)] - b.points[size_t(l)][size_t(j)])
                                   .GetLength() < 1e-6f;
    Check(same, "referenceLength 0 uses the bound surface's rest bounding-box diagonal");
    LaneSpec spec;
    spec.R = std::sqrt(5.0);
    Check(a.ok && Near(a.points[0][0][0], -spec.X(0, 0.0), 2e-6) &&
              Near(a.points[0][0][2], spec.Y(0, 0.0), 2e-6),
          "lane offsets scale with the reference length (W = 0.045 R / 2)");
    Scene unbound = full;
    Set(&unbound.params, "braid:referenceLength", VtValue(0.0));
    Result const u = Cook(unbound);
    Check(u.ok && u.points == target.points, "no surface and referenceLength 0 -> R = 1");
}

static void CheckLaneAssignment()
{
    // Six strands on the line d = s (1, 0, 0.5) around each of two guides:
    // pN = -0.5 s, pB = -s, theta = (pi - atan(4/3)) / 2, wide axis
    // A0 = -(2, 0, 1)/sqrt5 -> ascending key = descending s.
    std::vector<float> const s = {0.005f, -0.025f, 0.015f, -0.005f, 0.025f, -0.015f};
    Scene scene;
    for (int g = 0; g < 2; ++g)
        for (float v : s) scene.strandRoots.push_back(GfVec3f(float(g) + v, 0.0f, 0.5f * v));
    scene.guideRoots = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
    scene.params = FullPull();
    Result const r = Cook(scene);
    Check(r.ok, "two-guide braid cooks");
    if (!r.ok) return;
    auto expectedLane = [](float v) { return v > 0.01f ? 0 : (v > -0.01f ? 1 : 2); };
    bool ok = true;
    for (int g = 0; g < 2; ++g)
        for (size_t i = 0; i < s.size(); ++i)
            ok = ok && r.clumpId[size_t(g) * 6 + i] == 3 * g + expectedLane(s[i]);
    Check(ok, "clumpId = 3 * guide + lane, two members per lane by projection rank");

    // The upstream nearestGuide plane overrides the nearest-root rule.
    Scene forced = scene;
    forced.nearestGuide.assign(12, 1);
    forced.nearestGuide[3] = -1;   // not braided
    Result const f = Cook(forced);
    bool inG1 = f.ok;
    for (size_t c = 0; c < 12 && inG1; ++c)
        if (c != 3) inG1 = f.clumpId[c] >= 3 && f.clumpId[c] <= 5;
    Check(inG1, "nearestGuide = 1 puts every strand in guide 1's braid");
    Check(f.ok && f.clumpId[3] == -1, "nearestGuide = -1 leaves the strand unbraided (clumpId -1)");
    bool untouched = f.ok;
    for (int j = 0; j < 5 && untouched; ++j) {
        GfVec3f const p = f.points[3][size_t(j)];
        untouched = p == GfVec3f(-0.005f, 0.25f * float(j), -0.0025f);
    }
    Check(untouched, "an unbraided strand keeps its points");
    std::vector<int> count(6, 0);
    for (size_t c = 0; c < 12; ++c)
        if (f.ok && f.clumpId[c] >= 3) ++count[size_t(f.clumpId[c] - 3)];
    Check(count[0] == 4 && count[1] == 4 && count[2] == 3,
          "11 members split into lanes of floor(3 r / 11): 4, 4, 3");
}

static void CheckFade()
{
    Scene base = ThreeStrands();
    Result const flat = Cook(base);
    Scene faded = base;
    Set(&faded.params, "braid:fadeIn", VtValue(0.5));
    Set(&faded.params, "braid:fadeOut", VtValue(0.5));
    Result const r = Cook(faded);
    Check(flat.ok && r.ok, "faded braid cooks");
    if (!flat.ok || !r.ok) return;
    // fade(t) = sstep(t / fadeIn) (1 - sstep((t - (1 - fadeOut)) / fadeOut))
    double const fade[5] = {0.0, 0.5, 1.0, 0.5, 0.0};
    bool ok = true;
    for (int l = 0; l < 2; ++l)
        for (int j = 0; j < 5; ++j) {
            double const t = j / 4.0;
            double const ex = Sstep(t / 0.5) * (1.0 - Sstep((t - 0.5) / 0.5));
            ok = ok && Near(ex, fade[j], 1e-12);
            for (int d : {0, 2})
                ok = ok && Near(r.points[size_t(l)][size_t(j)][d],
                                fade[j] * flat.points[size_t(l)][size_t(j)][d], 1e-6);
        }
    Check(ok, "lane offsets scale by the fade-in/out envelope (0, .5, 1, .5, 0)");
    Check(Near(r.points[0][0][0], 0.0, 1e-7) && Near(r.points[0][4][0], 0.0, 1e-7),
          "root and tip sit on the guide when fully faded");
    // knot = fade * sstep(...): the crossings at u = .25/.75 are half faded.
    Check(Near(r.mask[0][1], 0.5, 1e-5) && Near(r.mask[0][3], 0.5, 1e-5) &&
              Near(r.mask[0][2], 1.0, 1e-6),
          "knot mask follows the fade envelope (1 - 0.5 at the faded crossings)");
}

static void CheckShortStrand()
{
    // The middle strand is half as long as the guide: CV k samples the braid
    // at index k * (Lf / Lg) * (n - 1) / (n - 1) = k / 2.
    Scene full = ThreeStrands();
    Result const ref = Cook(full);
    Scene shortScene = full;
    shortScene.strandLengths = {1.0f, 0.5f, 1.0f};
    Result const r = Cook(shortScene);
    Check(ref.ok && r.ok, "short-strand braid cooks");
    if (!ref.ok || !r.ok) return;
    auto lerp = [](GfVec3f a, GfVec3f b, float t) { return a * (1.0f - t) + b * t; };
    bool ok = true;
    for (int k = 0; k < 5; ++k) {
        int const i0 = k / 2;
        float const fr = (k % 2) ? 0.5f : 0.0f;
        GfVec3f const e = lerp(ref.points[1][size_t(i0)], ref.points[1][size_t(std::min(i0 + 1, 4))], fr);
        ok = ok && (r.points[1][size_t(k)] - e).GetLength() < 1e-6f;
    }
    Check(ok, "a strand shorter than its guide follows the braid to u * Lf / Lg");
    // knot mask at u = k/(n-1) * Lf/Lg: crossing at u = .25 -> k = 2
    Check(Near(r.mask[1][2], 0.0, 1e-5) && Near(r.mask[1][4], 1.0, 1e-5),
          "short strand knot mask is sampled at u * Lf / Lg");
}

static void CheckTail()
{
    // Strands lifted to z = 0.01 so the middle one starts off the guide.
    Scene scene = ThreeStrands();
    for (GfVec3f &p : scene.strandRoots) p[2] = 0.01f;
    Set(&scene.params, "braid:tailLength", VtValue(0.5));
    Set(&scene.params, "braid:tailRelax", VtValue(1.0));
    Set(&scene.params, "braid:tailCinch", VtValue(1.0));
    Result const r = Cook(scene);
    Check(r.ok, "tail braid cooks");
    if (!r.ok) return;
    // tailStart 0.5, tailFade max(0.04, fadeOut) = 0.04.
    // t = 1: relax weight 1 - sstep((1 - .5) / .5) = 0 -> the tip stays put.
    bool tips = true;
    for (int l = 0; l < 3; ++l) {
        GfVec3f const in = scene.strandRoots[size_t(l)] + GfVec3f(0, 1, 0);
        tips = tips && r.points[size_t(l)][4] == in;
    }
    Check(tips, "tail relax 1 leaves the tips at their input positions");
    // t = 0.75: weight 0.5, tail blend 1, lane fade 0, cinch back to 1:
    // middle strand target = C (bundle slot 0) -> z = lerp(0.01, 0, 0.5).
    GfVec3f const mid = r.points[1][3];
    Check(Near(mid[0], 0.0, 1e-7) && Near(mid[1], 0.75, 1e-6) && Near(mid[2], 0.005, 1e-7),
          "tail relax halves the pull at t = 0.75 (bundle centre)");
    // Outer strand: bundle slot u' = -sqrt(3) W along A = (-1,0,0).
    double const W = 0.0225, us = std::sqrt(3.0) * W;
    GfVec3f const outer = r.points[0][3];
    Check(Near(outer[0], 0.5 * us + 0.5 * 0.02, 1e-6) && Near(outer[2], 0.005, 1e-6),
          "outer strand relaxes toward its sqrt(3) W bundle slot");
    // t = 0.5: tail blend 1, fade 0, cinch = lerp(1, 0.02, 1) = 0.02, weight 1
    // -> the bundle is cinched to 2 % of its radius.
    GfVec3f const cinched = r.points[0][2];
    Check(Near(cinched[0], 0.02 * us, 1e-6) && Near(cinched[1], 0.5, 1e-6) &&
              Near(cinched[2], 0.0, 1e-6),
          "tail cinch shrinks the bundle at the tail start");
    Check(Near(r.points[1][2][2], 0.0, 1e-7), "middle strand is on the spine at the cinch");
}

static void CheckFlyaways()
{
    // strandIndex reversed: curve c has index 5 - c. u01(splitmix(s)) for
    // s = 0..5: .883 .567 .591 .113 .431 .387 -> fraction 0.4 picks s = 3, 5,
    // i.e. curves 2 and 0.
    Scene scene;
    std::vector<float> const s = {0.025f, 0.015f, 0.005f, -0.005f, -0.015f, -0.025f};
    for (float v : s) scene.strandRoots.push_back(GfVec3f(v, 0.0f, 0.5f * v));
    scene.guideRoots = {GfVec3f(0, 0, 0)};
    scene.strandIndex = {5, 4, 3, 2, 1, 0};
    scene.params = FullPull();
    Set(&scene.params, "braid:flyaway:fraction", VtValue(0.4));
    Set(&scene.params, "braid:flyaway:amount", VtValue(0.0));
    Result const r = Cook(scene);
    Check(r.ok, "flyaway braid cooks");
    if (!r.ok) return;
    double const u01[6] = {0.883311, 0.566562, 0.591190, 0.113450, 0.431456, 0.386768};
    bool ids = true, still = true, masks = true;
    for (int c = 0; c < 6; ++c) {
        bool const fly = 0.4 > u01[5 - c];
        ids = ids && (fly ? r.clumpId[size_t(c)] == -1 : r.clumpId[size_t(c)] == c / 2);
        for (int j = 0; j < 5; ++j) {
            GfVec3f const in = scene.strandRoots[size_t(c)] + GfVec3f(0, 0.25f * float(j), 0);
            if (fly) {
                still = still && r.points[size_t(c)][size_t(j)] == in;
                masks = masks && r.mask[size_t(c)][size_t(j)] == 1.0f;
            } else if (j > 0) {
                still = still && r.points[size_t(c)][size_t(j)] != in;
            }
        }
    }
    Check(ids, "flyaways (hash01(strandIndex) < fraction) get clumpId -1, others 3g + lane");
    Check(still, "flyaway amount 0 leaves flyaways in place; members are woven");
    Check(masks, "flyaways keep knot mask 1");

    Scene none = scene;
    Set(&none.params, "braid:flyaway:fraction", VtValue(0.0));
    Scene full = scene;
    Set(&full.params, "braid:flyaway:amount", VtValue(1.0));
    Result const a = Cook(none), b = Cook(full);
    Check(a.ok && b.ok && a.points == b.points,
          "flyaway amount 1 weaves flyaways like members");
    Check(b.ok && b.clumpId[0] == -1 && b.clumpId[2] == -1 && b.clumpId[1] == 0,
          "flyaways lose their clump id even when woven");
    Check(a.ok && a.clumpId[0] == 0 && a.clumpId[5] == 2, "no flyaways at fraction 0");
}

static void CheckNoGuides()
{
    Scene scene = ThreeStrands();
    scene.withGuides = false;
    Result const r = Cook(scene);
    Check(r.ok, "a braid without guides cooks (pass-through)");
    if (!r.ok) return;
    bool same = true;
    for (int c = 0; c < 3; ++c) {
        same = same && r.clumpId[size_t(c)] == -1;
        for (int j = 0; j < 5; ++j) {
            same = same && r.points[size_t(c)][size_t(j)] ==
                               scene.strandRoots[size_t(c)] + GfVec3f(0, 0.25f * float(j), 0);
            same = same && r.mask[size_t(c)][size_t(j)] == 1.0f;
        }
    }
    Check(same, "without guides: points unchanged, clumpId -1, mask 1");
    Check(!r.warnings.empty(), "without guides the braid warns");
}

static void CheckDefaultsAndDeterminism()
{
    // A larger groom (> 2000 strands -> parallel per-guide path) with default
    // parameters, tail, spread and flyaways.
    Scene scene;
    for (int g = 0; g < 8; ++g) scene.guideRoots.push_back(GfVec3f(float(g) * 0.2f, 0, 0));
    for (int i = 0; i < 2400; ++i) {
        float const a = float(i) * 2.399963f;
        float const rr = 0.05f * std::sqrt(float(i % 300) / 300.0f);
        scene.strandRoots.push_back(GfVec3f(float(i % 8) * 0.2f + rr * std::cos(a), 0.0f,
                                            rr * std::sin(a)));
    }
    scene.strandCvs = 12;
    scene.guideCvs = 7;
    scene.params = {{"braid:tailLength", VtValue(0.3)}, {"braid:rootSpread", VtValue(0.5)},
                    {"braid:tipSpread", VtValue(0.5)},  {"braid:flyaway:fraction", VtValue(0.2)},
                    {"braid:flyaway:amount", VtValue(0.3)}, {"braid:knotMask", VtValue(true)}};
    Result const a = Cook(scene, 1), b = Cook(scene, 4), c = Cook(scene, 4);
    Check(a.ok && b.ok && c.ok, "2400-strand braid cooks with 1 and 4 threads");
    if (!a.ok || !b.ok || !c.ok) return;
    auto bytesEqual = [](Result const &x, Result const &y) {
        return x.buffer.px == y.buffer.px && x.buffer.py == y.buffer.py &&
               x.buffer.pz == y.buffer.pz && x.clumpId == y.clumpId && x.mask == y.mask;
    };
    Check(bytesEqual(b, c), "two cooks are identical");
    Check(bytesEqual(a, b), "1-thread and 4-thread cooks are identical");
    int moved = 0, ids = 0;
    bool finite = true;
    for (size_t s = 0; s < a.points.size(); ++s) {
        if (a.clumpId[s] >= 0) ++ids;
        for (GfVec3f const &p : a.points[s])
            finite = finite && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        GfVec3f const tipIn = scene.strandRoots[s] + GfVec3f(0, 1, 0);
        if ((a.points[s].back() - tipIn).GetLength() > 1e-5f) ++moved;
    }
    Check(finite, "all points finite");
    Check(ids > 1700 && ids < 2100, "about 80 % of strands keep a braid clump id (" +
                                        std::to_string(ids) + ")");
    Check(moved > 1500, "default profile pulls the tips into the braid (" + std::to_string(moved) + ")");
}

int main()
{
    usdGenRegisterM1Operators();
    CheckRegistry();
    CheckWeave();
    CheckProfileAndReferenceLength();
    CheckLaneAssignment();
    CheckFade();
    CheckShortStrand();
    CheckTail();
    CheckFlyaways();
    CheckNoGuides();
    CheckDefaultsAndDeterminism();
    std::printf("testUsdGenBraid: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
