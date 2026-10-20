// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// UsdGenTwist and UsdGenFrizz: CurveSource-fed CPU cooks over hand-built
// strands with authored upstream planes (primaryGuide, clumpId_0, mirrorSign,
// strandSeed, strandIndex, restLength, braidKnotMask), checked against values
// computed by hand from the reference formulas:
//   Twist: per-point absolute angle from the root/mid/tip Bezier profile over
//          the group mean curve, amount multiplier, guide vs clump grouping,
//          mirror sign (first strand in strandIndex order), value-noise
//          angle scaling keyed by strandSeed, root re-snap with falloff.
//   Frizz: wave and value-noise offsets (sharpness 0 and 1) in the transport
//          frame, profile weighting (midShift 0..1), restLength as the
//          reference length, clump-shared seeds and lengths, knot-mask
//          attenuation; the root never moves; cooks are deterministic.
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/opRegistry.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <cstdio>
#include <string>
#include <vector>

using namespace usdGen;

static int failures = 0;
static void Check(bool v, std::string const &s)
{
    if (!v) { ++failures; std::printf("FAIL: %s\n", s.c_str()); }
    else std::printf("ok: %s\n", s.c_str());
}
static bool Near(float a, float b, float tol = 2e-5f) { return std::fabs(a - b) <= tol; }

constexpr double kPi = 3.14159265358979323846;

// --- reference formulas, restated independently of the engine -------------

static uint64_t SM(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static double U01(uint64_t z) { return double(z >> 11) / 9007199254740992.0; }

// 1-D value noise: lattice 2*U01(SM(j ^ key)) - 1, smoothstep or linear blend.
static double VNoise(double x, uint64_t key, double sharp)
{
    double const i = std::floor(x), f = x - i;
    double w = f * f * (3.0 - 2.0 * f);
    if (sharp > 0.0) w = (1.0 - sharp) * w + sharp * f;
    double const a = 2.0 * U01(SM(uint64_t(int64_t(i)) ^ key)) - 1.0;
    double const b = 2.0 * U01(SM(uint64_t(int64_t(i) + 1) ^ key)) - 1.0;
    return a * (1.0 - w) + b * w;
}

// Quadratic Bezier profile (0,r) (m,mid) (1,tip) at t: solve 2m u + (1-2m) u^2 = t.
static double Bezier(double t, double r, double mid, double m, double tip)
{
    double u;
    double const a = 1.0 - 2.0 * m;
    if (std::fabs(a) < 1e-9) u = m > 0 ? t / (2.0 * m) : t;
    else u = (-2.0 * m + std::sqrt(4.0 * m * m + 4.0 * a * t)) / (2.0 * a);
    return (1 - u) * (1 - u) * r + 2 * (1 - u) * u * mid + u * u * tip;
}

// --- graph fixture ---------------------------------------------------------

struct Planes
{
    std::vector<int> primaryGuide, clumpId, mirrorSign, strandSeed, strandIndex;
    std::vector<float> restLength, knotMask;
};

static void AddInt(UsdGenCurveSetDesc &set, char const *name, std::vector<int> const &v)
{
    if (v.empty()) return;
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Int32;
    p.domain = UsdGenAuthoredPlaneDomain::Primitive;
    p.intValues.assign(v.begin(), v.end());
    set.authoredPlanes.push_back(p);
}
static void AddFloat(UsdGenCurveSetDesc &set, char const *name, std::vector<float> const &v,
                     bool vertex)
{
    if (v.empty()) return;
    UsdGenAuthoredPlaneDesc p;
    p.name = TfToken(name);
    p.type = UsdGenAuthoredPlaneType::Float32;
    p.domain = vertex ? UsdGenAuthoredPlaneDomain::Point : UsdGenAuthoredPlaneDomain::Primitive;
    p.floatValues.assign(v.begin(), v.end());
    set.authoredPlanes.push_back(p);
}

// Straight strands along +Y from the given roots: cvs points, `step` apart.
static UsdGenCurveSetDesc Strands(std::vector<GfVec3f> const &roots, int cvs, float step,
                                  Planes const &planes)
{
    UsdGenCurveSetDesc set;
    set.path = SdfPath("/groom/hair");
    set.role = UsdGenRole::Curves;
    set.curveRole = TfToken("hair");
    size_t const n = roots.size();
    set.curveVertexCounts.assign(n, cvs);
    for (size_t c = 0; c < n; ++c)
        for (int i = 0; i < cvs; ++i)
            set.points.push_back(roots[c] + GfVec3f(0.0f, float(i) * step, 0.0f));
    set.rest = set.points;
    set.widths.assign(n * size_t(cvs), 0.01f);
    for (size_t c = 0; c < n; ++c) {
        set.curveId.push_back(uint64_t(c));
        set.skinPrim.push_back(int(c));
        set.skinPrimUv.push_back(GfVec2f(0.2f, 0.3f));
        set.rootFrame.push_back(GfMatrix4d(1.0));
    }
    AddInt(set, "primaryGuide", planes.primaryGuide);
    AddInt(set, "clumpId_0", planes.clumpId);
    AddInt(set, "mirrorSign", planes.mirrorSign);
    AddInt(set, "strandSeed", planes.strandSeed);
    AddInt(set, "strandIndex", planes.strandIndex);
    AddFloat(set, "restLength", planes.restLength, false);
    AddFloat(set, "braidKnotMask", planes.knotMask, true);
    return set;
}

using Params = std::vector<std::pair<char const *, VtValue>>;

static UsdGenGraphDesc Graph(UsdGenCurveSetDesc const &curves, char const *opType,
                             Params const &params, UsdGenSurfaceDesc const *surface = nullptr)
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/groom");
    desc.defaultWidth = 0.01f;
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/groom/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false},
                     {TfToken("useRest"), VtValue(true), false}};
    desc.nodes.push_back(source);
    UsdGenNodeDesc op;
    op.path = SdfPath("/groom/op");
    op.type = TfToken(opType);
    op.seed = 3;
    op.inputs = {source.path};
    for (auto const &p : params) op.params.push_back({TfToken(p.first), p.second, false});
    if (surface) {
        desc.surfaces.push_back(*surface);
        op.surfaces = {surface->path};
    }
    desc.nodes.push_back(op);
    desc.terminal = op.path;
    return desc;
}

static bool Cook(UsdGenGraphDesc const &desc, UsdGenCurveBuffer *out)
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
    *out = graph.Output();
    return true;
}

static GfVec3f Pt(UsdGenCurveBuffer const &b, size_t cv) { return GfVec3f(b.px[cv], b.py[cv], b.pz[cv]); }

static bool SameBits(UsdGenCurveBuffer const &a, UsdGenCurveBuffer const &b)
{
    if (a.px.size() != b.px.size()) return false;
    for (size_t i = 0; i < a.px.size(); ++i)
        if (a.px[i] != b.px[i] || a.py[i] != b.py[i] || a.pz[i] != b.pz[i]) return false;
    return true;
}

static VtValue F(double v) { return VtValue(float(v)); }
static VtValue B(bool v) { return VtValue(v); }

// --- registry ----------------------------------------------------------------

static void TestRegistry()
{
    UsdGenOpRegistry &registry = UsdGenOpRegistry::Get();
    for (char const *name : {"UsdGenTwist", "UsdGenFrizz"}) {
        TfToken const type(name);
        std::unique_ptr<UsdGenOp> op = registry.Create(type);
        Check(registry.HasKernel(type) && op && op->Type() == type, std::string(name) + " registered");
        if (!op) continue;
        Check(op->TopologyEffect() == UsdGenTopoFx::None && op->OutputPrimvars().empty(),
              std::string(name) + " keeps topology and emits no planes");
    }
}

// --- twist -------------------------------------------------------------------

// Angle (deg) of the offset of `p` from the axis line x = cx, z = 0 (the
// strands grow along +Y, so the twist axis is +Y): v' = (vx cos, 0, -vx sin).
static double AngleAbout(GfVec3f const &p, double cx, double vx)
{
    double const ox = (p[0] - cx) / vx, oz = -p[2] / vx;
    return std::atan2(oz, ox) * 180.0 / kPi;
}

static void TestTwistProfile()
{
    Planes pl;
    pl.primaryGuide = {0, 0};
    auto curves = Strands({GfVec3f(0.1f, 0, 0), GfVec3f(-0.1f, 0, 0)}, 5, 0.25f, pl);
    struct Case { double ms, amount; char const *label; };
    for (Case const &cs : {Case{0.5, 1.0, "midShift 0.5"}, Case{0.25, 1.0, "midShift 0.25"},
                           Case{0.5, 1.5, "amount 1.5"}}) {
        UsdGenCurveBuffer out;
        bool const ok = Cook(Graph(curves, "UsdGenTwist",
                                   {{"twist:root", F(10)}, {"twist:mid", F(40)},
                                    {"twist:tip", F(90)}, {"twist:midShift", F(cs.ms)},
                                    {"twist:amount", F(cs.amount)}}), &out);
        Check(ok && out.px.size() == 10, std::string("twist cook (") + cs.label + ")");
        if (!ok || out.px.size() != 10) continue;
        bool angles = true, radius = true, height = true;
        for (int k = 0; k < 5; ++k) {
            double const t = k / 4.0;
            double const expected = Bezier(t, 10, 40, cs.ms, 90) * cs.amount;
            GfVec3f const a = Pt(out, size_t(k)), b = Pt(out, size_t(5 + k));
            angles = angles && std::fabs(AngleAbout(a, 0.0, 0.1) - expected) < 1e-3 &&
                     std::fabs(AngleAbout(b, 0.0, -0.1) - expected) < 1e-3;
            radius = radius && Near(std::hypot(a[0], a[2]), 0.1f) && Near(std::hypot(b[0], b[2]), 0.1f);
            height = height && Near(a[1], 0.25f * float(k)) && Near(b[1], 0.25f * float(k));
            if (k == 0 || k == 2 || k == 4)
                std::printf("  %s k=%d angle %.4f expected %.4f\n", cs.label, k,
                            AngleAbout(a, 0.0, 0.1), expected);
        }
        Check(angles, std::string("twist angle at every point = Bezier profile x amount (") + cs.label + ")");
        Check(radius && height, std::string("twist is a rotation about the mean curve (") + cs.label + ")");
    }

    // Gate: amount 0 or all-zero profile leaves the strands untouched.
    UsdGenCurveBuffer off, zero;
    bool ok = Cook(Graph(curves, "UsdGenTwist", {{"twist:tip", F(90)}, {"twist:amount", F(0)}}), &off) &&
              Cook(Graph(curves, "UsdGenTwist", {}), &zero);
    Check(ok && Near(off.px[4], 0.1f) && off.pz[4] == 0.0f && zero.pz[4] == 0.0f,
          "twist gate: amount 0 / zero profile is a no-op");
}

static void TestTwistGrouping()
{
    // s0..s3 pair by guide (0: x=0.1,-0.1; 1: x=1.1,0.9) and cross-pair by
    // clump (0: s0,s2; 1: s1,s3). s4 is alone in guide 2, s5 has no group.
    Planes pl;
    pl.primaryGuide = {0, 0, 1, 1, 2, -1};
    pl.clumpId = {0, 1, 0, 1, -1, -1};
    std::vector<GfVec3f> const roots = {GfVec3f(0.1f, 0, 0), GfVec3f(-0.1f, 0, 0),
                                        GfVec3f(1.1f, 0, 0), GfVec3f(0.9f, 0, 0),
                                        GfVec3f(5, 0, 0), GfVec3f(7, 0, 0)};
    auto curves = Strands(roots, 3, 0.5f, pl);
    Params const base = {{"twist:root", F(30)}, {"twist:mid", F(30)}, {"twist:tip", F(30)}};
    double const c30 = std::cos(kPi / 6), s30 = std::sin(kPi / 6);

    UsdGenCurveBuffer guide, clump;
    Params withClumps = base;
    withClumps.push_back({"twist:useClumpIds", B(true)});
    bool const ok = Cook(Graph(curves, "UsdGenTwist", base), &guide) &&
                    Cook(Graph(curves, "UsdGenTwist", withClumps), &clump);
    Check(ok, "twist grouping cooks");
    if (!ok) return;
    // Guide groups: centres x = 0 and x = 1, offsets +-0.1.
    bool g = true;
    for (int k = 0; k < 3; ++k) {
        g = g && Near(guide.px[size_t(k)], float(0.1 * c30)) && Near(guide.pz[size_t(k)], float(-0.1 * s30));
        g = g && Near(guide.px[size_t(6 + k)], float(1.0 + 0.1 * c30)) &&
            Near(guide.pz[size_t(6 + k)], float(-0.1 * s30));
    }
    Check(g, "useClumpIds off: strands twist about their primaryGuide group mean");
    // Clump groups: clump 0 = {0.1, 1.1} -> centre 0.6, s0 offset -0.5.
    bool cl = true;
    for (int k = 0; k < 3; ++k) {
        cl = cl && Near(clump.px[size_t(k)], float(0.6 - 0.5 * c30)) &&
             Near(clump.pz[size_t(k)], float(0.5 * s30));
        cl = cl && Near(clump.px[size_t(6 + k)], float(0.6 + 0.5 * c30)) &&
             Near(clump.pz[size_t(6 + k)], float(-0.5 * s30));
        cl = cl && Near(clump.px[size_t(3 + k)], float(0.4 - 0.5 * c30));  // clump 1 = {-0.1, 0.9}
    }
    Check(cl, "useClumpIds on: strands twist about their clump mean");
    bool still = true;
    for (int k = 0; k < 3; ++k)
        for (int s : {4, 5})
            for (UsdGenCurveBuffer const *b : {&guide, &clump})
                still = still && b->px[size_t(3 * s + k)] == roots[size_t(s)][0] && b->pz[size_t(3 * s + k)] == 0.0f;
    Check(still, "a single-strand group and an ungrouped strand are not twisted");
}

static void TestTwistMirror()
{
    Params const p = {{"twist:root", F(30)}, {"twist:mid", F(30)}, {"twist:tip", F(30)}};
    for (int repIsMirrored = 0; repIsMirrored < 2; ++repIsMirrored) {
        Planes pl;
        pl.primaryGuide = {0, 0};
        pl.mirrorSign = {1, -1};
        // The group's representative is the first strand in strandIndex order.
        pl.strandIndex = repIsMirrored ? std::vector<int>{1, 0} : std::vector<int>{0, 1};
        auto curves = Strands({GfVec3f(0.1f, 0, 0), GfVec3f(-0.1f, 0, 0)}, 3, 0.5f, pl);
        UsdGenCurveBuffer out;
        bool const ok = Cook(Graph(curves, "UsdGenTwist", p), &out);
        double const expected = repIsMirrored ? -30.0 : 30.0;
        bool good = ok;
        for (int k = 0; ok && k < 3; ++k)
            good = good && std::fabs(AngleAbout(Pt(out, size_t(k)), 0.0, 0.1) - expected) < 1e-3 &&
                   std::fabs(AngleAbout(Pt(out, size_t(3 + k)), 0.0, -0.1) - expected) < 1e-3;
        Check(good, repIsMirrored ? "mirrorSign -1 on the representative strand reverses the twist"
                                  : "mirrorSign of a non-representative strand is ignored");
    }
}

static void TestTwistNoise()
{
    Planes pl;
    pl.primaryGuide = {0, 0};
    pl.strandSeed = {11, 12};
    int const n = 9;
    auto curves = Strands({GfVec3f(0.1f, 0, 0), GfVec3f(-0.1f, 0, 0)}, n, 0.125f, pl);
    double const noise = 0.5;
    Params const p = {{"twist:root", F(10)}, {"twist:mid", F(20)}, {"twist:tip", F(40)},
                      {"twist:noise", F(noise)}};
    UsdGenCurveBuffer out, again;
    bool const ok = Cook(Graph(curves, "UsdGenTwist", p), &out) &&
                    Cook(Graph(curves, "UsdGenTwist", p), &again);
    Check(ok, "twist noise cooks");
    if (!ok) return;
    uint64_t const G = 0x9E3779B97F4A7C15ull;
    auto expectedAngle = [&](uint32_t seed, int k) {
        uint64_t const h = SM(uint64_t(seed) ^ 0x7477697374ull);
        double const v0 = U01(SM(h));
        double const x = k * (4.0 / (n - 1));
        double const c = std::floor(x);
        double f = x - c;
        f = f * f * (3 - 2 * f);
        uint64_t const z = uint64_t(int64_t(c)) * G + h;
        double const a = U01(SM(z)), b = U01(SM(z + G));
        double const nz = ((2 * a - 1) * (1 - f) + (2 * b - 1) * f) - (2 * v0 - 1);
        return (1.0 + noise * nz) * Bezier(k / double(n - 1), 10, 20, 0.5, 40);
    };
    bool match = true, differs = false;
    for (int k = 0; k < n; ++k) {
        double const a = AngleAbout(Pt(out, size_t(k)), 0.0, 0.1);
        double const b = AngleAbout(Pt(out, size_t(n + k)), 0.0, -0.1);
        match = match && std::fabs(a - expectedAngle(11, k)) < 2e-3 &&
                std::fabs(b - expectedAngle(12, k)) < 2e-3;
        differs = differs || std::fabs(a - Bezier(k / double(n - 1), 10, 20, 0.5, 40)) > 0.5;
        if (k == 4) std::printf("  noise mid angle %.4f expected %.4f\n", a, expectedAngle(11, k));
    }
    Check(match, "twist noise: angle * (1 + noise * nz) keyed by strandSeed ^ 'twist'");
    Check(differs, "twist noise changes the angles away from the root");
    Check(std::fabs(AngleAbout(Pt(out, 0), 0.0, 0.1) - 10.0) < 1e-3, "twist noise is zero at the root");
    Check(SameBits(out, again), "twist noise cook is deterministic");
}

static void TestTwistResnap()
{
    // Strands in the plane z = 0 (the bound surface); a 90 degree root angle
    // swings the root to z = -0.1, the re-snap pulls it back.
    UsdGenSurfaceDesc plane;
    plane.path = SdfPath("/groom/scalp");
    plane.faceVertexCounts = {4};
    plane.faceVertexIndices = {0, 1, 2, 3};
    plane.points = {GfVec3f(-10, -10, 0), GfVec3f(10, -10, 0), GfVec3f(10, 10, 0), GfVec3f(-10, 10, 0)};
    plane.restPoints = plane.points;
    Planes pl;
    pl.primaryGuide = {0, 0};
    auto curves = Strands({GfVec3f(0.1f, 0, 0), GfVec3f(-0.1f, 0, 0)}, 5, 0.25f, pl);
    Params p = {{"twist:root", F(90)}, {"twist:mid", F(90)}, {"twist:tip", F(90)}};
    Params noSnap = p;
    noSnap.push_back({"twist:resnapRoot", B(false)});
    UsdGenCurveBuffer snapped, free, unbound;
    bool const ok = Cook(Graph(curves, "UsdGenTwist", p, &plane), &snapped) &&
                    Cook(Graph(curves, "UsdGenTwist", noSnap, &plane), &free) &&
                    Cook(Graph(curves, "UsdGenTwist", p), &unbound);
    Check(ok, "twist re-snap cooks");
    if (!ok) return;
    Check(Near(free.pz[0], -0.1f) && Near(unbound.pz[0], -0.1f) && Near(free.pz[1], -0.1f),
          "without re-snap (or without a surface) the root swings off the surface");
    // delta = (0,0,0.1); F = min(4|delta|, L/2) = 0.4; s1 = 0.25 -> w = 1 - smoothstep(0.625).
    double const x = 0.625, w1 = 1.0 - x * x * (3 - 2 * x);
    Check(Near(snapped.pz[0], 0.0f) && Near(snapped.px[0], 0.0f),
          "re-snap puts the root on the closest surface point");
    Check(Near(snapped.pz[1], float(-0.1 + 0.1 * w1)) && Near(snapped.pz[2], -0.1f) &&
              Near(snapped.pz[4], -0.1f),
          "re-snap falls off smoothly within min(4|delta|, L/2) of the root");
}

// --- frizz -------------------------------------------------------------------

struct FrizzExpect
{
    double amplitude = 0, wavesX = 0, wavesZ = 0, noise = 0, noiseSize = 0.1, sharp = 0;
    double root = 1, mid = 1, ms = 0.5, tip = 1;
    bool shared = false;
};

// Expected offset at arc length `arc` of a straight +Y strand: frame
// N = T x (1,0,0) = (0,0,-1), B = T x N = (-1,0,0).
static GfVec3f FrizzOffset(FrizzExpect const &e, uint32_t seed, double refL, double arc,
                           double mask = 1.0)
{
    double const u = arc / refL;
    double const w = std::min(1.0, std::max(0.0, Bezier(std::min(1.0, u), e.root, e.mid, e.ms, e.tip))) * mask;
    if (w <= 1e-4) return GfVec3f(0.0f);
    uint64_t const key = uint64_t(int64_t(int32_t(seed)));
    double const phi = U01(SM(key * 0x8DA6B343ull)) * 2 * kPi;
    double const rot = e.shared ? 0.0 : U01(SM(key * 0xD8163841ull)) * 2 * kPi;
    double const o1 = U01(SM(key * 0x1B873593ull)) * 1000.0;
    double const o2 = U01(SM(key * 0x9E3779B9ull)) * 1000.0;
    double wx = e.wavesX > 0 ? std::sin(u * e.wavesX * 2 * kPi + phi) - std::sin(phi) : 0.0;
    double wz = e.wavesZ > 0 ? std::sin(u * e.wavesZ * 2 * kPi + phi + kPi / 2) - std::sin(phi + kPi / 2) : 0.0;
    if (e.noise > 0) {
        double const x = u * refL / e.noiseSize;
        wx += e.noise * (2 * VNoise(x + o1, 0x9E3779B97F4A7C15ull, e.sharp) -
                         2 * VNoise(o1, 0x9E3779B97F4A7C15ull, e.sharp));
        wz += e.noise * (2 * VNoise(x + o2, 0x517CC1B727220A95ull, e.sharp) -
                         2 * VNoise(o2, 0x517CC1B727220A95ull, e.sharp));
    }
    double const ox = wx * std::cos(rot) - wz * std::sin(rot);
    double const oz = wx * std::sin(rot) + wz * std::cos(rot);
    double const s = w * e.amplitude * refL;
    return GfVec3f(float(-s * oz), 0.0f, float(-s * ox));
}

static Params FrizzParams(FrizzExpect const &e)
{
    return {{"frizz:amplitude", F(e.amplitude)}, {"frizz:wavesX", F(e.wavesX)},
            {"frizz:wavesZ", F(e.wavesZ)},       {"frizz:noise", F(e.noise)},
            {"frizz:noiseSize", F(e.noiseSize)}, {"frizz:sharpness", F(e.sharp)},
            {"frizz:root", F(e.root)},           {"frizz:mid", F(e.mid)},
            {"frizz:midShift", F(e.ms)},         {"frizz:tip", F(e.tip)}};
}

// Compares curve `c` (n CVs, `step` apart, root at `root`) against the expected
// offsets; the root must be untouched.
static bool MatchStrand(UsdGenCurveBuffer const &out, size_t c, int n, float step, GfVec3f root,
                        FrizzExpect const &e, uint32_t seed, double refL,
                        std::vector<float> const &mask = {}, float tol = 2e-5f)
{
    bool good = out.px.size() >= (c + 1) * size_t(n);
    for (int k = 0; good && k < n; ++k) {
        size_t const cv = c * size_t(n) + size_t(k);
        GfVec3f const base = root + GfVec3f(0.0f, step * float(k), 0.0f);
        GfVec3f const expected =
            k == 0 ? GfVec3f(0.0f)
                   : FrizzOffset(e, seed, refL, double(step) * k, mask.empty() ? 1.0 : mask[size_t(k)]);
        GfVec3f const got = Pt(out, cv) - base;
        good = good && Near(got[0], expected[0], tol) && Near(got[1], expected[1], tol) &&
               Near(got[2], expected[2], tol);
        if (!good)
            std::printf("  curve %zu k=%d got (%g %g %g) expected (%g %g %g)\n", c, k, got[0], got[1],
                        got[2], expected[0], expected[1], expected[2]);
    }
    return good;
}

static void TestFrizzWaves()
{
    Planes pl;
    pl.strandSeed = {7};
    int const n = 9;
    float const step = 0.125f;
    auto curves = Strands({GfVec3f(0.3f, 0, 0.2f)}, n, step, pl);
    FrizzExpect e;
    e.amplitude = 0.05; e.wavesX = 1.5; e.wavesZ = 0.5;
    UsdGenCurveBuffer out, again;
    bool const ok = Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &out) &&
                    Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &again);
    Check(ok, "frizz waves cook");
    if (!ok) return;
    Check(MatchStrand(out, 0, n, step, GfVec3f(0.3f, 0, 0.2f), e, 7, 1.0),
          "frizz waves only: offsets = amplitude * refL * (x N + z B), root fixed");
    Check(SameBits(out, again), "frizz cook is deterministic");
    float maxOff = 0.0f;
    for (int k = 0; k < n; ++k)
        maxOff = std::max(maxOff, std::fabs(out.px[size_t(k)] - 0.3f) + std::fabs(out.pz[size_t(k)] - 0.2f));
    Check(maxOff > 0.01f, "frizz waves visibly displace the strand");

    // restLength is the reference length: u = arc / 2, amplitude * 2.
    Planes rl = pl;
    rl.restLength = {2.0f};
    UsdGenCurveBuffer longer;
    bool const ok2 = Cook(Graph(Strands({GfVec3f(0.3f, 0, 0.2f)}, n, step, rl), "UsdGenFrizz",
                                FrizzParams(e)), &longer);
    Check(ok2 && MatchStrand(longer, 0, n, step, GfVec3f(0.3f, 0, 0.2f), e, 7, 2.0),
          "frizz uses the upstream restLength as the reference length");

    // Gates: no waves and no noise, or a negligible amplitude, is a no-op.
    FrizzExpect none = e;
    none.wavesX = none.wavesZ = 0.0;
    FrizzExpect tiny = e;
    tiny.amplitude = 1e-6;
    UsdGenCurveBuffer a, b;
    bool const ok3 = Cook(Graph(curves, "UsdGenFrizz", FrizzParams(none)), &a) &&
                     Cook(Graph(curves, "UsdGenFrizz", FrizzParams(tiny)), &b);
    bool same = ok3;
    for (int k = 0; ok3 && k < n; ++k)
        same = same && a.px[size_t(k)] == 0.3f && a.pz[size_t(k)] == 0.2f &&
               b.px[size_t(k)] == 0.3f && b.pz[size_t(k)] == 0.2f;
    Check(same, "frizz gates: no waves/noise or amplitude <= 2.5e-5 leaves strands untouched");
}

static void TestFrizzNoise()
{
    Planes pl;
    pl.strandSeed = {21};
    int const n = 17;
    float const step = 1.0f / 16.0f;
    auto curves = Strands({GfVec3f(0, 0, 0)}, n, step, pl);
    UsdGenCurveBuffer smooth, sharp;
    for (int s = 0; s < 2; ++s) {
        FrizzExpect e;
        e.amplitude = 0.05; e.noise = 0.8; e.noiseSize = 0.25; e.sharp = double(s);
        UsdGenCurveBuffer &out = s ? sharp : smooth;
        bool const ok = Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &out);
        Check(ok && MatchStrand(out, 0, n, step, GfVec3f(0.0f), e, 21, 1.0),
              s ? "frizz noise only, sharpness 1 (piecewise linear) matches the formula"
                : "frizz noise only, sharpness 0 (smoothstep) matches the formula");
    }
    bool differ = false;
    for (int k = 1; k < n; ++k)
        differ = differ || std::fabs(smooth.px[size_t(k)] - sharp.px[size_t(k)]) > 1e-5f ||
                 std::fabs(smooth.pz[size_t(k)] - sharp.pz[size_t(k)]) > 1e-5f;
    Check(differ, "frizz sharpness changes the noise shape");
}

static void TestFrizzProfile()
{
    Planes pl;
    pl.strandSeed = {5};
    int const n = 9;
    float const step = 0.125f;
    auto curves = Strands({GfVec3f(0, 0, 0)}, n, step, pl);
    struct Case { double root, mid, ms, tip; char const *label; };
    for (Case const &cs : {Case{0, 0.5, 0.5, 1, "default ramp (w = t)"},
                           Case{0, 1, 0.5, 0, "bump 2t(1-t)"},
                           Case{0, 1, 0.0, 0, "midShift 0 (frizz allows 0..1)"},
                           Case{0.2, 1, 0.8, 0.6, "midShift 0.8"}}) {
        FrizzExpect e;
        e.amplitude = 0.05; e.wavesX = 1.0;
        e.root = cs.root; e.mid = cs.mid; e.ms = cs.ms; e.tip = cs.tip;
        UsdGenCurveBuffer out;
        bool const ok = Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &out);
        Check(ok && MatchStrand(out, 0, n, step, GfVec3f(0.0f), e, 5, 1.0),
              std::string("frizz profile weighting: ") + cs.label);
    }
}

static void TestFrizzClumps()
{
    // Strands 0 and 1 share clump 0 (rest lengths 1 and 2 -> mean 1.5);
    // strand 2 is unclumped; strand 3 is unclumped with strandSeed 0, which
    // the reference also looks up in the clump table (clump 0's mean).
    Planes pl;
    pl.clumpId = {0, 0, -1, -1};
    pl.strandSeed = {3, 4, 5, 0};
    pl.restLength = {1.0f, 2.0f, 1.0f, 1.0f};
    int const n = 9;
    float const step = 0.125f;
    std::vector<GfVec3f> const roots = {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(2, 0, 0),
                                        GfVec3f(3, 0, 0)};
    auto curves = Strands(roots, n, step, pl);
    FrizzExpect e;
    e.amplitude = 0.05; e.wavesX = 1.0; e.wavesZ = 2.0; e.noise = 0.3; e.noiseSize = 0.2;
    Params on = FrizzParams(e);
    on.push_back({"frizz:useClumpIds", B(true)});
    UsdGenCurveBuffer shared, own;
    bool const ok = Cook(Graph(curves, "UsdGenFrizz", on), &shared) &&
                    Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &own);
    Check(ok, "frizz clump cooks");
    if (!ok) return;
    bool identical = true;
    for (int k = 0; k < n; ++k)
        identical = identical && Near(shared.px[size_t(k)], shared.px[size_t(n + k)] - 1.0f, 1e-6f) &&
                    shared.py[size_t(k)] == shared.py[size_t(n + k)] &&
                    shared.pz[size_t(k)] == shared.pz[size_t(n + k)];
    Check(identical, "useClumpIds: clump members frizz identically");
    FrizzExpect se = e;
    se.shared = true;
    Check(MatchStrand(shared, 0, n, step, roots[0], se, 0, 1.5) &&
              MatchStrand(shared, 1, n, step, roots[1], se, 0, 1.5),
          "useClumpIds: seed = clump id, reference length = clump mean restLength, no rotation");
    Check(MatchStrand(shared, 2, n, step, roots[2], se, 5, 1.0),
          "useClumpIds: unclumped strand keeps its own seed and restLength");
    Check(MatchStrand(shared, 3, n, step, roots[3], se, 0, 1.5),
          "useClumpIds: unclumped strand seed indexes the clump table (reference behaviour)");
    Check(MatchStrand(own, 0, n, step, roots[0], e, 3, 1.0) &&
              MatchStrand(own, 1, n, step, roots[1], e, 4, 2.0),
          "useClumpIds off: per-strand seed and restLength");
}

static void TestFrizzKnotMask()
{
    std::vector<float> const mask = {1.0f, 0.0f, 0.5f, 1.0f, 0.25f};
    Planes pl;
    pl.strandSeed = {9};
    pl.knotMask = mask;
    int const n = 5;
    float const step = 0.25f;
    auto curves = Strands({GfVec3f(0, 0, 0)}, n, step, pl);
    FrizzExpect e;
    e.amplitude = 0.05; e.wavesX = 1.0; e.wavesZ = 1.0;
    Params keep = FrizzParams(e);
    keep.push_back({"frizz:preserveKnots", B(true)});
    UsdGenCurveBuffer masked, unmasked;
    bool const ok = Cook(Graph(curves, "UsdGenFrizz", keep), &masked) &&
                    Cook(Graph(curves, "UsdGenFrizz", FrizzParams(e)), &unmasked);
    Check(ok && MatchStrand(masked, 0, n, step, GfVec3f(0.0f), e, 9, 1.0, mask),
          "preserveKnots: offsets attenuated by braidKnotMask");
    Check(ok && MatchStrand(unmasked, 0, n, step, GfVec3f(0.0f), e, 9, 1.0),
          "preserveKnots off: braidKnotMask ignored");
    Check(ok && masked.px[1] == 0.0f && masked.pz[1] == 0.0f, "knot mask 0 pins the point");
}

// Enough strands for the parallel capture path: two cooks agree bit for bit.
static void TestParallelDeterminism()
{
    Planes pl;
    std::vector<GfVec3f> roots;
    for (int i = 0; i < 600; ++i) {
        roots.push_back(GfVec3f(0.01f * float(i % 30), 0.0f, 0.013f * float(i / 30)));
        pl.primaryGuide.push_back(i % 7);
        pl.clumpId.push_back(i % 5 == 0 ? -1 : i % 11);
        pl.strandSeed.push_back(int((uint32_t(i) * 2654435761u) & 0x7fffffffu));
        pl.restLength.push_back(0.5f + 0.001f * float(i % 13));
    }
    auto curves = Strands(roots, 12, 0.05f, pl);
    Params const twist = {{"twist:root", F(5)}, {"twist:mid", F(60)}, {"twist:tip", F(120)},
                          {"twist:noise", F(0.7)}, {"twist:useClumpIds", B(true)}};
    FrizzExpect e;
    e.amplitude = 0.04; e.wavesX = 2.0; e.wavesZ = 1.0; e.noise = 0.5; e.noiseSize = 0.05; e.sharp = 0.3;
    Params frizz = FrizzParams(e);
    frizz.push_back({"frizz:useClumpIds", B(true)});
    UsdGenCurveBuffer t1, t2, f1, f2;
    bool const ok = Cook(Graph(curves, "UsdGenTwist", twist), &t1) &&
                    Cook(Graph(curves, "UsdGenTwist", twist), &t2) &&
                    Cook(Graph(curves, "UsdGenFrizz", frizz), &f1) &&
                    Cook(Graph(curves, "UsdGenFrizz", frizz), &f2);
    Check(ok && SameBits(t1, t2) && SameBits(f1, f2), "600-strand twist/frizz cooks are deterministic");
    bool moved = false;
    for (size_t i = 0; ok && i < t1.px.size(); ++i)
        moved = moved || t1.pz[i] != curves.points[i][2];
    Check(moved, "600-strand twist moves strands");
}

int main()
{
    TestRegistry();
    TestTwistProfile();
    TestTwistGrouping();
    TestTwistMirror();
    TestTwistNoise();
    TestTwistResnap();
    TestFrizzWaves();
    TestFrizzNoise();
    TestFrizzProfile();
    TestFrizzClumps();
    TestFrizzKnotMask();
    TestParallelDeterminism();
    std::printf(failures ? "FAILED (%d)\n" : "twist/frizz tests passed\n", failures);
    return failures ? 1 : 0;
}
