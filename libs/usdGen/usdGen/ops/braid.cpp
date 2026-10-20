// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenBraidOp implementation. See braid.h for the behaviour.
//
// The math follows the reference braid construction step by step (same
// constants, clamps, evaluation order and float rounding where it matters):
//   set level    fade/tail/cinch/spread constants, knot rate f/R * 2pi/6,
//                half width W = width * 0.045 * R / 2, depth D, lane scale K,
//                minimum lane separation S0 of the unit pattern, knot tension
//                (pinch/bulge), the clump profile and the tail-relax weight;
//   per guide    spine C[j] at j/(n-1) * Lg, transported frames, PCA lane
//                axis, projection ranking, lane slots, lane curves;
//   per member   target = lane curve + slot offset (bundle in the tail),
//                P = lerp(P, target, profile * relax * flyaway factor),
//                strands shorter than the guide sample the braid arrays at
//                u * Lf / Lg.
// Inferred (not decoded): the slot layout of the loose tail/bundle (one
// concentric-disk slot around the members' mean root, radii sqrt(3) W and
// sqrt(3) D), and the projection sort tie-break (strandIndex order).
#include "usdGen/ops/braid.h"

#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

TfTokenVector const kBraidReferences{TfToken("guides")};

TfToken const sFrequency("braid:frequency");
TfToken const sWidth("braid:width");
TfToken const sFlatness("braid:flatness");
TfToken const sKnotTightness("braid:knotTightness");
TfToken const sFadeIn("braid:fadeIn");
TfToken const sFadeOut("braid:fadeOut");
TfToken const sRootSpread("braid:rootSpread");
TfToken const sTipSpread("braid:tipSpread");
TfToken const sTailLength("braid:tailLength");
TfToken const sTailRelax("braid:tailRelax");
TfToken const sTailCinch("braid:tailCinch");
TfToken const sReferenceLength("braid:referenceLength");
TfToken const sRoot("braid:root");
TfToken const sMid("braid:mid");
TfToken const sMidShift("braid:midShift");
TfToken const sTip("braid:tip");
TfToken const sAmount("braid:amount");
TfToken const sFlyFraction("braid:flyaway:fraction");
TfToken const sFlyAmount("braid:flyaway:amount");
TfToken const sKnotMask("braid:knotMask");

TfToken const sClumpId("clumpId_0");
TfToken const sKnotMaskPlane("braidKnotMask");
TfToken const sNearestGuide("nearestGuide");

constexpr float kPi6 = 0.5235988f;        // pi / 6
constexpr float kTwoPi3 = 2.0943952f;     // 2 pi / 3
constexpr float kTwoPi = 6.2831855f;
constexpr float kSqrt3 = 1.7320508f;
constexpr float kDepthScale = 1.1547005f; // 2 / sqrt(3)
constexpr int kLanePerm[3] = {1, 2, 0};   // lane l runs at phase + perm[l] * 2pi/3

struct Settings
{
    float frequency = 8.0f, width = 1.0f, flatness = 0.55f, knotTightness = 0.5f;
    float fadeIn = 0.1f, fadeOut = 0.1f, rootSpread = 0.0f, tipSpread = 0.0f;
    float tailLength = 0.0f, tailRelax = 0.5f, tailCinch = 0.6f, referenceLength = 0.0f;
    float root = 0.0f, mid = 0.5f, midShift = 0.5f, tip = 1.0f, amount = 1.0f;
    float flyFraction = 0.0f, flyAmount = 0.0f;
    bool knotMask = false;
};

Settings ReadSettings(UsdGenParamView const *p)
{
    Settings s;
    if (!p) return s;
    auto f = [p](TfToken const &name, float fallback) {
        double const v = p->GetDouble(name, double(fallback));
        return std::isfinite(v) ? float(v) : fallback;
    };
    s.frequency = f(sFrequency, s.frequency);
    s.width = f(sWidth, s.width);
    s.flatness = f(sFlatness, s.flatness);
    s.knotTightness = f(sKnotTightness, s.knotTightness);
    s.fadeIn = f(sFadeIn, s.fadeIn);
    s.fadeOut = f(sFadeOut, s.fadeOut);
    s.rootSpread = f(sRootSpread, s.rootSpread);
    s.tipSpread = f(sTipSpread, s.tipSpread);
    s.tailLength = f(sTailLength, s.tailLength);
    s.tailRelax = f(sTailRelax, s.tailRelax);
    s.tailCinch = f(sTailCinch, s.tailCinch);
    s.referenceLength = f(sReferenceLength, s.referenceLength);
    s.root = f(sRoot, s.root);
    s.mid = f(sMid, s.mid);
    s.midShift = f(sMidShift, s.midShift);
    s.tip = f(sTip, s.tip);
    s.amount = f(sAmount, s.amount);
    s.flyFraction = f(sFlyFraction, s.flyFraction);
    s.flyAmount = f(sFlyAmount, s.flyAmount);
    s.knotMask = p->GetBool(sKnotMask, s.knotMask);
    return s;
}

inline float Clamp01(float x) { return strand::Clamp01(x); }

/// smoothstep of an already-mapped value, in the reference's operation order:
/// (3 - 2x) x x with x clamped to [0, 1].
inline float Ss(float x)
{
    x = Clamp01(x);
    return (3.0f - (x + x)) * x * x;
}

inline float Lerp(float a, float b, float t) { return (1.0f - t) * a + b * t; }

/// Lane cross-section: 2.2 cos(phi) / (1.2 |cos(phi)| + 1).
inline float LaneShape(float phi)
{
    float const c = std::cos(phi);
    return (c * 2.2f) / (std::fabs(c) * 1.2f + 1.0f);
}

/// Width normaliser: sqrt(3) / (f(0) - f(2pi/3)).
float LaneScale()
{
    float const d = LaneShape(0.0f) - LaneShape(kTwoPi3);
    return d > 1e-6f ? kSqrt3 / d : 0.0f;
}

/// Minimum pairwise 2-D distance between the three lanes of the unit pattern
/// (x = f(phi) K, y = sin(2 phi) 2/sqrt(3)) over 1024 phase samples.
float MinLaneSeparation()
{
    float const k = LaneScale();
    float best = 1e30f;
    for (int i = 0; i < 1024; ++i) {
        float const base = float(i) * 0.0009765625f * kTwoPi;
        float const phase[3] = {base + kTwoPi3, base + 4.1887903f, base + 0.0f};
        float x[3], y[3];
        for (int l = 0; l < 3; ++l) {
            x[l] = LaneShape(phase[l]) * k;
            y[l] = std::sin(phase[l] + phase[l]) * kDepthScale;
        }
        for (int a = 0; a < 3; ++a)
            for (int b = a + 1; b < 3; ++b) {
                float const dx = x[a] - x[b], dy = y[a] - y[b];
                best = std::min(best, std::sqrt(dy * dy + dx * dx));
            }
    }
    return best;
}

float CachedMinLaneSeparation()
{
    static float const value = MinLaneSeparation();
    return value;
}

/// Lane amplitude envelope at t: fade-in ramp times (1 - fade-out ramp); the
/// fade-out ends at the tail start when a tail is active.
float Fade(float t, float fadeIn, float fadeOut, float tail)
{
    float in = 1.0f;
    if (fadeIn > 1e-4f) in = Ss((t - 0.0f) / (fadeIn - 0.0f));
    float out;
    if (tail <= 1e-4f) {
        if (fadeOut <= 1e-4f) return in;
        out = Ss((t - (1.0f - fadeOut)) / (1.0f - (1.0f - fadeOut)));
    } else {
        float const a = (1.0f - tail) - std::max(0.04f, fadeOut);
        out = Ss((t - a) / ((1.0f - tail) - a));
    }
    return in * (1.0f - out);
}

/// Knot strength at strand parameter u (1 at a lane crossing).
float Knot(float u, float guideLength, float rate, bool noKnots, float fadeIn, float fadeOut,
           float tail)
{
    if (guideLength <= 1e-6f) return 0.0f;
    if (noKnots) {
        float r = 1.0f;
        if (tail > 1e-4f) {
            float const a = (1.0f - tail) - std::max(0.04f, fadeOut);
            float const x = Clamp01((u - a) / ((1.0f - tail) - a));
            r = 1.0f - (3.0f - (x + x)) * x * x;
        }
        return r;
    }
    float const c = std::cos((u * guideLength * rate + kPi6) * 6.0f);
    float const x = Clamp01(((c + 1.0f) * 0.5f - 0.21f) / 0.44f);
    return Fade(u, fadeIn, fadeOut, tail) * (3.0f - (x + x)) * x * x;
}

/// Transported frames along a polyline, seeded by a direction: the first
/// normal is normalize(seed x up) (up = +Y, +X when |seed.y| >= 0.95), then
/// each point projects the previous normal off its central-difference
/// tangent; binormal = T x N.
void Frames(std::vector<GfVec3f> const &C, GfVec3f const &seed, std::vector<GfVec3f> *N,
            std::vector<GfVec3f> *B)
{
    int const n = int(C.size());
    N->assign(size_t(n), GfVec3f(1, 0, 0));
    B->assign(size_t(n), GfVec3f(1, 0, 0));
    if (n <= 0) return;
    float const sx = seed[0], sy = seed[1], sz = seed[2];
    float upx = 0.0f, upy = 1.0f;
    if (std::fabs(sy) >= 0.95f) { upx = 1.0f; upy = 0.0f; }
    float nz = sx * upy - upx * sy;
    float nx = sy * 0.0f - sz * upy;
    float ny = upx * sz - sx * 0.0f;
    float l = std::sqrt(ny * ny + nx * nx + nz * nz);
    if (l > 1e-6f) { nx /= l; ny /= l; nz /= l; }
    else { nx = 1.0f; ny = 0.0f; nz = 0.0f; }
    float bz = ny * sx - sy * nx;
    float bx = sy * nz - sz * ny;
    float by = sz * nx - sx * nz;
    l = std::sqrt(by * by + bx * bx + bz * bz);
    if (l > 1e-6f) { bx /= l; by /= l; bz /= l; }
    else { bx = 1.0f; by = 0.0f; bz = 0.0f; }
    for (int j = 0; j < n; ++j) {
        int const i1 = std::min(j + 1, n - 1), i0 = std::max(j - 1, 0);
        float tx = C[size_t(i1)][0] - C[size_t(i0)][0];
        float ty = C[size_t(i1)][1] - C[size_t(i0)][1];
        float tz = C[size_t(i1)][2] - C[size_t(i0)][2];
        float tl = std::sqrt(tx * tx + ty * ty + tz * tz);
        if (tl <= 1e-6f) { tx = sx; ty = sy; tz = sz; }
        else { tx /= tl; ty /= tl; tz /= tl; }
        float const d = ty * ny + tx * nx + tz * nz;
        float px = nx - tx * d, py = ny - ty * d, pz = nz - tz * d;
        float pl = std::sqrt(px * px + py * py + pz * pz);
        bool valid = pl > 1e-5f;
        if (!valid) {
            float ux = 0.0f, uy = 1.0f;
            if (std::fabs(ty) >= 0.95f) { ux = 1.0f; uy = 0.0f; }
            px = ty * 0.0f - tz * uy;
            py = tz * ux - tx * 0.0f;
            pz = tx * uy - ty * ux;
            pl = std::sqrt(py * py + px * px + pz * pz);
            valid = pl > 1e-6f;
        }
        if (valid) {
            nx = px / pl; ny = py / pl; nz = pz / pl;
            bx = ty * nz - tz * ny;
            by = tz * nx - tx * nz;
            bz = tx * ny - ty * nx;
            float const bl = std::sqrt(by * by + bx * bx + bz * bz);
            if (bl > 1e-6f) { bx /= bl; by /= bl; bz /= bl; }
            else { bx = 1.0f; by = 0.0f; bz = 0.0f; }
        }
        (*N)[size_t(j)] = GfVec3f(nx, ny, nz);
        (*B)[size_t(j)] = GfVec3f(bx, by, bz);
    }
}

/// Dot product in the reference's evaluation order (x, y, then z).
inline float Dot(GfVec3f const &a, GfVec3f const &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/// Concentric square-to-disk slots: each member's offset from its group
/// centre, normalised per group by the largest |offset| along each axis,
/// mapped onto the disk (Shirley-Chiu) and scaled by (radiusA, radiusC).
void DiskSlots(std::vector<GfVec3f> const &rankRoots, std::vector<int> const &group,
               std::vector<GfVec3f> const &centres, GfVec3f const &axisA,
               GfVec3f const &axisC, float radiusA, float radiusC, std::vector<float> *u,
               std::vector<float> *v)
{
    size_t const m = rankRoots.size();
    std::vector<float> maxA(centres.size(), 0.0f), maxC(centres.size(), 0.0f);
    for (size_t r = 0; r < m; ++r) {
        GfVec3f const d = rankRoots[r] - centres[size_t(group[r])];
        maxA[size_t(group[r])] = std::max(std::fabs(Dot(axisA, d)), maxA[size_t(group[r])]);
        maxC[size_t(group[r])] = std::max(std::fabs(Dot(axisC, d)), maxC[size_t(group[r])]);
    }
    u->assign(m, 0.0f);
    v->assign(m, 0.0f);
    for (size_t r = 0; r < m; ++r) {
        size_t const k = size_t(group[r]);
        GfVec3f const d = rankRoots[r] - centres[k];
        float const a = maxA[k] <= 1e-8f ? 0.0f : Dot(d, axisA) / maxA[k];
        float const b = maxC[k] <= 1e-8f ? 0.0f : Dot(d, axisC) / maxC[k];
        float phi = 0.0f, rad = 0.0f;
        if (std::fabs(a) <= std::fabs(b)) {
            if (std::fabs(b) > 1e-8f) {
                phi = 1.5707964f - (a / b) * 0.7853982f;
                rad = b;
            }
        } else {
            phi = (b / a) * 0.7853982f;
            rad = a;
        }
        (*u)[r] = std::cos(phi) * rad * radiusA;
        (*v)[r] = std::sin(phi) * rad * radiusC;
    }
}

/// Reference length: the authored value, else the diagonal of the bound
/// surface's rest-point bounding box; 1 when degenerate or unbound.
float ReferenceLength(UsdGenCaptureContext const &ctx, Settings const &s)
{
    float length = 0.0f;
    if (s.referenceLength > 0.0f) {
        length = s.referenceLength;
    } else if (ctx.desc && ctx.surface < ctx.desc->surfaces.size()) {
        UsdGenSurfaceDesc const &surface = ctx.desc->surfaces[ctx.surface];
        VtVec3fArray const &points = surface.restPoints.empty() ? surface.points
                                                                : surface.restPoints;
        if (!points.empty()) {
            GfVec3f lo = points[0], hi = points[0];
            for (GfVec3f const &q : points)
                for (int d = 0; d < 3; ++d) {
                    lo[d] = std::min(lo[d], q[d]);
                    hi[d] = std::max(hi[d], q[d]);
                }
            GfVec3f const e = hi - lo;
            length = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        }
    }
    return length > 1e-4f && std::isfinite(length) ? length : 1.0f;
}

/// Set-level constants derived from the settings.
struct Weave
{
    float fadeIn, fadeOut, tail, tailStart, tailFade, cinch, relax;
    float rootSpread, tipSpread, knotTightness, sepScale, pinch, bulge;
    float halfWidth, depth, laneX, laneY, rate, flyAmount, flyFraction;
    bool tailActive, noKnots, knotMask;
};

Weave MakeWeave(Settings const &s, float referenceLength)
{
    Weave w;
    w.fadeIn = Clamp01(s.fadeIn);
    w.fadeOut = Clamp01(s.fadeOut);
    w.tail = std::min(std::max(s.tailLength, 0.0f), 0.9f);
    w.tailActive = w.tail > 1e-4f;
    w.tailStart = 1.0f - w.tail;
    w.tailFade = std::max(0.04f, w.fadeOut);
    w.noKnots = !(std::max(0.0f, s.frequency) > 1e-4f);
    float const diag = referenceLength;
    float const diagClamped = std::max(1e-6f, diag);
    w.rate = ((std::max(0.0f, s.frequency) / diagClamped) * kTwoPi) / 6.0f;
    w.halfWidth = std::max(0.0f, s.width) * 0.045f * diag * 0.5f;
    float const flat = Clamp01(s.flatness);
    w.depth = (flat * 0.35f + (1.0f - flat)) * w.halfWidth;
    w.laneX = w.halfWidth * LaneScale();
    w.laneY = w.depth * kDepthScale;
    w.knotTightness = Clamp01(s.knotTightness);
    float const sepTarget = Lerp(2.4f, 1.6f, w.knotTightness);
    float const s0 = CachedMinLaneSeparation();
    w.sepScale = s0 > 1e-6f ? sepTarget / s0 : 1.0f;
    float const room = (diagClamped / std::max(1e-4f, s.frequency)) /
                       std::max(1e-6f, w.halfWidth + w.halfWidth);
    float const atten = Ss(room - 0.2f);
    w.pinch = 1.0f - w.knotTightness * 0.55f * atten;
    w.bulge = w.knotTightness * 0.4f * atten + 1.0f;
    w.rootSpread = std::max(0.0f, s.rootSpread);
    w.tipSpread = std::max(0.0f, s.tipSpread);
    w.relax = Clamp01(s.tailRelax);
    w.cinch = Lerp(1.0f, 0.02f, Clamp01(s.tailCinch));
    w.flyAmount = Clamp01(s.flyAmount);
    w.flyFraction = Clamp01(s.flyFraction);
    w.knotMask = s.knotMask;
    return w;
}

/// Per CV-count tables: the clump profile (root/mid/tip Bezier times amount)
/// and the tail-relax weight, both at t = j / (n - 1).
struct Tables
{
    std::vector<float> profile, relax;
};

Tables MakeTables(size_t n, Settings const &s, Weave const &w)
{
    Tables t;
    t.profile.assign(n, 0.0f);
    t.relax.assign(n, 1.0f);
    float const root = Clamp01(s.root), mid = Clamp01(s.mid), tip = Clamp01(s.tip);
    float const amount = Clamp01(s.amount);
    for (size_t j = 0; j < n; ++j) {
        float const u = n > 1 ? float(j) / float(n - 1) : 0.0f;
        float const b = strand::BezierProfile(u, root, mid, s.midShift, tip);
        t.profile[j] = Clamp01(amount * b);
        if (w.tailActive && w.relax > 1e-4f)
            t.relax[j] = 1.0f - Ss((u - w.tailStart) / (1.0f - w.tailStart)) * w.relax;
    }
    return t;
}

struct GuideCurve
{
    std::vector<float> points;   // xyz
    std::vector<float> cum;      // arc length
    float length = 0.0f;
    GfVec3f root{0.0f};
    GfVec3f seed{0.0f, 1.0f, 0.0f};
};

struct StrandInfo
{
    float length = 0.0f;
    GfVec3f root{0.0f};
    bool fly = false;
};

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, bool parallel, F const &body)
{
    if (!dispatcher || !parallel || count < 2) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

/// Weaves the members of one guide. Writes only the members' points, clump
/// ids and mask values.
void BraidGuide(int g, GuideCurve const &guide, std::vector<uint32_t> const &members,
                Weave const &w, std::map<size_t, Tables> const &tables,
                std::vector<StrandInfo> const &info, UsdGenStrandSet &strands,
                std::vector<int32_t> &clumpIds, std::vector<float> &mask)
{
    size_t const m = members.size();
    if (m == 0) return;
    float const Lg = guide.length;
    if (Lg <= 1e-6f) return;
    size_t n = 0;
    for (uint32_t c : members) n = std::max(n, strands.Size(c));
    if (n < 2) return;
    Tables const &braidTables = tables.at(n);

    // Spine and frames.
    float const inv = 1.0f / float(int(n) - 1);
    std::vector<GfVec3f> C(n), N, B;
    size_t const gn = guide.points.size() / 3;
    for (size_t j = 0; j < n; ++j)
        C[j] = strand::SampleArc(guide.points.data(), gn, guide.cum, float(j) * inv * Lg);
    Frames(C, guide.seed, &N, &B);

    // Mean root and PCA of the root offsets in the root frame.
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (uint32_t c : members) {
        GfVec3f const &q = info[c].root;
        mx = q[0] + mx;
        my = q[1] + my;
        mz = q[2] + mz;
    }
    GfVec3f const mean(mx / float(m), my / float(m), mz / float(m));
    float sxx = 0.0f, sxy = 0.0f, syy = 0.0f;
    for (uint32_t c : members) {
        GfVec3f const d = info[c].root - mean;
        float const a = Dot(N[0], d), b = Dot(B[0], d);
        sxx = sxx + a * a;
        sxy = sxy + b * a;
        syy = syy + b * b;
    }
    float const theta = std::atan2(sxy + sxy, sxx - syy) * 0.5f;
    float const ct = std::cos(theta), st = std::sin(theta);
    GfVec3f const A0 = N[0] * ct + B[0] * st;
    GfVec3f const X0 = N[0] * -st + B[0] * ct;

    // Rank by projection on the wide axis; three lanes by rank.
    std::vector<std::pair<float, uint32_t>> keyed(m);
    for (size_t i = 0; i < m; ++i) {
        uint32_t const c = members[i];
        keyed[i] = {Dot(A0, info[c].root - mean), uint32_t(i)};
    }
    std::stable_sort(keyed.begin(), keyed.end(),
                     [](auto const &a, auto const &b) { return a.first < b.first; });
    std::vector<uint32_t> rankToMember(m);
    std::vector<GfVec3f> rankRoots(m);
    std::vector<int> lane(m);
    for (size_t r = 0; r < m; ++r) {
        rankToMember[r] = members[keyed[r].second];
        rankRoots[r] = info[rankToMember[r]].root;
        lane[r] = int((3 * r) / m);
    }

    // In-lane slots around each lane's mean root.
    std::vector<GfVec3f> laneCentre(3, GfVec3f(0.0f));
    int laneCount[3] = {0, 0, 0};
    for (size_t r = 0; r < m; ++r) {
        GfVec3f &sum = laneCentre[size_t(lane[r])];
        for (int d = 0; d < 3; ++d) sum[d] = rankRoots[r][d] + sum[d];
        ++laneCount[lane[r]];
    }
    for (int l = 0; l < 3; ++l)
        if (laneCount[l] > 0)
            for (int d = 0; d < 3; ++d)
                laneCentre[size_t(l)][d] = laneCentre[size_t(l)][d] / float(laneCount[l]);
    std::vector<float> slotU, slotV, tailU(m, 0.0f), tailV(m, 0.0f);
    DiskSlots(rankRoots, lane, laneCentre, A0, X0, w.halfWidth, w.depth, &slotU, &slotV);
    bool const bundle = w.tailActive || w.noKnots;
    if (bundle) {
        std::vector<int> one(m, 0);
        DiskSlots(rankRoots, one, std::vector<GfVec3f>{mean}, A0, X0, w.halfWidth * kSqrt3,
                  w.depth * kSqrt3, &tailU, &tailV);
    }

    // Lane cross-section offsets.
    std::vector<float> laneX(3 * n, 0.0f), laneY(3 * n, 0.0f), laneAbs(3 * n, 1.0f);
    if (!w.noKnots) {
        for (size_t j = 0; j < n; ++j) {
            float const phase = float(int(j)) * inv * Lg * w.rate + kPi6;
            for (int l = 0; l < 3; ++l) {
                float const phi = float(kLanePerm[l]) * kTwoPi3 + phase;
                float const f = LaneShape(phi);
                laneX[size_t(l) * n + j] = f * w.laneX;
                laneY[size_t(l) * n + j] = std::sin(phi + phi) * w.laneY;
                laneAbs[size_t(l) * n + j] = std::fabs(f);
            }
        }
        for (size_t i = 0; i < 3 * n; ++i) {
            laneX[i] = w.sepScale * laneX[i];
            laneY[i] = w.sepScale * laneY[i];
        }
    }

    // Envelopes and rotated frames per spine point.
    std::vector<float> fade(n, 1.0f), rad(n, 1.0f), tailBlend(n, 0.0f), cinch(n, 1.0f);
    std::vector<GfVec3f> A(n), X(n), tailA, tailX;
    if (bundle) { tailA.assign(n, GfVec3f(0.0f)); tailX.assign(n, GfVec3f(0.0f)); }
    float const tailFadeStart = w.tailStart - w.tailFade;
    for (size_t j = 0; j < n; ++j) {
        float const t = float(int(j)) * inv;
        fade[j] = Fade(t, w.fadeIn, w.fadeOut, w.tailActive ? w.tail : 0.0f);
        if (w.tailActive) {
            float const tb = Ss((t - tailFadeStart) / (w.tailStart - tailFadeStart));
            tailBlend[j] = tb;
            if (w.tailStart < t) {
                float const x = Clamp01((t - w.tailStart) / std::max(1e-6f, w.tail));
                float const k = Ss((x - 0.0f) / 0.35f);
                cinch[j] = (1.0f - k) * w.cinch + k;
            } else {
                cinch[j] = w.cinch * tb + (1.0f - tb);
            }
        }
        float spread = 1.0f;
        if (w.rootSpread > 1e-4f) spread = (1.0f - Ss((t - 0.0f) / 0.35f)) * w.rootSpread + 1.0f;
        if (w.tipSpread > 1e-4f) spread = spread + Ss((t - 0.65f) / 0.35000002f) * w.tipSpread;
        rad[j] = spread * cinch[j];
        A[j] = N[j] * ct + B[j] * st;
        X[j] = N[j] * -st + B[j] * ct;
        if (bundle) {
            tailA[j] = A[j] * rad[j];
            tailX[j] = X[j] * rad[j];
        }
    }

    // Lane curves plus the slot axes scaled by the pinch/bulge bundle radius.
    std::vector<GfVec3f> laneC(3 * n), laneA(3 * n), laneXs(3 * n);
    for (size_t l = 0; l < 3; ++l) {
        for (size_t j = 0; j < n; ++j) {
            size_t const i = l * n + j;
            float const cross = 1.0f - Ss((laneAbs[i] - 0.15f) / 0.70000005f);
            float const radius = ((1.0f - cross) * w.bulge + cross * w.pinch) * rad[j];
            float const rf = rad[j] * fade[j];
            float const x = laneX[i], y = laneY[i];
            laneC[i] = GfVec3f(C[j][0] + (A[j][0] * x + X[j][0] * y) * rf,
                               (X[j][1] * y + A[j][1] * x) * rf + C[j][1],
                               (X[j][2] * y + A[j][2] * x) * rf + C[j][2]);
            laneA[i] = A[j] * radius;
            laneXs[i] = X[j] * radius;
        }
    }

    // Members.
    for (size_t r = 0; r < m; ++r) {
        uint32_t const c = rankToMember[r];
        StrandInfo const &si = info[c];
        int const l = lane[r];
        clumpIds[c] = si.fly ? -1 : 3 * g + l;
        float const Lf = si.length;
        if (!(Lf > 1e-6f)) continue;
        size_t const ns = strands.Size(c);
        if (ns < 1) continue;
        Tables const &own = tables.at(ns);
        float const u = slotU[r], v = slotV[r];
        float const tu = bundle ? tailU[r] : 0.0f, tv = bundle ? tailV[r] : 0.0f;
        float const flyFactor = si.fly ? w.flyAmount : 1.0f;
        float const invS = ns > 1 ? 1.0f / float(int(ns) - 1) : 0.0f;
        size_t const row = size_t(l) * n;
        size_t const firstCv = strands.First(c);
        if (w.knotMask && !si.fly) {
            float const f = Lg <= Lf ? 1.0f : Lf / Lg;
            for (size_t k = 0; k < ns; ++k) {
                float const knot = Knot(float(int(k)) * invS * f, Lg, w.rate, w.noKnots,
                                        w.fadeIn, w.fadeOut, w.tailActive ? w.tail : 0.0f);
                mask[firstCv + k] = 1.0f - knot;
            }
        }
        float *P = strands.P(c);
        auto apply = [&](size_t k, float weight, GfVec3f const &target) {
            float const keep = 1.0f - weight;
            P[3 * k] = target[0] * weight + P[3 * k] * keep;
            P[3 * k + 1] = P[3 * k + 1] * keep + target[1] * weight;
            P[3 * k + 2] = P[3 * k + 2] * keep + target[2] * weight;
        };
        if (Lf < Lg || ns != n) {
            // Sample the braid arrays at u * Lf / Lg (linear in the index).
            float const ratio = Lf < Lg ? Lf / Lg : 1.0f;
            int const last = int(n) - 1;
            for (size_t k = 0; k < ns; ++k) {
                float const fi = float(int(k)) * invS * ratio * float(last);
                int i0 = std::min(std::max(int(fi), 0), last);
                int const i1 = std::min(i0 + 1, last);
                float const fr = fi - float(i0);
                float const om = 1.0f - fr;
                float const weight =
                    (braidTables.relax[size_t(i0)] * om + braidTables.relax[size_t(i1)] * fr) *
                    own.profile[k] * flyFactor;
                if (!(weight > 1e-6f)) continue;
                float const tb = w.noKnots ? 1.0f : fr * tailBlend[size_t(i1)] +
                                                        om * tailBlend[size_t(i0)];
                float const keepLane = 1.0f - tb;
                size_t const a0 = row + size_t(i0), a1 = row + size_t(i1);
                GfVec3f target;
                for (int d = 0; d < 3; ++d) {
                    float const xs = laneXs[a0][d] * om + fr * laneXs[a1][d];
                    float const as = laneA[a0][d] * om + fr * laneA[a1][d];
                    target[d] = keepLane * (xs * v + as * u) + laneC[a0][d] * om +
                                fr * laneC[a1][d];
                }
                if (bundle && tb > 1e-6f) {
                    for (int d = 0; d < 3; ++d) {
                        float const xs = tailX[size_t(i0)][d] * om + fr * tailX[size_t(i1)][d];
                        float const as = tailA[size_t(i0)][d] * om + fr * tailA[size_t(i1)][d];
                        target[d] = target[d] + tb * (tv * xs + tu * as);
                    }
                }
                apply(k, weight, target);
            }
        } else {
            for (size_t k = 0; k < n; ++k) {
                float const weight = braidTables.relax[k] * own.profile[k] * flyFactor;
                if (!(weight > 1e-6f)) continue;
                float const tb = w.noKnots ? 1.0f : tailBlend[k];
                float const keepLane = 1.0f - tb;
                size_t const i = row + k;
                GfVec3f target;
                for (int d = 0; d < 3; ++d)
                    target[d] = laneC[i][d] + keepLane * (laneXs[i][d] * v + laneA[i][d] * u);
                if (bundle && tb > 1e-6f)
                    for (int d = 0; d < 3; ++d)
                        target[d] = (tailA[k][d] * tu + tailX[k][d] * tv) * tb + target[d];
                apply(k, weight, target);
            }
        }
    }
}

}  // namespace

TfSpan<const TfToken> UsdGenBraidOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(kBraidReferences.data(), kBraidReferences.size());
}

TfTokenVector UsdGenBraidOp::StructuralParameters() const
{
    return {sFrequency,   sWidth,      sFlatness,    sKnotTightness, sFadeIn,
            sFadeOut,     sRootSpread, sTipSpread,   sTailLength,    sTailRelax,
            sTailCinch,   sReferenceLength, sRoot,   sMid,           sMidShift,
            sTip,         sAmount,     sFlyFraction, sFlyAmount,     sKnotMask};
}

std::vector<UsdGenStrandPlaneSpec> UsdGenBraidOp::OutputPlaneSpecs() const
{
    UsdGenStrandPlaneSpec clump;
    clump.name = sClumpId;
    clump.cv = false;
    clump.isInt = true;
    UsdGenStrandPlaneSpec knot;
    knot.name = sKnotMaskPlane;
    knot.cv = true;
    knot.isInt = false;
    return {clump, knot};
}

bool UsdGenBraidOp::Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                            std::vector<UsdGenStrandPlaneData> &planes,
                            UsdGenDiagnostics *diag)
{
    if (planes.size() != 2) {
        if (diag) diag->Error(Label() + ": output planes are not allocated");
        return false;
    }
    std::vector<int32_t> &clumpIds = planes[0].i;
    std::vector<float> &mask = planes[1].f;
    std::fill(mask.begin(), mask.end(), 1.0f);
    uint32_t const R = strands.Count();
    if (clumpIds.size() != R || mask.size() != strands.points.size() / 3) {
        if (diag) diag->Error(Label() + ": output planes do not match the strand topology");
        return false;
    }
    if (ctx.referenceCount != 1 || !ctx.resolvedReferences || !ctx.resolvedReferences[0] ||
        !ctx.resolvedReferences[0]->value) {
        if (diag) diag->Warn(Label() + ": usdGen:guides targets no guide curve set; strands are not braided");
        return true;
    }
    UsdGenCurveBuffer const &guides = ctx.resolvedReferences[0]->value->buffer;
    std::vector<uint32_t> guideSpans;
    std::string error;
    if (!opUtil::CurveSpans(guides, &guideSpans, &error)) {
        if (diag) diag->Error(Label() + ": guide topology: " + error);
        return false;
    }
    if (guides.px.size() != guides.totalCvs || guides.py.size() != guides.totalCvs ||
        guides.pz.size() != guides.totalCvs) {
        if (diag) diag->Error(Label() + ": guide point planes do not match their CV count");
        return false;
    }
    size_t const G = guides.totalCurves;
    if (G == 0) {
        if (diag) diag->Warn(Label() + ": the guide curve set has no curves; strands are not braided");
        return true;
    }

    Settings const s = ReadSettings(ctx.params);
    Weave const w = MakeWeave(s, ReferenceLength(ctx, s));

    // Guides: polyline, arc length, root and frame seed (first segment
    // direction, else the root-frame normal, else +Y).
    std::vector<GuideCurve> guideCurves(G);
    for (size_t g = 0; g < G; ++g) {
        GuideCurve &gc = guideCurves[g];
        size_t const first = guideSpans[g], count = guideSpans[g + 1] - first;
        gc.points.resize(3 * count);
        for (size_t i = 0; i < count; ++i) {
            gc.points[3 * i] = guides.px[first + i];
            gc.points[3 * i + 1] = guides.py[first + i];
            gc.points[3 * i + 2] = guides.pz[first + i];
        }
        gc.length = count ? strand::ArcLengths(gc.points.data(), count, &gc.cum) : 0.0f;
        if (count) gc.root = strand::Load(gc.points.data(), 0);
        GfVec3f seed(0.0f);
        float seedLength = 0.0f;
        if (count > 1) {
            seed = strand::Load(gc.points.data(), 1) - gc.root;
            seedLength = seed.GetLength();
        }
        if (seedLength > 1e-6f) {
            gc.seed = GfVec3f(seed[0] / seedLength, seed[1] / seedLength, seed[2] / seedLength);
        } else if (guides.rootN.size() == G) {
            gc.seed = strand::Normalize(guides.rootN[g], GfVec3f(0.0f, 1.0f, 0.0f));
        }
    }

    // Strands: length, root, flyaway flag (keyed by generation order).
    std::vector<StrandInfo> info(R);
    for (uint32_t c = 0; c < R; ++c) {
        size_t const ns = strands.Size(c);
        float const *P = strands.P(c);
        float length = 0.0f;
        for (size_t k = 1; k < ns; ++k) {
            float const dx = P[3 * k] - P[3 * k - 3];
            float const dy = P[3 * k + 1] - P[3 * k - 2];
            float const dz = P[3 * k + 2] - P[3 * k - 1];
            length += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        info[c].length = length;
        if (ns) info[c].root = strand::Load(P, 0);
        if (w.flyFraction > 0.0f) {
            uint64_t const key = uint64_t(int64_t(int32_t(strands.Index(c))));
            double const u01 =
                double(strand::SplitMix64(key) >> 11) * (1.0 / 9007199254740992.0);
            info[c].fly = double(w.flyFraction) > u01;
        }
    }

    // Membership: the upstream nearestGuide plane, else the nearest guide
    // root; members in strandIndex order.
    std::vector<uint32_t> order(R);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return strands.Index(a) < strands.Index(b);
    });
    int const *nearestPlane = strands.CurveInt(sNearestGuide);
    std::vector<std::vector<uint32_t>> members(G);
    for (uint32_t c : order) {
        if (strands.Size(c) == 0) continue;
        int g = -1;
        if (nearestPlane) {
            g = nearestPlane[c];
        } else {
            float best = 0.0f;
            for (size_t k = 0; k < G; ++k) {
                if (guideSpans[k + 1] == guideSpans[k]) continue;
                GfVec3f const d = info[c].root - guideCurves[k].root;
                float const d2 = Dot(d, d);
                if (g < 0 || d2 < best) { best = d2; g = int(k); }
            }
        }
        if (g >= 0 && size_t(g) < G) members[size_t(g)].push_back(c);
    }

    // Shared per CV-count tables (profile and tail relax), read-only below.
    std::map<size_t, Tables> tables;
    std::vector<int> active;
    for (size_t g = 0; g < G; ++g) {
        if (members[g].empty()) continue;
        active.push_back(int(g));
        size_t n = 0;
        for (uint32_t c : members[g]) {
            size_t const ns = strands.Size(c);
            n = std::max(n, ns);
            if (!tables.count(ns)) tables.emplace(ns, MakeTables(ns, s, w));
        }
        if (!tables.count(n)) tables.emplace(n, MakeTables(n, s, w));
    }

    ParallelFor(ctx.dispatcher, active.size(), R >= 2000, [&](size_t i) {
        int const g = active[i];
        BraidGuide(g, guideCurves[size_t(g)], members[size_t(g)], w, tables, info, strands,
                   clumpIds, mask);
    });
    return true;
}

}  // namespace usdGen
