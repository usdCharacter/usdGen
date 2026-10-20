// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — hair-card surface grids. See cardGeometry.h. Arithmetic is float32
// with the reference clamps and constants, so grids match the imported
// groom's own card mesh up to float rounding.
#include "usdGen/ops/cardGeometry.h"

#include "usdGen/ops/opUtil.h"

#include <algorithm>
#include <cmath>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace card {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = 0.017453292f;
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;

inline float Clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float Finite(float v, float fallback) { return std::isfinite(v) ? v : fallback; }

inline GfVec3f Nrm(GfVec3f const &v, GfVec3f const &fallback = GfVec3f(1.0f, 0.0f, 0.0f))
{
    return strand::Normalize(v, fallback, 1e-6f);
}

/// Mirror plane index -> mirrored coordinate axis (0: x, 1: z, 2: y).
inline int PlaneAxis(int plane) { return plane == 1 ? 2 : (plane == 2 ? 1 : 0); }

inline GfVec3f MirrorPoint(GfVec3f p, int plane, GfVec3f const &centre)
{
    int const a = PlaneAxis(plane);
    p[a] = 2.0f * centre[a] - p[a];
    return p;
}

inline GfVec3f MirrorVector(GfVec3f v, int plane)
{
    v[PlaneAxis(plane)] = -v[PlaneAxis(plane)];
    return v;
}

float CumulativeLengths(std::vector<GfVec3f> const &p, std::vector<float> *s)
{
    s->assign(p.size(), 0.0f);
    float acc = 0.0f;
    for (size_t i = 1; i < p.size(); ++i) {
        acc += (p[i] - p[i - 1]).GetLength();
        (*s)[i] = acc;
    }
    return acc;
}

/// Polyline point at arc length x: linear inside, linear extrapolation past
/// either end along the first / last segment direction.
GfVec3f SampleArcExtrapolated(std::vector<GfVec3f> const &p, std::vector<float> const &s,
                              float x)
{
    size_t const n = p.size();
    if (n < 2) return n ? p[0] : GfVec3f(0.0f);
    float const L = s[n - 1];
    if (x <= 0.0f) return p[0] + Nrm(p[1] - p[0]) * x;
    if (x >= L) return p[n - 1] + Nrm(p[n - 1] - p[n - 2]) * (x - L);
    size_t i = size_t(std::upper_bound(s.begin(), s.end(), x) - s.begin());
    i = std::min(std::max<size_t>(i, 1), n - 1);
    float const d = s[i] - s[i - 1];
    float const w = d <= 1e-8f ? 0.0f : (x - s[i - 1]) / d;
    return p[i - 1] + (p[i] - p[i - 1]) * w;
}

uint64_t SeedOf(Card const &card)
{
    return card.twinId != 0 ? std::min(card.id, card.twinId) : card.id;
}

Card const *FindTwin(Card const &card, Context const &ctx)
{
    if (!ctx.cards || card.twinId == 0) return nullptr;
    for (Card const &other : *ctx.cards)
        if (&other != &card && other.id == card.twinId) return &other;
    return nullptr;
}

// --- spine stages -------------------------------------------------------------------

void LengthScale(Card const &card, std::vector<GfVec3f> *P)
{
    float lo = Clamp(card.lengthMin, 0.01f, 2.0f), hi = Clamp(card.lengthMax, 0.01f, 2.0f);
    if (hi < lo) std::swap(lo, hi);
    float const r = strand::Clamp01(card.lengthRandomize);
    float scale = hi;
    if (r > 0.0f && hi - lo > 1e-6f)
        scale = (1.0f - r) * hi + (lo + strand::Hash01(SeedOf(card)) * (hi - lo)) * r;
    scale *= strand::Clamp01(card.lengthMask);
    std::vector<float> s;
    float const L = CumulativeLengths(*P, &s);
    if (!(std::fabs(scale - 1.0f) > 1e-4f) || !(L > 1e-6f)) return;
    std::vector<GfVec3f> const orig = *P;
    for (size_t i = 0; i < P->size(); ++i) (*P)[i] = SampleArcExtrapolated(orig, s, scale * s[i]);
}

void Frizz(Card const &card, std::vector<GfVec3f> *P)
{
    size_t const n = P->size();
    if (n < 3) return;
    float const amount = strand::Clamp01(card.frizzAmount);
    if (amount <= 1e-4f) return;
    float const fx = Clamp(card.frizzFrequency, 0.0f, 10.0f);
    float const fy = Clamp(card.frizzFrequencyZ, 0.0f, 10.0f);
    if (fx <= 1e-6f && fy <= 1e-6f) return;
    float const noise = strand::Clamp01(card.frizzNoise);
    std::vector<float> s;
    float const L = CumulativeLengths(*P, &s);
    if (L <= 1e-6f) return;
    float const A = amount * 0.25f * L;
    uint64_t const seed = SeedOf(card);
    float const phi = strand::Hash01(seed * 0xD8163841ull) * 2.0f * kPi;
    float const ofs = strand::Hash01(seed * 0x1B873593ull) * 1000.0f;
    GfVec3f const chord = Nrm((*P)[n - 1] - (*P)[0]);
    GfVec3f const R = std::fabs(chord[1]) >= 0.95f ? GfVec3f(1, 0, 0) : GfVec3f(0, 1, 0);
    float const root = strand::Clamp01(card.frizzRoot), mid = strand::Clamp01(card.frizzMid),
                shift = strand::Clamp01(card.frizzMidShift), tip = strand::Clamp01(card.frizzTip);
    float const cphi = std::cos(phi), sphi = std::sin(phi);
    std::vector<GfVec3f> disp(n, GfVec3f(0.0f));
    GfVec3f X(0.0f);
    bool haveX = false;
    for (size_t i = 1; i < n; ++i) {
        float const t = s[i] / L;
        float const w = strand::Clamp01(strand::BezierProfile(t, root, mid, shift, tip, 0.0f, 1.0f));
        GfVec3f d = i + 1 < n ? (*P)[i + 1] - (*P)[i] : (*P)[i] - (*P)[i - 1];
        float const dl = d.GetLength();
        if (dl <= 1e-6f) continue;
        d /= dl;
        bool found = false;
        GfVec3f Xn;
        if (haveX) {
            GfVec3f const xc = X - d * GfDot(X, d);
            float const l = xc.GetLength();
            if (l > 1e-6f) { Xn = xc / l; found = true; }
        }
        if (!found) {
            GfVec3f const xc = GfCross(d, R);
            float const l = xc.GetLength();
            if (l <= 1e-6f) continue;
            Xn = xc / l;
        }
        X = Xn;
        haveX = true;
        GfVec3f const Y = Nrm(GfCross(d, X));
        if (w > 1e-4f) {
            float pn = 0.0f;
            if (noise > 1e-4f) pn = strand::ValueNoise1(t * 4.0f + ofs, kGolden) * noise * 2.0f * kPi;
            float const a = std::sin(t * fx * 2.0f * kPi + pn);
            float const b = std::sin(t * fy * 2.0f * kPi + pn + 0.5f * kPi);
            float const a2 = a * cphi - b * sphi, b2 = a * sphi + b * cphi;
            disp[i] = (X * a2 + Y * b2) * (w * A);
        }
    }
    for (size_t i = 0; i < n; ++i) (*P)[i] += disp[i];
}

/// Signed distance to the scalp: closest point, interpolated normal, (p - q).n.
bool ScalpDistance(Context const &ctx, GfVec3f const &p, GfVec3f *normal, float *sd)
{
    if (!ctx.scalp || ctx.scalp->Empty()) return false;
    strand::ClosestHit const hit = ctx.scalp->Closest(p);
    if (!hit.hit) return false;
    *normal = ctx.scalp->Normal(hit.triangle, hit.bary);
    *sd = GfDot(p - hit.point, *normal);
    return true;
}

void PushOut(Card const &card, Context const &ctx, std::vector<GfVec3f> *P)
{
    size_t const n = P->size();
    if (!card.pushOutEnabled || n < 2 || !ctx.scalp || ctx.scalp->Empty()) return;
    float const d0 = std::max(1e-4f, card.pushOutRoot), d1 = std::max(1e-4f, card.pushOutTip);
    auto threshold = [&](size_t i) { return d0 + (d1 - d0) * (float(i) / float(n - 1)); };
    bool any = false;
    for (size_t i = 1; i < n && !any; ++i) {
        GfVec3f nrm;
        float sd = 0.0f;
        any = ScalpDistance(ctx, (*P)[i], &nrm, &sd) && sd < threshold(i);
    }
    if (!any) return;
    std::vector<GfVec3f> const orig = *P;
    std::vector<float> seg(n, 0.0f);
    for (size_t i = 1; i < n; ++i) seg[i] = (orig[i] - orig[i - 1]).GetLength();
    int const K = std::min(std::max(int(std::lround(strand::Clamp01(card.pushOutSmoothness) * 4.0f)), 1), 8);
    std::vector<GfVec3f> disp(n, GfVec3f(0.0f)), tmp(n);
    for (int iteration = 0; iteration < 16; ++iteration) {
        for (size_t i = 1; i < n; ++i) {
            GfVec3f nrm;
            float sd = 0.0f;
            float const thr = threshold(i);
            if (ScalpDistance(ctx, (*P)[i], &nrm, &sd) && sd < thr) disp[i] += nrm * ((thr - sd) * 0.8f);
        }
        for (size_t i = 0; i < n; ++i) {
            GfVec3f acc(0.0f);
            for (int k = -K; k <= K; ++k) {
                long const j = std::min(std::max(long(i) + k, 0L), long(n) - 1);
                acc += disp[size_t(j)];
            }
            tmp[i] = acc / float(2 * K + 1);
        }
        disp.swap(tmp);
        disp[0] = GfVec3f(0.0f);
        (*P)[0] = orig[0];
        for (size_t i = 1; i < n; ++i) (*P)[i] = orig[i] + disp[i];
        for (size_t i = 1; i < n; ++i)
            (*P)[i] = (*P)[i - 1] + Nrm((*P)[i] - (*P)[i - 1]) * seg[i];
    }
}

// --- root normal and side reference ------------------------------------------------

GfVec3f OwnRootNormal(Card const &card, Context const &ctx)
{
    std::vector<GfVec3f> const &p = card.points;
    strand::TriangleBvh const *scalp = ctx.scalp;
    if (scalp && !scalp->Empty() && card.rootTri >= 0 &&
        size_t(card.rootTri) < scalp->TriangleCount()) {
        GfVec3f b(Clamp(card.rootBary[0], 0.0f, 1.0f), Clamp(card.rootBary[1], 0.0f, 1.0f),
                  Clamp(card.rootBary[2], 0.0f, 1.0f));
        float const sum = b[0] + b[1] + b[2];
        b = sum > 1e-8f ? b / sum : GfVec3f(1.0f, 0.0f, 0.0f);
        // The binding indexes the scalp's fan triangles in face order. When
        // it does not land on the authored root (re-triangulated or
        // re-ordered mesh) the closest scalp point stands in.
        bool const bound = p.empty() ||
            (scalp->Position(card.rootTri, b) - p[0]).GetLength() <= ctx.bindTolerance;
        if (bound) return scalp->Normal(card.rootTri, b);
        strand::ClosestHit const hit = scalp->Closest(p[0]);
        if (hit.hit) return scalp->Normal(hit.triangle, hit.bary);
    }
    if (p.size() < 2) return GfVec3f(0.0f, 1.0f, 0.0f);
    return Nrm(p[1] - p[0]);
}

/// Free / Mesh Normal side: perpendicular to the root -> tip chord.
GfVec3f SideFromChord(std::vector<GfVec3f> const &spine, GfVec3f const &N, int excludedAxis)
{
    if (spine.size() >= 2) {
        GfVec3f const ch = spine.back() - spine.front();
        float const l = ch.GetLength();
        if (l > 1e-8f) {
            GfVec3f const c = ch / l;
            if ((c - N * GfDot(c, N)).GetLength() > 0.02f) {
                GfVec3f const x = GfCross(c, N);
                if (GfDot(x, x) > 1e-12f) return Nrm(x);
            }
        }
    }
    for (int axis : {1, 0, 2}) {
        if (axis == excludedAxis) continue;
        GfVec3f e(0.0f);
        e[axis] = 1.0f;
        if (std::fabs(GfDot(N, e)) < 0.95f) {
            GfVec3f const x = GfCross(N, e);
            if (GfDot(x, x) > 1e-8f) return Nrm(x);
        }
    }
    return GfVec3f(1.0f, 0.0f, 0.0f);
}

/// Surface / Conform side: perpendicular to the first near-root segment that
/// is not parallel to the root normal.
GfVec3f SideFromRootSegment(std::vector<GfVec3f> const &spine, GfVec3f const &N, int excludedAxis)
{
    if (spine.size() >= 2) {
        std::vector<float> s;
        float const L = CumulativeLengths(spine, &s);
        float acc = 0.0f;
        for (size_t i = 0; i + 1 < spine.size(); ++i) {
            if (0.25f * L < acc) break;
            GfVec3f const seg = spine[i + 1] - spine[i];
            float const l = seg.GetLength();
            acc += l;
            if (l >= 1e-8f) {
                GfVec3f const x = GfCross(seg / l, N);
                float const xl = x.GetLength();
                if (xl > 0.35f) return x / xl;
            }
        }
    }
    return SideFromChord(spine, N, excludedAxis);
}

void OwnRootFrame(Card const &card, Context const &ctx, GfVec3f *n0, GfVec3f *s0)
{
    *n0 = OwnRootNormal(card, ctx);
    int const excluded = card.twinId != 0 ? PlaneAxis(card.mirrorPlane) : -1;
    *s0 = (card.orientation == kSurface || card.orientation == kConform)
        ? SideFromRootSegment(card.points, *n0, excluded)
        : SideFromChord(card.points, *n0, excluded);
}

// --- frames ---------------------------------------------------------------------------

struct Frame { GfVec3f T, S, B; };

void Frames(std::vector<GfVec3f> const &C, GfVec3f const &side0, int orientation,
            std::vector<GfVec3f> const *conformN, float sign, std::vector<Frame> *out)
{
    size_t const m = C.size();
    out->resize(m);
    GfVec3f prev = side0;
    for (size_t i = 0; i < m; ++i) {
        GfVec3f const T = Nrm(C[std::min(i + 1, m - 1)] - C[i > 0 ? i - 1 : 0]);
        GfVec3f const ref = orientation == kMeshNormal ? side0 : prev;
        GfVec3f S = ref - T * GfDot(ref, T);
        if (GfDot(S, S) < 1e-8f) S = prev;
        S = Nrm(S);
        if (GfDot(S, S) < 0.5f) S = prev;
        if (orientation == kConform && conformN) {
            GfVec3f c = GfCross(T, (*conformN)[i]) * sign;
            float const cl = c.GetLength();
            float const w = strand::SmoothStep01((cl - 0.17365f) / 0.32635f);
            if (w > 0.0f) {
                c /= cl;
                float const angle = std::atan2(GfDot(GfCross(S, c), T), GfDot(S, c));
                GfVec3f const S2 = Nrm(strand::Rotate(S, T, angle * w));
                if (GfDot(S2, S2) > 0.5f) S = S2;
            }
        }
        prev = S;
        (*out)[i] = Frame{T, S, Nrm(GfCross(S, T))};
    }
}

/// Scalp normals along the sunk spine, smoothed twice (Conform). False when a
/// query fails (the card then behaves as Surface).
bool ConformNormals(std::vector<GfVec3f> const &C, Context const &ctx, bool mirrored,
                    int plane, std::vector<GfVec3f> *out)
{
    if (!ctx.scalp || ctx.scalp->Empty()) return false;
    size_t const m = C.size();
    out->resize(m);
    for (size_t i = 0; i < m; ++i) {
        GfVec3f const q = mirrored ? MirrorPoint(C[i], plane, ctx.bboxCentre) : C[i];
        strand::ClosestHit const hit = ctx.scalp->Closest(q);
        if (!hit.hit) return false;
        GfVec3f const n = ctx.scalp->Normal(hit.triangle, hit.bary);
        (*out)[i] = mirrored ? MirrorVector(n, plane) : n;
    }
    std::vector<GfVec3f> tmp(m);
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < m; ++i)
            tmp[i] = Nrm((*out)[i > 0 ? i - 1 : 0] + (*out)[i] * 2.0f + (*out)[std::min(i + 1, m - 1)],
                         (*out)[i]);
        out->swap(tmp);
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------

void Card::Sanitize()
{
    auto c01 = [](float &v, float fallback) { v = strand::Clamp01(Finite(v, fallback)); };
    cardType = std::min(std::max(cardType, 0), 3);
    orientation = std::min(std::max(orientation, 0), 3);
    widthSegments = std::min(std::max(widthSegments, 1), 32);
    for (float &b : vBias) b = Clamp(Finite(b, 0.5f), 0.05f, 0.95f);
    width = Clamp(Finite(width, 0.05f), 1e-9f, 1e9f);
    c01(widthRoot, 0.5f); c01(widthMid, 0.5f); c01(widthTip, 0.5f);
    widthMidShift = Clamp(Finite(widthMidShift, 0.5f), 0.1f, 0.9f);
    crossWidth = Clamp(Finite(crossWidth, 1.0f), 0.05f, 1.0f);
    tubeFlatten = Clamp(Finite(tubeFlatten, 0.0f), 0.0f, 0.98f);
    centerRidge = Clamp(Finite(centerRidge, 0.0f), -1.0f, 1.0f);
    c01(ridgeRootFlatten, 0.0f);
    sink = std::max(0.0f, Finite(sink, 0.0f));
    twistRoot = Clamp(Finite(twistRoot, 0.0f), -180.0f, 180.0f);
    twistMid = Clamp(Finite(twistMid, 0.0f), -180.0f, 180.0f);
    twistTip = Clamp(Finite(twistTip, 0.0f), -180.0f, 180.0f);
    twistMidShift = Clamp(Finite(twistMidShift, 0.5f), 0.1f, 0.9f);
    lengthMin = Clamp(Finite(lengthMin, 0.01f), 0.01f, 2.0f);
    lengthMax = Clamp(Finite(lengthMax, 1.0f), 0.01f, 2.0f);
    c01(lengthRandomize, 0.0f);
    c01(distributionMask, 1.0f);
    c01(lengthMask, 1.0f);
    c01(frizzAmount, 0.0f);
    frizzFrequency = Clamp(Finite(frizzFrequency, 0.5f), 0.0f, 10.0f);
    frizzFrequencyZ = Clamp(Finite(frizzFrequencyZ, 0.5f), 0.0f, 10.0f);
    c01(frizzNoise, 0.0f); c01(frizzRoot, 0.0f); c01(frizzMid, 0.5f); c01(frizzMidShift, 0.5f);
    c01(frizzTip, 1.0f);
    pushOutRoot = std::max(0.0f, Finite(pushOutRoot, 0.0f));
    pushOutTip = std::max(0.0f, Finite(pushOutTip, 0.01f));
    c01(pushOutSmoothness, 0.5f);
    for (int k = 0; k < 3; ++k) rootBary[k] = Finite(rootBary[k], 0.0f);
}

bool Included(Card const &card)
{
    return card.uvId >= 0 && card.uvId <= 9 && !card.erased && card.distributionMask >= 0.5f &&
        card.points.size() >= 2 && card.cardType != kBraid;
}

std::vector<GfVec3f> CatmullRom4(std::vector<GfVec3f> const &p)
{
    size_t const n = p.size();
    if (n < 2) return p;
    std::vector<GfVec3f> out;
    out.reserve(4 * n - 3);
    for (size_t i = 0; i + 1 < n; ++i) {
        GfVec3f const &P0 = p[i > 0 ? i - 1 : 0], &P1 = p[i], &P2 = p[i + 1],
                      &P3 = p[std::min(i + 2, n - 1)];
        for (int k = 0; k < 4; ++k) {
            float const t = 0.25f * float(k), t2 = t * t, t3 = t2 * t;
            out.push_back((P1 * 2.0f + (P2 - P0) * t + (P0 * 2.0f - P1 * 5.0f + P2 * 4.0f - P3) * t2 +
                           (P1 * 3.0f - P0 - P2 * 3.0f + P3) * t3) * 0.5f);
        }
    }
    out.push_back(p[n - 1]);
    return out;
}

std::vector<GfVec3f> ProcessSpine(Card const &card, Context const &ctx)
{
    std::vector<GfVec3f> P = card.points;
    if (P.size() < 2) return P;
    // Mirror-secondary cards share the twin's seed. The length stage commutes
    // with the mirror; frizz does not (its frame is handed), so it runs in the
    // twin's space.
    LengthScale(card, &P);
    if (card.MirrorSecondary()) {
        for (GfVec3f &p : P) p = MirrorPoint(p, card.mirrorPlane, ctx.bboxCentre);
        Frizz(card, &P);
        for (GfVec3f &p : P) p = MirrorPoint(p, card.mirrorPlane, ctx.bboxCentre);
    } else {
        Frizz(card, &P);
    }
    PushOut(card, ctx, &P);
    return P;
}

void RootFrame(Card const &card, Context const &ctx, GfVec3f *n0, GfVec3f *s0, bool *twinFound)
{
    *twinFound = false;
    if (card.MirrorSecondary()) {
        if (Card const *twin = FindTwin(card, ctx)) {
            OwnRootFrame(*twin, ctx, n0, s0);
            *n0 = MirrorVector(*n0, card.mirrorPlane);
            *s0 = MirrorVector(*s0, card.mirrorPlane);
            *twinFound = true;
            return;
        }
    }
    OwnRootFrame(card, ctx, n0, s0);
}

std::vector<Span> BuildCard(uint32_t index, Context const &ctx, bool honourInclude)
{
    std::vector<Span> spans;
    if (!ctx.cards || index >= ctx.cards->size()) return spans;
    Card const &card = (*ctx.cards)[index];
    if (card.cardType == kBraid || card.points.size() < 2) return spans;
    if (honourInclude && !Included(card)) return spans;

    bool const tube = card.cardType == kTube;
    std::vector<GfVec3f> P = ProcessSpine(card, ctx);
    size_t const n = P.size();
    if (ctx.smooth && n >= 2 && (n >= 3 || tube)) P = CatmullRom4(P);
    std::vector<float> s;
    float const L = CumulativeLengths(P, &s);
    float const invL = L > 1e-6f ? 1.0f / L : 0.0f;
    size_t const rows = P.size();

    GfVec3f N0, S0;
    bool twinFound = false;
    RootFrame(card, ctx, &N0, &S0, &twinFound);
    bool const secondary = card.MirrorSecondary();
    float const sT = secondary ? -1.0f : 1.0f;
    float const sN = (secondary && twinFound) ? -1.0f : 1.0f;

    // Sink the root along N0, faded out over F.
    std::vector<GfVec3f> C = P;
    if (card.sink > 1e-6f) {
        float const F = std::min(L, std::max(2.0f * card.width, 0.25f * L));
        for (size_t i = 0; i < rows; ++i) {
            float const w = F <= 1e-6f ? 1.0f : 1.0f - strand::SmoothStep01(s[i] / F);
            C[i] -= N0 * (card.sink * w);
        }
    }

    std::vector<GfVec3f> conformN;
    bool const conform = card.orientation == kConform &&
        ConformNormals(C, ctx, secondary && twinFound, card.mirrorPlane, &conformN);
    std::vector<Frame> frames;
    Frames(C, S0, card.orientation, conform ? &conformN : nullptr, sN, &frames);

    // Shape.
    float r = card.widthRoot, m = card.widthMid, tp = card.widthTip;
    if (r + m + tp < 1e-6f) m = 1.0f;
    float const ridge = (card.cardType == kRibbon && card.widthSegments > 1) ? card.centerRidge : 0.0f;
    float const rrf = card.ridgeRootFlatten;
    bool const hasTwist = !(card.twistRoot == 0.0f && card.twistMid == 0.0f &&
                            card.twistMidShift == 0.5f && card.twistTip == 0.0f);
    float const flat = std::max(0.02f, 1.0f - card.tubeFlatten);
    int const segs = std::min(std::max(card.widthSegments, tube ? 3 : 1), 32);
    uint32_t const cols = uint32_t(segs * ((ctx.smooth && tube) ? 4 : 1) + 1);

    struct Strip { float angle, widthFactor; int slot; };
    std::vector<Strip> strips{{0.0f, 1.0f, 0}};
    if (card.cardType == kCross) strips.push_back({0.5f * kPi, card.crossWidth, 1});

    for (size_t k = 0; k < strips.size(); ++k) {
        Strip const &strip = strips[k];
        Span span;
        span.card = index;
        span.strip = int(k);
        span.uvId = card.uvId;
        span.tile = card.tile[strip.slot];
        span.rows = uint32_t(rows);
        span.cols = cols;
        span.tube = tube;
        span.flipU = !tube && sN > 0.0f;
        span.vBias = card.vBias[strip.slot];
        span.V.resize(rows * cols);
        span.N.resize(rows * cols);
        span.rowParam.resize(rows);
        span.rowWidth.resize(rows);
        for (size_t i = 0; i < rows; ++i) {
            float const t = s[i] * invL;
            span.rowParam[i] = t;
            float const hw = strand::Clamp01(strand::BezierProfile(t, r, m, card.widthMidShift, tp)) *
                card.width * 0.5f * strip.widthFactor;
            GfVec3f const T = frames[i].T;
            GfVec3f S = frames[i].S, B = frames[i].B;
            if (hasTwist || std::fabs(strip.angle) > 1e-6f) {
                float const a = (strand::BezierProfile(t, card.twistRoot, card.twistMid,
                                                       card.twistMidShift, card.twistTip) *
                                     kDegToRad + strip.angle) * sT;
                S = Nrm(strand::Rotate(S, T, a));
                B = Nrm(GfCross(S, T));
            }
            float const flatR = rrf > 0.0f
                ? 1.0f - (1.0f - strand::SmoothStep01(t / (0.3f * rrf))) * rrf : 1.0f;
            for (uint32_t j = 0; j < cols; ++j) {
                float const u = cols < 2 ? 0.5f : float(j) / float(cols - 1);
                GfVec3f p, nv;
                if (!tube) {
                    float const x = 2.0f * u - 1.0f;
                    p = C[i] + S * (x * hw);
                    if (ridge != 0.0f) p += B * ((1.0f - x * x) * ridge * sT * flatR * hw);
                    nv = B * sN;
                } else {
                    float const th = u * 2.0f * kPi - 0.5f * kPi;
                    float const ct = std::cos(th), st = std::sin(th);
                    p = C[i] + S * (ct * hw) + B * (hw * flat * st);
                    nv = Nrm(S * (hw * flat * ct) + B * (st * hw), B);
                }
                span.V[i * cols + j] = p;
                span.N[i * cols + j] = nv;
            }
            GfVec3f const *row = &span.V[i * cols];
            if (tube) {
                float w = 0.0f;
                for (uint32_t j = 1; j < cols; ++j) w += (row[j] - row[j - 1]).GetLength();
                span.rowWidth[i] = w;
            } else {
                span.rowWidth[i] = (row[cols - 1] - row[0]).GetLength();
            }
        }
        uint32_t const mid = cols / 2;
        for (size_t i = 1; i < rows; ++i)
            span.length += (span.V[i * cols + mid] - span.V[(i - 1) * cols + mid]).GetLength();
        spans.push_back(std::move(span));
    }
    return spans;
}

void CardSurface(Span const &span, float u, float v, float depth, GfVec3f *position,
                 GfVec3f *normal)
{
    *position = GfVec3f(0.0f);
    *normal = GfVec3f(1.0f, 0.0f, 0.0f);
    uint32_t const rows = span.rows, cols = span.cols;
    if (rows < 2 || cols < 2 || span.V.size() < size_t(rows) * cols) return;
    v = strand::Clamp01(v);
    float const b = span.vBias;
    if (b != 0.5f) {
        float const den = (1.0f - v) * b + (1.0f - b) * v;
        v = den > 0.0f ? (1.0f - b) * v / den : v;
    }
    std::vector<float> const &rp = span.rowParam;
    size_t i = size_t(std::upper_bound(rp.begin(), rp.begin() + rows, v) - rp.begin());
    i = std::min(std::max<size_t>(i, 1), size_t(rows) - 1);
    float const d = rp[i] - rp[i - 1];
    float const t = d <= 0.0f ? 0.0f : strand::Clamp01((v - rp[i - 1]) / d);
    u = strand::Clamp01(u);
    if (span.flipU) u = 1.0f - u;
    float const x = float(cols - 1) * u;
    uint32_t const j = uint32_t(std::min(std::max(int(x), 0), int(cols) - 2));
    float const sx = x - float(j);
    auto bilerp = [&](std::vector<GfVec3f> const &A) {
        size_t const a = (i - 1) * cols + j, c = i * cols + j;
        return (A[a] * (1.0f - sx) + A[a + 1] * sx) * (1.0f - t) +
               (A[c] * (1.0f - sx) + A[c + 1] * sx) * t;
    };
    GfVec3f p = bilerp(span.V);
    GfVec3f const nrm = Nrm(bilerp(span.N));
    if (depth != 0.0f)
        p += nrm * (depth * ((1.0f - t) * span.rowWidth[i - 1] + t * span.rowWidth[i]));
    *position = p;
    *normal = nrm;
}

// --- reference-set planes ---------------------------------------------------------------

UsdGenPlane const *CurvePlane(UsdGenCurveBuffer const &b, TfToken const &name, uint8_t arity)
{
    for (UsdGenPlane const &p : b.extraCurve) {
        if (p.name != name || p.arity != arity || p.interpolation != TfToken("uniform")) continue;
        size_t const values = size_t(b.totalCurves) * arity;
        if ((p.type == TfToken("int") && p.i.size() == values) ||
            (p.type == TfToken("float") && p.f.size() == values))
            return &p;
    }
    return nullptr;
}

PlaneColumn::PlaneColumn(UsdGenCurveBuffer const &b, char const *name)
    : plane(CurvePlane(b, TfToken(name)))
{
}

int PlaneColumn::Int(size_t c, int fallback) const
{
    if (!plane) return fallback;
    if (plane->type == TfToken("int")) return plane->i[c];
    float const v = plane->f[c];
    return std::isfinite(v) ? int(std::lround(v)) : fallback;
}

float PlaneColumn::Float(size_t c, float fallback) const
{
    if (!plane) return fallback;
    return plane->type == TfToken("int") ? float(plane->i[c]) : plane->f[c];
}

bool IsSpineSet(UsdGenCurveBuffer const &b) { return PlaneColumn(b, "cardType").Has(); }
bool IsLibrarySet(UsdGenCurveBuffer const &b)
{
    return !IsSpineSet(b) && PlaneColumn(b, "tile").Has() && PlaneColumn(b, "index").Has();
}

static uint64_t U64(int lo, int hi)
{
    return uint64_t(uint32_t(lo)) | (uint64_t(uint32_t(hi)) << 32);
}

bool DecodeCards(UsdGenCurveBuffer const &b, std::vector<Card> *cards, std::string *error)
{
    std::vector<uint32_t> spans;
    if (!opUtil::CurveSpans(b, &spans, error)) return false;
    PlaneColumn const idLo(b, "cardIdLo"), idHi(b, "cardIdHi"), twLo(b, "twinIdLo"),
        twHi(b, "twinIdHi"), mirror(b, "mirrorPlane"), layer(b, "layer"), uvId(b, "uvId"),
        tile(b, "tile"), tile2(b, "tile2"), tile3(b, "tile3"), vb(b, "vBias"), vb2(b, "vBias2"),
        vb3(b, "vBias3"), type(b, "cardType"), orient(b, "orientation"), segs(b, "widthSegments"),
        rootTri(b, "rootTri"), erased(b, "erased"), dmask(b, "distributionMask"),
        lmask(b, "lengthMask"), width(b, "width"), wr(b, "widthRoot"), wm(b, "widthMid"),
        wms(b, "widthMidShift"), wt(b, "widthTip"), cross(b, "crossWidth"),
        tubeFlat(b, "tubeFlatten"), ridge(b, "centerRidge"), rrf(b, "ridgeRootFlatten"),
        sink(b, "sink"), tr(b, "twistRoot"), tm(b, "twistMid"), tms(b, "twistMidShift"),
        tt(b, "twistTip"), lmin(b, "lengthMin"), lmax(b, "lengthMax"), lrand(b, "lengthRandomize"),
        fa(b, "frizzAmount"), ff(b, "frizzFrequency"), ffz(b, "frizzFrequencyZ"),
        fn(b, "frizzNoise"), fr(b, "frizzRoot"), fm(b, "frizzMid"), fms(b, "frizzMidShift"),
        ft(b, "frizzTip"), po(b, "pushOutEnabled"), por(b, "pushOutRoot"), pot(b, "pushOutTip"),
        pos(b, "pushOutSmoothness");
    UsdGenPlane const *bary = CurvePlane(b, TfToken("rootBary"), 3);
    if (bary && bary->type != TfToken("float")) bary = nullptr;
    cards->assign(b.totalCurves, Card());
    for (uint32_t c = 0; c < b.totalCurves; ++c) {
        Card &k = (*cards)[c];
        for (uint32_t cv = spans[c]; cv < spans[c + 1]; ++cv) k.points.push_back(opUtil::RestPoint(b, cv));
        k.id = U64(idLo.Int(c, int(c + 1)), idHi.Int(c, 0));
        k.twinId = U64(twLo.Int(c, 0), twHi.Int(c, 0));
        k.mirrorPlane = mirror.Int(c, 0);
        k.layer = layer.Int(c, 0);
        k.uvId = uvId.Int(c, -1);
        k.tile[0] = tile.Int(c, -1);
        k.tile[1] = tile2.Int(c, k.tile[0]);
        k.tile[2] = tile3.Int(c, k.tile[0]);
        k.vBias[0] = vb.Float(c, 0.5f);
        k.vBias[1] = vb2.Float(c, k.vBias[0]);
        k.vBias[2] = vb3.Float(c, k.vBias[0]);
        k.cardType = type.Int(c, 0);
        k.orientation = orient.Int(c, kSurface);
        k.widthSegments = segs.Int(c, 1);
        k.rootTri = rootTri.Int(c, -1);
        if (bary) k.rootBary = GfVec3f(bary->f[3 * c], bary->f[3 * c + 1], bary->f[3 * c + 2]);
        k.erased = erased.Int(c, 0) != 0;
        k.distributionMask = dmask.Float(c, 1.0f);
        k.lengthMask = lmask.Float(c, 1.0f);
        k.width = width.Float(c, 0.05f);
        k.widthRoot = wr.Float(c, 0.5f);
        k.widthMid = wm.Float(c, 0.5f);
        k.widthMidShift = wms.Float(c, 0.5f);
        k.widthTip = wt.Float(c, 0.5f);
        k.crossWidth = cross.Float(c, 1.0f);
        k.tubeFlatten = tubeFlat.Float(c, 0.0f);
        k.centerRidge = ridge.Float(c, 0.0f);
        k.ridgeRootFlatten = rrf.Float(c, 0.0f);
        k.sink = sink.Float(c, 0.0f);
        k.twistRoot = tr.Float(c, 0.0f);
        k.twistMid = tm.Float(c, 0.0f);
        k.twistMidShift = tms.Float(c, 0.5f);
        k.twistTip = tt.Float(c, 0.0f);
        k.lengthMin = lmin.Float(c, 0.01f);
        k.lengthMax = lmax.Float(c, 1.0f);
        k.lengthRandomize = lrand.Float(c, 0.0f);
        k.frizzAmount = fa.Float(c, 0.0f);
        k.frizzFrequency = ff.Float(c, 0.5f);
        k.frizzFrequencyZ = ffz.Float(c, 0.5f);
        k.frizzNoise = fn.Float(c, 0.0f);
        k.frizzRoot = fr.Float(c, 0.0f);
        k.frizzMid = fm.Float(c, 0.5f);
        k.frizzMidShift = fms.Float(c, 0.5f);
        k.frizzTip = ft.Float(c, 1.0f);
        k.pushOutEnabled = po.Int(c, 0) != 0;
        k.pushOutRoot = por.Float(c, 0.0f);
        k.pushOutTip = pot.Float(c, 0.01f);
        k.pushOutSmoothness = pos.Float(c, 0.5f);
        k.Sanitize();
    }
    return true;
}


void MeshCounts(std::vector<Span> const &spans, uint64_t *vertices, uint64_t *indices)
{
    uint64_t v = 0, x = 0;
    for (Span const &s : spans) {
        v += uint64_t(s.rows) * s.cols;
        if (s.rows >= 1 && s.cols >= 1) x += 6ull * (s.rows - 1) * (s.cols - 1);
    }
    *vertices = v;
    *indices = x;
}

}  // namespace card
}  // namespace usdGen
