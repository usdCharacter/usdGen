// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — reference guide blending (see guideBlend.h).
//
// Conventions kept from the reference so results match bit for bit where
// the reference's helpers are known:
//   * every length is sqrtf((x*x + y*y) + z*z) and every dot product
//     (a.x*b.x + a.y*b.y) + a.z*b.z, evaluated in single precision;
//   * clamps are written as the reference writes them (lo first, then hi);
//   * "normalize" divides each component by the length and falls back to
//     (1,0,0) at length <= 1e-6 unless a different fallback is stated;
//   * float sums run in the reference's sequential order.
#include "usdGen/ops/guideBlend.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace usdGen {
namespace guideBlend {

namespace {

constexpr float kFltMax = 3.4028235e+38f;

inline V3 Make(float x, float y, float z)
{
    V3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}
inline V3 Sub(V3 const &a, V3 const &b) { return Make(a.x - b.x, a.y - b.y, a.z - b.z); }
inline float Dot(V3 const &a, V3 const &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float Len(V3 const &v) { return std::sqrt(Dot(v, v)); }
inline V3 Cross(V3 const &a, V3 const &b)
{
    return Make(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline V3 Div(V3 const &v, float l) { return Make(v.x / l, v.y / l, v.z / l); }

/// The reference's clamp: lower bound first, then upper.
inline float ClampRef(float x, float lo, float hi)
{
    float const a = lo <= x ? x : lo;
    return a <= hi ? a : hi;
}

/// Normalize with the (1,0,0) fallback at length <= 1e-6.
inline V3 NormalizeX(V3 const &v)
{
    float const l = Len(v);
    if (l <= 1e-6f) return Make(1.0f, 0.0f, 0.0f);
    return Div(v, l);
}

/// a * (1 - t) + b * t.
inline V3 Mix(V3 const &a, V3 const &b, float t)
{
    float const w = 1.0f - t;
    return Make(a.x * w + b.x * t, a.y * w + b.y * t, a.z * w + b.z * t);
}

inline float SmoothRef(float x)
{
    x = ClampRef(x, 0.0f, 1.0f);
    return (3.0f - (x + x)) * x * x;
}

/// Rodrigues rotation of v about unit k by angle a.
V3 Rotate(V3 const &v, V3 const &k, float a)
{
    float const c = std::cos(a);
    float const s = std::sin(a);
    float const d = v.x * k.x + v.y * k.y + v.z * k.z;
    float const omc = 1.0f - c;
    return Make((k.y * v.z - k.z * v.y) * s + v.x * c + k.x * d * omc,
                (v.x * k.z - k.x * v.z) * s + v.y * c + k.y * d * omc,
                (k.x * v.y - v.x * k.y) * s + v.z * c + k.z * d * omc);
}

/// d x (0,1,0) as the reference computes its last-resort bend axis.
inline V3 CrossUpFallback(V3 const &d)
{
    return Make(d.y * 0.0f - d.z, d.z * 0.0f - d.x * 0.0f, d.x - d.y * 0.0f);
}

/// The bend axis of a parting bend: d x target, else d x root normal, else
/// d x Y. False when all three degenerate. The strand-level bend also
/// rejects a final length of exactly 1e-6 (`strict`).
bool BendAxis(V3 const &d, V3 const &target, V3 const &rootNormal, bool strict, V3 *axis)
{
    V3 a = Cross(d, target);
    float l = Len(a);
    if (l < 1e-6f) {
        V3 const n = NormalizeX(rootNormal);
        a = Cross(d, n);
        l = Len(a);
        if (l < 1e-6f) {
            a = CrossUpFallback(d);
            l = Len(a);
            if (l < 1e-6f) return false;
        }
    }
    if (strict && !(1e-6f < l)) return false;
    *axis = Div(a, l);
    return true;
}

/// The tangent the reference's frame builder starts from, at arc length s on
/// the raw guide: the Catmull-Rom derivative of the segment holding s.
V3 CatmullRomTangent(V3 const *p, size_t count, float s)
{
    if (count < 2) return Make(0.0f, 1.0f, 0.0f);
    if (s <= 0.0f) s = 0.0f;
    for (size_t k = 0;; ++k) {
        V3 const seg = Sub(p[k + 1], p[k]);
        float const l = Len(seg);
        if (s <= l || k + 2 == count) {
            float u = 0.0f;
            if (!(l <= 1e-6f)) u = ClampRef(s / l, 0.0f, 1.0f);
            V3 const &a = p[k == 0 ? 0 : k - 1];
            V3 const &b = p[k];
            V3 const &c = p[k + 1];
            V3 const &d = p[k + 2 < count ? k + 2 : k + 1];
            float const u3 = u * 3.0f * u;
            float const u2 = u + u;
            auto comp = [&](float pa, float pb, float pc, float pd) {
                return ((((pa * 2.0f - pb * 5.0f) + pc * 4.0f) - pd) * u2 + (pc - pa) +
                        (((pb * 3.0f + (0.0f - pa)) - pc * 3.0f) + pd) * u3) * 0.5f;
            };
            V3 const tangent = Make(comp(a.x, b.x, c.x, d.x), comp(a.y, b.y, c.y, d.y),
                                    comp(a.z, b.z, c.z, d.z));
            float const tl = Len(tangent);
            if (tl > 1e-6f) return Div(tangent, tl);
            V3 const sd = Sub(c, b);
            float const sl = Len(sd);
            if (sl > 1e-6f) return Div(sd, sl);
            return Make(0.0f, 1.0f, 0.0f);
        }
        s = s - l;
    }
}

/// Rotation-minimising frames along `count` given tangents (projection
/// method), into T/N/B.
void TransportFrames(V3 const *tangent, size_t count, V3 *outT, V3 *outN, V3 *outB)
{
    if (count == 0) return;
    V3 const t0 = tangent[0];
    bool const steep = 0.95f <= std::fabs(t0.y);
    float const upX = steep ? 1.0f : 0.0f, upY = steep ? 0.0f : 1.0f;
    V3 n = Make(t0.y * 0.0f - t0.z * upY, upX * t0.z - t0.x * 0.0f, t0.x * upY - upX * t0.y);
    float l = Len(n);
    n = l > 1e-6f ? Div(n, l) : Make(1.0f, 0.0f, 0.0f);
    V3 b = Cross(t0, n);
    l = Len(b);
    b = l > 1e-6f ? Div(b, l) : Make(1.0f, 0.0f, 0.0f);
    V3 prevT = t0;
    for (size_t i = 0; i < count; ++i) {
        V3 t = tangent[i];
        float const lt = Len(t);
        t = lt <= 1e-6f ? t0 : Div(t, lt);
        float const d = Dot(t, n);
        V3 np = Make(n.x - t.x * d, n.y - t.y * d, n.z - t.z * d);
        float ln = Len(np);
        bool ok = 1e-5f < ln;
        if (!ok) {
            bool const s = 0.95f <= std::fabs(t.y);
            float const ux = s ? 1.0f : 0.0f, uy = s ? 0.0f : 1.0f;
            np = Make(t.y * 0.0f - t.z * uy, t.z * ux - t.x * 0.0f, t.x * uy - t.y * ux);
            ln = Len(np);
            ok = 1e-6f < ln;
        }
        if (ok) {
            n = Div(np, ln);
            V3 bb = Cross(t, n);
            float const lb = Len(bb);
            b = lb > 1e-6f ? Div(bb, lb) : Make(1.0f, 0.0f, 0.0f);
            prevT = t;
        }
        outT[i] = prevT;
        outN[i] = n;
        outB[i] = b;
    }
}

/// Point at arc length s on the resampled guide g (linear between samples;
/// extrapolates straight past the tip; the root for a degenerate guide).
V3 SampleTable(Guides const &guides, size_t g, float s)
{
    if (g >= guides.root.size()) return V3();
    int const ng = guides.ng;
    if (ng <= 0) return V3();
    size_t const base = g * size_t(ng);
    float const L = guides.length[g];
    if (guides.table.size() < base + size_t(ng) || L <= 1e-6f) return guides.root[g];
    V3 const *t = &guides.table[base];
    if (L < s && 1 < ng) {
        V3 const last = t[ng - 1];
        V3 const d = Sub(last, t[ng - 2]);
        float const l = Len(d);
        V3 const dir = l <= 1e-6f ? Make(0.0f, 1.0f, 0.0f) : Div(d, l);
        float const r = s - L;
        return Make(last.x + r * dir.x, last.y + r * dir.y, last.z + r * dir.z);
    }
    int const last = ng - 1;
    float const f = ClampRef(s / L, 0.0f, 1.0f);
    int const i0 = int(f * float(last));
    int const i1 = last < i0 + 1 ? last : i0 + 1;
    return Mix(t[i0], t[i1], f * float(last) - float(i0));
}

/// The guide's frame at arc length s (interpolated, re-orthonormalised).
bool FrameAt(Guides const &guides, size_t g, float s, V3 *T, V3 *N, V3 *B)
{
    if (!guides.frames || g >= guides.root.size()) return false;
    int const ng = guides.ng;
    size_t const base = size_t(ng) * g;
    if (base + size_t(ng) > guides.frameN.size()) return false;
    float const L = guides.length[g];
    if (!(1e-6f < L)) return false;
    int i0 = ng - 1, i1 = ng - 1;
    float f = 0.0f;
    if (s <= L && 1 < ng) {
        float const q = ClampRef(s / L, 0.0f, 1.0f);
        float const last = float(ng - 1);
        i0 = int(q * last);
        i1 = ng - 1 < i0 + 1 ? ng - 1 : i0 + 1;
        f = q * last - float(i0);
    }
    float const w = 1.0f - f;
    V3 const &ta = guides.frameT[base + size_t(i0)], &tb = guides.frameT[base + size_t(i1)];
    V3 const &na = guides.frameN[base + size_t(i0)], &nb = guides.frameN[base + size_t(i1)];
    V3 const ti = Make(w * ta.x + tb.x * f, w * ta.y + tb.y * f, w * ta.z + tb.z * f);
    V3 const ni = Make(w * na.x + nb.x * f, w * na.y + nb.y * f, w * na.z + nb.z * f);
    V3 const tn = NormalizeX(ti);
    float const d = Dot(tn, ni);
    V3 const nn = Make(ni.x - d * tn.x, ni.y - d * tn.y, ni.z - d * tn.z);
    float const l = Len(nn);
    if (1e-6f < l) {
        *T = tn;
        *N = Div(nn, l);
        *B = Cross(tn, *N);
        return true;
    }
    if (1e-6f < Len(ta) && 1e-6f < Len(na)) {
        *T = ta;
        *N = na;
        *B = guides.frameB[base + size_t(i0)];
        return true;
    }
    return false;
}

/// The root offset carried in the guide's frame at s (bendFollow), else the
/// offset itself.
V3 FrameOffset(Guides const &guides, size_t g, V3 const &delta, float s)
{
    if (guides.frames && g < guides.root.size()) {
        size_t const idx = size_t(guides.ng) * g;
        V3 T, N, B;
        if (idx < guides.frameN.size() && FrameAt(guides, g, s, &T, &N, &B)) {
            float const n0 = Dot(guides.frameN[idx], delta);
            float const b0 = Dot(guides.frameB[idx], delta);
            float const t0 = Dot(guides.frameT[idx], delta);
            return Make(N.x * n0 + B.x * b0 + T.x * t0, N.y * n0 + B.y * b0 + T.y * t0,
                        N.z * n0 + B.z * b0 + T.z * t0);
        }
    }
    return delta;
}

/// Guides on the far side of the same parting curve.
inline bool Across(PartInfo const &root, PartInfo const &guide)
{
    return guide.valid && 0 <= root.curve && root.curve == guide.curve &&
           root.side * guide.side < 0.0f;
}

/// Parting weight factor of guide g for a root (1 without parting).
float PartingFactor(Parting const &parting, PartInfo const &root, Guides const &guides, size_t g)
{
    if (!parting.active || !root.valid || !(1e-4f < root.weight) || g >= guides.part.size())
        return 1.0f;
    PartInfo const &guide = guides.part[g];
    if (!guide.valid) return 1.0f;
    float const wr = root.weight;
    float w = guide.weight * wr;
    if (w <= wr) w = wr;
    w = ClampRef(w, 0.0f, 1.0f);
    if (0 <= root.curve && root.curve == guide.curve && guide.side * root.side < 0.0f) {
        float const s = SmoothRef((w - 0.15f) / 0.70000005f);
        return s * 1e-6f + (1.0f - s);
    }
    if (1e-4f < guide.weight && 0.0f < root.offLine && 0.0f < guide.offLine) {
        float const c = ClampRef(Dot(guide.away, root.away), 0.0f, 1.0f);
        return c * w + c * w + 1.0f;
    }
    return 1.0f;
}

struct Slot
{
    int g = -1;
    float d2 = kFltMax;
};

/// The 3 nearest guides by parting-scaled distance; guides across the
/// parting are skipped first (root parting weight > 0.2), all guides
/// considered when that finds none. Ties keep the lower index.
void KNearest(Guides const &guides, Parting const &parting, PartInfo const &root, V3 const &p,
              Slot slots[3])
{
    size_t const count = guides.Count();
    auto pass = [&](bool skip) {
        for (size_t g = 0; g < count; ++g) {
            if (skip && parting.active && root.valid && 0.2f < root.weight &&
                g < guides.part.size() && Across(root, guides.part[g]))
                continue;
            V3 const d = Sub(p, guides.root[g]);
            float const f = PartingFactor(parting, root, guides, g);
            float const ff = f * f;
            float const den = 1e-8f <= ff ? ff : 1e-8f;
            float const d2 = (d.x * d.x + d.y * d.y + d.z * d.z) / den;
            for (int k = 0; k < 3; ++k) {
                if (slots[k].d2 <= d2) continue;
                for (int m = 2; m > k; --m) slots[m] = slots[m - 1];
                slots[k].g = int(g);
                slots[k].d2 = d2;
                break;
            }
        }
    };
    pass(true);
    if (slots[0].g < 0) pass(false);
}

/// Barycentric coordinates of p in the triangle (a, b, c), projected.
bool Barycentric(V3 const &p, V3 const &a, V3 const &b, V3 const &c, float w[3])
{
    V3 const e1 = Sub(b, a), e2 = Sub(c, a), v = Sub(p, a);
    float const d11 = Dot(e1, e1);
    float const d12 = Dot(e2, e1);
    float const d22 = Dot(e2, e2);
    float const p2 = Dot(v, e2);
    float const p1 = Dot(v, e1);
    float const den = d22 * d11 - d12 * d12;
    if (!(1e-12f <= std::fabs(den))) return false;
    float const b1 = (p1 * d22 - p2 * d12) / den;
    float const b2 = (p2 * d11 - p1 * d12) / den;
    w[0] = (1.0f - b1) - b2;
    w[1] = b1;
    w[2] = b2;
    return true;
}

/// One guide's contribution at strand parameter t / arc length s: the guide
/// point plus the (frame-carried) root offset, rotated about the root by the
/// parting bend computed from the guide's initial direction.
V3 GuideContribution(Guides const &guides, Parting const &parting, size_t g, V3 const &root,
                     V3 const &rootNormal, PartInfo const &rootPart, float t, float s)
{
    V3 const G = SampleTable(guides, g, s);
    V3 const &groot = guides.root[g];
    V3 const delta = Sub(root, groot);
    V3 const off = FrameOffset(guides, g, delta, s);
    V3 const e = Sub(off, delta);
    V3 const h = Make(G.x + off.x, G.y + off.y, G.z + off.z);
    if (!parting.active || !rootPart.valid || !(1e-4f < rootPart.weight) ||
        g >= guides.initialDir.size())
        return h;
    V3 d = guides.initialDir[g];
    float const l = Len(d);
    if (!(1e-6f < l)) return h;
    d = Div(d, l);
    float const tanA = parting.tanAngle;
    V3 const target = NormalizeX(Make(rootPart.away.x * tanA + d.x, rootPart.away.y * tanA + d.y,
                                      rootPart.away.z * tanA + d.z));
    float const theta = std::acos(ClampRef(Dot(d, target), -1.0f, 1.0f));
    if (!(1e-4f < theta)) return h;
    V3 axis;
    if (!BendAxis(d, target, rootNormal, false, &axis)) return h;
    float const a = (1.0f - (1.0f - parting.distance) * t) * rootPart.weight;
    if (!(1e-4f < a)) return h;
    V3 const r = Rotate(Sub(G, groot), axis, a * theta);
    return Make(r.x + root.x + e.x, r.y + root.y + e.y, r.z + root.z + e.z);
}

/// The strand-level parting bend (single-guide and straight strands): every
/// point j >= 1 rotated about the root.
void BendStrand(Parting const &parting, PartInfo const &rootPart, V3 const &root,
                V3 const &rootNormal, int n, float *p)
{
    if (!parting.active || !rootPart.valid || !(1e-4f < rootPart.weight) || n < 2) return;
    V3 d = Make(p[3] - p[0], p[4] - p[1], p[5] - p[2]);
    float const l = Len(d);
    d = l <= 1e-6f ? NormalizeX(rootNormal) : Div(d, l);
    float const tanA = parting.tanAngle;
    V3 const target = NormalizeX(Make(d.x + tanA * rootPart.away.x, d.y + tanA * rootPart.away.y,
                                      d.z + tanA * rootPart.away.z));
    float const theta = std::acos(ClampRef(Dot(target, d), -1.0f, 1.0f));
    if (!(1e-4f < theta)) return;
    V3 ax;
    if (!BendAxis(d, target, rootNormal, true, &ax)) return;
    float const wr = rootPart.weight;
    for (int j = 1; j < n; ++j) {
        float const t = n < 2 ? 0.0f : float(j) / float(n - 1);
        float const a = (1.0f - (1.0f - parting.distance) * t) * wr;
        if (!(1e-4f < a)) continue;
        float *q = p + 3 * j;
        float const vx = q[0] - root.x, vy = q[1] - root.y, vz = q[2] - root.z;
        float const ang = a * theta;
        float const c = std::cos(ang);
        float const s = std::sin(ang);
        float const dot = vx * ax.x + ax.y * vy + ax.z * vz;
        float const omc = 1.0f - c;
        q[0] = root.x + (ax.y * vz - ax.z * vy) * s + c * vx + dot * ax.x * omc;
        q[1] = root.y + vy * c + (ax.z * vx - ax.x * vz) * s + dot * ax.y * omc;
        q[2] = root.z + vz * c + (ax.x * vy - vx * ax.y) * s + dot * ax.z * omc;
    }
}

inline float StrandT(int j, int n) { return n < 2 ? 0.0f : float(j) / float(n - 1); }

/// glm::normalize: v * (1 / sqrt(dot(v, v))).
inline V3 GlmNormalize(V3 const &v)
{
    float const inv = 1.0f / std::sqrt(Dot(v, v));
    return Make(v.x * inv, v.y * inv, v.z * inv);
}

/// Symmetric 3x3 eigenvectors (cyclic Jacobi, double): columns sorted by
/// descending eigenvalue.
void SymmetricEigen(double a[3][3], double vec[3][3], double val[3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) vec[i][j] = i == j ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 64; ++sweep) {
        double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
        if (off < 1e-300) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (std::fabs(a[p][q]) < 1e-300) continue;
                double const theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                double const t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                double const c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    double const akp = a[k][p], akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for (int k = 0; k < 3; ++k) {
                    double const apk = a[p][k], aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for (int k = 0; k < 3; ++k) {
                    double const vkp = vec[k][p], vkq = vec[k][q];
                    vec[k][p] = c * vkp - s * vkq;
                    vec[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    int order[3] = {0, 1, 2};
    std::sort(order, order + 3, [&](int x, int y) {
        return a[x][x] != a[y][y] ? a[x][x] > a[y][y] : x < y;
    });
    double sorted[3][3];
    for (int c = 0; c < 3; ++c) {
        val[c] = a[order[c]][order[c]];
        for (int r = 0; r < 3; ++r) sorted[r][c] = vec[r][order[c]];
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) vec[r][c] = sorted[r][c];
}

/// The PCA plane of the guide roots: origin, U (largest spread), V.
void FitPlane(std::vector<V3> const &roots, V3 *origin, V3 *u, V3 *v)
{
    V3 o;
    for (V3 const &r : roots) {
        o.x = r.x + o.x;
        o.y = r.y + o.y;
        o.z = r.z + o.z;
    }
    float const count = float(roots.size());
    o = Make(o.x / count, o.y / count, o.z / count);
    float xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (V3 const &r : roots) {
        float const dx = r.x - o.x, dy = r.y - o.y, dz = r.z - o.z;
        xx = dx * dx + xx;
        xy = xy + dx * dy;
        xz = dx * dz + xz;
        yy = dy * dy + yy;
        yz = yz + dy * dz;
        zz = dz * dz + zz;
    }
    double a[3][3] = {{xx, xy, xz}, {xy, yy, yz}, {xz, yz, zz}};
    double vec[3][3], val[3];
    SymmetricEigen(a, vec, val);
    V3 v1 = Make(float(vec[0][0]), float(vec[1][0]), float(vec[2][0]));
    V3 v2 = Make(float(vec[0][1]), float(vec[1][1]), float(vec[2][1]));
    if (Len(Cross(v1, v2)) <= 1e-6f) {
        // Degenerate spread: any direction perpendicular to v1.
        float const l = Len(v1);
        V3 const a1 = 1e-6f < l ? Div(v1, l) : Make(1.0f, 0.0f, 0.0f);
        v1 = a1;
        bool const xAxis = 0.9f <= std::fabs(a1.x);
        float const ex = xAxis ? 0.0f : 1.0f, ey = xAxis ? 1.0f : 0.0f;
        v2 = NormalizeX(Make(ey * a1.z - a1.y * 0.0f, a1.x * 0.0f - a1.z * ex,
                             a1.y * ex - ey * a1.x));
    }
    float const l1 = Len(v1);
    V3 const U = 1e-6f < l1 ? Div(v1, l1) : Make(1.0f, 0.0f, 0.0f);
    V3 n = Cross(U, v2);
    float ln = Len(n);
    n = 1e-6f < ln ? Div(n, ln) : Make(1.0f, 0.0f, 0.0f);
    V3 w = Cross(n, U);
    float lw = Len(w);
    V3 const V = 1e-6f < lw ? Div(w, lw) : Make(1.0f, 0.0f, 0.0f);
    *origin = o;
    *u = U;
    *v = V;
}

/// (max(1e-6, 0.5 * mean 2-D nearest-neighbour distance))^2; 1e-6 for < 2.
float NearestNeighbourSigma2(std::vector<float> const &q)
{
    size_t const count = q.size() / 2;
    if (count < 2) return 1e-6f;
    int found = 0;
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float best = kFltMax;
        for (size_t j = 0; j < count; ++j) {
            if (i == j) continue;
            float const dx = q[2 * i] - q[2 * j], dy = q[2 * i + 1] - q[2 * j + 1];
            float const d = std::sqrt(dy * dy + dx * dx);
            if (d < best) best = d;
        }
        if (best < kFltMax) {
            sum = sum + best;
            ++found;
        }
    }
    float const mean = found < 1 ? 0.0f : sum / float(found);
    float const s = 1e-6f <= mean * 0.5f ? mean * 0.5f : 1e-6f;
    return s * s;
}

}  // namespace

// --- public helpers -----------------------------------------------------------

float PolylineLength(V3 const *points, size_t count)
{
    float l = 0.0f;
    for (size_t k = 1; k < count; ++k) l = l + Len(Sub(points[k], points[k - 1]));
    return l;
}

V3 SampleRaw(V3 const *p, size_t count, float s)
{
    if (count == 0) return V3();
    if (count == 1) return p[0];
    if (s <= 0.0f) s = 0.0f;
    for (size_t k = 0; k + 1 < count; ++k) {
        float const l = Len(Sub(p[k + 1], p[k]));
        if (s <= l) {
            float const u = l <= 1e-6f ? 0.0f : s / l;
            float const w = 1.0f - u;
            return Make(p[k + 1].x * u + p[k].x * w, p[k + 1].y * u + p[k].y * w,
                        p[k + 1].z * u + p[k].z * w);
        }
        s = s - l;
    }
    V3 const last = p[count - 1];
    V3 const d = Sub(last, p[count - 2]);
    float const l = Len(d);
    V3 const dir = l <= 1e-6f ? Make(0.0f, 1.0f, 0.0f) : Div(d, l);
    return Make(last.x + s * dir.x, last.y + s * dir.y, last.z + s * dir.z);
}

std::vector<int> Delaunay2D(std::vector<float> const &xy)
{
    size_t const n = xy.size() / 2;
    std::vector<int> out;
    if (n < 3) return out;
    struct P2 { double x, y; };
    std::vector<P2> pts(n + 3);
    double minX = xy[0], maxX = xy[0], minY = xy[1], maxY = xy[1];
    for (size_t i = 0; i < n; ++i) {
        pts[i] = {double(xy[2 * i]), double(xy[2 * i + 1])};
        minX = std::min(minX, pts[i].x);
        maxX = std::max(maxX, pts[i].x);
        minY = std::min(minY, pts[i].y);
        maxY = std::max(maxY, pts[i].y);
    }
    double const span = std::max(std::max(maxX - minX, maxY - minY), 1e-12);
    double const cx = 0.5 * (minX + maxX), cy = 0.5 * (minY + maxY);
    pts[n] = {cx - 40.0 * span, cy - 30.0 * span};
    pts[n + 1] = {cx + 40.0 * span, cy - 30.0 * span};
    pts[n + 2] = {cx, cy + 40.0 * span};
    struct Tri { int a, b, c; double ux, uy, r2; };
    auto make = [&](int a, int b, int c, Tri *t) {
        P2 const &A = pts[size_t(a)], &B = pts[size_t(b)], &C = pts[size_t(c)];
        double const d = 2.0 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
        if (std::fabs(d) < 1e-300) return false;
        double const a2 = A.x * A.x + A.y * A.y, b2 = B.x * B.x + B.y * B.y,
                     c2 = C.x * C.x + C.y * C.y;
        t->a = a; t->b = b; t->c = c;
        t->ux = (a2 * (B.y - C.y) + b2 * (C.y - A.y) + c2 * (A.y - B.y)) / d;
        t->uy = (a2 * (C.x - B.x) + b2 * (A.x - C.x) + c2 * (B.x - A.x)) / d;
        t->r2 = (A.x - t->ux) * (A.x - t->ux) + (A.y - t->uy) * (A.y - t->uy);
        return true;
    };
    std::vector<Tri> tris;
    Tri super;
    make(int(n), int(n + 1), int(n + 2), &super);
    tris.push_back(super);
    for (size_t i = 0; i < n; ++i) {
        P2 const &p = pts[i];
        bool duplicate = false;
        for (size_t j = 0; j < i && !duplicate; ++j)
            duplicate = pts[j].x == p.x && pts[j].y == p.y;
        if (duplicate) continue;
        std::vector<std::pair<int, int>> edges;
        std::vector<Tri> keep;
        keep.reserve(tris.size());
        for (Tri const &t : tris) {
            double const dx = p.x - t.ux, dy = p.y - t.uy;
            if (dx * dx + dy * dy < t.r2 * (1.0 - 1e-12)) {
                edges.push_back({t.a, t.b});
                edges.push_back({t.b, t.c});
                edges.push_back({t.c, t.a});
            } else {
                keep.push_back(t);
            }
        }
        for (size_t e = 0; e < edges.size(); ++e) {
            bool shared = false;
            for (size_t f = 0; f < edges.size() && !shared; ++f)
                shared = f != e && edges[e].first == edges[f].second &&
                         edges[e].second == edges[f].first;
            if (shared) continue;
            Tri t;
            if (make(edges[e].first, edges[e].second, int(i), &t)) keep.push_back(t);
        }
        tris.swap(keep);
    }
    for (Tri const &t : tris) {
        if (t.a >= int(n) || t.b >= int(n) || t.c >= int(n)) continue;
        out.push_back(t.a);
        out.push_back(t.b);
        out.push_back(t.c);
    }
    return out;
}

void Parting::Set(std::vector<V3> pts, std::vector<int> const &counts, float amount_, float angle_,
                  float radius_, float distance_)
{
    points = std::move(pts);
    offsets.assign(1, 0u);
    if (counts.empty()) {
        if (!points.empty()) offsets.push_back(uint32_t(points.size()));
    } else {
        for (int c : counts) offsets.push_back(offsets.back() + uint32_t(std::max(c, 0)));
    }
    if (offsets.size() == 1) offsets.clear();
    lengths.assign(CurveCount(), 0.0f);
    for (size_t c = 0; c < CurveCount(); ++c) {
        size_t const first = offsets[c], count = offsets[c + 1] - first;
        float l = 0.0f;
        for (size_t k = 1; k < count; ++k)
            l = Len(Sub(points[first + k], points[first + k - 1])) + l;
        lengths[c] = l;
    }
    active = 1e-4f < amount_ && 1e-4f < angle_ && CurveCount() > 0;
    amount = ClampRef(amount_, 0.0f, 1.0f);
    radius = ClampRef(radius_, 0.0f, 1.0f);
    tanAngle = std::tan(ClampRef(angle_, 0.0f, 89.0f) * 0.017453292f);
    distance = ClampRef(distance_, 0.0f, 1.0f);
}

PartInfo QueryParting(Parting const &parting, V3 const &x, V3 const &normal)
{
    PartInfo out;
    if (!parting.active) return out;
    float best = kFltMax;
    V3 q, dir = Make(0.0f, 1.0f, 0.0f);
    float curveLength = 0.0f;
    int curve = -1;
    for (size_t c = 0; c < parting.CurveCount(); ++c) {
        size_t const first = parting.offsets[c], count = parting.offsets[c + 1] - first;
        if (count < 2) continue;
        for (size_t k = 1; k < count; ++k) {
            V3 const &p0 = parting.points[first + k - 1];
            V3 const &p1 = parting.points[first + k];
            V3 const d = Sub(p1, p0);
            float const l2 = Dot(d, d);
            V3 c0;
            if (1e-8f <= l2) {
                float const u = ClampRef(Dot(d, Sub(x, p0)) / l2, 0.0f, 1.0f);
                c0 = Make(p0.x + d.x * u, p0.y + d.y * u, p0.z + d.z * u);
            } else {
                c0 = p0;
            }
            V3 const e = Sub(x, c0);
            float const dd = Dot(e, e);
            if (dd < best) {
                float const l = Len(d);
                dir = l <= 1e-6f ? Make(0.0f, 1.0f, 0.0f) : Div(d, l);
                q = c0;
                curveLength = c < parting.lengths.size() ? parting.lengths[c] : 0.0f;
                curve = int(c);
                best = dd;
            }
        }
    }
    if (best == kFltMax) return out;
    float const ln = Len(normal);
    V3 const n = ln <= 1e-6f ? Make(0.0f, 1.0f, 0.0f) : Div(normal, ln);
    V3 away = Cross(dir, n);
    float la = Len(away);
    if (la < 1e-6f) {
        float const ry = dir.y * 0.0f, rx = dir.x * 0.0f;
        away = Make(ry - dir.z, dir.z * 0.0f - rx, dir.x - ry);
        la = Len(away);
        if (la < 1e-6f) {
            away = Make(ry - dir.z * 0.0f, dir.z - rx, rx - dir.y);
            la = Len(away);
            if (la < 1e-6f) return out;
        }
    }
    away = Div(away, la);
    float const side = Dot(Sub(x, q), away);
    float offLine = side;
    if (!(0.0f <= side)) {
        away = Make(0.0f - away.x, 0.0f - away.y, 0.0f - away.z);
        offLine = -side;
    }
    out.valid = true;
    out.closest = q;
    out.dir = dir;
    out.away = away;
    out.dist2 = best;
    out.curveLength = curveLength;
    out.offLine = offLine <= 1e-5f ? 0.0f : 1.0f;
    out.curve = curve;
    out.side = side <= 1e-5f ? (-1e-5f <= side ? 0.0f : -1.0f) : 1.0f;
    float const R = parting.radius;
    if (1e-6f < R && 1e-6f < curveLength) {
        float const s = SmoothRef(1.0f - std::sqrt(best) / (R * curveLength));
        out.weight = s * parting.amount;
    }
    return out;
}

int PartKey(PartInfo const &info)
{
    if (info.valid && -1 < info.curve && info.side != 0.0f)
        return int(info.side <= 0.0f) + info.curve * 2;
    return -1;
}

void BuildGuides(float const *px, float const *py, float const *pz,
                 std::vector<uint32_t> const &spans, std::vector<V3> const &normals, int cvCount,
                 Weighting weighting, bool bendFollow, Parting const &parting, Guides *out)
{
    Guides &G = *out;
    G = Guides();
    G.rawOffset.push_back(0);
    size_t const curves = spans.empty() ? 0 : spans.size() - 1;
    std::vector<V3> pts;
    for (size_t c = 0; c < curves; ++c) {
        size_t const first = spans[c], count = spans[c + 1] - first;
        if (count < 2) continue;
        pts.resize(count);
        for (size_t i = 0; i < count; ++i) pts[i] = Make(px[first + i], py[first + i], pz[first + i]);
        float const L = PolylineLength(pts.data(), count);
        if (!(1e-6f < L)) continue;
        G.curve.push_back(uint32_t(c));
        G.raw.insert(G.raw.end(), pts.begin(), pts.end());
        G.rawOffset.push_back(uint32_t(G.raw.size()));
        G.root.push_back(pts[0]);
        G.rootNormal.push_back(c < normals.size() ? normals[c] : Make(0.0f, 1.0f, 0.0f));
        G.length.push_back(L);
    }
    size_t const count = G.root.size();
    int const ng = std::min(std::max(cvCount, 8), 1000);
    G.ng = ng;
    G.table.resize(count * size_t(ng));
    for (size_t g = 0; g < count; ++g) {
        V3 const *raw = &G.raw[G.rawOffset[g]];
        size_t const rawCount = G.rawOffset[g + 1] - G.rawOffset[g];
        for (int i = 0; i < ng; ++i)
            G.table[g * size_t(ng) + size_t(i)] =
                SampleRaw(raw, rawCount, StrandT(i, ng) * G.length[g]);
    }
    // Initial directions: first resampled segment, else the first raw
    // segment, else the root normal.
    G.initialDir.resize(count);
    for (size_t g = 0; g < count; ++g) {
        V3 d;
        if (1 < ng) d = Sub(G.table[g * size_t(ng) + 1], G.table[g * size_t(ng)]);
        float l = Len(d);
        if (l <= 1e-6f) {
            size_t const rawCount = G.rawOffset[g + 1] - G.rawOffset[g];
            if (1 < rawCount) {
                d = Sub(G.raw[G.rawOffset[g] + 1], G.raw[G.rawOffset[g]]);
                l = Len(d);
            }
        }
        G.initialDir[g] = l <= 1e-6f ? NormalizeX(G.rootNormal[g]) : GlmNormalize(d);
    }
    if (parting.active) {
        G.part.resize(count);
        for (size_t g = 0; g < count; ++g)
            G.part[g] = QueryParting(parting, G.root[g], G.rootNormal[g]);
    }
    if (bendFollow && count > 0) {
        G.frameT.resize(count * size_t(ng));
        G.frameN.resize(count * size_t(ng));
        G.frameB.resize(count * size_t(ng));
        std::vector<V3> tangents(static_cast<size_t>(ng));
        for (size_t g = 0; g < count; ++g) {
            V3 const *raw = &G.raw[G.rawOffset[g]];
            size_t const rawCount = G.rawOffset[g + 1] - G.rawOffset[g];
            for (int i = 0; i < ng; ++i)
                tangents[size_t(i)] = CatmullRomTangent(raw, rawCount, StrandT(i, ng) * G.length[g]);
            size_t const base = g * size_t(ng);
            TransportFrames(tangents.data(), size_t(ng), &G.frameT[base], &G.frameN[base],
                            &G.frameB[base]);
        }
        G.frames = true;
    }
    if (weighting == Weighting::Gaussian && count >= 3) {
        FitPlane(G.root, &G.origin, &G.axisU, &G.axisV);
        G.planar.resize(2 * count);
        for (size_t g = 0; g < count; ++g) {
            V3 const d = Sub(G.root[g], G.origin);
            G.planar[2 * g] = Dot(d, G.axisU);
            G.planar[2 * g + 1] = Dot(d, G.axisV);
        }
        G.plane = true;
        std::vector<int> const tris = Delaunay2D(G.planar);
        if (!tris.empty()) {
            int edges = 0;
            float sum = 0.0f;
            auto at = [&](int i, int c) { return G.planar[2 * size_t(i) + size_t(c)]; };
            for (size_t t = 0; t + 2 < tris.size(); t += 3) {
                int const a = tris[t], b = tris[t + 1], c = tris[t + 2];
                float const e0x = at(b, 0) - at(c, 0), e0y = at(b, 1) - at(c, 1);
                float const e1x = at(a, 0) - at(b, 0), e1y = at(a, 1) - at(b, 1);
                float const e2x = at(c, 0) - at(a, 0), e2y = at(c, 1) - at(a, 1);
                float const e0 = std::sqrt(e0y * e0y + e0x * e0x);
                float const e1 = std::sqrt(e1y * e1y + e1x * e1x);
                float const e2 = std::sqrt(e2y * e2y + e2x * e2x);
                edges += 3;
                sum = sum + e0 + e1 + e2;
            }
            float const mean = edges < 1 ? 0.0f : sum / float(edges);
            float const s = 1e-6f <= mean * 0.5f ? mean * 0.5f : 1e-6f;
            G.sigma2 = s * s;
        }
        if (!(1e-12f < G.sigma2)) G.sigma2 = NearestNeighbourSigma2(G.planar);
    }
}

int NearestGuide(Guides const &guides, Parting const &parting, PartInfo const &rootPart,
                 V3 const &root)
{
    Slot slots[3];
    KNearest(guides, parting, rootPart, root, slots);
    return slots[0].g;
}

void BuildStrand(Guides const &guides, Parting const &parting, StrandSettings const &settings,
                 V3 const &root, V3 const &normal, float *out, StrandResult *result,
                 Scratch *scratch)
{
    StrandResult &res = *result;
    res = StrandResult();
    int const n = settings.cvCount;
    size_t const count = guides.Count();
    PartInfo const rootPart = QueryParting(parting, root, normal);
    float const wr = rootPart.weight;
    res.partId = wr <= 0.2f ? -1 : PartKey(rootPart);

    Slot slots[3];
    if (count > 0) KNearest(guides, parting, rootPart, root, slots);
    res.nearest = slots[0].g;

    // A root on a guide root is that guide's own strand: a rigid copy of the
    // raw guide (the reference's pre-assigned guide).
    int assigned = -1;
    if (count > 0) {
        float best = kFltMax;
        for (size_t g = 0; g < count; ++g) {
            V3 const d = Sub(root, guides.root[g]);
            float const d2 = Dot(d, d);
            if (d2 < best) { best = d2; assigned = int(g); }
        }
        if (!(best <= 1e-12f)) assigned = -1;
    }

    enum class Path { Straight, Assigned, Single, Weighted, Kernel } path = Path::Straight;
    float L = settings.defaultLength;
    int primary = -1;
    float w3[3] = {0.0f, 0.0f, 0.0f};
    float kernelSum = 0.0f;
    std::vector<float> &w = scratch->weights;
    if (assigned >= 0) {
        path = Path::Assigned;
        primary = assigned;
        L = guides.length[size_t(assigned)];
    } else {
        bool kernel = false;
        if (settings.weighting == Weighting::Gaussian && count > 0) {
            w.assign(count, 0.0f);
            float sigma2 = guides.sigma2;
            if (sigma2 <= 1e-12f) sigma2 = 1e-6f;
            float const pw = ClampRef(settings.tightness, 0.25f, 8.0f);
            float qx = 0.0f, qy = 0.0f;
            if (guides.plane) {
                V3 const d = Sub(root, guides.origin);
                qx = Dot(d, guides.axisU);
                qy = Dot(d, guides.axisV);
            }
            auto d2Of = [&](size_t g) {
                if (guides.plane) {
                    float const dy = qy - guides.planar[2 * g + 1];
                    float const dx = qx - guides.planar[2 * g];
                    return dx * dx + dy * dy;
                }
                V3 const d = Sub(root, guides.root[g]);
                return Dot(d, d);
            };
            auto withFactor = [&]() {
                for (size_t g = 0; g < count; ++g) {
                    float const e = std::exp(-d2Of(g) / sigma2);
                    float const f = PartingFactor(parting, rootPart, guides, g);
                    w[g] = f * std::pow(e, pw);
                }
            };
            auto sum = [&]() {
                float s = 0.0f;
                for (float v : w) s = s + v;
                return s;
            };
            withFactor();
            float total = sum();
            if (!(1e-8f < total)) {
                // Drop the soft factor; guides across the parting keep their
                // (suppressed) weight.
                bool computed = false;
                for (size_t g = 0; g < count; ++g) {
                    bool const skip = parting.active && rootPart.valid && 0.2f < wr &&
                                      g < guides.part.size() && Across(rootPart, guides.part[g]);
                    if (skip) continue;
                    w[g] = std::pow(std::exp(-d2Of(g) / sigma2), pw);
                    computed = true;
                }
                if (!computed) withFactor();
                total = sum();
            }
            if (1e-8f < total) {
                kernel = true;
                kernelSum = total;
                float length = 0.0f, best = -1.0f;
                for (size_t g = 0; g < count; ++g) {
                    float const v = w[g];
                    if (!(v <= best)) { best = v; primary = int(g); }
                    length = length + (v / total) * guides.length[g];
                }
                L = length;
                path = Path::Kernel;
            }
        }
        if (!kernel) primary = slots[0].g;
        if (!kernel && count > 0) {
            if (settings.weighting == Weighting::Nearest || slots[1].g < 0) {
                path = Path::Single;
                if (slots[0].g >= 0) L = guides.length[size_t(slots[0].g)];
            } else {
                bool bary = false;
                if (settings.weighting == Weighting::Gaussian && slots[2].g >= 0) {
                    float b[3];
                    if (Barycentric(root, guides.root[size_t(slots[0].g)],
                                    guides.root[size_t(slots[1].g)],
                                    guides.root[size_t(slots[2].g)], b) &&
                        !(b[0] < 0.0f) && !(b[1] < 0.0f) && !(b[2] < 0.0f)) {
                        float const s = b[1] + b[0] + b[2];
                        if (1e-6f < s) {
                            w3[0] = b[0] / s;
                            w3[1] = b[1] / s;
                            w3[2] = b[2] / s;
                        } else {
                            w3[0] = 1.0f;
                            w3[1] = 0.0f;
                            w3[2] = 0.0f;
                        }
                        bary = true;
                    }
                }
                if (!bary) {
                    float const a0 = std::sqrt(slots[0].d2) + 1e-6f;
                    float const a1 = std::sqrt(slots[1].d2) + 1e-6f;
                    w3[2] = slots[2].g >= 0 ? 1.0f / (std::sqrt(slots[2].d2) + 1e-6f) : 0.0f;
                    w3[0] = 1.0f / a0;
                    w3[1] = 1.0f / a1;
                }
                float const s = w3[1] + w3[0] + w3[2];
                if (1e-6f < s) {
                    w3[2] = w3[2] / s;
                    w3[1] = w3[1] / s;
                    w3[0] = w3[0] / s;
                } else {
                    w3[0] = 1.0f;
                    w3[1] = 0.0f;
                    w3[2] = 0.0f;
                }
                float length = 0.0f;
                if (slots[0].g >= 0) length = w3[0] * guides.length[size_t(slots[0].g)] + 0.0f;
                length = length + guides.length[size_t(slots[1].g)] * w3[1];
                if (slots[2].g >= 0) length = length + w3[2] * guides.length[size_t(slots[2].g)];
                L = length;
                path = Path::Weighted;
            }
        }
    }

    if (L <= 1e-5f) {
        for (int j = 0; j < n; ++j) {
            out[3 * j] = root.x;
            out[3 * j + 1] = root.y;
            out[3 * j + 2] = root.z;
        }
        res.dropped = true;
        res.length = 0.0f;
        return;
    }
    res.primary = primary;

    switch (path) {
    case Path::Straight:
        for (int j = 0; j < n; ++j) {
            float const s = StrandT(j, n) * L;
            out[3 * j] = root.x + normal.x * s;
            out[3 * j + 1] = root.y + normal.y * s;
            out[3 * j + 2] = root.z + s * normal.z;
        }
        BendStrand(parting, rootPart, root, normal, n, out);
        res.length = L;
        break;
    case Path::Assigned: {
        size_t const g = size_t(assigned);
        V3 const delta = Sub(root, guides.root[g]);
        V3 const *raw = &guides.raw[guides.rawOffset[g]];
        size_t const rawCount = guides.rawOffset[g + 1] - guides.rawOffset[g];
        for (int j = 0; j < n; ++j) {
            V3 const p = SampleRaw(raw, rawCount, StrandT(j, n) * L);
            out[3 * j] = p.x + delta.x;
            out[3 * j + 1] = p.y + delta.y;
            out[3 * j + 2] = p.z + delta.z;
        }
        BendStrand(parting, rootPart, root, normal, n, out);
        res.index[0] = assigned;
        res.weight[0] = 1.0f;
        res.length = L;
        break;
    }
    case Path::Single: {
        size_t const g = size_t(slots[0].g);
        V3 const delta = Sub(root, guides.root[g]);
        for (int j = 0; j < n; ++j) {
            float const s = StrandT(j, n) * L;
            V3 const p = SampleTable(guides, g, s);
            V3 const off = FrameOffset(guides, g, delta, s);
            out[3 * j] = p.x + off.x;
            out[3 * j + 1] = p.y + off.y;
            out[3 * j + 2] = p.z + off.z;
        }
        BendStrand(parting, rootPart, root, normal, n, out);
        res.index[0] = slots[0].g;
        res.weight[0] = 1.0f;
        res.length = L;
        break;
    }
    case Path::Weighted:
        for (int j = 0; j < n; ++j) {
            float const t = StrandT(j, n);
            float const s = t * L;
            float ax = 0.0f, ay = 0.0f, az = 0.0f;
            for (int k = 0; k < 3; ++k) {
                if (slots[k].g < 0) continue;
                V3 const h = GuideContribution(guides, parting, size_t(slots[k].g), root, normal,
                                               rootPart, t, s);
                az = w3[k] * h.z + az;
                ay = w3[k] * h.y + ay;
                ax = h.x * w3[k] + ax;
            }
            out[3 * j] = ax;
            out[3 * j + 1] = ay;
            out[3 * j + 2] = az;
        }
        for (int k = 0; k < 3; ++k) {
            res.index[k] = slots[k].g;
            res.weight[k] = slots[k].g >= 0 ? w3[k] : 0.0f;
        }
        res.length = L;
        break;
    case Path::Kernel: {
        float const scale = L <= 1e-6f ? 0.0f : ClampRef(L / L, 0.0f, 1.0f);
        for (int j = 0; j < n; ++j) {
            float const t = StrandT(j, n);
            float ax = 0.0f, ay = 0.0f, az = 0.0f;
            for (size_t g = 0; g < count; ++g) {
                float const wn = w[g] / kernelSum;
                if (!(1e-8f < std::fabs(wn))) continue;
                float const tt = ClampRef(t * scale, 0.0f, 1.0f);
                V3 const h = GuideContribution(guides, parting, g, root, normal, rootPart, t,
                                               tt * guides.length[g]);
                az = wn * h.z + az;
                ay = wn * h.y + ay;
                ax = wn * h.x + ax;
            }
            out[3 * j] = ax;
            out[3 * j + 1] = ay;
            out[3 * j + 2] = az;
        }
        float length = 0.0f;
        for (int j = 1; j < n; ++j) {
            float const dx = out[3 * j] - out[3 * j - 3];
            float const dy = out[3 * j + 1] - out[3 * j - 2];
            float const dz = out[3 * j + 2] - out[3 * j - 1];
            length = length + std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        res.length = length;
        // The heaviest three, as blend weights.
        for (size_t g = 0; g < count; ++g) {
            float const v = w[g] / kernelSum;
            for (int k = 0; k < 3; ++k) {
                if (res.index[k] >= 0 && !(v > res.weight[k])) continue;
                for (int m = 2; m > k; --m) {
                    res.index[m] = res.index[m - 1];
                    res.weight[m] = res.weight[m - 1];
                }
                res.index[k] = int(g);
                res.weight[k] = v;
                break;
            }
        }
        break;
    }
    }
}

}  // namespace guideBlend
}  // namespace usdGen
