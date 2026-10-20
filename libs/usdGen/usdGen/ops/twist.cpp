// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTwistOp implementation. See twist.h.
//
// Reference behaviour, in order:
//   1. Group: clumpId_0 when useClumpIds and 0 <= id <= max id, else
//      (max id + 1) + primaryGuide, else none. Groups are further split by
//      CV count (the reference has one count per set).
//   2. Group mean curve C[k] = (1/count) * sum of P_s[k], summed in
//      strandIndex order. Fewer than two strands: no twist.
//   3. Per point k of C: central-difference unit axis, normalised arc length
//      t = s_k / L of C, angle = profile(t) * amount * (pi/180) * sign with
//      sign = -1 when the group's first strand has mirrorSign < 0.
//   4. Every point is rotated about (C[k], axis[k]) by its ABSOLUTE angle
//      (Rodrigues). With noise > 0 each strand scales the angle by
//      1 + noise * nz(k), nz a value noise over 4 cells along the point index,
//      keyed by strandSeed ^ "twist" and zero at the root.
//   5. When the root angle is non-zero and the root moved, the root is
//      re-snapped to the closest point of the bound surface with a smooth
//      falloff along the strand (strand::ResnapRoot).
// Floating-point evaluation order follows the reference so results match it
// bit for bit wherever the inputs do.
#include "usdGen/ops/twist.h"

#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

TfToken const kAmount("twist:amount");
TfToken const kRoot("twist:root");
TfToken const kMid("twist:mid");
TfToken const kMidShift("twist:midShift");
TfToken const kTip("twist:tip");
TfToken const kNoise("twist:noise");
TfToken const kUseClumpIds("twist:useClumpIds");
TfToken const kResnapRoot("twist:resnapRoot");

TfToken const kClumpPlane("clumpId_0");
TfToken const kGuidePlane("primaryGuide");
TfToken const kMirrorPlane("mirrorSign");

constexpr float kDegToRad = 0.017453292f;
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kTwistKey = 0x7477697374ull;  // ASCII "twist"

struct TwistSettings
{
    float amount = 1.0f;
    float root = 0.0f, mid = 0.0f, midShift = 0.5f, tip = 0.0f;
    float noise = 0.0f;
    bool useClumpIds = false;
    bool resnapRoot = true;
};

TwistSettings ReadSettings(UsdGenParamView const *p)
{
    TwistSettings s;
    if (!p) return s;
    s.amount = float(p->GetDouble(kAmount, 1.0));
    s.root = float(p->GetDouble(kRoot, 0.0));
    s.mid = float(p->GetDouble(kMid, 0.0));
    s.midShift = float(p->GetDouble(kMidShift, 0.5));
    s.tip = float(p->GetDouble(kTip, 0.0));
    s.noise = std::min(std::max(float(p->GetDouble(kNoise, 0.0)), 0.0f), 1.0f);
    s.useClumpIds = p->GetBool(kUseClumpIds, false);
    s.resnapRoot = p->GetBool(kResnapRoot, true);
    return s;
}

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (!dispatcher || count < 256) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

inline float Distance(float const *a, float const *b)
{
    float const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// One twist group: its mean curve and the per-point rotation tables.
struct TwistGroup
{
    uint32_t n = 0;              // CV count shared by every member
    uint32_t count = 0;          // members
    uint32_t representative = 0; // first member in strandIndex order
    bool valid = false;
    std::vector<float> center;   // 3n
    std::vector<float> axis;     // 3n, unit
    std::vector<float> sinA, cosA, angle;  // n
};

/// Angle/axis tables along the group's mean curve. False (no twist) when the
/// curve has no length.
bool BuildTables(TwistGroup &g, TwistSettings const &s, float sign)
{
    uint32_t const n = g.n;
    if (n < 2) return false;
    float const *C = g.center.data();
    float L = 0.0f;
    for (uint32_t k = 1; k < n; ++k) L = L + Distance(C + 3 * (k - 1), C + 3 * k);
    if (L <= 1e-6f) return false;

    // Initial axis: first point measurably away from the root.
    float T[3] = {0.0f, 0.0f, 0.0f};
    for (uint32_t j = 1; j < n; ++j) {
        float const len = Distance(C, C + 3 * j);
        if (len > 1e-6f) {
            T[0] = (C[3 * j] - C[0]) / len;
            T[1] = (C[3 * j + 1] - C[1]) / len;
            T[2] = (C[3 * j + 2] - C[2]) / len;
            break;
        }
    }
    g.axis.assign(3 * size_t(n), 0.0f);
    g.sinA.assign(n, 0.0f);
    g.cosA.assign(n, 1.0f);
    g.angle.assign(n, 0.0f);
    float arc = 0.0f;
    for (uint32_t k = 0; k < n; ++k) {
        if (k > 0) arc = arc + Distance(C + 3 * (k - 1), C + 3 * k);
        uint32_t const hi = std::min(k + 1, n - 1);
        uint32_t const lo = k > 0 ? k - 1 : 0;
        float const len = Distance(C + 3 * lo, C + 3 * hi);
        if (len > 1e-6f) {
            T[0] = (C[3 * hi] - C[3 * lo]) / len;
            T[1] = (C[3 * hi + 1] - C[3 * lo + 1]) / len;
            T[2] = (C[3 * hi + 2] - C[3 * lo + 2]) / len;
        }
        float const deg = strand::BezierProfile(arc / L, s.root, s.mid, s.midShift, s.tip);
        float const theta = deg * s.amount * kDegToRad * sign;
        g.axis[3 * k] = T[0];
        g.axis[3 * k + 1] = T[1];
        g.axis[3 * k + 2] = T[2];
        g.sinA[k] = std::sin(theta);
        g.cosA[k] = std::cos(theta);
        g.angle[k] = theta;
    }
    return true;
}

/// Rodrigues rotation of p about (c, t) with precomputed sin/cos, in the
/// reference's evaluation order for the table-driven pass.
inline void RotateTable(float *p, float const *c, float const *t, float sn, float cs)
{
    float const vx = p[0] - c[0], vy = p[1] - c[1], vz = p[2] - c[2];
    float const dot = (t[0] * vx + t[1] * vy + vz * t[2]) * (1.0f - cs);
    float const bx = vx * cs + c[0];
    float const by = vy * cs + c[1];
    float const bz = vz * cs + c[2];
    p[0] = (vz * t[1] - t[2] * vy) * sn + bx + t[0] * dot;
    p[1] = (t[2] * vx - vz * t[0]) * sn + by + t[1] * dot;
    p[2] = (vy * t[0] - t[1] * vx) * sn + bz + t[2] * dot;
}

/// Rodrigues rotation of p about (c, t) by `theta`, in the reference's
/// evaluation order for the noise pass.
inline void RotateNoise(float *p, float const *c, float const *t, float theta)
{
    float const sn = std::sin(theta), cs = std::cos(theta);
    float const vx = p[0] - c[0], vy = p[1] - c[1], vz = p[2] - c[2];
    float const dot = (vy * t[1] + t[0] * vx + vz * t[2]) * (1.0f - cs);
    p[0] = (t[1] * vz - vy * t[2]) * sn + vx * cs + c[0] + t[0] * dot;
    p[1] = (vx * t[2] - vz * t[0]) * sn + vy * cs + c[1] + t[1] * dot;
    p[2] = (vy * t[0] - t[1] * vx) * sn + vz * cs + c[2] + t[2] * dot;
}

/// Noise pass for one strand: angle * (1 + noise * nz), nz a 4-cell value
/// noise along the point index, offset to zero at the root.
void ApplyNoise(float *P, TwistGroup const &g, float noise, uint32_t seed)
{
    uint32_t const n = g.n;
    uint64_t const h = strand::SplitMix64(uint64_t(seed) ^ kTwistKey);
    float const v0 = strand::U01(strand::SplitMix64(h));
    float const step = n >= 2 ? 4.0f / float(int(n) - 1) : 0.0f;
    for (uint32_t k = 0; k < n; ++k) {
        float const x = float(int(k)) * step;
        float const cell = std::floor(x);
        float f = x - cell;
        uint64_t const z = uint64_t(int64_t(cell)) * kGolden + h;
        f = (3.0f - (f + f)) * f * f;
        float const a = strand::U01(strand::SplitMix64(z));
        float const b = strand::U01(strand::SplitMix64(z + kGolden));
        float const nz = (((a + a) - 1.0f) * (1.0f - f) + ((b + b) - 1.0f) * f) - ((v0 + v0) - 1.0f);
        float const theta = (nz * noise + 1.0f) * g.angle[k];
        if (theta != 0.0f)
            RotateNoise(P + 3 * size_t(k), g.center.data() + 3 * size_t(k),
                        g.axis.data() + 3 * size_t(k), theta);
    }
}

}  // namespace

TfTokenVector UsdGenTwistOp::StructuralParameters() const
{
    return {kAmount, kRoot, kMid, kMidShift, kTip, kNoise, kUseClumpIds, kResnapRoot};
}

std::vector<UsdGenStrandPlaneSpec> UsdGenTwistOp::OutputPlaneSpecs() const
{
    return {};
}

bool UsdGenTwistOp::Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                            std::vector<UsdGenStrandPlaneData> &, UsdGenDiagnostics *diag)
{
    TwistSettings const s = ReadSettings(ctx.params);
    if (!(s.amount > 0.0f) || (s.root == 0.0f && s.mid == 0.0f && s.tip == 0.0f)) return true;

    uint32_t const R = strands.Count();
    int const *clump = strands.CurveInt(kClumpPlane);
    int const *guide = strands.CurveInt(kGuidePlane);
    int const *mirror = strands.CurveInt(kMirrorPlane);
    if (!guide && !(s.useClumpIds && clump)) {
        if (diag)
            diag->Warn(Label() + ": upstream has no primaryGuide plane" +
                       (s.useClumpIds ? std::string(" or clumpId_0 plane") : std::string()) +
                       "; no strand has a twist group");
        return true;
    }

    // Generation order: sums and representatives follow strandIndex.
    std::vector<uint32_t> order(R);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return strands.Index(a) < strands.Index(b);
    });

    int64_t maxClump = -1;
    if (s.useClumpIds && clump)
        for (uint32_t c = 0; c < R; ++c) maxClump = std::max<int64_t>(maxClump, clump[c]);
    int64_t const base = maxClump + 1;
    auto groupKey = [&](uint32_t c) -> int64_t {
        if (clump) {
            int64_t const id = clump[c];
            if (id >= 0 && id <= maxClump) return id;
        }
        if (guide && guide[c] >= 0) return base + int64_t(guide[c]);
        return -1;
    };

    // Group membership and mean-curve sums.
    std::map<std::pair<int64_t, uint32_t>, uint32_t> slots;
    std::vector<TwistGroup> groups;
    std::vector<int32_t> slotOf(R, -1);
    for (uint32_t c : order) {
        uint32_t const n = uint32_t(strands.Size(c));
        if (n < 2) continue;
        int64_t const key = groupKey(c);
        if (key < 0) continue;
        auto const inserted = slots.emplace(std::make_pair(key, n), uint32_t(groups.size()));
        if (inserted.second) {
            TwistGroup g;
            g.n = n;
            g.representative = c;
            g.center.assign(3 * size_t(n), 0.0f);
            groups.push_back(std::move(g));
        }
        uint32_t const slot = inserted.first->second;
        slotOf[c] = int32_t(slot);
        TwistGroup &g = groups[slot];
        ++g.count;
        float const *P = strands.P(c);
        for (size_t i = 0; i < 3 * size_t(n); ++i) g.center[i] = P[i] + g.center[i];
    }

    ParallelFor(ctx.dispatcher, groups.size(), [&](size_t gi) {
        TwistGroup &g = groups[gi];
        if (g.count < 2) return;
        float const inv = 1.0f / float(int(g.count));
        for (float &v : g.center) v = inv * v;
        float const sign = (mirror && mirror[g.representative] < 0) ? -1.0f : 1.0f;
        g.valid = BuildTables(g, s, sign);
    });

    // Root re-snap onto the bound surface (current points).
    strand::TriangleBvh bvh;
    bool resnap = s.resnapRoot && s.root != 0.0f && ctx.desc &&
                  ctx.surface < ctx.desc->surfaces.size();
    if (resnap) {
        strand::BuildSurfaceBvh(ctx.desc->surfaces[ctx.surface], false, &bvh);
        resnap = !bvh.Empty();
    }

    ParallelFor(ctx.dispatcher, R, [&](size_t ci) {
        uint32_t const c = uint32_t(ci);
        if (slotOf[c] < 0) return;
        TwistGroup const &g = groups[size_t(slotOf[c])];
        if (!g.valid) return;
        float *P = strands.P(c);
        float const root0[3] = {P[0], P[1], P[2]};
        if (s.noise > 0.0f) {
            ApplyNoise(P, g, s.noise, strands.Seed(c));
        } else {
            for (uint32_t k = 0; k < g.n; ++k) {
                float const sn = g.sinA[k], cs = g.cosA[k];
                if (sn != 0.0f || cs != 1.0f)
                    RotateTable(P + 3 * size_t(k), g.center.data() + 3 * size_t(k),
                                g.axis.data() + 3 * size_t(k), sn, cs);
            }
        }
        if (!resnap) return;
        float const dx = P[0] - root0[0], dy = P[1] - root0[1], dz = P[2] - root0[2];
        if (!(dx * dx + dy * dy + dz * dz > 1e-14f)) return;
        GfVec3f const p0(P[0], P[1], P[2]);
        strand::ClosestHit const hit = bvh.Closest(p0);
        if (hit.hit) strand::ResnapRoot(P, g.n, hit.point - p0);
    });
    return true;
}

}  // namespace usdGen
