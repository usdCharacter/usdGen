// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenFrizzOp implementation. See frizz.h.
//
// Reference behaviour per strand (points P[0..n-1], n >= 3):
//   seed  = clumpId_0 when useClumpIds and the strand is clumped, else
//           strandSeed; phase phi, frame rotation rot (0 when clump-shared)
//           and noise offsets o1/o2 are hashes of seed * fixed odd constants.
//   refL  = restLength (else the current length); with useClumpIds the mean
//           restLength of the clump keyed by `seed` (the reference also looks
//           unclumped strands up by their strandSeed, reproduced here).
//   u     = arc_k / refL (unclamped), w = clamp01(Bezier(clamp01(u))) *
//           braidKnotMask[k] (when preserveKnots).
//   x     = sin(u * wavesX * 2pi + phi) - sin(phi)
//         + noise * (2 * vnoise(u * refL / noiseSize + o1) - its value at 0)
//   z     = sin(u * wavesZ * 2pi + phi + pi/2) - sin(phi + pi/2)
//         + noise * (2 * vnoise(u * refL / noiseSize + o2) - its value at 0)
//   (x, z) rotated by rot; D_k = w * amplitude * refL * (x * N_k + z * B_k)
//   with (N, B) a frame transported along the points that pass the weight
//   test (seeded from T x (0,1,0), or T x (1,0,0) when |T.y| >= 0.95).
//   Offsets come from the undisplaced curve; the root never moves.
// Floating-point evaluation order follows the reference.
#include "usdGen/ops/frizz.h"

#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

TfToken const kAmplitude("frizz:amplitude");
TfToken const kWavesX("frizz:wavesX");
TfToken const kWavesZ("frizz:wavesZ");
TfToken const kNoise("frizz:noise");
TfToken const kNoiseSize("frizz:noiseSize");
TfToken const kSharpness("frizz:sharpness");
TfToken const kRoot("frizz:root");
TfToken const kMid("frizz:mid");
TfToken const kMidShift("frizz:midShift");
TfToken const kTip("frizz:tip");
TfToken const kUseClumpIds("frizz:useClumpIds");
TfToken const kPreserveKnots("frizz:preserveKnots");

TfToken const kClumpPlane("clumpId_0");
TfToken const kRestLengthPlane("restLength");
TfToken const kKnotMaskPlane("braidKnotMask");

constexpr float kTwoPi = 6.2831855f;
constexpr float kHalfPi = 1.5707964f;
constexpr uint64_t kNoiseKeyX = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kNoiseKeyZ = 0x517CC1B727220A95ull;

inline float Clamp(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct FrizzSettings
{
    float amplitude = 0.0f;   // fraction of the reference length
    float wavesX = 0.0f, wavesZ = 0.0f;
    float noise = 0.0f;
    float noiseSize = 0.1f;   // scene units per noise cell
    float sharpness = 0.0f;
    float root = 0.0f, mid = 0.5f, midShift = 0.5f, tip = 1.0f;
    bool useClumpIds = false;
    bool preserveKnots = false;
};

FrizzSettings ReadSettings(UsdGenParamView const *p)
{
    FrizzSettings s;
    if (!p) return s;
    s.amplitude = std::max(float(p->GetDouble(kAmplitude, 0.0)), 0.0f);
    s.wavesX = std::max(float(p->GetDouble(kWavesX, 0.0)), 0.0f);
    s.wavesZ = std::max(float(p->GetDouble(kWavesZ, 0.0)), 0.0f);
    s.noise = Clamp(float(p->GetDouble(kNoise, 0.0)), 0.0f, 1.0f);
    s.noiseSize = std::max(float(p->GetDouble(kNoiseSize, 0.1)), 1e-4f);
    s.sharpness = Clamp(float(p->GetDouble(kSharpness, 0.0)), 0.0f, 1.0f);
    s.root = Clamp(float(p->GetDouble(kRoot, 0.0)), 0.0f, 1.0f);
    s.mid = Clamp(float(p->GetDouble(kMid, 0.5)), 0.0f, 1.0f);
    s.midShift = Clamp(float(p->GetDouble(kMidShift, 0.5)), 0.0f, 1.0f);
    s.tip = Clamp(float(p->GetDouble(kTip, 1.0)), 0.0f, 1.0f);
    s.useClumpIds = p->GetBool(kUseClumpIds, false);
    s.preserveKnots = p->GetBool(kPreserveKnots, false);
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

/// Frizz of one strand in place. `shared` (clump-shared seeds) drops the
/// per-strand rotation of the (x, z) offset. `mask` is n floats or null.
void FrizzStrand(float *P, size_t n, int32_t seed, FrizzSettings const &s, bool shared,
                 float refL, float const *mask)
{
    if (n <= 2) return;
    float const invNS = 1.0f / s.noiseSize;
    uint64_t const key = uint64_t(int64_t(seed));
    float const phi = strand::Hash01(key * 0x8DA6B343ull) * kTwoPi;
    float const rot = shared ? 0.0f : strand::Hash01(key * 0xD8163841ull) * kTwoPi;
    float const cr = std::cos(rot), sr = std::sin(rot);
    float const o1 = strand::Hash01(key * 0x1B873593ull) * 1000.0f;
    float const o2 = strand::Hash01(key * 0x9E3779B9ull) * 1000.0f;

    std::vector<float> arc;
    float const L = strand::ArcLengths(P, n, &arc);
    if (!(L > 1e-6f)) return;
    float const R = refL > 1e-6f ? refL : L;
    float const sx0 = s.wavesX > 1e-6f ? std::sin(phi) : 0.0f;
    float const sz0 = s.wavesZ > 1e-6f ? std::sin(phi + kHalfPi) : 0.0f;
    bool const noisy = s.noise > 1e-4f;
    float nx0 = 0.0f, nz0 = 0.0f;
    if (noisy) {
        nx0 = strand::ValueNoise1(o1 + 0.0f, kNoiseKeyX, s.sharpness);
        nx0 = nx0 + nx0;
        nz0 = strand::ValueNoise1(o2 + 0.0f, kNoiseKeyZ, s.sharpness);
        nz0 = nz0 + nz0;
    }
    if (refL <= 1e-6f) refL = L;
    float const amp = s.amplitude * refL;   // (amount/100 * 0.25) * refL

    std::vector<float> D(3 * n, 0.0f);
    bool framed = false;
    float Np[3] = {1.0f, 0.0f, 0.0f};
    for (size_t k = 1; k < n; ++k) {
        float const u = arc[k] / R;
        float const t = strand::Clamp01(u);
        float w = strand::Clamp01(
            strand::BezierProfile(t, s.root, s.mid, s.midShift, s.tip, 0.0f, 1.0f));
        if (mask) w = w * mask[k];
        if (!(w > 1e-4f)) continue;

        size_t const a = k < n - 1 ? k : k - 1;
        float tx = P[3 * (a + 1)] - P[3 * a];
        float ty = P[3 * (a + 1) + 1] - P[3 * a + 1];
        float tz = P[3 * (a + 1) + 2] - P[3 * a + 2];
        float const tl = std::sqrt(ty * ty + tx * tx + tz * tz);
        if (!(tl > 1e-6f)) continue;
        tx = tx / tl; ty = ty / tl; tz = tz / tl;

        float nx = 0.0f, ny = 0.0f, nz = 0.0f, nl = 0.0f;
        bool fresh = !framed;
        if (framed) {
            float const d = tx * Np[0] + ty * Np[1] + tz * Np[2];
            nx = Np[0] - d * tx;
            ny = Np[1] - d * ty;
            nz = Np[2] - d * tz;
            nl = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl <= 1e-6f) {
                framed = false;
                fresh = true;
            }
        }
        if (fresh) {
            bool const steep = std::fabs(ty) >= 0.95f;
            float const rx = steep ? 1.0f : 0.0f, ry = steep ? 0.0f : 1.0f, rz = 0.0f;
            nz = tx * ry - ty * rx;
            ny = tz * rx - tx * rz;
            nx = ty * rz - tz * ry;
            nl = std::sqrt(nx * nx + ny * ny + nz * nz);
            if (nl <= 1e-6f) continue;
            framed = true;
        }
        nx = nx / nl; ny = ny / nl; nz = nz / nl;
        float const bz = tx * ny - ty * nx;
        float const by = tz * nx - tx * nz;
        float const bx = ty * nz - tz * ny;
        float const invB = 1.0f / std::sqrt(bx * bx + by * by + bz * bz);
        Np[0] = nx; Np[1] = ny; Np[2] = nz;

        float wx = 0.0f, wz = 0.0f;
        if (s.wavesX > 1e-6f) wx = std::sin(u * s.wavesX * kTwoPi + phi) - sx0;
        if (s.wavesZ > 1e-6f) wz = std::sin(u * s.wavesZ * kTwoPi + phi + kHalfPi) - sz0;
        float gx = 0.0f, gz = 0.0f;
        if (noisy) {
            float const x = u * refL * invNS;
            float const vx = strand::ValueNoise1(x + o1, kNoiseKeyX, s.sharpness);
            gx = (vx + vx) - nx0;
            float const vz = strand::ValueNoise1(x + o2, kNoiseKeyZ, s.sharpness);
            gz = (vz + vz) - nz0;
        }
        wx = gx * s.noise + wx;
        wz = gz * s.noise + wz;
        float ox = wx, oz = wz;
        if (!shared) {
            ox = wx * cr - wz * sr;
            oz = wx * sr + wz * cr;
        }
        float const scale = w * amp;
        D[3 * k] = scale * (invB * bx * oz + ox * nx);
        D[3 * k + 1] = scale * (by * invB * oz + ox * ny);
        D[3 * k + 2] = scale * (bz * invB * oz + ox * nz);
    }
    for (size_t k = 1; k < n; ++k) {
        P[3 * k] = D[3 * k] + P[3 * k];
        P[3 * k + 1] = D[3 * k + 1] + P[3 * k + 1];
        P[3 * k + 2] = D[3 * k + 2] + P[3 * k + 2];
    }
}

}  // namespace

TfTokenVector UsdGenFrizzOp::StructuralParameters() const
{
    return {kAmplitude, kWavesX, kWavesZ, kNoise, kNoiseSize, kSharpness,
            kRoot, kMid, kMidShift, kTip, kUseClumpIds, kPreserveKnots};
}

std::vector<UsdGenStrandPlaneSpec> UsdGenFrizzOp::OutputPlaneSpecs() const
{
    return {};
}

bool UsdGenFrizzOp::Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                            std::vector<UsdGenStrandPlaneData> &, UsdGenDiagnostics *)
{
    FrizzSettings const s = ReadSettings(ctx.params);
    // Reference gates: amount/100 > 1e-4 (amplitude = amount/100 * 0.25),
    // and at least one of the waves or the noise is active.
    if (!(s.amplitude * 4.0f > 1e-4f)) return true;
    if (s.wavesX <= 1e-6f && s.wavesZ <= 1e-6f && s.noise <= 1e-4f) return true;

    uint32_t const R = strands.Count();
    int const *clump = strands.CurveInt(kClumpPlane);
    float const *restLength = strands.CurveFloat(kRestLengthPlane);
    float const *knotMask = s.preserveKnots ? strands.CvFloat(kKnotMaskPlane) : nullptr;

    // Reference length per strand: pre-length-scale length when known.
    std::vector<float> L0(R, 0.0f);
    ParallelFor(ctx.dispatcher, R, [&](size_t c) {
        if (restLength) {
            L0[c] = restLength[c];
        } else {
            std::vector<float> cum;
            L0[c] = strand::ArcLengths(strands.P(uint32_t(c)), strands.Size(uint32_t(c)), &cum);
        }
    });

    bool shared = false;
    int maxClump = -1;
    if (s.useClumpIds && clump)
        for (uint32_t c = 0; c < R; ++c) maxClump = std::max(maxClump, clump[c]);
    shared = maxClump >= 0;

    // Per-clump mean reference length, summed in strandIndex order.
    std::vector<float> clumpL0;
    if (shared) {
        std::vector<uint32_t> order(R);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return strands.Index(a) < strands.Index(b);
        });
        clumpL0.assign(size_t(maxClump) + 1, 0.0f);
        std::vector<int> counts(size_t(maxClump) + 1, 0);
        for (uint32_t c : order) {
            int const id = clump[c];
            if (id < 0) continue;
            clumpL0[size_t(id)] = L0[c] + clumpL0[size_t(id)];
            ++counts[size_t(id)];
        }
        for (size_t i = 0; i < clumpL0.size(); ++i)
            if (counts[i] > 0) clumpL0[i] = clumpL0[i] / float(counts[i]);
    }

    ParallelFor(ctx.dispatcher, R, [&](size_t ci) {
        uint32_t const c = uint32_t(ci);
        float refL = L0[c];
        int32_t seed;
        bool lookup = false;
        if (shared && clump[c] >= 0) {
            seed = clump[c];
            lookup = true;
        } else {
            seed = int32_t(strands.Seed(c));
            // Reference quirk: with shared clumps an unclumped strand looks
            // its own seed up in the per-clump table as well.
            lookup = shared && seed >= 0;
        }
        if (lookup && uint32_t(seed) < clumpL0.size() && clumpL0[size_t(seed)] > 1e-6f)
            refL = clumpL0[size_t(seed)];
        float const *mask = knotMask ? knotMask + strands.First(c) : nullptr;
        FrizzStrand(strands.P(c), strands.Size(c), seed, s, shared, refL, mask);
    });
    return true;
}

}  // namespace usdGen
