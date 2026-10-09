// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// deformEvaluateCpu.h — host mirror of deformEvaluate.comp's per-CV math.
//
// Every query runs the shader's operations in the shader's order (the same
// expression text, so the same FMA contraction applies): plain operators
// where the shader spells plain operators, std::fma/std::sqrt where it
// spells fma()/sqrt(). Queries are independent, so any range partition and
// any block alignment evaluates bitwise what the direct shader did; the
// non-finite flag is a single OR bit. Non-finite CVs and outputs are left
// unwritten, exactly like the shader's early returns.
//
// The W-wide blocking mirrors rbfField.cpp's DisplaceBlocked: W queries
// share one sample pass so the fp64 sqrt/FMA pipelines stay fed (a
// one-CV-at-a-time port exposes the full sqrt latency and runs ~15x
// slower). On AArch64 the fused hand-written NEON blocks below replace the
// auto-vectorized template (8-wide main, 4-wide remainder); elsewhere W=12
// is the measured sweet spot (eight leaves vector-sqrt throughput
// exposed, sixteen spills, thirty-two falls off a cliff). The tail runs
// the same scalar expression text as the block.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace usdGen::vulkan {

struct DeformEvalCpuParams {
    double const* samples = nullptr; // 3*n host-centered doubles (sample - center)
    double const* coef = nullptr;    // 3*m column-major doubles
    int n = 0;                       // samples
    int m = 0;                       // n + 4
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double invScale = 1.0;
    double scale = 1.0;
};

namespace deformCpuDetail {

// Exponent-bit finiteness: all-set exponent bits iff infinite or NaN (sNaN
// included), exactly `!(isinf || isnan)` on each lane with no FP work, so
// the per-CV checks run on the integer pipe (which the sample loop's FP
// mix leaves idle) instead of fabs/fcmp/fccmp plus short-circuit
// branches. The verdicts, the skip pattern, and the bad flag are
// unchanged, so every output bit matches.
inline bool FiniteF(float v) noexcept
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    return (u & 0x7F800000u) != 0x7F800000u;
}
inline bool FiniteD(double v) noexcept
{
    uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return (u & 0x7FF0000000000000ull) != 0x7FF0000000000000ull;
}

// One W-wide block over qs[0..W) -> ds[0..W): shader text, lane-arrayed so
// the vectorizer contracts across lanes. `bad` accumulates the non-finite
// flag. W must divide nothing in particular; the caller handles the tail.
template <size_t W>
inline void EvalBlock(DeformEvalCpuParams const& p, float const* qs, float* ds,
                      bool& bad) noexcept
{
    double qx[W], qy[W], qz[W];
    bool skip[W];
    for (size_t t = 0; t < W; ++t) {
        float const px = qs[t * 3], py = qs[t * 3 + 1], pz = qs[t * 3 + 2];
        bool const ok = FiniteF(px) && FiniteF(py) && FiniteF(pz);
        skip[t] = !ok;
        bad = bad || !ok;
        qx[t] = px;
        qy[t] = py;
        qz[t] = pz;
    }
    // Normalizer: (double(p) - c) * invScale, verbatim.
    double x[W], y[W], z[W];
    for (size_t t = 0; t < W; ++t) {
        x[t] = (qx[t] - p.cx) * p.invScale;
        y[t] = (qy[t] - p.cy) * p.invScale;
        z[t] = (qz[t] - p.cz) * p.invScale;
    }
    // Polynomial part, verbatim (last 4 coefficients of each column).
    double ox[W], oy[W], oz[W];
    double const* cx = p.coef;
    double const* cy = p.coef + size_t(p.m);
    double const* cz = p.coef + size_t(2) * size_t(p.m);
    size_t const n = size_t(p.n);
    for (size_t t = 0; t < W; ++t) {
        ox[t] = x[t] + cx[n] + cx[n + 1] * x[t] + cx[n + 2] * y[t] + cx[n + 3] * z[t];
        oy[t] = y[t] + cy[n] + cy[n + 1] * x[t] + cy[n + 2] * y[t] + cy[n + 3] * z[t];
        oz[t] = z[t] + cz[n] + cz[n + 1] * x[t] + cz[n + 2] * y[t] + cz[n + 3] * z[t];
    }
    for (size_t j = 0; j < n; ++j) {
        double const sx = p.samples[j * 3], sy = p.samples[j * 3 + 1], sz = p.samples[j * 3 + 2];
        double kk[W];
        for (size_t t = 0; t < W; ++t) {
            double const dx = std::fma(-sx, p.invScale, x[t]);
            double const dy = std::fma(-sy, p.invScale, y[t]);
            double const dz = std::fma(-sz, p.invScale, z[t]);
            double const rr = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)));
            kk[t] = rr * rr * rr;
        }
        for (size_t t = 0; t < W; ++t) {
            ox[t] += cx[j] * kk[t];
            oy[t] += cy[j] * kk[t];
            oz[t] += cz[j] * kk[t];
        }
    }
    for (size_t t = 0; t < W; ++t) {
        if (skip[t]) continue;
        double const oxw = ox[t] * p.scale + p.cx;
        double const oyw = oy[t] * p.scale + p.cy;
        double const ozw = oz[t] * p.scale + p.cz;
        if (!FiniteD(oxw) || !FiniteD(oyw) || !FiniteD(ozw)) {
            bad = true;
            continue;
        }
        ds[t * 3] = float(oxw);
        ds[t * 3 + 1] = float(oyw);
        ds[t * 3 + 2] = float(ozw);
    }
}

// Scalar tail query: the same expression text as EvalBlock's lanes.
inline void EvalOne(DeformEvalCpuParams const& p, float const* q, float* d,
                    bool& bad) noexcept
{
    float const px = q[0], py = q[1], pz = q[2];
    if (!FiniteF(px) || !FiniteF(py) || !FiniteF(pz)) {
        bad = true;
        return;
    }
    double const x = (double(px) - p.cx) * p.invScale;
    double const y = (double(py) - p.cy) * p.invScale;
    double const z = (double(pz) - p.cz) * p.invScale;
    double const* cx = p.coef;
    double const* cy = p.coef + size_t(p.m);
    double const* cz = p.coef + size_t(2) * size_t(p.m);
    size_t const n = size_t(p.n);
    double ox = x + cx[n] + cx[n + 1] * x + cx[n + 2] * y + cx[n + 3] * z;
    double oy = y + cy[n] + cy[n + 1] * x + cy[n + 2] * y + cy[n + 3] * z;
    double oz = z + cz[n] + cz[n + 1] * x + cz[n + 2] * y + cz[n + 3] * z;
    for (size_t j = 0; j < n; ++j) {
        double const dx = std::fma(-p.samples[j * 3], p.invScale, x);
        double const dy = std::fma(-p.samples[j * 3 + 1], p.invScale, y);
        double const dz = std::fma(-p.samples[j * 3 + 2], p.invScale, z);
        double r = std::sqrt(std::fma(dz, dz, std::fma(dx, dx, dy * dy)));
        r = r * r * r;
        ox += cx[j] * r;
        oy += cy[j] * r;
        oz += cz[j] * r;
    }
    double const oxw = ox * p.scale + p.cx;
    double const oyw = oy * p.scale + p.cy;
    double const ozw = oz * p.scale + p.cz;
    if (!FiniteD(oxw) || !FiniteD(oyw) || !FiniteD(ozw)) {
        bad = true;
        return;
    }
    d[0] = float(oxw);
    d[1] = float(oyw);
    d[2] = float(ozw);
}

} // namespace deformCpuDetail

#if defined(__aarch64__) || defined(_M_ARM64)
namespace deformCpuDetail {
// Fused 4-query NEON block: the same per-lane operations as EvalBlock in
// EvalBlock's order (the same FMA contraction association,
// correctly-rounded vector sqrt), but the kernel evaluation and the
// accumulation fuse into one sample pass with the normalised queries and
// accumulators held in vector registers. The auto-vectorized W12 template
// spills the queries to the stack and reloads them every sample; the fused
// loop keeps them live. The shader's fma(-s,inv,x) runs as vfmaq against
// a once-negated invScale (identical exact real value, so the one rounding
// gives the identical IEEE result; spelled directly, GCC expands vfmsq to
// fneg+fmla); the prologue seed is vadd-then-FMA exactly as the scalar
// text contracts. Queries load AoS via vld3; the outputs spill to the L1
// stack for the scalar skip/finiteness store pass, which is EvalBlock's
// epilogue verbatim. always_inline: the unrolled body is big enough that
// the outliner would otherwise charge a call per 4 CVs.
inline void __attribute__((always_inline))
EvalBlockNeon4(DeformEvalCpuParams const& p, float const* qs, float* ds,
               bool& bad) noexcept
{
    bool skip[4];
    for (size_t t = 0; t < 4; ++t) {
        float const px = qs[t * 3], py = qs[t * 3 + 1], pz = qs[t * 3 + 2];
        bool const ok = FiniteF(px) && FiniteF(py) && FiniteF(pz);
        skip[t] = !ok;
        bad = bad || !ok;
    }
    double const* cx = p.coef;
    double const* cy = p.coef + size_t(p.m);
    double const* cz = p.coef + size_t(2) * size_t(p.m);
    size_t const n = size_t(p.n);
    float64x2_t const c0 = vdupq_n_f64(p.cx);
    float64x2_t const c1 = vdupq_n_f64(p.cy);
    float64x2_t const c2 = vdupq_n_f64(p.cz);
    float64x2_t const invS = vdupq_n_f64(p.invScale);
    float64x2_t const nInvS = vnegq_f64(invS);
    float32x4x3_t const q = vld3q_f32(qs);
    float64x2_t const px0 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q.val[0])), c0), invS);
    float64x2_t const px1 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q.val[0])), c0), invS);
    float64x2_t const py0 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q.val[1])), c1), invS);
    float64x2_t const py1 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q.val[1])), c1), invS);
    float64x2_t const pz0 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q.val[2])), c2), invS);
    float64x2_t const pz1 =
        vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q.val[2])), c2), invS);
    float64x2_t ox0, ox1, oy0, oy1, oz0, oz1;
    {
        float64x2_t const a0 = vdupq_n_f64(cx[n]);
        float64x2_t const a1 = vdupq_n_f64(cx[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cx[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cx[n + 3]);
        ox0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px0, a0), a1, px0), a2, py0), a3, pz0);
        ox1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px1, a0), a1, px1), a2, py1), a3, pz1);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cy[n]);
        float64x2_t const a1 = vdupq_n_f64(cy[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cy[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cy[n + 3]);
        oy0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py0, a0), a1, px0), a2, py0), a3, pz0);
        oy1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py1, a0), a1, px1), a2, py1), a3, pz1);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cz[n]);
        float64x2_t const a1 = vdupq_n_f64(cz[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cz[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cz[n + 3]);
        oz0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz0, a0), a1, px0), a2, py0), a3, pz0);
        oz1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz1, a0), a1, px1), a2, py1), a3, pz1);
    }
#if defined(__clang__)
#pragma clang loop unroll_count(32)
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
    for (size_t j = 0; j < n; ++j) {
        float64x2_t const sx = vld1q_dup_f64(&p.samples[j * 3]);
        float64x2_t const sy = vld1q_dup_f64(&p.samples[j * 3 + 1]);
        float64x2_t const sz = vld1q_dup_f64(&p.samples[j * 3 + 2]);
        float64x2_t const qx = vld1q_dup_f64(&cx[j]);
        float64x2_t const qy = vld1q_dup_f64(&cy[j]);
        float64x2_t const qz = vld1q_dup_f64(&cz[j]);
        float64x2_t const dx0 = vfmaq_f64(px0, sx, nInvS);
        float64x2_t const dx1 = vfmaq_f64(px1, sx, nInvS);
        float64x2_t const dy0 = vfmaq_f64(py0, sy, nInvS);
        float64x2_t const dy1 = vfmaq_f64(py1, sy, nInvS);
        float64x2_t const dz0 = vfmaq_f64(pz0, sz, nInvS);
        float64x2_t const dz1 = vfmaq_f64(pz1, sz, nInvS);
        float64x2_t s0 = vfmaq_f64(vmulq_f64(dy0, dy0), dx0, dx0);
        s0 = vfmaq_f64(s0, dz0, dz0);
        float64x2_t s1 = vfmaq_f64(vmulq_f64(dy1, dy1), dx1, dx1);
        s1 = vfmaq_f64(s1, dz1, dz1);
        float64x2_t const rr0 = vsqrtq_f64(s0);
        float64x2_t const rr1 = vsqrtq_f64(s1);
        float64x2_t const kk0 = vmulq_f64(vmulq_f64(rr0, rr0), rr0);
        float64x2_t const kk1 = vmulq_f64(vmulq_f64(rr1, rr1), rr1);
        ox0 = vfmaq_f64(ox0, qx, kk0);
        oy0 = vfmaq_f64(oy0, qy, kk0);
        oz0 = vfmaq_f64(oz0, qz, kk0);
        ox1 = vfmaq_f64(ox1, qx, kk1);
        oy1 = vfmaq_f64(oy1, qy, kk1);
        oz1 = vfmaq_f64(oz1, qz, kk1);
    }
    float64x2_t const sc = vdupq_n_f64(p.scale);
    double oxw[4], oyw[4], ozw[4];
    vst1q_f64(oxw, vfmaq_f64(c0, ox0, sc));
    vst1q_f64(oxw + 2, vfmaq_f64(c0, ox1, sc));
    vst1q_f64(oyw, vfmaq_f64(c1, oy0, sc));
    vst1q_f64(oyw + 2, vfmaq_f64(c1, oy1, sc));
    vst1q_f64(ozw, vfmaq_f64(c2, oz0, sc));
    vst1q_f64(ozw + 2, vfmaq_f64(c2, oz1, sc));
    for (size_t t = 0; t < 4; ++t) {
        if (skip[t]) continue;
        double const ox = oxw[t], oy = oyw[t], oz = ozw[t];
        if (!FiniteD(ox) || !FiniteD(oy) || !FiniteD(oz)) {
            bad = true;
            continue;
        }
        ds[t * 3] = float(ox);
        ds[t * 3 + 1] = float(oy);
        ds[t * 3 + 2] = float(oz);
    }
}
// Eight-wide fused NEON block: two 4-query halves over one shared sample
// pass. Each lane runs the 4-wide block's operations in the 4-wide order,
// so the eight lanes are bitwise the 4-wide block run twice; only the
// sample/coefficient broadcasts and the prologue/epilogue amortize over
// eight queries instead of four. That amortization is what beats the
// auto-vectorized W12 (427 vs 480ps/pair single-threaded on this ARM64;
// the 4-wide fused block ties W12 at 493ps). Unroll x16: x8 under-feeds
// the pipes, x32 spills back over x16.
inline void __attribute__((always_inline))
EvalBlockNeon8(DeformEvalCpuParams const& p, float const* qs, float* ds,
               bool& bad) noexcept
{
    bool skip[8];
    for (size_t t = 0; t < 8; ++t) {
        float const px = qs[t * 3], py = qs[t * 3 + 1], pz = qs[t * 3 + 2];
        bool const ok = FiniteF(px) && FiniteF(py) && FiniteF(pz);
        skip[t] = !ok;
        bad = bad || !ok;
    }
    double const* cx = p.coef;
    double const* cy = p.coef + size_t(p.m);
    double const* cz = p.coef + size_t(2) * size_t(p.m);
    size_t const n = size_t(p.n);
    float64x2_t const c0 = vdupq_n_f64(p.cx);
    float64x2_t const c1 = vdupq_n_f64(p.cy);
    float64x2_t const c2 = vdupq_n_f64(p.cz);
    float64x2_t const invS = vdupq_n_f64(p.invScale);
    float64x2_t const nInvS = vnegq_f64(invS);
    float32x4x3_t const q0 = vld3q_f32(qs);
    float32x4x3_t const q1 = vld3q_f32(qs + 12);
    float64x2_t px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3;
    px0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q0.val[0])), c0), invS);
    px1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q0.val[0])), c0), invS);
    px2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q1.val[0])), c0), invS);
    px3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q1.val[0])), c0), invS);
    py0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q0.val[1])), c1), invS);
    py1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q0.val[1])), c1), invS);
    py2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q1.val[1])), c1), invS);
    py3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q1.val[1])), c1), invS);
    pz0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q0.val[2])), c2), invS);
    pz1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q0.val[2])), c2), invS);
    pz2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_low_f32(q1.val[2])), c2), invS);
    pz3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vget_high_f32(q1.val[2])), c2), invS);
    float64x2_t ox0, ox1, ox2, ox3, oy0, oy1, oy2, oy3, oz0, oz1, oz2, oz3;
    {
        float64x2_t const a0 = vdupq_n_f64(cx[n]);
        float64x2_t const a1 = vdupq_n_f64(cx[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cx[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cx[n + 3]);
        ox0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px0, a0), a1, px0), a2, py0), a3, pz0);
        ox1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px1, a0), a1, px1), a2, py1), a3, pz1);
        ox2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px2, a0), a1, px2), a2, py2), a3, pz2);
        ox3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(px3, a0), a1, px3), a2, py3), a3, pz3);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cy[n]);
        float64x2_t const a1 = vdupq_n_f64(cy[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cy[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cy[n + 3]);
        oy0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py0, a0), a1, px0), a2, py0), a3, pz0);
        oy1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py1, a0), a1, px1), a2, py1), a3, pz1);
        oy2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py2, a0), a1, px2), a2, py2), a3, pz2);
        oy3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(py3, a0), a1, px3), a2, py3), a3, pz3);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cz[n]);
        float64x2_t const a1 = vdupq_n_f64(cz[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cz[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cz[n + 3]);
        oz0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz0, a0), a1, px0), a2, py0), a3, pz0);
        oz1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz1, a0), a1, px1), a2, py1), a3, pz1);
        oz2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz2, a0), a1, px2), a2, py2), a3, pz2);
        oz3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(vaddq_f64(pz3, a0), a1, px3), a2, py3), a3, pz3);
    }
#if defined(__clang__)
#pragma clang loop unroll_count(16)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (size_t j = 0; j < n; ++j) {
        float64x2_t const sx = vld1q_dup_f64(&p.samples[j * 3]);
        float64x2_t const sy = vld1q_dup_f64(&p.samples[j * 3 + 1]);
        float64x2_t const sz = vld1q_dup_f64(&p.samples[j * 3 + 2]);
        float64x2_t const qx = vld1q_dup_f64(&cx[j]);
        float64x2_t const qy = vld1q_dup_f64(&cy[j]);
        float64x2_t const qz = vld1q_dup_f64(&cz[j]);
        float64x2_t const dx0 = vfmaq_f64(px0, sx, nInvS);
        float64x2_t const dx1 = vfmaq_f64(px1, sx, nInvS);
        float64x2_t const dx2 = vfmaq_f64(px2, sx, nInvS);
        float64x2_t const dx3 = vfmaq_f64(px3, sx, nInvS);
        float64x2_t const dy0 = vfmaq_f64(py0, sy, nInvS);
        float64x2_t const dy1 = vfmaq_f64(py1, sy, nInvS);
        float64x2_t const dy2 = vfmaq_f64(py2, sy, nInvS);
        float64x2_t const dy3 = vfmaq_f64(py3, sy, nInvS);
        float64x2_t const dz0 = vfmaq_f64(pz0, sz, nInvS);
        float64x2_t const dz1 = vfmaq_f64(pz1, sz, nInvS);
        float64x2_t const dz2 = vfmaq_f64(pz2, sz, nInvS);
        float64x2_t const dz3 = vfmaq_f64(pz3, sz, nInvS);
        float64x2_t s0 = vfmaq_f64(vmulq_f64(dy0, dy0), dx0, dx0);
        s0 = vfmaq_f64(s0, dz0, dz0);
        float64x2_t s1 = vfmaq_f64(vmulq_f64(dy1, dy1), dx1, dx1);
        s1 = vfmaq_f64(s1, dz1, dz1);
        float64x2_t s2 = vfmaq_f64(vmulq_f64(dy2, dy2), dx2, dx2);
        s2 = vfmaq_f64(s2, dz2, dz2);
        float64x2_t s3 = vfmaq_f64(vmulq_f64(dy3, dy3), dx3, dx3);
        s3 = vfmaq_f64(s3, dz3, dz3);
        float64x2_t const rr0 = vsqrtq_f64(s0);
        float64x2_t const rr1 = vsqrtq_f64(s1);
        float64x2_t const rr2 = vsqrtq_f64(s2);
        float64x2_t const rr3 = vsqrtq_f64(s3);
        float64x2_t const kk0 = vmulq_f64(vmulq_f64(rr0, rr0), rr0);
        float64x2_t const kk1 = vmulq_f64(vmulq_f64(rr1, rr1), rr1);
        float64x2_t const kk2 = vmulq_f64(vmulq_f64(rr2, rr2), rr2);
        float64x2_t const kk3 = vmulq_f64(vmulq_f64(rr3, rr3), rr3);
        ox0 = vfmaq_f64(ox0, qx, kk0);
        oy0 = vfmaq_f64(oy0, qy, kk0);
        oz0 = vfmaq_f64(oz0, qz, kk0);
        ox1 = vfmaq_f64(ox1, qx, kk1);
        oy1 = vfmaq_f64(oy1, qy, kk1);
        oz1 = vfmaq_f64(oz1, qz, kk1);
        ox2 = vfmaq_f64(ox2, qx, kk2);
        oy2 = vfmaq_f64(oy2, qy, kk2);
        oz2 = vfmaq_f64(oz2, qz, kk2);
        ox3 = vfmaq_f64(ox3, qx, kk3);
        oy3 = vfmaq_f64(oy3, qy, kk3);
        oz3 = vfmaq_f64(oz3, qz, kk3);
    }
    float64x2_t const sc = vdupq_n_f64(p.scale);
    double oxw[8], oyw[8], ozw[8];
    vst1q_f64(oxw, vfmaq_f64(c0, ox0, sc));
    vst1q_f64(oxw + 2, vfmaq_f64(c0, ox1, sc));
    vst1q_f64(oxw + 4, vfmaq_f64(c0, ox2, sc));
    vst1q_f64(oxw + 6, vfmaq_f64(c0, ox3, sc));
    vst1q_f64(oyw, vfmaq_f64(c1, oy0, sc));
    vst1q_f64(oyw + 2, vfmaq_f64(c1, oy1, sc));
    vst1q_f64(oyw + 4, vfmaq_f64(c1, oy2, sc));
    vst1q_f64(oyw + 6, vfmaq_f64(c1, oy3, sc));
    vst1q_f64(ozw, vfmaq_f64(c2, oz0, sc));
    vst1q_f64(ozw + 2, vfmaq_f64(c2, oz1, sc));
    vst1q_f64(ozw + 4, vfmaq_f64(c2, oz2, sc));
    vst1q_f64(ozw + 6, vfmaq_f64(c2, oz3, sc));
    for (size_t t = 0; t < 8; ++t) {
        if (skip[t]) continue;
        double const ox = oxw[t], oy = oyw[t], oz = ozw[t];
        if (!FiniteD(ox) || !FiniteD(oy) || !FiniteD(oz)) {
            bad = true;
            continue;
        }
        ds[t * 3] = float(ox);
        ds[t * 3 + 1] = float(oy);
        ds[t * 3 + 2] = float(oz);
    }
}
} // namespace deformCpuDetail
#endif

// Evaluates `count` CVs from qs (3 floats each) to ds (3 floats each).
// Bitwise deformEvaluate.comp over the same inputs. ORs 1 into *flag when
// any query or output is non-finite (leaving that output unwritten).
// Single-threaded; callers partition ranges across threads freely.
inline bool DeformEvaluateCpu(DeformEvalCpuParams const& p, float const* qs, float* ds,
                              size_t count, uint32_t* flag) noexcept
{
    if (count == 0) return true;
    if (!qs || !ds || !p.samples || !p.coef || !flag || p.n < 1 || p.m != p.n + 4)
        return false;
#if defined(__aarch64__) || defined(_M_ARM64)
    bool bad = false;
    size_t t = 0;
    size_t const blocks8 = count / 8;
    for (size_t b = 0; b < blocks8; ++b, t += 8)
        deformCpuDetail::EvalBlockNeon8(p, qs + t * 3, ds + t * 3, bad);
    if (count - t >= 4) {
        deformCpuDetail::EvalBlockNeon4(p, qs + t * 3, ds + t * 3, bad);
        t += 4;
    }
    for (; t < count; ++t) deformCpuDetail::EvalOne(p, qs + t * 3, ds + t * 3, bad);
    if (bad) *flag |= 1u;
    return true;
#else
    constexpr size_t W = 12;
    bool bad = false;
    size_t t = 0, blocks = count / W;
    for (size_t b = 0; b < blocks; ++b, t += W)
        deformCpuDetail::EvalBlock<W>(p, qs + t * 3, ds + t * 3, bad);
    for (; t < count; ++t) deformCpuDetail::EvalOne(p, qs + t * 3, ds + t * 3, bad);
    if (bad) *flag |= 1u;
    return true;
#endif
}

} // namespace usdGen::vulkan
