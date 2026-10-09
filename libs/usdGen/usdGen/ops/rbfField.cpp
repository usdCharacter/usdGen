#include "usdGen/ops/rbfField.h"
#include "usdGen/tbbFastCores.h"
#include "usdGen/workerPool.h"

#include "pxr/base/gf/vec3i.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_set>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace rbf {

namespace {

double Cube(double r) { return r * r * r; }

bool Finite(GfVec3d const &p)
{
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

// The rank test gpu::CudaRbfBinding applies to the polynomial Gram matrix:
// scaled elimination with a relative threshold, which also catches tilted
// coplanar layouts that never produce an exact zero.
bool FullAffineRank(std::vector<GfVec3d> const &y)
{
    double g[16] = {};
    for (GfVec3d const &p : y) {
        double const v[4] = {1.0, p[0], p[1], p[2]};
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) g[r * 4 + c] += v[r] * v[c];
    }
    double maxDiag = 0.0;
    for (int i = 0; i < 4; ++i) maxDiag = std::max(maxDiag, std::abs(g[i * 4 + i]));
    if (!(maxDiag > 0.0) || !std::isfinite(maxDiag)) return false;
    for (int c = 0; c < 4; ++c) {
        int pivot = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::abs(g[r * 4 + c]) > std::abs(g[pivot * 4 + c])) pivot = r;
        if (std::abs(g[pivot * 4 + c]) <= maxDiag * 1e-11) return false;
        if (pivot != c)
            for (int k = c; k < 4; ++k) std::swap(g[c * 4 + k], g[pivot * 4 + k]);
        for (int r = c + 1; r < 4; ++r) {
            double const q = g[r * 4 + c] / g[c * 4 + c];
            for (int k = c; k < 4; ++k) g[r * 4 + k] -= q * g[c * 4 + k];
        }
    }
    return true;
}

}  // namespace

bool CubicField::Bind(std::vector<GfVec3d> const &rest, std::string *error)
{
    auto fail = [&](char const *message) {
        _order = 0;
        _rest.clear();
        _restX.clear();
        _restY.clear();
        _restZ.clear();
        if (error) *error = message;
        return false;
    };
    size_t const n = rest.size();
    if (n < 4) return fail("an RBF needs at least four samples");
    GfVec3d lo(std::numeric_limits<double>::infinity());
    GfVec3d hi(-std::numeric_limits<double>::infinity());
    for (GfVec3d const &p : rest) {
        if (!Finite(p)) return fail("the RBF rest samples contain non-finite values");
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], p[d]);
            hi[d] = std::max(hi[d], p[d]);
        }
    }
    _centre = (lo + hi) * 0.5;
    _scale = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    if (!(_scale > 0.0) || !std::isfinite(_scale))
        return fail("the RBF rest samples have zero extent");
    _invScale = 1.0 / _scale;
    _rest.resize(n);
    for (size_t i = 0; i < n; ++i) _rest[i] = (rest[i] - _centre) / _scale;
    if (!FullAffineRank(_rest))
        return fail("the RBF samples do not span 3D (they are coplanar or collinear), "
                    "so the field's affine part is undetermined");

    size_t const m = n + 4;
    _lu.assign(m * m, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            _lu[i * m + j] = Cube((_rest[i] - _rest[j]).GetLength());
        double const p[4] = {1.0, _rest[i][0], _rest[i][1], _rest[i][2]};
        for (size_t q = 0; q < 4; ++q) {
            _lu[i * m + n + q] = p[q];
            _lu[(n + q) * m + i] = p[q];
        }
    }
    double maxAbs = 0.0;
    for (double v : _lu) maxAbs = std::max(maxAbs, std::abs(v));

    // Partial-pivoting LU. The augmented matrix is symmetric indefinite with
    // a zero block, so it is not a Cholesky candidate.
    _pivot.resize(m);
    for (size_t c = 0; c < m; ++c) {
        size_t pivot = c;
        for (size_t r = c + 1; r < m; ++r)
            if (std::abs(_lu[r * m + c]) > std::abs(_lu[pivot * m + c])) pivot = r;
        _pivot[c] = pivot;
        if (std::abs(_lu[pivot * m + c]) <= maxAbs * 1e-13)
            return fail("the RBF system is singular: two samples coincide");
        if (pivot != c)
            for (size_t k = 0; k < m; ++k) std::swap(_lu[c * m + k], _lu[pivot * m + k]);
        double const inverse = 1.0 / _lu[c * m + c];
        for (size_t r = c + 1; r < m; ++r) {
            double const factor = _lu[r * m + c] * inverse;
            _lu[r * m + c] = factor;
            if (factor == 0.0) continue;
            for (size_t k = c + 1; k < m; ++k) _lu[r * m + k] -= factor * _lu[c * m + k];
        }
    }
    _order = m;
    _coefficients.assign(3 * m, 0.0);   // unsolved: the identity field
    _restX.resize(n);
    _restY.resize(n);
    _restZ.resize(n);
    for (size_t i = 0; i < n; ++i) {
        _restX[i] = _rest[i][0];
        _restY[i] = _rest[i][1];
        _restZ[i] = _rest[i][2];
    }
    return true;
}

bool CubicField::Solve(std::vector<GfVec3d> const &current, std::string *error)
{
    size_t const n = _rest.size(), m = _order;
    if (!m) {
        if (error) *error = "the RBF field is not bound";
        return false;
    }
    if (current.size() != n) {
        if (error) *error = "the RBF pose has a different sample count than its rest";
        return false;
    }
    // One pass: Finite checks the whole sample, so the old d-outer loop
    // re-checked every sample three times; the first failure and the
    // message are unchanged.
    for (size_t i = 0; i < n; ++i) {
        if (!Finite(current[i])) {
            if (error) *error = "the RBF pose contains non-finite values";
            return false;
        }
    }
    // The three columns are independent (disjoint lanes over shared
    // read-only factors), so they run interleaved: each column keeps its
    // exact op sequence (same operations in the same order), giving the
    // dependent accumulation chain three times the ILP with
    // bitwise-identical coefficients.
    std::vector<double> w(size_t(3) * m);
    double *w0 = w.data(), *w1 = w.data() + m, *w2 = w.data() + size_t(2) * m;
    for (size_t i = 0; i < n; ++i) {
        w0[i] = (current[i][0] - _centre[0]) / _scale - _rest[i][0];
        w1[i] = (current[i][1] - _centre[1]) / _scale - _rest[i][1];
        w2[i] = (current[i][2] - _centre[2]) / _scale - _rest[i][2];
    }
    std::fill(w0 + n, w0 + m, 0.0);
    std::fill(w1 + n, w1 + m, 0.0);
    std::fill(w2 + n, w2 + m, 0.0);
    for (size_t c = 0; c < m; ++c) {
        size_t const p = _pivot[c];
        std::swap(w0[c], w0[p]);
        std::swap(w1[c], w1[p]);
        std::swap(w2[c], w2[p]);
    }
    for (size_t r = 1; r < m; ++r) {           // L (unit diagonal)
        double s0 = w0[r], s1 = w1[r], s2 = w2[r];
        for (size_t k = 0; k < r; ++k) {
            double const l = _lu[r * m + k];
            s0 -= l * w0[k];
            s1 -= l * w1[k];
            s2 -= l * w2[k];
        }
        w0[r] = s0;
        w1[r] = s1;
        w2[r] = s2;
    }
    for (size_t r = m; r-- > 0;) {             // U
        double s0 = w0[r], s1 = w1[r], s2 = w2[r];
        for (size_t k = r + 1; k < m; ++k) {
            double const l = _lu[r * m + k];
            s0 -= l * w0[k];
            s1 -= l * w1[k];
            s2 -= l * w2[k];
        }
        double const d = _lu[r * m + r];
        w0[r] = s0 / d;
        w1[r] = s1 / d;
        w2[r] = s2 / d;
    }
    std::copy(w.begin(), w.end(), _coefficients.begin());
    return true;
}

// CubicField::Displacement lives in rbfField.h (inlined at the per-CV call
// sites); see there.

namespace {

// One fixed-width block of DisplaceBatch: W queries share a single sample
// pass. Every query runs Displacement's operations in Displacement's order
// (the same expression text, so the same FMA contraction applies), which
// is what keeps the block bitwise; the accumulators stay one array per
// query so the vectorizer contracts across lanes instead of splitting mul
// and add. Eight is the measured sweet spot on ARM64 (four leaves FMA
// latency exposed, sixteen spills).
template <size_t W>
void DisplaceBlocked(GfVec3d const &centre, double invScale, double scale,
                     double const *restX, double const *restY, double const *restZ,
                     double const *cx, double const *cy, double const *cz,
                     size_t n, GfVec3d const *qs, GfVec3d *ds)
{
    double p[W][3];
    for (size_t t = 0; t < W; ++t) {
        p[t][0] = (qs[t][0] - centre[0]) * invScale;
        p[t][1] = (qs[t][1] - centre[1]) * invScale;
        p[t][2] = (qs[t][2] - centre[2]) * invScale;
    }
    double o[W][3];
    for (size_t t = 0; t < W; ++t) {
        o[t][0] = cx[n] + cx[n + 1] * p[t][0] + cx[n + 2] * p[t][1] + cx[n + 3] * p[t][2];
        o[t][1] = cy[n] + cy[n + 1] * p[t][0] + cy[n + 2] * p[t][1] + cy[n + 3] * p[t][2];
        o[t][2] = cz[n] + cz[n + 1] * p[t][0] + cz[n + 2] * p[t][1] + cz[n + 3] * p[t][2];
    }
    for (size_t i = 0; i < n; ++i) {
        double const sx = restX[i], sy = restY[i], sz = restZ[i];
        double kk[W];
        for (size_t t = 0; t < W; ++t) {
            double const dx = p[t][0] - sx, dy = p[t][1] - sy, dz = p[t][2] - sz;
            double const rr = std::sqrt(dx * dx + dy * dy + dz * dz);
            kk[t] = rr * rr * rr;
        }
        for (size_t t = 0; t < W; ++t) {
            o[t][0] += cx[i] * kk[t];
            o[t][1] += cy[i] * kk[t];
            o[t][2] += cz[i] * kk[t];
        }
    }
    for (size_t t = 0; t < W; ++t)
        ds[t] = GfVec3d(o[t][0] * scale, o[t][1] * scale, o[t][2] * scale);
}

#if defined(__aarch64__) || defined(_M_ARM64)
// Block pointer bundles for the NEON 4-block template: AoS GfVec3d runs
// (the DisplaceBatch shape) and planar float-in/double-out (the deform
// strands shape, whose queries already sit in planar float planes).
struct AosBlock4 {
    GfVec3d const *qs;
    GfVec3d *ds;
};
struct PlanarFBlock4 {
    float const *qx, *qy, *qz;
    double *dx, *dy, *dz;
};
struct PlanarFBlock8 {
    float const *qx, *qy, *qz;
    double *dx, *dy, *dz;
};
// Fused 4-query NEON block: the same per-lane operations as DisplaceBlocked
// (same FMA contraction association, correctly-rounded vector sqrt), but
// the kernel evaluation and the accumulation fuse into one sample pass
// with the normalised queries and accumulators held in vector registers.
// The auto-vectorized template above spills the queries to the stack and
// reloads them twice per sample; at 0.62ns/pair it runs 2.2x off the
// 0.28ns vector-sqrt roof, and the fused loop closes most of the
// schedulable gap (0.48ns/pair, on the FP-pipe wall for this op mix).
// Only the query load and the displacement store vary by bundle (vld3/vst3
// for AoS, plain loads/stores for planar); the affine prologue, the sample
// pass, and the FMA association below are one shared text, so both bundles
// evaluate bitwise what the other does.
template <class B>
void DisplaceBlockedNeon4T(GfVec3d const &centre, double invScale, double scale,
                           double const *restX, double const *restY, double const *restZ,
                           double const *cx, double const *cy, double const *cz,
                           size_t n, B blk)
{
    float64x2_t const c0 = vdupq_n_f64(centre[0]);
    float64x2_t const c1 = vdupq_n_f64(centre[1]);
    float64x2_t const c2 = vdupq_n_f64(centre[2]);
    float64x2_t const invS = vdupq_n_f64(invScale);
    // Plain locals, not arrays: the queries and accumulators must stay in
    // vector registers across the sample pass (arrays spill to the stack
    // and reload twice per sample).
    float64x2_t px0, px1, py0, py1, pz0, pz1;
    if constexpr (std::is_same<B, AosBlock4>::value) {
        float64x2x3_t const q01 = vld3q_f64(&blk.qs[0][0]);
        float64x2x3_t const q23 = vld3q_f64(&blk.qs[2][0]);
        px0 = vmulq_f64(vsubq_f64(q01.val[0], c0), invS);
        px1 = vmulq_f64(vsubq_f64(q23.val[0], c0), invS);
        py0 = vmulq_f64(vsubq_f64(q01.val[1], c1), invS);
        py1 = vmulq_f64(vsubq_f64(q23.val[1], c1), invS);
        pz0 = vmulq_f64(vsubq_f64(q01.val[2], c2), invS);
        pz1 = vmulq_f64(vsubq_f64(q23.val[2], c2), invS);
    } else {
        // float32x2 -> float64x2 is exact, so these six are bitwise the
        // vld3 path's six for the same query values.
        px0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx)), c0), invS);
        px1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx + 2)), c0), invS);
        py0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy)), c1), invS);
        py1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy + 2)), c1), invS);
        pz0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz)), c2), invS);
        pz1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz + 2)), c2), invS);
    }
    // One axis per scope: the twelve affine broadcasts must not all be
    // live at once, or the allocator spills the queries it just loaded.
    float64x2_t ox0, ox1, oy0, oy1, oz0, oz1;
    {
        float64x2_t const a0 = vdupq_n_f64(cx[n]);
        float64x2_t const a1 = vdupq_n_f64(cx[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cx[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cx[n + 3]);
        ox0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        ox1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cy[n]);
        float64x2_t const a1 = vdupq_n_f64(cy[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cy[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cy[n + 3]);
        oy0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        oy1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cz[n]);
        float64x2_t const a1 = vdupq_n_f64(cz[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cz[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cz[n + 3]);
        oz0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        oz1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
    }
    // Unroll ×32: iterations are independent (each query's accumulation
    // keeps sample order, so the bits match), and unrolling amortizes the
    // loop overhead plus the indexed-address setup across more samples.
    // The addressing overhead does not amortize monotonically (×2: 35.0,
    // ×4: 36.3, ×8: 34.4 instructions per sample versus 36.0 rolled), and
    // the measured curve tops out at ×32 (×8/×12 tie below, ×16/×24 tie
    // above those, ×32/×48 tie at the top at ~33 per sample; kernel
    // 9.96ms summed at ×8 versus 9.86ms at ×32 on this ARM64). No spills
    // at ×32; the remainder epilogue is one predictable dispatch per
    // block. Other compilers see no pragma and keep the rolled loop.
#if defined(__clang__)
#pragma clang loop unroll_count(32)
#elif defined(__GNUC__)
#pragma GCC unroll 32
#endif
    for (size_t i = 0; i < n; ++i) {
        // Broadcast loads keep one live register per sample value (the
        // compiler serves the coefficient lanes from scalar loads into
        // indexed FMA).
        float64x2_t const sx = vld1q_dup_f64(&restX[i]);
        float64x2_t const sy = vld1q_dup_f64(&restY[i]);
        float64x2_t const sz = vld1q_dup_f64(&restZ[i]);
        float64x2_t const qx = vld1q_dup_f64(&cx[i]);
        float64x2_t const qy = vld1q_dup_f64(&cy[i]);
        float64x2_t const qz = vld1q_dup_f64(&cz[i]);
        // Seed with dy*dy: the scalar blocks' codegen evaluates the sum
        // as fma(dz,dz,fma(dx,dx,dy*dy)), and the seed choice is
        // observable (exact-product-plus-rounded-addend commutes only in
        // the product, not across the addend).
        // Both pairs' differences first so each broadcast dies early.
        float64x2_t const dx0 = vsubq_f64(px0, sx);
        float64x2_t const dx1 = vsubq_f64(px1, sx);
        float64x2_t const dy0 = vsubq_f64(py0, sy);
        float64x2_t const dy1 = vsubq_f64(py1, sy);
        float64x2_t const dz0 = vsubq_f64(pz0, sz);
        float64x2_t const dz1 = vsubq_f64(pz1, sz);
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
    float64x2_t const sc = vdupq_n_f64(scale);
    if constexpr (std::is_same<B, AosBlock4>::value) {
        float64x2x3_t d01, d23;
        d01.val[0] = vmulq_f64(ox0, sc);
        d01.val[1] = vmulq_f64(oy0, sc);
        d01.val[2] = vmulq_f64(oz0, sc);
        d23.val[0] = vmulq_f64(ox1, sc);
        d23.val[1] = vmulq_f64(oy1, sc);
        d23.val[2] = vmulq_f64(oz1, sc);
        vst3q_f64(&blk.ds[0][0], d01);
        vst3q_f64(&blk.ds[2][0], d23);
    } else {
        vst1q_f64(blk.dx, vmulq_f64(ox0, sc));
        vst1q_f64(blk.dx + 2, vmulq_f64(ox1, sc));
        vst1q_f64(blk.dy, vmulq_f64(oy0, sc));
        vst1q_f64(blk.dy + 2, vmulq_f64(oy1, sc));
        vst1q_f64(blk.dz, vmulq_f64(oz0, sc));
        vst1q_f64(blk.dz + 2, vmulq_f64(oz1, sc));
    }
}

// Eight-wide planar NEON block: two 4-query halves over one shared sample
// pass. Each lane runs the 4-wide block's operations in the 4-wide order
// (same expression text per lane, same sample order), so the eight lanes
// are bitwise the 4-wide block run twice; only the sample/coefficient
// broadcasts amortize over eight queries instead of four. The wider pass
// runs the planar kernel ~4% faster (9.53 -> 9.14ms single-threaded over
// 200k x 100 bench-shaped pairs, 7/7 interleaved A/B pairs both orders).
// The allocator spills a few accumulators to the L1-resident stack, but
// the halved sample/coefficient broadcast traffic still wins net.
// Planar only: the AoS bundle's
// vld3/vst3 prologue and epilogue spill the 8-wide sample loop back over
// the 4-wide time (+1.5%), so AoS keeps the 4-wide block above. Unroll
// x16: x8/x16/x32 all tie (the overhead fully amortizes), so the middle
// keeps the code small. Other compilers see no pragma and keep the rolled
// loop.
void DisplaceBlockedNeon8Planar(GfVec3d const &centre, double invScale,
                               double scale, double const *restX,
                               double const *restY, double const *restZ,
                               double const *cx, double const *cy,
                               double const *cz, size_t n, PlanarFBlock8 blk)
{
    float64x2_t const c0 = vdupq_n_f64(centre[0]);
    float64x2_t const c1 = vdupq_n_f64(centre[1]);
    float64x2_t const c2 = vdupq_n_f64(centre[2]);
    float64x2_t const invS = vdupq_n_f64(invScale);
    float64x2_t px0, px1, px2, px3, py0, py1, py2, py3, pz0, pz1, pz2, pz3;
    px0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx)), c0), invS);
    px1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx + 2)), c0), invS);
    px2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx + 4)), c0), invS);
    px3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qx + 6)), c0), invS);
    py0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy)), c1), invS);
    py1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy + 2)), c1), invS);
    py2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy + 4)), c1), invS);
    py3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qy + 6)), c1), invS);
    pz0 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz)), c2), invS);
    pz1 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz + 2)), c2), invS);
    pz2 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz + 4)), c2), invS);
    pz3 = vmulq_f64(vsubq_f64(vcvt_f64_f32(vld1_f32(blk.qz + 6)), c2), invS);
    float64x2_t ox0, ox1, ox2, ox3, oy0, oy1, oy2, oy3, oz0, oz1, oz2, oz3;
    {
        float64x2_t const a0 = vdupq_n_f64(cx[n]);
        float64x2_t const a1 = vdupq_n_f64(cx[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cx[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cx[n + 3]);
        ox0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        ox1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
        ox2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px2), a2, py2), a3, pz2);
        ox3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px3), a2, py3), a3, pz3);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cy[n]);
        float64x2_t const a1 = vdupq_n_f64(cy[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cy[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cy[n + 3]);
        oy0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        oy1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
        oy2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px2), a2, py2), a3, pz2);
        oy3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px3), a2, py3), a3, pz3);
    }
    {
        float64x2_t const a0 = vdupq_n_f64(cz[n]);
        float64x2_t const a1 = vdupq_n_f64(cz[n + 1]);
        float64x2_t const a2 = vdupq_n_f64(cz[n + 2]);
        float64x2_t const a3 = vdupq_n_f64(cz[n + 3]);
        oz0 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px0), a2, py0), a3, pz0);
        oz1 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px1), a2, py1), a3, pz1);
        oz2 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px2), a2, py2), a3, pz2);
        oz3 = vfmaq_f64(vfmaq_f64(vfmaq_f64(a0, a1, px3), a2, py3), a3, pz3);
    }
#if defined(__clang__)
#pragma clang loop unroll_count(16)
#elif defined(__GNUC__)
#pragma GCC unroll 16
#endif
    for (size_t i = 0; i < n; ++i) {
        float64x2_t const sx = vld1q_dup_f64(&restX[i]);
        float64x2_t const sy = vld1q_dup_f64(&restY[i]);
        float64x2_t const sz = vld1q_dup_f64(&restZ[i]);
        float64x2_t const qx = vld1q_dup_f64(&cx[i]);
        float64x2_t const qy = vld1q_dup_f64(&cy[i]);
        float64x2_t const qz = vld1q_dup_f64(&cz[i]);
        float64x2_t const dx0 = vsubq_f64(px0, sx);
        float64x2_t const dx1 = vsubq_f64(px1, sx);
        float64x2_t const dx2 = vsubq_f64(px2, sx);
        float64x2_t const dx3 = vsubq_f64(px3, sx);
        float64x2_t const dy0 = vsubq_f64(py0, sy);
        float64x2_t const dy1 = vsubq_f64(py1, sy);
        float64x2_t const dy2 = vsubq_f64(py2, sy);
        float64x2_t const dy3 = vsubq_f64(py3, sy);
        float64x2_t const dz0 = vsubq_f64(pz0, sz);
        float64x2_t const dz1 = vsubq_f64(pz1, sz);
        float64x2_t const dz2 = vsubq_f64(pz2, sz);
        float64x2_t const dz3 = vsubq_f64(pz3, sz);
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
    float64x2_t const sc = vdupq_n_f64(scale);
    vst1q_f64(blk.dx, vmulq_f64(ox0, sc));
    vst1q_f64(blk.dx + 2, vmulq_f64(ox1, sc));
    vst1q_f64(blk.dx + 4, vmulq_f64(ox2, sc));
    vst1q_f64(blk.dx + 6, vmulq_f64(ox3, sc));
    vst1q_f64(blk.dy, vmulq_f64(oy0, sc));
    vst1q_f64(blk.dy + 2, vmulq_f64(oy1, sc));
    vst1q_f64(blk.dy + 4, vmulq_f64(oy2, sc));
    vst1q_f64(blk.dy + 6, vmulq_f64(oy3, sc));
    vst1q_f64(blk.dz, vmulq_f64(oz0, sc));
    vst1q_f64(blk.dz + 2, vmulq_f64(oz1, sc));
    vst1q_f64(blk.dz + 4, vmulq_f64(oz2, sc));
    vst1q_f64(blk.dz + 6, vmulq_f64(oz3, sc));
}
#endif

// Shared per-range parameters for the serial loops below: every query
// runs the same operations whatever its batch position, so any partition
// (one serial run, or one slice per pool worker) is bitwise identical.
struct DisplaceParams {
    GfVec3d centre{0.0};
    double invScale = 1.0;
    double scale = 1.0;
    double const *rx = nullptr, *ry = nullptr, *rz = nullptr;
    double const *cx = nullptr, *cy = nullptr, *cz = nullptr;
    size_t n = 0;
};

// The DisplaceBatch serial loop over [qs, qs+count): NEON blocks of 4
// (scalar blocks when forced, or off AArch64) plus the Displacement
// tail. One text for the serial entry and the pool workers' slices.
void DisplaceRangeAos(CubicField const &field, DisplaceParams const &p,
                      bool scalar, GfVec3d const *qs, GfVec3d *ds,
                      size_t count)
{
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!scalar) {
        while (count >= 4) {
            DisplaceBlockedNeon4T(p.centre, p.invScale, p.scale, p.rx, p.ry,
                                  p.rz, p.cx, p.cy, p.cz, p.n,
                                  AosBlock4{qs, ds});
            qs += 4;
            ds += 4;
            count -= 4;
        }
        for (size_t t = 0; t < count; ++t) ds[t] = field.Displacement(qs[t]);
        return;
    }
#else
    (void)scalar;
#endif
    while (count >= 8) {
        DisplaceBlocked<8>(p.centre, p.invScale, p.scale, p.rx, p.ry, p.rz,
                           p.cx, p.cy, p.cz, p.n, qs, ds);
        qs += 8;
        ds += 8;
        count -= 8;
    }
    if (count >= 4) {
        DisplaceBlocked<4>(p.centre, p.invScale, p.scale, p.rx, p.ry, p.rz,
                           p.cx, p.cy, p.cz, p.n, qs, ds);
        qs += 4;
        ds += 4;
        count -= 4;
    }
    for (size_t t = 0; t < count; ++t) ds[t] = field.Displacement(qs[t]);
}

// The DisplaceBatchPlanar serial loop over its planes: the 8-wide NEON
// block, then the 4-wide block for the remainder (scalar widths transpose
// through the stack, as before) plus the Displacement tails.
void DisplaceRangePlanar(CubicField const &field, DisplaceParams const &p,
                         bool scalar, float const *qx, float const *qy,
                         float const *qz, double *dx, double *dy, double *dz,
                         size_t count)
{
#if defined(__aarch64__) || defined(_M_ARM64)
    if (!scalar) {
        while (count >= 8) {
            DisplaceBlockedNeon8Planar(p.centre, p.invScale, p.scale, p.rx,
                                       p.ry, p.rz, p.cx, p.cy, p.cz, p.n,
                                       PlanarFBlock8{qx, qy, qz, dx, dy, dz});
            qx += 8;
            qy += 8;
            qz += 8;
            dx += 8;
            dy += 8;
            dz += 8;
            count -= 8;
        }
        while (count >= 4) {
            DisplaceBlockedNeon4T(p.centre, p.invScale, p.scale, p.rx, p.ry,
                                  p.rz, p.cx, p.cy, p.cz, p.n,
                                  PlanarFBlock4{qx, qy, qz, dx, dy, dz});
            qx += 4;
            qy += 4;
            qz += 4;
            dx += 4;
            dy += 4;
            dz += 4;
            count -= 4;
        }
        for (size_t t = 0; t < count; ++t) {
            GfVec3d const d = field.Displacement(GfVec3d(qx[t], qy[t], qz[t]));
            dx[t] = d[0];
            dy[t] = d[1];
            dz[t] = d[2];
        }
        return;
    }
#else
    (void)scalar;
#endif
    // Scalar widths transpose through the stack: the block template keeps
    // its AoS spelling (and its FMA association), and the gather/scatter
    // convert exactly.
    while (count >= 8) {
        GfVec3d q[8], d[8];
        for (size_t t = 0; t < 8; ++t) q[t] = GfVec3d(qx[t], qy[t], qz[t]);
        DisplaceBlocked<8>(p.centre, p.invScale, p.scale, p.rx, p.ry, p.rz,
                           p.cx, p.cy, p.cz, p.n, q, d);
        for (size_t t = 0; t < 8; ++t) {
            dx[t] = d[t][0];
            dy[t] = d[t][1];
            dz[t] = d[t][2];
        }
        qx += 8;
        qy += 8;
        qz += 8;
        dx += 8;
        dy += 8;
        dz += 8;
        count -= 8;
    }
    if (count >= 4) {
        GfVec3d q[4], d[4];
        for (size_t t = 0; t < 4; ++t) q[t] = GfVec3d(qx[t], qy[t], qz[t]);
        DisplaceBlocked<4>(p.centre, p.invScale, p.scale, p.rx, p.ry, p.rz,
                           p.cx, p.cy, p.cz, p.n, q, d);
        for (size_t t = 0; t < 4; ++t) {
            dx[t] = d[t][0];
            dy[t] = d[t][1];
            dz[t] = d[t][2];
        }
        qx += 4;
        qy += 4;
        qz += 4;
        dx += 4;
        dy += 4;
        dz += 4;
        count -= 4;
    }
    for (size_t t = 0; t < count; ++t) {
        GfVec3d const d = field.Displacement(GfVec3d(qx[t], qy[t], qz[t]));
        dx[t] = d[0];
        dy[t] = d[1];
        dz[t] = d[2];
    }
}

// Large-batch threading: a 200k-query batch is ~9ms of fp64 sqrt work on
// one core, compute-bound and per-query independent, so batches at or
// above kDisplaceThreadQueries split over a process-wide pool in
// kDisplaceBlock-query slices (a multiple of the NEON width, so only
// the last slice carries a tail). Production deform batches (~256
// queries per strand group, already spread over the scheduler pool)
// stay serial, as do nested calls from pool workers (the re-entry
// guard); concurrent top-level dispatches serialize on the pool mutex
// (one ParallelFor per pool at a time).
size_t constexpr kDisplaceThreadQueries = 16384;
size_t constexpr kDisplaceBlock = 1024;

UsdGenWorkerPool &DisplacePool()
{
    // Leaked like the TBB market: joining workers at DSO unload waits
    // for threads process shutdown may already have stopped.
    static UsdGenWorkerPool *const pool = [] {
        // The scheduler's sizing: the fast-core count where the topology
        // qualifies (pinning engages by construction), else the 8-thread
        // floor. 8/10/20 measured 1.44/1.14/1.30ms on the bench's 200k
        // batch (the 20-wide float loses to 10 pinned fast cores).
        return new UsdGenWorkerPool(std::max(8, FastCoreCount()));
    }();
    return *pool;
}

std::mutex &DisplacePoolMutex()
{
    static std::mutex mutex;
    return mutex;
}

thread_local bool t_inDisplaceBody = false;

struct DisplaceBodyGuard {
    DisplaceBodyGuard() { t_inDisplaceBody = true; }
    ~DisplaceBodyGuard() { t_inDisplaceBody = false; }
};

struct AosJob {
    CubicField const *field;
    DisplaceParams params;
    bool scalar;
    GfVec3d const *qs;
    GfVec3d *ds;
    size_t count;
};

void DisplaceAosBlock(size_t b, void *payload)
{
    DisplaceBodyGuard guard;
    AosJob const &job = *static_cast<AosJob const *>(payload);
    size_t const lo = b * kDisplaceBlock;
    size_t const hi = std::min(lo + kDisplaceBlock, job.count);
    DisplaceRangeAos(*job.field, job.params, job.scalar, job.qs + lo,
                     job.ds + lo, hi - lo);
}

struct PlanarJob {
    CubicField const *field;
    DisplaceParams params;
    bool scalar;
    float const *qx, *qy, *qz;
    double *dx, *dy, *dz;
    size_t count;
};

void DisplacePlanarBlock(size_t b, void *payload)
{
    DisplaceBodyGuard guard;
    PlanarJob const &job = *static_cast<PlanarJob const *>(payload);
    size_t const lo = b * kDisplaceBlock;
    size_t const hi = std::min(lo + kDisplaceBlock, job.count);
    DisplaceRangePlanar(*job.field, job.params, job.scalar, job.qx + lo,
                        job.qy + lo, job.qz + lo, job.dx + lo, job.dy + lo,
                        job.dz + lo, hi - lo);
}

// False when nested (the caller runs the serial range instead).
bool DisplaceDispatch(size_t count, void (*body)(size_t, void *),
                      void *payload)
{
    if (t_inDisplaceBody) return false;
    std::lock_guard<std::mutex> lock(DisplacePoolMutex());
    DisplacePool().ParallelFor((count + kDisplaceBlock - 1) / kDisplaceBlock,
                               body, payload);
    return true;
}

}  // namespace

namespace {
std::atomic<bool> g_forceScalarDisplace{false};
}  // namespace

void TestForceScalarDisplace(bool force) noexcept
{
    g_forceScalarDisplace.store(force, std::memory_order_relaxed);
}

void CubicField::DisplaceBatch(GfVec3d const *qs, GfVec3d *ds, size_t count) const
{
    if (count == 0) return;
    size_t const n = _rest.size(), m = _order;
    if (!m) {
        for (size_t t = 0; t < count; ++t) ds[t] = GfVec3d(0.0);
        return;
    }
    DisplaceParams const p{_centre,
                           _invScale,
                           _scale,
                           _restX.data(),
                           _restY.data(),
                           _restZ.data(),
                           &_coefficients[0],
                           &_coefficients[m],
                           &_coefficients[2 * m],
                           n};
    bool const scalar = g_forceScalarDisplace.load(std::memory_order_relaxed);
    if (count >= kDisplaceThreadQueries) {
        AosJob job{this, p, scalar, qs, ds, count};
        if (DisplaceDispatch(count, DisplaceAosBlock, &job)) return;
    }
    DisplaceRangeAos(*this, p, scalar, qs, ds, count);
}

void CubicField::DisplaceBatchPlanar(float const *qx, float const *qy, float const *qz,
                                     double *dx, double *dy, double *dz,
                                     size_t count) const
{
    if (count == 0) return;
    size_t const n = _rest.size(), m = _order;
    if (!m) {
        for (size_t t = 0; t < count; ++t) dx[t] = dy[t] = dz[t] = 0.0;
        return;
    }
    DisplaceParams const p{_centre,
                           _invScale,
                           _scale,
                           _restX.data(),
                           _restY.data(),
                           _restZ.data(),
                           &_coefficients[0],
                           &_coefficients[m],
                           &_coefficients[2 * m],
                           n};
    bool const scalar = g_forceScalarDisplace.load(std::memory_order_relaxed);
    if (count >= kDisplaceThreadQueries) {
        PlanarJob job{this, p, scalar, qx, qy, qz, dx, dy, dz, count};
        if (DisplaceDispatch(count, DisplacePlanarBlock, &job)) return;
    }
    DisplaceRangePlanar(*this, p, scalar, qx, qy, qz, dx, dy, dz, count);
}

std::vector<size_t> SelectSamples(std::vector<GfVec3d> const &points, size_t budget,
                                  double epsilon)
{
    // Exact duplicates first: they make the system singular.
    struct KeyHash {
        size_t operator()(GfVec3i const &k) const
        {
            return size_t(uint64_t(uint32_t(k[0])) * 73856093ULL ^
                          uint64_t(uint32_t(k[1])) * 19349663ULL ^
                          uint64_t(uint32_t(k[2])) * 83492791ULL);
        }
    };
    double const cell = epsilon > 0.0 ? epsilon : 1e-9;
    std::unordered_set<GfVec3i, KeyHash> seen;
    std::vector<size_t> unique;
    unique.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        if (!Finite(points[i])) continue;
        GfVec3d const q = points[i] / cell;
        auto clampInt = [](double v) {
            return int(std::max(-2.0e9, std::min(2.0e9, std::floor(v + 0.5))));
        };
        if (seen.insert(GfVec3i(clampInt(q[0]), clampInt(q[1]), clampInt(q[2]))).second)
            unique.push_back(i);
    }
    if (unique.size() <= budget) return unique;

    // Farthest-point sampling from the first point.
    std::vector<size_t> chosen;
    chosen.reserve(budget);
    std::vector<double> distance(unique.size(), std::numeric_limits<double>::infinity());
    size_t next = 0;
    while (chosen.size() < budget) {
        chosen.push_back(unique[next]);
        GfVec3d const &p = points[unique[next]];
        size_t best = 0;
        double bestDistance = -1.0;
        for (size_t k = 0; k < unique.size(); ++k) {
            distance[k] = std::min(distance[k], (points[unique[k]] - p).GetLengthSq());
            if (distance[k] > bestDistance) {
                bestDistance = distance[k];
                best = k;
            }
        }
        if (bestDistance <= 0.0) break;
        next = best;
    }
    std::sort(chosen.begin(), chosen.end());
    return chosen;
}

}  // namespace rbf
}  // namespace usdGen
