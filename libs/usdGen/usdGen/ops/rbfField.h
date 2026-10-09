// usdGen — host cubic RBF displacement field (plan/examples/rbf-deformation.md).
//
// The CPU twin of gpu::CudaRbfBinding's formulation, used by UsdGenDeform on
// the CPU lane:
//
//   y       = (x - centre) / scale       centre/scale: the rest samples' extent
//   K[i,j]  = |y_i - y_j|^3              P[i] = [1, y_i]
//   [K P; P^T 0] [a; b] = [d; 0]         d_i = (current_i - rest_i) / scale
//   F(x)    = x + scale * (sum_i a_i |y - y_i|^3 + b . [1, y])
//
// so F reproduces every sample exactly, and any affine motion of the samples
// (translation, rotation, scale) everywhere. Bind factors the system once per
// rest layout; Solve is a pair of triangular solves per pose.
#ifndef USDGEN_OPS_RBF_FIELD_H
#define USDGEN_OPS_RBF_FIELD_H

#include "pxr/base/gf/vec3d.h"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace rbf {

// Test-only DisplaceBatch path seam. While forced, DisplaceBatch runs the
// scalar blocks even where the NEON block is available, so a test can
// compare the two paths bitwise on the same field.
void TestForceScalarDisplace(bool force) noexcept;

class CubicField
{
public:
    /// Factors the system for `rest`. False, with the reason, for fewer than
    /// four samples, non-finite input, zero extent, or samples that do not
    /// span 3D (coplanar or collinear: the affine term is then undetermined).
    bool Bind(std::vector<GfVec3d> const &rest, std::string *error);

    /// Coefficients for the samples at `current` (same order as the bound
    /// rest samples).
    bool Solve(std::vector<GfVec3d> const &current, std::string *error);

    /// F(x) - x: the displacement of a rest-space point.
    ///
    /// Defined inline below: the deform strands loop and the field
    /// microbench call this once per CV, so a call boundary here costs a
    /// prologue, a 2KB stack carve, and a by-memory GfVec3d return on every
    /// query. The arithmetic is unchanged (same operations in the same
    /// order), so inlined queries are bitwise what the call returned.
    inline GfVec3d Displacement(GfVec3d const &x) const;

    /// A run of queries: ds[t] is bitwise Displacement(qs[t]) for every t.
    ///
    /// The single-query loop above is bound by its three serial FMA chains
    /// and the kernel row's round trip through the stack; blocking queries
    /// over one sample pass keeps every query's operations in the same
    /// order (so the bits match) while the samples load once and the
    /// accumulators overlap. On AArch64 a fused 4-wide NEON block holds
    /// the accumulators in registers (scalar blocks elsewhere). Defined
    /// out of line; tails fall back to Displacement, and an unbound field
    /// fills zeros. Large batches split over an internal worker pool;
    /// every query still runs the same operations, so the bits match the
    /// serial loop.
    void DisplaceBatch(GfVec3d const *qs, GfVec3d *ds, size_t count) const;

    /// A planar run of queries: (dx[t], dy[t], dz[t]) is bitwise
    /// Displacement(GfVec3d(qx[t], qy[t], qz[t])) for every t.
    ///
    /// The deform strands loop's queries already sit in planar float
    /// planes, so this entry skips the AoS transpose (and the vld3/vst3
    /// shuffles) while running each query's operations in the same order
    /// as DisplaceBatch, so the bits match. Tails fall back to
    /// Displacement, and an unbound field fills zeros. Large batches
    /// split over an internal worker pool, bitwise as above.
    void DisplaceBatchPlanar(float const *qx, float const *qy, float const *qz,
                             double *dx, double *dy, double *dz,
                             size_t count) const;

    bool Bound() const { return _order != 0; }
    size_t SampleCount() const { return _rest.size(); }

private:
    std::vector<GfVec3d> _rest;        // normalised rest samples
    std::vector<double> _restX, _restY, _restZ;   // the same, one array per axis
    GfVec3d _centre{0.0};
    double _scale = 1.0;
    // 1/_scale, computed once per bind: Displacement normalises every query
    // by _scale, and the compiler lowers that to a reciprocal plus three
    // multiplies, so caching the reciprocal bit-for-bit (same IEEE divide,
    // run once) removes a fully exposed fdiv from every query.
    double _invScale = 1.0;
    size_t _order = 0;                 // n + 4
    std::vector<double> _lu;           // row-major LU of the augmented matrix
    std::vector<size_t> _pivot;
    std::vector<double> _coefficients; // 3 * order: x, y, z columns
};

/// Up to `budget` well-spread indices into `points`: exact duplicates
/// (within `epsilon`) are dropped, then farthest-point sampling starts at the
/// first point. Deterministic for a given input.
std::vector<size_t> SelectSamples(std::vector<GfVec3d> const &points, size_t budget,
                                  double epsilon);

inline GfVec3d CubicField::Displacement(GfVec3d const &x) const
{
    size_t const n = _rest.size(), m = _order;
    if (!m) return GfVec3d(0.0);
    GfVec3d const y = (x - _centre) * _invScale;
    double const *cx = &_coefficients[0], *cy = &_coefficients[m], *cz = &_coefficients[2 * m];

    // The kernel row gets a loop of its own, which the compiler vectorizes
    // (the square root dominates a deform). The sums below depend on their
    // order, so they stay sample by sample and the result is unchanged.
    constexpr size_t kStackSamples = 256;
    double stackKernel[kStackSamples];
    double *k = stackKernel;
    if (n > kStackSamples) {
        thread_local std::vector<double> heapKernel;
        heapKernel.resize(n);
        k = heapKernel.data();
    }
    double const *rx = _restX.data(), *ry = _restY.data(), *rz = _restZ.data();
    double const px = y[0], py = y[1], pz = y[2];
    for (size_t i = 0; i < n; ++i) {
        double const dx = px - rx[i], dy = py - ry[i], dz = pz - rz[i];
        double const rr = std::sqrt(dx * dx + dy * dy + dz * dz);
        k[i] = rr * rr * rr;
    }

    double ox = cx[n] + cx[n + 1] * px + cx[n + 2] * py + cx[n + 3] * pz;
    double oy = cy[n] + cy[n + 1] * px + cy[n + 2] * py + cy[n + 3] * pz;
    double oz = cz[n] + cz[n + 1] * px + cz[n + 2] * py + cz[n + 3] * pz;
    for (size_t i = 0; i < n; ++i) {
        ox += cx[i] * k[i];
        oy += cy[i] * k[i];
        oz += cz[i] * k[i];
    }
    return GfVec3d(ox, oy, oz) * _scale;
}

}  // namespace rbf
}  // namespace usdGen

#endif  // USDGEN_OPS_RBF_FIELD_H
