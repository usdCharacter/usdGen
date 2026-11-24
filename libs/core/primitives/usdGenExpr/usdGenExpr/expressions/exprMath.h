// usdGen — the ONE implementation of every SeExpr2 builtin the expression
// language exposes, compiled for BOTH the host and the CUDA device.
//
// expressions/irExec.h includes this header; gpu/expression.cu and
// expressions/cpuEvaluator.cpp include irExec.h. There is therefore exactly one
// definition of every noise lattice, every hash, every colour conversion and
// every curve evaluation in the product, and the two lanes cannot drift.
//
// PARITY RULES OBSERVED HERE
//   * Everything is double. The vendored SeExpr2 host path (thirdparty/seexpr/
//     SeExpr2/Noise.cpp, thirdparty/seexprFrontend/SeExpr2/ExprBuiltins.cpp)
//     instantiates its noise at double, and CUDA is built with --fmad=false, so
//     +, -, *, /, sqrt, floor, ceil, fmod, fabs, fmin and fmax are IEEE-754
//     correctly rounded and therefore bit-identical on both lanes.
//   * The transcendentals (sin cos tan asin acos atan sinh cosh tanh exp log
//     log10 pow cbrt) come from the host CRT on one lane and libdevice on the
//     other. Each is correctly rounded to a fraction of an ulp but not
//     necessarily to the same bits, so anything built on them is documented as
//     "last bit may differ" and the parity test stays out of that set.
//   * Every double -> integer conversion goes through ExprInt/ExprUInt32Wrap.
//     A bare (int) or (uint32_t) cast of an out-of-range or negative double is
//     UB in C++ and genuinely differs between x86 (truncate through a 64-bit
//     register) and PTX (saturate), which would silently break cellnoise and
//     voronoi. These helpers pin the x86 answer on both lanes.
//   * No function keeps state. SeExpr2 caches voronoi cell points and prepared
//     curves in per-node Data objects; here the curve is prepared once by the
//     frontend into the instruction stream and voronoi recomputes its 27 cell
//     points, because a cache would be a second source of truth.
#ifndef USDGEN_EXPRESSIONS_EXPR_MATH_H
#define USDGEN_EXPRESSIONS_EXPR_MATH_H

#include "usdGen/expressions/exprNoiseTables.h"

#include <climits>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define USDGEN_EXPR_HD __host__ __device__
// nvcc resolves these to the CUDA device implementations in a device pass and
// to the C library in its host pass.
#define USDGEN_EXPR_FN(f) ::f
#else
#define USDGEN_EXPR_HD
#define USDGEN_EXPR_FN(f) std::f
#endif

namespace usdGen::expr {

// ---------------------------------------------------------------------------
// Tables. One macro payload, expanded once for each lane.
// ---------------------------------------------------------------------------
namespace tables {
#if defined(__CUDACC__)
static __device__ const double kNoiseG3Device[514][3] = USDGEN_EXPR_NOISE_G3_INIT;
static __device__ const int kNoiseP514Device[514] = USDGEN_EXPR_NOISE_P514_INIT;
static __device__ const unsigned char kHashP256Device[256] = USDGEN_EXPR_HASH_P256_INIT;
#endif
inline constexpr double kNoiseG3Host[514][3] = USDGEN_EXPR_NOISE_G3_INIT;
inline constexpr int kNoiseP514Host[514] = USDGEN_EXPR_NOISE_P514_INIT;
inline constexpr unsigned char kHashP256Host[256] = USDGEN_EXPR_HASH_P256_INIT;
} // namespace tables

USDGEN_EXPR_HD inline double NoiseG3(int lookup, int k)
{
#if defined(__CUDA_ARCH__)
    return tables::kNoiseG3Device[lookup][k];
#else
    return tables::kNoiseG3Host[lookup][k];
#endif
}
USDGEN_EXPR_HD inline int NoiseP514(int i)
{
#if defined(__CUDA_ARCH__)
    return tables::kNoiseP514Device[i];
#else
    return tables::kNoiseP514Host[i];
#endif
}
USDGEN_EXPR_HD inline unsigned char HashP256(int i)
{
#if defined(__CUDA_ARCH__)
    return tables::kHashP256Device[i];
#else
    return tables::kHashP256Host[i];
#endif
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// isfinite without depending on which namespace the platform put it in. NaN
/// fails both comparisons, so the result is exactly isfinite().
USDGEN_EXPR_HD inline bool ExprFinite(double value)
{
    return value > -INFINITY && value < INFINITY;
}

/// Truncating double -> int, saturating instead of invoking UB. NaN lands on
/// INT_MIN. Both lanes agree for every input, which a bare cast does not.
USDGEN_EXPR_HD inline int ExprInt(double v)
{
    if (!(v > -2147483649.0)) return INT_MIN;
    if (!(v < 2147483648.0)) return INT_MAX;
    return static_cast<int>(v);
}

/// SeExpr2's CellNoise casts a possibly-negative floor() result straight to
/// uint32_t. x86 truncates through a 64-bit register and wraps; PTX saturates
/// at zero. Going through int64_t explicitly keeps both lanes on the x86
/// answer, which is the one the vendored CPU oracle produces.
USDGEN_EXPR_HD inline uint32_t ExprUInt32Wrap(double v)
{
    if (!(v > -4.611686018427388e18)) return 0u;
    if (!(v < 4.611686018427388e18)) return 0xFFFFFFFFu;
    return static_cast<uint32_t>(static_cast<int64_t>(v));
}

USDGEN_EXPR_HD inline double ExprClamp(double x, double lo, double hi)
{
    return x < lo ? lo : x > hi ? hi : x;
}

// SeExpr2 ExprBuiltins.h: round(x) = x < 0 ? ceil(x - .5) : floor(x + .5).
// Written out rather than delegating to libm's round() so it is exact on both
// lanes by construction.
USDGEN_EXPR_HD inline double ExprRound(double x)
{
    return x < 0 ? USDGEN_EXPR_FN(ceil)(x - 0.5) : USDGEN_EXPR_FN(floor)(x + 0.5);
}

constexpr double kExprPi = 3.14159265358979323846;
constexpr double kExprE = 2.71828182845904523536;

// ---------------------------------------------------------------------------
// Blending / remapping (ExprBuiltins.cpp, verbatim).
//
// Only the ones an IROp::Call needs are here. SeExpr2's purely algebraic
// builtins -- deg rad invert compress fit gamma mix, and the cosd/sind/...
// family -- are lowered to arithmetic by expressions/frontend.cpp instead, so
// they cost no opcode and are bit-identical on both lanes for free. Keeping a
// second copy of their formulas here would be a second place to get them wrong.
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline double ExprExpand(double x, double lo, double hi)
{
    if (lo == hi) return x < lo ? 0 : 1;
    return (x - lo) / (hi - lo);
}
USDGEN_EXPR_HD inline double ExprBias(double x, double b)
{
    // SeExpr2 hoists C = 1/log(0.5) into a function-local static.
    const double C = 1 / USDGEN_EXPR_FN(log)(0.5);
    return USDGEN_EXPR_FN(pow)(x, USDGEN_EXPR_FN(log)(b) * C);
}
USDGEN_EXPR_HD inline double ExprContrast(double x, double c)
{
    if (x < 0.5) return 0.5 * ExprBias(1 - c, 2 * x);
    return 1 - 0.5 * ExprBias(1 - c, 2 - 2 * x);
}
USDGEN_EXPR_HD inline double ExprBoxStep(double x, double a) { return x < a ? 0.0 : 1.0; }
USDGEN_EXPR_HD inline double ExprLinearStep(double x, double a, double b)
{
    if (a < b) return x < a ? 0 : (x > b ? 1 : (x - a) / (b - a));
    if (a > b) return 1 - (x < b ? 0 : (x > a ? 1 : (x - b) / (a - b)));
    return ExprBoxStep(x, a);
}
USDGEN_EXPR_HD inline double ExprSmoothStep(double x, double a, double b)
{
    if (a < b) {
        if (x < a) return 0;
        if (x >= b) return 1;
        x = (x - a) / (b - a);
    } else if (a > b) {
        if (x <= b) return 1;
        if (x > a) return 0;
        x = 1 - (x - b) / (a - b);
    } else {
        return ExprBoxStep(x, a);
    }
    return x * x * (3 - 2 * x);
}
USDGEN_EXPR_HD inline double ExprGaussStep(double x, double a, double b)
{
    if (a < b) {
        if (x < a) return 0;
        if (x >= b) return 1;
        x = 1 - (x - a) / (b - a);
    } else if (a > b) {
        if (x <= b) return 1;
        if (x > a) return 0;
        x = (x - b) / (a - b);
    } else {
        return ExprBoxStep(x, a);
    }
    return USDGEN_EXPR_FN(pow)(2.0, -8 * x * x);
}
USDGEN_EXPR_HD inline double ExprRemap(double x, double source, double range,
                                       double falloff, double interp)
{
    range = USDGEN_EXPR_FN(fabs)(range);
    falloff = USDGEN_EXPR_FN(fabs)(falloff);
    if (falloff == 0) return USDGEN_EXPR_FN(fabs)(x - source) < range ? 1.0 : 0.0;
    double a, b;
    if (x > source) { a = source + range; b = a + falloff; }
    else            { a = source - range; b = a - falloff; }
    switch (ExprInt(interp)) {
    case 0: return ExprLinearStep(x, b, a);
    case 1: return ExprSmoothStep(x, b, a);
    default: return ExprGaussStep(x, b, a);
    }
}
USDGEN_EXPR_HD inline double ExprCycle(double index, double loRange, double hiRange)
{
    const int lo = ExprInt(loRange);
    const int hi = ExprInt(hiRange);
    const int range = hi - lo + 1;
    if (range <= 0) return lo;
    int result = ExprInt(index) % range;
    if (result < 0) result += range;
    return lo + result;
}

// ---------------------------------------------------------------------------
// Vectors (ExprBuiltins.cpp). `a`/`b` are 3-double blocks.
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline double ExprLength3(const double *v)
{
    return USDGEN_EXPR_FN(sqrt)(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
USDGEN_EXPR_HD inline double ExprDist(const double *a, const double *b)
{
    const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
    return USDGEN_EXPR_FN(sqrt)(x * x + y * y + z * z);
}
USDGEN_EXPR_HD inline double ExprHypot(double x, double y)
{
    // SeExpr2's hypot is sqrt(x*x + y*y), not libm hypot: exact on both lanes.
    return USDGEN_EXPR_FN(sqrt)(x * x + y * y);
}
USDGEN_EXPR_HD inline double ExprDot(const double *a, const double *b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
USDGEN_EXPR_HD inline void ExprCross(const double *a, const double *b, double *out)
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}
USDGEN_EXPR_HD inline void ExprNorm(const double *a, double *out)
{
    const double len = ExprLength3(a);
    if (len == 0) { out[0] = out[1] = out[2] = 0; return; }
    // SeExpr2's Vec::operator/(T) multiplies by the reciprocal rather than
    // dividing component by component (Vec.h operator/=). Dividing instead
    // costs a last-bit difference against the oracle, so do what it does.
    const double scale = 1.0 / len;
    out[0] = a[0] * scale; out[1] = a[1] * scale; out[2] = a[2] * scale;
}
USDGEN_EXPR_HD inline double ExprAngle(const double *a, const double *b)
{
    const double len = ExprLength3(a) * ExprLength3(b);
    if (len == 0) return 0;
    return USDGEN_EXPR_FN(acos)(ExprDot(a, b) / len);
}
USDGEN_EXPR_HD inline void ExprOrtho(const double *a, const double *b, double *out)
{
    double c[3];
    ExprCross(a, b, c);
    ExprNorm(c, out);
}
/// Vec<T,3>::rotateBy: c*v + (1-c)*dot(axis,v)*axis - s*cross(axis,v).
USDGEN_EXPR_HD inline void ExprRotateBy(const double *v, const double *axis,
                                        double angle, double *out)
{
    const double c = USDGEN_EXPR_FN(cos)(angle), s = USDGEN_EXPR_FN(sin)(angle);
    // Vec::rotateBy spells the last term as `this->cross(axis)`, so the
    // operands are (v, axis) in that order; swapping them negates the result.
    double cr[3];
    ExprCross(v, axis, cr);
    const double d = ExprDot(v, axis);
    for (int i = 0; i < 3; ++i)
        out[i] = c * v[i] + (1 - c) * d * axis[i] - s * cr[i];
}
USDGEN_EXPR_HD inline void ExprRotate(const double *P, const double *axis,
                                      double angle, double *out)
{
    const double len = ExprLength3(axis);
    if (len == 0) { out[0] = P[0]; out[1] = P[1]; out[2] = P[2]; return; }
    const double scale = 1.0 / len;   // Vec::operator/(T), as in ExprNorm
    const double unit[3] = {axis[0] * scale, axis[1] * scale, axis[2] * scale};
    // SeExpr2 narrows the angle to float before rotating; keep that.
    ExprRotateBy(P, unit, double(static_cast<float>(angle)), out);
}
USDGEN_EXPR_HD inline void ExprUp(const double *P, const double *upvec, double *out)
{
    const double yAxis[3] = {0, 1, 0};
    double axis[3];
    ExprOrtho(upvec, yAxis, axis);
    ExprRotateBy(P, axis, ExprAngle(upvec, yAxis), out);
}

// ---------------------------------------------------------------------------
// Colour (ExprBuiltins.cpp)
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline void ExprRgbToHsl(const double *rgb, double *out)
{
    const double R = rgb[0], G = rgb[1], B = rgb[2];
    const double x = R < G ? (R < B ? R : B) : (G < B ? G : B);
    const double y = R > G ? (R > B ? R : B) : (G > B ? G : B);
    const double sum = x + y, diff = y - x;
    const double L = sum / 2;
    if (diff < 1e-6) { out[0] = 0; out[1] = 0; out[2] = L; return; }
    double S;
    if (L <= .5) S = x < 0 ? 1 - x : diff / sum;
    else         S = y > 1 ? y : diff / (2 - sum);
    double H;
    if (R == y)      H = (G - B) / diff;
    else if (G == y) H = (B - R) / diff + 2;
    else             H = (R - G) / diff + 4;
    H *= 1 / 6.;
    H -= USDGEN_EXPR_FN(floor)(H);
    out[0] = H; out[1] = S; out[2] = L;
}
USDGEN_EXPR_HD inline double ExprHslValue(double x, double y, double H)
{
    H -= USDGEN_EXPR_FN(floor)(H);
    if (H < 1 / 6.) return x + (y - x) * H * 6;
    if (H < 3 / 6.) return y;
    if (H < 4 / 6.) return x + (y - x) * (4 / 6. - H) * 6;
    return x;
}
USDGEN_EXPR_HD inline void ExprHslToRgb(const double *hsl, double *out)
{
    const double H = hsl[0], S = hsl[1], L = hsl[2];
    if (S <= 0) { out[0] = out[1] = out[2] = L; return; }
    double y;
    if (L < 0.5) y = S > 1 ? 2 * L + S - 1 : L + L * S;
    else         y = S > 1 ? S : L + S - L * S;
    const double x = 2 * L - y;
    out[0] = ExprHslValue(x, y, H + (1 / 3.));
    out[1] = ExprHslValue(x, y, H);
    out[2] = ExprHslValue(x, y, H - (1 / 3.));
}
USDGEN_EXPR_HD inline void ExprSaturate(const double *Cin, double amt, double *out)
{
    const double lum[3] = {.2126, .7152, .0722};
    const double d = ExprDot(Cin, lum);
    for (int i = 0; i < 3; ++i) {
        out[i] = d * (1 - amt) + Cin[i] * amt;
        if (out[i] < 0) out[i] = 0;
    }
}
USDGEN_EXPR_HD inline void ExprHsiAdjust(const double *rgb, double h, double s,
                                         double i, double *out)
{
    double hsl[3];
    ExprRgbToHsl(rgb, hsl);
    hsl[0] += h * (1.0 / 360);
    hsl[1] *= s;
    ExprHslToRgb(hsl, out);
    out[0] *= i; out[1] *= i; out[2] *= i;
}
USDGEN_EXPR_HD inline void ExprHsi(const double *rgb, double h, double s, double i,
                                   double m, double *out)
{
    // The 4-argument spelling normalises to m = 1, which is the identity mask.
    h *= m;
    s = (s - 1) * m + 1;
    i = (i - 1) * m + 1;
    ExprHsiAdjust(rgb, h, s, i, out);
}
USDGEN_EXPR_HD inline void ExprMidHsi(const double *rgb, double h, double s, double i,
                                      double m, double falloff, double interp,
                                      double *out)
{
    m = m * 2 - 1;
    if (m < 0) m = -ExprRemap(-m, 1, 0, falloff, interp);
    else       m = ExprRemap(m, 1, 0, falloff, interp);
    h *= m;
    const double absm = double(static_cast<float>(USDGEN_EXPR_FN(fabs)(m)));
    s = s * absm + 1 - absm;
    i = i * absm + 1 - absm;
    if (m < 0) { s = 1 / s; i = 1 / i; }
    ExprHsiAdjust(rgb, h, s, i, out);
}

// ---------------------------------------------------------------------------
// Noise (thirdparty/seexpr/SeExpr2/Noise.cpp, the double instantiation)
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline double ExprSCurve(double t)
{
    return t * t * t * (t * (6 * t - 15) + 10);
}

USDGEN_EXPR_HD inline unsigned char ExprHashReduceChar3(const int index[3])
{
    uint32_t seed = 0;
    for (int k = 0; k < 3; k++) {
        const uint32_t M = 1664525u, C = 1013904223u;
        seed = seed * M + static_cast<uint32_t>(index[k]) + C;
    }
    seed ^= (seed >> 11);
    seed ^= (seed << 7) & 0x9d2c5680u;
    seed ^= (seed << 15) & 0xefc60000u;
    seed ^= (seed >> 18);
    return static_cast<unsigned char>((((seed & 0xff0000u) >> 4) + (seed & 0xffu)) & 0xffu);
}

USDGEN_EXPR_HD inline uint32_t ExprHashReduce3(const uint32_t index[3])
{
    uint32_t i = 0;
    for (int k = 0; k < 3; k++) {
        const uint32_t M = 1664525u, C = 1013904223u;
        i = i * M + index[k] + C;
    }
    i ^= (i >> 11);
    i ^= (i << 7) & 0x9d2c5680u;
    i ^= (i << 15) & 0xefc60000u;
    i ^= (i >> 18);
    // The union-of-bytes permutation, written out so endianness is explicit:
    // SeExpr2 reads c[0] as the low byte, which is what every platform usdGen
    // builds for does.
    const unsigned char c0 = static_cast<unsigned char>(i & 0xffu);
    const unsigned char c1 = static_cast<unsigned char>((i >> 8) & 0xffu);
    const unsigned char c2 = static_cast<unsigned char>((i >> 16) & 0xffu);
    const unsigned char c3 = static_cast<unsigned char>((i >> 24) & 0xffu);
    // p is the 514-entry table; SeExpr2 indexes it with an unsigned char plus
    // another unsigned char, which is why it is longer than 256 entries.
    const unsigned char o3 = static_cast<unsigned char>(NoiseP514(c0));
    const unsigned char o2 = static_cast<unsigned char>(NoiseP514(int(c1) + int(o3)));
    const unsigned char o1 = static_cast<unsigned char>(NoiseP514(int(c2) + int(o2)));
    const unsigned char o0 = static_cast<unsigned char>(NoiseP514(int(c3) + int(o1)));
    return (uint32_t(o3) << 24) | (uint32_t(o2) << 16) | (uint32_t(o1) << 8) | uint32_t(o0);
}

/// noiseHelper<3, double, periodic>.
USDGEN_EXPR_HD inline double ExprNoiseHelper3(const double *X, const int *period)
{
    double weights[2][3];
    int index[3];
    for (int k = 0; k < 3; k++) {
        const double f = USDGEN_EXPR_FN(floor)(X[k]);
        index[k] = ExprInt(f);
        if (period) {
            index[k] %= period[k];
            if (index[k] < 0) index[k] += period[k];
        }
        weights[0][k] = X[k] - f;
        weights[1][k] = weights[0][k] - 1;
    }
    double vals[8];
    for (int dummy = 0; dummy < 8; dummy++) {
        int latticeIndex[3];
        int offset[3];
        for (int k = 0; k < 3; k++) {
            offset[k] = ((dummy & (1 << k)) != 0);
            latticeIndex[k] = index[k] + offset[k];
        }
        const int lookup = ExprHashReduceChar3(latticeIndex);
        double val = 0;
        for (int k = 0; k < 3; k++) val += NoiseG3(lookup, k) * weights[offset[k]][k];
        vals[dummy] = val;
    }
    double alphas[3];
    for (int k = 0; k < 3; k++) alphas[k] = ExprSCurve(weights[0][k]);
    for (int newd = 2; newd >= 0; newd--) {
        const int newnum = 1 << newd;
        const int k = (3 - newd - 1);
        const double alpha = alphas[k];
        const double beta = 1.0 - alphas[k];
        for (int dummy = 0; dummy < newnum; dummy++) {
            const int i = dummy * (1 << (3 - newd));
            const int otherIndex = i + (1 << k);
            vals[i] = beta * vals[i] + alpha * vals[otherIndex];
        }
    }
    return vals[0];
}

/// Noise<3, dOut, double>. dOut is 1 or 3.
USDGEN_EXPR_HD inline void ExprNoise3(const double *in, int dOut, double *out)
{
    double P[3] = {in[0], in[1], in[2]};
    int i = 0;
    while (1) {
        out[i] = ExprNoiseHelper3(P, nullptr);
        if (++i >= dOut) break;
        for (int k = 0; k < dOut; k++) P[k] += 1000.0;
    }
}

/// PNoise<3, 1, double>. SeExpr2 takes std::max(1, (int)period[k]).
USDGEN_EXPR_HD inline double ExprPNoise3(const double *in, const double *period)
{
    int p[3];
    for (int k = 0; k < 3; ++k) {
        const int truncated = ExprInt(period[k]);
        p[k] = truncated < 1 ? 1 : truncated;
    }
    const double P[3] = {in[0], in[1], in[2]};
    return ExprNoiseHelper3(P, p);
}

/// CellNoise<3, dOut, double>.
USDGEN_EXPR_HD inline void ExprCellNoise3(const double *in, int dOut, double *out)
{
    uint32_t index[3];
    for (int k = 0; k < 3; k++) index[k] = ExprUInt32Wrap(USDGEN_EXPR_FN(floor)(in[k]));
    int dim = 0;
    while (1) {
        out[dim] = ExprHashReduce3(index) * (1.0 / 0xffffffffu);
        if (++dim >= dOut) break;
        for (int k = 0; k < 3; k++) index[k] += 1000u;
    }
}

/// FBM<3, dOut, turbulence, double>.
USDGEN_EXPR_HD inline void ExprFbm3(const double *in, int dOut, bool turbulence,
                                    int octaves, double lacunarity, double gain,
                                    double *out)
{
    double P[3] = {in[0], in[1], in[2]};
    double scale = 1;
    for (int k = 0; k < dOut; k++) out[k] = 0;
    int octave = 0;
    while (1) {
        double localResult[3];
        ExprNoise3(P, dOut, localResult);
        for (int k = 0; k < dOut; k++)
            out[k] += (turbulence ? USDGEN_EXPR_FN(fabs)(localResult[k]) : localResult[k]) * scale;
        if (++octave >= octaves) break;
        scale *= gain;
        for (int k = 0; k < 3; k++) { P[k] *= lacunarity; P[k] += 1234.0; }
    }
}

/// hash(n, args) — SeExpr2's standalone hash, not the noise lattice hash.
USDGEN_EXPR_HD inline double ExprHash(const double *args, int n)
{
    uint32_t seed = 0;
    for (int i = 0; i < n; i++) {
        int exponent = 0;
        const double frac = USDGEN_EXPR_FN(frexp)(args[i] * (kExprE * kExprPi), &exponent);
        const uint32_t s = ExprUInt32Wrap(frac * 4294967295.0) ^ static_cast<uint32_t>(exponent);
        const uint32_t M = 1664525u, C = 1013904223u;
        seed = seed * M + s + C;
    }
    seed ^= (seed >> 11);
    seed ^= (seed << 7) & 0x9d2c5680u;
    seed ^= (seed << 15) & 0xefc60000u;
    seed ^= (seed >> 18);
    const unsigned char c0 = static_cast<unsigned char>(seed & 0xffu);
    const unsigned char c1 = static_cast<unsigned char>((seed >> 8) & 0xffu);
    const unsigned char c2 = static_cast<unsigned char>((seed >> 16) & 0xffu);
    const unsigned char c3 = static_cast<unsigned char>((seed >> 24) & 0xffu);
    const unsigned char o3 = HashP256(c0);
    const unsigned char o2 = HashP256((int(c1) + int(o3)) & 0xff);
    const unsigned char o1 = HashP256((int(c2) + int(o2)) & 0xff);
    const unsigned char o0 = HashP256((int(c3) + int(o1)) & 0xff);
    const uint32_t out = (uint32_t(o3) << 24) | (uint32_t(o2) << 16) |
                         (uint32_t(o1) << 8) | uint32_t(o0);
    return out * (1.0 / 4294967295.0);
}

// ---------------------------------------------------------------------------
// Voronoi. SeExpr2 caches the 27 cell points per node; being stateless we
// recompute them, which is the same arithmetic in the same order.
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline void ExprVoronoiPoints(const double *cell, double jitter,
                                             double points[27][3])
{
    int n = 0;
    for (double i = -1; i <= 1; i++)
        for (double j = -1; j <= 1; j++)
            for (double k = -1; k <= 1; k++, n++) {
                const double testcell[3] = {cell[0] + i, cell[1] + j, cell[2] + k};
                double cc[3];
                ExprCellNoise3(testcell, 3, cc);
                for (int c = 0; c < 3; ++c)
                    points[n][c] = testcell[c] + jitter * (cc[c] - .5);
            }
}

USDGEN_EXPR_HD inline void ExprVoronoiF1(const double *p, double jitter,
                                         double *f1, double *pos1)
{
    const double thiscell[3] = {USDGEN_EXPR_FN(floor)(p[0]) + 0.5,
                                USDGEN_EXPR_FN(floor)(p[1]) + 0.5,
                                USDGEN_EXPR_FN(floor)(p[2]) + 0.5};
    double points[27][3];
    ExprVoronoiPoints(thiscell, jitter, points);
    double best = 1000;
    for (int i = 0; i < 27; ++i) {
        const double offset[3] = {points[i][0] - p[0], points[i][1] - p[1], points[i][2] - p[2]};
        const double d = ExprDot(offset, offset);
        if (d < best) {
            best = d;
            pos1[0] = points[i][0]; pos1[1] = points[i][1]; pos1[2] = points[i][2];
        }
    }
    *f1 = USDGEN_EXPR_FN(sqrt)(best);
}

USDGEN_EXPR_HD inline void ExprVoronoiF1F2(const double *p, double jitter,
                                           double *f1, double *pos1,
                                           double *f2, double *pos2)
{
    const double thiscell[3] = {USDGEN_EXPR_FN(floor)(p[0]) + 0.5,
                                USDGEN_EXPR_FN(floor)(p[1]) + 0.5,
                                USDGEN_EXPR_FN(floor)(p[2]) + 0.5};
    double points[27][3];
    ExprVoronoiPoints(thiscell, jitter, points);
    double b1 = 1000, b2 = 1000;
    for (int i = 0; i < 27; ++i) {
        const double offset[3] = {points[i][0] - p[0], points[i][1] - p[1], points[i][2] - p[2]};
        const double d = ExprDot(offset, offset);
        if (d < b1) {
            b2 = b1;
            pos2[0] = pos1[0]; pos2[1] = pos1[1]; pos2[2] = pos1[2];
            b1 = d;
            pos1[0] = points[i][0]; pos1[1] = points[i][1]; pos1[2] = points[i][2];
        } else if (d < b2) {
            b2 = d;
            pos2[0] = points[i][0]; pos2[1] = points[i][1]; pos2[2] = points[i][2];
        }
    }
    *f1 = USDGEN_EXPR_FN(sqrt)(b1);
    *f2 = USDGEN_EXPR_FN(sqrt)(b2);
}

/// The fbm warp every voronoi spelling shares.
USDGEN_EXPR_HD inline void ExprVoronoiWarp(const double *p, double fbmScale,
                                           double fbmOctaves, double fbmLacunarity,
                                           double fbmGain, double *out)
{
    out[0] = p[0]; out[1] = p[1]; out[2] = p[2];
    if (!(fbmScale > 0)) return;
    const double doubled[3] = {2 * p[0], 2 * p[1], 2 * p[2]};
    double warp[3];
    ExprFbm3(doubled, 3, false, ExprInt(ExprClamp(fbmOctaves, 1, 8)),
             fbmLacunarity, fbmGain, warp);
    for (int i = 0; i < 3; ++i) out[i] += fbmScale * warp[i];
}

/// voronoi(v, type, jitter, fbmScale, fbmOctaves, fbmLacunarity, fbmGain).
USDGEN_EXPR_HD inline double ExprVoronoi(const double *v, double typeArg, double jitterArg,
                                         double fbmScale, double fbmOctaves,
                                         double fbmLacunarity, double fbmGain)
{
    const int type = ExprInt(typeArg);
    const double jitter = ExprClamp(jitterArg, 1e-3, 1);
    double p[3];
    ExprVoronoiWarp(v, fbmScale, fbmOctaves, fbmLacunarity, fbmGain, p);
    double f1 = 0, f2 = 0, pos1[3] = {0, 0, 0}, pos2[3] = {0, 0, 0};
    if (type >= 3) ExprVoronoiF1F2(p, jitter, &f1, pos1, &f2, pos2);
    else           ExprVoronoiF1(p, jitter, &f1, pos1);
    switch (type) {
    case 1: {
        const double shifted[3] = {pos1[0] + 10, pos1[1], pos1[2]};
        double cell = 0;
        ExprCellNoise3(shifted, 1, &cell);
        return cell;
    }
    case 2: return f1;
    case 3: return f2;
    case 4: return f2 - f1;
    case 5: {
        const double d21[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
        const double d1p[3] = {pos1[0] - p[0], pos1[1] - p[1], pos1[2] - p[2]};
        const double d2p[3] = {pos2[0] - p[0], pos2[1] - p[1], pos2[2] - p[2]};
        const float scalefactor =
            static_cast<float>(ExprLength3(d21) / (ExprLength3(d1p) + ExprLength3(d2p)));
        return ExprSmoothStep(f2 - f1, 0, 0.1 * scalefactor);
    }
    default: return 0.0;
    }
}

/// cvoronoi(...) — the same walk, coloured by ccellnoise of the winning cell.
USDGEN_EXPR_HD inline void ExprCVoronoi(const double *v, double typeArg, double jitterArg,
                                        double fbmScale, double fbmOctaves,
                                        double fbmLacunarity, double fbmGain, double *out)
{
    const int type = ExprInt(typeArg);
    const double jitter = ExprClamp(jitterArg, 1e-3, 1);
    double p[3];
    ExprVoronoiWarp(v, fbmScale, fbmOctaves, fbmLacunarity, fbmGain, p);
    double f1 = 0, f2 = 0, pos1[3] = {0, 0, 0}, pos2[3] = {0, 0, 0};
    if (type >= 3) ExprVoronoiF1F2(p, jitter, &f1, pos1, &f2, pos2);
    else           ExprVoronoiF1(p, jitter, &f1, pos1);
    double color[3];
    ExprCellNoise3(pos1, 3, color);
    switch (type) {
    case 1: out[0] = color[0]; out[1] = color[1]; out[2] = color[2]; return;
    case 2: for (int i = 0; i < 3; ++i) out[i] = f1 * color[i]; return;
    case 3: for (int i = 0; i < 3; ++i) out[i] = f2 * color[i]; return;
    case 4: for (int i = 0; i < 3; ++i) out[i] = (f2 - f1) * color[i]; return;
    case 5: {
        const double d21[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
        const double d1p[3] = {pos1[0] - p[0], pos1[1] - p[1], pos1[2] - p[2]};
        const double d2p[3] = {pos2[0] - p[0], pos2[1] - p[1], pos2[2] - p[2]};
        const float scalefactor =
            static_cast<float>(ExprLength3(d21) / (ExprLength3(d1p) + ExprLength3(d2p)));
        const double blend = ExprSmoothStep(f2 - f1, 0, 0.1 * scalefactor);
        for (int i = 0; i < 3; ++i) out[i] = blend * color[i];
        return;
    }
    default: out[0] = out[1] = out[2] = 0; return;
    }
}

/// pvoronoi(v, jitter, fbmScale, fbmOctaves, fbmLacunarity, fbmGain) — the
/// centre of the winning cell.
USDGEN_EXPR_HD inline void ExprPVoronoi(const double *v, double jitterArg, double fbmScale,
                                        double fbmOctaves, double fbmLacunarity,
                                        double fbmGain, double *out)
{
    const double jitter = ExprClamp(jitterArg, 1e-3, 1);
    double p[3];
    ExprVoronoiWarp(v, fbmScale, fbmOctaves, fbmLacunarity, fbmGain, p);
    double f1 = 0;
    ExprVoronoiF1(p, jitter, &f1, out);
}

// ---------------------------------------------------------------------------
// Variations over a contiguous argument block.
// ---------------------------------------------------------------------------

/// spline(param, y1, y2, ...). `args` is the whole block, `n` its length.
USDGEN_EXPR_HD inline double ExprSpline(const double *args, int n)
{
    if (n < 5) return 0;
    double u = ExprClamp(args[0], 0, 1);
    if (u == 0) return args[2];
    if (u == 1) return args[n - 2];
    const int nsegs = n - 4;
    double seg = 0;
    u = USDGEN_EXPR_FN(modf)(u * nsegs, &seg);
    const double *p = &args[ExprInt(seg) + 1];
    const double u2 = u * u, u3 = u2 * u;
    return 0.5 * (p[0] * (-u3 + 2 * u2 - u) + p[1] * (3 * u3 - 5 * u2 + 2) +
                  p[2] * (-3 * u3 + 4 * u2 + u) + p[3] * (u3 - u2));
}

USDGEN_EXPR_HD inline double ExprChoose(const double *args, int n)
{
    if (n < 3) return 0;
    const double key = args[0];
    if (key != key) return 0;   // NaN protection, as SeExpr2
    const int nvals = n - 1;
    return args[1 + ExprInt(ExprClamp(key * nvals, 0, nvals - 1))];
}

USDGEN_EXPR_HD inline double ExprWChoose(const double *args, int n)
{
    if (n < 5) return 0;
    double key = args[0];
    if (key != key) return 0;
    const int nvals = (n - 1) / 2;
    double total = 0;
    for (int i = 0; i < nvals; i++) total += args[i * 2 + 2];
    if (total == 0) return args[1];
    key *= total;
    // SeExpr2 binary-searches a materialised cutoff table; the same table read
    // as a running prefix sum gives the same index without an allocation.
    auto cutoff = [&](int i) {
        double sum = 0;
        for (int k = 0; k <= i; ++k) sum += args[k * 2 + 2];
        return sum;
    };
    int lo = 0, hi = nvals - 1;
    while (lo < hi) {
        const int m = (lo + hi) / 2;
        if (key <= cutoff(m)) hi = m; else lo = m + 1;
    }
    if (args[lo * 2 + 2] == 0) {
        if (lo > 0 && cutoff(lo) > 0) while (--lo > 0 && args[lo * 2 + 2] == 0) {}
        else if (lo < nvals - 1) while (++lo < nvals - 1 && args[lo * 2 + 2] == 0) {}
    }
    return args[lo * 2 + 1];
}

/// pick(index, loRange, hiRange, [weights...]). The range is walked rather
/// than materialised; a range wider than kExprPickRange is refused (returns
/// loRange) so a runaway expression cannot hang a kernel.
constexpr int kExprPickRange = 4096;
USDGEN_EXPR_HD inline double ExprPick(const double *args, int n)
{
    if (n < 3) return 0;
    double index = ExprHash(&args[0], 1);
    const int loRange = ExprInt(args[1]);
    const int hiRange = ExprInt(args[2]);
    const int range = hiRange - loRange + 1;
    if (range <= 0 || range > kExprPickRange) return loRange;
    int numWeights = n - 3;
    if (numWeights > range) numWeights = range;
    auto weight = [&](int i) { return i < numWeights ? args[i + 3] : 1.0; };
    auto cutoff = [&](int i) {
        double sum = 0;
        for (int k = 0; k <= i; ++k) sum += weight(k);
        return sum;
    };
    double total = 0;
    for (int i = 0; i < range; i++) total += weight(i);
    if (total == 0) return loRange;
    index *= total;
    int lo = 0, hi = range - 1;
    while (lo < hi) {
        const int m = (lo + hi) / 2;
        if (index <= cutoff(m)) hi = m; else lo = m + 1;
    }
    if (weight(lo) == 0) {
        if (lo > 0 && cutoff(lo) > 0) while (--lo > 0 && weight(lo) == 0) {}
        else if (lo < range - 1) while (++lo < range - 1 && weight(lo) == 0) {}
    }
    return loRange + lo;
}

// ---------------------------------------------------------------------------
// curve()/ccurve(). The frontend prepares the knots (sorts them, adds SeExpr's
// two sentinels, computes the centred-difference derivatives and applies the
// monotone clamp) at COMPILE time and lays the result out as:
//
//   block[0]                 param
//   block[1]                 cv count, as a double
//   block[2 + S*i + 0]       pos
//   block[2 + S*i + 1]       interpolation code (0 none, 1 linear, 2 smooth,
//                            3 spline, 4 monotone spline)
//   block[2 + S*i + 2 ..]    value    (1 channel for curve, 3 for ccurve)
//   block[2 + S*i + 2 + k..] derivative, same channel count
//
// with stride S = 4 for curve and 8 for ccurve. Editing a curve control in the
// UI therefore only rewrites immediates in the instruction stream.
// ---------------------------------------------------------------------------
USDGEN_EXPR_HD inline double ExprCurveEval(const double *block, int n, int stride, int channel)
{
    if (stride < 4 || n < 2 + 2 * stride) return 0;
    const int channels = (stride - 2) / 2;
    if (channel < 0 || channel >= channels) return 0;
    const double param = block[0];
    const int count = ExprInt(block[1]);
    if (count < 2 || 2 + count * stride > n) return 0;
    auto pos = [&](int i) { return block[2 + i * stride + 0]; };
    auto interp = [&](int i) { return ExprInt(block[2 + i * stride + 1]); };
    auto val = [&](int i) { return block[2 + i * stride + 2 + channel]; };
    auto deriv = [&](int i) { return block[2 + i * stride + 2 + channels + channel]; };
    // std::upper_bound over _pos: the first cv whose position is > param.
    int index = count;
    for (int i = 0; i < count; ++i)
        if (param < pos(i)) { index = i; break; }
    if (index < 1) index = 1;
    if (index > count - 1) index = count - 1;
    const float t0 = static_cast<float>(pos(index - 1));
    const float t1 = static_cast<float>(pos(index));
    const double k0 = val(index - 1);
    const double k1 = val(index);
    switch (interp(index - 1)) {
    case 0: return k0;
    case 1: {
        const double u = (param - t0) / (t1 - t0);
        return k0 + u * (k1 - k0);
    }
    case 2: {
        const double u = (param - t0) / (t1 - t0);
        return k0 * (u - 1) * (u - 1) * (2 * u + 1) + k1 * u * u * (3 - 2 * u);
    }
    case 3:
    case 4: {
        const double x = param - pos(index - 1);
        const double h = pos(index) - pos(index - 1);
        const double y = k0;
        const double delta = k1 - k0;
        const double d1 = deriv(index - 1);
        const double d2 = deriv(index);
        return (x * (delta * (3 * h - 2 * x) * x + h * (-h + x) * (-(d1 * h) + (d1 + d2) * x))) /
                   (h * h * h) + y;
    }
    default: return 0;
    }
}

} // namespace usdGen::expr
#endif
