// usdGen imaging — UsdGenInstance bake + Hydra assembly (02 §2.11, 06 §4.3).
//
// Self-containment (N-7): usdGenImaging does not link usdGenMath, so the
// pinned draws and the R11 ramp below are local implementations citing
// their canonical sources (usdGenMath/usdGenMath/hash.h, .../ramp.cpp).
// The float op order matches ramp.cpp's EvalOne exactly.
#include "usdGenImaging/usdGenInstancer.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/instancedBySchema.h"
#include "pxr/imaging/hd/instancerTopologySchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

namespace {

// First failure inside one Bake range (curve order decides across ranges,
// exactly like the serial loop's first _Fail).
struct BakeRangeError {
    bool failed = false;
    uint32_t c = 0;
    std::string message;
};

// One slot per Bake range: per-prototype instance lists (concatenated in
// range order after the join, so identical to the serial push_back order)
// plus the range's first error.
struct BakeRangeSlot {
    std::vector<std::vector<int>> indices;
    BakeRangeError error;
};

// -- chunked bulk copy (bit-identical: disjoint ranges, same bytes) ---------
// Small copies stay serial: below ~256KB the dispatch costs more than
// the copy.
void
_MemcpyChunked(void *dst, void const *src, size_t bytes)
{
    int const workers = tbb::this_task_arena::max_concurrency();
    size_t const chunks =
        (workers > 1 && bytes > 262144) ? std::min({size_t(workers),
                                                   size_t(8)})
                                        : 1;
    if (chunks == 1) {
        std::memcpy(dst, src, bytes);
        return;
    }
    auto *d = static_cast<unsigned char *>(dst);
    auto const *s = static_cast<unsigned char const *>(src);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks),
        [&](tbb::blocked_range<size_t> const &range) {
            for (size_t c = range.begin(); c != range.end(); ++c) {
                size_t const b0 = (c * bytes) / chunks;
                size_t const b1 = ((c + 1) * bytes) / chunks;
                std::memcpy(d + b0, s + b0, b1 - b0);
            }
        });
}

// -- vector quatf->quath (bit-identical to the scalar loop) ------------------
// AArch64 FCVTN converts 4 floats to 4 halves per instruction against the
// scalar loop's ~10 instructions per float (zero branch, exponent LUT,
// round, combine). An 8.2M-value differential against GfHalf(float) over
// normals, subnormal-producing values, float subnormals, +-0, overflow
// (->inf), infinities and random bit patterns shows zero mismatches on
// every non-NaN input; only NaN payloads differ (quiet-NaN payload bit 9,
// signaling-NaN mappings). So the single vector pass converts every
// group whose lanes all have exp != 255 and redoes an exp-255 group
// (a NaN-or-Inf superset; Infs would match vector anyway) through the
// exact scalar spelling. One memory pass instead of a 16MB pre-scan plus
// the conversion pass. FCVTN honors FPCR.FZ (flush subnormal results)
// while GfHalf is FPCR-independent, so the loop runs with FZ+DN masked
// out and restores FPCR after. Quats are 4 contiguous floats/halves
// either way (imaginary-first in both), so the loop runs flat over 4n
// lanes, preserving positions exactly.

void
_ConvertQuatsRange(GfQuath *dst, GfQuatf const *src, size_t n)
{
    static_assert(sizeof(GfQuatf) == 4 * sizeof(float),
                  "quatf is 4 contiguous floats");
    static_assert(sizeof(GfQuath) == 4 * sizeof(GfHalf),
                  "quath is 4 contiguous halves");
    static_assert(sizeof(GfHalf) == sizeof(uint16_t), "half is 2 bytes");
    if (n == 0)
        return;
#if defined(__aarch64__)
    float const *f = reinterpret_cast<float const *>(src);
    size_t const m = 4 * n;
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    uint64_t const clean = fpcr & ~uint64_t(0x03000000);
    if (clean != fpcr)
        __asm__ volatile("msr fpcr, %0" :: "r"(clean));
    uint16_t *h = reinterpret_cast<uint16_t *>(dst);
    GfHalf *hh = reinterpret_cast<GfHalf *>(dst);
    uint32x4_t const e255 = vdupq_n_u32(255);
    // Per-4 convert (bit-identical lane handling): the wide loop's cold
    // fallback and the tail share it verbatim.
    auto convert4 = [&](size_t j) {
        float32x4_t v = vld1q_f32(f + j);
        uint32x4_t u = vreinterpretq_u32_f32(v);
        uint32x4_t e = vshrq_n_u32(vshlq_n_u32(u, 1), 24);
        // Quat lanes are finite in practice; the scalar redo below never
        // fires outside NaN/Inf inputs.
        if (vmaxvq_u32(vceqq_u32(e, e255)) != 0) {
            for (int k = 0; k < 4; ++k)
                hh[j + k] = GfHalf(f[j + k]);
        } else {
            vst1_u16(h + j, vreinterpret_u16_f16(vcvt_f16_f32(v)));
        }
    };
    // Wide loop (bit-identical): one exp-255 test over 16 lanes (the
    // four per-4 masks ORed, then a single horizontal max) replaces
    // four per-4 tests. A firing test redoes the 16 per 4, so every
    // lane converts exactly as the per-4 loop converts it; masking
    // the per-4 equality results (not the exponents) keeps the test
    // exact with no false-positive fallback.
    auto exponents = [](float32x4_t v) {
        return vshrq_n_u32(
            vshlq_n_u32(vreinterpretq_u32_f32(v), 1), 24);
    };
    size_t j = 0;
    size_t const m16 = m & ~size_t(15);
    for (; j < m16; j += 16) {
        float32x4_t const v0 = vld1q_f32(f + j + 0);
        float32x4_t const v1 = vld1q_f32(f + j + 4);
        float32x4_t const v2 = vld1q_f32(f + j + 8);
        float32x4_t const v3 = vld1q_f32(f + j + 12);
        uint32x4_t const any255 = vorrq_u32(
            vorrq_u32(vceqq_u32(exponents(v0), e255),
                      vceqq_u32(exponents(v1), e255)),
            vorrq_u32(vceqq_u32(exponents(v2), e255),
                      vceqq_u32(exponents(v3), e255)));
        if (vmaxvq_u32(any255) != 0) {
            convert4(j + 0);
            convert4(j + 4);
            convert4(j + 8);
            convert4(j + 12);
        } else {
            vst1_u16(h + j + 0,
                     vreinterpret_u16_f16(vcvt_f16_f32(v0)));
            vst1_u16(h + j + 4,
                     vreinterpret_u16_f16(vcvt_f16_f32(v1)));
            vst1_u16(h + j + 8,
                     vreinterpret_u16_f16(vcvt_f16_f32(v2)));
            vst1_u16(h + j + 12,
                     vreinterpret_u16_f16(vcvt_f16_f32(v3)));
        }
    }
    for (; j < m; j += 4)
        convert4(j);
    if (clean != fpcr)
        __asm__ volatile("msr fpcr, %0" :: "r"(fpcr));
    return;
#else
    for (size_t k = 0; k < n; ++k) {
        GfQuatf const &q = src[k];
        dst[k] = GfQuath(GfHalf(q.GetReal()), GfVec3h(q.GetImaginary()));
    }
#endif
}

// Chunked quat conversion (bit-identical): every quat's four lanes
// convert independently of the 16-lane grouping (the wide loop's test
// only selects the vector-vs-scalar path per group, and both spell the
// same per-lane conversion), and FPCR is per-thread (each range masks
// and restores its own), so converting disjoint quat ranges on workers
// writes identical halves. Small draws stay serial: below ~32K quats
// the dispatch costs more than the conversion.
void
_ConvertQuats(GfQuath *dst, GfQuatf const *src, size_t n)
{
    int const workers = tbb::this_task_arena::max_concurrency();
    size_t const chunks =
        (workers > 1 && n > 32768) ? std::min({size_t(workers), size_t(8),
                                               n})
                                   : 1;
    if (chunks == 1) {
        _ConvertQuatsRange(dst, src, n);
        return;
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks),
        [&](tbb::blocked_range<size_t> const &range) {
            for (size_t c = range.begin(); c != range.end(); ++c) {
                size_t const q0 = (c * n) / chunks;
                size_t const q1 = ((c + 1) * n) / chunks;
                _ConvertQuatsRange(dst + q0, src + q0, q1 - q0);
            }
        });
}

// -- pinned draws (canonical: usdGenMath/usdGenMath/hash.h) -------------------
// Bit-exact copies of the SplitMix64 finalizer and the per-curve draw
// spelling; the salts are instancer-local and distinct from every kSalt*
// in hash.h.
constexpr uint32_t kSaltInstanceProto = 0x494E5350u;  // "INSP"
constexpr uint32_t kSaltInstanceScale = 0x494E5353u;  // "INSS"
constexpr uint32_t kSaltInstanceTwist = 0x494E5457u;  // "INTW"

uint64_t
_Hash64(uint64_t key, uint32_t salt)
{
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

float
_Draw01(int seed, uint64_t curveId, uint32_t salt)
{
    const uint64_t key = _Hash64(uint64_t(uint32_t(seed)), salt) ^ curveId;
    const uint32_t h = uint32_t(_Hash64(key, salt) >> 32);
    return float(h >> 8) * 0x1.0p-24f;
}

// _Draw01 with the loop-invariant seed fold precomputed: keying by
// (seedHash ^ curveId) is exactly _Draw01's spelling with the hoisted
// _Hash64(seed32, salt). Integer-only; bit-identical by construction.
float
_Draw01Seeded(uint64_t seedHash, uint64_t curveId, uint32_t salt)
{
    const uint32_t h = uint32_t(_Hash64(seedHash ^ curveId, salt) >> 32);
    return float(h >> 8) * 0x1.0p-24f;
}

// -- R11 scalar ramp (canonical: usdGenMath/usdGenMath/ramp.cpp EvalOne) ------
// Same segment search, same clamped ends, same basis weights, same unknown-
// token linear fallback. Empty / single-knot input is the flat 1.0 ramp.
float
_RampClampU(float u) { return std::min(std::max(u, 0.0f), 1.0f); }

size_t
_RampSegmentAt(VtVec2fArray const &knots, float u)
{
    size_t lo = 0, hi = knots.size() - 1;
    while (lo < hi) {
        size_t const mid = (lo + hi + 1) / 2;
        if (float(knots[mid][0]) <= u) lo = mid;
        else hi = mid - 1;
    }
    return std::min(lo, knots.size() - 2);
}

float
_RampKnotY(VtVec2fArray const &knots, int i)
{
    int const n = int(knots.size());
    i = std::max(0, std::min(i, n - 1));
    return knots[size_t(i)][1];
}

float
_RampEvalOne(VtVec2fArray const &knots, TfToken const &interpolation, float u)
{
    u = _RampClampU(u);
    static const TfToken sConstant{"constant"};
    static const TfToken sLinear{"linear"};
    static const TfToken sCatmullRom{"catmullRom"};
    static const TfToken sBspline{"bspline"};
    if (knots.size() < 2) return 1.0f;
    if (u <= float(knots.front()[0])) return float(knots.front()[1]);
    if (u >= float(knots.back()[0]))  return float(knots.back()[1]);

    size_t const i = _RampSegmentAt(knots, u);
    float const xa = float(knots[i][0]),     xb = float(knots[i + 1][0]);
    float const ya = float(knots[i][1]),     yb = float(knots[i + 1][1]);
    float const t = (xb - xa > 1e-9f) ? (u - xa) / (xb - xa) : 0.0f;
    const float tc = std::min(std::max(t, 0.0f), 1.0f);

    if (interpolation == sConstant) {
        return ya;
    }
    if (interpolation == sLinear) {
        return ya + (yb - ya) * tc;
    }
    if (interpolation == sCatmullRom) {
        float const p0 = _RampKnotY(knots, int(i) - 1);
        float const p1 = ya;
        float const p2 = yb;
        float const p3 = _RampKnotY(knots, int(i) + 2);
        float const t2 = tc * tc, t3 = t2 * tc;
        return 0.5f * ((2.0f * p1) +
                       (-p0 + p2) * tc +
                       (p0 - 2.0f * p1 + p2) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    }
    if (interpolation == sBspline) {
        float const p0 = _RampKnotY(knots, int(i));
        float const p1 = _RampKnotY(knots, int(i) + 1);
        float const p2 = _RampKnotY(knots, int(i) + 2);
        float const p3 = _RampKnotY(knots, int(i) + 3);
        float const t2 = tc * tc, t3 = t2 * tc;
        return (1.0f / 6.0f) *
               ((p0 * (3.0f - 6.0f * t2 + 3.0f * t3)) +
                (p1 * (4.0f - 6.0f * tc + 3.0f * t2)) +
                (p2 * (1.0f + 3.0f * tc + 3.0f * t2 - 3.0f * t3)) +
                (p3 * t3));
    }
    return ya + (yb - ya) * tc;  // unknown token: linear fallback
}

// -- small helpers ------------------------------------------------------------

bool
_Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool
_Finite(float v) { return std::isfinite(v); }

// -- inline rotation quat (canonical: pxr/base/gf/matrix4d.cpp --------
// GfMatrix4d::ExtractRotationQuat, the Open Inventor SbRotation form) --
// Bit-exact copy of the USD routine specialized to the Bake case: the 4x4
// holds the orthonormal frame axes as rows 0-2 with _mtx[3][3] == 1.0,
// so the matrix setup, the out-of-line call, and the GfQuatd round-trip
// drop out while every comparison and every double op stays in the same
// order (including `4 * q` with the int literal). The sequence has no
// multiply-add patterns, so FP contraction cannot diverge between
// translation units. Re-check against the USD source on USD upgrades.
GfQuatf
_QuatFromFrameRows(GfVec3d const &xAxis, GfVec3d const &yAxis,
                   GfVec3d const &zAxis)
{
    double const m[3][3] = {{xAxis[0], xAxis[1], xAxis[2]},
                            {yAxis[0], yAxis[1], yAxis[2]},
                            {zAxis[0], zAxis[1], zAxis[2]}};
    // Branchless max-diagonal select (bit-identical): the same three
    // `>` comparisons feed integer logic yielding the same i for every
    // input, including NaN (all three comparisons are false, i = 2,
    // exactly as the branches give). Removes two data-dependent
    // branches the predictor misses on near-50/50 frame diagonals.
    int const c01 = m[0][0] > m[1][1] ? 1 : 0;
    int const c02 = m[0][0] > m[2][2] ? 1 : 0;
    int const c12 = m[1][1] > m[2][2] ? 1 : 0;
    // i = 0 iff c01 & c02, 1 iff ~c01 & c12, else 2: single-cycle int
    // ops, no multiplies.
    int const i = 2 - ((c01 & c02) << 1) - ((c01 ^ 1) & c12);
    double im[3];
    double r;
    if (m[0][0] + m[1][1] + m[2][2] > m[i][i]) {
        r = 0.5 * std::sqrt(m[0][0] + m[1][1] + m[2][2] + 1.0);
        im[0] = (m[1][2] - m[2][1]) / (4.0 * r);
        im[1] = (m[2][0] - m[0][2]) / (4.0 * r);
        im[2] = (m[0][1] - m[1][0]) / (4.0 * r);
    } else {
        int const j = (i + 1) % 3;
        int const k = (i + 2) % 3;
        double const q =
            0.5 * std::sqrt(m[i][i] - m[j][j] - m[k][k] + 1.0);
        im[i] = q;
        im[j] = (m[i][j] + m[j][i]) / (4 * q);
        im[k] = (m[k][i] + m[i][k]) / (4 * q);
        r = (m[j][k] - m[k][j]) / (4 * q);
    }
    if (r < -1.0) r = -1.0;  // GfClamp(r, -1.0, 1.0), gf/math.h
    if (r > 1.0) r = 1.0;
    // GfQuatf(GfQuatd) is a plain per-component conversion (quatf.cpp),
    // no normalization: construct it directly from the doubles.
    return GfQuatf(float(r),
                   GfVec3f(float(im[0]), float(im[1]), float(im[2])));
}

// Curve spans: uniform (totalCvs divisible by totalCurves) or ragged
// cvOffsets. Mirrors the engine's span validation (curveBuffer.h detail).
// Uniform spans are never materialized (bit-identical): spans[c] is
// exactly c*per in uint32 arithmetic, so callers compute curve starts
// inline and the ~4MB assign+fill+read traffic is gone. *perOut
// carries per; spans stays empty. Ragged fills as before.
bool
_Spans(usdGen::UsdGenCurveBuffer const &curves, std::vector<uint32_t> *spans,
       uint32_t *perOut, std::string *error)
{
    uint32_t const n = curves.totalCurves;
    if (curves.cvOffsets.empty()) {
        *perOut = 0;
        if (n == 0) {
            if (curves.totalCvs != 0)
                return _Fail("instance: uniform topology has CVs but no curves",
                             error);
            return true;
        }
        if (curves.totalCvs % n != 0)
            return _Fail("instance: uniform topology has non-integral CV count",
                         error);
        *perOut = curves.totalCvs / n;
        return true;
    }
    spans->assign(size_t(n) + 1, 0);
    if (curves.cvOffsets.size() != size_t(n) + 1 || curves.cvOffsets.front() != 0)
        return _Fail("instance: ragged topology has invalid offsets", error);
    for (uint32_t c = 0; c != n; ++c) {
        int const first = curves.cvOffsets[c];
        int const last = curves.cvOffsets[c + 1];
        if (first < 0 || last < first)
            return _Fail("instance: ragged topology has decreasing offsets",
                         error);
        (*spans)[c] = uint32_t(first);
        (*spans)[c + 1] = uint32_t(last);
    }
    if (spans->back() != curves.totalCvs)
        return _Fail("instance: ragged offsets disagree with CV count", error);
    return true;
}

GfVec3d
_Normalized(GfVec3d const &v, bool *ok)
{
    double const len = v.GetLength();
    if (!(len > 1e-12)) {
        *ok = false;
        return GfVec3d(0.0);
    }
    *ok = true;
    return v / len;
}

HdTokenDataSourceHandle
_Tok(TfToken const &t)
{
    return HdRetainedTypedSampledDataSource<TfToken>::New(t);
}

template <class T>
HdSampledDataSourceHandle
_Samp(T const &v)
{
    return HdRetainedTypedSampledDataSource<T>::New(v);
}

void
_Add(std::vector<TfToken> *names, std::vector<HdDataSourceBaseHandle> *values,
     TfToken const &name, HdDataSourceBaseHandle const &ds)
{
    if (ds) {
        names->push_back(name);
        values->push_back(ds);
    }
}

HdContainerDataSourceHandle
_Container(std::vector<TfToken> &&names,
           std::vector<HdDataSourceBaseHandle> &&values)
{
    return HdRetainedContainerDataSource::New(
        names.size(), names.data(), values.data());
}

// One primvars/<name> entry at `instance` interpolation (06 §4.3). The
// color role is set for displayColor only, matching USD's own color3f
// primvar convention; Storm applies no filtering to instance-rate primvars
// either way (02 §2.11).
HdContainerDataSourceHandle
_InstancePrimvar(HdSampledDataSourceHandle const &values,
                 TfToken const &role = TfToken(),
                 int elementSize = 0)
{
    HdPrimvarSchema::Builder b;
    b.SetPrimvarValue(values);
    b.SetInterpolation(_Tok(TfToken("instance")));
    if (!role.IsEmpty()) {
        b.SetRole(_Tok(role));
    }
    if (elementSize > 0) {
        b.SetElementSize(HdRetainedTypedSampledDataSource<int>::New(elementSize));
    }
    return b.Build();
}

}  // namespace

// -- vocabulary ---------------------------------------------------------------

/*static*/ TfToken const &
UsdGenInstancer::PrimitiveCards()
{
    static TfToken const token("cards");
    return token;
}

/*static*/ TfToken const &
UsdGenInstancer::PrimitiveSpheres()
{
    static TfToken const token("spheres");
    return token;
}

/*static*/ TfToken const &
UsdGenInstancer::PrimitiveArchives()
{
    static TfToken const token("archives");
    return token;
}

/*static*/ TfToken const &
UsdGenInstancer::OrientSurfaceFrame()
{
    static TfToken const token("surfaceFrame");
    return token;
}

/*static*/ TfToken const &
UsdGenInstancer::OrientCurveTangent()
{
    static TfToken const token("curveTangent");
    return token;
}

/*static*/ TfToken const &
UsdGenInstancer::OrientWorld()
{
    static TfToken const token("world");
    return token;
}

// -- paths --------------------------------------------------------------------

/*static*/ SdfPath
UsdGenInstancer::InstancerPath(SdfPath const &descriptionPath,
                               std::string const &opName)
{
    if (opName.empty()) return SdfPath::EmptyPath();
    return descriptionPath
        .AppendChild(TfToken("__usdGenRender"))
        .AppendChild(TfToken("inst_" + opName));
}

/*static*/ SdfPath
UsdGenInstancer::PrototypePath(SdfPath const &instancerPath,
                               std::string const &protoName)
{
    if (protoName.empty()) return SdfPath::EmptyPath();
    return instancerPath
        .AppendChild(TfToken("Prototypes"))
        .AppendChild(TfToken(protoName));
}

// -- bake ---------------------------------------------------------------------

/*static*/ bool
UsdGenInstancer::Validate(UsdGenInstanceParams const &params,
                          size_t prototypeCount,
                          std::string *error)
{
    if (params.primitive != PrimitiveCards() &&
        params.primitive != PrimitiveSpheres()) {
        if (params.primitive == PrimitiveArchives())
            return _Fail("instance: primitive \"archives\" is not fitted here; "
                         "subtree re-root mirroring is plugin work", error);
        return _Fail("instance: unknown primitive \"" +
                     params.primitive.GetString() + "\"", error);
    }
    if (params.orient != OrientSurfaceFrame() &&
        params.orient != OrientCurveTangent() &&
        params.orient != OrientWorld()) {
        if (params.orient == TfToken("camera"))
            return _Fail("instance: orient \"camera\" needs live camera state, "
                         "unavailable at publish time", error);
        return _Fail("instance: unknown orient \"" +
                     params.orient.GetString() + "\"", error);
    }
    if (prototypeCount == 0)
        return _Fail("instance: no prototypes bound", error);
    if (!params.weights.empty() && params.weights.size() != prototypeCount)
        return _Fail("instance: weights size does not match prototype count",
                     error);
    double total = 0.0;
    for (float w : params.weights) {
        if (!_Finite(w) || w < 0.0f)
            return _Fail("instance: weights must be finite and non-negative",
                         error);
        total += double(w);
    }
    if (!params.weights.empty() && !(total > 0.0))
        return _Fail("instance: weights sum to zero", error);
    if (!_Finite(params.scale) || !_Finite(params.twist) ||
        !_Finite(params.twistRandom) || !_Finite(params.normalOffset) ||
        !_Finite(params.scaleRandom[0]) || !_Finite(params.scaleRandom[1]))
        return _Fail("instance: non-finite scale/twist/offset parameter",
                     error);
    if (!_Finite(params.width) || params.width < 0.0f ||
        !_Finite(params.length) || params.length < 0.0f)
        return _Fail("instance: width/length must be finite and non-negative",
                     error);
    return true;
}

/*static*/ bool
UsdGenInstancer::Bake(UsdGenInstanceParams const &params,
                      UsdGenInstanceCurves const &input,
                      SdfPath const &instancerPath,
                      UsdGenInstanceResult *result,
                      std::string *error)
{
    if (!result)
        return _Fail("instance: no result output", error);
    if (!Validate(params, params.prototypes.size(), error))
        return false;
    if (instancerPath.IsEmpty())
        return _Fail("instance: empty instancer path", error);
    if (!input.curves)
        return _Fail("instance: no evaluated curves", error);
    usdGen::UsdGenCurveBuffer const &curves = *input.curves;
    uint32_t const n = curves.totalCurves;
    if (n == 0)
        return _Fail("instance: no curves to instance", error);
    if (curves.px.size() != curves.totalCvs ||
        curves.py.size() != curves.totalCvs ||
        curves.pz.size() != curves.totalCvs)
        return _Fail("instance: point planes disagree with CV count", error);
    if (curves.curveId.size() != n)
        return _Fail("instance: curveId is not per-curve", error);

    // Hoisted orient dispatch (bit-identical): params.orient is
    // loop-invariant, so the token comparisons (out-of-line calls) run once
    // instead of once per curve.
    bool const orientWorld = (params.orient == OrientWorld());
    bool const orientSurface = (params.orient == OrientSurfaceFrame());
    // Root frames are required exactly when consumed: every frame-relative
    // orient, plus any nonzero normalOffset (which pushes along rootN).
    bool const needFrames =
        !orientWorld || params.normalOffset != 0.0f;
    // Spheres ignore orientation, but a frame-relative orient still selects
    // its (discarded) frame path only for cards; spheres skip frames unless
    // the offset needs them.
    bool const wantFrames =
        params.primitive != PrimitiveSpheres() ? needFrames
                                              : params.normalOffset != 0.0f;
    if (wantFrames && (curves.rootT.size() != n ||
                       curves.rootN.size() != n ||
                       curves.rootB.size() != n))
        return _Fail("instance: root frames are not per-curve", error);

    std::vector<uint32_t> spans;  // ragged only; empty when uniform
    uint32_t uniformPer = 0;
    if (!_Spans(curves, &spans, &uniformPer, error))
        return false;
    // Same empty-cvOffsets test _Spans branches on: uniform curves read
    // spans[c] as c*uniformPer inline, ragged read the filled vector.
    bool const uniformSpans = curves.cvOffsets.empty();

    // Re-rooted prototype paths: the authored leaf re-homed under
    // <instancer>/Prototypes (06 §4.3). Leaves must be unique: two
    // targets with one leaf would collide as namespace children.
    size_t const nProtos = params.prototypes.size();
    std::vector<SdfPath> protoPaths;
    protoPaths.reserve(nProtos);
    for (SdfPath const &target : params.prototypes) {
        if (target.IsEmpty())
            return _Fail("instance: empty prototype target", error);
        std::string const leaf = target.GetName();
        if (leaf.empty())
            return _Fail("instance: prototype target has no name: " +
                         target.GetString(), error);
        SdfPath const rehomed = PrototypePath(instancerPath, leaf);
        if (std::find(protoPaths.begin(), protoPaths.end(), rehomed) !=
            protoPaths.end())
            return _Fail("instance: duplicate re-rooted prototype name: " +
                         leaf, error);
        protoPaths.push_back(rehomed);
    }

    // displayColor source: per-curve as-is, per-CV sampled at the root CV.
    bool const wantColorPerCv =
        !input.displayColor.empty() && input.displayColor.size() != size_t(n);
    if (wantColorPerCv && input.displayColor.size() != curves.totalCvs)
        return _Fail("instance: displayColor is neither per-curve nor per-CV",
                     error);

    bool const isCards = params.primitive == PrimitiveCards();
    float const cardProfile = isCards
        ? _RampEvalOne(params.widthKnots, params.widthInterpolation, 0.5f)
        : 1.0f;
    float const lo = std::min(params.scaleRandom[0], params.scaleRandom[1]);
    float const hi = std::max(params.scaleRandom[0], params.scaleRandom[1]);

    // Constant-twist fast path (bit-identical): with twistRandom == 0 the
    // angle is params.twist for every curve (twist + 0*x == twist for the
    // finite inputs Validate admits), so its cos/sin hoist out of the loop
    // and the per-curve twist draw is dead. With twist == 0 as well the
    // angle is exactly zero and the post-rotation always skips, as before.
    bool const twistConst = params.twistRandom == 0.0f;
    double const twistA =
        double(params.twist) * (3.141592653589793 / 180.0);
    double const twistCa = std::cos(twistA), twistSa = std::sin(twistA);

    double weightTotal = 0.0;
    for (float w : params.weights) weightTotal += double(w);

    // Hoisted draw seed folds (bit-identical CSE): params.seed is
    // loop-invariant, so each draw salt's _Hash64(seed32, salt) is computed
    // once per Bake instead of once per curve per draw.
    uint64_t const seed32 = uint64_t(uint32_t(params.seed));
    uint64_t const seedProto = _Hash64(seed32, kSaltInstanceProto);
    uint64_t const seedScale = _Hash64(seed32, kSaltInstanceScale);
    uint64_t const seedTwist = _Hash64(seed32, kSaltInstanceTwist);

    UsdGenInstanceResult out;
    out.instancerPath = instancerPath;
    out.prototypePaths = protoPaths;
    out.instanceIndices.assign(nProtos, VtIntArray());
    // Deferred-init sizing (bit-identical): resize(n) value-initializes
    // (~44MB of zeroes here) that the Bake loop below overwrites in full:
    // every lane of every array on both the spheres and the cards path,
    // while the _Fail exits discard `out` unread, so no uninitialized
    // element is ever observed. The arrays size through resize(n, fill)
    // with an empty filler over uninitialized storage instead. `out` is
    // function-local, so every array is fresh (null) and the fill runs
    // exactly once per array at full range.
    auto noInit = [](auto *b, auto *e) {
        (void)b;
        (void)e;
    };
    out.translations.resize(n, noInit);
    out.rotations.resize(n, noInit);
    out.scales.resize(n, noInit);
    out.prototypeIndex.resize(n, noInit);

    // Chunked over workers for big bakes: every curve's outputs are
    // per-curve independent (translations/rotations/scales/prototypeIndex
    // write disjoint lanes of the pre-sized arrays; all reads are shared
    // and immutable), the per-prototype index lists concatenate in range
    // order (identical to the serial push_back order), and the first
    // failure in curve order wins (same _Fail contract). Any chunking is
    // bit-identical. Small bakes stay serial (below ~32K curves the
    // dispatch costs more than the loop). The loop body below is
    // byte-for-byte the serial spelling, only wrapped in the range.
    int const workers = tbb::this_task_arena::max_concurrency();
    size_t const bakeChunks =
        (workers > 1 && n > 32768) ? std::min({size_t(workers), size_t(8),
                                               size_t(n)})
                                   : 1;
    auto bakeRange = [&](uint32_t c0, uint32_t c1, BakeRangeSlot &slot) {
    for (uint32_t c = c0; c != c1; ++c) {
        uint64_t const curveId = curves.curveId[c];
        uint32_t const first =
            uniformSpans ? c * uniformPer : spans[c];
        GfVec3f const root(curves.px[first], curves.py[first], curves.pz[first]);

        GfVec3d N(0.0, 0.0, 1.0);
        if (params.normalOffset != 0.0f || wantFrames)
            N = GfVec3d(curves.rootN[c]);
        out.translations[c] =
            root + GfVec3f(N * double(params.normalOffset));

        // Prototype choice: hashed weights, or a uniform hashed slot.
        uint32_t proto = 0;
        if (params.weights.empty()) {
            float const d =
                _Draw01Seeded(seedProto, curveId, kSaltInstanceProto);
            proto = std::min(uint32_t(d * float(nProtos)), uint32_t(nProtos - 1));
        } else {
            float const d =
                _Draw01Seeded(seedProto, curveId, kSaltInstanceProto);
            double const x = double(d) * weightTotal;
            double accum = 0.0;
            proto = uint32_t(nProtos - 1);
            for (size_t p = 0; p != nProtos; ++p) {
                accum += double(params.weights[p]);
                if (x < accum) {
                    proto = uint32_t(p);
                    break;
                }
            }
        }
        out.prototypeIndex[c] = int(proto);
        slot.indices[proto].push_back(int(c));

        float const sDraw =
            _Draw01Seeded(seedScale, curveId, kSaltInstanceScale);
        float const s = params.scale * (lo + (hi - lo) * sDraw);

        if (!isCards) {
            // Spheres: identity rotation (twist is a no-op); the width is
            // the sphere diameter over a unit-diameter prototype.
            out.rotations[c] = GfQuatf::GetIdentity();
            out.scales[c] = GfVec3f(params.width * s);
            continue;
        }

        // Cards: orientation frame + twist about the length axis.
        GfVec3d xAxis(1.0, 0.0, 0.0), yAxis(0.0, 1.0, 0.0),
            zAxis(0.0, 0.0, 1.0);
        if (orientWorld) {
            // Identity basis: nothing to derive.
        } else if (orientSurface) {
            bool ok = false;
            yAxis = _Normalized(GfVec3d(curves.rootB[c]), &ok);
            if (!ok) {
                slot.error.failed = true; slot.error.c = c;
                slot.error.message = "instance: degenerate rootB frame";
                break;
            }
            // Reuse N (bit-identical CSE): this path implies wantFrames,
            // so N above already holds exactly GfVec3d(curves.rootN[c]).
            GfVec3d z = N - yAxis * GfDot(N, yAxis);
            zAxis = _Normalized(z, &ok);
            if (!ok) {
                slot.error.failed = true; slot.error.c = c;
                slot.error.message = "instance: rootN parallel to rootB";
                break;
            }
            xAxis = GfCross(yAxis, zAxis);
        } else {
            // curveTangent: y follows the root segment; z is the surface
            // normal with the tangent projected out.
            GfVec3d t(0.0);
            bool tok = false;
            uint32_t const next =
                uniformSpans ? (c + 1) * uniformPer : spans[c + 1];
            if (next > first + 1) {
                uint32_t const next = first + 1;
                t = GfVec3d(double(curves.px[next]) - double(curves.px[first]),
                            double(curves.py[next]) - double(curves.py[first]),
                            double(curves.pz[next]) - double(curves.pz[first]));
                yAxis = _Normalized(t, &tok);
            }
            if (!tok) {
                // Single-CV span or a zero-length root segment: legal
                // topology, so fall back to the growth axis, not failure.
                yAxis = _Normalized(GfVec3d(curves.rootB[c]), &tok);
                if (!tok) {
                    slot.error.failed = true; slot.error.c = c;
                    slot.error.message =
                        "instance: degenerate tangent fallback";
                    break;
                }
            }
            // Reuse N (bit-identical CSE): this path implies wantFrames,
            // so N above already holds exactly GfVec3d(curves.rootN[c]).
            GfVec3d ref = N;
            // Squared parallel-fallback test (saves a normalize: 1 sqrt +
            // 3 divs): |dot(ref/len, y)| > 0.999 with len > 0 squares to
            // d*d > 0.999^2*l2. d and l2 are the exact dots; the
            // degenerate threshold sits a few ulp above (1e-12)^2 so a
            // truly-degenerate ref always falls back, as !rok does.
            // Matches except within rounding of either boundary.
            double const rd = GfDot(ref, yAxis);
            double const rl2 = GfDot(ref, ref);
            bool const refFallback = !(rl2 > 1.000000000000002e-24) ||
                rd * rd > (0.999 * 0.999) * rl2;
            if (refFallback)
                ref = GfVec3d(curves.rootT[c]);
            // Reuse rd on the common path (bit-identical CSE): ref is
            // unchanged when the fallback is not taken, so the tested dot
            // is exactly the projection dot; the fallback re-dots verbatim.
            double const rd2 = refFallback ? GfDot(ref, yAxis) : rd;
            GfVec3d z = ref - yAxis * rd2;
            zAxis = _Normalized(z, &tok);
            if (!tok) {
                slot.error.failed = true; slot.error.c = c;
                slot.error.message = "instance: tangent parallel to frame";
                break;
            }
            xAxis = GfCross(yAxis, zAxis);
        }

        float twDeg = params.twist;
        if (!twistConst) {
            float const tDraw =
                _Draw01Seeded(seedTwist, curveId, kSaltInstanceTwist);
            twDeg =
                params.twist + params.twistRandom * (tDraw * 2.0f - 1.0f);
        }
        if (!orientWorld && twDeg != 0.0f) {
            // Local-space post-rotation by R_y(+tw): newX = M*(ca,0,-sa),
            // newZ = M*(sa,0,ca). twist = +90 about Y sends local X to -Z.
            double ca, sa;
            if (twistConst) {
                ca = twistCa;
                sa = twistSa;
            } else {
                double const a =
                    double(twDeg) * (3.141592653589793 / 180.0);
                ca = std::cos(a);
                sa = std::sin(a);
            }
            GfVec3d const x = xAxis * ca - zAxis * sa;
            GfVec3d const z = xAxis * sa + zAxis * ca;
            xAxis = x;
            zAxis = z;
        }
        // Row-vector frame: the frame axes are ROWS (row i = image of
        // basis vector i), matching the root-frame row convention every
        // consumer uses. Columns would bake the transposed rotation.
        out.rotations[c] = _QuatFromFrameRows(xAxis, yAxis, zAxis);
        out.scales[c] = GfVec3f(
            params.width * cardProfile * s, params.length * s, s);
    }
    };
    std::vector<BakeRangeSlot> slots(bakeChunks);
    for (auto &s : slots)
        s.indices.resize(nProtos);
    if (bakeChunks == 1) {
        bakeRange(0, n, slots[0]);
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, bakeChunks),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t t = range.begin(); t != range.end(); ++t) {
                    uint32_t const c0 =
                        uint32_t((t * uint64_t(n)) / bakeChunks);
                    uint32_t const c1 =
                        uint32_t(((t + 1) * uint64_t(n)) / bakeChunks);
                    bakeRange(c0, c1, slots[t]);
                }
            });
    }
    // Ranges are curve-ordered, so the first failed range holds the serial
    // loop's first failure: same _Fail contract.
    for (size_t t = 0; t < bakeChunks; ++t) {
        if (!slots[t].error.failed)
            continue;
        return _Fail(slots[t].error.message, error);
    }
    // Concatenate the per-range index lists in range order: identical to
    // the serial push_back order (same values; capacities may differ,
    // which no reader observes).
    // Uninitialized sizing (bit-identical): total is the exact sum of the
    // copied sizes and the loop below writes every lane contiguously, so
    // resize's value-init (~4MB of zeroes here) is dead and folds into
    // the noInit filler over uninitialized storage. Same bytes either way.
    for (size_t p = 0; p < nProtos; ++p) {
        size_t total = 0;
        for (auto const &s : slots)
            total += s.indices[p].size();
        VtIntArray &dst = out.instanceIndices[p];
        dst.resize(total, noInit);
        int *w = dst.empty() ? nullptr : dst.data();
        for (auto const &s : slots) {
            if (!s.indices[p].empty()) {
                std::memcpy(w, s.indices[p].data(),
                            s.indices[p].size() * sizeof(int));
                w += s.indices[p].size();
            }
        }
    }

    // Variation primvars: default {"displayColor"} when nothing is authored.
    VtArray<TfToken> names = params.variationPrimvars;
    if (names.empty())
        names = VtArray<TfToken>{TfToken("displayColor")};
    std::vector<TfToken> seen;
    for (TfToken const &name : names) {
        if (name.IsEmpty() ||
            std::find(seen.begin(), seen.end(), name) != seen.end())
            continue;
        seen.push_back(name);
        if (name == TfToken("displayColor")) {
            if (input.displayColor.empty()) {
                out.unresolvedPrimvars.push_back(name);
                continue;
            }
            usdGen::UsdGenPlane plane;
            plane.name = name;
            plane.interpolation = TfToken("instance");
            plane.type = TfToken("float");
            plane.arity = 3;
            // Uninitialized sizing (bit-identical): the resize + full
            // overwrite below folds into resize(count, fill), whose filler
            // runs over uninitialized storage. `plane` is function-local, so
            // the fill runs exactly once at full range. Same bytes either way.
            size_t const count3 = size_t(n) * 3;
            if (!wantColorPerCv) {
                // Per-curve colors are contiguous GfVec3f == 3 floats: one
                // copy instead of a strided per-component loop. Same bytes.
                void const *src = input.displayColor.cdata();
                plane.f.resize(count3, [src](float *b, float *e) {
                    _MemcpyChunked(b, src,
                                   size_t(e - b) * sizeof(float));
                });
            } else {
                GfVec3f const *colors = input.displayColor.cdata();
                uint32_t const *offsets =
                    uniformSpans ? nullptr : spans.data();
                // Chunked over curve ranges for big bakes (bit-identical):
                // every curve reads its own root-CV color and writes a
                // disjoint output triple, so any range split writes the
                // same bytes. Small bakes stay serial: below ~32K curves
                // the dispatch costs more than the gather.
                size_t const gatherChunks =
                    (workers > 1 && n > 32768)
                        ? std::min({size_t(workers), size_t(8), size_t(n)})
                        : 1;
                plane.f.resize(count3, [colors, offsets, uniformSpans,
                                        uniformPer, gatherChunks](float *b,
                                                                 float *e) {
                    if (gatherChunks == 1) {
                        float *d = b;
                        for (uint32_t c = 0; d != e; ++c, d += 3) {
                            GfVec3f const v = colors[uniformSpans
                                                         ? c * uniformPer
                                                         : offsets[c]];
                            new (d + 0) float(v[0]);
                            new (d + 1) float(v[1]);
                            new (d + 2) float(v[2]);
                        }
                        return;
                    }
                    size_t const M = size_t(e - b) / 3;
                    tbb::parallel_for(
                        tbb::blocked_range<size_t>(0, gatherChunks),
                        [&](tbb::blocked_range<size_t> const &range) {
                            for (size_t t = range.begin(); t != range.end();
                                 ++t) {
                                size_t const c0 = (t * M) / gatherChunks;
                                size_t const c1 =
                                    ((t + 1) * M) / gatherChunks;
                                for (size_t c = c0; c < c1; ++c) {
                                    GfVec3f const v = colors[uniformSpans
                                                                 ? uint32_t(c) *
                                                                       uniformPer
                                                                 : offsets[c]];
                                    float *d = b + 3 * c;
                                    new (d + 0) float(v[0]);
                                    new (d + 1) float(v[1]);
                                    new (d + 2) float(v[2]);
                                }
                            }
                        });
                });
            }
            out.varyings.push_back(plane);
            continue;
        }
        usdGen::UsdGenPlane const *found = nullptr;
        for (usdGen::UsdGenPlane const &plane : curves.extraCurve) {
            if (plane.name == name) {
                found = &plane;
                break;
            }
        }
        if (!found || (found->type != TfToken("float") &&
                       found->type != TfToken("int")) ||
            found->arity == 0) {
            out.unresolvedPrimvars.push_back(name);
            continue;
        }
        size_t const expect = size_t(n) * found->arity;
        usdGen::UsdGenPlane plane;
        plane.name = name;
        plane.interpolation = TfToken("instance");
        plane.type = found->type;
        plane.arity = found->arity;
        if (found->interpolation == TfToken("constant")) {
            // One value shared by every instance: broadcast it.
            if ((found->type == TfToken("float") &&
                 found->f.size() != found->arity) ||
                (found->type == TfToken("int") &&
                 found->i.size() != found->arity)) {
                out.unresolvedPrimvars.push_back(name);
                continue;
            }
            if (found->type == TfToken("float")) {
                float const *vals = found->f.cdata();
                uint32_t const arity = found->arity;
                plane.f.resize(expect, [vals, arity](float *b, float *e) {
                    for (float *d = b; d != e;) {
                        for (uint32_t k = 0; k != arity; ++k, ++d)
                            new (d) float(vals[k]);
                    }
                });
            } else {
                int const *vals = found->i.cdata();
                uint32_t const arity = found->arity;
                plane.i.resize(expect, [vals, arity](int *b, int *e) {
                    for (int *d = b; d != e;) {
                        for (uint32_t k = 0; k != arity; ++k, ++d)
                            new (d) int(vals[k]);
                    }
                });
            }
        } else if (found->interpolation == TfToken("uniform")) {
            if ((found->type == TfToken("float") &&
                 found->f.size() != expect) ||
                (found->type == TfToken("int") &&
                 found->i.size() != expect)) {
                out.unresolvedPrimvars.push_back(name);
                continue;
            }
            plane.f = found->f;
            plane.i = found->i;
        } else {
            out.unresolvedPrimvars.push_back(name);
            continue;
        }
        out.varyings.push_back(plane);
    }

    *result = out;
    return true;
}

// -- Hydra assembly -----------------------------------------------------------

/*static*/ HdContainerDataSourceHandle
UsdGenInstancer::BuildInstancerDataSource(
    UsdGenInstanceResult const &result,
    SdfPath const &primOrigin)
{
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;

    {
        // 06 §4.3: prototypes are namespace children of the instancer;
        // instanceIndices is HdIntArrayVectorSchema (a vector of VtIntArray,
        // one per prototype). instanceLocations stays null: this is explicit
        // instancing. Mask stays absent: Bake writes all-true.
        HdInstancerTopologySchema::Builder tb;
        if (!result.prototypePaths.empty()) {
            VtArray<SdfPath> paths(result.prototypePaths.size());
            for (size_t i = 0; i != result.prototypePaths.size(); ++i)
                paths[i] = result.prototypePaths[i];
            tb.SetPrototypes(
                HdRetainedTypedSampledDataSource<VtArray<SdfPath>>::New(paths));
        }
        if (!result.instanceIndices.empty()) {
            std::vector<HdDataSourceBaseHandle> elems;
            elems.reserve(result.instanceIndices.size());
            for (VtIntArray const &arr : result.instanceIndices) {
                elems.push_back(HdDataSourceBaseHandle(
                    HdRetainedTypedSampledDataSource<VtIntArray>::New(arr)));
            }
            tb.SetInstanceIndices(HdRetainedSmallVectorDataSource::New(
                elems.size(), elems.data()));
        }
        if (!result.mask.empty()) {
            tb.SetMask(HdRetainedTypedSampledDataSource<VtBoolArray>::New(
                result.mask));
        }
        _Add(&names, &values, TfToken("instancerTopology"), tb.Build());
    }

    {
        std::vector<TfToken> pvNames;
        std::vector<HdDataSourceBaseHandle> pvValues;
        _Add(&pvNames, &pvValues, HdInstancerTokens->instanceTranslations,
             _InstancePrimvar(_Samp(result.translations)));
        // USD maps PointInstancer `orientations` (quath[]) straight through
        // to hydra:instanceRotations (dataSourcePointInstancer.cpp), and
        // hdMoonray only accepts VtQuathArray there: a VtQuatfArray
        // mis-syncs every instance. Storm accepts both encodings, so the
        // data source publishes quath -- the wire type every Hydra consumer
        // reads -- while Bake keeps full float precision (docs/moonray-fur.md).
        // Uninitialized sizing (bit-identical): VtQuathArray(n)
        // value-initializes (8MB of zeroes here) that the vector
        // quatf->quath conversion overwrites in full, so the array sizes
        // through resize(n, fill) with the conversion as the filler over
        // uninitialized storage. `quath` is function-local, so the fill
        // runs exactly once at full range.
        VtQuathArray quath;
        GfQuatf const *quatSrc = result.rotations.cdata();
        quath.resize(result.rotations.size(),
                     [quatSrc](GfQuath *b, GfQuath *e) {
                         _ConvertQuats(b, quatSrc, size_t(e - b));
                     });
        _Add(&pvNames, &pvValues, HdInstancerTokens->instanceRotations,
             _InstancePrimvar(_Samp(quath)));
        _Add(&pvNames, &pvValues, HdInstancerTokens->instanceScales,
             _InstancePrimvar(_Samp(result.scales)));
        for (usdGen::UsdGenPlane const &plane : result.varyings) {
            HdSampledDataSourceHandle sampled;
            TfToken role;
            int elementSize = 0;
            if (plane.type == TfToken("int")) {
                // Scalar ints ride through flat; hdMoonray reads them, and
                // only Storm honors elementSize on multi-arity int planes.
                sampled = _Samp(plane.i);
                elementSize = plane.arity;
            } else if (plane.arity == 3 && plane.f.size() % 3 == 0) {
                // hdMoonray ignores elementSize, so pack float3: a flat
                // VtFloatArray would be misread as one scalar per instance.
                // GfVec3f is 3 contiguous floats: one copy, same bytes.
                // Uninitialized sizing (bit-identical): the pack sizes
                // through resize(count, fill) with the copy as the filler
                // over uninitialized storage. Same bytes either way.
                VtVec3fArray packed;
                void const *packSrc = plane.f.cdata();
                packed.resize(plane.f.size() / 3,
                              [packSrc](GfVec3f *b, GfVec3f *e) {
                                  _MemcpyChunked(b, packSrc,
                                                 size_t(e - b) *
                                                 sizeof(GfVec3f));
                              });
                sampled = _Samp(packed);
                if (plane.name == TfToken("displayColor"))
                    role = TfToken("color");
            } else if (plane.arity == 2 && plane.f.size() % 2 == 0) {
                // Likewise: GfVec2f is 2 contiguous floats.
                VtVec2fArray packed;
                void const *packSrc = plane.f.cdata();
                packed.resize(plane.f.size() / 2,
                              [packSrc](GfVec2f *b, GfVec2f *e) {
                                  _MemcpyChunked(b, packSrc,
                                                 size_t(e - b) *
                                                 sizeof(GfVec2f));
                              });
                sampled = _Samp(packed);
            } else {
                sampled = _Samp(plane.f);
                elementSize = plane.arity;
            }
            _Add(&pvNames, &pvValues, plane.name,
                 _InstancePrimvar(sampled, role, elementSize));
        }
        _Add(&names, &values, TfToken("primvars"),
             _Container(std::move(pvNames), std::move(pvValues)));
    }

    // Instance translations are final (as tile points are): identity xform
    // with resetXformStack, replacing any inherited surface chain.
    _Add(&names, &values, TfToken("xform"),
         HdXformSchema::Builder()
             .SetMatrix(HdRetainedTypedSampledDataSource<GfMatrix4d>::New(
                 GfMatrix4d(1.0)))
             .SetResetXformStack(
                 HdRetainedTypedSampledDataSource<bool>::New(true))
             .Build());

    // Purpose/visibility exactly as tiles inherit them (see the publisher):
    // purpose only when authored ("default" is fatal); visibility always.
    if (!result.purpose.IsEmpty()) {
        _Add(&names, &values, TfToken("purpose"),
             _Container({TfToken("purpose")},
                        {HdDataSourceBaseHandle(_Tok(result.purpose))}));
    }
    _Add(&names, &values, TfToken("visibility"),
         HdVisibilitySchema::Builder()
             .SetVisibility(HdRetainedTypedSampledDataSource<bool>::New(
                 result.visibility != TfToken("invisible")))
             .Build());

    if (!primOrigin.IsEmpty()) {
        _Add(&names, &values, TfToken("primOrigin"),
             _Container({TfToken("scenePath")},
                        {HdDataSourceBaseHandle(_Samp(primOrigin))}));
    }
    return _Container(std::move(names), std::move(values));
}

/*static*/ HdContainerDataSourceHandle
UsdGenInstancer::BuildInstancedByDataSource(
    SdfPath const &instancerPath,
    SdfPath const &prototypeRoot)
{
    VtArray<SdfPath> paths(1, instancerPath);
    VtArray<SdfPath> roots(1, prototypeRoot);
    return HdInstancedBySchema::Builder()
        .SetPaths(HdRetainedTypedSampledDataSource<VtArray<SdfPath>>::New(paths))
        .SetPrototypeRoots(
            HdRetainedTypedSampledDataSource<VtArray<SdfPath>>::New(roots))
        .Build();
}

// -- notices ------------------------------------------------------------------

/*static*/ UsdGenInstancer::InstanceNotices
UsdGenInstancer::NoticesFor(bool topologyChanged,
                            bool transformsChanged,
                            std::vector<TfToken> const &varyingNames)
{
    InstanceNotices n;
    n.topologyDirty = topologyChanged;
    n.transformsDirty = transformsChanged;
    n.dirtyVaryings = varyingNames;
    return n;
}

/*static*/ std::vector<HdDataSourceLocator>
UsdGenInstancer::InstanceNotices::all() const
{
    std::vector<HdDataSourceLocator> out;
    if (topologyDirty) {
        out.emplace_back(TfToken("instancerTopology"));
    }
    if (transformsDirty) {
        // One BAR re-upload; never the topology (06 §4.3).
        out.emplace_back(TfToken("primvars"),
                         HdInstancerTokens->instanceTranslations,
                         TfToken("primvarValue"));
    }
    for (TfToken const &name : dirtyVaryings) {
        out.emplace_back(TfToken("primvars"), name, TfToken("primvarValue"));
    }
    return out;
}

}  // namespace usdGenImaging
