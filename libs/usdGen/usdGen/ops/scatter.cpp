// usdGen — UsdGenScatterOp implementation (M1, mode="random").
//
// 02-schema.md §2.6, 04-operators.md :620-676. Area-weighted scatter:
//   n_f = floor(expected) + (Hash01(Hash64(seed, f, kSaltScatter),
//                                    kSaltScatter) < frac(expected))
//   expected = density * areaRest(f)                        (plan/04 :634-635)
// Curve ids are UsdGenCurveId(seed, faceIndex, k) (plan/04 :637) so they stay
// stable across density edits (plan/04 :663).
// Each root position is drawn from three curveId-keyed draws (area-weighted
// fan-triangle pick + uniform in-triangle point): the M1 realization of the
// "low-discrepancy sample seeded by curveId" (plan/04 :638) — being purely
// curveId-keyed it is order-invariant across thread counts and chunk splits
// (review M-3 / engine rule E-8). Roots are Morton-sorted by rest position so
// chunk placement is surface-major (plan/04 :641).
#include "usdGen/ops/scatter.h"

#include "usdGen/opParams.h"
#include "usdGen/digest.h"
#include "usdGen/limitSurface.h"
#include "usdGen/scheduler.h"
#include "usdGen/ops/opUtil.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <type_traits>
#include <vector>
PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenScatterCapture final : public UsdGenCapture
{
    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        auto c = std::make_unique<UsdGenScatterCapture>();
        *c = *this;
        return c;
    }
    bool OwnsBuffer() const override { return true; }
};

GfVec3f Normalize3(GfVec3f const &v)
{
    const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return l > 1e-12f ? v / l : GfVec3f(0.0f, 1.0f, 0.0f);
}

// Pooled per-thread sort scratch (bit-identical): the morton/order/tmp
// vectors churn ~24MB of mmap/munmap per Capture; pooling across calls on
// the thread turns every rep after the first into pure reuse.
// Growth-only (resize never shrinks); every slot is overwritten before its
// read (key loop, radix passes, per-pass counts fill), so pooled contents
// never leak across calls. Deliberately sort-only: pooling the ~270MB
// emission over-reserve too measurably slows the Bake stages' fresh large
// allocs (cross-stage interference), dwarfing the extra capture win.
// thread_local keeps concurrent captures on distinct scratch; Capture
// never reenters itself, so one set per thread suffices.
struct ScatterSortScratch {
    std::vector<uint64_t> morton, tmpKeys;
    std::vector<uint32_t> order, tmpIdx;
    std::vector<uint32_t> counts;
    // Parallel-radix per-chunk histograms and scatter offsets, chunk-major
    // ([c * 65536 + d]): each chunk's worker touches only its own 256KB
    // region, so neither the histogram nor the scatter shares a line.
    // Pooled like the rest (every slot is rewritten each pass).
    std::vector<uint32_t> sortCounts, sortOff;
};
thread_local ScatterSortScratch t_scatterSortScratch;

// Per-range emission storage: the nine pre-sort planes. Ranges concatenate
// in face order after the join, so the merged arrays are identical for any
// partition count (every face's roots depend only on that face's corners
// plus loop-invariant seeds).
struct ScatterEmission {
    std::vector<float> ax, ay, az;
    std::vector<uint64_t> aids;
    std::vector<int> aPrim;
    std::vector<GfVec2f> aUv;
    std::vector<GfVec3f> aT, aN, aB;
    void reserve(size_t n)
    {
        ax.reserve(n); ay.reserve(n); az.reserve(n);
        aids.reserve(n); aPrim.reserve(n);
        aUv.reserve(n); aT.reserve(n); aN.reserve(n); aB.reserve(n);
    }
    size_t size() const { return aids.size(); }
    void moveTo(ScatterEmission &dst)
    {
        dst.ax = std::move(ax);
        dst.ay = std::move(ay);
        dst.az = std::move(az);
        dst.aids = std::move(aids);
        dst.aPrim = std::move(aPrim);
        dst.aUv = std::move(aUv);
        dst.aT = std::move(aT);
        dst.aN = std::move(aN);
        dst.aB = std::move(aB);
    }
};

// First error inside one range (face order decides across ranges, exactly
// like the serial loop's first failure). An empty message is the bare
// per-root limit failure, which reports no diagnostic.
struct ScatterEmitError {
    bool failed = false;
    size_t fi = 0;
    std::string message;
};

// One slot per range, cache-line disjoint: every range's thread hammers its
// own nine vector sizes on every root, and the unpadded stride would share
// a line between neighbours.
struct alignas(128) ScatterEmitSlot {
    ScatterEmission emission;
    ScatterEmitError error;
};

// Two-pass spill: pass 1's per-face fan outputs for big level-0 captures
// (see Capture). 32 bytes, naturally aligned, trivially constructible:
// the slice is uninitialized storage (a resize-zero here would serialize
// 8K first-touch faults every capture), so pass 1 fully writes every
// visited slot -- valid faces get all five fields, skipped faces get an
// explicit zero struct (nf == 0: pass 2 emits nothing for them, exactly
// like a face whose stochastic rounding drew no roots). Faces past the
// inline fan width (ntri > 2) spill zeros for the weights and recompute
// the fan in pass 2 from the same fanFace code.
struct ScatterSpill {
    double areaRest;
    float w0, w1;
    GfVec3f nrest;
    uint32_t nf;
};
static_assert(std::is_trivially_default_constructible<ScatterSpill>::value,
              "spill slices stay uninitialized");
static_assert(std::is_trivially_copyable<ScatterSpill>::value,
              "spill assignment is a plain store");
static_assert(sizeof(ScatterSpill) == 32, "spill is 32 bytes");
static_assert(offsetof(ScatterSpill, areaRest) == 0, "spill layout");
static_assert(offsetof(ScatterSpill, w0) == 8, "spill layout");
static_assert(offsetof(ScatterSpill, nrest) == 16, "spill layout");
static_assert(offsetof(ScatterSpill, nf) == 28, "spill layout");

// One slot per two-pass range: the range's root total (for the serial
// inter-range prefix) and the range's first error. The pass-1 spill
// slices live in the pooled emission scratch, flat across ranges.
struct ScatterSpillSlot {
    size_t total = 0;
    ScatterEmitError error;
};

// Two-pass merged roots: the nine planes as uninitialized buffers. Pass 2
// writes every slot exactly once (per-face cursors partition [0, total)),
// so no zero-fill runs and first-touch faults parallelize over the pass-2
// workers for free. Trivial element types only (asserted): assignment
// carries the same bytes push_back's copy-construction would.
struct ScatterRawRoots {
    std::unique_ptr<float[]> ax, ay, az;
    std::unique_ptr<uint64_t[]> aids;
    std::unique_ptr<int[]> aPrim;
    std::unique_ptr<GfVec2f[]> aUv;
    std::unique_ptr<GfVec3f[]> aT, aN, aB;
};

// Pooled two-pass emission scratch (bit-identical): pass 1's spill
// (~32MB at 1M faces) and pass 2's raw root planes (~68MB at 1M roots)
// churn through fresh alloc+fault+free every Capture; pooling across
// calls on the thread turns every rep after the first into pure reuse.
// Growth-only (never shrinks); pass 1 fully writes every visited spill
// slot (explicit zeros on skips) and pass 2 writes every raw slot
// [0, total) exactly once, so pooled contents never leak across calls.
// Retains the largest capture seen per thread (~100MB at bench scale),
// like the sort scratch and the convert/capture shells.
struct ScatterEmissionScratch {
    std::unique_ptr<ScatterSpill[]> spill;
    size_t spillFaces = 0;
    ScatterRawRoots raw;
    size_t rawTotal = 0;
};
thread_local ScatterEmissionScratch t_scatterEmissionScratch;
static_assert(std::is_trivially_default_constructible<GfVec2f>::value,
              "raw vec2 planes stay uninitialized");
static_assert(std::is_trivially_default_constructible<GfVec3f>::value,
              "raw vec3 planes stay uninitialized");
static_assert(std::is_trivially_copyable<GfVec2f>::value,
              "raw vec2 assignment is a plain store");
static_assert(std::is_trivially_copyable<GfVec3f>::value,
              "raw vec3 assignment is a plain store");

// Parallel-for over N slots: the scheduler-bound arena when Capture runs
// under a graph (ctx.dispatcher), plain TBB when Capture is called
// directly (bench, CUDA scatter input). Each slot runs exactly once and no
// slot's contents depend on scheduling, so results are identical under any
// worker count (or serially).
template <class F>
void ScatterParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (dispatcher) {
        struct Payload { F const *body; } payload{&body};
        dispatcher->ParallelFor(count, [](size_t i, void *p) {
            (*static_cast<Payload *>(p)->body)(i);
        }, &payload);
        return;
    }
    tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
        [&](tbb::blocked_range<size_t> const &range) {
            for (size_t i = range.begin(); i != range.end(); ++i)
                body(i);
        });
}

// Merge one plane's segments into the concatenated emission: size the plane
// then memcpy every segment to its prefix offset. Every plane is trivially
// copyable. Planes merge on different workers (first-touch faults
// parallelize); segments of one plane stay serial (one resize).
template <class T>
void ScatterCopyPlane(std::vector<T> ScatterEmission::*plane,
                      ScatterEmission &dst, ScatterEmitSlot *slots,
                      size_t const *segBase, size_t partitions, size_t total)
{
    std::vector<T> &d = dst.*plane;
    d.resize(total);
    T *out = d.data();
    for (size_t p = 0; p < partitions; ++p) {
        std::vector<T> const &s = slots[p].emission.*plane;
        if (!s.empty())
            std::memcpy(out + segBase[p], s.data(), s.size() * sizeof(T));
    }
}

inline uint64_t double_as_bits(double d)
{
    uint64_t u;
    static_assert(sizeof(u) == sizeof(d), "");
    std::memcpy(&u, &d, sizeof(d));
    return u;
}

// Inline UsdGenMortonKey3 (kernels.cpp is canonical, copied verbatim):
// the quantize is exact at every step (float->double is exact, the
// cellScale multiply is exact for the 64.0f power of two, floor/clamp/
// convert are exact) and the interleave is integer-only, so no
// contraction-sensitive pattern exists and the inline is bit-identical
// under any flags. Drops the cross-TU call per root and lets the key
// loop vectorize. Re-check against kernels.cpp on touch.
inline uint64_t ScatterMortonKey3(float x, float y, float z, float cellScale)
{
    auto split3 = [](uint32_t v) -> uint64_t {
        uint64_t s = v & 0x1FFFFFull;
        s = (s | (s << 32)) & 0x001F00000000FFFFull;
        s = (s | (s << 16)) & 0x001F0000FF0000FFull;
        s = (s | (s << 8)) & 0x100F00F00F00F00Full;
        s = (s | (s << 4)) & 0x10C30C30C30C30C3ull;
        s = (s | (s << 2)) & 0x1249249249249249ull;
        return s;
    };
    auto quant = [cellScale](float v) -> uint32_t {
        double q = std::floor(double(v) * double(cellScale));
        if (q < double(-(1 << 20))) q = double(-(1 << 20));
        if (q > double((1 << 20) - 1)) q = double((1 << 20) - 1);
        return uint32_t(std::int64_t(q) + (1 << 20));
    };
    return split3(quant(x)) | (split3(quant(y)) << 1) |
        (split3(quant(z)) << 2);
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("enabled"));      // generator: topology-class (02 §6.2)
    v.push_back(TfToken("density"));     // capture (02 §6.4)
    v.push_back(TfToken("flip"));
    v.push_back(TfToken("subdivisionLevel"));
    return v;
}();

// Scatter is a generator: it has no upstream curves to leave untouched, so it
// declares no usdGen:mask.
static TfTokenVector _valueParams = [] {
    return UsdGenBaseValueParams();
}();

TfSpan<const TfToken> UsdGenScatterOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenScatterOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenScatterOp::CreateCapture() const
{
    return std::make_unique<UsdGenScatterCapture>();
}
uint32_t UsdGenScatterOp::PlanesTouched() const
{
    return kPlanePoints | kPlaneHairT;
}

uint64_t UsdGenScatterOp::SubdivisionDigestCache::Digest(
    UsdGenSurfaceDesc const &surface)
{
    auto same = [](auto const &a, auto const &b) {
        return a.size() == b.size() && (a.empty() || a.cdata() == b.cdata());
    };
    if (valid_ && same(surface.faceVertexCounts, counts_) &&
        same(surface.faceVertexIndices, indices_) &&
        same(surface.holeIndices, holes_) &&
        same(surface.creaseIndices, creaseIndices_) &&
        same(surface.creaseLengths, creaseLengths_) &&
        same(surface.creaseSharpnesses, creaseSharpnesses_) &&
        same(surface.cornerIndices, cornerIndices_) &&
        same(surface.cornerSharpnesses, cornerSharpnesses_) &&
        surface.subdivisionScheme == scheme_ &&
        surface.orientation == orientation_ &&
        surface.interpolateBoundary == interpolateBoundary_ &&
        surface.faceVaryingLinearInterpolation ==
            faceVaryingLinearInterpolation_ &&
        surface.creaseMethod == creaseMethod_ &&
        surface.triangleSubdivisionRule == triangleSubdivisionRule_)
        return digest_;
    counts_ = surface.faceVertexCounts;
    indices_ = surface.faceVertexIndices;
    holes_ = surface.holeIndices;
    creaseIndices_ = surface.creaseIndices;
    creaseLengths_ = surface.creaseLengths;
    creaseSharpnesses_ = surface.creaseSharpnesses;
    cornerIndices_ = surface.cornerIndices;
    cornerSharpnesses_ = surface.cornerSharpnesses;
    scheme_ = surface.subdivisionScheme;
    orientation_ = surface.orientation;
    interpolateBoundary_ = surface.interpolateBoundary;
    faceVaryingLinearInterpolation_ =
        surface.faceVaryingLinearInterpolation;
    creaseMethod_ = surface.creaseMethod;
    triangleSubdivisionRule_ = surface.triangleSubdivisionRule;
    digest_ = UsdGenSubdivisionDigest(surface);
    valid_ = true;
    return digest_;
}

bool UsdGenScatterOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    int const level=params.GetInt(TfToken("subdivisionLevel"),0);
    if(level<0 || level>6) {
        if(diag)diag->Error("UsdGenScatter: subdivisionLevel must be in [0,6]");
        return false;
    }
    double const density = params.GetDouble(TfToken("density"), 100.0);
    if (!std::isfinite(density) || density < 0.0) {
        if (diag) diag->Error("UsdGenScatter: density must be finite and >= 0");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenScatterOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Scatter consumes only rest geometry. The general surfaceGeneration
    // includes posed points, shutter samples and world transforms, so it
    // would regenerate rest roots on every animation frame.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string const &k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    feed("density", p ? double_as_bits(p->GetDouble(TfToken("density"), 100.0)) : 0);
    feed("flip", p && p->GetBool(TfToken("flip"), false) ? 1u : 0u);
    feed("seed", ctx.seed);
    feed("subdivisionLevel", p ? p->GetInt(TfToken("subdivisionLevel"),0) : 0);
    UsdGenGraphDesc const *desc = ctx.desc;
    if (desc && ctx.surface < desc->surfaces.size()) {
        auto const &surface = desc->surfaces[ctx.surface];
        auto array = [&](char const *name, auto &cache, auto const &values) {
            // Word-at-a-time FNV-1a (opUtil::Digest::MixBytes): one round
            // per 8 bytes instead of one per byte. The epoch values differ
            // from byte mixing, but epochs are only compared for equality
            // within a process.
            feed(name, cache.Digest(values));
            feed(name, uint64_t(values.size()));
        };
        array("restPoints", restPointsDigest_, surface.restPoints);
        array("faceCounts", faceCountsDigest_, surface.faceVertexCounts);
        array("faceIndices", faceIndicesDigest_, surface.faceVertexIndices);
        array("subset", subsetFacesDigest_, surface.subsetFaces);
        feed("isSubset", surface.isSubset ? 1u : 0u);
        array("uv", uvDigest_, surface.uv);
        // Memoized on buffer identity (a hit is ~10 pointer compares), so
        // the steady state never re-streams the topology arrays; the cold
        // miss hashes with the word-wise FNV shared with the tags digest.
        feed("subdivision", subdivisionDigest_.Digest(surface));
        // The paint primvar edits no generation, so the multiplier content
        // itself joins the digest: without this a paint stroke would read
        // back the cached pre-stroke roots.
        auto const &mult = desc->surfaces[ctx.surface].densityMultiplier;
        feed("densityMultSize", uint64_t(mult.size()));
        feed("densityMult", densityMultDigest_.Digest(mult));
    }

    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenScatterOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    TF_UNUSED(upstream);
    UsdGenGraphDesc const *desc = ctx.desc;
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!desc) {
        if (diag) diag->Error("UsdGenScatter::Capture: no graph description");
        return false;
    }
    if (ctx.surface >= desc->surfaces.size()) {
        if (diag) diag->Error("UsdGenScatter::Capture: no bound usdGen:surface");
        return false;
    }
    UsdGenSurfaceDesc const &surf = desc->surfaces[ctx.surface];
    int const level=p ? p->GetInt(TfToken("subdivisionLevel"),0) : 0;
    UsdGenLimitSurface limit;
    if(level) {
        std::string error;
        if(!limit.Build(surf,level,&error)) {
            if(diag)diag->Error("UsdGenScatter: "+error);
            return false;
        }
    }
    if (surf.faceVertexCounts.empty() || surf.restPoints.empty()) {
        if (diag) diag->Warn(std::string("UsdGenScatter::Capture: surface '") +
                             surf.path.GetText() + "' has no topology; 0 roots");
    }

    // Face list: whole mesh or the GeomSubset restriction (R15). An empty
    // subset selects no face. Unrestricted capture walks faces 0..F-1
    // directly (f == fi) instead of materializing the identity table.
    bool const restricted = UsdGenSurfaceRestricted(surf);
    std::vector<int> faces;
    if (restricted) faces.assign(surf.subsetFaces.cbegin(), surf.subsetFaces.cend());
    size_t const faceCount =
        restricted ? faces.size() : surf.faceVertexCounts.size();

    UsdGenScatterCapture &cap = *static_cast<UsdGenScatterCapture *>(out);
    UsdGenCurveBuffer &buf = cap.MutableBuffer();
    buf.topologyVersion += 1;

    auto const *rest = surf.restPoints.empty() ? nullptr : surf.restPoints.data();
    auto const *fvc = surf.faceVertexCounts.empty() ? nullptr : surf.faceVertexCounts.data();
    auto const *fvi = surf.faceVertexIndices.empty() ? nullptr : surf.faceVertexIndices.data();
    auto const *uv = surf.uv.empty() ? nullptr : surf.uv.data();
    if (!rest || !fvc || !fvi) {
        buf.totalCurves = 0;
        buf.totalCvs = 0;
        return true;  // the empty-topology warning above already fired
    }

    // faceVertexIndices are FACE-RELATIVE (02 §2.2): face f's corners are
    // fvi[cornerOff[f] .. cornerOff[f] + fvc[f]). Unrestricted faces run in
    // order, so the loop below accumulates the running base instead of the
    // table; the table is built only for subset (out-of-order) capture.
    std::vector<size_t> cornerOff;
    if (restricted) {
        cornerOff.assign(surf.faceVertexCounts.size() + 1, 0);
        for (size_t f = 0; f < surf.faceVertexCounts.size(); ++f)
            cornerOff[f + 1] = cornerOff[f] + size_t(fvc[f]);
    }

    const double density = p ? p->GetDouble(TfToken("density"), 100.0) : 100.0;
    if (!std::isfinite(density) || density < 0.0) {
        if (diag) diag->Error("UsdGenScatter::Capture: density must be finite and >= 0");
        return false;
    }
    // Per-face density scale from usdGen:paint:density (the brush); empty
    // == all 1.0. Sized once here; the loop below only range-checks values.
    bool const hasMult = !surf.densityMultiplier.empty();
    if (hasMult &&
        surf.densityMultiplier.size() != surf.faceVertexCounts.size()) {
        if (diag) diag->Error("UsdGenScatter::Capture: density multiplier "
                              "has the wrong face count");
        return false;
    }
    const bool flip = p ? p->GetBool(TfToken("flip"), false) : false;

    // Hoisted SplitMix seed hashes (bit-identical CSE): every per-face and
    // per-root hash below folds ctx.seed through UsdGenHash64(seed32, salt)
    // first, and ctx.seed is loop-invariant, so the seed folding is computed
    // once per Capture. Integer-only: no FP association changes anywhere.
    // The uint32_t(int(ctx.seed)) round-trip reproduces ctx.seed exactly, so
    // hSeedScatter matches both the faceKey spelling (which folds
    // uint32_t(ctx.seed)) and the CurveId/Draw01 spellings (which fold
    // uint32_t of the int seed).
    uint64_t const seed32 = uint64_t(uint32_t(int(ctx.seed)));
    uint64_t const hSeedScatter = UsdGenHash64(seed32, kSaltScatter);
    uint64_t const hSeedBary0 = UsdGenHash64(seed32, kSaltScatterBary);
    uint64_t const hSeedBary1 = UsdGenHash64(seed32, kSaltScatterBary + 1u);
    uint64_t const hSeedBary2 = UsdGenHash64(seed32, kSaltScatterBary + 2u);

    // Per-face area-weighted emission (plan/04 :634-635), over face ranges
    // that concatenate in face order: the merged pre-sort arrays are
    // identical for any partition count, so small captures (and the
    // subdivision path, whose OpenSubdiv evaluation has no thread-safety
    // proof) take one range while big level-0 captures spread over the
    // arena. Range width targets ~4K faces for load balance (per-face root
    // counts vary with area). Ranges never exceed workers (extra ranges
    // only queue), and never exceed 8: measured on 10 pinned cores, P=8
    // captures fastest (44.9ms vs 48.1 at P=64 — less slot/concat churn)
    // while P>=12 leaves every downstream serial Bake rep ~5ms slower
    // than P<=10 (45 flat vs 40 partial warm-up; serial warms to 33),
    // so oversubscribing ranges trades a capture wash for a bake loss.
    // One worker means one range: partitions would only add concat traffic
    // for a lone worker to copy back.
    int const workers = ctx.dispatcher ? ctx.dispatcher->MaxConcurrency()
                                       : tbb::this_task_arena::max_concurrency();
    size_t partitions = 1;
    if (level == 0 && faceCount > 4096 && workers > 1)
        partitions = std::min({size_t(workers), size_t(8),
                               (faceCount + 4095) / 4096});
    std::vector<size_t> rangeStart(partitions + 1);
    for (size_t p = 0; p <= partitions; ++p)
        rangeStart[p] = (p * faceCount) / partitions;
    // Unrestricted corner bases: the serial loop's runBase at fi is the
    // face-vertex prefix sum below fi. Each range sums its own fvc slice
    // on its worker and one serial prefix over the <= 8 range sums fixes
    // every range's starting base (P+1 stores, no 8MB table). Integer
    // addition is exact in any order, so the bases match the serial
    // spelling bit-for-bit, empty ranges included. Restricted capture
    // keeps reading cornerOff directly, as before.
    std::vector<size_t> rangeBase(partitions + 1, 0);
    if (partitions > 1 && !restricted) {
        std::vector<size_t> rangeSum(partitions, 0);
        ScatterParallelFor(ctx.dispatcher, partitions, [&](size_t p) {
            size_t s = 0;
            for (size_t fi = rangeStart[p]; fi < rangeStart[p + 1]; ++fi)
                s += size_t(fvc[fi]);
            rangeSum[p] = s;
        });
        for (size_t p = 0; p < partitions; ++p)
            rangeBase[p + 1] = rangeBase[p] + rangeSum[p];
    }
    // Shared level-0 fan triangulation for the two-pass path (pass 1 and
    // the pass-2 overflow recompute call the same code, so spilled and
    // recomputed weights agree by construction): the same inline
    // area-cross fusion as emitRange's fan below (kernels.cpp canonical,
    // same float ops in the same order). triW takes nFan = nc - 2
    // weights; the caller provides the stack/spill buffer.
    auto fanFace = [&](size_t cbase, int nc, float *triW,
                       GfVec3f *nAccOut, double *areaOut) {
        GfVec3f const p0 = rest[fvi[cbase]];
        double areaRest = 0.0;
        GfVec3f nAcc(0.0f, 0.0f, 0.0f);
        for (int t = 1; t + 1 < nc; ++t) {
            GfVec3f const pb = rest[fvi[cbase + size_t(t)]];
            GfVec3f const pc = rest[fvi[cbase + size_t(t) + 1]];
            GfVec3f const eb = pb - p0;
            GfVec3f const ec = pc - p0;
            GfVec3f const x = GfCross(eb, ec);
            float const w = 0.5f * std::sqrt(
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            triW[size_t(t) - 1] = w;
            areaRest += double(w);
            nAcc += x;
        }
        *nAccOut = nAcc;
        *areaOut = areaRest;
    };
    auto emitRange = [&](size_t fi0, size_t fi1, size_t cbase0,
                         ScatterEmission &e, ScatterEmitError &err) {
    e.reserve((fi1 - fi0) * 4);
    // Per-face triangle weights: stack-backed for the common small fan
    // (quads need 2), spilling to a reused vector past 64 triangles or on
    // the subdivision path. Same values, same order, no per-face vector
    // traffic on the hot path.
    float triStack[64];
    std::vector<float> triSpill;
    triSpill.reserve(16);
    size_t runBase = cbase0;
    for (size_t fi = fi0; fi < fi1; ++fi) {
        int const f = restricted ? faces[fi] : int(fi);
        if (f < 0 || size_t(f) >= surf.faceVertexCounts.size()) continue;  // bad subset
        // Running corner base (unrestricted): fvc[f] is valid here (the
        // range check passed), and this advance runs for every face —
        // ahead of the skips below — so runBase == cornerOff[fi] exactly.
        size_t cbase;
        if (restricted) cbase = cornerOff[f];
        else {
            cbase = runBase;
            runBase += size_t(fvc[f]);
        }
        if(level && limit.IsHole(f)) continue;
        int const nc = fvc[f];
        if (nc < 3) continue;
        bool bad = false;
        for (int i = 0; i < nc; ++i)
            if (fvi[cbase + size_t(i)] < 0 ||
                size_t(fvi[cbase + size_t(i)]) >= surf.restPoints.size()) { bad = true; break; }
        if (bad) continue;  // malformed face; skip rather than read out of bounds
        // Fan triangulation of face f (the same decomposition
        // UsdGenPolygonRestArea uses): triW[t] is the area of the triangle
        // rest[c0], rest[c(t+1)], rest[c(t+2)], nAcc accumulates the face
        // normal. areaRest(f) = UsdGenPolygonRestArea (plan/04 :634); the
        // float sum below equals it up to fp-associativity and is used for the
        // sampling weights so the cumulative weights sum exactly to the
        // denominator.
        GfVec3f const p0 = rest[fvi[cbase]];
        size_t const nFan = size_t(nc) - 2;
        float *triW = triStack;
        size_t ntri = nFan;
        if (nFan > 64) {
            triSpill.assign(nFan, 0.0f);
            triW = triSpill.data();
        }
        double areaRest = 0.0;
        GfVec3f nAcc(0.0f, 0.0f, 0.0f);
        for (int t = 1; t + 1 < nc; ++t) {
            GfVec3f const pb = rest[fvi[cbase + size_t(t)]];
            GfVec3f const pc = rest[fvi[cbase + size_t(t) + 1]];
            // Inline UsdGenTriangleArea (kernels.cpp is canonical) and fuse
            // it with the nAcc cross: the area IS 0.5*|cross|, so one cross
            // feeds both the weight and the normal accumulator. Same float
            // ops in the same order (GfVec3f::operator- and GfCross spell
            // the identical subtractions/products as kernels.cpp); the
            // bench + suite checksums pin the bits. Drops the cross-TU
            // call per fan triangle plus the duplicate cross. This TU
            // builds with default contraction, so any respelling here
            // must re-verify against kernels.cpp.
            GfVec3f const eb = pb - p0;
            GfVec3f const ec = pc - p0;
            GfVec3f const x = GfCross(eb, ec);
            float const w = 0.5f * std::sqrt(
                x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
            triW[size_t(t) - 1] = w;
            areaRest += double(w);
            nAcc += x;
        }
        GfVec3f const Nrest = Normalize3(nAcc);
        // Area quadrature uses a 2^level grid on each coarse patch. Actual
        // roots and frames are evaluated on the bicubic/Gregory limit patch,
        // never on these quadrature triangles. Keep coarse face IDs for Ptex.
        int const grid=level ? (1<<level) : 0;
        if(level) {
            triSpill.clear();areaRest=0;
            std::vector<GfVec3f> samples((grid+1)*(grid+1));
            GfVec3f du,dv;
            for(int y=0;y<=grid;++y) for(int x=0;x<=grid;++x)
                if(!limit.Evaluate(f,float(x)/grid,float(y)/grid,&samples[y*(grid+1)+x],&du,&dv)) {
                    err.failed = true; err.fi = fi;
                    err.message = "UsdGenScatter: limit patch evaluation failed";
                    break;
                }
            if (err.failed) break;
            for(int y=0;y<grid;++y) for(int x=0;x<grid;++x) {
                auto const& a=samples[y*(grid+1)+x];auto const& b=samples[y*(grid+1)+x+1];
                auto const& c=samples[(y+1)*(grid+1)+x+1];auto const& d=samples[(y+1)*(grid+1)+x];
                float a0=.5f*GfCross(b-a,c-a).GetLength(),a1=.5f*GfCross(c-a,d-a).GetLength();
                triSpill.push_back(a0);triSpill.push_back(a1);areaRest+=double(a0)+double(a1);
            }
            triW = triSpill.data(); ntri = triSpill.size();
        }

        double mult = 1.0;
        if (hasMult) {
            mult = double(surf.densityMultiplier[size_t(f)]);
            if (!std::isfinite(mult) || mult < 0.0) {
                err.failed = true; err.fi = fi;
                err.message = "UsdGenScatter::Capture: density "
                              "multiplier must be finite and >= 0";
                break;
            }
        }
        double const expected = density * areaRest * mult;
        if (!std::isfinite(areaRest) || !std::isfinite(expected)) {
            err.failed = true; err.fi = fi;
            err.message = "UsdGenScatter::Capture: non-finite root count on face " +
                          std::to_string(f);
            break;
        }
        if (!(expected > 0.0)) continue;
        double const whole = std::floor(expected);
        double const frac = expected - whole;
        if (whole > double(std::numeric_limits<uint32_t>::max())) {
            err.failed = true; err.fi = fi;
            err.message = "UsdGenScatter::Capture: root count exceeds uint32 "
                          "cardinality on face " + std::to_string(f);
            break;
        }
        // UsdGenHash64(seed32, f32, kSaltScatter) with the hoisted seed
        // fold: Hash64(Hash64(seed32, salt) ^ f32, salt), verbatim.
        uint64_t const faceKey = UsdGenHash64(
            hSeedScatter ^ uint64_t(uint32_t(f)), kSaltScatter);
        uint64_t const nf = uint64_t(whole)
                          + (UsdGenHash01(faceKey, kSaltScatter) < float(frac) ? 1u : 0u);
        size_t const maxCurves = std::numeric_limits<uint32_t>::max();
        if (e.aids.size() > maxCurves || nf > maxCurves - e.aids.size()) {
            err.failed = true; err.fi = fi;
            err.message = "UsdGenScatter::Capture: total root count exceeds "
                          "uint32 cardinality at face " + std::to_string(f);
            break;
        }
        for (uint64_t k = 0; k < nf; ++k) {
            // UsdGenCurveId(seed, f, k) is Hash64(faceKey ^ k, salt): faceKey
            // above is exactly the id's documented inner fold, so one
            // finalizer replaces three. The draws likewise reuse the hoisted
            // seed folds via the Hash01 spelling Draw01 expands to.
            uint64_t const curveId = UsdGenHash64(
                faceKey ^ uint64_t(uint32_t(k)), kSaltScatter);
            // "low-discrepancy sample seeded by curveId" (plan/04 :638),
            // realized in M1 as: area-weighted fan-triangle pick + uniform
            // in-triangle point, three draws keyed only by curveId.
            float const u0 = UsdGenHash01(hSeedBary0 ^ curveId, kSaltScatterBary);
            float const u1 = UsdGenHash01(hSeedBary1 ^ curveId, kSaltScatterBary + 1u);
            float const u2 = UsdGenHash01(hSeedBary2 ^ curveId, kSaltScatterBary + 2u);
            // Peeled quad pick (bit-identical): at ntri == 2 the loop
            // computes ti = (target < triW[0]) ? 0 : 1 (ti starts at 1;
            // t = 0 sets 0 or falls to t = 1, which sets 1 or leaves 1).
            // Same comparisons in the same order, same ti, without the
            // data-dependent trip count. Other fan sizes keep the loop.
            double const target = double(u0) * areaRest;
            size_t ti;
            if (ntri == 2) {
                ti = (target < double(triW[0])) ? 0 : 1;
            } else {
                ti = ntri - 1;
                double cum = 0.0;
                for (size_t t = 0; t < ntri; ++t) {
                    cum += double(triW[t]);
                    if (target < cum) { ti = t; break; }
                }
            }
            GfVec3f pos,T,N=Nrest;
            GfVec2f puv;
            if(level) {
                int const cell=int(ti/2),x=cell%grid,y=cell/grid;
                GfVec2f a(float(x)/grid,float(y)/grid);
                GfVec2f b(float(x+1)/grid,float(y+(ti%2 ? 1:0))/grid);
                GfVec2f c(float(x+(ti%2 ? 0:1))/grid,float(y+1)/grid);
                float const r=std::sqrt(u1);
                puv=a*(1-r)+b*(u2*r)+c*((1-u2)*r);
                GfVec3f du,dv;
                if(!limit.Evaluate(f,puv[0],puv[1],&pos,&du,&dv)) {
                    err.failed = true; err.fi = fi;
                    break;
                }
                N=Normalize3(GfCross(du,dv));
                if(surf.orientation==TfToken("leftHanded")) N=-N;
                T=Normalize3(du-N*GfDot(du,N));
            } else {
            size_t const ib = cbase + ti + 1;
            size_t const ic = ib + 1;
            GfVec3f const pb = rest[fvi[ib]];
            GfVec3f const pc = rest[fvi[ic]];
            float const r1 = std::sqrt(u1);
            pos = p0 * (1.0f - r1) + pb * (u2 * r1) + pc * (r1 * (1.0f - u2));
            puv=GfVec2f(1.0f / 3.0f, 1.0f / 3.0f);
            if (uv)
                puv = uv[fvi[cbase]] * (1.0f - r1)
                    + uv[fvi[ib]] * (u2 * r1)
                    + uv[fvi[ic]] * (r1 * (1.0f - u2));
            // orthonormal frame: N_rest from the fan, T from a triangle
            // edge Gram-Schmidt'd against N (plan/04 :639). Replaces the M0
            // |dot(e0,N)| > 0.5 heuristic, which produced non-orthogonal
            // frames for near-axis-aligned faces.
            GfVec3f e0 = pb - p0;
            // Squared parallel-edge test (saves a sqrt plus a re-dot per
            // root): |dot| > 0.9*|e0| squares to dot*dot > 0.81*lenSq on
            // the exact float dots, and the keep path reuses the tested
            // dot for the Gram-Schmidt projection instead of re-dotting.
            // Matches except within rounding of the boundary (same class
            // as the r7 tangent test): a standalone differential over 20M
            // random + 20M boundary-aimed edges shows 0 flips random and
            // 1 flip aimed (window ~1e-7 relative), and the bench + suite
            // checksums pin the production meshes.
            float const e0d = GfDot(e0, Nrest);
            float const e0l2 = GfDot(e0, e0);
            if (e0d * e0d > 0.81f * e0l2) {
                e0 = pc - p0;  // e0 too normal-parallel; try the other edge
                T = e0 - Nrest * GfDot(e0, Nrest);
            } else {
                T = e0 - Nrest * e0d;
            }
            // Fuse the degenerate-length check with the normalize: T's
            // GetLengthSq is (T*T) and GetLength is its sqrt, so one
            // sqrt feeds both the 1e-9f check and the division. The hot
            // path (len >= 1e-9f) takes Normalize3's division branch
            // (len is far above its 1e-12f floor) with the identical
            // divisor; the cold path runs the original spelling verbatim.
            float tLen = GfSqrt(T * T);
            if (tLen < 1e-9f) {
                T = std::abs(Nrest[0]) > 0.9f ? GfVec3f(0.0f, 1.0f, 0.0f)
                                             : GfVec3f(1.0f, 0.0f, 0.0f);
                T = T - Nrest * GfDot(T, Nrest);
                T = Normalize3(T);
            } else {
                T = T / tLen;
            }
            }
            GfVec3f B = Normalize3(GfCross(N, T));
            if (flip) { T = -T; B = -B; }  // rest frame handedness (02 §2.6)
            e.ax.push_back(pos[0]); e.ay.push_back(pos[1]); e.az.push_back(pos[2]);
            e.aids.push_back(curveId);
            e.aPrim.push_back(f);
            e.aUv.push_back(puv);
            e.aT.push_back(T); e.aN.push_back(N); e.aB.push_back(B);
        }
        if (err.failed) break;
    }
    };
    // Pass 1 of the two-pass path (level 0 only): per-face validation,
    // fan triangulation and root-count spill. Mirrors emitRange's face
    // loop exactly (same checks, same messages, same skips) but writes a
    // ScatterSpill per face instead of emitting roots: the spill slice is
    // pre-sized and resize-zeroed, so skipped faces keep zeros (nf == 0)
    // and valid faces overwrite their slot in full. The per-range root
    // total accumulates for the inter-range prefix; the uint32 overflow
    // check runs globally after the join (exact first-crossing face).
    auto pass1Range = [&](size_t fi0, size_t fi1, size_t cbase0,
                          ScatterSpill *spill,
                          size_t *rangeTotal, ScatterEmitError &err) {
    float triStack[64];
    std::vector<float> triSpill;
    triSpill.reserve(16);
    // Explicit zero spill for skipped faces (the slice is uninitialized).
    auto skipFace = [&](size_t fi) { spill[fi - fi0] = ScatterSpill{}; };
    size_t runBase = cbase0;
    size_t rangeN = 0;
    for (size_t fi = fi0; fi < fi1; ++fi) {
        int const f = restricted ? faces[fi] : int(fi);
        if (f < 0 || size_t(f) >= surf.faceVertexCounts.size()) { skipFace(fi); continue; }
        size_t cbase;
        if (restricted) cbase = cornerOff[f];
        else {
            cbase = runBase;
            runBase += size_t(fvc[f]);
        }
        int const nc = fvc[f];
        if (nc < 3) { skipFace(fi); continue; }
        bool bad = false;
        for (int i = 0; i < nc; ++i)
            if (fvi[cbase + size_t(i)] < 0 ||
                size_t(fvi[cbase + size_t(i)]) >= surf.restPoints.size()) { bad = true; break; }
        if (bad) { skipFace(fi); continue; }
        size_t const nFan = size_t(nc) - 2;
        float *triW = triStack;
        if (nFan > 64) {
            triSpill.assign(nFan, 0.0f);
            triW = triSpill.data();
        }
        GfVec3f nAcc(0.0f, 0.0f, 0.0f);
        double areaRest = 0.0;
        fanFace(cbase, nc, triW, &nAcc, &areaRest);
        GfVec3f const Nrest = Normalize3(nAcc);
        double mult = 1.0;
        if (hasMult) {
            mult = double(surf.densityMultiplier[size_t(f)]);
            if (!std::isfinite(mult) || mult < 0.0) {
                err.failed = true; err.fi = fi;
                err.message = "UsdGenScatter::Capture: density "
                              "multiplier must be finite and >= 0";
                break;
            }
        }
        double const expected = density * areaRest * mult;
        if (!std::isfinite(areaRest) || !std::isfinite(expected)) {
            err.failed = true; err.fi = fi;
            err.message = "UsdGenScatter::Capture: non-finite root count on face " +
                          std::to_string(f);
            break;
        }
        if (!(expected > 0.0)) { skipFace(fi); continue; }
        double const whole = std::floor(expected);
        double const frac = expected - whole;
        if (whole > double(std::numeric_limits<uint32_t>::max())) {
            err.failed = true; err.fi = fi;
            err.message = "UsdGenScatter::Capture: root count exceeds uint32 "
                          "cardinality on face " + std::to_string(f);
            break;
        }
        uint64_t const faceKey = UsdGenHash64(
            hSeedScatter ^ uint64_t(uint32_t(f)), kSaltScatter);
        uint64_t const nf = uint64_t(whole)
                          + (UsdGenHash01(faceKey, kSaltScatter) < float(frac) ? 1u : 0u);
        ScatterSpill s;
        s.areaRest = areaRest;
        s.nrest = Nrest;
        s.nf = uint32_t(nf);
        if (nFan <= 2) {
            s.w0 = triW[0];
            s.w1 = nFan > 1 ? triW[1] : 0.0f;
        } else {
            s.w0 = 0.0f;
            s.w1 = 0.0f;
        }
        spill[fi - fi0] = s;
        rangeN += uint64_t(nf);
        if (err.failed) break;
    }
    *rangeTotal = rangeN;
    };
    // Pass 2 of the two-pass path (level 0 only): root sampling with
    // direct-indexed writes. The per-root computation is emitRange's
    // level-0 root loop verbatim (same draws, same pick, same blends,
    // same frame tests); only the sink differs (indexed stores into the
    // pre-sized raw planes instead of push_backs). Faces read their fan
    // outputs from the pass-1 spill; overflow fans (ntri > 2) recompute
    // triW from the same fanFace code. cbase/runBase advance exactly as
    // in pass 1 (every in-range face), and the cursor partitions
    // [cursor0, cursor0 + rangeTotal), so every slot is written once.
    auto pass2Range = [&](size_t fi0, size_t fi1, size_t cbase0,
                          size_t cursor0,
                          ScatterSpill const *spill,
                          ScatterRawRoots &roots) {
    float *ax = roots.ax.get(), *ay = roots.ay.get(), *az = roots.az.get();
    uint64_t *aids = roots.aids.get();
    int *aPrim = roots.aPrim.get();
    GfVec2f *aUv = roots.aUv.get();
    GfVec3f *aT = roots.aT.get(), *aN = roots.aN.get(), *aB = roots.aB.get();
    float triStack[64];
    std::vector<float> triSpill;
    triSpill.reserve(16);
    size_t runBase = cbase0;
    size_t cursor = cursor0;
    for (size_t fi = fi0; fi < fi1; ++fi) {
        int const f = restricted ? faces[fi] : int(fi);
        if (f < 0 || size_t(f) >= surf.faceVertexCounts.size()) continue;
        size_t cbase;
        if (restricted) cbase = cornerOff[f];
        else {
            cbase = runBase;
            runBase += size_t(fvc[f]);
        }
        ScatterSpill const &s = spill[fi - fi0];
        if (s.nf == 0) continue;
        // Valid spilled faces passed pass 1's nc >= 3 filter, so ntri >= 1.
        size_t const ntri = size_t(fvc[f]) - 2;
        float triLocal[2];
        float *triW = triLocal;
        if (ntri <= 2) {
            triLocal[0] = s.w0;
            triLocal[1] = s.w1;
        } else {
            if (ntri > 64) {
                triSpill.assign(ntri, 0.0f);
                triW = triSpill.data();
            } else {
                triW = triStack;
            }
            // Recomputed for triW only (areaRest/Nrest come from the
            // spill); same code as pass 1, so identical values.
            GfVec3f nAccIgnored(0.0f, 0.0f, 0.0f);
            double areaIgnored = 0.0;
            fanFace(cbase, fvc[f], triW, &nAccIgnored, &areaIgnored);
        }
        GfVec3f const p0 = rest[fvi[cbase]];
        GfVec3f const Nrest = s.nrest;
        double const areaRest = s.areaRest;
        uint64_t const faceKey = UsdGenHash64(
            hSeedScatter ^ uint64_t(uint32_t(f)), kSaltScatter);
        for (uint64_t k = 0; k < s.nf; ++k) {
            uint64_t const curveId = UsdGenHash64(
                faceKey ^ uint64_t(uint32_t(k)), kSaltScatter);
            float const u0 = UsdGenHash01(hSeedBary0 ^ curveId, kSaltScatterBary);
            float const u1 = UsdGenHash01(hSeedBary1 ^ curveId, kSaltScatterBary + 1u);
            float const u2 = UsdGenHash01(hSeedBary2 ^ curveId, kSaltScatterBary + 2u);
            // Peeled quad pick (bit-identical): at ntri == 2 the loop
            // computes ti = (target < triW[0]) ? 0 : 1 (ti starts at 1;
            // t = 0 sets 0 or falls to t = 1, which sets 1 or leaves 1).
            // Same comparisons in the same order, same ti, without the
            // data-dependent trip count. Other fan sizes keep the loop.
            double const target = double(u0) * areaRest;
            size_t ti;
            if (ntri == 2) {
                ti = (target < double(triW[0])) ? 0 : 1;
            } else {
                ti = ntri - 1;
                double cum = 0.0;
                for (size_t t = 0; t < ntri; ++t) {
                    cum += double(triW[t]);
                    if (target < cum) { ti = t; break; }
                }
            }
            GfVec3f pos, T, N = Nrest;
            GfVec2f puv;
            size_t const ib = cbase + ti + 1;
            size_t const ic = ib + 1;
            GfVec3f const pb = rest[fvi[ib]];
            GfVec3f const pc = rest[fvi[ic]];
            float const r1 = std::sqrt(u1);
            pos = p0 * (1.0f - r1) + pb * (u2 * r1) + pc * (r1 * (1.0f - u2));
            puv = GfVec2f(1.0f / 3.0f, 1.0f / 3.0f);
            if (uv)
                puv = uv[fvi[cbase]] * (1.0f - r1)
                    + uv[fvi[ib]] * (u2 * r1)
                    + uv[fvi[ic]] * (r1 * (1.0f - u2));
            GfVec3f e0 = pb - p0;
            float const e0d = GfDot(e0, Nrest);
            float const e0l2 = GfDot(e0, e0);
            if (e0d * e0d > 0.81f * e0l2) {
                e0 = pc - p0;
                T = e0 - Nrest * GfDot(e0, Nrest);
            } else {
                T = e0 - Nrest * e0d;
            }
            float tLen = GfSqrt(T * T);
            if (tLen < 1e-9f) {
                T = std::abs(Nrest[0]) > 0.9f ? GfVec3f(0.0f, 1.0f, 0.0f)
                                             : GfVec3f(1.0f, 0.0f, 0.0f);
                T = T - Nrest * GfDot(T, Nrest);
                T = Normalize3(T);
            } else {
                T = T / tLen;
            }
            GfVec3f B = Normalize3(GfCross(N, T));
            if (flip) { T = -T; B = -B; }
            size_t const w = cursor + size_t(k);
            ax[w] = pos[0]; ay[w] = pos[1]; az[w] = pos[2];
            aids[w] = curveId;
            aPrim[w] = f;
            aUv[w] = puv;
            aT[w] = T; aN[w] = N; aB[w] = B;
        }
        cursor += s.nf;
    }
    };
    // Big level-0 captures (past the threading threshold, restricted or
    // not) take the two-pass path: pass 1 spills per-face fan outputs +
    // root counts, a serial prefix over the <= 8 range totals sizes the
    // merged planes exactly, and pass 2 samples roots with direct-indexed
    // writes. No push_back growth, no 204MB concat: the same roots land
    // in the same pre-sort slots either way. Small captures and the
    // subdivision path keep the one-pass emission below.
    bool const twoPass = (level == 0 && faceCount > 4096);
    ScatterEmission all;
    // Pooled two-pass storage (see above): the spill slice and raw planes
    // alias the thread's emission scratch, grown (never shrunk) to the
    // largest capture seen. Unused by the one-pass path.
    ScatterEmissionScratch &emis = t_scatterEmissionScratch;
    ScatterRawRoots &raw = emis.raw;
    size_t total = 0;
    if (twoPass) {
        if (emis.spillFaces < faceCount) {
            emis.spill.reset(new ScatterSpill[faceCount]);
            emis.spillFaces = faceCount;
        }
        ScatterSpill *spillBase = emis.spill.get();
        std::vector<ScatterSpillSlot> spillSlots(partitions);
        if (partitions == 1) {
            pass1Range(rangeStart[0], rangeStart[1], 0,
                       spillBase + rangeStart[0], &spillSlots[0].total,
                       spillSlots[0].error);
        } else {
            ScatterParallelFor(ctx.dispatcher, partitions, [&](size_t p) {
                pass1Range(rangeStart[p], rangeStart[p + 1],
                           restricted ? 0 : rangeBase[p],
                           spillBase + rangeStart[p], &spillSlots[p].total,
                           spillSlots[p].error);
            });
        }
        // Ranges are face-ordered, so the first failed range holds the
        // serial loop's first failure: same return, same message.
        for (size_t p = 0; p < partitions; ++p) {
            if (!spillSlots[p].error.failed)
                continue;
            if (diag && !spillSlots[p].error.message.empty())
                diag->Error(spillSlots[p].error.message);
            return false;
        }
        std::vector<size_t> rangeRootBase(partitions + 1, 0);
        for (size_t p = 0; p < partitions; ++p)
            rangeRootBase[p + 1] =
                rangeRootBase[p] + spillSlots[p].total;
        total = rangeRootBase[partitions];
        size_t const maxTotal = std::numeric_limits<uint32_t>::max();
        if (total > maxTotal) {
            // Exact first-crossing face: the serial loop trips at the
            // first face whose roots would overflow uint32 cardinality.
            // (Unhittable in practice: 4G+ roots need 272GB+ of planes.)
            int face = 0;
            size_t acc = 0;
            for (size_t p = 0; p < partitions; ++p) {
                bool found = false;
                for (size_t fi = rangeStart[p]; fi < rangeStart[p + 1];
                     ++fi) {
                    size_t const nf = spillBase[fi].nf;
                    if (acc > maxTotal || nf > maxTotal - acc) {
                        face = restricted ? faces[fi] : int(fi);
                        found = true;
                        break;
                    }
                    acc += nf;
                }
                if (found) break;
            }
            if (diag) diag->Error("UsdGenScatter::Capture: total root count exceeds "
                                  "uint32 cardinality at face " + std::to_string(face));
            return false;
        }
        // Pooled uninitialized sizing: pass 2 writes every slot exactly
        // once, so growth needs no initialization and reuse is exact.
        if (emis.rawTotal < total) {
            raw.ax.reset(new float[total]);
            raw.ay.reset(new float[total]);
            raw.az.reset(new float[total]);
            raw.aids.reset(new uint64_t[total]);
            raw.aPrim.reset(new int[total]);
            raw.aUv.reset(new GfVec2f[total]);
            raw.aT.reset(new GfVec3f[total]);
            raw.aN.reset(new GfVec3f[total]);
            raw.aB.reset(new GfVec3f[total]);
            emis.rawTotal = total;
        }
        if (partitions == 1) {
            pass2Range(rangeStart[0], rangeStart[1], 0, rangeRootBase[0],
                       spillBase + rangeStart[0], raw);
        } else {
            ScatterParallelFor(ctx.dispatcher, partitions, [&](size_t p) {
                pass2Range(rangeStart[p], rangeStart[p + 1],
                           restricted ? 0 : rangeBase[p], rangeRootBase[p],
                           spillBase + rangeStart[p], raw);
            });
        }
        // The pooled spill and raw planes stay resident (no release):
        // they are cold during the sort/gather legs and hot again on the
        // next Capture, so releasing would only re-buy faults.
    } else {
        std::vector<ScatterEmitSlot> slots(partitions);
        // Reserve every slot up front, serially: 72 near-simultaneous mmaps
        // from 8 workers contend on the address-space lock (~6ms), while one
        // thread mapping the same total pays ~1ms. The per-range reserve inside
        // emitRange then no-ops (capacity already suffices).
        for (size_t p = 0; p < partitions; ++p)
            slots[p].emission.reserve(
                (rangeStart[p + 1] - rangeStart[p]) * 4);
        if (partitions == 1) {
            emitRange(rangeStart[0], rangeStart[1], 0,
                      slots[0].emission, slots[0].error);
        } else {
            ScatterParallelFor(ctx.dispatcher, partitions, [&](size_t p) {
                emitRange(rangeStart[p], rangeStart[p + 1],
                          restricted ? 0 : rangeBase[p],
                          slots[p].emission, slots[p].error);
            });
        }
        // Ranges are face-ordered, so the first failed range holds the serial
        // loop's first failure: same return, same message.
        for (size_t p = 0; p < partitions; ++p) {
            if (!slots[p].error.failed)
                continue;
            if (diag && !slots[p].error.message.empty())
                diag->Error(slots[p].error.message);
            return false;
        }
        for (auto const &s : slots)
            total += s.emission.size();
        size_t const maxTotal = std::numeric_limits<uint32_t>::max();
        if (total > maxTotal) {
            // The serial loop names the exact trip face, which needs per-face
            // counts (4MB kept for a >2^32-root path no process survives —
            // the emission vectors alone need 292GB+), so name the last face
            // of the first range whose end total overflows instead. Same
            // failure, same message shape; only the named face can differ.
            int face = 0;
            size_t acc = 0;
            for (size_t p = 0; p < partitions; ++p) {
                acc += slots[p].emission.size();
                if (acc > maxTotal) {
                    size_t const fiFace =
                        rangeStart[p + 1] ? rangeStart[p + 1] - 1 : 0;
                    face = restricted ? faces[fiFace] : int(fiFace);
                    break;
                }
            }
            if (diag) diag->Error("UsdGenScatter::Capture: total root count exceeds "
                                  "uint32 cardinality at face " + std::to_string(face));
            return false;
        }
        if (partitions == 1) {
            slots[0].emission.moveTo(all);
        } else if (total > 0) {
            std::vector<size_t> segBase(partitions + 1, 0);
            for (size_t p = 0; p < partitions; ++p)
                segBase[p + 1] = segBase[p] + slots[p].emission.size();
            ScatterEmitSlot *slotData = slots.data();
            size_t const *baseData = segBase.data();
            ScatterParallelFor(ctx.dispatcher, 9, [&](size_t i) {
                switch (i) {
                case 0: ScatterCopyPlane(&ScatterEmission::ax, all, slotData, baseData, partitions, total); break;
                case 1: ScatterCopyPlane(&ScatterEmission::ay, all, slotData, baseData, partitions, total); break;
                case 2: ScatterCopyPlane(&ScatterEmission::az, all, slotData, baseData, partitions, total); break;
                case 3: ScatterCopyPlane(&ScatterEmission::aids, all, slotData, baseData, partitions, total); break;
                case 4: ScatterCopyPlane(&ScatterEmission::aPrim, all, slotData, baseData, partitions, total); break;
                case 5: ScatterCopyPlane(&ScatterEmission::aUv, all, slotData, baseData, partitions, total); break;
                case 6: ScatterCopyPlane(&ScatterEmission::aT, all, slotData, baseData, partitions, total); break;
                case 7: ScatterCopyPlane(&ScatterEmission::aN, all, slotData, baseData, partitions, total); break;
                default: ScatterCopyPlane(&ScatterEmission::aB, all, slotData, baseData, partitions, total); break;
                }
            });
        }
        // Release the segments before the sort/gather legs: slots and the merged
        // arrays hold the same roots twice (~136MB live), and the downstream
        // passes are LLC-sensitive.
        slots.clear();
        slots.shrink_to_fit();
    }
    // Merged pre-sort planes from either emission path: vectors for the
    // one-pass concat, raw buffers for two-pass. Downstream only reads.
    float const *ax = nullptr, *ay = nullptr, *az = nullptr;
    uint64_t const *aids = nullptr;
    int const *aPrim = nullptr;
    GfVec2f const *aUv = nullptr;
    GfVec3f const *aT = nullptr, *aN = nullptr, *aB = nullptr;
    if (twoPass) {
        ax = raw.ax.get(); ay = raw.ay.get(); az = raw.az.get();
        aids = raw.aids.get();
        aPrim = raw.aPrim.get();
        aUv = raw.aUv.get();
        aT = raw.aT.get(); aN = raw.aN.get(); aB = raw.aB.get();
    } else {
        ax = all.ax.data(); ay = all.ay.data(); az = all.az.data();
        aids = all.aids.data();
        aPrim = all.aPrim.data();
        aUv = all.aUv.data();
        aT = all.aT.data(); aN = all.aN.data(); aB = all.aB.data();
    }

    // Morton-sort the roots by rest position: surface-major, locality-
    // preserving chunk placement (plan/04 :641, ADR §4.1). cellScale 64
    // cells/unit is an M1 choice — the plan does not pin a cell size; ties
    // keep the deterministic face-major emission order (stable passes), so
    // the result is bit-stable across thread counts and chunk splits (E-8).
    // LSD radix over 16-bit digits: each pass is stable and the digits run
    // least- to most-significant, so the final order is exactly the
    // stable_sort order (primary morton key, ties in emission order) at
    // linear cost. The even pass count lands the result back in the
    // morton/order pair.
    const size_t N = total;
    ScatterSortScratch &ss = t_scatterSortScratch;
    std::vector<uint64_t> &morton = ss.morton;
    morton.resize(N);
    // Indices and digit populations are 32-bit: the emission loop caps the
    // root total at uint32 cardinality, so every index and every population
    // is below 2^32. Halves the permutation traffic of the sort.
    std::vector<uint32_t> &order = ss.order;
    order.resize(N);
    // Morton keys + the varying-bits reduction, chunked over workers for
    // big captures (level-independent: this runs on the merged arrays).
    // Writes are disjoint and the |/& reductions are order-free, so any
    // chunking is bit-identical. Small captures stay serial: below ~32K
    // roots the dispatch costs more than the keys.
    uint64_t orKeys = 0, andKeys = ~uint64_t(0);
    size_t const mortonChunks =
        (workers > 1 && N > 32768) ? std::min({size_t(workers), size_t(8), N})
                                   : 1;
    if (mortonChunks == 1) {
        for (size_t i = 0; i < N; ++i) {
            uint64_t const key = ScatterMortonKey3(ax[i], ay[i], az[i], 64.0f);
            morton[i] = key;
            order[i] = uint32_t(i);
            orKeys |= key;
            andKeys &= key;
        }
    } else {
        uint64_t *mortonOut = morton.data();
        uint32_t *orderOut = order.data();
        float const *px = ax, *py = ay, *pz = az;
        struct MortonChunk {
            uint64_t orKeys = 0, andKeys = ~uint64_t(0);
        };
        std::vector<MortonChunk> partial(mortonChunks);
        ScatterParallelFor(ctx.dispatcher, mortonChunks, [&](size_t c) {
            size_t const i0 = (c * N) / mortonChunks;
            size_t const i1 = ((c + 1) * N) / mortonChunks;
            uint64_t orLocal = 0, andLocal = ~uint64_t(0);
            for (size_t i = i0; i < i1; ++i) {
                uint64_t const key =
                    ScatterMortonKey3(px[i], py[i], pz[i], 64.0f);
                mortonOut[i] = key;
                orderOut[i] = uint32_t(i);
                orLocal |= key;
                andLocal &= key;
            }
            partial[c].orKeys = orLocal;
            partial[c].andKeys = andLocal;
        });
        for (auto const &p : partial) {
            orKeys |= p.orKeys;
            andKeys &= p.andKeys;
        }
    }
    uint32_t const *perm = order.data();
    // Scratch pair at function scope: an odd surviving-pass count leaves the
    // permutation in tmpIdx, which the gather below still reads.
    std::vector<uint64_t> &tmpKeys = ss.tmpKeys;
    std::vector<uint32_t> &tmpIdx = ss.tmpIdx;
    if (N > 1) {
        tmpKeys.resize(N);
        tmpIdx.resize(N);
        std::vector<uint32_t> &counts = ss.counts;
        counts.resize(65536);
        uint64_t *keys = morton.data();
        uint32_t *idx = order.data();
        uint64_t *keysOut = tmpKeys.data();
        uint32_t *idxOut = tmpIdx.data();
        // A counting pass whose digit is constant across all keys is the
        // identity permutation (every element scatters to its own slot, in
        // input order), so only varying digits run. orKeys ^ andKeys marks
        // exactly the bit positions where some key pair differs; the
        // surviving passes still run least- to most-significant, so the
        // final order is exactly the 4-pass order. The permutation lands in
        // whichever index array the last surviving pass wrote.
        uint64_t const vary = orKeys ^ andKeys;
        // Parallel stable counting passes for big captures (same N>32768
        // rule as the morton keys): each chunk histograms its own input
        // range, one serial prefix over the chunk-major histograms assigns
        // every (digit, chunk) pair a disjoint output range in chunk order,
        // and each chunk scatters its range in input order. Within a digit
        // the landing order is chunk order then input order — exactly the
        // serial scatter's global input order — so any chunking is
        // bit-identical. Small captures keep the serial spelling below.
        size_t const sortChunks =
            (workers > 1 && N > 32768)
                ? std::min({size_t(workers), size_t(8), N})
                : 1;
        std::vector<uint32_t> *parCounts = nullptr, *parOff = nullptr;
        if (sortChunks > 1) {
            parCounts = &ss.sortCounts;
            parOff = &ss.sortOff;
            parCounts->resize(sortChunks * 65536);
            parOff->resize(sortChunks * 65536);
        }
        for (int pass = 0; pass < 4; ++pass) {
            int const shift = pass * 16;
            if (((vary >> shift) & 0xffffu) == 0)
                continue;
            if (sortChunks == 1) {
                std::fill(counts.begin(), counts.end(), uint32_t(0));
                for (size_t i = 0; i < N; ++i)
                    ++counts[(keys[i] >> shift) & 0xffffu];
                uint32_t sum = 0;
                for (size_t c = 0; c < 65536; ++c) {
                    uint32_t const t = counts[c];
                    counts[c] = sum;
                    sum += t;
                }
                for (size_t i = 0; i < N; ++i) {
                    size_t const d = (keys[i] >> shift) & 0xffffu;
                    uint32_t const p = counts[d]++;
                    keysOut[p] = keys[i];
                    idxOut[p] = idx[i];
                }
            } else {
                uint32_t *cntBase = parCounts->data();
                uint32_t *offBase = parOff->data();
                std::fill(cntBase, cntBase + sortChunks * 65536, uint32_t(0));
                uint64_t const *keysIn = keys;
                ScatterParallelFor(ctx.dispatcher, sortChunks, [&](size_t c) {
                    size_t const i0 = (c * N) / sortChunks;
                    size_t const i1 = ((c + 1) * N) / sortChunks;
                    uint32_t *cnt = cntBase + c * 65536;
                    for (size_t i = i0; i < i1; ++i)
                        ++cnt[(keysIn[i] >> shift) & 0xffffu];
                });
                // Parallel prefix over digit ranges (bit-identical): the
                // serial form is a running sum over the (digit, chunk)
                // table in digit-major order, and every prefix value sits
                // below 2^32 (root totals cap at uint32 cardinality), so
                // plain integer associativity lets each worker prefix its
                // own digit range from that range's base. Phase 1 sums
                // each range's total on workers, one <=8-element serial
                // prefix fixes the bases, and phase 2 writes every
                // range's offsets from its base. Same values in the same
                // slots at any worker count.
                size_t const prefixRanges = sortChunks;
                uint32_t rangeTotal[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                uint32_t *rangeTotalOut = rangeTotal;
                uint32_t const *cntIn = cntBase;
                ScatterParallelFor(ctx.dispatcher, prefixRanges, [&](size_t w) {
                    size_t const d0 = (w * 65536) / prefixRanges;
                    size_t const d1 = ((w + 1) * 65536) / prefixRanges;
                    uint32_t acc = 0;
                    for (size_t d = d0; d < d1; ++d)
                        for (size_t c = 0; c < sortChunks; ++c)
                            acc += cntIn[c * 65536 + d];
                    rangeTotalOut[w] = acc;
                });
                uint32_t rangeOff[8];
                rangeOff[0] = 0;
                for (size_t w = 1; w < prefixRanges; ++w)
                    rangeOff[w] = rangeOff[w - 1] + rangeTotal[w - 1];
                uint32_t *offOut = offBase;
                uint32_t const *rbOff = rangeOff;
                ScatterParallelFor(ctx.dispatcher, prefixRanges, [&](size_t w) {
                    size_t const d0 = (w * 65536) / prefixRanges;
                    size_t const d1 = ((w + 1) * 65536) / prefixRanges;
                    uint32_t sum = rbOff[w];
                    for (size_t d = d0; d < d1; ++d)
                        for (size_t c = 0; c < sortChunks; ++c) {
                            size_t const s = c * 65536 + d;
                            uint32_t const t = cntIn[s];
                            offOut[s] = sum;
                            sum += t;
                        }
                });
                uint32_t const *idxIn = idx;
                uint64_t *kOut = keysOut;
                uint32_t *iOut = idxOut;
                ScatterParallelFor(ctx.dispatcher, sortChunks, [&](size_t c) {
                    size_t const i0 = (c * N) / sortChunks;
                    size_t const i1 = ((c + 1) * N) / sortChunks;
                    uint32_t *off = offBase + c * 65536;
                    for (size_t i = i0; i < i1; ++i) {
                        size_t const d = (keysIn[i] >> shift) & 0xffffu;
                        uint32_t const p = off[d]++;
                        kOut[p] = keysIn[i];
                        iOut[p] = idxIn[i];
                    }
                });
            }
            std::swap(keys, keysOut);
            std::swap(idx, idxOut);
        }
        perm = idx;
    }

    buf.totalCurves = uint32_t(N);
    buf.totalCvs = uint32_t(N);
    // Per-plane gather (bit-identical): the permuted read stream jumps over
    // the whole emission range, so gathering every plane in one loop keeps
    // ~68MB of random-read working set live and thrashes the LLC. One plane
    // per pass shrinks the random window to a single 4-12MB array (the
    // sequential perm re-read stays cache-hot); writes stay sequential in
    // both forms. Same bytes in the same slots.
    // Uninitialized sizing (bit-identical): VtArray(N) value-initializes
    // (~68MB of zeroes here) that the gather overwrites in full, so each
    // plane sizes through resize(N, fill) instead, whose filler runs over
    // uninitialized storage. clear() first keeps a reused capture exact:
    // resize alone is a no-op at equal size and would leave stale data.
    // Every plane is trivially copyable, so placement-new is a plain store.
    // Chunked over workers for big captures (same N>32768 rule as the
    // morton keys): every output slot is independent (disjoint writes,
    // shared-readonly perm), so any chunking writes identical bytes.
    size_t const gatherChunks =
        (workers > 1 && N > 32768) ? std::min({size_t(workers), size_t(8), N})
                                   : 1;
    UsdGenWorkDispatcher *gatherDispatcher = ctx.dispatcher;
    auto gather = [perm, N](auto &arr, auto const *src) {
        using T = typename std::decay_t<decltype(arr)>::value_type;
        arr.clear();
        arr.resize(N, [perm, src](T *b, T *e) {
            size_t i = 0;
            for (T *d = b; d != e; ++d, ++i)
                new (d) T(src[perm[i]]);
        });
    };
    if (gatherChunks == 1) {
        gather(buf.px, ax);
        gather(buf.py, ay);
        gather(buf.pz, az);
        gather(buf.curveId, aids);
        gather(buf.rootPrim, aPrim);
        gather(buf.rootUV, aUv);
        gather(buf.rootT, aT);
        gather(buf.rootN, aN);
        gather(buf.rootB, aB);
        // Same chunking for the zero plane: clear + uninitialized resize
        // with a zero filler (identical bytes to VtFloatArray(N, 0.0f)).
        buf.hairT.clear();
        buf.hairT.resize(N, [](float *b, float *e) {
            for (float *d = b; d != e; ++d)
                new (d) float(0.0f);
        });
        return true;
    }
    // Fused parallel gather (bit-identical): the nine planes plus the zero
    // plane share one dispatch instead of ten. Each worker gathers its own
    // row range plane by plane (planes outer, rows inner), so the random
    // window of each inner pass is still one plane, exactly as above; only
    // the nine dispatches and barriers are gone, and each worker's perm
    // chunk stays L2-hot across its ten inner passes. Every output slot is
    // an independent function of shared-readonly inputs into a disjoint
    // lane, so any chunking writes the same bytes.
    auto sizeNoInit = [N](auto &arr) {
        using T = typename std::decay_t<decltype(arr)>::value_type;
        arr.clear();
        arr.resize(N, [](T *b, T *e) {
            (void)b;
            (void)e;
        });
    };
    sizeNoInit(buf.px);
    sizeNoInit(buf.py);
    sizeNoInit(buf.pz);
    sizeNoInit(buf.curveId);
    sizeNoInit(buf.rootPrim);
    sizeNoInit(buf.rootUV);
    sizeNoInit(buf.rootT);
    sizeNoInit(buf.rootN);
    sizeNoInit(buf.rootB);
    sizeNoInit(buf.hairT);
    float *dpx = buf.px.data(), *dpy = buf.py.data(), *dpz = buf.pz.data();
    uint64_t *dids = buf.curveId.data();
    int *dprim = buf.rootPrim.data();
    GfVec2f *duv = buf.rootUV.data();
    GfVec3f *dt = buf.rootT.data(), *dn = buf.rootN.data(),
            *db = buf.rootB.data();
    float *dh = buf.hairT.data();
    ScatterParallelFor(gatherDispatcher, gatherChunks, [&](size_t c) {
        size_t const i0 = (c * N) / gatherChunks;
        size_t const i1 = ((c + 1) * N) / gatherChunks;
        for (size_t i = i0; i < i1; ++i)
            new (dpx + i) float(ax[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dpy + i) float(ay[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dpz + i) float(az[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dids + i) uint64_t(aids[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dprim + i) int(aPrim[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (duv + i) GfVec2f(aUv[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dt + i) GfVec3f(aT[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dn + i) GfVec3f(aN[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (db + i) GfVec3f(aB[perm[i]]);
        for (size_t i = i0; i < i1; ++i)
            new (dh + i) float(0.0f);
    });
    return true;
}

void UsdGenScatterOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &capture,
    UsdGenChunkView *view) const
{
    // A generator's Evaluate is identity: Capture filled the node buffer
    // (framework pre-copied the planes from the capture buffer).
    TF_UNUSED(ctx); TF_UNUSED(capture); TF_UNUSED(view);
}

}  // namespace usdGen
