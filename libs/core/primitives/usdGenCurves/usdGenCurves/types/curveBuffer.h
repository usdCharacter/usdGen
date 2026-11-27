// usdGen engine — curve buffers, chunk views and tile views. Publication
// types live in usdGenCurves/publish/curvePublication.h.
//
// Plan source: 03-execution-engine.md §1.2/§1.5/§6.3; tile contract C2:
// docs/freezes/C2.md. I1: per-CV planar SoA, per-curve AoS.
#ifndef USDGEN_CURVE_BUFFER_H
#define USDGEN_CURVE_BUFFER_H

#include "usdGen/api/primRef.h"
#include "usdGenCurves/export.h"
#include "usdGen/pool.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/types.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGenMath/math.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/token.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

/// One chunk (03 §1.2). Surface-major, then Morton order over rest roots (§1.7).
struct UsdGenChunkDesc
{
    uint32_t        firstCurve = 0;   // index into the per-curve arrays
    uint32_t        curveCount = 0;   // allocated curves, <= chunkSize
    uint32_t        liveCount  = 0;   // curves surviving density decimation; <= curveCount
    uint32_t        firstCv    = 0;   // index into the per-CV arrays
    uint32_t        cvCount    = 0;   // CVs per curve; 0 == ragged chunk (use cvOffsets)
    UsdGenTileId    tile      = 0;
    UsdGenSurfaceId surface   = 0;
    UsdGenRange3f       boundsRest;       // filled at capture; brush/footprint rejection, tile extents
};

/// One node's output. Per-CV data planar (I1); per-curve data AoS.
/// Handoff to imaging is by value (VtArray copy-on-write refcount bump, R20).
struct UsdGenCurveBuffer
{
    // ---- per CV ----------------------------------------------------------------
    UsdGenFloatArray px, py, pz;          // size == totalCvs
    // Authored C3 rest positions, kept separately from the current geometry.
    // Empty means the source did not supply a usable rest channel.  It is an
    // immutable COW pass-through plane: style/value operators never write it.
    UsdGenVec3fArray rest;
    UsdGenFloatArray width;               // empty == inherit upstream
    UsdGenFloatArray hairT;               // root->tip parameter, written once by the generator
    std::vector<UsdGenPlane> extraCv; // named per-CV planes, sorted by name

    // ---- per curve ---------------------------------------------------------------
    // primvars:usdGen:curveId (uniform uint64[], R12):
    //   UsdGenHash64(seed, faceIndex, k, kSaltScatter), see usdGenMath hash.h.
    // NOTE: OpenUSD 26.08 ships no UsdGenUInt64Array alias; the type is
    //       UsdGenCowArray<uint64_t> (VT_INTEGRAL_BUILTIN_VALUE_TYPES, pxr/base/vt/types.h:56).
    UsdGenCowArray<uint64_t> curveId;
    UsdGenIntArray   rootPrim;            // surface face index (== primvars:skinprim)
    UsdGenVec2fArray rootUV;              // == primvars:skinprimuv, and the "st" primvar
    UsdGenVec3fArray rootT, rootN, rootB; // rest root frame, 36 B/curve
    UsdGenIntArray   cvOffsets;           // size totalCurves+1; present iff any chunk is ragged
    std::vector<UsdGenPlane> extraCurve;  // clumpId_<level>, guideIndex, guideWeight

    // ---- geometry ------------------------------------------------------------------
    std::vector<UsdGenChunkDesc> chunks; // surface-major, then Morton order
    uint32_t totalCurves = 0, totalCvs = 0;
    uint64_t topologyVersion = 0;        // bumped when curve or CV counts change
    uint64_t valueVersion    = 0;        // bumped by every write to this buffer
};

/// Back to a default-constructed buffer, keeping storage: every array is
/// cleared, which keeps its capacity when this buffer is its only owner and
/// drops the reference otherwise (a shared array is never written). Used by
/// recycled captures so a warm re-capture refills the storage it had.
inline void UsdGenRecycleBuffer(UsdGenCurveBuffer &b)
{
    b.px.clear(); b.py.clear(); b.pz.clear();
    b.rest.clear(); b.width.clear(); b.hairT.clear();
    b.extraCv.clear();
    b.curveId.clear(); b.rootPrim.clear(); b.rootUV.clear();
    b.rootT.clear(); b.rootN.clear(); b.rootB.clear();
    b.cvOffsets.clear();
    b.extraCurve.clear();
    b.chunks.clear();
    b.totalCurves = b.totalCvs = 0;
    b.topologyVersion = b.valueVersion = 0;
}

/// Rebuilds named planes for a topology-preserving curve-count resample.
/// `target` must already contain the new CV topology. Vertex float payloads
/// use the same linear index parameter as CurveSource/Resample; integer
/// payloads use the nearest source sample so their type is never coerced.
/// Uniform and constant planes are copied into fresh storage. The source is
/// never modified and `target` is unchanged on failure.
bool UsdGenResampleExtraPlanes(UsdGenCurveBuffer const &source,
                               UsdGenCurveBuffer *target,
                               std::string *error = nullptr);

/// Rebuilds named planes for stable curve compaction. `survivors` is the
/// strictly ascending source-curve index for every target curve. Per-CV
/// payload follows each survivor's complete CV span; uniform payload follows
/// the same survivor order; constant payload is privately copied once. The
/// source is immutable and `target` is unchanged on failure.
bool UsdGenCompactExtraPlanes(UsdGenCurveBuffer const &source,
                              std::vector<uint32_t> const &survivors,
                              UsdGenCurveBuffer *target,
                              std::string *error = nullptr);

struct UsdGenCurveSetView;   // usdGenCurves/types/curveSet.h

/// View overloads: copy the view's planes and curve topology into a scratch
/// buffer first (capture time only).
USDGEN_CURVES_API bool UsdGenResampleExtraPlanes(UsdGenCurveSetView const &source,
                                               UsdGenCurveBuffer *target,
                                               std::string *error = nullptr);
USDGEN_CURVES_API bool UsdGenCompactExtraPlanes(UsdGenCurveSetView const &source,
                                              std::vector<uint32_t> const &survivors,
                                              UsdGenCurveBuffer *target,
                                              std::string *error = nullptr);
/// A buffer with the view's counts, offsets and named planes (copies; no
/// built-in columns). What the overloads above and UsdGenBuildClumpMotion's
/// view overload read.
USDGEN_CURVES_API UsdGenCurveBuffer UsdGenCurvePlanesBuffer(UsdGenCurveSetView const &source);

namespace detail {

inline bool
_UsdGenExtraPlaneFail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

// 'Curves': UsdGenCurveBuffer or UsdGenCurveSetView (same field names).
template <class Curves, class SpanVector>
inline bool
_UsdGenCurveSpans(Curves const &buffer,
                  SpanVector *spans, std::string *error)
{
    if (!spans)
        return _UsdGenExtraPlaneFail("extra-plane transform has no span output", error);
    spans->assign(size_t(buffer.totalCurves) + 1, 0);
    if (buffer.cvOffsets.empty()) {
        if (buffer.totalCurves == 0) {
            if (buffer.totalCvs != 0)
                return _UsdGenExtraPlaneFail(
                    "uniform extra-plane topology has CVs but no curves", error);
            return true;
        }
        if (buffer.totalCvs % buffer.totalCurves != 0)
            return _UsdGenExtraPlaneFail(
                "uniform extra-plane topology has non-integral CV count", error);
        uint32_t const perCurve = buffer.totalCvs / buffer.totalCurves;
        for (uint32_t curve = 0; curve != buffer.totalCurves; ++curve)
            (*spans)[curve + 1] = (*spans)[curve] + perCurve;
        return true;
    }
    if (buffer.cvOffsets.size() != size_t(buffer.totalCurves) + 1 ||
        buffer.cvOffsets.front() != 0)
        return _UsdGenExtraPlaneFail("ragged extra-plane topology has invalid offsets", error);
    for (uint32_t curve = 0; curve != buffer.totalCurves; ++curve) {
        int const first = buffer.cvOffsets[curve];
        int const last = buffer.cvOffsets[curve + 1];
        if (first < 0 || last < first)
            return _UsdGenExtraPlaneFail("ragged extra-plane topology has decreasing offsets", error);
        (*spans)[curve] = static_cast<uint32_t>(first);
        (*spans)[curve + 1] = static_cast<uint32_t>(last);
    }
    if ((*spans).back() != buffer.totalCvs)
        return _UsdGenExtraPlaneFail("ragged extra-plane topology count disagrees with offsets", error);
    return true;
}

inline bool
_UsdGenPlaneValueCount(size_t elements, uint8_t arity, size_t *values,
                       std::string *error)
{
    if (!values || arity == 0 ||
        elements > std::numeric_limits<size_t>::max() / arity)
        return _UsdGenExtraPlaneFail("extra-plane value count overflows this platform", error);
    *values = elements * arity;
    return true;
}

inline bool
_UsdGenValidatePlaneList(std::vector<UsdGenPlane> const &planes,
                         bool vertex, uint32_t elements,
                         std::string *error)
{
    UsdGenToken const expected = vertex ? UsdGenTokens().vertex : UsdGenTokens().uniform;
    UsdGenToken previous;
    for (UsdGenPlane const &plane : planes) {
        if (plane.name.IsEmpty() || plane.arity == 0 ||
            (!previous.IsEmpty() && !(previous < plane.name)))
            return _UsdGenExtraPlaneFail(
                "extra-plane names must be nonempty and strictly sorted", error);
        previous = plane.name;
        bool const constant = !vertex && plane.interpolation == UsdGenTokens().constant;
        if (plane.interpolation != expected && !constant)
            return _UsdGenExtraPlaneFail("extra-plane has unsupported interpolation '" +
                                         plane.interpolation.GetString() + "'", error);
        size_t values = 0;
        if (!_UsdGenPlaneValueCount(constant ? 1 : elements, plane.arity,
                                    &values, error))
            return false;
        if (plane.type == UsdGenTokens().float_) {
            if (!plane.i.empty() || plane.f.size() != values)
                return _UsdGenExtraPlaneFail("float extra-plane has invalid payload size", error);
        } else if (plane.type == UsdGenTokens().int_) {
            if (!plane.f.empty() || plane.i.size() != values)
                return _UsdGenExtraPlaneFail("int extra-plane has invalid payload size", error);
        } else {
            return _UsdGenExtraPlaneFail("extra-plane has unsupported type '" +
                                         plane.type.GetString() + "'", error);
        }
    }
    return true;
}

inline bool
_UsdGenValidateExtraPlanes(UsdGenCurveBuffer const &source, std::string *error)
{
    if (!_UsdGenValidatePlaneList(source.extraCv, true, source.totalCvs, error) ||
        !_UsdGenValidatePlaneList(source.extraCurve, false, source.totalCurves, error))
        return false;
    // A name has one published identity.  Do not silently choose one domain
    // when malformed input puts it in both vectors.
    size_t cv = 0, curve = 0;
    while (cv < source.extraCv.size() && curve < source.extraCurve.size()) {
        UsdGenToken const &a = source.extraCv[cv].name;
        UsdGenToken const &b = source.extraCurve[curve].name;
        if (a == b)
            return _UsdGenExtraPlaneFail("extra-plane name appears in both vertex and uniform domains", error);
        if (a < b) ++cv;
        else ++curve;
    }
    return true;
}

inline bool
_UsdGenPrivatePlane(UsdGenPlane const &source, uint32_t elements,
                    UsdGenPlane *result, std::string *error)
{
    if (!result)
        return _UsdGenExtraPlaneFail("extra-plane transform has no plane output", error);
    size_t values = 0;
    if (!_UsdGenPlaneValueCount(
            source.interpolation == UsdGenTokens().constant ? 1 : elements,
            source.arity, &values, error))
        return false;
    result->name = source.name;
    result->interpolation = source.interpolation;
    result->type = source.type;
    result->arity = source.arity;
    if (source.type == UsdGenTokens().int_) result->i = UsdGenIntArray(values, 0);
    else result->f = UsdGenFloatArray(values, 0.0f);
    return true;
}

inline void
_UsdGenCopyElements(UsdGenPlane const &source, size_t sourceElement,
                    UsdGenPlane *target, size_t targetElement)
{
    size_t const arity = source.arity;
    if (source.type == UsdGenTokens().int_)
        std::copy_n(source.i.cdata() + sourceElement * arity, arity,
                    target->i.begin() + targetElement * arity);
    else
        std::copy_n(source.f.cdata() + sourceElement * arity, arity,
                    target->f.begin() + targetElement * arity);
}

} // namespace detail

inline bool
UsdGenResampleExtraPlanes(UsdGenCurveBuffer const &source,
                          UsdGenCurveBuffer *target, std::string *error)
{
    if (!target)
        return detail::_UsdGenExtraPlaneFail("extra-plane resample has no target", error);
    UsdGenPoolVector<uint32_t> sourceSpans, targetSpans;   // no heap once warm
    if (!detail::_UsdGenCurveSpans(source, &sourceSpans, error) ||
        !detail::_UsdGenCurveSpans(*target, &targetSpans, error) ||
        !detail::_UsdGenValidateExtraPlanes(source, error))
        return false;
    if (target->totalCurves != source.totalCurves)
        return detail::_UsdGenExtraPlaneFail(
            "extra-plane resample must preserve the curve count", error);

    try {
    std::vector<UsdGenPlane> extraCv;
    std::vector<UsdGenPlane> extraCurve;
    extraCv.reserve(source.extraCv.size());
    extraCurve.reserve(source.extraCurve.size());
    for (UsdGenPlane const &plane : source.extraCv) {
        UsdGenPlane transformed;
        if (!detail::_UsdGenPrivatePlane(
                plane, target->totalCvs, &transformed, error)) return false;
        for (uint32_t curve = 0; curve != source.totalCurves; ++curve) {
            uint32_t const inputCount = sourceSpans[curve + 1] - sourceSpans[curve];
            uint32_t const outputCount = targetSpans[curve + 1] - targetSpans[curve];
            if (inputCount == 0 && outputCount != 0)
                return detail::_UsdGenExtraPlaneFail(
                    "cannot resample a vertex extra-plane from an empty curve", error);
            for (uint32_t output = 0; output != outputCount; ++output) {
                double const t = inputCount <= 1 || outputCount <= 1 ? 0.0 :
                    double(output) * double(inputCount - 1) / double(outputCount - 1);
                uint32_t const lower = static_cast<uint32_t>(t);
                uint32_t const upper = std::min(lower + 1, inputCount - 1);
                size_t const inBase = sourceSpans[curve];
                size_t const outIndex = size_t(targetSpans[curve]) + output;
                if (plane.type == UsdGenTokens().int_) {
                    uint32_t const nearest = static_cast<uint32_t>(std::floor(t + 0.5));
                    detail::_UsdGenCopyElements(plane, inBase + nearest, &transformed, outIndex);
                } else {
                    float const fraction = static_cast<float>(t - double(lower));
                    for (uint32_t component = 0; component != plane.arity; ++component) {
                        float const a = plane.f[(inBase + lower) * plane.arity + component];
                        float const b = plane.f[(inBase + upper) * plane.arity + component];
                        transformed.f[outIndex * plane.arity + component] = a + fraction * (b - a);
                    }
                }
            }
        }
        extraCv.push_back(std::move(transformed));
    }
    for (UsdGenPlane const &plane : source.extraCurve) {
        UsdGenPlane transformed;
        if (!detail::_UsdGenPrivatePlane(
                plane, target->totalCurves, &transformed, error)) return false;
        size_t const elements = plane.interpolation == UsdGenTokens().constant ? 1 : source.totalCurves;
        for (size_t element = 0; element != elements; ++element)
            detail::_UsdGenCopyElements(plane, element, &transformed, element);
        extraCurve.push_back(std::move(transformed));
    }
    target->extraCv = std::move(extraCv);
    target->extraCurve = std::move(extraCurve);
    return true;
    } catch (...) {
        return detail::_UsdGenExtraPlaneFail(
            "extra-plane resample allocation failed", error);
    }
}

inline bool
UsdGenCompactExtraPlanes(UsdGenCurveBuffer const &source,
                         std::vector<uint32_t> const &survivors,
                         UsdGenCurveBuffer *target, std::string *error)
{
    if (!target)
        return detail::_UsdGenExtraPlaneFail("extra-plane compaction has no target", error);
    UsdGenPoolVector<uint32_t> sourceSpans, targetSpans;   // no heap once warm
    if (!detail::_UsdGenCurveSpans(source, &sourceSpans, error) ||
        !detail::_UsdGenCurveSpans(*target, &targetSpans, error) ||
        !detail::_UsdGenValidateExtraPlanes(source, error))
        return false;
    if (target->totalCurves != survivors.size())
        return detail::_UsdGenExtraPlaneFail(
            "extra-plane compaction survivor count disagrees with target", error);
    uint32_t previous = 0;
    for (size_t output = 0; output != survivors.size(); ++output) {
        uint32_t const input = survivors[output];
        if (input >= source.totalCurves || (output != 0 && input <= previous))
            return detail::_UsdGenExtraPlaneFail(
                "extra-plane compaction survivors must be valid and strictly ascending", error);
        previous = input;
        if (targetSpans[output + 1] - targetSpans[output] !=
            sourceSpans[input + 1] - sourceSpans[input])
            return detail::_UsdGenExtraPlaneFail(
                "extra-plane compaction changed a survivor CV count", error);
    }

    try {
    std::vector<UsdGenPlane> extraCv;
    std::vector<UsdGenPlane> extraCurve;
    extraCv.reserve(source.extraCv.size());
    extraCurve.reserve(source.extraCurve.size());
    for (UsdGenPlane const &plane : source.extraCv) {
        UsdGenPlane transformed;
        if (!detail::_UsdGenPrivatePlane(
                plane, target->totalCvs, &transformed, error)) return false;
        for (size_t outputCurve = 0; outputCurve != survivors.size(); ++outputCurve) {
            uint32_t const inputCurve = survivors[outputCurve];
            uint32_t const count = sourceSpans[inputCurve + 1] - sourceSpans[inputCurve];
            for (uint32_t cv = 0; cv != count; ++cv)
                detail::_UsdGenCopyElements(plane, size_t(sourceSpans[inputCurve]) + cv,
                    &transformed, size_t(targetSpans[outputCurve]) + cv);
        }
        extraCv.push_back(std::move(transformed));
    }
    for (UsdGenPlane const &plane : source.extraCurve) {
        UsdGenPlane transformed;
        if (!detail::_UsdGenPrivatePlane(
                plane, target->totalCurves, &transformed, error)) return false;
        if (plane.interpolation == UsdGenTokens().constant) {
            detail::_UsdGenCopyElements(plane, 0, &transformed, 0);
        } else {
            for (size_t output = 0; output != survivors.size(); ++output)
                detail::_UsdGenCopyElements(plane, survivors[output], &transformed, output);
        }
        extraCurve.push_back(std::move(transformed));
    }
    target->extraCv = std::move(extraCv);
    target->extraCurve = std::move(extraCurve);
    return true;
    } catch (...) {
        return detail::_UsdGenExtraPlaneFail(
            "extra-plane compaction allocation failed", error);
    }
}

/// A read-only window over one tile's chunk range, used by the interleaver (03 §6.3).
struct UsdGenTileView
{
    UsdGenTileId tile;
    uint32_t     firstChunk, chunkCount;
    uint32_t     totalLiveCurves, totalLiveCvs;
    UsdGenRange3f    extent;                      // written by the interleave pass
    // Engine-internal tile-change flags, set by the interleave pass and
    // consumed by the session's generation builder (not part of C2).
    bool pointsDirty = false;
    bool widthsDirty = false;
};

}  // namespace usdGen

#endif  // USDGEN_CURVE_BUFFER_H
