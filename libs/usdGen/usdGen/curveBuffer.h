// usdGen engine — curve buffers, chunk views, tile views, publication boundary.
//
// Plan source: 03-execution-engine.md §1.2/§1.5/§6.3; tile contract C2:
// docs/freezes/C2.md. I1: per-CV planar SoA, per-curve AoS.
#ifndef USDGEN_CURVE_BUFFER_H
#define USDGEN_CURVE_BUFFER_H

#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"
#include "pxr/usd/sdf/path.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// One named extra plane (03 §1.2). R24 pins type/arity per emitted name:
///   clumpId_<level> : uniform int,       arity 1
///   guideIndex      : uniform int[],     arity 3
///   guideWeight     : uniform float[],   arity 3
/// `int[3]`/`float[3]` are not USD type names. Exactly one of `f`/`i` holds
/// the payload.
struct UsdGenPlane
{
    TfToken      name;
    TfToken      interpolation;   // vertex | uniform | constant
    TfToken      type;            // float | int
    uint8_t      arity = 1;
    VtFloatArray f;
    VtIntArray   i;
};

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
    GfRange3f       boundsRest;       // filled at capture; brush/footprint rejection, tile extents
};

/// One node's output. Per-CV data planar (I1); per-curve data AoS.
/// Handoff to imaging is by value (VtArray copy-on-write refcount bump, R20).
struct UsdGenCurveBuffer
{
    // ---- per CV ----------------------------------------------------------------
    VtFloatArray px, py, pz;          // size == totalCvs
    // Authored C3 rest positions, kept separately from the current geometry.
    // Empty means the source did not supply a usable rest channel.  It is an
    // immutable COW pass-through plane: style/value operators never write it.
    VtVec3fArray rest;
    VtFloatArray width;               // empty == inherit upstream
    VtFloatArray hairT;               // root->tip parameter, written once by the generator
    std::vector<UsdGenPlane> extraCv; // named per-CV planes, sorted by name

    // ---- per curve ---------------------------------------------------------------
    // primvars:usdGen:curveId (uniform uint64[], R12):
    //   UsdGenHash64(seed, faceIndex, k, kSaltScatter), see usdGenMath hash.h.
    // NOTE: OpenUSD 26.08 ships no VtUInt64Array alias; the type is
    //       VtArray<uint64_t> (VT_INTEGRAL_BUILTIN_VALUE_TYPES, pxr/base/vt/types.h:56).
    VtArray<uint64_t> curveId;
    VtIntArray   rootPrim;            // surface face index (== primvars:skinprim)
    VtVec2fArray rootUV;              // == primvars:skinprimuv, and the "st" primvar
    VtVec3fArray rootT, rootN, rootB; // rest root frame, 36 B/curve
    VtIntArray   cvOffsets;           // size totalCurves+1; present iff any chunk is ragged
    std::vector<UsdGenPlane> extraCurve;  // clumpId_<level>, guideIndex, guideWeight

    // ---- geometry ------------------------------------------------------------------
    std::vector<UsdGenChunkDesc> chunks; // surface-major, then Morton order
    uint32_t totalCurves = 0, totalCvs = 0;
    uint64_t topologyVersion = 0;        // bumped when curve or CV counts change
    uint64_t valueVersion    = 0;        // bumped by every write to this buffer
};

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

namespace detail {

inline bool
_UsdGenExtraPlaneFail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

inline bool
_UsdGenCurveSpans(UsdGenCurveBuffer const &buffer,
                  std::vector<uint32_t> *spans, std::string *error)
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
    TfToken const expected = vertex ? TfToken("vertex") : TfToken("uniform");
    TfToken previous;
    for (UsdGenPlane const &plane : planes) {
        if (plane.name.IsEmpty() || plane.arity == 0 ||
            (!previous.IsEmpty() && !(previous < plane.name)))
            return _UsdGenExtraPlaneFail(
                "extra-plane names must be nonempty and strictly sorted", error);
        previous = plane.name;
        bool const constant = !vertex && plane.interpolation == TfToken("constant");
        if (plane.interpolation != expected && !constant)
            return _UsdGenExtraPlaneFail("extra-plane has unsupported interpolation '" +
                                         plane.interpolation.GetString() + "'", error);
        size_t values = 0;
        if (!_UsdGenPlaneValueCount(constant ? 1 : elements, plane.arity,
                                    &values, error))
            return false;
        if (plane.type == TfToken("float")) {
            if (!plane.i.empty() || plane.f.size() != values)
                return _UsdGenExtraPlaneFail("float extra-plane has invalid payload size", error);
        } else if (plane.type == TfToken("int")) {
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
        TfToken const &a = source.extraCv[cv].name;
        TfToken const &b = source.extraCurve[curve].name;
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
            source.interpolation == TfToken("constant") ? 1 : elements,
            source.arity, &values, error))
        return false;
    result->name = source.name;
    result->interpolation = source.interpolation;
    result->type = source.type;
    result->arity = source.arity;
    if (source.type == TfToken("int")) result->i = VtIntArray(values, 0);
    else result->f = VtFloatArray(values, 0.0f);
    return true;
}

inline void
_UsdGenCopyElements(UsdGenPlane const &source, size_t sourceElement,
                    UsdGenPlane *target, size_t targetElement)
{
    size_t const arity = source.arity;
    if (source.type == TfToken("int"))
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
    std::vector<uint32_t> sourceSpans, targetSpans;
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
                if (plane.type == TfToken("int")) {
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
        size_t const elements = plane.interpolation == TfToken("constant") ? 1 : source.totalCurves;
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
    std::vector<uint32_t> sourceSpans, targetSpans;
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
        if (plane.interpolation == TfToken("constant")) {
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

/// One chunk's posed extent, recorded by the sweep that wrote the points
/// so the tile interleave can union per-chunk extents instead of re-reading
/// every point. The span pins the chunk layout the extent was recorded
/// under: a mid-run repartition changes the spans and the interleave falls
/// back to the point pass, so a stale slot is never read.
struct UsdGenChunkExtent
{
    GfRange3f extent;
    uint32_t firstCurve = 0;
    uint32_t curveCount = 0;
    uint32_t liveCount = 0;
    uint32_t firstCv = 0;
    uint32_t cvCount = 0;
};

/// What a kernel is allowed to touch. All pointers are chunk-local bases
/// (03 §1.2). cvCount == 0 on the ragged path.
struct UsdGenChunkView
{
    const UsdGenChunkDesc *desc;
    float       *px, *py, *pz;               // out: writable
    const float *inPx, *inPy, *inPz, *inWidth;  // in: read-only, never const_cast
    // Optional ordered second geometry input. WidthBlend merges only this
    // plane and inherits every other plane from the first input.
    const float *inWidth2 = nullptr;
    float       *width;                      // out when the node touches kPlaneWidths (scheduler binds fillOut); otherwise aliases upstream
    float       *hairT;                      // out when the node touches kPlaneHairT; otherwise aliases upstream
    const uint64_t *curveId;
    const int   *rootPrim;
    const int   *cvOffsets;                  // null on the uniform fast path
    const GfVec2f *rootUV;
    const GfVec3f *rootT, *rootN, *rootB;
    const float *rampLut;                    // 257 entries over hairT, or null
    // Extra planes this node declared in OutputPrimvars(), bound by slot at compile.
    // Per-curve slots have curveCount * arity entries; per-CV slots curveCount * cvCount.
    float       **outF;                      // writable float planes, null when the node emits none
    int         **outI;                      // writable int planes
    const float **inF;                       // upstream planes this node reads
    const int   **inI;
    uint32_t      outCount, inCount;
    // Extent record for this chunk, or null when the sweep does not record
    // (every op but the recording points-writer). A recording kernel writes
    // the extent over exactly the floats it stores, in store order, plus
    // the span from desc, before Evaluate returns.
    UsdGenChunkExtent *extentSlot = nullptr;
    uint32_t curveCount, cvCount;            // cvCount == 0 on the ragged path
    uint32_t inCvCount = 0;                  // upstream's CVs per curve (in* base + c*inCvCount + i)
    // Upstream ragged offsets for this chunk, shifted to firstCurve but still
    // absolute in the upstream CV array.  `inFirstCv` rebases them onto the
    // already-sliced inPx/inPy/inPz ports.
    const int *inCvOffsets = nullptr;
    uint32_t inFirstCv = 0;
    inline size_t Cv(uint32_t c, uint32_t i) const { return size_t(c) * cvCount + i; }
    inline size_t CvRagged(uint32_t c, uint32_t i) const { return size_t(cvOffsets[c]) + i; }
    inline float *OutF(uint32_t slot) const { return outF[slot]; }
    inline int   *OutI(uint32_t slot) const { return outI[slot]; }
    inline const float *InF(uint32_t slot) const { return inF[slot]; }
    inline const int   *InI(uint32_t slot) const { return inI[slot]; }
};

/// A read-only window over one tile's chunk range, used by the interleaver (03 §6.3).
struct UsdGenTileView
{
    UsdGenTileId tile;
    uint32_t     firstChunk, chunkCount;
    uint32_t     totalLiveCurves, totalLiveCvs;
    GfRange3f    extent;                      // written by the interleave pass
    // Engine-internal tile-change flags, set by the interleave pass and
    // consumed by the session's generation builder (not part of C2).
    bool pointsDirty = false;
    bool widthsDirty = false;
};

/// One reference set per reference node, per generation (03 §1.5, I3).
struct UsdGenReferenceSet
{
    UsdGenCurveBuffer buffer;                 // evaluated in full, never chunk-dirty
    VtFloatArray      localX, localY, localZ; // root-local offsets, refreshed per frame
    uint64_t          generation = 0;
};

/// Immutable compiler-resolved external reference value.  The shared payload
/// gives a capture a stable value identity without retaining a mutable graph,
/// descriptor, scene index, or stage object.
struct UsdGenResolvedReferenceValue
{
    SdfPath path;
    uint64_t curveGeneration = 0;
    uint64_t identity = 0;
    std::shared_ptr<const UsdGenReferenceSet> value;
};

// ---------------------------------------------------------------------------
// Imaging boundary types (contract C2, docs/freezes/C2.md; 03 §9.1: "the engine
// only fills them"). One UsdGenTilePublication per published tile prim
// <description>/__usdGenRender/tile_NNNN.
// ---------------------------------------------------------------------------

struct UsdGenTilePublication
{
    UsdGenTileId  tile = 0;
    SdfPath       primPath;                   // <description>/__usdGenRender/tile_NNNN (4-digit zero pad)

    // --- topology (C2) ---
    VtIntArray    curveVertexCounts;          // basisCurves/topology/curveVertexCounts
    std::string   basis = "bspline";          // basisCurves/topology/basis (usdGen:curve:basis)
    // Contract constants, never authorable: type="cubic", wrap="pinned".
    int           refineLevel = 2;            // displayStyle/refineLevel (S-9: no M1 tumble tier)

    // --- exact-size primvar arrays (C2): points.size() == sum(curveVertexCounts) (SI-1)
    VtVec3fArray  points;                     // primvars/points, vertex
    VtFloatArray  widths;                     // primvars/widths, vertex or constant; never varying
    VtFloatArray  hairT;                      // primvars/hairT, vertex, root 0 -> tip 1
    VtFloatArray  hairId;                     // primvars/hairId, uniform float [0,1)
    VtVec2fArray  st;                         // primvars/st, uniform (root UV)
    VtVec3fArray  displayColor;               // uniform (perCurve bake) or vertex (perCV)
    VtVec3fArray  bakeColor;                  // optional primvars:<bakePrimvar> (bakeTarget=primvar)
    // Operator-emitted planes (C2), preserving their constant/uniform/vertex
    // interpolation, float/int type, and element arity.  The historical member
    // name predates publication of per-CV extra planes.
    std::vector<UsdGenPlane> extraUniform;

    // Motion profile P1 only (post-M1); empty otherwise,
    // and the primvar is BLOCKED (HdBlockDataSource) in P0/P2.
    VtVec3fArray  velocities;

    // --- per-tile data (recomputed every deforming frame; C2 extent rule) ---
    GfVec3d       extentMin, extentMax;       // extent/min, extent/max

    // --- hand-authored per tile, post-flattening (ADR §5.3) ---
    GfMatrix4d    xformMatrix;                // xform/matrix, surface world matrix
    // xform/resetXformStack = true (constant), never a payload.
    TfToken       purpose;                    // inherited by hand from the description
    TfToken       visibility;                 // inherited by hand from the description
    SdfPath       materialPath;               // description's Material
    TfToken       materialPurpose = TfToken("allPurpose"); // the empty-token container child
    SdfPath       primOrigin;                 // the description prim; absolute outside prototypes
    SdfPath       dependencySurface;          // __dependencies: dependedOnPrimPath
};

/// The scalp-shadow cap: one synthetic Mesh per description,
/// <description>/__usdGenRender/scalpShadow. It is the haired part of the
/// groom's emitting surface, offset a hair's breadth along its own normal and
/// carrying the same furTauP/furTauN a strand carries, so a translucent
/// material can darken whatever skin shader the user bound underneath by the
/// fraction of light the hair above it absorbs. Points and normals are WORLD
/// space and the prim's xform is identity with resetXformStack, exactly as a
/// tile's is. Empty when the description has no occluder or no hair over it.
struct UsdGenScalpShadowPublication
{
    SdfPath       primPath;
    VtVec3fArray  points;                     // mesh/points, world space
    VtVec3fArray  normals;                    // primvars/normals, vertex
    VtIntArray    faceVertexCounts;           // all 3
    VtIntArray    faceVertexIndices;
    // furTauP / furTauN, vertex, arity 3 -- the same planes a tile publishes,
    // so the publisher packs them through the same path.
    std::vector<UsdGenPlane> extraUniform;
    GfVec3d       extentMin{0.0}, extentMax{0.0};
    TfToken       purpose;
    TfToken       visibility;
    SdfPath       materialPath;               // the synthetic scalp-shadow material
    // Content identity: the scene index dirties the prim when this changes and
    // shares the previous immutable arrays when it does not.
    uint64_t      digest = 0;
    // Carry identity for the rebuild path (furOcclusion.cpp): when the bake
    // rebuilds, the merged topology is a pure function of (topologyKey,
    // tessLevel), the normals add (occluderKey, inward), and the points add
    // lift. Matching keys let the build share the seeded previous cap's
    // arrays instead of merging fresh ones; every carry is size-checked and
    // fails closed. Internal equality-only, like digest: never golden.
    uint64_t      topologyKey = 0;
    uint64_t      occluderKey = 0;
    float         lift = 0.0f;
    float         inward = 0.0f;
    int           tessLevel = 0;

    bool IsEmpty() const { return points.empty() || faceVertexIndices.empty(); }
};

/// M6-only; empty in M1 (UsdGenInstance publishes instancers instead of tiles).
struct UsdGenInstancerPublication
{
    SdfPath       primPath;                   // <description>/__usdGenRender/inst_<opName>
    std::vector<SdfPath> prototypes;          // instancerTopology/prototypes (namespace children)
    std::vector<VtIntArray> instanceIndices;  // instancerTopology/instanceIndices, one per prototype
    VtBoolArray   mask;                       // instancerTopology/mask; empty == all true
};

/// Structural signature for the generation diff (03 §6.1): what changed in the
/// published prim set, independent of payload values.
struct UsdGenPrimSetSignature
{
    int           tileCount = 0;
    int           instancerCount = 0;
    std::vector<std::string> primPaths;       // sorted
    // per prim path (parallel to primPaths): sorted list of published primvar names
    std::vector<std::vector<std::string>> primvarNames;
    // per prim path (parallel to primPaths): prim type string
    std::vector<std::string> primTypes;

    bool operator==(UsdGenPrimSetSignature const &rhs) const;
};

/// Diff result consumed by the imaging notice emitter (06 §5.1 rules).
struct UsdGenTileDirty
{
    UsdGenTileId tile = 0;
    SdfPath      primPath;
    bool         removed = false;            // -> PrimsRemoved
    bool         added   = false;            // -> PrimsAdded
    bool         pointsDirty = false;        // -> primvars/points/primvarValue + extent/min|max
    bool         widthsDirty = false;        // -> primvars/widths/primvarValue
    bool         xformDirty  = false;        // -> xform/matrix
    std::vector<TfToken> newPrimvars;        // first appearance or new interpolation -> primvars/<name> once
    std::vector<TfToken> dirtyPrimvars;      // -> primvars/<name>/primvarValue (never with points)
};

struct UsdGenDirtyReport
{
    std::vector<UsdGenTileDirty> tiles;
    bool surfaceXformDirty = false;         // description-level xform moved
};

}  // namespace usdGen

#endif  // USDGEN_CURVE_BUFFER_H
