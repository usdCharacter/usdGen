// Paintable per-surface attribute maps and the stroke-accumulation model the
// host-groomer brushes operate on.
//
// A UsdGenPaintMap in per-face storage (plan/07-look-maps-expressions.md §5.2,
// §8) addresses values by (faceId, u, v) into the parent mesh's faces. This
// module is the engine-side value type for that storage: a CPU-side analog of
// the .ptx a Bake-map action writes (07 §8.3), one instance per surface mesh,
// which brushes mutate during a stroke and the capture loop samples like any
// other map. The read-only .ptx sampler next door (ptexMap.h) stays the bake
// output's reader; this map is the session's writable original.
//
// The stroke model implements the brush contract of plan/08-tools.md §2.1 in
// the engine: press snapshots the base, every move recomputes the whole map
// from that base (never incrementally), release commits the recomputed map,
// escape aborts without touching the base. Recomputing from the base is what
// makes a stroke idempotent and abortable, and why a dropped mouse-move
// cannot accumulate error.
//
// v1 limits, documented rather than silent:
//   - A dab affects only the face it addresses. Cross-face falloff needs
//     mesh adjacency, which this USD-free header does not carry; the brush
//     C ABI (usdGenImaging/usdGenBrushApi.h) expands footprints onto
//     neighbouring faces and smooths across them at the corner level.
//   - Resolution is uniform over faces (usdGen:paint:resolution, default 256
//     in 02-schema.md §2.12; the quantity a host groomer writes as `#3dpaint, N`).
//
// This header has no USD dependency, like ptexMap.h.
#ifndef USDGEN_MAPS_ATTRIBUTE_MAP_H
#define USDGEN_MAPS_ATTRIBUTE_MAP_H

#include "usdGen/export.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

// ---------------------------------------------------------------------------
// UsdGenAttributeMap: per-face texel storage for one surface
// ---------------------------------------------------------------------------

struct UsdGenAttributeMapSpec {
    // Faces of the parent mesh (UsdGenPtexFirstFaceIds counts Ptex ids; this
    // map is indexed by coarse face instead, one res x res grid per face).
    int numFaces = 0;
    // Texels per face edge. A power of two in [1, 256] so a Bake-map action
    // can carry the grid into a Ptex Res(ulog2, vlog2) unchanged.
    int resolution = 16;
    // 1 for a scalar mask (density, length), 3 for colour.
    int channels = 1;
    float defaultValue = 0.0f;
    // Clamp every texel write (Fill, SetTexel, dabs) into [0, 1].
    bool clamp01 = false;
};

enum class UsdGenAttributeMapInterp {
    Nearest,
    Bilinear,
};

class USDGEN_CORE_API UsdGenAttributeMap final {
public:
    // Returns null and sets *error (when non-null) for an invalid spec: no
    // faces, a non-power-of-two or out-of-range resolution, channels outside
    // {1, 3}, a non-finite default, or a grid that would exceed 2^28 floats.
    static std::shared_ptr<UsdGenAttributeMap> Create(
        UsdGenAttributeMapSpec const &spec, std::string *error);

    ~UsdGenAttributeMap();
    UsdGenAttributeMap(UsdGenAttributeMap const &) = delete;
    UsdGenAttributeMap &operator=(UsdGenAttributeMap const &) = delete;

    int NumFaces() const;
    int Resolution() const;
    int Channels() const;
    float DefaultValue() const;
    bool Clamp01() const;

    std::shared_ptr<UsdGenAttributeMap> Clone() const;

    // Texel (s, t) of face, channel c. s and t run [0, resolution); the texel
    // sits on its grid node at u = s / (resolution - 1). Returns false, leaving *value
    // untouched, for an out-of-range face, texel or channel, or a null value.
    // SetTexel also returns false for a non-finite value.
    bool GetTexel(int face, int s, int t, int channel, float *value) const;
    bool SetTexel(int face, int s, int t, int channel, float value);

    // Sample channel c at face-local (u, v). u and v are clamped to [0, 1];
    // bilinear interpolates the grid nodes. Returns false,
    // leaving *value untouched, for an out-of-range face or channel, a
    // non-finite u or v, or a null value.
    bool Sample(int face, float u, float v, int channel,
                UsdGenAttributeMapInterp interp, float *value) const;

    void Fill(float value);

    // 4-lane FNV-1a over the spec and every texel bit (usdGen/digest.h):
    // the map digest a capture epoch folds in (07 §5.4
    // UsdGenMapLibrary::Digest). Bitwise-identical maps digest identically;
    // values are internal keys, never persisted or golden-tested. The
    // digest is memoized until the next write (SetTexel, Fill, or any
    // MutableData fetch), so repeated reads over an unchanged map are
    // constant-time.
    uint64_t Digest() const;

    // Direct row-major plane access for capture loops and tests:
    // data()[((face * resolution) + t) * resolution + s) * channels + c].
    // The span has numFaces * resolution * resolution * channels floats.
    float const *Data() const;
    size_t FloatCount() const;
    // Mutable plane for bulk writers (base upsample, corner smoothing).
    // Writes bypass the clamp01/finite checks SetTexel applies: callers
    // write validated values only. Fetching the pointer dirties the
    // cached Digest(); complete all writes before the next Digest() —
    // holding the pointer across a Digest call with further writes would
    // read back a stale epoch key.
    float *MutableData();

private:
    UsdGenAttributeMap();
    struct State;
    std::unique_ptr<State> state_;
};

// ---------------------------------------------------------------------------
// UsdGenBrushStroke: press / move / release accumulation over a base map
// ---------------------------------------------------------------------------

enum class UsdGenBrushMode {
    // Mix toward value: tex = mix(tex, value, strength * weight).
    Set,
    // Offset: tex += value * strength * weight.
    Add,
    // Mix toward the 3x3 texel neighbourhood mean (computed from the
    // pre-dab map): tex = mix(tex, mean, strength * weight). Face-local:
    // the mesh-aware brush ABI (usdGenImaging/usdGenBrushApi.h) smooths
    // across faces at the corner level instead.
    Smooth,
    // Mix toward the map default: Set with value = DefaultValue().
    Erase,
};

enum class UsdGenBrushFalloff {
    // 08-tools.md §3.3 brushFalloff tokens, minus the tool's aliases.
    Constant,
    Linear,
    Smooth,
};

// One brush stamp. radius is in face-uv units (the face spans [0, 1]^2); a
// radius <= 0 paints the single nearest texel at full weight. strength is in
// [0, 1]. channel selects the painted channel, or every channel when
// negative. value is the Set target / Add offset; Smooth and Erase ignore
// it. hardness in [0, 1] is the inner-radius fraction: texels within
// hardness * radius take full weight, and the falloff curve runs from that
// inner radius out to the rim.
struct UsdGenBrushDab {
    int face = -1;
    float u = 0.0f;
    float v = 0.0f;
    float radius = 0.1f;
    float strength = 0.5f;
    float value = 1.0f;
    int channel = -1;
    UsdGenBrushMode mode = UsdGenBrushMode::Set;
    UsdGenBrushFalloff falloff = UsdGenBrushFalloff::Smooth;
    float hardness = 0.0f;
};

// The dab weight at `dist` from the centre for a dab of `radius` (both in
// face-uv units): 1 inside hardness * radius, the falloff curve from there to
// the rim, 0 past it. Shared by the texel kernel and the corner smoother.
USDGEN_CORE_API float UsdGenBrushWeight(UsdGenBrushFalloff falloff, float hardness,
                                        double dist, double radius);

// Applies one validated dab to a live map in place: the move kernel the
// stroke replays and the brush ABI stamps footprint entries with.
USDGEN_CORE_API void UsdGenApplyBrushDab(UsdGenAttributeMap *map,
                                         UsdGenBrushDab const &dab);

// Non-accumulating stroke application (a DCC's model). Along a move the
// stamps overlap heavily (spacing 0.5 * radius), and applying Set per stamp
// (tex += (value - tex) * k) saturates every texel under the trail to value
// after a few stamps: the falloff washes out and strength 1 paints a hard
// disc. Instead, Set / Erase / Smooth keep a per-texel max-weight buffer
// over every stamp of the current SEGMENT (a run of dabs with the same mode,
// target and channel):
//
//   wmax  = max(wmax, strength * falloffWeight)
//   texel = segmentBase + (target - segmentBase) * wmax
//
// where target is the value (Set), the map default (Erase) or the 3x3 mean
// of segmentBase (Smooth). The result never depends on how many stamps
// overlap a texel, only on the strongest one. Add keeps accumulating (that
// is its point) and ends the segment, as does any change of mode, target or
// channel: the next segment starts from the map as it then is.
class USDGEN_CORE_API UsdGenBrushAccumulator final {
public:
    // `map` is the live map the accumulator writes; it must outlive this.
    explicit UsdGenBrushAccumulator(UsdGenAttributeMap *map);
    ~UsdGenBrushAccumulator();
    UsdGenBrushAccumulator(UsdGenBrushAccumulator const &) = delete;
    UsdGenBrushAccumulator &operator=(UsdGenBrushAccumulator const &) = delete;

    // Applies one validated stamp (the brush ABI's footprint entries go
    // through here one by one).
    void Apply(UsdGenBrushDab const &dab);
    // Ends the current segment: the map as it is becomes the next base.
    // Callers that write the map directly (the corner smoother) flush first.
    void Flush();

private:
    struct State;
    std::unique_ptr<State> state_;
};

class USDGEN_CORE_API UsdGenBrushStroke final {
public:
    // The base is the press-time snapshot: Preview/Commit recompute from it,
    // so it must stay alive (and unmutated) for the stroke's lifetime.
    explicit UsdGenBrushStroke(std::shared_ptr<const UsdGenAttributeMap> base);
    ~UsdGenBrushStroke();
    UsdGenBrushStroke(UsdGenBrushStroke const &) = delete;
    UsdGenBrushStroke &operator=(UsdGenBrushStroke const &) = delete;

    // Records one dab. Fails closed (false + *error) for a face outside the
    // base map, non-finite u/v/radius/strength/value, a negative radius, a
    // strength outside [0, 1], or a channel below -1 / past the map's
    // channels. A strength-0 dab is valid and a no-op.
    bool AddDab(UsdGenBrushDab const &dab, std::string *error);

    // Records the move endpoint as interpolated dabs from the previous dab's
    // centre to (face, u, v) on the same face, spaced so no gap exceeds
    // spacing * radius (spacing in (0, 1], default 0.5). With no previous dab
    // — or a face change, which v1 never interpolates across — records one
    // dab. Validation is AddDab's plus spacing in (0, 1]. The dab-struct
    // overload carries every field (hardness included) of the endpoint.
    bool AddMove(UsdGenBrushDab const &endpoint, float spacing, std::string *error);
    bool AddMove(int face, float u, float v, float radius, float strength,
                 float value, int channel, UsdGenBrushMode mode,
                 UsdGenBrushFalloff falloff, float spacing, std::string *error);
    bool AddMove(int face, float u, float v, float radius, float strength,
                 float value, int channel, UsdGenBrushMode mode,
                 UsdGenBrushFalloff falloff, std::string *error)
    {
        return AddMove(face, u, v, radius, strength, value, channel, mode,
                       falloff, 0.5f, error);
    }

    size_t DabCount() const;
    UsdGenBrushDab const &Dab(size_t index) const;

    // Recompute the whole map from the press-time base plus every recorded
    // dab, in order (08-tools.md §2.1: the move path). Pure: calling it twice
    // yields bitwise-identical maps, and a dropped move cannot accumulate
    // error because no state accumulates. Never null with a valid base.
    // Dabs go through a UsdGenBrushAccumulator: overlapping Set / Erase /
    // Smooth stamps take the max weight, never re-apply.
    std::shared_ptr<UsdGenAttributeMap> Preview() const;

    // The release path: the same recomputed map the tool writes once into the
    // edit target. Like Preview, pure — persistence and undo bracketing are
    // the tool's job (07 §8.2).
    std::shared_ptr<UsdGenAttributeMap> Commit() const;

    // The escape path: forget every dab. The base was never touched, so the
    // abort is free.
    void Abort();

private:
    struct State;
    std::unique_ptr<State> state_;
};

}  // namespace usdGen

#endif  // USDGEN_MAPS_ATTRIBUTE_MAP_H
