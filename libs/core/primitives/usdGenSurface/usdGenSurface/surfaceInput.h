// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The Surface input kind (step 7.3a): the Mesh (or GeomSubset) payload a host
// captures (UsdGenSurfaceData), its record layout and InputDesc, the packer
// and the read view with the payload's field names.
#pragma once
#include "usdGen/api/graphSpec.h"   // pxr gf/tf/vt types (interim, step 11)
#include "usdGen/compile/types.h"
#include "usdGenSurface/export.h"
#include "usdGen/registries/engineTokens.h"


#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace usdGen {

class UsdGenRegistrar;

/// A surface geometry sample at an absolute time (R23).
struct UsdGenSurfaceSample { double time; VtVec3fArray points; };

// The Default-time mesh-normal interpolation transported by UsdGenRestAPI.
// Invalid is deliberately distinct from None: a nonempty malformed snapshot
// must not be silently converted into geometric-normal fallback by CUDA.
enum class UsdGenSurfaceNormalDomain : uint8_t {
    None,
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
    Invalid
};

/// One bound surface as a host captures it; travels as a UsdGenSurfaceInput record.
struct UsdGenSurfaceData
{
    UsdGenSurfaceId id = 0;
    VtIntArray     faceVertexCounts, faceVertexIndices;
    // USD subdivision opinions, captured identically by stage/Hydra builders.
    TfToken subdivisionScheme = UsdGenTokens().none, orientation = UsdGenTokens().rightHanded;
    TfToken interpolateBoundary = UsdGenTokens().edgeAndCorner,
        faceVaryingLinearInterpolation = UsdGenTokens().cornersPlus1;
    TfToken triangleSubdivisionRule = UsdGenTokens().catmullClark,
        creaseMethod = UsdGenTokens().uniform;
    VtIntArray holeIndices, creaseIndices, creaseLengths, cornerIndices;
    VtFloatArray creaseSharpnesses, cornerSharpnesses;
    VtVec3fArray   restPoints;        // usdGen/rest/points (S12), UsdTimeCode::Default()
    VtVec3fArray   restNormals;       // usdGen/rest/normals, Default-time Mesh normals
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    bool          restFromCurrentPoints = false; // compatibility fallback, never a valid RBF binding
    VtVec3fArray   points;            // deformed, at the graph time
    /// Sorted by time; samples[0].time == the graph time (R23).
    std::vector<UsdGenSurfaceSample> samples;
    VtVec3fArray   velocities;        // motion profile P1 only; empty otherwise
    VtVec2fArray   uv;                // the surface's primary uv set
    // A GeomSubset target (R15) carries its parent Mesh's full geometry
    // above, so face ids stay parent-mesh ids, plus its sorted, unique faces
    // here. isSubset makes subsetFaces authoritative even when empty (an
    // empty subset selects no face); otherwise empty == whole mesh.
    VtIntArray     subsetFaces;
    bool           isSubset = false;
    VtFloatArray   densityMultiplier; // per-face scatter density scale from
                                     // usdGen:paint:density (face means,
                                     // clamped >= 0, non-finite -> 0);
                                     // empty == all 1.0
    GfMatrix4d     worldMatrix{1.0};  // post-flattening, resetXformStack (S4)
    uint64_t       surfaceGeneration = 0; // bumped by any points/topology change
};

/// Record columns of the UsdGenSurfaceInput kind (free-length).
enum UsdGenSurfaceInputSlot : UsdGenSlot {
    kUsdGenSurfaceInputFaceVertexCounts,      // I32
    kUsdGenSurfaceInputFaceVertexIndices,     // I32
    kUsdGenSurfaceInputHoleIndices,           // I32
    kUsdGenSurfaceInputCreaseIndices,         // I32
    kUsdGenSurfaceInputCreaseLengths,         // I32
    kUsdGenSurfaceInputCornerIndices,         // I32
    kUsdGenSurfaceInputCreaseSharpnesses,     // F32
    kUsdGenSurfaceInputCornerSharpnesses,     // F32
    kUsdGenSurfaceInputRestPoints,            // F32x3
    kUsdGenSurfaceInputRestNormals,           // F32x3
    kUsdGenSurfaceInputPoints,                // F32x3
    kUsdGenSurfaceInputSamplePoints,          // F32x3, every sample in order
    kUsdGenSurfaceInputSampleTimes,           // F64 per sample
    kUsdGenSurfaceInputSampleCounts,          // I32 points per sample
    kUsdGenSurfaceInputVelocities,            // F32x3
    kUsdGenSurfaceInputUv,                    // F32x2
    kUsdGenSurfaceInputSubsetFaces,           // I32
    kUsdGenSurfaceInputDensityMultiplier,     // F32
    kUsdGenSurfaceInputSlots
};

enum UsdGenSurfaceInputValue : uint8_t {
    kUsdGenSurfaceValueId,                    // int (UsdGenSurfaceId; the compiler assigns it)
    kUsdGenSurfaceValueSubdivisionScheme,     // TfToken ...
    kUsdGenSurfaceValueOrientation,
    kUsdGenSurfaceValueInterpolateBoundary,
    kUsdGenSurfaceValueFaceVaryingLinearInterpolation,
    kUsdGenSurfaceValueTriangleSubdivisionRule,
    kUsdGenSurfaceValueCreaseMethod,
    kUsdGenSurfaceValueRestNormalDomain,      // int
    kUsdGenSurfaceValueRestFromCurrentPoints, // bool
    kUsdGenSurfaceValueIsSubset,              // bool
    kUsdGenSurfaceValueWorldMatrix,           // GfMatrix4d
    kUsdGenSurfaceValues
};

USDGEN_SURFACE_API UsdGenTypeId UsdGenSurfaceInputKind() noexcept;
USDGEN_SURFACE_API UsdGenTypeId UsdGenRegisterSurfaceInputKind(UsdGenRegistrar &);

/// One surface sample, read from a record or a payload.
struct UsdGenSurfaceSampleView
{
    double time = 0.0;
    UsdGenSpan<const GfVec3f> points;
};

/// The samples of a surface, in time order. No allocation.
class UsdGenSurfaceSamples
{
public:
    UsdGenSurfaceSamples() noexcept = default;
    USDGEN_SURFACE_API static UsdGenSurfaceSamples Of(UsdGenInputRecord const &) noexcept;
    USDGEN_SURFACE_API static UsdGenSurfaceSamples Of(std::vector<UsdGenSurfaceSample> const &) noexcept;
    size_t size() const noexcept { return _count; }
    bool empty() const noexcept { return _count == 0; }
    USDGEN_SURFACE_API UsdGenSurfaceSampleView operator[](size_t i) const noexcept;
    UsdGenSurfaceSampleView front() const noexcept { return (*this)[0]; }
    UsdGenSurfaceSampleView back() const noexcept { return (*this)[_count - 1]; }
    class iterator
    {
    public:
        using value_type = UsdGenSurfaceSampleView;
        using difference_type = std::ptrdiff_t;
        using reference = UsdGenSurfaceSampleView;
        using pointer = void;
        using iterator_category = std::input_iterator_tag;
        UsdGenSurfaceSampleView operator*() const noexcept { return (*_owner)[_i]; }
        iterator &operator++() noexcept { ++_i; return *this; }
        bool operator==(iterator const &o) const noexcept { return _i == o._i; }
        bool operator!=(iterator const &o) const noexcept { return _i != o._i; }
    private:
        friend class UsdGenSurfaceSamples;
        iterator(UsdGenSurfaceSamples const *owner, size_t i) noexcept : _owner(owner), _i(i) {}
        UsdGenSurfaceSamples const *_owner;
        size_t _i;
    };
    iterator begin() const noexcept { return iterator(this, 0); }
    iterator end() const noexcept { return iterator(this, _count); }

private:
    size_t _count = 0;
    UsdGenSurfaceSample const *_samples = nullptr;   // payload-backed
    UsdGenSpan<const GfVec3f> _points;               // record-backed
    UsdGenSpan<const double> _times;
    UsdGenSpan<const int> _counts;
};

/// Read view of a surface input with the payload's field names. Non-owning:
/// valid while its record (or payload) lives. No allocation.
struct UsdGenSurfaceInputView
{
    UsdGenSurfaceInputView() = default;
    /// The view of a record (Of(record)); implicit so a record passes where a view is read.
    UsdGenSurfaceInputView(UsdGenInputRecord const &record) noexcept : UsdGenSurfaceInputView(Of(record)) {}
    USDGEN_SURFACE_API static UsdGenSurfaceInputView Of(UsdGenInputRecord const &) noexcept;
    USDGEN_SURFACE_API static UsdGenSurfaceInputView Of(UsdGenSurfaceData const &,
                                                     UsdGenNodeKey key = {},
                                                     UsdGenNodeKey owner = {}) noexcept;

    UsdGenNodeKey key;
    UsdGenNodeKey owner;   // the Mesh a GeomSubset belongs to (host-filled); else invalid
    UsdGenSurfaceId id = 0;
    UsdGenSpan<const int> faceVertexCounts, faceVertexIndices;
    TfToken subdivisionScheme, orientation, interpolateBoundary, faceVaryingLinearInterpolation,
        triangleSubdivisionRule, creaseMethod;
    UsdGenSpan<const int> holeIndices, creaseIndices, creaseLengths, cornerIndices;
    UsdGenSpan<const float> creaseSharpnesses, cornerSharpnesses;
    UsdGenSpan<const GfVec3f> restPoints, restNormals;
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    bool restFromCurrentPoints = false;
    UsdGenSpan<const GfVec3f> points;
    UsdGenSurfaceSamples samples;
    UsdGenSpan<const GfVec3f> velocities;
    UsdGenSpan<const GfVec2f> uv;
    UsdGenSpan<const int> subsetFaces;
    bool isSubset = false;
    UsdGenSpan<const float> densityMultiplier;
    GfMatrix4d worldMatrix{1.0};
    uint64_t surfaceGeneration = 0;
    /// Keeps a record's columns alive: a view may outlive its spec.
    UsdGenRef<const UsdGenElementSet> data;
};

/// Whether consumers restrict a surface to subsetFaces: a GeomSubset target
/// (even an empty one) or a hand-built input that names faces.
inline bool UsdGenSurfaceRestricted(UsdGenSurfaceInputView const &surface)
{
    return surface.isSubset || !surface.subsetFaces.empty();
}

inline bool UsdGenSurfaceRestricted(UsdGenSurfaceData const &surface)
{
    return surface.isSubset || !surface.subsetFaces.empty();
}

/// The Mesh a surface's geometry and primvars live on: the target itself, or
/// a GeomSubset's parent.
inline UsdGenNodeKey UsdGenSurfaceMeshKey(UsdGenSurfaceInputView const &surface)
{
    return surface.isSubset ? surface.owner : surface.key;
}

/// Every surface input of 'spec'; a surface's id is its index here.
USDGEN_SURFACE_API UsdGenSpan<const UsdGenInputRecord> UsdGenSurfaceInputs(UsdGenGraphSpec const &spec) noexcept;
/// The view of surface 'id' (its index), else an empty view.
USDGEN_SURFACE_API UsdGenSurfaceInputView UsdGenSurfaceAt(UsdGenGraphSpec const &spec, size_t id) noexcept;
/// The surface input 'key', else null.
USDGEN_SURFACE_API UsdGenInputRecord const *UsdGenFindSurfaceInput(UsdGenGraphSpec const &spec,
                                                                UsdGenNodeKey key) noexcept;
/// Sets the compiler-assigned id of a surface record. No allocation.
USDGEN_SURFACE_API void UsdGenSetSurfaceId(UsdGenInputRecord &record, UsdGenSurfaceId id) noexcept;

/// Packs 'data' into a UsdGenSurfaceInput record. Compile phase: allocates.
USDGEN_SURFACE_API UsdGenInputRecord UsdGenSurfaceRecord(UsdGenSurfaceData const &data,
                                                      UsdGenNodeKey key, UsdGenNodeKey owner,
                                                      TfToken const &label,
                                                      UsdGenInputColumnCache *cache = nullptr);
/// The Default-space variant of a surface record: points (and the one sample
/// kept) are its rest points, no velocities, a rest-content generation.
/// Shares every column; false (no record) when the surface has no usable rest.
USDGEN_SURFACE_API bool UsdGenRestSurfaceRecord(UsdGenInputRecord const &posed, UsdGenInputRecord *out);

/// Copies a view's arrays into a payload (engine-made variants of an input).
USDGEN_SURFACE_API void UsdGenCopySurfaceData(UsdGenSurfaceInputView const &view, UsdGenSurfaceData *out);

}  // namespace usdGen
