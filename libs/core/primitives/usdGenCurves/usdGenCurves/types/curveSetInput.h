// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CurveSet input kind (step 7.3a): the C3 BasisCurves payload a host
// captures (UsdGenCurveSetData), its input record layout and InputDesc, the
// packer, the read view with the payload's field names, and the reference
// set its materialize hook derives.
#pragma once
#include "usdGen/api/graphSpec.h"   // pxr gf/tf/vt types (interim, step 11)
#include "usdGenCurves/types/curveTokens.h"
#include "usdGen/compile/types.h"
#include "usdGenCurves/export.h"
#include "usdGen/registries/engineTokens.h"


#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace usdGen {

class UsdGenRegistrar;

/// A named plane authored on a C3 curve set: Point becomes vertex, Primitive
/// uniform, and Groom one constant value in the CPU CurveSource buffer.
enum class UsdGenAuthoredPlaneType : uint8_t { Float32, Int32 };
enum class UsdGenAuthoredPlaneDomain : uint8_t { Point, Primitive, Groom };

struct UsdGenAuthoredPlaneDesc
{
    UsdGenToken name;
    UsdGenAuthoredPlaneType type = UsdGenAuthoredPlaneType::Float32;
    UsdGenAuthoredPlaneDomain domain = UsdGenAuthoredPlaneDomain::Point;
    uint8_t arity = 1;                 // [1, 16]
    UsdGenFloatArray floatValues;          // populated only for Float32
    UsdGenIntArray intValues;              // populated only for Int32
};

/// Immutable surface-cage controls captured alongside a CurveSource's sparse
/// C3 rails, in the authored owner/triangle order the runtime interpolator
/// uses. Chart positions are captured independently of the surface map so a
/// density or profile edit changes the capture identity without re-reading.
struct UsdGenSurfaceCagePayload
{
    UsdGenIntArray   ownerIds;
    UsdGenFloatArray ownerDensities;
    UsdGenIntArray   ownerSeeds;
    UsdGenIntArray   ownerCvCounts;
    UsdGenFloatArray ownerEdgeBias;
    UsdGenIntArray   ownerLengthProfileOffsets;
    UsdGenVec2fArray ownerLengthProfile;
    UsdGenFloatArray normalizedT;           // flattened per-point rail parameter
    UsdGenVec3iArray triangles;              // guide-curve indices
    UsdGenIntArray   triangleOwnerIndices;
    UsdGenVec2fArray triangleRootCharts;     // 3 entries per triangle
    UsdGenVec2fArray ownerChartCentroids;
    UsdGenFloatArray ownerChartMeanRadii;
};

/// The source curves' own primvars:displayColor, forwarded as an authored
/// plane under this reserved name. It rides the named-plane machinery so
/// resampling and compaction carry it, but the session cooker consumes it
/// into the tile's displayColor instead of publishing it. 'displayColor'
/// itself is reserved by the compiler for authored planes.
inline UsdGenToken const &UsdGenSourceColorPlane()
{
    static UsdGenToken const name("usdGenSourceColor");
    return name;
}

/// One authored C3 BasisCurves input as a host captures it: a guide set, a
/// freeze, an import, a sim cache. The only way curve data reaches the
/// engine (R23); it travels as a UsdGenCurveSetInput record.
struct UsdGenCurveSetData
{
    UsdGenLane      role;              // Reference (guides, clump centres) | Chunked (a CurveSource)
    UsdGenToken         curveRole;         // primvars:usdGen:role: hair | guide (C3 marker)
    UsdGenIntArray      curveVertexCounts;
    UsdGenVec3fArray    points, rest;      // rest == primvars:rest; may share points' buffer
    // Compatibility builders may lack a Default-time C3 snapshot. CUDA
    // admission must not mistake a current-frame fallback for bound rest.
    bool            restFromCurrentPoints = false;
    UsdGenMatrix4d      worldMatrix{1.0}; // source object space must be explicit for deformation
    UsdGenFloatArray    widths;
    UsdGenToken         type = UsdGenCurveTokens().cubic;
    UsdGenToken         basis = UsdGenCurveTokens().bspline;
    UsdGenToken         wrap = UsdGenCurveTokens().pinned;
    UsdGenToken         widthsInterpolation = UsdGenCurveTokens().vertexWidths;
    UsdGenIntArray      skinPrim;          // primvars:skinprim (uniform int)
    UsdGenCowArray<uint64_t> curveId;         // primvars:usdGen:curveId (uniform uint64[], R12)
    UsdGenVec2fArray    skinPrimUv;        // primvars:skinprimuv (uniform texCoord2f, not "st")
    UsdGenMatrix4dArray rootFrame;         // primvars:usdGen:rootFrame; may be empty
    std::string     frozenEpoch;       // constant string primvar, "usdgen1:sha1:..." (S42)
    std::vector<UsdGenAuthoredPlaneDesc> authoredPlanes;
    std::shared_ptr<const UsdGenSurfaceCagePayload> surfaceCage;
    uint64_t        curveGeneration = 0;
    /// Keeps the record's columns alive (a view may outlive its spec).
    UsdGenRef<const UsdGenElementSet> data;   // bumped by any points/topology/id change on the prim
};

/// Record columns of the UsdGenCurveSetInput kind (slot == enumerator). Every
/// column is free-length: the payload array as captured.
enum UsdGenCurveInputSlot : UsdGenSlot {
    kUsdGenCurveInputPoints,             // F32x3
    kUsdGenCurveInputCounts,             // I32 curveVertexCounts
    kUsdGenCurveInputRest,               // F32x3
    kUsdGenCurveInputWidths,             // F32
    kUsdGenCurveInputSkinPrim,           // I32
    kUsdGenCurveInputCurveId,            // U64
    kUsdGenCurveInputSkinPrimUv,         // F32x2
    kUsdGenCurveInputRootFrame,          // F64x16
    kUsdGenCurveInputPlaneInfo,          // I32x5 per authored plane: type, domain, arity, floats, ints
    kUsdGenCurveInputPlaneFloats,        // F32, every Float32 plane in order
    kUsdGenCurveInputPlaneInts,          // I32, every Int32 plane in order
    kUsdGenCurveInputCageOwnerIds,       // the surface cage (UsdGenSurfaceCagePayload) ...
    kUsdGenCurveInputCageOwnerDensities,
    kUsdGenCurveInputCageOwnerSeeds,
    kUsdGenCurveInputCageOwnerCvCounts,
    kUsdGenCurveInputCageOwnerEdgeBias,
    kUsdGenCurveInputCageOwnerLengthProfileOffsets,
    kUsdGenCurveInputCageOwnerLengthProfile,
    kUsdGenCurveInputCageNormalizedT,
    kUsdGenCurveInputCageTriangles,
    kUsdGenCurveInputCageTriangleOwnerIndices,
    kUsdGenCurveInputCageTriangleRootCharts,
    kUsdGenCurveInputCageOwnerChartCentroids,
    kUsdGenCurveInputCageOwnerChartMeanRadii,
    kUsdGenCurveInputSlots
};

/// Record values of the kind, in the InputDesc's attribute order.
enum UsdGenCurveInputValue : uint8_t {
    kUsdGenCurveValueRole,                  // int (UsdGenLane)
    kUsdGenCurveValueCurveRole,             // UsdGenToken
    kUsdGenCurveValueType,                  // UsdGenToken
    kUsdGenCurveValueBasis,                 // UsdGenToken
    kUsdGenCurveValueWrap,                  // UsdGenToken
    kUsdGenCurveValueWidthsInterpolation,   // UsdGenToken
    kUsdGenCurveValueRestFromCurrentPoints, // bool
    kUsdGenCurveValueWorldMatrix,           // UsdGenMatrix4d
    kUsdGenCurveValueFrozenEpoch,           // std::string
    kUsdGenCurveValuePlaneNames,            // UsdGenTokenArray, authored order
    kUsdGenCurveValueSurfaceCage,           // bool: the cage columns are present
    kUsdGenCurveValues
};

/// Valid after usdGenRegister_usdGenCurves ran (the op registry's first use).
USDGEN_CURVES_API UsdGenTypeId UsdGenCurveSetInputKind() noexcept;
/// Registers "UsdGenCurveSetInput" with its layout, InputDesc and the
/// reference-set materialize hook.
USDGEN_CURVES_API UsdGenTypeId UsdGenRegisterCurveSetInputKind(UsdGenRegistrar &);

/// One authored plane, read from a record or a payload.
struct UsdGenAuthoredPlaneView
{
    UsdGenToken name;
    UsdGenAuthoredPlaneType type = UsdGenAuthoredPlaneType::Float32;
    UsdGenAuthoredPlaneDomain domain = UsdGenAuthoredPlaneDomain::Point;
    uint8_t arity = 1;
    UsdGenSpan<const float> floatValues;
    UsdGenSpan<const int> intValues;
};

/// The authored planes of a curve set, in authored order (duplicates and
/// invalid enumerators kept for the compiler's checks). No allocation.
class UsdGenAuthoredPlanes
{
public:
    UsdGenAuthoredPlanes() noexcept = default;
    USDGEN_CURVES_API static UsdGenAuthoredPlanes Of(UsdGenInputRecord const &) noexcept;
    USDGEN_CURVES_API static UsdGenAuthoredPlanes Of(std::vector<UsdGenAuthoredPlaneDesc> const &) noexcept;
    size_t size() const noexcept { return _count; }
    bool empty() const noexcept { return _count == 0; }
    USDGEN_CURVES_API UsdGenAuthoredPlaneView operator[](size_t i) const noexcept;
    class iterator
    {
    public:
        using value_type = UsdGenAuthoredPlaneView;
        using difference_type = std::ptrdiff_t;
        using reference = UsdGenAuthoredPlaneView;
        using pointer = void;
        using iterator_category = std::input_iterator_tag;
        UsdGenAuthoredPlaneView operator*() const noexcept { return _owner->_At(_i, _floats, _ints); }
        iterator &operator++() noexcept
        {
            _owner->_Advance(_i, &_floats, &_ints);
            ++_i;
            return *this;
        }
        bool operator==(iterator const &o) const noexcept { return _i == o._i; }
        bool operator!=(iterator const &o) const noexcept { return _i != o._i; }
    private:
        friend class UsdGenAuthoredPlanes;
        iterator(UsdGenAuthoredPlanes const *owner, size_t i) noexcept : _owner(owner), _i(i) {}
        UsdGenAuthoredPlanes const *_owner;
        size_t _i;
        size_t _floats = 0, _ints = 0;
    };
    iterator begin() const noexcept { return iterator(this, 0); }
    iterator end() const noexcept { return iterator(this, _count); }

private:
    USDGEN_CURVES_API UsdGenAuthoredPlaneView _At(size_t i, size_t floats, size_t ints) const noexcept;
    USDGEN_CURVES_API void _Advance(size_t i, size_t *floats, size_t *ints) const noexcept;
    size_t _count = 0;
    UsdGenAuthoredPlaneDesc const *_descs = nullptr;   // payload-backed
    UsdGenTokenArray _names;                               // record-backed (shared, no copy)
    UsdGenSpan<const int> _info;
    UsdGenSpan<const float> _floats;
    UsdGenSpan<const int> _ints;
};

/// The surface cage, field for field (UsdGenSurfaceCagePayload).
struct UsdGenSurfaceCageView
{
    UsdGenSpan<const int>     ownerIds;
    UsdGenSpan<const float>   ownerDensities;
    UsdGenSpan<const int>     ownerSeeds;
    UsdGenSpan<const int>     ownerCvCounts;
    UsdGenSpan<const float>   ownerEdgeBias;
    UsdGenSpan<const int>     ownerLengthProfileOffsets;
    UsdGenSpan<const UsdGenVec2f> ownerLengthProfile;
    UsdGenSpan<const float>   normalizedT;
    UsdGenSpan<const UsdGenVec3i> triangles;
    UsdGenSpan<const int>     triangleOwnerIndices;
    UsdGenSpan<const UsdGenVec2f> triangleRootCharts;
    UsdGenSpan<const UsdGenVec2f> ownerChartCentroids;
    UsdGenSpan<const float>   ownerChartMeanRadii;
};

/// An optional cage: tests like the payload's shared_ptr.
class UsdGenSurfaceCageRef
{
public:
    UsdGenSurfaceCageRef() noexcept = default;
    explicit UsdGenSurfaceCageRef(UsdGenSurfaceCageView const &cage) noexcept
        : _cage(cage), _present(true) {}
    explicit operator bool() const noexcept { return _present; }
    UsdGenSurfaceCageView const *operator->() const noexcept { return &_cage; }
    UsdGenSurfaceCageView const &operator*() const noexcept { return _cage; }

private:
    UsdGenSurfaceCageView _cage;
    bool _present = false;
};

/// Read view of a curve set input with the payload's field names. Non-owning:
/// valid while its record (or payload) lives. No allocation.
struct UsdGenCurveSetInputView
{
    UsdGenCurveSetInputView() = default;
    /// The view of a record (Of(record)); implicit so a record passes where a view is read.
    UsdGenCurveSetInputView(UsdGenInputRecord const &record) noexcept : UsdGenCurveSetInputView(Of(record)) {}
    USDGEN_CURVES_API static UsdGenCurveSetInputView Of(UsdGenInputRecord const &) noexcept;
    /// A payload not packed into a record (engine-made transients, tests).
    USDGEN_CURVES_API static UsdGenCurveSetInputView Of(UsdGenCurveSetData const &,
                                                      UsdGenNodeKey key = {}) noexcept;

    UsdGenNodeKey   key;
    UsdGenLane      role = UsdGenLane::Chunked;
    UsdGenToken         curveRole;
    UsdGenSpan<const int> curveVertexCounts;
    UsdGenSpan<const UsdGenVec3f> points, rest;
    bool            restFromCurrentPoints = false;
    UsdGenMatrix4d      worldMatrix{1.0};
    UsdGenSpan<const float> widths;
    UsdGenToken         type, basis, wrap, widthsInterpolation;
    UsdGenSpan<const int> skinPrim;
    UsdGenSpan<const uint64_t> curveId;
    UsdGenSpan<const UsdGenVec2f> skinPrimUv;
    UsdGenSpan<const UsdGenMatrix4d> rootFrame;
    std::string_view frozenEpoch;
    UsdGenAuthoredPlanes authoredPlanes;
    UsdGenSurfaceCageRef surfaceCage;
    uint64_t        curveGeneration = 0;
    /// Keep a record's columns and values alive: a view may outlive its spec.
    UsdGenRef<const UsdGenElementSet> data;
    UsdGenValue         frozenEpochHolder;
};

/// Every curve set input of 'spec', builder order.
USDGEN_CURVES_API UsdGenSpan<const UsdGenInputRecord> UsdGenCurveSetInputs(UsdGenGraphSpec const &spec) noexcept;
/// The curve set input 'key', else null.
USDGEN_CURVES_API UsdGenInputRecord const *UsdGenFindCurveSetInput(UsdGenGraphSpec const &spec,
                                                                 UsdGenNodeKey key) noexcept;

/// Packs 'data' into a UsdGenCurveSetInput record. Compile phase: allocates.
USDGEN_CURVES_API UsdGenInputRecord UsdGenCurveSetRecord(UsdGenCurveSetData const &data,
                                                       UsdGenNodeKey key, UsdGenToken const &label,
                                                       UsdGenInputColumnCache *cache = nullptr);

/// Copies a view's arrays into a payload (engine-made variants of an input).
USDGEN_CURVES_API void UsdGenCopyCurveSetData(UsdGenCurveSetInputView const &view,
                                            UsdGenCurveSetData *out);

}  // namespace usdGen
