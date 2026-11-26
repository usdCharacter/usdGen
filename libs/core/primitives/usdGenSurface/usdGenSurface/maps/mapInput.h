// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The Map input kind (step 7.3a): a UsdGenMap prim as a host captures it
// (UsdGenMapData), its record layout and InputDesc, the packer and the read
// view with the payload's field names.
#pragma once
#include "usdGen/api/graphSpec.h"   // pxr gf/tf/vt types (interim, step 11)
#include "usdGen/registries/typeRegistry.h"


#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace usdGen {

class UsdGenImagePayload;
class UsdGenRegistrar;

/// One map prim as a host captures it; travels as a UsdGenMapInput record.
struct UsdGenMapData
{
    TfToken      type;                // UsdGenImageMap | UsdGenPtexMap | UsdGenExprMap | ...
    std::string  resolvedAssetPath;   // stage-free (S13)
    uint64_t     textureGeneration = 0;  // bumped by ReloadMaps()
    // UsdGenPaintMap payload (empty unless type == UsdGenPaintMap): the
    // surface primvar snapshotted at capture, scalar-folded through
    // usdGen:map:channel. v1 carries faceVarying float data only; any other
    // interpolation, a missing primvar, or a size mismatch leaves
    // paintValues empty and the sampler fails closed.
    TfToken      paintPrimvar;
    TfToken      paintInterpolation;
    VtFloatArray paintValues;
    std::vector<UsdGenParamValue> params;
    // Optional stage-free decoded pixels for a typed ImageMap consumer. The
    // immutable payload is backend-neutral; callers replacing it must also
    // advance textureGeneration, which is the deterministic cache identity.
    std::shared_ptr<const UsdGenImagePayload> imagePayload;
    UsdGenTypeId typeId = UsdGenTypeId::Invalid;   // engine-filled at intake
};

enum UsdGenMapInputSlot : UsdGenSlot {
    kUsdGenMapInputPaintValues,     // F32
    kUsdGenMapInputSlots
};

/// Record values of the kind; the map's own parameters follow them.
enum UsdGenMapInputValue : uint8_t {
    kUsdGenMapValueType,                // TfToken
    kUsdGenMapValueResolvedAssetPath,   // std::string
    kUsdGenMapValuePaintPrimvar,        // TfToken
    kUsdGenMapValuePaintInterpolation,  // TfToken
    kUsdGenMapValueImagePayload,        // std::shared_ptr<const UsdGenImagePayload>
    kUsdGenMapValueTypeId,              // int (UsdGenTypeId; the intake hook fills it)
    kUsdGenMapValues
};

USDGEN_CORE_API UsdGenTypeId UsdGenMapInputKind() noexcept;
USDGEN_CORE_API UsdGenTypeId UsdGenRegisterMapInputKind(UsdGenRegistrar &);

/// Read view of a map input with the payload's field names. Non-owning:
/// valid while its record (or payload) lives. No allocation.
struct UsdGenMapInputView
{
    UsdGenMapInputView() = default;
    /// The view of a record (Of(record)); implicit so a record passes where a view is read.
    UsdGenMapInputView(UsdGenInputRecord const &record) noexcept : UsdGenMapInputView(Of(record)) {}
    USDGEN_CORE_API static UsdGenMapInputView Of(UsdGenInputRecord const &) noexcept;
    USDGEN_CORE_API static UsdGenMapInputView Of(UsdGenMapData const &, UsdGenNodeKey key = {},
                                                 UsdGenNodeKey paintSurface = {}) noexcept;

    UsdGenNodeKey key;
    UsdGenNodeKey paintSurface;       // UsdGenPaintMap: the surface carrying the primvar
    TfToken      type;
    std::string_view resolvedAssetPath;
    uint64_t     textureGeneration = 0;
    TfToken      paintPrimvar;
    TfToken      paintInterpolation;
    UsdGenSpan<const float> paintValues;
    UsdGenSpan<const UsdGenParamValue> params;
    std::shared_ptr<const UsdGenImagePayload> imagePayload;
    UsdGenTypeId typeId = UsdGenTypeId::Invalid;
    /// Keep a record's columns and path alive; 'params' still points into
    /// the record, so a stored map view needs its record.
    UsdGenRef<const UsdGenElementSet> data;
    VtValue resolvedAssetPathHolder;
};

/// The map's type id: the intake-resolved one, else a registry probe (inputs
/// built outside the compiler). Never interns.
inline UsdGenTypeId UsdGenMapTypeId(UsdGenMapInputView const &map) noexcept
{
    return map.typeId != UsdGenTypeId::Invalid ? map.typeId
                                               : UsdGenTypeRegistry::Get().Find(map.type);
}

/// True when the map's registered type carries \p trait (kUsdGenTypeMap*).
inline bool UsdGenMapHas(UsdGenMapInputView const &map, uint32_t trait) noexcept
{
    return UsdGenTypeRegistry::Get().Has(UsdGenMapTypeId(map), trait);
}

/// Every map input of 'spec', builder order.
USDGEN_CORE_API UsdGenSpan<const UsdGenInputRecord> UsdGenMapInputs(UsdGenGraphSpec const &spec) noexcept;
/// The map input 'key', else null.
USDGEN_CORE_API UsdGenInputRecord const *UsdGenFindMapInput(UsdGenGraphSpec const &spec,
                                                            UsdGenNodeKey key) noexcept;

/// Packs 'data' into a UsdGenMapInput record. Compile phase: allocates.
USDGEN_CORE_API UsdGenInputRecord UsdGenMapRecord(UsdGenMapData const &data, UsdGenNodeKey key,
                                                  UsdGenNodeKey paintSurface, TfToken const &label,
                                                  UsdGenInputColumnCache *cache = nullptr);

/// Copies a view into a payload (engine-made variants of an input).
USDGEN_CORE_API void UsdGenCopyMapData(UsdGenMapInputView const &view, UsdGenMapData *out);

}  // namespace usdGen
