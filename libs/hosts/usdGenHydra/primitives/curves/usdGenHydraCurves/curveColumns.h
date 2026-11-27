// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The CurveSet's columns and the surface inputs, read through usdGen/api by
// channel and attribute name: the adapter never links a primitive. Names are
// interned when the adapter registers (usdGenRegister_usdGenHydraCurves).
#pragma once
#include "usdGenHydraCurves/export.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"

#include <cstddef>
#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// One dynamic channel of a curve set as a primvar plane (constant, uniform
/// or vertex; "float" or "int").
struct UsdGenCurvePlane {
    UsdGenToken name, interpolation, type;
    uint8_t arity = 1;
    UsdGenSpan<const float> f;
    UsdGenSpan<const int> i;
};

/// The CurveSet static channel slots of one channel table, cached on its
/// digest. Caller-owned; one per reader, not shared between threads.
struct UsdGenCurveSlots {
    enum : uint8_t { Px, Py, Pz, Rest, Width, HairT, CurveId, RootPrim, RootUV, RootT, RootN, RootB, Count };
    uint64_t digest = 0;               // 0: unresolved
    UsdGenSlot slot[Count] = {};
    bool ok = false;
};

/// A curve set's columns with UsdGenCurveBuffer's field names. Non-owning:
/// valid while the set lives. No allocation.
struct UsdGenCurveColumns {
    /// Empty when the set lacks a CurveSet static channel or the adapter has
    /// not registered. 'slots' (optional) caches the name lookups.
    USDGEN_HYDRA_CURVES_API static UsdGenCurveColumns Of(UsdGenElementSetView const &,
                                                         UsdGenCurveSlots *slots = nullptr) noexcept;
    static UsdGenCurveColumns Of(UsdGenElementSet const &s, UsdGenCurveSlots *slots = nullptr) noexcept
    {
        return Of(s.View(), slots);
    }
    UsdGenSpan<const float> px, py, pz, width, hairT;
    UsdGenSpan<const GfVec3f> rest, rootT, rootN, rootB;
    UsdGenSpan<const uint64_t> curveId;
    UsdGenSpan<const int> rootPrim;
    UsdGenSpan<const GfVec2f> rootUV;
    UsdGenSpan<const int> cvOffsets;       // empty = uniform (stored uint32, same bits)
    uint32_t totalCurves = 0, totalCvs = 0;
    uint64_t topologyVersion = 0, valueVersion = 0;
    /// The levels 0-1 dynamic run and the level-2 run, in name order.
    USDGEN_HYDRA_CURVES_API uint32_t ExtraCurveCount() const noexcept;
    USDGEN_HYDRA_CURVES_API UsdGenCurvePlane ExtraCurve(uint32_t) const noexcept;
    USDGEN_HYDRA_CURVES_API uint32_t ExtraCvCount() const noexcept;
    USDGEN_HYDRA_CURVES_API UsdGenCurvePlane ExtraCv(uint32_t) const noexcept;
private:
    UsdGenElementSetView _set{};
};

/// The type named UsdGenCurveSet; Invalid before the curve primitive registered.
USDGEN_HYDRA_CURVES_API UsdGenTypeId UsdGenHydraCurveSetType() noexcept;

/// The source curves' forwarded displayColor plane (reserved name
/// "usdGenSourceColor"): consumed into a tile's displayColor, never published.
USDGEN_HYDRA_CURVES_API UsdGenToken const &UsdGenHydraSourceColorPlane() noexcept;

/// The spec's surface inputs (the input kind with the Surface role), in
/// compiler id order.
USDGEN_HYDRA_CURVES_API UsdGenSpan<const UsdGenInputRecord>
UsdGenHydraSurfaceInputs(UsdGenGraphSpec const &spec) noexcept;
/// The surface 'key', else null.
USDGEN_HYDRA_CURVES_API UsdGenInputRecord const *
UsdGenHydraFindSurface(UsdGenGraphSpec const &spec, UsdGenNodeKey key) noexcept;
/// A surface record's compiler-assigned id.
USDGEN_HYDRA_CURVES_API uint32_t UsdGenHydraSurfaceId(UsdGenInputRecord const &) noexcept;

/// The triangles a surface record carries, at the graph time.
struct UsdGenHydraSurfaceMesh {
    UsdGenSpan<const GfVec3f> points;
    UsdGenSpan<const int> faceVertexCounts, faceVertexIndices;
    GfMatrix4d worldMatrix{1.0};
};
USDGEN_HYDRA_CURVES_API UsdGenHydraSurfaceMesh UsdGenHydraSurfaceMeshOf(UsdGenInputRecord const &) noexcept;

/// Registration only (usdGenRegister_usdGenHydraCurves): interns the names
/// above and resolves the curve type and the surface kind. False with a
/// reason when UsdGenCurveSet is not registered.
bool UsdGenHydraCurvesResolveNames(char const **reason);

}  // namespace usdGen
