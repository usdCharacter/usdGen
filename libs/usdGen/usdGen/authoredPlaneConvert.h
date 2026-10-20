// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — generic authored-plane transport for curve sets.
//
// Any primvar named `primvars:usdGen:plane:<name>` on a C3 / reference
// BasisCurves prim is forwarded as the authored plane `<name>` (then an extra
// plane on the curve set's buffer), so operators can read per-curve or per-CV
// data that has no dedicated schema slot: card parameters on card spines,
// tile ids on a strand library, per-guide flags, per-curve trim settings.
//
// Domain follows the primvar interpolation (constant -> Groom, uniform ->
// Primitive, vertex/varying -> Point); arity follows the value type
// (float/int/bool/double -> 1, float2 -> 2, float3 -> 3, float4 -> 4). Both
// builders (stage and Hydra) call ConvertAuthoredPlane with the flattened
// value so the two paths cannot disagree.
#ifndef USDGEN_AUTHORED_PLANE_CONVERT_H
#define USDGEN_AUTHORED_PLANE_CONVERT_H

#include "usdGen/graphDesc.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// "usdGen:plane:" — the primvar namespace (without "primvars:").
inline std::string const &UsdGenAuthoredPlanePrefix()
{
    static std::string const prefix("usdGen:plane:");
    return prefix;
}

/// Converts one flattened primvar value into an authored plane. Returns false
/// (leaving *out untouched) for an unsupported type, an empty name, or an
/// element count that disagrees with the interpolation (`curves` per-curve
/// count, `points` total CV count).
inline bool ConvertAuthoredPlane(TfToken const &planeName, VtValue const &value,
                                 TfToken const &interpolation, size_t curves,
                                 size_t points, UsdGenAuthoredPlaneDesc *out)
{
    if (planeName.IsEmpty() || !out) return false;
    UsdGenAuthoredPlaneDesc plane;
    plane.name = planeName;
    auto floats = [&](float const *data, size_t count, uint8_t arity) {
        plane.type = UsdGenAuthoredPlaneType::Float32;
        plane.arity = arity;
        plane.floatValues.assign(data, data + count * arity);
        return count;
    };
    size_t count = 0;
    if (value.IsHolding<VtFloatArray>()) {
        VtFloatArray const &a = value.UncheckedGet<VtFloatArray>();
        count = floats(a.cdata(), a.size(), 1);
    } else if (value.IsHolding<VtDoubleArray>()) {
        VtDoubleArray const &a = value.UncheckedGet<VtDoubleArray>();
        plane.type = UsdGenAuthoredPlaneType::Float32;
        plane.arity = 1;
        plane.floatValues.resize(a.size());
        for (size_t i = 0; i < a.size(); ++i) plane.floatValues[i] = float(a[i]);
        count = a.size();
    } else if (value.IsHolding<VtVec2fArray>()) {
        VtVec2fArray const &a = value.UncheckedGet<VtVec2fArray>();
        count = floats(a.empty() ? nullptr : a.cdata()->data(), a.size(), 2);
    } else if (value.IsHolding<VtVec3fArray>()) {
        VtVec3fArray const &a = value.UncheckedGet<VtVec3fArray>();
        count = floats(a.empty() ? nullptr : a.cdata()->data(), a.size(), 3);
    } else if (value.IsHolding<VtVec4fArray>()) {
        VtVec4fArray const &a = value.UncheckedGet<VtVec4fArray>();
        count = floats(a.empty() ? nullptr : a.cdata()->data(), a.size(), 4);
    } else if (value.IsHolding<VtIntArray>()) {
        VtIntArray const &a = value.UncheckedGet<VtIntArray>();
        plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.arity = 1;
        plane.intValues = a;
        count = a.size();
    } else if (value.IsHolding<VtBoolArray>()) {
        VtBoolArray const &a = value.UncheckedGet<VtBoolArray>();
        plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.arity = 1;
        plane.intValues.resize(a.size());
        for (size_t i = 0; i < a.size(); ++i) plane.intValues[i] = a[i] ? 1 : 0;
        count = a.size();
    } else if (value.IsHolding<float>() || value.IsHolding<double>()) {
        plane.type = UsdGenAuthoredPlaneType::Float32;
        plane.arity = 1;
        plane.floatValues.assign(1, value.IsHolding<float>() ? value.UncheckedGet<float>()
                                                             : float(value.UncheckedGet<double>()));
        count = 1;
    } else if (value.IsHolding<int>() || value.IsHolding<bool>()) {
        plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.arity = 1;
        plane.intValues.assign(1, value.IsHolding<int>() ? value.UncheckedGet<int>()
                                                         : (value.UncheckedGet<bool>() ? 1 : 0));
        count = 1;
    } else {
        return false;
    }
    if (count == 0) return false;
    bool const vertexLike = interpolation == TfToken("vertex") || interpolation == TfToken("varying");
    if (interpolation == TfToken("uniform") && count == curves) {
        plane.domain = UsdGenAuthoredPlaneDomain::Primitive;
    } else if (vertexLike && count == points) {
        plane.domain = UsdGenAuthoredPlaneDomain::Point;
    } else if (count == 1 && (interpolation == TfToken("constant") || interpolation.IsEmpty())) {
        plane.domain = UsdGenAuthoredPlaneDomain::Groom;
    } else {
        return false;
    }
    *out = std::move(plane);
    return true;
}

}  // namespace usdGen

#endif  // USDGEN_AUTHORED_PLANE_CONVERT_H
