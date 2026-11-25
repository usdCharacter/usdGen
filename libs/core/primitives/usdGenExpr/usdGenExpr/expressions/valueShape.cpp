#include "usdGen/expressions/valueShape.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix2d.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec4i.h"
#include "pxr/base/tf/type.h"
#include "pxr/usd/sdf/schema.h"

#include <limits>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen::expr {
namespace {

ScalarType
_ScalarTypeFor(TfType const& type)
{
    if (type == TfType::Find<bool>()) return ScalarType::Bool;
    if (type == TfType::Find<int>()) return ScalarType::Int32;
    if (type == TfType::Find<unsigned int>()) return ScalarType::UInt32;
    if (type == TfType::Find<int64_t>()) return ScalarType::Int64;
    if (type == TfType::Find<uint64_t>()) return ScalarType::UInt64;
    if (type == TfType::Find<GfHalf>()) return ScalarType::Float16;
    if (type == TfType::Find<float>()) return ScalarType::Float32;
    if (type == TfType::Find<double>()) return ScalarType::Float64;

    if (type == TfType::Find<GfVec2i>() || type == TfType::Find<GfVec3i>() ||
        type == TfType::Find<GfVec4i>()) return ScalarType::Int32;
    if (type == TfType::Find<GfVec2h>() || type == TfType::Find<GfVec3h>() ||
        type == TfType::Find<GfVec4h>() || type == TfType::Find<GfQuath>())
        return ScalarType::Float16;
    if (type == TfType::Find<GfVec2f>() || type == TfType::Find<GfVec3f>() ||
        type == TfType::Find<GfVec4f>() || type == TfType::Find<GfQuatf>())
        return ScalarType::Float32;
    if (type == TfType::Find<GfVec2d>() || type == TfType::Find<GfVec3d>() ||
        type == TfType::Find<GfVec4d>() || type == TfType::Find<GfQuatd>() ||
        type == TfType::Find<GfMatrix2d>() || type == TfType::Find<GfMatrix3d>() ||
        type == TfType::Find<GfMatrix4d>()) return ScalarType::Float64;
    // `uchar` deliberately remains unsupported: ScalarType has no UInt8,
    // and treating its byte storage as UInt32 would corrupt transport.
    return ScalarType::Invalid;
}

ValueShape
_InvalidArrayShape(bool isArray)
{
    ValueShape result;
    result.isArray = isArray;
    result.elementCount = 0;
    result.scalar = ScalarType::Invalid;
    return result;
}

} // namespace

ValueShape
ValueShapeFromSdfType(SdfValueTypeName const& type,
                      uint32_t arrayElementCount,
                      bool arrayCountKnown)
{
    if (!type) return _InvalidArrayShape(false);
    bool const isArray = type.IsArray();
    if (isArray && !arrayCountKnown) return _InvalidArrayShape(true);

    SdfValueTypeName const scalarType = type.GetScalarType();
    if (!scalarType) return _InvalidArrayShape(isArray);
    ScalarType const scalar = _ScalarTypeFor(scalarType.GetType());
    if (scalar == ScalarType::Invalid) return _InvalidArrayShape(isArray);

    ValueShape result;
    result.scalar = scalar;
    result.isArray = isArray;
    result.elementCount = isArray ? arrayElementCount : 1;
    SdfTupleDimensions const dimensions = scalarType.GetDimensions();
    if (dimensions.size == 0) return result;
    if (dimensions.size == 1 && dimensions.d[0] >= 2 && dimensions.d[0] <= 4) {
        result.components = static_cast<uint32_t>(dimensions.d[0]);
        return result;
    }
    if (dimensions.size == 2 && dimensions.d[0] >= 2 && dimensions.d[0] <= 4 &&
        dimensions.d[1] >= 2 && dimensions.d[1] <= 4) {
        result.rows = static_cast<uint32_t>(dimensions.d[0]);
        result.columns = static_cast<uint32_t>(dimensions.d[1]);
        return result;
    }
    return _InvalidArrayShape(isArray);
}

ValueShape
ValueShapeFromNativeType(TfToken const& nativeType,
                         uint32_t arrayElementCount,
                         bool arrayCountKnown)
{
    return ValueShapeFromSdfType(SdfSchema::GetInstance().FindType(nativeType),
                                 arrayElementCount, arrayCountKnown);
}

bool
FixedArrayElementCount(SdfValueTypeName const& type, VtValue const& value,
                       uint32_t* count)
{
    if (!count || !type.IsArray() || value.IsEmpty() || !value.IsArrayValued() ||
        !type.CanRepresent(value) ||
        value.GetArraySize() > std::numeric_limits<uint32_t>::max()) return false;
    *count = static_cast<uint32_t>(value.GetArraySize());
    return true;
}

} // namespace usdGen::expr
