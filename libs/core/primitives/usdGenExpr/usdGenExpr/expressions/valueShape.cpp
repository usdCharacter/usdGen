// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenExpr/expressions/valueShape.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"

#include "usdGenMath/math.h"
#include "usdGen/api/cowArray.h"

#include <cstddef>
#include <cstring>
#include <string>


namespace usdGen::expr {
namespace {

template <class T> UsdGenValue _Value() { return UsdGenValue(T()); }
template <class T> UsdGenValue _Identity() { return UsdGenValue(T(1.0)); }
template <class T> UsdGenValue _Array() { return UsdGenValue(UsdGenCowArray<T>()); }
UsdGenValue _Half() { return UsdGenValue(UsdGenHalf(0.0f)); }
// UsdGenTimeCode comes with vt/types.h (through vt/value.h).

// Sdf's value type names plus the legacy names SdfSchema still resolves
// (Vec3f, Point, Transform, PointIndex, ...). Sorted by name (byte order);
// _Sorted() checks it at compile time.
constexpr UsdGenValueTypeInfo kTypes[] = {
    {"Color", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"ColorFloat", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"ColorFloat[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"Color[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"EdgeIndex", ScalarType::Int32, 0, 0, 0, false, &_Value<int>},
    {"EdgeIndex[]", ScalarType::Int32, 0, 0, 0, true, &_Array<int>},
    {"FaceIndex", ScalarType::Int32, 0, 0, 0, false, &_Value<int>},
    {"FaceIndex[]", ScalarType::Int32, 0, 0, 0, true, &_Array<int>},
    {"Frame", ScalarType::Float64, 2, 4, 4, false, &_Identity<UsdGenMatrix4d>},
    {"Frame[]", ScalarType::Float64, 2, 4, 4, true, &_Array<UsdGenMatrix4d>},
    {"Matrix2d", ScalarType::Float64, 2, 2, 2, false, &_Identity<UsdGenMatrix2d>},
    {"Matrix2d[]", ScalarType::Float64, 2, 2, 2, true, &_Array<UsdGenMatrix2d>},
    {"Matrix3d", ScalarType::Float64, 2, 3, 3, false, &_Identity<UsdGenMatrix3d>},
    {"Matrix3d[]", ScalarType::Float64, 2, 3, 3, true, &_Array<UsdGenMatrix3d>},
    {"Matrix4d", ScalarType::Float64, 2, 4, 4, false, &_Identity<UsdGenMatrix4d>},
    {"Matrix4d[]", ScalarType::Float64, 2, 4, 4, true, &_Array<UsdGenMatrix4d>},
    {"Normal", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"NormalFloat", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"NormalFloat[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"Normal[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"Point", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"PointFloat", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"PointFloat[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"PointIndex", ScalarType::Int32, 0, 0, 0, false, &_Value<int>},
    {"PointIndex[]", ScalarType::Int32, 0, 0, 0, true, &_Array<int>},
    {"Point[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"Quatd", ScalarType::Float64, 1, 4, 0, false, &_Identity<UsdGenQuatd>},
    {"Quatd[]", ScalarType::Float64, 1, 4, 0, true, &_Array<UsdGenQuatd>},
    {"Quatf", ScalarType::Float32, 1, 4, 0, false, &_Identity<UsdGenQuatf>},
    {"Quatf[]", ScalarType::Float32, 1, 4, 0, true, &_Array<UsdGenQuatf>},
    {"Quath", ScalarType::Float16, 1, 4, 0, false, &_Identity<UsdGenQuath>},
    {"Quath[]", ScalarType::Float16, 1, 4, 0, true, &_Array<UsdGenQuath>},
    {"Transform", ScalarType::Float64, 2, 4, 4, false, &_Identity<UsdGenMatrix4d>},
    {"Transform[]", ScalarType::Float64, 2, 4, 4, true, &_Array<UsdGenMatrix4d>},
    {"Vec2d", ScalarType::Float64, 1, 2, 0, false, &_Value<UsdGenVec2d>},
    {"Vec2d[]", ScalarType::Float64, 1, 2, 0, true, &_Array<UsdGenVec2d>},
    {"Vec2f", ScalarType::Float32, 1, 2, 0, false, &_Value<UsdGenVec2f>},
    {"Vec2f[]", ScalarType::Float32, 1, 2, 0, true, &_Array<UsdGenVec2f>},
    {"Vec2h", ScalarType::Float16, 1, 2, 0, false, &_Value<UsdGenVec2h>},
    {"Vec2h[]", ScalarType::Float16, 1, 2, 0, true, &_Array<UsdGenVec2h>},
    {"Vec2i", ScalarType::Int32, 1, 2, 0, false, &_Value<UsdGenVec2i>},
    {"Vec2i[]", ScalarType::Int32, 1, 2, 0, true, &_Array<UsdGenVec2i>},
    {"Vec3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"Vec3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"Vec3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"Vec3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"Vec3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"Vec3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"Vec3i", ScalarType::Int32, 1, 3, 0, false, &_Value<UsdGenVec3i>},
    {"Vec3i[]", ScalarType::Int32, 1, 3, 0, true, &_Array<UsdGenVec3i>},
    {"Vec4d", ScalarType::Float64, 1, 4, 0, false, &_Value<UsdGenVec4d>},
    {"Vec4d[]", ScalarType::Float64, 1, 4, 0, true, &_Array<UsdGenVec4d>},
    {"Vec4f", ScalarType::Float32, 1, 4, 0, false, &_Value<UsdGenVec4f>},
    {"Vec4f[]", ScalarType::Float32, 1, 4, 0, true, &_Array<UsdGenVec4f>},
    {"Vec4h", ScalarType::Float16, 1, 4, 0, false, &_Value<UsdGenVec4h>},
    {"Vec4h[]", ScalarType::Float16, 1, 4, 0, true, &_Array<UsdGenVec4h>},
    {"Vec4i", ScalarType::Int32, 1, 4, 0, false, &_Value<UsdGenVec4i>},
    {"Vec4i[]", ScalarType::Int32, 1, 4, 0, true, &_Array<UsdGenVec4i>},
    {"Vector", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"VectorFloat", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"VectorFloat[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"Vector[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"asset", ScalarType::Invalid, 0, 0, 0, false, nullptr},
    {"asset[]", ScalarType::Invalid, 0, 0, 0, true, nullptr},
    {"bool", ScalarType::Bool, 0, 0, 0, false, &_Value<bool>},
    {"bool[]", ScalarType::Bool, 0, 0, 0, true, &_Array<bool>},
    {"color3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"color3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"color3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"color3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"color3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"color3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"color4d", ScalarType::Float64, 1, 4, 0, false, &_Value<UsdGenVec4d>},
    {"color4d[]", ScalarType::Float64, 1, 4, 0, true, &_Array<UsdGenVec4d>},
    {"color4f", ScalarType::Float32, 1, 4, 0, false, &_Value<UsdGenVec4f>},
    {"color4f[]", ScalarType::Float32, 1, 4, 0, true, &_Array<UsdGenVec4f>},
    {"color4h", ScalarType::Float16, 1, 4, 0, false, &_Value<UsdGenVec4h>},
    {"color4h[]", ScalarType::Float16, 1, 4, 0, true, &_Array<UsdGenVec4h>},
    {"double", ScalarType::Float64, 0, 0, 0, false, &_Value<double>},
    {"double2", ScalarType::Float64, 1, 2, 0, false, &_Value<UsdGenVec2d>},
    {"double2[]", ScalarType::Float64, 1, 2, 0, true, &_Array<UsdGenVec2d>},
    {"double3", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"double3[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"double4", ScalarType::Float64, 1, 4, 0, false, &_Value<UsdGenVec4d>},
    {"double4[]", ScalarType::Float64, 1, 4, 0, true, &_Array<UsdGenVec4d>},
    {"double[]", ScalarType::Float64, 0, 0, 0, true, &_Array<double>},
    {"float", ScalarType::Float32, 0, 0, 0, false, &_Value<float>},
    {"float2", ScalarType::Float32, 1, 2, 0, false, &_Value<UsdGenVec2f>},
    {"float2[]", ScalarType::Float32, 1, 2, 0, true, &_Array<UsdGenVec2f>},
    {"float3", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"float3[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"float4", ScalarType::Float32, 1, 4, 0, false, &_Value<UsdGenVec4f>},
    {"float4[]", ScalarType::Float32, 1, 4, 0, true, &_Array<UsdGenVec4f>},
    {"float[]", ScalarType::Float32, 0, 0, 0, true, &_Array<float>},
    {"frame4d", ScalarType::Float64, 2, 4, 4, false, &_Identity<UsdGenMatrix4d>},
    {"frame4d[]", ScalarType::Float64, 2, 4, 4, true, &_Array<UsdGenMatrix4d>},
    {"group", ScalarType::Invalid, 0, 0, 0, false, nullptr},
    {"half", ScalarType::Float16, 0, 0, 0, false, &_Half},
    {"half2", ScalarType::Float16, 1, 2, 0, false, &_Value<UsdGenVec2h>},
    {"half2[]", ScalarType::Float16, 1, 2, 0, true, &_Array<UsdGenVec2h>},
    {"half3", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"half3[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"half4", ScalarType::Float16, 1, 4, 0, false, &_Value<UsdGenVec4h>},
    {"half4[]", ScalarType::Float16, 1, 4, 0, true, &_Array<UsdGenVec4h>},
    {"half[]", ScalarType::Float16, 0, 0, 0, true, &_Array<UsdGenHalf>},
    {"int", ScalarType::Int32, 0, 0, 0, false, &_Value<int>},
    {"int2", ScalarType::Int32, 1, 2, 0, false, &_Value<UsdGenVec2i>},
    {"int2[]", ScalarType::Int32, 1, 2, 0, true, &_Array<UsdGenVec2i>},
    {"int3", ScalarType::Int32, 1, 3, 0, false, &_Value<UsdGenVec3i>},
    {"int3[]", ScalarType::Int32, 1, 3, 0, true, &_Array<UsdGenVec3i>},
    {"int4", ScalarType::Int32, 1, 4, 0, false, &_Value<UsdGenVec4i>},
    {"int4[]", ScalarType::Int32, 1, 4, 0, true, &_Array<UsdGenVec4i>},
    {"int64", ScalarType::Int64, 0, 0, 0, false, &_Value<int64_t>},
    {"int64[]", ScalarType::Int64, 0, 0, 0, true, &_Array<int64_t>},
    {"int[]", ScalarType::Int32, 0, 0, 0, true, &_Array<int>},
    {"matrix2d", ScalarType::Float64, 2, 2, 2, false, &_Identity<UsdGenMatrix2d>},
    {"matrix2d[]", ScalarType::Float64, 2, 2, 2, true, &_Array<UsdGenMatrix2d>},
    {"matrix3d", ScalarType::Float64, 2, 3, 3, false, &_Identity<UsdGenMatrix3d>},
    {"matrix3d[]", ScalarType::Float64, 2, 3, 3, true, &_Array<UsdGenMatrix3d>},
    {"matrix4d", ScalarType::Float64, 2, 4, 4, false, &_Identity<UsdGenMatrix4d>},
    {"matrix4d[]", ScalarType::Float64, 2, 4, 4, true, &_Array<UsdGenMatrix4d>},
    {"normal3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"normal3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"normal3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"normal3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"normal3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"normal3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"opaque", ScalarType::Invalid, 0, 0, 0, false, nullptr},
    {"pathExpression", ScalarType::Invalid, 0, 0, 0, false, nullptr},
    {"pathExpression[]", ScalarType::Invalid, 0, 0, 0, true, nullptr},
    {"point3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"point3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"point3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"point3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"point3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"point3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"quatd", ScalarType::Float64, 1, 4, 0, false, &_Identity<UsdGenQuatd>},
    {"quatd[]", ScalarType::Float64, 1, 4, 0, true, &_Array<UsdGenQuatd>},
    {"quatf", ScalarType::Float32, 1, 4, 0, false, &_Identity<UsdGenQuatf>},
    {"quatf[]", ScalarType::Float32, 1, 4, 0, true, &_Array<UsdGenQuatf>},
    {"quath", ScalarType::Float16, 1, 4, 0, false, &_Identity<UsdGenQuath>},
    {"quath[]", ScalarType::Float16, 1, 4, 0, true, &_Array<UsdGenQuath>},
    {"string", ScalarType::Invalid, 0, 0, 0, false, &_Value<std::string>},
    {"string[]", ScalarType::Invalid, 0, 0, 0, true, &_Array<std::string>},
    {"texCoord2d", ScalarType::Float64, 1, 2, 0, false, &_Value<UsdGenVec2d>},
    {"texCoord2d[]", ScalarType::Float64, 1, 2, 0, true, &_Array<UsdGenVec2d>},
    {"texCoord2f", ScalarType::Float32, 1, 2, 0, false, &_Value<UsdGenVec2f>},
    {"texCoord2f[]", ScalarType::Float32, 1, 2, 0, true, &_Array<UsdGenVec2f>},
    {"texCoord2h", ScalarType::Float16, 1, 2, 0, false, &_Value<UsdGenVec2h>},
    {"texCoord2h[]", ScalarType::Float16, 1, 2, 0, true, &_Array<UsdGenVec2h>},
    {"texCoord3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"texCoord3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"texCoord3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"texCoord3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"texCoord3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"texCoord3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
    {"timecode", ScalarType::Invalid, 0, 0, 0, false, &_Value<UsdGenTimeCode>},
    {"timecode[]", ScalarType::Invalid, 0, 0, 0, true, &_Array<UsdGenTimeCode>},
    {"token", ScalarType::Invalid, 0, 0, 0, false, &_Value<UsdGenToken>},
    {"token[]", ScalarType::Invalid, 0, 0, 0, true, &_Array<UsdGenToken>},
    {"uchar", ScalarType::Invalid, 0, 0, 0, false, &_Value<unsigned char>},
    {"uchar[]", ScalarType::Invalid, 0, 0, 0, true, &_Array<unsigned char>},
    {"uint", ScalarType::UInt32, 0, 0, 0, false, &_Value<unsigned int>},
    {"uint64", ScalarType::UInt64, 0, 0, 0, false, &_Value<uint64_t>},
    {"uint64[]", ScalarType::UInt64, 0, 0, 0, true, &_Array<uint64_t>},
    {"uint[]", ScalarType::UInt32, 0, 0, 0, true, &_Array<unsigned int>},
    {"vector3d", ScalarType::Float64, 1, 3, 0, false, &_Value<UsdGenVec3d>},
    {"vector3d[]", ScalarType::Float64, 1, 3, 0, true, &_Array<UsdGenVec3d>},
    {"vector3f", ScalarType::Float32, 1, 3, 0, false, &_Value<UsdGenVec3f>},
    {"vector3f[]", ScalarType::Float32, 1, 3, 0, true, &_Array<UsdGenVec3f>},
    {"vector3h", ScalarType::Float16, 1, 3, 0, false, &_Value<UsdGenVec3h>},
    {"vector3h[]", ScalarType::Float16, 1, 3, 0, true, &_Array<UsdGenVec3h>},
};
constexpr size_t kTypeCount = sizeof(kTypes) / sizeof(kTypes[0]);

constexpr int _Compare(char const *a, char const *b)
{
    while (*a && *a == *b) { ++a; ++b; }
    return static_cast<unsigned char>(*a) - static_cast<unsigned char>(*b);
}

constexpr bool _Sorted()
{
    for (size_t i = 1; i < kTypeCount; ++i)
        if (_Compare(kTypes[i - 1].name, kTypes[i].name) >= 0) return false;
    return true;
}
static_assert(_Sorted(), "kTypes must be sorted by name with no duplicates");

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

UsdGenValueTypeInfo
FindValueType(char const *nativeType)
{
    if (!nativeType || !*nativeType) return {};
    size_t lo = 0, hi = kTypeCount;
    while (lo < hi) {
        size_t const mid = lo + (hi - lo) / 2;
        int const c = std::strcmp(kTypes[mid].name, nativeType);
        if (c == 0) return kTypes[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return {};
}

UsdGenValueTypeInfo
FindValueType(UsdGenToken const &nativeType)
{
    return FindValueType(nativeType.GetText());
}

UsdGenSpan<const UsdGenValueTypeInfo>
AllValueTypes()
{
    return UsdGenSpan<const UsdGenValueTypeInfo>(kTypes, kTypeCount);
}

UsdGenValue
ValueTypeDefault(UsdGenValueTypeInfo const &type)
{
    return type.makeDefault ? type.makeDefault() : UsdGenValue();
}

ValueShape
ValueShapeFromValueType(UsdGenValueTypeInfo const &type,
                        uint32_t arrayElementCount,
                        bool arrayCountKnown)
{
    if (!type) return _InvalidArrayShape(false);
    bool const isArray = type.isArray;
    if (isArray && !arrayCountKnown) return _InvalidArrayShape(true);
    // `uchar` deliberately remains unsupported: ScalarType has no UInt8,
    // and treating its byte storage as UInt32 would corrupt transport.
    if (type.scalar == ScalarType::Invalid) return _InvalidArrayShape(isArray);

    ValueShape result;
    result.scalar = type.scalar;
    result.isArray = isArray;
    result.elementCount = isArray ? arrayElementCount : 1;
    uint32_t const d0 = type.dimension0;
    uint32_t const d1 = type.dimension1;
    if (type.dimensionCount == 0) return result;
    if (type.dimensionCount == 1 && d0 >= 2 && d0 <= 4) {
        result.components = d0;
        return result;
    }
    if (type.dimensionCount == 2 && d0 >= 2 && d0 <= 4 && d1 >= 2 && d1 <= 4) {
        result.rows = d0;
        result.columns = d1;
        return result;
    }
    return _InvalidArrayShape(isArray);
}

ValueShape
ValueShapeFromNativeType(UsdGenToken const &nativeType,
                         uint32_t arrayElementCount,
                         bool arrayCountKnown)
{
    return ValueShapeFromValueType(FindValueType(nativeType),
                                   arrayElementCount, arrayCountKnown);
}

} // namespace usdGen::expr
