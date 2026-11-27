// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd/convert/math.h: the value conversions.
#include "usdGenUsd/convert/math.h"
#include "usdGenUsd/convert/token.h"

#include <cstdint>
#include <string>
#include <vector>

namespace usdGenUsd {

namespace {

template <class H>
bool EngineScalar(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    if (v.IsHolding<H>()) {
        *out = PXR_NS::VtValue(ToEngine(v.UncheckedGet<H>()));
        return true;
    }
    return false;
}

template <class E>
bool HostScalar(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    if (v.IsHolding<E>()) {
        *out = PXR_NS::VtValue(ToHost(v.UncheckedGet<E>()));
        return true;
    }
    return false;
}

template <class H>
bool EngineArray(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    if (!v.IsHolding<PXR_NS::VtArray<H>>()) return false;
    *out = PXR_NS::VtValue(ToCow(v.UncheckedGet<PXR_NS::VtArray<H>>()));
    return true;
}

template <class E>
bool HostArray(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    if (!v.IsHolding<usdGen::UsdGenCowArray<E>>()) return false;
    *out = PXR_NS::VtValue(ToVt(v.UncheckedGet<usdGen::UsdGenCowArray<E>>()));
    return true;
}

template <template <class> class Op, class... T>
bool TryConvert(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    return (... || Op<T>::Run(v, out));
}

template <template <class> class Op, class... T>
PXR_NS::VtValue Convert(PXR_NS::VtValue const &v)
{
    PXR_NS::VtValue out;
    return TryConvert<Op, T...>(v, &out) ? out : v;
}

template <class H> struct ToEngineOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return EngineScalar<H>(v, o); } };
template <class H> struct ToHostOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return HostScalar<EngineMathOf<H>>(v, o); } };
template <class H> struct ToEngineArrayOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return EngineArray<H>(v, o); } };
template <class H> struct ToHostArrayOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return HostArray<EngineElemOf<H>>(v, o); } };

#define USDGEN_MATH_HOST_TYPES                                                                    \
    PXR_NS::GfVec2f, PXR_NS::GfVec3f, PXR_NS::GfVec4f, PXR_NS::GfVec2d, PXR_NS::GfVec3d,          \
    PXR_NS::GfVec4d, PXR_NS::GfVec2i, PXR_NS::GfVec3i, PXR_NS::GfVec4i, PXR_NS::GfVec2h,          \
    PXR_NS::GfVec3h, PXR_NS::GfVec4h, PXR_NS::GfMatrix2d, PXR_NS::GfMatrix3d, PXR_NS::GfMatrix4d, \
    PXR_NS::GfMatrix4f, PXR_NS::GfQuatf, PXR_NS::GfQuatd, PXR_NS::GfQuath, PXR_NS::GfRange3f,     \
    PXR_NS::GfRange3d, PXR_NS::GfHalf

// Every VtArray element the engine has an array of (the value-type table's rows).
#define USDGEN_ARRAY_HOST_TYPES                                                                  \
    bool, unsigned char, int, unsigned int, int64_t, uint64_t, float, double, PXR_NS::GfTimeCode, \
    PXR_NS::TfToken, USDGEN_MATH_HOST_TYPES

using StringArray = std::vector<std::string>;

}  // namespace

PXR_NS::VtValue ToEngineValue(PXR_NS::VtValue const &v)
{
    if (v.IsEmpty()) return v;
    if (v.IsHolding<PXR_NS::TfToken>()) return PXR_NS::VtValue(ToToken(v.UncheckedGet<PXR_NS::TfToken>()));
    if (v.IsArrayValued()) {
        if (v.IsHolding<PXR_NS::VtArray<std::string>>()) {
            PXR_NS::VtArray<std::string> const &a = v.UncheckedGet<PXR_NS::VtArray<std::string>>();
            return PXR_NS::VtValue(StringArray(a.begin(), a.end()));
        }
        return Convert<ToEngineArrayOp, USDGEN_ARRAY_HOST_TYPES>(v);
    }
    return Convert<ToEngineOp, USDGEN_MATH_HOST_TYPES>(v);
}

PXR_NS::VtValue ToHostValue(PXR_NS::VtValue const &v)
{
    if (v.IsEmpty()) return v;
    if (v.IsHolding<usdGen::UsdGenToken>()) return PXR_NS::VtValue(ToTf(v.UncheckedGet<usdGen::UsdGenToken>()));
    if (v.IsHolding<StringArray>()) {
        StringArray const &a = v.UncheckedGet<StringArray>();
        return PXR_NS::VtValue(PXR_NS::VtArray<std::string>(a.begin(), a.end()));
    }
    PXR_NS::VtValue array;
    if (TryConvert<ToHostArrayOp, USDGEN_ARRAY_HOST_TYPES>(v, &array)) return array;
    return Convert<ToHostOp, USDGEN_MATH_HOST_TYPES>(v);
}

}  // namespace usdGenUsd
