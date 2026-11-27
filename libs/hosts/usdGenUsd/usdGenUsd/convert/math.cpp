// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd/convert/math.h: the value conversions.
#include "usdGenUsd/convert/math.h"

namespace usdGenUsd {

namespace {

template <class H>
bool EngineScalar(PXR_NS::VtValue const &v, PXR_NS::VtValue *out)
{
    if (v.IsHolding<H>()) {
        *out = PXR_NS::VtValue(ToEngine(v.UncheckedGet<H>()));
        return true;
    }
    if (v.IsHolding<PXR_NS::VtArray<H>>()) {
        *out = PXR_NS::VtValue(ToEngine(v.UncheckedGet<PXR_NS::VtArray<H>>()));
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
    if (v.IsHolding<PXR_NS::VtArray<E>>()) {
        *out = PXR_NS::VtValue(ToHost(v.UncheckedGet<PXR_NS::VtArray<E>>()));
        return true;
    }
    return false;
}

template <template <class> class Op, class... T>
PXR_NS::VtValue Convert(PXR_NS::VtValue const &v)
{
    PXR_NS::VtValue out;
    bool const done = (... || Op<T>::Run(v, &out));
    return done ? out : v;
}

template <class H> struct ToEngineOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return EngineScalar<H>(v, o); } };
template <class H> struct ToHostOp { static bool Run(PXR_NS::VtValue const &v, PXR_NS::VtValue *o) { return HostScalar<EngineMathOf<H>>(v, o); } };

#define USDGEN_MATH_HOST_TYPES                                                                    \
    PXR_NS::GfVec2f, PXR_NS::GfVec3f, PXR_NS::GfVec4f, PXR_NS::GfVec2d, PXR_NS::GfVec3d,          \
    PXR_NS::GfVec4d, PXR_NS::GfVec2i, PXR_NS::GfVec3i, PXR_NS::GfVec4i, PXR_NS::GfVec2h,          \
    PXR_NS::GfVec3h, PXR_NS::GfVec4h, PXR_NS::GfMatrix2d, PXR_NS::GfMatrix3d, PXR_NS::GfMatrix4d, \
    PXR_NS::GfMatrix4f, PXR_NS::GfQuatf, PXR_NS::GfQuatd, PXR_NS::GfQuath, PXR_NS::GfRange3f,     \
    PXR_NS::GfRange3d, PXR_NS::GfHalf

}  // namespace

PXR_NS::VtValue ToEngineValue(PXR_NS::VtValue const &v)
{
    if (v.IsEmpty()) return v;
    return Convert<ToEngineOp, USDGEN_MATH_HOST_TYPES>(v);
}

PXR_NS::VtValue ToHostValue(PXR_NS::VtValue const &v)
{
    if (v.IsEmpty()) return v;
    return Convert<ToHostOp, USDGEN_MATH_HOST_TYPES>(v);
}

}  // namespace usdGenUsd
