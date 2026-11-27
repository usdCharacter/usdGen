// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd/convert/value.h: the value conversions.
#include "usdGenUsd/convert/value.h"
#include "usdGenUsd/convert/token.h"

#include "pxr/base/gf/timeCode.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/opaqueValue.h"
#include "pxr/usd/sdf/pathExpression.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace usdGenUsd {

namespace {

using usdGen::UsdGenValue;
using usdGen::UsdGenValueKind;

/// A host value with no engine row.
struct HostValueBox final : usdGen::UsdGenBoxed, usdGen::UsdGenPooled {
    static constexpr uint64_t kBoxTag = usdGen::UsdGenBoxTag("usdGenUsd.vtValue");
    explicit HostValueBox(PXR_NS::VtValue v) : value(std::move(v)) {}
    uint64_t Hash() const noexcept override { return uint64_t(value.GetHash()); }
    bool Equal(usdGen::UsdGenBoxed const &o) const noexcept override
    {
        return value == static_cast<HostValueBox const &>(o).value;
    }
    PXR_NS::VtValue value;
};

usdGen::UsdGenAssetPath ToEngineAsset(PXR_NS::SdfAssetPath const &a)
{
    return usdGen::UsdGenAssetPath{a.GetAuthoredPath(), a.GetEvaluatedPath(), a.GetResolvedPath()};
}

PXR_NS::SdfAssetPath ToHostAsset(usdGen::UsdGenAssetPath const &a)
{
    if (a.authored.empty() && a.evaluated.empty() && a.resolved.empty()) return PXR_NS::SdfAssetPath();
    return PXR_NS::SdfAssetPath(
        PXR_NS::SdfAssetPathParams().Authored(a.authored).Evaluated(a.evaluated).Resolved(a.resolved));
}

PXR_NS::SdfPathExpression ToHostExpression(usdGen::UsdGenPathExpression const &e)
{
    return e.text.empty() ? PXR_NS::SdfPathExpression() : PXR_NS::SdfPathExpression(e.text);
}

/// Host scalar H -> its engine row.
usdGen::UsdGenToken EngineScalar(PXR_NS::TfToken const &t) { return ToToken(t); }
usdGen::UsdGenTimeCode EngineScalar(PXR_NS::GfTimeCode const &t) { return usdGen::UsdGenTimeCode(t.GetValue()); }
usdGen::UsdGenAssetPath EngineScalar(PXR_NS::SdfAssetPath const &a) { return ToEngineAsset(a); }
usdGen::UsdGenPathExpression EngineScalar(PXR_NS::SdfPathExpression const &e) { return usdGen::UsdGenPathExpression{e.GetText()}; }
usdGen::UsdGenOpaque EngineScalar(PXR_NS::SdfOpaqueValue const &) { return usdGen::UsdGenOpaque(); }
template <class H, std::enable_if_t<std::is_arithmetic<H>::value || std::is_same<H, std::string>::value, int> = 0>
H EngineScalar(H const &h) { return h; }
template <class H, std::enable_if_t<!std::is_void<EngineMathOf<H>>::value, int> = 0>
EngineMathOf<H> EngineScalar(H const &h) { return ToEngine(h); }

template <class H>
bool TryScalar(PXR_NS::VtValue const &v, UsdGenValue *out)
{
    if (!v.IsHolding<H>()) return false;
    *out = UsdGenValue(EngineScalar(v.UncheckedGet<H>()));
    return true;
}

template <class H>
bool TryArray(PXR_NS::VtValue const &v, UsdGenValue *out)
{
    if (!v.IsHolding<PXR_NS::VtArray<H>>()) return false;
    *out = UsdGenValue(ToCow(v.UncheckedGet<PXR_NS::VtArray<H>>()));
    return true;
}

template <class H, class E, class F>
bool TryVector(PXR_NS::VtValue const &v, UsdGenValue *out, F convert)
{
    if (!v.IsHolding<PXR_NS::VtArray<H>>()) return false;
    PXR_NS::VtArray<H> const &a = v.UncheckedGet<PXR_NS::VtArray<H>>();
    std::vector<E> e;
    e.reserve(a.size());
    for (H const &h : a) e.push_back(convert(h));
    *out = UsdGenValue(std::move(e));
    return true;
}

/// An engine value already in a VtValue (tests, wrapped values).
template <class T>
bool TryEngine(PXR_NS::VtValue const &v, UsdGenValue *out)
{
    if (!v.IsHolding<T>()) return false;
    *out = UsdGenValue(v.UncheckedGet<T>());
    return true;
}

template <class... H>
bool AnyScalar(PXR_NS::VtValue const &v, UsdGenValue *out) { return (... || TryScalar<H>(v, out)); }
template <class... H>
bool AnyArray(PXR_NS::VtValue const &v, UsdGenValue *out) { return (... || TryArray<H>(v, out)); }
template <class... T>
bool AnyEngine(PXR_NS::VtValue const &v, UsdGenValue *out) { return (... || TryEngine<T>(v, out)); }

#define USDGEN_HOST_MATH_TYPES                                                                    \
    PXR_NS::GfVec2f, PXR_NS::GfVec3f, PXR_NS::GfVec4f, PXR_NS::GfVec2d, PXR_NS::GfVec3d,          \
    PXR_NS::GfVec4d, PXR_NS::GfVec2i, PXR_NS::GfVec3i, PXR_NS::GfVec4i, PXR_NS::GfVec2h,          \
    PXR_NS::GfVec3h, PXR_NS::GfVec4h, PXR_NS::GfMatrix2d, PXR_NS::GfMatrix3d, PXR_NS::GfMatrix4d, \
    PXR_NS::GfQuatf, PXR_NS::GfQuatd, PXR_NS::GfQuath, PXR_NS::GfHalf
#define USDGEN_HOST_SCALAR_TYPES                                                                  \
    bool, unsigned char, int, unsigned int, int64_t, uint64_t, float, double, std::string,       \
    PXR_NS::GfTimeCode, PXR_NS::TfToken, PXR_NS::SdfAssetPath, PXR_NS::SdfPathExpression,         \
    PXR_NS::SdfOpaqueValue, USDGEN_HOST_MATH_TYPES
#define USDGEN_HOST_ARRAY_TYPES                                                                   \
    bool, unsigned char, int, unsigned int, int64_t, uint64_t, float, double, PXR_NS::GfTimeCode, \
    PXR_NS::TfToken, USDGEN_HOST_MATH_TYPES

#define USDGEN_ENGINE_TYPE(K, T) , T
#define USDGEN_ENGINE_ARRAY_TYPE(K, T) , usdGen::UsdGenCowArray<T>
#define USDGEN_ENGINE_VECTOR_TYPE(K, T) , std::vector<T>

bool EngineHeld(PXR_NS::VtValue const &v, UsdGenValue *out)
{
    using namespace usdGen;
    return AnyEngine<bool USDGEN_VALUE_INLINE_KINDS(USDGEN_ENGINE_TYPE) USDGEN_VALUE_HEAP_KINDS(USDGEN_ENGINE_TYPE)
                     USDGEN_VALUE_ARRAY_ELEMENTS(USDGEN_ENGINE_ARRAY_TYPE)
                     USDGEN_VALUE_VECTOR_ELEMENTS(USDGEN_ENGINE_VECTOR_TYPE)>(v, out);
}

/// Engine row -> host value.
template <class T> PXR_NS::VtValue HostOf(T const &t) { return PXR_NS::VtValue(t); }
PXR_NS::VtValue HostOf(usdGen::UsdGenToken t) { return PXR_NS::VtValue(ToTf(t)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenTimeCode t) { return PXR_NS::VtValue(PXR_NS::GfTimeCode(t.value)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenAssetPath const &a) { return PXR_NS::VtValue(ToHostAsset(a)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenPathExpression const &e) { return PXR_NS::VtValue(ToHostExpression(e)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenOpaque) { return PXR_NS::VtValue(PXR_NS::SdfOpaqueValue()); }
PXR_NS::VtValue HostOf(usdGen::UsdGenGroup) { return PXR_NS::VtValue(PXR_NS::SdfOpaqueValue()); }
PXR_NS::VtValue HostOf(usdGen::UsdGenHalf h) { return PXR_NS::VtValue(ToHost(h)); }
template <class S, int N>
PXR_NS::VtValue HostOf(usdGen::UsdGenVec<S, N> const &v) { return PXR_NS::VtValue(ToHost(v)); }
template <class S>
PXR_NS::VtValue HostOf(usdGen::UsdGenQuat<S> const &q) { return PXR_NS::VtValue(ToHost(q)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenMatrix2d const &m) { return PXR_NS::VtValue(ToHost(m)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenMatrix3d const &m) { return PXR_NS::VtValue(ToHost(m)); }
PXR_NS::VtValue HostOf(usdGen::UsdGenMatrix4d const &m) { return PXR_NS::VtValue(ToHost(m)); }
template <class E>
PXR_NS::VtValue HostOf(usdGen::UsdGenCowArray<E> const &a) { return PXR_NS::VtValue(ToVt(a)); }
PXR_NS::VtValue HostOf(std::vector<std::string> const &a)
{
    return PXR_NS::VtValue(PXR_NS::VtArray<std::string>(a.begin(), a.end()));
}
PXR_NS::VtValue HostOf(std::vector<usdGen::UsdGenAssetPath> const &a)
{
    PXR_NS::VtArray<PXR_NS::SdfAssetPath> out;
    out.reserve(a.size());
    for (usdGen::UsdGenAssetPath const &e : a) out.push_back(ToHostAsset(e));
    return PXR_NS::VtValue(std::move(out));
}
PXR_NS::VtValue HostOf(std::vector<usdGen::UsdGenPathExpression> const &a)
{
    PXR_NS::VtArray<PXR_NS::SdfPathExpression> out;
    out.reserve(a.size());
    for (usdGen::UsdGenPathExpression const &e : a) out.push_back(ToHostExpression(e));
    return PXR_NS::VtValue(std::move(out));
}

}  // namespace

usdGen::UsdGenValue ToEngineValue(PXR_NS::VtValue const &v)
{
    UsdGenValue out;
    if (v.IsEmpty()) return out;
    if (v.IsHolding<UsdGenValue>()) return v.UncheckedGet<UsdGenValue>();
    if (v.IsArrayValued()) {
        if (AnyArray<USDGEN_HOST_ARRAY_TYPES>(v, &out)) return out;
        if (TryVector<std::string, std::string>(v, &out, [](std::string const &s) { return s; })) return out;
        if (TryVector<PXR_NS::SdfAssetPath, usdGen::UsdGenAssetPath>(v, &out, &ToEngineAsset)) return out;
        if (TryVector<PXR_NS::SdfPathExpression, usdGen::UsdGenPathExpression>(
                v, &out, [](PXR_NS::SdfPathExpression const &e) { return usdGen::UsdGenPathExpression{e.GetText()}; }))
            return out;
    } else if (AnyScalar<USDGEN_HOST_SCALAR_TYPES>(v, &out)) {
        return out;
    }
    if (EngineHeld(v, &out)) return out;
    return UsdGenValue::Box(HostValueBox::kBoxTag, usdGen::UsdGenRef<const usdGen::UsdGenBoxed>(new HostValueBox(v)));
}

PXR_NS::VtValue ToHostValue(usdGen::UsdGenValue const &v)
{
    using namespace usdGen;
    switch (v.Kind()) {
    case UsdGenValueKind::Empty: return PXR_NS::VtValue();
    case UsdGenValueKind::Boxed:
        if (HostValueBox const *box = v.GetBoxed<HostValueBox>()) return box->value;
        return PXR_NS::VtValue(v);
#define USDGEN_HOST_OF(K, T) case UsdGenValueKind::K: return HostOf(v.UncheckedGet<T>());
    USDGEN_VALUE_INLINE_KINDS(USDGEN_HOST_OF)
    USDGEN_VALUE_HEAP_KINDS(USDGEN_HOST_OF)
#undef USDGEN_HOST_OF
    case UsdGenValueKind::Array:
        switch (v.ElementKind()) {
#define USDGEN_HOST_OF_ARRAY(K, T) case UsdGenValueKind::K: return HostOf(v.UncheckedGet<usdGen::UsdGenCowArray<T>>());
#define USDGEN_HOST_OF_VECTOR(K, T) case UsdGenValueKind::K: return HostOf(v.UncheckedGet<std::vector<T>>());
        USDGEN_VALUE_ARRAY_ELEMENTS(USDGEN_HOST_OF_ARRAY)
        USDGEN_VALUE_VECTOR_ELEMENTS(USDGEN_HOST_OF_VECTOR)
#undef USDGEN_HOST_OF_ARRAY
#undef USDGEN_HOST_OF_VECTOR
        default: break;
        }
        break;
    default: break;
    }
    return PXR_NS::VtValue();
}

}  // namespace usdGenUsd
