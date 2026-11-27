// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/value.h"

#include <cstring>

namespace usdGen {
namespace {

using value_detail::HeapOf;
using value_detail::Store;

// FNV-1a 64, one round per byte of each word.
uint64_t Mix(uint64_t h, uint64_t bits) noexcept { return math_detail::HashMix(h, bits); }
constexpr uint64_t kSeed = 0xcbf29ce484222325ull;

uint64_t Bits(bool v) noexcept { return v ? 1u : 0u; }
uint64_t Bits(unsigned char v) noexcept { return v; }
uint64_t Bits(int v) noexcept { return uint64_t(uint32_t(v)); }
uint64_t Bits(unsigned int v) noexcept { return v; }
uint64_t Bits(int64_t v) noexcept { return uint64_t(v); }
uint64_t Bits(uint64_t v) noexcept { return v; }
uint64_t Bits(float v) noexcept { return math_detail::HashBits(v); }
uint64_t Bits(double v) noexcept { return math_detail::HashBits(v); }
uint64_t Bits(UsdGenHalf v) noexcept { return math_detail::HashBits(v); }

uint64_t HashText(uint64_t h, std::string const &s) noexcept
{
    h = Mix(h, s.size());
    for (char c : s) {
        h ^= uint64_t(static_cast<unsigned char>(c));
        h *= 0x100000001b3ull;
    }
    return h;
}

template <class S>
uint64_t HashScalars(uint64_t h, S const *p, size_t n) noexcept
{
    for (size_t i = 0; i < n; ++i) h = Mix(h, Bits(p[i]));
    return h;
}

template <class T> uint64_t HashOf(uint64_t h, T const &v) noexcept { return Mix(h, Bits(v)); }
uint64_t HashOf(uint64_t h, UsdGenTimeCode const &v) noexcept { return Mix(h, Bits(v.value)); }
uint64_t HashOf(uint64_t h, UsdGenToken const &v) noexcept { return Mix(h, v.Hash()); }
uint64_t HashOf(uint64_t h, UsdGenOpaque const &) noexcept { return h; }
uint64_t HashOf(uint64_t h, UsdGenGroup const &) noexcept { return h; }
template <class S, int N>
uint64_t HashOf(uint64_t h, UsdGenVec<S, N> const &v) noexcept { return HashScalars(h, v.data(), size_t(N)); }
uint64_t HashOf(uint64_t h, UsdGenMatrix2d const &m) noexcept { return HashScalars(h, m.data(), 4); }
uint64_t HashOf(uint64_t h, UsdGenMatrix3d const &m) noexcept { return HashScalars(h, m.data(), 9); }
uint64_t HashOf(uint64_t h, UsdGenMatrix4d const &m) noexcept { return HashScalars(h, m.data(), 16); }
template <class S>
uint64_t HashOf(uint64_t h, UsdGenQuat<S> const &q) noexcept
{
    h = HashOf(h, q.GetImaginary());
    return Mix(h, Bits(q.GetReal()));
}
uint64_t HashOf(uint64_t h, std::string const &v) noexcept { return HashText(h, v); }
uint64_t HashOf(uint64_t h, UsdGenAssetPath const &v) noexcept
{
    return HashText(HashText(HashText(h, v.authored), v.evaluated), v.resolved);
}
uint64_t HashOf(uint64_t h, UsdGenPathExpression const &v) noexcept { return HashText(h, v.text); }
template <class E>
uint64_t HashOf(uint64_t h, UsdGenCowArray<E> const &a) noexcept
{
    h = Mix(h, a.size());
    for (E const &e : a) h = HashOf(h, e);
    return h;
}
template <class E>
uint64_t HashOf(uint64_t h, std::vector<E> const &a) noexcept
{
    h = Mix(h, a.size());
    for (E const &e : a) h = HashOf(h, e);
    return h;
}

/// Calls f(T const *) with the held object (null when the kind is unknown).
template <class F>
void Visit(UsdGenValue const &v, F &&f)
{
    switch (v.Kind()) {
#define USDGEN_VISIT(K, T) case UsdGenValueKind::K: f(&v.UncheckedGet<T>()); return;
    USDGEN_VALUE_INLINE_KINDS(USDGEN_VISIT)
    USDGEN_VALUE_HEAP_KINDS(USDGEN_VISIT)
#undef USDGEN_VISIT
    case UsdGenValueKind::Array:
        switch (v.ElementKind()) {
#define USDGEN_VISIT_ARRAY(K, T) case UsdGenValueKind::K: f(&v.UncheckedGet<UsdGenCowArray<T>>()); return;
#define USDGEN_VISIT_VECTOR(K, T) case UsdGenValueKind::K: f(&v.UncheckedGet<std::vector<T>>()); return;
        USDGEN_VALUE_ARRAY_ELEMENTS(USDGEN_VISIT_ARRAY)
        USDGEN_VALUE_VECTOR_ELEMENTS(USDGEN_VISIT_VECTOR)
#undef USDGEN_VISIT_ARRAY
#undef USDGEN_VISIT_VECTOR
        default: return;
        }
    default: return;
    }
}

template <class T> struct Tag { using Type = T; };

/// Calls f(Tag<T>) for a heap kind's C++ type.
template <class F>
void VisitHeapType(UsdGenValueKind kind, UsdGenValueKind element, F &&f)
{
    if (kind == UsdGenValueKind::Array) {
        switch (element) {
#define USDGEN_VISIT_VECTOR(K, T) case UsdGenValueKind::K: f(Tag<std::vector<T>>()); return;
        USDGEN_VALUE_VECTOR_ELEMENTS(USDGEN_VISIT_VECTOR)
#undef USDGEN_VISIT_VECTOR
        default: return;
        }
    }
    switch (kind) {
#define USDGEN_VISIT_HEAP(K, T) case UsdGenValueKind::K: f(Tag<T>()); return;
    USDGEN_VALUE_HEAP_KINDS(USDGEN_VISIT_HEAP)
#undef USDGEN_VISIT_HEAP
    default: return;
    }
}

/// Calls f(Tag<UsdGenCowArray<E>>) for an inline array's element kind.
template <class F>
void VisitArrayType(UsdGenValueKind element, F &&f)
{
    switch (element) {
#define USDGEN_VISIT_ARRAY(K, T) case UsdGenValueKind::K: f(Tag<UsdGenCowArray<T>>()); return;
    USDGEN_VALUE_ARRAY_ELEMENTS(USDGEN_VISIT_ARRAY)
#undef USDGEN_VISIT_ARRAY
    default: return;
    }
}

char const *KindName(UsdGenValueKind k) noexcept
{
    switch (k) {
    case UsdGenValueKind::Empty: return "";
    case UsdGenValueKind::Bool: return "bool";
    case UsdGenValueKind::UChar: return "uchar";
    case UsdGenValueKind::Int: return "int";
    case UsdGenValueKind::UInt: return "uint";
    case UsdGenValueKind::Int64: return "int64";
    case UsdGenValueKind::UInt64: return "uint64";
    case UsdGenValueKind::Half: return "half";
    case UsdGenValueKind::Float: return "float";
    case UsdGenValueKind::Double: return "double";
    case UsdGenValueKind::TimeCode: return "timecode";
    case UsdGenValueKind::Token: return "token";
    case UsdGenValueKind::String: return "string";
    case UsdGenValueKind::AssetPath: return "asset";
    case UsdGenValueKind::PathExpression: return "pathExpression";
    case UsdGenValueKind::Opaque: return "opaque";
    case UsdGenValueKind::Group: return "group";
    case UsdGenValueKind::Vec2h: return "half2";
    case UsdGenValueKind::Vec2f: return "float2";
    case UsdGenValueKind::Vec2d: return "double2";
    case UsdGenValueKind::Vec2i: return "int2";
    case UsdGenValueKind::Vec3h: return "half3";
    case UsdGenValueKind::Vec3f: return "float3";
    case UsdGenValueKind::Vec3d: return "double3";
    case UsdGenValueKind::Vec3i: return "int3";
    case UsdGenValueKind::Vec4h: return "half4";
    case UsdGenValueKind::Vec4f: return "float4";
    case UsdGenValueKind::Vec4d: return "double4";
    case UsdGenValueKind::Vec4i: return "int4";
    case UsdGenValueKind::Matrix2d: return "matrix2d";
    case UsdGenValueKind::Matrix3d: return "matrix3d";
    case UsdGenValueKind::Matrix4d: return "matrix4d";
    case UsdGenValueKind::Quath: return "quath";
    case UsdGenValueKind::Quatf: return "quatf";
    case UsdGenValueKind::Quatd: return "quatd";
    case UsdGenValueKind::Array: return "[]";
    case UsdGenValueKind::Boxed: return "boxed";
    }
    return "";
}

char const *ArrayName(UsdGenValueKind e) noexcept
{
    switch (e) {
    case UsdGenValueKind::Bool: return "bool[]";
    case UsdGenValueKind::UChar: return "uchar[]";
    case UsdGenValueKind::Int: return "int[]";
    case UsdGenValueKind::UInt: return "uint[]";
    case UsdGenValueKind::Int64: return "int64[]";
    case UsdGenValueKind::UInt64: return "uint64[]";
    case UsdGenValueKind::Half: return "half[]";
    case UsdGenValueKind::Float: return "float[]";
    case UsdGenValueKind::Double: return "double[]";
    case UsdGenValueKind::TimeCode: return "timecode[]";
    case UsdGenValueKind::Token: return "token[]";
    case UsdGenValueKind::String: return "string[]";
    case UsdGenValueKind::AssetPath: return "asset[]";
    case UsdGenValueKind::PathExpression: return "pathExpression[]";
    case UsdGenValueKind::Vec2h: return "half2[]";
    case UsdGenValueKind::Vec2f: return "float2[]";
    case UsdGenValueKind::Vec2d: return "double2[]";
    case UsdGenValueKind::Vec2i: return "int2[]";
    case UsdGenValueKind::Vec3h: return "half3[]";
    case UsdGenValueKind::Vec3f: return "float3[]";
    case UsdGenValueKind::Vec3d: return "double3[]";
    case UsdGenValueKind::Vec3i: return "int3[]";
    case UsdGenValueKind::Vec4h: return "half4[]";
    case UsdGenValueKind::Vec4f: return "float4[]";
    case UsdGenValueKind::Vec4d: return "double4[]";
    case UsdGenValueKind::Vec4i: return "int4[]";
    case UsdGenValueKind::Matrix2d: return "matrix2d[]";
    case UsdGenValueKind::Matrix3d: return "matrix3d[]";
    case UsdGenValueKind::Matrix4d: return "matrix4d[]";
    case UsdGenValueKind::Quath: return "quath[]";
    case UsdGenValueKind::Quatf: return "quatf[]";
    case UsdGenValueKind::Quatd: return "quatd[]";
    default: return "[]";
    }
}

}  // namespace

void UsdGenValue::_SetHeap(UsdGenValueKind kind, UsdGenValueKind element, void const *src)
{
    UsdGenRefCounted const *block = nullptr;
    VisitHeapType(kind, element, [&](auto tag) {
        using T = typename decltype(tag)::Type;
        auto *h = new HeapOf<T>(*static_cast<T const *>(src));
        h->AddRef();
        block = h;
    });
    std::memcpy(_bytes, &block, sizeof(block));
}

void UsdGenValue::_CopySlow(UsdGenValue const &o) noexcept
{
    _Relocate(this, const_cast<UsdGenValue *>(&o));   // bytes first, then the reference
    if (o._store == Store::Heap || o._store == Store::Box) {
        if (UsdGenRefCounted const *p = _Ptr()) p->AddRef();
    } else if (o._store == Store::Array) {
        VisitArrayType(o._element, [&](auto tag) {
            using A = typename decltype(tag)::Type;
            ::new (static_cast<void *>(_bytes)) A(o.UncheckedGet<A>());
        });
    }
}

void UsdGenValue::_Release() noexcept
{
    if (_store == Store::Heap || _store == Store::Box) {
        if (UsdGenRefCounted const *p = _Ptr()) p->Release();
    } else if (_store == Store::Array) {
        VisitArrayType(_element, [&](auto tag) {
            using A = typename decltype(tag)::Type;
            std::launder(reinterpret_cast<A *>(_bytes))->~A();
        });
    }
    _kind = UsdGenValueKind::Empty;
    _element = UsdGenValueKind::Empty;
    _store = Store::Inline;
}

size_t UsdGenValue::GetArraySize() const noexcept
{
    size_t n = 0;
    if (!IsArrayValued()) return n;
    Visit(*this, [&](auto const *v) {
        using T = std::decay_t<decltype(*v)>;
        if constexpr (value_detail::KindOf<T>::kind == UsdGenValueKind::Array) n = v->size();
    });
    return n;
}

uint64_t UsdGenValue::GetHash() const noexcept
{
    uint64_t h = Mix(kSeed, (uint64_t(_kind) << 8) | uint64_t(_element));
    if (_kind == UsdGenValueKind::Boxed) {
        auto const *b = static_cast<UsdGenBoxed const *>(_Ptr());
        return Mix(Mix(h, BoxedTag()), b ? b->Hash() : 0);
    }
    Visit(*this, [&](auto const *v) { h = HashOf(h, *v); });
    return h;
}

char const *UsdGenValue::GetTypeName() const noexcept
{
    return _kind == UsdGenValueKind::Array ? ArrayName(_element) : KindName(_kind);
}

bool UsdGenValue::_Equal(UsdGenValue const &a, UsdGenValue const &b) noexcept
{
    if (a._kind != b._kind || a._element != b._element) return false;
    if (a._kind == UsdGenValueKind::Empty) return true;
    if (a._kind == UsdGenValueKind::Boxed) {
        if (a.BoxedTag() != b.BoxedTag()) return false;
        auto const *x = static_cast<UsdGenBoxed const *>(a._Ptr());
        auto const *y = static_cast<UsdGenBoxed const *>(b._Ptr());
        return x == y || (x && y && x->Equal(*y));
    }
    bool same = false;
    Visit(a, [&](auto const *x) {
        using T = std::decay_t<decltype(*x)>;
        same = *x == b.UncheckedGet<T>();
    });
    return same;
}

}  // namespace usdGen
