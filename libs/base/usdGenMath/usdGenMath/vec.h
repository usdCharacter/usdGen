// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenVec<S, N>: Gf 26.08's vectors (GfVec{2,3,4}{f,d,h,i}) bit for bit:
// scalar * and / take double and v / s is v * (1.0 / s) (int: componentwise
// /), Normalize divides by max(len, eps), GetLength is std::sqrt in S, the
// default constructor leaves the value uninitialised, and the conversion
// table is Gf's (float, half and int widen implicitly; narrowing is
// explicit; nothing converts to int). testUsdGenMathGfParity proves it.
#pragma once
#include "usdGenMath/half.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244)   // Gf's implicit narrowing in *= and conversions
#endif

namespace usdGen {

/// A host type with the same bytes (Gf). usdGenUsd/convert/math.h
/// specialises it; the primary admits nothing, so core code never names a
/// host type.
template <class Host> struct UsdGenHostMath { using Type = void; };

/// GF_MIN_VECTOR_LENGTH: the default Normalize epsilon.
constexpr double kUsdGenMinVectorLength = 1e-10;

template <class S, int N> class UsdGenVec;

namespace math_detail {
template <class S> constexpr bool kFloating = std::is_floating_point<S>::value || std::is_same<S, UsdGenHalf>::value;
// Gf's conversion table (vec3f.h, vec3d.h, vec3h.h).
template <class From, class To> constexpr bool kImplicit =
    (std::is_same<To, double>::value && !std::is_same<From, double>::value) ||
    (std::is_same<To, float>::value && (std::is_same<From, UsdGenHalf>::value || std::is_same<From, int>::value)) ||
    (std::is_same<To, UsdGenHalf>::value && std::is_same<From, int>::value);
template <class From, class To> constexpr bool kExplicit =
    !std::is_same<From, To>::value && !kImplicit<From, To> && !std::is_same<To, int>::value;
template <class S> using DivArg = std::conditional_t<std::is_same<S, int>::value, int, double>;
template <class Host, class T> constexpr bool kHost = std::is_same<typename UsdGenHostMath<Host>::Type, T>::value;

// Gf: float and double eps = GF_MIN_VECTOR_LENGTH, half eps = 0.001.
template <class S> inline S DefaultEps() noexcept { return S(kUsdGenMinVectorLength); }
template <> inline UsdGenHalf DefaultEps<UsdGenHalf>() noexcept { return UsdGenHalf(float(0.001)); }

// GfSqrt: float stays float, half goes through float.
inline float Sqrt(float f) noexcept { return std::sqrt(f); }
inline double Sqrt(double d) noexcept { return std::sqrt(d); }
inline float Sqrt(UsdGenHalf h) noexcept { return std::sqrt(float(h)); }

// Process-local value hash for VtValue/VtArray (TfHash finds hash_value):
// FNV-1a 64 over the scalars' bits, +0 == -0.
inline uint64_t HashMix(uint64_t h, uint64_t bits) noexcept
{
    for (int i = 0; i < 8; ++i) { h ^= (bits >> (8 * i)) & 0xffu; h *= 0x100000001b3ull; }
    return h;
}
inline uint64_t HashBits(float f) noexcept { if (f == 0.0f) f = 0.0f; uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline uint64_t HashBits(double d) noexcept { if (d == 0.0) d = 0.0; uint64_t b; std::memcpy(&b, &d, 8); return b; }
inline uint64_t HashBits(UsdGenHalf h) noexcept { return (h.bits() & 0x7fffu) ? h.bits() : 0u; }
inline uint64_t HashBits(int i) noexcept { return uint64_t(uint32_t(i)); }
template <class S> inline size_t HashScalars(S const *p, size_t n) noexcept
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) h = HashMix(h, HashBits(p[i]));
    return size_t(h);
}
}  // namespace math_detail

template <class S, int N>
class UsdGenVec {
    static_assert(N >= 2 && N <= 4, "2-4 components");
    template <class T> using IfFloat = std::enable_if_t<math_detail::kFloating<T>, int>;

public:
    using ScalarType = S;
    static constexpr size_t dimension = size_t(N);

    UsdGenVec() = default;                                   // uninitialised
    template <int M = N, std::enable_if_t<M == 2, int> = 0>
    constexpr explicit UsdGenVec(S value) noexcept : _data{value, value} {}
    template <int M = N, std::enable_if_t<M == 3, int> = 0>
    constexpr explicit UsdGenVec(S value) noexcept : _data{value, value, value} {}
    template <int M = N, std::enable_if_t<M == 4, int> = 0>
    constexpr explicit UsdGenVec(S value) noexcept : _data{value, value, value, value} {}
    template <int M = N, std::enable_if_t<M == 2, int> = 0>
    constexpr UsdGenVec(S s0, S s1) noexcept : _data{s0, s1} {}
    template <int M = N, std::enable_if_t<M == 3, int> = 0>
    constexpr UsdGenVec(S s0, S s1, S s2) noexcept : _data{s0, s1, s2} {}
    template <int M = N, std::enable_if_t<M == 4, int> = 0>
    constexpr UsdGenVec(S s0, S s1, S s2, S s3) noexcept : _data{s0, s1, s2, s3} {}
    // Gf's "Scl const *p": brace initialisation, so narrowing does not compile.
    template <class P, int M = N, std::enable_if_t<M == 2, int> = 0>
    constexpr explicit UsdGenVec(P const *p) noexcept : _data{p[0], p[1]} {}
    template <class P, int M = N, std::enable_if_t<M == 3, int> = 0>
    constexpr explicit UsdGenVec(P const *p) noexcept : _data{p[0], p[1], p[2]} {}
    template <class P, int M = N, std::enable_if_t<M == 4, int> = 0>
    constexpr explicit UsdGenVec(P const *p) noexcept : _data{p[0], p[1], p[2], p[3]} {}
    template <class S2, std::enable_if_t<math_detail::kImplicit<S2, S>, int> = 0>
    UsdGenVec(UsdGenVec<S2, N> const &o) noexcept
    {
        for (int i = 0; i < N; ++i) _data[i] = o[size_t(i)];
    }
    template <class S2, std::enable_if_t<math_detail::kExplicit<S2, S>, int> = 0>
    explicit UsdGenVec(UsdGenVec<S2, N> const &o) noexcept
    {
        for (int i = 0; i < N; ++i) _data[i] = o[size_t(i)];
    }
    /// Host vector with the same bytes: implicit in, explicit out.
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenVec>, int> = 0>
    UsdGenVec(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenVec), "host layout");
        std::memcpy(static_cast<void *>(_data), static_cast<void const *>(&h), sizeof(_data));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenVec>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(_data), sizeof(_data));
        return h;
    }

    static UsdGenVec Axis(size_t i) noexcept
    {
        UsdGenVec r(S(0));
        if (i < size_t(N)) r[i] = S(1);
        return r;
    }
    static UsdGenVec XAxis() noexcept { return Axis(0); }
    static UsdGenVec YAxis() noexcept { return Axis(1); }
    template <int M = N, std::enable_if_t<(M >= 3), int> = 0>
    static UsdGenVec ZAxis() noexcept { return Axis(2); }
    template <int M = N, std::enable_if_t<(M >= 4), int> = 0>
    static UsdGenVec WAxis() noexcept { return Axis(3); }

    template <int M = N, std::enable_if_t<M == 2, int> = 0>
    UsdGenVec &Set(S s0, S s1) noexcept { _data[0] = s0; _data[1] = s1; return *this; }
    template <int M = N, std::enable_if_t<M == 3, int> = 0>
    UsdGenVec &Set(S s0, S s1, S s2) noexcept { _data[0] = s0; _data[1] = s1; _data[2] = s2; return *this; }
    template <int M = N, std::enable_if_t<M == 4, int> = 0>
    UsdGenVec &Set(S s0, S s1, S s2, S s3) noexcept { _data[0] = s0; _data[1] = s1; _data[2] = s2; _data[3] = s3; return *this; }
    UsdGenVec &Set(S const *a) noexcept { for (int i = 0; i < N; ++i) _data[i] = a[i]; return *this; }

    S const *data() const noexcept { return _data; }
    S *data() noexcept { return _data; }
    S const *GetArray() const noexcept { return _data; }
    S const &operator[](size_t i) const noexcept { return _data[i]; }
    S &operator[](size_t i) noexcept { return _data[i]; }

    friend size_t hash_value(UsdGenVec const &v) noexcept { return math_detail::HashScalars(v._data, size_t(N)); }

    bool operator==(UsdGenVec const &o) const noexcept
    {
        if constexpr (N == 2) return _data[0] == o[0] && _data[1] == o[1];
        else if constexpr (N == 3) return _data[0] == o[0] && _data[1] == o[1] && _data[2] == o[2];
        else return _data[0] == o[0] && _data[1] == o[1] && _data[2] == o[2] && _data[3] == o[3];
    }
    bool operator!=(UsdGenVec const &o) const noexcept { return !(*this == o); }
    /// Gf's cross-type equality (vec3f.cpp): componentwise after promotion.
    template <class S2, std::enable_if_t<!std::is_same<S2, S>::value, int> = 0>
    bool operator==(UsdGenVec<S2, N> const &o) const noexcept
    {
        for (int i = 0; i < N; ++i) if (!(_data[i] == o[size_t(i)])) return false;
        return true;
    }
    template <class S2, std::enable_if_t<!std::is_same<S2, S>::value, int> = 0>
    bool operator!=(UsdGenVec<S2, N> const &o) const noexcept { return !(*this == o); }

    UsdGenVec operator-() const noexcept
    {
        UsdGenVec r;
        for (int i = 0; i < N; ++i) r._data[i] = -_data[i];
        return r;
    }
    UsdGenVec &operator+=(UsdGenVec const &o) noexcept
    {
        for (int i = 0; i < N; ++i) _data[i] += o._data[i];
        return *this;
    }
    friend UsdGenVec operator+(UsdGenVec const &l, UsdGenVec const &r) noexcept { return UsdGenVec(l) += r; }
    UsdGenVec &operator-=(UsdGenVec const &o) noexcept
    {
        for (int i = 0; i < N; ++i) _data[i] -= o._data[i];
        return *this;
    }
    friend UsdGenVec operator-(UsdGenVec const &l, UsdGenVec const &r) noexcept { return UsdGenVec(l) -= r; }
    UsdGenVec &operator*=(double s) noexcept
    {
        for (int i = 0; i < N; ++i) _data[i] *= s;
        return *this;
    }
    UsdGenVec operator*(double s) const noexcept { return UsdGenVec(*this) *= s; }
    friend UsdGenVec operator*(double s, UsdGenVec const &v) noexcept { return v * s; }
    UsdGenVec &operator/=(math_detail::DivArg<S> s) noexcept
    {
        if constexpr (std::is_same<S, int>::value) {
            for (int i = 0; i < N; ++i) _data[i] /= s;
            return *this;
        } else {
            return *this *= (1.0 / s);
        }
    }
    UsdGenVec operator/(math_detail::DivArg<S> s) const noexcept
    {
        if constexpr (std::is_same<S, int>::value) return UsdGenVec(*this) /= s;
        else return *this * (1.0 / s);
    }
    /// Dot product, left to right in S's arithmetic (half through float).
    S operator*(UsdGenVec const &v) const noexcept
    {
        if constexpr (N == 2) return _data[0] * v[0] + _data[1] * v[1];
        else if constexpr (N == 3) return _data[0] * v[0] + _data[1] * v[1] + _data[2] * v[2];
        else return _data[0] * v[0] + _data[1] * v[1] + _data[2] * v[2] + _data[3] * v[3];
    }
    S GetLengthSq() const noexcept { return *this * *this; }

    template <class T = S, IfFloat<T> = 0>
    UsdGenVec GetProjection(UsdGenVec const &v) const noexcept { return v * (*this * v); }
    template <class T = S, IfFloat<T> = 0>
    UsdGenVec GetComplement(UsdGenVec const &b) const noexcept { return *this - this->GetProjection(b); }
    template <class T = S, IfFloat<T> = 0>
    S GetLength() const noexcept { return math_detail::Sqrt(GetLengthSq()); }
    template <class T = S, IfFloat<T> = 0>
    S Normalize(S eps = math_detail::DefaultEps<S>()) noexcept
    {
        S length = GetLength();
        *this /= (length > eps) ? length : eps;
        return length;
    }
    template <class T = S, IfFloat<T> = 0>
    UsdGenVec GetNormalized(S eps = math_detail::DefaultEps<S>()) const noexcept
    {
        UsdGenVec normalized(*this);
        normalized.Normalize(eps);
        return normalized;
    }

private:
    S _data[N];
};

using UsdGenVec2f = UsdGenVec<float, 2>;
using UsdGenVec3f = UsdGenVec<float, 3>;
using UsdGenVec4f = UsdGenVec<float, 4>;
using UsdGenVec2d = UsdGenVec<double, 2>;
using UsdGenVec3d = UsdGenVec<double, 3>;
using UsdGenVec4d = UsdGenVec<double, 4>;
using UsdGenVec2i = UsdGenVec<int, 2>;
using UsdGenVec3i = UsdGenVec<int, 3>;
using UsdGenVec4i = UsdGenVec<int, 4>;
using UsdGenVec2h = UsdGenVec<UsdGenHalf, 2>;
using UsdGenVec3h = UsdGenVec<UsdGenHalf, 3>;
using UsdGenVec4h = UsdGenVec<UsdGenHalf, 4>;

// GfDot: one overload per type, so mixed arguments convert as Gf's do
// (float promotes to double, never the reverse).
inline float UsdGenDot(UsdGenVec2f const &a, UsdGenVec2f const &b) noexcept { return a * b; }
inline float UsdGenDot(UsdGenVec3f const &a, UsdGenVec3f const &b) noexcept { return a * b; }
inline float UsdGenDot(UsdGenVec4f const &a, UsdGenVec4f const &b) noexcept { return a * b; }
inline double UsdGenDot(UsdGenVec2d const &a, UsdGenVec2d const &b) noexcept { return a * b; }
inline double UsdGenDot(UsdGenVec3d const &a, UsdGenVec3d const &b) noexcept { return a * b; }
inline double UsdGenDot(UsdGenVec4d const &a, UsdGenVec4d const &b) noexcept { return a * b; }
inline int UsdGenDot(UsdGenVec2i const &a, UsdGenVec2i const &b) noexcept { return a * b; }
inline int UsdGenDot(UsdGenVec3i const &a, UsdGenVec3i const &b) noexcept { return a * b; }
inline int UsdGenDot(UsdGenVec4i const &a, UsdGenVec4i const &b) noexcept { return a * b; }
inline UsdGenHalf UsdGenDot(UsdGenVec2h const &a, UsdGenVec2h const &b) noexcept { return a * b; }
inline UsdGenHalf UsdGenDot(UsdGenVec3h const &a, UsdGenVec3h const &b) noexcept { return a * b; }
inline UsdGenHalf UsdGenDot(UsdGenVec4h const &a, UsdGenVec4h const &b) noexcept { return a * b; }

// GfCross (no Vec3i overload, as Gf).
inline UsdGenVec3f UsdGenCross(UsdGenVec3f const &v1, UsdGenVec3f const &v2) noexcept
{
    return UsdGenVec3f(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}
inline UsdGenVec3d UsdGenCross(UsdGenVec3d const &v1, UsdGenVec3d const &v2) noexcept
{
    return UsdGenVec3d(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}
inline UsdGenVec3h UsdGenCross(UsdGenVec3h const &v1, UsdGenVec3h const &v2) noexcept
{
    return UsdGenVec3h(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}

/// GfRotation(from, to).TransformDir(v) in Gf's operation order
/// (SetRotateInto, GetQuat, the rotation matrix). matrix.cpp.
UsdGenVec3d UsdGenRotateInto(UsdGenVec3d const &from, UsdGenVec3d const &to, UsdGenVec3d const &v) noexcept;

static_assert(std::is_trivially_copyable<UsdGenVec3f>::value && std::is_standard_layout<UsdGenVec3f>::value &&
              sizeof(UsdGenVec3f) == 12 && alignof(UsdGenVec3f) == 4, "Gf layout");
static_assert(sizeof(UsdGenVec3h) == 6 && sizeof(UsdGenVec4d) == 32, "Gf layout");

}  // namespace usdGen

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
