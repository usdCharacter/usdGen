// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenQuat{f,d,h}: GfQuat's storage {imaginary i, j, k; real} and the
// header-level operations of gf/quat{f,d,h}.h.
#pragma once
#include "usdGenMath/vec.h"

#include <type_traits>

namespace usdGen {

template <class S>
class UsdGenQuat {
public:
    using ScalarType = S;
    using ImaginaryType = UsdGenVec<S, 3>;

    UsdGenQuat() = default;                                  // uninitialised, as Gf
    explicit UsdGenQuat(S realVal) noexcept : _imaginary(S(0)), _real(realVal) {}
    UsdGenQuat(S real, S i, S j, S k) noexcept : _imaginary(i, j, k), _real(real) {}
    UsdGenQuat(S real, ImaginaryType const &imaginary) noexcept : _imaginary(imaginary), _real(real) {}
    template <class S2, std::enable_if_t<math_detail::kImplicit<S2, S>, int> = 0>
    UsdGenQuat(UsdGenQuat<S2> const &o) noexcept : _imaginary(o.GetImaginary()), _real(o.GetReal()) {}
    template <class S2, std::enable_if_t<math_detail::kExplicit<S2, S>, int> = 0>
    explicit UsdGenQuat(UsdGenQuat<S2> const &o) noexcept : _imaginary(ImaginaryType(o.GetImaginary())), _real(S(o.GetReal())) {}
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenQuat>, int> = 0>
    UsdGenQuat(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenQuat), "host layout");
        std::memcpy(static_cast<void *>(this), static_cast<void const *>(&h), sizeof(UsdGenQuat));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenQuat>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(this), sizeof(UsdGenQuat));
        return h;
    }

    static UsdGenQuat GetZero() noexcept { return UsdGenQuat(S(0)); }
    static UsdGenQuat GetIdentity() noexcept { return UsdGenQuat(S(1)); }
    S GetReal() const noexcept { return _real; }
    void SetReal(S real) noexcept { _real = real; }
    ImaginaryType const &GetImaginary() const noexcept { return _imaginary; }
    void SetImaginary(ImaginaryType const &imaginary) noexcept { _imaginary = imaginary; }
    void SetImaginary(S i, S j, S k) noexcept { _imaginary = ImaginaryType(i, j, k); }

    S GetLength() const noexcept { return math_detail::Sqrt(_GetLengthSquared()); }
    UsdGenQuat GetNormalized(S eps = S(kUsdGenMinVectorLength)) const noexcept
    {
        UsdGenQuat ret(*this);
        ret.Normalize(eps);
        return ret;
    }
    S Normalize(S eps = S(kUsdGenMinVectorLength)) noexcept
    {
        S length = GetLength();
        if (length < eps) *this = GetIdentity();
        else *this /= length;
        return length;
    }
    UsdGenQuat GetConjugate() const noexcept { return UsdGenQuat(GetReal(), -GetImaginary()); }
    UsdGenQuat GetInverse() const noexcept { return GetConjugate() / _GetLengthSquared(); }

    friend size_t hash_value(UsdGenQuat const &q) noexcept
    {
        S const s[4] = {q._imaginary[0], q._imaginary[1], q._imaginary[2], q._real};
        return math_detail::HashScalars(s, 4);
    }
    UsdGenQuat operator-() const noexcept { return UsdGenQuat(-GetReal(), -GetImaginary()); }
    bool operator==(UsdGenQuat const &q) const noexcept { return GetReal() == q.GetReal() && GetImaginary() == q.GetImaginary(); }
    bool operator!=(UsdGenQuat const &q) const noexcept { return !(*this == q); }
    UsdGenQuat &operator*=(S s) noexcept { _real *= s; _imaginary *= s; return *this; }
    UsdGenQuat &operator/=(S s) noexcept { _real /= s; _imaginary /= s; return *this; }
    UsdGenQuat &operator+=(UsdGenQuat const &q) noexcept { _real += q._real; _imaginary += q._imaginary; return *this; }
    UsdGenQuat &operator-=(UsdGenQuat const &q) noexcept { _real -= q._real; _imaginary -= q._imaginary; return *this; }
    friend UsdGenQuat operator+(UsdGenQuat const &q1, UsdGenQuat const &q2) noexcept { return UsdGenQuat(q1) += q2; }
    friend UsdGenQuat operator-(UsdGenQuat const &q1, UsdGenQuat const &q2) noexcept { return UsdGenQuat(q1) -= q2; }
    friend UsdGenQuat operator*(UsdGenQuat const &q, S s) noexcept { return UsdGenQuat(q) *= s; }
    friend UsdGenQuat operator*(S s, UsdGenQuat const &q) noexcept { return UsdGenQuat(q) *= s; }
    friend UsdGenQuat operator/(UsdGenQuat const &q, S s) noexcept { return UsdGenQuat(q) /= s; }

private:
    S _GetLengthSquared() const noexcept { return UsdGenDot(_imaginary, _imaginary) + _real * _real; }
    ImaginaryType _imaginary;
    S _real;
};

using UsdGenQuatf = UsdGenQuat<float>;
using UsdGenQuatd = UsdGenQuat<double>;
using UsdGenQuath = UsdGenQuat<UsdGenHalf>;

template <class S>
inline S UsdGenDot(UsdGenQuat<S> const &q1, UsdGenQuat<S> const &q2) noexcept
{
    return UsdGenDot(q1.GetImaginary(), q2.GetImaginary()) + q1.GetReal() * q2.GetReal();
}

static_assert(sizeof(UsdGenQuatd) == 32 && sizeof(UsdGenQuatf) == 16 && sizeof(UsdGenQuath) == 8, "Gf layout");

}  // namespace usdGen
