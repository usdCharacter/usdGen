// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenRange3{f,d}: GfRange3{f,d}. The empty range is [FLT_MAX, -FLT_MAX]; the
// strict < and > updates keep a NaN from ever replacing a bound.
#pragma once
#include "usdGenMath/vec.h"

#include <cfloat>
#include <type_traits>

namespace usdGen {

template <class V>
class UsdGenRange3 {
public:
    using MinMaxType = V;
    using ScalarType = typename V::ScalarType;
    static constexpr size_t dimension = V::dimension;

    void SetEmpty() noexcept
    {
        // FLT_MAX for both, as GfRange3d.
        _min[0] = _min[1] = _min[2] = FLT_MAX;
        _max[0] = _max[1] = _max[2] = -FLT_MAX;
    }
    UsdGenRange3() noexcept { SetEmpty(); }
    UsdGenRange3(V const &min, V const &max) noexcept : _min(min), _max(max) {}
    template <class V2, std::enable_if_t<math_detail::kImplicit<typename V2::ScalarType, ScalarType>, int> = 0>
    UsdGenRange3(UsdGenRange3<V2> const &o) noexcept : _min(o.GetMin()), _max(o.GetMax()) {}
    template <class V2, std::enable_if_t<math_detail::kExplicit<typename V2::ScalarType, ScalarType>, int> = 0>
    explicit UsdGenRange3(UsdGenRange3<V2> const &o) noexcept : _min(V(o.GetMin())), _max(V(o.GetMax())) {}
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenRange3>, int> = 0>
    UsdGenRange3(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenRange3), "host layout");
        std::memcpy(static_cast<void *>(this), static_cast<void const *>(&h), sizeof(UsdGenRange3));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenRange3>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(this), sizeof(UsdGenRange3));
        return h;
    }

    V const &GetMin() const noexcept { return _min; }
    V const &GetMax() const noexcept { return _max; }
    V GetSize() const noexcept { return _max - _min; }
    V GetMidpoint() const noexcept { return static_cast<ScalarType>(0.5) * _min + static_cast<ScalarType>(0.5) * _max; }
    void SetMin(V const &min) noexcept { _min = min; }
    void SetMax(V const &max) noexcept { _max = max; }
    bool IsEmpty() const noexcept { return _min[0] > _max[0] || _min[1] > _max[1] || _min[2] > _max[2]; }
    void ExtendBy(V const &point) noexcept { UnionWith(point); }
    void ExtendBy(UsdGenRange3 const &range) noexcept { UnionWith(range); }
    bool Contains(V const &point) const noexcept
    {
        return (point[0] >= _min[0] && point[0] <= _max[0] && point[1] >= _min[1] && point[1] <= _max[1] &&
                point[2] >= _min[2] && point[2] <= _max[2]);
    }
    bool Contains(UsdGenRange3 const &range) const noexcept { return Contains(range._min) && Contains(range._max); }
    bool IsOutside(UsdGenRange3 const &range) const noexcept
    {
        return ((range._max[0] < _min[0] || range._min[0] > _max[0]) ||
                (range._max[1] < _min[1] || range._min[1] > _max[1]) ||
                (range._max[2] < _min[2] || range._min[2] > _max[2]));
    }
    static UsdGenRange3 GetUnion(UsdGenRange3 const &a, UsdGenRange3 const &b) noexcept
    {
        UsdGenRange3 res = a;
        _FindMin(res._min, b._min);
        _FindMax(res._max, b._max);
        return res;
    }
    UsdGenRange3 const &UnionWith(UsdGenRange3 const &b) noexcept
    {
        _FindMin(_min, b._min);
        _FindMax(_max, b._max);
        return *this;
    }
    UsdGenRange3 const &UnionWith(V const &b) noexcept
    {
        _FindMin(_min, b);
        _FindMax(_max, b);
        return *this;
    }
    static UsdGenRange3 GetIntersection(UsdGenRange3 const &a, UsdGenRange3 const &b) noexcept
    {
        UsdGenRange3 res = a;
        _FindMax(res._min, b._min);
        _FindMin(res._max, b._max);
        return res;
    }
    UsdGenRange3 const &IntersectWith(UsdGenRange3 const &b) noexcept
    {
        _FindMax(_min, b._min);
        _FindMin(_max, b._max);
        return *this;
    }
    UsdGenRange3 &operator+=(UsdGenRange3 const &b) noexcept { _min += b._min; _max += b._max; return *this; }
    UsdGenRange3 &operator-=(UsdGenRange3 const &b) noexcept { _min -= b._max; _max -= b._min; return *this; }
    UsdGenRange3 &operator*=(double m) noexcept
    {
        if (m > 0) {
            _min *= m;
            _max *= m;
        } else {
            V const tmp = _min;
            _min = _max * m;
            _max = tmp * m;
        }
        return *this;
    }
    UsdGenRange3 &operator/=(double m) noexcept { return *this *= (1.0 / m); }
    UsdGenRange3 operator+(UsdGenRange3 const &b) const noexcept { return UsdGenRange3(_min + b._min, _max + b._max); }
    UsdGenRange3 operator-(UsdGenRange3 const &b) const noexcept { return UsdGenRange3(_min - b._max, _max - b._min); }
    friend UsdGenRange3 operator*(double m, UsdGenRange3 const &r) noexcept
    {
        return (m > 0 ? UsdGenRange3(r._min * m, r._max * m) : UsdGenRange3(r._max * m, r._min * m));
    }
    friend UsdGenRange3 operator*(UsdGenRange3 const &r, double m) noexcept { return m * r; }
    friend UsdGenRange3 operator/(UsdGenRange3 const &r, double m) noexcept { return r * (1.0 / m); }
    friend size_t hash_value(UsdGenRange3 const &r) noexcept
    {
        ScalarType const s[6] = {r._min[0], r._min[1], r._min[2], r._max[0], r._max[1], r._max[2]};
        return math_detail::HashScalars(s, 6);
    }
    bool operator==(UsdGenRange3 const &b) const noexcept { return _min == b._min && _max == b._max; }
    bool operator!=(UsdGenRange3 const &b) const noexcept { return !(*this == b); }

private:
    static void _FindMin(V &dest, V const &point) noexcept
    {
        if (point[0] < dest[0]) dest[0] = point[0];
        if (point[1] < dest[1]) dest[1] = point[1];
        if (point[2] < dest[2]) dest[2] = point[2];
    }
    static void _FindMax(V &dest, V const &point) noexcept
    {
        if (point[0] > dest[0]) dest[0] = point[0];
        if (point[1] > dest[1]) dest[1] = point[1];
        if (point[2] > dest[2]) dest[2] = point[2];
    }
    V _min, _max;
};

using UsdGenRange3f = UsdGenRange3<UsdGenVec3f>;
using UsdGenRange3d = UsdGenRange3<UsdGenVec3d>;

static_assert(sizeof(UsdGenRange3f) == 24 && sizeof(UsdGenRange3d) == 48, "Gf layout");

}  // namespace usdGen
