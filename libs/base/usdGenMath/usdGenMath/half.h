// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenHalf: IEEE binary16 with GfHalf's bits, rounding (nearest even) and
// NaN/inf mapping. Arithmetic goes through float, as pxr_half's does.
#pragma once
#include <cstdint>
#include <cstring>

namespace usdGen {

class UsdGenHalf {
public:
    UsdGenHalf() = default;                                  // uninitialised
    UsdGenHalf(float f) noexcept : _h(FromFloat(f)) {}
    operator float() const noexcept { return ToFloat(_h); }

    UsdGenHalf operator-() const noexcept { UsdGenHalf h; h._h = uint16_t(_h ^ 0x8000u); return h; }
    UsdGenHalf &operator=(float f) noexcept { _h = FromFloat(f); return *this; }
    UsdGenHalf &operator+=(UsdGenHalf h) noexcept { return *this = UsdGenHalf(float(*this) + float(h)); }
    UsdGenHalf &operator+=(float f) noexcept { return *this = UsdGenHalf(float(*this) + f); }
    UsdGenHalf &operator-=(UsdGenHalf h) noexcept { return *this = UsdGenHalf(float(*this) - float(h)); }
    UsdGenHalf &operator-=(float f) noexcept { return *this = UsdGenHalf(float(*this) - f); }
    UsdGenHalf &operator*=(UsdGenHalf h) noexcept { return *this = UsdGenHalf(float(*this) * float(h)); }
    UsdGenHalf &operator*=(float f) noexcept { return *this = UsdGenHalf(float(*this) * f); }
    UsdGenHalf &operator/=(UsdGenHalf h) noexcept { return *this = UsdGenHalf(float(*this) / float(h)); }
    UsdGenHalf &operator/=(float f) noexcept { return *this = UsdGenHalf(float(*this) / f); }

    uint16_t bits() const noexcept { return _h; }
    void setBits(uint16_t b) noexcept { _h = b; }

    /// float -> half: round to nearest, ties to even; NaN keeps its top
    /// mantissa bits (at least one set); overflow becomes inf.
    static uint16_t FromFloat(float f) noexcept
    {
        uint32_t x;
        std::memcpy(&x, &f, 4);
        if (f == 0) return uint16_t(x >> 16);                // keeps the zero's sign
        int const se = int((x >> 23) & 0x1ffu);
        int const e = (se & 0xff) - (127 - 15);
        if (e > 0 && e < 30) {                               // normalized, no overflow possible
            int const m = int(x & 0x007fffffu);
            int const base = (e << 10) | ((se & 0x100) << 7);
            return uint16_t(base + ((m + 0x00000fff + ((m >> 13) & 1)) >> 13));
        }
        return Convert(int(x));
    }
    /// half -> float, exact.
    static float ToFloat(uint16_t h) noexcept
    {
        uint32_t const s = (uint32_t(h) >> 15) & 0x1u;
        int e = int((h >> 10) & 0x1fu);
        uint32_t m = uint32_t(h) & 0x3ffu;
        uint32_t bits;
        if (e == 0) {
            if (m == 0) {
                bits = s << 31;
            } else {                                         // denormal: renormalize
                while (!(m & 0x400u)) { m <<= 1; e -= 1; }
                e += 1;
                m &= ~0x400u;
                bits = (s << 31) | (uint32_t(e + (127 - 15)) << 23) | (m << 13);
            }
        } else if (e == 31) {
            bits = (s << 31) | 0x7f800000u | (m << 13);      // inf, NaN keeps its payload
        } else {
            bits = (s << 31) | (uint32_t(e + (127 - 15)) << 23) | (m << 13);
        }
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    }

private:
    static uint16_t Convert(int i) noexcept
    {
        int const s = (i >> 16) & 0x00008000;
        int e = ((i >> 23) & 0x000000ff) - (127 - 15);
        int m = i & 0x007fffff;
        if (e <= 0) {
            if (e < -10) return uint16_t(s);                 // below the smallest denormal
            m = m | 0x00800000;                              // denormal half
            int const t = 14 - e;
            int const a = (1 << (t - 1)) - 1;
            int const b = (m >> t) & 1;
            m = (m + a + b) >> t;
            return uint16_t(s | m);
        }
        if (e == 0xff - (127 - 15)) {
            if (m == 0) return uint16_t(s | 0x7c00);         // inf
            m >>= 13;                                        // NaN
            return uint16_t(s | 0x7c00 | m | (m == 0));
        }
        m = m + 0x00000fff + ((m >> 13) & 1);
        if (m & 0x00800000) {
            m = 0;
            e += 1;
        }
        if (e > 30) return uint16_t(s | 0x7c00);             // overflow
        return uint16_t(s | (e << 10) | (m >> 13));
    }

    uint16_t _h;
};

/// Hash of the bits (GfHalf's hash_value).
inline size_t hash_value(UsdGenHalf h) noexcept { return h.bits(); }

}  // namespace usdGen
