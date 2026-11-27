// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenMatrix4d (Gf 26.08's GfMatrix4d bit for bit: row vectors, no Vec3f
// overloads, Transform divides by w as GfProject does; GetInverse,
// GetDeterminant and operator*= live in matrix.cpp in gf/matrix4d.cpp's
// operation order) and the storage-only UsdGenMatrix{2d,3d,4f} the value
// table needs.
#pragma once
#include "usdGenMath/vec.h"

#include <cstring>
#include <type_traits>

namespace usdGen {

class UsdGenMatrix4f;

class UsdGenMatrix4d {
public:
    using ScalarType = double;
    static constexpr size_t numRows = 4;
    static constexpr size_t numColumns = 4;

    UsdGenMatrix4d() = default;                              // uninitialised, as Gf
    explicit UsdGenMatrix4d(double s) noexcept { SetDiagonal(s); }
    explicit UsdGenMatrix4d(UsdGenVec4d const &v) noexcept { SetDiagonal(v); }
    UsdGenMatrix4d(double m00, double m01, double m02, double m03,
                   double m10, double m11, double m12, double m13,
                   double m20, double m21, double m22, double m23,
                   double m30, double m31, double m32, double m33) noexcept
    {
        Set(m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33);
    }
    UsdGenMatrix4d(double const m[4][4]) noexcept { Set(m); }
    explicit UsdGenMatrix4d(UsdGenMatrix4f const &m) noexcept;
    /// Host matrix with the same bytes: implicit in, explicit out.
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenMatrix4d>, int> = 0>
    UsdGenMatrix4d(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenMatrix4d), "host layout");
        std::memcpy(static_cast<void *>(_mtx), static_cast<void const *>(&h), sizeof(_mtx));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenMatrix4d>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(_mtx), sizeof(_mtx));
        return h;
    }

    UsdGenMatrix4d &Set(double m00, double m01, double m02, double m03,
                        double m10, double m11, double m12, double m13,
                        double m20, double m21, double m22, double m23,
                        double m30, double m31, double m32, double m33) noexcept
    {
        _mtx[0][0] = m00; _mtx[0][1] = m01; _mtx[0][2] = m02; _mtx[0][3] = m03;
        _mtx[1][0] = m10; _mtx[1][1] = m11; _mtx[1][2] = m12; _mtx[1][3] = m13;
        _mtx[2][0] = m20; _mtx[2][1] = m21; _mtx[2][2] = m22; _mtx[2][3] = m23;
        _mtx[3][0] = m30; _mtx[3][1] = m31; _mtx[3][2] = m32; _mtx[3][3] = m33;
        return *this;
    }
    UsdGenMatrix4d &Set(double const m[4][4]) noexcept
    {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] = m[r][c];
        return *this;
    }
    UsdGenMatrix4d &SetIdentity() noexcept { return SetDiagonal(1); }
    UsdGenMatrix4d &SetZero() noexcept { return SetDiagonal(0); }
    UsdGenMatrix4d &SetDiagonal(double s) noexcept
    {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] = r == c ? s : 0.0;
        return *this;
    }
    UsdGenMatrix4d &SetDiagonal(UsdGenVec4d const &v) noexcept
    {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] = r == c ? v[size_t(r)] : 0.0;
        return *this;
    }
    /// Scale s on the diagonal, [3][3] = 1 (GetInverse's singular result).
    UsdGenMatrix4d &SetScale(double s) noexcept
    {
        SetDiagonal(s);
        _mtx[3][3] = 1.0;
        return *this;
    }
    UsdGenMatrix4d &SetScale(UsdGenVec3d const &s) noexcept
    {
        SetDiagonal(UsdGenVec4d(s[0], s[1], s[2], 1.0));
        return *this;
    }
    UsdGenMatrix4d &SetTranslate(UsdGenVec3d const &t) noexcept
    {
        SetDiagonal(1.0);
        return SetTranslateOnly(t);
    }
    UsdGenMatrix4d &SetTranslateOnly(UsdGenVec3d const &t) noexcept
    {
        _mtx[3][0] = t[0]; _mtx[3][1] = t[1]; _mtx[3][2] = t[2]; _mtx[3][3] = 1.0;
        return *this;
    }
    void SetRow(int i, UsdGenVec4d const &v) noexcept
    {
        _mtx[i][0] = v[0]; _mtx[i][1] = v[1]; _mtx[i][2] = v[2]; _mtx[i][3] = v[3];
    }
    void SetColumn(int i, UsdGenVec4d const &v) noexcept
    {
        _mtx[0][i] = v[0]; _mtx[1][i] = v[1]; _mtx[2][i] = v[2]; _mtx[3][i] = v[3];
    }
    UsdGenVec4d GetRow(int i) const noexcept { return UsdGenVec4d(_mtx[i][0], _mtx[i][1], _mtx[i][2], _mtx[i][3]); }
    UsdGenVec4d GetColumn(int i) const noexcept { return UsdGenVec4d(_mtx[0][i], _mtx[1][i], _mtx[2][i], _mtx[3][i]); }
    void SetRow3(int i, UsdGenVec3d const &v) noexcept { _mtx[i][0] = v[0]; _mtx[i][1] = v[1]; _mtx[i][2] = v[2]; }
    UsdGenVec3d GetRow3(int i) const noexcept { return UsdGenVec3d(_mtx[i][0], _mtx[i][1], _mtx[i][2]); }
    UsdGenVec3d ExtractTranslation() const noexcept { return UsdGenVec3d(_mtx[3][0], _mtx[3][1], _mtx[3][2]); }

    double *data() noexcept { return &_mtx[0][0]; }
    double const *data() const noexcept { return &_mtx[0][0]; }
    double *GetArray() noexcept { return &_mtx[0][0]; }
    double const *GetArray() const noexcept { return &_mtx[0][0]; }
    double *operator[](int i) noexcept { return _mtx[i]; }
    double const *operator[](int i) const noexcept { return _mtx[i]; }

    friend size_t hash_value(UsdGenMatrix4d const &m) noexcept { return math_detail::HashScalars(&m._mtx[0][0], 16); }

    bool operator==(UsdGenMatrix4d const &m) const noexcept
    {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) if (!(_mtx[r][c] == m._mtx[r][c])) return false;
        return true;
    }
    bool operator!=(UsdGenMatrix4d const &m) const noexcept { return !(*this == m); }

    UsdGenMatrix4d GetTranspose() const noexcept;
    UsdGenMatrix4d GetInverse(double *det = nullptr, double eps = 0) const noexcept;
    double GetDeterminant() const noexcept;
    double GetDeterminant3() const noexcept { return _GetDeterminant3(0, 1, 2, 0, 1, 2); }

    UsdGenMatrix4d &operator*=(UsdGenMatrix4d const &m) noexcept;
    UsdGenMatrix4d &operator*=(double d) noexcept;
    UsdGenMatrix4d &operator+=(UsdGenMatrix4d const &m) noexcept;
    UsdGenMatrix4d &operator-=(UsdGenMatrix4d const &m) noexcept;
    friend UsdGenMatrix4d operator-(UsdGenMatrix4d const &m) noexcept
    {
        UsdGenMatrix4d r;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) r._mtx[i][j] = -m._mtx[i][j];
        return r;
    }
    friend UsdGenMatrix4d operator*(UsdGenMatrix4d const &m1, double d) noexcept { UsdGenMatrix4d t(m1); t *= d; return t; }
    friend UsdGenMatrix4d operator*(double d, UsdGenMatrix4d const &m) noexcept { return m * d; }
    friend UsdGenMatrix4d operator+(UsdGenMatrix4d const &m1, UsdGenMatrix4d const &m2) noexcept { UsdGenMatrix4d t(m1); t += m2; return t; }
    friend UsdGenMatrix4d operator-(UsdGenMatrix4d const &m1, UsdGenMatrix4d const &m2) noexcept { UsdGenMatrix4d t(m1); t -= m2; return t; }
    friend UsdGenMatrix4d operator*(UsdGenMatrix4d const &m1, UsdGenMatrix4d const &m2) noexcept { UsdGenMatrix4d t(m1); t *= m2; return t; }
    friend UsdGenMatrix4d operator/(UsdGenMatrix4d const &m1, UsdGenMatrix4d const &m2) noexcept { return m1 * m2.GetInverse(); }
    friend UsdGenVec4d operator*(UsdGenMatrix4d const &m, UsdGenVec4d const &vec) noexcept
    {
        return UsdGenVec4d(vec[0] * m._mtx[0][0] + vec[1] * m._mtx[0][1] + vec[2] * m._mtx[0][2] + vec[3] * m._mtx[0][3],
                           vec[0] * m._mtx[1][0] + vec[1] * m._mtx[1][1] + vec[2] * m._mtx[1][2] + vec[3] * m._mtx[1][3],
                           vec[0] * m._mtx[2][0] + vec[1] * m._mtx[2][1] + vec[2] * m._mtx[2][2] + vec[3] * m._mtx[2][3],
                           vec[0] * m._mtx[3][0] + vec[1] * m._mtx[3][1] + vec[2] * m._mtx[3][2] + vec[3] * m._mtx[3][3]);
    }
    friend UsdGenVec4d operator*(UsdGenVec4d const &vec, UsdGenMatrix4d const &m) noexcept
    {
        return UsdGenVec4d(vec[0] * m._mtx[0][0] + vec[1] * m._mtx[1][0] + vec[2] * m._mtx[2][0] + vec[3] * m._mtx[3][0],
                           vec[0] * m._mtx[0][1] + vec[1] * m._mtx[1][1] + vec[2] * m._mtx[2][1] + vec[3] * m._mtx[3][1],
                           vec[0] * m._mtx[0][2] + vec[1] * m._mtx[1][2] + vec[2] * m._mtx[2][2] + vec[3] * m._mtx[3][2],
                           vec[0] * m._mtx[0][3] + vec[1] * m._mtx[1][3] + vec[2] * m._mtx[2][3] + vec[3] * m._mtx[3][3]);
    }

    /// Row vector, w = 1, divided by the resulting w (GfProject).
    UsdGenVec3d Transform(UsdGenVec3d const &vec) const noexcept
    {
        UsdGenVec4d const p(vec[0] * _mtx[0][0] + vec[1] * _mtx[1][0] + vec[2] * _mtx[2][0] + _mtx[3][0],
                            vec[0] * _mtx[0][1] + vec[1] * _mtx[1][1] + vec[2] * _mtx[2][1] + _mtx[3][1],
                            vec[0] * _mtx[0][2] + vec[1] * _mtx[1][2] + vec[2] * _mtx[2][2] + _mtx[3][2],
                            vec[0] * _mtx[0][3] + vec[1] * _mtx[1][3] + vec[2] * _mtx[2][3] + _mtx[3][3]);
        double const inv = (p[3] != 0.0) ? 1.0 / p[3] : 1.0;
        return UsdGenVec3d(inv * p[0], inv * p[1], inv * p[2]);
    }
    /// w = 0: translation ignored.
    UsdGenVec3d TransformDir(UsdGenVec3d const &vec) const noexcept
    {
        return UsdGenVec3d(vec[0] * _mtx[0][0] + vec[1] * _mtx[1][0] + vec[2] * _mtx[2][0],
                           vec[0] * _mtx[0][1] + vec[1] * _mtx[1][1] + vec[2] * _mtx[2][1],
                           vec[0] * _mtx[0][2] + vec[1] * _mtx[1][2] + vec[2] * _mtx[2][2]);
    }
    /// w = 1, column 3 ignored.
    UsdGenVec3d TransformAffine(UsdGenVec3d const &vec) const noexcept
    {
        return UsdGenVec3d(vec[0] * _mtx[0][0] + vec[1] * _mtx[1][0] + vec[2] * _mtx[2][0] + _mtx[3][0],
                           vec[0] * _mtx[0][1] + vec[1] * _mtx[1][1] + vec[2] * _mtx[2][1] + _mtx[3][1],
                           vec[0] * _mtx[0][2] + vec[1] * _mtx[1][2] + vec[2] * _mtx[2][2] + _mtx[3][2]);
    }

private:
    double _GetDeterminant3(size_t r1, size_t r2, size_t r3, size_t c1, size_t c2, size_t c3) const noexcept;
    double _mtx[4][4];
};

/// Storage-shaped NxM double/float matrix: diagonal constructor, element
/// access, equality, hash. GfMatrix{2d,3d,4f}'s layout.
template <class S, int N>
class UsdGenSquareMatrix {
public:
    using ScalarType = S;
    UsdGenSquareMatrix() = default;                          // uninitialised, as Gf
    explicit UsdGenSquareMatrix(S s) noexcept { SetDiagonal(s); }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenSquareMatrix>, int> = 0>
    UsdGenSquareMatrix(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenSquareMatrix), "host layout");
        std::memcpy(static_cast<void *>(_mtx), static_cast<void const *>(&h), sizeof(_mtx));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenSquareMatrix>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(_mtx), sizeof(_mtx));
        return h;
    }
    UsdGenSquareMatrix &SetDiagonal(S s) noexcept
    {
        for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) _mtx[r][c] = r == c ? s : S(0);
        return *this;
    }
    UsdGenSquareMatrix &SetIdentity() noexcept { return SetDiagonal(S(1)); }
    S *data() noexcept { return &_mtx[0][0]; }
    S const *data() const noexcept { return &_mtx[0][0]; }
    S const *GetArray() const noexcept { return &_mtx[0][0]; }
    S *operator[](int i) noexcept { return _mtx[i]; }
    S const *operator[](int i) const noexcept { return _mtx[i]; }
    friend size_t hash_value(UsdGenSquareMatrix const &m) noexcept { return math_detail::HashScalars(&m._mtx[0][0], size_t(N * N)); }
    bool operator==(UsdGenSquareMatrix const &m) const noexcept
    {
        for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) if (!(_mtx[r][c] == m._mtx[r][c])) return false;
        return true;
    }
    bool operator!=(UsdGenSquareMatrix const &m) const noexcept { return !(*this == m); }

protected:
    S _mtx[N][N];
};

using UsdGenMatrix2d = UsdGenSquareMatrix<double, 2>;
using UsdGenMatrix3d = UsdGenSquareMatrix<double, 3>;

class UsdGenMatrix4f : public UsdGenSquareMatrix<float, 4> {
public:
    UsdGenMatrix4f() = default;
    explicit UsdGenMatrix4f(float s) noexcept : UsdGenSquareMatrix<float, 4>(s) {}
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenMatrix4f>, int> = 0>
    UsdGenMatrix4f(H const &h) noexcept
    {
        static_assert(sizeof(H) == sizeof(UsdGenMatrix4f), "host layout");
        std::memcpy(static_cast<void *>(_mtx), static_cast<void const *>(&h), sizeof(_mtx));
    }
    template <class H, std::enable_if_t<math_detail::kHost<H, UsdGenMatrix4f>, int> = 0>
    explicit operator H() const noexcept
    {
        H h;
        std::memcpy(static_cast<void *>(&h), static_cast<void const *>(_mtx), sizeof(_mtx));
        return h;
    }
    /// GfMatrix4f(GfMatrix4d): element-wise narrowing.
    explicit UsdGenMatrix4f(UsdGenMatrix4d const &m) noexcept
    {
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] = float(m[r][c]);
    }
    friend size_t hash_value(UsdGenMatrix4f const &m) noexcept { return math_detail::HashScalars(m.data(), 16); }
};

inline UsdGenMatrix4d::UsdGenMatrix4d(UsdGenMatrix4f const &m) noexcept
{
    Set(m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
}

static_assert(std::is_trivially_copyable<UsdGenMatrix4d>::value && sizeof(UsdGenMatrix4d) == 128, "Gf layout");
static_assert(sizeof(UsdGenMatrix2d) == 32 && sizeof(UsdGenMatrix3d) == 72 && sizeof(UsdGenMatrix4f) == 64, "Gf layout");

}  // namespace usdGen
