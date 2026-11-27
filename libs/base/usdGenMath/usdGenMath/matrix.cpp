// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenMatrix4d's compiled members and UsdGenRotateInto, in Gf 26.08's
// operation order (gf/matrix4d.cpp, gf/rotation.cpp, gf/quatd.cpp).
#include "usdGenMath/matrix.h"

#include <cfloat>
#include <cmath>

namespace usdGen {

UsdGenMatrix4d UsdGenMatrix4d::GetTranspose() const noexcept
{
    UsdGenMatrix4d t;
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) t._mtx[c][r] = _mtx[r][c];
    return t;
}

UsdGenMatrix4d UsdGenMatrix4d::GetInverse(double *detPtr, double eps) const noexcept
{
    double x00, x01, x02, x03;
    double x10, x11, x12, x13;
    double x20, x21, x22, x23;
    double x30, x31, x32, x33;
    double y01, y02, y03, y12, y13, y23;
    double z00, z10, z20, z30;
    double z01, z11, z21, z31;
    double z02, z03, z12, z13, z22, z23, z32, z33;

    // First two columns and their six 2x2 determinants.
    x00 = _mtx[0][0]; x01 = _mtx[0][1];
    x10 = _mtx[1][0]; x11 = _mtx[1][1];
    x20 = _mtx[2][0]; x21 = _mtx[2][1];
    x30 = _mtx[3][0]; x31 = _mtx[3][1];
    y01 = x00 * x11 - x10 * x01;
    y02 = x00 * x21 - x20 * x01;
    y03 = x00 * x31 - x30 * x01;
    y12 = x10 * x21 - x20 * x11;
    y13 = x10 * x31 - x30 * x11;
    y23 = x20 * x31 - x30 * x21;

    // Last two columns; 3x3 cofactors for them.
    x02 = _mtx[0][2]; x03 = _mtx[0][3];
    x12 = _mtx[1][2]; x13 = _mtx[1][3];
    x22 = _mtx[2][2]; x23 = _mtx[2][3];
    x32 = _mtx[3][2]; x33 = _mtx[3][3];
    z33 = x02 * y12 - x12 * y02 + x22 * y01;
    z23 = x12 * y03 - x32 * y01 - x02 * y13;
    z13 = x02 * y23 - x22 * y03 + x32 * y02;
    z03 = x22 * y13 - x32 * y12 - x12 * y23;
    z32 = x13 * y02 - x23 * y01 - x03 * y12;
    z22 = x03 * y13 - x13 * y03 + x33 * y01;
    z12 = x23 * y03 - x33 * y02 - x03 * y23;
    z02 = x13 * y23 - x23 * y13 + x33 * y12;

    // 2x2 determinants of the last two columns; cofactors for the first two.
    y01 = x02 * x13 - x12 * x03;
    y02 = x02 * x23 - x22 * x03;
    y03 = x02 * x33 - x32 * x03;
    y12 = x12 * x23 - x22 * x13;
    y13 = x12 * x33 - x32 * x13;
    y23 = x22 * x33 - x32 * x23;
    z30 = x11 * y02 - x21 * y01 - x01 * y12;
    z20 = x01 * y13 - x11 * y03 + x31 * y01;
    z10 = x21 * y03 - x31 * y02 - x01 * y23;
    z00 = x11 * y23 - x21 * y13 + x31 * y12;
    z31 = x00 * y12 - x10 * y02 + x20 * y01;
    z21 = x10 * y03 - x30 * y01 - x00 * y13;
    z11 = x00 * y23 - x20 * y03 + x30 * y02;
    z01 = x20 * y13 - x30 * y12 - x10 * y23;

    double const det = x30 * z30 + x20 * z20 + x10 * z10 + x00 * z00;
    if (detPtr) *detPtr = det;

    UsdGenMatrix4d inverse;
    if (std::fabs(det) > eps) {
        double const rcp = 1.0 / det;
        inverse._mtx[0][0] = z00 * rcp;
        inverse._mtx[0][1] = z10 * rcp;
        inverse._mtx[1][0] = z01 * rcp;
        inverse._mtx[0][2] = z20 * rcp;
        inverse._mtx[2][0] = z02 * rcp;
        inverse._mtx[0][3] = z30 * rcp;
        inverse._mtx[3][0] = z03 * rcp;
        inverse._mtx[1][1] = z11 * rcp;
        inverse._mtx[1][2] = z21 * rcp;
        inverse._mtx[2][1] = z12 * rcp;
        inverse._mtx[1][3] = z31 * rcp;
        inverse._mtx[3][1] = z13 * rcp;
        inverse._mtx[2][2] = z22 * rcp;
        inverse._mtx[2][3] = z32 * rcp;
        inverse._mtx[3][2] = z23 * rcp;
        inverse._mtx[3][3] = z33 * rcp;
    } else {
        inverse.SetScale(FLT_MAX);
    }
    return inverse;
}

double UsdGenMatrix4d::_GetDeterminant3(size_t row1, size_t row2, size_t row3,
                                        size_t col1, size_t col2, size_t col3) const noexcept
{
    return (_mtx[row1][col1] * _mtx[row2][col2] * _mtx[row3][col3] +
            _mtx[row1][col2] * _mtx[row2][col3] * _mtx[row3][col1] +
            _mtx[row1][col3] * _mtx[row2][col1] * _mtx[row3][col2] -
            _mtx[row1][col1] * _mtx[row2][col3] * _mtx[row3][col2] -
            _mtx[row1][col2] * _mtx[row2][col1] * _mtx[row3][col3] -
            _mtx[row1][col3] * _mtx[row2][col2] * _mtx[row3][col1]);
}

double UsdGenMatrix4d::GetDeterminant() const noexcept
{
    return (-_mtx[0][3] * _GetDeterminant3(1, 2, 3, 0, 1, 2)
            + _mtx[1][3] * _GetDeterminant3(0, 2, 3, 0, 1, 2)
            - _mtx[2][3] * _GetDeterminant3(0, 1, 3, 0, 1, 2)
            + _mtx[3][3] * _GetDeterminant3(0, 1, 2, 0, 1, 2));
}

UsdGenMatrix4d &UsdGenMatrix4d::operator*=(UsdGenMatrix4d const &m) noexcept
{
    UsdGenMatrix4d const tmp = *this;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            _mtx[r][c] = tmp._mtx[r][0] * m._mtx[0][c] +
                         tmp._mtx[r][1] * m._mtx[1][c] +
                         tmp._mtx[r][2] * m._mtx[2][c] +
                         tmp._mtx[r][3] * m._mtx[3][c];
        }
    }
    return *this;
}

UsdGenMatrix4d &UsdGenMatrix4d::operator*=(double d) noexcept
{
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] *= d;
    return *this;
}

UsdGenMatrix4d &UsdGenMatrix4d::operator+=(UsdGenMatrix4d const &m) noexcept
{
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] += m._mtx[r][c];
    return *this;
}

UsdGenMatrix4d &UsdGenMatrix4d::operator-=(UsdGenMatrix4d const &m) noexcept
{
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) _mtx[r][c] -= m._mtx[r][c];
    return *this;
}

UsdGenVec3d UsdGenRotateInto(UsdGenVec3d const &rotateFrom, UsdGenVec3d const &rotateTo,
                             UsdGenVec3d const &vec) noexcept
{
    constexpr double kPi = 3.14159265358979323846;
    // GfRotation::SetRotateInto, then SetAxisAngle.
    UsdGenVec3d axis(1.0, 0.0, 0.0);
    double angle = 0.0;
    auto setAxisAngle = [&](UsdGenVec3d const &a, double degrees) {
        axis = a;
        angle = degrees;
        if (!(std::fabs(axis * axis - 1.0) < 1e-10)) axis.Normalize();
    };
    UsdGenVec3d const from = rotateFrom.GetNormalized();
    UsdGenVec3d const to = rotateTo.GetNormalized();
    double const cos = UsdGenDot(from, to);
    if (cos > 0.9999999) {
        // identity: axis (1, 0, 0), angle 0
    } else if (cos < -0.9999999) {
        UsdGenVec3d tmp = UsdGenCross(from, UsdGenVec3d(1.0, 0.0, 0.0));
        if (tmp.GetLength() < 0.00001) tmp = UsdGenCross(from, UsdGenVec3d(0.0, 1.0, 0.0));
        setAxisAngle(tmp.GetNormalized(), 180.0);
    } else {
        UsdGenVec3d const a = UsdGenCross(rotateFrom, rotateTo).GetNormalized();
        setAxisAngle(a, std::acos(cos) * (180.0 / kPi));
    }

    // GfRotation::GetQuat: half angle, then GfQuatd::GetNormalized.
    double const radians = angle * (kPi / 180.0) / 2.0;
    double const sinR = std::sin(radians);
    double const cosR = std::cos(radians);
    double real = cosR;
    UsdGenVec3d imag = axis * sinR;
    double const length = std::sqrt(UsdGenDot(imag, imag) + real * real);
    if (length < kUsdGenMinVectorLength) {
        real = 1.0;
        imag = UsdGenVec3d(0.0);
    } else {
        real /= length;
        imag /= length;
    }

    // GfMatrix4d::_SetRotateFromQuat (upper 3x3), then TransformDir.
    double const r = real;
    UsdGenVec3d const &i = imag;
    double m[3][3];
    m[0][0] = 1.0 - 2.0 * (i[1] * i[1] + i[2] * i[2]);
    m[0][1] = 2.0 * (i[0] * i[1] + i[2] * r);
    m[0][2] = 2.0 * (i[2] * i[0] - i[1] * r);
    m[1][0] = 2.0 * (i[0] * i[1] - i[2] * r);
    m[1][1] = 1.0 - 2.0 * (i[2] * i[2] + i[0] * i[0]);
    m[1][2] = 2.0 * (i[1] * i[2] + i[0] * r);
    m[2][0] = 2.0 * (i[2] * i[0] + i[1] * r);
    m[2][1] = 2.0 * (i[1] * i[2] - i[0] * r);
    m[2][2] = 1.0 - 2.0 * (i[1] * i[1] + i[0] * i[0]);
    return UsdGenVec3d(vec[0] * m[0][0] + vec[1] * m[1][0] + vec[2] * m[2][0],
                       vec[0] * m[0][1] + vec[1] * m[1][1] + vec[2] * m[2][1],
                       vec[0] * m[0][2] + vec[1] * m[1][2] + vec[2] * m[2][2]);
}

}  // namespace usdGen
