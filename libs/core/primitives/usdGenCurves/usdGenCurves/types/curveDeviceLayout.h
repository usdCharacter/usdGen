// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// How curves read UsdGenDeviceLayoutMetadata (kind, variant, wrap).
#pragma once
#include "usdGen/api/ext/device/deviceGeneration.h"
#include <cstdint>

namespace usdGen {

enum class UsdGenCurveLayoutType : uint8_t { Unknown = 0, Cubic = 1 };
enum class UsdGenCurveLayoutBasis : uint8_t { Unknown = 0, BSpline = 1, CatmullRom = 2 };
enum class UsdGenCurveLayoutWrap : uint8_t { Unknown = 0, Pinned = 1 };

constexpr UsdGenDeviceLayoutMetadata UsdGenCurveLayout(
    UsdGenCurveLayoutType type, UsdGenCurveLayoutBasis basis,
    UsdGenCurveLayoutWrap wrap) noexcept
{
    return UsdGenDeviceLayoutMetadata{static_cast<uint8_t>(type),
        static_cast<uint8_t>(basis), static_cast<uint8_t>(wrap)};
}
constexpr UsdGenCurveLayoutType UsdGenCurveLayoutTypeOf(
    UsdGenDeviceLayoutMetadata const& m) noexcept {
    return static_cast<UsdGenCurveLayoutType>(m.kind);
}
constexpr UsdGenCurveLayoutBasis UsdGenCurveLayoutBasisOf(
    UsdGenDeviceLayoutMetadata const& m) noexcept {
    return static_cast<UsdGenCurveLayoutBasis>(m.variant);
}
constexpr UsdGenCurveLayoutWrap UsdGenCurveLayoutWrapOf(
    UsdGenDeviceLayoutMetadata const& m) noexcept {
    return static_cast<UsdGenCurveLayoutWrap>(m.wrap);
}

/// All unspecified, or the one admitted cubic pinned layout (either basis).
constexpr bool UsdGenCurveLayoutValid(UsdGenDeviceLayoutMetadata const& m) noexcept
{
    using T = UsdGenCurveLayoutType;
    using B = UsdGenCurveLayoutBasis;
    using W = UsdGenCurveLayoutWrap;
    T const t = UsdGenCurveLayoutTypeOf(m);
    B const b = UsdGenCurveLayoutBasisOf(m);
    W const w = UsdGenCurveLayoutWrapOf(m);
    if (t == T::Unknown || b == B::Unknown || w == W::Unknown)
        return t == T::Unknown && b == B::Unknown && w == W::Unknown;
    return t == T::Cubic && (b == B::BSpline || b == B::CatmullRom) && w == W::Pinned;
}

}  // namespace usdGen
