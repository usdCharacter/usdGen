// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Channel descriptors: one typed column of an element set.
#pragma once
#include "usdGen/api/ids.h"
#include "usdGen/api/token.h"
#include <cstdint>

namespace usdGen {

enum class UsdGenScalar : uint8_t { F32, I32, U32, U64, F64 };
constexpr uint32_t UsdGenScalarBytes(UsdGenScalar s) noexcept
{
    return s == UsdGenScalar::U64 || s == UsdGenScalar::F64 ? 8u : 4u;
}

enum class UsdGenChannelRole : uint8_t { Other, Position, Rest, Width, Parameter, Id, Uv, Frame, Normal };

enum UsdGenChannelFlag : uint8_t {
    kUsdGenChannelPacked      = 1u << 0,   // arity values interleaved (rest, root frames, uv)
    kUsdGenChannelPassThrough = 1u << 1,   // immutable transport; only a capture that authors it writes it
};

struct UsdGenChannelDesc {                 // 16 bytes
    UsdGenToken       name;
    UsdGenScalar      scalar = UsdGenScalar::F32;
    uint8_t           arity = 1;
    uint8_t           level = 0;           // 0 constant (arity values), 1 element, 2.. sub-element
    UsdGenChannelRole role = UsdGenChannelRole::Other;
    uint8_t           component = 0;       // planar vector component (px 0, py 1, pz 2)
    uint8_t           flags = 0;
    uint16_t          reserved = 0;
};
static_assert(sizeof(UsdGenChannelDesc) == 16, "ChannelDesc is 16 bytes");

inline bool UsdGenSameLayout(UsdGenChannelDesc const &a, UsdGenChannelDesc const &b) noexcept
{
    return a.scalar == b.scalar && a.arity == b.arity && a.level == b.level;
}
inline bool operator==(UsdGenChannelDesc const &a, UsdGenChannelDesc const &b) noexcept
{
    return a.name == b.name && UsdGenSameLayout(a, b) && a.role == b.role &&
           a.component == b.component && a.flags == b.flags && a.reserved == b.reserved;
}
inline bool operator!=(UsdGenChannelDesc const &a, UsdGenChannelDesc const &b) noexcept
{
    return !(a == b);
}

}  // namespace usdGen
