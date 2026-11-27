// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// A displacement field over sample points of one surface and the rest points
// it deforms, per publication group. Immutable; a host plays it back without
// the primitive (was UsdGenRbfAnimationBinding).
#pragma once
#include "usdGen/api/nodeKey.h"
#include "usdGen/exec/epoch.h"
#include "usdGen/exec/pool.h"
#include "usdGenMath/rbf.h"
#include "usdGen/api/cowArray.h"          // UsdGenVec3fArray until step 11
#include <cstddef>
#include <cstdint>
#include <vector>

namespace usdGen {

struct UsdGenDeformBindingGroup {
    uint16_t group = 0;                  // publication group; the host names its prim
    UsdGenVec3fArray restPoints;
    double widthPadding = 0.0;
};

struct UsdGenDeformBinding : UsdGenRefCounted, UsdGenPooled {
    UsdGenNodeKey surface;
    std::vector<size_t> sampleIndices;
    usdGenMath::rbf::CubicField field;
    std::vector<UsdGenDeformBindingGroup> groups;
};

}  // namespace usdGen
