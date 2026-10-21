// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_IMAGING_RBF_ANIMATION_H
#define USDGEN_IMAGING_RBF_ANIMATION_H
#include "usdGen/rbfAnimation.h"
#include "usdGen/graphDesc.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/imaging/hd/dataSource.h"
#include <map>

namespace usdGenImaging {
// Immutable renderer snapshot. Static computation inputs are shared across
// poses; only the sparse coefficients, extents and transforms are replaced.
struct RbfAnimation {
    std::shared_ptr<const usdGen::UsdGenRbfAnimationBinding> binding;
    std::map<SdfPath, HdContainerDataSourceHandle> computations, overlays;
    std::map<SdfPath, GfRange3d> restBounds;
    SdfPath coefficientsPath;
};
std::shared_ptr<const RbfAnimation> BuildRbfAnimation(
    std::shared_ptr<const usdGen::UsdGenRbfAnimationBinding> const& binding,
    GfMatrix4d const& xform);
std::shared_ptr<const RbfAnimation> PoseRbfAnimation(
    RbfAnimation const& previous, usdGen::UsdGenGraphDesc const& desc);
}
#endif
