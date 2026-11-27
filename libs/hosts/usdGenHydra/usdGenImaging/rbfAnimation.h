// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_IMAGING_RBF_ANIMATION_H
#define USDGEN_IMAGING_RBF_ANIMATION_H
#include "usdGen/api/deformBinding.h"
#include "usdGen/api/primRef.h"
#include "usdGenUsd/graphDesc.h"
#include "usdGenUsd/lower.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/imaging/hd/dataSource.h"
#include <map>

namespace usdGenImaging {
// Immutable renderer snapshot. Static computation inputs are shared across
// poses; only the sparse coefficients, extents and transforms are replaced.
struct RbfAnimation {
    usdGen::UsdGenRef<const usdGen::UsdGenDeformBinding> binding;
    std::map<SdfPath, HdContainerDataSourceHandle> computations, overlays;
    std::map<SdfPath, GfRange3d> restBounds;
    SdfPath coefficientsPath;
    std::vector<SdfPath> tilePaths;        // per binding group
    usdGen::UsdGenHostDataRef host;      // the cooked spec's path table
};
/// groupPrims: the published prim of each binding group, in group order.
std::shared_ptr<const RbfAnimation> BuildRbfAnimation(
    usdGen::UsdGenRef<const usdGen::UsdGenDeformBinding> const& binding,
    std::vector<usdGen::UsdGenPrimRef> const& groupPrims,
    usdGen::UsdGenHostDataRef const &host, GfMatrix4d const& xform);
std::shared_ptr<const RbfAnimation> PoseRbfAnimation(
    RbfAnimation const& previous, usdGen::UsdGenGraphDesc const& desc);
}
#endif
