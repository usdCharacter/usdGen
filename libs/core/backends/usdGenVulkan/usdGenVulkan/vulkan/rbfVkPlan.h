// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// rbfVkPlan.h — header-only gap-config Deform admission for the Vulkan RBF
// device path (no Vulkan dependency; unit-testable, hook-callable).
//
// RbfVkValidateDeformGap mirrors ValidateDeform (executionPlan.cpp:380)
// for the configs that reject there: out-of-[4,100] rbfSamples budgets,
// non-identity source/surface transforms, and <4 selected samples. The
// integration hook delegates exactly those configs here; admitted configs
// keep the existing path byte-for-byte. On success the hook sets
// VulkanDeformControls::devicePath=true (field added by the hook) so the
// executor/job route the stage to RbfVkDeformPipeline.
//
// Sample spaces (all baked host-side; the device solver in rbfVkBinding
// is transform-agnostic, like cuSOLVER):
//   description rest   = restPoints verbatim (CPU-lane convention,
//                        ops/deform.cpp:257; rest-local == description);
//   description posed  = relative * points with
//                        relative = surface.worldMatrix * desc.xformMatrix^-1
//                        (ops/deform.cpp:256,260; Vulkan desc.xform is
//                        plan-wide identity, so this is surface.worldMatrix);
//   FPF selection runs on description rest (ops/deform.cpp:307-311);
//   stored rest/posed/rootTargets are source-LOCAL (toLocal = source
//   worldMatrix^-1), matching the immutable-plan convention that source
//   points stay source-local (executionPlan.cpp:83-86; root capture moves
//   rest-surface frames into that same space through the inverse).
// Budgets below the CUDA minimum (down to 1) are ACCEPTED here and fail
// at RbfVkBinding::Bind with the exact CUDA InvalidArgument result;
// budgets above kRbfVkPlanMaxBudget reject (shared signed matrix-index safety bound).
#pragma once

#include "usdGen/vulkan/executionPlan.h"
#include "usdGen/graphDesc.h"
#include "usdGen/op.h"
#include "usdGen/ops/rbfField.h"

#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/vec3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen::vulkan {

inline constexpr int kRbfVkPlanMinBudget = 1;
inline constexpr int kRbfVkPlanMaxBudget = 46336;

namespace rbfVkPlanDetail {

inline bool Fail(UsdGenDiagnostics* diagnostics, std::string const& message) {
    if (diagnostics) diagnostics->Error("Vulkan source-width plan: " + message);
    return false;
}

// Copy of the native-free check in executionPlan.cpp (aligned with C3
// root capture): row-vector affine, translation in matrix[3][0..2].
inline bool IsFiniteInvertibleAffine(GfMatrix4d const& matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column])) return false;
    if (matrix[0][3] != 0.0 || matrix[1][3] != 0.0 ||
        matrix[2][3] != 0.0 || matrix[3][3] != 1.0)
        return false;
    double determinant = 0.0;
    GfMatrix4d const inverse = matrix.GetInverse(&determinant);
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= std::numeric_limits<double>::epsilon())
        return false;
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(inverse[row][column])) return false;
    return inverse[0][3] == 0.0 && inverse[1][3] == 0.0 &&
        inverse[2][3] == 0.0 && inverse[3][3] == 1.0;
}

// Forward-only application needs no inverse (the surface matrix maps
// posed points description-ward; only the source matrix is inverted).
inline bool IsFiniteAffine(GfMatrix4d const& matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column])) return false;
    return matrix[0][3] == 0.0 && matrix[1][3] == 0.0 &&
        matrix[2][3] == 0.0 && matrix[3][3] == 1.0;
}

inline bool ReadFloat(UsdGenParamValue const& param, float* value) {
    if (!param.value.IsHolding<float>()) return false;
    *value = param.value.UncheckedGet<float>();
    return std::isfinite(*value);
}

// Barycentric (triangle) / bilinear (quad) interpolation of a surface face
// at a UV coordinate. Mirrors the CUDA GatherRoots kernel
// (gpu/surfaceBinding.cu RootPosition) exactly, in float arithmetic with the
// same evaluation order: triangle a*w+b*u+c*v with w=1-u-v, quad
// a*u0*v0+b*u*v0+c*u*v+d*u0*v. Spelled per-component like surfaceVk's
// GatherRootLocal so the two validators agree bitwise on identity fixtures.
// (The legacy RootPosition in executionPlan.cpp:400 shares the old triangle
// divergence; the identity path it serves is preserved bit-for-bit and out
// of scope.)
inline GfVec3f GatherRoot(UsdGenSurfaceDesc const& surface,
                          std::vector<uint32_t> const& faceStart,
                          int faceIdx, GfVec2f const& uv) {
    auto const& indices = surface.faceVertexIndices;
    auto const& pts = surface.points;
    uint32_t const base = faceStart[faceIdx];
    int const n = surface.faceVertexCounts[faceIdx];
    float const u = uv[0], v = uv[1];
    GfVec3f const a(pts[indices[base]]);
    GfVec3f const b(pts[indices[base + 1]]);
    GfVec3f const c(pts[indices[base + 2]]);
    if (n == 3) {
        float const w = 1.0f - u - v;
        return GfVec3f(a[0] * w + b[0] * u + c[0] * v,
                       a[1] * w + b[1] * u + c[1] * v,
                       a[2] * w + b[2] * u + c[2] * v);
    }
    GfVec3f const d(pts[indices[base + 3]]);
    float const u0 = 1.0f - u, v0 = 1.0f - v;
    return GfVec3f(a[0] * u0 * v0 + b[0] * u * v0 + c[0] * u * v + d[0] * u0 * v,
                   a[1] * u0 * v0 + b[1] * u * v0 + c[1] * u * v + d[1] * u0 * v,
                   a[2] * u0 * v0 + b[2] * u * v0 + c[2] * u * v + d[2] * u0 * v);
}

} // namespace rbfVkPlanDetail

inline bool RbfVkValidateDeformGap(UsdGenGraphDesc const& desc,
                                   UsdGenNodeDesc const& node,
                                   UsdGenCurveSetDesc const* source,
                                   VulkanDeformControls* out,
                                   UsdGenDiagnostics* diagnostics) {
    using rbfVkPlanDetail::Fail;
    if (!source || node.surfaces.size() != 1 ||
        !node.references.empty() || !node.curves.empty() ||
        !node.maps.empty() || !node.mapBindings.empty() ||
        !node.ramps.empty())
        return Fail(diagnostics, "Deform requires one pose surface and no references/maps");
    std::set<TfToken> seen;
    int budget = 100;
    for (auto const& param : node.params) {
        if (param.animated) return Fail(diagnostics, "Deform requires non-animated literal controls");
        if (!seen.insert(param.name).second) return Fail(diagnostics, "duplicate Deform parameter " + param.name.GetString());
        auto const& name = param.name.GetString();
        if (name == "rbfSamples") {
            if (!param.value.IsHolding<int>()) return Fail(diagnostics, "Deform rbfSamples must be an int literal");
            budget = param.value.UncheckedGet<int>();
        } else if (name == "lockRoots") {
            if (!param.value.IsHolding<bool>()) return Fail(diagnostics, "Deform lockRoots must be a bool literal");
            out->lockRoots = param.value.UncheckedGet<bool>();
        } else if (name == "mask") {
            if (!rbfVkPlanDetail::ReadFloat(param, &out->mask) || out->mask < 0 || out->mask > 1)
                return Fail(diagnostics, "Deform mask must be a finite [0,1] literal");
        } else if (name == "groomEnvelope") {
            if (!rbfVkPlanDetail::ReadFloat(param, &out->groomEnvelope) || out->groomEnvelope < 0 || out->groomEnvelope > 1)
                return Fail(diagnostics, "Deform groomEnvelope must be a finite [0,1] literal");
        } else if (name == "enabled") {
            if (!param.value.IsHolding<bool>()) return Fail(diagnostics, "Deform enabled must be a bool literal");
        } else {
            return Fail(diagnostics, "unsupported Deform parameter " + name);
        }
    }
    if (budget < kRbfVkPlanMinBudget || budget > kRbfVkPlanMaxBudget)
        return Fail(diagnostics, "Deform rbfSamples must be in [1,46336]");

    UsdGenSurfaceDesc const* surface = nullptr;
    for (auto const& item : desc.surfaces)
        if (item.path == node.surfaces.front()) surface = &item;
    if (!surface || surface->restPoints.empty() ||
        surface->points.size() != surface->restPoints.size() ||
        surface->faceVertexCounts.empty())
        return Fail(diagnostics, "Deform pose surface is invalid");
    // ValidateSource already proves the source matrix; re-check defensively
    // since this validator also runs standalone in tests.
    if (!rbfVkPlanDetail::IsFiniteInvertibleAffine(source->worldMatrix))
        return Fail(diagnostics, "C3 source worldMatrix must be finite, affine and invertible");
    if (!rbfVkPlanDetail::IsFiniteAffine(surface->worldMatrix))
        return Fail(diagnostics, "Deform surface worldMatrix must be finite and affine");

    uint32_t const curveCount = uint32_t(source->curveVertexCounts.size());
    if (source->skinPrim.size() != curveCount || source->skinPrimUv.size() != curveCount)
        return Fail(diagnostics, "Deform requires root bindings for every curve");

    // Description-space drivers: rest AND posed through
    // relative = surface.worldMatrix * desc.xformMatrix^-1, matching the
    // surfaceVk convention (a transform animation moves rest and pose
    // together, ops/deform.cpp:99-100), so in-budget and out-of-budget
    // Deform bake the same field for the same scene.
    GfMatrix4d const relative = surface->worldMatrix * desc.xformMatrix.GetInverse();
    size_t const nPts = surface->restPoints.size();
    std::vector<GfVec3d> driverRest(nPts), driverNow(nPts);
    GfRange3d extent;
    for (size_t i = 0; i < nPts; ++i) {
        driverRest[i] = relative.Transform(GfVec3d(surface->restPoints[i]));
        driverNow[i] = relative.Transform(GfVec3d(surface->points[i]));
        extent.UnionWith(driverRest[i]);
    }
    double const size = extent.IsEmpty() ? 0.0 : extent.GetSize().GetLength();
    auto chosen = usdGen::rbf::SelectSamples(driverRest, size_t(budget), std::max(1e-12, size * 1e-7));
    // No <4 rejection: below-minimum counts flow to RbfVkBinding::Bind,
    // which reports the exact CUDA InvalidArgument result.
    GfMatrix4d const toLocal = source->worldMatrix.GetInverse();
    out->sampleCount = int(chosen.size());
    out->restSamples.assign(3 * chosen.size(), 0.0f);
    out->posedSamples.assign(3 * chosen.size(), 0.0f);
    for (size_t k = 0; k < chosen.size(); ++k) {
        GfVec3d const restLocal = toLocal.Transform(driverRest[chosen[k]]);
        GfVec3d const posedLocal = toLocal.Transform(driverNow[chosen[k]]);
        out->restSamples[3*k]   = float(restLocal[0]);
        out->restSamples[3*k+1] = float(restLocal[1]);
        out->restSamples[3*k+2] = float(restLocal[2]);
        out->posedSamples[3*k]   = float(posedLocal[0]);
        out->posedSamples[3*k+1] = float(posedLocal[1]);
        out->posedSamples[3*k+2] = float(posedLocal[2]);
    }

    // Root targets: per-curve face+UV gather from the pose surface's
    // current points, through relative into description space, then into
    // source-local space with the samples.
    std::vector<uint32_t> faceStart(surface->faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < surface->faceVertexCounts.size(); ++f)
        faceStart[f + 1] = faceStart[f] + uint32_t(surface->faceVertexCounts[f]);
    out->rootTargets.assign(3 * curveCount, 0.0f);
    for (uint32_t c = 0; c < curveCount; ++c) {
        int const faceIdx = source->skinPrim[c];
        if (faceIdx < 0 || faceIdx >= int(surface->faceVertexCounts.size()))
            return Fail(diagnostics, "Deform root binding references an invalid face");
        GfVec3f gathered = rbfVkPlanDetail::GatherRoot(*surface, faceStart, faceIdx, source->skinPrimUv[c]);
        GfVec3d const local = toLocal.Transform(relative.Transform(GfVec3d(gathered)));
        out->rootTargets[3*c]   = float(local[0]);
        out->rootTargets[3*c+1] = float(local[1]);
        out->rootTargets[3*c+2] = float(local[2]);
    }
    return true;
}

} // namespace usdGen::vulkan
