// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/vulkan/surfaceVk.h"

#include "usdGen/op.h"
#include "usdGen/ops/rbfField.h"

#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/vec3d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace usdGen::vulkan {
namespace {

bool Fail(UsdGenDiagnostics* diagnostics, std::string const& message) {
    if (diagnostics) diagnostics->Error("Vulkan source-width plan: " + message);
    return false;
}

// Same native-free contract as executionPlan.cpp: GfMatrix4d is row-major
// with row-vector affine transforms (translation in matrix[3][0..2]).
bool IsFiniteAffine(GfMatrix4d const& matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column])) return false;
    return matrix[0][3] == 0.0 && matrix[1][3] == 0.0 &&
        matrix[2][3] == 0.0 && matrix[3][3] == 1.0;
}

bool IsFiniteInvertibleAffine(GfMatrix4d const& matrix) {
    if (!IsFiniteAffine(matrix)) return false;
    double determinant = 0.0;
    GfMatrix4d const inverse = matrix.GetInverse(&determinant);
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= std::numeric_limits<double>::epsilon())
        return false;
    return IsFiniteAffine(inverse);
}

// Barycentric (triangle) / bilinear (quad) interpolation of a surface face
// at a UV coordinate. Mirrors the CUDA GatherRoots kernel in
// gpu/surfaceBinding.cu exactly, in float arithmetic with the same
// evaluation order: triangle a*w+b*u+c*v with w=1-u-v, quad
// a*u0*v0+b*u*v0+c*u*v+d*u0*v.
GfVec3f GatherRootLocal(VtVec3fArray const& points, VtIntArray const& indices,
                        uint32_t base, int n, GfVec2f const& uv) {
    float const u = uv[0], v = uv[1];
    GfVec3f const a(points[indices[base]]);
    GfVec3f const b(points[indices[base + 1]]);
    GfVec3f const c(points[indices[base + 2]]);
    if (n == 3) {
        float const w = 1.0f - u - v;
        return GfVec3f(a[0] * w + b[0] * u + c[0] * v,
                       a[1] * w + b[1] * u + c[1] * v,
                       a[2] * w + b[2] * u + c[2] * v);
    }
    GfVec3f const d(points[indices[base + 3]]);
    float const u0 = 1.0f - u, v0 = 1.0f - v;
    return GfVec3f(a[0] * u0 * v0 + b[0] * u * v0 + c[0] * u * v + d[0] * u0 * v,
                   a[1] * u0 * v0 + b[1] * u * v0 + c[1] * u * v + d[1] * u0 * v,
                   a[2] * u0 * v0 + b[2] * u * v0 + c[2] * u * v + d[2] * u0 * v);
}

} // namespace

bool CompileSurfaceVkDeform(UsdGenCurveSetDesc const& source,
                            UsdGenSurfaceDesc const& surface, int budget,
                            VulkanDeformControls* out,
                            UsdGenDiagnostics* diagnostics) {
    if (!out) return Fail(diagnostics, "Deform surface binding output is null");
    if (budget < 4 || budget > 100)
        return Fail(diagnostics, "Deform rbfSamples must be in [4,100]");
    if (surface.restPoints.empty() ||
        surface.points.size() != surface.restPoints.size() ||
        surface.faceVertexCounts.empty())
        return Fail(diagnostics, "Deform pose surface is invalid");
    if (!IsFiniteInvertibleAffine(source.worldMatrix) ||
        !IsFiniteInvertibleAffine(surface.worldMatrix))
        return Fail(diagnostics,
            "Deform requires finite, affine and invertible source and surface transforms");

    uint32_t const curveCount = uint32_t(source.curveVertexCounts.size());
    if (source.skinPrim.size() != curveCount || source.skinPrimUv.size() != curveCount)
        return Fail(diagnostics, "Deform requires root bindings for every curve");

    // Row-vector convention (curveRootCapture.cpp): a surface-local point p
    // reaches source-local space as p * surface * source^-1.
    double determinant = 0.0;
    GfMatrix4d const sourceInverse = source.worldMatrix.GetInverse(&determinant);
    GfMatrix4d const surfaceToSource = surface.worldMatrix * sourceInverse;
    if (!IsFiniteAffine(surfaceToSource))
        return Fail(diagnostics,
            "Deform surface-to-source transform is non-finite or non-affine");

    // FPF on the source-local rest pose -> chosen indices; pair with the
    // source-local current pose. Same tolerance rule as ValidateDeform.
    size_t const nPts = surface.restPoints.size();
    std::vector<GfVec3d> driverRest(nPts), driverNow(nPts);
    GfRange3d extent;
    for (size_t i = 0; i < nPts; ++i) {
        driverRest[i] = surfaceToSource.TransformAffine(GfVec3d(surface.restPoints[i]));
        driverNow[i] = surfaceToSource.TransformAffine(GfVec3d(surface.points[i]));
        extent.UnionWith(driverRest[i]);
    }
    double const size = extent.IsEmpty() ? 0.0 : extent.GetSize().GetLength();
    auto chosen = usdGen::rbf::SelectSamples(driverRest, size_t(budget),
                                             std::max(1e-12, size * 1e-7));
    if (chosen.size() < 4)
        return Fail(diagnostics,
            "Deform pose surface has fewer than four spanning-3D samples for the RBF fit");
    out->sampleCount = int(chosen.size());
    out->restSamples.assign(3 * chosen.size(), 0.0f);
    out->posedSamples.assign(3 * chosen.size(), 0.0f);
    for (size_t k = 0; k < chosen.size(); ++k) {
        out->restSamples[3 * k] = float(driverRest[chosen[k]][0]);
        out->restSamples[3 * k + 1] = float(driverRest[chosen[k]][1]);
        out->restSamples[3 * k + 2] = float(driverRest[chosen[k]][2]);
        out->posedSamples[3 * k] = float(driverNow[chosen[k]][0]);
        out->posedSamples[3 * k + 1] = float(driverNow[chosen[k]][1]);
        out->posedSamples[3 * k + 2] = float(driverNow[chosen[k]][2]);
    }

    // Root targets: per-curve face+UV gather from the pose surface's current
    // points, mapped into source-local space.
    std::vector<uint32_t> faceStart(surface.faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < surface.faceVertexCounts.size(); ++f)
        faceStart[f + 1] = faceStart[f] + uint32_t(surface.faceVertexCounts[f]);
    out->rootTargets.assign(3 * curveCount, 0.0f);
    for (uint32_t c = 0; c < curveCount; ++c) {
        int const faceIdx = source.skinPrim[c];
        if (faceIdx < 0 || faceIdx >= int(surface.faceVertexCounts.size()))
            return Fail(diagnostics, "Deform root binding references an invalid face");
        GfVec3f local = GatherRootLocal(surface.points, surface.faceVertexIndices,
                                        faceStart[faceIdx],
                                        surface.faceVertexCounts[faceIdx],
                                        source.skinPrimUv[c]);
        GfVec3d mapped = surfaceToSource.TransformAffine(GfVec3d(local));
        out->rootTargets[3 * c] = float(mapped[0]);
        out->rootTargets[3 * c + 1] = float(mapped[1]);
        out->rootTargets[3 * c + 2] = float(mapped[2]);
    }
    return true;
}

} // namespace usdGen::vulkan
