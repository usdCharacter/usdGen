// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_CSOURCE_VK_CAPTURE_H
#define USDGEN_VULKAN_CSOURCE_VK_CAPTURE_H

#include "usdGen/curveLoader.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGen/op.h"
#include "usdGen/surfaceCageSource.h"
#include "usdGenCurves/vulkan/vkStages.h"
#include "usdGenMath/math.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace usdGen::vulkan {

inline bool IsVulkanSurfaceCage(UsdGenOpNode const& node) {
    UsdGenVulkanKernel const cs = UsdGenVulkanKernelFor(UsdGenVulkanStage::CurveSource);
    for (auto const& p : node.params)
        if (p.name == cs.Param(vulkanP::kCsInterpolationMode))
            return p.value == UsdGenValue(UsdGenVkTokens().surfaceCage);
    return false;
}

// This is descriptor admission only: the shared estimate never opens a map
// and allocates no dense curves. Keep its bounds for source preparation and
// the native graph's resource estimates, exactly as the CUDA source does.
inline bool ValidateVulkanSurfaceCage(
    UsdGenGraphSpec const& desc, UsdGenOpNode const& node,
    UsdGenCurveSetInputView const& sparse, uint64_t* maxCurves, uint64_t* maxCvs,
    UsdGenDiagnostics* diagnostics = nullptr) {
    auto fail = [&](std::string const& reason) {
        if (diagnostics) diagnostics->Error("Vulkan surfaceCage source: " + reason);
        return false;
    };
    if (!maxCurves || !maxCvs || !IsVulkanSurfaceCage(node) ||
        !node.mode.IsEmpty() || !node.inputs.empty() || !node.references.empty() ||
        node.inputSets.size() != 1 || node.inputSets.front() != sparse.key ||
        node.surfaces.size() != 1 || node.maps.size() != 1 ||
        node.mapBindings.size() != 1 || node.maps.front() != node.mapBindings.front().map ||
        node.mapBindings.front().relationship != UsdGenVkTokens().regionMapRel)
        return fail("requires one captured cage, one surface and one usdGen:regionMap");
    std::set<UsdGenToken> seen;
    for (auto const& p : node.params) {
        bool valid = !p.animated && seen.insert(p.name).second;
        auto const& name = p.name.GetString();
        if (name == "useRest") valid &= p.value.IsHolding<bool>();
        else if (name == "resampleTo" || name == "regionMapChannel")
            valid &= p.value.IsHolding<int>();
        else if (name == "densityMultiplier")
            valid &= p.value.IsHolding<float>() &&
                std::isfinite(p.value.UncheckedGet<float>()) && p.value.UncheckedGet<float>() > 0;
        else if (name == "expectMapGeneration") valid &= p.value.IsHolding<uint64_t>();
        else if (name == "expectEpoch") valid &= p.value.IsHolding<std::string>();
        else if (name == "interpolationMode" || name == "idSource" ||
                 name == "staleAction" || name == "rebind") valid &= p.value.IsHolding<UsdGenToken>();
        else valid = false;
        if (!valid) return fail("unsupported, duplicate or malformed control " + name);
    }
    for (auto const& ramp : node.ramps)
        if (!ramp.positions.empty() || !ramp.colors.empty() ||
            std::any_of(ramp.knots.begin(), ramp.knots.end(), [](auto const& k) {
                return k[1] != 1.0f || !std::isfinite(k[0]);
            })) return fail("non-identity source ramp");
    uint64_t curves = 0, points = 0;
    std::string error;
    if (!UsdGenEstimateSurfaceCageCurveSet(desc, node, sparse, &curves, &points, &error))
        return fail(error);
    if (curves > UINT32_MAX || points > uint64_t(std::numeric_limits<int>::max()))
        return fail("dense capacity exceeds the native cardinality domain");
    *maxCurves = curves;
    *maxCvs = points;
    return true;
}

// Worker-only immutable source capture. CPU map sampling and cage expansion
// are the same source-preparation contract used by CUDA; subsequent operators
// and the source resample stage still execute on the selected device.
inline bool CaptureVulkanSurfaceCage(
    UsdGenGraphSpec const& desc, UsdGenOpNode const& node,
    UsdGenCurveSetInputView const& sparse, uint64_t maxCurves, uint64_t maxCvs,
    UsdGenCurveBuffer* out, UsdGenDiagnostics* diagnostics = nullptr) {
    if (!out) return false;
    UsdGenCurveSetData dense;
    std::string error;
    if (!UsdGenBuildSurfaceCageCurveSet(desc, node, sparse, &dense, &error)) {
        if (diagnostics) diagnostics->Error("Vulkan surfaceCage source capture: " + error);
        return false;
    }
    if (dense.curveVertexCounts.size() > maxCurves || dense.points.size() > maxCvs) {
        if (diagnostics) diagnostics->Error("Vulkan surfaceCage source exceeds admitted capacity");
        return false;
    }
    auto captureNode = node;
    UsdGenToken const &resampleTo =
        UsdGenVulkanKernelFor(UsdGenVulkanStage::CurveSource).Param(vulkanP::kCsResampleTo);
    for (auto& p : captureNode.params)
        if (p.name == resampleTo) p.value = UsdGenValue(0);
    UsdGenParamView params{&desc, &captureNode};
    UsdGenCaptureContext context;
    context.desc = &desc;
    context.params = &params;
    context.seed = uint32_t(node.seed);
    context.diag = diagnostics;
    UsdGenCurveBuffer captured;
    if (!UsdGenCurveLoader::Load(context, UsdGenCurveSetInputView::Of(dense, sparse.key), &captured,
                                 diagnostics))
        return false;
    *out = std::move(captured);
    return true;
}

// surfaceCage roots do not exist during compilation. Gather immutable pose
// targets in description coordinates before upload, after stable-ID sorting and
// root rejection; later culling carries this primitive plane on the GPU.
inline bool GatherVulkanSourceRootTargets(
    UsdGenGraphSpec const& desc, UsdGenOpNode const& poseNode,
    UsdGenCurveBuffer const& captured, std::vector<float>* out,
    UsdGenDiagnostics* diagnostics = nullptr) {
    auto fail = [&](char const* reason) {
        if (diagnostics) diagnostics->Error(std::string("Vulkan surfaceCage pose roots: ") + reason);
        return false;
    };
    if (!out || poseNode.surfaces.size() != 1 ||
        captured.rootPrim.size() != captured.totalCurves ||
        captured.rootUV.size() != captured.totalCurves)
        return fail("incomplete captured root bindings");
    UsdGenInputRecord const* surfaceRecord = UsdGenFindSurfaceInput(desc, poseNode.surfaces.front());
    if (!surfaceRecord) return fail("unresolved pose surface");
    UsdGenSurfaceInputView const surfaceView = UsdGenSurfaceInputView::Of(*surfaceRecord);
    UsdGenSurfaceInputView const* surface = &surfaceView;
    std::vector<size_t> starts{0};
    for (int n : surface->faceVertexCounts) {
        if ((n != 3 && n != 4) || size_t(n) > SIZE_MAX - starts.back())
            return fail("pose surface requires triangles or quads");
        starts.push_back(starts.back() + size_t(n));
    }
    if (starts.back() != surface->faceVertexIndices.size())
        return fail("invalid pose topology");
    for (int index : surface->faceVertexIndices)
        if (index < 0 || size_t(index) >= surface->points.size())
            return fail("invalid pose vertex index");
    double determinant = 0.0;
    auto const inverse = desc.xformMatrix.GetInverse(&determinant);
    if (!std::isfinite(determinant) || std::abs(determinant) <= std::numeric_limits<double>::epsilon())
        return fail("invalid description transform");
    auto const relative = surface->worldMatrix * inverse;
    std::vector<float> targets(size_t(captured.totalCurves) * 3);
    for (size_t c = 0; c < captured.totalCurves; ++c) {
        int const face = captured.rootPrim[c];
        if (face < 0 || size_t(face) >= surface->faceVertexCounts.size())
            return fail("invalid captured face index");
        auto p = [&](int corner) {
            return surface->points[surface->faceVertexIndices[starts[size_t(face)] + size_t(corner)]];
        };
        float const u = captured.rootUV[c][0], v = captured.rootUV[c][1];
        UsdGenVec3f local;
        if (surface->faceVertexCounts[size_t(face)] == 3)
            local = p(0) * (1.0f-u-v) + p(1) * u + p(2) * v;
        else
            local = p(0) * ((1.0f-u)*(1.0f-v)) + p(1) * (u*(1.0f-v)) +
                    p(2) * (u*v) + p(3) * ((1.0f-u)*v);
        auto const world = relative.Transform(UsdGenVec3d(local));
        for (size_t k = 0; k < 3; ++k) {
            if (!std::isfinite(world[k]) || !std::isfinite(float(world[k])))
                return fail("non-finite pose target");
            targets[3*c+k] = float(world[k]);
        }
    }
    *out = std::move(targets);
    return true;
}

} // namespace usdGen::vulkan
#endif
