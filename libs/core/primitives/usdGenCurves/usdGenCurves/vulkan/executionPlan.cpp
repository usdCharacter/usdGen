// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGenCurves/types/curveTokens.h"
#include "usdGenVulkan/backendId.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenCurves/vulkan/csourceVkCapture.h"
#include "usdGenExpr/vulkan/exprVkPlan.h"
#include "usdGenCurves/vulkan/growVkPlan.h"
#include "usdGen/authoredNamedChannels.h"
#include "usdGen/surfaceRootFrames.h"
#include "usdGenCurves/vulkan/csourceVkAdmission.h"
#include "usdGenSurface/vulkan/surfaceVk.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "usdGenMath/math.h"
#include "usdGen/op.h"
#include "usdGenMath/rbf.h"
#include "usdGenCurves/vulkan/rbfVkPlan.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/token.h"
#include "usdGenCurves/vulkan/vkStages.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string>

namespace usdGen::vulkan {
namespace {

bool Fail(UsdGenDiagnostics *diagnostics, std::string const &message) {
    if (diagnostics)
        diagnostics->Error("Vulkan source-width plan: " + message);
    return false;
}

bool IsIdentity(UsdGenMatrix4d const &matrix) { return matrix == UsdGenMatrix4d(1.0); }

bool IsFiniteMatrix(UsdGenMatrix4d const &matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column]))
                return false;
    return true;
}

bool IsFiniteAffine(UsdGenMatrix4d const &matrix) {
    return IsFiniteMatrix(matrix) && matrix[0][3] == 0.0 && matrix[1][3] == 0.0 &&
           matrix[2][3] == 0.0 && matrix[3][3] == 1.0;
}

// UsdGenMatrix4d is row-major and uses row-vector affine transforms: translation
// occupies matrix[3][0..2].  Keep this native-free check aligned with C3 root
// capture, because the immutable descriptor remains source-local and root
// frames are converted through source.worldMatrix later.
bool IsFiniteInvertibleAffine(UsdGenMatrix4d const &matrix) {
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(matrix[row][column]))
                return false;
    if (matrix[0][3] != 0.0 || matrix[1][3] != 0.0 || matrix[2][3] != 0.0 || matrix[3][3] != 1.0)
        return false;
    double determinant = 0.0;
    UsdGenMatrix4d const inverse = matrix.GetInverse(&determinant);
    if (!std::isfinite(determinant) ||
        std::abs(determinant) <= std::numeric_limits<double>::epsilon())
        return false;
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            if (!std::isfinite(inverse[row][column]))
                return false;
    return inverse[0][3] == 0.0 && inverse[1][3] == 0.0 && inverse[2][3] == 0.0 &&
           inverse[3][3] == 1.0;
}

bool IsFinite(float value) { return std::isfinite(value); }

bool ReadFloat(UsdGenParamValue const &param, float *value) {
    if (!param.value.IsHolding<float>())
        return false;
    *value = param.value.UncheckedGet<float>();
    return IsFinite(*value);
}

// C1 rework: mask survives as one scalar `usdGen:mask` attribute per operator.
// Vulkan admits only the neutral value; a muted operator would alias its
// input, which the shader lanes express as their own envelope path instead.
bool IsIdentityMask(UsdGenParamValue const &param) {
    return param.name == UsdGenTokens().mask && param.value == UsdGenValue(1.0f);
}

bool ValidateSource(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                    UsdGenCurveSetInputView *source, VulkanSourceControls *controls,
                    UsdGenDiagnostics *diagnostics) {
    using namespace vulkanP;
    UsdGenEngineTokens const &et = UsdGenTokens();
    UsdGenVulkanTokens const &vk = UsdGenVkTokens();
    UsdGenVulkanKernel const cs = UsdGenVulkanKernelFor(UsdGenVulkanStage::CurveSource);
    if (IsVulkanSurfaceCage(node)) {
        if (node.inputSets.size() != 1)
            return Fail(diagnostics, "surfaceCage requires one sparse cage");
        UsdGenInputRecord const *foundRecord = UsdGenFindCurveSetInput(desc, node.inputSets.front());
        if (!foundRecord)
            return Fail(diagnostics, "surfaceCage requires one sparse cage");
        UsdGenCurveSetInputView const foundView = UsdGenCurveSetInputView::Of(*foundRecord);
        UsdGenCurveSetInputView const *found = &foundView;
        if (!ValidateVulkanSurfaceCage(desc, node, *found, &controls->maxCurves, &controls->maxCvs,
                                       diagnostics))
            return false;
        controls->surfaceCage = true;
        controls->hasAuthoredRest = !found->rest.empty();
        for (auto const &param : node.params) {
            if (param.name == cs.Param(kCsUseRest))
                controls->useRest = param.value.UncheckedGet<bool>();
            else if (param.name == cs.Param(kCsResampleTo)) {
                int target = 0;
                if (CsourceVkValidateResampleTarget(param.value, controls->maxCurves, &target,
                                                    nullptr) != CsourceVkResampleStatus::Ok)
                    return Fail(diagnostics, "surfaceCage resampleTo must be 0 or >= 2");
                controls->resampleTo = target;
            }
        }
        *source = *found;
        return true;
    }
    if (!node.mode.IsEmpty() || !node.inputs.empty() || !node.references.empty() ||
        !node.maps.empty() || !node.mapBindings.empty())
        return Fail(diagnostics, "CurveSource has unsupported enabled/input configuration");
    for (auto const &ramp : node.ramps)
        if (!ramp.positions.empty() || !ramp.colors.empty() ||
            std::any_of(ramp.knots.begin(), ramp.knots.end(), [](auto const &k) {
                return k[1] != 1.0f || !std::isfinite(k[0]);
            }))
            return Fail(diagnostics, "non-identity source ramps are not supported");
    if (node.inputSets.size() != 1)
        return Fail(diagnostics, "CurveSource requires exactly one C3 curve target");
    UsdGenInputRecord const *foundRecord = UsdGenFindCurveSetInput(desc, node.inputSets.front());
    UsdGenCurveSetInputView const foundView =
        foundRecord ? UsdGenCurveSetInputView::Of(*foundRecord) : UsdGenCurveSetInputView{};
    UsdGenCurveSetInputView const *found = &foundView;
    if (!foundRecord || found->role != UsdGenLane::Chunked ||
        found->curveRole != UsdGenCurveTokens().hair)
        return Fail(diagnostics, "CurveSource target must be one resolved C3 hair curve set");
    // Current/rest points stay source-local. Only constructing absent root
    // frames from a surface requires this matrix's inverse; authored frames
    // and explicitly absent surface-free root channels do not.
    if (!IsFiniteAffine(found->worldMatrix))
        return Fail(diagnostics, "C3 source worldMatrix must be finite and affine");
    // Vulkan uses the same immutable rest-surface/root-frame capture as the
    // native C3 loader.  Relationship resolution is a compiler fact, never a
    // later live-stage lookup.
    if (node.surfaces.size() > 1)
        return Fail(diagnostics, "CurveSource permits at most one rest surface");
    if (node.surfaces.size() == 1) {
        if (!node.surfaces.front().IsValid())
            return Fail(diagnostics, "CurveSource rest surface relationship is empty");
        UsdGenSpan<const UsdGenInputRecord> const surfaces = UsdGenSurfaceInputs(desc);
        size_t const resolved = std::count_if(
            surfaces.begin(), surfaces.end(),
            [&](UsdGenInputRecord const &item) { return item.key == node.surfaces.front(); });
        if (resolved != 1)
            return Fail(diagnostics,
                        "CurveSource rest surface relationship is unresolved or ambiguous");
        UsdGenInputRecord const *surface = UsdGenFindSurfaceInput(desc, node.surfaces.front());
        if (!IsFiniteInvertibleAffine(UsdGenSurfaceInputView::Of(*surface).worldMatrix))
            return Fail(diagnostics, "CurveSource rest surface worldMatrix must be "
                                     "finite, affine and invertible");
    }
    controls->hasAuthoredRest = !found->rest.empty();
    controls->restFromCurrentPoints = found->restFromCurrentPoints;
    if (found->type != UsdGenCurveTokens().cubic ||
        (found->basis != UsdGenCurveTokens().bspline && found->basis != UsdGenCurveTokens().catmullRom) ||
        (found->wrap != UsdGenCurveTokens().pinned && found->wrap != UsdGenCurveTokens().nonperiodic) ||
        (found->widthsInterpolation != et.vertex &&
         found->widthsInterpolation != et.constant))
        return Fail(diagnostics, "C3 source requires cubic bspline/catmullRom, "
                                 "pinned/nonperiodic and vertex/constant widths");
    size_t const widthCount = found->widthsInterpolation == et.constant
                                 ? 1 : found->points.size();
    if ((found->curveVertexCounts.empty() != found->points.empty()) ||
        (!found->rest.empty() && found->rest.size() != found->points.size()) ||
        (!found->widths.empty() && found->widths.size() != widthCount))
        return Fail(diagnostics, "C3 source has unsupported topology or binding data");
    uint64_t points = 0;
    std::set<uint64_t> ids;
    for (int count : found->curveVertexCounts) {
        if (count < 2 || uint64_t(count) > std::numeric_limits<uint64_t>::max() - points)
            return Fail(diagnostics, "C3 source has invalid curve topology");
        points += uint64_t(count);
    }
    if (points != found->points.size())
        return Fail(diagnostics, "C3 source point count disagrees with topology");
    if (points > uint64_t(std::numeric_limits<int>::max()) ||
        found->curveVertexCounts.size() > size_t(UINT32_MAX))
        return Fail(diagnostics, "C3 source exceeds the native offset/cardinality domain");
    std::string namedReason;
    if (!ValidateAuthoredNamedChannels(*found, found->curveVertexCounts.size(), size_t(points),
                                       &namedReason))
        return Fail(diagnostics, "C3 authored named channels: " + namedReason);
    for (auto const &point : found->points)
        for (int component = 0; component != 3; ++component)
            if (!std::isfinite(point[component]))
                return Fail(diagnostics, "C3 source contains a non-finite point");
    for (auto const &point : found->rest)
        for (int component = 0; component != 3; ++component)
            if (!std::isfinite(point[component]))
                return Fail(diagnostics, "C3 source contains a non-finite rest point");
    for (float width : found->widths)
        if (!IsFinite(width) || width < 0)
            return Fail(diagnostics, "C3 source contains an invalid width");
    std::set<UsdGenToken> seen;
    for (auto const &param : node.params) {
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate CurveSource parameter " + param.name.GetString());
        auto const &name = param.name.GetString();
        bool valid = false;
        if (name == "useRest") {
            valid = param.value.IsHolding<bool>();
            if (valid)
                controls->useRest = param.value.UncheckedGet<bool>();
        } else if (name == "lane")
            valid = param.value == UsdGenValue(UsdGenCurveTokens().hair);
        else if (name == "idSource") {
            valid = param.value == UsdGenValue(vk.primvar) ||
                    param.value == UsdGenValue(vk.index);
            if (valid)
                controls->idSource = param.value.UncheckedGet<UsdGenToken>();
        } else if (name == "resampleTo") {
            // Merged resample/csourceVk admission: CsourceVkValidateResampleTarget
            // implements the agreed contract (0 keeps layout, >= 2 resamples,
            // negative/1/non-int/overflow reject with CUDA-matching bounds).
            // The diagnostic carries both gaps' needles (ExpectReject matches
            // by substring): the generic CurveSource parameter prefix plus the
            // specific reason.
            int target = 0;
            std::string resampleReason;
            if (CsourceVkValidateResampleTarget(param.value, found->curveVertexCounts.size(),
                                                &target,
                                                &resampleReason) != CsourceVkResampleStatus::Ok) {
                if (resampleReason.empty())
                    return Fail(diagnostics,
                                "unsupported or malformed CurveSource parameter " + name);
                return Fail(diagnostics, "unsupported or malformed CurveSource parameter " + name +
                                             ": " + resampleReason);
            }
            controls->resampleTo = target;
            valid = true;
        } else if (name == "expectEpoch") {
            valid = param.value.IsHolding<std::string>();
            if (valid)
                controls->expectEpoch = param.value.UncheckedGet<std::string>();
        } else if (name == "staleAction") {
            valid = param.value == UsdGenValue(vk.warn) ||
                    param.value == UsdGenValue(vk.ignore) ||
                    param.value == UsdGenValue(vk.block);
            if (valid)
                controls->staleAction = param.value.UncheckedGet<UsdGenToken>();
        } else if (name == "rebind") {
            valid = param.value == UsdGenValue(vk.never) ||
                    param.value == UsdGenValue(vk.onError) ||
                    param.value == UsdGenValue(vk.always);
            if (valid)
                controls->rebind = param.value.UncheckedGet<UsdGenToken>();
        } else if (name == "label")
            valid = param.value.IsHolding<std::string>();
        else if (name == "interpolationMode")
            valid = param.value == UsdGenValue(et.none);
        else if (name == "densityMultiplier")
            valid = param.value.IsHolding<float>() &&
                    IsFinite(param.value.UncheckedGet<float>()) &&
                    param.value.UncheckedGet<float>() > 0.0f;
        else if (name == "regionMapChannel")
            valid = param.value.IsHolding<int>();
        else if (name == "expectMapGeneration")
            valid = param.value.IsHolding<uint64_t>();
        if (!valid)
            return Fail(diagnostics, "unsupported or malformed CurveSource parameter " + name);
    }
    if (controls->idSource == vk.primvar && found->curveId.empty()) {
        controls->stableIdsSynthesized = true;
        if (diagnostics)
            diagnostics->Warn("Vulkan source-width plan: CurveSource "
                              "idSource=primvar synthesizes index stable IDs");
    }
    if (controls->idSource == vk.primvar && !found->curveId.empty() &&
        found->curveId.size() != found->curveVertexCounts.size())
        return Fail(diagnostics, "CurveSource idSource=primvar has incomplete stable IDs");
    // Index mode deliberately ignores the authored ID array, including its
    // shape and duplicate values, just as the CPU source capture does.
    if (controls->idSource == vk.primvar)
        for (uint64_t id : found->curveId)
            if (!ids.insert(id).second)
                return Fail(diagnostics, "C3 source stable IDs must be unique");
    if (controls->useRest &&
        (found->restFromCurrentPoints || found->rest.size() != found->points.size()))
        return Fail(diagnostics, "CurveSource useRest=true requires an authored "
                                 "non-current rest sample");
    bool const preservesAbsentRoots = node.surfaces.empty() &&
        controls->rebind == vk.never && found->skinPrim.empty() &&
        found->skinPrimUv.empty() && found->rootFrame.empty();
    if (!found->curveVertexCounts.empty() && found->rootFrame.empty() &&
        !preservesAbsentRoots && !IsFiniteInvertibleAffine(found->worldMatrix))
        return Fail(diagnostics, "C3 root frame construction requires an invertible source worldMatrix");
    if (node.surfaces.empty() && !found->curveVertexCounts.empty() &&
        !CsourceVkSurfaceFreeRebindAdmitted(controls->rebind, found->curveVertexCounts.size(),
                                            found->rootFrame.size()))
        return Fail(diagnostics, "surface-free CurveSource requires rebind=never");
    if (!controls->expectEpoch.empty() && controls->expectEpoch != found->frozenEpoch) {
        if (controls->staleAction == vk.block)
            return Fail(diagnostics, "CurveSource expectEpoch mismatches frozenEpoch "
                                     "with staleAction=block");
        if (controls->staleAction == vk.warn && diagnostics)
            diagnostics->Warn("Vulkan source-width plan: CurveSource expectEpoch "
                              "mismatches frozenEpoch");
    }
    // Admission does not build a surface index or query roots.  Keep the
    // exact authored-frame contract in the shared helper; validity of an
    // individual face/UV is intentionally left to CurveLoader root capture,
    // where never drops and onError/always can rebind it.
    if (found->skinPrim.size() != found->skinPrimUv.size() && controls->rebind == vk.never)
        return Fail(diagnostics, "C3 rootPrim/rootUV cardinalities differ for rebind=never");
    if (!found->rootFrame.empty() && found->rootFrame.size() != found->curveVertexCounts.size())
        return Fail(diagnostics, "C3 authored rootFrame cardinality disagrees with curves");
    for (UsdGenMatrix4d const &frame : found->rootFrame)
        if (!IsVulkanCapturedScatter(desc, *found) && !UsdGenValidateAuthoredRootFrame(frame))
            return Fail(diagnostics, "C3 authored rootFrame violates the affine "
                                     "orthonormal frame contract");
    *source = *found;
    return true;
}

bool ValidateLength(UsdGenOpNode const &node, VulkanSourceWidthStage *stage,
                    UsdGenDiagnostics *diagnostics) {
    // C1 rework removed the node-level mute: `usdGen:mask` is the one
    // envelope control and it feeds the stage envelope directly.
    if (!node.mode.IsEmpty() || !node.references.empty() || !node.inputSets.empty() ||
        !node.surfaces.empty() || !node.maps.empty() || !node.mapBindings.empty() ||
        !node.ramps.empty())
        return Fail(diagnostics, "Length requires literal identity-envelope scale, "
                                 "set or cull configuration");
    UsdGenVulkanTokens const &vk = UsdGenVkTokens();
    UsdGenToken const &modeName =
        UsdGenVulkanKernelFor(UsdGenVulkanStage::Length).Param(vulkanP::kLenMode);
    bool cull = false, set = false;
    for (auto const &param : node.params)
        if (param.name == modeName && param.value == UsdGenValue(vk.cull))
            cull = true;
        else if (param.name == modeName && param.value == UsdGenValue(vk.set))
            set = true;
    stage->kind =
        cull ? VulkanSourceWidthStage::Kind::LengthCull : VulkanSourceWidthStage::Kind::LengthScale;
    stage->lengthMode =
        set ? VulkanSourceWidthStage::LengthMode::Set : VulkanSourceWidthStage::LengthMode::Scale;
    std::set<UsdGenToken> seen;
    for (auto const &param : node.params) {
        if (param.animated)
            return Fail(diagnostics, "Length requires non-animated literal controls");
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate Length parameter");
        auto const &name = param.name.GetString();
        if (name == "length:value") {
            double value;
            if (param.value.IsHolding<double>())
                value = param.value.UncheckedGet<double>();
            else if (param.value.IsHolding<float>())
                value = param.value.UncheckedGet<float>();
            else
                return Fail(diagnostics, "Length value requires a finite numeric literal");
            stage->factor = float(value);
            if (!std::isfinite(value) || value < 0 || !std::isfinite(stage->factor))
                return Fail(diagnostics, "Length factor must be finite, nonnegative "
                                         "and float-representable");
        } else if (name == "length:mode") {
            if (param.value != UsdGenValue(vk.scale) &&
                param.value != UsdGenValue(vk.set) && param.value != UsdGenValue(vk.cull))
                return Fail(diagnostics, "only literal Length scale, set or cull modes supported");
        } else if (name == "length:min" || name == "length:max" ||
                   name == "length:randomize") {
            // Inert randomRange controls (that mode is refused above).
            float inert = 0.0f;
            if (!ReadFloat(param, &inert))
                return Fail(diagnostics, "Length range controls must be finite literals");
        } else if (name == "length:shareByClump" || name == "length:followClumpMean") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Length range toggles must be bool literals");
        } else if (name == "length:method") {
            if (param.value == UsdGenValue(vk.scale))
                stage->lengthMethod = VulkanSourceWidthStage::LengthMethod::Scale;
            else if (param.value == UsdGenValue(vk.cutExtend))
                stage->lengthMethod = VulkanSourceWidthStage::LengthMethod::CutExtend;
            else
                return Fail(diagnostics, "Length method must be scale or cutExtend");
        } else if (name == "length:random") {
            if (!param.value.IsHolding<UsdGenVec2f>())
                return Fail(diagnostics,
                            "Length random requires a finite nonnegative vec2 literal");
            auto const range = param.value.UncheckedGet<UsdGenVec2f>();
            stage->randomLo = range[0];
            stage->randomHi = range[1];
            if (!std::isfinite(stage->randomLo) || !std::isfinite(stage->randomHi) ||
                stage->randomLo < 0 || stage->randomHi < 0)
                return Fail(diagnostics, "Length random requires finite nonnegative endpoints");
        } else if (name == "mask") {
            if (!ReadFloat(param, &stage->lengthMaskAmount) || stage->lengthMaskAmount < 0 ||
                stage->lengthMaskAmount > 1)
                return Fail(diagnostics, "Length mask amount must be a finite [0,1] literal");
        } else if (name == "cullThreshold") {
            double threshold;
            if (param.value.IsHolding<float>())
                threshold = param.value.UncheckedGet<float>();
            else if (param.value.IsHolding<double>())
                threshold = param.value.UncheckedGet<double>();
            else
                return Fail(diagnostics,
                            "Length cull threshold must be a finite nonnegative literal");
            stage->cullThreshold = float(threshold);
            if (!std::isfinite(threshold) || threshold < 0 || !std::isfinite(stage->cullThreshold))
                return Fail(diagnostics,
                            "Length cull threshold must be a finite nonnegative literal");
        } else if (name == "minRemainingLength") {
            double minimum;
            if (param.value.IsHolding<float>())
                minimum = param.value.UncheckedGet<float>();
            else if (param.value.IsHolding<double>())
                minimum = param.value.UncheckedGet<double>();
            else
                return Fail(diagnostics, "Length minimum must be a finite nonnegative literal");
            stage->minRemainingLength = float(minimum);
            if (!std::isfinite(minimum) || minimum < 0 || !std::isfinite(stage->minRemainingLength))
                return Fail(diagnostics, "Length minimum must be a finite nonnegative literal");
        } else if (name == "rebuild") {
            if (param.value == UsdGenValue(vk.keepParam))
                stage->lengthRebuild = VulkanSourceWidthStage::LengthRebuild::KeepParam;
            else if (param.value == UsdGenValue(vk.reparam))
                stage->lengthRebuild = VulkanSourceWidthStage::LengthRebuild::Reparam;
            else
                return Fail(diagnostics, "Length rebuild must be keepParam or reparam");
        } else if (name == "label") {
            if (!param.value.IsHolding<std::string>())
                return Fail(diagnostics, "Length label must be string");
        } else if (!IsIdentityMask(param))
            return Fail(diagnostics, "unsupported Length control " + name);
    }
    if (set && cull)
        return Fail(diagnostics, "Length set and cull are mutually exclusive");
    return true;
}

bool ValidateNoise(UsdGenOpNode const &node, VulkanLiteralNoiseControls *out,
                   UsdGenDiagnostics *diagnostics) {
    // Literal-only lane: no references/curves/surfaces/maps/expressions/ramps.
    // Magnitude ramps are excluded (the pipeline uses a flat 257-entry profile).
    if (!node.mode.IsEmpty() || !node.references.empty() || !node.inputSets.empty() ||
        !node.maps.empty() || !node.mapBindings.empty())
        return Fail(diagnostics, "Noise requires literal configuration");
    std::set<UsdGenToken> seen;
    UsdGenVec2fArray knots;
    UsdGenToken interpolation = UsdGenCurveTokens().catmullRom;
    for (auto const &param : node.params) {
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate Noise parameter");
        auto const &name = param.name.GetString();
        if (name == "noise:magnitude") {
            if (!ReadFloat(param, &out->magnitude) || out->magnitude < 0)
                return Fail(diagnostics, "Noise magnitude must be a finite nonnegative literal");
        } else if (name == "noise:frequency") {
            if (!ReadFloat(param, &out->frequency) || out->frequency <= 0)
                return Fail(diagnostics, "Noise frequency must be a finite positive literal");
        } else if (name == "noise:correlation") {
            if (!ReadFloat(param, &out->correlation) || out->correlation < 0 ||
                out->correlation > 1)
                return Fail(diagnostics, "Noise correlation must be a finite [0,1] literal");
        } else if (name == "noise:octaves") {
            if (!param.value.IsHolding<int>())
                return Fail(diagnostics, "Noise octaves must be an int literal");
            out->octaves = param.value.UncheckedGet<int>();
            if (out->octaves < 1 || out->octaves > 6)
                return Fail(diagnostics, "Noise octaves must be in [1,6]");
        } else if (name == "noise:lacunarity") {
            if (!ReadFloat(param, &out->lacunarity) || out->lacunarity <= 1)
                return Fail(diagnostics, "Noise lacunarity must be a finite >1 literal");
        } else if (name == "noise:gain") {
            if (!ReadFloat(param, &out->gain) || out->gain < 0 || out->gain > 1)
                return Fail(diagnostics, "Noise gain must be a finite [0,1] literal");
        } else if (name == "preserveLength") {
            if (!ReadFloat(param, &out->preserveLength) || out->preserveLength < 0 ||
                out->preserveLength > 1)
                return Fail(diagnostics, "Noise preserveLength must be a finite [0,1] literal");
        } else if (name == "mask") {
            if (!ReadFloat(param, &out->mask) || out->mask < 0 || out->mask > 1)
                return Fail(diagnostics, "Noise mask must be a finite [0,1] literal");
        } else if (name == "noise:seed") {
            if (!param.value.IsHolding<int>())
                return Fail(diagnostics, "Noise seed must be an int literal");
            out->seed = param.value.UncheckedGet<int>();
        } else if (name == "cumulative") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Noise cumulative must be a bool literal");
            out->cumulative = param.value.UncheckedGet<bool>();
        } else if (name == "noise:magnitude:knots") {
            if (!param.value.IsHolding<UsdGenVec2fArray>())
                return Fail(diagnostics, "Noise magnitude knots must be float2 array");
            knots = param.value.UncheckedGet<UsdGenVec2fArray>();
            float previous = -1;
            for (auto const &k : knots) {
                if (!std::isfinite(k[0]) || !std::isfinite(k[1]) || k[0] < 0 || k[0] > 1 ||
                    k[0] < previous)
                    return Fail(diagnostics, "malformed Noise magnitude knots");
                previous = k[0];
            }
        } else if (name == "noise:magnitude:interpolation") {
            if (!param.value.IsHolding<UsdGenToken>())
                return Fail(diagnostics, "Noise magnitude interpolation must be token");
            interpolation = param.value.UncheckedGet<UsdGenToken>();
        } else if (name == "enabled") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Noise enabled must be a bool literal");
            out->enabled = param.value.UncheckedGet<bool>();
        } else if (name == "label") {
            if (!param.value.IsHolding<std::string>())
                return Fail(diagnostics, "Noise label must be string");
        } else {
            return Fail(diagnostics, "unsupported Noise parameter " + name);
        }
    }
    for (auto const &ramp : node.ramps)
        if (!ramp.knots.empty() || !ramp.positions.empty() || !ramp.colors.empty())
            return Fail(diagnostics, "anonymous Noise ramp requires named parameter");
    if (!knots.empty())
        UsdGenBuildRampLut(knots, interpolation, out->profile.data(), 257);
    return true;
}

// Barycentric (triangle) / bilinear (quad) interpolation of a surface face at
// a UV coordinate — mirrors the CUDA GatherRoots kernel exactly.
UsdGenVec3f RootPosition(UsdGenSurfaceInputView const &surface, std::vector<uint32_t> const &faceStart,
                     int faceIdx, UsdGenVec2f const &uv) {
    auto const &indices = surface.faceVertexIndices;
    auto const &pts = surface.points;
    uint32_t const base = faceStart[faceIdx];
    int const n = surface.faceVertexCounts[faceIdx];
    auto P = [&](int j) { return UsdGenVec3f(pts[indices[base + j]]); };
    float const u = uv[0], v = uv[1];
    if (n == 3) {
        UsdGenVec3f const a = P(0), b = P(1), c = P(2);
        return a * (1.0f - u - v) + b * u + c * v;
    }
    UsdGenVec3f const a = P(0), b = P(1), c = P(2), d = P(3);
    float const u0 = 1.0f - u, v0 = 1.0f - v;
    return a * (u0 * v0) + b * (u * v0) + c * (u * v) + d * (u0 * v);
}

bool ValidateDeform(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                    UsdGenCurveSetInputView const *source, VulkanDeformControls *out,
                    UsdGenDiagnostics *diagnostics) {
    if (!source || node.surfaces.size() != 1 || !node.references.empty() || !node.inputSets.empty() ||
        !node.maps.empty() || !node.mapBindings.empty() || !node.ramps.empty())
        return Fail(diagnostics,
                    "Deform requires one pose surface and no references/maps/expressions");
    std::set<UsdGenToken> seen;
    int budget = 100;
    for (auto const &param : node.params) {
        if (param.animated)
            return Fail(diagnostics, "Deform requires non-animated literal controls");
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate Deform parameter " + param.name.GetString());
        auto const &name = param.name.GetString();
        if (name == "rbfSamples") {
            if (!param.value.IsHolding<int>())
                return Fail(diagnostics, "Deform rbfSamples must be an int literal");
            budget = param.value.UncheckedGet<int>();
        } else if (name == "lockRoots") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Deform lockRoots must be a bool literal");
            out->lockRoots = param.value.UncheckedGet<bool>();
        } else if (name == "mask") {
            if (!ReadFloat(param, &out->mask) || out->mask < 0 || out->mask > 1)
                return Fail(diagnostics, "Deform mask must be a finite [0,1] literal");
        } else if (name == "groomEnvelope") {
            if (!ReadFloat(param, &out->groomEnvelope) || out->groomEnvelope < 0 ||
                out->groomEnvelope > 1)
                return Fail(diagnostics, "Deform groomEnvelope must be a finite [0,1] literal");
        } else if (name == "enabled") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Deform enabled must be a bool literal");
        } else {
            return Fail(diagnostics, "unsupported Deform parameter " + name);
        }
    }
    // Phase-2 Deform gap routing (rbfVk + surfaceVk merge): out-of-range
    // budgets delegate to the device-path validator (devicePath=true on
    // success, served by RbfVkDeformPipeline); in-budget moved transforms
    // route to the surfaceVk source-local host fit below; admitted configs
    // keep the legacy host-gather path at the end of this function.
    if (budget < 4 || budget > 100) {
        if (!RbfVkValidateDeformGap(desc, node, source, out, diagnostics))
            return false;
        out->devicePath = true;
        return true;
    }

    UsdGenInputRecord const *surfaceRecord = nullptr;
    for (UsdGenInputRecord const &item : UsdGenSurfaceInputs(desc))
        if (item.key == node.surfaces.front())
            surfaceRecord = &item;
    UsdGenSurfaceInputView const surfaceView =
        surfaceRecord ? UsdGenSurfaceInputView::Of(*surfaceRecord) : UsdGenSurfaceInputView{};
    UsdGenSurfaceInputView const *surface = surfaceRecord ? &surfaceView : nullptr;
    if (!surface || surface->restPoints.empty() ||
        surface->points.size() != surface->restPoints.size() || surface->faceVertexCounts.empty())
        return Fail(diagnostics, "Deform pose surface is invalid");
    // Phase 2 (p2-surface): non-identity surface binding compiles the
    // frozen fit in source-local space; identity keeps the legacy
    // host-gather path below.
    if (!IsIdentity(source->worldMatrix) || !IsIdentity(surface->worldMatrix))
        return CompileSurfaceVkDeform(*source, *surface, budget, out, diagnostics);

    uint32_t const curveCount = uint32_t(source->curveVertexCounts.size());
    if (source->skinPrim.size() != curveCount || source->skinPrimUv.size() != curveCount)
        return Fail(diagnostics, "Deform requires root bindings for every curve");

    // FPF on rest pose → chosen indices; pair with current pose.
    size_t const nPts = surface->restPoints.size();
    std::vector<UsdGenVec3d> driverRest(nPts), driverNow(nPts);
    UsdGenRange3d extent;
    for (size_t i = 0; i < nPts; ++i) {
        driverRest[i] = UsdGenVec3d(surface->restPoints[i]);
        driverNow[i] = UsdGenVec3d(surface->points[i]);
        extent.UnionWith(driverRest[i]);
    }
    double const size = extent.IsEmpty() ? 0.0 : extent.GetSize().GetLength();
    auto chosen =
        usdGenMath::rbf::SelectSamples(driverRest, size_t(budget), std::max(1e-12, size * 1e-7));
    if (chosen.size() < 4)
        return Fail(diagnostics, "Deform pose surface has fewer than four "
                                 "spanning-3D samples for the RBF fit");
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
    // points.
    std::vector<uint32_t> faceStart(surface->faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < surface->faceVertexCounts.size(); ++f)
        faceStart[f + 1] = faceStart[f] + uint32_t(surface->faceVertexCounts[f]);
    out->rootTargets.assign(3 * curveCount, 0.0f);
    for (uint32_t c = 0; c < curveCount; ++c) {
        int const faceIdx = source->skinPrim[c];
        if (faceIdx < 0 || faceIdx >= int(surface->faceVertexCounts.size()))
            return Fail(diagnostics, "Deform root binding references an invalid face");
        UsdGenVec3f pos = RootPosition(*surface, faceStart, faceIdx, source->skinPrimUv[c]);
        out->rootTargets[3 * c] = pos[0];
        out->rootTargets[3 * c + 1] = pos[1];
        out->rootTargets[3 * c + 2] = pos[2];
    }
    return true;
}

bool ValidateResample(UsdGenOpNode const &node, VulkanSourceWidthStage *stage,
                      UsdGenDiagnostics *diagnostics) {
    // Indexed-CV (keepParam) subset only: uniform arc-length resampling,
    // mask blending and length restore have no Vulkan lowering and must
    // fail closed, never run silently as indexed interpolation.
    if (!node.mode.IsEmpty() || !node.references.empty() || !node.inputSets.empty() ||
        !node.surfaces.empty() || !node.maps.empty() || !node.mapBindings.empty() ||
        !node.expressionBindings.empty() || !node.ramps.empty())
        return Fail(diagnostics, "Resample requires literal keepParam configuration");
    int cvCount = 0;
    bool keepParamSeen = false;
    std::set<UsdGenToken> seen;
    for (auto const &param : node.params) {
        if (param.animated)
            return Fail(diagnostics, "Resample requires non-animated literal controls");
        if (!seen.insert(param.name).second)
            return Fail(diagnostics, "duplicate Resample parameter");
        auto const &name = param.name.GetString();
        if (name == "cvCount" || name == "usdGen:cvCount") {
            if (!param.value.IsHolding<int>())
                return Fail(diagnostics, "Resample cvCount must be an int literal");
            cvCount = param.value.UncheckedGet<int>();
        } else if (name == "distribution" || name == "usdGen:distribution") {
            if (param.value != UsdGenValue(UsdGenVkTokens().keepParam))
                return Fail(diagnostics, "Resample distribution must be keepParam");
            keepParamSeen = true;
        } else if (name == "mask" || name == "usdGen:mask") {
            if (param.value != UsdGenValue(1.0f))
                return Fail(diagnostics, "Resample mask must be neutral 1.0");
        } else if (name == "restoreSegmentLengths" || name == "usdGen:restoreSegmentLengths") {
            if (param.value != UsdGenValue(false))
                return Fail(diagnostics, "Resample restoreSegmentLengths must be false");
        } else if (name == "spacing" || name == "usdGen:spacing") {
            float spacing = 0.0f;
            if (!ReadFloat(param, &spacing) || spacing != 0.0f)
                return Fail(diagnostics, "Resample spacing requires the CPU reference backend");
        } else if (name == "enabled") {
            if (!param.value.IsHolding<bool>())
                return Fail(diagnostics, "Resample enabled must be a bool literal");
        } else if (name == "label") {
            if (!param.value.IsHolding<std::string>())
                return Fail(diagnostics, "Resample label must be string");
        } else if (!IsIdentityMask(param))
            return Fail(diagnostics, "unsupported Resample control " + name);
    }
    // Absent distribution means the CPU default (uniform), which has no
    // indexed lowering: require the explicit keepParam spelling.
    if (!keepParamSeen)
        return Fail(diagnostics, "Resample distribution must be keepParam");
    // CPU op bound is [1,64]; the indexed device contract needs >= 2.
    if (cvCount < 2 || cvCount > 64)
        return Fail(diagnostics, "Resample cvCount must be in [2,64]");
    stage->kind = VulkanSourceWidthStage::Kind::Resample;
    stage->resampleTarget = uint32_t(cvCount);
    return true;
}

bool Estimate(UsdGenCurveSetInputView const &source, bool rootBindings, int resampleTo, uint64_t *bytes,
              uint64_t *widthBytes, uint64_t capacityCurves = 0, uint64_t capacityPoints = 0) {
    uint64_t authoredPoints = capacityPoints ? capacityPoints : source.points.size(),
             curves = capacityCurves ? capacityCurves : source.curveVertexCounts.size(), total = 0,
             value = 0;
    uint64_t points = authoredPoints;
    if (!CsourceVkEffectivePointCount(authoredPoints, size_t(curves), resampleTo, &points))
        return false;
    if (curves == std::numeric_limits<uint64_t>::max())
        return false;
    bool const ok = UsdGenExecutionCheckedBytes::Multiply(points, 12, &value) &&
                    UsdGenExecutionCheckedBytes::Add(&total, value) &&
                    UsdGenExecutionCheckedBytes::Multiply(curves + 1, 4, &value) &&
                    UsdGenExecutionCheckedBytes::Add(&total, value) &&
                    UsdGenExecutionCheckedBytes::Multiply(curves, 8, &value) &&
                    UsdGenExecutionCheckedBytes::Add(&total, value) &&
                    UsdGenExecutionCheckedBytes::Multiply(points, 12, &value) &&
                    UsdGenExecutionCheckedBytes::Add(&total, value) &&
                    UsdGenExecutionCheckedBytes::Multiply(points, 4, widthBytes) &&
                    UsdGenExecutionCheckedBytes::Add(&total, *widthBytes) &&
                    UsdGenExecutionCheckedBytes::Add(&total, *widthBytes); // immutable hairT
    // Reserve the source cardinality, not the surviving cardinality: root
    // capture can compact geometrically dropped roots only at execution.
    if (ok && rootBindings &&
        (!UsdGenExecutionCheckedBytes::Multiply(curves, 48, &value) ||
         !UsdGenExecutionCheckedBytes::Add(&total, value)))
        return false;
    if (!ok)
        return false;
    for (auto const &plane : source.authoredPlanes) {
        // Domain accounting covers resampled point planes; with
        // resampleTo == 0 it equals the validated descriptor sizes.
        if (!CsourceVkAuthoredPlaneBytes(plane.domain, plane.arity, curves, points, &value) ||
            !UsdGenExecutionCheckedBytes::Add(&total, value))
            return false;
    }
    *bytes = total;
    return true;
}
bool ResampledEstimate(UsdGenCurveSetInputView const &source, bool rootBindings, uint32_t target,
                       uint64_t *bytes, uint64_t capacityCurves = 0) {
    // Upper bound for any Resample stage output packet. Curves never grow
    // downstream on this lane (no Grow/Scatter admitted), so every resample
    // emits at most sourceCurves * stageTarget points regardless of upstream
    // culls or earlier resamples.
    uint64_t curves = capacityCurves ? capacityCurves : source.curveVertexCounts.size(), points = 0,
             total = 0, value = 0;
    if (!UsdGenExecutionCheckedBytes::Multiply(curves, target, &points))
        return false;
    bool ok = UsdGenExecutionCheckedBytes::Multiply(points, 12, &value) &&
              UsdGenExecutionCheckedBytes::Add(&total, value) &&
              UsdGenExecutionCheckedBytes::Multiply(curves + 1, 4, &value) &&
              UsdGenExecutionCheckedBytes::Add(&total, value) &&
              UsdGenExecutionCheckedBytes::Multiply(curves, 8, &value) &&
              UsdGenExecutionCheckedBytes::Add(&total, value) &&
              UsdGenExecutionCheckedBytes::Multiply(points, 12, &value) &&
              UsdGenExecutionCheckedBytes::Add(&total, value);
    uint64_t widthBytes = 0;
    ok = ok && UsdGenExecutionCheckedBytes::Multiply(points, 4, &widthBytes) &&
         UsdGenExecutionCheckedBytes::Add(&total, widthBytes) &&
         UsdGenExecutionCheckedBytes::Add(&total, widthBytes);
    if (ok && rootBindings &&
        (!UsdGenExecutionCheckedBytes::Multiply(curves, 48, &value) ||
         !UsdGenExecutionCheckedBytes::Add(&total, value)))
        return false;
    if (!ok)
        return false;
    uint64_t namedTotal = 0;
    for (auto const &plane : source.authoredPlanes) {
        uint64_t scalars = plane.floatValues.size();
        if (!UsdGenExecutionCheckedBytes::Add(&scalars, plane.intValues.size()) ||
            !UsdGenExecutionCheckedBytes::Multiply(scalars, sizeof(uint32_t), &value) ||
            !UsdGenExecutionCheckedBytes::Add(&namedTotal, value))
            return false;
    }
    // Point-domain named bytes scale with the new cardinality; primitive and
    // groom bytes do not shrink, so scaling the whole total is conservative.
    uint64_t const sourcePoints = source.points.size();
    if (namedTotal && sourcePoints && points > sourcePoints) {
        uint64_t times = (points + sourcePoints - 1) / sourcePoints;
        if (!UsdGenExecutionCheckedBytes::Multiply(namedTotal, times, &namedTotal))
            return false;
    }
    if (!UsdGenExecutionCheckedBytes::Add(&total, namedTotal))
        return false;
    *bytes = total;
    return true;
}
} // namespace

VulkanSourceWidthPlan::VulkanSourceWidthPlan(std::shared_ptr<const UsdGenGraphSpec> descriptor,
                                             UsdGenCurveSetInputView source, UsdGenNodeKey sourcePath,
                                             UsdGenNodeKey widthPath, VulkanSourceControls sourceControls,
                                             VulkanLiteralWidthControls width, uint64_t generation,
                                             uint64_t topology, uint64_t value,
                                             bool rootBindings) noexcept
    : descriptor_(std::move(descriptor)), source_(std::move(source)),
      sourceNodeKey_(sourcePath), widthNodeKey_(widthPath),
      sourceControls_(std::move(sourceControls)), width_(width), curveGeneration_(generation),
      topologyVersion_(topology), valueVersion_(value), hasRootBindings_(rootBindings) {}

std::shared_ptr<const UsdGenExecutionPlanHandle>
CompileVulkanSourceWidthPlan(UsdGenGraphSpec const &original, UsdGenDiagnostics *diagnostics) {
    // Plan building is compile phase (synthetic path names intern).
    UsdGenCompileTokenScope tokenScope;
    UsdGenGraphSpec normalized;
    if (!NormalizeVulkanGeneratorGraph(original, &normalized, diagnostics))
        return {};
    auto const &desc = normalized;
    if (!desc.validationErrors.empty() || desc.executionBackend != UsdGenVulkanBackendId() ||
        desc.nodes.empty() || desc.nodes.size() >= size_t(UINT32_MAX) || UsdGenCurveSetInputs(desc).empty() ||
        !IsFiniteMatrix(desc.xformMatrix) || !IsFinite(desc.defaultWidth) || desc.defaultWidth < 0) {
        Fail(diagnostics, "requires a single-source Vulkan graph within the native index domain");
        return {};
    }
    using Stage = UsdGenVulkanStage;
    UsdGenVulkanTokens const &vk = UsdGenVkTokens();
    // Each node's Vulkan kernel row; the lane branches on its stage.
    std::vector<UsdGenVulkanKernel> kernels(desc.nodes.size());
    for (size_t i = 0; i < desc.nodes.size(); ++i)
        kernels[i] = UsdGenVulkanKernelOf(desc.nodes[i].type);
    size_t sourceIndex = desc.nodes.size(), terminalIndex = desc.nodes.size();
    std::set<UsdGenNodeKey> paths;
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        auto const &node = desc.nodes[i];
        if (!node.key.IsValid() || !paths.insert(node.key).second) {
            Fail(diagnostics, "node paths must be unique and nonempty");
            return {};
        }
        if (node.key == desc.terminal)
            terminalIndex = i;
    }
    if (!!desc.terminal.IsValid() && terminalIndex == desc.nodes.size()) {
        Fail(diagnostics, "missing terminal");
        return {};
    }
    auto indexFor = [&](UsdGenNodeKey path) {
        for (size_t i = 0; i < desc.nodes.size(); ++i)
            if (desc.nodes[i].key == path)
                return i;
        return desc.nodes.size();
    };
    std::vector<std::vector<size_t>> inputs(desc.nodes.size());
    std::vector<bool> consumed(desc.nodes.size(), false);
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        for (auto const &path : desc.nodes[i].inputs) {
            auto j = indexFor(path);
            if (j == desc.nodes.size() || j == i) {
                Fail(diagnostics, "invalid graph input");
                return {};
            }
            inputs[i].push_back(j);
            consumed[j] = true;
        }
    }
    if (!desc.terminal.IsValid()) {
        for (size_t i = 0; i < desc.nodes.size(); ++i) {
            if (consumed[i])
                continue;
            if (terminalIndex != desc.nodes.size()) {
                Fail(diagnostics, "graph with multiple sinks requires an explicit terminal");
                return {};
            }
            terminalIndex = i;
        }
        if (terminalIndex == desc.nodes.size()) {
            Fail(diagnostics, "graph has no terminal sink");
            return {};
        }
        // Retain the inferred selection for source capture and publication.
        normalized.terminal = desc.nodes[terminalIndex].key;
    }
    std::vector<bool> reachable(desc.nodes.size(), false);
    std::vector<size_t> pending{terminalIndex};
    while (!pending.empty()) {
        auto i = pending.back();
        pending.pop_back();
        if (reachable[i])
            continue;
        reachable[i] = true;
        pending.insert(pending.end(), inputs[i].begin(), inputs[i].end());
    }
    size_t reachableCount = 0;
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        if (!reachable[i])
            continue;
        ++reachableCount;
        if (kernels[i].Is(Stage::CurveSource)) {
            if (sourceIndex != desc.nodes.size()) {
                Fail(diagnostics, "selected terminal requires one source");
                return {};
            }
            sourceIndex = i;
        }
    }
    if (sourceIndex == desc.nodes.size()) {
        Fail(diagnostics, "missing selected source");
        return {};
    }
    UsdGenCurveSetInputView sourceView;
    UsdGenCurveSetInputView const *source = &sourceView;
    VulkanSourceControls sourceControls;
    auto const &sourceNode = desc.nodes[sourceIndex];
    if (!ValidateSource(desc, sourceNode, &sourceView, &sourceControls, diagnostics))
        return {};
    // Imported CurveSource planes can contain a baked native Clump quartet
    // without a Clump node in this graph. Vulkan Noise currently transports
    // those named channels but does not consume the motion weights, so a plan
    // would silently produce ungrouped field samples. ID-only legacy inputs
    // have no weight and retain the existing Noise behavior.
    bool const weightedClump = std::any_of(source->authoredPlanes.begin(),
        source->authoredPlanes.end(), [](auto const &plane) {
            return plane.name.GetString().rfind("clumpWeight_", 0) == 0;
        });
    bool reachableNoise = false;
    for (size_t i = 0; i < desc.nodes.size(); ++i)
        reachableNoise = reachableNoise ||
            (reachable[i] && kernels[i].Is(Stage::Noise));
    if (weightedClump && reachableNoise) {
        Fail(diagnostics, "Noise does not support imported clump motion planes");
        return {};
    }
    {
        std::vector<ExprVkStageBinding> bindings;
        std::vector<std::string> errors;
        if (!ExprVkCompileStageBindings(desc, sourceNode, &bindings, &errors)) {
            for (auto const &error : errors) {
                Fail(diagnostics, error);
            }
            return {};
        }
        UsdGenVulkanKernel const &cs = kernels[sourceIndex];
        for (auto const &b : bindings)
            if (cs.Names(vulkanP::kCsResampleTo, b.destination) ||
                cs.Names(vulkanP::kCsUsdGenResampleTo, b.destination))
                if (!sourceControls.resampleTo)
                    sourceControls.resampleTo = 2;
    }
    bool const hasRootBindings =
        (sourceControls.surfaceCage || !source->curveVertexCounts.empty()) &&
        (!sourceNode.surfaces.empty() || !source->skinPrim.empty() || !source->skinPrimUv.empty() ||
         !source->rootFrame.empty() || [&] {
             for (size_t i = 0; i < desc.nodes.size(); ++i)
                 if (reachable[i] && kernels[i].Is(Stage::Grow))
                     return true;
             return false;
         }());
    std::vector<uint32_t> nodeValue(desc.nodes.size(), UINT32_MAX);
    nodeValue[sourceIndex] = 0;
    std::vector<VulkanSourceWidthStage> steps;
    // An authored positive threshold after a fixed-topology Length lowers to
    // two native stages.  Keep its original descriptor ordinal separately:
    // the internal point transform has no semantic node, while the following
    // cull remains the one externally visible authored operator.
    std::vector<size_t> authoredIndices;
    std::vector<bool> internalStages;
    bool hasBlend = false;
    std::vector<bool> nodeIsDeform(desc.nodes.size(), false);
    size_t processedAuthored = 1;
    uint32_t syntheticPathOrdinal = 0;
    auto appendStage = [&](VulkanSourceWidthStage value, size_t authored, bool internal) {
        // Reserve the publication task and the UINT32_MAX missing-index sentinel.
        if (steps.size() >= size_t(UINT32_MAX) - 2)
            return false;
        // Job-private target plane names, interned here (compile phase).
        if (value.kind == VulkanSourceWidthStage::Kind::Deform)
            value.deform.rootTargetPlane =
                UsdGenInternToken("__usdGenDeformRoots" + std::to_string(steps.size()));
        else if (value.kind == VulkanSourceWidthStage::Kind::Grow && !value.emptyGenerator)
            value.growTargetPlane =
                UsdGenInternToken("__vulkanGrowTargets" + std::to_string(steps.size()));
        steps.push_back(std::move(value));
        authoredIndices.push_back(authored);
        internalStages.push_back(internal);
        return true;
    };
    auto uniqueTransformPath = [&](UsdGenNodeKey parent,
                                   char const *stem = "__vulkanLengthTransform") {
        // A synthetic child of the authored node, keyed like the host keys
        // <parent>/<stem>N; the ordinal advances past a taken key.
        UsdGenNodeKey key;
        do {
            key = normalized.DeriveKey(parent,
                UsdGenInternToken(std::string(stem) + std::to_string(syntheticPathOrdinal++)));
        } while (!paths.insert(key).second);
        return key;
    };
    // Source resampleTo lowers to an implicit leading internal Resample
    // stage (same internal-stage machinery as Length transforms): it
    // consumes value 0 and publishes value 1, so the source node alias
    // advances before the authored topo-sort runs.
    if (sourceControls.resampleTo > 0) {
        VulkanSourceWidthStage sourceResample;
        sourceResample.kind = VulkanSourceWidthStage::Kind::Resample;
        sourceResample.key = uniqueTransformPath(sourceNode.key, "__vulkanResampleTransform");
        sourceResample.input = 0;
        sourceResample.resampleTarget = uint32_t(sourceControls.resampleTo);
        if (!appendStage(std::move(sourceResample), sourceIndex, true)) {
            Fail(diagnostics, "expanded Vulkan stage count exceeds the native index domain");
            return {};
        }
        nodeValue[sourceIndex] = uint32_t(steps.size());
    }
    if (terminalIndex == sourceIndex && steps.empty()) {
        // The executor's legacy empty-stage shorthand means Width, so encode
        // source-only publication as an explicit internal alias. It submits
        // no operator work and preserves the source generation verbatim.
        VulkanSourceWidthStage alias;
        alias.disabled = true;
        alias.key = uniqueTransformPath(sourceNode.key, "__vulkanSourceAlias");
        if (!appendStage(std::move(alias), sourceIndex, true))
            return {};
    }
    while (processedAuthored < reachableCount) {
        size_t ready = desc.nodes.size();
        for (size_t i = 0; i < desc.nodes.size(); ++i) {
            if (!reachable[i] || nodeValue[i] != UINT32_MAX || inputs[i].empty())
                continue;
            bool available = true;
            for (auto j : inputs[i])
                if (nodeValue[j] == UINT32_MAX)
                    available = false;
            if (available) {
                ready = i;
                break;
            }
        }
        if (ready == desc.nodes.size()) {
            Fail(diagnostics, "cycle or unsupported additional root");
            return {};
        }
        auto const &node = desc.nodes[ready];
        UsdGenVulkanKernel const &kernel = kernels[ready];
        nodeIsDeform[ready] = nodeIsDeform[inputs[ready][0]];
        VulkanSourceWidthStage stage;
        stage.enabledLiteral = node.enabled;
        stage.key = node.key;
        stage.input = nodeValue[inputs[ready][0]];
        if (kernel.Is(Stage::Width)) {
            ExprVkWidthData widthData;
            std::vector<std::string> errors;
            if (inputs[ready].size() != 1 ||
                !ExprVkValidateWidthNode(desc, node, &widthData, &errors)) {
                for (auto const &error : errors) {
                    Fail(diagnostics, error);
                }
                return {};
            }
            stage.width = std::move(widthData.literal);
            stage.disabled = !node.enabled;
            if (ExprVkCompileNodeBindings(desc, node, true, true, &stage.expressionBindings,
                                          &errors) != ExprVkCompileStatus::Ok) {
                for (auto const &error : errors) {
                    Fail(diagnostics, error);
                }
                return {};
            }
            stage.expressionControls = ExprVkPlanControls(desc, node.seed, expr::Domain::Groom);
        } else if (kernel.Is(Stage::Length)) {
            stage.kind = VulkanSourceWidthStage::Kind::LengthScale;
            stage.randomSeed = node.seed;
            if (inputs[ready].size() != 1 || !ValidateLength(node, &stage, diagnostics))
                return {};
            stage.disabled = !node.enabled;
        } else if (kernel.Is(Stage::WidthBlend)) {
            stage.kind = VulkanSourceWidthStage::Kind::WidthBlend;
            bool paramsOk = true;
            float weight = 1.0f;
            for (auto const &param : node.params) {
                bool const ok = param.name == kernel.Param(vulkanP::kBlendWeight) && !param.animated &&
                                param.value.IsHolding<float>() &&
                                std::isfinite(param.value.UncheckedGet<float>()) &&
                                param.value.UncheckedGet<float>() >= 0.0f &&
                                param.value.UncheckedGet<float>() <= 1.0f;
                paramsOk = paramsOk && ok;
                if (ok)
                    weight = param.value.UncheckedGet<float>();
            }
            if (inputs[ready].size() != 2 || inputs[ready][0] == inputs[ready][1] ||
                !node.enabled || !paramsOk || !node.mode.IsEmpty() || !node.ramps.empty() ||
                !node.expressionBindings.empty() || !node.references.empty() ||
                !node.inputSets.empty() || !node.surfaces.empty() || !node.maps.empty() ||
                !node.mapBindings.empty()) {
                Fail(diagnostics, "WidthBlend requires two ordered inputs and literal blend");
                return {};
            }
            stage.rightInput = nodeValue[inputs[ready][1]];
            stage.blend = weight;
            hasBlend = true;
        } else if (kernel.Is(Stage::Noise)) {
            stage.kind = VulkanSourceWidthStage::Kind::Noise;
            stage.noise.seed = node.seed;
            stage.noise.enabled = node.enabled;
            if (inputs[ready].size() != 1 || !ValidateNoise(node, &stage.noise, diagnostics))
                return {};
            if (!hasRootBindings) {
                Fail(diagnostics, "Noise requires a root-bound source");
                return {};
            }
            stage.disabled = !node.enabled;
        } else if (kernel.Is(Stage::Deform)) {
            // CUDA currently requires identity description transforms only
            // for rest-to-pose Deform. Other graphs retain the description
            // transform for Session publication and shared source capture.
            if (!IsIdentity(desc.xformMatrix)) {
                Fail(diagnostics, "RBF currently requires an identity Description transform");
                return {};
            }
            UsdGenVulkanKernel const &cs = kernels[sourceIndex];
            if (!sourceControls.useRest &&
                !std::any_of(sourceNode.expressionBindings.begin(),
                             sourceNode.expressionBindings.end(), [&](auto const &b) {
                                 return cs.Names(vulkanP::kCsUseRest, b.destination) ||
                                        cs.Names(vulkanP::kCsUsdGenUseRest, b.destination);
                             })) {
                Fail(diagnostics, "already-deformed CurveSource cannot feed "
                                  "rest-to-animated RBF Deform");
                return {};
            }
            // Carry animated-space provenance through all intervening values:
            // a second rest-to-pose fit would apply surface motion twice.
            stage.kind = VulkanSourceWidthStage::Kind::Deform;
            if (inputs[ready].size() != 1)
                return {};
            if (nodeIsDeform[inputs[ready][0]]) {
                Fail(diagnostics, "a second rest-to-animated deformation would apply "
                                  "surface motion twice");
                return {};
            }
            if (sourceControls.surfaceCage) {
                // Dense roots are captured later. Fit planning depends only on
                // the pose surface, in the dense source's identity space.
                UsdGenCurveSetInputView fittingSource;   // identity worldMatrix
                if (!ValidateDeform(desc, node, &fittingSource, &stage.deform, diagnostics))
                    return {};
            } else if (!ValidateDeform(desc, node, source, &stage.deform, diagnostics))
                return {};
            stage.deform.rootStableIds.resize(
                sourceControls.surfaceCage ? 0 : source->curveVertexCounts.size());
            for (size_t c = 0; c < stage.deform.rootStableIds.size(); ++c)
                stage.deform.rootStableIds[c] =
                    sourceControls.idSource == vk.primvar && !source->curveId.empty()
                        ? source->curveId[c]
                        : uint64_t(c);
            stage.disabled = !node.enabled;
            nodeIsDeform[ready] = true;
        } else if (kernel.Is(Stage::Grow)) {
            stage.kind = VulkanSourceWidthStage::Kind::Grow;
            if (inputs[ready].size() != 1 ||
                !ValidateVulkanGrow(desc, node, &stage.grow, diagnostics))
                return {};
            // CUDA fuses Scatter->Grow into a generator, where a disabled
            // Grow empties the root set. Its ordinary C3 Grow operator still
            // executes the native controls independently of node.enabled.
            stage.emptyGenerator = !node.enabled && IsVulkanCapturedScatter(desc, *source);
            stage.disabled = false;
        } else if (kernel.Is(Stage::Resample)) {
            stage.kind = VulkanSourceWidthStage::Kind::Resample;
            if (inputs[ready].size() != 1 || !ValidateResample(node, &stage, diagnostics))
                return {};
            stage.disabled = !node.enabled;
        } else {
            Fail(diagnostics, "unsupported Vulkan operator");
            return {};
        }
        if (stage.kind != VulkanSourceWidthStage::Kind::Width && !node.expressionBindings.empty()) {
            std::vector<ExprVkStageBinding> checked;
            std::vector<std::string> errors;
            if (!ExprVkCompileStageBindings(desc, node, &checked, &errors) ||
                ExprVkCompileNodeBindings(desc, node, true, true, &stage.expressionBindings,
                                          &errors) != ExprVkCompileStatus::Ok) {
                for (auto const &error : errors) {
                    Fail(diagnostics, error);
                }
                return {};
            }
            stage.expressionControls = ExprVkPlanControls(desc, node.seed, expr::Domain::Groom);
        }
        for (auto const &binding : stage.expressionBindings)
            if (kernel.NamesEnabled(binding.binding.destination))
                stage.disabled = false;
        if ((stage.kind == VulkanSourceWidthStage::Kind::LengthScale ||
             stage.kind == VulkanSourceWidthStage::Kind::LengthCull) &&
            !stage.expressionBindings.empty()) {
            VulkanSourceWidthStage cull;
            cull.kind = VulkanSourceWidthStage::Kind::LengthCull;
            cull.disabled = stage.disabled;
            cull.enabledLiteral = stage.enabledLiteral;
            cull.lengthKeepInput = true;
            cull.key = stage.key;
            cull.input = uint32_t(steps.size() + 1);
            stage.lengthCullOnly = stage.kind == VulkanSourceWidthStage::Kind::LengthCull;
            stage.kind = VulkanSourceWidthStage::Kind::LengthScale;
            stage.lengthDeviceControls = true;
            stage.key = uniqueTransformPath(node.key);
            if (!appendStage(std::move(stage), ready, true) ||
                !appendStage(std::move(cull), ready, false)) {
                Fail(diagnostics, "expanded Vulkan stage count exceeds the native index domain");
                return {};
            }
        } else if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale &&
                   stage.cullThreshold > 0.0f) {
            // Do not ask compaction to reapply a scalar factor: it must
            // consume the actual scale/set transformed points so its native
            // length rounding is identical for both authored modes.
            VulkanSourceWidthStage cull;
            cull.kind = VulkanSourceWidthStage::Kind::LengthCull;
            cull.disabled = stage.disabled;
            cull.key = stage.key;
            cull.input = uint32_t(steps.size() + 1);
            cull.factor = stage.factor;
            cull.lengthMode = stage.lengthMode;
            cull.lengthMethod = stage.lengthMethod;
            cull.lengthRebuild = stage.lengthRebuild;
            cull.minRemainingLength = stage.minRemainingLength;
            cull.randomLo = stage.randomLo;
            cull.randomHi = stage.randomHi;
            cull.randomSeed = stage.randomSeed;
            cull.lengthBlend = stage.lengthBlend;
            cull.lengthMaskAmount = stage.lengthMaskAmount;
            cull.cullThreshold = stage.cullThreshold;
            stage.key = uniqueTransformPath(node.key);
            stage.cullThreshold = 0.0f;
            if (!appendStage(std::move(stage), ready, true) ||
                !appendStage(std::move(cull), ready, false)) {
                Fail(diagnostics, "expanded Vulkan stage count exceeds the native index domain");
                return {};
            }
        } else if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull &&
                   stage.lengthMaskAmount != 1.0f) {
            // Envelope phase 3 validates/preserves the fixed-topology
            // lineage before the authored topology owner performs culling.
            VulkanSourceWidthStage cull = stage;
            cull.input = uint32_t(steps.size() + 1);
            cull.lengthCullOnly = false;
            stage.kind = VulkanSourceWidthStage::Kind::LengthScale;
            stage.key = uniqueTransformPath(node.key);
            stage.cullThreshold = 0.0f;
            stage.lengthCullOnly = true;
            if (!appendStage(std::move(stage), ready, true) ||
                !appendStage(std::move(cull), ready, false)) {
                Fail(diagnostics, "expanded Vulkan stage count exceeds the native index domain");
                return {};
            }
        } else if (!appendStage(std::move(stage), ready, false)) {
            Fail(diagnostics, "expanded Vulkan stage count exceeds the native index domain");
            return {};
        }
        nodeValue[ready] = uint32_t(steps.size());
        ++processedAuthored;
    }
    if (authoredIndices.back() != terminalIndex) {
        Fail(diagnostics, "terminal must consume every authored node");
        return {};
    }
    uint64_t bytes = 0, widthBytes = 0, pointBytes = 0, peak = 0;
    uint64_t const sourceCurves =
        sourceControls.surfaceCage ? sourceControls.maxCurves : source->curveVertexCounts.size();
    uint64_t capturedPoints =
        sourceControls.surfaceCage ? sourceControls.maxCvs : source->points.size();
    // Source preparation keeps authored topology. Account the largest packet
    // any native topology stage can publish, including later Width/Length COW.
    for (auto const &stage : steps) {
        if (stage.disabled || stage.emptyGenerator)
            continue;
        uint32_t target = stage.kind == VulkanSourceWidthStage::Kind::Grow ? stage.grow.cvCount
                          : stage.kind == VulkanSourceWidthStage::Kind::Resample
                              ? stage.resampleTarget
                              : 0;
        uint64_t stagePoints = 0;
        if (target && !UsdGenExecutionCheckedBytes::Multiply(sourceCurves, target, &stagePoints)) {
            Fail(diagnostics, "payload estimate overflow");
            return {};
        }
        capturedPoints = std::max(capturedPoints, stagePoints);
    }
    if (!Estimate(*source, hasRootBindings, 0, &bytes, &widthBytes, sourceCurves, capturedPoints) ||
        !UsdGenExecutionCheckedBytes::Multiply(capturedPoints, 12, &pointBytes) ||
        !UsdGenExecutionCheckedBytes::Multiply(bytes, 2, &peak)) {
        Fail(diagnostics, "payload estimate overflow");
        return {};
    }
    // Targets become private immutable primitive planes during source upload;
    // account both staging and retained outputs before native admission.
    uint64_t privateRootBytes = 0;
    for (auto const &stage : steps) {
        if (stage.disabled || stage.kind != VulkanSourceWidthStage::Kind::Deform)
            continue;
        uint64_t rootBytes = 0;
        if (!UsdGenExecutionCheckedBytes::Multiply(sourceCurves, 12, &rootBytes) ||
            !UsdGenExecutionCheckedBytes::Add(&privateRootBytes, rootBytes) ||
            !UsdGenExecutionCheckedBytes::Add(&bytes, rootBytes) ||
            !UsdGenExecutionCheckedBytes::Add(&peak, rootBytes) ||
            !UsdGenExecutionCheckedBytes::Add(&peak, rootBytes)) {
            Fail(diagnostics, "root target estimate overflow");
            return {};
        }
    }
    auto capture = std::make_shared<UsdGenGraphSpec>(desc);
    auto payload = std::shared_ptr<VulkanSourceWidthPlan>(new VulkanSourceWidthPlan(
        capture, *source, sourceNode.key, desc.terminal, sourceControls,
        VulkanLiteralWidthControls{}, source->curveGeneration, source->curveGeneration,
        source->curveGeneration, hasRootBindings));
    payload->steps_ = steps;
    for (size_t i = 0; i < steps.size(); ++i) {
        auto const &stage = steps[i];
        if (stage.kind == VulkanSourceWidthStage::Kind::Width)
            payload->width_ = stage.width;
        if ((stage.kind == VulkanSourceWidthStage::Kind::LengthScale ||
             stage.kind == VulkanSourceWidthStage::Kind::LengthCull) &&
            !internalStages[i] && !payload->HasLength()) {
            payload->lengthNodeKey_ = stage.key;
            payload->lengthFactor_ = stage.factor;
        }
    }
    uint32_t const finalTask = uint32_t(steps.size());
    using K = UsdGenExecutionDataKind;
    using A = UsdGenExecutionResourceAccess;
    using S = UsdGenExecutionValueStorage;
    std::vector<UsdGenExecutionValueMetadata> values{
        {0, K::ElementGeometry, UINT32_MAX, 0, S::ExternalImmutable},
        {1, K::ElementGeometry, 0, 1, S::JobOwnedImmutable},
        {2, K::Widths, 0, 1, S::JobOwnedImmutable},
        {3, K::Widths, finalTask, finalTask + 1, S::PublishedImmutable},
        {4, K::ElementTopology, 0, 1, S::JobOwnedImmutable},
        {5, K::StableIds, 0, 1, S::JobOwnedImmutable},
        // One immutable source bundle: hairT plus every authored named plane.
        // Point-only stages retain/read it; topology-changing stages publish a
        // gathered replacement bundle for their downstream lineage.
        {6, K::NamedChannels, 0, 1, S::JobOwnedImmutable}};
    // Preserve the historical ids above.  Root bindings are a separate
    // structural bundle (rootPrim, rootUV, private T/B/N) and are appended
    // only when source capture can produce a non-empty bundle.
    std::vector<uint32_t> sourceImmutableIds{4u, 5u, 6u};
    if (hasRootBindings) {
        values.push_back({uint32_t(values.size()), K::RootBindings, 0, 1, S::JobOwnedImmutable});
        sourceImmutableIds.push_back(uint32_t(values.size() - 1));
    }
    // MapVk (phase 2 maps): CUDA carries desc.maps as a lazily-created
    // ImageMaps external value, materialized only when a task reads it
    // (cudaExecution.cpp externalValue()). Only Width admits bindings on
    // this lane, so the slot exists exactly when a Width binding does.
    // Historical ids above are never renumbered.
    bool needsImageMaps = false;
    for (auto const &node : desc.nodes)
        needsImageMaps = needsImageMaps || !node.mapBindings.empty();
    uint32_t const imageMapsValueId = uint32_t(values.size());
    if (needsImageMaps)
        values.push_back({imageMapsValueId, K::ImageMaps, UINT32_MAX, 0, S::ExternalImmutable});
    bool needsParameters = false;
    for (size_t i = 0; i < desc.nodes.size(); ++i)
        if (reachable[i] && !desc.nodes[i].expressionBindings.empty())
            needsParameters = true;
    uint32_t const parameterValueId = uint32_t(values.size());
    if (needsParameters)
        values.push_back(
            {parameterValueId, K::ParameterValues, UINT32_MAX, 0, S::ExternalImmutable});
    struct Bundle {
        uint32_t geometry, widths;
        std::vector<uint32_t> inherited;
    };
    std::vector<Bundle> bundles{{1, 2, sourceImmutableIds}};
    std::vector<UsdGenExecutionTaskMetadata> tasks;
    UsdGenExecutionTaskMetadata sourceTask;
    sourceTask.id = 0;
    sourceTask.semanticNode = uint32_t(sourceIndex);
    sourceTask.authoredOrderKey = uint32_t(sourceIndex);
    sourceTask.kind = UsdGenExecutionTaskKind::Source;
    sourceTask.key = sourceNode.key;
    sourceTask.type = original.nodes[sourceIndex].type;
    sourceTask.exclusiveWorkspace = false;
    sourceTask.resources = {{K::ElementGeometry, A::ReadWrite, UINT32_MAX, 0, 1},
                            {K::Widths, A::Write, UINT32_MAX, UINT32_MAX, 2}};
    for (uint32_t id : sourceImmutableIds)
        sourceTask.resources.push_back({values[id].resource, A::Write, UINT32_MAX, UINT32_MAX, id});
    if (!sourceNode.expressionBindings.empty())
        sourceTask.resources.push_back(
            {K::ParameterValues, A::Read, UINT32_MAX, parameterValueId, UINT32_MAX});
    sourceTask.estimate = {0, bytes, 0, 0, false, false, false, 0};
    tasks.push_back(std::move(sourceTask));
    std::vector<UsdGenCompiledOperatorCapability> operators{
        {sourceNode.key, original.nodes[sourceIndex].type, uint32_t(sourceIndex), 1,
         uint32_t(UsdGenCapabilityCopyOnWriteWrites |
                  (sourceNode.expressionBindings.empty() ? UsdGenCapabilityNone
                                                         : UsdGenCapabilityExpressions)),
         UsdGenExecutionCapabilityStatus::Supported}};
    for (size_t i = 0; i < steps.size(); ++i) {
        auto &stage = steps[i];
        // A topology writer can append geometry, widths and the four
        // inherited structural bundles. Check before any uint32 narrowing.
        if (values.size() > size_t(UINT32_MAX) - 6) {
            Fail(diagnostics, "Vulkan value count exceeds the native index domain");
            return {};
        }
        uint32_t taskId = uint32_t(i + 1);
        auto in = bundles[stage.input], out = in;
        UsdGenExecutionTaskMetadata task;
        auto const &authored = desc.nodes[authoredIndices[i]];
        task.id = taskId;
        task.semanticNode = internalStages[i] ? UINT32_MAX : uint32_t(authoredIndices[i]);
        task.authoredOrderKey = uint32_t(authoredIndices[i]);
        task.kind = UsdGenExecutionTaskKind::Operator;
        task.key = stage.key;
        task.type = !internalStages[i] ? authored.type
                    : stage.kind == VulkanSourceWidthStage::Kind::Resample
                        ? vk.resampleTransform
                    : stage.kind == VulkanSourceWidthStage::Kind::Width
                        ? vk.sourceAlias
                        : vk.lengthTransform;
        task.exclusiveWorkspace = false;
        bool const cull = stage.kind == VulkanSourceWidthStage::Kind::LengthCull;
        bool const resample = stage.kind == VulkanSourceWidthStage::Kind::Resample ||
                              stage.kind == VulkanSourceWidthStage::Kind::Grow;
        bool const length = stage.kind == VulkanSourceWidthStage::Kind::LengthScale || cull ||
                            resample || stage.kind == VulkanSourceWidthStage::Kind::Noise ||
                            stage.kind == VulkanSourceWidthStage::Kind::Deform;
        // Statically muted modifiers alias their input topology. Their value
        // revision still advances, but publication must retain the actual
        // topology producer instead of assigning an unwritten revision.
        bool const topologyChange = (cull || resample) && !stage.disabled;
        task.topologyBarrier = topologyChange;
        uint32_t output = taskId == finalTask ? 3u : uint32_t(values.size());
        if (output != 3)
            values.push_back({});
        values[output] = {output, length ? K::ElementGeometry : K::Widths, taskId, taskId + 1,
                          output == 3 ? S::PublishedImmutable : S::JobOwnedImmutable};
        if (length)
            out.geometry = output;
        else
            out.widths = output;
        task.resources.push_back({K::ElementGeometry, length ? A::ReadWrite : A::Read,
                                  values[in.geometry].producerTask, in.geometry,
                                  length ? output : UINT32_MAX});
        if (topologyChange) {
            out.widths = uint32_t(values.size());
            values.push_back({out.widths, K::Widths, taskId, taskId + 1, S::JobOwnedImmutable});
        }
        task.resources.push_back({K::Widths, length && !topologyChange ? A::Read : A::ReadWrite,
                                  values[in.widths].producerTask, in.widths,
                                  topologyChange ? out.widths
                                  : length           ? UINT32_MAX
                                                     : output});
        if (!stage.expressionBindings.empty())
            task.resources.push_back(
                {K::ParameterValues, A::Read, UINT32_MAX, parameterValueId, UINT32_MAX});
        // MapVk: CUDA declares an ImageMaps read for Width nodes with
        // bindings (cudaExecution.cpp); legacy node.maps alone add no edge.
        if (stage.kind == VulkanSourceWidthStage::Kind::Width && !authored.mapBindings.empty())
            task.resources.push_back(
                {K::ImageMaps, A::Read, UINT32_MAX, imageMapsValueId, UINT32_MAX});
        if (stage.kind == VulkanSourceWidthStage::Kind::WidthBlend) {
            auto right = bundles[stage.rightInput];
            stage.requiresNonWidthProof = right.geometry != in.geometry;
            if (stage.requiresNonWidthProof) {
                task.resources.push_back({K::ElementGeometry, A::Read,
                                          values[right.geometry].producerTask, right.geometry,
                                          UINT32_MAX});
                for (uint32_t id : right.inherited)
                    task.resources.push_back(
                        {values[id].resource, A::Read, values[id].producerTask, id, UINT32_MAX});
            }
            task.resources.push_back(
                {K::Widths, A::Read, values[right.widths].producerTask, right.widths, UINT32_MAX});
        }
        for (size_t j = 0; j < in.inherited.size(); ++j) {
            uint32_t const id = in.inherited[j];
            uint32_t replacement = UINT32_MAX;
            if (topologyChange) {
                replacement = uint32_t(values.size());
                values.push_back(
                    {replacement, values[id].resource, taskId, taskId + 1, S::JobOwnedImmutable});
                out.inherited[j] = replacement;
            }
            task.resources.push_back({values[id].resource,
                                      topologyChange ? A::ReadWrite : A::Read,
                                      values[id].producerTask, id, replacement});
        }
        uint64_t delta = cull ? bytes : length ? pointBytes : widthBytes;
        if (resample && !ResampledEstimate(*source, hasRootBindings,
                                           (stage.kind == VulkanSourceWidthStage::Kind::Grow
                                                ? stage.grow.cvCount
                                                : stage.resampleTarget),
                                           &delta, sourceControls.maxCurves)) {
            Fail(diagnostics, "resample payload estimate overflow");
            return {};
        }
        if (resample && !UsdGenExecutionCheckedBytes::Add(&delta, privateRootBytes)) {
            Fail(diagnostics, "resample root target estimate overflow");
            return {};
        }
        // The comparison packs both non-width packets with per-plane padding
        // and a scalar status. This remains a logical estimate, not a Vulkan
        // allocation-requirements bound; the pool enforces actual admission.
        uint64_t proofScratch = 0;
        if (stage.requiresNonWidthProof) {
            proofScratch = bytes;
            if (!UsdGenExecutionCheckedBytes::Add(&proofScratch, bytes) ||
                !UsdGenExecutionCheckedBytes::Add(&proofScratch, 64)) {
                Fail(diagnostics, "DAG comparison estimate overflow");
                return {};
            }
        }
        if (!UsdGenExecutionCheckedBytes::Add(&peak, delta)) {
            Fail(diagnostics, "DAG delta estimate overflow");
            return {};
        }
        if (!UsdGenExecutionCheckedBytes::Add(&peak, proofScratch)) {
            Fail(diagnostics, "DAG proof estimate overflow");
            return {};
        }
        task.estimate = {0, delta, bytes, proofScratch, false, false, false, 0};
        tasks.push_back(std::move(task));
        bundles.push_back(out);
        if (!internalStages[i])
            operators.push_back(
                {stage.key, authored.type, uint32_t(authoredIndices[i]), 1,
                 uint32_t(UsdGenCapabilityCopyOnWriteWrites |
                          (authored.expressionBindings.empty() ? UsdGenCapabilityNone
                                                               : UsdGenCapabilityExpressions)),
                 UsdGenExecutionCapabilityStatus::Supported});
    }
    UsdGenExecutionTaskMetadata publication;
    publication.id = finalTask + 1;
    publication.kind = UsdGenExecutionTaskKind::Publication;
    publication.type = vk.publication;
    publication.exclusiveWorkspace = false;
    auto terminal = bundles.back();
    publication.resources = {
        {K::ElementGeometry, A::Read, values[terminal.geometry].producerTask, terminal.geometry,
         UINT32_MAX},
        {K::Widths, A::Read, values[terminal.widths].producerTask, terminal.widths, UINT32_MAX}};
    for (uint32_t id : terminal.inherited)
        publication.resources.push_back(
            {values[id].resource, A::Read, values[id].producerTask, id, UINT32_MAX});
    for (uint32_t i = 0; i <= finalTask; ++i)
        publication.declaredDependencies.push_back(
            {i, UsdGenExecutionDependencyLifetimePublicationJoin});
    tasks.push_back(std::move(publication));
    payload->steps_ = steps;
    std::string reason;
    if (!UsdGenExecutionDependencyCompiler::Lower(&tasks, values, &reason)) {
        Fail(diagnostics, reason);
        return {};
    }
    UsdGenExecutionGraphMemoryEstimate memory{peak, bytes, false, false, false};
    auto metadata = std::make_shared<const UsdGenExecutionPlanMetadata>(
        "vulkan", 1,
        hasBlend ? UsdGenExecutionPlanShape::SourceRootedValueDag
                 : UsdGenExecutionPlanShape::LinearAuthoredChain,
        std::move(operators), std::move(tasks), std::move(values), finalTask + 1, memory);
    auto handle = UsdGenExecutionPlanHandle::Create(UsdGenVulkanBackendId(), metadata,
                                                    payload, &reason);
    if (!handle)
        Fail(diagnostics, reason);
    return handle;
}
} // namespace usdGen::vulkan
