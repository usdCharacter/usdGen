// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// csourceVkAdmission.h — Phase-2 CurveSource gap (csourceVk) admission predicates.
//
// Header-only so the plan-compiler fragment (applied by the phase-2
// integrator) and the parity test share one implementation. Every predicate
// mirrors an established CUDA/CPU contract cited below; Vulkan admits exactly
// what the CUDA twin executes and keeps rejecting what CUDA rejects.
#ifndef USDGEN_VULKAN_CSOURCE_VK_ADMISSION_H
#define USDGEN_VULKAN_CSOURCE_VK_ADMISSION_H

#include "usdGen/authoredNamedChannels.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenCurves/vulkan/vkStages.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace usdGen::vulkan {

// resampleTo validation. CUDA contract: 0 keeps the authored topology,
// >= 2 resamples every curve to a uniform count; negative values and 1
// reject (curveLoader.cpp ReadOptions, cudaSourceInput.cpp preparation,
// CudaCurveResample::Apply). Output cardinality is bounded by uint32/int
// (curveLoader.cpp Load, CudaCurveResample INT32_MAX rejection).
enum class CsourceVkResampleStatus { Ok, NotAnInteger, Negative, OneCV, Overflow };

inline CsourceVkResampleStatus CsourceVkValidateResampleTarget(
    VtValue const& value, size_t curveCount, int* target,
    std::string* reason = nullptr) {
    auto fail = [&](CsourceVkResampleStatus code, char const* text) {
        if (reason) *reason = text;
        return code;
    };
    if (!value.IsHolding<int>())
        return fail(CsourceVkResampleStatus::NotAnInteger,
                    "CurveSource parameter resampleTo must be int");
    int const t = value.UncheckedGet<int>();
    if (target) *target = t;
    if (t < 0 || t == 1)
        return fail(t < 0 ? CsourceVkResampleStatus::Negative
                          : CsourceVkResampleStatus::OneCV,
                    "CurveSource resampleTo must be zero or at least two");
    if (t > 0) {
        if (curveCount > size_t(std::numeric_limits<uint32_t>::max()))
            return fail(CsourceVkResampleStatus::Overflow,
                        "CurveSource resampleTo output exceeds uint32 cardinality");
        uint64_t const out = uint64_t(curveCount) * uint64_t(t);
        if (out > uint64_t(std::numeric_limits<uint32_t>::max()) ||
            out > uint64_t(std::numeric_limits<int>::max()))
            return fail(CsourceVkResampleStatus::Overflow,
                        "CurveSource resampleTo output exceeds uint32 cardinality");
    }
    return CsourceVkResampleStatus::Ok;
}

// Captured CV count under a validated resample target: curves * target, or
// the authored count when target == 0. The plan memory estimate must use
// this, not the authored count, so upsampling never under-reserves.
inline bool CsourceVkEffectivePointCount(uint64_t authoredPoints,
                                         size_t curveCount, int resampleTo,
                                         uint64_t* effective) {
    if (!effective || resampleTo < 0 || resampleTo == 1) return false;
    if (resampleTo == 0) {
        *effective = authoredPoints;
        return true;
    }
    if (curveCount > size_t(std::numeric_limits<uint32_t>::max()))
        return false;
    uint64_t const out = uint64_t(curveCount) * uint64_t(resampleTo);
    if (out > uint64_t(std::numeric_limits<uint32_t>::max()) ||
        out > uint64_t(std::numeric_limits<int>::max()))
        return false;
    *effective = out;
    return true;
}

// Authored-plane payload bytes under resample. Point-domain planes scale
// with the effective (possibly resampled) CV count; Primitive/Groom planes
// are untouched by resampling. With resampleTo == 0 this equals the
// descriptor byte count (enforced by ValidateAuthoredNamedChannels), so the
// plan estimate can use domain accounting unconditionally.
inline bool CsourceVkAuthoredPlaneBytes(UsdGenAuthoredPlaneDomain domain,
                                        uint32_t arity, uint64_t curveCount,
                                        uint64_t effectivePoints,
                                        uint64_t* bytes) {
    if (!bytes || arity == 0 || arity > 16) return false;
    uint64_t elements = 0;
    switch (domain) {
    case UsdGenAuthoredPlaneDomain::Point: elements = effectivePoints; break;
    case UsdGenAuthoredPlaneDomain::Primitive: elements = curveCount; break;
    case UsdGenAuthoredPlaneDomain::Groom: elements = 1; break;
    default: return false;
    }
    if (elements > uint64_t(std::numeric_limits<size_t>::max()) / arity ||
        elements * arity >
            uint64_t(std::numeric_limits<size_t>::max()) / sizeof(uint32_t))
        return false;
    *bytes = elements * arity * sizeof(uint32_t);
    return true;
}

// Surface-free rebind admission. Outcome table of the shared capture used by
// both backends (curveRootCapture.cpp): rebind=never always admits (the
// absence-preserving path); rebind=onError executes exactly when complete
// authored frames make repair unnecessary; rebind=always needs a rest
// surface for its unconditional repair and never executes surface-free.
inline bool CsourceVkSurfaceFreeRebindAdmitted(TfToken const& rebind,
                                              size_t curveCount,
                                              size_t rootFrameCount) {
    if (curveCount == 0) return true;
    UsdGenVulkanTokens const &t = UsdGenVkTokens();
    if (rebind == t.never) return true;
    if (rebind == t.onError) return rootFrameCount == curveCount;
    return false;
}

// Deform transform admission (CUDA-rule predicate; header-level only, not wired
// into the Vulkan plan). CUDA requires identity Description and CurveSource
// transforms for RBF Deform: ValidateCudaGraph never consults the pose surface
// transform (cudaExecution.cpp), but CUDA execution rejects moved surfaces in
// PrepareCudaSurface. Vulkan instead executes moved source/surface configs via
// the surfaceVk host fit (in-budget) or the rbf device path (out-of-budget).
// The plan compiler enforces description-transform identity on Deform
// stages; other graph shapes retain arbitrary finite publication transforms.
inline bool CsourceVkDeformSourceAdmitted(GfMatrix4d const& sourceWorld,
                                          std::string* reason = nullptr) {
    if (sourceWorld == GfMatrix4d(1.0)) return true;
    if (reason) *reason = "Deform requires identity source transforms";
    return false;
}

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_CSOURCE_VK_ADMISSION_H
