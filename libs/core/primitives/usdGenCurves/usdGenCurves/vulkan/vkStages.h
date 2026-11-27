// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Vulkan kernel rows: the backend stage of each op the lane lowers, backend
// parameter indices (UsdGenKernelDef::paramMap) and the lane's vocabulary.
// No Vulkan SDK types. Defined in usdGenCurvesVulkan (registerCurvesVulkan.cpp).
#pragma once
#include "usdGen/api/token.h"
#include "usdGenCurves/vulkan/export.h"
#include "usdGen/api/ext/backendRegistry.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/token.h"

#include <cstdint>

namespace usdGen {

class UsdGenRegistrar;

/// UsdGenKernelDef::stage on the Vulkan backend: the op family a descriptor
/// node lowers to (VulkanSourceWidthStage::Kind is the lowered form).
enum class UsdGenVulkanStage : uint16_t {
    CurveSource, ReferenceSource, Scatter, Grow, Width, Length, WidthBlend, Noise, Deform,
    Resample
};

namespace vulkanP {
/// Backend parameter indices; paramMap maps them to op slots. One enum per
/// stage listing the names the lane compares as tokens.
enum CurveSource : uint16_t { kCsUseRest, kCsUsdGenUseRest, kCsResampleTo, kCsUsdGenResampleTo,
                              kCsRebind, kCsInterpolationMode, kCsCount };
enum Width : uint16_t { kWidthProfileMode, kWidthCount };
enum Length : uint16_t { kLenMode, kLenCount };
enum WidthBlend : uint16_t { kBlendWeight, kBlendCount };
enum Grow : uint16_t { kGrowSegments, kGrowLength, kGrowLengthRandom, kGrowLift, kGrowAzimuth,
                       kGrowAzimuthRandom, kGrowDirection, kGrowDirectionVector, kGrowCount };
enum Deform : uint16_t { kDeformRbfSamples, kDeformCount };
enum Scatter : uint16_t { kScDensity, kScFlip, kScSubdivisionLevel, kScCount };
}  // namespace vulkanP

/// Values and names the lane compares that no op declares as data. Interned
/// once (compile scope); valid for the life of the process.
struct UsdGenVulkanTokens {
    // CurveSource idSource / staleAction / rebind values
    UsdGenToken primvar, index, warn, ignore, block, never, onError, always;
    // other ops' literal values the lane admits
    UsdGenToken surfaceCage, scale, set, cull, cutExtend, keepParam, reparam, surfaceNormal, vector;
    // lane vocabulary
    UsdGenToken regionMapRel /* usdGen:regionMap */, exprSource /* expr:source */,
        capturedRoots /* __vulkanCapturedRoots */, owners /* __vulkanOwners */,
        resampleTransform /* VulkanResampleTransform */, sourceAlias /* VulkanSourceAlias */,
        lengthTransform /* VulkanLengthTransform */, publication /* VulkanPublication */;
};

USDGEN_CURVES_VULKAN_API UsdGenVulkanTokens const &UsdGenVkTokens() noexcept;

/// One op's Vulkan kernel row; empty when the op has none.
struct UsdGenVulkanKernel {
    UsdGenOpRecord const *record = nullptr;
    UsdGenKernelDef const *kernel = nullptr;

    explicit operator bool() const noexcept { return kernel != nullptr; }
    bool Is(UsdGenVulkanStage s) const noexcept {
        return kernel && kernel->stage == uint16_t(s);
    }
    UsdGenParamSlot Slot(uint16_t k) const noexcept { return kernel->paramMap[k]; }
    /// The declared spelling of backend parameter k.
    UsdGenToken const &Param(uint16_t k) const noexcept { return record->ParamName(Slot(k)); }
    /// name spells backend parameter k as FindParam accepts it ("x" or
    /// "usdGen:x" unless the op declares "usdGen:x" separately).
    bool Names(uint16_t k, UsdGenToken const &name) const noexcept {
        return record->FindParam(name) == Slot(k);
    }
    /// name spells the op's enabled parameter, with or without "usdGen:".
    bool NamesEnabled(UsdGenToken const &name) const noexcept {
        UsdGenParamSlot const s = record->EnabledSlot();
        return s != kUsdGenNoParam && record->FindParam(name) == s;
    }
};

/// The Vulkan kernel of the op named type. Pointer probes; never interns.
USDGEN_CURVES_VULKAN_API UsdGenVulkanKernel UsdGenVulkanKernelOf(UsdGenToken const &type) noexcept;
/// The first op whose Vulkan kernel has stage s.
USDGEN_CURVES_VULKAN_API UsdGenVulkanKernel UsdGenVulkanKernelFor(UsdGenVulkanStage s) noexcept;

/// Op ids the Vulkan kernels attach to (register.cpp).
struct UsdGenVulkanKernelOps {
    UsdGenOpId curveSource = UsdGenOpId::Invalid, referenceSource = UsdGenOpId::Invalid,
               scatter = UsdGenOpId::Invalid, grow = UsdGenOpId::Invalid,
               width = UsdGenOpId::Invalid, length = UsdGenOpId::Invalid,
               widthBlend = UsdGenOpId::Invalid, noise = UsdGenOpId::Invalid,
               deform = UsdGenOpId::Invalid, resample = UsdGenOpId::Invalid;
};
void UsdGenRegisterVulkanKernels(UsdGenRegistrar &, UsdGenVulkanKernelOps const &);

}  // namespace usdGen
