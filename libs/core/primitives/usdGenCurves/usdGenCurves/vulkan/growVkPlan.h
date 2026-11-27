// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_GROW_VK_PLAN_H
#define USDGEN_VULKAN_GROW_VK_PLAN_H
#include "usdGenCurves/vulkan/curveGrowPipeline.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/op.h"
namespace usdGen::vulkan {
// Lower stage-free Scatter root capture and ReferenceSource C3 import to the
// existing immutable source packet. This never evaluates Grow on the CPU.
// Original source type/path remains available to the caller for capability
// metadata; native source decoding consumes the returned CurveSource shape.
bool NormalizeVulkanGeneratorGraph(UsdGenGraphSpec const&, UsdGenGraphSpec*,
    UsdGenDiagnostics* = nullptr);
bool ValidateVulkanGrow(UsdGenGraphSpec const&, UsdGenOpNode const&,
    CurveGrowPipeline::Controls*, UsdGenDiagnostics* = nullptr);
// The captured C3 decoder deliberately permits absent root channels. Grow
// requires a full private frame trio; this matches CUDA's source fallback.
void CompleteVulkanGrowSource(UsdGenCurveBuffer*);
bool IsVulkanCapturedScatter(UsdGenGraphSpec const&, UsdGenCurveSetInputView const&) noexcept;
bool CaptureVulkanScatterSource(UsdGenGraphSpec const&, UsdGenCurveSetInputView const&,
    uint64_t topologyVersion,
    uint64_t valueVersion, UsdGenCurveBuffer*);
} // namespace usdGen::vulkan
#endif
