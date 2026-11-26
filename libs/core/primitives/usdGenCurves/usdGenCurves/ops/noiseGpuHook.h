// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// GPU path for UsdGenNoiseOp::Capture. usdGenCurvesCuda sets it at
// registration (USE_GPU_NOISE builds); unset, Capture runs on the CPU.
#ifndef USDGEN_CURVES_OPS_NOISE_GPU_HOOK_H
#define USDGEN_CURVES_OPS_NOISE_GPU_HOOK_H

#include "usdGenCurves/export.h"

namespace usdGen {

// Returns false to fall back to the CPU loop. Uniform CV stride only.
using UsdGenNoiseCaptureGpuFn = bool (*)(float const *base,
                                         float const *cvT,
                                         float frequency,
                                         int octaves,
                                         float lacunarity,
                                         float gain,
                                         int nCurves,
                                         int nCV,
                                         float *field,
                                         float *perCurve);

USDGEN_CURVES_API void UsdGenSetNoiseCaptureGpu(UsdGenNoiseCaptureGpuFn fn) noexcept;

}  // namespace usdGen

#endif  // USDGEN_CURVES_OPS_NOISE_GPU_HOOK_H
