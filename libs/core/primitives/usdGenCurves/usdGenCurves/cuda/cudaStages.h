// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// CUDA lane stages and backend parameter indices. No CUDA SDK types: the
// kernel rows are registered (ops/registerCurvesCuda.cpp) in every build.
#pragma once
#include "usdGen/registries/backendRegistry.h"
#include "usdGen/registries/handles.h"
#include "usdGenCuda/backendId.h"

#include <cstdint>

namespace usdGen {

class UsdGenRegistrar;
struct UsdGenOpNode;

/// One per CUDA capability row; UsdGenKernelDef::stage.
enum class UsdGenCudaStage : uint16_t {
    Scatter, Grow, CurveSource, Width, Length, Noise, Deform, ReferenceSource, WidthBlend
};

namespace cudaP {
/// Backend parameter indices; UsdGenKernelDef::paramMap maps them to op
/// slots. One enum per stage, listing the names the CUDA lane reads.
enum Scatter : uint16_t { kScatterFlip, kScatterCount };
enum Grow : uint16_t {
    kGrowSegments, kGrowLength, kGrowLengthRandom, kGrowLift, kGrowAzimuth,
    kGrowAzimuthRandom, kGrowDirection, kGrowDirectionVector, kGrowCount
};
enum CurveSource : uint16_t {
    kSrcUseRest, kSrcResampleTo, kSrcInterpolationMode, kSrcRegionMapChannel,
    kSrcIdSource, kSrcStaleAction, kSrcExpectEpoch, kSrcRebind, kSrcCount
};
enum Width : uint16_t {
    kWidthWidth, kWidthMask, kWidthEnabled, kWidthReplace, kWidthKnots,
    kWidthInterpolation, kWidthCount
};
enum Length : uint16_t {
    kLenMode, kLenMethod, kLenRebuild, kLenValue, kLenRandom, kLenMinRemaining,
    kLenCull, kLenMask, kLenEnabled, kLenCount
};
enum Noise : uint16_t {
    kNoiseMagnitude, kNoiseFrequency, kNoiseCorrelation, kNoiseOctaves,
    kNoiseLacunarity, kNoiseGain, kNoisePreserveLength, kNoiseMask, kNoiseEnabled,
    kNoiseCumulative, kNoiseSeed, kNoiseMagKnots, kNoiseMagInterpolation, kNoiseCount
};
enum Deform : uint16_t {
    kDeformRbfSamples, kDeformMask, kDeformEnabled, kDeformLockRoots, kDeformCount
};
enum WidthBlend : uint16_t { kBlendWeight, kBlendCount };
}  // namespace cudaP

/// The ops behind the CUDA rows, as their register functions returned them.
struct UsdGenCudaKernelOps {
    UsdGenOpId scatter = UsdGenOpId::Invalid, grow = UsdGenOpId::Invalid,
        curveSource = UsdGenOpId::Invalid, width = UsdGenOpId::Invalid,
        length = UsdGenOpId::Invalid, noise = UsdGenOpId::Invalid,
        deform = UsdGenOpId::Invalid, referenceSource = UsdGenOpId::Invalid,
        widthBlend = UsdGenOpId::Invalid;
};

/// Registers the CUDA kernel rows, in capability-matrix order.
void UsdGenRegisterCurvesCuda(UsdGenRegistrar &, UsdGenCudaKernelOps const &);

/// The CUDA row of an op or of a node's op; null when CUDA cannot run it.
/// Never interns.
UsdGenKernelDef const *UsdGenCudaKernel(UsdGenOpId) noexcept;
UsdGenKernelDef const *UsdGenCudaKernel(UsdGenOpNode const &) noexcept;
bool UsdGenIsCudaStage(UsdGenOpNode const &, UsdGenCudaStage) noexcept;
/// The op slot behind a backend parameter index; kUsdGenNoParam if none.
UsdGenParamSlot UsdGenCudaSlot(UsdGenKernelDef const *, uint16_t index) noexcept;

/// The stage each cudaP enum indexes.
template <class E> struct UsdGenCudaStageOf;
#define USDGEN_CUDA_STAGE_OF(E, S) \
    template <> struct UsdGenCudaStageOf<cudaP::E> { \
        static constexpr UsdGenCudaStage value = UsdGenCudaStage::S; }
USDGEN_CUDA_STAGE_OF(Scatter, Scatter);
USDGEN_CUDA_STAGE_OF(Grow, Grow);
USDGEN_CUDA_STAGE_OF(CurveSource, CurveSource);
USDGEN_CUDA_STAGE_OF(Width, Width);
USDGEN_CUDA_STAGE_OF(Length, Length);
USDGEN_CUDA_STAGE_OF(Noise, Noise);
USDGEN_CUDA_STAGE_OF(Deform, Deform);
USDGEN_CUDA_STAGE_OF(WidthBlend, WidthBlend);
#undef USDGEN_CUDA_STAGE_OF

/// As UsdGenCudaSlot, and kUsdGenNoParam unless the kernel is E's stage.
template <class E>
UsdGenParamSlot UsdGenCudaStageSlot(UsdGenKernelDef const *kernel, E index) noexcept
{
    return kernel && kernel->stage == uint16_t(UsdGenCudaStageOf<E>::value)
        ? UsdGenCudaSlot(kernel, uint16_t(index)) : kUsdGenNoParam;
}

}  // namespace usdGen
