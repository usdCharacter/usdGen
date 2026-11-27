// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// CUDA kernel rows for the built-in curve ops (usdGenCurvesCuda).
// Registered in every build: the capability matrix reads them.
#include "usdGenCurves/cuda/cudaStages.h"

#include "usdGen/api/ext/device/executionPlan.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/registrar.h"

#include "usdGenCurves/ops/curveSource.h"
#include "usdGenCurves/ops/deform.h"
#include "usdGenCurves/ops/grow.h"
#include "usdGenCurves/ops/length.h"
#include "usdGenCurves/ops/noise.h"
#include "usdGenSurface/ops/scatter.h"
#include "usdGenCurves/ops/width.h"
#include "usdGenCurves/ops/widthBlend.h"
#include "usdGen/api/span.h"

namespace usdGen {

namespace {

using namespace cudaP;

constexpr UsdGenParamSlot kScatterMap[kScatterCount] = {scatterP::kFlip};
constexpr UsdGenParamSlot kGrowMap[kGrowCount] = {
    growP::kSegments, growP::kLength, growP::kLengthRandom, growP::kLift, growP::kAzimuth,
    growP::kAzimuthRandom, growP::kDirection, growP::kDirectionVector};
constexpr UsdGenParamSlot kSourceMap[kSrcCount] = {
    curveSourceP::kUseRest, curveSourceP::kResampleTo, curveSourceP::kInterpolationMode,
    curveSourceP::kRegionMapChannel, curveSourceP::kIdSource, curveSourceP::kStaleAction,
    curveSourceP::kExpectEpoch, curveSourceP::kRebind};
constexpr UsdGenParamSlot kWidthMap[kWidthCount] = {
    widthP::kWidth, widthP::kMask, widthP::kEnabled, widthP::kReplace, widthP::kKnots,
    widthP::kKnotsInterp};
constexpr UsdGenParamSlot kLengthMap[kLenCount] = {
    lengthP::kMode, lengthP::kMethod, lengthP::kRebuild, lengthP::kValue, lengthP::kRandom,
    lengthP::kMinRemaining, lengthP::kCull, lengthP::kMask, lengthP::kEnabled};
constexpr UsdGenParamSlot kNoiseMap[kNoiseCount] = {
    noiseP::kMagnitude, noiseP::kFrequency, noiseP::kCorrelation, noiseP::kOctaves,
    noiseP::kLacunarity, noiseP::kGain, noiseP::kPreserveLength, noiseP::kMask,
    noiseP::kEnabled, noiseP::kCumulative, noiseP::kNoiseSeed, noiseP::kMagKnots,
    noiseP::kMagInterp};
constexpr UsdGenParamSlot kDeformMap[kDeformCount] = {
    deformP::kRbfSamples, deformP::kMask, deformP::kEnabled, deformP::kLockRoots};
constexpr UsdGenParamSlot kBlendMap[kBlendCount] = {widthBlendP::kWeight};

void Add(UsdGenRegistrar &r, UsdGenOpId op, UsdGenCudaStage stage, uint32_t version,
         int minimumAlgorithm, int maximumAlgorithm, uint32_t flags,
         UsdGenSpan<const UsdGenParamSlot> paramMap = {})
{
    if (op == UsdGenOpId::Invalid) return;
    UsdGenKernelDef k;
    k.stage = uint16_t(stage);
    k.capabilityVersion = version;
    k.minimumAlgorithmVersion = minimumAlgorithm;
    k.maximumAlgorithmVersion = maximumAlgorithm;
    k.flags = flags;
    k.paramMap = paramMap;
    if (stage == UsdGenCudaStage::Width) k.profileLut = &UsdGenWidthBezierLut;
    r.RegisterKernel(op, UsdGenCudaBackendId(), k);
}

}  // namespace

// Rows, versions and flags as the CUDA capability matrix (version 35) has them.
void UsdGenRegisterCurvesCuda(UsdGenRegistrar &r, UsdGenCudaKernelOps const &ops)
{
    using S = UsdGenCudaStage;
    uint32_t const topology = UsdGenCapabilityChangesTopology | UsdGenCapabilityTopologyBarrier |
                              UsdGenCapabilityExclusiveWorkspace;
    Add(r, ops.scatter, S::Scatter, 1, 0, 0, topology, UsdGenSpan(kScatterMap));
    Add(r, ops.grow, S::Grow, 13, 0, 0, topology, UsdGenSpan(kGrowMap));
    Add(r, ops.curveSource, S::CurveSource, 2, 0, 1, UsdGenCapabilityExpressions | topology,
        UsdGenSpan(kSourceMap));
    Add(r, ops.width, S::Width, 6, 0, 1,
        UsdGenCapabilityExpressions | UsdGenCapabilityCopyOnWriteWrites, UsdGenSpan(kWidthMap));
    Add(r, ops.length, S::Length, 7, 0, 1, UsdGenCapabilityExpressions | topology,
        UsdGenSpan(kLengthMap));
    Add(r, ops.noise, S::Noise, 7, 0, 1,
        UsdGenCapabilityExpressions | UsdGenCapabilityCopyOnWriteWrites, UsdGenSpan(kNoiseMap));
    Add(r, ops.deform, S::Deform, 1, 0, 1,
        UsdGenCapabilityExpressions | UsdGenCapabilityExclusiveWorkspace |
            UsdGenCapabilitySurfaceInput | UsdGenCapabilityTransactionalPublication,
        UsdGenSpan(kDeformMap));
    Add(r, ops.referenceSource, S::ReferenceSource, 2, 0, 1, topology);
    Add(r, ops.widthBlend, S::WidthBlend, 6, 0, 0, UsdGenCapabilityCopyOnWriteWrites,
        UsdGenSpan(kBlendMap));
}

UsdGenKernelDef const *UsdGenCudaKernel(UsdGenOpId op) noexcept
{
    return op == UsdGenOpId::Invalid ? nullptr
                                     : UsdGenBackendRegistry::Get().Find(op, UsdGenCudaBackendId());
}

UsdGenKernelDef const *UsdGenCudaKernel(UsdGenOpNode const &node) noexcept
{
    return UsdGenCudaKernel(UsdGenOpRegistry::Get().Find(node.type));
}

bool UsdGenIsCudaStage(UsdGenOpNode const &node, UsdGenCudaStage stage) noexcept
{
    UsdGenKernelDef const *k = UsdGenCudaKernel(node);
    return k && k->stage == uint16_t(stage);
}

UsdGenParamSlot UsdGenCudaSlot(UsdGenKernelDef const *kernel, uint16_t index) noexcept
{
    return kernel && index < kernel->paramMap.size() ? kernel->paramMap[index] : kUsdGenNoParam;
}

}  // namespace usdGen
