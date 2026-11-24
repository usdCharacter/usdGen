// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Vulkan kernel rows for the built-in curve ops (step 8 moves them into
// usdGenCurvesVulkan). The lane reads stage and paramMap; it includes no op
// header.
#include "usdGen/vulkan/vkStages.h"

#include "usdGen/graphDesc.h"
#include "usdGen/ops/curveSource.h"
#include "usdGen/ops/deform.h"
#include "usdGen/ops/grow.h"
#include "usdGen/ops/length.h"
#include "usdGen/ops/scatter.h"
#include "usdGen/ops/width.h"
#include "usdGen/ops/widthBlend.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include <atomic>

namespace usdGen {
namespace {

constexpr UsdGenBackendId kVulkan = UsdGenBackendIdOf(uint8_t(UsdGenExecutionBackend::Vulkan));

constexpr UsdGenParamSlot kCurveSourceMap[vulkanP::kCsCount] = {
    curveSourceP::kUseRest, curveSourceP::kUsdGenUseRest, curveSourceP::kResampleTo,
    curveSourceP::kUsdGenResampleTo, curveSourceP::kRebind, curveSourceP::kInterpolationMode};
constexpr UsdGenParamSlot kWidthMap[vulkanP::kWidthCount] = {widthP::kProfileMode};
constexpr UsdGenParamSlot kLengthMap[vulkanP::kLenCount] = {lengthP::kMode};
constexpr UsdGenParamSlot kWidthBlendMap[vulkanP::kBlendCount] = {widthBlendP::kWeight};
constexpr UsdGenParamSlot kGrowMap[vulkanP::kGrowCount] = {
    growP::kSegments, growP::kLength, growP::kLengthRandom, growP::kLift,
    growP::kAzimuth, growP::kAzimuthRandom, growP::kDirection, growP::kDirectionVector};
constexpr UsdGenParamSlot kDeformMap[vulkanP::kDeformCount] = {deformP::kRbfSamples};
constexpr UsdGenParamSlot kScatterMap[vulkanP::kScCount] = {
    scatterP::kDensity, scatterP::kFlip, scatterP::kSubdivisionLevel};

std::atomic<UsdGenVulkanTokens const *> gVkTokens{nullptr};

UsdGenVulkanTokens *MakeTokens()
{
    UsdGenCompileTokenScope scope;
    auto *t = new UsdGenVulkanTokens;
    t->primvar = UsdGenInternToken("primvar");
    t->index = UsdGenInternToken("index");
    t->warn = UsdGenInternToken("warn");
    t->ignore = UsdGenInternToken("ignore");
    t->block = UsdGenInternToken("block");
    t->never = UsdGenInternToken("never");
    t->onError = UsdGenInternToken("onError");
    t->always = UsdGenInternToken("always");
    t->surfaceCage = UsdGenInternToken("surfaceCage");
    t->scale = UsdGenInternToken("scale");
    t->set = UsdGenInternToken("set");
    t->cull = UsdGenInternToken("cull");
    t->cutExtend = UsdGenInternToken("cutExtend");
    t->keepParam = UsdGenInternToken("keepParam");
    t->reparam = UsdGenInternToken("reparam");
    t->surfaceNormal = UsdGenInternToken("surfaceNormal");
    t->vector = UsdGenInternToken("vector");
    t->regionMapRel = UsdGenInternToken("usdGen:regionMap");
    t->exprSource = UsdGenInternToken("expr:source");
    t->capturedRoots = UsdGenInternToken("__vulkanCapturedRoots");
    t->owners = UsdGenInternToken("__vulkanOwners");
    t->resampleTransform = UsdGenInternToken("VulkanResampleTransform");
    t->sourceAlias = UsdGenInternToken("VulkanSourceAlias");
    t->lengthTransform = UsdGenInternToken("VulkanLengthTransform");
    t->publication = UsdGenInternToken("VulkanPublication");
    return t;
}

void Register(UsdGenRegistrar &r, UsdGenOpId op, UsdGenVulkanStage stage,
              TfSpan<const UsdGenParamSlot> paramMap = {},
              bool (*profileLut)(UsdGenParamView const &, float *, size_t) = nullptr)
{
    if (op == UsdGenOpId::Invalid) return;
    UsdGenKernelDef k;
    k.stage = uint16_t(stage);
    k.paramMap = paramMap;
    k.profileLut = profileLut;
    r.RegisterKernel(op, kVulkan, k);
}

}  // namespace

// Built on first use and published by CAS (registration builds it, so the
// lane never interns); a losing thread discards its copy. Never freed.
UsdGenVulkanTokens const &UsdGenVkTokens() noexcept
{
    if (UsdGenVulkanTokens const *t = gVkTokens.load(std::memory_order_acquire)) return *t;
    UsdGenVulkanTokens *fresh = MakeTokens();
    UsdGenVulkanTokens const *winner = nullptr;
    if (gVkTokens.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                          std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

UsdGenVulkanKernel UsdGenVulkanKernelOf(TfToken const &type) noexcept
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpId const id = ops.Find(type);
    if (id == UsdGenOpId::Invalid) return {};
    UsdGenKernelDef const *kernel = UsdGenBackendRegistry::Get().Find(id, kVulkan);
    if (!kernel) return {};
    return {&ops.Record(id), kernel};
}

UsdGenVulkanKernel UsdGenVulkanKernelFor(UsdGenVulkanStage s) noexcept
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenBackendRegistry const &kernels = UsdGenBackendRegistry::Get();
    for (UsdGenOpId id : kernels.Ops(kVulkan)) {
        UsdGenKernelDef const *kernel = kernels.Find(id, kVulkan);
        if (kernel && kernel->stage == uint16_t(s)) return {&ops.Record(id), kernel};
    }
    return {};
}

void UsdGenRegisterVulkanKernels(UsdGenRegistrar &r, UsdGenVulkanKernelOps const &ops)
{
    (void)UsdGenVkTokens();
    using S = UsdGenVulkanStage;
    Register(r, ops.curveSource, S::CurveSource, UsdGenSpan(kCurveSourceMap));
    Register(r, ops.referenceSource, S::ReferenceSource);
    Register(r, ops.scatter, S::Scatter, UsdGenSpan(kScatterMap));
    Register(r, ops.grow, S::Grow, UsdGenSpan(kGrowMap));
    Register(r, ops.width, S::Width, UsdGenSpan(kWidthMap), &UsdGenWidthBezierLut);
    Register(r, ops.length, S::Length, UsdGenSpan(kLengthMap));
    Register(r, ops.widthBlend, S::WidthBlend, UsdGenSpan(kWidthBlendMap));
    Register(r, ops.noise, S::Noise);
    Register(r, ops.deform, S::Deform, UsdGenSpan(kDeformMap));
    Register(r, ops.resample, S::Resample);
}

}  // namespace usdGen
