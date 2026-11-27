// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The curves' Vulkan program: the pipelines (built from the registered
// SPIR-V on a fresh device context) and the typed-plan route the runtime's
// Session provider shell forwards to.
#include "usdGenCurves/vulkan/curvesProgram.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGenCurves/vulkan/growVkPipeline.h"
#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/noisePipeline.h"
#include "usdGenCurves/vulkan/nonWidthComparePipeline.h"
#include "usdGenCurves/vulkan/picktileVk.h"
#include "usdGenCurves/vulkan/planExecutor.h"
#include "usdGenCurves/vulkan/rbfVkDeformPipeline.h"
#include "usdGenCurves/vulkan/resampleVkPipeline.h"
#include "usdGenCurves/vulkan/sessionProvider.h"
#include "usdGenCurves/vulkan/widthBlendPipeline.h"
#include "usdGenCurves/vulkan/widthPipeline.h"
#include "usdGenVulkan/exec/shaderLibrary.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenExpr/vulkan/exprVkApply.h"
#include "usdGenExpr/vulkan/exprVkContext.h"
#include "usdGenExpr/vulkan/exprVkEvaluate.h"
#include "usdGenExpr/vulkan/exprVkPack.h"
#include "usdGen/api/pipeline.h"
#include "usdGen/api/ext/device/executionTaskGraph.h"

#include <vulkan/vulkan.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen::vulkan {
namespace {

void SetReason(std::string *reason, char const *message) noexcept
{
    if (!reason) return;
    try { *reason = message; } catch (...) {}
}

// Copies a library module's words; false when the library has none.
bool LoadShader(ShaderLibrary const &library, char const *name, std::vector<uint32_t> *words)
{
    if (!library.Has(name)) return false;
    *words = library.Words(name);
    return true;
}

class CurvesProgram final : public Program {
public:
    explicit CurvesProgram(std::shared_ptr<DeviceContext> context) : context_(std::move(context)) {}
    ~CurvesProgram() override = default;

    bool Bind(ProgramBinding binding, std::string *reason) override
    {
        identity_ = binding.identity;
        domain_ = VulkanGenerationAdapterDomain::Create({binding.queueOwner, binding.completion});
        if (!domain_) {
            SetReason(reason, "Vulkan default provider cannot create a generation domain");
            return false;
        }
        preparer_ = UsdGenExecutionTaskGraph::GetOrCreate(*binding.runtime, "vulkan",
                                                          context_->resourceDeviceId());
        pipes_.domain = domain_;
        pipes_.preparer = preparer_;
        std::string executorReason;
        executor_ = VulkanPlanExecutor::Create(std::move(pipes_), &executorReason);
        if (!executor_) {
            std::string message = executorReason.empty()
                ? "Vulkan default provider cannot create a plan executor"
                : executorReason;
            SetReason(reason, message.c_str());
            return false;
        }
        std::string providerReason;
        inner_ = VulkanSessionProvider::Create(
            {executor_, context_, binding.queueOwner, binding.identity, 256}, &providerReason);
        if (!inner_) {
            std::string message = providerReason.empty()
                ? "Vulkan default provider cannot create a session provider"
                : providerReason;
            SetReason(reason, message.c_str());
            return false;
        }
        return true;
    }
    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const &desc,
                                          UsdGenDiagnostics *diagnostics) const override
    {
        return inner_ ? inner_->Compile(desc, diagnostics) : nullptr;
    }
    UsdGenSessionDeviceIdentity Identity() const noexcept override
    {
        return inner_ ? inner_->Identity() : identity_;
    }
    bool NotifyContextLost(UsdGenSessionDeviceIdentity const &lost) noexcept override
    {
        return inner_ && inner_->NotifyContextLost(lost);
    }
    bool Submit(UsdGenSessionDeviceRequest request) override
    {
        return inner_ ? inner_->Submit(std::move(request)) : false;
    }
    void Shutdown() noexcept override
    {
        if (inner_) inner_->Shutdown();
        if (executor_) executor_->Shutdown();
    }
    void StopAdmission() noexcept override
    {
        if (domain_) domain_->StopAdmission();
    }
    void DrainPreparer() noexcept override
    {
        if (preparer_) preparer_->Drain();
    }
    bool CloseDomain() noexcept override { return !domain_ || domain_->Close(); }
    bool RetireWhenIdle(std::function<void()> externalRetirement) noexcept override
    {
        return domain_ && domain_->RetireWhenIdle(std::move(externalRetirement));
    }

    // Declared so destruction runs inner, executor, preparer, domain, pipelines.
    std::shared_ptr<DeviceContext> context_;
    VulkanPlanExecutor::CreateInfo pipes_;
    std::shared_ptr<VulkanGenerationAdapterDomain> domain_;
    std::shared_ptr<UsdGenExecutionTaskGraph> preparer_;
    std::shared_ptr<VulkanPlanExecutor> executor_;
    std::shared_ptr<VulkanSessionProvider> inner_;
    UsdGenSessionDeviceIdentity identity_{};
};

}  // namespace

UsdGenRef<Program> CreateCurvesProgram(std::shared_ptr<DeviceContext> const &context,
                                       ShaderLibrary const &library, std::string *reason)
{
    if (!context) return {};
    bool const fp64 = context->shaderFloat64Enabled();
    bool const int64 = context->shaderInt64Enabled();
    // Missing optional modules only narrow later pipeline admission.
    std::vector<uint32_t> widthSpv;
    if (!LoadShader(library, "width", &widthSpv)) {
        SetReason(reason, "Vulkan default provider cannot load width.spv");
        return {};
    }
    std::vector<uint32_t> scale, set, cut, reparam, minimum, literal, envelope, lengthDevice;
    bool const hasScale = LoadShader(library, "lengthScale", &scale);
    bool const hasSet = LoadShader(library, "lengthSet", &set);
    bool const hasCut = LoadShader(library, "lengthCutExtend", &cut);
    bool const hasReparam = LoadShader(library, "lengthReparam", &reparam);
    bool const hasMinimum = LoadShader(library, "lengthMinimum", &minimum);
    bool const hasLiteral = LoadShader(library, "lengthLiteralV1", &literal);
    bool const hasEnvelope = LoadShader(library, "lengthEnvelopeV1", &envelope);
    bool const hasLengthDevice = LoadShader(library, "lengthDeviceV1", &lengthDevice);
    std::vector<uint32_t> blendSpv, compareSpv, compactSpv;
    bool const hasBlend = LoadShader(library, "widthBlend", &blendSpv);
    bool const hasCompare = LoadShader(library, "nonWidthCompare", &compareSpv);
    bool const hasCompact = LoadShader(library, "lengthCompaction", &compactSpv);
    std::vector<uint32_t> noiseValidateSpv, noiseGenerateSpv;
    bool const hasNoise =
        LoadShader(library, "noiseValidate", &noiseValidateSpv) &&
        LoadShader(library, "noiseGenerate", &noiseGenerateSpv);
    std::vector<uint32_t> resampleSpv;
    bool const hasResample = LoadShader(library, "resampleVk", &resampleSpv);
    std::vector<uint32_t> exprContextSpv, exprEvaluateSpv, exprWidthSpv, exprPackSpv, growSpv;
    bool const hasExprContext = LoadShader(library, "exprVkContext", &exprContextSpv);
    bool const hasExprEvaluate = LoadShader(library, "exprVkEvaluate", &exprEvaluateSpv);
    bool const hasExprWidth = LoadShader(library, "exprVkWidth", &exprWidthSpv);
    bool const hasExprPack = LoadShader(library, "exprVkPack", &exprPackSpv);
    bool const hasGrow = LoadShader(library, "curveGrow", &growSpv);
    // Phase 2 picktiles: optional tile-span/bounds finalization; missing
    // shaders only disable the tile stage (publications carry no tiles).
    std::vector<uint32_t> tilesSpv, boundsSpv;
    bool const hasTiles = LoadShader(library, "picktileVkTiles", &tilesSpv) &&
        LoadShader(library, "picktileVkBounds", &boundsSpv);
    // Phase-2 RBF device path (rbfVk hook): optional modules; missing
    // shaders narrow all Deform admission (see planExecutor.cpp).
    std::vector<uint32_t> rbfVkExtentSpv, rbfVkGramSpv, rbfVkBuildMatrixSpv, rbfVkLuSpv,
        rbfVkRhsSpv, rbfVkTriSolveSpv, rbfVkEvaluateSpv, rbfVkApplySpv;
    bool const hasRbfVk =
        LoadShader(library, "rbfVkExtent", &rbfVkExtentSpv) &&
        LoadShader(library, "rbfVkGram", &rbfVkGramSpv) &&
        LoadShader(library, "rbfVkBuildMatrix", &rbfVkBuildMatrixSpv) &&
        LoadShader(library, "rbfVkLu", &rbfVkLuSpv) &&
        LoadShader(library, "rbfVkRhs", &rbfVkRhsSpv) &&
        LoadShader(library, "rbfVkTriSolve", &rbfVkTriSolveSpv) &&
        LoadShader(library, "rbfVkEvaluate", &rbfVkEvaluateSpv) &&
        LoadShader(library, "rbfVkApply", &rbfVkApplySpv);
    // R-cache programs: optional, never gating. Missing shaders only
    // disable the RbfVkBinding evaluate cache (direct evaluation, as
    // before); the binding rejects a partial set at Create.
    std::vector<uint32_t> rbfVkVerifySpv, rbfVkFillSpv, rbfVkEvaluateCachedSpv;
    bool const hasRbfVkEvalCache =
        LoadShader(library, "rbfVkVerify", &rbfVkVerifySpv) &&
        LoadShader(library, "rbfVkFill", &rbfVkFillSpv) &&
        LoadShader(library, "rbfVkEvaluateCached", &rbfVkEvaluateCachedSpv);
    if (!hasRbfVkEvalCache) {
        rbfVkVerifySpv.clear();
        rbfVkFillSpv.clear();
        rbfVkEvaluateCachedSpv.clear();
    }
    VkResult status = VK_SUCCESS;
    // Pipelines need only the device context; build them before the
    // completion waiter exists so a pipeline failure destroys plain Vk
    // objects inline (safe on any thread) instead of needing the reaper.
    auto width = WidthPipeline::Create(context, widthSpv, &status);
    if (!width) {
        SetReason(reason, "Vulkan default provider cannot create the width pipeline");
        return {};
    }
    // Length scale/set/cut/reparam/minimum/literal/envelope chain: install
    // the largest contiguous combination present; missing shaders only
    // narrow admission for graphs needing those Length forms.
    std::shared_ptr<LengthScalePipeline> length;
    if (hasScale && hasSet && hasCut && hasReparam && hasMinimum && hasLiteral &&
        hasEnvelope && hasLengthDevice)
        length = LengthScalePipeline::CreateWithDeviceV1(
            context, scale, set, cut, reparam, minimum, literal,
            envelope, lengthDevice);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum && hasLiteral &&
        hasEnvelope)
        length = LengthScalePipeline::CreateWithEnvelopeV1(
            context, scale, set, cut, reparam, minimum, literal,
            envelope);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum &&
             hasLiteral)
        length = LengthScalePipeline::CreateWithLiteralV1(
            context, scale, set, cut, reparam, minimum, literal);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum)
        length = LengthScalePipeline::CreateWithMinimum(
            context, scale, set, cut, reparam, minimum);
    else if (hasScale && hasSet && hasCut && hasReparam)
        length = LengthScalePipeline::CreateWithReparam(context, scale,
                                                        set, cut, reparam);
    else if (hasScale && hasSet && hasCut)
        length = LengthScalePipeline::CreateWithCutExtend(context, scale,
                                                          set, cut);
    else if (hasScale && hasSet)
        length = LengthScalePipeline::CreateWithSet(context, scale, set);
    else if (hasScale)
        length = LengthScalePipeline::Create(context, scale);
    std::shared_ptr<WidthBlendPipeline> blend;
    if (hasBlend) blend = WidthBlendPipeline::Create(context, blendSpv);
    std::shared_ptr<NonWidthComparePipeline> compare;
    if (hasCompare)
        compare = NonWidthComparePipeline::Create(context, compareSpv);
    std::shared_ptr<LengthCompactionPipeline> compact;
    if (hasCompact)
        compact = LengthCompactionPipeline::CreateWithKeep(context, compactSpv);
    std::shared_ptr<NoisePipeline> noise;
    if (fp64 && hasNoise)
        noise = NoisePipeline::Create(context, noiseValidateSpv,
                                      noiseGenerateSpv);
    std::shared_ptr<ResampleVkPipeline> resample;
    if (fp64 && hasResample)
        resample = ResampleVkPipeline::Create(context, resampleSpv);
    std::shared_ptr<ExprVkContextPipeline> expressionContext;
    std::shared_ptr<ExprVkEvaluatePipeline> expressionEvaluate;
    std::shared_ptr<ExprVkWidthApplyPipeline> expressionWidth;
    std::shared_ptr<ExprVkPackPipeline> expressionPack;
    if (fp64 && int64 && hasExprContext && hasExprEvaluate) {
        expressionContext = ExprVkContextPipeline::Create(context, exprContextSpv);
        expressionEvaluate = ExprVkEvaluatePipeline::Create(context, exprEvaluateSpv);
    }
    if (hasExprWidth)
        expressionWidth = ExprVkWidthApplyPipeline::Create(context, exprWidthSpv);
    if (hasExprPack)
        expressionPack = ExprVkPackPipeline::Create(context, exprPackSpv);
    std::shared_ptr<GrowVkPipeline> grow;
    if (hasGrow && resample) {
        auto nativeGrow = CurveGrowPipeline::Create(context, growSpv);
        if (nativeGrow) grow = GrowVkPipeline::Create(nativeGrow, resample);
    }
    std::shared_ptr<PicktileVkTilesPipeline> tiles;
    std::shared_ptr<PicktileVkBoundsPipeline> bounds;
    if (hasTiles) {
        tiles = PicktileVkTilesPipeline::Create(context, tilesSpv);
        bounds = PicktileVkBoundsPipeline::Create(context, boundsSpv);
        if (!tiles || !bounds) { tiles.reset(); bounds.reset(); }
    }
    std::shared_ptr<RbfVkDeformPipeline> rbfVkDeform;
    if (fp64 && hasRbfVk) {
        RbfVkBindingSpirv rbfVkSpirv;
        rbfVkSpirv.extent = std::move(rbfVkExtentSpv);
        rbfVkSpirv.gram = std::move(rbfVkGramSpv);
        rbfVkSpirv.buildMatrix = std::move(rbfVkBuildMatrixSpv);
        rbfVkSpirv.lu = std::move(rbfVkLuSpv);
        rbfVkSpirv.rhs = std::move(rbfVkRhsSpv);
        rbfVkSpirv.triSolve = std::move(rbfVkTriSolveSpv);
        rbfVkSpirv.evaluate = std::move(rbfVkEvaluateSpv);
        rbfVkSpirv.verify = std::move(rbfVkVerifySpv);
        rbfVkSpirv.fill = std::move(rbfVkFillSpv);
        rbfVkSpirv.evaluateCached = std::move(rbfVkEvaluateCachedSpv);
        rbfVkDeform = RbfVkDeformPipeline::Create(context, rbfVkSpirv, rbfVkApplySpv);
    }
    auto program = UsdGenMakeRef<CurvesProgram>(context);
    VulkanPlanExecutor::CreateInfo &executorInfo = program->pipes_;
    executorInfo.widthPipeline = width;
    executorInfo.lengthPipeline = length;
    executorInfo.widthBlendPipeline = blend;
    executorInfo.nonWidthComparePipeline = compare;
    executorInfo.lengthCompactionPipeline = compact;
    executorInfo.noisePipeline = noise;
    // Production deformation requires the device solver. Missing FP64 or
    // RBF modules fail admission; legacy host-fit pipelines remain available
    // only to callers that explicitly inject a compatibility provider.
    executorInfo.resampleVkPipeline = resample;
    executorInfo.tilesPipeline = tiles;
    executorInfo.boundsPipeline = bounds;
    executorInfo.rbfVkDeformPipeline = rbfVkDeform;
    executorInfo.expressionContextPipeline = expressionContext;
    executorInfo.expressionEvaluatePipeline = expressionEvaluate;
    executorInfo.expressionWidthPipeline = expressionWidth;
    executorInfo.expressionPackPipeline = expressionPack;
    executorInfo.growVkPipeline = grow;
    return program;
}

}  // namespace usdGen::vulkan
