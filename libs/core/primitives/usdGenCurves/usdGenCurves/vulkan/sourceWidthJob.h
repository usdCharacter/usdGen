// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_SOURCE_WIDTH_JOB_H
#define USDGEN_VULKAN_SOURCE_WIDTH_JOB_H
#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGenCurves/vulkan/deformPipeline.h"
#include "usdGen/exec/sharedCell.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGenExpr/vulkan/exprVkApply.h"
#include "usdGenExpr/vulkan/exprVkPack.h"
#include "usdGenExpr/vulkan/exprVkRun.h"
#include "usdGenCurves/vulkan/growVkPipeline.h"
#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/noisePipeline.h"
#include "usdGenCurves/vulkan/nonWidthComparePipeline.h"
#include "usdGenCurves/vulkan/picktileVk.h"
#include "usdGenCurves/vulkan/rbfVkDeformPipeline.h"
#include "usdGenCurves/vulkan/resampleVkPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGen/exec/executionPipeline.h"
#include "usdGen/exec/executionTaskGraph.h"
#include "usdGenCurves/vulkan/widthBlendPipeline.h"
#include <atomic>
#include <functional>
namespace usdGen::vulkan {
// Owner-confined, value-indexed execution. Stages submit serially in validated
// topological order; fan-out shares immutable values and fan-in uses explicit
// ordered inputs. This does not claim concurrent GPU branch execution.
class VulkanSourceWidthJob final : public std::enable_shared_from_this<VulkanSourceWidthJob> {
  public:
    enum class State {
        Created,
        Preparing,
        UploadPending,
        WidthPending,
        Ready,
        Superseded,
        Failed,
        LostProof,
        LengthPending,
        BlendPending,
        ComparePending,
        CullCountsPending,
        CullScatterPending,
        NoisePending,
        DeformPending,
        ResamplePending,
        TileFinalizePending,
        ExpressionPending,
        ExpressionWidthPending,
        GrowPending,
        ExpressionPackPending,
        ExpressionHostPending
    };
    struct CreateInfo {
        VulkanSourceGenerationCreateInfo source;
        std::shared_ptr<WidthPipeline> widthPipeline;
        float width = 0;
        uint32_t replace = 0;
        uint64_t valueVersion = 0;
        std::shared_ptr<const void> requestLifetime;
        std::shared_ptr<VulkanCompletionService> completionService;
        // Shared bounded CPU task dispatcher. When supplied, validation and
        // packing run off-owner; native allocation/copy/submission remain on
        // owner. Null preserves the explicit synchronous-preparation probe
        // path, not the intended production scheduling policy.
        std::shared_ptr<UsdGenExecutionTaskGraph> preparer;
        std::shared_ptr<LengthScalePipeline> lengthPipeline = {};
        float lengthFactor = 1.0f;
        uint64_t lengthValueVersion = 0;
        std::shared_ptr<WidthBlendPipeline> widthBlendPipeline = {};
        // Empty stages retain the legacy optional-Length/Width shorthand.
        // Explicit stages require one strictly increasing revision per stage.
        std::vector<VulkanSourceWidthStage> stages = {};
        std::vector<uint64_t> stageValueVersions = {};
        std::shared_ptr<NonWidthComparePipeline> nonWidthComparePipeline = {};
        std::shared_ptr<LengthCompactionPipeline> lengthCompactionPipeline = {};
        std::shared_ptr<NoisePipeline> noisePipeline = {};
        std::shared_ptr<DeformPipeline> deformPipeline = {};
        // Phase 2 picktiles: tile-span/bounds finalization for Vulkan
        // publications. Either pipeline null disables the tile stage. When
        // both are set, non-empty identity-cardinality lanes finalize tiles
        // after the last stage; cull-lane survivor order is a follow-up.
        std::shared_ptr<PicktileVkTilesPipeline> tilesPipeline;
        std::shared_ptr<PicktileVkBoundsPipeline> boundsPipeline;
        PicktileVkTileOrder tileOrder = PicktileVkTileOrder::IdentityCaptureOrder;
        uint32_t tileChunkSize = 0; // 0 = CUDA defaults (512/64)
        uint32_t tileTarget = 0;
        std::shared_ptr<ResampleVkPipeline> resampleVkPipeline = {};
        // Phase-2 RBF device path (rbfVk hook). Must stay last.
        std::shared_ptr<RbfVkDeformPipeline> rbfVkDeformPipeline = {};
        std::shared_ptr<ExprVkContextPipeline> expressionContextPipeline;
        std::shared_ptr<ExprVkEvaluatePipeline> expressionEvaluatePipeline;
        std::shared_ptr<ExprVkWidthApplyPipeline> expressionWidthPipeline;
        std::shared_ptr<GrowVkPipeline> growVkPipeline;
        std::shared_ptr<ExprVkPackPipeline> expressionPackPipeline;
    };
    using Completion =
        std::function<void(std::shared_ptr<const VulkanSourceGeneration>, State, VkResult)>;
    static std::shared_ptr<VulkanSourceWidthJob> Create(CreateInfo, std::string *reason = nullptr);
    // Bind once and reserve progress, terminal and optional preparation-return
    // admission before submission.
    // False means no native work/callback admitted; Start is single-attempt,
    // including admission failure (create a fresh job to retry).
    // Externally retain owner and
    // stop/join notification producers through terminal completion.
    bool Start(UsdGenExecutionPipeline &, UsdGenExecutionPipeline::Cancellation, Completion);
    // Thread-safe after Start returns. Coalesced real completion notifications.
    // If both reserved owner routes fail, retain the entire job permanently
    // and report LostProof thread-neutrally. That exceptional callback must
    // not perform queue work or destroy native ownership.
    bool NotifyCompletion();
    // Thread-neutral failure ingress used by the completion service's owner
    // route.
    void NotifyFailure(VkResult) noexcept;
    void SuppressPublication() noexcept { suppressed_.store(true, std::memory_order_release); }
    State state() const noexcept { return state_.load(std::memory_order_acquire); }
    State failurePhase() const noexcept { return failurePhase_.load(std::memory_order_acquire); }

  private:
    struct Control;
    struct Relay;
    struct PrepareRelay;
    explicit VulkanSourceWidthJob(CreateInfo);
    void Advance();
    void BeginPreparation();
    void TakePrepared(std::shared_ptr<PrepareRelay>) noexcept;
    void FinishWidth();
    void BeginWidth();
    void BeginExpressions();
    void AdvanceExpressions();
    void BeginExpressionPack();
    void FinishExpressionPack();
    ExprVkParameterField const *PackedField(char const *) const;
    void FinishExpressions();
    void BeginExpressionHost();
    void TakeExpressionHost(std::shared_ptr<PrepareRelay>);
    void BeginExpressionWidth();
    void FinishExpressionWidth();
    void DispatchStage();
    void BeginLength();
    void FinishLength();
    void BeginNoise();
    void FinishNoise();
    void BeginDeform();
    void AdvanceDeform();
    void FinishDeform();
    void BeginRbfVkDeform();
    void FinishRbfVkDeform();
    void AdvanceRbfVkDeform();
    void BeginResample();
    void BeginGrow();
    void FinishGrow();
    void FinishResample();
    bool NeedsTiles(VulkanSourceGeneration const &) const noexcept;
    void BeginTiles();
    void BeginTileBounds();
    void FinishTiles();
    void BeginCull(bool scatter = false);
    void FinishCull(bool scatter);
    void BeginStage();
    void FinishStage(std::shared_ptr<const VulkanSourceGeneration>);
    void BeginBlend();
    void BeginCompare();
    void FinishCompare();
    void FinishBlend();
    void FailOnOwner(VkResult = VK_ERROR_DEVICE_LOST);
    void RouteFailure(VkResult = VK_ERROR_DEVICE_LOST) noexcept;
    bool Suppressed() const noexcept;
    void Terminal(std::shared_ptr<const VulkanSourceGeneration>, State, VkResult) noexcept;
    // `phase` is the failure phase to report (default: the current state).
    void Terminal(std::shared_ptr<const VulkanSourceGeneration>, State, VkResult, State phase) noexcept;
    CreateInfo info_;
    std::shared_ptr<VulkanSourceUpload> upload_;
    std::shared_ptr<const VulkanSourceGeneration> base_;
    std::unique_ptr<WidthPipeline::Candidate> width_;
    std::unique_ptr<ExprVkRun> expressions_;
    std::unique_ptr<ExprVkWidthApplyPipeline::Candidate> expressionWidth_;
    std::shared_ptr<const ChargedBuffer> expressionProfile_;
    std::unique_ptr<ExprVkPackPipeline::Candidate> expressionPack_;
    std::vector<ExprVkParameterField> packedFields_;
    size_t packedIndex_ = 0;
    std::unique_ptr<LengthScalePipeline::Candidate> length_;
    std::unique_ptr<LengthCompactionPipeline::Candidate> cull_;
    std::unique_ptr<WidthBlendPipeline::Candidate> blend_;
    std::unique_ptr<NonWidthComparePipeline::Candidate> compare_;
    std::unique_ptr<NoisePipeline::Candidate> noise_;
    std::unique_ptr<DeformPipeline::Candidate> deform_;
    bool deformMarked_ = false;  // the apply submit marked watch_
    std::unique_ptr<RbfVkDeformPipeline::Candidate> rbfVkDeform_;
    std::unique_ptr<ResampleVkPipeline::Candidate> resample_;
    std::unique_ptr<GrowVkPipeline::Candidate> grow_;
    std::unique_ptr<PicktileVkTilesPipeline::Candidate> tiles_;
    std::unique_ptr<PicktileVkBoundsPipeline::Candidate> bounds_;
    size_t tileCount_ = 0;
    std::vector<std::shared_ptr<const VulkanSourceGeneration>> values_;
    size_t stageIndex_ = 0;
    std::vector<std::shared_ptr<const ChargedBuffer>> lengthKeepValues_;
    std::shared_ptr<const ChargedBuffer> nextLengthKeep_;
    State stageFrom_ = State::UploadPending;
    UsdGenExecutionPipeline *owner_ = nullptr;
    UsdGenSharedCell<Control> control_;  // Start publishes; Settle/RouteFailure clear
    std::unique_ptr<VulkanCompletionService::Watch> watch_;
    bool watchProofReady_ = false;
    UsdGenExecutionPipeline::Cancellation cancellation_;
    std::shared_ptr<Relay> relay_;
    std::shared_ptr<PrepareRelay> prepareRelay_;
    std::shared_ptr<PrepareRelay> expressionHostRelay_;
    std::shared_ptr<const VulkanPreparedSource> prepared_;
    std::atomic<State> state_{State::Created};
    std::atomic<State> failurePhase_{State::Created};
    std::atomic<bool> suppressed_{false}, started_{false};
};
} // namespace usdGen::vulkan
#endif
