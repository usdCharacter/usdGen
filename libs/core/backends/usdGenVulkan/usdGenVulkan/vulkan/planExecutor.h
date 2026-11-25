// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Owner-admitted executor for typed Vulkan single-source geometry-value plans.
#ifndef USDGEN_VULKAN_PLAN_EXECUTOR_H
#define USDGEN_VULKAN_PLAN_EXECUTOR_H

#include "deformPipeline.h"
#include "executionPlan.h"
#include "exprVkApply.h"
#include "exprVkPack.h"
#include "exprVkRun.h"
#include "growVkPipeline.h"
#include "noisePipeline.h"
#include "picktileVk.h"
#include "publicationJob.h"
#include "rbfVkDeformPipeline.h"
#include "resampleVkPipeline.h"
#include "usdGen/executionValueRevisions.h"

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace usdGen::vulkan {

// This executor is deliberately injected rather than registered as a generic
// execution backend.  Its sole input is the native-free descriptor compiler's
// typed payload; it never interprets an arbitrary
// ExecutionPlanHandle::Payload(). CPU workers capture source data only; native
// operator stages and joins run on the shared device owner without production
// geometry readback or fallback.
class VulkanPlanExecutor final : public std::enable_shared_from_this<VulkanPlanExecutor> {
  public:
    struct CreateInfo {
        std::shared_ptr<VulkanGenerationAdapterDomain> domain;
        std::shared_ptr<WidthPipeline> widthPipeline;
        // Bounded worker graph used only to build the host source packet.
        std::shared_ptr<UsdGenExecutionTaskGraph> preparer;
        std::shared_ptr<LengthScalePipeline> lengthPipeline = {};
        std::shared_ptr<WidthBlendPipeline> widthBlendPipeline = {};
        std::shared_ptr<NonWidthComparePipeline> nonWidthComparePipeline = {};
        std::shared_ptr<LengthCompactionPipeline> lengthCompactionPipeline = {};
        std::shared_ptr<NoisePipeline> noisePipeline = {};
        std::shared_ptr<DeformPipeline> deformPipeline = {};
        std::shared_ptr<ResampleVkPipeline> resampleVkPipeline = {};
        // Phase 2 picktiles: tile-span/bounds finalization. Either pipeline
        // null disables the tile stage; both must share the request context.
        std::shared_ptr<PicktileVkTilesPipeline> tilesPipeline = {};
        std::shared_ptr<PicktileVkBoundsPipeline> boundsPipeline = {};
        PicktileVkTileOrder tileOrder = PicktileVkTileOrder::IdentityCaptureOrder;
        uint32_t tileChunkSize = 0; // 0 = CUDA defaults (512/64)
        uint32_t tileTarget = 0;
        // Phase-2 RBF device path (rbfVk hook): null when its shaders are
        // missing; gap admission then fails gracefully. Must stay last.
        std::shared_ptr<RbfVkDeformPipeline> rbfVkDeformPipeline = {};
        std::shared_ptr<ExprVkContextPipeline> expressionContextPipeline;
        std::shared_ptr<ExprVkEvaluatePipeline> expressionEvaluatePipeline;
        std::shared_ptr<ExprVkWidthApplyPipeline> expressionWidthPipeline;
        std::shared_ptr<GrowVkPipeline> growVkPipeline;
        std::shared_ptr<ExprVkPackPipeline> expressionPackPipeline;
    };
    struct Request {
        // When present, these are the Session's authoritative publication
        // revisions. A zero source revision is valid. Length plans require
        // one explicit intermediate revision, strictly between source and
        // final Width. Absent revisions preserve legacy Width-only behavior;
        // they are rejected for Length instead of inventing an intermediate.
        using AuthoritativeRevisions = UsdGenExecutionValueRevisions;
        // Must be the exact context retained by the injected domain service.
        std::shared_ptr<DeviceContext> context;
        // Version one is the fixed capability contract of the typed
        // VulkanSourceWidthPlan compiler. It is checked at admission rather
        // than inferred from any process-global backend registration.
        uint32_t capabilityVersion = 1;
        uint64_t generation = 0;
        std::optional<AuthoritativeRevisions> authoritativeRevisions;
        UsdGenDeviceToolMetadata tool;
        UsdGenExecutionPipeline::Cancellation cancellation;
        // Holds caller request state through terminal publication.  It is not
        // decoded by this executor and may be null.
        std::shared_ptr<const void> requestLifetime;
        // Absent values preserve direct callers' frozen-descriptor behavior.
        std::optional<double> evaluationFrame;
    };
    using Completion =
        std::function<void(std::shared_ptr<const UsdGenDeviceGeneration>, std::exception_ptr)>;

    static std::shared_ptr<VulkanPlanExecutor> Create(CreateInfo, std::string *reason = nullptr);

    // Owner-frame admission only.  It reserves the owner return ticket and
    // operation lease before queueing bounded CPU work.  False admits neither
    // source capture nor completion; once true, completion is exactly once.
    bool Submit(std::shared_ptr<const VulkanSourceWidthPlan> plan, Request request,
                Completion completion);

    // Stops new admission.  It intentionally does not wait for, destroy, or
    // detach accepted native work; external callers own domain Close().
    void Shutdown() noexcept;
    bool IsShutdown() const noexcept { return closing_.load(std::memory_order_acquire); }
    CreateInfo const &createInfo() const noexcept { return info_; }

  private:
    explicit VulkanPlanExecutor(CreateInfo) noexcept;
    CreateInfo info_;
    std::atomic<bool> closing_{false};
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_PLAN_EXECUTOR_H
