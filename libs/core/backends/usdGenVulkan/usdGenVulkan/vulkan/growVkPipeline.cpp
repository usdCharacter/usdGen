// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "growVkPipeline.h"
namespace usdGen::vulkan {
std::shared_ptr<GrowVkPipeline> GrowVkPipeline::Create(
    std::shared_ptr<CurveGrowPipeline> geometry, std::shared_ptr<ResampleVkPipeline> named) {
    if (!geometry || !named || geometry->context() != named->context()) return {};
    auto pipeline = std::make_shared<GrowVkPipeline>();
    pipeline->geometry_ = std::move(geometry); pipeline->named_ = std::move(named);
    return pipeline;
}
std::shared_ptr<DeviceContext> const& GrowVkPipeline::context() const noexcept { return geometry_->context(); }
std::unique_ptr<GrowVkPipeline::Candidate> GrowVkPipeline::Begin(
    std::shared_ptr<const VulkanSourceGeneration> source,
    CurveGrowPipeline::Controls const& controls, std::string const& targetPlane,
    uint64_t version, VkResult* result, CurveGrowPipeline::BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!source || source->context() != context() || version <= source->valueVersion()) return {};
    auto candidate = std::unique_ptr<Candidate>(new Candidate);
    candidate->input_ = source; candidate->target_ = controls.cvCount; candidate->version_ = version;
    CurveGrowPipeline::Input input;
    input.points = source->PlaneOwner("points"); input.rest = source->PlaneOwner("rest");
    input.widths = source->PlaneOwner("width"); input.curveOffsets = source->PlaneOwner("curveOffsets");
    input.stableIds = source->PlaneOwner("stableIds"); input.rootPrim = source->PlaneOwner("rootPrim");
    input.rootUV = source->PlaneOwner("rootUV"); input.rootT = source->SourceFrameOwner("sourceRootT");
    input.rootB = source->SourceFrameOwner("sourceRootB"); input.rootN = source->SourceFrameOwner("sourceRootN");
    input.targets = source->SourceFrameOwner(targetPlane);
    // Both Begin calls submit on the serialized owner. The admission proof is
    // acquired before the first submit and the caller arms its completion
    // marker only after both Begin calls have returned.
    // Empty geometry performs no native submit, but named resampling still
    // submits the canonical offset copy. Admit that first submission too.
    bool const empty = source->curveCount() == 0;
    candidate->geometry_ = geometry_->Begin(source, input, source->curveCount(), source->pointCount(),
        controls, result, empty ? CurveGrowPipeline::BeforeSubmit{} : beforeSubmit);
    if (!candidate->geometry_) return {};
    candidate->named_ = named_->Begin(source, controls.cvCount, result,
                                     empty ? std::move(beforeSubmit) : ResampleVkPipeline::BeforeSubmit{});
    if (!candidate->named_) return {};
    return candidate;
}
GrowVkPipeline::Candidate::~Candidate() = default;
void GrowVkPipeline::Candidate::Quarantine() noexcept {
    if (geometry_) geometry_->Quarantine();
    if (named_) named_->Quarantine();
}
VkResult GrowVkPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    if (semanticStatus) *semanticStatus = semantic_;
    if (proved_) return VK_SUCCESS;
    CurveGrowPipeline::Candidate::Semantic growStatus;
    VkResult const geometryProof = geometry_->Poll(&growStatus);
    uint32_t namedStatus = 0;
    VkResult const namedProof = named_->Poll(&namedStatus);
    if (geometryProof != VK_SUCCESS && geometryProof != VK_NOT_READY) return geometryProof;
    if (namedProof != VK_SUCCESS && namedProof != VK_NOT_READY) return namedProof;
    if (geometryProof == VK_NOT_READY || namedProof == VK_NOT_READY) return VK_NOT_READY;
    semantic_ = uint32_t(growStatus) ? uint32_t(growStatus) : namedStatus;
    proved_ = true;
    if (!semantic_) {
        output_ = VulkanSourceGeneration::WithGrown(input_, *geometry_, *named_, target_, version_);
        if (!output_) return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (semanticStatus) *semanticStatus = semantic_;
    return VK_SUCCESS;
}
} // namespace usdGen::vulkan
