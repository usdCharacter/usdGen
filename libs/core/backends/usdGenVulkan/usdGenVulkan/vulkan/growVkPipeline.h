// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_GROW_VK_PIPELINE_H
#define USDGEN_VULKAN_GROW_VK_PIPELINE_H
#include "sourceGeneration.h"
namespace usdGen::vulkan {
// Complete Grow transaction: geometry dispatch and named-plane resampling
// consume the same immutable input. Neither dispatch reads geometry on host.
class GrowVkPipeline final {
public:
    class Candidate;
    static std::shared_ptr<GrowVkPipeline> Create(
        std::shared_ptr<CurveGrowPipeline>, std::shared_ptr<ResampleVkPipeline>);
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const VulkanSourceGeneration>,
        CurveGrowPipeline::Controls const&, std::string const& targetPlane,
        uint64_t valueVersion, VkResult* = nullptr,
        CurveGrowPipeline::BeforeSubmit = {});
private:
    std::shared_ptr<CurveGrowPipeline> geometry_;
    std::shared_ptr<ResampleVkPipeline> named_;
};
class GrowVkPipeline::Candidate final {
public:
    ~Candidate();
    VkResult Poll(uint32_t* semanticStatus = nullptr);
    bool succeeded() const noexcept { return bool(output_); }
    std::shared_ptr<const VulkanSourceGeneration> output() const noexcept { return output_; }
    void Quarantine() noexcept;
private:
    friend class GrowVkPipeline;
    std::shared_ptr<const VulkanSourceGeneration> input_, output_;
    std::unique_ptr<CurveGrowPipeline::Candidate> geometry_;
    std::unique_ptr<ResampleVkPipeline::Candidate> named_;
    uint32_t target_ = 0;
    uint64_t version_ = 0;
    uint32_t semantic_ = 0;
    bool proved_ = false;
};
} // namespace usdGen::vulkan
#endif
