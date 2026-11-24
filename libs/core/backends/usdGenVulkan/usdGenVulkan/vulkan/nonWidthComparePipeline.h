// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_NON_WIDTH_COMPARE_PIPELINE_H
#define USDGEN_VULKAN_NON_WIDTH_COMPARE_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {
class VulkanSourceGeneration;
// Queue-owner-confined native byte equality proof for immutable non-width
// packets. Host admission checks semantic metadata; payloads (including
// private source frames and stride padding) are compared on the GPU, exactly
// by bits, not approximate floating-point equality. Only scalar status is read
// back. Both predecessor generations are retained for the entire proof.
class NonWidthComparePipeline final : public std::enable_shared_from_this<NonWidthComparePipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    // Trusted packaged SPIR-V ABI: three storage buffers (left/right/status),
    // uint32 word-count push constant, local size 256. No shader reflection.
    static std::shared_ptr<NonWidthComparePipeline> Create(std::shared_ptr<DeviceContext>,
        std::vector<uint32_t> const&, VkResult* = nullptr);
    ~NonWidthComparePipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Always submits a status job, including empty packets. BeforeSubmit must
    // reserve/mark the caller's completion watch before native submission.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const VulkanSourceGeneration>,
        std::shared_ptr<const VulkanSourceGeneration>, VkResult* = nullptr, BeforeSubmit = {});
private:
    struct Native;
    explicit NonWidthComparePipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};
class NonWidthComparePipeline::Candidate final {
public:
    enum class Status { NotReady, Ready, LostProof };
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS/Ready means completion proved; succeeded() distinguishes
    // equality from clean inequality. Lost proof is sticky and quarantined.
    VkResult Poll(Status* = nullptr);
    bool succeeded() const noexcept;
    std::shared_ptr<const VulkanSourceGeneration> leftOwner() const noexcept;
    std::shared_ptr<const VulkanSourceGeneration> rightOwner() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class NonWidthComparePipeline;
};
} // namespace usdGen::vulkan
#endif
