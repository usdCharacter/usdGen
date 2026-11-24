// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_WIDTH_PIPELINE_H
#define USDGEN_VULKAN_WIDTH_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <vector>

namespace usdGen::vulkan {
// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion.
class WidthPipeline final : public std::enable_shared_from_this<WidthPipeline> {
public:
    class Candidate;
    // Owner-side admission checkpoint immediately before the first native
    // submit. It runs at most once only for a fully recorded nonempty
    // candidate; invalid input, allocation/recording failures, and count==0
    // never invoke it. True precedes pending state and every native submit.
    // False or an exception rejects without arming or submitting the candidate.
    using BeforeSubmit = std::function<bool()>;
    // SPIR-V must be the trusted, externally spirv-val-validated width.comp
    // module with the documented three-binding/12-byte push-constant ABI.
    static std::shared_ptr<WidthPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~WidthPipeline();
    // Exact native identity is available before candidate allocation/submit.
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer> input,
        uint32_t count, float width, uint32_t replace, VkResult* result = nullptr,
        BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit WidthPipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginInternal(std::shared_ptr<const ChargedBuffer>,
        uint32_t, float, uint32_t, VkResult*, BeforeSubmit);
    std::shared_ptr<Native> native_;
};

class WidthPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained; semanticStatus must additionally
    // be zero before output is exposed. VK_NOT_READY preserves the candidate.
    VkResult Poll(uint32_t* semanticStatus = nullptr);
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    uint32_t count() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class WidthPipeline;
};
} // namespace usdGen::vulkan
#endif
