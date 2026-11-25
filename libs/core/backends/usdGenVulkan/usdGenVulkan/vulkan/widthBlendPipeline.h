// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_WIDTH_BLEND_PIPELINE_H
#define USDGEN_VULKAN_WIDTH_BLEND_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <vector>

namespace usdGen::vulkan {
class WidthBlendPipeline final : public std::enable_shared_from_this<WidthBlendPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<WidthBlendPipeline> Create(std::shared_ptr<DeviceContext>,
        std::vector<uint32_t> const&, VkResult* = nullptr);
    ~WidthBlendPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer>,
        std::shared_ptr<const ChargedBuffer>, uint32_t, float, VkResult* = nullptr,
        BeforeSubmit = {});
private:
    struct Native;
    explicit WidthBlendPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};
class WidthBlendPipeline::Candidate final {
public:
    ~Candidate(); Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    VkResult Poll(uint32_t* = nullptr);
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    std::shared_ptr<const ChargedBuffer> leftOwner() const noexcept;
    std::shared_ptr<const ChargedBuffer> rightOwner() const noexcept;
    uint32_t count() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    bool succeeded() const noexcept;
    void Quarantine() noexcept;
private:
    struct State; explicit Candidate(std::shared_ptr<State>); std::shared_ptr<State> state_;
    friend class WidthBlendPipeline;
};
} // namespace usdGen::vulkan
#endif
