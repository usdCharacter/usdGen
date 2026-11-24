// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// rbfVkDeformPipeline.h — gap-config RBF deform pipeline.
//
// Runs admitted DeformPipeline::BeginInfo sample sets through the device
// RBF solver and field-aware apply shader. The production source job uses
// this pipeline for every Deform; DeformPipeline remains a synchronous-fit
// compatibility surface for callers that explicitly inject it.
//
// The Begin/Candidate/Poll/Quarantine shape mirrors DeformPipeline
// exactly (including DeformSemantic reuse and the solve-before-empty-
// early-out order) so the WithDeform integration is a mechanical
// overload. Groom/primitive/point fields use the same checked ABI as the
// host-fit pipeline, including ownership through asynchronous completion.
#pragma once

#include "usdGen/vulkan/deformPipeline.h"
#include "usdGen/vulkan/rbfVkBinding.h"

#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

class RbfVkDeformPipeline final : public std::enable_shared_from_this<RbfVkDeformPipeline> {
public:
    class Candidate;
    using BeforeSubmit = DeformPipeline::BeforeSubmit;

    static std::shared_ptr<RbfVkDeformPipeline> Create(
        std::shared_ptr<DeviceContext> context,
        RbfVkBindingSpirv const& bindingSpirv,
        std::vector<uint32_t> const& applySpirv,
        VkResult* result = nullptr);

    ~RbfVkDeformPipeline();

    RbfVkDeformPipeline(RbfVkDeformPipeline const&) = delete;
    RbfVkDeformPipeline& operator=(RbfVkDeformPipeline const&) = delete;

    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Submit only device extent. Candidate Poll proves each phase; Advance
    // submits factorization, posed solve, evaluation, then apply.
    // `info` is the shared DeformPipeline::BeginInfo, including device fields.
    std::unique_ptr<Candidate> Begin(
        DeformPipeline::BeginInfo info,
        VkResult* result = nullptr,
        DeformSemantic* semantic = nullptr,
        BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit RbfVkDeformPipeline(std::shared_ptr<Native> native);
    std::shared_ptr<Native> native_;
};

class RbfVkDeformPipeline::Candidate final {
public:
    struct Output {
        std::shared_ptr<const ChargedBuffer> points;
    };

    ~Candidate();

    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // Poll never submits or waits. After a proved phase, Advance requires
    // a fresh admission hook and submits at most one subsequent phase.
    VkResult Poll(DeformSemantic* semantic = nullptr);
    bool Advance(BeforeSubmit beforeSubmit = {}, VkResult* result = nullptr);
    bool done() const noexcept;
    bool pending() const noexcept;

    Output output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    uint32_t pointCount() const noexcept;
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;

    friend class RbfVkDeformPipeline;
};

} // namespace usdGen::vulkan
