// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_TOPOLOGY_PIPELINE_H
#define USDGEN_VULKAN_TOPOLOGY_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {
// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion
// (only the fence status is probed; the compare result is read back through a
// host-visible result buffer once the fence proves). Mirrors WidthPipeline's
// lifecycle: Create(context,spirv,result) / Begin(...)->unique_ptr<Candidate>
// / Poll / succeeded / output / inputOwner / context.
class TopologyPipeline final : public std::enable_shared_from_this<TopologyPipeline> {
public:
    class Candidate;
    // Owner-side admission checkpoint immediately before the first native
    // submit, identical in contract to WidthPipeline::BeforeSubmit.
    using BeforeSubmit = std::function<bool()>;
    // SPIR-V must be the trusted, externally spirv-val-validated
    // topologyCompare.comp module with the documented five-binding/16-byte
    // push-constant ABI: b0 aOffs u32[curvesA+1], b1 aIds uvec2[curvesA],
    // b2 bOffs u32[curvesB+1], b3 bIds uvec2[curvesB],
    // b4 result {error u32, equal u32}.
    static std::shared_ptr<TopologyPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~TopologyPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin shape-check: each offsets buffer must be curvesX+1 u32, each
    // ids buffer curvesX u64 (8-byte) lanes, counts within uint32 and curvesX
    // below UINT32_MAX/pointCount. Compares only layout identity; publishes a
    // scalar result. BeforeSubmit runs at most once for a fully recorded
    // non-empty compare.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> offsetsA,
        std::shared_ptr<const ChargedBuffer> idsA,
        uint32_t curvesA, uint32_t pointsA,
        std::shared_ptr<const ChargedBuffer> offsetsB,
        std::shared_ptr<const ChargedBuffer> idsB,
        uint32_t curvesB, uint32_t pointsB,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit TopologyPipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginInternal(
        std::shared_ptr<const ChargedBuffer>, std::shared_ptr<const ChargedBuffer>,
        uint32_t, uint32_t, std::shared_ptr<const ChargedBuffer>,
        std::shared_ptr<const ChargedBuffer>, uint32_t, uint32_t,
        VkResult*, BeforeSubmit);
    std::shared_ptr<Native> native_;
};

class TopologyPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained for the compare. *semanticStatus
    // then carries the error code (0 == well-formed). EqualResult() is valid
    // only when semanticStatus == 0. VK_NOT_READY preserves the candidate.
    VkResult Poll(uint32_t* semanticStatus = nullptr);
    // Layout-identity outcome. Valid after a proved Poll with semantic==0.
    bool EqualResult() const noexcept;
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class TopologyPipeline;
};
} // namespace usdGen::vulkan
#endif // USDGEN_VULKAN_TOPOLOGY_PIPELINE_H