// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_RESAMPLE_VK_PIPELINE_H
#define USDGEN_VULKAN_RESAMPLE_VK_PIPELINE_H

#include "chargedBuffer.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/deviceGeneration.h"

#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {
class VulkanSourceGeneration;

// Indexed-CV resampling for complete Vulkan source packets. Vulkan port of
// gpu/curveResample.cu (target >= 2 uniform output) plus the resample-mode
// half of gpu/namedChannelTopology.cu: Point/Float32 planes interpolate with
// the same double-precision weights, Point/Int32 planes take the nearest
// source CV, Primitive planes and stable IDs copy verbatim (rootPrim/rootUV
// keep their CUDA content validation), Groom planes alias, and curveOffsets
// are re-emitted uniformly by the host. Every curve keeps its stable ID and
// ordinal; the point count becomes curveCount * target.
//
// Validation runs on-device before any payload write (a compute barrier plus
// an in-shader status gate), so a malformed input fails closed with a nonzero
// semantic status instead of producing a partial packet.
class ResampleVkPipeline final
    : public std::enable_shared_from_this<ResampleVkPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;

    static std::shared_ptr<ResampleVkPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&, VkResult* = nullptr);
    ~ResampleVkPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // target is the uniform output CV count per curve (must be >= 2; 0/1 are
    // rejected, matching the CUDA Apply contract). Counts are known upfront,
    // so one submission covers validation plus every plane dispatch.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const VulkanSourceGeneration>, uint32_t target,
        VkResult* = nullptr, BeforeSubmit = {});

private:
    struct Native;
    explicit ResampleVkPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class ResampleVkPipeline::Candidate final {
public:
    struct OutputPlane {
        UsdGenDeviceChannelMetadata metadata;
        std::shared_ptr<const ChargedBuffer> owner;
        bool privateFrame = false;
        VkDeviceSize bytes = 0;
    };

    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    VkResult Poll(uint32_t* semanticStatus = nullptr);
    // Unchanged by resampling; equals the input curve count.
    uint32_t curveCount() const noexcept;
    // curveCount() * target; 0 when the input has no curves.
    uint32_t pointCount() const noexcept;
    uint32_t target() const noexcept;
    std::shared_ptr<const VulkanSourceGeneration> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    bool succeeded() const noexcept;
    void Quarantine() noexcept;
    // Empty until Poll proves success; Groom planes alias input owners.
    std::vector<OutputPlane> const& outputs() const noexcept;
    std::vector<UsdGenChunkDesc> const& chunks() const noexcept;
    std::vector<UsdGenDeviceTileMetadata> const& tiles() const noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class ResampleVkPipeline;
};
} // namespace usdGen::vulkan

#endif
