// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_LENGTH_COMPACTION_PIPELINE_H
#define USDGEN_VULKAN_LENGTH_COMPACTION_PIPELINE_H

#include "chargedBuffer.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/deviceGeneration.h"

#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {
class VulkanSourceGeneration;

class LengthCompactionPipeline final
    : public std::enable_shared_from_this<LengthCompactionPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;

    static std::shared_ptr<LengthCompactionPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&, VkResult* = nullptr);
    // Opt in only with the updated trusted phase-zero external-keep ABI.
    static std::shared_ptr<LengthCompactionPipeline> CreateWithKeep(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&, VkResult* = nullptr);
    bool HasKeep() const noexcept;
    ~LengthCompactionPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const VulkanSourceGeneration>, float threshold,
        VkResult* = nullptr, BeforeSubmit = {});

    // Reuse the device-proven Length keep mask. No threshold or envelope
    // reevaluation occurs after the transform; the mask is retained to completion.
    std::unique_ptr<Candidate> BeginWithKeep(std::shared_ptr<const VulkanSourceGeneration>,
        std::shared_ptr<const ChargedBuffer> keep, VkResult* = nullptr, BeforeSubmit = {});
private:
    std::unique_ptr<Candidate> BeginImpl(std::shared_ptr<const VulkanSourceGeneration>, float,
        std::shared_ptr<const ChargedBuffer>, bool, VkResult*, BeforeSubmit);
    struct Native;
    explicit LengthCompactionPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class LengthCompactionPipeline::Candidate final {
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

    VkResult PollCounts(uint32_t* semanticStatus = nullptr);
    bool BeginScatter(VkResult* = nullptr, BeforeSubmit = {});
    VkResult PollScatter(uint32_t* semanticStatus = nullptr);
    uint32_t curveCount() const noexcept;
    uint32_t pointCount() const noexcept;
    std::shared_ptr<const VulkanSourceGeneration> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    bool succeeded() const noexcept;
    void Quarantine() noexcept;
    std::vector<OutputPlane> const& outputs() const noexcept;
    std::vector<UsdGenChunkDesc> const& chunks() const noexcept;
    std::vector<UsdGenDeviceTileMetadata> const& tiles() const noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class LengthCompactionPipeline;
};
} // namespace usdGen::vulkan

#endif
