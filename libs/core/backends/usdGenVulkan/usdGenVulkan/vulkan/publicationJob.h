// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_PUBLICATION_JOB_H
#define USDGEN_VULKAN_PUBLICATION_JOB_H

#include "deviceGenerationAdapter.h"
#include "sourceWidthJob.h"
#include "usdGen/executionBackend.h"

#include <atomic>
#include <memory>
#include <string>

namespace usdGen::vulkan {

// Owner-bound publication of a proved native source-width result.  This is a
// bridge only: it neither selects Vulkan as an execution backend nor provides
// a CPU/readback fallback.
class VulkanPublicationJob final
    : public std::enable_shared_from_this<VulkanPublicationJob> {
public:
    enum class State { Created, Running, Ready, Superseded, Failed, LostProof };

    struct CreateInfo {
        VulkanSourceWidthJob::CreateInfo nativeJob;
        std::shared_ptr<VulkanGenerationAdapterDomain> domain;
        uint64_t generation = 0;
        UsdGenDeviceToolMetadata tool;
    };

    static std::shared_ptr<VulkanPublicationJob> Create(CreateInfo,
                                                          std::string* reason = nullptr);

    // Single-attempt owner-frame admission. `owner` must be executing and be
    // the exact queue owner held by domain and nativeJob.completionService.
    // Accepted completion is exactly-once; a rejected Start has admitted
    // neither native work nor a callback.
    bool Start(UsdGenExecutionPipeline& owner,
               UsdGenExecutionPipeline::Cancellation cancellation,
               UsdGenExecutionCompletion completion);
    void SuppressPublication() noexcept;
    State state() const noexcept { return state_.load(std::memory_order_acquire); }
    VulkanSourceWidthJob::State nativeState() const noexcept {
        return nativeState_.load(std::memory_order_acquire);
    }

private:
    struct Relay;
    explicit VulkanPublicationJob(CreateInfo) noexcept;
    void NativeComplete(std::shared_ptr<const VulkanSourceGeneration>,
                        VulkanSourceWidthJob::State, VkResult) noexcept;
    void Settle(std::shared_ptr<const UsdGenDeviceGeneration>, State,
                std::exception_ptr, bool offOwner) noexcept;

    CreateInfo info_;
    std::shared_ptr<VulkanSourceWidthJob> native_;
    std::shared_ptr<Relay> relay_;
    std::atomic<State> state_{State::Created};
    std::atomic<VulkanSourceWidthJob::State> nativeState_{
        VulkanSourceWidthJob::State::Created};
    std::atomic<bool> started_{false};
};

} // namespace usdGen::vulkan

#endif
