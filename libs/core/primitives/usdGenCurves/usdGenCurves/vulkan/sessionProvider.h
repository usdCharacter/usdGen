// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Injected Session provider for the narrow Vulkan source/width route.
#ifndef USDGEN_VULKAN_SESSION_PROVIDER_H
#define USDGEN_VULKAN_SESSION_PROVIDER_H

#include "usdGen/api/ext/device/sessionDeviceProvider.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenCurves/vulkan/planExecutor.h"
#include "usdGen/api/epoch.h"

#include <atomic>
#include <array>
#include <memory>
#include <string>
#include "usdGenCurves/vulkan/export.h"

namespace usdGen::vulkan {

class USDGEN_CURVES_VULKAN_INTERNAL_API VulkanSessionProvider final : public UsdGenSessionDeviceProvider,
                                    public std::enable_shared_from_this<VulkanSessionProvider> {
public:
    ~VulkanSessionProvider() override;
    struct CreateInfo {
        std::shared_ptr<VulkanPlanExecutor> executor;
        std::shared_ptr<DeviceContext> context;
        // The same per-device serialized owner retained by the executor's
        // domain; it is used only for bounded ingress admission.
        std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
        UsdGenSessionDeviceIdentity identity;
        // At most this many concurrently-live compiled handles are admitted.
        // Expired records are atomically reclaimed; live handles are never
        // evicted from provenance while they may be submitted.
        uint32_t maxRegisteredPlans = 256;
    };

    static std::shared_ptr<VulkanSessionProvider> Create(CreateInfo,
                                                          std::string* reason = nullptr);

    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const&,
                                           UsdGenDiagnostics*) const override;
    UsdGenSessionDeviceIdentity Identity() const noexcept override;
    bool NotifyContextLost(UsdGenSessionDeviceIdentity const&) noexcept override;
    bool Submit(UsdGenSessionDeviceRequest) override;
    void Shutdown() noexcept override;

private:
    explicit VulkanSessionProvider(CreateInfo) noexcept;
    bool Matches(UsdGenSessionDeviceIdentity const&) const noexcept;
    struct Pending;
    void Deliver(std::shared_ptr<Pending> const&, std::shared_ptr<const UsdGenDeviceGeneration>,
                 std::exception_ptr) noexcept;

    CreateInfo info_;
    std::atomic<uint64_t> contextEpoch_{0};
    std::atomic<bool> lost_{false};
    std::atomic<bool> closing_{false};
    struct PlanRecord;
    static constexpr uint32_t kPlanSlots = 256;
    mutable std::array<UsdGenAtomicRef<PlanRecord>, kPlanSlots> plans_;
    uint32_t maxRegisteredPlans_ = 0;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_SESSION_PROVIDER_H
