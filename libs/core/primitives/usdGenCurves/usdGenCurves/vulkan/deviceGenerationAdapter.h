// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_DEVICE_GENERATION_ADAPTER_H
#define USDGEN_VULKAN_DEVICE_GENERATION_ADAPTER_H

#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGen/sharedCell.h"
#include "usdGen/deviceGeneration.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace usdGen::vulkan {

class VulkanGenerationAdapterDomainAccess;

class VulkanGenerationAdapterDomain final
    : public std::enable_shared_from_this<VulkanGenerationAdapterDomain> {
public:
    struct CreateInfo {
        std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
        std::shared_ptr<VulkanCompletionService> completionService;
    };
    static std::shared_ptr<VulkanGenerationAdapterDomain> Create(CreateInfo,
                                                                  std::string* reason = nullptr);
    // A bounded active-operation hold for sibling Vulkan jobs. It snapshots
    // the exact owner/service pair and prevents Close from detaching the
    // pipeline. It is move-only; the bound owner must Retire after proof (or
    // a pre-submit rollback). Unretired destruction deliberately fails closed.
    class OperationLease final {
    public:
        OperationLease() = default;
        ~OperationLease();
        OperationLease(OperationLease const&) = delete;
        OperationLease& operator=(OperationLease const&) = delete;
        OperationLease(OperationLease&&) noexcept = default;
        OperationLease& operator=(OperationLease&&) noexcept;
        explicit operator bool() const noexcept { return bool(domain_); }
        bool Retire() noexcept;
        bool CancelBeforeSubmit() noexcept { return Retire(); }
        std::shared_ptr<UsdGenExecutionPipeline> const& QueueOwner() const noexcept { return owner_; }
        std::shared_ptr<VulkanCompletionService> const& CompletionService() const noexcept { return service_; }
    private:
        OperationLease(std::shared_ptr<VulkanGenerationAdapterDomain> domain,
                       std::shared_ptr<UsdGenExecutionPipeline> owner,
                       std::shared_ptr<VulkanCompletionService> service) noexcept
            : domain_(std::move(domain)), owner_(std::move(owner)), service_(std::move(service)) {}
        std::shared_ptr<VulkanGenerationAdapterDomain> domain_;
        std::shared_ptr<UsdGenExecutionPipeline> owner_;
        std::shared_ptr<VulkanCompletionService> service_;
        friend class VulkanGenerationAdapterDomain;
    };
    OperationLease AcquireOperation() noexcept;
    // Read-only ingress identity. Native admission still requires an
    // OperationLease on this exact owner.
    std::shared_ptr<UsdGenExecutionPipeline> QueueOwnerForIngress() const noexcept;
    // Any-thread loss/shutdown latch. Existing leases retain their proof
    // route; new leases fail immediately. Close performs external retirement.
    void StopAdmission() noexcept;
    // External-only. It closes lease admission, drains the owner externally,
    // and detaches its pipeline reference only after every adapter consumer
    // has retired. It never closes the shared completion service. Failed or
    // premature close deliberately preserves self-retention for a later retry.
    bool Close() noexcept;
    // After a premature Close, transfer the sole external retirement boundary
    // here. It runs once on an external thread after consumer proofs retire.
    // False rejects a duplicate/invalid registration without retaining it.
    bool RetireWhenIdle(std::function<void()> externalRetirement) noexcept;
    bool IsClosing() const noexcept;
    bool IsClosed() const noexcept;
private:
    VulkanGenerationAdapterDomain() noexcept = default;
public:
    ~VulkanGenerationAdapterDomain() { delete deferredRetirement_.load(std::memory_order_acquire); }
private:
    bool BeginConsumer(std::shared_ptr<UsdGenExecutionPipeline>* owner,
                       std::shared_ptr<VulkanCompletionService>* service) noexcept;
    void RetireConsumer() noexcept;
    void ScheduleRetirement() noexcept;
    // Published at Create, cleared once by Close; readers copy under epochs.
    UsdGenSharedCell<UsdGenExecutionPipeline> queueOwner_;
    UsdGenSharedCell<VulkanCompletionService> completionService_;
    // Written at Create and by the one Close that wins draining_.
    std::shared_ptr<VulkanGenerationAdapterDomain> selfRetention_;
    // Owned by whoever exchanges it out; never read in place.
    std::atomic<std::function<void()>*> deferredRetirement_{nullptr};
    std::atomic<uint64_t> activeConsumers_{0};
    std::atomic<bool> closing_{false}, closed_{false}, draining_{false};
    friend class VulkanDeviceGenerationAdapter;
    friend class VulkanGenerationAdapterDomainAccess;
};

// Bridges an already-proved immutable Vulkan source snapshot into the neutral
// device-generation contract.  It deliberately does not advertise Vulkan as
// an execution backend or create a Vulkan context.
class VulkanDeviceGenerationAdapter final {
public:
    struct CreateInfo {
        std::shared_ptr<const VulkanSourceGeneration> source;
        std::shared_ptr<VulkanGenerationAdapterDomain> domain;
        // Must exactly encode source->context()->computeQueue(), never an
        // arbitrary application stream identifier.
        UsdGenDeviceStream queueStream = 0;
        // Publication identity uses the trusted context physicalIndex.  The
        // source and completion service must share that exact DeviceContext
        // (and therefore its factory-captured device UUID), rather than just
        // coincident numeric device indices.
        uint64_t generation = 0;
        UsdGenDeviceToolMetadata tool;
    };

    static std::shared_ptr<const UsdGenDeviceGeneration> Create(
        CreateInfo, std::string* reason = nullptr);
};

// Backend-private native view paired with the neutral lease.  `planes` is a
// public-channel copy only: source-frame T/B/N buffers are never exposed.
// The buffers remain valid exactly while `lease` remains valid.
class VulkanGenerationLease final {
public:
    VulkanGenerationLease() = default;
    explicit operator bool() const noexcept { return bool(lease_); }
    UsdGenDeviceStatus WaitUntilReady() const noexcept {
        return lease_.WaitUntilReady();
    }
    void Complete() noexcept { planes_.clear(); lease_.Complete(); }
    std::vector<VulkanSourceGeneration::PlaneView> const& Planes() const noexcept {
        return lease_ ? planes_ : NoPlanes();
    }
private:
    // Namespace-scope empty list in the .cpp: no guarded local static.
    static std::vector<VulkanSourceGeneration::PlaneView> const& NoPlanes() noexcept;
    VulkanGenerationLease(UsdGenDeviceLease lease,
                          std::vector<VulkanSourceGeneration::PlaneView> planes) noexcept
        : lease_(std::move(lease)), planes_(std::move(planes)) {}
    UsdGenDeviceLease lease_;
    std::vector<VulkanSourceGeneration::PlaneView> planes_;
    friend VulkanGenerationLease AcquireVulkanGeneration(
        std::shared_ptr<const UsdGenDeviceGeneration> const&, UsdGenDeviceStream) noexcept;
};

// Typed native consumer access for an adapter-created generation. Retains the
// generation and its domain; callers schedule native queue work on queueOwner
// and acquire a VulkanGenerationLease there before using any plane buffers.
// Access does not authorize consumption after domain closure.
struct VulkanGenerationAccess {
    std::shared_ptr<const UsdGenDeviceGeneration> generation;
    std::shared_ptr<DeviceContext> context;
    std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
    std::shared_ptr<VulkanGenerationAdapterDomain> domain;
    explicit operator bool() const noexcept {
        return generation && context && queueOwner && domain;
    }
};
VulkanGenerationAccess GetVulkanGenerationAccess(
    std::shared_ptr<const UsdGenDeviceGeneration> const&) noexcept;

// Acquires a Vulkan-native view only from an adapter-created Vulkan
// generation.  It rejects foreign backends/owners and stream tokens.
VulkanGenerationLease AcquireVulkanGeneration(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    UsdGenDeviceStream queueStream) noexcept;

} // namespace usdGen::vulkan
#endif
