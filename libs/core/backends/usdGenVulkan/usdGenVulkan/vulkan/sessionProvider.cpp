// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "sessionProvider.h"

#include "executionPlan.h"

#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace usdGen::vulkan {
namespace {
std::exception_ptr Failure(char const* message) noexcept {
    try { return std::make_exception_ptr(std::runtime_error(message)); }
    catch (...) { return std::current_exception(); }
}
} // namespace

struct VulkanSessionProvider::PlanRecord final : UsdGenRefCounted {
    std::weak_ptr<const UsdGenExecutionPlanHandle> handle;
};

VulkanSessionProvider::VulkanSessionProvider(CreateInfo info) noexcept
    : info_(std::move(info)), contextEpoch_(info_.identity.contextEpoch),
      maxRegisteredPlans_(info_.maxRegisteredPlans) {}

VulkanSessionProvider::~VulkanSessionProvider() = default;

std::shared_ptr<VulkanSessionProvider> VulkanSessionProvider::Create(
    CreateInfo info, std::string* reason) {
    if (!info.executor || !info.context || !info.queueOwner || !info.maxRegisteredPlans ||
        info.maxRegisteredPlans > kPlanSlots ||
        info.identity.backend != UsdGenDeviceBackend::Vulkan ||
        info.identity.capabilityVersion != 1 ||
        !info.identity.logicalContextIdentity ||
        // The neutral generation publishes the factory physical index. The
        // separate UUID and logical-context token retain pool/cache isolation.
        info.identity.deviceIndex != info.context->physicalIndex() ||
        info.identity.physicalDeviceUuid != info.context->deviceUUID() ||
        !info.executor->createInfo().domain ||
        info.executor->createInfo().widthPipeline->context() != info.context ||
        info.executor->createInfo().domain->QueueOwnerForIngress() != info.queueOwner) {
        if (reason) *reason = "Vulkan Session provider requires exact context, UUID, owner, and capability identity";
        return {};
    }
    try { return std::shared_ptr<VulkanSessionProvider>(new VulkanSessionProvider(std::move(info))); }
    catch (...) { if (reason) *reason = "Vulkan Session provider allocation failed"; return {}; }
}

UsdGenSessionDevicePlanHandle VulkanSessionProvider::Compile(
    UsdGenGraphDesc const& desc, UsdGenDiagnostics* diagnostics) const {
    if (closing_.load(std::memory_order_acquire) || lost_.load(std::memory_order_acquire)) return {};
    auto handle = CompileVulkanSourceWidthPlan(desc, diagnostics);
    if (!handle) return {};
    auto typed = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
    if (!typed) return {};
    UsdGenRef<PlanRecord> record;
    try {
        record = UsdGenMakeRef<PlanRecord>();
        record->handle = handle;
    } catch (...) { return {}; }
    // Each record snapshot (epoch-published) protects the slot while Submit
    // verifies it. Replacing only an empty/expired record cannot invalidate a
    // concurrently live handle; displaced records retire through the epochs.
    try {
        for (uint32_t i = 0; i != maxRegisteredPlans_; ++i) {
            auto prior = plans_[i].Load();
            if (prior && !prior->handle.expired()) continue;
            if (plans_[i].CompareExchange(prior, record)) return handle;
        }
    } catch (...) {}
    return {};
}

UsdGenSessionDeviceIdentity VulkanSessionProvider::Identity() const noexcept {
    auto result = info_.identity;
    result.contextEpoch = contextEpoch_.load(std::memory_order_acquire);
    return result;
}

bool VulkanSessionProvider::Matches(UsdGenSessionDeviceIdentity const& identity) const noexcept {
    return identity == Identity() && !lost_.load(std::memory_order_acquire) &&
        !closing_.load(std::memory_order_acquire);
}

bool VulkanSessionProvider::NotifyContextLost(UsdGenSessionDeviceIdentity const& lost) noexcept {
    if (lost != Identity()) return false;
    if (lost_.exchange(true, std::memory_order_acq_rel)) return false;
    uint64_t current = contextEpoch_.load(std::memory_order_acquire);
    if (current == std::numeric_limits<uint64_t>::max()) return false;
    if (!contextEpoch_.compare_exchange_strong(current, current + 1,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire)) return false;
    // A DeviceContext is immutable. A replacement context/provider is needed
    // after loss; advancing the epoch makes stale requests observably invalid.
    return true;
}

struct VulkanSessionProvider::Pending final : std::enable_shared_from_this<VulkanSessionProvider::Pending> {
    explicit Pending(UsdGenSessionDeviceRequest value) : request(std::move(value)) {}
    UsdGenSessionDeviceRequest request;
    std::shared_ptr<UsdGenSessionDeviceResult> result;
    std::exception_ptr superseded, rejected, cancelled, failed;
    std::atomic<bool> terminal{false};
    std::shared_ptr<VulkanSessionProvider> provider;
    // Deliberate cycle: a rejected return transport must not make native
    // ownership last-drop on its caller/transport thread.
    std::shared_ptr<Pending> quarantine;
    std::atomic<bool> quarantined{false};
    void Quarantine() noexcept {
        // The flag elects the one writer; nothing reads the cycle.
        if (quarantined.exchange(true, std::memory_order_acq_rel)) return;
        try { quarantine = shared_from_this(); } catch (...) {}
    }
};

void VulkanSessionProvider::Deliver(std::shared_ptr<Pending> const& pending,
                                    std::shared_ptr<const UsdGenDeviceGeneration> generation,
                                    std::exception_ptr error) noexcept {
    if (!pending || !pending->result) return;
    auto& result = *pending->result;
    // Identity and revision storage were frozen before ingress. In particular,
    // no intermediate-revision vector allocation is allowed in this noexcept
    // terminal path after native work has been admitted.
    result.generation = std::move(generation);
    result.error = error;
    // The supplied transport owns the pre-reserved Session return admission.
    // It receives only immutable data; this provider never reaches a cooker.
    bool accepted = false;
    try { accepted = pending->request.returnTransport(pending->result); } catch (...) {}
    if (!accepted) pending->Quarantine();
}

bool VulkanSessionProvider::Submit(UsdGenSessionDeviceRequest request) {
    if (!request.plan || !request.returnTransport || !Matches(request.identity) ||
        request.cancellation.Superseded()) return false;
    std::shared_ptr<const VulkanSourceWidthPlan> typed;
    for (uint32_t i = 0; i != maxRegisteredPlans_; ++i) {
        auto record = plans_[i].Load();
        if (!record) continue;
        auto known = record->handle.lock();
        if (known && known.get() == request.plan.get()) {
            typed = std::static_pointer_cast<const VulkanSourceWidthPlan>(known->Payload());
            break;
        }
    }
    if (request.plan->Backend() != UsdGenExecutionBackend::Vulkan || !typed ||
        (request.authoritativeRevisions &&
         !request.authoritativeRevisions->IsStrictlyOrdered(typed->IntermediateCount()))) return false;
    // Normalize the optional frozen-plan route once, before reserving either
    // owner. Every terminal result therefore carries the exact revisions that
    // native publication was asked to produce.
    if (!request.authoritativeRevisions) {
        if (typed->IntermediateCount() || typed->ValueVersion() == UINT64_MAX) return false;
        request.authoritativeRevisions = UsdGenSessionDeviceRevisions{
            typed->TopologyVersion(), typed->ValueVersion(), typed->ValueVersion() + 1};
    }
    auto ticket = info_.queueOwner->ReserveCommandTicket();
    if (!ticket) return false;
    std::shared_ptr<Pending> pending;
    try {
        pending = std::make_shared<Pending>(std::move(request));
        pending->result = std::make_shared<UsdGenSessionDeviceResult>();
        pending->result->identity = pending->request.identity;
        pending->result->revisions = *pending->request.authoritativeRevisions;
        pending->superseded = Failure("Vulkan Session request superseded");
        pending->rejected = Failure("Vulkan Session native admission rejected");
        pending->cancelled = Failure("Vulkan Session device ingress cancelled");
        pending->failed = Failure("Vulkan Session provider owner admission failed");
        pending->provider = shared_from_this();
    }
    catch (...) { return false; }
    auto self = shared_from_this();
    bool posted = false;
    try {
        posted = info_.queueOwner->PostCommand(std::move(ticket),
            [self, pending, typed]() mutable {
                try {
                auto const& request = pending->request;
                if (!self->Matches(request.identity) || request.cancellation.Superseded()) {
                    if (!pending->terminal.exchange(true, std::memory_order_acq_rel))
                        self->Deliver(pending, {}, pending->superseded);
                    return;
                }
                VulkanPlanExecutor::Request native;
                native.context = self->info_.context;
                native.capabilityVersion = request.identity.capabilityVersion;
                native.generation = request.publicationGeneration;
                native.tool = request.tool;
                native.cancellation = request.cancellation;
                native.requestLifetime = pending;
                native.authoritativeRevisions = request.authoritativeRevisions;
                native.evaluationFrame = request.evaluationFrame;
                if (!self->info_.executor->Submit(std::move(typed), std::move(native),
                    [self, pending](
                        std::shared_ptr<const UsdGenDeviceGeneration> generation,
                        std::exception_ptr error) mutable {
                        if (!pending->terminal.exchange(true, std::memory_order_acq_rel))
                            self->Deliver(pending, std::move(generation), error);
                        if (!self->info_.queueOwner->IsExecutingOwner()) pending->Quarantine();
                    })) {
                    if (!pending->terminal.exchange(true, std::memory_order_acq_rel))
                        self->Deliver(pending, {}, pending->rejected);
                }
                } catch (...) {
                    if (!pending->terminal.exchange(true, std::memory_order_acq_rel))
                        self->Deliver(pending, {}, pending->failed);
                }
            },
            [self, pending]() mutable {
                if (!pending->terminal.exchange(true, std::memory_order_acq_rel))
                    self->Deliver(pending, {}, pending->cancelled);
                if (!self->info_.queueOwner->IsExecutingOwner()) pending->Quarantine();
            });
    } catch (...) {}
    return posted;
}

void VulkanSessionProvider::Shutdown() noexcept {
    closing_.store(true, std::memory_order_release);
}

} // namespace usdGen::vulkan
