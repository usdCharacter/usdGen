// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "publicationJob.h"

#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace usdGen::vulkan {

struct VulkanPublicationJob::Relay final
    : public std::enable_shared_from_this<VulkanPublicationJob::Relay> {
    UsdGenExecutionCompletion completion;
    std::shared_ptr<VulkanPublicationJob> job;
    // The operation lease retains the exact domain/owner/service identity
    // while native work is live, preventing Close from detaching the queue.
    VulkanGenerationAdapterDomain::OperationLease operation;
    std::exception_ptr failed;
    std::exception_ptr superseded;
    std::exception_ptr lostProof;
    std::shared_ptr<Relay> quarantine;
    std::atomic<bool> quarantined{false};
    std::atomic<bool> settled{false};

    void RetainForLostProof() noexcept {
        // The flag elects the one writer; nothing reads the cycle.
        if (quarantined.exchange(true, std::memory_order_acq_rel)) return;
        try { quarantine = shared_from_this(); } catch (...) {}
    }
};

VulkanPublicationJob::VulkanPublicationJob(CreateInfo info) noexcept
    : info_(std::move(info)) {}

std::shared_ptr<VulkanPublicationJob>
VulkanPublicationJob::Create(CreateInfo info, std::string* reason)
{
    if (!info.domain || !info.nativeJob.completionService || !info.nativeJob.preparer ||
        !info.nativeJob.source.context ||
        info.nativeJob.completionService->context() != info.nativeJob.source.context) {
        if (reason) *reason = "Vulkan publication requires exact native context, service, and preparer";
        return {};
    }
    // Publication needs these two exact identity anchors at Start, but must
    // not duplicate the native job's CPU capture or request lifetime.
    auto context = info.nativeJob.source.context;
    auto service = info.nativeJob.completionService;
    std::string nativeReason;
    auto native = VulkanSourceWidthJob::Create(std::move(info.nativeJob), &nativeReason);
    if (!native) {
        if (reason) *reason = nativeReason.empty() ? "invalid Vulkan native publication job" : nativeReason;
        return {};
    }
    try {
        // Keep only admission identity in the outer wrapper. The inner job
        // exclusively retains source bytes, preparation inputs, and request.
        info.nativeJob = {};
        info.nativeJob.source.context = std::move(context);
        info.nativeJob.completionService = std::move(service);
        auto job = std::shared_ptr<VulkanPublicationJob>(new VulkanPublicationJob(std::move(info)));
        job->native_ = std::move(native);
        return job;
    } catch (std::bad_alloc const&) {
        if (reason) *reason = "Vulkan publication allocation failed";
        return {};
    }
}

bool VulkanPublicationJob::Start(UsdGenExecutionPipeline& owner,
                                 UsdGenExecutionPipeline::Cancellation cancellation,
                                 UsdGenExecutionCompletion completion)
{
    if (!completion || started_.exchange(true, std::memory_order_acq_rel) || !native_) return false;
    // Acquire before native Start: Close may otherwise detach the owner after
    // native admission but before its terminal callback retains publication.
    // Domain admission is intentionally owner-only.  It pairs the native
    // submission lifetime with the same serialized owner that can retire it.
    if (!owner.IsExecutingOwner()) return false;
    auto operation = info_.domain->AcquireOperation();
    if (!operation || operation.QueueOwner().get() != &owner ||
        operation.CompletionService() != info_.nativeJob.completionService ||
        operation.CompletionService()->context() != info_.nativeJob.source.context ||
        operation.CompletionService()->queueOwner() != &owner) {
        operation.CancelBeforeSubmit();
        return false;
    }
    try {
        auto relay = std::make_shared<Relay>();
        relay->completion = std::move(completion);
        // No allocation or construction on an off-owner loss route.
        relay->failed = std::make_exception_ptr(std::runtime_error("Vulkan publication failed"));
        relay->superseded = std::make_exception_ptr(std::runtime_error("Vulkan publication superseded"));
        relay->lostProof = std::make_exception_ptr(std::runtime_error("Vulkan publication lost proof"));
        // Transfer only after all potentially throwing relay setup is done:
        // failed setup can then explicitly cancel the local operation hold.
        relay->job = shared_from_this();
        relay->operation = std::move(operation);
        relay_ = std::move(relay);
    } catch (...) {
        operation.CancelBeforeSubmit();
        return false;
    }
    state_.store(State::Running, std::memory_order_release);
    try {
        auto self = shared_from_this();
        if (native_->Start(owner, std::move(cancellation),
            [self](std::shared_ptr<const VulkanSourceGeneration> source,
                   VulkanSourceWidthJob::State nativeState, VkResult status) noexcept {
                self->NativeComplete(std::move(source), nativeState, status);
            })) return true;
    } catch (...) {}
    // SourceWidthJob rejects before native work/callback admission. Roll the
    // operation lease back on this owner; Start remains single-attempt.
    auto relay = std::move(relay_);
    if (relay) {
        relay->operation.CancelBeforeSubmit();
        relay->job.reset();
    }
    State running = State::Running;
    state_.compare_exchange_strong(running, State::Created,
                                   std::memory_order_acq_rel);
    return false;
}

void VulkanPublicationJob::SuppressPublication() noexcept
{
    if (native_) native_->SuppressPublication();
}

void VulkanPublicationJob::NativeComplete(std::shared_ptr<const VulkanSourceGeneration> source,
                                          VulkanSourceWidthJob::State nativeState,
                                          VkResult status) noexcept
{
    nativeState_.store(nativeState, std::memory_order_release);
    auto relay = relay_;
    if (!relay) return;
    bool const offOwner = nativeState == VulkanSourceWidthJob::State::LostProof &&
        (!relay->operation.QueueOwner() || !relay->operation.QueueOwner()->IsExecutingOwner());
    if (nativeState == VulkanSourceWidthJob::State::Superseded) {
        Settle({}, State::Superseded, relay->superseded, offOwner); return;
    }
    if (nativeState == VulkanSourceWidthJob::State::LostProof) {
        // Lost native proof is unsafe even when delivery happened on owner:
        // retain the whole operation graph fail-closed.
        Settle({}, State::LostProof, relay->lostProof, true); return;
    }
    if (nativeState != VulkanSourceWidthJob::State::Ready || status != VK_SUCCESS || !source) {
        auto error = relay->failed;
        try {
            std::string phase;
            switch (native_->failurePhase()) {
            case VulkanSourceWidthJob::State::ExpressionPending: phase = "expression evaluation"; break;
            case VulkanSourceWidthJob::State::ExpressionHostPending: phase = "expression host step"; break;
            case VulkanSourceWidthJob::State::ExpressionPackPending: phase = "expression packing"; break;
            case VulkanSourceWidthJob::State::ExpressionWidthPending: phase = "expression Width"; break;
            default: phase = "native phase " + std::to_string(int(native_->failurePhase())); break;
            }
            error = std::make_exception_ptr(std::runtime_error(
                "Vulkan publication failed during " + phase + " (VkResult " + std::to_string(int(status)) + ")"));
        } catch (...) {}
        Settle({}, State::Failed, std::move(error), offOwner); return;
    }
    // The SourceWidthJob completion is owner-confined on all proved paths.
    // Adapter Create validates the exact retained domain/service/context again
    // before exposing the immutable neutral generation.
    try {
        VulkanDeviceGenerationAdapter::CreateInfo publication;
        publication.source = std::move(source);
        publication.domain = info_.domain;
        publication.queueStream = reinterpret_cast<UsdGenDeviceStream>(
            publication.source->context()->computeQueue());
        publication.generation = info_.generation;
        publication.tool = info_.tool;
        std::string reason;
        auto generation = VulkanDeviceGenerationAdapter::Create(std::move(publication), &reason);
        if (generation) { Settle(std::move(generation), State::Ready, {}, false); return; }
    } catch (...) {}
    Settle({}, State::Failed, relay->failed, false);
}

void VulkanPublicationJob::Settle(std::shared_ptr<const UsdGenDeviceGeneration> generation,
                                  State state, std::exception_ptr error,
                                  bool offOwner) noexcept
{
    auto relay = relay_;
    if (!relay || relay->settled.exchange(true, std::memory_order_acq_rel)) return;
    state_.store(state, std::memory_order_release);
    if (offOwner) {
        // Native loss may arrive on a transport thread. Keep all callbacks,
        // domain snapshots, and native ownership alive there; its only user
        // visible effect is a thread-neutral failure notification.
        relay->RetainForLostProof();
        try { relay->completion({}, error ? error : relay->lostProof); } catch (...) {}
        return;
    }
    auto completion = std::move(relay->completion);
    // Retain the exact request/domain/native graph through the owner's
    // completion callback.  This is the normal owner terminal boundary.
    try { completion(std::move(generation), error); } catch (...) {}
    if (!relay->operation.Retire()) {
        // Completion has already been delivered exactly once.  Preserve the
        // graph fail-closed if the owner proof boundary is unexpectedly gone.
        state_.store(State::LostProof, std::memory_order_release);
        relay->RetainForLostProof();
        return;
    }
    relay->job.reset();
}

} // namespace usdGen::vulkan
