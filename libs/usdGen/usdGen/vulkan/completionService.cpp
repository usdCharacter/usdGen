// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "completionService.h"

#include <limits>
#include <new>

namespace usdGen::vulkan {
namespace {
enum class SlotState : uint8_t { Free, Reserved, Marked, Arming, Armed, Proving, OwnerQueued, Delivered, Quarantined };
}

struct VulkanCompletionService::Slot {
    std::atomic<SlotState> state{SlotState::Free};
    std::atomic<uint64_t> generation{0};
    uint64_t jobEpoch = 0;
    uint64_t completionEpoch = 0;
    VkFence proof = VK_NULL_HANDLE;
    std::shared_ptr<const void> lifetime;
    Delivery delivery;
    Failure failure;
    UsdGenExecutionPipeline::CommandTicket ticket;
    std::unique_ptr<std::shared_ptr<VulkanCompletionService>> quarantine;
    std::atomic<bool> quarantined{false};  // elects the one quarantine writer
};

VulkanCompletionService::Watch::Watch(std::shared_ptr<VulkanCompletionService> service,
                                      uint32_t slot) noexcept
    : service_(std::move(service)), slot_(slot) {}
VulkanCompletionService::Watch::~Watch() { if (service_) service_->Abandon(*this); }
VulkanCompletionService::Watch::Watch(Watch&& other) noexcept = default;
VulkanCompletionService::Watch& VulkanCompletionService::Watch::operator=(Watch&& other) noexcept {
    if (this != &other) {
        if (service_) service_->Abandon(*this);
        service_ = std::move(other.service_); slot_ = other.slot_;
        marked_ = other.marked_; armed_ = other.armed_;
        other.slot_ = UINT32_MAX; other.marked_ = other.armed_ = false;
    }
    return *this;
}
bool VulkanCompletionService::Watch::MarkPhaseSubmitted() noexcept {
    return service_ && service_->Mark(*this);
}
bool VulkanCompletionService::Watch::Retire() noexcept {
    return service_ && service_->Retire(*this);
}

VulkanCompletionService::VulkanCompletionService(CreateInfo info)
    : context_(std::move(info.context)), queueOwner_(info.queueOwner), maxProofWaitNs_(info.maxProofWaitNs) {}

std::shared_ptr<VulkanCompletionService> VulkanCompletionService::Create(CreateInfo info, VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!info.context || !info.queueOwner || !info.maxWatches || !info.maxProofWaitNs ||
        info.maxProofWaitNs == UINT64_MAX || !info.context->timelineSemaphoreEnabled()) return {};
    try {
        auto service = std::shared_ptr<VulkanCompletionService>(new VulkanCompletionService(std::move(info)));
        service->quarantine_ = std::make_unique<std::shared_ptr<VulkanCompletionService>>();
        service->slots_.reserve(info.maxWatches);
        for (uint32_t i = 0; i != info.maxWatches; ++i) {
            auto slot = std::make_unique<Slot>();
            slot->quarantine = std::make_unique<std::shared_ptr<VulkanCompletionService>>();
            VkFenceCreateInfo fence{};
            fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            VkResult status = vkCreateFence(service->context_->device(), &fence, nullptr, &slot->proof);
            if (status != VK_SUCCESS) { if (result) *result = status; return {}; }
            service->slots_.push_back(std::move(slot));
        }
        VkSemaphoreTypeCreateInfo type{};
        type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        create.pNext = &type;
        VkResult status = vkCreateSemaphore(service->context_->device(), &create, nullptr, &service->completion_);
        if (status != VK_SUCCESS) { if (result) *result = status; return {}; }
        status = vkCreateSemaphore(service->context_->device(), &create, nullptr, &service->control_);
        if (status != VK_SUCCESS) { if (result) *result = status; return {}; }
        // The waiter owns a service reference until explicit external Close
        // wakes and joins it. Destruction is therefore never an implicit join.
        service->waiter_ = std::thread([service] { service->WaitMain(); });
        if (result) *result = VK_SUCCESS;
        return service;
    } catch (std::bad_alloc const&) { if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY; return {}; }
    catch (...) { if (result) *result = VK_ERROR_INITIALIZATION_FAILED; return {}; }
}

VulkanCompletionService::~VulkanCompletionService() {
    // Explicit CloseAndJoin releases the waiter's self-reference before this
    // point. A lost-proof service holds a permanent self-reference instead.
    if (!lost_.load(std::memory_order_acquire)) {
        if (completion_) vkDestroySemaphore(context_->device(), completion_, nullptr);
        if (control_) vkDestroySemaphore(context_->device(), control_, nullptr);
        for (auto const& slot : slots_)
            if (slot->proof) vkDestroyFence(context_->device(), slot->proof, nullptr);
    }
}

std::unique_ptr<VulkanCompletionService::Watch> VulkanCompletionService::Reserve(
    std::shared_ptr<const void> lifetime, Delivery delivery, Failure failure) noexcept {
    if (!lifetime || !delivery || closing_.load(std::memory_order_acquire) ||
        lost_.load(std::memory_order_acquire) || !queueOwner_->IsExecutingOwner()) return {};
    auto ticket = queueOwner_->ReserveCommandTicket();
    if (!ticket) return {};
    for (uint32_t i = 0; i != slots_.size(); ++i) {
        auto& slot = *slots_[i];
        auto expected = SlotState::Free;
        if (!slot.state.compare_exchange_strong(expected, SlotState::Reserved, std::memory_order_acq_rel)) continue;
        try {
            slot.lifetime = std::move(lifetime);
            slot.delivery = std::move(delivery);
            slot.failure = std::move(failure);
            slot.ticket = std::move(ticket);
            slot.generation.fetch_add(1, std::memory_order_relaxed);
            return std::unique_ptr<Watch>(new Watch(shared_from_this(), i));
        } catch (...) {
            slot.lifetime.reset(); slot.delivery = {}; slot.failure = {}; slot.ticket = {};
            slot.state.store(SlotState::Free, std::memory_order_release);
            return {};
        }
    }
    return {};
}

bool VulkanCompletionService::Mark(Watch& watch) noexcept {
    if (!queueOwner_->IsExecutingOwner() || watch.service_.get() != this || watch.slot_ >= slots_.size() ||
        watch.marked_ || watch.armed_ || closing_.load(std::memory_order_acquire) ||
        lost_.load(std::memory_order_acquire)) return false;
    auto& slot = *slots_[watch.slot_];
    auto expected = SlotState::Reserved;
    if (!slot.state.compare_exchange_strong(expected, SlotState::Marked, std::memory_order_acq_rel)) return false;
    watch.marked_ = true;
    return true;
}

bool VulkanCompletionService::Retire(Watch& watch) noexcept {
    if (!queueOwner_->IsExecutingOwner() || watch.service_.get() != this || watch.slot_ >= slots_.size() ||
        !watch.armed_) return false;
    auto& slot = *slots_[watch.slot_];
    auto expected = SlotState::Delivered;
    if (!slot.state.compare_exchange_strong(expected, SlotState::OwnerQueued, std::memory_order_acq_rel)) return false;
    // This owner-confined acknowledgement, never the waiter thread, is the
    // last-drop boundary for native work and callback captures.
    if (vkResetFences(context_->device(), 1, &slot.proof) != VK_SUCCESS) {
        Quarantine(watch.slot_); return false;
    }
    slot.lifetime.reset(); slot.delivery = {}; slot.failure = {}; slot.ticket = {};
    slot.state.store(SlotState::Free, std::memory_order_release);
    watch.service_.reset(); watch.slot_ = UINT32_MAX;
    watch.marked_ = watch.armed_ = false;
    return true;
}

bool VulkanCompletionService::Wake() noexcept {
    uint64_t const epoch = publishedControlEpoch_.fetch_add(1, std::memory_order_release) + 1;
    VkSemaphoreSignalInfo signal{};
    signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal.semaphore = control_;
    signal.value = epoch;
    return vkSignalSemaphore(context_->device(), &signal) == VK_SUCCESS;
}

bool VulkanCompletionService::Arm(Watch& watch, uint64_t jobEpoch, uint64_t* completionEpoch) noexcept {
    if (!queueOwner_->IsExecutingOwner() || watch.service_.get() != this || watch.slot_ >= slots_.size() ||
        !watch.marked_ || watch.armed_ || closing_.load(std::memory_order_acquire) ||
        lost_.load(std::memory_order_acquire)) return false;
    auto& slot = *slots_[watch.slot_];
    auto marked = SlotState::Marked;
    if (!slot.state.compare_exchange_strong(marked, SlotState::Arming, std::memory_order_acq_rel)) return false;
    uint64_t const epoch = ++nextGpuEpoch_;
    if (vkResetFences(context_->device(), 1, &slot.proof) != VK_SUCCESS) {
        slot.state.store(SlotState::Marked, std::memory_order_release);
        Quarantine(UINT32_MAX);
        for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
        (void)Wake(); return false;
    }
    VkTimelineSemaphoreSubmitInfo timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &epoch;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.pNext = &timeline;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &completion_;
    if (vkQueueSubmit(context_->computeQueue(), 1, &submit, slot.proof) != VK_SUCCESS) {
        slot.state.store(SlotState::Marked, std::memory_order_release);
        Quarantine(UINT32_MAX);
        for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
        (void)Wake(); return false;
    }
    // Publish the fixed immutable record before controlEpoch is released and
    // host-signaled.  The waiter loads the control epoch before scanning.
    slot.jobEpoch = jobEpoch;
    slot.completionEpoch = epoch;
    slot.state.store(SlotState::Armed, std::memory_order_release);
    watch.armed_ = true;
    if (closing_.load(std::memory_order_acquire) || lost_.load(std::memory_order_acquire)) {
        RouteFailure(watch.slot_, VK_ERROR_DEVICE_LOST); return false;
    }
    if (completionEpoch) *completionEpoch = epoch;
    if (!Wake()) {
        Quarantine(UINT32_MAX);
        for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
        return false;
    }
    return true;
}

bool VulkanCompletionService::CancelBeforeSubmit(Watch& watch) noexcept {
    if (!queueOwner_->IsExecutingOwner() || watch.service_.get() != this || watch.slot_ >= slots_.size() ||
        watch.marked_ || watch.armed_) return false;
    auto& slot = *slots_[watch.slot_];
    auto expected = SlotState::Reserved;
    if (!slot.state.compare_exchange_strong(expected, SlotState::Free, std::memory_order_acq_rel)) return false;
    slot.lifetime.reset(); slot.delivery = {}; slot.failure = {}; slot.ticket = {};
    watch.service_.reset(); watch.slot_ = UINT32_MAX;
    return true;
}

void VulkanCompletionService::Quarantine(uint32_t index) noexcept {
    lost_.store(true, std::memory_order_release);
    if (quarantine_ && !quarantined_.exchange(true, std::memory_order_acq_rel)) {
        try { *quarantine_ = shared_from_this(); } catch (...) {}
    }
    if (index < slots_.size()) {
        auto& slot = *slots_[index];
        auto prior = slot.state.exchange(SlotState::Quarantined, std::memory_order_acq_rel);
        if (prior != SlotState::Free && slot.quarantine &&
            !slot.quarantined.exchange(true, std::memory_order_acq_rel)) {
            try { *slot.quarantine = shared_from_this(); } catch (...) {}
        }
    }
    // Do not host-signal here: this path is also used by the waiter and by
    // arbitrary Watch destruction. Host timeline signals are serialized only
    // by the bound owner (Arm) or external Close after owner registration is
    // stopped, preserving strictly increasing call order.
}

void VulkanCompletionService::RouteFailure(uint32_t index, VkResult status) noexcept {
    if (index >= slots_.size()) return;
    auto& slot = *slots_[index];
    auto state = slot.state.load(std::memory_order_acquire);
    while (state == SlotState::Marked || state == SlotState::Armed || state == SlotState::Proving) {
        if (slot.state.compare_exchange_weak(state, SlotState::OwnerQueued, std::memory_order_acq_rel)) {
            uint64_t const generation = slot.generation.load(std::memory_order_acquire);
            auto self = shared_from_this();
            bool posted = false;
            try {
                auto ticket = std::move(slot.ticket);
                posted = queueOwner_->PostCommand(std::move(ticket),
                    [self, index, generation, status] { self->FailOnOwner(index, generation, status); },
                    [self, index, generation, status] { self->FailOnOwner(index, generation, status); });
            } catch (...) {}
            // The transport claim is OwnerQueued. A false post and a later
            // onCancel both converge through FailOnOwner's single CAS, so the
            // retained thread-neutral failure route runs exactly once.
            if (!posted) FailOnOwner(index, generation, status);
            return;
        }
    }
}

void VulkanCompletionService::Abandon(Watch& watch) noexcept {
    if (watch.slot_ >= slots_.size()) return;
    auto& slot = *slots_[watch.slot_];
    if (!watch.marked_) {
        auto expected = SlotState::Reserved;
        if (slot.state.compare_exchange_strong(expected, SlotState::Free, std::memory_order_acq_rel)) {
            slot.lifetime.reset(); slot.delivery = {}; slot.failure = {}; slot.ticket = {};
        }
    } else if (queueOwner_->IsExecutingOwner()) {
        // Claim only states before the waiter transfers the reserved ticket.
        // Once OwnerQueued wins, its immutable owner transport exclusively
        // owns the callback path and this destructor must not touch it.
        SlotState prior = slot.state.load(std::memory_order_acquire);
        bool claimed = false;
        while (prior == SlotState::Marked || prior == SlotState::Armed || prior == SlotState::Proving) {
            if (slot.state.compare_exchange_weak(prior, SlotState::Quarantined, std::memory_order_acq_rel)) {
                claimed = true;
                break;
            }
        }
        if (claimed) {
            // No transport can now move this ticket, so return this abandoned
            // slot's bounded command admission while retaining every native
            // and callback lifetime in the quarantined service graph.
            slot.ticket = {};
            Quarantine(UINT32_MAX);
            for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
            if (!closing_.load(std::memory_order_acquire)) (void)Wake();
        }
    }
    watch.slot_ = UINT32_MAX; watch.marked_ = watch.armed_ = false;
}

void VulkanCompletionService::DeliverOnOwner(uint32_t index, uint64_t generation, VkResult proof) noexcept {
    if (!queueOwner_->IsExecutingOwner() || index >= slots_.size()) return;
    auto& slot = *slots_[index];
    if (slot.generation.load(std::memory_order_acquire) != generation) return;
    auto expected = SlotState::OwnerQueued;
    if (!slot.state.compare_exchange_strong(expected, SlotState::Delivered, std::memory_order_acq_rel)) return;
    // Move every potentially-last native/job capture to this owner frame
    // before invoking user delivery. Watch::Retire may then safely release its
    // slot without destroying the function currently executing.
    auto delivery = std::move(slot.delivery);
    auto lifetime = std::move(slot.lifetime);
    auto failure = std::move(slot.failure);
    DeliveryResult result = DeliveryResult::LostProof;
    try { result = delivery(proof); } catch (...) {}
    // Retire can free and reuse this slot while `delivery` executes. Never
    // restore into a different generation. A normal Posted callback commonly
    // only enqueues a later job notification, so retain its captures in the
    // same Delivered slot until that later owner frame calls Retire.
    bool const unchanged = slot.generation.load(std::memory_order_acquire) == generation;
    if (result == DeliveryResult::LostProof && unchanged) {
        slot.delivery = std::move(delivery);
        slot.lifetime = std::move(lifetime);
        slot.failure = std::move(failure);
        Quarantine(index);
        // Losing the delivery route stops the service, just as losing a
        // native proof does. Settle every sibling still awaiting transport
        // before waking the waiter so accepted work cannot be stranded.
        // OwnerQueued siblings already own their exactly-once delivery.
        for (uint32_t i = 0; i != slots_.size(); ++i)
            RouteFailure(i, VK_ERROR_DEVICE_LOST);
        if (!closing_.load(std::memory_order_acquire)) (void)Wake();
    } else if (unchanged && slot.state.load(std::memory_order_acquire) == SlotState::Delivered) {
        slot.delivery = std::move(delivery);
        slot.lifetime = std::move(lifetime);
        slot.failure = std::move(failure);
    }
}

void VulkanCompletionService::FailOnOwner(uint32_t index, uint64_t generation, VkResult status) noexcept {
    if (index >= slots_.size()) return;
    auto& slot = *slots_[index];
    if (slot.generation.load(std::memory_order_acquire) != generation) return;
    if (!queueOwner_->IsExecutingOwner()) {
        auto expected = SlotState::OwnerQueued;
        if (!slot.state.compare_exchange_strong(expected, SlotState::Quarantined, std::memory_order_acq_rel)) return;
        // Shutdown may invoke onCancel outside the owner. This optional
        // closure is explicitly thread-neutral and never moves native state.
        try { if (slot.failure) slot.failure(status); } catch (...) {}
        Quarantine(index);
        return;
    }
    auto expected = SlotState::OwnerQueued;
    if (!slot.state.compare_exchange_strong(expected, SlotState::Quarantined, std::memory_order_acq_rel)) return;
    // Required delivery runs on the owner for every accepted terminal route.
    // State is already Quarantined, so Retire cannot free unproved ownership.
    try { if (slot.delivery) (void)slot.delivery(status); } catch (...) {}
    // Keep delivery/lifetime in the quarantined slot. They must never become
    // last-owned by this failure owner command.
    Quarantine(index);
}

void VulkanCompletionService::WaitMain() noexcept {
    while (!closing_.load(std::memory_order_acquire) && !lost_.load(std::memory_order_acquire)) {
        uint64_t gpu = 0;
        if (vkGetSemaphoreCounterValue(context_->device(), completion_, &gpu) != VK_SUCCESS) {
            Quarantine(UINT32_MAX);
            for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
            return;
        }
        // Acquire control publication before looking at armed records; Arm
        // publishes record -> control epoch -> host signal in that order.
        uint64_t const controlTarget = publishedControlEpoch_.load(std::memory_order_acquire) + 1;
        bool delivered = false;
        for (uint32_t i = 0; i != slots_.size(); ++i) {
            auto& slot = *slots_[i];
            // Completion fields are immutable only after this acquire load.
            if (slot.state.load(std::memory_order_acquire) != SlotState::Armed || slot.completionEpoch > gpu)
                continue;
            auto expected = SlotState::Armed;
            if (!slot.state.compare_exchange_strong(expected, SlotState::Proving, std::memory_order_acq_rel)) continue;
            // The timeline says the follow-up signal became visible; the
            // per-slot fence is the bounded proof that the ordered queue work
            // (including every prior phase submission) actually completed.
            VkResult proof = vkWaitForFences(context_->device(), 1, &slot.proof, VK_TRUE, maxProofWaitNs_);
            if (proof != VK_SUCCESS) {
                Quarantine(UINT32_MAX);
                for (uint32_t j = 0; j != slots_.size(); ++j) RouteFailure(j, proof);
                delivered = true; continue;
            }
            uint64_t const generation = slot.generation.load(std::memory_order_acquire);
            expected = SlotState::Proving;
            if (!slot.state.compare_exchange_strong(expected, SlotState::OwnerQueued, std::memory_order_acq_rel)) {
                delivered = true; continue;
            }
            auto self = shared_from_this();
            bool posted = false;
            try {
                auto ticket = std::move(slot.ticket);
                posted = queueOwner_->PostCommand(std::move(ticket),
                    [self, i, generation] { self->DeliverOnOwner(i, generation, VK_SUCCESS); },
                    [self, i, generation] { self->FailOnOwner(i, generation, VK_ERROR_DEVICE_LOST); });
            } catch (...) {}
            // Do not inspect this slot after a successful handoff: its owner
            // command can immediately Retire and reuse it.
            if (!posted) FailOnOwner(i, generation, VK_ERROR_DEVICE_LOST);
            delivered = true;
        }
        // An Arm signal can complete before this waiter reaches its first
        // blocking wait. Consume it now rather than waiting for another event.
        if (delivered || lost_.load(std::memory_order_acquire)) continue;
        uint64_t gpuTarget = gpu + 1;
        for (auto const& entry : slots_) {
            if (entry->state.load(std::memory_order_acquire) == SlotState::Armed &&
                entry->completionEpoch > gpu && entry->completionEpoch < gpuTarget)
                gpuTarget = entry->completionEpoch;
        }
        VkSemaphore semaphores[] = {completion_, control_};
        uint64_t values[] = {gpuTarget, controlTarget};
        VkSemaphoreWaitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
        wait.semaphoreCount = 2; wait.pSemaphores = semaphores; wait.pValues = values;
        // Close may have signaled control before this iteration captured its
        // next target. Recheck after that capture: otherwise we can wait for
        // a second shutdown signal that will never come, deadlocking Join.
        if (closing_.load(std::memory_order_acquire) || lost_.load(std::memory_order_acquire)) break;
        VkResult status = vkWaitSemaphores(context_->device(), &wait, UINT64_MAX);
        if (status != VK_SUCCESS) {
            Quarantine(UINT32_MAX);
            for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, status);
            return;
        }
        if (vkGetSemaphoreCounterValue(context_->device(), completion_, &gpu) != VK_SUCCESS) {
            Quarantine(UINT32_MAX);
            for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
            return;
        }
        // Re-enter the top of the loop, which claims only Armed records after
        // acquiring their immutable completion fields.
    }
}

bool VulkanCompletionService::CloseAndJoin() noexcept {
    if (UsdGenExecutionPipeline::IsExecuting()) return false;
    if (!closing_.exchange(true, std::memory_order_acq_rel)) {
        // Callers stop/join owner registration before this external boundary.
        // Do not touch OwnerQueued/Delivered records: their reserved owner
        // delivery has already claimed them and may still execute.
        for (uint32_t i = 0; i != slots_.size(); ++i) {
            auto state = slots_[i]->state.load(std::memory_order_acquire);
            if (state == SlotState::Reserved) {
                slots_[i]->lifetime.reset(); slots_[i]->delivery = {};
                slots_[i]->failure = {}; slots_[i]->ticket = {};
                slots_[i]->state.store(SlotState::Free, std::memory_order_release);
            }
        }
        // Wake first, then join. The post-join sweep below owns terminal
        // transport for only still-eligible Marked/Armed records.
        if (!Wake()) {
            Quarantine(UINT32_MAX);
            for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
            // A failed host wake cannot be joined safely: preserve the waiter
            // and every native handle through its preallocated self holder.
            if (waiter_.joinable()) waiter_.detach();
            return false;
        }
    }
    if (waiter_.joinable()) {
        waiter_.join();
        waiterJoined_ = true;
    }
    for (uint32_t i = 0; i != slots_.size(); ++i) RouteFailure(i, VK_ERROR_DEVICE_LOST);
    return waiterJoined_;
}
} // namespace usdGen::vulkan
