// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Dispatch proof for the host completion service without any operator lane.
// An empty same-queue submission stands in for phase work: the test proves
// Reserve/Mark/Arm/delivery/Retire ordering, close-pending failure routing,
// and idle-close shutdown against a real timeline device.
#include "hosts/usdGenUsd/vulkanNativeFixture.h"
#include "usdGenVulkan/vulkan/completionService.h"

#include <atomic>
#include <stdexcept>
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan completion service check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static DeviceContext::CreateInfo ContextInfo(std::shared_ptr<NativeOwner> const& native) {
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7016; info.nativeLifetime = native;
    info.resources = {size_t{8} << 20, 0};
    return info;
}

static int CheckDeliveryFailure(std::shared_ptr<DeviceContext> const& context,
                                UsdGenExecutionPipeline& owner, bool throwing) {
    using Result = VulkanCompletionService::DeliveryResult;
    struct State {
        std::unique_ptr<VulkanCompletionService::Watch> first, sibling;
        unsigned firstCalls = 0, siblingCalls = 0;
        bool admitted = false, firstOnOwner = false, siblingOnOwner = false;
    };
    // Quarantined delivery captures must own their test state too: no retained
    // callback may refer to a local whose lifetime ended after this check.
    auto state = std::make_shared<State>();
    auto service = VulkanCompletionService::Create({context, &owner, 2});
    CHECK(service);
    owner.Await([&](auto finish) {
        if (!owner.PostCommand([state, service, context, &owner, throwing, finish] {
            state->first = service->Reserve(state,
                [state, &owner, throwing](VkResult proof) -> Result {
                    ++state->firstCalls;
                    state->firstOnOwner = owner.IsExecutingOwner() && proof == VK_SUCCESS;
                    if (throwing) throw std::runtime_error("delivery failed");
                    return Result::LostProof;
                });
            state->sibling = service->Reserve(state,
                [state, &owner, finish](VkResult proof) {
                    ++state->siblingCalls;
                    state->siblingOnOwner = owner.IsExecutingOwner() && proof == VK_ERROR_DEVICE_LOST;
                    finish();
                    return Result::Posted;
                });
            VkSubmitInfo phases[2]{};
            for (auto& phase : phases) phase.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            state->admitted = state->first && state->sibling &&
                state->first->MarkPhaseSubmitted() && state->sibling->MarkPhaseSubmitted() &&
                vkQueueSubmit(context->computeQueue(), 2, phases, VK_NULL_HANDLE) == VK_SUCCESS &&
                service->Arm(*state->first, 1);
            // The sibling crossed native admission but has no completion
            // marker yet. Failure of the first delivery must settle it without
            // requiring a later Arm, explicit Close, or manual notification.
            if (!state->admitted) finish();
        })) finish();
    });
    CHECK(state->admitted && state->firstOnOwner && state->siblingOnOwner);
    CHECK(state->firstCalls == 1 && state->siblingCalls == 1);
    bool rejected = false;
    owner.InvokeOwner([&] {
        rejected = !service->Reserve(std::make_shared<int>(1), [](VkResult) { return Result::Stale; });
        state->first.reset(); state->sibling.reset();
    });
    CHECK(rejected);
    service->CloseAndJoin();
    owner.Drain();
    // Neither Close nor watch destruction may replay terminal callbacks.
    CHECK(state->firstCalls == 1 && state->siblingCalls == 1);
    CHECK(owner.OutstandingCommands() == 0);
    return 0;
}

// SubmitOnTimeline: one call submits through the queue timeline and runs
// the delivery on the owner once the value passed; the slot frees itself and
// the lifetime drops after delivery. Two rounds through a one-slot service.
static int CheckSubmitOnTimeline(std::shared_ptr<DeviceContext> const& context,
                                 UsdGenExecutionPipeline& owner) {
    using Result = VulkanCompletionService::DeliveryResult;
    VkDevice const device = context->device();
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = context->computeQueueFamily();
    VkCommandPool pool = VK_NULL_HANDLE;
    CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &pool) == VK_SUCCESS);
    VkCommandBufferAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc.commandPool = pool; alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    CHECK(vkAllocateCommandBuffers(device, &alloc, &cmd) == VK_SUCCESS);
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    CHECK(vkBeginCommandBuffer(cmd, &begin) == VK_SUCCESS && vkEndCommandBuffer(cmd) == VK_SUCCESS);
    auto service = VulkanCompletionService::Create({context, &owner, 1});
    CHECK(service);
    unsigned delivered = 0;
    bool ok = true;
    uint64_t lastValue = 0;
    for (unsigned round = 0; round != 2; ++round) {
        std::weak_ptr<int> life;
        uint64_t value = 0;
        owner.Await([&](auto finish) {
            if (!owner.PostCommand([&, finish] {
                auto lifetime = std::make_shared<int>(1); life = lifetime;
                VkResult r = VK_ERROR_UNKNOWN;
                value = SubmitOnTimeline(*service, &cmd, 1, std::move(lifetime),
                    [&, finish](VkResult proof) {
                        ++delivered;
                        ok = ok && owner.IsExecutingOwner() && proof == VK_SUCCESS &&
                             context->computeTimeline()->Check(value) == VK_SUCCESS;
                        finish();
                        return Result::Posted;
                    }, &r);
                ok = ok && r == VK_SUCCESS && value > lastValue;
                if (!value) finish();
            })) finish();
        });
        owner.Drain();
        CHECK(ok && delivered == round + 1 && life.expired());
        lastValue = value;
    }
    // Rejected before submission: nothing queued, delivery dropped unrun.
    bool rejected = false;
    owner.InvokeOwner([&] {
        VkResult r = VK_SUCCESS;
        rejected = SubmitOnTimeline(*service, &cmd, 0, std::make_shared<int>(1),
                                    [&](VkResult) { ++delivered; return Result::Posted; }, &r) == 0 &&
                   r != VK_SUCCESS;
    });
    CHECK(rejected && delivered == 2);
    CHECK(service->CloseAndJoin());
    owner.Drain();
    CHECK(delivered == 2 && owner.OutstandingCommands() == 0);
    vkDestroyCommandPool(device, pool, nullptr);
    return 0;
}

int main(int argc, char** argv) {
    bool const idleClose = argc == 2 && std::string(argv[1]) == "--idle-close";
    bool const unavailableOwner = argc == 2 && std::string(argv[1]) == "--owner-unavailable";
    bool const closePending = unavailableOwner || (argc == 2 && std::string(argv[1]) == "--close-pending");
    CHECK(argc == 1 || closePending || idleClose);
    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures supported{};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{}; features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &supported;
    vkGetPhysicalDeviceFeatures2(probe->physical, &features);
    if (!supported.timelineSemaphore) return 77;
    probe.reset();
    VkPhysicalDeviceTimelineSemaphoreFeatures enabled{};
    enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enabled.timelineSemaphore = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, &enabled); CHECK(native);
    UsdGenExecutionRuntime runtime{2};
    UsdGenExecutionPipeline owner(runtime), foreign(runtime);
    auto ci = ContextInfo(native);
    auto unclaimed = DeviceContext::Create(ci); CHECK(unclaimed);
    VkResult status = VK_SUCCESS;
    CHECK(!VulkanCompletionService::Create({unclaimed, &owner, 2}, &status));
    ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci); CHECK(context);
    CHECK(!VulkanCompletionService::Create({context, &owner, 0}, &status));
    CHECK(!VulkanCompletionService::Create({context, nullptr, 2}, &status));
    if (idleClose) {
        // Exercise Close racing the waiter's first target capture. No work
        // can supply a second wake if shutdown's first wake is missed.
        for (unsigned i = 0; i != 256; ++i) {
            auto idle = VulkanCompletionService::Create({context, &owner, 1}); CHECK(idle);
            idle->CloseAndJoin();
        }
        CHECK(owner.OutstandingCommands() == 0);
        return 0;
    }
    auto service = VulkanCompletionService::Create({context, &owner, 1}, &status);
    CHECK(service && status == VK_SUCCESS);
    using Result = VulkanCompletionService::DeliveryResult;
    auto noDelivery = [](VkResult) { return Result::Stale; };
    CHECK(!service->Reserve(std::make_shared<int>(1), noDelivery));

    bool admissionOk = false;
    foreign.InvokeOwner([&] { admissionOk = !service->Reserve(std::make_shared<int>(1), noDelivery); });
    CHECK(admissionOk);
    std::weak_ptr<int> cancelledLifetime;
    owner.InvokeOwner([&] {
        auto life = std::make_shared<int>(1); cancelledLifetime = life;
        auto watch = service->Reserve(std::move(life), noDelivery);
        admissionOk = watch && !cancelledLifetime.expired();
        admissionOk = admissionOk && !service->Reserve(std::make_shared<int>(2), noDelivery);
        admissionOk = admissionOk && service->CancelBeforeSubmit(*watch);
    });
    CHECK(admissionOk && cancelledLifetime.expired());

    std::unique_ptr<VulkanCompletionService::Watch> watch;
    if (closePending) {
        unsigned failures = 0;
        unsigned fallbackFailures = 0;
        bool ownerFailure = false;
        bool fallbackFailure = false;
        bool submitted = false;
        owner.InvokeOwner([&] {
            watch = service->Reserve(std::make_shared<int>(1), [&](VkResult proof) {
                ++failures; ownerFailure = owner.IsExecutingOwner() && proof != VK_SUCCESS;
                return Result::Posted;
            }, [&](VkResult proof) {
                ++fallbackFailures;
                fallbackFailure = !owner.IsExecutingOwner() && proof != VK_SUCCESS;
            });
            admissionOk = watch && watch->MarkPhaseSubmitted();
            VkSubmitInfo phase{}; phase.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitted = vkQueueSubmit(native->queue, 1, &phase, VK_NULL_HANDLE) == VK_SUCCESS;
            // Intentionally omit Arm: external close must settle this accepted
            // uncertain phase without waiting for a notification never issued.
        });
        CHECK(admissionOk && submitted);
        if (unavailableOwner) owner.Shutdown();
        service->CloseAndJoin(); owner.Drain();
        if (unavailableOwner) {
            CHECK(fallbackFailure && fallbackFailures == 1 && failures == 0);
            watch.reset();
        } else {
            CHECK(ownerFailure && failures == 1 && fallbackFailures == 0);
            owner.InvokeOwner([&] { watch.reset(); });
        }
        service.reset();
        CHECK(owner.OutstandingCommands() == 0);
        return 0;
    }

    // Real queue work advances only through the native timeline waiter. No
    // manual notification, fence wait, polling loop, or queue idle here.
    auto baseline = context->resources()->Snapshot();
    std::atomic<unsigned> notifications{0};
    bool completionOk = false;
    uint64_t completionEpoch = 0;
    owner.Await([&](auto finish) {
        if (!owner.PostCommand([&, finish] {
            auto ticket = std::make_shared<UsdGenExecutionPipeline::CommandTicket>(owner.ReserveCommandTicket());
            if (!*ticket) { finish(); return; }
            auto lifetime = std::make_shared<int>(1);
            watch = service->Reserve(lifetime, [&, ticket, finish](VkResult proof) {
                ++notifications;
                bool const posted = owner.PostCommand(std::move(*ticket), [&, proof, finish] {
                    completionOk = owner.IsExecutingOwner() && proof == VK_SUCCESS;
                    completionOk = watch->Retire() && completionOk;
                    finish();
                }, finish);
                return posted ? Result::Posted : Result::LostProof;
            });
            VkSubmitInfo phase{}; phase.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            if (!watch || !watch->MarkPhaseSubmitted() ||
                vkQueueSubmit(native->queue, 1, &phase, VK_NULL_HANDLE) != VK_SUCCESS ||
                !service->Arm(*watch, 41, &completionEpoch)) {
                // A failure remains visible; uncertain native work stays
                // retained by the watch/service quarantine paths.
                finish();
            }
        })) finish();
    });
    service->CloseAndJoin();
    owner.Drain();
    CHECK(completionOk && notifications == 1 && completionEpoch > 0);
    auto settled = context->resources()->Snapshot();
    CHECK(settled.usedBytes == baseline.usedBytes && settled.byKind == baseline.byKind);
    owner.InvokeOwner([&] { admissionOk = !service->Reserve(std::make_shared<int>(1), noDelivery); });
    CHECK(admissionOk);
    owner.InvokeOwner([&] { watch.reset(); });
    service.reset();
    CHECK(owner.OutstandingCommands() == 0);
    CHECK(CheckDeliveryFailure(context, owner, false) == 0);
    CHECK(CheckDeliveryFailure(context, owner, true) == 0);
    CHECK(CheckSubmitOnTimeline(context, owner) == 0);
    return 0;
}
