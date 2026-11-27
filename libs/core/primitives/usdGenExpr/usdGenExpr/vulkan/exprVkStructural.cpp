// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "exprVkStructural.h"
#include "usdGen/api/token.h"
#include <cstring>
namespace usdGen::vulkan {
namespace {
struct Packet : std::enable_shared_from_this<Packet> {
    std::shared_ptr<ExprVkContextPipeline> context;
    std::shared_ptr<VulkanCompletionService> service;
    std::unique_ptr<ExprVkRun> run;
    std::vector<UsdGenToken> names;
    std::unique_ptr<VulkanCompletionService::Watch> watch;
    std::function<void(ExprVkStructuralValues, VkResult)> completion;
    std::shared_ptr<Packet> quarantine;
    std::atomic<bool> quarantined{false};
    std::atomic<bool> settled{false};
    bool phaseMarked = false;
    void Finish(VkResult status) {
        if (settled.exchange(true, std::memory_order_acq_rel))
            return;
        ExprVkStructuralValues values;
        if (status == VK_SUCCESS) {
            for (auto const &name : names) {
                auto f = run->Find(name);
                void *p = nullptr;
                if (!f || f->domain != expr::Domain::Groom || f->count != 1 || f->components != 1 ||
                    !f->data ||
                    vkMapMemory(context->context()->device(), f->data->memory(), 0,
                                f->data->sizeBytes(), 0, &p) != VK_SUCCESS) {
                    status = VK_ERROR_VALIDATION_FAILED_EXT;
                    break;
                }
                uint32_t value;
                std::memcpy(&value, p, 4);
                vkUnmapMemory(context->context()->device(), f->data->memory());
                values.emplace(name, value);
            }
        }
        auto callback = std::move(completion);
        try {
            callback(std::move(values), status);
        } catch (...) {
        }
    }
    void Lost(VkResult status) {
        if (!quarantined.exchange(true, std::memory_order_acq_rel)) quarantine = shared_from_this();
        if (service->queueOwner()->IsExecutingOwner())
            run->Quarantine();
        Finish(status);
    }
    void Advance() noexcept {
        try {
            AdvanceImpl();
        } catch (...) {
            if (phaseMarked) {
                Lost(VK_ERROR_OUT_OF_HOST_MEMORY);
                return;
            }
            if (watch) {
                service->CancelBeforeSubmit(*watch);
                watch.reset();
            }
            Finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        }
    }
    void AdvanceImpl() {
        if (run->done()) {
            Finish(run->succeeded() ? VK_SUCCESS : VK_ERROR_VALIDATION_FAILED_EXT);
            return;
        }
        auto self = shared_from_this();
        watch = service->Reserve(
            self,
            [self](VkResult proof) {
                if (proof != VK_SUCCESS) {
                    self->Lost(proof);
                    return VulkanCompletionService::DeliveryResult::LostProof;
                }
                if (!self->watch->Retire()) {
                    self->watch.reset();
                    self->Lost(VK_ERROR_DEVICE_LOST);
                    return VulkanCompletionService::DeliveryResult::LostProof;
                }
                self->watch.reset();
                self->phaseMarked = false;
                auto poll = self->run->Poll();
                if (poll == VK_SUCCESS && self->run->NeedsHostStep()) {
                    poll = self->run->ComputeHostStep();
                    if (poll == VK_SUCCESS)
                        poll = self->run->CommitHostStep();
                }
                if (poll != VK_SUCCESS) {
                    self->Finish(poll);
                    return VulkanCompletionService::DeliveryResult::Posted;
                }
                self->Advance();
                return VulkanCompletionService::DeliveryResult::Posted;
            },
            [self](VkResult proof) { self->Lost(proof); });
        if (!watch) {
            Finish(VK_ERROR_OUT_OF_HOST_MEMORY);
            return;
        }
        bool marked = false;
        phaseMarked = false;
        VkResult status;
        if (!run->Advance(
                [&] {
                    marked = watch->MarkPhaseSubmitted();
                    phaseMarked = marked;
                    return marked;
                },
                &status)) {
            if (!marked)
                service->CancelBeforeSubmit(*watch);
            watch.reset();
            if (marked)
                Lost(status);
            else
                Finish(status);
            return;
        }
        if (!marked) {
            service->CancelBeforeSubmit(*watch);
            watch.reset();
            auto poll = run->Poll();
            if (poll == VK_SUCCESS && run->NeedsHostStep()) {
                poll = run->ComputeHostStep();
                if (poll == VK_SUCCESS)
                    poll = run->CommitHostStep();
            }
            if (poll != VK_SUCCESS) {
                Finish(poll);
                return;
            }
            Advance();
            return;
        }
        if (!service->Arm(*watch, 1)) {
            watch.reset();
            Lost(VK_ERROR_DEVICE_LOST);
        }
    }
};
} // namespace
void EvaluateExprVkStructural(std::shared_ptr<ExprVkContextPipeline> c,
                              std::shared_ptr<ExprVkEvaluatePipeline> e,
                              std::shared_ptr<VulkanCompletionService> service,
                              std::vector<ExprVkCompiledBinding> bindings, expr::Context controls,
                              std::function<void(ExprVkStructuralValues, VkResult)> callback) {
    if (!c || !e || !service || !service->queueOwner()->IsExecutingOwner()) {
        callback({}, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    auto packet = std::make_shared<Packet>();
    packet->context = c;
    packet->service = service;
    packet->completion = std::move(callback);
    for (auto const &b : bindings) {
        if (b.binding.domain != expr::Domain::Groom) {
            packet->Finish(VK_ERROR_VALIDATION_FAILED_EXT);
            return;
        }
        packet->names.push_back(b.binding.destination);
    }
    packet->run = std::make_unique<ExprVkRun>(c, e, std::move(bindings), ExprVkCurveGeometry{},
                                              ExprVkGeometryChannels{}, controls, false);
    packet->Advance();
}
} // namespace usdGen::vulkan
