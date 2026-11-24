// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkApply.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace usdGen::vulkan {
namespace {
bool ValidDom(uint32_t dom) { return dom == 0 || dom == 1 || dom == 2 || dom == 4; }
uint64_t DomSlots(uint32_t dom, uint32_t curves, uint32_t points) {
    if (dom == 1) return 1;
    if (dom == 2) return curves;
    return points;
}
bool ValidInput(std::shared_ptr<DeviceContext> const& context,
                std::shared_ptr<const ChargedBuffer> const& buffer,
                VkDeviceSize minimum) {
    return buffer && buffer->context() == context && !buffer->unproven() &&
        buffer->buffer() && (buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
        buffer->sizeBytes() >= minimum;
}
void BarrierBefore(VkCommandBuffer command) {
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
}
void BarrierAfter(VkCommandBuffer command) {
    VkMemoryBarrier after{};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask =
        VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0,
        nullptr, 0, nullptr);
}
} // namespace

struct ExprVkWidthApplyPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (pipeline) vkDestroyPipeline(d, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(d, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(d, shader, nullptr);
    }
};
struct ExprVkWidthApplyPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> input, owners, profile, hairT, curveOffsets;
    uint32_t points = 0, curves = 0;
    std::shared_ptr<const ChargedBuffer> fields[4];
    std::shared_ptr<ChargedBuffer> output, status, dummy;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

ExprVkWidthApplyPipeline::ExprVkWidthApplyPipeline(std::shared_ptr<Native> n)
    : native_(std::move(n)) {}
ExprVkWidthApplyPipeline::~ExprVkWidthApplyPipeline() = default;
std::shared_ptr<DeviceContext> const& ExprVkWidthApplyPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<ExprVkWidthApplyPipeline> ExprVkWidthApplyPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256 ||
        physical.limits.maxPushConstantsSize < 56 ||
        physical.limits.maxPerStageDescriptorStorageBuffers < 11 ||
        physical.limits.maxDescriptorSetStorageBuffers < 11)
        return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        VkShaderModuleCreateInfo sm{};
        sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t);
        sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetLayoutBinding bindings[11]{};
        for (uint32_t i = 0; i != 11; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 11;
        ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 56};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &n->descriptors;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(d, &pl, nullptr, &n->layout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkComputePipelineCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = n->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = n->shader;
        cp.stage.pName = "main";
        r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->pipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        finish(VK_SUCCESS);
        return std::shared_ptr<ExprVkWidthApplyPipeline>(
            new ExprVkWidthApplyPipeline(std::move(n)));
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

ExprVkWidthApplyPipeline::Candidate::Candidate(std::shared_ptr<State> s)
    : state_(std::move(s)) {}
ExprVkWidthApplyPipeline::Candidate::~Candidate() {
    if (state_ && state_->pending) Quarantine();
}
void ExprVkWidthApplyPipeline::Candidate::Quarantine() noexcept {
    if (!state_ || !state_->pending || state_->lost) return;
    state_->lost = true;
    state_->output->Quarantine();
    state_->status->Quarantine();
    state_->dummy->Quarantine();
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ExprVkWidthApplyPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> ExprVkWidthApplyPipeline::Candidate::output() const noexcept {
    return state_ && state_->points && state_->proved && state_->semantic == 0 ? state_->output : nullptr;
}
std::shared_ptr<const ChargedBuffer> ExprVkWidthApplyPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->input : nullptr;
}
uint32_t ExprVkWidthApplyPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->points : 0;
}
uint32_t ExprVkWidthApplyPipeline::Candidate::curveCount() const noexcept {
    return state_ ? state_->curves : 0;
}
std::shared_ptr<DeviceContext> ExprVkWidthApplyPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}
VkResult ExprVkWidthApplyPipeline::Candidate::Wait(uint64_t timeoutNs,
                                                        uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkWaitForFences(
            s.native->context->device(), 1, &s.fence, VK_TRUE, timeoutNs);
        if (r != VK_SUCCESS) {
            if (r != VK_TIMEOUT) Quarantine();
            return r;
        }
    }
    return Poll(semanticStatus);
}
VkResult ExprVkWidthApplyPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        r = s.output->PollComplete();
        if (r == VK_SUCCESS) r = s.status->PollComplete();
        if (r == VK_SUCCESS) r = s.dummy->PollComplete();
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        s.pending = false;
        void* data = nullptr;
        r = vkMapMemory(s.native->context->device(), s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.semantic, data, 4);
        vkUnmapMemory(s.native->context->device(), s.status->memory());
        s.proved = true;
    }
    if (semanticStatus) *semanticStatus = s.semantic;
    return VK_SUCCESS;
}

std::unique_ptr<ExprVkWidthApplyPipeline::Candidate> ExprVkWidthApplyPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> input, std::shared_ptr<const ChargedBuffer> owners,
    std::shared_ptr<const ChargedBuffer> widthField, std::shared_ptr<const ChargedBuffer> maskField,
    std::shared_ptr<const ChargedBuffer> replaceField,
    std::shared_ptr<const ChargedBuffer> enabledField, uint32_t pointCount,
    uint32_t curveCount, Controls controls, VkResult* result, BeforeSubmit beforeSubmit,
    std::shared_ptr<const ChargedBuffer> profile,
    std::shared_ptr<const ChargedBuffer> hairT,
    std::shared_ptr<const ChargedBuffer> curveOffsets) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    if (!ValidDom(controls.widthDom) || !ValidDom(controls.maskDom) ||
        !ValidDom(controls.replaceDom) || !ValidDom(controls.enabledDom) ||
        controls.replaceLiteral > 1 || controls.enabledLiteral > 1 ||
        !std::isfinite(controls.widthLiteral) || !std::isfinite(controls.maskLiteral) ||
        !std::isfinite(controls.baseLiteral) || controls.baseLiteral < 0.0f ||
        (pointCount == 0) != (curveCount == 0))
        return {};
    if ((pointCount && !ValidInput(context, input, VkDeviceSize(pointCount) * 4)) ||
        (!pointCount && input) ||
        (pointCount && !ValidInput(context, owners, VkDeviceSize(pointCount) * 4)) ||
        (profile && !ValidInput(context, profile, 257u * 4u)) ||
        (hairT && !ValidInput(context, hairT, VkDeviceSize(pointCount) * 4)) ||
        (curveOffsets && !ValidInput(context, curveOffsets, (VkDeviceSize(curveCount) + 1u) * 4u)) ||
        (pointCount && profile && !hairT && !curveOffsets))
        return {};
    std::shared_ptr<const ChargedBuffer> const fields[4] = {
        widthField, maskField, replaceField, enabledField};
    uint32_t const doms[4] = {
        controls.widthDom, controls.maskDom, controls.replaceDom, controls.enabledDom};
    for (int k = 0; k < 4; ++k) {
        if (doms[k] == 0) {
            if (fields[k]) return {};
        } else if (DomSlots(doms[k],curveCount,pointCount) && !ValidInput(context, fields[k],
                               DomSlots(doms[k], curveCount, pointCount) * 8))
            return {};
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t const work = std::max<uint64_t>(pointCount,
        std::max<uint64_t>(curveCount, profile ? 257u : 0u));
    uint64_t groups = (work + 255) / 256;
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};
    VkDeviceSize const outBytes = std::max<VkDeviceSize>(4u, VkDeviceSize(pointCount) * 4);
    if (outBytes > properties.limits.maxStorageBufferRange) return {};
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->input = std::move(input);
        s->owners = std::move(owners);
        s->profile = std::move(profile);
        s->hairT = std::move(hairT);
        s->curveOffsets = std::move(curveOffsets);
        s->points = pointCount; s->curves = curveCount;
        for (int k = 0; k < 4; ++k) s->fields[k] = fields[k];
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = outBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r;
        s->output = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active, &r);
        if (!s->output) { finish(r); return {}; }
        if (!pointCount) {
            s->proved = true; s->semantic = 0;
            finish(VK_SUCCESS); return candidate;
        }
        bi.size = 4;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        s->status = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->status) { finish(r); return {}; }
        s->dummy = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->dummy) { finish(r); return {}; }
        auto d = context->device();
        void* data = nullptr;
        r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        std::memset(data, 0, 4);
        vkUnmapMemory(d, s->status->memory());
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 11u};
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetAllocateInfo da{};
        da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        auto descriptorLayout = native_->descriptors;
        auto pipelineLayout = native_->layout;
        auto pipeline = native_->pipeline;
        da.descriptorPool = s->descriptors;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &descriptorLayout;
        VkDescriptorSet set;
        r = vkAllocateDescriptorSets(d, &da, &set);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorBufferInfo infos[11] = {{s->input->buffer(), 0, outBytes},
            {s->output->buffer(), 0, outBytes},
            {s->owners->buffer(), 0, VkDeviceSize(pointCount) * 4},
            {VK_NULL_HANDLE, 0, 0},
            {VK_NULL_HANDLE, 0, 0},
            {VK_NULL_HANDLE, 0, 0},
            {VK_NULL_HANDLE, 0, 0},
            {s->status->buffer(), 0, 4},
            {s->profile ? s->profile->buffer() : s->dummy->buffer(), 0, s->profile ? 257u * 4u : 4u},
            {s->hairT ? s->hairT->buffer() : s->dummy->buffer(), 0, s->hairT ? VkDeviceSize(pointCount) * 4u : 4u},
            {s->curveOffsets ? s->curveOffsets->buffer() : s->dummy->buffer(), 0,
                s->curveOffsets ? (VkDeviceSize(curveCount) + 1u) * 4u : 4u}};
        for (int k = 0; k < 4; ++k) {
            if (s->fields[k]) {
                infos[3 + k].buffer = s->fields[k]->buffer();
                infos[3 + k].range = DomSlots(doms[k], curveCount, pointCount) * 8;
            } else {
                infos[3 + k].buffer = s->dummy->buffer();
                infos[3 + k].range = 4;
            }
        }
        VkWriteDescriptorSet writes[11]{};
        for (uint32_t i = 0; i != 11u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 11, writes, 0, nullptr);
        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        VkCommandBuffer command;
        r = vkAllocateCommandBuffers(d, &ca, &command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        BarrierBefore(command);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &set, 0, nullptr);
        struct Push {
            uint32_t pointCount, curveCount;
            float widthLiteral, maskLiteral;
            uint32_t widthDom, maskDom, replaceDom, enabledDom;
            uint32_t replaceLiteral, enabledLiteral;
            float baseLiteral;
            uint32_t hasProfile, hasHairT, hasOffsets;
        };
        static_assert(sizeof(Push) == 56, "Shader ABI");
        Push push{pointCount, curveCount, controls.widthLiteral, controls.maskLiteral,
            controls.widthDom, controls.maskDom, controls.replaceDom, controls.enabledDom,
            controls.replaceLiteral, controls.enabledLiteral, controls.baseLiteral,
            s->profile ? 1u : 0u, s->hairT ? 1u : 0u, s->curveOffsets ? 1u : 0u};
        vkCmdPushConstants(
            command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 56, &push);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        BarrierAfter(command);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try { admitted = beforeSubmit(); } catch (...) {
                finish(VK_ERROR_UNKNOWN);
                return {};
            }
            if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        }
        r = s->output->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->status->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->dummy->MarkSubmitted(s->fence, s);
        if (r != VK_SUCCESS) {
            s->pending = true;
            candidate->Quarantine();
            finish(r);
            return {};
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        s->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

struct ExprVkExprOpPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (pipeline) vkDestroyPipeline(d, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(d, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(d, shader, nullptr);
    }
};
struct ExprVkExprOpPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inPoints, inWidths, owners, field, maskField;
    std::shared_ptr<ChargedBuffer> outPoints, outWidths, status, dummy;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

ExprVkExprOpPipeline::ExprVkExprOpPipeline(std::shared_ptr<Native> n)
    : native_(std::move(n)) {}
ExprVkExprOpPipeline::~ExprVkExprOpPipeline() = default;
std::shared_ptr<DeviceContext> const& ExprVkExprOpPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<ExprVkExprOpPipeline> ExprVkExprOpPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256 ||
        physical.limits.maxPushConstantsSize < 24)
        return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        VkShaderModuleCreateInfo sm{};
        sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t);
        sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetLayoutBinding bindings[8]{};
        for (uint32_t i = 0; i != 8; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 8;
        ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &n->descriptors;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(d, &pl, nullptr, &n->layout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkComputePipelineCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = n->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = n->shader;
        cp.stage.pName = "main";
        r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->pipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        finish(VK_SUCCESS);
        return std::shared_ptr<ExprVkExprOpPipeline>(
            new ExprVkExprOpPipeline(std::move(n)));
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

ExprVkExprOpPipeline::Candidate::Candidate(std::shared_ptr<State> s)
    : state_(std::move(s)) {}
ExprVkExprOpPipeline::Candidate::~Candidate() {
    if (state_ && state_->pending) Quarantine();
}
void ExprVkExprOpPipeline::Candidate::Quarantine() noexcept {
    if (!state_ || !state_->pending || state_->lost) return;
    state_->lost = true;
    state_->outPoints->Quarantine();
    state_->outWidths->Quarantine();
    state_->status->Quarantine();
    state_->dummy->Quarantine();
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ExprVkExprOpPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> ExprVkExprOpPipeline::Candidate::outPoints() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->outPoints : nullptr;
}
std::shared_ptr<const ChargedBuffer> ExprVkExprOpPipeline::Candidate::outWidths() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->outWidths : nullptr;
}
std::shared_ptr<DeviceContext> ExprVkExprOpPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}
VkResult ExprVkExprOpPipeline::Candidate::Wait(uint64_t timeoutNs,
                                                      uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkWaitForFences(
            s.native->context->device(), 1, &s.fence, VK_TRUE, timeoutNs);
        if (r != VK_SUCCESS) {
            if (r != VK_TIMEOUT) Quarantine();
            return r;
        }
    }
    return Poll(semanticStatus);
}
VkResult ExprVkExprOpPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        r = s.outPoints->PollComplete();
        if (r == VK_SUCCESS) r = s.outWidths->PollComplete();
        if (r == VK_SUCCESS) r = s.status->PollComplete();
        if (r == VK_SUCCESS) r = s.dummy->PollComplete();
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        s.pending = false;
        void* data = nullptr;
        r = vkMapMemory(s.native->context->device(), s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.semantic, data, 4);
        vkUnmapMemory(s.native->context->device(), s.status->memory());
        s.proved = true;
    }
    if (semanticStatus) *semanticStatus = s.semantic;
    return VK_SUCCESS;
}

std::unique_ptr<ExprVkExprOpPipeline::Candidate> ExprVkExprOpPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> inPoints, std::shared_ptr<const ChargedBuffer> inWidths,
    std::shared_ptr<const ChargedBuffer> owners, std::shared_ptr<const ChargedBuffer> field,
    std::shared_ptr<const ChargedBuffer> maskField, uint32_t pointCount,
    uint32_t curveCount, uint32_t returnType, bool perCurve, uint32_t maskDom,
    float maskLiteral, VkResult* result, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    if (!pointCount || !curveCount || returnType > 1 || !ValidDom(maskDom) ||
        !std::isfinite(maskLiteral))
        return {};
    if (!ValidInput(context, inPoints, VkDeviceSize(pointCount) * 12) ||
        !ValidInput(context, inWidths, VkDeviceSize(pointCount) * 4) ||
        !ValidInput(context, owners, VkDeviceSize(pointCount) * 4))
        return {};
    uint32_t const elements = perCurve ? curveCount : pointCount;
    uint32_t const components = returnType == 0 ? 3 : 1;
    if (!ValidInput(context, field, uint64_t(elements) * components * 8)) return {};
    if (maskDom == 0) {
        if (maskField) return {};
    } else if (!ValidInput(context, maskField, DomSlots(maskDom, curveCount, pointCount) * 8))
        return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t groups = (uint64_t(pointCount) + 255) / 256;
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};
    VkDeviceSize const pointsBytes = VkDeviceSize(pointCount) * 12;
    VkDeviceSize const widthsBytes = VkDeviceSize(pointCount) * 4;
    if (pointsBytes > properties.limits.maxStorageBufferRange ||
        widthsBytes > properties.limits.maxStorageBufferRange)
        return {};
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->inPoints = std::move(inPoints);
        s->inWidths = std::move(inWidths);
        s->owners = std::move(owners);
        s->field = std::move(field);
        s->maskField = std::move(maskField);
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r;
        bi.size = pointsBytes;
        s->outPoints = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active, &r);
        if (!s->outPoints) { finish(r); return {}; }
        bi.size = widthsBytes;
        s->outWidths = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active, &r);
        if (!s->outWidths) { finish(r); return {}; }
        bi.size = 4;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        s->status = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->status) { finish(r); return {}; }
        s->dummy = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->dummy) { finish(r); return {}; }
        auto d = context->device();
        void* data = nullptr;
        r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        std::memset(data, 0, 4);
        vkUnmapMemory(d, s->status->memory());
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8u};
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetAllocateInfo da{};
        da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        auto descriptorLayout = native_->descriptors;
        auto pipelineLayout = native_->layout;
        auto pipeline = native_->pipeline;
        da.descriptorPool = s->descriptors;
        da.descriptorSetCount = 1;
        da.pSetLayouts = &descriptorLayout;
        VkDescriptorSet set;
        r = vkAllocateDescriptorSets(d, &da, &set);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkBuffer const maskBuffer =
            s->maskField ? s->maskField->buffer() : s->dummy->buffer();
        VkDeviceSize const maskRange = s->maskField ? s->maskField->sizeBytes() : 4;
        VkDescriptorBufferInfo infos[8] = {{s->inPoints->buffer(), 0, pointsBytes},
            {s->outPoints->buffer(), 0, pointsBytes},
            {s->inWidths->buffer(), 0, widthsBytes},
            {s->outWidths->buffer(), 0, widthsBytes},
            {s->owners->buffer(), 0, VkDeviceSize(pointCount) * 4},
            {s->field->buffer(), 0, s->field->sizeBytes()},
            {maskBuffer, 0, maskRange},
            {s->status->buffer(), 0, 4}};
        VkWriteDescriptorSet writes[8]{};
        for (uint32_t i = 0; i != 8u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 8, writes, 0, nullptr);
        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        VkCommandBuffer command;
        r = vkAllocateCommandBuffers(d, &ca, &command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        BarrierBefore(command);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &set, 0, nullptr);
        struct Push {
            uint32_t pointCount, curveCount, returnType, rate, maskDom;
            float maskLiteral;
        };
        static_assert(sizeof(Push) == 24, "Shader ABI");
        Push push{pointCount, curveCount, returnType, perCurve ? 1u : 0u, maskDom,
            maskLiteral};
        vkCmdPushConstants(
            command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 24, &push);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        BarrierAfter(command);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try { admitted = beforeSubmit(); } catch (...) {
                finish(VK_ERROR_UNKNOWN);
                return {};
            }
            if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        }
        r = s->outPoints->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->outWidths->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->status->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->dummy->MarkSubmitted(s->fence, s);
        if (r != VK_SUCCESS) {
            s->pending = true;
            candidate->Quarantine();
            finish(r);
            return {};
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        s->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

} // namespace usdGen::vulkan
