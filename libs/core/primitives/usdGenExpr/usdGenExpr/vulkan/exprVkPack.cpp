// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "exprVkPack.h"
#include <algorithm>
#include <new>

namespace usdGen::vulkan {
struct ExprVkPackPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~Native() {
        auto device = context->device();
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(device, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(device, shader, nullptr);
    }
};
struct ExprVkPackPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> input;
    std::shared_ptr<ChargedBuffer> output;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, proved = false, lost = false;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};
ExprVkPackPipeline::~ExprVkPackPipeline() = default;
std::shared_ptr<DeviceContext> const& ExprVkPackPipeline::context() const noexcept {
    return native_->context;
}
std::shared_ptr<ExprVkPackPipeline> ExprVkPackPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code, VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context || code.size() < 5 || code[0] != 0x07230203u) return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    if (properties.limits.maxComputeWorkGroupInvocations < 256 ||
        properties.limits.maxComputeWorkGroupSize[0] < 256) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        auto device = native->context->device();
        VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(device, &sm, nullptr, &native->shader);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkDescriptorSetLayoutBinding bindings[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 2; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(device, &ds, nullptr, &native->descriptors);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
        VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &native->descriptors;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(device, &pl, nullptr, &native->layout);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = native->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = native->shader; cp.stage.pName = "main";
        r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &native->pipeline);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (result) *result = VK_SUCCESS;
        return std::shared_ptr<ExprVkPackPipeline>(new ExprVkPackPipeline(std::move(native)));
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}
ExprVkPackPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }
void ExprVkPackPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    state_->output->Quarantine();
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ExprVkPackPipeline::Candidate::succeeded() const noexcept {
    return state_->proved && !state_->lost;
}
std::shared_ptr<const ChargedBuffer> ExprVkPackPipeline::Candidate::output() const noexcept {
    return succeeded() ? state_->output : nullptr;
}
VkResult ExprVkPackPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        r = s.output->PollComplete();
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        s.pending = false; s.proved = true;
    }
    if (semanticStatus) *semanticStatus = 0;
    return VK_SUCCESS;
}
std::unique_ptr<ExprVkPackPipeline::Candidate> ExprVkPackPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> input, uint32_t count,
    VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    if (count && (!input || input->context() != context || input->unproven() ||
        !(input->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        input->sizeBytes() < VkDeviceSize(count) * 8u)) return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t const groups = (uint64_t(count) + 255u) / 256u;
    if (groups > properties.limits.maxComputeWorkGroupCount[0] ||
        VkDeviceSize(count) * 8u > properties.limits.maxStorageBufferRange) return {};
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_; s->input = std::move(input);
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        auto device = context->device();
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = std::max<VkDeviceSize>(4, VkDeviceSize(count) * 4);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r;
        s->output = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->output) { if (result) *result = r; return {}; }
        if (!count) { s->proved = true; if (result) *result = VK_SUCCESS; return candidate; }
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(device, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkDescriptorSetAllocateInfo da{}; da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        da.descriptorPool = s->descriptors; da.descriptorSetCount = 1;
        da.pSetLayouts = &native_->descriptors;
        VkDescriptorSet set;
        r = vkAllocateDescriptorSets(device, &da, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkDescriptorBufferInfo infos[2] = {{s->input->buffer(), 0, VkDeviceSize(count) * 8},
            {s->output->buffer(), 0, VkDeviceSize(count) * 4}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        VkCommandPoolCreateInfo pc{}; pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(device, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        VkCommandBuffer command;
        r = vkAllocateCommandBuffers(device, &ca, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, native_->pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, native_->layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, native_->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &count);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(device, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool accepted = false;
            try { accepted = beforeSubmit(); } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!accepted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        s->pending = true;
        r = s->output->MarkSubmitted(s->fence, s);
        if (r != VK_SUCCESS) { candidate->Quarantine(); if (result) *result = r; return {}; }
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); if (result) *result = r; return {}; }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}
} // namespace usdGen::vulkan
