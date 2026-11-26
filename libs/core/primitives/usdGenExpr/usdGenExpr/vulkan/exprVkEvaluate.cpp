// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkEvaluate.h"

#include <cstring>
#include <limits>
#include <new>

namespace usdGen::vulkan {

struct ExprVkEvaluatePipeline::Native {
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

struct ExprVkEvaluatePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> program, fields, owners;
    std::shared_ptr<ChargedBuffer> output, status;
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

ExprVkEvaluatePipeline::ExprVkEvaluatePipeline(std::shared_ptr<Native> n)
    : native_(std::move(n)) {}
ExprVkEvaluatePipeline::~ExprVkEvaluatePipeline() = default;
std::shared_ptr<DeviceContext> const& ExprVkEvaluatePipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<ExprVkEvaluatePipeline> ExprVkEvaluatePipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    // The shader emits Float64 and Int64 SPIR-V. Float64 carries the trusted
    // factory flag; Int64 is enabled by the factory wherever Float64 is (see
    // the defaultProvider fragment) and re-checked for physical support here.
    if (!context->shaderFloat64Enabled()) return {};
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(context->physicalDevice(), &features);
    if (features.shaderInt64 != VK_TRUE) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256) return {};
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
        VkDescriptorSetLayoutBinding bindings[5]{};
        for (uint32_t i = 0; i != 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 5;
        ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &n->descriptors;
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
        return std::shared_ptr<ExprVkEvaluatePipeline>(new ExprVkEvaluatePipeline(std::move(n)));
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

ExprVkEvaluatePipeline::Candidate::Candidate(std::shared_ptr<State> s)
    : state_(std::move(s)) {}
ExprVkEvaluatePipeline::Candidate::~Candidate() {
    if (state_ && state_->pending) Quarantine();
}
void ExprVkEvaluatePipeline::Candidate::Quarantine() noexcept {
    if (!state_ || !state_->pending || state_->lost) return;
    state_->lost = true;
    state_->output->Quarantine();
    state_->status->Quarantine();
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ExprVkEvaluatePipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> ExprVkEvaluatePipeline::Candidate::output() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->output : nullptr;
}
std::shared_ptr<DeviceContext> ExprVkEvaluatePipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}
VkResult ExprVkEvaluatePipeline::Candidate::Wait(uint64_t timeoutNs,
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
VkResult ExprVkEvaluatePipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        r = s.output->PollComplete();
        if (r == VK_SUCCESS) r = s.status->PollComplete();
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

std::unique_ptr<ExprVkEvaluatePipeline::Candidate> ExprVkEvaluatePipeline::Begin(
    std::shared_ptr<const ChargedBuffer> program, std::shared_ptr<const ChargedBuffer> fields,
    std::shared_ptr<const ChargedBuffer> owners, uint32_t count, uint32_t outSlots,
    VkResult* result, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    auto validInput = [&](std::shared_ptr<const ChargedBuffer> const& buffer,
                          VkDeviceSize minimum) {
        return buffer && buffer->context() == context && !buffer->unproven() &&
            buffer->buffer() && (buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
            buffer->sizeBytes() >= minimum;
    };
    if (uint64_t(outSlots) > uint64_t(std::numeric_limits<uint32_t>::max())) return {};
    VkDeviceSize outBytes = VkDeviceSize(outSlots) * 8;
    if (outSlots && !validInput(program, 96)) return {};
    if (outSlots && !validInput(fields, 8)) return {};
    if (outSlots && !validInput(owners, 4)) return {};
    if (!outSlots && (!program || !fields || !owners)) return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t groups = (uint64_t(count) + 255) / 256;
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};
    if (outBytes > properties.limits.maxStorageBufferRange) return {};
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->program = std::move(program);
        s->fields = std::move(fields);
        s->owners = std::move(owners);
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        if (!count || !outSlots) {
            s->proved = true;
            s->semantic = 0;
            finish(VK_SUCCESS);
            return candidate;
        }
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = outBytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r;
        s->output = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active, &r);
        if (!s->output) { finish(r); return {}; }
        bi.size = 4;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        s->status = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->status) { finish(r); return {}; }
        auto d = context->device();
        void* data = nullptr;
        r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        std::memset(data, 0, 4);
        vkUnmapMemory(d, s->status->memory());
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5u};
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
        VkDescriptorBufferInfo infos[5] = {
            {s->program->buffer(), 0, s->program->sizeBytes()},
            {s->fields->buffer(), 0, s->fields->sizeBytes()},
            {s->owners->buffer(), 0, s->owners->sizeBytes()},
            {s->output->buffer(), 0, outBytes},
            {s->status->buffer(), 0, 4}};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t i = 0; i != 5u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 5, writes, 0, nullptr);
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
        VkMemoryBarrier before{};
        before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask =
            VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &set, 0, nullptr);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        VkMemoryBarrier after{};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask =
            VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0,
            nullptr, 0, nullptr);
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
