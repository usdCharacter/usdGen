// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "rootFrameGatherPipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace usdGen::vulkan {

char const* RootFrameGatherSemanticName(RootFrameGatherSemantic s) noexcept {
    switch (s) {
        case RootFrameGatherSemantic::Ok:        return "Ok";
        case RootFrameGatherSemantic::MissingId: return "MissingId";
    }
    return "Invalid";
}

struct RootFrameGatherPipeline::Native {
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

struct RootFrameGatherPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    // Inputs (borrowed, kept alive for the dispatch).
    std::shared_ptr<const ChargedBuffer> sourceIds;
    std::shared_ptr<const ChargedBuffer> sourceOrigins;
    std::shared_ptr<const ChargedBuffer> sourceTangent;
    std::shared_ptr<const ChargedBuffer> sourceBinormal;
    std::shared_ptr<const ChargedBuffer> sourceNormal;
    std::shared_ptr<const ChargedBuffer> sourceValid;
    std::shared_ptr<const ChargedBuffer> sourceDrop;
    std::shared_ptr<const ChargedBuffer> survivorIds;
    // Outputs.
    std::shared_ptr<ChargedBuffer> outOrigin;
    std::shared_ptr<ChargedBuffer> outTangent;
    std::shared_ptr<ChargedBuffer> outBinormal;
    std::shared_ptr<ChargedBuffer> outNormal;
    std::shared_ptr<ChargedBuffer> outValid;
    std::shared_ptr<ChargedBuffer> outDrop;
    std::shared_ptr<ChargedBuffer> status;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;
    uint32_t survivorCount = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (cb) vkFreeCommandBuffers(d, commands, 1, &cb);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

RootFrameGatherPipeline::RootFrameGatherPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
RootFrameGatherPipeline::~RootFrameGatherPipeline() = default;
std::shared_ptr<DeviceContext> const& RootFrameGatherPipeline::context() const noexcept {
    return native_->context;
}

namespace {
RootFrameGatherSemantic MapSemantic(uint32_t code) {
    switch (code) {
        case 0u: return RootFrameGatherSemantic::Ok;
        case 1u: return RootFrameGatherSemantic::MissingId;
        default: return RootFrameGatherSemantic::MissingId;
    }
}
} // namespace

std::shared_ptr<RootFrameGatherPipeline> RootFrameGatherPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256) return {};
    try {
        auto n = std::make_shared<Native>(); n->context = std::move(context);
        auto d = n->context->device();
        VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        constexpr uint32_t kBindingCount = 15;
        VkDescriptorSetLayoutBinding bindings[kBindingCount]{};
        for (uint32_t i = 0; i < kBindingCount; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = kBindingCount; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &n->descriptors;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(d, &pl, nullptr, &n->layout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = n->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = n->shader; cp.stage.pName = "main";
        r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->pipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto pipeline = std::shared_ptr<RootFrameGatherPipeline>(
            new RootFrameGatherPipeline(std::move(n)));
        finish(VK_SUCCESS); return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

RootFrameGatherPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
RootFrameGatherPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }
void RootFrameGatherPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
RootFrameGatherPipeline::Candidate::Output
RootFrameGatherPipeline::Candidate::output() const noexcept {
    if (!state_->proved || state_->semantic != 0) return {};
    return Output{
        state_->outOrigin, state_->outTangent, state_->outBinormal,
        state_->outNormal, state_->outValid, state_->outDrop};
}
bool RootFrameGatherPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved &&
           state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> RootFrameGatherPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->sourceIds : nullptr;
}
std::shared_ptr<DeviceContext> RootFrameGatherPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}
uint32_t RootFrameGatherPipeline::Candidate::survivorCount() const noexcept {
    return state_ ? state_->survivorCount : 0;
}

VkResult RootFrameGatherPipeline::Candidate::Poll(RootFrameGatherSemantic* semantic) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (s.pending && !s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        s.pending = false;
        uint32_t error = 0;
        void* data = nullptr;
        r = vkMapMemory(s.native->context->device(), s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&error, data, 4);
        vkUnmapMemory(s.native->context->device(), s.status->memory());
        s.semantic = uint32_t(MapSemantic(error));
        s.proved = true;
    }
    if (semantic) *semantic = MapSemantic(s.semantic);
    return VK_SUCCESS;
}

std::unique_ptr<RootFrameGatherPipeline::Candidate> RootFrameGatherPipeline::Begin(
    BeginInfo info, VkResult* result, RootFrameGatherSemantic* semantic, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;

    // Validate inputs.
    const uint32_t baseN = info.baseCount;
    const uint32_t survN = info.survivorCount;
    const bool ok =
        info.sourceIds && info.sourceOrigins && info.sourceTangent &&
        info.sourceBinormal && info.sourceNormal && info.sourceValid &&
        info.sourceDrop && info.survivorIds &&
        info.sourceIds->context() == context &&
        info.sourceOrigins->context() == context &&
        info.sourceTangent->context() == context &&
        info.sourceBinormal->context() == context &&
        info.sourceNormal->context() == context &&
        info.sourceValid->context() == context &&
        info.sourceDrop->context() == context &&
        info.survivorIds->context() == context &&
        !info.sourceIds->unproven() && !info.sourceOrigins->unproven() &&
        !info.sourceTangent->unproven() && !info.sourceBinormal->unproven() &&
        !info.sourceNormal->unproven() && !info.sourceValid->unproven() &&
        !info.sourceDrop->unproven() && !info.survivorIds->unproven() &&
        info.sourceIds->buffer() && info.sourceOrigins->buffer() &&
        info.sourceTangent->buffer() && info.sourceBinormal->buffer() &&
        info.sourceNormal->buffer() && info.sourceValid->buffer() &&
        info.sourceDrop->buffer() && info.survivorIds->buffer();
    if (!ok) return {};

    const uint64_t f3Bytes = sizeof(float) * 3;
    const uint64_t u32Bytes = sizeof(uint32_t);
    const uint64_t u64Bytes = sizeof(uint64_t);
    // All size checks use >= (Vulkan minimum buffer size for 0-element allocs).
    if (baseN > 0) {
        if (info.sourceIds->sizeBytes() < VkDeviceSize(baseN) * u64Bytes ||
            info.sourceOrigins->sizeBytes() < VkDeviceSize(baseN) * f3Bytes ||
            info.sourceTangent->sizeBytes() < VkDeviceSize(baseN) * f3Bytes ||
            info.sourceBinormal->sizeBytes() < VkDeviceSize(baseN) * f3Bytes ||
            info.sourceNormal->sizeBytes() < VkDeviceSize(baseN) * f3Bytes ||
            info.sourceValid->sizeBytes() < VkDeviceSize(baseN) * u32Bytes ||
            info.sourceDrop->sizeBytes() < VkDeviceSize(baseN) * u32Bytes)
            return {};
    }
    if (survN > 0) {
        if (info.survivorIds->sizeBytes() < VkDeviceSize(survN) * u64Bytes)
            return {};
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    const uint64_t groups = std::max<uint64_t>(1, (uint64_t(survN) + 255u) / 256u);
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};

    auto s = std::make_shared<Candidate::State>();
    s->native = native_;
    s->sourceIds = info.sourceIds;
    s->sourceOrigins = info.sourceOrigins;
    s->sourceTangent = info.sourceTangent;
    s->sourceBinormal = info.sourceBinormal;
    s->sourceNormal = info.sourceNormal;
    s->sourceValid = info.sourceValid;
    s->sourceDrop = info.sourceDrop;
    s->survivorIds = info.survivorIds;
    s->survivorCount = survN;
    s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
    auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
    auto d = context->device();
    VkResult r;

    // Allocate outputs.
    const uint64_t outF3 = std::max<uint64_t>(VkDeviceSize(survN) * f3Bytes, 4u);
    const uint64_t outU32 = std::max<uint64_t>(VkDeviceSize(survN) * u32Bytes, 4u);
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    bi.size = uint64_t(outF3);
    s->outOrigin = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outOrigin) { finish(r); return {}; }
    s->outTangent = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outTangent) { finish(r); return {}; }
    s->outBinormal = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outBinormal) { finish(r); return {}; }
    s->outNormal = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outNormal) { finish(r); return {}; }
    bi.size = uint64_t(outU32);
    s->outValid = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outValid) { finish(r); return {}; }
    s->outDrop = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->outDrop) { finish(r); return {}; }

    // Status buffer (host-visible).
    bi.size = 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    s->status = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch, &r);
    if (!s->status) { finish(r); return {}; }

    // Descriptor pool + set.
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 15u};
    VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
    r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    VkDescriptorSetAllocateInfo da{}; da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = s->descriptors; da.descriptorSetCount = 1;
    da.pSetLayouts = &native_->descriptors;
    VkDescriptorSet set;
    r = vkAllocateDescriptorSets(d, &da, &set);
    if (r != VK_SUCCESS) { finish(r); return {}; }

    // Shader binding order: 0=origin, 1=tangent, 2=binormal, 3=normal,
    // 4=valid, 5=drop, 6=sourceIds, 7=survivorIds, 8-13=outputs, 14=status.
    VkDescriptorBufferInfo infos[15] = {
        {info.sourceOrigins->buffer(), 0, info.sourceOrigins->sizeBytes()},
        {info.sourceTangent->buffer(), 0, info.sourceTangent->sizeBytes()},
        {info.sourceBinormal->buffer(), 0, info.sourceBinormal->sizeBytes()},
        {info.sourceNormal->buffer(), 0, info.sourceNormal->sizeBytes()},
        {info.sourceValid->buffer(), 0, info.sourceValid->sizeBytes()},
        {info.sourceDrop->buffer(), 0, info.sourceDrop->sizeBytes()},
        {info.sourceIds->buffer(), 0, info.sourceIds->sizeBytes()},
        {info.survivorIds->buffer(), 0, info.survivorIds->sizeBytes()},
        {s->outOrigin->buffer(), 0, s->outOrigin->sizeBytes()},
        {s->outTangent->buffer(), 0, s->outTangent->sizeBytes()},
        {s->outBinormal->buffer(), 0, s->outBinormal->sizeBytes()},
        {s->outNormal->buffer(), 0, s->outNormal->sizeBytes()},
        {s->outValid->buffer(), 0, s->outValid->sizeBytes()},
        {s->outDrop->buffer(), 0, s->outDrop->sizeBytes()},
        {s->status->buffer(), 0, 4}};
    VkWriteDescriptorSet writes[15]{};
    for (uint32_t i = 0; i < 15u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d, 15, writes, 0, nullptr);

    // Command pool + fence.
    VkCommandPoolCreateInfo pc{}; pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pc.queueFamilyIndex = context->computeQueueFamily();
    r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d, &ca, &s->cb) != VK_SUCCESS) {
        finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {};
    }
    VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(d, &fi, nullptr, &s->fence);
    if (r != VK_SUCCESS) { finish(r); return {}; }

    // Record: fill status=0, barrier, dispatch gather.
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(s->cb, &begin) != VK_SUCCESS) {
        finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {};
    }
    VkMemoryBarrier hostBarrier{}; hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(s->cb, s->status->buffer(), 0, 4, 0u);
    VkMemoryBarrier fillBarrier = hostBarrier;
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier, 0, nullptr, 0, nullptr);

    struct Push { uint32_t survivorCount; uint32_t baseCount; };
    Push push{survN, baseN};
    static_assert(sizeof(Push) == 8, "Push constant ABI");
    vkCmdBindPipeline(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, native_->pipeline);
    vkCmdBindDescriptorSets(s->cb, VK_PIPELINE_BIND_POINT_COMPUTE, native_->layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(s->cb, native_->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push);
    vkCmdDispatch(s->cb, uint32_t(groups), 1, 1);

    VkMemoryBarrier doneBarrier{}; doneBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    doneBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    doneBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(s->cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &doneBarrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(s->cb) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }

    if (beforeSubmit) {
        bool admitted = false;
        try { admitted = beforeSubmit(); }
        catch (...) { finish(VK_ERROR_UNKNOWN); return {}; }
        if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    }
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &s->cb;
    s->pending = true;
    r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
    if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
    finish(VK_SUCCESS); return candidate;
}

} // namespace usdGen::vulkan
