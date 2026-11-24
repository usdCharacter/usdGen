// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "pointOverridePipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace usdGen::vulkan {
struct PointOverridePipeline::Native {
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

struct PointOverridePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> base, indices, replacements;
    std::shared_ptr<ChargedBuffer> output, status, occupancy;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer cb1 = VK_NULL_HANDLE, cb2 = VK_NULL_HANDLE, cb3 = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;   // PointOverridePipeline::Semantic ordinal
    uint32_t baseN = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (cb3) vkFreeCommandBuffers(d, commands, 1, &cb3);
        if (cb2) vkFreeCommandBuffers(d, commands, 1, &cb2);
        if (cb1) vkFreeCommandBuffers(d, commands, 1, &cb1);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

PointOverridePipeline::PointOverridePipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
PointOverridePipeline::~PointOverridePipeline() = default;
std::shared_ptr<DeviceContext> const& PointOverridePipeline::context() const noexcept {
    return native_->context;
}

namespace {
PointOverridePipeline::Semantic MapSemantic(uint32_t error) {
    // CUDA status-code mapping: kBadIndex=1 -> InvalidArgument,
    // kNonFinite=2 -> NonFinite, kDuplicate=3 -> Duplicate.
    switch (error) {
        case 0u: return PointOverridePipeline::Semantic::Ok;
        case 2u: return PointOverridePipeline::Semantic::NonFinite;
        case 3u: return PointOverridePipeline::Semantic::Duplicate;
        default: return PointOverridePipeline::Semantic::InvalidArgument;
    }
}
} // namespace

std::shared_ptr<PointOverridePipeline> PointOverridePipeline::Create(
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
        VkDescriptorSetLayoutBinding bindings[6]{};
        for (uint32_t i = 0; i != 6; ++i) {
            bindings[i].binding = i; bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 6; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 12};
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
        auto pipeline = std::shared_ptr<PointOverridePipeline>(new PointOverridePipeline(std::move(n)));
        finish(VK_SUCCESS); return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

PointOverridePipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
PointOverridePipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }
void PointOverridePipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PointOverridePipeline::Candidate::output() const noexcept {
    return state_->proved && state_->semantic == uint32_t(Semantic::Ok) ? state_->output : nullptr;
}
bool PointOverridePipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved &&
        state_->semantic == uint32_t(Semantic::Ok);
}
std::shared_ptr<const ChargedBuffer> PointOverridePipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->base : nullptr;
}
std::shared_ptr<DeviceContext> PointOverridePipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PointOverridePipeline::Candidate::Poll(Semantic* status) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (s.pending && !s.proved) {
        // Phase 3 scatter; probe the fence. The status buffer (error) is left
        // at 0 after clean validation; reading it keeps publication proof on
        // the same path as the synchronous validation readbacks.
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
    if (status) *status = s.semantic == UINT32_MAX
        ? Semantic::InvalidArgument : static_cast<Semantic>(s.semantic);
    return VK_SUCCESS;
}

std::unique_ptr<PointOverridePipeline::Candidate> PointOverridePipeline::Begin(
    std::shared_ptr<const ChargedBuffer> base,
    std::shared_ptr<const ChargedBuffer> indices,
    std::shared_ptr<const ChargedBuffer> replacements,
    uint32_t baseN, uint32_t slotN, VkResult* result, BeforeSubmit beforeSubmit) {
    return BeginInternal(std::move(base), std::move(indices), std::move(replacements),
                         baseN, slotN, result, std::move(beforeSubmit));
}

std::unique_ptr<PointOverridePipeline::Candidate> PointOverridePipeline::BeginInternal(
    std::shared_ptr<const ChargedBuffer> base, std::shared_ptr<const ChargedBuffer> indices,
    std::shared_ptr<const ChargedBuffer> replacements, uint32_t baseN, uint32_t slotN,
    VkResult* result, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    // Host Begin pre-checks (mirror CudaPointOverride::Apply validation).
    const bool ok =
        indices && replacements && base &&
        // >= (not ==): Vulkan cannot materialize zero-size buffers, so empty
        // (slotN == 0) planes arrive as bindable minimum-size planes; the
        // shader never reads past slotN. Matches the rf check-lambda contract.
        indices->sizeBytes() >= VkDeviceSize(slotN) * sizeof(int32_t) &&
        replacements->sizeBytes() >= VkDeviceSize(slotN) * sizeof(float) * 3 &&
        base->sizeBytes() >= VkDeviceSize(baseN) * sizeof(float) * 3 &&
        slotN <= size_t(std::numeric_limits<int32_t>::max()) &&
        slotN <= baseN &&
        indices->context() == context && replacements->context() == context &&
        base->context() == context && !indices->unproven() && !replacements->unproven() &&
        !base->unproven() && indices->buffer() && replacements->buffer() && base->buffer();
    if (!ok) return {};
    auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((base->usage() & usage) != usage || (indices->usage() & usage) != usage ||
        (replacements->usage() & usage) != usage) return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    const uint64_t baseBytes = uint64_t(baseN) * sizeof(float) * 3;
    const uint64_t validateWork = uint64_t(std::max(baseN, slotN));
    const uint64_t groups = std::max<uint64_t>(1, (validateWork + 255u) / 256u);
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};

    auto s = std::make_shared<Candidate::State>();
    s->native = native_;
    s->base = std::move(base); s->indices = std::move(indices);
    s->replacements = std::move(replacements);
    s->baseN = baseN;
    s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
    auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
    auto d = context->device();

    // Output buffer: private full-point buffer (copy + scatter target).
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = std::max<VkDeviceSize>(baseBytes, 4);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r;
    s->output = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!s->output) { finish(r); return {}; }
    bi.size = 4; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    s->status = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch, &r);
    if (!s->status) { finish(r); return {}; }
    if (baseN) {
        bi.size = VkDeviceSize(baseN) * sizeof(uint32_t);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        s->occupancy = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->occupancy) { finish(r); return {}; }
    }
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6u};
    VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
    r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    VkDescriptorSetAllocateInfo da{}; da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    auto descriptorLayout = native_->descriptors; auto pipelineLayout = native_->layout;
    da.descriptorPool = s->descriptors; da.descriptorSetCount = 1; da.pSetLayouts = &descriptorLayout;
    VkDescriptorSet set;
    r = vkAllocateDescriptorSets(d, &da, &set);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    VkDescriptorBufferInfo infos[6] = {
        {s->base->buffer(), 0, s->base->sizeBytes()},
        {s->indices->buffer(), 0, s->indices->sizeBytes()},
        {s->replacements->buffer(), 0, s->replacements->sizeBytes()},
        {s->output->buffer(), 0, s->output->sizeBytes()},
        {s->status->buffer(), 0, 4},
        {s->occupancy ? s->occupancy->buffer() : s->output->buffer(), 0,
         s->occupancy ? s->occupancy->sizeBytes() : s->output->sizeBytes()}};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i != 6u; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(d, 6, writes, 0, nullptr);
    VkCommandPoolCreateInfo pc{}; pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pc.queueFamilyIndex = context->computeQueueFamily();
    r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(d, &ca, &s->cb1) != VK_SUCCESS ||
        vkAllocateCommandBuffers(d, &ca, &s->cb2) != VK_SUCCESS ||
        vkAllocateCommandBuffers(d, &ca, &s->cb3) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(d, &fi, nullptr, &s->fence);
    if (r != VK_SUCCESS) { finish(r); return {}; }
    auto submitAndWait = [&](VkCommandBuffer cb) -> VkResult {
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
        s->pending = true;
        if (vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence) != VK_SUCCESS)
            return VK_ERROR_DEVICE_LOST;
        // Synchronous wait: required to gate phase 3 on phases 1-2 before any
        // output write is recorded (strict-ORDER parity).
        if (vkWaitForFences(d, 1, &s->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            return VK_ERROR_DEVICE_LOST;
        s->pending = false;
        if (vkResetFences(d, 1, &s->fence) != VK_SUCCESS)
            return VK_ERROR_DEVICE_LOST;
        return VK_SUCCESS;
    };
    auto readError = [&](uint32_t* out) -> VkResult {
        void* data = nullptr;
        if (vkMapMemory(d, s->status->memory(), 0, 4, 0, &data) != VK_SUCCESS)
            return VK_ERROR_DEVICE_LOST;
        std::memcpy(out, data, 4);
        vkUnmapMemory(d, s->status->memory());
        return VK_SUCCESS;
    };
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // Broad pre-submit barrier (matches widthPipeline.cpp): HOST|ALL_COMMANDS
    // source stages cover its HOST_WRITE|TRANSFER_WRITE|SHADER_WRITE accesses.
    VkMemoryBarrier shaderBarrier{}; shaderBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    shaderBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    shaderBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    // Narrow compute->compute carry between phases (status/register writes).
    VkMemoryBarrier phaseBarrier{}; phaseBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    phaseBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    phaseBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    // Narrow transfer->compute for fills/copy preceding a dispatch.
    VkMemoryBarrier fillBarrier{}; fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    VkMemoryBarrier hostBarrier{}; hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    struct Controls { uint32_t baseN; uint32_t slotN; uint32_t phase; } controls{baseN, slotN, 0u};
    static_assert(sizeof(Controls) == 12, "Shader ABI");

    // ---------- Phase 1: validate ----------
    if (vkBeginCommandBuffer(s->cb1, &begin) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    vkCmdPipelineBarrier(s->cb1, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &shaderBarrier, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(s->cb1, s->status->buffer(), 0, 4, 0u);   // reset error
    // Fill (TRANSFER_WRITE) must be visible to the validate dispatch before it
    // runs; otherwise the shader may observe a stale status register.
    vkCmdPipelineBarrier(s->cb1, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
    controls.phase = 0u;
    vkCmdBindPipeline(s->cb1, VK_PIPELINE_BIND_POINT_COMPUTE, native_->pipeline);
    vkCmdBindDescriptorSets(s->cb1, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(s->cb1, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &controls);
    vkCmdDispatch(s->cb1, uint32_t(groups), 1, 1);
    vkCmdPipelineBarrier(s->cb1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(s->cb1) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    if (beforeSubmit) {
        bool admitted = false;
        try { admitted = beforeSubmit(); }
        catch (...) { finish(VK_ERROR_UNKNOWN); return {}; }
        if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    }
    if (submitAndWait(s->cb1) != VK_SUCCESS) { finish(VK_ERROR_DEVICE_LOST); return {}; }
    uint32_t error = 0;
    if (readError(&error) != VK_SUCCESS) { finish(VK_ERROR_DEVICE_LOST); return {}; }
    if (error) {
        // Validation failed: no output writes were ever recorded. Report the
        // semantic failure synchronously; the candidate never arms pending.
        s->proved = true;
        s->semantic = uint32_t(MapSemantic(error));
        finish(VK_SUCCESS);
        return candidate;
    }

    // ---------- Phase 2: duplicate detection (only when edits exist) ----------
    if (slotN) {
        if (vkBeginCommandBuffer(s->cb2, &begin) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        // Carry phase-1 compute writes (status register) into phase 2 before
        // the occupancy fill; the fill then gets its own transfer->compute barrier.
        vkCmdPipelineBarrier(s->cb2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &phaseBarrier, 0, nullptr, 0, nullptr);
        // Occupancy sentinel fill (0xFFFFFFFF) before the duplicate kernel.
        vkCmdFillBuffer(s->cb2, s->occupancy->buffer(), 0, s->occupancy->sizeBytes(), 0xFFFFFFFFu);
        vkCmdPipelineBarrier(s->cb2, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        controls.phase = 1u;
        vkCmdBindPipeline(s->cb2, VK_PIPELINE_BIND_POINT_COMPUTE, native_->pipeline);
        vkCmdBindDescriptorSets(s->cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(s->cb2, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &controls);
        vkCmdDispatch(s->cb2, uint32_t(groups), 1, 1);
        vkCmdPipelineBarrier(s->cb2, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &hostBarrier, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(s->cb2) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        if (submitAndWait(s->cb2) != VK_SUCCESS) { finish(VK_ERROR_DEVICE_LOST); return {}; }
        if (readError(&error) != VK_SUCCESS) { finish(VK_ERROR_DEVICE_LOST); return {}; }
        if (error) {
            s->proved = true;
            s->semantic = uint32_t(MapSemantic(error));
            finish(VK_SUCCESS);
            return candidate;
        }
    }

    // ---------- Phase 3: copy base then scatter (recorded only when clean) ----------
    if (vkBeginCommandBuffer(s->cb3, &begin) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    if (baseBytes) {
        VkBufferCopy region{}; region.size = baseBytes;
        vkCmdCopyBuffer(s->cb3, s->base->buffer(), s->output->buffer(), 1, &region);
    }
    if (slotN) {
        vkCmdPipelineBarrier(s->cb3, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        controls.phase = 2u;
        vkCmdBindPipeline(s->cb3, VK_PIPELINE_BIND_POINT_COMPUTE, native_->pipeline);
        vkCmdBindDescriptorSets(s->cb3, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(s->cb3, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, &controls);
        vkCmdDispatch(s->cb3, uint32_t(groups), 1, 1);
    }
    VkMemoryBarrier outputBarrier = hostBarrier;
    outputBarrier.srcAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(s->cb3, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &outputBarrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(s->cb3) != VK_SUCCESS) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &s->cb3;
    s->pending = true;   // arm retention before native submission
    r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
    if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
    finish(VK_SUCCESS); return candidate;
}
} // namespace usdGen::vulkan