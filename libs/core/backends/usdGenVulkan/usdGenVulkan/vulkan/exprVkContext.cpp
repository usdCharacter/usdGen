// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkContext.h"

#include <cstring>
#include <limits>
#include <new>

namespace usdGen::vulkan {

struct ExprVkContextPipeline::Native {
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

struct ExprVkContextPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputs[11];
    std::shared_ptr<ChargedBuffer> params, fields, owners, arc, status, dummy;
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

ExprVkContextPipeline::ExprVkContextPipeline(std::shared_ptr<Native> n)
    : native_(std::move(n)) {}
ExprVkContextPipeline::~ExprVkContextPipeline() = default;
std::shared_ptr<DeviceContext> const& ExprVkContextPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<ExprVkContextPipeline> ExprVkContextPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
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
        VkDescriptorSetLayoutBinding bindings[16]{};
        for (uint32_t i = 0; i != 16; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 16;
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
        return std::shared_ptr<ExprVkContextPipeline>(new ExprVkContextPipeline(std::move(n)));
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

ExprVkContextPipeline::Candidate::Candidate(std::shared_ptr<State> s)
    : state_(std::move(s)) {}
ExprVkContextPipeline::Candidate::~Candidate() {
    if (state_ && state_->pending) Quarantine();
}
void ExprVkContextPipeline::Candidate::Quarantine() noexcept {
    if (!state_ || !state_->pending || state_->lost) return;
    state_->lost = true;
    state_->params->Quarantine();
    state_->fields->Quarantine();
    state_->owners->Quarantine();
    state_->arc->Quarantine();
    state_->status->Quarantine();
    state_->dummy->Quarantine();
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ExprVkContextPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<ChargedBuffer> ExprVkContextPipeline::Candidate::fields() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->fields : nullptr;
}
std::shared_ptr<const ChargedBuffer> ExprVkContextPipeline::Candidate::owners() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->owners : nullptr;
}
std::shared_ptr<DeviceContext> ExprVkContextPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}
VkResult ExprVkContextPipeline::Candidate::Wait(uint64_t timeoutNs,
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
VkResult ExprVkContextPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        r = s.params->PollComplete();
        if (r == VK_SUCCESS) r = s.fields->PollComplete();
        if (r == VK_SUCCESS) r = s.owners->PollComplete();
        if (r == VK_SUCCESS) r = s.arc->PollComplete();
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

std::unique_ptr<ExprVkContextPipeline::Candidate> ExprVkContextPipeline::Begin(
    ExprVkCurveGeometry const& geometry, ExprVkGeometryChannels const& channels,
    expr::Domain domain, uint32_t count, ExprVkFieldLayout const& layout,
    ExprVkContextStatus* status, VkResult* result, BeforeSubmit beforeSubmit) {
    auto finish = [&](ExprVkContextStatus s, VkResult r) {
        if (status) *status = s;
        if (result) *result = r;
    };
    finish(ExprVkContextStatus::InvalidArgument, VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    if (domain != expr::Domain::Groom && domain != expr::Domain::Primitive &&
        domain != expr::Domain::Point)
        return {};
    size_t const curves = geometry.curveCount, points = geometry.pointCount;
    size_t const n =
        domain == expr::Domain::Groom ? 1 : domain == expr::Domain::Primitive ? curves : points;
    if (n > UINT32_MAX || count != n || layout.varCount < ExprVkEncoding::kBaseVariables ||
        layout.offsets.size() != layout.varCount || layout.counts.size() != layout.varCount ||
        layout.domains.size() != layout.varCount || layout.components.size() != layout.varCount)
        return {};
    if (layout.totalDoubles > (uint64_t(UINT32_MAX) >> 3)) return {};
    auto validView = [&](std::shared_ptr<const ChargedBuffer> const& buffer) {
        return !buffer ||
            (buffer->context() == context && !buffer->unproven() && buffer->buffer() &&
             (buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    };
    if (!validView(geometry.points) || !validView(geometry.restPoints) ||
        !validView(geometry.widths) || !validView(geometry.curveOffsets) ||
        !validView(geometry.stableIds) || !validView(channels.hairT) ||
        !validView(channels.rootUV) || !validView(channels.rootN) ||
        !validView(channels.rootT) || !validView(channels.rootB) ||
        !validView(channels.rootPrim))
        return {};
    auto bytes = [&](std::shared_ptr<const ChargedBuffer> const& buffer) {
        return buffer ? buffer->sizeBytes() : 0;
    };
    bool const hasRest = !!geometry.restPoints, hasWidths = !!geometry.widths;
    bool const hasHairT = !!channels.hairT, hasRootUV = !!channels.rootUV;
    bool const hasRootN = !!channels.rootN, hasRootT = !!channels.rootT;
    bool const hasRootB = !!channels.rootB, hasRootPrim = !!channels.rootPrim;
    if (domain != expr::Domain::Groom) {
        // Host-side admission mirrors CudaExpressionContext::BuildImpl.
        if ((curves == 0) != (points == 0) || curves > uint64_t(UINT32_MAX) - 1 ||
            points > uint64_t(UINT32_MAX) || !geometry.curveOffsets ||
            bytes(geometry.curveOffsets) != (curves + 1) * 4 ||
            (points && !geometry.points) || bytes(geometry.points) != points * 12 ||
            (curves && !geometry.stableIds) || bytes(geometry.stableIds) != curves * 8) {
            finish(ExprVkContextStatus::InvalidGeometry, VK_SUCCESS);
            return {};
        }
        if ((hasRest && bytes(geometry.restPoints) != points * 12) ||
            (hasWidths && bytes(geometry.widths) != points * 4)) {
            finish(ExprVkContextStatus::InvalidGeometry, VK_SUCCESS);
            return {};
        }
        if ((hasHairT && bytes(channels.hairT) != points * 4) ||
            (hasRootUV && bytes(channels.rootUV) != curves * 8)) {
            finish(ExprVkContextStatus::InvalidChannel, VK_SUCCESS);
            return {};
        }
        if ((hasRootN && bytes(channels.rootN) != curves * 12) ||
            (hasRootT && bytes(channels.rootT) != curves * 12) ||
            (hasRootB && bytes(channels.rootB) != curves * 12) ||
            (hasRootPrim && bytes(channels.rootPrim) != curves * 4)) {
            finish(ExprVkContextStatus::InvalidChannel, VK_SUCCESS);
            return {};
        }
    }
    // The layout must agree with the actual channels (the evaluator builds
    // both from the same presence set; a mismatch is a caller bug).
    auto materialized = [&](expr::Variable v) {
        return layout.offsets[static_cast<unsigned>(v)] != UINT64_MAX;
    };
    bool layoutOk = true;
    if (domain == expr::Domain::Groom) {
        for (unsigned v = 0; v < ExprVkEncoding::kBaseVariables; ++v) {
            bool const want = v == static_cast<unsigned>(expr::Variable::Value);
            bool const got = layout.offsets[v] != UINT64_MAX;
            layoutOk = layoutOk && (want == got);
        }
    } else {
        layoutOk = layoutOk && materialized(expr::Variable::P) &&
            materialized(expr::Variable::RootP) &&
            (materialized(expr::Variable::PRef) == hasRest) &&
            (materialized(expr::Variable::RootPRef) == hasRest) &&
            (materialized(expr::Variable::CWidth) == hasWidths) &&
            (materialized(expr::Variable::U) == hasRootUV) &&
            (materialized(expr::Variable::V) == hasRootUV) &&
            (materialized(expr::Variable::N) == hasRootN) &&
            (materialized(expr::Variable::NRef) == hasRootN) &&
            (materialized(expr::Variable::DPdu) == hasRootT) &&
            (materialized(expr::Variable::DPduRef) == hasRootT) &&
            (materialized(expr::Variable::DPdv) == hasRootB) &&
            (materialized(expr::Variable::DPdvRef) == hasRootB) &&
            (materialized(expr::Variable::FaceId) == hasRootPrim) &&
            materialized(expr::Variable::T) && materialized(expr::Variable::PrimIndex) &&
            materialized(expr::Variable::PrimCount) && materialized(expr::Variable::CLength) &&
            materialized(expr::Variable::IdLo) && materialized(expr::Variable::IdHi) &&
            materialized(expr::Variable::Id) && materialized(expr::Variable::Value) &&
            (materialized(expr::Variable::PointIndex) == (domain == expr::Domain::Point)) &&
            (materialized(expr::Variable::PointCount) == (domain == expr::Domain::Point));
    }
    if (!layoutOk) return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t const major = curves > points ? curves : points;
    uint64_t const groups0 = major == 0 ? 1 : (major + 255) / 256;
    uint64_t const groups1 = (uint64_t(count) + 255) / 256;
    if (groups0 > properties.limits.maxComputeWorkGroupCount[0] ||
        (count && groups1 > properties.limits.maxComputeWorkGroupCount[0]))
        return {};
    VkDeviceSize const paramsBytes = 32 + VkDeviceSize(layout.varCount) * 16;
    VkDeviceSize const fieldBytes =
        layout.totalDoubles ? VkDeviceSize(layout.totalDoubles) * 8 : 8;
    VkDeviceSize const ownersBytes = points ? VkDeviceSize(points) * 4 : 4;
    VkDeviceSize const arcBytes = points ? VkDeviceSize(points) * 8 : 8;
    if (paramsBytes > properties.limits.maxStorageBufferRange ||
        fieldBytes > properties.limits.maxStorageBufferRange ||
        ownersBytes > properties.limits.maxStorageBufferRange ||
        arcBytes > properties.limits.maxStorageBufferRange)
        return {};
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->inputs[0] = geometry.points;
        s->inputs[1] = geometry.restPoints;
        s->inputs[2] = geometry.widths;
        s->inputs[3] = geometry.curveOffsets;
        s->inputs[4] = geometry.stableIds;
        s->inputs[5] = channels.hairT;
        s->inputs[6] = channels.rootUV;
        s->inputs[7] = channels.rootN;
        s->inputs[8] = channels.rootT;
        s->inputs[9] = channels.rootB;
        s->inputs[10] = channels.rootPrim;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        VkResult r;
        auto makeHostVisible = [&](VkDeviceSize size, UsdGenExecutionResourceKind kind,
                                   std::shared_ptr<ChargedBuffer>* out) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = size;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            *out = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                kind, &r);
            return !!*out;
        };
        if (!makeHostVisible(4, UsdGenExecutionResourceKind::Scratch, &s->dummy) ||
            !makeHostVisible(paramsBytes, UsdGenExecutionResourceKind::Scratch, &s->params) ||
            !makeHostVisible(fieldBytes, UsdGenExecutionResourceKind::Active, &s->fields) ||
            !makeHostVisible(ownersBytes, UsdGenExecutionResourceKind::Active, &s->owners) ||
            !makeHostVisible(arcBytes, UsdGenExecutionResourceKind::Scratch, &s->arc)) {
            finish(ExprVkContextStatus::DeviceError, r);
            return {};
        }
        // Groom runs no dispatches (CUDA validates nothing for groom either).
        if (domain == expr::Domain::Groom) {
            s->proved = true;
            s->semantic = 0;
            finish(ExprVkContextStatus::Ok, VK_SUCCESS);
            return candidate;
        }
        VkBufferCreateInfo sbi{};
        sbi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        sbi.size = 4;
        sbi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        sbi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        s->status = ChargedBuffer::Create(context, sbi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->status) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        auto d = context->device();
        void* mapped = nullptr;
        std::vector<uint32_t> params(size_t(paramsBytes) / 4, 0);
        params[0] = uint32_t(curves);
        params[1] = uint32_t(points);
        params[2] = count;
        params[3] = static_cast<unsigned>(domain);
        params[4] = (hasRest ? 1u : 0u) | (hasWidths ? 2u : 0u) | (hasHairT ? 4u : 0u) |
            (hasRootUV ? 8u : 0u) | (hasRootN ? 16u : 0u) | (hasRootT ? 32u : 0u) |
            (hasRootB ? 64u : 0u) | (hasRootPrim ? 128u : 0u);
        params[5] = 0;
        params[6] = layout.varCount;
        params[7] = 0;
        for (unsigned v = 0; v < layout.varCount; ++v) {
            size_t const w = 8 + size_t(v) * 4;
            params[w] = uint32_t(layout.offsets[v]);
            params[w + 1] = uint32_t(layout.offsets[v] >> 32);
            params[w + 2] = layout.counts[v];
            params[w + 3] = static_cast<unsigned>(layout.domains[v]) |
                (uint32_t(layout.components[v]) << 8);
        }
        r = vkMapMemory(d, s->params->memory(), 0, paramsBytes, 0, &mapped);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        std::memcpy(mapped, params.data(), size_t(paramsBytes));
        vkUnmapMemory(d, s->params->memory());
        r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &mapped);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        std::memset(mapped, 0, 4);
        vkUnmapMemory(d, s->status->memory());
        r = vkMapMemory(d, s->fields->memory(), 0, size_t(fieldBytes), 0, &mapped);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        std::memset(mapped, 0, size_t(fieldBytes));
        vkUnmapMemory(d, s->fields->memory());
        auto pick = [&](std::shared_ptr<const ChargedBuffer> const& buffer) {
            return buffer ? buffer->buffer() : s->dummy->buffer();
        };
        auto pickSize = [&](std::shared_ptr<const ChargedBuffer> const& buffer) {
            return buffer ? buffer->sizeBytes() : VkDeviceSize(4);
        };
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16u};
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
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
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        VkDescriptorBufferInfo infos[16] = {
            {s->params->buffer(), 0, paramsBytes},
            {pick(geometry.points), 0, pickSize(geometry.points)},
            {pick(geometry.restPoints), 0, pickSize(geometry.restPoints)},
            {pick(geometry.widths), 0, pickSize(geometry.widths)},
            {pick(geometry.curveOffsets), 0, pickSize(geometry.curveOffsets)},
            {pick(geometry.stableIds), 0, pickSize(geometry.stableIds)},
            {pick(channels.hairT), 0, pickSize(channels.hairT)},
            {pick(channels.rootUV), 0, pickSize(channels.rootUV)},
            {pick(channels.rootN), 0, pickSize(channels.rootN)},
            {pick(channels.rootT), 0, pickSize(channels.rootT)},
            {pick(channels.rootB), 0, pickSize(channels.rootB)},
            {pick(channels.rootPrim), 0, pickSize(channels.rootPrim)},
            {s->fields->buffer(), 0, fieldBytes},
            {s->owners->buffer(), 0, ownersBytes},
            {s->arc->buffer(), 0, arcBytes},
            {s->status->buffer(), 0, 4}};
        VkWriteDescriptorSet writes[16]{};
        for (uint32_t i = 0; i != 16u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 16, writes, 0, nullptr);
        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands;
        ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 1;
        VkCommandBuffer command;
        r = vkAllocateCommandBuffers(d, &ca, &command);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
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
        vkCmdDispatch(command, uint32_t(groups0), 1, 1);
        if (count) {
            // Flip the phase word on-device, then run field materialization.
            VkMemoryBarrier mid{};
            mid.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mid.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mid.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mid, 0, nullptr, 0, nullptr);
            uint32_t const one = 1;
            vkCmdUpdateBuffer(command, s->params->buffer(), 20, 4, &one);
            VkMemoryBarrier upd{};
            upd.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            upd.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            upd.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upd, 0, nullptr, 0, nullptr);
            vkCmdDispatch(command, uint32_t(groups1), 1, 1);
        }
        VkMemoryBarrier after{};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        after.dstAccessMask =
            VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0,
            nullptr, 0, nullptr);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(ExprVkContextStatus::DeviceError, r); return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try { admitted = beforeSubmit(); } catch (...) {
                finish(ExprVkContextStatus::DeviceError, VK_ERROR_UNKNOWN);
                return {};
            }
            if (!admitted) {
                finish(ExprVkContextStatus::DeviceError, VK_ERROR_OUT_OF_DEVICE_MEMORY);
                return {};
            }
        }
        r = s->params->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->fields->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->owners->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->arc->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->status->MarkSubmitted(s->fence, s);
        if (r == VK_SUCCESS) r = s->dummy->MarkSubmitted(s->fence, s);
        if (r != VK_SUCCESS) {
            s->pending = true;
            candidate->Quarantine();
            finish(ExprVkContextStatus::DeviceError, r);
            return {};
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        s->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            finish(ExprVkContextStatus::DeviceError, r);
            return {};
        }
        finish(ExprVkContextStatus::Ok, VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) {
        finish(ExprVkContextStatus::DeviceError, VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

} // namespace usdGen::vulkan
