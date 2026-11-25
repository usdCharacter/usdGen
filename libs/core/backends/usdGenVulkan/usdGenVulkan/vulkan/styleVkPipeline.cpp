// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "styleVkPipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace usdGen::vulkan {

char const* StyleVkSemanticName(StyleVkSemantic semantic) noexcept {
    switch (semantic) {
        case StyleVkSemantic::Ok: return "Ok";
        case StyleVkSemantic::BadOffsets: return "BadOffsets";
        case StyleVkSemantic::NonFinite: return "NonFinite";
        case StyleVkSemantic::BadValue: return "BadValue";
    }
    return "Invalid";
}

char const* StyleVkStatusName(StyleVkStatus status) noexcept {
    switch (status) {
        case StyleVkStatus::Ok: return "Ok";
        case StyleVkStatus::InvalidArgument: return "InvalidArgument";
        case StyleVkStatus::NonFiniteInput: return "NonFiniteInput";
        case StyleVkStatus::InvalidValue: return "InvalidValue";
        case StyleVkStatus::NotSupported: return "NotSupported";
        case StyleVkStatus::VulkanError: return "VulkanError";
    }
    return "Invalid";
}

StyleVkStatus StyleVkStatusForSemantic(StyleVkSemantic semantic) noexcept {
    switch (semantic) {
        case StyleVkSemantic::Ok: return StyleVkStatus::Ok;
        case StyleVkSemantic::BadOffsets: return StyleVkStatus::InvalidArgument;
        case StyleVkSemantic::NonFinite: return StyleVkStatus::NonFiniteInput;
        case StyleVkSemantic::BadValue: return StyleVkStatus::InvalidValue;
    }
    return StyleVkStatus::VulkanError;
}

namespace {
constexpr uint32_t kLocalSize = 256;
constexpr uint32_t kFlagHairT = 1u;
// All three styleVk ABIs place `phase` at push-constant word 2 (after
// curves and points); BeginImpl patches it per dispatch.
constexpr uint32_t kPhaseWord = 2u;

uint32_t Groups(uint32_t n) { return (n + kLocalSize - 1) / kLocalSize; }

// Push layouts must match styleVkWidthRamp.comp / styleVkLength.comp /
// styleVkGrow.comp exactly.
struct WidthRampPush {
    uint32_t curves, points, phase, flags;
    float rootLiteral;
    uint32_t rootDomain, rootHasData;
    float tipLiteral;
    uint32_t tipDomain, tipHasData;
};
struct LengthPush {
    uint32_t curves, points, phase;
    float scaleLiteral;
    uint32_t scaleDomain, scaleHasData;
};
struct GrowPush {
    uint32_t curves, points, phase, flags;
    float lengthLiteral;
    uint32_t lengthDomain, lengthHasData;
};
static_assert(sizeof(WidthRampPush) == 40, "widthRamp shader ABI");
static_assert(sizeof(LengthPush) == 24, "length shader ABI");
static_assert(sizeof(GrowPush) == 28, "grow shader ABI");

} // namespace

struct StyleVkPipeline::OpNative {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindingCount = 0;
    uint32_t pushBytes = 0;
};

struct StyleVkPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    OpNative widthRamp, length, grow;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        OpNative const* ops[3] = {&widthRamp, &length, &grow};
        for (auto op : ops) {
            if (op->pipeline) vkDestroyPipeline(d, op->pipeline, nullptr);
            if (op->layout) vkDestroyPipelineLayout(d, op->layout, nullptr);
            if (op->descriptors) vkDestroyDescriptorSetLayout(d, op->descriptors, nullptr);
            if (op->shader) vkDestroyShaderModule(d, op->shader, nullptr);
        }
    }
};

struct StyleVkPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    OpNative op;
    std::vector<std::shared_ptr<const ChargedBuffer>> owners;
    std::shared_ptr<ChargedBuffer> outValues;
    std::shared_ptr<ChargedBuffer> status;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    bool vectorOutput = false;
    uint32_t semantic = UINT32_MAX;
    uint32_t pointCount = 0;
    uint32_t curveCount = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

StyleVkPipeline::StyleVkPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
StyleVkPipeline::~StyleVkPipeline() = default;
std::shared_ptr<DeviceContext> const& StyleVkPipeline::context() const noexcept {
    return native_->context;
}

// ============================ Create ============================

std::shared_ptr<StyleVkPipeline> StyleVkPipeline::Create(
    std::shared_ptr<DeviceContext> context,
    std::vector<uint32_t> const& widthRampSpirv,
    std::vector<uint32_t> const& lengthSpirv,
    std::vector<uint32_t> const& growSpirv, VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto valid = [](std::vector<uint32_t> const& code) {
        return code.size() >= 5 && code.front() == 0x07230203u;
    };
    if (!context || !valid(widthRampSpirv) || !valid(lengthSpirv) || !valid(growSpirv))
        return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kLocalSize ||
        physical.limits.maxComputeWorkGroupSize[0] < kLocalSize)
        return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        struct Spec {
            OpNative* op;
            std::vector<uint32_t> const* code;
            uint32_t bindings;
            uint32_t pushBytes;
        };
        Spec specs[3] = {
            {&n->widthRamp, &widthRampSpirv, 6, sizeof(WidthRampPush)},
            {&n->length, &lengthSpirv, 5, sizeof(LengthPush)},
            {&n->grow, &growSpirv, 7, sizeof(GrowPush)},
        };
        for (auto& spec : specs) {
            spec.op->bindingCount = spec.bindings;
            spec.op->pushBytes = spec.pushBytes;
            VkShaderModuleCreateInfo sm{};
            sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sm.codeSize = spec.code->size() * sizeof(uint32_t);
            sm.pCode = spec.code->data();
            VkResult r = vkCreateShaderModule(d, &sm, nullptr, &spec.op->shader);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkDescriptorSetLayoutBinding bindings[7]{};
            for (uint32_t i = 0; i < spec.bindings; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = spec.bindings;
            ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &spec.op->descriptors);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, spec.pushBytes};
            VkPipelineLayoutCreateInfo pl{};
            pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &spec.op->descriptors;
            pl.pushConstantRangeCount = 1;
            pl.pPushConstantRanges = &push;
            r = vkCreatePipelineLayout(d, &pl, nullptr, &spec.op->layout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkComputePipelineCreateInfo cp{};
            cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout = spec.op->layout;
            cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cp.stage.module = spec.op->shader;
            cp.stage.pName = "main";
            r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr,
                                         &spec.op->pipeline);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        auto pipeline = std::shared_ptr<StyleVkPipeline>(new StyleVkPipeline(std::move(n)));
        finish(VK_SUCCESS);
        return pipeline;
    } catch (std::bad_alloc const&) {
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

// ============================ Candidate ============================

StyleVkPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
StyleVkPipeline::Candidate::~Candidate() {
    if (state_ && state_->pending) Quarantine();
}

void StyleVkPipeline::Candidate::Quarantine() noexcept {
    if (!state_ || !state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}

bool StyleVkPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}

StyleVkPipeline::Candidate::Output StyleVkPipeline::Candidate::output() const noexcept {
    if (!state_ || !state_->proved || state_->semantic != 0) return {};
    return Output{state_->outValues};
}

bool StyleVkPipeline::Candidate::vectorOutput() const noexcept {
    return state_ && state_->vectorOutput;
}

std::shared_ptr<DeviceContext> StyleVkPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

uint32_t StyleVkPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}

uint32_t StyleVkPipeline::Candidate::curveCount() const noexcept {
    return state_ ? state_->curveCount : 0;
}

VkResult StyleVkPipeline::Candidate::Poll(StyleVkSemantic* semantic) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        auto d = s.native->context->device();
        VkResult r = vkGetFenceStatus(d, s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) {
            Quarantine();
            return r;
        }
        void* data = nullptr;
        r = vkMapMemory(d, s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.semantic, data, 4);
        vkUnmapMemory(d, s.status->memory());
        s.pending = false;
        s.proved = true;
    }
    if (semantic) *semantic = static_cast<StyleVkSemantic>(s.semantic);
    return VK_SUCCESS;
}

// ============================ Begin ============================

namespace {
// Host shape contract mirrors CUDA styleOps.cu host validation (begin()
// plus validateField). Vulkan planes may carry a larger allocation than the
// op reads, so sizeBytes() >= required.
bool CheckPlane(std::shared_ptr<ChargedBuffer const> const& buffer,
                std::shared_ptr<DeviceContext> const& context, VkDeviceSize bytes) {
    return buffer && buffer->context() == context && !buffer->unproven() &&
        buffer->buffer() && (buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
        buffer->sizeBytes() >= bytes;
}

// CUDA validateField mirror: literal fields must be groom-scoped with no
// data; data fields must carry exactly the domain's element count.
StyleVkStatus ValidateScalarField(StyleVkPipeline::ScalarField const& field,
                                  std::shared_ptr<DeviceContext> const& context,
                                  uint32_t curves, uint32_t points) {
    using Domain = StyleVkDomain;
    if (field.domain != Domain::Groom && field.domain != Domain::Primitive &&
        field.domain != Domain::Point)
        return StyleVkStatus::InvalidArgument;
    if (!field.data) {
        if (field.count != 0 || field.domain != Domain::Groom)
            return StyleVkStatus::InvalidArgument;
        return std::isfinite(field.literal) ? StyleVkStatus::Ok
                                            : StyleVkStatus::NonFiniteInput;
    }
    uint32_t const expected = field.domain == Domain::Groom
        ? 1u : (field.domain == Domain::Primitive ? curves : points);
    if (field.count != expected)
        return StyleVkStatus::InvalidArgument;
    if (!CheckPlane(field.data, context, VkDeviceSize(expected) * 4u))
        return StyleVkStatus::InvalidArgument;
    return StyleVkStatus::Ok;
}

constexpr uint32_t kIntMax = uint32_t(std::numeric_limits<int>::max());
} // namespace

std::unique_ptr<StyleVkPipeline::Candidate> StyleVkPipeline::BeginImpl(
    OpNative const* op, std::shared_ptr<const ChargedBuffer> const* bindings,
    uint32_t bindingCount, void const* push, uint32_t pushBytes, uint32_t curves,
    uint32_t points, bool vectorOutput, VkResult* result, StyleVkStatus* status,
    BeforeSubmit beforeSubmit, std::shared_ptr<const ChargedBuffer> const* owners,
    uint32_t ownerCount) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (status) *status = StyleVkStatus::Ok;
    auto context = native_->context;
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->op = *op;
        s->owners.assign(owners, owners + ownerCount);
        s->pointCount = points;
        s->curveCount = curves;
        s->vectorOutput = vectorOutput;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        auto d = context->device();
        VkResult r;

        // Output plane: DEVICE_LOCAL scalar or packed-float3 plane (minimum
        // 4 bytes for the empty case, per the no-zero-size-buffer rule).
        VkDeviceSize const outBytes = points
            ? VkDeviceSize(points) * (vectorOutput ? 12u : 4u) : 4u;
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = outBytes;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            s->outValues = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                UsdGenExecutionResourceKind::Active, &r);
            if (!s->outValues) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(r);
                return {};
            }
        }
        // Empty geometry is a valid no-op (CUDA queues nothing and Finish
        // reports Ok); no submit, no status plane, no BeforeSubmit.
        if (curves == 0) {
            s->proved = true;
            s->semantic = 0;
            finish(VK_SUCCESS);
            return candidate;
        }
        // Status: 4-byte host-visible scratch, zeroed on host.
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            s->status = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &r);
            if (!s->status) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(r);
                return {};
            }
            void* data = nullptr;
            r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(r);
                return {};
            }
            std::memset(data, 0, 4);
            vkUnmapMemory(d, s->status->memory());
        }

        VkDescriptorPoolSize poolSize{
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount + 2};
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1;
        dp.poolSizeCount = 1;
        dp.pPoolSizes = &poolSize;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) {
            if (status) *status = StyleVkStatus::VulkanError;
            finish(r);
            return {};
        }
        VkDescriptorSet set = VK_NULL_HANDLE;
        {
            VkDescriptorSetAllocateInfo da{};
            da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            da.descriptorPool = s->descriptors;
            da.descriptorSetCount = 1;
            da.pSetLayouts = &s->op.descriptors;
            r = vkAllocateDescriptorSets(d, &da, &set);
            if (r != VK_SUCCESS) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(r);
                return {};
            }
        }
        // Absent optional planes (null input bindings) fall back to the
        // 4-byte status buffer, which the shader never reads when the
        // matching flag/count says the plane is absent (NoisePipeline
        // pattern). The caller passes input planes only; the candidate-owned
        // output and status planes occupy the trailing two bindings.
        if (bindingCount + 2 != s->op.bindingCount) {
            if (status) *status = StyleVkStatus::VulkanError;
            finish(VK_ERROR_INITIALIZATION_FAILED);
            return {};
        }
        VkDescriptorBufferInfo infos[7]{};
        VkWriteDescriptorSet writes[7]{};
        for (uint32_t i = 0; i < bindingCount; ++i) {
            if (bindings[i])
                infos[i] = {bindings[i]->buffer(), 0, bindings[i]->sizeBytes()};
            else
                infos[i] = {s->status->buffer(), 0, 4u};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        infos[bindingCount] = {s->outValues->buffer(), 0, outBytes};
        infos[bindingCount + 1] = {s->status->buffer(), 0, 4u};
        for (uint32_t i = bindingCount; i < bindingCount + 2; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, bindingCount + 2, writes, 0, nullptr);

        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) {
            if (status) *status = StyleVkStatus::VulkanError;
            finish(r);
            return {};
        }
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        {
            VkCommandBufferAllocateInfo ca{};
            ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ca.commandPool = s->commands;
            ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ca.commandBufferCount = 1;
            r = vkAllocateCommandBuffers(d, &ca, &cmd);
            if (r != VK_SUCCESS) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(r);
                return {};
            }
        }
        // One submit, two dispatches in CUDA stream order: ValidateOffsets
        // pre-pass (phase 0), then the op (phase 1), separated by a compute
        // barrier so the phase-1 gate observes validation before proceeding.
        uint8_t pushBytes_[40]{};
        std::memcpy(pushBytes_, push, pushBytes);
        r = [&]() {
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            VkResult record = vkBeginCommandBuffer(cmd, &begin);
            if (record != VK_SUCCESS) return record;
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT |
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr,
                0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s->op.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                s->op.layout, 0, 1, &set, 0, nullptr);
            auto dispatch = [&](uint32_t phase, uint32_t threads) {
                reinterpret_cast<uint32_t*>(pushBytes_)[kPhaseWord] = phase;
                vkCmdPushConstants(cmd, s->op.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                    0, pushBytes, pushBytes_);
                vkCmdDispatch(cmd, Groups(threads), 1, 1);
            };
            dispatch(0u, curves + 1u);
            VkMemoryBarrier middle{};
            middle.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            middle.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            middle.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &middle, 0, nullptr,
                0, nullptr);
            dispatch(1u, curves);
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT |
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            return vkEndCommandBuffer(cmd);
        }();
        if (r != VK_SUCCESS) {
            if (status) *status = StyleVkStatus::VulkanError;
            finish(r);
            return {};
        }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) {
            if (status) *status = StyleVkStatus::VulkanError;
            finish(r);
            return {};
        }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(VK_ERROR_UNKNOWN);
                return {};
            }
            if (!admitted) {
                if (status) *status = StyleVkStatus::VulkanError;
                finish(VK_ERROR_OUT_OF_DEVICE_MEMORY);
                return {};
            }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        s->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (status) *status = StyleVkStatus::VulkanError;
            finish(r);
            return {};
        }
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) {
        if (status) *status = StyleVkStatus::VulkanError;
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

std::unique_ptr<StyleVkPipeline::Candidate> StyleVkPipeline::BeginWidthRamp(
    WidthRampInfo info, VkResult* result, StyleVkStatus* status,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](StyleVkStatus code) {
        if (status) *status = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (status) *status = StyleVkStatus::Ok;
    auto context = native_->context;
    uint32_t const curves = info.curveCount, points = info.pointCount;
    // CUDA order: validateField runs before begin(), even for empty geometry.
    StyleVkStatus field = ValidateScalarField(info.root, context, curves, points);
    if (field != StyleVkStatus::Ok) return reject(field);
    field = ValidateScalarField(info.tip, context, curves, points);
    if (field != StyleVkStatus::Ok) return reject(field);
    // CUDA begin() numeric/shape contract.
    if (curves > kIntMax || points > kIntMax) return reject(StyleVkStatus::InvalidArgument);
    if (curves == 0 && points != 0) return reject(StyleVkStatus::InvalidArgument);
    if (curves > 0) {
        if (!CheckPlane(info.offsets, context, VkDeviceSize(curves + 1u) * 4u))
            return reject(StyleVkStatus::InvalidArgument);
        if (info.hairT && points > 0 &&
            !CheckPlane(info.hairT, context, VkDeviceSize(points) * 4u))
            return reject(StyleVkStatus::InvalidArgument);
    }
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (Groups(curves + 1u) > physical.limits.maxComputeWorkGroupCount[0] ||
        VkDeviceSize(points) * 4u > physical.limits.maxStorageBufferRange)
        return reject(StyleVkStatus::InvalidArgument);

    WidthRampPush push{};
    push.curves = curves;
    push.points = points;
    push.phase = 0;
    push.flags = (info.hairT && points > 0) ? kFlagHairT : 0u;
    push.rootLiteral = info.root.literal;
    push.rootDomain = info.root.domain;
    push.rootHasData = info.root.data ? 1u : 0u;
    push.tipLiteral = info.tip.literal;
    push.tipDomain = info.tip.domain;
    push.tipHasData = info.tip.data ? 1u : 0u;
    bool const useHairT = (push.flags & kFlagHairT) != 0u;
    // Binding order: offsets, rootData, tipData, hairT, then the
    // candidate-owned out/status planes appended by BeginImpl.
    std::shared_ptr<const ChargedBuffer> bindings[4] = {info.offsets,
        info.root.data, info.tip.data, useHairT ? info.hairT : nullptr};
    std::shared_ptr<const ChargedBuffer> owners[4] = {info.offsets,
        info.root.data, info.tip.data, info.hairT};
    return BeginImpl(&native_->widthRamp, bindings, 4, &push, sizeof(push),
        curves, points, false, result, status, std::move(beforeSubmit), owners,
        4);
}

std::unique_ptr<StyleVkPipeline::Candidate> StyleVkPipeline::BeginLength(
    LengthInfo info, VkResult* result, StyleVkStatus* status,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](StyleVkStatus code) {
        if (status) *status = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (status) *status = StyleVkStatus::Ok;
    auto context = native_->context;
    uint32_t const curves = info.curveCount, points = info.pointCount;
    // CUDA order: validateField runs before begin().
    StyleVkStatus field = ValidateScalarField(info.scale, context, curves, points);
    if (field != StyleVkStatus::Ok) return reject(field);
    if (curves > kIntMax || points > kIntMax) return reject(StyleVkStatus::InvalidArgument);
    if (curves == 0 && points != 0) return reject(StyleVkStatus::InvalidArgument);
    if (curves > 0) {
        if (!CheckPlane(info.offsets, context, VkDeviceSize(curves + 1u) * 4u))
            return reject(StyleVkStatus::InvalidArgument);
        // Length requires points (CUDA requirePoints); a zero pointCount
        // with curves is malformed offsets, diagnosed on the device.
        if (points > 0 &&
            !CheckPlane(info.points, context, VkDeviceSize(points) * 12u))
            return reject(StyleVkStatus::InvalidArgument);
    }
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (Groups(curves + 1u) > physical.limits.maxComputeWorkGroupCount[0] ||
        VkDeviceSize(points) * 12u > physical.limits.maxStorageBufferRange)
        return reject(StyleVkStatus::InvalidArgument);

    LengthPush push{};
    push.curves = curves;
    push.points = points;
    push.phase = 0;
    push.scaleLiteral = info.scale.literal;
    push.scaleDomain = info.scale.domain;
    push.scaleHasData = info.scale.data ? 1u : 0u;
    // Binding order: points, offsets, scaleData, then the
    // candidate-owned out/status planes appended by BeginImpl.
    std::shared_ptr<const ChargedBuffer> bindings[3] = {
        points > 0 ? info.points : nullptr, info.offsets, info.scale.data};
    std::shared_ptr<const ChargedBuffer> owners[3] = {info.points, info.offsets,
        info.scale.data};
    return BeginImpl(&native_->length, bindings, 3, &push, sizeof(push),
        curves, points, true, result, status, std::move(beforeSubmit), owners,
        3);
}

std::unique_ptr<StyleVkPipeline::Candidate> StyleVkPipeline::BeginGrow(
    GrowInfo info, VkResult* result, StyleVkStatus* status,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](StyleVkStatus code) {
        if (status) *status = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (status) *status = StyleVkStatus::Ok;
    auto context = native_->context;
    uint32_t const curves = info.curveCount, points = info.pointCount;
    // CUDA Grow order: roots/normals sizes, then validateField, then begin().
    if (curves > 0) {
        // CUDA Grow requires per-curve roots/normals sized by curveCount.
        if (!CheckPlane(info.roots, context, VkDeviceSize(curves) * 12u) ||
            !CheckPlane(info.normals, context, VkDeviceSize(curves) * 12u))
            return reject(StyleVkStatus::InvalidArgument);
    }
    StyleVkStatus field = ValidateScalarField(info.length, context, curves, points);
    if (field != StyleVkStatus::Ok) return reject(field);
    if (curves > kIntMax || points > kIntMax) return reject(StyleVkStatus::InvalidArgument);
    if (curves == 0 && points != 0) return reject(StyleVkStatus::InvalidArgument);
    if (curves > 0) {
        if (!CheckPlane(info.offsets, context, VkDeviceSize(curves + 1u) * 4u))
            return reject(StyleVkStatus::InvalidArgument);
        if (info.hairT && points > 0 &&
            !CheckPlane(info.hairT, context, VkDeviceSize(points) * 4u))
            return reject(StyleVkStatus::InvalidArgument);
    }
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (Groups(curves + 1u) > physical.limits.maxComputeWorkGroupCount[0] ||
        VkDeviceSize(points) * 12u > physical.limits.maxStorageBufferRange)
        return reject(StyleVkStatus::InvalidArgument);

    GrowPush push{};
    push.curves = curves;
    push.points = points;
    push.phase = 0;
    push.flags = (info.hairT && points > 0) ? kFlagHairT : 0u;
    push.lengthLiteral = info.length.literal;
    push.lengthDomain = info.length.domain;
    push.lengthHasData = info.length.data ? 1u : 0u;
    bool const useHairT = (push.flags & kFlagHairT) != 0u;
    // Binding order: offsets, roots, normals, lengthData, hairT, then the
    // candidate-owned out/status planes appended by BeginImpl.
    std::shared_ptr<const ChargedBuffer> bindings[5] = {info.offsets,
        info.roots, info.normals, info.length.data,
        useHairT ? info.hairT : nullptr};
    std::shared_ptr<const ChargedBuffer> owners[5] = {info.offsets, info.roots,
        info.normals, info.length.data, info.hairT};
    return BeginImpl(&native_->grow, bindings, 5, &push, sizeof(push), curves,
        points, true, result, status, std::move(beforeSubmit), owners, 5);
}

} // namespace usdGen::vulkan
