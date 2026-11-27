// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/resampleVkPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace usdGen::vulkan {
namespace {
struct Push {
    uint32_t curves = 0, srcPoints = 0, target = 0, phase = 0;
    uint32_t stride = 0, flags = 0, srcBytes = 0, dstBytes = 0;
};
static_assert(sizeof(Push) == 32, "ResampleVk shader ABI");
void Barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{}; b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}
bool IsFloatFamily(UsdGenDeviceValueType type) {
    return type == UsdGenDeviceValueType::Float32 || type == UsdGenDeviceValueType::Float32x2 ||
        type == UsdGenDeviceValueType::Float32x3 || type == UsdGenDeviceValueType::Float32x4;
}
} // namespace
struct ResampleVkPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits{};
    ~Native() {
        auto d = context->device();
        if (pipeline) vkDestroyPipeline(d, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(d, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(d, shader, nullptr);
    }
};
struct ResampleVkPipeline::Candidate::State : std::enable_shared_from_this<State> {
    std::shared_ptr<Native> native;
    std::shared_ptr<const VulkanSourceGeneration> base;
    std::vector<std::shared_ptr<ChargedBuffer>> owned;
    std::shared_ptr<ChargedBuffer> dummy, status;
    std::vector<OutputPlane> outputs;
    std::vector<UsdGenChunkDesc> chunks;
    std::vector<UsdGenDeviceTileMetadata> tiles;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    uint64_t value = 0;  // the submit's queue-timeline value
    Push push;
    uint32_t curves = 0, points = 0, target = 0, semantic = UINT32_MAX;
    bool pending = false, lost = false, proved = false;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
    void Retain() noexcept {
        if (!pending || lost) return;
        lost = true; *quarantine = shared_from_this(); (void)quarantine.release();
    }
    std::shared_ptr<ChargedBuffer> Buffer(uint64_t bytes, bool host, VkResult& r,
        UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Scratch) {
        bytes = std::max<uint64_t>(4, bytes);
        if (bytes > UINT32_MAX || bytes > native->limits.maxStorageBufferRange) {
            r = VK_ERROR_FEATURE_NOT_PRESENT; return {};
        }
        VkBufferCreateInfo b{}; b.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; b.size = bytes;
        b.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        b.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto out = ChargedBuffer::Create(native->context, b, host ?
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT :
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, kind, &r);
        if (out) owned.push_back(out);
        return out;
    }
    bool Write(std::shared_ptr<ChargedBuffer> const& b, void const* data, size_t bytes, VkResult& r) {
        void* target = nullptr;
        r = vkMapMemory(native->context->device(), b->memory(), 0, b->sizeBytes(), 0, &target);
        if (r != VK_SUCCESS) return false;
        std::memset(target, 0, size_t(b->sizeBytes()));
        if (bytes) std::memcpy(target, data, bytes);
        vkUnmapMemory(native->context->device(), b->memory()); return true;
    }
    bool Setup(uint32_t sets, VkResult& r) {
        auto d = native->context->device();
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, sets * 4u};
        VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = sets; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &descriptors); if (r != VK_SUCCESS) return false;
        VkCommandPoolCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex = native->context->computeQueueFamily();
        r = vkCreateCommandPool(d, &cp, nullptr, &commands); if (r != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        r = vkAllocateCommandBuffers(d, &ca, &command); if (r != VK_SUCCESS) return false;
        if (!native->context->computeTimeline()) { r = VK_ERROR_FEATURE_NOT_PRESENT; return false; }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin); if (r != VK_SUCCESS) return false;
        Barrier(command); vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, native->pipeline);
        return true;
    }
    using Bindings = std::array<std::shared_ptr<const ChargedBuffer>, 4>;
    bool Dispatch(Bindings const& bindings, Push const& controls, uint64_t work, VkResult& r) {
        uint64_t groups = (std::max<uint64_t>(work, 1) + 63u) / 64u;
        if (groups > native->limits.maxComputeWorkGroupCount[0]) { r = VK_ERROR_FEATURE_NOT_PRESENT; return false; }
        auto d = native->context->device();
        VkDescriptorSetAllocateInfo a{}; a.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        a.descriptorPool = descriptors; a.descriptorSetCount = 1; a.pSetLayouts = &native->descriptors;
        VkDescriptorSet set = VK_NULL_HANDLE; r = vkAllocateDescriptorSets(d, &a, &set); if (r != VK_SUCCESS) return false;
        VkDescriptorBufferInfo infos[4]{}; VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            infos[i] = {bindings[i]->buffer(), 0, bindings[i]->sizeBytes()};
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 4, writes, 0, nullptr);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, native->layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, native->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &controls);
        vkCmdDispatch(command, uint32_t(groups), 1, 1); Barrier(command); return true;
    }
    bool Submit(BeforeSubmit const& before, VkResult& r) {
        Barrier(command);
        r = vkEndCommandBuffer(command); if (r != VK_SUCCESS) return false;
        if (before) {
            bool admitted = false;
            try { admitted = before(); } catch (...) { r = VK_ERROR_UNKNOWN; return false; }
            if (!admitted) { r = VK_ERROR_OUT_OF_DEVICE_MEMORY; return false; }
        }
        pending = true; value = native->context->computeTimeline()->Submit(&command, 1u, &r);
        if (!value) { Retain(); return false; } return true;
    }
    VkResult Prove() {
        if (lost) return VK_ERROR_DEVICE_LOST;
        VkResult r = native->context->computeTimeline()->Check(value);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Retain(); return r; }
        pending = false; void* data = nullptr;
        r = vkMapMemory(native->context->device(), status->memory(), 0, status->sizeBytes(), 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&semantic, data, sizeof(uint32_t));
        vkUnmapMemory(native->context->device(), status->memory()); return VK_SUCCESS;
    }
};

ResampleVkPipeline::ResampleVkPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
ResampleVkPipeline::~ResampleVkPipeline() = default;
std::shared_ptr<DeviceContext> const& ResampleVkPipeline::context() const noexcept { return native_->context; }
std::shared_ptr<ResampleVkPipeline> ResampleVkPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code, VkResult* result) {
    VkResult r = VK_ERROR_INITIALIZATION_FAILED; if (result) *result = r;
    if (!context || code.size() < 5 || code[0] != 0x07230203u) return {};
    // The shader uses double intermediates for CUDA bit-parity; Float64
    // SPIR-V must reject contexts whose device lacks the enabled feature.
    if (!context->shaderFloat64Enabled()) return {};
    try {
        auto n = std::make_shared<Native>(); n->context = std::move(context);
        VkPhysicalDeviceProperties properties{}; vkGetPhysicalDeviceProperties(n->context->physicalDevice(), &properties);
        n->limits = properties.limits;
        if (n->limits.maxComputeWorkGroupInvocations < 64 || n->limits.maxComputeWorkGroupSize[0] < 64 ||
            n->limits.maxPerStageDescriptorStorageBuffers < 4 || n->limits.maxDescriptorSetStorageBuffers < 4) return {};
        auto d = n->context->device(); VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * 4; sm.pCode = code.data(); r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i < 4; ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; ds.bindingCount = 4; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors); if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; pl.setLayoutCount = 1; pl.pSetLayouts = &n->descriptors;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(d, &pl, nullptr, &n->layout); if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO; cp.layout = n->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = n->shader; cp.stage.pName = "main";
        r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->pipeline); if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        auto out = std::shared_ptr<ResampleVkPipeline>(new ResampleVkPipeline(std::move(n)));
        if (result) *result = VK_SUCCESS;
        return out;
    } catch (std::bad_alloc const&) { if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY; return {}; }
}
ResampleVkPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
ResampleVkPipeline::Candidate::~Candidate() { if (state_ && state_->pending) Quarantine(); }
void ResampleVkPipeline::Candidate::Quarantine() noexcept { if (state_) state_->Retain(); }
std::shared_ptr<const VulkanSourceGeneration> ResampleVkPipeline::Candidate::inputOwner() const noexcept { return state_->base; }
std::shared_ptr<DeviceContext> ResampleVkPipeline::Candidate::context() const noexcept { return state_->native->context; }
uint32_t ResampleVkPipeline::Candidate::curveCount() const noexcept { return state_->curves; }
uint32_t ResampleVkPipeline::Candidate::pointCount() const noexcept { return state_->points; }
uint32_t ResampleVkPipeline::Candidate::target() const noexcept { return state_->target; }
bool ResampleVkPipeline::Candidate::succeeded() const noexcept { return state_->proved && !state_->lost && !state_->semantic; }
namespace { std::vector<ResampleVkPipeline::Candidate::OutputPlane> const kNoResampleOutputs; }
std::vector<ResampleVkPipeline::Candidate::OutputPlane> const& ResampleVkPipeline::Candidate::outputs() const noexcept {
    return succeeded() ? state_->outputs : kNoResampleOutputs;
}
std::vector<UsdGenChunkDesc> const& ResampleVkPipeline::Candidate::chunks() const noexcept { return state_->chunks; }
std::vector<UsdGenDeviceTileMetadata> const& ResampleVkPipeline::Candidate::tiles() const noexcept { return state_->tiles; }

std::unique_ptr<ResampleVkPipeline::Candidate> ResampleVkPipeline::Begin(
    std::shared_ptr<const VulkanSourceGeneration> base, uint32_t target, VkResult* result, BeforeSubmit before) {
    VkResult r = VK_ERROR_INITIALIZATION_FAILED; if (result) *result = r;
    // CUDA parity: target 0 clones and 1 is invalid; this pipeline serves the
    // uniform >= 2 case (a resampleTo=0 plan compiles to no stage at all).
    if (!base || base->context() != context() || target < 2) return {};
    uint64_t const curves64 = base->curveCount(), points64 = base->pointCount();
    if (curves64 > UINT32_MAX || points64 > UINT32_MAX || (curves64 == 0) != (points64 == 0)) return {};
    uint64_t const outPoints64 = curves64 * uint64_t(target);
    // Mirror the CUDA Apply bounds: uint32 offsets and INT32 total.
    if (outPoints64 > UINT32_MAX || outPoints64 > uint64_t(std::numeric_limits<int32_t>::max())) return {};
    auto context = this->context();
    try {
        auto s = std::make_shared<Candidate::State>(); s->native = native_; s->base = base;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        s->curves = base->curveCount(); s->points = uint32_t(outPoints64); s->target = target;
        s->push.curves = s->curves; s->push.srcPoints = base->pointCount(); s->push.target = target;
        auto offsetsOwner = base->PlaneOwner(UsdGenCurveChannels().curveOffsets);
        uint64_t const offsetsBytes = (curves64 + 1) * 4;
        if (!offsetsOwner || offsetsOwner->context() != context || offsetsOwner->unproven() ||
            offsetsOwner->sizeBytes() < offsetsBytes || !(offsetsOwner->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
            return {};
        auto valid = [&](std::shared_ptr<const ChargedBuffer> const& b, uint64_t bytes) {
            return !bytes ? !b || (b->context() == context && !b->unproven()) : b && b->context() == context &&
                !b->unproven() && b->sizeBytes() >= bytes && b->sizeBytes() <= native_->limits.maxStorageBufferRange &&
                (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && (b->usage() & VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        };
        // Host metadata is the scheduling contract: every plane's typed bytes
        // must match its stored extent and its domain cardinality.
        auto consistent = [&](VulkanSourceGeneration::PlaneView const& plane, uint64_t elements) {
            size_t span = 0;
            return UsdGenDeviceChannelStorageBytes(plane.metadata, &span, nullptr) &&
                span == plane.bytes && plane.metadata.elementCount == elements;
        };
        s->chunks = base->chunks(); s->tiles = base->geometry().tiles;
        for (auto const& c : s->chunks) {
            if (c.firstCurve > s->curves || c.curveCount > s->curves - c.firstCurve ||
                c.liveCount > c.curveCount || c.firstCv > base->pointCount()) return {};
        }
        for (auto const& t : s->tiles) {
            if (t.firstElement > s->curves || t.elementCount > s->curves - t.firstElement ||
                t.firstPoint > base->pointCount() || t.pointCount > base->pointCount() - t.firstPoint) return {};
        }
        uint64_t const total = uint64_t(base->planes().size()) + base->sourceFrames().size();
        if (total > (UINT32_MAX / 4) - 1) return {};
        s->outputs.reserve(size_t(total));
        s->dummy = s->Buffer(4, true, r); if (!s->dummy || !s->Write(s->dummy, nullptr, 0, r)) goto failed;
        s->status = s->Buffer(4, true, r);
        if (!s->status || !s->Write(s->status, nullptr, 0, r)) goto failed;
        if (!s->Setup(uint32_t(total + 1), r)) goto failed;
        {
            Candidate::State::Bindings validate{offsetsOwner, s->dummy, s->dummy, s->status};
            auto p = s->push; p.phase = 0;
            if (!s->Dispatch(validate, p, s->curves, r)) goto failed;
            auto copy = [&](std::shared_ptr<const ChargedBuffer> const& from,
                            std::shared_ptr<ChargedBuffer> const& to, uint64_t bytes) {
                VkBufferCopy region{0, 0, bytes};
                vkCmdCopyBuffer(s->command, from->buffer(), to->buffer(), 1, &region);
                Barrier(s->command);
            };
            auto gather = [&](VulkanSourceGeneration::PlaneView const& plane, bool privateFrame) -> bool {
                Candidate::OutputPlane output{plane.metadata, {}, privateFrame, 0};
                auto owner = privateFrame ? base->SourceFrameOwner(plane.metadata.name) : base->PlaneOwner(plane.metadata.name);
                if (plane.bytes && (!owner || !valid(owner, plane.bytes))) return false;
                if (!plane.bytes && owner && !valid(owner, 0)) return false;
                using D = UsdGenDeviceDomain; using S = UsdGenDeviceChannelSemantic;
                if (plane.metadata.domain == D::Groom) {
                    if (!consistent(plane, plane.metadata.elementCount)) return false;
                    output.owner = owner; output.bytes = plane.bytes; s->outputs.push_back(std::move(output)); return true;
                }
                bool const offsets = plane.metadata.semantic == S::ElementOffsets;
                if (offsets) {
                    if (plane.metadata.domain != D::Topology || !consistent(plane, curves64 + 1)) return false;
                    output.metadata.elementCount = curves64 + 1; output.bytes = offsetsBytes;
                    auto staging = s->Buffer(offsetsBytes, true, r);
                    auto out = s->Buffer(offsetsBytes, false, r, UsdGenExecutionResourceKind::Active);
                    if (!staging || !out) return false;
                    std::vector<uint32_t> uniform(s->curves + 1);
                    for (uint32_t c = 0; c <= s->curves; ++c) uniform[c] = c * target;
                    if (!s->Write(staging, uniform.data(), uniform.size() * 4, r)) return false;
                    copy(staging, out, offsetsBytes);
                    output.owner = out; s->outputs.push_back(std::move(output)); return true;
                }
                if (plane.metadata.domain == D::Point) {
                    if (!consistent(plane, points64)) return false;
                    uint64_t const arity = plane.metadata.arity ? plane.metadata.arity : 1;
                    uint32_t const stride = plane.metadata.strideBytes ? plane.metadata.strideBytes : uint32_t(4 * arity);
                    // CUDA named-topology parity: point planes must be tightly
                    // packed float/int lanes; anything else fails closed.
                    if (stride != 4 * arity || arity > UINT32_MAX / 4) return false;
                    uint64_t const outBytes = outPoints64 * stride;
                    if (outBytes > UINT32_MAX || outBytes > native_->limits.maxStorageBufferRange) return false;
                    output.metadata.elementCount = outPoints64; output.bytes = outBytes;
                    if (!outBytes) { s->outputs.push_back(std::move(output)); return true; }
                    uint32_t phase = UINT32_MAX, flags = UINT32_MAX;
                    if (IsFloatFamily(plane.metadata.type)) {
                        phase = 1;
                        if (plane.metadata.semantic == S::Widths) flags = 1;
                        else if (plane.metadata.semantic == S::HairT) flags = 2;
                        else if (plane.metadata.semantic == S::Points || plane.metadata.semantic == S::RestPoints) flags = 0;
                        else if (plane.metadata.semantic == S::Generic) flags = 5;
                        else return false;
                    } else if (plane.metadata.type == UsdGenDeviceValueType::Int32) {
                        if (plane.metadata.semantic != S::Generic) return false;
                        phase = 2; flags = 6;
                    } else return false;
                    auto out = s->Buffer(outBytes, false, r, UsdGenExecutionResourceKind::Active);
                    if (!out) return false;
                    Candidate::State::Bindings b{offsetsOwner, owner, out, s->status};
                    auto pc = s->push; pc.phase = phase; pc.stride = stride; pc.flags = flags;
                    pc.srcBytes = uint32_t(plane.bytes); pc.dstBytes = uint32_t(outBytes);
                    if (!s->Dispatch(b, pc, outPoints64, r)) return false;
                    output.owner = out; s->outputs.push_back(std::move(output)); return true;
                }
                if (plane.metadata.domain == D::Primitive) {
                    if (!consistent(plane, curves64)) return false;
                    output.metadata.elementCount = curves64; output.bytes = plane.bytes;
                    if (!plane.bytes) { s->outputs.push_back(std::move(output)); return true; }
                    if (plane.bytes > UINT32_MAX) return false;
                    auto out = s->Buffer(plane.bytes, false, r, UsdGenExecutionResourceKind::Active);
                    if (!out) return false;
                    bool shaderCopy = false; uint32_t flags = 6;
                    if (plane.metadata.semantic == S::RootPrim) {
                        if (plane.metadata.type != UsdGenDeviceValueType::Int32 ||
                            plane.metadata.arity != 1 || plane.metadata.strideBytes != 4) return false;
                        shaderCopy = true; flags = 3;
                    } else if (plane.metadata.semantic == S::RootUV) {
                        if (plane.metadata.type != UsdGenDeviceValueType::Float32x2 ||
                            plane.metadata.arity != 2 || plane.metadata.strideBytes != 8) return false;
                        shaderCopy = true; flags = 4;
                    }
                    if (shaderCopy) {
                        Candidate::State::Bindings b{offsetsOwner, owner, out, s->status};
                        auto pc = s->push; pc.phase = 3; pc.flags = flags;
                        pc.srcBytes = uint32_t(plane.bytes); pc.dstBytes = uint32_t(plane.bytes);
                        if (!s->Dispatch(b, pc, plane.bytes / 4, r)) return false;
                    } else copy(owner, out, plane.bytes);
                    output.owner = out; s->outputs.push_back(std::move(output)); return true;
                }
                return false;
            };
            for (auto const& plane : base->planes()) if (!gather(plane, false)) goto failed;
            for (auto const& plane : base->sourceFrames()) if (!gather(plane, true)) goto failed;
            // Uniform output: chunk/tile curve ranges are unchanged; CV ranges
            // scale by the target and bounds are no longer valid.
            for (auto& c : s->chunks) {
                c.firstCv = c.firstCurve * target; c.cvCount = target;
            }
            for (auto& t : s->tiles) {
                t.firstPoint = uint64_t(t.firstElement) * target;
                t.pointCount = uint64_t(t.elementCount) * target;
                t.boundsValid = false;
            }
            if (!s->Submit(before, r)) goto failed;
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    failed:
        if (result) *result = r == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : r;
        return {};
    } catch (std::bad_alloc const&) { if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY; return {}; }
}
VkResult ResampleVkPipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_; if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        auto r = s.Prove(); if (r != VK_SUCCESS) return r;
        s.proved = true;
    }
    if (semanticStatus) *semanticStatus = s.semantic;
    return VK_SUCCESS;
}
} // namespace usdGen::vulkan
