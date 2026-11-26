// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "mapVkImage.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace usdGen::vulkan {
namespace {

// Push-constant layout mirror of mapVkSample.comp Controls (56 bytes).
struct MapVkControls {
    uint32_t sampleCount;
    uint32_t width, height, channels;
    uint32_t orientation, filterMode, wrap, channel;
    float scale, offset;
    uint32_t clampOutput;
    float outputMin, outputMax, defaultValue;
};
static_assert(sizeof(MapVkControls) == 56, "Shader ABI");
static_assert(uint32_t(UsdGenImageRowOrientation::TopDown) == 0 &&
                  uint32_t(UsdGenImageRowOrientation::BottomUp) == 1,
              "orientation encoding");
static_assert(uint32_t(UsdGenImageFilter::Nearest) == 0 &&
                  uint32_t(UsdGenImageFilter::Bilinear) == 1,
              "filter encoding");
static_assert(uint32_t(UsdGenImageWrap::Clamp) == 0 &&
                  uint32_t(UsdGenImageWrap::Repeat) == 1 &&
                  uint32_t(UsdGenImageWrap::Mirror) == 2 &&
                  uint32_t(UsdGenImageWrap::Black) == 3,
              "wrap encoding");
static_assert(uint32_t(UsdGenImageChannel::R) == 0 &&
                  uint32_t(UsdGenImageChannel::G) == 1 &&
                  uint32_t(UsdGenImageChannel::B) == 2 &&
                  uint32_t(UsdGenImageChannel::A) == 3 &&
                  uint32_t(UsdGenImageChannel::Luminance) == 4,
              "channel encoding");

// Range proof for the shader's 32-bit mirror fold. Every wrap mode maps a
// finite UV into [0,1] (or returns early for Black/outside/NaN), and IEEE
// round-to-nearest cannot push the wrapped coordinate outside [0,1]
// (a rounded result past one end would be farther from the exact value than
// the endpoint itself, and the same argument bounds the scaled position by
// [0, extent-1]). Hence the nearest index and x0/x1/y0/y1 always land in
// [0, extent], where the shader's fold equals the CUDA int64 period
// reduction exactly. Non-finite UVs return the default before indexing.
bool MulSize(size_t a, size_t b, size_t* out) {
    if (b && a > std::numeric_limits<size_t>::max() / b) return false;
    *out = a * b;
    return true;
}

} // namespace

MapVkImage::MapVkImage(std::shared_ptr<DeviceContext> context,
    std::shared_ptr<const UsdGenImagePayload> payload,
    std::shared_ptr<ChargedBuffer> texels)
    : context_(std::move(context)), payload_(std::move(payload)),
      texels_(std::move(texels)) {}

std::shared_ptr<MapVkImage> MapVkImage::Create(
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const UsdGenImagePayload> const& payload, VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || !payload || !payload->IsValid()) return {};
    size_t texels = 0, bytes = 0;
    if (!MulSize(payload->Width(), payload->Height(), &texels) ||
        !MulSize(texels, payload->Channels(), &texels) ||
        !MulSize(texels, sizeof(float), &bytes) || !bytes ||
        texels != payload->TexelCount())
        return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    if (VkDeviceSize(bytes) > properties.limits.maxStorageBufferRange) return {};
    try {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = VkDeviceSize(bytes);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult status = VK_SUCCESS;
        auto buffer = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active, &status);
        if (!buffer) { finish(status); return {}; }
        void* mapped = nullptr;
        status = vkMapMemory(context->device(), buffer->memory(), 0, bi.size, 0, &mapped);
        if (status != VK_SUCCESS) { finish(status); return {}; }
        std::memcpy(mapped, payload->Data(), bytes);
        vkUnmapMemory(context->device(), buffer->memory());
        auto image = std::shared_ptr<MapVkImage>(
            new MapVkImage(context, payload, std::move(buffer)));
        finish(VK_SUCCESS);
        return image;
    } catch (std::bad_alloc const&) {
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

MapVkImageView MapVkImage::View() const noexcept {
    return {texels_, payload_};
}

uint32_t MapVkImage::Width() const noexcept { return payload_->Width(); }
uint32_t MapVkImage::Height() const noexcept { return payload_->Height(); }
uint32_t MapVkImage::Channels() const noexcept { return payload_->Channels(); }

struct MapVkSamplePipeline::Native {
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

struct MapVkSamplePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    MapVkImageView image;
    std::shared_ptr<const ChargedBuffer> uv;
    std::shared_ptr<ChargedBuffer> output, status;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t count = 0;
    uint32_t semantic = UINT32_MAX;
    // Allocated before any submission; releasing this holder intentionally
    // retains the entire native graph and every charged allocation on loss.
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

MapVkSamplePipeline::MapVkSamplePipeline(std::shared_ptr<Native> n)
    : native_(std::move(n)) {}
MapVkSamplePipeline::~MapVkSamplePipeline() = default;
std::shared_ptr<DeviceContext> const& MapVkSamplePipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<MapVkSamplePipeline> MapVkSamplePipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256 ||
        physical.limits.maxPushConstantsSize < sizeof(MapVkControls))
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
        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i != 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 4;
        ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MapVkControls)};
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
        auto pipeline = std::shared_ptr<MapVkSamplePipeline>(
            new MapVkSamplePipeline(std::move(n)));
        finish(VK_SUCCESS);
        return pipeline;
    } catch (std::bad_alloc const&) {
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

MapVkSamplePipeline::Candidate::Candidate(std::shared_ptr<State> s)
    : state_(std::move(s)) {}
MapVkSamplePipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void MapVkSamplePipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> MapVkSamplePipeline::Candidate::output() const noexcept {
    return state_->proved && state_->semantic == 0 ? state_->output : nullptr;
}
bool MapVkSamplePipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> MapVkSamplePipeline::Candidate::uvOwner() const noexcept {
    return state_ ? state_->uv : nullptr;
}
MapVkImageView MapVkSamplePipeline::Candidate::imageOwner() const noexcept {
    return state_ ? state_->image : MapVkImageView{};
}
uint32_t MapVkSamplePipeline::Candidate::count() const noexcept {
    return state_ ? state_->count : 0;
}
std::shared_ptr<DeviceContext> MapVkSamplePipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

std::unique_ptr<MapVkSamplePipeline::Candidate> MapVkSamplePipeline::Begin(
    MapVkImageView image, std::shared_ptr<const ChargedBuffer> uv, uint32_t count,
    UsdGenImageSampleOptions const& options, VkResult* result,
    BeforeSubmit beforeSubmit) {
    return BeginInternal(std::move(image), std::move(uv), count, options, result,
        std::move(beforeSubmit));
}

VkResult MapVkSamplePipeline::Candidate::Poll(uint32_t* semanticStatus) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) return r;
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

std::unique_ptr<MapVkSamplePipeline::Candidate> MapVkSamplePipeline::BeginInternal(
    MapVkImageView image, std::shared_ptr<const ChargedBuffer> uv, uint32_t count,
    UsdGenImageSampleOptions const& options, VkResult* result,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;
    if (!image.texels || !image.payload || !image.payload->IsValid() ||
        image.texels->context() != context || image.texels->unproven() ||
        !image.texels->buffer() ||
        !(image.texels->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))
        return {};
    // Same option contract as CudaImage::Sample (luminance needs 3 channels,
    // finite scale/offset/limits, ordered clamp range).
    if (!ValidateUsdGenImageSampleOptions(*image.payload, options, nullptr))
        return {};
    size_t texelCount = 0, texelBytes = 0, uvBytes = 0, outBytes = 0;
    if (!MulSize(image.payload->Width(), image.payload->Height(), &texelCount) ||
        !MulSize(texelCount, image.payload->Channels(), &texelCount) ||
        !MulSize(texelCount, sizeof(float), &texelBytes) || !texelBytes ||
        texelCount != image.payload->TexelCount() ||
        VkDeviceSize(texelBytes) > image.texels->sizeBytes())
        return {};
    if (!MulSize(count, 2 * sizeof(float), &uvBytes) ||
        !MulSize(count, sizeof(float), &outBytes))
        return {};
    if ((count && !uv) || (uv && (uv->context() != context || uv->unproven() ||
                                !uv->buffer() ||
                                !(uv->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
                                uv->sizeBytes() < VkDeviceSize(uvBytes))))
        return {};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    if (VkDeviceSize(texelBytes) > properties.limits.maxStorageBufferRange ||
        VkDeviceSize(uvBytes) > properties.limits.maxStorageBufferRange ||
        VkDeviceSize(outBytes) > properties.limits.maxStorageBufferRange)
        return {};
    // Grid-stride cap mirroring the CUDA Blocks() 65,535 bound: dispatch at
    // most maxComputeWorkGroupCount[0] groups and let invocations loop.
    uint64_t wantGroups = (uint64_t(count) + 255) / 256;
    uint64_t const groupLimit = properties.limits.maxComputeWorkGroupCount[0];
    if (!groupLimit) return {};
    uint32_t const groups =
        wantGroups > groupLimit ? uint32_t(groupLimit) : uint32_t(wantGroups);
    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->image = std::move(image);
        s->uv = std::move(uv);
        s->count = count;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        if (!count) {
            s->proved = true;
            s->semantic = 0;
            finish(VK_SUCCESS);
            return candidate;
        }
        MapVkControls controls{};
        controls.sampleCount = count;
        controls.width = s->image.payload->Width();
        controls.height = s->image.payload->Height();
        controls.channels = s->image.payload->Channels();
        controls.orientation = uint32_t(s->image.payload->Orientation());
        controls.filterMode = uint32_t(options.filter);
        controls.wrap = uint32_t(options.wrap);
        controls.channel = uint32_t(options.channel);
        controls.scale = options.scale;
        controls.offset = options.offset;
        controls.clampOutput = options.clampOutput ? 1u : 0u;
        controls.outputMin = options.outputMin;
        controls.outputMax = options.outputMax;
        controls.defaultValue = options.defaultValue;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = VkDeviceSize(outBytes);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult r;
        s->output = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
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
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4u};
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
        VkDescriptorBufferInfo infos[4] = {
            {s->image.texels->buffer(), 0, VkDeviceSize(texelBytes)},
            {s->uv->buffer(), 0, VkDeviceSize(uvBytes)},
            {s->output->buffer(), 0, VkDeviceSize(outBytes)},
            {s->status->buffer(), 0, 4}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i != 4u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 4, writes, 0, nullptr);
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
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0,
            nullptr);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(controls), &controls);
        vkCmdDispatch(command, groups, 1, 1);
        VkMemoryBarrier after{};
        after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                              VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            1, &after, 0, nullptr, 0, nullptr);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        // Admission is checked after every reversible allocation and command
        // recording, immediately before pending state or Vulkan submission.
        // Rejecting here lets local State destruction return every permit.
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                finish(VK_ERROR_UNKNOWN);
                return {};
            }
            if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        s->pending = true; // Arm retention before native submission.
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) {
        finish(VK_ERROR_OUT_OF_HOST_MEMORY);
        return {};
    }
}

} // namespace usdGen::vulkan
