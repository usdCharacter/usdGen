// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "picktileVk.h"

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace usdGen::vulkan {
namespace {

constexpr uint32_t kWorkgroup = 256;

uint32_t GroupsFor(uint64_t work) {
    uint64_t groups = (work + kWorkgroup - 1u) / kWorkgroup;
    if (groups == 0) groups = 1;
    if (groups > uint64_t(UINT32_MAX)) groups = UINT32_MAX;
    return uint32_t(groups);
}

bool ValidCharged(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const ChargedBuffer> const& buffer, uint64_t exactBytes) {
    if (!buffer || buffer->context() != context || buffer->unproven() ||
        !buffer->buffer() ||
        (buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0)
        return false;
    if (exactBytes == 0) return true; // placeholder accepted; the shader reads none
    return buffer->sizeBytes() == VkDeviceSize(exactBytes);
}

std::shared_ptr<ChargedBuffer> MakeHostBuffer(std::shared_ptr<DeviceContext> const& context,
    VkDeviceSize bytes, VkResult* result) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4; // Vulkan cannot materialize zero-size buffers
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult local = VK_SUCCESS;
    auto buffer = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch, &local);
    if (result) *result = buffer ? VK_SUCCESS : local;
    return buffer;
}

bool FillHost(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<ChargedBuffer> const& buffer, void const* data, size_t bytes) {
    if (!buffer || VkDeviceSize(bytes) > buffer->sizeBytes()) return false;
    if (bytes == 0) return true;
    void* mapped = nullptr;
    if (vkMapMemory(context->device(), buffer->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS)
        return false;
    std::memcpy(mapped, data, bytes);
    vkUnmapMemory(context->device(), buffer->memory());
    return true;
}

bool ReadHost(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const ChargedBuffer> const& buffer, void* data, size_t bytes) {
    if (!buffer || VkDeviceSize(bytes) > buffer->sizeBytes()) return false;
    if (bytes == 0) return true;
    void* mapped = nullptr;
    if (vkMapMemory(context->device(), buffer->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS)
        return false;
    std::memcpy(data, mapped, bytes);
    vkUnmapMemory(context->device(), buffer->memory());
    return true;
}

struct PipeObjects {
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

void DestroyPipe(std::shared_ptr<DeviceContext> const& context, PipeObjects& pipe) {
    if (!context) return;
    VkDevice device = context->device();
    if (pipe.pipeline) vkDestroyPipeline(device, pipe.pipeline, nullptr);
    if (pipe.layout) vkDestroyPipelineLayout(device, pipe.layout, nullptr);
    if (pipe.descriptors) vkDestroyDescriptorSetLayout(device, pipe.descriptors, nullptr);
    if (pipe.shader) vkDestroyShaderModule(device, pipe.shader, nullptr);
    pipe = PipeObjects{};
}

bool CreatePipe(std::shared_ptr<DeviceContext> const& context,
    std::vector<uint32_t> const& code, uint32_t bindings, uint32_t pushSize,
    PipeObjects* pipe, VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context || !pipe || bindings == 0 || bindings > 16 ||
        code.size() < 5 || code.front() != 0x07230203u)
        return false;
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kWorkgroup ||
        physical.limits.maxComputeWorkGroupSize[0] < kWorkgroup ||
        physical.limits.maxPerStageDescriptorStorageBuffers < bindings ||
        physical.limits.maxPushConstantsSize < pushSize)
        return false;
    VkDevice device = context->device();
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = code.size() * sizeof(uint32_t);
    sm.pCode = code.data();
    VkResult r = vkCreateShaderModule(device, &sm, nullptr, &pipe->shader);
    if (r != VK_SUCCESS) { if (result) *result = r; return false; }
    VkDescriptorSetLayoutBinding layoutBindings[16]{};
    for (uint32_t i = 0; i < bindings; ++i) {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ds.bindingCount = bindings;
    ds.pBindings = layoutBindings;
    r = vkCreateDescriptorSetLayout(device, &ds, nullptr, &pipe->descriptors);
    if (r != VK_SUCCESS) { DestroyPipe(context, *pipe); if (result) *result = r; return false; }
    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = pushSize;
    VkPipelineLayoutCreateInfo pl{};
    pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &pipe->descriptors;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &push;
    r = vkCreatePipelineLayout(device, &pl, nullptr, &pipe->layout);
    if (r != VK_SUCCESS) { DestroyPipe(context, *pipe); if (result) *result = r; return false; }
    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.layout = pipe->layout;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = pipe->shader;
    cp.stage.pName = "main";
    r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipe->pipeline);
    if (r != VK_SUCCESS) { DestroyPipe(context, *pipe); if (result) *result = r; return false; }
    if (result) *result = VK_SUCCESS;
    return true;
}

VkResult AllocSet(std::shared_ptr<DeviceContext> const& context,
    VkDescriptorSetLayout layout, VkDescriptorBufferInfo const* infos, uint32_t count,
    VkDescriptorPool* pool, VkDescriptorSet* set) {
    VkDevice device = context->device();
    VkDescriptorPoolSize size{};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = count;
    VkDescriptorPoolCreateInfo dp{};
    dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp.maxSets = 1;
    dp.poolSizeCount = 1;
    dp.pPoolSizes = &size;
    VkResult r = vkCreateDescriptorPool(device, &dp, nullptr, pool);
    if (r != VK_SUCCESS) return r;
    VkDescriptorSetAllocateInfo da{};
    da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    da.descriptorPool = *pool;
    da.descriptorSetCount = 1;
    da.pSetLayouts = &layout;
    r = vkAllocateDescriptorSets(device, &da, set);
    if (r != VK_SUCCESS) return r;
    std::vector<VkWriteDescriptorSet> writes(count);
    for (uint32_t i = 0; i < count; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = *set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, count, writes.data(), 0, nullptr);
    return VK_SUCCESS;
}

struct PushDispatch {
    void const* bytes = nullptr;
    uint32_t size = 0;
    uint32_t groups = 1;
};

VkResult RecordOps(std::shared_ptr<DeviceContext> const& context, PipeObjects const& pipe,
    VkDescriptorSet set, PushDispatch const* dispatches, size_t count,
    VkCommandPool* pool, VkCommandBuffer* command) {
    VkDevice device = context->device();
    VkCommandPoolCreateInfo pc{};
    pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pc.queueFamilyIndex = context->computeQueueFamily();
    VkResult r = vkCreateCommandPool(device, &pc, nullptr, pool);
    if (r != VK_SUCCESS) return r;
    VkCommandBufferAllocateInfo ca{};
    ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ca.commandPool = *pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    r = vkAllocateCommandBuffers(device, &ca, command);
    if (r != VK_SUCCESS) return r;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    r = vkBeginCommandBuffer(*command, &begin);
    if (r != VK_SUCCESS) return r;
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask =
        VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(*command,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    vkCmdBindPipeline(*command, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.pipeline);
    vkCmdBindDescriptorSets(*command, VK_PIPELINE_BIND_POINT_COMPUTE, pipe.layout,
        0, 1, &set, 0, nullptr);
    for (size_t i = 0; i < count; ++i) {
        vkCmdPushConstants(*command, pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            dispatches[i].size, dispatches[i].bytes);
        vkCmdDispatch(*command, dispatches[i].groups, 1, 1);
        if (i + 1 < count) {
            VkMemoryBarrier mid{};
            mid.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mid.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mid.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(*command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mid, 0, nullptr, 0, nullptr);
        }
    }
    VkMemoryBarrier after{};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
        VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(*command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 1, &after, 0, nullptr, 0, nullptr);
    return vkEndCommandBuffer(*command);
}

struct QueryUpload {
    float vp[16]{};
    uint32_t width = 0, height = 0;
    float x = 0.0f, y = 0.0f, radius = 0.0f;
};
static_assert(sizeof(QueryUpload) == 84, "pick query ABI");

bool ValidQuery(PicktileVkQuery const& query) {
    if (query.width == 0 || query.height == 0) return false;
    if (!std::isfinite(query.x) || !std::isfinite(query.y) ||
        !std::isfinite(query.radiusPx) || query.radiusPx < 0 ||
        query.radiusPx > std::sqrt(std::numeric_limits<float>::max()))
        return false;
    for (float value : query.viewProj)
        if (!std::isfinite(value)) return false;
    return true;
}

bool ValidPickGeometry(uint32_t curveCount, uint32_t pointCount) {
    if (pointCount > uint32_t(INT32_MAX)) return false;
    // Vulkan u32-indexing bound (documented): flat float point planes index
    // 3*i in u32; beyond UINT32_MAX/12 points Begin fails closed instead of
    // emitting wrapped device indices. Physically unreachable (4.2GB+).
    if (pointCount > UINT32_MAX / 12) return false;
    if (curveCount == std::numeric_limits<uint32_t>::max()) return false;
    if (curveCount == 0 && pointCount != 0) return false;
    return true;
}

bool GroupsAdmitted(std::shared_ptr<DeviceContext> const& context,
    PushDispatch const* dispatches, size_t count) {
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    uint32_t const limit = physical.limits.maxComputeWorkGroupCount[0];
    for (size_t i = 0; i < count; ++i)
        if (dispatches[i].groups == 0 || dispatches[i].groups > limit) return false;
    return true;
}

// Directed-rounding double ops replicating the CUDA __d*_r* intrinsics used by
// FinalizeBounds in gpu/curveTileBounds.cu. Volatile operands defeat folding
// across the rounding-mode switch; the entry mode is restored by the caller.
double DirectedOp(double a, double b, char op, int mode, bool* ok) {
    if (fesetround(mode) != 0) { *ok = false; return 0.0; }
    volatile double x = a, y = b;
    volatile double value = (op == '*') ? x * y : (op == '+') ? x + y : x - y;
    return value;
}

bool DirectedFloat(double value, float* result, int mode) {
    if (!std::isfinite(value) || value < -double(FLT_MAX) || value > double(FLT_MAX))
        return false;
    if (fesetround(mode) != 0) return false;
    volatile double input = value;
    float rounded = float(input);
    if (!std::isfinite(rounded)) return false;
    *result = rounded;
    return true;
}

// Exact host port of FinalizeBounds (gpu/curveTileBounds.cu), including the
// Catmull-Rom signed-basis overshoot/width model and outward float bounds.
bool FinalizeTileBounds(PicktileVkTileBoundsBasis basis, float const* minimum,
    float const* maximum, float maxWidth, float* outMinimum, float* outMaximum) {
    for (int axis = 0; axis != 3; ++axis)
        if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis])) return false;
    if (!std::isfinite(maxWidth) || maxWidth < 0.0f) return false;
    int const entry = fegetround();
    bool ok = true;
    double const widthScale =
        basis == PicktileVkTileBoundsBasis::CatmullRom ? 9.0 / 16.0 : 1.0 / 2.0;
    double const widthPad = DirectedOp(double(maxWidth), widthScale, '*', FE_UPWARD, &ok);
    double const positionScale =
        basis == PicktileVkTileBoundsBasis::CatmullRom ? 1.0 / 8.0 : 0.0;
    float lo[3], hi[3];
    for (int axis = 0; ok && axis != 3; ++axis) {
        double const range =
            DirectedOp(double(maximum[axis]), double(minimum[axis]), '-', FE_UPWARD, &ok);
        double const positionPad = DirectedOp(range, positionScale, '*', FE_UPWARD, &ok);
        double const lowerInner =
            DirectedOp(double(minimum[axis]), positionPad, '-', FE_DOWNWARD, &ok);
        double const lower = DirectedOp(lowerInner, widthPad, '-', FE_DOWNWARD, &ok);
        double const upperInner =
            DirectedOp(double(maximum[axis]), positionPad, '+', FE_UPWARD, &ok);
        double const upper = DirectedOp(upperInner, widthPad, '+', FE_UPWARD, &ok);
        if (!DirectedFloat(lower, &lo[axis], FE_DOWNWARD) ||
            !DirectedFloat(upper, &hi[axis], FE_UPWARD))
            ok = false;
    }
    fesetround(entry);
    if (!ok) return false;
    for (int axis = 0; axis != 3; ++axis) {
        outMinimum[axis] = lo[axis];
        outMaximum[axis] = hi[axis];
    }
    return true;
}

uint32_t ClampTile(uint32_t value, uint32_t low, uint32_t high) {
    return std::max(low, std::min(value, high));
}

bool ValidIndexOption(PicktileVkIndexBasis basis, PicktileVkIndexWrap wrap,
    PicktileVkIndexMode mode) {
    return basis >= PicktileVkIndexBasis::Linear &&
        basis <= PicktileVkIndexBasis::CentripetalCatmullRom &&
        wrap >= PicktileVkIndexWrap::Nonperiodic && wrap <= PicktileVkIndexWrap::Segmented &&
        mode >= PicktileVkIndexMode::Curves && mode <= PicktileVkIndexMode::Points;
}

bool IndexCubic(PicktileVkIndexBasis basis, PicktileVkIndexMode mode) {
    return basis != PicktileVkIndexBasis::Linear && mode == PicktileVkIndexMode::Curves;
}

} // namespace

bool GetPicktileVkTileRequirements(uint32_t chunkSize, uint32_t tileTarget,
    size_t captureCurveCount, size_t survivorCurveCount, size_t survivorPointCount,
    PicktileVkTileRequirements* result) noexcept {
    if (!result || captureCurveCount > UINT32_MAX || survivorCurveCount > captureCurveCount ||
        survivorPointCount > UINT32_MAX || (survivorCurveCount == 0 && survivorPointCount != 0) ||
        survivorPointCount < survivorCurveCount)
        return false;
    uint32_t const chunk = chunkSize ? ClampTile(chunkSize, 128, 1024) : 512;
    uint32_t const target = tileTarget ? ClampTile(tileTarget, 32, 256) : 64;
    size_t const chunks = captureCurveCount ? (captureCurveCount + chunk - 1) / chunk : 0;
    size_t const chunksPerTile =
        chunks ? std::max<size_t>(1, (chunks + target - 1) / target) : 0;
    size_t const rawTiles = chunks ? (chunks + chunksPerTile - 1) / chunksPerTile : 0;
    size_t const tileCount = chunks
        ? std::min(chunks, std::max<size_t>(32, std::min<size_t>(rawTiles, 256)))
        : 0;
    *result = {chunks, tileCount, chunksPerTile, size_t(chunk) * chunksPerTile, chunk, target};
    return true;
}

bool GetPicktileVkIndexRequirements(PicktileVkIndexBasis basis, PicktileVkIndexWrap wrap,
    PicktileVkIndexMode mode, size_t curveCount, size_t pointCount,
    PicktileVkIndexRequirements* result) noexcept {
    if (!result || !ValidIndexOption(basis, wrap, mode) ||
        (wrap == PicktileVkIndexWrap::Segmented && basis != PicktileVkIndexBasis::Linear) ||
        (curveCount == 0 && pointCount != 0) || pointCount < curveCount ||
        curveCount >= size_t(INT_MAX) || pointCount > size_t(INT_MAX))
        return false;
    size_t max = pointCount;
    if (IndexCubic(basis, mode) && wrap == PicktileVkIndexWrap::Pinned &&
        basis != PicktileVkIndexBasis::Bezier) {
        size_t const extra = basis == PicktileVkIndexBasis::BSpline ? 3 : 1;
        if (curveCount > (size_t(INT_MAX) - pointCount) / extra) return false;
        max = pointCount + extra * curveCount;
    }
    if (max > size_t(INT_MAX)) return false;
    uint32_t const arity =
        mode == PicktileVkIndexMode::Points ? 1u : (IndexCubic(basis, mode) ? 4u : 2u);
    *result = {max, arity};
    return true;
}

// ---------------------------------------------------------------- Pick ---

struct PicktileVkPickPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    PipeObjects pipe;
    ~Native() { DestroyPipe(context, pipe); }
};

namespace {
struct PickResultUpload {
    uint32_t hit = 0, curve = UINT32_MAX, cv = UINT32_MAX, flatIndex = UINT32_MAX;
    uint32_t stableLo = 0, stableHi = 0;
    float distance = 0.0f;
};
static_assert(sizeof(PickResultUpload) == 28, "pick result ABI");
struct PickStateUpload {
    uint32_t error = 0, bestBits = 0x7F800000u, bestIndex = UINT32_MAX;
};
static_assert(sizeof(PickStateUpload) == 12, "pick state ABI");
struct PickPush {
    uint32_t pointCount = 0, curveCount = 0, phase = 0, hasIds = 0;
};
static_assert(sizeof(PickPush) == 16, "pick push ABI");
} // namespace

struct PicktileVkPickPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputPoints, inputOffsets, inputIds, query;
    std::shared_ptr<ChargedBuffer> distances, result, state, idPlaceholder;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    Semantic semantic = Semantic::Ok;
    PicktileVkResult pick{};
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};

PicktileVkPickPipeline::PicktileVkPickPipeline(std::shared_ptr<Native> native)
    : native_(std::move(native)) {}
PicktileVkPickPipeline::~PicktileVkPickPipeline() = default;
std::shared_ptr<DeviceContext> const& PicktileVkPickPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<PicktileVkPickPipeline> PicktileVkPickPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        if (!CreatePipe(native->context, code, 7, uint32_t(sizeof(PickPush)),
                &native->pipe, result))
            return {};
        auto pipeline =
            std::shared_ptr<PicktileVkPickPipeline>(new PicktileVkPickPipeline(std::move(native)));
        if (result) *result = VK_SUCCESS;
        return pipeline;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

PicktileVkPickPipeline::Candidate::Candidate(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
PicktileVkPickPipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void PicktileVkPickPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PicktileVkPickPipeline::Candidate::output() const noexcept {
    return state_->proved && state_->semantic == Semantic::Ok ? state_->result : nullptr;
}
bool PicktileVkPickPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == Semantic::Ok;
}
PicktileVkResult PicktileVkPickPipeline::Candidate::result() const noexcept {
    return state_->pick;
}
std::shared_ptr<const ChargedBuffer> PicktileVkPickPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputPoints : nullptr;
}
std::shared_ptr<DeviceContext> PicktileVkPickPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PicktileVkPickPipeline::Candidate::Poll(Semantic* status) {
    auto& state = *state_;
    if (state.lost) return VK_ERROR_DEVICE_LOST;
    if (!state.proved) {
        VkResult fence = vkGetFenceStatus(state.native->context->device(), state.fence);
        if (fence == VK_NOT_READY) return fence;
        if (fence != VK_SUCCESS) { Quarantine(); return fence; }
        state.pending = false;
        PickStateUpload host{};
        PickResultUpload upload{};
        auto context = state.native->context;
        VkResult r = ReadHost(context, state.state, &host, sizeof(host)) &&
                ReadHost(context, state.result, &upload, sizeof(upload))
            ? VK_SUCCESS
            : VK_ERROR_DEVICE_LOST;
        if (r != VK_SUCCESS) return r;
        state.semantic = host.error == 0 ? Semantic::Ok
            : host.error == 1 ? Semantic::InvalidArgument
            : Semantic::NonFiniteInput;
        state.pick = PicktileVkResult{};
        if (state.semantic == Semantic::Ok) {
            state.pick.hit = upload.hit != 0;
            state.pick.curve = upload.curve;
            state.pick.cv = upload.cv;
            state.pick.flatIndex = upload.flatIndex;
            state.pick.stableId =
                (uint64_t(upload.stableHi) << 32) | upload.stableLo;
            state.pick.distancePx = upload.distance;
        }
        state.proved = true;
    }
    if (status) *status = state.semantic;
    return VK_SUCCESS;
}

std::unique_ptr<PicktileVkPickPipeline::Candidate> PicktileVkPickPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> points, std::shared_ptr<const ChargedBuffer> offsets,
    std::shared_ptr<const ChargedBuffer> stableIds, uint32_t curveCount, uint32_t pointCount,
    PicktileVkQuery const& query, VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    bool const idsProvided = bool(stableIds);
    if (!ValidPickGeometry(curveCount, pointCount) || !ValidQuery(query) ||
        !ValidCharged(context, points, uint64_t(pointCount) * 12) ||
        !ValidCharged(context, offsets, (uint64_t(curveCount) + 1) * 4) ||
        (idsProvided && curveCount != 0 &&
            !ValidCharged(context, stableIds, uint64_t(curveCount) * 8)) ||
        (idsProvided && curveCount == 0 &&
            !ValidCharged(context, stableIds, 0)))
        return {};
    uint32_t const hasIds = idsProvided ? 1u : 0u;
    try {
        auto state = std::make_shared<Candidate::State>();
        state->native = native_;
        state->inputPoints = std::move(points);
        state->inputOffsets = std::move(offsets);
        state->inputIds = std::move(stableIds);
        state->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(state));
        VkResult r = VK_SUCCESS;
        state->distances = MakeHostBuffer(context, VkDeviceSize(pointCount) * 4, &r);
        if (!state->distances) { if (result) *result = r; return {}; }
        state->result = MakeHostBuffer(context, sizeof(PickResultUpload), &r);
        if (!state->result) { if (result) *result = r; return {}; }
        state->state = MakeHostBuffer(context, sizeof(PickStateUpload), &r);
        if (!state->state) { if (result) *result = r; return {}; }
        auto queryBuffer = MakeHostBuffer(context, sizeof(QueryUpload), &r);
        if (!queryBuffer) { if (result) *result = r; return {}; }
        if (!idsProvided) {
            // Even an unused descriptor must retain its allocation until the
            // submitted command completes, including a quarantined command.
            state->idPlaceholder = MakeHostBuffer(context, 4, &r);
            if (!state->idPlaceholder) { if (result) *result = r; return {}; }
        }
        QueryUpload upload{};
        std::memcpy(upload.vp, query.viewProj, sizeof(upload.vp));
        upload.width = query.width;
        upload.height = query.height;
        upload.x = query.x;
        upload.y = query.y;
        upload.radius = query.radiusPx;
        PickResultUpload miss{};
        PickStateUpload clean{};
        if (!FillHost(context, queryBuffer, &upload, sizeof(upload)) ||
            !FillHost(context, state->result, &miss, sizeof(miss)) ||
            !FillHost(context, state->state, &clean, sizeof(clean))) {
            if (result) *result = VK_ERROR_DEVICE_LOST;
            return {};
        }
        state->query = std::move(queryBuffer);
        auto idsBinding = idsProvided ? state->inputIds
            : std::shared_ptr<const ChargedBuffer>(state->idPlaceholder);
        VkDescriptorBufferInfo infos[7] = {
            {state->inputPoints->buffer(), 0, state->inputPoints->sizeBytes()},
            {state->inputOffsets->buffer(), 0, state->inputOffsets->sizeBytes()},
            {idsBinding->buffer(), 0, idsBinding->sizeBytes()},
            {state->distances->buffer(), 0, state->distances->sizeBytes()},
            {state->query->buffer(), 0, state->query->sizeBytes()},
            {state->result->buffer(), 0, state->result->sizeBytes()},
            {state->state->buffer(), 0, state->state->sizeBytes()}};
        VkDescriptorSet set = VK_NULL_HANDLE;
        r = AllocSet(context, native_->pipe.descriptors, infos, 7,
            &state->descriptors, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        PickPush head{pointCount, curveCount, 0u, hasIds};
        PickPush claim{pointCount, curveCount, 1u, hasIds};
        PickPush finish{pointCount, curveCount, 2u, hasIds};
        uint64_t const validateWork =
            std::max<uint64_t>(uint64_t(curveCount) + 1, pointCount);
        PushDispatch dispatches[3] = {
            {&head, uint32_t(sizeof(head)), GroupsFor(validateWork)},
            {&claim, uint32_t(sizeof(claim)), GroupsFor(pointCount)},
            {&finish, uint32_t(sizeof(finish)), 1u}};
        if (!GroupsAdmitted(context, dispatches, 3)) return {};
        VkCommandBuffer command = VK_NULL_HANDLE;
        r = RecordOps(context, native_->pipe, set, dispatches, 3,
            &state->commands, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(context->device(), &fi, nullptr, &state->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!admitted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        state->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, state->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (result) *result = r;
            return {};
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

// ------------------------------------------------------------- Footprint ---

struct PicktileVkFootprintPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    PipeObjects pipe;
    ~Native() { DestroyPipe(context, pipe); }
};

namespace {
struct FootprintStateUpload {
    uint32_t error = 0, count = 0;
};
static_assert(sizeof(FootprintStateUpload) == 8, "footprint state ABI");
struct FootprintPush {
    uint32_t pointCount = 0, curveCount = 0, phase = 0, stride = 0, srcIsB = 0;
};
static_assert(sizeof(FootprintPush) == 20, "footprint push ABI");
} // namespace

struct PicktileVkFootprintPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputPoints, inputOffsets, query;
    std::shared_ptr<ChargedBuffer> flags, scanA, scanB, output, state;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    Semantic semantic = Semantic::Ok;
    uint32_t count = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};

PicktileVkFootprintPipeline::PicktileVkFootprintPipeline(std::shared_ptr<Native> native)
    : native_(std::move(native)) {}
PicktileVkFootprintPipeline::~PicktileVkFootprintPipeline() = default;
std::shared_ptr<DeviceContext> const& PicktileVkFootprintPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<PicktileVkFootprintPipeline> PicktileVkFootprintPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        if (!CreatePipe(native->context, code, 8, uint32_t(sizeof(FootprintPush)),
                &native->pipe, result))
            return {};
        auto pipeline = std::shared_ptr<PicktileVkFootprintPipeline>(
            new PicktileVkFootprintPipeline(std::move(native)));
        if (result) *result = VK_SUCCESS;
        return pipeline;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

PicktileVkFootprintPipeline::Candidate::Candidate(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
PicktileVkFootprintPipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void PicktileVkFootprintPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PicktileVkFootprintPipeline::Candidate::footprint() const noexcept {
    return state_->proved && state_->semantic == Semantic::Ok ? state_->output : nullptr;
}
uint32_t PicktileVkFootprintPipeline::Candidate::footprintCount() const noexcept {
    return state_->proved && state_->semantic == Semantic::Ok ? state_->count : 0;
}
std::shared_ptr<const ChargedBuffer> PicktileVkFootprintPipeline::Candidate::output() const noexcept {
    return footprint();
}
bool PicktileVkFootprintPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == Semantic::Ok;
}
std::shared_ptr<const ChargedBuffer> PicktileVkFootprintPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputPoints : nullptr;
}
std::shared_ptr<DeviceContext> PicktileVkFootprintPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PicktileVkFootprintPipeline::Candidate::Poll(Semantic* status) {
    auto& state = *state_;
    if (state.lost) return VK_ERROR_DEVICE_LOST;
    if (!state.proved) {
        VkResult fence = vkGetFenceStatus(state.native->context->device(), state.fence);
        if (fence == VK_NOT_READY) return fence;
        if (fence != VK_SUCCESS) { Quarantine(); return fence; }
        state.pending = false;
        FootprintStateUpload host{};
        if (!ReadHost(state.native->context, state.state, &host, sizeof(host)))
            return VK_ERROR_DEVICE_LOST;
        state.semantic = host.error == 0 ? Semantic::Ok
            : host.error == 1 ? Semantic::InvalidArgument
            : Semantic::NonFiniteInput;
        state.count = state.semantic == Semantic::Ok ? host.count : 0;
        state.proved = true;
    }
    if (status) *status = state.semantic;
    return VK_SUCCESS;
}

std::unique_ptr<PicktileVkFootprintPipeline::Candidate> PicktileVkFootprintPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> points, std::shared_ptr<const ChargedBuffer> offsets,
    uint32_t curveCount, uint32_t pointCount, PicktileVkQuery const& query,
    VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    if (!ValidPickGeometry(curveCount, pointCount) || !ValidQuery(query) ||
        !ValidCharged(context, points, uint64_t(pointCount) * 12) ||
        !ValidCharged(context, offsets, (uint64_t(curveCount) + 1) * 4))
        return {};
    try {
        auto state = std::make_shared<Candidate::State>();
        state->native = native_;
        state->inputPoints = std::move(points);
        state->inputOffsets = std::move(offsets);
        state->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(state));
        VkResult r = VK_SUCCESS;
        state->flags = MakeHostBuffer(context, VkDeviceSize(pointCount) * 4, &r);
        if (!state->flags) { if (result) *result = r; return {}; }
        state->scanA = MakeHostBuffer(context, VkDeviceSize(pointCount) * 4, &r);
        if (!state->scanA) { if (result) *result = r; return {}; }
        state->scanB = MakeHostBuffer(context, VkDeviceSize(pointCount) * 4, &r);
        if (!state->scanB) { if (result) *result = r; return {}; }
        state->output = MakeHostBuffer(context, VkDeviceSize(pointCount) * 4, &r);
        if (!state->output) { if (result) *result = r; return {}; }
        state->state = MakeHostBuffer(context, sizeof(FootprintStateUpload), &r);
        if (!state->state) { if (result) *result = r; return {}; }
        auto queryBuffer = MakeHostBuffer(context, sizeof(QueryUpload), &r);
        if (!queryBuffer) { if (result) *result = r; return {}; }
        QueryUpload upload{};
        std::memcpy(upload.vp, query.viewProj, sizeof(upload.vp));
        upload.width = query.width;
        upload.height = query.height;
        upload.x = query.x;
        upload.y = query.y;
        upload.radius = query.radiusPx;
        FootprintStateUpload clean{};
        if (!FillHost(context, queryBuffer, &upload, sizeof(upload)) ||
            !FillHost(context, state->state, &clean, sizeof(clean))) {
            if (result) *result = VK_ERROR_DEVICE_LOST;
            return {};
        }
        state->query = std::move(queryBuffer);
        VkDescriptorBufferInfo infos[8] = {
            {state->inputPoints->buffer(), 0, state->inputPoints->sizeBytes()},
            {state->inputOffsets->buffer(), 0, state->inputOffsets->sizeBytes()},
            {state->flags->buffer(), 0, state->flags->sizeBytes()},
            {state->scanA->buffer(), 0, state->scanA->sizeBytes()},
            {state->scanB->buffer(), 0, state->scanB->sizeBytes()},
            {state->output->buffer(), 0, state->output->sizeBytes()},
            {state->query->buffer(), 0, state->query->sizeBytes()},
            {state->state->buffer(), 0, state->state->sizeBytes()}};
        VkDescriptorSet set = VK_NULL_HANDLE;
        r = AllocSet(context, native_->pipe.descriptors, infos, 8,
            &state->descriptors, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        // Phase schedule: validate/mark, one Hillis-Steele step per stride,
        // finalize. Step k reads A when k is even (srcIsB = k odd); the final
        // plane is B exactly when the step count is odd.
        std::vector<FootprintPush> pushes;
        pushes.push_back({pointCount, curveCount, 0u, 0u, 0u});
        uint32_t step = 0;
        for (uint64_t stride = 1; stride < pointCount; stride *= 2, ++step)
            pushes.push_back({pointCount, curveCount, 1u, uint32_t(stride), step & 1u});
        pushes.push_back({pointCount, curveCount, 2u, 0u, step & 1u});
        uint64_t const validateWork =
            std::max<uint64_t>(uint64_t(curveCount) + 1, pointCount);
        std::vector<PushDispatch> dispatches;
        dispatches.reserve(pushes.size());
        for (size_t i = 0; i < pushes.size(); ++i) {
            uint32_t groups = i == 0 ? GroupsFor(validateWork) : GroupsFor(pointCount);
            dispatches.push_back({&pushes[i], uint32_t(sizeof(FootprintPush)), groups});
        }
        if (!GroupsAdmitted(context, dispatches.data(), dispatches.size())) return {};
        VkCommandBuffer command = VK_NULL_HANDLE;
        r = RecordOps(context, native_->pipe, set, dispatches.data(), dispatches.size(),
            &state->commands, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(context->device(), &fi, nullptr, &state->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!admitted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        state->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, state->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (result) *result = r;
            return {};
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

// ----------------------------------------------------------------- Tiles ---

struct PicktileVkTilesPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    PipeObjects pipe;
    ~Native() { DestroyPipe(context, pipe); }
};

namespace {
struct TilesPush {
    uint32_t captureN = 0, survivorN = 0, pointCount = 0, order = 0;
    uint32_t chunkCount = 0, tileCount = 0, chunksPerTile = 0, chunkSize = 0, phase = 0;
};
static_assert(sizeof(TilesPush) == 36, "tiles push ABI");

bool SameTileRequirements(PicktileVkTileRequirements const& a,
    PicktileVkTileRequirements const& b) {
    return a.chunkCount == b.chunkCount && a.tileCount == b.tileCount &&
        a.chunksPerTile == b.chunksPerTile && a.curvesPerTile == b.curvesPerTile &&
        a.chunkSize == b.chunkSize && a.tileTarget == b.tileTarget;
}
} // namespace

struct PicktileVkTilesPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputCapture, inputSurvivors, inputOffsets;
    std::shared_ptr<const ChargedBuffer> inputSorted, inputOrdinals;
    std::shared_ptr<ChargedBuffer> spans, status, sortedPlaceholder, ordinalPlaceholder;
    size_t tileCount = 0;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t deviceStatus = UINT32_MAX;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};

PicktileVkTilesPipeline::PicktileVkTilesPipeline(std::shared_ptr<Native> native)
    : native_(std::move(native)) {}
PicktileVkTilesPipeline::~PicktileVkTilesPipeline() = default;
std::shared_ptr<DeviceContext> const& PicktileVkTilesPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<PicktileVkTilesPipeline> PicktileVkTilesPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        if (!CreatePipe(native->context, code, 7, uint32_t(sizeof(TilesPush)),
                &native->pipe, result))
            return {};
        auto pipeline = std::shared_ptr<PicktileVkTilesPipeline>(
            new PicktileVkTilesPipeline(std::move(native)));
        if (result) *result = VK_SUCCESS;
        return pipeline;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

PicktileVkTilesPipeline::Candidate::Candidate(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
PicktileVkTilesPipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void PicktileVkTilesPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PicktileVkTilesPipeline::Candidate::spans() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->spans : nullptr;
}
std::shared_ptr<const ChargedBuffer> PicktileVkTilesPipeline::Candidate::output() const noexcept {
    return spans();
}
bool PicktileVkTilesPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->deviceStatus == 0;
}
std::shared_ptr<const ChargedBuffer> PicktileVkTilesPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputCapture : nullptr;
}
std::shared_ptr<DeviceContext> PicktileVkTilesPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PicktileVkTilesPipeline::Candidate::Poll(uint32_t* deviceStatus) {
    auto& state = *state_;
    if (state.lost) return VK_ERROR_DEVICE_LOST;
    if (!state.proved) {
        VkResult fence = vkGetFenceStatus(state.native->context->device(), state.fence);
        if (fence == VK_NOT_READY) return fence;
        if (fence != VK_SUCCESS) { Quarantine(); return fence; }
        state.pending = false;
        uint32_t host = UINT32_MAX;
        if (!ReadHost(state.native->context, state.status, &host, sizeof(host)))
            return VK_ERROR_DEVICE_LOST;
        state.deviceStatus = host;
        state.proved = true;
    }
    if (deviceStatus) *deviceStatus = state.deviceStatus;
    return VK_SUCCESS;
}

std::unique_ptr<PicktileVkTilesPipeline::Candidate> PicktileVkTilesPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> captureIds,
    std::shared_ptr<const ChargedBuffer> survivorIds,
    std::shared_ptr<const ChargedBuffer> survivorOffsets,
    std::shared_ptr<const ChargedBuffer> sortedIds,
    std::shared_ptr<const ChargedBuffer> sortedOrdinals,
    uint32_t captureCurveCount, uint32_t survivorCurveCount, uint32_t survivorPointCount,
    PicktileVkTileOrder order, PicktileVkTileRequirements const& requirements,
    VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    PicktileVkTileRequirements expected{};
    bool validOrder = order == PicktileVkTileOrder::SortedSurvivorSubset ||
        order == PicktileVkTileOrder::IdentityCaptureOrder ||
        order == PicktileVkTileOrder::CaptureOrderSurvivorSubset;
    bool const lookup = order == PicktileVkTileOrder::CaptureOrderSurvivorSubset;
    bool const identity = order == PicktileVkTileOrder::IdentityCaptureOrder;
    if (!validOrder ||
        !GetPicktileVkTileRequirements(requirements.chunkSize, requirements.tileTarget,
            captureCurveCount, survivorCurveCount, survivorPointCount, &expected) ||
        !SameTileRequirements(requirements, expected) ||
        !ValidCharged(context, captureIds, uint64_t(captureCurveCount) * 8) ||
        !ValidCharged(context, survivorIds, uint64_t(survivorCurveCount) * 8) ||
        !ValidCharged(context, survivorOffsets, (uint64_t(survivorCurveCount) + 1) * 4) ||
        (lookup && (!ValidCharged(context, sortedIds, uint64_t(captureCurveCount) * 8) ||
            !ValidCharged(context, sortedOrdinals, uint64_t(captureCurveCount) * 4))) ||
        (identity && survivorCurveCount != captureCurveCount))
        return {};
    try {
        auto state = std::make_shared<Candidate::State>();
        state->native = native_;
        state->inputCapture = std::move(captureIds);
        state->inputSurvivors = std::move(survivorIds);
        state->inputOffsets = std::move(survivorOffsets);
        state->inputSorted = std::move(sortedIds);
        state->inputOrdinals = std::move(sortedOrdinals);
        state->tileCount = requirements.tileCount;
        state->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(state));
        VkResult r = VK_SUCCESS;
        state->spans = MakeHostBuffer(context,
            VkDeviceSize(requirements.tileCount) * sizeof(PicktileVkTileSpan), &r);
        if (!state->spans) { if (result) *result = r; return {}; }
        state->status = MakeHostBuffer(context, sizeof(uint32_t), &r);
        if (!state->status) { if (result) *result = r; return {}; }
        if (!lookup) {
            // Unused lookup bindings still need valid descriptors.
            state->sortedPlaceholder = MakeHostBuffer(context, 4, &r);
            if (!state->sortedPlaceholder) { if (result) *result = r; return {}; }
            state->ordinalPlaceholder = MakeHostBuffer(context, 4, &r);
            if (!state->ordinalPlaceholder) { if (result) *result = r; return {}; }
        }
        uint32_t const clean = 0;
        if (!FillHost(context, state->status, &clean, sizeof(clean))) {
            if (result) *result = VK_ERROR_DEVICE_LOST;
            return {};
        }
        auto sortedBinding = lookup ? state->inputSorted
            : std::shared_ptr<const ChargedBuffer>(state->sortedPlaceholder);
        auto ordinalBinding = lookup ? state->inputOrdinals
            : std::shared_ptr<const ChargedBuffer>(state->ordinalPlaceholder);
        VkDescriptorBufferInfo infos[7] = {
            {state->inputCapture->buffer(), 0, state->inputCapture->sizeBytes()},
            {state->inputSurvivors->buffer(), 0, state->inputSurvivors->sizeBytes()},
            {state->inputOffsets->buffer(), 0, state->inputOffsets->sizeBytes()},
            {sortedBinding->buffer(), 0, sortedBinding->sizeBytes()},
            {ordinalBinding->buffer(), 0, ordinalBinding->sizeBytes()},
            {state->spans->buffer(), 0, state->spans->sizeBytes()},
            {state->status->buffer(), 0, state->status->sizeBytes()}};
        VkDescriptorSet set = VK_NULL_HANDLE;
        r = AllocSet(context, native_->pipe.descriptors, infos, 7,
            &state->descriptors, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        TilesPush validate{captureCurveCount, survivorCurveCount, survivorPointCount,
            uint32_t(order), uint32_t(requirements.chunkCount), uint32_t(requirements.tileCount),
            uint32_t(requirements.chunksPerTile), requirements.chunkSize, 0u};
        TilesPush emit = validate;
        emit.phase = 1u;
        uint64_t const validateWork =
            std::max<uint64_t>(captureCurveCount, survivorCurveCount);
        // A zero-tile capture still validates (CUDA parity: the status scalar
        // is meaningful even when no spans emit).
        PushDispatch only[1] = {
            {&validate, uint32_t(sizeof(validate)), GroupsFor(validateWork)}};
        PushDispatch both[2] = {only[0],
            {&emit, uint32_t(sizeof(emit)), GroupsFor(requirements.tileCount)}};
        PushDispatch const* schedule = requirements.tileCount != 0 ? both : only;
        size_t const dispatchCount = requirements.tileCount != 0 ? 2 : 1;
        if (!GroupsAdmitted(context, schedule, dispatchCount)) return {};
        VkCommandBuffer command = VK_NULL_HANDLE;
        r = RecordOps(context, native_->pipe, set, schedule, dispatchCount,
            &state->commands, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(context->device(), &fi, nullptr, &state->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!admitted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        state->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, state->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (result) *result = r;
            return {};
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

// ---------------------------------------------------------------- Bounds ---

struct PicktileVkBoundsPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    PipeObjects pipe;
    ~Native() { DestroyPipe(context, pipe); }
};

namespace {
struct BoundsPush {
    uint32_t pointDomain = 0, tileCount = 0, phase = 0;
};
static_assert(sizeof(BoundsPush) == 12, "bounds push ABI");

bool ValidBoundsBasis(PicktileVkTileBoundsBasis basis) {
    return basis == PicktileVkTileBoundsBasis::Linear ||
        basis == PicktileVkTileBoundsBasis::BSpline ||
        basis == PicktileVkTileBoundsBasis::CatmullRom;
}
} // namespace

struct PicktileVkBoundsPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputPoints, inputWidths, inputSpans;
    std::shared_ptr<ChargedBuffer> scratchMin, scratchMax, scratchW;
    std::shared_ptr<ChargedBuffer> minimums, maximums, status;
    uint32_t tileCount = 0;
    PicktileVkTileBoundsBasis basis = PicktileVkTileBoundsBasis::BSpline;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false, empty = false;
    uint32_t deviceStatus = UINT32_MAX;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};

PicktileVkBoundsPipeline::PicktileVkBoundsPipeline(std::shared_ptr<Native> native)
    : native_(std::move(native)) {}
PicktileVkBoundsPipeline::~PicktileVkBoundsPipeline() = default;
std::shared_ptr<DeviceContext> const& PicktileVkBoundsPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<PicktileVkBoundsPipeline> PicktileVkBoundsPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        if (!CreatePipe(native->context, code, 7, uint32_t(sizeof(BoundsPush)),
                &native->pipe, result))
            return {};
        auto pipeline = std::shared_ptr<PicktileVkBoundsPipeline>(
            new PicktileVkBoundsPipeline(std::move(native)));
        if (result) *result = VK_SUCCESS;
        return pipeline;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

PicktileVkBoundsPipeline::Candidate::Candidate(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
PicktileVkBoundsPipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void PicktileVkBoundsPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PicktileVkBoundsPipeline::Candidate::minimums() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->minimums : nullptr;
}
std::shared_ptr<const ChargedBuffer> PicktileVkBoundsPipeline::Candidate::maximums() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->maximums : nullptr;
}
std::shared_ptr<const ChargedBuffer> PicktileVkBoundsPipeline::Candidate::output() const noexcept {
    return minimums();
}
bool PicktileVkBoundsPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->deviceStatus == 0;
}
std::shared_ptr<const ChargedBuffer> PicktileVkBoundsPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputPoints : nullptr;
}
std::shared_ptr<DeviceContext> PicktileVkBoundsPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PicktileVkBoundsPipeline::Candidate::Poll(uint32_t* deviceStatus) {
    auto& state = *state_;
    if (state.lost) return VK_ERROR_DEVICE_LOST;
    if (!state.proved) {
        if (!state.empty) {
            VkResult fence = vkGetFenceStatus(state.native->context->device(), state.fence);
            if (fence == VK_NOT_READY) return fence;
            if (fence != VK_SUCCESS) { Quarantine(); return fence; }
            state.pending = false;
        }
        auto context = state.native->context;
        uint32_t host = UINT32_MAX;
        if (!ReadHost(context, state.status, &host, sizeof(host)))
            return VK_ERROR_DEVICE_LOST;
        uint32_t semantic = host;
        // Host finalize into staging; publish to device buffers only when
        // every tile finalizes (aggregate-status-gated publish parity).
        if (semantic == 0 && state.tileCount != 0) {
            std::vector<float> scratchMin(size_t(state.tileCount) * 3);
            std::vector<float> scratchMax(size_t(state.tileCount) * 3);
            std::vector<float> scratchW(state.tileCount);
            std::vector<float> lo(size_t(state.tileCount) * 3);
            std::vector<float> hi(size_t(state.tileCount) * 3);
            if (!ReadHost(context, state.scratchMin, scratchMin.data(),
                    scratchMin.size() * 4) ||
                !ReadHost(context, state.scratchMax, scratchMax.data(),
                    scratchMax.size() * 4) ||
                !ReadHost(context, state.scratchW, scratchW.data(),
                    scratchW.size() * 4)) {
                return VK_ERROR_DEVICE_LOST;
            }
            for (uint32_t tile = 0; tile < state.tileCount; ++tile) {
                if (!FinalizeTileBounds(state.basis, &scratchMin[size_t(tile) * 3],
                        &scratchMax[size_t(tile) * 3], scratchW[tile],
                        &lo[size_t(tile) * 3], &hi[size_t(tile) * 3])) {
                    semantic = 1;
                    break;
                }
            }
            if (semantic == 0 &&
                (!FillHost(context, state.minimums, lo.data(), lo.size() * 4) ||
                    !FillHost(context, state.maximums, hi.data(), hi.size() * 4)))
                return VK_ERROR_DEVICE_LOST;
        }
        if (semantic != 0 && host == 0) {
            // A host finalize failure must also surface in the status scalar
            // (CUDA parity: finalize failures set device status before the
            // gated publish, which then publishes nothing).
            uint32_t const failed = 1;
            if (!FillHost(context, state.status, &failed, sizeof(failed)))
                return VK_ERROR_DEVICE_LOST;
        }
        state.deviceStatus = semantic;
        state.proved = true;
    }
    if (deviceStatus) *deviceStatus = state.deviceStatus;
    return VK_SUCCESS;
}

std::unique_ptr<PicktileVkBoundsPipeline::Candidate> PicktileVkBoundsPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> points, std::shared_ptr<const ChargedBuffer> widths,
    std::shared_ptr<const ChargedBuffer> spans, uint32_t pointDomain, uint32_t tileCount,
    PicktileVkTileBoundsBasis basis, VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    if (!ValidBoundsBasis(basis) || tileCount > uint32_t(INT_MAX) ||
        pointDomain > UINT32_MAX / 12 ||
        !ValidCharged(context, points, uint64_t(pointDomain) * 12) ||
        !ValidCharged(context, widths, uint64_t(pointDomain) * 4) ||
        !ValidCharged(context, spans, uint64_t(tileCount) * sizeof(PicktileVkTileSpan)))
        return {};
    try {
        auto state = std::make_shared<Candidate::State>();
        state->native = native_;
        state->inputPoints = std::move(points);
        state->inputWidths = std::move(widths);
        state->inputSpans = std::move(spans);
        state->tileCount = tileCount;
        state->basis = basis;
        state->empty = tileCount == 0;
        state->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(state));
        VkResult r = VK_SUCCESS;
        state->scratchMin = MakeHostBuffer(context, VkDeviceSize(tileCount) * 12, &r);
        if (!state->scratchMin) { if (result) *result = r; return {}; }
        state->scratchMax = MakeHostBuffer(context, VkDeviceSize(tileCount) * 12, &r);
        if (!state->scratchMax) { if (result) *result = r; return {}; }
        state->scratchW = MakeHostBuffer(context, VkDeviceSize(tileCount) * 4, &r);
        if (!state->scratchW) { if (result) *result = r; return {}; }
        state->minimums = MakeHostBuffer(context, VkDeviceSize(tileCount) * 12, &r);
        if (!state->minimums) { if (result) *result = r; return {}; }
        state->maximums = MakeHostBuffer(context, VkDeviceSize(tileCount) * 12, &r);
        if (!state->maximums) { if (result) *result = r; return {}; }
        state->status = MakeHostBuffer(context, sizeof(uint32_t), &r);
        if (!state->status) { if (result) *result = r; return {}; }
        uint32_t const clean = 0;
        if (!FillHost(context, state->status, &clean, sizeof(clean))) {
            if (result) *result = VK_ERROR_DEVICE_LOST;
            return {};
        }
        if (state->empty) {
            // Zero-tile capture is valid and clears status (CUDA parity: no
            // kernels run). Poll proves immediately without a fence.
            if (result) *result = VK_SUCCESS;
            return candidate;
        }
        VkDescriptorBufferInfo infos[7] = {
            {state->inputPoints->buffer(), 0, state->inputPoints->sizeBytes()},
            {state->inputWidths->buffer(), 0, state->inputWidths->sizeBytes()},
            {state->inputSpans->buffer(), 0, state->inputSpans->sizeBytes()},
            {state->scratchMin->buffer(), 0, state->scratchMin->sizeBytes()},
            {state->scratchMax->buffer(), 0, state->scratchMax->sizeBytes()},
            {state->scratchW->buffer(), 0, state->scratchW->sizeBytes()},
            {state->status->buffer(), 0, state->status->sizeBytes()}};
        VkDescriptorSet set = VK_NULL_HANDLE;
        r = AllocSet(context, native_->pipe.descriptors, infos, 7,
            &state->descriptors, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        BoundsPush validate{pointDomain, tileCount, 0u};
        BoundsPush reduce{pointDomain, tileCount, 1u};
        // One workgroup per tile (CUDA grid parity); each group is 256 wide.
        PushDispatch dispatches[2] = {
            {&validate, uint32_t(sizeof(validate)), tileCount},
            {&reduce, uint32_t(sizeof(reduce)), tileCount}};
        if (!GroupsAdmitted(context, dispatches, 2)) return {};
        VkCommandBuffer command = VK_NULL_HANDLE;
        r = RecordOps(context, native_->pipe, set, dispatches, 2,
            &state->commands, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(context->device(), &fi, nullptr, &state->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!admitted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        state->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, state->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (result) *result = r;
            return {};
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

// --------------------------------------------------------------- Indices ---

struct PicktileVkIndicesPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    PipeObjects pipe;
    ~Native() { DestroyPipe(context, pipe); }
};

namespace {
struct IndicesPush {
    uint32_t curveCount = 0, pointCount = 0, pointBase = 0;
    uint32_t basis = 0, wrap = 0, mode = 0;
    uint32_t maxRecords = 0, arity = 0;
    uint32_t phase = 0, stride = 0, srcIsB = 0, spanInvalid = 0;
};
static_assert(sizeof(IndicesPush) == 48, "indices push ABI");
struct IndicesTotal {
    uint32_t lo = 0, hi = 0;
};
static_assert(sizeof(IndicesTotal) == 8, "indices total ABI");
struct IndicesState {
    uint32_t status = 0, drawCount = 0;
};
static_assert(sizeof(IndicesState) == 8, "indices state ABI");
} // namespace

struct PicktileVkIndicesPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const ChargedBuffer> inputOffsets;
    std::shared_ptr<ChargedBuffer> countsA, scanB, offsetsOut;
    std::shared_ptr<ChargedBuffer> indices, primParam, total, state;
    size_t maxRecords = 0;
    uint32_t arity = 0;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t deviceStatus = UINT32_MAX;
    uint64_t records = 0;
    uint32_t draw = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto device = native->context->device();
        if (fence) vkDestroyFence(device, fence, nullptr);
        if (commands) vkDestroyCommandPool(device, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(device, descriptors, nullptr);
    }
};

PicktileVkIndicesPipeline::PicktileVkIndicesPipeline(std::shared_ptr<Native> native)
    : native_(std::move(native)) {}
PicktileVkIndicesPipeline::~PicktileVkIndicesPipeline() = default;
std::shared_ptr<DeviceContext> const& PicktileVkIndicesPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<PicktileVkIndicesPipeline> PicktileVkIndicesPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    if (!context) return {};
    try {
        auto native = std::make_shared<Native>();
        native->context = std::move(context);
        if (!CreatePipe(native->context, code, 8, uint32_t(sizeof(IndicesPush)),
                &native->pipe, result))
            return {};
        auto pipeline = std::shared_ptr<PicktileVkIndicesPipeline>(
            new PicktileVkIndicesPipeline(std::move(native)));
        if (result) *result = VK_SUCCESS;
        return pipeline;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

PicktileVkIndicesPipeline::Candidate::Candidate(std::shared_ptr<State> state)
    : state_(std::move(state)) {}
PicktileVkIndicesPipeline::Candidate::~Candidate() {
    if (state_->pending) Quarantine();
}
void PicktileVkIndicesPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
std::shared_ptr<const ChargedBuffer> PicktileVkIndicesPipeline::Candidate::indices() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->indices : nullptr;
}
std::shared_ptr<const ChargedBuffer> PicktileVkIndicesPipeline::Candidate::primitiveParam() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->primParam : nullptr;
}
uint64_t PicktileVkIndicesPipeline::Candidate::recordCount() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->records : 0;
}
uint32_t PicktileVkIndicesPipeline::Candidate::drawCount() const noexcept {
    return state_->proved && state_->deviceStatus == 0 ? state_->draw : 0;
}
std::shared_ptr<const ChargedBuffer> PicktileVkIndicesPipeline::Candidate::output() const noexcept {
    return indices();
}
bool PicktileVkIndicesPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->deviceStatus == 0;
}
std::shared_ptr<const ChargedBuffer> PicktileVkIndicesPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputOffsets : nullptr;
}
std::shared_ptr<DeviceContext> PicktileVkIndicesPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult PicktileVkIndicesPipeline::Candidate::Poll(uint32_t* deviceStatus) {
    auto& state = *state_;
    if (state.lost) return VK_ERROR_DEVICE_LOST;
    if (!state.proved) {
        VkResult fence = vkGetFenceStatus(state.native->context->device(), state.fence);
        if (fence == VK_NOT_READY) return fence;
        if (fence != VK_SUCCESS) { Quarantine(); return fence; }
        state.pending = false;
        IndicesTotal total{};
        IndicesState host{};
        auto context = state.native->context;
        if (!ReadHost(context, state.total, &total, sizeof(total)) ||
            !ReadHost(context, state.state, &host, sizeof(host)))
            return VK_ERROR_DEVICE_LOST;
        state.deviceStatus = host.status;
        state.records = state.deviceStatus == 0
            ? (uint64_t(total.hi) << 32) | total.lo
            : 0;
        state.draw = state.deviceStatus == 0 ? host.drawCount : 0;
        state.proved = true;
    }
    if (deviceStatus) *deviceStatus = state.deviceStatus;
    return VK_SUCCESS;
}

std::unique_ptr<PicktileVkIndicesPipeline::Candidate> PicktileVkIndicesPipeline::Begin(
    std::shared_ptr<const ChargedBuffer> offsets, uint32_t curveCount, uint32_t pointCount,
    uint32_t pointBase, PicktileVkIndexBasis basis, PicktileVkIndexWrap wrap,
    PicktileVkIndexMode mode, PicktileVkIndexRequirements const& requirements,
    VkResult* result, BeforeSubmit beforeSubmit) {
    if (result) *result = VK_ERROR_INITIALIZATION_FAILED;
    auto context = native_->context;
    PicktileVkIndexRequirements expected{};
    bool const spanInvalid =
        uint64_t(pointBase) + uint64_t(pointCount) > uint64_t(UINT32_MAX);
    if (!ValidIndexOption(basis, wrap, mode) ||
        !GetPicktileVkIndexRequirements(basis, wrap, mode, curveCount, pointCount, &expected) ||
        expected.maxRecords != requirements.maxRecords ||
        expected.indexArity != requirements.indexArity || requirements.indexArity == 0 ||
        uint64_t(requirements.maxRecords) * requirements.indexArity > uint64_t(UINT32_MAX) ||
        !ValidCharged(context, offsets, (uint64_t(curveCount) + 1) * 4))
        return {};
    try {
        auto state = std::make_shared<Candidate::State>();
        state->native = native_;
        state->inputOffsets = std::move(offsets);
        state->maxRecords = requirements.maxRecords;
        state->arity = requirements.indexArity;
        state->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(state));
        VkResult r = VK_SUCCESS;
        uint64_t const scanLanes = uint64_t(curveCount) + 1;
        state->countsA = MakeHostBuffer(context, scanLanes * 4, &r);
        if (!state->countsA) { if (result) *result = r; return {}; }
        state->scanB = MakeHostBuffer(context, scanLanes * 4, &r);
        if (!state->scanB) { if (result) *result = r; return {}; }
        state->offsetsOut = MakeHostBuffer(context, scanLanes * 4, &r);
        if (!state->offsetsOut) { if (result) *result = r; return {}; }
        state->indices = MakeHostBuffer(context,
            uint64_t(requirements.maxRecords) * requirements.indexArity * 4, &r);
        if (!state->indices) { if (result) *result = r; return {}; }
        state->primParam =
            MakeHostBuffer(context, uint64_t(requirements.maxRecords) * 4, &r);
        if (!state->primParam) { if (result) *result = r; return {}; }
        state->total = MakeHostBuffer(context, sizeof(IndicesTotal), &r);
        if (!state->total) { if (result) *result = r; return {}; }
        state->state = MakeHostBuffer(context, sizeof(IndicesState), &r);
        if (!state->state) { if (result) *result = r; return {}; }
        IndicesTotal cleanTotal{};
        IndicesState cleanState{};
        if (!FillHost(context, state->total, &cleanTotal, sizeof(cleanTotal)) ||
            !FillHost(context, state->state, &cleanState, sizeof(cleanState))) {
            if (result) *result = VK_ERROR_DEVICE_LOST;
            return {};
        }
        VkDescriptorBufferInfo infos[8] = {
            {state->inputOffsets->buffer(), 0, state->inputOffsets->sizeBytes()},
            {state->countsA->buffer(), 0, state->countsA->sizeBytes()},
            {state->scanB->buffer(), 0, state->scanB->sizeBytes()},
            {state->offsetsOut->buffer(), 0, state->offsetsOut->sizeBytes()},
            {state->indices->buffer(), 0, state->indices->sizeBytes()},
            {state->primParam->buffer(), 0, state->primParam->sizeBytes()},
            {state->total->buffer(), 0, state->total->sizeBytes()},
            {state->state->buffer(), 0, state->state->sizeBytes()}};
        VkDescriptorSet set = VK_NULL_HANDLE;
        r = AllocSet(context, native_->pipe.descriptors, infos, 8,
            &state->descriptors, &set);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        uint32_t const invalid = spanInvalid ? 1u : 0u;
        std::vector<IndicesPush> pushes;
        pushes.push_back({curveCount, pointCount, pointBase, uint32_t(basis),
            uint32_t(wrap), uint32_t(mode), uint32_t(requirements.maxRecords),
            requirements.indexArity, 0u, 0u, 0u, invalid});
        uint32_t step = 0;
        for (uint64_t stride = 1; stride < scanLanes; stride *= 2, ++step)
            pushes.push_back({curveCount, pointCount, pointBase, uint32_t(basis),
                uint32_t(wrap), uint32_t(mode), uint32_t(requirements.maxRecords),
                requirements.indexArity, 1u, uint32_t(stride), step & 1u, invalid});
        uint32_t const finalIsB = step & 1u;
        pushes.push_back({curveCount, pointCount, pointBase, uint32_t(basis),
            uint32_t(wrap), uint32_t(mode), uint32_t(requirements.maxRecords),
            requirements.indexArity, 2u, 0u, finalIsB, invalid});
        bool const emit = curveCount != 0;
        if (emit)
            pushes.push_back({curveCount, pointCount, pointBase, uint32_t(basis),
                uint32_t(wrap), uint32_t(mode), uint32_t(requirements.maxRecords),
                requirements.indexArity, 3u, 0u, finalIsB, invalid});
        pushes.push_back({curveCount, pointCount, pointBase, uint32_t(basis),
            uint32_t(wrap), uint32_t(mode), uint32_t(requirements.maxRecords),
            requirements.indexArity, 4u, 0u, finalIsB, invalid});
        std::vector<PushDispatch> dispatches;
        dispatches.reserve(pushes.size());
        for (size_t i = 0; i < pushes.size(); ++i) {
            uint32_t groups = 1;
            if (pushes[i].phase == 0u)
                groups = GroupsFor(std::max<uint64_t>(curveCount, 1));
            else if (pushes[i].phase == 1u)
                groups = GroupsFor(scanLanes);
            else if (pushes[i].phase == 3u)
                groups = curveCount; // one workgroup per curve (CUDA grid parity)
            dispatches.push_back({&pushes[i], uint32_t(sizeof(IndicesPush)), groups});
        }
        if (!GroupsAdmitted(context, dispatches.data(), dispatches.size())) return {};
        VkCommandBuffer command = VK_NULL_HANDLE;
        r = RecordOps(context, native_->pipe, set, dispatches.data(), dispatches.size(),
            &state->commands, &command);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(context->device(), &fi, nullptr, &state->fence);
        if (r != VK_SUCCESS) { if (result) *result = r; return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try {
                admitted = beforeSubmit();
            } catch (...) {
                if (result) *result = VK_ERROR_UNKNOWN;
                return {};
            }
            if (!admitted) { if (result) *result = VK_ERROR_OUT_OF_DEVICE_MEMORY; return {}; }
        }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        state->pending = true;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, state->fence);
        if (r != VK_SUCCESS) {
            candidate->Quarantine();
            if (result) *result = r;
            return {};
        }
        if (result) *result = VK_SUCCESS;
        return candidate;
    } catch (std::bad_alloc const&) {
        if (result) *result = VK_ERROR_OUT_OF_HOST_MEMORY;
        return {};
    }
}

// ----------------------------------------------------------------- Leases ---

namespace {

VulkanSourceGeneration::PlaneView const* FindPlane(
    std::vector<VulkanSourceGeneration::PlaneView> const& planes, char const* name) {
    for (auto const& plane : planes)
        if (plane.metadata.name == name) return &plane;
    return nullptr;
}

// Overflow-safe bytes == elements * elementBytes check (counts are u64).
bool BytesMatch(uint64_t bytes, uint64_t elements, uint32_t elementBytes) {
    return elementBytes != 0 && bytes / elementBytes == elements &&
        bytes % elementBytes == 0;
}

bool FillSlice(PicktileVkPlaneSlice* slice,
    std::vector<VulkanSourceGeneration::PlaneView> const& planes, char const* name,
    uint64_t elements, uint32_t elementBytes, bool required) {
    auto const* plane = FindPlane(planes, name);
    if (!plane) return !required;
    uint64_t const bytes = uint64_t(plane->bytes);
    if (!BytesMatch(bytes, elements, elementBytes)) return false;
    if (elements == 0) {
        // Canonical empty plane (null buffer, zero bytes) or a bindable
        // minimum-size placeholder: either carries no elements.
        slice->buffer = plane->buffer;
        slice->byteOffset = 0;
        slice->elementCount = 0;
        slice->elementBytes = elementBytes;
        return true;
    }
    if (plane->buffer == VK_NULL_HANDLE) return false;
    slice->buffer = plane->buffer;
    slice->byteOffset = 0;
    slice->elementCount = elements;
    slice->elementBytes = elementBytes;
    return true;
}

bool SlicePlane(PicktileVkPlaneSlice source, uint64_t begin, uint64_t count,
    PicktileVkPlaneSlice* result) {
    if (!result || begin > source.elementCount || count > source.elementCount - begin)
        return false;
    if (count != 0 && source.buffer == VK_NULL_HANDLE) return false;
    result->buffer = source.buffer;
    result->byteOffset = source.byteOffset + begin * source.elementBytes;
    result->elementCount = count;
    result->elementBytes = source.elementBytes;
    return true;
}

} // namespace

PicktileVkGeometryLease::PicktileVkGeometryLease(
    VulkanGenerationLease lease, PicktileVkGeometryView view) noexcept
    : lease_(std::move(lease)), view_(view), valid_(true) {}

UsdGenDeviceStatus PicktileVkGeometryLease::WaitUntilReady() const noexcept {
    return valid_ ? lease_.WaitUntilReady() : UsdGenDeviceStatus::InvalidLease;
}

void PicktileVkGeometryLease::Complete() noexcept {
    view_ = PicktileVkGeometryView{};
    valid_ = false;
    lease_.Complete();
}

PicktileVkGeometryLease AcquirePicktileVkGeometry(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    UsdGenDeviceStream queueStream) noexcept {
    if (!generation || generation->Identity().backend != UsdGenDeviceBackend::Vulkan)
        return {};
    VulkanGenerationLease inner = AcquireVulkanGeneration(generation, queueStream);
    if (!inner) return {};
    if (inner.WaitUntilReady() != UsdGenDeviceStatus::Ok) return {};
    auto const& geometry = generation->Geometry();
    uint64_t const curves = geometry.curveCount;
    uint64_t const points = geometry.pointCount;
    if (curves == std::numeric_limits<uint64_t>::max()) return {};
    auto const& planes = inner.Planes();
    PicktileVkGeometryView view{};
    view.curveCount = curves;
    view.pointCount = points;
    if (!FillSlice(&view.points, planes, "points", points, 12, true) ||
        !FillSlice(&view.curveOffsets, planes, "curveOffsets", curves + 1, 4, true) ||
        !FillSlice(&view.stableIds, planes, "stableIds", curves, 8, false) ||
        !FillSlice(&view.restPoints, planes, "rest", points, 12, false) ||
        !FillSlice(&view.widths, planes, "width", points, 4, false) ||
        !FillSlice(&view.hairT, planes, "hairT", points, 4, false) ||
        !FillSlice(&view.rootPrim, planes, "rootPrim", curves, 4, false) ||
        !FillSlice(&view.rootUV, planes, "rootUV", curves, 8, false))
        return {};
    // Presence implies validity here: a misshapen stableIds plane already
    // failed closed above.
    view.hasStableIds = FindPlane(planes, "stableIds") != nullptr;
    return PicktileVkGeometryLease(std::move(inner), view);
}

PicktileVkTileLease::PicktileVkTileLease(
    PicktileVkGeometryLease parent, PicktileVkTileView view) noexcept
    : parent_(std::move(parent)), view_(view), valid_(true) {}

PicktileVkTileLease AcquirePicktileVkTile(
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    uint32_t tileId, uint64_t expectedGeneration, UsdGenDeviceStream queueStream) noexcept {
    if (!generation || generation->Identity().backend != UsdGenDeviceBackend::Vulkan ||
        generation->Identity().generation != expectedGeneration)
        return {};
    UsdGenDeviceTileMetadata const* tile = nullptr;
    for (auto const& candidate : generation->Geometry().tiles) {
        if (candidate.tile == tileId) {
            tile = &candidate;
            break;
        }
    }
    if (!tile) return {};
    PicktileVkGeometryLease parent = AcquirePicktileVkGeometry(generation, queueStream);
    if (!parent) return {};
    PicktileVkGeometryView const& geometry = parent.View();
    if (tile->curveCount > tile->pointCount ||
        (tile->curveCount == 0 && tile->pointCount != 0) ||
        tile->firstCurve > geometry.curveCount ||
        tile->curveCount > geometry.curveCount - tile->firstCurve ||
        tile->curveCount == std::numeric_limits<uint64_t>::max() ||
        tile->firstPoint > std::numeric_limits<uint32_t>::max() ||
        tile->pointCount > std::numeric_limits<uint32_t>::max() - tile->firstPoint)
        return {};
    PicktileVkTileView view{};
    view.range = *tile;
    view.generation = generation->Identity().generation;
    view.topologyVersion = generation->Geometry().topologyVersion;
    view.valueVersion = generation->Geometry().valueVersion;
    // The extra offset sentinel is part of every non-empty and empty tile.
    uint64_t const offsetCount = tile->curveCount + 1;
    if (!SlicePlane(geometry.curveOffsets, tile->firstCurve, offsetCount, &view.curveOffsets) ||
        !SlicePlane(geometry.points, tile->firstPoint, tile->pointCount, &view.points) ||
        !SlicePlane(geometry.restPoints, tile->firstPoint, tile->pointCount, &view.restPoints) ||
        !SlicePlane(geometry.widths, tile->firstPoint, tile->pointCount, &view.widths) ||
        !SlicePlane(geometry.hairT, tile->firstPoint, tile->pointCount, &view.hairT) ||
        !SlicePlane(geometry.stableIds, tile->firstCurve, tile->curveCount, &view.stableIds))
        return {};
    // Optional channels absent from the generation slice to empty views; a
    // half-present root pair fails closed (CUDA AcquireGeometryTile parity).
    bool const rootsEmpty = geometry.rootPrim.elementCount == 0;
    bool const uvsEmpty = geometry.rootUV.elementCount == 0;
    if (rootsEmpty != uvsEmpty) return {};
    if (!SlicePlane(geometry.rootPrim, tile->firstCurve, tile->curveCount, &view.rootPrim) ||
        !SlicePlane(geometry.rootUV, tile->firstCurve, tile->curveCount, &view.rootUV))
        return {};
    return PicktileVkTileLease(std::move(parent), view);
}

std::shared_ptr<const UsdGenDeviceGeneration> PicktileVkWithTileMetadata(
    std::shared_ptr<const UsdGenDeviceGeneration> const& candidate,
    std::vector<UsdGenDeviceTileMetadata> tiles, std::string* reason,
    UsdGenDeviceCurveTopologyMetadata curveTopology) {
    auto fail = [&](char const* message) {
        if (reason) *reason = message;
        return std::shared_ptr<const UsdGenDeviceGeneration>{};
    };
    if (!candidate || !candidate->Owner())
        return fail("tile metadata requires an owned generation");
    auto const& geometry = candidate->Geometry();
    uint64_t curveEnd = 0, pointEnd = 0, previousId = 0;
    bool first = true;
    for (auto const& tile : tiles) {
        if (!first && tile.tile <= previousId)
            return fail("tile IDs must be strictly increasing");
        first = false;
        previousId = tile.tile;
        if (tile.curveCount > tile.pointCount ||
            (tile.curveCount == 0 && tile.pointCount != 0) ||
            tile.firstPoint > std::numeric_limits<uint32_t>::max() ||
            tile.pointCount > std::numeric_limits<uint32_t>::max() - tile.firstPoint ||
            tile.firstCurve != curveEnd || tile.firstPoint != pointEnd ||
            tile.curveCount > geometry.curveCount - curveEnd ||
            tile.pointCount > geometry.pointCount - pointEnd ||
            tile.curveCount > std::numeric_limits<uint64_t>::max() - curveEnd ||
            tile.pointCount > std::numeric_limits<uint64_t>::max() - pointEnd)
            return fail("tile ranges must be contiguous and in bounds");
        curveEnd += tile.curveCount;
        pointEnd += tile.pointCount;
    }
    if (tiles.empty()) {
        if (geometry.curveCount != 0 || geometry.pointCount != 0)
            return fail("non-empty geometry requires tile metadata");
    } else if (curveEnd != geometry.curveCount || pointEnd != geometry.pointCount) {
        return fail("tile metadata does not cover geometry");
    }
    UsdGenDeviceGeneration::CreateInfo info;
    info.identity = candidate->Identity();
    info.geometry = geometry;
    info.geometry.tiles = std::move(tiles);
    if (curveTopology.type != UsdGenDeviceCurveType::Unknown ||
        curveTopology.basis != UsdGenDeviceCurveBasis::Unknown ||
        curveTopology.wrap != UsdGenDeviceCurveWrap::Unknown) {
        info.geometry.curveTopology = curveTopology;
    }
    info.tool = candidate->Tool();
    info.channels = candidate->Channels();
    info.owner = candidate->Owner();
    return UsdGenDeviceGeneration::Create(std::move(info), reason);
}

bool ValidatePicktileVkToolChannels(UsdGenDeviceGeometryMetadata const& geometry,
    std::vector<UsdGenDeviceChannelMetadata> const& channels, std::string* reason) {
    auto fail = [&](char const* message) {
        if (reason) *reason = message;
        return false;
    };
    auto find = [&](char const* name) -> UsdGenDeviceChannelMetadata const* {
        for (auto const& channel : channels)
            if (channel.name == name) return &channel;
        return nullptr;
    };
    auto const* points = find("points");
    auto const* offsets = find("curveOffsets");
    if (geometry.curveCount == std::numeric_limits<uint64_t>::max())
        return fail("Vulkan tools reject a degenerate curve count");
    if (!points || points->elementCount != geometry.pointCount)
        return fail("Vulkan tools require an exact points channel");
    if (!offsets || offsets->elementCount != geometry.curveCount + 1)
        return fail("Vulkan tools require an exact curveOffsets channel");
    auto check = [&](char const* name, uint64_t elements) {
        auto const* channel = find(name);
        if (channel && channel->elementCount != elements)
            return fail("Vulkan tool channel has an inexact element count");
        return true;
    };
    if (!check("stableIds", geometry.curveCount) || !check("rest", geometry.pointCount) ||
        !check("width", geometry.pointCount) || !check("hairT", geometry.pointCount) ||
        !check("rootPrim", geometry.curveCount) || !check("rootUV", geometry.curveCount))
        return false;
    return true;
}

bool PicktileVkPublishTileMetadata(PicktileVkTileSpan const* spans,
    float const* minimums, float const* maximums, size_t tileCount,
    std::vector<UsdGenDeviceTileMetadata>* tiles) {
    if (!tiles) return false;
    if (tileCount != 0 && (!spans || !minimums || !maximums)) return false;
    std::vector<UsdGenDeviceTileMetadata> result;
    result.reserve(tileCount);
    for (size_t i = 0; i < tileCount; ++i) {
        UsdGenDeviceTileMetadata tile{};
        tile.tile = spans[i].tile;
        tile.firstCurve = spans[i].firstCurve;
        tile.curveCount = spans[i].curveCount;
        tile.firstPoint = spans[i].firstPoint;
        tile.pointCount = spans[i].pointCount;
        for (size_t axis = 0; axis != 3; ++axis) {
            float const lo = minimums[i * 3 + axis];
            float const hi = maximums[i * 3 + axis];
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) return false;
            tile.extentMin[axis] = lo;
            tile.extentMax[axis] = hi;
        }
        tile.boundsValid = true;
        result.push_back(tile);
    }
    *tiles = std::move(result);
    return true;
}

// ---------------------------------------------------------- Tool session ---

PicktileVkToolSession::PicktileVkToolSession(std::shared_ptr<DeviceContext> context)
    : context_(std::move(context)) {}
PicktileVkToolSession::~PicktileVkToolSession() = default;

bool PicktileVkToolSession::Fail(std::string const& message) {
    diagnostic_ = message;
    return false;
}

char const* PicktileVkToolSession::diagnostic() const noexcept {
    return diagnostic_.c_str();
}

std::shared_ptr<const ChargedBuffer> PicktileVkToolSession::footprint() const noexcept {
    return footprintBuffer_;
}

std::shared_ptr<PicktileVkToolSession> PicktileVkToolSession::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& pickSpirv,
    std::vector<uint32_t> const& footprintSpirv, std::string* reason) {
    if (!context) {
        if (reason) *reason = "picktile tool session requires a device context";
        return {};
    }
    try {
        auto session = std::shared_ptr<PicktileVkToolSession>(
            new PicktileVkToolSession(context));
        VkResult status = VK_SUCCESS;
        session->pick_ = PicktileVkPickPipeline::Create(context, pickSpirv, &status);
        if (!session->pick_) {
            if (reason) *reason = "picktile tool session cannot create the pick pipeline";
            return {};
        }
        session->footprint_ =
            PicktileVkFootprintPipeline::Create(context, footprintSpirv, &status);
        if (!session->footprint_) {
            if (reason) *reason = "picktile tool session cannot create the footprint pipeline";
            return {};
        }
        return session;
    } catch (std::bad_alloc const&) {
        if (reason) *reason = "picktile tool session allocation failed";
        return {};
    }
}

namespace {
// Blocking bounded wait through the candidate-private fence: poll with a 10s
// deadline (matching the repo readback fixture), sleeping between probes.
template <class Candidate, class Semantic>
VkResult WaitCandidate(Candidate& candidate, Semantic* semantic) {
    auto const deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    VkResult proved = candidate.Poll(semantic);
    while (proved == VK_NOT_READY && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        proved = candidate.Poll(semantic);
    }
    return proved;
}
} // namespace

bool PicktileVkToolSession::Pick(PicktileVkChargedGeometry const& geometry,
    PicktileVkQuery const& query, PicktileVkResult* result, std::string* reason) {
    auto fail = [&](char const* message) {
        if (reason) *reason = message;
        return Fail(message);
    };
    if (!result) return fail("pick result pointer is null");
    VkResult status = VK_SUCCESS;
    auto candidate = pick_->Begin(geometry.points, geometry.offsets, geometry.stableIds,
        geometry.curveCount, geometry.pointCount, query, &status);
    if (!candidate) return fail("pick cannot acquire the Vulkan geometry on this queue");
    PicktileVkPickPipeline::Semantic semantic;
    VkResult const proved = WaitCandidate(*candidate, &semantic);
    if (proved == VK_NOT_READY) {
        candidate->Quarantine();
        return fail("pick device proof timed out");
    }
    if (proved != VK_SUCCESS) return fail("pick device proof failed");
    if (semantic != PicktileVkPickPipeline::Semantic::Ok) {
        if (semantic == PicktileVkPickPipeline::Semantic::NonFiniteInput)
            return fail("pick diagnosed non-finite input");
        return fail("pick diagnosed invalid geometry");
    }
    *result = candidate->result();
    result_ = *result;
    diagnostic_.clear();
    if (reason) reason->clear();
    return true;
}

bool PicktileVkToolSession::Footprint(PicktileVkChargedGeometry const& geometry,
    PicktileVkQuery const& query, std::string* reason) {
    auto fail = [&](char const* message) {
        if (reason) *reason = message;
        return Fail(message);
    };
    VkResult status = VK_SUCCESS;
    auto candidate = footprint_->Begin(geometry.points, geometry.offsets,
        geometry.curveCount, geometry.pointCount, query, &status);
    if (!candidate) return fail("footprint cannot acquire the Vulkan geometry on this queue");
    PicktileVkFootprintPipeline::Semantic semantic;
    VkResult const proved = WaitCandidate(*candidate, &semantic);
    if (proved == VK_NOT_READY) {
        candidate->Quarantine();
        return fail("footprint device proof timed out");
    }
    if (proved != VK_SUCCESS) return fail("footprint device proof failed");
    if (semantic != PicktileVkFootprintPipeline::Semantic::Ok) {
        if (semantic == PicktileVkFootprintPipeline::Semantic::NonFiniteInput)
            return fail("footprint diagnosed non-finite input");
        return fail("footprint diagnosed invalid geometry");
    }
    footprintBuffer_ = candidate->footprint();
    footprintCount_ = candidate->footprintCount();
    if (!footprintBuffer_) return fail("footprint publication failed");
    diagnostic_.clear();
    if (reason) reason->clear();
    return true;
}

} // namespace usdGen::vulkan
