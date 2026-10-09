// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "rbfVkDeformPipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>

namespace usdGen::vulkan {

namespace {

constexpr uint32_t kLocalSize = 256;
constexpr uint32_t kApplyUboBytes = 80;
constexpr uint32_t kApplyStorage = 9;

uint32_t Groups(uint32_t n) { return (n + kLocalSize - 1) / kLocalSize; }

struct ApplyUbo {
    uint32_t counts[3];       // uvec3: 12B in std140, offset 0
    float groomEnvelope;      // offset 12
    float maskF[4];          // vec4: 16B in std140, offset 16
    float enabledF[4];       // vec4: offset 32
    float lockF[4];          // vec4: offset 48
    uint32_t fieldCounts[3]; // uvec3: 12B in std140, offset 64
    uint32_t strideCandidate; // offset 76; total 80, 16B-aligned block
};
static_assert(sizeof(ApplyUbo) == 80);
static_assert(offsetof(ApplyUbo, groomEnvelope) == 12);
static_assert(offsetof(ApplyUbo, maskF) == 16);
static_assert(offsetof(ApplyUbo, enabledF) == 32);
static_assert(offsetof(ApplyUbo, lockF) == 48);
static_assert(offsetof(ApplyUbo, fieldCounts) == 64);

DeformSemantic MapBindStatus(RbfVkStatus status) {
    switch (status) {
        case RbfVkStatus::Ok:              return DeformSemantic::Ok;
        case RbfVkStatus::InvalidArgument: return DeformSemantic::BadValue;
        case RbfVkStatus::NonFiniteInput:  return DeformSemantic::NonFinite;
        case RbfVkStatus::RankDeficient:   return DeformSemantic::RankDeficient;
        case RbfVkStatus::DeviceError:     return DeformSemantic::Ok; // reported via VkResult
        case RbfVkStatus::SolverError:     return DeformSemantic::SolverError;
    }
    return DeformSemantic::SolverError;
}

} // namespace

struct RbfVkDeformPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    RbfVkBindingSpirv bindingSpirv; // immutable programs; each candidate owns its solve
    // Rest is static across poses: fresh per-candidate bindings share one
    // factor cache so only the first pose pays the LU submit.
    std::shared_ptr<RbfVkFactorCache> factorCache = std::make_shared<RbfVkFactorCache>();
    // Same for the pose-invariant RBF radii: the CVs and rest are
    // static across poses, so only the first pose pays the sqrt fill.
    std::shared_ptr<RbfVkEvalCache> evalCache = std::make_shared<RbfVkEvalCache>();
    VkShaderModule applyShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout applyLayout = VK_NULL_HANDLE;
    VkPipelineLayout applyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline applyPipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (applyPipeline) vkDestroyPipeline(d, applyPipeline, nullptr);
        if (applyPipelineLayout) vkDestroyPipelineLayout(d, applyPipelineLayout, nullptr);
        if (applyLayout) vkDestroyDescriptorSetLayout(d, applyLayout, nullptr);
        if (applyShader) vkDestroyShaderModule(d, applyShader, nullptr);
    }
};

struct RbfVkDeformPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<RbfVkBinding> binding;
    enum class Phase { Extent, Factor, Pose, Evaluate, Apply };
    Phase phase = Phase::Extent;
    VkCommandBuffer applyCommand = VK_NULL_HANDLE;
    DeformPipeline::BeginInfo inputs;
    std::shared_ptr<ChargedBuffer> outPoints;
    std::shared_ptr<ChargedBuffer> warped;
    std::shared_ptr<ChargedBuffer> status;
    std::shared_ptr<ChargedBuffer> applyUbo;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX;
    uint32_t pointCount = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

RbfVkDeformPipeline::RbfVkDeformPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
RbfVkDeformPipeline::~RbfVkDeformPipeline() = default;
std::shared_ptr<DeviceContext> const& RbfVkDeformPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<RbfVkDeformPipeline> RbfVkDeformPipeline::Create(
    std::shared_ptr<DeviceContext> context, RbfVkBindingSpirv const& bindingSpirv,
    std::vector<uint32_t> const& applySpirv, VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || applySpirv.size() < 5 || applySpirv.front() != 0x07230203u) return {};
    if (!context->shaderFloat64Enabled()) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kLocalSize ||
        physical.limits.maxComputeWorkGroupSize[0] < kLocalSize ||
        physical.limits.maxPerStageDescriptorStorageBuffers < kApplyStorage ||
        physical.limits.maxDescriptorSetStorageBuffers < kApplyStorage)
        return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        VkResult r = VK_SUCCESS;
        auto validationBinding = RbfVkBinding::Create(n->context, bindingSpirv, &r);
        if (!validationBinding) { finish(r); return {}; }
        validationBinding.reset();
        n->bindingSpirv = bindingSpirv;
        VkShaderModuleCreateInfo sm{};
        sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = applySpirv.size() * sizeof(uint32_t);
        sm.pCode = applySpirv.data();
        r = vkCreateShaderModule(d, &sm, nullptr, &n->applyShader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        {
            uint32_t const total = kApplyStorage + 1;
            VkDescriptorSetLayoutBinding bindings[total]{};
            for (uint32_t i = 0; i < total; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                bindings[i].descriptorType = (i == kApplyStorage)
                    ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = total;
            ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->applyLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        {
            VkPipelineLayoutCreateInfo pl{};
            pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount = 1;
            pl.pSetLayouts = &n->applyLayout;
            r = vkCreatePipelineLayout(d, &pl, nullptr, &n->applyPipelineLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        {
            VkComputePipelineCreateInfo cp{};
            cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout = n->applyPipelineLayout;
            cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cp.stage.module = n->applyShader;
            cp.stage.pName = "main";
            r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->applyPipeline);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        auto pipeline = std::shared_ptr<RbfVkDeformPipeline>(new RbfVkDeformPipeline(std::move(n)));
        finish(VK_SUCCESS);
        return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

RbfVkDeformPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
RbfVkDeformPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }

void RbfVkDeformPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}

bool RbfVkDeformPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}

RbfVkDeformPipeline::Candidate::Output RbfVkDeformPipeline::Candidate::output() const noexcept {
    if (!state_ || !state_->proved || state_->semantic != 0) return {};
    return Output{state_->pointCount ? state_->outPoints : nullptr};
}

std::shared_ptr<const ChargedBuffer> RbfVkDeformPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputs.points : nullptr;
}

std::shared_ptr<DeviceContext> RbfVkDeformPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

uint32_t RbfVkDeformPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}

bool RbfVkDeformPipeline::Candidate::done() const noexcept { return state_->proved; }
bool RbfVkDeformPipeline::Candidate::pending() const noexcept { return state_->pending; }

VkResult RbfVkDeformPipeline::Candidate::Poll(DeformSemantic* semantic) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved && s.pending) {
        auto d = s.native->context->device();
        RbfVkStatus status = RbfVkStatus::Ok;
        VkResult r = VK_SUCCESS;
        if (s.phase == State::Phase::Apply) r = vkGetFenceStatus(d, s.fence);
        else if (s.phase == State::Phase::Evaluate) r = s.binding->PollEvaluate(&status);
        else r = s.binding->PollSolve(&status);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        s.pending = false;
        if (status != RbfVkStatus::Ok) {
            s.semantic = uint32_t(MapBindStatus(status));
            s.proved = true;
        } else if (s.phase == State::Phase::Apply) {
            void* data = nullptr;
            r = vkMapMemory(d, s.status->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) return r;
            std::memcpy(&s.semantic, data, 4);
            vkUnmapMemory(d, s.status->memory());
            s.proved = true;
        }
    }
    if (semantic) *semantic = s.proved ? static_cast<DeformSemantic>(s.semantic) : DeformSemantic::Ok;
    return VK_SUCCESS;
}

bool RbfVkDeformPipeline::Candidate::Advance(BeforeSubmit beforeSubmit, VkResult* result) {
    auto& s = *state_;
    auto finish = [&](VkResult r) { if (result) *result = r; return r == VK_SUCCESS; };
    if (s.lost || s.pending) return finish(VK_ERROR_INITIALIZATION_FAILED);
    if (s.proved) return finish(VK_SUCCESS);
    RbfVkStatus status = RbfVkStatus::Ok;
    if (s.phase == State::Phase::Extent) {
        status = s.binding->AdvanceBind(std::move(beforeSubmit));
        s.phase = State::Phase::Factor;
    } else if (s.phase == State::Phase::Factor) {
        status = s.binding->BeginSolve(s.inputs.posedSamples.data(), s.inputs.sampleCount,
                                       std::move(beforeSubmit));
        s.phase = State::Phase::Pose;
    } else {
        if (s.phase == State::Phase::Pose && s.inputs.curveCount == 0) {
            s.semantic = 0;
            s.proved = true;
            return finish(VK_SUCCESS);
        }
        if (s.phase == State::Phase::Pose) {
            status = s.binding->Evaluate(s.inputs.points, s.warped, s.pointCount,
                                         std::move(beforeSubmit));
            s.phase = State::Phase::Evaluate;
        } else if (s.phase == State::Phase::Evaluate) {
            if (beforeSubmit) {
                try { if (!beforeSubmit()) return finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); }
                catch (...) { return finish(VK_ERROR_UNKNOWN); }
            }
            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &s.applyCommand;
            s.pending = true;
            auto r = vkQueueSubmit(s.native->context->computeQueue(), 1, &submit, s.fence);
            s.phase = State::Phase::Apply;
            if (r != VK_SUCCESS) { Quarantine(); return finish(r); }
            return finish(VK_SUCCESS);
        } else return finish(VK_ERROR_INITIALIZATION_FAILED);
    }
    s.pending = s.binding->HasPendingSolve() || s.binding->HasPendingEvaluate();
    if (status == RbfVkStatus::DeviceError) {
        auto r = s.binding->lastResult();
        return finish(r == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : r);
    }
    if (status != RbfVkStatus::Ok) {
        s.semantic = uint32_t(MapBindStatus(status));
        s.proved = true;
    }
    return finish(VK_SUCCESS);
}

std::unique_ptr<RbfVkDeformPipeline::Candidate> RbfVkDeformPipeline::Begin(
    DeformPipeline::BeginInfo info, VkResult* result, DeformSemantic* semantic,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](DeformSemantic code) {
        if (semantic) *semantic = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (semantic) *semantic = DeformSemantic::Ok;
    auto context = info.points ? info.points->context()
                   : (info.curveOffsets ? info.curveOffsets->context() : nullptr);
    if (!context) context = native_->context;
    if (!context || context != native_->context) return {};

    uint32_t const curves = info.curveCount, points = info.pointCount;
    int const n = info.sampleCount;

    // Match the solver's signed-index bound; actual allocation limits are
    // checked by Bind. Counts below CUDA's minimum report InvalidArgument.
    if (n < 4 || n > kRbfVkMaxSamples) return reject(DeformSemantic::BadValue);
    if (info.restSamples.size() != size_t(3) * size_t(n) ||
        info.posedSamples.size() != size_t(3) * size_t(n))
        return reject(DeformSemantic::BadValue);
    for (float v : info.restSamples)
        if (!std::isfinite(v)) return reject(DeformSemantic::NonFinite);
    for (float v : info.posedSamples)
        if (!std::isfinite(v)) return reject(DeformSemantic::NonFinite);
    if (!std::isfinite(info.smoothing) || info.smoothing < 0) return reject(DeformSemantic::BadValue);
    if (!std::isfinite(info.groomEnvelope) || info.groomEnvelope < 0)
        return reject(DeformSemantic::BadValue);

    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    // Per-point apply: one thread per point (plus curve-span
    // validation for thread i < curves), so the dispatch covers
    // whichever domain is larger.
    uint32_t const applyGroups = Groups(std::max(curves, points));
    uint32_t const applyX = std::min(applyGroups, physical.limits.maxComputeWorkGroupCount[0]);
    uint32_t const applyY = applyX ? (applyGroups + applyX - 1) / applyX : 0;
    if (applyY > physical.limits.maxComputeWorkGroupCount[1]) return reject(DeformSemantic::BadValue);
    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return b && b->context() == context && !b->unproven() && b->buffer() &&
               (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes &&
               bytes <= physical.limits.maxStorageBufferRange;
    };
    if (curves == 0 && points != 0) return reject(DeformSemantic::BadOffsets);
    if (curves > 0) {
        if (!check(info.points, VkDeviceSize(points) * 12u) ||
            !check(info.curveOffsets, VkDeviceSize(curves + 1u) * 4u) ||
            !check(info.rootTargets, VkDeviceSize(curves) * 12u))
            return reject(DeformSemantic::BadOffsets);
    } else {
        if (info.points || info.curveOffsets || info.rootTargets)
            return reject(DeformSemantic::BadOffsets);
    }

    // Field validation.
    auto fieldCount = [&](uint32_t domain) -> uint32_t {
        switch (domain) {
            case 1u:       return 1u;
            case 2u:   return curves;
            case 4u:       return points;
            default: return 0;
        }
    };
    auto vScalar = [&](DeformPipeline::ScalarField const& f) {
        if (f.domain != 1u && f.domain != 2u && f.domain != 4u)
            return false;
        if (!f.data) {
            if (f.count != 0) return false;
            if (f.domain != 1u && fieldCount(f.domain) != 0) return false;
            if (!std::isfinite(f.literal)) return false;
            if (f.domain == 1u) {
                if (f.literal < 0.0f || f.literal > 1.0f) return false;
            }
            return true;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    auto vBool = [&](DeformPipeline::BoolField const& f, bool groomOnly, bool allowPrimitive) {
        if (f.domain != 1u && f.domain != 2u && f.domain != 4u)
            return false;
        if (groomOnly && f.domain != 1u) return false;
        if (!allowPrimitive && f.domain == 2u) return false;
        if (f.domain == 4u) return false;
        if (!f.data) {
            if (f.domain == 1u && f.count == 0) return true;
            if (f.domain == 2u && f.count == 0 && curves == 0) return true;
            return false;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    if (!vScalar(info.mask)) return reject(DeformSemantic::BadValue);
    if (!vBool(info.enabled, true, false)) return reject(DeformSemantic::BadValue);
    if (!vBool(info.lockRoots, false, true)) return reject(DeformSemantic::BadValue);

    // Device-resident topology is validated by the apply shader; only its
    // scalar diagnostic is read back before exposing any output.

    // Candidate-local solve state also isolates overlapping Begins.
    VkResult bindingResult = VK_SUCCESS;
    auto binding = RbfVkBinding::Create(context, native_->bindingSpirv, &bindingResult);
    if (!binding) { finish(bindingResult); return {}; }
    binding->SetFactorCache(native_->factorCache);
    binding->SetEvalCache(native_->evalCache);

    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->binding = binding;
        s->inputs = info;
        s->pointCount = points;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        auto d = context->device();
        VkResult r;

        VkDeviceSize const outBytes = points ? VkDeviceSize(points) * 12u : 4u;
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = outBytes;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            s->outPoints = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UsdGenExecutionResourceKind::Active, &r);
            if (!s->outPoints) { finish(r); return {}; }
        }
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            s->status = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &r);
            if (!s->status) { finish(r); return {}; }
            void* data = nullptr;
            r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            std::memset(data, 0, 4);
            vkUnmapMemory(d, s->status->memory());
        }

        // Prepare apply resources before the first accepted device submit.
        if (curves > 0) {
            {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = outBytes;
                bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                s->warped = ChargedBuffer::Create(context, bi,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UsdGenExecutionResourceKind::Active, &r);
                if (!s->warped) { finish(r); return {}; }
            }
            {
                VkBufferCreateInfo bi{};
                bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = kApplyUboBytes;
                bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                s->applyUbo = ChargedBuffer::Create(context, bi,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    UsdGenExecutionResourceKind::Scratch, &r);
                if (!s->applyUbo) { finish(r); return {}; }
                ApplyUbo a = {};
                a.counts[0] = curves;
                a.counts[1] = points;
                a.counts[2] = 0;
                a.groomEnvelope = info.groomEnvelope;
                a.maskF[0] = info.mask.literal;
                a.maskF[1] = info.mask.data ? 1.0f : 0.0f;
                a.maskF[2] = float(info.mask.domain);
                a.enabledF[0] = float(info.enabled.literal);
                a.enabledF[1] = info.enabled.data ? 1.0f : 0.0f;
                a.enabledF[2] = float(info.enabled.domain);
                a.lockF[0] = float(info.lockRoots.literal);
                a.lockF[1] = info.lockRoots.data ? 1.0f : 0.0f;
                a.lockF[2] = float(info.lockRoots.domain);
                a.fieldCounts[0] = info.mask.count;
                a.fieldCounts[1] = info.enabled.count;
                a.fieldCounts[2] = info.lockRoots.count;
                // Uniform-stride candidate for the apply shader's
                // point->curve map (verified against the offsets at
                // every use; 0 disables it). This fill runs only for
                // curves > 0, so the divisor is safe.
                a.strideCandidate =
                    points % curves == 0 ? points / curves : 0;
                void* data = nullptr;
                r = vkMapMemory(d, s->applyUbo->memory(), 0, kApplyUboBytes, 0, &data);
                if (r != VK_SUCCESS) { finish(r); return {}; }
                std::memcpy(data, &a, sizeof(a));
                vkUnmapMemory(d, s->applyUbo->memory());
            }

            VkDescriptorPoolSize sizes[2] = {
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kApplyStorage},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
            };
            VkDescriptorPoolCreateInfo dp{};
            dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dp.maxSets = 1;
            dp.poolSizeCount = 2;
            dp.pPoolSizes = sizes;
            r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkDescriptorSet applySet = VK_NULL_HANDLE;
            {
                VkDescriptorSetAllocateInfo da{};
                da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                da.descriptorPool = s->descriptors;
                da.descriptorSetCount = 1;
                da.pSetLayouts = &native_->applyLayout;
                r = vkAllocateDescriptorSets(d, &da, &applySet);
                if (r != VK_SUCCESS) { finish(r); return {}; }
            }
            VkDeviceSize const pointsBytes = VkDeviceSize(points) * 12u;
            VkDeviceSize const offsetsBytes = VkDeviceSize(curves + 1u) * 4u;
            VkDeviceSize const targetsBytes = VkDeviceSize(curves) * 12u;
            VkDescriptorBufferInfo aInfos[kApplyStorage + 1] = {
                {info.points->buffer(), 0, pointsBytes},
                {s->warped->buffer(), 0, outBytes},
                {info.curveOffsets->buffer(), 0, offsetsBytes},
                {info.rootTargets->buffer(), 0, targetsBytes},
                {s->outPoints->buffer(), 0, outBytes},
                {info.mask.data ? info.mask.data->buffer() : s->status->buffer(), 0,
                    info.mask.data ? std::max<VkDeviceSize>(4, VkDeviceSize(info.mask.count) * 4) : 4},
                {info.enabled.data ? info.enabled.data->buffer() : s->status->buffer(), 0,
                    info.enabled.data ? std::max<VkDeviceSize>(4, VkDeviceSize(info.enabled.count) * 4) : 4},
                {info.lockRoots.data ? info.lockRoots.data->buffer() : s->status->buffer(), 0,
                    info.lockRoots.data ? std::max<VkDeviceSize>(4, VkDeviceSize(info.lockRoots.count) * 4) : 4},
                {s->status->buffer(), 0, 4u},
                {s->applyUbo->buffer(), 0, kApplyUboBytes},
            };
            VkWriteDescriptorSet aWrites[kApplyStorage + 1]{};
            for (uint32_t i = 0; i < kApplyStorage + 1; ++i) {
                aWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                aWrites[i].dstSet = applySet;
                aWrites[i].dstBinding = i;
                aWrites[i].descriptorCount = 1;
                aWrites[i].descriptorType = (i == kApplyStorage) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                aWrites[i].pBufferInfo = &aInfos[i];
            }
            vkUpdateDescriptorSets(d, kApplyStorage + 1, aWrites, 0, nullptr);

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
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            r = vkAllocateCommandBuffers(d, &ca, &cmd);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            r = vkBeginCommandBuffer(cmd, &begin);
            if (r == VK_SUCCESS) {
                vkCmdFillBuffer(cmd, s->outPoints->buffer(), 0, VK_WHOLE_SIZE, 0u);
                VkMemoryBarrier before{};
                before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
                                       VK_ACCESS_SHADER_WRITE_BIT;
                before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, native_->applyPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    native_->applyPipelineLayout, 0, 1, &applySet, 0, nullptr);
                vkCmdDispatch(cmd, applyX, applyY, 1);
                VkMemoryBarrier after{};
                after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                      VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    0, 1, &after, 0, nullptr, 0, nullptr);
                r = vkEndCommandBuffer(cmd);
            }
            if (r != VK_SUCCESS) { finish(r); return {}; }
            VkFenceCreateInfo fi{};
            fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            r = vkCreateFence(d, &fi, nullptr, &s->fence);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            s->applyCommand = cmd;
        }
        auto status = binding->BeginBind(info.restSamples.data(), n, info.smoothing,
                                         std::move(beforeSubmit));
        s->pending = binding->HasPendingSolve();
        if (status == RbfVkStatus::DeviceError) {
            auto r = binding->lastResult();
            finish(r == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : r);
            return {};
        }
        if (status != RbfVkStatus::Ok) return reject(MapBindStatus(status));
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

} // namespace usdGen::vulkan
