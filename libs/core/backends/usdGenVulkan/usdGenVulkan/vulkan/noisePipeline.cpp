// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "noisePipeline.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <new>

namespace usdGen::vulkan {

char const* NoiseSemanticName(NoiseSemantic s) noexcept {
    switch (s) {
        case NoiseSemantic::Ok:         return "Ok";
        case NoiseSemantic::BadOffsets: return "BadOffsets";
        case NoiseSemantic::NonFinite:  return "NonFinite";
        case NoiseSemantic::BadValue:   return "BadValue";
    }
    return "Invalid";
}

namespace {
constexpr uint32_t kLocalSize = 256;
constexpr uint32_t kProfileSize = 257;
// std140 FieldDescs layout (noiseGenerate.comp binding 23): 11 vec4 = 176 B.
constexpr uint32_t kUboBytes = 176;
// Generate set: bindings 0..22 storage, 23 uniform.
constexpr uint32_t kGenerateStorage = 23;
// Validate set: bindings 0..10 storage.
constexpr uint32_t kValidateStorage = 11;

uint32_t Groups(uint32_t n) { return (n + kLocalSize - 1) / kLocalSize; }

struct Push {  // noiseValidate.comp + noiseGenerate.comp (5 x u32, 20 bytes)
    uint32_t curveCount, pointCount, frameCount, offsetsCount, flags;
};

enum : uint32_t {
    kFlagHairT = 1u,
    kFlagFrameIds = 2u,
};

// Domain codes (noiseGenerate.comp constants).
enum : uint32_t {
    kDomainGroom = 1u,
    kDomainPrimitive = 2u,
    kDomainPoint = 4u,
};

// Write a std140 vec4 (4 floats) into a 16-byte slot.
void writeVec4(uint8_t* p, float x, float y, float z) {
    float f4[4] = {x, y, z, 0.0f};
    std::memcpy(p, f4, 16);
}
// Write a std140 ivec4 (4 int32) into a 16-byte slot.
void writeIvec4(uint8_t* p, int32_t x, int32_t y, int32_t z) {
    int32_t i4[4] = {x, y, z, 0};
    std::memcpy(p, i4, 16);
}
} // namespace

struct NoisePipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule validateShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout validateLayout = VK_NULL_HANDLE;
    VkPipelineLayout validatePipelineLayout = VK_NULL_HANDLE;
    VkPipeline validatePipeline = VK_NULL_HANDLE;
    VkShaderModule generateShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout generateLayout = VK_NULL_HANDLE;
    VkPipelineLayout generatePipelineLayout = VK_NULL_HANDLE;
    VkPipeline generatePipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (validatePipeline) vkDestroyPipeline(d, validatePipeline, nullptr);
        if (validatePipelineLayout) vkDestroyPipelineLayout(d, validatePipelineLayout, nullptr);
        if (validateLayout) vkDestroyDescriptorSetLayout(d, validateLayout, nullptr);
        if (validateShader) vkDestroyShaderModule(d, validateShader, nullptr);
        if (generatePipeline) vkDestroyPipeline(d, generatePipeline, nullptr);
        if (generatePipelineLayout) vkDestroyPipelineLayout(d, generatePipelineLayout, nullptr);
        if (generateLayout) vkDestroyDescriptorSetLayout(d, generateLayout, nullptr);
        if (generateShader) vkDestroyShaderModule(d, generateShader, nullptr);
    }
};

struct NoisePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    BeginInfo inputs;
    // Outputs (owned by the candidate).
    std::shared_ptr<ChargedBuffer> outPoints;
    std::shared_ptr<ChargedBuffer> status;
    std::shared_ptr<ChargedBuffer> ubo;
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

NoisePipeline::NoisePipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
NoisePipeline::~NoisePipeline() = default;
std::shared_ptr<DeviceContext> const& NoisePipeline::context() const noexcept {
    return native_->context;
}

// ============================ Create ============================

std::shared_ptr<NoisePipeline> NoisePipeline::Create(
    std::shared_ptr<DeviceContext> context,
    std::vector<uint32_t> const& validateSpirv,
    std::vector<uint32_t> const& generateSpirv,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto valid = [](std::vector<uint32_t> const& c) {
        return c.size() >= 5 && c.front() == 0x07230203u;
    };
    if (!context || !valid(validateSpirv) || !valid(generateSpirv)) return {};
    if (!context->shaderFloat64Enabled()) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kLocalSize ||
        physical.limits.maxComputeWorkGroupSize[0] < kLocalSize ||
        physical.limits.maxPerStageDescriptorStorageBuffers < kGenerateStorage ||
        physical.limits.maxDescriptorSetStorageBuffers < kGenerateStorage) return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        auto mkModule = [&](std::vector<uint32_t> const& code, VkShaderModule* out) {
            VkShaderModuleCreateInfo sm{};
            sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sm.codeSize = code.size() * sizeof(uint32_t);
            sm.pCode = code.data();
            return vkCreateShaderModule(d, &sm, nullptr, out);
        };
        VkResult r = mkModule(validateSpirv, &n->validateShader);
        if (r == VK_SUCCESS) r = mkModule(generateSpirv, &n->generateShader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        {
            VkDescriptorSetLayoutBinding bindings[kValidateStorage]{};
            for (uint32_t i = 0; i < kValidateStorage; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = kValidateStorage; ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->validateLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        {
            uint32_t const total = kGenerateStorage + 1;
            VkDescriptorSetLayoutBinding bindings[total]{};
            for (uint32_t i = 0; i < kGenerateStorage; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            bindings[kGenerateStorage].binding = kGenerateStorage;
            bindings[kGenerateStorage].descriptorCount = 1;
            bindings[kGenerateStorage].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[kGenerateStorage].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = total; ds.pBindings = bindings;
            r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->generateLayout);
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        {
            VkPipelineLayoutCreateInfo pl{};
            pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount = 1; pl.pSetLayouts = &n->validateLayout;
            pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
            r = vkCreatePipelineLayout(d, &pl, nullptr, &n->validatePipelineLayout);
            if (r == VK_SUCCESS) {
                pl.pSetLayouts = &n->generateLayout;
                r = vkCreatePipelineLayout(d, &pl, nullptr, &n->generatePipelineLayout);
            }
            if (r != VK_SUCCESS) { finish(r); return {}; }
        }
        auto mkPipeline = [&](VkShaderModule module, VkPipelineLayout layout, VkPipeline* out) {
            VkComputePipelineCreateInfo cp{};
            cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout = layout;
            cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cp.stage.module = module; cp.stage.pName = "main";
            return vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, out);
        };
        r = mkPipeline(n->validateShader, n->validatePipelineLayout, &n->validatePipeline);
        if (r == VK_SUCCESS)
            r = mkPipeline(n->generateShader, n->generatePipelineLayout, &n->generatePipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto pipeline = std::shared_ptr<NoisePipeline>(new NoisePipeline(std::move(n)));
        finish(VK_SUCCESS);
        return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

// ============================ Candidate ============================

NoisePipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
NoisePipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }

void NoisePipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}

bool NoisePipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}

NoisePipeline::Candidate::Output NoisePipeline::Candidate::output() const noexcept {
    if (!state_ || !state_->proved || state_->semantic != 0) return {};
    return Output{state_->inputs.pointCount ? state_->outPoints : nullptr};
}

std::shared_ptr<const ChargedBuffer> NoisePipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputs.magnitudeProfile : nullptr;
}

std::shared_ptr<DeviceContext> NoisePipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

uint32_t NoisePipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}

VkResult NoisePipeline::Candidate::Poll(NoiseSemantic* semantic) {
    auto& s = *state_;
    if (s.lost) return VK_ERROR_DEVICE_LOST;
    if (!s.proved) {
        auto d = s.native->context->device();
        VkResult r = vkGetFenceStatus(d, s.fence);
        if (r == VK_NOT_READY) return r;
        if (r != VK_SUCCESS) { Quarantine(); return r; }
        void* data = nullptr;
        r = vkMapMemory(d, s.status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.semantic, data, 4);
        vkUnmapMemory(d, s.status->memory());
        s.pending = false;
        s.proved = true;
    }
    if (semantic) *semantic = static_cast<NoiseSemantic>(s.semantic);
    return VK_SUCCESS;
}
// ============================ Begin ============================

std::unique_ptr<NoisePipeline::Candidate> NoisePipeline::Begin(
    BeginInfo info, VkResult* result, NoiseSemantic* semantic, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](NoiseSemantic code) {
        if (semantic) *semantic = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (semantic) *semantic = NoiseSemantic::Ok;
    auto context = info.magnitudeProfile ? info.magnitudeProfile->context() : nullptr;
    if (!context || context != native_->context) return {};

    // Host shape contract mirrors CUDA noise.cu host validation: sizes are
    // element counts; Vulkan planes may carry a larger allocation, so
    // sizeBytes() >= required.
    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return b && b->context() == context && !b->unproven() && b->buffer() &&
            (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes;
    };
    uint32_t const curves = info.curveCount, points = info.pointCount;
    if (curves == 0 && points != 0) return reject(NoiseSemantic::BadOffsets);
    if (curves > 0) {
        if (!check(info.points, VkDeviceSize(points) * 12u) ||
            !check(info.restPoints, VkDeviceSize(points) * 12u) ||
            !check(info.curveOffsets, VkDeviceSize(curves + 1u) * 4u) ||
            !check(info.stableIds, VkDeviceSize(curves) * 8u))
            return reject(NoiseSemantic::BadOffsets);
        if (!info.frameTangent || !info.frameBinormal || !info.frameNormal ||
            info.frameCount == 0)
            return reject(NoiseSemantic::BadValue);
        VkDeviceSize const frameBytes = VkDeviceSize(info.frameCount) * 12u;
        if (!check(info.frameTangent, frameBytes) ||
            !check(info.frameBinormal, frameBytes) ||
            !check(info.frameNormal, frameBytes))
            return reject(NoiseSemantic::BadValue);
        if (info.frameStableIds) {
            if (!check(info.frameStableIds, VkDeviceSize(info.frameCount) * 8u))
                return reject(NoiseSemantic::BadValue);
        } else if (info.frameCount != curves) {
            return reject(NoiseSemantic::BadValue);
        }
    } else {
        if (info.points || info.restPoints || info.stableIds)
            return reject(NoiseSemantic::BadOffsets);
        if (info.frameTangent || info.frameBinormal || info.frameNormal)
            return reject(NoiseSemantic::BadValue);
        if (info.curveOffsets && info.curveOffsets->sizeBytes() > 4u)
            return reject(NoiseSemantic::BadOffsets);
    }
    if (info.hairT && !check(info.hairT, points ? VkDeviceSize(points) * 4u : 4u))
        return reject(NoiseSemantic::BadOffsets);
    if (!check(info.magnitudeProfile, kProfileSize * 4u))
        return reject(NoiseSemantic::BadValue);

    // Field-parameter validation (CUDA noise.cu validateScalar/Int/Bool/Seed).
    auto fieldCount = [&](uint32_t domain) -> uint32_t {
        switch (domain) {
            case kDomainGroom: return 1;
            case kDomainPrimitive: return curves;
            case kDomainPoint: return points;
            default: return 0;
        }
    };
    auto vScalar = [&](ScalarField const& f, float lo, float hi, bool strictLo) {
        if (f.domain != kDomainGroom && f.domain != kDomainPrimitive && f.domain != kDomainPoint)
            return false;
        if (!f.data) {
            if (f.count != 0) return false;
            if (f.domain != kDomainGroom && fieldCount(f.domain) != 0) return false;
            if (!std::isfinite(f.literal)) return false;
            if (f.domain == kDomainGroom) {
                if (strictLo ? f.literal <= lo : f.literal < lo) return false;
                if (f.literal > hi) return false;
            }
            return true;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    auto vInt = [&](IntField const& f, int32_t lo, int32_t hi, bool seed) {
        if (seed) {
            if (f.domain != kDomainGroom && f.domain != kDomainPrimitive) return false;
        } else if (f.domain != kDomainGroom && f.domain != kDomainPrimitive &&
                   f.domain != kDomainPoint) {
            return false;
        }
        if (!f.data) {
            if (f.count != 0) return false;
            if (seed) {
                if (f.domain == kDomainPrimitive && curves != 0) return false;
            } else {
                if (f.domain != kDomainGroom && fieldCount(f.domain) != 0) return false;
                if (f.literal < lo || f.literal > hi) return false;
            }
            return true;
        }
        uint32_t expected = fieldCount(f.domain);
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    auto vBool = [&](BoolField const& f, bool groomOnly, bool allowPrimitive) {
        if (f.domain != kDomainGroom && f.domain != kDomainPrimitive && f.domain != kDomainPoint)
            return false;
        if (groomOnly && f.domain != kDomainGroom) return false;
        if (!allowPrimitive && f.domain == kDomainPrimitive) return false;
        if (f.domain == kDomainPoint) return false;
        if (!f.data) {
            if (f.domain == kDomainGroom && f.count == 0) return true;
            if (f.domain == kDomainPrimitive && f.count == 0 && curves == 0) return true;
            return false;
        }
        uint32_t expected = f.domain == kDomainGroom ? 1 : curves;
        return f.count == expected && check(f.data, VkDeviceSize(expected) * 4u);
    };
    if (!vScalar(info.magnitude, 0.0f, std::numeric_limits<float>::max(), false))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.frequency, 0.0f, std::numeric_limits<float>::max(), true))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.correlation, 0.0f, 1.0f, false))
        return reject(NoiseSemantic::BadValue);
    if (!vInt(info.octaves, 1, 6, false))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.lacunarity, 1.0f, std::numeric_limits<float>::max(), true))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.gain, 0.0f, 1.0f, false))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.preserveLength, 0.0f, 1.0f, false))
        return reject(NoiseSemantic::BadValue);
    if (!vScalar(info.mask, 0.0f, 1.0f, false))
        return reject(NoiseSemantic::BadValue);
    if (!vInt(info.seed, 0, INT32_MAX, true))
        return reject(NoiseSemantic::BadValue);
    if (!vBool(info.enabled, true, false))
        return reject(NoiseSemantic::BadValue);
    if (!vBool(info.cumulative, false, true))
        return reject(NoiseSemantic::BadValue);

    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->inputs = info;
        s->pointCount = points;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        auto d = context->device();
        VkResult r;

        // Output plane: DEVICE_LOCAL, 3*pointCount floats (minimum 4 bytes for
        // the empty case, per the no-zero-size-buffer rule).
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
        // Status: 4-byte host-visible scratch, zeroed on host.
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
        // UBO: 176 bytes, host-filled std140 FieldDescs.
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = kUboBytes;
            bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            s->ubo = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &r);
            if (!s->ubo) { finish(r); return {}; }
            void* data = nullptr;
            r = vkMapMemory(d, s->ubo->memory(), 0, kUboBytes, 0, &data);
            if (r != VK_SUCCESS) { finish(r); return {}; }
            auto* p = static_cast<uint8_t*>(data);
            auto& i = info;
            writeVec4(p + 0,   i.magnitude.literal,    i.magnitude.data ? 1.0f : 0.0f,  float(i.magnitude.domain));
            writeVec4(p + 16,  i.frequency.literal,    i.frequency.data ? 1.0f : 0.0f,  float(i.frequency.domain));
            writeVec4(p + 32,  i.correlation.literal,  i.correlation.data ? 1.0f : 0.0f, float(i.correlation.domain));
            writeIvec4(p + 48, i.octaves.literal,      i.octaves.data ? 1 : 0,          int32_t(i.octaves.domain));
            writeVec4(p + 64,  i.lacunarity.literal,   i.lacunarity.data ? 1.0f : 0.0f, float(i.lacunarity.domain));
            writeVec4(p + 80,  i.gain.literal,         i.gain.data ? 1.0f : 0.0f,       float(i.gain.domain));
            writeVec4(p + 96,  i.preserveLength.literal, i.preserveLength.data ? 1.0f : 0.0f, float(i.preserveLength.domain));
            writeVec4(p + 112, i.mask.literal,         i.mask.data ? 1.0f : 0.0f,       float(i.mask.domain));
            writeVec4(p + 128, float(i.enabled.literal),  i.enabled.data ? 1.0f : 0.0f, float(i.enabled.domain));
            writeVec4(p + 144, float(i.cumulative.literal), i.cumulative.data ? 1.0f : 0.0f, float(i.cumulative.domain));
            writeIvec4(p + 160, i.seed.literal,        i.seed.data ? 1 : 0,            int32_t(i.seed.domain));
            vkUnmapMemory(d, s->ubo->memory());
        }

        VkDescriptorPoolSize sizes[2] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kValidateStorage + kGenerateStorage},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        VkDescriptorPoolCreateInfo dp{};
        dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 2; dp.poolSizeCount = 2; dp.pPoolSizes = sizes;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet* set) {
            VkDescriptorSetAllocateInfo da{};
            da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            da.descriptorPool = s->descriptors;
            da.descriptorSetCount = 1; da.pSetLayouts = &layout;
            return vkAllocateDescriptorSets(d, &da, set);
        };
        VkDescriptorSet validateSet = VK_NULL_HANDLE, generateSet = VK_NULL_HANDLE;
        r = allocSet(native_->validateLayout, &validateSet);
        if (r == VK_SUCCESS) r = allocSet(native_->generateLayout, &generateSet);
        if (r != VK_SUCCESS) { finish(r); return {}; }

        // Optional-plane fallback: 4-byte status buffer (never read when the
        // corresponding count/flag says the plane is absent).
        auto planeInfo = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes)
            -> VkDescriptorBufferInfo {
            if (b && bytes) return {b->buffer(), 0, bytes};
            return {s->status->buffer(), 0, 4u};
        };
        VkDeviceSize const pointsBytes = points ? VkDeviceSize(points) * 12u : 4u;
        VkDeviceSize const offsetsBytes = curves ? VkDeviceSize(curves + 1u) * 4u : 4u;
        VkDeviceSize const idsBytes = curves ? VkDeviceSize(curves) * 8u : 4u;
        VkDeviceSize const hairBytes = points ? VkDeviceSize(points) * 4u : 4u;
        VkDeviceSize const frameBytes = curves ? VkDeviceSize(info.frameCount) * 12u : 4u;
        VkDeviceSize const frameIdsBytes = curves ? VkDeviceSize(info.frameCount) * 8u : 4u;
        VkDescriptorBufferInfo vInfos[11] = {
            planeInfo(info.points, pointsBytes),
            planeInfo(info.restPoints, pointsBytes),
            planeInfo(info.curveOffsets, offsetsBytes),
            planeInfo(info.stableIds, idsBytes),
            planeInfo(info.hairT, hairBytes),
            planeInfo(info.frameTangent, frameBytes),
            planeInfo(info.frameBinormal, frameBytes),
            planeInfo(info.frameNormal, frameBytes),
            planeInfo(info.frameStableIds, frameIdsBytes),
            {info.magnitudeProfile->buffer(), 0, kProfileSize * 4u},
            {s->status->buffer(), 0, 4u},
        };
        VkWriteDescriptorSet vWrites[11]{};
        for (uint32_t i = 0; i != 11u; ++i) {
            vWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            vWrites[i].dstSet = validateSet; vWrites[i].dstBinding = i;
            vWrites[i].descriptorCount = 1;
            vWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            vWrites[i].pBufferInfo = &vInfos[i];
        }
        VkDescriptorBufferInfo gInfos[23]{};
        std::copy_n(vInfos, 11, gInfos);  // 0..10 mirror validate
        gInfos[10] = planeInfo(info.magnitude.data,
            info.magnitude.count ? VkDeviceSize(info.magnitude.count) * 4u : 4u);
        gInfos[11] = planeInfo(info.frequency.data,
            info.frequency.count ? VkDeviceSize(info.frequency.count) * 4u : 4u);
        gInfos[12] = planeInfo(info.correlation.data,
            info.correlation.count ? VkDeviceSize(info.correlation.count) * 4u : 4u);
        gInfos[13] = planeInfo(info.octaves.data,
            info.octaves.count ? VkDeviceSize(info.octaves.count) * 4u : 4u);
        gInfos[14] = planeInfo(info.lacunarity.data,
            info.lacunarity.count ? VkDeviceSize(info.lacunarity.count) * 4u : 4u);
        gInfos[15] = planeInfo(info.gain.data,
            info.gain.count ? VkDeviceSize(info.gain.count) * 4u : 4u);
        gInfos[16] = planeInfo(info.preserveLength.data,
            info.preserveLength.count ? VkDeviceSize(info.preserveLength.count) * 4u : 4u);
        gInfos[17] = planeInfo(info.mask.data,
            info.mask.count ? VkDeviceSize(info.mask.count) * 4u : 4u);
        gInfos[18] = planeInfo(info.enabled.data,
            info.enabled.count ? VkDeviceSize(info.enabled.count) * 4u : 4u);
        gInfos[19] = planeInfo(info.cumulative.data,
            info.cumulative.count ? VkDeviceSize(info.cumulative.count) * 4u : 4u);
        gInfos[20] = planeInfo(info.seed.data,
            info.seed.count ? VkDeviceSize(info.seed.count) * 4u : 4u);
        gInfos[21] = {s->outPoints->buffer(), 0, outBytes};
        gInfos[22] = {s->status->buffer(), 0, 4u};
        VkWriteDescriptorSet gWrites[24]{};
        for (uint32_t i = 0; i != 23u; ++i) {
            gWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            gWrites[i].dstSet = generateSet; gWrites[i].dstBinding = i;
            gWrites[i].descriptorCount = 1;
            gWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            gWrites[i].pBufferInfo = &gInfos[i];
        }
        gWrites[23].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        gWrites[23].dstSet = generateSet; gWrites[23].dstBinding = kGenerateStorage;
        gWrites[23].descriptorCount = 1;
        gWrites[23].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        VkDescriptorBufferInfo uboInfo{s->ubo->buffer(), 0, kUboBytes};
        gWrites[23].pBufferInfo = &uboInfo;
        vkUpdateDescriptorSets(d, 11, vWrites, 0, nullptr);
        vkUpdateDescriptorSets(d, 24, gWrites, 0, nullptr);

        VkCommandPoolCreateInfo pc{};
        pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{};
        ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = curves ? 2 : 1;
        VkCommandBuffer cmds[2];
        r = vkAllocateCommandBuffers(d, &ca, cmds);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        uint32_t const flags =
            (info.hairT ? kFlagHairT : 0u) | (info.frameStableIds ? kFlagFrameIds : 0u);
        uint32_t const offsetsCount =
            curves ? curves + 1u : (info.curveOffsets ? 1u : 0u);

        // cmd1: validate dispatch (synchronous proof inside Begin; CUDA
        // mid-Apply sync parity).
        r = vkBeginCommandBuffer(cmds[0], &begin);
        if (r == VK_SUCCESS) {
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmds[0],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->validatePipeline);
            vkCmdBindDescriptorSets(cmds[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->validatePipelineLayout, 0, 1, &validateSet, 0, nullptr);
            Push push{curves, points, info.frameCount, offsetsCount, flags};
            vkCmdPushConstants(cmds[0], native_->validatePipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            uint32_t const w = std::max({curves, points, kProfileSize});
            vkCmdDispatch(cmds[0], Groups(w), 1, 1);
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cmds[0]);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence proofFence = VK_NULL_HANDLE;
        r = vkCreateFence(d, &fi, nullptr, &proofFence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &cmds[0];
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, proofFence);
        if (r == VK_SUCCESS)
            r = vkWaitForFences(d, 1, &proofFence, VK_TRUE, 10000000000ull);
        uint32_t st = 0;
        if (r == VK_SUCCESS) {
            void* data = nullptr;
            r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
            if (r == VK_SUCCESS) {
                std::memcpy(&st, data, 4);
                vkUnmapMemory(d, s->status->memory());
            }
        }
        vkDestroyFence(d, proofFence, nullptr);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (st != 0) return reject(static_cast<NoiseSemantic>(st));

        // Empty topology: validate already proved the profile/offset; no
        // generate dispatch needed.
        if (curves == 0) {
            s->proved = true;
            s->semantic = 0;
            finish(VK_SUCCESS);
            return candidate;
        }

        // cmd2: zero-fill output + generate dispatch (async on candidate fence).
        r = vkBeginCommandBuffer(cmds[1], &begin);
        if (r == VK_SUCCESS) {
            vkCmdFillBuffer(cmds[1], s->outPoints->buffer(), 0, VK_WHOLE_SIZE, 0u);
            VkMemoryBarrier before{};
            before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmds[1],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(cmds[1], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->generatePipeline);
            vkCmdBindDescriptorSets(cmds[1], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->generatePipelineLayout, 0, 1, &generateSet, 0, nullptr);
            Push push{curves, points, info.frameCount, curves + 1u, flags};
            vkCmdPushConstants(cmds[1], native_->generatePipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(cmds[1], Groups(curves), 1, 1);
            VkMemoryBarrier after{};
            after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmds[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(cmds[1]);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try { admitted = beforeSubmit(); }
            catch (...) { finish(VK_ERROR_UNKNOWN); return {}; }
            if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        }
        s->pending = true;
        submit.pCommandBuffers = &cmds[1];
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS);
        return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

} // namespace usdGen::vulkan
