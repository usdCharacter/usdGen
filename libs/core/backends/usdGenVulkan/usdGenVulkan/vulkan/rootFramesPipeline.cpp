// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Vulkan root-frame pipeline (rootFramesValidate.comp + rootFrames.comp).
//
// Two-submit Begin (CUDA ValidateInput mid-Apply-sync parity):
//   cmd1: host->compute barrier, validate dispatch (rootFramesValidate.comp,
//         bindings 0..6, push {curves,faceCount,vertexCount,indexCount,
//         normalsCount,normalDomain}), compute->host barrier; submitted with a
//         private fence, waited, status read back INSIDE Begin. Any nonzero
//         status rejects the Begin with beginStatus and NO outputs.
//   cmd2: fill outputs/badRoot zero, build dispatch (rootFrames.comp,
//         bindings 0..13, push {curves,faceCount,vertexCount,indexCount,
//         normalDomain,hasNormals}); submitted asynchronously on the
//         candidate fence. Poll reads status + badRootCount.
//
// Authored-frames path (authoredFrames non-null): no dispatch at all. The
// host validates every matrix with the verbatim CUDA ValidAuthoredFrame port
// (full 16-double checks) and fills the output planes directly.

#include "rootFramesPipeline.h"

#include "chargedBuffer.h"
#include "deviceContext.h"
#include "usdGen/executionResources.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

namespace usdGen::vulkan {
namespace {

constexpr uint32_t kLocalSize = 256;
constexpr double kAuthoredTolerance = 1e-8;

// ---- ValidAuthoredFrame (CUDA host/device port, verbatim arithmetic) -------
// Row-major 16 doubles: row0 = T, row1 = B, row2 = N, row3 = origin/1.
// m[3]==m[7]==m[11]==0, m[15]==1 within 1e-8; rows unit and mutually
// orthogonal; right-handed: dot(cross(T,B),N) >= 1-tol; origin finite.
struct AuthoredFrame {
    float t[3], b[3], n[3], o[3];
};

bool ValidAuthoredFrame(double const* m, AuthoredFrame* out) {
    constexpr double eps = kAuthoredTolerance;
    for (int i = 0; i != 16; ++i)
        if (!std::isfinite(m[i])) return false;
    if (std::abs(m[3]) > eps || std::abs(m[7]) > eps || std::abs(m[11]) > eps)
        return false;
    if (std::abs(m[15] - 1.0) > eps) return false;
    const double t0 = m[0], t1 = m[1], t2 = m[2];
    const double b0 = m[4], b1 = m[5], b2 = m[6];
    const double n0 = m[8], n1 = m[9], n2 = m[10];
    const double ox = m[12], oy = m[13], oz = m[14];
    const double lt2 = t0 * t0 + t1 * t1 + t2 * t2;
    const double lb2 = b0 * b0 + b1 * b1 + b2 * b2;
    const double ln2 = n0 * n0 + n1 * n1 + n2 * n2;
    if (std::abs(lt2 - 1.0) > eps || std::abs(lb2 - 1.0) > eps ||
        std::abs(ln2 - 1.0) > eps)
        return false;
    if (std::abs(t0 * b0 + t1 * b1 + t2 * b2) > eps ||
        std::abs(t0 * n0 + t1 * n1 + t2 * n2) > eps ||
        std::abs(b0 * n0 + b1 * n1 + b2 * n2) > eps)
        return false;
    if (!std::isfinite(ox) || !std::isfinite(oy) || !std::isfinite(oz))
        return false;
    const double cx = t1 * b2 - t2 * b1;
    const double cy = t2 * b0 - t0 * b2;
    const double cz = t0 * b1 - t1 * b0;
    if (cx * n0 + cy * n1 + cz * n2 < 1.0 - eps) return false;
    if (out) {
        out->t[0] = float(t0); out->t[1] = float(t1); out->t[2] = float(t2);
        out->b[0] = float(b0); out->b[1] = float(b1); out->b[2] = float(b2);
        out->n[0] = float(n0); out->n[1] = float(n1); out->n[2] = float(n2);
        out->o[0] = float(ox); out->o[1] = float(oy); out->o[2] = float(oz);
    }
    return true;
}

// Normals-plane element count per domain (CUDA basis sizing). None is host-
// forced whenever the normals plane is absent.
uint32_t NormalsCount(RootFrameNormalDomain domain, uint32_t faceCount,
                      uint32_t vertexCount, uint32_t indexCount) {
    switch (domain) {
        case RootFrameNormalDomain::Constant: return 1;
        case RootFrameNormalDomain::Uniform: return faceCount;
        case RootFrameNormalDomain::Vertex: return vertexCount;
        case RootFrameNormalDomain::FaceVarying: return indexCount;
        case RootFrameNormalDomain::None: break;
    }
    return 0;
}

uint32_t Groups(uint32_t n) { return (n + kLocalSize - 1) / kLocalSize; }

struct PushValidate {  // rootFramesValidate.comp (6 x u32, std430)
    uint32_t curves, faceCount, vertexCount, indexCount, normalsCount, normalDomain;
};
static_assert(sizeof(PushValidate) == 24, "validate push ABI");

struct PushBuild {  // rootFrames.comp (6 x u32, std430)
    uint32_t curves, faceCount, vertexCount, indexCount, normalDomain, hasNormals;
};
static_assert(sizeof(PushBuild) == 24, "build push ABI");

} // namespace

char const* RootFramesBeginStatusName(RootFramesBeginStatus s) noexcept {
    switch (s) {
        case RootFramesBeginStatus::Ok: return "Ok";
        case RootFramesBeginStatus::BadShape: return "BadShape";
        case RootFramesBeginStatus::NonFinite: return "NonFinite";
        case RootFramesBeginStatus::BadTopology: return "BadTopology";
        case RootFramesBeginStatus::BadRoot: return "BadRoot";
        case RootFramesBeginStatus::BadAuthoredFrame: return "BadAuthoredFrame";
    }
    return "Unknown";
}

struct RootFramesPipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule validateShader = VK_NULL_HANDLE, buildShader = VK_NULL_HANDLE;
    VkDescriptorSetLayout validateLayout = VK_NULL_HANDLE, buildLayout = VK_NULL_HANDLE;
    VkPipelineLayout validatePipelineLayout = VK_NULL_HANDLE, buildPipelineLayout = VK_NULL_HANDLE;
    VkPipeline validatePipeline = VK_NULL_HANDLE, buildPipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (validatePipeline) vkDestroyPipeline(d, validatePipeline, nullptr);
        if (buildPipeline) vkDestroyPipeline(d, buildPipeline, nullptr);
        if (validatePipelineLayout) vkDestroyPipelineLayout(d, validatePipelineLayout, nullptr);
        if (buildPipelineLayout) vkDestroyPipelineLayout(d, buildPipelineLayout, nullptr);
        if (validateLayout) vkDestroyDescriptorSetLayout(d, validateLayout, nullptr);
        if (buildLayout) vkDestroyDescriptorSetLayout(d, buildLayout, nullptr);
        if (validateShader) vkDestroyShaderModule(d, validateShader, nullptr);
        if (buildShader) vkDestroyShaderModule(d, buildShader, nullptr);
    }
};

RootFramesPipeline::RootFramesPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
RootFramesPipeline::~RootFramesPipeline() = default;
std::shared_ptr<DeviceContext> const& RootFramesPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<RootFramesPipeline> RootFramesPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& validateSpirv,
    std::vector<uint32_t> const& buildSpirv, VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto valid = [](std::vector<uint32_t> const& c) {
        return c.size() >= 5 && c.front() == 0x07230203u;
    };
    if (!context || !valid(validateSpirv) || !valid(buildSpirv)) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < kLocalSize ||
        physical.limits.maxComputeWorkGroupSize[0] < kLocalSize ||
        physical.limits.maxPerStageDescriptorStorageBuffers < 14 ||
        physical.limits.maxDescriptorSetStorageBuffers < 14 ||
        physical.limits.maxPushConstantsSize < 24) return {};
    try {
        auto n = std::make_shared<Native>(); n->context = std::move(context);
        auto d = n->context->device();
        auto mkModule = [&](std::vector<uint32_t> const& code, VkShaderModule* out) {
            VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
            return vkCreateShaderModule(d, &sm, nullptr, out);
        };
        VkResult r = mkModule(validateSpirv, &n->validateShader);
        if (r == VK_SUCCESS) r = mkModule(buildSpirv, &n->buildShader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto mkLayout = [&](uint32_t count, VkDescriptorSetLayout* out) {
            VkDescriptorSetLayoutBinding bindings[14]{};
            for (uint32_t i = 0; i != count; ++i) {
                bindings[i].binding = i; bindings[i].descriptorCount = 1;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount = count; ds.pBindings = bindings;
            return vkCreateDescriptorSetLayout(d, &ds, nullptr, out);
        };
        r = mkLayout(7, &n->validateLayout);
        if (r == VK_SUCCESS) r = mkLayout(14, &n->buildLayout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
        auto mkPipelineLayout = [&](VkDescriptorSetLayout set, VkPipelineLayout* out) {
            VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount = 1; pl.pSetLayouts = &set;
            pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
            return vkCreatePipelineLayout(d, &pl, nullptr, out);
        };
        r = mkPipelineLayout(n->validateLayout, &n->validatePipelineLayout);
        if (r == VK_SUCCESS) r = mkPipelineLayout(n->buildLayout, &n->buildPipelineLayout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto mkPipeline = [&](VkShaderModule module, VkPipelineLayout layout, VkPipeline* out) {
            VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout = layout;
            cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = module;
            cp.stage.pName = "main";
            return vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, out);
        };
        r = mkPipeline(n->validateShader, n->validatePipelineLayout, &n->validatePipeline);
        if (r == VK_SUCCESS) r = mkPipeline(n->buildShader, n->buildPipelineLayout, &n->buildPipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto pipeline = std::shared_ptr<RootFramesPipeline>(new RootFramesPipeline(std::move(n)));
        finish(VK_SUCCESS); return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

// ============================ Candidate ============================

struct RootFramesPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    // Every borrowed input pinned for the candidate's lifetime.
    BeginInfo inputs;
    std::shared_ptr<ChargedBuffer> origin, tangent, binormal, normal, valid, drop;
    std::shared_ptr<ChargedBuffer> status, badRoot;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE, proofFence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t semantic = UINT32_MAX, badRootCount = 0, curves = 0;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (proofFence) vkDestroyFence(d, proofFence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

RootFramesPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
RootFramesPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }

void RootFramesPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}

bool RootFramesPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}

RootFramesPipeline::Candidate::Output RootFramesPipeline::Candidate::output() const noexcept {
    if (!state_ || !state_->proved || state_->semantic != 0) return {};
    return Output{state_->origin, state_->tangent, state_->binormal,
                  state_->normal, state_->valid, state_->drop};
}

std::shared_ptr<const ChargedBuffer> RootFramesPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->inputs.restVerts : nullptr;
}
uint32_t RootFramesPipeline::Candidate::count() const noexcept {
    return state_ ? state_->curves : 0;
}
std::shared_ptr<DeviceContext> RootFramesPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult RootFramesPipeline::Candidate::Poll(uint32_t* semanticStatus, uint32_t* badRootCount) {
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
        r = vkMapMemory(d, s.badRoot->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) return r;
        std::memcpy(&s.badRootCount, data, 4);
        vkUnmapMemory(d, s.badRoot->memory());
        s.pending = false;
        s.proved = true;
    }
    if (semanticStatus) *semanticStatus = s.semantic;
    if (badRootCount) *badRootCount = s.badRootCount;
    return VK_SUCCESS;
}

// ============================ Begin ============================

std::unique_ptr<RootFramesPipeline::Candidate> RootFramesPipeline::Begin(
    BeginInfo info, VkResult* result, RootFramesBeginStatus* beginStatus,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    auto reject = [&](RootFramesBeginStatus code) {
        if (beginStatus) *beginStatus = code;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        return std::unique_ptr<Candidate>{};
    };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (beginStatus) *beginStatus = RootFramesBeginStatus::Ok;
    auto context = info.restVerts ? info.restVerts->context() : nullptr;
    if (!context || context != native_->context) return {};

    // ---- Host shape contract (buffer-size contracts per shader header) ----
    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return b && b->context() == context && !b->unproven() && b->buffer() &&
            (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes;
    };
    const uint32_t curves = info.curves, faceCount = info.faceCount;
    const uint32_t vertexCount = info.vertexCount, indexCount = info.indexCount;
    if (curves == 0 && (info.authoredFrameCount != 0 || vertexCount != 0 ||
        faceCount != 0 || indexCount != 0)) return {};
    // Authored contract: non-null pointer requires exactly curves*16 doubles;
    // null pointer requires zero count. (Device-path builds carry 0/nullptr.)
    if (info.authoredFrames && info.authoredFrameCount != uint32_t(curves) * 16u) return {};
    if (!info.authoredFrames && info.authoredFrameCount != 0) return {};
    if (!check(info.restVerts, VkDeviceSize(vertexCount) * 12u)) return {};
    // Offsets plane required only when faces exist; faceCount == 0 performs no
    // offsets reads (validate shader guards its thread-0 structural reads too).
    if (faceCount && !check(info.faceOffsets, VkDeviceSize(faceCount + 1u) * 4u)) return {};
    if (indexCount && !check(info.faceIndices, VkDeviceSize(indexCount) * 4u)) return {};
    if (curves && (!check(info.rootPrim, VkDeviceSize(curves) * 4u) ||
        !check(info.rootUV, VkDeviceSize(curves) * 8u))) return {};
    RootFrameNormalDomain domain = info.normalDomain;
    if (!info.normals) domain = RootFrameNormalDomain::None;
    else if (uint32_t(domain) > 4u) return {};
    const uint32_t normalsCount = NormalsCount(domain, faceCount, vertexCount, indexCount);
    if (normalsCount && !check(info.normals, VkDeviceSize(normalsCount) * 12u)) return {};

    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_; s->inputs = info; s->curves = curves;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        if (!curves) { s->proved = true; s->semantic = 0; s->badRootCount = 0;
            finish(VK_SUCCESS); return candidate; }
        auto d = context->device();
        VkResult r;
        const bool deviceLocalOut = !info.authoredFrames;
        auto mkOut = [&](VkDeviceSize bytes, std::shared_ptr<ChargedBuffer>* out) -> bool {
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            // TSRC per width convention: test readback copies these planes out.
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            // Authored outputs are host-filled and stay host-visible; device
            // outputs are DEVICE_LOCAL like every other build plane.
            *out = ChargedBuffer::Create(context, bi,
                deviceLocalOut
                    ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                    : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
                UsdGenExecutionResourceKind::Active, &r);
            return *out != nullptr;
        };
        const VkDeviceSize plane = VkDeviceSize(curves) * 12u;
        const VkDeviceSize lane = VkDeviceSize(curves) * 4u;
        if (!mkOut(plane, &s->origin) || !mkOut(plane, &s->tangent) ||
            !mkOut(plane, &s->binormal) || !mkOut(plane, &s->normal) ||
            !mkOut(lane, &s->valid) || !mkOut(lane, &s->drop)) { finish(r); return {}; }
        auto mkStatus = [&](std::shared_ptr<ChargedBuffer>* out) -> bool {
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            *out = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &r);
            if (!*out) return false;
            void* data = nullptr;
            r = vkMapMemory(d, (*out)->memory(), 0, 4, 0, &data);
            if (r != VK_SUCCESS) return false;
            std::memset(data, 0, 4); vkUnmapMemory(d, (*out)->memory());
            return true;
        };
        if (!mkStatus(&s->status) || !mkStatus(&s->badRoot)) { finish(r); return {}; }

        // ---------------- Authored path: host fill, no dispatch ----------------
        if (info.authoredFrames) {
            // Plane mapping (rootFrames.comp): 6 origin, 7 tangent, 8 binormal,
            // 9 normal. Authored rows are T,B,N with origin in row 3.
            std::shared_ptr<ChargedBuffer>* planes[4] = {
                &s->origin, &s->tangent, &s->binormal, &s->normal};
            for (uint32_t c = 0; c != curves; ++c) {
                double const* matrix = info.authoredFrames + std::size_t(c) * 16u;
                for (int i = 0; i != 16; ++i)
                    if (!std::isfinite(matrix[i]))
                        return reject(RootFramesBeginStatus::NonFinite);
                AuthoredFrame f;
                if (!ValidAuthoredFrame(matrix, &f))
                    return reject(RootFramesBeginStatus::BadAuthoredFrame);
                // Plane mapping (rootFrames.comp): 6 origin, 7 tangent,
                // 8 binormal, 9 normal; authored rows are o,T,B,N.
                float const* planeRow[4] = {f.o, f.t, f.b, f.n};
                for (int p = 0; p != 4; ++p) {
                    void* data = nullptr;
                    r = vkMapMemory(d, (*planes[p])->memory(), sizeof(float) * 3u * c,
                        sizeof(float) * 3u, 0, &data);
                    if (r != VK_SUCCESS) { finish(r); return {}; }
                    std::memcpy(data, planeRow[p], sizeof(float) * 3u);
                    vkUnmapMemory(d, (*planes[p])->memory());
                }
                void* data = nullptr;
                const uint32_t one = 1, zero = 0;
                r = vkMapMemory(d, s->valid->memory(), sizeof(uint32_t) * c, 4, 0, &data);
                if (r != VK_SUCCESS) { finish(r); return {}; }
                std::memcpy(data, &one, 4); vkUnmapMemory(d, s->valid->memory());
                r = vkMapMemory(d, s->drop->memory(), sizeof(uint32_t) * c, 4, 0, &data);
                if (r != VK_SUCCESS) { finish(r); return {}; }
                std::memcpy(data, &zero, 4); vkUnmapMemory(d, s->drop->memory());
            }
            s->proved = true; s->semantic = 0; s->badRootCount = 0;
            finish(VK_SUCCESS); return candidate;
        }

        // ---------------- Device path: two-submit ----------------
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
        uint64_t const W = std::max({uint32_t(curves), faceCount, vertexCount,
                                     indexCount, normalsCount});
        uint64_t const validateGroups = Groups(uint32_t(W));
        if (validateGroups > properties.limits.maxComputeWorkGroupCount[0] ||
            uint64_t(Groups(curves)) > properties.limits.maxComputeWorkGroupCount[0])
            return {};

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 21u};
        VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 2; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto allocSet = [&](VkDescriptorSetLayout layout, VkDescriptorSet* set) {
            VkDescriptorSetAllocateInfo da{};
            da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            da.descriptorPool = s->descriptors; da.descriptorSetCount = 1; da.pSetLayouts = &layout;
            return vkAllocateDescriptorSets(d, &da, set);
        };
        VkDescriptorSet validateSet = VK_NULL_HANDLE, buildSet = VK_NULL_HANDLE;
        r = allocSet(native_->validateLayout, &validateSet);
        if (r == VK_SUCCESS) r = allocSet(native_->buildLayout, &buildSet);
        if (r != VK_SUCCESS) { finish(r); return {}; }

        // Optional-plane fallback: bind the zero-filled status buffer (4 bytes).
        auto planeBuffer = [&](std::shared_ptr<ChargedBuffer const> const& b,
                               VkDeviceSize bytes) -> VkDescriptorBufferInfo {
            if (b && bytes) return {b->buffer(), 0, bytes};
            return {s->status->buffer(), 0, 4u};
        };
        const VkDeviceSize restBytes = VkDeviceSize(vertexCount) * 12u;
        const VkDeviceSize offsetsBytes = VkDeviceSize(faceCount + 1u) * 4u;
        const VkDeviceSize indicesBytes = VkDeviceSize(indexCount) * 4u;
        const VkDeviceSize primBytes = VkDeviceSize(curves) * 4u;
        const VkDeviceSize uvBytes = VkDeviceSize(curves) * 8u;
        const VkDeviceSize normalsBytes = VkDeviceSize(normalsCount) * 12u;
        VkDescriptorBufferInfo validateInfos[7] = {
            planeBuffer(info.restVerts, restBytes), planeBuffer(info.faceOffsets, offsetsBytes),
            planeBuffer(info.faceIndices, indicesBytes), planeBuffer(info.rootPrim, primBytes),
            planeBuffer(info.rootUV, uvBytes), planeBuffer(info.normals, normalsBytes),
            {s->status->buffer(), 0, 4u}};
        VkWriteDescriptorSet validateWrites[7]{};
        for (uint32_t i = 0; i != 7u; ++i) {
            validateWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            validateWrites[i].dstSet = validateSet; validateWrites[i].dstBinding = i;
            validateWrites[i].descriptorCount = 1;
            validateWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            validateWrites[i].pBufferInfo = &validateInfos[i];
        }
        VkDescriptorBufferInfo buildInfos[14]{};
        std::copy_n(validateInfos, 7, buildInfos);  // 0..6 identical
        buildInfos[6] = {s->origin->buffer(), 0, plane};
        buildInfos[7] = {s->tangent->buffer(), 0, plane};
        buildInfos[8] = {s->binormal->buffer(), 0, plane};
        buildInfos[9] = {s->normal->buffer(), 0, plane};
        buildInfos[10] = {s->valid->buffer(), 0, lane};
        buildInfos[11] = {s->drop->buffer(), 0, lane};
        buildInfos[12] = {s->badRoot->buffer(), 0, 4u};
        // binding 13 = shared status buffer. It must be written EXPLICITLY:
        // the copy_n above only fills [0..6], so [13] starts as a zero
        // descriptor and a null-buffer write poisons the whole set.
        buildInfos[13] = {s->status->buffer(), 0, 4u};
        VkWriteDescriptorSet buildWrites[14]{};
        for (uint32_t i = 0; i != 14u; ++i) {
            buildWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            buildWrites[i].dstSet = buildSet; buildWrites[i].dstBinding = i;
            buildWrites[i].descriptorCount = 1;
            buildWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            buildWrites[i].pBufferInfo = &buildInfos[i];
        }
        vkUpdateDescriptorSets(d, 7, validateWrites, 0, nullptr);
        vkUpdateDescriptorSets(d, 14, buildWrites, 0, nullptr);

        VkCommandPoolCreateInfo pc{}; pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ca.commandBufferCount = 2;
        VkCommandBuffer commands[2];
        r = vkAllocateCommandBuffers(d, &ca, commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        // cmd1: validate dispatch (synchronous proof inside Begin).
        r = vkBeginCommandBuffer(commands[0], &begin);
        if (r == VK_SUCCESS) {
            VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands[0],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(commands[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->validatePipeline);
            vkCmdBindDescriptorSets(commands[0], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->validatePipelineLayout, 0, 1, &validateSet, 0, nullptr);
            PushValidate push{curves, faceCount, vertexCount, indexCount, normalsCount,
                              uint32_t(domain)};
            vkCmdPushConstants(commands[0], native_->validatePipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(commands[0], uint32_t(validateGroups), 1, 1);
            VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commands[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(commands[0]);
        }
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->proofFence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &commands[0];
        s->pending = true; // proof owns the same allocations as the final submit
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->proofFence);
        if (r == VK_SUCCESS)
            r = vkWaitForFences(d, 1, &s->proofFence, VK_TRUE, 10000000000ull);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        s->pending = false;
        uint32_t semantic = 0;
        if (r == VK_SUCCESS) {
            void* data = nullptr;
            r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
            if (r == VK_SUCCESS) {
                std::memcpy(&semantic, data, 4);
                vkUnmapMemory(d, s->status->memory());
            }
        }
        vkDestroyFence(d, s->proofFence, nullptr);
        s->proofFence = VK_NULL_HANDLE;
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (semantic != 0) return reject(RootFramesBeginStatus(semantic));

        // cmd2: zero-fill outputs + badRoot, then the build dispatch.
        r = vkBeginCommandBuffer(commands[1], &begin);
        if (r == VK_SUCCESS) {
            std::shared_ptr<ChargedBuffer>* outs[6] = {
                &s->origin, &s->tangent, &s->binormal, &s->normal, &s->valid, &s->drop};
            for (int p = 0; p != 6; ++p)
                vkCmdFillBuffer(commands[1], (*outs[p])->buffer(), 0, VK_WHOLE_SIZE, 0u);
            vkCmdFillBuffer(commands[1], s->badRoot->buffer(), 0, VK_WHOLE_SIZE, 0u);
            VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(commands[1],
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            vkCmdBindPipeline(commands[1], VK_PIPELINE_BIND_POINT_COMPUTE, native_->buildPipeline);
            vkCmdBindDescriptorSets(commands[1], VK_PIPELINE_BIND_POINT_COMPUTE,
                native_->buildPipelineLayout, 0, 1, &buildSet, 0, nullptr);
            PushBuild push{curves, faceCount, vertexCount, indexCount, uint32_t(domain),
                           info.normals && domain != RootFrameNormalDomain::None ? 1u : 0u};
            vkCmdPushConstants(commands[1], native_->buildPipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(commands[1], Groups(curves), 1, 1);
            VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(commands[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
            r = vkEndCommandBuffer(commands[1]);
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
        s->pending = true;  // arm retention before native submission
        submit.pCommandBuffers = &commands[1];
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS); return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

} // namespace usdGen::vulkan
