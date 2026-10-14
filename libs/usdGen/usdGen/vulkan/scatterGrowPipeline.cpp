// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "scatterGrowPipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

namespace usdGen::vulkan {
namespace {
constexpr VkDeviceSize kWordBytes = sizeof(uint32_t);
constexpr uint32_t kLocalSize = 256;

// SplitMix64 finalizer (UsdGenDraw01/hash.h) — the CUDA Hash64.
inline uint64_t Hash64(uint64_t key, uint64_t salt) noexcept {
    constexpr uint64_t K = 0x9E3779B97F4A7C15ull;
    uint64_t z = (key ^ (salt * K)) + K;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
constexpr uint32_t kGrowSalt = 0x47726F77u; // "Grow"
constexpr uint32_t kGrowAzimuthSalt = 0x4772417Au; // kSaltGrowAzimuth

// CUDA DrawGrow: salt'd finalizer draw; r is the high 24 bits as [0,1).
inline float DrawGrow(int seed, uint64_t id, uint32_t salt = kGrowSalt) noexcept {
    uint64_t key = Hash64(uint64_t(uint32_t(seed)), salt) ^ uint64_t(id);
    uint32_t hi = uint32_t(Hash64(key, salt) >> 32);
    uint32_t top = (hi >> 8) & 0x00FFFFFFu;
    return float(top) * 0x1.0p-24f;
}

// CUDA/scatterGrow Normalize parity: on a degenerate length, substitute (0,1,0)
// and CONTINUE — never reject.
std::array<float, 3> Normalize3(std::array<float, 3> const& v) noexcept {
    float d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (!(d > 1.0e-24f) || !std::isfinite(d)) return {0.0f, 1.0f, 0.0f};
    const float len = std::sqrt(d);
    return {v[0] / len, v[1] / len, v[2] / len};
}

// CUDA RotateAroundB: axis normalized internally, then Rodrigues.
std::array<float, 3> RotateAroundB(std::array<float, 3> const& p,
                                   std::array<float, 3> const& axis, float deg) noexcept {
    if (deg == 0.0f) return p;
    float a = deg * (3.14159265358979323846f / 180.0f);
    float c = std::cos(a), s = std::sin(a);
    std::array<float, 3> ax = Normalize3(axis);
    std::array<float, 3> out{};
    out[0] = p[0] * c + (ax[1] * p[2] - ax[2] * p[1]) * s +
             ax[0] * (ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2]) * (1.0f - c);
    out[1] = p[1] * c + (ax[2] * p[0] - ax[0] * p[2]) * s +
             ax[1] * (ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2]) * (1.0f - c);
    out[2] = p[2] * c + (ax[0] * p[1] - ax[1] * p[0]) * s +
             ax[2] * (ax[0] * p[0] + ax[1] * p[1] + ax[2] * p[2]) * (1.0f - c);
    return out;
}

void Barrier(VkCommandBuffer c, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
             VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) noexcept {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcAccess, dstAccess};
    vkCmdPipelineBarrier(c, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

struct ScatterPush {
    uint32_t curveCount, pointCount, cvCount, direction;
    float lift, fallbackWidth;
    float litDirX, litDirY, litDirZ;
    float azimuth, azimuthRandom;
    uint32_t seedBits;
};
static_assert(sizeof(ScatterPush) == 48, "scatterGrow.comp push ABI (12 words)");

} // namespace

// CPU oracle — exact CUDA double target math (scatterGrow.cu GrowKernel).
std::vector<float> ScatterGrowPipeline::BuildTargets(
    std::vector<uint64_t> const& stableIds, Controls const& controls) {
    std::vector<float> out;
    out.reserve(stableIds.size());
    double lo = controls.randomLo;
    double spread = controls.randomHi - controls.randomLo;
    for (uint64_t sid : stableIds) {
        float r = DrawGrow(controls.seed, sid);
        out.push_back(float(static_cast<double>(controls.length) *
                            (lo + static_cast<double>(r) * spread)));
    }
    return out;
}

// CPU-fallback geometry producer (host equivalent of scatterGrow.cu GrowKernel).
// Returns false on any semantic rejection (NonFinite), leaving `output`
// untouched — matching the CUDA kNonFinite=2 device status.
bool ScatterGrowPipeline::BuildCpu(std::vector<float> const& positions,
                                   std::vector<uint64_t> const& stableIds,
                                   std::vector<int32_t> const& rootPrim,
                                   std::vector<float> const& rootUV,
                                   std::vector<float> const& rootT,
                                   std::vector<float> const& rootB,
                                   std::vector<float> const& rootN,
                                   std::vector<float> const& targets,
                                   Controls const& controls, Outputs* output) {
    uint32_t cv = controls.cvCount;
    uint32_t curves = static_cast<uint32_t>(stableIds.size());
    auto bad = [&]() { return false; };
    if (curves == 0 || cv < 2 || cv > 64) return bad();
    const std::size_t n = stableIds.size();
    if (positions.size() != n * 3 || targets.size() != n ||
        rootPrim.size() != n || rootUV.size() != n * 2 ||
        rootT.size() != n * 3 || rootB.size() != n * 3 || rootN.size() != n * 3)
        return bad();
    // No whole-plane finite pre-scans: the per-curve body below checks
    // its own t/b/n/uv lanes explicitly, while positions and targets
    // stay covered by the in-loop d/opx checks (k=0 has h=0, so a
    // non-finite target yields d=NaN and a non-finite position yields a
    // non-finite opx). Same verdict, fused with cache-hot curve data.
    Outputs o;
    o.points.resize(3 * n * cv);
    o.rest.resize(3 * n * cv);
    o.widths.resize(n * cv);
    o.hairT.resize(n * cv);
    o.offsets.resize(n + 1);
    o.ids.resize(n);
    o.rootPrim.resize(n);
    o.rootUV.resize(2 * n);
    o.rootT.resize(3 * n);
    o.rootB.resize(3 * n);
    o.rootN.resize(3 * n);

    // The final offset lands up front (bit-identical): it is n * cv
    // regardless of which curve writes it, and failures discard o.
    o.offsets[n] = uint32_t(n * cv);
    // Per-curve body as a range driver (verdict-identical): every curve
    // reads only its own inputs plus shared-immutable controls and writes
    // only its own output lanes, so any curve order -- serial or chunked
    // over workers -- fills identical planes, and any failing curve
    // reports false with *output untouched either way.
    auto runRange = [&](size_t c0, size_t c1) -> bool {
        for (size_t c = c0; c < c1; ++c) {
            o.offsets[c] = uint32_t(c * cv);

            // Direction select + normalize + lift about root B.
            std::array<float, 3> t{rootT[3 * c], rootT[3 * c + 1], rootT[3 * c + 2]};
            std::array<float, 3> b{rootB[3 * c], rootB[3 * c + 1], rootB[3 * c + 2]};
            std::array<float, 3> nn{rootN[3 * c], rootN[3 * c + 1], rootN[3 * c + 2]};
            // Fused finite verdict for the frame + uv lanes (Normalize3
            // would substitute instead of rejecting; uv is only copied).
            if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2]) ||
                !std::isfinite(b[0]) || !std::isfinite(b[1]) || !std::isfinite(b[2]) ||
                !std::isfinite(nn[0]) || !std::isfinite(nn[1]) || !std::isfinite(nn[2]) ||
                !std::isfinite(rootUV[2 * c]) || !std::isfinite(rootUV[2 * c + 1]))
                return false;
            std::array<float, 3> dir;
            switch (controls.direction) {
                case Direction::RootTangent: dir = t; break;
                case Direction::Literal:
                    dir = {controls.literalDirection[0], controls.literalDirection[1],
                           controls.literalDirection[2]};
                    break;
                default: dir = nn; break; // RootNormal
            }
            dir = Normalize3(dir);
            if (controls.lift != 0.0f) dir = RotateAroundB(dir, b, controls.lift);
            // CUDA parity (scatterGrow.cu:106-108): azimuth about root N after lift.
            float const angle = controls.azimuth + controls.azimuthRandom * 360.0f *
                (DrawGrow(controls.seed, stableIds[c], kGrowAzimuthSalt) - 0.5f);
            if (angle != 0.0f) dir = RotateAroundB(dir, nn, angle);
            if (!std::isfinite(dir[0]) || !std::isfinite(dir[1]) || !std::isfinite(dir[2]))
                return false;

            float target = targets[c];
            float px = positions[3 * c], py = positions[3 * c + 1], pz = positions[3 * c + 2];
            uint32_t first = uint32_t(c * cv);
            for (uint32_t k = 0; k < cv; ++k) {
                float h = float(k) / float(cv - 1);
                float d = target * h;
                float opx = px + dir[0] * d, opy = py + dir[1] * d, opz = pz + dir[2] * d;
                if (!std::isfinite(d) || !std::isfinite(opx) || !std::isfinite(opy) ||
                    !std::isfinite(opz))
                    return false;
                uint32_t idx = first + k;
                o.points[3 * idx] = opx; o.points[3 * idx + 1] = opy; o.points[3 * idx + 2] = opz;
                o.rest[3 * idx] = opx; o.rest[3 * idx + 1] = opy; o.rest[3 * idx + 2] = opz;
                o.widths[idx] = controls.fallbackWidth;
                o.hairT[idx] = h;
            }
            o.ids[c] = stableIds[c];
            o.rootPrim[c] = rootPrim[c];
            o.rootUV[2 * c] = rootUV[2 * c];
            o.rootUV[2 * c + 1] = rootUV[2 * c + 1];
            o.rootT[3 * c] = t[0]; o.rootT[3 * c + 1] = t[1]; o.rootT[3 * c + 2] = t[2];
            o.rootB[3 * c] = b[0]; o.rootB[3 * c + 1] = b[1]; o.rootB[3 * c + 2] = b[2];
            o.rootN[3 * c] = nn[0]; o.rootN[3 * c + 1] = nn[1]; o.rootN[3 * c + 2] = nn[2];
        }
        return true;
    };
    // Threaded over curve ranges for big builds (plain TBB: BuildCpu is
    // pure host math with no scheduler context): per-curve work is ~20x
    // a finite scan's few cycles (sqrt + hash + the cv loop), so the
    // dispatch breakeven sits ~20x below the 32K scan rule. Small builds
    // keep the serial driver below.
    int const cpuWorkers = tbb::this_task_arena::max_concurrency();
    size_t const cpuChunks =
        (cpuWorkers > 1 && n > 4096) ? std::min({size_t(cpuWorkers), size_t(8), n})
                                     : 1;
    if (cpuChunks == 1) {
        if (!runRange(0, n))
            return bad();
    } else {
        unsigned char failed[8] = {};
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cpuChunks),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t p = range.begin(); p != range.end(); ++p) {
                    size_t const c0 = (p * n) / cpuChunks;
                    size_t const c1 = ((p + 1) * n) / cpuChunks;
                    if (!runRange(c0, c1))
                        failed[p] = 1;
                }
            });
        for (size_t p = 0; p < cpuChunks; ++p)
            if (failed[p])
                return bad();
    }
    *output = std::move(o);
    return true;
}

// ============================ Device half ============================
namespace {
struct ScatterNative {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~ScatterNative() {
        if (!context) return;
        auto d = context->device();
        if (pipeline) vkDestroyPipeline(d, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(d, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(d, shader, nullptr);
    }
};
} // namespace

struct ScatterGrowPipeline::Native : ScatterNative {};

struct ScatterGrowPipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const void> owner;
    std::shared_ptr<const ChargedBuffer> input;
    std::shared_ptr<ChargedBuffer> points, rest, widths, hairT, offsets, ids,
        rootPrim, rootUV, rootT, rootB, rootN, status;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    uint32_t curveCount = 0, pointCount = 0;
    uint32_t semantic = UINT32_MAX;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};

ScatterGrowPipeline::ScatterGrowPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
ScatterGrowPipeline::~ScatterGrowPipeline() = default;
std::shared_ptr<DeviceContext> const& ScatterGrowPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<ScatterGrowPipeline> ScatterGrowPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256 ||
        physical.limits.maxPerStageDescriptorStorageBuffers < 20 ||
        physical.limits.maxDescriptorSetStorageBuffers < 20 ||
        physical.limits.maxPushConstantsSize < 48) return {};
    try {
        auto n = std::make_shared<Native>();
        n->context = std::move(context);
        auto d = n->context->device();
        VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetLayoutBinding bindings[20]{};
        for (uint32_t i = 0; i != 20; ++i) {
            bindings[i].binding = i; bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 20; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 48};
        VkPipelineLayoutCreateInfo pl{}; pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &n->descriptors;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &push;
        r = vkCreatePipelineLayout(d, &pl, nullptr, &n->layout);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkComputePipelineCreateInfo cp{}; cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = n->layout;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = n->shader; cp.stage.pName = "main";
        r = vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &cp, nullptr, &n->pipeline);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        auto pipeline =
            std::shared_ptr<ScatterGrowPipeline>(new ScatterGrowPipeline(std::move(n)));
        finish(VK_SUCCESS); return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

ScatterGrowPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
ScatterGrowPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }
void ScatterGrowPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool ScatterGrowPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::output() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->points : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rest() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rest : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::widths() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->widths : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::hairT() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->hairT : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::offsets() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->offsets : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::ids() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->ids : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rootPrim() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootPrim : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rootUV() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootUV : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rootT() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootT : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rootB() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootB : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::rootN() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootN : nullptr;
}
std::shared_ptr<const ChargedBuffer> ScatterGrowPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->input : nullptr;
}
uint32_t ScatterGrowPipeline::Candidate::curveCount() const noexcept {
    return state_ ? state_->curveCount : 0;
}
uint32_t ScatterGrowPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}
std::shared_ptr<DeviceContext> ScatterGrowPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult ScatterGrowPipeline::Candidate::Poll(Semantic* semanticStatus) {
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
    if (semanticStatus) *semanticStatus = Semantic(s.semantic);
    return VK_SUCCESS;
}

std::unique_ptr<ScatterGrowPipeline::Candidate> ScatterGrowPipeline::Begin(
    std::shared_ptr<const void> owner, Input const& input, uint32_t curveCount,
    Controls const& controls, VkResult* result, BeforeSubmit beforeSubmit) {
    return BeginInternal(std::move(owner), input, curveCount, controls, result,
                         std::move(beforeSubmit));
}

std::unique_ptr<ScatterGrowPipeline::Candidate> ScatterGrowPipeline::BeginInternal(
    std::shared_ptr<const void> owner, Input const& input, uint32_t curveCount,
    Controls const& controls, VkResult* result, BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;

    // ---- Host validation (CUDA CudaScatterGrow::validate parity) ----
    const uint32_t cv = controls.cvCount;
    if (!owner || cv < 2u || cv > 64u ||
        !std::isfinite(controls.length) || controls.length < 0 ||
        !std::isfinite(controls.randomLo) || controls.randomLo < 0 ||
        !std::isfinite(controls.randomHi) || controls.randomHi < 0 ||
        !std::isfinite(controls.lift) || controls.lift < -90.0f || controls.lift > 90.0f ||
        !std::isfinite(controls.azimuth) || controls.azimuth < -360.0f || controls.azimuth > 360.0f ||
        !std::isfinite(controls.azimuthRandom) || controls.azimuthRandom < 0.0f || controls.azimuthRandom > 1.0f ||
        !std::isfinite(controls.fallbackWidth) || controls.fallbackWidth < 0 ||
        uint32_t(controls.direction) > 2u) return {};
    if (controls.direction == Direction::Literal &&
        (!std::isfinite(controls.literalDirection[0]) ||
         !std::isfinite(controls.literalDirection[1]) ||
         !std::isfinite(controls.literalDirection[2]))) return {};
    // pointCount is derived: curveCount * cv (roots-only contract).
    if (curveCount > std::numeric_limits<uint32_t>::max() / cv) return {};
    const uint32_t pointCount = curveCount * cv;
    const VkDeviceSize idsBytes = VkDeviceSize(curveCount) * 8u;
    const VkDeviceSize rootsBytes = VkDeviceSize(curveCount) * 3u * 4u;
    const VkDeviceSize uvBytes = VkDeviceSize(curveCount) * 2u * 4u;
    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return b && b->context() == context && !b->unproven() && b->buffer() &&
            (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes;
    };
    if (!check(input.positions, rootsBytes) ||
        !check(input.stableIds, idsBytes) ||
        !check(input.rootPrim, VkDeviceSize(curveCount) * 4u) ||
        !check(input.rootUV, uvBytes) ||
        !check(input.rootT, rootsBytes) ||
        !check(input.rootB, rootsBytes) ||
        !check(input.rootN, rootsBytes) ||
        !check(input.targets, VkDeviceSize(curveCount) * 4u)) return {};

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t groups = (uint64_t(curveCount) + 255) / 256;
    if (groups > properties.limits.maxComputeWorkGroupCount[0]) return {};

    try {
        auto s = std::make_shared<Candidate::State>();
        s->native = native_;
        s->owner = std::move(owner);
        s->input = input.positions;
        s->curveCount = curveCount; s->pointCount = pointCount;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        if (!curveCount) { s->proved = true; s->semantic = 0; finish(VK_SUCCESS); return candidate; }
        auto d = context->device();
        VkResult r;
        auto mkOut = [&](VkDeviceSize bytes, std::shared_ptr<ChargedBuffer>* out) -> bool {
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            *out = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         UsdGenExecutionResourceKind::Active, &r);
            return *out != nullptr;
        };
        const VkDeviceSize outPts = VkDeviceSize(pointCount) * 3u * 4u;
        const VkDeviceSize outPer = VkDeviceSize(pointCount) * 4u;
        if (!mkOut(outPts, &s->points) || !mkOut(outPts, &s->rest) ||
            !mkOut(outPer, &s->widths) || !mkOut(outPer, &s->hairT) ||
            !mkOut(VkDeviceSize(curveCount + 1u) * 4u, &s->offsets) ||
            !mkOut(VkDeviceSize(curveCount) * 8u, &s->ids) ||
            !mkOut(VkDeviceSize(curveCount) * 4u, &s->rootPrim) ||
            !mkOut(uvBytes, &s->rootUV) ||
            !mkOut(rootsBytes, &s->rootT) ||
            !mkOut(rootsBytes, &s->rootB) ||
            !mkOut(rootsBytes, &s->rootN)) { finish(r); return {}; }
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = 4; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        s->status = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch, &r);
        if (!s->status) { finish(r); return {}; }
        void* data = nullptr;
        r = vkMapMemory(d, s->status->memory(), 0, 4, 0, &data);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        std::memset(data, 0, 4); vkUnmapMemory(d, s->status->memory());

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20u};
        VkDescriptorPoolCreateInfo dp{}; dp.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets = 1; dp.poolSizeCount = 1; dp.pPoolSizes = &size;
        r = vkCreateDescriptorPool(d, &dp, nullptr, &s->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetAllocateInfo da{}; da.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        auto descriptorLayout = native_->descriptors;
        da.descriptorPool = s->descriptors; da.descriptorSetCount = 1; da.pSetLayouts = &descriptorLayout;
        VkDescriptorSet set;
        r = vkAllocateDescriptorSets(d, &da, &set);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        // 20 bindings (17 planes + status).
        VkDescriptorBufferInfo infos[20] = {
            {input.positions->buffer(), 0, rootsBytes},
            {input.rootT->buffer(), 0, rootsBytes},
            {input.rootB->buffer(), 0, rootsBytes},
            {input.rootN->buffer(), 0, rootsBytes},
            {input.stableIds->buffer(), 0, idsBytes},
            {input.rootPrim->buffer(), 0, VkDeviceSize(curveCount) * 4u},
            {input.rootUV->buffer(), 0, uvBytes},
            {input.targets->buffer(), 0, VkDeviceSize(curveCount) * 4u},
            {s->points->buffer(), 0, outPts},
            {s->rest->buffer(), 0, outPts},
            {s->widths->buffer(), 0, outPer},
            {s->hairT->buffer(), 0, outPer},
            {s->offsets->buffer(), 0, VkDeviceSize(curveCount + 1u) * 4u},
            {s->ids->buffer(), 0, idsBytes},
            {s->rootPrim->buffer(), 0, VkDeviceSize(curveCount) * 4u},
            {s->rootUV->buffer(), 0, uvBytes},
            {s->rootT->buffer(), 0, rootsBytes},
            {s->rootB->buffer(), 0, rootsBytes},
            {s->rootN->buffer(), 0, rootsBytes},
            {s->status->buffer(), 0, 4u}};
        VkWriteDescriptorSet writes[20]{};
        for (uint32_t i = 0; i != 20u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 20, writes, 0, nullptr);

        VkCommandPoolCreateInfo pc{}; pc.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pc.queueFamilyIndex = context->computeQueueFamily();
        r = vkCreateCommandPool(d, &pc, nullptr, &s->commands);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferAllocateInfo ca{}; ca.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool = s->commands; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
        VkCommandBuffer command;
        r = vkAllocateCommandBuffers(d, &ca, &command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r = vkBeginCommandBuffer(command, &begin);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        Barrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        auto pipelineLayout = native_->layout;
        auto pipeline = native_->pipeline;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        ScatterPush push{};
        push.curveCount = curveCount; push.pointCount = pointCount;
        push.cvCount = cv; push.direction = uint32_t(controls.direction);
        push.lift = controls.lift; push.fallbackWidth = controls.fallbackWidth;
        push.litDirX = controls.literalDirection[0];
        push.litDirY = controls.literalDirection[1];
        push.litDirZ = controls.literalDirection[2];
        push.azimuth = controls.azimuth; push.azimuthRandom = controls.azimuthRandom;
        std::memcpy(&push.seedBits, &controls.seed, sizeof(push.seedBits));
        vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 48, &push);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        Barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        r = vkEndCommandBuffer(command);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        r = vkCreateFence(d, &fi, nullptr, &s->fence);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        if (beforeSubmit) {
            bool admitted = false;
            try { admitted = beforeSubmit(); }
            catch (...) { finish(VK_ERROR_UNKNOWN); return {}; }
            if (!admitted) { finish(VK_ERROR_OUT_OF_DEVICE_MEMORY); return {}; }
        }
        s->pending = true;   // arm retention before native submission
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        r = vkQueueSubmit(context->computeQueue(), 1, &submit, s->fence);
        if (r != VK_SUCCESS) { candidate->Quarantine(); finish(r); return {}; }
        finish(VK_SUCCESS); return candidate;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

} // namespace usdGen::vulkan
