// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "curveGrowPipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

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

// bool, in the float sense the shader uses (0.0f/1.0f packed plane).
uint32_t BoolPlane(bool v) noexcept { return v ? 1u : 0u; }

struct Nav {
    std::array<float, 3> T{}, B{}, N{};
    float B0 = 0.0f; // original |B|
};

bool Normalize3(std::array<float, 3>& v) noexcept {
    // Shader/CUDA normalize3 parity: on a degenerate (non-positive-finite)
    // length, substitute (0,1,0) and CONTINUE — never reject.
    float d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (!(d > 1.0e-24f) || !std::isfinite(d)) {
        v = {0.0f, 1.0f, 0.0f};
        return true;
    }
    // Bit-exact CUDA parity (curveGrow.cu:34-37): length = sqrt(l2), then
    // component / length. Reciprocal-multiply differs in the last ulp.
    const float len = std::sqrt(d);
    v[0] /= len; v[1] /= len; v[2] /= len;
    return true;
}

// Rodrigues rotation of p about axis (assumed unit) by angle degrees.
std::array<float, 3> RotateAroundB(std::array<float, 3> const& p,
    std::array<float, 3> const& axis, float deg) noexcept {
    if (deg == 0.0f) return p;
    float a = deg * (3.14159265358979323846f / 180.0f);
    float c = std::cos(a), s = std::sin(a);
    std::array<float, 3> out{};
    out[0] = p[0] * c + (axis[1] * p[2] - axis[2] * p[1]) * s +
             axis[0] * (axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]) * (1.0f - c);
    out[1] = p[1] * c + (axis[2] * p[0] - axis[0] * p[2]) * s +
             axis[1] * (axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]) * (1.0f - c);
    out[2] = p[2] * c + (axis[0] * p[1] - axis[1] * p[0]) * s +
             axis[2] * (axis[0] * p[0] + axis[1] * p[1] + axis[2] * p[2]) * (1.0f - c);
    return out;
}


void Barrier(VkCommandBuffer c, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
             VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) noexcept {
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, srcAccess, dstAccess};
    vkCmdPipelineBarrier(c, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

} // namespace

// CPU oracle — exact CUDA double target math.
std::vector<float> CurveGrowPipeline::BuildTargets(std::vector<uint64_t> const& stableIds,
                                                   Controls const& controls) {
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

namespace {
using Controls = CurveGrowPipeline::Controls;
using Outputs = CurveGrowPipeline::Outputs;
using Direction = CurveGrowPipeline::Direction;
struct Builder {
    uint32_t cvCount;
    // Sorted frame table for binary search when hasFrames.
    std::vector<uint64_t> frames;   // frameStableIds sorted ascending
    bool hasFrames;
    std::vector<uint64_t> ids;      // curve id per cv = frame id (mapped) or ordinal
    explicit Builder(CurveGrowPipeline::Controls const& c)
        : cvCount(c.cvCount), hasFrames(false) {}

    // Resolve the frame ARRAY INDEX into the sorted frameStableIds for a curve
    // id. Mapped requires an exact frame-stable-id match; false on miss
    // triggers InvalidTopology. The index addresses rootT/B/N (they are ordered
    // to match the sorted frame table).
    bool ResolveFrame(uint64_t curveId, std::vector<uint64_t> const& frameStableIds,
                      uint32_t* indexOut) const {
        (void)frameStableIds;
        auto it = std::lower_bound(frames.begin(), frames.end(), curveId);
        if (it == frames.end() || *it != curveId) return false;
        *indexOut = static_cast<uint32_t>(it - frames.begin());
        return true;
    }

    bool Build(std::vector<float> const& points, std::vector<float> const& rest,
               std::vector<float> const& widths, std::vector<uint32_t> const& offsets,
               std::vector<uint64_t> const& stableIds, std::vector<int32_t> const& rootPrim,
               std::vector<float> const& rootUV, std::vector<float> const& rootT,
               std::vector<float> const& rootB, std::vector<float> const& rootN,
               std::vector<uint64_t> const& frameStableIds,
               std::vector<float> const& targets, CurveGrowPipeline::Controls const& controls,
               CurveGrowPipeline::Outputs* output) const;
};

// Shader WidthLowerIndex: h in [0,1] over inputCount points; returns the lower
// interpolant index, clamped before the cast (curveGrow.comp parity).
float WidthLowerIndex(float h, uint32_t inputCount) noexcept {
    float p = h * float(inputCount - 1);
    float c = std::min(std::max(p, 0.0f), float(inputCount - 1));
    return uint32_t(c);
}

bool FiniteF(std::vector<float> const& v) noexcept {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

bool Builder::Build(std::vector<float> const& points, std::vector<float> const& rest,
                    std::vector<float> const& widths, std::vector<uint32_t> const& offsets,
                    std::vector<uint64_t> const& stableIds,
                    std::vector<int32_t> const& rootPrim, std::vector<float> const& rootUV,
                    std::vector<float> const& rootT, std::vector<float> const& rootB,
                    std::vector<float> const& rootN,
                    std::vector<uint64_t> const& frameStableIds,
                    std::vector<float> const& targets, CurveGrowPipeline::Controls const& controls,
                    CurveGrowPipeline::Outputs* output) const {
    uint32_t cv = cvCount;
    uint32_t curves = static_cast<uint32_t>(stableIds.size());
    bool hasWidths = !widths.empty();
    auto bad = [&]() { return false; };
    // Shape contract BEFORE any indexing (CUDA ValidateInput parity): exact
    // plane sizes, canonical offsets ([0]==0, span>=2), one target per curve,
    // frame planes sized to the frame table.
    if (curves == 0 || cv < 2) return bad();
    if (offsets.size() != std::size_t(curves) + 1 || offsets[0] != 0) return bad();
    const uint32_t pointCount = offsets[curves];
    if (points.size() != std::size_t(pointCount) * 3 ||
        rest.size() != std::size_t(pointCount) * 3) return bad();
    if (hasWidths && widths.size() != pointCount) return bad();
    if (targets.size() != curves) return bad();
    if (rootPrim.size() != curves || rootUV.size() != std::size_t(curves) * 2)
        return bad();
    const std::size_t frameCount =
        frameStableIds.empty() ? curves : frameStableIds.size();
    if (rootT.size() != frameCount * 3 || rootB.size() != frameCount * 3 ||
        rootN.size() != frameCount * 3) return bad();
    for (uint32_t c = 0; c < curves; ++c)
        if (offsets[c + 1] < offsets[c] || offsets[c + 1] - offsets[c] < 2)
            return bad();
    // Finite scan (NonFinite).
    if (!FiniteF(points) || !FiniteF(rest)) return bad();
    if (hasWidths && !FiniteF(widths)) return bad();
    for (float t : targets)
        if (!std::isfinite(t)) return bad();

    Outputs o;
    o.points.resize(3 * size_t(curves) * cv);
    o.rest.resize(3 * size_t(curves) * cv);
    o.widths.resize(size_t(curves) * cv);
    o.hairT.resize(size_t(curves) * cv);
    o.offsets.resize(curves + 1);
    o.ids.resize(curves);
    o.rootPrim.resize(curves);
    o.rootUV.resize(2 * size_t(curves));
    o.rootT.resize(3 * size_t(curves));
    o.rootB.resize(3 * size_t(curves));
    o.rootN.resize(3 * size_t(curves));

    o.offsets[0] = 0;
    for (uint32_t c = 0; c < curves; ++c) {
        uint64_t id = stableIds[c];
        uint32_t frameIdx = 0; // index into sorted frame table (T/B/N arrays)
        if (!frameStableIds.empty()) {
            if (!ResolveFrame(id, frameStableIds, &frameIdx)) return bad();
        } else {
            frameIdx = c; // unmapped: frame array index == curve ordinal
        }
        o.ids[c] = stableIds[c]; // shader copies the curve's INPUT stable id

        uint32_t base = offsets[c];
        if (c + 1 > curves || offsets[c + 1] < base) return bad(); // invalid topology guard
        uint32_t seg = offsets[c + 1] - base; // last curve's seg = offsets[curves] - base

        // Root frame.
        uint32_t r = rootPrim[c];
        if (static_cast<size_t>(frameIdx) * 3 + 2 >= rootT.size() ||
            static_cast<size_t>(frameIdx) * 3 + 2 >= rootB.size() ||
            static_cast<size_t>(frameIdx) * 3 + 2 >= rootN.size() ||
            static_cast<size_t>(c) * 2 + 1 >= rootUV.size())
            return bad();
        std::array<float, 3> t{rootT[3 * frameIdx], rootT[3 * frameIdx + 1], rootT[3 * frameIdx + 2]};
        std::array<float, 3> b{rootB[3 * frameIdx], rootB[3 * frameIdx + 1], rootB[3 * frameIdx + 2]};
        std::array<float, 3> n{rootN[3 * frameIdx], rootN[3 * frameIdx + 1], rootN[3 * frameIdx + 2]};

        // CUDA parity (curveGrow.cu:112-115, shader :151-157): select the RAW
        // direction first, then normalize it once with the (0,1,0) fallback.
        // B is normalized only as the rotation axis inside nonzero-lift; the
        // stored outB keeps the RAW resolved frame value (shader stores bx,by,bz).
        std::array<float, 3> dir;
        switch (controls.direction) {
            case Direction::RootTangent: dir = t; break;
            case Direction::Literal:
                dir = {controls.literalDirection[0], controls.literalDirection[1],
                       controls.literalDirection[2]};
                break;
            default: // RootNormal
                dir = n;
                break;
        }
        Normalize3(dir); // degenerate -> (0,1,0), continue (never reject)
        if (controls.lift != 0.0f) {
            std::array<float, 3> axis = b;
            Normalize3(axis);
            dir = RotateAroundB(dir, axis, controls.lift);
        }
        // CUDA parity (curveGrow.cu:115-117): azimuth about root N after lift.
        float const angle = controls.azimuth + controls.azimuthRandom * 360.0f *
            (DrawGrow(controls.seed, id, kGrowAzimuthSalt) - 0.5f);
        if (angle != 0.0f) {
            std::array<float, 3> axis = n;
            Normalize3(axis);
            dir = RotateAroundB(dir, axis, angle);
        }

        // Per-CV resample + width + hairT. Width interpolates the curve's own input
        // width span (shader formula); points/rest seed from each curve's owned
        // root (points[first]/rest[first]); hairT uses raw CV parameter.
        float target = targets[c];
        float rootPx = points[3 * base + 0], rootPy = points[3 * base + 1], rootPz = points[3 * base + 2];
        float restRx = rest[3 * base + 0], restRy = rest[3 * base + 1], restRz = rest[3 * base + 2];
        for (uint32_t k = 0; k < cv; ++k) {
            float h = float(k) / float(cv - 1);
            float s = target * h;
            o.points[3 * (c * cv + k) + 0] = rootPx + dir[0] * s;
            o.points[3 * (c * cv + k) + 1] = rootPy + dir[1] * s;
            o.points[3 * (c * cv + k) + 2] = rootPz + dir[2] * s;
            o.rest[3 * (c * cv + k) + 0] = restRx + dir[0] * s;
            o.rest[3 * (c * cv + k) + 1] = restRy + dir[1] * s;
            o.rest[3 * (c * cv + k) + 2] = restRz + dir[2] * s;
            float w = controls.fallbackWidth;
            if (hasWidths) {
                uint32_t lo = WidthLowerIndex(h, seg);
                uint32_t hi = std::min(lo + 1, seg - 1);
                float a = h * float(seg - 1) - float(lo);
                w = widths[base + lo] + (widths[base + hi] - widths[base + lo]) * a;
            }
            o.widths[c * cv + k] = w;
            o.hairT[c * cv + k] = h;
        }
        o.offsets[c + 1] = o.offsets[c] + cv;

        o.rootPrim[c] = r;
        o.rootUV[2 * c] = rootUV[2 * c];
        o.rootUV[2 * c + 1] = rootUV[2 * c + 1];
        o.rootT[3 * c] = t[0]; o.rootT[3 * c + 1] = t[1]; o.rootT[3 * c + 2] = t[2];
        o.rootB[3 * c] = b[0]; o.rootB[3 * c + 1] = b[1]; o.rootB[3 * c + 2] = b[2];
        o.rootN[3 * c] = n[0]; o.rootN[3 * c + 1] = n[1]; o.rootN[3 * c + 2] = n[2];
    }
    *output = std::move(o);
    return true;
}
} // namespace

// BuildCpu front door: prepares the sorted frame table and runs the host
// oracle (exact CPU analogue of the CUDA Grow path).
bool CurveGrowPipeline::BuildCpu(std::vector<float> const& points,
                                 std::vector<float> const& rest,
                                 std::vector<float> const& widths,
                                 std::vector<uint32_t> const& curveOffsets,
                                 std::vector<uint64_t> const& stableIds,
                                 std::vector<int32_t> const& rootPrim,
                                 std::vector<float> const& rootUV,
                                 std::vector<float> const& rootT,
                                 std::vector<float> const& rootB,
                                 std::vector<float> const& rootN,
                                 std::vector<uint64_t> const& frameStableIds,
                                 std::vector<float> const& targets,
                                 Controls const& controls, Outputs* output) {
    Builder builder(controls);
    if (!frameStableIds.empty()) {
        // CUDA ValidateFrameStableIds parity: strictly increasing, never sorted
        // (sorting would silently accept input the device rejects).
        if (std::adjacent_find(frameStableIds.begin(), frameStableIds.end(),
                [](uint64_t a, uint64_t b) { return a >= b; }) != frameStableIds.end())
            return false;
        builder.frames = frameStableIds;
    }
    return builder.Build(points, rest, widths, curveOffsets, stableIds, rootPrim, rootUV,
                         rootT, rootB, rootN, frameStableIds, targets, controls, output);
}

// ============================ Device half ============================
namespace {

struct GrowPush {
    uint32_t phase, curves, cvCount, pointCount;
    uint32_t hasWidths, hasFrames, frameCount, direction;
    float lift, fallbackWidth, litDirX, litDirY, litDirZ;
    float azimuth, azimuthRandom;
    uint32_t seedBits;
};
static_assert(sizeof(GrowPush) == 64, "curveGrow.comp push ABI (16 words)");

} // namespace

struct CurveGrowPipeline::Native {
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

struct CurveGrowPipeline::Candidate::State {
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

CurveGrowPipeline::CurveGrowPipeline(std::shared_ptr<Native> n) : native_(std::move(n)) {}
CurveGrowPipeline::~CurveGrowPipeline() = default;
std::shared_ptr<DeviceContext> const& CurveGrowPipeline::context() const noexcept {
    return native_->context;
}

std::shared_ptr<CurveGrowPipeline> CurveGrowPipeline::Create(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
    VkResult* result) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties physical{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &physical);
    if (physical.limits.maxComputeWorkGroupInvocations < 256 ||
        physical.limits.maxComputeWorkGroupSize[0] < 256 ||
        physical.limits.maxPerStageDescriptorStorageBuffers < 24 ||
        physical.limits.maxDescriptorSetStorageBuffers < 24 ||
        physical.limits.maxPushConstantsSize < 64) return {};
    try {
        auto n = std::make_shared<Native>(); n->context = std::move(context);
        auto d = n->context->device();
        VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size() * sizeof(uint32_t); sm.pCode = code.data();
        VkResult r = vkCreateShaderModule(d, &sm, nullptr, &n->shader);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkDescriptorSetLayoutBinding bindings[24]{};
        for (uint32_t i = 0; i != 24; ++i) {
            bindings[i].binding = i; bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ds.bindingCount = 24; ds.pBindings = bindings;
        r = vkCreateDescriptorSetLayout(d, &ds, nullptr, &n->descriptors);
        if (r != VK_SUCCESS) { finish(r); return {}; }
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};
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
        auto pipeline = std::shared_ptr<CurveGrowPipeline>(new CurveGrowPipeline(std::move(n)));
        finish(VK_SUCCESS); return pipeline;
    } catch (std::bad_alloc const&) { finish(VK_ERROR_OUT_OF_HOST_MEMORY); return {}; }
}

CurveGrowPipeline::Candidate::Candidate(std::shared_ptr<State> s) : state_(std::move(s)) {}
CurveGrowPipeline::Candidate::~Candidate() { if (state_->pending) Quarantine(); }
void CurveGrowPipeline::Candidate::Quarantine() noexcept {
    if (!state_->pending || state_->lost) return;
    state_->lost = true;
    *state_->quarantine = state_;
    (void)state_->quarantine.release();
}
bool CurveGrowPipeline::Candidate::succeeded() const noexcept {
    return state_ && !state_->lost && state_->proved && state_->semantic == 0;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::output() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->points : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rest() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rest : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::widths() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->widths : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::hairT() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->hairT : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::offsets() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->offsets : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::ids() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->ids : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rootPrim() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootPrim : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rootUV() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootUV : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rootT() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootT : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rootB() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootB : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::rootN() const noexcept {
    return state_ && state_->proved && state_->semantic == 0 ? state_->rootN : nullptr;
}
std::shared_ptr<const ChargedBuffer> CurveGrowPipeline::Candidate::inputOwner() const noexcept {
    return state_ ? state_->input : nullptr;
}
uint32_t CurveGrowPipeline::Candidate::curveCount() const noexcept {
    return state_ ? state_->curveCount : 0;
}
uint32_t CurveGrowPipeline::Candidate::pointCount() const noexcept {
    return state_ ? state_->pointCount : 0;
}
std::shared_ptr<DeviceContext> CurveGrowPipeline::Candidate::context() const noexcept {
    return state_ && state_->native ? state_->native->context : nullptr;
}

VkResult CurveGrowPipeline::Candidate::Poll(Semantic* semanticStatus) {
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



std::unique_ptr<CurveGrowPipeline::Candidate> CurveGrowPipeline::Begin(
    std::shared_ptr<const void> owner, Input const& input, uint32_t curves,
    uint32_t pointCount, Controls const& controls, VkResult* result,
    BeforeSubmit beforeSubmit) {
    return BeginInternal(std::move(owner), input, curves, pointCount, controls,
                         result, std::move(beforeSubmit));
}

std::unique_ptr<CurveGrowPipeline::Candidate> CurveGrowPipeline::BeginInternal(
    std::shared_ptr<const void> owner, Input const& input, uint32_t curves,
    uint32_t pointCount, Controls const& controls, VkResult* result,
    BeforeSubmit beforeSubmit) {
    auto finish = [&](VkResult r) { if (result) *result = r; };
    finish(VK_ERROR_INITIALIZATION_FAILED);
    auto context = native_->context;

    // ---- Host validation (CUDA CudaCurveGrow::validate parity) ----
    auto in = [&](std::shared_ptr<ChargedBuffer const> const& b) { return b.get(); };
    const bool mapped = in(input.frameStableIds) != nullptr;
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
    if ((curves == 0u) != (pointCount == 0u)) return {};
    const VkDeviceSize idsBytes = VkDeviceSize(curves) * 8u;
    auto check = [&](std::shared_ptr<ChargedBuffer const> const& b, VkDeviceSize bytes) {
        return (!bytes && !b) || (b && b->context() == context && !b->unproven() && b->buffer() &&
            (b->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) && b->sizeBytes() >= bytes);
    };
    const VkDeviceSize pointsBytes = VkDeviceSize(pointCount) * 3u * 4u;
    const VkDeviceSize offsetsBytes = VkDeviceSize(curves + 1u) * 4u;
    if (!check(input.points, pointsBytes) || !check(input.rest, pointsBytes) ||
        !check(input.curveOffsets, offsetsBytes) ||
        !check(input.stableIds, idsBytes) ||
        !check(input.rootPrim, VkDeviceSize(curves) * 4u) ||
        !check(input.rootUV, VkDeviceSize(curves) * 2u * 4u) ||
        // With a frame table, rootT/B/N carry one row per TABLE entry (sized
        // exactly by frameCount*12 in the mapped checks below); without one
        // they are per-curve. Requiring curves-size in mapped mode is wrong.
        !check(input.rootT, mapped ? VkDeviceSize(0) : VkDeviceSize(curves) * 3u * 4u) ||
        !check(input.rootB, mapped ? VkDeviceSize(0) : VkDeviceSize(curves) * 3u * 4u) ||
        !check(input.rootN, mapped ? VkDeviceSize(0) : VkDeviceSize(curves) * 3u * 4u) ||
        !check(input.targets, VkDeviceSize(curves) * 4u)) return {};
    // Never reads device stableIds host-side; targets span already uploaded.
    if (input.widths && !check(input.widths, 0)) return {};
    if (input.widths && input.widths->sizeBytes() != VkDeviceSize(pointCount) * 4u) return {};
    const VkDeviceSize frameBytes = mapped
        ? input.frameStableIds->sizeBytes() : 0;
    if (mapped && (input.frameStableIds->sizeBytes() % 8u != 0 || frameBytes == 0 ||
        input.rootT->sizeBytes() < input.frameStableIds->sizeBytes() ||
        input.rootB->sizeBytes() < input.frameStableIds->sizeBytes() ||
        input.rootN->sizeBytes() < input.frameStableIds->sizeBytes())) return {};
    if (!mapped && in(input.frameStableIds)) return {};
    if (mapped && !check(input.frameStableIds, 8u)) return {};
    const uint32_t frameCount = mapped
        ? uint32_t(input.frameStableIds->sizeBytes() / 8u) : 0u;
    if (mapped && (input.rootT->sizeBytes() != VkDeviceSize(frameCount) * 12u ||
        input.rootB->sizeBytes() != VkDeviceSize(frameCount) * 12u ||
        input.rootN->sizeBytes() != VkDeviceSize(frameCount) * 12u)) return {};

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &properties);
    uint64_t groups = (uint64_t(curves) + 255) / 256;
    uint64_t frameGroups = (uint64_t(frameCount) + 255) / 256;
    if (groups > properties.limits.maxComputeWorkGroupCount[0] ||
        frameGroups > properties.limits.maxComputeWorkGroupCount[0]) return {};

    try {
        auto s = std::make_shared<Candidate::State>(); s->native = native_;
        s->owner = std::move(owner);
        s->input = input.points;
        s->curveCount = curves; s->pointCount = curves * cv;
        s->quarantine = std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate = std::unique_ptr<Candidate>(new Candidate(s));
        if (!curves) { s->proved = true; s->semantic = 0; finish(VK_SUCCESS); return candidate; }
        auto d = context->device();
        VkResult r;
        auto mkOut = [&](VkDeviceSize bytes, std::shared_ptr<ChargedBuffer>* out) -> bool {
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            *out = ChargedBuffer::Create(context, bi, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                         UsdGenExecutionResourceKind::Active, &r);
            return *out != nullptr;
        };
        const VkDeviceSize outPts = VkDeviceSize(curves) * cv * 3u * 4u;
        const VkDeviceSize outPer = VkDeviceSize(curves) * cv * 4u;
        if (!mkOut(outPts, &s->points) || !mkOut(outPts, &s->rest) ||
            !mkOut(outPer, &s->widths) || !mkOut(outPer, &s->hairT) ||
            !mkOut(VkDeviceSize(curves + 1u) * 4u, &s->offsets) ||
            !mkOut(VkDeviceSize(curves) * 8u, &s->ids) ||
            !mkOut(VkDeviceSize(curves) * 4u, &s->rootPrim) ||
            !mkOut(VkDeviceSize(curves) * 8u, &s->rootUV) ||
            !mkOut(VkDeviceSize(curves) * 12u, &s->rootT) ||
            !mkOut(VkDeviceSize(curves) * 12u, &s->rootB) ||
            !mkOut(VkDeviceSize(curves) * 12u, &s->rootN)) { finish(r); return {}; }
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

        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24u};
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
        VkDescriptorBufferInfo infos[24] = {
            {input.points->buffer(), 0, pointsBytes}, {input.rest->buffer(), 0, pointsBytes},
            {input.widths ? input.widths->buffer() : s->status->buffer(), 0, input.widths ? VkDeviceSize(pointCount) * 4u : 4u},
            {input.curveOffsets->buffer(), 0, offsetsBytes}, {input.stableIds->buffer(), 0, idsBytes},
            {input.rootPrim->buffer(), 0, VkDeviceSize(curves) * 4u},
            {input.rootUV->buffer(), 0, VkDeviceSize(curves) * 8u},
            {input.rootT->buffer(), 0, input.rootT->sizeBytes()},
            {input.rootB->buffer(), 0, input.rootB->sizeBytes()},
            {input.rootN->buffer(), 0, input.rootN->sizeBytes()},
            {mapped ? input.frameStableIds->buffer() : s->status->buffer(), 0, mapped ? frameBytes : 4u},
            {input.targets->buffer(), 0, VkDeviceSize(curves) * 4u},
            {s->points->buffer(), 0, outPts}, {s->rest->buffer(), 0, outPts},
            {s->widths->buffer(), 0, outPer}, {s->hairT->buffer(), 0, outPer},
            {s->offsets->buffer(), 0, VkDeviceSize(curves + 1u) * 4u},
            {s->ids->buffer(), 0, VkDeviceSize(curves) * 8u},
            {s->rootPrim->buffer(), 0, VkDeviceSize(curves) * 4u},
            {s->rootUV->buffer(), 0, VkDeviceSize(curves) * 8u},
            {s->rootT->buffer(), 0, VkDeviceSize(curves) * 12u},
            {s->rootB->buffer(), 0, VkDeviceSize(curves) * 12u},
            {s->rootN->buffer(), 0, VkDeviceSize(curves) * 12u},
            {s->status->buffer(), 0, 4u}};
        VkWriteDescriptorSet writes[24]{};
        for (uint32_t i = 0; i != 24u; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(d, 24, writes, 0, nullptr);

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
        VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        auto pipelineLayout = native_->layout;
        auto pipeline = native_->pipeline;
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        GrowPush push{};
        push.phase = 0u; push.curves = curves; push.cvCount = cv; push.pointCount = pointCount;
        push.hasWidths = input.widths ? 1u : 0u; push.hasFrames = mapped ? 1u : 0u;
        push.frameCount = frameCount; push.direction = uint32_t(controls.direction);
        push.lift = controls.lift; push.fallbackWidth = controls.fallbackWidth;
        push.litDirX = controls.literalDirection[0];
        push.litDirY = controls.literalDirection[1];
        push.litDirZ = controls.literalDirection[2];
        push.azimuth = controls.azimuth; push.azimuthRandom = controls.azimuthRandom;
        std::memcpy(&push.seedBits, &controls.seed, sizeof(push.seedBits));
        vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, &push);
        vkCmdDispatch(command, uint32_t(groups), 1, 1);
        if (mapped) {
            push.phase = 1u;
            vkCmdPushConstants(command, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 64, &push);
            vkCmdDispatch(command, uint32_t(frameGroups), 1, 1);
        }
        VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        after.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
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
