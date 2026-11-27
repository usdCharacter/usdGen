// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// rootFrames pipeline test: two-submit Begin validation semantics, device
// build vs CPU oracle (abs+rel tolerance per frozen Normalize criterion),
// authored host path, rejection codes, quarantine pinning.
// argv: <rootFramesValidate.spv> <rootFrames.spv> [--quarantine]

#include "usdGenSurface/vulkan/rootFramesPipeline.h"
#include "hosts/usdGenUsd/vulkanNativeFixture.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan RootFrames check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

template <typename T>
static std::shared_ptr<ChargedBuffer> Upload(std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<NativeOwner> const& native, std::vector<T> const& values) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = values.size() * sizeof(T);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Active);
    if (!buffer) return buffer;
    void* mapped = nullptr;
    if (vkMapMemory(native->device, buffer->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS)
        return std::shared_ptr<ChargedBuffer>{};
    std::memcpy(mapped, values.data(), bi.size); vkUnmapMemory(native->device, buffer->memory());
    return buffer;
}

template <typename T>
static bool Download(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<const ChargedBuffer> const& source, std::vector<T>* values) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = values->size() * sizeof(T); bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(source->context(), bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging) return false;
    VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkBufferCopy region{0, 0, bi.size};
    vkCmdCopyBuffer(command, source->buffer(), staging->buffer(), 1, &region);
    VkMemoryBarrier barrier{}; barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    struct Keep { std::shared_ptr<NativeOwner> native; std::shared_ptr<const ChargedBuffer> source; };
    auto keep = std::make_shared<Keep>(Keep{native, source});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS) return false;
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS) return false;
    std::memcpy(values->data(), mapped, bi.size); vkUnmapMemory(native->device, staging->memory());
    return true;
}

// ------------------------------------------------ CPU oracle (port of
// rootFrames.comp exact float arithmetic, count==3 faces, geometric normals)
struct Vec3f { float x, y, z; };
static Vec3f Sub(Vec3f a, Vec3f b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec3f Add(Vec3f a, Vec3f b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec3f Mul(Vec3f a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float Dot(Vec3f a, Vec3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3f Cross(Vec3f a, Vec3f b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static bool NormalizeP(Vec3f v, Vec3f* out) {  // shader: exact sqrt + per-component divide
    float l2 = Dot(v, v);
    if (!(l2 == l2) || !(l2 < INFINITY) || l2 < 1e-12f) return false;
    float len = std::sqrt(l2);
    *out = {v.x / len, v.y / len, v.z / len};
    return true;
}

static bool CloseAbsRel(float got, float want) {
    float const kAbs = 2e-5f, kRel = 2e-5f;  // tolerance gate; never bit-exact
    float d = std::abs(got - want);
    return d <= kAbs || d <= kRel * std::abs(want);
}
static bool CloseVec(std::vector<float> const& got, size_t c, Vec3f want) {
    return CloseAbsRel(got[3 * c], want.x) && CloseAbsRel(got[3 * c + 1], want.y) &&
        CloseAbsRel(got[3 * c + 2], want.z);
}

static std::vector<float> g_rest;
static std::vector<uint32_t> g_offsets, g_indices;
static Vec3f Rest(uint32_t v) { return {g_rest[3 * v], g_rest[3 * v + 1], g_rest[3 * v + 2]}; }

// count==3 barycentric with exact CUDA nesting: (v0*a + v1*u) + v2*w.
static Vec3f Inter3(Vec3f v0, Vec3f v1, Vec3f v2, float u, float w) {
    float a = 1.0f - u - w;
    return Add(Add(Mul(v0, a), Mul(v1, u)), Mul(v2, w));
}

// Per-curve oracle for one triangle face + geometric normal. valid=1 expected.
static bool OracleCurve(uint32_t face, float u, float w,
                        Vec3f* origin, Vec3f* tangent, Vec3f* binormal, Vec3f* normal) {
    uint32_t b = g_offsets[face];
    Vec3f v0 = Rest(g_indices[b]), v1 = Rest(g_indices[b + 1]), v2 = Rest(g_indices[b + 2]);
    Vec3f n;
    if (!NormalizeP(Cross(Sub(v1, v0), Sub(v2, v0)), &n)) return false;
    Vec3f deriv = Sub(v1, v0);
    Vec3f projected = Sub(deriv, Mul(n, Dot(n, deriv)));
    Vec3f t;
    if (!NormalizeP(projected, &t)) {
        Vec3f edge = Sub(v1, v0);
        projected = Sub(edge, Mul(n, Dot(n, edge)));
        if (!NormalizeP(projected, &t)) return false;
    }
    Vec3f bin;
    if (!NormalizeP(Cross(n, t), &bin)) return false;
    *normal = n; *tangent = t; *binormal = bin;
    *origin = Inter3(v0, v1, v2, u, w);
    return true;
}

int main(int argc, char** argv) {
    bool quarantine = argc == 4 && std::string(argv[3]) == "--quarantine";
    CHECK(argc == 3 || quarantine);
    auto load = [](char const* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return std::vector<uint32_t>{};
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
        if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) return std::vector<uint32_t>{};
        std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
        std::memcpy(code.data(), bytes.data(), bytes.size());
        return code;
    };
    auto validateSpirv = load(argv[1]);
    auto buildSpirv = load(argv[2]);
    CHECK(!validateSpirv.empty() && !buildSpirv.empty());

    bool unavailable = false;
    auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    {   // rootFrames build set = 14 storage bindings; skip weak devices (77).
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(native->physical, &props);
        if (props.limits.maxPerStageDescriptorStorageBuffers < 14 ||
            props.limits.maxDescriptorSetStorageBuffers < 14 ||
            props.limits.maxPushConstantsSize < 24) return 77;
    }
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7005; info.nativeLifetime = native;
    info.timelineSemaphoreEnabled = true;
    info.resources = {size_t{16} << 20, 0};
    auto context = DeviceContext::Create(info);
    CHECK(context); info.nativeLifetime.reset();
    auto pool = context->resources();
    VkResult status = VK_SUCCESS;
    CHECK(!RootFramesPipeline::Create(context, {0x07230203u}, {0x07230203u}, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);

    // Geometry: two triangle faces over 5 vertices (shared edge 0-1).
    // face0 = (0,1,2), face1 = (0,3,4).
    std::vector<uint32_t> faceOffsets = {0, 3, 6};
    std::vector<uint32_t> faceIndices = {0, 1, 2, 0, 3, 4};
    std::vector<float> restVerts;
    float const verts[5][3] = {
        {0.f, 0.f, 0.f}, {1.f, 0.f, 0.1f}, {0.2f, 1.f, 0.3f},
        {-0.5f, 1.f, 0.2f}, {1.5f, 0.8f, -0.4f}};
    for (auto& v : verts) { restVerts.push_back(v[0]); restVerts.push_back(v[1]); restVerts.push_back(v[2]); }
    g_rest = restVerts; g_offsets = faceOffsets; g_indices = faceIndices;

    // 3 curves: face0 uv(.2,.3); face1 uv(.75,.1); face0 uv(.5,.5).
    std::vector<int32_t> rootPrim = {0, 1, 0};
    std::vector<float> rootUV = {0.2f, 0.3f, 0.75f, 0.1f, 0.5f, 0.5f};
    uint32_t const curves = 3, vertexCount = 5, faceCount = 2, indexCount = 6;

    auto pipeline = RootFramesPipeline::Create(context, validateSpirv, buildSpirv, &status);
    CHECK(pipeline && status == VK_SUCCESS);

    auto makeInfo = [&](std::shared_ptr<DeviceContext> const&,
                        std::shared_ptr<ChargedBuffer> const& rest,
                        std::shared_ptr<ChargedBuffer> const& offs,
                        std::shared_ptr<ChargedBuffer> const& idx,
                        std::shared_ptr<ChargedBuffer> const& prim,
                        std::shared_ptr<ChargedBuffer> const& uv) {
        RootFramesPipeline::BeginInfo bi;
        bi.restVerts = rest; bi.faceOffsets = offs; bi.faceIndices = idx;
        bi.rootPrim = prim; bi.rootUV = uv;
        bi.normalDomain = RootFrameNormalDomain::None;
        bi.vertexCount = vertexCount; bi.faceCount = faceCount;
        bi.indexCount = indexCount; bi.curves = curves;
        return bi;
    };
    auto restBuf = Upload(context, native, restVerts);
    auto offsBuf = Upload(context, native, faceOffsets);
    auto idxBuf = Upload(context, native, faceIndices);
    auto primBuf = Upload(context, native, rootPrim);
    auto uvBuf = Upload(context, native, rootUV);
    CHECK(restBuf && offsBuf && idxBuf && primBuf && uvBuf);

    // ------------------------------------------------------------ quarantine
    if (quarantine) {
        auto bi = makeInfo(context, restBuf, offsBuf, idxBuf, primBuf, uvBuf);
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS && cand->count() == curves);
        auto charged = pool->Snapshot();
        std::weak_ptr<const ChargedBuffer> weakIn = cand->inputOwner();
        cand->Quarantine(); cand.reset(); pipeline.reset(); context.reset();
        CHECK(Prove(native)); native.reset();
        CHECK(!weakNative.expired());
        CHECK(!weakIn.expired());  // quarantine pins borrowed inputs forever
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes &&
              pool->Snapshot().byKind == charged.byKind);
        return 0;
    }

    // ------------------------------------------------------- (a) clean build
    // Inputs remain live for the whole scenario; candidate charges must return
    // to exactly the input charge after the candidate dies.
    auto const inputsOnly = pool->Snapshot();
    {
        auto bi = makeInfo(context, restBuf, offsBuf, idxBuf, primBuf, uvBuf);
        RootFramesBeginStatus beginStatus = RootFramesBeginStatus::Ok;
        auto cand = pipeline->Begin(bi, &status, &beginStatus);
        CHECK(cand && status == VK_SUCCESS && cand->count() == curves);
        CHECK(!cand->output().origin);  // hidden pre-proof
        CHECK(Prove(native));
        uint32_t semantic = 999, badRoot = 999;
        CHECK(cand->Poll(&semantic, &badRoot) == VK_SUCCESS);
        CHECK(semantic == 0 && badRoot == 0 && cand->succeeded());
        auto out = cand->output();
        CHECK(out.origin && out.tangent && out.binormal && out.normal && out.valid && out.drop);
        CHECK(out.origin->context() == context);
        std::vector<float> got(3 * curves);
        std::vector<uint32_t> valid(curves), drop(curves);
        CHECK(Download(native, out.valid, &valid) && Download(native, out.drop, &drop));
        for (uint32_t c = 0; c != curves; ++c) CHECK(valid[c] == 1u && drop[c] == 0u);
        for (int plane = 0; plane != 4; ++plane) {
            auto const& buf = plane == 0 ? out.origin : plane == 1 ? out.tangent :
                plane == 2 ? out.binormal : out.normal;
            CHECK(Download(native, buf, &got));
            for (uint32_t c = 0; c != curves; ++c) {
                Vec3f o, t, b, n;
                CHECK(OracleCurve(uint32_t(rootPrim[c]), rootUV[2 * c], rootUV[2 * c + 1],
                                  &o, &t, &b, &n));
                Vec3f want = plane == 0 ? o : plane == 1 ? t : plane == 2 ? b : n;
                CHECK(CloseVec(got, c, want));
            }
        }
        out = {};  // dropping all output refs so charges can be reclaimed
        cand.reset();
        CHECK(pool->Snapshot().usedBytes == inputsOnly.usedBytes &&
              pool->Snapshot().byKind == inputsOnly.byKind);
    }

    // --------------------------------------- (b) rejections (no candidate,
    // ----------------                                  no residual charge)
    // Returns nonzero iff a CHECK inside fails (CHECK yields int).
    auto rejected = [&](RootFramesPipeline::BeginInfo const& bi,
                        RootFramesBeginStatus want) -> int {
        auto before = pool->Snapshot();
        RootFramesBeginStatus code = RootFramesBeginStatus::Ok;
        auto cand = pipeline->Begin(bi, &status, &code);
        CHECK(!cand && status == VK_ERROR_INITIALIZATION_FAILED && code == want);
        CHECK(pool->Snapshot().usedBytes == before.usedBytes &&
              pool->Snapshot().byKind == before.byKind);
        return 0;
    };
    {
        auto bi = makeInfo(context, restBuf, offsBuf, idxBuf, primBuf, uvBuf);
        // BadRoot: rootPrim >= faceCount.
        auto badPrim = Upload(context, native, std::vector<int32_t>{0, 1, 9});
        CHECK(badPrim);
        auto probe = bi; probe.rootPrim = badPrim;
        CHECK(rejected(probe, RootFramesBeginStatus::BadRoot) == 0);
        // NonFinite: NaN vertex.
        auto nanVerts = restVerts; nanVerts[1] = std::nanf("");
        auto nanBuf = Upload(context, native, nanVerts);
        CHECK(nanBuf);
        probe = bi; probe.restVerts = nanBuf;
        CHECK(rejected(probe, RootFramesBeginStatus::NonFinite) == 0);
        // BadTopology: index >= vertexCount.
        auto badIdx = Upload(context, native, std::vector<uint32_t>{0, 1, 2, 0, 3, 99});
        CHECK(badIdx);
        probe = bi; probe.faceIndices = badIdx;
        CHECK(rejected(probe, RootFramesBeginStatus::BadTopology) == 0);
        // kBadRoot: the validate shader mirrors CUDA's combined root branch —
        // rootUV finiteness and rootPrim range both classify as kBadRoot(4).
        auto badUv = Upload(context, native, std::vector<float>{0.2f, 0.3f, 0.75f, std::nanf(""), 0.5f, 0.5f});
        CHECK(badUv);
        probe = bi; probe.rootUV = badUv;
        CHECK(rejected(probe, RootFramesBeginStatus::BadRoot) == 0);
        // Host shape contract: mis-sized offsets plane rejected with no status.
        probe = bi;
        probe.faceOffsets = Upload(context, native, std::vector<uint32_t>{0, 3});
        CHECK(probe.faceOffsets);
        auto before = pool->Snapshot();
        CHECK(!pipeline->Begin(probe, &status));
        CHECK(pool->Snapshot().usedBytes == before.usedBytes);
    }

    // ------------------------------------------------ (c) authored host path
    {
        std::vector<double> frames;
        for (uint32_t c = 0; c != curves; ++c) {
            double m[16] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,
                            double(c), 0.5, -0.25, 1};
            frames.insert(frames.end(), m, m + 16);
        }
        auto bi = makeInfo(context, restBuf, offsBuf, idxBuf, primBuf, uvBuf);
        bi.authoredFrames = frames.data();
        bi.authoredFrameCount = curves * 16;
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS);
        uint32_t semantic = 999, badRoot = 999;
        CHECK(cand->Poll(&semantic, &badRoot) == VK_SUCCESS);  // host-proved
        CHECK(semantic == 0 && badRoot == 0 && cand->succeeded());
        auto out = cand->output();
        std::vector<float> got(3 * curves);
        std::vector<uint32_t> valid(curves), drop(curves);
        CHECK(Download(native, out.origin, &got));
        for (uint32_t c = 0; c != curves; ++c) {
            CHECK(CloseAbsRel(got[3 * c], float(c)) && CloseAbsRel(got[3 * c + 1], 0.5f) &&
                  CloseAbsRel(got[3 * c + 2], -0.25f));
        }
        CHECK(Download(native, out.tangent, &got));
        CHECK(CloseAbsRel(got[0], 1.f) && CloseAbsRel(got[1], 0.f) && CloseAbsRel(got[2], 0.f));
        CHECK(Download(native, out.normal, &got));
        CHECK(CloseAbsRel(got[2], 1.f));
        CHECK(Download(native, out.valid, &valid) && Download(native, out.drop, &drop));
        for (uint32_t c = 0; c != curves; ++c) CHECK(valid[c] == 1u && drop[c] == 0u);
        out = {};  // drop refs so charges are reclaimable
        cand.reset();
        CHECK(pool->Snapshot().usedBytes == inputsOnly.usedBytes &&
              pool->Snapshot().byKind == inputsOnly.byKind);

        // Non-finite authored element classifies as NonFinite(2) (CUDA parity).
        auto broken = frames; broken[5] = std::nan("");
        bi.authoredFrames = broken.data();
        RootFramesBeginStatus code = RootFramesBeginStatus::Ok;
        auto before = pool->Snapshot();
        CHECK(!pipeline->Begin(bi, &status, &code));
        CHECK(code == RootFramesBeginStatus::NonFinite);
        CHECK(pool->Snapshot().usedBytes == before.usedBytes);

        // Structure failure (m[15] != 1) classifies as BadAuthoredFrame(5).
        broken = frames; broken[15] = 0.5;
        bi.authoredFrames = broken.data();
        CHECK(!pipeline->Begin(bi, &status, &code));
        CHECK(code == RootFramesBeginStatus::BadAuthoredFrame);

        // Left-handed frame also BadAuthoredFrame.
        broken = frames; broken[8] = -1.0;  // N = (0,0,-1)
        bi.authoredFrames = broken.data();
        CHECK(!pipeline->Begin(bi, &status, &code));
        CHECK(code == RootFramesBeginStatus::BadAuthoredFrame);

        // Mis-sized authored count is a host argument rejection (no code).
        bi.authoredFrameCount = curves * 16 - 1;
        CHECK(!pipeline->Begin(bi, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    }

    // ------------------------------------------------ empty-curve candidate
    {
        RootFramesPipeline::BeginInfo bi;
        bi.restVerts = restBuf;  // context bearer
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS && cand->count() == 0 && cand->succeeded());
        uint32_t semantic = 999, badRoot = 999;
        CHECK(cand->Poll(&semantic, &badRoot) == VK_SUCCESS);
        CHECK(semantic == 0 && badRoot == 0);
    }
    restBuf = offsBuf = idxBuf = primBuf = uvBuf = {};
    CHECK(pool->Snapshot().usedBytes == 0 &&
          pool->Snapshot().byKind == decltype(pool->Snapshot().byKind){});
    pool.reset();
    pipeline.reset(); context.reset(); native.reset();
    CHECK(weakNative.expired());
    std::fprintf(stderr, "testUsdGenVulkanRootFrames: OK\n");
    return 0;
}
