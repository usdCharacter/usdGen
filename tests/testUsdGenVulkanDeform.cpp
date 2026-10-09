// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deformEvaluateCpu.h"
#include "usdGen/vulkan/deformPipeline.h"
#include "usdGen/vulkan/deformRbfHost.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

inline float3 V(float x, float y, float z) { return {x, y, z}; }
inline bool Near(float3 a, float3 b, float tol) {
    return std::fabs(a.x - b.x) <= tol && std::fabs(a.y - b.y) <= tol &&
           std::fabs(a.z - b.z) <= tol;
}

inline std::vector<uint32_t> Code(char const* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}

inline bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

inline std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const& c, void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = ChargedBuffer::Create(c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b) return {};
    if (bytes) {
        void* p = nullptr;
        if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS) return {};
        std::memcpy(p, data, bytes);
        vkUnmapMemory(c->device(), b->memory());
    }
    return b;
}

// Read back the candidate's output points.
inline bool ReadOutput(std::shared_ptr<NativeOwner> const& native,
                       std::shared_ptr<DeviceContext> const& context,
                       std::unique_ptr<DeformPipeline::Candidate> const& c,
                       std::vector<float>* out) {
    auto o = c->output();
    std::vector<uint8_t> bytes;
    if (!ReadVulkanBytes(native, context, o.points->buffer(),
                         o.points->sizeBytes(), o.points, &bytes)) return false;
    out->resize(bytes.size() / sizeof(float));
    std::memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s deformEvaluate.spv deformApply.spv "
                             "deformEvaluateVerify.spv deformEvaluateFill.spv "
                             "deformEvaluateCached.spv\n",
                     argv[0]);
        return 1;
    }
    auto evalSpv = Code(argv[1]);
    auto applySpv = Code(argv[2]);
    CHECK(!evalSpv.empty() && !applySpv.empty());
    DeformEvalCacheSpirv cacheSpv;
    cacheSpv.verify = Code(argv[3]);
    cacheSpv.fill = Code(argv[4]);
    cacheSpv.cached = Code(argv[5]);
    CHECK(!cacheSpv.verify.empty() && !cacheSpv.fill.empty() && !cacheSpv.cached.empty());

    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64 = {};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, nullptr, &fp64);
    if (unavailable) return 77;
    CHECK(native);

    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(native->physical, &feats);
    if (!feats.shaderFloat64) return 77;

    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8020;
    ci.nativeLifetime = native;
    ci.resources = {size_t{32} << 20, 0};
    ci.shaderFloat64Enabled = true;
    auto context = DeviceContext::Create(ci);
    CHECK(context);

    VkResult status = VK_SUCCESS;
    auto pipe = DeformPipeline::Create(context, evalSpv, applySpv, &status);
    CHECK(pipe && status == VK_SUCCESS);

    // ---- Base geometry: 2 curves, 6 CVs ----
    // Curve 0: 3 points along +x near origin
    // Curve 1: 3 points along +x at x=1
    const std::vector<float> points{
        0.0f, 0.0f, 0.0f,   // curve 0, point 0
        0.1f, 0.0f, 0.1f,   // curve 0, point 1
        0.2f, 0.0f, 0.2f,   // curve 0, point 2
        1.0f, 0.0f, 0.0f,   // curve 1, point 0
        1.1f, 0.0f, 0.1f,   // curve 1, point 1
        1.2f, 0.0f, 0.2f,   // curve 1, point 2
    };
    const std::vector<uint32_t> offsets{0, 3, 6};
    const std::vector<float> rootTargets{
        0.0f, 0.0f, 0.0f,   // curve 0 root
        1.0f, 0.0f, 0.0f,   // curve 1 root
    };

    // 5 non-coplanar rest samples.
    const std::vector<float> restSamples{
        0.0f, 0.0f, 0.0f,   // sample 0
        1.0f, 0.0f, 0.0f,   // sample 1
        0.0f, 1.0f, 0.0f,   // sample 2
        0.0f, 0.0f, 1.0f,   // sample 3
        0.5f, 0.5f, 0.5f,   // sample 4
    };

    auto pointsBuf = Upload(context, points.data(), points.size() * sizeof(float));
    auto offsetsBuf = UploadVulkanDeviceBytes(native, context, offsets.data(),
        offsets.size() * sizeof(uint32_t));
    auto targetsBuf = Upload(context, rootTargets.data(), rootTargets.size() * sizeof(float));
    CHECK(pointsBuf && offsetsBuf && targetsBuf);

    auto makeParams = [](uint32_t lockLiteral = 0, float maskLiteral = 1.0f,
                         uint32_t enabledLiteral = 1, float groomEnvelope = 1.0f) {
        DeformApplyParams p;
        p.groomEnvelope = groomEnvelope;
        p.maskLiteral = maskLiteral; p.maskHasData = false; p.maskDomain = 1;
        p.enabledLiteral = enabledLiteral; p.enabledHasData = false; p.enabledDomain = 1;
        p.lockLiteral = lockLiteral; p.lockHasData = false; p.lockDomain = 1;
        return p;
    };

    // ---- Case 10: a caller-held Output keeps its bytes across reuse ----
    // Runs first, against an empty pool: the outPoints pool reuses a
    // destroyed candidate's buffer for the next same-size pose, but only
    // while the pool owns it alone. An Output held past destruction pins
    // its buffer (the next pose allocates fresh), and dropping the pin lets
    // a later pose reuse it with correct output.
    {
        const std::vector<float> points12{
            0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.2f, 0.0f, 0.2f,
            1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f, 1.2f, 0.0f, 0.2f,
            2.0f, 0.0f, 0.0f, 2.1f, 0.0f, 0.1f, 2.2f, 0.0f, 0.2f,
            3.0f, 0.0f, 0.0f, 3.1f, 0.0f, 0.1f, 3.2f, 0.0f, 0.2f,
        };
        const std::vector<uint32_t> offsets12{0, 3, 6, 9, 12};
        const std::vector<float> roots12{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
        };
        auto pointsBuf12 = Upload(context, points12.data(), points12.size() * sizeof(float));
        auto offsetsBuf12 = UploadVulkanDeviceBytes(native, context, offsets12.data(),
            offsets12.size() * sizeof(uint32_t));
        auto targetsBuf12 = Upload(context, roots12.data(), roots12.size() * sizeof(float));
        CHECK(pointsBuf12 && offsetsBuf12 && targetsBuf12);
        auto makeInfo = [&](std::vector<float> const& posed) {
            DeformPipeline::BeginInfo info;
            info.points = pointsBuf12;
            info.curveOffsets = offsetsBuf12;
            info.rootTargets = targetsBuf12;
            info.curveCount = 4;
            info.pointCount = 12;
            info.restSamples = restSamples;
            info.posedSamples = posed;
            info.sampleCount = 5;
            info.smoothing = 0.0;
            info.mask = {1.0f, 1, nullptr, 0};
            info.enabled = {1, 1, nullptr, 0};
            info.lockRoots = {0, 1, nullptr, 0};
            info.groomEnvelope = 1.0f;
            return info;
        };
        auto runPose = [&](std::vector<float> const& posed, std::vector<float>* out) {
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(makeInfo(posed), &status, &sem);
            if (!c || status != VK_SUCCESS) return false;
            if (!Prove(native)) return false;
            if (c->Poll(&sem) != VK_SUCCESS) return false;
            if (sem != DeformSemantic::Ok || !c->succeeded()) return false;
            return ReadOutput(native, context, c, out);
        };
        std::vector<float> posedA = restSamples, posedB = restSamples;
        for (size_t i = 0; i < posedA.size(); i += 3) posedA[i] += 0.5f;
        for (size_t i = 0; i < posedB.size(); i += 3) posedB[i] -= 0.25f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto cA = pipe->Begin(makeInfo(posedA), &status, &sem);
        CHECK(cA && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(cA->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok && cA->succeeded());
        auto held = cA->output();   // pins A's buffer past cA's death
        CHECK(held.points);
        std::vector<float> outA;
        CHECK(ReadOutput(native, context, cA, &outA));
        cA.reset();                 // A's buffer rejoins the pool, still shared

        std::vector<float> outB;
        CHECK(runPose(posedB, &outB));
        CHECK(outB.size() == outA.size() && outA.size() == 36 && outB != outA);
        for (int i = 0; i < 12; ++i) {
            CHECK(Near(V(outA[size_t(i * 3)], outA[size_t(i * 3 + 1)], outA[size_t(i * 3 + 2)]),
                       V(points12[size_t(i * 3)] + 0.5f, points12[size_t(i * 3 + 1)],
                         points12[size_t(i * 3 + 2)]), 1e-3f));
            CHECK(Near(V(outB[size_t(i * 3)], outB[size_t(i * 3 + 1)], outB[size_t(i * 3 + 2)]),
                       V(points12[size_t(i * 3)] - 0.25f, points12[size_t(i * 3 + 1)],
                         points12[size_t(i * 3 + 2)]), 1e-3f));
        }
        // The held Output still reads pose A's bytes: B must have allocated
        // fresh (or reused an older idle buffer), never A's pinned buffer.
        std::vector<uint8_t> bytesA;
        CHECK(ReadVulkanBytes(native, context, held.points->buffer(),
                              held.points->sizeBytes(), held.points, &bytesA));
        CHECK(bytesA.size() == outA.size() * sizeof(float));
        CHECK(std::memcmp(bytesA.data(), outA.data(), bytesA.size()) == 0);
        // Dropping the pin lets a later same-size pose reuse A's buffer;
        // the output is bitwise the earlier pose-B output, so no stale byte
        // survives the reuse.
        held.points.reset();
        std::vector<float> outC;
        CHECK(runPose(posedB, &outC));
        CHECK(outC.size() == outB.size());
        CHECK(std::memcmp(outC.data(), outB.data(), outC.size() * sizeof(float)) == 0);
        std::puts("Case 10 (held Output survives reuse): PASS");
    }

    // ---- Case 1: Empty topology ----
    {
        DeformPipeline::BeginInfo info;
        info.curveCount = 0;
        info.pointCount = 0;
        info.sampleCount = 5;
        info.restSamples = restSamples;
        info.posedSamples = restSamples;  // identity
        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());
        CHECK(c->pointCount() == 0);
        CHECK(!c->output().points);
        std::puts("Case 1 (empty topology): PASS");
    }

    // ---- Case 2: Identity pose (posed == rest) → output == input ----
    {
        std::vector<float> posed = restSamples;
        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);
        for (int i = 0; i < 6; ++i) {
            float3 got = V(out[i*3], out[i*3+1], out[i*3+2]);
            float3 exp = V(points[i*3], points[i*3+1], points[i*3+2]);
            CHECK(Near(got, exp, 1e-4f));
        }
        std::puts("Case 2 (identity pose): PASS");
    }

    // ---- Case 3: Rigid translation → output = input + d ----
    {
        const float d[3] = {0.3f, -0.2f, 0.5f};
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i)
            for (int ax = 0; ax < 3; ++ax)
                posed[i*3+ax] = restSamples[i*3+ax] + d[ax];

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);
        // Print all points first to diagnose.
        for (int i = 0; i < 6; ++i) {
            float3 got = V(out[i*3], out[i*3+1], out[i*3+2]);
            float3 exp = V(points[i*3]+d[0], points[i*3+1]+d[1], points[i*3+2]+d[2]);
            std::fprintf(stderr, "  case3 p%d got(%.6f,%.6f,%.6f) exp(%.6f,%.6f,%.6f)\n", i,
                got.x, got.y, got.z, exp.x, exp.y, exp.z);
        }
        for (int i = 0; i < 6; ++i) {
            float3 got = V(out[i*3], out[i*3+1], out[i*3+2]);
            float3 exp = V(points[i*3]+d[0], points[i*3+1]+d[1], points[i*3+2]+d[2]);
            CHECK(Near(got, exp, 1e-3f));
        }
        std::puts("Case 3 (rigid translation): PASS");
    }

    // Connected Groom fields contain one element and broadcast over two curves.
    {
        std::vector<float> posed=restSamples;
        for(size_t i=0;i<posed.size();i+=3){posed[i]+=.4f;posed[i+1]+=.2f;posed[i+2]+=.6f;}
        float mask=.25f;uint32_t enabled=1,lock=0;
        DeformPipeline::BeginInfo info;
        info.points=pointsBuf;info.curveOffsets=offsetsBuf;info.rootTargets=targetsBuf;
        info.curveCount=2;info.pointCount=6;info.restSamples=restSamples;info.posedSamples=posed;info.sampleCount=5;
        info.mask={1,1,Upload(context,&mask,4),1};
        info.enabled={1,1,Upload(context,&enabled,4),1};
        info.lockRoots={1,1,Upload(context,&lock,4),1};
        auto c=pipe->Begin(info,&status);CHECK(c&&status==VK_SUCCESS&&Prove(native));
        DeformSemantic sem=DeformSemantic::Ok;CHECK(c->Poll(&sem)==VK_SUCCESS&&sem==DeformSemantic::Ok&&c->succeeded());
        std::vector<float> out;CHECK(ReadOutput(native,context,c,&out));
        for(size_t i=0;i<points.size();i+=3)
            CHECK(Near(V(out[i],out[i+1],out[i+2]),V(points[i]+.1f,points[i+1]+.05f,points[i+2]+.15f),1e-4f));
        enabled=0;info.enabled.data=Upload(context,&enabled,4);
        c=pipe->Begin(info,&status);CHECK(c&&Prove(native)&&c->Poll(&sem)==VK_SUCCESS&&c->succeeded());
        CHECK(ReadOutput(native,context,c,&out)&&out==points);
        std::puts("Groom connected fields broadcast over every curve: PASS");
    }

    // ---- Case 4: General non-rigid pose vs CPU oracle ----
    {
        // Non-rigid: each sample displaced by a different amount.
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i) {
            posed[i*3+0] = restSamples[i*3+0] + 0.1f * i;
            posed[i*3+1] = restSamples[i*3+1] + 0.05f * (i * i);
            posed[i*3+2] = restSamples[i*3+2] + 0.02f * i * (i + 1);
        }

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);

        // CPU oracle.
        auto params = makeParams();
        DeformApplyParams p = params;
        RbfState rstate;
        SolveRbf(reinterpret_cast<float3 const*>(restSamples.data()),
                 reinterpret_cast<float3 const*>(posed.data()),
                 5, 0.0, rstate);
        CHECK(rstate.status == RbfStatus::Code::Ok);
        std::vector<float3> warped(6);
        for (int i = 0; i < 6; ++i) {
            float3 pt = V(points[i*3], points[i*3+1], points[i*3+2]);
            warped[i] = RbfEvaluate(rstate, reinterpret_cast<float3 const*>(restSamples.data()), pt);
        }
        p.offsets = offsets.data();
        p.targets = reinterpret_cast<float3 const*>(rootTargets.data());
        p.curveCount = 2; p.pointCount = 6;
        std::vector<float3> expected;
        CHECK(RbfApply(p, reinterpret_cast<float3 const*>(points.data()),
                       warped.data(), &expected));
        for (int i = 0; i < 6; ++i) {
            float3 got = V(out[i*3], out[i*3+1], out[i*3+2]);
            CHECK(Near(got, expected[i], 2e-3f));
        }
        std::puts("Case 4 (non-rigid vs oracle): PASS");
    }

    // ---- Case 5: lockRoots = 1 → oracle with correction ----
    {
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i) {
            posed[i*3+0] = restSamples[i*3+0] + 0.05f * i;
            posed[i*3+1] = restSamples[i*3+1] + 0.03f * (i * i);
            posed[i*3+2] = restSamples[i*3+2] + 0.01f * i;
        }

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {1, 1, nullptr, 0};  // lock enabled
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);

        // CPU oracle with lock.
        DeformApplyParams p = makeParams(1);  // lockLiteral=1
        RbfState rstate;
        SolveRbf(reinterpret_cast<float3 const*>(restSamples.data()),
                 reinterpret_cast<float3 const*>(posed.data()),
                 5, 0.0, rstate);
        CHECK(rstate.status == RbfStatus::Code::Ok);
        std::vector<float3> warped(6);
        for (int i = 0; i < 6; ++i) {
            float3 pt = V(points[i*3], points[i*3+1], points[i*3+2]);
            warped[i] = RbfEvaluate(rstate, reinterpret_cast<float3 const*>(restSamples.data()), pt);
        }
        p.offsets = offsets.data();
        p.targets = reinterpret_cast<float3 const*>(rootTargets.data());
        p.curveCount = 2; p.pointCount = 6;
        std::vector<float3> expected;
        CHECK(RbfApply(p, reinterpret_cast<float3 const*>(points.data()),
                       warped.data(), &expected));
        for (int i = 0; i < 6; ++i) {
            float3 got = V(out[i*3], out[i*3+1], out[i*3+2]);
            CHECK(Near(got, expected[i], 2e-3f));
        }
        std::puts("Case 5 (lockRoots): PASS");
    }

    // ---- Case 6: mask literal 0.0 → bitwise input passthrough ----
    {
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i)
            for (int ax = 0; ax < 3; ++ax)
                posed[i*3+ax] = restSamples[i*3+ax] + 0.1f * i;

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {0.0f, 1, nullptr, 0};  // mask = 0
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);
        CHECK(std::memcmp(out.data(), points.data(), points.size() * sizeof(float)) == 0);
        std::puts("Case 6 (mask=0 passthrough): PASS");
    }

    // ---- Case 7: enabled literal 0 → bitwise input passthrough ----
    {
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i)
            for (int ax = 0; ax < 3; ++ax)
                posed[i*3+ax] = restSamples[i*3+ax] + 0.1f * i;

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {0, 1, nullptr, 0};  // disabled
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);
        CHECK(std::memcmp(out.data(), points.data(), points.size() * sizeof(float)) == 0);
        std::puts("Case 7 (enabled=0 passthrough): PASS");
    }

    // ---- Case 8: Device offsets and host sample validation ----
    // 8a: Bad initial/final endpoints, zero-length and decreasing segments
    // are semantic failures, discovered asynchronously from device memory.
    for (auto const& badOffsets : std::vector<std::vector<uint32_t>>{
            {1, 3, 6}, {0, 3, 5}, {0, 0, 6}, {0, 7, 6}, {0, 3, 3}}) {
        auto badOffBuf = UploadVulkanDeviceBytes(native, context, badOffsets.data(),
            badOffsets.size() * sizeof(uint32_t));
        CHECK(badOffBuf);

        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = badOffBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = restSamples;
        info.sampleCount = 5;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::BadOffsets && !c->succeeded());
        std::puts("Case 8a (reject bad offsets): PASS");
    }
    // 8b: NaN in restSamples → NonFinite
    {
        std::vector<float> nanSamples = restSamples;
        nanSamples[0] = std::nanf("");
        DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 6;
        info.restSamples = nanSamples;
        info.posedSamples = restSamples;
        info.sampleCount = 5;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(!c && status == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(sem == DeformSemantic::NonFinite);
        std::puts("Case 8b (reject NaN samples): PASS");
    }
    // 8c: the same coplanar rest twice rejects identically. The first Begin
    // binds and fails; the second replays the cached bind failure.
    {
        std::vector<float> flatSamples = restSamples;
        for (size_t i = 0; i < 5; ++i) flatSamples[i * 3 + 2] = 0.0f;
        for (int attempt = 0; attempt < 2; ++attempt) {
            DeformPipeline::BeginInfo info;
            info.points = pointsBuf;
            info.curveOffsets = offsetsBuf;
            info.rootTargets = targetsBuf;
            info.curveCount = 2;
            info.pointCount = 6;
            info.restSamples = flatSamples;
            info.posedSamples = restSamples;
            info.sampleCount = 5;
            info.mask = {1.0f, 1, nullptr, 0};
            info.enabled = {1, 1, nullptr, 0};
            info.lockRoots = {0, 1, nullptr, 0};

            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &status, &sem);
            CHECK(!c && status == VK_ERROR_INITIALIZATION_FAILED);
            CHECK(sem == DeformSemantic::RankDeficient);
        }
        std::puts("Case 8c (repeat coplanar rest rejects twice): PASS");
    }

    // ---- Case 9: ragged spans (1, 2, 3 points) with per-curve fields ----
    // The per-point apply binary-searches offsets for its curve; ragged
    // spans (including a span of 1) plus primitive-domain fields pin the
    // mapping (groom literals are curve-independent and cannot). Under a
    // rigid translation d: a primitive mask of [0, 1, 0] holds curves 0
    // and 2 at their input bitwise while curve 1 moves by d; a primitive
    // lock of [1, 0, 1] holds the locked curves at their roots while the
    // unlocked one moves by d.
    {
        const std::vector<float> rPoints{
            0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f,
            2.0f, 0.0f, 0.0f, 2.1f, 0.0f, 0.1f, 2.2f, 0.0f, 0.2f,
        };
        const std::vector<uint32_t> rOffsets{0, 1, 3, 6};
        const std::vector<float> rTargets{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
        };
        const std::vector<float> rMask{0.0f, 1.0f, 0.0f};
        const std::vector<uint32_t> rLock{1u, 0u, 1u};
        auto rPointsBuf = Upload(context, rPoints.data(), rPoints.size() * sizeof(float));
        auto rOffsetsBuf = UploadVulkanDeviceBytes(native, context, rOffsets.data(),
            rOffsets.size() * sizeof(uint32_t));
        auto rTargetsBuf = Upload(context, rTargets.data(), rTargets.size() * sizeof(float));
        auto rMaskBuf = Upload(context, rMask.data(), rMask.size() * sizeof(float));
        auto rLockBuf = Upload(context, rLock.data(), rLock.size() * sizeof(uint32_t));
        CHECK(rPointsBuf && rOffsetsBuf && rTargetsBuf && rMaskBuf && rLockBuf);
        const float d[3] = {0.3f, -0.2f, 0.5f};
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i)
            for (int ax = 0; ax < 3; ++ax)
                posed[size_t(i * 3 + ax)] = restSamples[size_t(i * 3 + ax)] + d[ax];

        // 9a: primitive mask [0, 1, 0].
        DeformPipeline::BeginInfo info;
        info.points = rPointsBuf;
        info.curveOffsets = rOffsetsBuf;
        info.rootTargets = rTargetsBuf;
        info.curveCount = 3;
        info.pointCount = 6;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {0.0f, 2, rMaskBuf, 3};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 18);
        // Mask-0 curves pass the input through bitwise.
        CHECK(std::memcmp(out.data(), rPoints.data(), 3 * sizeof(float)) == 0);
        CHECK(std::memcmp(out.data() + 9, rPoints.data() + 9, 9 * sizeof(float)) == 0);
        for (int i = 1; i < 3; ++i) {
            float3 got = V(out[size_t(i * 3)], out[size_t(i * 3 + 1)], out[size_t(i * 3 + 2)]);
            float3 exp = V(rPoints[size_t(i * 3)] + d[0], rPoints[size_t(i * 3 + 1)] + d[1],
                           rPoints[size_t(i * 3 + 2)] + d[2]);
            CHECK(Near(got, exp, 1e-4f));
        }
        std::puts("Case 9a (ragged primitive mask): PASS");

        // 9b: primitive lock [1, 0, 1].
        DeformPipeline::BeginInfo info2;
        info2.points = rPointsBuf;
        info2.curveOffsets = rOffsetsBuf;
        info2.rootTargets = rTargetsBuf;
        info2.curveCount = 3;
        info2.pointCount = 6;
        info2.restSamples = restSamples;
        info2.posedSamples = posed;
        info2.sampleCount = 5;
        info2.smoothing = 0.0;
        info2.mask = {1.0f, 1, nullptr, 0};
        info2.enabled = {1, 1, nullptr, 0};
        info2.lockRoots = {0, 2, rLockBuf, 3};
        info2.groomEnvelope = 1.0f;
        auto c2 = pipe->Begin(std::move(info2), &status, &sem);
        CHECK(c2 && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c2->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c2->succeeded());
        std::vector<float> out2;
        CHECK(ReadOutput(native, context, c2, &out2));
        CHECK(out2.size() == 18);
        for (int i = 0; i < 6; ++i) {
            bool const locked = (i < 1 || i >= 3);
            float3 got = V(out2[size_t(i * 3)], out2[size_t(i * 3 + 1)], out2[size_t(i * 3 + 2)]);
            float3 exp = locked ? V(rPoints[size_t(i * 3)], rPoints[size_t(i * 3 + 1)],
                                          rPoints[size_t(i * 3 + 2)])
                                : V(rPoints[size_t(i * 3)] + d[0], rPoints[size_t(i * 3 + 1)] + d[1],
                                    rPoints[size_t(i * 3 + 2)] + d[2]);
            CHECK(Near(got, exp, 1e-4f));
        }
        std::puts("Case 9b (ragged primitive lock): PASS");
    }

    // ---- Case 9c: ragged spans (1, 2, 4 points), indivisible count ----
    // 7 points over 3 curves leave a remainder, so the stride candidate
    // is 0 and every point runs the pure binary search (cases 9a/9b
    // divide evenly and mix division with the search fallback). Same
    // rigid translation and primitive mask shape as 9a.
    {
        const std::vector<float> rPoints{
            0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f,
            2.0f, 0.0f, 0.0f, 2.1f, 0.0f, 0.1f, 2.2f, 0.0f, 0.2f, 2.3f, 0.0f, 0.3f,
        };
        const std::vector<uint32_t> rOffsets{0, 1, 3, 7};
        const std::vector<float> rTargets{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f,
        };
        const std::vector<float> rMask{0.0f, 1.0f, 0.0f};
        auto rPointsBuf = Upload(context, rPoints.data(), rPoints.size() * sizeof(float));
        auto rOffsetsBuf = UploadVulkanDeviceBytes(native, context, rOffsets.data(),
            rOffsets.size() * sizeof(uint32_t));
        auto rTargetsBuf = Upload(context, rTargets.data(), rTargets.size() * sizeof(float));
        auto rMaskBuf = Upload(context, rMask.data(), rMask.size() * sizeof(float));
        CHECK(rPointsBuf && rOffsetsBuf && rTargetsBuf && rMaskBuf);
        const float d[3] = {0.3f, -0.2f, 0.5f};
        std::vector<float> posed(15);
        for (int i = 0; i < 5; ++i)
            for (int ax = 0; ax < 3; ++ax)
                posed[size_t(i * 3 + ax)] = restSamples[size_t(i * 3 + ax)] + d[ax];

        DeformPipeline::BeginInfo info;
        info.points = rPointsBuf;
        info.curveOffsets = rOffsetsBuf;
        info.rootTargets = rTargetsBuf;
        info.curveCount = 3;
        info.pointCount = 7;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = 5;
        info.smoothing = 0.0;
        info.mask = {0.0f, 2, rMaskBuf, 3};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;

        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == DeformSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(ReadOutput(native, context, c, &out));
        CHECK(out.size() == 21);
        // Mask-0 curves pass the input through bitwise.
        CHECK(std::memcmp(out.data(), rPoints.data(), 3 * sizeof(float)) == 0);
        CHECK(std::memcmp(out.data() + 9, rPoints.data() + 9, 12 * sizeof(float)) == 0);
        for (int i = 1; i < 3; ++i) {
            float3 got = V(out[size_t(i * 3)], out[size_t(i * 3 + 1)], out[size_t(i * 3 + 2)]);
            float3 exp = V(rPoints[size_t(i * 3)] + d[0], rPoints[size_t(i * 3 + 1)] + d[1],
                           rPoints[size_t(i * 3 + 2)] + d[2]);
            CHECK(Near(got, exp, 1e-4f));
        }
        std::puts("Case 9c (ragged indivisible mask): PASS");
    }

    // ---- Cases 11-18: cached-R evaluate ----
    // One cached pipe on a funded context; each case compares the cached
    // path against the direct path (via the disable seam) bitwise and
    // asserts the hit/miss/bypass/unfunded counter deltas.
    {
        DeviceContext::CreateInfo bigCi;
        bigCi.instance = native->instance;
        bigCi.physicalDevice = native->physical;
        bigCi.device = native->device;
        bigCi.computeQueue = native->queue;
        bigCi.computeQueueFamily = native->family;
        bigCi.physicalIndex = native->physicalIndex;
        bigCi.resourceDeviceId = 8030;
        bigCi.nativeLifetime = native;
        bigCi.resources = {size_t{1024} << 20, 0};
        bigCi.shaderFloat64Enabled = true;
        auto bigContext = DeviceContext::Create(bigCi);
        CHECK(bigContext);
        VkResult cstatus = VK_SUCCESS;
        std::shared_ptr<DeformPipeline> cachedPipe;
        auto freshPipe = [&]() {
            cachedPipe = DeformPipeline::Create(bigContext, evalSpv, applySpv, &cstatus, cacheSpv);
            return cachedPipe && cstatus == VK_SUCCESS;
        };
        CHECK(freshPipe());
        // A partial cache set is a wiring bug, failed loudly.
        DeformEvalCacheSpirv partial = cacheSpv;
        partial.fill.clear();
        CHECK(!DeformPipeline::Create(bigContext, evalSpv, applySpv, &cstatus, partial));

        auto bigUpload = [&](void const* data, VkDeviceSize bytes) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes ? bytes : 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto b = ChargedBuffer::Create(bigContext, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            if (!b) return std::shared_ptr<ChargedBuffer>{};
            if (bytes) {
                void* p = nullptr;
                if (vkMapMemory(bigContext->device(), b->memory(), 0, bytes, 0, &p) !=
                    VK_SUCCESS)
                    return std::shared_ptr<ChargedBuffer>{};
                std::memcpy(p, data, bytes);
                vkUnmapMemory(bigContext->device(), b->memory());
            }
            return b;
        };
        auto rewrite = [&](std::shared_ptr<ChargedBuffer> const& b, void const* data,
                            VkDeviceSize bytes) {
            void* p = nullptr;
            if (vkMapMemory(bigContext->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS)
                return false;
            std::memcpy(p, data, bytes);
            vkUnmapMemory(bigContext->device(), b->memory());
            return true;
        };
        auto runPose = [&](DeformPipeline::BeginInfo info, std::vector<float>* out,
                            DeformSemantic* semOut) {
            DeformSemantic sem = DeformSemantic::Ok;
            VkResult st = VK_SUCCESS;
            auto c = cachedPipe->Begin(std::move(info), &st, &sem);
            if (!c || st != VK_SUCCESS) return false;
            if (!Prove(native)) return false;
            if (c->Poll(&sem) != VK_SUCCESS) return false;
            if (semOut) *semOut = sem;
            if (sem != DeformSemantic::Ok || !c->succeeded()) return false;
            return ReadOutput(native, bigContext, c, out);
        };
        auto runDirect = [&](DeformPipeline::BeginInfo info, std::vector<float>* out) {
            TestDisableDeformEvalCache(true);
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = cachedPipe->Begin(std::move(info), &cstatus, &sem);
            bool ok = c && cstatus == VK_SUCCESS && Prove(native) &&
                c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok && c->succeeded() &&
                ReadOutput(native, bigContext, c, out);
            TestDisableDeformEvalCache(false);
            return ok;
        };
        auto bitEq = [](std::vector<float> const& a, std::vector<float> const& b) {
            return a.size() == b.size() &&
                std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };

        // 12 CVs over 4 curves, n=5 (the file's rest set).
        const std::vector<float> cPoints{
            0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.2f, 0.0f, 0.2f,
            1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f, 1.2f, 0.0f, 0.2f,
            2.0f, 0.0f, 0.0f, 2.1f, 0.0f, 0.1f, 2.2f, 0.0f, 0.2f,
            3.0f, 0.0f, 0.0f, 3.1f, 0.0f, 0.1f, 3.2f, 0.0f, 0.2f,
        };
        const std::vector<uint32_t> cOffsets{0, 3, 6, 9, 12};
        const std::vector<float> cTargets{
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            2.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f,
        };
        std::vector<float> cPosed = restSamples;
        for (size_t i = 0; i < cPosed.size(); i += 3) cPosed[i] += 0.5f;
        auto cPointsBuf = bigUpload(cPoints.data(), cPoints.size() * sizeof(float));
        auto cOffsetsBuf = UploadVulkanDeviceBytes(native, bigContext, cOffsets.data(),
            cOffsets.size() * sizeof(uint32_t));
        auto cTargetsBuf = bigUpload(cTargets.data(), cTargets.size() * sizeof(float));
        CHECK(cPointsBuf && cOffsetsBuf && cTargetsBuf);
        auto makeInfo = [&](std::vector<float> const& rest, std::vector<float> const& posed,
                            int n) {
            DeformPipeline::BeginInfo info;
            info.points = cPointsBuf;
            info.curveOffsets = cOffsetsBuf;
            info.rootTargets = cTargetsBuf;
            info.curveCount = 4;
            info.pointCount = 12;
            info.restSamples = rest;
            info.posedSamples = posed;
            info.sampleCount = n;
            info.smoothing = 0.0;
            info.mask = {1.0f, 1, nullptr, 0};
            info.enabled = {1, 1, nullptr, 0};
            info.lockRoots = {0, 1, nullptr, 0};
            info.groomEnvelope = 1.0f;
            return info;
        };

        // Case 11: fill then hit, both bitwise the direct path.
        {
            uint64_t h0 = DeformEvalCacheHitsForTesting();
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            std::vector<float> fillOut, hitOut, directOut;
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &fillOut, nullptr));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &hitOut, nullptr));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 1);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 1);
            CHECK(runDirect(makeInfo(restSamples, cPosed, 5), &directOut));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 1);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 1);
            CHECK(bitEq(fillOut, directOut));
            CHECK(bitEq(hitOut, directOut));
            std::puts("Case 11 (cache fill/hit bitwise direct): PASS");
        }

        // Case 12: a CV change (same buffer, rewritten bytes) mismatch-refills,
        // bitwise direct. The repeat between the fills resets the miss
        // streak so no bypass engages.
        {
            CHECK(freshPipe());
            uint64_t h0 = DeformEvalCacheHitsForTesting();
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            uint64_t b0 = DeformEvalCacheBypassedForTesting();
            std::vector<float> moved = cPoints;
            for (size_t i = 0; i < moved.size(); i += 3) moved[i] += 0.25f;
            CHECK(rewrite(cPointsBuf, moved.data(), moved.size() * sizeof(float)));
            std::vector<float> fillOut, hitOut, refillOut, hit2Out, directOut;
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &fillOut, nullptr));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &hitOut, nullptr));
            CHECK(rewrite(cPointsBuf, cPoints.data(), cPoints.size() * sizeof(float)));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &refillOut, nullptr));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &hit2Out, nullptr));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 2);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 2);
            CHECK(DeformEvalCacheBypassedForTesting() - b0 == 0);
            CHECK(runDirect(makeInfo(restSamples, cPosed, 5), &directOut));
            CHECK(bitEq(refillOut, directOut));
            CHECK(bitEq(hit2Out, directOut));
            std::puts("Case 12 (CV refill bitwise direct): PASS");
        }

        // Case 13: rest, n-grow, and n-shrink changes force, bitwise direct.
        // Each force is followed by a repeat hit so no bypass engages.
        {
            CHECK(freshPipe());
            uint64_t h0 = DeformEvalCacheHitsForTesting();
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            uint64_t b0 = DeformEvalCacheBypassedForTesting();
            std::vector<float> rest2 = restSamples;
            rest2[0] += 0.125f;
            std::vector<float> posed2 = cPosed;
            posed2[0] += 0.125f;
            std::vector<float> got, direct;
            CHECK(runPose(makeInfo(rest2, posed2, 5), &got, nullptr));
            CHECK(runPose(makeInfo(rest2, posed2, 5), &got, nullptr));
            CHECK(runDirect(makeInfo(rest2, posed2, 5), &direct));
            CHECK(bitEq(got, direct));
            // n = 8 (grow: R reallocates) then n = 5 (shrink: rows reused).
            std::vector<float> rest8 = restSamples;
            const float extra[9] = {2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f};
            rest8.insert(rest8.end(), extra, extra + 9);
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.5f;
            CHECK(runPose(makeInfo(rest8, posed8, 8), &got, nullptr));
            CHECK(runPose(makeInfo(rest8, posed8, 8), &got, nullptr));
            CHECK(runDirect(makeInfo(rest8, posed8, 8), &direct));
            CHECK(bitEq(got, direct));
            CHECK(runPose(makeInfo(rest2, posed2, 5), &got, nullptr));
            CHECK(runPose(makeInfo(rest2, posed2, 5), &got, nullptr));
            CHECK(runDirect(makeInfo(rest2, posed2, 5), &direct));
            CHECK(bitEq(got, direct));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 3);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 3);
            CHECK(DeformEvalCacheBypassedForTesting() - b0 == 0);
            std::puts("Case 13 (rest/n-change force bitwise direct): PASS");
        }

        // Case 14: point-count growth forces, bitwise direct.
        {
            CHECK(freshPipe());
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            std::vector<float> pts24 = cPoints;
            pts24.insert(pts24.end(), cPoints.begin(), cPoints.end());
            std::vector<uint32_t> off24{0, 3, 6, 9, 12, 15, 18, 21, 24};
            std::vector<float> tgt24 = cTargets;
            tgt24.insert(tgt24.end(), cTargets.begin(), cTargets.end());
            auto ptsBuf = bigUpload(pts24.data(), pts24.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, bigContext, off24.data(),
                off24.size() * sizeof(uint32_t));
            auto tgtBuf = bigUpload(tgt24.data(), tgt24.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto info24 = makeInfo(restSamples, cPosed, 5);
            info24.points = ptsBuf;
            info24.curveOffsets = offBuf;
            info24.rootTargets = tgtBuf;
            info24.curveCount = 8;
            info24.pointCount = 24;
            std::vector<float> got, direct;
            // Fill at 12 points first so the 24-point pose is a growth
            // force (24 > filled 12), not just a first-use force.
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
            CHECK(runPose(info24, &got, nullptr));
            auto direct24 = makeInfo(restSamples, cPosed, 5);
            direct24.points = ptsBuf;
            direct24.curveOffsets = offBuf;
            direct24.rootTargets = tgtBuf;
            direct24.curveCount = 8;
            direct24.pointCount = 24;
            CHECK(runDirect(std::move(direct24), &direct));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 2);
            CHECK(bitEq(got, direct));
            std::puts("Case 14 (growth force bitwise direct): PASS");
        }

        // Case 15: NaN CVs flag NonFinite on the cached path, then recovery.
        {
            CHECK(freshPipe());
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            uint64_t b0 = DeformEvalCacheBypassedForTesting();
            std::vector<float> nanPts = cPoints;
            nanPts[4] = std::numeric_limits<float>::quiet_NaN();
            CHECK(rewrite(cPointsBuf, nanPts.data(), nanPts.size() * sizeof(float)));
            DeformSemantic sem = DeformSemantic::Ok;
            VkResult st = VK_SUCCESS;
            auto c = cachedPipe->Begin(makeInfo(restSamples, cPosed, 5), &st, &sem);
            CHECK(!c && sem == DeformSemantic::NonFinite);
            CHECK(rewrite(cPointsBuf, cPoints.data(), cPoints.size() * sizeof(float)));
            // Recovery: the NaN fill neither poisons the cache nor the
            // output. (Two consecutive refills arm the bypass, so the
            // second recovery pose runs direct; both agree bitwise.)
            std::vector<float> got, direct;
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
            CHECK(runDirect(makeInfo(restSamples, cPosed, 5), &direct));
            CHECK(bitEq(got, direct));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
            CHECK(bitEq(got, direct));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 2);
            CHECK(DeformEvalCacheBypassedForTesting() - b0 == 1);
            std::puts("Case 15 (NaN flags, then recovers): PASS");
        }

        // Case 16: two consecutive refills bypass; a proof-matching probe resumes.
        {
            CHECK(freshPipe());
            uint64_t h0 = DeformEvalCacheHitsForTesting();
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            uint64_t b0 = DeformEvalCacheBypassedForTesting();
            std::vector<float> cvA = cPoints, cvB = cPoints;
            for (size_t i = 0; i < cvB.size(); i += 3) cvB[i] += 1.5f;
            CHECK(rewrite(cPointsBuf, cvA.data(), cvA.size() * sizeof(float)));
            std::vector<float> got, direct;
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
            CHECK(runDirect(makeInfo(restSamples, cPosed, 5), &direct));
            CHECK(bitEq(got, direct));
            CHECK(rewrite(cPointsBuf, cvB.data(), cvB.size() * sizeof(float)));
            CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
            auto infoB = makeInfo(restSamples, cPosed, 5);
            CHECK(runDirect(std::move(infoB), &direct));
            CHECK(bitEq(got, direct));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 2);
            // 32 churning poses run direct under backoff.
            for (int k = 0; k < 32; ++k) {
                std::vector<float> cvK = cPoints;
                for (size_t i = 0; i < cvK.size(); i += 3)
                    cvK[i] += 3.0f + 0.5f * float(k);
                CHECK(rewrite(cPointsBuf, cvK.data(), cvK.size() * sizeof(float)));
                CHECK(runPose(makeInfo(restSamples, cPosed, 5), &got, nullptr));
                auto infoK = makeInfo(restSamples, cPosed, 5);
                CHECK(runDirect(std::move(infoK), &direct));
                CHECK(bitEq(got, direct));
            }
            CHECK(DeformEvalCacheBypassedForTesting() - b0 == 32);
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 2);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 0);
            // Probe with the last-filled CVs: verify matches, hits resume.
            CHECK(rewrite(cPointsBuf, cvB.data(), cvB.size() * sizeof(float)));
            auto infoP = makeInfo(restSamples, cPosed, 5);
            CHECK(runPose(std::move(infoP), &got, nullptr));
            auto infoQ = makeInfo(restSamples, cPosed, 5);
            CHECK(runDirect(std::move(infoQ), &direct));
            CHECK(bitEq(got, direct));
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 1);
            CHECK(rewrite(cPointsBuf, cPoints.data(), cPoints.size() * sizeof(float)));
            std::puts("Case 16 (bypass then probe resumes): PASS");
        }

        // Case 17: past-prefix suffix (600k points) bitwise direct. The
        // 48MiB budget funds only a ~340k prefix at n=8 (~8.1MiB of inputs
        // plus ~14.4MiB of pose scratch charge first, then ~24.5MiB funds
        // R+proof at 76B/CV), so the suffix dispatch covers the remaining
        // ~260k points; the funded-prefix seam pins that the suffix ran
        // (a full prefix would silently drop that coverage).
        {
            DeviceContext::CreateInfo suffixCi;
            suffixCi.instance = native->instance;
            suffixCi.physicalDevice = native->physical;
            suffixCi.device = native->device;
            suffixCi.computeQueue = native->queue;
            suffixCi.computeQueueFamily = native->family;
            suffixCi.physicalIndex = native->physicalIndex;
            suffixCi.resourceDeviceId = 8031;
            suffixCi.nativeLifetime = native;
            suffixCi.resources = {size_t{48} << 20, 0};
            suffixCi.shaderFloat64Enabled = true;
            auto suffixContext = DeviceContext::Create(suffixCi);
            CHECK(suffixContext);
            VkResult sstatus = VK_SUCCESS;
            auto suffixPipe = DeformPipeline::Create(suffixContext, evalSpv, applySpv,
                &sstatus, cacheSpv);
            CHECK(suffixPipe && sstatus == VK_SUCCESS);
            auto runSuffix = [&](DeformPipeline::BeginInfo info,
                                 std::vector<float>* out) {
                DeformSemantic sem = DeformSemantic::Ok;
                VkResult st = VK_SUCCESS;
                auto c = suffixPipe->Begin(std::move(info), &st, &sem);
                if (!c || st != VK_SUCCESS) return false;
                if (!Prove(native)) return false;
                if (c->Poll(&sem) != VK_SUCCESS) return false;
                if (sem != DeformSemantic::Ok || !c->succeeded()) return false;
                return ReadOutput(native, suffixContext, c, out);
            };
            auto runSuffixDirect = [&](DeformPipeline::BeginInfo info,
                                       std::vector<float>* out) {
                TestDisableDeformEvalCache(true);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = suffixPipe->Begin(std::move(info), &sstatus, &sem);
                bool ok = c && sstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, suffixContext, c, out);
                TestDisableDeformEvalCache(false);
                return ok;
            };
            uint64_t h0 = DeformEvalCacheHitsForTesting();
            uint64_t m0 = DeformEvalCacheMissesForTesting();
            // Forced GPU: this case pins exact cache accounting, and the
            // hetero balance cap (plus engagement) is timing-dependent.
            // Hetero-vs-GPU bitwise equivalence is cases 20-22's job.
            TestForceDeformHeteroSuffix(-1);
            uint32_t const curves = 60000, perCurve = 10, points = curves * perCurve;
            int const n = 8;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x12345678u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = Upload(suffixContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, suffixContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(suffixContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkBig = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            std::vector<float> fillOut, hitOut, directOut;
            CHECK(runSuffix(mkBig(), &fillOut));
            CHECK(runSuffix(mkBig(), &hitOut));
            CHECK(DeformEvalCacheMissesForTesting() - m0 == 1);
            CHECK(DeformEvalCacheHitsForTesting() - h0 == 1);
            uint32_t const funded = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded >= 4096 && funded < points);
            CHECK(runSuffixDirect(mkBig(), &directOut));
            CHECK(bitEq(fillOut, directOut));
            CHECK(bitEq(hitOut, directOut));
            TestForceDeformHeteroSuffix(0);
            std::puts("Case 17 (600k suffix bitwise direct): PASS");
        }

        // Case 18: over budget runs direct (unfunded), bitwise direct.
        // The 1MiB budget cannot fund the 4096-CV floor at n=100 (the 1MiB
        // funding margin alone meets what is free), so the pose runs the
        // direct shader with no vkAllocateMemory probes.
        {
            uint64_t u0 = DeformEvalCacheUnfundedForTesting();
            DeviceContext::CreateInfo tinyCi;
            tinyCi.instance = native->instance;
            tinyCi.physicalDevice = native->physical;
            tinyCi.device = native->device;
            tinyCi.computeQueue = native->queue;
            tinyCi.computeQueueFamily = native->family;
            tinyCi.physicalIndex = native->physicalIndex;
            tinyCi.resourceDeviceId = 8032;
            tinyCi.nativeLifetime = native;
            tinyCi.resources = {size_t{1} << 20, 0};
            tinyCi.shaderFloat64Enabled = true;
            auto tinyContext = DeviceContext::Create(tinyCi);
            CHECK(tinyContext);
            VkResult st2 = VK_SUCCESS;
            auto smallPipe =
                DeformPipeline::Create(tinyContext, evalSpv, applySpv, &st2, cacheSpv);
            CHECK(smallPipe && st2 == VK_SUCCESS);
            const std::vector<float> tinyPts{
                0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.2f, 0.0f, 0.2f,
                1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f, 1.2f, 0.0f, 0.2f,
            };
            const std::vector<uint32_t> tinyOff{0, 3, 6};
            const std::vector<float> tinyTgt{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            };
            auto tinyPtsBuf =
                Upload(tinyContext, tinyPts.data(), tinyPts.size() * sizeof(float));
            auto tinyOffBuf = UploadVulkanDeviceBytes(native, tinyContext, tinyOff.data(),
                tinyOff.size() * sizeof(uint32_t));
            auto tinyTgtBuf =
                Upload(tinyContext, tinyTgt.data(), tinyTgt.size() * sizeof(float));
            CHECK(tinyPtsBuf && tinyOffBuf && tinyTgtBuf);
            std::vector<float> rest100(size_t(100) * 3);
            for (int i = 0; i < 100; ++i) {
                rest100[size_t(i) * 3] = float(i % 10);
                rest100[size_t(i) * 3 + 1] = float((i / 10) % 10);
                rest100[size_t(i) * 3 + 2] = float(i % 3);
            }
            std::vector<float> posed100 = rest100;
            for (size_t i = 0; i < posed100.size(); i += 3) posed100[i] += 0.1f;
            DeformPipeline::BeginInfo info;
            info.points = tinyPtsBuf;
            info.curveOffsets = tinyOffBuf;
            info.rootTargets = tinyTgtBuf;
            info.curveCount = 2;
            info.pointCount = 6;
            info.restSamples = rest100;
            info.posedSamples = posed100;
            info.sampleCount = 100;
            info.smoothing = 0.0;
            info.mask = {1.0f, 1, nullptr, 0};
            info.enabled = {1, 1, nullptr, 0};
            info.lockRoots = {0, 1, nullptr, 0};
            info.groomEnvelope = 1.0f;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = smallPipe->Begin(info, &st2, &sem);
            CHECK(c && st2 == VK_SUCCESS);
            CHECK(Prove(native));
            CHECK(c->Poll(&sem) == VK_SUCCESS);
            CHECK(sem == DeformSemantic::Ok && c->succeeded());
            std::vector<float> got;
            CHECK(ReadOutput(native, tinyContext, c, &got));
            CHECK(DeformEvalCacheUnfundedForTesting() - u0 == 1);
            // The disable seam on the same pipe agrees bitwise.
            TestDisableDeformEvalCache(true);
            DeformSemantic sem2 = DeformSemantic::Ok;
            auto c2 = smallPipe->Begin(std::move(info), &st2, &sem2);
            bool directOk = c2 && st2 == VK_SUCCESS && Prove(native) &&
                c2->Poll(&sem2) == VK_SUCCESS && sem2 == DeformSemantic::Ok &&
                c2->succeeded();
            std::vector<float> direct;
            directOk = directOk && ReadOutput(native, tinyContext, c2, &direct);
            TestDisableDeformEvalCache(false);
            CHECK(directOk);
            CHECK(bitEq(got, direct));
            std::puts("Case 18 (unfunded runs direct): PASS");
        }

        // Case 19: a pose its own cache starves evicts and succeeds. The
        // 20MiB budget funds a ~12k prefix for the 200k-point pose (~9.6MiB
        // of inputs and scratch charge first at n=100), leaving ~1MiB free;
        // the 150k-point pose's scratch then misses the pool checkout and
        // would fail, so Begin evicts the cache, retries, and refounds a
        // smaller prefix instead of failing. Bitwise direct throughout.
        {
            DeviceContext::CreateInfo evictCi;
            evictCi.instance = native->instance;
            evictCi.physicalDevice = native->physical;
            evictCi.device = native->device;
            evictCi.computeQueue = native->queue;
            evictCi.computeQueueFamily = native->family;
            evictCi.physicalIndex = native->physicalIndex;
            evictCi.resourceDeviceId = 8033;
            evictCi.nativeLifetime = native;
            evictCi.resources = {size_t{20} << 20, 0};
            evictCi.shaderFloat64Enabled = true;
            auto evictContext = DeviceContext::Create(evictCi);
            CHECK(evictContext);
            VkResult estatus = VK_SUCCESS;
            auto evictPipe = DeformPipeline::Create(evictContext, evalSpv, applySpv,
                &estatus, cacheSpv);
            CHECK(evictPipe && estatus == VK_SUCCESS);
            int const n = 100;
            std::vector<float> rest100(size_t(n) * 3);
            for (int i = 0; i < n; ++i) {
                rest100[size_t(i) * 3] = float(i % 10);
                rest100[size_t(i) * 3 + 1] = float((i / 10) % 10);
                rest100[size_t(i) * 3 + 2] = float(i % 3);
            }
            std::vector<float> posed100 = rest100;
            for (size_t i = 0; i < posed100.size(); i += 3) posed100[i] += 0.1f;
            auto makeShape = [&](uint32_t curves, uint32_t perCurve) {
                DeformPipeline::BeginInfo info;
                uint32_t const points = curves * perCurve;
                std::vector<float> pts(size_t(points) * 3);
                uint64_t state = 0x9e3779b9u;
                for (auto& v : pts) {
                    state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                    v = float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
                }
                std::vector<uint32_t> off(curves + 1);
                for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
                std::vector<float> tgt(size_t(curves) * 3);
                for (uint32_t i = 0; i < curves; ++i)
                    for (int a = 0; a < 3; ++a)
                        tgt[size_t(i) * 3 + size_t(a)] =
                            pts[size_t(i) * perCurve * 3 + size_t(a)];
                info.points = Upload(evictContext, pts.data(), pts.size() * sizeof(float));
                info.curveOffsets = UploadVulkanDeviceBytes(native, evictContext, off.data(),
                    off.size() * sizeof(uint32_t));
                info.rootTargets =
                    Upload(evictContext, tgt.data(), tgt.size() * sizeof(float));
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest100;
                info.posedSamples = posed100;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            auto runEvict = [&](DeformPipeline::BeginInfo info,
                                std::vector<float>* out) {
                DeformSemantic sem = DeformSemantic::Ok;
                VkResult st = VK_SUCCESS;
                auto c = evictPipe->Begin(std::move(info), &st, &sem);
                if (!c || st != VK_SUCCESS) return false;
                if (!Prove(native)) return false;
                if (c->Poll(&sem) != VK_SUCCESS) return false;
                if (sem != DeformSemantic::Ok || !c->succeeded()) return false;
                return ReadOutput(native, evictContext, c, out);
            };
            auto info1 = makeShape(20000, 10);
            auto info2 = makeShape(15000, 10);
            CHECK(info1.points && info1.curveOffsets && info1.rootTargets);
            CHECK(info2.points && info2.curveOffsets && info2.rootTargets);
            // Saved for the direct comparison (no re-upload: the budget is full).
            auto pts2 = info2.points;
            auto off2 = info2.curveOffsets;
            auto tgt2 = info2.rootTargets;
            uint64_t u0 = DeformEvalCacheUnfundedForTesting();
            std::vector<float> got1, got2;
            // By copy: both shapes' inputs stay alive across both poses,
            // so pose 2's scratch genuinely does not fit next to the cache
            // pose 1 funded (moved-from inputs would free early and the
            // eviction would never engage).
            CHECK(runEvict(info1, &got1));
            uint32_t const funded1 = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded1 >= 10000 && funded1 < 200000);
            CHECK(runEvict(info2, &got2));
            // Exactly the eviction: the refound pose runs cached, not direct.
            CHECK(DeformEvalCacheUnfundedForTesting() - u0 == 1);
            uint32_t const funded2 = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded2 >= 4096 && funded2 < 150000);
            DeformPipeline::BeginInfo directInfo;
            directInfo.points = pts2;
            directInfo.curveOffsets = off2;
            directInfo.rootTargets = tgt2;
            directInfo.curveCount = 15000;
            directInfo.pointCount = 150000;
            directInfo.restSamples = rest100;
            directInfo.posedSamples = posed100;
            directInfo.sampleCount = n;
            directInfo.smoothing = 0.0;
            directInfo.mask = {1.0f, 1, nullptr, 0};
            directInfo.enabled = {1, 1, nullptr, 0};
            directInfo.lockRoots = {0, 1, nullptr, 0};
            directInfo.groomEnvelope = 1.0f;
            TestDisableDeformEvalCache(true);
            DeformSemantic sem2 = DeformSemantic::Ok;
            auto c2 = evictPipe->Begin(std::move(directInfo), &estatus, &sem2);
            bool directOk = c2 && estatus == VK_SUCCESS && Prove(native) &&
                c2->Poll(&sem2) == VK_SUCCESS && sem2 == DeformSemantic::Ok &&
                c2->succeeded();
            std::vector<float> direct;
            directOk = directOk && ReadOutput(native, evictContext, c2, &direct);
            TestDisableDeformEvalCache(false);
            CHECK(directOk);
            CHECK(bitEq(got2, direct));
            std::puts("Case 19 (eviction succeeds, bitwise direct): PASS");
        }

        // Case 20: forced hetero vs forced GPU bitwise on the 600k/48MiB
        // suffix shape (fill + hit). The run counter pins that both hetero
        // poses ran the host suffix; the funded-prefix seam pins that a
        // suffix existed to run.
        {
            DeviceContext::CreateInfo hetCi;
            hetCi.instance = native->instance;
            hetCi.physicalDevice = native->physical;
            hetCi.device = native->device;
            hetCi.computeQueue = native->queue;
            hetCi.computeQueueFamily = native->family;
            hetCi.physicalIndex = native->physicalIndex;
            hetCi.resourceDeviceId = 8034;
            hetCi.nativeLifetime = native;
            hetCi.resources = {size_t{48} << 20, 0};
            hetCi.shaderFloat64Enabled = true;
            auto hetContext = DeviceContext::Create(hetCi);
            CHECK(hetContext);
            VkResult hstatus = VK_SUCCESS;
            auto hetPipe = DeformPipeline::Create(hetContext, evalSpv, applySpv,
                &hstatus, cacheSpv);
            CHECK(hetPipe && hstatus == VK_SUCCESS);
            auto runForced = [&](int force, DeformPipeline::BeginInfo info,
                                 std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = hetPipe->Begin(std::move(info), &hstatus, &sem);
                bool ok = c && hstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, hetContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            uint32_t const curves = 60000, perCurve = 10, points = curves * perCurve;
            int const n = 8;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x12345678u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = Upload(hetContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, hetContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(hetContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            // Host-visible points: this case covers the direct-map hetero
            // path (case 20b covers the staged copy).
            CHECK((ptsBuf->memoryPropertyFlags() &
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            auto mkHet = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> hetFill, hetHit, gpuHit;
            CHECK(runForced(1, mkHet(), &hetFill));
            CHECK(runForced(1, mkHet(), &hetHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            uint32_t const funded = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded >= 4096 && funded < points);
            CHECK(runForced(-1, mkHet(), &gpuHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            CHECK(bitEq(hetFill, gpuHit));
            CHECK(bitEq(hetHit, gpuHit));
            std::puts("Case 20 (forced hetero bitwise forced GPU): PASS");
        }

        // Case 20b: staged-copy hetero. Case 20's points are host-visible,
        // so they map directly; device-local points take the staging copy
        // instead. Same forced-hetero-vs-forced-GPU bitwise bar on the
        // 600k/48MiB suffix shape, plus a flags check pinning that the
        // points really are device-only (else this would re-cover the
        // direct path).
        {
            DeviceContext::CreateInfo dlCi;
            dlCi.instance = native->instance;
            dlCi.physicalDevice = native->physical;
            dlCi.device = native->device;
            dlCi.computeQueue = native->queue;
            dlCi.computeQueueFamily = native->family;
            dlCi.physicalIndex = native->physicalIndex;
            dlCi.resourceDeviceId = 8041;
            dlCi.nativeLifetime = native;
            dlCi.resources = {size_t{48} << 20, 0};
            dlCi.shaderFloat64Enabled = true;
            auto dlContext = DeviceContext::Create(dlCi);
            CHECK(dlContext);
            VkResult dlstatus = VK_SUCCESS;
            auto dlPipe = DeformPipeline::Create(dlContext, evalSpv, applySpv,
                &dlstatus, cacheSpv);
            CHECK(dlPipe && dlstatus == VK_SUCCESS);
            auto runForcedDl = [&](int force, DeformPipeline::BeginInfo info,
                                    std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = dlPipe->Begin(std::move(info), &dlstatus, &sem);
                bool ok = c && dlstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, dlContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            auto bitEqDl = [](std::vector<float> const& a, std::vector<float> const& b) {
                return a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
            };
            uint32_t const curves = 60000, perCurve = 10, points = curves * perCurve;
            int const n = 8;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x12345678u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = UploadVulkanDeviceBytes(native, dlContext, pts.data(),
                pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, dlContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(dlContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            CHECK((ptsBuf->memoryPropertyFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0);
            auto mkHetDl = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> hetFill, hetHit, gpuHit;
            CHECK(runForcedDl(1, mkHetDl(), &hetFill));
            CHECK(runForcedDl(1, mkHetDl(), &hetHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            uint32_t const funded = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded >= 4096 && funded < points);
            CHECK(runForcedDl(-1, mkHetDl(), &gpuHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            CHECK(bitEqDl(hetFill, gpuHit));
            CHECK(bitEqDl(hetHit, gpuHit));
            std::puts("Case 20b (staged-copy hetero bitwise forced GPU): PASS");
        }

        // Case 21: adversarial hetero. 100007 points (odd: block + TBB tails),
        // n=100, exact zeros, subnormals-adjacent tinies, larges, and
        // near-coincident CVs in both the prefix and the suffix.
        {
            DeviceContext::CreateInfo advCi;
            advCi.instance = native->instance;
            advCi.physicalDevice = native->physical;
            advCi.device = native->device;
            advCi.computeQueue = native->queue;
            advCi.computeQueueFamily = native->family;
            advCi.physicalIndex = native->physicalIndex;
            advCi.resourceDeviceId = 8035;
            advCi.nativeLifetime = native;
            advCi.resources = {size_t{32} << 20, 0};
            advCi.shaderFloat64Enabled = true;
            auto advContext = DeviceContext::Create(advCi);
            CHECK(advContext);
            VkResult astatus = VK_SUCCESS;
            auto advPipe = DeformPipeline::Create(advContext, evalSpv, applySpv,
                &astatus, cacheSpv);
            CHECK(advPipe && astatus == VK_SUCCESS);
            auto runForced = [&](int force, DeformPipeline::BeginInfo info,
                                 std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = advPipe->Begin(std::move(info), &astatus, &sem);
                bool ok = c && astatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, advContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            std::vector<float> rest100(size_t(100) * 3);
            for (int i = 0; i < 100; ++i) {
                rest100[size_t(i) * 3] = float(i % 10);
                rest100[size_t(i) * 3 + 1] = float((i / 10) % 10);
                rest100[size_t(i) * 3 + 2] = float(i % 3);
            }
            std::vector<float> posed100 = rest100;
            for (size_t i = 0; i < posed100.size(); i += 3) posed100[i] += 0.1f;
            uint32_t const curves = 10001, points = 100007;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0xabcdef01u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 18.0f - 9.0f;
            };
            for (auto& v : pts) v = next();
            pts[0] = 0.0f; pts[1] = 0.0f; pts[2] = 0.0f;
            pts[3] = -0.0f; pts[4] = 1e-20f; pts[5] = -1e-20f;
            pts[6] = 1e-30f; pts[7] = -1e-30f; pts[8] = 1e4f;
            for (int a = 0; a < 3; ++a) {
                pts[9 + a] = rest100[a] + 1e-7f;
                pts[size_t(90000) * 3 + size_t(a)] = rest100[size_t(5) * 3 + size_t(a)] + 1e-7f;
            }
            pts[size_t(99999) * 3] = -1e4f;
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i < curves; ++i) off[i] = i * 10;
            off[curves] = points;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * 10 * 3 + size_t(a)];
            auto ptsBuf = Upload(advContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, advContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(advContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkAdv = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest100;
                info.posedSamples = posed100;
                info.sampleCount = 100;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> hetFill, hetHit, gpuHit;
            CHECK(runForced(1, mkAdv(), &hetFill));
            CHECK(runForced(1, mkAdv(), &hetHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            uint32_t const funded = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded >= 4096 && funded < points);
            CHECK(runForced(-1, mkAdv(), &gpuHit));
            CHECK(bitEq(hetFill, gpuHit));
            CHECK(bitEq(hetHit, gpuHit));
            std::puts("Case 21 (adversarial hetero bitwise GPU): PASS");
        }

        // Case 22: the host port is partition-transparent: one range, odd
        // splits, and TBB-sized chunks evaluate bitwise identically, with
        // matching flags. Empty ranges no-op; NaN CVs flag and stay
        // unwritten, like the shader's early return.
        {
            int const n = 100, m = n + 4;
            std::vector<double> samples(size_t(3) * size_t(n));
            std::vector<double> coef(size_t(3) * size_t(m));
            uint64_t state = 0x5bd1e995u;
            auto nextD = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return double(state >> 11) * 0x1p-53 * 20.0 - 10.0;
            };
            for (auto& v : samples) v = nextD();
            for (auto& v : coef) v = nextD() * 0.01;
            DeformEvalCpuParams p;
            p.samples = samples.data();
            p.coef = coef.data();
            p.n = n; p.m = m;
            p.cx = 1.5; p.cy = -2.5; p.cz = 0.5;
            p.invScale = 0.25; p.scale = 4.0;
            uint32_t const count = 100007;
            std::vector<float> qs(size_t(count) * 3);
            for (auto& v : qs) v = float(nextD());
            qs[0] = 0.0f; qs[1] = -0.0f; qs[2] = 1e-30f;
            std::vector<float> whole(size_t(count) * 3, 12345.0f);
            uint32_t flagWhole = 0;
            CHECK(DeformEvaluateCpu(p, qs.data(), whole.data(), count, &flagWhole));
            CHECK(flagWhole == 0);
            auto runSplit = [&](std::vector<uint32_t> const& cuts,
                               std::vector<float>* out, uint32_t* flag) {
                out->assign(size_t(count) * 3, 12345.0f);
                *flag = 0;
                uint32_t prev = 0;
                for (uint32_t cut : cuts) {
                    uint32_t f = 0;
                    if (!DeformEvaluateCpu(p, qs.data() + size_t(prev) * 3,
                            out->data() + size_t(prev) * 3, cut - prev, &f))
                        return false;
                    *flag |= f;
                    prev = cut;
                }
                return true;
            };
            std::vector<float> oddOut;
            uint32_t oddFlag = 0;
            CHECK(runSplit({7, 65537, count}, &oddOut, &oddFlag));
            CHECK(oddFlag == flagWhole);
            CHECK(bitEq(oddOut, whole));
            std::vector<uint32_t> grains;
            for (uint32_t o = 2048; o < count; o += 2048) grains.push_back(o);
            grains.push_back(count);
            std::vector<float> chunkedOut;
            uint32_t chunkedFlag = 0;
            CHECK(runSplit(grains, &chunkedOut, &chunkedFlag));
            CHECK(chunkedFlag == flagWhole);
            CHECK(bitEq(chunkedOut, whole));
            std::vector<float> emptyOut(3, 7.0f);
            uint32_t emptyFlag = 0;
            CHECK(DeformEvaluateCpu(p, qs.data(), emptyOut.data(), 0, &emptyFlag));
            CHECK(emptyFlag == 0 && emptyOut[0] == 7.0f);
            std::vector<float> nanQ{0.0f, 0.0f, 0.0f,
                std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f};
            std::vector<float> nanD(6, 765.0f);
            uint32_t nanFlag = 0;
            CHECK(DeformEvaluateCpu(p, nanQ.data(), nanD.data(), 2, &nanFlag));
            CHECK(nanFlag == 1);
            CHECK(nanD[3] == 765.0f && nanD[4] == 765.0f && nanD[5] == 765.0f);
            std::puts("Case 22 (host port partition-transparent): PASS");
        }

        // Case 23: a NaN CV in the suffix flags NonFinite on hetero exactly
        // like GPU, then recovery matches after the rewrite.
        {
            DeviceContext::CreateInfo nanCi;
            nanCi.instance = native->instance;
            nanCi.physicalDevice = native->physical;
            nanCi.device = native->device;
            nanCi.computeQueue = native->queue;
            nanCi.computeQueueFamily = native->family;
            nanCi.physicalIndex = native->physicalIndex;
            nanCi.resourceDeviceId = 8036;
            nanCi.nativeLifetime = native;
            nanCi.resources = {size_t{8} << 20, 0};
            nanCi.shaderFloat64Enabled = true;
            auto nanContext = DeviceContext::Create(nanCi);
            CHECK(nanContext);
            VkResult nstatus = VK_SUCCESS;
            auto nanPipe = DeformPipeline::Create(nanContext, evalSpv, applySpv,
                &nstatus, cacheSpv);
            CHECK(nanPipe && nstatus == VK_SUCCESS);
            auto runForcedSem = [&](int force, DeformPipeline::BeginInfo info,
                                    std::vector<float>* out, DeformSemantic* semOut) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = nanPipe->Begin(std::move(info), &nstatus, &sem);
                bool submitted = c && nstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS;
                *semOut = sem;
                bool ok = submitted && sem == DeformSemantic::Ok && c->succeeded() &&
                    ReadOutput(native, nanContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            uint32_t const curves = 10000, perCurve = 10, points = curves * perCurve;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x2545f491u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = Upload(nanContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, nanContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(nanContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkNan = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = 8;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            auto poison = [&](uint32_t cv, float v) {
                void* mm = nullptr;
                if (vkMapMemory(nanContext->device(), ptsBuf->memory(),
                        VkDeviceSize(cv) * 12u, 4, 0, &mm) != VK_SUCCESS)
                    return false;
                std::memcpy(mm, &v, 4);
                vkUnmapMemory(nanContext->device(), ptsBuf->memory());
                return true;
            };
            uint32_t funded = 0;
            {
                std::vector<float> tmp;
                DeformSemantic s = DeformSemantic::Ok;
                CHECK(runForcedSem(1, mkNan(), &tmp, &s));
                CHECK(s == DeformSemantic::Ok);
                funded = DeformEvalCacheFundedPrefixForTesting();
            }
            CHECK(funded >= 4096 && funded < points);
            uint32_t const badCv = funded + (points - funded) / 2;
            float const nan = std::numeric_limits<float>::quiet_NaN();
            CHECK(poison(badCv, nan));
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> noOut;
            DeformSemantic hetSem = DeformSemantic::Ok, gpuSem = DeformSemantic::Ok;
            CHECK(!runForcedSem(1, mkNan(), &noOut, &hetSem));
            CHECK(hetSem == DeformSemantic::NonFinite);
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 1);
            CHECK(!runForcedSem(-1, mkNan(), &noOut, &gpuSem));
            CHECK(gpuSem == DeformSemantic::NonFinite);
            CHECK(poison(badCv, pts[size_t(badCv) * 3]));
            std::vector<float> hetOut, gpuOut;
            CHECK(runForcedSem(1, mkNan(), &hetOut, &hetSem));
            CHECK(hetSem == DeformSemantic::Ok);
            CHECK(runForcedSem(-1, mkNan(), &gpuOut, &gpuSem));
            CHECK(gpuSem == DeformSemantic::Ok);
            CHECK(bitEq(hetOut, gpuOut));
            std::puts("Case 23 (suffix NaN flags both, recovers): PASS");
        }

        // Case 24: forced hetero on an unfunded pose still runs GPU direct
        // (no suffix path without a cache), bitwise the forced GPU pose.
        {
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            DeviceContext::CreateInfo tinyCi;
            tinyCi.instance = native->instance;
            tinyCi.physicalDevice = native->physical;
            tinyCi.device = native->device;
            tinyCi.computeQueue = native->queue;
            tinyCi.computeQueueFamily = native->family;
            tinyCi.physicalIndex = native->physicalIndex;
            tinyCi.resourceDeviceId = 8037;
            tinyCi.nativeLifetime = native;
            tinyCi.resources = {size_t{1} << 20, 0};
            tinyCi.shaderFloat64Enabled = true;
            auto tinyContext = DeviceContext::Create(tinyCi);
            CHECK(tinyContext);
            VkResult st2 = VK_SUCCESS;
            auto smallPipe =
                DeformPipeline::Create(tinyContext, evalSpv, applySpv, &st2, cacheSpv);
            CHECK(smallPipe && st2 == VK_SUCCESS);
            const std::vector<float> tinyPts{
                0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.1f, 0.2f, 0.0f, 0.2f,
                1.0f, 0.0f, 0.0f, 1.1f, 0.0f, 0.1f, 1.2f, 0.0f, 0.2f,
            };
            const std::vector<uint32_t> tinyOff{0, 3, 6};
            const std::vector<float> tinyTgt{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            };
            auto tinyPtsBuf =
                Upload(tinyContext, tinyPts.data(), tinyPts.size() * sizeof(float));
            auto tinyOffBuf = UploadVulkanDeviceBytes(native, tinyContext, tinyOff.data(),
                tinyOff.size() * sizeof(uint32_t));
            auto tinyTgtBuf =
                Upload(tinyContext, tinyTgt.data(), tinyTgt.size() * sizeof(float));
            CHECK(tinyPtsBuf && tinyOffBuf && tinyTgtBuf);
            std::vector<float> rest100(size_t(100) * 3);
            for (int i = 0; i < 100; ++i) {
                rest100[size_t(i) * 3] = float(i % 10);
                rest100[size_t(i) * 3 + 1] = float((i / 10) % 10);
                rest100[size_t(i) * 3 + 2] = float(i % 3);
            }
            std::vector<float> posed100 = rest100;
            for (size_t i = 0; i < posed100.size(); i += 3) posed100[i] += 0.1f;
            auto mkTiny = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = tinyPtsBuf;
                info.curveOffsets = tinyOffBuf;
                info.rootTargets = tinyTgtBuf;
                info.curveCount = 2;
                info.pointCount = 6;
                info.restSamples = rest100;
                info.posedSamples = posed100;
                info.sampleCount = 100;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            auto runForced = [&](int force, std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = smallPipe->Begin(mkTiny(), &st2, &sem);
                bool ok = c && st2 == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, tinyContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            std::vector<float> hetOut, gpuOut;
            CHECK(runForced(1, &hetOut));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 0);
            CHECK(runForced(-1, &gpuOut));
            CHECK(bitEq(hetOut, gpuOut));
            std::puts("Case 24 (unfunded forced hetero runs GPU): PASS");
        }

        // Case 25: the automatic policy is correct whatever it picks: three
        // auto poses match the forced-GPU reference bitwise. (Engagement is
        // timing-dependent, so only correctness is asserted.)
        {
            DeviceContext::CreateInfo autoCi;
            autoCi.instance = native->instance;
            autoCi.physicalDevice = native->physical;
            autoCi.device = native->device;
            autoCi.computeQueue = native->queue;
            autoCi.computeQueueFamily = native->family;
            autoCi.physicalIndex = native->physicalIndex;
            autoCi.resourceDeviceId = 8038;
            autoCi.nativeLifetime = native;
            autoCi.resources = {size_t{48} << 20, 0};
            autoCi.shaderFloat64Enabled = true;
            auto autoContext = DeviceContext::Create(autoCi);
            CHECK(autoContext);
            VkResult st3 = VK_SUCCESS;
            auto autoPipe = DeformPipeline::Create(autoContext, evalSpv, applySpv,
                &st3, cacheSpv);
            CHECK(autoPipe && st3 == VK_SUCCESS);
            uint32_t const curves = 60000, perCurve = 10, points = curves * perCurve;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x12345678u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = Upload(autoContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, autoContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(autoContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkAuto = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = 8;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            auto runAuto = [&](std::vector<float>* out) {
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = autoPipe->Begin(mkAuto(), &st3, &sem);
                return c && st3 == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, autoContext, c, out);
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> a0, a1, a2, ref;
            CHECK(runAuto(&a0));
            CHECK(runAuto(&a1));
            CHECK(runAuto(&a2));
            TestForceDeformHeteroSuffix(-1);
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = autoPipe->Begin(mkAuto(), &st3, &sem);
            bool ok = c && st3 == VK_SUCCESS && Prove(native) &&
                c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                c->succeeded() && ReadOutput(native, autoContext, c, &ref);
            TestForceDeformHeteroSuffix(0);
            CHECK(ok);
            CHECK(bitEq(a0, ref));
            CHECK(bitEq(a1, ref));
            CHECK(bitEq(a2, ref));
            std::printf("(auto hetero runs over 3 poses: %llu)\n",
                (unsigned long long)(DeformHeteroSuffixRunsForTesting() - r0));
            std::puts("Case 25 (auto policy correct): PASS");
        }

        // Case 26: the hetero balance cap. A budget that funds the whole
        // 600k prefix would run GPU-only; the cap holds the active prefix
        // at the host/device balance point so forced hetero still runs the
        // suffix (fill + hit), bitwise the forced-GPU full prefix.
        {
            DeviceContext::CreateInfo capCi;
            capCi.instance = native->instance;
            capCi.physicalDevice = native->physical;
            capCi.device = native->device;
            capCi.computeQueue = native->queue;
            capCi.computeQueueFamily = native->family;
            capCi.physicalIndex = native->physicalIndex;
            capCi.resourceDeviceId = 8039;
            capCi.nativeLifetime = native;
            capCi.resources = {size_t{128} << 20, 0};
            capCi.shaderFloat64Enabled = true;
            auto capContext = DeviceContext::Create(capCi);
            CHECK(capContext);
            VkResult cstatus = VK_SUCCESS;
            auto capPipe = DeformPipeline::Create(capContext, evalSpv, applySpv,
                &cstatus, cacheSpv);
            CHECK(capPipe && cstatus == VK_SUCCESS);
            auto runForced = [&](int force, DeformPipeline::BeginInfo info,
                                 std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = capPipe->Begin(std::move(info), &cstatus, &sem);
                bool ok = c && cstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, capContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            uint32_t const curves = 60000, perCurve = 10, points = curves * perCurve;
            int const n = 8;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x12345678u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            std::vector<float> rest8{
                0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f,
                0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
            };
            std::vector<float> posed8 = rest8;
            for (size_t i = 0; i < posed8.size(); i += 3) posed8[i] += 0.25f;
            auto ptsBuf = Upload(capContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, capContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(capContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkCap = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest8;
                info.posedSamples = posed8;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> hetFill, hetHit, gpuHit;
            CHECK(runForced(1, mkCap(), &hetFill));
            CHECK(runForced(1, mkCap(), &hetHit));
            // Full funding (the cap's precondition) with a host suffix both
            // poses: without the cap the funded prefix covers every point
            // and the counter stays flat.
            CHECK(DeformEvalCacheFundedPrefixForTesting() == points);
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            CHECK(runForced(-1, mkCap(), &gpuHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            CHECK(bitEq(hetFill, gpuHit));
            CHECK(bitEq(hetHit, gpuHit));
            std::puts("Case 26 (balance cap caps full funding): PASS");
        }

        // Case 27: small-split hetero transparency. The 6MiB budget funds
        // only a few thousand prefix CVs at n=100, so forced hetero runs
        // a ~4-16k prefix with a large host suffix (fill + hit), bitwise
        // the forced-GPU run. The funded band pins the small-split shape:
        // a future funding change that unfunds the pose (or funds past
        // 16k) fails here instead of silently testing another shape.
        {
            DeviceContext::CreateInfo smallCi;
            smallCi.instance = native->instance;
            smallCi.physicalDevice = native->physical;
            smallCi.device = native->device;
            smallCi.computeQueue = native->queue;
            smallCi.computeQueueFamily = native->family;
            smallCi.physicalIndex = native->physicalIndex;
            smallCi.resourceDeviceId = 8040;
            smallCi.nativeLifetime = native;
            smallCi.resources = {size_t{6} << 20, 0};
            smallCi.shaderFloat64Enabled = true;
            auto smallContext = DeviceContext::Create(smallCi);
            CHECK(smallContext);
            VkResult sstatus = VK_SUCCESS;
            auto smallPipe = DeformPipeline::Create(smallContext, evalSpv, applySpv,
                &sstatus, cacheSpv);
            CHECK(smallPipe && sstatus == VK_SUCCESS);
            auto runForced = [&](int force, DeformPipeline::BeginInfo info,
                                 std::vector<float>* out) {
                TestForceDeformHeteroSuffix(force);
                DeformSemantic sem = DeformSemantic::Ok;
                auto c = smallPipe->Begin(std::move(info), &sstatus, &sem);
                bool ok = c && sstatus == VK_SUCCESS && Prove(native) &&
                    c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok &&
                    c->succeeded() && ReadOutput(native, smallContext, c, out);
                TestForceDeformHeteroSuffix(0);
                return ok;
            };
            uint32_t const curves = 2000, perCurve = 10, points = curves * perCurve;
            int const n = 100;
            std::vector<float> rest100(size_t(n) * 3);
            for (int i = 0; i < n; ++i) {
                rest100[size_t(i) * 3] = float(i % 10);
                rest100[size_t(i) * 3 + 1] = float((i / 10) % 10);
                rest100[size_t(i) * 3 + 2] = float(i % 3);
            }
            std::vector<float> posed100 = rest100;
            for (size_t i = 0; i < posed100.size(); i += 3) posed100[i] += 0.1f;
            std::vector<float> pts(size_t(points) * 3);
            uint64_t state = 0x5eed2701u;
            auto next = [&]() {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                return float((state >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
            };
            for (auto& v : pts) v = next();
            std::vector<uint32_t> off(curves + 1);
            for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
            std::vector<float> tgt(size_t(curves) * 3);
            for (uint32_t i = 0; i < curves; ++i)
                for (int a = 0; a < 3; ++a)
                    tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
            auto ptsBuf = Upload(smallContext, pts.data(), pts.size() * sizeof(float));
            auto offBuf = UploadVulkanDeviceBytes(native, smallContext, off.data(),
                off.size() * sizeof(uint32_t));
            auto tgtBuf = Upload(smallContext, tgt.data(), tgt.size() * sizeof(float));
            CHECK(ptsBuf && offBuf && tgtBuf);
            auto mkSmall = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = ptsBuf;
                info.curveOffsets = offBuf;
                info.rootTargets = tgtBuf;
                info.curveCount = curves;
                info.pointCount = points;
                info.restSamples = rest100;
                info.posedSamples = posed100;
                info.sampleCount = n;
                info.smoothing = 0.0;
                info.mask = {1.0f, 1, nullptr, 0};
                info.enabled = {1, 1, nullptr, 0};
                info.lockRoots = {0, 1, nullptr, 0};
                info.groomEnvelope = 1.0f;
                return info;
            };
            uint64_t r0 = DeformHeteroSuffixRunsForTesting();
            std::vector<float> hetFill, hetHit, gpuHit;
            CHECK(runForced(1, mkSmall(), &hetFill));
            CHECK(runForced(1, mkSmall(), &hetHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            uint32_t const funded = DeformEvalCacheFundedPrefixForTesting();
            CHECK(funded >= 4096 && funded <= 16384);
            CHECK(runForced(-1, mkSmall(), &gpuHit));
            CHECK(DeformHeteroSuffixRunsForTesting() - r0 == 2);
            CHECK(bitEq(hetFill, gpuHit));
            CHECK(bitEq(hetHit, gpuHit));
            std::puts("Case 27 (small-split hetero bitwise GPU): PASS");
        }
    }

    std::puts("Vulkan deform pipeline: PASS");
    return 0;
}
