// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/noisePipeline.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include "usdGenMath/usdGenMath/hash.h"
#include "SeExpr2/Noise.h"

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

struct V3 { float x, y, z; };
inline V3 V(float x, float y, float z) { return {x, y, z}; }
inline V3 Add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 Mul(V3 a, float f) { return {a.x * f, a.y * f, a.z * f}; }
inline bool Near(V3 a, V3 b, float tol = 2e-3f) {
    return std::fabs(a.x - b.x) <= tol && std::fabs(a.y - b.y) <= tol && std::fabs(a.z - b.z) <= tol;
}
inline V3 FrameVector(V3 t, V3 b, V3 n, V3 local) {
    return Add(Add(Mul(t, local.x), Mul(b, local.y)), Mul(n, local.z));
}

inline V3 CpuVfbm(V3 restRoot, uint64_t id, int seed, float t,
                  float frequency, float correlation, int octaves,
                  float lacunarity, float gain) {
    const V3 hash = V(UsdGenDraw01(seed, id, kSaltNoise),
                      UsdGenDraw01(seed, id, kSaltNoise + 1u),
                      UsdGenDraw01(seed, id, kSaltNoise + 2u));
    const double input[3] = {
        double(restRoot.x * correlation + hash.x * (1.0f - correlation)),
        double(restRoot.y * correlation + hash.y * (1.0f - correlation)),
        double(restRoot.z * correlation + hash.z * (1.0f - correlation) + t * frequency)};
    double result[3]{};
    SeExpr2::FBM<3, 3, false, double>(input, result, octaves,
                                      double(lacunarity), double(gain));
    return V(float(result[0]), float(result[1]), float(result[2]));
}

inline float Sample257(std::vector<float> const& values, float t) {
    const int lower = std::min(255, std::max(0, int(std::floor(t * 256.0f))));
    const float fraction = t * 256.0f - float(lower);
    return values[lower] + (values[lower + 1] - values[lower]) * fraction;
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s noiseValidate.spv noiseGenerate.spv\n", argv[0]);
        return 1;
    }
    auto validateSpv = Code(argv[1]);
    auto generateSpv = Code(argv[2]);
    CHECK(!validateSpv.empty() && !generateSpv.empty());

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
    auto pipe = NoisePipeline::Create(context, validateSpv, generateSpv, &status);
    CHECK(pipe && status == VK_SUCCESS);

    auto readOutput = [&](std::unique_ptr<NoisePipeline::Candidate> const& c,
                          std::vector<float>* out) {
        auto o = c->output();
        std::vector<uint8_t> bytes;
        if (!ReadVulkanBytes(native, context, o.points->buffer(),
                             o.points->sizeBytes(), o.points, &bytes)) return false;
        out->resize(bytes.size() / sizeof(float));
        std::memcpy(out->data(), bytes.data(), bytes.size());
        return true;
    };

    // ---- Base test data (mirror CUDA test case 1) ----
    const std::vector<float> points{0,0,0, 0,0,1, 0,0,2, 3,0,0, 3,1,0, 3,2,0, 3,3,0};
    const std::vector<float> rest{0,0,0, 0,0,1, 0,0,2, 3,0,0, 3,1,0, 3,2,0, 3,3,0};
    const std::vector<uint32_t> offsets{0, 3, 7};
    const std::vector<uint32_t> idsPacked{41, 0, 99, 0};  // lo/hi pairs for u64 ids
    const std::vector<float> hairT{0, .5f, 1, 0, .25f, .65f, 1};
    const std::vector<float> tangent{1,0,0, 0,1,0};
    const std::vector<float> binormal{0,1,0, 0,0,1};
    const std::vector<float> normal{0,0,1, 1,0,0};
    const std::vector<float> profile(257, 1.0f);
    const uint64_t stableIds[2] = {41, 99};

    auto pointsBuf = Upload(context, points.data(), points.size() * sizeof(float));
    auto restBuf = Upload(context, rest.data(), rest.size() * sizeof(float));
    auto offsetsBuf = Upload(context, offsets.data(), offsets.size() * sizeof(uint32_t));
    auto idsBuf = Upload(context, idsPacked.data(), idsPacked.size() * sizeof(uint32_t));
    auto hairTBuf = Upload(context, hairT.data(), hairT.size() * sizeof(float));
    auto tanBuf = Upload(context, tangent.data(), tangent.size() * sizeof(float));
    auto binBuf = Upload(context, binormal.data(), binormal.size() * sizeof(float));
    auto nrmBuf = Upload(context, normal.data(), normal.size() * sizeof(float));
    auto profileBuf = Upload(context, profile.data(), profile.size() * sizeof(float));
    CHECK(pointsBuf && restBuf && offsetsBuf && idsBuf && hairTBuf && tanBuf && binBuf && nrmBuf && profileBuf);

    auto fillGeometry = [&](NoisePipeline::BeginInfo& info) {
        info.points = pointsBuf;
        info.restPoints = restBuf;
        info.curveOffsets = offsetsBuf;
        info.stableIds = idsBuf;
        info.curveCount = 2;
        info.pointCount = 7;
        info.hairT = hairTBuf;
        info.frameTangent = tanBuf;
        info.frameBinormal = binBuf;
        info.frameNormal = nrmBuf;
        info.frameCount = 2;
        info.magnitudeProfile = profileBuf;
    };

    auto setGroomLiterals = [&](NoisePipeline::BeginInfo& info) {
        info.magnitude = {0.2f, 1, nullptr, 0};
        info.frequency = {1.7f, 1, nullptr, 0};
        info.correlation = {0.4f, 1, nullptr, 0};
        info.octaves = {3, 1, nullptr, 0};
        info.lacunarity = {2.0f, 1, nullptr, 0};
        info.gain = {0.5f, 1, nullptr, 0};
        info.preserveLength = {0.0f, 1, nullptr, 0};
        info.mask = {1.0f, 1, nullptr, 0};
        info.seed = {17, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.cumulative = {0, 1, nullptr, 0};
    };

    // ---- Case 1: Basic groom literals (oracle comparison) ----
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == NoiseSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(readOutput(c, &out));
        CHECK(out.size() == 21);

        for (size_t cv = 0; cv < 2; ++cv) {
            const V3 restRoot = V(rest[offsets[cv] * 3], rest[offsets[cv] * 3 + 1], rest[offsets[cv] * 3 + 2]);
            const V3 t = V(tangent[cv * 3], tangent[cv * 3 + 1], tangent[cv * 3 + 2]);
            const V3 b = V(binormal[cv * 3], binormal[cv * 3 + 1], binormal[cv * 3 + 2]);
            const V3 n = V(normal[cv * 3], normal[cv * 3 + 1], normal[cv * 3 + 2]);
            for (uint32_t pt = offsets[cv]; pt < offsets[cv + 1]; ++pt) {
                const V3 field = CpuVfbm(restRoot, stableIds[cv], 17, hairT[pt], 1.7f, 0.4f, 3, 2.0f, 0.5f);
                const V3 local = Mul(field, 0.2f);
                const V3 expected = Add(V(points[pt * 3], points[pt * 3 + 1], points[pt * 3 + 2]),
                                        FrameVector(t, b, n, local));
                const V3 got = V(out[pt * 3], out[pt * 3 + 1], out[pt * 3 + 2]);
                CHECK(Near(got, expected));
            }
        }
        std::puts("Case 1 (basic groom literals): PASS");
    }

    // ---- Case 2: Point-domain fields + primitive seed/cumulative ----
    {
        const std::vector<float> pointMags{.1f, .2f, .3f, .35f, .25f, .15f, .05f};
        const std::vector<int32_t> pointOctaves{3, 3, 3, 3, 3, 3, 3};
        const std::vector<float> pointMasks{0.0f, .75f, .5f, 1.0f, .25f, .5f, 1.0f};
        const std::vector<int32_t> primSeeds{17, 18};
        const std::vector<uint32_t> cumData{1, 0};

        std::vector<float> ramp(257, 1.0f);
        ramp[0] = 0.0f;
        ramp[128] = 0.5f;

        auto magBuf = Upload(context, pointMags.data(), pointMags.size() * sizeof(float));
        auto octBuf = Upload(context, pointOctaves.data(), pointOctaves.size() * sizeof(int32_t));
        auto maskBuf = Upload(context, pointMasks.data(), pointMasks.size() * sizeof(float));
        auto seedBuf = Upload(context, primSeeds.data(), primSeeds.size() * sizeof(int32_t));
        auto cumBuf = Upload(context, cumData.data(), cumData.size() * sizeof(uint32_t));
        auto rampBuf = Upload(context, ramp.data(), ramp.size() * sizeof(float));
        CHECK(magBuf && octBuf && maskBuf && seedBuf && cumBuf && rampBuf);

        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        info.magnitudeProfile = rampBuf;
        info.magnitude = {.0f, 4, magBuf, 7};
        info.frequency = {1.7f, 1, nullptr, 0};
        info.correlation = {0.4f, 1, nullptr, 0};
        info.octaves = {0, 4, octBuf, 7};
        info.lacunarity = {2.0f, 1, nullptr, 0};
        info.gain = {0.5f, 1, nullptr, 0};
        info.preserveLength = {0.0f, 1, nullptr, 0};
        info.mask = {.0f, 4, maskBuf, 7};
        info.seed = {0, 2, seedBuf, 2};
        info.enabled = {1, 1, nullptr, 0};
        info.cumulative = {0, 2, cumBuf, 2};

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == NoiseSemantic::Ok);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(readOutput(c, &out));
        CHECK(out.size() == 21);

        for (size_t cv = 0; cv < 2; ++cv) {
            const bool isCum = (cv == 0);
            const int seed = primSeeds[cv];
            const V3 restRoot = V(rest[offsets[cv] * 3], rest[offsets[cv] * 3 + 1], rest[offsets[cv] * 3 + 2]);
            const V3 t = V(tangent[cv * 3], tangent[cv * 3 + 1], tangent[cv * 3 + 2]);
            const V3 b = V(binormal[cv * 3], binormal[cv * 3 + 1], binormal[cv * 3 + 2]);
            const V3 n = V(normal[cv * 3], normal[cv * 3 + 1], normal[cv * 3 + 2]);
            V3 running = V(0, 0, 0);
            for (uint32_t pt = offsets[cv]; pt < offsets[cv + 1]; ++pt) {
                const V3 field = CpuVfbm(restRoot, stableIds[cv], seed, hairT[pt],
                                         1.7f, 0.4f, pointOctaves[pt], 2.0f, 0.5f);
                if (isCum) running = Add(running, field);
                const float scale = pointMags[pt] * Sample257(ramp, hairT[pt]) * pointMasks[pt];
                const V3 local = Mul(isCum ? running : field, scale);
                const V3 expected = Add(V(points[pt * 3], points[pt * 3 + 1], points[pt * 3 + 2]),
                                        FrameVector(t, b, n, local));
                const V3 got = V(out[pt * 3], out[pt * 3 + 1], out[pt * 3 + 2]);
                CHECK(Near(got, expected));
            }
        }
        std::puts("Case 2 (device fields + cumulative): PASS");
    }

    // ---- Case 3: Reject — empty terminal curve (bad offsets) ----
    {
        const std::vector<uint32_t> badOffsets{0, 3, 3};
        auto badOffBuf = Upload(context, badOffsets.data(), badOffsets.size() * sizeof(uint32_t));
        CHECK(badOffBuf);

        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        info.curveOffsets = badOffBuf;
        setGroomLiterals(info);

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(!c && status == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(sem == NoiseSemantic::BadOffsets);
        std::puts("Case 3 (reject bad offsets): PASS");
    }

    // ---- Case 4: Reject — invalid seed domain (host validation) ----
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        info.seed = {17, 4, nullptr, 0};  // domain=Point (4) is invalid for seed

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(!c && status == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(sem == NoiseSemantic::BadValue);
        std::puts("Case 4 (reject invalid seed domain): PASS");
    }

    // ---- Case 5: Empty topology ----
    {
        NoisePipeline::BeginInfo info;
        info.curveCount = 0;
        info.pointCount = 0;
        info.magnitudeProfile = profileBuf;
        setGroomLiterals(info);

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(sem == NoiseSemantic::Ok);
        CHECK(c->succeeded());
        CHECK(c->pointCount() == 0);
        CHECK(!c->output().points);
        std::puts("Case 5 (empty topology): PASS");
    }

    // ---- Case 6: Mask=0 → bitwise input pass-through ----
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        info.mask = {0.0f, 1, nullptr, 0};

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(readOutput(c, &out));
        CHECK(out.size() == 21);
        CHECK(std::memcmp(out.data(), points.data(), points.size() * sizeof(float)) == 0);
        std::puts("Case 6 (mask=0 pass-through): PASS");
    }

    // ---- Case 7: Enabled=0 → bitwise input pass-through ----
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        info.enabled = {0, 1, nullptr, 0};

        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Prove(native));
        CHECK(c->Poll(&sem) == VK_SUCCESS);
        CHECK(c->succeeded());

        std::vector<float> out;
        CHECK(readOutput(c, &out));
        CHECK(out.size() == 21);
        CHECK(std::memcmp(out.data(), points.data(), points.size() * sizeof(float)) == 0);
        std::puts("Case 7 (enabled=0 pass-through): PASS");
    }

    std::puts("Vulkan noise pipeline: PASS");
    return 0;
}
