// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenVulkanDeformStress — DeformPipeline Begin/Launch from many
// threads on one cached pipeline. The poses race for the proof state (the
// loser runs direct, bitwise the same), for the idle scratch sets and the
// outPoints pool, and every submit goes through the context's queue
// timeline. Half the poses use the synchronous Begin, half Launch + Poll
// (the non-blocking proof stage); some candidates are dropped mid-proof.
// Every output must match the single-thread reference bit for bit.
#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deformPipeline.h"
#include "usdGen/vulkan/deviceContext.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
std::vector<uint32_t> Code(char const* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}
std::shared_ptr<const ChargedBuffer> Upload(std::shared_ptr<DeviceContext> const& c,
                                            void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = ChargedBuffer::Create(c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b) return {};
    void* p = nullptr;
    if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS) return {};
    std::memcpy(p, data, bytes);
    vkUnmapMemory(c->device(), b->memory());
    return b;
}
// Drives a candidate to completion without the native queue (other threads
// submit concurrently through the timeline).
VkResult Finish(DeformPipeline::Candidate& c, DeformSemantic* sem) {
    VkResult r;
    while ((r = c.Poll(sem)) == VK_NOT_READY) std::this_thread::yield();
    return r;
}
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s deformEvaluate.spv deformApply.spv "
                             "deformEvaluateVerify.spv deformEvaluateFill.spv "
                             "deformEvaluateCached.spv\n", argv[0]);
        return 1;
    }
    auto evalSpv = Code(argv[1]), applySpv = Code(argv[2]);
    DeformEvalCacheSpirv cacheSpv;
    cacheSpv.verify = Code(argv[3]);
    cacheSpv.fill = Code(argv[4]);
    cacheSpv.cached = Code(argv[5]);
    CHECK(!evalSpv.empty() && !applySpv.empty() && !cacheSpv.verify.empty() &&
          !cacheSpv.fill.empty() && !cacheSpv.cached.empty());
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64{};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateTimelineNative(&unavailable, &fp64);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8102;
    ci.nativeLifetime = native;
    ci.resources = {size_t{512} << 20, 0};
    ci.shaderFloat64Enabled = true;
    ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci);
    CHECK(context);
    VkResult status = VK_SUCCESS;
    auto pipe = DeformPipeline::Create(context, evalSpv, applySpv, &status, cacheSpv);
    CHECK(pipe && status == VK_SUCCESS);

    // Geometry: 2000 curves x 8 CVs, pseudo-random.
    uint32_t const curves = 2000, perCurve = 8, points = curves * perCurve;
    std::vector<float> pts(size_t(points) * 3);
    uint64_t s = 0x2545F4914F6CDD1Dull;
    for (auto& v : pts) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        v = float((s >> 11) & 0xffffff) / float(0xffffff) * 2.0f - 1.0f;
    }
    std::vector<uint32_t> off(curves + 1);
    for (uint32_t i = 0; i <= curves; ++i) off[i] = i * perCurve;
    std::vector<float> tgt(size_t(curves) * 3);
    for (uint32_t i = 0; i < curves; ++i)
        for (int a = 0; a < 3; ++a) tgt[size_t(i) * 3 + size_t(a)] = pts[size_t(i) * perCurve * 3 + size_t(a)];
    auto ptsBuf = Upload(context, pts.data(), pts.size() * sizeof(float));
    auto offBuf = UploadVulkanDeviceBytes(native, context, off.data(), off.size() * sizeof(uint32_t));
    auto tgtBuf = Upload(context, tgt.data(), tgt.size() * sizeof(float));
    CHECK(ptsBuf && offBuf && tgtBuf);
    std::vector<float> const rest{
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
        0.5f, 0.5f, 0.5f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 2.0f,
    };
    constexpr int kPoses = 6;
    auto mkInfo = [&](int pose) {
        DeformPipeline::BeginInfo info;
        info.points = ptsBuf;
        info.curveOffsets = offBuf;
        info.rootTargets = tgtBuf;
        info.curveCount = curves;
        info.pointCount = points;
        info.restSamples = rest;
        info.posedSamples = rest;
        for (size_t i = 0; i < info.posedSamples.size(); i += 3)
            info.posedSamples[i] += 0.05f * float(pose + 1);
        info.sampleCount = 8;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;
        return info;
    };
    auto read = [&](std::shared_ptr<const ChargedBuffer> const& b, std::vector<float>* out) {
        std::vector<uint8_t> bytes;
        if (!ReadVulkanBytes(native, context, b->buffer(), b->sizeBytes(), b, &bytes)) return false;
        out->resize(bytes.size() / sizeof(float));
        std::memcpy(out->data(), bytes.data(), bytes.size());
        return true;
    };

    // Single-thread reference per pose.
    std::vector<std::vector<float>> ref(kPoses);
    for (int p = 0; p < kPoses; ++p) {
        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(mkInfo(p), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        CHECK(Finish(*c, &sem) == VK_SUCCESS && sem == DeformSemantic::Ok && c->succeeded());
        CHECK(read(c->output().points, &ref[p]));
        CHECK(ref[p].size() == size_t(points) * 3);
    }

    // Stress: every thread runs a mix of synchronous and staged poses.
    constexpr int kThreads = 8, kRounds = 24;
    struct Result { int pose = -1; std::shared_ptr<const ChargedBuffer> points; };
    std::vector<std::vector<Result>> results(kThreads);
    std::atomic<int> failures{0}, staged{0}, dropped{0};
    std::atomic<int> go{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < kRounds; ++i) {
                int const pose = (t + i) % kPoses;
                DeformSemantic sem = DeformSemantic::Ok;
                VkResult r = VK_SUCCESS;
                std::unique_ptr<DeformPipeline::Candidate> c;
                if ((t + i) % 2) {
                    c = pipe->Begin(mkInfo(pose), &r, &sem);
                } else {
                    c = pipe->Launch(mkInfo(pose), &r, &sem);
                    staged.fetch_add(1, std::memory_order_relaxed);
                    if (c && i % 7 == 3) {
                        // Dropped mid-proof: destruction settles the proof.
                        c.reset();
                        dropped.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                }
                if (!c || r != VK_SUCCESS) { failures.fetch_add(1); continue; }
                if (Finish(*c, &sem) != VK_SUCCESS || sem != DeformSemantic::Ok || !c->succeeded()) {
                    failures.fetch_add(1);
                    continue;
                }
                results[t].push_back({pose, c->output().points});
            }
        });
    }
    go.store(1, std::memory_order_release);
    for (auto& th : threads) th.join();
    CHECK(failures.load() == 0);
    size_t compared = 0;
    for (int t = 0; t < kThreads; ++t)
        for (auto const& res : results[t]) {
            std::vector<float> got;
            CHECK(read(res.points, &got));
            CHECK(got.size() == ref[res.pose].size());
            CHECK(std::memcmp(got.data(), ref[res.pose].data(), got.size() * sizeof(float)) == 0);
            ++compared;
        }
    CHECK(compared + size_t(dropped.load()) == size_t(kThreads * kRounds));
    auto ts = context->computeTimeline()->stats();
    std::printf("deform stress: %zu outputs bitwise, %d staged, %d dropped mid-proof; "
                "cache hits %llu misses %llu; timeline %llu submits, %llu combined, %llu parked\n",
        compared, staged.load(), dropped.load(),
        (unsigned long long)DeformEvalCacheHitsForTesting(),
        (unsigned long long)DeformEvalCacheMissesForTesting(),
        (unsigned long long)ts.submits, (unsigned long long)ts.combined,
        (unsigned long long)ts.parks);
    std::puts("deform stress: PASS");
    return 0;
}
