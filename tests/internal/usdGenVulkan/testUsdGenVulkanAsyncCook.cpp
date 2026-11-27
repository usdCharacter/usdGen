// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenVulkanAsyncCook: warm Vulkan cooks never block on the queue
// timeline. (1) 20 warm param-edit Session cooks on the default provider
// make no VulkanQueueTimeline::Wait call while the timeline advances.
// (2) 20 DeformPipeline poses through the asynchronous begin (Launch, then
// Poll checks the proof and apply values) make no Wait call either and
// match the synchronous Begin bit for bit.
#include "utils/pxrInterop.h"
#include "api/testBackend.h"
#include "usdGen/api/session.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/vulkan/deformPipeline.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenVulkan/vulkan/defaultProvider.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/queueTimeline.h"

#include "hosts/usdGenUsd/vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "async cook check: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
constexpr int kWarmCooks = 20;

void ShaderDirectory(std::string const& value) {
#ifdef _WIN32
    CHECK(_putenv_s("USDGEN_VULKAN_SHADER_DIR", value.c_str()) == 0);
#else
    CHECK(setenv("USDGEN_VULKAN_SHADER_DIR", value.c_str(), 1) == 0);
#endif
}

UsdGenGraphDesc Descriptor(float width) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/AsyncCook");
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/AsyncCook/curves");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {3, 2}; curves.curveId = {20, 10};
    curves.points = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0}};
    curves.rest = {{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1}};
    curves.widths = {1,2,3,4,5}; curves.widthsInterpolation = TfToken("vertex");
    curves.curveGeneration = 41; desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/AsyncCook/source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc widthNode;
    widthNode.path = SdfPath("/AsyncCook/width"); widthNode.type = TfToken("UsdGenWidth");
    widthNode.inputs = {source.path};
    widthNode.params = {{TfToken("width"), VtValue(width), false},
                        {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, widthNode}; desc.terminal = widthNode.path;
    return desc;
}

int SessionCooks() {
    UsdGenSession session(2);
    usdGenUsd::SetGraphDesc(session, Descriptor(1.25f));
    session.SetDevicePublicationEnabled(true);
    auto cold = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    if (cold) for (auto const& error : cold->diagnostics.errors)
        std::fprintf(stderr, "cold: %s\n", error.c_str());
    CHECK(cold && !cold->diagnostics.HasErrors() && cold->generation && cold->generation->device);
    auto access = GetVulkanGenerationAccess(cold->generation->device);
    CHECK(access && access.context->computeTimeline());
    auto* timeline = access.context->computeTimeline();
    uint64_t const waits = VulkanQueueTimelineWaitsForTesting();
    uint64_t const submits = timeline->stats().submits;
    for (int i = 0; i != kWarmCooks; ++i) {
        usdGenUsd::SetGraphDesc(session, Descriptor(1.5f + float(i) / 8.f));
        auto warm = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        if (warm) for (auto const& error : warm->diagnostics.errors)
            std::fprintf(stderr, "warm %d: %s\n", i, error.c_str());
        CHECK(warm && !warm->diagnostics.HasErrors() && warm->generation && warm->generation->device);
        auto warmAccess = GetVulkanGenerationAccess(warm->generation->device);
        CHECK(warmAccess && warmAccess.context == access.context);
    }
    uint64_t const waited = VulkanQueueTimelineWaitsForTesting() - waits;
    uint64_t const submitted = timeline->stats().submits - submits;
    std::printf("session: %d warm cooks, %llu timeline waits, %llu timeline submits\n", kWarmCooks,
                (unsigned long long)waited, (unsigned long long)submitted);
    CHECK(waited == 0);
    CHECK(submitted > 0);
    session.Drain();
    return 0;
}

std::vector<uint32_t> Code(char const* path) {
    std::ifstream f(path, std::ios::binary);
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
    std::memcpy(p, data, size_t(bytes));
    vkUnmapMemory(c->device(), b->memory());
    return b;
}

int DeformPoses(char const* evaluatePath, char const* applyPath) {
    auto evalSpv = Code(evaluatePath), applySpv = Code(applyPath);
    CHECK(!evalSpv.empty() && !applySpv.empty());
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
    ci.resourceDeviceId = 8108;
    ci.nativeLifetime = native;
    ci.resources = {size_t{256} << 20, 0};
    ci.shaderFloat64Enabled = true;
    ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci);
    CHECK(context);
    VkResult status = VK_SUCCESS;
    auto pipe = DeformPipeline::Create(context, evalSpv, applySpv, &status);
    CHECK(pipe && status == VK_SUCCESS);

    uint32_t const curves = 256, perCurve = 8, points = curves * perCurve;
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
    auto mkInfo = [&] {
        DeformPipeline::BeginInfo info;
        info.points = ptsBuf;
        info.curveOffsets = offBuf;
        info.rootTargets = tgtBuf;
        info.curveCount = curves;
        info.pointCount = points;
        info.restSamples = rest;
        info.posedSamples = rest;
        for (size_t i = 0; i < info.posedSamples.size(); i += 3) info.posedSamples[i] += 0.1f;
        info.sampleCount = 8;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;
        return info;
    };
    auto read = [&](std::shared_ptr<const ChargedBuffer> const& b, std::vector<uint8_t>* out) {
        return ReadVulkanBytes(native, context, b->buffer(), b->sizeBytes(), b, out);
    };

    // Reference through the synchronous Begin (it waits; not counted).
    DeformSemantic sem = DeformSemantic::Ok;
    std::vector<uint8_t> reference;
    {
        auto c = pipe->Begin(mkInfo(), &status, &sem);
        CHECK(c && status == VK_SUCCESS);
        VkResult r;
        while ((r = c->Poll(&sem)) == VK_NOT_READY) std::this_thread::yield();
        CHECK(r == VK_SUCCESS && sem == DeformSemantic::Ok && c->succeeded());
        CHECK(read(c->output().points, &reference) && reference.size() == size_t(points) * 12);
    }

    uint64_t const waits = VulkanQueueTimelineWaitsForTesting();
    std::vector<std::shared_ptr<const ChargedBuffer>> outputs;
    int proofPending = 0;
    for (int pose = 0; pose != kWarmCooks; ++pose) {
        auto c = pipe->Launch(mkInfo(), &status, &sem);
        CHECK(c && status == VK_SUCCESS && sem == DeformSemantic::Ok);
        // The proof stage is a timeline check: Advance reports VK_NOT_READY
        // until ProofValue signals, then submits the apply.
        VkResult r;
        while (c->ProofValue()) {
            r = c->Advance(&sem);
            if (r == VK_NOT_READY) { ++proofPending; std::this_thread::yield(); continue; }
            CHECK(r == VK_SUCCESS && sem == DeformSemantic::Ok);
        }
        while ((r = c->Poll(&sem)) == VK_NOT_READY) std::this_thread::yield();
        CHECK(r == VK_SUCCESS && sem == DeformSemantic::Ok && c->succeeded());
        outputs.push_back(c->output().points);
    }
    uint64_t const waited = VulkanQueueTimelineWaitsForTesting() - waits;
    std::printf("deform: %d asynchronous begins, %llu timeline waits, %d pending proof checks\n",
                kWarmCooks, (unsigned long long)waited, proofPending);
    CHECK(waited == 0);
    for (auto const& out : outputs) {
        std::vector<uint8_t> got;
        CHECK(read(out, &got) && got == reference);
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s width.spv deformEvaluate.spv deformApply.spv\n", argv[0]);
        return 1;
    }
    if (!HasDefaultVulkanDevice()) return 77;
    ShaderDirectory(std::filesystem::path(argv[1]).parent_path().string());
    if (int r = SessionCooks()) return r;
    if (int r = DeformPoses(argv[2], argv[3])) return r;
    std::puts("Vulkan async cook: no timeline waits over warm cooks: PASS");
    return 0;
}
