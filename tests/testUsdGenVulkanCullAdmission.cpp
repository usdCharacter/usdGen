// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan cull admission check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

using Packet = std::map<std::string, std::vector<uint8_t>>;

static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> result(raw.size() / sizeof(uint32_t));
    std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static VulkanSourceGenerationCreateInfo Fixture(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& s = info.source;
    s.totalCurves = 4; s.totalCvs = 8; s.topologyVersion = 41; s.valueVersion = 42;
    s.px = {0,1, 0,1, 0,1, 0,1}; s.py = VtFloatArray(8, 0.f); s.pz = VtFloatArray(8, 0.f);
    s.rest.reserve(8); s.width.reserve(8); s.hairT.reserve(8);
    for (uint32_t i = 0; i != 8; ++i) {
        s.rest.push_back(GfVec3f(float(i), 0, 0)); s.width.push_back(float(i) + .5f);
        s.hairT.push_back(float(i) / 8.f);
    }
    s.cvOffsets = {0,2,4,6,8}; s.curveId = {10,11,12,13};
    s.rootPrim = {20,21,22,23}; s.rootUV = {{0,0},{1,0},{0,1},{1,1}};
    s.rootT = {{1,0,0},{1,0,0},{1,0,0},{1,0,0}};
    s.rootB = {{0,1,0},{0,1,0},{0,1,0},{0,1,0}};
    s.rootN = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
    UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{1.f,1.f,1.f,1.f}}; s.extraCurve = {maskPlane};
    s.chunks = {{0,2,2,0,0,3,1,{}}, {2,2,2,4,0,4,1,{}}};
    info.geometry.alreadyDeformed = true;
    info.geometry.tiles = {{3,0,2,0,4}, {4,2,2,4,4}};
    return info;
}

static bool Capture(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const VulkanSourceGeneration> const& generation, Packet* packet) {
    packet->clear();
    auto add = [&](VulkanSourceGeneration::PlaneView const& plane) {
        std::vector<uint8_t> bytes;
        return ReadVulkanBytes(native, context, plane.buffer, plane.bytes, generation, &bytes) &&
            packet->emplace(plane.metadata.name, std::move(bytes)).second;
    };
    for (auto const& plane : generation->planes()) if (!add(plane)) return false;
    for (auto const& plane : generation->sourceFrames()) if (!add(plane)) return false;
    return true;
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    auto code = Code(argv[1]); CHECK(!code.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 7053; ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true;
    ci.resources = {size_t{16} << 20, 0};
    auto context = DeviceContext::Create(ci); CHECK(context);
    VkResult status = VK_SUCCESS;
    auto baseUpload = VulkanSourceUpload::Create(Fixture(context), &status); CHECK(baseUpload);
    CHECK(baseUpload->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(baseUpload->Poll() == SourceGenerationStatus::Ready);
    auto base = baseUpload->TakeReady(); CHECK(base && base->curveCount() == 4 && base->pointCount() == 8);
    baseUpload.reset();
    Packet original; CHECK(Capture(native, context, base, &original));
    auto pipeline = LengthCompactionPipeline::Create(context, code, &status); CHECK(pipeline && status == VK_SUCCESS);
    auto pool = context->resources();

    // Exhaust the actual usable budget before phase 0.  Begin must fail before
    // its counts submission/admission callback, and must return every charge.
    auto beforeCounts = pool->Snapshot();
    auto pressure = pool->TryReserve(beforeCounts.usableBytes - beforeCounts.usedBytes,
                                     UsdGenExecutionResourceKind::Scratch);
    CHECK(pressure);
    unsigned countHook = 0;
    CHECK(!pipeline->Begin(base, 1.f, &status, [&] { ++countHook; return true; }));
    CHECK(countHook == 0 && pool->Snapshot().usedBytes == beforeCounts.usedBytes + pressure->Bytes());
    pressure->Release();
    CHECK(pool->Snapshot().usedBytes == beforeCounts.usedBytes);

    // Establish the exact phase-2 charge from a real admitted candidate, then
    // retry with one byte less headroom. This is ledger admission, not timing.
    auto measured = pipeline->Begin(base, 1.f, &status); CHECK(measured && Prove(native));
    uint32_t semantic = UINT32_MAX;
    CHECK(measured->PollCounts(&semantic) == VK_SUCCESS && semantic == 0);
    auto phase2Before = pool->Snapshot();
    CHECK(measured->BeginScatter(&status) && Prove(native));
    CHECK(measured->PollScatter(&semantic) == VK_SUCCESS && semantic == 0 && measured->succeeded());
    auto phase2After = pool->Snapshot();
    CHECK(phase2After.usedBytes > phase2Before.usedBytes);
    size_t phase2Bytes = phase2After.usedBytes - phase2Before.usedBytes;
    measured.reset();
    CHECK(pool->Snapshot().usedBytes == beforeCounts.usedBytes);

    auto rejected = pipeline->Begin(base, 1.f, &status); CHECK(rejected && Prove(native));
    CHECK(rejected->PollCounts(&semantic) == VK_SUCCESS && semantic == 0);
    auto available = pool->Snapshot();
    CHECK(available.usableBytes - available.usedBytes >= phase2Bytes);
    auto phase2Admission = pool->TryReserve(
        available.usableBytes - available.usedBytes - (phase2Bytes - 1), UsdGenExecutionResourceKind::Scratch);
    CHECK(phase2Admission);
    auto phase2Baseline = pool->Snapshot();
    unsigned scatterHook = 0;
    CHECK(!rejected->BeginScatter(&status, [&] { ++scatterHook; return true; }));
    CHECK(scatterHook == 0 && !rejected->succeeded());
    // A failed unsubmitted candidate may retain partially allocated scratch
    // until destruction; none may escape that owner or remain charged after it.
    CHECK(pool->Snapshot().usedBytes >= phase2Baseline.usedBytes);
    rejected.reset(); phase2Admission->Release();
    CHECK(pool->Snapshot().usedBytes == beforeCounts.usedBytes);
    Packet after; CHECK(Capture(native, context, base, &after));
    CHECK(after == original && base->curveCount() == 4 && base->pointCount() == 8);
    pipeline.reset(); base.reset();
    CHECK(pool->Snapshot().usedBytes == 0);
    std::puts("Vulkan cull resource admission: PASS");
    return 0;
}
