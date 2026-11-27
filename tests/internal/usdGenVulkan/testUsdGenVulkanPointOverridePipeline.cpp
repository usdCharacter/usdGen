// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/pointOverridePipeline.h"

#include "hosts/usdGenUsd/vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

using usdGen::UsdGenExecutionResourceKind;
using usdGen::vulkan::ChargedBuffer;
using usdGen::vulkan::DeviceContext;
using usdGen::vulkan::PointOverridePipeline;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

static std::vector<uint32_t> Code(char const* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}
static std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const& c, void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
static bool FloatsEqual(std::vector<uint8_t> const& bytes, float const* expected, size_t n) {
    if (bytes.size() != n * sizeof(float)) return false;
    float const* got = reinterpret_cast<float const*>(bytes.data());
    for (size_t i = 0; i != n; ++i)
        if (got[i] != expected[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s pointOverride.spv\n", argv[0]); return 1; }
    auto code = Code(argv[1]); CHECK(!code.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(native->physical, &props);
    if (props.limits.maxPerStageDescriptorStorageBuffers < 6 ||
        props.limits.maxDescriptorSetStorageBuffers < 6 ||
        props.limits.maxPushConstantsSize < 12) return 77;
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true; ci.instance = native->instance;
    ci.physicalDevice = native->physical; ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8012; ci.nativeLifetime = native; ci.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(ci); CHECK(context);
    VkResult status = VK_SUCCESS;
    auto pipe = PointOverridePipeline::Create(context, code, &status);
    CHECK(pipe && status == VK_SUCCESS);

    // Base of 3 points.
    const uint32_t baseN = 3;
    float base[9] = {0.f, 1.f, 2.f, 10.f, 20.f, 30.f, 100.f, 200.f, 300.f};
    auto baseBuf = Upload(context, base, sizeof(base)); CHECK(baseBuf);
    auto readOut = [&](std::unique_ptr<PointOverridePipeline::Candidate> const& c,
                       std::vector<uint8_t>* out) {
        return ReadVulkanBytes(native, context, c->output()->buffer(),
            c->output()->sizeBytes(), c->output(), out);
    };

    // (1) No-op empty indices: exact base copy.
    auto emptyIdx = Upload(context, nullptr, 0); CHECK(emptyIdx);
    auto emptyRep = Upload(context, nullptr, 0); CHECK(emptyRep);
    auto c1 = pipe->Begin(baseBuf, emptyIdx, emptyRep, baseN, 0, &status);
    CHECK(c1 && status == VK_SUCCESS);
    CHECK(Prove(native));
    PointOverridePipeline::Semantic sem = PointOverridePipeline::Semantic::InvalidArgument;
    CHECK(c1->Poll(&sem) == VK_SUCCESS && sem == PointOverridePipeline::Semantic::Ok);
    CHECK(c1->succeeded() && c1->output());
    std::vector<uint8_t> out1; CHECK(readOut(c1, &out1));
    CHECK(FloatsEqual(out1, base, 9));
    c1.reset();

    // (2) Single override index 1.
    int32_t idx1[1] = {1};
    float rep1[3] = {7.f, 8.f, 9.f};
    auto idxB = Upload(context, idx1, sizeof(idx1)); CHECK(idxB);
    auto repB = Upload(context, rep1, sizeof(rep1)); CHECK(repB);
    auto c2 = pipe->Begin(baseBuf, idxB, repB, baseN, 1, &status);
    CHECK(c2 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c2->Poll(&sem) == VK_SUCCESS && sem == PointOverridePipeline::Semantic::Ok);
    CHECK(c2->succeeded() && c2->output());
    std::vector<uint8_t> out2; CHECK(readOut(c2, &out2));
    float expectSingle[9] = {0.f, 1.f, 2.f, 7.f, 8.f, 9.f, 100.f, 200.f, 300.f};
    CHECK(FloatsEqual(out2, expectSingle, 9));
    c2.reset();

    // (3) Override ordering independence: {2,0} and {0,2} must match.
    int32_t ordA[2] = {2, 0};
    float repA[6] = {10.f, 11.f, 12.f, 20.f, 21.f, 22.f};
    int32_t ordB[2] = {0, 2};
    float repB2[6] = {20.f, 21.f, 22.f, 10.f, 11.f, 12.f};
    auto ia = Upload(context, ordA, sizeof(ordA)); auto ra = Upload(context, repA, sizeof(repA));
    auto ib = Upload(context, ordB, sizeof(ordB)); auto rb = Upload(context, repB2, sizeof(repB2));
    CHECK(ia && ra && ib && rb);
    auto ca = pipe->Begin(baseBuf, ia, ra, baseN, 2, &status); CHECK(ca && status == VK_SUCCESS);
    CHECK(Prove(native)); CHECK(ca->Poll(&sem) == VK_SUCCESS && ca->succeeded());
    std::vector<uint8_t> oa; CHECK(readOut(ca, &oa));
    auto cb = pipe->Begin(baseBuf, ib, rb, baseN, 2, &status); CHECK(cb && status == VK_SUCCESS);
    CHECK(Prove(native)); CHECK(cb->Poll(&sem) == VK_SUCCESS && cb->succeeded());
    std::vector<uint8_t> ob; CHECK(readOut(cb, &ob));
    CHECK(oa == ob);
    float expectOrder[9] = {20.f, 21.f, 22.f, 10.f, 20.f, 30.f, 10.f, 11.f, 12.f};
    CHECK(FloatsEqual(oa, expectOrder, 9));
    ca.reset(); cb.reset();

    // (4) Out-of-range index -> InvalidArgument.
    int32_t badIdx[1] = {5};
    auto ib1 = Upload(context, badIdx, sizeof(badIdx)); CHECK(ib1);
    auto c4 = pipe->Begin(baseBuf, ib1, repB, baseN, 1, &status);
    CHECK(c4 && status == VK_SUCCESS);   // synchronous validation failure candidate
    CHECK(c4->Poll(&sem) == VK_SUCCESS && sem == PointOverridePipeline::Semantic::InvalidArgument);
    CHECK(!c4->succeeded() && !c4->output());
    c4.reset();

    // (5) Non-finite base -> NonFinite.
    float nanBase[9] = {0.f, NAN, 2.f, 10.f, 20.f, 30.f, 100.f, 200.f, 300.f};
    auto nanBuf = Upload(context, nanBase, sizeof(nanBase)); CHECK(nanBuf);
    auto c5 = pipe->Begin(nanBuf, ib1, repB, baseN, 1, &status);
    CHECK(c5 && status == VK_SUCCESS);
    CHECK(c5->Poll(&sem) == VK_SUCCESS && sem == PointOverridePipeline::Semantic::NonFinite);
    CHECK(!c5->succeeded() && !c5->output());
    c5.reset();

    // (6) Duplicate indices -> Duplicate.
    int32_t dupIdx[2] = {1, 1};
    float dupRep[6] = {4.f, 4.f, 4.f, 5.f, 5.f, 5.f};
    auto id = Upload(context, dupIdx, sizeof(dupIdx)); auto rd = Upload(context, dupRep, sizeof(dupRep));
    CHECK(id && rd);
    auto c6 = pipe->Begin(baseBuf, id, rd, baseN, 2, &status);
    CHECK(c6 && status == VK_SUCCESS);
    CHECK(c6->Poll(&sem) == VK_SUCCESS && sem == PointOverridePipeline::Semantic::Duplicate);
    CHECK(!c6->succeeded() && !c6->output());
    c6.reset();

    std::puts("Vulkan pointOverride pipeline: PASS");
    return 0;
}