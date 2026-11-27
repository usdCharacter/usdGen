// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/topologyPipeline.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

using usdGen::UsdGenExecutionResourceKind;
using usdGen::vulkan::ChargedBuffer;
using usdGen::vulkan::DeviceContext;
using usdGen::vulkan::TopologyPipeline;

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
// Host-visible storage buffer upload; the pipeline's pre-compute host barrier
// flushes the coherent writes before the shader reads them.
static std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const& c, void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;  // Vulkan cannot materialize zero-size buffers
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s topologyCompare.spv\n", argv[0]); return 1; }
    auto code = Code(argv[1]); CHECK(!code.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(native->physical, &props);
    if (props.limits.maxPerStageDescriptorStorageBuffers < 5 ||
        props.limits.maxDescriptorSetStorageBuffers < 5 ||
        props.limits.maxPushConstantsSize < 16) return 77;
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true; ci.instance = native->instance;
    ci.physicalDevice = native->physical; ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8011; ci.nativeLifetime = native; ci.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(ci); CHECK(context);
    VkResult status = VK_SUCCESS;
    auto pipe = TopologyPipeline::Create(context, code, &status);
    CHECK(pipe && status == VK_SUCCESS);

    // A: two curves, points [0,3). Offsets {0,2,3}, ids {100,200}.
    const uint32_t curves = 2, points = 3;
    uint32_t offsA[3] = {0, 2, 3};
    uint64_t idsA[2] = {100ull, 200ull};
    auto oA = Upload(context, offsA, sizeof(offsA));
    auto iA = Upload(context, idsA, sizeof(idsA));
    CHECK(oA && iA);

    // Identical set B => equal.
    auto oE = Upload(context, offsA, sizeof(offsA));
    auto iE = Upload(context, idsA, sizeof(idsA));
    CHECK(oE && iE);
    auto c1 = pipe->Begin(oA, iA, curves, points, oE, iE, curves, points, &status);
    CHECK(c1 && status == VK_SUCCESS);
    CHECK(Prove(native));
    uint32_t sem = UINT32_MAX;
    CHECK(c1->Poll(&sem) == VK_SUCCESS && sem == 0);
    CHECK(c1->succeeded() && c1->EqualResult() && c1->output());
    c1.reset();

    // Different stable id => unequal, well-formed (semantic 0).
    uint64_t idsB[2] = {100ull, 201ull};
    auto iB = Upload(context, idsB, sizeof(idsB)); CHECK(iB);
    auto c2 = pipe->Begin(oA, iA, curves, points, oA, iB, curves, points, &status);
    CHECK(c2 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c2->Poll(&sem) == VK_SUCCESS && sem == 0);
    CHECK(c2->succeeded() && !c2->EqualResult());
    c2.reset();

    // Different offsets (same point count) => unequal.
    uint32_t offsC[3] = {0, 1, 3};
    auto oC = Upload(context, offsC, sizeof(offsC)); CHECK(oC);
    auto c3 = pipe->Begin(oA, iA, curves, points, oC, iA, curves, points, &status);
    CHECK(c3 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c3->Poll(&sem) == VK_SUCCESS && sem == 0);
    CHECK(c3->succeeded() && !c3->EqualResult());
    c3.reset();

    // Different point count => unequal. The compare kernel requires the last
    // offset to equal pointCount, so the count change comes with its offsets.
    uint32_t offsD[3] = {0, 1, 4};
    auto oD = Upload(context, offsD, sizeof(offsD)); CHECK(oD);
    auto c4 = pipe->Begin(oA, iA, curves, points, oD, iA, curves, 4, &status);
    CHECK(c4 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c4->Poll(&sem) == VK_SUCCESS && sem == 0);
    CHECK(c4->succeeded() && !c4->EqualResult());
    c4.reset();

    // Empty set vs itself: canonical {0} offset array, no ids => equal.
    uint32_t emptyOffs[1] = {0};
    auto o0 = Upload(context, emptyOffs, sizeof(emptyOffs)); CHECK(o0);
    auto emptyIds = Upload(context, nullptr, 0); CHECK(emptyIds);
    auto c5 = pipe->Begin(o0, emptyIds, 0, 0, o0, emptyIds, 0, 0, &status);
    CHECK(c5 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c5->Poll(&sem) == VK_SUCCESS && sem == 0);
    CHECK(c5->succeeded() && c5->EqualResult());
    c5.reset();

    // Malformed offset: final offset != pointCount -> InvalidArgument (semantic != 0).
    uint32_t badOffs[3] = {0, 2, 5}; // 5 > pointCount 3
    auto oBad = Upload(context, badOffs, sizeof(badOffs)); CHECK(oBad);
    auto c6 = pipe->Begin(oA, iA, curves, points, oBad, iA, curves, points, &status);
    CHECK(c6 && status == VK_SUCCESS);
    CHECK(Prove(native));
    CHECK(c6->Poll(&sem) == VK_SUCCESS && sem != 0);
    CHECK(!c6->succeeded() && !c6->output());
    c6.reset();

    std::puts("Vulkan topologyCompare pipeline: PASS");
    return 0;
}