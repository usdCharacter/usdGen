// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "vulkanReadbackFixture.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan source accounting check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static VulkanSourceGenerationCreateInfo MinimalSource(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info;
    info.context = std::move(context);
    auto& source = info.source;
    source.totalCurves = 1;
    source.totalCvs = 4;
    source.topologyVersion = 1;
    source.valueVersion = 1;
    source.px = {0.f, 1.f, 2.f, 3.f};
    source.py = {0.f, 0.f, 0.f, 0.f};
    source.pz = {0.f, 0.f, 0.f, 0.f};
    source.width = {1.f, 1.f, 1.f, 1.f};
    source.curveId = {42};
    source.cvOffsets = {0, 4};
    return info;
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!raw.empty() && raw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());

    bool unavailable = false;
    auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 7015;
    ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true;
    ci.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(ci);
    CHECK(context);
    ci.nativeLifetime.reset();
    auto pool = context->resources();
    auto const baseline = pool->Snapshot();

    VkResult status = VK_SUCCESS;
    std::string reason;
    auto capture = MinimalSource(context);
    VulkanSourcePrepareInfo prepareInfo{capture.source, capture.geometry, capture.additionalNamed};
    auto prepared = VulkanPreparedSource::Prepare(prepareInfo, &reason);
    CHECK(prepared && pool->Snapshot().usedBytes == baseline.usedBytes);
    // Packing owns its bytes: later changes to either caller copy cannot
    // affect upload, and sharing the CPU packet makes no native allocation.
    capture.source.width[0] = 91.f;
    prepareInfo.source.width[0] = 92.f;
    auto upload = VulkanSourceUpload::Create(context, prepared, &status, &reason);
    CHECK(upload && status == VK_SUCCESS);
    CHECK(upload->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    auto base = upload->TakeReady();
    CHECK(base);
    auto const rootBytes = base->ExclusiveRetainedBytes();
    CHECK(rootBytes && base->InclusiveRetainedBytes() == rootBytes);
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes);

    std::vector<uint8_t> copied;
    CHECK(ReadVulkanBytes(native, context, base->PlaneOwner("width")->buffer(),
        4 * sizeof(float), base, &copied));
    float capturedWidth[4]{};
    std::memcpy(capturedWidth, copied.data(), sizeof(capturedWidth));
    for (float value : capturedWidth) CHECK(value == 1.f);

    // A second upload from one immutable host packet owns independent native
    // storage. This differs from TakeReady aliases and width-only COW below.
    auto secondUpload = VulkanSourceUpload::Create(context, prepared, &status, &reason);
    CHECK(secondUpload && secondUpload->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(secondUpload->Poll() == SourceGenerationStatus::Ready);
    auto secondBase = secondUpload->TakeReady();
    CHECK(secondBase && secondBase->PlaneOwner("points") != base->PlaneOwner("points"));
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes + secondBase->ExclusiveRetainedBytes());
    secondUpload.reset(); secondBase.reset(); prepared.reset();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes);

    // Multiple ready wrappers alias the one immutable source packet.  They
    // must not produce another physical pool charge, so do not add their
    // per-generation reports together.
    auto readyAlias = upload->TakeReady();
    CHECK(readyAlias && readyAlias->PlaneOwner("points") == base->PlaneOwner("points"));
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes);
    readyAlias.reset();
    upload.reset();

    auto pipeline = WidthPipeline::Create(context, code, &status);
    CHECK(pipeline);
    auto first = pipeline->Begin(base->PlaneOwner("width"), base->pointCount(), 2.f, 0, &status);
    CHECK(first && Prove(native));
    uint32_t semantic = UINT32_MAX;
    CHECK(first->Poll(&semantic) == VK_SUCCESS && semantic == 0 && first->succeeded());
    auto const firstBytes = size_t(first->output()->allocationBytes());
    auto child = VulkanSourceGeneration::WithWidth(base, *first, 2, &reason);
    CHECK(child && child->ExclusiveRetainedBytes() == firstBytes);
    CHECK(child->InclusiveRetainedBytes() == rootBytes + firstBytes);
    auto const afterChild = pool->Snapshot().usedBytes;

    // Candidate output adoption is an alias: two sibling wrappers reference
    // one allocation, and creating the second must not reserve again.
    auto sibling = VulkanSourceGeneration::WithWidth(base, *first, 3, &reason);
    CHECK(sibling && sibling->PlaneOwner("width") == child->PlaneOwner("width"));
    CHECK(pool->Snapshot().usedBytes == afterChild);
    first.reset(); // Releases its status scratch allocation, not the adopted output.
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes + firstBytes);

    auto second = pipeline->Begin(child->PlaneOwner("width"), child->pointCount(), 3.f, 0, &status);
    CHECK(second && Prove(native));
    CHECK(second->Poll(&semantic) == VK_SUCCESS && semantic == 0 && second->succeeded());
    auto const secondBytes = size_t(second->output()->allocationBytes());
    auto grandchild = VulkanSourceGeneration::WithWidth(child, *second, 4, &reason);
    CHECK(grandchild && grandchild->ExclusiveRetainedBytes() == secondBytes);
    CHECK(grandchild->InclusiveRetainedBytes() == rootBytes + firstBytes + secondBytes);
    second.reset();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes + firstBytes + secondBytes);

    // The last descendant retains both predecessors.  It is the sole live
    // ownership path after these resets, and releasing it returns every
    // source/COW permit to the baseline.
    std::weak_ptr<const VulkanSourceGeneration> weakBase = base;
    std::weak_ptr<const VulkanSourceGeneration> weakChild = child;
    sibling.reset();
    base.reset();
    child.reset();
    CHECK(!weakBase.expired() && !weakChild.expired());
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes + rootBytes + firstBytes + secondBytes);
    grandchild.reset();
    CHECK(weakBase.expired() && weakChild.expired());
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes);
    return 0;
}
