// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "vulkanNativeFixture.h"
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan lifecycle check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)
struct SubmissionOwner { std::shared_ptr<NativeOwner> native; };

int main(int argc, char** argv) {
    bool quarantine = argc == 2 && std::string(argv[1]) == "--quarantine";
    CHECK(argc == 1 || quarantine);
    VkResult status = VK_SUCCESS;
    CHECK(!DeviceContext::Create({}, &status) && status == VK_ERROR_INITIALIZATION_FAILED);
    VkBufferCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create.size = 1028; create.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(!ChargedBuffer::Create({}, create, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7002; // Isolated trusted factory fixture key.
    info.nativeLifetime = native; info.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(info, &status);
    CHECK(context && status == VK_SUCCESS);
    auto pool = context->resources();
    auto baseline = pool->Snapshot();
    auto unchanged = [&] {
        auto now = pool->Snapshot();
        return now.usedBytes == baseline.usedBytes && now.byKind == baseline.byKind;
    };
    for (int invalid = 0; invalid != 10; ++invalid) {
        auto bad = info;
        switch (invalid) {
        case 0: bad.instance = VK_NULL_HANDLE; break;
        case 1: bad.physicalDevice = VK_NULL_HANDLE; break;
        case 2: bad.device = VK_NULL_HANDLE; break;
        case 3: bad.computeQueue = VK_NULL_HANDLE; break;
        case 4: bad.computeQueueFamily = UINT32_MAX; break;
        case 5: bad.computeQueueFamily = UINT32_MAX - 1; break;
        case 6: bad.physicalIndex = -1; break;
        case 7: bad.resourceDeviceId = -1; break;
        case 8: bad.nativeLifetime.reset(); break;
        case 9: bad.resources.limitBytes = 0; break;
        }
        status = VK_SUCCESS;
        CHECK(!DeviceContext::Create(bad, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED && unchanged());
    }
    {
        auto same = DeviceContext::Create(info, &status);
        CHECK(same && same->resources() == pool && status == VK_SUCCESS);
        CHECK(same->deviceUUID() == context->deviceUUID());
        auto mismatch = info; mismatch.resources.limitBytes += 4096;
        CHECK(!DeviceContext::Create(mismatch, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED && unchanged());
    }
    for (int invalid = 0; invalid != 2; ++invalid) {
        auto bad = create;
        if (invalid == 0) bad.size = 0;
        else bad.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        status = VK_SUCCESS;
        CHECK(!ChargedBuffer::Create(context, bad, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            UsdGenExecutionResourceKind::Active, &status));
        CHECK(status == VK_ERROR_INITIALIZATION_FAILED && unchanged());
    }
    auto buffer = ChargedBuffer::Create(context, create, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        UsdGenExecutionResourceKind::Active, &status);
    CHECK(buffer && status == VK_SUCCESS);
    CHECK(buffer->context() == context && buffer->sizeBytes() == create.size);
    CHECK(buffer->usage() == create.usage);
    auto charged = pool->Snapshot();
    CHECK(charged.usedBytes == baseline.usedBytes + buffer->allocationBytes());
    CHECK(buffer->PollComplete() == VK_ERROR_INITIALIZATION_FAILED);
    CHECK(buffer->MarkSubmitted(VK_NULL_HANDLE, native) == VK_ERROR_INITIALIZATION_FAILED);
    CHECK(buffer->MarkSubmitted(native->fence, {}) == VK_ERROR_INITIALIZATION_FAILED);
    CHECK(!buffer->unproven());
    auto submission = std::make_shared<SubmissionOwner>(); submission->native = native;
    std::weak_ptr<SubmissionOwner> weakSubmission = submission;
    CHECK(buffer->MarkSubmitted(native->fence, submission) == VK_SUCCESS);
    submission.reset();
    CHECK(!weakSubmission.expired());
    CHECK(buffer->MarkSubmitted(native->fence, native) == VK_ERROR_INITIALIZATION_FAILED);
    CHECK(buffer->PollComplete() == VK_NOT_READY && buffer->unproven());
    CHECK(pool->Snapshot().usedBytes == charged.usedBytes);
    info.nativeLifetime.reset();
    if (quarantine) {
        // Destructor proof-loss injection: retain native owners and exact charge.
        buffer.reset();
        CHECK(!weakSubmission.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes);
    }
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    CHECK(vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS);
    CHECK(vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS);
    if (quarantine) {
        context.reset(); native.reset();
        CHECK(!weakSubmission.expired() && !weakNative.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes && pool->Snapshot().byKind == charged.byKind);
        return 0; // Later completion cannot reclaim a lost-proof owner.
    }
    CHECK(buffer->PollComplete() == VK_SUCCESS && !buffer->unproven());
    CHECK(weakSubmission.expired());
    CHECK(buffer->PollComplete() == VK_ERROR_INITIALIZATION_FAILED);
    CHECK(buffer->MarkSubmitted(native->fence, native) == VK_ERROR_INITIALIZATION_FAILED);
    context.reset(); native.reset(); CHECK(!weakNative.expired());
    buffer.reset(); CHECK(weakNative.expired() && unchanged());
    return 0;
}
