// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// rootFrameGather pipeline test: binary-search gather of 6 planes by stable-ID,
// missing-ID rejection, quarantine pinning.
// argv: <rootFrameGather.spv> [--quarantine]

#include "usdGenVulkan/vulkan/rootFrameGatherPipeline.h"
#include "vulkanNativeFixture.h"

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
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan RootFrameGather check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

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
    bi.size = values.empty() ? 4 : values.size() * sizeof(T);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Active);
    if (!buffer) return buffer;
    if (!values.empty()) {
        void* mapped = nullptr;
        if (vkMapMemory(native->device, buffer->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS)
            return std::shared_ptr<ChargedBuffer>{};
        std::memcpy(mapped, values.data(), values.size() * sizeof(T));
        vkUnmapMemory(native->device, buffer->memory());
    }
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

// Packed float3 helper: store 3 floats per element in a flat vector.
static void PackF3(std::vector<float>* out, float x, float y, float z) {
    out->push_back(x); out->push_back(y); out->push_back(z);
}

int main(int argc, char** argv) {
    bool quarantine = argc == 3 && std::string(argv[2]) == "--quarantine";
    CHECK(argc == 2 || quarantine);
    auto load = [](char const* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return std::vector<uint32_t>{};
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
        if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) return std::vector<uint32_t>{};
        std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
        std::memcpy(code.data(), bytes.data(), bytes.size());
        return code;
    };
    auto spv = load(argv[1]);
    CHECK(!spv.empty());

    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    {   // rootFrameGather set = 15 storage bindings; skip weak devices.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(native->physical, &props);
        if (props.limits.maxPerStageDescriptorStorageBuffers < 15 ||
            props.limits.maxDescriptorSetStorageBuffers < 15 ||
            props.limits.maxPushConstantsSize < 8) return 77;
    }
    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue;
    info.computeQueueFamily = native->family; info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7005; info.nativeLifetime = native;
    info.resources = {size_t{16} << 20, 0};
    auto context = DeviceContext::Create(info);
    CHECK(context); info.nativeLifetime.reset();
    auto pool = context->resources();
    VkResult status = VK_SUCCESS;
    CHECK(!RootFrameGatherPipeline::Create(context, {0x07230203u}, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);

    // 5 source frames, 3 survivors (indices 0, 2, 4 — non-contiguous).
    uint32_t const baseCount = 5, survivorCount = 3;
    std::vector<uint64_t> sourceIds = {10, 20, 30, 40, 50};
    std::vector<uint64_t> survivorIds = {10, 30, 50};
    // float3 planes: each source has distinct values per component.
    std::vector<float> srcOrigin, srcTangent, srcBinormal, srcNormal;
    std::vector<uint32_t> srcValid, srcDrop;
    for (uint32_t i = 0; i < baseCount; ++i) {
        float base = float(i + 1);
        PackF3(&srcOrigin,   base, base*2.f, base*3.f);
        PackF3(&srcTangent,  base*0.5f, base*1.5f, base*2.5f);
        PackF3(&srcBinormal, base*10.f, base*20.f, base*30.f);
        PackF3(&srcNormal,   base*0.1f, base*0.2f, base*0.3f);
        srcValid.push_back(i % 2 == 0 ? 1u : 0u);
        srcDrop.push_back(i == 3 ? 1u : 0u);
    }

    auto sourceIdsBuf = Upload(context, native, sourceIds);
    auto survivorIdsBuf = Upload(context, native, survivorIds);
    auto srcOriginBuf = Upload(context, native, srcOrigin);
    auto srcTangentBuf = Upload(context, native, srcTangent);
    auto srcBinormalBuf = Upload(context, native, srcBinormal);
    auto srcNormalBuf = Upload(context, native, srcNormal);
    auto srcValidBuf = Upload(context, native, srcValid);
    auto srcDropBuf = Upload(context, native, srcDrop);
    CHECK(sourceIdsBuf && survivorIdsBuf && srcOriginBuf && srcTangentBuf &&
          srcBinormalBuf && srcNormalBuf && srcValidBuf && srcDropBuf);

    auto pipeline = RootFrameGatherPipeline::Create(context, spv, &status);
    CHECK(pipeline && status == VK_SUCCESS);

    auto makeInfo = [&](std::shared_ptr<const ChargedBuffer> sIds,
                        std::shared_ptr<const ChargedBuffer> sOrigin,
                        std::shared_ptr<const ChargedBuffer> sTangent,
                        std::shared_ptr<const ChargedBuffer> sBinormal,
                        std::shared_ptr<const ChargedBuffer> sNormal,
                        std::shared_ptr<const ChargedBuffer> sValid,
                        std::shared_ptr<const ChargedBuffer> sDrop,
                        std::shared_ptr<const ChargedBuffer> survIds,
                        uint32_t baseN, uint32_t survN) {
        RootFrameGatherPipeline::BeginInfo bi;
        bi.sourceIds = sIds; bi.sourceOrigins = sOrigin;
        bi.sourceTangent = sTangent; bi.sourceBinormal = sBinormal;
        bi.sourceNormal = sNormal; bi.sourceValid = sValid;
        bi.sourceDrop = sDrop; bi.survivorIds = survIds;
        bi.baseCount = baseN; bi.survivorCount = survN;
        return bi;
    };

    // ------------------------------------------------------------ quarantine
    if (quarantine) {
        auto bi = makeInfo(sourceIdsBuf, srcOriginBuf, srcTangentBuf, srcBinormalBuf,
                           srcNormalBuf, srcValidBuf, srcDropBuf, survivorIdsBuf,
                           baseCount, survivorCount);
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS && cand->survivorCount() == survivorCount);
        auto charged = pool->Snapshot();
        std::weak_ptr<const ChargedBuffer> weakIn = cand->inputOwner();
        cand->Quarantine(); cand.reset(); pipeline.reset(); context.reset();
        CHECK(Prove(native)); native.reset();
        CHECK(!weakNative.expired());
        CHECK(!weakIn.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes &&
              pool->Snapshot().byKind == charged.byKind);
        return 0;
    }

    // ------------------------------------------------------- (a) clean gather
    auto const inputsOnly = pool->Snapshot();
    {
        auto bi = makeInfo(sourceIdsBuf, srcOriginBuf, srcTangentBuf, srcBinormalBuf,
                           srcNormalBuf, srcValidBuf, srcDropBuf, survivorIdsBuf,
                           baseCount, survivorCount);
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS);
        CHECK(!cand->output().origins);  // hidden pre-proof
        CHECK(Prove(native));
        RootFrameGatherSemantic semantic = RootFrameGatherSemantic::MissingId;
        CHECK(cand->Poll(&semantic) == VK_SUCCESS);
        CHECK(semantic == RootFrameGatherSemantic::Ok && cand->succeeded());
        auto out = cand->output();
        CHECK(out.origins && out.tangent && out.binormal && out.normal &&
              out.valid && out.drop);
        CHECK(out.origins->context() == context);
        // Expected: survivors are source indices 0, 2, 4.
        uint32_t const expectedIdx[] = {0, 2, 4};
        std::vector<float> got(3 * survivorCount);
        std::vector<uint32_t> gotValid(survivorCount), gotDrop(survivorCount);
        CHECK(Download(native, out.valid, &gotValid));
        CHECK(Download(native, out.drop, &gotDrop));
        for (uint32_t s = 0; s < survivorCount; ++s) {
            uint32_t const src = expectedIdx[s];
            CHECK(gotValid[s] == srcValid[src]);
            CHECK(gotDrop[s] == srcDrop[src]);
        }
        for (int plane = 0; plane < 4; ++plane) {
            auto const& buf = plane == 0 ? out.origins : plane == 1 ? out.tangent :
                plane == 2 ? out.binormal : out.normal;
            CHECK(Download(native, buf, &got));
            for (uint32_t s = 0; s < survivorCount; ++s) {
                uint32_t const src = expectedIdx[s];
                float const base = float(src + 1);
                float want[3];
                switch (plane) {
                    case 0: want[0]=base; want[1]=base*2.f; want[2]=base*3.f; break;
                    case 1: want[0]=base*0.5f; want[1]=base*1.5f; want[2]=base*2.5f; break;
                    case 2: want[0]=base*10.f; want[1]=base*20.f; want[2]=base*30.f; break;
                    default: want[0]=base*0.1f; want[1]=base*0.2f; want[2]=base*0.3f; break;
                }
                for (int c = 0; c < 3; ++c) {
                    float diff = std::abs(got[3*s+c] - want[c]);
                    CHECK(diff < 1e-5f || diff <= 1e-5f * std::abs(want[c]));
                }
            }
        }
        out = {};
        cand.reset();
        CHECK(pool->Snapshot().usedBytes == inputsOnly.usedBytes &&
              pool->Snapshot().byKind == inputsOnly.byKind);
    }


    // --------------------------------------- (c) rejection: null inputs
    {
        auto bi = makeInfo(nullptr, srcOriginBuf, srcTangentBuf, srcBinormalBuf,
                           srcNormalBuf, srcValidBuf, srcDropBuf, survivorIdsBuf,
                           baseCount, survivorCount);
        RootFrameGatherSemantic code = RootFrameGatherSemantic::Ok;
        auto cand = pipeline->Begin(bi, &status, &code);
        CHECK(!cand && status == VK_ERROR_INITIALIZATION_FAILED);
        CHECK(pool->Snapshot().usedBytes == inputsOnly.usedBytes &&
              pool->Snapshot().byKind == inputsOnly.byKind);
    }

    // --------------------------------------- (d) rejection: size mismatch
    {
        auto bi = makeInfo(sourceIdsBuf, srcOriginBuf, srcTangentBuf, srcBinormalBuf,
                           srcNormalBuf, srcValidBuf, srcDropBuf, survivorIdsBuf,
                           baseCount + 1, survivorCount);  // claim 6 sources but have 5
        auto cand = pipeline->Begin(bi, &status);
        CHECK(!cand && status == VK_ERROR_INITIALIZATION_FAILED);
    }

    // --------------------------------------- (e) missing ID rejection
    {
        // Survivor id 99 is not in sourceIds; shader reports MissingId.
        std::vector<uint64_t> badSurvivors = {10, 99, 50};
        auto badSurv = Upload(context, native, badSurvivors);
        CHECK(badSurv);
        auto bi = makeInfo(sourceIdsBuf, srcOriginBuf, srcTangentBuf, srcBinormalBuf,
                           srcNormalBuf, srcValidBuf, srcDropBuf, badSurv,
                           baseCount, survivorCount);
        auto cand = pipeline->Begin(bi, &status);
        CHECK(cand && status == VK_SUCCESS);
        CHECK(Prove(native));
        RootFrameGatherSemantic semantic = RootFrameGatherSemantic::Ok;
        CHECK(cand->Poll(&semantic) == VK_SUCCESS);
        CHECK(semantic == RootFrameGatherSemantic::MissingId);
        CHECK(!cand->succeeded());
        CHECK(!cand->output().origins);
        cand.reset();  // polled (pending=false) -> no quarantine, buffers freed
    }

    // Release input buffers so the context can be destroyed.
    sourceIdsBuf = srcOriginBuf = srcTangentBuf = srcBinormalBuf = srcNormalBuf =
                   srcValidBuf = srcDropBuf = survivorIdsBuf = nullptr;
    pipeline.reset();
    context.reset();
    CHECK(Prove(native));
    native.reset();
    CHECK(weakNative.expired());
    return 0;
}
