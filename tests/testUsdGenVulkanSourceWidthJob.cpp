// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceWidthJob.h"
#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

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
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan source width job check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context,
    bool withWidth = true) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& b = info.source;
    b.totalCurves = 1; b.totalCvs = 2; b.topologyVersion = 3; b.valueVersion = 4;
    b.px = {0, 1}; b.py = {0, 1}; b.pz = {0, 1}; b.curveId = {7}; b.cvOffsets = {0, 2};
    if (withWidth) b.width = {1.f, 2.f};
    return info;
}

static std::vector<uint32_t> ReadSpirv(char const* path) {
    std::ifstream shader(path, std::ios::binary);
    if (!shader) return {};
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());
    return code;
}

int main(int argc, char** argv) {
    bool const testDag = argc == 4 && std::string(argv[2]) == "--dag";
    bool const testCross = argc == 6 && std::string(argv[2]) == "--cross";
    bool const testCull = argc == 4 && std::string(argv[2]) == "--cull";
    bool const testDisabled = argc == 3 && std::string(argv[2]) == "--disabled";
    bool const testTiles = argc == 5 && std::string(argv[2]) == "--tiles";
    CHECK(argc == 2 || testDisabled || testTiles ||
        (argc == 4 && (std::string(argv[2]) == "--length" || testDag || testCull)) || testCross);
    std::vector<uint32_t> code = ReadSpirv(argv[1]);
    CHECK(!code.empty());
    bool const testLength = argc == 4 && std::string(argv[2]) == "--length";
    std::vector<uint32_t> lengthCode;
    if (testLength && !testDag) {
        lengthCode = ReadSpirv(argv[3]);
        CHECK(!lengthCode.empty());
    }
    std::vector<uint32_t> cullCode;
    if (testCull) { cullCode = ReadSpirv(argv[3]); CHECK(!cullCode.empty()); }
    std::vector<uint32_t> tilesCode, boundsCode;
    if (testTiles) {
        tilesCode = ReadSpirv(argv[3]); CHECK(!tilesCode.empty());
        boundsCode = ReadSpirv(argv[4]); CHECK(!boundsCode.empty());
    }
    std::vector<uint32_t> blendCode;
    if (testDag) { blendCode = ReadSpirv(argv[3]); CHECK(!blendCode.empty()); }
    std::vector<uint32_t> compareCode;
    if (testCross) {
        blendCode = ReadSpirv(argv[3]); CHECK(!blendCode.empty());
        lengthCode = ReadSpirv(argv[4]); CHECK(!lengthCode.empty());
        compareCode = ReadSpirv(argv[5]); CHECK(!compareCode.empty());
    }
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue; ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex; ci.resourceDeviceId = 7014;
    ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true; ci.resources = {size_t{8} << 20, 0};
    VkResult status = VK_ERROR_UNKNOWN; auto context = DeviceContext::Create(ci, &status);
    CHECK(context && status == VK_SUCCESS);
    auto pipeline = WidthPipeline::Create(context, code, &status); CHECK(pipeline && status == VK_SUCCESS);
    std::shared_ptr<LengthScalePipeline> lengthPipeline;
    if (testLength || testCross) {
        lengthPipeline = LengthScalePipeline::Create(context, lengthCode, &status);
        CHECK(lengthPipeline && status == VK_SUCCESS);
    }
    std::shared_ptr<LengthCompactionPipeline> cullPipeline;
    if (testCull) {
        cullPipeline = LengthCompactionPipeline::Create(context, cullCode, &status);
        CHECK(cullPipeline && status == VK_SUCCESS);
    }
    UsdGenExecutionRuntime runtime{2};

    // Creation validates all controls before any owner admission.
    auto lifetime = std::make_shared<int>(1);
    VulkanSourceWidthJob::CreateInfo base{Source(context), pipeline, 3.f, 1, 5, lifetime, {}, {}, {}};
    std::string reason;
    auto bad = base; bad.width = -1.f; CHECK(!VulkanSourceWidthJob::Create(bad, &reason));
    bad = base; bad.replace = 2; CHECK(!VulkanSourceWidthJob::Create(bad, &reason));
    bad = base; bad.valueVersion = 4; CHECK(!VulkanSourceWidthJob::Create(bad, &reason));
    bad = base; bad.requestLifetime.reset(); CHECK(!VulkanSourceWidthJob::Create(bad, &reason));
    bad = base; bad.widthPipeline.reset(); CHECK(!VulkanSourceWidthJob::Create(bad, &reason));

    if (testDag) {
        auto blendPipeline = WidthBlendPipeline::Create(context, blendCode, &status);
        CHECK(blendPipeline && status == VK_SUCCESS);
        UsdGenExecutionPipeline owner(runtime);
        auto request = std::make_shared<int>(8); std::weak_ptr<int> weakRequest = request;
        auto dagInfo = base;
        dagInfo.requestLifetime = request;
        dagInfo.widthBlendPipeline = blendPipeline;
        dagInfo.width = 0.f; dagInfo.replace = 0;
        dagInfo.valueVersion = 7;
        VulkanSourceWidthStage left; left.kind = VulkanSourceWidthStage::Kind::Width;
        left.input = 0; left.width.width = 1.f; left.width.replace = false;
        VulkanSourceWidthStage right; right.kind = VulkanSourceWidthStage::Kind::Width;
        right.input = 0; right.width.width = 3.f; right.width.replace = false;
        VulkanSourceWidthStage blend; blend.kind = VulkanSourceWidthStage::Kind::WidthBlend;
        blend.input = 1; blend.rightInput = 2; blend.blend = .25f;
        dagInfo.stages = {left, right, blend}; dagInfo.stageValueVersions = {5, 6, 7};
        auto job = VulkanSourceWidthJob::Create(std::move(dagInfo), &reason); CHECK(job);
        request.reset();
        int callbacks = 0; bool callbackOk = false;
        std::shared_ptr<const VulkanSourceGeneration> result;
        CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
            ++callbacks; result = std::move(generation);
            callbackOk = !weakRequest.expired() && !result &&
                state == VulkanSourceWidthJob::State::Superseded && resultStatus == VK_SUCCESS;
        }));
        owner.Drain(); CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(job->state() == VulkanSourceWidthJob::State::WidthPending);
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(job->state() == VulkanSourceWidthJob::State::WidthPending);
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(job->state() == VulkanSourceWidthJob::State::BlendPending);
        job->SuppressPublication(); CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(callbackOk && callbacks == 1 && !result && job->state() == VulkanSourceWidthJob::State::Superseded);
        CHECK(owner.OutstandingCommands() == 0 && weakRequest.expired());
        job.reset(); owner.Drain();
        CHECK(context->resources()->Snapshot().usedBytes == 0);
        CHECK(owner.OutstandingCommands() == 0);
    }

    if (testCross) {
        auto blendPipeline = WidthBlendPipeline::Create(context, blendCode, &status);
        CHECK(blendPipeline && status == VK_SUCCESS);
        auto comparePipeline = NonWidthComparePipeline::Create(context, compareCode, &status);
        CHECK(comparePipeline && status == VK_SUCCESS);
        UsdGenExecutionPipeline owner(runtime);
        auto request = std::make_shared<int>(9); std::weak_ptr<int> weakRequest = request;
        auto cross = base;
        cross.requestLifetime = request;
        cross.lengthPipeline = lengthPipeline;
        cross.widthBlendPipeline = blendPipeline;
        cross.nonWidthComparePipeline = comparePipeline;
        cross.valueVersion = 7;
        cross.lengthValueVersion = 0; // Explicit stages carry the revisions.
        VulkanSourceWidthStage left; left.kind = VulkanSourceWidthStage::Kind::LengthScale;
        left.factor = 0.5f; left.input = 0;
        VulkanSourceWidthStage right; right.kind = VulkanSourceWidthStage::Kind::LengthScale;
        right.factor = 1.5f; right.input = 0;
        VulkanSourceWidthStage blend; blend.kind = VulkanSourceWidthStage::Kind::WidthBlend;
        blend.input = 1; blend.rightInput = 2; blend.blend = .5f;
        cross.stages = {left, right, blend};
        cross.stageValueVersions = {5, 6, 7};
        auto job = VulkanSourceWidthJob::Create(std::move(cross), &reason); CHECK(job);
        request.reset();
        int callbacks = 0; bool callbackOk = false;
        std::shared_ptr<const VulkanSourceGeneration> result;
        CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
            ++callbacks; result = std::move(generation);
            callbackOk = !weakRequest.expired() && !result &&
                state == VulkanSourceWidthJob::State::Superseded && resultStatus == VK_SUCCESS;
        }));
        owner.Drain();                         // Created -> UploadPending
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(job->state() == VulkanSourceWidthJob::State::LengthPending);
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain(); // left Length
        CHECK(job->state() == VulkanSourceWidthJob::State::LengthPending);
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain(); // right Length
        CHECK(job->state() == VulkanSourceWidthJob::State::ComparePending);
        auto const before = context->resources()->Snapshot();
        job->SuppressPublication();
        CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(callbackOk && callbacks == 1 && !result &&
              job->state() == VulkanSourceWidthJob::State::Superseded);
        CHECK(owner.OutstandingCommands() == 0 && weakRequest.expired());
        job.reset(); owner.Drain();
        CHECK(context->resources()->Snapshot().usedBytes == 0);
        CHECK(context->resources()->Snapshot().usedBytes <= before.usedBytes);
    }

    // Suppression before Start settles on the owner without creating native work.
    {
        UsdGenExecutionPipeline owner(runtime, 4, 2);
        auto request = std::make_shared<int>(2); std::weak_ptr<int> weakRequest = request;
        auto jobInfo = base; jobInfo.requestLifetime = request;
        auto job = VulkanSourceWidthJob::Create(std::move(jobInfo), &reason); CHECK(job);
        request.reset();
        job->SuppressPublication();
        int callbacks = 0; bool callbackOk = true; VulkanSourceWidthJob::State finalState{}; VkResult finalStatus = VK_ERROR_UNKNOWN;
        CHECK(job->Start(owner, {}, [&](auto result, auto state, VkResult resultStatus) {
            ++callbacks; finalState = state; finalStatus = resultStatus; callbackOk = !result;
        }));
        CHECK(!job->Start(owner, {}, {})); owner.Drain();
        CHECK(callbackOk && callbacks == 1 && finalState == VulkanSourceWidthJob::State::Superseded &&
            finalStatus == VK_SUCCESS && weakRequest.expired());
        CHECK(owner.OutstandingCommands() == 0);
        auto next = VulkanSourceWidthJob::Create(base, &reason); CHECK(next);
        next->SuppressPublication();
        CHECK(next->Start(owner, {}, [&](auto result, auto state, VkResult resultStatus) {
            ++callbacks; callbackOk = !result && state == VulkanSourceWidthJob::State::Superseded && resultStatus == VK_SUCCESS;
        }));
        owner.Drain();
        CHECK(callbackOk && callbacks == 2 && owner.OutstandingCommands() == 0);
        CHECK(!job->NotifyCompletion() && !next->NotifyCompletion());
        next.reset();
        job.reset(); owner.Drain();
    }

    // Explicitly drive upload proof, width proof, and terminal publication.
    {
        UsdGenExecutionPipeline owner(runtime);
        auto request = std::make_shared<int>(3); std::weak_ptr<int> weakRequest = request;
        auto jobInfo = base; jobInfo.requestLifetime = request;
        auto job = VulkanSourceWidthJob::Create(std::move(jobInfo), &reason); CHECK(job);
        request.reset();
        int callbacks = 0; bool callbackOk = true; std::shared_ptr<const VulkanSourceGeneration> result;
        VulkanSourceWidthJob::State finalState{}; VkResult finalStatus = VK_ERROR_UNKNOWN;
        CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
            ++callbacks; result = std::move(generation); finalState = state; finalStatus = resultStatus;
            callbackOk = !weakRequest.expired() && result && result->context() == context && result->valueVersion() == 5;
        }));
        CHECK(job->Start(owner, {}, {}) == false);
        owner.Drain();                         // Created -> UploadPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion()); owner.Drain(); // UploadPending -> WidthPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion()); owner.Drain(); // WidthPending -> terminal callback
        CHECK(callbackOk && callbacks == 1 && finalState == VulkanSourceWidthJob::State::Ready && finalStatus == VK_SUCCESS);
        auto width = result->PlaneOwner("width"); CHECK(width && width->sizeBytes() == 8);
        CHECK(job->state() == VulkanSourceWidthJob::State::Ready);
        CHECK(owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
        result.reset(); job.reset(); owner.Drain(); CHECK(weakRequest.expired());
    }

    // Suppression after upload submission still proves the upload before settling.
    {
        UsdGenExecutionPipeline owner(runtime);
        auto job = VulkanSourceWidthJob::Create(base, &reason); CHECK(job);
        int callbacks = 0; bool callbackOk = true; VulkanSourceWidthJob::State finalState{};
        CHECK(job->Start(owner, {}, [&](auto, auto state, VkResult statusResult) {
            ++callbacks; finalState = state; callbackOk = statusResult == VK_SUCCESS;
        }));
        owner.Drain(); job->SuppressPublication(); CHECK(Prove(native));
        CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(callbackOk && callbacks == 1 && finalState == VulkanSourceWidthJob::State::Superseded);
        job.reset(); owner.Drain();
    }

    // A missing width input reaches the terminal failure callback on the owner.
    {
        UsdGenExecutionPipeline owner(runtime);
        auto missing = base; missing.source = Source(context, false);
        auto jobInfo = missing; jobInfo.requestLifetime = std::make_shared<int>(4);
        auto job = VulkanSourceWidthJob::Create(jobInfo, &reason); CHECK(job);
        int callbacks = 0; bool callbackOk = true; VulkanSourceWidthJob::State finalState{};
        CHECK(job->Start(owner, {}, [&](auto result, auto state, VkResult statusResult) {
            ++callbacks; finalState = state; callbackOk = !result && statusResult != VK_SUCCESS;
        }));
        owner.Drain(); CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
        CHECK(callbackOk && callbacks == 1 && finalState == VulkanSourceWidthJob::State::Failed);
        job.reset(); owner.Drain();
    }

    if (testTiles) {
        auto tilesPipeline = PicktileVkTilesPipeline::Create(context, tilesCode, &status);
        CHECK(tilesPipeline && status == VK_SUCCESS);
        auto boundsPipeline = PicktileVkBoundsPipeline::Create(context, boundsCode, &status);
        CHECK(boundsPipeline && status == VK_SUCCESS);

        // The lookup order needs survivor lookup planes the job does not
        // plumb yet: engaged jobs reject it at Create, fail-closed.
        auto badOrder = base;
        badOrder.requestLifetime = std::make_shared<int>(11);
        badOrder.tilesPipeline = tilesPipeline;
        badOrder.boundsPipeline = boundsPipeline;
        badOrder.tileOrder = PicktileVkTileOrder::CaptureOrderSurvivorSubset;
        CHECK(!VulkanSourceWidthJob::Create(badOrder, &reason));
        badOrder.tileOrder = PicktileVkTileOrder::IdentityCaptureOrder;
        CHECK(VulkanSourceWidthJob::Create(badOrder, &reason));

        // Foreign-context tile pipelines are rejected at Create; a
        // half-wired job is admitted and runs with tiles disabled.
        DeviceContext::CreateInfo foreignInfo; foreignInfo.timelineSemaphoreEnabled = true;
        foreignInfo.instance = native->instance; foreignInfo.physicalDevice = native->physical;
        foreignInfo.device = native->device; foreignInfo.computeQueue = native->queue;
        foreignInfo.computeQueueFamily = native->family; foreignInfo.physicalIndex = native->physicalIndex;
        foreignInfo.resourceDeviceId = 7015; foreignInfo.nativeLifetime = native; foreignInfo.timelineSemaphoreEnabled = true;
        foreignInfo.resources = {size_t{8} << 20, 0};
        auto foreignContext = DeviceContext::Create(foreignInfo, &status);
        CHECK(foreignContext && status == VK_SUCCESS);
        auto foreignTiles = PicktileVkTilesPipeline::Create(foreignContext, tilesCode, &status);
        CHECK(foreignTiles && status == VK_SUCCESS);
        auto badContext = base;
        badContext.requestLifetime = std::make_shared<int>(15);
        badContext.tilesPipeline = foreignTiles;
        badContext.boundsPipeline = boundsPipeline;
        CHECK(!VulkanSourceWidthJob::Create(badContext, &reason));
        auto halfWired = base;
        halfWired.requestLifetime = std::make_shared<int>(16);
        halfWired.tilesPipeline = tilesPipeline;
        CHECK(VulkanSourceWidthJob::Create(halfWired, &reason));

        // Tile finalization runs after the last stage: spans, then bounds,
        // then a metadata-only attach that Terminals Ready directly.
        std::shared_ptr<const VulkanSourceGeneration> tiled;
        {
            UsdGenExecutionPipeline owner(runtime);
            auto request = std::make_shared<int>(12); std::weak_ptr<int> weakRequest = request;
            auto info = base;
            info.requestLifetime = request;
            info.tilesPipeline = tilesPipeline;
            info.boundsPipeline = boundsPipeline;
            auto job = VulkanSourceWidthJob::Create(std::move(info), &reason); CHECK(job);
            request.reset();
            int callbacks = 0; bool callbackOk = false;
            std::shared_ptr<const VulkanSourceGeneration> result;
            VulkanSourceWidthJob::State finalState{}; VkResult finalStatus = VK_ERROR_UNKNOWN;
            CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
                ++callbacks; result = std::move(generation); finalState = state; finalStatus = resultStatus;
                callbackOk = !weakRequest.expired() && result && result->context() == context &&
                    result->valueVersion() == 5;
            }));
            owner.Drain();                         // Created -> UploadPending
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // Upload -> WidthPending
            CHECK(job->state() == VulkanSourceWidthJob::State::WidthPending);
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // Width -> TileFinalizePending
            CHECK(job->state() == VulkanSourceWidthJob::State::TileFinalizePending);
            CHECK(callbacks == 0);
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // spans proved -> bounds submitted
            CHECK(job->state() == VulkanSourceWidthJob::State::TileFinalizePending);
            CHECK(callbacks == 0);
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // bounds proved -> Ready
            CHECK(callbackOk && callbacks == 1 &&
                finalState == VulkanSourceWidthJob::State::Ready && finalStatus == VK_SUCCESS);
            CHECK(job->state() == VulkanSourceWidthJob::State::Ready);
            auto const& tiles = result->geometry().tiles;
            CHECK(tiles.size() == 1 && tiles[0].tile == 0 && tiles[0].boundsValid);
            CHECK(tiles[0].firstElement == 0 && tiles[0].elementCount == 1 &&
                tiles[0].firstPoint == 0 && tiles[0].pointCount == 2);
            for (int axis = 0; axis != 3; ++axis)
                CHECK(std::isfinite(tiles[0].extentMin[axis]) && std::isfinite(tiles[0].extentMax[axis]) &&
                    tiles[0].extentMin[axis] <= tiles[0].extentMax[axis]);
            CHECK(result->valueVersion() == 5 && result->topologyVersion() == 3);
            auto width = result->PlaneOwner("width"); CHECK(width && width->sizeBytes() == 8);

            // WithTiles validation, host-side on the tiled result.
            std::vector<UsdGenDeviceTileMetadata> gap{{0, 0, 1, 1, 2, {0, 0, 0}, {1, 1, 1}, true}};
            CHECK(!VulkanSourceGeneration::WithTiles(result, gap, &reason));
            auto unfinalized = tiles; unfinalized[0].boundsValid = false;
            CHECK(!VulkanSourceGeneration::WithTiles(result, unfinalized, &reason));
            std::vector<UsdGenDeviceTileMetadata> emptyTiles;
            CHECK(!VulkanSourceGeneration::WithTiles(result, emptyTiles, &reason));
            auto flipped = tiles; flipped[0].extentMin[0] = 9.f; flipped[0].extentMax[0] = -9.f;
            CHECK(!VulkanSourceGeneration::WithTiles(result, flipped, &reason));
            auto reattached = VulkanSourceGeneration::WithTiles(result, tiles, &reason);
            CHECK(reattached && reattached->valueVersion() == 5 && reattached->topologyVersion() == 3 &&
                reattached->NonWidthIdentity() == result->NonWidthIdentity() &&
                reattached->PlaneOwner("points") == result->PlaneOwner("points"));
            CHECK(owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
            tiled = std::move(result); job.reset(); owner.Drain();
            CHECK(context->resources()->Snapshot().usedBytes > 0);
        }
        // The tiled child aliases its ancestors: dropping it retires every
        // charged plane, including the tile candidates already released.
        tiled.reset();
        CHECK(context->resources()->Snapshot().usedBytes == 0);

        // Empty grooms keep the Terminal path: Ready with no tiles.
        {
            UsdGenExecutionPipeline owner(runtime);
            auto info = base;
            VulkanSourceGenerationCreateInfo emptySource;
            emptySource.context = context;
            emptySource.source.topologyVersion = 3;
            emptySource.source.valueVersion = 4;
            info.source = std::move(emptySource);
            info.requestLifetime = std::make_shared<int>(13);
            info.tilesPipeline = tilesPipeline;
            info.boundsPipeline = boundsPipeline;
            auto job = VulkanSourceWidthJob::Create(std::move(info), &reason); CHECK(job);
            int callbacks = 0; bool callbackOk = false;
            std::shared_ptr<const VulkanSourceGeneration> result;
            VulkanSourceWidthJob::State finalState{}; VkResult finalStatus = VK_ERROR_UNKNOWN;
            CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
                ++callbacks; result = std::move(generation); finalState = state; finalStatus = resultStatus;
                callbackOk = result && result->curveCount() == 0 && result->pointCount() == 0;
            }));
            owner.Drain();                         // Created -> UploadPending
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // Upload -> Width -> Ready (no tiles)
            CHECK(callbackOk && callbacks == 1 &&
                finalState == VulkanSourceWidthJob::State::Ready && finalStatus == VK_SUCCESS);
            CHECK(result->geometry().tiles.empty());
            CHECK(job->state() == VulkanSourceWidthJob::State::Ready);
            CHECK(owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
            result.reset(); job.reset(); owner.Drain();
            CHECK(context->resources()->Snapshot().usedBytes == 0);
        }

        // Status-gated failure: unsorted survivors under the Sorted order
        // fail device validation, so the job fails and publishes nothing.
        {
            UsdGenExecutionPipeline owner(runtime);
            auto info = base;
            VulkanSourceGenerationCreateInfo unsorted;
            unsorted.context = context;
            auto& b = unsorted.source;
            b.totalCurves = 2; b.totalCvs = 4; b.topologyVersion = 3; b.valueVersion = 4;
            b.px = {0, 1, 2, 3}; b.py = {0, 1, 2, 3}; b.pz = {0, 1, 2, 3};
            b.curveId = {9, 3}; b.cvOffsets = {0, 2, 4}; b.width = {1.f, 1.f, 1.f, 1.f};
            info.source = std::move(unsorted);
            info.requestLifetime = std::make_shared<int>(14);
            info.tilesPipeline = tilesPipeline;
            info.boundsPipeline = boundsPipeline;
            info.tileOrder = PicktileVkTileOrder::SortedSurvivorSubset;
            auto job = VulkanSourceWidthJob::Create(std::move(info), &reason); CHECK(job);
            int callbacks = 0; bool callbackOk = false;
            VulkanSourceWidthJob::State finalState{}; VkResult finalStatus = VK_SUCCESS;
            CHECK(job->Start(owner, {}, [&](auto result, auto state, VkResult resultStatus) {
                ++callbacks; finalState = state; finalStatus = resultStatus;
                callbackOk = !result;
            }));
            owner.Drain();                         // Created -> UploadPending
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // Upload -> WidthPending
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // Width -> TileFinalizePending
            CHECK(job->state() == VulkanSourceWidthJob::State::TileFinalizePending);
            CHECK(callbacks == 0);
            CHECK(Prove(native));
            CHECK(job->NotifyCompletion()); owner.Drain(); // spans rejected -> Failed
            CHECK(callbackOk && callbacks == 1 &&
                finalState == VulkanSourceWidthJob::State::Failed && finalStatus != VK_SUCCESS);
            CHECK(job->state() == VulkanSourceWidthJob::State::Failed);
            CHECK(owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
            job.reset(); owner.Drain();
            CHECK(context->resources()->Snapshot().usedBytes == 0);
        }
    }

    if (testDisabled) {
        // A muted Width aliases its source planes: the job never enters
        // WidthPending, and the published widths are the source widths
        // bit-exactly (a factor-9 replace would read {9,9} instead).
        UsdGenExecutionPipeline owner(runtime);
        auto request = std::make_shared<int>(10);
        auto info = base;
        info.requestLifetime = request;
        info.valueVersion = 5;
        VulkanSourceWidthStage muted;
        muted.kind = VulkanSourceWidthStage::Kind::Width;
        muted.input = 0;
        muted.width.width = 9.f;
        muted.width.replace = true;
        muted.disabled = true;
        info.stages = {muted};
        info.stageValueVersions = {5};
        auto job = VulkanSourceWidthJob::Create(std::move(info), &reason); CHECK(job);
        request.reset();
        int callbacks = 0; bool callbackOk = false;
        std::shared_ptr<const VulkanSourceGeneration> result;
        CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
            ++callbacks; result = std::move(generation);
            // The alias retains the source planes but publishes the muted
            // stage's authoritative value revision.
            callbackOk = result && state == VulkanSourceWidthJob::State::Ready &&
                resultStatus == VK_SUCCESS && result->valueVersion() == 5 &&
                result->geometry().valueVersion == 5;
        }));
        owner.Drain();                         // Created -> UploadPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion()); owner.Drain(); // Upload -> muted skip -> Ready
        CHECK(job->state() == VulkanSourceWidthJob::State::Ready);
        CHECK(callbackOk && callbacks == 1);
        {
            // Readback borrows the plane; release every shared owner before
            // the ledger check below.
            auto widths = result->PlaneOwner("width");
            CHECK(widths && widths->sizeBytes() == 8);
            std::vector<uint8_t> bytes;
            CHECK(ReadVulkanBytes(native, context, widths->buffer(), 8, result, &bytes) &&
                  bytes.size() == 8);
            float values[2]; std::memcpy(values, bytes.data(), 8);
            CHECK(values[0] == 1.f && values[1] == 2.f);
        }
        CHECK(owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
        result.reset(); job.reset(); owner.Drain();
        CHECK(context->resources()->Snapshot().usedBytes == 0);
    }

    if (testLength) {
        // Length is an optional, strictly intermediate native stage: its
        // revision must exist and lie strictly between source and width.
        auto lengthInfo = base;
        lengthInfo.lengthPipeline = lengthPipeline;
        CHECK(!VulkanSourceWidthJob::Create(lengthInfo, &reason));
        lengthInfo.lengthValueVersion = 4;
        CHECK(!VulkanSourceWidthJob::Create(lengthInfo, &reason));
        lengthInfo.lengthValueVersion = 5;
        CHECK(!VulkanSourceWidthJob::Create(lengthInfo, &reason));

        // Context equality is exact, rather than raw-device equality. This
        // must reject at Create time, before upload allocates or takes a
        // command credit from either resource ledger.
        auto foreignInfo = ci;
        foreignInfo.resourceDeviceId = 7015;
        auto foreignContext = DeviceContext::Create(foreignInfo, &status);
        CHECK(foreignContext && status == VK_SUCCESS);
        auto foreignLength = LengthScalePipeline::Create(foreignContext, lengthCode, &status);
        CHECK(foreignLength && status == VK_SUCCESS);
        auto const localLedger = context->resources()->Snapshot();
        auto const foreignLedger = foreignContext->resources()->Snapshot();
        auto foreignJob = base;
        foreignJob.lengthPipeline = foreignLength;
        foreignJob.lengthValueVersion = 5;
        foreignJob.valueVersion = 6;
        CHECK(!VulkanSourceWidthJob::Create(foreignJob, &reason));
        auto const localAfter = context->resources()->Snapshot();
        auto const foreignAfter = foreignContext->resources()->Snapshot();
        CHECK(localAfter.usedBytes == localLedger.usedBytes &&
              localAfter.byKind == localLedger.byKind);
        CHECK(foreignAfter.usedBytes == foreignLedger.usedBytes &&
              foreignAfter.byKind == foreignLedger.byKind);
        foreignLength.reset();
        foreignContext.reset();

        // This fixture is intentionally explicit/manual: each proof below
        // advances one submitted phase without spinning a notification loop.
        UsdGenExecutionPipeline owner(runtime);
        auto request = std::make_shared<int>(6);
        auto valid = base;
        valid.requestLifetime = request;
        valid.lengthPipeline = lengthPipeline;
        valid.lengthFactor = 0.5f;
        valid.lengthValueVersion = 5;
        valid.valueVersion = 6;
        auto job = VulkanSourceWidthJob::Create(std::move(valid), &reason);
        CHECK(job);
        request.reset();
        int callbacks = 0;
        bool callbackOk = false;
        std::shared_ptr<const VulkanSourceGeneration> result;
        CHECK(job->Start(owner, {}, [&](auto generation, auto state, VkResult resultStatus) {
            ++callbacks;
            result = std::move(generation);
            callbackOk = result && state == VulkanSourceWidthJob::State::Ready &&
                         resultStatus == VK_SUCCESS && result->context() == context &&
                         result->valueVersion() == 6 && result->PlaneOwner("points");
        }));
        owner.Drain();                         // Created -> UploadPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion());
        owner.Drain();                         // UploadPending -> LengthPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion());
        owner.Drain();                         // LengthPending -> WidthPending
        CHECK(Prove(native));
        CHECK(job->NotifyCompletion());
        owner.Drain();                         // WidthPending -> terminal callback
        CHECK(callbackOk && callbacks == 1 &&
              owner.OutstandingCommands() == 0 && !job->NotifyCompletion());
        result.reset();
        job.reset();
        owner.Drain();
        CHECK(context->resources()->Snapshot().usedBytes == 0);
    }
    if (testCull) {
        auto cullInfo = [&] {
            auto info = base;
            info.lengthCompactionPipeline = cullPipeline;
            VulkanSourceWidthStage cull;
            cull.kind = VulkanSourceWidthStage::Kind::LengthCull;
            cull.input = 0;
            cull.cullThreshold = 0.f;
            info.stages = {cull};
            info.stageValueVersions = {5};
            return info;
        };

        // Counts has completed, but its owner notification has not yet
        // admitted scatter. Suppression at this boundary must finish without
        // allocating or submitting phase two.
        {
            UsdGenExecutionPipeline owner(runtime);
            auto jobInfo = cullInfo();
            auto job = VulkanSourceWidthJob::Create(std::move(jobInfo), &reason); CHECK(job);
            int callbacks = 0; bool callbackOk = false;
            CHECK(job->Start(owner, {}, [&](auto result, auto state, VkResult resultStatus) {
                ++callbacks;
                callbackOk = !result && state == VulkanSourceWidthJob::State::Superseded && resultStatus == VK_SUCCESS;
            }));
            owner.Drain();
            CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
            CHECK(job->state() == VulkanSourceWidthJob::State::CullCountsPending);
            CHECK(Prove(native));
            job->SuppressPublication();
            CHECK(job->NotifyCompletion()); owner.Drain();
            CHECK(callbackOk && callbacks == 1 &&
                  job->state() == VulkanSourceWidthJob::State::Superseded && owner.OutstandingCommands() == 0);
            job.reset(); owner.Drain();
            CHECK(context->resources()->Snapshot().usedBytes == 0);
        }

        // Scatter was admitted before suppression, so it remains retained and
        // must prove before the same superseded terminal callback is emitted.
        {
            UsdGenExecutionPipeline owner(runtime);
            auto jobInfo = cullInfo();
            auto job = VulkanSourceWidthJob::Create(std::move(jobInfo), &reason); CHECK(job);
            int callbacks = 0; bool callbackOk = false;
            CHECK(job->Start(owner, {}, [&](auto result, auto state, VkResult resultStatus) {
                ++callbacks;
                callbackOk = !result && state == VulkanSourceWidthJob::State::Superseded && resultStatus == VK_SUCCESS;
            }));
            owner.Drain();
            CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
            CHECK(job->state() == VulkanSourceWidthJob::State::CullCountsPending);
            CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
            CHECK(job->state() == VulkanSourceWidthJob::State::CullScatterPending && callbacks == 0);
            job->SuppressPublication();
            CHECK(callbacks == 0 && job->state() == VulkanSourceWidthJob::State::CullScatterPending &&
                  context->resources()->Snapshot().usedBytes > 0);
            CHECK(Prove(native)); CHECK(job->NotifyCompletion()); owner.Drain();
            CHECK(callbackOk && callbacks == 1 &&
                  job->state() == VulkanSourceWidthJob::State::Superseded && owner.OutstandingCommands() == 0);
            CHECK(!job->NotifyCompletion() && callbacks == 1);
            job.reset(); owner.Drain();
            CHECK(context->resources()->Snapshot().usedBytes == 0);
        }
    }
    cullPipeline.reset(); lengthPipeline.reset(); pipeline.reset(); context.reset(); native.reset();
    return 0;
}
