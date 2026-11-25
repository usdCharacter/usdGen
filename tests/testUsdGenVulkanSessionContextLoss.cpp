// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Native fixture for the Session Vulkan provider contract.
//
// This deliberately keeps the reply/return owner separate from the queue
// owner.  The provider is expected to reserve its reply transport before it
// enters the native owner; no callback in this fixture is permitted to touch
// mutable Session state directly from the Vulkan waiter.

#include "usdGen/executionPipeline.h"
#include "nodeSets.h"
#include "usdGen/executionResources.h"
#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/completionService.h"
#include "usdGen/vulkan/deviceGenerationAdapter.h"
#include "usdGen/vulkan/planExecutor.h"
#include "usdGen/vulkan/widthPipeline.h"
#include "usdGen/executionTaskGraph.h"
#include "usdGen/vulkan/sessionProvider.h"
#include "usdGen/sessionDeviceIntegration.h"
#include "usdGen/executionCache.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "vulkanNativeFixture.h"
#include "usdGen/vulkan/chargedBuffer.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <future>
#include <cstring>
#include <cstdlib>

#define CHECK(condition) do { if (!(condition)) { std::fprintf(stderr, "CHECK failed: %s at %d\n", #condition, __LINE__); std::abort(); } } while (false)

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

struct SixPlaneOracle {
    std::vector<GfVec3f> points, rest;
    std::vector<uint32_t> curveOffsets;
    std::vector<uint64_t> stableIds;
    std::vector<float> hairT, width;
};

template <class T> static std::vector<uint8_t> Bytes(std::vector<T> const& v) {
    std::vector<uint8_t> out(v.size() * sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}

struct BatchRead final {
    std::shared_ptr<NativeOwner> native;
    std::vector<std::shared_ptr<ChargedBuffer>> staging;
    std::vector<std::vector<uint8_t>> expected;
    std::unique_ptr<VulkanCompletionService::Watch> watch;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool proved = false;
    ~BatchRead() {
        CHECK(proved);
        if (command) vkFreeCommandBuffers(native->device, native->commands, 1, &command);
        if (fence) vkDestroyFence(native->device, fence, nullptr);
    }
};

void CheckPlanes(std::shared_ptr<NativeOwner> const& native,
                 std::shared_ptr<DeviceContext> const& context,
                 std::shared_ptr<VulkanCompletionService> const& service,
                 UsdGenExecutionPipeline& owner,
                 std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                 SixPlaneOracle const& expected,
                 VulkanGenerationLease priorLease = {}) {
    VulkanGenerationLease lease = std::move(priorLease);
    auto const stream = reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
    auto reads = std::make_shared<BatchRead>(); reads->native = native;
    auto completed = std::make_shared<std::promise<void>>(); auto done = completed->get_future();
    owner.InvokeOwner([&] {
        if (!lease) lease = AcquireVulkanGeneration(generation, stream);
        CHECK(lease && lease.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        auto const& planes = lease.Planes(); CHECK(planes.size() == 6);
        std::vector<std::pair<char const*, std::vector<uint8_t>>> want{
            {"points", Bytes(expected.points)}, {"rest", Bytes(expected.rest)},
            {"curveOffsets", Bytes(expected.curveOffsets)}, {"curveId", Bytes(expected.stableIds)},
            {"hairT", Bytes(expected.hairT)}, {"widths", Bytes(expected.width)}};
        reads->watch = service->Reserve(reads, [reads, completed, &owner](VkResult proof) {
            CHECK(owner.IsExecutingOwner() && proof == VK_SUCCESS);
            for (size_t i = 0; i != reads->staging.size(); ++i) {
                CHECK(reads->staging[i]->PollComplete() == VK_SUCCESS);
                void* mapped = nullptr;
                CHECK(vkMapMemory(reads->native->device, reads->staging[i]->memory(), 0,
                    reads->expected[i].size(), 0, &mapped) == VK_SUCCESS);
                CHECK(std::memcmp(mapped, reads->expected[i].data(), reads->expected[i].size()) == 0);
                vkUnmapMemory(reads->native->device, reads->staging[i]->memory());
            }
            CHECK(reads->watch->Retire()); reads->watch.reset(); reads->proved = true;
            completed->set_value(); return VulkanCompletionService::DeliveryResult::Posted;
        }, [](VkResult) { CHECK(false); });
        CHECK(reads->watch);
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK(vkCreateFence(native->device, &fi, nullptr, &reads->fence) == VK_SUCCESS);
        VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(native->device, &ai, &reads->command) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CHECK(vkBeginCommandBuffer(reads->command, &begin) == VK_SUCCESS);
        VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(reads->command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &before, 0, nullptr, 0, nullptr);
        for (auto const& item : want) {
            auto plane = std::find_if(planes.begin(), planes.end(), [&](auto const& p) { return p.metadata.name == item.first; });
            CHECK(plane != planes.end() && plane->bytes == item.second.size());
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size = plane->bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto staging = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch); CHECK(staging);
            CHECK(staging->MarkSubmitted(reads->fence, reads) == VK_SUCCESS);
            VkBufferCopy copy{0, 0, plane->bytes}; vkCmdCopyBuffer(reads->command, plane->buffer, staging->buffer(), 1, &copy);
            reads->staging.push_back(std::move(staging)); reads->expected.push_back(item.second);
        }
        VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(reads->command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &after, 0, nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(reads->command) == VK_SUCCESS);
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &reads->command;
        CHECK(reads->watch->MarkPhaseSubmitted());
        CHECK(vkQueueSubmit(native->queue, 1, &submit, reads->fence) == VK_SUCCESS);
    });
    // The observer owns the copy proof; source leases may retire before its
    // completion signal is armed on the same serialized owner.
    lease.Complete();
    owner.InvokeOwner([&] { CHECK(service->Arm(*reads->watch, 1)); });
    CHECK(done.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    owner.Drain(); // Retire callback owns the final watcher/local references.
    CHECK(reads->proved);
}

// CPU-reference oracle used by the Session provider parity checks.  Keeping
// this here makes the Vulkan proof independent of device-only fixture tests.
static bool BuildCpuReference(UsdGenGraphDesc desc, SixPlaneOracle* out) {
    if (!out) return false;
    desc.executionBackend = UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = compiler.Compile(desc, &graph);
    if (!compiled.ok) {
        for (auto const& error : compiled.errors) std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Desc();
    auto run = scheduler.Run(graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const& error : run.diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    UsdGenCurveBuffer const buffer = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    out->points.resize(buffer.totalCvs);
    out->rest.assign(buffer.rest.begin(), buffer.rest.end());
    for (size_t i = 0; i != buffer.totalCvs; ++i)
        out->points[i] = {buffer.px[i], buffer.py[i], buffer.pz[i]};
    out->curveOffsets.assign(buffer.cvOffsets.begin(), buffer.cvOffsets.end());
    out->stableIds.assign(buffer.curveId.begin(), buffer.curveId.end());
    out->hairT.assign(buffer.hairT.begin(), buffer.hairT.end());
    out->width.assign(buffer.width.begin(), buffer.width.end());
    return true;
}

static UsdGenGraphDesc MakeProviderDescriptor() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/SessionProvider");
    desc.executionBackend = UsdGenExecutionBackend::Vulkan;
    desc.defaultWidth = .5f;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/SessionProvider/curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {3, 2};
    curves.curveId = {20, 10};
    curves.points = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0}};
    curves.rest = {{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1}};
    curves.widths = {1, 2, 3, 4, 5};
    curves.widthsInterpolation = TfToken("vertex");
    curves.curveGeneration = 41;
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/SessionProvider/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/SessionProvider/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params = {{TfToken("width"), VtValue(1.25f), false},
                    {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, width};
    desc.terminal = width.path;
    return desc;
}

struct NativeLifetime {
    std::shared_ptr<NativeOwner> native;
};

struct ReturnRelay {
    UsdGenExecutionPipeline::CommandTicket ticket;
    std::shared_ptr<std::promise<std::shared_ptr<const UsdGenSessionDeviceResult>>> promise;
};

// Shared by the new provider tests and intentionally not a device-only test:
// the two owner identities are observable and are part of the contract.
struct VulkanSessionProviderFixture final {
    bool unavailable = false;
    std::shared_ptr<NativeOwner> native;
    std::shared_ptr<DeviceContext> context;
    std::shared_ptr<UsdGenExecutionRuntime> runtime;
    std::shared_ptr<UsdGenExecutionPipeline> returnOwner;
    std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
    std::shared_ptr<VulkanCompletionService> completion;
    std::shared_ptr<VulkanGenerationAdapterDomain> domain;

    explicit VulkanSessionProviderFixture(uint64_t commandCapacity = 8) {
        // Timeline support is an enabled-device fact, not physical-device
        // capability probing. CreateNative owns the complete native domain.
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
        timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        timeline.timelineSemaphore = VK_TRUE;
        native = CreateNative(&unavailable, {}, &timeline);
        if (!native) return;

        auto lifetime = std::make_shared<NativeLifetime>();
        lifetime->native = native;
        DeviceContext::CreateInfo info;
        info.instance = native->instance;
        info.physicalDevice = native->physical;
        info.device = native->device;
        info.computeQueue = native->queue;
        info.computeQueueFamily = native->family;
        info.physicalIndex = native->physicalIndex;
        info.resourceDeviceId = native->physicalIndex;
        info.nativeLifetime = lifetime;
        info.gpuLabel = "session-provider-fixture";
        info.resources = {64u * 1024u * 1024u, 4u * 1024u * 1024u};
        info.timelineSemaphoreEnabled = true;
        VkResult result = VK_SUCCESS;
        context = DeviceContext::Create(info, &result);
        if (!context || result != VK_SUCCESS) { native.reset(); return; }

        runtime = std::make_shared<UsdGenExecutionRuntime>(4);
        // Keep capacities deliberately small: admission/cancel tests can
        // deterministically exercise rejected and retained requests.
        returnOwner = std::make_shared<UsdGenExecutionPipeline>(*runtime,
                                                                 commandCapacity,
                                                                 commandCapacity);
        queueOwner = std::make_shared<UsdGenExecutionPipeline>(*runtime, 16, 16);
        completion = VulkanCompletionService::Create({context, queueOwner.get(), 8});
        if (!completion) return;
        domain = VulkanGenerationAdapterDomain::Create({queueOwner, completion});
    }

    ~VulkanSessionProviderFixture() {
        // External service/domain close is explicit and ordered.  No owner
        // callback is allowed to be running when the native domain dies.
        if (domain) { queueOwner->Drain(); domain->Close(); }
        if (completion) completion->CloseAndJoin();
        if (returnOwner) returnOwner->Drain();
        queueOwner.reset(); returnOwner.reset(); completion.reset(); domain.reset();
        context.reset(); runtime.reset(); native.reset();
    }

    explicit operator bool() const noexcept {
        return bool(context && returnOwner && queueOwner && completion && domain);
    }
};

} // namespace

int main(int argc, char** argv) {
    // CMake supplies the compiled width shader as the sole argument.
    if (argc < 2) { std::puts("Vulkan Session provider shader unavailable"); return 77; }
    std::ifstream shader(argv[1], std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return 77;
    VulkanSessionProviderFixture fixture;
    if (fixture.unavailable) {
        std::puts("Vulkan Session provider native fixture unavailable");
        return 77;
    }
    if (!fixture) {
        std::fprintf(stderr, "Vulkan Session provider fixture construction failed\n");
        return 1;
    }
    std::vector<uint32_t> spirv(raw.size() / sizeof(uint32_t));
    std::memcpy(spirv.data(), raw.data(), raw.size());
    auto pipeline = WidthPipeline::Create(fixture.context, spirv);
    if (!pipeline) return 77;
    auto preparer = UsdGenExecutionTaskGraph::GetOrCreate(*fixture.runtime,
                                                           "vulkan-session-provider", 7044);
    std::string reason;
    auto executor = VulkanPlanExecutor::Create({fixture.domain, pipeline, preparer}, &reason);
    if (!executor) { std::fprintf(stderr, "%s\n", reason.c_str()); return 1; }
    UsdGenSessionDeviceIdentity identity;
    identity.backend = UsdGenDeviceBackend::Vulkan;
    identity.deviceIndex = fixture.context->physicalIndex();
    identity.capabilityVersion = 1;
    identity.logicalContextIdentity = reinterpret_cast<uintptr_t>(fixture.context.get());
    identity.contextEpoch = 0;
    std::copy(fixture.context->deviceUUID().begin(), fixture.context->deviceUUID().end(),
              identity.physicalDeviceUuid.begin());
    auto provider = VulkanSessionProvider::Create({executor, fixture.context,
                                                    fixture.queueOwner, identity, 256}, &reason);
    if (!provider) { std::fprintf(stderr, "%s\n", reason.c_str()); return 1; }
    auto cacheDomain = UsdGenExecutionCacheDomain::AcquireShared(
        {UsdGenDeviceBackend::Vulkan, identity.deviceIndex,
         identity.logicalContextIdentity}, size_t{64} << 20);
    auto firstSession = CreateUsdGenDeviceSession(2, 8, cacheDomain, provider);
    auto secondSession = CreateUsdGenDeviceSession(2, 8, cacheDomain, provider);
    CHECK(firstSession && secondSession);
    auto descriptor = MakeProviderDescriptor();
    SixPlaneOracle oracle;
    CHECK(BuildCpuReference(descriptor, &oracle));
    firstSession->SetGraphDesc(descriptor);
    firstSession->SetDevicePublicationEnabled(true);
    secondSession->SetGraphDesc(descriptor);
    secondSession->SetDevicePublicationEnabled(true);
    auto first = firstSession->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    auto second = secondSession->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(first && second && !first->diagnostics.HasErrors() && !second->diagnostics.HasErrors());
    CHECK(first->generation && second->generation && first->generation->device && second->generation->device);
    CHECK(second->stats.executionCacheHits == 1);
    VulkanGenerationLease heldFirst, heldSecond;
    fixture.queueOwner->InvokeOwner([&] {
        auto stream = reinterpret_cast<UsdGenDeviceStream>(fixture.context->computeQueue());
        heldFirst = AcquireVulkanGeneration(first->generation->device, stream);
        heldSecond = AcquireVulkanGeneration(second->generation->device, stream);
        CHECK(heldFirst && heldSecond);
        CHECK(heldFirst.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        CHECK(heldSecond.WaitUntilReady() == UsdGenDeviceStatus::Ok);
    });
    auto const beforeProvider = provider->Identity();
    auto const beforeCache = cacheDomain->Epoch();
    // Logical loss injection exercises invalidation without destroying the
    // test device while existing, already-proved consumer leases still exist.
    firstSession->NotifyDeviceContextLost(UsdGenDeviceBackend::Vulkan, identity.deviceIndex);
    firstSession->Drain();
    CHECK(provider->Identity().contextEpoch > beforeProvider.contextEpoch);
    CHECK(cacheDomain->Epoch() > beforeCache);
    CHECK(cacheDomain->Size() == 0);
    UsdGenDiagnostics rejected;
    CHECK(!provider->Compile(descriptor, &rejected));
    auto failedSecond = secondSession->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(failedSecond && failedSecond->diagnostics.HasErrors());
    CHECK(failedSecond->generation == second->generation);
    CHECK(secondSession->NeedsCommit());
    auto failedFirst = firstSession->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(failedFirst && failedFirst->diagnostics.HasErrors());
    CHECK(failedFirst->generation == first->generation);
    CHECK(firstSession->NeedsCommit());
    // Leases acquired before invalidation still expose the same immutable
    // public buffers; no physical Vulkan device loss is fabricated here.
    fixture.queueOwner->InvokeOwner([&] {
        CHECK(heldFirst.Planes().size() == 6 && heldSecond.Planes().size() == 6);
        for (auto const& plane : heldFirst.Planes()) CHECK(plane.buffer != VK_NULL_HANDLE);
        for (auto const& plane : heldSecond.Planes()) CHECK(plane.buffer != VK_NULL_HANDLE);
    });
    // Copy directly through the leases acquired BEFORE loss. Each helper
    // completes its borrowed lease after queue submission and then proves
    // both the native copies and consumer retirement through the observer.
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, first->generation->device, oracle, std::move(heldFirst));
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, second->generation->device, oracle, std::move(heldSecond));
    {
        auto const retainedBeforeReplacement = fixture.context->resources()->Snapshot().usedBytes;
        VulkanSessionProviderFixture replacement;
        CHECK(replacement && !replacement.unavailable);
        CHECK(replacement.native->device != fixture.native->device);
        CHECK(replacement.context->deviceUUID() == fixture.context->deviceUUID());
        CHECK(replacement.context->resources() == fixture.context->resources());
        auto replacementPipeline = WidthPipeline::Create(replacement.context, spirv);
        CHECK(replacementPipeline);
        auto replacementPreparer = UsdGenExecutionTaskGraph::GetOrCreate(*replacement.runtime,
            "vulkan-session-context-replacement", 7045);
        auto replacementExecutor = VulkanPlanExecutor::Create(
            {replacement.domain, replacementPipeline, replacementPreparer}, &reason);
        CHECK(replacementExecutor);
        auto replacementIdentity = identity;
        replacementIdentity.logicalContextIdentity = reinterpret_cast<uintptr_t>(replacement.context.get());
        CHECK(replacementIdentity.logicalContextIdentity != identity.logicalContextIdentity);
        auto replacementProvider = VulkanSessionProvider::Create({replacementExecutor,
            replacement.context, replacement.queueOwner, replacementIdentity}, &reason);
        CHECK(replacementProvider);
        auto replacementCache = UsdGenExecutionCacheDomain::AcquireShared(
            {replacementIdentity.backend, replacementIdentity.deviceIndex,
             replacementIdentity.logicalContextIdentity}, size_t{64} << 20);
        CHECK(replacementCache && replacementCache != cacheDomain);
        CHECK(!CreateUsdGenDeviceSession(2, 8, cacheDomain, replacementProvider));
        auto replacementSession = CreateUsdGenDeviceSession(2, 8, replacementCache, replacementProvider);
        CHECK(replacementSession);
        replacementSession->SetGraphDesc(descriptor);
        replacementSession->SetDevicePublicationEnabled(true);
        auto fresh = replacementSession->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(fresh && !fresh->diagnostics.HasErrors() && fresh->generation && fresh->generation->device);
        CHECK(fresh->stats.executionCacheHits == 0 && fresh->stats.executionCacheMisses == 1);
        CheckPlanes(replacement.native, replacement.context, replacement.completion,
                    *replacement.queueOwner, fresh->generation->device, oracle);
        // A new logical device cannot mutate the retained old publications.
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, first->generation->device, oracle);
        CHECK(cacheDomain->Size() == 0 && replacementCache->Size() == 1);
        replacementSession->Drain(); replacementSession.reset();
        CHECK(replacementCache->Invalidate());
        replacement.queueOwner->InvokeOwner([&] { fresh.reset(); });
        replacementProvider->Shutdown(); replacementExecutor->Shutdown(); replacementPreparer->Drain();
        CHECK(replacement.domain->Close()); replacement.completion->CloseAndJoin();
        CHECK(replacement.queueOwner->OutstandingCommands() == 0);
        CHECK(replacement.context->resources()->Snapshot().usedBytes == retainedBeforeReplacement);
    }
    firstSession->Drain(); secondSession->Drain();
    firstSession.reset(); secondSession.reset();
    CHECK(cacheDomain->Invalidate());
    fixture.queueOwner->InvokeOwner([&] {
        failedFirst.reset(); failedSecond.reset(); first.reset(); second.reset();
    });
    provider->Shutdown(); executor->Shutdown(); preparer->Drain();
    fixture.queueOwner->Drain();
    if (!fixture.domain->Close()) return 1;
    fixture.completion->CloseAndJoin();
    CHECK(fixture.queueOwner->OutstandingCommands() == 0);
    CHECK(fixture.queueOwner->CallbackFailures() == 0);
    CHECK(fixture.context->resources()->Snapshot().usedBytes == 0);
    if (fixture.returnOwner.get() == fixture.queueOwner.get() ||
        fixture.completion->queueOwner() != fixture.queueOwner.get() ||
        !fixture.context->timelineSemaphoreEnabled() ||
        fixture.context->computeQueue() != fixture.native->queue) {
        std::fprintf(stderr, "Vulkan Session provider owner/context contract failed\n");
        return 1;
    }
    std::puts("Vulkan Session context-loss fixture: PASS");
    return 0;
}
