// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Native fixture for the Session Vulkan provider contract.
//
// This deliberately keeps the reply/return owner separate from the queue
// owner.  The provider is expected to reserve its reply transport before it
// enters the native owner; no callback in this fixture is permitted to touch
// mutable Session state directly from the Vulkan waiter.

#include "utils/pxrInterop.h"
#include "usdGen/exec/executionPipeline.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGen/exec/device/executionResources.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenCurves/vulkan/planExecutor.h"
#include "usdGenCurves/vulkan/widthPipeline.h"
#include "usdGen/exec/executionTaskGraph.h"
#include "usdGenCurves/vulkan/sessionProvider.h"
#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"
#include "usdGen/exec/scheduler.h"
#include "vulkanNativeFixture.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <future>
#include <thread>
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
                 SixPlaneOracle const& expected) {
    VulkanGenerationLease lease;
    auto const stream = reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
    auto reads = std::make_shared<BatchRead>(); reads->native = native;
    auto completed = std::make_shared<std::promise<void>>(); auto done = completed->get_future();
    owner.InvokeOwner([&] {
        lease = AcquireVulkanGeneration(generation, stream);
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
    desc.executionBackend = kUsdGenBackendCpu;
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) {
        for (auto const& error : compiled.errors) std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    UsdGenScheduler scheduler(2);
    UsdGenEvalContext context;
    context.desc = &graph.Spec();
    auto run = scheduler.Run(graph, context, 1);
    if (run.diagnostics.HasErrors()) {
        for (auto const& error : run.diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
        return false;
    }
    UsdGenCurveBuffer const buffer = NodeBuffer(graph, usdGenUsd::NodeIdForPath(graph, desc.terminal));
    out->points.resize(buffer.totalCvs);
    auto const hostRest = usdGenUsd::ToHost(buffer.rest);
    out->rest.assign(hostRest.begin(), hostRest.end());
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
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");
    desc.defaultWidth = .5f;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/SessionProvider/curves");
    curves.role = UsdGenLane::Chunked;
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

static UsdGenGraphDesc MakeProviderLengthDescriptor() {
    auto desc = MakeProviderDescriptor();
    auto source = desc.nodes[0];
    auto length = UsdGenNodeDesc{};
    length.path = SdfPath("/SessionProvider/length");
    length.type = TfToken("UsdGenLength");
    length.inputs = {source.path};
    length.params = {{TfToken("length:value"), VtValue(0.25), false},
                     {TfToken("length:mode"), VtValue(TfToken("scale")), false},
                     {TfToken("length:method"), VtValue(TfToken("scale")), false}};
    desc.nodes[1].inputs = {length.path};
    desc.nodes.insert(desc.nodes.begin() + 1, length);
    desc.nodes[2].path = SdfPath("/SessionProvider/width");
    desc.nodes[2].type = TfToken("UsdGenWidth");
    desc.nodes[2].inputs = {length.path};
    desc.nodes[2].params = {{TfToken("width"), VtValue(1.25f), false},
                            {TfToken("replace"), VtValue(false), false}};
    desc.terminal = desc.nodes[2].path;
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
    identity.backend = usdGenTestUtils::Backend("vulkan");
    identity.deviceIndex = fixture.context->physicalIndex();
    identity.capabilityVersion = 1;
    identity.logicalContextIdentity = reinterpret_cast<uintptr_t>(fixture.context.get());
    identity.contextEpoch = 0;
    std::copy(fixture.context->deviceUUID().begin(), fixture.context->deviceUUID().end(),
              identity.physicalDeviceUuid.begin());
    auto provider = VulkanSessionProvider::Create({executor, fixture.context,
                                                    fixture.queueOwner, identity, 2}, &reason);
    if (!provider) { std::fprintf(stderr, "%s\n", reason.c_str()); return 1; }
    SixPlaneOracle oracle;
    auto descriptor = MakeProviderDescriptor();
    if (!BuildCpuReference(descriptor, &oracle) || oracle.points.empty() ||
        oracle.rest.size() != oracle.points.size() ||
        oracle.hairT.size() != oracle.points.size() ||
        oracle.width.size() != oracle.points.size() ||
        oracle.curveOffsets.size() != 3 || oracle.stableIds.size() != 2) {
        std::fprintf(stderr, "Vulkan Session provider CPU six-plane oracle failed\n");
        return 1;
    }
    UsdGenDiagnostics diagnostics;
    auto plan = provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics);
    if (!plan || diagnostics.HasErrors()) return 1;
    // Live records consume the bounded provenance capacity. Releasing one
    // must reclaim its slot without evicting the earlier live handle.
    auto dropped = provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics);
    CHECK(dropped);
    CHECK(!provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics));
    dropped.reset();
    auto reclaimed = provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics);
    CHECK(reclaimed);
    // Reuse the second slot repeatedly while retaining `plan`; provenance
    // reclamation must remain bounded and must never displace that live plan.
    for (int i = 0; i != 16; ++i) {
        reclaimed.reset();
        reclaimed = provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics);
        CHECK(reclaimed);
    }
    reclaimed.reset();
    // Concurrent compiler callers may contend for the one slot not held by
    // `plan`; bounded rejection is legitimate, but no foreign/native work is
    // issued from these threads and the retained plan must remain usable.
    std::promise<void> startPromise;
    auto start = startPromise.get_future().share();
    std::atomic<uint32_t> compiled{0}, rejected{0}, badDiagnostics{0};
    std::vector<std::thread> compilers;
    for (int worker = 0; worker != 4; ++worker) {
        compilers.emplace_back([&, start] {
            start.wait();
            for (int i = 0; i != 16; ++i) {
                UsdGenDiagnostics local;
                auto transient = provider->Compile(usdGenUsd::Lower(descriptor), &local);
                if (local.HasErrors()) { ++badDiagnostics; continue; }
                if (transient) ++compiled;
                else ++rejected;
            }
        });
    }
    startPromise.set_value();
    for (auto& worker : compilers) worker.join();
    CHECK(!badDiagnostics && compiled + rejected == 64 && compiled);
    auto postStress = provider->Compile(usdGenUsd::Lower(descriptor), &diagnostics);
    CHECK(postStress);
    postStress.reset();
    UsdGenExecutionPipeline::Cancellation cancellation;

    // Length plans are descriptor-compilable even when this fixture was
    // launched with only the width shader. Revision shape is nevertheless a
    // provider admission contract and must reject before queue/native ingress.
    auto lengthDescriptor = MakeProviderLengthDescriptor();
    auto lengthPlan = provider->Compile(usdGenUsd::Lower(lengthDescriptor), &diagnostics);
    CHECK(lengthPlan && !diagnostics.HasErrors());
    auto const lengthLedger = fixture.context->resources()->Snapshot();
    std::vector<UsdGenExecutionPipeline::CommandTicket> heldCommandCredits;
    for (uint64_t i = 0; i + 1 != fixture.queueOwner->CommandCapacity(); ++i) {
        auto ticket = fixture.queueOwner->ReserveCommandTicket();
        CHECK(ticket); heldCommandCredits.push_back(std::move(ticket));
    }
    std::atomic<uint32_t> lengthCallbacks{0};
    auto lengthRequest = [&](std::optional<UsdGenSessionDeviceRevisions> revisions) {
        UsdGenSessionDeviceRequest bad;
        bad.plan = lengthPlan; bad.identity = identity; bad.publicationGeneration = 100;
        bad.cancellation = cancellation;
        bad.tool = {"vulkan-session-provider", 7044, 100};
        bad.authoritativeRevisions = std::move(revisions);
        bad.returnTransport = [&](auto) { ++lengthCallbacks; return true; };
        CHECK(!provider->Submit(std::move(bad)));
        auto probe = fixture.queueOwner->ReserveCommandTicket();
        CHECK(probe);
        CHECK(fixture.context->resources()->Snapshot().usedBytes == lengthLedger.usedBytes);
    };
    lengthRequest(std::nullopt);
    lengthRequest(UsdGenSessionDeviceRevisions{41, 42, 43, {}});
    lengthRequest(UsdGenSessionDeviceRevisions{41, 42, 43, {44, 45}});
    lengthRequest(UsdGenSessionDeviceRevisions{41, 42, 44, {42}});
    lengthRequest(UsdGenSessionDeviceRevisions{41, 42, 43, {40}});
    CHECK(lengthCallbacks.load() == 0);
    // The width shape has no intermediate operator and rejects even a
    // correctly ordered-looking extraneous intermediate revision.
    {
        UsdGenSessionDeviceRequest bad;
        bad.plan = plan; bad.identity = identity; bad.cancellation = cancellation;
        bad.authoritativeRevisions = UsdGenSessionDeviceRevisions{41, 42, 44, {43}};
        bad.returnTransport = [&](auto) { ++lengthCallbacks; return true; };
        CHECK(!provider->Submit(std::move(bad)));
        auto probe = fixture.queueOwner->ReserveCommandTicket();
        CHECK(probe);
        CHECK(fixture.context->resources()->Snapshot().usedBytes == lengthLedger.usedBytes);
    }
    heldCommandCredits.clear();
    lengthPlan.reset();

    fixture.queueOwner->Await([&cancellation, owner = fixture.queueOwner](auto done) {
        auto signal = std::make_shared<std::function<void()>>(std::move(done));
        if (!owner->Submit([&cancellation, signal](auto const& token) {
            cancellation = token; return UsdGenExecutionPipeline::Publish([signal] { (*signal)(); });
        })) (*signal)();
    });
    auto returned = std::make_shared<std::promise<std::shared_ptr<const UsdGenSessionDeviceResult>>>();
    auto future = returned->get_future();
    auto returnTicket = fixture.returnOwner->ReserveCommandTicket();
    if (!returnTicket) return 1;
    auto relay = std::make_shared<ReturnRelay>();
    relay->ticket = std::move(returnTicket);
    relay->promise = returned;
    UsdGenSessionDeviceRequest request;
    request.plan = plan; request.identity = identity; request.publicationGeneration = 99;
    request.cancellation = cancellation;
    request.tool = {"vulkan-session-provider", 7044, 99};
    request.authoritativeRevisions = UsdGenSessionDeviceRevisions{41, 42, 43};
    request.returnTransport = [relay, owner = fixture.returnOwner](auto result) mutable {
        auto payload = std::move(result);
        auto ticket = std::move(relay->ticket);
        if (!ticket) return false;
        return owner->PostCommand(std::move(ticket), [relay, owner, payload = std::move(payload)] {
            CHECK(owner->IsExecutingOwner());
            relay->promise->set_value(payload);
        }, [relay] { relay->promise->set_value({}); });
    };
    fixture.returnOwner->InvokeOwner([&] {
        auto bad = request;
        ++bad.identity.logicalContextIdentity;
        CHECK(!provider->Submit(std::move(bad)));
        bad = request;
        bad.authoritativeRevisions->finalValueVersion = 42;
        CHECK(!provider->Submit(std::move(bad)));
        bad = request;
        // Matching backend metadata alone must not authorize a foreign
        // opaque payload or a plan not compiled by this exact provider.
        bad.plan = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(descriptor));
        CHECK(bad.plan && !provider->Submit(std::move(bad)));
        // `plan` predates an expired-slot replacement and remains an exact
        // provider-provenance handle, not merely a matching descriptor.
        CHECK(provider->Submit(std::move(request)));
    });
    CHECK(future.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    auto result = future.get();
    if (!result || result->error || !result->generation ||
        result->identity != identity || result->revisions.topologyVersion != 41) return 1;
    fixture.returnOwner->Drain();
    CHECK(result->generation->Geometry().topologyVersion == 41);
    CHECK(result->generation->Geometry().valueVersion == 43);
    CHECK(result->generation->Identity().generation == 99);
    CHECK(result->generation->Identity().deviceIndex == identity.deviceIndex);
    CheckPlanes(fixture.native, fixture.context, fixture.completion, *fixture.queueOwner, result->generation, oracle);
    provider->Shutdown(); executor->Shutdown(); preparer->Drain();
    fixture.queueOwner->Drain();
    if (!fixture.domain->Close()) return 1;
    fixture.completion->CloseAndJoin();
    if (fixture.returnOwner.get() == fixture.queueOwner.get() ||
        fixture.completion->queueOwner() != fixture.queueOwner.get() ||
        !fixture.context->timelineSemaphoreEnabled() ||
        fixture.context->computeQueue() != fixture.native->queue) {
        std::fprintf(stderr, "Vulkan Session provider owner/context contract failed\n");
        return 1;
    }
    std::puts("Vulkan Session provider native fixture: PASS");
    return 0;
}
