// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Native fixture for the Session Vulkan provider contract.
//
// This deliberately keeps the reply/return owner separate from the queue
// owner.  The provider is expected to reserve its reply transport before it
// enters the native owner; no callback in this fixture is permitted to touch
// mutable Session state directly from the Vulkan waiter.

#include "usdGen/executionPipeline.h"
#include "usdGen/executionResources.h"
#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/completionService.h"
#include "usdGen/vulkan/deviceGenerationAdapter.h"
#include "usdGen/vulkan/planExecutor.h"
#include "usdGen/vulkan/lengthScalePipeline.h"
#include "usdGen/vulkan/lengthCompactionPipeline.h"
#include "usdGen/vulkan/widthPipeline.h"
#include "usdGen/vulkan/widthBlendPipeline.h"
#include "usdGen/vulkan/nonWidthComparePipeline.h"
#include "usdGen/executionTaskGraph.h"
#include "usdGen/vulkan/sessionProvider.h"
#include "usdGen/sessionDeviceIntegration.h"
#include "usdGen/executionCache.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "../libs/base/usdGenMath/usdGenMath/hash.h"
#include "vulkanNativeFixture.h"
#include "floatUlpFixture.h"
#include "usdGen/vulkan/chargedBuffer.h"

#include <atomic>
#include <chrono>
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
    struct Named {
        UsdGenDeviceChannelMetadata metadata;
        std::vector<uint8_t> bytes;
    };
    std::vector<GfVec3f> points, rest;
    std::vector<uint32_t> curveOffsets;
    std::vector<uint64_t> stableIds;
    std::vector<float> hairT, width;
    std::vector<int32_t> rootPrim;
    std::vector<GfVec2f> rootUV;
    std::vector<Named> named;
    uint32_t computedPointUlps = 0;
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
    std::vector<uint32_t> ulps;
    std::vector<char const*> names;
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
        auto const& planes = lease.Planes();
        // Every publication retains the six core channels, including an
        // empty packet. Optional roots are public only when they have values.
        auto const rootPlanes = expected.rootPrim.empty() ? 0u : 2u;
        CHECK(planes.size() == 6 + rootPlanes + expected.named.size());
        for (auto const& plane : planes)
            CHECK(plane.metadata.name != "sourceRootT" && plane.metadata.name != "sourceRootB" &&
                  plane.metadata.name != "sourceRootN");
        std::vector<std::pair<char const*, std::vector<uint8_t>>> want{
            {"points", Bytes(expected.points)}, {"rest", Bytes(expected.rest)},
            {"curveOffsets", Bytes(expected.curveOffsets)}, {"curveId", Bytes(expected.stableIds)},
            {"hairT", Bytes(expected.hairT)}, {"widths", Bytes(expected.width)}};
        if (!expected.rootPrim.empty()) {
            CHECK(expected.rootUV.size() == expected.rootPrim.size());
            want.emplace_back("skinprim", Bytes(expected.rootPrim));
            want.emplace_back("skinprimuv", Bytes(expected.rootUV));
        }
        for (auto const& named : expected.named) {
            want.emplace_back(named.metadata.name.c_str(), named.bytes);
            auto found = std::find_if(planes.begin(), planes.end(), [&](auto const& plane) {
                return plane.metadata.name == named.metadata.name;
            });
            CHECK(found != planes.end());
            auto const& actual = found->metadata;
            auto const& schema = named.metadata;
            CHECK(actual.type == schema.type && actual.domain == schema.domain &&
                  actual.elementCount == schema.elementCount && actual.arity == schema.arity &&
                  actual.strideBytes == schema.strideBytes && actual.readOnly &&
                  actual.semantic == UsdGenDeviceChannelSemantic::Generic);
        }
        reads->watch = service->Reserve(reads, [reads, completed, &owner](VkResult proof) {
            CHECK(owner.IsExecutingOwner() && proof == VK_SUCCESS);
            for (size_t i = 0; i != reads->staging.size(); ++i) {
                CHECK(reads->staging[i]->PollComplete() == VK_SUCCESS);
                void* mapped = nullptr;
                CHECK(vkMapMemory(reads->native->device, reads->staging[i]->memory(), 0,
                    reads->expected[i].size(), 0, &mapped) == VK_SUCCESS);
                if (reads->ulps[i])
                    CHECK(FloatBytesWithinUlps(mapped, reads->expected[i].data(),
                                              reads->expected[i].size(), reads->ulps[i]));
                else
                if (std::memcmp(mapped, reads->expected[i].data(), reads->expected[i].size()) != 0) {
                    auto const* first = static_cast<unsigned char const*>(mapped);
                    for (size_t b = 0; b != reads->expected[i].size(); ++b)
                        if (first[b] != reads->expected[i][b]) {
                            std::fprintf(stderr, "plane mismatch %s at byte %zu (got %u want %u)\n",
                                         reads->names[i], b, first[b], reads->expected[i][b]);
                            break;
                        }
                    CHECK(false);
                }
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
            if (plane == planes.end() || plane->bytes != item.second.size())
                std::fprintf(stderr, "Plane %s: native bytes=%llu expected=%zu curves=%zu points=%zu\n",
                    item.first, static_cast<unsigned long long>(plane == planes.end() ? UINT64_MAX : plane->bytes),
                    item.second.size(), expected.stableIds.size(), expected.points.size());
            CHECK(plane != planes.end() && plane->bytes == item.second.size());
            if (std::strcmp(item.first, "skinprim") == 0)
                CHECK(plane->metadata.type == UsdGenDeviceValueType::Int32 &&
                      plane->metadata.domain == UsdGenDeviceDomain::Primitive &&
                      plane->metadata.semantic == UsdGenDeviceChannelSemantic::RootPrim &&
                      plane->metadata.elementCount == expected.stableIds.size() && plane->metadata.arity == 1);
            if (std::strcmp(item.first, "skinprimuv") == 0)
                CHECK(plane->metadata.type == UsdGenDeviceValueType::Float32x2 &&
                      plane->metadata.domain == UsdGenDeviceDomain::Primitive &&
                      plane->metadata.semantic == UsdGenDeviceChannelSemantic::RootUV &&
                      plane->metadata.elementCount == expected.stableIds.size() && plane->metadata.arity == 2);
            if (item.second.empty()) {
                CHECK(plane->buffer == VK_NULL_HANDLE);
                continue;
            }
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size = plane->bytes;
            bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto staging = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch); CHECK(staging);
            CHECK(staging->MarkSubmitted(reads->fence, reads) == VK_SUCCESS);
            VkBufferCopy copy{0, 0, plane->bytes}; vkCmdCopyBuffer(reads->command, plane->buffer, staging->buffer(), 1, &copy);
            reads->staging.push_back(std::move(staging)); reads->expected.push_back(item.second);
            reads->ulps.push_back(std::strcmp(item.first, "points") == 0 ? expected.computedPointUlps : 0);
            reads->names.push_back(item.first);
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
    auto const& buffer = graph.Node(graph.NodeIdForPath(desc.terminal)).buffer;
    out->points.resize(buffer.totalCvs);
    out->rest.assign(buffer.rest.begin(), buffer.rest.end());
    for (size_t i = 0; i != buffer.totalCvs; ++i)
        out->points[i] = {buffer.px[i], buffer.py[i], buffer.pz[i]};
    out->curveOffsets.assign(buffer.cvOffsets.begin(), buffer.cvOffsets.end());
    // CurveSource represents an all-dropped capture without offsets; native
    // publication canonicalizes that empty topology to its required sentinel.
    if (out->points.empty() && out->curveOffsets.empty()) out->curveOffsets = {0};
    out->stableIds.assign(buffer.curveId.begin(), buffer.curveId.end());
    out->hairT.assign(buffer.hairT.begin(), buffer.hairT.end());
    out->width.assign(buffer.width.begin(), buffer.width.end());
    out->rootPrim.assign(buffer.rootPrim.begin(), buffer.rootPrim.end());
    out->rootUV.assign(buffer.rootUV.begin(), buffer.rootUV.end());
    out->named.clear();
    for (auto const* collection : {&buffer.extraCv, &buffer.extraCurve}) {
        for (auto const& plane : *collection) {
            bool const floats = plane.type == TfToken("float");
            auto const domain = plane.interpolation == TfToken("vertex") ? UsdGenDeviceDomain::Point :
                plane.interpolation == TfToken("uniform") ? UsdGenDeviceDomain::Primitive : UsdGenDeviceDomain::Groom;
            uint64_t const elements = domain == UsdGenDeviceDomain::Point ? buffer.totalCvs :
                domain == UsdGenDeviceDomain::Primitive ? buffer.totalCurves : 1;
            SixPlaneOracle::Named named;
            named.metadata = {plane.name.GetString(), floats ? UsdGenDeviceValueType::Float32 : UsdGenDeviceValueType::Int32,
                              domain, elements, plane.arity, uint32_t(plane.arity * 4), true,
                              UsdGenDeviceChannelSemantic::Generic};
            named.bytes.resize((floats ? plane.f.size() : plane.i.size()) * 4);
            if (!named.bytes.empty()) std::memcpy(named.bytes.data(),
                floats ? static_cast<void const*>(plane.f.cdata()) : static_cast<void const*>(plane.i.cdata()), named.bytes.size());
            out->named.push_back(std::move(named));
        }
    }
    return true;
}

static UsdGenGraphDesc MakeProviderDescriptor(bool withLength = false) {
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
    if (withLength) {
        // This is the only non-identity Length form admitted by the Vulkan
        // plan: fixed topology, literal scale, followed by Width.
        UsdGenNodeDesc length;
        length.path = SdfPath("/SessionProvider/length");
        length.type = TfToken("UsdGenLength");
        length.inputs = {source.path};
        length.params = {{TfToken("length:value"), VtValue(0.5f), false},
                         {TfToken("length:mode"), VtValue(TfToken("scale")), false}};
        width.inputs = {length.path};
        desc.nodes = {source, length, width};
    } else {
        desc.nodes = {source, width};
    }
    desc.terminal = width.path;
    return desc;
}

static UsdGenGraphDesc MakeProviderDagDescriptor(bool withLength) {
    auto desc = MakeProviderDescriptor(withLength);
    auto right = desc.nodes.back();
    right.path = SdfPath("/SessionProvider/rightWidth");
    right.params[0].value = VtValue(2.0f);
    desc.nodes.push_back(right);
    UsdGenNodeDesc blend;
    blend.path = SdfPath("/SessionProvider/blend");
    blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {desc.nodes[desc.nodes.size() - 2].path, right.path};
    blend.params = {{TfToken("widthBlend:weight"), VtValue(0.25f), false}};
    desc.nodes.push_back(blend);
    desc.terminal = blend.path;
    return desc;
}

static void AddRootAuthoredPlanes(UsdGenCurveSetDesc* curves) {
    auto add = [&](char const* name, UsdGenAuthoredPlaneType type,
                   UsdGenAuthoredPlaneDomain domain, uint8_t arity) {
        UsdGenAuthoredPlaneDesc plane;
        plane.name = TfToken(name); plane.type = type; plane.domain = domain; plane.arity = arity;
        size_t const elements = domain == UsdGenAuthoredPlaneDomain::Point ? curves->points.size() :
            domain == UsdGenAuthoredPlaneDomain::Primitive ? curves->curveVertexCounts.size() : 1;
        if (type == UsdGenAuthoredPlaneType::Float32) {
            plane.floatValues.resize(elements * arity);
            for (size_t i = 0; i != plane.floatValues.size(); ++i) plane.floatValues[i] = float(i) + .125f;
        } else {
            plane.intValues.resize(elements * arity);
            for (size_t i = 0; i != plane.intValues.size(); ++i) plane.intValues[i] = int(i * 7 + 3);
        }
        curves->authoredPlanes.push_back(std::move(plane));
    };
    add("pointFloat16", UsdGenAuthoredPlaneType::Float32, UsdGenAuthoredPlaneDomain::Point, 16);
    add("pointInt3", UsdGenAuthoredPlaneType::Int32, UsdGenAuthoredPlaneDomain::Point, 3);
    add("primitiveFloat2", UsdGenAuthoredPlaneType::Float32, UsdGenAuthoredPlaneDomain::Primitive, 2);
    add("primitiveInt1", UsdGenAuthoredPlaneType::Int32, UsdGenAuthoredPlaneDomain::Primitive, 1);
    add("groomFloat4", UsdGenAuthoredPlaneType::Float32, UsdGenAuthoredPlaneDomain::Groom, 4);
    add("groomInt16", UsdGenAuthoredPlaneType::Int32, UsdGenAuthoredPlaneDomain::Groom, 16);
}

// This mirrors the C3 root fixture exercised by the direct executor test,
// but reaches it solely through the Session provider's source capture.
static UsdGenGraphDesc MakeProviderRootDescriptor(bool withLength, bool useRest, TfToken rebind,
                                                  bool affine = false) {
    auto desc = MakeProviderDescriptor(withLength);
    auto& curves = desc.curveSets.front();
    curves.rest = curves.points;
    curves.skinPrim = {0, 0}; curves.skinPrimUv = {{.20f,.30f}, {.60f,.20f}};
    curves.rootFrame.resize(2, GfMatrix4d(1.0));
    curves.rootFrame[0][3][0] = 20.0; curves.rootFrame[1][3][0] = 10.0;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/SessionProvider/Scalp"); surface.id = 31;
    surface.faceVertexCounts = {4}; surface.faceVertexIndices = {0,1,2,3};
    surface.restPoints = {{0,0,0},{4,0,0},{4,0,4},{0,0,4}};
    surface.points = surface.restPoints; surface.uv = {{0,0},{1,0},{1,1},{0,1}};
    if (affine) {
        // Keep source points/rest in local space while making the bound
        // surface live in a genuinely different affine object space.
        GfMatrix4d relative(1.0);
        relative[0][0] = 0.; relative[0][1] = 2.;
        relative[1][0] = -3.; relative[1][1] = 0.;
        relative[2][2] = .5;
        relative[3][0] = 5.; relative[3][1] = -2.; relative[3][2] = 7.;
        GfMatrix4d sourceWorld(1.0);
        sourceWorld[0][0] = 0.; sourceWorld[0][1] = -1.;
        sourceWorld[1][0] = 1.; sourceWorld[1][1] = 0.;
        sourceWorld[3][0] = 11.; sourceWorld[3][1] = -13.; sourceWorld[3][2] = 17.;
        curves.worldMatrix = sourceWorld;
        surface.worldMatrix = relative * sourceWorld;
        curves.rootFrame.clear(); // capture the actual source-relative frames
    }
    desc.surfaces = {surface};
    auto& source = desc.nodes.front(); source.surfaces = {surface.path};
    source.params = {{TfToken("useRest"), VtValue(useRest), false},
                     {TfToken("rebind"), VtValue(rebind), false}};
    AddRootAuthoredPlanes(&curves);
    return desc;
}

static UsdGenGraphDesc MakeProviderRootDagDescriptor(bool withLength, bool affine = false) {
    auto desc = MakeProviderRootDescriptor(withLength, true, TfToken("onError"), affine);
    auto right = desc.nodes.back();
    right.path = SdfPath("/SessionProvider/rightWidth"); right.params[0].value = VtValue(2.0f);
    desc.nodes.push_back(right);
    UsdGenNodeDesc blend;
    blend.path = SdfPath("/SessionProvider/blend"); blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {desc.nodes[desc.nodes.size() - 2].path, right.path};
    blend.params = {{TfToken("widthBlend:weight"), VtValue(.25f), false}};
    desc.nodes.push_back(blend); desc.terminal = blend.path;
    return desc;
}

struct NativeLifetime {
    std::shared_ptr<NativeOwner> native;
};

struct ReturnRelay {
    UsdGenExecutionPipeline::CommandTicket ticket;
    std::shared_ptr<std::promise<std::shared_ptr<const UsdGenSessionDeviceResult>>> promise;
};

// Holds an actual, proved native result without entering the Session return
// transport. The future synchronizes access to the single retained payload.
struct HeldDeviceReturn {
    std::promise<void> ready;
    std::shared_ptr<const UsdGenSessionDeviceResult> result;
    UsdGenSessionDeviceReturn transport;
    std::atomic<unsigned> returns{0};
};

struct HoldingProvider final : UsdGenSessionDeviceProvider {
    std::shared_ptr<UsdGenSessionDeviceProvider> inner;
    std::shared_ptr<HeldDeviceReturn> held;
    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphDesc const& desc,
                                         UsdGenDiagnostics* diagnostics) const override {
        return inner->Compile(desc, diagnostics);
    }
    UsdGenSessionDeviceIdentity Identity() const noexcept override { return inner->Identity(); }
    bool NotifyContextLost(UsdGenSessionDeviceIdentity const& identity) noexcept override {
        return inner->NotifyContextLost(identity);
    }
    bool Submit(UsdGenSessionDeviceRequest request) override {
        held->transport = std::move(request.returnTransport);
        request.returnTransport = [gate = held](auto result) {
            CHECK(gate->returns.fetch_add(1) == 0);
            gate->result = std::move(result);
            gate->ready.set_value();
            return true; // This transport really retains the native result.
        };
        return inner->Submit(std::move(request));
    }
    void Shutdown() noexcept override { inner->Shutdown(); }
};

// Test-only transport checks allocated stage identities independently of the
// compiler's value-lineage lookup, and can inject an otherwise valid native
// result with the wrong topology revision to exercise Session rejection.
struct CullRevisionProvider final : UsdGenSessionDeviceProvider {
    std::shared_ptr<UsdGenSessionDeviceProvider> inner;
    std::atomic<bool> corruptTopology{false};
    std::atomic<size_t> topologyOrdinal{SIZE_MAX};
    UsdGenSessionDevicePlanHandle Compile(UsdGenGraphDesc const& desc,
                                         UsdGenDiagnostics* diagnostics) const override {
        return inner->Compile(desc, diagnostics);
    }
    UsdGenSessionDeviceIdentity Identity() const noexcept override { return inner->Identity(); }
    bool NotifyContextLost(UsdGenSessionDeviceIdentity const& identity) noexcept override {
        return inner->NotifyContextLost(identity);
    }
    bool Submit(UsdGenSessionDeviceRequest request) override {
        CHECK(request.authoritativeRevisions &&
              !request.authoritativeRevisions->intermediateValueVersions.empty());
        auto const revisions = *request.authoritativeRevisions;
        bool const corrupt = corruptTopology.load();
        size_t const ordinal = topologyOrdinal.load();
        uint64_t const expectedTopology = ordinal == SIZE_MAX ? revisions.intermediateValueVersions.back()
            : revisions.intermediateValueVersions.at(ordinal);
        request.returnTransport = [transport = std::move(request.returnTransport), revisions, corrupt, expectedTopology](auto result) {
            if (result && !result->error && result->generation) {
                auto const& generation = result->generation;
                CHECK(generation->Geometry().topologyVersion == expectedTopology);
                CHECK(generation->Geometry().valueVersion == revisions.finalValueVersion);
                if (corrupt) {
                    auto geometry = generation->Geometry();
                    geometry.topologyVersion = revisions.topologyVersion;
                    auto altered = std::make_shared<UsdGenSessionDeviceResult>(*result);
                    altered->generation = UsdGenDeviceGeneration::Create({generation->Identity(),
                        geometry, generation->Tool(), generation->Channels(), generation->Owner()});
                    CHECK(altered->generation);
                    return transport(std::move(altered));
                }
            }
            return transport(std::move(result));
        };
        return inner->Submit(std::move(request));
    }
    void Shutdown() noexcept override { inner->Shutdown(); }
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
    // CMake supplies a flat Width shader first. Optional shader variants use
    // explicit flags so the standalone return/admission modes keep their
    // original two-argument form:
    //   width.spv [--roots [--affine]]
    //            [--named] [--length length-scale.spv] [--dag blend.spv]
    //            [--compare compare.spv] [--cut cut-extend.spv] [--reparam reparam.spv]
    //            [--minimum minimum-length.spv] [--literal-v1 literal-v1.spv] [--envelope-v1 envelope-v1.spv]
    if (argc < 2) { std::puts("Vulkan Session provider shader unavailable"); return 77; }
    bool lengthMode = false, dagMode = false, lostReturnMode = false, admissionRollbackMode = false;
    char const* lengthShaderPath = nullptr;
    char const* cullShaderPath = nullptr;
    char const* setShaderPath = nullptr;
    char const* cutShaderPath = nullptr;
    char const* reparamShaderPath = nullptr;
    char const* minimumShaderPath = nullptr;
    char const* literalV1ShaderPath = nullptr;
    char const* envelopeV1ShaderPath = nullptr;
    char const* blendShaderPath = nullptr;
    char const* compareShaderPath = nullptr;
    bool namedMode = false, rootsMode = false, affineMode = false;
    for (int argument = 2; argument < argc; ++argument) {
        std::string const option(argv[argument]);
        if (option == "--length" && !lengthMode && argument + 1 < argc) {
            lengthMode = true; lengthShaderPath = argv[++argument];
        } else if (option == "--cull" && !cullShaderPath && argument + 1 < argc) {
            cullShaderPath = argv[++argument];
        } else if (option == "--set" && !setShaderPath && argument + 1 < argc) {
            setShaderPath = argv[++argument];
        } else if (option == "--cut" && !cutShaderPath && argument + 1 < argc) {
            cutShaderPath = argv[++argument];
        } else if (option == "--reparam" && !reparamShaderPath && argument + 1 < argc) {
            reparamShaderPath = argv[++argument];
        } else if (option == "--minimum" && !minimumShaderPath && argument + 1 < argc) {
            minimumShaderPath = argv[++argument];
        } else if (option == "--literal-v1" && !literalV1ShaderPath && argument + 1 < argc) {
            literalV1ShaderPath = argv[++argument];
        } else if (option == "--envelope-v1" && !envelopeV1ShaderPath && argument + 1 < argc) {
            envelopeV1ShaderPath = argv[++argument];
        } else if (option == "--dag" && !dagMode && argument + 1 < argc) {
            dagMode = true; blendShaderPath = argv[++argument];
        } else if (option == "--compare" && !compareShaderPath && argument + 1 < argc) {
            compareShaderPath = argv[++argument];
        } else if (option == "--named" && !namedMode) {
            namedMode = true;
        } else if (option == "--roots" && !rootsMode) {
            rootsMode = true;
        } else if (option == "--affine" && !affineMode) {
            affineMode = true;
        } else if (option == "--lost-return" && !lostReturnMode) {
            lostReturnMode = true;
        } else if (option == "--admission-rollback" && !admissionRollbackMode) {
            admissionRollbackMode = true;
        } else {
            std::fprintf(stderr, "invalid Vulkan Session shader/mode arguments\n"); return 1;
        }
    }
    if (compareShaderPath && (!dagMode || (!lengthMode && !cullShaderPath))) return 1;
    if (namedMode && !dagMode) return 1;
    if (affineMode && !rootsMode) return 1;
    if (cullShaderPath && ((lengthMode && !setShaderPath) || lostReturnMode ||
                          admissionRollbackMode || affineMode || namedMode)) return 1;
    if (cullShaderPath && dagMode && !compareShaderPath) return 1;
    if (setShaderPath && !lengthMode) return 1;
    if (cutShaderPath && (!lengthMode || !setShaderPath)) return 1;
    if (reparamShaderPath && (!lengthMode || !setShaderPath || !cutShaderPath)) return 1;
    if (minimumShaderPath && (!lengthMode || !setShaderPath || !cutShaderPath || !reparamShaderPath)) return 1;
    if (literalV1ShaderPath && (!lengthMode || !setShaderPath || !cutShaderPath || !reparamShaderPath ||
                                !minimumShaderPath)) return 1;
    if (envelopeV1ShaderPath && (!lengthMode || !setShaderPath || !cutShaderPath || !reparamShaderPath ||
                                 !minimumShaderPath || !literalV1ShaderPath)) return 1;
    if ((lostReturnMode && (admissionRollbackMode || lengthMode || dagMode || rootsMode)) ||
        (admissionRollbackMode && (lostReturnMode || lengthMode || dagMode || rootsMode))) {
        std::fprintf(stderr, "standalone Vulkan Session modes cannot select shader variants\n"); return 1;
    }
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
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(fixture.native->physical, &properties);
    if (reparamShaderPath &&
        (properties.limits.maxPerStageDescriptorStorageBuffers < 5 ||
         properties.limits.maxDescriptorSetStorageBuffers < 5 ||
         properties.limits.maxPushConstantsSize < 20)) return 77;
    if (minimumShaderPath &&
        (properties.limits.maxPerStageDescriptorStorageBuffers < 5 ||
         properties.limits.maxDescriptorSetStorageBuffers < 5 ||
         properties.limits.maxPushConstantsSize < 32)) return 77;
    if (literalV1ShaderPath &&
        (properties.limits.maxPerStageDescriptorStorageBuffers < 6 ||
         properties.limits.maxDescriptorSetStorageBuffers < 6 ||
         properties.limits.maxPushConstantsSize < 48)) return 77;
    if (envelopeV1ShaderPath &&
        (properties.limits.maxPerStageDescriptorStorageBuffers < 6 ||
         properties.limits.maxDescriptorSetStorageBuffers < 6 ||
         properties.limits.maxPushConstantsSize < 56)) return 77;
    std::vector<uint32_t> spirv(raw.size() / sizeof(uint32_t));
    std::memcpy(spirv.data(), raw.data(), raw.size());
    auto pipeline = WidthPipeline::Create(fixture.context, spirv);
    if (!pipeline) return 77;
    std::shared_ptr<LengthScalePipeline> lengthPipeline;
    if (lengthMode) {
        std::ifstream lengthShader(lengthShaderPath, std::ios::binary);
        std::vector<char> lengthRaw((std::istreambuf_iterator<char>(lengthShader)), {});
        CHECK(!lengthRaw.empty() && lengthRaw.size() % sizeof(uint32_t) == 0);
        std::vector<uint32_t> lengthCode(lengthRaw.size() / sizeof(uint32_t));
        std::memcpy(lengthCode.data(), lengthRaw.data(), lengthRaw.size());
        lengthPipeline = LengthScalePipeline::Create(fixture.context, lengthCode);
        CHECK(lengthPipeline);
        CHECK(!lengthPipeline->HasSet());
        if (setShaderPath) {
            std::ifstream setShader(setShaderPath, std::ios::binary);
            std::vector<char> setRaw((std::istreambuf_iterator<char>(setShader)), {});
            CHECK(!setRaw.empty() && setRaw.size() % sizeof(uint32_t) == 0);
            std::vector<uint32_t> setCode(setRaw.size() / sizeof(uint32_t));
            std::memcpy(setCode.data(), setRaw.data(), setRaw.size());
            lengthPipeline = LengthScalePipeline::CreateWithSet(fixture.context, lengthCode, setCode);
            CHECK(lengthPipeline && lengthPipeline->HasSet());
            if (cutShaderPath) {
                std::ifstream cutShader(cutShaderPath, std::ios::binary);
                std::vector<char> cutRaw((std::istreambuf_iterator<char>(cutShader)), {});
                CHECK(!cutRaw.empty() && cutRaw.size() % sizeof(uint32_t) == 0);
                std::vector<uint32_t> cutCode(cutRaw.size() / sizeof(uint32_t));
                std::memcpy(cutCode.data(), cutRaw.data(), cutRaw.size());
                lengthPipeline = LengthScalePipeline::CreateWithCutExtend(
                    fixture.context, lengthCode, setCode, cutCode);
                CHECK(lengthPipeline && lengthPipeline->HasSet() && lengthPipeline->HasCutExtend());
                if (reparamShaderPath) {
                    std::ifstream reparamShader(reparamShaderPath, std::ios::binary);
                    std::vector<char> reparamRaw((std::istreambuf_iterator<char>(reparamShader)), {});
                    CHECK(!reparamRaw.empty() && reparamRaw.size() % sizeof(uint32_t) == 0);
                    std::vector<uint32_t> reparamCode(reparamRaw.size() / sizeof(uint32_t));
                    std::memcpy(reparamCode.data(), reparamRaw.data(), reparamRaw.size());
                    if (minimumShaderPath) {
                        std::ifstream minimumShader(minimumShaderPath, std::ios::binary);
                        std::vector<char> minimumRaw((std::istreambuf_iterator<char>(minimumShader)), {});
                        CHECK(!minimumRaw.empty() && minimumRaw.size() % sizeof(uint32_t) == 0);
                        std::vector<uint32_t> minimumCode(minimumRaw.size() / sizeof(uint32_t));
                        std::memcpy(minimumCode.data(), minimumRaw.data(), minimumRaw.size());
                        if (literalV1ShaderPath) {
                            std::ifstream literalShader(literalV1ShaderPath, std::ios::binary);
                            std::vector<char> literalRaw((std::istreambuf_iterator<char>(literalShader)), {});
                            CHECK(!literalRaw.empty() && literalRaw.size() % sizeof(uint32_t) == 0);
                            std::vector<uint32_t> literalCode(literalRaw.size() / sizeof(uint32_t));
                            std::memcpy(literalCode.data(), literalRaw.data(), literalRaw.size());
                            if (envelopeV1ShaderPath) {
                                std::ifstream envelopeShader(envelopeV1ShaderPath, std::ios::binary);
                                std::vector<char> envelopeRaw((std::istreambuf_iterator<char>(envelopeShader)), {});
                                CHECK(!envelopeRaw.empty() && envelopeRaw.size() % sizeof(uint32_t) == 0);
                                std::vector<uint32_t> envelopeCode(envelopeRaw.size() / sizeof(uint32_t));
                                std::memcpy(envelopeCode.data(), envelopeRaw.data(), envelopeRaw.size());
                                lengthPipeline = LengthScalePipeline::CreateWithEnvelopeV1(
                                    fixture.context, lengthCode, setCode, cutCode, reparamCode, minimumCode,
                                    literalCode, envelopeCode);
                            } else lengthPipeline = LengthScalePipeline::CreateWithLiteralV1(
                                fixture.context, lengthCode, setCode, cutCode, reparamCode, minimumCode, literalCode);
                        } else lengthPipeline = LengthScalePipeline::CreateWithMinimum(
                            fixture.context, lengthCode, setCode, cutCode, reparamCode, minimumCode);
                    } else {
                        lengthPipeline = LengthScalePipeline::CreateWithReparam(
                            fixture.context, lengthCode, setCode, cutCode, reparamCode);
                    }
                    CHECK(lengthPipeline && lengthPipeline->HasSet() &&
                          lengthPipeline->HasCutExtend() && lengthPipeline->HasReparam() &&
                          (!minimumShaderPath || lengthPipeline->HasMinimum()) &&
                          (!literalV1ShaderPath || lengthPipeline->HasLiteralV1()) &&
                          (!envelopeV1ShaderPath || lengthPipeline->HasEnvelopeV1()));
                }
            }
        }
    }
    std::shared_ptr<WidthBlendPipeline> blendPipeline;
    if (dagMode) {
        std::ifstream blendShader(blendShaderPath, std::ios::binary);
        std::vector<char> blendRaw((std::istreambuf_iterator<char>(blendShader)), {});
        CHECK(!blendRaw.empty() && blendRaw.size() % sizeof(uint32_t) == 0);
        std::vector<uint32_t> blendCode(blendRaw.size() / sizeof(uint32_t));
        std::memcpy(blendCode.data(), blendRaw.data(), blendRaw.size());
        blendPipeline = WidthBlendPipeline::Create(fixture.context, blendCode);
        CHECK(blendPipeline);
    }
    std::shared_ptr<NonWidthComparePipeline> comparePipeline;
    if (compareShaderPath) {
        std::ifstream compareShader(compareShaderPath, std::ios::binary);
        std::vector<char> compareRaw((std::istreambuf_iterator<char>(compareShader)), {});
        CHECK(!compareRaw.empty() && compareRaw.size() % sizeof(uint32_t) == 0);
        std::vector<uint32_t> compareCode(compareRaw.size() / sizeof(uint32_t));
        std::memcpy(compareCode.data(), compareRaw.data(), compareRaw.size());
        comparePipeline = NonWidthComparePipeline::Create(fixture.context, compareCode);
        CHECK(comparePipeline);
    }
    auto preparer = UsdGenExecutionTaskGraph::GetOrCreate(*fixture.runtime,
                                                           "vulkan-session-provider", 7044);
    std::shared_ptr<LengthCompactionPipeline> compactionPipeline;
    if (cullShaderPath) {
        std::ifstream codeFile(cullShaderPath, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(codeFile)), {});
        CHECK(!bytes.empty() && bytes.size() % sizeof(uint32_t) == 0);
        std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
        std::memcpy(code.data(), bytes.data(), bytes.size());
        compactionPipeline = LengthCompactionPipeline::Create(fixture.context, code);
        CHECK(compactionPipeline);
    }
    std::string reason;
    auto executor = VulkanPlanExecutor::Create(
        {fixture.domain, pipeline, preparer, lengthPipeline, blendPipeline, comparePipeline,
         compactionPipeline}, &reason);
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
    if (lostReturnMode) {
        auto held = std::make_shared<HeldDeviceReturn>();
        auto ready = held->ready.get_future();
        auto wrapper = std::make_shared<HoldingProvider>();
        wrapper->inner = provider; wrapper->held = held;
        std::promise<void> closedPromise;
        auto closed = closedPromise.get_future();
        UsdGenSessionDeviceObservers observers;
        observers.admissionClosed = [&] { closedPromise.set_value(); };
        auto losingSession = CreateUsdGenDeviceSession(2, 8, cacheDomain, wrapper, observers);
        CHECK(losingSession);
        losingSession->SetGraphDesc(MakeProviderDescriptor());
        losingSession->SetDevicePublicationEnabled(true);
        std::atomic<unsigned> terminals{0};
        std::promise<UsdGenExecutionPipeline::Outcome> terminalPromise;
        auto terminal = terminalPromise.get_future();
        CHECK(losingSession->CommitAsync(1.0, UsdGenCommitReason::SetTime,
            [&](auto snapshot, auto outcome) {
                CHECK(terminals.fetch_add(1) == 0);
                CHECK(!snapshot || !snapshot->generation);
                terminalPromise.set_value(outcome);
            }));
        CHECK(ready.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        ready.get();
        CHECK(held->result && !held->result->error && held->result->generation);
        // Drain only the independent native owner, never the still-occupied
        // Session async gate. No worker is blocked to manufacture the race.
        fixture.queueOwner->Drain();
        auto const charged = fixture.context->resources()->Snapshot().usedBytes;
        CHECK(charged > 0);
        std::weak_ptr<const UsdGenDeviceGeneration> retained = held->result->generation;
        auto destroying = std::async(std::launch::async,
            [session = std::move(losingSession)]() mutable { session.reset(); });
        CHECK(closed.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        closed.get(); // Actual admission is now closed, not a timing guess.
        CHECK(!held->transport(held->result));
        CHECK(!held->transport(held->result)); // Duplicate cannot settle again.
        CHECK(terminal.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        CHECK(terminal.get() == UsdGenExecutionPipeline::Outcome::Superseded);
        CHECK(destroying.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
        destroying.get();
        CHECK(terminals.load() == 1 && held->returns.load() == 1);
        held->result.reset(); held->transport = {};
        wrapper.reset(); held.reset();
        CHECK(!retained.expired());
        CHECK(cacheDomain->Size() == 0);
        provider->Shutdown(); executor->Shutdown(); preparer->Drain();
        fixture.queueOwner->Drain();
        CHECK(fixture.queueOwner->OutstandingCommands() == 0);
        CHECK(fixture.queueOwner->CallbackFailures() == 0);
        CHECK(fixture.context->resources()->Snapshot().usedBytes == charged);
        // Native work is already proved and no consumer lease is outstanding:
        // adapter close is allowed even though Session quarantine retains the
        // immutable allocation owner for the remainder of this process.
        CHECK(fixture.domain->Close());
        fixture.completion->CloseAndJoin();
        CHECK(!retained.expired());
        CHECK(fixture.context->resources()->Snapshot().usedBytes == charged);
        std::puts("Vulkan Session lost-return quarantine: PASS");
        return 0;
    }
    std::shared_ptr<CullRevisionProvider> cullProvider;
    std::shared_ptr<UsdGenSessionDeviceProvider> sessionProvider = provider;
    if (cullShaderPath) {
        cullProvider = std::make_shared<CullRevisionProvider>();
        cullProvider->inner = provider;
        sessionProvider = cullProvider;
    }
    auto session = CreateUsdGenDeviceSession(2, 8, cacheDomain, sessionProvider);
    if (!session) return 1;
    if (cullShaderPath) {
        auto descriptor = rootsMode ? MakeProviderRootDescriptor(true, true, TfToken("onError"))
                                    : MakeProviderDescriptor(true);
        auto& controls = descriptor.nodes[1].params;
        controls[0].value = VtValue(1.0f);
        controls[1].value = VtValue(TfToken("cull"));
        controls.push_back({TfToken("cullThreshold"), VtValue(1.5f), false});
        session->SetDevicePublicationEnabled(true);
        auto commitChecked = [&](UsdGenGraphDesc const& desc, SixPlaneOracle const& oracle) {
            session->SetGraphDesc(desc);
            auto snapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            CHECK(snapshot && !snapshot->diagnostics.HasErrors() && snapshot->generation &&
                  snapshot->generation->device);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, snapshot->generation->device, oracle);
            return snapshot;
        };
        auto publish = [&](UsdGenGraphDesc const& desc, SixPlaneOracle* oracle) {
            // The older CPU Length implementation collapses short curves in
            // this path; it is not a topology oracle. Use CPU source/Width
            // evaluation followed by independent survivor gathering instead.
            auto referenceDesc = desc;
            float threshold = 0;
            for (auto& node : referenceDesc.nodes) if (node.type == TfToken("UsdGenLength")) {
                for (auto& param : node.params) {
                    if (param.name == TfToken("cullThreshold")) {
                        threshold = std::max(threshold, param.value.UncheckedGet<float>());
                        param.value = VtValue(0.f);
                    } else if (param.name == TfToken("length:mode")) param.value = VtValue(TfToken("scale"));
                }
            }
            CHECK(BuildCpuReference(referenceDesc, oracle));
            std::vector<uint32_t> curves, points, offsets{0};
            for (uint32_t c = 0; c != oracle->stableIds.size(); ++c) {
                uint32_t const begin = oracle->curveOffsets[c], end = oracle->curveOffsets[c+1];
                float length = 0;
                for (uint32_t p = begin + 1; p < end; ++p)
                    length += (oracle->points[p] - oracle->points[p-1]).GetLength();
                if (length < threshold) continue;
                curves.push_back(c);
                for (uint32_t p = begin; p < end; ++p) points.push_back(p);
                offsets.push_back(uint32_t(points.size()));
            }
            auto gather = [](auto* values, std::vector<uint32_t> const& indices) {
                auto before = std::move(*values); values->clear();
                if (before.empty()) return;
                for (auto index : indices) values->push_back(before[index]);
            };
            gather(&oracle->points, points); gather(&oracle->rest, points);
            gather(&oracle->width, points); gather(&oracle->hairT, points);
            gather(&oracle->stableIds, curves); gather(&oracle->rootPrim, curves);
            gather(&oracle->rootUV, curves); oracle->curveOffsets = std::move(offsets);
            for (auto& named : oracle->named) {
                if (named.metadata.domain == UsdGenDeviceDomain::Groom) continue;
                auto const& indices = named.metadata.domain == UsdGenDeviceDomain::Point ? points : curves;
                auto before = std::move(named.bytes);
                size_t const stride = named.metadata.strideBytes;
                named.bytes.resize(stride * indices.size());
                for (size_t i = 0; i != indices.size(); ++i)
                    std::memcpy(named.bytes.data() + stride*i, before.data() + stride*indices[i], stride);
                named.metadata.elementCount = indices.size();
            }
            return commitChecked(desc, *oracle);
        };
        // Axis-aligned lengths (1 and 2) avoid cross-backend rounding ambiguity.
        // value=1 is deliberately neutral for source/Width reference capture.
        SixPlaneOracle firstOracle;
        auto first = publish(descriptor, &firstOracle);
        CHECK(firstOracle.stableIds == std::vector<uint64_t>({20}));
        CHECK(first->generation->device->Geometry().topologyVersion !=
              first->generation->device->Geometry().valueVersion);
        auto bothDesc = descriptor;
        bothDesc.nodes[1].params.back().value = VtValue(1.0f);
        SixPlaneOracle bothOracle;
        auto both = publish(bothDesc, &bothOracle);
        CHECK(bothOracle.stableIds == std::vector<uint64_t>({10,20}));
        CHECK(first->generation->device->Geometry().topologyVersion !=
              both->generation->device->Geometry().topologyVersion);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, first->generation->device, firstOracle);
        auto repeatedDesc = bothDesc;
        auto secondCull = repeatedDesc.nodes[1];
        secondCull.path = SdfPath("/SessionProvider/secondCull");
        secondCull.inputs = {repeatedDesc.nodes[1].path};
        secondCull.params.back().value = VtValue(1.5f);
        repeatedDesc.nodes.back().inputs = {secondCull.path};
        repeatedDesc.nodes.insert(repeatedDesc.nodes.end() - 1, secondCull);
        SixPlaneOracle repeatedOracle;
        auto repeated = publish(repeatedDesc, &repeatedOracle);
        CHECK(repeatedOracle.stableIds == firstOracle.stableIds);
        auto emptyDesc = descriptor;
        emptyDesc.nodes[1].params.back().value = VtValue(3.0f);
        SixPlaneOracle emptyOracle;
        auto empty = publish(emptyDesc, &emptyOracle);
        CHECK(emptyOracle.stableIds.empty() && emptyOracle.points.empty() &&
              emptyOracle.curveOffsets == std::vector<uint32_t>({0}));
        SixPlaneOracle cachedOracle;
        auto cached = publish(descriptor, &cachedOracle);
        CHECK(cached->generation->device->Owner() == first->generation->device->Owner());
        CHECK(cached->generation->device->Geometry().topologyVersion ==
              first->generation->device->Geometry().topologyVersion);
        auto badRevisionDesc = descriptor;
        badRevisionDesc.nodes[1].params.back().value = VtValue(1.25f);
        cullProvider->corruptTopology = true;
        session->SetGraphDesc(badRevisionDesc);
        auto wrongTopology = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(wrongTopology && wrongTopology->diagnostics.HasErrors() &&
              wrongTopology->generation == cached->generation && session->NeedsCommit());
        cullProvider->corruptTopology = false;
        SixPlaneOracle retryOracle;
        auto retried = publish(badRevisionDesc, &retryOracle);
        CHECK(retryOracle.stableIds == firstOracle.stableIds);
        cached = publish(descriptor, &cachedOracle);
        if (dagMode) {
            // Independent allocations/revisions are not a semantic mismatch:
            // both thresholds retain the same full non-width packet.
            auto branch = descriptor;
            auto rightCull = descriptor.nodes[1];
            rightCull.path = SdfPath("/SessionProvider/rightCull");
            rightCull.params.back().value = VtValue(1.25f);
            auto rightWidth = descriptor.nodes.back();
            rightWidth.path = SdfPath("/SessionProvider/rightWidth");
            rightWidth.inputs = {rightCull.path};
            rightWidth.params[0].value = VtValue(2.f);
            UsdGenNodeDesc join;
            join.path = SdfPath("/SessionProvider/cullJoin");
            join.type = TfToken("UsdGenWidthBlend");
            join.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
            join.inputs = {descriptor.nodes.back().path, rightWidth.path};
            branch.nodes.push_back(rightCull); branch.nodes.push_back(rightWidth);
            branch.nodes.push_back(join); branch.terminal = join.path;
            cullProvider->topologyOrdinal = 0; // left Cull is the first operator
            SixPlaneOracle joinedOracle;
            auto joined = publish(branch, &joinedOracle);
            CHECK(joinedOracle.stableIds == firstOracle.stableIds);
            auto unequal = branch;
            unequal.nodes[3].params.back().value = VtValue(.5f);
            session->SetGraphDesc(unequal);
            auto rejectedJoin = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            CHECK(rejectedJoin && rejectedJoin->diagnostics.HasErrors() &&
                  rejectedJoin->generation == joined->generation && session->NeedsCommit());
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, joined->generation->device, joinedOracle);
            auto retryBranch = branch;
            retryBranch.nodes[3].params.back().value = VtValue(1.75f);
            SixPlaneOracle retryJoinOracle;
            auto retryJoin = publish(retryBranch, &retryJoinOracle);
            auto emptyBranch = branch;
            emptyBranch.nodes[1].params.back().value = VtValue(3.f);
            emptyBranch.nodes[3].params.back().value = VtValue(4.f);
            SixPlaneOracle emptyJoinOracle;
            auto emptyJoin = publish(emptyBranch, &emptyJoinOracle);
            CHECK(emptyJoinOracle.stableIds.empty());
            auto sharedBranch = branch;
            sharedBranch.nodes[4].inputs = {sharedBranch.nodes[1].path};
            sharedBranch.nodes.erase(sharedBranch.nodes.begin() + 3);
            SixPlaneOracle sharedOracle;
            auto sharedJoin = publish(sharedBranch, &sharedOracle);
            CHECK(sharedOracle.stableIds == joinedOracle.stableIds);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, first->generation->device, firstOracle);
            fixture.queueOwner->InvokeOwner([&] {
                joined.reset(); rejectedJoin.reset(); retryJoin.reset(); emptyJoin.reset(); sharedJoin.reset();
            });
            cullProvider->topologyOrdinal = SIZE_MAX;
            cached = publish(descriptor, &cachedOracle);
        }
        if (setShaderPath) {
            if (cutShaderPath) {
                // Exercise cut/extend through the rooted capture as well: its
                // compacted packet must retain roots and every authored plane.
                auto cutDesc = rootsMode ? MakeProviderRootDescriptor(true, true, TfToken("never"))
                                         : MakeProviderDescriptor(true);
                auto& cutCurves = cutDesc.curveSets.front();
                cutCurves.curveVertexCounts = {3};
                cutCurves.curveId = {20};
                cutCurves.points = {{0,0,0},{2,0,0},{2,2,0}};
                cutCurves.rest = cutCurves.points;
                cutCurves.widths = {1,2,3};
                if (rootsMode) {
                    cutCurves.skinPrim = {0};
                    cutCurves.skinPrimUv = {{.20f,.30f}};
                    cutCurves.rootFrame.assign(1, GfMatrix4d(1.0));
                    cutCurves.authoredPlanes.clear();
                    AddRootAuthoredPlanes(&cutCurves);
                }
                auto& cutNode = cutDesc.nodes[1];
                cutNode.params = {{TfToken("length:value"), VtValue(3.f), false},
                                  {TfToken("length:mode"), VtValue(TfToken("set")), false},
                                  {TfToken("length:method"), VtValue(TfToken("cutExtend")), false},
                                  {TfToken("rebuild"), VtValue(TfToken("keepParam")), false},
                                  {TfToken("cullThreshold"), VtValue(.25f), false}};
                SixPlaneOracle cutOracle;
                auto neutralCut = cutDesc;
                neutralCut.nodes[1].params[0].value = VtValue(1.f);
                neutralCut.nodes[1].params[1].value = VtValue(TfToken("scale"));
                neutralCut.nodes[1].params[2].value = VtValue(TfToken("scale"));
                neutralCut.nodes[1].params[4].value = VtValue(0.f);
                CHECK(BuildCpuReference(neutralCut, &cutOracle));
                cutOracle.curveOffsets = {0,3};
                // Do not use the CPU Length implementation as the cut oracle.
                // This is the neutral source/Width packet with exact bent CVs.
                cutOracle.points = {{0,0,0},{2,0,0},{2,1,0}};
                if (rootsMode) {
                    cutOracle.rootPrim = {0};
                    cutOracle.rootUV = {{.20f,.30f}};
                    CHECK(cutOracle.rootPrim.size() == 1 && cutOracle.rootUV.size() == 1 &&
                          cutOracle.named.size() >= 6);
                }
                cullProvider->topologyOrdinal = 1;
                auto cutSnapshot = commitChecked(cutDesc, cutOracle);
                auto extendDesc = cutDesc;
                extendDesc.nodes[1].params[0].value = VtValue(6.f);
                SixPlaneOracle extendOracle = cutOracle;
                extendOracle.points = {{0,0,0},{2,0,0},{2,4,0}};
                if (rootsMode) {
                    CHECK(extendOracle.rootPrim == std::vector<int32_t>({0}) &&
                          extendOracle.rootUV == std::vector<GfVec2f>({{.20f,.30f}}) &&
                          !extendOracle.named.empty());
                }
                auto extendSnapshot = commitChecked(extendDesc, extendOracle);
                // A positive set target is undefined for a zero-current
                // strand.  The failed publish must retain the last good packet.
                auto zeroCurrent = extendDesc;
                auto& zeroCurves = zeroCurrent.curveSets.front();
                zeroCurves.points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroCurves.rest = zeroCurves.points;
                session->SetGraphDesc(zeroCurrent);
                auto rejectedZero = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(rejectedZero && rejectedZero->diagnostics.HasErrors() &&
                      rejectedZero->generation == extendSnapshot->generation && session->NeedsCommit());
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, extendSnapshot->generation->device, extendOracle);
                // In contrast, target zero is well-defined. Its .25 cull
                // threshold removes the zero-length result.
                zeroCurrent.nodes[1].params[0].value = VtValue(0.f);
                SixPlaneOracle emptyCutOracle = cutOracle;
                emptyCutOracle.points.clear(); emptyCutOracle.rest.clear();
                emptyCutOracle.width.clear(); emptyCutOracle.hairT.clear();
                emptyCutOracle.stableIds.clear(); emptyCutOracle.rootPrim.clear();
                emptyCutOracle.rootUV.clear(); emptyCutOracle.curveOffsets = {0};
                for (auto& named : emptyCutOracle.named) {
                    if (named.metadata.domain == UsdGenDeviceDomain::Groom) continue;
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                cullProvider->topologyOrdinal = 1;
                auto zeroSnapshot = commitChecked(zeroCurrent, emptyCutOracle);
                cullProvider->topologyOrdinal = SIZE_MAX;
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, cutSnapshot->generation->device, cutOracle);
                // A leading Cull may compact to empty before cut/extend runs.
                // The later Cut must accept that bounded empty packet without
                // manufacturing a topology revision of its own.
                UsdGenNodeDesc leadingCull;
                leadingCull.path = SdfPath("/SessionProvider/leadingCutCull");
                leadingCull.type = TfToken("UsdGenLength");
                leadingCull.inputs = {cutDesc.nodes[0].path};
                leadingCull.params = {{TfToken("length:value"), VtValue(1.f), false},
                                      {TfToken("length:mode"), VtValue(TfToken("cull")), false},
                                      {TfToken("cullThreshold"), VtValue(100.f), false}};
                auto cullBeforeCut = cutDesc;
                cullBeforeCut.nodes[1].inputs = {leadingCull.path};
                cullBeforeCut.nodes.insert(cullBeforeCut.nodes.begin() + 1, leadingCull);
                cullBeforeCut.nodes[2].params[4].value = VtValue(0.f);
                cullProvider->topologyOrdinal = 0;
                auto emptyBeforeCut = commitChecked(cullBeforeCut, emptyCutOracle);

                if (dagMode) {
                    // Both Cut branches start from the same compacted packet.
                    // They may join only when their complete point packets
                    // agree; matching IDs and CV counts alone is insufficient.
                    auto cutDag = cutDesc;
                    auto leftCut = cutDesc.nodes[1];
                    leftCut.path = SdfPath("/SessionProvider/leftBoundedCut");
                    leftCut.inputs = {SdfPath("/SessionProvider/sharedCutCull")};
                    leftCut.params[4].value = VtValue(0.f);
                    auto leftWidth = cutDesc.nodes.back();
                    leftWidth.path = SdfPath("/SessionProvider/leftBoundedWidth");
                    leftWidth.inputs = {leftCut.path};
                    auto rightCut = leftCut;
                    rightCut.path = SdfPath("/SessionProvider/rightBoundedCut");
                    auto rightWidth = leftWidth;
                    rightWidth.path = SdfPath("/SessionProvider/rightBoundedWidth");
                    rightWidth.inputs = {rightCut.path};
                    auto sharedCull = leadingCull;
                    sharedCull.path = SdfPath("/SessionProvider/sharedCutCull");
                    sharedCull.params.back().value = VtValue(.25f);
                    UsdGenNodeDesc cutJoin;
                    cutJoin.path = SdfPath("/SessionProvider/boundedCutJoin");
                    cutJoin.type = TfToken("UsdGenWidthBlend");
                    cutJoin.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
                    cutJoin.inputs = {leftWidth.path, rightWidth.path};
                    cutDag.nodes = {cutDesc.nodes[0], sharedCull, leftCut, leftWidth,
                                    rightCut, rightWidth, cutJoin};
                    cutDag.terminal = cutJoin.path;
                    cullProvider->topologyOrdinal = 0;
                    auto equalCutJoin = commitChecked(cutDag, cutOracle);
                    auto unequalCutJoin = cutDag;
                    unequalCutJoin.nodes[4].params[0].value = VtValue(2.f);
                    session->SetGraphDesc(unequalCutJoin);
                    auto rejectedCutJoin = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                    CHECK(rejectedCutJoin && rejectedCutJoin->diagnostics.HasErrors() &&
                          rejectedCutJoin->generation == equalCutJoin->generation && session->NeedsCommit());
                    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                                *fixture.queueOwner, equalCutJoin->generation->device, cutOracle);
                    auto equalCutRetry = commitChecked(cutDag, cutOracle);
                    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                                *fixture.queueOwner, equalCutRetry->generation->device, cutOracle);
                    fixture.queueOwner->InvokeOwner([&] {
                        equalCutJoin.reset(); rejectedCutJoin.reset(); equalCutRetry.reset();
                    });
                }

                // Cut/extend decides whether a strand is degenerate from the
                // actual float points.  This tiny arc has no usable segment,
                // so Set leaves it intact and the following threshold culls it.
                auto tinyCut = cutDesc;
                auto& tinyCurves = tinyCut.curveSets.front();
                tinyCurves.points = {{0,0,0},{1e-13f,0,0},{2e-13f,0,0}};
                tinyCurves.rest = tinyCurves.points;
                tinyCurves.widths = {1,2,3};
                tinyCut.nodes[1].params[0].value = VtValue(1.f);
                tinyCut.nodes[1].params[4].value = VtValue(.5f);
                cullProvider->topologyOrdinal = 1;
                auto tinyCutSnapshot = commitChecked(tinyCut, emptyCutOracle);
                cullProvider->topologyOrdinal = SIZE_MAX;
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, emptyBeforeCut->generation->device, emptyCutOracle);
                fixture.queueOwner->InvokeOwner([&] {
                    cutSnapshot.reset(); extendSnapshot.reset(); rejectedZero.reset(); zeroSnapshot.reset();
                    emptyBeforeCut.reset(); tinyCutSnapshot.reset();
                });
            }
            if (reparamShaderPath) {
                auto reparamDesc = rootsMode
                    ? MakeProviderRootDescriptor(true, true, TfToken("onError"))
                    : MakeProviderDescriptor(true);
                auto& curves = reparamDesc.curveSets.front();
                curves.curveVertexCounts = {3}; curves.curveId = {20};
                curves.points = {{0,0,0},{1,0,0},{1,3,0}}; curves.rest = curves.points;
                curves.widths = {1,2,3};
                if (rootsMode) {
                    curves.skinPrim = {0}; curves.skinPrimUv = {{.25f,.25f}};
                    curves.rootFrame = {GfMatrix4d(1.0)};
                    curves.authoredPlanes.clear();
                    AddRootAuthoredPlanes(&curves);
                }
                reparamDesc.nodes[1].params = {
                    {TfToken("length:value"), VtValue(4.f), false},
                    {TfToken("length:mode"), VtValue(TfToken("set")), false},
                    {TfToken("length:method"), VtValue(TfToken("cutExtend")), false},
                    {TfToken("rebuild"), VtValue(TfToken("reparam")), false},
                    {TfToken("cullThreshold"), VtValue(.25f), false}};
                auto neutral = reparamDesc;
                neutral.nodes[1].params[0].value = VtValue(1.f);
                neutral.nodes[1].params[1].value = VtValue(TfToken("scale"));
                neutral.nodes[1].params[2].value = VtValue(TfToken("scale"));
                neutral.nodes[1].params[3].value = VtValue(TfToken("keepParam"));
                neutral.nodes[1].params[4].value = VtValue(0.f);
                SixPlaneOracle expected;
                CHECK(BuildCpuReference(neutral, &expected));
                // Division by three in arc interpolation can differ by an
                // ULP from the exact mathematical point. Only these newly
                // computed points allow four ULPs; all retained data is exact.
                expected.computedPointUlps = 4;
                expected.curveOffsets = {0,3};
                expected.points = {{0,0,0},{1,1,0},{1,3,0}};

                SixPlaneOracle emptyExpected = expected;
                emptyExpected.curveOffsets = {0}; emptyExpected.points.clear();
                emptyExpected.rest.clear(); emptyExpected.width.clear();
                emptyExpected.hairT.clear(); emptyExpected.stableIds.clear();
                emptyExpected.rootPrim.clear(); emptyExpected.rootUV.clear();
                for (auto& named : emptyExpected.named) {
                    if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                        named.bytes.clear(); named.metadata.elementCount = 0;
                    }
                }
                cullProvider->topologyOrdinal = 1;
                auto reparamSnapshot = commitChecked(reparamDesc, expected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, reparamSnapshot->generation->device, expected);
                // The leading Cull owns topology. Reparam then computes only
                // the middle CV; roots, IDs, widths, and named planes are
                // retained exactly from the surviving source packet.
                UsdGenNodeDesc pureCull;
                pureCull.path = SdfPath("/SessionProvider/beforeReparamCull");
                pureCull.type = TfToken("UsdGenLength");
                pureCull.inputs = {reparamDesc.nodes[0].path};
                pureCull.params = {
                    {TfToken("length:value"), VtValue(1.f), false},
                    {TfToken("length:mode"), VtValue(TfToken("cull")), false},
                    {TfToken("cullThreshold"), VtValue(.5f), false}};
                auto cullThenReparam = reparamDesc;
                cullThenReparam.nodes[1].inputs = {pureCull.path};
                cullThenReparam.nodes[1].params[4].value = VtValue(0.f);
                cullThenReparam.nodes.insert(cullThenReparam.nodes.begin() + 1, pureCull);
                cullProvider->topologyOrdinal = 0;
                auto cullReparamSnapshot = commitChecked(cullThenReparam, expected);
                auto emptyCullThenReparam = cullThenReparam;
                emptyCullThenReparam.nodes[1].params[2].value = VtValue(100.f);
                cullProvider->topologyOrdinal = 0;
                auto emptyCullReparamSnapshot = commitChecked(emptyCullThenReparam, emptyExpected);
                auto highThreshold = reparamDesc;
                highThreshold.nodes[1].params[4].value = VtValue(3.75f);
                cullProvider->topologyOrdinal = 1;
                auto highEmpty = expected;
                highEmpty.curveOffsets = {0}; highEmpty.points.clear(); highEmpty.rest.clear();
                highEmpty.width.clear(); highEmpty.hairT.clear(); highEmpty.stableIds.clear();
                highEmpty.rootPrim.clear(); highEmpty.rootUV.clear();
                for (auto& named : highEmpty.named) if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                auto highSnapshot = commitChecked(highThreshold, highEmpty);
                auto restoredReparam = commitChecked(reparamDesc, expected);
                auto zeroCurrent = reparamDesc;
                zeroCurrent.curveSets.front().points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroCurrent.curveSets.front().rest = zeroCurrent.curveSets.front().points;
                session->SetGraphDesc(zeroCurrent);
                auto failedZero = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(failedZero && failedZero->diagnostics.HasErrors() &&
                      failedZero->generation == restoredReparam->generation && session->NeedsCommit());
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, failedZero->generation->device, expected);
                auto zeroTarget = zeroCurrent;
                zeroTarget.nodes[1].params[0].value = VtValue(0.f);
                auto emptyRetry = commitChecked(zeroTarget, emptyExpected);
                CHECK(emptyRetry && emptyRetry->generation);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, reparamSnapshot->generation->device, expected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, cullReparamSnapshot->generation->device, expected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, emptyCullReparamSnapshot->generation->device, emptyExpected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, highSnapshot->generation->device, highEmpty);
                fixture.queueOwner->InvokeOwner([&] {
                    reparamSnapshot.reset(); highSnapshot.reset(); emptyRetry.reset(); failedZero.reset();
                    restoredReparam.reset(); cullReparamSnapshot.reset(); emptyCullReparamSnapshot.reset();
                });
                cullProvider->topologyOrdinal = SIZE_MAX;
            }
            if (minimumShaderPath) {
                auto minimumDesc = rootsMode
                    ? MakeProviderRootDescriptor(true, true, TfToken("never"))
                    : MakeProviderDescriptor(true);
                auto& minimumCurves = minimumDesc.curveSets.front();
                minimumCurves.curveVertexCounts = {3}; minimumCurves.curveId = {20};
                minimumCurves.points = {{0,0,0},{1,0,0},{1,3,0}};
                minimumCurves.rest = minimumCurves.points; minimumCurves.widths = {1,2,3};
                if (rootsMode) {
                    minimumCurves.skinPrim = {0}; minimumCurves.skinPrimUv = {{.25f,.25f}};
                    minimumCurves.rootFrame = {GfMatrix4d(1.0)};
                    minimumCurves.authoredPlanes.clear(); AddRootAuthoredPlanes(&minimumCurves);
                }
                auto& minimumNode = minimumDesc.nodes[1];
                minimumNode.params = {
                    {TfToken("length:value"), VtValue(.25f), false},
                    {TfToken("length:mode"), VtValue(TfToken("scale")), false},
                    {TfToken("length:method"), VtValue(TfToken("scale")), false},
                    {TfToken("rebuild"), VtValue(TfToken("keepParam")), false},
                    {TfToken("minRemainingLength"), VtValue(3.f), false},
                    {TfToken("cullThreshold"), VtValue(.25f), false}};
                // CPU's old minimum path applies its legacy threshold rule.
                // Capture only neutral Source/Width data, then state the
                // minimum shader's literal geometry by hand.
                auto neutralMinimum = minimumDesc;
                neutralMinimum.nodes[1].params[0].value = VtValue(1.f);
                neutralMinimum.nodes[1].params[4].value = VtValue(0.f);
                neutralMinimum.nodes[1].params[5].value = VtValue(0.f);
                SixPlaneOracle radial;
                CHECK(BuildCpuReference(neutralMinimum, &radial));
                radial.curveOffsets = {0,3};
                radial.points = {{0,0,0},{.75f,0,0},{.75f,2.25f,0}};
                radial.computedPointUlps = 4;
                if (rootsMode) {
                    radial.rootPrim = {0}; radial.rootUV = {{.25f,.25f}};
                    CHECK(radial.named.size() >= 6);
                }
                auto emptyMinimum = radial;
                emptyMinimum.points.clear(); emptyMinimum.rest.clear(); emptyMinimum.width.clear();
                emptyMinimum.hairT.clear(); emptyMinimum.stableIds.clear();
                emptyMinimum.rootPrim.clear(); emptyMinimum.rootUV.clear(); emptyMinimum.curveOffsets = {0};
                for (auto& named : emptyMinimum.named) if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                cullProvider->topologyOrdinal = 1;
                auto radialScale = commitChecked(minimumDesc, radial);
                auto scaleKeepDesc = minimumDesc;
                scaleKeepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle scaleKeep = radial;
                scaleKeep.points = {{0,0,0},{1,0,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto scaleKeepSnapshot = commitChecked(scaleKeepDesc, scaleKeep);
                auto scaleReparamDesc = scaleKeepDesc;
                scaleReparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle scaleReparam = radial;
                scaleReparam.points = {{0,0,0},{1,.5f,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto scaleReparamSnapshot = commitChecked(scaleReparamDesc, scaleReparam);
                auto radialSetDesc = minimumDesc;
                radialSetDesc.nodes[1].params[0].value = VtValue(1.f);
                radialSetDesc.nodes[1].params[1].value = VtValue(TfToken("set"));
                cullProvider->topologyOrdinal = 1;
                auto radialSet = commitChecked(radialSetDesc, radial);
                auto keepDesc = radialSetDesc;
                keepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle keep = radial;
                keep.points = {{0,0,0},{1,0,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto keepSnapshot = commitChecked(keepDesc, keep);
                auto reparamDesc = keepDesc;
                reparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle reparam = radial;
                reparam.points = {{0,0,0},{1,.5f,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto reparamSnapshot = commitChecked(reparamDesc, reparam);
                // Reparam's actual arc is sqrt(1.25)+1.5, below 2.75.
                auto reparamCull = reparamDesc;
                reparamCull.nodes[1].params[5].value = VtValue(2.75f);
                cullProvider->topologyOrdinal = 1;
                auto reparamEmpty = commitChecked(reparamCull, emptyMinimum);
                auto sameCurrent = reparamDesc;
                sameCurrent.nodes[1].params[0].value = VtValue(0.f);
                sameCurrent.nodes[1].params[4].value = VtValue(4.f);
                sameCurrent.nodes[1].params[5].value = VtValue(.25f);
                SixPlaneOracle same = radial;
                same.points = {{0,0,0},{1,1,0},{1,3,0}};
                cullProvider->topologyOrdinal = 1;
                auto sameSnapshot = commitChecked(sameCurrent, same);
                UsdGenNodeDesc pureCull;
                pureCull.path = SdfPath("/SessionProvider/beforeMinimumCull");
                pureCull.type = TfToken("UsdGenLength"); pureCull.inputs = {minimumDesc.nodes[0].path};
                pureCull.params = {{TfToken("length:value"), VtValue(1.f), false},
                                   {TfToken("length:mode"), VtValue(TfToken("cull")), false},
                                   {TfToken("minRemainingLength"), VtValue(1000.f), false},
                                   {TfToken("cullThreshold"), VtValue(.5f), false}};
                auto cullThenMinimum = minimumDesc;
                cullThenMinimum.nodes[1].inputs = {pureCull.path};
                cullThenMinimum.nodes[1].params[5].value = VtValue(0.f);
                cullThenMinimum.nodes.insert(cullThenMinimum.nodes.begin() + 1, pureCull);
                cullProvider->topologyOrdinal = 0;
                auto cullMinimum = commitChecked(cullThenMinimum, radial);
                auto emptyCullThenMinimum = cullThenMinimum;
                emptyCullThenMinimum.nodes[1].params[3].value = VtValue(100.f);
                cullProvider->topologyOrdinal = 0;
                auto emptyCullMinimum = commitChecked(emptyCullThenMinimum, emptyMinimum);
                cullProvider->topologyOrdinal = 1;
                auto restoredSame = commitChecked(sameCurrent, same);
                auto zeroCurrent = minimumDesc;
                zeroCurrent.curveSets.front().points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroCurrent.curveSets.front().rest = zeroCurrent.curveSets.front().points;
                session->SetGraphDesc(zeroCurrent);
                auto failedZero = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(failedZero && failedZero->diagnostics.HasErrors() &&
                      failedZero->generation == restoredSame->generation && session->NeedsCommit());
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, failedZero->generation->device, same);
                auto zeroRetry = zeroCurrent;
                zeroRetry.nodes[1].params[0].value = VtValue(0.f);
                zeroRetry.nodes[1].params[4].value = VtValue(0.f);
                cullProvider->topologyOrdinal = 1;
                auto zeroEmpty = commitChecked(zeroRetry, emptyMinimum);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, radialScale->generation->device, radial);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, scaleKeepSnapshot->generation->device, scaleKeep);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, scaleReparamSnapshot->generation->device, scaleReparam);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, restoredSame->generation->device, same);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, cullMinimum->generation->device, radial);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, emptyCullMinimum->generation->device, emptyMinimum);
                fixture.queueOwner->InvokeOwner([&] {
                    radialScale.reset(); scaleKeepSnapshot.reset(); scaleReparamSnapshot.reset();
                    radialSet.reset(); keepSnapshot.reset(); reparamSnapshot.reset();
                    reparamEmpty.reset(); sameSnapshot.reset(); cullMinimum.reset(); emptyCullMinimum.reset();
                    restoredSame.reset(); failedZero.reset(); zeroEmpty.reset();
                });
                cullProvider->topologyOrdinal = SIZE_MAX;
            }
            if (literalV1ShaderPath) {
                auto literalDesc = rootsMode
                    ? MakeProviderRootDescriptor(true, true, TfToken("never"))
                    : MakeProviderDescriptor(true);
                auto& literalCurves = literalDesc.curveSets.front();
                literalCurves.curveVertexCounts = {3}; literalCurves.curveId = {0x100000007ull};
                literalCurves.points = {{0,0,0},{1,0,0},{1,3,0}};
                literalCurves.rest = literalCurves.points; literalCurves.widths = {1,2,3};
                if (rootsMode) {
                    literalCurves.skinPrim = {0}; literalCurves.skinPrimUv = {{.25f,.25f}};
                    literalCurves.rootFrame = {GfMatrix4d(1.0)};
                    literalCurves.authoredPlanes.clear(); AddRootAuthoredPlanes(&literalCurves);
                }
                auto& literalNode = literalDesc.nodes[1]; literalNode.seed = -17;
                literalNode.params = {{TfToken("length:value"), VtValue(1.f), false},
                                      {TfToken("length:mode"), VtValue(TfToken("scale")), false},
                                      {TfToken("length:method"), VtValue(TfToken("scale")), false},
                                      {TfToken("rebuild"), VtValue(TfToken("keepParam")), false},
                                      {TfToken("minRemainingLength"), VtValue(0.f), false},
                                      {TfToken("length:random"), VtValue(GfVec2f(.75f,.75f)), false},
                                      {TfToken("cullThreshold"), VtValue(.25f), false}};
                auto neutralLiteral = literalDesc;
                neutralLiteral.nodes[1].params[5].value = VtValue(GfVec2f(1.f,1.f));
                neutralLiteral.nodes[1].params[6].value = VtValue(0.f);
                SixPlaneOracle literalRadial; CHECK(BuildCpuReference(neutralLiteral, &literalRadial));
                literalRadial.curveOffsets = {0,3};
                literalRadial.points = {{0,0,0},{.75f,0,0},{.75f,2.25f,0}};
                literalRadial.computedPointUlps = 4;
                if (rootsMode) { literalRadial.rootPrim = {0}; literalRadial.rootUV = {{.25f,.25f}};
                                 CHECK(literalRadial.named.size() >= 6); }
                auto literalEmpty = literalRadial;
                literalEmpty.points.clear(); literalEmpty.rest.clear(); literalEmpty.width.clear();
                literalEmpty.hairT.clear(); literalEmpty.stableIds.clear(); literalEmpty.rootPrim.clear();
                literalEmpty.rootUV.clear(); literalEmpty.curveOffsets = {0};
                for (auto& named : literalEmpty.named) if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                cullProvider->topologyOrdinal = 1;
                auto literalScale = commitChecked(literalDesc, literalRadial);
                auto literalScaleKeepDesc = literalDesc;
                literalScaleKeepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle literalScaleKeep = literalRadial;
                literalScaleKeep.points = {{0,0,0},{1,0,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto literalScaleKeepSnapshot = commitChecked(literalScaleKeepDesc, literalScaleKeep);
                auto literalScaleReparamDesc = literalScaleKeepDesc;
                literalScaleReparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle literalScaleReparam = literalRadial;
                literalScaleReparam.points = {{0,0,0},{1,.5f,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto literalScaleReparamSnapshot = commitChecked(literalScaleReparamDesc, literalScaleReparam);
                auto literalSetDesc = literalDesc;
                literalSetDesc.nodes[1].params[0].value = VtValue(4.f);
                literalSetDesc.nodes[1].params[1].value = VtValue(TfToken("set"));
                cullProvider->topologyOrdinal = 1;
                auto literalSet = commitChecked(literalSetDesc, literalRadial);
                auto literalKeepDesc = literalSetDesc;
                literalKeepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle literalKeep = literalRadial;
                literalKeep.points = {{0,0,0},{1,0,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto literalKeepSnapshot = commitChecked(literalKeepDesc, literalKeep);
                auto literalReparamDesc = literalKeepDesc;
                literalReparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle literalReparam = literalRadial;
                literalReparam.points = {{0,0,0},{1,.5f,0},{1,2,0}};
                cullProvider->topologyOrdinal = 1;
                auto literalReparamSnapshot = commitChecked(literalReparamDesc, literalReparam);
                auto literalReparamCull = literalReparamDesc;
                literalReparamCull.nodes[1].params[6].value = VtValue(2.75f);
                cullProvider->topologyOrdinal = 1;
                auto literalReparamEmpty = commitChecked(literalReparamCull, literalEmpty);
                auto literalSameDesc = literalReparamDesc;
                literalSameDesc.nodes[1].params[4].value = VtValue(4.f);
                SixPlaneOracle literalSameExpected = literalRadial;
                literalSameExpected.points = {{0,0,0},{1,1,0},{1,3,0}};
                cullProvider->topologyOrdinal = 1;
                auto literalSameSnapshot = commitChecked(literalSameDesc, literalSameExpected);
                // A cull may compact curve order, never the ID used by the
                // LiteralV1 draw.  The surviving unit curve is keyed by ID.
                auto randomDesc = literalDesc;
                auto& randomCurves = randomDesc.curveSets.front();
                uint64_t const shortId = 1ull, liveId = 0x100000007ull;
                randomCurves.curveVertexCounts = {2,2}; randomCurves.curveId = {shortId, liveId};
                randomCurves.points = {{0,0,0},{.1f,0,0},{0,0,0},{1,0,0}};
                randomCurves.rest = randomCurves.points; randomCurves.widths = {1,2,3,4};
                if (rootsMode) { randomCurves.skinPrim = {0,0}; randomCurves.skinPrimUv = {{.25f,.25f},{.5f,.25f}};
                                 randomCurves.rootFrame = {GfMatrix4d(1.0), GfMatrix4d(1.0)};
                                 randomCurves.authoredPlanes.clear(); AddRootAuthoredPlanes(&randomCurves); }
                randomDesc.nodes[1].params[5].value = VtValue(GfVec2f(0.f,1.f));
                UsdGenNodeDesc literalCull;
                literalCull.path = SdfPath("/SessionProvider/beforeLiteralCull"); literalCull.type = TfToken("UsdGenLength");
                literalCull.inputs = {randomDesc.nodes[0].path};
                literalCull.params = {{TfToken("length:value"), VtValue(1.f), false},
                                      {TfToken("length:mode"), VtValue(TfToken("cull")), false},
                                      {TfToken("length:random"), VtValue(GfVec2f(0.f,1.f)), false},
                                      {TfToken("cullThreshold"), VtValue(.5f), false}};
                randomDesc.nodes[1].inputs = {literalCull.path}; randomDesc.nodes[1].params[6].value = VtValue(0.f);
                randomDesc.nodes.insert(randomDesc.nodes.begin() + 1, literalCull);
                // CPU Cull does not gather every retained plane. Capture the
                // neutral full packet, then independently gather the known
                // canonical survivor (source curve 1, CVs 2 and 3).
                SixPlaneOracle randomExpected;
                auto randomNeutral = randomDesc;
                randomNeutral.nodes.erase(randomNeutral.nodes.begin() + 1);
                randomNeutral.nodes[1].inputs = {randomNeutral.nodes[0].path};
                randomNeutral.nodes[1].params[5].value = VtValue(GfVec2f(1.f,1.f));
                randomNeutral.nodes[1].params[6].value = VtValue(0.f);
                CHECK(BuildCpuReference(randomNeutral, &randomExpected));
                // Uniform CPU curves may use implicit offsets; the authored
                // two-CV counts above define the equivalent explicit topology.
                if (randomExpected.curveOffsets.empty()) randomExpected.curveOffsets = {0,2,4};
                CHECK(randomExpected.curveOffsets == std::vector<uint32_t>({0,2,4}) &&
                      randomExpected.points.size() == 4 && randomExpected.rest.size() == 4 &&
                      randomExpected.width.size() == 4 && randomExpected.hairT.size() == 4 &&
                      randomExpected.stableIds.size() == 2);
                if (rootsMode) CHECK(randomExpected.rootPrim.size() == 2 &&
                                     randomExpected.rootUV.size() == 2 &&
                                     randomExpected.named.size() >= 6);
                auto gather = [](auto* values, std::initializer_list<uint32_t> indices) {
                    auto before = std::move(*values); values->clear();
                    if (before.empty()) return;
                    for (auto index : indices) values->push_back(before[index]);
                };
                gather(&randomExpected.points, {2,3}); gather(&randomExpected.rest, {2,3});
                gather(&randomExpected.width, {2,3}); gather(&randomExpected.hairT, {2,3});
                gather(&randomExpected.stableIds, {1}); gather(&randomExpected.rootPrim, {1});
                gather(&randomExpected.rootUV, {1});
                for (auto& named : randomExpected.named) {
                    auto before = std::move(named.bytes);
                    size_t const stride = named.metadata.strideBytes;
                    CHECK(stride && before.size() == stride * named.metadata.elementCount);
                    if (named.metadata.domain == UsdGenDeviceDomain::Point) {
                        CHECK(named.metadata.elementCount == 4);
                        named.bytes.assign(before.begin() + 2 * stride, before.begin() + 4 * stride);
                        named.metadata.elementCount = 2;
                    } else if (named.metadata.domain == UsdGenDeviceDomain::Primitive) {
                        CHECK(named.metadata.elementCount == 2);
                        named.bytes.assign(before.begin() + stride, before.begin() + 2 * stride);
                        named.metadata.elementCount = 1;
                    } else {
                        CHECK(named.metadata.domain == UsdGenDeviceDomain::Groom &&
                              named.metadata.elementCount == 1);
                        named.bytes = std::move(before);
                    }
                }
                float const draw = UsdGenDraw01(-17, liveId, kSaltLength);
                randomExpected.points = {{0,0,0},{draw,0,0}};
                randomExpected.curveOffsets = {0,2}; randomExpected.computedPointUlps = 0;

                cullProvider->topologyOrdinal = 0;
                auto randomSnapshot = commitChecked(randomDesc, randomExpected);
                auto emptyRandom = randomDesc;
                emptyRandom.nodes[1].params[3].value = VtValue(100.f);
                auto emptyRandomExpected = randomExpected;
                emptyRandomExpected.points.clear(); emptyRandomExpected.rest.clear(); emptyRandomExpected.width.clear();
                emptyRandomExpected.hairT.clear(); emptyRandomExpected.stableIds.clear();
                emptyRandomExpected.rootPrim.clear(); emptyRandomExpected.rootUV.clear(); emptyRandomExpected.curveOffsets = {0};
                for (auto& named : emptyRandomExpected.named) if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                cullProvider->topologyOrdinal = 0;
                auto emptyRandomSnapshot = commitChecked(emptyRandom, emptyRandomExpected);
                cullProvider->topologyOrdinal = 1;
                auto restoredLiteral = commitChecked(literalSameDesc, literalSameExpected);
                auto zeroLiteral = literalDesc;
                zeroLiteral.curveSets.front().points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroLiteral.curveSets.front().rest = zeroLiteral.curveSets.front().points;
                zeroLiteral.nodes[1].params[4].value = VtValue(1.f);
                session->SetGraphDesc(zeroLiteral);
                auto failedLiteral = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(failedLiteral && failedLiteral->diagnostics.HasErrors() &&
                      failedLiteral->generation == restoredLiteral->generation && session->NeedsCommit());
                auto zeroRetry = zeroLiteral; zeroRetry.nodes[1].params[0].value = VtValue(0.f);
                zeroRetry.nodes[1].params[4].value = VtValue(0.f);
                zeroRetry.nodes[1].params[5].value = VtValue(GfVec2f(1.f,1.f));
                cullProvider->topologyOrdinal = 1;
                auto zeroLiteralEmpty = commitChecked(zeroRetry, literalEmpty);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, randomSnapshot->generation->device, randomExpected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, restoredLiteral->generation->device, literalSameExpected);
                fixture.queueOwner->InvokeOwner([&] {
                    literalScale.reset(); literalScaleKeepSnapshot.reset(); literalScaleReparamSnapshot.reset();
                    literalSet.reset(); literalKeepSnapshot.reset(); literalReparamSnapshot.reset();
                    literalReparamEmpty.reset(); literalSameSnapshot.reset(); randomSnapshot.reset(); emptyRandomSnapshot.reset();
                    restoredLiteral.reset(); failedLiteral.reset(); zeroLiteralEmpty.reset();
                });
                cullProvider->topologyOrdinal = SIZE_MAX;
            }
            if (envelopeV1ShaderPath) {
                auto envelopeDesc = rootsMode
                    ? MakeProviderRootDescriptor(true, true, TfToken("never"))
                    : MakeProviderDescriptor(true);
                auto& envelopeCurves = envelopeDesc.curveSets.front();
                envelopeCurves.curveVertexCounts = {3}; envelopeCurves.curveId = {0x100000007ull};
                envelopeCurves.points = {{0,0,0},{1,0,0},{1,3,0}};
                envelopeCurves.rest = envelopeCurves.points; envelopeCurves.widths = {1,2,3};
                if (rootsMode) { envelopeCurves.skinPrim = {0}; envelopeCurves.skinPrimUv = {{.25f,.25f}};
                                 envelopeCurves.rootFrame = {GfMatrix4d(1.0)};
                                 envelopeCurves.authoredPlanes.clear(); AddRootAuthoredPlanes(&envelopeCurves); }
                auto& envelopeNode = envelopeDesc.nodes[1]; envelopeNode.seed = -17;
                envelopeNode.params = {{TfToken("length:value"), VtValue(1.f), false},
                                       {TfToken("length:mode"), VtValue(TfToken("scale")), false},
                                       {TfToken("length:method"), VtValue(TfToken("scale")), false},
                                       {TfToken("rebuild"), VtValue(TfToken("keepParam")), false},
                                       {TfToken("minRemainingLength"), VtValue(0.f), false},
                                       {TfToken("length:random"), VtValue(GfVec2f(.75f,.75f)), false},
                                       {TfToken("mask"), VtValue(.5f), false},
                                       {TfToken("cullThreshold"), VtValue(.25f), false}};
                auto neutralEnvelope = envelopeDesc; neutralEnvelope.nodes[1].params[6].value = VtValue(1.f);
                neutralEnvelope.nodes[1].params[5].value = VtValue(GfVec2f(1.f,1.f));
                neutralEnvelope.nodes[1].params[7].value = VtValue(0.f);
                SixPlaneOracle envelopeRadial; CHECK(BuildCpuReference(neutralEnvelope, &envelopeRadial));
                SixPlaneOracle envelopeSource = envelopeRadial;
                envelopeSource.curveOffsets = {0,3};
                envelopeSource.computedPointUlps = 0;
                envelopeRadial.curveOffsets = {0,3};
                envelopeRadial.points = {{0,0,0},{.875f,0,0},{.875f,2.625f,0}};
                envelopeRadial.computedPointUlps = 4;
                if (rootsMode) { envelopeRadial.rootPrim = {0}; envelopeRadial.rootUV = {{.25f,.25f}};
                                 CHECK(envelopeRadial.named.size() >= 6); }
                auto envelopeEmpty = envelopeRadial;
                envelopeEmpty.points.clear(); envelopeEmpty.rest.clear(); envelopeEmpty.width.clear();
                envelopeEmpty.hairT.clear(); envelopeEmpty.stableIds.clear(); envelopeEmpty.rootPrim.clear();
                envelopeEmpty.rootUV.clear(); envelopeEmpty.curveOffsets = {0};
                for (auto& named : envelopeEmpty.named) if (named.metadata.domain != UsdGenDeviceDomain::Groom) {
                    named.bytes.clear(); named.metadata.elementCount = 0;
                }
                cullProvider->topologyOrdinal = 1;
                auto envelopeScale = commitChecked(envelopeDesc, envelopeRadial);
                auto envelopeScaleKeepDesc = envelopeDesc;
                envelopeScaleKeepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle envelopeScaleKeep = envelopeRadial;
                envelopeScaleKeep.points = {{0,0,0},{1,0,0},{1,2.5f,0}};
                cullProvider->topologyOrdinal = 1;
                auto envelopeScaleKeepSnapshot = commitChecked(envelopeScaleKeepDesc, envelopeScaleKeep);
                auto envelopeScaleReparamDesc = envelopeScaleKeepDesc;
                envelopeScaleReparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle envelopeScaleReparam = envelopeRadial;
                envelopeScaleReparam.points = {{0,0,0},{1,.25f,0},{1,2.5f,0}};
                cullProvider->topologyOrdinal = 1;
                auto envelopeScaleReparamSnapshot = commitChecked(envelopeScaleReparamDesc, envelopeScaleReparam);
                auto envelopeSet = envelopeDesc; envelopeSet.nodes[1].params[0].value = VtValue(4.f);
                envelopeSet.nodes[1].params[1].value = VtValue(TfToken("set"));
                cullProvider->topologyOrdinal = 1;
                auto envelopeSetSnapshot = commitChecked(envelopeSet, envelopeRadial);
                auto envelopeKeepDesc = envelopeSet; envelopeKeepDesc.nodes[1].params[2].value = VtValue(TfToken("cutExtend"));
                SixPlaneOracle envelopeKeepExpected = envelopeRadial;
                envelopeKeepExpected.points = {{0,0,0},{1,0,0},{1,2.5f,0}};
                cullProvider->topologyOrdinal = 1;
                auto envelopeKeepSnapshot = commitChecked(envelopeKeepDesc, envelopeKeepExpected);
                auto envelopeReparamDesc = envelopeKeepDesc; envelopeReparamDesc.nodes[1].params[3].value = VtValue(TfToken("reparam"));
                SixPlaneOracle envelopeReparamExpected = envelopeRadial;
                envelopeReparamExpected.points = {{0,0,0},{1,.25f,0},{1,2.5f,0}};
                cullProvider->topologyOrdinal = 1;
                auto envelopeReparamSnapshot = commitChecked(envelopeReparamDesc, envelopeReparamExpected);
                auto envelopeKeepThreshold = envelopeReparamDesc;
                envelopeKeepThreshold.nodes[1].params[7].value = VtValue(3.2f);
                cullProvider->topologyOrdinal = 1;
                auto envelopeKeeps = commitChecked(envelopeKeepThreshold, envelopeReparamExpected);
                auto envelopeCull = envelopeReparamDesc;
                envelopeCull.nodes[1].params[7].value = VtValue(3.4f);
                cullProvider->topologyOrdinal = 1;
                auto envelopeCulls = commitChecked(envelopeCull, envelopeEmpty);
                // mask=.5 resolves to the same .5 envelope.
                auto composedEnvelope = envelopeDesc;
                composedEnvelope.nodes[1].params[6].value = VtValue(.5f);
                cullProvider->topologyOrdinal = 1;
                auto composedSnapshot = commitChecked(composedEnvelope, envelopeRadial);
                // Pure Cull still validates its source packet, but an exact
                // zero envelope suppresses its geometric threshold decision.
                auto pureEnvelopeCull = envelopeDesc;
                pureEnvelopeCull.nodes[1].params[0].value = VtValue(100.f);
                pureEnvelopeCull.nodes[1].params[1].value = VtValue(TfToken("cull"));
                pureEnvelopeCull.nodes[1].params[4].value = VtValue(100.f);
                pureEnvelopeCull.nodes[1].params[7].value = VtValue(100.f);
                pureEnvelopeCull.nodes[1].params[6].value = VtValue(0.f);
                cullProvider->topologyOrdinal = 1;
                auto pureEnvelopeKeeps = commitChecked(pureEnvelopeCull, envelopeSource);
                auto partialEnvelopeCull = pureEnvelopeCull;
                partialEnvelopeCull.nodes[1].params[6].value = VtValue(.5f);
                cullProvider->topologyOrdinal = 1;
                auto partialEnvelopeEmpty = commitChecked(partialEnvelopeCull, envelopeEmpty);
                auto restoreEnvelope = commitChecked(envelopeReparamDesc, envelopeReparamExpected);
                auto zeroEnvelope = envelopeReparamDesc;
                zeroEnvelope.curveSets.front().points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroEnvelope.curveSets.front().rest = zeroEnvelope.curveSets.front().points;
                zeroEnvelope.nodes[1].params[4].value = VtValue(1.f);
                cullProvider->topologyOrdinal = 1;
                session->SetGraphDesc(zeroEnvelope);
                auto failedEnvelope = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(failedEnvelope && failedEnvelope->diagnostics.HasErrors() &&
                      failedEnvelope->generation == restoreEnvelope->generation && session->NeedsCommit());
                auto zeroEnvelopeRetry = zeroEnvelope;
                zeroEnvelopeRetry.nodes[1].params[7].value = VtValue(100.f);
                zeroEnvelopeRetry.nodes[1].params[6].value = VtValue(0.f);
                SixPlaneOracle zeroEnvelopeExpected = envelopeReparamExpected;
                zeroEnvelopeExpected.points = {{0,0,0},{0,0,0},{0,0,0}};
                zeroEnvelopeExpected.rest = zeroEnvelopeExpected.points;
                zeroEnvelopeExpected.computedPointUlps = 0;
                cullProvider->topologyOrdinal = 1;
                auto zeroEnvelopeSnapshot = commitChecked(zeroEnvelopeRetry, zeroEnvelopeExpected);
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, restoreEnvelope->generation->device, envelopeReparamExpected);
                fixture.queueOwner->InvokeOwner([&] {
                    envelopeScale.reset(); envelopeScaleKeepSnapshot.reset(); envelopeScaleReparamSnapshot.reset();
                    envelopeSetSnapshot.reset(); envelopeKeepSnapshot.reset();
                    envelopeReparamSnapshot.reset(); envelopeKeeps.reset(); envelopeCulls.reset();
                    composedSnapshot.reset(); pureEnvelopeKeeps.reset(); partialEnvelopeEmpty.reset();
                    restoreEnvelope.reset(); failedEnvelope.reset(); zeroEnvelopeSnapshot.reset();
                });
                cullProvider->topologyOrdinal = SIZE_MAX;
            }
            auto setPoints = [](SixPlaneOracle* packet, float target) {
                for (size_t c = 0; c != packet->stableIds.size(); ++c) {
                    auto begin = packet->curveOffsets[c], end = packet->curveOffsets[c+1];
                    float current = 0;
                    for (auto p = begin + 1; p < end; ++p)
                        current += (packet->points[p] - packet->points[p-1]).GetLength();
                    CHECK(current > 0);
                    auto const root = packet->points[begin];
                    for (auto p = begin; p < end; ++p)
                        packet->points[p] = root + (packet->points[p] - root) * (target/current);
                }
            };
            UsdGenNodeDesc set;
            set.path = SdfPath("/SessionProvider/mixedSet"); set.type = TfToken("UsdGenLength");
            set.params = {{TfToken("length:mode"), VtValue(TfToken("set")), false},
                          {TfToken("length:value"), VtValue(2.f), false}};
            auto setThenCull = descriptor;
            set.inputs = {descriptor.nodes[0].path};
            setThenCull.nodes[1].inputs = {set.path};
            setThenCull.nodes.insert(setThenCull.nodes.begin()+1, set);
            SixPlaneOracle extended = bothOracle; setPoints(&extended, 2.f);
            // Both now have length 2; a threshold of 1.5 must keep even the
            // originally short strand. All non-point planes retain source data.
            cullProvider->topologyOrdinal = 1;
            auto extendedSnapshot = commitChecked(setThenCull, extended);
            auto shortenedDesc = setThenCull;
            shortenedDesc.nodes[1].params[1].value = VtValue(.5f);
            auto shortenedSnapshot = commitChecked(shortenedDesc, emptyOracle);
            // Reversing the order filters the original geometry first, then
            // sets only the surviving strand. Its topology belongs to Cull.
            auto cullThenSet = descriptor;
            set.inputs = {descriptor.nodes[1].path}; set.params[1].value = VtValue(.5f);
            cullThenSet.nodes.back().inputs = {set.path};
            cullThenSet.nodes.insert(cullThenSet.nodes.end()-1, set);
            SixPlaneOracle filteredSet = firstOracle; setPoints(&filteredSet, .5f);
            cullProvider->topologyOrdinal = 0;
            auto filteredSnapshot = commitChecked(cullThenSet, filteredSet);
            // Empty compaction is also a valid input to a subsequent Set.
            auto emptyThenSet = cullThenSet;
            emptyThenSet.nodes[1].params.back().value = VtValue(3.f);
            auto emptySetSnapshot = commitChecked(emptyThenSet, emptyOracle);
            auto twice = setThenCull;
            set.path = SdfPath("/SessionProvider/secondMixedSet");
            set.inputs = {twice.nodes[2].path}; set.params[1].value = VtValue(1.f);
            twice.nodes.back().inputs = {set.path};
            twice.nodes.insert(twice.nodes.end()-1, set);
            SixPlaneOracle twiceOracle = extended; setPoints(&twiceOracle, 1.f);
            cullProvider->topologyOrdinal = 1;
            auto twiceSnapshot = commitChecked(twice, twiceOracle);
            // One authored Length with a threshold lowers to the same two
            // explicit immutable stages as the manually composed graph.
            auto compoundSet = descriptor;
            compoundSet.nodes[1].params[0].value = VtValue(2.f);
            compoundSet.nodes[1].params[1].value = VtValue(TfToken("set"));
            auto compoundSetSnapshot = commitChecked(compoundSet, extended);
            auto compoundScale = descriptor;
            compoundScale.nodes[1].params[0].value = VtValue(.5f);
            compoundScale.nodes[1].params[1].value = VtValue(TfToken("scale"));
            compoundScale.nodes[1].params.back().value = VtValue(.75f);
            SixPlaneOracle scaledFiltered = firstOracle; setPoints(&scaledFiltered, 1.f);
            auto compoundScaleSnapshot = commitChecked(compoundScale, scaledFiltered);
            if (dagMode) {
                auto compoundDag = compoundScale;
                auto rightLength = compoundScale.nodes[1];
                rightLength.path = SdfPath("/SessionProvider/rightCompoundLength");
                rightLength.params.back().value = VtValue(.9f);
                auto rightWidth = compoundScale.nodes.back();
                rightWidth.path = SdfPath("/SessionProvider/rightCompoundWidth");
                rightWidth.inputs = {rightLength.path};
                UsdGenNodeDesc join;
                join.path = SdfPath("/SessionProvider/compoundJoin");
                join.type = TfToken("UsdGenWidthBlend");
            join.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
                join.inputs = {compoundScale.nodes.back().path, rightWidth.path};
                compoundDag.nodes.push_back(rightLength); compoundDag.nodes.push_back(rightWidth);
                compoundDag.nodes.push_back(join); compoundDag.terminal = join.path;
                auto joined = commitChecked(compoundDag, scaledFiltered);
                // Matching survivor IDs/counts is insufficient: the right
                // branch retains the same strand but different point values.
                auto mismatch = compoundDag;
                mismatch.nodes[3].params[0].value = VtValue(.25f);
                mismatch.nodes[3].params.back().value = VtValue(.4f);
                session->SetGraphDesc(mismatch);
                auto rejected = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
                CHECK(rejected && rejected->diagnostics.HasErrors() &&
                      rejected->generation == joined->generation && session->NeedsCommit());
                CheckPlanes(fixture.native, fixture.context, fixture.completion,
                            *fixture.queueOwner, joined->generation->device, scaledFiltered);
                fixture.queueOwner->InvokeOwner([&] { joined.reset(); rejected.reset(); });
            }
            auto compoundEmpty = compoundSet;
            compoundEmpty.nodes[1].params[0].value = VtValue(0.f);
            compoundEmpty.nodes[1].params.back().value = VtValue(.25f);
            auto compoundEmptySnapshot = commitChecked(compoundEmpty, emptyOracle);
            // At 2^24, rounding after the .5 scale produces arc length 4,
            // although source length6 * factor.5 is3. Threshold3.5 must keep
            // the strand; multiplying lengths on the host would cull it.
            auto rounded = MakeProviderDescriptor(true);
            auto& roundedSource = rounded.curveSets.front();
            roundedSource.curveVertexCounts = {4}; roundedSource.curveId = {999};
            roundedSource.points = {{16777216.f,0,0},{16777218.f,0,0},
                                    {16777220.f,0,0},{16777222.f,0,0}};
            roundedSource.rest = roundedSource.points;
            roundedSource.widths = {1,2,3,4};
            rounded.nodes[1].params[0].value = VtValue(.5f);
            rounded.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(3.5f), false});
            auto roundedReference = rounded;
            roundedReference.nodes[1].params[0].value = VtValue(1.f);
            roundedReference.nodes[1].params.back().value = VtValue(0.f);
            SixPlaneOracle roundedOracle; CHECK(BuildCpuReference(roundedReference, &roundedOracle));
            CHECK(roundedOracle.stableIds == std::vector<uint64_t>({999}) && roundedOracle.points.size() == 4);
            // CPU's uniform single-curve layout omits offsets; Vulkan exposes
            // the canonical topology plane for both uniform and ragged data.
            roundedOracle.curveOffsets = {0,4};
            roundedOracle.points = {{16777216.f,0,0},{16777216.f,0,0},
                                    {16777218.f,0,0},{16777220.f,0,0}};
            auto roundedSnapshot = commitChecked(rounded, roundedOracle);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, extendedSnapshot->generation->device, extended);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, first->generation->device, firstOracle);
            fixture.queueOwner->InvokeOwner([&] {
                extendedSnapshot.reset(); shortenedSnapshot.reset(); filteredSnapshot.reset();
                emptySetSnapshot.reset(); twiceSnapshot.reset();
                compoundSetSnapshot.reset(); compoundScaleSnapshot.reset(); compoundEmptySnapshot.reset();
                roundedSnapshot.reset();
            });
            cullProvider->topologyOrdinal = SIZE_MAX;
            cached = publish(descriptor, &cachedOracle);
        }
        auto invalid = descriptor;
        invalid.nodes.back().type = TfToken("UnsupportedVulkanOperator");
        session->SetGraphDesc(invalid);
        auto failed = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(failed && failed->diagnostics.HasErrors() && failed->generation == cached->generation);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, failed->generation->device, firstOracle);
        session->Drain(); session.reset();
        CHECK(cacheDomain->Invalidate());
        fixture.queueOwner->InvokeOwner([&] {
            first.reset(); both.reset(); repeated.reset(); empty.reset(); cached.reset(); failed.reset();
            wrongTopology.reset(); retried.reset();
        });
        provider->Shutdown(); executor->Shutdown(); preparer->Drain();
        fixture.queueOwner->Drain();
        CHECK(fixture.domain->Close());
        fixture.completion->CloseAndJoin();
        std::puts("Vulkan Session complete-packet Length Cull COW: PASS");
        return 0;
    }
    auto noncanonical = std::make_shared<UsdGenExecutionCacheDomain>(
        cacheDomain->Key(), cacheDomain->MaxBytes());
    CHECK(!CreateUsdGenDeviceSession(2, 8, noncanonical, provider));
    SixPlaneOracle oracle;
    auto descriptor = rootsMode ? (dagMode ? MakeProviderRootDagDescriptor(lengthMode, affineMode)
                                           : MakeProviderRootDescriptor(lengthMode, true, TfToken("onError"), affineMode))
                                : (dagMode ? MakeProviderDagDescriptor(lengthMode)
                                           : MakeProviderDescriptor(lengthMode));
    if (setShaderPath) {
        for (auto& node : descriptor.nodes) if (node.type == TfToken("UsdGenLength"))
            for (auto& param : node.params) if (param.name == TfToken("length:mode"))
                param.value = VtValue(TfToken("set"));
    }
    if (namedMode) {
        auto& channels = descriptor.curveSets.front().authoredPlanes;
        UsdGenAuthoredPlaneDesc point;
        point.name = TfToken("pointPair"); point.arity = 2;
        point.floatValues = {1,2,3,4,5,6,7,8,9,10};
        channels.push_back(point);
        UsdGenAuthoredPlaneDesc primitive;
        primitive.name = TfToken("curveTags"); primitive.type = UsdGenAuthoredPlaneType::Int32;
        primitive.domain = UsdGenAuthoredPlaneDomain::Primitive; primitive.arity = 3;
        primitive.intValues = {20,21,22,10,11,12};
        channels.push_back(primitive);
        UsdGenAuthoredPlaneDesc groom;
        groom.name = TfToken("groomTuple"); groom.domain = UsdGenAuthoredPlaneDomain::Groom; groom.arity = 16;
        groom.floatValues = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
        channels.push_back(groom);
    }
    size_t const widthIndex = dagMode ? descriptor.nodes.size() - 3 : descriptor.nodes.size() - 1;
    if (!BuildCpuReference(descriptor, &oracle) || oracle.points.empty() ||
        oracle.rest.size() != oracle.points.size() ||
        oracle.hairT.size() != oracle.points.size() ||
        oracle.width.size() != oracle.points.size() ||
        oracle.curveOffsets.size() != 3 || oracle.stableIds.size() != 2) {
        std::fprintf(stderr, "Vulkan Session provider CPU six-plane oracle failed\n");
        return 1;
    }
    if (rootsMode) {
        CHECK(oracle.stableIds == std::vector<uint64_t>({10,20}) &&
              oracle.rootPrim.size() == oracle.stableIds.size() &&
              oracle.rootUV.size() == oracle.stableIds.size() && oracle.named.size() >= 6);
    }
    if (namedMode) {
        CHECK(oracle.named.size() == (rootsMode ? 9u : 3u));
        for (auto const& channel : oracle.named) {
            if (channel.metadata.name == "pointPair")
                CHECK(channel.bytes == Bytes(std::vector<float>{7,8,9,10,1,2,3,4,5,6}));
            else if (channel.metadata.name == "curveTags")
                CHECK(channel.bytes == Bytes(std::vector<int32_t>{10,11,12,20,21,22}));
            else if (channel.metadata.name == "groomTuple") {
                CHECK(channel.metadata.name == "groomTuple");
                CHECK(channel.bytes == Bytes(std::vector<float>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}));
            }
        }
    }
    session->SetGraphDesc(descriptor);
    session->SetDevicePublicationEnabled(true);
    if (admissionRollbackMode) {
        // Exhaust native ingress by retaining exact-owner tickets, not by
        // blocking a queue worker. The Session return ticket is independent.
        std::vector<UsdGenExecutionPipeline::CommandTicket> occupied;
        for (uint64_t i = 0; i != fixture.queueOwner->CommandCapacity(); ++i) {
            auto ticket = fixture.queueOwner->ReserveCommandTicket();
            CHECK(ticket);
            occupied.push_back(std::move(ticket));
        }
        CHECK(!fixture.queueOwner->ReserveCommandTicket());
        auto const before = fixture.context->resources()->Snapshot().usedBytes;
        auto rejected = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(rejected && rejected->diagnostics.HasErrors() && !rejected->generation);
        CHECK(session->NeedsCommit() && cacheDomain->Size() == 0);
        CHECK(fixture.context->resources()->Snapshot().usedBytes == before && before == 0);
        CHECK(fixture.queueOwner->OutstandingCommands() == occupied.size());
        occupied.clear();
        CHECK(fixture.queueOwner->OutstandingCommands() == 0);
        // The ordinary full fixture below retries this same dirty Session,
        // proving rollback did not strand its return gate/coalesced leader.
    }
    auto sessionSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    if (!sessionSnapshot || !sessionSnapshot->generation) return 1;
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, sessionSnapshot->generation->device, oracle);
    CHECK(sessionSnapshot->generation->device);
    if (rootsMode) {
        CHECK(!sessionSnapshot->generation->device->Geometry().alreadyDeformed);
        // Root bindings participate in the Session cache identity even when
        // source/surface generations are unchanged. Verify a root-only COW
        // publication and retain/read the earlier immutable generation.
        auto rootMutation = descriptor;
        rootMutation.curveSets.front().skinPrimUv[0] = GfVec2f(.83f, .17f);
        rootMutation.nodes.front().params[1].value = VtValue(TfToken("always"));
        CHECK(rootMutation.curveSets.front().curveGeneration == descriptor.curveSets.front().curveGeneration);
        SixPlaneOracle rootMutationOracle; CHECK(BuildCpuReference(rootMutation, &rootMutationOracle));
        session->SetGraphDesc(rootMutation);
        auto rootMutationSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(rootMutationSnapshot && !rootMutationSnapshot->diagnostics.HasErrors() &&
              rootMutationSnapshot->generation && rootMutationSnapshot->generation->device);
        CHECK(rootMutationSnapshot->generation->device->Geometry().valueVersion !=
              sessionSnapshot->generation->device->Geometry().valueVersion);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, rootMutationSnapshot->generation->device, rootMutationOracle);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, sessionSnapshot->generation->device, oracle);
        fixture.queueOwner->InvokeOwner([&] { rootMutationSnapshot.reset(); });
        session->SetGraphDesc(descriptor);
    }
    auto changed = descriptor;
    // Keep the source generation constant and change only the final Width or
    // the intervening Length value.  This exercises Session's authoritative
    // intermediate revisions and verifies value COW rather than a topology
    // rebuild.
    auto& changedControl = lengthMode ? changed.nodes[1].params[0]
                                      : changed.nodes[widthIndex].params[0];
    changedControl.value = VtValue(lengthMode ? 0.75f : 0.5f);
    CHECK(changed.curveSets[0].curveGeneration == descriptor.curveSets[0].curveGeneration);
    SixPlaneOracle changedOracle;
    CHECK(BuildCpuReference(changed, &changedOracle));
    session->SetGraphDesc(changed);
    auto changedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(changedSnapshot && !changedSnapshot->diagnostics.HasErrors());
    CHECK(changedSnapshot->generation && changedSnapshot->generation->device);
    CHECK(changedSnapshot->generation->device->Geometry().valueVersion !=
          sessionSnapshot->generation->device->Geometry().valueVersion);
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, changedSnapshot->generation->device, changedOracle);
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, sessionSnapshot->generation->device, oracle);

    // Returning to the identical descriptor uses the real Session cache,
    // retaining native ownership and immutable value revisions.
    session->SetGraphDesc(descriptor);
    auto cachedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(cachedSnapshot && !cachedSnapshot->diagnostics.HasErrors());
    CHECK(cachedSnapshot->generation && cachedSnapshot->generation->device);
    std::fprintf(stderr, "Session cache: entries=%zu hits=%llu misses=%llu admissions=%llu failures=%llu versions=%llu/%llu\n",
        cacheDomain->Size(),
        static_cast<unsigned long long>(cachedSnapshot->stats.executionCacheHits),
        static_cast<unsigned long long>(cachedSnapshot->stats.executionCacheMisses),
        static_cast<unsigned long long>(cachedSnapshot->stats.executionCacheAdmissions),
        static_cast<unsigned long long>(cachedSnapshot->stats.executionCacheAdmissionFailures),
        static_cast<unsigned long long>(sessionSnapshot->generation->device->Geometry().valueVersion),
        static_cast<unsigned long long>(cachedSnapshot->generation->device->Geometry().valueVersion));
    CHECK(cachedSnapshot->generation->device->Owner() ==
          sessionSnapshot->generation->device->Owner());
    CHECK(cachedSnapshot->generation->device->Geometry().valueVersion ==
          sessionSnapshot->generation->device->Geometry().valueVersion);
    CHECK(cachedSnapshot->generation->device->Identity().generation !=
          sessionSnapshot->generation->device->Identity().generation);
    CHECK(cachedSnapshot->stats.executionCacheHits > 0);
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, cachedSnapshot->generation->device, oracle);

    if (setShaderPath) {
        auto degenerate = descriptor;
        auto& curves = degenerate.curveSets.front();
        size_t offset = 0;
        for (auto count : curves.curveVertexCounts) {
            auto const root = curves.points[offset];
            for (int i = 0; i != count; ++i) curves.points[offset + size_t(i)] = root;
            offset += size_t(count);
        }
        curves.rest = curves.points;
        session->SetGraphDesc(degenerate);
        auto failedSet = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(failedSet && failedSet->diagnostics.HasErrors() &&
              failedSet->generation == cachedSnapshot->generation && session->NeedsCommit());
        // Zero current length with a positive target cannot invent a tangent.
        // Changing the target to zero gives an exact degenerate no-op, with
        // topology and all non-point data retained and the dirty request retryable.
        auto zeroTarget = degenerate;
        for (auto& node : zeroTarget.nodes) if (node.type == TfToken("UsdGenLength"))
            for (auto& param : node.params) if (param.name == TfToken("length:value"))
                param.value = VtValue(0.f);
        SixPlaneOracle zeroOracle; CHECK(BuildCpuReference(zeroTarget, &zeroOracle));
        session->SetGraphDesc(zeroTarget);
        auto zeroSet = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(zeroSet && !zeroSet->diagnostics.HasErrors() && zeroSet->generation && zeroSet->generation->device);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, zeroSet->generation->device, zeroOracle);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, cachedSnapshot->generation->device, oracle);
        fixture.queueOwner->InvokeOwner([&] { failedSet.reset(); zeroSet.reset(); });
        session->SetGraphDesc(descriptor);
        cachedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(cachedSnapshot && !cachedSnapshot->diagnostics.HasErrors() && cachedSnapshot->generation);
    }

    if (affineMode) {
        // A transform-only source edit must partition the real Session cache
        // while preserving the source payload and both authored generations.
        auto matrixMutation = descriptor;
        auto const sourcePoints = matrixMutation.curveSets.front().points;
        auto const sourceRest = matrixMutation.curveSets.front().rest;
        matrixMutation.curveSets.front().worldMatrix[3][0] += 0.125;
        CHECK(matrixMutation.curveSets.front().curveGeneration == descriptor.curveSets.front().curveGeneration &&
              matrixMutation.surfaces.front().surfaceGeneration == descriptor.surfaces.front().surfaceGeneration);
        CHECK(matrixMutation.curveSets.front().points == sourcePoints &&
              matrixMutation.curveSets.front().rest == sourceRest);
        SixPlaneOracle matrixOracle;
        CHECK(BuildCpuReference(matrixMutation, &matrixOracle));
        session->SetGraphDesc(matrixMutation);
        auto matrixSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(matrixSnapshot && !matrixSnapshot->diagnostics.HasErrors() &&
              matrixSnapshot->generation && matrixSnapshot->generation->device);
        CHECK(matrixSnapshot->stats.executionCacheMisses > cachedSnapshot->stats.executionCacheMisses &&
              matrixSnapshot->generation->device->Geometry().valueVersion !=
                  cachedSnapshot->generation->device->Geometry().valueVersion);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, matrixSnapshot->generation->device, matrixOracle);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, cachedSnapshot->generation->device, oracle);
        fixture.queueOwner->InvokeOwner([&] { matrixSnapshot.reset(); });
        session->SetGraphDesc(descriptor);
        cachedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(cachedSnapshot && !cachedSnapshot->diagnostics.HasErrors() &&
              cachedSnapshot->generation && cachedSnapshot->generation->device);
    }

    if (dagMode) {
        // Exercise the ordered fan-in, authored-vector sorting and a longer
        // branch through the real Session, not a hard-coded fork fixture.
        for (int variant = 0; variant != (namedMode ? 8 : 6); ++variant) {
            auto dag = descriptor;
            if (variant < 2) dag.nodes.back().params[0].value = VtValue(float(variant));
            else if (variant == 2) std::swap(dag.nodes.back().inputs[0], dag.nodes.back().inputs[1]);
            else if (variant == 3) std::reverse(dag.nodes.begin(), dag.nodes.end());
            else if (variant == 4) {
                auto extra = dag.nodes[widthIndex];
                extra.path = SdfPath("/SessionProvider/extraWidth");
                extra.inputs = {dag.nodes[widthIndex].path};
                extra.params = {{TfToken("width"), VtValue(0.5f), false},
                                {TfToken("replace"), VtValue(false), false}};
                dag.nodes.back().inputs[0] = extra.path;
                dag.nodes.insert(dag.nodes.end() - 1, extra);
            } else if (variant == 5) {
                auto second = dag.nodes.back();
                second.path = SdfPath("/SessionProvider/secondBlend");
                second.inputs = {dag.terminal, dag.nodes[widthIndex].path};
                second.params[0].value = VtValue(0.6f);
                dag.nodes.push_back(second);
                dag.terminal = second.path;
            } else if (variant == 6) {
                // Same authored source generation, different immutable named
                // payload: the descriptor/cache key must distinguish them.
                dag.curveSets.front().authoredPlanes.front().floatValues[0] += 100.f;
            } else {
                auto& channel = dag.curveSets.front().authoredPlanes.front();
                channel.type = UsdGenAuthoredPlaneType::Int32;
                channel.intValues.resize(channel.floatValues.size());
                for (size_t i = 0; i != channel.intValues.size(); ++i)
                    channel.intValues[i] = static_cast<int>(i + 1);
                channel.floatValues.clear();
            }
            SixPlaneOracle expected;
            CHECK(BuildCpuReference(dag, &expected));
            session->SetGraphDesc(dag);
            auto result = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            CHECK(result && !result->diagnostics.HasErrors() && result->generation && result->generation->device);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, result->generation->device, expected);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, sessionSnapshot->generation->device, oracle);
            fixture.queueOwner->InvokeOwner([&] { result.reset(); });
        }
        if (comparePipeline) {
            auto separateOrigins = descriptor;
            auto secondLength = separateOrigins.nodes[1];
            secondLength.path = SdfPath("/SessionProvider/rightLength");
            // The two Length stages allocate distinct points owners, even
            // though identical controls produce equal non-width data.
            separateOrigins.nodes[widthIndex + 1].inputs = {secondLength.path};
            separateOrigins.nodes.insert(separateOrigins.nodes.begin() + 2, secondLength);
            SixPlaneOracle expected;
            CHECK(BuildCpuReference(separateOrigins, &expected));
            session->SetGraphDesc(separateOrigins);
            auto equalSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            CHECK(equalSnapshot && !equalSnapshot->diagnostics.HasErrors() &&
                  equalSnapshot->generation && equalSnapshot->generation->device);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, equalSnapshot->generation->device, expected);

            auto unequalOrigins = separateOrigins;
            unequalOrigins.nodes[2].params[0].value = VtValue(0.75f);
            session->SetGraphDesc(unequalOrigins);
            auto unequalSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            CHECK(unequalSnapshot && unequalSnapshot->diagnostics.HasErrors());
            CHECK(unequalSnapshot->generation == equalSnapshot->generation);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, unequalSnapshot->generation->device, expected);
            CheckPlanes(fixture.native, fixture.context, fixture.completion,
                        *fixture.queueOwner, sessionSnapshot->generation->device, oracle);
            fixture.queueOwner->InvokeOwner([&] { equalSnapshot.reset(); unequalSnapshot.reset(); });
        }
        // Empty publication still owns the native offset sentinel. The other
        // five core channels retain metadata without allocations or payloads.
        auto emptyDag = descriptor;
        auto& emptyCurves = emptyDag.curveSets.front();
        emptyCurves.curveVertexCounts.clear();
        emptyCurves.curveId.clear();
        emptyCurves.points.clear();
        emptyCurves.rest.clear();
        emptyCurves.widths.clear();
        emptyCurves.skinPrim.clear();
        emptyCurves.skinPrimUv.clear();
        emptyCurves.rootFrame.clear();
        for (auto& channel : emptyCurves.authoredPlanes) {
            if (channel.domain != UsdGenAuthoredPlaneDomain::Groom) {
                channel.floatValues.clear(); channel.intValues.clear();
            }
        }
        ++emptyCurves.curveGeneration;
        session->SetGraphDesc(emptyDag);
        auto emptySnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(emptySnapshot && !emptySnapshot->diagnostics.HasErrors() &&
              emptySnapshot->generation && emptySnapshot->generation->device);
        SixPlaneOracle emptyNative;
        emptyNative.curveOffsets = {0};
        emptyNative.named = oracle.named;
        for (auto& channel : emptyNative.named) {
            if (channel.metadata.domain != UsdGenDeviceDomain::Groom) {
                channel.metadata.elementCount = 0; channel.bytes.clear();
            }
        }
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, emptySnapshot->generation->device, emptyNative);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, sessionSnapshot->generation->device, oracle);
        fixture.queueOwner->InvokeOwner([&] { emptySnapshot.reset(); });
        session->SetGraphDesc(descriptor);
        cachedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(cachedSnapshot && cachedSnapshot->generation && !cachedSnapshot->diagnostics.HasErrors());
    }

    if (rootsMode) {
        // `always` resolves current-space roots against the rest surface;
        // source current/rest choice is intentionally observable through the
        // published alreadyDeformed flag while root channels stay public.
        auto rebound = MakeProviderRootDescriptor(lengthMode, false, TfToken("always"));
        SixPlaneOracle reboundOracle; CHECK(BuildCpuReference(rebound, &reboundOracle));
        CHECK(reboundOracle.rootPrim.size() == reboundOracle.stableIds.size());
        session->SetGraphDesc(rebound);
        auto reboundSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(reboundSnapshot && !reboundSnapshot->diagnostics.HasErrors() &&
              reboundSnapshot->generation && reboundSnapshot->generation->device);
        CHECK(reboundSnapshot->generation->device->Geometry().alreadyDeformed);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, reboundSnapshot->generation->device, reboundOracle);

        // A never-rebind source with no valid root drops every C3.  It still
        // publishes the required empty points/IDs/offset sentinel convention
        // and its Groom constants, but no optional roots or private frames.
        auto dropped = MakeProviderRootDescriptor(lengthMode, true, TfToken("never"));
        dropped.curveSets.front().skinPrim = {-1, -1};
        SixPlaneOracle droppedOracle; CHECK(BuildCpuReference(dropped, &droppedOracle));
        CHECK(droppedOracle.points.empty() && droppedOracle.rest.empty() &&
              droppedOracle.curveOffsets == std::vector<uint32_t>({0}) &&
              droppedOracle.stableIds.empty() && droppedOracle.rootPrim.empty() &&
              droppedOracle.rootUV.empty() && droppedOracle.named.size() == 6);
        for (auto const& plane : droppedOracle.named)
            if (plane.metadata.domain == UsdGenDeviceDomain::Groom) CHECK(!plane.bytes.empty());
            else CHECK(plane.bytes.empty() && plane.metadata.elementCount == 0);
        session->SetGraphDesc(dropped);
        auto droppedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(droppedSnapshot && !droppedSnapshot->diagnostics.HasErrors() &&
              droppedSnapshot->generation && droppedSnapshot->generation->device);
        CHECK(!droppedSnapshot->generation->device->Geometry().alreadyDeformed);
        CheckPlanes(fixture.native, fixture.context, fixture.completion,
                    *fixture.queueOwner, droppedSnapshot->generation->device, droppedOracle);
        fixture.queueOwner->InvokeOwner([&] { reboundSnapshot.reset(); droppedSnapshot.reset(); });
        session->SetGraphDesc(descriptor);
        cachedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
        CHECK(cachedSnapshot && !cachedSnapshot->diagnostics.HasErrors() &&
              cachedSnapshot->generation && cachedSnapshot->generation->device);
    }

    auto unsupported = descriptor;
    unsupported.nodes.back().type = TfToken("UnsupportedVulkanOperator");
    session->SetGraphDesc(unsupported);
    auto failedSnapshot = session->CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(failedSnapshot && failedSnapshot->diagnostics.HasErrors());
    CHECK(failedSnapshot->generation == cachedSnapshot->generation);
    CheckPlanes(fixture.native, fixture.context, fixture.completion,
                *fixture.queueOwner, failedSnapshot->generation->device, oracle);
    session->Drain();
    session.reset();
    CHECK(cacheDomain->Invalidate());
    fixture.queueOwner->InvokeOwner([&] {
        failedSnapshot.reset(); cachedSnapshot.reset();
        changedSnapshot.reset(); sessionSnapshot.reset();
    });
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
