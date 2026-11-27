// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// End-to-end proof for the deliberately narrow descriptor -> native Vulkan
// source/width executor.  CPU construction below is an independent oracle for
// C3 canonicalization; GPU bytes are always read back from the publication.
#include "utils/pxrInterop.h"
#include "usdGenCurves/vulkan/planExecutor.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "nodeSets.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/widthBlendPipeline.h"
#include "usdGen/curveLoader.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "vulkanNativeFixture.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan plan executor check failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
template <class T> std::vector<uint8_t> Bytes(std::vector<T> const& value) {
    std::vector<uint8_t> result(value.size() * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), value.data(), result.size());
    return result;
}

UsdGenGraphDesc MakeDescriptor(bool sourceWidths, bool useRest, float literal,
                               bool replace, uint64_t generation) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Executor");
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");
    desc.defaultWidth = .8f;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Executor/C3"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.curveVertexCounts = {3, 2, 4};
    // Deliberately unsorted stable IDs and ragged source order.
    curves.curveId = {30, 10, 20};
    curves.points = {{300,0,0},{301,1,0},{302,2,0},
                     {100,0,1},{101,1,1},
                     {200,0,2},{201,1,2},{202,2,2},{203,3,2}};
    curves.rest = {{3000,0,0},{3001,1,0},{3002,2,0},
                   {1000,0,1},{1001,1,1},
                   {2000,0,2},{2001,1,2},{2002,2,2},{2003,3,2}};
    if (sourceWidths) curves.widths = {3,4,5, 6,7, 8,9,10,11};
    curves.widthsInterpolation = TfToken("vertex");
    curves.curveGeneration = generation; curves.frozenEpoch = "locked";
    desc.curveSets = {curves};
    UsdGenNodeDesc source;
    source.path = SdfPath("/Executor/Ops/source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("useRest"), VtValue(useRest), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("expectEpoch"), VtValue(std::string("locked")), false},
                     {TfToken("staleAction"), VtValue(TfToken("block")), false},
                     {TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Executor/Ops/width"); width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params = {{TfToken("width"), VtValue(literal), false},
                    {TfToken("replace"), VtValue(replace), false}};
    desc.nodes = {source, width}; desc.terminal = width.path;
    return desc;
}

UsdGenGraphDesc MakeFanoutDescriptor() {
    auto desc = MakeDescriptor(true, true, .5f, true, 27);
    auto source = desc.nodes[0];
    auto left = desc.nodes[1]; left.path = SdfPath("/Executor/Ops/widthL");
    left.params[0].value = VtValue(.5f);
    auto right = desc.nodes[1]; right.path = SdfPath("/Executor/Ops/widthR");
    right.params[0].value = VtValue(1.5f);
    auto blend = UsdGenNodeDesc{};
    blend.path = SdfPath("/Executor/Ops/blend"); blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {left.path, right.path};
    blend.params = {{TfToken("widthBlend:weight"), VtValue(.25f), false}};
    desc.nodes = {source, blend, right, left}; desc.terminal = blend.path;
    return desc;
}

void AddAuthoredPlanes(UsdGenGraphDesc* desc);

// A real C3 root fixture: the loader resolves this source through the
// CurveSource surface relationship, then canonicalizes the deliberately
// ragged/unsorted C3 order.  rootFrame is intentionally authored (rather
// than inferred) so the executor's private T/B/N capture path is exercised
// without making those implementation planes public generation channels.
UsdGenGraphDesc MakeRootBoundDescriptor(bool useRest, TfToken rebind) {
    auto desc = MakeDescriptor(true, useRest, .75f, true, 31);
    auto& curves = desc.curveSets.front();
    // Keep default-time roots on the fixture mesh so `always` can perform a
    // real resolved-rest-surface repair rather than deliberately dropping it.
    curves.rest = curves.points;
    curves.skinPrim = {0, 0, 0};
    curves.skinPrimUv = {{.20f,.30f}, {.40f,.10f}, {.60f,.20f}};
    curves.rootFrame.resize(3, usdGen::UsdGenMatrix4d(1.0));
    curves.rootFrame[0][3][0] = 30.0;
    curves.rootFrame[1][3][0] = 10.0;
    curves.rootFrame[2][3][0] = 20.0;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Executor/Scalp");
    surface.id = 31;
    surface.faceVertexCounts = {4}; surface.faceVertexIndices = {0, 1, 2, 3};
    surface.restPoints = {{0,0,0},{400,0,0},{400,0,4},{0,0,4}};
    surface.points = surface.restPoints; surface.uv = {{0,0},{1,0},{1,1},{0,1}};
    desc.surfaces = {surface};
    desc.nodes[0].surfaces = {surface.path};
    desc.nodes[0].params[0].value = VtValue(useRest);
    desc.nodes[0].params[4].value = VtValue(rebind);
    AddAuthoredPlanes(&desc);
    return desc;
}

// Keep C3 data in its own object space while deriving the private root frame
// through a genuinely different source/surface world-space relationship.  The
// public source planes must not be world-baked as a side effect of capture.
UsdGenGraphDesc MakeAffineRootBoundDescriptor() {
    auto desc = MakeRootBoundDescriptor(true, TfToken("never"));
    desc.curveSets.front().rootFrame.clear();
    usdGen::UsdGenMatrix4d sourceWorld(1.0);
    sourceWorld[0][0] = 0.0; sourceWorld[0][1] = -1.0;
    sourceWorld[1][0] = 1.0; sourceWorld[1][1] = 0.0;
    sourceWorld[3][0] = 11.0; sourceWorld[3][1] = -13.0; sourceWorld[3][2] = 17.0;
    usdGen::UsdGenMatrix4d relative(1.0);
    relative[0][0] = 0.0; relative[0][1] = 2.0;
    relative[1][0] = -3.0; relative[1][1] = 0.0;
    relative[2][2] = .5;
    relative[3][0] = 5.0; relative[3][1] = -2.0; relative[3][2] = 7.0;
    desc.curveSets.front().worldMatrix = sourceWorld;
    desc.surfaces.front().worldMatrix = relative * sourceWorld;
    return desc;
}

// The Length branch has a different immutable non-width geometry bundle from
// the direct Width branch.  Its WidthBlend therefore requires a native
// non-width comparator even though both branches originate at one source.
UsdGenGraphDesc MakeLengthBlendDescriptor() {
    auto desc = MakeDescriptor(true, true, .5f, true, 28);
    auto source = desc.nodes[0];
    auto left = desc.nodes[1]; left.path = SdfPath("/Executor/Ops/widthL");
    auto length = UsdGenNodeDesc{};
    length.path = SdfPath("/Executor/Ops/lengthR");
    length.type = TfToken("UsdGenLength"); length.inputs = {source.path};
    length.params = {{TfToken("length:value"), VtValue(1.0), false}};
    auto blend = UsdGenNodeDesc{};
    blend.path = SdfPath("/Executor/Ops/lengthBlend");
    blend.type = TfToken("UsdGenWidthBlend");
    blend.inputs = {left.path, length.path};
    blend.params = {{TfToken("widthBlend:weight"), VtValue(.5f), false}};
    desc.nodes = {source, left, length, blend}; desc.terminal = blend.path;
    return desc;
}

struct Oracle {
    std::vector<usdGen::UsdGenVec3f> points, rest;
    std::vector<uint32_t> offsets;
    std::vector<uint64_t> ids;
    std::vector<float> hairT, widths;
    std::vector<int32_t> rootPrim;
    std::vector<usdGen::UsdGenVec2f> rootUV;
    struct NamedPlane {
        std::string name;
        UsdGenDeviceValueType type;
        UsdGenDeviceDomain domain;
        uint32_t arity;
        std::vector<uint8_t> bytes;
    };
    std::vector<NamedPlane> named;
};

void AddAuthoredPlane(UsdGenCurveSetDesc* curves, char const* name,
                      UsdGenAuthoredPlaneType type,
                      UsdGenAuthoredPlaneDomain domain, uint8_t arity) {
    if (!curves) std::abort();
    UsdGenAuthoredPlaneDesc plane;
    plane.name = TfToken(name); plane.type = type; plane.domain = domain;
    plane.arity = arity;
    size_t const elements = domain == UsdGenAuthoredPlaneDomain::Point
        ? curves->points.size() : domain == UsdGenAuthoredPlaneDomain::Primitive
        ? curves->curveVertexCounts.size() : 1;
    size_t const values = elements * arity;
    if (type == UsdGenAuthoredPlaneType::Float32) {
        plane.floatValues.resize(values);
        for (size_t i = 0; i != values; ++i) plane.floatValues[i] = float(i) + .125f;
    } else {
        plane.intValues.resize(values);
        for (size_t i = 0; i != values; ++i) plane.intValues[i] = int(i * 7 + 3);
    }
    curves->authoredPlanes.push_back(std::move(plane));
}

void AddAuthoredPlanes(UsdGenGraphDesc* desc) {
    if (!desc || desc->curveSets.empty()) std::abort();
    auto* curves = &desc->curveSets.front();
    // Deliberately use the ragged, unsorted C3 fixture's point/curve order.
    // CpuReference below is the oracle for canonical stable-ID gathering.
    AddAuthoredPlane(curves, "pointFloat16", UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Point, 16);
    AddAuthoredPlane(curves, "pointInt3", UsdGenAuthoredPlaneType::Int32,
                     UsdGenAuthoredPlaneDomain::Point, 3);
    AddAuthoredPlane(curves, "primitiveFloat2", UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Primitive, 2);
    AddAuthoredPlane(curves, "primitiveInt1", UsdGenAuthoredPlaneType::Int32,
                     UsdGenAuthoredPlaneDomain::Primitive, 1);
    AddAuthoredPlane(curves, "groomFloat4", UsdGenAuthoredPlaneType::Float32,
                     UsdGenAuthoredPlaneDomain::Groom, 4);
    AddAuthoredPlane(curves, "groomInt16", UsdGenAuthoredPlaneType::Int32,
                     UsdGenAuthoredPlaneDomain::Groom, 16);
}

// Arithmetic cross-check only.  The authoritative oracle below is the CPU
// compiler/scheduler on the same descriptor.
Oracle ManualExpected(bool sourceWidths, float literal, bool replace, bool indexIds = false) {
    auto source = MakeDescriptor(sourceWidths, true, literal, replace, 17).curveSets.front();
    std::vector<size_t> order{0, 1, 2};
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return (indexIds ? a : source.curveId[a]) < (indexIds ? b : source.curveId[b]);
    });
    Oracle result; result.offsets.push_back(0);
    size_t input = 0;
    std::vector<size_t> begins;
    for (int count : source.curveVertexCounts) { begins.push_back(input); input += size_t(count); }
    for (size_t curve : order) {
        auto const count = size_t(source.curveVertexCounts[curve]);
        result.ids.push_back(indexIds ? curve : source.curveId[curve]);
        for (size_t i = 0; i != count; ++i) {
            auto const at = begins[curve] + i;
            result.points.push_back(source.points[at]); result.rest.push_back(source.rest[at]);
            result.hairT.push_back(float(i) / float(count - 1));
            float const inputWidth = sourceWidths ? source.widths[at] : .8f;
            result.widths.push_back(replace ? inputWidth + (literal - inputWidth) :
                inputWidth * (1.0f + (literal - 1.0f)));
        }
        result.offsets.push_back(uint32_t(result.points.size()));
    }
    return result;
}

bool CpuReference(UsdGenGraphDesc desc, Oracle* expected) {
    if (!expected) return false;
    desc.executionBackend = kUsdGenBackendCpu;
    UsdGenCompiler compiler; UsdGenGraph graph;
    auto compiled = usdGenUsd::Compile(compiler, desc, &graph);
    if (!compiled.ok) return false;
    UsdGenScheduler scheduler(2); UsdGenEvalContext context; context.desc = &graph.Spec();
    auto result = scheduler.Run(graph, context, 1);
    if (result.diagnostics.HasErrors()) return false;
    UsdGenCurveBuffer const buffer = NodeBuffer(graph, usdGenUsd::NodeIdForPath(graph, desc.terminal));
    Oracle output;
    output.points.resize(buffer.totalCvs); output.rest.assign(buffer.rest.begin(), buffer.rest.end());
    for (size_t i = 0; i != output.points.size(); ++i)
        output.points[i] = {buffer.px[i], buffer.py[i], buffer.pz[i]};
    output.offsets.assign(buffer.cvOffsets.begin(), buffer.cvOffsets.end());
    output.ids.assign(buffer.curveId.begin(), buffer.curveId.end());
    output.hairT.assign(buffer.hairT.begin(), buffer.hairT.end());
    output.widths.assign(buffer.width.begin(), buffer.width.end());
    output.rootPrim.assign(buffer.rootPrim.begin(), buffer.rootPrim.end());
    output.rootUV.assign(buffer.rootUV.begin(), buffer.rootUV.end());
    auto appendNamed = [&](UsdGenPlane const& plane, UsdGenDeviceDomain domain) {
        Oracle::NamedPlane named;
        named.name = plane.name.GetString();
        named.type = plane.type == TfToken("float") ? UsdGenDeviceValueType::Float32
                                                     : UsdGenDeviceValueType::Int32;
        named.domain = domain; named.arity = plane.arity;
        if (plane.type == TfToken("float")) {
            named.bytes.resize(plane.f.size() * sizeof(float));
            if (!named.bytes.empty()) std::memcpy(named.bytes.data(), plane.f.cdata(), named.bytes.size());
        } else {
            named.bytes.resize(plane.i.size() * sizeof(int));
            if (!named.bytes.empty()) std::memcpy(named.bytes.data(), plane.i.cdata(), named.bytes.size());
        }
        output.named.push_back(std::move(named));
    };
    for (auto const& plane : buffer.extraCv)
        appendNamed(plane, UsdGenDeviceDomain::Point);
    for (auto const& plane : buffer.extraCurve) {
        auto const domain = plane.interpolation == TfToken("constant")
            ? UsdGenDeviceDomain::Groom : UsdGenDeviceDomain::Primitive;
        appendNamed(plane, domain);
    }
    *expected = std::move(output); return true;
}

struct Result { std::shared_ptr<const UsdGenDeviceGeneration> generation; std::exception_ptr error; };

bool CopyCancellation(UsdGenExecutionPipeline& owner,
                      UsdGenExecutionPipeline::Cancellation* cancellation) {
    bool accepted = false;
    owner.Await([&](auto done) {
        auto signal = std::make_shared<std::function<void()>>(std::move(done));
        accepted = owner.Submit([cancellation, signal](auto const& token) {
            *cancellation = token;
            return UsdGenExecutionPipeline::Publish([signal] { (*signal)(); });
        }) != 0;
        if (!accepted) (*signal)();
    });
    return accepted && cancellation->epoch != 0 && !cancellation->Superseded();
}

Result SubmitAndWait(UsdGenExecutionPipeline& owner,
                     std::shared_ptr<VulkanPlanExecutor> const& executor,
                     std::shared_ptr<const VulkanSourceWidthPlan> const& plan,
                     std::shared_ptr<DeviceContext> const& context, uint64_t generation,
                     std::shared_ptr<const void> request) {
    auto promise = std::make_shared<std::promise<Result>>();
    auto future = promise->get_future();
    std::weak_ptr<const void> requestWeak = request;
    owner.InvokeOwner([&] {
        VulkanPlanExecutor::Request requestInfo;
        requestInfo.context = context; requestInfo.generation = generation;
        if (plan->IntermediateCount() != 0) {
            VulkanPlanExecutor::Request::AuthoritativeRevisions revisions;
            revisions.topologyVersion = 40; revisions.sourceValueVersion = 41;
            for (size_t i = 0; i != plan->IntermediateCount(); ++i)
                revisions.intermediateValueVersions.push_back(42 + i);
            revisions.finalValueVersion = 42 + plan->IntermediateCount();
            requestInfo.authoritativeRevisions = std::move(revisions);
        }
        requestInfo.tool = {"vulkan-plan-executor", 9, generation};
        requestInfo.requestLifetime = std::move(request);
        CHECK(executor->Submit(plan, std::move(requestInfo), [promise, requestWeak, &owner](auto output, auto error) {
            CHECK(owner.IsExecutingOwner());
            // An async terminal callback must still have the request owner;
            // this is deliberately fatal because use-after-free is not a
            // recoverable test failure.
            CHECK(!requestWeak.expired());
            promise->set_value({std::move(output), error});
        }));
    });
    // This is an external bounded future wait, never a queue-owner GPU wait.
    CHECK(future.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    auto result = future.get(); CHECK(!result.error && result.generation);
    owner.Drain(); // Completion may wake the promise just before returning.
    return result;
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
                 Oracle const& expected) {
    VulkanGenerationLease lease;
    auto const stream = reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
    auto reads = std::make_shared<BatchRead>(); reads->native = native;
    auto completed = std::make_shared<std::promise<void>>(); auto done = completed->get_future();
    owner.InvokeOwner([&] {
        lease = AcquireVulkanGeneration(generation, stream);
        CHECK(lease && lease.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        auto const& planes = lease.Planes();
        auto const rootPlanes = expected.rootPrim.empty() ? 0u : 2u;
        CHECK(planes.size() == 6 + rootPlanes + expected.named.size());
        for (auto const& plane : planes)
            CHECK(plane.metadata.name != "sourceRootT" && plane.metadata.name != "sourceRootB" &&
                  plane.metadata.name != "sourceRootN");
        std::vector<std::pair<char const*, std::vector<uint8_t>>> want{
            {"points", Bytes(expected.points)}, {"rest", Bytes(expected.rest)},
            {"curveOffsets", Bytes(expected.offsets)}, {"curveId", Bytes(expected.ids)},
            {"hairT", Bytes(expected.hairT)}, {"widths", Bytes(expected.widths)}};
        if (!expected.rootPrim.empty()) {
            CHECK(expected.rootUV.size() == expected.rootPrim.size());
            want.emplace_back("skinprim", Bytes(expected.rootPrim));
            want.emplace_back("skinprimuv", Bytes(expected.rootUV));
        }
        for (auto const& named : expected.named) {
            auto plane = std::find_if(planes.begin(), planes.end(), [&](auto const& p) {
                return p.metadata.name == named.name;
            });
            CHECK(plane != planes.end());
            CHECK(plane->metadata.type == named.type && plane->metadata.domain == named.domain &&
                  plane->metadata.elementCount == (named.domain == UsdGenDeviceDomain::Point
                      ? expected.points.size() : named.domain == UsdGenDeviceDomain::Primitive
                      ? expected.ids.size() : 1) &&
                  plane->metadata.arity == named.arity &&
                  plane->metadata.strideBytes == 4 * named.arity &&
                  plane->metadata.readOnly &&
                  plane->metadata.semantic == UsdGenDeviceChannelSemantic::Generic);
            want.emplace_back(named.name.c_str(), named.bytes);
        }
        reads->watch = service->Reserve(reads, [reads, completed, &owner](VkResult proof) {
            CHECK(owner.IsExecutingOwner() && proof == VK_SUCCESS);
            for (size_t i = 0; i != reads->staging.size(); ++i) {
                CHECK(reads->staging[i]->PollComplete() == VK_SUCCESS);
                void* mapped = nullptr;
                CHECK(vkMapMemory(reads->native->device, reads->staging[i]->memory(), 0,
                    reads->expected[i].size(), 0, &mapped) == VK_SUCCESS);
                if (std::memcmp(mapped, reads->expected[i].data(), reads->expected[i].size()) != 0) {
                    auto const* actual = static_cast<unsigned char const*>(mapped);
                    for (size_t byte = 0; byte < reads->expected[i].size(); ++byte) {
                        if (actual[byte] == reads->expected[i][byte]) continue;
                        std::fprintf(stderr, "plane %zu first mismatch byte %zu: actual %02x expected %02x\n",
                            i, byte, unsigned(actual[byte]), unsigned(reads->expected[i][byte]));
                        if ((i == 0 || i == 1 || i == 4 || i == 5) && (byte / 4 + 1) * 4 <= reads->expected[i].size()) {
                            float got = 0, want = 0;
                            std::memcpy(&got, actual + (byte / 4) * 4, 4);
                            std::memcpy(&want, reads->expected[i].data() + (byte / 4) * 4, 4);
                            std::fprintf(stderr, "scalar %zu actual %.9g expected %.9g\n", byte / 4, double(got), double(want));
                        }
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
            CHECK(plane != planes.end() && plane->bytes == item.second.size());
            if (std::strcmp(item.first, "skinprim") == 0)
                CHECK(plane->metadata.type == UsdGenDeviceValueType::Int32 &&
                      plane->metadata.domain == UsdGenDeviceDomain::Primitive &&
                      plane->metadata.semantic == UsdGenDeviceChannelSemantic::RootPrim &&
                      plane->metadata.elementCount == expected.ids.size() && plane->metadata.arity == 1);
            if (std::strcmp(item.first, "skinprimuv") == 0)
                CHECK(plane->metadata.type == UsdGenDeviceValueType::Float32x2 &&
                      plane->metadata.domain == UsdGenDeviceDomain::Primitive &&
                      plane->metadata.semantic == UsdGenDeviceChannelSemantic::RootUV &&
                      plane->metadata.elementCount == expected.ids.size() && plane->metadata.arity == 2);
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
// Read back the byte contents of one named plane from a published generation.
// Mirrors the owner/watch/fence plumbing of CheckPlanes but captures a single
// plane into host memory so a caller can compare it against another generation.
std::vector<uint8_t> ReadPlaneBytes(
    std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<VulkanCompletionService> const& service,
    UsdGenExecutionPipeline& owner,
    std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
    char const* planeName) {
    struct OneRead final {
        std::shared_ptr<NativeOwner> native;
        std::shared_ptr<ChargedBuffer> staging;
        std::vector<uint8_t> bytes;
        std::unique_ptr<VulkanCompletionService::Watch> watch;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool got = false;
        ~OneRead() {
            if (command) vkFreeCommandBuffers(native->device, native->commands, 1, &command);
            if (fence) vkDestroyFence(native->device, fence, nullptr);
        }
    };
    VulkanGenerationLease lease;
    auto const stream = reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
    auto reads = std::make_shared<OneRead>(); reads->native = native;
    auto completed = std::make_shared<std::promise<void>>(); auto done = completed->get_future();
    owner.InvokeOwner([&] {
        lease = AcquireVulkanGeneration(generation, stream);
        CHECK(lease && lease.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        auto const& planes = lease.Planes();
        auto plane = std::find_if(planes.begin(), planes.end(), [&](auto const& p) {
            return p.metadata.name == planeName;
        });
        CHECK(plane != planes.end());
        uint32_t const size = plane->bytes;
        reads->watch = service->Reserve(reads, [reads, completed, &owner](VkResult proof) {
            CHECK(owner.IsExecutingOwner() && proof == VK_SUCCESS);
            CHECK(reads->staging->PollComplete() == VK_SUCCESS);
            void* mapped = nullptr;
            CHECK(vkMapMemory(reads->native->device, reads->staging->memory(), 0,
                reads->bytes.size(), 0, &mapped) == VK_SUCCESS);
            std::memcpy(reads->bytes.data(), mapped, reads->bytes.size());
            vkUnmapMemory(reads->native->device, reads->staging->memory());
            CHECK(reads->watch->Retire()); reads->watch.reset(); reads->got = true;
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
        reads->bytes.resize(size);
        VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size = size;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        reads->staging = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Scratch); CHECK(reads->staging);
        CHECK(reads->staging->MarkSubmitted(reads->fence, reads) == VK_SUCCESS);
        VkBufferCopy copy{0, 0, size}; vkCmdCopyBuffer(reads->command, plane->buffer, reads->staging->buffer(), 1, &copy);
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
    lease.Complete();
    owner.InvokeOwner([&] { CHECK(service->Arm(*reads->watch, 1)); });
    CHECK(done.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    owner.Drain();
    CHECK(reads->got);
    return reads->bytes;
}
} // namespace

int main(int argc, char** argv) {
    CHECK(argc == 11);
    std::ifstream shader(argv[1], std::ios::binary); CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!raw.empty() && raw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> spirv(raw.size() / sizeof(uint32_t));
    std::memcpy(spirv.data(), raw.data(), raw.size());
    std::ifstream lengthShader(argv[2], std::ios::binary); CHECK(lengthShader);
    std::vector<char> lengthRaw((std::istreambuf_iterator<char>(lengthShader)), {});
    CHECK(!lengthRaw.empty() && lengthRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> lengthSpirv(lengthRaw.size() / sizeof(uint32_t));
    std::memcpy(lengthSpirv.data(), lengthRaw.data(), lengthRaw.size());
    std::ifstream blendShader(argv[3], std::ios::binary); CHECK(blendShader);
    std::vector<char> blendRaw((std::istreambuf_iterator<char>(blendShader)), {});
    CHECK(!blendRaw.empty() && blendRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> blendSpirv(blendRaw.size() / sizeof(uint32_t));
    std::memcpy(blendSpirv.data(), blendRaw.data(), blendRaw.size());
    std::ifstream setShader(argv[4], std::ios::binary); CHECK(setShader);
    std::vector<char> setRaw((std::istreambuf_iterator<char>(setShader)), {});
    CHECK(!setRaw.empty() && setRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> setSpirv(setRaw.size() / sizeof(uint32_t));
    std::memcpy(setSpirv.data(), setRaw.data(), setRaw.size());
    std::ifstream cutShader(argv[5], std::ios::binary); CHECK(cutShader);
    std::vector<char> cutRaw((std::istreambuf_iterator<char>(cutShader)), {});
    CHECK(!cutRaw.empty() && cutRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> cutSpirv(cutRaw.size() / sizeof(uint32_t));
    std::memcpy(cutSpirv.data(), cutRaw.data(), cutRaw.size());
    std::ifstream reparamShader(argv[6], std::ios::binary); CHECK(reparamShader);
    std::vector<char> reparamRaw((std::istreambuf_iterator<char>(reparamShader)), {});
    CHECK(!reparamRaw.empty() && reparamRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> reparamSpirv(reparamRaw.size() / sizeof(uint32_t));
    std::memcpy(reparamSpirv.data(), reparamRaw.data(), reparamRaw.size());
    std::ifstream minimumShader(argv[7], std::ios::binary); CHECK(minimumShader);
    std::vector<char> minimumRaw((std::istreambuf_iterator<char>(minimumShader)), {});
    CHECK(!minimumRaw.empty() && minimumRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> minimumSpirv(minimumRaw.size() / sizeof(uint32_t));
    std::memcpy(minimumSpirv.data(), minimumRaw.data(), minimumRaw.size());
    std::ifstream literalShader(argv[8], std::ios::binary); CHECK(literalShader);
    std::vector<char> literalRaw((std::istreambuf_iterator<char>(literalShader)), {});
    CHECK(!literalRaw.empty() && literalRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> literalSpirv(literalRaw.size() / sizeof(uint32_t));
    std::memcpy(literalSpirv.data(), literalRaw.data(), literalRaw.size());
    std::ifstream noiseValShader(argv[9], std::ios::binary); CHECK(noiseValShader);
    std::vector<char> noiseValRaw((std::istreambuf_iterator<char>(noiseValShader)), {});
    CHECK(!noiseValRaw.empty() && noiseValRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> noiseValidateSpirv(noiseValRaw.size() / sizeof(uint32_t));
    std::memcpy(noiseValidateSpirv.data(), noiseValRaw.data(), noiseValRaw.size());
    std::ifstream noiseGenShader(argv[10], std::ios::binary); CHECK(noiseGenShader);
    std::vector<char> noiseGenRaw((std::istreambuf_iterator<char>(noiseGenShader)), {});
    CHECK(!noiseGenRaw.empty() && noiseGenRaw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> noiseGenerateSpirv(noiseGenRaw.size() / sizeof(uint32_t));
    std::memcpy(noiseGenerateSpirv.data(), noiseGenRaw.data(), noiseGenRaw.size());
    bool unavailable = false; auto probe = CreateNative(&unavailable); if (unavailable) return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures supported{};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{}; features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2; features.pNext = &supported;
    vkGetPhysicalDeviceFeatures2(probe->physical, &features); if (!supported.timelineSemaphore) return 77;
    // FP64 is required only by the noise pipeline; enable it on the logical
    // device solely when the physical device supports it, keeping this test
    // portable on GPUs without shaderFloat64 (the noise case then skips).
    VkPhysicalDeviceFeatures physicalFeatures{};
    vkGetPhysicalDeviceFeatures(probe->physical, &physicalFeatures);
    bool const hasFp64 = physicalFeatures.shaderFloat64;
    probe.reset(); supported.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceFeatures core{};
    core.shaderFloat64 = hasFp64 ? VK_TRUE : VK_FALSE;
    auto native = CreateNative(&unavailable, {}, &supported, hasFp64 ? &core : nullptr); CHECK(native);
    DeviceContext::CreateInfo info; info.instance = native->instance; info.physicalDevice = native->physical;
    info.device = native->device; info.computeQueue = native->queue; info.computeQueueFamily = native->family;
    info.physicalIndex = native->physicalIndex; info.resourceDeviceId = 7044; info.nativeLifetime = native;
    info.timelineSemaphoreEnabled = true; info.resources = {size_t{16} << 20, 0};
    info.shaderFloat64Enabled = hasFp64;
    auto context = DeviceContext::Create(info); CHECK(context);
    UsdGenExecutionRuntime runtime{3}; auto owner = std::make_shared<UsdGenExecutionPipeline>(runtime);
    auto service = VulkanCompletionService::Create({context, owner.get(), 8}); CHECK(service);
    std::string reason;
    auto domain = VulkanGenerationAdapterDomain::Create({owner, service}, &reason); CHECK(domain);
    auto pipeline = WidthPipeline::Create(context, spirv); CHECK(pipeline);
    auto lengthPipeline = LengthScalePipeline::Create(context, lengthSpirv); CHECK(lengthPipeline);
    auto oldCutPipeline = LengthScalePipeline::CreateWithCutExtend(
        context, lengthSpirv, setSpirv, cutSpirv); CHECK(oldCutPipeline);
    CHECK(oldCutPipeline->HasCutExtend() && !oldCutPipeline->HasReparam());
    auto reparamPipeline = LengthScalePipeline::CreateWithReparam(
        context, lengthSpirv, setSpirv, cutSpirv, reparamSpirv); CHECK(reparamPipeline);
    CHECK(reparamPipeline->HasSet() && reparamPipeline->HasCutExtend() && reparamPipeline->HasReparam() &&
          !reparamPipeline->HasMinimum());
    auto blendPipeline = WidthBlendPipeline::Create(context, blendSpirv); CHECK(blendPipeline);
    auto preparer = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "vulkan-plan-executor", 7044);
    VkResult noiseStatus = VK_SUCCESS;
    std::shared_ptr<NoisePipeline> noisePipeline;
    if (hasFp64) {
        noisePipeline = NoisePipeline::Create(context, noiseValidateSpirv, noiseGenerateSpirv, &noiseStatus);
        CHECK(noisePipeline && noiseStatus == VK_SUCCESS);
    }
    auto executor = VulkanPlanExecutor::Create({domain, pipeline, preparer, lengthPipeline, blendPipeline}, &reason); CHECK(executor);
    std::shared_ptr<VulkanPlanExecutor> noiseExecutor;
    if (noisePipeline) {
        VulkanPlanExecutor::CreateInfo noiseCI;
        noiseCI.domain = domain;
        noiseCI.widthPipeline = pipeline;
        noiseCI.preparer = preparer;
        noiseCI.lengthPipeline = lengthPipeline;
        noiseCI.widthBlendPipeline = blendPipeline;
        noiseCI.noisePipeline = noisePipeline;
        noiseExecutor = VulkanPlanExecutor::Create(noiseCI, &reason);
        CHECK(noiseExecutor);
    }
    auto oldCutExecutor = VulkanPlanExecutor::Create(
        {domain, pipeline, preparer, oldCutPipeline, blendPipeline}, &reason); CHECK(oldCutExecutor);
    auto reparamExecutor = VulkanPlanExecutor::Create(
        {domain, pipeline, preparer, reparamPipeline, blendPipeline}, &reason); CHECK(reparamExecutor);
    auto minimumPlanDesc = MakeDescriptor(true, true, 0.0f, false, 17);
    UsdGenNodeDesc minimumNode;
    minimumNode.path = SdfPath("/Executor/Ops/minimum");
    minimumNode.type = TfToken("UsdGenLength");
    minimumNode.inputs = {minimumPlanDesc.nodes[0].path};
    minimumNode.params = {{TfToken("length:value"), VtValue(.25f), false},
                          {TfToken("length:mode"), VtValue(TfToken("scale")), false},
                          {TfToken("minRemainingLength"), VtValue(3.0f), false}};
    minimumPlanDesc.nodes[1].inputs = {minimumNode.path};
    minimumPlanDesc.nodes.insert(minimumPlanDesc.nodes.begin() + 1, minimumNode);
    auto minimumPlanHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumPlanDesc), nullptr);
    CHECK(minimumPlanHandle);
    auto minimumPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumPlanHandle->Payload());
    CHECK(minimumPlan && minimumPlan->Steps().size() == 2 &&
          minimumPlan->Steps()[0].minRemainingLength == 3.0f);
    auto minimumReject = [&](std::shared_ptr<const VulkanSourceWidthPlan> const& plan) {
        auto before = context->resources()->Snapshot();
        uint32_t callbacks = 0;
        owner->InvokeOwner([&] {
            VulkanPlanExecutor::Request request;
            request.context = context;
            request.authoritativeRevisions =
                VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
            CHECK(!reparamExecutor->Submit(plan, std::move(request), [&](auto, auto) { ++callbacks; }));
            CHECK(owner->OutstandingCommands() == 0);
        });
        preparer->Drain(); owner->Drain();
        auto after = context->resources()->Snapshot();
        CHECK(callbacks == 0 && owner->OutstandingCommands() == 0 &&
              after.limitBytes == before.limitBytes && after.headroomBytes == before.headroomBytes &&
              after.usableBytes == before.usableBytes && after.usedBytes == before.usedBytes &&
              after.byKind == before.byKind);
    };
    minimumReject(minimumPlan);
    auto minimumSetDesc = minimumPlanDesc;
    for (auto& parameter : minimumSetDesc.nodes[1].params)
        if (parameter.name == TfToken("length:mode")) parameter.value = VtValue(TfToken("set"));
    auto minimumSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumSetDesc), nullptr); CHECK(minimumSetHandle);
    minimumReject(std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumSetHandle->Payload()));
    auto minimumCutDesc = minimumSetDesc;
    minimumCutDesc.nodes[1].params.push_back({TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    auto minimumCutHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumCutDesc), nullptr); CHECK(minimumCutHandle);
    minimumReject(std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumCutHandle->Payload()));
    minimumCutDesc.nodes[1].params.push_back({TfToken("rebuild"), VtValue(TfToken("reparam")), false});
    auto minimumReparamHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(minimumCutDesc), nullptr); CHECK(minimumReparamHandle);
    minimumReject(std::static_pointer_cast<const VulkanSourceWidthPlan>(minimumReparamHandle->Payload()));
    auto zeroMinimumDesc = minimumPlanDesc;
    for (auto& parameter : zeroMinimumDesc.nodes[1].params)
        if (parameter.name == TfToken("minRemainingLength")) parameter.value = VtValue(0.f);
    auto zeroMinimumHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(zeroMinimumDesc), nullptr); CHECK(zeroMinimumHandle);
    auto zeroMinimumPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(zeroMinimumHandle->Payload());
    auto zeroMinimumResult = SubmitAndWait(*owner, reparamExecutor, zeroMinimumPlan, context, 146,
                                           std::make_shared<unsigned>(146));
    CHECK(zeroMinimumResult.generation);
    owner->InvokeOwner([&] { zeroMinimumResult.generation.reset(); });

    // All previous Length modules are present. Only the versioned random
    // module is missing, so rejection cannot be attributed to an older gate.
    {
        auto oldMinimum = LengthScalePipeline::CreateWithMinimum(
            context, lengthSpirv, setSpirv, cutSpirv, reparamSpirv, minimumSpirv);
        CHECK(oldMinimum && oldMinimum->HasMinimum() && oldMinimum->HasReparam() &&
              oldMinimum->HasCutExtend() && oldMinimum->HasSet() && !oldMinimum->HasLiteralV1());
        auto noRandom = VulkanPlanExecutor::Create(
            {domain, pipeline, preparer, oldMinimum, blendPipeline}, &reason);
        CHECK(noRandom);
        for (auto mode : {TfToken("scale"), TfToken("set")}) {
            for (unsigned method = 0; method != 3; ++method) {
                for (float floor : {0.f, 3.f}) {
                    auto desc = minimumPlanDesc;
                    desc.nodes[1].seed = -17;
                    desc.nodes[1].params = {
                        {TfToken("length:value"), VtValue(1.f), false},
                        {TfToken("length:mode"), VtValue(mode), false},
                        {TfToken("length:method"), VtValue(TfToken(method ? "cutExtend" : "scale")), false},
                        {TfToken("rebuild"), VtValue(TfToken(method == 2 ? "reparam" : "keepParam")), false},
                        {TfToken("minRemainingLength"), VtValue(floor), false},
                        {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(.5f, 1.5f)), false}};
                    auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc)); CHECK(handle);
                    auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
                    CHECK(plan && plan->Steps().size() == 2 &&
                          plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale);
                    auto const before = context->resources()->Snapshot();
                    unsigned callbacks = 0;
                    owner->InvokeOwner([&] {
                        VulkanPlanExecutor::Request request;
                        request.context = context;
                        request.authoritativeRevisions =
                            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
                        CHECK(!noRandom->Submit(plan, std::move(request),
                            [&](auto, auto) { ++callbacks; }));
                        CHECK(owner->OutstandingCommands() == 0);
                    });
                    preparer->Drain(); owner->Drain();
                    auto const after = context->resources()->Snapshot();
                    CHECK(callbacks == 0 && owner->OutstandingCommands() == 0 &&
                          after.limitBytes == before.limitBytes && after.headroomBytes == before.headroomBytes &&
                          after.usableBytes == before.usableBytes && after.usedBytes == before.usedBytes &&
                          after.byKind == before.byKind);
                }
            }
        }
        owner->InvokeOwner([&] {
            std::vector<UsdGenExecutionPipeline::CommandTicket> credits;
            for (uint64_t i = 0; i != owner->CommandCapacity(); ++i) {
                auto credit = owner->ReserveCommandTicket(); CHECK(credit);
                credits.push_back(std::move(credit));
            }
            CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
            credits.clear(); CHECK(owner->OutstandingCommands() == 0);
        });
        auto neutral = minimumPlanDesc;
        neutral.nodes[1].seed = -17;
        neutral.nodes[1].params.push_back(
            {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(1, 1)), false});
        auto neutralHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(neutral)); CHECK(neutralHandle);
        auto neutralPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(neutralHandle->Payload());
        auto reuse = SubmitAndWait(*owner, noRandom, neutralPlan, context, 147,
                                   std::make_shared<unsigned>(147));
        CHECK(reuse.generation);
        owner->InvokeOwner([&] { reuse.generation.reset(); });
        noRandom->Shutdown(); CHECK(noRandom->IsShutdown());
    }

    // Nonidentity envelope needs its own module, even with every older
    // random/minimum/method module available and an exactly-zero envelope.
    {
        auto oldLiteral = LengthScalePipeline::CreateWithLiteralV1(
            context, lengthSpirv, setSpirv, cutSpirv, reparamSpirv, minimumSpirv, literalSpirv);
        CHECK(oldLiteral && oldLiteral->HasLiteralV1() && oldLiteral->HasMinimum() &&
              oldLiteral->HasReparam() && !oldLiteral->HasEnvelopeV1());
        auto noEnvelope = VulkanPlanExecutor::Create(
            {domain, pipeline, preparer, oldLiteral, blendPipeline}, &reason);
        CHECK(noEnvelope);
        for (float blend : {0.f, .5f}) for (auto mode : {TfToken("scale"), TfToken("set")}) {
            for (unsigned method = 0; method != 3; ++method) {
                auto desc = minimumPlanDesc;
                desc.nodes[1].seed = -17;
                desc.nodes[1].params = {
                    {TfToken("length:value"), VtValue(1.f), false},
                    {TfToken("length:mode"), VtValue(mode), false},
                    {TfToken("length:method"), VtValue(TfToken(method ? "cutExtend" : "scale")), false},
                    {TfToken("rebuild"), VtValue(TfToken(method == 2 ? "reparam" : "keepParam")), false},
                    {TfToken("minRemainingLength"), VtValue(3.f), false},
                    {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(.5f, 1.5f)), false},
                    {TfToken("mask"), VtValue(blend), false}};
                auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc)); CHECK(handle);
                auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
                CHECK(plan && plan->Steps().size() == 2 &&
                      plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale);
                auto const before = context->resources()->Snapshot();
                unsigned callbacks = 0;
                owner->InvokeOwner([&] {
                    VulkanPlanExecutor::Request request;
                    request.context = context;
                    request.authoritativeRevisions =
                        VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
                    CHECK(!noEnvelope->Submit(plan, std::move(request),
                        [&](auto, auto) { ++callbacks; }));
                    CHECK(owner->OutstandingCommands() == 0);
                });
                preparer->Drain(); owner->Drain();
                auto const after = context->resources()->Snapshot();
                CHECK(callbacks == 0 && owner->OutstandingCommands() == 0 &&
                      after.limitBytes == before.limitBytes && after.headroomBytes == before.headroomBytes &&
                      after.usableBytes == before.usableBytes && after.usedBytes == before.usedBytes &&
                      after.byKind == before.byKind);
            }
        }
        owner->InvokeOwner([&] {
            std::vector<UsdGenExecutionPipeline::CommandTicket> credits;
            for (uint64_t i = 0; i != owner->CommandCapacity(); ++i) {
                auto credit = owner->ReserveCommandTicket(); CHECK(credit);
                credits.push_back(std::move(credit));
            }
            CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
            credits.clear(); CHECK(owner->OutstandingCommands() == 0);
        });
        auto neutral = minimumPlanDesc;
        neutral.nodes[1].params.push_back(
            {TfToken("length:random"), VtValue(usdGen::UsdGenVec2f(.5f, 1.5f)), false});
        neutral.nodes[1].params.push_back({TfToken("mask"), VtValue(1.f), false});
        auto neutralHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(neutral)); CHECK(neutralHandle);
        auto neutralPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(neutralHandle->Payload());
        auto reuse = SubmitAndWait(*owner, noEnvelope, neutralPlan, context, 148,
                                   std::make_shared<unsigned>(148));
        CHECK(reuse.generation);
        owner->InvokeOwner([&] { reuse.generation.reset(); });
        noEnvelope->Shutdown(); CHECK(noEnvelope->IsShutdown());
    }

    // Capability/context admission rejects before retaining or scheduling work.
    CHECK(!VulkanPlanExecutor::Create({domain, {}, preparer}, &reason));
    auto firstDesc = MakeDescriptor(true, true, .25f, false, 17);
    auto firstHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(firstDesc), nullptr); CHECK(firstHandle);
    auto first = std::static_pointer_cast<const VulkanSourceWidthPlan>(firstHandle->Payload()); CHECK(first);
    owner->InvokeOwner([&] { VulkanPlanExecutor::Request invalid; invalid.generation = 1;
        CHECK(!executor->Submit(first, std::move(invalid), {})); });
    owner->InvokeOwner([&] { VulkanPlanExecutor::Request invalid;
        invalid.context = context; invalid.capabilityVersion = 0; invalid.generation = 1;
        CHECK(!executor->Submit(first, std::move(invalid), [](auto, auto) {})); });
    UsdGenExecutionPipeline::Cancellation cancelled;
    CHECK(CopyCancellation(*owner, &cancelled));
    CHECK(owner->CancelPending() != 0);
    owner->InvokeOwner([&] { CHECK(cancelled.Superseded()); VulkanPlanExecutor::Request invalid;
        invalid.context = context; invalid.generation = 1; invalid.cancellation = cancelled;
        CHECK(!executor->Submit(first, std::move(invalid), [](auto, auto) {})); });

    auto retained = std::make_shared<unsigned>(7); std::weak_ptr<unsigned> weakRetained = retained;
    auto firstResult = SubmitAndWait(*owner, executor, first, context, 101, std::move(retained));
    CHECK(weakRetained.expired());
    Oracle firstExpected; CHECK(CpuReference(firstDesc, &firstExpected));
    auto firstManual = ManualExpected(true, .25f, false); CHECK(Bytes(firstExpected.widths) == Bytes(firstManual.widths));
    CheckPlanes(native, context, service, *owner, firstResult.generation, firstExpected);

    auto lengthDesc = firstDesc;
    UsdGenNodeDesc lengthNode;
    lengthNode.path = SdfPath("/Executor/Ops/length");
    lengthNode.type = TfToken("UsdGenLength");
    lengthNode.inputs = {lengthDesc.nodes[0].path};
    lengthNode.params = {{TfToken("length:value"), VtValue(1.25), false}};
    lengthDesc.nodes[1].inputs = {lengthNode.path};
    lengthDesc.nodes.insert(lengthDesc.nodes.begin() + 1, lengthNode);
    AddAuthoredPlanes(&lengthDesc);
    auto lengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(lengthDesc), nullptr); CHECK(lengthHandle);
    auto lengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(lengthHandle->Payload()); CHECK(lengthPlan);

    // A literal Length `set` is a valid fixed-topology compiled plan, but
    // this production executor was deliberately constructed with only the
    // existing scale module (argv[2]).  Admission must reject it before the
    // source preparer, native allocation, or queue-owner return credit.
    auto lengthSetDesc = lengthDesc;
    lengthSetDesc.nodes[1].params.push_back(
        {TfToken("length:mode"), VtValue(TfToken("set")), false});
    auto lengthSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(lengthSetDesc), nullptr);
    CHECK(lengthSetHandle);
    auto lengthSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        lengthSetHandle->Payload());
    CHECK(lengthSetPlan && lengthSetPlan->Steps().size() == 2 &&
          lengthSetPlan->IntermediateCount() == 1 &&
          lengthSetPlan->Steps()[0].kind == VulkanSourceWidthStage::Kind::LengthScale &&
          lengthSetPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
          lengthSetPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::Width);
    CHECK(!lengthPipeline->HasSet());
    auto const beforeLengthSetAdmission = context->resources()->Snapshot();
    uint32_t lengthSetCallbacks = 0;
    owner->InvokeOwner([&] {
        VulkanPlanExecutor::Request request;
        request.context = context;
        // One Length intermediate sits strictly between Source and Width.
        request.authoritativeRevisions =
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
        CHECK(!executor->Submit(lengthSetPlan, std::move(request),
            [&](auto, auto) { ++lengthSetCallbacks; }));
        CHECK(owner->OutstandingCommands() == 0);
    });
    // Draining makes an accidental preparer/native handoff observable before
    // checking the ledger and callback boundary.
    preparer->Drain(); owner->Drain();
    auto const afterLengthSetAdmission = context->resources()->Snapshot();
    CHECK(lengthSetCallbacks == 0 && owner->OutstandingCommands() == 0 &&
          afterLengthSetAdmission.limitBytes == beforeLengthSetAdmission.limitBytes &&
          afterLengthSetAdmission.headroomBytes == beforeLengthSetAdmission.headroomBytes &&
          afterLengthSetAdmission.usableBytes == beforeLengthSetAdmission.usableBytes &&
          afterLengthSetAdmission.usedBytes == beforeLengthSetAdmission.usedBytes &&
          afterLengthSetAdmission.byKind == beforeLengthSetAdmission.byKind);

    // Cut/extend is a distinct native operation.  In particular, a cut set
    // must be gated by its own capability, not by the ordinary absolute-set
    // capability (this injected pipeline deliberately has neither).
    auto cutExtendScaleDesc = lengthDesc;
    cutExtendScaleDesc.nodes[1].params.push_back(
        {TfToken("length:method"), VtValue(TfToken("cutExtend")), false});
    cutExtendScaleDesc.nodes[1].params.push_back(
        {TfToken("rebuild"), VtValue(TfToken("keepParam")), false});
    auto cutExtendScaleHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cutExtendScaleDesc), nullptr);
    CHECK(cutExtendScaleHandle);
    auto cutExtendScalePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        cutExtendScaleHandle->Payload());
    CHECK(cutExtendScalePlan && cutExtendScalePlan->Steps().size() == 2 &&
          cutExtendScalePlan->Steps()[0].lengthMethod ==
              VulkanSourceWidthStage::LengthMethod::CutExtend &&
          cutExtendScalePlan->Steps()[0].lengthMode ==
              VulkanSourceWidthStage::LengthMode::Scale);
    auto cutExtendSetDesc = cutExtendScaleDesc;
    cutExtendSetDesc.nodes[1].params.push_back(
        {TfToken("length:mode"), VtValue(TfToken("set")), false});
    auto cutExtendSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(cutExtendSetDesc), nullptr);
    CHECK(cutExtendSetHandle);
    auto cutExtendSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        cutExtendSetHandle->Payload());
    CHECK(cutExtendSetPlan && cutExtendSetPlan->Steps().size() == 2 &&
          cutExtendSetPlan->Steps()[0].lengthMethod ==
              VulkanSourceWidthStage::LengthMethod::CutExtend &&
          cutExtendSetPlan->Steps()[0].lengthMode ==
              VulkanSourceWidthStage::LengthMode::Set);
    CHECK(!lengthPipeline->HasCutExtend());
    auto const beforeCutExtendAdmission = context->resources()->Snapshot();
    uint32_t cutExtendCallbacks = 0;
    auto rejectCutExtend = [&](std::shared_ptr<const VulkanSourceWidthPlan> const& plan) {
        owner->InvokeOwner([&] {
            VulkanPlanExecutor::Request request;
            request.context = context;
            request.authoritativeRevisions =
                VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
            CHECK(!executor->Submit(plan, std::move(request),
                [&](auto, auto) { ++cutExtendCallbacks; }));
            CHECK(owner->OutstandingCommands() == 0);
        });
        // A false Submit must not schedule source capture or consume a
        // resource lease/ticket; drain both potential ingress queues to make
        // an accidental handoff observable.
        preparer->Drain(); owner->Drain();
        auto const after = context->resources()->Snapshot();
        CHECK(cutExtendCallbacks == 0 && owner->OutstandingCommands() == 0 &&
              after.limitBytes == beforeCutExtendAdmission.limitBytes &&
              after.headroomBytes == beforeCutExtendAdmission.headroomBytes &&
              after.usableBytes == beforeCutExtendAdmission.usableBytes &&
              after.usedBytes == beforeCutExtendAdmission.usedBytes &&
              after.byKind == beforeCutExtendAdmission.byKind);
    };
    rejectCutExtend(cutExtendScalePlan);
    rejectCutExtend(cutExtendSetPlan);
    auto reparamScaleDesc = cutExtendScaleDesc;
    for (auto& param : reparamScaleDesc.nodes[1].params)
        if (param.name == TfToken("rebuild")) param.value = VtValue(TfToken("reparam"));
    auto reparamScaleHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reparamScaleDesc), nullptr);
    CHECK(reparamScaleHandle);
    auto reparamScalePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        reparamScaleHandle->Payload());
    CHECK(reparamScalePlan && reparamScalePlan->Steps()[0].lengthRebuild ==
          VulkanSourceWidthStage::LengthRebuild::Reparam);
    rejectCutExtend(reparamScalePlan);
    auto reparamSetDesc = cutExtendSetDesc;
    for (auto& param : reparamSetDesc.nodes[1].params)
        if (param.name == TfToken("rebuild")) param.value = VtValue(TfToken("reparam"));
    auto reparamSetHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reparamSetDesc), nullptr);
    CHECK(reparamSetHandle);
    auto reparamSetPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        reparamSetHandle->Payload());
    CHECK(reparamSetPlan && reparamSetPlan->Steps()[0].lengthRebuild ==
          VulkanSourceWidthStage::LengthRebuild::Reparam &&
          reparamSetPlan->Steps()[0].lengthMode == VulkanSourceWidthStage::LengthMode::Set);
    rejectCutExtend(reparamSetPlan);
    // This executor has the legacy cut/set module but deliberately lacks the
    // reparam module.  Reparam is a separate gate: reject both modes before
    // source capture, callbacks, owner-credit use, or resource allocation.
    auto rejectOldCutReparam = [&](std::shared_ptr<const VulkanSourceWidthPlan> const& plan) {
        auto const before = context->resources()->Snapshot();
        uint32_t callbacks = 0;
        owner->InvokeOwner([&] {
            VulkanPlanExecutor::Request request;
            request.context = context;
            request.authoritativeRevisions =
                VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
            CHECK(!oldCutExecutor->Submit(plan, std::move(request),
                [&](auto, auto) { ++callbacks; }));
            CHECK(owner->OutstandingCommands() == 0);
        });
        preparer->Drain(); owner->Drain();
        auto const after = context->resources()->Snapshot();
        CHECK(callbacks == 0 && owner->OutstandingCommands() == 0 &&
              after.limitBytes == before.limitBytes && after.headroomBytes == before.headroomBytes &&
              after.usableBytes == before.usableBytes && after.usedBytes == before.usedBytes &&
              after.byKind == before.byKind);
        owner->InvokeOwner([&] {
            std::vector<UsdGenExecutionPipeline::CommandTicket> credits;
            credits.reserve(owner->CommandCapacity());
            for (uint64_t i = 0; i != owner->CommandCapacity(); ++i) {
                auto credit = owner->ReserveCommandTicket(); CHECK(credit);
                credits.push_back(std::move(credit));
            }
            CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
            credits.clear(); CHECK(owner->OutstandingCommands() == 0);
        });
    };
    rejectOldCutReparam(reparamScalePlan);
    rejectOldCutReparam(reparamSetPlan);
    // The same executor remains usable for its supported keep-param cut
    // operation, proving the rejection was specific to reparam capability.
    auto oldCutKeepParam = SubmitAndWait(*owner, oldCutExecutor, cutExtendScalePlan, context, 106,
                                         std::make_shared<unsigned>(106));
    CHECK(oldCutKeepParam.generation);
    owner->InvokeOwner([&] { oldCutKeepParam.generation.reset(); });
    auto radialReparamDesc = lengthDesc;
    radialReparamDesc.nodes[1].params.push_back(
        {TfToken("rebuild"), VtValue(TfToken("reparam")), false});
    auto radialReparamHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(radialReparamDesc), nullptr);
    CHECK(radialReparamHandle);
    auto radialReparamPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        radialReparamHandle->Payload());
    CHECK(radialReparamPlan && radialReparamPlan->Steps()[0].lengthMethod ==
          VulkanSourceWidthStage::LengthMethod::Scale &&
          radialReparamPlan->Steps()[0].lengthRebuild ==
              VulkanSourceWidthStage::LengthRebuild::Reparam);
    auto radialReparamResult = SubmitAndWait(*owner, executor, radialReparamPlan, context, 108,
                                             std::make_shared<unsigned>(108));
    CHECK(radialReparamResult.generation);
    owner->InvokeOwner([&] { radialReparamResult.generation.reset(); });
    // Rejection leaves the injected executor reusable for an admitted scale.
    auto cutAdmissionReuse = SubmitAndWait(*owner, executor, lengthPlan, context, 107,
                                           std::make_shared<unsigned>(107));
    owner->InvokeOwner([&] { cutAdmissionReuse.generation.reset(); });
    // A scale module alone also cannot admit a positive-threshold Length:
    // lowering adds a real Cull stage requiring its own native capability.
    auto thresholdDesc = lengthDesc;
    thresholdDesc.nodes[1].params.push_back({TfToken("cullThreshold"), VtValue(.25f), false});
    auto thresholdHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(thresholdDesc)); CHECK(thresholdHandle);
    auto thresholdPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(thresholdHandle->Payload());
    CHECK(thresholdPlan && thresholdPlan->Steps().size() == 3 && thresholdPlan->IntermediateCount() == 2 &&
          thresholdPlan->Steps()[1].kind == VulkanSourceWidthStage::Kind::LengthCull);
    owner->InvokeOwner([&] {
        VulkanPlanExecutor::Request request; request.context = context;
        request.authoritativeRevisions =
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 44, {42,43}};
        CHECK(!executor->Submit(thresholdPlan, std::move(request),
            [&](auto, auto) { ++lengthSetCallbacks; }));
    });
    preparer->Drain(); owner->Drain();
    CHECK(lengthSetCallbacks == 0 && owner->OutstandingCommands() == 0 &&
          context->resources()->Snapshot().usedBytes == beforeLengthSetAdmission.usedBytes &&
          context->resources()->Snapshot().byKind == beforeLengthSetAdmission.byKind);
    // Prove the rejection did not consume a bounded queue-owner credit.
    owner->InvokeOwner([&] {
        std::vector<UsdGenExecutionPipeline::CommandTicket> credits;
        credits.reserve(owner->CommandCapacity());
        for (uint64_t i = 0; i != owner->CommandCapacity(); ++i) {
            auto credit = owner->ReserveCommandTicket();
            CHECK(credit); credits.push_back(std::move(credit));
        }
        CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
        credits.clear();
        CHECK(owner->OutstandingCommands() == 0);
    });

    // Length admission is rejected before operation/ticket reservation,
    // source capture, or any readback when authoritative intermediate
    // revisions are absent or malformed.
    auto const beforeLengthRejects = context->resources()->Snapshot().usedBytes;
    uint32_t lengthRejectCallbacks = 0;
    auto rejectLength = [&](VulkanPlanExecutor::Request request) {
        owner->InvokeOwner([&] {
            CHECK(!executor->Submit(lengthPlan, std::move(request),
                [&](auto, auto) { ++lengthRejectCallbacks; }));
        });
        CHECK(lengthRejectCallbacks == 0);
        CHECK(owner->OutstandingCommands() == 0);
        CHECK(context->resources()->Snapshot().usedBytes == beforeLengthRejects);
    };
    VulkanPlanExecutor::Request missingRevisions;
    missingRevisions.context = context;
    rejectLength(std::move(missingRevisions));
    for (auto revisions : {
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {}},
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42, 43}},
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 42, {41}},
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {40}}}) {
        VulkanPlanExecutor::Request malformed;
        malformed.context = context; malformed.authoritativeRevisions = revisions;
        rejectLength(std::move(malformed));
    }
    auto noLengthExecutor = VulkanPlanExecutor::Create({domain, pipeline, preparer}, &reason);
    CHECK(noLengthExecutor);
    VulkanPlanExecutor::Request noPipeline;
    noPipeline.context = context;
    noPipeline.authoritativeRevisions = VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
    owner->InvokeOwner([&] {
        CHECK(!noLengthExecutor->Submit(lengthPlan, std::move(noPipeline),
            [&](auto, auto) { ++lengthRejectCallbacks; }));
    });
    CHECK(lengthRejectCallbacks == 0 && context->resources()->Snapshot().usedBytes == beforeLengthRejects);
    CHECK(owner->OutstandingCommands() == 0);
    noLengthExecutor->Shutdown();
    auto foreignContext = DeviceContext::Create(info); CHECK(foreignContext);
    VulkanPlanExecutor::Request wrongContext;
    wrongContext.context = foreignContext;
    wrongContext.authoritativeRevisions = VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 43, {42}};
    owner->InvokeOwner([&] {
        CHECK(!executor->Submit(lengthPlan, std::move(wrongContext),
            [&](auto, auto) { ++lengthRejectCallbacks; }));
    });
    CHECK(lengthRejectCallbacks == 0 && context->resources()->Snapshot().usedBytes == beforeLengthRejects);
    CHECK(owner->OutstandingCommands() == 0);

    // A Length-to-WidthBlend fan-in crosses non-width bundle origins.  The
    // plan compiles as proof-required, but this executor intentionally has no
    // compare module. Reject it at Submit admission, before source capture,
    // tickets/owner credits, resource allocation, or a completion callback.
    auto lengthBlendDesc = MakeLengthBlendDescriptor();
    auto lengthBlendHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(lengthBlendDesc), nullptr);
    CHECK(lengthBlendHandle);
    auto lengthBlendPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        lengthBlendHandle->Payload());
    CHECK(lengthBlendPlan && std::any_of(lengthBlendPlan->Steps().begin(),
        lengthBlendPlan->Steps().end(), [](auto const& stage) {
            return stage.requiresNonWidthProof;
        }));
    auto const beforeMissingComparator = context->resources()->Snapshot().usedBytes;
    uint32_t missingComparatorCallbacks = 0;
    owner->InvokeOwner([&] {
        VulkanPlanExecutor::Request request;
        request.context = context;
        request.authoritativeRevisions =
            VulkanPlanExecutor::Request::AuthoritativeRevisions{40, 41, 44, {42, 43}};
        CHECK(!executor->Submit(lengthBlendPlan, std::move(request),
            [&](auto, auto) { ++missingComparatorCallbacks; }));
    });
    preparer->Drain(); owner->Drain();
    CHECK(missingComparatorCallbacks == 0);
    CHECK(owner->OutstandingCommands() == 0);
    CHECK(context->resources()->Snapshot().usedBytes == beforeMissingComparator);
    // A neutral Length still establishes a distinct DAG geometry origin for
    // Vulkan proof admission. The CPU oracle verifies that every authored
    // named payload survives the same Source -> Length -> WidthBlend path.
    auto namedLengthBlendDesc = MakeLengthBlendDescriptor();
    AddAuthoredPlanes(&namedLengthBlendDesc);
    auto namedLengthBlendHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(namedLengthBlendDesc));
    CHECK(namedLengthBlendHandle);
    auto namedLengthBlendPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(
        namedLengthBlendHandle->Payload());
    CHECK(namedLengthBlendPlan && std::any_of(namedLengthBlendPlan->Steps().begin(),
        namedLengthBlendPlan->Steps().end(), [](auto const& stage) {
            return stage.requiresNonWidthProof;
        }));
    Oracle namedLengthBlendExpected;
    CHECK(CpuReference(namedLengthBlendDesc, &namedLengthBlendExpected));
    CHECK(namedLengthBlendExpected.named.size() == 6);

    auto lengthResult = SubmitAndWait(*owner, executor, lengthPlan, context, 108, std::make_shared<unsigned>(14));
    Oracle lengthExpected; CHECK(CpuReference(lengthDesc, &lengthExpected));
    CheckPlanes(native, context, service, *owner, lengthResult.generation, lengthExpected);
    CHECK(lengthResult.generation->Geometry().valueVersion == 43 &&
          lengthResult.generation->Geometry().topologyVersion == 40);
    for (int factor = 0; factor <= 1; ++factor) {
        auto boundary = lengthDesc;
        boundary.nodes[1].params[0].value = VtValue(double(factor));
        auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundary)); CHECK(handle);
        auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
        auto result = SubmitAndWait(*owner, executor, plan, context, 120 + factor, std::make_shared<unsigned>(20 + factor));
        Oracle expected; CHECK(CpuReference(boundary, &expected));
        CheckPlanes(native, context, service, *owner, result.generation, expected);
        CheckPlanes(native, context, service, *owner, firstResult.generation, firstExpected);
        owner->InvokeOwner([&] { result.generation.reset(); });
    }

    auto fanoutDesc = MakeFanoutDescriptor();
    AddAuthoredPlanes(&fanoutDesc);
    auto fanoutHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(fanoutDesc), nullptr); CHECK(fanoutHandle);
    auto fanoutPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(fanoutHandle->Payload()); CHECK(fanoutPlan);
    CHECK(fanoutPlan->Steps().size() == 3 && fanoutPlan->IntermediateCount() == 2);
    auto fanoutResult = SubmitAndWait(*owner, executor, fanoutPlan, context, 109, std::make_shared<unsigned>(15));
    Oracle fanoutExpected; CHECK(CpuReference(fanoutDesc, &fanoutExpected));
    CheckPlanes(native, context, service, *owner, fanoutResult.generation, fanoutExpected);
    CheckPlanes(native, context, service, *owner, firstResult.generation, firstExpected);

    // Bound C3s take the production CurveLoader path, including stable-ID
    // canonicalization of ragged curves, root-frame capture, and named-plane
    // gathering.  The two root channels are public; the captured T/B/N frame
    // must remain executor-private and therefore does not add public planes.
    auto boundDesc = MakeRootBoundDescriptor(true, TfToken("onError"));
    auto boundHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundDesc), nullptr); CHECK(boundHandle);
    auto boundPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(boundHandle->Payload());
    CHECK(boundPlan && boundPlan->HasRootBindings());
    auto boundResult = SubmitAndWait(*owner, executor, boundPlan, context, 130,
                                     std::make_shared<unsigned>(30));
    Oracle boundExpected; CHECK(CpuReference(boundDesc, &boundExpected));
    CHECK(boundExpected.ids == std::vector<uint64_t>({10,20,30}) &&
          boundExpected.rootPrim.size() == 3 && boundExpected.rootUV.size() == 3 &&
          boundExpected.named.size() == 6);
    CheckPlanes(native, context, service, *owner, boundResult.generation, boundExpected);
    CHECK(!boundResult.generation->Geometry().alreadyDeformed);

    // Admission rollback must happen before rooted source capture.  Fill the
    // queue-owner's bounded return ingress while already executing its owner
    // frame: no worker is blocked, but Submit cannot reserve its return
    // ticket.  The retained rooted plan is the exact immutable snapshot that
    // was just published above, so the later retry also proves caller-side
    // descriptor/surface mutations cannot leak into capture.
    auto const rootedRejectLedger = context->resources()->Snapshot();
    uint32_t rootedRejectCallbacks = 0;
    auto rejectedLifetime = std::make_shared<unsigned>(35);
    std::weak_ptr<unsigned> rejectedLifetimeWeak = rejectedLifetime;
    owner->InvokeOwner([&] {
        std::vector<UsdGenExecutionPipeline::CommandTicket> heldTickets;
        heldTickets.reserve(owner->CommandCapacity());
        for (uint64_t i = 0; i != owner->CommandCapacity(); ++i) {
            auto ticket = owner->ReserveCommandTicket();
            CHECK(ticket); heldTickets.push_back(std::move(ticket));
        }
        CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
        {
            VulkanPlanExecutor::Request request;
            request.context = context; request.generation = 135;
            request.tool = {"vulkan-plan-executor", 9, 135};
            request.requestLifetime = std::move(rejectedLifetime);
            CHECK(!executor->Submit(boundPlan, std::move(request),
                [&](auto, auto) { ++rootedRejectCallbacks; }));
        }
        CHECK(rootedRejectCallbacks == 0 && rejectedLifetimeWeak.expired());
        CHECK(owner->OutstandingCommands() == owner->CommandCapacity());
        auto const afterReject = context->resources()->Snapshot();
        CHECK(afterReject.usedBytes == rootedRejectLedger.usedBytes &&
              afterReject.byKind == rootedRejectLedger.byKind);
        heldTickets.clear();
        CHECK(owner->OutstandingCommands() == 0);
    });
    // A fresh lease on this exact owner is the observable operation-credit
    // balance after the rejected pre-capture admission.
    owner->InvokeOwner([&] {
        auto releasedOperation = domain->AcquireOperation();
        CHECK(releasedOperation && releasedOperation.QueueOwner().get() == owner.get());
        CHECK(releasedOperation.CancelBeforeSubmit());
    });
    CHECK(owner->OutstandingCommands() == 0);
    CheckPlanes(native, context, service, *owner, boundResult.generation, boundExpected);

    // Mutate every caller-owned rooted input only after compilation.  Retrying
    // the same captured plan must still agree with the oracle from its frozen
    // descriptor/surface snapshot, while the previous publication remains
    // readable after the rejected admission and after the retry.
    auto pristineBoundDesc = boundDesc;
    boundDesc.curveSets.front().points[0] = {-99, -98, -97};
    boundDesc.curveSets.front().rootFrame[0][3][0] = -77.0;
    boundDesc.surfaces.front().points[0] = {-66, -65, -64};
    boundDesc.surfaces.front().restPoints[0] = {-55, -54, -53};
    boundDesc.surfaces.front().faceVertexIndices = {0, 0, 0};
    boundDesc.nodes[1].params[0].value = VtValue(123.0f);
    auto rootedRetry = SubmitAndWait(*owner, executor, boundPlan, context, 135,
                                     std::make_shared<unsigned>(36));
    CheckPlanes(native, context, service, *owner, rootedRetry.generation, boundExpected);
    CheckPlanes(native, context, service, *owner, boundResult.generation, boundExpected);
    boundDesc = std::move(pristineBoundDesc);

    // `always` repairs bindings against the resolved rest surface.  It must
    // retain the same public root schema and canonical named-channel gather.
    auto reboundDesc = MakeRootBoundDescriptor(false, TfToken("always"));
    auto reboundHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(reboundDesc), nullptr); CHECK(reboundHandle);
    auto reboundPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(reboundHandle->Payload());
    CHECK(reboundPlan && reboundPlan->HasRootBindings());
    auto reboundResult = SubmitAndWait(*owner, executor, reboundPlan, context, 131,
                                       std::make_shared<unsigned>(31));
    Oracle reboundExpected; CHECK(CpuReference(reboundDesc, &reboundExpected));
    CHECK(reboundExpected.rootPrim.size() == reboundExpected.ids.size() &&
          reboundExpected.named.size() == 6);
    CheckPlanes(native, context, service, *owner, reboundResult.generation, reboundExpected);
    CHECK(reboundResult.generation->Geometry().alreadyDeformed);

    // A `never` source drops only its invalid root.  The ragged canonical
    // order and all named planes must follow the two surviving stable IDs.
    auto droppedDesc = MakeRootBoundDescriptor(true, TfToken("never"));
    droppedDesc.curveSets.front().skinPrim[0] = -1; // ID 30 is dropped after source-order capture.
    auto droppedHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(droppedDesc), nullptr); CHECK(droppedHandle);
    auto droppedPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(droppedHandle->Payload());
    CHECK(droppedPlan && droppedPlan->HasRootBindings());
    auto droppedResult = SubmitAndWait(*owner, executor, droppedPlan, context, 132,
                                       std::make_shared<unsigned>(32));
    Oracle droppedExpected; CHECK(CpuReference(droppedDesc, &droppedExpected));
    CHECK(droppedExpected.ids == std::vector<uint64_t>({10,20}) &&
          droppedExpected.rootPrim.size() == 2 && droppedExpected.named.size() == 6);
    CheckPlanes(native, context, service, *owner, droppedResult.generation, droppedExpected);

    // Root bindings must survive both an actual Length -> Width chain and a
    // WidthBlend fan-in.  Their GPU readback is compared against the existing
    // independent CPU compiler/scheduler oracle, not a hand-built root copy.
    auto boundLengthDesc = boundDesc;
    UsdGenNodeDesc boundLength;
    boundLength.path = SdfPath("/Executor/Ops/boundLength");
    boundLength.type = TfToken("UsdGenLength"); boundLength.inputs = {boundLengthDesc.nodes[0].path};
    boundLength.params = {{TfToken("length:value"), VtValue(1.25), false}};
    boundLengthDesc.nodes[1].inputs = {boundLength.path};
    boundLengthDesc.nodes.insert(boundLengthDesc.nodes.begin() + 1, boundLength);
    auto boundLengthHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundLengthDesc), nullptr); CHECK(boundLengthHandle);
    auto boundLengthPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(boundLengthHandle->Payload());
    CHECK(boundLengthPlan);
    auto boundLengthResult = SubmitAndWait(*owner, executor, boundLengthPlan, context, 133,
                                           std::make_shared<unsigned>(33));
    Oracle boundLengthExpected; CHECK(CpuReference(boundLengthDesc, &boundLengthExpected));
    CheckPlanes(native, context, service, *owner, boundLengthResult.generation, boundLengthExpected);

    auto boundBlendDesc = boundDesc;
    auto boundSource = boundBlendDesc.nodes[0];
    auto boundLeft = boundBlendDesc.nodes[1]; boundLeft.path = SdfPath("/Executor/Ops/rootWidthL");
    boundLeft.params[0].value = VtValue(.5f);
    auto boundRight = boundLeft; boundRight.path = SdfPath("/Executor/Ops/rootWidthR");
    boundRight.params[0].value = VtValue(1.5f);
    UsdGenNodeDesc boundBlend;
    boundBlend.path = SdfPath("/Executor/Ops/rootBlend"); boundBlend.type = TfToken("UsdGenWidthBlend");
    boundBlend.inputs = {boundLeft.path, boundRight.path};
    boundBlend.params = {{TfToken("widthBlend:weight"), VtValue(.25f), false}};
    boundBlendDesc.nodes = {boundSource, boundBlend, boundRight, boundLeft};
    boundBlendDesc.terminal = boundBlend.path;
    auto boundBlendHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(boundBlendDesc), nullptr); CHECK(boundBlendHandle);
    auto boundBlendPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(boundBlendHandle->Payload());
    CHECK(boundBlendPlan);
    auto boundBlendResult = SubmitAndWait(*owner, executor, boundBlendPlan, context, 134,
                                          std::make_shared<unsigned>(34));
    Oracle boundBlendExpected; CHECK(CpuReference(boundBlendDesc, &boundBlendExpected));
    CheckPlanes(native, context, service, *owner, boundBlendResult.generation, boundBlendExpected);

    // This is deliberately a source-local C3: only root-frame capture sees
    // the unequal affine spaces.  The actual native Source -> Length -> Width
    // submission must retain the six ordinary planes, two public root planes,
    // and six named planes exactly as the compiler/scheduler oracle does.
    auto affineDesc = MakeAffineRootBoundDescriptor();
    auto affineSourcePoints = affineDesc.curveSets.front().points;
    auto affineSourceRest = affineDesc.curveSets.front().rest;
    UsdGenNodeDesc affineLength;
    affineLength.path = SdfPath("/Executor/Ops/affineLength");
    affineLength.type = TfToken("UsdGenLength");
    affineLength.inputs = {affineDesc.nodes[0].path};
    affineLength.params = {{TfToken("length:value"), VtValue(1.25), false}};
    affineDesc.nodes[1].inputs = {affineLength.path};
    affineDesc.nodes.insert(affineDesc.nodes.begin() + 1, affineLength);
    auto affineHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(affineDesc), nullptr); CHECK(affineHandle);
    auto affinePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(affineHandle->Payload());
    CHECK(affinePlan && affinePlan->HasRootBindings() && affineDesc.curveSets.front().rootFrame.empty());
    auto affineResult = SubmitAndWait(*owner, executor, affinePlan, context, 136,
                                      std::make_shared<unsigned>(37));
    Oracle affineExpected; CHECK(CpuReference(affineDesc, &affineExpected));
    CHECK(affineExpected.rootPrim.size() == affineExpected.ids.size() &&
          affineExpected.named.size() == 6 &&
          affineDesc.curveSets.front().points == affineSourcePoints &&
          affineDesc.curveSets.front().rest == affineSourceRest);
    CheckPlanes(native, context, service, *owner, affineResult.generation, affineExpected);

    // The plan owns its source and surface snapshots.  Matrix edits after
    // compilation cannot change either a retry or the already published
    // generation; private derived-frame readback belongs to RootFrameCow.
    affineDesc.curveSets.front().worldMatrix[3][0] = -91.0;
    affineDesc.surfaces.front().worldMatrix[3][1] = 83.0;
    auto affineRetry = SubmitAndWait(*owner, executor, affinePlan, context, 137,
                                     std::make_shared<unsigned>(38));
    CheckPlanes(native, context, service, *owner, affineRetry.generation, affineExpected);
    CheckPlanes(native, context, service, *owner, affineResult.generation, affineExpected);
    // Noise is topology-preserving vector frizz (root T/B/N) with no CPU
    // oracle in this suite: prove the executor wiring by submitting the same
    // root-bound source with and without the noise op and requiring the GPU
    // points to differ while point count and the width plane are preserved.
    Result noiseResult, plainResult;
    if (noiseExecutor) {
    auto noiseDesc = MakeRootBoundDescriptor(true, TfToken("never"));
    UsdGenNodeDesc noiseNode;
    noiseNode.path = SdfPath("/Executor/Ops/noise");
    noiseNode.type = TfToken("UsdGenNoise");
    noiseNode.inputs = {noiseDesc.nodes[0].path};
    noiseNode.params = {
        {TfToken("noise:magnitude"), VtValue(0.1f), false},
        {TfToken("noise:frequency"), VtValue(1.7f), false},
        {TfToken("noise:correlation"), VtValue(0.4f), false},
        {TfToken("noise:octaves"), VtValue(3), false},
        {TfToken("noise:lacunarity"), VtValue(2.0f), false},
        {TfToken("noise:gain"), VtValue(0.5f), false},
        {TfToken("mask"), VtValue(1.0f), false},
        {TfToken("preserveLength"), VtValue(0.0f), false},
        {TfToken("cumulative"), VtValue(false), false},
        {TfToken("noise:seed"), VtValue(12345), false}};
    noiseDesc.nodes[1].inputs = {noiseNode.path};
    noiseDesc.nodes.insert(noiseDesc.nodes.begin() + 1, noiseNode);
    auto noiseHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(noiseDesc), nullptr); CHECK(noiseHandle);
    auto noisePlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(noiseHandle->Payload());
    CHECK(noisePlan && noisePlan->HasRootBindings());
    noiseResult = SubmitAndWait(*owner, noiseExecutor, noisePlan, context, 141,
                                     std::make_shared<unsigned>(41));
    CHECK(noiseResult.generation);
    // Baseline: the identical source with no noise op, published through the
    // main executor.  The points planes must differ once noise is admitted.
    auto plainDesc = MakeRootBoundDescriptor(true, TfToken("never"));
    auto plainHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(plainDesc), nullptr); CHECK(plainHandle);
    auto plainPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(plainHandle->Payload());
    CHECK(plainPlan && plainPlan->HasRootBindings());
    plainResult = SubmitAndWait(*owner, executor, plainPlan, context, 140,
                                     std::make_shared<unsigned>(40));
    CHECK(plainResult.generation);
    CHECK(noiseResult.generation->Geometry().pointCount ==
          plainResult.generation->Geometry().pointCount);
    auto plainPoints = ReadPlaneBytes(native, context, service, *owner,
                                     plainResult.generation, "points");
    auto noisePoints = ReadPlaneBytes(native, context, service, *owner,
                                     noiseResult.generation, "points");
    CHECK(!plainPoints.empty() && plainPoints.size() == noisePoints.size());
    CHECK(std::memcmp(plainPoints.data(), noisePoints.data(), plainPoints.size()) != 0);
    auto noiseWidth = ReadPlaneBytes(native, context, service, *owner,
                                     noiseResult.generation, "widths");
    CHECK(!noiseWidth.empty());
    std::printf("Noise executor wiring: points moved, width present\n");
    } // if (noiseExecutor)

    // A second plan uses the default-width path and useRest=false.  The prior
    // published immutable result must stay readable while this job publishes.
    auto secondDesc = MakeDescriptor(false, false, .5f, true, 18);
    // index identity deliberately ignores the authored (unsorted) IDs.
    secondDesc.nodes[0].params[1].value = VtValue(TfToken("index"));
    auto secondHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(secondDesc), nullptr); CHECK(secondHandle);
    auto second = std::static_pointer_cast<const VulkanSourceWidthPlan>(secondHandle->Payload()); CHECK(second);
    auto secondResult = SubmitAndWait(*owner, executor, second, context, 102, std::make_shared<unsigned>(8));
    Oracle secondExpected; CHECK(CpuReference(secondDesc, &secondExpected));
    auto secondManual = ManualExpected(false, .5f, true, true); CHECK(Bytes(secondExpected.widths) == Bytes(secondManual.widths));
    CheckPlanes(native, context, service, *owner, firstResult.generation, firstExpected);
    CheckPlanes(native, context, service, *owner, secondResult.generation, secondExpected);


    // `idSource=index` is authoritative: malformed authored primvar IDs are
    // deliberately ignored, rather than becoming a hidden validation path.
    auto shortIndexDesc = secondDesc;
    shortIndexDesc.curveSets.front().curveId = {999};
    auto shortIndexHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(shortIndexDesc), nullptr); CHECK(shortIndexHandle);
    auto shortIndex = std::static_pointer_cast<const VulkanSourceWidthPlan>(shortIndexHandle->Payload()); CHECK(shortIndex);
    auto shortIndexResult = SubmitAndWait(*owner, executor, shortIndex, context, 105, std::make_shared<unsigned>(11));
    Oracle shortIndexExpected; CHECK(CpuReference(shortIndexDesc, &shortIndexExpected));
    CheckPlanes(native, context, service, *owner, shortIndexResult.generation, shortIndexExpected);
    auto duplicateIndexDesc = secondDesc;
    duplicateIndexDesc.curveSets.front().curveId = {5, 5, 6};
    auto duplicateIndexHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(duplicateIndexDesc), nullptr); CHECK(duplicateIndexHandle);
    auto duplicateIndex = std::static_pointer_cast<const VulkanSourceWidthPlan>(duplicateIndexHandle->Payload()); CHECK(duplicateIndex);
    auto duplicateIndexResult = SubmitAndWait(*owner, executor, duplicateIndex, context, 106, std::make_shared<unsigned>(12));
    Oracle duplicateIndexExpected; CHECK(CpuReference(duplicateIndexDesc, &duplicateIndexExpected));
    CheckPlanes(native, context, service, *owner, duplicateIndexResult.generation, duplicateIndexExpected);

    // Empty C3 remains a real admitted publication, not a null-handle shortcut.
    auto emptyDesc = MakeDescriptor(false, false, .5f, true, 19);
    auto& empty = emptyDesc.curveSets.front(); empty.curveVertexCounts.clear(); empty.points.clear();
    empty.rest.clear(); empty.widths.clear(); empty.curveId.clear();
    auto emptyHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(emptyDesc), nullptr); CHECK(emptyHandle);
    auto emptyPlan = std::static_pointer_cast<const VulkanSourceWidthPlan>(emptyHandle->Payload()); CHECK(emptyPlan);
    auto emptyResult = SubmitAndWait(*owner, executor, emptyPlan, context, 103, std::make_shared<unsigned>(9));
    CHECK(emptyResult.generation->Geometry().elementCount == 0 && emptyResult.generation->Geometry().pointCount == 0);

    // Canonical empty C3 needs no rest sample even when useRest=true.
    auto badEmpty = emptyDesc; badEmpty.nodes[0].params[0].value = VtValue(true);
    auto emptyRestHandle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(badEmpty), nullptr); CHECK(emptyRestHandle);
    auto emptyRest = std::static_pointer_cast<const VulkanSourceWidthPlan>(emptyRestHandle->Payload()); CHECK(emptyRest);
    auto emptyRestResult = SubmitAndWait(*owner, executor, emptyRest, context, 104, std::make_shared<unsigned>(10));
    CHECK(emptyRestResult.generation->Geometry().elementCount == 0 && emptyRestResult.generation->Geometry().pointCount == 0);

    owner->InvokeOwner([&] { firstResult.generation.reset(); secondResult.generation.reset(); lengthResult.generation.reset(); fanoutResult.generation.reset(); boundResult.generation.reset(); rootedRetry.generation.reset(); reboundResult.generation.reset(); droppedResult.generation.reset(); boundLengthResult.generation.reset(); boundBlendResult.generation.reset(); affineResult.generation.reset(); affineRetry.generation.reset(); shortIndexResult.generation.reset(); duplicateIndexResult.generation.reset(); emptyResult.generation.reset(); emptyRestResult.generation.reset(); plainResult.generation.reset(); noiseResult.generation.reset(); });
    oldCutExecutor->Shutdown(); CHECK(oldCutExecutor->IsShutdown());
    executor->Shutdown(); CHECK(executor->IsShutdown());
    noiseExecutor->Shutdown(); CHECK(noiseExecutor->IsShutdown());
    owner->InvokeOwner([&] { CHECK(!executor->Submit(first, {}, {})); });
    CHECK(domain->Close()); service->CloseAndJoin(); owner->Drain(); preparer->Drain();
    CHECK(owner->CallbackFailures() == 0 && context->resources()->Snapshot().usedBytes == 0);
    return 0;
}
