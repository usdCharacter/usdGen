// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// End-to-end parity through the two production Session executors. Device
// readback below is confined to the test; no host geometry oracle is published.
#include "usdGenHydraCurves/publication.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "testBackend.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenUsd/lower.h"
#include "usdGen/session.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGenVulkan/vulkan/defaultProvider.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenVulkan/vulkan/externalRetirement.h"
#include "vulkanSurfaceCageFixture.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Session CUDA/Vulkan parity: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
std::vector<std::weak_ptr<DeviceContext>> previousContexts;

void RetirePreviousCase() {
    // Cases own independent Sessions. Bound their native-device footprint by
    // proving the preceding case released every observed context before the
    // next case starts. The separate Session lifecycle test deliberately
    // overlaps providers; retained generations within each case still overlap.
    if (previousContexts.empty()) return;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!std::all_of(previousContexts.begin(), previousContexts.end(),
                       [](auto const& context) { return context.expired(); })) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Weak expiry precedes the last native destructor. This FIFO barrier also
    // proves earlier cleanup captures have finished destroying their devices.
    auto reply = std::make_shared<std::promise<void>>();
    auto complete = reply->get_future();
    CHECK(EnqueueVulkanExternalRetirement([reply] { reply->set_value(); }));
    CHECK(complete.wait_until(deadline) == std::future_status::ready);
    previousContexts.clear();
}

struct Plane {
    UsdGenDeviceChannelMetadata metadata;
    std::vector<unsigned char> bytes;
};
struct Packet {
    UsdGenDeviceGeometryMetadata geometry;
    std::map<std::string, Plane> planes;
};
using DeviceGeneration = std::shared_ptr<const UsdGenDeviceGeneration>;

Packet ReadCuda(DeviceGeneration const& generation) {
    CHECK(generation && generation->Identity().backend == usdGenTestUtils::Backend("cuda"));
    Packet packet; packet.geometry = generation->Geometry();
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
    {
        auto lease = gpu::AcquireGeometry(generation, stream); CHECK(lease);
        auto view = lease.Geometry();
        for (auto const& metadata : generation->Channels()) {
            void const* pointer = nullptr;
            gpu::CudaNamedChannelLease named;
            switch (metadata.semantic) {
                case UsdGenDeviceChannelSemantic::Points: pointer = view.points.data; break;
                case UsdGenDeviceChannelSemantic::RestPoints: pointer = view.restPoints.data; break;
                case UsdGenDeviceChannelSemantic::Widths: pointer = view.widths.data; break;
                case UsdGenDeviceChannelSemantic::HairT: pointer = lease.HairT().data; break;
                case UsdGenDeviceChannelSemantic::ElementOffsets: pointer = view.curveOffsets.data; break;
                case UsdGenDeviceChannelSemantic::StableIds: pointer = view.stableIds.data; break;
                case UsdGenDeviceChannelSemantic::RootPrim: pointer = lease.RootPrim().data; break;
                case UsdGenDeviceChannelSemantic::RootUV: pointer = lease.RootUV().data; break;
                case UsdGenDeviceChannelSemantic::Generic:
                    named = gpu::AcquireNamedChannel(generation, metadata.name, stream);
                    CHECK(named); pointer = named.Bytes().data; break;
            }
            size_t bytes = 0; CHECK(UsdGenDeviceChannelStorageBytes(metadata, &bytes));
            Plane plane{metadata, std::vector<unsigned char>(bytes)};
            CHECK(!bytes || (pointer && cudaMemcpyAsync(plane.bytes.data(), pointer,
                bytes, cudaMemcpyDeviceToHost, stream) == cudaSuccess));
            // The named lease must outlive its submitted copy.
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            CHECK(packet.planes.emplace(metadata.name, std::move(plane)).second);
        }
    }
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    return packet;
}

Packet ReadVulkan(DeviceGeneration const& generation) {
    CHECK(generation && generation->Identity().backend == usdGenTestUtils::Backend("vulkan"));
    auto access = GetVulkanGenerationAccess(generation); CHECK(access);
    auto context = access.context; auto device = context->device();
    previousContexts.push_back(context);
    Packet packet; packet.geometry = generation->Geometry();
    VulkanGenerationLease lease;
    VkCommandPool pool = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<ChargedBuffer>> staging;
    std::vector<std::string> names;
    access.queueOwner->InvokeOwner([&] {
        lease = AcquireVulkanGeneration(generation,
            reinterpret_cast<UsdGenDeviceStream>(context->computeQueue()));
        CHECK(lease && lease.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        CHECK(lease.Planes().size() == generation->Channels().size());
        VkCommandPoolCreateInfo pi{}; pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.queueFamilyIndex = context->computeQueueFamily();
        CHECK(vkCreateCommandPool(device, &pi, nullptr, &pool) == VK_SUCCESS);
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        CHECK(vkCreateFence(device, &fi, nullptr, &fence) == VK_SUCCESS);
        VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(device, &ai, &command) == VK_SUCCESS);
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        CHECK(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
        VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &before, 0, nullptr, 0, nullptr);
        for (auto const& plane : lease.Planes()) {
            size_t bytes = 0; CHECK(UsdGenDeviceChannelStorageBytes(plane.metadata, &bytes));
            CHECK(bytes == plane.bytes);
            CHECK(packet.planes.emplace(plane.metadata.name,
                Plane{plane.metadata, std::vector<unsigned char>(bytes)}).second);
            if (!bytes) continue;
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto output = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            CHECK(output && output->MarkSubmitted(fence, generation) == VK_SUCCESS);
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(command, plane.buffer, output->buffer(), 1, &copy);
            names.push_back(plane.metadata.name); staging.push_back(std::move(output));
        }
        VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &after, 0, nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(command) == VK_SUCCESS);
        VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        CHECK(vkQueueSubmit(context->computeQueue(), 1, &submit, fence) == VK_SUCCESS);
    });
    CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 20000000000ull) == VK_SUCCESS);
    access.queueOwner->InvokeOwner([&] {
        for (size_t i = 0; i != staging.size(); ++i) {
            CHECK(staging[i]->PollComplete() == VK_SUCCESS);
            auto& bytes = packet.planes.at(names[i]).bytes;
            void* mapped = nullptr;
            CHECK(vkMapMemory(device, staging[i]->memory(), 0, bytes.size(), 0, &mapped) == VK_SUCCESS);
            std::memcpy(bytes.data(), mapped, bytes.size());
            vkUnmapMemory(device, staging[i]->memory());
        }
        lease.Complete(); staging.clear();
        vkDestroyFence(device, fence, nullptr); vkDestroyCommandPool(device, pool, nullptr);
    });
    return packet;
}

Plane const& Semantic(Packet const& packet, UsdGenDeviceChannelSemantic semantic) {
    auto found = std::find_if(packet.planes.begin(), packet.planes.end(), [&](auto const& value) {
        return value.second.metadata.semantic == semantic;
    });
    CHECK(found != packet.planes.end()); return found->second;
}
template<class T> std::vector<T> Values(Plane const& plane) {
    CHECK(plane.bytes.size() % sizeof(T) == 0);
    std::vector<T> values(plane.bytes.size() / sizeof(T));
    if (!values.empty()) std::memcpy(values.data(), plane.bytes.data(), plane.bytes.size());
    return values;
}
template<class T> std::vector<T> Values(Packet const& packet, UsdGenDeviceChannelSemantic semantic) {
    return Values<T>(Semantic(packet, semantic));
}

void Compare(Packet const& cuda, Packet const& vulkan, float pointTolerance = 0,
             bool generatedRest = false) {
    CHECK(cuda.geometry.elementCount == vulkan.geometry.elementCount);
    CHECK(cuda.geometry.pointCount == vulkan.geometry.pointCount);
    CHECK(cuda.geometry.alreadyDeformed == vulkan.geometry.alreadyDeformed);
    CHECK(UsdGenCurveLayoutTypeOf(cuda.geometry.layoutMetadata) == UsdGenCurveLayoutTypeOf(vulkan.geometry.layoutMetadata));
    CHECK(UsdGenCurveLayoutBasisOf(cuda.geometry.layoutMetadata) == UsdGenCurveLayoutBasisOf(vulkan.geometry.layoutMetadata));
    CHECK(UsdGenCurveLayoutWrapOf(cuda.geometry.layoutMetadata) == UsdGenCurveLayoutWrapOf(vulkan.geometry.layoutMetadata));
    if (cuda.planes.size() != vulkan.planes.size()) {
        std::fprintf(stderr, "CUDA planes:");
        for (auto const& plane : cuda.planes) std::fprintf(stderr, " %s", plane.first.c_str());
        std::fprintf(stderr, "\nVulkan planes:");
        for (auto const& plane : vulkan.planes) std::fprintf(stderr, " %s", plane.first.c_str());
        std::fprintf(stderr, "\n");
    }
    CHECK(cuda.planes.size() == vulkan.planes.size());
    for (auto const& entry : cuda.planes) {
        auto found = vulkan.planes.find(entry.first);
        if (found == vulkan.planes.end()) {
            std::fprintf(stderr, "Vulkan is missing CUDA plane %s; Vulkan planes:", entry.first.c_str());
            for (auto const& plane : vulkan.planes) std::fprintf(stderr, " %s", plane.first.c_str());
            std::fprintf(stderr, "\n");
        }
        CHECK(found != vulkan.planes.end());
        auto const& a = entry.second; auto const& b = found->second;
        CHECK(a.metadata.type == b.metadata.type && a.metadata.domain == b.metadata.domain &&
            a.metadata.elementCount == b.metadata.elementCount && a.metadata.arity == b.metadata.arity &&
            a.metadata.strideBytes == b.metadata.strideBytes && a.metadata.semantic == b.metadata.semantic &&
            a.metadata.readOnly && b.metadata.readOnly && a.bytes.size() == b.bytes.size());
        // Topology, stable IDs, inherited fields and binding/rest channels
        // are always bitwise. Arithmetic widths allow one small rounding
        // difference; nonlinear FBM/RBF/Grow points use the explicit case bound.
        float tolerance = a.metadata.semantic == UsdGenDeviceChannelSemantic::Widths ? 3.e-6f :
            (a.metadata.semantic == UsdGenDeviceChannelSemantic::Points ||
             (generatedRest && a.metadata.semantic == UsdGenDeviceChannelSemantic::RestPoints)) ? pointTolerance : 0;
        if (!tolerance) {
            if (a.bytes != b.bytes) {
                std::fprintf(stderr, "exact plane mismatch: %s\n", entry.first.c_str());
                for (size_t i = 0; i != a.bytes.size(); ++i) {
                    if (a.bytes[i] == b.bytes[i]) continue;
                    std::fprintf(stderr, "first differing byte: %zu\n", i);
                    if (a.metadata.type == UsdGenDeviceValueType::Float32) {
                        auto av = Values<float>(a); auto bv = Values<float>(b);
                        size_t const index = i / sizeof(float);
                        std::fprintf(stderr, "%s[%zu] CUDA=%.9g Vulkan=%.9g\n",
                            entry.first.c_str(), index, av[index], bv[index]);
                    }
                    break;
                }
            }
            CHECK(a.bytes == b.bytes); continue;
        }
        auto av = Values<float>(a); auto bv = Values<float>(b);
        for (size_t i = 0; i != av.size(); ++i) {
            bool withinTolerance = std::isfinite(av[i]) && std::isfinite(bv[i]) && std::fabs(av[i]-bv[i]) <= tolerance;
            if (!withinTolerance) std::fprintf(stderr, "%s[%zu] CUDA=%g Vulkan=%g tolerance=%g\n",
                entry.first.c_str(), i, av[i], bv[i], tolerance);
            CHECK(withinTolerance);
        }
    }
}

UsdGenNodeDesc Node(char const* path, char const* type, SdfPath input = {}) {
    UsdGenNodeDesc node; node.path = SdfPath(path); node.type = TfToken(type);
    if (!input.IsEmpty()) node.inputs = {input};
    return node;
}
UsdGenGraphDesc Source() {
    UsdGenGraphDesc desc; desc.description = SdfPath("/Parity"); desc.defaultWidth = 1.f;
    UsdGenSurfaceDesc surface; surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
    surface.points = surface.restPoints;
    for (auto& point : surface.points) point += GfVec3f(.25f,.5f,-.125f);
    surface.faceVertexCounts = {3,3,3}; surface.faceVertexIndices = {0,1,2,0,1,3,1,2,4};
    desc.surfaces = {surface};
    UsdGenCurveSetDesc curves; curves.path = SdfPath("/Curves");
    curves.role = UsdGenLane::Chunked; curves.curveRole = TfToken("hair");
    curves.type = TfToken("cubic"); curves.basis = TfToken("bspline"); curves.wrap = TfToken("pinned");
    curves.curveGeneration = 41; curves.curveVertexCounts = {2,3,2};
    curves.curveId = {99,42,11};
    curves.points = {{.25f,0,0},{.25f,.5f,0},{.5f,0,0},{.5f,.0625f,0},
                     {.5f,.125f,0},{.75f,0,0},{.75f,1,0}};
    curves.rest = curves.points; curves.widths = {1,2,3,4,5,6,7};
    curves.widthsInterpolation = TfToken("vertex");
    curves.skinPrim = {0,1,2}; curves.skinPrimUv = {{.125f,.25f},{.25f,.375f},{.375f,.5f}};
    curves.rootFrame = {GfMatrix4d(1),GfMatrix4d(1),GfMatrix4d(1)};
    UsdGenAuthoredPlaneDesc point; point.name = TfToken("density");
    point.type = UsdGenAuthoredPlaneType::Float32; point.domain = UsdGenAuthoredPlaneDomain::Point;
    point.arity = 1; point.floatValues = {10,11,20,21,22,30,31};
    UsdGenAuthoredPlaneDesc primitive; primitive.name = TfToken("classPair");
    primitive.type = UsdGenAuthoredPlaneType::Int32; primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primitive.arity = 2; primitive.intValues = {99,1,42,2,11,3};
    UsdGenAuthoredPlaneDesc groom; groom.name = TfToken("groomWeight");
    groom.type = UsdGenAuthoredPlaneType::Float32; groom.domain = UsdGenAuthoredPlaneDomain::Groom;
    groom.arity = 1; groom.floatValues = {.25f};
    curves.authoredPlanes = {point, primitive, groom}; desc.curveSets = {curves};
    auto source = Node("/Ops/Source", "UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    source.params = {{TfToken("useRest"),VtValue(true),false},
                     {TfToken("rebind"),VtValue(TfToken("never")),false}};
    desc.nodes = {source}; desc.terminal = source.path; return desc;
}
void Append(UsdGenGraphDesc& desc, UsdGenNodeDesc node) {
    desc.terminal = node.path; desc.nodes.push_back(std::move(node));
}
void Expression(UsdGenGraphDesc& desc, size_t nodeIndex, char const* path, char const* source,
                char const* destination, expr::Domain domain, expr::ScalarType type,
                char const* native, VtValue literal) {
    expr::ValueShape shape{type,1,1,1,1,false};
    UsdGenExpressionDesc expression; expression.path = SdfPath(path); expression.source = source;
    expression.outputs.push_back({TfToken("result"), TfToken(native), shape});
    desc.expressions.push_back(expression);
    UsdGenExpressionBinding binding; binding.expression = expression.path;
    binding.destination = TfToken(destination); binding.domain = domain; binding.destinationShape = shape;
    binding.nativeType = TfToken(native); binding.literal = std::move(literal);
    desc.nodes.at(nodeIndex).expressionBindings.push_back(std::move(binding));
}
UsdGenNodeDesc Cull(SdfPath input, bool enabled = true) {
    auto node = Node("/Ops/Cull", "UsdGenLength", input); node.enabled = enabled;
    node.params = {{TfToken("length:mode"),VtValue(TfToken("cull")),false},
                   {TfToken("cullThreshold"),VtValue(.25f),false}};
    return node;
}
DeviceGeneration Commit(UsdGenSession& session, double frame) {
    auto snapshot = session.CommitSnapshot(frame, UsdGenCommitReason::SetTime);
    CHECK(snapshot);
    for (auto const& error : snapshot->diagnostics.errors) std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(!snapshot->diagnostics.HasErrors() && snapshot->generation && snapshot->generation->device);
    CHECK(usdGen::UsdGenCurvePublicationOf(*snapshot->generation).tiles.empty()); return snapshot->generation->device;
}
struct Pair {
    UsdGenSession cuda{2}, vulkan{2};
    // Run this at the next case's start, leaving the final case's asynchronous
    // cleanup pending at process exit so the automatic exit join is exercised.
    Pair() { RetirePreviousCase(); }
    void Set(UsdGenGraphDesc desc) {
        std::fprintf(stderr, "Session parity terminal: %s\n", desc.terminal.GetText());
        cuda.SetDevicePublicationEnabled(true); vulkan.SetDevicePublicationEnabled(true);
        desc.executionBackend = usdGenTestUtils::Backend("cuda"); usdGenUsd::SetGraphDesc(cuda, desc);
        desc.executionBackend = usdGenTestUtils::Backend("vulkan"); usdGenUsd::SetGraphDesc(vulkan, desc);
    }
    Packet Run(double frame = 1, float tolerance = 0, bool generatedRest = false) {
        std::fprintf(stderr, "Session parity frame: %g\n", frame);
        std::fprintf(stderr, "  CUDA commit\n");
        auto cudaGeneration = Commit(cuda, frame);
        std::fprintf(stderr, "  CUDA readback\n");
        auto a = ReadCuda(cudaGeneration);
        std::fprintf(stderr, "  Vulkan commit\n");
        auto vulkanGeneration = Commit(vulkan, frame);
        std::fprintf(stderr, "  Vulkan readback\n");
        auto b = ReadVulkan(vulkanGeneration);
        Compare(a, b, tolerance, generatedRest); return a;
    }
    ~Pair() { cuda.Drain(); vulkan.Drain(); }
};
void ExpectedSource(Packet const& packet, bool culled = false) {
    CHECK(Values<uint64_t>(packet, UsdGenDeviceChannelSemantic::StableIds) ==
        (culled ? std::vector<uint64_t>{11,99} : std::vector<uint64_t>{11,42,99}));
    CHECK(Values<uint32_t>(packet, UsdGenDeviceChannelSemantic::ElementOffsets) ==
        (culled ? std::vector<uint32_t>{0,2,4} : std::vector<uint32_t>{0,2,5,7}));
    CHECK(Values<float>(packet.planes.at("density")) ==
        (culled ? std::vector<float>{30,31,10,11} : std::vector<float>{30,31,20,21,22,10,11}));
    CHECK(Values<int32_t>(packet.planes.at("classPair")) ==
        (culled ? std::vector<int32_t>{11,3,99,1} : std::vector<int32_t>{11,3,42,2,99,1}));
    CHECK(Values<float>(packet.planes.at("groomWeight")) == std::vector<float>{.25f});
}
}

int main(int argc, char** argv) {
    CHECK(argc == 2);
#ifdef _WIN32
    CHECK(_putenv_s("USDGEN_VULKAN_SHADER_DIR", argv[1]) == 0);
#else
    CHECK(setenv("USDGEN_VULKAN_SHADER_DIR", argv[1], 1) == 0);
#endif
    usdGenTest::EnsureHydraPublishers();
    int devices = 0; auto status = cudaGetDeviceCount(&devices);
    bool const cudaUnavailable = status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && devices == 0);
    if (!cudaUnavailable && status != cudaSuccess)
        std::fprintf(stderr, "CUDA preflight failed: %s (%d)\n", cudaGetErrorString(status), int(status));
    CHECK(status == cudaSuccess || cudaUnavailable);
    if (cudaUnavailable || !HasDefaultVulkanDevice()) return 77;
    {
        auto folder = std::filesystem::temp_directory_path() /
            ("usdGen-cage-session-parity-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(folder);
        struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code error; std::filesystem::remove_all(path,error); } } cleanup{folder};
        auto map = (folder / "owners.ptx").string();
        CHECK(test::WriteSurfaceCageOwnerMap(map));
        auto desc = test::SurfaceCageDesc(map);
        Pair pair; pair.Set(desc); auto dense = pair.Run();
        CHECK(dense.geometry.elementCount > 3 && dense.geometry.pointCount == 4*dense.geometry.elementCount);
        CHECK(Values<int32_t>(dense.planes.at("tubeId")) == std::vector<int32_t>(dense.geometry.elementCount,41));
        CHECK(Values<int32_t>(dense.planes.at("regionId")) == std::vector<int32_t>(dense.geometry.elementCount,8));
        CHECK(Values<int32_t>(dense.planes.at("hierarchyLevel")) == std::vector<int32_t>(dense.geometry.elementCount,2));
        auto retainedCuda = Commit(pair.cuda,1); auto retainedVk = Commit(pair.vulkan,1);
        desc.nodes[0].params.push_back({TfToken("resampleTo"),VtValue(7),false});
        auto width = Node("/CageWidth","UsdGenWidth",desc.terminal);
        width.params = {{TfToken("width"),VtValue(.3f),false}};
        Append(desc,width); pair.Set(desc); auto resampled = pair.Run(2,3.e-6f);
        CHECK(resampled.geometry.elementCount == dense.geometry.elementCount &&
            resampled.geometry.pointCount == 7*dense.geometry.elementCount);
        CHECK(Values<uint64_t>(resampled,UsdGenDeviceChannelSemantic::StableIds) ==
            Values<uint64_t>(dense,UsdGenDeviceChannelSemantic::StableIds));
        Compare(dense,ReadCuda(retainedCuda)); Compare(dense,ReadVulkan(retainedVk));
    }
    {
        Pair pair; pair.Set(Source()); auto source = pair.Run(); ExpectedSource(source);
        CHECK(Values<float>(source, UsdGenDeviceChannelSemantic::Widths) == std::vector<float>({6,7,3,4,5,1,2}));
        CHECK(Values<GfVec3f>(source, UsdGenDeviceChannelSemantic::Points) ==
            std::vector<GfVec3f>({{.75f,0,0},{.75f,1,0},{.5f,0,0},{.5f,.0625f,0},{.5f,.125f,0},{.25f,0,0},{.25f,.5f,0}}));
    }
    {
        auto desc = Source(); auto& curves = desc.curveSets[0];
        curves.role = UsdGenLane::Reference; curves.curveRole = TfToken("guide");
        // CUDA ReferenceSource accepts bare reference geometry; named planes
        // and authored root frames are outside its admitted contract.
        curves.authoredPlanes.clear(); curves.rootFrame.clear();
        auto reference = Node("/Ops/Reference", "UsdGenReferenceSource"); reference.references = {curves.path};
        desc.nodes = {reference}; desc.terminal = reference.path;
        auto width = Node("/Ops/ReferenceWidth", "UsdGenWidth", desc.terminal);
        width.params = {{TfToken("width"), VtValue(1.f), false}};
        Append(desc, width);
        Pair pair; pair.Set(desc); auto packet = pair.Run();
        CHECK(Values<uint64_t>(packet,UsdGenDeviceChannelSemantic::StableIds) == std::vector<uint64_t>({11,42,99}));
        CHECK(Values<uint32_t>(packet,UsdGenDeviceChannelSemantic::ElementOffsets) == std::vector<uint32_t>({0,2,5,7}));
    }
    {
        auto desc = Source(); Append(desc, Cull(desc.terminal, false));
        Pair pair; pair.Set(desc); auto disabled = pair.Run(); ExpectedSource(disabled);
        auto sourceDesc = Source(); pair.Set(sourceDesc); auto source = pair.Run(); Compare(disabled, source);
        desc.nodes[1].enabled = true; pair.Set(desc); ExpectedSource(pair.Run(), true);
    }
    {
        auto desc = Source(); auto width = Node("/Ops/Width", "UsdGenWidth", desc.terminal);
        width.params = {{TfToken("width"),VtValue(.5f),false},{TfToken("mask"),VtValue(.5f),false},
                        {TfToken("width:knots"),VtValue(VtVec2fArray{{0,1},{1,.5f}}),false},
                        {TfToken("width:interpolation"),VtValue(TfToken("linear")),false}};
        Append(desc, width);
        Expression(desc,1,"/Expr/Width","$t * .5 + .25","usdGen:width",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(.5f));
        Expression(desc,1,"/Expr/Replace","$primIndex == 0","replace",expr::Domain::Primitive,
            expr::ScalarType::Bool,"bool",VtValue(false));
        Expression(desc,1,"/Expr/Enabled","$frame < 3","enabled",expr::Domain::Groom,
            expr::ScalarType::Bool,"bool",VtValue(true));
        Pair pair; pair.Set(desc); auto first = pair.Run(); ExpectedSource(first);
        CHECK(Values<float>(first, UsdGenDeviceChannelSemantic::Widths) ==
            std::vector<float>({3.125f,3.6875f,1.875f,2.75f,3.4375f,.625f,1.375f}));
        auto disabled = pair.Run(3);
        CHECK(Values<float>(disabled, UsdGenDeviceChannelSemantic::Widths) == std::vector<float>({6,7,3,4,5,1,2}));
    }
    for (bool cutExtend : {false,true}) {
        auto desc = Source(); auto length = Node("/Ops/Length", "UsdGenLength", desc.terminal);
        length.params = {{TfToken("length:mode"),VtValue(TfToken("set")),false},
                         {TfToken("length:value"),VtValue(.75f),false},
                         {TfToken("length:method"),VtValue(TfToken(cutExtend ? "cutExtend" : "scale")),false},
                         {TfToken("rebuild"),VtValue(TfToken("reparam")),false}};
        Append(desc, length); Pair pair; pair.Set(desc); auto packet = pair.Run(); ExpectedSource(packet);
        auto points = Values<GfVec3f>(packet, UsdGenDeviceChannelSemantic::Points);
        CHECK(points[1] == GfVec3f(.75f,.75f,0) && points[4] == GfVec3f(.5f,.75f,0) &&
              points[6] == GfVec3f(.25f,.75f,0));
    }
    {
        auto desc = Source(); Append(desc, Cull(desc.terminal));
        auto deform = Node("/Ops/Deform", "UsdGenDeform", desc.terminal);
        deform.surfaces = {desc.surfaces[0].path};
        deform.params = {{TfToken("rbfSamples"),VtValue(5),false},
                         {TfToken("lockRoots"),VtValue(false),false}}; Append(desc, deform);
        Expression(desc,2,"/Expr/DeformMask","$frame < 2 ? .25 : 1","mask",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(1.f));
        Pair pair; pair.Set(desc);
        for (int frame : {1,2}) {
            auto packet = pair.Run(frame, 2.e-5f); ExpectedSource(packet,true);
            auto points = Values<GfVec3f>(packet, UsdGenDeviceChannelSemantic::Points);
            auto rest = Values<GfVec3f>(packet, UsdGenDeviceChannelSemantic::RestPoints);
            CHECK(points.size() == rest.size());
            auto shift = GfVec3f(.25f,.5f,-.125f) * (frame == 1 ? .25f : 1.f);
            for (size_t i=0; i!=points.size(); ++i) CHECK((points[i]-rest[i]-shift).GetLength() < 2.e-5f);
        }
    }
    {
        // Nonzero triangle u and v distinguish barycentric root locking
        // from bilinear weights; source roots are deliberately displaced.
        auto desc = Source();
        auto deform = Node("/Ops/RootedDeform", "UsdGenDeform", desc.terminal);
        deform.surfaces = {desc.surfaces[0].path};
        deform.params = {{TfToken("rbfSamples"),VtValue(5),false},
                         {TfToken("lockRoots"),VtValue(true),false}};
        Append(desc, deform); Pair pair; pair.Set(desc);
        auto packet = pair.Run(1, 2.e-5f); ExpectedSource(packet);
        auto points = Values<GfVec3f>(packet, UsdGenDeviceChannelSemantic::Points);
        auto offsets = Values<uint32_t>(packet, UsdGenDeviceChannelSemantic::ElementOffsets);
        CHECK(points.size() == 7 && offsets.size() == 4);
        // Stable IDs sort authored curve 2, 1, 0 into output order.
        auto const& surface = desc.surfaces[0];
        for (size_t c=0; c!=3; ++c) {
            size_t authored = 2-c; auto uv = desc.curveSets[0].skinPrimUv[authored];
            int face = desc.curveSets[0].skinPrim[authored];
            auto a = surface.points[surface.faceVertexIndices[3*face]];
            auto b = surface.points[surface.faceVertexIndices[3*face+1]];
            auto d = surface.points[surface.faceVertexIndices[3*face+2]];
            auto target = a*(1.f-uv[0]-uv[1])+b*uv[0]+d*uv[1];
            CHECK((points[offsets[c]]-target).GetLength() < 2.e-5f);
        }
    }
    {
        auto desc = Source(); Append(desc, Cull(desc.terminal));
        auto noise = Node("/Ops/Noise", "UsdGenNoise", desc.terminal);
        noise.params = {{TfToken("noise:frequency"),VtValue(1.3f),false},
                        {TfToken("noise:magnitude"),VtValue(.05f),false},
                        {TfToken("noise:correlation"),VtValue(.5f),false},
                        {TfToken("noise:seed"),VtValue(23),false},
                        {TfToken("noise:octaves"),VtValue(2),false},
                        {TfToken("preserveLength"),VtValue(0.f),false}};
        Append(desc,noise);
        Expression(desc,2,"/Expr/NoiseFrequency","$frame < 2 ? 1 : 3","noise:frequency",expr::Domain::Groom,
            expr::ScalarType::Float32,"float",VtValue(1.f));
        Expression(desc,2,"/Expr/NoiseSeed","$primIndex + 7","noise:seed",expr::Domain::Primitive,
            expr::ScalarType::Int32,"int",VtValue(7));
        Expression(desc,2,"/Expr/NoiseMagnitude","$P[0] * 0 + .05","noise:magnitude",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(.05f));
        Pair pair; pair.Set(desc); auto first = pair.Run(1,2.e-5f); ExpectedSource(first,true);
        auto second = pair.Run(2,2.e-5f); ExpectedSource(second,true);
        CHECK(Semantic(first,UsdGenDeviceChannelSemantic::Points).bytes != Semantic(second,UsdGenDeviceChannelSemantic::Points).bytes);
        CHECK(Semantic(first,UsdGenDeviceChannelSemantic::Points).bytes != Semantic(first,UsdGenDeviceChannelSemantic::RestPoints).bytes);
    }
    for (bool randomized : {false,true}) {
        auto desc = Source(); auto grow = Node("/Ops/Grow", "UsdGenGrow", desc.terminal); grow.seed = 19;
        grow.params = {{TfToken("segments"),VtValue(5),false},{TfToken("length"),VtValue(2.f),false},
                       {TfToken("lengthRandom"),VtValue(randomized ? GfVec2f(.5f,1.8f) : GfVec2f(1.f)),false},
                       {TfToken("direction"),VtValue(TfToken("surfaceNormal")),false}};
        if (randomized) {
            grow.params.push_back({TfToken("lift"),VtValue(23.f),false});
            grow.params.push_back({TfToken("azimuth"),VtValue(57.f),false});
            grow.params.push_back({TfToken("azimuthRandom"),VtValue(.8f),false});
        }
        Append(desc,grow); Pair pair; pair.Set(desc); auto packet = pair.Run(1,2.e-5f,true);
        CHECK(packet.geometry.elementCount == 3 && packet.geometry.pointCount == 15);
        CHECK(Values<uint64_t>(packet,UsdGenDeviceChannelSemantic::StableIds) == std::vector<uint64_t>({11,42,99}));
        CHECK(Values<uint32_t>(packet,UsdGenDeviceChannelSemantic::ElementOffsets) == std::vector<uint32_t>({0,5,10,15}));
        if (!randomized) {
            auto points = Values<GfVec3f>(packet,UsdGenDeviceChannelSemantic::Points);
            for (size_t c=0;c!=3;++c) for(size_t i=0;i!=5;++i)
                CHECK(points[c*5+i] == GfVec3f(c==0?.75f:c==1?.5f:.25f,0,float(i)*.5f));
        }
    }
    {
        auto desc = Source();
        for (auto& point : desc.curveSets[0].points) point += GfVec3f(2,0,0);
        Expression(desc,0,"/Expr/SourceUseRest","$frame < 2","useRest",expr::Domain::Groom,
            expr::ScalarType::Bool,"bool",VtValue(true));
        Expression(desc,0,"/Expr/SourceResample","$frame < 2 ? 0 : 5","resampleTo",expr::Domain::Groom,
            expr::ScalarType::Int32,"int",VtValue(0));
        Pair pair; pair.Set(desc); auto rest = pair.Run(1); ExpectedSource(rest);
        CHECK(!rest.geometry.alreadyDeformed);
        auto current = pair.Run(2,3.e-6f);
        CHECK(current.geometry.alreadyDeformed && current.geometry.pointCount == 15);
        auto points = Values<GfVec3f>(current,UsdGenDeviceChannelSemantic::Points);
        CHECK(points[0] == GfVec3f(2.75f,0,0) && points[4] == GfVec3f(2.75f,1,0));
        CHECK(Values<float>(current.planes.at("density")) ==
            std::vector<float>({30,30.25f,30.5f,30.75f,31,20,20.5f,21,21.5f,22,10,10.25f,10.5f,10.75f,11}));
    }
    {
        auto desc = Source();
        desc.curveSets[0].skinPrim = {99,-1,99}; desc.curveSets[0].rootFrame.clear();
        desc.nodes[0].params[1].value = VtValue(TfToken("onError"));
        Pair pair; pair.Set(desc); auto repaired = pair.Run(); ExpectedSource(repaired);
        auto bindings = Values<int32_t>(repaired,UsdGenDeviceChannelSemantic::RootPrim);
        CHECK(bindings.size() == 3);
        for (int32_t binding : bindings) CHECK(binding >= 0 && binding < 3);
    }
    {
        auto desc = Source(); auto source = desc.nodes[0];
        auto left = Node("/Ops/WidthLeft","UsdGenWidth",source.path);
        auto right = Node("/Ops/WidthRight","UsdGenWidth",source.path);
        left.params = {{TfToken("width"),VtValue(.25f),false}};
        right.params = {{TfToken("width"),VtValue(.75f),false}};
        auto blend = Node("/Ops/Blend","UsdGenWidthBlend");
        blend.inputs = {left.path,right.path};
        blend.params = {{TfToken("widthBlend:weight"),VtValue(.25f),false}};
        desc.nodes = {blend,right,source,left}; desc.terminal = blend.path;
        Pair pair; pair.Set(desc); auto packet = pair.Run(); ExpectedSource(packet);
        CHECK(Values<float>(packet,UsdGenDeviceChannelSemantic::Widths) == std::vector<float>(7,.375f));
        // Authored fan-in order matters, independently of descriptor order.
        std::reverse(desc.nodes[0].inputs.begin(),desc.nodes[0].inputs.end()); pair.Set(desc);
        auto reversed = pair.Run(2); ExpectedSource(reversed);
        CHECK(Values<float>(reversed,UsdGenDeviceChannelSemantic::Widths) == std::vector<float>(7,.625f));
    }
    {
        auto desc = Source(); Append(desc,Cull(desc.terminal));
        Expression(desc,1,"/Expr/CullEnabled","$frame < 2","enabled",expr::Domain::Groom,
            expr::ScalarType::Bool,"bool",VtValue(true));
        Expression(desc,1,"/Expr/CullThreshold","$primIndex == 1 ? .25 : 0","cullThreshold",expr::Domain::Primitive,
            expr::ScalarType::Float32,"float",VtValue(.25f));
        Expression(desc,1,"/Expr/CullMask","$t == 0 ? 0 : 1","mask",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(1.f));
        Pair pair; pair.Set(desc); ExpectedSource(pair.Run(1),true); ExpectedSource(pair.Run(2));
    }
    {
        auto desc = Source(); auto length = Node("/Ops/FieldLength","UsdGenLength",desc.terminal);
        length.params = {{TfToken("length:mode"),VtValue(TfToken("scale")),false},
                         {TfToken("length:value"),VtValue(1.f),false}};
        Append(desc,length);
        Expression(desc,1,"/Expr/LengthValue","$t * .5 + .25","length:value",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(1.f));
        Expression(desc,1,"/Expr/LengthMask","$t == 0 ? 0 : .5","mask",expr::Domain::Point,
            expr::ScalarType::Float32,"float",VtValue(1.f));
        Expression(desc,1,"/Expr/LengthMinimum","$primIndex == 1 ? .125 : 0","minRemainingLength",expr::Domain::Primitive,
            expr::ScalarType::Float32,"float",VtValue(0.f));
        Pair pair; pair.Set(desc); auto packet = pair.Run(); ExpectedSource(packet);
        auto points = Values<GfVec3f>(packet,UsdGenDeviceChannelSemantic::Points);
        CHECK(points[1] == GfVec3f(.75f,.875f,0) && points[3] == GfVec3f(.5f,.0625f,0) &&
              points[4] == GfVec3f(.5f,.125f,0) && points[6] == GfVec3f(.25f,.4375f,0));
    }
    {
        // Source remains a selectable immutable terminal after descendants
        // have produced topology and point revisions.
        auto desc = Source(); auto selected = desc.terminal;
        Append(desc,Cull(desc.terminal));
        auto width = Node("/Ops/UnusedWidth","UsdGenWidth",desc.terminal);
        width.params = {{TfToken("width"),VtValue(.125f),false}}; Append(desc,width);
        desc.terminal = selected; Pair pair; pair.Set(desc); ExpectedSource(pair.Run());
    }
    {
        auto desc = Source();
        desc.nodes[0].params.push_back({TfToken("resampleTo"),VtValue(5),false});
        Pair pair; pair.Set(desc); auto resampled = pair.Run(1,3.e-6f);
        CHECK(resampled.geometry.elementCount == 3 && resampled.geometry.pointCount == 15);
        CHECK(Values<uint32_t>(resampled,UsdGenDeviceChannelSemantic::ElementOffsets) ==
            std::vector<uint32_t>({0,5,10,15}));
        CHECK(Values<float>(resampled.planes.at("density")) ==
            std::vector<float>({30,30.25f,30.5f,30.75f,31,20,20.5f,21,21.5f,22,10,10.25f,10.5f,10.75f,11}));
    }
    {
        auto desc = Source(); auto& curves = desc.curveSets[0];
        curves.curveVertexCounts.clear(); curves.curveId.clear(); curves.points.clear(); curves.rest.clear();
        curves.widths.clear(); curves.skinPrim.clear(); curves.skinPrimUv.clear(); curves.rootFrame.clear();
        curves.authoredPlanes.clear(); Pair pair; pair.Set(desc); auto empty = pair.Run();
        CHECK(empty.geometry.elementCount == 0 && empty.geometry.pointCount == 0);
        CHECK(Values<uint64_t>(empty,UsdGenDeviceChannelSemantic::StableIds).empty());
    }
    {
        auto desc = Source(); desc.curveSets.clear();
        // Production Scatter captures the same deterministic root packet on
        // both backends, then each GPU expands it using its native Grow route.
        auto& surface = desc.surfaces[0];
        surface.restPoints = {{0,0,0},{1,0,.2f},{1,1,.2f},{0,1,0}};
        surface.points = surface.restPoints;
        surface.faceVertexCounts = {4}; surface.faceVertexIndices = {0,1,2,3};
        surface.uv = {{0,0},{1,0},{1,1},{0,1}};
        auto scatter = Node("/Ops/Scatter","UsdGenScatter"); scatter.seed = 41;
        scatter.surfaces = {surface.path};
        scatter.params = {{TfToken("density"),VtValue(12.f),false}};
        auto grow = Node("/Ops/Grow","UsdGenGrow",scatter.path); grow.seed = 19;
        grow.params = {{TfToken("segments"),VtValue(5),false},{TfToken("length"),VtValue(2.f),false},
                       {TfToken("lengthRandom"),VtValue(GfVec2f(.5f,1.5f)),false}};
        desc.nodes = {grow,scatter}; desc.terminal = grow.path;
        Pair pair; pair.Set(desc); auto packet = pair.Run(1,2.e-5f,true);
        CHECK(packet.geometry.elementCount > 0 && packet.geometry.pointCount == 5*packet.geometry.elementCount);
        // Changing topology must create a fresh generation while the old
        // published packet and its owner remain consumable.
        auto retainedCuda = Commit(pair.cuda,1); auto retainedVk = Commit(pair.vulkan,1);
        desc.nodes[0].params[0].value = VtValue(7); pair.Set(desc); auto revised = pair.Run(2,2.e-5f,true);
        CHECK(revised.geometry.elementCount == packet.geometry.elementCount &&
            revised.geometry.pointCount == 7*packet.geometry.elementCount);
        CHECK(Values<uint64_t>(packet,UsdGenDeviceChannelSemantic::StableIds) ==
            Values<uint64_t>(revised,UsdGenDeviceChannelSemantic::StableIds));
        Compare(packet,ReadCuda(retainedCuda),2.e-5f,true);
        Compare(packet,ReadVulkan(retainedVk),2.e-5f,true);
    }
    {
        auto desc = Source(); Pair pair; pair.Set(desc);
        auto cudaGood = Commit(pair.cuda,1); auto vkGood = Commit(pair.vulkan,1);
        auto cudaBytes = ReadCuda(cudaGood); auto vkBytes = ReadVulkan(vkGood); Compare(cudaBytes,vkBytes);
        auto width = Node("/Ops/BadWidth", "UsdGenWidth", desc.terminal);
        width.params = {{TfToken("width"),VtValue(-1.f),false}}; Append(desc,width); pair.Set(desc);
        auto a = pair.cuda.CommitSnapshot(2,UsdGenCommitReason::SetTime);
        auto b = pair.vulkan.CommitSnapshot(2,UsdGenCommitReason::SetTime);
        CHECK(a && b && a->diagnostics.HasErrors() && b->diagnostics.HasErrors());
        CHECK(a->generation && b->generation && a->generation->device == cudaGood && b->generation->device == vkGood);
        CHECK(pair.cuda.NeedsCommit() && pair.vulkan.NeedsCommit());
        Compare(cudaBytes,ReadCuda(cudaGood)); Compare(vkBytes,ReadVulkan(vkGood));
        desc.nodes.back().params[0].value = VtValue(.25f); pair.Set(desc);
        auto recovered = pair.Run(2); ExpectedSource(recovered);
        CHECK(Values<float>(recovered,UsdGenDeviceChannelSemantic::Widths) == std::vector<float>(7,.25f));
        Compare(cudaBytes,ReadCuda(cudaGood)); Compare(vkBytes,ReadVulkan(vkGood));
    }
    {
        // Connected fields must use the generated topology and stable-ID
        // primitive order after native Grow, including the split
        // transcendental lane used by both expression evaluators.
        auto desc = Source();
        auto grow = Node("/Ops/ExpressionGrow", "UsdGenGrow", desc.terminal);
        grow.params = {{TfToken("segments"),VtValue(5),false},
                       {TfToken("length"),VtValue(2.f),false},
                       {TfToken("direction"),VtValue(TfToken("vector")),false},
                       {TfToken("directionVector"),VtValue(GfVec3f(0,1,0)),false}};
        Append(desc, grow);
        auto width = Node("/Ops/GeneratedWidth", "UsdGenWidth", desc.terminal);
        width.params = {{TfToken("width"),VtValue(1.f),false}}; Append(desc, width);
        Expression(desc,2,"/Expr/GeneratedWidth",".25 + sin($t) * .125","width",
            expr::Domain::Point,expr::ScalarType::Float32,"float",VtValue(1.f));
        Expression(desc,2,"/Expr/GeneratedMask","$t","mask",
            expr::Domain::Point,expr::ScalarType::Float32,"float",VtValue(1.f));
        Expression(desc,2,"/Expr/GeneratedReplace","$primIndex == 0","replace",
            expr::Domain::Primitive,expr::ScalarType::Bool,"bool",VtValue(false));
        Pair pair; pair.Set(desc); auto packet = pair.Run(1,3.e-6f,true);
        CHECK(packet.geometry.elementCount == 3 && packet.geometry.pointCount == 15);
        CHECK(Values<uint64_t>(packet,UsdGenDeviceChannelSemantic::StableIds) ==
            std::vector<uint64_t>({11,42,99}));
        CHECK(Values<uint32_t>(packet,UsdGenDeviceChannelSemantic::ElementOffsets) ==
            std::vector<uint32_t>({0,5,10,15}));
        CHECK(Values<int32_t>(packet.planes.at("classPair")) == std::vector<int32_t>({11,3,42,2,99,1}));
        auto widths = Values<float>(packet,UsdGenDeviceChannelSemantic::Widths);
        for (size_t c=0; c!=3; ++c) for (size_t i=0; i!=5; ++i) {
            float h = float(i)*.25f;
            float base = c == 0 ? 6.f+h : c == 1 ? 3.f+2.f*h : 1.f+h;
            float expression = float(.25 + std::sin(double(h))*.125);
            float target = c == 0 ? expression : base*expression;
            CHECK(std::abs(widths[5*c+i]-(base+(target-base)*h)) < 3.e-6f);
        }
    }
    for (int unsupported = 0; unsupported != 3; ++unsupported) {
        // The CUDA production matrix has no ExprOp terminal or connected
        // Grow/Scatter fields. Both Sessions must reject these captures and
        // keep the previously proved immutable generation consumable.
        auto desc = Source(); Pair pair; pair.Set(desc);
        auto cudaGood = Commit(pair.cuda,1); auto vkGood = Commit(pair.vulkan,1);
        auto cudaBytes = ReadCuda(cudaGood); auto vkBytes = ReadVulkan(vkGood);
        if (unsupported == 0) {
            auto op = Node("/Ops/UnsupportedExpr", "UsdGenExprOp", desc.terminal);
            op.params = {{TfToken("expr:source"),VtValue(std::string("$t + 1")),false},
                         {TfToken("expr:returnType"),VtValue(TfToken("float")),false},
                         {TfToken("mode"),VtValue(TfToken("widths")),false}};
            Append(desc, op);
        } else if (unsupported == 1) {
            auto grow = Node("/Ops/ConnectedGrow", "UsdGenGrow", desc.terminal);
            grow.params = {{TfToken("segments"),VtValue(5),false}}; Append(desc,grow);
            Expression(desc,1,"/Expr/GrowLength","$primIndex + 1","length",
                expr::Domain::Primitive,expr::ScalarType::Float32,"float",VtValue(1.f));
        } else {
            desc.curveSets.clear();
            auto scatter = Node("/Ops/ConnectedScatter", "UsdGenScatter");
            scatter.surfaces = {desc.surfaces[0].path};
            scatter.params = {{TfToken("density"),VtValue(1.f),false}};
            auto grow = Node("/Ops/ScatterGrow", "UsdGenGrow", scatter.path);
            grow.params = {{TfToken("segments"),VtValue(5),false}};
            desc.nodes = {scatter,grow}; desc.terminal = grow.path;
            Expression(desc,0,"/Expr/ScatterDensity","$frame + 1","density",
                expr::Domain::Groom,expr::ScalarType::Float32,"float",VtValue(1.f));
        }
        pair.Set(desc);
        auto a = pair.cuda.CommitSnapshot(2,UsdGenCommitReason::SetTime);
        auto b = pair.vulkan.CommitSnapshot(2,UsdGenCommitReason::SetTime);
        CHECK(a && b && a->diagnostics.HasErrors() && b->diagnostics.HasErrors());
        CHECK(a->generation && b->generation && a->generation->device == cudaGood &&
            b->generation->device == vkGood);
        CHECK(pair.cuda.NeedsCommit() && pair.vulkan.NeedsCommit());
        Compare(cudaBytes,ReadCuda(cudaGood)); Compare(vkBytes,ReadVulkan(vkGood));
        pair.Set(Source()); ExpectedSource(pair.Run(2));
    }
    {
        // Commit time is request data even when the compiled descriptor and
        // its expression programs are retained across successive captures.
        auto desc = Source(); desc.time = 500; desc.timeCodesPerSecond = 10;
        auto width = Node("/Ops/ExpressionClock", "UsdGenWidth", desc.terminal);
        width.params = {{TfToken("width"),VtValue(1.f),false}}; Append(desc,width);
        Expression(desc,1,"/Expr/Clock","$frame + $time","width",
            expr::Domain::Groom,expr::ScalarType::Float32,"float",VtValue(1.f));
        Pair pair; pair.Set(desc);
        for (double frame : {1.,5.,1.}) {
            auto packet = pair.Run(frame); ExpectedSource(packet);
            auto widths = Values<float>(packet,UsdGenDeviceChannelSemantic::Widths);
            for (float width : widths) CHECK(std::abs(width-float(frame+frame/10.)) < 3.e-6f);
        }
    }
    {
        // A cached structural source program can resolve to either an
        // identity layout or a new native topology. Both publish the exact
        // authoritative topology revision declared by that original plan.
        auto desc = Source();
        Expression(desc,0,"/Expr/AlternatingResample","$frame == 0 ? 0 : 5","resampleTo",
            expr::Domain::Groom,expr::ScalarType::Int32,"int",VtValue(0));
        Pair pair; pair.Set(desc);
        for (double frame : {0.,1.,0.}) {
            auto packet = pair.Run(frame,3.e-6f);
            CHECK(packet.geometry.elementCount == 3);
            if (frame == 0) {
                ExpectedSource(packet);
                CHECK(packet.geometry.pointCount == 7);
            } else {
                CHECK(packet.geometry.pointCount == 15);
                CHECK(Values<uint32_t>(packet,UsdGenDeviceChannelSemantic::ElementOffsets) ==
                    std::vector<uint32_t>({0,5,10,15}));
            }
        }
    }
    {
        // A disabled Deform preserves bytes but still marks the lineage as
        // animated-space geometry. The independent source branch stays rest.
        auto desc = Source(); Pair pair; pair.Set(desc);
        auto source = pair.Run(); CHECK(!source.geometry.alreadyDeformed);
        auto deform = Node("/Ops/DisabledDeform", "UsdGenDeform", desc.terminal);
        deform.enabled = false; deform.surfaces = {desc.surfaces.front().path};
        deform.params = {{TfToken("rbfSamples"),VtValue(5),false}};
        Append(desc,deform); pair.Set(desc);
        auto disabled = pair.Run(); CHECK(disabled.geometry.alreadyDeformed);
        CHECK(disabled.planes.size() == source.planes.size());
        for (auto const& plane : source.planes)
            CHECK(disabled.planes.at(plane.first).bytes == plane.second.bytes);
        desc.terminal = desc.nodes.front().path; pair.Set(desc);
        CHECK(!pair.Run().geometry.alreadyDeformed);
    }
    for (bool empty : {false,true}) {
        // Grow always emits a cubic BSpline topology, even when its C3
        // predecessor was CatmullRom or the source publishes no curves.
        // Empty Grow must complete through the production notification path:
        // geometry submits nothing, but named topology copies offset zero.
        auto desc = Source(); desc.curveSets.front().basis = TfToken("catmullRom");
        desc.nodes.front().enabled = !empty;
        auto grow = Node("/Ops/CatmullGrow", "UsdGenGrow", desc.terminal);
        grow.params = {{TfToken("segments"),VtValue(5),false},
                       {TfToken("length"),VtValue(2.f),false},
                       {TfToken("direction"),VtValue(TfToken("surfaceNormal")),false}};
        Append(desc,grow); Pair pair; pair.Set(desc);
        auto packet = pair.Run(1,2.e-5f,true);
        CHECK(UsdGenCurveLayoutBasisOf(packet.geometry.layoutMetadata) == UsdGenCurveLayoutBasis::BSpline);
        CHECK(packet.geometry.pointCount == (empty ? 0 : 15));
        CHECK(!packet.geometry.alreadyDeformed);
        if (empty)
            CHECK(Values<uint32_t>(packet,UsdGenDeviceChannelSemantic::ElementOffsets) ==
                std::vector<uint32_t>{0});
        desc.nodes.front().params.front().value = VtValue(false); pair.Set(desc);
        auto current = pair.Run(2,2.e-5f,true);
        CHECK(current.geometry.alreadyDeformed);
        desc.nodes.back().enabled = false; pair.Set(desc);
        auto disabled = pair.Run(3,2.e-5f,true);
        CHECK(UsdGenCurveLayoutBasisOf(disabled.geometry.layoutMetadata) == UsdGenCurveLayoutBasis::BSpline);
        CHECK(disabled.geometry.pointCount == (empty ? 0 : 15) && disabled.geometry.alreadyDeformed);
        for (auto const& plane : current.planes)
            CHECK(disabled.planes.at(plane.first).bytes == plane.second.bytes);
    }
    std::puts("CUDA/Vulkan production Session parity: PASS");
    return 0;
}
