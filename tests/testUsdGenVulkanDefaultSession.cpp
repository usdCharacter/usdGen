// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Plain Session production-provider parity. Geometry readback is test-only.
#include "usdGen/session.h"
#include "nodeSets.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/scheduler.h"
#include "usdGen/vulkan/defaultProvider.h"
#include "usdGen/vulkan/deviceGenerationAdapter.h"
#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/externalRetirement.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "default Session check: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
void ShaderDirectory(std::string const& value) {
#ifdef _WIN32
    CHECK(_putenv_s("USDGEN_VULKAN_SHADER_DIR", value.c_str()) == 0);
#else
    CHECK(setenv("USDGEN_VULKAN_SHADER_DIR", value.c_str(), 1) == 0);
#endif
}
void ClearShaderDirectory() {
#ifdef _WIN32
    CHECK(_putenv_s("USDGEN_VULKAN_SHADER_DIR", "") == 0);
#else
    CHECK(unsetenv("USDGEN_VULKAN_SHADER_DIR") == 0);
#endif
}
UsdGenGraphDesc Descriptor() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/DefaultSession");
    desc.executionBackend = UsdGenExecutionBackend::Vulkan;
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/DefaultSession/curves");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {3, 2}; curves.curveId = {20, 10};
    curves.points = {{0,0,0},{1,0,0},{2,0,0},{0,1,0},{1,1,0}};
    curves.rest = {{0,0,1},{1,0,1},{2,0,1},{0,1,1},{1,1,1}};
    curves.widths = {1,2,3,4,5}; curves.widthsInterpolation = TfToken("vertex");
    curves.curveGeneration = 41; desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/DefaultSession/source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/DefaultSession/width"); width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params = {{TfToken("width"), VtValue(1.25f), false},
                    {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, width}; desc.terminal = width.path;
    return desc;
}
template<class T> std::vector<uint8_t> Bytes(std::vector<T> const& values) {
    std::vector<uint8_t> result(values.size() * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), values.data(), result.size());
    return result;
}
using Oracle = std::vector<std::pair<std::string, std::vector<uint8_t>>>;
Oracle CpuOracle(UsdGenGraphDesc desc) {
    desc.executionBackend = UsdGenExecutionBackend::CpuReference;
    UsdGenCompiler compiler; UsdGenGraph graph;
    CHECK(compiler.Compile(desc, &graph).ok);
    UsdGenEvalContext evaluation; evaluation.desc = &graph.Desc();
    UsdGenScheduler scheduler(2);
    CHECK(!scheduler.Run(graph, evaluation, 1).diagnostics.HasErrors());
    auto const& b = NodeBuffer(graph, graph.NodeIdForPath(desc.terminal));
    std::vector<GfVec3f> points(b.totalCvs);
    for (size_t i = 0; i != b.totalCvs; ++i) points[i] = {b.px[i], b.py[i], b.pz[i]};
    return {{"points", Bytes(points)},
            {"rest", Bytes(std::vector<GfVec3f>(b.rest.begin(), b.rest.end()))},
            {"curveOffsets", Bytes(std::vector<uint32_t>(b.cvOffsets.begin(), b.cvOffsets.end()))},
            {"curveId", Bytes(std::vector<uint64_t>(b.curveId.begin(), b.curveId.end()))},
            {"hairT", Bytes(std::vector<float>(b.hairT.begin(), b.hairT.end()))},
            {"widths", Bytes(std::vector<float>(b.width.begin(), b.width.end()))}};
}

// The generation owns the device/context; the typed consumer access supplies
// its exact queue owner. All queue operations and lease admission run there.
void ExactPlanes(VulkanGenerationAccess const& access, Oracle const& oracle,
                 VulkanGenerationLease held = {}) {
    CHECK(access);
    auto context = access.context;
    auto device = context->device();
    VkCommandPool pool = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    std::vector<std::shared_ptr<ChargedBuffer>> staging;
    access.queueOwner->InvokeOwner([&] {
        if (!held) held = AcquireVulkanGeneration(access.generation,
            reinterpret_cast<UsdGenDeviceStream>(context->computeQueue()));
        CHECK(held && held.WaitUntilReady() == UsdGenDeviceStatus::Ok);
        CHECK(held.Planes().size() == oracle.size());
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
        for (auto const& expected : oracle) {
            auto const& planes = held.Planes();
            auto found = std::find_if(planes.begin(), planes.end(), [&](auto const& p) {
                return p.metadata.name == expected.first;
            });
            CHECK(found != planes.end() && found->bytes == expected.second.size());
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = found->bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto output = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            CHECK(output && output->MarkSubmitted(fence, access.generation) == VK_SUCCESS);
            VkBufferCopy copy{0, 0, found->bytes};
            vkCmdCopyBuffer(command, found->buffer, output->buffer(), 1, &copy);
            staging.push_back(std::move(output));
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
            void* mapped = nullptr;
            CHECK(vkMapMemory(device, staging[i]->memory(), 0, oracle[i].second.size(), 0, &mapped) == VK_SUCCESS);
            CHECK(std::memcmp(mapped, oracle[i].second.data(), oracle[i].second.size()) == 0);
            vkUnmapMemory(device, staging[i]->memory());
        }
        held.Complete();
        staging.clear();
        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
    });
}

unsigned lifecycleCompleted = 0;
void VerifyLifecycleExit() {
    CHECK(lifecycleCompleted == 32);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
}
int Lifecycle(std::string const& shaderPath) {
    if (!HasDefaultVulkanDevice()) return 77;
    ShaderDirectory(std::filesystem::path(shaderPath).parent_path().string());
    CHECK(std::atexit(VerifyLifecycleExit) == 0);
    std::vector<std::weak_ptr<DeviceContext>> retiredContexts;
    for (unsigned iteration = 0; iteration != 32; ++iteration) {
        {
            auto descriptor = Descriptor();
            descriptor.nodes[1].params[0].value = VtValue(1.25f + float(iteration) / 32.f);
            auto oracle = CpuOracle(descriptor);
            UsdGenSession session(2);
            session.SetGraphDesc(descriptor);
            session.SetDevicePublicationEnabled(true);
            auto snapshot = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
            if (snapshot) for (auto const& error : snapshot->diagnostics.errors)
                std::fprintf(stderr, "lifecycle %u: %s\n", iteration, error.c_str());
            CHECK(snapshot && !snapshot->diagnostics.HasErrors() && snapshot->generation &&
                  snapshot->generation->device && snapshot->generation->tiles.empty());
            auto access = GetVulkanGenerationAccess(snapshot->generation->device);
            CHECK(access && access.context->shaderInt64Enabled() &&
                  access.context->timelineSemaphoreEnabled());
            retiredContexts.push_back(access.context);
            ExactPlanes(access, oracle);
            session.Drain();
        }
        ++lifecycleCompleted;
        std::fprintf(stderr, "Vulkan Session lifecycle %u/32 destroyed\n", lifecycleCompleted);
    }
    // Consumer proof and native retirement may overlap the next provider's
    // bring-up. Keep only weak observations during the loop, then prove every
    // context was released before the process-wide retirement join.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (!std::all_of(retiredContexts.begin(), retiredContexts.end(),
                       [](auto const& context) { return context.expired(); })) {
        CHECK(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CloseVulkanExternalRetirement();
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
    std::puts("32 plain Vulkan Session lifecycles and final native cleanup: PASS");
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[2]) == "--lifecycle") return Lifecycle(argv[1]);
    auto descriptor = Descriptor();
    UsdGenSession session(2);
    session.SetGraphDesc(descriptor); session.SetDevicePublicationEnabled(true);
    ShaderDirectory("__usdGen_absent_default_shaders__");
    auto missing = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(missing && missing->diagnostics.HasErrors() && !missing->generation && session.NeedsCommit());
    bool precise = false;
    for (auto const& error : missing->diagnostics.errors)
        precise |= error.find("width.spv") != std::string::npos;
    CHECK(precise);
    if (argc == 2 && std::string(argv[1]) == "--missing") return 0;
    if (!HasDefaultVulkanDevice()) return 77;
    CHECK(argc == 2);
    if (std::string(argv[1]) == "--auto") ClearShaderDirectory();
    else ShaderDirectory(std::filesystem::path(argv[1]).parent_path().string());
    auto oracle = CpuOracle(descriptor);
    auto first = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    if (first) for (auto const& error : first->diagnostics.errors)
        std::fprintf(stderr, "%s\n", error.c_str());
    CHECK(first && !first->diagnostics.HasErrors() && first->generation && first->generation->device);
    CHECK(first->generation->tiles.empty());
    auto old = GetVulkanGenerationAccess(first->generation->device); CHECK(old);
    CHECK(old.context->shaderInt64Enabled() && old.context->timelineSemaphoreEnabled());
    ExactPlanes(old, oracle);
    UsdGenSession sibling(2);
    sibling.SetGraphDesc(descriptor); sibling.SetDevicePublicationEnabled(true);
    auto concurrent = sibling.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(concurrent && !concurrent->diagnostics.HasErrors() && concurrent->generation->device);
    auto other = GetVulkanGenerationAccess(concurrent->generation->device); CHECK(other);
    CHECK(other.context != old.context && other.context->resources() == old.context->resources());
    CHECK(other.context->resourceDeviceId() == old.context->resourceDeviceId());
    ExactPlanes(other, oracle);
    VulkanGenerationLease retained;
    old.queueOwner->InvokeOwner([&] {
        retained = AcquireVulkanGeneration(old.generation,
            reinterpret_cast<UsdGenDeviceStream>(old.context->computeQueue()));
        CHECK(retained);
    });
    session.NotifyDeviceContextLost(UsdGenDeviceBackend::Vulkan, old.context->physicalIndex());
    auto lost = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    CHECK(lost && lost->diagnostics.HasErrors() && lost->generation == first->generation && session.NeedsCommit());
    CHECK(!GetVulkanGenerationAccess(old.generation));
    auto recovered = session.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    if (recovered) for (auto const& error : recovered->diagnostics.errors)
        std::fprintf(stderr, "recovery: %s\n", error.c_str());
    CHECK(recovered && !recovered->diagnostics.HasErrors() && recovered->generation->device);
    auto fresh = GetVulkanGenerationAccess(recovered->generation->device); CHECK(fresh);
    CHECK(fresh.context != old.context && fresh.context->deviceUUID() == old.context->deviceUUID());
    CHECK(fresh.context->resources() == old.context->resources());
    ExactPlanes(fresh, oracle);
    ExactPlanes(old, oracle, std::move(retained));
    session.Drain(); sibling.Drain();
    std::puts("Plain Vulkan Session exact parity, initialization retry and context recovery: PASS");
    return 0;
}
