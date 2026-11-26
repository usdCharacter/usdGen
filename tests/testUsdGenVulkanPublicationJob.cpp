// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/publicationJob.h"
#include "testBackend.h"
#include "vulkanReadbackFixture.h"

#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <future>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, \
    "Vulkan publication job check failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info;
    info.context = std::move(context);
    auto& source = info.source;
    source.totalCurves = 1; source.totalCvs = 2;
    source.topologyVersion = 3; source.valueVersion = 4;
    source.px = {0.f, 1.f}; source.py = {0.f, 1.f}; source.pz = {0.f, 1.f};
    source.curveId = {9}; source.cvOffsets = {0, 2}; source.width = {1.f, 2.f};
    return info;
}

static VulkanPublicationJob::CreateInfo MakeInfo(
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<WidthPipeline> const& pipeline,
    std::shared_ptr<VulkanCompletionService> const& service,
    std::shared_ptr<UsdGenExecutionTaskGraph> const& preparer,
    std::shared_ptr<const void> lifetime, float width, uint64_t generation) {
    VulkanPublicationJob::CreateInfo info;
    info.nativeJob.source = Source(context);
    info.nativeJob.widthPipeline = pipeline;
    info.nativeJob.width = width;
    info.nativeJob.replace = 1;
    info.nativeJob.valueVersion = generation + 4;
    info.nativeJob.requestLifetime = std::move(lifetime);
    info.nativeJob.completionService = service;
    info.nativeJob.preparer = preparer;
    info.generation = generation;
    info.tool.toolId = "vulkan-publication-test";
    info.tool.graphVersion = 41;
    info.tool.snapshotVersion = generation;
    return info;
}

template <class T>
static void Pump(UsdGenExecutionPipeline& owner, std::future<T>& done) {
    CHECK(done.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
    owner.Drain();
}

int main(int argc, char** argv) {
    CHECK(argc == 2 || argc == 3);
    std::string mode=argc==3?argv[2]:"";
    CHECK(mode.empty() || mode=="--queued-cancel" || mode=="--prepare-reject" || mode=="--owner-unavailable");
    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!raw.empty() && raw.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());

    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &timeline;
    vkGetPhysicalDeviceFeatures2(probe->physical, &features);
    if (!timeline.timelineSemaphore) return 77;
    probe.reset();
    auto native = CreateNative(&unavailable, {}, &timeline);
    CHECK(native);

    DeviceContext::CreateInfo ci;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family; ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 7417; ci.nativeLifetime = native;
    ci.resources = {size_t{8} << 20, 0}; ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci); CHECK(context);
    auto pipeline = WidthPipeline::Create(context, code); CHECK(pipeline);

    UsdGenExecutionRuntime runtime{4};
    auto owner = std::make_shared<UsdGenExecutionPipeline>(runtime);
    auto service = VulkanCompletionService::Create({context, owner.get(), 16});
    CHECK(service);
    UsdGenExecutionTaskGraph::Limits limits;
    limits.maxJobs=mode=="--prepare-reject"?1:8;
    limits.maxTasks=16;limits.maxDependencyEdges=16;limits.maxActiveTasks=1;
    auto preparer = UsdGenExecutionTaskGraph::GetOrCreate(
        runtime, "vulkan-publication-preparation", 7417,limits);
    CHECK(preparer);
    auto domain = VulkanGenerationAdapterDomain::Create({owner, service});
    CHECK(domain);

    if(!mode.empty()) {
        // Hold an asynchronous task admission, not a blocked worker thread.
        std::shared_ptr<UsdGenExecutionTaskGraph::TaskCompletion> held;
        std::promise<void> started;auto startedFuture=started.get_future();
        UsdGenExecutionTaskGraph::Job blocker;blocker.tasks.resize(1);
        blocker.tasks[0].run=[&](auto const&,auto completion) {
            held=std::make_shared<UsdGenExecutionTaskGraph::TaskCompletion>(std::move(completion));
            started.set_value();
        };
        blocker.completion=[](auto publish,auto error){CHECK(!error && publish);publish();};
        CHECK(preparer->Submit(std::move(blocker)));
        CHECK(startedFuture.wait_for(std::chrono::seconds(15))==std::future_status::ready && held);
        UsdGenExecutionPipeline::Cancellation cancellation;
        CHECK(owner->Submit([&](auto const& value) {
            cancellation=value;return UsdGenExecutionPipeline::Publish([]{});
        })!=0);
        owner->Drain();CHECK(cancellation.epoch && !cancellation.Superseded());
        auto request=std::make_shared<int>(17);std::weak_ptr<int> weakRequest=request;
        auto info=MakeInfo(context,pipeline,service,preparer,request,3.f,20);
        info.domain=domain;
        auto job=VulkanPublicationJob::Create(std::move(info));CHECK(job);request.reset();
        std::weak_ptr<VulkanPublicationJob> weakJob=job;
        std::promise<void> completed;auto completedFuture=completed.get_future();
        std::atomic<unsigned> callbacks{0};std::atomic<bool> onOwner{false};
        owner->InvokeOwner([&] {
            CHECK(job->Start(*owner,cancellation,[&](auto result,auto error) {
                CHECK(!result && error);
                onOwner.store(owner->IsExecutingOwner(),std::memory_order_release);
                CHECK(callbacks.fetch_add(1,std::memory_order_acq_rel)==0);
                completed.set_value();
            }));
        });
        owner->Drain();
        CHECK(context->resources()->Snapshot().usedBytes==0);
        if(mode=="--prepare-reject") {
            Pump(*owner,completedFuture);
            CHECK(job->state()==VulkanPublicationJob::State::Failed && callbacks==1 && onOwner);
        } else {
            CHECK(preparer->GetAdmissionUsage().jobs==2); // Blocker and queued preparation.
            CHECK(job->state()==VulkanPublicationJob::State::Running);
            // Publication has no native plane yet, but its operation activity
            // must prevent the domain from dropping the queue during packing.
            CHECK(!domain->Close() && domain->IsClosing() && !domain->IsClosed());
            if(mode=="--owner-unavailable") owner->Shutdown();
            else {
                CHECK(owner->CancelPending()!=0);
                owner->InvokeOwner([]{});
                CHECK(cancellation.Superseded());
            }
        }
        (*held)([] {},{});held.reset();
        CHECK(completedFuture.wait_for(std::chrono::seconds(15))==std::future_status::ready);
        preparer->Drain();owner->Drain();
        CHECK(callbacks==1 && context->resources()->Snapshot().usedBytes==0);
        CHECK(owner->OutstandingCommands()==0 && owner->CallbackFailures()==0);
        CHECK(preparer->GetAdmissionUsage().jobs==0);
        if(mode=="--owner-unavailable") {
            CHECK(!onOwner && job->state()==VulkanPublicationJob::State::LostProof);
            CHECK(job->nativeState()==VulkanSourceWidthJob::State::LostProof);
            job.reset();CHECK(!weakJob.expired() && !weakRequest.expired());
            CHECK(!domain->Close()); // Deliberate process-isolated quarantine.
        } else {
            CHECK(onOwner && job->state()==(mode=="--prepare-reject"?
                VulkanPublicationJob::State::Failed:VulkanPublicationJob::State::Superseded));
            job.reset();CHECK(weakJob.expired() && weakRequest.expired());
            CHECK(domain->Close() && domain->IsClosed());
        }
        service->CloseAndJoin();
        return 0;
    }

    // Creation is fail-closed when the shared preparation/service contract is incomplete.
    {
        auto missing = MakeInfo(context, pipeline, service, preparer,
                                std::make_shared<int>(1), 3.f, 10);
        missing.domain = domain;
        std::string reason;
        auto noService = missing; noService.nativeJob.completionService.reset();
        CHECK(!VulkanPublicationJob::Create(std::move(noService), &reason));
        auto noPreparer = missing; noPreparer.nativeJob.preparer.reset();
        CHECK(!VulkanPublicationJob::Create(std::move(noPreparer), &reason));
    }

    // Suppression before admission produces no neutral generation/publication.
    {
        auto info = MakeInfo(context, pipeline, service, preparer,
                             std::make_shared<int>(2), 3.f, 10);
        info.domain = domain;
        auto job = VulkanPublicationJob::Create(std::move(info)); CHECK(job);
        job->SuppressPublication();
        unsigned callbacks = 0; std::shared_ptr<const UsdGenDeviceGeneration> result;
        std::exception_ptr error;
        std::promise<void> promise; auto done = promise.get_future();
        owner->InvokeOwner([&] {
            CHECK(job->Start(*owner, {}, [&](auto generation, auto failure) {
                ++callbacks; result = std::move(generation); error = failure;
                promise.set_value();
            }));
        });
        Pump(*owner, done);
        CHECK(callbacks == 1 && !result && error);
        CHECK(job->state() == VulkanPublicationJob::State::Superseded);
    }

    // Two real publications retain immutable, independent width planes.
    {
        auto info = MakeInfo(context, pipeline, service, preparer,
                             std::make_shared<int>(3), 3.f, 10);
        info.domain = domain;
        auto job = VulkanPublicationJob::Create(std::move(info)); CHECK(job);
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> promise;
        auto done = promise.get_future();
        owner->InvokeOwner([&] {
            CHECK(job->Start(*owner, {}, [&](auto generation, auto failure) {
                CHECK(!failure && generation); promise.set_value(std::move(generation));
            }));
        });
        Pump(*owner, done);
        auto first = done.get();
        CHECK(job->state() == VulkanPublicationJob::State::Ready);
        CHECK(first->Identity().backend == usdGenTestUtils::Backend("vulkan") &&
              first->Identity().generation == 10 && first->Geometry().valueVersion == 14);
        CHECK(first->Tool().toolId == "vulkan-publication-test" &&
              first->Tool().snapshotVersion == 10);
        CHECK(first->Channels().size() >= 4);
        auto widthMeta = std::find_if(first->Channels().begin(), first->Channels().end(),
            [](auto const& channel) { return channel.name == "widths"; });
        CHECK(widthMeta != first->Channels().end() &&
              widthMeta->type == UsdGenDeviceValueType::Float32 &&
              widthMeta->domain == UsdGenDeviceDomain::Point &&
              widthMeta->elementCount == 2 && widthMeta->arity == 1 &&
              widthMeta->semantic == UsdGenDeviceChannelSemantic::Widths);

        auto secondInfo = MakeInfo(context, pipeline, service, preparer,
                                    std::make_shared<int>(4), 7.f, 11);
        secondInfo.domain = domain;
        auto secondJob = VulkanPublicationJob::Create(std::move(secondInfo)); CHECK(secondJob);
        std::promise<std::shared_ptr<const UsdGenDeviceGeneration>> secondPromise;
        auto secondDone = secondPromise.get_future();
        owner->InvokeOwner([&] {
            CHECK(secondJob->Start(*owner, {}, [&](auto generation, auto failure) {
                CHECK(!failure && generation); secondPromise.set_value(std::move(generation));
            }));
        });
        Pump(*owner, secondDone);
        auto second = secondDone.get();
        CHECK(secondJob->state() == VulkanPublicationJob::State::Ready);
        CHECK(second->Identity().generation == 11 && second->Geometry().valueVersion == 15);
        CHECK(second->Tool().snapshotVersion == 11 && first->Owner() != second->Owner());

        auto readWidth = [&](std::shared_ptr<const UsdGenDeviceGeneration> const& generation,
                             float expected) {
            VulkanGenerationLease lease;
            std::unique_ptr<VulkanCompletionService::Watch> watch;
            std::shared_ptr<ChargedBuffer> staging;
            VkCommandBuffer command = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE;
            std::promise<void> copied; auto copiedFuture = copied.get_future();
            owner->InvokeOwner([&] {
                lease = AcquireVulkanGeneration(generation,
                    reinterpret_cast<UsdGenDeviceStream>(native->queue));
                CHECK(lease && lease.WaitUntilReady() == UsdGenDeviceStatus::Ok);
                auto it = std::find_if(lease.Planes().begin(), lease.Planes().end(),
                    [](auto const& p) { return p.metadata.name == "widths"; });
                CHECK(it != lease.Planes().end() && it->bytes == 8 && it->buffer);
                VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size = 8; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                staging = ChargedBuffer::Create(context, bi,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    UsdGenExecutionResourceKind::Scratch);
                CHECK(staging);
                VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                CHECK(vkCreateFence(native->device, &fi, nullptr, &fence) == VK_SUCCESS);
                VkCommandBufferAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                ai.commandPool = native->commands; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                ai.commandBufferCount = 1;
                CHECK(vkAllocateCommandBuffers(native->device, &ai, &command) == VK_SUCCESS);
                VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                CHECK(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS);
                VkMemoryBarrier before{}; before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
                VkBufferCopy copy{0, 0, 8}; vkCmdCopyBuffer(command, it->buffer, staging->buffer(), 1, &copy);
                VkMemoryBarrier after{}; after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                after.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
                CHECK(vkEndCommandBuffer(command) == VK_SUCCESS);
                CHECK(staging->MarkSubmitted(fence, generation) == VK_SUCCESS);
                watch = service->Reserve(generation, [&](VkResult proof) {
                    CHECK(proof == VK_SUCCESS && staging->PollComplete() == VK_SUCCESS);
                    void* mapped = nullptr;
                    CHECK(vkMapMemory(native->device, staging->memory(), 0, 8, 0, &mapped) == VK_SUCCESS);
                    float values[2]{}; std::memcpy(values, mapped, sizeof(values));
                    vkUnmapMemory(native->device, staging->memory());
                    CHECK(values[0] == expected && values[1] == expected);
                    CHECK(watch->Retire()); watch.reset(); copied.set_value();
                    return VulkanCompletionService::DeliveryResult::Posted;
                }, [&](VkResult) { CHECK(false); });
                CHECK(watch && watch->MarkPhaseSubmitted());
                VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
                CHECK(vkQueueSubmit(native->queue, 1, &submit, fence) == VK_SUCCESS);
            });
            // Complete queues the consumer's retirement signal. Queue the
            // observer after it, so observer proof also precedes safe domain
            // closure; draining the owner alone cannot prove GPU retirement.
            lease.Complete();
            owner->InvokeOwner([&] {
                CHECK(service->Arm(*watch, generation->Identity().generation));
            });
            Pump(*owner, copiedFuture);
            vkFreeCommandBuffers(native->device, native->commands, 1, &command);
            vkDestroyFence(native->device, fence, nullptr);
        };
        readWidth(first, 3.f);
        readWidth(second, 7.f);

        first.reset(); second.reset(); secondJob.reset(); job.reset(); owner->Drain();
    }

    // Closing is an external lifecycle boundary and rejects a later admission
    // without invoking native work or a completion callback.
    CHECK(domain->Close());
    auto closedInfo = MakeInfo(context, pipeline, service, preparer,
                               std::make_shared<int>(5), 3.f, 12);
    closedInfo.domain = domain;
    auto closedJob = VulkanPublicationJob::Create(std::move(closedInfo)); CHECK(closedJob);
    bool called = false;
    owner->InvokeOwner([&] {
        CHECK(!closedJob->Start(*owner, {}, [&](auto, auto) { called = true; }));
    });
    CHECK(!called && closedJob->state() == VulkanPublicationJob::State::Created);

    // Every successful job/consumer proof returned its plane reservations;
    // a closed domain is not reopened by a later publication request.
    owner->Drain();
    CHECK(context->resources()->Snapshot().usedBytes==0);
    CHECK(owner->OutstandingCommands()==0 && owner->CallbackFailures()==0);
    service->CloseAndJoin();
    return 0;
}
