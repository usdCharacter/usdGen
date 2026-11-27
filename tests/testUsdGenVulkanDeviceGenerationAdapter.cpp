// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Vulkan device-generation adapter contract.  This test keeps the source
// capture/readback boundary separate from neutral publication: production
// publication must expose metadata and owners, never geometry bytes.
#include "utils/testToken.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "testBackend.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenCurves/vulkan/deviceGenerationAdapter.h"
#include "usdGenVulkan/vulkan/completionService.h"
#include "usdGenVulkan/vulkan/externalRetirement.h"
#include "usdGen/deviceGeneration.h"
#include "vulkanNativeFixture.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <string>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/types.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan adapter check failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

struct HeldExitCheck {
    std::shared_ptr<VulkanExternalRetirementDomain> exitDomain;
    std::shared_ptr<VulkanGenerationAdapterDomain> domain;
    std::shared_ptr<std::atomic<bool>> retired;
    VulkanGenerationLease held;
};
static HeldExitCheck* heldExitCheck = nullptr;
static void VerifyHeldExit() {
    CHECK(heldExitCheck && heldExitCheck->exitDomain->IsQuiesced());
    CHECK(!heldExitCheck->retired->load() && !heldExitCheck->domain->IsClosed());
    heldExitCheck->held.Complete(); // Late release must retain unproved storage.
    CHECK(!heldExitCheck->held && !heldExitCheck->retired->load());
}

template<class T> static std::vector<uint8_t> Bytes(T const* p, size_t n) {
    std::vector<uint8_t> out(n * sizeof(T));
    if (!out.empty()) std::memcpy(out.data(), p, out.size());
    return out;
}

// Neutral publication uses the established names shared by every backend;
// native source planes retain their private Vulkan names.
static std::string PublicName(UsdGenDeviceChannelMetadata const& metadata) {
    switch (metadata.semantic) {
    case UsdGenDeviceChannelSemantic::Generic: return metadata.name.GetString();
    case UsdGenDeviceChannelSemantic::Points: return "points";
    case UsdGenDeviceChannelSemantic::RestPoints: return "rest";
    case UsdGenDeviceChannelSemantic::Widths: return "widths";
    case UsdGenDeviceChannelSemantic::HairT: return "hairT";
    case UsdGenDeviceChannelSemantic::ElementOffsets: return "curveOffsets";
    case UsdGenDeviceChannelSemantic::StableIds: return "curveId";
    case UsdGenDeviceChannelSemantic::RootPrim: return "skinprim";
    case UsdGenDeviceChannelSemantic::RootUV: return "skinprimuv";
    }
    CHECK(false);
    return {};
}

static VulkanSourcePrepareInfo SourceFixture() {
    VulkanSourcePrepareInfo in;
    auto& b = in.source;
    b.totalCurves = 2; b.totalCvs = 4; b.topologyVersion = 11; b.valueVersion = 23;
    b.px = {0,1,2,3}; b.py = {0,1,0,1}; b.pz = {0,0,1,1};
    b.rest = usdGen::UsdGenVec3fArray(4, usdGen::UsdGenVec3f(2,3,4)); b.width = usdGen::UsdGenFloatArray(4, 1.f);
    b.hairT = {0,.5f,.5f,1}; b.curveId = {5,9}; b.cvOffsets = {0,2,4};
    b.rootPrim = {3,7}; b.rootUV = {usdGen::UsdGenVec2f(.2f,.3f),usdGen::UsdGenVec2f(.4f,.5f)};
    b.rootT = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(1,0,0));
    b.rootB = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(0,1,0));
    b.rootN = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(0,0,1));
    in.geometry.topologyVersion = 11; in.geometry.valueVersion = 23;
    in.geometry.elementCount = 2; in.geometry.pointCount = 4;
    in.geometry.alreadyDeformed = true;
    in.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic,
        UsdGenCurveLayoutBasis::BSpline, UsdGenCurveLayoutWrap::Pinned);
    UsdGenDeviceTileMetadata tile; tile.tile = 4; tile.elementCount = 2;
    tile.pointCount = 4; tile.boundsValid = true; tile.extentMin = {0,0,0};
    tile.extentMax = {3,1,1}; in.geometry.tiles = {tile};
    uint32_t guide[] = {3,7};
    in.additionalNamed.push_back({{TestToken("guideId"), UsdGenDeviceValueType::UInt32,
        UsdGenDeviceDomain::Primitive, 2, 1, 4, true,
        UsdGenDeviceChannelSemantic::Generic}, Bytes(guide, 2)});
    return in;
}

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE,
                        10000000000ull) == VK_SUCCESS;
}

// The test copy has its own native proof. No scheduled owner waits for GPU
// completion; the service callback reads staging only after that proof.
struct ReadState {
    std::shared_ptr<NativeOwner> native;
    std::vector<std::shared_ptr<ChargedBuffer>> staging;
    std::vector<std::vector<uint8_t>> expected;
    std::unique_ptr<VulkanCompletionService::Watch> watch;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore gate = VK_NULL_HANDLE;
    bool proved = false;
    ~ReadState() {
        CHECK(proved);
        if(command) vkFreeCommandBuffers(native->device,native->commands,1,&command);
        if(fence) vkDestroyFence(native->device,fence,nullptr);
        if(gate) vkDestroySemaphore(native->device,gate,nullptr);
    }
};

int main(int argc, char** argv) {
    CHECK(argc==2 || (argc==3 && (std::string(argv[2])=="--close-held" ||
                                std::string(argv[2])=="--exit-held")));
    std::ifstream shader(argv[1],std::ios::binary); CHECK(shader);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)),{});
    CHECK(!raw.empty() && raw.size()%sizeof(uint32_t)==0);
    std::vector<uint32_t> code(raw.size()/sizeof(uint32_t));
    std::memcpy(code.data(),raw.data(),raw.size());
    bool unavailable = false;
    auto probe = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures supported{};
    supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{}; features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features.pNext = &supported; vkGetPhysicalDeviceFeatures2(probe->physical, &features);
    if (!supported.timelineSemaphore) return 77;
    probe.reset();
    VkPhysicalDeviceTimelineSemaphoreFeatures enabled{};
    enabled.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    enabled.timelineSemaphore = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, &enabled); CHECK(native);
    DeviceContext::CreateInfo contextInfo;
    contextInfo.instance = native->instance; contextInfo.physicalDevice = native->physical;
    contextInfo.device = native->device; contextInfo.computeQueue = native->queue;
    contextInfo.computeQueueFamily = native->family; contextInfo.physicalIndex = native->physicalIndex;
    contextInfo.resourceDeviceId = 7020; contextInfo.nativeLifetime = native;
    contextInfo.timelineSemaphoreEnabled = true; contextInfo.resources = {size_t{8} << 20, 0};
    auto context = DeviceContext::Create(contextInfo); CHECK(context);
    auto fixture = SourceFixture();
    std::string reason;
    auto missingContext = VulkanSourceUpload::Create(
        VulkanSourceGenerationCreateInfo{nullptr, fixture.source, fixture.geometry,
                                         fixture.additionalNamed}, nullptr, &reason);
    CHECK(!missingContext);
    auto prepared = VulkanPreparedSource::Prepare(std::move(fixture), &reason);
    CHECK(prepared);
    CHECK(prepared->topologyVersion() == 11 && prepared->valueVersion() == 23);
    CHECK(prepared->geometry().alreadyDeformed);
    CHECK(UsdGenCurveLayoutTypeOf(prepared->geometry().layoutMetadata) == UsdGenCurveLayoutType::Cubic);
    CHECK(UsdGenCurveLayoutBasisOf(prepared->geometry().layoutMetadata) == UsdGenCurveLayoutBasis::BSpline);
    CHECK(prepared->geometry().tiles.size() == 1 && prepared->geometry().tiles[0].boundsValid);

    bool sawT = false, sawB = false, sawN = false, sawPublicT = false;
    for (auto const& plane : prepared->planes()) {
        CHECK((plane.metadata.name != "sourceRootT" && plane.metadata.name != "sourceRootB" && plane.metadata.name != "sourceRootN") || plane.privateFrame);
        sawT |= plane.metadata.name == "sourceRootT" && plane.privateFrame;
        sawB |= plane.metadata.name == "sourceRootB" && plane.privateFrame;
        sawN |= plane.metadata.name == "sourceRootN" && plane.privateFrame;
        sawPublicT |= plane.metadata.name == "points" || plane.metadata.name == "rest";
    }
    CHECK(sawT && sawB && sawN && sawPublicT);

    // Exercise the real native upload owner used by the eventual adapter.
    VulkanSourceGenerationCreateInfo uploadInfo;
    uploadInfo.context = context; uploadInfo.source = SourceFixture().source;
    uploadInfo.geometry = prepared->geometry();
    uploadInfo.additionalNamed = SourceFixture().additionalNamed;
    VkResult uploadStatus = VK_SUCCESS;
    auto upload = VulkanSourceUpload::Create(std::move(uploadInfo), &uploadStatus, &reason);
    CHECK(upload && uploadStatus == VK_SUCCESS);
    CHECK(upload->Submit() == SourceGenerationStatus::Submitted);
    CHECK(Prove(native));
    CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    auto nativeGeneration = upload->TakeReady(); CHECK(nativeGeneration);
    CHECK(nativeGeneration->geometry().alreadyDeformed && nativeGeneration->planes().size() >= 3);
    CHECK(nativeGeneration->sourceFrames().size() == 3);
    for (auto const& plane : nativeGeneration->planes())
        CHECK(plane.metadata.name != "sourceRootT" && plane.metadata.name != "sourceRootB" && plane.metadata.name != "sourceRootN");

    auto widthPipeline=WidthPipeline::Create(context,code); CHECK(widthPipeline);
    auto candidate=widthPipeline->Begin(nativeGeneration->PlaneOwner(TestToken("width")),4,2.f,0);
    CHECK(candidate && Prove(native));
    uint32_t semantic=UINT32_MAX;
    CHECK(candidate->Poll(&semantic)==VK_SUCCESS && semantic==0);
    auto child=VulkanSourceGeneration::WithWidth(nativeGeneration,*candidate,24,&reason); CHECK(child);
    candidate.reset(); widthPipeline.reset(); upload.reset();
    auto const physicalBytes=context->resources()->Snapshot().usedBytes;
    CHECK(physicalBytes==child->InclusiveRetainedBytes());

    UsdGenExecutionRuntime runtime{2};
    auto queueOwner = std::make_shared<UsdGenExecutionPipeline>(runtime);
    auto service = VulkanCompletionService::Create({context, queueOwner.get(), 3});
    CHECK(service);
    auto domain=VulkanGenerationAdapterDomain::Create({queueOwner,service},&reason); CHECK(domain);
    if (argc == 3) {
        bool const atExit = std::string(argv[2]) == "--exit-held";
        if (atExit) CHECK(std::atexit(VerifyHeldExit) == 0);
        auto exitDomain = TrackVulkanExternalRetirementDomain(queueOwner, service); CHECK(exitDomain);
        auto const stream = reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
        auto publication = VulkanDeviceGenerationAdapter::Create({nativeGeneration, domain,
            stream, 41, {"held-exit", 17, 41}}, &reason); CHECK(publication);
        VulkanGenerationLease held;
        queueOwner->InvokeOwner([&] { held = AcquireVulkanGeneration(publication, stream); CHECK(held); });
        CHECK(!domain->Close() && domain->IsClosing());
        auto retired = std::make_shared<std::atomic<bool>>(false);
        CHECK(domain->RetireWhenIdle([domain, service, retired, exitDomain] {
            CHECK(domain->Close());
            service->CloseAndJoin();
            retired->store(true);
        }));
        if (atExit) {
            // No cleanup callback is queued: the weak service registry must
            // still stop the native waiter while this real consumer is held.
            heldExitCheck = new HeldExitCheck{exitDomain, domain, retired, std::move(held)};
            CHECK(!exitDomain->IsQuiesced());
            return 0;
        }
        // A held consumer does not keep process shutdown open. Its deferred
        // callback remains owned by the domain, outside the accepted queue.
        CloseVulkanExternalRetirement();
        CHECK(!retired->load() && !domain->IsClosed());
        // Close must join this waiter even though no consumer proof became
        // ready. Late consumer ingress fails without releasing native state.
        CHECK(exitDomain->IsQuiesced());
        CHECK(!queueOwner->PostCommand([] { std::abort(); }));
        held.Complete();
        CHECK(!held && !retired->load() && !domain->IsClosed());
        return 0;
    }
    CHECK(!VulkanDeviceGenerationAdapter::Create({nativeGeneration, {},
        reinterpret_cast<UsdGenDeviceStream>(context->computeQueue()), 41,
        {"bad-service", 0, 41}}, &reason));
    CHECK(!VulkanDeviceGenerationAdapter::Create({nativeGeneration, domain,
        1, 41, {"bad-stream", 0, 41}}, &reason));
    auto publication = VulkanDeviceGenerationAdapter::Create({
        nativeGeneration, domain,
        reinterpret_cast<UsdGenDeviceStream>(context->computeQueue()), 41,
        {"vulkan-source", 17, 41}}, &reason);
    CHECK(publication && publication->Identity().backend == usdGenTestUtils::Backend("vulkan"));
    CHECK(publication->Identity().generation == 41 && publication->Tool().snapshotVersion == 41);
    CHECK(publication->Geometry().tiles[0].extentMax[0] == 3);
    size_t publicCount = 0;
    for (auto const& plane : nativeGeneration->planes()) {
        if (plane.metadata.name == "sourceRootT" || plane.metadata.name == "sourceRootB" ||
            plane.metadata.name == "sourceRootN") continue;
        CHECK(publicCount < publication->Channels().size());
        auto const& actual = publication->Channels()[publicCount++];
        CHECK(actual.name == PublicName(plane.metadata) && actual.type == plane.metadata.type &&
              actual.domain == plane.metadata.domain && actual.elementCount == plane.metadata.elementCount &&
              actual.arity == plane.metadata.arity && actual.strideBytes == plane.metadata.strideBytes &&
              actual.semantic == plane.metadata.semantic && actual.readOnly == plane.metadata.readOnly);
    }
    CHECK(publicCount == publication->Channels().size());
    CHECK(publication->ExclusiveRetainedBytes() > 0 &&
          publication->InclusiveRetainedBytes() >= publication->ExclusiveRetainedBytes());
    auto republished = publication->Republish(42, &reason);
    CHECK(republished && republished->Identity().generation == 42);
    CHECK(republished->Owner() == publication->Owner());
    auto childPublication=VulkanDeviceGenerationAdapter::Create({child,domain,
        reinterpret_cast<UsdGenDeviceStream>(context->computeQueue()),43,{"vulkan-source",17,43}},&reason);
    CHECK(childPublication && childPublication->Geometry().valueVersion==24);
    CHECK(childPublication->ExclusiveRetainedBytes()==child->ExclusiveRetainedBytes());
    CHECK(childPublication->InclusiveRetainedBytes()==physicalBytes);
    CHECK(context->resources()->Snapshot().usedBytes==physicalBytes);
    auto const stream=reinterpret_cast<UsdGenDeviceStream>(context->computeQueue());
    CHECK(!AcquireVulkanGeneration(publication,stream)); // Off-owner admission.
    {
        UsdGenExecutionPipeline otherOwner(runtime);
        otherOwner.InvokeOwner([&] {
            CHECK(!AcquireVulkanGeneration(publication,stream));
            CHECK(!domain->AcquireOperation());
        });
        CHECK(queueOwner->OutstandingCommands()==0);
    }
    auto foreign = UsdGenDeviceGeneration::Create({
        {kUsdGenBackendCpu, 0, 41}, publication->Geometry(),
        publication->Tool(), publication->Channels(), publication->Owner()}, &reason);
    CHECK(foreign);
    std::weak_ptr<const VulkanSourceGeneration> weakBase=nativeGeneration,weakChild=child;
    std::weak_ptr<UsdGenExecutionPipeline> weakQueue=queueOwner;
    std::weak_ptr<VulkanGenerationAdapterDomain> weakDomain=domain;
    auto* queueRaw=queueOwner.get();
    VulkanGenerationLease baseLease,childLease;
    auto reads=std::make_shared<ReadState>(); reads->native=native;
    queueOwner->Await([&](auto finish) {
        queueOwner->InvokeOwner([&] {
            CHECK(!AcquireVulkanGeneration(foreign,stream));
            CHECK(!AcquireVulkanGeneration(publication,stream+1));
            CHECK(!domain->Close()); // A graph callback cannot close a domain.
            CHECK(!domain->IsClosing());
            baseLease=AcquireVulkanGeneration(publication,stream);
            childLease=AcquireVulkanGeneration(childPublication,stream);
            CHECK(baseLease && childLease && baseLease.WaitUntilReady()==UsdGenDeviceStatus::Ok);
            CHECK(childLease.WaitUntilReady()==UsdGenDeviceStatus::Ok);
            auto copied=baseLease;
            baseLease.Complete(); CHECK(!baseLease && baseLease.Planes().empty());
            baseLease=std::move(copied); CHECK(baseLease);
            CHECK(baseLease.Planes().size()==childLease.Planes().size());
            for(size_t i=0;i<baseLease.Planes().size();++i) {
                auto const& a=baseLease.Planes()[i];auto const& b=childLease.Planes()[i];
                CHECK(a.metadata.name==b.metadata.name);
                CHECK((a.buffer==b.buffer)==(a.metadata.name!="widths"));
            }
            reads->watch=service->Reserve(reads,[&,reads,finish](VkResult proof) {
                CHECK(queueRaw->IsExecutingOwner() && proof==VK_SUCCESS);
                for(size_t i=0;i<reads->staging.size();++i) {
                    auto const& buffer=reads->staging[i];
                    CHECK(buffer->PollComplete()==VK_SUCCESS);
                    void* mapped=nullptr;
                    CHECK(vkMapMemory(native->device,buffer->memory(),0,reads->expected[i].size(),0,&mapped)==VK_SUCCESS);
                    CHECK(std::memcmp(mapped,reads->expected[i].data(),reads->expected[i].size())==0);
                    vkUnmapMemory(native->device,buffer->memory());
                }
                CHECK(reads->watch->Retire()); reads->watch.reset();
                reads->proved=true;
                finish();
                return VulkanCompletionService::DeliveryResult::Posted;
            },[](VkResult){CHECK(false);});
            CHECK(reads->watch);
            auto const credits=queueOwner->OutstandingCommands();
            CHECK(!AcquireVulkanGeneration(publication,stream)); // Three slots exhausted.
            CHECK(queueOwner->OutstandingCommands()==credits);
            VkFenceCreateInfo fi{};fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            CHECK(vkCreateFence(native->device,&fi,nullptr,&reads->fence)==VK_SUCCESS);
            VkSemaphoreTypeCreateInfo ti{};ti.sType=VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            ti.semaphoreType=VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo si{};si.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;si.pNext=&ti;
            CHECK(vkCreateSemaphore(native->device,&si,nullptr,&reads->gate)==VK_SUCCESS);
            VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool=native->commands;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
            CHECK(vkAllocateCommandBuffers(native->device,&ai,&reads->command)==VK_SUCCESS);
            VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            CHECK(vkBeginCommandBuffer(reads->command,&begin)==VK_SUCCESS);
            VkMemoryBarrier before{};before.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
            before.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(reads->command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,1,&before,0,nullptr,0,nullptr);
            for(auto const* lease:{&baseLease,&childLease}) for(auto const& plane:lease->Planes()) {
                auto expected=std::find_if(prepared->planes().begin(),prepared->planes().end(),[&](auto const& p){
                    return !p.privateFrame && p.metadata.semantic == plane.metadata.semantic &&
                        PublicName(p.metadata) == plane.metadata.name;
                });
                CHECK(expected!=prepared->planes().end() && expected->bytes.size()==plane.bytes);
                if(!plane.bytes) {CHECK(!plane.buffer);continue;}
                auto bytes=expected->bytes;
                if(lease==&childLease && plane.metadata.name=="widths") {
                    float widths[4]={2.f,2.f,2.f,2.f};bytes=Bytes(widths,4);
                }
                VkBufferCreateInfo bi{};bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                bi.size=plane.bytes;bi.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
                auto staging=ChargedBuffer::Create(context,bi,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    UsdGenExecutionResourceKind::Scratch);CHECK(staging);
                CHECK(staging->MarkSubmitted(reads->fence,native)==VK_SUCCESS);
                VkBufferCopy region{0,0,plane.bytes};
                vkCmdCopyBuffer(reads->command,plane.buffer,staging->buffer(),1,&region);
                reads->staging.push_back(std::move(staging));reads->expected.push_back(std::move(bytes));
            }
            VkMemoryBarrier after{};after.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(reads->command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
                0,1,&after,0,nullptr,0,nullptr);
            CHECK(vkEndCommandBuffer(reads->command)==VK_SUCCESS);
            uint64_t waitValue=1;
            VkTimelineSemaphoreSubmitInfo timeline{};timeline.sType=VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timeline.waitSemaphoreValueCount=1;timeline.pWaitSemaphoreValues=&waitValue;
            VkPipelineStageFlags waitStage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;submit.pNext=&timeline;
            submit.waitSemaphoreCount=1;submit.pWaitSemaphores=&reads->gate;submit.pWaitDstStageMask=&waitStage;
            submit.commandBufferCount=1;submit.pCommandBuffers=&reads->command;
            CHECK(reads->watch->MarkPhaseSubmitted());
            CHECK(vkQueueSubmit(native->queue,1,&submit,reads->fence)==VK_SUCCESS);
        });
        // The GPU is held at a timeline gate. Only the leases now own source
        // and COW child storage; no generation wrapper or native upload does.
        publication.reset();republished.reset();childPublication.reset();foreign.reset();
        nativeGeneration.reset();child.reset();
        CHECK(!weakBase.expired() && !weakChild.expired());
        CHECK(!domain->Close() && domain->IsClosing() && !domain->IsClosed());
        baseLease.Complete();childLease.Complete(); // Thread-neutral retirement ingress.
        queueOwner->InvokeOwner([&]{CHECK(service->Arm(*reads->watch,100));});
        // Premature external last-drop must neither free the raw queue route
        // nor transfer pipeline destruction into a completion callback.
        queueOwner.reset();domain.reset();
        CHECK(!weakQueue.expired() && !weakDomain.expired());
        VkSemaphoreSignalInfo signal{};signal.sType=VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signal.semaphore=reads->gate;signal.value=1;
        CHECK(vkSignalSemaphore(native->device,&signal)==VK_SUCCESS);
    });
    queueOwner=weakQueue.lock();domain=weakDomain.lock();
    CHECK(queueOwner && domain);
    queueOwner->Drain();
    CHECK(reads->proved && weakChild.expired() && weakBase.expired());
    reads.reset();
    CHECK(context->resources()->Snapshot().usedBytes==0);
    CHECK(queueOwner->OutstandingCommands()==0 && queueOwner->CallbackFailures()==0);
    CHECK(domain->Close() && domain->IsClosed() && domain->Close());
    service->CloseAndJoin();
    queueOwner->Drain();
    domain.reset(); service.reset(); queueOwner.reset();
    std::puts("testUsdGenVulkanDeviceGenerationAdapter: PASS");
    return 0;
}
