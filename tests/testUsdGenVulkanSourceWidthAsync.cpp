// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceWidthJob.h"
#include "vulkanReadbackFixture.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"Vulkan async check failed: %s (%d)\n",#x,__LINE__); return 1; } } while(false)
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context=std::move(context);
    auto& b=info.source;
    b.totalCurves=1; b.totalCvs=2; b.topologyVersion=3; b.valueVersion=4;
    b.px={0,1}; b.py={0,1}; b.pz={0,1}; b.curveId={9}; b.cvOffsets={0,2}; b.width={1.f,2.f};
    return info;
}
static std::vector<uint32_t> ReadSpirv(char const* path) {
    std::ifstream shader(path,std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)),{});
    if(!shader || raw.empty() || raw.size()%4!=0) return {};
    std::vector<uint32_t> code(raw.size()/4); std::memcpy(code.data(),raw.data(),raw.size());
    return code;
}
int main(int argc,char** argv) {
    CHECK(argc>=2);
    bool preparedMode=false,early=false;
    std::vector<uint32_t> tilesCode,boundsCode;
    int flag0=2;
    if(argc>=5 && std::string(argv[2])=="--tiles") {
        tilesCode=ReadSpirv(argv[3]); boundsCode=ReadSpirv(argv[4]);
        CHECK(!tilesCode.empty() && !boundsCode.empty()); flag0=5;
    }
    for(int i=flag0;i<argc;++i) {
        if(std::string(argv[i])=="--prepared")preparedMode=true;
        else if(std::string(argv[i])=="--early-notifications")early=true;
        else CHECK(false);
    }
    std::vector<uint32_t> code=ReadSpirv(argv[1]); CHECK(!code.empty());
    bool unavailable=false; auto probe=CreateNative(&unavailable);
    if(unavailable)return 77;
    CHECK(probe);
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
    timeline.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    VkPhysicalDeviceFeatures2 features{};
    features.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2; features.pNext=&timeline;
    vkGetPhysicalDeviceFeatures2(probe->physical,&features);
    if(!timeline.timelineSemaphore)return 77;
    probe.reset(); std::fprintf(stderr,"async: probe complete\n");
    auto native=CreateNative(&unavailable,{},&timeline); CHECK(native);
    std::fprintf(stderr,"async: enabled device\n");
    DeviceContext::CreateInfo ci;
    ci.instance=native->instance; ci.physicalDevice=native->physical; ci.device=native->device;
    ci.computeQueue=native->queue; ci.computeQueueFamily=native->family; ci.physicalIndex=native->physicalIndex;
    ci.resourceDeviceId=7017; ci.nativeLifetime=native; ci.resources={size_t{8}<<20,0}; ci.timelineSemaphoreEnabled=true;
    auto context=DeviceContext::Create(ci); CHECK(context);
    auto pool=context->resources(); auto pipeline=WidthPipeline::Create(context,code); CHECK(pipeline);
    std::shared_ptr<PicktileVkTilesPipeline> tilesPipeline;
    std::shared_ptr<PicktileVkBoundsPipeline> boundsPipeline;
    if(!tilesCode.empty()) {
        tilesPipeline=PicktileVkTilesPipeline::Create(context,tilesCode); CHECK(tilesPipeline);
        boundsPipeline=PicktileVkBoundsPipeline::Create(context,boundsCode); CHECK(boundsPipeline);
    }
    UsdGenExecutionRuntime runtime{4}; UsdGenExecutionPipeline owner(runtime);
    auto service=VulkanCompletionService::Create({context,&owner,16}); CHECK(service);
    struct CloseService { std::shared_ptr<VulkanCompletionService> service; ~CloseService(){service->CloseAndJoin();} } closeService{service};
    std::shared_ptr<UsdGenExecutionTaskGraph> preparer;
    if(preparedMode) {preparer=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"vulkan-source-preparation",7017);CHECK(preparer);}
    std::fprintf(stderr,"async: service created\n");
    auto makeInfo=[&](unsigned i) {
        VulkanSourceWidthJob::CreateInfo info; info.source=Source(context); info.widthPipeline=pipeline;
        info.width=float(i+3); info.replace=1; info.valueVersion=5+i;
        info.requestLifetime=std::make_shared<unsigned>(i); info.completionService=service; info.preparer=preparer;
        info.tilesPipeline=tilesPipeline; info.boundsPipeline=boundsPipeline;
        return info;
    };
    {
        UsdGenExecutionPipeline foreign(runtime);
        auto wrongOwner=VulkanSourceWidthJob::Create(makeInfo(0)); CHECK(wrongOwner);
        CHECK(!wrongOwner->Start(foreign,{},[](auto,auto,VkResult){}));
        CHECK(foreign.OutstandingCommands()==0);
    }
    bool suppressedOk=false; auto suppressedInfo=makeInfo(0);
    std::weak_ptr<const void> suppressedLifetime=suppressedInfo.requestLifetime;
    auto suppressed=VulkanSourceWidthJob::Create(std::move(suppressedInfo)); CHECK(suppressed);
    suppressed->SuppressPublication();
    owner.Await([&](auto finish) {
        if(!suppressed->Start(owner,{},[&,finish](auto result,auto state,VkResult status) {
            suppressedOk=owner.IsExecutingOwner() && !result && state==VulkanSourceWidthJob::State::Superseded &&
                status==VK_SUCCESS && !suppressedLifetime.expired(); finish();
        }))finish();
    });
    owner.Drain(); CHECK(suppressedOk && suppressedLifetime.expired());
    owner.InvokeOwner([&]{suppressed.reset();}); CHECK(pool->Snapshot().usedBytes==0);
    std::fprintf(stderr,"async: suppression done\n");
    constexpr size_t count=8;
    std::array<std::shared_ptr<VulkanSourceWidthJob>,count> jobs;
    std::array<std::shared_ptr<const VulkanSourceGeneration>,count> results;
    std::array<std::weak_ptr<const void>,count> lifetimes;
    std::array<unsigned,count> calls{};
    std::atomic<unsigned> remaining{count}; std::atomic<bool> ok{true};
    for(size_t i=0;i<count;++i) {auto info=makeInfo(unsigned(i));lifetimes[i]=info.requestLifetime;jobs[i]=VulkanSourceWidthJob::Create(std::move(info));CHECK(jobs[i]);}
    auto epoch=owner.AcceptedEpoch();
    std::fprintf(stderr,"async: batch start prepared=%d early=%d\n",preparedMode,early);
    owner.Await([&](auto finish) {
        for(size_t i=0;i<count;++i) {
            if(!jobs[i]->Start(owner,{},[&,i,finish](auto generation,auto state,VkResult status) {
                ++calls[i];
                if(!owner.IsExecutingOwner() || state!=VulkanSourceWidthJob::State::Ready || status!=VK_SUCCESS || !generation || lifetimes[i].expired())ok.store(false);
                results[i]=std::move(generation); if(remaining.fetch_sub(1)==1)finish();
            })) {ok.store(false);if(remaining.fetch_sub(1)==1)finish();}
            if(early)for(unsigned n=0;n<32;++n)jobs[i]->NotifyCompletion();
        }
    });
    std::fprintf(stderr,"async: batch done\n");
    if(preparer)preparer->Drain();
    service->CloseAndJoin(); owner.Drain(); std::fprintf(stderr,"async: service joined\n");
    CHECK(ok.load() && owner.OutstandingCommands()==0 && owner.AcceptedEpoch()==epoch);
    for(size_t i=0;i<count;++i) {
        CHECK(calls[i]==1 && lifetimes[i].expired() && results[i]);
        CHECK(results[i]->context()==context && results[i]->valueVersion()==5+i && results[i]->topologyVersion()==3 && results[i]->pointCount()==2);
        std::vector<uint8_t> bytes;
        CHECK(results[i]->PlaneOwner("width") && results[i]->PlaneOwner("points"));
        CHECK(ReadVulkanBytes(native,context,results[i]->PlaneOwner("width")->buffer(),8,results[i],&bytes));
        CHECK(bytes.size()==8);
        float widths[2]{};std::memcpy(widths,bytes.data(),sizeof(widths));CHECK(widths[0]==float(i+3) && widths[1]==float(i+3));
        CHECK(ReadVulkanBytes(native,context,results[i]->PlaneOwner("points")->buffer(),24,results[i],&bytes));
        CHECK(bytes.size()==24);
        float points[6]{};std::memcpy(points,bytes.data(),sizeof(points));
        CHECK(points[0]==0 && points[1]==0 && points[2]==0 && points[3]==1 && points[4]==1 && points[5]==1);
        if(tilesPipeline) {
            auto const& tiles=results[i]->geometry().tiles;
            CHECK(tiles.size()==1 && tiles[0].tile==0 && tiles[0].boundsValid);
            CHECK(tiles[0].firstElement==0 && tiles[0].elementCount==1);
            CHECK(tiles[0].firstPoint==0 && tiles[0].pointCount==2);
        }
    }
    owner.InvokeOwner([&]{results={};jobs={};}); CHECK(pool->Snapshot().usedBytes==0);
    return 0;
}
