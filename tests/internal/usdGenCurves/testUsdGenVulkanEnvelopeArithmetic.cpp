// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "internal/usdGenVulkan/vulkanReadbackFixture.h"
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"Arithmetic check failed: %s (%d)\n",#x,__LINE__);std::abort();} } while(false)
static uint32_t Bits(float value) { uint32_t bits; std::memcpy(&bits,&value,4); return bits; }
static float Float(uint32_t bits) { float value; std::memcpy(&value,&bits,4); return value; }
static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path,std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
    CHECK(!bytes.empty()&&bytes.size()%4==0);
    std::vector<uint32_t> code(bytes.size()/4); std::memcpy(code.data(),bytes.data(),bytes.size()); return code;
}
int main(int argc,char** argv) {
    CHECK(argc==9&&std::fesetround(FE_TONEAREST)==0);
    std::vector<std::vector<uint32_t>> code;
    for(int i=1;i<argc;++i) code.push_back(Code(argv[i]));
    bool unavailable=false; auto native=CreateTimelineNative(&unavailable); if(unavailable)return 77; CHECK(native);
    VkPhysicalDeviceProperties properties{}; vkGetPhysicalDeviceProperties(native->physical,&properties);
    if(properties.limits.maxPerStageDescriptorStorageBuffers<6||
       properties.limits.maxDescriptorSetStorageBuffers<6||properties.limits.maxPushConstantsSize<56)return 77;
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true; ci.instance=native->instance;ci.physicalDevice=native->physical;
    ci.device=native->device;ci.computeQueue=native->queue;ci.computeQueueFamily=native->family;
    ci.physicalIndex=native->physicalIndex;ci.resourceDeviceId=7937;ci.nativeLifetime=native;
    ci.resources={size_t{32}<<20,0};
    auto context=DeviceContext::Create(ci);CHECK(context);auto baseline=context->resources()->Snapshot();
    {
        // Test-only shader, exact production helper and native dispatch/proof.
        auto pipeline=LengthScalePipeline::CreateWithEnvelopeV1(context,code[0],code[1],code[2],
            code[3],code[4],code[5],code[7]);CHECK(pipeline);
        std::vector<uint32_t> input,expected;
        auto addCase=[&](uint32_t a,uint32_t b,uint32_t e) {
            input.insert(input.end(),{a,b,e});
            volatile float av=Float(a),bv=Float(b),ev=Float(e);
            volatile float sum=av+bv,product=av*bv,delta=bv-av;
            volatile float weighted=delta*ev,result=av+weighted;
            expected.insert(expected.end(),{Bits(sum),Bits(product),Bits(result)});
        };
        std::vector<uint32_t> edges={0,0x80000000u,1,0x80000001u,0x007fffffu,0x807fffffu,
            0x00800000u,0x80800000u,0x3f800000u,0xbf800000u,0x3f800001u,0x3f7fffffu,
            0x33800000u,0x34000000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc00001u};
        for(auto a:edges)for(auto b:edges)for(auto e:{0u,1u,0x00400000u,0x3f000000u,0x3f800000u})addCase(a,b,e);
        uint32_t state=0x72e49719u;
        auto random=[&]{state^=state<<13;state^=state>>17;state^=state<<5;return state;};
        for(unsigned i=0;i<20000;++i){auto a=random(),b=random(),e=random()%0x3f800001u;addCase(a,b,e);}
        auto upload=[&](std::vector<uint32_t> const& data) {
            VkBufferCreateInfo info{};info.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size=data.size()*4;info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
            auto buffer=ChargedBuffer::Create(context,info,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,UsdGenExecutionResourceKind::Scratch);CHECK(buffer);
            void* mapped=nullptr;CHECK(vkMapMemory(native->device,buffer->memory(),0,info.size,0,&mapped)==VK_SUCCESS);
            std::memcpy(mapped,data.data(),info.size);vkUnmapMemory(native->device,buffer->memory());return buffer;
        };
        uint32_t count=uint32_t(input.size()/3);
        auto points=upload(input),offsets=upload({0,count}),ids=upload({7,0});
        VkResult status=VK_SUCCESS;
        auto candidate=pipeline->BeginEnvelopeV1(points,offsets,{},ids,1,count,{},&status);
        CHECK(candidate&&status==VK_SUCCESS);
        CHECK(vkResetFences(native->device,1,&native->fence)==VK_SUCCESS);
        VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        CHECK(vkQueueSubmit(native->queue,1,&submit,native->fence)==VK_SUCCESS&&
              vkWaitForFences(native->device,1,&native->fence,VK_TRUE,10000000000ull)==VK_SUCCESS);
        uint32_t semantic=UINT32_MAX;CHECK(candidate->Poll(&semantic)==VK_SUCCESS&&semantic==0);
        auto output=candidate->output();CHECK(output);std::vector<uint8_t> bytes;
        CHECK(ReadVulkanBytes(native,context,output->buffer(),output->sizeBytes(),output,&bytes)&&bytes.size()==expected.size()*4);
        for(size_t i=0;i<expected.size();++i) {
            uint32_t actual;std::memcpy(&actual,bytes.data()+i*4,4);
            bool bothNan=std::isnan(Float(actual))&&std::isnan(Float(expected[i]));
            if(actual!=expected[i]&&!bothNan) {
                size_t k=(i/3)*3;
                std::fprintf(stderr,"case=%zu op=%zu a=%08x b=%08x e=%08x actual=%08x expected=%08x\n",
                    i/3,i%3,input[k],input[k+1],input[k+2],actual,expected[i]);
                CHECK(false);
            }
        }
        std::printf("GPU binary32 arithmetic: %u cases, %zu exact results PASS\n",count,expected.size());
    }
    auto final=context->resources()->Snapshot();
    CHECK(final.usedBytes==baseline.usedBytes&&final.byKind==baseline.byKind);
    return 0;
}
