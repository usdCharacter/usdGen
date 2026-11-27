// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"
#include "floatUlpFixture.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>
using namespace usdGen; using namespace usdGen::vulkan;
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"Vulkan DeviceV1 failed: %s (%d)\n",#x,__LINE__);std::abort();} } while(false)
namespace {
template<class T> std::vector<uint8_t> Bytes(T const*p,size_t n){std::vector<uint8_t>r(n*sizeof(T));if(n)std::memcpy(r.data(),p,r.size());return r;}
static std::vector<uint32_t> Code(char const*p){std::ifstream f(p,std::ios::binary);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});if(b.empty()||b.size()%4)return{};std::vector<uint32_t>r(b.size()/4);std::memcpy(r.data(),b.data(),b.size());return r;}
static bool Prove(std::shared_ptr<NativeOwner>const&n){if(vkResetFences(n->device,1,&n->fence)!=VK_SUCCESS)return false;VkSubmitInfo s{};s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;return vkQueueSubmit(n->queue,1,&s,n->fence)==VK_SUCCESS&&vkWaitForFences(n->device,1,&n->fence,VK_TRUE,10000000000ull)==VK_SUCCESS;}
using Packet=std::map<std::string,std::vector<uint8_t>>;
static bool Capture(std::shared_ptr<NativeOwner>const&n,std::shared_ptr<DeviceContext>const&c,std::shared_ptr<const VulkanSourceGeneration>const&g,Packet*out){out->clear();auto add=[&](VulkanSourceGeneration::PlaneView const&p){std::vector<uint8_t>b;return ReadVulkanBytes(n,c,p.buffer,p.bytes,g,&b)&&out->emplace(p.metadata.name,std::move(b)).second;};for(auto const&p:g->planes())if(!add(p))return false;for(auto const&p:g->sourceFrames())if(!add(p))return false;return true;}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext>c,std::vector<uint64_t> ids={0,1},bool bent=false,bool zero=false){VulkanSourceGenerationCreateInfo i;i.context=std::move(c);auto&s=i.source;s.totalCurves=2;s.totalCvs=bent?7:4;s.topologyVersion=95;s.valueVersion=17;s.cvOffsets=bent?VtIntArray{0,3,7}:VtIntArray{0,2,4};s.curveId=VtArray<uint64_t>(ids.begin(),ids.end());s.rootPrim={12,44};s.rootUV={{.2f,.3f},{.7f,.8f}};s.rootT={{1,0,0},{0,1,0}};s.rootB={{0,1,0},{0,0,1}};s.rootN={{0,0,1},{1,0,0}};UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{.25f,.75f}};if(bent){s.px=zero?VtFloatArray(7,0):VtFloatArray{0,1,1,10,11,12,14};s.py=zero?VtFloatArray(7,0):VtFloatArray{0,0,3,0,0,0,0};s.hairT={0,.75f,1,.5f,0,.75f,.25f};}else{s.px=zero?VtFloatArray(4,0):VtFloatArray{0,1,0,1};s.py=VtFloatArray(4,0);s.hairT={0,1,0,1};}s.pz=VtFloatArray(s.totalCvs,0);s.rest.resize(s.totalCvs);s.width=VtFloatArray(s.totalCvs,1);UsdGenChunkDesc x;x.firstCurve=0;x.curveCount=2;x.liveCount=2;x.firstCv=0;x.cvCount=0;x.tile=3;x.surface=1;s.chunks={x};UsdGenPlane p;p.name=TfToken("privateVertex");p.interpolation=TfToken("vertex");p.type=TfToken("float");p.arity=1;p.f=VtFloatArray(s.totalCvs,3);s.extraCv={p};UsdGenPlane q;q.name=TfToken("privateUniform");q.interpolation=TfToken("uniform");q.type=TfToken("int");q.i={18,19};s.extraCurve={maskPlane, q};i.geometry.alreadyDeformed=true;i.geometry.layoutMetadata=UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic,UsdGenCurveLayoutBasis::BSpline,UsdGenCurveLayoutWrap::Pinned);i.geometry.tiles={{3,0,2,0,s.totalCvs}};uint32_t named[]={73,81};i.additionalNamed.push_back({{TfToken("namedCurve"),UsdGenDeviceValueType::UInt32,UsdGenDeviceDomain::Primitive,2,1,4,true},Bytes(named,2)});return i;}
static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner>const&n,VulkanSourceGenerationCreateInfo i){VkResult r=VK_SUCCESS;std::string why;auto u=VulkanSourceUpload::Create(std::move(i),&r,&why);if(!u||u->Submit()!=SourceGenerationStatus::Submitted||!Prove(n)||u->Poll()!=SourceGenerationStatus::Ready)return{};return u->TakeReady();}
}
int main(int argc,char**argv) {
    CHECK(argc==10);
    std::vector<std::vector<uint32_t>> code;
    for(int i=1;i<argc;++i){code.push_back(Code(argv[i]));CHECK(!code.back().empty());}
    bool unavailable=false;auto native=CreateNative(&unavailable);if(unavailable)return 77;CHECK(native);
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(native->physical,&props);
    if(props.limits.maxPerStageDescriptorStorageBuffers<13||props.limits.maxDescriptorSetStorageBuffers<13||props.limits.maxPushConstantsSize<88)return 77;
    DeviceContext::CreateInfo ci;ci.instance=native->instance;ci.physicalDevice=native->physical;
    ci.device=native->device;ci.computeQueue=native->queue;ci.computeQueueFamily=native->family;
    ci.physicalIndex=native->physicalIndex;ci.resourceDeviceId=7936;ci.nativeLifetime=native;
    ci.resources={size_t{32}<<20,0};auto context=DeviceContext::Create(ci);CHECK(context);
    auto pipe=LengthScalePipeline::CreateWithDeviceV1(context,code[0],code[1],code[2],code[3],code[4],code[5],code[6],code[7]);
    auto compact=LengthCompactionPipeline::CreateWithKeep(context,code[8]);CHECK(pipe&&pipe->HasDeviceV1()&&compact);
    auto base=Upload(native,Source(context,{0x8000000000000001ull,0xffffffffull},true));CHECK(base);
    Packet original;CHECK(Capture(native,context,base,&original));
    using Controls=LengthScalePipeline::DeviceV1Controls;
    auto field=[&](auto const& values,uint32_t domain){
        auto owner=UploadVulkanDeviceBytes(native,context,values.data(),values.size()*sizeof(values[0]));
        CHECK(owner);return LengthScalePipeline::DeviceField{owner,domain};
    };
    auto read=[&](std::shared_ptr<const ChargedBuffer> const& owner){std::vector<uint8_t> bytes;
        CHECK(owner&&ReadVulkanBytes(native,context,owner->buffer(),owner->sizeBytes(),owner,&bytes));return bytes;};
    auto points=[&](auto const& candidate,std::vector<float> const& expected){
        auto bytes=read(candidate->output());CHECK(bytes.size()==expected.size()*4);
        for(size_t i=0;i<expected.size();++i){float v;std::memcpy(&v,bytes.data()+4*i,4);CHECK(FloatBytesWithinUlps(&v,&expected[i],4,4));}
    };
    auto begin=[&](Controls const& controls){return pipe->BeginDeviceV1(base->PlaneOwner(TfToken("points")),
        base->PlaneOwner(TfToken("curveOffsets")),base->PlaneOwner(TfToken("hairT")),base->PlaneOwner(TfToken("stableIds")),2,7,controls);};
    auto prove=[&](auto const& candidate,bool success=true){CHECK(candidate&&Prove(native));uint32_t semantic=UINT32_MAX;
        CHECK(candidate->Poll(&semantic)==VK_SUCCESS&&candidate->succeeded()==success&&(semantic==0)==success);
        if(!success)CHECK(!candidate->output()&&!candidate->keep());};
    auto keep=[&](auto const& candidate,std::vector<uint32_t> const& expected){CHECK(read(candidate->keep())==Bytes(expected.data(),expected.size()));};

    Controls c;c.valueField=field(std::vector<float>{1,.5f,2,1,1,1,1},4);
    c.minimumField=field(std::vector<float>{0,3,0,0,0,0,0},4);
    c.maskField=field(std::vector<float>{0,1,.5f,1,1,1,1},4);
    c.randomField=field(std::vector<float>{1,1,2,2},2);
    c.enabledField=field(std::vector<uint32_t>{1,0},2);
    c.cullThresholdField=field(std::vector<float>{5.5f,100},2);
    auto candidate=begin(c);prove(candidate);
    points(candidate,{0,0,0,.75f,0,0,1.5f,4.5f,0,10,0,0,11,0,0,12,0,0,14,0,0});keep(candidate,{0,1});
    // Connected fields override fallback literals, including unused invalid values.
    auto overridden=c;overridden.value=overridden.minimum=overridden.randomLo=overridden.randomHi=
        overridden.maskAmount=overridden.cullThreshold=std::numeric_limits<float>::quiet_NaN();
    auto overrideCandidate=begin(overridden);prove(overrideCandidate);
    CHECK(read(overrideCandidate->output())==read(candidate->output()));keep(overrideCandidate,{0,1});
    auto child=VulkanSourceGeneration::WithPoints(base,*candidate,100);CHECK(child);
    auto oldCompact=LengthCompactionPipeline::Create(context,code[8]);CHECK(oldCompact&&!oldCompact->HasKeep());
    unsigned hookCalls=0;CHECK(!oldCompact->BeginWithKeep(child,candidate->keep(),nullptr,[&](){++hookCalls;return true;}));CHECK(hookCalls==0);
    auto scatter=compact->BeginWithKeep(child,candidate->keep());CHECK(scatter&&Prove(native));
    uint32_t status=UINT32_MAX;CHECK(scatter->PollCounts(&status)==VK_SUCCESS&&status==0&&scatter->curveCount()==1&&scatter->pointCount()==4);
    CHECK(scatter->BeginScatter()&&Prove(native)&&scatter->PollScatter(&status)==VK_SUCCESS&&status==0&&scatter->succeeded());
    auto culled=VulkanSourceGeneration::WithCompacted(child,*scatter,101);CHECK(culled&&culled->curveCount()==1);
    CHECK(read(culled->PlaneOwner(TfToken("stableIds")))==Bytes(std::vector<uint64_t>{0xffffffffull}.data(),1));
    CHECK(read(culled->PlaneOwner(TfToken("namedCurve")))==Bytes(std::vector<uint32_t>{81}.data(),1));

    // A zero mask preserves topology even under a threshold above the arc.
    c.maskField=field(std::vector<float>{0},1);auto zero=begin(c);prove(zero);keep(zero,{1,1});
    CHECK(read(zero->output())==original["points"]);
    // A primitive random range changes enabled curves independently.
    c=Controls{};c.randomField=field(std::vector<float>{1,1,2,2},2);auto random=begin(c);prove(random);
    points(random,{0,0,0,1,0,0,1,3,0,10,0,0,12,0,0,14,0,0,18,0,0});keep(random,{1,1});
    // Per-point Set targets and Reparam read each target and each hairT.
    c=Controls{};c.absolute=true;c.method=Controls::Method::CutExtend;c.rebuild=Controls::Rebuild::Reparam;
    c.valueField=field(std::vector<float>{0,2,6,4,4,4,4},4);auto reparam=begin(c);prove(reparam);
    points(reparam,{0,0,0,1,.5f,0,1,5,0,12,0,0,10,0,0,13,0,0,11,0,0});
    // Semantic failures never expose either staging output.
    c.enabledField=field(std::vector<uint32_t>{2,1},2);auto badBool=begin(c);prove(badBool,false);
    c=Controls{};c.cullThresholdField=field(std::vector<float>{std::numeric_limits<float>::quiet_NaN()},1);
    auto badThreshold=begin(c);prove(badThreshold,false);
    c=Controls{};c.maskField=field(std::vector<float>{std::numeric_limits<float>::quiet_NaN()},1);
    c.enabledField=field(std::vector<uint32_t>{0},1);auto muted=begin(c);prove(muted);keep(muted,{1,1});
    c.cullOnly=true;auto cullBadMask=begin(c);prove(cullBadMask,false);
    c=Controls{};c.cullThresholdField=field(std::vector<float>(7,1),4);CHECK(!begin(c));
    c=Controls{};c.randomField=field(std::vector<float>(14,1),4);CHECK(!begin(c));
    c=Controls{};c.enabledField=field(std::vector<uint32_t>(7,1),4);CHECK(!begin(c));
    c=Controls{};c.valueField=field(std::vector<float>{1,2},4);CHECK(!begin(c));
    c=Controls{};c.minimumField=field(std::vector<float>{-1},1);auto badMinimum=begin(c);prove(badMinimum,false);
    // Every retained predecessor channel remains byte-identical after edits.
    Packet finalSource;CHECK(Capture(native,context,base,&finalSource)&&finalSource==original);
    CHECK(read(candidate->keep())==Bytes(std::vector<uint32_t>{0,1}.data(),2));
    std::puts("Vulkan evaluated Length native fields and keep compaction passed");
}
