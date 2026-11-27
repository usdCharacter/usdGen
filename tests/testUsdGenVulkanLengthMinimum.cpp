// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "utils/testToken.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"
#include "floatUlpFixture.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/types.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan minimum-length failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
template<class T> std::vector<uint8_t> Bytes(T const* value, size_t count) {
    std::vector<uint8_t> result(count * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), value, result.size());
    return result;
}
static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> result(raw.size() / sizeof(uint32_t));
    std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}
using Packet = std::map<std::string, std::vector<uint8_t>>;
static bool Capture(std::shared_ptr<NativeOwner> const& native,
                    std::shared_ptr<DeviceContext> const& context,
                    std::shared_ptr<const VulkanSourceGeneration> const& generation,
                    Packet* packet) {
    packet->clear();
    auto add = [&](VulkanSourceGeneration::PlaneView const& view) {
        std::vector<uint8_t> bytes;
        return ReadVulkanBytes(native, context, view.buffer, view.bytes, generation, &bytes) &&
            packet->emplace(view.metadata.name.GetString(), std::move(bytes)).second;
    };
    for (auto const& view : generation->planes()) if (!add(view)) return false;
    for (auto const& view : generation->sourceFrames()) if (!add(view)) return false;
    return true;
}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context,
                                                bool zero = false) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context); auto& source = info.source;
    source.totalCurves = 2; source.totalCvs = 7; source.topologyVersion = 94; source.valueVersion = 17;
    source.cvOffsets = {0, 3, 7}; source.curveId = {401, 907}; source.rootPrim = {12, 44};
    source.rootUV = {{.2f,.3f},{.7f,.8f}}; source.rootT = {{1,0,0},{0,1,0}};
    source.rootB = {{0,1,0},{0,0,1}}; source.rootN = {{0,0,1},{1,0,0}}; UsdGenPlane maskPlane; maskPlane.name = TestToken("curveMask"); maskPlane.interpolation = TestToken("uniform"); maskPlane.type = TestToken("float"); maskPlane.f = {{.25f,.75f}};
    source.px = zero ? usdGen::UsdGenFloatArray(7, 0.f) : usdGen::UsdGenFloatArray{0,1,1,10,11,12,14};
    source.py = zero ? usdGen::UsdGenFloatArray(7, 0.f) : usdGen::UsdGenFloatArray{0,0,3,0,0,0,0}; source.pz = usdGen::UsdGenFloatArray(7, 0.f);
    source.rest = {usdGen::UsdGenVec3f(0,0,0),usdGen::UsdGenVec3f(0,10,0),usdGen::UsdGenVec3f(0,30,0),usdGen::UsdGenVec3f(9,9,9),usdGen::UsdGenVec3f(19,9,9),usdGen::UsdGenVec3f(29,9,9),usdGen::UsdGenVec3f(49,9,9)};
    source.width = usdGen::UsdGenFloatArray(7, 1.f); source.hairT = {0,.75f,1,.5f,0,.75f,.25f};
    UsdGenChunkDesc chunk; chunk.firstCurve=0; chunk.curveCount=2; chunk.liveCount=2; chunk.firstCv=0; chunk.cvCount=0; chunk.tile=3; chunk.surface=1; source.chunks={chunk};
    UsdGenPlane vertex; vertex.name=TestToken("privateVertex"); vertex.interpolation=TestToken("vertex"); vertex.type=TestToken("float"); vertex.arity=1; vertex.f={3,4,5,6,7,8,9}; source.extraCv={vertex};
    UsdGenPlane uniform; uniform.name=TestToken("privateUniform"); uniform.interpolation=TestToken("uniform"); uniform.type=TestToken("int"); uniform.i={18,19}; source.extraCurve={maskPlane, uniform};
    info.geometry.alreadyDeformed=true; info.geometry.layoutMetadata=UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic,UsdGenCurveLayoutBasis::BSpline,UsdGenCurveLayoutWrap::Pinned); info.geometry.tiles={{3,0,2,0,7}};
    uint32_t named[] = {73,81}; info.additionalNamed.push_back({{TestToken("namedCurve"),UsdGenDeviceValueType::UInt32,UsdGenDeviceDomain::Primitive,2,1,4,true},Bytes(named,2)});
    return info;
}
static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner> const& native,
                                                              VulkanSourceGenerationCreateInfo info) {
    VkResult status = VK_SUCCESS; std::string why; auto upload = VulkanSourceUpload::Create(std::move(info), &status, &why);
    if (!upload || upload->Submit()!=SourceGenerationStatus::Submitted || !Prove(native) || upload->Poll()!=SourceGenerationStatus::Ready) return {};
    return upload->TakeReady();
}
}

int main(int argc, char** argv) {
    CHECK(argc == 6);
    auto scale=Code(argv[1]), set=Code(argv[2]), cut=Code(argv[3]), reparam=Code(argv[4]), minimum=Code(argv[5]);
    CHECK(!scale.empty() && !set.empty() && !cut.empty() && !reparam.empty() && !minimum.empty());
    bool unavailable=false; auto native=CreateTimelineNative(&unavailable); if (unavailable) return 77; CHECK(native);
    VkPhysicalDeviceProperties limits{}; vkGetPhysicalDeviceProperties(native->physical, &limits);
    if (limits.limits.maxPerStageDescriptorStorageBuffers < 5 || limits.limits.maxDescriptorSetStorageBuffers < 5 || limits.limits.maxPushConstantsSize < 32) return 77;
    DeviceContext::CreateInfo ci; ci.instance=native->instance; ci.physicalDevice=native->physical; ci.device=native->device; ci.computeQueue=native->queue; ci.computeQueueFamily=native->family; ci.physicalIndex=native->physicalIndex; ci.resourceDeviceId=7934; ci.nativeLifetime=native; ci.timelineSemaphoreEnabled=true; ci.resources={size_t{32}<<20,0};
    auto context=DeviceContext::Create(ci); CHECK(context); auto pool=context->resources(); auto ledger=pool->Snapshot(); VkResult status=VK_SUCCESS;
    auto base=Upload(native,Source(context)); CHECK(base);
    auto old=LengthScalePipeline::CreateWithReparam(context,scale,set,cut,reparam,&status); CHECK(old && !old->HasMinimum());
    auto pipeline=LengthScalePipeline::CreateWithMinimum(context,scale,set,cut,reparam,minimum,&status); CHECK(pipeline && pipeline->HasMinimum());
    unsigned missingHooks=0;
    LengthScalePipeline::MinimumControls missingControls;
    CHECK(!old->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,missingControls,&status,[&]{++missingHooks;return true;}) && status==VK_ERROR_FEATURE_NOT_PRESENT && missingHooks==0);
    auto run=[&](std::shared_ptr<const VulkanSourceGeneration> const& generation, LengthScalePipeline::MinimumControls controls) {
        auto candidate=pipeline->BeginMinimum(generation->PlaneOwner(TestToken("points")),generation->PlaneOwner(TestToken("curveOffsets")),generation->PlaneOwner(TestToken("hairT")),generation->curveCount(),generation->pointCount(),controls,&status);
        if (!candidate || status!=VK_SUCCESS || !Prove(native)) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);
        uint32_t semantic=UINT32_MAX; if (candidate->Poll(&semantic)!=VK_SUCCESS) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);
        return std::pair<std::unique_ptr<LengthScalePipeline::Candidate>,uint32_t>(std::move(candidate),semantic);
    };
    auto runRaw=[&](std::shared_ptr<const ChargedBuffer> hair, LengthScalePipeline::MinimumControls controls) {
        auto candidate=pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),std::move(hair),base->curveCount(),base->pointCount(),controls,&status);
        if (!candidate || status!=VK_SUCCESS || !Prove(native)) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);
        uint32_t semantic=UINT32_MAX; if (candidate->Poll(&semantic)!=VK_SUCCESS) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);
        return std::pair<std::unique_ptr<LengthScalePipeline::Candidate>,uint32_t>(std::move(candidate),semantic);
    };
    auto sameLedger=[](UsdGenExecutionResourceSnapshot const& a, UsdGenExecutionResourceSnapshot const& b) {
        return a.limitBytes==b.limitBytes && a.headroomBytes==b.headroomBytes && a.usableBytes==b.usableBytes && a.usedBytes==b.usedBytes && a.byKind==b.byKind;
    };
    Packet original; CHECK(Capture(native,context,base,&original)); CHECK(original.count("stableIds") && original.count("privateVertex") && original.count("sourceRootT"));
    auto verify=[&](std::shared_ptr<const VulkanSourceGeneration> const& child,float const* expected) {
        CHECK(child && child->PlaneOwner(TestToken("points"))!=base->PlaneOwner(TestToken("points")));
        for(auto const& plane:base->planes()) if(plane.metadata.name!="points") CHECK(child->PlaneOwner(plane.metadata.name)==base->PlaneOwner(plane.metadata.name));
        for(auto const& frame:base->sourceFrames()) CHECK(child->SourceFrameOwner(frame.metadata.name)==base->SourceFrameOwner(frame.metadata.name));
        CHECK(child->topologyVersion()==base->topologyVersion() &&
              child->geometry().alreadyDeformed==base->geometry().alreadyDeformed &&
              UsdGenCurveLayoutTypeOf(child->geometry().layoutMetadata)==UsdGenCurveLayoutTypeOf(base->geometry().layoutMetadata) &&
              UsdGenCurveLayoutBasisOf(child->geometry().layoutMetadata)==UsdGenCurveLayoutBasisOf(base->geometry().layoutMetadata) &&
              UsdGenCurveLayoutWrapOf(child->geometry().layoutMetadata)==UsdGenCurveLayoutWrapOf(base->geometry().layoutMetadata) &&
              child->geometry().tiles.size()==base->geometry().tiles.size() &&
              child->chunks().size()==base->chunks().size());
        for(size_t i=0;i<child->geometry().tiles.size();++i){auto const&a=child->geometry().tiles[i];auto const&b=base->geometry().tiles[i];CHECK(a.tile==b.tile&&a.firstElement==b.firstElement&&a.elementCount==b.elementCount&&a.firstPoint==b.firstPoint&&a.pointCount==b.pointCount&&a.extentMin==b.extentMin&&a.extentMax==b.extentMax&&a.boundsValid==b.boundsValid);}
        auto const&a=child->chunks()[0];auto const&b=base->chunks()[0];CHECK(a.firstCurve==b.firstCurve&&a.curveCount==b.curveCount&&a.liveCount==b.liveCount&&a.firstCv==b.firstCv&&a.cvCount==b.cvCount&&a.tile==b.tile&&a.surface==b.surface);
        Packet packet; CHECK(Capture(native,context,child,&packet) && packet.size()==original.size()); for(auto const& p:original) if(p.first!="points") CHECK(packet.at(p.first)==p.second); CHECK(packet.at("points").size()==21*sizeof(float)); CHECK(FloatBytesWithinUlps(packet.at("points").data(),expected,21*sizeof(float),4)); return packet;
    };
    auto controls=[](float value,bool absolute,float floor,LengthScalePipeline::MinimumControls::Method method,LengthScalePipeline::MinimumControls::Rebuild rebuild){LengthScalePipeline::MinimumControls c;c.value=value;c.absolute=absolute;c.minimum=floor;c.method=method;c.rebuild=rebuild;return c;};
    float radial[]={0,0,0,.75f,0,0,.75f,2.25f,0,10,0,0,10.75f,0,0,11.5f,0,0,13,0,0};
    float keep[]={0,0,0,1,0,0,1,2,0,10,0,0,11,0,0,12,0,0,13,0,0};
    float rep[]={0,0,0,1,1.25f,0,1,2,0,11.5f,0,0,10,0,0,12.25f,0,0,10.75f,0,0};
    std::vector<std::shared_ptr<const VulkanSourceGeneration>> children;
    Packet firstInitial;
    for (auto pair : {std::pair<float,bool>{1.f,true}, { .25f,false}}) {
        for (auto item : {std::pair<LengthScalePipeline::MinimumControls::Method,LengthScalePipeline::MinimumControls::Rebuild>{LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam}, {LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::Reparam}, {LengthScalePipeline::MinimumControls::Method::CutExtend,LengthScalePipeline::MinimumControls::Rebuild::KeepParam}, {LengthScalePipeline::MinimumControls::Method::CutExtend,LengthScalePipeline::MinimumControls::Rebuild::Reparam}}) {
            auto result=run(base,controls(pair.first,pair.second,3,item.first,item.second)); CHECK(result.first&&result.second==0&&result.first->succeeded()&&result.first->usesHairT()&&result.first->usesReparam()==(item.first==LengthScalePipeline::MinimumControls::Method::CutExtend&&item.second==LengthScalePipeline::MinimumControls::Rebuild::Reparam));
            std::string why; auto child=VulkanSourceGeneration::WithPoints(base,*result.first,18+children.size(),&why); children.push_back(child); verify(child,item.first==LengthScalePipeline::MinimumControls::Method::Scale?radial:item.second==LengthScalePipeline::MinimumControls::Rebuild::KeepParam?keep:rep); result.first.reset();
            if (children.size()==1) CHECK(Capture(native,context,children.front(),&firstInitial));
        }
    }
    auto binding=run(base,controls(1,true,6,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam)); CHECK(binding.first&&binding.second==0); std::string why;auto bindingChild=VulkanSourceGeneration::WithPoints(base,*binding.first,30,&why);float six[]={0,0,0,1.5f,0,0,1.5f,4.5f,0,10,0,0,11.5f,0,0,13,0,0,16,0,0};verify(bindingChild,six);
    auto nonbinding=run(base,controls(6,true,1,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam)); CHECK(nonbinding.first&&nonbinding.second==0);auto nonbindingChild=VulkanSourceGeneration::WithPoints(base,*nonbinding.first,31,&why);verify(nonbindingChild,six);
    auto same=run(base,controls(0,true,4,LengthScalePipeline::MinimumControls::Method::CutExtend,LengthScalePipeline::MinimumControls::Rebuild::Reparam)); CHECK(same.first&&same.second==0&&same.first->usesReparam());auto sameChild=VulkanSourceGeneration::WithPoints(base,*same.first,32,&why);float sameExpected[]={0,0,0,1,2,0,1,3,0,12,0,0,10,0,0,13,0,0,11,0,0};verify(sameChild,sameExpected);
    Packet sourceAgain, firstAgain; CHECK(Capture(native,context,base,&sourceAgain)&&sourceAgain==original&&Capture(native,context,children.front(),&firstAgain)&&firstAgain==firstInitial);
    auto radialControls=controls(1,true,3,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam);
    auto omitted=runRaw({},radialControls); CHECK(omitted.first&&omitted.second==0&&omitted.first->succeeded()&&omitted.first->usesHairT()&&!omitted.first->usesReparam()&&!VulkanSourceGeneration::WithPoints(base,*omitted.first,34));
    auto foreign=Upload(native,Source(context)); CHECK(foreign&&foreign->PlaneOwner(TestToken("hairT"))!=base->PlaneOwner(TestToken("hairT")));
    auto foreignCandidate=runRaw(foreign->PlaneOwner(TestToken("hairT")),radialControls); CHECK(foreignCandidate.first&&foreignCandidate.second==0&&foreignCandidate.first->succeeded()&&foreignCandidate.first->usesHairT()&&!foreignCandidate.first->usesReparam()&&!VulkanSourceGeneration::WithPoints(base,*foreignCandidate.first,35));
    auto badHair=[&](float value, LengthScalePipeline::MinimumControls controlsForCase) {
        VkBufferCreateInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size=base->pointCount()*sizeof(float); bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        auto hair=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,UsdGenExecutionResourceKind::Scratch); CHECK(hair);
        void* mapped=nullptr; CHECK(vkMapMemory(native->device,hair->memory(),0,bi.size,0,&mapped)==VK_SUCCESS); for(uint32_t i=0;i<base->pointCount();++i) static_cast<float*>(mapped)[i]=value; vkUnmapMemory(native->device,hair->memory());
        auto candidate=runRaw(hair,controlsForCase); CHECK(candidate.first&&candidate.second!=0&&!candidate.first->succeeded()&&!candidate.first->output()&&!VulkanSourceGeneration::WithPoints(base,*candidate.first,36));
    };
    badHair(-.1f,radialControls); badHair(1.1f,radialControls); badHair(std::numeric_limits<float>::quiet_NaN(),radialControls); badHair(std::numeric_limits<float>::infinity(),radialControls);
    badHair(-.1f,controls(1,true,3,LengthScalePipeline::MinimumControls::Method::CutExtend,LengthScalePipeline::MinimumControls::Rebuild::Reparam));
    // A genuinely absent predecessor hairT permits the ordinal fallback;
    // omission above must not silently override an existing predecessor plane.
    {
        auto fallbackInfo = Source(context); fallbackInfo.source.hairT.clear();
        auto fallback = Upload(native, std::move(fallbackInfo)); CHECK(fallback && !fallback->PlaneOwner(TestToken("hairT")));
        Packet beforeFallback; CHECK(Capture(native, context, fallback, &beforeFallback));
        auto fallbackResult = run(fallback, controls(1,true,3,
            LengthScalePipeline::MinimumControls::Method::CutExtend,
            LengthScalePipeline::MinimumControls::Rebuild::Reparam));
        CHECK(fallbackResult.first && fallbackResult.second == 0 && fallbackResult.first->usesHairT() &&
              fallbackResult.first->usesReparam() && !fallbackResult.first->hairTOwner());
        auto fallbackChild = VulkanSourceGeneration::WithPoints(fallback, *fallbackResult.first, 37);
        CHECK(fallbackChild && !fallbackChild->PlaneOwner(TestToken("hairT")));
        Packet fallbackPacket, fallbackAgain;
        CHECK(Capture(native, context, fallbackChild, &fallbackPacket) &&
              fallbackPacket.size() == beforeFallback.size());
        float fallbackPoints[] = {0,0,0, 1,.5f,0, 1,2,0, 10,0,0, 11,0,0, 12,0,0, 13,0,0};
        CHECK(fallbackPacket.at("points").size() == sizeof(fallbackPoints) &&
              FloatBytesWithinUlps(fallbackPacket.at("points").data(), fallbackPoints, sizeof(fallbackPoints), 4));
        for (auto const& plane : beforeFallback)
            if (plane.first != "points") CHECK(fallbackPacket.at(plane.first) == plane.second);
        CHECK(Capture(native, context, fallback, &fallbackAgain) && fallbackAgain == beforeFallback);
    }
    auto liveBaseline=pool->Snapshot();
    auto invalidHair=[&](std::shared_ptr<const ChargedBuffer> hair, UsdGenExecutionResourceSnapshot const& before) {
        unsigned hooks=0; auto candidate=pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),std::move(hair),base->curveCount(),base->pointCount(),radialControls,&status,[&]{++hooks;return true;});
        CHECK(!candidate&&status==VK_ERROR_INITIALIZATION_FAILED&&hooks==0&&sameLedger(pool->Snapshot(),before));
    };
    { VkBufferCreateInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size=base->pointCount()*sizeof(float)-sizeof(float); bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE; auto hair=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,UsdGenExecutionResourceKind::Scratch); CHECK(hair); auto before=pool->Snapshot(); invalidHair(hair,before); }
    { VkBufferCreateInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size=base->pointCount()*sizeof(float); bi.usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE; auto hair=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,UsdGenExecutionResourceKind::Scratch); CHECK(hair); auto before=pool->Snapshot(); invalidHair(hair,before); }
    { auto foreignContext=DeviceContext::Create(ci); CHECK(foreignContext); VkBufferCreateInfo bi{}; bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size=base->pointCount()*sizeof(float); bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE; auto hair=ChargedBuffer::Create(foreignContext,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,UsdGenExecutionResourceKind::Scratch); CHECK(hair); auto before=pool->Snapshot(); invalidHair(hair,before); }
    CHECK(sameLedger(pool->Snapshot(),liveBaseline));
    auto malformed=radialControls; malformed.method=static_cast<LengthScalePipeline::MinimumControls::Method>(99); unsigned malformedHooks=0; CHECK(!pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,malformed,&status,[&]{++malformedHooks;return true;})&&malformedHooks==0&&sameLedger(pool->Snapshot(),liveBaseline));
    auto pressure=pool->TryReserve(liveBaseline.usableBytes-liveBaseline.usedBytes,UsdGenExecutionResourceKind::Scratch); CHECK(pressure); auto saturated=pool->Snapshot(); unsigned pressureHooks=0; CHECK(!pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,radialControls,&status,[&]{++pressureHooks;return true;})&&status==VK_ERROR_OUT_OF_DEVICE_MEMORY&&pressureHooks==0&&sameLedger(pool->Snapshot(),saturated)); pressure->Release(); CHECK(sameLedger(pool->Snapshot(),liveBaseline));
    unsigned admissionHooks=0; auto hookFalse=pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,radialControls,&status,[&]{++admissionHooks;return false;}); CHECK(!hookFalse&&status==VK_ERROR_OUT_OF_DEVICE_MEMORY&&admissionHooks==1&&sameLedger(pool->Snapshot(),liveBaseline)); auto hookThrow=pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,radialControls,&status,[&]() -> bool {++admissionHooks;throw 1;}); CHECK(!hookThrow&&status==VK_ERROR_UNKNOWN&&admissionHooks==2&&sameLedger(pool->Snapshot(),liveBaseline));
    Packet sourceFinal, firstFinal; CHECK(Capture(native,context,base,&sourceFinal)&&sourceFinal==original&&Capture(native,context,children.front(),&firstFinal)&&firstFinal==firstInitial);
    auto zero=Upload(native,Source(context,true));CHECK(zero);auto failed=run(zero,controls(0,true,3,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam));CHECK(failed.first&&failed.second!=0&&!failed.first->succeeded()&&!failed.first->output()&&!VulkanSourceGeneration::WithPoints(zero,*failed.first,33));auto empty=pipeline->BeginMinimum({}, {}, {},0,0,controls(1,true,3,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam),&status);CHECK(empty&&empty->succeeded());
    for(float invalid:{-1.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){auto c=controls(1,true,invalid,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam);unsigned hooks=0;CHECK(!pipeline->BeginMinimum(base->PlaneOwner(TestToken("points")),base->PlaneOwner(TestToken("curveOffsets")),base->PlaneOwner(TestToken("hairT")),2,7,c,&status,[&]{++hooks;return true;})&&hooks==0);auto e=controls(1,true,invalid,LengthScalePipeline::MinimumControls::Method::Scale,LengthScalePipeline::MinimumControls::Rebuild::KeepParam);CHECK(!pipeline->BeginMinimum({}, {}, {},0,0,e,&status));}
    empty.reset();failed.first.reset();zero.reset();foreignCandidate.first.reset();foreign.reset();omitted.first.reset();sameChild.reset();same.first.reset();nonbindingChild.reset();nonbinding.first.reset();bindingChild.reset();binding.first.reset();children.clear();old.reset();pipeline.reset();base.reset();CHECK(sameLedger(pool->Snapshot(),ledger));std::puts("Vulkan minimum Length COW: PASS");return 0;
}
