// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/vulkan/lengthScalePipeline.h"
#include "usdGen/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"
#include "floatUlpFixture.h"
#include "../libs/base/usdGenMath/usdGenMath/hash.h"

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
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"Vulkan LiteralV1 failed: %s (%d)\n",#x,__LINE__);std::abort();} } while(false)
namespace {
template<class T> std::vector<uint8_t> Bytes(T const*p,size_t n){std::vector<uint8_t>r(n*sizeof(T));if(n)std::memcpy(r.data(),p,r.size());return r;}
static std::vector<uint32_t> Code(char const*p){std::ifstream f(p,std::ios::binary);std::vector<char>b((std::istreambuf_iterator<char>(f)),{});if(b.empty()||b.size()%4)return{};std::vector<uint32_t>r(b.size()/4);std::memcpy(r.data(),b.data(),b.size());return r;}
static bool Prove(std::shared_ptr<NativeOwner>const&n){if(vkResetFences(n->device,1,&n->fence)!=VK_SUCCESS)return false;VkSubmitInfo s{};s.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;return vkQueueSubmit(n->queue,1,&s,n->fence)==VK_SUCCESS&&vkWaitForFences(n->device,1,&n->fence,VK_TRUE,10000000000ull)==VK_SUCCESS;}
using Packet=std::map<std::string,std::vector<uint8_t>>;
static bool Capture(std::shared_ptr<NativeOwner>const&n,std::shared_ptr<DeviceContext>const&c,std::shared_ptr<const VulkanSourceGeneration>const&g,Packet*out){out->clear();auto add=[&](VulkanSourceGeneration::PlaneView const&p){std::vector<uint8_t>b;return ReadVulkanBytes(n,c,p.buffer,p.bytes,g,&b)&&out->emplace(p.metadata.name,std::move(b)).second;};for(auto const&p:g->planes())if(!add(p))return false;for(auto const&p:g->sourceFrames())if(!add(p))return false;return true;}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext>c,std::vector<uint64_t> ids={0,1},bool bent=false,bool zero=false){VulkanSourceGenerationCreateInfo i;i.context=std::move(c);auto&s=i.source;s.totalCurves=2;s.totalCvs=bent?7:4;s.topologyVersion=95;s.valueVersion=17;s.cvOffsets=bent?VtIntArray{0,3,7}:VtIntArray{0,2,4};s.curveId=VtArray<uint64_t>(ids.begin(),ids.end());s.rootPrim={12,44};s.rootUV={{.2f,.3f},{.7f,.8f}};s.rootT={{1,0,0},{0,1,0}};s.rootB={{0,1,0},{0,0,1}};s.rootN={{0,0,1},{1,0,0}};UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{.25f,.75f}};if(bent){s.px=zero?VtFloatArray(7,0):VtFloatArray{0,1,1,10,11,12,14};s.py=zero?VtFloatArray(7,0):VtFloatArray{0,0,3,0,0,0,0};s.hairT={0,.75f,1,.5f,0,.75f,.25f};}else{s.px=zero?VtFloatArray(4,0):VtFloatArray{0,1,0,1};s.py=VtFloatArray(4,0);s.hairT={0,1,0,1};}s.pz=VtFloatArray(s.totalCvs,0);s.rest.resize(s.totalCvs);s.width=VtFloatArray(s.totalCvs,1);UsdGenChunkDesc x;x.firstCurve=0;x.curveCount=2;x.liveCount=2;x.firstCv=0;x.cvCount=0;x.tile=3;x.surface=1;s.chunks={x};UsdGenPlane p;p.name=TfToken("privateVertex");p.interpolation=TfToken("vertex");p.type=TfToken("float");p.arity=1;p.f=VtFloatArray(s.totalCvs,3);s.extraCv={p};UsdGenPlane q;q.name=TfToken("privateUniform");q.interpolation=TfToken("uniform");q.type=TfToken("int");q.i={18,19};s.extraCurve={maskPlane, q};i.geometry.alreadyDeformed=true;i.geometry.curveTopology={UsdGenDeviceCurveType::Cubic,UsdGenDeviceCurveBasis::BSpline,UsdGenDeviceCurveWrap::Pinned};i.geometry.tiles={{3,0,2,0,s.totalCvs}};uint32_t named[]={73,81};i.additionalNamed.push_back({{"namedCurve",UsdGenDeviceValueType::UInt32,UsdGenDeviceDomain::Primitive,2,1,4,true},Bytes(named,2)});return i;}
static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner>const&n,VulkanSourceGenerationCreateInfo i){VkResult r=VK_SUCCESS;std::string why;auto u=VulkanSourceUpload::Create(std::move(i),&r,&why);if(!u||u->Submit()!=SourceGenerationStatus::Submitted||!Prove(n)||u->Poll()!=SourceGenerationStatus::Ready)return{};return u->TakeReady();}
}
static void CheckProofs(std::shared_ptr<NativeOwner> const& native,
                        std::shared_ptr<DeviceContext> const& context,
                        DeviceContext::CreateInfo const& contextInfo,
                        std::shared_ptr<LengthScalePipeline> const& pipe) {
    using Controls = LengthScalePipeline::LiteralV1Controls;
    auto pool = context->resources();
    auto sameLedger = [](auto const& a, auto const& b) {
        return a.usedBytes == b.usedBytes && a.byKind == b.byKind &&
               a.limitBytes == b.limitBytes && a.headroomBytes == b.headroomBytes &&
               a.usableBytes == b.usableBytes;
    };
    auto baseline = pool->Snapshot();
    {
        auto base = Upload(native, Source(context, {0x8000000000000001ull, 0xffffffffull}));
        auto foreign = Upload(native, Source(context, {0x8000000000000001ull, 0xffffffffull}));
        CHECK(base && foreign);
        Packet original; CHECK(Capture(native, context, base, &original));
        Controls controls; controls.randomLo = 0; controls.randomHi = 1; controls.seed = -17;
        VkResult status = VK_SUCCESS;
        auto begin = [&](std::shared_ptr<const VulkanSourceGeneration> const& source,
                         std::shared_ptr<const ChargedBuffer> hair,
                         std::shared_ptr<const ChargedBuffer> ids, Controls const& c,
                         LengthScalePipeline::BeforeSubmit hook = {}) {
            return pipe->BeginLiteralV1(source->PlaneOwner("points"), source->PlaneOwner("curveOffsets"),
                std::move(hair), std::move(ids), source->curveCount(), source->pointCount(), c,
                &status, std::move(hook));
        };
        auto prove = [&](auto const& candidate, bool success) {
            CHECK(candidate && status == VK_SUCCESS && Prove(native));
            uint32_t semantic = UINT32_MAX;
            CHECK(candidate->Poll(&semantic) == VK_SUCCESS &&
                  (semantic == 0) == success && candidate->succeeded() == success);
            if (!success) CHECK(!candidate->output());
        };
        // A valid buffer from another predecessor, or omission, must never
        // change which IDs/hairT the output claims to have consumed.
        for (bool testHair : {false, true}) for (bool omit : {false, true}) {
            auto hair = testHair ? (omit ? nullptr : foreign->PlaneOwner("hairT")) : base->PlaneOwner("hairT");
            auto ids = !testHair ? (omit ? nullptr : foreign->PlaneOwner("stableIds")) : base->PlaneOwner("stableIds");
            auto candidate = begin(base, hair, ids, controls);
            prove(candidate, true);
            CHECK(candidate->usesStableIds() && candidate->usesHairT() &&
                  candidate->stableIdsOwner() == ids && candidate->hairTOwner() == hair &&
                  !VulkanSourceGeneration::WithPoints(base, *candidate, 100));
        }
        // Source generations require stable IDs. Test the raw native ordinal
        // fallback, but never publish it over a predecessor with IDs.
        auto fallbackInfo = Source(context);
        fallbackInfo.source.hairT.clear();
        auto fallback = Upload(native, std::move(fallbackInfo));
        CHECK(fallback && fallback->PlaneOwner("stableIds") && !fallback->PlaneOwner("hairT"));
        auto fallbackCandidate = begin(fallback, {}, {}, controls); prove(fallbackCandidate, true);
        CHECK(!VulkanSourceGeneration::WithPoints(fallback, *fallbackCandidate, 101));
        std::vector<uint8_t> fallbackBytes;
        CHECK(ReadVulkanBytes(native, context, fallbackCandidate->output()->buffer(),
                              fallbackCandidate->output()->sizeBytes(), fallback, &fallbackBytes));
        float fallbackPoints[] = {0,0,0,UsdGenDraw01(-17,0,kSaltLength),0,0,
                                  0,0,0,UsdGenDraw01(-17,1,kSaltLength),0,0};
        CHECK(fallbackBytes == Bytes(fallbackPoints, 12));
        auto honest = begin(fallback, {}, fallback->PlaneOwner("stableIds"), controls);
        prove(honest, true);
        auto honestChild = VulkanSourceGeneration::WithPoints(fallback, *honest, 102);
        CHECK(honestChild && !honestChild->PlaneOwner("hairT"));
        auto falsePresence = begin(fallback, base->PlaneOwner("hairT"),
                                   fallback->PlaneOwner("stableIds"), controls);
        prove(falsePresence, true);
        CHECK(!VulkanSourceGeneration::WithPoints(fallback, *falsePresence, 102));

        // Reversed nonconstant endpoints must produce the identical packet.
        controls.randomLo = .25f; controls.randomHi = 1.75f;
        auto ordered = begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), controls);
        prove(ordered, true);
        auto orderedChild = VulkanSourceGeneration::WithPoints(base, *ordered, 103); CHECK(orderedChild);
        Packet orderedPacket; CHECK(Capture(native, context, orderedChild, &orderedPacket));
        std::swap(controls.randomLo, controls.randomHi);
        auto reversed = begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), controls);
        prove(reversed, true);
        auto reversedChild = VulkanSourceGeneration::WithPoints(base, *reversed, 104); CHECK(reversedChild);
        Packet reversedPacket; CHECK(Capture(native, context, reversedChild, &reversedPacket));
        CHECK(orderedPacket == reversedPacket);
        CHECK(orderedChild->PlaneOwner("points") != base->PlaneOwner("points"));
        CHECK(orderedChild->topologyVersion() == base->topologyVersion() &&
              orderedChild->curveCount() == base->curveCount() &&
              orderedChild->pointCount() == base->pointCount() &&
              orderedChild->geometry().alreadyDeformed == base->geometry().alreadyDeformed &&
              orderedChild->geometry().curveTopology.type == base->geometry().curveTopology.type &&
              orderedChild->geometry().curveTopology.basis == base->geometry().curveTopology.basis &&
              orderedChild->geometry().curveTopology.wrap == base->geometry().curveTopology.wrap);
        CHECK(orderedChild->geometry().tiles.size() == base->geometry().tiles.size() &&
              orderedChild->chunks().size() == base->chunks().size());
        for (size_t i = 0; i < base->geometry().tiles.size(); ++i) {
            auto const& a = orderedChild->geometry().tiles[i]; auto const& b = base->geometry().tiles[i];
            CHECK(a.tile == b.tile && a.firstCurve == b.firstCurve && a.curveCount == b.curveCount &&
                  a.firstPoint == b.firstPoint && a.pointCount == b.pointCount &&
                  a.extentMin == b.extentMin && a.extentMax == b.extentMax && a.boundsValid == b.boundsValid);
        }
        for (size_t i = 0; i < base->chunks().size(); ++i) {
            auto const& a = orderedChild->chunks()[i]; auto const& b = base->chunks()[i];
            CHECK(a.firstCurve == b.firstCurve && a.curveCount == b.curveCount &&
                  a.liveCount == b.liveCount && a.firstCv == b.firstCv && a.cvCount == b.cvCount &&
                  a.tile == b.tile && a.surface == b.surface);
        }
        for (auto const& frame : base->sourceFrames())
            CHECK(orderedChild->SourceFrameOwner(frame.metadata.name) == base->SourceFrameOwner(frame.metadata.name) &&
                  orderedPacket.at(frame.metadata.name) == original.at(frame.metadata.name));
        for (auto const& plane : base->planes()) if (plane.metadata.name != "points")
            CHECK(orderedChild->PlaneOwner(plane.metadata.name) == base->PlaneOwner(plane.metadata.name) &&
                  orderedPacket.at(plane.metadata.name) == original.at(plane.metadata.name));

        auto live = pool->Snapshot();
        for (float invalid : {-1.f, std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()}) for (bool high : {false,true}) {
            auto bad = controls; (high ? bad.randomHi : bad.randomLo) = invalid;
            unsigned hooks = 0;
            CHECK(!begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), bad,
                         [&] { ++hooks; return true; }) && hooks == 0);
            CHECK(!pipe->BeginLiteralV1({}, {}, {}, {}, 0, 0, bad, &status));
            CHECK(sameLedger(pool->Snapshot(), live));
        }
        for (int invalidKind : {0,1,2}) {
            auto bufferContext = invalidKind == 2 ? DeviceContext::Create(contextInfo) : context;
            CHECK(bufferContext);
            VkBufferCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = invalidKind == 0 ? 8 : 16;
            info.usage = invalidKind == 1 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto ids = ChargedBuffer::Create(bufferContext, info, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                             UsdGenExecutionResourceKind::Scratch);
            CHECK(ids);
            auto before = pool->Snapshot(); unsigned hooks = 0;
            CHECK(!begin(base, base->PlaneOwner("hairT"), ids, controls,
                         [&] { ++hooks; return true; }) && hooks == 0 &&
                  sameLedger(pool->Snapshot(), before));
        }
        CHECK(sameLedger(pool->Snapshot(), live));
        for (float invalid : {-.1f, 1.1f, std::numeric_limits<float>::quiet_NaN(),
                               std::numeric_limits<float>::infinity()}) {
            VkBufferCreateInfo info{}; info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            info.size = base->pointCount()*sizeof(float);
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto hair = ChargedBuffer::Create(context, info,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            CHECK(hair);
            void* mapped = nullptr;
            CHECK(vkMapMemory(native->device, hair->memory(), 0, info.size, 0, &mapped) == VK_SUCCESS);
            for (uint32_t i = 0; i < base->pointCount(); ++i) static_cast<float*>(mapped)[i] = invalid;
            vkUnmapMemory(native->device, hair->memory());
            auto badHair = begin(base, hair, base->PlaneOwner("stableIds"), controls);
            prove(badHair, false);
            CHECK(!VulkanSourceGeneration::WithPoints(base, *badHair, 107));
        }
        CHECK(sameLedger(pool->Snapshot(), live));
        auto pressure = pool->TryReserve(live.usableBytes-live.usedBytes, UsdGenExecutionResourceKind::Scratch);
        CHECK(pressure);
        auto saturated = pool->Snapshot(); unsigned hooks = 0;
        CHECK(!begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), controls,
                     [&] { ++hooks; return true; }) && hooks == 0 &&
              status == VK_ERROR_OUT_OF_DEVICE_MEMORY && sameLedger(pool->Snapshot(), saturated));
        pressure->Release();
        CHECK(!begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), controls,
                     [&] { ++hooks; return false; }) && hooks == 1 && sameLedger(pool->Snapshot(), live));
        CHECK(!begin(base, base->PlaneOwner("hairT"), base->PlaneOwner("stableIds"), controls,
                     [&]() -> bool { ++hooks; throw 1; }) && hooks == 2 &&
              status == VK_ERROR_UNKNOWN && sameLedger(pool->Snapshot(), live));

        // A finite arc times FLT_MAX overflows, then *0 is NaN. Match CUDA
        // fmaxf recovery of the finite minimum rather than rejecting raw NaN.
        auto bent = Upload(native, Source(context, {401,907}, true)); CHECK(bent);
        controls = Controls{}; controls.value = std::numeric_limits<float>::max();
        controls.minimum = 3; controls.randomLo = controls.randomHi = 0;
        auto recovery = begin(bent, bent->PlaneOwner("hairT"), bent->PlaneOwner("stableIds"), controls);
        prove(recovery, true);
        auto recovered = VulkanSourceGeneration::WithPoints(bent, *recovery, 105); CHECK(recovered);
        Packet recoveredPacket; CHECK(Capture(native, context, recovered, &recoveredPacket));
        float recoveredPoints[] = {0,0,0,.75f,0,0,.75f,2.25f,0,10,0,0,10.75f,0,0,11.5f,0,0,13,0,0};
        CHECK(recoveredPacket.at("points").size() == sizeof(recoveredPoints) &&
              FloatBytesWithinUlps(recoveredPacket.at("points").data(), recoveredPoints, sizeof(recoveredPoints), 4));
        controls.randomLo = controls.randomHi = .5f;
        auto overflow = begin(bent, bent->PlaneOwner("hairT"), bent->PlaneOwner("stableIds"), controls);
        prove(overflow, false);
        CHECK(!VulkanSourceGeneration::WithPoints(bent, *overflow, 106));
        Packet sourceAgain; CHECK(Capture(native, context, base, &sourceAgain) && sourceAgain == original);
    }
    CHECK(sameLedger(pool->Snapshot(), baseline));
}

int main(int argc,char**argv){CHECK(argc==7);auto scale=Code(argv[1]),set=Code(argv[2]),cut=Code(argv[3]),reparam=Code(argv[4]),minimum=Code(argv[5]),literal=Code(argv[6]);CHECK(!scale.empty()&&!set.empty()&&!cut.empty()&&!reparam.empty()&&!minimum.empty()&&!literal.empty());bool unavailable=false;auto native=CreateNative(&unavailable);if(unavailable)return 77;CHECK(native);VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(native->physical,&p);if(p.limits.maxPerStageDescriptorStorageBuffers<6||p.limits.maxDescriptorSetStorageBuffers<6||p.limits.maxPushConstantsSize<48)return 77;DeviceContext::CreateInfo ci;ci.instance=native->instance;ci.physicalDevice=native->physical;ci.device=native->device;ci.computeQueue=native->queue;ci.computeQueueFamily=native->family;ci.physicalIndex=native->physicalIndex;ci.resourceDeviceId=7935;ci.nativeLifetime=native;ci.resources={size_t{32}<<20,0};auto context=DeviceContext::Create(ci);CHECK(context);auto pool=context->resources();auto ledger=pool->Snapshot();VkResult status=VK_SUCCESS;auto old=LengthScalePipeline::CreateWithMinimum(context,scale,set,cut,reparam,minimum,&status);CHECK(old&&!old->HasLiteralV1());auto pipe=LengthScalePipeline::CreateWithLiteralV1(context,scale,set,cut,reparam,minimum,literal,&status);CHECK(pipe&&pipe->HasLiteralV1());
 CheckProofs(native, context, ci, pipe);
 auto run=[&](std::shared_ptr<const VulkanSourceGeneration>const&g,LengthScalePipeline::LiteralV1Controls c){auto x=pipe->BeginLiteralV1(g->PlaneOwner("points"),g->PlaneOwner("curveOffsets"),g->PlaneOwner("hairT"),g->PlaneOwner("stableIds"),g->curveCount(),g->pointCount(),c,&status);if(!x||status!=VK_SUCCESS||!Prove(native))return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);uint32_t sem=UINT32_MAX;if(x->Poll(&sem)!=VK_SUCCESS)return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{},UINT32_MAX);return std::pair<std::unique_ptr<LengthScalePipeline::Candidate>,uint32_t>(std::move(x),sem);};
 auto base=Upload(native,Source(context));CHECK(base);LengthScalePipeline::LiteralV1Controls c;unsigned hooks=0;CHECK(!old->BeginLiteralV1(base->PlaneOwner("points"),base->PlaneOwner("curveOffsets"),base->PlaneOwner("hairT"),base->PlaneOwner("stableIds"),2,4,c,&status,[&]{++hooks;return true;})&&status==VK_ERROR_FEATURE_NOT_PRESENT&&hooks==0);Packet original;CHECK(Capture(native,context,base,&original));
 std::vector<uint64_t> ids={0,1,0xffffffffull,0x100000000ull,0x8000000000000000ull,0xffffffffffffffffull,0x1234567800000001ull};for(int seed:{0,-1,std::numeric_limits<int>::min(),std::numeric_limits<int>::max(),11})for(size_t k=0;k<ids.size();++k){auto firstId=ids[k],secondId=ids[(k+1)%ids.size()];auto g=Upload(native,Source(context,{secondId,firstId}));CHECK(g);c.value=1;c.minimum=0;c.absolute=false;c.method=LengthScalePipeline::LiteralV1Controls::Method::Scale;c.rebuild=LengthScalePipeline::LiteralV1Controls::Rebuild::KeepParam;c.randomLo=0;c.randomHi=1;c.seed=seed;auto r=run(g,c);CHECK(r.first&&r.second==0&&r.first->succeeded());std::string why;auto child=VulkanSourceGeneration::WithPoints(g,*r.first,18,&why);Packet got;CHECK(Capture(native,context,child,&got));float second=UsdGenDraw01(seed,secondId,kSaltLength),first=UsdGenDraw01(seed,firstId,kSaltLength);float expected[]={0,0,0,second,0,0,0,0,0,first,0,0};CHECK(got.at("points")==Bytes(expected,12));r.first.reset();child.reset();g.reset();}
 c.seed=11;c.randomLo=.75f;c.randomHi=.75f;c.value=1;c.minimum=0;c.absolute=false;auto bent=Upload(native,Source(context,{401,907},true));CHECK(bent);Packet bentOriginal;CHECK(Capture(native,context,bent,&bentOriginal));float radial[]={0,0,0,.75f,0,0,.75f,2.25f,0,10,0,0,10.75f,0,0,11.5f,0,0,13,0,0};float keep[]={0,0,0,1,0,0,1,2,0,10,0,0,11,0,0,12,0,0,13,0,0};float rep[]={0,0,0,1,1.25f,0,1,2,0,11.5f,0,0,10,0,0,12.25f,0,0,10.75f,0,0};Packet first;std::shared_ptr<const VulkanSourceGeneration> firstChild;for(auto pair:{std::pair<LengthScalePipeline::LiteralV1Controls::Method,LengthScalePipeline::LiteralV1Controls::Rebuild>{LengthScalePipeline::LiteralV1Controls::Method::Scale,LengthScalePipeline::LiteralV1Controls::Rebuild::KeepParam},{LengthScalePipeline::LiteralV1Controls::Method::CutExtend,LengthScalePipeline::LiteralV1Controls::Rebuild::KeepParam},{LengthScalePipeline::LiteralV1Controls::Method::CutExtend,LengthScalePipeline::LiteralV1Controls::Rebuild::Reparam}}){c.method=pair.first;c.rebuild=pair.second;auto r=run(bent,c);CHECK(r.first&&r.second==0);std::string why;auto child=VulkanSourceGeneration::WithPoints(bent,*r.first,20+first.size(),&why);Packet got;CHECK(Capture(native,context,child,&got));float const*e=pair.first==LengthScalePipeline::LiteralV1Controls::Method::Scale?radial:pair.second==LengthScalePipeline::LiteralV1Controls::Rebuild::KeepParam?keep:rep;CHECK(got.at("points").size()==21*sizeof(float));CHECK(FloatBytesWithinUlps(got.at("points").data(),e,21*sizeof(float),4));for(auto const&p:bentOriginal)if(p.first!="points")CHECK(got.at(p.first)==p.second);if(first.empty()){first=got;firstChild=child;}r.first.reset();child.reset();}Packet reread,firstAgain;CHECK(Capture(native,context,bent,&reread)&&reread==bentOriginal&&Capture(native,context,firstChild,&firstAgain)&&firstAgain==first);
 auto zero=Upload(native,Source(context,{0,1},false,true));CHECK(zero);c.randomLo=1;c.randomHi=1;c.value=1;c.minimum=1;auto bad=run(zero,c);CHECK(bad.first&&bad.second!=0&&!bad.first->output());auto empty=pipe->BeginLiteralV1({}, {}, {}, {},0,0,c,&status);CHECK(empty&&empty->succeeded());empty.reset();bad.first.reset();zero.reset();firstChild.reset();bent.reset();old.reset();pipe.reset();base.reset();CHECK(pool->Snapshot().usedBytes==ledger.usedBytes);std::puts("Vulkan LiteralV1 Length: PASS");return 0;}
