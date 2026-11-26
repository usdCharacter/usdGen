// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/types/curveDeviceLayout.h"
#include "vulkanReadbackFixture.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenCurves/vulkan/widthBlendPipeline.h"
#include <map>
#include <fstream>
#include <iterator>
#include <limits>
#include <cmath>
#include <string>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Source regression failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

template<class T> static std::vector<uint8_t> Bytes(T const* data, size_t count) {
    std::vector<uint8_t> result(count * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), data, result.size());
    return result;
}
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}
using Snapshot = std::map<std::string, std::vector<uint8_t>>;
static bool Capture(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const VulkanSourceGeneration> const& generation, Snapshot* result) {
    result->clear();
    for (auto const& plane : generation->planes()) {
        std::vector<uint8_t> bytes;
        if (!ReadVulkanBytes(native, context, plane.buffer, plane.bytes, generation, &bytes) ||
            !result->emplace(plane.metadata.name, std::move(bytes)).second) return false;
        if (plane.bytes && !plane.buffer) return false;
    }
    return true;
}
static VulkanSourceGenerationCreateInfo Fixture(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& b = info.source;
    b.totalCurves = 2; b.totalCvs = 8; b.topologyVersion = 3; b.valueVersion = 5;
    b.px = {0,1,2,3,4,5,6,7}; b.py = {1,2,1,2,1,2,1,2}; b.pz = {2,2,2,2,2,2,2,2};
    b.rest = VtVec3fArray(8, GfVec3f(3,4,5)); b.width = VtFloatArray(8, 3.f);
    b.hairT = {0,.25f,.75f,1,0,.25f,.75f,1}; b.curveId = {7,19}; b.cvOffsets = {0,4,8};
    b.rootPrim = {3,7}; b.rootUV = {GfVec2f(.2f,.4f), GfVec2f(.7f,.8f)};
    b.rootT = VtVec3fArray(2, GfVec3f(0,1,0)); b.rootB = VtVec3fArray(2, GfVec3f(0,0,1));
    b.rootN = VtVec3fArray(2, GfVec3f(1,0,0)); UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{.25f,1.f}};
    UsdGenChunkDesc chunk; chunk.curveCount = 2; chunk.liveCount = 2; chunk.cvCount = 4;
    chunk.tile = 9; chunk.surface = 2; chunk.boundsRest = GfRange3f(GfVec3f(0), GfVec3f(8));
    b.chunks = {chunk};
    UsdGenPlane vf; vf.name = TfToken("vFloat"); vf.interpolation = TfToken("vertex");
    vf.type = TfToken("float"); vf.arity = 3; vf.f = VtFloatArray(24, 1.25f);
    UsdGenPlane vi; vi.name = TfToken("vInt"); vi.interpolation = TfToken("vertex");
    vi.type = TfToken("int"); vi.i = {1,2,3,4,5,6,7,8}; b.extraCv = {vf,vi};
    UsdGenPlane cf; cf.name = TfToken("constantFloat"); cf.interpolation = TfToken("constant");
    cf.type = TfToken("float"); cf.arity = 2; cf.f = {2,7};
    UsdGenPlane gi; gi.name = TfToken("guideIndex"); gi.interpolation = TfToken("uniform");
    gi.type = TfToken("int"); gi.arity = 3; gi.i = {3,2,1,7,8,9};
    UsdGenPlane gf; gf.name = TfToken("guideWeight"); gf.interpolation = TfToken("uniform");
    gf.type = TfToken("float"); gf.arity = 3; gf.f = {.2f,.3f,.5f,.1f,.2f,.7f};
    b.extraCurve = {cf,maskPlane,gi,gf};
    info.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::CatmullRom, UsdGenCurveLayoutWrap::Pinned);
    info.geometry.alreadyDeformed = true;
    UsdGenDeviceTileMetadata tile{9,0,2,0,8}; tile.extentMin = {0,0,0}; tile.extentMax = {8,8,8}; tile.boundsValid = true;
    info.geometry.tiles = {tile};
    float tileValue = 2.5f;
    info.additionalNamed.push_back({{"tileValue",UsdGenDeviceValueType::Float32,UsdGenDeviceDomain::Tile,1,1,4,true}, Bytes(&tileValue,1)});
    uint64_t toolValue = 123;
    info.additionalNamed.push_back({{"toolValue",UsdGenDeviceValueType::UInt64,UsdGenDeviceDomain::Tool,1,1,8,true}, Bytes(&toolValue,1)});
    return info;
}
int main(int argc, char** argv) {
    bool quarantine = argc == 2 && std::string(argv[1]) == "--quarantine";
    bool cow = argc == 3 && std::string(argv[1]) == "--cow";
    bool length = argc == 4 && std::string(argv[2]) == "--length";
    bool blend = argc == 4 && std::string(argv[1]) == "--blend";
    CHECK(argc == 1 || quarantine || cow || length || blend);
    bool unavailable = false; auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native); std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue; ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex; ci.resourceDeviceId = 7004;
    ci.nativeLifetime = native; ci.resources = {size_t{8} << 20,0};
    auto context = DeviceContext::Create(ci); CHECK(context); ci.nativeLifetime.reset();
    auto pool = context->resources(); auto baseline = pool->Snapshot();
    auto info = Fixture(context);
    VkResult status = VK_SUCCESS; std::string reason;
    auto upload = VulkanSourceUpload::Create(info, &status, &reason);
    if (!upload) std::fprintf(stderr, "Source Create: %s\n", reason.c_str());
    CHECK(upload && status == VK_SUCCESS && !upload->TakeReady());
    CHECK(upload->Poll() == SourceGenerationStatus::InvalidInput);
    auto staged = pool->Snapshot();
    CHECK(staged.byKind[size_t(UsdGenExecutionResourceKind::Scratch)] > baseline.byKind[size_t(UsdGenExecutionResourceKind::Scratch)]);
    CHECK(upload->Submit() == SourceGenerationStatus::Submitted);
    CHECK(upload->Submit() == SourceGenerationStatus::InvalidInput);
    if (quarantine) {
        auto charged = pool->Snapshot(); CHECK(charged.usedBytes > baseline.usedBytes);
        upload->Quarantine(); upload.reset(); info.context.reset(); context.reset();
        CHECK(Prove(native)); native.reset();
        CHECK(!weakNative.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes && pool->Snapshot().byKind == charged.byKind);
        return 0;
    }
    CHECK(Prove(native)); CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    CHECK(pool->Snapshot().byKind[size_t(UsdGenExecutionResourceKind::Scratch)] == baseline.byKind[size_t(UsdGenExecutionResourceKind::Scratch)]);
    auto a = upload->TakeReady(); CHECK(a);
    CHECK(a->ExclusiveRetainedBytes()==pool->Snapshot().usedBytes-baseline.usedBytes);
    CHECK(a->InclusiveRetainedBytes()==a->ExclusiveRetainedBytes());
    {
        auto again = upload->TakeReady(); CHECK(again);
        CHECK(again->PlaneOwner("points") == a->PlaneOwner("points"));
        CHECK(!a->PlaneOwner("sourceRootT"));
    }
    CHECK(a->curveCount() == 2 && a->pointCount() == 8 && a->topologyVersion() == 3 && a->valueVersion() == 5);
    CHECK(a->geometry().alreadyDeformed && UsdGenCurveLayoutBasisOf(a->geometry().layoutMetadata) == UsdGenCurveLayoutBasis::CatmullRom);
    CHECK(a->chunks().size() == 1 && a->chunks()[0].boundsRest == info.source.chunks[0].boundsRest);
    CHECK(a->geometry().tiles.size() == 1 && a->geometry().tiles[0].boundsValid && a->geometry().tiles[0].extentMax[0] == 8);
    CHECK(upload->Submit() == SourceGenerationStatus::InvalidInput);
    upload->Quarantine(); // A Ready handle cannot poison its already-published snapshot.
    upload.reset();
    Snapshot original, observed;
    CHECK(Capture(native, context, a, &original));
    CHECK(original.size() == 16 && a->sourceFrames().size() == 3);
    auto checkFrames = [&](std::shared_ptr<const VulkanSourceGeneration> const& generation) {
        for (auto const& plane : generation->sourceFrames()) {
            std::vector<uint8_t> values;
            if (!ReadVulkanBytes(native, context, plane.buffer, plane.bytes, generation, &values)) return false;
            auto const* expected = plane.metadata.name == "sourceRootT" ? &info.source.rootT :
                plane.metadata.name == "sourceRootB" ? &info.source.rootB :
                plane.metadata.name == "sourceRootN" ? &info.source.rootN : nullptr;
            if (!expected || values != Bytes(expected->cdata(), expected->size())) return false;
        }
        return generation->sourceFrames().size() == 3;
    };
    CHECK(checkFrames(a));
    std::vector<float> packed;
    for (size_t i=0;i<8;++i) { packed.push_back(info.source.px[i]); packed.push_back(info.source.py[i]); packed.push_back(info.source.pz[i]); }
    CHECK(original.at("points") == Bytes(packed.data(),packed.size()));
    CHECK(original.at("rest") == Bytes(info.source.rest.cdata(),8));
    CHECK(original.at("width") == Bytes(info.source.width.cdata(),8));
    CHECK(original.at("hairT") == Bytes(info.source.hairT.cdata(),8));
    CHECK(original.at("stableIds") == Bytes(info.source.curveId.cdata(),2));
    std::vector<uint32_t> offsets{0,4,8};
    CHECK(original.at("curveOffsets") == Bytes(offsets.data(),offsets.size()));
    CHECK(original.at("rootPrim") == Bytes(info.source.rootPrim.cdata(),2));
    CHECK(original.at("rootUV") == Bytes(info.source.rootUV.cdata(),2));
    CHECK(original.at("curveMask") == Bytes(info.source.extraCurve[1].f.cdata(),2));
    for (auto const& p:info.source.extraCv) CHECK(original.at(p.name.GetString()) ==
        (p.type == TfToken("int") ? Bytes(p.i.cdata(),p.i.size()) : Bytes(p.f.cdata(),p.f.size())));
    for (auto const& p:info.source.extraCurve) CHECK(original.at(p.name.GetString()) ==
        (p.type == TfToken("int") ? Bytes(p.i.cdata(),p.i.size()) : Bytes(p.f.cdata(),p.f.size())));
    for (auto const& p:info.additionalNamed) CHECK(original.at(p.metadata.name) == p.bytes);
    CHECK(original.count("sourceRootT") == 0 && original.count("sourceRootB") == 0 && original.count("sourceRootN") == 0);
    for (auto const& p:a->planes()) {
        if (p.metadata.name == "width") CHECK(p.metadata.semantic == UsdGenDeviceChannelSemantic::Widths);
        if (p.metadata.name == "rest") CHECK(p.metadata.semantic == UsdGenDeviceChannelSemantic::RestPoints);
        if (p.metadata.name == "guideIndex") CHECK(p.metadata.type == UsdGenDeviceValueType::Int32 && p.metadata.arity == 3 && p.metadata.elementCount == 2);
    }
    auto publish = [&](VulkanSourceGenerationCreateInfo const& input) {
        auto candidate = VulkanSourceUpload::Create(input, &status, &reason);
        if (!candidate || candidate->Submit() != SourceGenerationStatus::Submitted || !Prove(native) ||
            candidate->Poll() != SourceGenerationStatus::Ready) return std::shared_ptr<const VulkanSourceGeneration>{};
        return candidate->TakeReady();
    };
    auto revised = info; revised.source.px[0] = 10; revised.source.rest[0] = GfVec3f(9);
    revised.source.width[0] = 7; revised.source.extraCv[0].f[0] = 9; revised.source.valueVersion = 6;
    auto b = publish(revised); CHECK(b);
    CHECK(b->geometry().alreadyDeformed && b->geometry().tiles.size() == 1 &&
          b->geometry().tiles[0].boundsValid);
    CHECK(Capture(native, context, b, &observed)); CHECK(observed != original);
    CHECK(Capture(native, context, a, &observed) && observed == original);
    CHECK(checkFrames(a));
    auto permuted = info; permuted.source.curveId = {19,7};
    auto permutedGeneration = publish(permuted); CHECK(permutedGeneration);
    CHECK(Capture(native,context,permutedGeneration,&observed));
    CHECK(observed.at("stableIds") == Bytes(permuted.source.curveId.cdata(),2));
    auto absent = info; absent.source.rest.clear(); absent.source.hairT.clear(); absent.source.rootPrim.clear();
    absent.source.rootUV.clear(); absent.source.rootT.clear(); absent.source.rootB.clear(); absent.source.rootN.clear();
    absent.source.extraCurve.erase(absent.source.extraCurve.begin() + 1);
    auto c = publish(absent); CHECK(c); CHECK(Capture(native,context,c,&observed));
    CHECK(c->sourceFrames().empty());
    CHECK(!observed.count("rest") && !observed.count("hairT") && !observed.count("rootPrim") && !observed.count("rootUV") && !observed.count("curveMask"));
    VulkanSourceGenerationCreateInfo empty; empty.context = context;
    auto e = publish(empty); CHECK(e && !e->curveCount() && !e->pointCount());
    CHECK(e->sourceFrames().empty());
    CHECK(Capture(native,context,e,&observed)); uint32_t zero=0;
    CHECK(observed.at("curveOffsets") == Bytes(&zero,1));
    for (auto const& p:e->planes()) if (p.metadata.name != "curveOffsets") CHECK(!p.bytes && !p.buffer);
    if (cow) {
        std::ifstream shader(argv[2], std::ios::binary); CHECK(shader);
        std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
        CHECK(!raw.empty() && raw.size() % 4 == 0);
        std::vector<uint32_t> code(raw.size()/4); std::memcpy(code.data(),raw.data(),raw.size());
        auto pipeline = WidthPipeline::Create(context,code,&status); CHECK(pipeline);
        auto candidate = pipeline->Begin(a->PlaneOwner("width"),a->pointCount(),4.f,1,&status);
        CHECK(candidate && !candidate->succeeded());
        CHECK(!VulkanSourceGeneration::WithWidth(a,*candidate,6,&reason));
        CHECK(Prove(native)); uint32_t semantic=UINT32_MAX;
        CHECK(candidate->Poll(&semantic)==VK_SUCCESS && semantic==0 && candidate->succeeded());
        CHECK(!VulkanSourceGeneration::WithWidth(a,*candidate,5,&reason));
        CHECK(!VulkanSourceGeneration::WithWidth(a,*candidate,4,&reason));
        CHECK(!VulkanSourceGeneration::WithWidth(b,*candidate,7,&reason));
        auto beforeChild=pool->Snapshot();
        auto child=VulkanSourceGeneration::WithWidth(a,*candidate,6,&reason); CHECK(child);
        CHECK(child->ExclusiveRetainedBytes()==candidate->output()->allocationBytes());
        CHECK(child->InclusiveRetainedBytes()==a->InclusiveRetainedBytes()+child->ExclusiveRetainedBytes());
        CHECK(pool->Snapshot().usedBytes==beforeChild.usedBytes && pool->Snapshot().byKind==beforeChild.byKind);
        CHECK(child->valueVersion()==6 && child->geometry().valueVersion==6 && a->valueVersion()==5);
        CHECK(child->topologyVersion()==a->topologyVersion() && child->geometry().topologyVersion==a->geometry().topologyVersion);
        CHECK(child->geometry().alreadyDeformed==a->geometry().alreadyDeformed);
        CHECK(UsdGenCurveLayoutBasisOf(child->geometry().layoutMetadata)==UsdGenCurveLayoutBasisOf(a->geometry().layoutMetadata));
        CHECK(&child->chunks()==&a->chunks() && &child->sourceFrames()==&a->sourceFrames());
        for(auto const& plane:a->planes()) {
            auto baseOwner=a->PlaneOwner(plane.metadata.name);
            auto childOwner=child->PlaneOwner(plane.metadata.name);
            if(plane.metadata.semantic==UsdGenDeviceChannelSemantic::Widths) CHECK(childOwner!=baseOwner && childOwner==candidate->output());
            else CHECK(childOwner==baseOwner);
        }
        CHECK(Capture(native,context,child,&observed));
        auto expected=original; std::vector<float> four(8,4.f); expected["width"]=Bytes(four.data(),four.size());
        CHECK(observed==expected && checkFrames(child));
        candidate.reset();
        auto next=pipeline->Begin(child->PlaneOwner("width"),8,2.f,0,&status); CHECK(next && Prove(native));
        CHECK(next->Poll(&semantic)==VK_SUCCESS && semantic==0);
        auto grandchild=VulkanSourceGeneration::WithWidth(child,*next,7,&reason); CHECK(grandchild);
        CHECK(grandchild->InclusiveRetainedBytes()==child->InclusiveRetainedBytes()+grandchild->ExclusiveRetainedBytes());
        CHECK(grandchild->valueVersion()==7 && grandchild->geometry().valueVersion==7);
        CHECK(Capture(native,context,grandchild,&observed));
        std::vector<float> eight(8,8.f); expected["width"]=Bytes(eight.data(),eight.size()); CHECK(observed==expected);
        CHECK(Capture(native,context,a,&observed) && observed==original && checkFrames(a));
        CHECK(grandchild->PlaneOwner("points")==a->PlaneOwner("points"));
        auto shortCandidate=pipeline->Begin(a->PlaneOwner("width"),4,1.f,0,&status); CHECK(shortCandidate && Prove(native));
        CHECK(shortCandidate->Poll(&semantic)==VK_SUCCESS && semantic==0);
        CHECK(!VulkanSourceGeneration::WithWidth(a,*shortCandidate,8,&reason));
        auto emptyCandidate=pipeline->Begin({},0,1.f,0,&status); CHECK(emptyCandidate && emptyCandidate->succeeded());
        CHECK(!VulkanSourceGeneration::WithWidth(a,*emptyCandidate,8,&reason));
        auto emptyChild=VulkanSourceGeneration::WithWidth(e,*emptyCandidate,1,&reason); CHECK(emptyChild);
        CHECK(emptyChild->ExclusiveRetainedBytes()==0 && emptyChild->InclusiveRetainedBytes()==e->InclusiveRetainedBytes());
        CHECK(emptyChild->pointCount()==0 && emptyChild->valueVersion()==1 && e->valueVersion()==0);
        CHECK(emptyChild->geometry().valueVersion==1);
        bool hasEmptyWidth=false;
        for(auto const& plane:emptyChild->planes()) if(plane.metadata.semantic==UsdGenDeviceChannelSemantic::Widths) {
            CHECK(plane.metadata.elementCount==0 && plane.bytes==0 && !plane.buffer); hasEmptyWidth=true;
        }
        CHECK(hasEmptyWidth);
        auto otherInfo=ci; otherInfo.nativeLifetime=native;
        auto otherContext=DeviceContext::Create(otherInfo); CHECK(otherContext && otherContext!=context);
        auto otherPipeline=WidthPipeline::Create(otherContext,code,&status); CHECK(otherPipeline);
        auto wrongEmpty=otherPipeline->Begin({},0,1.f,0,&status); CHECK(wrongEmpty && wrongEmpty->succeeded());
        CHECK(!VulkanSourceGeneration::WithWidth(e,*wrongEmpty,2,&reason));
        auto badSemantic=pipeline->Begin(a->PlaneOwner("width"),8,std::numeric_limits<float>::max(),0,&status);
        CHECK(badSemantic && Prove(native));
        CHECK(badSemantic->Poll(&semantic)==VK_SUCCESS && semantic==2 && !badSemantic->succeeded());
        CHECK(!VulkanSourceGeneration::WithWidth(a,*badSemantic,9,&reason));
        next.reset(); child.reset();
        std::weak_ptr<const VulkanSourceGeneration> weakBase=a; a.reset();
        CHECK(!weakBase.expired());
        CHECK(Capture(native,context,grandchild,&observed) && observed==expected);
        a=weakBase.lock(); CHECK(a);
    }
    if (length) {
        auto loadCode = [](char const* path) {
            std::ifstream file(path, std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
            if (bytes.empty() || bytes.size() % sizeof(uint32_t)) return std::vector<uint32_t>{};
            std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
            std::memcpy(code.data(), bytes.data(), bytes.size());
            return code;
        };
        auto widthPipeline = WidthPipeline::Create(context, loadCode(argv[1]), &status);
        auto lengthPipeline = LengthScalePipeline::Create(context, loadCode(argv[3]), &status);
        CHECK(widthPipeline && lengthPipeline);
        auto run = [&](float factor) {
            auto candidate = lengthPipeline->Begin(a->PlaneOwner("points"),
                a->PlaneOwner("curveOffsets"), a->curveCount(), a->pointCount(), factor, &status);
            if (!candidate || status != VK_SUCCESS || candidate->succeeded() || !Prove(native)) return decltype(candidate){};
            uint32_t semantic = UINT32_MAX;
            if (candidate->Poll(&semantic) != VK_SUCCESS || semantic != 0 || !candidate->succeeded()) return decltype(candidate){};
            return candidate;
        };
        auto zeroRun = run(0.f); CHECK(zeroRun);
        auto zeroChild = VulkanSourceGeneration::WithPoints(a, *zeroRun, 20, &reason); CHECK(zeroChild);
        CHECK(zeroChild->geometry().alreadyDeformed == a->geometry().alreadyDeformed &&
              !zeroChild->geometry().tiles[0].boundsValid &&
              zeroChild->ExclusiveRetainedBytes() == zeroRun->output()->allocationBytes());
        Snapshot zeroObserved; CHECK(Capture(native, context, zeroChild, &zeroObserved));
        CHECK(checkFrames(zeroChild));
        std::vector<float> zeroPoints(24);
        for (size_t i = 0; i < 8; ++i) {
            size_t root = i < 4 ? 0 : 4;
            zeroPoints[3*i] = info.source.px[root]; zeroPoints[3*i+1] = info.source.py[root];
            zeroPoints[3*i+2] = info.source.pz[root];
        }
        auto zeroExpected = original; zeroExpected["points"] = Bytes(zeroPoints.data(), zeroPoints.size());
        CHECK(zeroObserved == zeroExpected);
        auto oneRun = run(1.f); CHECK(oneRun);
        auto oneChild = VulkanSourceGeneration::WithPoints(a, *oneRun, 21, &reason); CHECK(oneChild);
        CHECK(oneChild->geometry().alreadyDeformed == a->geometry().alreadyDeformed &&
              !oneChild->geometry().tiles[0].boundsValid &&
              oneChild->ExclusiveRetainedBytes() == oneRun->output()->allocationBytes());
        Snapshot oneObserved; CHECK(Capture(native, context, oneChild, &oneObserved));
        CHECK(checkFrames(oneChild));
        CHECK(oneObserved == original);
        auto scaledRun = run(2.5f); CHECK(scaledRun);
        auto scaledChild = VulkanSourceGeneration::WithPoints(a, *scaledRun, 22, &reason); CHECK(scaledChild);
        CHECK(scaledChild->geometry().alreadyDeformed == a->geometry().alreadyDeformed &&
              !scaledChild->geometry().tiles[0].boundsValid &&
              scaledChild->ExclusiveRetainedBytes() == scaledRun->output()->allocationBytes());
        Snapshot scaledObserved; CHECK(Capture(native, context, scaledChild, &scaledObserved));
        CHECK(checkFrames(scaledChild));
        std::vector<float> scaledPoints(24);
        for (size_t i = 0; i < 8; ++i) {
            size_t root = i < 4 ? 0 : 4;
            scaledPoints[3*i] = info.source.px[root] + (info.source.px[i] - info.source.px[root]) * 2.5f;
            scaledPoints[3*i+1] = info.source.py[root] + (info.source.py[i] - info.source.py[root]) * 2.5f;
            scaledPoints[3*i+2] = info.source.pz[root] + (info.source.pz[i] - info.source.pz[root]) * 2.5f;
        }
        auto scaledExpected = original; scaledExpected["points"] = Bytes(scaledPoints.data(), scaledPoints.size());
        CHECK(scaledObserved == scaledExpected);
        auto pointsChild = scaledChild;
        auto notDeformedInfo = info;
        notDeformedInfo.geometry.alreadyDeformed = false;
        auto notDeformed = publish(notDeformedInfo); CHECK(notDeformed);
        auto undeformedRun = lengthPipeline->Begin(notDeformed->PlaneOwner("points"),
            notDeformed->PlaneOwner("curveOffsets"), 2, 8, 2.5f, &status);
        CHECK(undeformedRun && Prove(native));
        uint32_t undeformedStatus = UINT32_MAX;
        CHECK(undeformedRun->Poll(&undeformedStatus) == VK_SUCCESS && undeformedStatus == 0);
        auto undeformedChild = VulkanSourceGeneration::WithPoints(notDeformed, *undeformedRun, 30, &reason);
        CHECK(undeformedChild && !undeformedChild->geometry().alreadyDeformed &&
              !undeformedChild->geometry().tiles[0].boundsValid);
        CHECK(undeformedChild->PlaneOwner("rest") == notDeformed->PlaneOwner("rest"));
        CHECK(pointsChild->PlaneOwner("points") != a->PlaneOwner("points"));
        for (auto const& plane : a->planes()) {
            if (plane.metadata.name == "points") CHECK(pointsChild->PlaneOwner("points") == scaledRun->output());
            else CHECK(pointsChild->PlaneOwner(plane.metadata.name) == a->PlaneOwner(plane.metadata.name));
        }
        CHECK(pointsChild->geometry().alreadyDeformed && !pointsChild->geometry().tiles[0].boundsValid);
        CHECK(checkFrames(pointsChild));
        auto widthCandidate = widthPipeline->Begin(pointsChild->PlaneOwner("width"), pointsChild->pointCount(), 2.f, 0, &status);
        CHECK(widthCandidate && Prove(native)); uint32_t widthSemantic = UINT32_MAX;
        CHECK(widthCandidate->Poll(&widthSemantic) == VK_SUCCESS && widthSemantic == 0);
        auto pointsAndWidth = VulkanSourceGeneration::WithWidth(pointsChild, *widthCandidate, 23, &reason);
        CHECK(pointsAndWidth && pointsAndWidth->PlaneOwner("points") == pointsChild->PlaneOwner("points"));
        CHECK(pointsAndWidth->geometry().alreadyDeformed && !pointsAndWidth->geometry().tiles[0].boundsValid);
        CHECK(checkFrames(pointsAndWidth));

        auto beforeHook = pool->Snapshot(); int hookCalls = 0;
        auto rejected = lengthPipeline->Begin(a->PlaneOwner("points"), a->PlaneOwner("curveOffsets"),
            a->curveCount(), a->pointCount(), 2.f, &status, [&] { ++hookCalls; return false; });
        CHECK(!rejected && hookCalls == 1);
        CHECK(pool->Snapshot().usedBytes == beforeHook.usedBytes && pool->Snapshot().byKind == beforeHook.byKind);
        CHECK(!lengthPipeline->Begin(a->PlaneOwner("points"), a->PlaneOwner("curveOffsets"),
            a->curveCount(), a->pointCount(), std::numeric_limits<float>::quiet_NaN(), &status));
        CHECK(!lengthPipeline->Begin(a->PlaneOwner("points"), a->PlaneOwner("curveOffsets"),
            a->curveCount(), a->pointCount(), std::numeric_limits<float>::infinity(), &status));

        VkBufferCreateInfo malformedInfo{}; malformedInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        malformedInfo.size = 3 * sizeof(uint32_t); malformedInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        malformedInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto malformedOffsets = ChargedBuffer::Create(context, malformedInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active, &status); CHECK(malformedOffsets);
        uint32_t badOffsets[3] = {0, 9, 8}; void* malformedMapped = nullptr;
        CHECK(vkMapMemory(native->device, malformedOffsets->memory(), 0, malformedInfo.size, 0, &malformedMapped) == VK_SUCCESS);
        std::memcpy(malformedMapped, badOffsets, malformedInfo.size); vkUnmapMemory(native->device, malformedOffsets->memory());
        auto malformed = lengthPipeline->Begin(a->PlaneOwner("points"), malformedOffsets, 2, 8, 2.f, &status);
        CHECK(malformed && Prove(native)); uint32_t malformedStatus = UINT32_MAX;
        CHECK(malformed->Poll(&malformedStatus) == VK_SUCCESS && malformedStatus != 0 && !malformed->succeeded());
        CHECK(!VulkanSourceGeneration::WithPoints(a, *malformed, 24, &reason));
        auto overflow = lengthPipeline->Begin(a->PlaneOwner("points"), a->PlaneOwner("curveOffsets"), 2, 8,
            std::numeric_limits<float>::max(), &status);
        CHECK(overflow && Prove(native)); uint32_t overflowStatus = UINT32_MAX;
        CHECK(overflow->Poll(&overflowStatus) == VK_SUCCESS && overflowStatus != 0 && !overflow->succeeded());
        CHECK(!VulkanSourceGeneration::WithPoints(a, *overflow, 25, &reason));
        auto foreignOffsets = lengthPipeline->Begin(a->PlaneOwner("points"), b->PlaneOwner("curveOffsets"), 2, 8, 1.f, &status);
        CHECK(foreignOffsets && Prove(native)); uint32_t foreignStatus = UINT32_MAX;
        CHECK(foreignOffsets->Poll(&foreignStatus) == VK_SUCCESS && foreignStatus == 0);
        CHECK(!VulkanSourceGeneration::WithPoints(a, *foreignOffsets, 26, &reason));
        auto empty = lengthPipeline->Begin(e->PlaneOwner("points"), e->PlaneOwner("curveOffsets"), 0, 0, 1.f, &status);
        CHECK(empty && empty->succeeded());
        auto emptyChild = VulkanSourceGeneration::WithPoints(e, *empty, 1, &reason); CHECK(emptyChild);
        CHECK(emptyChild->pointCount() == 0 && emptyChild->ExclusiveRetainedBytes() == 0);
    }
    if (blend) {
        auto loadCode = [](char const* path) {
            std::ifstream file(path, std::ios::binary);
            std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
            if (bytes.empty() || bytes.size() % sizeof(uint32_t)) return std::vector<uint32_t>{};
            std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
            std::memcpy(code.data(), bytes.data(), bytes.size());
            return code;
        };
        auto widths = WidthPipeline::Create(context, loadCode(argv[2]), &status); CHECK(widths);
        auto blends = WidthBlendPipeline::Create(context, loadCode(argv[3]), &status); CHECK(blends);
        auto leftRun = widths->Begin(a->PlaneOwner("width"), 8, 2.f, 0, &status);
        auto rightRun = widths->Begin(a->PlaneOwner("width"), 8, 3.f, 0, &status);
        CHECK(leftRun && rightRun && Prove(native));
        uint32_t semantic = UINT32_MAX;
        CHECK(leftRun->Poll(&semantic) == VK_SUCCESS && semantic == 0);
        CHECK(rightRun->Poll(&semantic) == VK_SUCCESS && semantic == 0);
        auto left = VulkanSourceGeneration::WithWidth(a, *leftRun, 6, &reason);
        auto right = VulkanSourceGeneration::WithWidth(a, *rightRun, 7, &reason);
        CHECK(left && right && left->NonWidthIdentity() == right->NonWidthIdentity());
        CHECK(left->NonWidthIdentity() != b->NonWidthIdentity());
        for (float weight : {0.f, 0.25f, 1.f}) {
            auto candidate = blends->Begin(left->PlaneOwner("width"), right->PlaneOwner("width"), 8, weight, &status);
            CHECK(candidate && !candidate->output() && Prove(native));
            CHECK(candidate->Poll(&semantic) == VK_SUCCESS && semantic == 0);
            auto child = VulkanSourceGeneration::WithWidthBlend(left, right, *candidate, 8, &reason);
            CHECK(child && child->NonWidthIdentity() == a->NonWidthIdentity());
            CHECK(!VulkanSourceGeneration::WithWidthBlend(right, left, *candidate, 8, &reason));
            CHECK(!VulkanSourceGeneration::WithWidthBlend(left, right, *candidate, 7, &reason));
            CHECK(child->ExclusiveRetainedBytes() == candidate->output()->allocationBytes());
            for (auto const& plane : a->planes())
                if (plane.metadata.name != "width") CHECK(child->PlaneOwner(plane.metadata.name) == a->PlaneOwner(plane.metadata.name));
            std::vector<float> expectedWidths(8);
            for (size_t i = 0; i != expectedWidths.size(); ++i) {
                float l = info.source.width[i] * 2.f, r = info.source.width[i] * 3.f;
                expectedWidths[i] = weight == 0.f ? l : weight == 1.f ? r : l + (r - l) * weight;
            }
            auto expected = original;
            expected["width"] = Bytes(expectedWidths.data(), expectedWidths.size());
            CHECK(Capture(native, context, child, &observed) && observed == expected);
            CHECK(checkFrames(child));
            CHECK(Capture(native, context, a, &observed) && observed == original);
        }
        auto foreign = blends->Begin(left->PlaneOwner("width"), b->PlaneOwner("width"), 8, 0.5f, &status);
        CHECK(foreign && Prove(native) && foreign->Poll(&semantic) == VK_SUCCESS && semantic == 0);
        CHECK(!VulkanSourceGeneration::WithWidthBlend(left, b, *foreign, 9, &reason));
        auto emptyOther = publish(empty); CHECK(emptyOther);
        auto emptyRun = blends->Begin({}, {}, 0, 0.5f, &status);
        CHECK(emptyRun && emptyRun->succeeded());
        CHECK(VulkanSourceGeneration::WithWidthBlend(e, e, *emptyRun, 1, &reason));
        CHECK(!VulkanSourceGeneration::WithWidthBlend(e, emptyOther, *emptyRun, 1, &reason));
        auto before = pool->Snapshot();
        int hooks = 0;
        CHECK(!blends->Begin(left->PlaneOwner("width"), right->PlaneOwner("width"), 8, 0.5f, &status,
                            [&] { ++hooks; return false; }));
        CHECK(hooks == 1 && pool->Snapshot().usedBytes == before.usedBytes);
        CHECK(!blends->Begin(left->PlaneOwner("width"), right->PlaneOwner("width"), 8,
                            std::numeric_limits<float>::quiet_NaN(), &status));
        VkBufferCreateInfo malformedInfo{};
        malformedInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        malformedInfo.size = 8 * sizeof(float);
        malformedInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        malformedInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto malformed = ChargedBuffer::Create(context, malformedInfo,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active, &status); CHECK(malformed);
        for (float bad : {-1.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            float data[8] = {1.f}; data[2] = bad;
            void* mapped = nullptr;
            CHECK(vkMapMemory(native->device, malformed->memory(), 0, sizeof(data), 0, &mapped) == VK_SUCCESS);
            std::memcpy(mapped, data, sizeof(data));
            vkUnmapMemory(native->device, malformed->memory());
            // Even endpoint zero validates the unselected right input.
            auto invalid = blends->Begin(left->PlaneOwner("width"), malformed, 8, 0.f, &status);
            CHECK(invalid && Prove(native));
            CHECK(invalid->Poll(&semantic) == VK_SUCCESS && semantic != 0 && !invalid->output());
        }
    }
    for (int kind=0;kind<10;++kind) {
        auto bad = info;
        switch(kind) {
        case 0: bad.source.px.pop_back(); break;
        case 1: bad.source.cvOffsets[1] = -1; break;
        case 2: bad.source.rootN.clear(); break;
        case 3: bad.source.extraCv[0].type = TfToken("unknown"); break;
        case 4: bad.source.extraCv[0].f.pop_back(); break;
        case 5: bad.additionalNamed[0].bytes.pop_back(); break;
        case 6: bad.additionalNamed[0].metadata.name = "width"; break;
        case 7: bad.geometry.tiles[0].pointCount = 999; break;
        case 8: bad.source.chunks[0].curveCount = 999; break;
        case 9: bad.source.curveId = {7,7}; break;
        }
        auto before = pool->Snapshot(); CHECK(!VulkanSourceUpload::Create(bad,&status,&reason));
        CHECK(pool->Snapshot().usedBytes == before.usedBytes && pool->Snapshot().byKind == before.byKind);
    }
    {
        auto before = pool->Snapshot();
        auto filler = pool->TryReserve(before.usableBytes-before.usedBytes,UsdGenExecutionResourceKind::Active); CHECK(filler);
        auto full = pool->Snapshot(); CHECK(!VulkanSourceUpload::Create(info,&status,&reason));
        CHECK(status == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        CHECK(pool->Snapshot().usedBytes == full.usedBytes && pool->Snapshot().byKind == full.byKind);
    }
    CHECK(Capture(native,context,a,&observed) && observed == original);
    e.reset(); c.reset(); b.reset(); a.reset(); permutedGeneration.reset();
    info.context.reset(); revised.context.reset(); absent.context.reset(); empty.context.reset();
    permuted.context.reset();
    context.reset(); native.reset();
    CHECK(weakNative.expired());
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes && pool->Snapshot().byKind == baseline.byKind);
    return 0;
}
