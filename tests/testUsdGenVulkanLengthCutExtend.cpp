// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "utils/testToken.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/types.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan cut/extend failed: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
template<class T> std::vector<uint8_t> Bytes(T const* p, size_t n) {
    std::vector<uint8_t> r(n * sizeof(T));
    if (!r.empty()) std::memcpy(r.data(), p, r.size());
    return r;
}
static std::vector<uint32_t> Code(char const* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> r(raw.size() / sizeof(uint32_t));
    std::memcpy(r.data(), raw.data(), raw.size()); return r;
}
static bool Prove(std::shared_ptr<NativeOwner> const& n) {
    if (vkResetFences(n->device, 1, &n->fence) != VK_SUCCESS) return false;
    VkSubmitInfo s{}; s.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(n->queue, 1, &s, n->fence) == VK_SUCCESS &&
        vkWaitForFences(n->device, 1, &n->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}
using Packet = std::map<std::string, std::vector<uint8_t>>;
static bool Capture(std::shared_ptr<NativeOwner> const& n,
    std::shared_ptr<DeviceContext> const& c,
    std::shared_ptr<const VulkanSourceGeneration> const& g, Packet* out) {
    out->clear();
    auto add = [&](VulkanSourceGeneration::PlaneView const& p) {
        std::vector<uint8_t> b;
        return ReadVulkanBytes(n, c, p.buffer, p.bytes, g, &b) &&
            out->emplace(p.metadata.name.GetString(), std::move(b)).second;
    };
    for (auto const& p : g->planes()) if (!add(p)) return false;
    for (auto const& p : g->sourceFrames()) if (!add(p)) return false;
    return true;
}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> c,
                                                bool zero = false) {
    VulkanSourceGenerationCreateInfo i; i.context = std::move(c); auto& s = i.source;
    s.totalCurves = 2; s.totalCvs = 7; s.topologyVersion = 92; s.valueVersion = 17;
    s.cvOffsets = {0, 3, 7}; s.curveId = {401, 907}; s.rootPrim = {12, 44};
    s.rootUV = {{.2f,.3f},{.7f,.8f}}; s.rootT = {{1,0,0},{0,1,0}};
    s.rootB = {{0,1,0},{0,0,1}}; s.rootN = {{0,0,1},{1,0,0}}; UsdGenPlane maskPlane; maskPlane.name = TestToken("curveMask"); maskPlane.interpolation = TestToken("uniform"); maskPlane.type = TestToken("float"); maskPlane.f = {{.25f,.75f}};
    s.px = zero ? VtFloatArray(7, 0.f) : VtFloatArray{0,2,2, 10,10,13,13};
    s.py = zero ? VtFloatArray(7, 0.f) : VtFloatArray{0,0,2, 10,12,12,12};
    s.pz = VtFloatArray(7, 0.f);
    s.rest = {usdGen::UsdGenVec3f(0,0,0),usdGen::UsdGenVec3f(0,10,0),usdGen::UsdGenVec3f(0,20,0),
              usdGen::UsdGenVec3f(9,9,9),usdGen::UsdGenVec3f(9,19,9),usdGen::UsdGenVec3f(39,19,9),usdGen::UsdGenVec3f(39,19,9)};
    s.width = VtFloatArray(7, 1.f); s.hairT = {0,.5f,1,0,.4f,.8f,1};
    UsdGenChunkDesc chunk; chunk.firstCurve = 0; chunk.curveCount = 2; chunk.liveCount = 2;
    chunk.firstCv = 0; chunk.cvCount = 0; chunk.tile = 3; chunk.surface = 1; s.chunks = {chunk};
    UsdGenPlane p; p.name = TestToken("privateVertex"); p.interpolation = TestToken("vertex");
    p.type = TestToken("float"); p.arity = 1; p.f = {3,4,5,6,7,8,9}; s.extraCv = {p};
    UsdGenPlane q; q.name = TestToken("privateUniform"); q.interpolation = TestToken("uniform");
    q.type = TestToken("int"); q.i = {18,19}; s.extraCurve = {maskPlane, q};
    i.geometry.alreadyDeformed = true;
    i.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline,
                                UsdGenCurveLayoutWrap::Pinned);
    i.geometry.tiles = {{3,0,2,0,7}};
    uint32_t named[] = {73, 81};
    i.additionalNamed.push_back({{TestToken("namedCurve"), UsdGenDeviceValueType::UInt32,
        UsdGenDeviceDomain::Primitive, 2, 1, 4, true}, Bytes(named, 2)});
    return i;
}
static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner> const& n,
                                                              VulkanSourceGenerationCreateInfo i) {
    VkResult r = VK_SUCCESS; std::string why; auto u = VulkanSourceUpload::Create(std::move(i), &r, &why);
    if (!u || u->Submit() != SourceGenerationStatus::Submitted || !Prove(n) ||
        u->Poll() != SourceGenerationStatus::Ready) return {};
    return u->TakeReady();
}
}

int main(int argc, char** argv) {
    CHECK(argc == 4); auto scaleCode = Code(argv[1]); auto setCode = Code(argv[2]); auto cutCode = Code(argv[3]);
    CHECK(!scaleCode.empty() && !setCode.empty() && !cutCode.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable); if (unavailable) return 77; CHECK(native);
    DeviceContext::CreateInfo ci; ci.instance=native->instance; ci.physicalDevice=native->physical;
    ci.device=native->device; ci.computeQueue=native->queue; ci.computeQueueFamily=native->family;
    ci.physicalIndex=native->physicalIndex; ci.resourceDeviceId=7932; ci.nativeLifetime=native; ci.timelineSemaphoreEnabled=true;
    ci.resources={size_t{32}<<20,0};
    auto context = DeviceContext::Create(ci); CHECK(context); auto pool = context->resources();
    auto initialBaseline = pool->Snapshot();
    VkResult status = VK_SUCCESS; auto base = Upload(native, Source(context)); CHECK(base);
    auto liveBaseline = pool->Snapshot();
    unsigned featureHooks = 0;
    auto old = LengthScalePipeline::Create(context, scaleCode, &status);
    CHECK(old && !old->HasCutExtend());
    auto oldCut = old->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
        base->curveCount(), base->pointCount(), 2.f, true, &status,
        [&] { ++featureHooks; return true; });
    CHECK(!oldCut && status == VK_ERROR_FEATURE_NOT_PRESENT && featureHooks == 0 &&
          pool->Snapshot().usedBytes == liveBaseline.usedBytes &&
          pool->Snapshot().byKind == liveBaseline.byKind); old.reset();
    auto oldSet = LengthScalePipeline::CreateWithSet(context, scaleCode, setCode, &status);
    CHECK(oldSet && !oldSet->HasCutExtend());
    oldCut = oldSet->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
        base->curveCount(), base->pointCount(), 2.f, true, &status,
        [&] { ++featureHooks; return true; });
    CHECK(!oldCut && status == VK_ERROR_FEATURE_NOT_PRESENT && featureHooks == 0 &&
          pool->Snapshot().usedBytes == liveBaseline.usedBytes &&
          pool->Snapshot().byKind == liveBaseline.byKind); oldSet.reset();
    auto baseline = pool->Snapshot();
    auto pipeline = LengthScalePipeline::CreateWithCutExtend(context, scaleCode, setCode, cutCode, &status);
    CHECK(pipeline && pipeline->HasSet() && pipeline->HasCutExtend());
    auto run = [&](std::shared_ptr<const VulkanSourceGeneration> const& g, float value, bool absolute) {
        auto c = pipeline->BeginCutExtend(g->PlaneOwner(TestToken("points")), g->PlaneOwner(TestToken("curveOffsets")),
            g->curveCount(), g->pointCount(), value, absolute, &status);
        if (!c || status != VK_SUCCESS || !Prove(native))
            return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{}, UINT32_MAX);
        uint32_t sem = UINT32_MAX;
        if (c->Poll(&sem) != VK_SUCCESS) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{}, UINT32_MAX);
        return std::pair<std::unique_ptr<LengthScalePipeline::Candidate>, uint32_t>(std::move(c), sem);
    };
    auto makeChild = [&](std::shared_ptr<const VulkanSourceGeneration> const& g,
                         LengthScalePipeline::Candidate const& c, uint64_t version) {
        std::string why; return VulkanSourceGeneration::WithPoints(g, c, version, &why);
    };
    Packet original; CHECK(Capture(native, context, base, &original));
    CHECK(original.count("stableIds") && original.count("rest") && original.count("sourceRootT"));
    auto verifyChild = [&](std::shared_ptr<const VulkanSourceGeneration> const& child, float const* points) {
        CHECK(child && child->PlaneOwner(TestToken("points")) != base->PlaneOwner(TestToken("points")));
        CHECK(child->topologyVersion() == base->topologyVersion());
        for (auto const& plane : base->planes()) if (plane.metadata.name != "points")
            CHECK(child->PlaneOwner(plane.metadata.name) == base->PlaneOwner(plane.metadata.name));
        for (auto const& frame : base->sourceFrames()) {
            CHECK(child->SourceFrameOwner(frame.metadata.name) == base->SourceFrameOwner(frame.metadata.name));
            CHECK(!child->PlaneOwner(frame.metadata.name));
        }
        CHECK(child->geometry().alreadyDeformed == base->geometry().alreadyDeformed &&
              UsdGenCurveLayoutTypeOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutTypeOf(base->geometry().layoutMetadata) &&
              UsdGenCurveLayoutBasisOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutBasisOf(base->geometry().layoutMetadata) &&
              UsdGenCurveLayoutWrapOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutWrapOf(base->geometry().layoutMetadata) &&
              child->geometry().tiles.size() == base->geometry().tiles.size() &&
              child->chunks().size() == base->chunks().size());
        for (size_t i = 0; i != base->geometry().tiles.size(); ++i) {
            auto const& a = child->geometry().tiles[i];
            auto const& b = base->geometry().tiles[i];
            CHECK(a.tile == b.tile && a.firstElement == b.firstElement &&
                  a.elementCount == b.elementCount && a.firstPoint == b.firstPoint &&
                  a.pointCount == b.pointCount && a.extentMin == b.extentMin &&
                  a.extentMax == b.extentMax && a.boundsValid == b.boundsValid);
        }
        auto const& a = child->chunks()[0]; auto const& b = base->chunks()[0];
        CHECK(a.firstCurve == b.firstCurve && a.curveCount == b.curveCount &&
              a.liveCount == b.liveCount && a.firstCv == b.firstCv && a.cvCount == b.cvCount &&
              a.tile == b.tile && a.surface == b.surface);
        Packet packet; CHECK(Capture(native, context, child, &packet) && packet.size() == original.size());
        for (auto const& p : original) if (p.first != "points") CHECK(packet.at(p.first) == p.second);
        CHECK(packet.at("points") == Bytes(points, 21));
        return true;
    };
    auto pressure = pool->TryReserve(baseline.usableBytes - baseline.usedBytes,
        UsdGenExecutionResourceKind::Scratch);
    CHECK(pressure);
    auto saturatedBaseline = pool->Snapshot();
    unsigned admissionHooks = 0;
    auto noRoom = pipeline->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
        base->curveCount(), base->pointCount(), 3.f, true, &status,
        [&] { ++admissionHooks; return true; });
    auto saturatedAfter = pool->Snapshot();
    CHECK(!noRoom && status == VK_ERROR_OUT_OF_DEVICE_MEMORY && admissionHooks == 0 &&
          saturatedAfter.usedBytes == saturatedBaseline.usedBytes &&
          saturatedAfter.byKind == saturatedBaseline.byKind);
    pressure->Release();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes &&
          pool->Snapshot().byKind == baseline.byKind);
    auto beforeHookFailure = pool->Snapshot();
    auto hookFalse = pipeline->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
        base->curveCount(), base->pointCount(), 3.f, true, &status,
        [&] { ++admissionHooks; return false; });
    auto afterHookFalse = pool->Snapshot();
    CHECK(!hookFalse && status == VK_ERROR_OUT_OF_DEVICE_MEMORY && admissionHooks == 1 &&
          afterHookFalse.usedBytes == beforeHookFailure.usedBytes && afterHookFalse.byKind == beforeHookFailure.byKind);
    auto hookThrow = pipeline->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
        base->curveCount(), base->pointCount(), 3.f, true, &status,
        [&]() -> bool { ++admissionHooks; throw std::runtime_error("cut admission"); });
    auto afterHookThrow = pool->Snapshot();
    CHECK(!hookThrow && status == VK_ERROR_UNKNOWN && admissionHooks == 2 &&
          afterHookThrow.usedBytes == beforeHookFailure.usedBytes && afterHookThrow.byKind == beforeHookFailure.byKind);
    auto empty = pipeline->BeginCutExtend({}, {}, 0, 0, 1.f, true, &status,
        [&] { ++admissionHooks; return true; });
    CHECK(empty && status == VK_SUCCESS && empty->succeeded() && admissionHooks == 2);
    auto invalid = [&](float value) {
        return pipeline->BeginCutExtend(base->PlaneOwner(TestToken("points")), base->PlaneOwner(TestToken("curveOffsets")),
            base->curveCount(), base->pointCount(), value, true, &status,
            [&] { ++admissionHooks; return true; });
    };
    CHECK(!invalid(-1.f) && !invalid(std::numeric_limits<float>::quiet_NaN()) &&
          !invalid(std::numeric_limits<float>::infinity()) && admissionHooks == 2 &&
          pool->Snapshot().usedBytes == beforeHookFailure.usedBytes &&
          pool->Snapshot().byKind == beforeHookFailure.byKind);
    empty.reset();
    auto cut = run(base, 3.f, true); CHECK(cut.second == 0 && cut.first->succeeded());
    auto cutChild = makeChild(base, *cut.first, 18);
    float cutPoints[] = {0,0,0, 2,0,0, 2,1,0, 10,10,0, 10,12,0, 11,12,0, 11,12,0};
    CHECK(verifyChild(cutChild, cutPoints));
    Packet cutPacket; CHECK(Capture(native, context, cutChild, &cutPacket));
    auto extend = run(base, 6.f, true); CHECK(extend.second == 0 && extend.first->succeeded());
    auto extendChild = makeChild(base, *extend.first, 19);
    float extendPoints[] = {0,0,0, 2,0,0, 2,4,0, 10,10,0, 10,12,0, 13,12,0, 14,12,0};
    CHECK(verifyChild(extendChild, extendPoints));
    auto half = run(base, .5f, false); CHECK(half.second == 0 && half.first->succeeded());
    auto halfChild = makeChild(base, *half.first, 20);
    float halfPoints[] = {0,0,0, 2,0,0, 2,0,0, 10,10,0, 10,12,0, 10.5f,12,0, 10.5f,12,0};
    CHECK(verifyChild(halfChild, halfPoints));
    auto oneHalf = run(base, 1.5f, false); CHECK(oneHalf.second == 0 && oneHalf.first->succeeded());
    auto oneHalfChild = makeChild(base, *oneHalf.first, 21);
    float oneHalfPoints[] = {0,0,0, 2,0,0, 2,4,0, 10,10,0, 10,12,0, 13,12,0, 15.5f,12,0};
    CHECK(verifyChild(oneHalfChild, oneHalfPoints));
    auto identity = run(base, 1.f, false); CHECK(identity.second == 0 && identity.first->succeeded());
    auto identityChild = makeChild(base, *identity.first, 22); Packet identityPacket;
    CHECK(Capture(native, context, identityChild, &identityPacket) && identityPacket == original);
    Packet sourceAfter; CHECK(Capture(native, context, base, &sourceAfter) && sourceAfter == original);
    Packet cutAfter; CHECK(Capture(native, context, cutChild, &cutAfter) && cutAfter == cutPacket);
    auto tinyInfo = Source(context);
    tinyInfo.source.px = {0.f, 1e-13f, 2e-13f, 0.f, 1e-13f, 2e-13f, 3e-13f};
    tinyInfo.source.py = VtFloatArray(7, 0.f);
    auto tiny = Upload(native, std::move(tinyInfo)); CHECK(tiny);
    Packet tinyBefore, tinyAfter;
    CHECK(Capture(native, context, tiny, &tinyBefore));
    auto tinyNoTangent = run(tiny, 1.f, true);
    CHECK(tinyNoTangent.first && tinyNoTangent.second == 0 && tinyNoTangent.first->succeeded());
    auto tinyChild = makeChild(tiny, *tinyNoTangent.first, 25);
    CHECK(Capture(native, context, tinyChild, &tinyAfter) && tinyAfter == tinyBefore);
    auto numericRejected = [&](VulkanSourceGenerationCreateInfo info, float value, bool absolute,
                               uint64_t version) {
        auto g = Upload(native, std::move(info)); CHECK(g);
        Packet before, after; CHECK(Capture(native, context, g, &before));
        auto candidate = run(g, value, absolute);
        CHECK(candidate.first && candidate.second != 0 && !candidate.first->succeeded() &&
              !candidate.first->output() && !VulkanSourceGeneration::WithPoints(g, *candidate.first, version));
        CHECK(Capture(native, context, g, &after) && after == before);
    };
    auto quotientOverflow = Source(context);
    quotientOverflow.source.px = {0.f, 5e-11f, 1e-10f, 0.f, 5e-11f, 1e-10f, 1e-10f};
    quotientOverflow.source.py = VtFloatArray(7, 0.f);
    numericRejected(std::move(quotientOverflow), 1e30f, true, 26);
    auto productOverflow = Source(context);
    productOverflow.source.px = {0.f, 5e9f, 1e10f, 0.f, 5e9f, 1e10f, 1e10f};
    productOverflow.source.py = VtFloatArray(7, 0.f);
    numericRejected(std::move(productOverflow), 1e30f, false, 27);
    auto arcSquareOverflow = Source(context);
    arcSquareOverflow.source.px = {0.f, .5f, 1.f, 0.f, .5f, 1.f, 1.f};
    arcSquareOverflow.source.py = VtFloatArray(7, 0.f);
    numericRejected(std::move(arcSquareOverflow), 1e20f, true, 28);
    auto currentArcOverflow = Source(context);
    currentArcOverflow.source.px = {-1e30f, 0.f, 1e30f, -1e30f, 0.f, 1e30f, 1e30f};
    currentArcOverflow.source.py = VtFloatArray(7, 0.f);
    numericRejected(std::move(currentArcOverflow), 2.f, true, 29);
    auto zero = Upload(native, Source(context, true)); CHECK(zero);
    auto rejected = run(zero, 2.f, true); CHECK(rejected.first && rejected.second != 0 && !rejected.first->succeeded() &&
        !rejected.first->output() && !VulkanSourceGeneration::WithPoints(zero, *rejected.first, 23));
    auto zeroNoop = run(zero, 0.f, true); CHECK(zeroNoop.second == 0 && zeroNoop.first->succeeded());
    auto zeroChild = makeChild(zero, *zeroNoop.first, 24); Packet zeroBefore, zeroAfter;
    CHECK(Capture(native, context, zero, &zeroBefore) && Capture(native, context, zeroChild, &zeroAfter) && zeroBefore == zeroAfter);
    zeroChild.reset(); rejected.first.reset(); zeroNoop.first.reset(); tinyChild.reset(); tinyNoTangent.first.reset();
    tiny.reset(); identityChild.reset(); oneHalfChild.reset();
    halfChild.reset(); extendChild.reset(); cutChild.reset(); identity.first.reset(); oneHalf.first.reset(); half.first.reset();
    extend.first.reset(); cut.first.reset(); pipeline.reset(); zero.reset(); base.reset();
    CHECK(pool->Snapshot().usedBytes == initialBaseline.usedBytes &&
          pool->Snapshot().byKind == initialBaseline.byKind);
    std::puts("Vulkan cut/extend Length COW: PASS");
    return 0;
}
