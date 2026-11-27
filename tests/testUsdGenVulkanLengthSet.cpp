// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan length-set failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

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
        return ReadVulkanBytes(n, c, p.buffer, p.bytes, g, &b) && out->emplace(p.metadata.name, std::move(b)).second;
    };
    for (auto const& p : g->planes()) if (!add(p)) return false;
    for (auto const& p : g->sourceFrames()) if (!add(p)) return false;
    return true;
}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> c, bool zero = false) {
    VulkanSourceGenerationCreateInfo i; i.context = std::move(c); auto& s = i.source;
    s.totalCurves = 2; s.totalCvs = 7; s.topologyVersion = 91; s.valueVersion = 17;
    s.cvOffsets = {0, 3, 7}; s.curveId = {401, 907}; s.rootPrim = {12, 44};
    s.rootUV = {{.2f,.3f},{.7f,.8f}}; s.rootT = {{1,0,0},{0,1,0}};
    s.rootB = {{0,1,0},{0,0,1}}; s.rootN = {{0,0,1},{1,0,0}}; UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{.25f,.75f}};
    s.px = zero ? VtFloatArray(7, 0.f) : VtFloatArray{0, 1, 2, 0, 0, 0, 0};
    s.py = VtFloatArray(7, 0.f); s.pz = zero ? VtFloatArray(7, 0.f) : VtFloatArray{0,0,0, 0,1,2,3};
    // Rest lengths differ by 10x, so using the rest plane for normalization
    // would fail the exact current-polyline target test below.
    s.rest = {GfVec3f(0,0,0),GfVec3f(0,10,0),GfVec3f(0,20,0),GfVec3f(0,0,0),GfVec3f(0,0,10),GfVec3f(0,0,20),GfVec3f(0,0,30)};
    s.width = VtFloatArray(7, 1.f); s.hairT = {0,.5f,1,0,.333f,.667f,1};
    UsdGenChunkDesc chunk; chunk.firstCurve = 0; chunk.curveCount = 2; chunk.liveCount = 2; chunk.firstCv = 0; chunk.cvCount = 0; chunk.tile = 3; chunk.surface = 1; s.chunks = {chunk};
    UsdGenPlane p; p.name = TfToken("privateVertex"); p.interpolation = TfToken("vertex"); p.type = TfToken("float"); p.arity = 1; p.f = {3,4,5,6,7,8,9}; s.extraCv = {p};
    UsdGenPlane q; q.name = TfToken("privateUniform"); q.interpolation = TfToken("uniform"); q.type = TfToken("int"); q.i = {18,19}; s.extraCurve = {maskPlane, q};
    i.geometry.alreadyDeformed = true; i.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline, UsdGenCurveLayoutWrap::Pinned);
    i.geometry.tiles = {{3,0,2,0,7}};
    uint32_t named[] = {73, 81}; i.additionalNamed.push_back({{"namedCurve", UsdGenDeviceValueType::UInt32, UsdGenDeviceDomain::Primitive, 2, 1, 4, true}, Bytes(named,2)});
    return i;
}
static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner> const& n, VulkanSourceGenerationCreateInfo i) {
    VkResult r = VK_SUCCESS; std::string why; auto u = VulkanSourceUpload::Create(std::move(i), &r, &why);
    if (!u || u->Submit() != SourceGenerationStatus::Submitted || !Prove(n) || u->Poll() != SourceGenerationStatus::Ready) return {};
    return u->TakeReady();
}
}

int main(int argc, char** argv) {
    CHECK(argc == 3); auto scaleCode = Code(argv[1]); auto setCode = Code(argv[2]); CHECK(!scaleCode.empty() && !setCode.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable); if (unavailable) return 77; CHECK(native);
    DeviceContext::CreateInfo ci; ci.instance=native->instance; ci.physicalDevice=native->physical; ci.device=native->device; ci.computeQueue=native->queue; ci.computeQueueFamily=native->family; ci.physicalIndex=native->physicalIndex; ci.resourceDeviceId=7931; ci.nativeLifetime=native; ci.timelineSemaphoreEnabled=true; ci.resources={size_t{32}<<20,0};
    auto context = DeviceContext::Create(ci); CHECK(context); auto pool = context->resources(); auto baseline = pool->Snapshot(); VkResult status = VK_SUCCESS;
    auto base = Upload(native, Source(context)); CHECK(base);
    auto scaleOnly = LengthScalePipeline::Create(context, scaleCode, &status); CHECK(scaleOnly && !scaleOnly->HasSet());
    CHECK(!scaleOnly->BeginSet(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), 2, 7, 2.f, &status) && status == VK_ERROR_FEATURE_NOT_PRESENT);
    auto pipeline = LengthScalePipeline::CreateWithSet(context, scaleCode, setCode, &status); CHECK(pipeline && pipeline->HasSet());
    auto run = [&](std::shared_ptr<const VulkanSourceGeneration> const& g, float target) {
        auto c = pipeline->BeginSet(g->PlaneOwner("points"), g->PlaneOwner("curveOffsets"), g->curveCount(), g->pointCount(), target, &status);
        if (!c || status != VK_SUCCESS || !Prove(native)) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{}, UINT32_MAX);
        uint32_t sem = UINT32_MAX; if (c->Poll(&sem) != VK_SUCCESS) return std::make_pair(std::unique_ptr<LengthScalePipeline::Candidate>{}, UINT32_MAX);
        return std::pair<std::unique_ptr<LengthScalePipeline::Candidate>,uint32_t>(std::move(c), sem);
    };
    Packet original; CHECK(Capture(native, context, base, &original));
    auto exact = run(base, 6.f); CHECK(exact.second == 0 && exact.first->succeeded());
    std::string reason; auto child = VulkanSourceGeneration::WithPoints(base, *exact.first, 18, &reason); CHECK(child);
    CHECK(child->PlaneOwner("points") == exact.first->output() && child->PlaneOwner("points") != base->PlaneOwner("points"));
    CHECK(child->topologyVersion() == base->topologyVersion() && child->valueVersion() == 18);
    for (auto const& plane : base->planes()) if (plane.metadata.name != "points")
        CHECK(child->PlaneOwner(plane.metadata.name) == base->PlaneOwner(plane.metadata.name));
    for (auto const& frame : base->sourceFrames()) {
        CHECK(child->SourceFrameOwner(frame.metadata.name) == base->SourceFrameOwner(frame.metadata.name));
        CHECK(!child->PlaneOwner(frame.metadata.name));
    }
    CHECK(child->SourceFrameOwner("sourceRootT") == base->SourceFrameOwner("sourceRootT")); CHECK(child->geometry().alreadyDeformed == base->geometry().alreadyDeformed); CHECK(UsdGenCurveLayoutTypeOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutTypeOf(base->geometry().layoutMetadata) && UsdGenCurveLayoutBasisOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutBasisOf(base->geometry().layoutMetadata) && UsdGenCurveLayoutWrapOf(child->geometry().layoutMetadata) == UsdGenCurveLayoutWrapOf(base->geometry().layoutMetadata)); CHECK(child->chunks().size() == base->chunks().size() && child->chunks()[0].firstCurve == base->chunks()[0].firstCurve && child->chunks()[0].curveCount == base->chunks()[0].curveCount);
    Packet changed; CHECK(Capture(native, context, child, &changed) && changed.size() == original.size());
    CHECK(original.count("stableIds") && original.count("rest") && original.count("sourceRootT"));
    for (auto const& plane : original) if (plane.first != "points") CHECK(changed.at(plane.first) == plane.second);
    float exactPoints[] = {0,0,0, 3,0,0, 6,0,0, 0,0,0, 0,0,2, 0,0,4, 0,0,6};
    CHECK(changed.at("points") == Bytes(exactPoints, 21));
    Packet unchanged; CHECK(Capture(native, context, base, &unchanged) && unchanged == original);
    auto zeroCurrent = Upload(native, Source(context, true)); CHECK(zeroCurrent); auto rejectedSemantic = run(zeroCurrent, 2.f); CHECK(rejectedSemantic.second != 0 && !rejectedSemantic.first->succeeded());
    CHECK(!rejectedSemantic.first->output() && !VulkanSourceGeneration::WithPoints(zeroCurrent, *rejectedSemantic.first, 20));
    auto zeroNoop = run(zeroCurrent, 0.f); CHECK(zeroNoop.first && zeroNoop.second == 0 && zeroNoop.first->succeeded());
    auto noOpChild = VulkanSourceGeneration::WithPoints(zeroCurrent, *zeroNoop.first, 21); CHECK(noOpChild);
    Packet zeroBefore, zeroAfter;
    CHECK(Capture(native, context, zeroCurrent, &zeroBefore) && Capture(native, context, noOpChild, &zeroAfter) && zeroBefore == zeroAfter);
    auto zeroTarget = run(base, 0.f); CHECK(zeroTarget.second == 0 && zeroTarget.first->succeeded()); auto collapsed = VulkanSourceGeneration::WithPoints(base, *zeroTarget.first, 19, &reason); CHECK(collapsed); Packet collapsedPacket; CHECK(Capture(native, context, collapsed, &collapsedPacket)); std::vector<float> collapsedPoints(21); std::memcpy(collapsedPoints.data(), collapsedPacket["points"].data(), collapsedPacket["points"].size()); CHECK(collapsedPoints[0] == collapsedPoints[3] && collapsedPoints[3] == collapsedPoints[6] && collapsedPoints[2] == collapsedPoints[5] && collapsedPoints[5] == collapsedPoints[8] && collapsedPoints[9] == collapsedPoints[12] && collapsedPoints[12] == collapsedPoints[18] && collapsedPoints[11] == collapsedPoints[14] && collapsedPoints[14] == collapsedPoints[20]);
    auto badHook = pipeline->BeginSet(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), 2, 7, 2.f, &status, [] { return false; }); CHECK(!badHook && status == VK_ERROR_OUT_OF_DEVICE_MEMORY);
    for (float v : {-1.f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) CHECK(!pipeline->BeginSet(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), 2, 7, v, &status));
    auto empty = pipeline->BeginSet({}, {}, 0, 0, 3.f, &status); CHECK(empty && empty->succeeded());
    CHECK(!pipeline->BeginSet(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), 2, 0, 1.f, &status));
    CHECK(!pipeline->BeginSet(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), UINT32_MAX, 7, 1.f, &status));
    exact.first.reset(); zeroTarget.first.reset(); rejectedSemantic.first.reset(); empty.reset();
    zeroNoop.first.reset(); noOpChild.reset();
    collapsed.reset(); pipeline.reset(); scaleOnly.reset(); child.reset(); zeroCurrent.reset(); base.reset();
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes);
    std::puts("Vulkan absolute Length set COW: PASS");
    return 0;
}
