// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGen/curveLoader.h"
#include "usdGenUsd/lower.h"
#include "usdGen/vulkan/lengthScalePipeline.h"
#include "usdGen/vulkan/nonWidthComparePipeline.h"
#include "usdGen/vulkan/sourceGeneration.h"
#include "usdGen/vulkan/widthBlendPipeline.h"
#include "usdGen/vulkan/widthPipeline.h"
#include "vulkanReadbackFixture.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "root-frame COW failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

namespace {

using BytesByName = std::map<std::string, std::vector<uint8_t>>;

static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());
    return code;
}

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static UsdGenGraphDesc RootedC3() {
    UsdGenGraphDesc desc; desc.description = SdfPath("/Groom");
    UsdGenSurfaceDesc scalp; scalp.path = SdfPath("/Scalp");
    scalp.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}}; scalp.points = scalp.restPoints;
    scalp.faceVertexCounts = {3}; scalp.faceVertexIndices = {0,1,2};
    scalp.uv = {{0,0}, {1,0}, {0,1}}; desc.surfaces = {scalp};
    UsdGenCurveSetDesc c3; c3.path = SdfPath("/Groom/C3"); c3.role = UsdGenLane::Chunked;
    c3.curveRole = TfToken("hair"); c3.type = TfToken("cubic"); c3.basis = TfToken("bspline");
    c3.wrap = TfToken("pinned"); c3.curveVertexCounts = {3,3};
    c3.points = {{.2f,.2f,0}, {.2f,.2f,1}, {.2f,.2f,2}, {.7f,.1f,0}, {.7f,.1f,1}, {.7f,.1f,2}};
    c3.rest = c3.points; c3.widths = VtFloatArray(6, .02f); c3.curveId = {42,7};
    c3.skinPrim = {0,0}; c3.skinPrimUv = {{.2f,.2f}, {.7f,.1f}}; desc.curveSets = {c3};
    UsdGenNodeDesc source; source.path = SdfPath("/Groom/Ops/source");
    source.type = TfToken("UsdGenCurveSource"); source.curves = {c3.path}; source.surfaces = {scalp.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("rebind"), VtValue(TfToken("never")), false}};
    desc.nodes = {source}; desc.terminal = source.path;
    return desc;
}

// Keep the capture boundary shared with the production CurveSource: these
// tests deliberately do not manufacture a post-loader Vulkan input.
static bool LoadRootedC3(UsdGenGraphDesc const& desc, UsdGenCurveBuffer* out) {
    usdGen::UsdGenGraphSpec const lowered = usdGenUsd::Lower(desc);
    UsdGenParamView params{&lowered, &lowered.nodes[0]};
    UsdGenCaptureContext capture; capture.desc = &lowered; capture.params = &params;
    UsdGenDiagnostics diagnostics;
    return UsdGenCurveLoader::Load(capture, UsdGenCurveSetInputs(lowered)[0], out, &diagnostics) &&
        !diagnostics.HasErrors();
}

static bool Near(float value, float expected, float tolerance = 2.e-6f) {
    return std::abs(value - expected) <= tolerance;
}

static bool NearVec(GfVec3f const& value, GfVec3f const& expected) {
    return Near(value[0], expected[0]) && Near(value[1], expected[1]) &&
        Near(value[2], expected[2]);
}

static bool ReadFrames(std::shared_ptr<NativeOwner> const& native,
                       std::shared_ptr<DeviceContext> const& context,
                       std::shared_ptr<const VulkanSourceGeneration> const& generation,
                       BytesByName* result) {
    result->clear();
    for (auto const& frame : generation->sourceFrames()) {
        std::vector<uint8_t> bytes;
        if (!ReadVulkanBytes(native, context, frame.buffer, frame.bytes, generation, &bytes) ||
            !result->emplace(frame.metadata.name, std::move(bytes)).second) return false;
    }
    return true;
}

static bool SameFrames(std::shared_ptr<NativeOwner> const& native,
                       std::shared_ptr<DeviceContext> const& context,
                       std::shared_ptr<const VulkanSourceGeneration> const& a,
                       std::shared_ptr<const VulkanSourceGeneration> const& b) {
    BytesByName left, right;
    return ReadFrames(native, context, a, &left) && ReadFrames(native, context, b, &right) && left == right;
}

static bool SamePublicPlanes(std::shared_ptr<NativeOwner> const& native,
                             std::shared_ptr<DeviceContext> const& context,
                             std::shared_ptr<const VulkanSourceGeneration> const& a,
                             std::shared_ptr<const VulkanSourceGeneration> const& b) {
    auto read = [&](std::shared_ptr<const VulkanSourceGeneration> const& generation,
                    BytesByName* result) {
        result->clear();
        for (auto const& plane : generation->planes()) {
            std::vector<uint8_t> bytes;
            if (!ReadVulkanBytes(native, context, plane.buffer, plane.bytes, generation, &bytes) ||
                !result->emplace(plane.metadata.name, std::move(bytes)).second) return false;
        }
        return true;
    };
    BytesByName left, right;
    return read(a, &left) && read(b, &right) && left == right;
}

} // namespace

int main(int argc, char** argv) {
    // width.spv lengthScale.spv widthBlend.spv nonWidthCompare.spv
    CHECK(argc == 5);
    auto widthCode = Code(argv[1]), lengthCode = Code(argv[2]);
    auto blendCode = Code(argv[3]), compareCode = Code(argv[4]);
    CHECK(!widthCode.empty() && !lengthCode.empty() && !blendCode.empty() && !compareCode.empty());

    // Root frames are derived, rather than authored, from a genuinely bound
    // surface.  The row-vector relative transform is surface * inverse(source):
    // it rotates +90 degrees about Z, has non-uniform scale, and translates.
    auto rootedDesc = RootedC3();
    rootedDesc.curveSets[0].rootFrame.clear();
    auto const sourcePoints = rootedDesc.curveSets[0].points;
    auto const sourceRest = rootedDesc.curveSets[0].rest;
    GfMatrix4d relative(1.0);
    relative[0][0] = 0.; relative[0][1] = 2.;
    relative[1][0] = -3.; relative[1][1] = 0.;
    relative[2][2] = .5;
    relative[3][0] = 5.; relative[3][1] = -2.; relative[3][2] = 7.;
    GfMatrix4d sourceWorld(1.0);
    sourceWorld[0][0] = 0.; sourceWorld[0][1] = -1.;
    sourceWorld[1][0] = 1.; sourceWorld[1][1] = 0.;
    sourceWorld[3][0] = 11.; sourceWorld[3][1] = -13.; sourceWorld[3][2] = 17.;
    rootedDesc.curveSets[0].worldMatrix = sourceWorld;
    auto& affineSurface = rootedDesc.surfaces[0].worldMatrix;
    affineSurface = relative * sourceWorld;
    UsdGenCurveBuffer rooted;
    CHECK(LoadRootedC3(rootedDesc, &rooted));
    CHECK(rooted.totalCurves == 2 && rooted.totalCvs == 6 && rooted.rootPrim.size() == 2 &&
          rooted.rootUV.size() == 2 && rooted.rootT.size() == 2 && rooted.rootB.size() == 2 && rooted.rootN.size() == 2);
    // Canonical stable-id order is 7 then 42.  Inverse-transpose keeps the
    // normal at +Z; Gram-Schmidt plus N x T reconstructs a unit basis.
    for (size_t curve = 0; curve != rooted.totalCurves; ++curve)
        CHECK(NearVec(rooted.rootT[curve], {0,1,0}) &&
              NearVec(rooted.rootB[curve], {-1,0,0}) &&
              NearVec(rooted.rootN[curve], {0,0,1}));
    CHECK(rooted.rootUV == VtVec2fArray({GfVec2f(.7f,.1f), GfVec2f(.2f,.2f)}));
    CHECK(NearVec(GfVec3f(rooted.px[0], rooted.py[0], rooted.pz[0]), {.7f,.1f,0}) &&
          rooted.rest[0] == GfVec3f(.7f,.1f,0) &&
          NearVec(GfVec3f(rooted.px[3], rooted.py[3], rooted.pz[3]), {.2f,.2f,0}) &&
          rooted.rest[3] == GfVec3f(.2f,.2f,0) &&
          rootedDesc.curveSets[0].points == sourcePoints && rootedDesc.curveSets[0].rest == sourceRest);

    // Equal object transforms cancel in the same relative expression.  This
    // is a second derived-frame load, not an authored-rootFrame shortcut.
    auto equalRelative = rootedDesc;
    equalRelative.curveSets[0].worldMatrix = affineSurface;
    UsdGenCurveBuffer equalRooted;
    CHECK(equalRelative.curveSets[0].rootFrame.empty() && LoadRootedC3(equalRelative, &equalRooted));
    for (size_t curve = 0; curve != equalRooted.totalCurves; ++curve)
        CHECK(NearVec(equalRooted.rootT[curve], {1,0,0}) &&
              NearVec(equalRooted.rootB[curve], {0,1,0}) &&
              NearVec(equalRooted.rootN[curve], {0,0,1}));
    CHECK(NearVec(GfVec3f(equalRooted.px[0], equalRooted.py[0], equalRooted.pz[0]), {.7f,.1f,0}) &&
          equalRooted.rest[0] == GfVec3f(.7f,.1f,0));

    bool unavailable = false; auto native = CreateNative(&unavailable); if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci; ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue; ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex; ci.resourceDeviceId = 7027; ci.nativeLifetime = native;
    ci.resources = {size_t{8} << 20, 0}; auto context = DeviceContext::Create(ci); CHECK(context);
    VkResult status = VK_SUCCESS; std::string reason;
    auto upload = [&](UsdGenCurveBuffer const& source) {
        VulkanSourceGenerationCreateInfo info; info.context = context; info.source = source;
        auto pending = VulkanSourceUpload::Create(std::move(info), &status, &reason);
        if (!pending || pending->Submit() != SourceGenerationStatus::Submitted || !Prove(native) ||
            pending->Poll() != SourceGenerationStatus::Ready) return std::shared_ptr<const VulkanSourceGeneration>{};
        return pending->TakeReady();
    };
    auto base = upload(rooted); CHECK(base);
    CHECK(base->PlaneOwner("rootPrim") && base->PlaneOwner("rootUV") && base->sourceFrames().size() == 3);
    for (auto const& plane : base->planes())
        CHECK(plane.metadata.name != "sourceRootT" && plane.metadata.name != "sourceRootB" && plane.metadata.name != "sourceRootN");
    BytesByName originalFrames; CHECK(ReadFrames(native, context, base, &originalFrames));
    CHECK(originalFrames.size() == 3 &&
          originalFrames.at("sourceRootT") == std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(rooted.rootT.cdata()), reinterpret_cast<uint8_t const*>(rooted.rootT.cdata()) + rooted.rootT.size() * sizeof(GfVec3f)) &&
          originalFrames.at("sourceRootB") == std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(rooted.rootB.cdata()), reinterpret_cast<uint8_t const*>(rooted.rootB.cdata()) + rooted.rootB.size() * sizeof(GfVec3f)) &&
          originalFrames.at("sourceRootN") == std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(rooted.rootN.cdata()), reinterpret_cast<uint8_t const*>(rooted.rootN.cdata()) + rooted.rootN.size() * sizeof(GfVec3f)));
    // The private GPU frame payload is the derived affine basis above, not a
    // CPU operator result or an authored rootFrame.
    CHECK(originalFrames.at("sourceRootT") == std::vector<uint8_t>(
              reinterpret_cast<uint8_t const*>(rooted.rootT.cdata()),
              reinterpret_cast<uint8_t const*>(rooted.rootT.cdata()) + rooted.rootT.size() * sizeof(GfVec3f)) &&
          NearVec(rooted.rootT[0], {0,1,0}) && NearVec(rooted.rootB[0], {-1,0,0}) &&
          NearVec(rooted.rootN[0], {0,0,1}));

    auto width = WidthPipeline::Create(context, widthCode, &status);
    auto length = LengthScalePipeline::Create(context, lengthCode, &status);
    auto blend = WidthBlendPipeline::Create(context, blendCode, &status);
    auto compare = NonWidthComparePipeline::Create(context, compareCode, &status);
    CHECK(width && length && blend && compare);
    auto makeWidth = [&](float factor, uint64_t version) {
        auto candidate = width->Begin(base->PlaneOwner("width"), base->pointCount(), factor, 0, &status);
        if (!candidate || !Prove(native)) return std::shared_ptr<const VulkanSourceGeneration>{};
        uint32_t semantic = UINT32_MAX;
        if (candidate->Poll(&semantic) != VK_SUCCESS || semantic || !candidate->succeeded()) return std::shared_ptr<const VulkanSourceGeneration>{};
        if (candidate->inputOwner() != base->PlaneOwner("width") ||
            candidate->context() != base->context()) return std::shared_ptr<const VulkanSourceGeneration>{};
        return VulkanSourceGeneration::WithWidth(base, *candidate, version, &reason);
    };
    auto widthLeft = makeWidth(1.f, 11), widthRight = makeWidth(2.f, 12); CHECK(widthLeft && widthRight);
    for (auto const& child : {widthLeft, widthRight}) {
        CHECK(child->PlaneOwner("rootPrim") == base->PlaneOwner("rootPrim") &&
              child->PlaneOwner("rootUV") == base->PlaneOwner("rootUV") &&
              &child->sourceFrames() == &base->sourceFrames() && SameFrames(native, context, base, child));
    }
    auto pointRun = length->Begin(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"),
        base->curveCount(), base->pointCount(), 1.f, &status);
    CHECK(pointRun && Prove(native)); uint32_t pointStatus = UINT32_MAX;
    CHECK(pointRun->Poll(&pointStatus) == VK_SUCCESS && pointStatus == 0 && pointRun->succeeded());
    CHECK(pointRun->inputOwner() == base->PlaneOwner("points") &&
          pointRun->offsetsOwner() == base->PlaneOwner("curveOffsets") && pointRun->context() == base->context());
    auto points = VulkanSourceGeneration::WithPoints(base, *pointRun, 13, &reason); CHECK(points);
    CHECK(points->PlaneOwner("points") == pointRun->output() && points->PlaneOwner("points") != base->PlaneOwner("points") &&
          points->PlaneOwner("rootPrim") == base->PlaneOwner("rootPrim") &&
          points->PlaneOwner("rootUV") == base->PlaneOwner("rootUV") &&
          &points->sourceFrames() == &base->sourceFrames() && SameFrames(native, context, base, points));

    // An equality proof binds its exact rooted predecessors; it is then the
    // admission witness for a WidthBlend COW result retaining all frame data.
    auto proof = compare->Begin(widthLeft, widthRight, &status); CHECK(proof);
    CHECK(proof->leftOwner() == widthLeft && proof->rightOwner() == widthRight && Prove(native));
    NonWidthComparePipeline::Candidate::Status compareStatus{};
    CHECK(proof->Poll(&compareStatus) == VK_SUCCESS && compareStatus == NonWidthComparePipeline::Candidate::Status::Ready && proof->succeeded());
    auto blendRun = blend->Begin(widthLeft->PlaneOwner("width"), widthRight->PlaneOwner("width"),
        base->pointCount(), .25f, &status); CHECK(blendRun && Prove(native));
    uint32_t blendStatus = UINT32_MAX;
    CHECK(blendRun->Poll(&blendStatus) == VK_SUCCESS && blendStatus == 0 && blendRun->succeeded());
    CHECK(blendRun->leftOwner() == widthLeft->PlaneOwner("width") &&
          blendRun->rightOwner() == widthRight->PlaneOwner("width") && blendRun->context() == base->context());
    auto blended = VulkanSourceGeneration::WithWidthBlend(widthLeft, widthRight, *blendRun, 14, &reason, proof.get());
    CHECK(blended && blended->PlaneOwner("rootPrim") == base->PlaneOwner("rootPrim") &&
          blended->PlaneOwner("rootUV") == base->PlaneOwner("rootUV") &&
          &blended->sourceFrames() == &base->sourceFrames() && SameFrames(native, context, base, blended));
    CHECK(!VulkanSourceGeneration::WithWidthBlend(widthRight, widthLeft, *blendRun, 14, &reason, proof.get()));

    // A separately uploaded rooted packet with one changed private frame must
    // fail GPU non-width equality, even though private frames are not planes().
    auto changedRooted = rooted; changedRooted.rootT[0][0] += 1.f;
    auto changed = upload(changedRooted); CHECK(changed && changed->sourceFrames().size() == 3);
    CHECK(changed->PlaneOwner("rootPrim") != base->PlaneOwner("rootPrim"));
    // Only a derived private frame changed: every public geometry plane is
    // byte-identical, but GPU non-width comparison must still reject it.
    CHECK(changedRooted.px == rooted.px && changedRooted.py == rooted.py &&
          changedRooted.pz == rooted.pz && changedRooted.rest == rooted.rest &&
          changedRooted.rootPrim == rooted.rootPrim && changedRooted.rootUV == rooted.rootUV &&
          SamePublicPlanes(native, context, base, changed));
    BytesByName changedFrames; CHECK(ReadFrames(native, context, changed, &changedFrames) && changedFrames != originalFrames);
    auto unequal = compare->Begin(base, changed, &status); CHECK(unequal);
    CHECK(unequal->leftOwner() == base && unequal->rightOwner() == changed && Prove(native));
    CHECK(unequal->Poll(&compareStatus) == VK_SUCCESS && compareStatus == NonWidthComparePipeline::Candidate::Status::Ready && !unequal->succeeded());

    auto noFrames = rooted; noFrames.rootT.clear(); noFrames.rootB.clear(); noFrames.rootN.clear();
    auto absent = upload(noFrames); CHECK(absent && absent->sourceFrames().empty());
    // Frame absence is an explicitly different packet shape and is rejected
    // at metadata admission, before any private-frame vector is traversed.
    CHECK(!compare->Begin(base, absent, &status));
    return 0;
}
