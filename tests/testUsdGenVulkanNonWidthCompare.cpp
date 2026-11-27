// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/nonWidthComparePipeline.h"
#include "usdGenCurves/vulkan/lengthScalePipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenCurves/vulkan/widthBlendPipeline.h"
#include "usdGenCurves/vulkan/widthPipeline.h"
#include "vulkanNativeFixture.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "non-width compare failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> out(raw.size() / sizeof(uint32_t));
    std::memcpy(out.data(), raw.data(), raw.size()); return out;
}
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    vkResetFences(native->device, 1, &native->fence);
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}
static bool SameLedger(UsdGenExecutionResourceSnapshot const& a,
                       UsdGenExecutionResourceSnapshot const& b) {
    return a.limitBytes == b.limitBytes && a.headroomBytes == b.headroomBytes &&
        a.usableBytes == b.usableBytes && a.usedBytes == b.usedBytes &&
        a.byKind == b.byKind;
}
static VulkanSourceGenerationCreateInfo Source(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& b = info.source; b.totalCurves = 1; b.totalCvs = 3;
    b.topologyVersion = 4; b.valueVersion = 4; b.px = {0,1,2}; b.py = {0,1,0}; b.pz = {0,0,0};
    b.rest = {GfVec3f(0,0,1), GfVec3f(1,1,1), GfVec3f(2,0,1)};
    b.width = {1,2,3}; b.hairT = {0,.5f,1}; b.curveId = {7}; b.cvOffsets = {0,3};
    return info;
}
static VulkanSourceGenerationCreateInfo EmptySource(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    info.source.topologyVersion = 31; info.source.valueVersion = 31;
    return info;
}
int main(int argc, char** argv) {
    CHECK(argc == 5);
    auto flatCode = Code(argv[1]); auto lengthCode = Code(argv[2]); auto compareCode = Code(argv[3]);
    auto blendCode = Code(argv[4]);
    CHECK(!flatCode.empty() && !lengthCode.empty() && !compareCode.empty() && !blendCode.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable); if (unavailable) return 77; CHECK(native);
    DeviceContext::CreateInfo ci; ci.timelineSemaphoreEnabled = true; ci.instance=native->instance; ci.physicalDevice=native->physical;
    ci.device=native->device; ci.computeQueue=native->queue; ci.computeQueueFamily=native->family;
    ci.physicalIndex=native->physicalIndex; ci.resourceDeviceId=7016; ci.nativeLifetime=native; ci.resources={size_t{8}<<20,0};
    auto context = DeviceContext::Create(ci); CHECK(context);
    VkResult status = VK_SUCCESS; auto width = WidthPipeline::Create(context, flatCode, &status); CHECK(width);
    auto length = LengthScalePipeline::Create(context, lengthCode, &status); CHECK(length);
    auto compare = NonWidthComparePipeline::Create(context, compareCode, &status); CHECK(compare);
    auto blend = WidthBlendPipeline::Create(context, blendCode, &status); CHECK(blend);
    std::string reason; auto upload = VulkanSourceUpload::Create(Source(context), &status, &reason); CHECK(upload);
    CHECK(upload->Submit() == SourceGenerationStatus::Submitted && Prove(native)); CHECK(upload->Poll() == SourceGenerationStatus::Ready);
    auto base = upload->TakeReady(); CHECK(base);
    auto makeLength = [&](float factor, uint64_t version) {
        auto candidate = length->Begin(base->PlaneOwner("points"), base->PlaneOwner("curveOffsets"), 1, 3, factor, &status);
        if (!candidate || !Prove(native)) return std::shared_ptr<const VulkanSourceGeneration>{};
        uint32_t semantic = UINT32_MAX;
        if (candidate->Poll(&semantic) != VK_SUCCESS || semantic != 0 || !candidate->succeeded())
            return std::shared_ptr<const VulkanSourceGeneration>{};
        auto out = VulkanSourceGeneration::WithPoints(base, *candidate, version, &reason);
        return out;
    };
    auto sameLeft = makeLength(1.f, 5); auto sameRight = makeLength(1.f, 6); CHECK(sameLeft && sameRight);
    auto same = compare->Begin(sameLeft, sameRight, &status); CHECK(same);
    CHECK(same->leftOwner() == sameLeft && same->rightOwner() == sameRight);
    CHECK(Prove(native)); NonWidthComparePipeline::Candidate::Status compareStatus{};
    CHECK(same->Poll(&compareStatus) == VK_SUCCESS && compareStatus == NonWidthComparePipeline::Candidate::Status::Ready && same->succeeded());
    // Polling an already proved comparator must be stable: it remains the
    // ordered proof used by the WidthBlend admission below.
    CHECK(same->Poll(&compareStatus) == VK_SUCCESS &&
          compareStatus == NonWidthComparePipeline::Candidate::Status::Ready && same->succeeded());
    auto different = makeLength(2.f, 7); CHECK(different);
    auto unequal = compare->Begin(sameLeft, different, &status); CHECK(unequal);
    CHECK(Prove(native)); CHECK(unequal->Poll(&compareStatus) == VK_SUCCESS && !unequal->succeeded());

    // Independent Length outputs do not share a non-width identity.  An
    // ordered equality proof is therefore required to blend their widths.
    auto orderedBlend = blend->Begin(sameLeft->PlaneOwner("width"), sameRight->PlaneOwner("width"),
        3, .25f, &status); CHECK(orderedBlend && Prove(native));
    uint32_t blendStatus = UINT32_MAX;
    CHECK(orderedBlend->Poll(&blendStatus) == VK_SUCCESS && blendStatus == 0 && orderedBlend->succeeded());
    CHECK(!VulkanSourceGeneration::WithWidthBlend(sameLeft, sameRight, *orderedBlend, 10, &reason));
    auto joined = VulkanSourceGeneration::WithWidthBlend(sameLeft, sameRight, *orderedBlend, 10,
        &reason, same.get());
    CHECK(joined && joined->NonWidthIdentity() == sameLeft->NonWidthIdentity());
    for (auto const& plane : sameLeft->planes())
        if (plane.metadata.name != "width")
            CHECK(joined->PlaneOwner(plane.metadata.name) == sameLeft->PlaneOwner(plane.metadata.name));

    // Proofs bind the precise, authored left/right pair, not merely equal
    // payloads or compatible width-buffer owners.
    CHECK(!VulkanSourceGeneration::WithWidthBlend(sameRight, sameLeft, *orderedBlend, 10,
        &reason, same.get()));
    auto differentBlend = blend->Begin(sameLeft->PlaneOwner("width"), different->PlaneOwner("width"),
        3, .25f, &status); CHECK(differentBlend && Prove(native));
    CHECK(differentBlend->Poll(&blendStatus) == VK_SUCCESS && blendStatus == 0);
    CHECK(!VulkanSourceGeneration::WithWidthBlend(sameLeft, different, *differentBlend, 10,
        &reason, same.get()));

    // Width-only revisions preserve the data bundle but make the old ordered
    // proof stale: only the exact generations it proved may be joined.
    auto reviseWidth = [&](std::shared_ptr<const VulkanSourceGeneration> input, uint64_t version) {
        auto candidate = width->Begin(input->PlaneOwner("width"), 3, 1.f, 0, &status);
        if (!candidate || !Prove(native)) return std::shared_ptr<const VulkanSourceGeneration>{};
        uint32_t semantic = UINT32_MAX;
        if (candidate->Poll(&semantic) != VK_SUCCESS || semantic != 0 || !candidate->succeeded())
            return std::shared_ptr<const VulkanSourceGeneration>{};
        return VulkanSourceGeneration::WithWidth(input, *candidate, version, &reason);
    };
    auto revisedLeft = reviseWidth(sameLeft, 11); auto revisedRight = reviseWidth(sameRight, 12);
    CHECK(revisedLeft && revisedRight);
    auto staleBlend = blend->Begin(revisedLeft->PlaneOwner("width"), revisedRight->PlaneOwner("width"),
        3, .25f, &status); CHECK(staleBlend && Prove(native));
    CHECK(staleBlend->Poll(&blendStatus) == VK_SUCCESS && blendStatus == 0);
    CHECK(!VulkanSourceGeneration::WithWidthBlend(revisedLeft, revisedRight, *staleBlend, 13,
        &reason, same.get()));

    // A rejected pre-submit admission must leave the complete resource ledger
    // unchanged, not just its aggregate byte total.
    auto beforeRejectedBlend = context->resources()->Snapshot(); int rejectedHooks = 0;
    CHECK(!blend->Begin(sameLeft->PlaneOwner("width"), sameRight->PlaneOwner("width"), 3, .25f, &status,
                        [&] { ++rejectedHooks; return false; }));
    CHECK(rejectedHooks == 1 && SameLedger(beforeRejectedBlend, context->resources()->Snapshot()));
    auto beforeRejectedCompare = context->resources()->Snapshot();
    int rejectedCompareHooks = 0;
    CHECK(!compare->Begin(sameLeft, sameRight, &status,
                         [&] { ++rejectedCompareHooks; return false; }));
    CHECK(rejectedCompareHooks == 1 && SameLedger(beforeRejectedCompare, context->resources()->Snapshot()));

    // Width-only COW revisions retain the same non-width bundle.  The
    // compare must therefore prove equality even though the width payloads
    // differ.
    auto makeWidth = [&](float factor, uint64_t version) {
        auto candidate = width->Begin(base->PlaneOwner("width"), 3, factor, 0, &status);
        if (!candidate || !Prove(native)) return std::shared_ptr<const VulkanSourceGeneration>{};
        uint32_t semantic = UINT32_MAX;
        if (candidate->Poll(&semantic) != VK_SUCCESS || semantic != 0 || !candidate->succeeded())
            return std::shared_ptr<const VulkanSourceGeneration>{};
        return VulkanSourceGeneration::WithWidth(base, *candidate, version, &reason);
    };
    auto widthLeft = makeWidth(1.f, 8); auto widthRight = makeWidth(3.f, 9);
    CHECK(widthLeft && widthRight);
    auto widthOnly = compare->Begin(widthLeft, widthRight, &status); CHECK(widthOnly);
    CHECK(Prove(native)); CHECK(widthOnly->Poll(&compareStatus) == VK_SUCCESS &&
          compareStatus == NonWidthComparePipeline::Candidate::Status::Ready && widthOnly->succeeded());

    // Empty independently uploaded sources still have a valid equal
    // non-width contract and must reach scalar readback proof.
    auto emptyUploadA = VulkanSourceUpload::Create(EmptySource(context), &status, &reason);
    auto emptyUploadB = VulkanSourceUpload::Create(EmptySource(context), &status, &reason);
    CHECK(emptyUploadA && emptyUploadB);
    CHECK(emptyUploadA->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(emptyUploadA->Poll() == SourceGenerationStatus::Ready);
    CHECK(emptyUploadB->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(emptyUploadB->Poll() == SourceGenerationStatus::Ready);
    auto emptyA = emptyUploadA->TakeReady(); auto emptyB = emptyUploadB->TakeReady();
    CHECK(emptyA && emptyB);
    auto emptyEqual = compare->Begin(emptyA, emptyB, &status); CHECK(emptyEqual);
    CHECK(Prove(native)); CHECK(emptyEqual->Poll(&compareStatus) == VK_SUCCESS && emptyEqual->succeeded());
    // Empty sources retain the full core schema. Replacing their empty width
    // view must preserve that schema and non-width pairing in either direction.
    {
        auto emptyWidthCandidate = width->Begin(emptyB->PlaneOwner("width"), 0, 1.f, 0, &status);
        CHECK(emptyWidthCandidate);
        CHECK(emptyWidthCandidate->Poll() == VK_SUCCESS && emptyWidthCandidate->succeeded());
        auto emptyWithWidth = VulkanSourceGeneration::WithWidth(emptyB, *emptyWidthCandidate, 32, &reason);
        CHECK(emptyWithWidth && emptyA->planes().size() == 6 &&
              emptyWithWidth->planes().size() == emptyA->planes().size());
        CHECK(std::count_if(emptyWithWidth->planes().begin(), emptyWithWidth->planes().end(),
            [](auto const& plane) { return plane.metadata.semantic == UsdGenDeviceChannelSemantic::Widths &&
                plane.metadata.elementCount == 0 && plane.bytes == 0 && plane.buffer == VK_NULL_HANDLE; }) == 1);
        auto forward = compare->Begin(emptyA, emptyWithWidth, &status);
        CHECK(forward && Prove(native));
        CHECK(forward->Poll() == VK_SUCCESS && forward->succeeded());
        auto reverse = compare->Begin(emptyWithWidth, emptyA, &status);
        CHECK(reverse && Prove(native));
        CHECK(reverse->Poll() == VK_SUCCESS && reverse->succeeded());
    }

    // A generic channel with a non-word-aligned physical tail is part of the
    // non-width payload.  The comparator must include its final byte.
    auto tailInfoA = Source(context); auto tailInfoB = Source(context);
    VulkanSourceNamedChannel tailA;
    tailA.metadata = {"tail", UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Point,
                      3, 1, 5, true, UsdGenDeviceChannelSemantic::Generic};
    tailA.bytes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    auto tailB = tailA; tailB.bytes.back() = 15;
    tailInfoA.additionalNamed.push_back(tailA); tailInfoB.additionalNamed.push_back(tailB);
    auto tailUploadA = VulkanSourceUpload::Create(std::move(tailInfoA), &status, &reason);
    auto tailUploadB = VulkanSourceUpload::Create(std::move(tailInfoB), &status, &reason);
    CHECK(tailUploadA && tailUploadB);
    CHECK(tailUploadA->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(tailUploadA->Poll() == SourceGenerationStatus::Ready);
    CHECK(tailUploadB->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(tailUploadB->Poll() == SourceGenerationStatus::Ready);
    auto tailGenA = tailUploadA->TakeReady(); auto tailGenB = tailUploadB->TakeReady();
    CHECK(tailGenA && tailGenB);
    auto tailUnequal = compare->Begin(tailGenA, tailGenB, &status); CHECK(tailUnequal);
    CHECK(Prove(native)); CHECK(tailUnequal->Poll(&compareStatus) == VK_SUCCESS && !tailUnequal->succeeded());

    // Captured root T/B/N frames are private source-frame planes, but remain
    // part of the non-width contract and are compared by payload bytes.
    auto frameInfoA = Source(context); auto frameInfoB = Source(context);
    frameInfoA.source.rootT = {GfVec3f(1, 0, 0)};
    frameInfoA.source.rootB = {GfVec3f(0, 1, 0)};
    frameInfoA.source.rootN = {GfVec3f(0, 0, 1)};
    frameInfoB.source.rootT = frameInfoA.source.rootT;
    frameInfoB.source.rootB = frameInfoA.source.rootB;
    frameInfoB.source.rootN = frameInfoA.source.rootN;
    frameInfoB.source.rootT[0][0] = 2.f;
    auto frameUploadA = VulkanSourceUpload::Create(std::move(frameInfoA), &status, &reason);
    auto frameUploadB = VulkanSourceUpload::Create(std::move(frameInfoB), &status, &reason);
    CHECK(frameUploadA && frameUploadB);
    CHECK(frameUploadA->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(frameUploadA->Poll() == SourceGenerationStatus::Ready);
    CHECK(frameUploadB->Submit() == SourceGenerationStatus::Submitted && Prove(native));
    CHECK(frameUploadB->Poll() == SourceGenerationStatus::Ready);
    auto frameGenA = frameUploadA->TakeReady(); auto frameGenB = frameUploadB->TakeReady();
    CHECK(frameGenA && frameGenB);
    auto frameUnequal = compare->Begin(frameGenA, frameGenB, &status); CHECK(frameUnequal);
    CHECK(Prove(native)); CHECK(frameUnequal->Poll(&compareStatus) == VK_SUCCESS && !frameUnequal->succeeded());
    auto mismatchInfo = Source(context); mismatchInfo.geometry.alreadyDeformed = true;
    auto mismatchUpload = VulkanSourceUpload::Create(mismatchInfo, &status, &reason); CHECK(mismatchUpload);
    CHECK(mismatchUpload->Submit() == SourceGenerationStatus::Submitted && Prove(native)); CHECK(mismatchUpload->Poll() == SourceGenerationStatus::Ready);
    auto mismatch = mismatchUpload->TakeReady(); CHECK(mismatch);
    CHECK(!compare->Begin(base, mismatch, &status));
    CHECK(context->resources()->Snapshot().usedBytes > 0);
    same.reset(); unequal.reset(); orderedBlend.reset(); differentBlend.reset(); staleBlend.reset();
    joined.reset(); revisedLeft.reset(); revisedRight.reset(); widthOnly.reset(); emptyEqual.reset(); tailUnequal.reset(); frameUnequal.reset();
    different.reset(); widthLeft.reset(); widthRight.reset(); sameLeft.reset(); sameRight.reset();
    mismatch.reset(); base.reset(); emptyA.reset(); emptyB.reset(); tailGenA.reset(); tailGenB.reset();
    frameGenA.reset(); frameGenB.reset(); mismatchUpload.reset(); upload.reset(); emptyUploadA.reset();
    emptyUploadB.reset(); tailUploadA.reset(); tailUploadB.reset(); frameUploadA.reset(); frameUploadB.reset();
    CHECK(context->resources()->Snapshot().usedBytes == 0);
    return 0;
}
