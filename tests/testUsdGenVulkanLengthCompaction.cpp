// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "vulkanReadbackFixture.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan length compaction failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

namespace {
using Packet = std::map<std::string, std::vector<uint8_t>>;

template <class T> std::vector<uint8_t> Bytes(T const* values, size_t count) {
    std::vector<uint8_t> result(count * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), values, result.size());
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

static bool Capture(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context,
    std::shared_ptr<const VulkanSourceGeneration> const& generation, Packet* packet) {
    packet->clear();
    auto add = [&](VulkanSourceGeneration::PlaneView const& plane) {
        std::vector<uint8_t> bytes;
        return ReadVulkanBytes(native, context, plane.buffer, plane.bytes, generation, &bytes) &&
            packet->emplace(plane.metadata.name, std::move(bytes)).second;
    };
    for (auto const& plane : generation->planes()) if (!add(plane)) return false;
    for (auto const& plane : generation->sourceFrames()) if (!add(plane)) return false;
    return true;
}

static size_t ValueBytes(UsdGenDeviceChannelMetadata const& metadata) {
    size_t scalar = metadata.type == UsdGenDeviceValueType::Float32 ||
                            metadata.type == UsdGenDeviceValueType::Int32 ||
                            metadata.type == UsdGenDeviceValueType::UInt32 ? 4 :
                    metadata.type == UsdGenDeviceValueType::UInt64 ? 8 : 4;
    if (metadata.type == UsdGenDeviceValueType::Float32x2) return 8;
    if (metadata.type == UsdGenDeviceValueType::Float32x3) return 12;
    if (metadata.type == UsdGenDeviceValueType::Float32x4) return 16;
    return scalar * metadata.arity;
}

static std::vector<uint8_t> Gather(std::vector<uint8_t> const& source,
    UsdGenDeviceChannelMetadata const& metadata, std::vector<uint32_t> const& indices) {
    size_t const valueBytes = ValueBytes(metadata);
    uint32_t const stride = metadata.strideBytes ? metadata.strideBytes : uint32_t(valueBytes);
    std::vector<uint8_t> result;
    if (indices.empty()) return result;
    result.resize(size_t(stride) * (indices.size() - 1) + valueBytes);
    for (size_t i = 0; i != indices.size(); ++i) {
        size_t const copyBytes = i + 1 < indices.size() ? stride : valueBytes;
        std::memcpy(result.data() + size_t(i) * stride,
            source.data() + size_t(indices[i]) * stride, copyBytes);
    }
    return result;
}

static VulkanSourceGenerationCreateInfo Fixture(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& source = info.source;
    source.totalCurves = 4; source.totalCvs = 11; source.topologyVersion = 17; source.valueVersion = 23;
    // Lengths are exactly 1, 0, 2, and 3.  The equality case is deliberately
    // curve zero: threshold 1 must retain it, preserving source order 0,2,3.
    source.px = {0,1, 4,4,4, 0,2, 0,0,0,0};
    source.py = VtFloatArray(11, 0); source.pz = {0,0, 0,0,0, 0,0, 0,1,2,3};
    source.rest.reserve(11); source.width.reserve(11); source.hairT.reserve(11);
    for (uint32_t i = 0; i != 11; ++i) {
        source.rest.push_back(GfVec3f(float(i), float(i + 10), float(i + 20)));
        source.width.push_back(float(i) + .25f); source.hairT.push_back(float(i) / 10.f);
    }
    source.cvOffsets = {0,2,5,7,11}; source.curveId = {100,101,102,103};
    source.rootPrim = {10,11,12,13};
    source.rootUV = {{.1f,.2f},{.3f,.4f},{.5f,.6f},{.7f,.8f}};
    source.rootT = {{1,0,0},{0,1,0},{0,0,1},{1,1,0}};
    source.rootB = {{0,1,0},{0,0,1},{1,0,0},{0,1,1}};
    source.rootN = {{0,0,1},{1,0,0},{0,1,0},{1,0,1}};
    UsdGenPlane maskPlane; maskPlane.name = TfToken("curveMask"); maskPlane.interpolation = TfToken("uniform"); maskPlane.type = TfToken("float"); maskPlane.f = {{.1f,.2f,.3f,.4f}};
    UsdGenPlane point; point.name = TfToken("pointExtra"); point.interpolation = TfToken("vertex");
    point.type = TfToken("float"); point.arity = 2;
    for (uint32_t i = 0; i != 22; ++i) point.f.push_back(float(i) + .125f);
    source.extraCv = {point};
    UsdGenPlane primitive; primitive.name = TfToken("primitiveExtra"); primitive.interpolation = TfToken("uniform");
    primitive.type = TfToken("int"); primitive.arity = 1; primitive.i = {30,31,32,33};
    source.extraCurve = {maskPlane, primitive};
    source.chunks.resize(2);
    source.chunks[0].firstCurve = 0; source.chunks[0].curveCount = 2; source.chunks[0].liveCount = 2;
    source.chunks[0].firstCv = 0; source.chunks[0].cvCount = 0; source.chunks[0].tile = 7; source.chunks[0].surface = 1;
    source.chunks[1].firstCurve = 2; source.chunks[1].curveCount = 2; source.chunks[1].liveCount = 2;
    source.chunks[1].firstCv = 5; source.chunks[1].cvCount = 0; source.chunks[1].tile = 9; source.chunks[1].surface = 1;
    info.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline, UsdGenCurveLayoutWrap::Pinned);
    info.geometry.alreadyDeformed = true;
    info.geometry.tiles = {{7,0,2,0,5}, {9,2,2,5,6}};
    float groom[] = {9.5f, 7.25f};
    info.additionalNamed.push_back({{"GroomNamed", UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Groom, 1, 2, 8, true}, Bytes(groom, 2)});
    uint32_t primitiveNamed[] = {400,401,402,403};
    info.additionalNamed.push_back({{"PrimitiveNamed", UsdGenDeviceValueType::UInt32, UsdGenDeviceDomain::Primitive, 4, 1, 4, true}, Bytes(primitiveNamed, 4)});
    // 5-byte records prove byte-addressed scatter does not overlap odd strides.
    std::vector<uint8_t> odd(5 * 10 + 4);
    for (size_t i = 0; i != odd.size(); ++i) odd[i] = uint8_t(17 + i);
    info.additionalNamed.push_back({{"PointNamed", UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Point, 11, 1, 5, true}, std::move(odd)});
    return info;
}

// Deliberately crosses the 64-lane workgroup boundary several times.  The
// zero-length entries and varying spans also ensure that the independent
// curve/point prefixes cannot accidentally be inferred from a uniform stride.
static VulkanSourceGenerationCreateInfo LargeFixture(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& s = info.source; constexpr uint32_t curves = 513;
    s.totalCurves = curves; s.topologyVersion = 117; s.valueVersion = 123;
    s.cvOffsets.resize(curves + 1, 0); s.curveId.resize(curves); s.rootPrim.resize(curves);
    s.rootUV.resize(curves); s.rootT.resize(curves); s.rootB.resize(curves); s.rootN.resize(curves);
    for (uint32_t c = 0; c != curves; ++c) {
        uint32_t const n = c % 5; s.cvOffsets[c + 1] = s.cvOffsets[c] + n;
        s.curveId[c] = 10000 + c; s.rootPrim[c] = int(c);
        s.rootUV[c] = {float(c), float(c + 1)};
        s.rootT[c] = {1,0,0}; s.rootB[c] = {0,1,0}; s.rootN[c] = {0,0,1};
    }
    uint32_t const points = s.cvOffsets.back();
    s.totalCvs = points;
    s.px.reserve(points); s.py = VtFloatArray(points, 0.f); s.pz = VtFloatArray(points, 0.f);
    s.rest.reserve(points); s.width.reserve(points); s.hairT.reserve(points);
    for (uint32_t c = 0; c != curves; ++c) for (uint32_t j = 0; j != c % 5; ++j) {
        s.px.push_back(float(j)); s.rest.push_back(GfVec3f(float(c), float(j), 0));
        s.width.push_back(float(c) + .5f); s.hairT.push_back(float(j));
    }
    for (uint32_t first : {0u,171u,342u}) {
        uint32_t const last = first == 342 ? curves : first + 171;
        UsdGenChunkDesc chunk; chunk.firstCurve = first; chunk.curveCount = last - first;
        chunk.liveCount = chunk.curveCount; chunk.firstCv = s.cvOffsets[first]; chunk.cvCount = 0;
        chunk.tile = 20 + first / 171; chunk.surface = 3; s.chunks.push_back(chunk);
        info.geometry.tiles.push_back({chunk.tile, first, chunk.curveCount, chunk.firstCv,
                                       s.cvOffsets[last] - chunk.firstCv});
    }
    info.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline,
                                   UsdGenCurveLayoutWrap::Pinned);
    info.geometry.alreadyDeformed = true;
    float groom[] = {3.25f};
    info.additionalNamed.push_back({{"LargeGroom", UsdGenDeviceValueType::Float32,
                                     UsdGenDeviceDomain::Groom, 1, 1, 4, true}, Bytes(groom, 1)});
    return info;
}

static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner> const& native,
    VulkanSourceGenerationCreateInfo info) {
    VkResult status = VK_SUCCESS; std::string reason;
    auto upload = VulkanSourceUpload::Create(std::move(info), &status, &reason);
    if (!upload || upload->Submit() != SourceGenerationStatus::Submitted || !Prove(native) ||
        upload->Poll() != SourceGenerationStatus::Ready) return {};
    return upload->TakeReady();
}

static bool CompletePacket(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context, std::shared_ptr<const VulkanSourceGeneration> const& value,
    Packet const& input, std::vector<uint32_t> const& curves, std::vector<uint32_t> const& points) {
    Packet got;
    if (!Capture(native, context, value, &got)) return false;
    if (got.size() != input.size() || !got.count("sourceRootT") || !got.count("sourceRootB") || !got.count("sourceRootN")) return false;
    std::vector<uint32_t> offsets{0};
    for (uint32_t curve : curves) {
        static uint32_t const ends[] = {2,5,7,11};
        static uint32_t const begins[] = {0,2,5,7};
        offsets.push_back(offsets.back() + ends[curve] - begins[curve]);
    }
    for (auto const& plane : value->planes()) {
        auto const inputIt = input.find(plane.metadata.name); if (inputIt == input.end()) return false;
        if (plane.metadata.name == "curveOffsets") {
            if (got.at(plane.metadata.name) != Bytes(offsets.data(), offsets.size())) return false;
        } else if (plane.metadata.domain == UsdGenDeviceDomain::Point) {
            if (got.at(plane.metadata.name) != Gather(inputIt->second, plane.metadata, points)) return false;
        } else if (plane.metadata.domain == UsdGenDeviceDomain::Primitive) {
            if (got.at(plane.metadata.name) != Gather(inputIt->second, plane.metadata, curves)) return false;
        } else if (got.at(plane.metadata.name) != inputIt->second) return false;
    }
    for (auto const& frame : value->sourceFrames()) {
        auto const inputIt = input.find(frame.metadata.name); if (inputIt == input.end() ||
            got.at(frame.metadata.name) != Gather(inputIt->second, frame.metadata, curves)) return false;
    }
    return true;
}

static bool CompleteRaggedPacket(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<DeviceContext> const& context, std::shared_ptr<const VulkanSourceGeneration> const& value,
    Packet const& input, std::vector<uint32_t> const& curves, std::vector<uint32_t> const& points,
    std::vector<uint32_t> const& sourceOffsets) {
    Packet got; if (!Capture(native, context, value, &got) || got.size() != input.size()) return false;
    std::vector<uint32_t> offsets{0};
    for (uint32_t c : curves) offsets.push_back(offsets.back() + sourceOffsets[c + 1] - sourceOffsets[c]);
    for (auto const& plane : value->planes()) {
        auto it = input.find(plane.metadata.name); if (it == input.end()) return false;
        if (plane.metadata.name == "curveOffsets") {
            if (got.at(plane.metadata.name) != Bytes(offsets.data(), offsets.size())) return false;
        } else if (plane.metadata.domain == UsdGenDeviceDomain::Point) {
            if (got.at(plane.metadata.name) != Gather(it->second, plane.metadata, points)) return false;
        } else if (plane.metadata.domain == UsdGenDeviceDomain::Primitive) {
            if (got.at(plane.metadata.name) != Gather(it->second, plane.metadata, curves)) return false;
        } else if (got.at(plane.metadata.name) != it->second) return false;
    }
    for (auto const& frame : value->sourceFrames()) {
        auto it = input.find(frame.metadata.name); if (it == input.end() || got.at(frame.metadata.name) != Gather(it->second, frame.metadata, curves)) return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    CHECK(argc == 2); auto code = Code(argv[1]); CHECK(!code.empty());
    bool unavailable = false; auto native = CreateTimelineNative(&unavailable); if (unavailable) return 77;
    CHECK(native);
    DeviceContext::CreateInfo ci; ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue; ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex; ci.resourceDeviceId = 7041; ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true; ci.resources = {size_t{16} << 20, 0};
    auto context = DeviceContext::Create(ci); CHECK(context);
    auto base = Upload(native, Fixture(context)); CHECK(base && base->curveCount() == 4 && base->pointCount() == 11);
    Packet original; CHECK(Capture(native, context, base, &original));
    CHECK(original.count("PointNamed") && original.count("PrimitiveNamed") && original.count("GroomNamed") && base->sourceFrames().size() == 3);
    VkResult status = VK_SUCCESS; auto pipeline = LengthCompactionPipeline::Create(context, code, &status); CHECK(pipeline && status == VK_SUCCESS);
    CHECK(!pipeline->Begin(base, -1.f, &status));
    CHECK(!pipeline->Begin(base, std::numeric_limits<float>::infinity(), &status));
    CHECK(!pipeline->Begin(base, std::numeric_limits<float>::quiet_NaN(), &status));
    CHECK(!base->PlaneOwner("sourceRootT") && base->SourceFrameOwner("sourceRootT"));
    bool admissionCalled = false;
    CHECK(!pipeline->Begin(base, 1.f, &status, [&] { admissionCalled = true; return false; }) && admissionCalled);
    auto compact = [&](std::shared_ptr<const VulkanSourceGeneration> input, float threshold, uint64_t version) {
        auto candidate = pipeline->Begin(input, threshold, &status); if (!candidate || !Prove(native)) return std::shared_ptr<const VulkanSourceGeneration>{};
        uint32_t semantic = UINT32_MAX;
        if (candidate->PollCounts(&semantic) != VK_SUCCESS || semantic || candidate->succeeded() || !candidate->BeginScatter(&status) || !Prove(native) ||
            candidate->PollScatter(&semantic) != VK_SUCCESS || semantic || !candidate->succeeded() || candidate->inputOwner() != input || candidate->context() != context)
            return std::shared_ptr<const VulkanSourceGeneration>{};
        return VulkanSourceGeneration::WithCompacted(input, *candidate, version);
    };
    // Equality keeps curve 0; the zero-length curve 1 is discarded.  This
    // verifies counts before scatter and every public/private output plane.
    auto subset = compact(base, 1.f, 24); CHECK(subset && subset->curveCount() == 3 && subset->pointCount() == 8);
    CHECK(CompletePacket(native, context, subset, original, {0,2,3}, {0,1,5,6,7,8,9,10}));
    CHECK(subset->geometry().alreadyDeformed && subset->geometry().topologyVersion == 24 && subset->geometry().valueVersion == 24);
    CHECK(subset->chunks().size() == 2 && subset->geometry().tiles.size() == 2 && subset->chunks()[0].curveCount == 1 && subset->geometry().tiles[0].elementCount == 1);
    // Repeated culls retain predecessors: first COW packet and original remain immutable.
    auto final = compact(subset, 2.5f, 25); CHECK(final && final->curveCount() == 1 && final->pointCount() == 4);
    CHECK(CompletePacket(native, context, final, original, {3}, {7,8,9,10}));
    CHECK(CompletePacket(native, context, subset, original, {0,2,3}, {0,1,5,6,7,8,9,10}));
    Packet stillOriginal; CHECK(Capture(native, context, base, &stillOriginal) && stillOriginal == original);
    // Zero preserves all curves (including the degenerate curve); all-cull retains
    // empty chunk/tile membership descriptors and the canonical zero offset.
    auto zero = compact(base, 0.f, 26); CHECK(zero && zero->curveCount() == 4 && zero->pointCount() == 11);
    CHECK(CompletePacket(native, context, zero, original, {0,1,2,3}, {0,1,2,3,4,5,6,7,8,9,10}));
    auto empty = compact(base, 4.f, 27); CHECK(empty && !empty->curveCount() && !empty->pointCount());
    CHECK(empty->chunks().size() == 2 && empty->geometry().tiles.size() == 2 && empty->chunks()[0].curveCount == 0 && empty->chunks()[1].curveCount == 0 && empty->geometry().tiles[0].elementCount == 0 && empty->geometry().tiles[1].elementCount == 0);
    CHECK(CompletePacket(native, context, empty, original, {}, {}));

    // Density-live curves are a prefix within each allocated chunk. A cull
    // must not resurrect previously non-live curves in this control metadata.
    auto partialInfo = Fixture(context);
    partialInfo.source.chunks[1].liveCount = 1;
    auto partial = Upload(native, std::move(partialInfo)); CHECK(partial);
    auto partialSubset = compact(partial, 1.f, 28); CHECK(partialSubset);
    CHECK(partialSubset->chunks()[1].curveCount == 2 && partialSubset->chunks()[1].liveCount == 1);
    auto partialFinal = compact(partialSubset, 2.5f, 29); CHECK(partialFinal);
    CHECK(partialFinal->chunks()[1].curveCount == 1 && partialFinal->chunks()[1].liveCount == 0);
    CHECK(CompletePacket(native, context, partialFinal, original, {3}, {7,8,9,10}));

    // A second, independent packet exercises the global scan across eight
    // full workgroups plus a partial ninth, checking ragged source offsets.
    auto large = Upload(native, LargeFixture(context)); CHECK(large && large->curveCount() == 513);
    Packet largeOriginal; CHECK(Capture(native, context, large, &largeOriginal));
    CHECK(large->PlaneOwner("sourceRootT") == nullptr && large->SourceFrameOwner("sourceRootT"));
    CHECK(large->PlaneOwner("LargeGroom"));
    auto largeOffsets = std::vector<uint32_t>(514, 0);
    for (uint32_t c = 0; c != 513; ++c) largeOffsets[c + 1] = largeOffsets[c] + c % 5;
    std::vector<uint32_t> survivors, survivorPoints;
    for (uint32_t c = 0; c != 513; ++c) if (c % 5 >= 3) {
        survivors.push_back(c);
        for (uint32_t p = largeOffsets[c]; p != largeOffsets[c + 1]; ++p) survivorPoints.push_back(p);
    }
    auto largeCandidate = pipeline->Begin(large, 2.f, &status); CHECK(largeCandidate && Prove(native));
    // Scatter is gated by a successful count proof, and its admission hook is
    // still invoked after that proof (before any payload write is submitted).
    CHECK(!largeCandidate->BeginScatter(&status));
    uint32_t semantic = UINT32_MAX;
    CHECK(largeCandidate->PollCounts(&semantic) == VK_SUCCESS && semantic == 0 && !largeCandidate->succeeded());
    bool scatterAdmission = false;
    CHECK(!largeCandidate->BeginScatter(&status, [&] { scatterAdmission = true; return false; }) && scatterAdmission);
    auto largeSubset = compact(large, 2.f, 124); CHECK(largeSubset && largeSubset->curveCount() == survivors.size());
    CHECK(CompleteRaggedPacket(native, context, largeSubset, largeOriginal, survivors, survivorPoints, largeOffsets));
    CHECK(largeSubset->PlaneOwner("LargeGroom") == large->PlaneOwner("LargeGroom"));
    // Every source chunk/tile retains its identity while its range is prefix-
    // compacted.  In particular, a chunk with no survivors has zero counts
    // but still remains present for downstream tile scheduling.
    uint32_t compactCurve = 0, compactPoint = 0;
    for (size_t i = 0; i != large->chunks().size(); ++i) {
        auto const& in = large->chunks()[i]; auto const& out = largeSubset->chunks()[i];
        uint32_t live = 0, pointsIn = 0;
        for (uint32_t c = in.firstCurve; c != in.firstCurve + in.curveCount; ++c)
            if (c % 5 >= 3) { ++live; pointsIn += largeOffsets[c + 1] - largeOffsets[c]; }
        CHECK(out.tile == in.tile && out.firstCurve == compactCurve && out.curveCount == live &&
              out.liveCount == live && out.firstCv == compactPoint && out.cvCount == 0);
        CHECK(largeSubset->geometry().tiles[i].tile == in.tile &&
              largeSubset->geometry().tiles[i].firstElement == compactCurve &&
              largeSubset->geometry().tiles[i].elementCount == live &&
              largeSubset->geometry().tiles[i].firstPoint == compactPoint &&
              largeSubset->geometry().tiles[i].pointCount == pointsIn);
        compactCurve += live; compactPoint += pointsIn;
    }
    auto largeEmpty = compact(largeSubset, 4.f, 125); CHECK(largeEmpty && !largeEmpty->curveCount() && !largeEmpty->pointCount());
    CHECK(CompleteRaggedPacket(native, context, largeEmpty, largeOriginal, {}, {}, largeOffsets));
    for (size_t i = 0; i != largeEmpty->chunks().size(); ++i) {
        auto const& out = largeEmpty->chunks()[i]; auto const& tile = largeEmpty->geometry().tiles[i];
        CHECK(out.tile == large->chunks()[i].tile && out.firstCurve == 0 && out.curveCount == 0 &&
              out.liveCount == 0 && out.firstCv == 0 && out.cvCount == 0 && tile.firstElement == 0 &&
              tile.elementCount == 0 && tile.firstPoint == 0 && tile.pointCount == 0);
    }
    largeCandidate.reset(); largeEmpty.reset(); largeSubset.reset(); large.reset();
    partialFinal.reset(); partialSubset.reset(); partial.reset();
    empty.reset(); zero.reset(); final.reset(); subset.reset(); base.reset(); pipeline.reset();
    CHECK(context->resources()->Snapshot().usedBytes == 0);
    std::puts("Vulkan complete-packet Length compaction COW: PASS");
    return 0;
}
