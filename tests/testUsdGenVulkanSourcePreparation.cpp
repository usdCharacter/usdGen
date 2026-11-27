// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGen/curveLoader.h"
#include "usdGenUsd/lower.h"
#include "usdGen/op.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "utils/pxrInterop.h"

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan source preparation check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static VulkanSourcePrepareInfo Source() {
    VulkanSourcePrepareInfo info;
    auto& b = info.source;
    b.totalCurves = 2; b.totalCvs = 4; b.topologyVersion = 11; b.valueVersion = 12;
    b.px = {0,1,2,3}; b.py = {1,2,3,4}; b.pz = {2,3,4,5};
    b.curveId = {17, 29}; b.cvOffsets = {0,2,4}; b.width = {1,2,3,4};
    b.rootT = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(1,0,0));
    b.rootB = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(0,1,0));
    b.rootN = usdGen::UsdGenVec3fArray(2, usdGen::UsdGenVec3f(0,0,1));
    std::vector<uint8_t> payload{1,2,3,4,5,6,7,8,9};
    info.additionalNamed.push_back({{TfToken("stride5"), UsdGenDeviceValueType::Float32,
        UsdGenDeviceDomain::Primitive, 2, 1, 5, true, UsdGenDeviceChannelSemantic::Generic}, payload});
    return info;
}

static VulkanPreparedSource::Plane const* Find(VulkanPreparedSource const& source, std::string const& name) {
    for (auto const& plane : source.planes()) if (plane.metadata.name == name) return &plane;
    return nullptr;
}

static bool LoadAffine(UsdGenGraphDesc const& desc, UsdGenCurveBuffer* out,
                       UsdGenDiagnostics* diagnostics) {
    usdGen::UsdGenGraphSpec const lowered = usdGenUsd::Lower(desc);
    UsdGenParamView params{&lowered, &lowered.nodes.front()};
    UsdGenCaptureContext context; context.desc = &lowered; context.params = &params;
    return UsdGenCurveLoader::Load(context, UsdGenCurveSetInputs(lowered).front(), out, diagnostics);
}

static UsdGenGraphDesc AffineDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Affine");
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Affine/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {0,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3}; surface.faceVertexIndices = {0,1,2};
    surface.uv = {{0,0}, {1,0}, {0,1}};
    usdGen::UsdGenMatrix4d relative(1.0);
    relative[0][0] = 0.; relative[0][1] = 1.;
    relative[1][0] = -1.; relative[1][1] = 0.;
    relative[3][0] = 3.; relative[3][1] = -2.; relative[3][2] = 4.;
    usdGen::UsdGenMatrix4d sourceWorld(1.0);
    sourceWorld[0][0] = 0.; sourceWorld[0][1] = -1.;
    sourceWorld[1][0] = 1.; sourceWorld[1][1] = 0.;
    sourceWorld[3][0] = 7.; sourceWorld[3][1] = -5.; sourceWorld[3][2] = 2.;
    surface.worldMatrix = relative * sourceWorld;
    desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Affine/Curves"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair"); curves.curveVertexCounts = {3};
    curves.points = {{.2f,.2f,0}, {.2f,.2f,1}, {.2f,.2f,2}};
    curves.rest = curves.points; curves.curveId = {17};
    curves.skinPrim = {0}; curves.skinPrimUv = {{.2f,.2f}};
    curves.worldMatrix = sourceWorld; // rootFrame remains absent: derive it.
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/Affine/Source"); source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path}; source.surfaces = {surface.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("rebind"), VtValue(TfToken("always")), false}};
    desc.nodes.push_back(source); desc.terminal = source.path;
    return desc;
}

template <class T> static std::vector<uint8_t> Bytes(VtArray<T> const& values) {
    std::vector<uint8_t> bytes(values.size() * sizeof(T));
    if (!bytes.empty()) std::memcpy(bytes.data(), values.cdata(), bytes.size());
    return bytes;
}

int main() {
    std::string reason;
    auto input = Source();
    auto prepared = VulkanPreparedSource::Prepare(input, &reason);
    CHECK(prepared && prepared->curveCount() == 2 && prepared->pointCount() == 4);
    CHECK(prepared->topologyVersion() == 11 && prepared->valueVersion() == 12);
    auto stride = Find(*prepared, "stride5");
    CHECK(stride && stride->metadata.strideBytes == 5 && stride->metadata.elementCount == 2 &&
        stride->bytes == std::vector<uint8_t>({1,2,3,4,5,6,7,8,9}));
    size_t privateFrames = 0;
    for (auto const& plane : prepared->planes()) if (plane.privateFrame) ++privateFrames;
    CHECK(privateFrames == 3 && Find(*prepared, "sourceRootT") &&
        Find(*prepared, "sourceRootB") && Find(*prepared, "sourceRootN"));

    // Preparation owns an immutable deep copy, independent of caller mutation.
    input.source.px[0] = 99; input.source.curveId[0] = 999;
    input.additionalNamed[0].bytes[0] = 99;
    CHECK(Find(*prepared, "points") && Find(*prepared, "stride5") &&
        Find(*prepared, "stride5")->bytes[0] == 1);
    float firstPoint[3]{};
    std::memcpy(firstPoint, Find(*prepared, "points")->bytes.data(), sizeof(firstPoint));
    CHECK(firstPoint[0] == 0 && firstPoint[1] == 1 && firstPoint[2] == 2);
    uint64_t firstId = 0;
    CHECK(Find(*prepared, "stableIds"));
    std::memcpy(&firstId, Find(*prepared, "stableIds")->bytes.data(), sizeof(firstId));
    CHECK(firstId == 17);

    // CurveLoader derives source-local T/B/N from a genuinely nonidentity
    // source/surface affine pair. Prepare must own those private planes even
    // after both the descriptor and loader result are destroyed/mutated.
    auto affineDesc = AffineDesc();
    auto const sourcePoints = affineDesc.curveSets.front().points;
    auto const sourceRest = affineDesc.curveSets.front().rest;
    UsdGenCurveBuffer affineLoaded;
    UsdGenDiagnostics affineDiagnostics;
    CHECK(LoadAffine(affineDesc, &affineLoaded, &affineDiagnostics));
    CHECK(affineDesc.curveSets.front().rootFrame.empty() &&
          affineLoaded.rootT.size() == 1 && affineLoaded.rootB.size() == 1 &&
          affineLoaded.rootN.size() == 1);
    CHECK(affineLoaded.rootT[0] == usdGen::UsdGenVec3f(0,1,0) &&
          affineLoaded.rootB[0] == usdGen::UsdGenVec3f(-1,0,0) &&
          affineLoaded.rootN[0] == usdGen::UsdGenVec3f(0,0,1));
    VulkanSourcePrepareInfo affineInfo;
    affineInfo.source = affineLoaded;
    auto affinePrepared = VulkanPreparedSource::Prepare(affineInfo, &reason);
    CHECK(affinePrepared && affinePrepared->pointCount() == affineLoaded.totalCvs);
    auto affineT = Find(*affinePrepared, "sourceRootT");
    auto affineB = Find(*affinePrepared, "sourceRootB");
    auto affineN = Find(*affinePrepared, "sourceRootN");
    CHECK(affineT && affineB && affineN && affineT->privateFrame &&
          affineB->privateFrame && affineN->privateFrame &&
          affineT->bytes == Bytes(affineLoaded.rootT) &&
          affineB->bytes == Bytes(affineLoaded.rootB) &&
          affineN->bytes == Bytes(affineLoaded.rootN));
    affineDesc.curveSets.front().points.clear(); affineDesc.surfaces.front().points.clear();
    affineLoaded.px.clear(); affineLoaded.py.clear(); affineLoaded.pz.clear();
    affineLoaded.rest.clear(); affineLoaded.rootT[0] = usdGen::UsdGenVec3f(9,9,9);
    CHECK(affineDesc.curveSets.front().points != sourcePoints &&
          affineDesc.curveSets.front().rest == sourceRest &&
          affineT->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(0,1,0))) &&
          affineB->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(-1,0,0))) &&
          affineN->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(0,0,1))));
    affineInfo = {}; affineLoaded = {}; affineDesc = {};
    CHECK(Find(*affinePrepared, "points") &&
          Find(*affinePrepared, "points")->bytes == Bytes(sourcePoints) &&
          affineT->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(0,1,0))) &&
          affineB->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(-1,0,0))) &&
          affineN->bytes == Bytes(usdGen::UsdGenVec3fArray(1, usdGen::UsdGenVec3f(0,0,1))));

    auto permuted = Source(); permuted.source.curveId = {29,17};
    CHECK(VulkanPreparedSource::Prepare(permuted, &reason));

    for (size_t count : {size_t{8}, size_t{10}}) {
        auto bad = Source(); bad.additionalNamed[0].bytes.resize(count);
        CHECK(!VulkanPreparedSource::Prepare(std::move(bad), &reason));
    }
    auto duplicate = Source(); duplicate.source.curveId = {17,17};
    CHECK(!VulkanPreparedSource::Prepare(std::move(duplicate), &reason));
    auto mismatch = Source(); mismatch.source.px.pop_back();
    CHECK(!VulkanPreparedSource::Prepare(std::move(mismatch), &reason));
    auto badStride = Source(); badStride.additionalNamed[0].metadata.strideBytes = 4;
    CHECK(!VulkanPreparedSource::Prepare(std::move(badStride), &reason));

    VulkanSourcePrepareInfo empty;
    auto emptyPrepared = VulkanPreparedSource::Prepare(std::move(empty), &reason);
    CHECK(emptyPrepared && emptyPrepared->curveCount() == 0 && emptyPrepared->pointCount() == 0);
    CHECK(emptyPrepared->planes().size() == 6);
    CHECK(Find(*emptyPrepared, "curveOffsets") && Find(*emptyPrepared, "curveOffsets")->bytes.size() == 4);
    for (char const* name : {"points", "stableIds", "rest", "width", "hairT"}) {
        auto const* plane = Find(*emptyPrepared, name);
        CHECK(plane && plane->metadata.elementCount == 0 && plane->bytes.empty());
    }
    CHECK(Find(*emptyPrepared, "rest")->metadata.semantic == UsdGenDeviceChannelSemantic::RestPoints &&
          Find(*emptyPrepared, "width")->metadata.semantic == UsdGenDeviceChannelSemantic::Widths &&
          Find(*emptyPrepared, "hairT")->metadata.semantic == UsdGenDeviceChannelSemantic::HairT);

    std::atomic<bool> concurrentOk{true};
    std::array<std::thread, 4> workers;
    for (auto& worker : workers) worker = std::thread([&] {
        auto copy = Source(); std::string localReason;
        auto result = VulkanPreparedSource::Prepare(std::move(copy), &localReason);
        if (!result || !Find(*result, "stride5") || Find(*result, "stride5")->bytes.size() != 9)
            concurrentOk.store(false, std::memory_order_release);
    });
    for (auto& worker : workers) worker.join();
    CHECK(concurrentOk.load(std::memory_order_acquire));
    return 0;
}
