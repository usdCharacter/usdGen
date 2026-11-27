// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_TEST_VULKAN_SURFACE_CAGE_FIXTURE_H
#define USDGEN_TEST_VULKAN_SURFACE_CAGE_FIXTURE_H
#include "usdGenUsd/lower.h"
#include <Ptexture.h>
#include <memory>
#include <string>
#include "utils/pxrInterop.h"

namespace usdGen::test {
// Categorical maps reserve zero for outside; a tube ID is encoded as ID+1.
inline bool WriteSurfaceCageOwnerMap(std::string const& path, float owner = 42.f) {
    Ptex::String error;
    auto* writer = PtexWriter::open(path.c_str(), Ptex::mt_quad, Ptex::dt_float,
        1, -1, 1, error, false);
    if (!writer) return false;
    int adjacentFaces[4] = {-1,-1,-1,-1}, adjacentEdges[4] = {0,0,0,0};
    bool ok = writer->writeConstantFace(0,
        Ptex::FaceInfo(Ptex::Res(int8_t(0),int8_t(0)), adjacentFaces, adjacentEdges), &owner);
    ok = writer->close(error) && ok;
    writer->release();
    return ok;
}
inline UsdGenGraphDesc SurfaceCageDesc(std::string const& mapFile) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/SurfaceCage"); desc.defaultWidth = .2f;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {4}; surface.faceVertexIndices = {0,1,2,3};
    desc.surfaces = {surface};
    UsdGenMapDesc map;
    map.path = SdfPath("/RegionMap"); map.type = TfToken("UsdGenPtexMap");
    map.resolvedAssetPath = mapFile;
    map.params = {{TfToken("map:filter"),VtValue(TfToken("nearest")),false},
        {TfToken("map:firstChannel"),VtValue(0),false},
        {TfToken("map:channelCount"),VtValue(1),false},
        {TfToken("map:blur"),VtValue(0.f),false},
        {TfToken("map:textureGeneration"),VtValue(uint64_t(91)),false}};
    desc.maps = {map};
    UsdGenCurveSetDesc sparse;
    // CurveSource consumes its sparse rails through the curve slot. Marking
    // these as Reference makes the general compiler resolve a reference slot
    // that CurveSource does not declare, before either backend is selected.
    sparse.path = SdfPath("/Sparse"); sparse.role = UsdGenLane::Chunked;
    sparse.curveRole = TfToken("hair"); sparse.curveVertexCounts = {3,3,3};
    sparse.points = {{0,0,0},{0,0,.25f},{0,0,1},
                     {1,0,0},{1,0,.25f},{1,0,1},
                     {0,1,0},{0,1,.25f},{0,1,1}};
    sparse.rest = sparse.points;
    auto cage = std::make_shared<UsdGenSurfaceCagePayload>();
    cage->ownerIds = {41}; cage->ownerDensities = {30.f}; cage->ownerSeeds = {73};
    cage->ownerCvCounts = {4}; cage->ownerEdgeBias = {0.f};
    cage->ownerChartCentroids = {GfVec2f(0.f)}; cage->ownerChartMeanRadii = {1.f};
    cage->ownerLengthProfileOffsets = {0,2};
    cage->ownerLengthProfile = {{0,1},{1,1}};
    cage->triangles = {GfVec3i(0,1,2)}; cage->triangleOwnerIndices = {0};
    cage->triangleRootCharts = {{0,0},{1,0},{0,1}};
    cage->normalizedT = {0,.25f,1,0,.25f,1,0,.25f,1};
    sparse.surfaceCage = cage;
    for (auto const& item : {std::pair<char const*,int>{"tubeId",41},
                            {"regionId",8},{"hierarchyLevel",2}}) {
        UsdGenAuthoredPlaneDesc plane;
        plane.name = TfToken(item.first); plane.type = UsdGenAuthoredPlaneType::Int32;
        plane.domain = UsdGenAuthoredPlaneDomain::Primitive; plane.arity = 1;
        plane.intValues = {item.second,item.second,item.second};
        sparse.authoredPlanes.push_back(std::move(plane));
    }
    desc.curveSets = {sparse};
    UsdGenNodeDesc node;
    node.path = SdfPath("/Source"); node.type = TfToken("UsdGenCurveSource");
    node.curves = {sparse.path}; node.surfaces = {surface.path}; node.maps = {map.path};
    node.mapBindings = {{map.path,TfToken("usdGen:regionMap")}};
    node.params = {{TfToken("interpolationMode"),VtValue(TfToken("surfaceCage")),false},
        {TfToken("expectMapGeneration"),VtValue(uint64_t(91)),false},
        {TfToken("densityMultiplier"),VtValue(1.f),false},
        {TfToken("regionMapChannel"),VtValue(0),false},
        {TfToken("useRest"),VtValue(true),false},
        {TfToken("rebind"),VtValue(TfToken("never")),false}};
    desc.nodes = {node}; desc.terminal = node.path;
    return desc;
}
} // namespace usdGen::test
#endif
