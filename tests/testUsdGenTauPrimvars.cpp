// testUsdGenTauPrimvars.cpp — tau planes reach Hydra as zero-copy vec3 views.
//
// The tile/scalp publishers expose furTauP/furTauN (float-triple operator
// planes) as vec3 primvars without copying: GfVec3f is three contiguous
// floats, so a VtArray foreign-source view reads exactly the engine's bytes.
// This test pins:
//   (a) the published values are bit-exact (float-compare per component);
//   (b) the published array shares the plane's storage (pointer identity),
//       which is what makes the publish O(1) instead of O(bytes);
//   (c) the size/arity guards still route malformed planes to the generic
//       float path instead of wrapping them.
// Exit: 0 pass, 1 fail.
// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenImaging/usdGenTilePublisher.h"

#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
using usdGenImaging::UsdGenTilePublisher;

namespace {

int failures = 0;

void Check(bool value, std::string const &message)
{
    std::printf("%s: %s\n", value ? "ok  " : "FAIL", message.c_str());
    if (!value) ++failures;
}

// primvars/<name>/primvarValue off a built prim, as the render delegate
// would read it.
VtValue Primvar(HdContainerDataSourceHandle const &prim, char const *name,
                TfToken *interpolation)
{
    HdPrimvarsSchema pvs = HdPrimvarsSchema::GetFromParent(prim);
    HdPrimvarSchema pv = pvs.GetPrimvar(TfToken(name));
    if (!pv.IsDefined()) return VtValue();
    if (interpolation && pv.GetInterpolation())
        *interpolation = pv.GetInterpolation()->GetTypedValue(0.0);
    HdSampledDataSourceHandle values = pv.GetPrimvarValue();
    return values ? values->GetValue(0.0) : VtValue();
}

UsdGenPlane FloatPlane(char const *name, char const *interp, uint8_t arity,
                       std::vector<float> const &values)
{
    UsdGenPlane p;
    p.name = TfToken(name);
    p.interpolation = TfToken(interp);
    p.type = TfToken("float");
    p.arity = arity;
    p.f = VtFloatArray(values.data(), values.data() + values.size());
    return p;
}

// Two curves (3 + 4 CVs): 7 CVs, 21 floats per tau plane.
UsdGenTilePublication TileWithTaus()
{
    UsdGenTilePublication tile;
    tile.tile = 0;
    tile.primPath = SdfPath("/groom/__usdGenRender/tile_0000");
    tile.curveVertexCounts = VtIntArray{3, 4};
    tile.points = VtVec3fArray{
        GfVec3f(0, 0, 0), GfVec3f(0, 0, 1), GfVec3f(0, 0, 2),
        GfVec3f(1, 0, 0), GfVec3f(1, 0, 1), GfVec3f(1, 0, 2), GfVec3f(1, 0, 3)};
    std::vector<float> p, n;
    for (int i = 0; i < 21; ++i) {
        p.push_back(float(i) * 0.5f - 3.0f);
        n.push_back(float(20 - i) * -0.25f);
    }
    tile.extraUniform.push_back(FloatPlane("furTauP", "vertex", 3, p));
    tile.extraUniform.push_back(FloatPlane("furTauN", "vertex", 3, n));
    tile.extentMin = GfVec3d(0, 0, 0);
    tile.extentMax = GfVec3d(1, 0, 3);
    tile.xformMatrix = GfMatrix4d(1.0);
    return tile;
}

bool TauBytesEqual(VtVec3fArray const &vec, VtFloatArray const &flat)
{
    if (vec.size() * 3 != flat.size()) return false;
    for (size_t i = 0; i < vec.size(); ++i)
        if (vec[i][0] != flat[i * 3] || vec[i][1] != flat[i * 3 + 1] ||
            vec[i][2] != flat[i * 3 + 2])
            return false;
    return true;
}

}  // namespace

int main()
{
    // (a)+(b) Tile taus: exact bytes, shared storage.
    {
        UsdGenTilePublication tile = TileWithTaus();
        HdContainerDataSourceHandle prim =
            UsdGenTilePublisher::BuildTileDataSource(tile, 7);
        Check(bool(prim), "a tile with tau planes builds");
        for (char const *name : {"furTauP", "furTauN"}) {
            TfToken interp;
            VtValue v = Primvar(prim, name, &interp);
            UsdGenPlane const *plane = nullptr;
            for (auto const &p : tile.extraUniform)
                if (p.name == TfToken(name)) plane = &p;
            Check(v.IsHolding<VtVec3fArray>(),
                  std::string(name) + " publishes as a vec3 primvar");
            Check(interp == TfToken("vertex"),
                  std::string(name) + " keeps vertex interpolation");
            if (v.IsHolding<VtVec3fArray>() && plane) {
                VtVec3fArray const &vec = v.UncheckedGet<VtVec3fArray>();
                Check(TauBytesEqual(vec, plane->f),
                      std::string(name) + " bytes are bit-exact");
                Check(static_cast<void const *>(vec.data()) ==
                          static_cast<void const *>(plane->f.data()),
                      std::string(name) + " shares the plane's storage");
            }
        }
        HdSampledDataSourceHandle gen = HdSampledDataSource::Cast(
            HdContainerDataSource::Get(prim,
                HdDataSourceLocator(TfToken("generation"))));
        Check(gen && gen->GetValue(0.0).IsHolding<int>() &&
                  gen->GetValue(0.0).UncheckedGet<int>() == 7,
              "the generation stamp still rides the tile");
    }

    // (c) Guard: a mis-sized tau plane takes the generic float path.
    {
        UsdGenTilePublication tile = TileWithTaus();
        tile.extraUniform.clear();
        tile.extraUniform.push_back(
            FloatPlane("furTauP", "vertex", 3, {1.0f, 2.0f, 3.0f}));
        HdContainerDataSourceHandle prim =
            UsdGenTilePublisher::BuildTileDataSource(tile, 7);
        VtValue v = Primvar(prim, "furTauP", nullptr);
        Check(v.IsHolding<VtFloatArray>() &&
                  v.UncheckedGet<VtFloatArray>().size() == 3,
              "a mis-sized furTauP plane publishes as generic floats");
    }

    // (a)+(b) Scalp-cap taus: same view, same guarantees.
    {
        UsdGenScalpShadowPublication cap;
        cap.primPath = SdfPath("/groom/__usdGenRender/scalpShadow");
        cap.points = VtVec3fArray{GfVec3f(0, 0, 0), GfVec3f(1, 0, 0)};
        cap.normals =
            VtVec3fArray{GfVec3f(0, 0, 1), GfVec3f(0, 0, 1)};
        cap.faceVertexCounts = VtIntArray{3};
        cap.faceVertexIndices = VtIntArray{0, 1, 0};
        cap.extraUniform.push_back(FloatPlane(
            "furTauP", "vertex", 3, {0.5f, -1.0f, 2.0f, 3.0f, 4.0f, -5.0f}));
        cap.extentMin = GfVec3d(0, 0, 0);
        cap.extentMax = GfVec3d(1, 0, 0);
        HdContainerDataSourceHandle prim =
            UsdGenTilePublisher::BuildScalpShadowDataSource(cap, 7);
        Check(bool(prim), "a cap with a tau plane builds");
        VtValue v = Primvar(prim, "furTauP", nullptr);
        Check(v.IsHolding<VtVec3fArray>() &&
                  v.UncheckedGet<VtVec3fArray>().size() == 2,
              "the cap's tau plane publishes as vec3");
        if (v.IsHolding<VtVec3fArray>()) {
            VtVec3fArray const &vec = v.UncheckedGet<VtVec3fArray>();
            // Through const access only: a mutable data() would CoW-detach
            // the shared plane and move it, failing the comparison below.
            UsdGenScalpShadowPublication const &ccap = cap;
            VtFloatArray const &flat = ccap.extraUniform[0].f;
            Check(TauBytesEqual(vec, flat),
                  "the cap's tau bytes are bit-exact");
            Check(static_cast<void const *>(vec.data()) ==
                      static_cast<void const *>(flat.data()),
                  "the cap's tau view shares the plane's storage");
        }
    }

    if (failures == 0) std::printf("testUsdGenTauPrimvars: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
