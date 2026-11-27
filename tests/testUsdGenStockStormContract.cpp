// Stock-Hydra data-source contract: this intentionally uses only public
// BasisCurves and primvar schema payloads.  It does not load or reference a
// private HdSt provider.
#include "usdGen/curveBuffer.h"
#include "usdGenTestUtils/hydraPublishers.h"
#include "usdGenHydraCurves/usdGenTilePublisher.h"
#include "usdGenUsd/lower.h"

#include "pxr/pxr.h"
#include "pxr/base/vt/array.h"
#include "pxr/imaging/hd/basisCurvesSchema.h"
#include "pxr/imaging/hd/basisCurvesTopologySchema.h"
#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/tokens.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

bool
Check(bool const condition, char const *const message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

HdContainerDataSourceHandle
ContainerChild(HdContainerDataSourceHandle const &container,
               TfToken const &name)
{
    return container ? HdContainerDataSource::Cast(container->Get(name))
                     : HdContainerDataSourceHandle();
}

template <class T>
bool
SampleEquals(HdContainerDataSourceHandle const &container,
             TfToken const &name, T const &expected)
{
    HdSampledDataSourceHandle const value = HdSampledDataSource::Cast(
        container ? container->Get(name) : HdDataSourceBaseHandle());
    return value && value->GetValue(0.0).IsHolding<T>() &&
        value->GetValue(0.0).UncheckedGet<T>() == expected;
}

} // namespace

int
main()
{
    usdGenTest::EnsureHydraPublishers();
    usdGen::UsdGenTilePublication tile;
    usdGenUsd::PathRecorder keys;   // the cooked spec's path table, by hand
    tile.prim = usdGen::UsdGenPrimRef{keys(SdfPath("/stock/tile_0000"))};
    tile.curveVertexCounts = VtIntArray{2, 3};
    tile.points = VtVec3fArray{
        GfVec3f(0.0f, 0.0f, 0.0f), GfVec3f(0.0f, 1.0f, 0.0f),
        GfVec3f(1.0f, 0.0f, 0.0f), GfVec3f(1.0f, 1.0f, 0.0f),
        GfVec3f(1.0f, 2.0f, 0.0f)};
    tile.widths = VtFloatArray{0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    HdContainerDataSourceHandle const root =
        usdGenImaging::UsdGenTilePublisher::BuildTileDataSource(tile, keys.Table());
    HdContainerDataSourceHandle const basis =
        ContainerChild(root, TfToken("basisCurves"));
    HdContainerDataSourceHandle const topology =
        ContainerChild(basis, TfToken("topology"));
    HdContainerDataSourceHandle const primvars =
        ContainerChild(root, TfToken("primvars"));
    HdContainerDataSourceHandle const points =
        ContainerChild(primvars, TfToken("points"));
    HdContainerDataSourceHandle const widths =
        ContainerChild(primvars, TfToken("widths"));

    bool okay = true;
    okay &= Check(root && basis && topology && primvars,
                  "public basisCurves/topology/primvars containers exist");
    okay &= Check(!root->Get(TfToken("hdStBasisCurvesGpu")),
                  "stock payload does not expose a private GPU datasource");
    okay &= Check(SampleEquals(topology, TfToken("curveVertexCounts"),
                               tile.curveVertexCounts),
                  "standard curveVertexCounts preserve ragged topology");

    // type/basis/wrap are read from INSIDE the topology container
    // (pxr/imaging/hd/basisCurvesTopologySchema.h:35-42). Anything authored
    // as a direct child of `basisCurves` is invisible to stock Hydra, which
    // then falls back to type = linear / basis = bezier / wrap = nonperiodic.
    HdBasisCurvesTopologySchema const topologySchema =
        HdBasisCurvesSchema::GetFromParent(root).GetTopology();
    okay &= Check(topologySchema.IsDefined(),
                  "HdBasisCurvesSchema::GetTopology() resolves on the tile");
    okay &= Check(topologySchema.GetType() &&
                      topologySchema.GetType()->GetTypedValue(0) ==
                          HdTokens->cubic,
                  "topology/type == cubic through the public schema");
    okay &= Check(topologySchema.GetBasis() &&
                      topologySchema.GetBasis()->GetTypedValue(0) ==
                          HdTokens->bspline,
                  "topology/basis == bspline through the public schema");
    okay &= Check(topologySchema.GetWrap() &&
                      topologySchema.GetWrap()->GetTypedValue(0) ==
                          HdTokens->pinned,
                  "topology/wrap == pinned through the public schema");
    okay &= Check(bool(topologySchema.GetCurveVertexCounts()),
                  "topology/curveVertexCounts through the public schema");
    okay &= Check(!basis->Get(TfToken("type")) &&
                      !basis->Get(TfToken("basis")) &&
                      !basis->Get(TfToken("wrap")),
                  "no type/basis/wrap directly under basisCurves "
                  "(stock Hydra never reads them there)");
    okay &= Check(SampleEquals(points, TfToken("primvarValue"), tile.points),
                  "standard points primvar preserves every CV");
    okay &= Check(SampleEquals(widths, TfToken("primvarValue"), tile.widths),
                  "standard widths primvar preserves every CV");
    okay &= Check(SampleEquals(points, TfToken("interpolation"),
                               TfToken("vertex")) &&
                  SampleEquals(widths, TfToken("interpolation"),
                               TfToken("vertex")),
                  "points and widths use standard vertex primvars");
    return okay ? 0 : 1;
}
