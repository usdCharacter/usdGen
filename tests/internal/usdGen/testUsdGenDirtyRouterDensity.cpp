// testUsdGenDirtyRouterDensity.cpp — T1: a paint-density primvar dirty on a
// surface-bound node's surface routes UsdGenDirtyCapture to that node (the
// brush bake -> groom recook path). Value-leaf and structural-primvar
// dirties re-capture; a bare coarse container dirty keeps the
// pre-existing SurfacePoints route; an unrelated surface stays quiet.
#include "utils/pxrInterop.h"
#include "usdGenImaging/usdGenDirtyRouter.h"
#include "usdGen/compile/graph.h"
#include "usdGen/api/session.h"
#include "usdGen/api/types.h"
#include "usdGenUsd/lower.h"

#include "pxr/imaging/hd/dataSourceLocator.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

// Keys the hand-built snapshots name prims by (the router reads the table).
usdGenUsd::PathRecorder gKeys;

UsdGenGraphRoutingSnapshot Snapshot()
{
    UsdGenGraphRoutingSnapshot snap;
    snap.description = gKeys(SdfPath("/Groom"));
    snap.surfaceKeys.push_back(gKeys(SdfPath("/Scalp")));
    UsdGenGraphRoutingNode node;
    node.id = 0;
    node.key = gKeys(SdfPath("/Groom/scatter"));
    node.surface = 0;
    node.hasSurface = true;
    snap.nodes.push_back(node);
    snap.terminal = 0;
    snap.host = gKeys.Table();
    return snap;
}

HdSceneIndexObserver::DirtiedPrimEntries Dirtied(
    SdfPath const &prim, HdDataSourceLocator const &locator)
{
    HdSceneIndexObserver::DirtiedPrimEntry entry;
    entry.primPath = prim;
    entry.dirtyLocators.insert(locator);
    return {entry};
}

HdDataSourceLocator Loc(std::initializer_list<TfToken> tokens)
{
    HdDataSourceLocator loc;
    for (TfToken const &t : tokens) {
        loc = loc.Append(t);
    }
    return loc;
}

}  // namespace

int main()
{
    usdGenImaging::UsdGenDirtyRouter router;
    router.Rebuild(Snapshot());

    // 1. A value-leaf dirty (what the brush bake emits) re-captures node 0.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:density"),
                                  TfToken("primvarValue")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.surfaceBits.empty());
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[0] == UsdGenDirtyCapture);
    }

    // 2. The structural primvar locator (bake creating the primvar) too.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:density")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[0] == UsdGenDirtyCapture);
    }

    // 3. A bare coarse container dirty keeps the pre-existing
    // SurfacePoints route (the deeper points prefix wins); real bakes
    // always carry the fine locator too (cases 1-2).
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.empty());
        CHECK(pending.surfaceBits.size() == 1);
        CHECK(pending.surfaceBits[0] == UsdGenDirtySurfacePoints);
    }

    // 4. An unrelated surface prim stays silent.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Other"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:density"),
                                  TfToken("primvarValue")})),
                     &pending);
        CHECK(!pending.Any());
    }

    // 5. Rest-normals routing still lands Capture (the sibling row).
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("usdGen"), TfToken("rest"),
                                  TfToken("normals")})),
                     &pending);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[0] == UsdGenDirtyCapture);
    }

    // 6. A GeomSubset-bound node (02 §2.20): the subset's desc reads its
    // parent mesh, so points, topology and paint rows key on the mesh; the
    // subset's own indices/type (bare and prefixed) re-capture the node.
    {
        UsdGenGraphRoutingSnapshot snap = Snapshot();
        snap.surfaceKeys = {gKeys(SdfPath("/Scalp/crown"))};
        snap.meshKeys = {gKeys(SdfPath("/Scalp"))};
        snap.host = gKeys.Table();
        usdGenImaging::UsdGenDirtyRouter subsetRouter;
        subsetRouter.Rebuild(snap);

        UsdGenPendingDirty points;
        subsetRouter.Route(Dirtied(SdfPath("/Scalp"),
                                   Loc({TfToken("primvars"), TfToken("points")})),
                           &points);
        CHECK(points.surfaceBits.size() == 1);
        CHECK(points.surfaceBits[0] == UsdGenDirtySurfacePoints);

        UsdGenPendingDirty paint;
        subsetRouter.Route(Dirtied(SdfPath("/Scalp"),
                                   Loc({TfToken("primvars"),
                                        TfToken("usdGen:paint:density")})),
                           &paint);
        CHECK(paint.nodeBits.size() == 1 && paint.nodeBits[0] == UsdGenDirtyCapture);

        for (HdDataSourceLocator const &loc :
             {Loc({TfToken("indices")}), Loc({TfToken("type")}),
              Loc({TfToken("geomSubset"), TfToken("indices")})}) {
            UsdGenPendingDirty edit;
            subsetRouter.Route(Dirtied(SdfPath("/Scalp/crown"), loc), &edit);
            CHECK(edit.nodeBits.size() == 1 && edit.nodeBits[0] == UsdGenDirtyCapture);
        }
    }

    std::printf("ok: paint-density dirties re-capture the surface node\n");
    return 0;
}
