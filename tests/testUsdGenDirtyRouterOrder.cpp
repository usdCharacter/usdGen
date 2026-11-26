// testUsdGenDirtyRouterOrder.cpp — T1: an operatorOrder-aggregate dirty on
// the description routes structural (reordering the chain is a
// recompile, not a value sweep). The tileTarget sibling still routes
// Topology on the terminal; an unknown usdGen leaf on the description
// stays quiet; a whole-container resync still routes structural.
#include "usdGenImaging/usdGenDirtyRouter.h"
#include "usdGen/graph.h"
#include "usdGen/session.h"
#include "usdGen/types.h"
#include "usdGenUsd/lower.h"

#include "pxr/imaging/hd/dataSourceLocator.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

// Keys the hand-built snapshot names prims by (the router reads the table).
usdGenUsd::PathRecorder gKeys;

UsdGenGraphRoutingSnapshot Snapshot()
{
    UsdGenGraphRoutingSnapshot snap;
    snap.description = gKeys(SdfPath("/Groom/Description"));
    snap.surfaceKeys.push_back(gKeys(SdfPath("/Scalp")));
    UsdGenGraphRoutingNode node;
    node.id = 0;
    node.key = gKeys(SdfPath("/Groom/Description/Ops/grow"));
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

    // 1. The authored order array routes structural, with no value bits.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Groom/Description"),
                             Loc({TfToken("usdGen"),
                                  TfToken("operatorOrder"),
                                  TfToken("value")})),  // value-leaf form
                     &pending);
        CHECK(pending.structural);
        CHECK(pending.nodeBits.empty());
        CHECK(pending.surfaceBits.empty());
    }

    // 2. The container-leaf form too.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Groom/Description"),
                             Loc({TfToken("usdGen"),
                                  TfToken("operatorOrder")})),
                     &pending);
        CHECK(pending.structural);
        CHECK(pending.nodeBits.empty());
        CHECK(pending.surfaceBits.empty());
    }

    // 3. The tileTarget sibling still routes Topology on the terminal.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Groom/Description"),
                             Loc({TfToken("usdGen"),
                                  TfToken("tileTarget")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[0] == UsdGenDirtyTopology);
    }

    // 4. An unknown usdGen leaf on the description stays quiet.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Groom/Description"),
                             Loc({TfToken("usdGen"),
                                  TfToken("noSuchLeaf")})),
                     &pending);
        CHECK(!pending.Any());
    }

    // 5. A whole-container resync still routes structural.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Groom/Description"),
                             Loc({TfToken("usdGen")})),
                     &pending);
        CHECK(pending.structural);
    }

    std::printf("ok: operatorOrder dirties recompile the chain\n");
    return 0;
}
