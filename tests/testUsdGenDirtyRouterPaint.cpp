// testUsdGenDirtyRouterPaint.cpp — T1: a paint-map primvar dirty on the
// surface routes UsdGenDirtyMap | UsdGenDirtyCapture to the sampling node
// (the length/width/clump/curl bake -> groom recook path). The snapshot
// fill resolves both ref lists that name paint maps: direct map slots
// (mapRefs) and expression inputs (geometryRefs, where the length
// wiring's ptex("lengthPaint") lands). A non-paint map contributes no
// paint ref; a bare coarse container dirty keeps the pre-existing
// SurfacePoints route; unrelated surfaces and primvars stay quiet.
#include "usdGenImaging/usdGenDirtyRouter.h"
#include "usdGenUsd/lower.h"
#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/session.h"
#include "usdGen/types.h"

#include "pxr/imaging/hd/dataSourceLocator.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

UsdGenGraphDesc Desc()
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Description");
    UsdGenSurfaceDesc scalp;
    scalp.path = SdfPath("/Scalp");
    scalp.restPoints = {{0,0,0},{1,0,0},{0,1,0},
                        {2,0,0},{3,0,0},{2,1,0}};
    scalp.points = scalp.restPoints;
    scalp.faceVertexCounts = {3,3};
    scalp.faceVertexIndices = {0,1,2,3,4,5};
    d.surfaces.push_back(scalp);

    UsdGenMapDesc length;
    length.path = SdfPath("/Groom/Description/Maps/lengthPaint");
    length.type = TfToken("UsdGenPaintMap");
    length.paintSurface = scalp.path;
    length.paintPrimvar = TfToken("usdGen:paint:length");
    d.maps.push_back(length);
    UsdGenMapDesc width;
    width.path = SdfPath("/Groom/Description/Maps/widthPaint");
    width.type = TfToken("UsdGenPaintMap");
    width.paintSurface = scalp.path;
    width.paintPrimvar = TfToken("usdGen:paint:width");
    d.maps.push_back(width);
    // A non-paint map in a direct slot: no paint ref, map-prim row only.
    UsdGenMapDesc ptex;
    ptex.path = SdfPath("/Groom/Description/Maps/region");
    ptex.type = TfToken("UsdGenPtexMap");
    d.maps.push_back(ptex);

    UsdGenExpressionDesc expr;
    expr.path = SdfPath("/Groom/Description/Expressions/lengthScale");
    expr.source = "ptex(\"lengthPaint\") * 0.25";
    UsdGenExpressionOutputDesc out;
    out.name = TfToken("result");
    out.nativeType = TfToken("float");
    out.shape = {expr::ScalarType::Float32, 1, 1, 1, 1, false};
    expr.outputs.push_back(out);
    UsdGenExpressionInputDesc in;
    in.name = TfToken("lengthPaint");
    in.targets.push_back(length.path);
    in.maps.push_back(length.path);
    expr.inputs.push_back(in);
    d.expressions.push_back(expr);

    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/Scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.surfaces = {scalp.path};
    scatter.params = {{TfToken("density"), VtValue(20.0), false}};
    d.nodes.push_back(scatter);

    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Grow");
    grow.type = TfToken("UsdGenGrow");
    grow.surfaces = {scalp.path};
    grow.inputs = {scatter.path};
    grow.params = {{TfToken("seed"), VtValue(19), false},
                   {TfToken("segments"), VtValue(5), false},
                   {TfToken("length"), VtValue(0.25f), false}};
    grow.maps = {width.path, ptex.path};
    UsdGenExpressionBinding binding;
    binding.expression = expr.path;
    binding.destination = TfToken("usdGen:length");
    binding.domain = expr::Domain::Primitive;
    binding.nativeType = TfToken("float");
    binding.destinationShape = {expr::ScalarType::Float32, 1, 1, 1, 1, false};
    binding.literal = VtValue(0.25f);
    grow.expressionBindings.push_back(binding);
    d.nodes.push_back(grow);
    d.terminal = grow.path;
    return d;
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

    // 1. The compiled snapshot resolves paint maps from both ref lists.
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    auto compiled = usdGenUsd::Compile(compiler, Desc(), &graph);
    for (auto const &e : compiled.errors) std::fprintf(stderr, "%s\n", e.c_str());
    CHECK(compiled.ok);
    auto snapshot = graph.RoutingSnapshot();
    CHECK(snapshot->nodes.size() == 2);
    UsdGenGraphRoutingNode const *grow = nullptr;
    for (auto const &node : snapshot->nodes)
        if (usdGenUsd::PathOf(usdGenUsd::Paths(*snapshot), node.key) == SdfPath("/Grow")) grow = &node;
    CHECK(grow != nullptr);
    CHECK(grow->paintRefs.size() == 2);
    bool sawLength = false, sawWidth = false;
    for (auto const &ref : grow->paintRefs) {
        CHECK(ref.surface == 0);
        sawLength |= (ref.primvar == TfToken("usdGen:paint:length"));
        sawWidth |= (ref.primvar == TfToken("usdGen:paint:width"));
    }
    CHECK(sawLength && sawWidth);

    // 2. Routing from the compiled snapshot: the length value-leaf dirty
    // (what the brush bake emits) re-captures the grow node as a map.
    usdGenImaging::UsdGenDirtyRouter router;
    router.Rebuild(*snapshot);
    UsdGenNodeId const growId = grow->id;
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:length"),
                                  TfToken("primvarValue")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.surfaceBits.empty());
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[growId] ==
              (UsdGenDirtyMap | UsdGenDirtyCapture));
    }

    // 3. The structural primvar locator (bake creating the primvar) too,
    // on the direct-slot map as well as the expression-input one.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:width")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[growId] ==
              (UsdGenDirtyMap | UsdGenDirtyCapture));
    }

    // 4. A bare coarse container dirty keeps the pre-existing
    // SurfacePoints route.
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

    // 5. An unrelated surface prim stays silent.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Other"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:length"),
                                  TfToken("primvarValue")})),
                     &pending);
        CHECK(!pending.Any());
    }

    // 6. An unrelated primvar on the bound surface stays silent.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),
                             Loc({TfToken("primvars"),
                                  TfToken("usdGen:paint:unknown"),
                                  TfToken("primvarValue")})),
                     &pending);
        CHECK(!pending.Any());
    }

    // 7. The sibling rows are intact: the direct slot's map-prim
    // container edits route DirtyMap, while the expression input's
    // map-prim edits route Capture through the geometry row.
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(
                         SdfPath("/Groom/Description/Maps/widthPaint"),
                         Loc({TfToken("usdGen"), TfToken("map")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[growId] == UsdGenDirtyMap);
    }
    {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(
                         SdfPath("/Groom/Description/Maps/lengthPaint"),
                         Loc({TfToken("usdGen"), TfToken("map")})),
                     &pending);
        CHECK(!pending.structural);
        CHECK(pending.nodeBits.size() == 1);
        CHECK(pending.nodeBits[growId] == UsdGenDirtyCapture);
    }

    for(auto const* field:{"subdivisionScheme","subdivisionTags","topology"}) {
        UsdGenPendingDirty pending;
        router.Route(Dirtied(SdfPath("/Scalp"),Loc({TfToken("mesh"),TfToken(field)})),&pending);
        CHECK(pending.surfaceBits.size()==1);
        CHECK(pending.surfaceBits[0]==UsdGenDirtySurfaceTopo);
    }
    std::printf("ok: paint and subdivision dirties re-capture consumers\n");
    return 0;
}
