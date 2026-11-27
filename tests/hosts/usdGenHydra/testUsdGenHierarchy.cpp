// Hierarchy topology contract: composed reverse-sibling post-order is the
// entire operator stack.  Deliberately authored legacy edges below are
// malicious/no-op inputs and must not affect the derived graph.
#include "utils/pxrInterop.h"
#include "usdGenUsd/walk/usdGenGraphDescBuilderStage.h"
#include "usdGenUsd/lower.h"
#include "usdGenImaging/usdGenGraphDescBuilder.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/layer.h"
#include "pxr/usd/sdf/namespaceEdit.h"
#include "pxr/usd/sdf/primSpec.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"
#include "pxr/usdImaging/usdImaging/stageSceneIndex.h"
#include "pxr/imaging/hd/sceneIndexObserver.h"

#include <algorithm>
#include <cstdio>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

class _NoticeObserver final : public HdSceneIndexObserver
{
public:
    bool sawOpsOrDescription = false;
    std::vector<SdfPath> added;
    std::vector<SdfPath> removed;
    std::vector<SdfPath> dirtied;
    std::vector<SdfPath> renamedOld;
    std::vector<SdfPath> renamedNew;

    void Reset()
    {
        sawOpsOrDescription = false;
        added.clear();
        removed.clear();
        dirtied.clear();
        renamedOld.clear();
        renamedNew.clear();
    }

    void PrimsAdded(HdSceneIndexBase const &, AddedPrimEntries const &e) override
    { _Record(e, &added); }
    void PrimsRemoved(HdSceneIndexBase const &, RemovedPrimEntries const &e) override
    { _Record(e, &removed); }
    void PrimsDirtied(HdSceneIndexBase const &, DirtiedPrimEntries const &e) override
    { _Record(e, &dirtied); }
    void PrimsRenamed(HdSceneIndexBase const &, RenamedPrimEntries const &e) override
    {
        for (auto const &entry : e) {
            _RecordPath(entry.oldPrimPath);
            _RecordPath(entry.newPrimPath);
            renamedOld.push_back(entry.oldPrimPath);
            renamedNew.push_back(entry.newPrimPath);
        }
    }

private:
    template <class Entries>
    void _Record(Entries const &entries, std::vector<SdfPath> *paths)
    {
        for (auto const &entry : entries) {
            _RecordPath(entry.primPath);
            paths->push_back(entry.primPath);
        }
    }
    void _RecordPath(SdfPath const &path)
    {
        sawOpsOrDescription |= path.HasPrefix(SdfPath("/hair/Ops")) ||
                                 path == SdfPath("/hair");
    }
};

bool
_HasPath(std::vector<SdfPath> const &paths, SdfPath const &path)
{
    return std::find(paths.begin(), paths.end(), path) != paths.end();
}

bool
_HasRenameOrRemoveAdd(_NoticeObserver const &observer,
                      SdfPath const &oldPath, SdfPath const &newPath)
{
    return (_HasPath(observer.renamedOld, oldPath) &&
            _HasPath(observer.renamedNew, newPath)) ||
           (_HasPath(observer.removed, oldPath) &&
            _HasPath(observer.added, newPath));
}

}  // namespace

int
main()
{
    SdfLayerRefPtr layer = SdfLayer::CreateAnonymous("hierarchy.usda");
    if (!layer->ImportFromString(R"(#usda 1.0
def UsdGenDescription "hair"
{
    def Scope "Ops"
    {
        def UsdGenDeform "deform"
        {
            rel usdGen:input = </hair/Ops/width>
        }
        def UsdGenWidth "width" {}
        def Scope "style"
        {
            def UsdGenNoise "frizz" {}
            def UsdGenClump "clump" {}
        }
        def UsdGenGuideInterpolate "interpolate"
        {
            def UsdGenScatter "scatter" {}
        }
        def UsdGenNoise "inactive" ( active = false ) {}
    }
}
)") ) {
        return 1;
    }
    UsdStageRefPtr const stage = UsdStage::Open(layer);
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    UsdImagingSceneIndices const indices = UsdImagingCreateSceneIndices(info);
    auto buildHydra = [&]() {
        return usdGenImaging::BuildGraphDescFromHydra(
            *indices.finalSceneIndex, SdfPath("/hair"));
    };
    _NoticeObserver observer;
    HdSceneIndexObserverPtr const observerPtr = TfCreateWeakPtr(&observer);
    indices.finalSceneIndex->AddObserver(observerPtr);
    usdGen::UsdGenGraphDesc const graph =
        usdGenImaging::BuildGraphDescFromStage(
            stage, SdfPath("/hair"));
    usdGen::UsdGenGraphDesc const hydra = buildHydra();
    std::vector<SdfPath> const expected{
        SdfPath("/hair/Ops/interpolate/scatter"),
        SdfPath("/hair/Ops/interpolate"),
        SdfPath("/hair/Ops/style/clump"),
        SdfPath("/hair/Ops/style/frizz"),
        SdfPath("/hair/Ops/width"),
        SdfPath("/hair/Ops/deform"),
    };
    auto checkGraph = [&](usdGen::UsdGenGraphDesc const &candidate,
                          int failureBase) {
        if (candidate.nodes.size() != expected.size() ||
            candidate.terminal != expected.back()) {
            return failureBase;
        }
        for (size_t i = 0; i != expected.size(); ++i) {
            if (candidate.nodes[i].path != expected[i] ||
                candidate.nodes[i].type.IsEmpty() ||
                (i == 0 ? !candidate.nodes[i].inputs.empty()
                        : candidate.nodes[i].inputs !=
                              SdfPathVector{expected[i - 1]})) {
                std::fprintf(stderr, "hierarchy order mismatch at %zu\n", i);
                return failureBase + 1;
            }
        }
        return 0;
    };
    if (int const status = checkGraph(graph, 2)) {
        return status;
    }
    if (int const status = checkGraph(hydra, 4)) {
        return status;
    }
    // Composed `reorder nameChildren` must alter the derived stack; no
    // relationship edit is involved.  This uses the public Sdf API rather
    // than assuming stage-index child enumeration preserves the order.
    SdfPrimSpecHandle const ops = layer->GetPrimAtPath(SdfPath("/hair/Ops"));
    ops->SetNameChildrenOrder(TfTokenVector{
        TfToken("interpolate"), TfToken("style"), TfToken("width"),
        TfToken("deform"), TfToken("inactive")});
    indices.stageSceneIndex->ApplyPendingUpdates();
    usdGen::UsdGenGraphDesc const reordered =
        usdGenImaging::BuildGraphDescFromStage(stage, SdfPath("/hair"));
    usdGen::UsdGenGraphDesc const hydraReordered = buildHydra();
    if (reordered.nodes.size() != expected.size() ||
        reordered.nodes.front().path != SdfPath("/hair/Ops/deform") ||
        reordered.terminal != SdfPath("/hair/Ops/interpolate")) {
        return 6;
    }
    if (hydraReordered.nodes.size() != expected.size() ||
        hydraReordered.nodes.front().path != SdfPath("/hair/Ops/deform") ||
        hydraReordered.terminal != SdfPath("/hair/Ops/interpolate")) {
        return 7;
    }
    if (!observer.sawOpsOrDescription) {
        std::fprintf(stderr, "no live scene-index notice for hierarchy reorder\n");
        return 8;
    }

    // Add a nested Scope after the scene-index chain is live.  Scopes are
    // traversal-only: their children still participate in one reverse-sibling
    // post-order stream, and the live adapter must publish the new subtree
    // without rebuilding the chain.
    observer.Reset();
    SdfPrimSpecHandle const style = layer->GetPrimAtPath(
        SdfPath("/hair/Ops/style"));
    SdfPrimSpecHandle const nested = SdfPrimSpec::New(
        style, "nested", SdfSpecifierDef, "Scope");
    if (!nested || !SdfPrimSpec::New(
            nested, "low", SdfSpecifierDef, "UsdGenNoise") ||
        !SdfPrimSpec::New(
            nested, "high", SdfSpecifierDef, "UsdGenWidth")) {
        return 9;
    }
    indices.stageSceneIndex->ApplyPendingUpdates();
    usdGen::UsdGenGraphDesc const nestedGraph = buildHydra();
    std::vector<SdfPath> const nestedExpected{
        SdfPath("/hair/Ops/deform"),
        SdfPath("/hair/Ops/width"),
        SdfPath("/hair/Ops/style/nested/high"),
        SdfPath("/hair/Ops/style/nested/low"),
        SdfPath("/hair/Ops/style/clump"),
        SdfPath("/hair/Ops/style/frizz"),
        SdfPath("/hair/Ops/interpolate/scatter"),
        SdfPath("/hair/Ops/interpolate"),
    };
    if (nestedGraph.terminal != nestedExpected.back() ||
        nestedGraph.nodes.size() != nestedExpected.size()) {
        return 10;
    }
    for (size_t i = 0; i != nestedExpected.size(); ++i) {
        if (nestedGraph.nodes[i].path != nestedExpected[i] ||
            (i == 0 ? !nestedGraph.nodes[i].inputs.empty()
                    : nestedGraph.nodes[i].inputs !=
                          SdfPathVector{nestedExpected[i - 1]})) {
            std::fprintf(stderr, "nested hierarchy order mismatch at %zu\n", i);
            return 11;
        }
    }
    if (!_HasPath(observer.added, SdfPath("/hair/Ops/style/nested")) ||
        !_HasPath(observer.added,
                  SdfPath("/hair/Ops/style/nested/low")) ||
        !_HasPath(observer.added,
                  SdfPath("/hair/Ops/style/nested/high")) ||
        !observer.sawOpsOrDescription) {
        std::fprintf(stderr, "missing live nested Scope add notices\n");
        return 12;
    }

    // Reparent the whole Scope subtree in one namespace edit.  The same
    // scene-index chain must report the old/new namespace and recompute the
    // post-order with the Scope now a direct Ops child.
    observer.Reset();
    SdfBatchNamespaceEdit reparent;
    reparent.Add(SdfNamespaceEdit::Reparent(
        SdfPath("/hair/Ops/style/nested"), SdfPath("/hair/Ops"),
        SdfNamespaceEdit::AtEnd));
    if (!layer->Apply(reparent)) {
        std::fprintf(stderr, "failed to apply live Scope reparent\n");
        return 13;
    }
    indices.stageSceneIndex->ApplyPendingUpdates();
    usdGen::UsdGenGraphDesc const reparentedGraph = buildHydra();
    std::vector<SdfPath> const reparentedExpected{
        SdfPath("/hair/Ops/nested/high"),
        SdfPath("/hair/Ops/nested/low"),
        SdfPath("/hair/Ops/deform"),
        SdfPath("/hair/Ops/width"),
        SdfPath("/hair/Ops/style/clump"),
        SdfPath("/hair/Ops/style/frizz"),
        SdfPath("/hair/Ops/interpolate/scatter"),
        SdfPath("/hair/Ops/interpolate"),
    };
    if (reparentedGraph.terminal != reparentedExpected.back() ||
        reparentedGraph.nodes.size() != reparentedExpected.size()) {
        return 14;
    }
    for (size_t i = 0; i != reparentedExpected.size(); ++i) {
        if (reparentedGraph.nodes[i].path != reparentedExpected[i] ||
            (i == 0 ? !reparentedGraph.nodes[i].inputs.empty()
                    : reparentedGraph.nodes[i].inputs !=
                          SdfPathVector{reparentedExpected[i - 1]})) {
            std::fprintf(stderr,
                         "reparented hierarchy order mismatch at %zu\n", i);
            return 15;
        }
    }
    if (!_HasRenameOrRemoveAdd(
            observer, SdfPath("/hair/Ops/style/nested"),
            SdfPath("/hair/Ops/nested")) ||
        !_HasRenameOrRemoveAdd(
            observer, SdfPath("/hair/Ops/style/nested/high"),
            SdfPath("/hair/Ops/nested/high")) ||
        !_HasRenameOrRemoveAdd(
            observer, SdfPath("/hair/Ops/style/nested/low"),
            SdfPath("/hair/Ops/nested/low")) ||
        !observer.sawOpsOrDescription) {
        std::fprintf(stderr, "missing live Scope reparent notices\n");
        return 16;
    }
    indices.finalSceneIndex->RemoveObserver(observerPtr);
    return 0;
}
