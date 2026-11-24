// usdGen imaging — the live paint-preview overlay (app -> Hydra).
//
// While a brush stroke is in flight its Preview() map is shown in the
// viewport without touching the stage: the app pushes the live map into this
// filtering scene index, which overlays primvars/displayColor on the brushed
// surface (per-face colours) and on the groom (per-strand colours sampled at
// the strand roots), exactly as the capture loop will sample them. The stage
// still carries the last baked .ptx; the overlay is purely a Hydra opinion
// and vanishes on ClearPreview, so release-then-bake-then-swap never fights
// the interaction.
//
// One index serves every target: SetMapPreview per brushed surface,
// SetGroomPreview per groom. Colours are computed eagerly at Set time from an
// immutable snapshot (the stroke's Preview()/Clone()), through the same
// UsdGenPreviewColor maps the value preview uses, so the overlay and the
// preview agree by construction. Later mutation of the caller's map cannot
// tear the published colours.
//
// Two sources feed one index. The per-index Set* calls below serve the T1
// tests and in-process callers. The brush tool reaches the index from Python
// through the C ABI (usdGenBrushApi.h UsdGenBrush_PreviewSet/Clear), which
// writes the process-global UsdGenAttributePreviewRegistry; every index the
// UsdGenAttributePreviewSceneIndexPlugin puts in a render chain attaches to
// that registry on construction (detaches on destruction) and overlays its
// entries too. A per-index entry wins over a registry entry on the same path.
// The plugin registers for all renderers after the groom index (plugInfo
// tag usdGen:attributePreview; USDGEN_BRUSH_PREVIEW_ENABLE kill switch).
//
// Registry entries are faceVarying (four colours per quad, the corners the
// bake keeps), and GetPrim only applies one whose size matches the mesh's
// faceVertexIndices, so a stale overlay can never hand Storm a mis-sized
// primvar after a topology edit.
//
// Dirty discipline: the first Set on a target dirties primvars/displayColor
// (structural: the primvar appears); a re-Set dirties
// primvars/displayColor/primvarValue only (no interpolation change, so Hydra
// re-reads the values in place); ClearPreview dirties primvars/displayColor
// (the primvar goes away and upstream shows through again).
//
// Threading: the tables are copy-on-write maps published by atomic pointer
// (usdGen/sharedCell.h). GetPrim and the queries read them from any thread and
// never wait. Writers install by CAS. Set/Clear notices, index construction
// and index destruction belong to the frontend thread (as Hydra's own
// notices do); a notice holds a strong reference to each index it calls.
#ifndef USDGEN_IMAGING_ATTRIBUTE_PREVIEW_SCENE_INDEX_H
#define USDGEN_IMAGING_ATTRIBUTE_PREVIEW_SCENE_INDEX_H

#include "usdGen/sharedCell.h"
#include "usdGen/maps/attributeMap.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"
#include "pxr/imaging/hd/filteringSceneIndex.h"
#include "pxr/usd/sdf/path.h"

#include <map>
#include <memory>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

class UsdGenAttributePreviewSceneIndex : public HdSingleInputFilteringSceneIndexBase {
public:
    // attachGlobal: also serve the process-global registry's overlays (the
    // render-chain plugin passes true; tests choose).
    static TfRefPtr<UsdGenAttributePreviewSceneIndex> New(
        HdSceneIndexBaseRefPtr const &input, bool attachGlobal = false)
    {
        return TfCreateRefPtr(new UsdGenAttributePreviewSceneIndex(input, attachGlobal));
    }

    ~UsdGenAttributePreviewSceneIndex() override;

    // Registry callback: dirty `target`'s displayColor (structural, or
    // values only). Not for general use.
    void _NotifyRegistryChange(SdfPath const &target, bool structural);

    // Overlays `target` (a mesh) with one colour per map face: the face's
    // mean value over `channel`, through `colorMap`/`range`
    // (UsdGenPreviewColor). The mean is exact at res <= 4 and a 4x4 bilinear
    // grid above that — a 256-res scalp face has 65K texels, and the overlay
    // must stay inside the interaction budget. Returns false, leaving any
    // previous preview in place, for a null map or a channel outside the map.
    bool SetMapPreview(SdfPath const &target,
                       std::shared_ptr<const usdGen::UsdGenAttributeMap> map,
                       TfToken const &colorMap, GfVec2f const &range, int channel = 0);

    // Overlays `target` (curves) with one colour per strand root, sampled
    // bilinearly from `channel` at (rootFaces[i], rootUV[i]) exactly as the
    // capture loop samples. A root the map cannot sample shows the no-value
    // colour rather than failing the Set. Returns false, leaving any previous
    // preview in place, for a null map, a channel outside the map, mismatched
    // or empty root arrays.
    bool SetGroomPreview(SdfPath const &target,
                         std::shared_ptr<const usdGen::UsdGenAttributeMap> map,
                         std::vector<int> const &rootFaces,
                         std::vector<GfVec2f> const &rootUV, TfToken const &colorMap,
                         GfVec2f const &range, int channel = 0);

    // Overlays `target` (a mesh) with caller-computed faceVarying colours,
    // one per face-vertex. Returns false for an empty array.
    bool SetFaceVaryingPreview(SdfPath const &target, VtVec3fArray const &colors);

    // Removes the overlay; upstream shows through again. Returns false when
    // the target had no preview.
    bool ClearPreview(SdfPath const &target);
    void ClearAllPreviews();
    bool HasPreview(SdfPath const &target) const;

    HdSceneIndexPrim GetPrim(SdfPath const &path) const override;
    SdfPathVector GetChildPrimPaths(SdfPath const &path) const override;

protected:
    UsdGenAttributePreviewSceneIndex(HdSceneIndexBaseRefPtr const &input,
                                     bool attachGlobal);

    void _PrimsAdded(HdSceneIndexBase const &,
                     HdSceneIndexObserver::AddedPrimEntries const &entries) override
    {
        _SendPrimsAdded(entries);
    }
    void _PrimsRemoved(HdSceneIndexBase const &,
                       HdSceneIndexObserver::RemovedPrimEntries const &entries) override
    {
        _SendPrimsRemoved(entries);
    }
    void _PrimsDirtied(HdSceneIndexBase const &,
                       HdSceneIndexObserver::DirtiedPrimEntries const &entries) override
    {
        _SendPrimsDirtied(entries);
    }

private:
    struct Entry {
        VtVec3fArray colors;  // uniform: one per face / strand; or faceVarying
        bool faceVarying = false;
    };
    using Previews = std::map<SdfPath, Entry>;
    bool _Store(SdfPath const &target, Entry entry);
    usdGen::UsdGenSharedCell<Previews const> _previews;
    bool _attached = false;
};

// The process-global overlay table the brush C ABI writes. Deliberately
// leaked (like PomadeRegistry): scene indices may be destroyed during static
// destruction and must still be able to detach.
class UsdGenAttributePreviewRegistry {
public:
    static UsdGenAttributePreviewRegistry &Get();

    // faceVarying colours for `target`; false for an empty array. Notifies
    // every attached index (structural the first time the path appears).
    bool Set(SdfPath const &target, VtVec3fArray const &colors);
    bool Clear(SdfPath const &target);
    // Drop every overlay (stage replace, palette close); the count cleared.
    size_t ClearAll();
    bool Lookup(SdfPath const &target, VtVec3fArray *colors) const;
    size_t EntryCount() const;

    void Attach(UsdGenAttributePreviewSceneIndex *index);
    void Detach(UsdGenAttributePreviewSceneIndex *index);
    size_t IndexCount() const;

private:
    UsdGenAttributePreviewRegistry() = default;
    void _Notify(SdfPath const &target, bool structural);
    using Entries = std::map<SdfPath, VtVec3fArray>;
    usdGen::UsdGenSharedCell<Entries const> _entries;
    // Attached indices. The raw pointer is the identity Detach removes; the
    // weak pointer gives a notice a strong reference, or nothing once the
    // index is gone (an observer may destroy an index mid-notice).
    struct Attached {
        UsdGenAttributePreviewSceneIndex *index = nullptr;
        TfWeakPtr<UsdGenAttributePreviewSceneIndex> weak;
    };
    using Indices = std::vector<Attached>;
    usdGen::UsdGenSharedCell<Indices const> _indices;
};

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_ATTRIBUTE_PREVIEW_SCENE_INDEX_H
