// usdGen imaging — the live paint-preview overlay implementation.
#include "usdGenImaging/attributePreviewSceneIndex.h"

#include "usdGenHydraCurves/previewColor.h"

#include "pxr/base/tf/envSetting.h"
#include "pxr/base/tf/registryManager.h"
#include "pxr/base/tf/type.h"
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexPlugin.h"
#include "pxr/imaging/hd/sceneIndexPluginRegistry.h"

#include <algorithm>

PXR_NAMESPACE_OPEN_SCOPE

// The render-chain registration: every renderer gets one preview index after
// the groom index (plugInfo ordering), attached to the global registry the
// brush C ABI writes.
TF_DEFINE_ENV_SETTING(USDGEN_BRUSH_PREVIEW_ENABLE, true,
                      "Enable the usdGen attribute-brush preview overlay scene index.");

class UsdGenAttributePreviewSceneIndexPlugin : public HdSceneIndexPlugin {
public:
    UsdGenAttributePreviewSceneIndexPlugin() = default;

protected:
    HdSceneIndexBaseRefPtr _AppendSceneIndex(std::string const &,
                                             HdSceneIndexBaseRefPtr const &input,
                                             HdContainerDataSourceHandle const &) override
    {
        return usdGenImaging::UsdGenAttributePreviewSceneIndex::New(input, true);
    }
    bool _IsEnabled(HdContainerDataSourceHandle const &) const override
    {
        return TfGetEnvSetting(USDGEN_BRUSH_PREVIEW_ENABLE);
    }
};

TF_REGISTRY_FUNCTION(TfType)
{
    HdSceneIndexPluginRegistry::Define<UsdGenAttributePreviewSceneIndexPlugin>();
}
TF_REGISTRY_FUNCTION(HdSceneIndexPlugin)
{
    HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
        HdSceneIndexPluginRegistryTokens->allRenderers.GetString(),
        TfToken("UsdGenAttributePreviewSceneIndexPlugin"), nullptr, 0,
        HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
}

PXR_NAMESPACE_CLOSE_SCOPE

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

namespace {

// Above this face resolution the per-face mean is a GRID x GRID bilinear
// sample grid rather than every texel (see the header).
constexpr int kExactRes = 4;
constexpr int kGrid = 4;

GfVec3f FaceColor(usdGen::UsdGenAttributeMap const &map, int face, int channel,
                  TfToken const &colorMap, GfVec2f const &range)
{
    int const res = map.Resolution();
    double sum = 0.0;
    size_t n = 0;
    if (res <= kExactRes) {
        for (int t = 0; t < res; ++t) {
            for (int s = 0; s < res; ++s) {
                float v = 0.0f;
                if (map.GetTexel(face, s, t, channel, &v)) {
                    sum += v;
                    ++n;
                }
            }
        }
    } else {
        for (int t = 0; t < kGrid; ++t) {
            for (int s = 0; s < kGrid; ++s) {
                float v = 0.0f;
                if (map.Sample(face, (float(s) + 0.5f) / kGrid,
                               (float(t) + 0.5f) / kGrid, channel,
                               usdGen::UsdGenAttributeMapInterp::Bilinear, &v)) {
                    sum += v;
                    ++n;
                }
            }
        }
    }
    if (n == 0) return usdGen::UsdGenPreviewMissingColor();
    double const mean = sum / double(n);
    return usdGen::UsdGenPreviewColor(colorMap, range, &mean, 1);
}

// Built at library load, not on first use (no guarded function statics).
TfToken const s_uniform("uniform");
TfToken const s_faceVarying("faceVarying");
HdDataSourceLocator const s_structuralLocator(TfToken("primvars"), TfToken("displayColor"));
HdDataSourceLocator const s_valueLocator(TfToken("primvars"), TfToken("displayColor"),
                                         TfToken("primvarValue"));

HdContainerDataSourceHandle DisplayColorOverlay(VtVec3fArray const &colors,
                                                bool faceVarying)
{
    // HdOverlayContainerDataSource merges nested containers, so an upstream
    // INDEXED displayColor would keep its indexedPrimvarValue + indices, and
    // HdPrimvarSchema::IsIndexed (both present) makes the scene-delegate
    // adapter read those instead of our primvarValue. Block both.
    TfToken const names[] = {
        HdPrimvarSchemaTokens->primvarValue,
        HdPrimvarSchemaTokens->interpolation,
        HdPrimvarSchemaTokens->role,
        HdPrimvarSchemaTokens->indexedPrimvarValue,
        HdPrimvarSchemaTokens->indices,
    };
    HdDataSourceBaseHandle const values[] = {
        HdRetainedTypedSampledDataSource<VtVec3fArray>::New(colors),
        HdRetainedTypedSampledDataSource<TfToken>::New(faceVarying ? s_faceVarying : s_uniform),
        HdRetainedTypedSampledDataSource<TfToken>::New(TfToken("color")),
        HdBlockDataSource::New(),
        HdBlockDataSource::New(),
    };
    return HdRetainedContainerDataSource::New(
        TfToken("primvars"),
        HdRetainedContainerDataSource::New(
            TfToken("displayColor"),
            HdRetainedContainerDataSource::New(5, names, values)));
}

// Face-vertex count of an upstream mesh prim, or -1 when it is not a mesh
// (or states no topology).
int64_t FaceVertexCount(HdContainerDataSourceHandle const &prim)
{
    HdMeshSchema const mesh = HdMeshSchema::GetFromParent(prim);
    if (!mesh.IsDefined()) return -1;
    HdMeshTopologySchema const topology = mesh.GetTopology();
    if (!topology.IsDefined()) return -1;
    HdIntArrayDataSourceHandle const indices = topology.GetFaceVertexIndices();
    if (!indices) return -1;
    return int64_t(indices->GetTypedValue(0.0f).size());
}

HdDataSourceLocator const &StructuralLocator() { return s_structuralLocator; }
HdDataSourceLocator const &ValueLocator() { return s_valueLocator; }

}  // namespace

// ---------------------------------------------------------------------------
// UsdGenAttributePreviewRegistry
// ---------------------------------------------------------------------------

namespace {
std::atomic<UsdGenAttributePreviewRegistry *> s_previewRegistry{nullptr};
}

UsdGenAttributePreviewRegistry &UsdGenAttributePreviewRegistry::Get()
{
    return usdGen::UsdGenLazyLeaked(s_previewRegistry,
                            [] { return new UsdGenAttributePreviewRegistry(); });
}

bool UsdGenAttributePreviewRegistry::Set(SdfPath const &target, VtVec3fArray const &colors)
{
    if (target.IsEmpty() || colors.empty()) return false;
    bool structural = false;
    usdGen::UsdGenCowUpdate(_entries, [&](Entries &entries) {
        auto found = entries.find(target);
        if (found == entries.end()) {
            structural = true;
            entries.emplace(target, colors);
        } else {
            // A size change is an interpolation-level change for Hydra's
            // primvar descriptors: republish structurally.
            structural = found->second.size() != colors.size();
            found->second = colors;
        }
    });
    _Notify(target, structural);
    return true;
}

bool UsdGenAttributePreviewRegistry::Clear(SdfPath const &target)
{
    if (!usdGen::UsdGenCowUpdateIf(_entries, [&](Entries &entries) {
            return entries.erase(target) != 0;
        }))
        return false;
    _Notify(target, true);
    return true;
}

size_t UsdGenAttributePreviewRegistry::ClearAll()
{
    std::shared_ptr<Entries const> const cleared = _entries.Exchange(nullptr);
    if (!cleared) return 0;
    for (auto const &entry : *cleared) _Notify(entry.first, true);
    return cleared->size();
}

bool UsdGenAttributePreviewRegistry::Lookup(SdfPath const &target, VtVec3fArray *colors) const
{
    std::shared_ptr<Entries const> const entries = _entries.Load();
    if (!entries) return false;
    auto found = entries->find(target);
    if (found == entries->end()) return false;
    if (colors) *colors = found->second;
    return true;
}

size_t UsdGenAttributePreviewRegistry::EntryCount() const
{
    std::shared_ptr<Entries const> const entries = _entries.Load();
    return entries ? entries->size() : 0;
}

void UsdGenAttributePreviewRegistry::Attach(UsdGenAttributePreviewSceneIndex *index)
{
    // Called from the index constructor: weak, not strong, so the count the
    // TfCreateRefPtr in New() takes stays the first.
    TfWeakPtr<UsdGenAttributePreviewSceneIndex> weak(index);
    usdGen::UsdGenCowUpdateIf(_indices, [&](Indices &indices) {
        for (Attached const &a : indices)
            if (a.index == index) return false;
        indices.push_back({index, weak});
        return true;
    });
}

void UsdGenAttributePreviewRegistry::Detach(UsdGenAttributePreviewSceneIndex *index)
{
    usdGen::UsdGenCowUpdateIf(_indices, [&](Indices &indices) {
        auto const end = std::remove_if(indices.begin(), indices.end(),
            [&](Attached const &a) { return a.index == index; });
        if (end == indices.end()) return false;
        indices.erase(end, indices.end());
        return true;
    });
}

size_t UsdGenAttributePreviewRegistry::IndexCount() const
{
    std::shared_ptr<Indices const> const indices = _indices.Load();
    return indices ? indices->size() : 0;
}

void UsdGenAttributePreviewRegistry::_Notify(SdfPath const &target, bool structural)
{
    // The snapshot is immutable; an observer that attaches or destroys an
    // index while notified changes only later snapshots. A destroyed index
    // has an expired weak pointer and is skipped; a live one is held for the
    // call.
    std::shared_ptr<Indices const> const indices = _indices.Load();
    if (!indices) return;
    for (Attached const &a : *indices) {
        TfRefPtr<UsdGenAttributePreviewSceneIndex> const live =
            TfCreateRefPtrFromProtectedWeakPtr(a.weak);
        if (live) live->_NotifyRegistryChange(target, structural);
    }
}

// ---------------------------------------------------------------------------
// UsdGenAttributePreviewSceneIndex
// ---------------------------------------------------------------------------

UsdGenAttributePreviewSceneIndex::UsdGenAttributePreviewSceneIndex(
    HdSceneIndexBaseRefPtr const &input, bool attachGlobal)
    : HdSingleInputFilteringSceneIndexBase(input), _attached(attachGlobal)
{
    if (_attached) UsdGenAttributePreviewRegistry::Get().Attach(this);
}

UsdGenAttributePreviewSceneIndex::~UsdGenAttributePreviewSceneIndex()
{
    if (_attached) UsdGenAttributePreviewRegistry::Get().Detach(this);
}

void UsdGenAttributePreviewSceneIndex::_NotifyRegistryChange(SdfPath const &target,
                                                             bool structural)
{
    try {
        _SendPrimsDirtied({{target, HdDataSourceLocatorSet{
                                        structural ? StructuralLocator() : ValueLocator()}}});
    } catch (...) {
        TF_WARN("usdGen brush preview: a dirty observer threw");
    }
}

bool UsdGenAttributePreviewSceneIndex::_Store(SdfPath const &target, Entry entry)
{
    bool structural = false;
    usdGen::UsdGenCowUpdate(_previews, [&](Previews &previews) {
        auto found = previews.find(target);
        structural = found == previews.end() ||
                     found->second.faceVarying != entry.faceVarying ||
                     found->second.colors.size() != entry.colors.size();
        previews[target] = entry;  // copy: the edit may run again
    });
    // Structural the first time (the primvar appears), values-only after.
    _SendPrimsDirtied({{target, HdDataSourceLocatorSet{
                                    structural ? StructuralLocator() : ValueLocator()}}});
    return true;
}

bool UsdGenAttributePreviewSceneIndex::SetFaceVaryingPreview(SdfPath const &target,
                                                             VtVec3fArray const &colors)
{
    if (colors.empty()) return false;
    Entry entry;
    entry.colors = colors;
    entry.faceVarying = true;
    return _Store(target, std::move(entry));
}

bool UsdGenAttributePreviewSceneIndex::SetMapPreview(
    SdfPath const &target, std::shared_ptr<const usdGen::UsdGenAttributeMap> map,
    TfToken const &colorMap, GfVec2f const &range, int channel)
{
    if (!map || channel < 0 || channel >= map->Channels()) return false;
    Entry entry;
    entry.colors.resize(size_t(map->NumFaces()));
    for (int f = 0; f < map->NumFaces(); ++f)
        entry.colors[size_t(f)] = FaceColor(*map, f, channel, colorMap, range);
    return _Store(target, std::move(entry));
}

bool UsdGenAttributePreviewSceneIndex::SetGroomPreview(
    SdfPath const &target, std::shared_ptr<const usdGen::UsdGenAttributeMap> map,
    std::vector<int> const &rootFaces, std::vector<GfVec2f> const &rootUV,
    TfToken const &colorMap, GfVec2f const &range, int channel)
{
    if (!map || channel < 0 || channel >= map->Channels()) return false;
    if (rootFaces.empty() || rootFaces.size() != rootUV.size()) return false;
    Entry entry;
    entry.colors.resize(rootFaces.size());
    for (size_t i = 0; i < rootFaces.size(); ++i) {
        float v = 0.0f;
        if (!map->Sample(rootFaces[i], rootUV[i][0], rootUV[i][1], channel,
                         usdGen::UsdGenAttributeMapInterp::Bilinear, &v)) {
            entry.colors[i] = usdGen::UsdGenPreviewMissingColor();
            continue;
        }
        double const d = v;
        entry.colors[i] = usdGen::UsdGenPreviewColor(colorMap, range, &d, 1);
    }
    return _Store(target, std::move(entry));
}

bool UsdGenAttributePreviewSceneIndex::ClearPreview(SdfPath const &target)
{
    if (!usdGen::UsdGenCowUpdateIf(_previews, [&](Previews &previews) {
            return previews.erase(target) != 0;
        }))
        return false;
    _SendPrimsDirtied({{target, HdDataSourceLocatorSet{HdDataSourceLocator(
                                    TfToken("primvars"), TfToken("displayColor"))}}});
    return true;
}

void UsdGenAttributePreviewSceneIndex::ClearAllPreviews()
{
    std::shared_ptr<Previews const> const cleared = _previews.Exchange(nullptr);
    if (!cleared) return;
    HdSceneIndexObserver::DirtiedPrimEntries entries;
    for (auto const &kv : *cleared) {
        SdfPath const &target = kv.first;
        entries.push_back({target, HdDataSourceLocatorSet{HdDataSourceLocator(
                                        TfToken("primvars"), TfToken("displayColor"))}});
    }
    if (!entries.empty()) _SendPrimsDirtied(entries);
}

bool UsdGenAttributePreviewSceneIndex::HasPreview(SdfPath const &target) const
{
    std::shared_ptr<Previews const> const previews = _previews.Load();
    return previews && previews->find(target) != previews->end();
}

HdSceneIndexPrim UsdGenAttributePreviewSceneIndex::GetPrim(SdfPath const &path) const
{
    HdSceneIndexPrim prim = _GetInputSceneIndex()->GetPrim(path);
    if (!prim.dataSource) return prim;
    VtVec3fArray colors;
    bool faceVarying = false;
    bool found = false;
    if (std::shared_ptr<Previews const> const previews = _previews.Load()) {
        auto const local = previews->find(path);
        if (local != previews->end()) {
            colors = local->second.colors;
            faceVarying = local->second.faceVarying;
            found = true;
        }
    }
    if (!found && _attached &&
        UsdGenAttributePreviewRegistry::Get().Lookup(path, &colors)) {
        faceVarying = true;
        found = true;
    }
    if (!found) return prim;
    // A faceVarying overlay must match the mesh it lands on; a stale one
    // (topology edited since the Set) is withheld rather than mis-sized.
    if (faceVarying && FaceVertexCount(prim.dataSource) != int64_t(colors.size()))
        return prim;
    // Upstream first, overlay wins: every other primvar (and the rest of the
    // prim) shows through unchanged. The overlay sets the whole displayColor
    // triple (values, interpolation, role) and blocks the indexed pair
    // (indexedPrimvarValue, indices), so no upstream value field survives.
    prim.dataSource = HdOverlayContainerDataSource::New(
        DisplayColorOverlay(colors, faceVarying), prim.dataSource);
    return prim;
}

SdfPathVector UsdGenAttributePreviewSceneIndex::GetChildPrimPaths(
    SdfPath const &path) const
{
    return _GetInputSceneIndex()->GetChildPrimPaths(path);
}

}  // namespace usdGenImaging
