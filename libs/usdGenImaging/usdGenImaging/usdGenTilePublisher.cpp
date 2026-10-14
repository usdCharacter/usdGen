// usdGen imaging — C2 tile publication -> Hydra retained data sources.
//
// Assembly contract: plan/06-imaging.md §4.1 (the C2 table; docs/freezes/C2.md
// carries the frozen field list). The engine fills EVERY field of
// UsdGenTilePublication at commit time (curveBuffer.h) — this file reads it
// and never writes to it (no hashing, no hairId derivation here).
#include "usdGenImaging/usdGenTilePublisher.h"

#include "usdGenImaging/usdGenTokens.h"

#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/trace/trace.h"
#include "pxr/imaging/hd/basisCurvesSchema.h"
#include "pxr/imaging/hd/basisCurvesTopologySchema.h"
#include "pxr/imaging/hd/dependencySchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/materialConnectionSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeParameterSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

namespace {

// All payload names are spelled as literal TfTokens: TfToken interning makes
// them identical to the corresponding HdTokens without pulling the whole
// token table into the diff of this contract file.
HdTokenDataSourceHandle
_Tok(TfToken const &t)
{
    return HdRetainedTypedSampledDataSource<TfToken>::New(t);
}

template <class T>
HdSampledDataSourceHandle
_Samp(T const &v)
{
    return HdRetainedTypedSampledDataSource<T>::New(v);
}

HdDataSourceBaseHandle
_Block()
{
    return HdDataSourceBaseHandle(HdBlockDataSource::New());
}

TfToken
_HydraMaterialPurpose(TfToken const &purpose)
{
    return purpose.IsEmpty() || purpose == TfToken("allPurpose")
        ? HdMaterialBindingsSchemaTokens->allPurpose : purpose;
}

// One primvars/<name> entry: { values, interpolation, type?, role?,
// elementSize? } (06 §4.1; interpolation NEVER "varying").
HdContainerDataSourceHandle
_Primvar(HdSampledDataSourceHandle const &values,
         TfToken const &interpolation,
         TfToken const &role = TfToken(),
         int elementSize = 0)
{
    HdPrimvarSchema::Builder b;
    b.SetPrimvarValue(values);
    if (!interpolation.IsEmpty()) {
        b.SetInterpolation(_Tok(interpolation));
    }
    if (!role.IsEmpty()) {
        b.SetRole(_Tok(role));
    }
    if (elementSize > 0) {
        b.SetElementSize(HdRetainedTypedSampledDataSource<int>::New(elementSize));
    }
    return b.Build();
}

void
_Add(std::vector<TfToken> *names, std::vector<HdDataSourceBaseHandle> *values,
     TfToken const &name, HdDataSourceBaseHandle const &ds)
{
    if (ds) {
        names->push_back(name);
        values->push_back(ds);
    }
}

HdContainerDataSourceHandle
_Container(std::vector<TfToken> &&names,
           std::vector<HdDataSourceBaseHandle> &&values)
{
    return HdRetainedContainerDataSource::New(
        names.size(), names.data(), values.data());
}

// A zero-copy vec3 view over a float-triple plane (furTauP/furTauN). GfVec3f
// is three contiguous floats, so the view reads exactly the bytes the pack
// memcpy used to copy, with no allocation and no copy. The holder keeps the
// source VtFloatArray (and through it the engine generation) alive until the
// last view dies; foreign arrays detach on write, so the engine buffer is
// never mutated through the view.
class _PlaneVec3Wrap : public Vt_ArrayForeignDataSource {
public:
    explicit _PlaneVec3Wrap(VtFloatArray const &src)
        : Vt_ArrayForeignDataSource(&_Detach), _src(src) {}
private:
    static void _Detach(Vt_ArrayForeignDataSource *self) {
        delete static_cast<_PlaneVec3Wrap *>(self);
    }
    VtFloatArray _src;
};

VtVec3fArray
_WrapVec3(VtFloatArray const &f, size_t n)
{
    static_assert(sizeof(GfVec3f) == 3 * sizeof(float),
                  "GfVec3f must be three contiguous floats for the vec3 view");
    static_assert(alignof(GfVec3f) == alignof(float),
                  "GfVec3f must share float alignment for the vec3 view");
    if (n == 0 || f.empty()) return VtVec3fArray();
    // Non-const pointer: the VtArray foreign-source constructor takes
    // ElementType*. Nothing writes through it (all readers are const, and a
    // foreign array detaches on write instead of writing through).
    auto *held = new _PlaneVec3Wrap(f);
    return VtVec3fArray(held,
        reinterpret_cast<GfVec3f *>(const_cast<float *>(f.data())), n);
}

HdContainerDataSourceHandle
_Topology(usdGen::UsdGenTilePublication const &tile)
{
    // C2 (docs/freezes/C2.md:20-23; 06 §4.1): the Hydra BasisCurves schema
    // nests EVERYTHING one level down — curveVertexCounts, type, basis and
    // wrap are all children of basisCurves/topology
    // (pxr/imaging/hd/basisCurvesTopologySchema.h:35-42; the consumer reads
    // them from that container in
    // hd/sceneIndexAdapterSceneDelegate.cpp:865-911). Anything authored as a
    // direct child of `basisCurves` is invisible to Hydra, which then falls
    // back to type = linear / basis = bezier / wrap = nonperiodic and the
    // tile can never render as a smooth cubic curve at any refineLevel.
    // Build through the schema builders so the token names cannot drift.
    // NOTE: basis carries the open/closed semantics; open is Hydra-exact.
    return HdBasisCurvesSchema::Builder()
        .SetTopology(
            HdBasisCurvesTopologySchema::Builder()
                .SetCurveVertexCounts(
                    HdRetainedTypedSampledDataSource<VtIntArray>::New(
                        tile.curveVertexCounts))
                .SetType(_Tok(HdTokens->cubic))
                .SetBasis(_Tok(tile.basis.empty() ? HdTokens->bezier
                                                  : TfToken(tile.basis)))
                .SetWrap(_Tok(HdTokens->pinned))
                .Build())
        .Build();
}

// size-derived interpolation for per-curve / per-CV colour arrays (06 §4.1
// displayColor row: uniform per curve, vertex per CV).
TfToken
_CurveOrVertexInterp(size_t n, size_t numCurves, size_t totalCvs,
                     char const *what, SdfPath const &path)
{
    if (n == numCurves) {
        return TfToken("uniform");
    }
    if (n == totalCvs) {
        return TfToken("vertex");
    }
    TF_CODING_ERROR("usdGen tile %s: primvars/%s has %zu entries, which is "
                    "neither one per curve (%zu) nor one per CV (%zu).",
                    path.GetText(), what, n, numCurves, totalCvs);
    return TfToken("uniform");
}

HdContainerDataSourceHandle
_Assemble(usdGen::UsdGenTilePublication const &tile, bool isGuide,
          TfToken const &guideRole, int64_t generation = -1)
{
    size_t totalCvs = 0;
    for (int c : tile.curveVertexCounts) {
        totalCvs += size_t(c);
    }
    size_t const numCurves = tile.curveVertexCounts.size();
    if (numCurves == 0) {
        // S27 / 06 §4: an empty basisCurves prim is NEVER published — the
        // engine drops empty tiles before Commit publishes them.
        TF_CODING_ERROR("usdGen tile %s: empty tile reached the publisher; "
                        "empty tiles must not be published (S27).",
                        tile.primPath.GetText());
        return nullptr;
    }
    if (tile.points.size() != totalCvs) {
        TF_CODING_ERROR("usdGen tile %s: points.size() (%zu) != sum of "
                        "curveVertexCounts (%zu).",
                        tile.primPath.GetText(),
                        size_t(tile.points.size()), totalCvs);
    }

    // --- primvars ---------------------------------------------------------
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;

    _Add(&pvNames, &pvValues, TfToken("points"),
         _Primvar(_Samp(tile.points), TfToken("vertex"), TfToken("point")));

    if (!tile.widths.empty()) {
        // widths: vertex, or CONSTANT for the single-tuple width attribute
        // — never "varying" (06 §4.1 widths row).
        _Add(&pvNames, &pvValues, TfToken("widths"),
             _Primvar(_Samp(tile.widths),
                      tile.widths.size() == 1 ? TfToken("constant")
                                              : TfToken("vertex")));
    }

    // C2 (docs/freezes/C2.md:35): constant 1.0. Storm claims
    // minScreenSpaceWidths as a builtin primvar name so it survives primvar
    // filtering (hdSt/basisCurves.cpp:1371-1385) and clamps the rasterized
    // strand to at least one pixel (hdSt/shaders/basisCurves.glslfx:781-784),
    // which is what stops thin hair aliasing away at distance.
    _Add(&pvNames, &pvValues, TfToken("minScreenSpaceWidths"),
         _Primvar(_Samp(1.0f), TfToken("constant")));

    if (!tile.hairT.empty()) {
        _Add(&pvNames, &pvValues, TfToken("hairT"),
             _Primvar(_Samp(tile.hairT), TfToken("vertex")));
    }
    if (!tile.hairId.empty()) {
        _Add(&pvNames, &pvValues, TfToken("hairId"),
             _Primvar(_Samp(tile.hairId), TfToken("uniform")));
    }
    if (!tile.st.empty()) {
        _Add(&pvNames, &pvValues, TfToken("st"),
             _Primvar(_Samp(tile.st), TfToken("uniform"),
                      TfToken("textureCoordinate")));
    }
    if (!tile.displayColor.empty()) {
        _Add(&pvNames, &pvValues, TfToken("displayColor"),
             _Primvar(_Samp(tile.displayColor),
                      _CurveOrVertexInterp(tile.displayColor.size(),
                                           numCurves, totalCvs,
                                           "displayColor", tile.primPath)));
    }
    if (!tile.bakeColor.empty()) {
        // Contract gap: UsdGenTilePublication carries the bake colours but
        // NOT the authored bakePrimvar NAME; published under "bakeColor".
        _Add(&pvNames, &pvValues, TfToken("bakeColor"),
             _Primvar(_Samp(tile.bakeColor),
                      _CurveOrVertexInterp(tile.bakeColor.size(),
                                           numCurves, totalCvs,
                                           "bakeColor", tile.primPath)));
    }

    // Engine-internal uniform planes (clumpId_<level>, guideIndex/guideWeight
    // with elementSize = arity 3, minScreenSpaceWidths constant 1.0).
    for (usdGen::UsdGenPlane const &plane : tile.extraUniform) {
        if (plane.name == TfToken("minScreenSpaceWidths")) {
            // Published unconditionally above; never publish it twice.
            continue;
        }
        HdSampledDataSourceHandle values;
        if (plane.name == "hairTipColor" && plane.arity == 3 &&
            plane.interpolation == "constant" && plane.f.size() == 3) {
            // The look's tip colour as a real color3f, not a float[3]: the
            // shader declares it `vec3` and Storm's codegen types the primvar
            // from the value, not from elementSize. See the cooker for why it
            // is published at all (a user-bound material otherwise gets the
            // Sdr default tips).
            VtVec3fArray packed(1, GfVec3f(plane.f[0], plane.f[1], plane.f[2]));
            _Add(&pvNames, &pvValues, plane.name,
                 _Primvar(_Samp(packed), plane.interpolation, TfToken("color")));
            continue;
        }
        if ((plane.name == "furTauP" || plane.name == "furTauN") &&
            plane.arity == 3 && plane.f.size() == totalCvs * 3) {
            // View optical depth as vec3 (zero-copy). Separate scalars
            // exhaust GL's per-stage SSBO slots on instanced curve draws.
            VtVec3fArray wrapped = _WrapVec3(plane.f, totalCvs);
            _Add(&pvNames, &pvValues, plane.name,
                 _Primvar(_Samp(wrapped), plane.interpolation));
            continue;
        }
        if (plane.type == "int") {
            values = _Samp(plane.i);
        } else {
            values = _Samp(plane.f);
        }
        _Add(&pvNames, &pvValues, plane.name,
             _Primvar(values, plane.interpolation, TfToken(), plane.arity));
    }

    // Motion (06 §4.1 velocities row): real data in P1, an HdBlockDataSource
    // on P0/P2 so an authored primvar on the C3 source cannot leak through.
    if (!tile.velocities.empty()) {
        _Add(&pvNames, &pvValues, TfToken("velocities"),
             _Primvar(_Samp(tile.velocities), TfToken("vertex")));
        _Add(&pvNames, &pvValues, TfToken("accelerations"), _Block());
    } else {
        _Add(&pvNames, &pvValues, TfToken("velocities"), _Block());
        _Add(&pvNames, &pvValues, TfToken("accelerations"), _Block());
    }

    if (isGuide) {
        // 06 §4.2: guides carry primvars/usdGen:role = guide.
        _Add(&pvNames, &pvValues, TfToken("usdGen:role"),
             _Primvar(_Tok(guideRole), TfToken("constant")));
    }
    // NO primvars/normals, NO primvars/hairTangent (06 §4.1: tangent is a
    // Storm extension opt-in, never default-on).

    // --- the rest of the prim container ------------------------------------
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;

    _Add(&names, &values, TfToken("basisCurves"), _Topology(tile));
    // StormSurgery generation stamp (P0 race contract): the publication
    // snapshot id, threaded as a parameter (the engine tile carries no
    // generation field — libs/usdGen, not this lane). -1 = unstamped.
    if (generation >= 0) {
        _Add(&names, &values, TfToken("generation"), _Samp((int)generation));
    }
    _Add(&names, &values, TfToken("primvars"),
         _Container(std::move(pvNames), std::move(pvValues)));

    _Add(&names, &values, TfToken("extent"),
         _Container(
             {TfToken("min"), TfToken("max")},
             {HdDataSourceBaseHandle(_Samp(tile.extentMin)),
              HdDataSourceBaseHandle(_Samp(tile.extentMax))}));

    // UsdImaging xform semantics: our own matrix must REPLACE the inherited
    // surface chain (02 §1.7), hence resetXformStack.
    _Add(&names, &values, TfToken("xform"),
         HdXformSchema::Builder()
             .SetMatrix(HdRetainedTypedSampledDataSource<GfMatrix4d>::New(
                 tile.xformMatrix))
             .SetResetXformStack(
                 HdRetainedTypedSampledDataSource<bool>::New(true))
             .Build());

    // Storm P0 (06 §4.1): the publication's refineLevel (engine default 2);
    // tessellation stays at the Osd default (S14: never USDGEN_TESSELLATION_LEVEL).
    // Storm P0 (06 §4.1): the publication's refineLevel (engine default 2);
    // tessellation stays at the Osd default (S14: never USDGEN_TESSELLATION_LEVEL).
    _Add(&names, &values, TfToken("displayStyle"),
         _Container({TfToken("refineLevel")},
                    {HdDataSourceBaseHandle(_Samp(tile.refineLevel))}));

    // Purpose: publish ONLY when authored. An empty purpose omits the
    // container, which Hydra resolves to the geometry render tag
    // (HdSceneIndexAdapterSceneDelegate::GetRenderTag falls back to geometry
    // iff the container is absent/empty). Publishing purpose="default" is
    // FATAL — "default" is not a render tag, so the tile matches no
    // collection and Storm never syncs it (2026-09-12: 49 published tiles,
    // itemsDrawn == 1, zero HdStBasisCurves sync lines). Authored values
    // (render/guide/proxy) pass through verbatim — they ARE render tags.
    if (!tile.purpose.IsEmpty()) {
        _Add(&names, &values, TfToken("purpose"),
             _Container({TfToken("purpose")},
                        {HdDataSourceBaseHandle(_Tok(tile.purpose))}));
    }
    _Add(&names, &values, TfToken("visibility"),
         HdVisibilitySchema::Builder()
             .SetVisibility(HdRetainedTypedSampledDataSource<bool>::New(
                 tile.visibility != TfToken("invisible")))
             .Build());

    {
        // An AUTHORED binding always wins. With nothing authored the tile
        // used to carry no materialBindings at all, so Storm fell back to
        // flat displayColor shading and the hair lobes never ran; bind the
        // synthetic <description>/__usdGenRender/material_storm instead
        // (C2 06 §4.4 slot, synthesized by UsdGenGroomSceneIndex::GetPrim).
        bool const authored = !tile.materialPath.IsEmpty();
        SdfPath const path = authored
            ? tile.materialPath
            : UsdGenTilePublisher::DefaultMaterialPath(tile.primPath);
        // The app-facing allPurpose spelling maps to Hydra's empty-token
        // default binding child; the value is a MaterialBindingSchema.
        HdDataSourceBaseHandle binding = HdMaterialBindingSchema::Builder()
            .SetPath(HdRetainedTypedSampledDataSource<SdfPath>::New(path))
            .Build();
        _Add(&names, &values, TfToken("materialBindings"),
             _Container({_HydraMaterialPurpose(
                             authored ? tile.materialPurpose : TfToken())},
                        {binding}));
    }
    if (!tile.primOrigin.IsEmpty()) {
        _Add(&names, &values, TfToken("primOrigin"),
             _Container({TfToken("scenePath")},
                        {HdDataSourceBaseHandle(_Samp(tile.primOrigin))}));
    }
    if (!tile.dependencySurface.IsEmpty()) {
        // Exactly ONE __dependencies entry: the bound surface's points feed
        // our points (02 §1.7; HdDependencySchema field names verified).
        HdDataSourceLocator const pointsLoc(
            TfToken("primvars"), TfToken("points"));
        HdContainerDataSourceHandle dep =
            HdDependencySchema::Builder()
                .SetDependedOnPrimPath(
                    HdRetainedTypedSampledDataSource<SdfPath>::New(
                        tile.dependencySurface))
                .SetDependedOnDataSourceLocator(
                    HdRetainedTypedSampledDataSource<HdDataSourceLocator>::New(
                        pointsLoc))
                .SetAffectedDataSourceLocator(
                    HdRetainedTypedSampledDataSource<HdDataSourceLocator>::New(
                        pointsLoc))
                .Build();
        _Add(&names, &values, TfToken("__dependencies"),
             _Container({TfToken("usdGenSurface")},
                        {HdDataSourceBaseHandle(dep)}));
    }

    return _Container(std::move(names), std::move(values));
}

}  // namespace

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildTileDataSource(
    usdGen::UsdGenTilePublication const &tile)
{
    return _Assemble(tile, /*isGuide=*/false, TfToken());
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildTileDataSource(
    usdGen::UsdGenTilePublication const &tile, int64_t generation)
{
    TRACE_FUNCTION();
    return _Assemble(tile, /*isGuide=*/false, TfToken(), generation);
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildGuideDataSource(
    usdGen::UsdGenTilePublication const &guide,
    std::string const &setName)
{
    (void)setName;  // the set name lives in the prim path (GuidePath); the
                    // payload marker is primvars/usdGen:role = guide.
    return _Assemble(guide, /*isGuide=*/true, TfToken("guide"));
}

/*static*/
UsdGenTilePublisher::TileNotices
UsdGenTilePublisher::NoticesFor(usdGen::UsdGenTileDirty const &tileDirty)
{
    TileNotices n;
    n.pointsDirty = tileDirty.pointsDirty;
    n.widthsDirty = tileDirty.widthsDirty;
    n.xformDirty = tileDirty.xformDirty;
    n.newPrimvarLocators.reserve(tileDirty.newPrimvars.size());
    for (TfToken const &name : tileDirty.newPrimvars) {
        // A NEW primvar dirties primvars/<name> ONCE — never /primvarValue
        // (06 §5.1: the container-level dirty makes the client build it).
        n.newPrimvarLocators.emplace_back(TfToken("primvars"), name);
    }
    n.dirtyPrimvarLocators.reserve(tileDirty.dirtyPrimvars.size());
    for (TfToken const &name : tileDirty.dirtyPrimvars) {
        n.dirtyPrimvarLocators.emplace_back(
            TfToken("primvars"), name, TfToken("primvarValue"));
    }
    return n;
}

/*static*/
std::vector<HdDataSourceLocator>
UsdGenTilePublisher::TileNotices::all() const
{
    std::vector<HdDataSourceLocator> out;
    if (pointsDirty) {
        // Bare leaf + the two extent leaves; the container sentinel of
        // ComputeExtents is never used (06 §5.1).
        out.emplace_back(TfToken("primvars"), TfToken("points"),
                         TfToken("primvarValue"));
        out.emplace_back(TfToken("extent"), TfToken("min"));
        out.emplace_back(TfToken("extent"), TfToken("max"));
    }
    if (widthsDirty) {
        out.emplace_back(TfToken("primvars"), TfToken("widths"),
                         TfToken("primvarValue"));
    }
    if (xformDirty) {
        out.emplace_back(TfToken("xform"), TfToken("matrix"));
    }
    out.insert(out.end(), newPrimvarLocators.begin(), newPrimvarLocators.end());
    out.insert(out.end(), dirtyPrimvarLocators.begin(),
               dirtyPrimvarLocators.end());
    return out;
}

/*static*/
SdfPath
UsdGenTilePublisher::TilePath(
    SdfPath const &descriptionPath, usdGen::UsdGenTileId tile)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "tile_%04u", unsigned(tile));
    return descriptionPath.AppendChild(RenderNamespace())
        .AppendChild(TfToken(buf));
}

/*static*/
SdfPath
UsdGenTilePublisher::GuidePath(
    SdfPath const &descriptionPath, TfToken const &setName)
{
    return descriptionPath.AppendChild(RenderNamespace())
        .AppendChild(TfToken("guides"))
        .AppendChild(setName);
}

/*static*/
SdfPath
UsdGenTilePublisher::MaterialPath(SdfPath const &descriptionPath)
{
    return descriptionPath.AppendChild(RenderNamespace())
        .AppendChild(MaterialName());
}

/*static*/
SdfPath
UsdGenTilePublisher::DefaultMaterialPath(SdfPath const &tilePath)
{
    // tilePath is <description>/__usdGenRender/tile_NNNN; the material is its
    // sibling.
    return tilePath.GetParentPath().AppendChild(MaterialName());
}

/*static*/
SdfPath
UsdGenTilePublisher::PreviewMaterialPath(SdfPath const &descriptionPath, bool flat)
{
    static TfToken const lit("material_preview"), flatName("material_preview_flat");
    return descriptionPath.AppendChild(RenderNamespace())
        .AppendChild(flat ? flatName : lit);
}

/*static*/
bool
UsdGenTilePublisher::IsPreviewMaterialPath(SdfPath const &descriptionPath,
                                           SdfPath const &path, bool *flat)
{
    for (bool const candidate : {false, true}) {
        if (path == PreviewMaterialPath(descriptionPath, candidate)) {
            if (flat) *flat = candidate;
            return true;
        }
    }
    return false;
}

namespace {

// One `surface`-terminal network around one `surface` node.
HdContainerDataSourceHandle
_Network(TfToken const &identifier, HdContainerDataSourceHandle const &parameters,
         HdContainerDataSourceHandle const &inputConnections)
{
    static TfToken const surface("surface");
    HdContainerDataSourceHandle const node =
        HdMaterialNodeSchema::Builder()
            .SetNodeIdentifier(_Tok(identifier))
            .SetParameters(parameters)
            .SetInputConnections(inputConnections)
            .Build();
    HdContainerDataSourceHandle const terminal =
        HdMaterialConnectionSchema::Builder()
            .SetUpstreamNodePath(_Tok(surface))
            .SetUpstreamNodeOutputName(_Tok(surface))
            .Build();
    return HdMaterialNetworkSchema::Builder()
        .SetNodes(HdRetainedContainerDataSource::New(surface, node))
        .SetTerminals(
            HdRetainedContainerDataSource::New(surface, terminal))
        .Build();
}

HdDataSourceBaseHandle
_Param(float value)
{
    return HdMaterialNodeParameterSchema::Builder()
        .SetValue(HdRetainedTypedSampledDataSource<float>::New(value))
        .Build();
}

HdDataSourceBaseHandle
_Param(GfVec3f const &value)
{
    return HdMaterialNodeParameterSchema::Builder()
        .SetValue(HdRetainedTypedSampledDataSource<GfVec3f>::New(value))
        .Build();
}

HdDataSourceBaseHandle
_Param(TfToken const &value)
{
    return HdMaterialNodeParameterSchema::Builder()
        .SetValue(HdRetainedTypedSampledDataSource<TfToken>::New(value))
        .Build();
}

// The universal-context fallback network: a UsdPreviewSurface whose
// diffuseColor reads the per-curve displayColor the tiles bake, falling
// back to `rootFallback` where the prim carries none. hdMoonray reads the
// universal context when no moonray context exists; a glslfx-only node
// there makes it look for an RDL DSO of that name, so the fallback is
// what keeps synthetic materials rendering outside Storm
// (docs/moonray-fur.md). Roughness is the same constant the fur
// generator's fallback uses.
HdContainerDataSourceHandle
_PreviewNetwork(GfVec3f const &rootFallback)
{
    static TfToken const surface("surface");
    static TfToken const reader("DisplayColor");
    static TfToken const result("result");
    HdDataSourceBaseHandle const connection =
        HdMaterialConnectionSchema::Builder()
            .SetUpstreamNodePath(_Tok(reader))
            .SetUpstreamNodeOutputName(_Tok(result))
            .Build();
    HdDataSourceBaseHandle const connectionVector =
        HdRetainedSmallVectorDataSource::New(1, &connection);
    HdContainerDataSourceHandle const preview =
        HdMaterialNodeSchema::Builder()
            .SetNodeIdentifier(_Tok(TfToken("UsdPreviewSurface")))
            .SetParameters(HdRetainedContainerDataSource::New(
                TfToken("roughness"), _Param(0.6f)))
            .SetInputConnections(HdRetainedContainerDataSource::New(
                TfToken("diffuseColor"), connectionVector))
            .Build();
    TfToken const readerNames[] = {TfToken("varname"), TfToken("fallback")};
    HdDataSourceBaseHandle const readerValues[] = {
        _Param(TfToken("displayColor")), _Param(rootFallback)};
    HdContainerDataSourceHandle const readerNode =
        HdMaterialNodeSchema::Builder()
            .SetNodeIdentifier(_Tok(TfToken("UsdPrimvarReader_float3")))
            .SetParameters(HdRetainedContainerDataSource::New(
                2, readerNames, readerValues))
            .SetInputConnections(HdRetainedContainerDataSource::New())
            .Build();
    HdContainerDataSourceHandle const terminal =
        HdMaterialConnectionSchema::Builder()
            .SetUpstreamNodePath(_Tok(surface))
            .SetUpstreamNodeOutputName(_Tok(surface))
            .Build();
    TfToken const nodeNames[] = {surface, reader};
    HdDataSourceBaseHandle const nodeValues[] = {preview, readerNode};
    return HdMaterialNetworkSchema::Builder()
        .SetNodes(HdRetainedContainerDataSource::New(2, nodeNames, nodeValues))
        .SetTerminals(
            HdRetainedContainerDataSource::New(surface, terminal))
        .Build();
}

// A material with both networks: the identifier's shading under the
// `glslfx` context Storm prefers, the preview under the universal one
// every other delegate falls back to. Storm's pixels are unchanged: it
// reads the same node and parameters as before, from its own context.
HdContainerDataSourceHandle
_SurfaceMaterial(TfToken const &identifier, HdContainerDataSourceHandle const &parameters,
                 GfVec3f const &previewFallback)
{
    HdContainerDataSourceHandle const shaded = _Network(
        identifier, parameters, HdRetainedContainerDataSource::New());
    HdContainerDataSourceHandle const preview = _PreviewNetwork(previewFallback);
    TfToken const contextNames[] = {HdMaterialSchemaTokens->universalRenderContext,
                                    TfToken("glslfx")};
    HdDataSourceBaseHandle const contextValues[] = {preview, shaded};
    return HdRetainedContainerDataSource::New(
        HdMaterialSchemaTokens->material,
        HdMaterialSchema::BuildRetained(2, contextNames, contextValues));
}

// A two-colour approximation of the look's ramp: the authored root/tip pair,
// or the first and last stop when a multi-stop usdGen:look:colorRamp exists
// (the shader has one lerp, not a ramp). Stop ORDER is the authored order;
// usdGen:look:colorRamp:positions is required to be ascending (C1 §2.12).
void
_LookColors(usdGen::UsdGenLookDesc const &look, GfVec3f *root, GfVec3f *tip)
{
    *root = look.rootColor;
    *tip = look.tipColor;
    if (look.rampColors.size() >= 2) {
        *root = look.rampColors[0];
        *tip = look.rampColors[look.rampColors.size() - 1];
    }
}

void
_MixBits(uint64_t *h, uint64_t v)
{
    // splitmix64's finalizer: cheap, and every input bit reaches every output
    // bit, which is what a change-detection digest needs.
    *h ^= v + 0x9e3779b97f4a7c15ull + (*h << 6) + (*h >> 2);
    *h ^= *h >> 30; *h *= 0xbf58476d1ce4e5b9ull;
    *h ^= *h >> 27; *h *= 0x94d049bb133111ebull;
    *h ^= *h >> 31;
}

void
_MixFloat(uint64_t *h, float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    _MixBits(h, bits);
}

}  // namespace

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildDefaultMaterialDataSource()
{
    // Parameters are left unset so the shader def's Sdr defaults apply. This
    // overload exists for callers with no description in hand (the C2 contract
    // test); the groom scene index always uses the look-carrying one. The
    // preview fallback is the default look's root, as it would be there.
    usdGen::UsdGenLookDesc const look;
    return _SurfaceMaterial(DefaultMaterialIdentifier(),
                            HdRetainedContainerDataSource::New(), look.rootColor);
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildDefaultMaterialDataSource(
    usdGen::UsdGenLookDesc const &look)
{
    GfVec3f root, tip;
    _LookColors(look, &root, &tip);

    // baseColor is only consulted where the prim carries no displayColor
    // (usdGen:look:bakeTarget "none"); the tile otherwise bakes rootColor into
    // displayColor per curve and the shader reads that as the root albedo.
    // Authoring it anyway keeps the two routes agreeing.
    TfToken const names[] = {
        TfToken("baseColor"), TfToken("tipColor"), TfToken("colorRamp"),
        TfToken("randomHue"), TfToken("randomValue"),
    };
    HdDataSourceBaseHandle const values[] = {
        _Param(root), _Param(tip), _Param(std::max(look.rampExponent, 0.001f)),
        _Param(std::max(look.hueJitter, 0.0f)),
        _Param(std::max(look.valueJitter, 0.0f)),
    };
    return _SurfaceMaterial(
        DefaultMaterialIdentifier(),
        HdRetainedContainerDataSource::New(
            sizeof(names) / sizeof(names[0]), names, values),
        root);
}

/*static*/
TfToken const &
UsdGenTilePublisher::DefaultMaterialIdentifier()
{
    // The OPAQUE variant. The two differ only in materialTag, and the choice
    // is between two failure modes of stock Storm, both measured on
    // examples/head-hair-closeup.usda at 1280x960 (docs/storm-fur.md):
    //   * translucent/OIT composites the host renderer coverage exactly, but Storm's OIT
    //     pool is 8 * width * height fragments for the WHOLE frame, handed out
    //     by one atomic counter, and a fragment past the end is silently
    //     dropped (hdx/shaders/renderPass.glslfx RenderOutputImpl). This groom
    //     exhausts it and loses the crown; at a quarter of the density it does
    //     not. Whole regions of hair disappear, in draw order. It also costs
    //     18.8 ms against 12.7 ms.
    //   * defaultMaterialTag's alpha-to-coverage has no such cliff. Plain
    //     alpha-to-coverage cannot composite sub-pixel hair either, because
    //     its sample mask comes from the alpha value and the pixel position
    //     alone, so the glslfx snaps alpha against a per-strand hash instead
    //     and recovers 1-(1-a)^N in expectation.
    // A groom is exactly the case the OIT budget cannot take, so the default
    // is the opaque one. Bind UsdGenHairStrandsTranslucent by hand for a hero
    // shot of a groom that fits the budget: it is smoother, with no dither.
    static TfToken const identifier("UsdGenHairStrands");
    return identifier;
}

/*static*/
uint64_t
UsdGenTilePublisher::DefaultMaterialLookDigest(
    usdGen::UsdGenLookDesc const &look)
{
    GfVec3f root, tip;
    _LookColors(look, &root, &tip);
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (int i = 0; i < 3; ++i) { _MixFloat(&h, root[i]); _MixFloat(&h, tip[i]); }
    _MixFloat(&h, look.rampExponent);
    _MixFloat(&h, look.hueJitter);
    _MixFloat(&h, look.valueJitter);
    return h;
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildPreviewMaterialDataSource(bool flat)
{
    static TfToken const shading("shading");
    HdDataSourceBaseHandle const value = HdMaterialNodeParameterSchema::Builder()
        .SetValue(HdRetainedTypedSampledDataSource<float>::New(flat ? 0.0f : 1.0f))
        .Build();
    // The preview mode always repaints displayColor, so the fallback only
    // shows where the prim carries none; mid-gray is deliberately neutral.
    return _SurfaceMaterial(TfToken("UsdGenValuePreview"),
                            HdRetainedContainerDataSource::New(shading, value),
                            GfVec3f(0.5f));
}

/*static*/
SdfPath
UsdGenTilePublisher::ScalpShadowPath(SdfPath const &descriptionPath)
{
    static TfToken const name("scalpShadow");
    return descriptionPath.AppendChild(RenderNamespace()).AppendChild(name);
}

/*static*/
SdfPath
UsdGenTilePublisher::ScalpShadowMaterialPath(SdfPath const &descriptionPath)
{
    static TfToken const name("material_scalpShadow");
    return descriptionPath.AppendChild(RenderNamespace()).AppendChild(name);
}

/*static*/
TfToken const &
UsdGenTilePublisher::ScalpShadowIdentifier()
{
    static TfToken const identifier("UsdGenScalpShadow");
    return identifier;
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildScalpShadowDataSource(
    usdGen::UsdGenScalpShadowPublication const &cap, int64_t generation)
{
    TRACE_FUNCTION();
    if (cap.IsEmpty()) return nullptr;
    size_t const points = cap.points.size();

    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;

    // Hydra's Mesh schema nests topology the way BasisCurves does. The cap is
    // doubleSided so a left-handed emitter cannot cull it away: nothing about
    // it depends on which way it faces.
    _Add(&names, &values, TfToken("mesh"),
         _Container(
             {TfToken("topology"), TfToken("doubleSided")},
             {HdDataSourceBaseHandle(_Container(
                  {TfToken("faceVertexCounts"), TfToken("faceVertexIndices"),
                   TfToken("orientation")},
                  {HdDataSourceBaseHandle(_Samp(cap.faceVertexCounts)),
                   HdDataSourceBaseHandle(_Samp(cap.faceVertexIndices)),
                   HdDataSourceBaseHandle(_Tok(TfToken("rightHanded")))})),
              HdDataSourceBaseHandle(
                  HdRetainedTypedSampledDataSource<bool>::New(true))}));

    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, TfToken("points"),
         _Primvar(_Samp(cap.points), TfToken("vertex"), TfToken("point")));
    if (cap.normals.size() == points) {
        _Add(&pvNames, &pvValues, TfToken("normals"),
             _Primvar(_Samp(cap.normals), TfToken("vertex"), TfToken("normal")));
    }
    for (usdGen::UsdGenPlane const &plane : cap.extraUniform) {
        if (plane.arity != 3 || plane.f.size() != points * 3) continue;
        // The same vec3 view a tile's depths get: separate scalar buffers
        // exhaust GL's per-stage SSBO slots.
        VtVec3fArray wrapped = _WrapVec3(plane.f, points);
        _Add(&pvNames, &pvValues, plane.name,
             _Primvar(_Samp(wrapped), plane.interpolation));
    }
    _Add(&names, &values, TfToken("primvars"),
         _Container(std::move(pvNames), std::move(pvValues)));

    _Add(&names, &values, TfToken("extent"),
         _Container({TfToken("min"), TfToken("max")},
                    {HdDataSourceBaseHandle(_Samp(cap.extentMin)),
                     HdDataSourceBaseHandle(_Samp(cap.extentMax))}));

    // Points are already world space, so the inherited surface chain must not
    // apply again — the same reason a tile resets the stack.
    _Add(&names, &values, TfToken("xform"),
         HdXformSchema::Builder()
             .SetMatrix(HdRetainedTypedSampledDataSource<GfMatrix4d>::New(
                 GfMatrix4d(1.0)))
             .SetResetXformStack(
                 HdRetainedTypedSampledDataSource<bool>::New(true))
             .Build());

    if (!cap.purpose.IsEmpty()) {
        _Add(&names, &values, TfToken("purpose"),
             _Container({TfToken("purpose")},
                        {HdDataSourceBaseHandle(_Tok(cap.purpose))}));
    }
    _Add(&names, &values, TfToken("visibility"),
         HdVisibilitySchema::Builder()
             .SetVisibility(HdRetainedTypedSampledDataSource<bool>::New(
                 cap.visibility != TfToken("invisible")))
             .Build());

    HdDataSourceBaseHandle binding = HdMaterialBindingSchema::Builder()
        .SetPath(HdRetainedTypedSampledDataSource<SdfPath>::New(cap.materialPath))
        .Build();
    _Add(&names, &values, TfToken("materialBindings"),
         _Container({_HydraMaterialPurpose(TfToken())}, {binding}));

    _Add(&names, &values, TfToken("generation"), _Samp(generation));
    return _Container(std::move(names), std::move(values));
}

/*static*/
HdContainerDataSourceHandle
UsdGenTilePublisher::BuildScalpShadowMaterialDataSource(
    usdGen::UsdGenLookDesc const &look)
{
    static TfToken const baseColor("baseColor");
    // The transmitted light is tinted by what the hair absorbs, so the cap
    // needs the same root albedo the strands use.
    GfVec3f root = look.rootColor;
    if (!look.rampColors.empty()) root = look.rampColors.front();
    TfToken const names[] = {baseColor};
    HdDataSourceBaseHandle const values[] = {_Param(root)};
    return _SurfaceMaterial(
        ScalpShadowIdentifier(),
        HdRetainedContainerDataSource::New(1, names, values), root);
}

}  // namespace usdGenImaging
