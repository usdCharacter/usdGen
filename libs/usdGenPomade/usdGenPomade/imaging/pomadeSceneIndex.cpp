// usdGenPomade imaging — Pomade scene index implementation (plan/18 §2.1-§2.2).
#include "usdGenPomade/imaging/pomadeSceneIndex.h"
#include "usdGenUsd/convert/math.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadePublish.h"
#include "usdGenPomade/pomadeRegistry.h"
#include "usdGenPomade/pomadeTransport.h"

#include "usdGen/api/epoch.h"

#include "pxr/base/tf/envSetting.h"
#include "pxr/base/tf/registryManager.h"
#include "pxr/imaging/hd/basisCurvesSchema.h"
#include "pxr/imaging/hd/basisCurvesTopologySchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/materialConnectionSchema.h"
#include "pxr/imaging/hd/materialNetworkSchema.h"
#include "pxr/imaging/hd/materialNodeSchema.h"
#include "pxr/imaging/hd/materialNodeParameterSchema.h"
#include "pxr/imaging/hd/materialSchema.h"
#include "pxr/imaging/hd/meshSchema.h"
#include "pxr/imaging/hd/meshTopologySchema.h"
#include "pxr/imaging/hd/subdivisionTagsSchema.h"
#include "pxr/imaging/hd/legacyDisplayStyleSchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/sceneIndexPluginRegistry.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"
#include "pxr/usd/usdGeom/tokens.h"

#include <array>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

TF_DEFINE_ENV_SETTING(USDGENPOMADE_ENABLE, true,
                      "Enable the usdGenPomade scene index plugin.");
TF_DEFINE_ENV_SETTING(USDGENPOMADE_TEST_TUBE, false,
                      "Publish the static test tube while no model is "
                      "active. Opt-in scene-index scaffolding: an empty "
                      "stage must open clean, so interactive and recorded "
                      "frames never carry it unless a harness asks for it "
                      "explicitly. Activating a model removes it either "
                      "way.");
TF_REGISTRY_FUNCTION(TfType) {
    HdSceneIndexPluginRegistry::Define<UsdGenPomadeSceneIndexPlugin>();
}
TF_REGISTRY_FUNCTION(HdSceneIndexPlugin) {
    HdSceneIndexPluginRegistry::GetInstance().RegisterSceneIndexForRenderer(
        HdSceneIndexPluginRegistryTokens->allRenderers.GetString(),
        TfToken("UsdGenPomadeSceneIndexPlugin"), nullptr, 0,
        HdSceneIndexPluginRegistry::InsertionOrderAtEnd);
}

HdSceneIndexBaseRefPtr
UsdGenPomadeSceneIndexPlugin::_AppendSceneIndex(
    std::string const&, HdSceneIndexBaseRefPtr const& input,
    HdContainerDataSourceHandle const&) {
    return UsdGenPomadeSceneIndex::New(input);
}

bool
UsdGenPomadeSceneIndexPlugin::_IsEnabled(
    HdContainerDataSourceHandle const&) const {
    return TfGetEnvSetting(USDGENPOMADE_ENABLE);
}

namespace {

TfToken const _scopeType("scope");
TfToken const _meshType("mesh");
TfToken const _curvesType("basisCurves");
TfToken const _pointsType("points");
TfToken const _materialType("material");

TfToken const _tokPoints("points");
TfToken const _tokNormals("normals");
TfToken const _tokFaceVarying("faceVarying");
TfToken const _tokWidths("widths");
TfToken const _tokDisplayColor("displayColor");
TfToken const _tokPrimvars("primvars");
TfToken const _tokPrimvarValue("primvarValue");
TfToken const _tokExtent("extent");
TfToken const _tokMin("min");
TfToken const _tokMax("max");
TfToken const _tokTopology("topology");
TfToken const _tokMesh("mesh");
TfToken const _tokBasisCurves("basisCurves");
TfToken const _tokVisibility("visibility");
TfToken const _tokTubeId("tubeId");
TfToken const _tokClumpColor("clumpColor");
TfToken const _tokSelected("selected");
TfToken const _tokXray("xray");
TfToken const _tokHierarchyLevel("hierarchyLevel");
TfToken const _tokCvIndex("cvIndex");
TfToken const _tokSectionIndex("sectionIndex");
TfToken const _tokHairT("hairT");
TfToken const _tokHandleId("handleId");
TfToken const _tokActive("active");
TfToken const _tokUnlit("unlit");
TfToken const _tokOpacity("opacity");
TfToken const _tokSurface("surface");
TfToken const _tokRenderScope("__usdGenRender");
TfToken const _tokMaterialBindings("materialBindings");
TfToken const _tokConstant("constant");
TfToken const _tokUniform("uniform");
TfToken const _tokVertex("vertex");
TfToken const _tokColorRole("color");
TfToken const _tokPointRole("point");
TfToken const _tokNormalRole("normal");

HdTokenDataSourceHandle
_Tok(TfToken const &t)
{
    return HdRetainedTypedSampledDataSource<TfToken>::New(t);
}

template <class T>
HdSampledDataSourceHandle
_Samp(T const &v)
{
    static_assert(!usdGenUsd::kIsEngineMath<T>, "convert engine math with usdGenUsd::ToHost");
    return HdRetainedTypedSampledDataSource<T>::New(v);
}

HdContainerDataSourceHandle
_Primvar(HdSampledDataSourceHandle const &values,
         TfToken const &interpolation,
         TfToken const &role = TfToken())
{
    HdPrimvarSchema::Builder b;
    b.SetPrimvarValue(values);
    if (!interpolation.IsEmpty()) {
        b.SetInterpolation(_Tok(interpolation));
    }
    if (!role.IsEmpty()) {
        b.SetRole(_Tok(role));
    }
    return b.Build();
}

void
_Add(std::vector<TfToken> *names,
     std::vector<HdDataSourceBaseHandle> *values,
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
    if (names.empty()) {
        return HdRetainedContainerDataSource::New();
    }
    return HdRetainedContainerDataSource::New(
        names.size(), names.data(), values.data());
}

HdContainerDataSourceHandle
_IdentityXform()
{
    return HdXformSchema::Builder()
        .SetMatrix(HdRetainedTypedSampledDataSource<GfMatrix4d>::New(
            GfMatrix4d(1.0)))
        .SetResetXformStack(
            HdRetainedTypedSampledDataSource<bool>::New(true))
        .Build();
}

HdContainerDataSourceHandle
_Visible(bool visible)
{
    return HdVisibilitySchema::Builder()
        .SetVisibility(HdRetainedTypedSampledDataSource<bool>::New(visible))
        .Build();
}

// A one-entry container that forces `visibility = false`. Overlaid on a
// tile prim it hides it without disturbing anything else the groom scene
// index published for it (plan/17 §3.2). Each index makes one.
HdContainerDataSourceHandle
_MakeHiddenOverlay()
{
    return HdRetainedContainerDataSource::New(_tokVisibility, _Visible(false));
}

HdDataSourceBaseHandle
_Extent(GfVec3f const &mn, GfVec3f const &mx)
{
    return _Container({_tokMin, _tokMax},
                      {HdDataSourceBaseHandle(_Samp(mn)),
                       HdDataSourceBaseHandle(_Samp(mx))});
}

// HOW AN OVERLAY GETS DRAWN ON TOP (plan/18 §2.4a, V9).
//
// Centers, CV dots, rings, the gizmo and the brush ring all sit INSIDE or
// ON the tube they describe, so with an opaque tube in front of them they
// are simply not in the frame. Storm offers no per-prim "ignore the depth
// test", and the two things that look like it do not apply here:
//
//   * `displayInOverlay` (HdLegacyDisplayStyleSchema, and the material tag
//     HdStMaterialTagTokens->displayInOverlay it produces) is real, and
//     mesh/basisCurves/points all honour it — but HdxTaskController only
//     builds render tasks for defaultMaterialTag, masked, additive,
//     translucent and volume (hdx/taskController.cpp _CreateRenderGraph).
//     usdview uses that task controller, so a prim tagged displayInOverlay
//     lands in no collection at all and disappears instead of coming
//     forward. Same for `occludedSelectionShowsThrough`, whose
//     translucentToSelection tag only the shadow task knows.
//   * a depth bias in the shader would need the displacement terminal,
//     which Storm evaluates for meshes only (hdSt/shaders/mesh.glslfx is
//     the sole caller of DisplacementTerminal) — the overlays are curves
//     and points.
//
// What does work is to stop the OCCLUDER from writing depth. A material
// tagged `translucent` is drawn by HdxOitRenderTask after the opaque pass,
// composited out of the OIT buffers, and never writes to the depth buffer.
// So an x-rayed tube no longer hides anything: the overlays are ordinary
// opaque geometry, they draw first, and the ghosted tube blends over them.
//
// HdSt derives that tag from the material, and the strongest opinion below
// a hardcoded glslfx `materialTag` is the terminal's authored `opacity`
// (hdSt/materialNetwork.cpp _GetMaterialTag: "Weakest opinion is an
// authored terminal.opacity value" — anything under 1 means translucent).
// pomadeTube.glslfx therefore no longer hardcodes a tag, and the index
// publishes the same shader twice: material_tube with opacity 1 (opaque
// pass) and material_tubeXray with opacity below 1 (OIT pass). A level's
// mesh binds whichever its x-ray state calls for.
//
// The number itself is never the alpha on screen. The alpha is the per-
// level `xray` primvar, so one x-ray material serves the focused level at
// 25 % and the levels behind it at 10 %; this constant exists only to put
// the draw items in the OIT pass.
float const _kXrayMaterialOpacity = 0.25f;

// One `surface` node under the universal render context (the tile
// publisher's _SurfaceMaterial pattern, plan/17 §4.4). `unlit` picks the
// glslfx's flat branch, which is what the overlay material is: one shader,
// two bindings, so the centers/rings/CV dots read as pure colour while the
// tubes are shaded.
HdContainerDataSourceHandle
_SurfaceMaterial(TfToken const &identifier,
                 std::vector<TfToken> const &paramNames,
                 std::vector<HdSampledDataSourceHandle> const &paramValues)
{
    TfToken const &surface = _tokSurface;
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    for (size_t i = 0; i < paramNames.size() && i < paramValues.size(); ++i) {
        names.push_back(paramNames[i]);
        values.push_back(HdMaterialNodeParameterSchema::Builder()
                             .SetValue(paramValues[i])
                             .Build());
    }
    HdContainerDataSourceHandle parameters =
        _Container(std::move(names), std::move(values));
    HdContainerDataSourceHandle const node =
        HdMaterialNodeSchema::Builder()
            .SetNodeIdentifier(_Tok(identifier))
            .SetParameters(parameters)
            .SetInputConnections(HdRetainedContainerDataSource::New())
            .Build();
    HdContainerDataSourceHandle const terminal =
        HdMaterialConnectionSchema::Builder()
            .SetUpstreamNodePath(_Tok(surface))
            .SetUpstreamNodeOutputName(_Tok(surface))
            .Build();
    HdContainerDataSourceHandle const network =
        HdMaterialNetworkSchema::Builder()
            .SetNodes(HdRetainedContainerDataSource::New(surface, node))
            .SetTerminals(
                HdRetainedContainerDataSource::New(surface, terminal))
            .Build();
    TfToken const contextName =
        HdMaterialSchemaTokens->universalRenderContext;
    HdDataSourceBaseHandle const contextValue = network;
    return HdRetainedContainerDataSource::New(
        HdMaterialSchemaTokens->material,
        HdMaterialSchema::BuildRetained(1, &contextName, &contextValue));
}

// The guide preview's material: UsdGenHairPreview, parameterised so the
// strand keeps the clump colour it is published with (plan/18 §2.4a,
// "guides ... coloured by their clump").
//
// The shader's albedo is mix(root, tipColor, hairT^colorRamp) with `root`
// taken from the per-curve displayColor when the prim has one. With the
// stock defaults that is a lerp from the clump colour at the root to a
// fixed dark brown at the tip, tuned for rendered fur, so a Pomade guide
// arrived at the viewport as brown hair rather than as a coloured guide.
// Here the tip goes white instead, so the strand leaves its root in the
// clump colour and lightens along its length. That gradient is the point,
// not a decoration: a guide grown from a tube is drawn against that same
// tube, in that same clump colour, so a guide painted the flat clump
// colour is invisible — which is what the 2026-09-19 braid frame showed
// (the preview was published, drawn, and indistinguishable from the tube
// behind it). plan/18 §2.4a allows the lighter tint for exactly this
// reason. The diffuse gain and wrap are raised on the same grounds: this
// is an editing overlay that has to read under the viewer's default
// headlight, not a render of hair.
HdContainerDataSourceHandle
_GuidePreviewMaterial()
{
    return _SurfaceMaterial(
        TfToken("UsdGenHairPreview"),
        {TfToken("tipColor"), TfToken("colorRamp"), TfToken("diffuseGain"),
         TfToken("diffuseWrap"), TfToken("transmissionGain"),
         TfToken("specular1Gain")},
        {_Samp(GfVec3f(1.0f, 1.0f, 1.0f)), _Samp(1.2f), _Samp(0.90f),
         _Samp(0.70f), _Samp(0.10f), _Samp(0.12f)});
}

HdDataSourceBaseHandle
_MaterialBinding(SdfPath const &path)
{
    HdDataSourceBaseHandle binding =
        HdMaterialBindingSchema::Builder()
            .SetPath(
                HdRetainedTypedSampledDataSource<SdfPath>::New(path))
            .Build();
    return HdRetainedContainerDataSource::New(
        HdMaterialBindingsSchemaTokens->allPurpose, binding);
}

HdContainerDataSourceHandle
_MeshTopology(VtIntArray const &counts, VtIntArray const &indices,
              usdGenPomade::PomadeSubdivisionCage const *cage = nullptr)
{
    HdMeshSchema::Builder builder;
    builder
        .SetTopology(HdMeshTopologySchema::Builder()
                         .SetFaceVertexCounts(
                             HdRetainedTypedSampledDataSource<VtIntArray>::
                                 New(counts))
                         .SetFaceVertexIndices(
                             HdRetainedTypedSampledDataSource<VtIntArray>::
                                 New(indices))
                         .SetOrientation(_Tok(HdTokens->rightHanded))
                         .Build())
        .SetSubdivisionScheme(_Tok(cage ? UsdGeomTokens->catmullClark
                                        : UsdGeomTokens->none))
        .SetDoubleSided(HdRetainedTypedSampledDataSource<bool>::New(true));
    if (cage) {
        builder.SetSubdivisionTags(HdSubdivisionTagsSchema::Builder()
            .SetInterpolateBoundary(_Tok(UsdGeomTokens->edgeAndCorner))
            .SetCornerIndices(HdRetainedTypedSampledDataSource<VtIntArray>::New(
                cage->cornerIndices))
            .SetCornerSharpnesses(HdRetainedTypedSampledDataSource<VtFloatArray>::New(
                cage->cornerSharpnesses)).Build());
    }
    return builder.Build();
}

template <class T>
VtArray<T> _CageFaceValues(VtArray<T> const &values, VtIntArray const &sourceFaces)
{
    VtArray<T> result(sourceFaces.size());
    for (size_t i = 0; i < sourceFaces.size(); ++i)
        result[i] = values[size_t(sourceFaces[i])];
    return result;
}

HdContainerDataSourceHandle
_CurvesTopology(VtIntArray const &counts, VtIntArray const &indices,
                TfToken const &type, TfToken const &basis,
                TfToken const &wrap)
{
    HdBasisCurvesTopologySchema::Builder b;
    b.SetCurveVertexCounts(
        HdRetainedTypedSampledDataSource<VtIntArray>::New(counts));
    b.SetCurveIndices(
        HdRetainedTypedSampledDataSource<VtIntArray>::New(indices));
    b.SetBasis(_Tok(basis));
    b.SetType(_Tok(type));
    b.SetWrap(_Tok(wrap));
    return HdBasisCurvesSchema::Builder().SetTopology(b.Build()).Build();
}

// The mesh data source for one level: subdivision cage, end tags, the
// per-face identity primvars and the level's display state. Purpose is
// deliberately omitted (an absent purpose resolves to the geometry render
// tag; authoring purpose="default" would match no collection and Storm
// would never sync the prim — the 2026-09-12 tile lesson).
//
// tubeId/clumpColor/selected are UNIFORM, i.e. one value per face, expanded
// to the real face count. The P3 index published them as single-element
// `constant` arrays because it had one tube per mesh and a 1-element uniform
// on a 32-face mesh is rejected; a level mesh carries many tubes, so uniform
// is both the correct interpolation and correctly sized. The two the shader
// reads (clumpColor, selected) are floats: the repo's glslfx attributes only
// spell float/vec2/vec3 and the HD_HAS_-guarded HdGet_ reads need a matching
// type. tubeId and hierarchyLevel are ints — no shader reads them, only the
// tools and the tests.
HdContainerDataSourceHandle
_BuildLevelMeshDataSource(usdGenPomade::PomadeStagedLevel const &level,
                          SdfPath const &materialPath)
{
    auto const &cage = level.subdivision;
    VtFloatArray selected(cage.sourceFaces.size());
    for (size_t i = 0; i < selected.size(); ++i)
        selected[i] = float(level.faceSelected[size_t(cage.sourceFaces[i])]);
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(cage.points), _tokVertex, _tokPointRole));
    // Let subdivision generate smooth limit normals, including the bevel.
    // The old face-varying strip normals would split that rounded surface.
    _Add(&pvNames, &pvValues, _tokTubeId,
         _Primvar(_Samp(_CageFaceValues(level.faceTubeId, cage.sourceFaces)),
                  _tokUniform));
    _Add(&pvNames, &pvValues, _tokClumpColor,
         _Primvar(_Samp(_CageFaceValues(level.faceClumpColor, cage.sourceFaces)),
                  _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokSelected,
         _Primvar(_Samp(selected), _tokUniform));
    _Add(&pvNames, &pvValues, _tokHierarchyLevel,
         _Primvar(_Samp(VtIntArray(1, level.level)), _tokConstant));
    // The ABSOLUTE alpha the level draws with, 0 when it is opaque. It was
    // a 0/1 flag through V8, with one `xrayOpacity` material parameter
    // behind it; plan/18 §2.4a needs the focused level and the levels
    // behind it ghosted by different amounts in the same frame, and a
    // primvar is the only one of the two that is per level.
    _Add(&pvNames, &pvValues, _tokXray,
         _Primvar(_Samp(VtFloatArray(1, level.xray ? level.xrayOpacity
                                                   : 0.0f)),
                  _tokConstant));

    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokMesh,
         _MeshTopology(cage.faceVertexCounts, cage.faceVertexIndices, &cage));
    _Add(&names, &values, HdLegacyDisplayStyleSchemaTokens->displayStyle,
         HdLegacyDisplayStyleSchema::Builder().SetRefineLevel(
             HdRetainedTypedSampledDataSource<int>::New(2)).Build());
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent,
         _Extent(level.extentMin, level.extentMax));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    // Centers-only keeps the center overlay and drops everything heavy:
    // the ladder's fourth step (plan/18 section 3.7).
    _Add(&names, &values, _tokVisibility,
         _Visible(level.visible && !level.centersOnly));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// One linear basisCurves overlay: center curves or section rings. Colour and
// width are uniform (per curve), so the focused level's curves thicken
// without touching a point.
HdContainerDataSourceHandle
_BuildCurvesDataSource(VtVec3fArray const &points, VtIntArray const &counts,
                       VtIntArray const &indices, VtVec3fArray const &colors,
                       VtFloatArray const &widths, VtIntArray const &tubeIds,
                       VtIntArray const &selected,
                       TfToken const &extraName, VtIntArray const &extra,
                       GfVec3f const &mn, GfVec3f const &mx, bool visible,
                       SdfPath const &materialPath)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(colors), _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokWidths,
         _Primvar(_Samp(widths), _tokUniform));
    _Add(&pvNames, &pvValues, _tokTubeId,
         _Primvar(_Samp(tubeIds), _tokUniform));
    if (!selected.empty()) {
        // FLOAT, exactly as the level mesh publishes it, and for the same
        // reason: `selected` is one of the three primvars pomadeTube.glslfx
        // declares in its sdrMetadata, HdSt binds it for every prim that
        // carries the material, and the shader reads it as a float. The
        // mesh converted; the curves did not, and an int under a float
        // declaration shifts the rest of the uniform block -- which is
        // what drew one center curve pure green in the 2026-09-19 braid
        // frames, the first frames in which the curves were visible at
        // all (V9 x-ray). Nothing reads tubeId or sectionIndex in a
        // shader, so those stay ints.
        VtFloatArray asFloat(selected.size());
        for (size_t i = 0; i < selected.size(); ++i) {
            asFloat[i] = float(selected[i]);
        }
        _Add(&pvNames, &pvValues, _tokSelected,
             _Primvar(_Samp(asFloat), _tokUniform));
    }
    if (!extra.empty()) {
        _Add(&pvNames, &pvValues, extraName,
             _Primvar(_Samp(extra), _tokUniform));
    }
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokBasisCurves,
         _CurvesTopology(counts, indices, TfToken("linear"),
                         TfToken("bezier"), TfToken("nonperiodic")));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(visible));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// The CV dots: a points prim with per-vertex colour and width.
HdContainerDataSourceHandle
_BuildPointsDataSource(VtVec3fArray const &points, VtVec3fArray const &colors,
                       VtFloatArray const &widths, VtIntArray const &tubeIds,
                       TfToken const &extraName, VtIntArray const &extra,
                       GfVec3f const &mn, GfVec3f const &mx, bool visible,
                       SdfPath const &materialPath)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(colors), _tokVertex, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokWidths,
         _Primvar(_Samp(widths), _tokVertex));
    _Add(&pvNames, &pvValues, _tokTubeId,
         _Primvar(_Samp(tubeIds), _tokVertex));
    if (!extra.empty()) {
        _Add(&pvNames, &pvValues, extraName,
             _Primvar(_Samp(extra), _tokVertex));
    }
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(visible));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// The lowest refine level whose Storm repr carries the ribbon orientation
// vector UsdGenHairPreview reads (hdSt/basisCurves.cpp: 0 = WIRE lines,
// 1 = RIBBON + HAIR normal, 2 = RIBBON + ROUND). 2 is also what the tile
// pipeline treats as bound-and-shaded, so the preview matches it.
int const _kGuideRefineLevel = 2;

// The guide preview: basisCurves (cubic bspline, pinned — the committed
// Guides contract) with per-vertex widths + hairT for UsdGenHairPreview
// (variant A: patch reprs at refineLevel >= 2, the tile rule, 06 §4.1), plus
// the per-curve clump colour the 2014/2018 stills show.
HdContainerDataSourceHandle
_BuildGuideDataSource(usdGenPomade::PomadeStagedLevel const &level,
                      SdfPath const &materialPath)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(level.guidePoints), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokWidths,
         _Primvar(_Samp(level.guideWidths), _tokVertex));
    _Add(&pvNames, &pvValues, _tokHairT,
         _Primvar(_Samp(level.guideHairT), _tokVertex));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(level.guideCurveColor), _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokTubeId,
         _Primvar(_Samp(level.guideCurveTubeId), _tokUniform));
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokBasisCurves,
         _CurvesTopology(level.guideVertexCounts, level.guideIndices,
                         TfToken("cubic"), TfToken("bspline"),
                         TfToken("pinned")));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(level.guideMin, level.guideMax));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    // The guide preview states its own refine level instead of taking the
    // host's fallback, because the material bound below cannot survive the
    // level the host would otherwise choose. UsdGenHairPreview reads the
    // ribbon orientation vector `inData.Neye`, and the WIRE repr Storm
    // draws at refineLevel 0 declares a curve vertex block that has no
    // Neye member (basisCurves.glslfx:1212-1218), so the shader fails to
    // COMPILE -- and Storm retries the compilation on every draw, which
    // turns the preview into hundreds of milliseconds per frame rather
    // than into a missing shade. usdGen's own tiles solve the same problem
    // at the other end, by hiding the binding below refineLevel 2
    // (groomSceneIndexPlugin.cpp DescriptionOverlay); this prim is an
    // overlay the tool owns, and the preview is only worth drawing shaded,
    // so it pins the level instead of dropping the shader.
    _Add(&names, &values, TfToken("displayStyle"),
         _Container({TfToken("refineLevel")},
                    {HdDataSourceBaseHandle(_Samp(_kGuideRefineLevel))}));
    // Centers-only keeps the center overlay and drops everything heavy:
    // the ladder's fourth step (plan/18 section 3.7). guidesVisible is the
    // other half of the "show amplified hair" swap (plan/17 section 3.2):
    // the preview steps aside for the cook's tiles.
    _Add(&names, &values, _tokVisibility,
         _Visible(level.visible && !level.centersOnly &&
                  level.guidesVisible));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// The gizmo and the brush ring (plan/18 §2.4): linear basisCurves with a
// per-curve colour, the handle id and whether that handle is the one being
// dragged. They bind the unlit overlay material, so what the artist sees is
// exactly the colour the record asked for, and they are ordinary prims, so
// they depth-sort against the tubes they manipulate.
HdContainerDataSourceHandle
_BuildOverlayDataSource(usdGenPomade::PomadeOverlayCurves const &curves,
                        SdfPath const &materialPath)
{
    size_t const cvCount = curves.points.size() / 3;
    VtVec3fArray points(cvCount);
    VtIntArray indices(cvCount);
    for (size_t i = 0; i < cvCount; ++i) {
        points[i] = GfVec3f(curves.points[i * 3 + 0],
                            curves.points[i * 3 + 1],
                            curves.points[i * 3 + 2]);
        indices[i] = int(i);
    }
    size_t const curveCount = curves.vertexCounts.size();
    VtIntArray counts(curveCount);
    VtVec3fArray colors(curveCount);
    VtFloatArray widths(curveCount);
    VtIntArray handleIds(curveCount);
    VtIntArray active(curveCount);
    for (size_t c = 0; c < curveCount; ++c) {
        counts[c] = curves.vertexCounts[c];
        colors[c] = GfVec3f(curves.colors[c * 3 + 0],
                            curves.colors[c * 3 + 1],
                            curves.colors[c * 3 + 2]);
        widths[c] = curves.widths[c];
        handleIds[c] = curves.handleIds[c];
        active[c] = curves.active[c];
    }
    GfVec3f mn(0.0f), mx(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0) {
            mn = mx = points[0];
        } else {
            for (int a = 0; a < 3; ++a) {
                mn[a] = std::min(mn[a], points[i][a]);
                mx[a] = std::max(mx[a], points[i][a]);
            }
        }
    }
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(colors), _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokWidths,
         _Primvar(_Samp(widths), _tokUniform));
    _Add(&pvNames, &pvValues, _tokHandleId,
         _Primvar(_Samp(handleIds), _tokUniform));
    _Add(&pvNames, &pvValues, _tokActive,
         _Primvar(_Samp(active), _tokUniform));
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokBasisCurves,
         _CurvesTopology(counts, indices, TfToken("linear"),
                         TfToken("bezier"), TfToken("nonperiodic")));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(true));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// The P0 test tube, published while no model is active.
HdContainerDataSourceHandle
_BuildTestTubeDataSource(usdGenPomade::PomadeStagedTubeMesh const &tube,
                         SdfPath const &materialPath)
{
    usdGenPomade::PomadeRgb const rgb = usdGenPomade::PomadeClumpColor(0, 1, -1);
    size_t const faceCount = tube.faceVertexCounts.size();
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(tube.points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokNormals,
         _Primvar(_Samp(tube.normals), _tokFaceVarying, _tokNormalRole));
    _Add(&pvNames, &pvValues, _tokTubeId,
         _Primvar(_Samp(VtIntArray(faceCount, 0)), _tokUniform));
    _Add(&pvNames, &pvValues, _tokClumpColor,
         _Primvar(_Samp(VtVec3fArray(faceCount,
                                     GfVec3f(rgb.r, rgb.g, rgb.b))),
                  _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokSelected,
         _Primvar(_Samp(VtFloatArray(faceCount, 0.0f)), _tokUniform));
    _Add(&pvNames, &pvValues, _tokHierarchyLevel,
         _Primvar(_Samp(VtIntArray(1, 1)), _tokConstant));
    _Add(&pvNames, &pvValues, _tokXray,
         _Primvar(_Samp(VtFloatArray(1, 0.0f)), _tokConstant));
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokMesh,
         _MeshTopology(tube.faceVertexCounts, tube.faceVertexIndices));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(tube.extentMin, tube.extentMax));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(true));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// Graph nodes as a points prim: positions + per-point display colour
// (white; unwelded coincident nodes warn orange — the HUD ring's data).
// Bound to the overlay material like every other overlay dot: Storm does
// not draw these points under the fallback material, so without the
// binding a click that lands in the model leaves no dot on the screen.
HdContainerDataSourceHandle
_BuildNodesDataSource(VtVec3fArray const &points, VtVec3fArray const &colors,
                      VtFloatArray const &widths,
                      SdfPath const &materialPath)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(colors), _tokVertex, _tokColorRole));
    _Add(&pvNames, &pvValues, _tokWidths,
         _Primvar(_Samp(widths), _tokVertex));
    GfVec3f mn(0.0f), mx(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0) {
            mn = mx = points[0];
        } else {
            for (int a = 0; a < 3; ++a) {
                mn[a] = std::min(mn[a], points[i][a]);
                mx[a] = std::max(mx[a], points[i][a]);
            }
        }
    }
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(true));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// Graph edges as linear basisCurves: one curve per edge polyline, uniform
// per-curve colour (shared edges white, border edges grey — plan/17 §5.1).
// Bound to the overlay material with the nodes: same Storm, same rule.
HdContainerDataSourceHandle
_BuildEdgesDataSource(VtIntArray const &curveVertexCounts,
                      VtIntArray const &curveIndices,
                      VtVec3fArray const &points,
                      VtVec3fArray const &curveColors,
                      SdfPath const &materialPath)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(curveColors), _tokUniform, _tokColorRole));
    GfVec3f mn(0.0f), mx(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0) {
            mn = mx = points[0];
        } else {
            for (int a = 0; a < 3; ++a) {
                mn[a] = std::min(mn[a], points[i][a]);
                mx[a] = std::max(mx[a], points[i][a]);
            }
        }
    }
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokBasisCurves,
         _CurvesTopology(curveVertexCounts, curveIndices, TfToken("linear"),
                         TfToken("bezier"), TfToken("nonperiodic")));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(true));
    if (!materialPath.IsEmpty()) {
        _Add(&names, &values, _tokMaterialBindings,
             _MaterialBinding(materialPath));
    }
    return _Container(std::move(names), std::move(values));
}

// Live region tint: the scalp topology with per-face display colours and
// the pomadeRegion primvar (plan/17 §5.1 HUD overlays). The colours come from
// the same palette the tubes use, so the patch on the head and the tube
// rooted in it match (plan/18 §2.4a).
HdContainerDataSourceHandle
_BuildRegionsDataSource(VtVec3fArray const &points,
                        VtIntArray const &faceVertexCounts,
                        VtIntArray const &faceVertexIndices,
                        VtVec3fArray const &faceColors,
                        VtIntArray const &faceRegions)
{
    std::vector<TfToken> pvNames;
    std::vector<HdDataSourceBaseHandle> pvValues;
    _Add(&pvNames, &pvValues, _tokPoints,
         _Primvar(_Samp(points), _tokVertex, _tokPointRole));
    _Add(&pvNames, &pvValues, _tokDisplayColor,
         _Primvar(_Samp(faceColors), _tokUniform, _tokColorRole));
    _Add(&pvNames, &pvValues, TfToken("usdGen:pomadeRegion"),
         _Primvar(_Samp(faceRegions), _tokUniform));
    GfVec3f mn(0.0f), mx(0.0f);
    for (size_t i = 0; i < points.size(); ++i) {
        if (i == 0) {
            mn = mx = points[0];
        } else {
            for (int a = 0; a < 3; ++a) {
                mn[a] = std::min(mn[a], points[i][a]);
                mx[a] = std::max(mx[a], points[i][a]);
            }
        }
    }
    std::vector<TfToken> names;
    std::vector<HdDataSourceBaseHandle> values;
    _Add(&names, &values, _tokMesh,
         _MeshTopology(faceVertexCounts, faceVertexIndices));
    _Add(&names, &values, _tokPrimvars,
         _Container(std::move(pvNames), std::move(pvValues)));
    _Add(&names, &values, _tokExtent, _Extent(mn, mx));
    _Add(&names, &values, TfToken("xform"), _IdentityXform());
    _Add(&names, &values, _tokVisibility, _Visible(true));
    return _Container(std::move(names), std::move(values));
}

// The live K3 map is deliberately one value per coarse scalp face: that is
// the inexpensive channel the bake and root sampler share.  It is not a
// sufficient display representation, though.  A graph boundary can cross a
// single quad and leave two different regions on it.  The helpers below make
// a small, display-only patch mesh by triangulating each region contour in
// its chart and clipping those triangles against the scalp's existing face
// fan.  Only faces touched by a patch acquire extra triangles; untouched
// scalp stays at its authored resolution.
struct _Patch2 {
    float x = 0.0f;
    float y = 0.0f;
};

float
_Cross2(_Patch2 const &a, _Patch2 const &b, _Patch2 const &c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

float
_PolygonArea2(std::vector<_Patch2> const &p)
{
    float area = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) {
        _Patch2 const &a = p[i];
        _Patch2 const &b = p[(i + 1) % p.size()];
        area += a.x * b.y - a.y * b.x;
    }
    return area;
}

bool
_InsideTriangle2(_Patch2 const &p, _Patch2 const &a, _Patch2 const &b,
                 _Patch2 const &c, float sign)
{
    // Contours retain the K2 samples along a straight edge.  Those
    // collinear samples lie on a candidate ear but do not occupy it, and
    // must not prevent ear clipping from ever getting started.
    float const eps = 1.0e-7f;
    return sign * _Cross2(a, b, p) > eps &&
           sign * _Cross2(b, c, p) > eps &&
           sign * _Cross2(c, a, p) > eps;
}

// Ear clipping is local to one graph region.  It deliberately preserves a
// concave loop's indentation; a fan from the first sample would paint the
// indentation and turn an outside gap into a coloured tile.
std::vector<std::array<int, 3>>
_TriangulatePatch(std::vector<_Patch2> const &polygon)
{
    std::vector<std::array<int, 3>> triangles;
    if (polygon.size() < 3) {
        return triangles;
    }
    float const area = _PolygonArea2(polygon);
    if (std::fabs(area) <= 1.0e-10f) {
        return triangles;
    }
    float const sign = area > 0.0f ? 1.0f : -1.0f;
    std::vector<int> remaining(polygon.size());
    for (size_t i = 0; i < remaining.size(); ++i) {
        remaining[i] = int(i);
    }
    // A bad contour should not make publishing unbounded.  The graph's
    // contours are simple by construction, so this limit only catches an
    // accidental duplicate/self-intersection and leaves that patch absent.
    size_t guard = polygon.size() * polygon.size();
    while (remaining.size() > 2 && guard-- > 0) {
        bool clipped = false;
        for (size_t i = 0; i < remaining.size(); ++i) {
            int const ia = remaining[(i + remaining.size() - 1) %
                                     remaining.size()];
            int const ib = remaining[i];
            int const ic = remaining[(i + 1) % remaining.size()];
            if (sign * _Cross2(polygon[ia], polygon[ib], polygon[ic]) <=
                1.0e-7f) {
                continue;
            }
            bool contains = false;
            for (int ip : remaining) {
                if (ip != ia && ip != ib && ip != ic &&
                    _InsideTriangle2(polygon[ip], polygon[ia], polygon[ib],
                                     polygon[ic], sign)) {
                    contains = true;
                    break;
                }
            }
            if (!contains) {
                triangles.push_back({ia, ib, ic});
                remaining.erase(remaining.begin() + i);
                clipped = true;
                break;
            }
        }
        if (!clipped) {
            triangles.clear();
            return triangles;
        }
    }
    return triangles;
}

// K2 deliberately retains samples along every traced edge for a smooth white
// display curve.  They are useful to the graph, but a run of collinear
// vertices can leave ear clipping with only degenerate candidate ears after
// it has already emitted valid triangles.  Simplify only the fill polygon;
// the boundary curve still publishes every original CV.
std::vector<_Patch2>
_RemoveCollinearPatchPoints(std::vector<_Patch2> polygon)
{
    bool changed = true;
    while (changed && polygon.size() > 3) {
        changed = false;
        for (size_t i = 0; i < polygon.size() && polygon.size() > 3; ++i) {
            _Patch2 const &a = polygon[(i + polygon.size() - 1) %
                                       polygon.size()];
            _Patch2 const &b = polygon[i];
            _Patch2 const &c = polygon[(i + 1) % polygon.size()];
            float const abx = b.x - a.x;
            float const aby = b.y - a.y;
            float const bcx = c.x - b.x;
            float const bcy = c.y - b.y;
            float const scale = abx * abx + aby * aby + bcx * bcx + bcy * bcy;
            if (std::fabs(_Cross2(a, b, c)) <= 1.0e-6f *
                                                   std::max(1.0f, scale)) {
                polygon.erase(polygon.begin() + i);
                changed = true;
                break;
            }
        }
    }
    return polygon;
}

_Patch2
_Intersect2(_Patch2 const &a, _Patch2 const &b, _Patch2 const &c,
            _Patch2 const &d)
{
    float const abx = b.x - a.x;
    float const aby = b.y - a.y;
    float const cdx = d.x - c.x;
    float const cdy = d.y - c.y;
    float const denom = abx * cdy - aby * cdx;
    if (std::fabs(denom) <= 1.0e-12f) {
        return a;
    }
    float const t = ((c.x - a.x) * cdy - (c.y - a.y) * cdx) / denom;
    return {a.x + t * abx, a.y + t * aby};
}

// Convex clip: the scalp face fan gives us triangles, so the clip polygon is
// always convex even when an authored coarse n-gon is not.
std::vector<_Patch2>
_ClipPatchToTriangle(std::vector<_Patch2> subject, _Patch2 const &a,
                     _Patch2 const &b, _Patch2 const &c)
{
    _Patch2 const clip[3] = {a, b, c};
    float const sign = _Cross2(a, b, c) >= 0.0f ? 1.0f : -1.0f;
    for (int edge = 0; edge != 3 && !subject.empty(); ++edge) {
        std::vector<_Patch2> out;
        _Patch2 const &e0 = clip[edge];
        _Patch2 const &e1 = clip[(edge + 1) % 3];
        _Patch2 prev = subject.back();
        bool prevInside = sign * _Cross2(e0, e1, prev) >= -1.0e-6f;
        for (_Patch2 const &current : subject) {
            bool const currentInside =
                sign * _Cross2(e0, e1, current) >= -1.0e-6f;
            if (currentInside != prevInside) {
                out.push_back(_Intersect2(prev, current, e0, e1));
            }
            if (currentInside) {
                out.push_back(current);
            }
            prev = current;
            prevInside = currentInside;
        }
        subject.swap(out);
    }
    return subject;
}

GfVec3f
_LiftedPatchPoint(_Patch2 const &p, _Patch2 const &a, _Patch2 const &b,
                  _Patch2 const &c, GfVec3f const &pa, GfVec3f const &pb,
                  GfVec3f const &pc, GfVec3f const &na, GfVec3f const &nb,
                  GfVec3f const &nc, float offset)
{
    float const denom = _Cross2(a, b, c);
    if (std::fabs(denom) <= 1.0e-12f) {
        return pa;
    }
    float const wb = _Cross2(a, p, c) / denom;
    float const wc = _Cross2(a, b, p) / denom;
    float const wa = 1.0f - wb - wc;
    GfVec3f point = wa * pa + wb * pb + wc * pc;
    GfVec3f normal = wa * na + wb * nb + wc * nc;
    if (normal.GetLengthSq() > 1.0e-20f) {
        point += normal.GetNormalized() * offset;
    }
    return point;
}

// SnapshotGraph is the renderer-safe boundary handoff.  Build the same
// chart-local projection data as PomadeFlattenLoops from its immutable region
// contours; reading the live graph here would race a graph edit and could
// combine a new contour with an old scalp snapshot.
usdGenPomade::PomadeRegionLoops
_SnapshotDisplayLoops(usdGenPomade::PomadeModel::GraphSnapshot const &snap)
{
    usdGenPomade::PomadeRegionLoops loops;
    for (size_t r = 0; r < snap.regionBoundaries.size(); ++r) {
        std::vector<float> const &boundary = snap.regionBoundaries[r];
        size_t const count = boundary.size() / 3;
        if (count < 3 || count * 3 != boundary.size()) {
            continue;
        }
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        for (size_t i = 0; i < count; ++i) {
            float const *a = &boundary[i * 3];
            float const *b = &boundary[((i + 1) % count) * 3];
            nx += (a[1] - b[1]) * (a[2] + b[2]);
            ny += (a[2] - b[2]) * (a[0] + b[0]);
            nz += (a[0] - b[0]) * (a[1] + b[1]);
        }
        float const nl = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!(nl > 1.0e-10f)) {
            continue;
        }
        nx /= nl;
        ny /= nl;
        nz /= nl;
        float rx = std::fabs(nx) > 0.9f ? 0.0f : 1.0f;
        float rz = std::fabs(nx) > 0.9f ? 1.0f : 0.0f;
        float ux = rx - nx * (rx * nx + rz * nz);
        float uy = -ny * (rx * nx + rz * nz);
        float uz = rz - nz * (rx * nx + rz * nz);
        float const ul = std::sqrt(ux * ux + uy * uy + uz * uz);
        if (!(ul > 1.0e-10f)) {
            continue;
        }
        ux /= ul;
        uy /= ul;
        uz /= ul;
        int interp = int(r);
        for (size_t f = 0; f < snap.faceRegionIds.size() &&
                           f < snap.faceRegions.size(); ++f) {
            if (snap.faceRegionIds[f] == int(r)) {
                interp = snap.faceRegions[f];
                break;
            }
        }
        loops.loopBegin.push_back(int(loops.points.size() / 3));
        loops.loopCount.push_back(int(count));
        loops.points.insert(loops.points.end(), boundary.begin(),
                            boundary.end());
        loops.planeN.insert(loops.planeN.end(), {nx, ny, nz});
        loops.planeP.insert(loops.planeP.end(),
                            {boundary[0], boundary[1], boundary[2]});
        loops.basisU.insert(loops.basisU.end(), {ux, uy, uz});
        loops.basisV.insert(loops.basisV.end(),
                            {ny * uz - nz * uy, nz * ux - nx * uz,
                             nx * uy - ny * ux});
        loops.interpIds.push_back(interp);
        loops.regionIds.push_back(int(r));
    }
    loops.valid = true;
    return loops;
}

SdfPath _LevelPath(SdfPath const &scope, int level)
{
    return scope.AppendChild(TfToken("L" + std::to_string(level)));
}

} // namespace

// -- paths --------------------------------------------------------------------

namespace {
// The fixed namespace, built at load (no function-local statics).
SdfPath const _kRootPath("/__usdGenPomade");
SdfPath const _kTestTubePath("/__usdGenPomade/testTube");
SdfPath const _kTubesScopePath("/__usdGenPomade/tubes");
SdfPath const _kCentersScopePath("/__usdGenPomade/centers");
SdfPath const _kCenterCVsScopePath("/__usdGenPomade/centerCVs");
SdfPath const _kRingsScopePath("/__usdGenPomade/rings");
SdfPath const _kRingCVsScopePath("/__usdGenPomade/ringCVs");
SdfPath const _kGuidesScopePath("/__usdGenPomade/guides");
SdfPath const _kGraphNodesPath("/__usdGenPomade/graphNodes");
SdfPath const _kGraphEdgesPath("/__usdGenPomade/graphEdges");
SdfPath const _kGraphRegionsPath("/__usdGenPomade/graphRegions");
SdfPath const _kGizmoPath("/__usdGenPomade/gizmo");
SdfPath const _kBrushRingPath("/__usdGenPomade/brushRing");
SdfPath const _kTubeMaterialPath("/__usdGenPomade/material_tube");
SdfPath const _kTubeXrayMaterialPath("/__usdGenPomade/material_tubeXray");
SdfPath const _kHairMaterialPath("/__usdGenPomade/material_hairPreview");
SdfPath const _kOverlayMaterialPath("/__usdGenPomade/material_overlay");
}  // namespace

SdfPath const &
UsdGenPomadeSceneIndex::RootPath()
{
    return _kRootPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::TestTubePath()
{
    return _kTestTubePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::TubesScopePath()
{
    return _kTubesScopePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::CentersScopePath()
{
    return _kCentersScopePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::CenterCVsScopePath()
{
    return _kCenterCVsScopePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::RingsScopePath()
{
    return _kRingsScopePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::RingCVsScopePath()
{
    return _kRingCVsScopePath;
}

SdfPath const &
UsdGenPomadeSceneIndex::GuidesScopePath()
{
    return _kGuidesScopePath;
}

SdfPath UsdGenPomadeSceneIndex::TubesPath(int level)
{
    return _LevelPath(TubesScopePath(), level);
}

SdfPath UsdGenPomadeSceneIndex::CentersPath(int level)
{
    return _LevelPath(CentersScopePath(), level);
}

SdfPath UsdGenPomadeSceneIndex::CenterCVsPath(int level)
{
    return _LevelPath(CenterCVsScopePath(), level);
}

SdfPath UsdGenPomadeSceneIndex::RingsPath(int level)
{
    return _LevelPath(RingsScopePath(), level);
}

SdfPath UsdGenPomadeSceneIndex::RingCVsPath(int level)
{
    return _LevelPath(RingCVsScopePath(), level);
}

SdfPath UsdGenPomadeSceneIndex::GuidesPath(int level)
{
    return _LevelPath(GuidesScopePath(), level);
}

SdfPath const &
UsdGenPomadeSceneIndex::GraphNodesPath()
{
    return _kGraphNodesPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::GraphEdgesPath()
{
    return _kGraphEdgesPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::GraphRegionsPath()
{
    return _kGraphRegionsPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::GizmoPath()
{
    return _kGizmoPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::BrushRingPath()
{
    return _kBrushRingPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::TubeMaterialPath()
{
    return _kTubeMaterialPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::TubeXrayMaterialPath()
{
    return _kTubeXrayMaterialPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::HairMaterialPath()
{
    return _kHairMaterialPath;
}

SdfPath const &
UsdGenPomadeSceneIndex::OverlayMaterialPath()
{
    return _kOverlayMaterialPath;
}

// -- shared state -------------------------------------------------------------

struct UsdGenPomadeSceneIndex::_Shared {
    // One published view. Immutable once stored.
    struct View : usdGen::UsdGenRefCounted {
        std::map<SdfPath, _Prim> prims;
        SdfPath hiddenGuides;  // the committed guides hidden while active
        // Face / point / tube census per published level.
        std::map<int, std::array<int, 3>> levels;
        int restagedTubeCount = 0;
        bool testTube = false;
    };
    struct Tiles : usdGen::UsdGenRefCounted {
        std::set<SdfPath> paths;
    };
    usdGen::UsdGenAtomicRef<View> view;
    usdGen::UsdGenAtomicRef<Tiles> tiles;
    // Frontend thread: the census PublishModel stages for the next _Commit.
    std::map<int, std::array<int, 3>> nextLevels;
    int nextRestaged = 0;

    // Frontend thread: publish a copy of the current view after `edit`.
    template <class Edit>
    void EditView(Edit &&edit)
    {
        usdGen::UsdGenRef<View> const current = view.Load();
        usdGen::UsdGenRef<View> next = current
            ? usdGen::UsdGenMakeRef<View>(*current)
            : usdGen::UsdGenMakeRef<View>();
        edit(*next);
        view.Store(std::move(next));
    }
    // Any thread: copy-on-write with a CAS retry; `edit` returns false to
    // leave the set alone.
    template <class Edit>
    void EditTiles(Edit &&edit)
    {
        (void)usdGen::UsdGenCowUpdateIf(
            tiles, [&](Tiles &t) { return edit(t.paths); });
    }
};

// -- lifetime -----------------------------------------------------------------

HdSceneIndexBaseRefPtr
UsdGenPomadeSceneIndex::New(HdSceneIndexBaseRefPtr const &inputScene)
{
    return TfCreateRefPtr(new UsdGenPomadeSceneIndex(inputScene));
}

UsdGenPomadeSceneIndex::UsdGenPomadeSceneIndex(
    HdSceneIndexBaseRefPtr const &inputScene)
    : HdSingleInputFilteringSceneIndexBase(inputScene)
    , _publisher(new usdGenPomade::PomadePublisher())
    , _published(new usdGenPomade::PomadeStagedModel())
    , _shared(new _Shared())
    , _hiddenOverlay(_MakeHiddenOverlay())
{
    // The root scope always exists, so GetChildPrimPaths can be a single
    // walk of the prim table.
    _shared->EditView([](_Shared::View &v) {
        v.prims[RootPath()] =
            _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    });
    usdGenPomade::PomadeRegistry::Get().AttachIndex(this);
    // Publish whatever is already active (usually nothing, and then the
    // test tube unless a headless record opted out).
    Refresh(~0u);
}

UsdGenPomadeSceneIndex::~UsdGenPomadeSceneIndex()
{
    usdGenPomade::PomadeRegistry::Get().DetachIndex(this);
}

// -- queries ------------------------------------------------------------------

HdSceneIndexPrim
UsdGenPomadeSceneIndex::GetPrim(SdfPath const &primPath) const
{
    SdfPath hiddenGuides;
    {
        usdGen::UsdGenEpochGuard guard;
        _Shared::View const *view = _shared->view.Peek();
        auto it = view->prims.find(primPath);
        if (it != view->prims.end()) {
            return {it->second.primType, it->second.dataSource};
        }
        hiddenGuides = view->hiddenGuides;
    }
    if (primPath.HasPrefix(RootPath())) {
        // Our namespace, nothing published there: never fall through to the
        // input, which does not own /__usdGenPomade (plan/17 R5).
        return {TfToken(), nullptr};
    }
    HdSceneIndexBaseRefPtr const &input = _GetInputSceneIndex();
    if (!input) {
        return {TfToken(), nullptr};
    }
    HdSceneIndexPrim prim = input->GetPrim(primPath);
    // plan/17 section 3.2: the usdGen cook's amplified tiles are hidden
    // while a gesture is live (they carry a stage version older than the
    // drag) and whenever "show amplified hair" is off. The override is an
    // overlay, so the tile keeps every other data source the groom index
    // published and comes straight back when the flag flips.
    if (prim.dataSource && !_amplifiedTilesVisible.load() &&
        _IsAmplifiedTilePath(primPath)) {
        prim.dataSource = HdOverlayContainerDataSource::New(
            _hiddenOverlay, prim.dataSource);
    }
    // plan/18 §2.4a: while a model is live, the committed <groom>/Guides
    // prim is the PREVIOUS commit's curves. It is an ordinary BasisCurves
    // with no displayColor, so Storm draws it plain white on top of every
    // tube the tool is publishing — which is exactly what the 2026-09-18
    // workspace frame showed. Hiding it is a Hydra opinion and nothing
    // more: the amplifier reads the stage, not this index, so the cook
    // and its tiles are untouched. Deactivating the model clears the path
    // and the prim comes straight back.
    if (prim.dataSource && !hiddenGuides.IsEmpty() &&
        primPath.HasPrefix(hiddenGuides)) {
        prim.dataSource = HdOverlayContainerDataSource::New(
            _hiddenOverlay, prim.dataSource);
    }
    return prim;
}

bool
UsdGenPomadeSceneIndex::_IsAmplifiedTilePath(SdfPath const &path)
{
    for (SdfPath p = path; !p.IsEmpty() && p != SdfPath::AbsoluteRootPath();
         p = p.GetParentPath()) {
        if (p.GetNameToken() == _tokRenderScope) {
            return true;
        }
    }
    return false;
}

SdfPathVector
UsdGenPomadeSceneIndex::GetChildPrimPaths(SdfPath const &path) const
{
    // The pomade subtree is additive: input children pass through untouched.
    SdfPathVector children;
    if (!path.HasPrefix(RootPath()) || path == SdfPath::AbsoluteRootPath()) {
        HdSceneIndexBaseRefPtr const &input = _GetInputSceneIndex();
        if (input) {
            children = input->GetChildPrimPaths(path);
        }
    }
    usdGen::UsdGenRef<_Shared::View> const view = _shared->view.Load();
    for (auto const &kv : view->prims) {
        if (kv.first.GetParentPath() == path) {
            children.push_back(kv.first);
        }
    }
    return children;
}

std::vector<int>
UsdGenPomadeSceneIndex::PublishedLevels() const
{
    usdGen::UsdGenRef<_Shared::View> const view = _shared->view.Load();
    std::vector<int> levels;
    for (auto const &kv : view->levels) {
        levels.push_back(kv.first);
    }
    return levels;
}

bool
UsdGenPomadeSceneIndex::PublishedLevelInfo(int level, int *outFaceCount,
                                          int *outPointCount,
                                          int *outTubeCount) const
{
    usdGen::UsdGenRef<_Shared::View> const view = _shared->view.Load();
    auto it = view->levels.find(level);
    if (it == view->levels.end()) {
        return false;
    }
    if (outFaceCount) {
        *outFaceCount = it->second[0];
    }
    if (outPointCount) {
        *outPointCount = it->second[1];
    }
    if (outTubeCount) {
        *outTubeCount = it->second[2];
    }
    return true;
}

bool
UsdGenPomadeSceneIndex::QueryPublishedLevel(int level, int *outFaceCount,
                                           int *outPointCount,
                                           int *outTubeCount) const
{
    return PublishedLevelInfo(level, outFaceCount, outPointCount,
                              outTubeCount);
}

int
UsdGenPomadeSceneIndex::LastRestagedTubeCount() const
{
    return _shared->view.Load()->restagedTubeCount;
}

bool
UsdGenPomadeSceneIndex::HasTestTube() const
{
    return _shared->view.Load()->testTube;
}

// -- notices ------------------------------------------------------------------

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::NoticesFor(uint32_t dirty)
{
    HdDataSourceLocatorSet locators;
    if (dirty & usdGenPomade::PomadeDirty_Topology) {
        locators.insert(HdDataSourceLocator(_tokMesh, _tokTopology));
        locators.insert(HdDataSourceLocator(_tokMesh,
                                            HdMeshSchemaTokens->subdivisionTags));
    }
    if (dirty & (usdGenPomade::PomadeDirty_Points |
                 usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    }
    if (dirty & (Dirty_Uniforms | usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokTubeId, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokClumpColor,
                                            _tokPrimvarValue));
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokSelected, _tokPrimvarValue));
    }
    if (dirty & usdGenPomade::PomadeDirty_Selection) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokSelected, _tokPrimvarValue));
    }
    if (dirty & Dirty_Xray) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokXray, _tokPrimvarValue));
    }
    if (dirty & Dirty_Material) {
        // The level swapped between the opaque and the translucent twin of
        // the tube shader, which is a different binding, not a different
        // parameter (see _kXrayMaterialOpacity).
        locators.insert(HdDataSourceLocator(_tokMaterialBindings));
    }
    if (dirty & Dirty_Visibility) {
        locators.insert(HdDataSourceLocator(_tokVisibility, _tokVisibility));
    }
    return locators;
}

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::CurveNoticesFor(uint32_t dirty)
{
    HdDataSourceLocatorSet locators;
    if (dirty & usdGenPomade::PomadeDirty_Topology) {
        locators.insert(HdDataSourceLocator(_tokBasisCurves, _tokTopology));
    }
    if (dirty & (usdGenPomade::PomadeDirty_Points |
                 usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    }
    if (dirty & (Dirty_Uniforms | usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokTubeId, _tokPrimvarValue));
    }
    if (dirty & (Dirty_Widths | usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokWidths, _tokPrimvarValue));
    }
    if (dirty & usdGenPomade::PomadeDirty_Selection) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokSelected, _tokPrimvarValue));
    }
    if (dirty & Dirty_Visibility) {
        locators.insert(HdDataSourceLocator(_tokVisibility, _tokVisibility));
    }
    return locators;
}

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::PointNoticesFor(uint32_t dirty)
{
    HdDataSourceLocatorSet locators;
    if (dirty & (usdGenPomade::PomadeDirty_Points |
                 usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    }
    if (dirty & (Dirty_Uniforms | usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokTubeId, _tokPrimvarValue));
    }
    if (dirty & (Dirty_Widths | usdGenPomade::PomadeDirty_Topology)) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokWidths, _tokPrimvarValue));
    }
    if (dirty & usdGenPomade::PomadeDirty_Selection) {
        // A CV dot carries its state in its colour (white selected, yellow
        // hovered), so selection dirties displayColor and nothing else.
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
    }
    if (dirty & Dirty_Visibility) {
        locators.insert(HdDataSourceLocator(_tokVisibility, _tokVisibility));
    }
    return locators;
}

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::GuideNoticesFor(uint32_t dirty, bool countChanged)
{
    HdDataSourceLocatorSet locators;
    if (dirty & Dirty_Visibility) {
        locators.insert(HdDataSourceLocator(_tokVisibility, _tokVisibility));
    }
    if (!(dirty & usdGenPomade::PomadeDirty_Guides)) {
        return locators;
    }
    if (countChanged) {
        locators.insert(HdDataSourceLocator(_tokBasisCurves, _tokTopology));
    }
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokWidths, _tokPrimvarValue));
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokHairT, _tokPrimvarValue));
    locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
    locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    return locators;
}

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::GraphNoticesFor(uint32_t dirty)
{
    // One locator set per overlay prim would be most precise; the graph and
    // the tint restage together (both derive from one snapshot), so one set
    // covers the overlay: nodes/edges points + topology on graph edits, the
    // tint leaves on region edits. Each entry below is addressed to its own
    // prim by the Refresh fan-out.
    HdDataSourceLocatorSet locators;
    if (dirty & usdGenPomade::PomadeDirty_Graph) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
        // Widths ride with points: the dot array grows alongside them, and
        // without this locator Storm keeps the stale size and drops every
        // node placed after the first.
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokWidths, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokBasisCurves, _tokTopology));
        // Region patches are clipped against the graph's current contours,
        // so a graph edit can add or remove patch triangles inside the same
        // coarse scalp face.
        locators.insert(HdDataSourceLocator(_tokMesh, _tokTopology));
        locators.insert(HdDataSourceLocator(
            _tokPrimvars, TfToken("usdGen:pomadeRegion"), _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    }
    if (dirty & usdGenPomade::PomadeDirty_Regions) {
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(
            _tokPrimvars, TfToken("usdGen:pomadeRegion"), _tokPrimvarValue));
        locators.insert(HdDataSourceLocator(_tokMesh, _tokTopology));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
        locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    }
    if (dirty & usdGenPomade::PomadeDirty_Selection) {
        locators.insert(HdDataSourceLocator(_tokPrimvars, _tokDisplayColor,
                                            _tokPrimvarValue));
    }
    return locators;
}

HdDataSourceLocatorSet
UsdGenPomadeSceneIndex::OverlayNoticesFor(uint32_t dirty, bool countChanged)
{
    HdDataSourceLocatorSet locators;
    if (!(dirty & (usdGenPomade::PomadeDirty_Gizmo |
                   usdGenPomade::PomadeDirty_Brush))) {
        return locators;
    }
    if (countChanged) {
        locators.insert(HdDataSourceLocator(_tokBasisCurves, _tokTopology));
        locators.insert(
            HdDataSourceLocator(_tokPrimvars, _tokHandleId, _tokPrimvarValue));
    }
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokPoints, _tokPrimvarValue));
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokDisplayColor, _tokPrimvarValue));
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokWidths, _tokPrimvarValue));
    locators.insert(
        HdDataSourceLocator(_tokPrimvars, _tokActive, _tokPrimvarValue));
    locators.insert(HdDataSourceLocator(_tokExtent, _tokMin));
    locators.insert(HdDataSourceLocator(_tokExtent, _tokMax));
    return locators;
}

// -- publication --------------------------------------------------------------

void
UsdGenPomadeSceneIndex::_BuildTestTubePrims(
    std::map<SdfPath, _Prim> *prims) const
{
    usdGenPomade::PomadeStagedTubeMesh tube;
    usdGenPomade::PomadeStageTestTubeMesh(usdGenPomade::PomadeTubeShape(), &tube);
    (*prims)[TestTubePath()] =
        _Prim{_meshType, _BuildTestTubeDataSource(tube, TubeMaterialPath())};
}

void
UsdGenPomadeSceneIndex::_BuildLevelPrims(
    usdGenPomade::PomadeStagedModel const &staged,
    std::map<SdfPath, _Prim> *prims) const
{
    if (staged.levels.empty()) {
        return;
    }
    (*prims)[TubesScopePath()] =
        _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    (*prims)[CentersScopePath()] =
        _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    (*prims)[CenterCVsScopePath()] =
        _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    (*prims)[RingsScopePath()] =
        _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    (*prims)[RingCVsScopePath()] =
        _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    for (auto const &kv : staged.levels) {
        int const level = kv.first;
        usdGenPomade::PomadeStagedLevel const &staging = kv.second;
        // A point glyph spells one explicitly active component domain. The
        // display policy carries those bits independently of ring geometry:
        // a selected ring in Center/Hierarchy must not hide center dots just
        // because its ring vertices happen to be staged.
        bool const showCenterCVs = staging.visible && staging.centers &&
            staging.centerCVDots;
        bool const showRingCVs = staging.visible && staging.centers &&
            !staging.centersOnly && staging.ringCVDots &&
            !staging.ringCVPoints.empty();
        // An x-rayed level binds the translucent twin so it stops writing
        // depth and the overlays inside it come through (see
        // _kXrayMaterialOpacity).
        (*prims)[TubesPath(level)] =
            _Prim{_meshType,
                  _BuildLevelMeshDataSource(
                      staging, staging.xray ? TubeXrayMaterialPath()
                                            : TubeMaterialPath())};
        (*prims)[CentersPath(level)] =
            _Prim{_curvesType,
                  _BuildCurvesDataSource(
                      staging.centerPoints, staging.centerVertexCounts,
                      staging.centerIndices, staging.centerCurveColor,
                      staging.centerCurveWidth, staging.centerCurveTubeId,
                      staging.centerCurveSelected, TfToken(), VtIntArray(),
                      staging.centerMin, staging.centerMax,
                      staging.visible && staging.centers,
                      OverlayMaterialPath())};
        (*prims)[CenterCVsPath(level)] =
            _Prim{_pointsType,
                  _BuildPointsDataSource(
                      staging.centerPoints, staging.centerCVColor,
                      staging.centerCVWidth, staging.centerCVTubeId,
                      _tokCvIndex, staging.centerCVIndex, staging.centerMin,
                      staging.centerMax, showCenterCVs,
                      OverlayMaterialPath())};
        (*prims)[RingsPath(level)] =
            _Prim{_curvesType,
                  _BuildCurvesDataSource(
                      staging.ringPoints, staging.ringVertexCounts,
                      staging.ringIndices, staging.ringCurveColor,
                      staging.ringCurveWidth, staging.ringCurveTubeId,
                      staging.ringCurveSelected, _tokSectionIndex,
                      staging.ringCurveSection,
                      staging.ringMin, staging.ringMax,
                      staging.visible && staging.centers &&
                          !staging.centersOnly &&
                          !staging.ringPoints.empty(),
                      OverlayMaterialPath())};
        (*prims)[RingCVsPath(level)] =
            _Prim{_pointsType,
                  _BuildPointsDataSource(
                      staging.ringCVPoints, staging.ringCVColor,
                      staging.ringCVWidth, staging.ringCVTubeId,
                      _tokSectionIndex, staging.ringCVSection, staging.ringMin,
                      staging.ringMax, showRingCVs,
                      OverlayMaterialPath())};
        if (staging.guideCount > 0 && !staging.guidePoints.empty()) {
            (*prims)[GuidesScopePath()] =
                _Prim{_scopeType, HdRetainedContainerDataSource::New()};
            (*prims)[GuidesPath(level)] =
                _Prim{_curvesType,
                      _BuildGuideDataSource(staging, HairMaterialPath())};
        }
    }
}

void
UsdGenPomadeSceneIndex::_BuildGraphPrims(usdGenPomade::PomadeModel &model,
                                        std::map<SdfPath, _Prim> *prims) const
{
    if (!model.HasScalp()) {
        return;
    }
    usdGenPomade::PomadeModel::GraphSnapshot snap = model.SnapshotGraph();
    std::shared_ptr<usdGenPomade::PomadeScalpMesh const> scalp =
        model.GetScalp();
    // Selected graph items draw white, the hovered one yellow (plan/18
    // §2.4a); the palette matches the CV dots so one rule reads across
    // the whole overlay.
    static GfVec3f const kSelected(1.0f, 1.0f, 1.0f);
    static GfVec3f const kHover(1.0f, 0.85f, 0.10f);
    std::set<int> selectedNodes;
    std::set<int> selectedEdges;
    for (usdGenPomade::PomadeSelectionItem const &item :
         model.SelectionItems(usdGenPomade::PomadePick_GraphNode |
                              usdGenPomade::PomadePick_GraphEdge)) {
        if (item.kind == usdGenPomade::PomadePick_GraphNode) {
            selectedNodes.insert(item.id);
        } else {
            selectedEdges.insert(item.id);
        }
    }
    usdGenPomade::PomadeSelectionItem const hover = model.SelectionHover();
    // The overlay stack, bottom to top: scalp, region tint, graph. The
    // tint copies the scalp lifted 2e-3 of its diagonal off it, and the
    // graph sits exactly ON the scalp — so without its own (larger) lift
    // the tint covers every node and edge and clicks land in the model
    // but leave nothing on the screen. Twice the tint offset keeps the
    // stack ordered at every framing; the graph stays depth-tested, so it
    // still hides behind the scalp from the back.
    float const graphLift = usdGenPomade::PomadeGraphDisplayLift(scalp.get());
    float const tintOffset = 0.5f * graphLift;
    // Nodes: white, coincident (unwelded) sets warn orange, selected
    // white-bright and the hovered one yellow.
    VtVec3fArray nodePoints;
    VtVec3fArray nodeColors;
    nodePoints.reserve(snap.nodes.size());
    nodeColors.reserve(snap.nodes.size());
    std::set<int> coincident;
    for (auto const &group :
         model.GetGraph().CoincidentSets(model.GetSnapRadius())) {
        coincident.insert(group.begin(), group.end());
    }
    for (auto const &nd : snap.nodes) {
        float displayP[3];
        usdGenPomade::PomadeGraphDisplayPosition(nd, scalp.get(), displayP);
        GfVec3f const np(displayP[0], displayP[1], displayP[2]);
        nodePoints.push_back(np);
        bool const hovered =
            hover.kind == usdGenPomade::PomadePick_GraphNode &&
            hover.id == nd.id;
        if (hovered) {
            nodeColors.push_back(kHover);
        } else if (selectedNodes.count(nd.id)) {
            nodeColors.push_back(kSelected);
        } else {
            nodeColors.push_back(coincident.count(nd.id)
                                     ? GfVec3f(1.0f, 0.5f, 0.1f)
                                     : GfVec3f(1.0f));
        }
    }
    // Completed graph boundaries are white, providing a clean seam between
    // neighbouring saturated region patches.
    VtIntArray curveCounts;
    VtIntArray curveIndices;
    VtVec3fArray edgePoints;
    VtVec3fArray edgeColors;
    {
        int cursor = 0;
        for (auto const &e : model.GetGraph().Edges()) {
            if (!e.alive || e.polyline.size() < 6) {
                continue;
            }
            int const n = int(e.polyline.size() / 3);
            curveCounts.push_back(n);
            for (int i = 0; i < n; ++i) {
                curveIndices.push_back(cursor + i);
                float const q[3] = {e.polyline[size_t(i) * 3 + 0],
                                    e.polyline[size_t(i) * 3 + 1],
                                    e.polyline[size_t(i) * 3 + 2]};
                float displayP[3];
                usdGenPomade::PomadeGraphDisplaySurfacePosition(
                    q, scalp.get(), displayP);
                GfVec3f const ep(displayP[0], displayP[1], displayP[2]);
                edgePoints.push_back(ep);
            }
            cursor += n;
            bool const hovered =
                hover.kind == usdGenPomade::PomadePick_GraphEdge &&
                hover.id == e.id;
            if (hovered) {
                edgeColors.push_back(kHover);
            } else if (selectedEdges.count(e.id)) {
                edgeColors.push_back(kSelected);
            } else {
                edgeColors.push_back(GfVec3f(1.0f));
            }
        }
    }
    // Tint mesh: scalp topology + per-face region colours from the clump
    // palette, so the scalp patch and the tube rooted in it are the same
    // colour (plan/18 §2.4a). Uncovered is dark red, intersected magenta.
    //
    // The colour is keyed on the face's REGION id (`faceRegionIds`, the
    // lowest region claiming the face), which is the id the L1 tube rooted
    // there carries in its desc, so the two palette lookups are the same
    // lookup. Keying it on the face's interp id — the union-find channel
    // the bake writes — was the V0..V7 spelling and is not the same thing:
    // linked regions share an interp id but keep their own region ids and
    // their own tubes, so the patch took whichever member the map happened
    // to see first while the tubes each took their own colour.
    //
    // The mesh is also pushed off the scalp along its own vertex normals.
    // It carries the scalp's points verbatim otherwise, and two coincident
    // opaque meshes resolve by depth-buffer luck: the golden braid frame
    // showed the tint winning 36 % of the scalp's pixels and the bound
    // mesh the rest, and in the workspace shot the tint lost everywhere,
    // so the artist saw an untinted head. The offset is a fixed fraction
    // of the scalp's bounding diagonal, which puts it two orders of
    // magnitude above the depth resolution at any sane framing while
    // staying under a pixel or two of parallax.
    VtVec3fArray scalpPoints;
    VtIntArray scalpCounts;
    VtIntArray scalpIndices;
    VtVec3fArray faceColors;
    VtIntArray faceRegions;
    if (scalp && scalp->finalized) {
        size_t const vertexCount = scalp->points.size() / 3;
        // A face GeomSubset scalp tints its own faces only: the rest of the
        // parent mesh is not scalp, so painting it "uncovered" would be a
        // lie (plan/02 §2.20). The whole-mesh case keeps the bulk copy.
        if (scalp->activeFaces.empty()) {
            scalpCounts.assign(scalp->faceVertexCounts.begin(),
                               scalp->faceVertexCounts.end());
            scalpIndices.assign(scalp->faceVertexIndices.begin(),
                                scalp->faceVertexIndices.end());
        } else {
            for (int f : scalp->activeFaces) {
                int const off = scalp->faceOffsets[size_t(f)];
                int const n = scalp->faceVertexCounts[size_t(f)];
                scalpCounts.push_back(n);
                scalpIndices.insert(
                    scalpIndices.end(),
                    scalp->faceVertexIndices.begin() + off,
                    scalp->faceVertexIndices.begin() + off + n);
            }
        }
        scalpPoints.assign(vertexCount, GfVec3f(0.0f));
        std::vector<GfVec3f> vertexNormal(vertexCount, GfVec3f(0.0f));
        for (size_t f = 0; f + 1 < scalp->faceOffsets.size(); ++f) {
            if (f * 3 + 2 >= scalp->faceNormals.size()) {
                break;
            }
            GfVec3f const n(scalp->faceNormals[f * 3 + 0],
                            scalp->faceNormals[f * 3 + 1],
                            scalp->faceNormals[f * 3 + 2]);
            for (int k = scalp->faceOffsets[f]; k < scalp->faceOffsets[f + 1];
                 ++k) {
                int const v = scalp->faceVertexIndices[size_t(k)];
                if (v >= 0 && size_t(v) < vertexCount) {
                    vertexNormal[size_t(v)] += n;
                }
            }
        }
        for (size_t i = 0; i < vertexCount; ++i) {
            scalpPoints[i] = GfVec3f(scalp->points[i * 3 + 0],
                                     scalp->points[i * 3 + 1],
                                     scalp->points[i * 3 + 2]);
        }
        float const offset = tintOffset;
        if (offset > 0.0f) {
            for (size_t i = 0; i < vertexCount; ++i) {
                GfVec3f n = vertexNormal[i];
                if (n.GetLengthSq() > 1e-20f) {
                    scalpPoints[i] += n.GetNormalized() * offset;
                }
            }
        }
        // A coarse base represents the uncovered scalp.  The loop patches
        // below are sparse: only a contour-covered face receives extra
        // triangles, so this does not turn every scalp quad into a tile grid.
        for (size_t f = 0; f < scalpCounts.size(); ++f) {
            faceColors.push_back(GfVec3f(0.35f, 0.05f, 0.05f));
            faceRegions.push_back(-1);
        }
        usdGenPomade::PomadeRegionLoops const loops =
            _SnapshotDisplayLoops(snap);
        if (loops.valid) {
            float const patchOffset = 1.5f * tintOffset;
            auto restPoint = [&](int vertex) {
                return GfVec3f(scalp->points[size_t(vertex) * 3 + 0],
                               scalp->points[size_t(vertex) * 3 + 1],
                               scalp->points[size_t(vertex) * 3 + 2]);
            };
            auto project = [&](size_t region, GfVec3f const &p) {
                float const *pp = &loops.planeP[region * 3];
                float const *u = &loops.basisU[region * 3];
                float const *v = &loops.basisV[region * 3];
                float const dx = p[0] - pp[0];
                float const dy = p[1] - pp[1];
                float const dz = p[2] - pp[2];
                return _Patch2{dx * u[0] + dy * u[1] + dz * u[2],
                               dx * v[0] + dy * v[1] + dz * v[2]};
            };
            for (size_t region = 0; region < loops.loopCount.size();
                 ++region) {
                int const begin = loops.loopBegin[region];
                int const count = loops.loopCount[region];
                if (count < 3 || size_t(begin + count) * 3 >
                                     loops.points.size()) {
                    continue;
                }
                std::vector<_Patch2> contour;
                contour.reserve(size_t(count));
                for (int i = 0; i < count; ++i) {
                    size_t const at = size_t(begin + i) * 3;
                    _Patch2 const p = project(
                        region, GfVec3f(loops.points[at],
                                        loops.points[at + 1],
                                        loops.points[at + 2]));
                    if (contour.empty() ||
                        std::fabs(p.x - contour.back().x) > 1.0e-6f ||
                        std::fabs(p.y - contour.back().y) > 1.0e-6f) {
                        contour.push_back(p);
                    }
                }
                if (contour.size() > 2 &&
                    std::fabs(contour.front().x - contour.back().x) <=
                        1.0e-6f &&
                    std::fabs(contour.front().y - contour.back().y) <=
                        1.0e-6f) {
                    contour.pop_back();
                }
                contour = _RemoveCollinearPatchPoints(std::move(contour));
                std::vector<std::array<int, 3>> const triangles =
                    _TriangulatePatch(contour);
                if (triangles.empty()) {
                    continue;
                }
                int const regionId = loops.regionIds[region];
                int const interp = loops.interpIds[region];
                usdGenPomade::PomadeRgb const rgb =
                    usdGenPomade::PomadeClumpColor(regionId, 1, -1);
                GfVec3f const colour(rgb.r, rgb.g, rgb.b);
                size_t const scalpFaceCount = scalp->faceVertexCounts.size();
                for (size_t f = 0; f < scalpFaceCount; ++f) {
                    if (!usdGenPomade::PomadeScalpFaceActive(*scalp, int(f))) {
                        continue;  // outside the face subset: no patch
                    }
                    int const n = scalp->faceVertexCounts[f];
                    int const off = scalp->faceOffsets[f];
                    if (n < 3 || off < 0 ||
                        size_t(off + n) > scalp->faceVertexIndices.size()) {
                        continue;
                    }
                    int const ia = scalp->faceVertexIndices[size_t(off)];
                    if (ia < 0 || size_t(ia) >= vertexCount) {
                        continue;
                    }
                    for (int corner = 1; corner + 1 < n; ++corner) {
                        int const ib =
                            scalp->faceVertexIndices[size_t(off + corner)];
                        int const ic = scalp->faceVertexIndices[
                            size_t(off + corner + 1)];
                        if (ib < 0 || ic < 0 || size_t(ib) >= vertexCount ||
                            size_t(ic) >= vertexCount) {
                            continue;
                        }
                        GfVec3f const pa = restPoint(ia);
                        GfVec3f const pb = restPoint(ib);
                        GfVec3f const pc = restPoint(ic);
                        _Patch2 const qa = project(region, pa);
                        _Patch2 const qb = project(region, pb);
                        _Patch2 const qc = project(region, pc);
                        if (std::fabs(_Cross2(qa, qb, qc)) <= 1.0e-10f) {
                            continue;
                        }
                        float const minX = std::min(qa.x, std::min(qb.x, qc.x));
                        float const maxX = std::max(qa.x, std::max(qb.x, qc.x));
                        float const minY = std::min(qa.y, std::min(qb.y, qc.y));
                        float const maxY = std::max(qa.y, std::max(qb.y, qc.y));
                        for (std::array<int, 3> const &tri : triangles) {
                            _Patch2 const &ta = contour[size_t(tri[0])];
                            _Patch2 const &tb = contour[size_t(tri[1])];
                            _Patch2 const &tc = contour[size_t(tri[2])];
                            float const triMinX = std::min(ta.x, std::min(tb.x, tc.x));
                            float const triMaxX = std::max(ta.x, std::max(tb.x, tc.x));
                            float const triMinY = std::min(ta.y, std::min(tb.y, tc.y));
                            float const triMaxY = std::max(ta.y, std::max(tb.y, tc.y));
                            if (triMaxX < minX || triMinX > maxX ||
                                triMaxY < minY || triMinY > maxY) {
                                continue;
                            }
                            std::vector<_Patch2> clipped = _ClipPatchToTriangle(
                                {ta, tb, tc}, qa, qb, qc);
                            if (clipped.size() < 3 ||
                                std::fabs(_PolygonArea2(clipped)) <= 1.0e-10f) {
                                continue;
                            }
                            int const first = int(scalpPoints.size());
                            GfVec3f const na = vertexNormal[size_t(ia)];
                            GfVec3f const nb = vertexNormal[size_t(ib)];
                            GfVec3f const nc = vertexNormal[size_t(ic)];
                            for (_Patch2 const &p : clipped) {
                                scalpPoints.push_back(_LiftedPatchPoint(
                                    p, qa, qb, qc, pa, pb, pc, na, nb, nc,
                                    patchOffset));
                            }
                            // A graph loop may be walked either direction,
                            // while the scalp's winding is authoritative for
                            // its visible side.  Match every clipped fan to
                            // that local face triangle; otherwise a valid
                            // region drawn on a -Y (or inward-wound) scalp
                            // can be back-face culled even though its base
                            // overlay is double sided.
                            bool const sameWinding =
                                _PolygonArea2(clipped) * _Cross2(qa, qb, qc) >=
                                0.0f;
                            for (size_t i = 1; i + 1 < clipped.size(); ++i) {
                                scalpCounts.push_back(3);
                                scalpIndices.push_back(first);
                                scalpIndices.push_back(
                                    first + int(sameWinding ? i : i + 1));
                                scalpIndices.push_back(
                                    first + int(sameWinding ? i + 1 : i));
                                faceColors.push_back(colour);
                                faceRegions.push_back(interp);
                            }
                        }
                    }
                }
            }
        }
    }
    if (!nodePoints.empty()) {
        // Node dots are a fixed pixel size like every other overlay dot
        // (plan/18 §2.4a); before V8 they published no `widths` at all and
        // took whatever Storm's default point size happened to be. With no
        // camera resolved they fall back to a fraction of the scalp's mean
        // edge, which is the only length the graph knows about itself.
        float const nodeWidth = usdGenPomade::PomadeOverlayWidth(
            usdGenPomade::PomadeOverlayPixels::kGraphNode,
            model.GetDisplayScale(),
            scalp && scalp->meanEdgeLength > 0.0f
                ? scalp->meanEdgeLength * 0.12f
                : 0.05f);
        (*prims)[GraphNodesPath()] =
            _Prim{_pointsType,
                  _BuildNodesDataSource(nodePoints, nodeColors,
                                        VtFloatArray(nodePoints.size(),
                                                     nodeWidth),
                                        OverlayMaterialPath())};
    }
    if (!edgePoints.empty()) {
        (*prims)[GraphEdgesPath()] =
            _Prim{_curvesType,
                  _BuildEdgesDataSource(curveCounts, curveIndices,
                                        edgePoints, edgeColors,
                                        OverlayMaterialPath())};
    }
    if (!scalpPoints.empty()) {
        (*prims)[GraphRegionsPath()] =
            _Prim{_meshType,
                  _BuildRegionsDataSource(scalpPoints, scalpCounts,
                                          scalpIndices, faceColors,
                                          faceRegions)};
    }
}

UsdGenPomadeSceneIndex::_OverlayDirty
UsdGenPomadeSceneIndex::_BuildOverlayPrims(usdGenPomade::PomadeModel &model,
                                          std::map<SdfPath, _Prim> *prims)
{
    _OverlayDirty dirty;
    usdGenPomade::PomadeGizmoRecord const gizmo = model.GetGizmo();
    usdGenPomade::PomadeBrushRingRecord const brush = model.GetBrushRing();
    usdGenPomade::PomadeOverlayCurves curves;
    if (usdGenPomade::PomadeBuildGizmoCurves(gizmo, &curves)) {
        (*prims)[GizmoPath()] =
            _Prim{_curvesType,
                  _BuildOverlayDataSource(curves, OverlayMaterialPath())};
        if (gizmo != _publishedGizmo) {
            dirty.gizmo = usdGenPomade::PomadeDirty_Gizmo;
        }
        dirty.gizmoCountChanged =
            curves.CurveCount() != _publishedGizmoCurves;
        _publishedGizmoCurves = curves.CurveCount();
    } else {
        _publishedGizmoCurves = 0;
    }
    if (usdGenPomade::PomadeBuildBrushRingCurves(brush, &curves)) {
        (*prims)[BrushRingPath()] =
            _Prim{_curvesType,
                  _BuildOverlayDataSource(curves, OverlayMaterialPath())};
        if (brush != _publishedBrush) {
            dirty.brush = usdGenPomade::PomadeDirty_Brush;
        }
        dirty.brushCountChanged =
            curves.CurveCount() != _publishedBrushCurves;
        _publishedBrushCurves = curves.CurveCount();
    } else {
        _publishedBrushCurves = 0;
    }
    _publishedGizmo = gizmo;
    _publishedBrush = brush;
    return dirty;
}

void
UsdGenPomadeSceneIndex::_CollectLevelDirties(
    usdGenPomade::PomadeStagedModel const &previous,
    usdGenPomade::PomadeStagedModel const &current, uint32_t modelDirty,
    HdSceneIndexObserver::DirtiedPrimEntries *out) const
{
    for (auto const &kv : current.levels) {
        int const level = kv.first;
        usdGenPomade::PomadeStagedLevel const &now = kv.second;
        auto prevIt = previous.levels.find(level);
        if (prevIt == previous.levels.end()) {
            continue;  // brand new level: PrimsAdded covers it
        }
        usdGenPomade::PomadeStagedLevel const &was = prevIt->second;
        uint32_t dirty = 0;
        if (was.topologyHash != now.topologyHash) {
            dirty |= usdGenPomade::PomadeDirty_Topology;
        } else if (was.pointsHash != now.pointsHash) {
            dirty |= usdGenPomade::PomadeDirty_Points;
        }
        if (was.uniformHash != now.uniformHash) {
            dirty |= Dirty_Uniforms;
        }
        if (was.selectionHash != now.selectionHash) {
            dirty |= usdGenPomade::PomadeDirty_Selection;
        }
        if (was.focused != now.focused ||
            was.displayScale != now.displayScale) {
            dirty |= Dirty_Widths;
        }
        if (was.xray != now.xray || was.xrayOpacity != now.xrayOpacity) {
            dirty |= Dirty_Xray;
        }
        if (was.xray != now.xray) {
            // Opaque and x-ray are two materials, not two parameters.
            dirty |= Dirty_Material;
        }
        if (was.visible != now.visible ||
            was.centersOnly != now.centersOnly ||
            was.centerCVDots != now.centerCVDots ||
            was.ringCVDots != now.ringCVDots ||
            was.guidesVisible != now.guidesVisible) {
            dirty |= Dirty_Visibility;
        }
        HdDataSourceLocatorSet const mesh = NoticesFor(dirty);
        if (!mesh.IsEmpty()) {
            out->push_back({TubesPath(level), mesh});
        }
        // The overlays never carry x-ray and never swap material;
        // everything else applies. Center curves and the two explicit dot
        // domains can change visibility without touching the mesh.
        uint32_t overlay = dirty & ~(Dirty_Xray | Dirty_Material);
        if (was.centers != now.centers ||
            was.centerCVDots != now.centerCVDots ||
            was.ringCVDots != now.ringCVDots) {
            overlay |= Dirty_Visibility;
        }
        // Ring visibility (plan/18 §2.4a) rides on its own hash, so that
        // selecting a tube dirties the two ring prims and nothing else.
        uint32_t const ringOverlay =
            was.ringHash == now.ringHash
                ? overlay
                : (overlay | usdGenPomade::PomadeDirty_Topology |
                   Dirty_Visibility);
        HdDataSourceLocatorSet const curves = CurveNoticesFor(overlay);
        if (!curves.IsEmpty()) {
            out->push_back({CentersPath(level), curves});
        }
        HdDataSourceLocatorSet const ringCurves = CurveNoticesFor(ringOverlay);
        if (!ringCurves.IsEmpty()) {
            out->push_back({RingsPath(level), ringCurves});
        }
        HdDataSourceLocatorSet const points = PointNoticesFor(overlay);
        if (!points.IsEmpty()) {
            out->push_back({CenterCVsPath(level), points});
        }
        HdDataSourceLocatorSet const ringPoints = PointNoticesFor(ringOverlay);
        if (!ringPoints.IsEmpty()) {
            out->push_back({RingCVsPath(level), ringPoints});
        }
        if (now.guideCount > 0 && was.guideCount > 0) {
            uint32_t const guideDirty =
                (modelDirty & usdGenPomade::PomadeDirty_Guides) |
                (dirty & Dirty_Visibility);
            HdDataSourceLocatorSet const guides = GuideNoticesFor(
                guideDirty, now.guideCount != was.guideCount);
            if (!guides.IsEmpty()) {
                out->push_back({GuidesPath(level), guides});
            }
        }
    }
}

void
UsdGenPomadeSceneIndex::_Commit(
    std::map<SdfPath, _Prim> &&prims,
    HdSceneIndexObserver::DirtiedPrimEntries const &dirtied)
{
    HdSceneIndexObserver::AddedPrimEntries added;
    HdSceneIndexObserver::RemovedPrimEntries removed;
    {
        usdGen::UsdGenRef<_Shared::View> const current = _shared->view.Load();
        for (auto const &kv : current->prims) {
            if (!prims.count(kv.first)) {
                removed.push_back({kv.first});
            }
        }
        for (auto const &kv : prims) {
            auto it = current->prims.find(kv.first);
            if (it == current->prims.end() ||
                it->second.primType != kv.second.primType) {
                added.push_back({kv.first, kv.second.primType});
            }
        }
        // One store publishes the table with the census PublishModel
        // staged; GetPrim sees either the old view or the new one.
        usdGen::UsdGenRef<_Shared::View> next =
            usdGen::UsdGenMakeRef<_Shared::View>();
        next->prims = std::move(prims);
        next->hiddenGuides = current->hiddenGuides;
        next->levels = std::move(_shared->nextLevels);
        next->restagedTubeCount = _shared->nextRestaged;
        next->testTube = next->prims.count(TestTubePath()) != 0;
        _shared->view.Store(std::move(next));
    }
    if (!removed.empty()) {
        _SendPrimsRemoved(removed);
    }
    if (!added.empty()) {
        _SendPrimsAdded(added);
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
}

bool
UsdGenPomadeSceneIndex::PublishModel(usdGenPomade::PomadeModel *model,
                                    uint32_t dirtyMask)
{
    std::map<SdfPath, _Prim> prims;
    prims[RootPath()] = _Prim{_scopeType, HdRetainedContainerDataSource::New()};
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    usdGenPomade::PomadeStagedModel staged;
    uint32_t modelDirty = dirtyMask;
    _OverlayDirty overlayDirty;
    bool anything = false;
    int restaged = 0;

    if (model != _stagedModel) {
        _publisher->Clear();
        _stagedModel = model;
    }
    if (model) {
        // A model has been active: the test tube is a harness convenience,
        // not a fallback, and never shares the frame with real geometry.
        _testTubeRetired = true;
        modelDirty |= model->TakeDirty();
        if (_publisher->Stage(*model)) {
            staged = _publisher->Current();
            restaged = staged.restagedTubeCount;
            _BuildLevelPrims(staged, &prims);
            anything = true;
        }
        _BuildGraphPrims(*model, &prims);
        overlayDirty = _BuildOverlayPrims(*model, &prims);
        anything = anything || model->HasScalp() ||
                   prims.count(GizmoPath()) || prims.count(BrushRingPath());
    } else if (!_testTubeRetired && TfGetEnvSetting(USDGENPOMADE_TEST_TUBE)) {
        _BuildTestTubePrims(&prims);
        anything = true;
    }

    if (anything) {
        prims[TubeMaterialPath()] =
            _Prim{_materialType,
                  _SurfaceMaterial(TfToken("UsdGenPomadeTube"),
                                   {_tokUnlit, _tokOpacity},
                                   {_Samp(0.0f), _Samp(1.0f)})};
        // The x-ray twin. Same shader, same parameters but one: `opacity`
        // below 1. That single number is the whole mechanism by which an
        // x-rayed tube stops hiding the curves inside it (see the comment
        // on _kXrayMaterialOpacity above).
        prims[TubeXrayMaterialPath()] =
            _Prim{_materialType,
                  _SurfaceMaterial(TfToken("UsdGenPomadeTube"),
                                   {_tokUnlit, _tokOpacity},
                                   {_Samp(0.0f),
                                    _Samp(_kXrayMaterialOpacity)})};
        prims[HairMaterialPath()] =
            _Prim{_materialType, _GuidePreviewMaterial()};
        prims[OverlayMaterialPath()] =
            _Prim{_materialType,
                  _SurfaceMaterial(TfToken("UsdGenPomadeTube"),
                                   {_tokUnlit}, {_Samp(1.0f)})};
    }

    if (!model) {
        // No active model: the overlays belong to it and go with it.
        _publishedGizmo = usdGenPomade::PomadeGizmoRecord();
        _publishedBrush = usdGenPomade::PomadeBrushRingRecord();
        _publishedGizmoCurves = 0;
        _publishedBrushCurves = 0;
    }
    // plan/17 §3.2: the cook's tiles follow "show amplified hair" and the
    // gesture bracket. With no model there is nothing suppressing them.
    _SetAmplifiedTilesVisible(model ? staged.amplifiedTilesVisible : true);
    _SetHiddenGuidesPath(model ? CommittedGuidesPath(model->GetGroomPath())
                               : SdfPath());
    _CollectLevelDirties(*_published, staged, modelDirty, &dirtied);
    // The graph overlay restages wholesale; its notices stay as they were.
    HdDataSourceLocatorSet const graph = GraphNoticesFor(modelDirty);
    usdGen::UsdGenRef<_Shared::View> const shown = _shared->view.Load();
    if (!graph.IsEmpty()) {
        for (SdfPath const &path : {GraphNodesPath(), GraphEdgesPath(),
                                    GraphRegionsPath()}) {
            if (shown->prims.count(path) && prims.count(path)) {
                dirtied.push_back({path, graph});
            }
        }
    }

    // The overlays, one prim at a time: moving the brush ring must not
    // dirty a leaf of the gizmo. A record that appeared or vanished is an
    // add/remove, which _Commit sends instead.
    {
        HdDataSourceLocatorSet const gizmoSet = OverlayNoticesFor(
            overlayDirty.gizmo | (modelDirty & usdGenPomade::PomadeDirty_Gizmo),
            overlayDirty.gizmoCountChanged);
        if (!gizmoSet.IsEmpty() && shown->prims.count(GizmoPath()) &&
            prims.count(GizmoPath())) {
            dirtied.push_back({GizmoPath(), gizmoSet});
        }
        HdDataSourceLocatorSet const brushSet = OverlayNoticesFor(
            overlayDirty.brush | (modelDirty & usdGenPomade::PomadeDirty_Brush),
            overlayDirty.brushCountChanged);
        if (!brushSet.IsEmpty() && shown->prims.count(BrushRingPath()) &&
            prims.count(BrushRingPath())) {
            dirtied.push_back({BrushRingPath(), brushSet});
        }
    }

    _shared->nextLevels.clear();
    for (auto const &kv : staged.levels) {
        _shared->nextLevels[kv.first] = {
            int(kv.second.subdivision.faceVertexCounts.size()),
            int(kv.second.subdivision.points.size()),
            int(kv.second.tubes.size())};
    }
    _shared->nextRestaged = restaged;
    *_published = std::move(staged);
    _Commit(std::move(prims), dirtied);
    return anything;
}

bool
UsdGenPomadeSceneIndex::Refresh(uint32_t dirtyMask)
{
    return PublishModel(usdGenPomade::PomadeRegistry::Get().Active(), dirtyMask);
}

bool
UsdGenPomadeSceneIndex::AmplifiedTilesVisible() const
{
    return _amplifiedTilesVisible.load();
}

size_t
UsdGenPomadeSceneIndex::AmplifiedTileCount() const
{
    usdGen::UsdGenRef<_Shared::Tiles> const tiles = _shared->tiles.Load();
    return tiles ? tiles->paths.size() : 0;
}

bool
UsdGenPomadeSceneIndex::_SetAmplifiedTilesVisible(bool visible)
{
    if (_amplifiedTilesVisible.load() == visible) {
        return false;
    }
    _amplifiedTilesVisible.store(visible);
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    if (usdGen::UsdGenRef<_Shared::Tiles> const tiles =
            _shared->tiles.Load()) {
        HdDataSourceLocatorSet const locators{
            HdDataSourceLocator(_tokVisibility)};
        for (SdfPath const &path : tiles->paths) {
            dirtied.push_back({path, locators});
        }
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
    return true;
}

SdfPath
UsdGenPomadeSceneIndex::HiddenGuidesPath() const
{
    return _shared->view.Load()->hiddenGuides;
}

SdfPath
UsdGenPomadeSceneIndex::CommittedGuidesPath(std::string const &groomPath)
{
    if (groomPath.empty() || !SdfPath::IsValidPathString(groomPath)) {
        return SdfPath();
    }
    SdfPath const groom(groomPath);
    if (!groom.IsAbsolutePath() || !groom.IsPrimPath()) {
        return SdfPath();
    }
    return groom.AppendChild(TfToken("Guides"));
}

bool
UsdGenPomadeSceneIndex::_SetHiddenGuidesPath(SdfPath const &path)
{
    HdSceneIndexObserver::DirtiedPrimEntries dirtied;
    {
        // Frontend thread: the only writer of the view.
        SdfPath const previous = _shared->view.Load()->hiddenGuides;
        if (previous == path) {
            return false;
        }
        HdDataSourceLocatorSet const locators{
            HdDataSourceLocator(_tokVisibility)};
        // Both ends: the prim that stops being hidden and the one that
        // starts. Either may be absent from the input, and a dirty on a
        // path no observer knows is harmless.
        if (!previous.IsEmpty()) {
            dirtied.push_back({previous, locators});
        }
        if (!path.IsEmpty()) {
            dirtied.push_back({path, locators});
        }
        _shared->EditView([&](_Shared::View &v) { v.hiddenGuides = path; });
    }
    if (!dirtied.empty()) {
        _SendPrimsDirtied(dirtied);
    }
    return true;
}

// -- passthrough --------------------------------------------------------------

void
UsdGenPomadeSceneIndex::_PrimsAdded(
    HdSceneIndexBase const &,
    HdSceneIndexObserver::AddedPrimEntries const &entries)
{
    // Input prims pass through; the input never announces pomade paths.
    // The amplified tiles are remembered so a "show amplified hair" flip
    // can dirty exactly their visibility (plan/17 §3.2).
    std::vector<SdfPath> tiles;
    for (auto const &entry : entries) {
        if (_IsAmplifiedTilePath(entry.primPath)) {
            tiles.push_back(entry.primPath);
        }
    }
    if (!tiles.empty()) {
        _shared->EditTiles([&](std::set<SdfPath> &paths) {
            paths.insert(tiles.begin(), tiles.end());
            return true;
        });
    }
    _SendPrimsAdded(entries);
}

void
UsdGenPomadeSceneIndex::_PrimsRemoved(
    HdSceneIndexBase const &,
    HdSceneIndexObserver::RemovedPrimEntries const &entries)
{
    if (usdGen::UsdGenRef<_Shared::Tiles> const shown =
            _shared->tiles.Load()) {
        auto touches = [&](std::set<SdfPath> const &paths) {
            for (auto const &entry : entries) {
                auto it = paths.lower_bound(entry.primPath);
                if (it != paths.end() && it->HasPrefix(entry.primPath)) {
                    return true;
                }
            }
            return false;
        };
        if (touches(shown->paths)) {
            _shared->EditTiles([&](std::set<SdfPath> &paths) {
                bool changed = false;
                for (auto const &entry : entries) {
                    auto it = paths.lower_bound(entry.primPath);
                    while (it != paths.end() &&
                           it->HasPrefix(entry.primPath)) {
                        it = paths.erase(it);
                        changed = true;
                    }
                }
                return changed;
            });
        }
    }
    _SendPrimsRemoved(entries);
}

void
UsdGenPomadeSceneIndex::_PrimsDirtied(
    HdSceneIndexBase const &,
    HdSceneIndexObserver::DirtiedPrimEntries const &entries)
{
    _SendPrimsDirtied(entries);
}

PXR_NAMESPACE_CLOSE_SCOPE
