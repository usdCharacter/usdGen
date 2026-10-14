// usdGen imaging — graph description builder implementation.
//
// The engine must never see a UsdStage (S8; link-enforced by B-1), so
// everything the graph needs is pulled to pure values here (06 §3.10,
// 03 §2.5). Missing properties keep the C1 defaults already materialized in
// the UsdGenGraphDesc member initializers; the S14 "pull everything" rule is
// honored by sweeping every usdGen:* attribute of every participating prim
// into UsdGenNodeDesc::params, whether or not a dedicated field exists.
#include "usdGenImaging/usdGenGraphDescBuilder.h"
#include "usdGenImaging/imageMapCache.h"
#include "usdGenImaging/usdGenGraphDescShared.h"
#include "usdGenImaging/usdGenTokens.h"

#include "usdGen/debugCodes.h"
#include "usdGen/expressions/valueShape.h"

#include "pxr/base/trace/trace.h"
#include "usdGen/executionBackend.h"

#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/vt/array.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/sdf/schema.h"

#include "pxr/imaging/hd/dataSource.h"
#include "pxr/imaging/hd/dataSourceTypeDefs.h"
#include "pxr/imaging/hd/geomSubsetSchema.h"
#include "pxr/imaging/hd/materialBindingSchema.h"
#include "pxr/imaging/hd/materialBindingsSchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/purposeSchema.h"
#include "pxr/imaging/hd/visibilitySchema.h"
#include "pxr/imaging/hd/xformSchema.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

class UsdGenGraphDescCaptureCache
{
public:
    struct NodeCapture {
        SdfPath path;
        usdGen::UsdGenNodeDesc node;       // no inherited surfaces / inputs
        std::vector<std::string> validationErrors;
        SdfPathVector guides;
        SdfPathVector colliders;           // usdGen:colliders (UsdGenCollide only)
        bool exists = false;
    };

    // Geometry pools, keyed by prim path: a pooled desc is reused when its
    // prim (and, for subsets, the parent mesh it reads) saw no dirty since
    // the cached capture. Errors ride along so a reused entry splices the
    // same diagnostics at the same point; a unioned subset also records
    // its member set, which must match to reuse.
    struct SurfaceCapture {
        SdfPath path;
        usdGen::UsdGenSurfaceDesc surface;
        std::vector<std::string> validationErrors;
        SdfPathVector unionMembers;
        bool exists = false;
    };
    struct CurveCapture {
        SdfPath path;
        usdGen::UsdGenCurveSetDesc curves;
        bool exists = false;
    };
    struct MapCapture {
        SdfPath path;
        usdGen::UsdGenMapDesc map;
        std::vector<std::string> validationErrors;
        // Paint maps snapshot a surface primvar outside the map prim; a
        // repaint dirties that surface, so it gates reuse with the map.
        SdfPathVector paintSources;
    };
    struct GeometryCapture {
        SdfPath path;
        usdGen::UsdGenGeometryDesc geometry;
        std::vector<std::string> validationErrors;
    };

    SdfPath description;
    double time = 0.0;
    SdfPathVector operatorOrder;
    std::vector<NodeCapture> nodes;
    std::map<std::string, SurfaceCapture> surfaces;
    std::map<std::string, CurveCapture> curves;
    std::map<std::string, MapCapture> maps;
    std::map<std::string, GeometryCapture> geometries;
};

namespace {

using usdGen::UsdGenCurveSetDesc;
using usdGen::UsdGenGraphDesc;
using usdGen::UsdGenNodeDesc;
using usdGen::UsdGenParamValue;
using usdGen::UsdGenRole;
using usdGen::UsdGenLookDesc;
using usdGen::UsdGenSurfaceDesc;
using usdGen::UsdGenSurfaceSample;
using usdGen::UsdGenSurfaceNormalDomain;
using usdGen::UsdGenSurfaceCagePayload;



// ---- Hydra-sourced reads (production path, 13 §7 V2-9 homing table) -----
//
// Mirror of the stage helpers above, reading UsdImaging data sources instead
// of Usd prims. Locator contract: the adapter overlays its mapped source at
// the prim root, so adapter-published usdGen:* properties are served FLAT
// with 02 §0.7 relative elements (usdGen:width:knots -> width/knots); an
// ancestor in a name-collision pair takes a "-value" (attribute) / "-rel"
// (relationship) suffix on its final element (primAdapter.cpp
// LocatorForProperty), which _HLocate retries transparently. Stock geometry
// arrives through the standard Hydra schemas (points, primvars,
// xform/matrix, purpose, visibility, materialBindings); the final scene
// index is flattened, so xform/matrix is already the world matrix (S4),
// matching ComputeLocalToWorldTransform. The S14 sweep prunes stock
// subtrees via _HIsStock (no usdGen prefix survives the overlay flattening).

using _HdTime = HdSampledDataSource::Time;

_HdTime
_HTime(double t)
{
    return static_cast<_HdTime>(t);
}

bool
_HPrim(HdSceneIndexBase &input, SdfPath const &path,
       HdContainerDataSourceHandle *outDs, TfToken *outType)
{
    HdSceneIndexPrim const prim = input.GetPrim(path);
    if (!prim.dataSource) {
        return false;
    }
    *outDs = prim.dataSource;
    if (outType) {
        *outType = prim.primType;
    }
    return true;
}

HdContainerDataSourceHandle
_HChild(HdContainerDataSourceHandle const &parent, char const *name)
{
    if (!parent) {
        return nullptr;
    }
    return HdContainerDataSource::Cast(parent->Get(TfToken(name)));
}

HdContainerDataSourceHandle
_HUsdGen(HdContainerDataSourceHandle const &primDs)
{
    // The adapter overlays its mapped source at the prim root
    // (OverlayedContainerDataSources(usdGen, base)), so mapped usdGen:*
    // properties are served FLAT: usdGen:tileTarget -> `tileTarget`,
    // usdGen:curve:basis -> curve/basis. There is no `usdGen` container in
    // the served tree (the `usdGen` prefix lives only on invalidation
    // locators). The mapped root IS the prim data source; stock subtrees
    // sharing it are pruned by _HIsStock during the S14 sweep.
    return primDs;
}

// Stock-served top-level names sharing the overlaid prim root. The S14 sweep
// must not mistake them for mapped usdGen:* properties (the stage builder's
// prefix filter has no Hydra equivalent once the overlay flattens serving).
bool
_HIsStock(std::string const &name)
{
    static std::unordered_set<std::string> const stock{
        "__usdGenValidationErrors",
        "primvars", "xform", "mesh", "basisCurves", "geomSubset",
        "points", "widths", "visibility", "purpose", "materialBindings",
        "extent", "proxyPrim", "xformOpOrder", "model", "geomModel",
        "primOrigin", "__usdPrimInfo", "__usdUpAxis", "skelBinding",
        "coordSysBinding", "usdMaterialBindings", "displayStyle",
        "categories",
        // Adapter-owned derived metadata is consumed explicitly by the
        // builder and must not enter S14's authored parameter sweep.
        "expressionBindings", "expressions", "operatorOrder",
        "usdGenRuntime", "usdGenCurveRest",
    };
    if (stock.count(name) != 0) {
        return true;
    }
    return name.rfind("xformOp:", 0) == 0;
}

// USD type name without a UsdPrim (B-2: UsdImagingUsdPrimInfoSchema's symbol
// contains UsdPrim, so the nm half would bite; raw reads stay clean).
// Defined after _HGetTyped below.

// Navigate intermediate containers exactly; the final element retries the
// LocatorForProperty ancestor-deviation suffixes ("-rel", then "-value").
HdDataSourceBaseHandle
_HLocate(HdContainerDataSourceHandle const &root,
         std::initializer_list<char const*> elems)
{
    HdContainerDataSourceHandle cur = root;
    size_t i = 0;
    size_t const n = elems.size();
    for (char const *e : elems) {
        if (!cur || n == 0) {
            return nullptr;
        }
        HdDataSourceBaseHandle h = cur->Get(TfToken(e));
        if (!h) {
            std::string const b(e);
            h = cur->Get(TfToken(b + "-rel"));
            if (!h) {
                h = cur->Get(TfToken(b + "-value"));
            }
        }
        if (!h) {
            return nullptr;
        }
        if (++i == n) {
            return h;
        }
        cur = HdContainerDataSource::Cast(h);
    }
    return nullptr;
}

bool
_HSampledValue(HdContainerDataSourceHandle const &root, _HdTime t,
               VtValue *out, std::initializer_list<char const*> elems)
{
    HdSampledDataSourceHandle const s =
        HdSampledDataSource::Cast(_HLocate(root, elems));
    if (!s) {
        return false;
    }
    *out = s->GetValue(t);
    return !out->IsEmpty();
}

template <class T>
bool
_HGetTyped(HdContainerDataSourceHandle const &root, _HdTime t, T *out,
           std::initializer_list<char const*> elems)
{
    VtValue value;
    if (!_HSampledValue(root, t, &value, elems)) {
        return false;
    }
    if (value.IsHolding<T>()) {
        *out = value.UncheckedGet<T>();
        return true;
    }
    if (value.CanCast<T>()) {
        *out = value.template Cast<T>().template UncheckedGet<T>();
        return true;
    }
    return false;
}

template <class T>
void
_HGetDedicated(HdContainerDataSourceHandle const &root, _HdTime t, T *out,
               std::vector<std::string> *errors, SdfPath const &path,
               char const *name, std::initializer_list<char const*> elems)
{
    VtValue value;
    if (!_HSampledValue(root, t, &value, elems)) return;
    if (value.IsHolding<T>()) {
        *out = value.UncheckedGet<T>();
        return;
    }
    errors->push_back(path.GetString() + ": " + name + " has wrong authored type");
}

bool
_HGetToken(HdContainerDataSourceHandle const &root, _HdTime t, TfToken *out,
           std::initializer_list<char const*> elems)
{
    TfToken v;
    if (_HGetTyped(root, t, &v, elems) && !v.IsEmpty()) {
        *out = v;
        return true;
    }
    return false;
}

// Backend selection is intentionally stricter than the ordinary token
// helpers.  Those helpers perform VtValue casts and cannot distinguish an
// absent property from a malformed value.  Hydra's mapped schema source
// includes the resolved "cuda" fallback, so only a missing leaf keeps the
// legacy CPU baseline.  Any present non-token/empty/unknown value is an
// explicit invalid request and must never fall through to CPU.
usdGen::UsdGenExecutionBackend
_HResolveExecutionBackend(HdContainerDataSourceHandle const &root,
                          _HdTime t)
{
    HdDataSourceBaseHandle const source =
        _HLocate(root, {"execution", "backend"});
    if (!source) {
        return usdGen::UsdGenExecutionBackend::CpuReference;
    }
    HdSampledDataSourceHandle const sampled =
        HdSampledDataSource::Cast(source);
    if (!sampled) {
        return usdGen::UsdGenExecutionBackend::Invalid;
    }
    VtValue const value = sampled->GetValue(t);
    if (!value.IsHolding<TfToken>()) {
        return usdGen::UsdGenExecutionBackend::Invalid;
    }
    TfToken const backend = value.UncheckedGet<TfToken>();
    return usdGen::ParseUsdGenExecutionBackend(backend);
}

bool
_BindingArrayCount(TfToken const& nativeType, VtValue const& literal,
                   uint32_t* count)
{
    SdfValueTypeName const type = SdfSchema::GetInstance().FindType(nativeType);
    return !type.IsArray() || usdGen::expr::FixedArrayElementCount(type, literal, count);
}

TfToken
_HUsdTypeName(HdContainerDataSourceHandle const &primDs)
{
    TfToken type;
    _HGetTyped(_HChild(primDs, "__usdPrimInfo"), _HTime(0.0), &type,
               {"typeName"});
    return type;
}

bool
_HGetPathArray(HdContainerDataSourceHandle const &root, SdfPathVector *out,
               std::initializer_list<char const*> elems)
{
    VtValue value;
    if (!_HSampledValue(root, _HTime(0.0), &value, elems)) {
        return false;
    }
    if (value.IsHolding<VtArray<SdfPath>>()) {
        VtArray<SdfPath> const &arr = value.UncheckedGet<VtArray<SdfPath>>();
        out->assign(arr.begin(), arr.end());
        return true;
    }
    if (value.IsHolding<SdfPathVector>()) {
        *out = value.UncheckedGet<SdfPathVector>();
        return true;
    }
    if (value.IsHolding<SdfPath>()) {
        out->push_back(value.UncheckedGet<SdfPath>());
        return true;
    }
    return false;
}

bool
_HPrimvarValue(HdContainerDataSourceHandle const &primDs, char const *name,
               _HdTime t, VtValue *out)
{
    // Indexed primvars arrive flattened (primvarSchema.h); fixtures author
    // plain arrays, matching the raw stage reads.
    HdSampledDataSourceHandle const s =
        HdPrimvarsSchema::GetFromParent(primDs)
            .GetPrimvar(TfToken(name))
            .GetPrimvarValue();
    if (!s) {
        return false;
    }
    *out = s->GetValue(t);
    return !out->IsEmpty();
}

// The source curves' own displayColor, forwarded so a groom that styles
// nothing shows the colour its asset already carries. Any interpolation is
// accepted: constant/uniform/vertex map onto the three authored-plane domains.
void
_HForwardSourceColor(HdContainerDataSourceHandle const &primDs, _HdTime t,
                     usdGen::UsdGenCurveSetDesc *out)
{
    HdPrimvarSchema const primvar =
        HdPrimvarsSchema::GetFromParent(primDs).GetPrimvar(TfToken("displayColor"));
    HdSampledDataSourceHandle const values = primvar.GetPrimvarValue();
    if (!values) return;
    VtValue const value = values->GetValue(t);
    if (!value.IsHolding<VtVec3fArray>()) return;
    VtVec3fArray const &colors = value.UncheckedGet<VtVec3fArray>();
    if (colors.empty()) return;

    TfToken interpolation;
    if (HdTokenDataSourceHandle const i = primvar.GetInterpolation()) {
        interpolation = i->GetTypedValue(t);
    }
    usdGen::UsdGenAuthoredPlaneDesc plane;
    plane.name = usdGen::UsdGenSourceColorPlane();
    plane.type = usdGen::UsdGenAuthoredPlaneType::Float32;
    plane.arity = 3;
    if (interpolation == TfToken("constant") || colors.size() == 1) {
        plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Groom;
    } else if (interpolation == TfToken("uniform")) {
        plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Primitive;
    } else {
        plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Point;
    }
    size_t const elements =
        plane.domain == usdGen::UsdGenAuthoredPlaneDomain::Groom ? 1 : colors.size();
    plane.floatValues.reserve(elements * 3);
    for (size_t i = 0; i < elements; ++i) {
        plane.floatValues.push_back(colors[i][0]);
        plane.floatValues.push_back(colors[i][1]);
        plane.floatValues.push_back(colors[i][2]);
    }
    out->authoredPlanes.push_back(std::move(plane));
}

// OutputCurves ownership is carried by one uniform integer per source curve.
// Do not reinterpret a malformed or differently interpolated primvar: the
// authored-plane contract requires a primitive-domain scalar.
void
_HForwardOwnershipPlane(HdContainerDataSourceHandle const &primDs,
                        TfToken const &name, _HdTime t,
                        usdGen::UsdGenCurveSetDesc *out)
{
    HdPrimvarSchema const primvar =
        HdPrimvarsSchema::GetFromParent(primDs).GetPrimvar(name);
    HdSampledDataSourceHandle const values = primvar.GetPrimvarValue();
    if (!values) return;
    HdTokenDataSourceHandle const interpolation = primvar.GetInterpolation();
    if (!interpolation || interpolation->GetTypedValue(t) != TfToken("uniform")) {
        return;
    }
    VtValue const value = values->GetValue(t);
    if (!value.IsHolding<VtIntArray>()) return;
    VtIntArray const &integers = value.UncheckedGet<VtIntArray>();
    if (integers.size() != out->curveVertexCounts.size()) return;

    usdGen::UsdGenAuthoredPlaneDesc plane;
    plane.name = name;
    plane.type = usdGen::UsdGenAuthoredPlaneType::Int32;
    plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Primitive;
    plane.arity = 1;
    plane.intValues = integers;
    out->authoredPlanes.push_back(std::move(plane));
}

template <class T>
bool
_HPrimvarTyped(HdContainerDataSourceHandle const &primDs, char const *name,
               _HdTime t, T *out)
{
    VtValue value;
    if (!_HPrimvarValue(primDs, name, t, &value)) {
        return false;
    }
    if (value.IsHolding<T>()) {
        *out = value.UncheckedGet<T>();
        return true;
    }
    if (value.CanCast<T>()) {
        *out = value.template Cast<T>().template UncheckedGet<T>();
        return true;
    }
    return false;
}

UsdGenSurfaceNormalDomain
_HNormalDomain(TfToken const& interpolation)
{
    if (interpolation == TfToken("constant")) return UsdGenSurfaceNormalDomain::Constant;
    if (interpolation == TfToken("uniform")) return UsdGenSurfaceNormalDomain::Uniform;
    if (interpolation == TfToken("vertex")) return UsdGenSurfaceNormalDomain::Vertex;
    if (interpolation == TfToken("faceVarying")) return UsdGenSurfaceNormalDomain::FaceVarying;
    return UsdGenSurfaceNormalDomain::Invalid;
}

// Suffix left by LocatorForProperty on a collision ancestor's final element.
std::string
_HStripSuffix(std::string s)
{
    for (char const *suf : {"-value", "-rel"}) {
        size_t const n = std::char_traits<char>::length(suf);
        if (s.size() > n && s.compare(s.size() - n, n, suf) == 0) {
            s.resize(s.size() - n);
            break;
        }
    }
    return s;
}

bool
_HIsPathValue(VtValue const &v)
{
    return v.IsHolding<SdfPath>() || v.IsHolding<VtArray<SdfPath>>() ||
           v.IsHolding<SdfPathVector>();
}

void
_HAppendPaths(VtValue const &v, SdfPathVector *out)
{
    if (v.IsHolding<SdfPath>()) {
        out->push_back(v.UncheckedGet<SdfPath>());
    } else if (v.IsHolding<VtArray<SdfPath>>()) {
        VtArray<SdfPath> const &arr = v.UncheckedGet<VtArray<SdfPath>>();
        out->insert(out->end(), arr.begin(), arr.end());
    } else if (v.IsHolding<SdfPathVector>()) {
        SdfPathVector const &vec = v.UncheckedGet<SdfPathVector>();
        out->insert(out->end(), vec.begin(), vec.end());
    }
}

// One recursive walk over the usdGen container fills BOTH the S14 param
// sweep (attribute leaves) and the graph edges (path-valued leaves),
// applying the stage builder's bucket switch on (base name, full name) so
// the two builders classify identically. A null node skips edge classification
// (map prims: params only).
void
_HPullUsdGen(HdContainerDataSourceHandle const &usdGen, _HdTime t,
             UsdGenNodeDesc *node, std::vector<UsdGenParamValue> *params,
             std::string const &prefix)
{
    if (!usdGen) {
        return;
    }
    bool const top = prefix == "usdGen";
    for (TfToken const &child : usdGen->GetNames()) {
        if (top && _HIsStock(child.GetString())) {
            continue;
        }
        HdDataSourceBaseHandle const h = usdGen->Get(child);
        if (HdContainerDataSourceHandle const sub =
                HdContainerDataSource::Cast(h)) {
            _HPullUsdGen(sub, t, node, params,
                         prefix + ":" + child.GetString());
            continue;
        }
        HdSampledDataSourceHandle const s = HdSampledDataSource::Cast(h);
        if (!s) {
            continue;
        }
        VtValue const v = s->GetValue(t);
        if (v.IsEmpty()) {
            continue;  // no authored/default opinion at all
        }
        std::string const leaf = _HStripSuffix(child.GetString());
        std::string const full = prefix + ":" + leaf;
        if (_HIsPathValue(v)) {
            if (node) {
                SdfPathVector *bucket = nullptr;
                if (leaf == "input") {
                    bucket = &node->inputs;
                } else if (leaf == "references" || leaf == "reference") {
                    bucket = &node->references;
                } else if (leaf == "guides" || leaf == "curves" ||
                           full == "usdGen:direction:source") {
                    // node.curves, not node.references: operator sources only
                    // resolve through curveRefs (guides rule).
                    bucket = &node->curves;
                } else {
                    continue;  // not a graph edge (base-name match only)
                }
                _HAppendPaths(v, bucket);
            }
            continue;
        }
        if (_isDedicated(TfToken(full))) {
            continue;
        }
        UsdGenParamValue p;
        p.name = TfToken(full.substr(7));  // strip "usdGen:" (02 §0.7)
        p.value = v;
        // P6 follow-up: HdSampledDataSource exposes no ValueMightBeTimeVarying
        // equivalent; fixtures are static so false holds. Animated-param
        // detection (e.g. via GetContributingSampleTimesForInterval) arrives
        // with the P6 animated scalp.
        p.animated = false;
        params->push_back(std::move(p));
    }
}

// Follows usdGen:input edges back from the terminal over Hydra path arrays;
// guides rels mark their curve sets as the Reference lane (I3). Mirrors
// _GraphWalker exactly (including: guides targets are roles, not operators).
struct _HydraWalker
{
    HdSceneIndexBase *input = nullptr;
    std::unordered_set<std::string> operatorPaths;
    std::unordered_set<std::string> visited;
    // guides rels mark their curve sets as the Reference lane (I3)
    std::unordered_map<std::string, UsdGenRole> curveRoles;

    HdContainerDataSourceHandle _UsdGen(SdfPath const &path)
    {
        HdSceneIndexPrim const prim = input->GetPrim(path);
        if (!prim.dataSource) {
            return nullptr;
        }
        return _HUsdGen(prim.dataSource);
    }

    void Walk(SdfPath const &path)
    {
        if (!visited.insert(path.GetString()).second) {
            return;
        }
        HdContainerDataSourceHandle const ug = _UsdGen(path);
        if (!ug) {
            return;
        }
        operatorPaths.insert(path.GetString());
        SdfPathVector targets;
        if (_HGetPathArray(ug, &targets, {"input"})) {
            for (SdfPath const &target : targets) {
                Walk(target);
            }
        }
        targets.clear();
        if (_HGetPathArray(ug, &targets, {"guides"})) {
            for (SdfPath const &target : targets) {
                curveRoles[target.GetString()] = UsdGenRole::Reference;
            }
        }
    }
};

// S26 namespace order over Hydra: depth-first pre-order from the description,
// children in scene-index order (sorted, matching USD namespace order).
void
_HCollectSubtree(HdSceneIndexBase &input, SdfPath const &root,
                 SdfPathVector *out)
{
    out->push_back(root);
    for (SdfPath const &c : input.GetChildPrimPaths(root)) {
        _HCollectSubtree(input, c, out);
    }
}

// Hydra mirror of the stage-side density fill: face means of the composed
// usdGen:paint:density face-varying primvar, clamped >= 0 (non-finite to
// 0). Missing/wrong-shaped data leaves the multiplier empty (== all 1.0).
void
_HCaptureDensityMultiplier(HdContainerDataSourceHandle const &primDs,
                           _HdTime t, UsdGenSurfaceDesc *out)
{
    HdPrimvarSchema const primvar =
        HdPrimvarsSchema::GetFromParent(primDs).GetPrimvar(
            TfToken("usdGen:paint:density"));
    HdSampledDataSourceHandle const values = primvar.GetPrimvarValue();
    if (!values) return;
    HdTokenDataSourceHandle const interp = primvar.GetInterpolation();
    if (!interp || interp->GetTypedValue(t) != TfToken("faceVarying")) return;
    VtValue const value = values->GetValue(t);
    if (!value.IsHolding<VtFloatArray>()) return;
    VtFloatArray const &flat = value.UncheckedGet<VtFloatArray>();
    size_t total = 0;
    for (int c : out->faceVertexCounts) total += size_t(c);
    if (flat.size() != total || out->faceVertexCounts.empty()) return;
    out->densityMultiplier.resize(out->faceVertexCounts.size());
    size_t k = 0;
    for (size_t f = 0; f < out->faceVertexCounts.size(); ++f) {
        int const n = out->faceVertexCounts[f];
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += double(flat[k++]);
        float mean = n > 0 ? float(sum / double(n)) : 1.0f;
        if (!std::isfinite(mean) || mean < 0.0f) mean = 0.0f;
        out->densityMultiplier[f] = mean;
    }
}

// Fills a surface desc from a Mesh prim in the flattened index. A GeomSubset
// target is detected by the caller (R15); xform/matrix is already the world
// matrix post-flattening (S4).
void
_HBuildSurface(HdSceneIndexBase &input, SdfPath const &path, double time,
               _HdTime t, UsdGenSurfaceDesc *out, std::vector<std::string>* errors)
{
    HdContainerDataSourceHandle primDs;
    TfToken primType;
    if (!_HPrim(input, path, &primDs, &primType)) {
        TF_CODING_ERROR("usdGen: usdGen:surface target %s does not exist.",
                        path.GetText());
        return;
    }
    if (primType != TfToken("mesh")) {
        TF_CODING_ERROR("usdGen: usdGen:surface target %s is a %s; M1 "
                        "surfaces must be Mesh or GeomSubset (R15).",
                        path.GetText(), primType.GetText());
        return;
    }

    out->path = path;
    HdContainerDataSourceHandle const mesh = _HChild(primDs, "mesh");
    HdContainerDataSourceHandle const topo = _HChild(mesh, "topology");
    _HGetTyped(mesh, t, &out->subdivisionScheme, {"subdivisionScheme"});
    _HGetTyped(topo, t, &out->orientation, {"orientation"});
    _HGetTyped(topo, t, &out->holeIndices, {"holeIndices"});
    auto const tags = _HChild(mesh, "subdivisionTags");
    _HGetTyped(tags, t, &out->interpolateBoundary, {"interpolateBoundary"});
    _HGetTyped(tags, t, &out->faceVaryingLinearInterpolation, {"faceVaryingLinearInterpolation"});
    _HGetTyped(tags, t, &out->triangleSubdivisionRule, {"triangleSubdivisionRule"});
    _HGetTyped(tags, t, &out->creaseMethod, {"creaseMethod"});
    _HGetTyped(tags, t, &out->creaseIndices, {"creaseIndices"});
    _HGetTyped(tags, t, &out->creaseLengths, {"creaseLengths"});
    _HGetTyped(tags, t, &out->creaseSharpnesses, {"creaseSharpnesses"});
    _HGetTyped(tags, t, &out->cornerIndices, {"cornerIndices"});
    _HGetTyped(tags, t, &out->cornerSharpnesses, {"cornerSharpnesses"});
    _HGetTyped(topo, t, &out->faceVertexCounts, {"faceVertexCounts"});
    _HGetTyped(topo, t, &out->faceVertexIndices, {"faceVertexIndices"});
    if (!_HGetTyped(primDs, t, &out->points, {"points"})) {
        // Flattened mesh points arrive as primvars/points (no top-level
        // points source on the final index).
        _HPrimvarTyped(primDs, "points", t, &out->points);
    }
    auto rest = _HChild(_HChild(primDs, "usdGen"), "rest");
    if (rest) {
        VtIntArray restCounts, restIndices;
        // The RestAPI leaves expose the live authored Default-time opinions,
        // but are time-invariant from Hydra's frame-sampling view. Sampling
        // their explicit rest time is valid; sampling mesh/current primvars
        // at a zero shutter offset would not be.
        bool valid = _HGetTyped(rest, 0.0, &out->restPoints, {"points"}) &&
            _HGetTyped(rest, 0.0, &restCounts, {"faceVertexCounts"}) &&
            _HGetTyped(rest, 0.0, &restIndices, {"faceVertexIndices"});
        // A deform moves points, never faces, so the rest and live topology
        // are usually the same Hydra shares: identical shares prove equality
        // in O(1) and skip the ~20MB deep compare, which runs only when the
        // shares actually differ (a retopology). Same predicate, same
        // short-circuit order, either way.
        auto countsMatch = [&] {
            return restCounts.IsIdentical(out->faceVertexCounts) ||
                restCounts == out->faceVertexCounts;
        };
        auto indicesMatch = [&] {
            return restIndices.IsIdentical(out->faceVertexIndices) ||
                restIndices == out->faceVertexIndices;
        };
        if (!valid || !countsMatch() || !indicesMatch()) {
            errors->push_back(path.GetString() + ": missing rest data or animated/rest topology mismatch");
            out->restFromCurrentPoints = true;
        }
    } else {
        // Legacy reference clients may omit RestAPI. Keep their old data
        // visible, but CUDA RBF must not bind from a posed first-pull sample.
        _HPrimvarTyped(primDs, "rest", t, &out->restPoints);
        if (out->restPoints.empty()) out->restPoints = out->points;
        out->restFromCurrentPoints = true;
    }
    _HPrimvarTyped(primDs, "st", t, &out->uv);
    _HPrimvarTyped(primDs, "velocities", t, &out->velocities);
    _HCaptureDensityMultiplier(primDs, t, out);

    // Only the RestAPI's live Default-time snapshot is authoritative for
    // F_rest. It updates on an authored Default edit, never per shutter.
    // Do not substitute a current mesh primvar sampled at t (or at 0): the
    // latter is a shutter offset, not UsdTimeCode::Default().
    if (rest) {
        HdSampledDataSourceHandle const normals = HdSampledDataSource::Cast(
            rest->Get(TfToken("normals")));
        if (normals) {
            VtValue const normalValue = normals->GetValue(0.0);
            if (!normalValue.IsEmpty()) {
                if (!normalValue.IsHolding<VtVec3fArray>()) {
                    out->restNormalDomain = UsdGenSurfaceNormalDomain::Invalid;
                    errors->push_back(path.GetString() + ": rest normals have invalid value type");
                } else {
                    out->restNormals = normalValue.UncheckedGet<VtVec3fArray>();
                    if (!out->restNormals.empty()) {
                        HdSampledDataSourceHandle const interpolation =
                            HdSampledDataSource::Cast(rest->Get(TfToken("normalsInterpolation")));
                        if (!interpolation) {
                            out->restNormalDomain = UsdGenSurfaceNormalDomain::Invalid;
                            errors->push_back(path.GetString() + ": rest normals are missing interpolation");
                        } else {
                            VtValue const interpolationValue = interpolation->GetValue(0.0);
                            if (!interpolationValue.IsHolding<TfToken>()) {
                                out->restNormalDomain = UsdGenSurfaceNormalDomain::Invalid;
                                errors->push_back(path.GetString() + ": rest normal interpolation has invalid value type");
                            } else {
                                out->restNormalDomain = _HNormalDomain(
                                    interpolationValue.UncheckedGet<TfToken>());
                                if (out->restNormalDomain == UsdGenSurfaceNormalDomain::Invalid)
                                    errors->push_back(path.GetString() + ": unsupported rest normal interpolation");
                            }
                        }
                    }
                }
            }
        }
    }

    out->samples.push_back(UsdGenSurfaceSample{time, out->points});
    out->worldMatrix = GfMatrix4d(1.0);
    HdMatrixDataSourceHandle const m =
        HdXformSchema::GetFromParent(primDs).GetMatrix();
    if (m) {
        out->worldMatrix = m->GetTypedValue(t);
    }
}

// The Mesh a surface target reads (R15): the target itself, a geomSubset's
// parent mesh, or empty for anything else.
SdfPath
_HSurfaceMesh(HdSceneIndexBase &input, SdfPath const &path)
{
    HdContainerDataSourceHandle primDs, parentDs;
    TfToken primType, parentType;
    if (!_HPrim(input, path, &primDs, &primType)) return SdfPath();
    if (primType == TfToken("mesh")) return path;
    if (HdGeomSubsetSchema::GetFromParent(primDs).IsDefined() &&
        _HPrim(input, path.GetParentPath(), &parentDs, &parentType) &&
        parentType == TfToken("mesh"))
        return path.GetParentPath();
    return SdfPath();
}

// Whether a geomSubset is a face set (02 §2.20 rule 1: the adapter maps
// elementType "face" to typeFaceSet). An absent type passes: hand-built
// sources may publish indices only. Otherwise `type` names the offender.
bool
_HFaceSubset(HdGeomSubsetSchema const &schema, _HdTime t, std::string *type)
{
    HdTokenDataSourceHandle const typeDs = schema.GetType();
    if (!typeDs) return true;
    TfToken const value = typeDs->GetTypedValue(t);
    if (value == HdGeomSubsetSchemaTokens->typeFaceSet) return true;
    *type = value.GetString();
    return false;
}

// Hydra mirror of the stage-side _BuildSubsetSurface: a geomSubset target's
// desc carries its parent mesh's full geometry plus the sorted, unique
// PARENT-mesh faces it (and any rule-4 union members) selects.
void
_HBuildSubsetSurface(HdSceneIndexBase &input, SdfPath const &path,
                     SdfPathVector const &unioned, double time, _HdTime t,
                     UsdGenSurfaceDesc *out, std::vector<std::string> *errors)
{
    SdfPath const meshPath = path.GetParentPath();
    HdContainerDataSourceHandle meshDs;
    TfToken meshType;
    if (!_HPrim(input, meshPath, &meshDs, &meshType) ||
        meshType != TfToken("mesh")) {
        errors->push_back(UsdGenSubsetParentError(path));
        return;
    }
    _HBuildSurface(input, meshPath, time, t, out, errors);
    out->path = path;
    out->isSubset = true;
    size_t const faceCount = out->faceVertexCounts.size();
    std::vector<int> faces;
    bool whole = false;
    SdfPathVector members{path};
    members.insert(members.end(), unioned.begin(), unioned.end());
    for (SdfPath const &member : members) {
        if (member == meshPath) {
            whole = true;
            continue;
        }
        HdContainerDataSourceHandle memberDs;
        if (!_HPrim(input, member, &memberDs, nullptr)) continue;
        HdGeomSubsetSchema const schema =
            HdGeomSubsetSchema::GetFromParent(memberDs);
        std::string type;
        if (!_HFaceSubset(schema, t, &type)) {
            errors->push_back(UsdGenSubsetElementTypeError(member, type));
            continue;
        }
        VtIntArray indices;
        if (HdIntArrayDataSourceHandle const idx = schema.GetIndices())
            indices = idx->GetTypedValue(t);
        UsdGenAppendSubsetFaces(indices, faceCount, member, meshPath, &faces,
                                errors);
    }
    out->subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), whole, faceCount);
}

template <class T>
bool
_HSurfaceCageField(HdContainerDataSourceHandle const &root, _HdTime t,
                   char const *leaf, T *out)
{
    if (_HGetTyped(root, t, out, {leaf})) return true;
    if (_HGetTyped(root, t, out, {"surfaceCage", leaf})) return true;
    std::string const flat = std::string("surfaceCage:") + leaf;
    return _HGetTyped(root, t, out, {flat.c_str()});
}

bool
_HBuildSurfaceCagePayload(HdContainerDataSourceHandle const &root,
                          _HdTime t,
                          std::shared_ptr<const UsdGenSurfaceCagePayload> *out)
{
    if (!root || !out) return false;
    HdContainerDataSourceHandle source = _HChild(root, "surfaceCage");
    if (!source) {
        HdContainerDataSourceHandle const rest =
            _HChild(root, "usdGenCurveRest");
        source = _HChild(rest, "surfaceCage");
    }
    if (!source) source = root;
    static char const *const names[] = {
        "ownerIds", "ownerDensities", "ownerSeeds", "ownerCvCounts",
        "ownerEdgeBias", "ownerLengthProfileOffsets", "ownerLengthProfile",
        "normalizedT", "triangles", "triangleOwnerIndices", "triangleRootCharts",
        "ownerChartCentroids", "ownerChartMeanRadii"};
    bool present = false;
    for (char const *leaf : names) {
        std::string const flat = std::string("surfaceCage:") + leaf;
        if (_HLocate(source, {leaf}) || _HLocate(source, {flat.c_str()})) {
            present = true;
            break;
        }
    }
    if (!present) return false;
    std::shared_ptr<UsdGenSurfaceCagePayload> cage =
        std::make_shared<UsdGenSurfaceCagePayload>();
    if (!_HSurfaceCageField(source, t, "ownerIds", &cage->ownerIds) ||
        !_HSurfaceCageField(source, t, "ownerDensities", &cage->ownerDensities) ||
        !_HSurfaceCageField(source, t, "ownerSeeds", &cage->ownerSeeds) ||
        !_HSurfaceCageField(source, t, "ownerCvCounts", &cage->ownerCvCounts) ||
        !_HSurfaceCageField(source, t, "ownerEdgeBias", &cage->ownerEdgeBias) ||
        !_HSurfaceCageField(source, t, "ownerLengthProfileOffsets",
                            &cage->ownerLengthProfileOffsets) ||
        !_HSurfaceCageField(source, t, "ownerLengthProfile",
                            &cage->ownerLengthProfile) ||
        !_HSurfaceCageField(source, t, "normalizedT", &cage->normalizedT) ||
        !_HSurfaceCageField(source, t, "triangles", &cage->triangles) ||
        !_HSurfaceCageField(source, t, "triangleOwnerIndices",
                            &cage->triangleOwnerIndices) ||
        !_HSurfaceCageField(source, t, "triangleRootCharts",
                            &cage->triangleRootCharts) ||
        !_HSurfaceCageField(source, t, "ownerChartCentroids",
                            &cage->ownerChartCentroids) ||
        !_HSurfaceCageField(source, t, "ownerChartMeanRadii",
                            &cage->ownerChartMeanRadii)) {
        return false;
    }
    *out = std::move(cage);
    return true;
}

void
_HBuildCurveSet(HdSceneIndexBase &input, SdfPath const &path,
                UsdGenRole role, _HdTime t, UsdGenCurveSetDesc *out)
{
    HdContainerDataSourceHandle primDs;
    if (!_HPrim(input, path, &primDs, nullptr)) {
        TF_CODING_ERROR("usdGen: curve set %s does not exist.",
                        path.GetText());
        return;
    }
    HdContainerDataSourceHandle const ug = _HUsdGen(primDs);
    out->path = path;
    out->role = role;
    if (auto matrix = HdXformSchema::GetFromParent(primDs).GetMatrix())
        out->worldMatrix = matrix->GetTypedValue(t);

    HdContainerDataSourceHandle const topo =
        _HChild(_HChild(primDs, "basisCurves"), "topology");
    out->type = out->basis = out->wrap = TfToken();
    _HGetToken(topo, t, &out->type, {"type"});
    _HGetToken(topo, t, &out->basis, {"basis"});
    _HGetToken(topo, t, &out->wrap, {"wrap"});
    _HGetToken(primDs, t, &out->widthsInterpolation,
               {"primvars", "widths", "interpolation"});
    _HGetTyped(topo, t, &out->curveVertexCounts, {"curveVertexCounts"});
    if (!_HGetTyped(primDs, t, &out->points, {"points"})) {
        _HPrimvarTyped(primDs, "points", t, &out->points);
    }
    // UsdGenCurveAPI publishes a live Default-time rest source.  Presence of
    // the source, rather than array emptiness, is significant: an explicitly
    // authored empty primvar must remain empty and must not be replaced by
    // current-frame points.  The provenance leaf distinguishes that source
    // from the compatibility fallback used when the API adapter is absent.
    bool curveRestRead = false;
    bool curveRestProvenanceRead = false;
    HdContainerDataSourceHandle const curveRest =
        _HChild(primDs, "usdGenCurveRest");
    if (curveRest) {
        HdSampledDataSourceHandle const restPoints =
            HdSampledDataSource::Cast(curveRest->Get(TfToken("points")));
        if (restPoints) {
            VtValue const value = restPoints->GetValue(t);
            if (value.IsHolding<VtVec3fArray>()) {
                out->rest = value.UncheckedGet<VtVec3fArray>();
                curveRestRead = true;
            }
        }
        HdSampledDataSourceHandle const provenance =
            HdSampledDataSource::Cast(
                curveRest->Get(TfToken("hasAuthoredRest")));
        if (provenance) {
            VtValue const value = provenance->GetValue(t);
            if (value.IsHolding<bool>()) {
                // The bool records provenance, not whether the valid
                // Default-time fallback is usable.  Both authored rest and
                // Default-time points are proper C3 rest sources.
                curveRestProvenanceRead = true;
            }
        }
    }
    if (curveRestRead) {
        // A valid adapter read is never a current-frame fallback, regardless
        // of hasAuthoredRest's value.  Missing/malformed provenance remains
        // marked conservatively for CUDA admission.
        out->restFromCurrentPoints = !curveRestProvenanceRead;
    } else {
        // Missing adapter/current fallback is retained for compatibility but
        // explicitly marked so CUDA admission cannot treat it as bound rest.
        _HPrimvarTyped(primDs, "rest", t, &out->rest);
        if (out->rest.empty()) {
            out->rest = out->points;
        }
        out->restFromCurrentPoints = true;
    }
    if (!_HGetTyped(primDs, t, &out->widths, {"widths"})) {
        _HPrimvarTyped(primDs, "widths", t, &out->widths);
    }
    _HBuildSurfaceCagePayload(ug, t, &out->surfaceCage);
    _HPrimvarTyped(primDs, "skinprim", t, &out->skinPrim);
    _HPrimvarTyped(primDs, "usdGen:curveId", t, &out->curveId);
    _HPrimvarTyped(primDs, "skinprimuv", t, &out->skinPrimUv);
    _HPrimvarTyped(primDs, "usdGen:rootFrame", t, &out->rootFrame);

    _HForwardSourceColor(primDs, t, out);
    _HForwardOwnershipPlane(primDs, TfToken("tubeId"), t, out);
    _HForwardOwnershipPlane(primDs, TfToken("regionId"), t, out);
    _HForwardOwnershipPlane(primDs, TfToken("hierarchyLevel"), t, out);

    TfToken curveRole;
    _HPrimvarTyped(primDs, "usdGen:role", t, &curveRole);
    if (curveRole.IsEmpty()) {
        _HGetToken(ug, t, &curveRole, {"role"});
    }
    if (!curveRole.IsEmpty()) {
        out->curveRole = curveRole;
    }

    // frozenEpoch: constant string primvar "usdgen1:sha1:..." (S42).
    VtValue epoch;
    if (_HPrimvarValue(primDs, "usdGen:frozenEpoch", t, &epoch)) {
        if (epoch.IsHolding<VtStringArray>()) {
            VtStringArray const &arr = epoch.UncheckedGet<VtStringArray>();
            if (!arr.empty()) {
                out->frozenEpoch = arr[0].c_str();
            }
        } else if (epoch.IsHolding<std::string>()) {
            out->frozenEpoch = epoch.UncheckedGet<std::string>();
        } else if (epoch.IsHolding<TfToken>()) {
            out->frozenEpoch = epoch.UncheckedGet<TfToken>().GetString();
        }
    }

}

// Resolves one input:<name> target into the gprims and map prims it names. A
// target that is neither contributes its descendants, in child order.
void
_HCollectInputTarget(HdSceneIndexBase &input, SdfPath const &path, int depth,
                     SdfPathVector *geometries, SdfPathVector *maps)
{
    HdContainerDataSourceHandle primDs;
    TfToken primType;
    if (!_HPrim(input, path, &primDs, &primType)) return;
    if (UsdGenIsMapTypeName(_HUsdTypeName(primDs))) {
        maps->push_back(path);
        return;
    }
    if (primType == TfToken("mesh") || primType == TfToken("basisCurves") ||
        primType == TfToken("points")) {
        geometries->push_back(path);
        return;
    }
    // A mesh's geomSubset samples the mesh restricted to its faces (R15).
    if (HdGeomSubsetSchema::GetFromParent(primDs).IsDefined()) {
        if (!_HSurfaceMesh(input, path).IsEmpty()) geometries->push_back(path);
        return;
    }
    if (depth > 64) return;
    for (SdfPath const &child : input.GetChildPrimPaths(path))
        _HCollectInputTarget(input, child, depth + 1, geometries, maps);
}

// A gprim an expression samples. Rest follows the RestAPI/CurveAPI adapters
// when applied, then an authored primvars:rest; otherwise it stays empty and
// the sampler reads the current points as rest. A face geomSubset samples its
// parent mesh restricted to the sorted, unique faces it names (R15).
void
_HBuildGeometry(HdSceneIndexBase &input, SdfPath const &path, _HdTime t,
                usdGen::UsdGenGeometryDesc *out, std::vector<std::string> *errors)
{
    HdContainerDataSourceHandle primDs;
    TfToken primType;
    out->path = path;
    if (!_HPrim(input, path, &primDs, &primType)) return;
    HdGeomSubsetSchema const subset = HdGeomSubsetSchema::GetFromParent(primDs);
    if (subset.IsDefined()) {
        SdfPath const meshPath = path.GetParentPath();
        _HBuildGeometry(input, meshPath, t, out, errors);
        out->path = path;
        out->isSubset = true;
        std::string type;
        if (!_HFaceSubset(subset, t, &type)) {
            errors->push_back(UsdGenSubsetElementTypeError(path, type));
        } else {
            VtIntArray indices;
            if (HdIntArrayDataSourceHandle const idx = subset.GetIndices())
                indices = idx->GetTypedValue(t);
            std::vector<int> faces;
            UsdGenAppendSubsetFaces(indices, out->counts.size(), path, meshPath,
                                    &faces, errors);
            out->subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), false, 0);
        }
        out->generation = UsdGenGeometryContentHash(*out);
        return;
    }
    if (auto matrix = HdXformSchema::GetFromParent(primDs).GetMatrix())
        out->worldMatrix = matrix->GetTypedValue(t);
    if (!_HGetTyped(primDs, t, &out->points, {"points"}))
        _HPrimvarTyped(primDs, "points", t, &out->points);
    if (primType == TfToken("mesh")) {
        out->kind = usdGen::UsdGenGeometryKind::Mesh;
        HdContainerDataSourceHandle const topo = _HChild(_HChild(primDs, "mesh"), "topology");
        _HGetTyped(topo, t, &out->counts, {"faceVertexCounts"});
        _HGetTyped(topo, t, &out->indices, {"faceVertexIndices"});
        if (HdContainerDataSourceHandle rest = _HChild(_HChild(primDs, "usdGen"), "rest"))
            _HGetTyped(rest, 0.0, &out->rest, {"points"});
        else
            _HPrimvarTyped(primDs, "rest", t, &out->rest);
    } else if (primType == TfToken("basisCurves")) {
        out->kind = usdGen::UsdGenGeometryKind::Curves;
        HdContainerDataSourceHandle const topo =
            _HChild(_HChild(primDs, "basisCurves"), "topology");
        _HGetTyped(topo, t, &out->counts, {"curveVertexCounts"});
        if (HdContainerDataSourceHandle rest = _HChild(primDs, "usdGenCurveRest"))
            _HGetTyped(rest, t, &out->rest, {"points"});
        else
            _HPrimvarTyped(primDs, "rest", t, &out->rest);
        _HPrimvarTyped(primDs, "usdGen:curveId", t, &out->ids);
    } else {
        out->kind = usdGen::UsdGenGeometryKind::Points;
        _HPrimvarTyped(primDs, "rest", t, &out->rest);
        _HPrimvarTyped(primDs, "normals", t, &out->normals);
    }
    out->generation = UsdGenGeometryContentHash(*out);
}

// Read precisely the operator-owned portion.  Inputs and inherited surfaces
// are intentionally absent from this value: both are assembled from the
// current composed hierarchy below, even if this raw read is reused.
UsdGenGraphDescCaptureCache::NodeCapture
_HReadNode(HdSceneIndexBase &input, SdfPath const &p, _HdTime t)
{
    UsdGenGraphDescCaptureCache::NodeCapture captured;
    captured.path = p;
    HdContainerDataSourceHandle primDs;
    TfToken primType;
    if (!_HPrim(input, p, &primDs, &primType)) return captured;
    captured.exists = true;
    HdContainerDataSourceHandle const ug = _HUsdGen(primDs);
    VtStringArray operatorErrors;
    _HGetTyped(primDs, t, &operatorErrors, {"__usdGenValidationErrors"});
    captured.validationErrors.insert(captured.validationErrors.end(),
        operatorErrors.begin(), operatorErrors.end());

    UsdGenNodeDesc &node = captured.node;
    node.path = p;
    TfToken type;
    _HGetToken(ug, t, &type, {"type"});
    if (type.IsEmpty()) type = primType.IsEmpty() ? _HUsdTypeName(primDs) : primType;
    node.type = type;
    _HGetToken(ug, t, &node.mode, {"mode"});
    bool enabled = true;
    _HGetDedicated(ug, t, &enabled, &captured.validationErrors,
        node.path, "enabled", {"enabled"});
    node.enabled = enabled;
    _HGetDedicated(ug, t, &node.seed, &captured.validationErrors,
        node.path, "seed", {"seed"});
    _HPullUsdGen(ug, t, &node, &node.params, "usdGen");
    // Legacy authored inputs are observed for S14 but never become graph
    // topology.  Keep the cached payload pre-derived.
    node.inputs.clear();
    if (HdContainerDataSourceHandle bindings = _HChild(ug, "expressionBindings")) {
        for (TfToken const &bn : bindings->GetNames()) {
            HdContainerDataSourceHandle const b = HdContainerDataSource::Cast(bindings->Get(bn));
            usdGen::UsdGenExpressionBinding binding;
            _HGetTyped(b, t, &binding.expression, {"expression"});
            _HGetToken(b, t, &binding.output, {"output"});
            _HGetToken(b, t, &binding.destination, {"destination"});
            _HGetToken(b, t, &binding.nativeType, {"nativeType"});
            TfToken domain; _HGetToken(b, t, &domain, {"domain"});
            binding.domain = domain == TfToken("point") ? usdGen::expr::Domain::Point :
                (domain == TfToken("primitive") ? usdGen::expr::Domain::Primitive : usdGen::expr::Domain::Groom);
            _HGetTyped(b, t, &binding.literal, {"literal"});
            uint32_t arrayElementCount = 0;
            bool const arrayCountKnown = _BindingArrayCount(
                binding.nativeType, binding.literal, &arrayElementCount);
            binding.destinationShape = usdGen::expr::ValueShapeFromNativeType(
                binding.nativeType, arrayElementCount, arrayCountKnown);
            node.expressionBindings.push_back(std::move(binding));
        }
    }
    _HGetPathArray(ug, &captured.guides, {"guides"});
    // usdGen:part:curves nests as usdGen/part/curves; its targets join the
    // Reference lane (I3) exactly like guides.
    SdfPathVector partCurves;
    if (_HGetPathArray(ug, &partCurves, {"part", "curves"}))
        captured.guides.insert(captured.guides.end(),
                               partCurves.begin(), partCurves.end());
    // usdGen:direction:source nests as usdGen/direction/source; same lane.
    SdfPathVector directionSource;
    if (_HGetPathArray(ug, &directionSource, {"direction", "source"}))
        captured.guides.insert(captured.guides.end(),
                               directionSource.begin(), directionSource.end());
    // usdGen:frozen:curves nests as usdGen/frozen/curves; its targets join
    // the Reference lane (I3) exactly like part:curves, so Freeze can
    // snapshot an explicit curve set instead of the chain input.
    SdfPathVector frozenCurves;
    if (_HGetPathArray(ug, &frozenCurves, {"frozen", "curves"}))
        captured.guides.insert(captured.guides.end(),
                               frozenCurves.begin(), frozenCurves.end());
    // usdGen:colliders is flat (usdGen/colliders), like usdGen:guides. Only
    // UsdGenCollide declares it; the targets append to that node's surfaces
    // after surface inheritance below.
    if (node.type == TfToken("UsdGenCollide"))
        _HGetPathArray(ug, &captured.colliders, {"colliders"});
    if (node.type == TfToken("UsdGenCurveSource") || node.type == TfToken("UsdGenDeform") ||
        node.type == TfToken("UsdGenGuideInterpolate")) {
        SdfPathVector regionMap;
        if (_HGetPathArray(ug, &regionMap, {"regionMap"})) {
            // The schema relationship exists even when unauthored, so an
            // empty target list is "not set". surfaceCage admission still
            // requires a bound map later. More than one target is invalid.
            if (regionMap.size() > 1) {
                captured.validationErrors.push_back(
                    node.path.GetString() +
                    ": usdGen:regionMap requires exactly one target");
            } else if (regionMap.size() == 1) {
                node.maps.push_back(regionMap.front());
                node.mapBindings.push_back(
                    {regionMap.front(), TfToken("usdGen:regionMap")});
            }
        }
    }
    return captured;
}

bool
_HNodeDirty(SdfPath const &node, SdfPathVector const &dirtyPaths)
{
    for (SdfPath const &dirty : dirtyPaths)
        if (node.HasPrefix(dirty)) return true;
    return false;
}

// Hydra mirror of the stage-side _CapturePaintMap (usdGenGraphDescBuilderStage.cpp):
// snapshots a UsdGenPaintMap's surface primvar into the map desc (scalar,
// folded through usdGen:map:channel). v1 carries faceVarying float data
// only; anything else fails closed with a validation error and leaves the
// payload empty, which the sampler treats as unusable. A missing primvar
// reads usdGen:map:default everywhere instead of failing.
void
_HCapturePaintMap(HdSceneIndexBase &input,
                 HdContainerDataSourceHandle const &primDs, _HdTime t,
                 usdGen::UsdGenMapDesc *map,
                 std::vector<std::string> *errors,
                 SdfPathVector *readPrims)
{
    auto fail = [&](std::string const &what) {
        if (errors) {
            errors->push_back(map->path.GetString() + ": " + what);
        }
    };
    HdContainerDataSourceHandle const ug = _HUsdGen(primDs);
    SdfPathVector targets;
    _HGetPathArray(ug, &targets, {"paint", "surface"});
    if (targets.size() != 1) {
        fail("usdGen:paint:surface requires exactly one target");
        return;
    }
    // The snapshot reads through to the paint surface, outside the map
    // prim: report every prim touched so the capture cache can gate reuse
    // on them and keep them in the groom's dependencies.
    if (readPrims) readPrims->push_back(targets.front());
    // A face geomSubset target (R15) paints its parent mesh: the primvar is
    // read there and the corners outside the subset read usdGen:map:default.
    SdfPath const meshPath = _HSurfaceMesh(input, targets.front());
    HdContainerDataSourceHandle surfaceDs;
    if (!meshPath.IsEmpty() && readPrims) readPrims->push_back(meshPath);
    if (meshPath.IsEmpty() || !_HPrim(input, meshPath, &surfaceDs, nullptr)) {
        fail("usdGen:paint:surface must target a UsdGeomMesh or a face "
             "GeomSubset of one");
        return;
    }
    bool const subset = meshPath != targets.front();
    VtIntArray subsetFaces;
    if (subset) {
        HdContainerDataSourceHandle subsetDs;
        _HPrim(input, targets.front(), &subsetDs, nullptr);
        HdGeomSubsetSchema const schema = HdGeomSubsetSchema::GetFromParent(subsetDs);
        std::string type;
        if (!_HFaceSubset(schema, t, &type)) {
            fail(UsdGenSubsetElementTypeError(targets.front(), type));
            return;
        }
        VtIntArray faceCounts, indices;
        _HGetTyped(_HChild(_HChild(surfaceDs, "mesh"), "topology"), t,
                   &faceCounts, {"faceVertexCounts"});
        if (HdIntArrayDataSourceHandle const idx = schema.GetIndices())
            indices = idx->GetTypedValue(t);
        std::vector<int> faces;
        UsdGenAppendSubsetFaces(indices, faceCounts.size(), targets.front(),
                                meshPath, &faces, errors);
        subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), false, 0);
    }
    TfToken primvarName;
    _HGetToken(ug, t, &primvarName, {"paint", "primvar"});
    if (primvarName.IsEmpty()) {
        fail("usdGen:paint:primvar is empty");
        return;
    }
    HdPrimvarSchema const primvar =
        HdPrimvarsSchema::GetFromParent(surfaceDs).GetPrimvar(primvarName);
    HdSampledDataSourceHandle const values = primvar.GetPrimvarValue();
    VtValue value;
    if (values) {
        value = values->GetValue(t);
    }
    if (!values || value.IsEmpty()) {
        // No baked primvar yet: the map reads its authored default
        // everywhere (the BaseGridFromStage semantic), so a wired but
        // unpainted map cooks instead of rejecting the commit.
        float defaultValue = 0.0f;
        if (!_HGetTyped(ug, t, &defaultValue, {"map", "default"})) {
            fail("surface " + meshPath.GetString() + " has no primvar " +
                 primvarName.GetString());
            return;
        }
        TfToken promised;
        _HGetToken(ug, t, &promised, {"paint", "interpolation"});
        if (!promised.IsEmpty() && promised != TfToken("faceVarying")) {
            fail("primvar " + primvarName.GetString() + " is " + promised.GetString() +
                 ", v1 paints faceVarying only");
            return;
        }
        VtIntArray defaultCounts;
        HdContainerDataSourceHandle const defaultTopo =
            _HChild(_HChild(surfaceDs, "mesh"), "topology");
        _HGetTyped(defaultTopo, t, &defaultCounts, {"faceVertexCounts"});
        size_t defaultFaceVarying = 0;
        for (int c : defaultCounts) defaultFaceVarying += size_t(c);
        map->paintSurface = meshPath;
        map->paintPrimvar = primvarName;
        map->paintInterpolation = TfToken("faceVarying");
        map->paintValues.assign(defaultFaceVarying, defaultValue);
        return;
    }
    TfToken interp;
    if (HdTokenDataSourceHandle const i = primvar.GetInterpolation()) {
        interp = i->GetTypedValue(t);
    }
    if (interp != TfToken("faceVarying")) {
        fail("primvar " + primvarName.GetString() + " is " + interp.GetString() +
             ", v1 paints faceVarying only");
        return;
    }
    TfToken channel;
    _HGetToken(ug, t, &channel, {"map", "channel"});
    if (channel.IsEmpty()) {
        channel = TfToken("r");
    }
    VtFloatArray folded;
    if (value.IsHolding<VtFloatArray>()) {
        if (channel != TfToken("r") && channel != TfToken("luminance")) {
            fail("scalar primvar " + primvarName.GetString() +
                 " cannot supply channel " + channel.GetString());
            return;
        }
        folded = value.UncheckedGet<VtFloatArray>();
    } else if (value.IsHolding<VtVec3fArray>()) {
        VtVec3fArray const &v = value.UncheckedGet<VtVec3fArray>();
        size_t pick = 0;
        bool luminance = false;
        if (channel == TfToken("r")) pick = 0;
        else if (channel == TfToken("g")) pick = 1;
        else if (channel == TfToken("b")) pick = 2;
        else if (channel == TfToken("luminance")) luminance = true;
        else {
            fail("primvar " + primvarName.GetString() +
                 " cannot supply channel " + channel.GetString());
            return;
        }
        folded.resize(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            folded[i] = luminance
                ? 0.2126f * v[i][0] + 0.7152f * v[i][1] + 0.0722f * v[i][2]
                : v[i][pick];
        }
    } else {
        fail("primvar " + primvarName.GetString() + " must be float or "
             "float3-valued");
        return;
    }
    VtIntArray counts;
    HdContainerDataSourceHandle const topo =
        _HChild(_HChild(surfaceDs, "mesh"), "topology");
    _HGetTyped(topo, t, &counts, {"faceVertexCounts"});
    size_t faceVarying = 0;
    for (int c : counts) faceVarying += size_t(c);
    if (folded.size() != faceVarying) {
        fail("primvar " + primvarName.GetString() + " has " +
             std::to_string(folded.size()) + " values for " +
             std::to_string(faceVarying) + " face vertices");
        return;
    }
    if (subset) {
        float fallback = 0.0f;
        _HGetTyped(ug, t, &fallback, {"map", "default"});
        UsdGenMaskPaintValues(counts, subsetFaces, fallback, &folded);
    }
    map->paintSurface = meshPath;
    map->paintPrimvar = primvarName;
    map->paintInterpolation = interp;
    map->paintValues = folded;
}

}  // namespace


UsdGenGraphDescCapture
CaptureGraphDescFromHydra(
    HdSceneIndexBase &input,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options)
{
    TRACE_FUNCTION();
    UsdGenGraphDescCapture result;
    UsdGenGraphDesc &desc = result.desc;
    desc.description = descriptionPath;
    desc.time = options.time;
    double const time = options.time;
    _HdTime const t = _HTime(time);

    HdContainerDataSourceHandle descDs;
    if (!_HPrim(input, descriptionPath, &descDs, nullptr)) {
        TF_CODING_ERROR("usdGen: description prim %s does not exist.",
                        descriptionPath.GetText());
        return result;
    }
    HdContainerDataSourceHandle const descUg = _HUsdGen(descDs);
    VtStringArray descriptionErrors;
    _HGetTyped(descDs, t, &descriptionErrors, {"__usdGenValidationErrors"});
    desc.validationErrors.insert(desc.validationErrors.end(), descriptionErrors.begin(), descriptionErrors.end());

    // The description adapter exposes the composed stage rate as a live
    // metadata source.  Inspect its raw VtValue: malformed, non-finite, and
    // non-positive rates must be diagnosed instead of reaching $time math.
    if (HdContainerDataSourceHandle const runtime =
            _HChild(descUg, "usdGenRuntime")) {
        if (HdSampledDataSourceHandle const rateSource =
                HdSampledDataSource::Cast(
                    runtime->Get(TfToken("timeCodesPerSecond")))) {
            VtValue const value = rateSource->GetValue(t);
            if (!value.IsHolding<double>()) {
                desc.validationErrors.push_back(
                    desc.description.GetString() +
                    ": timeCodesPerSecond has wrong authored type");
            } else {
                double const rate = value.UncheckedGet<double>();
                if (!std::isfinite(rate) || rate <= 0.0) {
                    desc.validationErrors.push_back(
                        desc.description.GetString() +
                        ": timeCodesPerSecond must be finite and positive");
                } else {
                    desc.timeCodesPerSecond = rate;
                }
            }
        }
    }

    // The Description adapter carries composed child order explicitly: a
    // scene-index child enumeration is not an authoring-order API.  This is
    // reverse-sibling post-order, and it is also the execution order.
    SdfPathVector operatorOrder;
    _HGetPathArray(descUg, &operatorOrder, {"operatorOrder"});
    if (operatorOrder.empty()) {
        TF_CODING_ERROR("usdGen: %s has no usdGen:operatorOrder.",
                        descriptionPath.GetText());
        return result;
    }
    desc.terminal = operatorOrder.back();
    std::unordered_map<std::string, UsdGenRole> curveRoles;

    if (HdContainerDataSourceHandle expressions = _HChild(descUg, "expressions")) {
        for (TfToken const &name : expressions->GetNames()) {
            HdContainerDataSourceHandle const e = HdContainerDataSource::Cast(expressions->Get(name));
            usdGen::UsdGenExpressionDesc expression;
            _HGetTyped(e, t, &expression.path, {"path"});
            _HGetTyped(e, t, &expression.source, {"source"});
            if (HdContainerDataSourceHandle outputs = _HChild(e, "outputs")) for (TfToken const &on : outputs->GetNames()) {
                HdContainerDataSourceHandle const o = HdContainerDataSource::Cast(outputs->Get(on));
                usdGen::UsdGenExpressionOutputDesc output;
                _HGetToken(o, t, &output.name, {"name"});
                _HGetToken(o, t, &output.nativeType, {"nativeType"});
                uint32_t arrayElementCount = 0;
                bool arrayCountKnown = false;
                _HGetTyped(o, t, &arrayElementCount, {"arrayElementCount"});
                _HGetTyped(o, t, &arrayCountKnown, {"arrayCountKnown"});
                output.shape = usdGen::expr::ValueShapeFromNativeType(
                    output.nativeType, arrayElementCount, arrayCountKnown);
                expression.outputs.push_back(std::move(output));
            }
            if (HdContainerDataSourceHandle inputs = _HChild(e, "inputs")) for (TfToken const &in : inputs->GetNames()) {
                HdContainerDataSourceHandle const i = HdContainerDataSource::Cast(inputs->Get(in));
                usdGen::UsdGenExpressionInputDesc slot;
                _HGetToken(i, t, &slot.name, {"name"});
                _HGetPathArray(i, &slot.targets, {"targets"});
                expression.inputs.push_back(std::move(slot));
            }
            desc.expressions.push_back(std::move(expression));
        }
    }

    // ---- nodes, supplied composed reverse-sibling post-order -------------
    // Collider targets ride along keyed by node path; they append to the
    // Collide nodes' surfaces after surface inheritance below.
    std::map<std::string, SdfPathVector> colliderTargets;
    auto const previousCache =
        (options.reuseNodes || options.reuseGeometry) ? options.previousCache
                                                      : nullptr;
    // Geometry values flow only from Hydra data, so a prim no dirty touched
    // still reads what the cached capture read -- across frames, unlike
    // operator params, which can read the frame procedurally ($frame/$time)
    // and stay gated on it below.
    bool const reuseGeometry = options.reuseGeometry && previousCache &&
        previousCache->description == descriptionPath &&
        previousCache->time == options.time;
    auto cache = std::make_shared<UsdGenGraphDescCaptureCache>();
    {
        TRACE_SCOPE("usdGen capture operators");
        auto const previous = options.reuseNodes ? previousCache : nullptr;
        bool const reusable = previous && previous->description == descriptionPath &&
            previous->time == options.time && previous->operatorOrder == operatorOrder;
        cache->description = descriptionPath;
        cache->time = options.time;
        cache->operatorOrder = operatorOrder;
        cache->nodes.reserve(operatorOrder.size());
        for (size_t index = 0; index != operatorOrder.size(); ++index) {
            SdfPath const &p = operatorOrder[index];
            UsdGenGraphDescCaptureCache::NodeCapture captured;
            if (reusable && index < previous->nodes.size() &&
                previous->nodes[index].path == p &&
                !_HNodeDirty(p, options.dirtyPrimPaths)) {
                captured = previous->nodes[index];
            } else {
                captured = _HReadNode(input, p, t);
            }
            cache->nodes.push_back(captured);
            desc.validationErrors.insert(desc.validationErrors.end(),
                captured.validationErrors.begin(), captured.validationErrors.end());
            if (!captured.exists) continue;
            UsdGenNodeDesc node = captured.node;
            for (SdfPath const &guide : captured.guides)
                curveRoles[guide.GetString()] = UsdGenRole::Reference;
            // Hierarchy is the topology contract.  Do not permit a legacy
            // authored input relationship to alter it.
            node.inputs.clear();
            if (!desc.nodes.empty()) {
                node.inputs.push_back(desc.nodes.back().path);
            }
            if (!captured.colliders.empty())
                colliderTargets[captured.path.GetString()] = captured.colliders;
            desc.nodes.push_back(std::move(node));
        }
    }

    // ---- surface inheritance (02 §2) -------------------------------------
    // Targets on the bound target's mesh union into it (02 §2.20 rule 4).
    std::map<SdfPath, SdfPathVector> surfaceUnions;
    {
        SdfPathVector descSurfaces;
        _HGetPathArray(descUg, &descSurfaces, {"surface"});
        SdfPathVector unioned = UsdGenUnionSurfaceTargets(&descSurfaces,
            [&](SdfPath const &p) { return _HSurfaceMesh(input, p); });
        if (!unioned.empty())
            surfaceUnions.emplace(descSurfaces.front(), std::move(unioned));
        if (!descSurfaces.empty()) {
            for (UsdGenNodeDesc &node : desc.nodes) {
                node.surfaces = descSurfaces;
            }
        }
    }

    // UsdGenCollide (02 §2.8): usdGen:colliders targets ride the shared
    // surface path. They append AFTER the inherited bound surface, so
    // surfaces.front() — the root surface every consumer resolves — is
    // unchanged, and the pool loop below builds their descs like any other.
    // A collider equal to the bound surface is already present, not doubled.
    if (!colliderTargets.empty()) {
        for (UsdGenNodeDesc &node : desc.nodes) {
            auto const it = colliderTargets.find(node.path.GetString());
            if (it == colliderTargets.end()) continue;
            for (SdfPath const &c : it->second) {
                if (std::find(node.surfaces.begin(), node.surfaces.end(), c) ==
                    node.surfaces.end())
                    node.surfaces.push_back(c);
            }
        }
    }

    // ---- shared pools: surfaces / curve sets / maps ----------------------
    std::map<std::string, size_t> surfaceIndex;
    std::map<std::string, size_t> curveIndex;
    std::map<std::string, size_t> mapIndex;

    auto surfaceFor = [&](SdfPath const &p) {
        if (surfaceIndex.count(p.GetString())) {
            return;
        }
        size_t const index = desc.surfaces.size();
        if (reuseGeometry) {
            auto const prev = previousCache->surfaces.find(p.GetString());
            if (prev != previousCache->surfaces.end() && prev->second.exists &&
                !_HNodeDirty(p, options.dirtyPrimPaths)) {
                // A subset reads its parent mesh and its union members, so
                // all of those must be clean too, with the same member set.
                bool stale = false;
                if (prev->second.surface.isSubset) {
                    if (_HNodeDirty(p.GetParentPath(), options.dirtyPrimPaths))
                        stale = true;
                    auto const u = surfaceUnions.find(p);
                    SdfPathVector const empty;
                    SdfPathVector const &unioned =
                        u == surfaceUnions.end() ? empty : u->second;
                    if (unioned != prev->second.unionMembers) stale = true;
                    for (SdfPath const &m : unioned)
                        if (_HNodeDirty(m, options.dirtyPrimPaths)) stale = true;
                }
                if (!stale) {
                    UsdGenSurfaceDesc reused = prev->second.surface;
                    reused.id = usdGen::UsdGenSurfaceId(index);
                    surfaceIndex.emplace(p.GetString(), index);
                    desc.surfaces.push_back(reused);
                    desc.validationErrors.insert(
                        desc.validationErrors.end(),
                        prev->second.validationErrors.begin(),
                        prev->second.validationErrors.end());
                    cache->surfaces.emplace(p.GetString(), prev->second);
                    result.reusedInputs.push_back(p);
                    if (prev->second.surface.isSubset) {
                        result.reusedInputs.push_back(p.GetParentPath());
                        for (SdfPath const &m : prev->second.unionMembers)
                            result.reusedInputs.push_back(m);
                    }
                    return;
                }
            }
        }
        UsdGenSurfaceDesc surface;
        HdContainerDataSourceHandle primDs;
        std::vector<std::string> surfaceErrors;
        SdfPathVector unionMembers;
        if (_HPrim(input, p, &primDs, nullptr) &&
            HdGeomSubsetSchema::GetFromParent(primDs).IsDefined()) {
            // R15: the subset's desc carries its parent mesh's geometry and
            // the PARENT-mesh face indices it selects.
            auto const u = surfaceUnions.find(p);
            if (u != surfaceUnions.end()) unionMembers = u->second;
            _HBuildSubsetSurface(input, p, unionMembers,
                                 time, t, &surface, &surfaceErrors);
        } else {
            _HBuildSurface(input, p, time, t, &surface, &surfaceErrors);
        }
        // Invariant sub-hash memo: a deform moves points, never rest or
        // topology, so when the previous capture's rest/topology shares are
        // identical to this build's, its sub-hash is this build's (same CoW
        // rule as the occluder digest: the cache holds the shares, so equal
        // (size, cdata) is equal content). Anything else recomputes fresh
        // in Finalize; the values are identical either way.
        if (reuseGeometry) {
            auto const prev = previousCache->surfaces.find(p.GetString());
            if (prev != previousCache->surfaces.end() && prev->second.exists) {
                usdGen::UsdGenSurfaceDesc const &ps = prev->second.surface;
                if (ps.restNormalDomain == surface.restNormalDomain &&
                    ps.faceVertexCounts.IsIdentical(surface.faceVertexCounts) &&
                    ps.faceVertexIndices.IsIdentical(surface.faceVertexIndices) &&
                    ps.restPoints.IsIdentical(surface.restPoints) &&
                    ps.restNormals.IsIdentical(surface.restNormals))
                    surface.surfaceInvariantGeneration =
                        ps.surfaceInvariantGeneration;
            }
        }
        // Both builders set out->path past their existence/type checks, so
        // an unset path marks a miss: a missing prim rebuilds (and re-emits
        // its diagnostic) every capture, exactly as before. The published
        // desc still names the target, as the historical slot preset did.
        bool const exists = surface.path == p;
        if (!exists) surface.path = p;
        surface.id = usdGen::UsdGenSurfaceId(index);
        surfaceIndex.emplace(p.GetString(), index);
        desc.surfaces.push_back(surface);
        desc.validationErrors.insert(desc.validationErrors.end(),
            surfaceErrors.begin(), surfaceErrors.end());
        UsdGenGraphDescCaptureCache::SurfaceCapture captured;
        captured.path = p;
        captured.surface = surface;
        captured.validationErrors = std::move(surfaceErrors);
        captured.unionMembers = std::move(unionMembers);
        captured.exists = exists;
        cache->surfaces.emplace(p.GetString(), std::move(captured));
    };

    auto curveFor = [&](SdfPath const &p, UsdGenRole role) {
        auto it = curveIndex.find(p.GetString());
        if (it == curveIndex.end()) {
            if (reuseGeometry) {
                auto const prev = previousCache->curves.find(p.GetString());
                if (prev != previousCache->curves.end() &&
                    prev->second.exists &&
                    prev->second.curves.role == role &&
                    !_HNodeDirty(p, options.dirtyPrimPaths)) {
                    it = curveIndex.emplace(p.GetString(),
                                            desc.curveSets.size()).first;
                    desc.curveSets.push_back(prev->second.curves);
                    cache->curves.emplace(p.GetString(), prev->second);
                    result.reusedInputs.push_back(p);
                    return;
                }
            }
            UsdGenCurveSetDesc cs;
            _HBuildCurveSet(input, p, role, t, &cs);
            bool const exists = cs.path == p;
            it = curveIndex.emplace(p.GetString(), desc.curveSets.size())
                     .first;
            desc.curveSets.push_back(std::move(cs));
            UsdGenGraphDescCaptureCache::CurveCapture captured;
            captured.path = p;
            captured.curves = desc.curveSets.back();
            captured.exists = exists;
            cache->curves.emplace(p.GetString(), std::move(captured));
        } else if (role == UsdGenRole::Reference &&
                   desc.curveSets[it->second].role != UsdGenRole::Reference) {
            // Reference is the stronger lane claim (I3). The role feeds the
            // generation, so a reused entry drops its carried one and
            // Finalize re-hashes; a fresh entry was never finalized.
            desc.curveSets[it->second].role = UsdGenRole::Reference;
            desc.curveSets[it->second].curveGeneration = 0;
            auto const kept = cache->curves.find(p.GetString());
            if (kept != cache->curves.end()) {
                kept->second.curves.role = UsdGenRole::Reference;
                kept->second.curves.curveGeneration = 0;
            }
        }
    };

    auto mapFor = [&](SdfPath const &p) {
        if (mapIndex.count(p.GetString())) {
            return;
        }
        if (reuseGeometry) {
            auto const prev = previousCache->maps.find(p.GetString());
            // A missing map builds a silent default either way, so every
            // cached entry is reusable while its prim and any paint
            // surface stay clean.
            bool paintClean = true;
            if (prev != previousCache->maps.end())
                for (SdfPath const &s : prev->second.paintSources)
                    if (_HNodeDirty(s, options.dirtyPrimPaths)) paintClean = false;
            if (prev != previousCache->maps.end() && paintClean &&
                !_HNodeDirty(p, options.dirtyPrimPaths)) {
                mapIndex.emplace(p.GetString(), desc.maps.size());
                desc.maps.push_back(prev->second.map);
                desc.validationErrors.insert(desc.validationErrors.end(),
                    prev->second.validationErrors.begin(),
                    prev->second.validationErrors.end());
                cache->maps.emplace(p.GetString(), prev->second);
                result.reusedInputs.push_back(p);
                for (SdfPath const &s : prev->second.paintSources)
                    result.reusedInputs.push_back(s);
                return;
            }
        }
        usdGen::UsdGenMapDesc map;
        map.path = p;
        HdContainerDataSourceHandle primDs;
        TfToken primType;
        std::vector<std::string> mapErrors;
        SdfPathVector paintSources;
        if (_HPrim(input, p, &primDs, &primType)) {
            HdContainerDataSourceHandle const ug = _HUsdGen(primDs);
            TfToken type;
            _HGetToken(ug, t, &type, {"type"});
            if (type.IsEmpty()) {
                type = primType.IsEmpty() ? _HUsdTypeName(primDs) : primType;
            }
            map.type = type;
            SdfAssetPath asset;
            if (_HGetTyped(ug, t, &asset, {"map", "file"})) {
                // Stage-free by value (S13): the RESOLVED path travels.
                map.resolvedAssetPath = asset.GetResolvedPath();
                if (map.resolvedAssetPath.empty()) {
                    map.resolvedAssetPath = asset.GetAssetPath();
                }
            }
            _HPullUsdGen(ug, t, nullptr, &map.params, "usdGen");
            if (map.type == TfToken("UsdGenPaintMap")) {
                _HCapturePaintMap(input, primDs, t, &map, &mapErrors,
                                  &paintSources);
            }
        }
        mapIndex.emplace(p.GetString(), desc.maps.size());
        desc.maps.push_back(map);
        desc.validationErrors.insert(desc.validationErrors.end(),
            mapErrors.begin(), mapErrors.end());
        UsdGenGraphDescCaptureCache::MapCapture captured;
        captured.path = p;
        captured.map = map;
        captured.validationErrors = std::move(mapErrors);
        captured.paintSources = std::move(paintSources);
        cache->maps.emplace(p.GetString(), std::move(captured));
    };

    // Expression inputs: what geoSampler()/ptex() read. Each target is read
    // through this index, so its edits dirty the description like a surface.
    std::map<std::string, size_t> geometryIndex;
    for (usdGen::UsdGenExpressionDesc &expression : desc.expressions) {
        for (usdGen::UsdGenExpressionInputDesc &in : expression.inputs) {
            for (SdfPath const &target : in.targets) {
                size_t const before = in.geometries.size() + in.maps.size();
                _HCollectInputTarget(input, target, 0, &in.geometries, &in.maps);
                if (in.geometries.size() + in.maps.size() == before)
                    desc.validationErrors.push_back(expression.path.GetString() + ": input:" +
                        in.name.GetString() + " target " + target.GetString() +
                        " is not a mesh, curves, points or map prim, and contains none");
            }
            for (SdfPath const &g : in.geometries) {
                if (geometryIndex.count(g.GetString())) continue;
                if (reuseGeometry) {
                    auto const prev =
                        previousCache->geometries.find(g.GetString());
                    // A missing geometry builds a silent default either
                    // way; a subset also reads its parent mesh.
                    bool const parentClean =
                        prev == previousCache->geometries.end() ||
                        !prev->second.geometry.isSubset ||
                        !_HNodeDirty(g.GetParentPath(),
                                     options.dirtyPrimPaths);
                    if (prev != previousCache->geometries.end() &&
                        parentClean &&
                        !_HNodeDirty(g, options.dirtyPrimPaths)) {
                        geometryIndex.emplace(g.GetString(),
                                              desc.geometries.size());
                        desc.geometries.push_back(prev->second.geometry);
                        desc.validationErrors.insert(
                            desc.validationErrors.end(),
                            prev->second.validationErrors.begin(),
                            prev->second.validationErrors.end());
                        cache->geometries.emplace(g.GetString(), prev->second);
                        result.reusedInputs.push_back(g);
                        if (prev->second.geometry.isSubset)
                            result.reusedInputs.push_back(g.GetParentPath());
                        continue;
                    }
                }
                usdGen::UsdGenGeometryDesc geometry;
                std::vector<std::string> geometryErrors;
                _HBuildGeometry(input, g, t, &geometry, &geometryErrors);
                geometryIndex.emplace(g.GetString(), desc.geometries.size());
                desc.geometries.push_back(geometry);
                desc.validationErrors.insert(desc.validationErrors.end(),
                    geometryErrors.begin(), geometryErrors.end());
                UsdGenGraphDescCaptureCache::GeometryCapture captured;
                captured.path = g;
                captured.geometry = geometry;
                captured.validationErrors = std::move(geometryErrors);
                cache->geometries.emplace(g.GetString(), std::move(captured));
            }
            for (SdfPath const &m : in.maps) mapFor(m);
        }
    }

    TRACE_SCOPE("usdGen capture surfaces, curves and maps");
    for (UsdGenNodeDesc &node : desc.nodes) {
        for (SdfPath const &s : node.surfaces) {
            surfaceFor(s);
        }
        for (SdfPath const &c : node.curves) {
            UsdGenRole role = UsdGenRole::Curves;
            auto roleIt = curveRoles.find(c.GetString());
            if (roleIt != curveRoles.end()) {
                role = roleIt->second;
            }
            curveFor(c, role);
        }
        for (SdfPath const &reference : node.references) {
            curveFor(reference, UsdGenRole::Reference);
        }
        for (SdfPath const &m : node.maps) {
            mapFor(m);
        }
    }

    // ---- description-level fields (02 §2.3/§2.12/§2.14) -------------------
    _HGetDedicated(descUg, t, &desc.defaultWidth, &desc.validationErrors,
                   desc.description, "width:default", {"width", "default"});
    _HGetDedicated(descUg, t, &desc.tileTarget, &desc.validationErrors,
                   desc.description, "tileTarget", {"tileTarget"});
    _HGetToken(descUg, t, &desc.curveBasis, {"curve", "basis"});

    HdContainerDataSourceHandle groomDs;
    if (_HPrim(input, descriptionPath.GetParentPath(), &groomDs, nullptr)) {
        desc.executionBackend = _HResolveExecutionBackend(
            _HUsdGen(groomDs), t);
    }

    UsdGenLookDesc &look = desc.look;
    HdContainerDataSourceHandle const lookDs = _HChild(descUg, "look");
    bool const gotRoot = _HGetTyped(lookDs, t, &look.rootColor, {"rootColor"});
    _HGetTyped(lookDs, t, &look.tipColor, {"tipColor"});
    // usdGen:look:* lives on UsdGenLookAPI, which a scene applies per
    // description. A mapping table built from the concrete type alone serves
    // no `look` container and every field here silently takes its schema
    // default, which renders as a slightly different dark brown -- invisible
    // unless you read the values back. Say which happened.
    // "Authored" is the predicate the session cooker uses to decide whether
    // the look beats the source curves' own displayColor: an opinion that
    // DIFFERS from the schema fallback. Applying UsdGenLookAPI without setting
    // anything leaves a full set of fallbacks, which must not count.
    TF_DEBUG(USDGEN_INGRESS).Msg(
        "usdGen capture look on %s: %s, rootColor %.4f %.4f %.4f\n",
        desc.description.GetText(),
        !lookDs ? "NO look container (is UsdGenLookAPI mapped?)"
                : (gotRoot && look.rootColor != UsdGenLookDesc().rootColor)
                      ? "authored"
                      : "schema fallback",
        double(look.rootColor[0]), double(look.rootColor[1]),
        double(look.rootColor[2]));
    _HGetTyped(lookDs, t, &look.rampColors, {"colorRamp", "colors"});
    _HGetTyped(lookDs, t, &look.rampPositions, {"colorRamp", "positions"});
    _HGetToken(lookDs, t, &look.rampInterpolation,
               {"colorRamp", "interpolation"});
    _HGetTyped(lookDs, t, &look.rampExponent, {"rampExponent"});
    _HGetToken(lookDs, t, &look.bakeMode, {"bakeMode"});
    _HGetToken(lookDs, t, &look.bakeTarget, {"bakeTarget"});
    _HGetToken(lookDs, t, &look.bakePrimvar, {"bakePrimvar"});
    _HGetTyped(lookDs, t, &look.hueJitter, {"hueJitter"});
    _HGetTyped(lookDs, t, &look.valueJitter, {"valueJitter"});
    _HGetTyped(lookDs, t, &look.jitterSeed, {"jitterSeed"});

    // usdGen:preview:* (viewport value preview); twin of the stage builder.
    usdGen::UsdGenPreviewDesc &preview = desc.preview;
    HdContainerDataSourceHandle const previewDs = _HChild(descUg, "preview");
    {
        SdfPathVector targets;
        if (_HGetPathArray(previewDs, &targets, {"source"}) && !targets.empty())
            preview.source = targets.front();
    }
    _HGetToken(previewDs, t, &preview.colorMap, {"colorMap"});
    _HGetTyped(previewDs, t, &preview.range, {"range"});
    _HGetToken(previewDs, t, &preview.evaluation, {"evaluation"});
    _HGetToken(previewDs, t, &preview.shading, {"shading"});
    if (preview.source.IsPrimPath()) {
        HdContainerDataSourceHandle targetDs;
        if (_HPrim(input, preview.source, &targetDs, nullptr) &&
            UsdGenIsMapTypeName(_HUsdTypeName(targetDs)))
            mapFor(preview.source);
    }

    // Purpose keeps the V2-9a absence semantics for free: Hydra carries only
    // authored opinions, so an unauthored purpose simply has no data source
    // and the desc stays EMPTY (the publisher then omits the purpose
    // container and Storm falls back to the geometry render tag). A served
    // "default" can only be the schema fallback leaking through, never an
    // authored render tag, so it maps to empty too — mirroring the stage
    // path's HasAuthoredValueOpinion gate without a stage query.
    // Visibility keeps its (valid) fallback: "inherited" is a real Hydra
    // visibility value.
    {
        HdTokenDataSourceHandle const ps =
            HdPurposeSchema::GetFromParent(descDs).GetPurpose();
        if (ps) {
            TfToken const v = ps->GetTypedValue(t);
            if (!v.IsEmpty() && v != TfToken("default")) {
                desc.purpose = v;
            }
        }
    }
    {
        HdBoolDataSourceHandle const vs =
            HdVisibilitySchema::GetFromParent(descDs).GetVisibility();
        bool visible = true;
        if (vs) {
            visible = vs->GetTypedValue(t);
        }
        desc.visibility = visible ? TfToken("inherited") : TfToken("invisible");
    }

    {
        HdMaterialBindingSchema const b =
            HdMaterialBindingsSchema::GetFromParent(descDs)
                .GetMaterialBinding();
        HdPathDataSourceHandle const p = b.GetPath();
        if (p) {
            SdfPath const mp = p->GetTypedValue(t);
            if (!mp.IsEmpty()) {
                desc.materialPath = mp;
            }
        }
    }

    desc.xformMatrix = GfMatrix4d(1.0);
    {
        HdMatrixDataSourceHandle const m =
            HdXformSchema::GetFromParent(descDs).GetMatrix();
        if (m) {
            desc.xformMatrix = m->GetTypedValue(t);
        }
    }

    ResolveUsdGenImageMaps(&desc);
    UsdGenFinalizeInputGenerations(&desc);
    // The pooled descs were cached before their generations existed; carry
    // the fresh ones back so a reuse restores a fully finalized desc and
    // Finalize can skip its proven-identical arrays outright.
    for (usdGen::UsdGenSurfaceDesc const &surface : desc.surfaces) {
        auto const kept = cache->surfaces.find(surface.path.GetString());
        if (kept != cache->surfaces.end()) {
            kept->second.surface.surfaceGeneration = surface.surfaceGeneration;
            kept->second.surface.surfaceInvariantGeneration =
                surface.surfaceInvariantGeneration;
        }
    }
    for (usdGen::UsdGenCurveSetDesc const &curves : desc.curveSets) {
        auto const kept = cache->curves.find(curves.path.GetString());
        if (kept != cache->curves.end())
            kept->second.curves.curveGeneration = curves.curveGeneration;
    }
    result.cache = std::move(cache);
    return result;
}

usdGen::UsdGenGraphDesc
BuildGraphDescFromHydra(
    HdSceneIndexBase &input,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options)
{
    return CaptureGraphDescFromHydra(input, descriptionPath, options).desc;
}

}  // namespace usdGenImaging
