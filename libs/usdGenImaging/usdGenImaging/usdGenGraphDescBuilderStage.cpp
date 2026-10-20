// usdGen imaging — stage-sourced graph description builder implementation.
//
// This translation unit is the offline UsdStage oracle. It is deliberately
// separate from the production Hydra builder so the production object has no
// UsdStage dependencies (B-2/V2-11).
#include "usdGenImaging/usdGenGraphDescBuilderStage.h"
#include "usdGenImaging/expressionConnection.h"
#include "usdGenImaging/imageMapCache.h"
#include "usdGenImaging/usdGenGraphDescShared.h"

#include "usdGen/authoredPlaneConvert.h"
#include "usdGen/expressions/valueShape.h"
#include "usdGen/executionBackend.h"

#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/vt/array.h"
#include "pxr/usd/sdf/assetPath.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usd/relationship.h"
#include "pxr/usd/usdGeom/basisCurves.h"
#include "pxr/usd/usdGeom/imageable.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/points.h"
#include "pxr/usd/usdGeom/primvar.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/subset.h"
#include "pxr/usd/usdShade/material.h"
#include "pxr/usd/usdShade/materialBindingAPI.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

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


template <class T>
bool
_GetTyped(UsdAttribute const &attr, UsdTimeCode time, T *out)
{
    if (!attr) {
        return false;
    }
    VtValue value;
    if (!attr.Get(&value, time) || value.IsEmpty()) {
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
_GetDedicated(UsdAttribute const &attr, UsdTimeCode time, T *out,
              std::vector<std::string> *errors, SdfPath const &path,
              char const *name)
{
    if (!attr) return;
    VtValue value;
    if (!attr.Get(&value, time) || value.IsEmpty()) return;
    if (value.IsHolding<T>()) {
        *out = value.UncheckedGet<T>();
        return;
    }
    errors->push_back(path.GetString() + ": " + name + " has wrong authored type");
}

/// point3f[] → VtVec3fArray elementwise conversion. USD mesh points author as point3f[];
/// the engine surface descs carry VtVec3fArray.
// rest/points pulls come back empty, Scatter captures 0 roots and the
// commit publishes 0 tiles (surgery gen=0 root cause family).
bool
_GetVec3fArray(UsdAttribute const &attr, UsdTimeCode time, VtVec3fArray *out)
{
    if (!attr) {
        return false;
    }
    VtValue value;
    if (!attr.Get(&value, time) || value.IsEmpty()) {
        return false;
    }
    if (value.IsHolding<VtVec3fArray>()) {
        *out = value.UncheckedGet<VtVec3fArray>();
        return true;
    }
    return _GetTyped(attr, time, out);
}

void
_GetToken(UsdPrim const &prim, char const *name, TfToken *out)
{
    TfToken v;
    if (_GetTyped(prim.GetAttribute(TfToken(name)), UsdTimeCode::Default(), &v)
        && !v.IsEmpty()) {
        *out = v;
    }
}

// Do not use _GetToken here.  The compatibility contract treats a genuinely
// absent property as the legacy CPU baseline, but a schema property with its
// resolved "cuda" fallback is still an explicit CUDA selection.  Inspect the
// raw VtValue and reject every present value except the approved token.
usdGen::UsdGenExecutionBackend
_ResolveExecutionBackend(UsdPrim const &prim)
{
    UsdAttribute const attr =
        prim.GetAttribute(TfToken("usdGen:execution:backend"));
    if (!attr) {
        return usdGen::UsdGenExecutionBackend::CpuReference;
    }
    VtValue value;
    if (!attr.Get(&value, UsdTimeCode::Default()) ||
        !value.IsHolding<TfToken>()) {
        return usdGen::UsdGenExecutionBackend::Invalid;
    }
    return usdGen::ParseUsdGenExecutionBackend(value.UncheckedGet<TfToken>());
}

void
_GetPrimvar(UsdPrim const &prim, TfToken const &name, UsdTimeCode time,
            VtValue *out)
{
    UsdGeomPrimvar pv = UsdGeomPrimvarsAPI(prim).GetPrimvar(name);
    if (!pv) {
        return;
    }
    pv.Get(out, time);
    if (out->IsEmpty() && time != UsdTimeCode::Default()) {
        pv.Get(out, UsdTimeCode::Default());
    }
}

template <class T>
void
_GetPrimvarTyped(UsdPrim const &prim, TfToken const &name, UsdTimeCode time,
                 T *out)
{
    VtValue v;
    _GetPrimvar(prim, name, time, &v);
    if (v.IsHolding<T>()) {
        *out = v.UncheckedGet<T>();
    } else if (v.CanCast<T>()) {
        *out = v.template Cast<T>().template UncheckedGet<T>();
    }
}

UsdGenSurfaceNormalDomain
_NormalDomain(TfToken const& interpolation)
{
    if (interpolation == TfToken("constant")) return UsdGenSurfaceNormalDomain::Constant;
    if (interpolation == TfToken("uniform")) return UsdGenSurfaceNormalDomain::Uniform;
    if (interpolation == TfToken("vertex")) return UsdGenSurfaceNormalDomain::Vertex;
    if (interpolation == TfToken("faceVarying")) return UsdGenSurfaceNormalDomain::FaceVarying;
    return UsdGenSurfaceNormalDomain::Invalid;
}

void
_PullParams(UsdPrim const &prim, double time,
            std::vector<UsdGenParamValue> *params)
{
    for (TfToken const &attrName : prim.GetPrimDefinition().GetPropertyNames()) {
        std::string const s = attrName.GetString();
        if (s.compare(0, 7, "usdGen:") != 0 || _isDedicated(attrName)) {
            continue;
        }
        UsdAttribute const attr = prim.GetAttribute(attrName);
        if (!attr) {
            continue;  // a relationship carries the name, not an attribute
        }
        VtValue value;
        if (!attr.Get(&value, UsdTimeCode(time)) || value.IsEmpty()) {
            continue;  // no authored/default opinion at all
        }
        UsdGenParamValue p;
        p.name = TfToken(s.substr(7));  // strip "usdGen:" (02 §0.7)
        p.value = std::move(value);
        p.animated = attr.ValueMightBeTimeVarying();
        params->push_back(std::move(p));
    }
    // ramps (S11) stay empty here: the capture adapter resolves ramp
    // encodings at capture time (graphDesc.h: "already resolved by the
    // adapter"); the knot/position/color attributes themselves were swept
    // into params above, so S14's pull-all contract holds.
}

// The Mesh a surface target reads (R15): the target itself, a GeomSubset's
// parent Mesh, or empty for anything else.
SdfPath
_SurfaceMesh(UsdStageRefPtr const &stage, SdfPath const &path)
{
    UsdPrim const prim = stage->GetPrimAtPath(path);
    if (!prim) return SdfPath();
    if (prim.IsA<UsdGeomMesh>()) return path;
    if (prim.IsA<UsdGeomSubset>() && prim.GetParent().IsA<UsdGeomMesh>())
        return path.GetParentPath();
    return SdfPath();
}

// Snapshots a UsdGenPaintMap's surface primvar into the map desc (scalar,
// folded through usdGen:map:channel). v1 carries faceVarying float data
// only; anything else fails closed with a validation error and leaves the
// payload empty, which the sampler treats as unusable. A missing primvar
// reads usdGen:map:default everywhere instead of failing.
void
_CapturePaintMap(UsdStageRefPtr const &stage, UsdPrim const &prim,
                 double time, usdGen::UsdGenMapDesc *map,
                 std::vector<std::string> *errors)
{
    auto fail = [&](std::string const &what) {
        if (errors) {
            errors->push_back(map->path.GetString() + ": " + what);
        }
    };
    UsdRelationship rel = prim.GetRelationship(TfToken("usdGen:paint:surface"));
    SdfPathVector targets;
    if (rel) {
        rel.GetTargets(&targets);
    }
    if (targets.size() != 1) {
        fail("usdGen:paint:surface requires exactly one target");
        return;
    }
    // A face GeomSubset target (R15) paints its parent Mesh: the primvar is
    // read there and the corners outside the subset read usdGen:map:default.
    SdfPath const meshPath = _SurfaceMesh(stage, targets.front());
    UsdPrim surface = meshPath.IsEmpty() ? UsdPrim() : stage->GetPrimAtPath(meshPath);
    if (!surface) {
        fail("usdGen:paint:surface must target a UsdGeomMesh or a face "
             "GeomSubset of one");
        return;
    }
    bool const subset = meshPath != targets.front();
    VtIntArray subsetFaces;
    if (subset) {
        UsdGeomSubset const geomSubset(stage->GetPrimAtPath(targets.front()));
        TfToken elementType;
        geomSubset.GetElementTypeAttr().Get(&elementType);
        if (elementType != UsdGeomTokens->face) {
            fail(UsdGenSubsetElementTypeError(targets.front(),
                                              elementType.GetString()));
            return;
        }
        VtIntArray faceCounts, indices;
        UsdGeomMesh(surface).GetFaceVertexCountsAttr().Get(&faceCounts);
        _GetTyped(geomSubset.GetIndicesAttr(), UsdTimeCode(time), &indices);
        std::vector<int> faces;
        UsdGenAppendSubsetFaces(indices, faceCounts.size(), targets.front(),
                                meshPath, &faces, errors);
        subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), false, 0);
    }
    TfToken primvarName;
    _GetToken(prim, "usdGen:paint:primvar", &primvarName);
    if (primvarName.IsEmpty()) {
        fail("usdGen:paint:primvar is empty");
        return;
    }
    UsdGeomPrimvar pv = UsdGeomPrimvarsAPI(surface).GetPrimvar(primvarName);
    VtValue value;
    if (pv) {
        _GetPrimvar(surface, primvarName, UsdTimeCode(time), &value);
    }
    if (!pv || value.IsEmpty()) {
        // No baked primvar yet: the map reads its authored default
        // everywhere (the BaseGridFromStage semantic), so a wired but
        // unpainted map cooks instead of rejecting the commit.
        UsdAttribute defaultAttr = prim.GetAttribute(TfToken("usdGen:map:default"));
        float defaultValue = 0.0f;
        if (!defaultAttr || !defaultAttr.Get(&defaultValue)) {
            fail("surface " + meshPath.GetString() + " has no primvar " +
                 primvarName.GetString());
            return;
        }
        TfToken promised;
        _GetToken(prim, "usdGen:paint:interpolation", &promised);
        if (!promised.IsEmpty() && promised != UsdGeomTokens->faceVarying) {
            fail("primvar " + primvarName.GetString() + " is " + promised.GetString() +
                 ", v1 paints faceVarying only");
            return;
        }
        VtIntArray defaultCounts;
        UsdGeomMesh(surface).GetFaceVertexCountsAttr().Get(&defaultCounts);
        size_t defaultFaceVarying = 0;
        for (int c : defaultCounts) defaultFaceVarying += size_t(c);
        map->paintSurface = meshPath;
        map->paintPrimvar = primvarName;
        map->paintInterpolation = UsdGeomTokens->faceVarying;
        map->paintValues.assign(defaultFaceVarying, defaultValue);
        return;
    }
    TfToken const interp = pv.GetInterpolation();
    if (interp != UsdGeomTokens->faceVarying) {
        fail("primvar " + primvarName.GetString() + " is " + interp.GetString() +
             ", v1 paints faceVarying only");
        return;
    }
    TfToken channel;
    _GetToken(prim, "usdGen:map:channel", &channel);
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
    UsdGeomMesh(surface).GetFaceVertexCountsAttr().Get(&counts);
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
        if (UsdAttribute defaultAttr = prim.GetAttribute(TfToken("usdGen:map:default")))
            defaultAttr.Get(&fallback);
        UsdGenMaskPaintValues(counts, subsetFaces, fallback, &folded);
    }
    map->paintSurface = meshPath;
    map->paintPrimvar = primvarName;
    map->paintInterpolation = interp;
    map->paintValues = folded;
}

// Follows usdGen:input edges back from the terminal, collecting the operator
// set. Relationships are NOT traversed as data edges (02 §2.3).
struct _GraphWalker
{
    UsdStageRefPtr stage;
    std::unordered_set<std::string> operatorPaths;
    std::unordered_set<std::string> visited;
    // guides rels mark their curve sets as the Reference lane (I3)
    std::unordered_map<std::string, UsdGenRole> curveRoles;

    void Walk(UsdPrim const &prim)
    {
        if (!prim || !visited.insert(prim.GetPath().GetString()).second) {
            return;
        }
        operatorPaths.insert(prim.GetPath().GetString());
        for (UsdRelationship const &rel : prim.GetRelationships()) {
            // GetBaseName strips namespaces ("usdGen:input" -> "input").
            std::string const name = rel.GetBaseName().GetString();
            if (name == "input") {
                SdfPathVector targets;
                rel.GetTargets(&targets);
                for (SdfPath const &target : targets) {
                    Walk(stage->GetPrimAtPath(target));
                }
            } else if (name == "guides") {
                SdfPathVector targets;
                rel.GetTargets(&targets);
                for (SdfPath const &target : targets) {
                    curveRoles[target.GetString()] = UsdGenRole::Reference;
                }
            }
        }
    }
};

// Offline oracle equivalent of the Description aggregate.  GetChildren()
// returns final composed child order; reversing at every level gives the
// required lower-sibling-first post-order.  Scopes are traversal-only.
void
_AppendStageOperatorPostOrder(UsdPrim const &prim, SdfPathVector *out)
{
    std::vector<UsdPrim> children;
    for (UsdPrim const &child : prim.GetChildren()) children.push_back(child);
    for (auto child = children.rbegin(); child != children.rend(); ++child) {
        _AppendStageOperatorPostOrder(*child, out);
    }
    if (prim.IsA(TfToken("UsdGenOperator"))) {
        out->push_back(prim.GetPath());
    }
}

SdfPathVector
_StageOperatorOrder(UsdPrim const &description)
{
    SdfPathVector out;
    UsdPrim const ops = description.GetChild(TfToken("Ops"));
    if (!ops) {
        return out;
    }
    // Oracle twin of the adapter rule: bottom-up hierarchy, always.
    // Reverse prim order, post-order over groups; reordering the Ops
    // children reorders execution, and no metadata overrides it.
    std::vector<UsdPrim> children;
    for (UsdPrim const &child : ops.GetChildren()) children.push_back(child);
    for (auto child = children.rbegin(); child != children.rend(); ++child) {
        _AppendStageOperatorPostOrder(*child, &out);
    }
    return out;
}

usdGen::expr::ValueShape
_ExpressionShape(SdfValueTypeName const &type, VtValue const *declaration)
{
    uint32_t count = 0;
    bool const countKnown = !type.IsArray() ||
        (declaration && usdGen::expr::FixedArrayElementCount(
            type, *declaration, &count));
    return usdGen::expr::ValueShapeFromSdfType(type, count, countKnown);
}

usdGen::expr::Domain
_ExpressionDomain(UsdAttribute const &attr)
{
    VtValue v = attr.GetCustomDataByKey(TfToken("usdGen:evaluation"));
    TfToken const d = v.IsHolding<TfToken>() ? v.UncheckedGet<TfToken>() :
        (v.IsHolding<std::string>() ? TfToken(v.UncheckedGet<std::string>()) : TfToken("groom"));
    if (d == TfToken("primitive")) return usdGen::expr::Domain::Primitive;
    if (d == TfToken("point")) return usdGen::expr::Domain::Point;
    return usdGen::expr::Domain::Groom;
}

// Per-face scatter density scale from the composed usdGen:paint:density
// face-varying primvar (the brush bake and live scratch write it): face
// means, clamped >= 0, non-finite clamped to 0. A missing primvar, wrong
// interpolation/type or a size mismatch leaves the multiplier empty,
// which scatter reads as all 1.0.
void
_CaptureDensityMultiplier(UsdPrim const &meshPrim, double time,
                          UsdGenSurfaceDesc *out)
{
    UsdGeomPrimvar pv = UsdGeomPrimvarsAPI(meshPrim).GetPrimvar(
        TfToken("usdGen:paint:density"));
    if (!pv) return;
    if (pv.GetInterpolation() != TfToken("faceVarying")) return;
    VtValue value;
    if (!pv.GetAttr().Get(&value, UsdTimeCode(time)) ||
        !value.IsHolding<VtFloatArray>()) return;
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

// Fills a surface desc from a Mesh; a GeomSubset target instead records
// subsetFaces onto the parent mesh's desc (R15; caller handles the parent).
void
_BuildSurface(UsdStageRefPtr const &stage, SdfPath const &path, double time,
              UsdGenSurfaceDesc *out)
{
    UsdPrim prim = stage->GetPrimAtPath(path);
    if (!prim) {
        TF_CODING_ERROR("usdGen: usdGen:surface target %s does not exist.",
                        path.GetText());
        return;
    }
    UsdGeomMesh mesh(prim);
    if (!mesh) {
        TF_CODING_ERROR("usdGen: usdGen:surface target %s is a %s; M1 "
                        "surfaces must be Mesh or GeomSubset (R15).",
                        path.GetText(),
                        prim.GetPrimTypeInfo().GetTypeName().GetText());
        return;
    }
    UsdPrim const meshPrim = mesh.GetPrim();

    out->path = path;
    mesh.GetSubdivisionSchemeAttr().Get(&out->subdivisionScheme);
    mesh.GetOrientationAttr().Get(&out->orientation);
    mesh.GetInterpolateBoundaryAttr().Get(&out->interpolateBoundary);
    mesh.GetFaceVaryingLinearInterpolationAttr().Get(&out->faceVaryingLinearInterpolation);
    mesh.GetTriangleSubdivisionRuleAttr().Get(&out->triangleSubdivisionRule);
    mesh.GetHoleIndicesAttr().Get(&out->holeIndices);
    mesh.GetCreaseIndicesAttr().Get(&out->creaseIndices);
    mesh.GetCreaseLengthsAttr().Get(&out->creaseLengths);
    mesh.GetCreaseSharpnessesAttr().Get(&out->creaseSharpnesses);
    mesh.GetCornerIndicesAttr().Get(&out->cornerIndices);
    mesh.GetCornerSharpnessesAttr().Get(&out->cornerSharpnesses);
    _GetTyped(mesh.GetFaceVertexCountsAttr(), UsdTimeCode::Default(),
              &out->faceVertexCounts);
    _GetTyped(mesh.GetFaceVertexIndicesAttr(), UsdTimeCode::Default(),
              &out->faceVertexIndices);
    _GetVec3fArray(mesh.GetPointsAttr(), UsdTimeCode(time), &out->points);
    auto restPrimvar = UsdGeomPrimvarsAPI(meshPrim).GetPrimvar(TfToken("rest"));
    TfToken restSource("default");
    if (auto attr = meshPrim.GetAttribute(TfToken("usdGen:rest:source")))
        attr.Get(&restSource, UsdTimeCode::Default());
    bool authoredRest = restPrimvar && restPrimvar.GetAttr().GetResolveInfo().HasAuthoredValueOpinion();
    if (restSource != TfToken("default")) {
        // Match the adapter's fail-closed unsupported rest-source behavior.
        out->restFromCurrentPoints = true;
    } else if (authoredRest) {
        _GetPrimvarTyped(meshPrim, TfToken("rest"), UsdTimeCode::Default(), &out->restPoints);
    } else {
        // S12: no authored rest -> the Default-time deformed opinion IS the
        // rest (the UsdGenRestAPI adapter publishes the same fallback).
        _GetVec3fArray(mesh.GetPointsAttr(), UsdTimeCode::Default(),
                       &out->restPoints);
    }
    _GetPrimvarTyped(meshPrim, TfToken("st"), UsdTimeCode::Default(), &out->uv);
    // Rest normals are a paired Default-time Mesh attribute/interpolation,
    // never the current evaluation-time mesh normal. Empty is the valid
    // geometric-normal fallback; nonempty bad typing/interpolation stays
    // Invalid for CUDA admission.
    VtValue normals;
    if (mesh.GetNormalsAttr().Get(&normals, UsdTimeCode::Default()) && !normals.IsEmpty()) {
        if (normals.IsHolding<VtVec3fArray>()) {
            out->restNormals = normals.UncheckedGet<VtVec3fArray>();
            if (!out->restNormals.empty())
                out->restNormalDomain = _NormalDomain(mesh.GetNormalsInterpolation());
        } else {
            out->restNormalDomain = UsdGenSurfaceNormalDomain::Invalid;
        }
    }
    _GetPrimvarTyped(meshPrim, TfToken("velocities"), UsdTimeCode(time),
                     &out->velocities);  // motion profile P1 only
    _CaptureDensityMultiplier(meshPrim, time, out);

    out->samples.push_back(UsdGenSurfaceSample{time, out->points});
    out->worldMatrix = UsdGeomImageable(prim).ComputeLocalToWorldTransform(
        UsdTimeCode(time));
}

// Fills a surface desc for a GeomSubset target (02 §2.20): its parent Mesh's
// full geometry, so face, ptex and root-binding ids stay parent-mesh ids,
// plus the sorted, unique faces it selects. `unioned` are further targets on
// the same mesh (sibling subsets, or the Mesh itself) folded in by rule 4.
// A non-face subset or a non-Mesh parent is a hard diagnostic (rules 1, 7).
void
_BuildSubsetSurface(UsdStageRefPtr const &stage, SdfPath const &path,
                    SdfPathVector const &unioned, double time,
                    UsdGenSurfaceDesc *out, std::vector<std::string> *errors)
{
    SdfPath const meshPath = path.GetParentPath();
    if (!UsdGeomMesh(stage->GetPrimAtPath(meshPath))) {
        errors->push_back(UsdGenSubsetParentError(path));
        return;
    }
    _BuildSurface(stage, meshPath, time, out);
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
        UsdGeomSubset const subset(stage->GetPrimAtPath(member));
        TfToken elementType;
        subset.GetElementTypeAttr().Get(&elementType);
        if (elementType != UsdGeomTokens->face) {
            errors->push_back(
                UsdGenSubsetElementTypeError(member, elementType.GetString()));
            continue;
        }
        VtIntArray indices;
        _GetTyped(subset.GetIndicesAttr(), UsdTimeCode(time), &indices);
        UsdGenAppendSubsetFaces(indices, faceCount, member, meshPath, &faces,
                                errors);
    }
    out->subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), whole, faceCount);
}

bool
_BuildSurfaceCagePayload(UsdPrim const &prim, UsdTimeCode time,
                         std::shared_ptr<const UsdGenSurfaceCagePayload> *out)
{
    if (!prim || !out) return false;
    static TfToken const names[] = {
        TfToken("usdGen:surfaceCage:ownerIds"),
        TfToken("usdGen:surfaceCage:ownerDensities"),
        TfToken("usdGen:surfaceCage:ownerSeeds"),
        TfToken("usdGen:surfaceCage:ownerCvCounts"),
        TfToken("usdGen:surfaceCage:ownerEdgeBias"),
        TfToken("usdGen:surfaceCage:ownerLengthProfileOffsets"),
        TfToken("usdGen:surfaceCage:ownerLengthProfile"),
        TfToken("usdGen:surfaceCage:normalizedT"),
        TfToken("usdGen:surfaceCage:triangles"),
        TfToken("usdGen:surfaceCage:triangleOwnerIndices"),
        TfToken("usdGen:surfaceCage:triangleRootCharts"),
        TfToken("usdGen:surfaceCage:ownerChartCentroids"),
        TfToken("usdGen:surfaceCage:ownerChartMeanRadii")};
    bool present = false;
    for (TfToken const &name : names) {
        if (prim.GetAttribute(name)) {
            present = true;
            break;
        }
    }
    if (!present) return false;

    std::shared_ptr<UsdGenSurfaceCagePayload> cage =
        std::make_shared<UsdGenSurfaceCagePayload>();
    if (!_GetTyped(prim.GetAttribute(names[0]), time, &cage->ownerIds) ||
        !_GetTyped(prim.GetAttribute(names[1]), time, &cage->ownerDensities) ||
        !_GetTyped(prim.GetAttribute(names[2]), time, &cage->ownerSeeds) ||
        !_GetTyped(prim.GetAttribute(names[3]), time, &cage->ownerCvCounts) ||
        !_GetTyped(prim.GetAttribute(names[4]), time, &cage->ownerEdgeBias) ||
        !_GetTyped(prim.GetAttribute(names[5]), time,
                   &cage->ownerLengthProfileOffsets) ||
        !_GetTyped(prim.GetAttribute(names[6]), time,
                   &cage->ownerLengthProfile) ||
        !_GetTyped(prim.GetAttribute(names[7]), time, &cage->normalizedT) ||
        !_GetTyped(prim.GetAttribute(names[8]), time, &cage->triangles) ||
        !_GetTyped(prim.GetAttribute(names[9]), time,
                   &cage->triangleOwnerIndices) ||
        !_GetTyped(prim.GetAttribute(names[10]), time,
                   &cage->triangleRootCharts) ||
        !_GetTyped(prim.GetAttribute(names[11]), time,
                   &cage->ownerChartCentroids) ||
        !_GetTyped(prim.GetAttribute(names[12]), time,
                   &cage->ownerChartMeanRadii)) {
        return false;
    }
    *out = std::move(cage);
    return true;
}

void
_BuildCurveSet(UsdStageRefPtr const &stage, SdfPath const &path,
               UsdGenRole role, double time, UsdGenCurveSetDesc *out)
{
    UsdPrim prim = stage->GetPrimAtPath(path);
    if (!prim) {
        TF_CODING_ERROR("usdGen: curve set %s does not exist.",
                        path.GetText());
        return;
    }
    UsdGeomBasisCurves curves(prim);
    if (!curves) return;
    out->path = path;
    out->role = role;
    out->worldMatrix = UsdGeomImageable(prim).ComputeLocalToWorldTransform(UsdTimeCode(time));

    _GetTyped(curves.GetTypeAttr(), UsdTimeCode(time), &out->type);
    _GetTyped(curves.GetBasisAttr(), UsdTimeCode(time), &out->basis);
    _GetTyped(curves.GetWrapAttr(), UsdTimeCode(time), &out->wrap);
    out->widthsInterpolation = curves.GetWidthsInterpolation();

    _GetTyped(curves.GetCurveVertexCountsAttr(), UsdTimeCode(time),
              &out->curveVertexCounts);
    _GetTyped(curves.GetPointsAttr(), UsdTimeCode(time), &out->points);
    _GetPrimvarTyped(prim, TfToken("rest"), UsdTimeCode::Default(), &out->rest);
    if (out->rest.empty()) {
        _GetTyped(curves.GetPointsAttr(), UsdTimeCode::Default(), &out->rest);
    }
    UsdGeomPrimvar widthsPrimvar = UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken("widths"));
    if (widthsPrimvar && widthsPrimvar.HasValue()) {
        widthsPrimvar.Get(&out->widths, UsdTimeCode(time));
        out->widthsInterpolation = widthsPrimvar.GetInterpolation();
    } else {
        _GetTyped(curves.GetWidthsAttr(), UsdTimeCode(time), &out->widths);
    }
    _BuildSurfaceCagePayload(prim, UsdTimeCode(time), &out->surfaceCage);
    _GetPrimvarTyped(prim, TfToken("skinprim"), UsdTimeCode(time),
                     &out->skinPrim);
    _GetPrimvarTyped(prim, TfToken("usdGen:curveId"), UsdTimeCode(time),
                     &out->curveId);
    _GetPrimvarTyped(prim, TfToken("skinprimuv"), UsdTimeCode(time),
                     &out->skinPrimUv);
    _GetPrimvarTyped(prim, TfToken("usdGen:rootFrame"), UsdTimeCode(time),
                     &out->rootFrame);

    // The source curves' own displayColor, forwarded so a groom that styles
    // nothing shows the colour its asset already carries. Any interpolation is
    // accepted: constant/uniform/vertex map onto the three plane domains.
    if (UsdGeomPrimvar pv =
            UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken("displayColor"))) {
        VtVec3fArray colors;
        if (pv.Get(&colors, UsdTimeCode(time)) && !colors.empty()) {
            usdGen::UsdGenAuthoredPlaneDesc plane;
            plane.name = usdGen::UsdGenSourceColorPlane();
            plane.type = usdGen::UsdGenAuthoredPlaneType::Float32;
            plane.arity = 3;
            TfToken const interpolation = pv.GetInterpolation();
            if (interpolation == UsdGeomTokens->constant || colors.size() == 1) {
                plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Groom;
            } else if (interpolation == UsdGeomTokens->uniform) {
                plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Primitive;
            } else {
                plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Point;
            }
            size_t const elements =
                plane.domain == usdGen::UsdGenAuthoredPlaneDomain::Groom
                    ? 1 : colors.size();
            plane.floatValues.reserve(elements * 3);
            for (size_t i = 0; i < elements; ++i) {
                plane.floatValues.push_back(colors[i][0]);
                plane.floatValues.push_back(colors[i][1]);
                plane.floatValues.push_back(colors[i][2]);
            }
            out->authoredPlanes.push_back(std::move(plane));
        }
    }

    // OutputCurves ownership is carried by one uniform integer per source
    // curve.  Do not reinterpret a malformed or differently interpolated
    // primvar: the authored-plane contract requires a primitive-domain scalar.
    auto forwardOwnership = [&](TfToken const &name) {
        UsdGeomPrimvar const pv = UsdGeomPrimvarsAPI(prim).GetPrimvar(name);
        if (!pv || pv.GetInterpolation() != UsdGeomTokens->uniform) return;
        VtIntArray values;
        if (!pv.Get(&values, UsdTimeCode(time)) ||
            values.size() != out->curveVertexCounts.size()) {
            return;
        }
        usdGen::UsdGenAuthoredPlaneDesc plane;
        plane.name = name;
        plane.type = usdGen::UsdGenAuthoredPlaneType::Int32;
        plane.domain = usdGen::UsdGenAuthoredPlaneDomain::Primitive;
        plane.arity = 1;
        plane.intValues = std::move(values);
        out->authoredPlanes.push_back(std::move(plane));
    };
    forwardOwnership(TfToken("tubeId"));
    forwardOwnership(TfToken("regionId"));
    forwardOwnership(TfToken("hierarchyLevel"));

    // Generic transport: primvars:usdGen:plane:<name> -> authored plane
    // <name> (usdGen/authoredPlaneConvert.h). Indexed primvars are flattened.
    {
        size_t points = 0;
        for (int count : out->curveVertexCounts) points += size_t(std::max(count, 0));
        std::string const &prefix = usdGen::UsdGenAuthoredPlanePrefix();
        for (UsdGeomPrimvar const &pv : UsdGeomPrimvarsAPI(prim).GetPrimvarsWithValues()) {
            std::string const name = pv.GetPrimvarName().GetString();
            if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0)
                continue;
            VtValue value;
            if (!pv.ComputeFlattened(&value, UsdTimeCode(time))) continue;
            usdGen::UsdGenAuthoredPlaneDesc plane;
            if (usdGen::ConvertAuthoredPlane(TfToken(name.substr(prefix.size())), value,
                                             pv.GetInterpolation(),
                                             out->curveVertexCounts.size(), points, &plane))
                out->authoredPlanes.push_back(std::move(plane));
        }
    }

    TfToken curveRole;
    _GetPrimvarTyped(prim, TfToken("usdGen:role"), UsdTimeCode::Default(),
                     &curveRole);
    if (curveRole.IsEmpty()) {
        _GetToken(prim, "usdGen:role", &curveRole);
    }
    if (!curveRole.IsEmpty()) {
        out->curveRole = curveRole;
    }

    // frozenEpoch: constant string primvar "usdgen1:sha1:..." (S42).
    VtValue epoch;
    _GetPrimvar(prim, TfToken("usdGen:frozenEpoch"), UsdTimeCode::Default(),
                &epoch);
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



// Oracle twin of the Hydra builder's _HCollectInputTarget.
void
_CollectInputTarget(UsdStageRefPtr const &stage, SdfPath const &path, int depth,
                    SdfPathVector *geometries, SdfPathVector *maps)
{
    UsdPrim const prim = stage->GetPrimAtPath(path);
    if (!prim) return;
    if (UsdGenIsMapTypeName(prim.GetPrimTypeInfo().GetTypeName())) {
        maps->push_back(path);
        return;
    }
    if (prim.IsA<UsdGeomMesh>() || prim.IsA<UsdGeomBasisCurves>() ||
        prim.IsA<UsdGeomPoints>()) {
        geometries->push_back(path);
        return;
    }
    // A Mesh's GeomSubset samples the mesh restricted to its faces (R15).
    if (prim.IsA<UsdGeomSubset>()) {
        if (!_SurfaceMesh(stage, path).IsEmpty()) geometries->push_back(path);
        return;
    }
    if (depth > 64) return;
    for (UsdPrim const &child : prim.GetChildren())
        _CollectInputTarget(stage, child.GetPath(), depth + 1, geometries, maps);
}

// Oracle twin of _HBuildGeometry: rest is what the RestAPI/CurveAPI adapters
// publish when applied (authored primvars:rest, else Default-time points),
// otherwise an authored primvars:rest, otherwise empty. A face GeomSubset
// samples its parent Mesh restricted to the faces it names (R15).
void
_BuildGeometry(UsdStageRefPtr const &stage, SdfPath const &path, double time,
               usdGen::UsdGenGeometryDesc *out, std::vector<std::string> *errors)
{
    out->path = path;
    UsdPrim const prim = stage->GetPrimAtPath(path);
    if (!prim) return;
    if (UsdGeomSubset const subset{prim}) {
        SdfPath const meshPath = path.GetParentPath();
        _BuildGeometry(stage, meshPath, time, out, errors);
        out->path = path;
        out->isSubset = true;
        TfToken elementType;
        subset.GetElementTypeAttr().Get(&elementType);
        if (elementType != UsdGeomTokens->face) {
            errors->push_back(
                UsdGenSubsetElementTypeError(path, elementType.GetString()));
        } else {
            VtIntArray indices;
            _GetTyped(subset.GetIndicesAttr(), UsdTimeCode(time), &indices);
            std::vector<int> faces;
            UsdGenAppendSubsetFaces(indices, out->counts.size(), path, meshPath,
                                    &faces, errors);
            out->subsetFaces = UsdGenFinalizeSubsetFaces(std::move(faces), false, 0);
        }
        out->generation = UsdGenGeometryContentHash(*out);
        return;
    }
    out->worldMatrix =
        UsdGeomImageable(prim).ComputeLocalToWorldTransform(UsdTimeCode(time));
    UsdGeomPointBased const pointBased(prim);
    if (pointBased) _GetVec3fArray(pointBased.GetPointsAttr(), UsdTimeCode(time), &out->points);
    UsdGeomPrimvar const rest = UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken("rest"));
    bool const authoredRest = rest && rest.GetAttr().GetResolveInfo().HasAuthoredValueOpinion();
    auto readRest = [&](bool adapterApplied) {
        if (authoredRest) {
            _GetPrimvarTyped(prim, TfToken("rest"), UsdTimeCode::Default(), &out->rest);
        } else if (adapterApplied && pointBased) {
            _GetVec3fArray(pointBased.GetPointsAttr(), UsdTimeCode::Default(), &out->rest);
        }
    };
    if (UsdGeomMesh const mesh{prim}) {
        out->kind = usdGen::UsdGenGeometryKind::Mesh;
        _GetTyped(mesh.GetFaceVertexCountsAttr(), UsdTimeCode(time), &out->counts);
        _GetTyped(mesh.GetFaceVertexIndicesAttr(), UsdTimeCode(time), &out->indices);
        readRest(prim.HasAPI(TfToken("UsdGenRestAPI")));
    } else if (UsdGeomBasisCurves const curves{prim}) {
        out->kind = usdGen::UsdGenGeometryKind::Curves;
        _GetTyped(curves.GetCurveVertexCountsAttr(), UsdTimeCode(time), &out->counts);
        readRest(prim.HasAPI(TfToken("UsdGenCurveAPI")));
        _GetPrimvarTyped(prim, TfToken("usdGen:curveId"), UsdTimeCode(time), &out->ids);
    } else {
        out->kind = usdGen::UsdGenGeometryKind::Points;
        readRest(false);
        _GetPrimvarTyped(prim, TfToken("normals"), UsdTimeCode(time), &out->normals);
    }
    out->generation = UsdGenGeometryContentHash(*out);
}

}  // namespace

usdGen::UsdGenGraphDesc
BuildGraphDescFromStage(
    UsdStageRefPtr const &stage,
    SdfPath const &descriptionPath,
    UsdGenGraphDescBuildOptions const &options)
{
    UsdGenGraphDesc desc;
    desc.description = descriptionPath;
    desc.time = options.time;
    if (!stage) {
        return desc;
    }
    UsdPrim const descPrim = stage->GetPrimAtPath(descriptionPath);
    if (!descPrim) {
        TF_CODING_ERROR("usdGen: description prim %s does not exist.",
                        descriptionPath.GetText());
        return desc;
    }

    // Stage metadata is the oracle counterpart of the live Hydra description
    // source.  Read the raw value so malformed/non-positive rates become
    // validation errors rather than silently using a default.
    VtValue timeCodesPerSecond;
    if (stage->HasAuthoredMetadata(TfToken("timeCodesPerSecond"))) {
        stage->GetMetadata(TfToken("timeCodesPerSecond"),
                           &timeCodesPerSecond);
    } else {
        // GetTimeCodesPerSecond applies USD's framesPerSecond fallback and
        // the schema's default 24 when no explicit timeCodesPerSecond exists.
        timeCodesPerSecond = VtValue(stage->GetTimeCodesPerSecond());
    }
    if (timeCodesPerSecond.IsEmpty() ||
        !timeCodesPerSecond.IsHolding<double>()) {
        desc.validationErrors.push_back(
            desc.description.GetString() +
            ": timeCodesPerSecond has wrong authored type");
    } else {
        double const rate = timeCodesPerSecond.UncheckedGet<double>();
        if (!std::isfinite(rate) || rate <= 0.0) {
            desc.validationErrors.push_back(
                desc.description.GetString() +
                ": timeCodesPerSecond must be finite and positive");
        } else {
            desc.timeCodesPerSecond = rate;
        }
    }
    double const time = options.time;

    SdfPathVector const operatorOrder = _StageOperatorOrder(descPrim);
    if (operatorOrder.empty()) {
        TF_CODING_ERROR("usdGen: %s has no Ops operators.",
                        descriptionPath.GetText());
        return desc;
    }
    desc.terminal = operatorOrder.back();
    std::unordered_map<std::string, UsdGenRole> curveRoles;

    // Expressions are providers, never stack nodes.  Preserve native output
    // type and source verbatim; compiler policy decides whether evaluation is
    // available, rather than silently discarding a connected authoring edit.
    if (UsdPrim const expressions = descPrim.GetChild(TfToken("Expressions"))) {
        for (UsdPrim const &prim : expressions.GetChildren()) {
            if (prim.GetPrimTypeInfo().GetTypeName() != TfToken("UsdGenExpression")) continue;
            usdGen::UsdGenExpressionDesc e;
            e.path = prim.GetPath();
            _GetTyped(prim.GetAttribute(TfToken("usdGen:expr:source")),
                      UsdTimeCode::Default(), &e.source);
            for (UsdAttribute const &a : prim.GetAttributes()) {
                std::string const n = a.GetName().GetString();
                if (n.rfind("outputs:", 0) != 0) continue;
                usdGen::UsdGenExpressionOutputDesc output;
                output.name = TfToken(n.substr(8));
                output.nativeType = a.GetTypeName().GetAsToken();
                VtValue declaration;
                output.shape = _ExpressionShape(
                    a.GetTypeName(), a.Get(&declaration, UsdTimeCode::Default())
                        ? &declaration : nullptr);
                e.outputs.push_back(std::move(output));
            }
            for (UsdRelationship const &rel : prim.GetRelationships()) {
                std::string const n = rel.GetName().GetString();
                if (n.rfind("input:", 0) != 0 || n.size() <= 6) continue;
                usdGen::UsdGenExpressionInputDesc input;
                input.name = TfToken(n.substr(6));
                rel.GetForwardedTargets(&input.targets);
                e.inputs.push_back(std::move(input));
            }
            desc.expressions.push_back(std::move(e));
        }
    }

    // ---- nodes, composed reverse-sibling post-order ----------------------
    // Collider targets ride along keyed by node path; they append to the
    // Collide nodes' surfaces after surface inheritance below.
    std::map<std::string, SdfPathVector> colliderTargets;
    for (SdfPath const &operatorPath : operatorOrder) {
        UsdPrim const prim = stage->GetPrimAtPath(operatorPath);
        UsdGenNodeDesc node;
        node.path = prim.GetPath();
        TfToken type;
        _GetToken(prim, "usdGen:type", &type);
        node.type = type.IsEmpty() ? prim.GetPrimTypeInfo().GetTypeName() : type;
        _GetToken(prim, "usdGen:mode", &node.mode);
        bool enabled = true;
        _GetDedicated(prim.GetAttribute(TfToken("usdGen:enabled")),
                  UsdTimeCode::Default(), &enabled, &desc.validationErrors, node.path, "usdGen:enabled");
        node.enabled = enabled;
        _GetDedicated(prim.GetAttribute(TfToken("usdGen:seed")),
                  UsdTimeCode::Default(), &node.seed, &desc.validationErrors, node.path, "usdGen:seed");

        // usdGen:colliders (UsdGenCollide only) collects to a sidecar:
        // node.surfaces is overwritten by the description inheritance
        // below, so colliders append after it instead of bucketing here.
        SdfPathVector nodeColliders;
        bool const isCollide = node.type == TfToken("UsdGenCollide");
        for (UsdRelationship const &rel : prim.GetRelationships()) {
            std::string const name = rel.GetBaseName().GetString();
            // usdGen:part:curves nests one level deeper; match it by full
            // name so its bucketing cannot depend on how GetBaseName
            // strips a nested namespace.
            bool const isPartCurves =
                rel.GetName().GetString() == "usdGen:part:curves";
            bool const isDirectionSource =
                rel.GetName().GetString() == "usdGen:direction:source";
            // usdGen:frozen:curves nests like part:curves; match it by full
            // name for the same reason, and mark its targets Reference so
            // the compiler resolves them into Freeze's reference slot.
            bool const isFrozenCurves =
                rel.GetName().GetString() == "usdGen:frozen:curves";
            bool const isColliders =
                isCollide && rel.GetName().GetString() == "usdGen:colliders";
            bool const isRegionMap =
                (node.type == TfToken("UsdGenCurveSource") || node.type == TfToken("UsdGenDeform") ||
                 node.type == TfToken("UsdGenGuideInterpolate")) &&
                rel.GetName().GetString() == "usdGen:regionMap";
            // Reference-lane curve sets of the whole-strand operators: trim
            // blades (Trim), card spines and the tile strand library
            // (CardSource). Matched by full name like part:curves.
            std::string const fullName = rel.GetName().GetString();
            bool const isExtraReference =
                fullName == "usdGen:trim:curves" || fullName == "usdGen:cards" ||
                fullName == "usdGen:cardStrands";
            // Mask maps sampled at strand roots: Scatter's density masks and
            // Length's length masks. Any number of targets (multiplied).
            bool const isMaskMap =
                (node.type == TfToken("UsdGenScatter") && fullName == "usdGen:densityMap") ||
                (node.type == TfToken("UsdGenLength") && fullName == "usdGen:lengthMap");
            SdfPathVector *bucket = &node.references;
            if (isMaskMap) {
                SdfPathVector targets;
                rel.GetTargets(&targets);
                for (SdfPath const &target : targets) {
                    node.maps.push_back(target);
                    node.mapBindings.push_back({target, TfToken(fullName)});
                }
                continue;
            } else if (isExtraReference) {
                SdfPathVector targets;
                rel.GetTargets(&targets);
                for (SdfPath const &target : targets) {
                    node.curves.push_back(target);
                    curveRoles[target.GetString()] = UsdGenRole::Reference;
                }
                continue;
            } else if (name == "input") {
                bucket = &node.inputs;
            } else if (name == "references" || name == "reference") {
                bucket = &node.references;
            } else if (name == "guides" || name == "curves" ||
                       name == "frozen:curves" || isPartCurves ||
                       isDirectionSource) {
                // node.curves, not node.references: the compiler resolves
                // every node.references path as a curve set, while curveRefs
                // also admit reference-lane operator sources (guides rule).
                bucket = &node.curves;
            } else if (isColliders) {
                bucket = &nodeColliders;
            } else if (isRegionMap) {
                SdfPathVector targets;
                rel.GetTargets(&targets);
                // Schema relationships are present with zero targets when
                // unauthored. Empty means optional/unset; surfaceCage still
                // rejects a missing map at operator admission. More than one
                // target is a validation error.
                if (targets.size() > 1) {
                    desc.validationErrors.push_back(
                        node.path.GetString() +
                        ": usdGen:regionMap requires exactly one target");
                } else if (targets.size() == 1) {
                    node.maps.push_back(targets.front());
                    node.mapBindings.push_back(
                        {targets.front(), TfToken("usdGen:regionMap")});
                }
                continue;
            } else {
                continue;  // not a graph edge (base-name match only)
            }
            SdfPathVector relTargets;
            rel.GetTargets(&relTargets);
            for (SdfPath const &t : relTargets) {
                bucket->push_back(t);
                if (name == "guides" || isPartCurves || isDirectionSource || isFrozenCurves) {
                    curveRoles[t.GetString()] = UsdGenRole::Reference;
                }
            }
        }

        node.inputs.clear();
        if (!desc.nodes.empty()) {
            node.inputs.push_back(desc.nodes.back().path);
        }

        _PullParams(prim, time, &node.params);
        for (UsdAttribute const &a : prim.GetAttributes()) {
            SdfPathVector connections;
            a.GetConnections(&connections);
            for (SdfPath const &c : connections) {
                usdGen::UsdGenExpressionBinding binding;
                binding.expression = c.GetPrimPath();
                // A connection may name the expression prim instead of one of
                // its outputs; resolve both spellings to the same binding.
                binding.output =
                    UsdGenResolveExpressionConnectionOutput(prim, c);
                binding.destination = a.GetName();
                binding.nativeType = a.GetTypeName().GetAsToken();
                binding.domain = _ExpressionDomain(a);
                VtValue literal;
                bool const hasLiteral = a.Get(&literal, UsdTimeCode(time));
                binding.destinationShape = _ExpressionShape(
                    a.GetTypeName(), hasLiteral ? &literal : nullptr);
                binding.literal = std::move(literal);
                node.expressionBindings.push_back(std::move(binding));
            }
        }
        if (!nodeColliders.empty())
            colliderTargets[operatorPath.GetString()] = nodeColliders;
        desc.nodes.push_back(std::move(node));
    }

    // 02 §2: the bound surface is the Description's. There is no per-operator
    // override, so every node sees the same target set. Targets on the bound
    // target's mesh union into it (02 §2.20 rule 4).
    std::map<SdfPath, SdfPathVector> surfaceUnions;
    {
        SdfPathVector descSurfaces;
        if (UsdRelationship rel = descPrim.GetRelationship(
                TfToken("usdGen:surface"))) {
            rel.GetTargets(&descSurfaces);
        }
        SdfPathVector unioned = UsdGenUnionSurfaceTargets(&descSurfaces,
            [&](SdfPath const &p) { return _SurfaceMesh(stage, p); });
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

    // Returns the pool slot; ensures desc.nodes references resolve. Nodes
    // store SdfPath vectors, so the pool membership itself is the contract
    // (the engine re-derives ids in SetGraphDesc; 03 §2.5).
    auto surfaceFor = [&](SdfPath const &p) {
        if (surfaceIndex.count(p.GetString())) {
            return;
        }
        UsdGenSurfaceDesc surface;
        surface.path = p;
        surface.id = usdGen::UsdGenSurfaceId(desc.surfaces.size());
        surfaceIndex.emplace(p.GetString(), desc.surfaces.size());
        desc.surfaces.push_back(std::move(surface));
        UsdPrim prim = stage->GetPrimAtPath(p);
        if (prim && prim.IsA<UsdGeomSubset>()) {
            // R15: the subset's desc carries its parent Mesh's geometry and
            // the PARENT-mesh face indices it selects.
            auto const u = surfaceUnions.find(p);
            _BuildSubsetSurface(stage, p,
                                u == surfaceUnions.end() ? SdfPathVector() : u->second,
                                time, &desc.surfaces[surfaceIndex[p.GetString()]],
                                &desc.validationErrors);
        } else {
            _BuildSurface(stage, p, time,
                          &desc.surfaces[surfaceIndex[p.GetString()]]);
        }
    };

    auto curveFor = [&](SdfPath const &p, UsdGenRole role) {
        auto it = curveIndex.find(p.GetString());
        if (it == curveIndex.end()) {
            UsdGenCurveSetDesc cs;
            _BuildCurveSet(stage, p, role, time, &cs);
            it = curveIndex.emplace(p.GetString(), desc.curveSets.size())
                     .first;
            desc.curveSets.push_back(std::move(cs));
        } else if (role == UsdGenRole::Reference) {
            // Reference is the stronger lane claim (I3).
            desc.curveSets[it->second].role = UsdGenRole::Reference;
        }
    };

    auto mapFor = [&](SdfPath const &p) {
        if (mapIndex.count(p.GetString())) {
            return;
        }
        usdGen::UsdGenMapDesc map;
        map.path = p;
        if (UsdPrim prim = stage->GetPrimAtPath(p)) {
            TfToken type;
            _GetToken(prim, "usdGen:type", &type);
            map.type = type.IsEmpty() ? prim.GetPrimTypeInfo().GetTypeName() : type;
            SdfAssetPath asset;
            if (_GetTyped(prim.GetAttribute(TfToken("usdGen:map:file")),
                          UsdTimeCode::Default(), &asset)) {
                // Stage-free by value (S13): the RESOLVED path travels.
                map.resolvedAssetPath = asset.GetResolvedPath();
                if (map.resolvedAssetPath.empty()) {
                    map.resolvedAssetPath = asset.GetAssetPath();
                }
            }
            _PullParams(prim, time, &map.params);
            if (map.type == TfToken("UsdGenPaintMap")) {
                _CapturePaintMap(stage, prim, time, &map,
                                 &desc.validationErrors);
            }
        }
        mapIndex.emplace(p.GetString(), desc.maps.size());
        desc.maps.push_back(std::move(map));
    };

    std::map<std::string, size_t> geometryIndex;
    for (usdGen::UsdGenExpressionDesc &expression : desc.expressions) {
        for (usdGen::UsdGenExpressionInputDesc &in : expression.inputs) {
            for (SdfPath const &target : in.targets) {
                size_t const before = in.geometries.size() + in.maps.size();
                _CollectInputTarget(stage, target, 0, &in.geometries, &in.maps);
                if (in.geometries.size() + in.maps.size() == before)
                    desc.validationErrors.push_back(expression.path.GetString() + ": input:" +
                        in.name.GetString() + " target " + target.GetString() +
                        " is not a mesh, curves, points or map prim, and contains none");
            }
            for (SdfPath const &g : in.geometries) {
                if (geometryIndex.count(g.GetString())) continue;
                usdGen::UsdGenGeometryDesc geometry;
                _BuildGeometry(stage, g, time, &geometry, &desc.validationErrors);
                geometryIndex.emplace(g.GetString(), desc.geometries.size());
                desc.geometries.push_back(std::move(geometry));
            }
            for (SdfPath const &m : in.maps) mapFor(m);
        }
    }

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
    _GetDedicated(descPrim.GetAttribute(TfToken("usdGen:width:default")),
              UsdTimeCode(options.time), &desc.defaultWidth, &desc.validationErrors, desc.description, "usdGen:width:default");
    _GetDedicated(descPrim.GetAttribute(TfToken("usdGen:tileTarget")),
              UsdTimeCode::Default(), &desc.tileTarget, &desc.validationErrors, desc.description, "usdGen:tileTarget");
    _GetToken(descPrim, "usdGen:curve:basis", &desc.curveBasis);

    UsdGenLookDesc &look = desc.look;
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:rootColor")),
              UsdTimeCode::Default(), &look.rootColor);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:tipColor")),
              UsdTimeCode::Default(), &look.tipColor);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:colorRamp:colors")),
              UsdTimeCode::Default(), &look.rampColors);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:colorRamp:positions")),
              UsdTimeCode::Default(), &look.rampPositions);
    _GetToken(descPrim, "usdGen:look:colorRamp:interpolation",
              &look.rampInterpolation);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:rampExponent")),
              UsdTimeCode::Default(), &look.rampExponent);
    _GetToken(descPrim, "usdGen:look:bakeMode", &look.bakeMode);
    _GetToken(descPrim, "usdGen:look:bakeTarget", &look.bakeTarget);
    _GetToken(descPrim, "usdGen:look:bakePrimvar", &look.bakePrimvar);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:hueJitter")),
              UsdTimeCode::Default(), &look.hueJitter);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:valueJitter")),
              UsdTimeCode::Default(), &look.valueJitter);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:look:jitterSeed")),
              UsdTimeCode::Default(), &look.jitterSeed);

    // usdGen:preview:* (viewport value preview). The source stays unresolved
    // except that a map target joins the map pool, so the cooker can read it.
    usdGen::UsdGenPreviewDesc &preview = desc.preview;
    if (UsdRelationship const rel =
            descPrim.GetRelationship(TfToken("usdGen:preview:source"))) {
        SdfPathVector targets;
        rel.GetForwardedTargets(&targets);
        if (!targets.empty()) preview.source = targets.front();
    }
    _GetToken(descPrim, "usdGen:preview:colorMap", &preview.colorMap);
    _GetTyped(descPrim.GetAttribute(TfToken("usdGen:preview:range")),
              UsdTimeCode::Default(), &preview.range);
    _GetToken(descPrim, "usdGen:preview:evaluation", &preview.evaluation);
    _GetToken(descPrim, "usdGen:preview:shading", &preview.shading);
    if (preview.source.IsPrimPath()) {
        UsdPrim const target = stage->GetPrimAtPath(preview.source);
        if (target && UsdGenIsMapTypeName(target.GetPrimTypeInfo().GetTypeName()))
            mapFor(preview.source);
    }

    // purpose / visibility inherited by hand to every tile (C2): read the
    // description's own opinions; the engine copies them onto publications.
    // NOTE (2026-09-12): this read "usdGeom:purpose", which is not a USD
    // attribute name — the Imageable attribute is "purpose" — so the read
    // never matched. Fixed to "purpose" with one more subtlety: the schema
    // declares fallback `uniform token purpose = "default"`
    // (usdGeom/imageable.h), which UsdAttribute::Get returns when nothing is
    // authored — so only an AUTHORED opinion inherits. An unauthored purpose
    // stays EMPTY: the publisher omits the purpose container then, which
    // Hydra resolves to the geometry render tag
    // (HdSceneIndexAdapterSceneDelegate::GetRenderTag falls back to geometry
    // only when the container is absent or empty). Publishing
    // purpose="default" instead is FATAL: "default" is not a render tag, so
    // the tile matches no collection and Storm never syncs it (2026-09-12:
    // 49 published tiles, itemsDrawn == 1, zero HdStBasisCurves sync lines).
    // Visibility keeps its (valid) fallback: "inherited" is a real Hydra
    // visibility value.
    if (UsdAttribute const purposeAttr =
            descPrim.GetAttribute(TfToken("purpose"))) {
        if (purposeAttr.GetResolveInfo().HasAuthoredValueOpinion()) {
            _GetToken(descPrim, "purpose", &desc.purpose);
        }
    }
    TfToken visibility;
    _GetToken(descPrim, "visibility", &visibility);
    desc.visibility = visibility.IsEmpty() ? TfToken("inherited") : visibility;

    UsdShadeMaterialBindingAPI const binding(descPrim);
    if (UsdShadeMaterial material = binding.ComputeBoundMaterial()) {
        desc.materialPath = material.GetPath();
    }

    desc.xformMatrix = UsdGeomImageable(descPrim).ComputeLocalToWorldTransform(
        UsdTimeCode(time));

    desc.executionBackend = _ResolveExecutionBackend(descPrim.GetParent());

    ResolveUsdGenImageMaps(&desc);
    UsdGenFinalizeInputGenerations(&desc);
    return desc;
}


}  // namespace usdGenImaging
