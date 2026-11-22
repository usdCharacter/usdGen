// usdGen imaging — UsdGenRestAPI data sourcing (02-schema.md §2.15, S12;
// 06-imaging.md §2.6).
//
// The RestAPI container publishes the REST surface of a bound Mesh under
// the `usdGen` container of that prim's imaging data source:
//   usdGen/rest/points              VtVec3fArray
//   usdGen/rest/faceVertexCounts    VtIntArray
//   usdGen/rest/faceVertexIndices   VtIntArray
//   usdGen/rest/st                  VtVec2fArray (primary uv set, may be empty)
//   usdGen/rest/normals             VtVec3fArray (Mesh normals at Default)
//   usdGen/rest/normalsInterpolation TfToken
//
// Rest points are `primvars:rest` when authored (the host application convention,
// S12 — it passes through the chain untouched), otherwise the deformed
// `points` opinion at UsdTimeCode::Default(). Live leaves re-read Default
// time after edits, including through cached handles. They never register
// as time-varying; editing rest still emits its dedicated dirty locator.
// Capture-on-first-cook is rejected (S12): the first drawn frame is not
// necessarily the rest frame.
//
// UsdGenRestAPI may be applied only to the parent Mesh, never to a
// GeomSubset (ADR §9 R15): a subset carries only face indices into the
// parent mesh. A geomSubset target is a hard diagnostic, reported once per
// prim (06 §9 diagnostic 2), and no container is published.
#include "usdGenImaging/usdGenRestApiDataSource.h"

#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/ts/spline.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/usd/usdGeom/primvar.h"
#include "pxr/usd/usdGeom/primvarsAPI.h"
#include "pxr/usd/usdGeom/subset.h"

#include "usdGenImaging/usdGenTokens.h"

#include <atomic>
#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

namespace {

class _LiveRestValueDataSource final : public HdSampledDataSource
{
public:
    HD_DECLARE_DATASOURCE(_LiveRestValueDataSource);
    enum Leaf { Points, Counts, Indices, St, Normals, NormalsInterpolation };
    _LiveRestValueDataSource(UsdPrim const& prim, Leaf leaf) : _prim(prim), _leaf(leaf) {}

    VtValue GetValue(Time) override {
        UsdGeomMesh mesh(_prim);
        if (!mesh) return VtValue();
        if (_leaf == Points) {
            TfToken source("default");
            if (auto attr = _prim.GetAttribute(TfToken("usdGen:rest:source")))
                attr.Get(&source, UsdTimeCode::Default());
            // Alternate rest assets/named primvars are not implemented here.
            // An absent value makes descriptor validation fail closed instead
            // of silently substituting a different binding surface.
            if (source != TfToken("default")) return VtValue();
            UsdGeomPrimvar rest = UsdGeomPrimvarsAPI(_prim).GetPrimvar(TfToken("rest"));
            UsdAttribute attr = rest ? rest.GetAttr() : UsdAttribute();
            // ResolveInfo distinguishes an authored empty opinion from no
            // opinion; the former must not fall through to points.
            if (attr && attr.GetResolveInfo().HasAuthoredValueOpinion()) {
                VtValue value;
                return attr.Get(&value, UsdTimeCode::Default()) &&
                    value.IsHolding<VtVec3fArray>() ? value : VtValue(VtVec3fArray());
            }
            VtValue value;
            return mesh.GetPointsAttr().Get(&value, UsdTimeCode::Default()) &&
                value.IsHolding<VtVec3fArray>() ? value : VtValue(VtVec3fArray());
        }
        if (_leaf == St) {
            UsdGeomPrimvar st = UsdGeomPrimvarsAPI(_prim).GetPrimvar(TfToken("st"));
            VtValue value;
            return st && st.GetAttr().Get(&value, UsdTimeCode::Default()) &&
                value.IsHolding<VtVec2fArray>() ? value : VtValue(VtVec2fArray());
        }
        if (_leaf == Normals) {
            // Preserve a malformed value for the Hydra builder to reject;
            // returning an empty normal array here would erase Invalid into
            // the valid geometric-fallback None domain.
            VtValue value;
            return mesh.GetNormalsAttr().Get(&value, UsdTimeCode::Default())
                ? value : VtValue();
        }
        if (_leaf == NormalsInterpolation) {
            return VtValue(mesh.GetNormalsInterpolation());
        }
        VtValue value;
        UsdAttribute attr = _leaf == Counts ? mesh.GetFaceVertexCountsAttr() : mesh.GetFaceVertexIndicesAttr();
        return attr.Get(&value, UsdTimeCode::Default()) &&
            value.IsHolding<VtIntArray>() ? value : VtValue(VtIntArray());
    }
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
private:
    UsdPrim _prim; Leaf _leaf;
};

// One hard diagnostic per prim, not per pull (06 §9: "emitted once per groom
// per compile"; a scene-index pull happens many times a frame).
// Lock-free warn-once: a fixed open-addressed set of path hashes claimed by
// CAS. Pulls run on any render thread; a full table still warns.
struct _WarnedPaths {
    static constexpr size_t kSlots = 1024;
    std::atomic<uint64_t> slots[kSlots] = {};
};

bool
_WarnOnce(_WarnedPaths *warned, SdfPath const &path, char const *msg)
{
    uint64_t const hash = uint64_t(path.GetHash()) | 1u;   // 0 marks empty
    for (size_t probe = 0; probe < _WarnedPaths::kSlots; ++probe) {
        std::atomic<uint64_t> &slot =
            warned->slots[(hash + probe) & (_WarnedPaths::kSlots - 1)];
        uint64_t seen = slot.load(std::memory_order_acquire);
        if (seen == hash) return false;
        if (seen == 0 && slot.compare_exchange_strong(
                seen, hash, std::memory_order_acq_rel)) break;
        if (seen == hash) return false;
    }
    TF_WARN("usdGen: %s", msg);
    return true;
}

}  // namespace

HdContainerDataSourceHandle UsdGenRestApiContainerFactory(
    UsdPrim const &prim,
    UsdImagingDataSourceStageGlobals const &globals)
{
    TF_UNUSED(globals);

    static _WarnedPaths _warned;

    // R15: a GeomSubset target is the sibling case of hard diagnostic 2.
    if (prim.IsA<UsdGeomSubset>()) {
        _WarnOnce(&_warned, prim.GetPath(),
            ("usdGen: UsdGenRestAPI applied to geomSubset " +
             std::string(prim.GetPath().GetText()) +
             " - apply it to the parent Mesh instead (ADR §9 R15).").c_str());
        return nullptr;
    }

    if (!prim.IsA<UsdGeomMesh>()) {
        std::string text = "usdGen: UsdGenRestAPI applied to '" +
                           prim.GetPrimTypeInfo().GetTypeName().GetString() +
                           "' at " + prim.GetPath().GetText() +
                           " — expected a UsdGenDescription-bound Mesh surface.";
        _WarnOnce(&_warned, prim.GetPath(), text.c_str());
        return nullptr;
    }

    HdSampledDataSourceHandle pointsSrc = _LiveRestValueDataSource::New(prim, _LiveRestValueDataSource::Points);
    HdSampledDataSourceHandle countsSrc = _LiveRestValueDataSource::New(prim, _LiveRestValueDataSource::Counts);
    HdSampledDataSourceHandle indicesSrc = _LiveRestValueDataSource::New(prim, _LiveRestValueDataSource::Indices);
    HdSampledDataSourceHandle uvSrc = _LiveRestValueDataSource::New(prim, _LiveRestValueDataSource::St);
    HdSampledDataSourceHandle normalsSrc = _LiveRestValueDataSource::New(prim, _LiveRestValueDataSource::Normals);
    HdSampledDataSourceHandle normalsInterpolationSrc = _LiveRestValueDataSource::New(
        prim, _LiveRestValueDataSource::NormalsInterpolation);

    // Build the retained tree by hand: usdGen/rest/{points,faceVertexCounts,
    // faceVertexIndices,st,normals,normalsInterpolation}. Only containers are retained: live leaves
    // sample Default time without registering per-frame variability.
    TfTokenVector restNames{TfToken("points"), TfToken("faceVertexCounts"), TfToken("faceVertexIndices"),
                            TfToken("st"), TfToken("normals"), TfToken("normalsInterpolation")};
    HdDataSourceBaseHandle restValues[6] = {
        pointsSrc, countsSrc, indicesSrc, uvSrc, normalsSrc, normalsInterpolationSrc};
    HdRetainedContainerDataSourceHandle restContainer =
        HdRetainedContainerDataSource::New(6, restNames.data(), restValues);

    // Contract shape usdGen/rest/* (header + 02 §2.15): the four leaves
    // sit under an intermediate `rest` container, itself under `usdGen`.
    // (The factory previously returned usdGen/{leaves} directly — one level
    // short of the documented tree; the invalidation map already emitted
    // usdGen/rest/*.)
    TfToken const restTok("rest");
    HdDataSourceBaseHandle midValues[1] = { restContainer };
    HdRetainedContainerDataSourceHandle mid =
        HdRetainedContainerDataSource::New(1, &restTok, midValues);
    HdDataSourceBaseHandle rootValues[1] = { mid };
    return HdRetainedContainerDataSource::New(
        1, &UsdGenContainerToken(), rootValues);
}

HdDataSourceLocatorSet UsdGenRestApiInvalidateMapping(
    UsdPrim const &prim,
    TfTokenVector const &properties,
    UsdImagingPropertyInvalidationType invalidationType)
{
    TF_UNUSED(prim);
    TF_UNUSED(invalidationType);

    HdDataSourceLocatorSet result;
    static HdDataSourceLocator const points{
        TfToken("usdGen"), TfToken("rest"), TfToken("points")};
    static HdDataSourceLocator const fvc{
        TfToken("usdGen"), TfToken("rest"), TfToken("faceVertexCounts")};
    static HdDataSourceLocator const fvi{
        TfToken("usdGen"), TfToken("rest"), TfToken("faceVertexIndices")};
    static HdDataSourceLocator const st{
        TfToken("usdGen"), TfToken("rest"), TfToken("st")};
    static HdDataSourceLocator const normals{
        TfToken("usdGen"), TfToken("rest"), TfToken("normals")};
    static HdDataSourceLocator const normalsInterpolation{
        TfToken("usdGen"), TfToken("rest"), TfToken("normalsInterpolation")};

    for (TfToken const &property : properties) {
        const std::string p = property.GetString();
        if (p == "points" || p == "primvars:rest") {
            result.insert(points);
        } else if (p == "faceVertexCounts") {
            result.insert(fvc);
        } else if (p == "faceVertexIndices") {
            result.insert(fvi);
        } else if (p == "st" || p == "primvars:st") {
            result.insert(st);
        } else if (p == "normals") {
            result.insert(normals);
            result.insert(normalsInterpolation);
        } else if (p == "normalsInterpolation") {
            result.insert(normalsInterpolation);
        } else if (p.rfind("usdGen:rest:", 0) == 0) {
            // The API's own authored knobs (source/primvar/file, S12) all
            // reach the evaluator through the points leaf; the file asset
            // path additionally dirty-points via the asset dependency.
            result.insert(points);
        }
    }
    return result;
}

}  // namespace usdGenImaging
