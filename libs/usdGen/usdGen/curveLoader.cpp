#include "usdGen/curveLoader.h"
#include "usdGen/curveRootCapture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

bool Fail(UsdGenDiagnostics *diag, std::string const &message)
{
    if (diag) diag->Error("UsdGenCurveSource: " + message);
    return false;
}
void Warn(UsdGenDiagnostics *diag, std::string const &message)
{
    if (diag) diag->Warn("UsdGenCurveSource: " + message);
}

// Every C3 validation is reported against the actual source prim.  Helpers
// can stay small and return early while this scope appends that indispensable
// context uniformly to both warnings and errors.
struct SourceDiagnosticContext {
    UsdGenDiagnostics *diag = nullptr;
    size_t errors = 0, warnings = 0;
    std::string source;
    SourceDiagnosticContext(UsdGenDiagnostics *d, SdfPath const &path)
        : diag(d), source(path.IsEmpty() ? std::string("<empty>") : path.GetString())
    {
        if (diag) { errors = diag->errors.size(); warnings = diag->warnings.size(); }
    }
    ~SourceDiagnosticContext()
    {
        if (!diag) return;
        std::string const prefix = "C3 source '" + source + "': ";
        for (size_t i = errors; i < diag->errors.size(); ++i)
            diag->errors[i] = prefix + diag->errors[i];
        for (size_t i = warnings; i < diag->warnings.size(); ++i)
            diag->warnings[i] = prefix + diag->warnings[i];
    }
};

bool Finite(GfVec3f const &v)
{
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
bool Finite(GfMatrix4d const &m)
{
    for (int r = 0; r != 4; ++r)
        for (int c = 0; c != 4; ++c)
            if (!std::isfinite(m[r][c])) return false;
    return true;
}

bool ReadBool(UsdGenParamView const *params, TfToken const &name,
              bool fallback, bool *out, UsdGenDiagnostics *diag)
{
    if (!params) { *out = fallback; return true; }
    VtValue const value = params->GetVtValue(name, VtValue());
    if (value.IsEmpty()) { *out = fallback; return true; }
    if (!value.IsHolding<bool>())
        return Fail(diag, "parameter '" + name.GetString() + "' must be bool");
    *out = value.UncheckedGet<bool>();
    return true;
}

bool ReadInt(UsdGenParamView const *params, TfToken const &name,
             int fallback, int *out, UsdGenDiagnostics *diag)
{
    if (!params) { *out = fallback; return true; }
    VtValue const value = params->GetVtValue(name, VtValue());
    if (value.IsEmpty()) { *out = fallback; return true; }
    if (!value.IsHolding<int>())
        return Fail(diag, "parameter '" + name.GetString() + "' must be int");
    *out = value.UncheckedGet<int>();
    return true;
}

bool ReadToken(UsdGenParamView const *params, TfToken const &name,
               TfToken const &fallback, TfToken *out, UsdGenDiagnostics *diag)
{
    if (!params) { *out = fallback; return true; }
    VtValue const value = params->GetVtValue(name, VtValue());
    if (value.IsEmpty()) { *out = fallback; return true; }
    if (!value.IsHolding<TfToken>())
        return Fail(diag, "parameter '" + name.GetString() + "' must be token");
    *out = value.UncheckedGet<TfToken>();
    return true;
}

bool ReadString(UsdGenParamView const *params, TfToken const &name,
                std::string *out, UsdGenDiagnostics *diag)
{
    out->clear();
    if (!params) return true;
    VtValue const value = params->GetVtValue(name, VtValue());
    if (value.IsEmpty()) return true;
    if (!value.IsHolding<std::string>())
        return Fail(diag, "parameter '" + name.GetString() + "' must be string");
    *out = value.UncheckedGet<std::string>();
    return true;
}

struct Options {
    bool useRest = true;
    int resampleTo = 0;
    TfToken idSource{"primvar"};
    TfToken staleAction{"warn"};
    TfToken rebind{"onError"};
    std::string expectedEpoch;
};

bool ReadOptions(UsdGenCaptureContext const &ctx, Options *out,
                 UsdGenDiagnostics *diag)
{
    UsdGenParamView const *params = ctx.params ? &*ctx.params : nullptr;
    if (!ReadBool(params, TfToken("useRest"), true, &out->useRest, diag) ||
        !ReadInt(params, TfToken("resampleTo"), 0, &out->resampleTo, diag) ||
        !ReadToken(params, TfToken("idSource"), TfToken("primvar"), &out->idSource, diag) ||
        !ReadToken(params, TfToken("staleAction"), TfToken("warn"), &out->staleAction, diag) ||
        !ReadToken(params, TfToken("rebind"), TfToken("onError"), &out->rebind, diag) ||
        !ReadString(params, TfToken("expectEpoch"), &out->expectedEpoch, diag))
        return false;
    // Pure-value builders normally map usdGen:useRest to `useRest`, but
    // preserve the legacy direct descriptor spelling when the canonical
    // locator is absent.  Both remain type-checked rather than defaulted.
    if (params && params->GetVtValue(TfToken("useRest"), VtValue()).IsEmpty() &&
        !ReadBool(params, TfToken("usdGen:useRest"), out->useRest, &out->useRest, diag))
        return false;
    if (out->resampleTo < 0 || out->resampleTo == 1)
        return Fail(diag, "resampleTo must be zero or at least two");
    if (out->idSource != TfToken("primvar") && out->idSource != TfToken("index"))
        return Fail(diag, "idSource must be 'primvar' or 'index'");
    if (out->staleAction != TfToken("warn") && out->staleAction != TfToken("ignore") &&
        out->staleAction != TfToken("block"))
        return Fail(diag, "staleAction must be 'warn', 'ignore', or 'block'");
    if (out->rebind != TfToken("never") && out->rebind != TfToken("onError") &&
        out->rebind != TfToken("always"))
        return Fail(diag, "rebind must be 'never', 'onError', or 'always'");
    return true;
}

bool BuildSourcePlanes(UsdGenCurveSetDesc const &source,
                       uint32_t totalCurves, uint32_t totalCvs,
                       std::vector<UsdGenPlane> *cv,
                       std::vector<UsdGenPlane> *curve,
                       UsdGenDiagnostics *diag)
{
    cv->clear(); curve->clear();
    try {
        for (UsdGenAuthoredPlaneDesc const &authored : source.authoredPlanes) {
            if (authored.name.IsEmpty() || authored.arity == 0 || authored.arity > 16)
                return Fail(diag, "authored plane has an invalid name or arity");
            size_t elements = 0;
            TfToken interpolation;
            std::vector<UsdGenPlane> *destination = cv;
            switch (authored.domain) {
            case UsdGenAuthoredPlaneDomain::Point:
                elements = totalCvs; interpolation = TfToken("vertex"); break;
            case UsdGenAuthoredPlaneDomain::Primitive:
                elements = totalCurves; interpolation = TfToken("uniform"); destination = curve; break;
            case UsdGenAuthoredPlaneDomain::Groom:
                elements = 1; interpolation = TfToken("constant"); destination = curve; break;
            default:
                return Fail(diag, "authored plane '" + authored.name.GetString() + "' has an invalid domain");
            }
            if (elements > std::numeric_limits<size_t>::max() / authored.arity)
                return Fail(diag, "authored plane '" + authored.name.GetString() + "' cardinality overflows");
            size_t const expected = elements * authored.arity;
            UsdGenPlane plane;
            plane.name = authored.name;
            plane.interpolation = interpolation;
            plane.arity = authored.arity;
            if (authored.type == UsdGenAuthoredPlaneType::Float32) {
                if (authored.floatValues.size() != expected || !authored.intValues.empty())
                    return Fail(diag, "authored plane '" + authored.name.GetString() + "' cardinality is invalid");
                for (float value : authored.floatValues)
                    if (!std::isfinite(value))
                        return Fail(diag, "authored plane '" + authored.name.GetString() + "' has non-finite values");
                plane.type = TfToken("float");
                plane.f = authored.floatValues;
            } else if (authored.type == UsdGenAuthoredPlaneType::Int32) {
                if (authored.intValues.size() != expected || !authored.floatValues.empty())
                    return Fail(diag, "authored plane '" + authored.name.GetString() + "' cardinality is invalid");
                plane.type = TfToken("int");
                plane.i = authored.intValues;
            } else {
                return Fail(diag, "authored plane '" + authored.name.GetString() + "' has an unsupported type");
            }
            destination->push_back(std::move(plane));
        }
        auto sortName = [](UsdGenPlane const &a, UsdGenPlane const &b) { return a.name < b.name; };
        std::sort(cv->begin(), cv->end(), sortName);
        std::sort(curve->begin(), curve->end(), sortName);
        for (size_t i = 1; i < cv->size(); ++i)
            if ((*cv)[i - 1].name == (*cv)[i].name)
                return Fail(diag, "duplicate authored vertex plane '" + (*cv)[i].name.GetString() + "'");
        for (size_t i = 1; i < curve->size(); ++i)
            if ((*curve)[i - 1].name == (*curve)[i].name)
                return Fail(diag, "duplicate authored curve plane '" + (*curve)[i].name.GetString() + "'");
        return true;
    } catch (...) {
        return Fail(diag, "authored plane materialization failed");
    }
}

// UsdGenCompactExtraPlanes deliberately accepts only ascending survivors: it
// models a filter/compaction, not a canonical stable-id permutation.  C3
// canonicalization is a permutation, so retain the same private-owner and
// typed-copy rules here without pretending it is a compaction.
bool ReorderSourcePlanes(UsdGenCurveBuffer const &source,
                         std::vector<size_t> const &order,
                         UsdGenCurveBuffer *target,
                         std::string *error)
{
    if (!target || source.totalCurves < order.size() || target->totalCurves != order.size() ||
        source.cvOffsets.size() != size_t(source.totalCurves) + 1 ||
        target->cvOffsets.size() != size_t(target->totalCurves) + 1) {
        if (error) *error = "invalid C3 plane permutation topology";
        return false;
    }
    try {
        std::vector<UsdGenPlane> cv;
        std::vector<UsdGenPlane> curve;
        cv.reserve(source.extraCv.size());
        curve.reserve(source.extraCurve.size());
        for (UsdGenPlane const &plane : source.extraCv) {
            UsdGenPlane copy = plane;
            size_t const elements = target->totalCvs;
            if (plane.type == TfToken("float")) {
                copy.f.assign(elements * plane.arity, 0.0f);
                copy.i.clear();
            } else if (plane.type == TfToken("int")) {
                copy.i.assign(elements * plane.arity, 0);
                copy.f.clear();
            } else {
                if (error) *error = "unsupported C3 vertex plane type";
                return false;
            }
            for (size_t outputCurve = 0; outputCurve != order.size(); ++outputCurve) {
                size_t const inputCurve = order[outputCurve];
                if (inputCurve >= source.totalCurves) {
                    if (error) *error = "invalid C3 plane permutation index";
                    return false;
                }
                uint32_t const inBegin = uint32_t(source.cvOffsets[inputCurve]);
                uint32_t const inEnd = uint32_t(source.cvOffsets[inputCurve + 1]);
                uint32_t const outBegin = uint32_t(target->cvOffsets[outputCurve]);
                if (inEnd - inBegin != uint32_t(target->cvOffsets[outputCurve + 1]) - outBegin) {
                    if (error) *error = "C3 plane permutation changed curve CV count";
                    return false;
                }
                size_t const n = size_t(inEnd - inBegin) * plane.arity;
                if (plane.type == TfToken("float"))
                    std::copy_n(plane.f.cdata() + size_t(inBegin) * plane.arity, n,
                                copy.f.begin() + size_t(outBegin) * plane.arity);
                else
                    std::copy_n(plane.i.cdata() + size_t(inBegin) * plane.arity, n,
                                copy.i.begin() + size_t(outBegin) * plane.arity);
            }
            cv.push_back(std::move(copy));
        }
        for (UsdGenPlane const &plane : source.extraCurve) {
            UsdGenPlane copy = plane;
            size_t const elements = plane.interpolation == TfToken("constant") ? 1 : target->totalCurves;
            if (plane.type == TfToken("float")) {
                copy.f.assign(elements * plane.arity, 0.0f);
                copy.i.clear();
            } else if (plane.type == TfToken("int")) {
                copy.i.assign(elements * plane.arity, 0);
                copy.f.clear();
            } else {
                if (error) *error = "unsupported C3 curve plane type";
                return false;
            }
            if (plane.interpolation == TfToken("constant")) {
                if (plane.type == TfToken("float")) copy.f = plane.f;
                else copy.i = plane.i;
            } else {
                for (size_t outputCurve = 0; outputCurve != order.size(); ++outputCurve) {
                    size_t const inputCurve = order[outputCurve];
                    if (plane.type == TfToken("float"))
                        std::copy_n(plane.f.cdata() + inputCurve * plane.arity, plane.arity,
                                    copy.f.begin() + outputCurve * plane.arity);
                    else
                        std::copy_n(plane.i.cdata() + inputCurve * plane.arity, plane.arity,
                                    copy.i.begin() + outputCurve * plane.arity);
                }
            }
            curve.push_back(std::move(copy));
        }
        target->extraCv = std::move(cv);
        target->extraCurve = std::move(curve);
        return true;
    } catch (...) {
        if (error) *error = "C3 plane permutation allocation failed";
        return false;
    }
}


UsdGenSurfaceDesc const *FindSurface(UsdGenCaptureContext const &ctx)
{
    if (!ctx.desc || !ctx.params || !ctx.params->node || ctx.params->node->surfaces.size() != 1)
        return nullptr;
    SdfPath const &path = ctx.params->node->surfaces.front();
    auto const it = std::find_if(ctx.desc->surfaces.begin(), ctx.desc->surfaces.end(),
        [&](UsdGenSurfaceDesc const &candidate) { return candidate.path == path; });
    return it == ctx.desc->surfaces.end() ? nullptr : &*it;
}

void PutFrame(GfMatrix4d const &frame, UsdGenCurveBuffer *out, size_t index)
{
    GfVec3d const t = frame.GetRow3(0);
    GfVec3d const b = frame.GetRow3(1);
    GfVec3d const n = frame.GetRow3(2);
    out->rootT[index] = GfVec3f(t);
    out->rootB[index] = GfVec3f(b);
    out->rootN[index] = GfVec3f(n);
}

} // namespace

bool UsdGenCurveLoader::Load(UsdGenCaptureContext const &ctx,
                             UsdGenCurveSetDesc const &source,
                             UsdGenCurveBuffer *out,
                             UsdGenDiagnostics *diag,
                             std::shared_ptr<const UsdGenRestSurfaceBindingCache> *bindingCache)
{
    SourceDiagnosticContext const sourceContext(diag, source.path);
    if (!out || !ctx.desc) return Fail(diag, "no output buffer or graph description");
    if (!std::isfinite(ctx.desc->defaultWidth) || ctx.desc->defaultWidth < 0.0f)
        return Fail(diag, "description defaultWidth must be finite and >= 0");
    Options options;
    if (!ReadOptions(ctx, &options, diag)) return false;
    if (source.path.IsEmpty() || source.role != UsdGenRole::Curves ||
        source.curveRole != TfToken("hair"))
        return Fail(diag, "CurveSource requires a C3 hair BasisCurves source; guide/reference lanes are not implemented here");
    if (!Finite(source.worldMatrix))
        return Fail(diag, "C3 worldMatrix contains non-finite values");
    // Linear polylines are accepted as-is (the basis is ignored for linear
    // curves); cubic curves must be bspline or catmullRom.
    bool const linear = source.type == TfToken("linear");
    if ((!linear && (source.type != TfToken("cubic") ||
                     (source.basis != TfToken("bspline") && source.basis != TfToken("catmullRom")))) ||
        source.wrap == TfToken("periodic"))
        return Fail(diag, "C3 requires linear, or cubic bspline/catmullRom, curves with non-periodic wrapping");
    if (source.wrap != TfToken("pinned") && source.wrap != TfToken("nonperiodic"))
        return Fail(diag, "C3 wrap must be 'pinned' or 'nonperiodic'");
    if (source.wrap == TfToken("nonperiodic") && !linear)
        Warn(diag, "nonperiodic C3 curves are promoted to pinned semantics without changing CVs");
    if (!options.expectedEpoch.empty() && options.expectedEpoch != source.frozenEpoch) {
        if (options.staleAction == TfToken("block"))
            return Fail(diag, "source frozenEpoch does not match expectEpoch");
        if (options.staleAction == TfToken("warn"))
            Warn(diag, "source frozenEpoch does not match expectEpoch; evaluating by request");
    }

    size_t total = 0;
    for (int count : source.curveVertexCounts) {
        if (count < 2 || total > std::numeric_limits<uint32_t>::max() - size_t(count))
            return Fail(diag, "C3 curveVertexCounts must contain only counts >= 2 within uint32 cardinality");
        total += size_t(count);
    }
    size_t const curves = source.curveVertexCounts.size();
    if (curves > std::numeric_limits<uint32_t>::max() ||
        total > size_t(std::numeric_limits<int>::max()) ||
        source.points.size() != total)
        return Fail(diag, "C3 points cardinality must equal sum(curveVertexCounts)");
    for (GfVec3f const &point : source.points)
        if (!Finite(point)) return Fail(diag, "C3 points contain non-finite values");
    // Validate an unselected rest snapshot as well.  It is still descriptor
    // data and must not become usable later through a retained COW capture.
    if (!source.rest.empty() && source.rest.size() != total)
        return Fail(diag, "C3 rest cardinality must equal sum(curveVertexCounts)");
    for (GfVec3f const &point : source.rest)
        if (!Finite(point)) return Fail(diag, "C3 rest contains non-finite values");
    if (options.useRest) {
        if (source.restFromCurrentPoints || source.rest.size() != total)
            return Fail(diag, "useRest requires an authored/default-time C3 rest snapshot");
    }
    if (source.widthsInterpolation != TfToken("vertex") &&
        source.widthsInterpolation != TfToken("constant"))
        return Fail(diag, "C3 widths interpolation must be vertex or constant");
    if (!source.widths.empty()) {
        size_t const expected = source.widthsInterpolation == TfToken("vertex") ? total : 1;
        if (source.widths.size() != expected)
            return Fail(diag, "C3 widths cardinality does not match its interpolation");
        for (float width : source.widths)
            if (!std::isfinite(width) || width < 0.0f)
                return Fail(diag, "C3 widths must be finite and >= 0");
    } else {
        Warn(diag, "C3 widths are absent; using description defaultWidth");
    }

    std::vector<uint64_t> ids;
    if (options.idSource == TfToken("primvar") && !source.curveId.empty()) {
        if (source.curveId.size() != curves)
            return Fail(diag, "C3 usdGen:curveId cardinality must be uniform per curve");
        ids.assign(source.curveId.begin(), source.curveId.end());
    } else {
        ids.resize(curves);
        for (size_t i = 0; i != curves; ++i) ids[i] = i;
        Warn(diag, "C3 stable curve ids are synthesized from source indices");
    }
    // Identity is a source contract, not merely a property of the surviving
    // frame set.  A degenerate curve must not hide duplicate authored ids.
    std::vector<size_t> allIdOrder(curves);
    for (size_t curve = 0; curve != curves; ++curve) allIdOrder[curve] = curve;
    std::stable_sort(allIdOrder.begin(), allIdOrder.end(),
                     [&](size_t a, size_t b) { return ids[a] < ids[b]; });
    for (size_t i = 1; i < allIdOrder.size(); ++i)
        if (ids[allIdOrder[i - 1]] == ids[allIdOrder[i]])
            return Fail(diag, "C3 contains duplicate stable curve ids");
    UsdGenSurfaceDesc const *surface = FindSurface(ctx);
    bool const hasSurfaceRelationship = ctx.params && ctx.params->node &&
        !ctx.params->node->surfaces.empty();
    if (hasSurfaceRelationship &&
        (ctx.params->node->surfaces.size() != 1 || !surface))
        return Fail(diag, "C3 bound surface relationship is missing or ambiguous");
    UsdGenCurveRootCaptureResult roots;
    auto candidateCache = bindingCache ? *bindingCache
        : std::shared_ptr<const UsdGenRestSurfaceBindingCache>{};
    std::string rootError;
    if (!UsdGenCaptureCurveRoots(source, surface, options.rebind, &roots, &rootError, &candidateCache))
        return Fail(diag, "cannot capture C3 root bindings/frames: " + rootError);
    std::vector<size_t> order;
    order.reserve(curves);
    for (size_t curve = 0; curve != curves; ++curve)
        if (roots.valid[curve]) order.push_back(curve);
    if (roots.dropped)
        Warn(diag, std::to_string(roots.dropped) +
             " C3 curves were dropped because root capture was unresolved or degenerate");
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return ids[a] < ids[b]; });

    std::vector<uint32_t> sourceOffsets(curves + 1, 0);
    for (size_t c = 0; c != curves; ++c)
        sourceOffsets[c + 1] = sourceOffsets[c] + uint32_t(source.curveVertexCounts[c]);

    // Reorder named planes by canonical id before optional resampling.  The
    // helpers create fresh plane owners and preserve typed interpolation.
    UsdGenCurveBuffer sourcePlanes;
    sourcePlanes.totalCurves = static_cast<uint32_t>(curves);
    sourcePlanes.totalCvs = static_cast<uint32_t>(total);
    sourcePlanes.cvOffsets.resize(curves + 1);
    for (size_t i = 0; i != sourceOffsets.size(); ++i)
        sourcePlanes.cvOffsets[i] = static_cast<int>(sourceOffsets[i]);
    if (!BuildSourcePlanes(source, sourcePlanes.totalCurves, sourcePlanes.totalCvs,
                           &sourcePlanes.extraCv, &sourcePlanes.extraCurve, diag))
        return false;
    UsdGenCurveBuffer orderedPlanes;
    orderedPlanes.totalCurves = static_cast<uint32_t>(order.size());
    orderedPlanes.cvOffsets.resize(order.size() + 1);
    uint32_t orderedTotal = 0;
    for (size_t i = 0; i != order.size(); ++i) {
        orderedPlanes.cvOffsets[i] = static_cast<int>(orderedTotal);
        orderedTotal += uint32_t(source.curveVertexCounts[order[i]]);
    }
    orderedPlanes.cvOffsets[order.size()] = static_cast<int>(orderedTotal);
    orderedPlanes.totalCvs = orderedTotal;
    std::string planeError;
    if ((!sourcePlanes.extraCv.empty() || !sourcePlanes.extraCurve.empty()) &&
        !ReorderSourcePlanes(sourcePlanes, order, &orderedPlanes, &planeError))
        return Fail(diag, "cannot reorder authored C3 planes: " + planeError);

    uint64_t outputCvs64 = options.resampleTo
        ? uint64_t(order.size()) * uint64_t(options.resampleTo) : uint64_t(orderedTotal);
    if (outputCvs64 > std::numeric_limits<uint32_t>::max() ||
        outputCvs64 > uint64_t(std::numeric_limits<int>::max()))
        return Fail(diag, "C3 resample output exceeds uint32 cardinality");
    UsdGenCurveBuffer candidate;
    candidate.totalCurves = static_cast<uint32_t>(order.size());
    candidate.totalCvs = static_cast<uint32_t>(outputCvs64);
    bool const uniformSource = order.empty() || std::all_of(order.begin(), order.end(),
        [&](size_t curve) { return source.curveVertexCounts[curve] ==
                                    source.curveVertexCounts[order.front()]; });
    if (!options.resampleTo && !uniformSource) {
        candidate.cvOffsets = orderedPlanes.cvOffsets;
    }
    candidate.px.resize(candidate.totalCvs);
    candidate.py.resize(candidate.totalCvs);
    candidate.pz.resize(candidate.totalCvs);
    // CUDA's C3 producer preserves a usable rest channel for deformed
    // imports with no authored rest by snapshotting current points.  Keep the
    // same explicit fallback here; useRest=true was rejected above unless an
    // authored/default-time rest snapshot exists.
    candidate.rest.resize(candidate.totalCvs);
    candidate.width.resize(candidate.totalCvs);
    candidate.hairT.resize(candidate.totalCvs);
    candidate.curveId.resize(order.size());
    candidate.rootPrim.resize(roots.rootPrim.empty() ? 0 : order.size());
    candidate.rootUV.resize(roots.rootUV.empty() ? 0 : order.size());
    candidate.rootT.resize(roots.frames.empty() ? 0 : order.size());
    candidate.rootB.resize(roots.frames.empty() ? 0 : order.size());
    candidate.rootN.resize(roots.frames.empty() ? 0 : order.size());

    // C3 carries two distinct channels.  `points` is the geometry handed to
    // the evaluator (including a rest-space source); `rest` is transported
    // independently for binding/deformation.  useRest is admission/space
    // policy, never permission to overwrite current points with rest.
    VtVec3fArray const &positions = source.points;
    VtVec3fArray const &restPositions = source.rest.empty() ? source.points : source.rest;
    size_t dst = 0;
    for (size_t sorted = 0; sorted != order.size(); ++sorted) {
        size_t const old = order[sorted];
        uint32_t const first = sourceOffsets[old];
        uint32_t const count = sourceOffsets[old + 1] - first;
        uint32_t const outputCount = options.resampleTo ? uint32_t(options.resampleTo) : count;
        candidate.curveId[sorted] = ids[old];
        if (!roots.rootPrim.empty()) candidate.rootPrim[sorted] = roots.rootPrim[old];
        if (!roots.rootUV.empty()) candidate.rootUV[sorted] = roots.rootUV[old];
        if (!roots.frames.empty()) PutFrame(roots.frames[old], &candidate, sorted);
        for (uint32_t i = 0; i != outputCount; ++i, ++dst) {
            float const u = outputCount > 1 ? float(i) / float(outputCount - 1) : 0.0f;
            float const position = u * float(count - 1);
            // A load without resampling preserves authored CVs exactly.
            // Reconstructing their indices from normalized float hairT can
            // introduce interpolation (e.g. 7/13*13 is slightly above 7).
            uint32_t const lo = options.resampleTo ? static_cast<uint32_t>(position) : i;
            uint32_t const hi = options.resampleTo ? std::min(lo + 1, count - 1) : i;
            float const alpha = options.resampleTo ? position - float(lo) : 0.0f;
            GfVec3f const point = positions[first + lo] +
                (positions[first + hi] - positions[first + lo]) * alpha;
            candidate.px[dst] = point[0];
            candidate.py[dst] = point[1];
            candidate.pz[dst] = point[2];
            GfVec3f const rest = restPositions[first + lo] +
                (restPositions[first + hi] - restPositions[first + lo]) * alpha;
            candidate.rest[dst] = rest;
            if (source.widths.empty()) {
                candidate.width[dst] = ctx.desc->defaultWidth;
            } else if (source.widthsInterpolation == TfToken("constant")) {
                candidate.width[dst] = source.widths[0];
            } else {
                float const a = source.widths[first + lo];
                float const b = source.widths[first + hi];
                candidate.width[dst] = a + (b - a) * alpha;
            }
            candidate.hairT[dst] = u;
        }
    }
    if (options.resampleTo) {
        if ((!orderedPlanes.extraCv.empty() || !orderedPlanes.extraCurve.empty()) &&
            !UsdGenResampleExtraPlanes(orderedPlanes, &candidate, &planeError))
            return Fail(diag, "cannot resample authored C3 planes: " + planeError);
    } else {
        candidate.extraCv = std::move(orderedPlanes.extraCv);
        candidate.extraCurve = std::move(orderedPlanes.extraCurve);
    }
    *out = std::move(candidate);
    if (bindingCache) *bindingCache = std::move(candidateCache);
    return true;
}

} // namespace usdGen
