// usdGen — UsdGenGuideInterpolateOp implementation. 04-operators.md §2.3.
//
// usdGen:weighting selects how a strand follows the guides.
//
// "falloff" (the default) — capture (the whole result, like Clump; the guide
// set is a reference input whose identity already re-runs it):
//   * every guide is resampled to usdGen:cvCount CVs by arc length and kept
//     as root-local offsets (linearBlend) or segment vectors
//     (extrudeAndBlend);
//   * a guide's root normal is its authored usdGen:rootFrame, else the rest
//     normal of the strand rooted nearest it (the surface frame there);
//   * a guide's region is the region value of the strand rooted nearest it,
//     so one connected field (usdGen:region) regions both. A voronoi
//     expression seeded by the guide roots therefore gives every guide its
//     own cell, and a ptex region map gives each guide the region it sits in;
//   * per strand, the nearest guides within usdGen:influenceRadius, of its own
//     region (others scaled by usdGen:regionCrossover), within
//     usdGen:maxGuideAngle, weighted (1 - d/R)^influenceDecay, the heaviest
//     usdGen:maxGuides kept and normalised. A strand with no candidate follows
//     the nearest guide of its region regardless of distance; a region with
//     no guide falls back to the nearest guide overall and is reported.
//   * shapes blend in skin space: each guide's offsets are turned by the
//     rotation taking its root normal onto the strand's (scaled by
//     usdGen:blendInSkinSpace) before weighting.
//
// "nearest" / "inverseDistance" / "gaussian" — the imported-groom blend
// (ops/guideBlend.h): a rigid copy of the nearest guide, inverse-distance
// weights over the 3 nearest guides, or a Gaussian kernel over all guides
// (usdGen:tightness), with optional rotation-minimising frames
// (usdGen:bendFollow) and parting (usdGen:part:*: authored curves, amount,
// angle, radius as a fraction of the curve length, distance). The
// falloff-only controls (maxGuides, influenceRadius, influenceDecay,
// maxGuideAngle, blendMethod, blendInSkinSpace, useUniqueGuide,
// randomizeGuide, region, regionCrossover, regionMap) do not apply there.
// Without guides every strand grows straight along its root normal
// (usdGen:defaultLength).
//
// Every weighting emits guideIndex / guideWeight (arity 3) and the uniform
// int planes primaryGuide (dominant guide), nearestGuide (parting-aware
// nearest guide root), partId (parting key, -1 off the parting) and
// mirrorSign (the guide set's mirrorSign plane at primaryGuide, default +1);
// guide indices are curve indices of the guide set. Upstream uniform planes
// (strandIndex, strandSeed, ...) are carried into the generated buffer.
// Evaluate writes the captured strands, following their roots.
#include "usdGen/ops/guideInterpolate.h"

#include "usdGen/clumpMotion.h"
#include "usdGen/opParams.h"
#include "usdGen/ops/guideBlend.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/regionMap.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"
#include "usdGenMath/usdGenMath/hash.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/vt/types.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

const TfToken sCvCount{"cvCount"}, sMaxGuides{"maxGuides"}, sRadius{"influenceRadius"},
    sDecay{"influenceDecay"}, sMaxAngle{"maxGuideAngle"}, sBlendMethod{"blendMethod"},
    sSkinSpace{"blendInSkinSpace"}, sUnique{"useUniqueGuide"}, sRandomize{"randomizeGuide"},
    sCrossover{"regionCrossover"}, sRegion{"region"}, sLinear{"linearBlend"},
    sExtrude{"extrudeAndBlend"}, sGuideIndex{"guideIndex"}, sGuideWeight{"guideWeight"};
const TfToken sWeighting{"weighting"}, sFalloff{"falloff"}, sNearest{"nearest"},
    sInverseDistance{"inverseDistance"}, sGaussian{"gaussian"}, sTightness{"tightness"},
    sBendFollow{"bendFollow"}, sDefaultLength{"defaultLength"}, sPartPoints{"part:points"},
    sPartCounts{"part:counts"}, sPartNormals{"part:normals"}, sPartAmount{"part:amount"},
    sPartAngle{"part:angle"}, sPartRadius{"part:radius"}, sPartDistance{"part:distance"},
    sPrimaryGuide{"primaryGuide"}, sNearestGuide{"nearestGuide"}, sPartId{"partId"},
    sMirrorSign{"mirrorSign"};

constexpr uint32_t kSaltGuide = 0x47554944u;   // "GUID"
constexpr size_t kMaxCandidates = 32;

struct UsdGenGuideInterpolateCapture final : public UsdGenCapture
{
    uint32_t cvCount = 0;
    std::vector<float> result;   // 3 * curves * cvCount, rooted at `roots`
    std::vector<float> roots;    // 3 * curves
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenGuideInterpolateCapture>(*this);
    }
    bool OwnsBuffer() const override { return true; }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves;
    }
};

struct Cloud {
    std::vector<float> points;
    size_t kdtree_get_point_count() const { return points.size() / 3; }
    float kdtree_get_pt(size_t i, size_t d) const { return points[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B &) const { return false; }
};
using CloudIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (!dispatcher || count < 256) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

enum class Mode { Falloff, Nearest, InverseDistance, Gaussian };

struct Settings {
    int cvCount = 8, maxGuides = 3;
    double radius = 4.0, decay = 2.0, maxAngle = 90.0, skin = 1.0, randomize = 0.0,
           crossover = 0.0;
    bool unique = false, extrude = false;
    Mode mode = Mode::Falloff;
    double tightness = 3.0, defaultLength = 0.01;
    bool bendFollow = false;
    double partAmount = 0.0, partAngle = 80.0, partRadius = 0.5, partDistance = 1.0;
};

/// A point/normal array parameter as float vectors (point3f[] / normal3f[],
/// double spellings tolerated). False when the value has another type.
bool ReadVec3s(VtValue const &value, std::vector<guideBlend::V3> *out)
{
    out->clear();
    if (value.IsEmpty()) return true;
    if (value.IsHolding<VtVec3fArray>()) {
        for (GfVec3f const &p : value.UncheckedGet<VtVec3fArray>())
            out->push_back({p[0], p[1], p[2]});
        return true;
    }
    if (value.IsHolding<VtVec3dArray>()) {
        for (GfVec3d const &p : value.UncheckedGet<VtVec3dArray>())
            out->push_back({float(p[0]), float(p[1]), float(p[2])});
        return true;
    }
    return false;
}

bool ReadInts(VtValue const &value, std::vector<int> *out)
{
    out->clear();
    if (value.IsEmpty()) return true;
    if (!value.IsHolding<VtIntArray>()) return false;
    VtIntArray const &a = value.UncheckedGet<VtIntArray>();
    out->assign(a.begin(), a.end());
    return true;
}

/// The authored parting curves, validated.
bool ReadParting(UsdGenParamView const *p, Settings const &s, guideBlend::Parting *parting,
                 std::string *error)
{
    std::vector<guideBlend::V3> points, normals;
    std::vector<int> counts;
    if (p) {
        if (!ReadVec3s(p->GetVtValue(sPartPoints, VtValue()), &points)) {
            *error = "usdGen:part:points must be a point3f[]";
            return false;
        }
        if (!ReadInts(p->GetVtValue(sPartCounts, VtValue()), &counts)) {
            *error = "usdGen:part:counts must be an int[]";
            return false;
        }
        if (!ReadVec3s(p->GetVtValue(sPartNormals, VtValue()), &normals)) {
            *error = "usdGen:part:normals must be a normal3f[]";
            return false;
        }
    }
    size_t total = 0;
    for (int c : counts) {
        if (c < 0) {
            *error = "usdGen:part:counts must be >= 0";
            return false;
        }
        total += size_t(c);
    }
    if (!counts.empty() && total != points.size()) {
        *error = "usdGen:part:counts must sum to the size of usdGen:part:points";
        return false;
    }
    if (!normals.empty() && normals.size() != points.size()) {
        *error = "usdGen:part:normals must be empty or one per usdGen:part:points entry";
        return false;
    }
    for (guideBlend::V3 const &v : points) {
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            *error = "usdGen:part:points must be finite";
            return false;
        }
    }
    if (parting)
        parting->Set(std::move(points), counts, float(s.partAmount), float(s.partAngle),
                     float(s.partRadius), float(s.partDistance));
    return true;
}

bool ReadSettings(UsdGenParamView const *p, Settings *s, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error(std::string("UsdGenGuideInterpolate: ") + message);
        return false;
    };
    if (!p) return true;
    s->cvCount = p->GetInt(sCvCount, 8);
    s->maxGuides = p->GetInt(sMaxGuides, 3);
    s->radius = p->GetDouble(sRadius, 4.0);
    s->decay = p->GetDouble(sDecay, 2.0);
    s->maxAngle = p->GetDouble(sMaxAngle, 90.0);
    s->skin = p->GetDouble(sSkinSpace, 1.0);
    s->randomize = p->GetDouble(sRandomize, 0.0);
    s->crossover = p->GetDouble(sCrossover, 0.0);
    s->unique = p->GetBool(sUnique, false);
    TfToken const method = p->GetToken(sBlendMethod, sLinear);
    if (method != sLinear && method != sExtrude)
        return fail("usdGen:blendMethod must be linearBlend or extrudeAndBlend");
    s->extrude = method == sExtrude;
    TfToken const weighting = p->GetToken(sWeighting, sFalloff);
    if (weighting == sFalloff) s->mode = Mode::Falloff;
    else if (weighting == sNearest) s->mode = Mode::Nearest;
    else if (weighting == sInverseDistance) s->mode = Mode::InverseDistance;
    else if (weighting == sGaussian) s->mode = Mode::Gaussian;
    else return fail("usdGen:weighting must be falloff, nearest, inverseDistance or gaussian");
    s->tightness = p->GetDouble(sTightness, 3.0);
    s->bendFollow = p->GetBool(sBendFollow, false);
    s->defaultLength = p->GetDouble(sDefaultLength, 0.01);
    s->partAmount = p->GetDouble(sPartAmount, 0.0);
    s->partAngle = p->GetDouble(sPartAngle, 80.0);
    s->partRadius = p->GetDouble(sPartRadius, 0.5);
    s->partDistance = p->GetDouble(sPartDistance, 1.0);
    if (s->cvCount < 2 || s->cvCount > int(kUsdGenMaxCvsPerCurve))
        return fail("usdGen:cvCount must be in [2, " + std::to_string(kUsdGenMaxCvsPerCurve) + "]");
    if (s->maxGuides < 1 || s->maxGuides > 8) return fail("usdGen:maxGuides must be in [1, 8]");
    if (!std::isfinite(s->radius) || s->radius <= 0.0)
        return fail("usdGen:influenceRadius must be finite and > 0");
    if (!std::isfinite(s->decay) || s->decay < 0.0)
        return fail("usdGen:influenceDecay must be finite and >= 0");
    if (!(s->maxAngle >= 0.0 && s->maxAngle <= 180.0))
        return fail("usdGen:maxGuideAngle must be in [0, 180] degrees");
    if (!(s->skin >= 0.0 && s->skin <= 1.0) || !(s->randomize >= 0.0 && s->randomize <= 1.0) ||
        !(s->crossover >= 0.0 && s->crossover <= 1.0))
        return fail("usdGen:blendInSkinSpace, randomizeGuide and regionCrossover must be in [0, 1]");
    if (!std::isfinite(s->tightness)) return fail("usdGen:tightness must be finite");
    if (!std::isfinite(s->defaultLength) || s->defaultLength < 0.0)
        return fail("usdGen:defaultLength must be finite and >= 0");
    if (!std::isfinite(s->partAmount) || !std::isfinite(s->partAngle) ||
        !std::isfinite(s->partRadius) || !std::isfinite(s->partDistance))
        return fail("usdGen:part:amount, angle, radius and distance must be finite");
    std::string error;
    if (!ReadParting(p, *s, nullptr, &error)) return fail(error);
    return true;
}

/// A guide resampled to `n` CVs by arc length, into `out` (xyz).
void Resample(float const *points, size_t count, size_t n, float *out)
{
    std::vector<double> arc(count, 0.0);
    for (size_t i = 1; i < count; ++i) {
        double const dx = double(points[i * 3]) - points[(i - 1) * 3];
        double const dy = double(points[i * 3 + 1]) - points[(i - 1) * 3 + 1];
        double const dz = double(points[i * 3 + 2]) - points[(i - 1) * 3 + 2];
        arc[i] = arc[i - 1] + std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    double const total = count ? arc.back() : 0.0;
    size_t segment = 0;
    for (size_t k = 0; k < n; ++k) {
        if (count < 2 || total <= 0.0) {
            for (int d = 0; d < 3; ++d) out[k * 3 + d] = count ? points[d] : 0.0f;
            continue;
        }
        double const target = total * double(k) / double(n - 1);
        while (segment + 2 < count && arc[segment + 1] < target) ++segment;
        double const span = arc[segment + 1] - arc[segment];
        double const f = span > 0.0 ? std::clamp((target - arc[segment]) / span, 0.0, 1.0) : 0.0;
        for (int d = 0; d < 3; ++d)
            out[k * 3 + d] = float(points[segment * 3 + d] +
                (double(points[(segment + 1) * 3 + d]) - points[segment * 3 + d]) * f);
    }
}

/// The guide set's per-curve plane `name` as ints (int or float uniform,
/// arity 1), or empty.
std::vector<int> CurveInts(UsdGenCurveBuffer const &buffer, TfToken const &name)
{
    for (UsdGenPlane const &plane : buffer.extraCurve) {
        if (plane.name != name || plane.arity != 1 ||
            plane.interpolation != TfToken("uniform"))
            continue;
        if (plane.type == TfToken("int") && plane.i.size() == buffer.totalCurves)
            return std::vector<int>(plane.i.begin(), plane.i.end());
        if (plane.type == TfToken("float") && plane.f.size() == buffer.totalCurves) {
            std::vector<int> out(plane.f.size());
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = std::isfinite(plane.f[i]) ? int(std::lround(plane.f[i])) : 0;
            return out;
        }
    }
    return {};
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("guides"));
    v.push_back(TfToken("regionMap"));
    for (char const *name : {"cvCount", "maxGuides", "influenceRadius", "influenceDecay",
                             "maxGuideAngle", "blendMethod", "blendInSkinSpace",
                             "useUniqueGuide", "randomizeGuide", "regionCrossover", "region",
                             "weighting", "tightness", "bendFollow", "defaultLength",
                             "part:points", "part:counts", "part:normals", "part:amount",
                             "part:angle", "part:radius", "part:distance"})
        v.push_back(TfToken(name));
    return v;
}();

static TfTokenVector _valueParams = [] {
    return UsdGenBaseValueParams();
}();

static TfTokenVector const _referenceInputs{TfToken("guides")};

TfSpan<const TfToken> UsdGenGuideInterpolateOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenGuideInterpolateOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
TfSpan<const TfToken> UsdGenGuideInterpolateOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(_referenceInputs.data(), _referenceInputs.size());
}
std::unique_ptr<UsdGenCapture> UsdGenGuideInterpolateOp::CreateCapture() const
{
    return std::make_unique<UsdGenGuideInterpolateCapture>();
}
uint32_t UsdGenGuideInterpolateOp::PlanesTouched() const
{
    return kPlanePoints | kPlaneWidths | kPlaneHairT;
}

bool UsdGenGuideInterpolateOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    Settings settings;
    return ReadSettings(&params, &settings, diag);
}

UsdGenEpoch UsdGenGuideInterpolateOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    opUtil::Digest d;
    if (UsdGenParamView const *p = ctx.params) {
        for (TfToken const &name : _topoParams) {
            d.Mix(name);
            d.Mix(p->GetVtValue(name, VtValue()));
        }
    }
    d.Mix(uint64_t(ctx.seed));
    d.Mix(double(ctx.desc ? ctx.desc->defaultWidth : 0.01f));
    return d.Epoch(0x4775696465ull);
}

bool UsdGenGuideInterpolateOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenGuideInterpolateCapture *>(out);
    Settings s;
    if (!ReadSettings(p, &s, diag)) return false;
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenGuideInterpolate: " + message);
        return false;
    };
    bool const falloff = s.mode == Mode::Falloff;

    // --- the guide set (optional for the reference weightings) -----------------
    static UsdGenCurveBuffer const kNoGuides;
    bool const haveRef = ctx.referenceCount == 1 && ctx.resolvedReferences &&
        ctx.resolvedReferences[0] && ctx.resolvedReferences[0]->value;
    if (falloff ? !haveRef : ctx.referenceCount > 1 || (ctx.referenceCount == 1 && !haveRef))
        return fail("usdGen:guides must target one guide curve set");
    UsdGenCurveBuffer const &guides =
        haveRef ? ctx.resolvedReferences[0]->value->buffer : kNoGuides;
    UsdGenClumpMotion guideClumps;
    if (haveRef) {
        std::string clumpError;
        if (!UsdGenBuildClumpMotion(guides, &guideClumps, &clumpError))
            return fail("guide clump motion metadata " + clumpError);
    }
    size_t const G = guides.totalCurves;
    if (falloff && G == 0) return fail("the guide curve set has no curves");

    std::vector<uint32_t> guideSpans, rootSpans;
    std::string error;
    if (!opUtil::CurveSpans(guides, &guideSpans, &error))
        return fail("guide topology: " + error);
    if (!opUtil::CurveSpans(upstream, &rootSpans, &error))
        return fail("root topology: " + error);
    if (guides.px.size() != guides.totalCvs || upstream.px.size() != upstream.totalCvs)
        return fail("point planes do not match their CV counts");

    guideBlend::Parting parting;
    if (!ReadParting(p, s, &parting, &error)) return fail(error);

    size_t const R = upstream.totalCurves;
    size_t const n = size_t(s.cvCount);
    if (R > std::numeric_limits<uint32_t>::max() / n)
        return fail("CV topology exceeds uint32 cardinality");
    cap.cvCount = uint32_t(n);
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;

    // --- strand roots, frames and regions ------------------------------------
    std::vector<float> roots(R * 3), rest(R * 3);
    for (size_t c = 0; c < R; ++c) {
        GfVec3f const cur = opUtil::Point(upstream, rootSpans[c]);
        GfVec3f const r = opUtil::RestPoint(upstream, rootSpans[c]);
        for (int d = 0; d < 3; ++d) { roots[c * 3 + d] = cur[d]; rest[c * 3 + d] = r[d]; }
    }
    auto hairNormal = [&](size_t c) {
        return upstream.rootN.size() == R ? opUtil::Normalized(upstream.rootN[c])
                                          : GfVec3f(0, 1, 0);
    };
    // The root normal as authored (unit from the scatter), for the reference math.
    auto rawNormal = [&](size_t c) -> guideBlend::V3 {
        if (upstream.rootN.size() == R) {
            GfVec3f const nrm = upstream.rootN[c];
            if (nrm.GetLengthSq() > 1e-24f) return {nrm[0], nrm[1], nrm[2]};
        }
        return {0.0f, 1.0f, 0.0f};
    };
    std::vector<int64_t> hairRegion(R, 0);
    bool mapped = false;
    if (falloff) {
        UsdGenParamField const regionField =
            p ? p->GetScalarField(sRegion, 0.0) : UsdGenParamField{};
        if (regionField.connected)
            for (size_t c = 0; c < R; ++c)
                hairRegion[c] = opUtil::RegionKey(regionField.Value(c, rootSpans[c]));
        std::vector<int> mapRegions;
        if (!UsdGenReadRootRegions(ctx, upstream, &mapRegions, &mapped, &error)) return fail(error);
        if (mapped) {
            if (s.crossover != 0)
                return fail("regionMap requires regionCrossover=0 for hard partitions");
            for (size_t c = 0; c < R; ++c) hairRegion[c] = opUtil::RegionKey(double(mapRegions[c]));
        }
    }
    uint64_t const *ids = upstream.curveId.size() == R ? upstream.curveId.cdata() : nullptr;

    Cloud hairCloud;
    hairCloud.points = rest;
    std::unique_ptr<CloudIndex> hairIndex;
    if (R) hairIndex = std::make_unique<CloudIndex>(3, hairCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    auto nearestHair = [&](float const *q) -> int64_t {
        if (!hairIndex) return -1;
        uint32_t found = 0;
        float distance = 0.0f;
        return hairIndex->knnSearch(q, 1, &found, &distance) ? int64_t(found) : -1;
    };

    // --- guide widths (resampled by CV index) and root normals ----------------
    bool const guidesHaveWidth = G > 0 && guides.width.size() == guides.totalCvs;
    std::vector<float> guideWidth;
    if (guidesHaveWidth) {
        guideWidth.assign(G * n, 0.0f);
        for (size_t g = 0; g < G; ++g) {
            size_t const first = guideSpans[g], count = guideSpans[g + 1] - first;
            if (count == 0) continue;
            for (size_t k = 0; k < n; ++k) {
                float const at = n > 1 ? float(k) / float(n - 1) * float(count - 1) : 0.0f;
                size_t const lo = std::min(size_t(at), count - 1), hi = std::min(lo + 1, count - 1);
                guideWidth[g * n + k] = guides.width[first + lo] +
                    (guides.width[first + hi] - guides.width[first + lo]) * (at - float(lo));
            }
        }
    }
    // Reference root normal of every guide: the authored root frame, else the
    // bound surface's normal at the guide root (parting only), else the rest
    // normal of the strand rooted nearest it.
    std::vector<guideBlend::V3> guideNormals(G, guideBlend::V3{0.0f, 1.0f, 0.0f});
    {
        strand::TriangleBvh bvh;
        bool const useSurface = parting.active && guides.rootN.size() != G && ctx.desc &&
            p && p->node && !p->node->surfaces.empty() && ctx.surface < ctx.desc->surfaces.size();
        if (useSurface) strand::BuildSurfaceBvh(ctx.desc->surfaces[ctx.surface], true, &bvh);
        for (size_t g = 0; g < G; ++g) {
            if (guideSpans[g + 1] == guideSpans[g]) continue;
            size_t const first = guideSpans[g];
            if (guides.rootN.size() == G) {
                GfVec3f const nrm = guides.rootN[g];
                guideNormals[g] = {nrm[0], nrm[1], nrm[2]};
                continue;
            }
            GfVec3f const root(guides.px[first], guides.py[first], guides.pz[first]);
            if (!bvh.Empty()) {
                strand::ClosestHit const hit = bvh.Closest(root);
                if (hit.hit) {
                    GfVec3f const nrm = bvh.Normal(hit.triangle, hit.bary);
                    guideNormals[g] = {nrm[0], nrm[1], nrm[2]};
                    continue;
                }
            }
            float const q[3] = {root[0], root[1], root[2]};
            int64_t const near = nearestHair(q);
            if (near >= 0) {
                guideBlend::V3 const nrm = rawNormal(size_t(near));
                guideNormals[g] = nrm;
            }
        }
    }

    // --- reference guide tables ------------------------------------------------
    guideBlend::Weighting const weighting =
        s.mode == Mode::Gaussian ? guideBlend::Weighting::Gaussian
        : s.mode == Mode::InverseDistance ? guideBlend::Weighting::InverseDistance
                                          : guideBlend::Weighting::Nearest;
    guideBlend::Guides tables;
    guideBlend::BuildGuides(guides.px.cdata(), guides.py.cdata(), guides.pz.cdata(), guideSpans,
                            guideNormals, s.cvCount, weighting, !falloff && s.bendFollow,
                            parting, &tables);
    auto curveOf = [&](int tableIndex) {
        return tableIndex >= 0 && size_t(tableIndex) < tables.curve.size()
            ? int(tables.curve[size_t(tableIndex)]) : -1;
    };

    // --- per strand ---------------------------------------------------------------
    float const fallbackWidth = ctx.desc ? ctx.desc->defaultWidth : 0.01f;
    cap.roots = roots;
    cap.result.assign(R * n * 3, 0.0f);
    std::vector<float> restResult(R * n * 3, 0.0f);
    std::vector<float> widths(R * n, fallbackWidth);
    std::vector<int> planeIndex(R * 3, -1);
    std::vector<float> planeWeight(R * 3, 0.0f);
    std::vector<int> selectedGuide(R * 8, -1);
    std::vector<float> selectedWeight(R * 8, 0.0f);
    std::vector<int> primaryGuide(R, -1), nearestGuide(R, -1), partId(R, -1);
    std::atomic<size_t> orphans{0};
    bool const rootWidth = upstream.width.size() == upstream.totalCvs;
    auto widthAt = [&](size_t c, size_t i) {
        if (guidesHaveWidth) {
            // The published (heaviest three) weights, renormalised.
            float w = 0.0f, total = 0.0f;
            for (size_t j = 0; j < 3; ++j) {
                int const g = planeIndex[c * 3 + j];
                if (g < 0) continue;
                w += guideWidth[size_t(g) * n + i] * planeWeight[c * 3 + j];
                total += planeWeight[c * 3 + j];
            }
            if (total > 0.0f) return w / total;
        }
        return rootWidth ? upstream.width[rootSpans[c]] : fallbackWidth;
    };

    if (!falloff) {
        guideBlend::StrandSettings strandSettings;
        strandSettings.weighting = weighting;
        strandSettings.tightness = float(s.tightness);
        strandSettings.defaultLength = float(s.defaultLength);
        strandSettings.cvCount = s.cvCount;
        strandSettings.bendFollow = s.bendFollow;
        ParallelFor(ctx.dispatcher, R, [&](size_t c) {
            thread_local guideBlend::Scratch scratch;
            guideBlend::V3 const root{rest[c * 3], rest[c * 3 + 1], rest[c * 3 + 2]};
            guideBlend::StrandResult result;
            float *outRest = &restResult[c * n * 3];
            guideBlend::BuildStrand(tables, parting, strandSettings, root, rawNormal(c), outRest,
                                    &result, &scratch);
            primaryGuide[c] = curveOf(result.primary);
            nearestGuide[c] = curveOf(result.nearest);
            partId[c] = result.partId;
            for (size_t j = 0; j < 3; ++j) {
                int const g = result.dropped ? -1 : curveOf(result.index[j]);
                planeIndex[c * 3 + j] = g;
                planeWeight[c * 3 + j] = g >= 0 ? result.weight[j] : 0.0f;
            }
            for (size_t i = 0; i < n; ++i) widths[c * n + i] = widthAt(c, i);
            float const delta[3] = {roots[c * 3] - root.x, roots[c * 3 + 1] - root.y,
                                    roots[c * 3 + 2] - root.z};
            for (size_t i = 0; i < n; ++i)
                for (int d = 0; d < 3; ++d)
                    cap.result[(c * n + i) * 3 + d] = outRest[i * 3 + d] + delta[d];
        });
    } else {
        // --- falloff: guides -------------------------------------------------
        std::vector<float> guideRoot(G * 3), guideShape(G * n * 3), guideLength(G * n, 0.0f);
        std::vector<GfVec3f> guideNormal(G, GfVec3f(0, 1, 0));
        std::vector<int64_t> guideRegion(G, 0);
        std::vector<float> points;
        for (size_t g = 0; g < G; ++g) {
            size_t const first = guideSpans[g], count = guideSpans[g + 1] - first;
            points.resize(count * 3);
            for (size_t i = 0; i < count; ++i) {
                points[i * 3] = guides.px[first + i];
                points[i * 3 + 1] = guides.py[first + i];
                points[i * 3 + 2] = guides.pz[first + i];
            }
            float *shape = &guideShape[g * n * 3];
            Resample(points.data(), count, n, shape);
            for (int d = 0; d < 3; ++d) guideRoot[g * 3 + d] = shape[d];
            for (size_t k = n; k-- > 0;)
                for (int d = 0; d < 3; ++d) shape[k * 3 + d] -= shape[d];   // root-local
            for (size_t k = 1; k < n; ++k) {
                GfVec3f const seg(shape[k * 3] - shape[(k - 1) * 3],
                                  shape[k * 3 + 1] - shape[(k - 1) * 3 + 1],
                                  shape[k * 3 + 2] - shape[(k - 1) * 3 + 2]);
                guideLength[g * n + k] = seg.GetLength();
            }
            int64_t const near = nearestHair(&guideRoot[g * 3]);
            if (guides.rootN.size() == G) guideNormal[g] = opUtil::Normalized(guides.rootN[g]);
            else if (near >= 0) guideNormal[g] = hairNormal(size_t(near));
            if (near >= 0) guideRegion[g] = hairRegion[size_t(near)];
        }
        Cloud guideCloud;
        guideCloud.points = guideRoot;
        CloudIndex guideIndex(3, guideCloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        double const cosMax = std::cos(s.maxAngle * 3.14159265358979323846 / 180.0);
        size_t const maxKeep = s.unique ? 1 : size_t(s.maxGuides);
        std::vector<uint8_t> eligible(G, 0);
        for (uint32_t g : tables.curve) eligible[g] = 1;

        ParallelFor(ctx.dispatcher, R, [&](size_t c) {
            GfVec3f const normal = hairNormal(c);
            float const *q = &rest[c * 3];
            struct Candidate { uint32_t guide; double weight; };
            Candidate keep[kMaxCandidates];
            size_t kept = 0;
            uint32_t found[kMaxCandidates];
            float distances[kMaxCandidates];
            size_t const k = std::min(G, kMaxCandidates);
            size_t const hits = guideIndex.knnSearch(q, k, found, distances);
            auto angleOk = [&](uint32_t g) {
                return s.maxAngle >= 180.0 || double(GfDot(normal, guideNormal[g])) >= cosMax - 1e-9;
            };
            for (size_t j = 0; j < hits; ++j) {
                uint32_t const g = found[j];
                double const d = std::sqrt(double(distances[j]));
                if (d >= s.radius || !angleOk(g)) continue;
                double w = std::pow(1.0 - d / s.radius, s.decay);
                if (guideRegion[g] != hairRegion[c]) w *= s.crossover;
                if (s.randomize > 0.0)
                    w *= 1.0 - s.randomize *
                        double(UsdGenHash01(UsdGenHash64(ids ? ids[c] : c, uint64_t(g), kSaltGuide),
                                            kSaltGuide + uint32_t(ctx.seed)));
                if (w <= 0.0) continue;
                keep[kept++] = {g, w};
            }
            if (kept == 0) {
                // Nearest guide of the strand's own region, however far.
                auto nearestIn = [&](bool sameRegion, bool checkAngle) -> int64_t {
                    for (size_t j = 0; j < hits; ++j)
                        if ((!sameRegion || guideRegion[found[j]] == hairRegion[c]) &&
                            (!checkAngle || angleOk(found[j])))
                            return found[j];
                    int64_t best = -1;
                    double bestDistance = std::numeric_limits<double>::infinity();
                    for (uint32_t g = 0; g < G; ++g) {
                        if ((sameRegion && guideRegion[g] != hairRegion[c]) ||
                            (checkAngle && !angleOk(g))) continue;
                        double const dx = double(q[0]) - guideRoot[g * 3];
                        double const dy = double(q[1]) - guideRoot[g * 3 + 1];
                        double const dz = double(q[2]) - guideRoot[g * 3 + 2];
                        double const distance = dx * dx + dy * dy + dz * dz;
                        if (distance < bestDistance) { bestDistance = distance; best = g; }
                    }
                    return best;
                };
                int64_t g = nearestIn(true, true);
                if (g < 0) g = nearestIn(true, false);
                if (g < 0) { g = nearestIn(false, false); ++orphans; }
                keep[kept++] = {uint32_t(g), 1.0};
            }
            std::sort(keep, keep + kept, [](Candidate const &a, Candidate const &b) {
                return a.weight != b.weight ? a.weight > b.weight : a.guide < b.guide;
            });
            kept = std::min(kept, maxKeep);
            double total = 0.0;
            for (size_t j = 0; j < kept; ++j) total += keep[j].weight;
            for (size_t j = 0; j < kept; ++j) keep[j].weight = total > 0.0 ? keep[j].weight / total : 1.0 / double(kept);
            for (size_t j = 0; j < 3; ++j) {
                planeIndex[c * 3 + j] = j < kept ? int(keep[j].guide) : -1;
                planeWeight[c * 3 + j] = j < kept ? float(keep[j].weight) : 0.0f;
            }
            for (size_t j = 0; j < kept; ++j) {
                selectedGuide[c * 8 + j] = int(keep[j].guide);
                selectedWeight[c * 8 + j] = float(keep[j].weight);
            }
            primaryGuide[c] = kept ? int(keep[0].guide) : -1;
            guideBlend::V3 const root{q[0], q[1], q[2]};
            guideBlend::PartInfo const part = guideBlend::QueryParting(parting, root, rawNormal(c));
            partId[c] = part.weight <= 0.2f ? -1 : guideBlend::PartKey(part);
            // Without parting the nearest guide root is the kd-tree's nearest
            // eligible hit (same squared distance; ties keep the lower index).
            int nearest = -1;
            if (!parting.active) {
                float bestDistance = 0.0f;
                for (size_t j = 0; j < hits; ++j) {
                    if (nearest >= 0 && distances[j] > bestDistance) break;
                    if (!eligible[found[j]]) continue;
                    if (nearest < 0 || int(found[j]) < nearest) {
                        nearest = int(found[j]);
                        bestDistance = distances[j];
                    }
                }
            }
            nearestGuide[c] = nearest >= 0
                ? nearest : curveOf(guideBlend::NearestGuide(tables, parting, part, root));

            // Compose, at the rest root.
            float *outRest = &restResult[c * n * 3];
            GfVec3f previous(q[0], q[1], q[2]);
            for (size_t i = 0; i < n; ++i) {
                GfVec3f value(0.0f);
                float length = 0.0f;
                for (size_t j = 0; j < kept; ++j) {
                    uint32_t const g = keep[j].guide;
                    float const w = float(keep[j].weight);
                    float const *shape = &guideShape[g * n * 3];
                    GfVec3f local = s.extrude && i > 0
                        ? GfVec3f(shape[i * 3] - shape[(i - 1) * 3],
                                  shape[i * 3 + 1] - shape[(i - 1) * 3 + 1],
                                  shape[i * 3 + 2] - shape[(i - 1) * 3 + 2])
                        : GfVec3f(shape[i * 3], shape[i * 3 + 1], shape[i * 3 + 2]);
                    local = opUtil::RotateOnto(local, guideNormal[g], normal, float(s.skin));
                    value += local * w;
                    length += guideLength[g * n + i] * w;
                }
                GfVec3f point;
                if (s.extrude && i > 0) {
                    float const l = value.GetLength();
                    point = previous + (l > 1e-12f ? value * (length / l) : GfVec3f(0.0f));
                } else {
                    point = GfVec3f(q[0], q[1], q[2]) + value;
                }
                previous = point;
                for (int d = 0; d < 3; ++d) outRest[i * 3 + d] = point[d];
                if (guidesHaveWidth) {
                    float w = 0.0f;
                    for (size_t j = 0; j < kept; ++j)
                        w += guideWidth[keep[j].guide * n + i] * float(keep[j].weight);
                    widths[c * n + i] = w;
                } else if (rootWidth) {
                    widths[c * n + i] = upstream.width[rootSpans[c]];
                }
            }
            // The current strand: the rest strand moved with its root.
            float const delta[3] = {roots[c * 3] - q[0], roots[c * 3 + 1] - q[1], roots[c * 3 + 2] - q[2]};
            for (size_t i = 0; i < n; ++i)
                for (int d = 0; d < 3; ++d)
                    cap.result[(c * n + i) * 3 + d] = outRest[i * 3 + d] + delta[d];
        });
        if (mapped && orphans.load()) return fail("regionMap contains a region with no growth guide");
        if (orphans.load() && diag)
            diag->Warn("UsdGenGuideInterpolate: " + std::to_string(orphans.load()) +
                       " strands lie in regions without a guide and follow the nearest guide of"
                       " another region");
    }

    // --- the output buffer ----------------------------------------------------------
    UsdGenCurveBuffer &buf = cap.MutableBuffer();
    buf.totalCurves = uint32_t(R);
    buf.totalCvs = uint32_t(R * n);
    buf.topologyVersion = upstream.topologyVersion + 1;
    buf.cvOffsets.clear();
    buf.px = VtFloatArray(R * n);
    buf.py = VtFloatArray(R * n);
    buf.pz = VtFloatArray(R * n);
    buf.hairT = VtFloatArray(R * n);
    buf.width = VtFloatArray(widths.begin(), widths.end());
    buf.rest = VtVec3fArray(R * n);
    for (size_t o = 0; o < R * n; ++o) {
        buf.px[o] = cap.result[o * 3];
        buf.py[o] = cap.result[o * 3 + 1];
        buf.pz[o] = cap.result[o * 3 + 2];
        buf.hairT[o] = float(o % n) / float(n - 1);
        buf.rest[o] = GfVec3f(restResult[o * 3], restResult[o * 3 + 1], restResult[o * 3 + 2]);
    }
    // Upstream planes (strandIndex, strandSeed, ...) follow their strands.
    std::string planeError;
    if (!UsdGenResampleExtraPlanes(upstream, &buf, &planeError))
        return fail("cannot carry named planes across the CV topology change: " + planeError);
    // A complete native guide quartet follows the chosen guide group. A
    // mixed guide blend retains only the selected group's share of cohesion;
    // the other guides continue shaping geometry without claiming that this
    // strand moves rigidly with a group it did not fully follow. Numeric
    // levels not present on guides retain the root-source planes above.
    auto findGuideIds = [&guides](int level) -> UsdGenPlane const * {
        TfToken const name("clumpId_" + std::to_string(level));
        auto const it = std::lower_bound(guides.extraCurve.begin(), guides.extraCurve.end(), name,
            [](UsdGenPlane const &plane, TfToken const &needle) { return plane.name < needle; });
        return it != guides.extraCurve.end() && it->name == name ? &*it : nullptr;
    };
    auto erasePlane = [&buf](TfToken const &name) {
        auto eraseNamed = [&name](std::vector<UsdGenPlane> *planes) {
            planes->erase(std::remove_if(planes->begin(), planes->end(),
                [&name](UsdGenPlane const &plane) { return plane.name == name; }), planes->end());
        };
        eraseNamed(&buf.extraCurve);
        eraseNamed(&buf.extraCv);
    };
    auto insertNamed = [&buf](UsdGenPlane plane, bool vertex) {
        auto &planes = vertex ? buf.extraCv : buf.extraCurve;
        auto it = std::lower_bound(planes.begin(), planes.end(), plane.name,
            [](UsdGenPlane const &a, TfToken const &b) { return a.name < b; });
        if (it != planes.end() && it->name == plane.name) *it = std::move(plane);
        else planes.insert(it, std::move(plane));
    };
    for (UsdGenClumpMotionLevel const &level : guideClumps.levels) {
        if (level.weight.empty()) continue; // legacy ID only
        UsdGenPlane const *guideIds = findGuideIds(level.level);
        if (!guideIds || guideIds->i.size() != G) continue;
        std::string const suffix = std::to_string(level.level);
        UsdGenPlane outId, outCenter, outCenterId, outWeight;
        outId.name = TfToken("clumpId_" + suffix);
        outId.interpolation = TfToken("uniform");
        outId.type = TfToken("int");
        outId.i = VtIntArray(R, -1);
        outCenter.name = TfToken("clumpCenter_" + suffix);
        outCenter.interpolation = TfToken("uniform");
        outCenter.type = TfToken("float");
        outCenter.arity = 3;
        outCenter.f = VtFloatArray(R * 3, 0.0f);
        outCenterId.name = TfToken("clumpCenterId_" + suffix);
        outCenterId.interpolation = TfToken("uniform");
        outCenterId.type = TfToken("int");
        outCenterId.arity = 2;
        outCenterId.i = VtIntArray(R * 2, 0);
        outWeight.name = TfToken("clumpWeight_" + suffix);
        outWeight.interpolation = TfToken("vertex");
        outWeight.type = TfToken("float");
        outWeight.f = VtFloatArray(R * n, 0.0f);
        for (size_t c = 0; c < R; ++c) {
            uint32_t selected = UINT32_MAX;
            float selectedShare = 0.0f;
            for (size_t j = 0; j < 8; ++j) {
                int const guide = selectedGuide[c * 8 + j];
                if (guide < 0 || size_t(guide) >= G) continue;
                uint32_t const group = level.groupForCurve[size_t(guide)];
                if (group >= level.groups.size()) continue;
                float share = 0.0f;
                for (size_t k = 0; k < 8; ++k) {
                    int const other = selectedGuide[c * 8 + k];
                    if (other >= 0 && size_t(other) < G &&
                        level.groupForCurve[size_t(other)] == group)
                        share += selectedWeight[c * 8 + k];
                }
                if (share > selectedShare || (share == selectedShare &&
                    selected < level.groups.size() &&
                    level.groups[group].centerId < level.groups[selected].centerId)) {
                    selected = group;
                    selectedShare = share;
                }
            }
            if (selected >= level.groups.size() || selectedShare <= 0.0f) continue;
            UsdGenClumpMotionGroup const &group = level.groups[selected];
            for (size_t j = 0; j < 8; ++j) {
                int const guide = selectedGuide[c * 8 + j];
                if (guide < 0 || size_t(guide) >= G ||
                    level.groupForCurve[size_t(guide)] != selected) continue;
                outId.i[c] = guideIds->i[size_t(guide)];
                break;
            }
            for (int d = 0; d < 3; ++d) outCenter.f[c * 3 + size_t(d)] = group.restAnchor[d];
            uint32_t const words[2] = {uint32_t(group.centerId),
                                       uint32_t(group.centerId >> 32)};
            std::memcpy(&outCenterId.i[c * 2], words, sizeof(words));
            for (size_t i = 0; i < n; ++i) {
                float const t = n > 1 ? float(i) / float(n - 1) : 0.0f;
                float blended = 0.0f;
                for (size_t j = 0; j < 8; ++j) {
                    int const guide = selectedGuide[c * 8 + j];
                    if (guide < 0 || size_t(guide) >= G ||
                        level.groupForCurve[size_t(guide)] != selected) continue;
                    size_t const first = guideSpans[size_t(guide)];
                    size_t const count = guideSpans[size_t(guide) + 1] - first;
                    if (count == 0) continue;
                    float const at = count > 1 ? t * float(count - 1) : 0.0f;
                    size_t const lo = std::min(size_t(at), count - 1);
                    size_t const hi = std::min(lo + 1, count - 1);
                    float const f = at - float(lo);
                    float const a = level.weight[first + lo];
                    float const b = level.weight[first + hi];
                    blended += selectedWeight[c * 8 + j] * (a + (b - a) * f);
                }
                outWeight.f[c * n + i] = std::clamp(blended, 0.0f, 1.0f);
            }
        }
        for (TfToken const &name : {outId.name, outCenter.name,
                                    outCenterId.name, outWeight.name}) erasePlane(name);
        insertNamed(std::move(outId), false);
        insertNamed(std::move(outCenter), false);
        insertNamed(std::move(outCenterId), false);
        insertNamed(std::move(outWeight), true);
    }
    auto insertPlane = [&buf](UsdGenPlane plane) {
        auto &planes = buf.extraCurve;
        auto it = std::lower_bound(planes.begin(), planes.end(), plane.name,
            [](UsdGenPlane const &a, TfToken const &b) { return a.name < b; });
        if (it != planes.end() && it->name == plane.name) *it = std::move(plane);
        else planes.insert(it, std::move(plane));
    };
    auto intPlane = [&](TfToken const &name, std::vector<int> const &values, uint8_t arity) {
        UsdGenPlane plane;
        plane.name = name;
        plane.interpolation = TfToken("uniform");
        plane.type = TfToken("int");
        plane.arity = arity;
        plane.i = VtIntArray(values.begin(), values.end());
        insertPlane(std::move(plane));
    };
    intPlane(sGuideIndex, planeIndex, 3);
    UsdGenPlane weight;
    weight.name = sGuideWeight;
    weight.interpolation = TfToken("uniform");
    weight.type = TfToken("float");
    weight.arity = 3;
    weight.f = VtFloatArray(planeWeight.begin(), planeWeight.end());
    insertPlane(std::move(weight));
    intPlane(sPrimaryGuide, primaryGuide, 1);
    intPlane(sNearestGuide, nearestGuide, 1);
    intPlane(sPartId, partId, 1);
    std::vector<int> const guideMirror = CurveInts(guides, sMirrorSign);
    std::vector<int> mirror(R, 1);
    for (size_t c = 0; c < R; ++c) {
        int const g = primaryGuide[c];
        if (g >= 0 && size_t(g) < guideMirror.size()) mirror[c] = guideMirror[size_t(g)] < 0 ? -1 : 1;
    }
    intPlane(sMirrorSign, mirror, 1);
    return true;
}

void UsdGenGuideInterpolateOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    TF_UNUSED(ctx);
    auto const &cap = static_cast<UsdGenGuideInterpolateCapture const &>(captureIn);
    size_t const n = cap.cvCount;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const curve = curveBase + c;
        if ((curve + 1) * n * 3 > cap.result.size()) continue;
        // Input chunks may be ragged roots; rebase like Grow does.
        uint32_t const r = view->inCvOffsets
            ? static_cast<uint32_t>(view->inCvOffsets[c] - int(view->inFirstCv))
            : c * view->inCvCount;
        float const delta[3] = {
            view->inPx ? view->inPx[r] - cap.roots[curve * 3] : 0.0f,
            view->inPy ? view->inPy[r] - cap.roots[curve * 3 + 1] : 0.0f,
            view->inPz ? view->inPz[r] - cap.roots[curve * 3 + 2] : 0.0f};
        for (uint32_t i = 0; i < view->cvCount && i < n; ++i) {
            size_t const o = view->Cv(c, i);
            float const *p = &cap.result[(curve * n + i) * 3];
            view->px[o] = p[0] + delta[0];
            view->py[o] = p[1] + delta[1];
            view->pz[o] = p[2] + delta[2];
            if (view->hairT) view->hairT[o] = n > 1 ? float(i) / float(n - 1) : 0.0f;
        }
    }
}

}  // namespace usdGen
