// usdGen — UsdGenClumpOp implementation. 02-schema.md §2.7, 04 §2.8.
//
// Capture resolves the whole result: membership per level, the centre strand
// of every clump, and the pulled CV positions. The pull reads other strands
// (a clump's centre), and a chunk kernel may only read its own chunk (I3), so
// the centre shapes are snapshots of the input; the capture is re-run
// whenever the input's values move (ValidForTopology checks valueVersion,
// exactly like Grow). Evaluate applies usdGen:mask, the one value-class
// control, and writes the clumpId_<level> planes.
//
// Membership, level 0:
//   usdGen:clump:map connected -> strands whose map value is equal (10-bit
//     quantised) form one clump; its centre is the member rooted nearest the
//     group's rest centroid. A ptex region map, a geoSampler voronoi or any
//     expression can drive it.
//   unconnected -> round(density * restArea) centres are picked from the
//     strands in hash order with a minimum spacing (a jittered, even spread)
//     and every strand joins its nearest centre.
// Level L > 0 splits each level L-1 clump the same implicit way, with
// density / sizeReduction^(2L) centres, size * sizeReduction^L reach and
// amount * tightnessReduction^L strength; its centre shapes are the strands
// as level L-1 left them (goal feedback).
// A strand farther than the level's usdGen:clump:size from its centre is not
// clumped at that level.
//
// Pull, per CV:  w = amount * tightness^L * profile(t) * (1 - stray * t^falloff)
//   linearBlend      P = lerp(P, C(t), w)                  tips converge
//   extrudeAndBlend  P = lerp(P, C(t) + (root - rootC), w)  strands align
// then usdGen:preserveLength restores the input segment lengths, root locked.
//
// Reference clump modes (usdGen:clump:centers, map unconnected; level 0 only,
// further levels split their clumps by density as above). Strand ORDER is the
// upstream strandIndex plane (else the curve index), never the Morton-sorted
// curve order. Helpers live in ops/clumpModes.{h,cpp}.
//   count        K = min(clump:count, N) seed strands: mt19937(usdGen:seed),
//                first pick A[rng() % N], farthest-point seeding (an MSVC-style
//                shuffle when K > 255), grouped by the partId plane when it
//                splits the strands; every strand joins its nearest seed
//                (see clumpRef::AssignCenters on the reference's capacity
//                pass); the target curve is the seed strand's input curve,
//                point-index matched.
//   guides       clump = the strand's guide (nearestGuide plane, else the
//                guide rooted nearest), target = the guide curve resampled to
//                clamp(n, 8, 1000) uniform-arc points, sampled at
//                min(u Ls, Lg) (strand no longer than the guide) or u Lg.
//   guideSeeded  seeds bucketed by guide, per-bucket farthest point; target =
//                a curve grown from the seed root along its guide's segments.
// These blend absolutely, P = (1 - w) P + w T for w > 1e-6, with
// w = clamp01(clamp01(amount) * profile(u)) and, with
// usdGen:clump:profile:mode = "bezier", profile = the quadratic Bezier
// through (0, root), (midShift, mid), (1, tip) at u = i/(n-1) (also used by
// the density/map modes in place of the knots ramp).
// Every mode then applies, at level 0: flyaways (hash01(strandIndex) <
// clump:flyaway:fraction scale their weight by clump:flyaway:amount and lose
// their clump id), saved overrides (usdGen:clump:overrides, indexed by
// strandIndex, used only when usdGen:clump:overrideCount equals the strand
// count) and sub-clumps (usdGen:clump:sub:*, sector centres projected onto
// the bound surface).
#include "usdGen/ops/clump.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/clumpModes.h"
#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"
#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "pxr/base/gf/range3d.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

const TfToken sAmount{"clump:amount"}, sMap{"clump:map"}, sDensity{"clump:density"},
    sSize{"clump:size"}, sLevels{"clump:levels"}, sLevel{"clump:level"},
    sSizeReduction{"clump:sizeReduction"}, sTightness{"clump:tightnessReduction"},
    sMethod{"clump:method"}, sProfileKnots{"clump:profile:knots"},
    sProfileInterp{"clump:profile:interpolation"}, sStrayAmount{"clump:stray:amount"},
    sStrayRate{"clump:stray:rate"}, sStrayFalloff{"clump:stray:falloff"},
    sPreserveLength{"preserveLength"}, sExtrude{"extrudeAndBlend"},
    sLinear{"linearBlend"}, sCatmullRom{"catmullRom"},
    // Reference clump modes.
    sCenters{"clump:centers"}, sCount{"clump:count"}, sProfileMode{"clump:profile:mode"},
    sRoot{"clump:root"}, sMid{"clump:mid"}, sMidShift{"clump:midShift"}, sTip{"clump:tip"},
    sFlyFraction{"clump:flyaway:fraction"}, sFlyAmount{"clump:flyaway:amount"},
    sSubEnabled{"clump:sub:enabled"}, sSubCount{"clump:sub:count"},
    sSubSpread{"clump:sub:spread"}, sSubAmount{"clump:sub:amount"}, sSubIds{"clump:sub:ids"},
    sOverrides{"clump:overrides"}, sOverrideCount{"clump:overrideCount"},
    sModeDensity{"density"}, sModeCount{"count"}, sModeGuides{"guides"},
    sModeGuideSeeded{"guideSeeded"}, sRamp{"ramp"}, sBezier{"bezier"},
    sStrandIndex{"strandIndex"}, sPartId{"partId"}, sNearestGuide{"nearestGuide"};

constexpr uint32_t kSaltClumpStray = kSaltClump + 64u;

struct UsdGenClumpCapture final : public UsdGenCapture
{
    std::vector<float> result;                    // 3 * totalCvs
    std::vector<std::vector<int32_t>> clumpId;    // per level, per strand; -1 = none
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenClumpCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

struct Cloud {
    std::vector<float> points;   // xyz
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

VtVec2fArray ReadKnots(UsdGenParamView const *p, TfToken const &name)
{
    if (!p) return {};
    VtValue const v = p->GetVtValue(name, VtValue());
    return v.IsHolding<VtVec2fArray>() ? v.UncheckedGet<VtVec2fArray>() : VtVec2fArray();
}

enum class Centers { Density, Count, Guides, GuideSeeded };

struct Settings {
    int levels = 1;
    double density = 4.0, size = 1.0, sizeReduction = 0.5, tightness = 0.8;
    double strayAmount = 0.0, strayRate = 0.0, strayFalloff = 1.0;
    bool extrude = false;
    // Reference modes.
    Centers centers = Centers::Density;
    int count = 10;
    bool bezier = false;
    clumpRef::BezierControls controls;
    float flyFraction = 0.0f, flyAmount = 0.0f;
    bool subEnabled = false;
    clumpRef::SubClumpSettings sub;
    VtIntArray overrides;
    int overrideCount = 0;
};

bool ReadSettings(UsdGenParamView const *p, Settings *s, UsdGenDiagnostics *diag)
{
    auto fail = [&](char const *message) {
        if (diag) diag->Error(std::string("UsdGenClump: ") + message);
        return false;
    };
    if (!p) return true;
    s->levels = p->GetInt(sLevels, 1);
    s->density = p->GetDouble(sDensity, 4.0);
    s->size = p->GetDouble(sSize, 1.0);
    s->sizeReduction = p->GetDouble(sSizeReduction, 0.5);
    s->tightness = p->GetDouble(sTightness, 0.8);
    s->strayAmount = p->GetDouble(sStrayAmount, 0.0);
    s->strayRate = p->GetDouble(sStrayRate, 0.0);
    s->strayFalloff = p->GetDouble(sStrayFalloff, 1.0);
    TfToken const method = p->GetToken(sMethod, sLinear);
    if (method != sLinear && method != sExtrude)
        return fail("usdGen:clump:method must be linearBlend or extrudeAndBlend");
    s->extrude = method == sExtrude;
    if (s->levels < 1 || s->levels > 4) return fail("usdGen:clump:levels must be in [1, 4]");
    if (!std::isfinite(s->density) || s->density < 0.0)
        return fail("usdGen:clump:density must be finite and >= 0");
    if (!std::isfinite(s->size) || s->size <= 0.0)
        return fail("usdGen:clump:size must be finite and > 0");
    if (!(s->sizeReduction > 0.0 && s->sizeReduction <= 1.0))
        return fail("usdGen:clump:sizeReduction must be in (0, 1]");
    if (!(s->tightness >= 0.0 && s->tightness <= 1.0))
        return fail("usdGen:clump:tightnessReduction must be in [0, 1]");
    if (!(s->strayAmount >= 0.0 && s->strayAmount <= 1.0) ||
        !(s->strayRate >= 0.0 && s->strayRate <= 1.0) ||
        !(std::isfinite(s->strayFalloff) && s->strayFalloff >= 0.0))
        return fail("usdGen:clump:stray amount/rate must be in [0, 1] and falloff >= 0");

    TfToken const centers = p->GetToken(sCenters, sModeDensity);
    if (centers == sModeDensity) s->centers = Centers::Density;
    else if (centers == sModeCount) s->centers = Centers::Count;
    else if (centers == sModeGuides) s->centers = Centers::Guides;
    else if (centers == sModeGuideSeeded) s->centers = Centers::GuideSeeded;
    else return fail("usdGen:clump:centers must be density, count, guides or guideSeeded");
    s->count = p->GetInt(sCount, 10);
    if (s->count < 0) return fail("usdGen:clump:count must be >= 0");
    TfToken const profileMode = p->GetToken(sProfileMode, sRamp);
    if (profileMode != sRamp && profileMode != sBezier)
        return fail("usdGen:clump:profile:mode must be ramp or bezier");
    s->bezier = profileMode == sBezier;
    s->controls.root = float(p->GetDouble(sRoot, 0.0));
    s->controls.mid = float(p->GetDouble(sMid, 0.5));
    s->controls.midShift = float(p->GetDouble(sMidShift, 0.5));
    s->controls.tip = float(p->GetDouble(sTip, 1.0));
    s->flyFraction = float(p->GetDouble(sFlyFraction, 0.0));
    s->flyAmount = float(p->GetDouble(sFlyAmount, 0.0));
    s->subEnabled = p->GetBool(sSubEnabled, false);
    s->sub.count = p->GetInt(sSubCount, 2);
    s->sub.spread = float(p->GetDouble(sSubSpread, 0.5));
    s->sub.amount = float(p->GetDouble(sSubAmount, 1.0));
    s->sub.ids = p->GetBool(sSubIds, false);
    for (float v : {s->controls.root, s->controls.mid, s->controls.midShift, s->controls.tip,
                    s->flyFraction, s->flyAmount, s->sub.spread, s->sub.amount})
        if (!std::isfinite(v))
            return fail("usdGen:clump profile, flyaway and sub-clump controls must be finite");
    VtValue const overrides = p->GetVtValue(sOverrides, VtValue());
    if (overrides.IsHolding<VtIntArray>()) s->overrides = overrides.UncheckedGet<VtIntArray>();
    else if (!overrides.IsEmpty()) return fail("usdGen:clump:overrides must be an int[]");
    s->overrideCount = p->GetInt(sOverrideCount, 0);
    return true;
}

/// A uniform int plane (arity 1) of the upstream buffer, or null.
int const *IntCurvePlane(UsdGenCurveBuffer const &buffer, TfToken const &name)
{
    for (UsdGenPlane const &plane : buffer.extraCurve)
        if (plane.name == name && plane.type == TfToken("int") && plane.arity == 1 &&
            plane.interpolation == TfToken("uniform") && plane.i.size() == buffer.totalCurves)
            return plane.i.cdata();
    return nullptr;
}

/// Picks up to `want` centres among `members` (hash order, minimum spacing)
/// and returns them. `roots` is xyz per strand.
std::vector<uint32_t> PickCentres(std::vector<uint32_t> const &members,
                                  std::vector<float> const &roots,
                                  uint64_t const *ids, int seed, uint32_t salt,
                                  size_t want, double spacing)
{
    std::vector<std::pair<float, uint32_t>> order;
    order.reserve(members.size());
    for (uint32_t c : members)
        order.emplace_back(UsdGenDraw01(seed, ids ? ids[c] : c, salt), c);
    std::sort(order.begin(), order.end());
    std::vector<uint32_t> centres;
    if (want == 0 || order.empty()) return centres;
    // Uniform grid over the accepted centres: a candidate is rejected when an
    // accepted centre lies closer than `spacing`.
    double const cell = spacing > 0.0 ? spacing : 1.0;
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    auto key = [&](int64_t x, int64_t y, int64_t z) {
        return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
    };
    auto cellOf = [&](float v) { return int64_t(std::floor(double(v) / cell)); };
    for (auto const &entry : order) {
        if (centres.size() >= want) break;
        uint32_t const c = entry.second;
        float const *p = &roots[size_t(c) * 3];
        int64_t const cx = cellOf(p[0]), cy = cellOf(p[1]), cz = cellOf(p[2]);
        bool ok = true;
        if (spacing > 0.0) {
            for (int64_t dx = -1; dx <= 1 && ok; ++dx)
                for (int64_t dy = -1; dy <= 1 && ok; ++dy)
                    for (int64_t dz = -1; dz <= 1 && ok; ++dz) {
                        auto found = grid.find(key(cx + dx, cy + dy, cz + dz));
                        if (found == grid.end()) continue;
                        for (uint32_t other : found->second) {
                            float const *q = &roots[size_t(other) * 3];
                            double const ddx = double(p[0]) - q[0], ddy = double(p[1]) - q[1],
                                         ddz = double(p[2]) - q[2];
                            if (ddx * ddx + ddy * ddy + ddz * ddz < spacing * spacing) {
                                ok = false;
                                break;
                            }
                        }
                    }
        }
        if (!ok) continue;
        centres.push_back(c);
        grid[key(cx, cy, cz)].push_back(c);
    }
    return centres;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    // Clump resolves its whole result at capture (see the file comment), so
    // every control except the envelope is capture-class.
    for (char const *name : {"clump:amount", "clump:map", "clump:density", "clump:size",
                             "clump:levels", "clump:level", "clump:sizeReduction",
                             "clump:tightnessReduction", "clump:method",
                             "clump:profile:knots", "clump:profile:interpolation",
                             "clump:stray:amount", "clump:stray:rate",
                             "clump:stray:falloff", "preserveLength",
                             // Reference modes and the guide relationship.
                             "guides", "clump:centers", "clump:count",
                             "clump:profile:mode", "clump:root", "clump:mid",
                             "clump:midShift", "clump:tip", "clump:flyaway:fraction",
                             "clump:flyaway:amount", "clump:sub:enabled",
                             "clump:sub:count", "clump:sub:spread", "clump:sub:amount",
                             "clump:sub:ids", "clump:overrides", "clump:overrideCount"})
        v.push_back(TfToken(name));
    return v;
}();

static TfTokenVector const _referenceInputs{TfToken("guides")};

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("mask"));
    return v;
}();

TfSpan<const TfToken> UsdGenClumpOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenClumpOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
TfSpan<const TfToken> UsdGenClumpOp::OutputPrimvars() const
{
    return TfSpan<const TfToken>(_outputs.data(), _outputs.size());
}
TfSpan<const TfToken> UsdGenClumpOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(_referenceInputs.data(), _referenceInputs.size());
}
std::unique_ptr<UsdGenCapture> UsdGenClumpOp::CreateCapture() const
{
    return std::make_unique<UsdGenClumpCapture>();
}
uint32_t UsdGenClumpOp::PlanesTouched() const
{
    return kPlanePoints;
}

void UsdGenClumpOp::Configure(UsdGenParamView const &params)
{
    // usdGen:clump:level = -1 numbers this node by its ordinal among every
    // UsdGenClump of the description, disabled ones included, so toggling a
    // node never renumbers another's published planes (04 §2.8).
    int base = params.GetInt(sLevel, -1);
    if (base < 0) {
        base = 0;
        if (params.desc && params.node) {
            for (UsdGenNodeDesc const &node : params.desc->nodes) {
                if (&node == params.node || node.path == params.node->path) break;
                if (node.type == TfToken("UsdGenClump")) ++base;
            }
        }
    }
    int const levels = std::clamp(params.GetInt(sLevels, 1), 1, 4);
    _outputs.clear();
    for (int l = 0; l < levels; ++l)
        _outputs.push_back(TfToken("clumpId_" + std::to_string(base + l)));
}

bool UsdGenClumpOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    Settings settings;
    return ReadSettings(&params, &settings, diag);
}

UsdGenEpoch UsdGenClumpOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    opUtil::Digest d;
    if (UsdGenParamView const *p = ctx.params) {
        for (TfToken const &name : _topoParams) {
            d.Mix(name);
            d.Mix(p->GetVtValue(name, VtValue()));
        }
    }
    d.Mix(uint64_t(ctx.seed));
    d.Mix(ctx.upstreamGeneration);
    if (ctx.desc && ctx.surface < ctx.desc->surfaces.size())
        d.Mix(ctx.desc->surfaces[ctx.surface].surfaceGeneration);
    // The guide set (usdGen:guides) feeds the guides / guideSeeded modes.
    for (uint32_t i = 0; i < ctx.referenceCount; ++i) {
        if (ctx.resolvedReferences && ctx.resolvedReferences[i]) {
            d.Mix(ctx.resolvedReferences[i]->identity);
            d.Mix(ctx.resolvedReferences[i]->curveGeneration);
        } else if (ctx.references && ctx.references[i]) {
            d.Mix(ctx.references[i]->generation);
        }
    }
    return d.Epoch(0x436C756D70ull);
}

bool UsdGenClumpOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenClumpCapture *>(out);
    Settings s;
    if (!ReadSettings(p, &s, diag)) return false;

    size_t const R = upstream.totalCurves;
    size_t const totalCvs = upstream.totalCvs;
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    cap.clumpId.assign(size_t(s.levels), std::vector<int32_t>(R, -1));
    cap.result.clear();
    if (R == 0) return true;

    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(upstream, &spans, &error)) {
        if (diag) diag->Error("UsdGenClump: " + error);
        return false;
    }
    if (upstream.px.size() != totalCvs || upstream.py.size() != totalCvs ||
        upstream.pz.size() != totalCvs) {
        if (diag) diag->Error("UsdGenClump: input point planes do not match the CV count");
        return false;
    }

    // Input shapes, strand roots and parameters.
    std::vector<float> shape(totalCvs * 3);
    for (size_t i = 0; i < totalCvs; ++i) {
        shape[i * 3] = upstream.px[i];
        shape[i * 3 + 1] = upstream.py[i];
        shape[i * 3 + 2] = upstream.pz[i];
    }
    std::vector<float> const input = shape;
    std::vector<float> roots(R * 3);
    for (size_t c = 0; c < R; ++c) {
        GfVec3f const r = opUtil::RestPoint(upstream, spans[c]);
        roots[c * 3] = r[0]; roots[c * 3 + 1] = r[1]; roots[c * 3 + 2] = r[2];
    }
    uint64_t const *ids = upstream.curveId.size() == R ? upstream.curveId.cdata() : nullptr;
    auto hairT = [&](size_t cv, size_t i, size_t n) {
        return upstream.hairT.size() == totalCvs
            ? upstream.hairT[cv]
            : (n > 1 ? float(i) / float(n - 1) : 0.0f);
    };
    UsdGenParamField const amountField = p ? p->GetScalarField(sAmount, 0.5) : UsdGenParamField{0.5};
    UsdGenParamField const mapField = p ? p->GetScalarField(sMap, 0.0) : UsdGenParamField{};
    UsdGenParamField const preserveField =
        p ? p->GetScalarField(sPreserveLength, 1.0) : UsdGenParamField{1.0};
    bool const mapMode = mapField.connected;

    // Knots ramp, unless the Bezier root/mid/tip profile replaces it.
    std::vector<float> profile;
    VtVec2fArray const knots = s.bezier ? VtVec2fArray() : ReadKnots(p, sProfileKnots);
    if (!knots.empty()) {
        profile.assign(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(knots, p ? p->GetToken(sProfileInterp, sCatmullRom) : sCatmullRom,
                           profile.data(), kUsdGenRampLutSize);
    }

    std::vector<float> stray(R, 0.0f);
    if (s.strayRate > 0.0 && s.strayAmount > 0.0)
        for (size_t c = 0; c < R; ++c)
            if (UsdGenDraw01(int(ctx.seed), ids ? ids[c] : c, kSaltClumpStray) < s.strayRate)
                stray[c] = float(s.strayAmount);

    // Strand order (upstream strandIndex, else curve index), flyaways, live
    // roots and lengths for the reference modes, the Bezier profile,
    // overrides and sub-clumps.
    bool const referenceMode = !mapMode && s.centers != Centers::Density;
    std::vector<int> strandIndex(R);
    if (int const *plane = IntCurvePlane(upstream, sStrandIndex))
        std::copy(plane, plane + R, strandIndex.begin());
    else
        std::iota(strandIndex.begin(), strandIndex.end(), 0);
    std::vector<uint32_t> order(R);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return strandIndex[a] < strandIndex[b];
    });
    std::vector<char> flyaway(R, 0);
    if (strand::Clamp01(s.flyFraction) > 0.0f)
        for (size_t c = 0; c < R; ++c)
            flyaway[c] = clumpRef::IsFlyaway(uint64_t(int64_t(strandIndex[c])), s.flyFraction);
    float const flyAmount = strand::Clamp01(s.flyAmount);
    std::vector<float> liveRoots(R * 3, 0.0f), lengths(R, 0.0f);
    {
        std::vector<float> cum;
        for (size_t c = 0; c < R; ++c) {
            size_t const n = spans[c + 1] - spans[c];
            if (n == 0) continue;
            float const *pc = &input[size_t(spans[c]) * 3];
            for (int d = 0; d < 3; ++d) liveRoots[c * 3 + d] = pc[d];
            lengths[c] = strand::ArcLengths(pc, n, &cum);
        }
    }
    // Level-0 clump weight of CV i (of n) of curve c, before stray and
    // flyaway scaling: clamp01(clamp01(amount) * profile).
    auto baseWeight = [&](uint32_t c, size_t i, size_t n) -> float {
        size_t const cv = size_t(spans[c]) + i;
        float const amount = float(amountField.Value(c, cv));
        if (s.bezier) {
            float const u = n > 1 ? float(i) / float(n - 1) : 0.0f;
            return clumpRef::ClumpWeight(u, lengths[c], amount, s.controls);
        }
        float const ramp = profile.empty()
            ? 1.0f : UsdGenEvalLut257(profile.data(), hairT(cv, i, n));
        return strand::Clamp01(strand::Clamp01(amount) * ramp);
    };

    double area = opUtil::RestSurfaceArea(ctx.desc, ctx.surface, ctx.desc &&
        ctx.surface < ctx.desc->surfaces.size());
    if (!(area > 0.0)) {
        // No bound surface: the roots' bounding box stands in for the area.
        GfRange3d box;
        for (size_t c = 0; c < R; ++c)
            box.UnionWith(GfVec3d(roots[c * 3], roots[c * 3 + 1], roots[c * 3 + 2]));
        GfVec3d const e = box.GetSize();
        double const a = e[0] * e[1], b = e[1] * e[2], d = e[0] * e[2];
        area = std::max({a, b, d, 1e-12});
    }

    // --- reference modes (count / guides / guideSeeded), level 0 -------------
    // Membership into `member`, then the absolute lerp toward each clump's
    // target curve. Selection reads the rest roots, targets the live curves.
    auto referenceLevel = [&](std::vector<int32_t> &member) -> bool {
        UsdGenCurveBuffer const *guides =
            ctx.referenceCount >= 1 && ctx.resolvedReferences && ctx.resolvedReferences[0] &&
                    ctx.resolvedReferences[0]->value
                ? &ctx.resolvedReferences[0]->value->buffer : nullptr;
        size_t G = 0;
        clumpRef::GuideTable table;
        std::vector<float> guideRoots;
        if (guides && guides->totalCurves > 0 && s.centers != Centers::Count) {
            size_t steps = 0;
            for (size_t c = 0; c < R; ++c)
                steps = std::max<size_t>(steps, spans[c + 1] - spans[c]);
            std::vector<uint32_t> guideSpans;
            if (!table.Build(*guides, std::clamp<size_t>(steps, 8, 1000), &error) ||
                !opUtil::CurveSpans(*guides, &guideSpans, &error)) {
                if (diag) diag->Error("UsdGenClump: usdGen:guides: " + error);
                return false;
            }
            G = guides->totalCurves;
            guideRoots.assign(G * 3, 0.0f);
            for (size_t g = 0; g < G; ++g) {
                if (guideSpans[g + 1] == guideSpans[g]) continue;
                GfVec3f const r = opUtil::RestPoint(*guides, guideSpans[g]);
                for (int d = 0; d < 3; ++d) guideRoots[g * 3 + d] = r[d];
            }
        }
        int const *partKey = IntCurvePlane(upstream, sPartId);
        // The strand's guide: the upstream nearestGuide plane (parting-aware,
        // from guide interpolation) when valid, else the guide rooted nearest.
        std::vector<int> nearest;
        if (G > 0) {
            int const *plane = IntCurvePlane(upstream, sNearestGuide);
            clumpRef::NearestFinder finder;
            finder.Build(guideRoots);
            nearest.assign(R, -1);
            for (size_t c = 0; c < R; ++c) {
                int g = plane ? plane[c] : -1;
                if (g < 0 || size_t(g) >= G) g = finder.Nearest(&roots[c * 3]);
                nearest[c] = g;
            }
        }

        std::mt19937 rng(ctx.seed);
        size_t const K = std::min(size_t(s.count), R);
        std::vector<uint32_t> centres;
        std::vector<std::vector<float>> centreCurves;   // guideSeeded targets
        if (s.centers == Centers::Guides) {
            if (G == 0) {
                if (diag)
                    diag->Warn("UsdGenClump: usdGen:clump:centers = guides needs a "
                               "usdGen:guides target; no strand is clumped");
            } else {
                for (size_t c = 0; c < R; ++c) member[c] = nearest[c];
            }
        } else if (K > 0) {
            std::vector<std::vector<uint32_t>> candidates;   // centre-index lists
            std::vector<uint32_t> groupOf(R, 0);              // by strand-order position
            // Centres bucketed by parting key; a strand whose key has no
            // centre falls back to group 0 (every centre).
            auto bucketByPart = [&]() {
                std::map<int, uint32_t> bucketOf;
                for (size_t i = 0; i < centres.size(); ++i) {
                    auto const entry = bucketOf.emplace(partKey[centres[i]],
                                                        uint32_t(candidates.size()));
                    if (entry.second) candidates.emplace_back();
                    candidates[entry.first->second].push_back(uint32_t(i));
                }
                for (size_t pos = 0; pos < R; ++pos) {
                    auto const it = bucketOf.find(partKey[order[pos]]);
                    groupOf[pos] = it == bucketOf.end() ? 0u : it->second;
                }
            };
            if (s.centers == Centers::Count) {
                centres = clumpRef::RandomSeedCenters(order, roots.data(), partKey, K, rng);
            } else {
                centres = clumpRef::GuideSeededCenters(order, roots.data(),
                                                       G ? nearest.data() : nullptr,
                                                       guideRoots.data(), G, partKey, K, rng);
                centreCurves.resize(centres.size());
                for (size_t i = 0; i < centres.size(); ++i) {
                    uint32_t const c = centres[i];
                    size_t const n = spans[c + 1] - spans[c];
                    std::vector<float> &curve = centreCurves[i];
                    if (G > 0 && nearest[c] >= 0) {
                        curve.resize(3 * n);
                        clumpRef::SynthesizeCenterCurve(table, size_t(nearest[c]),
                                                        strand::Load(liveRoots.data(), c),
                                                        lengths[c], n, curve.data());
                    } else {
                        curve.assign(input.begin() + std::ptrdiff_t(3 * size_t(spans[c])),
                                     input.begin() + std::ptrdiff_t(3 * size_t(spans[c + 1])));
                    }
                }
            }
            size_t const k = centres.size();
            if (k > 0) {
                candidates.emplace_back(k);
                std::iota(candidates.back().begin(), candidates.back().end(), 0u);
                if (s.centers == Centers::GuideSeeded && G > 0) {
                    // Candidates: the centres of the strand's guide, else of
                    // the guide rooted nearest that has centres.
                    std::vector<std::vector<uint32_t>> ofGuide(G);
                    for (size_t i = 0; i < k; ++i)
                        ofGuide[size_t(nearest[centres[i]])].push_back(uint32_t(i));
                    std::vector<int64_t> groupOfGuide(G, -1);
                    std::vector<float> withPoints;
                    std::vector<size_t> withGuide;
                    for (size_t g = 0; g < G; ++g) {
                        if (ofGuide[g].empty()) continue;
                        groupOfGuide[g] = int64_t(candidates.size());
                        candidates.push_back(ofGuide[g]);
                        withPoints.insert(withPoints.end(), &guideRoots[g * 3],
                                          &guideRoots[g * 3] + 3);
                        withGuide.push_back(g);
                    }
                    clumpRef::NearestFinder withCentres;
                    withCentres.Build(std::move(withPoints));
                    for (size_t pos = 0; pos < R; ++pos) {
                        uint32_t const c = order[pos];
                        int64_t group = groupOfGuide[size_t(nearest[c])];
                        if (group < 0) {
                            int32_t const j = withCentres.Nearest(&roots[size_t(c) * 3]);
                            group = j < 0 ? 0 : groupOfGuide[withGuide[size_t(j)]];
                        }
                        groupOf[pos] = uint32_t(group);
                    }
                } else if (partKey) {
                    bucketByPart();
                }
                std::vector<int32_t> clumpOf(R, -1);
                clumpRef::AssignCenters(order, roots.data(), centres, candidates, groupOf,
                                        &clumpOf);
                for (size_t c = 0; c < R; ++c) member[c] = clumpOf[c];
            }
        }

        // Absolute lerp toward the clump's target curve.
        ParallelFor(ctx.dispatcher, R, [&](size_t c) {
            int32_t const id = member[c];
            if (id < 0) return;
            size_t const first = spans[c], n = spans[c + 1] - spans[c];
            if (n == 0) return;
            float const *tp = nullptr;
            size_t tn = 0, g = 0;
            float gl = 0.0f, Ls = 0.0f;
            if (s.centers == Centers::Guides) {
                g = size_t(id);
                gl = std::max(table.Length(g), 0.0f);
                Ls = lengths[c] > 1e-6f ? lengths[c] : gl;
            } else if (s.centers == Centers::Count) {
                uint32_t const k = centres[size_t(id)];
                tp = &input[size_t(spans[k]) * 3];
                tn = spans[k + 1] - spans[k];
                if (tn == 0) return;
            } else {
                tp = centreCurves[size_t(id)].data();
                tn = centreCurves[size_t(id)].size() / 3;
                if (tn == 0) return;
            }
            auto target = [&](size_t i) -> GfVec3f {
                if (tp) return clumpRef::SampleAtParameter(tp, tn, i, n);
                // Follow the guide by arc length when the strand is no longer
                // than it, else stretch the guide over the strand.
                float const u = n > 1 ? float(i) / float(n - 1) : 0.0f;
                float const a = gl <= 1e-6f ? 0.0f
                    : (Ls <= gl ? std::min(u * Ls, gl) : u * gl);
                return table.Sample(g, a);
            };
            GfVec3f offset(0.0f);
            if (s.extrude) offset = strand::Load(liveRoots.data(), c) - target(0);
            for (size_t i = 0; i < n; ++i) {
                size_t const cv = first + i;
                float w = baseWeight(uint32_t(c), i, n);
                if (stray[c] > 0.0f)
                    w *= 1.0f - stray[c] *
                        float(std::pow(double(hairT(cv, i, n)), s.strayFalloff));
                if (flyaway[c]) w *= flyAmount;
                if (!(w > 1e-6f)) continue;
                GfVec3f const T = target(i) + offset;
                float const keep = 1.0f - w;
                for (int d = 0; d < 3; ++d)
                    shape[cv * 3 + d] = keep * shape[cv * 3 + d] + w * T[d];
            }
        });
        return true;
    };

    // --- level 0, every mode: flyaways lose their id, then saved overrides
    // and sub-clumps.
    auto finishLevel0 = [&](std::vector<int32_t> &member) {
        for (size_t c = 0; c < R; ++c)
            if (flyaway[c]) member[c] = -1;
        if (!s.overrides.empty() || s.overrideCount != 0) {
            if (s.overrideCount == int(R) && s.overrides.size() == R) {
                clumpRef::ApplyOverrides(order, strandIndex, s.overrides, roots.data(), input,
                                         &shape, spans, &member, flyaway, flyAmount, baseWeight);
            } else if (diag) {
                diag->Warn("UsdGenClump: clump overrides ignored: usdGen:clump:overrideCount " +
                           std::to_string(s.overrideCount) + " and " +
                           std::to_string(s.overrides.size()) +
                           " overrides must both equal the strand count " + std::to_string(R));
            }
        }
        if (s.subEnabled && s.sub.count > 1) {
            strand::TriangleBvh bvh;
            bool const bound = ctx.desc && ctx.surface < ctx.desc->surfaces.size() &&
                (!p || !p->node || !p->node->surfaces.empty());
            if (bound) strand::BuildSurfaceBvh(ctx.desc->surfaces[ctx.surface], false, &bvh);
            clumpRef::ApplySubClumps(order, liveRoots.data(), &shape, spans, &member, flyaway,
                                     s.sub, baseWeight, bound && !bvh.Empty() ? &bvh : nullptr);
        }
    };

    std::vector<int32_t> parent(R, 0);
    size_t parentCount = 1;
    for (int level = 0; level < s.levels; ++level) {
        double const reach = s.size * std::pow(s.sizeReduction, level);
        double const density = s.density / std::pow(s.sizeReduction, 2.0 * level);
        float const strength = float(std::pow(s.tightness, level));
        uint32_t const salt = kSaltClumpLevel(level);

        std::vector<int32_t> &member = cap.clumpId[size_t(level)];

        if (level == 0 && referenceMode) {
            if (!referenceLevel(member)) return false;
        } else {
            // --- membership ----------------------------------------------------
            std::vector<std::vector<uint32_t>> groups(parentCount);
            for (size_t c = 0; c < R; ++c)
                if (parent[c] >= 0) groups[size_t(parent[c])].push_back(uint32_t(c));
            std::vector<uint32_t> centreOf;          // clump -> centre strand

            if (level == 0 && mapMode) {
                std::map<std::pair<int32_t, int64_t>, std::vector<uint32_t>> regions;
                for (size_t c = 0; c < R; ++c) {
                    if (parent[c] < 0) continue;
                    double const value = mapField.Value(c, spans[c]);
                    regions[{parent[c], opUtil::RegionKey(value)}].push_back(uint32_t(c));
                }
                for (auto const &region : regions) {
                    GfVec3d centroid(0.0);
                    for (uint32_t c : region.second)
                        centroid += GfVec3d(roots[c * 3], roots[c * 3 + 1], roots[c * 3 + 2]);
                    centroid /= double(region.second.size());
                    uint32_t best = region.second.front();
                    double bestDistance = std::numeric_limits<double>::infinity();
                    for (uint32_t c : region.second) {
                        GfVec3d const d = GfVec3d(roots[c * 3], roots[c * 3 + 1], roots[c * 3 + 2]) - centroid;
                        double const distance = d.GetLengthSq();
                        if (distance < bestDistance) { bestDistance = distance; best = c; }
                    }
                    int32_t const clump = int32_t(centreOf.size());
                    centreOf.push_back(best);
                    for (uint32_t c : region.second) member[c] = clump;
                }
            } else {
                std::vector<uint32_t> centreParent;
                for (size_t g = 0; g < groups.size(); ++g) {
                    auto const &members = groups[g];
                    if (members.empty()) continue;
                    double const share = area * double(members.size()) / double(R);
                    size_t const want = std::max<size_t>(1, size_t(std::llround(density * share)));
                    double const spacing = 0.75 * std::sqrt(share / (3.14159265358979 * double(want)));
                    for (uint32_t c : PickCentres(members, roots, ids, int(ctx.seed), salt, want, spacing)) {
                        centreOf.push_back(c);
                        centreParent.push_back(uint32_t(g));
                    }
                }
                // Nearest centre of the same parent.
                Cloud cloud;
                cloud.points.reserve(centreOf.size() * 3);
                for (uint32_t c : centreOf)
                    for (int d = 0; d < 3; ++d) cloud.points.push_back(roots[c * 3 + d]);
                if (!centreOf.empty()) {
                    CloudIndex index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
                    size_t const k = std::min<size_t>(centreOf.size(), 16);
                    std::vector<std::vector<uint32_t>> byParent(groups.size());
                    for (size_t i = 0; i < centreOf.size(); ++i) byParent[centreParent[i]].push_back(uint32_t(i));
                    ParallelFor(ctx.dispatcher, R, [&](size_t c) {
                        if (parent[c] < 0) return;
                        uint32_t found[16];
                        float distances[16];
                        size_t const n = index.knnSearch(&roots[c * 3], k, found, distances);
                        for (size_t j = 0; j < n; ++j)
                            if (centreParent[found[j]] == uint32_t(parent[c])) {
                                member[c] = int32_t(found[j]);
                                return;
                            }
                        // Every nearby centre belongs to another parent: scan this one's.
                        double best = std::numeric_limits<double>::infinity();
                        for (uint32_t i : byParent[size_t(parent[c])]) {
                            float const *q = &cloud.points[size_t(i) * 3];
                            double const dx = double(roots[c * 3]) - q[0], dy = double(roots[c * 3 + 1]) - q[1],
                                         dz = double(roots[c * 3 + 2]) - q[2];
                            double const distance = dx * dx + dy * dy + dz * dz;
                            if (distance < best) { best = distance; member[c] = int32_t(i); }
                        }
                    });
                }
            }

            // --- reach -----------------------------------------------------------
            for (size_t c = 0; c < R; ++c) {
                if (member[c] < 0) continue;
                uint32_t const centre = centreOf[size_t(member[c])];
                double const dx = double(roots[c * 3]) - roots[centre * 3];
                double const dy = double(roots[c * 3 + 1]) - roots[centre * 3 + 1];
                double const dz = double(roots[c * 3 + 2]) - roots[centre * 3 + 2];
                if (dx * dx + dy * dy + dz * dz > reach * reach) member[c] = -1;
            }

            // --- pull (centre shapes are this level's input: goal feedback) -----
            std::vector<float> const before = shape;
            ParallelFor(ctx.dispatcher, R, [&](size_t c) {
                if (member[c] < 0) return;
                uint32_t const centre = centreOf[size_t(member[c])];
                if (centre == c) return;
                size_t const first = spans[c], n = spans[c + 1] - spans[c];
                size_t const cFirst = spans[centre], cN = spans[centre + 1] - spans[centre];
                if (n == 0 || cN == 0) return;
                float const offset[3] = {
                    s.extrude ? roots[c * 3] - roots[centre * 3] : 0.0f,
                    s.extrude ? roots[c * 3 + 1] - roots[centre * 3 + 1] : 0.0f,
                    s.extrude ? roots[c * 3 + 2] - roots[centre * 3 + 2] : 0.0f};
                for (size_t i = 0; i < n; ++i) {
                    size_t const cv = first + i;
                    float const t = hairT(cv, i, n);
                    // The centre sampled at the same parameter, for a centre with
                    // a different CV count.
                    float const at = cN > 1 ? t * float(cN - 1) : 0.0f;
                    size_t const lo = std::min(size_t(at), cN - 1);
                    size_t const hi = std::min(lo + 1, cN - 1);
                    float const f = at - float(lo);
                    float w = std::clamp(float(amountField.Value(c, cv)), 0.0f, 1.0f) * strength;
                    if (s.bezier)
                        w *= clumpRef::ClumpWeight(n > 1 ? float(i) / float(n - 1) : 0.0f,
                                                   lengths[c], 1.0f, s.controls);
                    else if (!profile.empty())
                        w *= UsdGenEvalLut257(profile.data(), t);
                    if (stray[c] > 0.0f)
                        w *= 1.0f - stray[c] * float(std::pow(double(t), s.strayFalloff));
                    if (level == 0 && flyaway[c]) w *= flyAmount;
                    w = std::clamp(w, 0.0f, 1.0f);
                    if (w == 0.0f) continue;
                    for (int d = 0; d < 3; ++d) {
                        float const target = before[(cFirst + lo) * 3 + d] +
                            (before[(cFirst + hi) * 3 + d] - before[(cFirst + lo) * 3 + d]) * f +
                            offset[d];
                        float &v = shape[cv * 3 + d];
                        v += (target - v) * w;
                    }
                }
            });
        }

        // --- level 0: flyaways, saved overrides, sub-clumps ----------------
        if (level == 0) finishLevel0(member);

        // --- next level's regions (dense ids of this level's clumps) -------
        if (level + 1 < s.levels) {
            std::vector<int32_t> keys;
            for (size_t c = 0; c < R; ++c)
                if (member[c] >= 0) keys.push_back(member[c]);
            std::sort(keys.begin(), keys.end());
            keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
            for (size_t c = 0; c < R; ++c)
                parent[c] = member[c] < 0 ? -1 : int32_t(std::lower_bound(
                    keys.begin(), keys.end(), member[c]) - keys.begin());
            parentCount = keys.size();
        }
    }

    // Restore the input segment lengths, root locked.
    ParallelFor(ctx.dispatcher, R, [&](size_t c) {
        size_t const first = spans[c], n = spans[c + 1] - spans[c];
        for (size_t i = 1; i < n; ++i) {
            size_t const cv = first + i;
            float const keep = std::clamp(float(preserveField.Value(c, cv)), 0.0f, 1.0f);
            if (keep == 0.0f) continue;
            GfVec3f const prev(shape[(cv - 1) * 3], shape[(cv - 1) * 3 + 1], shape[(cv - 1) * 3 + 2]);
            GfVec3f seg(shape[cv * 3] - prev[0], shape[cv * 3 + 1] - prev[1], shape[cv * 3 + 2] - prev[2]);
            GfVec3f const was(input[cv * 3] - input[(cv - 1) * 3],
                              input[cv * 3 + 1] - input[(cv - 1) * 3 + 1],
                              input[cv * 3 + 2] - input[(cv - 1) * 3 + 2]);
            float const now = seg.GetLength();
            if (now < 1e-9f) continue;
            float const length = now + (was.GetLength() - now) * keep;
            seg *= length / now;
            for (int d = 0; d < 3; ++d) shape[cv * 3 + d] = prev[d] + seg[d];
        }
    });
    cap.result = std::move(shape);
    return true;
}

void UsdGenClumpOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenClumpCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params;
    UsdGenParamField const maskField = p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const firstCv = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    bool const ready = cap.result.size() == size_t(cap.upstreamCvs) * 3;
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        size_t const curve = curveBase + c;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            size_t const cv = firstCv + o;
            float const in[3] = {view->inPx[o], view->inPy[o], view->inPz[o]};
            float const m = ready
                ? std::clamp(float(maskField.Value(curve, cv)), 0.0f, 1.0f) : 0.0f;
            if (m == 0.0f || cv * 3 + 2 >= cap.result.size()) {
                view->px[o] = in[0]; view->py[o] = in[1]; view->pz[o] = in[2];
                continue;
            }
            float const *r = &cap.result[cv * 3];
            view->px[o] = m == 1.0f ? r[0] : in[0] + (r[0] - in[0]) * m;
            view->py[o] = m == 1.0f ? r[1] : in[1] + (r[1] - in[1]) * m;
            view->pz[o] = m == 1.0f ? r[2] : in[2] + (r[2] - in[2]) * m;
        }
        for (uint32_t slot = 0; slot < view->outCount && slot < cap.clumpId.size(); ++slot) {
            if (!view->outI || !view->outI[slot]) continue;
            auto const &ids = cap.clumpId[slot];
            view->outI[slot][c] = curve < ids.size() ? ids[curve] : -1;
        }
    }
}

}  // namespace usdGen
