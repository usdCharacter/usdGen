// usdGen — UsdGenDirectionOp implementation. 02-schema.md §2.7.1, 04 §2.13.
// Combs every strand toward a shaped target direction (see ops/direction.h
// for the full contract). Capture validates literals, tokens and connected
// values and, when direction:source targets a curve set, bakes the
// nearest-sample tangent per strand root; Evaluate shapes the target in the
// root frame and rotates rigidly or per segment. Roots never move.
#include "usdGen/ops/direction.h"
#include "usdGen/scratch.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include <nanoflann.hpp>

#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/usdGenMath/ramp.h"

#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/diagnostic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenDirectionCapture final : public UsdGenCapture
{
    VtVec3fArray sourceDir;   // per strand, nearest-sample tangent; empty when unbound
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenDirectionCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves;
    }
};

struct Cloud {
    std::vector<float> points;   // xyz per direction:source sample
    size_t kdtree_get_point_count() const { return points.size() / 3; }
    float kdtree_get_pt(size_t i, size_t d) const { return points[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B &) const { return false; }
};
using CloudIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

bool Finite(double v) { return std::isfinite(v); }

VtVec2fArray ReadRampKnots(UsdGenParamView const *p, TfToken const &name)
{
    if (!p) return {};
    if (VtValue const v = p->GetVtValue(name, VtValue());
        v.IsHolding<VtVec2fArray>()) {
        return v.UncheckedGet<VtVec2fArray>();
    }
    return {};
}

inline GfVec3f Normalized(GfVec3f v, GfVec3f const &fallback)
{
    float const l = v.GetLength();
    return l > 1e-12f ? v / l : fallback;
}

constexpr float kPi = 3.14159265358979323846f;

// Rodrigues rotation of v about axis by radians. A zero-length axis is the
// identity (never a NaN); length is preserved up to float rounding.
inline GfVec3f RotateAxisAngle(GfVec3f const &v, GfVec3f const &axis, float radians)
{
    float const l = axis.GetLength();
    if (!(l > 1e-12f) || radians == 0.0f) return v;
    GfVec3f const a = axis / l;
    float const c = std::cos(radians), s = std::sin(radians);
    GfVec3f const cross(a[1] * v[2] - a[2] * v[1],
                        a[2] * v[0] - a[0] * v[2],
                        a[0] * v[1] - a[1] * v[0]);
    float const t = 1.0f - c, d = GfDot(a, v);
    return GfVec3f(v[0] * c + cross[0] * s + a[0] * d * t,
                   v[1] * c + cross[1] * s + a[1] * d * t,
                   v[2] * c + cross[2] * s + a[2] * d * t);
}

inline GfVec3f RotateAboutAxisDeg(GfVec3f const &v, GfVec3f const &axis, float degrees)
{
    if (degrees == 0.0f) return v;
    return RotateAxisAngle(v, axis, degrees * (kPi / 180.0f));
}

// Axis and angle of the minimal rotation taking unit a onto unit b. Aligned
// vectors give angle 0; opposed vectors pick a perpendicular axis and pi.
inline void RotationFromTo(GfVec3f const &a, GfVec3f const &b,
                           GfVec3f *axis, float *angle)
{
    GfVec3f const c = GfCross(a, b);
    float const s = c.GetLength(), d = GfDot(a, b);
    if (s < 1e-9f) {
        if (d > 0.0f) {
            *axis = GfVec3f(1.0f, 0.0f, 0.0f);
            *angle = 0.0f;
            return;
        }
        *axis = std::fabs(a[0]) < 0.9f
            ? Normalized(GfCross(a, GfVec3f(1.0f, 0.0f, 0.0f)), GfVec3f(0.0f, 1.0f, 0.0f))
            : Normalized(GfCross(a, GfVec3f(0.0f, 1.0f, 0.0f)), GfVec3f(1.0f, 0.0f, 0.0f));
        *angle = kPi;
        return;
    }
    *axis = c / s;
    *angle = std::atan2(s, d);
}

// v rotated by `amount` (clamped to [0,1]) of the minimal from->to angle.
inline GfVec3f RotateToward(GfVec3f const &v, GfVec3f const &from,
                            GfVec3f const &to, float amount)
{
    if (amount <= 0.0f) return v;
    if (amount > 1.0f) amount = 1.0f;
    GfVec3f axis(1.0f, 0.0f, 0.0f);
    float angle = 0.0f;
    RotationFromTo(from, to, &axis, &angle);
    if (angle == 0.0f) return v;
    return RotateAxisAngle(v, axis, angle * amount);
}

struct Mat3 {
    float m[3][3];
    static Mat3 Identity()
    {
        Mat3 r{};
        r.m[0][0] = r.m[1][1] = r.m[2][2] = 1.0f;
        return r;
    }
    // The Rodrigues matrix: (FromAxisAngle(a, t) * v) == RotateAxisAngle(v, a, t).
    static Mat3 FromAxisAngle(GfVec3f const &a, float radians)
    {
        float const c = std::cos(radians), s = std::sin(radians), t = 1.0f - c;
        float const x = a[0], y = a[1], z = a[2];
        Mat3 r{};
        r.m[0][0] = t * x * x + c;     r.m[0][1] = t * x * y - s * z; r.m[0][2] = t * x * z + s * y;
        r.m[1][0] = t * x * y + s * z; r.m[1][1] = t * y * y + c;     r.m[1][2] = t * y * z - s * x;
        r.m[2][0] = t * x * z - s * y; r.m[2][1] = t * y * z + s * x; r.m[2][2] = t * z * z + c;
        return r;
    }
    // Composition: (a * b) * v == a * (b * v).
    Mat3 operator*(Mat3 const &b) const
    {
        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r.m[i][j] = m[i][0] * b.m[0][j] + m[i][1] * b.m[1][j] +
                    m[i][2] * b.m[2][j];
        return r;
    }
    GfVec3f operator*(GfVec3f const &v) const
    {
        return GfVec3f(m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                       m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                       m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]);
    }
};

// The shaped unit target: tiltU/V/N and aroundN pre-rotations, the Grow-axis
// lift, then the followSkinContour blend toward the tangent-plane projection.
// The result is always unit: a zero-length base direction falls back to the
// root normal (the strand then combs toward N rather than emitting NaNs).
inline GfVec3f ShapeTarget(GfVec3f d, GfVec3f const &T, GfVec3f const &B,
                           GfVec3f const &N, float tiltU, float tiltV,
                           float tiltN, float aroundN, float lift, float follow)
{
    GfVec3f const fallback = Normalized(N, GfVec3f(0.0f, 0.0f, 1.0f));
    d = Normalized(d, fallback);
    if (tiltU != 0.0f) d = RotateAboutAxisDeg(d, T, tiltU);
    if (tiltV != 0.0f) d = RotateAboutAxisDeg(d, B, tiltV);
    if (tiltN != 0.0f) d = RotateAboutAxisDeg(d, N, tiltN);
    if (aroundN != 0.0f) d = RotateAboutAxisDeg(d, N, aroundN);
    if (lift != 0.0f) d = RotateAboutAxisDeg(d, B, lift);
    d = Normalized(d, fallback);
    if (follow != 0.0f) {
        GfVec3f const c = d - fallback * GfDot(d, fallback);
        if (c.GetLength() > 1e-12f)
            d = Normalized(d + (c / c.GetLength() - d) * follow, d);
    }
    return Normalized(d, fallback);
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("mode"));              // structural: kernel branch (02 §6.1)
    v.push_back(TfToken("direction:source"));  // structural: graph edge (02 §6.1)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));
    v.push_back(TfToken("direction"));
    v.push_back(TfToken("amount"));
    v.push_back(TfToken("lift"));
    v.push_back(TfToken("tiltU"));
    v.push_back(TfToken("tiltV"));
    v.push_back(TfToken("tiltN"));
    v.push_back(TfToken("aroundN"));
    v.push_back(TfToken("followSkinContour"));
    v.push_back(TfToken("direction:knots"));
    v.push_back(TfToken("direction:interpolation"));
    v.push_back(TfToken("mask"));
    return v;
}();

static TfTokenVector const _referenceInputs{TfToken("direction:source")};

TfSpan<const TfToken> UsdGenDirectionOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenDirectionOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
TfSpan<const TfToken> UsdGenDirectionOp::ReferenceInputs() const
{
    return TfSpan<const TfToken>(_referenceInputs.data(), _referenceInputs.size());
}
std::unique_ptr<UsdGenCapture> UsdGenDirectionOp::CreateCapture() const
{
    return std::make_unique<UsdGenDirectionCapture>();
}
uint32_t UsdGenDirectionOp::PlanesTouched() const
{
    return kPlanePoints;
}

bool UsdGenDirectionOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sMode, sRigid);
    if (mode != sRigid && mode != sPerSegment) {
        if (diag) diag->Error("UsdGenDirection: unknown usdGen:mode '" + mode.GetString() + "'");
        return false;
    }
    TfToken const interp = params.GetToken(sInterp, sCatmullRom);
    if (interp != sLinear && interp != sCatmullRom && interp != sBspline &&
        interp != sConstant) {
        if (diag)
            diag->Error("UsdGenDirection: unknown usdGen:direction:interpolation '" +
                        interp.GetString() + "'");
        return false;
    }
    double const amount = params.GetDoubleLiteral(sAmount, 0.0);
    if (!Finite(amount) || amount < 0.0 || amount > 1.0) {
        if (diag) diag->Error("UsdGenDirection: usdGen:amount must be finite and in [0, 1]");
        return false;
    }
    double const lift = params.GetDoubleLiteral(sLift, 0.0);
    if (!Finite(lift) || lift < -90.0 || lift > 90.0) {
        if (diag)
            diag->Error("UsdGenDirection: usdGen:lift must be finite and in [-90, 90] degrees");
        return false;
    }
    for (char const *name : {"tiltU", "tiltV", "tiltN", "aroundN"}) {
        if (!Finite(params.GetDoubleLiteral(TfToken(name), 0.0))) {
            if (diag)
                diag->Error(std::string("UsdGenDirection: usdGen:") + name + " must be finite");
            return false;
        }
    }
    double const follow = params.GetDoubleLiteral(sFollow, 0.0);
    if (!Finite(follow) || follow < 0.0 || follow > 1.0) {
        if (diag)
            diag->Error("UsdGenDirection: usdGen:followSkinContour must be finite and in [0, 1]");
        return false;
    }
    VtValue const direction = params.GetVtValue(sDirection, VtValue());
    if (!direction.IsEmpty() && !direction.IsHolding<GfVec3f>()) {
        if (diag) diag->Error("UsdGenDirection: usdGen:direction must be vector3f");
        return false;
    }
    if (direction.IsHolding<GfVec3f>()) {
        GfVec3f const v = direction.UncheckedGet<GfVec3f>();
        if (!Finite(v[0]) || !Finite(v[1]) || !Finite(v[2])) {
            if (diag) diag->Error("UsdGenDirection: usdGen:direction must be finite");
            return false;
        }
    }
    VtValue const knots = params.GetVtValue(sKnots, VtValue());
    if (!knots.IsEmpty() && !knots.IsHolding<VtVec2fArray>()) {
        if (diag) diag->Error("UsdGenDirection: usdGen:direction:knots must be float2[]");
        return false;
    }
    if (knots.IsHolding<VtVec2fArray>()) {
        for (GfVec2f const &knot : knots.UncheckedGet<VtVec2fArray>()) {
            if (!Finite(knot[0]) || !Finite(knot[1])) {
                if (diag) diag->Error("UsdGenDirection: usdGen:direction:knots must be finite");
                return false;
            }
        }
    }
    return true;
}

UsdGenEpoch UsdGenDirectionOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Structural tokens plus upstream generation plus seed only: value edits
    // (including connected-value edits, which the scheduler folds into the
    // capture identity separately, and the direction:source identity, which
    // arrives via the resolved reference values as it does for Part) sweep
    // without recapturing.
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](uint64_t v) { h ^= v; h *= 0x100000001b3ULL; };
    feed(p ? uint64_t(p->GetToken(sMode, sRigid).Hash()) : 0u);
    feed(ctx.upstreamGeneration);
    feed(uint64_t(ctx.seed));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenDirectionOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params;
    auto &cap = *static_cast<UsdGenDirectionCapture *>(out);
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error("UsdGenDirection: " + message);
        return false;
    };
    if (p && !Bind(*p, diag)) return false;
    for (uint32_t i = 0; i < ctx.mapBindingCount; ++i) {
        if (ctx.mapBindings && ctx.mapBindings[i].relationship == sSource)
            return fail("map-driven direction:source is not implemented by this kernel "
                        "revision; target a curve set or an operator instead");
    }
    std::vector<uint32_t> spans;
    std::string error;
    if (!opUtil::CurveSpans(upstream, &spans, &error))
        return fail("strand topology: " + error);
    size_t const R = upstream.totalCurves;
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.sourceDir.clear();
    // Literals were validated by Bind above; connected values (which the CPU
    // lane reads here, never through Bind) are validated per element, at the
    // same granularity Evaluate samples them: direction/amount/lift per curve
    // root, the mask per CV, the groom-wide tilts through their groom fold.
    UsdGenParamField const directionField =
        p ? p->GetScalarField(sDirection, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const amountField =
        p ? p->GetScalarField(sAmount, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const liftField =
        p ? p->GetScalarField(sLift, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    if (directionField.connected && directionField.components != 3)
        return fail("connected direction must be vector3f");
    if (p) {
        for (char const *name : {"tiltU", "tiltV", "tiltN", "aroundN"}) {
            if (!Finite(p->GetDouble(TfToken(name), 0.0)))
                return fail(std::string("groom usdGen:") + name + " must be finite");
        }
        if (!Finite(p->GetDouble(sFollow, 0.0)))
            return fail("groom usdGen:followSkinContour must be finite");
    }
    for (size_t c = 0; c < R; ++c) {
        uint32_t const root = spans[c];
        if (directionField.connected &&
            (!Finite(directionField.Value(c, root, 0)) ||
             !Finite(directionField.Value(c, root, 1)) ||
             !Finite(directionField.Value(c, root, 2))))
            return fail("per-strand direction must be finite");
        double const amount = amountField.Value(c, root);
        if (!Finite(amount) || amount < 0.0 || amount > 1.0)
            return fail("per-strand amount must be finite and in [0, 1]");
        double const lift = liftField.Value(c, root);
        if (!Finite(lift) || lift < -90.0 || lift > 90.0)
            return fail("per-strand lift must be finite and in [-90, 90] degrees");
        for (uint32_t cv = spans[c]; cv != spans[c + 1]; ++cv) {
            if (!Finite(maskField.Value(c, cv)))
                return fail("per-CV mask must be finite");
        }
    }
    if (ctx.referenceCount == 0) return true;
    if (ctx.referenceCount != 1 || !ctx.resolvedReferences ||
        !ctx.resolvedReferences[0] || !ctx.resolvedReferences[0]->value)
        return fail("usdGen:direction:source must target exactly one curve set");
    UsdGenCurveBuffer const &guides = ctx.resolvedReferences[0]->value->buffer;
    size_t const M = guides.totalCurves;
    if (M == 0) return fail("the direction:source curve set has no curves");
    std::vector<uint32_t> guideSpans;
    if (!opUtil::CurveSpans(guides, &guideSpans, &error))
        return fail("direction:source topology: " + error);
    // Nearest-sample tangents on rest positions, Part-style: the per-strand
    // base target when a curve set is bound.
    size_t const nSamples = guides.totalCvs;
    Cloud cloud;
    cloud.points.resize(nSamples * 3);
    std::vector<float> tangents(nSamples * 3, 0.0f);
    for (size_t m = 0; m < M; ++m) {
        size_t const first = guideSpans[m], count = guideSpans[m + 1] - first;
        for (size_t j = 0; j < count; ++j) {
            size_t const s = first + j;
            GfVec3f const pos = opUtil::RestPoint(guides, s);
            cloud.points[s * 3] = pos[0];
            cloud.points[s * 3 + 1] = pos[1];
            cloud.points[s * 3 + 2] = pos[2];
            size_t const a = first + (j > 0 ? j - 1 : 0);
            size_t const b = first + (j + 1 < count ? j + 1 : j);
            GfVec3f t = opUtil::RestPoint(guides, b) - opUtil::RestPoint(guides, a);
            float const l = t.GetLength();
            if (l <= 1e-12f) t = GfVec3f(0.0f, 1.0f, 0.0f);
            else t /= l;
            tangents[s * 3] = t[0];
            tangents[s * 3 + 1] = t[1];
            tangents[s * 3 + 2] = t[2];
        }
    }
    CloudIndex index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    cap.sourceDir.resize(R);
    for (size_t c = 0; c < R; ++c) {
        GfVec3f const rootPos = opUtil::RestPoint(upstream, spans[c]);
        float const query[3] = {rootPos[0], rootPos[1], rootPos[2]};
        uint32_t found = 0;
        float distSq = 0.0f;
        if (!index.knnSearch(query, 1, &found, &distSq) || found >= nSamples)
            return fail("direction:source query failed");
        cap.sourceDir[c] = GfVec3f(tangents[size_t(found) * 3],
                                   tangents[size_t(found) * 3 + 1],
                                   tangents[size_t(found) * 3 + 2]);
    }
    return true;
}

void UsdGenDirectionOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenDirectionCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const directionField =
        p ? p->GetScalarField(sDirection, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const amountField =
        p ? p->GetScalarField(sAmount, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const liftField =
        p ? p->GetScalarField(sLift, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    GfVec3f litDirection(0.0f, 1.0f, 0.0f);
    double tiltU = 0.0, tiltV = 0.0, tiltN = 0.0, aroundN = 0.0, follow = 0.0;
    bool perSegment = false;
    if (p) {
        VtValue const v = p->GetVtValue(sDirection, VtValue());
        if (v.IsHolding<GfVec3f>()) litDirection = v.UncheckedGet<GfVec3f>();
        tiltU = p->GetDouble(sTiltU, 0.0);
        tiltV = p->GetDouble(sTiltV, 0.0);
        tiltN = p->GetDouble(sTiltN, 0.0);
        aroundN = p->GetDouble(sAroundN, 0.0);
        follow = std::clamp(p->GetDouble(sFollow, 0.0), 0.0, 1.0);
        perSegment = p->GetToken(sMode, sRigid) == sPerSegment;
    }
    bool const sourceBound = !cap.sourceDir.empty();
    float const *lutPtr = nullptr;
    VtVec2fArray const knots = ReadRampKnots(p, sKnots);
    bool const rampFlat = knots.empty();
    UsdGenScratchScope scratch;   // chunk-local; rewinds when Evaluate returns
    if (!rampFlat) {
        float *lut = scratch.Filled<float>(kUsdGenRampLutSize, 1.0f);
        UsdGenBuildRampLut(knots,
            p ? p->GetToken(sInterp, sCatmullRom) : sCatmullRom,
            lut, kUsdGenRampLutSize);
        lutPtr = lut;
    }
    float *px = view->px, *py = view->py, *pz = view->pz;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *hairT = view->hairT;
    auto const *rootT = view->rootT, *rootN = view->rootN, *rootB = view->rootB;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    GfVec3f const fallbackT(1.0f, 0.0f, 0.0f), fallbackB(0.0f, 1.0f, 0.0f),
        fallbackN(0.0f, 0.0f, 1.0f);

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        size_t const root = cvBase + g;
        float const amount = static_cast<float>(amountField.Value(curve, root));
        float const lift = static_cast<float>(liftField.Value(curve, root));
        GfVec3f base;
        if (sourceBound && curve < cap.sourceDir.size()) {
            base = cap.sourceDir[curve];
        } else if (directionField.connected) {
            base = GfVec3f(float(directionField.Value(curve, root, 0)),
                           float(directionField.Value(curve, root, 1)),
                           float(directionField.Value(curve, root, 2)));
        } else {
            base = litDirection;
        }
        GfVec3f const T = rootT ? rootT[c] : fallbackT;
        GfVec3f const B = rootB ? rootB[c] : fallbackB;
        GfVec3f const N = rootN ? rootN[c] : fallbackN;
        GfVec3f const target = ShapeTarget(base, T, B, N, float(tiltU), float(tiltV),
                                           float(tiltN), float(aroundN), lift,
                                           float(follow));
        auto rampAt = [&](size_t i) {
            float const t = hairT ? hairT[g + i]
                : (n > 1 ? float(i) / float(n - 1) : 0.0f);
            return rampFlat ? 1.0f : UsdGenEvalLut257(lutPtr, t);
        };
        auto maskAt = [&](size_t i) {
            return std::clamp(
                static_cast<float>(maskField.Value(curve, cvBase + g + i)), 0.0f, 1.0f);
        };
        GfVec3f const rootPos(inPx[g], inPy[g], inPz[g]);
        px[g] = inPx[g]; py[g] = inPy[g]; pz[g] = inPz[g];
        if (n < 2) continue;
        if (amount == 0.0f) {
            for (size_t i = 1; i < n; ++i) {
                px[g + i] = inPx[g + i]; py[g + i] = inPy[g + i]; pz[g + i] = inPz[g + i];
            }
            continue;
        }
        if (!perSegment) {
            // Rigid: every CV offset turns about the root by w_i of the
            // root-segment->target angle. A zero-length root segment has no
            // frame, so the strand keeps its input shape.
            GfVec3f const from = Normalized(
                GfVec3f(inPx[g + 1] - inPx[g], inPy[g + 1] - inPy[g],
                        inPz[g + 1] - inPz[g]),
                GfVec3f(0.0f));
            if (from == GfVec3f(0.0f)) {
                for (size_t i = 1; i < n; ++i) {
                    px[g + i] = inPx[g + i];
                    py[g + i] = inPy[g + i];
                    pz[g + i] = inPz[g + i];
                }
                continue;
            }
            for (size_t i = 1; i < n; ++i) {
                size_t const o = g + i;
                float const w = amount * rampAt(i) * maskAt(i);
                if (w == 0.0f) {
                    px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                    continue;
                }
                GfVec3f const turned = RotateToward(
                    GfVec3f(inPx[o] - rootPos[0], inPy[o] - rootPos[1],
                            inPz[o] - rootPos[2]),
                    from, target, w);
                px[o] = rootPos[0] + turned[0];
                py[o] = rootPos[1] + turned[1];
                pz[o] = rootPos[2] + turned[2];
            }
            continue;
        }
        // Per-segment: accumulate the turn down the strand Bend-style. While
        // the cumulative rotation is still the identity, unturned segments
        // copy the input bitwise.
        Mat3 cumul = Mat3::Identity();
        bool identQ = true;
        float prevR = 0.0f;   // R(t_{-1}) := 0, so a flat ramp turns at the root
        for (size_t k = 0; k + 1 < n; ++k) {
            float const rK = rampAt(k);
            float w = amount * (rK - prevR) * maskAt(k);
            prevR = rK;
            size_t const o = g + k, q = g + k + 1;
            GfVec3f const seg(inPx[q] - inPx[o], inPy[q] - inPy[o], inPz[q] - inPz[o]);
            if (seg.GetLength() <= 1e-12f) {
                if (identQ) {
                    px[q] = inPx[q]; py[q] = inPy[q]; pz[q] = inPz[q];
                } else {
                    px[q] = px[o]; py[q] = py[o]; pz[q] = pz[o];
                }
                continue;
            }
            w = std::clamp(w, -1.0f, 1.0f);
            if (w == 0.0f && identQ) {
                px[q] = inPx[q]; py[q] = inPy[q]; pz[q] = inPz[q];
                continue;
            }
            if (w != 0.0f) {
                GfVec3f const d = Normalized(cumul * seg, GfVec3f(0.0f));
                if (d != GfVec3f(0.0f)) {
                    GfVec3f axis(1.0f, 0.0f, 0.0f);
                    float angle = 0.0f;
                    RotationFromTo(d, target, &axis, &angle);
                    if (angle != 0.0f) {
                        cumul = Mat3::FromAxisAngle(axis, angle * w) * cumul;
                        identQ = false;
                    }
                }
            }
            GfVec3f const turned = cumul * seg;
            px[q] = px[o] + turned[0];
            py[q] = py[o] + turned[1];
            pz[q] = pz[o] + turned[2];
        }
    }
}

UsdGenOpId UsdGenRegisterDirectionOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenDirectionOp().Type(), [] { return std::make_unique<UsdGenDirectionOp>(); });
}

}  // namespace usdGen
