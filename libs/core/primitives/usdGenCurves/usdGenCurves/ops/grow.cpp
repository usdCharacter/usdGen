// usdGen — UsdGenGrowOp implementation (M1). 02-schema.md §2.6, 04 §2.2.
// Roots -> straight strands: sets the CV count (usdGen:segments), the length
// and root-frame-B lift followed by azimuth around root-frame N.
// Capture fixes the topology (cvCount) and the
// per-curve target length (kSaltGrow draw); Evaluate writes the CV
// positions and hairT. Direction: surfaceNormal (root frame N), attribute
// (M1: root frame T), vector (usdGen:directionVector).
#include "usdGen/ops/grow.h"
#include "usdGenCurves/types/curveCapture.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenSurface/types/rootSet.h"

#include "usdGenMath/usdGenMath/hash.h"
#include "usdGenMath/usdGenMath/kernels.h"
#include "usdGenMath/math.h"
#include "usdGenMath/math.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "usdGen/registries/registrar.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"


namespace usdGen {

namespace {

// usdGen:lengthRandom is a float2 (lo, hi); the property is stored as a
// UsdGenVec2f or a len-2 UsdGenFloatArray depending on the source (02 §2.6).
UsdGenValue const kNoValue;

inline void ReadVec2Prop(UsdGenParamView const &p, UsdGenParamSlot slot,
                         double &lo, double &hi)
{
    lo = 1.0; hi = 1.0;
    if (UsdGenValue const &v = p.GetValue(slot, kNoValue); v.IsHolding<UsdGenFloatArray>()) {
        UsdGenFloatArray const a = v.UncheckedGet<UsdGenFloatArray>();
        if (a.size() >= 2) { lo = a[0]; hi = a[1]; }
    } else if (v.IsHolding<UsdGenVec2f>()) {
        UsdGenVec2f const r = v.UncheckedGet<UsdGenVec2f>();
        lo = r[0]; hi = r[1];
    }
}

inline uint64_t doubleAsBits(double d)
{
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}

inline uint32_t floatAsBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

inline UsdGenVec3f NormalizeGrowDirection(UsdGenVec3f v)
{
    float const l2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    float const length = std::sqrt(l2);
    return length > 1.0e-12f ? v / length : UsdGenVec3f(0.0f, 1.0f, 0.0f);
}

// plan/04 §2.2: lift is an angle in degrees about root-frame B.  Rodrigues'
// form keeps the CPU and CUDA lowerings on the same float arithmetic path.
inline UsdGenVec3f RotateGrowDirection(UsdGenVec3f direction, UsdGenVec3f axis,
                                   float degrees)
{
    if (degrees == 0.0f) return direction;
    axis = NormalizeGrowDirection(axis);
    constexpr float pi = 3.14159265358979323846f;
    float const radians = degrees * (pi / 180.0f);
    float const c = std::cos(radians);
    float const s = std::sin(radians);
    float const dot = axis[0] * direction[0] + axis[1] * direction[1] +
        axis[2] * direction[2];
    UsdGenVec3f const cross(
        axis[1] * direction[2] - axis[2] * direction[1],
        axis[2] * direction[0] - axis[0] * direction[2],
        axis[0] * direction[1] - axis[1] * direction[0]);
    float const oneMinusC = 1.0f - c;
    return UsdGenVec3f(
        direction[0] * c + cross[0] * s + axis[0] * dot * oneMinusC,
        direction[1] * c + cross[1] * s + axis[1] * dot * oneMinusC,
        direction[2] * c + cross[2] * s + axis[2] * dot * oneMinusC);
}

// Azimuth is captured once per stable strand, independently of length draws.
struct UsdGenGrowCapture final : public UsdGenCapturePayload
{
    // perCurve[c] = target length of curve c (kSaltGrow draw applied).
    // liftPerCurve/dirPerCurve hold the connected per-strand lift (degrees)
    // and directionVector; empty when those controls are plain literals, in
    // which case the scalar lift/directionVector below apply to every curve.
    int cvCount = 0;
    float lift = 0.0f;
    UsdGenToken direction;   // empty until Capture; anything but "vector" grows along the normal
    UsdGenVec3f directionVector{0.0f, 1.0f, 0.0f};
    UsdGenFloatArray liftPerCurve;
    UsdGenFloatArray azimuthPerCurve;
    UsdGenVec3fArray dirPerCurve;
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenGrowCapture>(*this);
    }
    bool OwnsBuffer() const override { return true; }
    bool Recyclable() const override { return true; }
    bool Recycle() override
    {
        UsdGenRecycleBuffer(_buffer);
        perCurve.clear();
        perCv.clear();
        rampLut.clear();
        cvCount = 0;
        lift = 0.0f;
        direction = UsdGenToken();
        directionVector = UsdGenVec3f(0.0f, 1.0f, 0.0f);
        liftPerCurve.clear();
        azimuthPerCurve.clear();
        dirPerCurve.clear();
        upstreamTopologyVersion = upstreamValueVersion = 0;
        upstreamCurves = 0;
        return true;
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves;
    }
};

}  // namespace

namespace {
using namespace growP;
using D = expr::Domain;

constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kIntGroom{expr::ScalarType::Int32, 1, "int", D::Groom};
constexpr UsdGenExprAdmission kFloatRoot{expr::ScalarType::Float32, 1, "float", D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kFloat2Root{expr::ScalarType::Float32, 2, "float2", D::Groom | D::Primitive};
constexpr UsdGenExprAdmission kVector3Root{expr::ScalarType::Float32, 3, "vector3f", D::Groom | D::Primitive};
constexpr char const *kDirections[] = {"surfaceNormal", "vector"};

// A generator: no upstream curves to leave untouched, no mask. Order is the old
// TopologyParameters() then ValueParameters() list.
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Capture, true, kBoolGroom},
    {"segments", UsdGenParamClass::Structural, true, kIntGroom},
    {"direction", UsdGenParamClass::Structural, true, {}, UsdGenSpan(kDirections)},
    {"length", UsdGenParamClass::Capture, true, kFloatRoot},
    {"lengthRandom", UsdGenParamClass::Capture, true, kFloat2Root},
    {"directionVector", UsdGenParamClass::Value, true, kVector3Root},
    {"lift", UsdGenParamClass::Value, true, kFloatRoot},
    {"azimuth", UsdGenParamClass::Value, true, kFloatRoot},
    {"azimuthRandom", UsdGenParamClass::Value, true, kFloatRoot},
};
constexpr char const *kTokens[] = {"surfaceNormal", "vector"};
// Roots (Scatter), or curves whose first CVs are the roots; filled at registration.
char const *gInputTypes[2] = {nullptr, nullptr};

std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &record, void *)
{
    return std::make_unique<UsdGenGrowOp>(record);
}
}  // namespace

std::unique_ptr<UsdGenCapture> UsdGenGrowOp::CreateCapture() const
{
    return std::make_unique<UsdGenGrowCapture>();
}
uint32_t UsdGenGrowOp::PlanesTouched() const
{
    // Capture materializes a fresh width owner because Grow changes the CV
    // layout.  Mark it written so generic preparation does not replace that
    // transformed owner with an incompatible upstream alias.
    return kPlanePoints | kPlaneWidths | kPlaneHairT;
}

bool UsdGenGrowOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    int segments = params.GetInt(kSegments, 8);
    if (segments < 2 || segments > int(kUsdGenMaxCvsPerCurve)) {
        if (diag)
            diag->Error("UsdGenGrow: usdGen:segments must be in [2, 4096]");
        return false;
    }
    UsdGenToken const &dir = params.GetToken(kDirection, Record().Token(kTokSurfaceNormal));
    if (dir != Record().Token(kTokSurfaceNormal) && dir != Record().Token(kTokVector)) {
        if (diag)
            diag->Error("UsdGenGrow: unknown usdGen:direction token '" +
                        dir.GetString() + "'");
        return false;
    }
    double const lift = params.GetDouble(kLift, 0.0);
    if (!std::isfinite(lift) || lift < -90.0 || lift > 90.0) {
        if (diag)
            diag->Error("UsdGenGrow: usdGen:lift must be finite and in [-90, 90] degrees");
        return false;
    }
    double const azimuth = params.GetDouble(kAzimuth, 0.0);
    double const azimuthRandom = params.GetDouble(kAzimuthRandom, 0.0);
    if (!std::isfinite(azimuth) || azimuth < -360.0 || azimuth > 360.0 ||
        !std::isfinite(azimuthRandom) || azimuthRandom < 0.0 || azimuthRandom > 1.0) {
        if (diag) diag->Error("UsdGenGrow: azimuth must be in [-360,360] degrees and azimuthRandom in [0,1]");
        return false;
    }
    UsdGenValue const &directionValue = params.GetValue(kDirectionVector, kNoValue);
    if (!directionValue.IsEmpty() && !directionValue.IsHolding<UsdGenVec3f>()) {
        if (diag)
            diag->Error("UsdGenGrow: usdGen:directionVector must be vector3f");
        return false;
    }
    if (directionValue.IsHolding<UsdGenVec3f>()) {
        UsdGenVec3f const v = directionValue.UncheckedGet<UsdGenVec3f>();
        if (!std::isfinite(v[0]) || !std::isfinite(v[1]) ||
            !std::isfinite(v[2])) {
            if (diag)
                diag->Error("UsdGenGrow: usdGen:directionVector must be finite");
            return false;
        }
    }
    return true;
}

UsdGenEpoch UsdGenGrowOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // Parameter-class inputs only; the upstream topology validity is checked
    // through ValidForTopology() at commit time (03 §3.4).
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    uint64_t h = 1469598103934665603ULL;
    auto feed = [&](std::string_view k, uint64_t v) {
        h ^= v; h *= 0x100000001b3ULL;
        h ^= k.size(); h *= 0x100000001b3ULL;
    };
    if (p) {
        feed("segments", uint64_t(p->GetInt(kSegments, 8)));
        feed("length", doubleAsBits(p->GetDouble(kLength, 1.0)));
        double lo = 1.0, hi = 1.0;
        ReadVec2Prop(*p, kLengthRandom, lo, hi);
        feed("lengthRandomLo", doubleAsBits(lo));
        feed("lengthRandomHi", doubleAsBits(hi));
        feed("direction",
             uint64_t(p->GetToken(kDirection, Record().Token(kTokSurfaceNormal)).Hash()));
        feed("lift", doubleAsBits(p->GetDouble(kLift, 0.0)));
        feed("azimuth", doubleAsBits(p->GetDouble(kAzimuth, 0.0)));
        feed("azimuthRandom", doubleAsBits(p->GetDouble(kAzimuthRandom, 0.0)));
        UsdGenVec3f directionVector(0.0f, 1.0f, 0.0f);
        if (UsdGenValue const &v = p->GetValue(kDirectionVector, kNoValue);
            v.IsHolding<UsdGenVec3f>())
            directionVector = v.UncheckedGet<UsdGenVec3f>();
        feed("directionVector.x", floatAsBits(directionVector[0]));
        feed("directionVector.y", floatAsBits(directionVector[1]));
        feed("directionVector.z", floatAsBits(directionVector[2]));
        feed("seed", ctx.seed);
    }
    // Grow captures the fallback only when its input has no width plane, but
    // including it unconditionally is cheap and prevents a stale capture if
    // a description edit adds/removes the inherited plane at the same time.
    feed("defaultWidth", doubleAsBits(ctx.desc ? ctx.desc->defaultWidth : 0.01));
    return {h, h ^ 0x9E3779B97F4A7C15ull};
}

bool UsdGenGrowOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenOpInputs const &in,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    auto &cap = *static_cast<UsdGenGrowCapture *>(out);

    int segments = p ? p->GetInt(kSegments, 8) : 8;
    if (segments < 2 || segments > int(kUsdGenMaxCvsPerCurve)) {
        if (diag) diag->Error("UsdGenGrow: usdGen:segments must be in [2, 4096]");
        return false;
    }
    double const lift = p ? p->GetDouble(kLift, 0.0) : 0.0;
    if (!std::isfinite(lift) || lift < -90.0 || lift > 90.0) {
        if (diag) diag->Error("UsdGenGrow: usdGen:lift must be finite and in [-90, 90] degrees");
        return false;
    }
    UsdGenVec3f directionVector(0.0f, 1.0f, 0.0f);
    if (p) {
        UsdGenValue const &v = p->GetValue(kDirectionVector, kNoValue);
        if (!v.IsEmpty() && !v.IsHolding<UsdGenVec3f>()) {
            if (diag) diag->Error("UsdGenGrow: usdGen:directionVector must be vector3f");
            return false;
        }
        if (v.IsHolding<UsdGenVec3f>()) directionVector = v.UncheckedGet<UsdGenVec3f>();
    }
    if (!std::isfinite(directionVector[0]) || !std::isfinite(directionVector[1]) ||
        !std::isfinite(directionVector[2])) {
        if (diag) diag->Error("UsdGenGrow: usdGen:directionVector must be finite");
        return false;
    }
    // A description-supplied fallback is authored data, not a best-effort
    // hint.  Reject invalid values before changing this capture's buffer so
    // the scheduler can retain the last-good generation.
    float const fallbackWidth = ctx.desc ? ctx.desc->defaultWidth : 0.01f;
    if (!std::isfinite(fallbackWidth) || fallbackWidth < 0.0f) {
        if (diag) diag->Error("UsdGenGrow: description defaultWidth must be "
                              "finite and >= 0");
        return false;
    }
    cap.cvCount = segments;
    cap.lift = static_cast<float>(lift);
    cap.direction = p ? p->GetToken(kDirection, Record().Token(kTokSurfaceNormal))
                      : Record().Token(kTokSurfaceNormal);
    cap.directionVector = directionVector;
    double length = p ? p->GetDouble(kLength, 1.0) : 1.0;
    double lo = 1.0, hi = 1.0;
    if (p) ReadVec2Prop(*p, kLengthRandom, lo, hi);
    if (lo > hi) std::swap(lo, hi);

    if (!std::isfinite(length) || !std::isfinite(lo) || !std::isfinite(hi)) {
        if (diag) diag->Error("UsdGenGrow: usdGen:length and lengthRandom must be finite");
        return false;
    }
    // Connected per-strand controls sample each curve's root below; the
    // literals above remain the $value fallback. GetDouble already folds a
    // groom-domain expression into `length`/`lift`, so the fields below agree
    // with the literals unless a primitive-domain expression is connected.
    UsdGenParamField const lengthField =
        p ? p->GetScalarField(kLength, length) : UsdGenParamField{length};
    UsdGenParamField const liftField =
        p ? p->GetScalarField(kLift, lift) : UsdGenParamField{lift};
    UsdGenParamField const randomField =
        p ? p->GetScalarField(kLengthRandom, 1.0) : UsdGenParamField{1.0};
    UsdGenParamField const dirField =
        p ? p->GetScalarField(kDirectionVector, 0.0) : UsdGenParamField{0.0};
    if ((randomField.connected && randomField.components != 2) ||
        (dirField.connected && dirField.components != 3)) {
        if (diag) diag->Error("UsdGenGrow: connected lengthRandom must be float2 and "
                              "directionVector must be vector3f");
        return false;
    }

    uint32_t const R = upstream.totalCurves;

    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;

    UsdGenCurveBuffer &buf = cap.MutableBuffer();
    if (R > std::numeric_limits<uint32_t>::max() /
                static_cast<uint32_t>(cap.cvCount)) {
        if (diag) diag->Error("UsdGenGrow: CV topology exceeds uint32 cardinality");
        return false;
    }
    buf.totalCurves = R;
    buf.totalCvs = R * uint32_t(cap.cvCount);
    buf.topologyVersion = upstream.topologyVersion + 1;
    buf.cvOffsets.clear();  // fixed cvCount: Grow's output is uniform.

    // A CV-topology producer cannot retain the old width owner: it either
    // resamples a valid vertex-width plane to the new span or creates the
    // description fallback.  This happens during capture, before workers
    // receive raw output pointers, and is deliberately private storage.
    // Refills a recycled capture's width storage (fresh when shared).
    UsdGenFloatArray grownWidths;
    grownWidths.swap(buf.width);
    grownWidths.assign(buf.totalCvs, fallbackWidth);
    if (!upstream.width.empty()) {
        if (upstream.width.size() != upstream.totalCvs) {
            if (diag) diag->Error("UsdGenGrow: upstream width plane cardinality "
                                  "does not match its CV topology");
            return false;
        }
        UsdGenPoolVector<uint32_t> sourceSpans;   // recycled storage, no heap once warm
        if (upstream.cvOffsets.empty()) {
            if (R && upstream.totalCvs % R != 0) {
                if (diag) diag->Error("UsdGenGrow: upstream width plane has "
                                      "non-uniform CV topology without offsets");
                return false;
            }
            sourceSpans.resize(size_t(R) + 1);
            uint32_t const perCurve = R ? upstream.totalCvs / R : 0;
            for (uint32_t c = 0; c != R; ++c)
                sourceSpans[c + 1] = sourceSpans[c] + perCurve;
        } else {
            if (upstream.cvOffsets.size() != size_t(R) + 1 ||
                upstream.cvOffsets.front() != 0 ||
                upstream.cvOffsets.back() != static_cast<int>(upstream.totalCvs)) {
                if (diag) diag->Error("UsdGenGrow: upstream width plane has "
                                      "invalid ragged CV offsets");
                return false;
            }
            sourceSpans.resize(size_t(R) + 1);
            for (uint32_t c = 0; c != R; ++c) {
                int const first = upstream.cvOffsets[c];
                int const last = upstream.cvOffsets[c + 1];
                if (first < 0 || last < first) {
                    if (diag) diag->Error("UsdGenGrow: upstream width plane has "
                                          "decreasing ragged CV offsets");
                    return false;
                }
                sourceSpans[c] = static_cast<uint32_t>(first);
                sourceSpans[c + 1] = static_cast<uint32_t>(last);
            }
        }
        for (uint32_t c = 0; c != R; ++c) {
            uint32_t const sourceFirst = sourceSpans[c];
            uint32_t const sourceCount = sourceSpans[c + 1] - sourceFirst;
            if (sourceCount == 0) {
                if (diag) diag->Error("UsdGenGrow: cannot resample a zero-CV "
                                      "upstream width curve");
                return false;
            }
            for (uint32_t i = 0; i != static_cast<uint32_t>(cap.cvCount); ++i) {
                float const u = cap.cvCount > 1
                    ? float(i) / float(cap.cvCount - 1) : 0.0f;
                float const sourcePosition = u * float(sourceCount - 1);
                // Large spans may round the float endpoint above the last
                // valid index, including to 2^32. Clamp before converting.
                uint32_t const loIndex = sourcePosition >= float(sourceCount - 1)
                    ? sourceCount - 1 : static_cast<uint32_t>(sourcePosition);
                uint32_t const hiIndex = std::min(loIndex + 1, sourceCount - 1);
                float const alpha = sourcePosition - float(loIndex);
                float const loValue = upstream.width[sourceFirst + loIndex];
                float const hiValue = upstream.width[sourceFirst + hiIndex];
                grownWidths[size_t(c) * cap.cvCount + i] =
                    loValue + (hiValue - loValue) * alpha;
            }
        }
    }
    buf.width = std::move(grownWidths);

    // Grow changes every curve's CV span. Named vertex planes therefore need
    // a fresh resample and named uniform/constant planes need fresh copies;
    // retaining the old VtArray handles would pair the new topology with the
    // prior cardinality and let later writes detach the wrong owner.
    std::string planeError;
    if (!UsdGenResampleExtraPlanes(upstream, &buf, &planeError)) {
        if (diag) diag->Error("UsdGenGrow: cannot preserve named planes across "
                              "CV topology change: " + planeError);
        return false;
    }

    // The per-curve loop samples connected controls at each curve's root CV,
    // so the upstream point topology is established here, before it.
    if (upstream.px.size() != upstream.totalCvs || upstream.py.size() != upstream.totalCvs ||
        upstream.pz.size() != upstream.totalCvs) {
        if (diag) diag->Error("UsdGenGrow: upstream point plane cardinality does not match its CV topology");
        return false;
    }
    UsdGenPoolVector<uint32_t> rootSpans(size_t(R) + 1, 0);
    if (upstream.cvOffsets.empty()) {
        if (R && upstream.totalCvs % R != 0) {
            if (diag) diag->Error("UsdGenGrow: upstream CV topology is non-uniform without offsets");
            return false;
        }
        uint32_t const perCurve = R ? upstream.totalCvs / R : 0;
        for (uint32_t c = 0; c != R; ++c) rootSpans[c + 1] = rootSpans[c] + perCurve;
    } else {
        if (upstream.cvOffsets.size() != size_t(R) + 1 || upstream.cvOffsets.front() != 0 ||
            upstream.cvOffsets.back() != static_cast<int>(upstream.totalCvs)) {
            if (diag) diag->Error("UsdGenGrow: upstream CV topology has invalid ragged offsets");
            return false;
        }
        for (uint32_t c = 0; c != R; ++c) {
            int const first = upstream.cvOffsets[c], last = upstream.cvOffsets[c + 1];
            if (first < 0 || last <= first) {
                if (diag) diag->Error("UsdGenGrow: upstream CV topology has an empty/decreasing curve");
                return false;
            }
            rootSpans[c] = static_cast<uint32_t>(first);
            rootSpans[c + 1] = static_cast<uint32_t>(last);
        }
    }

    // Per-curve target lengths: len = length_c * (lo_c + (hi_c-lo_c) * Draw01),
    // where the _c values are the connected per-strand controls sampled at the
    // curve's root, or the literals when nothing is connected. Lift and the
    // direction vector bake the same way so the rest channel below and
    // Evaluate agree without re-reading expressions per frame.
    cap.perCurve.clear();
    cap.liftPerCurve.clear();
    cap.azimuthPerCurve.clear();
    cap.dirPerCurve.clear();
    UsdGenParamField const azimuthField = p
        ? p->GetScalarField(kAzimuth, 0.0) : UsdGenParamField{0.0};
    UsdGenParamField const azimuthRandomField = p
        ? p->GetScalarField(kAzimuthRandom, 0.0) : UsdGenParamField{0.0};
    // Validate literal controls even for an empty groom. Bind also checks
    // these, but direct Capture callers must fail closed independently.
    double const azimuth = p ? p->GetDouble(kAzimuth, 0.0) : 0.0;
    double const azimuthRandom = p ? p->GetDouble(kAzimuthRandom, 0.0) : 0.0;
    if (!std::isfinite(azimuth) || azimuth < -360.0 || azimuth > 360.0 ||
        !std::isfinite(azimuthRandom) || azimuthRandom < 0.0 || azimuthRandom > 1.0) {
        if (diag) diag->Error("UsdGenGrow: azimuth must be in [-360,360] degrees and azimuthRandom in [0,1]");
        return false;
    }
    if (R) {
        cap.perCurve.resize(R);
        cap.azimuthPerCurve.resize(R);
        if (liftField.connected) cap.liftPerCurve.resize(R);
        if (dirField.connected) cap.dirPerCurve.resize(R);
        auto const *ids = upstream.curveId.empty() ? nullptr : upstream.curveId.data();
        auto *len = cap.perCurve.data();
        for (uint32_t c = 0; c < R; ++c) {
            uint32_t const root = rootSpans[c];
            double const azimuthC = azimuthField.Value(c, root);
            double const azimuthRandomC = azimuthRandomField.Value(c, root);
            if (!std::isfinite(azimuthC) || azimuthC < -360.0 || azimuthC > 360.0 ||
                !std::isfinite(azimuthRandomC) || azimuthRandomC < 0.0 || azimuthRandomC > 1.0) {
                if (diag) diag->Error("UsdGenGrow: per-strand azimuth must be in [-360,360] degrees and azimuthRandom in [0,1]");
                return false;
            }
            cap.azimuthPerCurve[c] = static_cast<float>(azimuthC) +
                static_cast<float>(azimuthRandomC) * 360.0f *
                (UsdGenDraw01(int(ctx.seed), ids ? ids[c] : 0, kSaltGrowAzimuth) - 0.5f);
            double const lengthC = lengthField.Value(c, root);
            double loC = lo, hiC = hi;
            if (randomField.connected) {
                loC = randomField.Value(c, root, 0);
                hiC = randomField.Value(c, root, 1);
                if (loC > hiC) std::swap(loC, hiC);
            }
            if (!std::isfinite(lengthC) || lengthC < 0.0 ||
                !std::isfinite(loC) || loC < 0.0 || !std::isfinite(hiC) || hiC < 0.0) {
                if (diag) diag->Error("UsdGenGrow: per-strand length and lengthRandom "
                                      "must be finite and >= 0");
                return false;
            }
            float const r = UsdGenDraw01(int(ctx.seed), ids ? ids[c] : 0,
                                         kSaltGrow);
            double const random = loC + static_cast<double>(r) * (hiC - loC);
            double const target = lengthC * random;
            float const targetFloat = static_cast<float>(target);
            if (!std::isfinite(target) || target < 0.0 ||
                !std::isfinite(targetFloat) || targetFloat < 0.0f) {
                if (diag) diag->Error("UsdGenGrow: length target must be finite and non-negative");
                return false;
            }
            len[c] = targetFloat;
            if (liftField.connected) {
                double const liftC = liftField.Value(c, root);
                if (!std::isfinite(liftC) || liftC < -90.0 || liftC > 90.0) {
                    if (diag) diag->Error("UsdGenGrow: per-strand lift must be finite "
                                          "and in [-90, 90] degrees");
                    return false;
                }
                cap.liftPerCurve[c] = static_cast<float>(liftC);
            }
            if (dirField.connected) {
                UsdGenVec3f const dirC(
                    static_cast<float>(dirField.Value(c, root, 0)),
                    static_cast<float>(dirField.Value(c, root, 1)),
                    static_cast<float>(dirField.Value(c, root, 2)));
                if (!std::isfinite(dirC[0]) || !std::isfinite(dirC[1]) ||
                    !std::isfinite(dirC[2])) {
                    if (diag) diag->Error("UsdGenGrow: per-strand directionVector "
                                          "must be finite");
                    return false;
                }
                cap.dirPerCurve[c] = dirC;
            }
        }
    }

    // Grow owns a new CV topology, so it also owns a new rest channel.  Use
    // the same direction/length formula as Evaluate, rooted at the C3
    // rest position when present (or at current roots for Scatter).  No
    // worker writes rest and no old-cardinality VtArray can leak through.
    if (upstream.rest.size() && upstream.rest.size() != upstream.totalCvs) {
        if (diag) diag->Error("UsdGenGrow: upstream rest plane cardinality does not match its CV topology");
        return false;
    }
    for (UsdGenVec3f const &point : upstream.rest) {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1]) || !std::isfinite(point[2])) {
            if (diag) diag->Error("UsdGenGrow: upstream rest plane contains non-finite values");
            return false;
        }
    }
    if ((!upstream.rootT.empty() && upstream.rootT.size() != R) ||
        (!upstream.rootN.empty() && upstream.rootN.size() != R) ||
        (!upstream.rootB.empty() && upstream.rootB.size() != R)) {
        if (diag) diag->Error("UsdGenGrow: upstream root-frame plane cardinality does not match curve topology");
        return false;
    }
    auto finiteFrame = [](UsdGenSpan<const UsdGenVec3f> const &values) {
        for (UsdGenVec3f const &v : values)
            if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2])) return false;
        return true;
    };
    if (!finiteFrame(upstream.rootT) || !finiteFrame(upstream.rootN) || !finiteFrame(upstream.rootB)) {
        if (diag) diag->Error("UsdGenGrow: upstream root-frame plane contains non-finite values");
        return false;
    }
    // Point cardinality and rootSpans were established before the per-curve
    // loop so connected controls sample each curve's root.
    UsdGenVec3fArray grownRest;
    grownRest.swap(buf.rest);   // a recycled capture's storage
    grownRest.assign(buf.totalCvs, UsdGenVec3f(0.0f));
    for (uint32_t c = 0; c != R; ++c) {
        uint32_t const rootIndex = rootSpans[c];
        UsdGenVec3f root = upstream.rest.empty() ? UsdGenVec3f(upstream.px[rootIndex], upstream.py[rootIndex], upstream.pz[rootIndex])
                                              : upstream.rest[rootIndex];
        UsdGenVec3f const dirLiteral = cap.dirPerCurve.size() == R
            ? cap.dirPerCurve[c] : cap.directionVector;
        UsdGenVec3f dir = cap.direction == Record().Token(kTokVector) ? dirLiteral
            : (upstream.rootN.empty() ? UsdGenVec3f(0.0f, 1.0f, 0.0f) : upstream.rootN[c]);
        dir = NormalizeGrowDirection(dir);
        UsdGenVec3f const axis = upstream.rootB.empty()
            ? UsdGenVec3f(0.0f, 1.0f, 0.0f) : upstream.rootB[c];
        float const liftC = cap.liftPerCurve.size() == R ? cap.liftPerCurve[c] : cap.lift;
        dir = RotateGrowDirection(dir, axis, liftC);
        dir = RotateGrowDirection(dir, upstream.rootN.empty()
            ? UsdGenVec3f(0.0f, 1.0f, 0.0f) : upstream.rootN[c], cap.azimuthPerCurve[c]);
        for (uint32_t i = 0; i != static_cast<uint32_t>(cap.cvCount); ++i) {
            float const t = cap.cvCount > 1 ? float(i) / float(cap.cvCount - 1) : 0.0f;
            float const distance = cap.perCurve[c] * t;
            grownRest[size_t(c) * cap.cvCount + i] = root + dir * distance;
        }
    }
    buf.rest = std::move(grownRest);
    return true;
}

void UsdGenGrowOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    (void)ctx;
    UsdGenGrowCapture const &cap =
        static_cast<UsdGenGrowCapture const &>(captureIn);

    UsdGenToken const &vectorToken = Record().Token(kTokVector);
    float const invSpan = view->cvCount > 1 ? 1.0f / float(view->cvCount - 1) : 0.0f;
    float *px = view->px;
    float *py = view->py;
    float *pz = view->pz;
    auto const *inPx = view->inPx;
    auto const *inPy = view->inPy;
    auto const *inPz = view->inPz;

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        // Input chunks may be ragged C3 curves.  Their plane ports are
        // already sliced at inFirstCv, so rebase the absolute upstream
        // offsets instead of treating inCvCount==0 as every root at CV 0.
        uint32_t const r = view->inCvOffsets
            ? static_cast<uint32_t>(view->inCvOffsets[c] - int(view->inFirstCv))
            : c * view->inCvCount;
        float const rx = inPx[r], ry = inPy[r], rz = inPz[r];

        // Direction compares use the record's token (no per-curve lookup).
        // liftPerCurve/dirPerCurve are whole-buffer capture payloads like
        // perCurve: chunk views pre-offset the plane arrays only, so index by
        // absolute curve.
        size_t const absolute = view->desc->firstCurve + c;
        UsdGenVec3f dir;
        if (cap.direction == vectorToken) {
            dir = !cap.dirPerCurve.empty() && absolute < cap.dirPerCurve.size()
                ? cap.dirPerCurve[absolute] : cap.directionVector;
        } else {  // surfaceNormal
            dir = view->rootN ? view->rootN[c] : UsdGenVec3f(0.0f, 1.0f, 0.0f);
        }
        UsdGenVec3f d = NormalizeGrowDirection(dir);
        UsdGenVec3f const axis = view->rootB ? view->rootB[c]
            : UsdGenVec3f(0.0f, 1.0f, 0.0f);
        float const liftC = !cap.liftPerCurve.empty() &&
            absolute < cap.liftPerCurve.size()
            ? cap.liftPerCurve[absolute] : cap.lift;
        d = RotateGrowDirection(d, axis, liftC);
        d = RotateGrowDirection(d, view->rootN ? view->rootN[c]
            : UsdGenVec3f(0.0f, 1.0f, 0.0f), cap.azimuthPerCurve[absolute]);
        // perCurve is a whole-buffer capture payload: chunk views pre-offset
        // the plane/per-curve arrays only, so index by absolute curve.
        const float targetLen = cap.perCurve.empty()
            ? 0.0f
            : cap.perCurve[absolute];
        for (uint32_t i = 0; i < view->cvCount; ++i) {
            const float t = i * invSpan;
            const float s = targetLen * t;
            uint32_t const o = static_cast<uint32_t>(view->Cv(c, i));
            px[o] = rx + d[0] * s;
            py[o] = ry + d[1] * s;
            pz[o] = rz + d[2] * s;
            if (view->hairT) view->hairT[o] = t;
        }
    }
}

UsdGenOpId UsdGenRegisterGrowOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenGrow";
    d.displayName = "Grow";
    d.create = &Create;
    d.countChanges = UsdGenCountsAtLevel(2);
    d.traits = kUsdGenOpGenerator | kUsdGenOpCaptureDigestComplete |
        kUsdGenOpCaptureOwnsRestAndPlanes;
    gInputTypes[0] = UsdGenRootSetTypeName();
    d.inputTypes = UsdGenSpan(gInputTypes);
    d.params = UsdGenSpan(kParams);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
