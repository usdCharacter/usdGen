// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — UsdGenTieClumpOp implementation. See tieClump.h for the contract.
//
// Deviation from the reference: its tie test compares tieArc with the strand
// length recorded right after length-scale/trim; here the strand length on
// entry to this operator is used (the upstream stack in between only adds
// frizz, which lengthens strands slightly).
#include "usdGen/ops/tieClump.h"

#include "usdGen/ops/opUtil.h"
#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include "usdGenMath/math.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/cowArray.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"


namespace usdGen {

namespace {

using namespace tieClumpP;
using D = expr::Domain;
constexpr UsdGenExprAdmission kBoolGroom{expr::ScalarType::Bool, 1, "bool", D::Groom};
constexpr UsdGenExprAdmission kFloatAll{expr::ScalarType::Float32, 1, "float", D::All};
// input, surface, seed, the structural parameters, then enabled and mask.
UsdGenParamDecl const kParams[kCount] = {
    {"input", UsdGenParamClass::Capture},
    {"surface", UsdGenParamClass::Capture},
    {"seed", UsdGenParamClass::Capture},
    {"guides", UsdGenParamClass::Capture},
    {"tie:amount", UsdGenParamClass::Capture},
    {"tie:spread", UsdGenParamClass::Capture},
    {"tie:positions", UsdGenParamClass::Capture},
    {"tie:entryCounts", UsdGenParamClass::Capture},
    {"tie:guideIndices", UsdGenParamClass::Capture},
    {"tie:vertexIndices", UsdGenParamClass::Capture},
    {"enabled", UsdGenParamClass::Value, true, kBoolGroom},
    {"mask", UsdGenParamClass::Value, true, kFloatAll},
};
constexpr char const *kReferences[] = {"guides"};
constexpr char const *kTokens[] = {"primaryGuide"};
std::unique_ptr<UsdGenOp> Create(UsdGenOpRecord const &r, void *)
{
    return std::make_unique<UsdGenTieClumpOp>(r);
}

constexpr double kGate = 1e-4;          // amount threshold
constexpr float kMinLength = 1e-6f;     // strands this short are skipped
constexpr float kMinWeight = 1e-6f;

struct Tie
{
    UsdGenVec3f position;
    float arc = 0.0f;   // arc length along the guide to the tied point
};

bool ReadPositions(UsdGenValue const &v, std::vector<UsdGenVec3f> *out)
{
    out->clear();
    if (v.IsEmpty()) return true;
    if (v.IsHolding<UsdGenVec3fArray>()) {
        UsdGenVec3fArray const &a = v.UncheckedGet<UsdGenVec3fArray>();
        out->assign(a.cbegin(), a.cend());
        return true;
    }
    if (v.IsHolding<UsdGenVec3dArray>()) {
        UsdGenVec3dArray const &a = v.UncheckedGet<UsdGenVec3dArray>();
        for (UsdGenVec3d const &p : a) out->push_back(UsdGenVec3f(p));
        return true;
    }
    return false;
}

bool ReadInts(UsdGenValue const &v, std::vector<int> *out)
{
    out->clear();
    if (v.IsEmpty()) return true;
    if (v.IsHolding<UsdGenIntArray>()) {
        UsdGenIntArray const &a = v.UncheckedGet<UsdGenIntArray>();
        out->assign(a.cbegin(), a.cend());
        return true;
    }
    return false;
}

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

// Arc position of the polyline point closest to t: per segment the clamped
// projection, the first segment of minimum distance wins.
float ClosestArc(float const *p, size_t n, std::vector<float> const &arc, UsdGenVec3f const &t)
{
    float best = std::numeric_limits<float>::max();
    float s = 0.0f;
    for (size_t k = 0; k + 1 < n; ++k) {
        UsdGenVec3f const a = strand::Load(p, k), b = strand::Load(p, k + 1);
        UsdGenVec3f const ab = b - a;
        float const len2 = UsdGenDot(ab, ab);
        float const u = len2 > 0.0f ? strand::Clamp01(UsdGenDot(t - a, ab) / len2) : 0.0f;
        UsdGenVec3f const q(a[0] + ab[0] * u, a[1] + ab[1] * u, a[2] + ab[2] * u);
        UsdGenVec3f const d = t - q;
        float const d2 = UsdGenDot(d, d);
        if (d2 < best) {
            best = d2;
            s = arc[k] + std::sqrt(len2) * u;
        }
    }
    return s;
}

void PullStrand(float *p, size_t n, std::vector<Tie> const &ties, float amount, float spread)
{
    std::vector<float> arc;
    float length = strand::ArcLengths(p, n, &arc);
    if (!(length > kMinLength)) return;
    float const entryLength = length;
    float const m = float(std::max<size_t>(1, n - 1));
    for (Tie const &tie : ties) {
        if (!(tie.arc < entryLength)) continue;
        float const inner = std::max(1e-4f, std::max(0.04f * length, 1.5f * length / m));
        float const outer = std::max(0.4f * length, 8.0f * length / m);
        float const reach = std::max(inner, outer);
        float const sStar = ClosestArc(p, n, arc, tie.position);
        float const radius = reach * spread + (1.0f - spread) * inner;
        for (size_t k = 1; k < n; ++k) {
            float const x = strand::Clamp01(std::fabs(arc[k] - sStar) / radius);
            float const f = 1.0f - strand::SmoothStep01(x);
            float const w = strand::Clamp01((1.0f - spread) * f +
                                            spread * std::sqrt(strand::Clamp01(f))) * amount;
            if (!(w > kMinWeight)) continue;
            UsdGenVec3f const q = strand::Load(p, k);
            UsdGenVec3f const d = tie.position - q;
            strand::Store(p, k, UsdGenVec3f(q[0] + d[0] * w, q[1] + d[1] * w, q[2] + d[2] * w));
        }
        length = strand::ArcLengths(p, n, &arc);
        if (!(length > kMinLength)) break;
    }
}

}  // namespace

bool UsdGenTieClumpOp::Compute(UsdGenCaptureContext const &ctx, UsdGenStrandSet &strands,
                               std::vector<UsdGenStrandPlaneData> &, UsdGenDiagnostics *diag)
{
    auto fail = [&](std::string const &message) {
        if (diag) diag->Error(Label() + ": " + message);
        return false;
    };
    UsdGenParamView const *p = ctx.params;
    if (!p) return fail("no parameters");
    double const amount = p->GetDouble(kAmount, 1.0);
    double const spread = p->GetDouble(kSpread, 0.25);
    if (!std::isfinite(amount) || !std::isfinite(spread))
        return fail("usdGen:tie:amount and usdGen:tie:spread must be finite");

    std::vector<UsdGenVec3f> positions;
    std::vector<int> counts, guideIndices, vertexIndices;
    if (!ReadPositions(UsdGenAuthoredValue(*p, kPositions), &positions))
        return fail("usdGen:tie:positions must be a point3f[]");
    if (!ReadInts(UsdGenAuthoredValue(*p, kEntryCounts), &counts) ||
        !ReadInts(UsdGenAuthoredValue(*p, kGuideIndices), &guideIndices) ||
        !ReadInts(UsdGenAuthoredValue(*p, kVertexIndices), &vertexIndices))
        return fail("usdGen:tie:entryCounts/guideIndices/vertexIndices must be int[]");
    if (counts.size() != positions.size())
        return fail("usdGen:tie:entryCounts needs one entry count per tie position");
    size_t entries = 0;
    for (int c : counts) {
        if (c < 0) return fail("usdGen:tie:entryCounts has a negative count");
        entries += size_t(c);
    }
    if (guideIndices.size() != entries || vertexIndices.size() != entries)
        return fail("usdGen:tie:guideIndices/vertexIndices must hold sum(entryCounts) entries");
    for (UsdGenVec3f const &t : positions)
        if (!std::isfinite(t[0]) || !std::isfinite(t[1]) || !std::isfinite(t[2]))
            return fail("usdGen:tie:positions must be finite");
    if (entries == 0 || !(amount > kGate)) return true;   // gate: identity

    if (ctx.referenceCount != 1 || !ctx.resolvedReferences || !ctx.resolvedReferences[0] ||
        !ctx.resolvedReferences[0]->value)
        return fail("usdGen:guides must target one guide curve set when ties are authored");
    UsdGenCurveSetView const guides = UsdGenCurveSetView::Of(*ctx.resolvedReferences[0]->value);
    std::vector<uint32_t> guideSpans;
    std::string error;
    if (!opUtil::CurveSpans(guides, &guideSpans, &error)) return fail("guide topology: " + error);
    if (guides.px.size() != guides.totalCvs || guides.py.size() != guides.totalCvs ||
        guides.pz.size() != guides.totalCvs)
        return fail("guide point planes do not match their CV count");
    size_t const G = guides.totalCurves;

    // Tie table: per guide, {position, arc length to the tied guide point},
    // in tie order then entry order.
    std::vector<std::vector<Tie>> tiesOf(G);
    size_t entry = 0, skipped = 0;
    for (size_t t = 0; t < positions.size(); ++t) {
        for (int e = 0; e < counts[t]; ++e, ++entry) {
            int const g = guideIndices[entry];
            if (g < 0 || size_t(g) >= G) {
                ++skipped;
                continue;
            }
            size_t const first = guideSpans[size_t(g)];
            int const nv = int(guideSpans[size_t(g) + 1] - first);
            float arc = 0.0f;
            if (nv >= 2) {
                int const last = std::min(std::max(vertexIndices[entry], 1), nv - 1);
                for (int v = 1; v <= last; ++v) {
                    size_t const a = first + size_t(v) - 1, b = first + size_t(v);
                    UsdGenVec3f const d(guides.px[b] - guides.px[a], guides.py[b] - guides.py[a],
                                    guides.pz[b] - guides.pz[a]);
                    arc += d.GetLength();
                }
            }
            tiesOf[size_t(g)].push_back({positions[t], arc});
        }
    }
    if (skipped && diag)
        diag->Warn(Label() + ": skipped " + std::to_string(skipped) +
                   " tie entr(ies) naming a guide outside the guide set");
    bool anyTied = false;
    for (auto const &list : tiesOf) anyTied = anyTied || !list.empty();
    if (!anyTied) return true;

    int const *primary = strands.CurveInt(Record().Token(kTokPrimaryGuide));
    if (!primary)
        return fail("the upstream strands carry no primaryGuide plane "
                    "(place a GuideInterpolate upstream)");
    float const amt = float(std::clamp(amount, 0.0, 1.0));
    float const spr = float(std::clamp(spread, 0.0, 1.0));
    ParallelFor(ctx.dispatcher, strands.Count(), [&](size_t c) {
        int const g = primary[c];
        if (g < 0 || size_t(g) >= G || tiesOf[size_t(g)].empty()) return;
        PullStrand(strands.P(uint32_t(c)), strands.Size(uint32_t(c)), tiesOf[size_t(g)], amt, spr);
    });
    return true;
}

UsdGenOpId UsdGenRegisterTieClumpOp(UsdGenRegistrar &r)
{
    UsdGenOpDef d;
    d.name = "UsdGenTieClump";
    d.displayName = "TieClump";
    d.create = &Create;
    d.params = UsdGenSpan(kParams);
    d.referenceInputs = UsdGenSpan(kReferences);
    d.tokens = UsdGenSpan(kTokens);
    return r.RegisterOp(d);
}

}  // namespace usdGen
