// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — whole-strand styler base. See wholeStrand.h.
#include "usdGen/ops/wholeStrand.h"
#include "usdGenSurface/surfaceInput.h"

#include "usdGen/ops/opUtil.h"
#include "usdGen/api/token.h"

#include <algorithm>
#include <cmath>
#include <string>


namespace usdGen {

namespace {

struct UsdGenWholeStrandCapture final : public UsdGenCapture
{
    std::vector<float> result;                       // 3 * totalCvs
    std::vector<UsdGenStrandPlaneData> planes;
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;
    uint32_t upstreamCurves = 0;
    uint32_t upstreamCvs = 0;
    bool ok = false;
    std::vector<uint32_t> spans;   // capture scratch, kept across Recycle()

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWholeStrandCapture>(*this);
    }
    bool Recyclable() const override { return true; }
    bool Recycle() override
    {
        // Capture refills result, planes and spans; their storage stays.
        upstreamTopologyVersion = upstreamValueVersion = 0;
        upstreamCurves = upstreamCvs = 0;
        ok = false;
        return true;
    }
    bool ValidForTopology(UsdGenElementSetView const &in) const override
    {
        UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in);
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

// A named plane of the upstream view: curve rate (levels 0-1) or vertex rate.
UsdGenPlaneView const *FindPlane(UsdGenCurveSetView const &v, bool perCv, UsdGenToken const &name,
                                 UsdGenPlaneView *storage)
{
    uint32_t const n = perCv ? v.ExtraCvCount() : v.ExtraCurveCount();
    for (uint32_t k = 0; k < n; ++k) {
        *storage = perCv ? v.ExtraCv(k) : v.ExtraCurve(k);
        if (storage->name == name) return storage;
    }
    return nullptr;
}

const UsdGenToken sStrandSeed = UsdGenInternToken("strandSeed");
const UsdGenToken sStrandIndex = UsdGenInternToken("strandIndex");

}  // namespace

int const *UsdGenStrandSet::CurveInt(UsdGenToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlaneView plane;
    UsdGenPlaneView const *p = FindPlane(*upstream, false, name, &plane);
    if (!p || p->type != UsdGenTokens().int_ || p->arity != 1 ||
        p->interpolation != UsdGenTokens().uniform || p->i.size() != upstream->totalCurves)
        return nullptr;
    return p->i.cdata();
}

float const *UsdGenStrandSet::CurveFloat(UsdGenToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlaneView plane;
    UsdGenPlaneView const *p = FindPlane(*upstream, false, name, &plane);
    if (!p || p->type != UsdGenTokens().float_ || p->arity != 1 ||
        p->interpolation != UsdGenTokens().uniform || p->f.size() != upstream->totalCurves)
        return nullptr;
    return p->f.cdata();
}

float const *UsdGenStrandSet::CvFloat(UsdGenToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlaneView plane;
    UsdGenPlaneView const *p = FindPlane(*upstream, true, name, &plane);
    if (!p || p->type != UsdGenTokens().float_ || p->arity != 1 || p->f.size() != upstream->totalCvs)
        return nullptr;
    return p->f.cdata();
}

uint32_t UsdGenStrandSet::Seed(uint32_t c) const
{
    if (int const *s = CurveInt(sStrandSeed)) return uint32_t(s[c]);
    return Index(c);
}

uint32_t UsdGenStrandSet::Index(uint32_t c) const
{
    if (int const *s = CurveInt(sStrandIndex)) return uint32_t(s[c]);
    return c;
}

UsdGenWholeStrandOp::UsdGenWholeStrandOp(UsdGenOpRecord const &record)
    : UsdGenOp(record)
    , _mask(record.FindParam(UsdGenTokens().mask))
{
    for (UsdGenChannelDesc const &plane : record.OutputPlanes()) {
        UsdGenStrandPlaneSpec spec;
        spec.name = plane.name;
        spec.cv = plane.level == 2;   // vertex
        spec.isInt = plane.scalar == UsdGenScalar::I32;
        _specs.push_back(spec);
    }
}

bool UsdGenWholeStrandOp::Bind(UsdGenParamView const &, UsdGenDiagnostics *)
{
    return true;
}

std::unique_ptr<UsdGenCapture> UsdGenWholeStrandOp::CreateCapture() const
{
    return std::make_unique<UsdGenWholeStrandCapture>();
}

UsdGenEpoch UsdGenWholeStrandOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    opUtil::Digest d;
    d.Mix(Record().Name());
    if (UsdGenParamView const *p = ctx.params) {
        UsdGenOpRecord const &record = Record();
        for (UsdGenParamSlot s = 0; s < record.ParamCount(); ++s) {
            if (record.ParamClass(s) == UsdGenParamClass::Value) continue;
            d.Mix(record.ParamName(s));
            d.Mix(UsdGenAuthoredValue(*p, s));
        }
    }
    d.Mix(uint64_t(ctx.seed));
    d.Mix(ctx.upstreamGeneration);
    if (ctx.desc && ctx.surface < UsdGenSurfaceInputs(*ctx.desc).size())
        d.Mix(UsdGenSurfaceAt(*ctx.desc, ctx.surface).surfaceGeneration);
    for (uint32_t i = 0; i < ctx.referenceCount; ++i) {
        if (ctx.resolvedReferences && ctx.resolvedReferences[i]) {
            d.Mix(ctx.resolvedReferences[i]->identity);
            d.Mix(ctx.resolvedReferences[i]->generation);
        } else if (ctx.references && ctx.references[i]) {
            d.Mix(ctx.references[i]->generation);
        }
    }
    for (uint32_t i = 0; i < ctx.mapCount; ++i)
        if (ctx.maps && ctx.maps[i]) d.Mix(ctx.maps[i]->identity);
    return d.Epoch(0x5768537472ull);  // "WhStr"
}

bool UsdGenWholeStrandOp::Capture(UsdGenCaptureContext const &ctx,
                                  UsdGenOpInputs const &in,
                                  UsdGenCapture *out, UsdGenDiagnostics *diag)
{
    UsdGenCurveSetView const upstream = UsdGenCurveSetView::Of(in.Primary());
    auto &cap = *static_cast<UsdGenWholeStrandCapture *>(out);
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    cap.ok = false;
    // A recycled capture's vectors keep their storage: resize, never rebuild.
    cap.planes.resize(_specs.size());

    size_t const R = upstream.totalCurves, totalCvs = upstream.totalCvs;
    for (size_t s = 0; s < _specs.size(); ++s) {
        size_t const n = _specs[s].cv ? totalCvs : R;
        UsdGenStrandPlaneData &plane = cap.planes[s];
        if (_specs[s].isInt) {
            plane.i.assign(n, -1);
            plane.f.clear();
        } else {
            plane.f.assign(n, 0.0f);
            plane.i.clear();
        }
    }
    UsdGenStrandSet strands;
    strands.upstream = &upstream;
    strands.spans.swap(cap.spans);
    strands.points.swap(cap.result);
    // Hands the scratch back on every return; result is the computed points
    // only on success.
    struct Restore {
        UsdGenWholeStrandCapture &cap;
        UsdGenStrandSet &strands;
        ~Restore()
        {
            cap.spans.swap(strands.spans);
            cap.result.swap(strands.points);
            if (!cap.ok) cap.result.clear();
        }
    } restore{cap, strands};
    std::string error;
    if (!opUtil::CurveSpans(upstream, &strands.spans, &error)) {
        if (diag) diag->Error(Label() + ": " + error);
        return false;
    }
    if (upstream.px.size() != totalCvs || upstream.py.size() != totalCvs ||
        upstream.pz.size() != totalCvs) {
        if (diag) diag->Error(Label() + ": input point planes do not match the CV count");
        return false;
    }
    strands.points.resize(3 * totalCvs);
    for (size_t i = 0; i < totalCvs; ++i) {
        strands.points[3 * i] = upstream.px[i];
        strands.points[3 * i + 1] = upstream.py[i];
        strands.points[3 * i + 2] = upstream.pz[i];
    }
    if (R == 0) {
        cap.ok = true;
        return true;
    }
    if (!Compute(ctx, strands, cap.planes, diag)) return false;
    for (float v : strands.points) {
        if (!std::isfinite(v)) {
            if (diag) diag->Error(Label() + ": computed a non-finite point");
            return false;
        }
    }
    cap.ok = true;
    return true;
}

void UsdGenWholeStrandOp::Evaluate(UsdGenEvalContext const &ctx,
                                   UsdGenCapture const &captureIn,
                                   UsdGenChunkView const &chunk) const
{
    UsdGenCurveChunk curve(chunk);
    UsdGenCurveChunk const *view = &curve;
    auto const &cap = static_cast<UsdGenWholeStrandCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params;
    UsdGenParamField const maskField = p ? p->GetScalarField(_mask, 1.0) : UsdGenParamField{1.0};
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const firstCv = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;
    bool const ready = cap.ok && cap.result.size() == size_t(cap.upstreamCvs) * 3;
    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        size_t const curve = curveBase + c;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            size_t const cv = firstCv + o;
            float const in[3] = {view->inPx[o], view->inPy[o], view->inPz[o]};
            float const m = ready ? std::clamp(float(maskField.Value(curve, cv)), 0.0f, 1.0f) : 0.0f;
            if (m == 0.0f || cv * 3 + 2 >= cap.result.size()) {
                view->px[o] = in[0]; view->py[o] = in[1]; view->pz[o] = in[2];
                continue;
            }
            float const *r = &cap.result[cv * 3];
            view->px[o] = m == 1.0f ? r[0] : in[0] + (r[0] - in[0]) * m;
            view->py[o] = m == 1.0f ? r[1] : in[1] + (r[1] - in[1]) * m;
            view->pz[o] = m == 1.0f ? r[2] : in[2] + (r[2] - in[2]) * m;
        }
        for (uint32_t slot = 0; slot < view->outCount && slot < _specs.size(); ++slot) {
            UsdGenStrandPlaneSpec const &spec = _specs[slot];
            UsdGenStrandPlaneData const *data = slot < cap.planes.size() ? &cap.planes[slot] : nullptr;
            if (spec.isInt) {
                int *dst = view->outI ? view->outI[slot] : nullptr;
                if (!dst) continue;
                if (spec.cv) {
                    for (size_t i = 0; i < n; ++i) {
                        size_t const cv = firstCv + g + i;
                        dst[g + i] = data && cv < data->i.size() ? data->i[cv] : -1;
                    }
                } else {
                    dst[c] = data && curve < data->i.size() ? data->i[curve] : -1;
                }
            } else {
                float *dst = view->outF ? view->outF[slot] : nullptr;
                if (!dst) continue;
                if (spec.cv) {
                    for (size_t i = 0; i < n; ++i) {
                        size_t const cv = firstCv + g + i;
                        dst[g + i] = data && cv < data->f.size() ? data->f[cv] : 0.0f;
                    }
                } else {
                    dst[c] = data && curve < data->f.size() ? data->f[curve] : 0.0f;
                }
            }
        }
    }
}

}  // namespace usdGen
