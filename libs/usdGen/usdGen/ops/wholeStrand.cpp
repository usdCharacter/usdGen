// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — whole-strand styler base. See wholeStrand.h.
#include "usdGen/ops/wholeStrand.h"

#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include <algorithm>
#include <cmath>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

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

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenWholeStrandCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        return upstream.topologyVersion == upstreamTopologyVersion &&
               upstream.valueVersion == upstreamValueVersion &&
               upstream.totalCurves == upstreamCurves &&
               upstream.totalCvs == upstreamCvs;
    }
};

UsdGenPlane const *FindPlane(std::vector<UsdGenPlane> const &planes, TfToken const &name)
{
    for (UsdGenPlane const &p : planes)
        if (p.name == name) return &p;
    return nullptr;
}

const TfToken sMask{"mask"};

}  // namespace

int const *UsdGenStrandSet::CurveInt(TfToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlane const *p = FindPlane(upstream->extraCurve, name);
    if (!p || p->type != TfToken("int") || p->arity != 1 ||
        p->interpolation != TfToken("uniform") || p->i.size() != upstream->totalCurves)
        return nullptr;
    return p->i.cdata();
}

float const *UsdGenStrandSet::CurveFloat(TfToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlane const *p = FindPlane(upstream->extraCurve, name);
    if (!p || p->type != TfToken("float") || p->arity != 1 ||
        p->interpolation != TfToken("uniform") || p->f.size() != upstream->totalCurves)
        return nullptr;
    return p->f.cdata();
}

float const *UsdGenStrandSet::CvFloat(TfToken const &name) const
{
    if (!upstream) return nullptr;
    UsdGenPlane const *p = FindPlane(upstream->extraCv, name);
    if (!p || p->type != TfToken("float") || p->arity != 1 || p->f.size() != upstream->totalCvs)
        return nullptr;
    return p->f.cdata();
}

uint32_t UsdGenStrandSet::Seed(uint32_t c) const
{
    static TfToken const sSeed("strandSeed");
    if (int const *s = CurveInt(sSeed)) return uint32_t(s[c]);
    return Index(c);
}

uint32_t UsdGenStrandSet::Index(uint32_t c) const
{
    static TfToken const sIndex("strandIndex");
    if (int const *s = CurveInt(sIndex)) return uint32_t(s[c]);
    return c;
}

void UsdGenWholeStrandOp::EnsureTables() const
{
    if (_tablesReady) return;
    _topo = UsdGenBaseTopologyParams();
    for (TfToken const &t : StructuralParameters()) _topo.push_back(t);
    _value = UsdGenBaseValueParams();
    _value.push_back(TfToken("enabled"));
    _value.push_back(sMask);
    _specs = OutputPlaneSpecs();
    _outputs.clear();
    for (auto const &s : _specs) _outputs.push_back(s.name);
    _tablesReady = true;
}

TfSpan<const TfToken> UsdGenWholeStrandOp::TopologyParameters() const
{
    EnsureTables();
    return TfSpan<const TfToken>(_topo.data(), _topo.size());
}

TfSpan<const TfToken> UsdGenWholeStrandOp::ValueParameters() const
{
    EnsureTables();
    return TfSpan<const TfToken>(_value.data(), _value.size());
}

TfSpan<const TfToken> UsdGenWholeStrandOp::OutputPrimvars() const
{
    EnsureTables();
    return TfSpan<const TfToken>(_outputs.data(), _outputs.size());
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
    EnsureTables();
    opUtil::Digest d;
    d.Mix(Type());
    if (UsdGenParamView const *p = ctx.params) {
        for (TfToken const &name : _topo) {
            d.Mix(name);
            d.Mix(p->GetVtValue(name, VtValue()));
        }
    }
    d.Mix(uint64_t(ctx.seed));
    d.Mix(ctx.upstreamGeneration);
    if (ctx.desc && ctx.surface < ctx.desc->surfaces.size())
        d.Mix(ctx.desc->surfaces[ctx.surface].surfaceGeneration);
    for (uint32_t i = 0; i < ctx.referenceCount; ++i) {
        if (ctx.resolvedReferences && ctx.resolvedReferences[i]) {
            d.Mix(ctx.resolvedReferences[i]->identity);
            d.Mix(ctx.resolvedReferences[i]->curveGeneration);
        } else if (ctx.references && ctx.references[i]) {
            d.Mix(ctx.references[i]->generation);
        }
    }
    for (uint32_t i = 0; i < ctx.mapCount; ++i)
        if (ctx.maps && ctx.maps[i]) d.Mix(ctx.maps[i]->identity);
    return d.Epoch(0x5768537472ull);  // "WhStr"
}

bool UsdGenWholeStrandOp::Capture(UsdGenCaptureContext const &ctx,
                                  UsdGenCurveBuffer const &upstream,
                                  UsdGenCapture *out, UsdGenDiagnostics *diag)
{
    EnsureTables();
    auto &cap = *static_cast<UsdGenWholeStrandCapture *>(out);
    cap.upstreamTopologyVersion = upstream.topologyVersion;
    cap.upstreamValueVersion = upstream.valueVersion;
    cap.upstreamCurves = upstream.totalCurves;
    cap.upstreamCvs = upstream.totalCvs;
    cap.ok = false;
    cap.result.clear();
    cap.planes.assign(_specs.size(), UsdGenStrandPlaneData());

    size_t const R = upstream.totalCurves, totalCvs = upstream.totalCvs;
    for (size_t s = 0; s < _specs.size(); ++s) {
        size_t const n = _specs[s].cv ? totalCvs : R;
        if (_specs[s].isInt) cap.planes[s].i.assign(n, -1);
        else cap.planes[s].f.assign(n, 0.0f);
    }
    UsdGenStrandSet strands;
    strands.upstream = &upstream;
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
    cap.result = std::move(strands.points);
    cap.ok = true;
    return true;
}

void UsdGenWholeStrandOp::Evaluate(UsdGenEvalContext const &ctx,
                                   UsdGenCapture const &captureIn,
                                   UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenWholeStrandCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params;
    UsdGenParamField const maskField = p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
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
