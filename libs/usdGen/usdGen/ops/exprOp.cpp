// usdGen — UsdGenExprOp implementation. 02-schema.md §2.7, 04 §3, 07 §7.8.
//
// Capture compiles usdGen:expr:source through the existing connected-parameter
// lane (expr::Frontend::Compile, then CpuExpressionContext::Build and
// expr::EvaluateProgram over the upstream curves — the cpuParameters.cpp path,
// no new machinery) and stores one float per element per component. Evaluate
// adds (displacement) or sets (width) from that array under the mask envelope.
// mode=cv evaluates at point rate, mode=curve at primitive rate with a
// per-root broadcast. Every edge fails closed with a UsdGenDiagnostics
// message; see exprOp.h for the contract.
#include "usdGen/ops/exprOp.h"

#include "usdGen/expressions/cpuEvaluator.h"
#include "usdGen/expressions/frontend.h"
#include "usdGen/opParams.h"
#include "usdGen/ops/opUtil.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include "usdGen/registries/registrar.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

struct UsdGenExprOpCapture final : public UsdGenCapturePayload
{
    uint8_t components = 0;  // 3 = displacement, 1 = width
    uint8_t modeCv = 1;      // 1: perCv holds totalCvs * components;
                             // 0: perCurve holds totalCurves * components
    uint32_t curves = 0;
    uint32_t cvs = 0;
    uint64_t upstreamTopologyVersion = 0;
    uint64_t upstreamValueVersion = 0;

    std::unique_ptr<UsdGenCapture> Clone() const override
    {
        return std::make_unique<UsdGenExprOpCapture>(*this);
    }
    bool ValidForTopology(UsdGenCurveBuffer const &upstream) const override
    {
        // The expression may read upstream values ($P, $cWidth, ...), so a
        // value move recaptures exactly like a topology move (Clump rule).
        if (upstream.topologyVersion != upstreamTopologyVersion ||
            upstream.valueVersion != upstreamValueVersion ||
            upstream.totalCurves != curves || upstream.totalCvs != cvs)
            return false;
        if (components != 3 && components != 1) return false;
        if (modeCv)
            return perCurve.empty() && perCv.size() == size_t(cvs) * components;
        return perCv.empty() && perCurve.size() == size_t(curves) * components;
    }
};

bool IsBlank(std::string const &text)
{
    for (char c : text)
        if (!std::isspace(static_cast<unsigned char>(c))) return false;
    return true;
}

}  // namespace

static TfTokenVector _topoParams = [] {
    TfTokenVector v = UsdGenBaseTopologyParams();
    v.push_back(TfToken("mode"));             // structural kernel branch (02 §6.1)
    v.push_back(TfToken("expr:returnType"));  // structural output branch (02 §6.1)
    v.push_back(TfToken("expr:source"));      // capture: recompiled at Capture
    v.push_back(TfToken("expr:maps"));        // rel retarget recompiles (02 §6.1)
    return v;
}();

static TfTokenVector _valueParams = [] {
    TfTokenVector v = UsdGenBaseValueParams();
    v.push_back(TfToken("enabled"));  // value-toggle styler (02 §6.3)
    v.push_back(TfToken("mask"));     // operator envelope, literal or connected
    return v;
}();

TfSpan<const TfToken> UsdGenExprOp::TopologyParameters() const
{
    return TfSpan<const TfToken>(_topoParams.data(), _topoParams.size());
}
TfSpan<const TfToken> UsdGenExprOp::ValueParameters() const
{
    return TfSpan<const TfToken>(_valueParams.data(), _valueParams.size());
}
std::unique_ptr<UsdGenCapture> UsdGenExprOp::CreateCapture() const
{
    return std::make_unique<UsdGenExprOpCapture>();
}
uint32_t UsdGenExprOp::PlanesTouched() const
{
    // Static per type: displacement rewrites points, width rewrites widths,
    // and Evaluate always fills both planes (one transformed, one copied).
    return kPlanePoints | kPlaneWidths;
}

bool UsdGenExprOp::Bind(UsdGenParamView const &params, UsdGenDiagnostics *diag)
{
    TfToken const mode = params.GetToken(sMode, sCv);
    if (mode != sCv && mode != sCurve) {
        if (diag) diag->Error("UsdGenExprOp: unknown usdGen:mode '" + mode.GetString() + "'");
        return false;
    }
    TfToken const returnType = params.GetToken(sReturnType, sDisplacement);
    if (returnType != sDisplacement && returnType != sWidth && returnType != sColor) {
        if (diag) diag->Error("UsdGenExprOp: unknown usdGen:expr:returnType '" +
                              returnType.GetString() + "'");
        return false;
    }
    if (returnType == sColor) {
        if (diag) diag->Error("UsdGenExprOp: usdGen:expr:returnType 'color' is not supported: "
                              "the operator-emitted primvar transport carries no vec3 color "
                              "plane, so a color field has nowhere to publish");
        return false;
    }
    UsdGenParamValue const *source = params.FindParam(sSource);
    std::string text;
    if (source && !source->value.IsEmpty()) {
        if (source->value.IsHolding<std::string>())
            text = source->value.UncheckedGet<std::string>();
        else if (source->value.IsHolding<TfToken>())
            text = source->value.UncheckedGet<TfToken>().GetString();
        else {
            if (diag) diag->Error("UsdGenExprOp: usdGen:expr:source must be a string");
            return false;
        }
    }
    if (IsBlank(text)) {
        if (diag) diag->Error("UsdGenExprOp: usdGen:expr:source is empty");
        return false;
    }
    return true;
}

UsdGenEpoch UsdGenExprOp::CaptureDigest(UsdGenCaptureContext const &ctx) const
{
    // The captured field is a pure function of the source text, the two
    // structural tokens, the upstream topology and the seed; a source edit
    // must recapture (a re-sweep alone would keep the stale array). Map
    // identities join the digest so a map retarget recaptures for direct
    // callers too (the scheduler folds them in separately as well).
    opUtil::Digest digest;
    if (UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr) {
        digest.Mix(p->GetVtValue(sSource, VtValue(std::string())));
        digest.Mix(p->GetToken(sMode, sCv));
        digest.Mix(p->GetToken(sReturnType, sDisplacement));
    }
    digest.Mix(uint64_t(ctx.upstreamGeneration));
    digest.Mix(uint64_t(ctx.seed));
    if (ctx.maps) {
        for (uint32_t i = 0; i < ctx.mapCount; ++i) {
            if (!ctx.maps[i]) continue;
            digest.Mix(ctx.maps[i]->identity);
            digest.Mix(ctx.maps[i]->textureGeneration);
        }
    }
    return digest.Epoch(0x9E3779B97F4A7C15ull);
}

bool UsdGenExprOp::Capture(
    UsdGenCaptureContext const &ctx,
    UsdGenCurveBuffer const &upstream,
    UsdGenCapture *out,
    UsdGenDiagnostics *diag)
{
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    if (!p) return true;
    // Literals and tokens are validated here (the CPU lane never calls Bind).
    if (!Bind(*p, diag)) return false;
    auto *cap = dynamic_cast<UsdGenExprOpCapture *>(out);
    if (!cap) {
        if (diag) diag->Error("UsdGenExprOp: capture payload has the wrong type");
        return false;
    }
    std::vector<uint32_t> spans;
    std::string spansError;
    if (!opUtil::CurveSpans(upstream, &spans, &spansError)) {
        if (diag) diag->Error("UsdGenExprOp: upstream CV topology " + spansError);
        return false;
    }
    bool const modeCv = p->GetToken(sMode, sCv) == sCv;
    bool const displacement = p->GetToken(sReturnType, sDisplacement) == sDisplacement;
    expr::Domain const domain =
        modeCv ? expr::Domain::Point : expr::Domain::Primitive;
    uint32_t const components = displacement ? 3u : 1u;

    std::string source;
    if (UsdGenParamValue const *authored = p->FindParam(sSource)) {
        if (authored->value.IsHolding<std::string>())
            source = authored->value.UncheckedGet<std::string>();
        else if (authored->value.IsHolding<TfToken>())
            source = authored->value.UncheckedGet<TfToken>().GetString();
    }
    expr::FrontendOptions options;
    options.domain = domain;
    options.destination = expr::ScalarType::Float32;
    options.components = components;
    expr::CompileResult compiled = expr::Frontend::Compile(source, options);
    if (!compiled.ok) {
        if (diag) {
            if (compiled.diagnostics.empty())
                diag->Error("UsdGenExprOp: expression compilation failed");
            for (auto const &message : compiled.diagnostics)
                diag->Error("UsdGenExprOp: " + message);
        }
        return false;
    }
    expr::IRProgram const ir = compiled.program.IR();
    if (!ir.samplers.empty()) {
        if (diag) {
            expr::IRSampler const &slot = ir.samplers.front();
            std::string const call = slot.kind == expr::SamplerKind::Geometry
                ? "geoSampler(\"" + slot.input + "\")" : "ptex(\"" + slot.input + "\")";
            std::string message = "UsdGenExprOp: expression calls " + call +
                ", but a bare usdGen:expr:source has no input:<name> bindings to "
                "resolve it against";
            if (ctx.maps && ctx.mapCount) {
                message += "; usdGen:expr:maps targets:";
                for (uint32_t i = 0; i < ctx.mapCount; ++i)
                    if (ctx.maps[i]) message += " " + ctx.maps[i]->path.GetString();
            } else {
                message += "; usdGen:expr:maps names no maps";
            }
            diag->Error(message);
        }
        return false;
    }
    for (expr::IRInstruction const &instruction : ir.instructions) {
        if (instruction.op != expr::IROp::LoadVariable) continue;
        if (instruction.variable == expr::Variable::Value) {
            if (diag) diag->Error("UsdGenExprOp: expression reads $value, which a bare "
                                  "usdGen:expr:source does not define (width authors read "
                                  "the upstream width as $cWidth)");
            return false;
        }
        if (instruction.variable == expr::Variable::Frame ||
            instruction.variable == expr::Variable::Time) {
            if (diag) diag->Error("UsdGenExprOp: expression reads " +
                                  std::string(instruction.variable == expr::Variable::Frame
                                                  ? "$frame" : "$time") +
                                  ", but the operator evaluates once per capture and "
                                  "cannot see time");
            return false;
        }
    }
    if (ctx.maps && ctx.mapCount > 0 && diag)
        diag->Warn("UsdGenExprOp: usdGen:expr:maps names " +
                   std::to_string(ctx.mapCount) +
                   " map(s), but the expression calls no sampler; the maps are ignored");

    // --- geometry view over the node's INPUT curves (cpuParameters.cpp) ----
    size_t const curveCount = upstream.totalCurves;
    size_t const pointCount = upstream.totalCvs;
    std::vector<uint32_t> offsets(curveCount + 1, 0);
    if (!upstream.cvOffsets.empty() &&
        size_t(upstream.cvOffsets.size()) == curveCount + 1) {
        for (size_t c = 0; c <= curveCount; ++c)
            offsets[c] = static_cast<uint32_t>(upstream.cvOffsets[c]);
    } else if (curveCount) {
        uint32_t const perCurve = static_cast<uint32_t>(pointCount / curveCount);
        for (size_t c = 0; c <= curveCount; ++c)
            offsets[c] = static_cast<uint32_t>(c * perCurve);
    }
    std::vector<uint64_t> fallbackIds;
    uint64_t const *ids = upstream.curveId.empty() ? nullptr : upstream.curveId.cdata();
    if (!ids && curveCount) {
        fallbackIds.resize(curveCount);
        for (size_t c = 0; c < curveCount; ++c) fallbackIds[c] = c;
        ids = fallbackIds.data();
    }
    expr::CpuCurveGeometryView view;
    view.px = upstream.px.empty() ? nullptr : upstream.px.cdata();
    view.py = upstream.py.empty() ? nullptr : upstream.py.cdata();
    view.pz = upstream.pz.empty() ? nullptr : upstream.pz.cdata();
    view.rest = upstream.rest.empty() ? nullptr
        : reinterpret_cast<float const *>(upstream.rest.cdata());
    view.widths = upstream.width.empty() ? nullptr : upstream.width.cdata();
    view.hairT = upstream.hairT.empty() ? nullptr : upstream.hairT.cdata();
    view.rootUV = upstream.rootUV.empty() ? nullptr
        : reinterpret_cast<float const *>(upstream.rootUV.cdata());
    auto rootFrame = [&](VtVec3fArray const &plane) -> float const * {
        return curveCount && plane.size() == curveCount
            ? reinterpret_cast<float const *>(plane.cdata()) : nullptr;
    };
    view.rootN = rootFrame(upstream.rootN);
    view.rootT = rootFrame(upstream.rootT);
    view.rootB = rootFrame(upstream.rootB);
    view.rootPrim = curveCount && upstream.rootPrim.size() == curveCount
        ? upstream.rootPrim.cdata() : nullptr;
    view.stableIds = ids;
    view.curveOffsets = offsets.data();
    view.curveCount = curveCount;
    view.pointCount = pointCount;

    expr::Context controls;
    controls.seed = static_cast<int32_t>(ctx.seed);
    controls.descId = ctx.desc
        ? expr::DescriptionId(ctx.desc->description.GetText()) : 0;
    controls.domain = domain;
    expr::CpuExpressionContext context;
    std::string contextError;
    if (context.Build(view, controls, &contextError) != expr::CpuExpressionStatus::Ok) {
        if (diag) diag->Error("UsdGenExprOp: expression geometry context validation failed: " +
                              contextError);
        return false;
    }
    expr::CpuExpressionInputs const inputs = context.Inputs();
    VtFloatArray field(inputs.count * components, 0.0f);
    expr::CpuExpressionOutput output;
    output.data = field.empty() ? nullptr : &field[0];
    output.count = inputs.count;
    output.type = expr::ScalarType::Float32;
    output.components = components;
    expr::CpuExpressionStatus const status = expr::EvaluateProgram(ir, inputs, output);
    if (status != expr::CpuExpressionStatus::Ok) {
        if (diag) diag->Error(status == expr::CpuExpressionStatus::InvalidValue
            ? "UsdGenExprOp: expression produced a value the destination cannot "
              "represent (a non-finite result, or a variable with no upstream channel)"
            : "UsdGenExprOp: expression evaluation was refused");
        return false;
    }
    cap->components = static_cast<uint8_t>(components);
    cap->modeCv = modeCv ? 1 : 0;
    cap->curves = upstream.totalCurves;
    cap->cvs = upstream.totalCvs;
    cap->upstreamTopologyVersion = upstream.topologyVersion;
    cap->upstreamValueVersion = upstream.valueVersion;
    cap->perCurve.clear();
    cap->perCv.clear();
    if (modeCv) cap->perCv = field;
    else cap->perCurve = field;
    return true;
}

void UsdGenExprOp::Evaluate(
    UsdGenEvalContext const &ctx,
    UsdGenCapture const &captureIn,
    UsdGenChunkView *view) const
{
    auto const &cap = static_cast<UsdGenExprOpCapture const &>(captureIn);
    UsdGenParamView const *p = ctx.params ? &*ctx.params : nullptr;
    UsdGenParamField const maskField =
        p ? p->GetScalarField(sMask, 1.0) : UsdGenParamField{1.0};
    float defaultInputWidth = ctx.desc ? ctx.desc->defaultWidth : 0.01f;
    if (!std::isfinite(defaultInputWidth) || defaultInputWidth < 0.0f)
        defaultInputWidth = 0.01f;
    float *px = view->px, *py = view->py, *pz = view->pz;
    float *width = view->width;
    auto const *inPx = view->inPx, *inPy = view->inPy, *inPz = view->inPz;
    auto const *inWidth = view->inWidth;
    size_t const curveBase = view->desc ? size_t(view->desc->firstCurve) : 0;
    size_t const cvBase = view->desc ? size_t(view->desc->firstCv) : 0;
    auto const *cvOff = view->cvOffsets;
    bool const ragged = cvOff != nullptr;

    // A capture/topology mismatch cannot happen through the scheduler
    // (ValidForTopology recaptures first); a direct caller with a stale
    // capture gets a bit-for-bit pass-through instead of an overrun.
    bool const isDisplacement = cap.components == 3;
    bool const isWidth = cap.components == 1;
    auto arrayData = [](VtFloatArray const &a) -> float const * {
        return a.empty() ? nullptr : a.cdata();
    };
    float const *field = cap.modeCv ? arrayData(cap.perCv) : arrayData(cap.perCurve);
    bool layoutOk = (isDisplacement || isWidth) && field != nullptr;
    if (layoutOk && view->curveCount) {
        if (cap.modeCv) {
            size_t chunkCvs = 0;
            if (ragged) chunkCvs = size_t(cvOff[view->curveCount] - cvOff[0]);
            else chunkCvs = size_t(view->curveCount) * size_t(view->cvCount);
            layoutOk = cap.perCv.size() >=
                (cvBase + chunkCvs) * size_t(cap.components);
        } else {
            layoutOk = cap.perCurve.size() >=
                (curveBase + size_t(view->curveCount)) * size_t(cap.components);
        }
    }

    for (uint32_t c = 0; c < view->curveCount; ++c) {
        size_t const n = ragged ? size_t(cvOff[c + 1] - cvOff[c]) : size_t(view->cvCount);
        size_t const g = ragged ? size_t(cvOff[c] - cvOff[0]) : view->Cv(c, 0);
        if (!n) continue;
        size_t const curve = curveBase + c;
        for (size_t i = 0; i < n; ++i) {
            size_t const o = g + i;
            size_t const cv = cvBase + o;
            float const input = inWidth ? inWidth[o] : defaultInputWidth;
            if (!layoutOk) {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                width[o] = input;
                continue;
            }
            float const mask = std::clamp(
                static_cast<float>(maskField.Value(curve, cv)), 0.0f, 1.0f);
            size_t const e = cap.modeCv ? cv : curve;
            if (isDisplacement) {
                float const dx = field[e * 3] * mask;
                float const dy = field[e * 3 + 1] * mask;
                float const dz = field[e * 3 + 2] * mask;
                if (dx == 0.0f && dy == 0.0f && dz == 0.0f) {
                    px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                } else {
                    px[o] = inPx[o] + dx;
                    py[o] = inPy[o] + dy;
                    pz[o] = inPz[o] + dz;
                }
                width[o] = input;
            } else {
                px[o] = inPx[o]; py[o] = inPy[o]; pz[o] = inPz[o];
                width[o] = std::max(0.0f, input + (field[e] - input) * mask);
            }
        }
    }
}

UsdGenOpId UsdGenRegisterExprOpOp(UsdGenRegistrar &r)
{
    return r.RegisterLegacyOp(UsdGenExprOp().Type(), [] { return std::make_unique<UsdGenExprOp>(); });
}

}  // namespace usdGen
