// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/preview/valuePreview.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenExpr/expressions/valueShape.h"
#include "usdGen/api/compile.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/op.h"
#include "usdGen/api/ext/tokenScope.h"

#include "usdGenMath/math.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGenCurves/types/curveTokens.h"
#include "usdGen/api/span.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>


namespace usdGen {

/// The values a preview colours: `count` elements of `components` doubles,
/// indexed over `geometry`'s curves (primitive) or CVs (point).
struct UsdGenValuePreview::Values
{
    expr::Domain domain = expr::Domain::Groom;
    unsigned components = 1;
    size_t count = 0;
    double const *data = nullptr;
    UsdGenCurveSetView const *geometry = nullptr;
    UsdGenCurveSetView input;      // an operator's input strands (FromAttribute)
};

namespace {

using Domain = expr::Domain;

// Preview names live in UsdGenCurveTokens (interned at registration, never at
// load). previewChannels are the per-channel destinations of a map preview,
// rgbChannels the channel each one reads.

/// Text of a property name without `usdGen:`.
std::string_view CanonicalText(UsdGenToken const &name)
{
    std::string_view text(name.GetString());
    constexpr std::string_view kPrefix("usdGen:");
    return text.substr(0, kPrefix.size()) == kPrefix ? text.substr(kPrefix.size()) : text;
}

UsdGenToken const &PreviewDestination()
{
    return UsdGenCurveTokens().previewDestination;
}

uint64_t Mix(uint64_t h, uint64_t value)
{
    return h ^ (value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

template <class T>
bool ScalarValue(UsdGenValue const &value, std::vector<double> *out)
{
    if (!value.IsHolding<T>()) return false;
    out->assign(1, double(value.UncheckedGet<T>()));
    return true;
}

template <class T>
bool VectorValue(UsdGenValue const &value, std::vector<double> *out)
{
    if (!value.IsHolding<T>()) return false;
    T const &v = value.UncheckedGet<T>();
    out->resize(T::dimension);
    for (size_t i = 0; i < T::dimension; ++i) (*out)[i] = double(v[i]);
    return true;
}

/// An authored operator value as doubles, or false when it has no colour
/// (a token, an array, a ramp).
bool LiteralValues(UsdGenValue const &value, std::vector<double> *out)
{
    return ScalarValue<float>(value, out) || ScalarValue<double>(value, out) ||
        ScalarValue<int>(value, out) || ScalarValue<bool>(value, out) ||
        ScalarValue<UsdGenHalf>(value, out) || ScalarValue<unsigned int>(value, out) ||
        ScalarValue<int64_t>(value, out) || ScalarValue<uint64_t>(value, out) ||
        VectorValue<UsdGenVec2f>(value, out) || VectorValue<UsdGenVec3f>(value, out) ||
        VectorValue<UsdGenVec4f>(value, out) || VectorValue<UsdGenVec2d>(value, out) ||
        VectorValue<UsdGenVec3d>(value, out) || VectorValue<UsdGenVec4d>(value, out) ||
        VectorValue<UsdGenVec2h>(value, out) || VectorValue<UsdGenVec3h>(value, out) ||
        VectorValue<UsdGenVec4h>(value, out) || VectorValue<UsdGenVec2i>(value, out) ||
        VectorValue<UsdGenVec3i>(value, out) || VectorValue<UsdGenVec4i>(value, out);
}

/// First CV of every curve, plus the end.
std::vector<size_t> CurveOffsets(UsdGenCurveSetView const &buffer)
{
    size_t const curves = buffer.totalCurves;
    std::vector<size_t> offsets(curves + 1, 0);
    if (buffer.cvOffsets.size() == curves + 1) {
        for (size_t c = 0; c <= curves; ++c) offsets[c] = size_t(buffer.cvOffsets[c]);
    } else if (curves) {
        size_t const perCurve = buffer.totalCvs / curves;
        for (size_t c = 0; c <= curves; ++c) offsets[c] = c * perCurve;
    }
    return offsets;
}

size_t const kNoMatch = std::numeric_limits<size_t>::max();

/// For each curve of `to`, its index in `from`: the same index when the two
/// are one buffer or carry no ids, else the curve with the same id.
std::vector<size_t> MatchCurves(UsdGenCurveSetView const &from, UsdGenCurveSetView const &to)
{
    std::vector<size_t> match(to.totalCurves, kNoMatch);
    bool const ids = from.curveId.size() == from.totalCurves &&
                     to.curveId.size() == to.totalCurves;
    bool sameOrder = &from == &to ||
        (from.totalCurves == to.totalCurves &&
         (!ids || std::equal(from.curveId.begin(), from.curveId.end(), to.curveId.begin())));
    if (sameOrder) {
        for (size_t c = 0; c < match.size(); ++c) match[c] = c;
        return match;
    }
    if (!ids) return match;
    std::unordered_map<uint64_t, size_t> index;
    index.reserve(from.totalCurves);
    for (size_t c = 0; c < from.totalCurves; ++c) index.emplace(from.curveId[c], c);
    for (size_t c = 0; c < to.totalCurves; ++c) {
        auto const found = index.find(to.curveId[c]);
        if (found != index.end()) match[c] = found->second;
    }
    return match;
}

/// The surface every operator of the description roots on (ADR R15), which is
/// what ptex() needs to place a strand root on a face.
UsdGenPoolVector<UsdGenNodeKey> RootSurfaces(UsdGenGraphSpec const &desc)
{
    for (UsdGenOpNode const &node : desc.nodes)
        if (!node.surfaces.empty()) return node.surfaces;
    return {};
}

UsdGenOpNode PreviewNode(UsdGenGraphSpec const &desc)
{
    UsdGenOpNode node;
    node.key = desc.description;
    // No op type: the preview evaluates bindings only.
    node.surfaces = RootSurfaces(desc);
    return node;
}

void SetParam(UsdGenMapData *map, UsdGenToken const &name, UsdGenValue value)
{
    for (UsdGenParamValue &param : map->params) {
        if (param.name == name) {
            param.value = std::move(value);
            return;
        }
    }
    map->params.push_back({name, std::move(value), false});
}

void EraseParam(UsdGenMapData *map, char const *name)
{
    map->params.erase(std::remove_if(map->params.begin(), map->params.end(),
                                     [&](UsdGenParamValue const &p) { return p.name == name; }),
                      map->params.end());
}

std::string Join(std::vector<std::string> const &messages)
{
    std::string text;
    for (std::string const &message : messages) {
        if (!text.empty()) text += "; ";
        text += message;
    }
    return text;
}

/// The previewed source's text: its prim, then '.' and the attribute.
std::string SourceText(UsdGenGraphSpec const &desc)
{
    std::string text = desc.Label(desc.preview.source);
    if (!desc.preview.attribute.IsEmpty()) {
        text += '.';
        text += desc.preview.attribute.GetString();
    }
    return text;
}

} // namespace

bool UsdGenValuePreview::Evaluate(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                                  UsdGenCurveSetView const &terminal, double frame,
                                  std::string *message)
{
    double const rate = desc.timeCodesPerSecond;
    double const seconds = std::isfinite(rate) && rate > 0.0 ? frame / rate : frame;
    std::vector<std::string> diagnostics;
    bool changed = false;
    bool const ok = _parameters.Evaluate(desc, node, terminal, frame, seconds, 0,
                                         &changed, &diagnostics);
    for (std::string &warning : _parameters.TakeWarnings())
        diagnostics.push_back(std::move(warning));
    *message = Join(diagnostics);
    return ok;
}

bool UsdGenValuePreview::FromExpression(UsdGenGraphSpec const &desc,
                                        UsdGenCurveSetView const &terminal, double frame,
                                        Values *values, std::string *warning)
{
    UsdGenPreviewSpec const &preview = desc.preview;
    UsdGenInputRecord const *expressionRecord = UsdGenFindExpressionInput(desc, preview.source);
    UsdGenExpressionInputView const expressionView = expressionRecord
        ? UsdGenExpressionInputView::Of(*expressionRecord) : UsdGenExpressionInputView{};
    UsdGenExpressionInputView const *expression = &expressionView;
    UsdGenExpressionOutputDesc const *output =
        UsdGenFindExpressionOutput(*expression, UsdGenToken());
    if (!output) {
        *warning = SourceText(desc) +
            " declares no outputs:result, and not exactly one other output";
        return false;
    }
    Domain domain = Domain::None;
    if (preview.evaluation == UsdGenCurveTokens().groom) domain = Domain::Groom;
    else if (preview.evaluation == UsdGenTokens().primitive) domain = Domain::Primitive;
    else if (preview.evaluation == UsdGenCurveTokens().point) domain = Domain::Point;
    else {
        *warning = "usdGen:preview:evaluation \"" + preview.evaluation.GetString() +
            "\" is not groom, primitive or point";
        return false;
    }

    UsdGenOpNode node = PreviewNode(desc);
    UsdGenParamBinding binding;
    binding.expression = expression->key;
    binding.output = output->name;
    binding.nativeType = output->nativeType;
    binding.destination = PreviewDestination();
    binding.destinationShape = output->shape;
    binding.domain = domain;
    // $value is the literal of the attribute an expression drives; a preview
    // drives nothing, so it reads zero of the output's own type.
    binding.literal = expr::ValueTypeDefault(expr::FindValueType(output->nativeType));
    node.expressionBindings.push_back(std::move(binding));

    std::string message;
    if (!Evaluate(desc, node, terminal, frame, &message)) {
        *warning = message;
        return false;
    }
    UsdGenExpressionValue const *value = _parameters.Find(PreviewDestination());
    if (!value) {
        *warning = SourceText(desc) + " produced no value";
        return false;
    }
    *warning = message;
    values->domain = value->domain;
    values->components = value->components;
    values->count = value->count;
    values->data = value->values.data();
    values->geometry = &terminal;
    return true;
}

bool UsdGenValuePreview::FromMap(UsdGenGraphSpec const &desc,
                                 UsdGenCurveSetView const &terminal, double frame,
                                 Values *values, std::string *warning)
{
    UsdGenPreviewSpec const &preview = desc.preview;
    UsdGenInputRecord const *mapRecord = UsdGenFindMapInput(desc, preview.source);
    UsdGenMapInputView const mapView =
        mapRecord ? UsdGenMapInputView::Of(*mapRecord) : UsdGenMapInputView{};
    UsdGenMapInputView const *map = &mapView;
    if (!UsdGenMapHas(*map, kUsdGenTypeMapPtex)) {
        *warning = SourceText(desc) + " is a " + map->type.GetString() +
            "; only a UsdGenPtexMap can be previewed";
        return false;
    }
    // ptex("map") in a world of its own: the description's surfaces and
    // placement, and a copy of the map the expression's input targets.
    _mapDesc.description = desc.description;
    _mapDesc.labels = desc.labels;
    _mapDesc.xformMatrix = desc.xformMatrix;
    UsdGenSpan<const UsdGenInputRecord> const surfaces = UsdGenSurfaceInputs(desc);
    _mapDesc.timeCodesPerSecond = desc.timeCodesPerSecond;
    expr::ValueShape const floatShape = expr::ValueShapeFromNativeType(UsdGenTokens().float_);

    // "rgb" reads the first three texture channels unscaled; a map with fewer
    // falls back to the value an operator reads, usdGen:map:channel.
    std::string colourFailure;
    for (int attempt = preview.mapChannels == 3 ? 0 : 1; attempt < 2; ++attempt) {
        bool const colour = attempt == 0;
        unsigned const channels = colour ? 3u : 1u;
        _mapDesc.inputs.assign(surfaces.begin(), surfaces.end());
        UsdGenOpNode node = PreviewNode(desc);
        for (unsigned k = 0; k < channels; ++k) {
            UsdGenMapData copy;
            UsdGenCopyMapData(*map, &copy);
            UsdGenNodeKey const copyKey = _mapDesc.DeriveKey(map->key, UsdGenCurveTokens().previewChannels[k]);
            if (colour) {
                SetParam(&copy, UsdGenCurveTokens().mapChannel, UsdGenValue(UsdGenCurveTokens().rgbChannels[k]));
                SetParam(&copy, UsdGenCurveTokens().mapFirstChannel, UsdGenValue(0));
                SetParam(&copy, UsdGenCurveTokens().mapChannelCount, UsdGenValue(3));
                EraseParam(&copy, "map:scale");
                EraseParam(&copy, "map:offset");
                EraseParam(&copy, "map:clamp");
            }
            UsdGenExpressionData expression;
            expression.source = "ptex(\"map\")";
            expression.inputs.push_back({UsdGenCurveTokens().map, {copyKey}, {}, {copyKey}});
            expression.outputs.push_back({UsdGenTokens().result, UsdGenTokens().float_, floatShape});
            UsdGenParamBinding binding;
            binding.expression = copyKey;
            binding.output = UsdGenTokens().result;
            binding.nativeType = UsdGenTokens().float_;
            binding.destination = UsdGenCurveTokens().previewChannels[k];
            binding.destinationShape = floatShape;
            binding.domain = Domain::Primitive;
            binding.literal = UsdGenValue(0.0f);
            node.expressionBindings.push_back(std::move(binding));
            _mapDesc.inputs.push_back(UsdGenMapRecord(copy, copyKey, map->paintSurface, UsdGenText()));
            _mapDesc.inputs.push_back(UsdGenExpressionRecord(expression, copyKey, UsdGenText()));
        }
        _mapDesc.SortInputs();
        std::string message;
        if (!Evaluate(_mapDesc, node, terminal, frame, &message)) {
            if (colour) {
                colourFailure = message;
                continue;
            }
            *warning = message;
            return false;
        }
        size_t const curves = terminal.totalCurves;
        _scratch.assign(curves * channels, 0.0);
        for (unsigned k = 0; k < channels; ++k) {
            UsdGenExpressionValue const *value = _parameters.Find(UsdGenCurveTokens().previewChannels[k]);
            if (!value || value->count != curves) {
                *warning = SourceText(desc) + " produced no value per strand";
                return false;
            }
            for (size_t c = 0; c < curves; ++c) _scratch[c * channels + k] = value->values[c];
        }
        *warning = colourFailure.empty() ? message
            : SourceText(desc) + " cannot be shown as rgb (" + colourFailure +
              "); showing its usdGen:map:channel instead";
        values->domain = Domain::Primitive;
        values->components = channels;
        values->count = curves;
        values->data = _scratch.data();
        values->geometry = &terminal;
        return true;
    }
    return false;
}

bool UsdGenValuePreview::FromAttribute(UsdGenGraphSpec const &desc, UsdGenGraph const &graph,
                                       Values *values, std::string *warning)
{
    UsdGenNodeKey const operatorKey = desc.preview.source;
    UsdGenToken const &attribute = desc.preview.attribute;
    std::string_view const name = CanonicalText(attribute);
    for (int id = 0; id < UsdGenCompiledNodeCount(graph); ++id) {
        UsdGenCompiledNodeView const node = UsdGenCompiledNodeAt(graph, UsdGenNodeId(id));
        if (!node.desc || node.desc->key != operatorKey) continue;
        // Connected values were evaluated over the operator's INPUT strands.
        values->input = UsdGenCurveSetView::Of(
            UsdGenCompiledNodeOutput(graph, node.inputs.empty() ? node.id : node.inputs.front()));
        UsdGenCurveSetView const &input = values->input;
        bool const connected = std::any_of(
            node.desc->expressionBindings.begin(), node.desc->expressionBindings.end(),
            [&](UsdGenParamBinding const &b) {
                return CanonicalText(b.destination) == name;
            });
        if (connected) {
            UsdGenEvaluatedParam const *value =
                node.expressions ? node.expressions->Find(attribute) : nullptr;
            if (!value || value->values.empty()) {
                *warning = SourceText(desc) +
                    " is connected to an expression that did not evaluate";
                return false;
            }
            Domain const domain = value->level == 1 ? Domain::Primitive
                : value->level >= 2 ? Domain::Point : Domain::Groom;
            size_t const expected = domain == Domain::Primitive ? input.totalCurves
                : domain == Domain::Point ? input.totalCvs : 1;
            if (value->count != expected) {
                *warning = SourceText(desc) + " was evaluated over " +
                    std::to_string(value->count) + " elements, but the operator's input has " +
                    std::to_string(expected);
                return false;
            }
            values->domain = domain;
            values->components = value->components;
            values->count = value->count;
            values->data = value->values.data();
            values->geometry = &input;
            return true;
        }
        for (UsdGenParamValue const &param : node.desc->params) {
            if (std::string_view(param.name.GetString()) != name) continue;
            if (!LiteralValues(param.value, &_scratch)) {
                *warning = SourceText(desc) + " holds a " + param.value.GetTypeName() +
                    ", which has no colour";
                return false;
            }
            values->domain = Domain::Groom;
            values->components = unsigned(_scratch.size());
            values->count = 1;
            values->data = _scratch.data();
            values->geometry = &input;
            return true;
        }
        *warning = SourceText(desc) + " has no value on this operator";
        return false;
    }
    *warning = desc.Label(operatorKey) + " is not an operator of " +
        desc.Label(desc.description);
    return false;
}

void UsdGenValuePreview::Build(UsdGenGraphSpec const &desc, UsdGenGraph const &graph,
                               UsdGenCurveSetView const &terminal, double frame,
                               UsdGenPreviewValues *out, UsdGenDiagnostics *diag)
{
    *out = UsdGenPreviewValues{};
    UsdGenPreviewSpec const &preview = desc.preview;
    if (!preview.Active()) {
        _lastWarning.clear();
        return;
    }
    out->active = true;

    Values values;
    std::string warning;
    bool ok = false;
    UsdGenNodeKey const source = preview.source;
    if (!preview.attribute.IsEmpty()) {
        ok = FromAttribute(desc, graph, &values, &warning);
    } else if (UsdGenFindExpressionInput(desc, source)) {
        ok = FromExpression(desc, terminal, frame, &values, &warning);
    } else if (UsdGenFindMapInput(desc, source)) {
        ok = FromMap(desc, terminal, frame, &values, &warning);
    } else {
        warning = SourceText(desc) + " is not a UsdGenExpression under " +
            desc.Label(desc.description) + "/Expressions, a UsdGenPtexMap, or an operator "
            "attribute of this description";
    }

    bool const perCv = ok && values.domain == Domain::Point;
    size_t const elements = perCv ? terminal.totalCvs : terminal.totalCurves;
    unsigned const k = values.components;
    _values.clear();
    _present.clear();
    if (ok && values.count && values.data) {
        _values.assign(elements * k, 0.0);
        _present.assign(elements, 0);
        auto take = [&](size_t target, size_t index) {
            std::copy(values.data + index * k, values.data + (index + 1) * k,
                      _values.data() + target * k);
            _present[target] = 1;
        };
        if (values.domain == Domain::Groom) {
            for (size_t i = 0; i < elements; ++i) take(i, 0);
        } else {
            UsdGenCurveSetView const &from = *values.geometry;
            std::vector<size_t> const match = MatchCurves(from, terminal);
            size_t matched = 0;
            if (values.domain == Domain::Primitive) {
                for (size_t c = 0; c < match.size(); ++c) {
                    if (match[c] >= values.count) continue;
                    take(c, match[c]);
                    ++matched;
                }
            } else {
                std::vector<size_t> const fromOffsets = CurveOffsets(from);
                std::vector<size_t> const toOffsets = CurveOffsets(terminal);
                for (size_t c = 0; c < match.size(); ++c) {
                    size_t const m = match[c];
                    if (m >= from.totalCurves) continue;
                    size_t const length = toOffsets[c + 1] - toOffsets[c];
                    size_t const sourceLength = fromOffsets[m + 1] - fromOffsets[m];
                    if (!sourceLength) continue;
                    ++matched;
                    // A resampled strand reads the source CV at the same
                    // fraction of its length.
                    for (size_t v = 0; v < length; ++v) {
                        size_t const sv = sourceLength == length ? v
                            : length > 1 ? size_t(std::llround(double(v) * double(sourceLength - 1) /
                                                               double(length - 1)))
                            : 0;
                        size_t const index = fromOffsets[m] + std::min(sv, sourceLength - 1);
                        size_t const target = toOffsets[c] + v;
                        if (index < values.count && target < elements) take(target, index);
                    }
                }
            }
            if (matched < match.size() && warning.empty())
                warning = SourceText(desc) + ": " + std::to_string(match.size() - matched) +
                    " of " + std::to_string(match.size()) +
                    " published strands have no counterpart where the value was evaluated";
        }
    }
    out->level = perCv ? 2 : 1;
    out->components = uint8_t(k);
    out->values = UsdGenSpan<const double>(_values.data(), _values.size());
    out->present = UsdGenSpan<const uint8_t>(_present.data(), _present.size());

    uint64_t digest = Mix(1469598103934665603ULL, out->level);
    digest = Mix(digest, k);
    for (double const v : _values) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        digest = Mix(digest, bits);
    }
    for (uint8_t const p : _present) digest = Mix(digest, p);
    out->digest = digest | 1u;

    if (warning != _lastWarning) {
        if (!warning.empty() && diag) diag->Warn("usdGen:preview: " + warning);
        _lastWarning = warning;
    }
}

} // namespace usdGen
