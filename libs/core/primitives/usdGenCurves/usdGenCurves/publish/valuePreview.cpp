#include "usdGen/valuePreview.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/token.h"

#include "usdGen/expressions/valueShape.h"
#include "usdGen/graph.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec4i.h"
#include "pxr/usd/sdf/schema.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string_view>
#include <unordered_map>

PXR_NAMESPACE_USING_DIRECTIVE

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

// Made at load, not on first use: no function-local static to guard.
TfToken const kPreviewDestination = UsdGenInternToken("usdGenPreview");
TfToken const kColorRgb = UsdGenInternToken("rgb"), kColorIds = UsdGenInternToken("ids"),
    kColorGray = UsdGenInternToken("gray"), kColorViridis = UsdGenInternToken("viridis");
TfToken const kShadingFlat = UsdGenInternToken("flat");
TfToken const kGroom = UsdGenInternToken("groom"), kPoint = UsdGenInternToken("point");
TfToken const kMapInput = UsdGenInternToken("map");
TfToken const kMapChannel = UsdGenInternToken("map:channel"),
    kMapFirstChannel = UsdGenInternToken("map:firstChannel"),
    kMapChannelCount = UsdGenInternToken("map:channelCount");
// Per-channel destinations of a map preview, and the channel each one reads.
TfToken const kPreviewChannels[] = {UsdGenInternToken("previewR"), UsdGenInternToken("previewG"),
                                    UsdGenInternToken("previewB")};
TfToken const kRgbChannels[] = {UsdGenInternToken("r"), UsdGenInternToken("g"),
                                UsdGenInternToken("b")};

/// Text of a property name without `usdGen:`.
std::string_view CanonicalText(TfToken const &name)
{
    std::string_view text(name.GetString());
    constexpr std::string_view kPrefix("usdGen:");
    return text.substr(0, kPrefix.size()) == kPrefix ? text.substr(kPrefix.size()) : text;
}

TfToken const &PreviewDestination()
{
    return kPreviewDestination;
}

uint64_t Mix(uint64_t h, uint64_t value)
{
    return h ^ (value + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

double Clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

float SrgbToLinear(double c)
{
    c = Clamp01(c);
    return float(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
}

GfVec3f SrgbToLinear(GfVec3f const &c)
{
    return GfVec3f(SrgbToLinear(c[0]), SrgbToLinear(c[1]), SrgbToLinear(c[2]));
}

// Eleven evenly spaced sRGB stops (matplotlib's viridis and inferno).
using Stops = unsigned char const[11][3];
Stops kViridis = {
    {0x44, 0x01, 0x54}, {0x48, 0x24, 0x75}, {0x41, 0x44, 0x87}, {0x35, 0x5f, 0x8d},
    {0x2a, 0x78, 0x8e}, {0x21, 0x91, 0x8c}, {0x22, 0xa8, 0x84}, {0x44, 0xbf, 0x70},
    {0x7a, 0xd1, 0x51}, {0xbd, 0xdf, 0x26}, {0xfd, 0xe7, 0x25}};
Stops kInferno = {
    {0x00, 0x00, 0x04}, {0x16, 0x0b, 0x39}, {0x42, 0x0a, 0x68}, {0x6a, 0x17, 0x6e},
    {0x93, 0x26, 0x67}, {0xbc, 0x37, 0x54}, {0xdd, 0x51, 0x3a}, {0xf3, 0x78, 0x19},
    {0xfc, 0xa5, 0x0a}, {0xf6, 0xd7, 0x46}, {0xfc, 0xff, 0xa4}};

GfVec3f Ramp(Stops const &stops, double t)
{
    double const x = Clamp01(t) * 10.0;
    int const i = std::min(9, int(x));
    double const f = x - i;
    GfVec3f out;
    for (int k = 0; k < 3; ++k)
        out[k] = float((stops[i][k] * (1.0 - f) + stops[i + 1][k] * f) / 255.0);
    return out;
}

// One colour per distinct value, keyed in 1/1024 steps like opUtil::RegionKey
// so strands an operator treats as one region or clump share a colour.
GfVec3f IdColor(double v)
{
    double const scaled = std::max(-9.0e18, std::min(9.0e18, v * 1024.0));
    uint64_t x = uint64_t(std::llround(scaled)) + 0x9e3779b97f4a7c15ULL;  // splitmix64
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    double const h = double(x & 0xffff) / 65536.0 * 6.0;
    double const s = 0.55 + 0.4 * double((x >> 16) & 0xff) / 255.0;
    double const value = 0.7 + 0.3 * double((x >> 24) & 0xff) / 255.0;
    double const f = h - std::floor(h);
    float const p = float(value * (1.0 - s));
    float const q = float(value * (1.0 - s * f));
    float const t = float(value * (1.0 - s * (1.0 - f)));
    float const w = float(value);
    switch (int(h) % 6) {
    case 0: return GfVec3f(w, t, p);
    case 1: return GfVec3f(q, w, p);
    case 2: return GfVec3f(p, w, t);
    case 3: return GfVec3f(p, q, w);
    case 4: return GfVec3f(t, p, w);
    default: return GfVec3f(w, p, q);
    }
}

template <class T>
bool ScalarValue(VtValue const &value, std::vector<double> *out)
{
    if (!value.IsHolding<T>()) return false;
    out->assign(1, double(value.UncheckedGet<T>()));
    return true;
}

template <class T>
bool VectorValue(VtValue const &value, std::vector<double> *out)
{
    if (!value.IsHolding<T>()) return false;
    T const &v = value.UncheckedGet<T>();
    out->resize(T::dimension);
    for (size_t i = 0; i < T::dimension; ++i) (*out)[i] = double(v[i]);
    return true;
}

/// An authored operator value as doubles, or false when it has no colour
/// (a token, an array, a ramp).
bool LiteralValues(VtValue const &value, std::vector<double> *out)
{
    return ScalarValue<float>(value, out) || ScalarValue<double>(value, out) ||
        ScalarValue<int>(value, out) || ScalarValue<bool>(value, out) ||
        ScalarValue<GfHalf>(value, out) || ScalarValue<unsigned int>(value, out) ||
        ScalarValue<int64_t>(value, out) || ScalarValue<uint64_t>(value, out) ||
        VectorValue<GfVec2f>(value, out) || VectorValue<GfVec3f>(value, out) ||
        VectorValue<GfVec4f>(value, out) || VectorValue<GfVec2d>(value, out) ||
        VectorValue<GfVec3d>(value, out) || VectorValue<GfVec4d>(value, out) ||
        VectorValue<GfVec2h>(value, out) || VectorValue<GfVec3h>(value, out) ||
        VectorValue<GfVec4h>(value, out) || VectorValue<GfVec2i>(value, out) ||
        VectorValue<GfVec3i>(value, out) || VectorValue<GfVec4i>(value, out);
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
std::vector<UsdGenNodeKey> RootSurfaces(UsdGenGraphSpec const &desc)
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

void SetParam(UsdGenMapInput *map, TfToken const &name, VtValue value)
{
    for (UsdGenParamValue &param : map->params) {
        if (param.name == name) {
            param.value = std::move(value);
            return;
        }
    }
    map->params.push_back({name, std::move(value), false});
}

void EraseParam(UsdGenMapInput *map, char const *name)
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

GfVec3f UsdGenPreviewColor(TfToken const &colorMap, GfVec2f const &range,
                           double const *values, unsigned components)
{
    TfToken const &rgb = kColorRgb, &ids = kColorIds, &gray = kColorGray, &viridis = kColorViridis;
    if (!values || components == 0) return UsdGenPreviewMissingColor();
    for (unsigned k = 0; k < components; ++k)
        if (!std::isfinite(values[k])) return GfVec3f(1.0f, 0.0f, 1.0f);
    double const lo = range[0], hi = range[1];
    auto normalise = [&](double v) {
        if (hi == lo) return v >= hi ? 1.0 : 0.0;
        return Clamp01((v - lo) / (hi - lo));
    };
    if (colorMap == rgb) {
        GfVec3f out(0.0f);
        for (unsigned k = 0; k < 3; ++k)
            out[k] = float(normalise(components == 1 ? values[0]
                                     : k < components ? values[k] : lo));
        return out;
    }
    double v = values[0];
    if (components > 1) {
        double sum = 0.0;
        for (unsigned k = 0; k < components; ++k) sum += values[k] * values[k];
        v = std::sqrt(sum);
    }
    if (colorMap == ids) return SrgbToLinear(IdColor(v));
    double const t = normalise(v);
    if (colorMap == gray) return SrgbToLinear(GfVec3f(float(t)));
    return SrgbToLinear(Ramp(colorMap == viridis ? kViridis : kInferno, t));
}

GfVec3f UsdGenPreviewMissingColor()
{
    return GfVec3f(0.02f, 0.02f, 0.02f);
}

UsdGenPublishRole UsdGenPreviewMaterialRole(TfToken const &shading)
{
    return shading == kShadingFlat ? UsdGenPublishRole::PreviewMaterialFlat
                                   : UsdGenPublishRole::PreviewMaterial;
}

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
    auto const expression = std::find_if(desc.expressions.begin(), desc.expressions.end(),
        [&](UsdGenExpressionInput const &e) { return e.key == preview.source; });
    UsdGenExpressionOutputDesc const *output =
        UsdGenFindExpressionOutput(*expression, TfToken());
    if (!output) {
        *warning = SourceText(desc) +
            " declares no outputs:result, and not exactly one other output";
        return false;
    }
    Domain domain = Domain::None;
    if (preview.evaluation == kGroom) domain = Domain::Groom;
    else if (preview.evaluation == UsdGenTokens().primitive) domain = Domain::Primitive;
    else if (preview.evaluation == kPoint) domain = Domain::Point;
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
    binding.literal = SdfSchema::GetInstance().FindType(output->nativeType).GetDefaultValue();
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
    auto const map = std::find_if(desc.maps.begin(), desc.maps.end(),
        [&](UsdGenMapInput const &m) { return m.key == preview.source; });
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
    _mapDesc.surfaces = desc.surfaces;
    _mapDesc.timeCodesPerSecond = desc.timeCodesPerSecond;
    expr::ValueShape const floatShape = expr::ValueShapeFromNativeType(UsdGenTokens().float_);

    // "rgb" reads the first three texture channels unscaled; a map with fewer
    // falls back to the value an operator reads, usdGen:map:channel.
    std::string colourFailure;
    for (int attempt = preview.colorMap == kColorRgb ? 0 : 1; attempt < 2; ++attempt) {
        bool const colour = attempt == 0;
        unsigned const channels = colour ? 3u : 1u;
        _mapDesc.maps.clear();
        _mapDesc.expressions.clear();
        UsdGenOpNode node = PreviewNode(desc);
        for (unsigned k = 0; k < channels; ++k) {
            UsdGenMapInput copy = *map;
            copy.key = _mapDesc.DeriveKey(map->key, kPreviewChannels[k]);
            if (colour) {
                SetParam(&copy, kMapChannel, VtValue(kRgbChannels[k]));
                SetParam(&copy, kMapFirstChannel, VtValue(0));
                SetParam(&copy, kMapChannelCount, VtValue(3));
                EraseParam(&copy, "map:scale");
                EraseParam(&copy, "map:offset");
                EraseParam(&copy, "map:clamp");
            }
            UsdGenExpressionInput expression;
            expression.key = copy.key;
            expression.source = "ptex(\"map\")";
            expression.inputs.push_back({kMapInput, {copy.key}, {}, {copy.key}});
            expression.outputs.push_back({UsdGenTokens().result, UsdGenTokens().float_, floatShape});
            UsdGenParamBinding binding;
            binding.expression = copy.key;
            binding.output = UsdGenTokens().result;
            binding.nativeType = UsdGenTokens().float_;
            binding.destination = kPreviewChannels[k];
            binding.destinationShape = floatShape;
            binding.domain = Domain::Primitive;
            binding.literal = VtValue(0.0f);
            node.expressionBindings.push_back(std::move(binding));
            _mapDesc.maps.push_back(std::move(copy));
            _mapDesc.expressions.push_back(std::move(expression));
        }
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
            UsdGenExpressionValue const *value = _parameters.Find(kPreviewChannels[k]);
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
    TfToken const &attribute = desc.preview.attribute;
    std::string_view const name = CanonicalText(attribute);
    for (int id = 0; id < graph.NodeCount(); ++id) {
        UsdGenCompiledNode const &node = graph.Node(id);
        if (!node.desc || node.desc->key != operatorKey) continue;
        // Connected values were evaluated over the operator's INPUT strands.
        values->input = UsdGenCurveSetView::Of(
            graph.NodeOutput(node.inputs.empty() ? node.id : node.inputs.front()));
        UsdGenCurveSetView const &input = values->input;
        bool const connected = std::any_of(
            node.desc->expressionBindings.begin(), node.desc->expressionBindings.end(),
            [&](UsdGenParamBinding const &b) {
                return CanonicalText(b.destination) == name;
            });
        if (connected) {
            UsdGenExpressionValue const *value = node.expressions.Find(attribute);
            if (!value || value->values.empty()) {
                *warning = SourceText(desc) +
                    " is connected to an expression that did not evaluate";
                return false;
            }
            size_t const expected = value->domain == Domain::Primitive ? input.totalCurves
                : value->domain == Domain::Point ? input.totalCvs : 1;
            if (value->count != expected) {
                *warning = SourceText(desc) + " was evaluated over " +
                    std::to_string(value->count) + " elements, but the operator's input has " +
                    std::to_string(expected);
                return false;
            }
            values->domain = value->domain;
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

UsdGenPreviewColors UsdGenValuePreview::Build(UsdGenGraphSpec const &desc,
                                              UsdGenGraph const &graph,
                                              UsdGenCurveSetView const &terminal,
                                              double frame,
                                              std::vector<std::string> *warnings)
{
    UsdGenPreviewColors out;
    UsdGenPreviewSpec const &preview = desc.preview;
    if (!preview.Active()) {
        _lastWarning.clear();
        return out;
    }
    out.active = true;

    Values values;
    std::string warning;
    bool ok = false;
    UsdGenNodeKey const source = preview.source;
    auto const isExpression = [&](UsdGenExpressionInput const &e) { return e.key == source; };
    auto const isMap = [&](UsdGenMapInput const &m) { return m.key == source; };
    if (!preview.attribute.IsEmpty()) {
        ok = FromAttribute(desc, graph, &values, &warning);
    } else if (std::any_of(desc.expressions.begin(), desc.expressions.end(), isExpression)) {
        ok = FromExpression(desc, terminal, frame, &values, &warning);
    } else if (std::any_of(desc.maps.begin(), desc.maps.end(), isMap)) {
        ok = FromMap(desc, terminal, frame, &values, &warning);
    } else {
        warning = SourceText(desc) + " is not a UsdGenExpression under " +
            desc.Label(desc.description) + "/Expressions, a UsdGenPtexMap, or an operator "
            "attribute of this description";
    }

    out.perCv = ok && values.domain == Domain::Point;
    size_t const elements = out.perCv ? terminal.totalCvs : terminal.totalCurves;
    out.colors.assign(elements, UsdGenPreviewMissingColor());
    if (ok && values.count && values.data) {
        unsigned const k = values.components;
        auto colour = [&](size_t i) {
            return UsdGenPreviewColor(preview.colorMap, preview.range, values.data + i * k, k);
        };
        if (values.domain == Domain::Groom) {
            std::fill(out.colors.begin(), out.colors.end(), colour(0));
        } else {
            UsdGenCurveSetView const &from = *values.geometry;
            std::vector<size_t> const match = MatchCurves(from, terminal);
            size_t matched = 0;
            if (values.domain == Domain::Primitive) {
                for (size_t c = 0; c < match.size(); ++c) {
                    if (match[c] >= values.count) continue;
                    out.colors[c] = colour(match[c]);
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
                        if (index < values.count && target < elements)
                            out.colors[target] = colour(index);
                    }
                }
            }
            if (matched < match.size() && warning.empty())
                warning = SourceText(desc) + ": " + std::to_string(match.size() - matched) +
                    " of " + std::to_string(match.size()) +
                    " published strands have no counterpart where the value was evaluated";
        }
    }

    uint64_t digest = Mix(1469598103934665603ULL, out.perCv ? 2u : 1u);
    digest = Mix(digest, preview.shading.Hash());
    for (GfVec3f const &c : out.colors) {
        uint32_t bits[3];
        std::memcpy(bits, c.data(), sizeof(bits));
        digest = Mix(Mix(Mix(digest, bits[0]), bits[1]), bits[2]);
    }
    out.digest = digest | 1u;

    if (warning != _lastWarning) {
        if (!warning.empty() && warnings)
            warnings->push_back("usdGen:preview: " + warning);
        _lastWarning = warning;
    }
    return out;
}

} // namespace usdGen
