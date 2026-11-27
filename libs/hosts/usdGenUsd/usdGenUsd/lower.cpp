// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/lower.h"

#include "usdGenUsd/convert/math.h"

#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenSurface/surfaceInput.h"

#include <algorithm>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {

using namespace usdGen;

PathTable::PathTable(std::vector<std::pair<UsdGenNodeKey, SdfPath>> entries,
                     UsdGenPresentation presentation)
    : _byKey(std::move(entries)), _presentation(std::move(presentation))
{
    std::sort(_byKey.begin(), _byKey.end(),
              [](auto const &a, auto const &b) { return a.first < b.first; });
    _byKey.erase(std::unique(_byKey.begin(), _byKey.end(),
                             [](auto const &a, auto const &b) { return a.first == b.first; }),
                 _byKey.end());
    _byPath.reserve(_byKey.size());
    for (auto const &entry : _byKey) _byPath.emplace_back(entry.second, entry.first);
    std::sort(_byPath.begin(), _byPath.end(),
              [](auto const &a, auto const &b) { return a.first < b.first; });
}

SdfPath const &PathTable::Path(UsdGenNodeKey key) const
{
    auto const it = std::lower_bound(_byKey.begin(), _byKey.end(), key,
        [](auto const &e, UsdGenNodeKey k) { return e.first < k; });
    return it != _byKey.end() && it->first == key ? it->second : SdfPath::EmptyPath();
}

UsdGenNodeKey PathTable::Key(SdfPath const &path) const
{
    auto const it = std::lower_bound(_byPath.begin(), _byPath.end(), path,
        [](auto const &e, SdfPath const &p) { return e.first < p; });
    return it != _byPath.end() && it->first == path ? it->second : UsdGenNodeKey{};
}

void PathTable::KeysUnder(SdfPath const &prim, std::vector<UsdGenNodeKey> *out) const
{
    if (!out || prim.IsEmpty()) return;
    auto it = std::lower_bound(_byPath.begin(), _byPath.end(), prim,
        [](auto const &e, SdfPath const &p) { return e.first < p; });
    for (; it != _byPath.end() && it->first.HasPrefix(prim); ++it) out->push_back(it->second);
}

LoweredParams::LoweredParams(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                             UsdGenOpRecord const &record)
    : _spec(Lower(desc)), _node(Lower(node))
{
    UsdGenResolveParams(record, _node, _slots);
    for (UsdGenResolvedParam &slot : _slots)
        if (slot.authored)
            slot.authored = &node.params[size_t(slot.authored - _node.params.data())];
    _view.desc = &_spec;
    _view.node = &_node;
    _view.op = &record;
    _view.slots = _slots;
}

UsdGenNodeKey PathRecorder::operator()(SdfPath const &path)
{
    UsdGenNodeKey const key = _map.Key(path);
    if (key.IsValid()) _entries.emplace_back(key, path);
    return key;
}

std::shared_ptr<const PathTable> PathRecorder::Table() const
{
    return std::make_shared<PathTable>(_entries);
}

namespace {

class Lowering {
public:
    explicit Lowering(NodeKeyMap &map) : _map(map) {}

    UsdGenInputColumnCache *Columns() noexcept { return &_map.Columns(); }

    UsdGenNodeKey Key(SdfPath const &path)
    {
        UsdGenNodeKey const key = _map.Key(path);
        if (key.IsValid()) _entries.emplace_back(key, path);
        return key;
    }
    std::vector<UsdGenNodeKey> Keys(SdfPathVector const &paths)
    {
        std::vector<UsdGenNodeKey> keys;
        keys.reserve(paths.size());
        for (SdfPath const &path : paths) keys.push_back(Key(path));
        return keys;
    }
    void Finish(UsdGenGraphSpec *spec, UsdGenPresentation presentation = UsdGenPresentation())
    {
        auto table = std::make_shared<PathTable>(std::move(_entries), std::move(presentation));
        spec->labels.clear();
        spec->labels.reserve(table->Size());
        for (auto const &entry : table->Entries())
            spec->labels.push_back(UsdGenNodeLabel{entry.first, UsdGenNodeKey{},
                                                   entry.second.GetAsToken()});
        spec->host = std::move(table);
    }

private:
    NodeKeyMap &_map;
    std::vector<std::pair<UsdGenNodeKey, SdfPath>> _entries;
};

UsdGenInputRecord LowerCurves(UsdGenCurveSetDesc const &curves, Lowering &keys)
{
    return UsdGenCurveSetRecord(curves, keys.Key(curves.path), curves.path.GetAsToken(),
                                keys.Columns());
}

UsdGenInputRecord LowerSurface(UsdGenSurfaceDesc const &surface, Lowering &keys)
{
    UsdGenNodeKey const key = keys.Key(surface.path);
    // The parent prim: a GeomSubset's Mesh (consumers that find a
    // topology-less target look its geometry up here).
    UsdGenNodeKey owner;
    if (!surface.path.IsEmpty() && !surface.path.IsAbsoluteRootPath())
        owner = keys.Key(surface.path.GetParentPath());
    return UsdGenSurfaceRecord(surface, key, owner, surface.path.GetAsToken(), keys.Columns());
}

UsdGenInputRecord LowerMap(UsdGenMapDesc const &map, Lowering &keys)
{
    UsdGenNodeKey const key = keys.Key(map.path);
    UsdGenMapData data = map;
    for (UsdGenParamValue &p : data.params) p.value = ToEngineValue(p.value);
    return UsdGenMapRecord(data, key, keys.Key(map.paintSurface), map.path.GetAsToken(),
                           keys.Columns());
}

UsdGenInputRecord LowerExpression(UsdGenExpressionDesc const &expression, Lowering &keys)
{
    UsdGenExpressionData data;
    UsdGenNodeKey const key = keys.Key(expression.path);
    data.source = expression.source;
    data.outputs = expression.outputs;
    data.inputs.reserve(expression.inputs.size());
    for (UsdGenExpressionInputDesc const &input : expression.inputs)
        data.inputs.push_back(UsdGenExpressionSlot{input.name, keys.Keys(input.targets),
                                                   keys.Keys(input.geometries),
                                                   keys.Keys(input.maps)});
    return UsdGenExpressionRecord(data, key, expression.path.GetAsToken());
}

UsdGenInputRecord LowerGeometry(UsdGenGeometryDesc const &geometry, Lowering &keys)
{
    return UsdGenGeometryRecord(geometry, keys.Key(geometry.path), geometry.path.GetAsToken(),
                                keys.Columns());
}

UsdGenOpNode LowerNode(UsdGenNodeDesc const &node, Lowering &keys)
{
    UsdGenOpNode out;
    out.key = keys.Key(node.path);
    out.type = node.type;
    out.mode = node.mode;
    out.enabled = node.enabled;
    out.seed = node.seed;
    out.inputs = keys.Keys(node.inputs);
    out.references = keys.Keys(node.references);
    out.inputSets = keys.Keys(node.curves);
    out.surfaces = keys.Keys(node.surfaces);
    out.maps = keys.Keys(node.maps);
    out.mapBindings.reserve(node.mapBindings.size());
    for (UsdGenMapBindingDesc const &binding : node.mapBindings)
        out.mapBindings.push_back(UsdGenMapBinding{keys.Key(binding.map), binding.relationship});
    out.params = node.params;
    for (UsdGenParamValue &p : out.params) p.value = ToEngineValue(p.value);
    out.expressionBindings.reserve(node.expressionBindings.size());
    for (UsdGenExpressionBinding const &binding : node.expressionBindings) {
        UsdGenParamBinding b;
        b.expression = keys.Key(binding.expression);
        b.output = binding.output;
        b.nativeType = binding.nativeType;
        b.destination = binding.destination;
        b.destinationShape = binding.destinationShape;
        b.domain = binding.domain;
        b.literal = ToEngineValue(binding.literal);
        out.expressionBindings.push_back(std::move(b));
    }
    out.ramps = node.ramps;
    out.opId = node.opId;
    return out;
}

inline constexpr char kRgbText[] = "rgb";
UsdGenTextToken<kRgbText> const sRgb{};

// The source's one root-binding surface: device admission permits at most
// one, and a binding absent from the surfaces is rejected, so none is named
// then (never invented).
UsdGenNodeKey SourceDependencySurface(UsdGenGraphSpec const &spec)
{
    if (spec.nodes.empty() || spec.nodes.front().surfaces.size() != 1) return UsdGenNodeKey();
    UsdGenNodeKey const candidate = spec.nodes.front().surfaces.front();
    UsdGenInputRecord const *found =
        spec.FindInput(UsdGenInputKindWithRole(UsdGenInputRole::Surface), candidate);
    return found ? found->key : UsdGenNodeKey();
}

}  // namespace

UsdGenGraphSpec Lower(UsdGenGraphDesc const &desc, NodeKeyMap &map)
{
    map.Columns().Advance();
    Lowering keys(map);
    UsdGenGraphSpec spec;
    spec.description = keys.Key(desc.description);
    spec.terminal = keys.Key(desc.terminal);
    spec.nodes.reserve(desc.nodes.size());
    for (UsdGenNodeDesc const &node : desc.nodes) spec.nodes.push_back(LowerNode(node, keys));
    spec.inputs.reserve(desc.curveSets.size() + desc.surfaces.size() + desc.maps.size() +
                        desc.expressions.size() + desc.geometries.size());
    for (UsdGenCurveSetDesc const &curves : desc.curveSets)
        spec.inputs.push_back(LowerCurves(curves, keys));
    for (UsdGenSurfaceDesc const &surface : desc.surfaces)
        spec.inputs.push_back(LowerSurface(surface, keys));
    for (UsdGenMapDesc const &map : desc.maps) spec.inputs.push_back(LowerMap(map, keys));
    for (UsdGenExpressionDesc const &expression : desc.expressions)
        spec.inputs.push_back(LowerExpression(expression, keys));
    for (UsdGenGeometryDesc const &geometry : desc.geometries)
        spec.inputs.push_back(LowerGeometry(geometry, keys));
    spec.SortInputs();
    UsdGenPresentation presentation;
    presentation.look = desc.look;
    UsdGenPreviewDesc const &preview = desc.preview;
    if (preview.source.IsPropertyPath()) {
        spec.preview.source = keys.Key(preview.source.GetPrimPath());
        spec.preview.attribute = preview.source.GetNameToken();
    } else {
        spec.preview.source = keys.Key(preview.source);
    }
    presentation.previewColorMap = preview.colorMap;
    presentation.previewRange = preview.range;
    spec.preview.evaluation = preview.evaluation;
    spec.preview.mapChannels = preview.colorMap == *sRgb ? 3 : 1;
    presentation.previewShading = preview.shading;
    spec.xformMatrix = desc.xformMatrix;
    presentation.purpose = desc.purpose;
    presentation.visibility = desc.visibility;
    presentation.material = keys.Key(desc.materialPath);
    spec.defaultWidth = desc.defaultWidth;
    spec.tileTarget = desc.tileTarget;
    presentation.basis = desc.curveBasis;
    spec.time = desc.time;
    spec.timeCodesPerSecond = desc.timeCodesPerSecond;
    spec.executionBackend = desc.executionBackend;
    spec.validationErrors = desc.validationErrors;
    presentation.description = spec.description;
    presentation.xformMatrix = ToHost(spec.xformMatrix);
    presentation.dependency = SourceDependencySurface(spec);
    spec.hostDigest = PresentationDigest(presentation);
    keys.Finish(&spec, std::move(presentation));
    return spec;
}

UsdGenGraphSpec Lower(UsdGenGraphDesc const &desc)
{
    NodeKeyMap map;
    return Lower(desc, map);
}

UsdGenOpNode Lower(UsdGenNodeDesc const &node)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerNode(node, keys);
}
UsdGenInputRecord Lower(UsdGenCurveSetDesc const &curves)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerCurves(curves, keys);
}
UsdGenInputRecord Lower(UsdGenSurfaceDesc const &surface)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerSurface(surface, keys);
}
UsdGenInputRecord Lower(UsdGenMapDesc const &map)
{
    NodeKeyMap keyMap;
    Lowering keys(keyMap);
    return LowerMap(map, keys);
}
UsdGenInputRecord Lower(UsdGenGeometryDesc const &geometry)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerGeometry(geometry, keys);
}
UsdGenInputRecord Lower(UsdGenExpressionDesc const &expression)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerExpression(expression, keys);
}

UsdGenCurveSetInputView View(UsdGenCurveSetDesc const &curves)
{
    return UsdGenCurveSetInputView::Of(Lower(curves));
}

UsdGenSurfaceInputView View(UsdGenSurfaceDesc const &surface)
{
    return UsdGenSurfaceInputView::Of(Lower(surface));
}

UsdGenParamBinding Lower(UsdGenExpressionBinding const &binding)
{
    UsdGenParamBinding b;
    b.expression = KeyOf(binding.expression);
    b.output = binding.output;
    b.nativeType = binding.nativeType;
    b.destination = binding.destination;
    b.destinationShape = binding.destinationShape;
    b.domain = binding.domain;
    b.literal = ToEngineValue(binding.literal);
    return b;
}

std::shared_ptr<const UsdGenGraphSpec> LowerShared(UsdGenGraphDesc const &desc, NodeKeyMap *map)
{
    return std::make_shared<const UsdGenGraphSpec>(map ? Lower(desc, *map) : Lower(desc));
}

UsdGenNodeKey KeyOf(SdfPath const &path)
{
    if (path.IsEmpty()) return UsdGenNodeKey{};
    std::string const &text = path.GetString();
    for (uint32_t salt = 0;; ++salt) {
        UsdGenNodeKey const key{NodeKeyMap::Hash(text, salt)};
        if (key.IsValid()) return key;
    }
}

PathTable const *Paths(std::shared_ptr<const void> const &host)
{
    return static_cast<PathTable const *>(host.get());
}
PathTable const *Paths(UsdGenGraphSpec const &spec) { return Paths(spec.host); }

UsdGenPresentation const *Presentation(std::shared_ptr<const void> const &host)
{
    PathTable const *table = Paths(host);
    return table ? &table->Presentation() : nullptr;
}
UsdGenPresentation const *Presentation(UsdGenGraphSpec const &spec)
{
    return Presentation(spec.host);
}
UsdGenPresentation const *Presentation(UsdGenGeneration const &generation)
{
    return Presentation(generation.host);
}

std::optional<UsdGenDevicePresentationMetadata> DevicePresentation(
    UsdGenGeneration const &generation)
{
    if (!generation.device) return std::nullopt;
    UsdGenPresentation const fallback;
    UsdGenPresentation const *p = Presentation(generation.host);
    if (!p) p = &fallback;
    UsdGenDevicePresentationMetadata m;
    m.description = p->description;
    m.xformMatrix = p->xformMatrix;
    m.purpose = p->purpose;
    m.visibility = p->visibility;
    m.material = UsdGenPrimRef{p->material};
    m.origin = UsdGenPrimRef{p->description};
    m.dependency = UsdGenPrimRef{p->dependency};
    return m;
}
PathTable const *Paths(UsdGenGeneration const &generation) { return Paths(generation.host); }
PathTable const *Paths(UsdGenGraphRoutingSnapshot const &routing) { return Paths(routing.host); }

SdfPath PathOf(PathTable const *table, UsdGenNodeKey key)
{
    return table ? table->Path(key) : SdfPath();
}

UsdGenNodeKey KeyFor(PathTable const *table, SdfPath const &path)
{
    if (table) {
        UsdGenNodeKey const key = table->Key(path);
        if (key.IsValid()) return key;
    }
    return KeyOf(path);
}

UsdGenCompileResult Compile(UsdGenCompiler &compiler, UsdGenGraphDesc const &desc,
                            UsdGenGraph *graph, NodeKeyMap *map)
{
    UsdGenGraphSpec const spec = map ? Lower(desc, *map) : Lower(desc);
    return compiler.Compile(spec, graph);
}

UsdGenCompileResult Recompile(UsdGenCompiler &compiler, UsdGenGraphDesc const &desc,
                              UsdGenGraph *graph, NodeKeyMap *map)
{
    UsdGenGraphSpec const spec = map ? Lower(desc, *map) : Lower(desc);
    return compiler.Recompile(spec, graph);
}

UsdGenCompileResult CompileInjectedDevice(
    UsdGenCompiler &compiler, UsdGenGraphDesc const &desc, UsdGenGraph *graph,
    UsdGenCompiler::DevicePlanCompiler const &device,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan, NodeKeyMap *map)
{
    UsdGenGraphSpec const spec = map ? Lower(desc, *map) : Lower(desc);
    return compiler.CompileInjectedDevice(spec, graph, device, plan);
}

void SetGraphDesc(UsdGenSession &session, UsdGenGraphDesc const &desc, NodeKeyMap *map)
{
    session.SetGraphSpec(map ? Lower(desc, *map) : Lower(desc));
}

bool PostGraphDesc(UsdGenSession &session, UsdGenGraphDesc const &desc, NodeKeyMap *map)
{
    return session.PostGraphSpec(map ? Lower(desc, *map) : Lower(desc));
}

UsdGenNodeId NodeIdForPath(UsdGenGraph const &graph, SdfPath const &path)
{
    return graph.NodeIdForKey(KeyFor(Paths(graph.Spec()), path));
}

UsdGenNodeId NodeIdForPath(UsdGenSessionGraphSnapshot const &graph, SdfPath const &path)
{
    return graph.NodeIdForKey(KeyFor(Paths(graph.Spec()), path));
}

SdfPath NodePath(UsdGenGraph const &graph, UsdGenNodeId id)
{
    if (id >= UsdGenNodeId(graph.NodeCount())) return SdfPath();
    UsdGenCompiledNode const &node = graph.Node(id);
    return node.desc ? PathOf(Paths(graph.Spec()), node.desc->key) : SdfPath();
}

void DirtyGeometry(UsdGenGraph &graph, SdfPath const &prim)
{
    std::vector<UsdGenNodeKey> keys;
    if (PathTable const *table = Paths(graph.Spec())) table->KeysUnder(prim, &keys);
    for (UsdGenNodeKey key : keys) graph.DirtyGeometry(key);
}

void DirtyInput(UsdGenGraph &graph, SdfPath const &input, uint64_t generation)
{
    graph.DirtyInput(KeyFor(Paths(graph.Spec()), input), generation);
}

void DirtyMap(UsdGenGraph &graph, SdfPath const &map, uint64_t generation)
{
    graph.DirtyMap(KeyFor(Paths(graph.Spec()), map), generation);
}

}  // namespace usdGenUsd
