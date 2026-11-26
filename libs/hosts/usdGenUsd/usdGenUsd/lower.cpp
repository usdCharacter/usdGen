// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/lower.h"

#include <algorithm>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {

using namespace usdGen;

PathTable::PathTable(std::vector<std::pair<UsdGenNodeKey, SdfPath>> entries)
    : _byKey(std::move(entries))
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
    void Finish(UsdGenGraphSpec *spec)
    {
        auto table = std::make_shared<PathTable>(std::move(_entries));
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

UsdGenCurveSetInput LowerCurves(UsdGenCurveSetDesc const &curves, Lowering &keys)
{
    UsdGenCurveSetInput in;
    static_cast<UsdGenCurveSetData &>(in) = curves;
    in.key = keys.Key(curves.path);
    return in;
}

UsdGenSurfaceInput LowerSurface(UsdGenSurfaceDesc const &surface, Lowering &keys)
{
    UsdGenSurfaceInput in;
    static_cast<UsdGenSurfaceData &>(in) = surface;
    in.key = keys.Key(surface.path);
    // The parent prim: a GeomSubset's Mesh (consumers that find a
    // topology-less target look its geometry up here).
    if (!surface.path.IsEmpty() && !surface.path.IsAbsoluteRootPath())
        in.owner = keys.Key(surface.path.GetParentPath());
    return in;
}

UsdGenMapInput LowerMap(UsdGenMapDesc const &map, Lowering &keys)
{
    UsdGenMapInput in;
    static_cast<UsdGenMapData &>(in) = map;
    in.key = keys.Key(map.path);
    in.paintSurface = keys.Key(map.paintSurface);
    return in;
}

UsdGenExpressionInput LowerExpression(UsdGenExpressionDesc const &expression, Lowering &keys)
{
    UsdGenExpressionInput in;
    in.key = keys.Key(expression.path);
    in.source = expression.source;
    in.outputs = expression.outputs;
    in.inputs.reserve(expression.inputs.size());
    for (UsdGenExpressionInputDesc const &input : expression.inputs)
        in.inputs.push_back(UsdGenExpressionSlot{input.name, keys.Keys(input.targets),
                                                 keys.Keys(input.geometries),
                                                 keys.Keys(input.maps)});
    return in;
}

UsdGenGeometryInput LowerGeometry(UsdGenGeometryDesc const &geometry, Lowering &keys)
{
    UsdGenGeometryInput in;
    static_cast<UsdGenGeometryData &>(in) = geometry;
    in.key = keys.Key(geometry.path);
    return in;
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
    out.curves = keys.Keys(node.curves);
    out.surfaces = keys.Keys(node.surfaces);
    out.maps = keys.Keys(node.maps);
    out.mapBindings.reserve(node.mapBindings.size());
    for (UsdGenMapBindingDesc const &binding : node.mapBindings)
        out.mapBindings.push_back(UsdGenMapBinding{keys.Key(binding.map), binding.relationship});
    out.params = node.params;
    out.expressionBindings.reserve(node.expressionBindings.size());
    for (UsdGenExpressionBinding const &binding : node.expressionBindings) {
        UsdGenParamBinding b;
        b.expression = keys.Key(binding.expression);
        b.output = binding.output;
        b.nativeType = binding.nativeType;
        b.destination = binding.destination;
        b.destinationShape = binding.destinationShape;
        b.domain = binding.domain;
        b.literal = binding.literal;
        out.expressionBindings.push_back(std::move(b));
    }
    out.ramps = node.ramps;
    out.opId = node.opId;
    return out;
}

}  // namespace

UsdGenGraphSpec Lower(UsdGenGraphDesc const &desc, NodeKeyMap &map)
{
    Lowering keys(map);
    UsdGenGraphSpec spec;
    spec.description = keys.Key(desc.description);
    spec.terminal = keys.Key(desc.terminal);
    spec.nodes.reserve(desc.nodes.size());
    for (UsdGenNodeDesc const &node : desc.nodes) spec.nodes.push_back(LowerNode(node, keys));
    spec.curveSets.reserve(desc.curveSets.size());
    for (UsdGenCurveSetDesc const &curves : desc.curveSets)
        spec.curveSets.push_back(LowerCurves(curves, keys));
    spec.surfaces.reserve(desc.surfaces.size());
    for (UsdGenSurfaceDesc const &surface : desc.surfaces)
        spec.surfaces.push_back(LowerSurface(surface, keys));
    spec.maps.reserve(desc.maps.size());
    for (UsdGenMapDesc const &map : desc.maps) spec.maps.push_back(LowerMap(map, keys));
    spec.expressions.reserve(desc.expressions.size());
    for (UsdGenExpressionDesc const &expression : desc.expressions)
        spec.expressions.push_back(LowerExpression(expression, keys));
    spec.geometries.reserve(desc.geometries.size());
    for (UsdGenGeometryDesc const &geometry : desc.geometries)
        spec.geometries.push_back(LowerGeometry(geometry, keys));
    spec.look = desc.look;
    UsdGenPreviewDesc const &preview = desc.preview;
    if (preview.source.IsPropertyPath()) {
        spec.preview.source = keys.Key(preview.source.GetPrimPath());
        spec.preview.attribute = preview.source.GetNameToken();
    } else {
        spec.preview.source = keys.Key(preview.source);
    }
    spec.preview.colorMap = preview.colorMap;
    spec.preview.range = preview.range;
    spec.preview.evaluation = preview.evaluation;
    spec.preview.shading = preview.shading;
    spec.xformMatrix = desc.xformMatrix;
    spec.purpose = desc.purpose;
    spec.visibility = desc.visibility;
    spec.material = keys.Key(desc.materialPath);
    spec.defaultWidth = desc.defaultWidth;
    spec.tileTarget = desc.tileTarget;
    spec.curveBasis = desc.curveBasis;
    spec.time = desc.time;
    spec.timeCodesPerSecond = desc.timeCodesPerSecond;
    spec.executionBackend = desc.executionBackend;
    spec.validationErrors = desc.validationErrors;
    keys.Finish(&spec);
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
UsdGenCurveSetInput Lower(UsdGenCurveSetDesc const &curves)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerCurves(curves, keys);
}
UsdGenSurfaceInput Lower(UsdGenSurfaceDesc const &surface)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerSurface(surface, keys);
}
UsdGenMapInput Lower(UsdGenMapDesc const &map)
{
    NodeKeyMap keyMap;
    Lowering keys(keyMap);
    return LowerMap(map, keys);
}
UsdGenGeometryInput Lower(UsdGenGeometryDesc const &geometry)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerGeometry(geometry, keys);
}
UsdGenExpressionInput Lower(UsdGenExpressionDesc const &expression)
{
    NodeKeyMap map;
    Lowering keys(map);
    return LowerExpression(expression, keys);
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
    b.literal = binding.literal;
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
