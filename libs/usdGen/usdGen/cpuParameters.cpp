#include "usdGen/cpuParameters.h"

#include "usdGen/expressions/frontend.h"
#include "usdGen/expressions/irExec.h"
#include "usdGen/maps/ptexMap.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec4i.h"

#include <algorithm>
#include <cmath>
#include <cstring>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace {

using Type = expr::ScalarType;
using Domain = expr::Domain;

size_t ScalarBytes(Type type)
{
    switch (type) {
    case Type::Bool: return 1;
    case Type::Float16: return 2;
    case Type::Int32: case Type::UInt32: case Type::Float32: return 4;
    case Type::Int64: case Type::UInt64: case Type::Float64: return 8;
    default: return 0;
    }
}

uint64_t Mix(uint64_t h, uint64_t value)
{
    h ^= value; h *= 0x100000001b3ULL;
    return h;
}
uint64_t MixText(uint64_t h, char const *text)
{
    for (auto const *p = reinterpret_cast<unsigned char const *>(text); p && *p; ++p)
        h = Mix(h, *p);
    return h;
}
uint64_t MixDouble(uint64_t h, double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return Mix(h, bits);
}

// Closest-point barycentric of p in triangle (a, b, c):
// p = a + s(b-a) + t(c-a). False when the triangle is degenerate.
bool TriCoords(GfVec3d const &a, GfVec3d const &b, GfVec3d const &c,
               GfVec3d const &p, double *s, double *t)
{
    GfVec3d const e0 = b - a, e1 = c - a, ap = p - a;
    double const d00 = GfDot(e0, e0);
    double const d01 = GfDot(e0, e1);
    double const d11 = GfDot(e1, e1);
    double const denom = d00 * d11 - d01 * d01;
    if (!(denom > 1e-18 * d00 * d11)) return false;
    double const d20 = GfDot(ap, e0);
    double const d21 = GfDot(ap, e1);
    *s = (d11 * d20 - d01 * d21) / denom;
    *t = (d00 * d21 - d01 * d20) / denom;
    return true;
}

// Face-local (u, v) of root p on the quad whose corners start at
// indices[begin]: the brush tool's pick convention (brushPick.pickFace:
// v0=(0,0), v1=(1,0), v2=(1,1), v3=(0,1), diagonal v0-v2, triangle A
// first). False when the quad is degenerate, a corner index is out of
// range, or p sits outside both halves past tolerance.
bool PaintQuadUV(float const *points, size_t numPoints, int const *indices,
                 int begin, GfVec3f const &p, double *u, double *v)
{
    int iv[4];
    for (int k = 0; k < 4; ++k) {
        int const id = indices[begin + k];
        if (id < 0 || size_t(id) >= numPoints) return false;
        iv[k] = id;
    }
    auto at = [&](int k) -> GfVec3d {
        return GfVec3d(double(points[size_t(iv[k]) * 3]),
                       double(points[size_t(iv[k]) * 3 + 1]),
                       double(points[size_t(iv[k]) * 3 + 2]));
    };
    GfVec3d const pp{double(p[0]), double(p[1]), double(p[2])};
    double s = 0.0, t = 0.0;
    // Triangle A (v0, v1, v2): u = s + t, v = t.
    if (TriCoords(at(0), at(1), at(2), pp, &s, &t) && s >= -1e-6 &&
        t >= -1e-6 && s + t <= 1.0 + 1e-6) {
        *u = std::min(1.0, std::max(0.0, s + t));
        *v = std::min(1.0, std::max(0.0, t));
        return true;
    }
    // Triangle B (v0, v2, v3): u = s, v = s + t.
    if (TriCoords(at(0), at(2), at(3), pp, &s, &t) && s >= -1e-6 &&
        t >= -1e-6 && s + t <= 1.0 + 1e-6) {
        *u = std::min(1.0, std::max(0.0, s));
        *v = std::min(1.0, std::max(0.0, s + t));
        return true;
    }
    return false;
}

template <class T>
bool ScalarLiteral(VtValue const &value, std::vector<double> *output)
{
    if (!value.IsHolding<T>()) return false;
    output->assign(1, static_cast<double>(value.UncheckedGet<T>()));
    return std::isfinite((*output)[0]);
}
template <class T>
bool VectorLiteral(VtValue const &value, unsigned count, std::vector<double> *output)
{
    if (!value.IsHolding<T>() || count != T::dimension) return false;
    output->resize(count);
    auto const &vector = value.UncheckedGet<T>();
    for (unsigned i = 0; i < count; ++i) {
        (*output)[i] = static_cast<double>(vector[i]);
        if (!std::isfinite((*output)[i])) return false;
    }
    return true;
}

/// The literal the expression receives as $value. Identical admission to
/// cudaParameters.cpp's Literal(): the exact native type, finite, and int64
/// limited to the range SeExpr's doubles represent exactly.
bool Literal(VtValue const &value, expr::ValueShape const &shape,
             std::vector<double> *output)
{
    const unsigned n = shape.components;
    if (n == 1) {
        switch (shape.scalar) {
        case Type::Bool: return ScalarLiteral<bool>(value, output);
        case Type::Int32: return ScalarLiteral<int32_t>(value, output);
        case Type::UInt32: return ScalarLiteral<uint32_t>(value, output);
        case Type::Int64:
            if (!value.IsHolding<int64_t>() ||
                value.UncheckedGet<int64_t>() < -(int64_t(1) << 53) ||
                value.UncheckedGet<int64_t>() > (int64_t(1) << 53)) return false;
            return ScalarLiteral<int64_t>(value, output);
        case Type::UInt64:
            if (!value.IsHolding<uint64_t>() ||
                value.UncheckedGet<uint64_t>() > (uint64_t(1) << 53)) return false;
            return ScalarLiteral<uint64_t>(value, output);
        case Type::Float16: return ScalarLiteral<GfHalf>(value, output);
        case Type::Float32: return ScalarLiteral<float>(value, output);
        case Type::Float64: return ScalarLiteral<double>(value, output);
        default: return false;
        }
    }
    switch (shape.scalar) {
    case Type::Float16:
        return VectorLiteral<GfVec2h>(value, n, output) ||
            VectorLiteral<GfVec3h>(value, n, output) || VectorLiteral<GfVec4h>(value, n, output);
    case Type::Float32:
        return VectorLiteral<GfVec2f>(value, n, output) ||
            VectorLiteral<GfVec3f>(value, n, output) || VectorLiteral<GfVec4f>(value, n, output);
    case Type::Float64:
        return VectorLiteral<GfVec2d>(value, n, output) ||
            VectorLiteral<GfVec3d>(value, n, output) || VectorLiteral<GfVec4d>(value, n, output);
    case Type::Int32:
        return VectorLiteral<GfVec2i>(value, n, output) ||
            VectorLiteral<GfVec3i>(value, n, output) || VectorLiteral<GfVec4i>(value, n, output);
    default: return false;
    }
}

bool SupportedShape(expr::ValueShape const &shape)
{
    return ScalarBytes(shape.scalar) && !shape.isArray && shape.elementCount == 1 &&
        shape.rows == 1 && shape.columns == 1 && shape.components >= 1 &&
        shape.components <= 4;
}
bool SameShape(expr::ValueShape const &a, expr::ValueShape const &b)
{
    return a.scalar == b.scalar && a.elementCount == b.elementCount &&
        a.components == b.components && a.rows == b.rows && a.columns == b.columns &&
        a.isArray == b.isArray;
}

size_t DomainIndex(Domain domain)
{
    return domain == Domain::Groom ? 0 : domain == Domain::Primitive ? 1 : 2;
}

/// Decodes one element of a typed destination buffer back to a double. The
/// buffer was written by expr::Store, so the value is already known to be
/// exactly representable in the destination type.
double Decode(unsigned char const *bytes, Type type, size_t index)
{
    switch (type) {
    case Type::Bool: return bytes[index] ? 1.0 : 0.0;
    case Type::Int32: { int32_t v; std::memcpy(&v, bytes + index * 4, 4); return double(v); }
    case Type::UInt32: { uint32_t v; std::memcpy(&v, bytes + index * 4, 4); return double(v); }
    case Type::Int64: { int64_t v; std::memcpy(&v, bytes + index * 8, 8); return double(v); }
    case Type::UInt64: { uint64_t v; std::memcpy(&v, bytes + index * 8, 8); return double(v); }
    case Type::Float16: {
        uint16_t bits; std::memcpy(&bits, bytes + index * 2, 2);
        GfHalf half; half.setBits(bits); return double(float(half));
    }
    case Type::Float32: { float v; std::memcpy(&v, bytes + index * 4, 4); return double(v); }
    case Type::Float64: { double v; std::memcpy(&v, bytes + index * 8, 8); return v; }
    default: return 0.0;
    }
}

UsdGenExpressionInputDesc const *FindInput(UsdGenExpressionDesc const &expression,
                                            std::string const &name)
{
    for (auto const &input : expression.inputs)
        if (input.name.GetString() == name) return &input;
    return nullptr;
}

UsdGenGeometryDesc const *FindGeometry(UsdGenGraphDesc const &desc, SdfPath const &path)
{
    for (auto const &geometry : desc.geometries)
        if (geometry.path == path) return &geometry;
    return nullptr;
}

UsdGenMapDesc const *FindMap(UsdGenGraphDesc const &desc, SdfPath const &path)
{
    for (auto const &map : desc.maps)
        if (map.path == path) return &map;
    return nullptr;
}

UsdGenParamValue const *FindMapParam(UsdGenMapDesc const &map, char const *name)
{
    for (auto const &param : map.params)
        if (param.name == name) return &param;
    return nullptr;
}

bool MapToken(UsdGenMapDesc const &map, char const *name, std::string *out)
{
    UsdGenParamValue const *param = FindMapParam(map, name);
    if (!param) return false;
    if (param->value.IsHolding<TfToken>()) *out = param->value.UncheckedGet<TfToken>().GetString();
    else if (param->value.IsHolding<std::string>()) *out = param->value.UncheckedGet<std::string>();
    else return false;
    return true;
}

double MapNumber(UsdGenMapDesc const &map, char const *name, double fallback)
{
    UsdGenParamValue const *param = FindMapParam(map, name);
    if (!param) return fallback;
    VtValue const &v = param->value;
    if (v.IsHolding<float>()) return v.UncheckedGet<float>();
    if (v.IsHolding<double>()) return v.UncheckedGet<double>();
    if (v.IsHolding<int>()) return v.UncheckedGet<int>();
    return fallback;
}

/// The space the groom is generated and published in: the description's.
/// Sampled geometry is brought into it through its own world matrix.
GfMatrix4d ToGroomSpace(UsdGenGraphDesc const &desc, UsdGenGeometryDesc const &geometry)
{
    return geometry.worldMatrix * desc.xformMatrix.GetInverse();
}

void TransformPoints(VtVec3fArray const &in, GfMatrix4d const &m, std::vector<float> *out)
{
    out->resize(in.size() * 3);
    for (size_t i = 0; i < in.size(); ++i) {
        GfVec3d const p = m.Transform(GfVec3d(in[i]));
        (*out)[i * 3] = float(p[0]);
        (*out)[i * 3 + 1] = float(p[1]);
        (*out)[i * 3 + 2] = float(p[2]);
    }
}

expr::SamplerGeometrySource ToSamplerSource(UsdGenGraphDesc const &desc,
                                           UsdGenGeometryDesc const &geometry)
{
    expr::SamplerGeometrySource source;
    switch (geometry.kind) {
    case UsdGenGeometryKind::Mesh: source.kind = expr::SamplerGeometrySource::Kind::Mesh; break;
    case UsdGenGeometryKind::Curves: source.kind = expr::SamplerGeometrySource::Kind::Curves; break;
    case UsdGenGeometryKind::Points: source.kind = expr::SamplerGeometrySource::Kind::Points; break;
    }
    GfMatrix4d const m = ToGroomSpace(desc, geometry);
    TransformPoints(geometry.points, m, &source.points);
    if (!geometry.rest.empty()) TransformPoints(geometry.rest, m, &source.rest);
    if (!geometry.normals.empty()) {
        GfMatrix4d const normalMatrix = m.GetInverse().GetTranspose();
        source.normals.resize(geometry.normals.size() * 3);
        for (size_t i = 0; i < geometry.normals.size(); ++i) {
            GfVec3d n = normalMatrix.TransformDir(GfVec3d(geometry.normals[i]));
            double const length = n.GetLength();
            if (length > 0.0) n /= length;
            source.normals[i * 3] = float(n[0]);
            source.normals[i * 3 + 1] = float(n[1]);
            source.normals[i * 3 + 2] = float(n[2]);
        }
    }
    source.counts.assign(geometry.counts.cbegin(), geometry.counts.cend());
    source.indices.assign(geometry.indices.cbegin(), geometry.indices.cend());
    source.ids.assign(geometry.ids.cbegin(), geometry.ids.cend());
    source.subset = geometry.isSubset;
    source.faces.assign(geometry.subsetFaces.cbegin(), geometry.subsetFaces.cend());
    return source;
}

/// The surface a node's strands root on. A GeomSubset binding carries no
/// topology of its own; its parent mesh does (R15).
UsdGenSurfaceDesc const *RootSurface(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node)
{
    if (node.surfaces.empty()) return nullptr;
    SdfPath path = node.surfaces.front();
    for (int pass = 0; pass < 2; ++pass) {
        for (auto const &surface : desc.surfaces) {
            if (surface.path != path) continue;
            if (!surface.faceVertexCounts.empty()) return &surface;
            path = path.GetParentPath();
            break;
        }
    }
    return nullptr;
}

bool ReadsVariable(expr::IRProgram const &program, expr::Variable a, expr::Variable b)
{
    for (auto const &op : program.instructions)
        if (op.op == expr::IROp::LoadVariable && (op.variable == a || op.variable == b))
            return true;
    return false;
}

// Only the connected single-source ancestry qualifies. Another description
// scattering on the same surface must not reinterpret this chain's UVs.
bool HasLimitScatter(UsdGenGraphDesc const& desc, UsdGenNodeDesc const& node) {
    auto const* current=&node;
    for(size_t step=0;step<=desc.nodes.size();++step) {
        if(current->type==TfToken("UsdGenScatter")) {
            for(auto const& p:current->params)
                if(p.name==TfToken("subdivisionLevel") && p.value.IsHolding<int>())
                    return p.value.UncheckedGet<int>()>0;
            return false;
        }
        if(current->inputs.size()!=1) return false;
        auto it=std::find_if(desc.nodes.begin(),desc.nodes.end(),[&](auto const& n){return n.path==current->inputs.front();});
        if(it==desc.nodes.end()) return false;
        current=&*it;
    }
    return false;
}

} // namespace

std::vector<std::string> UsdGenCpuParameters::TakeWarnings()
{
    std::vector<std::string> result;
    result.swap(warnings_);
    return result;
}

bool UsdGenCpuParameters::ResolveSamplers(UsdGenGraphDesc const &desc,
                                          UsdGenExpressionDesc const &expression,
                                          Item *item, std::vector<std::string> *diagnostics)
{
    auto fail = [&](std::string const &message) {
        if (diagnostics)
            diagnostics->push_back(item->binding.destination.GetString() + ": expression " +
                                   expression.path.GetString() + ": " + message);
        return false;
    };
    item->samplers.clear();
    item->table.slots.clear();
    for (expr::IRSampler const &spec : item->ir.samplers) {
        auto state = std::make_unique<SamplerState>();
        state->spec = spec;
        state->expression = expression.path;
        UsdGenExpressionInputDesc const *input = FindInput(expression, spec.input);
        std::string const call = spec.kind == expr::SamplerKind::Geometry
            ? "geoSampler(\"" + spec.input + "\")" : "ptex(\"" + spec.input + "\")";
        if (!input)
            return fail(call + " names no input:" + spec.input + " relationship on the expression prim");
        expr::CpuExpressionSamplers::Slot slot;
        slot.kind = spec.kind;
        slot.components = spec.components;
        if (spec.kind == expr::SamplerKind::Geometry) {
            if (input->geometries.empty())
                return fail(call + ": input:" + spec.input + " targets no mesh, curves or points" +
                            (input->maps.empty() ? "" : " (it targets a map; read maps with ptex())"));
            for (SdfPath const &path : input->geometries) {
                if (!FindGeometry(desc, path))
                    return fail(call + ": geometry " + path.GetString() + " was not resolved");
                state->geometries.push_back(path);
            }
            state->readsTime = ReadsVariable(*spec.element, expr::Variable::Frame,
                                             expr::Variable::Time);
            state->geometry = std::make_unique<expr::GeometrySampler>();
        } else {
            if (input->maps.size() != 1)
                return fail(call + ": input:" + spec.input + " must target exactly one UsdGenPtexMap or UsdGenPaintMap");
            UsdGenMapDesc const *map = FindMap(desc, input->maps.front());
            if (!map)
                return fail(call + ": map " + input->maps.front().GetString() + " was not resolved");
            if (map->type != TfToken("UsdGenPtexMap") && map->type != TfToken("UsdGenPaintMap"))
                return fail(call + ": " + map->path.GetString() + " is a " + map->type.GetString() +
                            ", not a UsdGenPtexMap or UsdGenPaintMap");
            state->map = map->path;
            state->paint = map->type == TfToken("UsdGenPaintMap");
            UsdGenPtexMapOptions options;
            MapToken(*map, "map:filter", &options.filter);
            MapToken(*map, "map:borderMode", &options.borderMode);
            options.blur = float(MapNumber(*map, "map:blur", 0.0));
            options.firstChannel = int(MapNumber(*map, "map:firstChannel", 0.0));
            options.channelCount = int(MapNumber(*map, "map:channelCount", 1.0));
            std::string channel = "r";
            MapToken(*map, "map:channel", &channel);
            state->scale = MapNumber(*map, "map:scale", 1.0);
            state->offset = MapNumber(*map, "map:offset", 0.0);
            state->fallback = MapNumber(*map, "map:default", 0.0);
            if (UsdGenParamValue const *clamp = FindMapParam(*map, "map:clamp")) {
                if (clamp->value.IsHolding<GfVec2f>()) {
                    GfVec2f const range = clamp->value.UncheckedGet<GfVec2f>();
                    state->clampLo = range[0];
                    state->clampHi = range[1];
                }
            }
            if (channel == "r") state->channel = 0;
            else if (channel == "g") state->channel = 1;
            else if (channel == "b") state->channel = 2;
            else if (channel == "a") state->channel = 3;
            else if (channel == "luminance") state->channel = -1;
            else
                return fail(call + ": usdGen:map:channel \"" + channel +
                            "\" cannot drive a scalar; use r, g, b, a or luminance");
            if (state->paint && channel != "r" && channel != "luminance")
                return fail(call + ": " + map->path.GetString() +
                            " is a scalar paint snapshot; usdGen:map:channel \"" + channel +
                            "\" cannot drive it, use r or luminance");
            std::string error;
            if (!state->paint)
                state->texture = UsdGenPtexTexture::Open(map->resolvedAssetPath, options, &error);
            else if (map->paintValues.empty())
                warnings_.push_back(call + ": " + map->path.GetString() +
                    " carries no paint payload (see the capture errors); using usdGen:map:default");
            if (state->texture) {
                const int window = state->texture->SampleChannels();
                if ((state->channel < 0 && window < 3) || state->channel >= window)
                    return fail(call + ": usdGen:map:channel \"" + channel + "\" is outside the " +
                                std::to_string(window) + "-channel window of " +
                                map->resolvedAssetPath);
            } else if (!state->paint) {
                // A map that cannot be read is not a broken groom: every strand
                // reads usdGen:map:default, and the reason is reported (07 §5.1).
                warnings_.push_back(call + ": " + (map->resolvedAssetPath.empty()
                    ? std::string("the map has no usdGen:map:file") : error) +
                    "; using usdGen:map:default");
            }
        }
        item->samplers.push_back(std::move(state));
        item->table.slots.push_back(slot);
    }
    return true;
}

bool UsdGenCpuParameters::PrepareSamplers(UsdGenGraphDesc const &desc,
                                          UsdGenNodeDesc const &node,
                                          UsdGenCurveBuffer const &geometry,
                                          expr::Context const &controls, Item *item,
                                          std::vector<std::string> *diagnostics)
{
    auto fail = [&](SamplerState const &state, std::string const &message) {
        if (diagnostics)
            diagnostics->push_back(item->binding.destination.GetString() + ": expression " +
                                   state.expression.GetString() + ": " + message);
        return false;
    };
    for (size_t i = 0; i < item->samplers.size(); ++i) {
        SamplerState &state = *item->samplers[i];
        expr::CpuExpressionSamplers::Slot &slot = item->table.slots[i];
        if (state.spec.kind == expr::SamplerKind::Geometry) {
            bool const stale = !state.built ||
                (state.readsTime && (state.builtFrame != controls.frame ||
                                     state.builtTime != controls.time));
            if (stale) {
                std::vector<expr::SamplerGeometrySource> sources;
                sources.reserve(state.geometries.size());
                for (SdfPath const &path : state.geometries) {
                    UsdGenGeometryDesc const *found = FindGeometry(desc, path);
                    if (!found) return fail(state, "geometry " + path.GetString() + " is gone");
                    sources.push_back(ToSamplerSource(desc, *found));
                }
                std::string error;
                if (!state.geometry->Build(sources, state.spec, controls, &error))
                    return fail(state, error);
                state.built = true;
                state.builtFrame = controls.frame;
                state.builtTime = controls.time;
            }
            slot.geometry = state.geometry.get();
            continue;
        }

        // Ptex: one value per strand, read at the strand root.
        size_t const curves = geometry.totalCurves;
        state.values.assign(curves, state.fallback);
        slot.values = state.values.data();
        slot.count = state.values.size();
        if (!curves) continue;
        // A ptex file that could not be read leaves the fallback values
        // Resolve already warned about; paint (which never opens a
        // texture) always reaches its branch below.
        if (!state.paint && !state.texture) continue;
        std::string const call = "ptex(\"" + state.spec.input + "\")";
        UsdGenSurfaceDesc const *surface = RootSurface(desc, node);
        if (!surface)
            return fail(state, call + " needs strands rooted on a usdGen:surface mesh");
        if (geometry.rootPrim.size() != curves)
            return fail(state, call + " needs root face ids, and this input carries none");
        VtVec3fArray const &corners = surface->restPoints.empty() ? surface->points
                                                                  : surface->restPoints;
        size_t const faceCount = surface->faceVertexCounts.size();
        bool const limitUV=HasLimitScatter(desc,node) && geometry.rootUV.size()==curves;
        std::vector<int> faceOffsets(faceCount + 1, 0);
        for (size_t f = 0; f < faceCount; ++f)
            faceOffsets[f + 1] = faceOffsets[f] + surface->faceVertexCounts[f];
        if (size_t(faceOffsets.back()) != surface->faceVertexIndices.size())
            return fail(state, call + ": surface " + surface->path.GetString() +
                        " has inconsistent topology");
        if (state.paint) {
            // Paint: one value per strand, bilinearly sampled from the
            // faceVarying snapshot at the strand root's face-local (u, v).
            // v1 requires the paint surface to be the root surface's mesh:
            // root face ids index that mesh's faces (a GeomSubset root
            // surface keeps its parent's ids, R15). Non-quad faces, degenerate
            // quads and unreadable corners fall back to the face mean, so a
            // paint read never fails a cook on sampling alone.
            UsdGenMapDesc const *map = FindMap(desc, state.map);
            if (!map || map->paintValues.empty()) continue;  // Resolve warned; strands read the fallback
            SdfPath const rootMesh = UsdGenSurfaceMeshPath(*surface);
            if (map->paintSurface != rootMesh)
                return fail(state, call + ": " + map->path.GetString() + " paints " +
                            map->paintSurface.GetString() + ", not the root surface " +
                            rootMesh.GetString());
            if (map->paintValues.size() != size_t(faceOffsets.back()))
                return fail(state, call + ": " + map->path.GetString() + " carries " +
                            std::to_string(map->paintValues.size()) + " paint values for " +
                            std::to_string(faceOffsets.back()) + " face vertices of " +
                            surface->path.GetString());
            float const *positions = corners.empty()
                ? nullptr
                : reinterpret_cast<float const *>(corners.cdata());
            int const *cornerIds = surface->faceVertexIndices.cdata();
            size_t misses = 0;
            for (size_t c = 0; c < curves; ++c) {
                int const face = geometry.rootPrim[c];
                if (face < 0 || size_t(face) >= faceCount) { ++misses; continue; }
                int const begin = faceOffsets[size_t(face)];
                int const n = surface->faceVertexCounts[size_t(face)];
                if (n <= 0) { ++misses; continue; }
                double sum = 0.0;
                for (int k = 0; k < n; ++k) sum += double(map->paintValues[size_t(begin + k)]);
                double sample = sum / double(n);
                if (n == 4 && positions && cornerIds) {
                    size_t const root = offsets_[c];
                    GfVec3f const p =
                        geometry.rest.size() == geometry.totalCvs && root < geometry.rest.size()
                        ? geometry.rest[root]
                        : GfVec3f(geometry.px[root], geometry.py[root], geometry.pz[root]);
                    double u = 0.0, v = 0.0;
                    bool located=false;
                    if(limitUV) {u=geometry.rootUV[c][0];v=geometry.rootUV[c][1];located=true;}
                    else located=PaintQuadUV(positions, corners.size(), cornerIds, begin, p, &u, &v);
                    if (located) {
                        double const c0 = double(map->paintValues[size_t(begin)]);
                        double const c1 = double(map->paintValues[size_t(begin + 1)]);
                        double const c2 = double(map->paintValues[size_t(begin + 2)]);
                        double const c3 = double(map->paintValues[size_t(begin + 3)]);
                        sample = (c0 * (1.0 - u) + c1 * u) * (1.0 - v) +
                                 (c3 * (1.0 - u) + c2 * u) * v;
                    }
                }
                double value = sample * state.scale + state.offset;
                if (state.clampLo < state.clampHi)
                    value = std::min(state.clampHi, std::max(state.clampLo, value));
                state.values[c] = value;
            }
            // Reported when the count changes, not on every cook.
            if (misses != state.reportedMisses && misses)
                warnings_.push_back(call + ": " + std::to_string(misses) + " of " +
                                    std::to_string(curves) + " strand roots name no face of " +
                                    surface->path.GetString() + "; they read usdGen:map:default");
            state.reportedMisses = misses;
            continue;
        }

        std::vector<int> const firstIds =
            UsdGenPtexFirstFaceIds(surface->faceVertexCounts.cdata(), faceCount);
        float const *points = corners.empty() ? nullptr
            : reinterpret_cast<float const *>(corners.cdata());
        auto sampler = state.texture->MakeSampler();
        bool const triangles = state.texture->IsTriangleMesh();
        float texel[16]{};
        size_t misses = 0;
        for (size_t c = 0; c < curves; ++c) {
            size_t const root = offsets_[c];
            GfVec3f const p = geometry.rest.size() == geometry.totalCvs && root < geometry.rest.size()
                ? geometry.rest[root]
                : GfVec3f(geometry.px[root], geometry.py[root], geometry.pz[root]);
            int ptexFace = -1;
            float u = 0.0f, v = 0.0f;
            bool located=false;
            int const face=geometry.rootPrim[c];
            if(limitUV && !triangles && face>=0 && size_t(face)<faceCount) {
                ptexFace=firstIds[face];u=geometry.rootUV[c][0];v=geometry.rootUV[c][1];located=true;
            } else if(points) located=UsdGenPtexFaceCoordinate(points, corners.size(), surface->faceVertexCounts.cdata(),
                                          surface->faceVertexIndices.cdata(), faceOffsets.data(),
                                          faceCount, firstIds.data(), triangles,
                                          geometry.rootPrim[c], p[0], p[1], p[2],
                                          &ptexFace, &u, &v);
            if (!located || !sampler->Sample(ptexFace, u, v, texel)) {
                ++misses;
                continue;
            }
            double value = state.channel < 0
                ? 0.2126 * texel[0] + 0.7152 * texel[1] + 0.0722 * texel[2]
                : double(texel[state.channel]);
            value = value * state.scale + state.offset;
            if (state.clampLo < state.clampHi)
                value = std::min(state.clampHi, std::max(state.clampLo, value));
            state.values[c] = value;
        }
        // Reported when the count changes, not on every cook.
        if (misses != state.reportedMisses && misses)
            warnings_.push_back(call + ": " + std::to_string(misses) + " of " +
                                std::to_string(curves) + " strand roots are outside " +
                                state.texture->Path() + "; they read usdGen:map:default");
        state.reportedMisses = misses;
    }
    return true;
}

bool UsdGenCpuParameters::Clear() noexcept
{
    const bool had = !values_.empty();
    values_.clear();
    digest_ = 0;
    return had;
}

UsdGenExpressionValue const *UsdGenCpuParameters::Find(TfToken const &destination) const
{
    // Fast path: a node with no connected expressions matches nothing, so
    // skip the canonicalization (a string copy plus a token-registry
    // interning) that the empty scan below would do anyway. Every
    // per-chunk param read in an expression-less graph pays this.
    if (values_.empty()) return nullptr;
    const TfToken canonical = UsdGenCanonicalParamName(destination);
    for (auto const &value : values_)
        if (value.destination == canonical) return &value;
    return nullptr;
}

bool UsdGenCpuParameters::Compile(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                                  std::vector<std::string> *diagnostics)
{
    auto fail = [&](std::string const &destination, std::string const &message) {
        if (diagnostics) diagnostics->push_back(destination + ": " + message);
        items_.clear();
        compiled_ = false;
        return false;
    };
    items_.clear();
    compiled_ = false;
    for (auto const &binding : node.expressionBindings) {
        auto const name = binding.destination.GetString();
        if (binding.destination.IsEmpty() || binding.expression.IsEmpty())
            return fail(name, "empty expression destination or path");
        auto expression = std::find_if(desc.expressions.begin(), desc.expressions.end(),
            [&](auto const &candidate) { return candidate.path == binding.expression; });
        if (expression == desc.expressions.end())
            return fail(name, "missing expression " + binding.expression.GetString());
        // An empty output is a connection to the expression PRIM; it resolves
        // to outputs:result, or to a single declared output.
        UsdGenExpressionOutputDesc const *result =
            UsdGenFindExpressionOutput(*expression, binding.output);
        if (!result)
            return fail(name, binding.output.IsEmpty()
                ? "connection to prim " + binding.expression.GetString() +
                      " declaring no outputs:result and no single outputs:* attribute"
                : "missing expression output " + binding.output.GetString());
        if (!expr::ExprIsDomain(binding.domain))
            return fail(name, "invalid evaluation domain");
        if (!SupportedShape(binding.destinationShape))
            return fail(name, "unsupported array/matrix/type shape");
        if (binding.nativeType.IsEmpty() || result->nativeType != binding.nativeType ||
            !SameShape(result->shape, binding.destinationShape))
            return fail(name, "output native type/shape mismatch");
        Item item;
        item.binding = binding;
        item.canonical = UsdGenCanonicalParamName(binding.destination);
        if (!Literal(binding.literal, binding.destinationShape, &item.literal))
            return fail(name, "literal must have the exact finite native type; "
                              "int64/uint64 currently limited to 2^53");
        auto compiled = expr::Frontend::Compile(expression->source,
            {binding.domain, binding.destinationShape.scalar,
             binding.destinationShape.components});
        if (!compiled.ok) {
            if (diagnostics)
                for (auto const &message : compiled.diagnostics)
                    diagnostics->push_back(name + ": " + message);
            return fail(name, "SeExpr compilation failed");
        }
        item.ir = compiled.program.IR();
        if (!ResolveSamplers(desc, *expression, &item, diagnostics))
            return fail(name, "an expression input could not be resolved");
        items_.push_back(std::move(item));
    }
    compiled_ = true;
    return true;
}

bool UsdGenCpuParameters::Evaluate(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                                   UsdGenCurveBuffer const &geometry, double frame,
                                   double time, uint32_t seed, bool *changed,
                                   std::vector<std::string> *diagnostics)
{
    const uint64_t previous = digest_;
    auto publish = [&](bool ok) {
        if (changed) *changed = digest_ != previous;
        return ok;
    };
    if (node.expressionBindings.empty()) {
        Clear();
        return publish(true);
    }
    // Recompile whenever a binding or its source text moved. This is what lets
    // an expression edit land without the graph being rebuilt underneath us.
    uint64_t sourceDigest = 1469598103934665603ULL;
    for (auto const &binding : node.expressionBindings) {
        sourceDigest = MixText(sourceDigest, binding.destination.GetText());
        sourceDigest = MixText(sourceDigest, binding.expression.GetText());
        sourceDigest = MixText(sourceDigest, binding.output.GetText());
        sourceDigest = MixText(sourceDigest, binding.nativeType.GetText());
        sourceDigest = Mix(sourceDigest, uint64_t(binding.domain));
        sourceDigest = Mix(sourceDigest, uint64_t(binding.destinationShape.scalar));
        sourceDigest = Mix(sourceDigest, binding.destinationShape.components);
        sourceDigest = Mix(sourceDigest,
            binding.literal.IsEmpty() ? 0u : uint64_t(binding.literal.GetHash()));
        auto expression = std::find_if(desc.expressions.begin(), desc.expressions.end(),
            [&](auto const &candidate) { return candidate.path == binding.expression; });
        if (expression == desc.expressions.end()) continue;
        sourceDigest = MixText(sourceDigest, expression->source.c_str());
        // A geoSampler()/ptex() input is compiled into the sampler state, so
        // retargeting it, or editing the data it names, recompiles.
        for (auto const &input : expression->inputs) {
            sourceDigest = MixText(sourceDigest, input.name.GetText());
            for (SdfPath const &path : input.geometries) {
                sourceDigest = MixText(sourceDigest, path.GetText());
                UsdGenGeometryDesc const *sampled = FindGeometry(desc, path);
                sourceDigest = Mix(sourceDigest, sampled ? sampled->generation : ~uint64_t(0));
                if (sampled) {
                    GfMatrix4d const m = ToGroomSpace(desc, *sampled);
                    for (int r = 0; r < 4; ++r)
                        for (int c = 0; c < 4; ++c) sourceDigest = MixDouble(sourceDigest, m[r][c]);
                }
            }
            for (SdfPath const &path : input.maps) {
                sourceDigest = MixText(sourceDigest, path.GetText());
                if (UsdGenMapDesc const *map = FindMap(desc, path)) {
                    sourceDigest = MixText(sourceDigest, map->type.GetText());
                    sourceDigest = MixText(sourceDigest, map->resolvedAssetPath.c_str());
                    sourceDigest = Mix(sourceDigest, map->textureGeneration);
                    for (auto const &param : map->params) {
                        sourceDigest = MixText(sourceDigest, param.name.GetText());
                        sourceDigest = Mix(sourceDigest, uint64_t(param.value.GetHash()));
                    }
                }
            }
        }
    }
    if (!compiled_ || sourceDigest != sourceDigest_) {
        sourceDigest_ = sourceDigest;
        if (!Compile(desc, node, diagnostics)) { Clear(); return publish(false); }
    }

    // --- geometry view over the node's INPUT curves ------------------------
    // Uniform buffers carry no cvOffsets; synthesize the offsets the shared
    // context builder needs so both layouts take the same path.
    const size_t curveCount = geometry.totalCurves;
    const size_t pointCount = geometry.totalCvs;
    offsets_.assign(curveCount + 1, 0);
    if (!geometry.cvOffsets.empty() &&
        size_t(geometry.cvOffsets.size()) == curveCount + 1) {
        for (size_t c = 0; c <= curveCount; ++c)
            offsets_[c] = static_cast<uint32_t>(geometry.cvOffsets[c]);
    } else if (curveCount) {
        const uint32_t perCurve = static_cast<uint32_t>(pointCount / curveCount);
        for (size_t c = 0; c <= curveCount; ++c)
            offsets_[c] = static_cast<uint32_t>(c * perCurve);
    }
    // A generic source may publish no stable ids; $id/$idLo/$idHi then read the
    // curve index, which is what the CPU operators use as the id fallback.
    ids_.clear();
    uint64_t const *ids = geometry.curveId.empty() ? nullptr : geometry.curveId.cdata();
    if (!ids && curveCount) {
        ids_.resize(curveCount);
        for (size_t c = 0; c < curveCount; ++c) ids_[c] = c;
        ids = ids_.data();
    }
    expr::CpuCurveGeometryView view;
    view.px = geometry.px.empty() ? nullptr : geometry.px.cdata();
    view.py = geometry.py.empty() ? nullptr : geometry.py.cdata();
    view.pz = geometry.pz.empty() ? nullptr : geometry.pz.cdata();
    view.rest = geometry.rest.empty() ? nullptr
        : reinterpret_cast<float const *>(geometry.rest.cdata());
    view.widths = geometry.width.empty() ? nullptr : geometry.width.cdata();
    view.hairT = geometry.hairT.empty() ? nullptr : geometry.hairT.cdata();
    view.rootUV = geometry.rootUV.empty() ? nullptr
        : reinterpret_cast<float const *>(geometry.rootUV.cdata());
    // The rest root frame and the surface face index feed $N/$Nref, $dPdu/
    // $dPduref, $dPdv/$dPdvref and $faceId. A producer that did not publish a
    // full per-curve plane leaves the channel null, which keeps the variable
    // unavailable rather than inventing a frame.
    auto rootFrame = [&](VtVec3fArray const &plane) -> float const * {
        return curveCount && plane.size() == curveCount
            ? reinterpret_cast<float const *>(plane.cdata()) : nullptr;
    };
    view.rootN = rootFrame(geometry.rootN);
    view.rootT = rootFrame(geometry.rootT);
    view.rootB = rootFrame(geometry.rootB);
    view.rootPrim = curveCount && geometry.rootPrim.size() == curveCount
        ? geometry.rootPrim.cdata() : nullptr;
    view.stableIds = ids;
    view.curveOffsets = offsets_.data();
    view.curveCount = curveCount;
    view.pointCount = pointCount;

    expr::Context controls;
    controls.frame = frame;
    controls.time = time;
    controls.seed = static_cast<int32_t>(seed);
    controls.descId = expr::DescriptionId(desc.description.GetText());

    expr::CpuExpressionContext contexts[3];
    bool built[3]{};
    for (auto const &item : items_) {
        const size_t index = DomainIndex(item.binding.domain);
        if (built[index]) continue;
        auto domainControls = controls;
        domainControls.domain = index == 0 ? Domain::Groom
            : index == 1 ? Domain::Primitive : Domain::Point;
        std::string diagnostic;
        if (contexts[index].Build(view, domainControls, &diagnostic) !=
            expr::CpuExpressionStatus::Ok) {
            if (diagnostics)
                diagnostics->push_back(node.path.GetString() +
                    ": expression geometry context validation failed: " + diagnostic);
            Clear();
            return publish(false);
        }
        built[index] = true;
    }

    for (auto &item : items_) {
        if (item.samplers.empty()) continue;
        expr::Context itemControls = controls;
        itemControls.domain = item.binding.domain;
        if (!PrepareSamplers(desc, node, geometry, itemControls, &item, diagnostics)) {
            Clear();
            return publish(false);
        }
    }

    std::vector<UsdGenExpressionValue> candidate;
    candidate.reserve(items_.size());
    for (auto const &item : items_) {
        auto inputs = contexts[DomainIndex(item.binding.domain)].Inputs();
        inputs.samplers = item.table.slots.empty() ? nullptr : &item.table;
        const unsigned components = item.binding.destinationShape.components;
        const Type type = item.binding.destinationShape.scalar;
        // $value is the authored literal, broadcast at groom rate exactly as
        // CudaParameterEvaluator installs it.
        inputs.fields[static_cast<unsigned>(expr::Variable::Value)] =
            {item.literal.data(), 1, Domain::Groom, components};
        std::vector<unsigned char> raw(inputs.count * components * ScalarBytes(type), 0);
        expr::CpuExpressionOutput output;
        output.data = raw.empty() ? nullptr : raw.data();
        output.count = inputs.count;
        output.type = type;
        output.components = components;
        const auto status = expr::EvaluateProgram(item.ir, inputs, output);
        if (status != expr::CpuExpressionStatus::Ok) {
            if (diagnostics)
                diagnostics->push_back(item.binding.destination.GetString() + ": " +
                    (status == expr::CpuExpressionStatus::InvalidValue
                        ? "expression produced a value the destination cannot represent"
                        : "expression evaluation was refused"));
            Clear();
            return publish(false);
        }
        UsdGenExpressionValue value;
        value.destination = item.canonical;
        value.domain = item.binding.domain;
        value.type = type;
        value.components = components;
        value.count = inputs.count;
        value.values.resize(inputs.count * components);
        for (size_t i = 0; i < value.values.size(); ++i)
            value.values[i] = Decode(raw.data(), type, i);
        candidate.push_back(std::move(value));
    }

    uint64_t digest = 1469598103934665603ULL;
    for (auto const &value : candidate) {
        digest = MixText(digest, value.destination.GetText());
        digest = Mix(digest, uint64_t(value.domain));
        digest = Mix(digest, uint64_t(value.type));
        digest = Mix(digest, value.components);
        digest = Mix(digest, value.count);
        for (double v : value.values) digest = MixDouble(digest, v);
    }
    values_ = std::move(candidate);
    digest_ = digest;
    return publish(true);
}

} // namespace usdGen
