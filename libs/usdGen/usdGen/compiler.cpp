// usdGen engine — compiler implementation (03-execution-engine.md §3).
//
// Compile() walks UsdGenGraphDesc::nodes and compiler-owned dependency edges
// derived from the composed hierarchy by the descriptor builder; Kahn sort
// with namespace tie-break; cycle detection (compile error
// naming the offending pair); dense node ids in topological order;
// reference-lane ordering (§1.5); OutputPrimvars
// slot binding (§1.2); Merkle structural digests (§3.3); tile arithmetic
// (R21); dirty routing table rebuild data (§5.1).
//
// Recompile() is the incremental path (gate E-6): every node's structural
// digest is recomputed; a node whose digest is unchanged (same type,
// inputs, topology-class params and relationship targets) keeps its op,
// capture and buffer from the previous graph. Exactly one appended node therefore rebuilds exactly one node.
#include "usdGen/compiler.h"

#include "usdGen/expressionTargets.h"
#include "usdGen/ops/clump.h"

#include "usdGen/opRegistry.h"
#include "usdGen/cudaExecution.h"
#include "usdGen/executionBackend.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/trace/trace.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <limits>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

PXR_NAMESPACE_USING_DIRECTIVE


namespace usdGen {
namespace {

int _PopCount64(uint64_t value) noexcept
{
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
#endif
}

int _CountTrailingZeros64(uint64_t value) noexcept
{
    // All callers pass a nonzero bitmap word.
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int count = 0;
    while ((value & 1u) == 0) { value >>= 1; ++count; }
    return count;
#endif
}

bool ValidateAuthoredPlanes(UsdGenGraphDesc const &desc,
                            UsdGenCompileResult &result)
{
    static std::set<TfToken> const reservedNames{
        TfToken("points"), TfToken("rest"), TfToken("widths"),
        TfToken("hairT"), TfToken("curveOffsets"), TfToken("curveId"),
        TfToken("skinprim"), TfToken("skinprimuv"), TfToken("displayColor")};
    bool ok = true;
    for (UsdGenCurveSetDesc const &curveSet : desc.curveSets) {
        size_t pointCount = 0;
        for (int count : curveSet.curveVertexCounts) {
            if (count < 0 || pointCount > std::numeric_limits<size_t>::max() -
                                  static_cast<size_t>(count)) {
                result.errors.push_back("UsdGenCompiler: curve set '" +
                    curveSet.path.GetString() + "' has invalid topology for authored planes");
                ok = false;
                pointCount = 0;
                break;
            }
            pointCount += static_cast<size_t>(count);
        }
        std::set<TfToken> names;
        for (UsdGenAuthoredPlaneDesc const &plane : curveSet.authoredPlanes) {
            bool const validName = !plane.name.IsEmpty() && names.insert(plane.name).second;
            if (!validName) {
                result.errors.push_back("UsdGenCompiler: curve set '" +
                    curveSet.path.GetString() + "' has an empty or duplicate authored plane '" +
                    plane.name.GetString() + "'");
                ok = false;
                continue;
            }
            if (reservedNames.count(plane.name)) {
                result.errors.push_back("UsdGenCompiler: authored plane '" +
                    plane.name.GetString() + "' on curve set '" +
                    curveSet.path.GetString() +
                    "' collides with a reserved C3 channel");
                ok = false;
                continue;
            }
            if (plane.arity < 1 || plane.arity > kUsdGenMaxExtraPlaneSlots) {
                result.errors.push_back("UsdGenCompiler: authored plane '" +
                    plane.name.GetString() + "' on curve set '" + curveSet.path.GetString() +
                    "' has arity outside [1,16]");
                ok = false;
                continue;
            }
            size_t elements = 0;
            switch (plane.domain) {
            case UsdGenAuthoredPlaneDomain::Point: elements = pointCount; break;
            case UsdGenAuthoredPlaneDomain::Primitive:
                elements = curveSet.curveVertexCounts.size(); break;
            case UsdGenAuthoredPlaneDomain::Groom: elements = 1; break;
            default:
                result.errors.push_back("UsdGenCompiler: authored plane '" +
                    plane.name.GetString() + "' has an invalid domain");
                ok = false;
                continue;
            }
            if (elements > std::numeric_limits<size_t>::max() / plane.arity) {
                result.errors.push_back("UsdGenCompiler: authored plane '" +
                    plane.name.GetString() + "' value count overflows this platform");
                ok = false;
                continue;
            }
            size_t const expected = elements * plane.arity;
            bool payload = false;
            switch (plane.type) {
            case UsdGenAuthoredPlaneType::Float32:
                payload = plane.floatValues.size() == expected && plane.intValues.empty();
                break;
            case UsdGenAuthoredPlaneType::Int32:
                payload = plane.intValues.size() == expected && plane.floatValues.empty();
                break;
            default: break;
            }
            if (!payload) {
                result.errors.push_back("UsdGenCompiler: authored plane '" +
                    plane.name.GetString() + "' has invalid type or cardinality");
                ok = false;
            }
        }
    }
    return ok;
}

bool BindExtraPlaneSlots(UsdGenOp const &op, UsdGenCompiledNode *node,
                         UsdGenCompileResult *result)
{
    auto bind = [&](TfSpan<const TfToken> names, std::vector<TfToken> *slots,
                    char const *direction) {
        if (names.size() > kUsdGenMaxExtraPlaneSlots) {
            result->errors.push_back(std::string("UsdGenCompiler: operator '") +
                node->desc->path.GetString() + "' declares too many " +
                direction + " primvar slots");
            return false;
        }
        std::set<TfToken> seen;
        slots->clear();
        slots->reserve(names.size());
        for (TfToken const &name : names) {
            if (name.IsEmpty() || !seen.insert(name).second) {
                result->errors.push_back(std::string("UsdGenCompiler: operator '") +
                    node->desc->path.GetString() + "' declares an empty or duplicate " +
                    direction + " primvar slot");
                return false;
            }
            slots->push_back(name);
        }
        return true;
    };
    return bind(op.OutputPrimvars(), &node->outputPrimvars, "output") &&
        bind(op.InputPrimvars(), &node->inputPrimvars, "input");
}

bool ValidateExpressionBindings(UsdGenGraphDesc const &desc,
                                UsdGenCompileResult &result,
                                bool injectedDevice = false)
{
    bool ok = true;
    ok = ValidateAuthoredPlanes(desc, result) && ok;
    if (!desc.validationErrors.empty()) {
        result.errors.insert(result.errors.end(), desc.validationErrors.begin(), desc.validationErrors.end());
        ok = false;
    }
    UsdGenDiagnostics backendDiagnostics;
    if (!injectedDevice &&
        !ValidateUsdGenExecutionBackend(desc.executionBackend, &backendDiagnostics)) {
        result.errors.insert(result.errors.end(), backendDiagnostics.errors.begin(),
                             backendDiagnostics.errors.end());
        ok = false;
    }
    if (ok && desc.executionBackend == UsdGenExecutionBackend::Cuda) {
        UsdGenDiagnostics diagnostics;
        if (!ValidateCudaGraph(desc, &diagnostics)) {
            result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
            ok = false;
        }
    }
    std::set<SdfPath> expressionPaths;
    std::set<SdfPath> geometryPaths, mapPaths;
    for (auto const &g : desc.geometries) {
        if (g.path.IsEmpty() || !geometryPaths.insert(g.path).second) {
            result.errors.push_back("invalid or duplicate expression geometry " + g.path.GetString());
            ok = false;
        }
    }
    for (auto const &m : desc.maps) mapPaths.insert(m.path);
    for (auto const &e : desc.expressions) {
        if (e.path.IsEmpty() || e.source.empty() || !expressionPaths.insert(e.path).second) {
            result.errors.push_back("expression has missing path or source"); ok = false;
        }
        std::set<TfToken> inputNames;
        for (auto const &input : e.inputs) {
            if (input.name.IsEmpty() || !inputNames.insert(input.name).second) {
                result.errors.push_back("expression " + e.path.GetString() +
                    " has an empty or duplicate input:" + input.name.GetString());
                ok = false;
            }
            for (SdfPath const &g : input.geometries)
                if (!geometryPaths.count(g)) {
                    result.errors.push_back("expression " + e.path.GetString() + " input:" +
                        input.name.GetString() + " names unresolved geometry " + g.GetString());
                    ok = false;
                }
            for (SdfPath const &m : input.maps)
                if (!mapPaths.count(m)) {
                    result.errors.push_back("expression " + e.path.GetString() + " input:" +
                        input.name.GetString() + " names unresolved map " + m.GetString());
                    ok = false;
                }
        }
        std::set<TfToken> seen;
        for (auto const &o : e.outputs)
            if (o.name.IsEmpty() || o.nativeType.IsEmpty() || !seen.insert(o.name).second) { result.errors.push_back("expression " + e.path.GetString() + " has invalid or duplicate output"); ok = false; }
    }
    for (auto const &node : desc.nodes) {
        std::set<TfToken> destinations;
        // Both lanes run an expression evaluator, so a connected parameter is
        // no longer backend-gated. What each OPERATOR admits is the one shared
        // table in expressionTargets.cpp, used here and by the CUDA admission.
        if (!UsdGenValidateExpressionTargets(node, &result.errors)) ok = false;
        for (auto const &b : node.expressionBindings) {
            auto ei = std::find_if(desc.expressions.begin(), desc.expressions.end(), [&](auto const &e){ return e.path == b.expression; });
            if (ei == desc.expressions.end()) { result.errors.push_back("expression binding references missing expression " + b.expression.GetString()); ok = false; continue; }
            // An empty output is a connection to the expression PRIM; it
            // resolves to outputs:result, or to a single declared output.
            UsdGenExpressionOutputDesc const *oi =
                UsdGenFindExpressionOutput(*ei, b.output);
            if (!oi) {
                result.errors.push_back(b.output.IsEmpty()
                    ? "expression binding connects to prim " + b.expression.GetString() +
                          " which declares no outputs:result and no single outputs:* attribute"
                    : "expression binding references missing or ambiguous output " + b.output.GetString());
                ok = false; continue;
            }
            if (!destinations.insert(b.destination).second) { result.errors.push_back("duplicate expression consumer " + node.path.GetString() + "." + b.destination.GetString()); ok = false; }
            if (!(b.domain == expr::Domain::Groom || b.domain == expr::Domain::Primitive || b.domain == expr::Domain::Point)) { result.errors.push_back("invalid expression evaluation domain"); ok = false; }
            if (b.destination.IsEmpty() || b.nativeType.IsEmpty()) { result.errors.push_back("expression binding has empty destination or native type"); ok = false; }
            // Hydra transports native property names; direct descriptor
            // clients may use operator-local names. Strip only our namespace.
            std::string destination = b.destination.GetString();
            if (destination.compare(0, 7, "usdGen:") == 0)
                destination.erase(0, 7);
            if ((destination == "enabled" || destination == "seed" ||
                 destination == "segments" || destination == "cvCount") &&
                b.domain != expr::Domain::Groom) {
                result.errors.push_back("topology/control expression must evaluate at groom domain"); ok = false;
            }
            auto const &a = oi->shape; auto const &d = b.destinationShape;
            if (b.nativeType != oi->nativeType || a.scalar == expr::ScalarType::Invalid || d.scalar == expr::ScalarType::Invalid || a.scalar != d.scalar || a.elementCount != d.elementCount || a.components != d.components || a.rows != d.rows || a.columns != d.columns || a.isArray != d.isArray) {
                result.errors.push_back("expression output type/shape does not match destination " + b.destination.GetString()); ok = false;
            }
        }
    }
    return ok;
}

// FNV-1a 64-bit.
uint64_t Fnv1a(uint64_t h, void const *data, size_t n)
{
    auto const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

uint64_t Fnv1aI64(uint64_t h, int64_t v)
{
    uint64_t u = static_cast<uint64_t>(v);
    return Fnv1a(h, &u, sizeof(u));
}

uint64_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
uint64_t Fnv1aCstr(uint64_t h, char const *s)
{
    for (auto const *p = reinterpret_cast<unsigned char const *>(s); *p; ++p) {
        h ^= *p; h *= 0x100000001b3ULL;
    }
    return h;
}
uint64_t Fnv1aTfToken(uint64_t h, TfToken const &t)
{
    // GetText(): the interned c-string; GetString() would allocate a
    // std::string per call on the E-6 hot path.
    return Fnv1aI64(Fnv1aCstr(h, t.GetText()),
                    static_cast<int64_t>(t.Hash()));
}

uint64_t MapValueIdentity(UsdGenMapDesc const &map)
{
    uint64_t h = Fnv1aCstr(1469598103934665603ULL, map.path.GetText());
    h = Fnv1aTfToken(h, map.type);
    h = Fnv1aCstr(h, map.resolvedAssetPath.c_str());
    h = Fnv1aI64(h, static_cast<int64_t>(map.textureGeneration));
    // A UsdGenPaintMap's payload is the surface primvar snapshot, not a
    // file: a repaint changes only paintValues, so they are the identity a
    // direct map-slot consumer re-captures on (the execution-cache key in
    // sessionCooker.cpp folds the same fields).
    if (!map.paintValues.empty() || !map.paintSurface.IsEmpty()) {
        h = Fnv1aCstr(h, map.paintSurface.GetText());
        h = Fnv1aTfToken(h, map.paintPrimvar);
        h = Fnv1aTfToken(h, map.paintInterpolation);
        h = Fnv1aI64(h, static_cast<int64_t>(map.paintValues.size()));
        for (float v : map.paintValues) {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            h = Fnv1aI64(h, static_cast<int64_t>(bits));
        }
    }
    std::vector<UsdGenParamValue const *> params;
    params.reserve(map.params.size());
    for (UsdGenParamValue const &param : map.params) params.push_back(&param);
    std::sort(params.begin(), params.end(), [](auto const *a, auto const *b) {
        if (a->name != b->name) return a->name < b->name;
        if (a->value.GetHash() != b->value.GetHash())
            return a->value.GetHash() < b->value.GetHash();
        return a->animated < b->animated;
    });
    for (UsdGenParamValue const *param : params) {
        h = Fnv1aTfToken(h, param->name);
        h = Fnv1aI64(h, static_cast<int64_t>(param->value.GetHash()));
        h = Fnv1aI64(h, param->animated ? 1 : 0);
    }
    return h;
}

// mapBindings is the canonical transport.  The old maps array remains a
// source-compatible input surface, but when both forms are present they must
// describe the same ordered target sequence.
bool EffectiveMapBindings(UsdGenNodeDesc const &node,
                          std::vector<UsdGenMapBindingDesc> *out,
                          std::string *error)
{
    out->clear();
    if (node.mapBindings.empty()) {
        out->reserve(node.maps.size());
        for (SdfPath const &path : node.maps)
            out->push_back({path, TfToken()});
        return true;
    }
    if (!node.maps.empty()) {
        if (node.maps.size() != node.mapBindings.size()) {
            *error = "legacy map paths and typed map bindings disagree";
            return false;
        }
        for (size_t i = 0; i < node.maps.size(); ++i) {
            if (node.maps[i] != node.mapBindings[i].map) {
                *error = "legacy map paths and typed map bindings disagree";
                return false;
            }
        }
    }
    *out = node.mapBindings;
    return true;
}

bool ValidateMapBinding(UsdGenMapBindingDesc const &binding,
                        std::string *error)
{
    if (binding.map.IsEmpty()) {
        *error = "binding has an empty map target";
        return false;
    }
    // Empty relationship is deliberately accepted for direct typed clients
    // and maps-only compatibility descriptors. `usdGen:regionMap` is the
    // categorical Ptex owner-map slot of CurveSource surfaceCage; the
    // operator admission layer enforces that it is used only there.
    if (binding.relationship.IsEmpty()) return true;
    // usdGen:densityMap (Scatter) and usdGen:lengthMap (Length) are root-sampled
    // mask maps; the operators enforce their own map types.
    if (binding.relationship != TfToken("usdGen:map") &&
        binding.relationship != TfToken("usdGen:regionMap") &&
        binding.relationship != TfToken("usdGen:densityMap") &&
        binding.relationship != TfToken("usdGen:lengthMap")) {
        *error = "relationship '" + binding.relationship.GetString() +
            "' is not a map relationship";
        return false;
    }
    return true;
}

bool BuildReferenceValue(UsdGenCurveSetDesc const &curves,
                         UsdGenResolvedReferenceValue *out,
                         std::string *error)
{
    if (curves.path.IsEmpty()) {
        *error = "reference curve set has an empty path";
        return false;
    }
    uint64_t totalCvs = 0;
    for (int count : curves.curveVertexCounts) {
        if (count < 0) {
            *error = "reference curve set '" + curves.path.GetString() +
                "' has a negative curve vertex count";
            return false;
        }
        totalCvs += static_cast<uint32_t>(count);
    }
    if (totalCvs > UINT32_MAX || curves.points.size() != totalCvs ||
        (!curves.rest.empty() && curves.rest.size() != totalCvs) ||
        (!curves.widths.empty() && curves.widths.size() != totalCvs) ||
        (!curves.curveId.empty() &&
         curves.curveId.size() != curves.curveVertexCounts.size()) ||
        (!curves.skinPrim.empty() &&
         curves.skinPrim.size() != curves.curveVertexCounts.size()) ||
        (!curves.skinPrimUv.empty() &&
         curves.skinPrimUv.size() != curves.curveVertexCounts.size()) ||
        (!curves.rootFrame.empty() &&
         curves.rootFrame.size() != curves.curveVertexCounts.size())) {
        *error = "reference curve set '" + curves.path.GetString() +
            "' has inconsistent topology/value array sizes";
        return false;
    }

    // A reference value owns the optional C3 rest plane as immutable COW
    // payload.  Validate it before constructing the shared value: an absent
    // rest plane follows the CUDA/reference contract and falls back to the
    // authored points, while a present plane must be complete and finite.
    for (GfVec3f const &rest : curves.rest) {
        if (!std::isfinite(rest[0]) || !std::isfinite(rest[1]) ||
            !std::isfinite(rest[2])) {
            *error = "reference curve set '" + curves.path.GetString() +
                "' has non-finite rest points";
            return false;
        }
    }

    auto value = std::make_shared<UsdGenReferenceSet>();
    value->generation = curves.curveGeneration;
    UsdGenCurveBuffer &buffer = value->buffer;
    buffer.totalCurves = static_cast<uint32_t>(curves.curveVertexCounts.size());
    buffer.totalCvs = static_cast<uint32_t>(totalCvs);
    buffer.topologyVersion = curves.curveGeneration;
    buffer.valueVersion = curves.curveGeneration;
    bool uniform = true;
    int const firstCount = curves.curveVertexCounts.empty()
        ? 0 : curves.curveVertexCounts.front();
    for (int count : curves.curveVertexCounts) uniform = uniform && count == firstCount;
    if (!uniform) {
        buffer.cvOffsets.resize(buffer.totalCurves + 1, 0);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i)
            buffer.cvOffsets[i + 1] = buffer.cvOffsets[i] + curves.curveVertexCounts[i];
    }
    buffer.width = curves.widths;
    buffer.rest = curves.rest.empty() ? curves.points : curves.rest;
    buffer.curveId = curves.curveId;
    if (buffer.curveId.empty()) {
        buffer.curveId.resize(buffer.totalCurves);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i) buffer.curveId[i] = i;
    }
    buffer.rootPrim = curves.skinPrim;
    buffer.rootUV = curves.skinPrimUv;
    if (!curves.rootFrame.empty()) {
        // GfMatrix4d is row-major/row-vector.  Root-frame consumers use the
        // same tangent/binormal/normal row convention as the CPU kernels.
        buffer.rootT.resize(buffer.totalCurves);
        buffer.rootB.resize(buffer.totalCurves);
        buffer.rootN.resize(buffer.totalCurves);
        for (uint32_t i = 0; i < buffer.totalCurves; ++i) {
            GfMatrix4d const &frame = curves.rootFrame[i];
            GfVec3d const tangent = frame.GetRow3(0);
            GfVec3d const binormal = frame.GetRow3(1);
            GfVec3d const normal = frame.GetRow3(2);
            buffer.rootT[i] = GfVec3f(tangent);
            buffer.rootB[i] = GfVec3f(binormal);
            buffer.rootN[i] = GfVec3f(normal);
        }
    }
    buffer.px.resize(buffer.totalCvs);
    buffer.py.resize(buffer.totalCvs);
    buffer.pz.resize(buffer.totalCvs);
    buffer.hairT.resize(buffer.totalCvs);
    value->localX.resize(buffer.totalCvs);
    value->localY.resize(buffer.totalCvs);
    value->localZ.resize(buffer.totalCvs);
    uint32_t cv = 0;
    for (int count : curves.curveVertexCounts) {
        for (int i = 0; i < count; ++i, ++cv) {
            GfVec3f const &point = curves.points[cv];
            buffer.px[cv] = value->localX[cv] = point[0];
            buffer.py[cv] = value->localY[cv] = point[1];
            buffer.pz[cv] = value->localZ[cv] = point[2];
            buffer.hairT[cv] = count > 1 ? float(i) / float(count - 1) : 0.0f;
        }
    }
    for (UsdGenAuthoredPlaneDesc const &authored : curves.authoredPlanes) {
        UsdGenPlane plane;
        plane.name = authored.name;
        plane.arity = authored.arity;
        switch (authored.domain) {
        case UsdGenAuthoredPlaneDomain::Point:
            plane.interpolation = TfToken("vertex");
            break;
        case UsdGenAuthoredPlaneDomain::Primitive:
            plane.interpolation = TfToken("uniform");
            break;
        case UsdGenAuthoredPlaneDomain::Groom:
            plane.interpolation = TfToken("constant");
            break;
        default:
            *error = "reference curve set '" + curves.path.GetString() +
                "' has an invalid authored plane domain";
            return false;
        }
        if (authored.type == UsdGenAuthoredPlaneType::Float32) {
            plane.type = TfToken("float");
            plane.f = authored.floatValues;
        } else if (authored.type == UsdGenAuthoredPlaneType::Int32) {
            plane.type = TfToken("int");
            plane.i = authored.intValues;
        } else {
            *error = "reference curve set '" + curves.path.GetString() +
                "' has an invalid authored plane type";
            return false;
        }
        if (plane.interpolation == TfToken("vertex"))
            buffer.extraCv.push_back(std::move(plane));
        else
            buffer.extraCurve.push_back(std::move(plane));
    }
    auto byName = [](UsdGenPlane const &a, UsdGenPlane const &b) {
        return a.name < b.name;
    };
    std::sort(buffer.extraCv.begin(), buffer.extraCv.end(), byName);
    std::sort(buffer.extraCurve.begin(), buffer.extraCurve.end(), byName);
    out->path = curves.path;
    out->curveGeneration = curves.curveGeneration;
    out->identity = Fnv1aI64(Fnv1aCstr(1469598103934665603ULL,
                                       curves.path.GetText()),
                              static_cast<int64_t>(curves.curveGeneration));
    out->value = std::move(value);
    return true;
}


/// 128-bit digest = two independent FNV lanes over the term list
/// (ADR §4.2.1). `enabled` and `seed` are NOT digest terms; only the
/// topology-class parameter values (02 §6) enter, together with type, mode,
/// sorted input paths, sorted reference/map/surface relationship targets,
/// and the children's digests.
UsdGenEpoch ComputeNodeDigest(
    UsdGenNodeDesc const &nd,
    TfToken const &type,
    TfSpan<const TfToken> topoParams,
    std::vector<SdfPath> const &inputPaths,
    std::vector<SdfPath> const &refPaths,
    std::vector<UsdGenMapBindingDesc> const &mapBindings,
    std::vector<std::pair<SdfPath, UsdGenEpoch>> const &childDigests)
{
    uint64_t h0 = 1469598103934665603ULL;
    uint64_t h1 = 1469598103934665603ULL;
    auto mix = [&h0, &h1](uint64_t v) {
        h0 = Fnv1aI64(h0, static_cast<int64_t>(v));
        h1 = Fnv1aI64(h1, static_cast<int64_t>(~v));
    };

    mix(0xa5a5a5a5ULL ^ Fnv1aCstr(0, type.GetText()));
    mix(0x00020002ULL ^ Fnv1aTfToken(0, nd.mode));
    for (SdfPath const &p : inputPaths) mix(0x00050005ULL ^ Fnv1aCstr(0, p.GetText()));
    for (SdfPath const &p : refPaths)   mix(0x00060006ULL ^ Fnv1aCstr(0, p.GetText()));
    for (UsdGenMapBindingDesc const &binding : mapBindings) {
        mix(0x00060007ULL ^ Fnv1aCstr(0, binding.map.GetText()));
        mix(0x00060009ULL ^ Fnv1aTfToken(0, binding.relationship));
    }
    for (auto const &param : nd.params) {
        for (TfToken const &t : topoParams) {
            if (param.name == t) {
                mix(0x00070007ULL ^ Fnv1aTfToken(0, t));
                mix(0x00080008ULL ^ param.value.GetHash());
                break;
            }
        }
    }
    for (SdfPath const &p : inputPaths) {
        for (auto const &kv : childDigests)
            if (kv.first == p) { mix(kv.second[0] ^ (kv.second[1] << 1)); break; }
    }
    return {h0, h1};
}

// 02-schema.md §6 dirty classification, per (type, param). The §6.1
// graph-structural rows are DIGEST terms (a change recompiles the node);
// the §6.2 topology rows and the §6.4 capture rows are not.

// Eager library-owned tokens preserve E-6's pointer-comparison path without
// lazy function-static construction/destruction. They initialize before the
// runtime-owner singleton is first constructed, so the owner quiesces before
// these TU statics are destroyed at process exit.
namespace tok {
namespace {
const TfToken t{"input"},
    guides{"guides"}, curves{"curves"}, surface{"surface"},
    enabled{"enabled"}, seed{"seed"},
    segments{"segments"}, direction{"direction"},
    lengthMethod{"length:method"}, rebuild{"rebuild"}, replace{"replace"},
    lengthMode{"length:mode"}, cullThreshold{"cullThreshold"},
    grow{"UsdGenGrow"}, length{"UsdGenLength"},
    width{"UsdGenWidth"}, clump{"UsdGenClump"},
    clumpLevel{"clump:level"}, clumpLevels{"clump:levels"},
    curl{"UsdGenCurl"}, bend{"UsdGenBend"}, part{"UsdGenPart"},
    axisMode{"axisMode"}, partCurves{"part:curves"},
    directionOp{"UsdGenDirection"}, smooth{"UsdGenSmooth"},
    resample{"UsdGenResample"}, displace{"UsdGenDisplace"},
    mode{"mode"}, directionSource{"direction:source"},
    displaceMap{"displace:map"}, distribution{"distribution"},
    sculpt{"UsdGenSculptLayer"}, sculptSpace{"sculpt:space"},
    exprOp{"UsdGenExprOp"}, exprMaps{"expr:maps"},
    exprReturnType{"expr:returnType"},
    freeze{"UsdGenFreeze"}, frozenMode{"frozen:mode"},
    frozenCurves{"frozen:curves"};
}
const TfToken &T() { return t; }
const TfToken &Guides() { return guides; }
const TfToken &Curves() { return curves; }
const TfToken &Surface() { return surface; }
const TfToken &Enabled() { return enabled; }
const TfToken &Seed() { return seed; }
const TfToken &Segments() { return segments; }
const TfToken &Direction() { return direction; }
const TfToken &LengthMethod() { return lengthMethod; }
const TfToken &Rebuild() { return rebuild; }
const TfToken &Replace() { return replace; }
const TfToken &LengthMode() { return lengthMode; }
const TfToken &CullThreshold() { return cullThreshold; }
const TfToken &Grow() { return grow; }
const TfToken &Length() { return length; }
const TfToken &Width() { return width; }
const TfToken &Clump() { return clump; }
const TfToken &ClumpLevel() { return clumpLevel; }
const TfToken &ClumpLevels() { return clumpLevels; }
const TfToken &Curl() { return curl; }
const TfToken &Bend() { return bend; }
const TfToken &Part() { return part; }
const TfToken &AxisMode() { return axisMode; }
const TfToken &PartCurves() { return partCurves; }
const TfToken &DirectionOp() { return directionOp; }
const TfToken &Smooth() { return smooth; }
const TfToken &Resample() { return resample; }
const TfToken &Displace() { return displace; }
const TfToken &Mode() { return mode; }
const TfToken &DirectionSource() { return directionSource; }
const TfToken &DisplaceMap() { return displaceMap; }
const TfToken &Distribution() { return distribution; }
const TfToken &Sculpt() { return sculpt; }
const TfToken &SculptSpace() { return sculptSpace; }
const TfToken &ExprOp() { return exprOp; }
const TfToken &ExprMaps() { return exprMaps; }
const TfToken &ExprReturnType() { return exprReturnType; }
const TfToken &Freeze() { return freeze; }
const TfToken &FrozenMode() { return frozenMode; }
const TfToken &FrozenCurves() { return frozenCurves; }
}  // namespace tok

/// §6.1: this parameter is a term of the node's Merkle structural digest
/// (recompile on edit).
bool IsDigestParam(TfToken const &type, TfToken const &param)
{
    if (type == tok::Grow())
        return param == tok::Segments() || param == tok::Direction();
    if (type == tok::Length())
        return param == tok::LengthMethod() || param == tok::Rebuild() ||
               param == tok::LengthMode();
    if (type == tok::Width())
        return param == tok::Replace();
    // Clump's emitted plane names (clumpId_<level>...) follow these two.
    if (type == tok::Clump())
        return param == tok::ClumpLevel() || param == tok::ClumpLevels();
    // Curl/Bend select their kernel frame via axisMode; Part's parting set is
    // a graph edge (02 §2.7.2 structural rows).
    if (type == tok::Curl() || type == tok::Bend())
        return param == tok::AxisMode();
    if (type == tok::Part())
        return param == tok::PartCurves();
    // Direction/Smooth/Displace select kernel branches via mode; the
    // direction:source graph edge and the displace:map dependency are
    // structural like part:curves; Resample's distribution reselects the
    // resampling kernel (cvCount stays capture-class: same recompile-free
    // recapture as Grow's segments).
    if (type == tok::DirectionOp())
        return param == tok::Mode() || param == tok::DirectionSource();
    if (type == tok::Smooth())
        return param == tok::Mode();
    if (type == tok::Displace())
        return param == tok::Mode() || param == tok::DisplaceMap();
    if (type == tok::Resample())
        return param == tok::Distribution();
    // SculptLayer's delta frame, ExprOp's program identity (mode, the maps
    // edge and the write target) and Freeze's snapshot identity (mode plus
    // the frozen:curves edge, a part:curves-style graph edge) reselect the
    // kernel branch or the captured source: structural like the rows above.
    if (type == tok::Sculpt())
        return param == tok::SculptSpace();
    if (type == tok::ExprOp())
        return param == tok::Mode() || param == tok::ExprMaps() ||
            param == tok::ExprReturnType();
    if (type == tok::Freeze())
        return param == tok::FrozenMode() || param == tok::FrozenCurves();
    // Collide's relax branch publishes collideUnresolved (Configure), so its
    // resolveType rebuilds the node like Clump's level rows.
    if (type == TfToken("UsdGenCollide"))
        return param == TfToken("resolveType");
    return false;
}

/// §6.2: topology-class rows that re-allocate without touching the digest.
bool IsTopologyParam(TfToken const &type, TfToken const &param)
{
    if (type == tok::Length())
        return param == tok::LengthMode() || param == tok::CullThreshold();
    return false;
}

/// Classify one authored parameter into UsdGenDirtyBits (02 §6 row table).
/// `inTopoList` states that the caller already KNOWS the param is one of
/// op.TopologyParameters() — the membership tail loops are the O(n^2) term
/// of building every node's routing table and are pure waste when the
/// caller iterates that very list.
uint32_t ClassifyParamBits(
    UsdGenOp const &op, TfToken const &type, TfToken const &param,
    UsdGenTopoFx topoFx, bool inTopoList = false)
{
    if (param == tok::T()) {
        return UsdGenDirtyStructural;
    }
    // Relationship retargets are graph-structural (02 §6.1): they change a
    // graph edge / the digest's relationship list.
    if (param == tok::Guides() || param == tok::Curves()) {
        return UsdGenDirtyStructural;
    }
    if (param == tok::Surface()) {
        return UsdGenDirtyStructural;  // bound-surface retarget: recompile
    }
    if (param == tok::Enabled()) {
        // 02 §6.2/§6.3: topology for generators and Length (its static
        // TopologyEffect() is CurveCount), value-toggle for the rest. A
        // topology-class toggle must also re-capture: a disabled generator
        // publishes an empty curve set and re-deriving it is a capture.
        return (topoFx != UsdGenTopoFx::None)
            ? (UsdGenDirtyTopology | UsdGenDirtyCapture)
            : UsdGenDirtyParameter;
    }
    if (param == tok::Seed()) {
        return UsdGenDirtyCapture;
    }
    if (IsDigestParam(type, param)) {
        return UsdGenDirtyStructural;
    }
    if (IsTopologyParam(type, param)) {
        return UsdGenDirtyTopology;
    }
    if (inTopoList) return UsdGenDirtyCapture;       // §6.4 capture-class edit
    for (TfToken const &t : op.TopologyParameters()) {
        if (t == param) return UsdGenDirtyCapture;   // §6.4 capture-class edit
    }
    for (TfToken const &t : op.ValueParameters()) {
        if (t == param) return UsdGenDirtyParameter;
    }
    // Unknown usdGen:* leaf: value-dirty is the safe superset (02 §6).
    return UsdGenDirtyParameter;
}

/// The (routing table, digest-param list) of an operator TYPE is a static
/// function of the type: build it ONCE per type, not once per node. E-6
/// walks 200+ nodes in one recompile; without this cache the classification
/// is O(nodes x params x param-list) per call.
struct TypeClassTable
{
    std::vector<std::pair<TfToken, uint32_t>> routing;  // paramRouting content
    TfTokenVector digestParams;   // TopologyParameters() ∩ §6.1 digest rows
};

TypeClassTable const &TypeClassification(UsdGenOp const &op)
{
    thread_local std::unordered_map<std::string, TypeClassTable> cache;
    TfToken const type = op.Type();
    auto it = cache.find(type.GetString());
    if (it != cache.end()) return it->second;

    TypeClassTable tbl;
    UsdGenTopoFx const topoFx = op.TopologyEffect();
    for (TfToken const &t : op.TopologyParameters()) {
        tbl.routing.push_back({t, ClassifyParamBits(op, type, t, topoFx, true)});
        if (IsDigestParam(type, t)) tbl.digestParams.push_back(t);
    }
    for (TfToken const &t : op.ValueParameters())
        tbl.routing.push_back({t, UsdGenDirtyParameter});
    return cache.emplace(type.GetString(), std::move(tbl)).first->second;
}

}  // namespace

// ---------------------------------------------------------------------------

UsdGenCompileResult UsdGenCompiler::Compile(UsdGenGraphDesc const &desc, UsdGenGraph *out)
{
    TRACE_FUNCTION();
    UsdGenCompileResult result;
    if (!ValidateExpressionBindings(desc, result)) return result;
    UsdGenGraph candidate;
    _Build(desc, &candidate, /*reuse=*/nullptr, result);
    if (result.errors.empty() && desc.executionBackend == UsdGenExecutionBackend::Cuda) {
        UsdGenDiagnostics diagnostics;
        candidate._cudaPlan = CompileCudaGraph(desc, &diagnostics);
        result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
        if (!candidate._cudaPlan && result.errors.empty())
            result.errors.push_back("CUDA plan compilation failed");
    }
    if (result.errors.empty()) {
        *out = std::move(candidate);
        out->_inputVersions =
            UsdGenExecutionInputVersions::FromDescription(out->Desc());
        result.ok = true;
        // Fresh compile: every node re-captures; all chunks value-dirty so
        // the first commit evaluates the whole chain.
        for (auto &np : out->_nodes) {
            np->captureNeeded = true;
            std::fill(np->chunkDirty.begin(), np->chunkDirty.end(),
                      UsdGenDirtyParameter);
        }
    }
    return result;
}

UsdGenCompileResult UsdGenCompiler::CompileInjectedDevice(
    UsdGenGraphDesc const& desc, UsdGenGraph* out,
    DevicePlanCompiler const& compile,
    std::shared_ptr<const UsdGenExecutionPlanHandle>* plan)
{
    UsdGenCompileResult result;
    if (!out || !plan || !compile ||
        desc.executionBackend != UsdGenExecutionBackend::Vulkan) {
        result.errors.push_back("injected device compilation requires a Vulkan provider and valid outputs");
        return result;
    }
    // Injection bypasses only the process-wide availability lookup. All
    // authoring, space, expression, factory and graph checks remain in force.
    if (!ValidateExpressionBindings(desc, result, true)) return result;
    UsdGenDiagnostics diagnostics;
    auto nativePlan = compile(desc, &diagnostics);
    result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
    result.warnings.insert(result.warnings.end(), diagnostics.warnings.begin(), diagnostics.warnings.end());
    if (!nativePlan || nativePlan->Backend() != desc.executionBackend ||
        !nativePlan->Metadata() || !nativePlan->Payload()) {
        result.errors.push_back("injected device provider returned no matching immutable plan");
    }
    if (!result.errors.empty()) return result;
    UsdGenGraph candidate;
    _Build(desc, &candidate, nullptr, result);
    if (!result.errors.empty()) return result;
    candidate._inputVersions = UsdGenExecutionInputVersions::FromDescription(candidate.Desc());
    for (auto& node : candidate._nodes) {
        node->captureNeeded = true;
        std::fill(node->chunkDirty.begin(), node->chunkDirty.end(), UsdGenDirtyParameter);
    }
    *out = std::move(candidate);
    *plan = std::move(nativePlan);
    result.ok = true;
    return result;
}

UsdGenCompileResult UsdGenCompiler::Recompile(UsdGenGraphDesc const &newDesc, UsdGenGraph *out)
{
    // CUDA programs are immutable per descriptor. Compile transactionally so
    // a failed edit cannot invalidate the last usable program or generation.
    if (newDesc.executionBackend == UsdGenExecutionBackend::Cuda)
        return Compile(newDesc, out);
    UsdGenCompileResult result;
    if (!ValidateExpressionBindings(newDesc, result)) return result;
    _Build(newDesc, out, out, result);
    if (result.errors.empty()) {
        result.ok = true;
        out->_inputVersions =
            UsdGenExecutionInputVersions::FromDescription(out->Desc());
        // Rebuilt nodes re-capture; digest-stable nodes keep their runtime
        // state (capture, buffer, chunk + dirty bytes) moved over by
        // _Build. The runtime partition is re-established by the first
        // commit once a generator installs its topology — a recompile
        // cannot know the live curve count of an unpopulated output, so it
        // must not call Repartition() here (that would wipe the moved
        // chunk/dirty state and capture flags).
        for (auto &np : out->_nodes) np->captureNeeded = false;
        for (UsdGenNodeId id : result.rebuilt) out->Node(id).captureNeeded = true;
    }
    return result;
}

void UsdGenCompiler::_Build(
    UsdGenGraphDesc const &desc,
    UsdGenGraph *out,
    UsdGenGraph const *reuse,
    UsdGenCompileResult &result)
{
    usdGenRegisterM1Operators();   // idempotent

    if (desc.nodes.empty()) {
        result.errors.push_back("UsdGenCompiler: graph has no operator nodes");
        return;
    }
    if (desc.terminal.IsEmpty()) {
        result.errors.push_back("UsdGenCompiler: graph has no terminal operator");
        return;
    }

    // desc index by path (namespace order == desc.nodes' order, S26).
    // Sorted vector, not unordered_map (E-6): one allocation + sort instead
    // of ~200 per-insert map-node mallocs; binary search per lookup.
    // SdfPath::operator< orders by full path.
    std::vector<std::pair<SdfPath, int>> descIdxByPath;
    descIdxByPath.reserve(desc.nodes.size());
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        descIdxByPath.emplace_back(desc.nodes[i].path, static_cast<int>(i));
    }
    std::sort(descIdxByPath.begin(), descIdxByPath.end(),
              [](auto const &a, auto const &b) { return a.first < b.first; });
    auto findDescIdx = [&](SdfPath const &p) {
        auto it = std::lower_bound(
            descIdxByPath.begin(), descIdxByPath.end(), p,
            [](std::pair<SdfPath, int> const &e, SdfPath const &v) { return e.first < v; });
        return (it != descIdxByPath.end() && !(p < it->first)) ? it
            : descIdxByPath.end();
    };
    auto termIt = findDescIdx(desc.terminal);
    if (termIt == descIdxByPath.end()) {
        result.errors.push_back(
            std::string("UsdGenCompiler: terminal '") +
            desc.terminal.GetText() + "' is not an operator prim in the graph");
        return;
    }

    int const n = static_cast<int>(desc.nodes.size());
    // Edge resolution, fused single pass (E-6): every input's desc index is
    // looked up ONCE into flat arrays shared by the counters, the CSR fill
    // and Kahn — not re-searched per pass.
    std::vector<int> edgeOff(size_t(n) + 1, 0);
    for (int i = 0; i < n; ++i)
        edgeOff[size_t(i) + 1] = edgeOff[size_t(i)] + int(desc.nodes[i].inputs.size());
    std::vector<int> edgeIds;
    edgeIds.assign(size_t(edgeOff[size_t(n)]), -1);
    std::vector<int> inDeg(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int k = edgeOff[size_t(i)]; k < edgeOff[size_t(i) + 1]; ++k) {
            SdfPath const &in = desc.nodes[i].inputs[size_t(k - edgeOff[size_t(i)])];
            auto it = findDescIdx(in);
            if (it == descIdxByPath.end()) {
                result.errors.push_back(
                    std::string("UsdGenCompiler: usdGen:input target '") + in.GetText() +
                    "' of '" + desc.nodes[i].path.GetText() + "' is not in the graph");
                return;
            }
            edgeIds[size_t(k)] = it->second;
            ++inDeg[i];
        }
    }
    // Consumers in CSR form (E-6): one allocation instead of n per-node
    // vectors. consumersOf[i] = [begin,end) into consumerIds.
    std::vector<int> consumersOf(size_t(n) + 1, 0);
    for (int id : edgeIds) ++consumersOf[size_t(id) + 1];
    for (int i = 0; i < n; ++i) consumersOf[size_t(i) + 1] += consumersOf[size_t(i)];
    std::vector<int> consumerIds;
    consumerIds.resize(size_t(consumersOf[size_t(n)]));
    {
        std::vector<int> cursor(consumersOf.begin(), consumersOf.begin() + n);
        for (int i = 0; i < n; ++i)
            for (int k = edgeOff[size_t(i)]; k < edgeOff[size_t(i) + 1]; ++k)
                consumerIds[size_t(cursor[edgeIds[size_t(k)]]++)] = i;
    }

    // Kahn with namespace-order tie-break: among the ready nodes, always
    // pick the lowest desc index (02 §3.2 dense ids). Min-heap, not std::set:
    // identical order, zero per-node allocations (E-6: 200 tree-node mallocs).
    std::vector<int> order;
    order.reserve(n);
    {
        std::vector<int> ready;
        ready.reserve(n);
        for (int i = 0; i < n; ++i) if (inDeg[i] == 0) ready.push_back(i);
        std::make_heap(ready.begin(), ready.end(), std::greater<int>());
        while (!ready.empty()) {
            std::pop_heap(ready.begin(), ready.end(), std::greater<int>());
            int const i = ready.back();
            ready.pop_back();
            order.push_back(i);
            for (int k = consumersOf[size_t(i)]; k < consumersOf[size_t(i) + 1]; ++k) {
                int const j = consumerIds[size_t(k)];
                if (--inDeg[j] == 0) {
                    ready.push_back(j);
                    std::push_heap(ready.begin(), ready.end(), std::greater<int>());
                }
            }
        }
    }
    if (static_cast<int>(order.size()) != n) {
        for (int i = 0; i < n; ++i) {
            if (inDeg[i] > 0) {
                result.errors.push_back(
                    std::string("UsdGenCompiler: input cycle involving '") +
                    desc.nodes[i].path.GetText() + "'");
                break;
            }
        }
        return;
    }
    // Dense node ids: topo position == id.
    std::vector<int> topoOfDesc(n, -1);
    for (int pos = 0; pos < n; ++pos) topoOfDesc[order[pos]] = pos;

    // Allocate auto Clump levels in compiled topological order. Reserve every
    // emitted level, including disabled nodes, and reject pinned collisions
    // before moving any reusable graph state.
    std::vector<int> clumpBase(size_t(n), -1);
    std::map<int, SdfPath> clumpOwner;
    // CurveSource can import an existing Clump quartet. Its level names are
    // already part of the graph namespace, so a following auto Clump must
    // allocate around them rather than replace those authored planes.
    for (UsdGenCurveSetDesc const &curves : desc.curveSets) {
        for (UsdGenAuthoredPlaneDesc const &plane : curves.authoredPlanes) {
            std::string const name = plane.name.GetString();
            constexpr char prefix[] = "clumpId_";
            if (name.compare(0, sizeof(prefix) - 1, prefix) != 0) continue;
            std::string const suffix = name.substr(sizeof(prefix) - 1);
            if (suffix.empty() || (suffix.size() > 1 && suffix.front() == '0')) continue;
            int level = 0;
            bool valid = true;
            for (char digit : suffix) {
                if (digit < '0' || digit > '9' ||
                    level > (std::numeric_limits<int>::max() - (digit - '0')) / 10) {
                    valid = false;
                    break;
                }
                level = level * 10 + (digit - '0');
            }
            if (valid) clumpOwner.emplace(level, curves.path);
        }
    }
    int nextAutoLevel = 0;
    for (int di : order) {
        UsdGenNodeDesc const &nd = desc.nodes[di];
        if (nd.type != tok::Clump()) continue;
        UsdGenParamView p;
        p.desc = &desc;
        p.node = &nd;
        int const levels = p.GetInt(tok::ClumpLevels(), 1);
        int const pinned = p.GetInt(tok::ClumpLevel(), -1);
        if (levels < 1 || levels > 4 || pinned < -1 ||
            (pinned >= 0 && pinned > std::numeric_limits<int>::max() - levels) ||
            nextAutoLevel > std::numeric_limits<int>::max() - levels) {
            result.errors.push_back("UsdGenCompiler: invalid Clump level range on '" +
                                    nd.path.GetString() + "'");
            return;
        }
        int base = pinned >= 0 ? pinned : nextAutoLevel;
        if (pinned < 0) {
            // Find the first contiguous run of free level names. Each node's
            // multilevel output remains consecutive even around imports.
            for (;;) {
                if (base > std::numeric_limits<int>::max() - levels) {
                    result.errors.push_back("UsdGenCompiler: no free Clump level range on '" +
                                            nd.path.GetString() + "'");
                    return;
                }
                bool occupied = false;
                for (int offset = 0; offset < levels; ++offset)
                    occupied = occupied || clumpOwner.count(base + offset) != 0;
                if (!occupied) break;
                ++base;
            }
        }
        clumpBase[size_t(di)] = base;
        nextAutoLevel = std::max(nextAutoLevel, base + levels);
        for (int offset = 0; offset < levels; ++offset) {
            int const level = base + offset;
            auto const inserted = clumpOwner.emplace(level, nd.path);
            if (!inserted.second) {
                result.errors.push_back("UsdGenCompiler: clumpId_" +
                    std::to_string(level) + " is emitted by both '" +
                    inserted.first->second.GetString() + "' and '" +
                    nd.path.GetString() + "'");
                return;
            }
        }
    }

    UsdGenNodeId const terminalId =
        static_cast<UsdGenNodeId>(topoOfDesc[termIt->second]);

    // Strict descendants as dense-id bitmaps propagated in REVERSE
    // topological order — O(edges * words) word-ORs. This replaces a
    // per-node std::set BFS whose O(n^2) allocations were the dominant
    // term of the E-6 recompile budget on a 200-node chain.
    // M-4: the DIRECT consumer is a descendant too.
    std::vector<std::vector<UsdGenNodeId>> descOf;
    descOf.resize(size_t(n));
    {
        int const words = (n + 63) / 64;
        std::vector<uint64_t> bits(size_t(n) * words, 0);
        // Counts per row first, so every row allocates exactly once below.
        std::vector<int> rowCount(size_t(n), 0);
        for (int pos = n - 1; pos >= 0; --pos) {
            int const di = order[pos];
            uint64_t *dst = bits.data() + size_t(pos) * words;
            for (int k = consumersOf[size_t(di)]; k < consumersOf[size_t(di) + 1]; ++k) {
                int const c = consumerIds[size_t(k)];
                int const cp = topoOfDesc[c];   // cp > pos (topological)
                uint64_t const *src = bits.data() + size_t(cp) * words;
                // OR + popcount in one row pass (E-6): the count feeds the
                // exact reserve below, so rows never regrow.
                int add = 0;
                for (int w = 0; w < words; ++w) {
                    uint64_t const before = dst[w];
                    uint64_t const after = before | src[w];
                    dst[w] = after;
                    add += _PopCount64(after & ~before);
                }
                uint64_t const bit = 1ull << (cp & 63);
                if (!(dst[cp >> 6] & bit)) { dst[cp >> 6] |= bit; ++add; }
                rowCount[size_t(pos)] += add;
            }
        }
        for (int pos = 0; pos < n; ++pos) {
            uint64_t const *row = bits.data() + size_t(pos) * words;
            auto &v = descOf[size_t(pos)];
            v.reserve(size_t(rowCount[size_t(pos)]));
            for (int w = 0; w < words; ++w) {
                uint64_t b = row[w];
                while (b) {
                    v.push_back(static_cast<UsdGenNodeId>(
                        w * 64 + _CountTrailingZeros64(b)));
                    b &= b - 1;   // ascending bit order == dense-id order
                }
            }
        }
    }
    // Reuse matching (E-6): index the previous graph's nodes by path WITHOUT
    // copying anything (no Clone, no buffer/vector copies). Stable nodes are
    // moved whole — op, capture, buffer, chunks, dirty bytes, epochs — into
    // the new graph; only their desc pointers and digests refresh. This
    // replaces the OldNode snapshot whose per-node Clone + deep copies were
    // the dominant term of the chain-200 append budget.
    // Validate factories and their geometry dataflow contracts before moving
    // any retained runtime state. A failed incremental compile must leave the
    // old graph intact, just like Compile. In particular, no executor may
    // silently select the first edge of an unsupported fan-in.
    struct StaticOperatorContract {
        size_t geometryInputArity = 0;
        size_t referenceInputArity = 0;
        UsdGenRole role = UsdGenRole::Curves;
    };
    std::vector<StaticOperatorContract> operatorContracts(desc.nodes.size());
    for (size_t nodeIndex = 0; nodeIndex != desc.nodes.size(); ++nodeIndex) {
        UsdGenNodeDesc const& node = desc.nodes[nodeIndex];
        if (!UsdGenOpRegistry::Get().HasKernel(node.type)) {
            result.errors.push_back("UsdGenCompiler: no kernel registered for '" +
                node.type.GetString() + "' (prim " + node.path.GetString() + ")");
            return;
        }
        StaticOperatorContract& contract = operatorContracts[nodeIndex];
        if (!UsdGenOpRegistry::Get().GetOperatorContract(
                node.type, &contract.geometryInputArity,
                &contract.referenceInputArity, &contract.role)) {
            result.errors.push_back("UsdGenCompiler: no input contract registered for '" +
                node.type.GetString() + "' (prim " + node.path.GetString() + ")");
            return;
        }
        if (node.inputs.size() != contract.geometryInputArity) {
            std::ostringstream message;
            message << "UsdGenCompiler: operator '" << node.path.GetText()
                    << "' (type '" << node.type.GetText() << "') requires exactly "
                    << contract.geometryInputArity << " geometry input"
                    << (contract.geometryInputArity == 1 ? "" : "s") << "; found "
                    << node.inputs.size();
            result.errors.push_back(message.str());
            return;
        }
        // WidthBlend is the first admitted binary value fan-in and therefore
        // has a deliberately closed descriptor ABI.  Validate it before any
        // incremental state moves; disabled operators bypass runtime Bind(),
        // so this contract cannot live only in the operator implementation.
        if (node.type == TfToken("UsdGenWidthBlend")) {
            bool paramsOk = true;
            for (UsdGenParamValue const &param : node.params)
                paramsOk = paramsOk && param.name == TfToken("widthBlend:weight");
            double weight = 1.0;
            {
                UsdGenParamView const view{&desc, &node};
                weight = view.GetDouble(TfToken("widthBlend:weight"), 1.0);
            }
            if (!node.enabled || !std::isfinite(weight) || weight < 0.0 ||
                weight > 1.0 || !node.mode.IsEmpty() ||
                !paramsOk || !node.ramps.empty() ||
                !node.expressionBindings.empty() || !node.references.empty() ||
                !node.curves.empty() || !node.surfaces.empty() ||
                !node.maps.empty() || !node.mapBindings.empty()) {
                result.errors.push_back(
                    "UsdGenCompiler: WidthBlend requires enabled=true, a finite "
                    "usdGen:widthBlend:weight in [0,1], and no auxiliary inputs");
                return;
            }
            if (node.inputs[0] == node.inputs[1]) {
                result.errors.push_back(
                    "UsdGenCompiler: WidthBlend requires two distinct ordered geometry inputs");
                return;
            }
        }
    }

    // Fully validate and materialize external descriptor values BEFORE an
    // incremental build moves the old graph's nodes/descriptor.  A malformed
    // reference/map edit must leave the prior graph usable.
    std::vector<UsdGenResolvedReferenceValue> resolvedReferenceValues;
    std::vector<UsdGenResolvedMapValue> resolvedMapValues;
    std::unordered_map<SdfPath, uint32_t, SdfPath::Hash> referenceByPath;
    std::unordered_map<SdfPath, uint32_t, SdfPath::Hash> mapByPath;
    for (UsdGenCurveSetDesc const &curves : desc.curveSets) {
        if (curves.role != UsdGenRole::Reference) continue;
        if (referenceByPath.count(curves.path)) {
            result.errors.push_back("UsdGenCompiler: duplicate reference curve set '" +
                                    curves.path.GetString() + "'");
            return;
        }
        UsdGenResolvedReferenceValue value;
        std::string error;
        if (!BuildReferenceValue(curves, &value, &error)) {
            result.errors.push_back("UsdGenCompiler: " + error);
            return;
        }
        referenceByPath.emplace(curves.path,
                                static_cast<uint32_t>(resolvedReferenceValues.size()));
        resolvedReferenceValues.push_back(std::move(value));
    }
    for (UsdGenMapDesc const &map : desc.maps) {
        if (map.path.IsEmpty() || mapByPath.count(map.path)) {
            result.errors.push_back("UsdGenCompiler: invalid or duplicate map descriptor '" +
                                    map.path.GetString() + "'");
            return;
        }
        UsdGenResolvedMapValue value;
        value.path = map.path;
        value.type = map.type;
        value.resolvedAssetPath = map.resolvedAssetPath;
        value.textureGeneration = map.textureGeneration;
        value.identity = MapValueIdentity(map);
        mapByPath.emplace(map.path, static_cast<uint32_t>(resolvedMapValues.size()));
        resolvedMapValues.push_back(std::move(value));
    }
    for (size_t nodeIndex = 0; nodeIndex != desc.nodes.size(); ++nodeIndex) {
        UsdGenNodeDesc const& node = desc.nodes[nodeIndex];
        StaticOperatorContract const& contract = operatorContracts[nodeIndex];
        bool const hasExternalInputs = !node.references.empty() ||
            !node.curves.empty() || !node.maps.empty() ||
            !node.mapBindings.empty();
        if (!hasExternalInputs) {
            // A reference-role producer cannot silently omit its declared
            // slots.  This is the only empty-input rule; every nonempty
            // reference, curve, or map path continues through the complete
            // duplicate/target/typed-binding validation below.
            if (contract.role == UsdGenRole::Reference &&
                contract.referenceInputArity != 0) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' has 0 resolved reference values but its operator declares " +
                    std::to_string(contract.referenceInputArity) + " reference slots");
                return;
            }
            continue;
        }
        std::set<SdfPath> references;
        auto addReference = [&](SdfPath const &path) {
            if (!references.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' declares duplicate reference value '" + path.GetString() + "'");
                return false;
            }
            if (!referenceByPath.count(path)) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' references unresolved reference value '" + path.GetString() + "'");
                return false;
            }
            return true;
        };
        for (SdfPath const &path : node.references)
            if (!addReference(path)) return;
        for (SdfPath const &path : node.curves)
            if (referenceByPath.count(path) && !addReference(path)) return;
        // Preserve the reference-free incremental fast path: factory
        // existence and the complete version-resolved static contract were
        // already checked above.  Do not allocate a probe per node here:
        // role/reference-slot metadata came from the registration-time
        // probe, exactly like geometry input arity.
        if (!references.empty()) {
            if (references.size() != contract.referenceInputArity) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' has " + std::to_string(references.size()) +
                    " resolved reference values but its operator declares " +
                    std::to_string(contract.referenceInputArity) + " reference slots");
                return;
            }
        } else if (contract.role == UsdGenRole::Reference &&
                   contract.referenceInputArity != 0) {
            result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                "' has 0 resolved reference values but its operator declares " +
                std::to_string(contract.referenceInputArity) + " reference slots");
            return;
        }
        std::vector<UsdGenMapBindingDesc> bindings;
        std::string bindingError;
        if (!EffectiveMapBindings(node, &bindings, &bindingError)) {
            result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                "' has ambiguous map bindings: " + bindingError);
            return;
        }
        std::set<SdfPath> maps;
        for (UsdGenMapBindingDesc const &binding : bindings) {
            SdfPath const &path = binding.map;
            if (!ValidateMapBinding(binding, &bindingError)) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' has ambiguous map binding: " + bindingError);
                return;
            }
            if (!maps.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' declares duplicate map binding '" + path.GetString() + "'");
                return;
            }
            if (!mapByPath.count(path)) {
                result.errors.push_back("UsdGenCompiler: node '" + node.path.GetString() +
                    "' references unresolved map value '" + path.GetString() + "'");
                return;
            }
        }
    }
    std::vector<std::unique_ptr<UsdGenCompiledNode>> oldNodes;
    // descIdx → node index over the OLD graph (E-6): the merge and the node
    // loop map by position, with a linear path scan only for reorder misses
    // (rare) — no index vector, no sort, no per-lookup searches on the hit
    // path. SdfPath::operator== is an integer compare (no strings).
    std::vector<size_t> nodeByDesc;
    if (reuse && reuse != out) {
        // Defensive: Recompile always passes out as reuse; a foreign graph
        // is indexed read-only (no moves) — nodes rebuild fully below.
    } else if (reuse) {
        oldNodes = std::move(out->_nodes);
    }
    if (!oldNodes.empty() && out->_desc) {
        nodeByDesc.assign(out->_desc->nodes.size(), SIZE_MAX);
        for (size_t i = 0; i < oldNodes.size(); ++i)
            if (oldNodes[i] && oldNodes[i]->descIdx >= 0 &&
                size_t(oldNodes[i]->descIdx) < nodeByDesc.size())
                nodeByDesc[size_t(oldNodes[i]->descIdx)] = i;
    }
    // Desc copy elision (E-6): node entries byte-identical to the previous
    // graph's copy move instead of copying — a 201-entry copy is ~600 small
    // allocs (vectors + VtValue holders per param). Scalars always copy.
    // Entry identity is by path; equality is field-wise (no allocs on hit).
    // Correctness: nodes read their desc entry read-only through desc/paramView.
    auto sameNodeDesc = [](UsdGenNodeDesc const &a, UsdGenNodeDesc const &b) {
        // Until expression program identity participates in incremental
        // digests, copy connected descriptors instead of reusing stale source
        // bindings/literals. This is conservative, not a semantic fallback.
        if (!a.expressionBindings.empty() || !b.expressionBindings.empty()) return false;
        if (a.path != b.path || a.type != b.type || a.mode != b.mode ||
            a.enabled != b.enabled ||
            a.seed != b.seed || a.inputs != b.inputs ||
            a.references != b.references || a.curves != b.curves ||
            a.surfaces != b.surfaces || a.maps != b.maps ||
            a.mapBindings.size() != b.mapBindings.size() ||
            a.params.size() != b.params.size() || a.ramps.size() != b.ramps.size())
            return false;
        for (size_t i = 0; i < a.mapBindings.size(); ++i) {
            if (a.mapBindings[i].map != b.mapBindings[i].map ||
                a.mapBindings[i].relationship != b.mapBindings[i].relationship)
                return false;
        }
        for (size_t i = 0; i < a.params.size(); ++i) {
            if (a.params[i].name != b.params[i].name ||
                a.params[i].animated != b.params[i].animated ||
                !(a.params[i].value == b.params[i].value))
                return false;
        }
        for (size_t i = 0; i < a.ramps.size(); ++i) {
            if (a.ramps[i].knots != b.ramps[i].knots ||
                a.ramps[i].positions != b.ramps[i].positions ||
                a.ramps[i].colors != b.ramps[i].colors ||
                a.ramps[i].interpolation != b.ramps[i].interpolation)
                return false;
        }
        return true;
    };
    UsdGenGraphDesc *oldDescPtr = (reuse && out->_desc) ? out->_desc.get() : nullptr;
    std::vector<char> descChanged(desc.nodes.size(), 1);
    // oldNodeForNewDesc[di] = old node index, or -1 (recorded by the merge,
    // so the node loop needs no per-node searches on the hit path).
    std::vector<int> oldNodeForNewDesc(desc.nodes.size(), -1);
    // oldNodeAtPath[di] = the path-matched old node index even when the desc
    // entry changed (-1 when the path is new). A rebuilt node continues that
    // node's buffer stamps; see the rebuild branch of the node loop.
    std::vector<int> oldNodeAtPath(desc.nodes.size(), -1);
    // entries copy from the input. Entry ORDER follows the input (S26).
    // Identity is positional (nodeByDesc) with a linear-scan fallback.
    static_assert(sizeof(UsdGenGraphDesc) ==
                      512 + sizeof(std::vector<UsdGenGeometryDesc>) + sizeof(UsdGenPreviewDesc),
        "UsdGenGraphDesc changed size: update the Recompile shell merge below");
    {
        auto fresh = std::make_unique<UsdGenGraphDesc>();
        fresh->description = desc.description;
        fresh->terminal = desc.terminal;
        fresh->curveSets = desc.curveSets;
        fresh->surfaces = desc.surfaces;
        fresh->maps = desc.maps;
        fresh->expressions = desc.expressions;
        fresh->geometries = desc.geometries;
        fresh->validationErrors = desc.validationErrors;
        fresh->executionBackend = desc.executionBackend;
        fresh->look = desc.look;
        fresh->preview = desc.preview;
        fresh->xformMatrix = desc.xformMatrix;
        fresh->purpose = desc.purpose;
        fresh->visibility = desc.visibility;
        fresh->materialPath = desc.materialPath;
        fresh->defaultWidth = desc.defaultWidth;
        fresh->tileTarget = desc.tileTarget;
        fresh->curveBasis = desc.curveBasis;
        fresh->time = desc.time;
        fresh->timeCodesPerSecond = desc.timeCodesPerSecond;
        fresh->nodes.reserve(desc.nodes.size());
        // Per-input-order entry change flags for digest-change propagation
        // below (plan §3.5: only changed nodes and their descendants pay).
        if (oldDescPtr) {
            size_t const oldSize = oldDescPtr->nodes.size();
            for (size_t i = 0; i < desc.nodes.size(); ++i) {
                // Positional pre-check (E-6): appends/param-edits keep
                // namespace order — one pointer-fast SdfPath == instead of
                // a binary search on the hit path.
                size_t o = oldSize;  // sentinel: linear-scan fallback below
                size_t on = oldSize; // old NODE index for oldNodeForNewDesc
                if (i < oldSize && oldDescPtr->nodes[i].path == desc.nodes[i].path) {
                    o = i;
                    if (i < nodeByDesc.size() && nodeByDesc[i] != SIZE_MAX) on = nodeByDesc[i];
                } else {
                    // Reorder miss (rare): linear path scan with integer
                    // SdfPath compares; node index rides nodeByDesc.
                    for (size_t j = 0; j < oldSize; ++j) {
                        if (oldDescPtr->nodes[j].path == desc.nodes[i].path) {
                            o = j;
                            if (j < nodeByDesc.size() && nodeByDesc[j] != SIZE_MAX)
                                on = nodeByDesc[j];
                            break;
                        }
                    }
                }
                if (on < oldSize) oldNodeAtPath[i] = int(on);
                if (o < oldSize && sameNodeDesc(oldDescPtr->nodes[o], desc.nodes[i])) {
                    fresh->nodes.push_back(std::move(oldDescPtr->nodes[o]));
                    descChanged[i] = 0;
                    if (on < oldSize) oldNodeForNewDesc[i] = int(on);
                } else {
                    fresh->nodes.push_back(desc.nodes[i]);
                }
            }
        } else {
            fresh->nodes = desc.nodes;
        }
        out->_desc = std::move(fresh);
    }

    // External values were fully validated before any reusable graph state
    // moved.  Transfer the immutable candidates only after the new descriptor
    // shell has been installed.
    out->_referenceValues = std::move(resolvedReferenceValues);
    out->_mapValues = std::move(resolvedMapValues);
    out->_nodes.clear();
    out->_nodes.resize(n);
    out->_nodeByPath.clear();
    out->_terminal = terminalId;
    out->_chunkSize = kUsdGenDefaultChunkSize;   // USDGEN_CHUNK_SIZE (S23)
    out->_tileTarget = desc.tileTarget;
    std::vector<UsdGenEpoch> digests(n);
    std::vector<char> digestChanged(n, 0);
    for (int pos = 0; pos < n; ++pos) {
        int const di = order[pos];
        UsdGenNodeDesc const &nd = desc.nodes[di];
        bool const clumpSlotsChanged = nd.type == tok::Clump() &&
            oldNodeForNewDesc[di] >= 0 &&
            oldNodes[size_t(oldNodeForNewDesc[di])] &&
            (oldNodes[size_t(oldNodeForNewDesc[di])]->outputPrimvars.empty() ||
             oldNodes[size_t(oldNodeForNewDesc[di])]->outputPrimvars.front() !=
                 TfToken("clumpId_" + std::to_string(clumpBase[size_t(di)])) ||
             oldNodes[size_t(oldNodeForNewDesc[di])]->outputPrimvars.size() !=
                 size_t(4 * UsdGenParamView{out->_desc.get(), &out->_desc->nodes[di]}
                              .GetInt(tok::ClumpLevels(), 1)));
        // Change propagation (plan §3.5): a node whose desc entry is
        // byte-identical AND whose inputs' digests all survived keeps its
        // old digest verbatim — no FNV, no path vectors, no GetHash. Only
        // changed nodes and downstream consumers recompute.
        // Inputs come pre-resolved from the fused edge pass (E-6): no
        // per-node binary searches for the change check or the refill.
        int const e0 = edgeOff[size_t(di)], e1 = edgeOff[size_t(di) + 1];
        bool inputChanged = false;
        for (int k = e0; !inputChanged && k < e1; ++k)
            inputChanged = digestChanged[topoOfDesc[edgeIds[size_t(k)]]] != 0;
        if (!oldNodes.empty() && !descChanged[di] && !inputChanged &&
            !clumpSlotsChanged &&
            oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode &oldS = *oldNodes[size_t(oldNodeForNewDesc[di])];
            if (oldS.type == nd.type) {
                // Verbatim reuse: refresh desc-owned fields and rebuild the
                // dense-id edge list in retained capacity (no alloc when the
                // fan-in is stable). sameNodeDesc() above already proved all
                // value-class inputs, enabled state and ramps are
                // unchanged, so retain the existing value digest instead of
                // rescanning every operator parameter on every reused node.
                auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                moved->id = static_cast<UsdGenNodeId>(pos);
                moved->desc = &out->_desc->nodes[di];
                moved->descIdx = di;
                moved->enabled = nd.enabled;
                moved->paramView.desc = out->_desc.get();
                moved->paramView.node = &out->_desc->nodes[di];
                moved->paramView.expressions = &moved->expressions;
                moved->inputs.clear();
                for (int k = e0; k < e1; ++k)
                    moved->inputs.push_back(static_cast<UsdGenNodeId>(topoOfDesc[edgeIds[size_t(k)]]));
                // Preserve authored edge order.  Unary operators do not give
                // that order semantic meaning, but ordered fan-in operators
                // (UsdGenWidthBlend) use inputs[0] as left and inputs[1] as
                // right and must retain it through recompile.
                moved->input = moved->inputs.empty() ? kUsdGenInvalidNode : moved->inputs.front();
                moved->descendants = std::move(descOf[size_t(pos)]);
                digests[pos] = moved->structuralDigest;
                out->_nodeByPath[nd.path] = moved->id;
                out->_nodes[pos] = std::move(moved);
                continue;
            }
        }
        // E-6 fast path: a path-matched old node of the same type and
        // version reuses its OWN op for the stability check — no op Create,
        // no vector building, no routing copy. On stability the old node
        // moves whole with only desc pointers/digests refreshed; the
        // expensive per-node allocs (node, inputs, refs, routing table) and
        // their matching destructions never happen.
        if (!oldNodes.empty() && oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode &old0 = *oldNodes[size_t(oldNodeForNewDesc[di])];
            if (old0.type == nd.type) {
                TypeClassTable const &tbl0 = TypeClassification(*old0.op);
                std::vector<SdfPath> inputPaths0 = nd.inputs;
                if (!old0.op->GeometryInputsOrdered())
                    std::sort(inputPaths0.begin(), inputPaths0.end());
                std::vector<SdfPath> refPaths0 = nd.references;
                std::vector<UsdGenMapBindingDesc> bindings0;
                std::string bindingError0;
                if (!EffectiveMapBindings(nd, &bindings0, &bindingError0)) {
                    result.errors.push_back("UsdGenCompiler: node '" + nd.path.GetString() +
                        "' has ambiguous map bindings: " + bindingError0);
                    return;
                }
                for (UsdGenMapBindingDesc const &binding : bindings0)
                    refPaths0.push_back(binding.map);
                refPaths0.insert(refPaths0.end(), nd.surfaces.begin(), nd.surfaces.end());
                std::sort(refPaths0.begin(), refPaths0.end());
                std::vector<std::pair<SdfPath, UsdGenEpoch>> childDigests0;
                for (SdfPath const &p : inputPaths0) {
                    auto const it = findDescIdx(p);
                    if (it != descIdxByPath.end())
                        childDigests0.emplace_back(p, digests[topoOfDesc[it->second]]);
                }
                UsdGenEpoch const dg0 = ComputeNodeDigest(
                    nd, nd.type,
                    TfSpan<const TfToken>(tbl0.digestParams.data(), tbl0.digestParams.size()),
                    inputPaths0, refPaths0, bindings0, childDigests0);
                if (dg0 == old0.structuralDigest && !clumpSlotsChanged &&
                    old0.topoFx == old0.op->TopologyEffect() &&
                    old0.role == old0.op->Role()) {
                    // Stable: move whole, refresh desc-owned fields only.
                    // inputs/descendants-aside everything rides along:
                    // op, capture, buffer, chunks, dirty bytes, epochs,
                    // lastParamDigest (scheduler skip state survives).
                    auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                    moved->id = static_cast<UsdGenNodeId>(pos);
                    moved->desc = &out->_desc->nodes[di];
                    moved->descIdx = di;
                    moved->structuralDigest = dg0;
                    moved->enabled = nd.enabled;
                    moved->paramView.desc = out->_desc.get();
                    moved->paramView.node = &out->_desc->nodes[di];
                    moved->paramView.expressions = &moved->expressions;
                    moved->descendants = std::move(descOf[size_t(pos)]);
                    {   // value digest refreshes (cheap, no allocs).
                        uint64_t vh = 1469598103934665603ULL;
                        auto feed = [&vh](uint64_t v) { vh ^= v; vh *= 0x100000001b3ULL; };
                        feed(static_cast<uint64_t>(nd.enabled));
                        for (auto const &param : nd.params) {
                            bool inValue = false, inTopo = false;
                            for (TfToken const &t : moved->op->ValueParameters())
                                inValue = inValue || (t == param.name);
                            for (TfToken const &t : moved->op->TopologyParameters())
                                inTopo = inTopo || (t == param.name);
                            if (inValue || (!inTopo && param.name != tok::Seed())) feed(param.value.GetHash());
                        }
                        moved->paramValueDigest = vh;
                    }
                    digests[pos] = dg0;
                    digestChanged[pos] = 0;
                    out->_nodeByPath[nd.path] = moved->id;
                    out->_nodes[pos] = std::move(moved);
                    continue;
                }
            }
        }

        auto node = std::make_unique<UsdGenCompiledNode>();
        node->id = static_cast<UsdGenNodeId>(pos);
        node->type = nd.type;
        node->desc = &out->_desc->nodes[di];
        node->descIdx = di;

        std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(nd.type);
        if (!op) {
            result.errors.push_back(
                "UsdGenCompiler: no kernel registered for '" +
                nd.type.GetString() + "' (prim " + nd.path.GetText() + ")");
            return;
        }
        node->op = std::move(op);
        // Parameter view over the graph's desc copy. Set before the slots
        // are bound: an operator may name its planes from its parameters.
        node->paramView.desc = out->_desc.get();
        node->paramView.node = &out->_desc->nodes[di];
        node->paramView.expressions = &node->expressions;
        if (nd.type == tok::Clump())
            static_cast<UsdGenClumpOp *>(node->op.get())->SetLevelBase(clumpBase[size_t(di)]);
        node->op->Configure(node->paramView);
        if (!BindExtraPlaneSlots(*node->op, node.get(), &result)) return;

        node->topoFx = node->op->TopologyEffect();
        node->role = node->op->Role();
        node->enabled = nd.enabled;

        // Input edges (dense ids).
        for (SdfPath const &in : nd.inputs) {
            auto const it = findDescIdx(in);
            if (it != descIdxByPath.end())
                node->inputs.push_back(static_cast<UsdGenNodeId>(topoOfDesc[it->second]));
        }
        // Keep dense IDs in the same authored order as UsdGenNodeDesc::inputs;
        // WidthBlend's ordered operands depend on this correspondence.
        node->input = node->inputs.empty() ? kUsdGenInvalidNode : node->inputs.front();

        // Relationship targets.
        node->curveRefs = nd.curves;
        std::vector<UsdGenMapBindingDesc> bindings;
        std::string bindingError;
        if (!EffectiveMapBindings(nd, &bindings, &bindingError)) {
            result.errors.push_back("UsdGenCompiler: node '" + nd.path.GetString() +
                "' has ambiguous map bindings: " + bindingError);
            return;
        }
        node->mapBindingRefs = bindings;
        node->mapRefs.clear();
        node->mapRefs.reserve(bindings.size());
        for (UsdGenMapBindingDesc const &binding : bindings)
            node->mapRefs.push_back(binding.map);
        if (!nd.surfaces.empty()) {
            node->hasSurface = true;
            node->surface = 0;   // filled below when desc.surfaces is indexed
        }

        // Strict descendants, ascending dense-id order (bitmap pass above).
        node->descendants = std::move(descOf[size_t(pos)]);

        // Structural digest (ADR §4.2.1 term list).
        std::vector<SdfPath> inputPaths = nd.inputs;
        if (!node->op->GeometryInputsOrdered())
            std::sort(inputPaths.begin(), inputPaths.end());
        std::vector<SdfPath> refPaths = nd.references;
        for (UsdGenMapBindingDesc const &binding : bindings)
            refPaths.push_back(binding.map);
        refPaths.insert(refPaths.end(), nd.surfaces.begin(), nd.surfaces.end());
        std::sort(refPaths.begin(), refPaths.end());
        std::vector<std::pair<SdfPath, UsdGenEpoch>> childDigests;
        for (SdfPath const &p : inputPaths) {
            auto const it = findDescIdx(p);
            if (it != descIdxByPath.end())
                childDigests.emplace_back(p, digests[topoOfDesc[it->second]]);
        }
        // Structural digest: only §6.1-class (recompile) parameter values
        // are digest terms — capture- and topology-class edits must NOT
        // move the digest (03 §3.3/§3.6). The §6.1 ∩ TopologyParameters()
        // filter and the routing table are properties of the TYPE and are
        // built once per type, not once per node (E-6 budget).
        TypeClassTable const &tbl = TypeClassification(*node->op);
        digests[pos] = ComputeNodeDigest(
            nd, node->type,
            TfSpan<const TfToken>(tbl.digestParams.data(), tbl.digestParams.size()),
            inputPaths, refPaths, bindings, childDigests);
        node->structuralDigest = digests[pos];

        // Value-class digest (skip-signature term, 03 §3.2): recompile keeps
        // this stable unless a value-class param or the enabled flag moved.
        // Unknown (unclassified) params feed it too — a value-class edit is
        // the safe superset for an unlisted property.
        {
            uint64_t vh = 1469598103934665603ULL;
            auto feed = [&vh](uint64_t v) { vh ^= v; vh *= 0x100000001b3ULL; };
            feed(static_cast<uint64_t>(nd.enabled));
            for (auto const &param : nd.params) {
                bool inValue = false, inTopo = false;
                for (TfToken const &t : node->op->ValueParameters())
                    inValue = inValue || (t == param.name);
                for (TfToken const &t : node->op->TopologyParameters())
                    inTopo = inTopo || (t == param.name);
                if (inValue || (!inTopo && param.name != tok::Seed())) feed(param.value.GetHash());
            }
            node->paramValueDigest = vh;
        }

        // Parameter routing table (02 §6): every C1 parameter of this type
        // gets exactly one entry — copied from the cached type table.
        node->paramRouting = tbl.routing;

        // E-6 reuse: a digest-STABLE node moves whole — op, capture, buffer,
        // chunks, dirty bytes, epochs — out of the previous graph. Only the
        // desc pointers and digests refresh. A null capture (never-run node)
        // is NOT a rebuild: the capture installs lazily on the next commit.
        if (!oldNodes.empty() && oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode const &oldRef = *oldNodes[size_t(oldNodeForNewDesc[di])];
            bool const stable = !clumpSlotsChanged &&
                oldRef.structuralDigest == node->structuralDigest &&
                oldRef.type == node->type &&
                oldRef.topoFx == node->topoFx &&
                oldRef.role == node->role;
            if (stable) {
                // Move the whole node; refresh only what the new desc
                // owns. Op, capture, buffer, chunks, dirty bytes,
                // capture epoch, topologySeq and eval signature ride
                // along untouched — including lastParamDigest, so the
                // scheduler's skip logic survives the recompile.
                auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                moved->id = node->id;
                moved->desc = node->desc;
                moved->descIdx = node->descIdx;
                moved->structuralDigest = node->structuralDigest;
                moved->paramValueDigest = node->paramValueDigest;
                moved->enabled = node->enabled;
                moved->input = node->input;
                moved->inputs = std::move(node->inputs);
                moved->descendants = std::move(node->descendants);
                moved->curveRefs = std::move(node->curveRefs);
                moved->mapRefs = std::move(node->mapRefs);
                moved->mapBindingRefs = std::move(node->mapBindingRefs);
                moved->hasSurface = node->hasSurface;
                moved->surface = node->surface;
                moved->paramView = node->paramView;
                moved->paramView.expressions = &moved->expressions;
                moved->paramRouting = std::move(node->paramRouting);
                moved->outputPrimvars = std::move(node->outputPrimvars);
                moved->inputPrimvars = std::move(node->inputPrimvars);
                node = std::move(moved);
                digestChanged[pos] = 0;
            } else {
                result.rebuilt.push_back(node->id);
                digestChanged[pos] = 1;
            }
        } else {
            result.rebuilt.push_back(node->id);
            digestChanged[pos] = 1;
        }

        // A rebuilt node REPLACES its path-matched predecessor: its buffer
        // stamps continue the old sequence instead of restarting at zero.
        // Downstream captures (Clump, Curl, Expr, ...) key their validity on
        // the upstream (topologyVersion, valueVersion); a fresh node counting
        // 0 -> 1 again in one cook reproduces the very stamps they recorded,
        // so they keep serving the old geometry. Every expression-bound node
        // rebuilds on every recompile (sameNodeDesc), so a length-paint
        // repaint rebuilt Grow and the groom stayed unchanged until a
        // density edit changed the curve count under the whole chain.
        if (node->buffer.topologyVersion == 0 && node->buffer.valueVersion == 0 &&
            oldNodeAtPath[di] >= 0 && size_t(oldNodeAtPath[di]) < oldNodes.size() &&
            oldNodes[size_t(oldNodeAtPath[di])]) {
            UsdGenCompiledNode const &prior = *oldNodes[size_t(oldNodeAtPath[di])];
            node->topologySeq = prior.topologySeq;
            node->buffer.topologyVersion = prior.buffer.topologyVersion;
            node->buffer.valueVersion = prior.buffer.valueVersion;
        }
        out->_nodeByPath[nd.path] = node->id;
        out->_nodes[pos] = std::move(node);
    }

    // Bind external descriptor values after all node instances have been
    // created.  `guides` paths live in curveRefs in both builders; generic
    // `references` paths are accepted too.  Preserve authored order: duplicate
    // paths were rejected during the transactional preflight above.
    for (auto const &nodePtr : out->_nodes) {
        UsdGenCompiledNode &node = *nodePtr;
        node.referenceValues.clear();
        node.mapValues.clear();
        // Everything a connected expression samples (geoSampler/ptex inputs):
        // an edit there re-evaluates this node, like a curve or map edit.
        node.geometryRefs.clear();
        for (UsdGenExpressionBinding const &binding : node.desc->expressionBindings) {
            for (UsdGenExpressionDesc const &expression : out->_desc->expressions) {
                if (expression.path != binding.expression) continue;
                for (UsdGenExpressionInputDesc const &input : expression.inputs) {
                    auto add = [&node](SdfPath const &path) {
                        if (std::find(node.geometryRefs.begin(), node.geometryRefs.end(), path) ==
                            node.geometryRefs.end())
                            node.geometryRefs.push_back(path);
                    };
                    for (SdfPath const &path : input.targets) add(path);
                    for (SdfPath const &path : input.geometries) add(path);
                    for (SdfPath const &path : input.maps) add(path);
                }
            }
        }
        if (node.desc->references.empty() && node.curveRefs.empty() &&
            node.mapBindingRefs.empty()) {
            // Static empty-slot validation completed before any old runtime
            // state moved.  Avoid constructing empty ordered sets and making
            // a virtual ReferenceInputs() call for the common no-external-
            // inputs case; stale resolved handles were cleared above.
            continue;
        }
        std::set<SdfPath> seenReferences;
        auto addReference = [&](SdfPath const &path) {
            if (!seenReferences.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    node.desc->path.GetString() + "' declares duplicate reference value '" +
                    path.GetString() + "'");
                return false;
            }
            auto const it = referenceByPath.find(path);
            if (it == referenceByPath.end()) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    node.desc->path.GetString() + "' references unresolved reference value '" +
                    path.GetString() + "'");
                return false;
            }
            node.referenceValues.push_back(it->second);
            return true;
        };
        for (SdfPath const &path : node.desc->references)
            if (!addReference(path)) return;
        for (SdfPath const &path : node.curveRefs)
            if (referenceByPath.count(path) && !addReference(path)) return;

        TfSpan<const TfToken> const declaredReferences = node.op->ReferenceInputs();
        if ((!node.referenceValues.empty() || node.role == UsdGenRole::Reference) &&
            node.referenceValues.size() != declaredReferences.size()) {
            result.errors.push_back("UsdGenCompiler: node '" +
                node.desc->path.GetString() + "' has " +
                std::to_string(node.referenceValues.size()) +
                " resolved reference values but its operator declares " +
                std::to_string(declaredReferences.size()) + " reference slots");
            return;
        }
        std::set<SdfPath> seenMaps;
        for (UsdGenMapBindingDesc const &binding : node.mapBindingRefs) {
            SdfPath const &path = binding.map;
            if (!seenMaps.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    node.desc->path.GetString() + "' declares duplicate map binding '" +
                    path.GetString() + "'");
                return;
            }
            auto const it = mapByPath.find(path);
            if (it == mapByPath.end()) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    node.desc->path.GetString() + "' references unresolved map value '" +
                    path.GetString() + "'");
                return;
            }
            node.mapValues.push_back(it->second);
        }
    }

    // Surface id assignment: index desc.surfaces; bind node->surface.
    std::unordered_map<SdfPath, UsdGenSurfaceId, SdfPath::Hash> surfaceIdByPath;
    for (size_t s = 0; s < out->_desc->surfaces.size(); ++s) {
        surfaceIdByPath[out->_desc->surfaces[s].path] =
            static_cast<UsdGenSurfaceId>(s);
        out->_desc->surfaces[s].id = static_cast<UsdGenSurfaceId>(s);
    }
    for (auto &np : out->_nodes) {
        if (!np->hasSurface) continue;
        auto const it = surfaceIdByPath.find(np->desc->surfaces.front());
        np->surface = (it != surfaceIdByPath.end()) ? it->second : 0;
    }
    result.structuralDigest = ComputeNodeDigest(
        out->_desc->nodes[termIt->second], TfToken("UsdGenTerminal"), {},
        {desc.terminal}, {},
        {},
        std::vector<std::pair<SdfPath, UsdGenEpoch>>{
            {desc.terminal, digests[terminalId]}});
}


}  // namespace usdGen
