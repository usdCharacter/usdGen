// usdGen imaging — private shared decl for the two graph-desc builder TUs.
//
// Split of the S14 dedicated-field set out of usdGenGraphDescBuilder.cpp so
// the stage reference path (usdGenGraphDescBuilderStage.cpp) and the Hydra
// production path (usdGenGraphDescBuilder.cpp) share one definition without
// the B-2-fenced production TU needing any Usd header to reach it.
#ifndef USDGEN_IMAGING_GRAPH_DESC_SHARED_H
#define USDGEN_IMAGING_GRAPH_DESC_SHARED_H

#include "usdGenUsd/graphDesc.h"

#include "pxr/base/tf/token.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

namespace usdGenImaging {

/// UsdGenGeometryDesc::generation: FNV-1a over everything the descriptor
/// carries, so both builders give the same prim content the same identity.
inline uint64_t
UsdGenGeometryContentHash(usdGen::UsdGenGeometryDesc const &g)
{
    uint64_t h = 1469598103934665603ULL;
    auto bytes = [&h](void const *data, size_t count) {
        auto const *p = static_cast<unsigned char const *>(data);
        for (size_t i = 0; i < count; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
    };
    auto array = [&](auto const &a) {
        uint64_t const n = a.size();
        bytes(&n, sizeof(n));
        if (n) bytes(a.cdata(), a.size() * sizeof(a[0]));
    };
    uint8_t const kind = static_cast<uint8_t>(g.kind);
    bytes(&kind, 1);
    array(g.counts);
    array(g.indices);
    array(g.points);
    array(g.rest);
    array(g.normals);
    array(g.ids);
    array(g.subsetFaces);
    uint8_t const subset = g.isSubset ? 1 : 0;
    bytes(&subset, 1);
    bytes(g.worldMatrix.GetArray(), 16 * sizeof(double));
    return h;
}

/// Word-at-a-time FNV-style content hash for the generation counters below;
/// large meshes are hashed on every capture, so bytes go eight at a time.
class UsdGenContentHasher
{
public:
    void Bytes(void const *data, size_t count)
    {
        auto const *p = static_cast<unsigned char const *>(data);
        size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            uint64_t word;
            std::memcpy(&word, p + i, 8);
            Word(word);
        }
        uint64_t tail = 0;
        if (i < count) {
            std::memcpy(&tail, p + i, count - i);
            Word(tail ^ (uint64_t(count - i) << 56));
        }
    }
    void Word(uint64_t word)
    {
        _h = (_h ^ word) * 0x100000001b3ULL;
        _h ^= _h >> 29;
    }
    template <class A> void Array(A const &a)
    {
        Word(uint64_t(a.size()));
        if (!a.empty()) Bytes(a.cdata(), a.size() * sizeof(a[0]));
    }
    void Token(TfToken const &t) { Bytes(t.GetText(), t.size()); Word(t.size()); }
    void Token(usdGen::UsdGenToken t) { Bytes(t.GetText(), t.size()); Word(t.size()); }
    uint64_t Value() const { return _h ? _h : 1; }

private:
    uint64_t _h = 1469598103934665603ULL;
};

/// Sets UsdGenCurveSetDesc::curveGeneration and
/// UsdGenSurfaceDesc::surfaceGeneration from the content each carries, so a
/// capture digest keyed by them follows every points/topology/id edit (and
/// animation) while both builders give equal content the same value.
/// Provenance flags and the curves' world matrix are not terms: consumers
/// that read the matrix digest it themselves.
inline void
UsdGenFinalizeInputGenerations(usdGen::UsdGenGraphDesc *desc)
{
    for (usdGen::UsdGenCurveSetDesc &curves : desc->curveSets) {
        UsdGenContentHasher h;
        h.Word(uint64_t(curves.role));
        h.Token(curves.curveRole);
        h.Array(curves.curveVertexCounts);
        h.Array(curves.points);
        h.Array(curves.rest);
        h.Array(curves.widths);
        h.Token(curves.type);
        h.Token(curves.basis);
        h.Token(curves.wrap);
        h.Token(curves.widthsInterpolation);
        h.Array(curves.skinPrim);
        h.Array(curves.curveId);
        h.Array(curves.skinPrimUv);
        h.Array(curves.rootFrame);
        h.Bytes(curves.frozenEpoch.data(), curves.frozenEpoch.size());
        for (auto const &plane : curves.authoredPlanes) {
            h.Token(plane.name);
            h.Word((uint64_t(plane.type) << 16) | (uint64_t(plane.domain) << 8) | plane.arity);
            h.Array(plane.floatValues);
            h.Array(plane.intValues);
        }
        h.Word(curves.surfaceCage ? 1u : 0u);
        if (curves.surfaceCage) {
            usdGen::UsdGenSurfaceCagePayload const &cage =
                *curves.surfaceCage;
            h.Array(cage.ownerIds);
            h.Array(cage.ownerDensities);
            h.Array(cage.ownerSeeds);
            h.Array(cage.ownerCvCounts);
            h.Array(cage.ownerEdgeBias);
            h.Array(cage.ownerLengthProfileOffsets);
            h.Array(cage.ownerLengthProfile);
            h.Array(cage.normalizedT);
            h.Array(cage.triangles);
            h.Array(cage.triangleOwnerIndices);
            h.Array(cage.triangleRootCharts);
            h.Array(cage.ownerChartCentroids);
            h.Array(cage.ownerChartMeanRadii);
        }
        curves.curveGeneration = h.Value();
    }
    for (usdGen::UsdGenSurfaceDesc &surface : desc->surfaces) {
        UsdGenContentHasher h;
        h.Array(surface.faceVertexCounts);
        h.Array(surface.faceVertexIndices);
        h.Array(surface.restPoints);
        h.Array(surface.restNormals);
        h.Word(uint64_t(surface.restNormalDomain));
        h.Array(surface.points);
        for (auto const &sample : surface.samples) {
            h.Bytes(&sample.time, sizeof(sample.time));
            h.Array(sample.points);
        }
        h.Array(surface.velocities);
        h.Array(surface.uv);
        h.Array(surface.subsetFaces);
        h.Word(surface.isSubset ? 1u : 0u);
        h.Bytes(surface.worldMatrix.GetArray(), 16 * sizeof(double));
        surface.surfaceGeneration = h.Value();
    }
}

/// R15 (02 §2.20 rule 4): targets of one usdGen:surface that resolve to the
/// bound (front) target's mesh — sibling GeomSubsets, or the Mesh itself —
/// union into the front target. They leave `targets` and are returned as its
/// union members; targets on other meshes stay (only the front one binds).
/// `meshOf` maps a target to the Mesh it reads (itself for a Mesh, the
/// parent for a GeomSubset), or an empty path for anything else.
template <class MeshOf>
inline SdfPathVector
UsdGenUnionSurfaceTargets(SdfPathVector *targets, MeshOf const &meshOf)
{
    SdfPathVector unioned;
    if (!targets || targets->size() < 2) return unioned;
    SdfPath const front = targets->front();
    SdfPath const mesh = meshOf(front);
    if (mesh.IsEmpty()) return unioned;
    SdfPathVector kept{front};
    for (size_t i = 1; i < targets->size(); ++i) {
        SdfPath const &target = (*targets)[i];
        if (target == front) continue;
        if (meshOf(target) != mesh) {
            kept.push_back(target);
        } else if (std::find(unioned.begin(), unioned.end(), target) ==
                   unioned.end()) {
            unioned.push_back(target);
        }
    }
    *targets = std::move(kept);
    return unioned;
}

/// R15: appends the faces one GeomSubset's `indices` name to `faces`. An
/// index outside the parent's `faceCount` faces is a hard diagnostic (rule
/// 7) and is dropped; a parent with no faces yet skips the check, its
/// consumers already warn about the missing topology.
inline void
UsdGenAppendSubsetFaces(VtIntArray const &indices, size_t faceCount,
                        SdfPath const &subset, SdfPath const &mesh,
                        std::vector<int> *faces,
                        std::vector<std::string> *errors)
{
    for (int face : indices) {
        if (faceCount && (face < 0 || size_t(face) >= faceCount)) {
            if (errors) {
                errors->push_back(subset.GetString() + ": GeomSubset names face " +
                    std::to_string(face) + ", outside the " +
                    std::to_string(faceCount) + " faces of " + mesh.GetString() +
                    " (R15)");
            }
            continue;
        }
        faces->push_back(face);
    }
}

/// R15: the sorted, unique subsetFaces a union of GeomSubsets selects, so
/// a face named twice scatters once and authored index order is inert.
/// `whole` (the Mesh itself is a union member) selects every face.
inline VtIntArray
UsdGenFinalizeSubsetFaces(std::vector<int> faces, bool whole, size_t faceCount)
{
    if (whole) {
        faces.resize(faceCount);
        for (size_t f = 0; f < faceCount; ++f) faces[f] = int(f);
    }
    std::sort(faces.begin(), faces.end());
    faces.erase(std::unique(faces.begin(), faces.end()), faces.end());
    return VtIntArray(faces.begin(), faces.end());
}

/// R15: a UsdGenPaintMap whose usdGen:paint:surface is a GeomSubset reads
/// its parent Mesh's faceVarying primvar; the corners of faces outside
/// `faces` (sorted parent-mesh ids) read `fallback`, usdGen:map:default.
/// `values` must hold exactly one value per face vertex of the parent.
inline void
UsdGenMaskPaintValues(VtIntArray const &faceVertexCounts,
                      VtIntArray const &faces, float fallback,
                      VtFloatArray *values)
{
    size_t corner = 0;
    for (size_t f = 0; f < faceVertexCounts.size(); ++f) {
        size_t const n = size_t(std::max(0, faceVertexCounts[f]));
        if (!std::binary_search(faces.cbegin(), faces.cend(), int(f))) {
            for (size_t k = 0; k < n && corner + k < values->size(); ++k)
                (*values)[corner + k] = fallback;
        }
        corner += n;
    }
}

/// The diagnostic both builders report for a subset surface that is not a
/// face set (rule 1) or whose parent is not a Mesh (rule 7).
inline std::string
UsdGenSubsetElementTypeError(SdfPath const &subset, std::string const &type)
{
    return subset.GetString() + ": GeomSubset has elementType '" + type +
        "'; usdGen surfaces need a face subset (R15)";
}
inline std::string
UsdGenSubsetParentError(SdfPath const &subset)
{
    return subset.GetString() + ": GeomSubset parent " +
        subset.GetParentPath().GetString() + " is not a Mesh (R15)";
}

/// True for the usdGen map prim types (UsdGenPtexMap, UsdGenImageMap, ...).
inline bool
UsdGenIsMapTypeName(TfToken const &typeName)
{
    std::string const &s = typeName.GetString();
    return s.size() > 9 && s.compare(0, 6, "UsdGen") == 0 &&
        s.compare(s.size() - 3, 3, "Map") == 0;
}

// Attributes that already own a dedicated desc field; everything else
// reaches the engine through params (S14 pull-all).
inline bool
_isDedicated(TfToken const &name)
{
    // Constant data, not a guarded function static.
    static constexpr char const *dedicated[] = {
        "usdGen:type", "usdGen:mode",
        "usdGen:enabled", "usdGen:seed",
        "usdGen:references", "usdGen:guides", "usdGen:curves",
        "usdGen:frozen:curves", "usdGen:surface",
        // description-level dedicated fields
        "usdGen:tileTarget", "usdGen:curve:basis",
    };
    bool const isDedicated = std::any_of(std::begin(dedicated), std::end(dedicated),
        [&](char const *d) { return name.GetString() == d; });
    // usdGen:look:* lives in UsdGenLookDesc, not params.
    return isDedicated ||
           name.GetString().rfind("usdGen:look:", 0) == 0 ||
           name.GetString().rfind("usdGen:surfaceCage:", 0) == 0;
}

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_GRAPH_DESC_SHARED_H
