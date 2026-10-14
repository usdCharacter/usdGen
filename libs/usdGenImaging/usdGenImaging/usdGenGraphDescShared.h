// usdGen imaging — private shared decl for the two graph-desc builder TUs.
//
// Split of the S14 dedicated-field set out of usdGenGraphDescBuilder.cpp so
// the stage reference path (usdGenGraphDescBuilderStage.cpp) and the Hydra
// production path (usdGenGraphDescBuilder.cpp) share one definition without
// the B-2-fenced production TU needing any Usd header to reach it.
#ifndef USDGEN_IMAGING_GRAPH_DESC_SHARED_H
#define USDGEN_IMAGING_GRAPH_DESC_SHARED_H

#include "usdGen/digest.h"
#include "usdGen/graphDesc.h"

#include "pxr/base/tf/token.h"

#include <algorithm>
#include <cstring>
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

/// Generation feeds. Same coverage and feed order as the word-at-a-time
/// hasher they replace, mixed with the shared 4-lane word mixer whose
/// contract names capture digests explicitly. Values change; the
/// generations are in-memory equality-only (cross-builder parity tested,
/// never golden).
inline void _GenMixWord(uint64_t &h, uint64_t word)
{
    usdGen::UsdGenDigestMixWord(h, word);
}
inline void _GenMixBytes(uint64_t &h, void const *data, size_t n)
{
    if (n) h = usdGen::UsdGenDigestBytes(data, n, h);
}
template <class A> void _GenMixArray(uint64_t &h, A const &a)
{
    _GenMixWord(h, uint64_t(a.size()));
    if (!a.empty()) _GenMixBytes(h, a.cdata(), a.size() * sizeof(a[0]));
}
inline void _GenMixToken(uint64_t &h, TfToken const &t)
{
    _GenMixBytes(h, t.GetText(), t.size());
    _GenMixWord(h, uint64_t(t.size()));
}

/// Sets UsdGenCurveSetDesc::curveGeneration and
/// UsdGenSurfaceDesc::surfaceGeneration from the content each carries, so a
/// capture digest keyed by them follows every points/topology/id edit (and
/// animation) while both builders give equal content the same value.
/// Provenance flags and the curves' world matrix are not terms: consumers
/// that read the matrix digest it themselves. An entry that already carries
/// a generation keeps it: only the Hydra builder restores carried
/// generations, from descs its dirty gate proved identical, so the content
/// hash is identical too and the re-hash is skipped.
///
/// The surface generation combines two sub-hashes: the slowly-varying rest
/// and topology bytes, and the per-cook points, samples, and attributes. A
/// deform re-cooks every frame with identical rest/topology shares, so the
/// Hydra builder memoizes the invariant sub-hash across cooks (see
/// UsdGenSurfaceDesc::surfaceInvariantGeneration) and only the moving bytes
/// re-hash. A zero invariant hash always recomputes here, identically in
/// both builders, so a memo miss (or a hash that is legitimately zero) is
/// exactly the fresh value.
inline void
UsdGenFinalizeInputGenerations(usdGen::UsdGenGraphDesc *desc)
{
    for (usdGen::UsdGenCurveSetDesc &curves : desc->curveSets) {
        if (curves.curveGeneration) continue;
        uint64_t h = usdGen::UsdGenDigestOffset;
        _GenMixWord(h, uint64_t(curves.role));
        _GenMixToken(h, curves.curveRole);
        _GenMixArray(h, curves.curveVertexCounts);
        _GenMixArray(h, curves.points);
        _GenMixArray(h, curves.rest);
        _GenMixArray(h, curves.widths);
        _GenMixToken(h, curves.type);
        _GenMixToken(h, curves.basis);
        _GenMixToken(h, curves.wrap);
        _GenMixToken(h, curves.widthsInterpolation);
        _GenMixArray(h, curves.skinPrim);
        _GenMixArray(h, curves.curveId);
        _GenMixArray(h, curves.skinPrimUv);
        _GenMixArray(h, curves.rootFrame);
        _GenMixBytes(h, curves.frozenEpoch.data(), curves.frozenEpoch.size());
        for (auto const &plane : curves.authoredPlanes) {
            _GenMixToken(h, plane.name);
            _GenMixWord(h, (uint64_t(plane.type) << 16) | (uint64_t(plane.domain) << 8) | plane.arity);
            _GenMixArray(h, plane.floatValues);
            _GenMixArray(h, plane.intValues);
        }
        _GenMixWord(h, curves.surfaceCage ? 1u : 0u);
        if (curves.surfaceCage) {
            usdGen::UsdGenSurfaceCagePayload const &cage =
                *curves.surfaceCage;
            _GenMixArray(h, cage.ownerIds);
            _GenMixArray(h, cage.ownerDensities);
            _GenMixArray(h, cage.ownerSeeds);
            _GenMixArray(h, cage.ownerCvCounts);
            _GenMixArray(h, cage.ownerEdgeBias);
            _GenMixArray(h, cage.ownerLengthProfileOffsets);
            _GenMixArray(h, cage.ownerLengthProfile);
            _GenMixArray(h, cage.normalizedT);
            _GenMixArray(h, cage.triangles);
            _GenMixArray(h, cage.triangleOwnerIndices);
            _GenMixArray(h, cage.triangleRootCharts);
            _GenMixArray(h, cage.ownerChartCentroids);
            _GenMixArray(h, cage.ownerChartMeanRadii);
        }
        curves.curveGeneration = h ? h : 1;
    }
    for (usdGen::UsdGenSurfaceDesc &surface : desc->surfaces) {
        if (surface.surfaceGeneration) continue;
        uint64_t inv = surface.surfaceInvariantGeneration;
        if (!inv) {
            inv = usdGen::UsdGenDigestOffset;
            _GenMixArray(inv, surface.faceVertexCounts);
            _GenMixArray(inv, surface.faceVertexIndices);
            _GenMixArray(inv, surface.restPoints);
            _GenMixArray(inv, surface.restNormals);
            _GenMixWord(inv, uint64_t(surface.restNormalDomain));
            surface.surfaceInvariantGeneration = inv;
        }
        uint64_t h = usdGen::UsdGenDigestOffset;
        _GenMixWord(h, inv);
        _GenMixArray(h, surface.points);
        for (size_t i = 0; i < surface.samples.size(); ++i) {
            auto const &sample = surface.samples[i];
            _GenMixBytes(h, &sample.time, sizeof(sample.time));
            // The builders seed each sample's points from the surface points,
            // so a sample usually shares them: identical shares are already
            // covered above (or by an earlier sample), and re-hashing the
            // same ~12MB buys no sensitivity. Times still mix, so a new
            // sample always moves the generation. Values change; generations
            // are in-memory equality-only, and both builders share this
            // function so parity holds.
            bool covered = sample.points.IsIdentical(surface.points);
            for (size_t j = 0; !covered && j < i; ++j)
                covered = sample.points.IsIdentical(
                    surface.samples[j].points);
            if (!covered)
                _GenMixArray(h, sample.points);
        }
        _GenMixArray(h, surface.velocities);
        _GenMixArray(h, surface.uv);
        _GenMixArray(h, surface.subsetFaces);
        _GenMixWord(h, surface.isSubset ? 1u : 0u);
        _GenMixBytes(h, surface.worldMatrix.GetArray(), 16 * sizeof(double));
        surface.surfaceGeneration = h ? h : 1;
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
    static std::unordered_set<std::string> const dedicated{
        "usdGen:type", "usdGen:mode",
        "usdGen:enabled", "usdGen:seed",
        "usdGen:references", "usdGen:guides", "usdGen:curves",
        "usdGen:frozen:curves", "usdGen:surface",
        // description-level dedicated fields
        "usdGen:tileTarget", "usdGen:curve:basis",
    };
    // usdGen:look:* lives in UsdGenLookDesc, not params.
    return dedicated.count(name.GetString()) != 0 ||
           name.GetString().rfind("usdGen:look:", 0) == 0 ||
           name.GetString().rfind("usdGen:surfaceCage:", 0) == 0;
}

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_GRAPH_DESC_SHARED_H
