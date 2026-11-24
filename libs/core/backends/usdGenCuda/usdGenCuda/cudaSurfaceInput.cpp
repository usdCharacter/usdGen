#ifdef USDGEN_ENABLE_CUDA
#include "cudaSurfaceInput.h"

#include <cmath>
#include <limits>
#include <new>

namespace usdGen {
namespace {
void Diag(std::vector<std::string>* d, std::string const& s) { if (d) d->push_back(s); }
bool Finite(VtVec3fArray const& points) {
    for (auto const& p : points)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) return false;
    return true;
}
bool Identity(GfMatrix4d const& m) {
    GfMatrix4d const identity(1.0);
    for (int r = 0; r != 4; ++r) for (int c = 0; c != 4; ++c)
        if (m[r][c] != identity[r][c]) return false;
    return true;
}
bool SameKey(CudaSurfaceBindingKey const& a, CudaSurfaceBindingKey const& b) {
    return a.path == b.path && a.restPoints == b.restPoints &&
        a.restNormals == b.restNormals && a.restNormalDomain == b.restNormalDomain &&
        a.faceVertexCounts == b.faceVertexCounts &&
        a.faceVertexIndices == b.faceVertexIndices &&
        a.sampleBudget == b.sampleBudget;
}
bool ValidNormalShape(UsdGenSurfaceDesc const& source) {
    const size_t faceCount = source.faceVertexCounts.size();
    const size_t vertexCount = source.restPoints.size();
    const size_t faceVaryingCount = source.faceVertexIndices.size();
    switch (source.restNormalDomain) {
    case UsdGenSurfaceNormalDomain::None: return source.restNormals.empty();
    case UsdGenSurfaceNormalDomain::Constant: return source.restNormals.size() == 1;
    case UsdGenSurfaceNormalDomain::Uniform: return source.restNormals.size() == faceCount;
    case UsdGenSurfaceNormalDomain::Vertex: return source.restNormals.size() == vertexCount;
    case UsdGenSurfaceNormalDomain::FaceVarying: return source.restNormals.size() == faceVaryingCount;
    case UsdGenSurfaceNormalDomain::Invalid: return false;
    }
    return false;
}
} // namespace

CudaSurfacePreparationStatus PrepareCudaSurface(
    UsdGenSurfaceDesc const& source, uint32_t sampleBudget,
    CudaSurfacePrepared* output, std::vector<std::string>* diagnostics) {
    if (!output) { Diag(diagnostics, "null CUDA surface output"); return CudaSurfacePreparationStatus::InvalidArgument; }
    auto fail = [&](CudaSurfacePreparationStatus status, char const* message) {
        Diag(diagnostics, message); return status;
    };
    if (sampleBudget < 4) return fail(CudaSurfacePreparationStatus::InvalidArgument, "surface sampleBudget must be at least 4");
    if (source.restFromCurrentPoints)
        return fail(CudaSurfacePreparationStatus::UnsupportedFeature, "RBF requires a Default-time rest surface; current-frame fallback is not a binding");
    // A GeomSubset surface carries its parent mesh's full topology (R15):
    // like the CPU lane, the RBF field samples the whole parent, so
    // subsetFaces (a Scatter/Collide restriction) is not an RBF input.
    if (!Identity(source.worldMatrix)) return fail(CudaSurfacePreparationStatus::UnsupportedFeature, "non-identity surface transforms are unsupported");
    if (source.restPoints.empty() || source.points.empty() || source.restPoints.size() != source.points.size())
        return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface rest/current vertex counts do not match");
    if (source.restPoints.size() > std::numeric_limits<uint32_t>::max() ||
        source.faceVertexIndices.size() > std::numeric_limits<uint32_t>::max())
        return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface exceeds uint32 index limits");
    if (!Finite(source.restPoints) || !Finite(source.points) || !Finite(source.restNormals))
        return fail(CudaSurfacePreparationStatus::NonFiniteInput,
                    "surface positions and rest normals must be finite");
    size_t indexCount = 0;
    for (int count : source.faceVertexCounts) {
        if (count != 3 && count != 4) return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface faces must be triangles or quads");
        if (indexCount > std::numeric_limits<size_t>::max() - size_t(count))
            return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface face index count overflow");
        indexCount += size_t(count);
    }
    if (indexCount != source.faceVertexIndices.size())
        return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface face counts do not match indices");
    for (int index : source.faceVertexIndices)
        if (index < 0 || size_t(index) >= source.points.size())
            return fail(CudaSurfacePreparationStatus::InvalidTopology, "surface face index is out of range");
    if (!ValidNormalShape(source))
        return fail(CudaSurfacePreparationStatus::InvalidArgument,
                    "rest normals must have a supported interpolation and exact cardinality");

    try {
        CudaSurfacePrepared candidate;
        candidate.restPoints.reserve(source.restPoints.size());
        candidate.currentPoints.reserve(source.points.size());
        candidate.restNormals.reserve(source.restNormals.size());
        candidate.faceOffsets.reserve(source.faceVertexCounts.size() + 1);
        candidate.faceVertexIndices.reserve(source.faceVertexIndices.size());
        for (auto const& p : source.restPoints) candidate.restPoints.push_back({p[0], p[1], p[2]});
        for (auto const& p : source.points) candidate.currentPoints.push_back({p[0], p[1], p[2]});
        for (auto const& p : source.restNormals) candidate.restNormals.push_back({p[0], p[1], p[2]});
        candidate.restNormalDomain = source.restNormalDomain;
        candidate.faceOffsets.push_back(0);
        uint64_t offset = 0;
        for (int count : source.faceVertexCounts) {
            for (int i = 0; i < count; ++i)
                candidate.faceVertexIndices.push_back(static_cast<uint32_t>(source.faceVertexIndices[offset + i]));
            offset += static_cast<uint64_t>(count);
            candidate.faceOffsets.push_back(static_cast<uint32_t>(offset));
        }
        candidate.key = {source.path, source.restPoints, source.restNormals, source.restNormalDomain,
                         source.faceVertexCounts,
                         source.faceVertexIndices, sampleBudget};
        candidate.sampleBudget = sampleBudget;
        *output = std::move(candidate);
        return CudaSurfacePreparationStatus::Ok;
    } catch (std::bad_alloc const&) {
        return fail(CudaSurfacePreparationStatus::AllocationFailure, "surface preparation allocation failed");
    }
}

bool RestBindingMatches(CudaSurfacePrepared const& a, CudaSurfacePrepared const& b) {
    return SameKey(a.key, b.key);
}
} // namespace usdGen
#endif // USDGEN_ENABLE_CUDA
