#ifndef USDGEN_CUDA_SURFACE_INPUT_H
#define USDGEN_CUDA_SURFACE_INPUT_H

#include "gpu/curveGeometry.h"
#include "usdGen/api/graphSpec.h"
#include <cstdint>
#include <string>
#include <vector>

namespace usdGen {

struct CudaSurfaceBindingKey {
    UsdGenNodeKey surface;
    VtVec3fArray restPoints;
    VtVec3fArray restNormals;
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    VtIntArray faceVertexCounts;
    VtIntArray faceVertexIndices;
    uint32_t sampleBudget = 0;
};

struct CudaSurfacePrepared {
    std::vector<float3> restPoints, currentPoints;
    std::vector<float3> restNormals;
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    std::vector<uint32_t> faceOffsets, faceVertexIndices;
    CudaSurfaceBindingKey key;
    uint32_t sampleBudget = 0;
};

enum class CudaSurfacePreparationStatus {
    Ok, InvalidArgument, InvalidTopology, NonFiniteInput,
    UnsupportedFeature, AllocationFailure
};

CudaSurfacePreparationStatus PrepareCudaSurface(
    UsdGenSurfaceInput const&, uint32_t sampleBudget,
    CudaSurfacePrepared*, std::vector<std::string>* diagnostics = nullptr);

bool RestBindingMatches(CudaSurfacePrepared const&, CudaSurfacePrepared const&);

} // namespace usdGen
#endif
