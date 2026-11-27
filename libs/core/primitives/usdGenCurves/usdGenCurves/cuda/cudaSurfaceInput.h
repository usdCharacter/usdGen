#ifndef USDGEN_CUDA_SURFACE_INPUT_H
#define USDGEN_CUDA_SURFACE_INPUT_H

#include "usdGenCurves/cuda/curveGeometry.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/cowArray.h"
#include <cstdint>
#include <string>
#include <vector>

namespace usdGen {

struct CudaSurfaceBindingKey {
    UsdGenNodeKey surface;
    UsdGenVec3fArray restPoints;
    UsdGenVec3fArray restNormals;
    UsdGenSurfaceNormalDomain restNormalDomain = UsdGenSurfaceNormalDomain::None;
    UsdGenIntArray faceVertexCounts;
    UsdGenIntArray faceVertexIndices;
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
    UsdGenSurfaceInputView const&, uint32_t sampleBudget,
    CudaSurfacePrepared*, std::vector<std::string>* diagnostics = nullptr);

bool RestBindingMatches(CudaSurfacePrepared const&, CudaSurfacePrepared const&);

} // namespace usdGen
#endif
