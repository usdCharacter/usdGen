#ifndef USDGEN_CUDA_SOURCE_INPUT_H
#define USDGEN_CUDA_SOURCE_INPUT_H

#include "gpu/curveSource.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace usdGen {

enum class CudaSourceIdSource { Primvar, Index };
enum class CudaSourceStaleAction { Warn, Ignore, Block };

struct CudaSourcePreparationOptions {
    float defaultWidth = 0.01f;
    bool useRest = true;
    CudaSourceIdSource idSource = CudaSourceIdSource::Primvar;
    std::string expectedEpoch;
    std::string actualEpoch;
    CudaSourceStaleAction staleAction = CudaSourceStaleAction::Warn;
    int resampleTo = 0;
    std::string rebind = "never";
    bool hasRootFrame = false;
};

struct CudaSourcePreparationInput {
    std::vector<int32_t> curveVertexCounts;
    std::vector<float3> points;
    std::vector<float3> rest;
    std::vector<float> widths;
    std::vector<float> hairT;
    std::vector<uint64_t> curveId;
    std::vector<int32_t> rootPrim;
    std::vector<float2> rootUV;
    // USD/Gf row-major doubles, one optional rest frame per curve. CUDA
    // upload and device validation intentionally happen in a later layer.
    std::vector<std::array<double, 16>> rootFrames;
};

struct CudaSourcePrepared {
    std::vector<int32_t> curveVertexCounts;
    std::vector<float3> points, rest;
    std::vector<float> widths, hairT;
    std::vector<uint64_t> curveId;
    std::vector<int32_t> rootPrim;
    std::vector<float2> rootUV;
    std::vector<std::array<double, 16>> rootFrames;
    bool useRest = true;
    uint32_t warningFlags = gpu::CurveSourceWarningNone;

    gpu::CurveSourceInput Input() const;
};

enum class CudaSourcePreparationStatus {
    Ok, InvalidArgument, InvalidTopology, NonFiniteInput,
    DuplicateStableId, StaleEpoch, UnsupportedFeature, AllocationFailure
};

// Compact a validated source into fresh owners without renumbering stable IDs.
// Retained IDs may be unordered; duplicates/unknown IDs and malformed channel
// cardinalities reject transactionally. Empty optional channels stay absent.
bool CompactCudaSource(CudaSourcePrepared* prepared,
                       std::vector<uint64_t> const& retainedStableIds);

CudaSourcePreparationStatus PrepareCudaSource(
    CudaSourcePreparationInput const& source,
    CudaSourcePreparationOptions const& options,
    CudaSourcePrepared* prepared,
    std::vector<std::string>* diagnostics = nullptr);

} // namespace usdGen
#endif
