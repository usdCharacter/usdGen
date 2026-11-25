#ifndef USDGEN_CUDA_SCATTER_INPUT_H
#define USDGEN_CUDA_SCATTER_INPUT_H

#include "usdGen/graphDesc.h"
#include "usdGen/gpu/scatterGrow.h"

#include <memory>
#include <string>

namespace usdGen {

// CPU-authoritative preparation boundary for the narrow Scatter -> Grow CUDA
// topology slice.  It deliberately runs the registered CPU Scatter capture
// kernel, then copies its already Morton-ordered roots into an immutable CUDA
// input owner.  `out` is written only on success.
enum class CudaScatterInputStatus {
    Ok,
    InvalidArgument,
    Unsupported,
    InvalidSurface,
    CaptureFailed
};

CudaScatterInputStatus PrepareCudaScatterInput(
    UsdGenGraphDesc const& desc, SdfPath const& scatterPath,
    std::shared_ptr<const gpu::ScatterGrowRoots>* out,
    std::string* reason = nullptr);

} // namespace usdGen

#endif
