#ifndef USDGEN_CUDA_SCATTER_INPUT_H
#define USDGEN_CUDA_SCATTER_INPUT_H

#include "usdGen/api/graphSpec.h"
#include "usdGenCurves/cuda/scatterGrow.h"

#include <memory>
#include <string>
#include "usdGenCurves/cuda/export.h"

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

USDGEN_CURVES_CUDA_INTERNAL_API CudaScatterInputStatus PrepareCudaScatterInput(
    UsdGenGraphSpec const& desc, UsdGenNodeKey scatterKey,
    std::shared_ptr<const gpu::ScatterGrowRoots>* out,
    std::string* reason = nullptr);

} // namespace usdGen

#endif
