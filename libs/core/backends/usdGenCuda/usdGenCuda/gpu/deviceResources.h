#ifndef USDGEN_GPU_DEVICE_RESOURCES_H
#define USDGEN_GPU_DEVICE_RESOURCES_H

#include "usdGen/api/ext/device/executionResources.h"

#include "usdGenCuda/export.h"

#include <optional>

namespace usdGen::gpu {
USDGEN_CUDA_API bool ConfigureCudaExecutionResources(int device,
                                     UsdGenExecutionResourceConfig) noexcept;
USDGEN_CUDA_API std::optional<UsdGenExecutionResourcePermit> TryReserveCudaExecutionBytes(
    size_t bytes, UsdGenExecutionResourceKind = UsdGenExecutionResourceKind::Active,
    UsdGenExecutionMemoryReservation* reservation = nullptr) noexcept;
// Upfront job admission counterpart to the byte helper. It atomically charges
// Pending in the current device's shared pool; no snapshot preflight decides
// admission.
USDGEN_CUDA_API std::optional<UsdGenExecutionMemoryReservation> TryReserveCudaExecutionMemory(
    size_t bytes) noexcept;
} // namespace usdGen::gpu
#endif
