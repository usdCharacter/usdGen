#ifndef USDGEN_GPU_DEVICE_RESOURCES_H
#define USDGEN_GPU_DEVICE_RESOURCES_H

#include "usdGen/executionResources.h"

#include <optional>

namespace usdGen::gpu {
bool ConfigureCudaExecutionResources(int device,
                                     UsdGenExecutionResourceConfig) noexcept;
std::optional<UsdGenExecutionResourcePermit> TryReserveCudaExecutionBytes(
    size_t bytes, UsdGenExecutionResourceKind = UsdGenExecutionResourceKind::Active,
    UsdGenExecutionMemoryReservation* reservation = nullptr) noexcept;
// Upfront job admission counterpart to the byte helper. It atomically charges
// Pending in the current device's shared pool; no snapshot preflight decides
// admission.
std::optional<UsdGenExecutionMemoryReservation> TryReserveCudaExecutionMemory(
    size_t bytes) noexcept;
} // namespace usdGen::gpu
#endif
