// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Binding statistics of a CUDA cook, carried as the provider's report.
#pragma once
#include "usdGenCurves/cuda/export.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/registries/publisher.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace usdGen {

struct UsdGenSessionSnapshot;

/// UsdGenSessionDeviceResult::report of the CUDA provider.
class UsdGenCudaBindingReport final : public UsdGenPublication {
public:
    static constexpr uint8_t kKind = 0xCB;
    UsdGenCudaBindingReport() noexcept : UsdGenPublication(UsdGenTypeId::Invalid, kKind) {}
    std::vector<UsdGenCudaBindingStats> stats;
};

/// The binding statistics of the snapshot's last executed CUDA cook; empty
/// without one.
USDGEN_CURVES_CUDA_API std::vector<UsdGenCudaBindingStats> UsdGenCudaBindingStatsOf(
    UsdGenSessionSnapshot const *);
inline std::vector<UsdGenCudaBindingStats> UsdGenCudaBindingStatsOf(
    std::shared_ptr<const UsdGenSessionSnapshot> const &snapshot)
{
    return UsdGenCudaBindingStatsOf(snapshot.get());
}

}  // namespace usdGen
