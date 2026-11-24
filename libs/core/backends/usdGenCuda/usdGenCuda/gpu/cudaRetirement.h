#ifndef USDGEN_GPU_CUDA_RETIREMENT_H
#define USDGEN_GPU_CUDA_RETIREMENT_H

#include "usdGen/executionRetirement.h"

#include <optional>

namespace usdGen::gpu {

/// Reserve one bounded retirement slot for a CUDA device before accepting a
/// published owner or a consumer lease.  The slot is deliberately independent
/// of byte admission: it bounds callback/payload retention, including a
/// permanently quarantined device fault.
std::optional<UsdGenExecutionRetirementTicket> TryReserveCudaRetirement(
    int device) noexcept;
// Irreversibly closes CUDA retirement admission for this process. Call this
// before cudaDeviceReset or unloading CUDA in an embedded host. Existing and
// late tickets are retained/quarantined by the neutral service without CUDA
// cleanup calls. The automatic atexit fallback is not a dlclose guarantee.
void CloseCudaRetirementBackend() noexcept;
bool IsCudaRetirementBackendOpen() noexcept;

} // namespace usdGen::gpu
#endif
