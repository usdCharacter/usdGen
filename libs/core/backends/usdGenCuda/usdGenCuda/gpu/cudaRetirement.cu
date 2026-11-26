#include "cudaRetirement.h"
#include "usdGenCuda/backendId.h"

#include <atomic>
#include <cstdlib>

#include <cuda_runtime.h>

namespace usdGen::gpu {
namespace {
void CloseCudaRetirementAtExit() noexcept {
    // This only closes/quiesces neutral retention; it deliberately makes no
    // CUDA runtime call, so it can run before loader CUDA finalization.
    CloseCudaRetirementBackend();
}
// Constant-initialized and trivially destructible: no guard, no static dtor.
std::atomic<bool> g_atexitRegistered{false};
std::atomic<bool>& AtexitRegistered() { return g_atexitRegistered; }
bool RegisterCudaRetirementAtExit() noexcept {
    if (AtexitRegistered().load(std::memory_order_acquire)) return true;
    // Do not publish readiness before registration succeeds. Concurrent first
    // callers may register duplicate callbacks; backend close is idempotent,
    // while a failed registration remains fail-closed for that caller.
    if (std::atexit(CloseCudaRetirementAtExit) != 0) return false;
    AtexitRegistered().store(true, std::memory_order_release);
    return true;
}
bool InitializeCudaDeviceForRetirement(int device) noexcept {
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess)
        return false;
    if (previous != device && cudaSetDevice(previous) != cudaSuccess) return false;
    return true;
}
}

bool IsCudaRetirementBackendOpen() noexcept {
    return IsUsdGenExecutionRetirementBackendOpen(
        UsdGenCudaBackendId());
}
void CloseCudaRetirementBackend() noexcept {
    QuiesceUsdGenExecutionRetirementBackend(
        UsdGenCudaBackendId());
}

std::optional<UsdGenExecutionRetirementTicket>
TryReserveCudaRetirement(int device) noexcept
{
    if (device < 0 || !IsCudaRetirementBackendOpen() ||
        !InitializeCudaDeviceForRetirement(device)) return std::nullopt;
    auto service = GetOrCreateUsdGenExecutionRetirementService(
        {UsdGenCudaBackendId(), device}, {1024}, false);
    if (!service) return {};
    // Register only after a real CUDA retirement service exists.  The
    // backend-wide neutral gate is shared across DSOs, making duplicate
    // plugin atexit callbacks harmless and preventing later creation.
    if (!RegisterCudaRetirementAtExit()) return {};
    return service->TryReserve();
}
} // namespace usdGen::gpu
