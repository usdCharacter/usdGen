#include "deviceResources.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace usdGen::gpu {
namespace {
constexpr size_t OneGiB = size_t{1} << 30;
UsdGenExecutionResourceDevice Key(int device) { return {UsdGenExecutionResourceBackend::Cuda, device}; }
UsdGenExecutionResourceConfig DefaultConfig(size_t freeBytes) {
    size_t headroom = std::max(OneGiB, freeBytes / 5);
    return {freeBytes, std::min(headroom, freeBytes)};
}
}
bool ConfigureCudaExecutionResources(int device, UsdGenExecutionResourceConfig config) noexcept {
    return device >= 0 && static_cast<bool>(GetOrCreateUsdGenExecutionResourcePool(Key(device), config, true));
}
std::optional<UsdGenExecutionResourcePermit> TryReserveCudaExecutionBytes(
    size_t bytes, UsdGenExecutionResourceKind kind,
    UsdGenExecutionMemoryReservation* reservation) noexcept {
    // A job reservation is already globally charged. Do not inspect CUDA
    // free memory or fall back to pool admission: that would let concurrent
    // jobs steal/recharge its explicitly reserved balance.
    if (reservation) return reservation->Consume(bytes, kind);
    int device = -1; size_t freeBytes = 0, totalBytes = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
        return std::nullopt;
    auto pool = GetOrCreateUsdGenExecutionResourcePool(Key(device), DefaultConfig(freeBytes), false);
    if (!pool) return std::nullopt;
    auto const snapshot = pool->Snapshot();
    // Best-effort external-state check only; CUDA and third-party allocations
    // can still race it, so this is not a physical-VRAM hard-cap claim.
    if (freeBytes <= snapshot.headroomBytes || bytes > freeBytes - snapshot.headroomBytes)
        return std::nullopt;
    return pool->TryReserve(bytes, kind);
}
std::optional<UsdGenExecutionMemoryReservation> TryReserveCudaExecutionMemory(
    size_t bytes) noexcept {
    int device = -1; size_t freeBytes = 0, totalBytes = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
        return std::nullopt;
    auto pool = GetOrCreateUsdGenExecutionResourcePool(
        Key(device), DefaultConfig(freeBytes), false);
    if (!pool) return std::nullopt;
    auto const snapshot = pool->Snapshot();
    // This is only a best-effort external CUDA-state guard using immutable
    // pool headroom; pool admission itself is decided atomically below.
    // Third-party allocations can still race cudaMemGetInfo, so this is not a
    // physical-VRAM hard-cap claim and never replaces TryReserveMemory.
    if (freeBytes <= snapshot.headroomBytes ||
        bytes > freeBytes - snapshot.headroomBytes)
        return std::nullopt;
    return pool->TryReserveMemory(bytes);
}
} // namespace usdGen::gpu
