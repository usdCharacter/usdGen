#include "usdGen/gpu/deviceBuffer.h"
#include "usdGen/gpu/deviceResources.h"

#include <cstdio>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    // This process configures the device before its first DeviceBuffer.
    CHECK(gpu::ConfigureCudaExecutionResources(device, {1024, 128}));
    gpu::DeviceBuffer<float> empty, buffer;
    CHECK(empty.recordUse(nullptr) == cudaSuccess && empty.waitOn(nullptr) == cudaSuccess);
    CHECK(buffer.reset(33) == cudaSuccess);
    float const sentinel = 7.25f;
    CHECK(cudaMemcpy(buffer.data(), &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice) == cudaSuccess);
    auto resources = FindUsdGenExecutionResourcePool(
        {UsdGenExecutionResourceBackend::Cuda, device});
    CHECK(resources && resources->Snapshot().usedBytes == 33 * sizeof(float));
    auto const pointer = buffer.data();
    auto moved = std::move(buffer);
    CHECK(buffer.recordUse(nullptr) == cudaSuccess && buffer.waitOn(nullptr) == cudaSuccess);
    CHECK(moved.reset(200) == cudaErrorMemoryAllocation);
    float restored = 0;
    CHECK(moved.data() == pointer && moved.size() == 33 &&
          cudaMemcpy(&restored, moved.data(), sizeof(restored), cudaMemcpyDeviceToHost) == cudaSuccess &&
          restored == sentinel && resources->Snapshot().usedBytes == 33 * sizeof(float));
    moved.release();
    CHECK(resources->Snapshot().usedBytes == 0);
    CHECK(moved.reset(200) == cudaSuccess);
    CHECK(resources->Snapshot().usedBytes == 200 * sizeof(float));
    // DeviceBuffer reuse cache: retention, exact-size reuse, and
    // drain-on-pressure. Entering: `moved` holds 200 floats live (800B
    // charged); the 33-float block sits cached with its permit released.
    float const* movedData = moved.data();
    CHECK(gpu::UsdGenGpuDeviceBufferCacheBytes() == 33 * sizeof(float));
    moved.release();
    CHECK(resources->Snapshot().usedBytes == 0 &&
          gpu::UsdGenGpuDeviceBufferCacheBytes() == 233 * sizeof(float));
    gpu::DeviceBuffer<float> small, big, extra;
    CHECK(small.reset(33) == cudaSuccess);
    CHECK(small.data() == pointer &&
          resources->Snapshot().usedBytes == 33 * sizeof(float) &&
          gpu::UsdGenGpuDeviceBufferCacheBytes() == 200 * sizeof(float));
    small.release();
    CHECK(big.reset(200) == cudaSuccess);
    CHECK(big.data() == movedData &&
          resources->Snapshot().usedBytes == 200 * sizeof(float) &&
          gpu::UsdGenGpuDeviceBufferCacheBytes() == 33 * sizeof(float));
    // Pressure: 800B live + a 132B cached block, then a 132B admission
    // that overfills the 896B usable budget. It must fail exactly as
    // without the cache, after shedding the candidate and the cache.
    CHECK(extra.reset(33) == cudaErrorMemoryAllocation);
    CHECK(resources->Snapshot().usedBytes == 200 * sizeof(float) &&
          gpu::UsdGenGpuDeviceBufferCacheBytes() == 0);
    big.release();
    CHECK(resources->Snapshot().usedBytes == 0);
    CHECK(extra.reset(200) == cudaSuccess);
    CHECK(extra.data() == movedData &&
          resources->Snapshot().usedBytes == 200 * sizeof(float) &&
          gpu::UsdGenGpuDeviceBufferCacheBytes() == 0);
    return 0;
}
