// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// DeviceBuffer size classes: a count that moves within its class keeps its
// block (in place or from the pool), permits charge the requested bytes, and
// concurrent resizes keep data and accounting exact.
#include "usdGenCuda/gpu/deviceBuffer.h"
#include "testBackend.h"
#include "usdGenCuda/gpu/deviceResources.h"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(gpu::ConfigureCudaExecutionResources(device, {size_t{1} << 30, 0}));
    auto resources = FindUsdGenExecutionResourcePool({usdGenTestUtils::Backend("cuda"), device});
    CHECK(resources);
    {
        gpu::DeviceBuffer<float> a;
        CHECK(a.reset(1000) == cudaSuccess);
        float* const block = a.data();
        // 4000 and 4040 bytes share the 4096-byte class: same block, exact charge.
        CHECK(a.reset(1010) == cudaSuccess && a.data() == block && a.size() == 1010);
        CHECK(resources->Snapshot().usedBytes == 1010 * sizeof(float));
        // 8000 bytes is another class: a new block; the old one is pooled.
        CHECK(a.reset(2000) == cudaSuccess && a.data() != block);
        CHECK(resources->Snapshot().usedBytes == 2000 * sizeof(float));
        CHECK(gpu::UsdGenGpuDeviceBufferCacheBytes() == 1010 * sizeof(float));
        // A third size in the first class takes the pooled block back.
        gpu::DeviceBuffer<float> b;
        CHECK(b.reset(1020) == cudaSuccess && b.data() == block);
        CHECK(gpu::UsdGenGpuDeviceBufferCacheBytes() == 0);
        CHECK(resources->Snapshot().usedBytes == 3020 * sizeof(float));
    }
    CHECK(resources->Snapshot().usedBytes == 0);

    // Every class from 256 bytes to 64MiB has its own cache bucket: a freed
    // block of each class comes back for a smaller request in that class.
    {
        std::vector<size_t> classes{256};
        for (size_t top = 8; top != 26; ++top)
            for (size_t k = 1; k <= 4; ++k)
                classes.push_back((size_t{1} << top) + k * (size_t{1} << (top - 2)));
        std::vector<void*> blocks;
        for (size_t bytes : classes) {
            gpu::DeviceBuffer<unsigned char> b;
            CHECK(b.reset(bytes) == cudaSuccess);
            blocks.push_back(b.data());
        }
        CHECK(gpu::UsdGenGpuDeviceBufferCacheBytes() > 0);
        for (size_t i = 0; i != classes.size(); ++i) {
            // The smallest request that still rounds up to this class.
            size_t const below = i ? classes[i - 1] + 1 : 1;
            gpu::DeviceBuffer<unsigned char> b;
            CHECK(b.reset(below) == cudaSuccess && b.data() == blocks[i]);
        }
    }
    CHECK(resources->Snapshot().usedBytes == 0);

    // Threads resize within and across classes and check their own bytes.
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t != 16; ++t) {
        threads.emplace_back([t, device, &errors] {
            cudaSetDevice(device);
            gpu::DeviceBuffer<uint32_t> buffer;
            std::vector<uint32_t> host;
            for (int round = 0; round != 200; ++round) {
                size_t const count = 512 + size_t((t * 131 + round * 37) % 3000);
                if (buffer.reset(count) != cudaSuccess || buffer.size() != count) { errors.fetch_add(1); return; }
                host.assign(count, uint32_t(t * 1000 + round));
                std::vector<uint32_t> back(count, 0);
                if (cudaMemcpy(buffer.data(), host.data(), count * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
                    cudaMemcpy(back.data(), buffer.data(), count * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
                    back != host)
                    errors.fetch_add(1);
                if (round % 7 == 0) buffer.release();
            }
        });
    }
    for (auto& thread : threads) thread.join();
    CHECK(errors.load() == 0);
    CHECK(resources->Snapshot().usedBytes == 0);
    std::printf("PASS\n");
    return 0;
}
