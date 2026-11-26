// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCudaScatterGrowOverlapFault — a validation throw on the big-grow
// overlap path may discard its buffers only once every queued copy is proven
// idle. The copies run on their own stream and `stream` waits on them only
// through copied_; when a copy fails to issue, copied_ is never recorded, so
// idling `stream` proves nothing about the copies still queued on the copy
// stream: the work must stay unproven (and the destructor quarantine) rather
// than hand those blocks back to the device-buffer cache.
#include "usdGenCurves/cuda/scatterGrow.h"
#include "testBackend.h"
#include "usdGenCurves/cuda/generation.h"
#include "usdGen/executionResources.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace usdGen::gpu;
using namespace usdGen;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

static std::shared_ptr<ScatterGrowRoots> MakeRoots(size_t n)
{
    auto roots = std::make_shared<ScatterGrowRoots>();
    for (size_t i = 0; i < n; ++i) {
        float const f = float(i);
        roots->positions.push_back(make_float3(f, 2.0f * f, 3.0f * f));
        roots->stableIds.push_back(uint64_t(i + 1));
        roots->rootPrim.push_back(int32_t(i % 7));
        roots->rootUV.push_back(make_float2(0.1f, 0.2f));
        roots->rootT.push_back(make_float3(1.0f, 0.0f, 0.0f));
        roots->rootB.push_back(make_float3(0.0f, 1.0f, 0.0f));
        roots->rootN.push_back(make_float3(0.0f, 0.0f, 2.0f));
    }
    return roots;
}

static bool BeginThrows(CudaScatterGrow& grow, std::shared_ptr<ScatterGrowRoots> const& roots,
                        ScatterGrowControls const& controls, cudaStream_t stream)
{
    try {
        (void)grow.BeginFresh(roots, controls, stream);
    } catch (std::runtime_error const&) {
        return true;
    }
    return false;
}

int main() {
    int count = 0, device = -1;
    CHECK(cudaGetDeviceCount(&count) == cudaSuccess);
    if (!count) return 77;
    CHECK(cudaGetDevice(&device) == cudaSuccess);
    constexpr size_t kBig = 70000;  // above the 65536 overlap threshold
    ScatterGrowControls controls;
    controls.cvCount = 8;
    controls.seed = 42;
    controls.length = 1.0;
    ScatterGrowRequirements requirements;
    CHECK(GetScatterGrowRequirements(kBig, controls.cvCount, &requirements) == ScatterGrowStatus::Ok);
    CHECK(ConfigureCudaExecutionResources(device, {4 * requirements.peakBytes + (size_t{64} << 20), 0}));
    auto resources = FindUsdGenExecutionResourcePool({usdGenTestUtils::Backend("cuda"), device});
    CHECK(resources);
    auto const baseline = resources->Snapshot();
    cudaStream_t stream = nullptr;
    CHECK(cudaStreamCreate(&stream) == cudaSuccess);
    auto roots = MakeRoots(kBig);
    {
        // Throw with every copy issued: `stream` covers the copies, so the
        // object discards clean, releases its accounting and stays usable.
        CudaScatterGrow grow;
        SetScatterGrowOverlapFaultsForTesting(false, true);
        CHECK(BeginThrows(grow, roots, controls, stream));
        CHECK(!grow.pending() && !grow.HasUnprovenWork());
        CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);
    }
    CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);
    {
        // Throw after a copy failed to issue: earlier copies may still run on
        // the copy stream, so nothing is discarded and the work stays
        // unproven; destruction quarantines (its charge stays abandoned).
        CudaScatterGrow grow;
        SetScatterGrowOverlapFaultsForTesting(true, true);
        CHECK(BeginThrows(grow, roots, controls, stream));
        CHECK(grow.pending() && grow.HasUnprovenWork());
        CHECK(resources->Snapshot().usedBytes > baseline.usedBytes);
    }
    SetScatterGrowOverlapFaultsForTesting(false, false);
    CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
    std::puts("testUsdGenCudaScatterGrowOverlapFault: PASS");
    return 0;
}
