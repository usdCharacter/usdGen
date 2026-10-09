// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCudaScatterGrowOverlap — pin the big-grow overlap failure
// discipline (r45): above 65536 curves BeginFresh spec-allocates from the
// cheap validation prefix and overlaps the H2D copies with the validation
// scans. A scan failure there must report exactly ValidateRoots' status,
// release all accounting, and leave the object clean and reusable.
#include "gpu/scatterGrow.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/executionResources.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

using namespace usdGen::gpu;
using namespace usdGen;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)
struct Relay { std::atomic<int> status{-1}; };
static void Done(cudaStream_t, cudaError_t status, void* p) noexcept { static_cast<Relay*>(p)->status.store(int(status)); }

static std::shared_ptr<ScatterGrowRoots> MakeRoots(size_t n)
{
    auto roots = std::make_shared<ScatterGrowRoots>();
    roots->positions.reserve(n);
    roots->stableIds.reserve(n);
    roots->rootPrim.reserve(n);
    roots->rootUV.reserve(n);
    roots->rootT.reserve(n);
    roots->rootB.reserve(n);
    roots->rootN.reserve(n);
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

int main() {
    int count=0, device=-1; CHECK(cudaGetDeviceCount(&count)==cudaSuccess); if(!count) return 77; CHECK(cudaGetDevice(&device)==cudaSuccess);
    constexpr size_t kBig = 70000; // above the 65536 overlap threshold
    ScatterGrowControls controls;
    controls.cvCount = 8;
    controls.seed = 42;
    controls.length = 1.0;
    controls.randomLo = 0.8;
    controls.randomHi = 1.2;
    ScatterGrowRequirements requirements;
    CHECK(GetScatterGrowRequirements(kBig, controls.cvCount, &requirements) == ScatterGrowStatus::Ok);
    CHECK(ConfigureCudaExecutionResources(device, {requirements.peakBytes + (size_t{64} << 20), 0}));
    auto resources=FindUsdGenExecutionResourcePool({UsdGenExecutionResourceBackend::Cuda,device}); CHECK(resources);
    auto const baseline=resources->Snapshot();
    cudaStream_t stream=nullptr; CHECK(cudaStreamCreate(&stream)==cudaSuccess);
    {
    CudaScatterGrow grow;
    size_t total = 0;
    // A mid-range NaN fails the finite-inputs leg after the speculative
    // copies are already in flight: same status as ValidateRoots, same
    // clean object as the pre-alloc failure, accounting fully released.
    auto nanRoots = MakeRoots(kBig);
    nanRoots->positions[40000] = make_float3(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f);
    CHECK(CudaScatterGrow::ValidateRoots(nanRoots, controls, &total) == ScatterGrowStatus::NonFiniteInput);
    CHECK(grow.BeginFresh(nanRoots, controls, stream) == ScatterGrowStatus::NonFiniteInput);
    CHECK(!grow.pending() && !grow.HasUnprovenWork());
    CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);
    // A duplicate id fails the dup leg on the same path.
    auto dupRoots = MakeRoots(kBig);
    dupRoots->stableIds[60000] = dupRoots->stableIds[0];
    CHECK(CudaScatterGrow::ValidateRoots(dupRoots, controls, &total) == ScatterGrowStatus::DuplicateStableId);
    CHECK(grow.BeginFresh(dupRoots, controls, stream) == ScatterGrowStatus::DuplicateStableId);
    CHECK(!grow.pending() && !grow.HasUnprovenWork());
    CHECK(resources->Snapshot().usedBytes == baseline.usedBytes);
    // The same object still grows valid roots to completion.
    auto roots = MakeRoots(kBig);
    CHECK(CudaScatterGrow::ValidateRoots(roots, controls, &total) == ScatterGrowStatus::Ok);
    CHECK(grow.BeginFresh(roots, controls, stream) == ScatterGrowStatus::Ok);
    Relay relay; CHECK(grow.FinishFreshAsync(stream,Done,&relay)==ScatterGrowStatus::Ok);
    CHECK(cudaStreamSynchronize(stream)==cudaSuccess && relay.status.load()==int(cudaSuccess));
    CHECK(grow.CommitFreshFinish()==ScatterGrowStatus::Ok);
    CHECK(grow.curveCount()==kBig && grow.pointCount()==kBig*controls.cvCount && grow.generation()==1);
    }
    CHECK(resources->Snapshot().usedBytes==baseline.usedBytes && resources->Snapshot().byKind==baseline.byKind);
    CHECK(cudaStreamDestroy(stream)==cudaSuccess); std::puts("testUsdGenCudaScatterGrowOverlap: PASS"); return 0;
}
