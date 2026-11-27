// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// Warm CUDA Session cooks never block a thread on the GPU: 20 cooks after
// warm-up make no CudaFenceStream call, and their completions arrive through
// the completion lane (step 8.7). Allocations per cook are printed (R11).
#include "testBackend.h"
#include "usdGen/exec/allocCounter.h"
#include "usdGen/exec/session.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenUsd/lower.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/array.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

namespace {

constexpr int kNX = 100, kNY = 50;   // 50 area units at density 100: ~5k roots

// scatter -> grow -> noise -> length -> width, as benchUsdGenRestructure's G3.
UsdGenGraphDesc MakeGraph(double noiseMagnitude)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;
    d.executionBackend = usdGenTestUtils::Backend("cuda");
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    s.restPoints = VtVec3fArray((kNX + 1) * (kNY + 1));
    s.uv = VtVec2fArray((kNX + 1) * (kNY + 1));
    for (int j = 0; j <= kNY; ++j)
        for (int i = 0; i <= kNX; ++i) {
            int const k = j * (kNX + 1) + i;
            s.restPoints[k] = GfVec3f(i * 0.1f, j * 0.1f, 0.0f);
            s.uv[k] = GfVec2f(float(i) / kNX, float(j) / kNY);
        }
    s.faceVertexCounts = VtIntArray(kNX * kNY, 4);
    s.faceVertexIndices = VtIntArray(kNX * kNY * 4);
    for (int j = 0; j < kNY; ++j)
        for (int i = 0; i < kNX; ++i) {
            int const a = j * (kNX + 1) + i, o = (j * kNX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + kNX + 2;
            s.faceVertexIndices[o + 3] = a + kNX + 1;
        }
    d.surfaces.push_back(std::move(s));
    auto addNode = [&](std::string const &name, char const *type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = TfToken(type);
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (n.type == TfToken("UsdGenScatter")) n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", "UsdGenScatter", "", 42);
    addNode("grow", "UsdGenGrow", "scatter", 43);
    addNode("noise", "UsdGenNoise", "grow", 44);
    addNode("length", "UsdGenLength", "noise", 45);
    addNode("width", "UsdGenWidth", "length", 46);
    // Schema attributes are float; the device admission checks types.
    auto setp = [&](std::string const &name, char const *p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{TfToken(p), v, false});
    };
    setp("grow", "segments", VtValue(8));
    setp("grow", "length", VtValue(1.0f));
    setp("noise", "noise:magnitude", VtValue(float(noiseMagnitude)));
    setp("noise", "noise:frequency", VtValue(3.0f));
    setp("noise", "noise:octaves", VtValue(2));
    setp("noise", "noise:correlation", VtValue(0.5f));
    setp("length", "length:mode", VtValue(TfToken("scale")));
    setp("length", "length:value", VtValue(1.2f));
    setp("width", "width", VtValue(0.02f));
    return d;
}

bool Commit(UsdGenSession &s, char const *what)
{
    UsdGenSession::SnapshotPtr const snap = s.CommitSnapshot(1.0, UsdGenCommitReason::SetTime);
    if (snap && snap->generation && snap->generation->device && !snap->diagnostics.HasErrors())
        return true;
    std::fprintf(stderr, "%s commit failed%s\n", what,
                 snap && snap->generation && !snap->generation->device ? " (no device generation)" : "");
    if (snap)
        for (auto const &e : snap->diagnostics.errors) std::fprintf(stderr, "  %s\n", e.c_str());
    return false;
}

}  // namespace

int main()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        std::printf("SKIP: no CUDA device\n");
        return 77;
    }
    CHECK(usdGenTestUtils::Backend("cuda") != UsdGenBackendId::Invalid);

    constexpr int kWarmUp = 3, kCooks = 20;
    usdGenUsd::NodeKeyMap keys;
    std::vector<UsdGenGraphSpec> specs;
    for (int i = 0; i != kWarmUp + kCooks; ++i)
        specs.push_back(usdGenUsd::Lower(MakeGraph(0.05 + 0.001 * i), keys));

    UsdGenSession s(4);
    s.SetDevicePublicationEnabled(true);
    // Warm-up: first commit plus edits, so workspaces and pools exist.
    for (int i = 0; i != kWarmUp; ++i) {
        s.SetGraphSpec(specs[size_t(i)]);
        CHECK(Commit(s, "warm-up"));
    }

    UsdGenEngineAllocCounting(true);
    uint64_t const fences0 = gpu::CudaFenceStreamCallsForTesting();
    uint64_t const passes0 = gpu::CudaCompletionLanePassesForTesting();
    uint64_t allocs = 0;
    for (int i = 0; i != kCooks; ++i) {
        // A new noise value each time: a real device cook, no cache hit.
        s.SetGraphSpec(specs[size_t(kWarmUp + i)]);
        uint64_t const a0 = UsdGenEngineAllocCount();
        CHECK(Commit(s, "warm"));
        allocs += UsdGenEngineAllocCount() - a0;
    }
    uint64_t const fences = gpu::CudaFenceStreamCallsForTesting() - fences0;
    uint64_t const passes = gpu::CudaCompletionLanePassesForTesting() - passes0;
    s.Drain();
    std::printf("warm cooks=%d fence calls=%llu completion-lane passes=%llu allocations/cook=%.1f\n",
                kCooks, (unsigned long long)fences, (unsigned long long)passes,
                double(allocs) / kCooks);
    CHECK(fences == 0);
    CHECK(passes > 0);
    std::printf("PASS\n");
    return 0;
}
