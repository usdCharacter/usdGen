// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// CUDA cook timings: Scatter -> Grow -> Width -> Length on one quad, cooked
// through the direct executor (ExecuteCudaGraph) and through a Session with
// device publication (the asynchronous relay path). Prints medians in ms.
//   benchUsdGenCudaCook [--density N] [--runs N]
#include "usdGen/cudaExecution.h"
#include "../testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/session.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace usdGen;

static UsdGenGraphDesc Desc(float density) {
    UsdGenGraphDesc d;
    d.description = SdfPath("/Groom/Bench");
    d.executionBackend = usdGenTestUtils::Backend("cuda");
    d.defaultWidth = .025f;
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Scalp");
    surface.restPoints = {{0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}};
    surface.points = surface.restPoints;
    surface.uv = {{0,0}, {1,0}, {1,1}, {0,1}};
    surface.faceVertexCounts = {4};
    surface.faceVertexIndices = {0,1,2,3};
    d.surfaces.push_back(surface);
    UsdGenNodeDesc scatter;
    scatter.path = SdfPath("/Ops/Scatter");
    scatter.type = TfToken("UsdGenScatter");
    scatter.seed = 41;
    scatter.surfaces = {surface.path};
    scatter.params = {{TfToken("density"), VtValue(density), false}};
    UsdGenNodeDesc grow;
    grow.path = SdfPath("/Ops/Grow");
    grow.type = TfToken("UsdGenGrow");
    grow.seed = 19;
    grow.inputs = {scatter.path};
    grow.params = {{TfToken("segments"), VtValue(8), false},
                   {TfToken("length"), VtValue(2.0f), false},
                   {TfToken("lengthRandom"), VtValue(GfVec2f(.5f, 1.5f)), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Ops/Width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {grow.path};
    width.params = {{TfToken("width"), VtValue(.02f), false}};
    UsdGenNodeDesc length;
    length.path = SdfPath("/Ops/Length");
    length.type = TfToken("UsdGenLength");
    length.inputs = {width.path};
    length.params = {{TfToken("length:mode"), VtValue(TfToken("scale")), false},
                     {TfToken("length:value"), VtValue(.75f), false}};
    d.nodes = {scatter, grow, width, length};
    d.terminal = length.path;
    return d;
}

static double Median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main(int argc, char** argv) {
    float density = 200000.f;
    int runs = 30;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--density")) density = float(std::atof(argv[i + 1]));
        else if (!std::strcmp(argv[i], "--runs")) runs = std::max(3, std::atoi(argv[i + 1]));
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    UsdGenGraphDesc desc = Desc(density);
    UsdGenDiagnostics diagnostics;
    auto plan = CompileCudaGraph(usdGenUsd::Lower(desc), &diagnostics);
    if (!plan) { std::fprintf(stderr, "compile failed\n"); return 1; }
    auto workspace = CreateCudaExecutionWorkspace(-1, &diagnostics);
    if (!workspace) { std::fprintf(stderr, "workspace failed\n"); return 1; }
    std::shared_ptr<const UsdGenDeviceGeneration> previous;
    std::vector<double> direct;
    size_t curves = 0;
    for (int run = 0; run != runs + 3; ++run) {
        auto const t0 = Clock::now();
        auto next = ExecuteCudaGraph(*plan, *workspace, 1.0 + run, 1000 + uint64_t(run),
                                     &diagnostics, previous);
        auto const t1 = Clock::now();
        if (!next || diagnostics.HasErrors()) {
            for (auto const& e : diagnostics.errors) std::fprintf(stderr, "%s\n", e.c_str());
            return 1;
        }
        curves = next->Geometry().elementCount;
        previous = std::move(next);
        if (run >= 3) direct.push_back(ms(t0, t1));
    }
    previous.reset();
    std::vector<double> session;
    {
        UsdGenSession s;
        s.SetDevicePublicationEnabled(true);
        usdGenUsd::SetGraphDesc(s, desc);
        for (int run = 0; run != runs + 3; ++run) {
            auto const t0 = Clock::now();
            auto gen = s.Commit(1.0 + run, UsdGenCommitReason::SetTime);
            auto const t1 = Clock::now();
            if (!gen || !gen->device || s.LastDiagnostics().HasErrors()) {
                for (auto const& e : s.LastDiagnostics().errors) std::fprintf(stderr, "%s\n", e.c_str());
                return 1;
            }
            if (run >= 3) session.push_back(ms(t0, t1));
        }
    }
    std::printf("cuda.cook curves=%zu runs=%d direct.median=%.3f ms direct.min=%.3f ms "
                "session.median=%.3f ms session.min=%.3f ms\n",
                curves, runs, Median(direct), *std::min_element(direct.begin(), direct.end()),
                Median(session), *std::min_element(session.begin(), session.end()));
    return 0;
}
