// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "utils/pxrInterop.h"
#include "usdGenCurves/vulkan/csourceVkCapture.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/vulkan/growVkPlan.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "vulkanSurfaceCageFixture.h"
#include <chrono>
#include <cstdio>
#include <filesystem>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"surfaceCage source: %s (%d)\n",#x,__LINE__); return 1; } } while(false)
int main() {
    auto folder = std::filesystem::temp_directory_path() /
        ("usdGen-vulkan-cage-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(folder);
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code error; std::filesystem::remove_all(path,error); } } cleanup{folder};
    auto map = (folder / "owners.ptx").string();
    auto desc = test::SurfaceCageDesc(map);
    uint64_t curves = 0, points = 0;
    UsdGenDiagnostics diagnostics;
    // Compile-time capacity validation must succeed without opening the asset.
    CHECK(ValidateVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),&curves,&points,&diagnostics));
    CHECK(curves == 30 && points == 120 && !std::filesystem::exists(map));
    auto planned = desc;
    planned.executionBackend = usdGenTestUtils::Backend("vulkan");
    CHECK(CompileVulkanSourceWidthPlan(usdGenUsd::Lower(planned),&diagnostics));
    CHECK(!diagnostics.HasErrors() && !std::filesystem::exists(map));
    UsdGenCurveBuffer output; output.totalCurves = 17;
    CHECK(!CaptureVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),curves,points,&output));
    CHECK(output.totalCurves == 17);
    CHECK(test::WriteSurfaceCageOwnerMap(map));
    CHECK(CaptureVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),curves,points,&output,&diagnostics));
    CHECK(output.totalCurves > 3 && output.totalCurves <= curves);
    CHECK(output.totalCvs == output.totalCurves * 4);
    CHECK(output.rootPrim.size() == output.totalCurves && output.rootUV.size() == output.totalCurves);
    CHECK(output.extraCurve.size() == 3);
    for (size_t c = 1; c < output.curveId.size(); ++c) CHECK(output.curveId[c-1] < output.curveId[c]);
    auto saved = output.totalCurves;
    CHECK(!CaptureVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),0,0,&output));
    CHECK(output.totalCurves == saved);
    // Device resampling remains a later stage: capture keeps four authored
    // dense CVs even when the source asks the device for seven CVs.
    desc.nodes[0].params.push_back({TfToken("resampleTo"),VtValue(7),false});
    CHECK(CaptureVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),curves,points,&output));
    CHECK(output.totalCvs == output.totalCurves * 4);
    for (auto& p : desc.surfaces[0].points) p += GfVec3f(.25f,0,.5f);
    std::vector<float> targets;
    CHECK(GatherVulkanSourceRootTargets(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),output,&targets));
    CHECK(targets.size() == size_t(output.totalCurves)*3);
    for (size_t c = 0; c < output.totalCurves; ++c) {
        CHECK(std::abs(targets[c*3] - output.rootUV[c][0] - .25f) < 1.e-6f);
        CHECK(std::abs(targets[c*3+1] - output.rootUV[c][1]) < 1.e-6f);
        CHECK(std::abs(targets[c*3+2] - .5f) < 1.e-6f);
    }
    output.rootPrim[0] = 2;
    auto savedTargets = targets;
    CHECK(!GatherVulkanSourceRootTargets(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),output,&targets));
    CHECK(targets == savedTargets);
    auto triangle = desc;
    triangle.surfaces[0].points = {{0,0,0},{1,0,0},{0,1,0}};
    triangle.surfaces[0].faceVertexCounts = {3};
    triangle.surfaces[0].faceVertexIndices = {0,1,2};
    triangle.surfaces[0].worldMatrix[3][0] = 3;
    triangle.xformMatrix[3][0] = 1;
    UsdGenCurveBuffer oneRoot;
    oneRoot.totalCurves = 1; oneRoot.rootPrim = {0}; oneRoot.rootUV = {{.25f,.5f}};
    CHECK(GatherVulkanSourceRootTargets(usdGenUsd::Lower(triangle),usdGenUsd::Lower(triangle.nodes[0]),oneRoot,&targets));
    CHECK(targets == std::vector<float>({2.25f,.5f,0}));
    desc.nodes[0].params[1].value = VtValue(uint64_t(92));
    CHECK(!ValidateVulkanSurfaceCage(usdGenUsd::Lower(desc),usdGenUsd::Lower(desc.nodes[0]),usdGenUsd::Lower(desc.curveSets[0]),&curves,&points));
    // Disabling one Grow cannot erase Scatter's sibling source branch.
    auto scatter = desc.nodes[0]; scatter.type = TfToken("UsdGenScatter");
    scatter.curves.clear(); scatter.maps.clear(); scatter.mapBindings.clear();
    scatter.params = {{TfToken("density"),VtValue(30.f),false}};
    UsdGenNodeDesc grow; grow.path = SdfPath("/DisabledGrow"); grow.type = TfToken("UsdGenGrow");
    grow.inputs = {scatter.path}; grow.enabled = false;
    auto scattered = desc; scattered.curveSets.clear(); scattered.nodes = {scatter,grow};
    scattered.terminal = grow.path;
    UsdGenGraphSpec normalized;
    CHECK(NormalizeVulkanGeneratorGraph(usdGenUsd::Lower(scattered),&normalized,&diagnostics));
    CHECK(UsdGenCurveSetInputs(normalized).size() == 1 &&
          !UsdGenCurveSetInputView(UsdGenCurveSetInputs(normalized)[0]).points.empty());
    CHECK(!normalized.nodes[1].enabled);
    std::puts("Vulkan surfaceCage source preflight, capture, pose roots and generator isolation PASS");
    return 0;
}
