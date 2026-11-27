// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanCsourceVkParity — Phase-2 CurveSource gap (csourceVk) parity gate.
//
// Covers the four CurveSource options ported to Vulkan source capture:
// resampleTo, useRest-with-rest, surface-free rebind modes, and moved
// source/surface admission.
//
// Legs (mirroring production call sequences, not reimplementations):
//   Vulkan: UsdGenCurveLoader::Load (planExecutor::Capture) ->
//           VulkanPreparedSource::Prepare -> VulkanSourceUpload device upload.
//   CUDA:   UsdGenCaptureCurveRoots -> PrepareCudaSource -> CompactCudaSource ->
//           CudaCurveSource::Set/Finish -> CudaCurveResample -> device readback,
//           plus CudaNamedChannelTopologyCandidate for authored planes.
//   Deform: Vulkan DeformPipeline vs CudaRbfBinding/CudaRbfCurveDeformer on a
//           moved-surface fixture (pose surface worldMatrix[3][1] = -2, the
//           GapRejections spelling); both backends fit/apply the RBF in desc
//           space, so the matrix is admitted without changing any deform input.
//
// Tolerances cite the CUDA tests' own contracts:
//   resampled float channels: |a-b| < 1e-5 (testUsdGenCudaCurveResample.cpp:41)
//   deform cross-compare:     per-component < 4e-4 (testUsdGenCudaDeform.cu:13)
//   deform vs host oracle:    Vulkan leg <= 2e-3 (testUsdGenVulkanDeform.cpp:324)
// Unresampled float channels, integer channels, offsets, stable IDs, and
// root bindings are verbatim copies on both backends and compare bitwise.
// Private root frames have no CUDA device twin (CUDA consumes the same shared
// capture on the host); both legs are checked bitwise against the float-cast
// rows of the shared UsdGenCaptureCurveRoots result.
//
// Host admission tables and loader checks always run. Each device leg runs
// when its backend is available; the test fails only on a run check, and the
// summary banner states explicitly which parity legs were proven.
#include "utils/pxrInterop.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "api/testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "hosts/usdGenUsd/usdGenOpFixture.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenCurves/vulkan/deformPipeline.h"
#include "usdGenCurves/vulkan/deformRbfHost.h"
#include "usdGenCurves/vulkan/csourceVkAdmission.h"

#include "usdGenCurves/curveLoader.h"
#include "usdGenCurves/curveRootCapture.h"
#include "usdGenMath/rbf.h"
#include "pxr/base/gf/range3d.h"

#include "hosts/usdGenUsd/vulkanNativeFixture.h"
#include "internal/usdGenVulkan/vulkanReadbackFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "usdGenCurves/cuda/curveSource.h"
#include "usdGenCurves/cuda/curveResample.h"
#include "usdGenCurves/cuda/namedChannelTopology.h"
#include "usdGenCurves/cuda/deformCurves.h"
#include "usdGenCurves/cuda/rbf.h"
#include "usdGenCurves/cuda/cudaSourceInput.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#ifdef __linux__
#include <unistd.h>
#endif

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

int g_checks = 0, g_fail = 0;
bool g_vulkanDeviceProven = false, g_cudaDeviceProven = false;
bool g_deformProven = false;

#define CHECK(x) do { ++g_checks; if (!(x)) { \
    ++g_fail; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (false)
#define REQUIRE(x) do { ++g_checks; if (!(x)) { \
    ++g_fail; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return false; } } while (false)

constexpr float kResampleTolerance = 1e-5f;
constexpr float kDeformCrossTolerance = 4e-4f;
constexpr float kDeformOracleTolerance = 2e-3f;

float WorstAbs(std::vector<float> const& a, std::vector<float> const& b,
               bool* sizeOk = nullptr) {
    if (a.size() != b.size()) {
        if (sizeOk) *sizeOk = false;
        return std::numeric_limits<float>::infinity();
    }
    if (sizeOk) *sizeOk = true;
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

std::vector<float> AsFloats(std::vector<uint8_t> const& bytes) {
    std::vector<float> out(bytes.size() / sizeof(float), 0.0f);
    if (!bytes.empty())
        std::memcpy(out.data(), bytes.data(),
                    out.size() * sizeof(float));
    return out;
}

template <class T>
std::vector<T> AsValues(std::vector<uint8_t> const& bytes) {
    std::vector<T> out(bytes.size() / sizeof(T), T{});
    if (!bytes.empty())
        std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
}

// ---------------------------------------------------------------- fixtures

UsdGenGraphDesc BaseDesc() {
    UsdGenGraphDesc d;
    d.description = SdfPath("/CsourceVk");
    d.executionBackend = usdGenTestUtils::Backend("vulkan");
    d.defaultWidth = 0.01f;
    return d;
}

void AddWidthTerminal(UsdGenGraphDesc& d, SdfPath const& sourcePath) {
    UsdGenNodeDesc width;
    width.path = SdfPath("/CsourceVk/Ops/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {sourcePath};
    width.params = {{TfToken("width"), VtValue(0.125f), false},
                    {TfToken("replace"), VtValue(false), false}};
    d.nodes.push_back(width);
    d.terminal = width.path;
}

UsdGenSurfaceDesc TriSurface() {
    UsdGenSurfaceDesc s;
    s.path = SdfPath("/CsourceVk/Scalp");
    s.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    s.points = s.restPoints;
    s.faceVertexCounts = {3};
    s.faceVertexIndices = {0, 1, 2};
    s.uv = {{0, 0}, {1, 0}, {0, 1}};
    return s;
}

void AddRaggedRootedCurves(UsdGenGraphDesc& d) {
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/CsourceVk/C3");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.widthsInterpolation = TfToken("vertex");
    hair.curveVertexCounts = {3, 2, 4};
    for (int c = 0; c != 3; ++c)
        for (int i = 0; i != hair.curveVertexCounts[c]; ++i) {
            hair.points.push_back(usdGen::UsdGenVec3f(float(10 * c + i), float(c), 1.0f + i));
            hair.rest.push_back(
                usdGen::UsdGenVec3f(float(100 + 10 * c + i), float(10 + c), 2.0f + i));
            hair.widths.push_back(0.02f + 0.001f * float(hair.widths.size()));
        }
    hair.curveId = {30, 10, 20};
    hair.skinPrim = {0, 99, 0};
    hair.skinPrimUv = {{.3f, .7f}, {.5f, .5f}, {.8f, .2f}};
    UsdGenAuthoredPlaneDesc pointF;
    pointF.name = TfToken("pointFloat");
    pointF.type = UsdGenAuthoredPlaneType::Float32;
    pointF.domain = UsdGenAuthoredPlaneDomain::Point;
    pointF.arity = 2;
    for (int i = 0; i != 9; ++i) {
        pointF.floatValues.push_back(1000.f + i);
        pointF.floatValues.push_back(2000.f + i);
    }
    UsdGenAuthoredPlaneDesc pointI;
    pointI.name = TfToken("pointInt");
    pointI.type = UsdGenAuthoredPlaneType::Int32;
    pointI.domain = UsdGenAuthoredPlaneDomain::Point;
    pointI.arity = 1;
    for (int i = 0; i != 9; ++i) pointI.intValues.push_back(3000 + i);
    UsdGenAuthoredPlaneDesc primF;
    primF.name = TfToken("primFloat");
    primF.type = UsdGenAuthoredPlaneType::Float32;
    primF.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primF.arity = 1;
    primF.floatValues = {30.f, 10.f, 20.f};
    UsdGenAuthoredPlaneDesc primI;
    primI.name = TfToken("primInt");
    primI.type = UsdGenAuthoredPlaneType::Int32;
    primI.domain = UsdGenAuthoredPlaneDomain::Primitive;
    primI.arity = 2;
    primI.intValues = {301, 302, 101, 102, 201, 202};
    UsdGenAuthoredPlaneDesc groomF;
    groomF.name = TfToken("groomFloat");
    groomF.type = UsdGenAuthoredPlaneType::Float32;
    groomF.domain = UsdGenAuthoredPlaneDomain::Groom;
    groomF.arity = 1;
    groomF.floatValues = {7.5f};
    UsdGenAuthoredPlaneDesc groomI;
    groomI.name = TfToken("groomInt");
    groomI.type = UsdGenAuthoredPlaneType::Int32;
    groomI.domain = UsdGenAuthoredPlaneDomain::Groom;
    groomI.arity = 2;
    groomI.intValues = {81, 82};
    hair.authoredPlanes = {pointF, pointI, primF, primI, groomF, groomI};
    d.curveSets.push_back(hair);
}

// Fixture A (resampleTo=8) / B (resampleTo=0): rooted ragged source with an
// invalid middle root (dropped on both backends) and named planes.
UsdGenGraphDesc FixtureResample(int resampleTo) {
    UsdGenGraphDesc d = BaseDesc();
    d.surfaces.push_back(TriSurface());
    AddRaggedRootedCurves(d);
    UsdGenNodeDesc source;
    source.path = SdfPath("/CsourceVk/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {SdfPath("/CsourceVk/C3")};
    source.surfaces = {SdfPath("/CsourceVk/Scalp")};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("resampleTo"), VtValue(resampleTo), false},
                     {TfToken("rebind"), VtValue(TfToken("never")), false}};
    d.nodes.push_back(source);
    AddWidthTerminal(d, source.path);
    return d;
}

// Fixture C: surface-free source with authored frames/bindings + rebind=onError.
// The shared capture takes its repair-free path (no surface needed).
UsdGenGraphDesc FixtureSurfaceFree() {
    UsdGenGraphDesc d = BaseDesc();
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/CsourceVk/C3");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.widthsInterpolation = TfToken("vertex");
    hair.curveVertexCounts = {3, 3};
    hair.points = {{0, 0, 0}, {0, 1, .5f}, {0, 2, 1},
                   {3, 0, 0}, {3, 1, .5f}, {3, 2, 1}};
    hair.rest = hair.points;
    for (int i = 0; i != 6; ++i)
        hair.widths.push_back(0.02f + 0.005f * float(i));
    hair.curveId = {5, 6};
    hair.skinPrim = {0, 0};
    hair.skinPrimUv = {{.2f, .3f}, {.7f, .8f}};
    usdGen::UsdGenMatrix4d rotZ90(1.0);
    rotZ90[0][0] = 0.0;
    rotZ90[0][1] = 1.0;
    rotZ90[1][0] = -1.0;
    rotZ90[1][1] = 0.0;
    hair.rootFrame = {usdGen::UsdGenMatrix4d(1.0), rotZ90};
    d.curveSets.push_back(hair);
    UsdGenNodeDesc source;
    source.path = SdfPath("/CsourceVk/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("resampleTo"), VtValue(0), false},
                     {TfToken("rebind"), VtValue(TfToken("onError")), false}};
    d.nodes.push_back(source);
    AddWidthTerminal(d, source.path);
    return d;
}

// Fixture D: moved source AND moved surface (no deform node — moved sources
// stay rejected under RBF Deform on CUDA only; Vulkan executes them via the
// surfaceVk/rbf baked twins) with a differentiated rest snapshot
// (rest != points, useRest=true).
UsdGenGraphDesc FixtureMoved() {
    UsdGenGraphDesc d = BaseDesc();
    auto surface = TriSurface();
    surface.worldMatrix[0][0] = 0.0;
    surface.worldMatrix[0][1] = 1.0;
    surface.worldMatrix[1][0] = -1.0;
    surface.worldMatrix[1][1] = 0.0;
    surface.worldMatrix[3][0] = 5.0;
    surface.worldMatrix[3][1] = -2.0;
    surface.worldMatrix[3][2] = 7.0;
    d.surfaces.push_back(surface);
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/CsourceVk/C3");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.widthsInterpolation = TfToken("vertex");
    hair.worldMatrix[3][0] = 3.0;
    hair.worldMatrix[3][1] = -4.0;
    hair.worldMatrix[3][2] = 2.0;
    hair.curveVertexCounts = {3, 3};
    hair.points = {{.1f, .2f, .3f}, {.4f, .5f, .6f}, {.7f, .8f, .9f},
                   {1.1f, 1.2f, 1.3f}, {1.4f, 1.5f, 1.6f}, {1.7f, 1.8f, 1.9f}};
    for (auto const& p : hair.points)
        hair.rest.push_back(usdGen::UsdGenVec3f(p[0] + 10.f, p[1] + 20.f, p[2] + 30.f));
    for (int i = 0; i != 6; ++i)
        hair.widths.push_back(0.02f + 0.003f * float(i));
    hair.curveId = {7, 19};
    hair.skinPrim = {0, 0};
    hair.skinPrimUv = {{.7f, .1f}, {.2f, .2f}};
    d.curveSets.push_back(hair);
    UsdGenNodeDesc source;
    source.path = SdfPath("/CsourceVk/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.surfaces = {SdfPath("/CsourceVk/Scalp")};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("resampleTo"), VtValue(0), false},
                     {TfToken("rebind"), VtValue(TfToken("never")), false}};
    d.nodes.push_back(source);
    AddWidthTerminal(d, source.path);
    return d;
}

// Fixture E: deform input with a moved pose surface (GapRejections spelling:
// surface worldMatrix[3][1] = -2). Corner UVs make the root targets
// hand-known pose vertices, independent of any gather formula.
struct DeformFixture {
    UsdGenGraphDesc desc;
    std::vector<float> restSamples;  // 3*n desc-space floats
    std::vector<float> posedSamples; // 3*n desc-space floats
    std::vector<float> targets;      // 3*curves desc-space floats
    std::vector<float> points;       // 3*points rest-domain CVs
    std::vector<uint32_t> offsets;
    std::vector<float> oracle;       // 3*points expected deformed CVs
};

DeformFixture FixtureDeformMovedSurface() {
    DeformFixture fx;
    fx.desc = BaseDesc();
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/CsourceVk/Scalp");
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 1, 0}, {1, 0, 1}};
    // Affine posed motion (row-vector Rz90 plus translation), exactly
    // representable by the cubic RBF on both backends.
    for (auto const& p : surface.restPoints)
        surface.points.push_back(
            usdGen::UsdGenVec3f(-p[2] + 5.f, p[0] - 2.f, p[1] + 7.f));
    surface.faceVertexCounts = {3, 3};
    surface.faceVertexIndices = {0, 1, 2, 3, 4, 5};
    surface.uv = {{0, 0}, {1, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 1}};
    surface.worldMatrix[3][1] = -2.0;
    fx.desc.surfaces.push_back(surface);
    UsdGenCurveSetDesc hair;
    hair.path = SdfPath("/CsourceVk/C3");
    hair.role = UsdGenLane::Chunked;
    hair.curveRole = TfToken("hair");
    hair.type = TfToken("cubic");
    hair.basis = TfToken("bspline");
    hair.wrap = TfToken("pinned");
    hair.widthsInterpolation = TfToken("vertex");
    hair.curveVertexCounts = {3, 3};
    hair.points = {{.2f, .1f, .3f}, {.4f, .2f, .5f}, {.1f, .6f, .2f},
                   {1.2f, .3f, .1f}, {1.1f, .9f, .4f}, {1.4f, .2f, .8f}};
    hair.rest = hair.points;
    hair.widths = VtFloatArray(6, .02f);
    hair.curveId = {3, 4};
    hair.skinPrim = {0, 1};
    hair.skinPrimUv = {{0.f, 0.f}, {1.f, 0.f}};
    fx.desc.curveSets.push_back(hair);
    UsdGenNodeDesc source;
    source.path = SdfPath("/CsourceVk/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {hair.path};
    source.surfaces = {SdfPath("/CsourceVk/Scalp")};
    source.params = {{TfToken("useRest"), VtValue(true), false},
                     {TfToken("idSource"), VtValue(TfToken("primvar")), false},
                     {TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc deform;
    deform.path = SdfPath("/CsourceVk/Ops/deform");
    deform.type = TfToken("UsdGenDeform");
    deform.inputs = {source.path};
    deform.surfaces = {SdfPath("/CsourceVk/Scalp")};
    UsdGenNodeDesc width;
    width.path = SdfPath("/CsourceVk/Ops/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {deform.path};
    width.params = {{TfToken("width"), VtValue(.125f), false},
                    {TfToken("replace"), VtValue(false), false}};
    fx.desc.nodes = {source, deform, width};
    fx.desc.terminal = width.path;

    auto const& rest = fx.desc.surfaces[0].restPoints;
    auto const& posed = fx.desc.surfaces[0].points;
    // Same FPF call the Vulkan plan compiler makes (executionPlan.cpp
    // ValidateDeform): deterministic shared code, identical inputs.
    std::vector<usdGen::UsdGenVec3d> drivers;
    usdGen::UsdGenRange3d extent;
    for (auto const& p : rest) {
        drivers.emplace_back(p);
        extent.UnionWith(drivers.back());
    }
    double const size =
        extent.IsEmpty() ? 0.0 : extent.GetSize().GetLength();
    auto chosen = usdGenMath::rbf::SelectSamples(
        drivers, 100, std::max(1e-12, size * 1e-7));
    for (size_t k : chosen) {
        fx.restSamples.insert(fx.restSamples.end(),
                              {rest[k][0], rest[k][1], rest[k][2]});
        fx.posedSamples.insert(fx.posedSamples.end(),
                               {posed[k][0], posed[k][1], posed[k][2]});
    }
    // Corner-UV gather: curve 0 reads posed vertex 0 of face 0, curve 1
    // reads posed vertex 1 of face 1 — true for the tri gather on both
    // backends without consulting any formula.
    auto const& idx = fx.desc.surfaces[0].faceVertexIndices;
    size_t const v0 = size_t(idx[0]);
    size_t const v1 = size_t(idx[3 + 1]);
    fx.targets = {posed[v0][0], posed[v0][1], posed[v0][2],
                  posed[v1][0], posed[v1][1], posed[v1][2]};
    for (auto const& p : hair.points)
        fx.points.insert(fx.points.end(), {p[0], p[1], p[2]});
    fx.offsets = {0, 3, 6};
    // Oracle: affine warp plus per-curve root-lock correction, in doubles.
    auto warp = [](double x, double y, double z) {
        return std::array<double, 3>({-z + 5.0, x - 2.0, y + 7.0});
    };
    for (uint32_t curve = 0; curve != 2; ++curve) {
        uint32_t const root = curve * 3;
        auto wRoot = warp(fx.points[3 * root], fx.points[3 * root + 1],
                          fx.points[3 * root + 2]);
        double const cx = double(fx.targets[3 * curve]) - wRoot[0];
        double const cy = double(fx.targets[3 * curve + 1]) - wRoot[1];
        double const cz = double(fx.targets[3 * curve + 2]) - wRoot[2];
        for (uint32_t i = 0; i != 3; ++i) {
            auto w = warp(fx.points[3 * (root + i)], fx.points[3 * (root + i) + 1],
                          fx.points[3 * (root + i) + 2]);
            fx.oracle.insert(fx.oracle.end(),
                             {float(w[0] + cx), float(w[1] + cy), float(w[2] + cz)});
        }
    }
    return fx;
}

// ------------------------------------------------- host admission tables

void HostAdmissionTables() {
    // resampleTo boundary table (CUDA: loader/prep/kernel all agree).
    struct ResampleCase {
        VtValue value;
        size_t curves;
        CsourceVkResampleStatus want;
    };
    std::vector<ResampleCase> const cases{
        {VtValue(0), 3, CsourceVkResampleStatus::Ok},
        {VtValue(2), 3, CsourceVkResampleStatus::Ok},
        {VtValue(8), 1, CsourceVkResampleStatus::Ok},
        {VtValue(8), 0, CsourceVkResampleStatus::Ok},
        {VtValue(-1), 3, CsourceVkResampleStatus::Negative},
        {VtValue(-100), 3, CsourceVkResampleStatus::Negative},
        {VtValue(1), 3, CsourceVkResampleStatus::OneCV},
        {VtValue(2.5f), 3, CsourceVkResampleStatus::NotAnInteger},
        {VtValue(TfToken("8")), 3, CsourceVkResampleStatus::NotAnInteger},
        {VtValue(std::numeric_limits<int>::max()), 2,
         CsourceVkResampleStatus::Overflow},
        {VtValue(8), size_t(std::numeric_limits<uint32_t>::max()) + 1,
         CsourceVkResampleStatus::Overflow},
    };
    for (auto const& c : cases) {
        int target = -999;
        std::string reason;
        auto got = CsourceVkValidateResampleTarget(c.value, c.curves, &target,
                                                   &reason);
        CHECK(got == c.want);
        CHECK((got == CsourceVkResampleStatus::Ok) == reason.empty());
        if (c.value.IsHolding<int>())
            CHECK(target == c.value.UncheckedGet<int>());
    }

    // Effective captured points feeding the plan memory estimate.
    {
        uint64_t eff = 0;
        CHECK(CsourceVkEffectivePointCount(9, 3, 8, &eff) && eff == 24);
        CHECK(CsourceVkEffectivePointCount(9, 3, 0, &eff) && eff == 9);
        CHECK(CsourceVkEffectivePointCount(0, 0, 8, &eff) && eff == 0);
        CHECK(!CsourceVkEffectivePointCount(9, 3, 1, &eff));
        CHECK(!CsourceVkEffectivePointCount(9, 3, -2, &eff));
        CHECK(!CsourceVkEffectivePointCount(
            9, 2, std::numeric_limits<int>::max(), &eff));
    }

    // Authored-plane bytes under resample (domain accounting).
    {
        uint64_t bytes = 0;
        using D = UsdGenAuthoredPlaneDomain;
        CHECK(CsourceVkAuthoredPlaneBytes(D::Point, 2, 3, 24, &bytes) &&
              bytes == 24 * 2 * 4);
        CHECK(CsourceVkAuthoredPlaneBytes(D::Point, 1, 2, 16, &bytes) &&
              bytes == 16 * 4);
        CHECK(CsourceVkAuthoredPlaneBytes(D::Primitive, 2, 3, 24, &bytes) &&
              bytes == 3 * 2 * 4);
        CHECK(CsourceVkAuthoredPlaneBytes(D::Groom, 2, 3, 24, &bytes) &&
              bytes == 8);
        // resampleTo == 0 reproduces descriptor sizes exactly.
        CHECK(CsourceVkAuthoredPlaneBytes(D::Point, 2, 3, 9, &bytes) &&
              bytes == 9 * 2 * 4);
        CHECK(!CsourceVkAuthoredPlaneBytes(D::Point, 0, 3, 24, &bytes));
        CHECK(!CsourceVkAuthoredPlaneBytes(D::Point, 17, 3, 24, &bytes));
        CHECK(!CsourceVkAuthoredPlaneBytes(D(99), 1, 3, 24, &bytes));
    }

    // Surface-free rebind admission (shared-capture outcome table).
    {
        CHECK(CsourceVkSurfaceFreeRebindAdmitted(TfToken("never"), 1, 0));
        CHECK(CsourceVkSurfaceFreeRebindAdmitted(TfToken("never"), 1, 1));
        CHECK(CsourceVkSurfaceFreeRebindAdmitted(TfToken("onError"), 1, 1));
        CHECK(!CsourceVkSurfaceFreeRebindAdmitted(TfToken("onError"), 1, 0));
        CHECK(!CsourceVkSurfaceFreeRebindAdmitted(TfToken("onError"), 2, 1));
        CHECK(!CsourceVkSurfaceFreeRebindAdmitted(TfToken("always"), 1, 1));
        CHECK(!CsourceVkSurfaceFreeRebindAdmitted(TfToken("always"), 1, 0));
        CHECK(CsourceVkSurfaceFreeRebindAdmitted(TfToken("always"), 0, 0));
        CHECK(CsourceVkSurfaceFreeRebindAdmitted(TfToken("onError"), 0, 0));
    }

    // Deform transform admission (CUDA rule: source must stay identity;
    // surfaceVk consults both matrices, and CUDA execution rejects moved
    // surfaces in PrepareCudaSurface — only ValidateCudaGraph skips it).
    {
        std::string reason;
        CHECK(CsourceVkDeformSourceAdmitted(usdGen::UsdGenMatrix4d(1.0), &reason) &&
              reason.empty());
        usdGen::UsdGenMatrix4d moved(1.0);
        moved[3][0] = 5.0;
        CHECK(!CsourceVkDeformSourceAdmitted(moved, &reason));
        CHECK(reason == "Deform requires identity source transforms");
        CHECK(!CsourceVkDeformSourceAdmitted(moved, nullptr));
    }
    std::puts("csourceVk: host admission tables PASS");
}

// ------------------------------------------------------- loader checks

bool LoadDesc(UsdGenGraphDesc const& d, UsdGenCurveBuffer* out) {
    if (!out) return false;
    usdGen::UsdGenGraphSpec const lowered = usdGenUsd::Lower(d);
    UsdGenParamView params{&lowered, &lowered.nodes[0]};
    UsdGenCaptureContext ctx;
    ctx.desc = &lowered;
    ctx.params = &params;
    ctx.seed = static_cast<uint32_t>(d.nodes[0].seed);
    UsdGenDiagnostics diag;
    UsdGenCurveBuffer result;
    if (!UsdGenCurveLoader::Load(ctx, UsdGenCurveSetInputs(lowered)[0], &result, &diag))
        return false;
    *out = std::move(result);
    return true;
}

void LoaderChecks() {
    // Fixture A: drop + resample to 8 -> uniform 2x8, canonical {20,30}.
    {
        UsdGenCurveBuffer out;
        CHECK(LoadDesc(FixtureResample(8), &out));
        CHECK(out.totalCurves == 2 && out.totalCvs == 16);
        CHECK(out.curveId == VtArray<uint64_t>({20, 30}));
        CHECK(out.cvOffsets.empty());
        CHECK(out.px.size() == 16 && out.rest.size() == 16 &&
              out.width.size() == 16 && out.hairT.size() == 16);
        CHECK(out.rootPrim == VtIntArray({0, 0}));
        CHECK(out.rootUV.size() == 2);
        CHECK(out.rootT.size() == 2 && out.rootB.size() == 2 &&
              out.rootN.size() == 2);
        CHECK(out.extraCv.size() == 2 && out.extraCurve.size() == 4);
        // Resampled hairT is canonical per curve.
        CHECK(out.hairT[0] == 0.f && out.hairT[7] == 1.f &&
              out.hairT[8] == 0.f && out.hairT[15] == 1.f);
    }
    // Fixture B: drop without resample -> ragged {0,4,7}.
    {
        UsdGenCurveBuffer out;
        CHECK(LoadDesc(FixtureResample(0), &out));
        CHECK(out.totalCurves == 2 && out.totalCvs == 7);
        CHECK(out.curveId == VtArray<uint64_t>({20, 30}));
        CHECK(out.cvOffsets == VtIntArray({0, 4, 7}));
        CHECK(out.extraCv.size() == 2 && out.extraCurve.size() == 4);
    }
    // Fixture C: surface-free onError with authored frames executes.
    {
        UsdGenCurveBuffer out;
        CHECK(LoadDesc(FixtureSurfaceFree(), &out));
        CHECK(out.totalCurves == 2 && out.totalCvs == 6);
        CHECK(out.rootPrim == VtIntArray({0, 0}));
        CHECK(out.rootUV ==
              usdGen::UsdGenVec2fArray({usdGen::UsdGenVec2f(.2f, .3f), usdGen::UsdGenVec2f(.7f, .8f)}));
        CHECK(out.rootT.size() == 2);
        // Authored frames are transported exactly (identity, then Rz90).
        CHECK(out.rootT[0] == usdGen::UsdGenVec3f(1, 0, 0) &&
              out.rootB[0] == usdGen::UsdGenVec3f(0, 1, 0) &&
              out.rootN[0] == usdGen::UsdGenVec3f(0, 0, 1));
        CHECK(out.rootT[1] == usdGen::UsdGenVec3f(0, 1, 0) &&
              out.rootB[1] == usdGen::UsdGenVec3f(-1, 0, 0) &&
              out.rootN[1] == usdGen::UsdGenVec3f(0, 0, 1));
    }
    // Fixture D: moved source/surface with differentiated rest executes.
    {
        auto d = FixtureMoved();
        UsdGenCurveBuffer out;
        CHECK(LoadDesc(d, &out));
        CHECK(out.totalCurves == 2 && out.totalCvs == 6);
        CHECK(out.rootT.size() == 2);
        // Points and rest stay distinct channels (useRest-with-rest).
        CHECK(out.px[0] == d.curveSets[0].points[0][0]);
        CHECK(out.rest[0] == d.curveSets[0].rest[0]);
        CHECK(out.rest[0] != d.curveSets[0].points[0]);
    }
    // Keep-reject set: configs CUDA itself cannot execute still fail
    // capture on the shared loader (compile rejections stay valid).
    {
        auto noFrames = FixtureSurfaceFree();
        noFrames.curveSets[0].rootFrame.clear();
        UsdGenCurveBuffer out;
        CHECK(!LoadDesc(noFrames, &out));
        auto always = FixtureSurfaceFree();
        for (auto& p : always.nodes[0].params)
            if (p.name == TfToken("rebind")) p.value = VtValue(TfToken("always"));
        CHECK(!LoadDesc(always, &out));
        auto noRest = FixtureMoved();
        noRest.curveSets[0].rest.clear();
        CHECK(!LoadDesc(noRest, &out));
        auto badTarget = FixtureResample(1);
        CHECK(!LoadDesc(badTarget, &out));
        auto negTarget = FixtureResample(-2);
        CHECK(!LoadDesc(negTarget, &out));
    }
    std::puts("csourceVk: loader checks PASS");
}

// ------------------------------------------------------- snapshots

struct NamedSnapshot {
    std::string name;
    UsdGenDeviceValueType type = UsdGenDeviceValueType::Float32;
    UsdGenDeviceDomain domain = UsdGenDeviceDomain::Point;
    uint64_t elements = 0;
    uint32_t arity = 0;
    std::vector<uint8_t> bytes;
};

struct SourceSnapshot {
    uint32_t curves = 0, pointCount = 0;
    std::vector<float> points;   // 3N interleaved
    std::vector<float> rest;     // 3N interleaved
    std::vector<float> widths;   // N
    std::vector<float> hairT;    // N
    std::vector<uint32_t> offsets;
    std::vector<uint64_t> ids;
    std::vector<int32_t> rootPrim;
    std::vector<float> rootUV;   // 2C interleaved
    std::vector<float> frameT, frameB, frameN; // 3C each
    std::map<std::string, NamedSnapshot> named;
};

bool HostSnapshot(UsdGenCurveBuffer const& b, SourceSnapshot* out) {
    if (!out) return false;
    SourceSnapshot s;
    s.curves = b.totalCurves;
    s.pointCount = b.totalCvs;
    if (b.px.size() != b.totalCvs || b.py.size() != b.totalCvs ||
        b.pz.size() != b.totalCvs || b.curveId.size() != b.totalCurves)
        return false;
    for (uint32_t i = 0; i != b.totalCvs; ++i) {
        s.points.insert(s.points.end(), {b.px[i], b.py[i], b.pz[i]});
        if (!b.rest.empty())
            s.rest.insert(s.rest.end(),
                          {b.rest[i][0], b.rest[i][1], b.rest[i][2]});
    }
    if (!b.width.empty())
        s.widths.assign(b.width.begin(), b.width.end());
    if (!b.hairT.empty())
        s.hairT.assign(b.hairT.begin(), b.hairT.end());
    if (!b.rootUV.empty())
        for (auto const& uv : b.rootUV)
            s.rootUV.insert(s.rootUV.end(), {uv[0], uv[1]});
    if (!b.rootPrim.empty())
        s.rootPrim.assign(b.rootPrim.begin(), b.rootPrim.end());
    s.ids.assign(b.curveId.begin(), b.curveId.end());
    if (b.cvOffsets.empty()) {
        if (b.totalCurves && b.totalCvs % b.totalCurves) return false;
        uint32_t const per = b.totalCurves ? b.totalCvs / b.totalCurves : 0;
        for (uint32_t c = 0; c <= b.totalCurves; ++c)
            s.offsets.push_back(c * per);
    } else {
        if (b.cvOffsets.size() != size_t(b.totalCurves) + 1) return false;
        for (auto v : b.cvOffsets) {
            if (v < 0) return false;
            s.offsets.push_back(uint32_t(v));
        }
    }
    if (!b.rootT.empty()) {
        if (b.rootT.size() != b.totalCurves ||
            b.rootB.size() != b.totalCurves ||
            b.rootN.size() != b.totalCurves)
            return false;
        for (uint32_t c = 0; c != b.totalCurves; ++c) {
            s.frameT.insert(s.frameT.end(),
                            {b.rootT[c][0], b.rootT[c][1], b.rootT[c][2]});
            s.frameB.insert(s.frameB.end(),
                            {b.rootB[c][0], b.rootB[c][1], b.rootB[c][2]});
            s.frameN.insert(s.frameN.end(),
                            {b.rootN[c][0], b.rootN[c][1], b.rootN[c][2]});
        }
    }
    auto plane = [&](UsdGenPlane const& p, bool cv) {
        NamedSnapshot n;
        n.name = p.name.GetString();
        n.type = p.type == TfToken("float") ? UsdGenDeviceValueType::Float32
                                            : UsdGenDeviceValueType::Int32;
        n.domain = p.interpolation == TfToken("vertex")
            ? UsdGenDeviceDomain::Point
            : p.interpolation == TfToken("uniform") ? UsdGenDeviceDomain::Primitive
                                                   : UsdGenDeviceDomain::Groom;
        (void)cv;
        n.elements = n.domain == UsdGenDeviceDomain::Point ? b.totalCvs
            : n.domain == UsdGenDeviceDomain::Primitive   ? b.totalCurves
                                                          : 1;
        n.arity = p.arity;
        if (n.type == UsdGenDeviceValueType::Float32) {
            n.bytes.resize(p.f.size() * 4);
            if (!p.f.empty())
                std::memcpy(n.bytes.data(), p.f.cdata(), n.bytes.size());
        } else {
            n.bytes.resize(p.i.size() * 4);
            if (!p.i.empty())
                std::memcpy(n.bytes.data(), p.i.cdata(), n.bytes.size());
        }
        s.named[n.name] = std::move(n);
    };
    for (auto const& p : b.extraCv) plane(p, true);
    for (auto const& p : b.extraCurve) plane(p, false);
    *out = std::move(s);
    return true;
}

// Float-cast rows of the shared capture result in canonical id order: the
// frame oracle both backends must reproduce bitwise.
bool FrameOracle(UsdGenCurveSetDesc const& curves,
                 UsdGenCurveRootCaptureResult const& captured,
                 std::vector<float>* t, std::vector<float>* b,
                 std::vector<float>* n) {
    if (!t || !b || !n) return false;
    size_t const count = curves.curveVertexCounts.size();
    if (captured.frames.size() != count || captured.valid.size() != count)
        return false;
    std::vector<size_t> order;
    for (size_t c = 0; c != count; ++c)
        if (captured.valid[c]) order.push_back(c);
    auto idOf = [&](size_t c) {
        return !curves.curveId.empty() &&
                curves.curveId.size() == count
            ? curves.curveId[c]
            : uint64_t(c);
    };
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t c) { return idOf(a) < idOf(c); });
    for (size_t c : order) {
        usdGen::UsdGenMatrix4d const& f = captured.frames[c];
        t->insert(t->end(), {float(f[0][0]), float(f[0][1]), float(f[0][2])});
        b->insert(b->end(), {float(f[1][0]), float(f[1][1]), float(f[1][2])});
        n->insert(n->end(), {float(f[2][0]), float(f[2][1]), float(f[2][2])});
    }
    return true;
}

// ------------------------------------------------------- Vulkan leg

std::shared_ptr<NativeOwner> CreateNativeRetry(bool* unavailable) {
    VkPhysicalDeviceFeatures fp64{};
    fp64.shaderFloat64 = VK_TRUE;
    return CreateTimelineNative(unavailable, &fp64);
}

struct VulkanDevice {
    std::shared_ptr<NativeOwner> native;
    std::shared_ptr<DeviceContext> context;
    bool shaderFloat64 = false;
};

bool InitVulkan(VulkanDevice* dev, bool* unavailable) {
    if (!dev || !unavailable) return false;
    *unavailable = false;
    dev->native = CreateNativeRetry(unavailable);
    if (!dev->native) return false;
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(dev->native->physical, &feats);
    dev->shaderFloat64 = feats.shaderFloat64 == VK_TRUE;
    DeviceContext::CreateInfo ci;
    ci.instance = dev->native->instance;
    ci.physicalDevice = dev->native->physical;
    ci.device = dev->native->device;
    ci.computeQueue = dev->native->queue;
    ci.computeQueueFamily = dev->native->family;
    ci.physicalIndex = dev->native->physicalIndex;
    ci.resourceDeviceId = 7041;
    ci.nativeLifetime = dev->native;
    ci.resources = {size_t{8} << 20, 0};
    ci.shaderFloat64Enabled = dev->shaderFloat64;
    ci.timelineSemaphoreEnabled = true;
    VkResult status = VK_ERROR_UNKNOWN;
    dev->context = DeviceContext::Create(ci, &status);
    if (!dev->context || status != VK_SUCCESS) {
        dev->native.reset();
        dev->context.reset();
        return false;
    }
    return true;
}

bool ProveNative(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS)
        return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) ==
        VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE,
                        10000000000ull) == VK_SUCCESS;
}

bool ReadPlane(VulkanDevice const& dev,
               std::shared_ptr<const VulkanSourceGeneration> const& gen,
               std::string const& name, std::vector<uint8_t>* out,
               bool priv = false) {
    auto const& planes = priv ? gen->sourceFrames() : gen->planes();
    for (auto const& p : planes) {
        if (p.metadata.name != name) continue;
        if (p.bytes == 0) {
            out->clear();
            return true;
        }
        auto owner = priv ? gen->SourceFrameOwner(TfToken(name))
                          : gen->PlaneOwner(TfToken(name));
        if (p.buffer == VK_NULL_HANDLE || !owner) return false;
        return ReadVulkanBytes(dev.native, dev.context, p.buffer,
                               size_t(p.bytes), owner, out);
    }
    return false;
}

bool VulkanSourceSnapshot(VulkanDevice const& dev, UsdGenGraphDesc const& d,
                           SourceSnapshot* out, UsdGenCurveBuffer* loaderOut) {
    if (!out) return false;
    UsdGenCurveBuffer loaded;
    if (!LoadDesc(d, &loaded)) return false;
    if (loaderOut) *loaderOut = loaded;
    // Session revisions are authoritative (planExecutor::Capture): stamp the
    // buffer before preparation so geometry metadata agrees with it.
    loaded.topologyVersion = 11;
    loaded.valueVersion = 12;
    VulkanSourcePrepareInfo info;
    info.source = loaded;
    info.geometry.topologyVersion = 11;
    info.geometry.valueVersion = 12;
    info.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic,
                                  UsdGenCurveLayoutBasis::BSpline,
                                  UsdGenCurveLayoutWrap::Pinned);
    bool useRest = true;
    for (auto const& p : d.nodes[0].params)
        if (p.name == TfToken("useRest") && p.value.IsHolding<bool>())
            useRest = p.value.UncheckedGet<bool>();
    info.geometry.alreadyDeformed = !useRest;
    std::string reason;
    auto prepared = VulkanPreparedSource::Prepare(std::move(info), &reason);
    if (!prepared) {
        std::fprintf(stderr, "csourceVk prepare: %s\n", reason.c_str());
        return false;
    }
    VkResult status = VK_ERROR_UNKNOWN;
    auto upload =
        VulkanSourceUpload::Create(dev.context, prepared, &status);
    if (!upload || status != VK_SUCCESS) return false;
    if (upload->Submit() != SourceGenerationStatus::Submitted) return false;
    bool ready = false;
    for (int i = 0; i != 100000; ++i) {
        auto s = upload->Poll();
        if (s == SourceGenerationStatus::Ready) {
            ready = true;
            break;
        }
        if (s != SourceGenerationStatus::NotReady) return false;
        if (i % 1000 == 0 && !ProveNative(dev.native)) return false;
    }
    if (!ready) return false;
    auto gen = upload->TakeReady();
    if (!gen) return false;
    SourceSnapshot s;
    s.curves = gen->curveCount();
    s.pointCount = gen->pointCount();
    std::vector<uint8_t> bytes;
    if (!ReadPlane(dev, gen, "points", &bytes)) return false;
    s.points = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "rest", &bytes)) return false;
    s.rest = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "width", &bytes)) return false;
    s.widths = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "hairT", &bytes)) return false;
    s.hairT = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "curveOffsets", &bytes)) return false;
    s.offsets = AsValues<uint32_t>(bytes);
    if (!ReadPlane(dev, gen, "stableIds", &bytes)) return false;
    s.ids = AsValues<uint64_t>(bytes);
    if (!ReadPlane(dev, gen, "rootPrim", &bytes)) return false;
    s.rootPrim = AsValues<int32_t>(bytes);
    if (!ReadPlane(dev, gen, "rootUV", &bytes)) return false;
    s.rootUV = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "sourceRootT", &bytes, true)) return false;
    s.frameT = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "sourceRootB", &bytes, true)) return false;
    s.frameB = AsFloats(bytes);
    if (!ReadPlane(dev, gen, "sourceRootN", &bytes, true)) return false;
    s.frameN = AsFloats(bytes);
    for (auto const& plane : gen->planes()) {
        if (plane.metadata.semantic !=
            UsdGenDeviceChannelSemantic::Generic)
            continue;
        NamedSnapshot n;
        n.name = plane.metadata.name.GetString();
        n.type = plane.metadata.type;
        n.domain = plane.metadata.domain;
        n.elements = plane.metadata.elementCount;
        n.arity = plane.metadata.arity;
        if (!ReadPlane(dev, gen, n.name, &n.bytes)) return false;
        s.named[n.name] = std::move(n);
    }
    *out = std::move(s);
    return true;
}

// ------------------------------------------------------- CUDA leg

#ifdef USDGEN_ENABLE_CUDA
struct AsyncStatus {
    std::atomic<int> value{INT_MIN};
    std::atomic<unsigned> calls{0};
};
void AsyncDone(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* result = static_cast<AsyncStatus*>(data);
    result->value.store(static_cast<int>(status), std::memory_order_release);
    result->calls.fetch_add(1, std::memory_order_release);
}

template <class T>
bool DownloadView(gpu::DeviceView<const T> view, std::vector<T>* out) {
    out->assign(view.size, T{});
    if (!view.size) return true;
    if (!view.data) return false;
    return cudaMemcpy(out->data(), view.data, view.size * sizeof(T),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
}

// Canonical authored-plane gather mirroring
// ExecutionState::UploadAuthoredPlanes (id-ordered, post-compaction).
bool GatherPlaneBytes(UsdGenCurveSetDesc const& curves,
                      CudaSourcePrepared const& prepared, bool indexIds,
                      UsdGenAuthoredPlaneDesc const& authored,
                      std::vector<uint32_t>* staging) {
    if (!staging) return false;
    std::map<uint64_t, size_t> sourceCurveById;
    for (size_t c = 0; c != curves.curveVertexCounts.size(); ++c) {
        uint64_t const id =
            indexIds || curves.curveId.empty() ? uint64_t(c)
                                               : curves.curveId[c];
        sourceCurveById.emplace(id, c);
    }
    std::vector<size_t> sourceOffsets(curves.curveVertexCounts.size() + 1, 0);
    for (size_t c = 0; c != curves.curveVertexCounts.size(); ++c)
        sourceOffsets[c + 1] =
            sourceOffsets[c] + size_t(curves.curveVertexCounts[c]);
    staging->clear();
    auto append = [&](size_t valueIndex) {
        uint32_t bits = 0;
        if (authored.type == UsdGenAuthoredPlaneType::Float32) {
            float const v = authored.floatValues[valueIndex];
            std::memcpy(&bits, &v, sizeof(bits));
        } else {
            int32_t const v = authored.intValues[valueIndex];
            std::memcpy(&bits, &v, sizeof(bits));
        }
        staging->push_back(bits);
    };
    if (authored.domain == UsdGenAuthoredPlaneDomain::Groom) {
        for (size_t lane = 0; lane != authored.arity; ++lane) append(lane);
        return true;
    }
    for (uint64_t id : prepared.curveId) {
        auto const found = sourceCurveById.find(id);
        if (found == sourceCurveById.end()) return false;
        size_t const curve = found->second;
        size_t const first =
            authored.domain == UsdGenAuthoredPlaneDomain::Point
            ? sourceOffsets[curve]
            : curve;
        size_t const count =
            authored.domain == UsdGenAuthoredPlaneDomain::Point
            ? size_t(curves.curveVertexCounts[curve])
            : 1;
        for (size_t e = 0; e != count; ++e)
            for (size_t lane = 0; lane != authored.arity; ++lane)
                append((first + e) * authored.arity + lane);
    }
    return true;
}

struct CudaSourceLeg {
    std::unique_ptr<gpu::CudaCurveSource> source;
    std::unique_ptr<gpu::CudaCurveResample> resampled;
    std::unique_ptr<gpu::CudaNamedChannelTopologyCandidate> namedTopo;
    std::vector<std::unique_ptr<gpu::DeviceBuffer<unsigned char>>> planeBytes;
    std::vector<gpu::CudaNamedChannelPlane> rebuilt;
    CudaSourcePrepared prepared;
    UsdGenCurveRootCaptureResult captured;
    int resampleTo = 0;

    bool Run(UsdGenGraphDesc const& d) {
        // The Vulkan leg runs first; re-select the CUDA device defensively.
        if (cudaSetDevice(0) != cudaSuccess) return false;
        auto const& node = d.nodes[0];
        auto const& curves = d.curveSets[0];
        usdGenTest::NodeParams params(node);
        TfToken const rebind =
            params.GetToken(TfToken("rebind"), TfToken("onError"));
        resampleTo = params.GetInt(TfToken("resampleTo"), 0);
        bool const useRest = params.GetBool(TfToken("useRest"), true);
        TfToken const idSource =
            params.GetToken(TfToken("idSource"), TfToken("primvar"));
        UsdGenSurfaceDesc const* surface = nullptr;
        if (!node.surfaces.empty()) {
            for (auto const& s : d.surfaces)
                if (s.path == node.surfaces.front()) surface = &s;
            if (!surface) return false;
        }
        std::string rootError;
        if (!UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves), (surface ? usdGenUsd::Addr(usdGenUsd::View(*surface)) : nullptr), rebind, &captured,
                                     &rootError)) {
            std::fprintf(stderr, "csourceVk cuda capture: %s\n",
                         rootError.c_str());
            return false;
        }
        CudaSourcePreparationInput input;
        input.curveVertexCounts.assign(curves.curveVertexCounts.begin(),
                                       curves.curveVertexCounts.end());
        for (auto const& p : curves.points)
            input.points.push_back(::make_float3(p[0], p[1], p[2]));
        for (auto const& p : curves.rest)
            input.rest.push_back(::make_float3(p[0], p[1], p[2]));
        input.widths.assign(curves.widths.begin(), curves.widths.end());
        input.curveId.assign(curves.curveId.begin(), curves.curveId.end());
        input.rootPrim.assign(captured.rootPrim.begin(),
                              captured.rootPrim.end());
        for (auto const& uv : captured.rootUV)
            input.rootUV.push_back(::make_float2(uv[0], uv[1]));
        for (auto const& f : captured.frames) {
            std::array<double, 16> raw{};
            for (int r = 0; r != 4; ++r)
                for (int c = 0; c != 4; ++c)
                    raw[size_t(r) * 4 + size_t(c)] = f[r][c];
            input.rootFrames.push_back(raw);
        }
        CudaSourcePreparationOptions options;
        options.defaultWidth = d.defaultWidth;
        options.useRest = useRest;
        options.idSource = idSource == TfToken("index")
            ? CudaSourceIdSource::Index
            : CudaSourceIdSource::Primvar;
        options.expectedEpoch =
            params.GetVtValue(TfToken("expectEpoch"), VtValue(std::string()))
                .IsHolding<std::string>()
            ? params.GetVtValue(TfToken("expectEpoch"),
                                VtValue(std::string()))
                  .UncheckedGet<std::string>()
            : std::string();
        options.actualEpoch = curves.frozenEpoch;
        options.resampleTo = resampleTo;
        options.rebind = "never"; // shared capture resolved policy
        options.hasRootFrame = !input.rootFrames.empty();
        std::vector<std::string> messages;
        if (PrepareCudaSource(input, options, &prepared, &messages) !=
            CudaSourcePreparationStatus::Ok) {
            for (auto const& m : messages)
                std::fprintf(stderr, "csourceVk cuda prep: %s\n", m.c_str());
            return false;
        }
        if (captured.dropped) {
            std::vector<uint64_t> retained;
            for (size_t c = 0; c != captured.valid.size(); ++c) {
                if (!captured.valid[c]) continue;
                retained.push_back(options.idSource ==
                            CudaSourceIdSource::Index ||
                        curves.curveId.empty()
                    ? uint64_t(c)
                    : curves.curveId[c]);
            }
            if (!CompactCudaSource(&prepared, retained)) return false;
        }
        source = std::make_unique<gpu::CudaCurveSource>();
        if (source->Set(prepared.Input(), nullptr) !=
                gpu::CurveSourceStatus::Ok ||
            source->Finish(nullptr) != gpu::CurveSourceStatus::Ok)
            return false;
        if (resampleTo) {
            resampled = std::make_unique<gpu::CudaCurveResample>();
            if (resampled->Apply(source->view(), source->hairT(),
                                 source->rootPrim(), source->rootUV(),
                                 resampleTo, nullptr) !=
                    gpu::CurveResampleStatus::Ok ||
                resampled->Finish(nullptr) != gpu::CurveResampleStatus::Ok)
                return false;
        }
        if (!curves.authoredPlanes.empty()) {
            bool const indexIds =
                options.idSource == CudaSourceIdSource::Index;
            std::vector<gpu::CudaNamedChannelTopologyCandidate::RawInputPlane>
                inputs;
            for (auto const& authored : curves.authoredPlanes) {
                uint64_t const elements =
                    authored.domain == UsdGenAuthoredPlaneDomain::Point
                    ? uint64_t(prepared.points.size())
                    : authored.domain == UsdGenAuthoredPlaneDomain::Primitive
                        ? uint64_t(prepared.curveVertexCounts.size())
                        : 1;
                size_t const bytes =
                    size_t(elements * authored.arity) * sizeof(uint32_t);
                std::vector<uint32_t> staging;
                if (!GatherPlaneBytes(curves, prepared, indexIds, authored,
                                      &staging) ||
                    staging.size() * sizeof(uint32_t) != bytes)
                    return false;
                auto buffer =
                    std::make_unique<gpu::DeviceBuffer<unsigned char>>();
                if (buffer->reset(bytes) != cudaSuccess) return false;
                if (bytes &&
                    cudaMemcpy(buffer->data(), staging.data(), bytes,
                               cudaMemcpyHostToDevice) != cudaSuccess)
                    return false;
                gpu::CudaNamedChannelTopologyCandidate::RawInputPlane in;
                in.metadata = {
                    authored.name,
                    authored.type == UsdGenAuthoredPlaneType::Float32
                        ? UsdGenDeviceValueType::Float32
                        : UsdGenDeviceValueType::Int32,
                    authored.domain == UsdGenAuthoredPlaneDomain::Point
                        ? UsdGenDeviceDomain::Point
                        : authored.domain ==
                                UsdGenAuthoredPlaneDomain::Primitive
                            ? UsdGenDeviceDomain::Primitive
                            : UsdGenDeviceDomain::Groom,
                    elements, authored.arity,
                    uint32_t(authored.arity * sizeof(uint32_t)), true,
                    UsdGenDeviceChannelSemantic::Generic};
                in.bytes = {buffer->data(), buffer->size()};
                in.readyBuffer = nullptr; // sync upload already visible
                planeBytes.push_back(std::move(buffer));
                inputs.push_back(in);
            }
            if (resampleTo) {
                namedTopo = std::make_unique<
                    gpu::CudaNamedChannelTopologyCandidate>();
                std::string reason;
                if (namedTopo->BeginRaw(
                        source->view(), resampled->view(), std::move(inputs),
                        nullptr, gpu::CudaNamedChannelTopologyMode::Resample,
                        {}, &reason) != cudaSuccess) {
                    std::fprintf(stderr, "csourceVk named BeginRaw: %s\n",
                                 reason.c_str());
                    return false;
                }
                AsyncStatus status;
                if (namedTopo->FinishAsync(AsyncDone, &status) !=
                        cudaSuccess ||
                    cudaStreamSynchronize(nullptr) != cudaSuccess ||
                    status.calls.load(std::memory_order_acquire) != 1 ||
                    status.value.load(std::memory_order_acquire) !=
                        cudaSuccess ||
                    !namedTopo->CommitPlanes(
                        cudaError_t(status.value.load(
                            std::memory_order_acquire)),
                        &rebuilt, &reason)) {
                    std::fprintf(stderr, "csourceVk named commit: %s\n",
                                 reason.c_str());
                    return false;
                }
            }
        }
        return true;
    }

    bool Snapshot(SourceSnapshot* out) {
        if (!out || !source) return false;
        bool const rw = resampled != nullptr;
        auto view = rw ? resampled->view() : source->view();
        auto hairT = rw ? resampled->hairT() : source->hairT();
        auto rootPrim = rw ? resampled->rootPrim() : source->rootPrim();
        auto rootUV = rw ? resampled->rootUV() : source->rootUV();
        SourceSnapshot s;
        s.curves = uint32_t(view.curveCount);
        s.pointCount = uint32_t(view.pointCount);
        std::vector<::float3> pts, rest;
        std::vector<float> widths;
        if (!DownloadView(view.points, &pts) ||
            !DownloadView(view.restPoints, &rest) ||
            !DownloadView(view.widths, &widths) ||
            !DownloadView(hairT, &s.hairT) ||
            !DownloadView(view.curveOffsets, &s.offsets) ||
            !DownloadView(view.stableIds, &s.ids) ||
            !DownloadView(rootPrim, &s.rootPrim))
            return false;
        if (pts.size() != view.pointCount ||
            rest.size() != view.pointCount ||
            widths.size() != view.pointCount)
            return false;
        for (auto const& p : pts)
            s.points.insert(s.points.end(), {p.x, p.y, p.z});
        for (auto const& p : rest)
            s.rest.insert(s.rest.end(), {p.x, p.y, p.z});
        s.widths = std::move(widths);
        std::vector<::float2> uvs;
        if (!DownloadView(rootUV, &uvs)) return false;
        for (auto const& uv : uvs)
            s.rootUV.insert(s.rootUV.end(), {uv.x, uv.y});
        if (!rebuilt.empty()) {
            for (auto const& plane : rebuilt) {
                if (!plane.bytes) return false;
                NamedSnapshot n;
                n.name = plane.metadata.name.GetString();
                n.type = plane.metadata.type;
                n.domain = plane.metadata.domain;
                n.elements = plane.metadata.elementCount;
                n.arity = plane.metadata.arity;
                n.bytes.assign(plane.bytes->size(), 0);
                if (!n.bytes.empty() &&
                    cudaMemcpy(n.bytes.data(), plane.bytes->data(),
                               n.bytes.size(),
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    return false;
                s.named[n.name] = std::move(n);
            }
        }
        // resampleTo == 0 leaves named empty here: uploaded canonical
        // planes are final and the caller reads them via UploadedPlanes().
        *out = std::move(s);
        return true;
    }

    // Canonical uploaded planes for the resampleTo == 0 path.
    bool UploadedPlanes(std::map<std::string, NamedSnapshot>* out,
                        UsdGenGraphDesc const& d) {
        if (!out) return false;
        if (planeBytes.size() != d.curveSets[0].authoredPlanes.size())
            return false;
        for (size_t i = 0; i != planeBytes.size(); ++i) {
            auto const& authored = d.curveSets[0].authoredPlanes[i];
            NamedSnapshot n;
            n.name = authored.name.GetString();
            n.type = authored.type == UsdGenAuthoredPlaneType::Float32
                ? UsdGenDeviceValueType::Float32
                : UsdGenDeviceValueType::Int32;
            n.domain = authored.domain == UsdGenAuthoredPlaneDomain::Point
                ? UsdGenDeviceDomain::Point
                : authored.domain == UsdGenAuthoredPlaneDomain::Primitive
                    ? UsdGenDeviceDomain::Primitive
                    : UsdGenDeviceDomain::Groom;
            n.elements = n.domain == UsdGenDeviceDomain::Point
                ? prepared.points.size()
                : n.domain == UsdGenDeviceDomain::Primitive
                    ? prepared.curveVertexCounts.size()
                    : 1;
            n.arity = authored.arity;
            n.bytes.assign(planeBytes[i]->size(), 0);
            if (!n.bytes.empty() &&
                cudaMemcpy(n.bytes.data(), planeBytes[i]->data(),
                           n.bytes.size(),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
            (*out)[n.name] = std::move(n);
        }
        return true;
    }
};

bool CudaAvailable() {
    int n = 0;
    cudaError_t const result = cudaGetDeviceCount(&n);
    bool const unavailable = result == cudaErrorNoDevice ||
        result == cudaErrorInsufficientDriver || (result == cudaSuccess && n == 0);
    if (unavailable) return false;
    CHECK(result == cudaSuccess);
    if (result != cudaSuccess) return false;
    bool const selected = cudaSetDevice(0) == cudaSuccess;
    CHECK(selected);
    return selected;
}

#endif // USDGEN_ENABLE_CUDA

// ------------------------------------------------------- compare

bool BitwiseFloats(std::vector<float> const& a, std::vector<float> const& b) {
    return a.size() == b.size() &&
        (a.empty() ||
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void CheckFloats(char const* label, std::vector<float> const& a,
                 std::vector<float> const& b, bool resampled) {
    bool sizeOk = false;
    float const worst = WorstAbs(a, b, &sizeOk);
    if (!sizeOk) {
        CHECK(false);
        std::fprintf(stderr, "csourceVk %s: size %zu vs %zu\n", label,
                     a.size(), b.size());
        return;
    }
    if (resampled) {
        std::printf("csourceVk %s: worst %.3g (tol %.3g)\n", label,
                    double(worst), double(kResampleTolerance));
        CHECK(worst < kResampleTolerance);
    } else {
        CHECK(BitwiseFloats(a, b));
    }
}

void CompareSnapshots(char const* fixture, SourceSnapshot const& a,
                       SourceSnapshot const& b, bool resampled,
                       bool compareFrames = true) {
    std::string const tag = std::string(fixture);
    CHECK(a.curves == b.curves && a.pointCount == b.pointCount);
    CHECK(a.offsets == b.offsets);
    CHECK(a.ids == b.ids);
    CHECK(a.rootPrim == b.rootPrim);
    CheckFloats((tag + " points").c_str(), a.points, b.points, resampled);
    CheckFloats((tag + " rest").c_str(), a.rest, b.rest, resampled);
    CheckFloats((tag + " widths").c_str(), a.widths, b.widths, resampled);
    CheckFloats((tag + " hairT").c_str(), a.hairT, b.hairT, resampled);
    // rootUV is per-curve passthrough on both backends: always bitwise.
    CHECK(BitwiseFloats(a.rootUV, b.rootUV));
    // Frames: bitwise against each other (same shared capture underneath).
    // Skipped for CUDA-involving pairings: CudaSourceLeg has no device frame
    // twin by design (empty vectors), and CUDA frame equivalence stays
    // covered by the prepared-rootFrames-vs-capture check plus FrameOracle.
    if (compareFrames) {
        CHECK(BitwiseFloats(a.frameT, b.frameT));
        CHECK(BitwiseFloats(a.frameB, b.frameB));
        CHECK(BitwiseFloats(a.frameN, b.frameN));
    }
    CHECK(a.named.size() == b.named.size());
    for (auto const& [name, na] : a.named) {
        auto const it = b.named.find(name);
        CHECK(it != b.named.end());
        if (it == b.named.end()) continue;
        auto const& nb = it->second;
        CHECK(na.type == nb.type && na.domain == nb.domain &&
              na.elements == nb.elements && na.arity == nb.arity);
        if (na.type == UsdGenDeviceValueType::Float32 && resampled) {
            std::vector<float> fa(na.bytes.size() / 4), fb(nb.bytes.size() / 4);
            if (!na.bytes.empty())
                std::memcpy(fa.data(), na.bytes.data(), na.bytes.size());
            if (!nb.bytes.empty())
                std::memcpy(fb.data(), nb.bytes.data(), nb.bytes.size());
            CheckFloats((tag + " named:" + name).c_str(), fa, fb, true);
        } else {
            CHECK(na.bytes == nb.bytes);
        }
    }
}

bool SourceParity(char const* fixture, UsdGenGraphDesc const& d,
                  bool resampled, VulkanDevice* vkDev, bool haveCuda) {
    UsdGenCurveBuffer loaded;
    if (!LoadDesc(d, &loaded)) {
        CHECK(false);
        return false;
    }
    SourceSnapshot host;
    if (!HostSnapshot(loaded, &host)) {
        CHECK(false);
        return false;
    }
    bool vkOk = false, cuOk = false;
    SourceSnapshot vk, cu;
    if (vkDev) {
        UsdGenCurveBuffer discard;
        vkOk = VulkanSourceSnapshot(*vkDev, d, &vk, &discard);
        CHECK(vkOk);
        if (vkOk) {
            g_vulkanDeviceProven = true;
            // Upload fidelity: device planes reproduce the loader packet.
            CompareSnapshots((std::string(fixture) + " vk==host").c_str(), vk,
                             host, false);
        }
    }
    (void)resampled;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda) {
        CudaSourceLeg leg;
        cuOk = leg.Run(d);
        CHECK(cuOk);
        if (cuOk) {
            if (leg.resampleTo == 0 && !leg.planeBytes.empty()) {
                // No transform runs for resampleTo == 0; uploaded
                // canonical planes are final on the CUDA path too.
                SourceSnapshot tmp;
                cuOk = leg.Snapshot(&tmp);
                CHECK(cuOk);
                cu = std::move(tmp);
                cuOk = leg.UploadedPlanes(&cu.named, d);
                CHECK(cuOk);
            } else {
                cuOk = leg.Snapshot(&cu);
                CHECK(cuOk);
            }
        }
        if (cuOk) {
            g_cudaDeviceProven = true;
            bool const rw = leg.resampleTo != 0;
            CompareSnapshots((std::string(fixture) + " cu==host").c_str(), cu,
                             host, rw, false);
            // CUDA-leg plumbing used the same shared capture the oracle
            // derives from: prepared root frames match capture frames in
            // canonical order.
            {
                auto const& curves = d.curveSets[0];
                size_t const count = curves.curveVertexCounts.size();
                std::vector<size_t> order;
                for (size_t c = 0; c != count; ++c)
                    if (leg.captured.valid[c]) order.push_back(c);
                auto idOf = [&](size_t c) {
                    return !curves.curveId.empty() &&
                            curves.curveId.size() == count
                        ? curves.curveId[c]
                        : uint64_t(c);
                };
                std::stable_sort(order.begin(), order.end(), [&](size_t a,
                                                                 size_t c) {
                    return idOf(a) < idOf(c);
                });
                CHECK(leg.prepared.rootFrames.size() == order.size());
                for (size_t i = 0;
                     i < order.size() &&
                     i < leg.prepared.rootFrames.size();
                     ++i) {
                    auto const& got = leg.prepared.rootFrames[i];
                    auto const& want = leg.captured.frames[order[i]];
                    bool same = true;
                    for (int r = 0; r != 4 && same; ++r)
                        for (int c = 0; c != 4 && same; ++c)
                            same = got[size_t(r) * 4 + size_t(c)] ==
                                want[r][c];
                    CHECK(same);
                }
            }
        }
    }
#else
    (void)haveCuda;
    (void)cu;
#endif
    // Frame oracle: device frames reproduce the shared capture bitwise.
    {
        auto const& node = d.nodes[0];
        auto const& curves = d.curveSets[0];
        UsdGenSurfaceDesc const* surface = nullptr;
        if (!node.surfaces.empty()) {
            for (auto const& s : d.surfaces)
                if (s.path == node.surfaces.front()) surface = &s;
        }
        TfToken rebind = TfToken("never");
        for (auto const& p : node.params)
            if (p.name == TfToken("rebind") && p.value.IsHolding<usdGen::UsdGenToken>())
                rebind = p.value.UncheckedGet<usdGen::UsdGenToken>();
        UsdGenCurveRootCaptureResult captured;
        std::string error;
        CHECK(UsdGenCaptureCurveRoots(usdGenUsd::Lower(curves), (surface ? usdGenUsd::Addr(usdGenUsd::View(*surface)) : nullptr), rebind, &captured,
                                       &error));
        std::vector<float> ot, ob, on;
        CHECK(FrameOracle(curves, captured, &ot, &ob, &on));
        if (vkOk) {
            CHECK(BitwiseFloats(vk.frameT, ot));
            CHECK(BitwiseFloats(vk.frameB, ob));
            CHECK(BitwiseFloats(vk.frameN, on));
        }
        CHECK(BitwiseFloats(host.frameT, ot));
        CHECK(BitwiseFloats(host.frameB, ob));
        CHECK(BitwiseFloats(host.frameN, on));
    }
    if (vkOk && cuOk)
        CompareSnapshots((std::string(fixture) + " vk==cu").c_str(), vk, cu,
                         resampled, false);
    std::printf("csourceVk: fixture %s source parity "
                "(vk=%d cu=%d resampled=%d) done\n",
                fixture, int(vkOk), int(cuOk), int(resampled));
    return true;
}

// ------------------------------------------------------- deform leg

std::vector<uint32_t> ReadSpirv(char const* path) {
    std::ifstream shader(path, std::ios::binary);
    if (!shader) return {};
    std::vector<char> raw((std::istreambuf_iterator<char>(shader)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> code(raw.size() / sizeof(uint32_t));
    std::memcpy(code.data(), raw.data(), raw.size());
    return code;
}

std::shared_ptr<const ChargedBuffer> UploadScratch(
    std::shared_ptr<DeviceContext> const& context, void const* data,
    VkDeviceSize bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = ChargedBuffer::Create(
        context, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!buffer) return {};
    if (bytes) {
        void* mapped = nullptr;
        if (vkMapMemory(context->device(), buffer->memory(), 0, bytes, 0,
                        &mapped) != VK_SUCCESS)
            return {};
        std::memcpy(mapped, data, bytes);
        vkUnmapMemory(context->device(), buffer->memory());
    }
    return buffer;
}

bool VulkanDeformLeg(VulkanDevice const& dev, DeformFixture const& fx,
                     std::vector<uint32_t> const& evalSpv,
                     std::vector<uint32_t> const& applySpv,
                     std::vector<float>* out) {
    if (!out) return false;
    VkResult status = VK_ERROR_UNKNOWN;
    auto pipe =
        DeformPipeline::Create(dev.context, evalSpv, applySpv, &status);
    if (!pipe || status != VK_SUCCESS) {
        std::fprintf(stderr, "csourceVk deform Create failed: status=%d\n",
                     int(status));
        return false;
    }
    DeformPipeline::BeginInfo info;
    info.points = UploadScratch(dev.context, fx.points.data(),
                                fx.points.size() * 4);
    info.curveOffsets = UploadScratch(dev.context, fx.offsets.data(),
                                      fx.offsets.size() * 4);
    info.rootTargets = UploadScratch(dev.context, fx.targets.data(),
                                     fx.targets.size() * 4);
    if (!info.points || !info.curveOffsets || !info.rootTargets)
        return false;
    info.curveCount = 2;
    info.pointCount = 6;
    info.restSamples = fx.restSamples;
    info.posedSamples = fx.posedSamples;
    info.sampleCount = int(fx.restSamples.size() / 3);
    info.smoothing = 0.0;
    info.mask = {1.0f, 1, nullptr, 0};
    info.enabled = {1, 1, nullptr, 0};
    info.lockRoots = {1, 1, nullptr, 0};
    info.groomEnvelope = 1.0f;
    DeformSemantic sem = DeformSemantic::Ok;
    auto candidate = pipe->Begin(std::move(info), &status, &sem);
    if (!candidate || status != VK_SUCCESS) {
        std::fprintf(stderr, "csourceVk deform Begin failed: status=%d sem=%u\n",
                     int(status), unsigned(sem));
        return false;
    }
    for (int i = 0; i != 100000; ++i) {
        VkResult r = candidate->Poll(&sem);
        if (r == VK_SUCCESS) break;
        if (r != VK_NOT_READY) {
            std::fprintf(stderr, "csourceVk deform Poll failed: r=%d\n",
                         int(r));
            return false;
        }
        if (i % 1000 == 0 && !ProveNative(dev.native)) return false;
        if (i == 99999) {
            std::fprintf(stderr, "csourceVk deform Poll timed out\n");
            return false;
        }
    }
    if (sem != DeformSemantic::Ok || !candidate->succeeded()) {
        std::fprintf(stderr, "csourceVk deform semantic=%u succeeded=%d\n",
                     unsigned(sem), int(candidate->succeeded()));
        return false;
    }
    auto output = candidate->output();
    std::vector<uint8_t> bytes;
    if (!ReadVulkanBytes(dev.native, dev.context, output.points->buffer(),
                         output.points->sizeBytes(), output.points, &bytes))
        return false;
    *out = AsFloats(bytes);
    return out->size() == fx.oracle.size();
}

#ifdef USDGEN_ENABLE_CUDA
bool CudaDeformLeg(DeformFixture const& fx, std::vector<float>* out) {
    if (!out) return false;
    auto f3 = [](std::vector<float> const& v) {
        std::vector<::float3> r(v.size() / 3);
        for (size_t i = 0; i != r.size(); ++i)
            r[i] = ::make_float3(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
        return r;
    };
    std::vector<::float3> restS = f3(fx.restSamples);
    std::vector<::float3> posedS = f3(fx.posedSamples);
    std::vector<::float3> targets = f3(fx.targets);
    std::vector<::float3> points = f3(fx.points);
    gpu::DeviceBuffer<::float3> restB, posedB, targetsB, pointsB, outputB;
    gpu::DeviceBuffer<uint32_t> offsetsB;
    auto upload = [](auto& buffer, auto const& values) {
        return buffer.reset(values.size()) == cudaSuccess &&
            (values.empty() ||
             cudaMemcpy(buffer.data(), values.data(),
                        values.size() * sizeof(values[0]),
                        cudaMemcpyHostToDevice) == cudaSuccess);
    };
    if (!upload(restB, restS) || !upload(posedB, posedS) ||
        !upload(targetsB, targets) || !upload(pointsB, points) ||
        !upload(offsetsB, fx.offsets) ||
        outputB.reset(points.size()) != cudaSuccess)
        return false;
    gpu::DeviceCurveGeometryView geometry{
        {pointsB.data(), pointsB.size()},
        {pointsB.data(), pointsB.size()},
        {},
        {offsetsB.data(), offsetsB.size()},
        {},
        2,
        pointsB.size()};
    gpu::CudaRbfBinding binding;
    if (binding.Bind({restB.data(), restB.size()}, 0.0, nullptr) !=
        gpu::RbfStatus::Ok)
        return false;
    if (binding.Solve({posedB.data(), posedB.size()}, nullptr) !=
        gpu::RbfStatus::Ok)
        return false;
    gpu::CudaRbfCurveDeformer deform;
    if (deform.Deform(binding, geometry, {targetsB.data(), targetsB.size()},
                      1.0f, {}, {}, outputB.view(), nullptr) !=
            gpu::RbfStatus::Ok ||
        deform.Finish(binding, nullptr) != gpu::RbfStatus::Ok)
        return false;
    std::vector<::float3> got;
    if (!DownloadView(gpu::DeviceView<const ::float3>{outputB.data(),
                                                      outputB.size()},
                      &got))
        return false;
    out->clear();
    for (auto const& p : got) out->insert(out->end(), {p.x, p.y, p.z});
    return out->size() == fx.oracle.size();
}
#endif

void DeformParity(VulkanDevice* vkDev, bool haveCuda,
                  std::vector<uint32_t> const& evalSpv,
                  std::vector<uint32_t> const& applySpv) {
    DeformFixture fx = FixtureDeformMovedSurface();
    size_t const samples = fx.restSamples.size() / 3;
    CHECK(samples >= 4 && samples <= 100);
    CHECK(fx.posedSamples.size() == fx.restSamples.size());
    CHECK(fx.targets.size() == 6 && fx.oracle.size() == 18);
    bool vkOk = false, cuOk = false;
    std::vector<float> vk, cu;
    if (vkDev) {
        vkOk = VulkanDeformLeg(*vkDev, fx, evalSpv, applySpv, &vk);
        CHECK(vkOk);
        if (vkOk) {
            bool sizeOk = false;
            float const worst = WorstAbs(vk, fx.oracle, &sizeOk);
            std::printf("csourceVk deform vk vs oracle: worst %.3g "
                        "(tol %.3g)\n",
                        double(worst), double(kDeformOracleTolerance));
            CHECK(sizeOk && worst <= kDeformOracleTolerance);
        }
    }
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda) {
        cuOk = CudaDeformLeg(fx, &cu);
        CHECK(cuOk);
        if (cuOk) {
            bool sizeOk = false;
            float const worst = WorstAbs(cu, fx.oracle, &sizeOk);
            std::printf("csourceVk deform cu vs oracle: worst %.3g "
                        "(tol %.3g)\n",
                        double(worst), double(kDeformCrossTolerance));
            CHECK(sizeOk && worst < kDeformCrossTolerance);
        }
    }
#else
    (void)haveCuda;
#endif
    if (vkOk && cuOk) {
        bool sizeOk = false;
        float const worst = WorstAbs(vk, cu, &sizeOk);
        std::printf("csourceVk deform vk vs cu: worst %.3g (tol %.3g)\n",
                    double(worst), double(kDeformCrossTolerance));
        CHECK(sizeOk && worst < kDeformCrossTolerance);
        g_deformProven = true;
    }
    std::puts("csourceVk: deform parity done");
}

} // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> evalSpv, applySpv;
    if (argc == 3) {
        evalSpv = ReadSpirv(argv[1]);
        applySpv = ReadSpirv(argv[2]);
        if (evalSpv.empty() || applySpv.empty()) {
            std::fprintf(stderr,
                         "csourceVk: cannot load deform SPIR-V modules\n");
            return 1;
        }
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [deformEvaluate.spv deformApply.spv]\n",
                     argv[0]);
        return 1;
    }

    HostAdmissionTables();
    LoaderChecks();

    VulkanDevice vkDev;
    bool vkUnavailable = false;
    bool const haveVulkan = InitVulkan(&vkDev, &vkUnavailable);
    if (!haveVulkan)
        std::printf("csourceVk: vulkan leg %s\n",
                    vkUnavailable ? "unavailable (no supported compute device)"
                                  : "failed to initialize");
    CHECK(haveVulkan || vkUnavailable);
    bool haveCuda = false;
#ifdef USDGEN_ENABLE_CUDA
    haveCuda = CudaAvailable();
    if (!haveCuda)
        std::puts("csourceVk: cuda leg unavailable (no CUDA device)");
#else
    std::puts("csourceVk: cuda leg not compiled (USDGEN_ENABLE_CUDA off)");
#endif

    SourceParity("A/resample8", FixtureResample(8), true,
                 haveVulkan ? &vkDev : nullptr, haveCuda);
    SourceParity("B/resample0", FixtureResample(0), false,
                 haveVulkan ? &vkDev : nullptr, haveCuda);
    SourceParity("C/surffree-onError", FixtureSurfaceFree(), false,
                 haveVulkan ? &vkDev : nullptr, haveCuda);
    SourceParity("D/moved+rest", FixtureMoved(), false,
                 haveVulkan ? &vkDev : nullptr, haveCuda);
    if (!evalSpv.empty() && (!haveVulkan || vkDev.shaderFloat64))
        DeformParity(haveVulkan ? &vkDev : nullptr, haveCuda, evalSpv,
                     applySpv);
    else if (evalSpv.empty())
        std::puts("csourceVk: deform leg skipped (no SPIR-V argv)");
    else
        std::puts("csourceVk: vulkan deform skipped (no shaderFloat64)");

    std::printf("csourceVk: device parity legs proven: vulkan-source=%d "
                "cuda-source=%d deform-cross=%d\n",
                int(g_vulkanDeviceProven), int(g_cudaDeviceProven),
                int(g_deformProven));
    std::printf("csourceVk: %d checks, %d failures\n", g_checks, g_fail);
    bool const complete = g_vulkanDeviceProven && g_cudaDeviceProven &&
        (evalSpv.empty() || g_deformProven);
    return g_fail ? 1 : (complete ? 0 : 77);
}
