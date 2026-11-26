// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanSurfaceVk — Phase 2 (p2-surface): non-identity surface
// binding for Vulkan Deform.
//
// The Vulkan plan compiler admits only identity source/surface transforms
// for Deform; moved transforms reject with "Deform requires identity source
// and surface transforms" (see testUsdGenVulkanGapRejections). The surfaceVk
// module closes that gap by compiling the frozen RBF fit (rest/posed
// samples plus per-curve root targets) in source-local space, mapped from
// the pose surface by surface.worldMatrix * source.worldMatrix^-1.
//
// CUDA itself rejects the same moved-transform graphs (PrepareCudaSurface
// rejects non-identity surface transforms; the RBF lane requires identity
// Description/CurveSource transforms), so the parity twin is a baked graph:
// the same motion with the transforms folded into the points and identity
// matrices, which CUDA accepts. Rigid-equivariance of the cubic RBF fit
// makes the twin's world-space output agree with the Vulkan source-local
// output (mapped by the source matrix) up to float rounding.
//
// Parts:
//   A (host, always): the helper accepts moved-source, moved-surface and
//     doubly-moved posed fixtures with the exact mapped samples/targets,
//     and still rejects malformed budgets, matrices, bindings and faces.
//   B (device): Vulkan DeformPipeline driven by the helper vs the CUDA
//     twin (CudaSurfaceBinding + CudaRbfBinding + legacy CudaRbfCurveDeformer)
//     within 4e-4/component, the Near() contract of testUsdGenCudaDeform.cu.
//     A leg that cannot initialize (no device, tenant OOM) is skipped with
//     a SKIP line; the Vulkan leg is additionally checked against the CPU
//     RBF oracle within 2e-3 like testUsdGenVulkanDeform cases 4/5.
//   C (plan): CompileVulkanSourceWidthPlan accepts the moved graphs and
//     freezes exactly the helper's controls; the identity graph still
//     accepts through the untouched legacy path. Requires the phase2
//     surface plan hook (applied by integration).
//
// Usage: testUsdGenVulkanSurfaceVk deformEvaluate.spv deformApply.spv

#include "usdGenVulkan/vulkan/surfaceVk.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGen/op.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/deformPipeline.h"
#include "usdGenCurves/vulkan/deformRbfHost.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "gpu/deformCurves.h"
#include "gpu/rbf.h"
#include "gpu/surfaceBinding.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

int g_failures = 0;
bool g_crossBackendCompared = false;

void Check(bool ok, char const* what) {
    if (ok) {
        std::printf("PASS: %s\n", what);
    } else {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

// Parity bound vs the CUDA twin: the per-component Near() contract of
// tests/testUsdGenCudaDeform.cu (4.e-4f absolute).
constexpr float kCudaParityTolerance = 4.0e-4f;
// Sample/root-target agreement between the helper and the CUDA binding
// (same fit inputs, float rounding only).
constexpr float kFitInputTolerance = 1.0e-5f;
// Host helper checks against the in-test matrix oracle.
constexpr float kHostTolerance = 1.0e-6f;
// Vulkan leg vs the CPU RBF oracle (testUsdGenVulkanDeform cases 4/5).
constexpr float kOracleTolerance = 2.0e-3f;

bool Near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

// Surface rest pose: two triangles plus one quad (10 verts).
std::vector<GfVec3f> RestVerts() {
    return {GfVec3f(0, 0, 0), GfVec3f(1, 0, 0), GfVec3f(0, 1, 0),
            GfVec3f(0, 0, 1), GfVec3f(1, 1, 0), GfVec3f(1, 0, 1),
            GfVec3f(2, 0, 0), GfVec3f(3, 0, 0), GfVec3f(3, 1, 0),
            GfVec3f(2, 1, 0)};
}

// Non-rigid pose deltas, one per rest vert.
std::vector<GfVec3f> PoseDeltas() {
    return {GfVec3f(0, 0, 0), GfVec3f(.1f, 0, 0), GfVec3f(0, .15f, .05f),
            GfVec3f(0, 0, .2f), GfVec3f(-.1f, .1f, 0), GfVec3f(.05f, -.05f, .1f),
            GfVec3f(.2f, 0, 0), GfVec3f(0, .2f, 0), GfVec3f(0, 0, .25f),
            GfVec3f(-.15f, 0, .05f)};
}

std::vector<GfVec3f> PosedVerts() {
    auto rest = RestVerts();
    auto deltas = PoseDeltas();
    std::vector<GfVec3f> posed(rest.size());
    for (size_t i = 0; i < rest.size(); ++i) posed[i] = rest[i] + deltas[i];
    return posed;
}

// Three curves x three CVs near faces 0, 1 and 2.
std::vector<GfVec3f> CurvePoints() {
    return {GfVec3f(.2f, .2f, 0), GfVec3f(.25f, .25f, .3f),
            GfVec3f(.3f, .3f, .6f), GfVec3f(.1f, .1f, .9f),
            GfVec3f(.15f, .15f, 1.2f), GfVec3f(.2f, .2f, 1.5f),
            GfVec3f(2.5f, .5f, 0), GfVec3f(2.55f, .55f, .3f),
            GfVec3f(2.6f, .6f, .6f)};
}

UsdGenSurfaceDesc MakeSurface() {
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    auto rest = RestVerts();
    auto posed = PosedVerts();
    surface.restPoints = VtVec3fArray(rest.data(), rest.data() + rest.size());
    surface.points = VtVec3fArray(posed.data(), posed.data() + posed.size());
    surface.faceVertexCounts = {3, 3, 4};
    surface.faceVertexIndices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    surface.uv = {{0, 0}, {1, 0}, {0, 1}, {0, 0}, {1, 0}, {0, 1},
                  {0, 0}, {1, 0}, {1, 1}, {0, 1}};
    return surface;
}

UsdGenCurveSetDesc MakeSource() {
    UsdGenCurveSetDesc source;
    source.path = SdfPath("/Groom/C3");
    source.role = UsdGenLane::Chunked;
    source.curveRole = TfToken("hair");
    source.curveVertexCounts = {3, 3, 3};
    auto points = CurvePoints();
    source.points = VtVec3fArray(points.data(), points.data() + points.size());
    source.rest = source.points;
    source.widths = {.02f, .03f, .04f, .02f, .03f, .04f, .02f, .03f, .04f};
    source.curveId = {41, 42, 43};
    source.curveGeneration = 19;
    source.frozenEpoch = "epoch-19";
    source.skinPrim = {0, 1, 2};
    source.skinPrimUv = {{.2f, .2f}, {.3f, .1f}, {.5f, .5f}};
    return source;
}

// Independent oracle: CUDA-formula gather (surfaceBinding.cu RootPosition)
// of a face+UV from float points, then an affine map in double.
GfVec3d OracleRootTarget(std::vector<GfVec3f> const& points,
                         std::vector<int> const& faceStart, int face, int n,
                         std::vector<int> const& indices, GfVec2f const& uv,
                         GfMatrix4d const& map) {
    float const u = uv[0], v = uv[1];
    auto P = [&](int j) { return points[size_t(indices[size_t(faceStart[size_t(face)]) + size_t(j)])]; };
    GfVec3f local(0);
    if (n == 3) {
        float const w = 1.0f - u - v;
        GfVec3f const a = P(0), b = P(1), c = P(2);
        local = GfVec3f(a[0] * w + b[0] * u + c[0] * v,
                        a[1] * w + b[1] * u + c[1] * v,
                        a[2] * w + b[2] * u + c[2] * v);
    } else {
        float const u0 = 1.0f - u, v0 = 1.0f - v;
        GfVec3f const a = P(0), b = P(1), c = P(2), d = P(3);
        local = GfVec3f(a[0] * u0 * v0 + b[0] * u * v0 + c[0] * u * v + d[0] * u0 * v,
                        a[1] * u0 * v0 + b[1] * u * v0 + c[1] * u * v + d[1] * u0 * v,
                        a[2] * u0 * v0 + b[2] * u * v0 + c[2] * u * v + d[2] * u0 * v);
    }
    return map.TransformAffine(GfVec3d(local));
}

bool SameSet(std::vector<float> const& a, std::vector<float> const& b, float tol) {
    if (a.size() != b.size() || a.size() % 3 != 0) return false;
    // Lexicographic sort on triples.
    auto order = [](std::vector<float> v) {
        size_t const n = v.size() / 3;
        std::vector<size_t> idx(n);
        for (size_t i = 0; i < n; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](size_t x, size_t y) {
            for (int k = 0; k < 3; ++k) {
                if (v[3 * x + size_t(k)] != v[3 * y + size_t(k)])
                    return v[3 * x + size_t(k)] < v[3 * y + size_t(k)];
            }
            return false;
        });
        std::vector<float> out(v.size());
        for (size_t i = 0; i < n; ++i)
            for (int k = 0; k < 3; ++k) out[3 * i + size_t(k)] = v[3 * idx[i] + size_t(k)];
        return out;
    };
    auto x = order(a), y = order(b);
    for (size_t i = 0; i < x.size(); ++i)
        if (!Near(x[i], y[i], tol)) return false;
    return true;
}

// Part A: host-side helper checks. Returns the compiled controls for the
// device legs to consume (empty on failure).
bool CheckHelperCase(char const* label, GfMatrix4d const& sourceMatrix,
                     GfMatrix4d const& surfaceMatrix, VulkanDeformControls* controls) {
    auto source = MakeSource();
    auto surface = MakeSurface();
    source.worldMatrix = sourceMatrix;
    surface.worldMatrix = surfaceMatrix;
    UsdGenDiagnostics diagnostics;
    VulkanDeformControls out;
    out.lockRoots = true;
    out.mask = 1.0f;
    out.groomEnvelope = 1.0f;
    bool const ok = CompileSurfaceVkDeform(usdGenUsd::Lower(source), usdGenUsd::Lower(surface), 100, &out, &diagnostics);
    if (!ok || diagnostics.HasErrors()) {
        std::printf("      diagnostic: %s\n",
                    diagnostics.errors.empty() ? "<none>" : diagnostics.errors.front().c_str());
        Check(false, label);
        return false;
    }
    // Ten verts under budget 100: every distinct position is selected.
    bool sane = out.sampleCount == 10 && out.restSamples.size() == 30 &&
        out.posedSamples.size() == 30 && out.rootTargets.size() == 9 &&
        out.lockRoots && out.mask == 1.0f && out.groomEnvelope == 1.0f;
    // Expected samples: the mapped rest/posed verts as sets.
    double det = 0.0;
    GfMatrix4d const relative = surfaceMatrix * sourceMatrix.GetInverse(&det);
    auto rest = RestVerts();
    auto posed = PosedVerts();
    std::vector<float> wantRest, wantPosed;
    for (size_t i = 0; i < rest.size(); ++i) {
        GfVec3d r = relative.TransformAffine(GfVec3d(rest[i]));
        GfVec3d p = relative.TransformAffine(GfVec3d(posed[i]));
        wantRest.insert(wantRest.end(), {float(r[0]), float(r[1]), float(r[2])});
        wantPosed.insert(wantPosed.end(), {float(p[0]), float(p[1]), float(p[2])});
    }
    sane = sane && SameSet(out.restSamples, wantRest, kHostTolerance) &&
        SameSet(out.posedSamples, wantPosed, kHostTolerance);
    // Expected root targets: CUDA-formula gather mapped to source-local.
    std::vector<int> faceStart{0, 3, 6, 10}, counts{3, 3, 4};
    std::vector<int> indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<GfVec2f> uvs{{.2f, .2f}, {.3f, .1f}, {.5f, .5f}};
    for (int c = 0; sane && c < 3; ++c) {
        GfVec3d want = OracleRootTarget(posed, faceStart, c, counts[size_t(c)],
                                       indices, uvs[size_t(c)], relative);
        for (int k = 0; k < 3; ++k)
            sane = sane && Near(out.rootTargets[size_t(3 * c + k)], float(want[k]), kHostTolerance);
    }
    Check(sane, label);
    if (sane && controls) *controls = out;
    return sane;
}

void PartA() {
    GfMatrix4d movedSource(1.0), movedSurface(1.0);
    movedSource[3][0] = 5.0;   // GapRejections moved-source transform.
    movedSurface[3][1] = -2.0; // GapRejections moved-surface transform.
    CheckHelperCase("A: helper accepts moved source", movedSource, GfMatrix4d(1.0), nullptr);
    CheckHelperCase("A: helper accepts moved surface", GfMatrix4d(1.0), movedSurface, nullptr);
    CheckHelperCase("A: helper accepts moved source+surface", movedSource, movedSurface, nullptr);
    CheckHelperCase("A: helper accepts identity", GfMatrix4d(1.0), GfMatrix4d(1.0), nullptr);
    {
        // Shared ancestor motion cancels: a coherent rigid move of source
        // and surface together freezes bitwise-identical controls, so only
        // the imaging xform (transport) changes, never the fit.
        auto source = MakeSource();
        auto surface = MakeSurface();
        source.worldMatrix = movedSource;
        surface.worldMatrix = movedSource;
        UsdGenDiagnostics d1, d2;
        VulkanDeformControls coherent, identity;
        coherent.lockRoots = identity.lockRoots = true;
        coherent.mask = identity.mask = 1.0f;
        coherent.groomEnvelope = identity.groomEnvelope = 1.0f;
        auto plain = MakeSource();
        auto surf = MakeSurface();
        bool const ok = CompileSurfaceVkDeform(usdGenUsd::Lower(source), usdGenUsd::Lower(surface), 100, &coherent, &d1) &&
            CompileSurfaceVkDeform(usdGenUsd::Lower(plain), usdGenUsd::Lower(surf), 100, &identity, &d2) &&
            coherent.sampleCount == identity.sampleCount &&
            coherent.restSamples == identity.restSamples &&
            coherent.posedSamples == identity.posedSamples &&
            coherent.rootTargets == identity.rootTargets;
        Check(ok, "A: coherent shared motion cancels bitwise");
    }

    // Rejections mirror the ValidateDeform messages where the check coincides.
    auto expectReject = [](char const* label, UsdGenCurveSetDesc source,
                           UsdGenSurfaceDesc surface, int budget, char const* needle) {
        UsdGenDiagnostics diagnostics;
        VulkanDeformControls out;
        bool const ok = CompileSurfaceVkDeform(usdGenUsd::Lower(source), usdGenUsd::Lower(surface), budget, &out, &diagnostics);
        bool found = false;
        for (auto const& error : diagnostics.errors)
            if (error.find(needle) != std::string::npos) found = true;
        if (!ok && !found)
            std::printf("      diagnostic: %s\n",
                        diagnostics.errors.empty() ? "<none>" : diagnostics.errors.front().c_str());
        Check(!ok && found, label);
    };
    expectReject("A: budget 3 rejects", MakeSource(), MakeSurface(), 3,
                 "Deform rbfSamples must be in [4,100]");
    expectReject("A: budget 101 rejects", MakeSource(), MakeSurface(), 101,
                 "Deform rbfSamples must be in [4,100]");
    {
        auto surface = MakeSurface();
        surface.points.resize(4);
        expectReject("A: mismatched pose count rejects", MakeSource(), surface, 100,
                     "Deform pose surface is invalid");
    }
    {
        auto source = MakeSource();
        source.worldMatrix[0][0] = 0.0; // Singular.
        expectReject("A: singular source matrix rejects", source, MakeSurface(), 100,
                     "Deform requires finite, affine and invertible source and surface transforms");
    }
    {
        auto surface = MakeSurface();
        surface.worldMatrix[3][1] = std::numeric_limits<double>::quiet_NaN();
        expectReject("A: non-finite surface matrix rejects", MakeSource(), surface, 100,
                     "Deform requires finite, affine and invertible source and surface transforms");
    }
    {
        auto source = MakeSource();
        source.skinPrimUv.resize(2);
        expectReject("A: root binding cardinality rejects", source, MakeSurface(), 100,
                     "Deform requires root bindings for every curve");
    }
    {
        auto source = MakeSource();
        source.skinPrim = {0, 1, 7};
        expectReject("A: invalid root face rejects", source, MakeSurface(), 100,
                     "Deform root binding references an invalid face");
    }
    {
        // Fewer than four distinct samples: the RBF fit cannot bind.
        auto surface = MakeSurface();
        for (size_t i = 1; i < surface.restPoints.size(); ++i)
            surface.restPoints[i] = surface.restPoints[0];
        expectReject("A: degenerate rest rejects", MakeSource(), surface, 100,
                     "Deform pose surface has fewer than four spanning-3D samples");
    }
}

// ---- Part B: device parity ----

inline std::vector<uint32_t> Code(char const* p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}

inline bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

inline std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const& c, void const* data, VkDeviceSize bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = ChargedBuffer::Create(c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b) return {};
    if (bytes) {
        void* p = nullptr;
        if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS) return {};
        std::memcpy(p, data, bytes);
        vkUnmapMemory(c->device(), b->memory());
    }
    return b;
}

inline bool ReadOutput(std::shared_ptr<NativeOwner> const& native,
                       std::shared_ptr<DeviceContext> const& context,
                       std::unique_ptr<DeformPipeline::Candidate> const& c,
                       std::vector<float>* out) {
    auto o = c->output();
    std::vector<uint8_t> bytes;
    if (!ReadVulkanBytes(native, context, o.points->buffer(),
                         o.points->sizeBytes(), o.points, &bytes)) return false;
    out->resize(bytes.size() / sizeof(float));
    std::memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

bool CompileForDevice(char const* label, GfMatrix4d const& sourceMatrix,
                      GfMatrix4d const& surfaceMatrix, VulkanDeformControls* controls) {
    auto source = MakeSource();
    auto surface = MakeSurface();
    source.worldMatrix = sourceMatrix;
    surface.worldMatrix = surfaceMatrix;
    UsdGenDiagnostics diagnostics;
    VulkanDeformControls out;
    out.lockRoots = true;
    out.mask = 1.0f;
    out.groomEnvelope = 1.0f;
    if (!CompileSurfaceVkDeform(usdGenUsd::Lower(source), usdGenUsd::Lower(surface), 100, &out, &diagnostics) ||
        diagnostics.HasErrors()) {
        std::printf("      diagnostic: %s\n",
                    diagnostics.errors.empty() ? "<none>" : diagnostics.errors.front().c_str());
        Check(false, label);
        return false;
    }
    *controls = out;
    return true;
}

struct VulkanLeg {
    std::shared_ptr<NativeOwner> native;
    std::shared_ptr<DeviceContext> context;
    std::shared_ptr<DeformPipeline> pipe;
    std::shared_ptr<const ChargedBuffer> pointsBuf, offsetsBuf;
    bool ready = false;
};

bool InitVulkanLeg(char const* evalPath, char const* applyPath, VulkanLeg* leg) {
    auto evalSpv = Code(evalPath);
    auto applySpv = Code(applyPath);
    if (evalSpv.empty() || applySpv.empty()) {
        Check(false, "B: deform SPIR-V loads");
        return false;
    }
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64{};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateTimelineNative(&unavailable, &fp64);
    if (unavailable) {
        std::printf("SKIP: no Vulkan compute device with shaderFloat64\n");
        return false;
    }
    if (!native) {
        Check(false, "B: Vulkan device creation succeeds");
        return false;
    }
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(native->physical, &feats);
    if (!feats.shaderFloat64) {
        std::printf("SKIP: Vulkan device lacks shaderFloat64\n");
        return false;
    }
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8021;
    ci.nativeLifetime = native;
    ci.resources = {size_t{32} << 20, 0};
    ci.shaderFloat64Enabled = true;
    ci.timelineSemaphoreEnabled = true;
    auto context = DeviceContext::Create(ci);
    if (!context) {
        Check(false, "B: Vulkan context creation succeeds");
        return false;
    }
    VkResult status = VK_SUCCESS;
    auto pipe = DeformPipeline::Create(context, evalSpv, applySpv, &status);
    if (!pipe || status != VK_SUCCESS) {
        Check(false, "B: Vulkan deform pipeline creation succeeds");
        return false;
    }
    auto points = CurvePoints();
    std::vector<float> flat(3 * points.size());
    for (size_t i = 0; i < points.size(); ++i)
        for (int k = 0; k < 3; ++k) flat[3 * i + size_t(k)] = points[i][k];
    std::vector<uint32_t> offsets{0, 3, 6, 9};
    auto pointsBuf = Upload(context, flat.data(), flat.size() * sizeof(float));
    auto offsetsBuf = Upload(context, offsets.data(), offsets.size() * sizeof(uint32_t));
    if (!pointsBuf || !offsetsBuf) {
        Check(false, "B: Vulkan upload succeeds");
        return false;
    }
    leg->native = native;
    leg->context = context;
    leg->pipe = pipe;
    leg->pointsBuf = pointsBuf;
    leg->offsetsBuf = offsetsBuf;
    leg->ready = true;
    return true;
}

bool RunVulkanCase(VulkanLeg const& leg, VulkanDeformControls const& controls,
                   std::vector<float>* output) {
    auto targetsBuf = Upload(leg.context, controls.rootTargets.data(),
                             controls.rootTargets.size() * sizeof(float));
    if (!targetsBuf) return false;
    DeformPipeline::BeginInfo info;
    info.points = leg.pointsBuf;
    info.curveOffsets = leg.offsetsBuf;
    info.rootTargets = targetsBuf;
    info.curveCount = 3;
    info.pointCount = 9;
    info.restSamples = controls.restSamples;
    info.posedSamples = controls.posedSamples;
    info.sampleCount = controls.sampleCount;
    info.smoothing = 0.0;
    info.mask = {1.0f, 1, nullptr, 0};
    info.enabled = {1, 1, nullptr, 0};
    info.lockRoots = {1, 1, nullptr, 0};
    info.groomEnvelope = 1.0f;
    VkResult status = VK_SUCCESS;
    DeformSemantic sem = DeformSemantic::Ok;
    auto candidate = leg.pipe->Begin(std::move(info), &status, &sem);
    if (!candidate || status != VK_SUCCESS || !Prove(leg.native)) return false;
    if (candidate->Poll(&sem) != VK_SUCCESS || sem != DeformSemantic::Ok) return false;
    if (!candidate->succeeded()) return false;
    return ReadOutput(leg.native, leg.context, candidate, output);
}

// CPU RBF oracle for the Vulkan leg (mirrors testUsdGenVulkanDeform case 5).
bool CpuOracle(VulkanDeformControls const& controls, std::vector<float>* expected) {
    RbfState rstate;
    SolveRbf(reinterpret_cast<vulkan::float3 const*>(controls.restSamples.data()),
             reinterpret_cast<vulkan::float3 const*>(controls.posedSamples.data()),
             controls.sampleCount, 0.0, rstate);
    if (rstate.status != RbfStatus::Code::Ok) return false;
    auto points = CurvePoints();
    std::vector<vulkan::float3> warped(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        vulkan::float3 pt{points[i][0], points[i][1], points[i][2]};
        warped[i] = RbfEvaluate(rstate,
            reinterpret_cast<vulkan::float3 const*>(controls.restSamples.data()), pt);
    }
    DeformApplyParams p;
    p.groomEnvelope = 1.0f;
    p.maskLiteral = 1.0f;
    p.maskHasData = false;
    p.maskDomain = 1;
    p.enabledLiteral = 1;
    p.enabledHasData = false;
    p.enabledDomain = 1;
    p.lockLiteral = 1;
    p.lockHasData = false;
    p.lockDomain = 1;
    std::vector<uint32_t> offsets{0, 3, 6, 9};
    p.offsets = offsets.data();
    p.targets = reinterpret_cast<vulkan::float3 const*>(controls.rootTargets.data());
    p.curveCount = 3;
    p.pointCount = 9;
    std::vector<vulkan::float3> styled(points.size());
    for (size_t i = 0; i < points.size(); ++i)
        styled[i] = vulkan::float3{points[i][0], points[i][1], points[i][2]};
    std::vector<vulkan::float3> out;
    if (!RbfApply(p, styled.data(), warped.data(), &out) || out.size() != 9) return false;
    expected->resize(27);
    for (size_t i = 0; i < 9; ++i) {
        (*expected)[3 * i] = out[i].x;
        (*expected)[3 * i + 1] = out[i].y;
        (*expected)[3 * i + 2] = out[i].z;
    }
    return true;
}

float MaxDelta(std::vector<float> const& a, std::vector<float> const& b) {
    if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
            return std::numeric_limits<float>::infinity();
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

#ifdef USDGEN_ENABLE_CUDA
template <class T>
bool CudaUpload(gpu::DeviceBuffer<T>& device, std::vector<T> const& values) {
    if (device.reset(values.size()) != cudaSuccess) return false;
    return values.empty() || cudaMemcpy(device.data(), values.data(),
                                        values.size() * sizeof(T),
                                        cudaMemcpyHostToDevice) == cudaSuccess;
}

template <class T>
bool CudaDownload(gpu::DeviceView<const T> view, std::vector<T>* values) {
    values->resize(view.size);
    return view.size == 0 || cudaMemcpy(values->data(), view.data,
                                        view.size * sizeof(T),
                                        cudaMemcpyDeviceToHost) == cudaSuccess;
}

template <class T>
gpu::DeviceView<const T> ConstView(gpu::DeviceBuffer<T> const& device) {
    return device.view();
}

struct CudaLeg {
    cudaStream_t producer = nullptr, consumer = nullptr;
    bool ready = false;
};

bool InitCudaLeg(CudaLeg* leg) {
    int devices = 0;
    cudaError_t const result = cudaGetDeviceCount(&devices);
    if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
        (result == cudaSuccess && devices == 0)) {
        std::printf("SKIP: CUDA device unavailable\n");
        return false;
    }
    if (result != cudaSuccess || cudaSetDevice(0) != cudaSuccess ||
        cudaStreamCreate(&leg->producer) != cudaSuccess ||
        cudaStreamCreate(&leg->consumer) != cudaSuccess) {
        Check(false, "B: CUDA device and stream creation succeed");
        if (leg->producer) cudaStreamDestroy(leg->producer);
        if (leg->consumer) cudaStreamDestroy(leg->consumer);
        leg->producer = leg->consumer = nullptr;
        return false;
    }
    leg->ready = true;
    return true;
}

void DoneCudaLeg(CudaLeg* leg) {
    if (leg->consumer) cudaStreamDestroy(leg->consumer);
    if (leg->producer) cudaStreamDestroy(leg->producer);
    leg->consumer = leg->producer = nullptr;
    leg->ready = false;
}

::float3 Bake(GfMatrix4d const& m, GfVec3f const& p) {
    GfVec3d q = m.TransformAffine(GfVec3d(p));
    return ::make_float3(float(q[0]), float(q[1]), float(q[2]));
}

// Baked twin: the same motion with the transforms folded into the points
// and identity matrices, so CUDA accepts it. Output is world-space.
bool RunCudaCase(CudaLeg const& leg, GfMatrix4d const& sourceMatrix,
                 GfMatrix4d const& surfaceMatrix, VulkanDeformControls const& controls,
                 std::vector<float>* outputWorld) {
    auto rest = RestVerts();
    auto posed = PosedVerts();
    auto points = CurvePoints();
    std::vector<::float3> bakedRest, bakedPosed, bakedCurves;
    for (auto const& p : rest) bakedRest.push_back(Bake(surfaceMatrix, p));
    for (auto const& p : posed) bakedPosed.push_back(Bake(surfaceMatrix, p));
    for (auto const& p : points) bakedCurves.push_back(Bake(sourceMatrix, p));
    std::vector<uint32_t> offsetsHost{0, 3, 6, 10};
    std::vector<uint32_t> indicesHost{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int32_t> primHost{0, 1, 2};
    std::vector<::float2> uvHost{::make_float2(.2f, .2f), ::make_float2(.3f, .1f), ::make_float2(.5f, .5f)};
    std::vector<uint32_t> curveOffsetsHost{0, 3, 6, 9};

    gpu::DeviceBuffer<::float3> restBuf, posedBuf, curveBuf, outBuf;
    gpu::DeviceBuffer<uint32_t> offsetsBuf, indicesBuf, curveOffsetsBuf;
    gpu::DeviceBuffer<int32_t> primBuf;
    gpu::DeviceBuffer<::float2> uvBuf;
    if (!CudaUpload(restBuf, bakedRest) || !CudaUpload(posedBuf, bakedPosed) ||
        !CudaUpload(curveBuf, bakedCurves) || !CudaUpload(offsetsBuf, offsetsHost) ||
        !CudaUpload(indicesBuf, indicesHost) || !CudaUpload(primBuf, primHost) ||
        !CudaUpload(uvBuf, uvHost) || !CudaUpload(curveOffsetsBuf, curveOffsetsHost) ||
        outBuf.reset(9) != cudaSuccess)
        return false;

    gpu::CudaSurfaceBinding binding;
    if (binding.Bind(ConstView(restBuf), ConstView(offsetsBuf), 3, ConstView(indicesBuf), 100,
                     leg.producer) != gpu::SurfaceBindingStatus::Ok ||
        binding.Finish(leg.producer) != gpu::SurfaceBindingStatus::Ok ||
        binding.sampleCount() != 10)
        return false;
    if (binding.Update(ConstView(posedBuf), ConstView(primBuf), ConstView(uvBuf),
                       leg.producer) != gpu::SurfaceBindingStatus::Ok ||
        binding.Finish(leg.producer) != gpu::SurfaceBindingStatus::Ok)
        return false;
    // The twin's fit inputs must be the helper's fit mapped forward by the
    // source matrix (world space), up to float rounding.
    std::vector<::float3> cudaRest, cudaPosed, cudaTargets;
    if (!CudaDownload(binding.restSamples(), &cudaRest) || cudaRest.size() != 10 ||
        !CudaDownload(binding.currentSamples(), &cudaPosed) || cudaPosed.size() != 10 ||
        !CudaDownload(binding.rootTargets(), &cudaTargets) || cudaTargets.size() != 3)
        return false;
    auto forward = [&](std::vector<float> const& local) {
        std::vector<float> world(local.size());
        for (size_t i = 0; i < local.size() / 3; ++i) {
            GfVec3d q = sourceMatrix.TransformAffine(
                GfVec3d(local[3 * i], local[3 * i + 1], local[3 * i + 2]));
            world[3 * i] = float(q[0]);
            world[3 * i + 1] = float(q[1]);
            world[3 * i + 2] = float(q[2]);
        }
        return world;
    };
    auto flat = [](std::vector<::float3> const& v) {
        std::vector<float> f(3 * v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            f[3 * i] = v[i].x;
            f[3 * i + 1] = v[i].y;
            f[3 * i + 2] = v[i].z;
        }
        return f;
    };
    if (!SameSet(flat(cudaRest), forward(controls.restSamples), kFitInputTolerance)) return false;
    if (!SameSet(flat(cudaPosed), forward(controls.posedSamples), kFitInputTolerance)) return false;
    auto wantTargets = forward(controls.rootTargets);
    auto gotTargets = flat(cudaTargets);
    for (size_t i = 0; i < wantTargets.size(); ++i)
        if (!Near(gotTargets[i], wantTargets[i], kFitInputTolerance)) return false;

    gpu::CudaRbfBinding rbf;
    if (rbf.Bind(binding.restSamples(), 0.0, leg.producer) != gpu::RbfStatus::Ok ||
        rbf.Solve(binding.currentSamples(), leg.producer) != gpu::RbfStatus::Ok)
        return false;
    gpu::DeviceCurveGeometryView geometry;
    geometry.points = {curveBuf.data(), curveBuf.size()};
    geometry.restPoints = {curveBuf.data(), curveBuf.size()};
    geometry.curveOffsets = {curveOffsetsBuf.data(), curveOffsetsBuf.size()};
    geometry.curveCount = 3;
    geometry.pointCount = 9;
    gpu::CudaRbfCurveDeformer deform;
    if (deform.Deform(rbf, geometry, binding.rootTargets(), 1.0f, {}, {}, outBuf.view(),
                      leg.producer) != gpu::RbfStatus::Ok ||
        deform.Finish(rbf, leg.consumer) != gpu::RbfStatus::Ok)
        return false;
    std::vector<::float3> world;
    gpu::DeviceView<const ::float3> outView{outBuf.data(), outBuf.size()};
    if (!CudaDownload(outView, &world) || world.size() != 9) return false;
    *outputWorld = flat(world);
    return true;
}
#endif // USDGEN_ENABLE_CUDA

struct Case {
    char const* name;
    GfMatrix4d source;
    GfMatrix4d surface;
};

std::vector<Case> Cases() {
    GfMatrix4d movedSource(1.0), movedSurface(1.0);
    movedSource[3][0] = 5.0;
    movedSurface[3][1] = -2.0;
    return {{"moved source", movedSource, GfMatrix4d(1.0)},
            {"moved surface", GfMatrix4d(1.0), movedSurface},
            {"moved source+surface", movedSource, movedSurface}};
}

void PartB(char const* evalPath, char const* applyPath) {
    VulkanLeg vleg;
    bool const vulkanReady = InitVulkanLeg(evalPath, applyPath, &vleg);
#ifdef USDGEN_ENABLE_CUDA
    CudaLeg cleg;
    bool const cudaReady = InitCudaLeg(&cleg);
#else
    std::printf("SKIP: CUDA twin unavailable (CUDA off)\n");
#endif
    auto points = CurvePoints();
    std::vector<float> inputLocal(27);
    for (size_t i = 0; i < 9; ++i)
        for (int k = 0; k < 3; ++k) inputLocal[3 * i + size_t(k)] = points[i][k];

    for (auto const& c : Cases()) {
        std::string tag = std::string("B: ") + c.name;
        VulkanDeformControls controls;
        if (!CompileForDevice(tag.c_str(), c.source, c.surface, &controls)) continue;
        bool vulkanRan = false, cudaRan = false;
        std::vector<float> vulkanOut, cudaWorld;
        if (vulkanReady) {
            if (!RunVulkanCase(vleg, controls, &vulkanOut) || vulkanOut.size() != 27) {
                Check(false, (tag + " (Vulkan leg runs)").c_str());
            } else {
                vulkanRan = true;
                std::vector<float> oracle;
                bool const oracleOk = CpuOracle(controls, &oracle);
                float const drift = oracleOk ? MaxDelta(vulkanOut, oracle) : -1.0f;
                std::printf("      vulkan-vs-oracle max delta: %.3g\n", drift);
                Check(oracleOk && drift <= kOracleTolerance,
                      (tag + " (Vulkan leg matches CPU oracle)").c_str());
                float const moved = MaxDelta(vulkanOut, inputLocal);
                std::printf("      vulkan deformation magnitude: %.3g\n", moved);
                Check(moved > 1.0e-3f, (tag + " (Vulkan leg deforms)").c_str());
            }
        } else {
            std::printf("SKIP: %s (Vulkan leg)\n", tag.c_str());
        }
#ifdef USDGEN_ENABLE_CUDA
        if (cudaReady) {
            if (!RunCudaCase(cleg, c.source, c.surface, controls, &cudaWorld) ||
                cudaWorld.size() != 27) {
                Check(false, (tag + " (CUDA twin runs with mapped fit inputs)").c_str());
            } else {
                cudaRan = true;
                Check(true, (tag + " (CUDA twin runs with mapped fit inputs)").c_str());
                std::vector<float> inputWorld(27);
                for (size_t i = 0; i < 9; ++i) {
                    GfVec3d q = c.source.TransformAffine(GfVec3d(points[i]));
                    inputWorld[3 * i] = float(q[0]);
                    inputWorld[3 * i + 1] = float(q[1]);
                    inputWorld[3 * i + 2] = float(q[2]);
                }
                float const moved = MaxDelta(cudaWorld, inputWorld);
                std::printf("      cuda deformation magnitude: %.3g\n", moved);
                Check(moved > 1.0e-3f, (tag + " (CUDA twin deforms)").c_str());
            }
        } else {
            std::printf("SKIP: %s (CUDA twin)\n", tag.c_str());
        }
#endif
        if (vulkanRan && cudaRan) {
            g_crossBackendCompared = true;
            // Map the twin back to source-local space and compare.
            double det = 0.0;
            GfMatrix4d const back = c.source.GetInverse(&det);
            std::vector<float> cudaLocal(27);
            for (size_t i = 0; i < 9; ++i) {
                GfVec3d q = back.TransformAffine(GfVec3d(
                    cudaWorld[3 * i], cudaWorld[3 * i + 1], cudaWorld[3 * i + 2]));
                cudaLocal[3 * i] = float(q[0]);
                cudaLocal[3 * i + 1] = float(q[1]);
                cudaLocal[3 * i + 2] = float(q[2]);
            }
            float const parity = MaxDelta(vulkanOut, cudaLocal);
            std::printf("      vulkan-vs-cuda max delta: %.3g (tolerance %.1g)\n", parity,
                        kCudaParityTolerance);
            Check(parity <= kCudaParityTolerance, (tag + " (parity within CUDA Near)").c_str());
        } else {
            std::printf("SKIP: %s (parity)\n", tag.c_str());
        }
    }
#ifdef USDGEN_ENABLE_CUDA
    if (cleg.ready) DoneCudaLeg(&cleg);
#endif
}

// ---- Part C: plan acceptance (requires the phase2 surface plan hook) ----

UsdGenGraphDesc MakeDeformDesc(GfMatrix4d const& sourceMatrix,
                               GfMatrix4d const& surfaceMatrix) {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom");
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");
    auto curves = MakeSource();
    curves.worldMatrix = sourceMatrix;
    desc.curveSets.push_back(curves);
    auto surface = MakeSurface();
    surface.worldMatrix = surfaceMatrix;
    desc.surfaces.push_back(surface);

    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.surfaces = {surface.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
        {TfToken("idSource"), VtValue(TfToken("primvar")), false},
        {TfToken("expectEpoch"), VtValue(std::string("epoch-19")), false},
        {TfToken("staleAction"), VtValue(TfToken("block")), false},
        {TfToken("rebind"), VtValue(TfToken("never")), false}};
    UsdGenNodeDesc deform;
    deform.path = SdfPath("/Groom/Ops/deform");
    deform.type = TfToken("UsdGenDeform");
    deform.inputs = {source.path};
    deform.surfaces = {surface.path};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Ops/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {deform.path};
    width.params = {{TfToken("width"), VtValue(.125f), false},
        {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, deform, width};
    desc.terminal = width.path;
    return desc;
}

void CheckPlanCase(char const* label, GfMatrix4d const& sourceMatrix,
                   GfMatrix4d const& surfaceMatrix, bool expectHooked) {
    auto desc = MakeDeformDesc(sourceMatrix, surfaceMatrix);
    UsdGenDiagnostics diagnostics;
    auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc), &diagnostics);
    if (!handle || diagnostics.HasErrors()) {
        std::printf("      diagnostic: %s\n",
                    diagnostics.errors.empty() ? "<none>" : diagnostics.errors.front().c_str());
        Check(false, label);
        return;
    }
    auto plan = std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
    bool ok = plan && plan->Steps().size() == 2 &&
        plan->Steps()[0].kind == VulkanSourceWidthStage::Kind::Deform &&
        plan->Steps()[0].deform.sampleCount == 10;
    if (ok && expectHooked) {
        // The hooked plan must freeze exactly the helper's controls.
        VulkanDeformControls want;
        want.lockRoots = true;
        want.mask = 1.0f;
        want.groomEnvelope = 1.0f;
        ok = CompileSurfaceVkDeform(usdGenUsd::Lower(desc.curveSets[0]), usdGenUsd::Lower(desc.surfaces[0]), 100, &want, nullptr) &&
            plan->Steps()[0].deform.restSamples == want.restSamples &&
            plan->Steps()[0].deform.posedSamples == want.posedSamples &&
            plan->Steps()[0].deform.rootTargets == want.rootTargets;
    }
    Check(ok, label);
}

void PartC() {
    GfMatrix4d movedSource(1.0), movedSurface(1.0);
    movedSource[3][0] = 5.0;
    movedSurface[3][1] = -2.0;
    CheckPlanCase("C: plan accepts moved source with helper controls", movedSource,
                  GfMatrix4d(1.0), true);
    CheckPlanCase("C: plan accepts moved surface with helper controls", GfMatrix4d(1.0),
                  movedSurface, true);
    CheckPlanCase("C: plan accepts moved source+surface with helper controls", movedSource,
                  movedSurface, true);
    // The admitted identity config keeps its untouched legacy host-gather
    // path: acceptance plus sample count only, no value comparison.
    CheckPlanCase("C: control identity plan still accepts", GfMatrix4d(1.0),
                  GfMatrix4d(1.0), false);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s deformEvaluate.spv deformApply.spv\n", argv[0]);
        return 1;
    }
    PartA();
    PartC();
    PartB(argv[1], argv[2]);
    std::printf("surfaceVk: %d failure(s)\n", g_failures);
    return g_failures ? 1 : (g_crossBackendCompared ? 0 : 77);
}
