// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// benchUsdGenRbfDeform — RBF deformation execution-time breakdown.
//
// Workload D1: scatter -> grow -> deform (surface-driven, rbfSamples=100) ->
// width over a wavy 400x250 grid surface (spans 3D so the RBF binds), plus a
// direct rbf::CubicField microbench that isolates Bind/Solve/Displacement.
//
// Usage: benchUsdGenRbfDeform [--threads N] [--density D] [--vulkan-spv dir]
//   [--vulkan-budget-mb MB]
// --threads sets the scheduler arena width; 0 (the default) resolves the
// product default (I8: max(8, fast-core count) on heterogeneous Linux).
// --vulkan-budget-mb sizes the deform leg's pool (default 512); larger
// budgets show what the budget-adaptive R-cache prefix does at
// production scale (a 2048MiB budget funds the full 1M-CV prefix).
// Output: one "METRIC <name>=<value_ms>" line per measured quantity
// (median of 9, fixture built once outside every timing window), plus
// CHECKSUM lines (FNV-1a over float bits) for bit-identity tracking.
// Exit is nonzero on hard failure only (compile/run errors, NaN output,
// nondeterministic repeat); timings are informational.
//
// With USDGEN_BENCH_CUDA the CUDA CudaRbfBinding path is measured too
// (Bind/Solve/Evaluate via CUDA events, n=100 samples, 1M CVs).
//
// With USDGEN_BENCH_VULKAN and --vulkan-spv <dir> (a directory holding the
// deformEvaluate/deformApply/rbfVk*.spv files), the Vulkan DeformPipeline
// per-pose time (1M CVs, n=100) and the RbfVkBinding Bind/Solve/
// EvaluateHost times (n=400, 1M CVs) are measured too. Without the flag
// the Vulkan legs are skipped; an unavailable device skips (exit 77).

#include "usdGen/compiler.h"
#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/opRegistry.h"
#include "usdGen/ops/rbfField.h"
#include "usdGen/scheduler.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef USDGEN_BENCH_CUDA
#include "usdGen/gpu/rbf.h"
#include <cuda_runtime.h>
#endif

#ifdef USDGEN_BENCH_VULKAN
#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deformPipeline.h"
#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/rbfVkBinding.h"
#include "../vulkanNativeFixture.h"
#include "../vulkanReadbackFixture.h"
#include <fstream>
#endif

PXR_NAMESPACE_USING_DIRECTIVE
using namespace usdGen;

namespace {

int g_hardFail = 0;

double Median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double NowMs()
{
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void EmitMetric(char const *name, double ms)
{
    std::printf("METRIC %s=%.4f\n", name, ms);
}

uint64_t Fnv1a(void const *data, size_t bytes)
{
    auto const *p = static_cast<unsigned char const *>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= uint64_t(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

double Hash01(uint64_t k)
{
    k += 0x9e3779b97f4a7c15ULL;
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    return double((k ^ (k >> 31)) >> 11) / double(1ull << 53);
}

// --- field microbench fixture (built once, outside timing) ------------------

struct FieldFixture {
    std::vector<GfVec3d> rest;
    std::vector<GfVec3d> posed;
    std::vector<GfVec3d> queries;
};

FieldFixture MakeFieldFixture(int samples, size_t queryCount)
{
    FieldFixture f;
    f.rest.resize(size_t(samples));
    f.posed.resize(size_t(samples));
    for (int i = 0; i < samples; ++i) {
        GfVec3d p(Hash01(uint64_t(i) * 3), Hash01(uint64_t(i) * 3 + 1),
                  Hash01(uint64_t(i) * 3 + 2));
        f.rest[i] = p * 2.0 - GfVec3d(1.0);
        f.posed[i] = f.rest[i] + GfVec3d(0.2 * std::sin(3.0 * p[1]), 0.1 * p[0] * p[2],
                                         -0.15 * std::cos(2.0 * p[0]));
    }
    f.queries.resize(queryCount);
    for (size_t i = 0; i < queryCount; ++i)
        f.queries[i] = GfVec3d(Hash01(1001 + i * 3), Hash01(1001 + i * 3 + 1),
                               Hash01(1001 + i * 3 + 2)) * 2.0 - GfVec3d(1.0);
    return f;
}

// --- D1 graph fixture --------------------------------------------------------
// Wavy grid: the RBF needs samples spanning 3D, so rest z is a smooth bump
// field, not a plane.

UsdGenGraphDesc MakeD1(double density)
{
    UsdGenGraphDesc d;
    d.description = SdfPath("/groom");
    d.terminal = SdfPath("/groom/width");
    d.time = 0.0;

    UsdGenSurfaceDesc s;
    s.path = SdfPath("/groom/surface");
    s.id = 0;
    const int NX = 400, NY = 250;
    s.restPoints = VtVec3fArray((NX + 1) * (NY + 1));
    s.uv = VtVec2fArray((NX + 1) * (NY + 1));
    for (int j = 0; j <= NY; ++j)
        for (int i = 0; i <= NX; ++i) {
            const int k = j * (NX + 1) + i;
            float const x = float(i) * 0.1f, y = float(j) * 0.1f;
            float const z = 2.0f * std::sin(x * 0.3f) * std::cos(y * 0.3f);
            s.restPoints[k] = GfVec3f(x, y, z);
            s.uv[k] = GfVec2f(float(i) / NX, float(j) / NY);
        }
    s.points = s.restPoints;
    s.faceVertexCounts = VtIntArray(NX * NY, 4);
    s.faceVertexIndices = VtIntArray(NX * NY * 4);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i) {
            const int a = j * (NX + 1) + i;
            const int o = (j * NX + i) * 4;
            s.faceVertexIndices[o + 0] = a;
            s.faceVertexIndices[o + 1] = a + 1;
            s.faceVertexIndices[o + 2] = a + NX + 2;
            s.faceVertexIndices[o + 3] = a + NX + 1;
        }
    d.surfaces.push_back(std::move(s));

    auto addNode = [&](std::string const &name, TfToken type,
                       std::string const &input, int seed) {
        UsdGenNodeDesc n;
        n.path = SdfPath("/groom/" + name);
        n.type = type;
        n.enabled = true;
        n.seed = seed;
        if (!input.empty()) n.inputs.push_back(SdfPath("/groom/" + input));
        if (type == TfToken("UsdGenScatter") || type == TfToken("UsdGenDeform"))
            n.surfaces.push_back(SdfPath("/groom/surface"));
        d.nodes.push_back(std::move(n));
    };
    addNode("scatter", TfToken("UsdGenScatter"), "", 42);
    addNode("grow", TfToken("UsdGenGrow"), "scatter", 43);
    addNode("deform", TfToken("UsdGenDeform"), "grow", 44);
    addNode("width", TfToken("UsdGenWidth"), "deform", 45);

    auto setp = [&](std::string const &name, TfToken p, VtValue v) {
        for (auto &n : d.nodes)
            if (n.path == SdfPath("/groom/" + name))
                n.params.push_back(UsdGenParamValue{p, v, false});
    };
    setp("scatter", TfToken("density"), VtValue(density));
    setp("grow", TfToken("segments"), VtValue(8));
    setp("grow", TfToken("length"), VtValue(1.0));
    setp("deform", TfToken("rbfSamples"), VtValue(100));
    setp("deform", TfToken("lockRoots"), VtValue(true));
    setp("width", TfToken("width"), VtValue(0.02));
    return d;
}

#ifdef USDGEN_BENCH_VULKAN
// --- Vulkan legs -------------------------------------------------------------

std::vector<uint32_t> LoadSpv(std::string const &path)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}

// Spin a deform candidate to completion. Production never blocks on the
// apply fence (sourceWidthJob drives Poll from its completion service and
// returns on VK_NOT_READY), so the bench must not either: the old empty
// submit plus blocking vkWaitForFences measured the futex wakeup lottery
// (~180us median, +-125us jitter over the ~144us of real apply work)
// instead of the GPU and host work. A pure spin observes the fence within
// one vkGetFenceStatus of the signal; the 30s deadline only trips on a
// wedged device (well under the ctest timeout) and fails the leg.
VkResult PollUntilReady(vulkan::DeformPipeline::Candidate *c,
                        vulkan::DeformSemantic *sem)
{
    double const deadline = NowMs() + 30000.0;
    for (;;) {
        VkResult const r = c->Poll(sem);
        if (r != VK_NOT_READY) return r;
        if (NowMs() > deadline) return VK_TIMEOUT;
    }
}

std::shared_ptr<const vulkan::ChargedBuffer> UploadHost(
    std::shared_ptr<vulkan::DeviceContext> const &c, void const *data, VkDeviceSize bytes)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = vulkan::ChargedBuffer::Create(c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b) return {};
    if (bytes) {
        void *p = nullptr;
        if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS) return {};
        std::memcpy(p, data, bytes);
        vkUnmapMemory(c->device(), b->memory());
    }
    return b;
}

// Main path: DeformPipeline per-pose (Begin + spin-poll to completion) over
// 100k curves x 10 CVs with 100 RBF samples. CHECKSUM covers the last
// pose's deformed points.
int VulkanDeformLeg(std::string const &spvDir, size_t budgetBytes)
{
    std::vector<uint32_t> const evalSpv = LoadSpv(spvDir + "/deformEvaluate.spv");
    std::vector<uint32_t> const applySpv = LoadSpv(spvDir + "/deformApply.spv");
    vulkan::DeformEvalCacheSpirv cacheSpv;
    cacheSpv.verify = LoadSpv(spvDir + "/deformEvaluateVerify.spv");
    cacheSpv.fill = LoadSpv(spvDir + "/deformEvaluateFill.spv");
    cacheSpv.cached = LoadSpv(spvDir + "/deformEvaluateCached.spv");
    if (evalSpv.empty() || applySpv.empty() || cacheSpv.verify.empty() ||
        cacheSpv.fill.empty() || cacheSpv.cached.empty()) {
        std::printf("Vulkan deform leg: spirv not found in %s\n", spvDir.c_str());
        return 1;
    }
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64 = {};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, nullptr, &fp64);
    if (unavailable || !native) {
        std::printf("Vulkan deform leg: no fp64 device\n");
        return 77;
    }
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(native->physical, &feats);
    if (!feats.shaderFloat64) {
        std::printf("Vulkan deform leg: no fp64 device\n");
        return 77;
    }
    vulkan::DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8020;
    ci.nativeLifetime = native;
    ci.resources = {budgetBytes, 0};
    ci.shaderFloat64Enabled = true;
    auto context = vulkan::DeviceContext::Create(ci);
    if (!context) {
        std::printf("Vulkan deform leg: context creation failed\n");
        return 1;
    }
    VkResult status = VK_SUCCESS;
    auto pipe = vulkan::DeformPipeline::Create(context, evalSpv, applySpv, &status, cacheSpv);
    if (!pipe || status != VK_SUCCESS) {
        std::printf("Vulkan deform leg: pipeline creation failed\n");
        return 1;
    }
    uint32_t const curves = 100000, perCurve = 10, points = curves * perCurve;
    int const n = 100;
    std::vector<float> pointsData(size_t(points) * 3);
    for (uint32_t i = 0; i < points; ++i)
        for (int a = 0; a < 3; ++a)
            pointsData[size_t(i) * 3 + size_t(a)] =
                float(2 * Hash01(1001 + size_t(i) * 3 + size_t(a)) - 1);
    std::vector<uint32_t> offsets(curves + 1);
    for (uint32_t c = 0; c <= curves; ++c) offsets[c] = c * perCurve;
    std::vector<float> targets(size_t(curves) * 3);
    for (uint32_t c = 0; c < curves; ++c)
        for (int a = 0; a < 3; ++a)
            targets[size_t(c) * 3 + size_t(a)] =
                pointsData[size_t(c) * perCurve * 3 + size_t(a)];
    std::vector<float> restSamples(size_t(n) * 3), posedSamples(size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
        float const rx = float(2 * Hash01(size_t(i) * 3) - 1);
        float const ry = float(2 * Hash01(size_t(i) * 3 + 1) - 1);
        float const rz = float(2 * Hash01(size_t(i) * 3 + 2) - 1);
        restSamples[size_t(i) * 3] = rx;
        restSamples[size_t(i) * 3 + 1] = ry;
        restSamples[size_t(i) * 3 + 2] = rz;
        posedSamples[size_t(i) * 3] = rx + 0.1f * float(std::sin(double(i)));
        posedSamples[size_t(i) * 3 + 1] = ry - 0.05f * float(i % 7);
        posedSamples[size_t(i) * 3 + 2] = rz + 0.07f * float(std::cos(2.0 * double(i)));
    }
    auto pointsBuf = UploadHost(context, pointsData.data(), pointsData.size() * sizeof(float));
    auto offsetsBuf = UploadVulkanDeviceBytes(native, context, offsets.data(),
                                              offsets.size() * sizeof(uint32_t));
    auto targetsBuf = UploadHost(context, targets.data(), targets.size() * sizeof(float));
    if (!pointsBuf || !offsetsBuf || !targetsBuf) {
        std::printf("Vulkan deform leg: upload failed\n");
        return 1;
    }
    std::vector<double> tPose;
    uint64_t checksum = 0;
    for (int iter = 0; iter < 11; ++iter) {
        std::vector<float> posed = posedSamples;
        float const w = 0.01f * float(iter);
        for (int i = 0; i < n; ++i) posed[size_t(i) * 3] += w * float(i % 5);
        vulkan::DeformPipeline::BeginInfo info;
        info.points = pointsBuf;
        info.curveOffsets = offsetsBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = curves;
        info.pointCount = points;
        info.restSamples = restSamples;
        info.posedSamples = posed;
        info.sampleCount = n;
        info.smoothing = 0.0;
        info.mask = {1.0f, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.lockRoots = {0, 1, nullptr, 0};
        info.groomEnvelope = 1.0f;
        double const t0 = NowMs();
        vulkan::DeformSemantic sem = vulkan::DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        VkResult const pr =
            (c && status == VK_SUCCESS) ? PollUntilReady(c.get(), &sem) : VK_NOT_READY;
        if (!c || status != VK_SUCCESS ||
            pr != VK_SUCCESS || sem != vulkan::DeformSemantic::Ok || !c->succeeded()) {
            std::printf("Vulkan deform leg: pose %d failed\n", iter);
            return 1;
        }
        if (iter >= 2) tPose.push_back(NowMs() - t0);
        if (iter == 10) {
            auto o = c->output();
            std::vector<uint8_t> bytes;
            if (!ReadVulkanBytes(native, context, o.points->buffer(), o.points->sizeBytes(),
                                 o.points, &bytes)) {
                std::printf("Vulkan deform leg: readback failed\n");
                return 1;
            }
            checksum = Fnv1a(bytes.data(), bytes.size());
        }
    }
    EmitMetric("vk_deform_pose_1m", Median(tPose));
    std::printf("CHECKSUM vk_deform_pose=%016llx\n", (unsigned long long)checksum);
    return 0;
}

// Device path: RbfVkBinding Bind/Solve/EvaluateHost, n=400, 1M CVs.
int VulkanBindLeg(std::string const &spvDir)
{
    vulkan::RbfVkBindingSpirv spirv;
    spirv.extent = LoadSpv(spvDir + "/rbfVkExtent.spv");
    spirv.gram = LoadSpv(spvDir + "/rbfVkGram.spv");
    spirv.buildMatrix = LoadSpv(spvDir + "/rbfVkBuildMatrix.spv");
    spirv.lu = LoadSpv(spvDir + "/rbfVkLu.spv");
    spirv.rhs = LoadSpv(spvDir + "/rbfVkRhs.spv");
    spirv.triSolve = LoadSpv(spvDir + "/rbfVkTriSolve.spv");
    spirv.evaluate = LoadSpv(spvDir + "/rbfVkEvaluate.spv");
    spirv.verify = LoadSpv(spvDir + "/rbfVkVerify.spv");
    spirv.fill = LoadSpv(spvDir + "/rbfVkFill.spv");
    spirv.evaluateCached = LoadSpv(spvDir + "/rbfVkEvaluateCached.spv");
    if (spirv.extent.empty() || spirv.evaluate.empty()) {
        std::printf("Vulkan bind leg: spirv not found in %s\n", spvDir.c_str());
        return 1;
    }
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64{};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, nullptr, &fp64);
    if (unavailable || !native) {
        std::printf("Vulkan bind leg: no fp64 device\n");
        return 77;
    }
    vulkan::DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8021;
    ci.nativeLifetime = native;
    // The R cache needs R (count*n doubles) plus a 12MB CV proof: 3.2GB
    // at the n=400/1M bench shape. Production pools are heap-sized, so
    // the old 256MB cap (which predates the cache) would only forbid an
    // engagement production takes.
    ci.resources = {size_t{4} << 30, 0};
    ci.shaderFloat64Enabled = true;
    auto context = vulkan::DeviceContext::Create(ci);
    if (!context) {
        std::printf("Vulkan bind leg: context creation failed\n");
        return 1;
    }
    VkResult status = VK_SUCCESS;
    auto binding = vulkan::RbfVkBinding::Create(context, spirv, &status);
    if (!binding || status != VK_SUCCESS) {
        std::printf("Vulkan bind leg: binding creation failed\n");
        return 1;
    }
    // Steady state, like production gap poses: rest is static across binds,
    // so the shared factor cache adopts the LU after the first bind.
    auto factorCache = std::make_shared<vulkan::RbfVkFactorCache>();
    binding->SetFactorCache(factorCache);
    // Same for the pose-invariant radii: the CVs are static across the
    // repeated evaluations, so only the first pays the sqrt fill.
    auto evalCache = std::make_shared<vulkan::RbfVkEvalCache>();
    binding->SetEvalCache(evalCache);
    int const n = 400;
    uint32_t const cvs = 1000000;
    std::vector<float> rest(size_t(n) * 3), posed(size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
        float const rx = float(2 * Hash01(size_t(i) * 3) - 1);
        float const ry = float(2 * Hash01(size_t(i) * 3 + 1) - 1);
        float const rz = float(2 * Hash01(size_t(i) * 3 + 2) - 1);
        rest[size_t(i) * 3] = rx;
        rest[size_t(i) * 3 + 1] = ry;
        rest[size_t(i) * 3 + 2] = rz;
        posed[size_t(i) * 3] = rx + 0.1f * float(std::sin(double(i)));
        posed[size_t(i) * 3 + 1] = ry - 0.05f * float(i % 7);
        posed[size_t(i) * 3 + 2] = rz + 0.07f * float(std::cos(2.0 * double(i)));
    }
    std::vector<float> cv(size_t(cvs) * 3);
    for (uint32_t i = 0; i < cvs; ++i)
        for (int a = 0; a < 3; ++a)
            cv[size_t(i) * 3 + size_t(a)] =
                float(2 * Hash01(1001 + size_t(i) * 3 + size_t(a)) - 1);
    std::vector<float> out;
    out.resize(size_t(cvs) * 3);
    std::vector<double> tBind, tSolve, tEval;
    for (int i = 0; i < 5; ++i) {
        double t0 = NowMs();
        if (binding->Bind(rest.data(), n, 0.0) != vulkan::RbfVkStatus::Ok) {
            std::printf("Vulkan bind leg: bind failed: %s\n", binding->diagnostic());
            return 1;
        }
        tBind.push_back(NowMs() - t0);
        t0 = NowMs();
        if (binding->Solve(posed.data(), n) != vulkan::RbfVkStatus::Ok) {
            std::printf("Vulkan bind leg: solve failed: %s\n", binding->diagnostic());
            return 1;
        }
        tSolve.push_back(NowMs() - t0);
        t0 = NowMs();
        if (binding->EvaluateHost(cv.data(), out.data(), cvs) != vulkan::RbfVkStatus::Ok) {
            std::printf("Vulkan bind leg: evaluate failed: %s\n", binding->diagnostic());
            return 1;
        }
        tEval.push_back(NowMs() - t0);
    }
    EmitMetric("vk_rbf_bind400", Median(tBind));
    EmitMetric("vk_rbf_solve400", Median(tSolve));
    EmitMetric("vk_rbf_evaluate_1m", Median(tEval));
    std::printf("CHECKSUM vk_rbf_eval=%016llx\n",
                (unsigned long long)Fnv1a(out.data(), out.size() * sizeof(float)));
    return 0;
}
#endif  // USDGEN_BENCH_VULKAN

// Deterministic animated pose k over the rest surface (smooth bend).
void PoseSurface(UsdGenSurfaceDesc *s, int k)
{
    // Read rest through a const reference: s is non-const, so s->restPoints[i]
    // would take the mutating subscript and detach (deep-copy) the shared
    // 1.2MB rest buffer on every pose.
    VtVec3fArray const &rp = s->restPoints;
    size_t const n = rp.size();
    s->points.resize(n);
    double const t = 0.35 * double(k);
    for (size_t i = 0; i < n; ++i) {
        GfVec3f const r = rp[i];
        float const bend = float(0.6 * std::sin(0.05 * r[0] + t) * std::cos(0.04 * r[1] - 0.5 * t));
        s->points[i] = GfVec3f(r[0], r[1], r[2] + bend);
    }
}

}  // namespace

int main(int argc, char **argv)
{
    int threads = 0;  // the product default (see the usage note above)
    double density = 25.0;   // ~25k curves, 225k CVs at 8 segments
    std::string vulkanSpvDir;
    size_t vulkanBudgetMb = 512;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        if (std::string(argv[i]) == "--density" && i + 1 < argc) density = std::atof(argv[++i]);
        if (std::string(argv[i]) == "--vulkan-spv" && i + 1 < argc) vulkanSpvDir = argv[++i];
        if (std::string(argv[i]) == "--vulkan-budget-mb" && i + 1 < argc)
            vulkanBudgetMb = size_t(std::atoll(argv[++i]));
    }
    if (vulkanBudgetMb < 1) vulkanBudgetMb = 1;
    usdGenRegisterM1Operators();

    // ---- field microbench: Bind / Solve / Displacement ----------------------
    {
        FieldFixture const f = MakeFieldFixture(100, 200000);
        std::string error;
        rbf::CubicField field;
        double t0 = NowMs();
        bool const bound = field.Bind(f.rest, &error);
        double const bindMs = NowMs() - t0;
        if (!bound) {
            std::printf("FIELD bind failed: %s\n", error.c_str());
            return 1;
        }
        EmitMetric("cpu_rbf_bind", bindMs);

        std::vector<double> tSolve, tDisp;
        std::vector<GfVec3d> disp(f.queries.size());
        uint64_t checksum = 0;
        for (int i = 0; i < 9; ++i) {
            t0 = NowMs();
            if (!field.Solve(f.posed, &error)) {
                std::printf("FIELD solve failed: %s\n", error.c_str());
                return 1;
            }
            tSolve.push_back(NowMs() - t0);
            t0 = NowMs();
            // The production batch API over the same 200k queries; the
            // accumulation below runs in query order as before, so the
            // checksum is comparable across the change.
            field.DisplaceBatch(f.queries.data(), disp.data(), f.queries.size());
            double acc[3] = {0, 0, 0};
            for (GfVec3d const &d : disp) {
                acc[0] += d[0]; acc[1] += d[1]; acc[2] += d[2];
            }
            tDisp.push_back(NowMs() - t0);
            if (i == 0) checksum = Fnv1a(acc, sizeof(acc));
            if (!std::isfinite(acc[0] + acc[1] + acc[2])) {
                std::printf("FIELD displacement produced non-finite output\n");
                return 1;
            }
        }
        EmitMetric("cpu_rbf_solve", Median(tSolve));
        EmitMetric("cpu_rbf_displace_200k", Median(tDisp));
        std::printf("CHECKSUM cpu_rbf_field=%016llx\n", (unsigned long long)checksum);
    }

    // ---- D1 graph: compile once, then animated re-commit --------------------
    UsdGenGraphDesc base = MakeD1(density);
    UsdGenCompiler compiler;
    UsdGenGraph graph;
    {
        std::vector<double> tCompile;
        for (int i = 0; i < 9; ++i) {
            UsdGenGraph g;
            double t0 = NowMs();
            UsdGenCompileResult const r = compiler.Compile(base, &g);
            tCompile.push_back(NowMs() - t0);
            if (!r.ok) {
                std::printf("D1 compile failed:");
                for (auto const &e : r.errors) std::printf(" %s", e.c_str());
                std::printf("\n");
                return 1;
            }
            if (i == 0) graph = std::move(g);
        }
        EmitMetric("cpu_compile", Median(tCompile));
    }
    UsdGenEvalContext ctx;
    UsdGenScheduler scheduler(threads);
    uint64_t gen = 0;
    UsdGenGraphDesc desc = base;
    PoseSurface(&desc.surfaces[0], 0);
    ctx.desc = nullptr;
    UsdGenCompileResult const warm = compiler.Recompile(desc, &graph);
    ctx.desc = &graph.Desc();
    if (!warm.ok || scheduler.Run(graph, ctx, ++gen).diagnostics.HasErrors()) {
        std::printf("D1 warmup run failed\n");
        return 1;
    }
    // Repeatability: the same pose twice must be bitwise identical.
    auto terminalChecksum = [&]() {
        UsdGenCurveBuffer const &t = graph.Node(graph.NodeIdForPath(desc.terminal)).buffer;
        uint64_t h = Fnv1a(t.px.cdata(), t.px.size() * sizeof(float));
        h ^= Fnv1a(t.py.cdata(), t.py.size() * sizeof(float)) * 1099511628211ULL;
        h ^= Fnv1a(t.pz.cdata(), t.pz.size() * sizeof(float)) * 1099511628211ULL;
        return h;
    };
    uint64_t const repeat0 = terminalChecksum();
    compiler.Recompile(desc, &graph);
    ctx.desc = &graph.Desc();
    scheduler.Run(graph, ctx, ++gen);
    if (terminalChecksum() != repeat0) {
        std::printf("D1 repeatability FAIL: same pose cooked different bits\n");
        return 1;
    }
    std::printf("  curves=%u cvs=%u\n",
                graph.Node(graph.NodeIdForPath(desc.terminal)).buffer.totalCurves,
                graph.Node(graph.NodeIdForPath(desc.terminal)).buffer.totalCvs);

    std::vector<double> tCommit, tDeformCapture, tDeformEval;
    uint64_t lastChecksum = 0;
    for (int i = 1; i <= 9; ++i) {
        PoseSurface(&desc.surfaces[0], i);
        double t0 = NowMs();
        UsdGenCompileResult const rr = compiler.Recompile(desc, &graph);
        ctx.desc = &graph.Desc();
        UsdGenRunResult const run = scheduler.Run(graph, ctx, ++gen);
        tCommit.push_back(NowMs() - t0);
        if (!rr.ok || run.diagnostics.HasErrors()) {
            std::printf("D1 commit %d failed\n", i);
            return 1;
        }
        for (UsdGenNodeRunStats const &st : run.nodeStats)
            if (graph.Node(st.id).type == TfToken("UsdGenDeform")) {
                tDeformCapture.push_back(st.captureMs);
                tDeformEval.push_back(st.evalMs);
            }
        lastChecksum = terminalChecksum();
    }
    EmitMetric("cpu_commit_pose", Median(tCommit));
    EmitMetric("cpu_deform_capture", Median(tDeformCapture));
    EmitMetric("cpu_deform_evaluate", Median(tDeformEval));
    std::printf("CHECKSUM cpu_d1_terminal=%016llx\n", (unsigned long long)lastChecksum);

#ifdef USDGEN_BENCH_CUDA
    // ---- CUDA: direct CudaRbfBinding Bind/Solve/Evaluate --------------------
    // 77 (no usable device) records a skip and continues to the Vulkan
    // legs; bind/solve/evaluate errors are hard failures.
    int cudaLegExit = [&]() {
    {
        const int n = 100, cvs = 1000000, m = n + 4;
        std::vector<float3> rest(n), posed(n), cv(cvs);
        for (int i = 0; i < n; ++i) {
            rest[i] = make_float3(float(2 * Hash01(uint64_t(i) * 3) - 1),
                                  float(2 * Hash01(uint64_t(i) * 3 + 1) - 1),
                                  float(2 * Hash01(uint64_t(i) * 3 + 2) - 1));
            posed[i] = make_float3(rest[i].x + 0.1f * float(std::sin(double(i))),
                                   rest[i].y - 0.05f * float(i % 7),
                                   rest[i].z + 0.07f * float(std::cos(2.0 * double(i))));
        }
        for (int i = 0; i < cvs; ++i)
            cv[i] = make_float3(float(2 * Hash01(1001 + uint64_t(i) * 3) - 1),
                                float(2 * Hash01(1001 + uint64_t(i) * 3 + 1) - 1),
                                float(2 * Hash01(1001 + uint64_t(i) * 3 + 2) - 1));
        float3 *dRest = nullptr, *dPosed = nullptr, *dCv = nullptr, *dOut = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t start = nullptr, stop = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
            cudaMalloc(&dRest, size_t(n) * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&dPosed, size_t(n) * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&dCv, size_t(cvs) * sizeof(float3)) != cudaSuccess ||
            cudaMalloc(&dOut, size_t(cvs) * sizeof(float3)) != cudaSuccess) {
            // Environment, not code: no usable device (or no free memory).
            // Bind/solve/evaluate errors below stay hard failures.
            std::printf("CUDA setup unavailable; skipping CUDA leg\n");
            return 77;
        }
        cudaMemcpy(dRest, rest.data(), size_t(n) * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(dPosed, posed.data(), size_t(n) * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(dCv, cv.data(), size_t(cvs) * sizeof(float3), cudaMemcpyHostToDevice);
        auto eventMs = [&](cudaEvent_t a, cudaEvent_t b) {
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return double(ms);
        };
        gpu::CudaRbfBinding binding;
        std::vector<double> tBind, tSolve, tEval;
        for (int i = 0; i < 9; ++i) {
            cudaEventRecord(start, stream);
            gpu::RbfStatus st = binding.Bind({dRest, size_t(n)}, 0.0, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            if (st != gpu::RbfStatus::Ok) {
                std::printf("CUDA bind failed: %s\n", binding.diagnostic());
                return 1;
            }
            tBind.push_back(eventMs(start, stop));
            cudaEventRecord(start, stream);
            st = binding.Solve({dPosed, size_t(n)}, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            if (st != gpu::RbfStatus::Ok) {
                std::printf("CUDA solve failed: %s\n", binding.diagnostic());
                return 1;
            }
            tSolve.push_back(eventMs(start, stop));
            cudaEventRecord(start, stream);
            st = binding.Evaluate({dCv, size_t(cvs)}, {dOut, size_t(cvs)}, stream);
            if (st == gpu::RbfStatus::Ok) st = binding.Finish(stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            if (st != gpu::RbfStatus::Ok) {
                std::printf("CUDA evaluate failed: %s\n", binding.diagnostic());
                return 1;
            }
            tEval.push_back(eventMs(start, stop));
        }
        EmitMetric("cuda_rbf_bind", Median(tBind));
        EmitMetric("cuda_rbf_solve", Median(tSolve));
        EmitMetric("cuda_rbf_evaluate_1m", Median(tEval));
        std::vector<float3> out;
        out.resize(size_t(cvs));
        cudaMemcpy(out.data(), dOut, size_t(cvs) * sizeof(float3), cudaMemcpyDeviceToHost);
        uint64_t h = Fnv1a(out.data(), size_t(cvs) * sizeof(float3));
        std::printf("CHECKSUM cuda_rbf_eval=%016llx\n", (unsigned long long)h);
        (void)m;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        cudaFree(dRest);
        cudaFree(dPosed);
        cudaFree(dCv);
        cudaFree(dOut);
        return 0;
    }
    }();
    if (cudaLegExit != 0 && cudaLegExit != 77) return cudaLegExit;
#endif

#ifdef USDGEN_BENCH_VULKAN
    // ---- Vulkan legs (need --vulkan-spv <dir>) ------------------------------
    int vulkanLegExit = 0;
    if (!vulkanSpvDir.empty()) {
        vulkanLegExit = VulkanDeformLeg(vulkanSpvDir, vulkanBudgetMb << 20);
        if (vulkanLegExit == 0) vulkanLegExit = VulkanBindLeg(vulkanSpvDir);
        if (vulkanLegExit != 0 && vulkanLegExit != 77) return vulkanLegExit;
    } else {
        std::printf("Vulkan legs skipped (no --vulkan-spv dir)\n");
    }
#endif

    if (g_hardFail) return 1;
#ifdef USDGEN_BENCH_CUDA
    if (cudaLegExit == 77) {
        std::printf("rbf-deform bench: partial skip (CUDA leg unavailable)\n");
        return 77;
    }
#endif
#ifdef USDGEN_BENCH_VULKAN
    if (!vulkanSpvDir.empty() && vulkanLegExit == 77) {
        std::printf("rbf-deform bench: partial skip (Vulkan legs unavailable)\n");
        return 77;
    }
#endif
    std::printf("rbf-deform bench: done\n");
    return 0;
}
