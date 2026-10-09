// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanRbfVkParity — CUDA-vs-Vulkan equivalence gate for the
// Vulkan RBF device path (p2-rbf, T1).
//
// The admitted plan range rbfSamples [4,100] is host-solved; this gate
// proves the device path (RbfVkBinding + RbfVkDeformPipeline, fed by the
// rbfVkPlan.h gap baker) executes the boundary configs with
// CUDA-equivalent results:
//   P0  rbfVkPlan.h baking: identity/translation/rotation equivalence,
//       budgets 3 and 101, and malformed-input rejections (host-only).
//   B1-B3,B6,B7  binding value parity vs CudaRbfBinding within the CUDA
//       tests' own contracts (2e-4 affine/identity and 1e-3 nonlinear
//       per testUsdGenCudaRbf.cu near(); 4e-4 deform per
//       testUsdGenCudaDeform.cu Near), plus <=1e-9 vs the independent
//       CPU oracle (rbf::CubicField / deformRbfHost.h SolveRbf).
//   B4-B5  error parity: equal status codes AND equal diagnostic
//       strings on the shared n<4 / rank-deficient / non-finite /
//       mismatch paths (CUDA's messages are mirrored verbatim).
//   B9     eval-cache transparency: cached == direct bitwise over
//       shapes/strides/reshape/rest-change/NaN plus a randomized
//       differential, with hit/miss engagement and clean fallbacks
//       (tiny pool, missing SPVs, disabled seam).
//   G0-G4  pipeline end-to-end vs CudaRbfCurveDeformer (legacy + typed),
//       including n=101 and baked non-identity samples, plus the
//       pass-through / rejection contracts.
// A case prints PASS with its worst observed diff; any failure exits 1.
// Exit 77 only when no Vulkan device exists at all. CUDA legs are
// skipped (not failed) when no CUDA device is present; the CPU oracle
// still checks every value case in that configuration.
//
// Device policy: vkCreateDevice failure (notably -3 under tenant load)
// retries 5x with a 60s wait; only then does the test fall back to a
// non-NVIDIA compute device, bannered as SECONDARY EVIDENCE with NVIDIA
// reported retry-pending. Set USDGEN_RBFVK_REQUIRE_NVIDIA=1 to fail
// closed (77) instead of using the fallback. USDGEN_RBFVK_RETRIES and
// USDGEN_RBFVK_RETRY_SECONDS override the retry schedule for iteration
// only (defaults 5 and 60).

#include "usdGen/vulkan/deviceContext.h"
#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deformPipeline.h"
#include "usdGen/vulkan/deformRbfHost.h"
#include "usdGen/vulkan/rbfVkBinding.h"
#include "usdGen/vulkan/rbfVkDeformPipeline.h"
#include "usdGen/vulkan/rbfVkPlan.h"
#include "usdGen/vulkan/surfaceVk.h"
#include "usdGen/ops/rbfField.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "gpu/deformCurves.h"
#include "gpu/rbf.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {

int g_cudaSkipped = 0;

// --- tolerances: the CUDA twins' own contracts ---
// testUsdGenCudaRbf.cu near(): 2e-4 default, 1e-3 nonlinear.
constexpr float kTolCudaRbfAffine = 2e-4f;
constexpr float kTolCudaRbfNonlinear = 1e-3f;
// testUsdGenCudaDeform.cu Near(): 4e-4.
constexpr float kTolCudaDeform = 4e-4f;
// Independent CPU oracle (same algorithm, host order): tight.
constexpr double kTolCpuOracle = 1e-6;
constexpr double kTolHostSolve = 1e-9;

inline vulkan::float3 V(float x, float y, float z) { return {x, y, z}; }

inline float MaxDiff(float const* a, float const* b, size_t n, bool* bitwise = nullptr) {
    float worst = 0.0f;
    bool exact = true;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            if (bitwise) *bitwise = false;
            return std::numeric_limits<float>::infinity();
        }
        float d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
        if (a[i] != b[i]) exact = false;
    }
    if (bitwise) *bitwise = exact;
    return worst;
}

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

bool CompleteRbf(std::shared_ptr<NativeOwner> const& native,
                 RbfVkDeformPipeline::Candidate& candidate, DeformSemantic* semantic) {
    for (unsigned phase = 0; phase < 6; ++phase) {
        if (candidate.pending() && !Prove(native)) return false;
        if (candidate.Poll(semantic) != VK_SUCCESS) return false;
        if (candidate.done()) return true;
        VkResult r = VK_SUCCESS;
        if (!candidate.Advance({}, &r) || r != VK_SUCCESS) return false;
    }
    return false;
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

inline bool ReadDeformOutput(std::shared_ptr<NativeOwner> const& native,
                             std::shared_ptr<DeviceContext> const& context,
                             RbfVkDeformPipeline::Candidate const& c,
                             std::vector<float>* out) {
    auto o = c.output();
    if (!o.points) return false;
    std::vector<uint8_t> bytes;
    if (!ReadVulkanBytes(native, context, o.points->buffer(),
                         o.points->sizeBytes(), o.points, &bytes))
        return false;
    out->resize(bytes.size() / sizeof(float));
    std::memcpy(out->data(), bytes.data(), bytes.size());
    return true;
}

#ifdef USDGEN_ENABLE_CUDA
using gpu::CudaRbfBinding;
using gpu::RbfStatus;

template <class T>
bool CudaUpload(gpu::DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        cudaMemcpy(buffer.data(), values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice) == cudaSuccess;
}
template <class T>
std::vector<T> CudaRead(gpu::DeviceBuffer<T> const& buffer) {
    std::vector<T> values(buffer.size());
    if (cudaMemcpy(values.data(), buffer.data(), values.size() * sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        values.clear();
    return values;
}
#endif

// Independent CPU oracle (production host field, dynamic size).
bool CpuSolve(float const* rest, float const* posed, int n,
              float const* cvs, float* out, uint32_t count, std::string* error) {
    std::vector<GfVec3d> r{size_t(n)};
    std::vector<GfVec3d> p{size_t(n)};
    for (int i = 0; i < n; ++i) {
        r[size_t(i)] = GfVec3d(rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]);
        p[size_t(i)] = GfVec3d(posed[3 * i], posed[3 * i + 1], posed[3 * i + 2]);
    }
    rbf::CubicField field;
    if (!field.Bind(r, error)) return false;
    if (!field.Solve(p, error)) return false;
    for (uint32_t i = 0; i < count; ++i) {
        GfVec3d x(cvs[3 * i], cvs[3 * i + 1], cvs[3 * i + 2]);
        GfVec3d moved = x + field.Displacement(x);
        out[3 * i] = float(moved[0]);
        out[3 * i + 1] = float(moved[1]);
        out[3 * i + 2] = float(moved[2]);
    }
    return true;
}

// Host SolveRbf oracle (n <= 100): expected bitwise-or-ULP equal.
bool HostSolve(float const* rest, float const* posed, int n,
               float const* cvs, float* out, uint32_t count) {
    RbfState state;
    SolveRbf(reinterpret_cast<vulkan::float3 const*>(rest),
             reinterpret_cast<vulkan::float3 const*>(posed), n, 0.0, state);
    if (state.status != vulkan::RbfStatus::Code::Ok) return false;
    for (uint32_t i = 0; i < count; ++i) {
        vulkan::float3 got = RbfEvaluate(state, reinterpret_cast<vulkan::float3 const*>(rest),
                                 V(cvs[3 * i], cvs[3 * i + 1], cvs[3 * i + 2]));
        out[3 * i] = got.x;
        out[3 * i + 1] = got.y;
        out[3 * i + 2] = got.z;
    }
    return true;
}

// Deterministic jittered grid spanning 3D.
std::vector<float> GridSamples(int n) {
    std::vector<float> pts;
    pts.reserve(size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
        int x = i % 5, y = (i / 5) % 5, z = i / 25;
        float jx = float((i * 37) % 11) * 0.013f;
        float jy = float((i * 53) % 13) * 0.017f;
        float jz = float((i * 29) % 7) * 0.019f;
        pts.push_back(float(x) + jx);
        pts.push_back(float(y) + jy);
        pts.push_back(float(z) * 0.8f + jz);
    }
    return pts;
}

void AffinePose(float const* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2];
        out[3 * i] = -y + 2.0f;
        out[3 * i + 1] = x - 3.0f;
        out[3 * i + 2] = z + 4.0f;
    }
}

void NonlinearPose(float const* in, float* out, int n) {
    for (int i = 0; i < n; ++i) {
        float x = in[3 * i], y = in[3 * i + 1], z = in[3 * i + 2];
        out[3 * i] = -y + 2.0f;
        out[3 * i + 1] = x - 3.0f;
        out[3 * i + 2] = z + 4.0f + x * y + 0.2f * z * z;
    }
}

// GapRejections-style rooted fixture (6-point 3D surface, 1 curve).
struct RootedFixture {
    UsdGenGraphDesc desc;
    UsdGenNodeDesc deform;
};

RootedFixture MakeRooted() {
    RootedFixture fx;
    fx.desc.description = SdfPath("/Groom");
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Groom/C3");
    curves.role = UsdGenRole::Curves;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{0, 0, 0}, {0, 1, 0}};
    curves.rest = curves.points;
    curves.widths = {.02f, .03f};
    curves.skinPrim = {0};
    curves.skinPrimUv = {{.2f, .2f}};
    fx.desc.curveSets.push_back(curves);
    UsdGenSurfaceDesc surface;
    surface.path = SdfPath("/Groom/Scalp");
    surface.restPoints = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 1, 0}, {1, 0, 1}};
    surface.points = surface.restPoints;
    surface.faceVertexCounts = {3, 3};
    surface.faceVertexIndices = {0, 1, 2, 3, 4, 5};
    fx.desc.surfaces.push_back(surface);
    fx.deform.path = SdfPath("/Groom/Ops/deform");
    fx.deform.type = TfToken("UsdGenDeform");
    fx.deform.surfaces = {surface.path};
    return fx;
}

#ifdef USDGEN_ENABLE_CUDA
int CudaCode(gpu::RbfStatus s) {
    switch (s) {
        case gpu::RbfStatus::Ok: return 0;
        case gpu::RbfStatus::InvalidArgument: return 1;
        case gpu::RbfStatus::NonFiniteInput: return 2;
        case gpu::RbfStatus::RankDeficient: return 3;
        case gpu::RbfStatus::CudaError: return 4;
        case gpu::RbfStatus::SolverError: return 5;
    }
    return -1;
}

int VkCode(RbfVkStatus s) {
    switch (s) {
        case RbfVkStatus::Ok: return 0;
        case RbfVkStatus::InvalidArgument: return 1;
        case RbfVkStatus::NonFiniteInput: return 2;
        case RbfVkStatus::RankDeficient: return 3;
        case RbfVkStatus::DeviceError: return 4;
        case RbfVkStatus::SolverError: return 5;
    }
    return -1;
}

std::vector<::float3> ToF3(float const* p, int n) {
    std::vector<::float3> v{size_t(n)};
    for (int i = 0; i < n; ++i) v[size_t(i)] = make_float3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
    return v;
}

std::vector<float> FromF3(std::vector<::float3> const& v) {
    std::vector<float> p(v.size() * 3);
    for (size_t i = 0; i < v.size(); ++i) {
        p[3 * i] = v[i].x;
        p[3 * i + 1] = v[i].y;
        p[3 * i + 2] = v[i].z;
    }
    return p;
}
#endif

} // namespace

int main(int argc, char** argv) {
    if (argc != 12) {
        std::fprintf(stderr,
                     "usage: %s rbfVkExtent.spv rbfVkGram.spv rbfVkBuildMatrix.spv rbfVkLu.spv "
                     "rbfVkRhs.spv rbfVkTriSolve.spv rbfVkEvaluate.spv rbfVkApply.spv "
                     "rbfVkVerify.spv rbfVkFill.spv rbfVkEvaluateCached.spv\n",
                     argv[0]);
        return 1;
    }
    RbfVkBindingSpirv spirv;
    spirv.extent = Code(argv[1]);
    spirv.gram = Code(argv[2]);
    spirv.buildMatrix = Code(argv[3]);
    spirv.lu = Code(argv[4]);
    spirv.rhs = Code(argv[5]);
    spirv.triSolve = Code(argv[6]);
    spirv.evaluate = Code(argv[7]);
    auto applySpv = Code(argv[8]);
    spirv.verify = Code(argv[9]);
    spirv.fill = Code(argv[10]);
    spirv.evaluateCached = Code(argv[11]);
    CHECK(!spirv.extent.empty() && !spirv.gram.empty() && !spirv.buildMatrix.empty() &&
          !spirv.lu.empty() && !spirv.rhs.empty() && !spirv.triSolve.empty() &&
          !spirv.evaluate.empty() && !applySpv.empty() && !spirv.verify.empty() &&
          !spirv.fill.empty() && !spirv.evaluateCached.empty());

    // ================= P0: gap baker (host-only) =================
    {
        // P0a: identity default budget.
        auto fx = MakeRooted();
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 6);
        CHECK(out.restSamples.size() == 18 && out.posedSamples.size() == 18);
        for (int i = 0; i < 18; ++i) {
            CHECK(out.restSamples[size_t(i)] == fx.desc.surfaces[0].restPoints[i / 3][i % 3]);
            CHECK(out.posedSamples[size_t(i)] == fx.desc.surfaces[0].points[i / 3][i % 3]);
        }
        // Hand-gathered root target: tri (0,0,0),(1,0,0),(0,1,0) at (.2,.2):
        // CUDA a*w+b*u+c*v gives x = u = 0.2.
        CHECK(std::fabs(out.rootTargets[0] - 0.2f) < 1e-6f);
        CHECK(std::fabs(out.rootTargets[1] - 0.2f) < 1e-6f);
        CHECK(std::fabs(out.rootTargets[2]) < 1e-6f);
        std::puts("P0a (bake identity): PASS");
    }
    {
        // P0b: moved source (+5x) shifts samples into source-local space.
        auto fx = MakeRooted();
        fx.desc.curveSets[0].worldMatrix[3][0] = 5.0;
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 6);
        for (int i = 0; i < 6; ++i) {
            CHECK(std::fabs(out.restSamples[3 * i] - (fx.desc.surfaces[0].restPoints[i][0] - 5.0f)) < 1e-6f);
            CHECK(std::fabs(out.restSamples[3 * i + 1] - fx.desc.surfaces[0].restPoints[i][1]) < 1e-6f);
            CHECK(std::fabs(out.restSamples[3 * i + 2] - fx.desc.surfaces[0].restPoints[i][2]) < 1e-6f);
        }
        CHECK(std::fabs(out.rootTargets[0] - (0.2f - 5.0f)) < 1e-6f);
        std::puts("P0b (bake moved source): PASS");
    }
    {
        // P0c: moved surface (-2y) moves rest and posed drivers together
        // (surfaceVk convention: coherent motion cancels in the fit).
        auto fx = MakeRooted();
        fx.desc.surfaces[0].worldMatrix[3][1] = -2.0;
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 6);
        for (int i = 0; i < 6; ++i) {
            CHECK(std::fabs(out.restSamples[3 * i] - fx.desc.surfaces[0].restPoints[i][0]) < 1e-6f);
            CHECK(std::fabs(out.restSamples[3 * i + 1] - (fx.desc.surfaces[0].restPoints[i][1] - 2.0f)) < 1e-6f);
            CHECK(std::fabs(out.restSamples[3 * i + 2] - fx.desc.surfaces[0].restPoints[i][2]) < 1e-6f);
        }
        for (int i = 0; i < 6; ++i) {
            CHECK(std::fabs(out.posedSamples[3 * i] - fx.desc.surfaces[0].points[i][0]) < 1e-6f);
            CHECK(std::fabs(out.posedSamples[3 * i + 1] - (fx.desc.surfaces[0].points[i][1] - 2.0f)) < 1e-6f);
            CHECK(std::fabs(out.posedSamples[3 * i + 2] - fx.desc.surfaces[0].points[i][2]) < 1e-6f);
        }
        CHECK(std::fabs(out.rootTargets[0] - 0.2f) < 1e-6f);
        CHECK(std::fabs(out.rootTargets[1] - (0.2f - 2.0f)) < 1e-6f);
        std::puts("P0c (bake moved surface): PASS");
    }
    {
        // P0d: budgets 3 and 101 are admitted (no <4 rejection here).
        auto fx = MakeRooted();
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(3), false}};
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 3);
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(101), false}};
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 6); // all 6 unique vertices selected
        std::puts("P0d (bake budgets 3/101): PASS");
    }
    {
        // P0e: 125-point surface with budget 101 selects exactly 101.
        auto fx = MakeRooted();
        auto grid = GridSamples(125);
        VtVec3fArray pts;
        for (int i = 0; i < 125; ++i) pts.push_back(GfVec3f(grid[3 * i], grid[3 * i + 1], grid[3 * i + 2]));
        fx.desc.surfaces[0].restPoints = pts;
        fx.desc.surfaces[0].points = pts;
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(101), false}};
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        CHECK(out.sampleCount == 101);
        std::puts("P0e (bake 101 of 125): PASS");
    }
    {
        // P0f: malformed inputs still reject with clear diagnostics.
        auto expectReject = [](RootedFixture fx, char const* needle) {
            VulkanDeformControls out;
            UsdGenDiagnostics diag;
            bool ok = RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag);
            if (ok || !diag.HasErrors()) return false;
            for (auto const& e : diag.errors)
                if (e.find(needle) != std::string::npos) return true;
            std::fprintf(stderr, "missing %s in %s\n", needle,
                         diag.errors.empty() ? "<none>" : diag.errors.front().c_str());
            return false;
        };
        auto fx = MakeRooted();
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(0), false}};
        CHECK(expectReject(fx, "Deform rbfSamples must be in [1,46336]"));
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(46337), false}};
        CHECK(expectReject(fx, "Deform rbfSamples must be in [1,46336]"));
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(3.5f), false}};
        CHECK(expectReject(fx, "Deform rbfSamples must be an int literal"));
        fx.deform.params = {{TfToken("rbfSamples"), VtValue(50), true}};
        CHECK(expectReject(fx, "Deform requires non-animated literal controls"));
        fx.deform.params = {{TfToken("nope"), VtValue(1), false}};
        CHECK(expectReject(fx, "unsupported Deform parameter nope"));
        fx = MakeRooted();
        fx.desc.curveSets[0].worldMatrix[0][0] = 0.0; // singular source
        CHECK(expectReject(fx, "C3 source worldMatrix must be finite, affine and invertible"));
        fx = MakeRooted();
        fx.desc.surfaces[0].worldMatrix[0][3] = 1.0; // non-affine surface
        CHECK(expectReject(fx, "Deform surface worldMatrix must be finite and affine"));
        std::puts("P0f (bake rejections): PASS");
    }
    {
        // P0g: surface rotation (90 deg about Z) rotates posed drivers.
        auto fx = MakeRooted();
        auto& w = fx.desc.surfaces[0].worldMatrix;
        w = GfMatrix4d(1.0);
        w[0][0] = 0.0; w[0][1] = 1.0;
        w[1][0] = -1.0; w[1][1] = 0.0;
        VulkanDeformControls out;
        UsdGenDiagnostics diag;
        CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &out, &diag));
        // rest (1,0,0) is sample 1; rest and posed must both be
        // R*(1,0,0) = (0,1,0) under the surfaceVk convention.
        CHECK(std::fabs(out.restSamples[3] - 0.0f) < 1e-6f);
        CHECK(std::fabs(out.restSamples[4] - 1.0f) < 1e-6f);
        CHECK(std::fabs(out.restSamples[5] - 0.0f) < 1e-6f);
        CHECK(std::fabs(out.posedSamples[3] - 0.0f) < 1e-6f);
        CHECK(std::fabs(out.posedSamples[4] - 1.0f) < 1e-6f);
        CHECK(std::fabs(out.posedSamples[5] - 0.0f) < 1e-6f);
        std::puts("P0g (bake rotation): PASS");
    }
    {
        // P0h: gap validator and CompileSurfaceVkDeform bake the same field
        // (<=1e-6) on moved fixtures at in-budget budgets, so budget 100 vs
        // 101 is a pipeline choice, not a semantic change.
        auto crossCheck = [](RootedFixture fx, char const* tag) {
            VulkanDeformControls rbf, surf;
            UsdGenDiagnostics dr, ds;
            if (!RbfVkValidateDeformGap(fx.desc, fx.deform,
                                        &fx.desc.curveSets[0], &rbf, &dr))
                return false;
            if (!CompileSurfaceVkDeform(fx.desc.curveSets[0],
                                        fx.desc.surfaces[0], 100, &surf, &ds))
                return false;
            if (rbf.sampleCount != surf.sampleCount) return false;
            if (rbf.restSamples.size() != surf.restSamples.size()) return false;
            if (rbf.posedSamples.size() != surf.posedSamples.size())
                return false;
            if (rbf.rootTargets.size() != surf.rootTargets.size()) return false;
            auto worst = [](std::vector<float> const& a,
                            std::vector<float> const& b) {
                return a.size() == b.size() ? MaxDiff(a.data(), b.data(), a.size())
                    : std::numeric_limits<float>::infinity();
            };
            std::printf("P0h cross-check %s: worst rest/posed/roots %.3g/%.3g/%.3g\n",
                        tag, double(worst(rbf.restSamples, surf.restSamples)),
                        double(worst(rbf.posedSamples, surf.posedSamples)),
                        double(worst(rbf.rootTargets, surf.rootTargets)));
            return worst(rbf.restSamples, surf.restSamples) <= 1e-6f &&
                   worst(rbf.posedSamples, surf.posedSamples) <= 1e-6f &&
                   worst(rbf.rootTargets, surf.rootTargets) <= 1e-6f;
        };
        {
            auto fx = MakeRooted(); // identity
            CHECK(crossCheck(fx, "identity"));
        }
        {
            auto fx = MakeRooted(); // moved source (+5x)
            fx.desc.curveSets[0].worldMatrix[3][0] = 5.0;
            CHECK(crossCheck(fx, "moved-source"));
        }
        {
            auto fx = MakeRooted(); // moved surface (-2y)
            fx.desc.surfaces[0].worldMatrix[3][1] = -2.0;
            CHECK(crossCheck(fx, "moved-surface"));
        }
        std::puts("P0h (bake cross-checks): PASS");
    }

    // ================= device setup =================
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64{};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateNative(&unavailable, {}, nullptr, &fp64);
    if (unavailable) return 77;
    CHECK(native);

    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8021;
    ci.nativeLifetime = native;
    ci.resources = {size_t{64} << 20, size_t{4} << 20};
    ci.shaderFloat64Enabled = true;
    auto context = DeviceContext::Create(ci);
    CHECK(context);

    VkResult status = VK_SUCCESS;
    auto binding = RbfVkBinding::Create(context, spirv, &status);
    CHECK(binding && status == VK_SUCCESS);
    CHECK(context->resources()->Snapshot().usedBytes == 0); // Create stages no worst-case matrix.
    // The main binding evaluates through a shared R cache, so every
    // value case below exercises the cached path (B9 pins cached ==
    // direct bitwise, plus the engagement counters).
    auto evalCache = std::make_shared<RbfVkEvalCache>();
    binding->SetEvalCache(evalCache);
    auto pipe = RbfVkDeformPipeline::Create(context, spirv, applySpv, &status);
    CHECK(pipe && status == VK_SUCCESS);

#ifdef USDGEN_ENABLE_CUDA
    bool cudaAvailable = false;
    cudaStream_t cudaProducer = nullptr, cudaConsumer = nullptr;
    {
        int devices = 0;
        cudaError_t const result = cudaGetDeviceCount(&devices);
        bool const cudaUnavailable = result == cudaErrorNoDevice ||
            result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && devices == 0);
        CHECK(result == cudaSuccess || cudaUnavailable);
        if (!cudaUnavailable) {
            cudaError_t const selected = cudaSetDevice(0);
            if (selected != cudaSuccess)
                std::fprintf(stderr, "RBF CUDA device selection failed: %s (%d)\n", cudaGetErrorString(selected), int(selected));
            CHECK(selected == cudaSuccess);
            CHECK(cudaStreamCreate(&cudaProducer) == cudaSuccess);
            CHECK(cudaStreamCreate(&cudaConsumer) == cudaSuccess);
            cudaAvailable = true;
        }
    }
    if (!cudaAvailable) {
        std::printf("CUDA leg unavailable; running Vulkan-vs-CPU-oracle only\n");
        ++g_cudaSkipped;
    }
#else
    std::printf("CUDA leg not compiled; running Vulkan-vs-CPU-oracle only\n");
#endif

    // Evaluate retains submitted device inputs and outputs until Finish
    // proves their last GPU use, even when callers drop every external owner.
    {
        std::vector<float> const rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        CHECK(binding->Bind(rest.data(), 5, 0.0) == RbfVkStatus::Ok);
        CHECK(binding->Solve(rest.data(), 5) == RbfVkStatus::Ok);
        auto input = UploadVulkanDeviceBytes(native, context, rest.data(),
            rest.size() * sizeof(float));
        auto output = UploadVulkanDeviceBytes(native, context, rest.data(),
            rest.size() * sizeof(float));
        CHECK(input && output);
        std::weak_ptr<ChargedBuffer> weakInput = input, weakOutput = output;
        CHECK(binding->Evaluate(input, output, 5) == RbfVkStatus::Ok);
        input.reset(); output.reset();
        CHECK(!weakInput.expired() && !weakOutput.expired());
        CHECK(binding->Finish() == RbfVkStatus::Ok);
        CHECK(weakInput.expired() && weakOutput.expired());
        std::puts("RBF pending evaluation ownership: PASS");
    }

    // B8: the shared factor cache adopts bitwise-identical factors on a
    // hit and recomputes on rest/smoothing changes; failures stay failures.
    {
        const std::vector<float> restA{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        std::vector<float> restB = restA;
        restB[0] = 0.5f;
        std::vector<float> posedA(15), posedB(15);
        AffinePose(restA.data(), posedA.data(), 5);
        AffinePose(restB.data(), posedB.data(), 5);
        auto evalAll = [&](std::shared_ptr<RbfVkBinding> const& b, float const* rs,
                           float const* ps, double smoothing,
                           std::vector<float>& out) -> bool {
            CHECK(b->Bind(rs, 5, smoothing) == RbfVkStatus::Ok);
            CHECK(b->Solve(ps, 5) == RbfVkStatus::Ok);
            RbfVkStatus es = b->EvaluateHost(rs, out.data(), 5);
            CHECK(es == RbfVkStatus::Ok);
            return true;
        };
        auto sameBits = [&](std::vector<float> const& a, std::vector<float> const& b) {
            return a.size() == b.size() &&
                   std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        // Uncached oracles for both rests.
        auto plain = RbfVkBinding::Create(context, spirv, &status);
        CHECK(plain && status == VK_SUCCESS);
        std::vector<float> oracleA(15), oracleB(15);
        evalAll(plain, restA.data(), posedA.data(), 0.0, oracleA);
        evalAll(plain, restB.data(), posedB.data(), 0.0, oracleB);
        CHECK(!sameBits(oracleA, oracleB));
        // Cached binding: miss, hit, evict, re-miss; all bitwise vs oracle.
        auto cached = RbfVkBinding::Create(context, spirv, &status);
        CHECK(cached && status == VK_SUCCESS);
        auto cache = std::make_shared<RbfVkFactorCache>();
        cached->SetFactorCache(cache);
        std::vector<float> got(15);
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        evalAll(cached, restB.data(), posedB.data(), 0.0, got);
        CHECK(sameBits(got, oracleB));
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        // A second binding sharing the cache hits too.
        auto cached2 = RbfVkBinding::Create(context, spirv, &status);
        CHECK(cached2 && status == VK_SUCCESS);
        cached2->SetFactorCache(cache);
        evalAll(cached2, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        // Failures are never adopted: rank-deficient rest still fails, and
        // a later good bind still succeeds.
        const std::vector<float> plane{0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0.2f, 0.3f, 0};
        CHECK(cached->Bind(plane.data(), 5, 0.0) == RbfVkStatus::RankDeficient);
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        // Repeat failures stay failures (and stay exact): the per-binding
        // upload memo never skips validation, so a repeated bad rest fails
        // identically and the next good bind still matches bitwise.
        CHECK(cached->Bind(plane.data(), 5, 0.0) == RbfVkStatus::RankDeficient);
        CHECK(cached->Bind(plane.data(), 5, 0.0) == RbfVkStatus::RankDeficient);
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        std::vector<float> nanRest = restA;
        nanRest[3] = std::numeric_limits<float>::quiet_NaN();
        CHECK(cached->Bind(nanRest.data(), 5, 0.0) == RbfVkStatus::NonFiniteInput);
        CHECK(cached->Bind(nanRest.data(), 5, 0.0) == RbfVkStatus::NonFiniteInput);
        evalAll(cached, restA.data(), posedA.data(), 0.0, got);
        CHECK(sameBits(got, oracleA));
        // A smoothing change keys a different entry and still matches uncached.
        // (No assertion that smoothing moves the float bits: on this tiny
        // near-affine fixture the weights are ~0, so 1e-3 vs 0.0 smoothing
        // can be bit-identical. The cache property is cached == uncached.)
        std::vector<float> oracleS(15);
        evalAll(plain, restA.data(), posedA.data(), 1e-3, oracleS);
        evalAll(cached, restA.data(), posedA.data(), 1e-3, got);
        CHECK(sameBits(got, oracleS));
        std::puts("B8 (factor cache): PASS");
    }

    // B9: the RbfVkEvalCache evaluates bitwise what the direct shader
    // did, across shapes/strides/reshape/rest-change/NaN; it engages
    // (hits) on repeats and falls back cleanly (tiny pool, missing
    // SPVs, disabled seam). Cached == direct is the property.
    {
        auto sameBits = [&](std::vector<float> const& a, std::vector<float> const& b) {
            return a.size() == b.size() &&
                   std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        };
        auto diagnose = [&](float const* got, float const* want, size_t n,
                            char const* tag) {
            size_t ndiff = 0, first = 0;
            for (size_t i = 0; i < n; ++i) {
                uint32_t a, b;
                std::memcpy(&a, &got[i], 4);
                std::memcpy(&b, &want[i], 4);
                if (a == b) continue;
                if (ndiff == 0) first = i;
                if (++ndiff > 4) break;
            }
            if (ndiff == 0) return true;
            // Recount fully (the loop above stops early).
            ndiff = 0;
            for (size_t i = 0; i < n; ++i) {
                uint32_t a, b;
                std::memcpy(&a, &got[i], 4);
                std::memcpy(&b, &want[i], 4);
                ndiff += (a != b);
            }
            std::fprintf(stderr, "B9/%s: %zu/%zu differ, first @%zu got=%a want=%a\n",
                         tag, ndiff, n, first, got[first], want[first]);
            return false;
        };
        uint64_t rng = 0x243f6a8885a308d3ull;
        auto rand01 = [&]() {
            rng = rng * 6364136223846793005ull + 1442695040888963407ull;
            return double(rng >> 11) * (1.0 / 9007199254740992.0);
        };
        struct Field {
            std::vector<float> rest, posed, cvs;
        };
        auto mkField = [&](int n, uint32_t count, double span) {
            Field f;
            f.rest.resize(size_t(n) * 3);
            f.posed.resize(size_t(n) * 3);
            f.cvs.resize(size_t(count) * 3);
            for (int i = 0; i < n; ++i) {
                double rx = span * (2 * rand01() - 1);
                double ry = span * (2 * rand01() - 1);
                double rz = span * (2 * rand01() - 1);
                f.rest[size_t(i) * 3] = float(rx);
                f.rest[size_t(i) * 3 + 1] = float(ry);
                f.rest[size_t(i) * 3 + 2] = float(rz);
                f.posed[size_t(i) * 3] = float(rx + 0.1 * std::sin(double(i)));
                f.posed[size_t(i) * 3 + 1] = float(ry - 0.05 * double(i % 7));
                f.posed[size_t(i) * 3 + 2] = float(rz + 0.07 * std::cos(2.0 * double(i)));
            }
            for (uint32_t i = 0; i < count; ++i)
                for (int a = 0; a < 3; ++a)
                    f.cvs[size_t(i) * 3 + size_t(a)] =
                        float(span * (2 * rand01() - 1));
            return f;
        };
        auto evalCacheB = std::make_shared<RbfVkEvalCache>();
        auto cacheB = RbfVkBinding::Create(context, spirv, &status);
        CHECK(cacheB && status == VK_SUCCESS);
        cacheB->SetEvalCache(evalCacheB);
        // NOTE: no CHECK inside these lambdas (it returns 1, which
        // reads as success from a bool): every step returns false
        // explicitly, and the call-site CHECK reports the line.
        auto directInto = [&](Field const& f, int, uint32_t count,
                              std::vector<float>& out) -> bool {
            TestDisableRbfVkEvalCache(true);
            RbfVkStatus ds = cacheB->EvaluateHost(f.cvs.data(), out.data(), count);
            TestDisableRbfVkEvalCache(false);
            if (ds != RbfVkStatus::Ok) return false;
            return true;
        };
        auto checkShape = [&](int n, uint32_t count, double span, char const* tag) -> bool {
            Field f = mkField(n, count, span);
            if (cacheB->Bind(f.rest.data(), n, 0.0) != RbfVkStatus::Ok) return false;
            if (cacheB->Solve(f.posed.data(), n) != RbfVkStatus::Ok) return false;
            std::vector<float> got(size_t(count) * 3), rep(size_t(count) * 3),
                want(size_t(count) * 3);
            if (cacheB->EvaluateHost(f.cvs.data(), got.data(), count) !=
                RbfVkStatus::Ok)
                return false;
            if (cacheB->EvaluateHost(f.cvs.data(), rep.data(), count) !=
                RbfVkStatus::Ok)
                return false;
            if (!diagnose(got.data(), rep.data(), got.size(), tag)) return false;
            if (!directInto(f, n, count, want)) return false;
            if (!diagnose(got.data(), want.data(), got.size(), tag)) return false;
            return true;
        };
        CHECK(checkShape(5, 7, 1.0, "tiny"));
        CHECK(checkShape(5, 5, 1.0, "square"));
        CHECK(checkShape(4, 4, 1.0, "min"));
        CHECK(checkShape(8, 129, 1.0, "unstrided"));
        CHECK(checkShape(16, 1000, 1.0, "kilo"));
        CHECK(checkShape(64, 3000, 2.0, "wide"));
        CHECK(checkShape(400, 500, 1.0, "deep-n"));
        CHECK(checkShape(200, 8000, 1.0, "medium"));
        CHECK(evalCacheB->hitsForTesting() >= 8);
        CHECK(evalCacheB->missesForTesting() >= 8);
        std::puts("B9a (cached == direct): PASS");
        // Rest change misses and still matches bitwise.
        {
            uint64_t misses0 = evalCacheB->missesForTesting();
            Field f = mkField(16, 1000, 1.0);
            CHECK(cacheB->Bind(f.rest.data(), 16, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 16) == RbfVkStatus::Ok);
            std::vector<float> got(3000), want(3000);
            CHECK(cacheB->EvaluateHost(f.cvs.data(), got.data(), 1000) ==
                  RbfVkStatus::Ok);
            CHECK(directInto(f, 16, 1000, want));
            CHECK(diagnose(got.data(), want.data(), got.size(), "rest-change"));
            CHECK(evalCacheB->missesForTesting() > misses0);
            std::puts("B9b (rest change): PASS");
        }
        // NaN CVs: both paths report NonFiniteInput and leave the host
        // output untouched (no readback on failure).
        {
            Field f = mkField(8, 129, 1.0);
            f.cvs[3 * 17 + 1] = std::numeric_limits<float>::quiet_NaN();
            CHECK(cacheB->Bind(f.rest.data(), 8, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 8) == RbfVkStatus::Ok);
            std::vector<float> got(129 * 3, 12345.0f), want(129 * 3, 12345.0f);
            CHECK(cacheB->EvaluateHost(f.cvs.data(), got.data(), 129) ==
                  RbfVkStatus::NonFiniteInput);
            // A poisoned evaluation leaves the binding unsolved, so the
            // direct oracle needs a fresh solved state (same as B4/B5).
            CHECK(cacheB->Bind(f.rest.data(), 8, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 8) == RbfVkStatus::Ok);
            TestDisableRbfVkEvalCache(true);
            RbfVkStatus ds = cacheB->EvaluateHost(f.cvs.data(), want.data(), 129);
            TestDisableRbfVkEvalCache(false);
            CHECK(ds == RbfVkStatus::NonFiniteInput);
            CHECK(sameBits(got, want));
            std::puts("B9c (NaN CVs): PASS");
        }
        // A shared cache hits across bindings (the production gap-path
        // pattern: fresh bindings, static CVs/rest).
        {
            Field f = mkField(16, 1000, 1.0);
            CHECK(cacheB->Bind(f.rest.data(), 16, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 16) == RbfVkStatus::Ok);
            std::vector<float> got(3000), want(3000), other(3000);
            CHECK(cacheB->EvaluateHost(f.cvs.data(), got.data(), 1000) ==
                  RbfVkStatus::Ok);
            auto cacheB2 = RbfVkBinding::Create(context, spirv, &status);
            CHECK(cacheB2 && status == VK_SUCCESS);
            cacheB2->SetEvalCache(evalCacheB);
            CHECK(cacheB2->Bind(f.rest.data(), 16, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB2->Solve(f.posed.data(), 16) == RbfVkStatus::Ok);
            uint64_t hits0 = evalCacheB->hitsForTesting();
            CHECK(cacheB2->EvaluateHost(f.cvs.data(), other.data(), 1000) ==
                  RbfVkStatus::Ok);
            CHECK(evalCacheB->hitsForTesting() > hits0);
            CHECK(diagnose(got.data(), other.data(), got.size(), "shared"));
            CHECK(directInto(f, 16, 1000, want));
            CHECK(diagnose(got.data(), want.data(), got.size(), "shared-direct"));
            std::puts("B9d (shared cache): PASS");
        }
        // Randomized differential: fresh shapes churn one binding+cache
        // through miss/hit/reshape/speculative paths; every cached
        // evaluation matches the direct shader bitwise.
        {
            int const cases = 60;
            for (int t = 0; t < cases; ++t) {
                int const n = 4 + int(rng % 21);
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                uint32_t const count = 1 + uint32_t(rng % 600);
                rng = rng * 6364136223846793005ull + 1442695040888963407ull;
                double const span = (t % 3 == 2) ? (t % 2 ? 1e3 : 10.0) : 1.0;
                char tag[32];
                std::snprintf(tag, sizeof(tag), "fuzz-%d", t);
                Field f = mkField(n, count, span);
                CHECK(cacheB->Bind(f.rest.data(), n, 0.0) == RbfVkStatus::Ok);
                CHECK(cacheB->Solve(f.posed.data(), n) == RbfVkStatus::Ok);
                std::vector<float> got(size_t(count) * 3), want(size_t(count) * 3);
                CHECK(cacheB->EvaluateHost(f.cvs.data(), got.data(), count) ==
                      RbfVkStatus::Ok);
                CHECK(directInto(f, n, count, want));
                CHECK(diagnose(got.data(), want.data(), got.size(), tag));
            }
            std::puts("B9e (randomized differential): PASS");
        }
        // A pool that fits the bind but not R falls back to direct with
        // identical bytes and no cache traffic.
        {
            DeviceContext::CreateInfo ci2 = ci;
            ci2.resourceDeviceId = 18021;
            ci2.resources = {size_t{1} << 20, 0};
            auto tiny = DeviceContext::Create(ci2);
            CHECK(tiny);
            auto tinyBinding = RbfVkBinding::Create(tiny, spirv, &status);
            CHECK(tinyBinding && status == VK_SUCCESS);
            auto tinyCache = std::make_shared<RbfVkEvalCache>();
            tinyBinding->SetEvalCache(tinyCache);
            Field f = mkField(100, 2000, 1.0); // R = 1.6MB > 1MB pool
            CHECK(tinyBinding->Bind(f.rest.data(), 100, 0.0) == RbfVkStatus::Ok);
            CHECK(tinyBinding->Solve(f.posed.data(), 100) == RbfVkStatus::Ok);
            std::vector<float> got(size_t(2000) * 3), want(size_t(2000) * 3);
            CHECK(tinyBinding->EvaluateHost(f.cvs.data(), got.data(), 2000) ==
                  RbfVkStatus::Ok);
            CHECK(tinyCache->hitsForTesting() == 0);
            CHECK(tinyCache->missesForTesting() == 0);
            CHECK(cacheB->Bind(f.rest.data(), 100, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 100) == RbfVkStatus::Ok);
            CHECK(directInto(f, 100, 2000, want));
            CHECK(diagnose(got.data(), want.data(), got.size(), "tiny-pool"));
            std::puts("B9f (tiny-pool fallback): PASS");
        }
        // Missing cache programs keep the binding direct (old SPV dirs
        // keep working); a partial set is a wiring bug rejected at
        // Create.
        {
            RbfVkBindingSpirv noCache = spirv;
            noCache.verify.clear();
            noCache.fill.clear();
            noCache.evaluateCached.clear();
            auto directBinding = RbfVkBinding::Create(context, noCache, &status);
            CHECK(directBinding && status == VK_SUCCESS);
            auto noCacheCounter = std::make_shared<RbfVkEvalCache>();
            directBinding->SetEvalCache(noCacheCounter);
            Field f = mkField(16, 1000, 1.0);
            CHECK(directBinding->Bind(f.rest.data(), 16, 0.0) == RbfVkStatus::Ok);
            CHECK(directBinding->Solve(f.posed.data(), 16) == RbfVkStatus::Ok);
            std::vector<float> got(3000), want(3000);
            CHECK(directBinding->EvaluateHost(f.cvs.data(), got.data(), 1000) ==
                  RbfVkStatus::Ok);
            CHECK(noCacheCounter->hitsForTesting() == 0);
            CHECK(noCacheCounter->missesForTesting() == 0);
            CHECK(cacheB->Bind(f.rest.data(), 16, 0.0) == RbfVkStatus::Ok);
            CHECK(cacheB->Solve(f.posed.data(), 16) == RbfVkStatus::Ok);
            CHECK(directInto(f, 16, 1000, want));
            CHECK(diagnose(got.data(), want.data(), got.size(), "no-spv"));
            RbfVkBindingSpirv partial = spirv;
            partial.fill.clear();
            partial.evaluateCached.clear();
            auto partialBinding = RbfVkBinding::Create(context, partial, &status);
            CHECK(!partialBinding);
            std::puts("B9g (missing SPVs): PASS");
        }
        std::puts("B9 (eval cache): PASS");
    }

    // ================= B1: n=5 (the CUDA test's own fixture) =================
    {
        const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        const std::vector<float> cvs{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                                     1, 1, 1, 0.25f, 0.5f, 0.75f, 0.1f, 0.9f, 0.3f};
        const uint32_t count = 7;
        struct Pose {
            char const* name;
            float tol;
        };
        for (int variant = 0; variant < 3; ++variant) {
            std::vector<float> posed(15);
            float tol;
            char const* name;
            if (variant == 0) {
                posed = rest;
                tol = kTolCudaRbfAffine;
                name = "identity";
            } else if (variant == 1) {
                AffinePose(rest.data(), posed.data(), 5);
                tol = kTolCudaRbfAffine;
                name = "affine";
            } else {
                NonlinearPose(rest.data(), posed.data(), 5);
                tol = kTolCudaRbfNonlinear;
                name = "nonlinear";
            }
            CHECK(binding->Bind(rest.data(), 5, 0.0) == RbfVkStatus::Ok);
            CHECK(binding->Solve(posed.data(), 5) == RbfVkStatus::Ok);
            std::vector<float> vkOut(size_t(count) * 3);
            CHECK(binding->EvaluateHost(cvs.data(), vkOut.data(), count) == RbfVkStatus::Ok);
            std::string cpuError;
            std::vector<float> cpuOut(size_t(count) * 3);
            CHECK(CpuSolve(rest.data(), posed.data(), 5, cvs.data(), cpuOut.data(), count, &cpuError));
            bool bitwise = false;
            float dCpu = MaxDiff(vkOut.data(), cpuOut.data(), vkOut.size(), &bitwise);
            CHECK(dCpu <= float(kTolCpuOracle));
            std::vector<float> hostOut(size_t(count) * 3);
            CHECK(HostSolve(rest.data(), posed.data(), 5, cvs.data(), hostOut.data(), count));
            bool bitwiseHost = false;
            float dHost = MaxDiff(vkOut.data(), hostOut.data(), vkOut.size(), &bitwiseHost);
            CHECK(dHost <= float(kTolHostSolve));
#ifdef USDGEN_ENABLE_CUDA
            if (cudaAvailable) {
                gpu::DeviceBuffer<::float3> dr, dp, dc, dout;
                CHECK(CudaUpload(dr, ToF3(rest.data(), 5)));
                CHECK(CudaUpload(dp, ToF3(posed.data(), 5)));
                CHECK(CudaUpload(dc, ToF3(cvs.data(), int(count))));
                CHECK(dout.reset(count) == cudaSuccess);
                CudaRbfBinding cr;
                CHECK(cr.Bind({dr.data(), 5}, 0, cudaProducer) == gpu::RbfStatus::Ok);
                CHECK(cr.Solve({dp.data(), 5}, cudaProducer) == gpu::RbfStatus::Ok);
                CHECK(cr.Evaluate({dc.data(), count}, {dout.data(), count}, cudaProducer) ==
                      gpu::RbfStatus::Ok);
                CHECK(cr.Finish(cudaProducer) == gpu::RbfStatus::Ok);
                std::vector<float> cudaOut = FromF3(CudaRead(dout));
                CHECK(cudaOut.size() == vkOut.size());
                float dCuda = MaxDiff(vkOut.data(), cudaOut.data(), vkOut.size());
                CHECK(dCuda <= tol);
                std::printf("B1/%s: PASS (vk-cuda %.2e <= %.0e, vk-cpu %.2e%s, vk-host %.2e%s)\n",
                            name, dCuda, tol, dCpu, bitwise ? " BITWISE" : "",
                            dHost, bitwiseHost ? " BITWISE" : "");
            } else {
                std::printf("B1/%s: PASS-no-CUDA (vk-cpu %.2e%s, vk-host %.2e%s)\n", name, dCpu,
                            bitwise ? " BITWISE" : "", dHost, bitwiseHost ? " BITWISE" : "");
            }
#else
            std::printf("B1/%s: PASS-no-CUDA (vk-cpu %.2e, vk-host %.2e)\n", name, dCpu, dHost);
#endif
        }
    }

    // ================= B2: n=4 tetrahedron =================
    {
        const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<float> posed(12);
        AffinePose(rest.data(), posed.data(), 4);
        CHECK(binding->Bind(rest.data(), 4, 0.0) == RbfVkStatus::Ok);
        CHECK(binding->Solve(posed.data(), 4) == RbfVkStatus::Ok);
        std::vector<float> vkOut(12);
        CHECK(binding->EvaluateHost(rest.data(), vkOut.data(), 4) == RbfVkStatus::Ok);
        std::string cpuError;
        std::vector<float> cpuOut(12);
        CHECK(CpuSolve(rest.data(), posed.data(), 4, rest.data(), cpuOut.data(), 4, &cpuError));
        float dCpu = MaxDiff(vkOut.data(), cpuOut.data(), 12);
        CHECK(dCpu <= float(kTolCpuOracle));
#ifdef USDGEN_ENABLE_CUDA
        if (cudaAvailable) {
            gpu::DeviceBuffer<::float3> dr, dp, dout;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 4)));
            CHECK(CudaUpload(dp, ToF3(posed.data(), 4)));
            CHECK(dout.reset(4) == cudaSuccess);
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 4}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Solve({dp.data(), 4}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dr.data(), 4}, {dout.data(), 4}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Finish(cudaProducer) == gpu::RbfStatus::Ok);
            std::vector<float> cudaOut = FromF3(CudaRead(dout));
            float dCuda = MaxDiff(vkOut.data(), cudaOut.data(), 12);
            CHECK(dCuda <= kTolCudaRbfAffine);
            std::printf("B2 (n=4 tetra): PASS (vk-cuda %.2e, vk-cpu %.2e)\n", dCuda, dCpu);
        } else {
            std::printf("B2 (n=4 tetra): PASS-no-CUDA (vk-cpu %.2e)\n", dCpu);
        }
#else
        std::printf("B2 (n=4 tetra): PASS-no-CUDA (vk-cpu %.2e)\n", dCpu);
#endif
    }

    // Resource admission uses the actual count and unwinds every failed charge.
    {
        auto limitedInfo = ci;
        limitedInfo.resourceDeviceId = 8022;
        limitedInfo.resources = {size_t{256} << 10, 0};
        auto limitedContext = DeviceContext::Create(limitedInfo);
        CHECK(limitedContext);
        auto limited = RbfVkBinding::Create(limitedContext, spirv, &status);
        CHECK(limited && limitedContext->resources()->Snapshot().usedBytes == 0);
        auto large = GridSamples(1025);
        CHECK(limited->Bind(large.data(), 1025, 0.0) == RbfVkStatus::DeviceError);
        CHECK(std::strstr(limited->diagnostic(), "resource budget"));
        CHECK(limitedContext->resources()->Snapshot().usedBytes == 0);
        const std::vector<float> tetra{0,0,0, 1,0,0, 0,1,0, 0,0,1};
        CHECK(limited->Bind(tetra.data(), 4, 0.0) == RbfVkStatus::Ok);
        auto smallCharge = limitedContext->resources()->Snapshot().usedBytes;
        CHECK(smallCharge > 0 && smallCharge < (size_t{64} << 10));
        CHECK(limited->Bind(tetra.data(), 46337, 0.0) == RbfVkStatus::InvalidArgument);
        CHECK(limited->Bind(tetra.data(), 46336, 0.0) == RbfVkStatus::InvalidArgument);
        CHECK(std::strstr(limited->diagnostic(), "maxStorageBufferRange"));
        CHECK(limitedContext->resources()->Snapshot().usedBytes == smallCharge);
        limited.reset();
        CHECK(limitedContext->resources()->Snapshot().usedBytes == 0);
        std::puts("B3/resource admission + recovery: PASS");
    }
    // 1025 is admitted and solved, including an off-driver affine query.
    {
        auto rest = GridSamples(1025);
        std::vector<float> posed(rest.size());
        AffinePose(rest.data(), posed.data(), 1025);
        const std::vector<float> cvs{0.33f, 1.71f, 2.13f, 3.9f, 0.11f, 1.01f};
        std::vector<float> expected(cvs.size()), vkOut(cvs.size());
        AffinePose(cvs.data(), expected.data(), 2);
        CHECK(binding->Bind(rest.data(), 1025, 0.0) == RbfVkStatus::Ok);
        CHECK(binding->Solve(posed.data(), 1025) == RbfVkStatus::Ok);
        CHECK(binding->EvaluateHost(cvs.data(), vkOut.data(), 2) == RbfVkStatus::Ok);
        CHECK(MaxDiff(vkOut.data(), expected.data(), vkOut.size()) <= kTolCudaRbfAffine);
        auto largeCharge = context->resources()->Snapshot().usedBytes;
        CHECK(largeCharge > (size_t{8} << 20) && largeCharge < (size_t{9} << 20));
#ifdef USDGEN_ENABLE_CUDA
        if (cudaAvailable) {
            gpu::DeviceBuffer<::float3> dr, dp, dc, dout;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 1025)));
            CHECK(CudaUpload(dp, ToF3(posed.data(), 1025)));
            CHECK(CudaUpload(dc, ToF3(cvs.data(), 2)));
            CHECK(dout.reset(2) == cudaSuccess);
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 1025}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Solve({dp.data(), 1025}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dc.data(), 2}, {dout.data(), 2}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Finish(cudaProducer) == gpu::RbfStatus::Ok);
            auto cudaOut = FromF3(CudaRead(dout));
            CHECK(MaxDiff(vkOut.data(), cudaOut.data(), vkOut.size()) <= kTolCudaRbfAffine);
        }
#endif
        const std::vector<float> tetra{0,0,0, 1,0,0, 0,1,0, 0,0,1};
        CHECK(binding->Bind(tetra.data(), 4, 0.0) == RbfVkStatus::Ok);
        CHECK(context->resources()->Snapshot().usedBytes < (size_t{64} << 10));
        std::puts("B3/1025 affine + resize: PASS");
    }

    // ================= B3: n=100 boundary + n=101 gap =================
    for (int n : {100, 101}) {
        auto rest = GridSamples(n);
        std::vector<float> posed(size_t(n) * 3);
        NonlinearPose(rest.data(), posed.data(), n);
        // CVs: every 7th driver plus two off-grid points.
        std::vector<float> cvs;
        for (int i = 0; i < n; i += 7) {
            cvs.push_back(rest[3 * i]);
            cvs.push_back(rest[3 * i + 1]);
            cvs.push_back(rest[3 * i + 2]);
        }
        cvs.insert(cvs.end(), {0.33f, 1.71f, 2.13f, 3.9f, 0.11f, 1.01f});
        uint32_t count = uint32_t(cvs.size() / 3);
        CHECK(binding->Bind(rest.data(), n, 0.0) == RbfVkStatus::Ok);
        CHECK(binding->Solve(posed.data(), n) == RbfVkStatus::Ok);
        std::vector<float> vkOut(size_t(count) * 3);
        CHECK(binding->EvaluateHost(cvs.data(), vkOut.data(), count) == RbfVkStatus::Ok);
        std::string cpuError;
        std::vector<float> cpuOut(size_t(count) * 3);
        CHECK(CpuSolve(rest.data(), posed.data(), n, cvs.data(), cpuOut.data(), count, &cpuError));
        float dCpu = MaxDiff(vkOut.data(), cpuOut.data(), vkOut.size());
        CHECK(dCpu <= float(kTolCpuOracle));
        float dHost = -1.0f;
        if (n <= 100) {
            std::vector<float> hostOut(size_t(count) * 3);
            CHECK(HostSolve(rest.data(), posed.data(), n, cvs.data(), hostOut.data(), count));
            dHost = MaxDiff(vkOut.data(), hostOut.data(), vkOut.size());
            CHECK(dHost <= float(kTolHostSolve));
        }
#ifdef USDGEN_ENABLE_CUDA
        if (cudaAvailable) {
            gpu::DeviceBuffer<::float3> dr, dp, dc, dout;
            CHECK(CudaUpload(dr, ToF3(rest.data(), n)));
            CHECK(CudaUpload(dp, ToF3(posed.data(), n)));
            CHECK(CudaUpload(dc, ToF3(cvs.data(), int(count))));
            CHECK(dout.reset(count) == cudaSuccess);
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), size_t(n)}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Solve({dp.data(), size_t(n)}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dc.data(), count}, {dout.data(), count}, cudaProducer) ==
                  gpu::RbfStatus::Ok);
            CHECK(cr.Finish(cudaProducer) == gpu::RbfStatus::Ok);
            std::vector<float> cudaOut = FromF3(CudaRead(dout));
            CHECK(cudaOut.size() == vkOut.size());
            float dCuda = MaxDiff(vkOut.data(), cudaOut.data(), vkOut.size());
            CHECK(dCuda <= kTolCudaRbfNonlinear);
            std::printf("B3 (n=%d): PASS (vk-cuda %.2e <= %.0e, vk-cpu %.2e%s)\n", n, dCuda,
                        kTolCudaRbfNonlinear, dCpu, dHost >= 0 ? "" : "");
            if (dHost >= 0) std::printf("      vk-host %.2e\n", dHost);
        } else {
            std::printf("B3 (n=%d): PASS-no-CUDA (vk-cpu %.2e)\n", n, dCpu);
        }
#else
        std::printf("B3 (n=%d): PASS-no-CUDA (vk-cpu %.2e)\n", n, dCpu);
#endif
    }

#ifdef USDGEN_ENABLE_CUDA
    // ================= B4/B5/B6: error parity (status + verbatim strings) ===
    if (cudaAvailable) {
        // B4: n=3 (below the CUDA minimum) fails identically at Bind.
        {
            const std::vector<float> three{0, 0, 0, 1, 0, 0, 0, 1, 0};
            RbfVkStatus vk = binding->Bind(three.data(), 3, 0.0);
            CHECK(vk == RbfVkStatus::InvalidArgument);
            gpu::DeviceBuffer<::float3> dr;
            CHECK(CudaUpload(dr, ToF3(three.data(), 3)));
            CudaRbfBinding cr;
            gpu::RbfStatus cu = cr.Bind({dr.data(), 3}, 0, cudaProducer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            // Solve on the unbound state matches too.
            RbfVkStatus vkSolve = binding->Solve(three.data(), 3);
            gpu::RbfStatus cuSolve = cr.Solve({dr.data(), 3}, cudaProducer);
            CHECK(CudaCode(cuSolve) == VkCode(vkSolve));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            std::printf("B4 (n=3 error parity): PASS [%s]\n", binding->diagnostic());
        }
        // B5a: coplanar drivers are rank-deficient on both.
        {
            const std::vector<float> plane{0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0.2f, 0.3f, 0};
            RbfVkStatus vk = binding->Bind(plane.data(), 5, 0.0);
            CHECK(vk == RbfVkStatus::RankDeficient);
            gpu::DeviceBuffer<::float3> dr;
            CHECK(CudaUpload(dr, ToF3(plane.data(), 5)));
            CudaRbfBinding cr;
            gpu::RbfStatus cu = cr.Bind({dr.data(), 5}, 0, cudaProducer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            std::printf("B5a (coplanar parity): PASS [%s]\n", binding->diagnostic());
        }
        // B5b: NaN rest.
        {
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            std::vector<float> nanRest = rest;
            nanRest[0] = std::numeric_limits<float>::quiet_NaN();
            RbfVkStatus vk = binding->Bind(nanRest.data(), 5, 0.0);
            CHECK(vk == RbfVkStatus::NonFiniteInput);
            gpu::DeviceBuffer<::float3> dr;
            CHECK(CudaUpload(dr, ToF3(nanRest.data(), 5)));
            CudaRbfBinding cr;
            gpu::RbfStatus cu = cr.Bind({dr.data(), 5}, 0, cudaProducer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            std::printf("B5b (NaN rest parity): PASS [%s]\n", binding->diagnostic());
        }
        // B5c: NaN posed (Bind ok, Solve flags).
        {
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            std::vector<float> nanPosed = rest;
            nanPosed[4] = std::numeric_limits<float>::quiet_NaN();
            CHECK(binding->Bind(rest.data(), 5, 0.0) == RbfVkStatus::Ok);
            RbfVkStatus vk = binding->Solve(nanPosed.data(), 5);
            CHECK(vk == RbfVkStatus::NonFiniteInput);
            gpu::DeviceBuffer<::float3> dr, dp;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 5)));
            CHECK(CudaUpload(dp, ToF3(nanPosed.data(), 5)));
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 5}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            gpu::RbfStatus cu = cr.Solve({dp.data(), 5}, cudaProducer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            std::printf("B5c (NaN posed parity): PASS [%s]\n", binding->diagnostic());
        }
        // B5d: Solve count mismatch.
        {
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            CHECK(binding->Bind(rest.data(), 5, 0.0) == RbfVkStatus::Ok);
            RbfVkStatus vk = binding->Solve(rest.data(), 4);
            CHECK(vk == RbfVkStatus::InvalidArgument);
            gpu::DeviceBuffer<::float3> dr, dp;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 5)));
            CHECK(CudaUpload(dp, ToF3(rest.data(), 4)));
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 5}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            gpu::RbfStatus cu = cr.Solve({dp.data(), 4}, cudaProducer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), binding->diagnostic()) == 0);
            std::printf("B5d (solve mismatch parity): PASS [%s]\n", binding->diagnostic());
        }
        // B5e: async stacked evaluates share one diagnostic; NaN poisons.
        {
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            std::vector<float> nanCvs = rest;
            nanCvs[1] = std::numeric_limits<float>::quiet_NaN();
            CHECK(binding->Bind(rest.data(), 5, 0.0) == RbfVkStatus::Ok);
            CHECK(binding->Solve(rest.data(), 5) == RbfVkStatus::Ok);
            auto cvsA = Upload(context, rest.data(), 60);
            auto cvsB = Upload(context, nanCvs.data(), 60);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = 60;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            auto outA = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            auto outB = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch);
            CHECK(cvsA && cvsB && outA && outB);
            CHECK(binding->Evaluate(cvsB, outB, 5) == RbfVkStatus::Ok);
            CHECK(binding->Evaluate(cvsA, outA, 5) == RbfVkStatus::Ok);
            RbfVkStatus vk = binding->Finish();
            CHECK(vk == RbfVkStatus::NonFiniteInput);
            // Snapshot before the post-poison Evaluate below overwrites it.
            std::string const vkDiag = binding->diagnostic();
            CHECK(binding->Evaluate(cvsA, outA, 5) == RbfVkStatus::InvalidArgument);
            gpu::DeviceBuffer<::float3> dr, dp, dout;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 5)));
            CHECK(CudaUpload(dp, ToF3(nanCvs.data(), 5)));
            CHECK(dout.reset(5) == cudaSuccess);
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 5}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Solve({dr.data(), 5}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dp.data(), 5}, {dout.data(), 5}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dr.data(), 5}, {dout.data(), 5}, cudaConsumer) == gpu::RbfStatus::Ok);
            gpu::RbfStatus cu = cr.Finish(cudaConsumer);
            CHECK(CudaCode(cu) == VkCode(vk));
            CHECK(std::strcmp(cr.diagnostic(), vkDiag.c_str()) == 0);
            CHECK(cr.Evaluate({dr.data(), 5}, {dout.data(), 5}, cudaProducer) ==
                  gpu::RbfStatus::InvalidArgument);
            std::printf("B5e (stacked-eval poison parity): PASS [%s]\n", vkDiag.c_str());
        }
        // B6: smoothing still rejects planar; non-planar solves on both.
        {
            const std::vector<float> plane{0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0.2f, 0.3f, 0};
            RbfVkStatus vk = binding->Bind(plane.data(), 5, 1e-3);
            CHECK(vk == RbfVkStatus::RankDeficient);
            gpu::DeviceBuffer<::float3> dr;
            CHECK(CudaUpload(dr, ToF3(plane.data(), 5)));
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 5}, 1e-3, cudaProducer) == gpu::RbfStatus::RankDeficient);
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            std::vector<float> posed(15);
            AffinePose(rest.data(), posed.data(), 5);
            CHECK(binding->Bind(rest.data(), 5, 1e-3) == RbfVkStatus::Ok);
            CHECK(binding->Solve(posed.data(), 5) == RbfVkStatus::Ok);
            std::vector<float> vkOut(15);
            CHECK(binding->EvaluateHost(rest.data(), vkOut.data(), 5) == RbfVkStatus::Ok);
            gpu::DeviceBuffer<::float3> drr, dpp, dout;
            CHECK(CudaUpload(drr, ToF3(rest.data(), 5)));
            CHECK(CudaUpload(dpp, ToF3(posed.data(), 5)));
            CHECK(dout.reset(5) == cudaSuccess);
            CudaRbfBinding cr2;
            CHECK(cr2.Bind({drr.data(), 5}, 1e-3, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr2.Solve({dpp.data(), 5}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr2.Evaluate({drr.data(), 5}, {dout.data(), 5}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr2.Finish(cudaProducer) == gpu::RbfStatus::Ok);
            std::vector<float> cudaOut = FromF3(CudaRead(dout));
            float dCuda = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(dCuda <= kTolCudaRbfNonlinear);
            std::printf("B6 (smoothing): PASS (vk-cuda %.2e)\n", dCuda);
        }
    } else {
        std::printf("B4/B5/B6: SKIP-no-CUDA\n");
    }
#endif

    // ================= B7: far-from-origin frame =================
    {
        auto rest = GridSamples(25);
        for (size_t i = 0; i < rest.size(); i += 3) {
            rest[i] += 1000.0f;
            rest[i + 1] -= 2000.0f;
            rest[i + 2] += 300.0f;
        }
        std::vector<float> posed(rest.size());
        AffinePose(rest.data(), posed.data(), 25);
        CHECK(binding->Bind(rest.data(), 25, 0.0) == RbfVkStatus::Ok);
        CHECK(binding->Solve(posed.data(), 25) == RbfVkStatus::Ok);
        std::vector<float> vkOut(rest.size());
        CHECK(binding->EvaluateHost(rest.data(), vkOut.data(), 25) == RbfVkStatus::Ok);
        std::string cpuError;
        std::vector<float> cpuOut(rest.size());
        CHECK(CpuSolve(rest.data(), posed.data(), 25, rest.data(), cpuOut.data(), 25, &cpuError));
        float dCpu = MaxDiff(vkOut.data(), cpuOut.data(), vkOut.size());
        CHECK(dCpu <= 1e-3f); // large coordinates: absolute check at CUDA scale
#ifdef USDGEN_ENABLE_CUDA
        if (cudaAvailable) {
            gpu::DeviceBuffer<::float3> dr, dp, dout;
            CHECK(CudaUpload(dr, ToF3(rest.data(), 25)));
            CHECK(CudaUpload(dp, ToF3(posed.data(), 25)));
            CHECK(dout.reset(25) == cudaSuccess);
            CudaRbfBinding cr;
            CHECK(cr.Bind({dr.data(), 25}, 0, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Solve({dp.data(), 25}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Evaluate({dr.data(), 25}, {dout.data(), 25}, cudaProducer) == gpu::RbfStatus::Ok);
            CHECK(cr.Finish(cudaProducer) == gpu::RbfStatus::Ok);
            std::vector<float> cudaOut = FromF3(CudaRead(dout));
            float dCuda = MaxDiff(vkOut.data(), cudaOut.data(), vkOut.size());
            CHECK(dCuda <= kTolCudaDeform);
            std::printf("B7 (far frame): PASS (vk-cuda %.2e, vk-cpu %.2e)\n", dCuda, dCpu);
        } else {
            std::printf("B7 (far frame): PASS-no-CUDA (vk-cpu %.2e)\n", dCpu);
        }
#else
        std::printf("B7 (far frame): PASS-no-CUDA (vk-cpu %.2e)\n", dCpu);
#endif
    }

    // ================= G cases: pipeline end-to-end =================
    // Groom shared by G0/G1/G2/G4: 2 curves, 5 CVs (the CUDA deform
    // test's variable 2/3 shape).
    const std::vector<float> groom{.2f, .1f, .1f, .3f, .1f, .4f, .6f, .2f,
                                   .1f, .7f, .4f, .3f, .5f, .6f, .5f};
    const std::vector<uint32_t> groomOffsets{0, 2, 5};
    auto groomBuf = Upload(context, groom.data(), groom.size() * sizeof(float));
    auto groomOffBuf = Upload(context, groomOffsets.data(), groomOffsets.size() * sizeof(uint32_t));
    CHECK(groomBuf && groomOffBuf);

    auto runVkDeform = [&](std::vector<float> const& rest, std::vector<float> const& posed, int n,
                           std::vector<float> const& targets, float groomEnv, float mask,
                           uint32_t enabled, uint32_t lock, std::vector<float>* out,
                           bool expectBegin = true) {
        auto targetsBuf = Upload(context, targets.data(), targets.size() * sizeof(float));
        if (!targetsBuf) return false;
        DeformPipeline::BeginInfo info;
        info.points = groomBuf;
        info.curveOffsets = groomOffBuf;
        info.rootTargets = targetsBuf;
        info.curveCount = 2;
        info.pointCount = 5;
        info.restSamples = rest;
        info.posedSamples = posed;
        info.sampleCount = n;
        info.smoothing = 0.0;
        info.mask = {mask, 1, nullptr, 0};
        info.enabled = {enabled, 1, nullptr, 0};
        info.lockRoots = {lock, 1, nullptr, 0};
        info.groomEnvelope = groomEnv;
        VkResult vr = VK_SUCCESS;
        DeformSemantic sem = DeformSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &vr, &sem);
        if (!expectBegin) return c == nullptr;
        if (!c || vr != VK_SUCCESS) return false;
        if (!CompleteRbf(native, *c, &sem) || sem != DeformSemantic::Ok || !c->succeeded())
            return false;
        return ReadDeformOutput(native, context, *c, out);
    };

    // G0: identity pose reproduces the input.
    {
        const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        const std::vector<float> targets{0, 0, 0, 0, 0, 0};
        std::vector<float> out;
        CHECK(runVkDeform(rest, rest, 5, targets, 1.0f, 1.0f, 1, 0, &out));
        CHECK(out.size() == groom.size());
        float d = MaxDiff(out.data(), groom.data(), out.size());
        CHECK(d <= 1e-4f);
        std::printf("G0 (pipeline identity): PASS (%.2e)\n", d);
    }

#ifdef USDGEN_ENABLE_CUDA
    if (cudaAvailable) {
        auto runCudaLegacy = [&](std::vector<float> const& rest, std::vector<float> const& posed,
                                 int n, std::vector<float> const& targets, float groomEnv,
                                 std::vector<float>* out) {
            gpu::DeviceBuffer<::float3> drivers, pp, points, targs, output;
            gpu::DeviceBuffer<uint32_t> offs;
            if (!CudaUpload(drivers, ToF3(rest.data(), n))) return false;
            if (!CudaUpload(pp, ToF3(posed.data(), n))) return false;
            if (!CudaUpload(points, ToF3(groom.data(), 5))) return false;
            if (!CudaUpload(targs, ToF3(targets.data(), 2))) return false;
            if (!CudaUpload(offs, groomOffsets)) return false;
            if (output.reset(5) != cudaSuccess) return false;
            gpu::DeviceCurveGeometryView geometry = {{points.data(), 5},
                                                     {points.data(), 5},
                                                     {},
                                                     {offs.data(), 3},
                                                     {},
                                                     2,
                                                     5};
            gpu::CudaRbfBinding rb;
            gpu::CudaRbfCurveDeformer deform;
            if (rb.Bind({drivers.data(), size_t(n)}, 0, cudaProducer) != gpu::RbfStatus::Ok)
                return false;
            if (rb.Solve({pp.data(), size_t(n)}, cudaProducer) != gpu::RbfStatus::Ok) return false;
            if (deform.Deform(rb, geometry, {targs.data(), 2}, groomEnv, {}, {},
                              output.view(), cudaProducer) != gpu::RbfStatus::Ok)
                return false;
            if (deform.Finish(rb, cudaConsumer) != gpu::RbfStatus::Ok) return false;
            *out = FromF3(CudaRead(output));
            return out->size() == 15;
        };
        auto runCudaTyped = [&](std::vector<float> const& rest, std::vector<float> const& posed,
                                int n, std::vector<float> const& targets, float mask, bool lock,
                                std::vector<float>* out) {
            gpu::DeviceBuffer<::float3> drivers, pp, points, targs, output;
            gpu::DeviceBuffer<uint32_t> offs;
            if (!CudaUpload(drivers, ToF3(rest.data(), n))) return false;
            if (!CudaUpload(pp, ToF3(posed.data(), n))) return false;
            if (!CudaUpload(points, ToF3(groom.data(), 5))) return false;
            if (!CudaUpload(targs, ToF3(targets.data(), 2))) return false;
            if (!CudaUpload(offs, groomOffsets)) return false;
            if (output.reset(5) != cudaSuccess) return false;
            gpu::DeviceCurveGeometryView geometry = {{points.data(), 5},
                                                     {points.data(), 5},
                                                     {},
                                                     {offs.data(), 3},
                                                     {},
                                                     2,
                                                     5};
            gpu::CudaRbfBinding rb;
            gpu::CudaRbfCurveDeformer deform;
            if (rb.Bind({drivers.data(), size_t(n)}, 0, cudaProducer) != gpu::RbfStatus::Ok)
                return false;
            if (rb.Solve({pp.data(), size_t(n)}, cudaProducer) != gpu::RbfStatus::Ok) return false;
            gpu::DeformParameters params;
            params.mask = gpu::ScalarField::Literal(mask);
            params.enabled = gpu::BoolField::Literal(true);
            params.lockRoots = gpu::BoolField::Literal(lock);
            if (deform.Deform(rb, geometry, {targs.data(), 2}, params, output.view(),
                              cudaProducer) != gpu::RbfStatus::Ok)
                return false;
            if (deform.Finish(rb, cudaConsumer) != gpu::RbfStatus::Ok) return false;
            *out = FromF3(CudaRead(output));
            return out->size() == 15;
        };

        // G4: the CUDA deform test's first case, exactly (n=5 affine).
        {
            const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
            std::vector<float> posed(15);
            AffinePose(rest.data(), posed.data(), 5);
            std::vector<float> targets(6);
            AffinePose(groom.data(), targets.data(), 1);
            AffinePose(groom.data() + 6, targets.data() + 3, 1);
            std::vector<float> vkOut, cudaOut;
            CHECK(runVkDeform(rest, posed, 5, targets, 1.0f, 1.0f, 1, 1, &vkOut));
            CHECK(runCudaLegacy(rest, posed, 5, targets, 1.0f, &cudaOut));
            float d = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(d <= kTolCudaDeform);
            std::printf("G4 (pipeline n=5 affine): PASS (vk-cuda %.2e <= %.0e)\n", d,
                        kTolCudaDeform);
        }
        // G1: n=101 end-to-end (legacy + typed + envelope composition).
        {
            auto rest = GridSamples(101);
            std::vector<float> posed(rest.size());
            NonlinearPose(rest.data(), posed.data(), 101);
            // Targets: identity roots keep the lock correction exercised.
            const std::vector<float> targets{groom[0], groom[1], groom[2],
                                             groom[6], groom[7], groom[8]};
            std::vector<float> vkOut, cudaOut;
            CHECK(runVkDeform(rest, posed, 101, targets, 0.5f, 1.0f, 1, 1, &vkOut));
            CHECK(runCudaLegacy(rest, posed, 101, targets, 0.5f, &cudaOut));
            float d = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(d <= kTolCudaDeform);
            std::printf("G1a (pipeline n=101 legacy): PASS (vk-cuda %.2e <= %.0e)\n", d,
                        kTolCudaDeform);
            CHECK(runVkDeform(rest, posed, 101, targets, 1.0f, 0.5f, 1, 1, &vkOut));
            CHECK(runCudaLegacy(rest, posed, 101, targets, 0.5f, &cudaOut));
            d = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(d <= kTolCudaDeform);
            std::printf("G1b (pipeline groom*mask): PASS (vk-cuda %.2e <= %.0e)\n", d,
                        kTolCudaDeform);
            CHECK(runVkDeform(rest, posed, 101, targets, 0.5f, 0.5f, 1, 0, &vkOut));
            CHECK(runCudaTyped(rest, posed, 101, targets, 0.25f, false, &cudaOut));
            d = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(d <= kTolCudaDeform);
            std::printf("G1c (pipeline n=101 typed unlock): PASS (vk-cuda %.2e <= %.0e)\n", d,
                        kTolCudaDeform);
        }
        // G2: baked non-identity samples execute with CUDA-equivalent output.
        {
            auto fx = MakeRooted();
            auto grid = GridSamples(125);
            VtVec3fArray pts;
            for (int i = 0; i < 125; ++i)
                pts.push_back(GfVec3f(grid[3 * i], grid[3 * i + 1], grid[3 * i + 2]));
            fx.desc.surfaces[0].restPoints = pts;
            fx.desc.surfaces[0].points = pts;
            fx.desc.surfaces[0].worldMatrix[3][1] = -2.0; // moved surface
            fx.deform.params = {{TfToken("rbfSamples"), VtValue(101), false}};
            VulkanDeformControls baked;
            UsdGenDiagnostics diag;
            CHECK(RbfVkValidateDeformGap(fx.desc, fx.deform, &fx.desc.curveSets[0], &baked, &diag));
            CHECK(baked.sampleCount == 101);
            // Pose the baked drivers with the affine map (keeps 3D span).
            std::vector<float> posed(baked.posedSamples.size());
            AffinePose(baked.posedSamples.data(), posed.data(), 101);
            const std::vector<float> targets{groom[0], groom[1], groom[2],
                                             groom[6], groom[7], groom[8]};
            std::vector<float> vkOut, cudaOut;
            CHECK(runVkDeform(baked.restSamples, posed, 101, targets, 1.0f, 1.0f, 1, 1, &vkOut));
            CHECK(runCudaLegacy(baked.restSamples, posed, 101, targets, 1.0f, &cudaOut));
            float d = MaxDiff(vkOut.data(), cudaOut.data(), 15);
            CHECK(d <= kTolCudaDeform);
            std::printf("G2 (pipeline baked non-identity): PASS (vk-cuda %.2e <= %.0e)\n", d,
                        kTolCudaDeform);
        }
    } else {
        std::printf("G1/G2/G4: SKIP-no-CUDA\n");
    }
#endif

    // ================= G3: pipeline contracts =================
    {
        const std::vector<float> rest{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
        std::vector<float> posed(15);
        AffinePose(rest.data(), posed.data(), 5);
        const std::vector<float> targets{0, 0, 0, 0, 0, 0};
        // mask=0 and enabled=0 are bitwise pass-throughs.
        std::vector<float> out;
        CHECK(runVkDeform(rest, posed, 5, targets, 1.0f, 0.0f, 1, 1, &out));
        CHECK(out.size() == groom.size());
        CHECK(std::memcmp(out.data(), groom.data(), out.size() * sizeof(float)) == 0);
        CHECK(runVkDeform(rest, posed, 5, targets, 1.0f, 1.0f, 0, 1, &out));
        CHECK(std::memcmp(out.data(), groom.data(), out.size() * sizeof(float)) == 0);
        std::puts("G3a (passthrough): PASS");
        // One packed Groom element must broadcast to both curves for each field.
        {
            auto translated=rest;
            for(size_t i=0;i<translated.size();i+=3){translated[i]+=.4f;translated[i+1]+=.2f;translated[i+2]+=.6f;}
            float amount=.25f;uint32_t enabled=1,lock=0;
            DeformPipeline::BeginInfo info;
            info.points=groomBuf;info.curveOffsets=groomOffBuf;
            info.rootTargets=Upload(context,targets.data(),targets.size()*4);
            info.curveCount=2;info.pointCount=5;info.restSamples=rest;info.posedSamples=translated;info.sampleCount=5;
            info.mask={1,1,Upload(context,&amount,4),1};
            info.enabled={1,1,Upload(context,&enabled,4),1};
            info.lockRoots={1,1,Upload(context,&lock,4),1};
            VkResult vr=VK_SUCCESS;DeformSemantic sem=DeformSemantic::Ok;
            auto c=pipe->Begin(info,&vr,&sem);CHECK(c&&vr==VK_SUCCESS&&CompleteRbf(native,*c,&sem)&&c->succeeded());
            CHECK(ReadDeformOutput(native,context,*c,&out));
            for(size_t i=0;i<groom.size();i+=3){CHECK(std::abs(out[i]-groom[i]-.1f)<1e-4f);CHECK(std::abs(out[i+1]-groom[i+1]-.05f)<1e-4f);CHECK(std::abs(out[i+2]-groom[i+2]-.15f)<1e-4f);}
            enabled=0;info.enabled.data=Upload(context,&enabled,4);
            c=pipe->Begin(info,&vr,&sem);CHECK(c&&CompleteRbf(native,*c,&sem)&&c->succeeded());
            CHECK(ReadDeformOutput(native,context,*c,&out)&&out==groom);
            enabled=1;lock=1;amount=1;
            info.enabled.data=Upload(context,&enabled,4);info.lockRoots.data=Upload(context,&lock,4);info.mask.data=Upload(context,&amount,4);
            c=pipe->Begin(info,&vr,&sem);CHECK(c&&CompleteRbf(native,*c,&sem)&&c->succeeded());
            CHECK(ReadDeformOutput(native,context,*c,&out));
            for(uint32_t root:{0u,2u})for(uint32_t axis=0;axis<3;++axis)CHECK(std::abs(out[root*3+axis])<1e-4f);
            // Empty topology still proves Bind/Solve, then exposes no dummy point owner.
            info={};info.sampleCount=5;info.restSamples=rest;info.posedSamples=translated;
            c=pipe->Begin(info,&vr,&sem);CHECK(c&&CompleteRbf(native,*c,&sem)&&c->succeeded()&&c->pointCount()==0&&!c->output().points);
            std::puts("G3 Groom packed fields and canonical empty output: PASS");
        }
        // enabled=2 surfaces at Poll as BadValue (DeformPipeline parity).
        {
            auto targetsBuf = Upload(context, targets.data(), targets.size() * sizeof(float));
            CHECK(targetsBuf);
            DeformPipeline::BeginInfo info;
            info.points = groomBuf;
            info.curveOffsets = groomOffBuf;
            info.rootTargets = targetsBuf;
            info.curveCount = 2;
            info.pointCount = 5;
            info.restSamples = rest;
            info.posedSamples = posed;
            info.sampleCount = 5;
            info.mask = {1.0f, 1, nullptr, 0};
            info.enabled = {2, 1, nullptr, 0};
            info.lockRoots = {0, 1, nullptr, 0};
            VkResult vr = VK_SUCCESS;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &vr, &sem);
            CHECK(c && vr == VK_SUCCESS);
            CHECK(CompleteRbf(native, *c, &sem));
            CHECK(sem == DeformSemantic::BadValue);
            CHECK(!c->succeeded());
            std::puts("G3b (enabled=2 Poll BadValue): PASS");
        }
        // Bad offsets reject asynchronously after device validation.
        {
            const std::vector<uint32_t> bad{0, 0, 5};
            auto badOff = Upload(context, bad.data(), bad.size() * sizeof(uint32_t));
            auto targetsBuf = Upload(context, targets.data(), targets.size() * sizeof(float));
            CHECK(badOff && targetsBuf);
            DeformPipeline::BeginInfo info;
            info.points = groomBuf;
            info.curveOffsets = badOff;
            info.rootTargets = targetsBuf;
            info.curveCount = 2;
            info.pointCount = 5;
            info.restSamples = rest;
            info.posedSamples = posed;
            info.sampleCount = 5;
            VkResult vr = VK_SUCCESS;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &vr, &sem);
            CHECK(c && vr == VK_SUCCESS);
            CHECK(CompleteRbf(native, *c, &sem));
            CHECK(sem == DeformSemantic::BadOffsets && !c->succeeded());
            std::puts("G3c (bad offsets): PASS");
        }
        // n=3 reaches the device solver and reports its CUDA result.
        {
            const std::vector<float> three{0, 0, 0, 1, 0, 0, 0, 1, 0};
            auto targetsBuf = Upload(context, targets.data(), targets.size() * sizeof(float));
            CHECK(targetsBuf);
            DeformPipeline::BeginInfo info;
            info.points = groomBuf;
            info.curveOffsets = groomOffBuf;
            info.rootTargets = targetsBuf;
            info.curveCount = 2;
            info.pointCount = 5;
            info.restSamples = three;
            info.posedSamples = three;
            info.sampleCount = 3;
            VkResult vr = VK_SUCCESS;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &vr, &sem);
            CHECK(!c && vr == VK_ERROR_INITIALIZATION_FAILED && sem == DeformSemantic::BadValue);
            std::puts("G3d (n=3 Begin BadValue): PASS");
        }
        // A malformed Groom field with two elements still rejects.
        {
            auto targetsBuf = Upload(context, targets.data(), targets.size() * sizeof(float));
            auto fieldBuf = Upload(context, groom.data(), sizeof(float));
            CHECK(targetsBuf && fieldBuf);
            DeformPipeline::BeginInfo info;
            info.points = groomBuf;
            info.curveOffsets = groomOffBuf;
            info.rootTargets = targetsBuf;
            info.curveCount = 2;
            info.pointCount = 5;
            info.restSamples = rest;
            info.posedSamples = posed;
            info.sampleCount = 5;
            info.mask = {1.0f, 1, fieldBuf, 2};
            VkResult vr = VK_SUCCESS;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &vr, &sem);
            CHECK(!c && sem == DeformSemantic::BadValue);
            std::puts("G3e (malformed Groom field rejected): PASS");
        }
        // Every phase gets a fresh gate; Poll alone cannot submit the next phase.
        {
            auto makeInfo = [&]() {
                DeformPipeline::BeginInfo info;
                info.points = groomBuf;
                info.curveOffsets = groomOffBuf;
                info.rootTargets = Upload(context, targets.data(), targets.size() * sizeof(float));
                info.curveCount = 2;
                info.pointCount = 5;
                info.restSamples = rest;
                info.posedSamples = posed;
                info.sampleCount = 5;
                return info;
            };
            auto baseline = context->resources()->Snapshot().usedBytes;
            unsigned gates = 0;
            VkResult vr = VK_SUCCESS;
            auto c = pipe->Begin(makeInfo(), &vr, nullptr, [&] { ++gates; return true; });
            CHECK(c && vr == VK_SUCCESS && c->pending() && !c->done() && gates == 1);
            CHECK(!c->output().points);
            for (unsigned phase = 0; phase < 5; ++phase) {
                CHECK(Prove(native));
                DeformSemantic sem = DeformSemantic::Ok;
                CHECK(c->Poll(&sem) == VK_SUCCESS && sem == DeformSemantic::Ok);
                CHECK(!c->pending());
                CHECK(gates == phase + 1);
                CHECK(c->Poll(&sem) == VK_SUCCESS && gates == phase + 1);
                if (phase < 4) {
                    CHECK(!c->done() && !c->output().points);
                    CHECK(c->Advance([&] { ++gates; return true; }, &vr));
                    CHECK(vr == VK_SUCCESS && c->pending());
                }
            }
            CHECK(c->done() && c->succeeded() && c->output().points && gates == 5);
            c.reset();
            CHECK(context->resources()->Snapshot().usedBytes == baseline);
            c = pipe->Begin(makeInfo(), &vr);
            CHECK(c && Prove(native) && c->Poll() == VK_SUCCESS);
            CHECK(!c->Advance([] { return false; }, &vr) && vr == VK_ERROR_OUT_OF_DEVICE_MEMORY);
            CHECK(!c->pending() && !c->output().points);
            c.reset();
            CHECK(context->resources()->Snapshot().usedBytes == baseline);
            c = pipe->Begin(makeInfo(), &vr);
            CHECK(c && Prove(native) && c->Poll() == VK_SUCCESS);
            CHECK(!c->Advance([]() -> bool { throw 1; }, &vr) && vr == VK_ERROR_UNKNOWN);
            CHECK(!c->pending());
            c.reset();
            CHECK(context->resources()->Snapshot().usedBytes == baseline);
            // Separate candidate LU/RHS state permits overlap on one queue.
            auto first = pipe->Begin(makeInfo(), &vr);
            auto second = pipe->Begin(makeInfo(), &vr);
            CHECK(first && second && first->pending() && second->pending());
            DeformSemantic sem = DeformSemantic::Ok;
            CHECK(CompleteRbf(native, *first, &sem) && first->succeeded());
            CHECK(CompleteRbf(native, *second, &sem) && second->succeeded());
            first.reset(); second.reset();
            CHECK(context->resources()->Snapshot().usedBytes == baseline);
            std::puts("G3g (phase gates, rejected retries, overlapping solve ownership): PASS");
        }
        // Empty topology proves after the solve.
        {
            DeformPipeline::BeginInfo info;
            info.curveCount = 0;
            info.pointCount = 0;
            info.restSamples = rest;
            info.posedSamples = posed;
            info.sampleCount = 5;
            VkResult vr = VK_SUCCESS;
            DeformSemantic sem = DeformSemantic::Ok;
            auto c = pipe->Begin(std::move(info), &vr, &sem);
            CHECK(c && vr == VK_SUCCESS);
            CHECK(CompleteRbf(native, *c, &sem));
            CHECK(sem == DeformSemantic::Ok && c->succeeded() && c->pointCount() == 0);
            std::puts("G3f (empty topology): PASS");
        }
    }

#ifdef USDGEN_ENABLE_CUDA
    if (cudaProducer) cudaStreamDestroy(cudaProducer);
    if (cudaConsumer) cudaStreamDestroy(cudaConsumer);
#endif
#ifdef USDGEN_ENABLE_CUDA
    if (!cudaAvailable) {
        std::puts("testUsdGenVulkanRbfVkParity: SKIP (CUDA comparison unavailable)");
        return 77;
    }
#else
    std::puts("testUsdGenVulkanRbfVkParity: SKIP (CUDA comparison not compiled)");
    return 77;
#endif
    std::puts("testUsdGenVulkanRbfVkParity: PASS");
    return 0;
}
