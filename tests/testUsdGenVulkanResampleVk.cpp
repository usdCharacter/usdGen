// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanResampleVk — indexed-CV resample parity gate (phase 2, resample).
//
// The Vulkan ResampleVkPipeline must reproduce the CUDA CudaCurveResample
// results exactly (same double-precision index math) for points, rest,
// widths, hairT, offsets, stable IDs and root channels, and the CUDA
// resample-mode named-channel transform for Generic point/primitive/groom
// planes. The shared fixture mirrors tests/testUsdGenCudaCurveResample.cpp
// (ragged {2,4} curves resampled to 4 uniform CVs) plus Generic named planes
// in every domain. hairT compares bitwise against the CPU and CUDA oracles.
// Other arithmetic float lanes compare within the CUDA tests' own Near()
// contract (|a-b| < 1e-5, testUsdGenCudaCurveResample.cpp and
// testUsdGenCudaNamedChannelTopology.cu); verbatim copies (offsets, stable
// IDs, rootPrim/rootUV, int/uniform/constant planes) compare bitwise.
//
// Builds with and without CUDA: the live CUDA twin runs under
// USDGEN_ENABLE_CUDA, while hardcoded oracle vectors (transcribed from the
// CUDA test's expectedX/expectedWidths/expectedHairT) verify the Vulkan leg
// in every build. Exits 77 when either backend is unavailable, after running
// the available Vulkan oracle checks. Available-device failures fail the test.
#include "utils/pxrInterop.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "testBackend.h"
#include "usdGenUsd/lower.h"
#include "usdGenCurves/vulkan/resampleVkPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"
#include "usdGenCurves/vulkan/executionPlan.h"
#include "usdGen/exec/device/executionBackend.h"
#include "usdGen/registries/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGenCurves/types/curveBuffer.h"
#include "usdGen/exec/device/deviceGeneration.h"
#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "usdGenCurves/cuda/curveResample.h"
#include "usdGenCurves/cuda/namedChannelTopology.h"
#include "usdGen/exec/device/executionResources.h"
#include "usdGen/exec/device/executionRetirement.h"
#include "usdGenCuda/gpu/cudaRetirement.h"

#include <cuda_runtime.h>
#endif

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan resampleVk failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

// Tolerance of one float lane against the CUDA twin. This is the CUDA
// tests' own Near() contract (see header comment), not a new bound: the
// Vulkan shader evaluates the identical double-precision formula, so the
// observed worst-case difference is expected to be ~0.
static bool Near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

namespace {
using Packet = std::map<std::string, std::vector<uint8_t>>;

template <class T> std::vector<uint8_t> Bytes(T const* values, size_t count) {
    std::vector<uint8_t> result(count * sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), values, result.size());
    return result;
}
template <class T> std::vector<T> As(std::vector<uint8_t> const& bytes) {
    std::vector<T> result(bytes.size() / sizeof(T));
    if (!result.empty()) std::memcpy(result.data(), bytes.data(), result.size() * sizeof(T));
    return result;
}

static bool ExactHairT(std::vector<uint8_t> const& actual,
                      std::vector<uint8_t> const& expected,
                      uint32_t target, char const* oracle) {
    if (actual == expected) return true;
    if (actual.size() != expected.size()) {
        std::fprintf(stderr, "hairT target=%u %s size mismatch: Vulkan=%zu expected=%zu\n",
                     target, oracle, actual.size(), expected.size());
        return false;
    }
    auto const got = As<float>(actual), want = As<float>(expected);
    auto const gotBits = As<uint32_t>(actual), wantBits = As<uint32_t>(expected);
    for (size_t i = 0; i != got.size(); ++i) {
        if (gotBits[i] == wantBits[i]) continue;
        std::fprintf(stderr,
            "hairT target=%u curve=%zu cv=%zu %s: Vulkan=%a (0x%08x) expected=%a (0x%08x)\n",
            target, i / target, i % target, oracle,
            double(got[i]), gotBits[i], double(want[i]), wantBits[i]);
    }
    return false;
}

static std::vector<uint32_t> Code(char const* path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t)) return {};
    std::vector<uint32_t> result(raw.size() / sizeof(uint32_t));
    std::memcpy(result.data(), raw.data(), raw.size());
    return result;
}

static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{}; submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static bool Capture(std::vector<ResampleVkPipeline::Candidate::OutputPlane> const& outputs,
    std::shared_ptr<NativeOwner> const& native, std::shared_ptr<DeviceContext> const& context, Packet* packet) {
    packet->clear();
    for (auto const& plane : outputs) {
        std::vector<uint8_t> bytes;
        if (plane.bytes) {
            if (!plane.owner) return false;
            if (!ReadVulkanBytes(native, context, plane.owner->buffer(), size_t(plane.bytes), plane.owner, &bytes))
                return false;
        }
        if (!packet->emplace(plane.metadata.name.GetString(), std::move(bytes)).second) return false;
    }
    return true;
}

// Shared fixture. Core channels transcribe testUsdGenCudaCurveResample.cpp
// verbatim; named planes cover every Generic domain in both CUDA-admitted
// value types (Float32/Int32, the resample-mode ValidateMetadata domain).
static VulkanSourceGenerationCreateInfo Fixture(std::shared_ptr<DeviceContext> context) {
    VulkanSourceGenerationCreateInfo info; info.context = std::move(context);
    auto& s = info.source;
    s.totalCurves = 2; s.totalCvs = 6; s.topologyVersion = 19; s.valueVersion = 23;
    s.px = {0, 4, 10, 11, 14, 15};
    s.py = VtFloatArray(6, 0); s.pz = VtFloatArray(6, 0);
    s.rest.reserve(6); s.width.reserve(6); s.hairT.reserve(6);
    float const rx[6] = {0, 4, 10, 11, 14, 15};
    float const w[6] = {1, 9, 2, 4, 6, 8};
    float const t[6] = {0, 1, 0, .25f, .75f, 1};
    for (uint32_t i = 0; i != 6; ++i) {
        s.rest.push_back(GfVec3f(rx[i], 1, 0)); s.width.push_back(w[i]); s.hairT.push_back(t[i]);
    }
    s.cvOffsets = {0, 2, 6}; s.curveId = {9, 3};
    s.rootPrim = {7, 8};
    s.rootUV = {{.2f, .3f}, {.7f, .8f}};
    UsdGenPlane f; f.name = TfToken("f"); f.interpolation = TfToken("vertex"); f.type = TfToken("float");
    f.f = {0, 1, 2, 3, 4, 5};
    UsdGenPlane iv; iv.name = TfToken("i"); iv.interpolation = TfToken("vertex"); iv.type = TfToken("int");
    iv.i = {10, 11, 12, 13, 14, 15};
    s.extraCv = {f, iv};
    UsdGenPlane u; u.name = TfToken("u"); u.interpolation = TfToken("uniform"); u.type = TfToken("int");
    u.i = {30, 31};
    UsdGenPlane c; c.name = TfToken("c"); c.interpolation = TfToken("constant"); c.type = TfToken("float");
    c.f = {3.5f};
    s.extraCurve = {c, u};
    s.chunks.resize(1);
    s.chunks[0].firstCurve = 0; s.chunks[0].curveCount = 2; s.chunks[0].liveCount = 2;
    s.chunks[0].firstCv = 0; s.chunks[0].cvCount = 0; s.chunks[0].tile = 5; s.chunks[0].surface = 1;
    info.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline, UsdGenCurveLayoutWrap::Pinned);
    info.geometry.tiles = {{5, 0, 2, 0, 6}};
    return info;
}

static std::shared_ptr<const VulkanSourceGeneration> Upload(std::shared_ptr<NativeOwner> const& native,
    VulkanSourceGenerationCreateInfo info) {
    VkResult status = VK_SUCCESS; std::string reason;
    auto upload = VulkanSourceUpload::Create(std::move(info), &status, &reason);
    if (!upload || upload->Submit() != SourceGenerationStatus::Submitted || !Prove(native) ||
        upload->Poll() != SourceGenerationStatus::Ready) return {};
    return upload->TakeReady();
}

static std::unique_ptr<ResampleVkPipeline::Candidate> Run(ResampleVkPipeline& pipeline,
    std::shared_ptr<NativeOwner> const& native, std::shared_ptr<const VulkanSourceGeneration> const& base,
    uint32_t target, uint32_t* semantic) {
    VkResult status = VK_ERROR_UNKNOWN;
    auto candidate = pipeline.Begin(base, target, &status);
    if (!candidate || status != VK_SUCCESS || !Prove(native)) return {};
    *semantic = UINT32_MAX;
    if (candidate->Poll(semantic) != VK_SUCCESS) return {};
    return candidate;
}

// Host oracle for the float lanes, transcribed from the CUDA test's
// expected vectors (target 4). See the header comment for provenance.
static bool CheckTarget4Oracle(Packet const& got, float* worst) {
    *worst = 0;
    auto const& points = As<float>(got.at("points"));
    auto const& rest = As<float>(got.at("rest"));
    auto const& widths = As<float>(got.at("width"));
    auto const& hairT = As<float>(got.at("hairT"));
    auto const& f = As<float>(got.at("f"));
    if (points.size() != 24 || rest.size() != 24 || widths.size() != 8 || hairT.size() != 8 || f.size() != 8)
        return false;
    float const ex[8] = {0, 4.f / 3.f, 8.f / 3.f, 4, 10, 11, 14, 15};
    float const ew[8] = {1, 11.f / 3.f, 19.f / 3.f, 9, 2, 4, 6, 8};
    float const et[8] = {0, 1.f / 3.f, 2.f / 3.f, 1, 0, .25f, .75f, 1};
    float const ef[8] = {0, 1.f / 3.f, 2.f / 3.f, 1, 2, 3, 4, 5};
    if (!ExactHairT(got.at("hairT"), Bytes(et, 8), 4, "CPU")) return false;
    for (size_t p = 0; p != 8; ++p) {
        float const diffs[7] = {std::fabs(points[3 * p] - ex[p]), std::fabs(points[3 * p + 1]),
            std::fabs(points[3 * p + 2]), std::fabs(rest[3 * p] - ex[p]), std::fabs(rest[3 * p + 1] - 1),
            std::fabs(widths[p] - ew[p]), std::fabs(f[p] - ef[p])};
        for (float d : diffs) {
            *worst = std::max(*worst, d);
            if (!(d < 1e-5f)) return false;
        }
        if (std::fabs(rest[3 * p + 2]) >= 1e-5f) return false;
    }
    return true;
}

// Extend the existing fixture's host oracle to fractional target intervals.
// The first curve has linear {0,1} hairT; the second has deliberately
// nonuniform {0,.25,.75,1}. Preserve the CUDA contract's float fraction before
// interpolation so this also checks the authored parameter, not just a newly
// synthesized normalized CV index.
static std::vector<uint8_t> HairTOracle(uint32_t target) {
    std::vector<float> result(2 * target);
    for (uint32_t point = 0; point != target; ++point) {
        result[point] = float(double(point) / double(target - 1));
        double const position = double(point) * 3.0 / double(target - 1);
        uint32_t const segment = uint32_t(position);
        float const fraction = float(position - double(segment));
        double const base = segment == 0 ? 0.0 : segment == 1 ? .25 : .75;
        double const span = segment == 1 ? .5 : .25;
        result[target + point] = segment == 3 ? 1.0f : float(base + span * double(fraction));
    }
    return Bytes(result.data(), result.size());
}

static bool CheckVerbatim(Packet const& got) {
    if (As<uint32_t>(got.at("curveOffsets")) != std::vector<uint32_t>({0, 4, 8})) return false;
    if (As<uint64_t>(got.at("stableIds")) != std::vector<uint64_t>({9, 3})) return false;
    if (As<int32_t>(got.at("rootPrim")) != std::vector<int32_t>({7, 8})) return false;
    if (As<float>(got.at("rootUV")) != std::vector<float>({.2f, .3f, .7f, .8f})) return false;
    if (As<int32_t>(got.at("i")) != std::vector<int32_t>({10, 10, 11, 11, 12, 13, 14, 15})) return false;
    if (As<int32_t>(got.at("u")) != std::vector<int32_t>({30, 31})) return false;
    if (As<float>(got.at("c")) != std::vector<float>({3.5f})) return false;
    return true;
}

#ifdef USDGEN_ENABLE_CUDA
namespace cudaTwin {
using namespace gpu;

template <class T>
static bool Upload(DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        (!values.size() || cudaMemcpy(buffer.data(), values.data(), values.size() * sizeof(T),
                                      cudaMemcpyHostToDevice) == cudaSuccess);
}
template <class T>
static DeviceView<const T> Read(DeviceBuffer<T> const& buffer) {
    return buffer.view();
}
template <class T>
static std::vector<T> Download(DeviceView<const T> view) {
    std::vector<T> result(view.size);
    if (view.size && cudaMemcpy(result.data(), view.data, view.size * sizeof(T),
                                cudaMemcpyDeviceToHost) != cudaSuccess)
        result.clear();
    return result;
}
struct Completion {
    std::atomic<int> status{std::numeric_limits<int>::min()};
    std::atomic<unsigned> calls{0};
};
static void Callback(cudaStream_t, cudaError_t status, void* data) noexcept {
    auto* c = static_cast<Completion*>(data);
    c->status.store(static_cast<int>(status), std::memory_order_release);
    c->calls.fetch_add(1, std::memory_order_relaxed);
}
// Owns every raw device allocation borrowed by the resample + named-topology
// twins through their terminal callbacks.
struct Lifetime {
    DeviceBuffer<::float3> points, restPoints;
    DeviceBuffer<float> widths, hairT, fBytes, cBytes;
    DeviceBuffer<int32_t> iBytes, uBytes;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> stableIds;
    DeviceBuffer<int32_t> rootPrim;
    DeviceBuffer<unsigned char> rootUVBytes;
    CudaCurveResample resampled;
};
} // namespace cudaTwin
#endif

// The neutral resampleTo=0 source control must keep compiling after the
// resample gap closes (GapRejections control, replicated here as the
// integration anchor for this gap's test).
static bool NeutralResampleControl() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom");
    desc.executionBackend = usdGenTestUtils::Backend("vulkan");
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Groom/C3");
    curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2};
    curves.points = {{0, 0, 0}, {0, 1, 0}};
    curves.rest = curves.points;
    curves.widths = {.02f, .03f};
    curves.curveId = {41};
    curves.curveGeneration = 19;
    curves.frozenEpoch = "epoch-19";
    desc.curveSets.push_back(curves);
    UsdGenNodeDesc source;
    source.path = SdfPath("/Groom/Ops/source");
    source.type = TfToken("UsdGenCurveSource");
    source.curves = {curves.path};
    source.params = {{TfToken("useRest"), VtValue(true), false},
        {TfToken("idSource"), VtValue(TfToken("primvar")), false},
        {TfToken("expectEpoch"), VtValue(std::string("epoch-19")), false},
        {TfToken("staleAction"), VtValue(TfToken("block")), false},
        {TfToken("rebind"), VtValue(TfToken("never")), false},
        {TfToken("resampleTo"), VtValue(0), false}};
    UsdGenNodeDesc width;
    width.path = SdfPath("/Groom/Ops/width");
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params = {{TfToken("width"), VtValue(.125f), false},
        {TfToken("replace"), VtValue(false), false}};
    desc.nodes = {source, width};
    desc.terminal = width.path;
    UsdGenDiagnostics diagnostics;
    auto handle = CompileVulkanSourceWidthPlan(usdGenUsd::Lower(desc), &diagnostics);
    return handle && !diagnostics.HasErrors();
}
} // namespace

int main(int argc, char** argv) {
    CHECK(argc == 2); auto code = Code(argv[1]); CHECK(!code.empty());
    CHECK(NeutralResampleControl());
    bool unavailable = false;
    VkPhysicalDeviceFeatures fp64{}; fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateTimelineNative(&unavailable, &fp64);
    if (unavailable) {
        std::puts("resampleVk: no supported Vulkan device; SKIP");
        return 77;
    }
    CHECK(native);
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(native->physical, &feats);
    if (!feats.shaderFloat64) { std::puts("resampleVk: no Float64 support; SKIP"); return 77; }
    DeviceContext::CreateInfo ci; ci.instance = native->instance; ci.physicalDevice = native->physical;
    ci.device = native->device; ci.computeQueue = native->queue; ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex; ci.resourceDeviceId = 7042; ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true; ci.resources = {size_t{16} << 20, 0};
    ci.shaderFloat64Enabled = true;
    auto context = DeviceContext::Create(ci); CHECK(context);
    {
        // Float64 SPIR-V must reject contexts whose device lacks the flag.
        DeviceContext::CreateInfo plain = ci; plain.shaderFloat64Enabled = false;
        plain.resourceDeviceId = 7043; plain.resources = {size_t{1} << 20, 0};
        auto plainContext = DeviceContext::Create(plain); CHECK(plainContext);
        VkResult rejected = VK_SUCCESS;
        CHECK(!ResampleVkPipeline::Create(plainContext, code, &rejected));
        CHECK(plainContext->resources()->Snapshot().usedBytes == 0);
    }
    VkResult status = VK_SUCCESS;
    auto pipeline = ResampleVkPipeline::Create(context, code, &status); CHECK(pipeline && status == VK_SUCCESS);
    auto base = Upload(native, Fixture(context)); CHECK(base && base->curveCount() == 2 && base->pointCount() == 6);
    CHECK(!pipeline->Begin(base, 0, &status) && !pipeline->Begin(base, 1, &status));
    bool admissionCalled = false;
    CHECK(!pipeline->Begin(base, 4, &status, [&] { admissionCalled = true; return false; }) && admissionCalled);
    uint32_t semantic = UINT32_MAX;
    auto four = Run(*pipeline, native, base, 4, &semantic);
    CHECK(four && semantic == 0 && four->succeeded() && four->curveCount() == 2 && four->pointCount() == 8 &&
        four->target() == 4 && four->inputOwner() == base && four->context() == context);
    Packet got;
    CHECK(Capture(four->outputs(), native, context, &got));
    float worst = 0;
    CHECK(CheckTarget4Oracle(got, &worst) && CheckVerbatim(got));
    std::printf("resampleVk target=4 oracle worst diff: %.3g\n", double(worst));
    CHECK(four->chunks().size() == 1 && four->chunks()[0].firstCurve == 0 && four->chunks()[0].curveCount == 2 &&
        four->chunks()[0].liveCount == 2 && four->chunks()[0].firstCv == 0 && four->chunks()[0].cvCount == 4 &&
        four->chunks()[0].tile == 5 && four->tiles().size() == 1 && four->tiles()[0].firstElement == 0 &&
        four->tiles()[0].elementCount == 2 && four->tiles()[0].firstPoint == 0 && four->tiles()[0].pointCount == 8 &&
        !four->tiles()[0].boundsValid);
    for (auto const& plane : four->outputs()) {
        if (plane.metadata.domain == UsdGenDeviceDomain::Point) CHECK(plane.metadata.elementCount == 8);
        if (plane.metadata.domain == UsdGenDeviceDomain::Primitive) CHECK(plane.metadata.elementCount == 2);
        if (plane.metadata.semantic == UsdGenDeviceChannelSemantic::ElementOffsets)
            CHECK(plane.metadata.elementCount == 3);
    }
    // Determinism: a second run reproduces every output byte bit-for-bit.
    auto rerun = Run(*pipeline, native, base, 4, &semantic);
    CHECK(rerun && semantic == 0 && rerun->succeeded());
    Packet again;
    CHECK(Capture(rerun->outputs(), native, context, &again) && again == got);
    rerun.reset();

    // Minimum (2) and odd (7) targets exercise the same twin comparison.
    // Hand-derived from the documented CUDA index formulas (t =
    // j*(n-1)/(T-1), lower/upper+fraction blend, nearest(t+0.5)): target 2
    // keeps each curve's first/last CV; target 7 is worked below.
    auto two = Run(*pipeline, native, base, 2, &semantic);
    CHECK(two && semantic == 0 && two->succeeded() && two->pointCount() == 4);
    Packet got2;
    CHECK(Capture(two->outputs(), native, context, &got2));
    CHECK(As<uint32_t>(got2.at("curveOffsets")) == std::vector<uint32_t>({0, 2, 4}));
    {
        auto px = As<float>(got2.at("points"));
        auto w = As<float>(got2.at("width"));
        auto ht = As<float>(got2.at("hairT"));
        auto f = As<float>(got2.at("f"));
        float const ex[4] = {0, 4, 10, 15};
        float const ew[4] = {1, 9, 2, 8};
        float const et[4] = {0, 1, 0, 1};
        float const ef[4] = {0, 1, 2, 5};
        CHECK(px.size() == 12 && w.size() == 4 && ht.size() == 4 && f.size() == 4);
        CHECK(ExactHairT(got2.at("hairT"), Bytes(et, 4), 2, "CPU"));
        for (size_t p = 0; p != 4; ++p)
            CHECK(Near(px[3 * p], ex[p]) && Near(w[p], ew[p]) && Near(f[p], ef[p]));
        CHECK(As<int32_t>(got2.at("i")) == std::vector<int32_t>({10, 11, 12, 15}));
    }
    auto seven = Run(*pipeline, native, base, 7, &semantic);
    CHECK(seven && semantic == 0 && seven->succeeded() && seven->pointCount() == 14);
    Packet got7;
    CHECK(Capture(seven->outputs(), native, context, &got7));
    CHECK(As<uint32_t>(got7.at("curveOffsets")) == std::vector<uint32_t>({0, 7, 14}));
    CHECK(seven->chunks()[0].firstCv == 0 && seven->chunks()[0].cvCount == 7);
    {
        auto px = As<float>(got7.at("points"));
        auto w = As<float>(got7.at("width"));
        auto ht = As<float>(got7.at("hairT"));
        auto f = As<float>(got7.at("f"));
        float const ex[14] = {0, 2.f / 3, 4.f / 3, 2, 8.f / 3, 10.f / 3, 4,
            10, 10.5f, 11, 12.5f, 14, 14.5f, 15};
        float const ew[14] = {1, 7.f / 3, 11.f / 3, 5, 19.f / 3, 23.f / 3, 9,
            2, 3, 4, 5, 6, 7, 8};
        float const et[14] = {0, 1.f / 6, 2.f / 6, 3.f / 6, 4.f / 6, 5.f / 6, 1,
            0, .125f, .25f, .5f, .75f, .875f, 1};
        float const ef[14] = {0, 1.f / 6, 2.f / 6, 3.f / 6, 4.f / 6, 5.f / 6, 1,
            2, 2.5f, 3, 3.5f, 4, 4.5f, 5};
        CHECK(px.size() == 42 && w.size() == 14 && ht.size() == 14 && f.size() == 14);
        CHECK(ExactHairT(got7.at("hairT"), Bytes(et, 14), 7, "CPU"));
        for (size_t p = 0; p != 14; ++p)
            CHECK(Near(px[3 * p], ex[p]) && Near(w[p], ew[p]) && Near(f[p], ef[p]));
        CHECK(As<int32_t>(got7.at("i")) ==
            std::vector<int32_t>({10, 10, 10, 11, 11, 11, 11, 12, 13, 13, 14, 14, 15, 15}));
    }

    std::map<uint32_t, Packet> targets{{2, got2}, {4, got}, {7, got7}};
    // Seven-, eleven- and sixty-three-interval divisions exercise rounding
    // that dyadic fractions and the existing short fixtures do not expose.
    for (uint32_t target : {3u, 8u, 12u, 64u}) {
        auto candidate = Run(*pipeline, native, base, target, &semantic);
        CHECK(candidate && semantic == 0 && candidate->succeeded() &&
            candidate->pointCount() == 2 * target);
        Packet packet;
        CHECK(Capture(candidate->outputs(), native, context, &packet));
        CHECK(ExactHairT(packet.at("hairT"), HairTOracle(target), target, "CPU"));
        targets.emplace(target, std::move(packet));
    }

    // Empty input owns only its required {0} offset, matching CUDA.
    VulkanSourceGenerationCreateInfo emptyInfo; emptyInfo.context = context;
    emptyInfo.source.totalCurves = 0; emptyInfo.source.totalCvs = 0;
    emptyInfo.source.topologyVersion = 31; emptyInfo.source.valueVersion = 32;
    emptyInfo.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic, UsdGenCurveLayoutBasis::BSpline,
        UsdGenCurveLayoutWrap::Pinned);
    auto emptyBase = Upload(native, std::move(emptyInfo));
    CHECK(emptyBase && emptyBase->curveCount() == 0 && emptyBase->pointCount() == 0);
    auto emptyOut = Run(*pipeline, native, emptyBase, 4, &semantic);
    CHECK(emptyOut && semantic == 0 && emptyOut->succeeded() && emptyOut->pointCount() == 0);
    Packet gotEmpty;
    CHECK(Capture(emptyOut->outputs(), native, context, &gotEmpty));
    CHECK(As<uint32_t>(gotEmpty.at("curveOffsets")) == std::vector<uint32_t>({0}));

    bool cudaCompared = false;
#ifdef USDGEN_ENABLE_CUDA
    int cudaDevices = 0;
    cudaError_t const deviceStatus = cudaGetDeviceCount(&cudaDevices);
    CHECK(deviceStatus == cudaSuccess || deviceStatus == cudaErrorNoDevice ||
        deviceStatus == cudaErrorInsufficientDriver);
    bool const cudaAvailable = deviceStatus == cudaSuccess && cudaDevices > 0;
    if (!cudaAvailable) std::puts("resampleVk: CUDA leg unavailable; twin comparison skipped");
    if (cudaAvailable) {
        CHECK(cudaSetDevice(0) == cudaSuccess);
        using namespace cudaTwin;
        CHECK(ConfigureCudaExecutionResources(0, {size_t{16} << 20, 0}));
        UsdGenExecutionResourceDevice const resourceKey{usdGenTestUtils::Backend("cuda"), 0};
        auto cudaResources = FindUsdGenExecutionResourcePool(resourceKey);
        auto cudaRetirement = GetOrCreateUsdGenExecutionRetirementService(resourceKey, {64});
        CHECK(cudaResources && cudaRetirement);
        auto const cudaBefore = cudaResources->Snapshot().usedBytes;
        cudaStream_t stream = nullptr;
        CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
        auto lifetime = std::make_shared<Lifetime>();
        auto& points = lifetime->points; auto& restPoints = lifetime->restPoints;
        auto& widths = lifetime->widths; auto& hairT = lifetime->hairT;
        auto& offsets = lifetime->offsets; auto& stableIds = lifetime->stableIds;
        auto& rootPrim = lifetime->rootPrim; auto& rootUVBytes = lifetime->rootUVBytes;
        CHECK(Upload(points, {{0, 0, 0}, {4, 0, 0}, {10, 0, 0}, {11, 0, 0}, {14, 0, 0}, {15, 0, 0}}));
        CHECK(Upload(restPoints, {{0, 1, 0}, {4, 1, 0}, {10, 1, 0}, {11, 1, 0}, {14, 1, 0}, {15, 1, 0}}));
        CHECK(Upload(widths, {1, 9, 2, 4, 6, 8}));
        CHECK(Upload(hairT, {0, 1, 0, .25f, .75f, 1}));
        CHECK(Upload(offsets, {0u, 2u, 6u}));
        CHECK(Upload(stableIds, {uint64_t(9), uint64_t(3)}));
        CHECK(Upload(rootPrim, {7, 8}));
        std::vector<::float2> hostRootUV{{.2f, .3f}, {.7f, .8f}};
        CHECK(rootUVBytes.reset(hostRootUV.size() * sizeof(::float2)) == cudaSuccess);
        CHECK(cudaMemcpy(rootUVBytes.data(), hostRootUV.data(), rootUVBytes.size(), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(Upload(lifetime->fBytes, {0, 1, 2, 3, 4, 5}));
        CHECK(Upload(lifetime->iBytes, {10, 11, 12, 13, 14, 15}));
        CHECK(Upload(lifetime->uBytes, {30, 31}));
        CHECK(Upload(lifetime->cBytes, {3.5f}));
        DeviceView<const ::float2> rootUV{reinterpret_cast<::float2 const*>(rootUVBytes.data()), 2};
        auto asBytes = [](auto const& buffer) {
            return DeviceView<const unsigned char>{reinterpret_cast<unsigned char const*>(buffer.data()),
                buffer.size() * sizeof(*buffer.data())};
        };
        float twinWorst = 0;
        for (auto const& targetPacket : targets) {
            uint32_t const target = targetPacket.first;
            Packet const& vulkan = targetPacket.second;
            CudaCurveResample resample;
            DeviceCurveGeometryView input{Read(points), Read(restPoints), Read(widths), Read(offsets),
                Read(stableIds), 2, 6};
            CHECK(resample.Apply(input, Read(hairT), Read(rootPrim), rootUV, int32_t(target), stream) ==
                CurveResampleStatus::Ok);
            CHECK(resample.Finish(stream) == CurveResampleStatus::Ok);
            auto const view = resample.view();
            CHECK(view.curveCount == 2 && view.pointCount == 2 * target);
            auto twinOffsets = Download(view.curveOffsets);
            auto twinIds = Download(view.stableIds);
            auto twinPoints = Download(view.points);
            auto twinRest = Download(view.restPoints);
            auto twinWidths = Download(view.widths);
            auto twinHairT = Download(resample.hairT());
            auto twinRootPrim = Download(resample.rootPrim());
            auto twinRootUV = Download(resample.rootUV());
            CHECK(As<uint32_t>(vulkan.at("curveOffsets")) == std::vector<uint32_t>(twinOffsets.begin(), twinOffsets.end()));
            CHECK(As<uint64_t>(vulkan.at("stableIds")) == std::vector<uint64_t>(twinIds.begin(), twinIds.end()));
            CHECK(As<int32_t>(vulkan.at("rootPrim")) == std::vector<int32_t>(twinRootPrim.begin(), twinRootPrim.end()));
            auto const& vkPoints = As<float>(vulkan.at("points"));
            auto const& vkRest = As<float>(vulkan.at("rest"));
            auto const& vkWidths = As<float>(vulkan.at("width"));
            auto const& vkHairT = As<float>(vulkan.at("hairT"));
            auto const& vkRootUV = As<float>(vulkan.at("rootUV"));
            CHECK(vkPoints.size() == twinPoints.size() * 3 && vkRest.size() == twinRest.size() * 3 &&
                vkWidths.size() == twinWidths.size() && vkHairT.size() == twinHairT.size() &&
                vkRootUV.size() == twinRootUV.size() * 2);
            CHECK(ExactHairT(vulkan.at("hairT"), Bytes(twinHairT.data(), twinHairT.size()), target, "CUDA"));
            for (size_t p = 0; p != twinPoints.size(); ++p) {
                float const diffs[9] = {std::fabs(vkPoints[3 * p] - twinPoints[p].x),
                    std::fabs(vkPoints[3 * p + 1] - twinPoints[p].y), std::fabs(vkPoints[3 * p + 2] - twinPoints[p].z),
                    std::fabs(vkRest[3 * p] - twinRest[p].x), std::fabs(vkRest[3 * p + 1] - twinRest[p].y),
                    std::fabs(vkRest[3 * p + 2] - twinRest[p].z), std::fabs(vkWidths[p] - twinWidths[p]),
                    std::fabs(vkRootUV[2 * (p / target)] - twinRootUV[p / target].x),
                    std::fabs(vkRootUV[2 * (p / target) + 1] - twinRootUV[p / target].y)};
                for (float d : diffs) {
                    twinWorst = std::max(twinWorst, d);
                    CHECK(d < 1e-5f);
                }
            }
            // Named-channel twin through the resample-mode topology transform.
            CudaNamedChannelTopologyCandidate named;
            std::vector<CudaNamedChannelTopologyCandidate::RawInputPlane> planes{
                {{TfToken("f"), UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Point, 6, 1, 4, true}, asBytes(lifetime->fBytes)},
                {{TfToken("i"), UsdGenDeviceValueType::Int32, UsdGenDeviceDomain::Point, 6, 1, 4, true}, asBytes(lifetime->iBytes)},
                {{TfToken("u"), UsdGenDeviceValueType::Int32, UsdGenDeviceDomain::Primitive, 2, 1, 4, true}, asBytes(lifetime->uBytes)},
                {{TfToken("c"), UsdGenDeviceValueType::Float32, UsdGenDeviceDomain::Groom, 1, 1, 4, true}, asBytes(lifetime->cBytes)}};
            std::string reason;
            CHECK(named.BeginRaw(input, view, std::move(planes), stream, CudaNamedChannelTopologyMode::Resample,
                lifetime, &reason) == cudaSuccess);
            Completion done;
            CHECK(named.FinishAsync(Callback, &done) == cudaSuccess);
            CHECK(cudaStreamSynchronize(stream) == cudaSuccess);
            CHECK(done.calls.load(std::memory_order_acquire) == 1 &&
                done.status.load(std::memory_order_acquire) == cudaSuccess);
            std::vector<CudaNamedChannelPlane> twinPlanes;
            CHECK(named.CommitPlanes(static_cast<cudaError_t>(done.status.load(std::memory_order_acquire)), &twinPlanes, &reason));
            CHECK(twinPlanes.size() == 4);
            for (auto& plane : twinPlanes) {
                CHECK(plane.bytes->waitOn(nullptr) == cudaSuccess);
                std::vector<uint8_t> bytes(plane.bytes->size());
                CHECK(cudaMemcpy(bytes.data(), plane.bytes->data(), bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess);
                auto it = vulkan.find(plane.metadata.name.GetString());
                CHECK(it != vulkan.end());
                if (plane.metadata.name == "f") {
                    auto tf = As<float>(bytes);
                    auto vf = As<float>(it->second);
                    CHECK(tf.size() == vf.size() && tf.size() == 2 * target);
                    for (size_t k = 0; k != tf.size(); ++k) {
                        twinWorst = std::max(twinWorst, std::fabs(vf[k] - tf[k]));
                        CHECK(Near(vf[k], tf[k]));
                    }
                } else CHECK(bytes == it->second);
            }
        }
        std::printf("resampleVk CUDA-twin worst diff: %.3g\n", double(twinWorst));
        // Empty twin: CUDA accepts the degenerate geometry with {0} offsets.
        {
            DeviceBuffer<uint32_t> emptyOffsets;
            CHECK(Upload(emptyOffsets, {0u}));
            DeviceCurveGeometryView empty{{}, {}, {}, Read(emptyOffsets), {}, 0, 0};
            CudaCurveResample resample;
            CHECK(resample.Apply(empty, {}, {}, {}, 4, stream) == CurveResampleStatus::Ok);
            CHECK(resample.Finish(stream) == CurveResampleStatus::Ok);
            CHECK(Download(resample.view().curveOffsets) == std::vector<uint32_t>({0}));
        }
        CHECK(cudaStreamDestroy(stream) == cudaSuccess);
        lifetime.reset();
        cudaRetirement->Drain();
        CHECK(cudaResources->Snapshot().usedBytes == cudaBefore);
        cudaCompared = true;
    }
#else
    std::puts("resampleVk: CUDA off; twin comparison skipped");
#endif

    four.reset(); two.reset(); seven.reset(); emptyOut.reset();
    emptyBase.reset(); base.reset(); pipeline.reset();
    CHECK(context->resources()->Snapshot().usedBytes == 0);
    std::puts(cudaCompared ? "Vulkan indexed-CV resample parity: PASS" :
        "Vulkan indexed-CV resample parity: SKIP (CUDA comparison unavailable)");
    return cudaCompared ? 0 : 77;
}
