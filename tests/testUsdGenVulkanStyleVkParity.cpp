// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanStyleVkParity — CUDA-vs-Vulkan equivalence gate for the
// device-only style primitives (gpu/styleOps.cu vs StyleVkPipeline).
//
// Every finished CudaStyleOps operator runs per op on both GPU lanes over a
// shared deterministic fixture (two ragged curves, the exact geometry from
// tests/testUsdGenCudaStyle.cu): WidthRamp (primitive broadcast, point
// domain, groom literal, groom data), Length (primitive, literal, point
// domain), and Grow (primitive + hairT, literal, point domain). Both legs
// are compared against host oracles transcribed from the CUDA test's own
// expectations, and against each other, within kTolerance.
//
// Tolerance contract: kTolerance (2e-5) is the Near() bound from
// tests/testUsdGenCudaStyle.cu, the CUDA twin's own contract. Bitwise
// equality is reported per scenario (expected: identical IEEE op order on
// both lanes, CUDA --fmad=false vs GLSL `precise`) but asserted only as
// information; the gate is the tolerance.
//
// Functional parity is covered alongside the numerics: both legs must map
// malformed offsets to InvalidArgument, non-finite inputs to NonFiniteInput,
// and bad values (negative widths, zero normals, out-of-range hairT) to
// InvalidValue; both must withhold publication on failure (CUDA leaves the
// caller buffer untouched, Vulkan exposes no output plane); host-invalid
// calls (field count mismatch, empty-shape mismatch, non-finite literal)
// must fail synchronously on both legs; empty geometry is a valid no-op on
// both; Noise reports NotSupported on both; and a rerun reproduces output
// bit-for-bit on both.
//
// Missing backends return 77 after the available lane runs its host oracles.
// Initialization and execution failures fail the test; success requires both lanes.

#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/styleVkPipeline.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "usdGenCurves/cuda/styleOps.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <chrono>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

int g_failures = 0;
void Check(bool ok, char const* what) {
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    } else {
        std::printf("PASS: %s\n", what);
    }
}

// Near() bound from tests/testUsdGenCudaStyle.cu (the CUDA twin's contract).
constexpr float kTolerance = 2.0e-5f;

// Backend-neutral outcome for cross-leg comparison.
enum class Outcome {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidValue,
    NotSupported,
    BackendError,
};
char const* Name(Outcome o) {
    switch (o) {
        case Outcome::Ok: return "Ok";
        case Outcome::InvalidArgument: return "InvalidArgument";
        case Outcome::NonFiniteInput: return "NonFiniteInput";
        case Outcome::InvalidValue: return "InvalidValue";
        case Outcome::NotSupported: return "NotSupported";
        case Outcome::BackendError: return "BackendError";
    }
    return "?";
}

bool WithinTolerance(std::vector<float> const& got,
                     std::vector<float> const& expected, float tolerance,
                     float* maxDiff) {
    if (got.size() != expected.size()) return false;
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i])) return false;
        worst = std::max(worst, std::fabs(got[i] - expected[i]));
    }
    if (maxDiff) *maxDiff = worst;
    return worst <= tolerance;
}

bool BitEqual(std::vector<float> const& a, std::vector<float> const& b) {
    return a.size() == b.size() &&
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::vector<uint32_t> LoadSpirv(char const* path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4) return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}

// Shared deterministic fixture: the exact geometry from
// tests/testUsdGenCudaStyle.cu (two ragged curves, 2 + 3 CVs).
struct Fixture {
    std::vector<float> points{0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 2};
    std::vector<uint32_t> offsets{0, 2, 5};
    std::vector<float> roots{0, 0, 0, 1, 0, 0};
    std::vector<float> normals{0, 2, 0, 0, 0, 3};
    std::vector<float> primitive{2.0f, 4.0f};
    std::vector<float> point{0.0f, 0.25f, 0.0f, 0.5f, 1.0f};
    std::vector<float> groom{3.0f};
    static constexpr uint32_t kCurves = 2, kPoints = 5;
};

struct LegResult {
    Outcome outcome = Outcome::BackendError;
    std::vector<float> values; // scalar or packed-float3 plane
    bool published = false;    // output observable (CUDA buffer written /
                               // Vulkan output plane exposed)
};

// ---- Vulkan leg ----

struct VulkanLeg {
    std::shared_ptr<NativeOwner> native;
    std::shared_ptr<DeviceContext> context;
    std::shared_ptr<StyleVkPipeline> pipeline;
};

std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const& c, void const* data,
    VkDeviceSize bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = ChargedBuffer::Create(c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b) return {};
    if (bytes) {
        void* p = nullptr;
        if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) != VK_SUCCESS)
            return {};
        std::memcpy(p, data, bytes);
        vkUnmapMemory(c->device(), b->memory());
    }
    return b;
}

template <class T>
std::shared_ptr<const ChargedBuffer> UploadVec(
    std::shared_ptr<DeviceContext> const& c, std::vector<T> const& v) {
    return Upload(c, v.data(), v.size() * sizeof(T));
}

bool SetupVulkan(VulkanLeg* leg, std::vector<uint32_t> const& widthRampSpv,
                 std::vector<uint32_t> const& lengthSpv,
                 std::vector<uint32_t> const& growSpv, bool* unavailable) {
    *unavailable = false;
    auto native = CreateTimelineNative(unavailable);
    if (!native) return false;
    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8042;
    ci.nativeLifetime = native; ci.timelineSemaphoreEnabled = true;
    ci.resources = {size_t{32} << 20, 0};
    leg->native = native;
    leg->context = DeviceContext::Create(ci);
    if (!leg->context) return false;
    VkResult status = VK_SUCCESS;
    leg->pipeline = StyleVkPipeline::Create(leg->context, widthRampSpv,
                                            lengthSpv, growSpv, &status);
    return leg->pipeline && status == VK_SUCCESS;
}

Outcome MapVk(StyleVkStatus s) {
    switch (s) {
        case StyleVkStatus::Ok: return Outcome::Ok;
        case StyleVkStatus::InvalidArgument: return Outcome::InvalidArgument;
        case StyleVkStatus::NonFiniteInput: return Outcome::NonFiniteInput;
        case StyleVkStatus::InvalidValue: return Outcome::InvalidValue;
        case StyleVkStatus::NotSupported: return Outcome::NotSupported;
        case StyleVkStatus::VulkanError: return Outcome::BackendError;
    }
    return Outcome::BackendError;
}

// Runs one candidate to proof and reads back `wantFloats` floats.
bool ProveCandidate(VulkanLeg const& leg,
                    std::unique_ptr<StyleVkPipeline::Candidate>* candidate,
                    size_t wantFloats, LegResult* out) {
    auto& c = *candidate;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    if (vkResetFences(leg.native->device, 1, &leg.native->fence) != VK_SUCCESS)
        return false;
    if (vkQueueSubmit(leg.native->queue, 1, &submit, leg.native->fence) !=
            VK_SUCCESS ||
        vkWaitForFences(leg.native->device, 1, &leg.native->fence, VK_TRUE,
                        10000000000ull) != VK_SUCCESS)
        return false;
    StyleVkSemantic semantic = StyleVkSemantic::Ok;
    VkResult poll = c->Poll(&semantic);
    if (poll == VK_NOT_READY) {
        // Single queue shared with the proof submit above; a second proof
        // cannot be needed, but poll honestly once more after a wait.
        if (vkResetFences(leg.native->device, 1, &leg.native->fence) !=
                VK_SUCCESS ||
            vkQueueSubmit(leg.native->queue, 1, &submit, leg.native->fence) !=
                VK_SUCCESS ||
            vkWaitForFences(leg.native->device, 1, &leg.native->fence, VK_TRUE,
                            10000000000ull) != VK_SUCCESS)
            return false;
        poll = c->Poll(&semantic);
    }
    if (poll != VK_SUCCESS) return false;
    out->outcome = MapVk(StyleVkStatusForSemantic(semantic));
    auto output = c->output();
    out->published = bool(output.values);
    if (!output.values) {
        out->values.clear();
        return true;
    }
    std::vector<uint8_t> bytes;
    if (!ReadVulkanBytes(leg.native, leg.context, output.values->buffer(),
                         wantFloats * sizeof(float), output.values, &bytes))
        return false;
    out->values.resize(wantFloats);
    if (wantFloats)
        std::memcpy(out->values.data(), bytes.data(), bytes.size());
    return true;
}

StyleVkPipeline::ScalarField VkLiteral(float value) {
    StyleVkPipeline::ScalarField f;
    f.literal = value;
    f.domain = StyleVkDomain::Groom;
    return f;
}

StyleVkPipeline::ScalarField VkField(
    std::shared_ptr<const ChargedBuffer> const& data, uint32_t count,
    uint32_t domain) {
    StyleVkPipeline::ScalarField f;
    f.domain = domain;
    f.data = data;
    f.count = count;
    return f;
}

struct VkPlanes {
    std::shared_ptr<const ChargedBuffer> points, offsets, roots, normals,
        primitive, point, groom, hairT;
};

bool UploadFixture(VulkanLeg const& leg, Fixture const& fx, VkPlanes* planes) {
    planes->points = UploadVec(leg.context, fx.points);
    planes->offsets = UploadVec(leg.context, fx.offsets);
    planes->roots = UploadVec(leg.context, fx.roots);
    planes->normals = UploadVec(leg.context, fx.normals);
    planes->primitive = UploadVec(leg.context, fx.primitive);
    planes->point = UploadVec(leg.context, fx.point);
    planes->groom = UploadVec(leg.context, fx.groom);
    planes->hairT = planes->point; // the point field doubles as hairT
    return planes->points && planes->offsets && planes->roots &&
        planes->normals && planes->primitive && planes->point && planes->groom;
}

bool VkWidthRamp(VulkanLeg const& leg, VkPlanes const& planes,
                 StyleVkPipeline::ScalarField root,
                 StyleVkPipeline::ScalarField tip,
                 std::shared_ptr<const ChargedBuffer> hairT, uint32_t curves,
                 uint32_t points, LegResult* out) {
    StyleVkPipeline::WidthRampInfo info;
    info.offsets = planes.offsets;
    info.curveCount = curves;
    info.pointCount = points;
    info.root = std::move(root);
    info.tip = std::move(tip);
    info.hairT = std::move(hairT);
    VkResult result = VK_SUCCESS;
    StyleVkStatus status = StyleVkStatus::Ok;
    auto c = leg.pipeline->BeginWidthRamp(std::move(info), &result, &status);
    if (!c) {
        out->outcome = MapVk(status);
        out->published = false;
        out->values.clear();
        return status == StyleVkStatus::InvalidArgument ||
            status == StyleVkStatus::NonFiniteInput;
    }
    return ProveCandidate(leg, &c, points, out);
}

bool VkLength(VulkanLeg const& leg, VkPlanes const& planes,
              StyleVkPipeline::ScalarField scale, uint32_t curves,
              uint32_t points, LegResult* out) {
    StyleVkPipeline::LengthInfo info;
    info.points = planes.points;
    info.offsets = planes.offsets;
    info.curveCount = curves;
    info.pointCount = points;
    info.scale = std::move(scale);
    VkResult result = VK_SUCCESS;
    StyleVkStatus status = StyleVkStatus::Ok;
    auto c = leg.pipeline->BeginLength(std::move(info), &result, &status);
    if (!c) {
        out->outcome = MapVk(status);
        out->published = false;
        out->values.clear();
        return status == StyleVkStatus::InvalidArgument ||
            status == StyleVkStatus::NonFiniteInput;
    }
    return ProveCandidate(leg, &c, size_t{points} * 3, out);
}

bool VkGrow(VulkanLeg const& leg, VkPlanes const& planes,
            StyleVkPipeline::ScalarField length,
            std::shared_ptr<const ChargedBuffer> hairT, uint32_t curves,
            uint32_t points, LegResult* out) {
    StyleVkPipeline::GrowInfo info;
    info.offsets = planes.offsets;
    info.roots = planes.roots;
    info.normals = planes.normals;
    info.curveCount = curves;
    info.pointCount = points;
    info.length = std::move(length);
    info.hairT = std::move(hairT);
    VkResult result = VK_SUCCESS;
    StyleVkStatus status = StyleVkStatus::Ok;
    auto c = leg.pipeline->BeginGrow(std::move(info), &result, &status);
    if (!c) {
        out->outcome = MapVk(status);
        out->published = false;
        out->values.clear();
        return status == StyleVkStatus::InvalidArgument ||
            status == StyleVkStatus::NonFiniteInput;
    }
    return ProveCandidate(leg, &c, size_t{points} * 3, out);
}

#ifdef USDGEN_ENABLE_CUDA
// ---- CUDA leg ----

namespace cudaLeg {
using namespace usdGen::gpu;

template <class T>
bool Upload(DeviceBuffer<T>& device, std::vector<T> const& host,
            cudaStream_t stream) {
    return (device.size() == host.size() ||
            device.reset(host.size()) == cudaSuccess) &&
        (!host.size() ||
         cudaMemcpyAsync(device.data(), host.data(),
                         host.size() * sizeof(T), cudaMemcpyHostToDevice,
                         stream) == cudaSuccess);
}

template <class T>
bool Download(DeviceBuffer<T> const& device, std::vector<T>* host,
              cudaStream_t stream) {
    host->resize(device.size());
    return (!host->size() ||
            cudaMemcpyAsync(host->data(), device.data(),
                            host->size() * sizeof(T), cudaMemcpyDeviceToHost,
                            stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}

Outcome MapCuda(StyleStatus s) {
    switch (s) {
        case StyleStatus::Ok: return Outcome::Ok;
        case StyleStatus::InvalidArgument: return Outcome::InvalidArgument;
        case StyleStatus::NonFiniteInput: return Outcome::NonFiniteInput;
        case StyleStatus::InvalidValue: return Outcome::InvalidValue;
        case StyleStatus::NotSupported: return Outcome::NotSupported;
        case StyleStatus::CudaError: return Outcome::BackendError;
    }
    return Outcome::BackendError;
}

float3 V3(float x, float y, float z) { return make_float3(x, y, z); }

std::vector<float3> ToFloat3(std::vector<float> const& flat) {
    std::vector<float3> v(flat.size() / 3);
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = V3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
    return v;
}

std::vector<float> ToFlat(std::vector<float3> const& v) {
    std::vector<float> flat(v.size() * 3);
    for (size_t i = 0; i < v.size(); ++i) {
        flat[i * 3] = v[i].x;
        flat[i * 3 + 1] = v[i].y;
        flat[i * 3 + 2] = v[i].z;
    }
    return flat;
}

struct Context {
    cudaStream_t producer = nullptr, consumer = nullptr;
    DeviceBuffer<float3> points, roots, normals, vectorOutput;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<float> primitive, point, groom, widths, hairT;
};

bool Setup(Context* ctx, Fixture const& fx, bool* unavailable = nullptr) {
    if (unavailable) *unavailable = false;
    int devices = 0;
    cudaError_t const deviceStatus = cudaGetDeviceCount(&devices);
    if (deviceStatus == cudaErrorNoDevice || deviceStatus == cudaErrorInsufficientDriver ||
        (deviceStatus == cudaSuccess && devices == 0)) {
        if (unavailable) *unavailable = true;
        return false;
    }
    if (deviceStatus != cudaSuccess) return false;
    cudaError_t set = cudaSetDevice(0);
    cudaError_t prod = set == cudaSuccess ? cudaStreamCreateWithFlags(
        &ctx->producer, cudaStreamNonBlocking) : set;
    cudaError_t cons = prod == cudaSuccess ? cudaStreamCreateWithFlags(
        &ctx->consumer, cudaStreamNonBlocking) : prod;
    if (set != cudaSuccess || prod != cudaSuccess || cons != cudaSuccess) {
        if (ctx->producer) cudaStreamDestroy(ctx->producer);
        if (ctx->consumer) cudaStreamDestroy(ctx->consumer);
        ctx->producer = ctx->consumer = nullptr;
        return false;
    }
    if (!Upload(ctx->points, ToFloat3(fx.points), ctx->producer) ||
        !Upload(ctx->offsets, fx.offsets, ctx->producer) ||
        !Upload(ctx->roots, ToFloat3(fx.roots), ctx->producer) ||
        !Upload(ctx->normals, ToFloat3(fx.normals), ctx->producer) ||
        !Upload(ctx->primitive, fx.primitive, ctx->producer) ||
        !Upload(ctx->point, fx.point, ctx->producer) ||
        !Upload(ctx->groom, fx.groom, ctx->producer) ||
        !Upload(ctx->hairT, fx.point, ctx->producer) ||
        ctx->widths.reset(Fixture::kPoints) != cudaSuccess ||
        ctx->vectorOutput.reset(Fixture::kPoints) != cudaSuccess ||
        cudaStreamSynchronize(ctx->producer) != cudaSuccess)
        return false;
    return true;
}

void Teardown(Context* ctx) {
    if (ctx->producer) cudaStreamDestroy(ctx->producer);
    if (ctx->consumer) cudaStreamDestroy(ctx->consumer);
}

DeviceCurveGeometryView Geometry(Context const& ctx, uint32_t curves,
                                 uint32_t points) {
    DeviceCurveGeometryView geometry{
        {ctx.points.data(), points}, {}, {}, {ctx.offsets.data(), curves + 1},
        {}, curves, points};
    return geometry;
}

ScalarField CudaField(DeviceBuffer<float> const& buffer,
                       expr::Domain domain) {
    return ScalarField::Device({buffer.data(), buffer.size()}, domain);
}

// hFile==nullptr selects the no-hairT overload; otherwise the hairT overload.
bool RunWidthRamp(Context* ctx, Fixture const& fx, ScalarField root,
                  ScalarField tip, bool useHairT, LegResult* out) {
    (void)fx;
    CudaStyleOps style;
    StyleStatus queued;
    if (useHairT)
        queued = style.WidthRamp(Geometry(*ctx, Fixture::kCurves,
                                          Fixture::kPoints),
                                 root, tip, {ctx->widths.data(), ctx->widths.size()},
                                 ctx->producer, {ctx->hairT.data(), ctx->hairT.size()});
    else
        queued = style.WidthRamp(Geometry(*ctx, Fixture::kCurves,
                                          Fixture::kPoints),
                                 root, tip, {ctx->widths.data(), ctx->widths.size()},
                                 ctx->producer);
    if (queued != StyleStatus::Ok) {
        out->outcome = MapCuda(queued);
        out->published = false;
        out->values.clear();
        return true;
    }
    StyleStatus done = style.Finish(ctx->consumer);
    out->outcome = MapCuda(done);
    out->published = (done == StyleStatus::Ok);
    if (done != StyleStatus::Ok) {
        out->values.clear();
        return true;
    }
    std::vector<float> host;
    if (!Download(ctx->widths, &host, ctx->consumer)) return false;
    out->values = host;
    return true;
}

bool RunLength(Context* ctx, ScalarField scale, LegResult* out) {
    CudaStyleOps style;
    StyleStatus queued = style.Length(
        Geometry(*ctx, Fixture::kCurves, Fixture::kPoints), scale,
        {ctx->vectorOutput.data(), ctx->vectorOutput.size()}, ctx->producer);
    if (queued != StyleStatus::Ok) {
        out->outcome = MapCuda(queued);
        out->published = false;
        out->values.clear();
        return true;
    }
    StyleStatus done = style.Finish(ctx->consumer);
    out->outcome = MapCuda(done);
    out->published = (done == StyleStatus::Ok);
    if (done != StyleStatus::Ok) {
        out->values.clear();
        return true;
    }
    std::vector<float3> host;
    if (!Download(ctx->vectorOutput, &host, ctx->consumer)) return false;
    out->values = ToFlat(host);
    return true;
}

bool RunGrow(Context* ctx, ScalarField length, bool useHairT,
             LegResult* out) {
    CudaStyleOps style;
    DeviceView<const float3> roots{ctx->roots.data(), Fixture::kCurves};
    DeviceView<const float3> normals{ctx->normals.data(), Fixture::kCurves};
    DeviceView<float3> output{ctx->vectorOutput.data(),
                              ctx->vectorOutput.size()};
    StyleStatus queued;
    if (useHairT)
        queued = style.Grow(Geometry(*ctx, Fixture::kCurves, Fixture::kPoints),
                            roots, normals, length,
                            {ctx->hairT.data(), ctx->hairT.size()}, output,
                            ctx->producer);
    else
        queued = style.Grow(Geometry(*ctx, Fixture::kCurves, Fixture::kPoints),
                            roots, normals, length, output, ctx->producer);
    if (queued != StyleStatus::Ok) {
        out->outcome = MapCuda(queued);
        out->published = false;
        out->values.clear();
        return true;
    }
    StyleStatus done = style.Finish(ctx->consumer);
    out->outcome = MapCuda(done);
    out->published = (done == StyleStatus::Ok);
    if (done != StyleStatus::Ok) {
        out->values.clear();
        return true;
    }
    std::vector<float3> host;
    if (!Download(ctx->vectorOutput, &host, ctx->consumer)) return false;
    out->values = ToFlat(host);
    return true;
}

} // namespace cudaLeg
#endif // USDGEN_ENABLE_CUDA

// ---- Scenario checks ----

void CheckScenario(char const* name, bool haveVulkan, LegResult const& vk,
                   bool haveCuda, LegResult const& cuda,
                   std::vector<float> const& oracle, Outcome want) {
    char label[256];
    char const* legs = haveVulkan && haveCuda ? "both legs"
        : haveVulkan                          ? "vulkan leg"
                                              : "cuda leg";
    std::snprintf(label, sizeof(label), "%s: %s reports %s", name, legs,
                  Name(want));
    bool outcomes = true;
    if (haveVulkan) outcomes = outcomes && vk.outcome == want;
    if (haveCuda) outcomes = outcomes && cuda.outcome == want;
    if (haveVulkan && haveCuda) outcomes = outcomes && vk.outcome == cuda.outcome;
    Check(outcomes, label);
    if (want != Outcome::Ok) {
        std::snprintf(label, sizeof(label),
                      "%s: failing op publishes nothing", name);
        bool withheld = true;
        if (haveVulkan) withheld = withheld && !vk.published;
        if (haveCuda) withheld = withheld && !cuda.published;
        Check(withheld, label);
        if (haveVulkan)
            std::printf("      vulkan outcome %s published %d\n",
                        Name(vk.outcome), int(vk.published));
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda)
            std::printf("      cuda outcome %s published %d\n",
                        Name(cuda.outcome), int(cuda.published));
#endif
        return;
    }
    float worst = 0.0f;
    if (haveVulkan) {
        bool near =
            WithinTolerance(vk.values, oracle, kTolerance, &worst);
        std::printf("%s vulkan vs oracle: max diff %.6g (tol %.6g)%s\n", name,
                    worst, double(kTolerance),
                    BitEqual(vk.values, oracle) ? " [bitwise]" : "");
        std::snprintf(label, sizeof(label), "%s: vulkan matches oracle", name);
        Check(near, label);
    }
    if (haveCuda) {
        bool near =
            WithinTolerance(cuda.values, oracle, kTolerance, &worst);
        std::printf("%s cuda vs oracle: max diff %.6g (tol %.6g)%s\n", name,
                    worst, double(kTolerance),
                    BitEqual(cuda.values, oracle) ? " [bitwise]" : "");
        std::snprintf(label, sizeof(label), "%s: cuda matches oracle", name);
        Check(near, label);
    }
    if (haveVulkan && haveCuda) {
        bool near =
            WithinTolerance(vk.values, cuda.values, kTolerance, &worst);
        std::printf("%s vulkan vs cuda: max diff %.6g (tol %.6g)%s\n", name,
                    worst, double(kTolerance),
                    BitEqual(vk.values, cuda.values) ? " [bitwise]" : "");
        std::snprintf(label, sizeof(label), "%s: vulkan matches cuda", name);
        Check(near, label);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
                     "usage: %s styleVkWidthRamp.spv styleVkLength.spv "
                     "styleVkGrow.spv\n",
                     argv[0]);
        return 1;
    }
    auto widthRampSpv = LoadSpirv(argv[1]);
    auto lengthSpv = LoadSpirv(argv[2]);
    auto growSpv = LoadSpirv(argv[3]);
    if (widthRampSpv.empty() || lengthSpv.empty() || growSpv.empty()) {
        std::fprintf(stderr, "FAIL: cannot load SPIR-V modules\n");
        return 1;
    }
    Fixture fx;

    VulkanLeg leg;
    bool vulkanUnavailable = false;
    bool haveVulkan =
        SetupVulkan(&leg, widthRampSpv, lengthSpv, growSpv, &vulkanUnavailable);

    VkPlanes planes;
    if (haveVulkan && !UploadFixture(leg, fx, &planes)) haveVulkan = false;

#ifdef USDGEN_ENABLE_CUDA
    cudaLeg::Context cuda;
    bool cudaUnavailable = false;
    bool haveCuda = cudaLeg::Setup(&cuda, fx, &cudaUnavailable);
#else
    bool haveCuda = false;
    bool cudaUnavailable = true;
    std::printf("cuda leg not compiled (USDGEN_ENABLE_CUDA off)\n");
#endif

    Check(haveVulkan || vulkanUnavailable, "Vulkan bring-up and upload succeed");
    Check(haveCuda || cudaUnavailable, "CUDA bring-up and upload succeed");
    if (g_failures) return 1;
    if (!haveVulkan && !haveCuda) {
        std::printf("testUsdGenVulkanStyleVkParity: no GPU backend available "
                    "(vulkan %s, cuda %s), skipping\n",
                    vulkanUnavailable ? "unavailable" : "failed",
                    cudaUnavailable ? "unavailable" : "failed");
        return 77;
    }
    if (haveVulkan != haveCuda)
        std::printf("styleVk parity: cross-backend comparisons skipped "
                    "(%s only)\n",
                    haveVulkan ? "vulkan" : "cuda");

    LegResult vk, cudaResult;
    bool ran = true;

    // W1: WidthRamp primitive broadcast (CUDA test's first scenario).
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes, VkLiteral(1.0f),
                                 VkField(planes.primitive, 2,
                                         StyleVkDomain::Primitive),
                                 nullptr, 2, 5, &vk);
    LegResult vkW1 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx, cudaLeg::ScalarField::Literal(1.0f),
                         cudaLeg::CudaField(cuda.primitive,
                                            expr::Domain::Primitive),
                         false, &cudaResult);
#endif
    LegResult cudaW1 = cudaResult;
    CheckScenario("W1 widthRamp/primitive", haveVulkan, vkW1, haveCuda,
                  cudaW1, {1.0f, 2.0f, 1.0f, 2.5f, 4.0f}, Outcome::Ok);

    // W2: WidthRamp point-domain root (CUDA test's second scenario).
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes,
                                 VkField(planes.point, 5,
                                         StyleVkDomain::Point),
                                 VkLiteral(10.0f), nullptr, 2, 5, &vk);
    LegResult vkW2 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx,
                         cudaLeg::CudaField(cuda.point, expr::Domain::Point),
                         cudaLeg::ScalarField::Literal(10.0f), false,
                         &cudaResult);
#endif
    LegResult cudaW2 = cudaResult;
    CheckScenario("W2 widthRamp/point", haveVulkan, vkW2, haveCuda, cudaW2,
                  {0.0f, 10.0f, 0.0f, 5.25f, 10.0f}, Outcome::Ok);

    // W3: WidthRamp groom literals, index-interpolated t.
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes, VkLiteral(1.0f),
                                 VkLiteral(2.0f), nullptr, 2, 5, &vk);
    LegResult vkW3 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx, cudaLeg::ScalarField::Literal(1.0f),
                         cudaLeg::ScalarField::Literal(2.0f), false,
                         &cudaResult);
#endif
    LegResult cudaW3 = cudaResult;
    CheckScenario("W3 widthRamp/literals", haveVulkan, vkW3, haveCuda,
                  cudaW3, {1.0f, 2.0f, 1.0f, 1.5f, 2.0f}, Outcome::Ok);

    // W4: WidthRamp groom data plane plus primitive tip.
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes,
                                 VkField(planes.groom, 1,
                                         StyleVkDomain::Groom),
                                 VkField(planes.primitive, 2,
                                         StyleVkDomain::Primitive),
                                 nullptr, 2, 5, &vk);
    LegResult vkW4 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx,
                         cudaLeg::CudaField(cuda.groom, expr::Domain::Groom),
                         cudaLeg::CudaField(cuda.primitive,
                                            expr::Domain::Primitive),
                         false, &cudaResult);
#endif
    LegResult cudaW4 = cudaResult;
    CheckScenario("W4 widthRamp/groom-data", haveVulkan, vkW4, haveCuda,
                  cudaW4, {3.0f, 2.0f, 3.0f, 3.5f, 4.0f}, Outcome::Ok);

    // W5: WidthRamp with supplied hairT (point field as t).
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes, VkLiteral(0.0f),
                                 VkLiteral(8.0f), planes.hairT, 2, 5, &vk);
    LegResult vkW5 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx, cudaLeg::ScalarField::Literal(0.0f),
                         cudaLeg::ScalarField::Literal(8.0f), true,
                         &cudaResult);
#endif
    LegResult cudaW5 = cudaResult;
    CheckScenario("W5 widthRamp/hairT", haveVulkan, vkW5, haveCuda, cudaW5,
                  {0.0f, 2.0f, 0.0f, 4.0f, 8.0f}, Outcome::Ok);

    // L1: Length primitive scale (CUDA test's third scenario).
    if (haveVulkan)
        ran = ran && VkLength(leg, planes,
                              VkField(planes.primitive, 2,
                                      StyleVkDomain::Primitive),
                              2, 5, &vk);
    LegResult vkL1 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunLength(
                         &cuda,
                         cudaLeg::CudaField(cuda.primitive,
                                            expr::Domain::Primitive),
                         &cudaResult);
#endif
    LegResult cudaL1 = cudaResult;
    CheckScenario("L1 length/primitive", haveVulkan, vkL1, haveCuda, cudaL1,
                  {0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 0, 4, 1, 0, 8},
                  Outcome::Ok);

    // L2: Length literal scale 1 (expression is still evaluated, like CUDA).
    if (haveVulkan)
        ran = ran && VkLength(leg, planes, VkLiteral(1.0f), 2, 5, &vk);
    LegResult vkL2 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunLength(
                         &cuda, cudaLeg::ScalarField::Literal(1.0f),
                         &cudaResult);
#endif
    LegResult cudaL2 = cudaResult;
    CheckScenario("L2 length/literal-1", haveVulkan, vkL2, haveCuda, cudaL2,
                  fx.points, Outcome::Ok);

    // L3: Length point-domain scale.
    if (haveVulkan)
        ran = ran && VkLength(leg, planes,
                              VkField(planes.point, 5, StyleVkDomain::Point),
                              2, 5, &vk);
    LegResult vkL3 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunLength(
                         &cuda,
                         cudaLeg::CudaField(cuda.point, expr::Domain::Point),
                         &cudaResult);
#endif
    LegResult cudaL3 = cudaResult;
    CheckScenario("L3 length/point", haveVulkan, vkL3, haveCuda, cudaL3,
                  {0, 0, 0, 0, 0, 0.25f, 1, 0, 0, 1, 0, 0.5f, 1, 0, 2},
                  Outcome::Ok);

    // G1: Grow primitive length with supplied hairT (CUDA test scenario).
    if (haveVulkan)
        ran = ran && VkGrow(leg, planes,
                            VkField(planes.primitive, 2,
                                    StyleVkDomain::Primitive),
                            planes.hairT, 2, 5, &vk);
    LegResult vkG1 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunGrow(
                         &cuda,
                         cudaLeg::CudaField(cuda.primitive,
                                            expr::Domain::Primitive),
                         true, &cudaResult);
#endif
    LegResult cudaG1 = cudaResult;
    CheckScenario("G1 grow/primitive-hairT", haveVulkan, vkG1, haveCuda,
                  cudaG1,
                  {0, 0, 0, 0, 0.5f, 0, 1, 0, 0, 1, 0, 2, 1, 0, 4},
                  Outcome::Ok);

    // G2: Grow literal length, index-interpolated t.
    if (haveVulkan)
        ran = ran && VkGrow(leg, planes, VkLiteral(2.0f), nullptr, 2, 5,
                            &vk);
    LegResult vkG2 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunGrow(
                         &cuda, cudaLeg::ScalarField::Literal(2.0f), false,
                         &cudaResult);
#endif
    LegResult cudaG2 = cudaResult;
    CheckScenario("G2 grow/literal", haveVulkan, vkG2, haveCuda, cudaG2,
                  {0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2},
                  Outcome::Ok);

    // G3: Grow point-domain length, index-interpolated t.
    if (haveVulkan)
        ran = ran && VkGrow(leg, planes,
                            VkField(planes.point, 5, StyleVkDomain::Point),
                            nullptr, 2, 5, &vk);
    LegResult vkG3 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunGrow(
                         &cuda,
                         cudaLeg::CudaField(cuda.point, expr::Domain::Point),
                         false, &cudaResult);
#endif
    LegResult cudaG3 = cudaResult;
    CheckScenario("G3 grow/point", haveVulkan, vkG3, haveCuda, cudaG3,
                  {0, 0, 0, 0, 0.25f, 0, 1, 0, 0, 1, 0, 0.25f, 1, 0, 1},
                  Outcome::Ok);

    // F1/F2: malformed offsets diagnose InvalidArgument on both legs.
    {
        Fixture bad = fx;
        bad.offsets = {0u, 2u, 9u};
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF1 = true;
        if (haveVulkan) {
            ranF1 = ranF1 && UploadFixture(leg, bad, &badPlanes) &&
                VkWidthRamp(leg, badPlanes, VkLiteral(1.0f), VkLiteral(2.0f),
                            nullptr, 2, 5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF1 = ranF1 && cudaLeg::RunWidthRamp(
                                     &badCtx, bad,
                                     cudaLeg::ScalarField::Literal(1.0f),
                                     cudaLeg::ScalarField::Literal(2.0f),
                                     false, &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF1 = false;
            }
        }
#endif
        ran = ran && ranF1;
        CheckScenario("F1 widthRamp/bad-offsets-end", haveVulkan, badVk,
                      haveCuda, badCuda, {}, Outcome::InvalidArgument);
    }
    {
        Fixture bad = fx;
        bad.offsets = {1u, 2u, 5u};
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF2 = true;
        if (haveVulkan) {
            ranF2 = ranF2 && UploadFixture(leg, bad, &badPlanes) &&
                VkLength(leg, badPlanes, VkLiteral(1.0f), 2, 5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF2 = ranF2 && cudaLeg::RunLength(
                                     &badCtx,
                                     cudaLeg::ScalarField::Literal(1.0f),
                                     &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF2 = false;
            }
        }
#endif
        ran = ran && ranF2;
        CheckScenario("F2 length/bad-offsets-base", haveVulkan, badVk,
                      haveCuda, badCuda, {}, Outcome::InvalidArgument);
    }

    // F3: non-finite field data diagnoses NonFiniteInput on both legs.
    {
        Fixture bad = fx;
        bad.primitive = {std::numeric_limits<float>::quiet_NaN(), 4.0f};
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF3 = true;
        if (haveVulkan) {
            ranF3 = ranF3 && UploadFixture(leg, bad, &badPlanes) &&
                VkWidthRamp(leg, badPlanes, VkLiteral(1.0f),
                            VkField(badPlanes.primitive, 2,
                                    StyleVkDomain::Primitive),
                            nullptr, 2, 5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF3 = ranF3 && cudaLeg::RunWidthRamp(
                                     &badCtx, bad,
                                     cudaLeg::ScalarField::Literal(1.0f),
                                     cudaLeg::CudaField(badCtx.primitive,
                                                       expr::Domain::Primitive),
                                     false, &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF3 = false;
            }
        }
#endif
        ran = ran && ranF3;
        CheckScenario("F3 widthRamp/nan-field", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::NonFiniteInput);
    }

    // F4: negative widths diagnose InvalidValue on both legs.
    if (haveVulkan)
        ran = ran && VkWidthRamp(leg, planes, VkLiteral(-1.0f),
                                 VkLiteral(2.0f), nullptr, 2, 5, &vk);
    LegResult vkF4 = vk;
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        ran = ran && cudaLeg::RunWidthRamp(
                         &cuda, fx, cudaLeg::ScalarField::Literal(-1.0f),
                         cudaLeg::ScalarField::Literal(2.0f), false,
                         &cudaResult);
#endif
    LegResult cudaF4 = cudaResult;
    CheckScenario("F4 widthRamp/negative", haveVulkan, vkF4, haveCuda,
                  cudaF4, {}, Outcome::InvalidValue);

    // F5: non-finite points diagnose NonFiniteInput on Length.
    {
        Fixture bad = fx;
        bad.points[3 * 3] = std::numeric_limits<float>::quiet_NaN();
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF5 = true;
        if (haveVulkan) {
            ranF5 = ranF5 && UploadFixture(leg, bad, &badPlanes) &&
                VkLength(leg, badPlanes, VkLiteral(1.0f), 2, 5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF5 = ranF5 && cudaLeg::RunLength(
                                     &badCtx,
                                     cudaLeg::ScalarField::Literal(1.0f),
                                     &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF5 = false;
            }
        }
#endif
        ran = ran && ranF5;
        CheckScenario("F5 length/nan-point", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::NonFiniteInput);
    }

    // F6: zero normals diagnose InvalidValue on Grow.
    {
        Fixture bad = fx;
        bad.normals = {0, 0, 0, 0, 0, 3};
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF6 = true;
        if (haveVulkan) {
            ranF6 = ranF6 && UploadFixture(leg, bad, &badPlanes) &&
                VkGrow(leg, badPlanes, VkLiteral(2.0f), nullptr, 2, 5,
                       &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF6 = ranF6 && cudaLeg::RunGrow(
                                     &badCtx,
                                     cudaLeg::ScalarField::Literal(2.0f),
                                     false, &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF6 = false;
            }
        }
#endif
        ran = ran && ranF6;
        CheckScenario("F6 grow/zero-normal", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::InvalidValue);
    }

    // F7: non-finite normals diagnose NonFiniteInput on Grow.
    {
        Fixture bad = fx;
        bad.normals[3] = std::numeric_limits<float>::quiet_NaN();
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF7 = true;
        if (haveVulkan) {
            ranF7 = ranF7 && UploadFixture(leg, bad, &badPlanes) &&
                VkGrow(leg, badPlanes, VkLiteral(2.0f), nullptr, 2, 5,
                       &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF7 = ranF7 && cudaLeg::RunGrow(
                                     &badCtx,
                                     cudaLeg::ScalarField::Literal(2.0f),
                                     false, &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF7 = false;
            }
        }
#endif
        ran = ran && ranF7;
        CheckScenario("F7 grow/nan-normal", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::NonFiniteInput);
    }

    // F8/F9: out-of-range and non-finite hairT diagnose InvalidValue /
    // NonFiniteInput (ReadHairT contract in styleOps.cu).
    {
        Fixture bad = fx;
        bad.point = {0.0f, 0.25f, 0.0f, 2.0f, 1.0f};
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF8 = true;
        if (haveVulkan) {
            ranF8 = ranF8 && UploadFixture(leg, bad, &badPlanes) &&
                VkGrow(leg, badPlanes, VkLiteral(2.0f), badPlanes.hairT, 2,
                       5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF8 = ranF8 && cudaLeg::RunGrow(
                                     &badCtx,
                                     cudaLeg::ScalarField::Literal(2.0f), true,
                                     &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF8 = false;
            }
        }
#endif
        ran = ran && ranF8;
        CheckScenario("F8 grow/hairT-range", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::InvalidValue);
    }
    {
        Fixture bad = fx;
        bad.point[1] = std::numeric_limits<float>::quiet_NaN();
        VkPlanes badPlanes;
        LegResult badVk, badCuda;
        bool ranF9 = true;
        if (haveVulkan) {
            ranF9 = ranF9 && UploadFixture(leg, bad, &badPlanes) &&
                VkWidthRamp(leg, badPlanes, VkLiteral(0.0f), VkLiteral(8.0f),
                            badPlanes.hairT, 2, 5, &badVk);
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::Context badCtx;
            if (cudaLeg::Setup(&badCtx, bad)) {
                ranF9 = ranF9 && cudaLeg::RunWidthRamp(
                                     &badCtx, bad,
                                     cudaLeg::ScalarField::Literal(0.0f),
                                     cudaLeg::ScalarField::Literal(8.0f), true,
                                     &badCuda);
                cudaLeg::Teardown(&badCtx);
            } else {
                ranF9 = false;
            }
        }
#endif
        ran = ran && ranF9;
        CheckScenario("F9 widthRamp/hairT-nan", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::NonFiniteInput);
    }

    // H1: field count mismatch fails synchronously (host validation).
    {
        LegResult badVk, badCuda;
        bool ranH1 = true;
        if (haveVulkan) {
            StyleVkPipeline::WidthRampInfo info;
            info.offsets = planes.offsets;
            info.curveCount = 2;
            info.pointCount = 5;
            info.root = VkField(planes.primitive, 1,
                                StyleVkDomain::Primitive);
            info.tip = VkLiteral(2.0f);
            VkResult result = VK_SUCCESS;
            StyleVkStatus status = StyleVkStatus::Ok;
            auto c = leg.pipeline->BeginWidthRamp(info, &result, &status);
            badVk.outcome = MapVk(status);
            badVk.published = false;
            ranH1 = ranH1 && !c &&
                status == StyleVkStatus::InvalidArgument;
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::CudaStyleOps style;
            cudaLeg::ScalarField bad =
                cudaLeg::ScalarField::Device({cuda.primitive.data(), 1},
                                            expr::Domain::Primitive);
            cudaLeg::StyleStatus queued = style.WidthRamp(
                cudaLeg::Geometry(cuda, Fixture::kCurves, Fixture::kPoints),
                bad, cudaLeg::ScalarField::Literal(2.0f),
                {cuda.widths.data(), cuda.widths.size()}, cuda.producer);
            badCuda.outcome = cudaLeg::MapCuda(queued);
            badCuda.published = false;
            ranH1 = ranH1 && queued == cudaLeg::StyleStatus::InvalidArgument;
        }
#endif
        ran = ran && ranH1;
        CheckScenario("H1 host/field-count", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::InvalidArgument);
    }

    // H2: curves==0 with points!=0 fails synchronously on both legs.
    {
        LegResult badVk, badCuda;
        bool ranH2 = true;
        if (haveVulkan) {
            StyleVkPipeline::WidthRampInfo info;
            info.curveCount = 0;
            info.pointCount = 5;
            info.root = VkLiteral(1.0f);
            info.tip = VkLiteral(2.0f);
            VkResult result = VK_SUCCESS;
            StyleVkStatus status = StyleVkStatus::Ok;
            auto c = leg.pipeline->BeginWidthRamp(info, &result, &status);
            badVk.outcome = MapVk(status);
            badVk.published = false;
            ranH2 = ranH2 && !c &&
                status == StyleVkStatus::InvalidArgument;
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::CudaStyleOps style;
            cudaLeg::DeviceCurveGeometryView geometry =
                cudaLeg::Geometry(cuda, 0, Fixture::kPoints);
            cudaLeg::StyleStatus queued = style.WidthRamp(
                geometry, cudaLeg::ScalarField::Literal(1.0f),
                cudaLeg::ScalarField::Literal(2.0f),
                {cuda.widths.data(), cuda.widths.size()}, cuda.producer);
            badCuda.outcome = cudaLeg::MapCuda(queued);
            badCuda.published = false;
            ranH2 = ranH2 && queued == cudaLeg::StyleStatus::InvalidArgument;
        }
#endif
        ran = ran && ranH2;
        CheckScenario("H2 host/empty-shape", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::InvalidArgument);
    }

    // H3: non-finite literal fails host-side (validateField contract).
    {
        LegResult badVk, badCuda;
        bool ranH3 = true;
        float nan = std::numeric_limits<float>::quiet_NaN();
        if (haveVulkan) {
            ranH3 = ranH3 && VkWidthRamp(leg, planes, VkLiteral(nan),
                                         VkLiteral(2.0f), nullptr, 2, 5,
                                         &badVk) &&
                badVk.outcome == Outcome::NonFiniteInput;
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::CudaStyleOps style;
            cudaLeg::StyleStatus queued = style.WidthRamp(
                cudaLeg::Geometry(cuda, Fixture::kCurves, Fixture::kPoints),
                cudaLeg::ScalarField::Literal(nan),
                cudaLeg::ScalarField::Literal(2.0f),
                {cuda.widths.data(), cuda.widths.size()}, cuda.producer);
            badCuda.outcome = cudaLeg::MapCuda(queued);
            badCuda.published = false;
            ranH3 = ranH3 &&
                queued == cudaLeg::StyleStatus::NonFiniteInput;
        }
#endif
        ran = ran && ranH3;
        CheckScenario("H3 host/nan-literal", haveVulkan, badVk, haveCuda,
                      badCuda, {}, Outcome::NonFiniteInput);
    }

    // E1: empty geometry is a valid no-op reporting Ok with zero points.
    {
        LegResult emptyVk, emptyCuda;
        bool ranE1 = true;
        if (haveVulkan) {
            StyleVkPipeline::WidthRampInfo winfo;
            winfo.root = VkLiteral(1.0f);
            winfo.tip = VkLiteral(2.0f);
            VkResult result = VK_SUCCESS;
            StyleVkStatus status = StyleVkStatus::Ok;
            auto w = leg.pipeline->BeginWidthRamp(winfo, &result, &status);
            StyleVkSemantic semantic = StyleVkSemantic::BadValue;
            bool ok = w && result == VK_SUCCESS &&
                status == StyleVkStatus::Ok &&
                w->Poll(&semantic) == VK_SUCCESS &&
                semantic == StyleVkSemantic::Ok && w->succeeded() &&
                w->pointCount() == 0 && w->output().values;
            StyleVkPipeline::LengthInfo linfo;
            linfo.scale = VkLiteral(1.0f);
            auto l = leg.pipeline->BeginLength(linfo, &result, &status);
            ok = ok && l && l->Poll(&semantic) == VK_SUCCESS &&
                semantic == StyleVkSemantic::Ok;
            StyleVkPipeline::GrowInfo ginfo;
            ginfo.length = VkLiteral(1.0f);
            auto g = leg.pipeline->BeginGrow(ginfo, &result, &status);
            ok = ok && g && g->Poll(&semantic) == VK_SUCCESS &&
                semantic == StyleVkSemantic::Ok;
            emptyVk.outcome = ok ? Outcome::Ok : Outcome::BackendError;
            emptyVk.published = ok;
            ranE1 = ranE1 && ok;
        }
#ifdef USDGEN_ENABLE_CUDA
        if (haveCuda) {
            cudaLeg::CudaStyleOps style;
            cudaLeg::DeviceCurveGeometryView empty{
                {}, {}, {}, {}, {}, 0, 0};
            cudaLeg::DeviceView<float> noOut{};
            cudaLeg::DeviceView<float3> noVec{};
            bool ok =
                style.WidthRamp(empty, 1.0f, 2.0f, noOut, cuda.producer) ==
                    cudaLeg::StyleStatus::Ok &&
                style.Finish(cuda.consumer) == cudaLeg::StyleStatus::Ok &&
                style.Length(empty, 1.0f, noVec, cuda.producer) ==
                    cudaLeg::StyleStatus::Ok &&
                style.Finish(cuda.consumer) == cudaLeg::StyleStatus::Ok;
            cudaLeg::DeviceView<const float3> noRoots{}, noNormals{};
            ok = ok &&
                style.Grow(empty, noRoots, noNormals, 1.0f, noVec,
                           cuda.producer) == cudaLeg::StyleStatus::Ok &&
                style.Finish(cuda.consumer) == cudaLeg::StyleStatus::Ok;
            emptyCuda.outcome = ok ? Outcome::Ok : Outcome::BackendError;
            emptyCuda.published = ok;
            ranE1 = ranE1 && ok;
        }
#endif
        ran = ran && ranE1;
        CheckScenario("E1 empty/no-op", haveVulkan, emptyVk, haveCuda,
                      emptyCuda, {}, Outcome::Ok);
    }

    // N1: Noise is explicitly unfinished on both backends.
    {
        bool vkNoise =
            StyleVkPipeline::NoiseStatus() == StyleVkStatus::NotSupported;
#ifdef USDGEN_ENABLE_CUDA
        cudaLeg::CudaStyleOps style;
        cudaLeg::DeviceCurveGeometryView empty{{}, {}, {}, {}, {}, 0, 0};
        bool cudaNoise =
            style.Noise(empty, 1.0f, 7, {}, cuda.producer) ==
            cudaLeg::StyleStatus::NotSupported;
#else
        bool cudaNoise = true;
#endif
        if (haveVulkan) Check(vkNoise, "N1: vulkan Noise reports NotSupported");
        if (haveCuda) Check(cudaNoise, "N1: cuda Noise reports NotSupported");
    }

    // D1: determinism — a rerun reproduces the W1/G1 outputs bit-for-bit.
    if (haveVulkan) {
        LegResult rerun;
        bool ok = VkWidthRamp(leg, planes, VkLiteral(1.0f),
                              VkField(planes.primitive, 2,
                                      StyleVkDomain::Primitive),
                              nullptr, 2, 5, &rerun) &&
            rerun.outcome == Outcome::Ok && BitEqual(rerun.values, vkW1.values);
        Check(ok, "D1: vulkan rerun reproduces W1 bit-for-bit");
        ok = VkGrow(leg, planes,
                    VkField(planes.primitive, 2, StyleVkDomain::Primitive),
                    planes.hairT, 2, 5, &rerun) &&
            rerun.outcome == Outcome::Ok && BitEqual(rerun.values, vkG1.values);
        Check(ok, "D1: vulkan rerun reproduces G1 bit-for-bit");
    }
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda) {
        LegResult rerun;
        bool ok = cudaLeg::RunWidthRamp(
                      &cuda, fx, cudaLeg::ScalarField::Literal(1.0f),
                      cudaLeg::CudaField(cuda.primitive,
                                         expr::Domain::Primitive),
                      false, &rerun) &&
            rerun.outcome == Outcome::Ok &&
            BitEqual(rerun.values, cudaW1.values);
        Check(ok, "D1: cuda rerun reproduces W1 bit-for-bit");
    }
#endif

    Check(ran, "all scenarios executed without backend errors");

#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda) cudaLeg::Teardown(&cuda);
#endif

    std::printf("testUsdGenVulkanStyleVkParity: %s\n",
                g_failures ? "FAILED" : (haveVulkan && haveCuda ? "PASS" :
                    "SKIP (cross-backend comparison unavailable)"));
    return g_failures ? 1 : (haveVulkan && haveCuda ? 0 : 77);
}
