// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenCudaVulkanParity — CUDA-vs-Vulkan equivalence gate (T1).
//
// One shared deterministic fixture (two ragged curves, rotated second root
// frame) runs through the Noise operator on every GPU backend compiled in:
// the Vulkan NoisePipeline and, when USDGEN_ENABLE_CUDA is set, the CUDA
// CudaNoise kernel. Both outputs are compared against the same host oracle
// (SeExpr double-precision vfbm, the oracle both single-backend tests use)
// within the 2e-3/component FP-contract tolerance those tests establish,
// and against each other within 4e-3 (triangle inequality over the oracle).
// Functional parity is covered alongside the numerics: both legs must accept
// the same inputs with a clean status, publish the same point count, honour
// the mask=0 bitwise pass-through contract, and reproduce their own output
// bit-for-bit on a second run (elementwise kernels, no atomics/reductions).
//
// A leg that cannot run (no NVIDIA Vulkan device, no CUDA device) is
// skipped; the test exits 77 only when no GPU backend is available at all,
// so CUDA-only, Vulkan-only, and dual-backend builds all stay green.

#include "usdGenVulkan/vulkan/deviceContext.h"
#include "usdGenVulkan/vulkan/chargedBuffer.h"
#include "usdGenCurves/vulkan/noisePipeline.h"

#include "vulkanNativeFixture.h"
#include "vulkanReadbackFixture.h"

#include "usdGenMath/usdGenMath/hash.h"
#include "SeExpr2/Noise.h"

#ifdef USDGEN_ENABLE_CUDA
#include "gpu/noise.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;

namespace {

int g_failures = 0;
void Check(bool ok, char const *what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what);
    }
}

// Per-component absolute tolerance of one float GPU lane against the double
// oracle. Matches testUsdGenCudaNoise and testUsdGenVulkanNoise (2e-3): both
// lanes are deliberately float while the oracle is SeExpr double FBM, built
// with --fmad=false / -ffp-contract=off so op order is exact and only the
// float rounding differs.
constexpr float kOracleTolerance = 2.0e-3f;
// GPU-vs-GPU bound: both lanes within kOracleTolerance of the same oracle.
constexpr float kCrossTolerance = 2.0f * kOracleTolerance;

struct V3 {
    float x, y, z;
};
inline V3 V(float x, float y, float z) { return {x, y, z}; }
inline V3 Add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 Mul(V3 a, float f) { return {a.x * f, a.y * f, a.z * f}; }
inline V3 FrameVector(V3 t, V3 b, V3 n, V3 local)
{
    return Add(Add(Mul(t, local.x), Mul(b, local.y)), Mul(n, local.z));
}

// Double-precision SeExpr vfbm oracle, identical to the one in
// testUsdGenCudaNoise and testUsdGenVulkanNoise.
inline V3 CpuVfbm(V3 restRoot, uint64_t id, int seed, float t, float frequency,
                  float correlation, int octaves, float lacunarity, float gain)
{
    const V3 hash = V(UsdGenDraw01(seed, id, kSaltNoise),
                      UsdGenDraw01(seed, id, kSaltNoise + 1u),
                      UsdGenDraw01(seed, id, kSaltNoise + 2u));
    const double input[3] = {
        double(restRoot.x * correlation + hash.x * (1.0f - correlation)),
        double(restRoot.y * correlation + hash.y * (1.0f - correlation)),
        double(restRoot.z * correlation + hash.z * (1.0f - correlation) +
               t * frequency)};
    double result[3]{};
    SeExpr2::FBM<3, 3, false, double>(input, result, octaves,
                                      double(lacunarity), double(gain));
    return V(float(result[0]), float(result[1]), float(result[2]));
}

inline float Sample257(std::vector<float> const &values, float t)
{
    const int lower =
        std::min(255, std::max(0, int(std::floor(t * 256.0f))));
    const float fraction = t * 256.0f - float(lower);
    return values[lower] + (values[lower + 1] - values[lower]) * fraction;
}

struct Fixture {
    std::vector<float> points{0, 0, 0, 0, 0, 1, 0, 0, 2, 3, 0, 0,
                              3, 1, 0, 3, 2, 0, 3, 3, 0};
    std::vector<float> rest{0, 0, 0, 0, 0, 1, 0, 0, 2, 3, 0, 0,
                            3, 1, 0, 3, 2, 0, 3, 3, 0};
    std::vector<uint32_t> offsets{0, 3, 7};
    std::vector<uint64_t> ids{41, 99};
    std::vector<float> hairT{0, .5f, 1, 0, .25f, .65f, 1};
    std::vector<float> tangent{1, 0, 0, 0, 1, 0};
    std::vector<float> binormal{0, 1, 0, 0, 0, 1};
    std::vector<float> normal{0, 0, 1, 1, 0, 0};
};

// Scenario A: groom literals (magnitude .2, frequency 1.7, correlation .4,
// octaves 3, lacunarity 2, gain .5, preserveLength 0, mask 1, seed 17,
// flat magnitude profile).
std::vector<float> OracleA(Fixture const &fx)
{
    std::vector<float> expected(fx.points.size());
    for (size_t cv = 0; cv < 2; ++cv) {
        const V3 restRoot =
            V(fx.rest[fx.offsets[cv] * 3], fx.rest[fx.offsets[cv] * 3 + 1],
              fx.rest[fx.offsets[cv] * 3 + 2]);
        const V3 t = V(fx.tangent[cv * 3], fx.tangent[cv * 3 + 1],
                       fx.tangent[cv * 3 + 2]);
        const V3 b = V(fx.binormal[cv * 3], fx.binormal[cv * 3 + 1],
                       fx.binormal[cv * 3 + 2]);
        const V3 n = V(fx.normal[cv * 3], fx.normal[cv * 3 + 1],
                       fx.normal[cv * 3 + 2]);
        for (uint32_t pt = fx.offsets[cv]; pt < fx.offsets[cv + 1]; ++pt) {
            const V3 field = CpuVfbm(restRoot, fx.ids[cv], 17, fx.hairT[pt],
                                     1.7f, 0.4f, 3, 2.0f, 0.5f);
            const V3 out =
                Add(V(fx.points[pt * 3], fx.points[pt * 3 + 1],
                      fx.points[pt * 3 + 2]),
                    FrameVector(t, b, n, Mul(field, 0.2f)));
            expected[pt * 3] = out.x;
            expected[pt * 3 + 1] = out.y;
            expected[pt * 3 + 2] = out.z;
        }
    }
    return expected;
}

struct ScenarioB {
    std::vector<float> pointMags{.1f, .2f, .3f, .35f, .25f, .15f, .05f};
    std::vector<int32_t> pointOctaves{3, 3, 3, 3, 3, 3, 3};
    std::vector<float> pointMasks{0.0f, .75f, .5f, 1.0f, .25f, .5f, 1.0f};
    std::vector<int32_t> primSeeds{17, 18};
    std::vector<uint32_t> cumulative{1, 0};
    std::vector<float> ramp;
    ScenarioB() : ramp(257, 1.0f)
    {
        ramp[0] = 0.0f;
        ramp[128] = 0.5f;
    }
};

// Scenario B: point-domain magnitude/octaves/mask, primitive seed and
// cumulative flag, ramped magnitude profile.
std::vector<float> OracleB(Fixture const &fx, ScenarioB const &sc)
{
    std::vector<float> expected(fx.points.size());
    for (size_t cv = 0; cv < 2; ++cv) {
        const bool isCum = sc.cumulative[cv] != 0;
        const V3 restRoot =
            V(fx.rest[fx.offsets[cv] * 3], fx.rest[fx.offsets[cv] * 3 + 1],
              fx.rest[fx.offsets[cv] * 3 + 2]);
        const V3 t = V(fx.tangent[cv * 3], fx.tangent[cv * 3 + 1],
                       fx.tangent[cv * 3 + 2]);
        const V3 b = V(fx.binormal[cv * 3], fx.binormal[cv * 3 + 1],
                       fx.binormal[cv * 3 + 2]);
        const V3 n = V(fx.normal[cv * 3], fx.normal[cv * 3 + 1],
                       fx.normal[cv * 3 + 2]);
        V3 running = V(0, 0, 0);
        for (uint32_t pt = fx.offsets[cv]; pt < fx.offsets[cv + 1]; ++pt) {
            const V3 field =
                CpuVfbm(restRoot, fx.ids[cv], sc.primSeeds[cv], fx.hairT[pt],
                        1.7f, 0.4f, sc.pointOctaves[pt], 2.0f, 0.5f);
            if (isCum)
                running = Add(running, field);
            const float scale = sc.pointMags[pt] *
                Sample257(sc.ramp, fx.hairT[pt]) * sc.pointMasks[pt];
            const V3 out =
                Add(V(fx.points[pt * 3], fx.points[pt * 3 + 1],
                      fx.points[pt * 3 + 2]),
                    FrameVector(t, b, n, Mul(isCum ? running : field, scale)));
            expected[pt * 3] = out.x;
            expected[pt * 3 + 1] = out.y;
            expected[pt * 3 + 2] = out.z;
        }
    }
    return expected;
}

bool AllFinite(std::vector<float> const &values)
{
    for (float v : values)
        if (!std::isfinite(v))
            return false;
    return true;
}

// Max absolute component difference; false on size mismatch or nonfinite.
bool WithinTolerance(std::vector<float> const &got,
                     std::vector<float> const &expected, float tolerance,
                     float *maxDiff)
{
    if (got.size() != expected.size() || got.empty())
        return false;
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(expected[i]))
            return false;
        worst = std::max(worst, std::fabs(got[i] - expected[i]));
    }
    if (maxDiff)
        *maxDiff = worst;
    return worst <= tolerance;
}

std::vector<uint32_t> LoadSpirv(char const *path)
{
    std::ifstream f(path, std::ios::binary);
    std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.empty() || b.size() % 4)
        return {};
    std::vector<uint32_t> r(b.size() / 4);
    std::memcpy(r.data(), b.data(), b.size());
    return r;
}

bool Prove(std::shared_ptr<NativeOwner> const &native)
{
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS)
        return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) ==
            VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE,
                        10000000000ull) == VK_SUCCESS;
}

inline std::shared_ptr<const ChargedBuffer> Upload(
    std::shared_ptr<DeviceContext> const &c, void const *data,
    VkDeviceSize bytes)
{
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes ? bytes : 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto b = ChargedBuffer::Create(
        c, bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!b)
        return {};
    if (bytes) {
        void *p = nullptr;
        if (vkMapMemory(c->device(), b->memory(), 0, bytes, 0, &p) !=
            VK_SUCCESS)
            return {};
        std::memcpy(p, data, bytes);
        vkUnmapMemory(c->device(), b->memory());
    }
    return b;
}

struct LegOutputs {
    std::vector<float> a; // scenario A displaced points (float[3*points])
    std::vector<float> b; // scenario B displaced points
    bool maskPassthrough = false;
    bool deterministic = false;
};

// Runs scenarios A/B plus the mask=0 pass-through and a determinism rerun on
// the Vulkan noise pipeline. Sets *unavailable when no usable device exists.
bool RunVulkanLeg(std::vector<uint32_t> const &validateSpv,
                  std::vector<uint32_t> const &generateSpv,
                  Fixture const &fx, ScenarioB const &sc, LegOutputs *out,
                  bool *unavailable)
{
    *unavailable = false;
    VkPhysicalDeviceFeatures fp64 = {};
    fp64.shaderFloat64 = VK_TRUE;
    auto native = CreateNative(unavailable, {}, nullptr, &fp64);
    if (*unavailable)
        return false;
    if (!native)
        return false;
    VkPhysicalDeviceFeatures feats{};
    vkGetPhysicalDeviceFeatures(native->physical, &feats);
    if (!feats.shaderFloat64) {
        *unavailable = true;
        return false;
    }

    DeviceContext::CreateInfo ci;
    ci.instance = native->instance;
    ci.physicalDevice = native->physical;
    ci.device = native->device;
    ci.computeQueue = native->queue;
    ci.computeQueueFamily = native->family;
    ci.physicalIndex = native->physicalIndex;
    ci.resourceDeviceId = 8020;
    ci.nativeLifetime = native;
    ci.resources = {size_t{32} << 20, 0};
    ci.shaderFloat64Enabled = true;
    auto context = DeviceContext::Create(ci);
    if (!context)
        return false;

    VkResult status = VK_SUCCESS;
    auto pipe =
        NoisePipeline::Create(context, validateSpv, generateSpv, &status);
    if (!pipe || status != VK_SUCCESS)
        return false;

    auto readOutput = [&](std::unique_ptr<NoisePipeline::Candidate> const &c,
                          std::vector<float> *result) {
        auto o = c->output();
        std::vector<uint8_t> bytes;
        if (!ReadVulkanBytes(native, context, o.points->buffer(),
                             o.points->sizeBytes(), o.points, &bytes))
            return false;
        result->resize(bytes.size() / sizeof(float));
        std::memcpy(result->data(), bytes.data(), bytes.size());
        return true;
    };

    const std::vector<float> profile(257, 1.0f);
    const std::vector<uint32_t> idsPacked{41, 0, 99, 0};
    auto pointsBuf = Upload(context, fx.points.data(),
                            fx.points.size() * sizeof(float));
    auto restBuf =
        Upload(context, fx.rest.data(), fx.rest.size() * sizeof(float));
    auto offsetsBuf = Upload(context, fx.offsets.data(),
                             fx.offsets.size() * sizeof(uint32_t));
    auto idsBuf = Upload(context, idsPacked.data(),
                         idsPacked.size() * sizeof(uint32_t));
    auto hairTBuf = Upload(context, fx.hairT.data(),
                           fx.hairT.size() * sizeof(float));
    auto tanBuf = Upload(context, fx.tangent.data(),
                         fx.tangent.size() * sizeof(float));
    auto binBuf = Upload(context, fx.binormal.data(),
                         fx.binormal.size() * sizeof(float));
    auto nrmBuf = Upload(context, fx.normal.data(),
                         fx.normal.size() * sizeof(float));
    auto profileBuf =
        Upload(context, profile.data(), profile.size() * sizeof(float));
    if (!pointsBuf || !restBuf || !offsetsBuf || !idsBuf || !hairTBuf ||
        !tanBuf || !binBuf || !nrmBuf || !profileBuf)
        return false;

    auto fillGeometry = [&](NoisePipeline::BeginInfo &info) {
        info.points = pointsBuf;
        info.restPoints = restBuf;
        info.curveOffsets = offsetsBuf;
        info.stableIds = idsBuf;
        info.curveCount = 2;
        info.pointCount = 7;
        info.hairT = hairTBuf;
        info.frameTangent = tanBuf;
        info.frameBinormal = binBuf;
        info.frameNormal = nrmBuf;
        info.frameCount = 2;
        info.magnitudeProfile = profileBuf;
    };
    auto setGroomLiterals = [&](NoisePipeline::BeginInfo &info) {
        info.magnitude = {0.2f, 1, nullptr, 0};
        info.frequency = {1.7f, 1, nullptr, 0};
        info.correlation = {0.4f, 1, nullptr, 0};
        info.octaves = {3, 1, nullptr, 0};
        info.lacunarity = {2.0f, 1, nullptr, 0};
        info.gain = {0.5f, 1, nullptr, 0};
        info.preserveLength = {0.0f, 1, nullptr, 0};
        info.mask = {1.0f, 1, nullptr, 0};
        info.seed = {17, 1, nullptr, 0};
        info.enabled = {1, 1, nullptr, 0};
        info.cumulative = {0, 1, nullptr, 0};
    };
    auto run = [&](NoisePipeline::BeginInfo info, std::vector<float> *got) {
        NoiseSemantic sem = NoiseSemantic::Ok;
        auto c = pipe->Begin(std::move(info), &status, &sem);
        if (!c || status != VK_SUCCESS)
            return false;
        if (!Prove(native))
            return false;
        if (c->Poll(&sem) != VK_SUCCESS || sem != NoiseSemantic::Ok ||
            !c->succeeded())
            return false;
        return readOutput(c, got);
    };

    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        if (!run(std::move(info), &out->a))
            return false;
    }
    {
        auto magBuf = Upload(context, sc.pointMags.data(),
                             sc.pointMags.size() * sizeof(float));
        auto octBuf = Upload(context, sc.pointOctaves.data(),
                             sc.pointOctaves.size() * sizeof(int32_t));
        auto maskBuf = Upload(context, sc.pointMasks.data(),
                              sc.pointMasks.size() * sizeof(float));
        auto seedBuf = Upload(context, sc.primSeeds.data(),
                              sc.primSeeds.size() * sizeof(int32_t));
        auto cumBuf = Upload(context, sc.cumulative.data(),
                             sc.cumulative.size() * sizeof(uint32_t));
        auto rampBuf = Upload(context, sc.ramp.data(),
                              sc.ramp.size() * sizeof(float));
        if (!magBuf || !octBuf || !maskBuf || !seedBuf || !cumBuf || !rampBuf)
            return false;
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        info.magnitudeProfile = rampBuf;
        info.magnitude = {.0f, 4, magBuf, 7};
        info.frequency = {1.7f, 1, nullptr, 0};
        info.correlation = {0.4f, 1, nullptr, 0};
        info.octaves = {0, 4, octBuf, 7};
        info.lacunarity = {2.0f, 1, nullptr, 0};
        info.gain = {0.5f, 1, nullptr, 0};
        info.preserveLength = {0.0f, 1, nullptr, 0};
        info.mask = {.0f, 4, maskBuf, 7};
        info.seed = {0, 2, seedBuf, 2};
        info.enabled = {1, 1, nullptr, 0};
        info.cumulative = {0, 2, cumBuf, 2};
        if (!run(std::move(info), &out->b))
            return false;
    }
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        info.mask = {0.0f, 1, nullptr, 0};
        std::vector<float> pass;
        if (!run(std::move(info), &pass))
            return false;
        out->maskPassthrough =
            pass.size() == fx.points.size() &&
            std::memcmp(pass.data(), fx.points.data(),
                        fx.points.size() * sizeof(float)) == 0;
    }
    {
        NoisePipeline::BeginInfo info;
        fillGeometry(info);
        setGroomLiterals(info);
        std::vector<float> rerun;
        if (!run(std::move(info), &rerun))
            return false;
        out->deterministic =
            rerun.size() == out->a.size() &&
            std::memcmp(rerun.data(), out->a.data(),
                        out->a.size() * sizeof(float)) == 0;
    }
    return true;
}

#ifdef USDGEN_ENABLE_CUDA
namespace cudaLeg {
using namespace usdGen::gpu;

template <class T>
bool Upload(DeviceBuffer<T> &device, std::vector<T> const &host,
            cudaStream_t stream)
{
    return (device.size() == host.size() ||
            device.reset(host.size()) == cudaSuccess) &&
        (!host.size() ||
         cudaMemcpyAsync(device.data(), host.data(),
                         host.size() * sizeof(T), cudaMemcpyHostToDevice,
                         stream) == cudaSuccess);
}
template <class T>
bool Download(DeviceBuffer<T> const &device, std::vector<T> *host,
              cudaStream_t stream)
{
    host->resize(device.size());
    return (!host->size() ||
            cudaMemcpyAsync(host->data(), device.data(),
                            host->size() * sizeof(T), cudaMemcpyDeviceToHost,
                            stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
template <class T>
DeviceView<const T> View(DeviceBuffer<T> const &buffer)
{
    return {buffer.data(), buffer.size()};
}

NoiseParameters GroomLiterals(DeviceView<const float> profile)
{
    NoiseParameters parameters;
    parameters.magnitudeProfile = profile;
    parameters.magnitude = ScalarField::Literal(.2f);
    parameters.frequency = ScalarField::Literal(1.7f);
    parameters.correlation = ScalarField::Literal(.4f);
    parameters.octaves = IntField::Literal(3);
    parameters.lacunarity = ScalarField::Literal(2.0f);
    parameters.gain = ScalarField::Literal(.5f);
    parameters.preserveLength = ScalarField::Literal(0.0f);
    parameters.seed = IntField::Literal(17);
    return parameters;
}

bool Run(Fixture const &fx, ScenarioB const &sc, LegOutputs *out,
         bool *unavailable)
{
    *unavailable = false;
    int devices = 0;
    cudaError_t const deviceStatus = cudaGetDeviceCount(&devices);
    if (deviceStatus == cudaErrorNoDevice || deviceStatus == cudaErrorInsufficientDriver ||
        (deviceStatus == cudaSuccess && devices == 0)) {
        *unavailable = true;
        return false;
    }
    if (deviceStatus != cudaSuccess) return false;
    cudaStream_t producer = nullptr, consumer = nullptr;
    if (cudaSetDevice(0) != cudaSuccess ||
        cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) !=
            cudaSuccess ||
        cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) !=
            cudaSuccess) {
        if (producer) cudaStreamDestroy(producer);
        if (consumer) cudaStreamDestroy(consumer);
        return false;
    }

    auto toFloat3 = [](std::vector<float> const &flat) {
        std::vector<float3> v(flat.size() / 3);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = make_float3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]);
        return v;
    };
    auto toFlat = [](std::vector<float3> const &v) {
        std::vector<float> flat(v.size() * 3);
        for (size_t i = 0; i < v.size(); ++i) {
            flat[i * 3] = v[i].x;
            flat[i * 3 + 1] = v[i].y;
            flat[i * 3 + 2] = v[i].z;
        }
        return flat;
    };
    const std::vector<float3> points = toFloat3(fx.points);
    const std::vector<float3> rest = toFloat3(fx.rest);
    const std::vector<float3> tangent = toFloat3(fx.tangent);
    const std::vector<float3> binormal = toFloat3(fx.binormal);
    const std::vector<float3> normal = toFloat3(fx.normal);
    const std::vector<float> profileValues(257, 1.0f);

    DeviceBuffer<float3> devicePoints, deviceRest, deviceTangent,
        deviceBinormal, deviceNormal, output;
    DeviceBuffer<uint32_t> deviceOffsets;
    DeviceBuffer<uint64_t> deviceIds;
    DeviceBuffer<float> deviceHairT, profile, pointMask, pointMagnitude;
    bool ok = Upload(devicePoints, points, producer) &&
        Upload(deviceRest, rest, producer) &&
        Upload(deviceOffsets, fx.offsets, producer) &&
        Upload(deviceIds, fx.ids, producer) &&
        Upload(deviceHairT, fx.hairT, producer) &&
        Upload(deviceTangent, tangent, producer) &&
        Upload(deviceNormal, normal, producer) &&
        Upload(deviceBinormal, binormal, producer) &&
        profile.reset(257) == cudaSuccess &&
        output.reset(points.size()) == cudaSuccess &&
        cudaMemcpyAsync(profile.data(), profileValues.data(),
                        257 * sizeof(float), cudaMemcpyHostToDevice,
                        producer) == cudaSuccess &&
        cudaStreamSynchronize(producer) == cudaSuccess;
    if (!ok) {
        cudaStreamDestroy(producer);
        cudaStreamDestroy(consumer);
        return false;
    }

    DeviceCurveGeometryView geometry{View(devicePoints), View(deviceRest), {},
                                     View(deviceOffsets), View(deviceIds), 2,
                                     points.size()};
    RestRootFrames frames{View(deviceTangent), View(deviceBinormal),
                          View(deviceNormal), {}};
    CudaNoise noise;
    std::vector<float3> raw;
    auto run = [&](NoiseParameters const &parameters,
                   std::vector<float> *got) {
        if (noise.Apply(geometry, View(deviceHairT), frames, parameters,
                        {output.data(), output.size()},
                        producer) != StyleStatus::Ok)
            return false;
        if (noise.Finish(consumer) != StyleStatus::Ok)
            return false;
        if (!Download(output, &raw, consumer))
            return false;
        *got = toFlat(raw);
        return true;
    };

    NoiseParameters groom = GroomLiterals(View(profile));
    ok = run(groom, &out->a);
    if (ok) {
        DeviceBuffer<int32_t> pointOctaves, primitiveSeed;
        DeviceBuffer<unsigned char> cumulative;
        ok = Upload(pointOctaves, sc.pointOctaves, producer) &&
            Upload(primitiveSeed, sc.primSeeds, producer) &&
            Upload(cumulative,
                   std::vector<unsigned char>{1u, 0u}, producer) &&
            Upload(pointMagnitude, sc.pointMags, producer) &&
            Upload(pointMask, sc.pointMasks, producer) &&
            Upload(profile, sc.ramp, producer);
        if (ok) {
            NoiseParameters fields = GroomLiterals(View(profile));
            fields.magnitude = ScalarField::Device(View(pointMagnitude),
                                                  expr::Domain::Point);
            fields.octaves = IntField::Device(View(pointOctaves),
                                              expr::Domain::Point);
            fields.seed = IntField::Device(View(primitiveSeed),
                                           expr::Domain::Primitive);
            fields.mask = ScalarField::Device(View(pointMask),
                                              expr::Domain::Point);
            fields.cumulative = BoolField::Device(
                {reinterpret_cast<uint8_t const *>(cumulative.data()),
                 cumulative.size()},
                expr::Domain::Primitive);
            ok = run(fields, &out->b);
        }
    }
    if (ok) {
        ok = Upload(profile, profileValues, producer);
        if (ok) {
            NoiseParameters passthrough = GroomLiterals(View(profile));
            passthrough.mask = ScalarField::Literal(0.0f);
            std::vector<float> pass;
            ok = run(passthrough, &pass);
            out->maskPassthrough = ok && pass.size() == fx.points.size() &&
                std::memcmp(pass.data(), fx.points.data(),
                            fx.points.size() * sizeof(float)) == 0;
        }
    }
    if (ok) {
        NoiseParameters rerunParams = GroomLiterals(View(profile));
        std::vector<float> rerun;
        ok = run(rerunParams, &rerun);
        out->deterministic = ok && rerun.size() == out->a.size() &&
            std::memcmp(rerun.data(), out->a.data(),
                        out->a.size() * sizeof(float)) == 0;
    }

    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
    return ok;
}

} // namespace cudaLeg
#endif // USDGEN_ENABLE_CUDA

void CheckLeg(char const *name, LegOutputs const &leg,
              std::vector<float> const &expectedA,
              std::vector<float> const &expectedB)
{
    float worst = 0.0f;
    Check(leg.a.size() == expectedA.size(), "leg publishes scenario A point count");
    Check(leg.b.size() == expectedB.size(), "leg publishes scenario B point count");
    Check(AllFinite(leg.a) && AllFinite(leg.b), "leg outputs are finite");
    bool nearA = WithinTolerance(leg.a, expectedA, kOracleTolerance, &worst);
    std::printf("%s scenario A vs oracle: max diff %.6g (tol %.6g)\n", name,
                worst, double(kOracleTolerance));
    Check(nearA, "leg matches the oracle on scenario A");
    bool nearB = WithinTolerance(leg.b, expectedB, kOracleTolerance, &worst);
    std::printf("%s scenario B vs oracle: max diff %.6g (tol %.6g)\n", name,
                worst, double(kOracleTolerance));
    Check(nearB, "leg matches the oracle on scenario B");
    Check(leg.maskPassthrough, "leg honours mask=0 bitwise pass-through");
    Check(leg.deterministic, "leg reproduces scenario A bit-for-bit");
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s noiseValidate.spv noiseGenerate.spv\n",
                     argv[0]);
        return 1;
    }
    auto validateSpv = LoadSpirv(argv[1]);
    auto generateSpv = LoadSpirv(argv[2]);
    if (validateSpv.empty() || generateSpv.empty()) {
        std::fprintf(stderr, "FAIL: cannot load SPIR-V modules\n");
        return 1;
    }

    Fixture fx;
    ScenarioB sc;
    const std::vector<float> expectedA = OracleA(fx);
    const std::vector<float> expectedB = OracleB(fx, sc);

    LegOutputs vulkan;
    bool vulkanUnavailable = false;
    const bool haveVulkan = RunVulkanLeg(validateSpv, generateSpv, fx, sc,
                                         &vulkan, &vulkanUnavailable);

    LegOutputs cuda;
    bool cudaUnavailable = true;
#ifdef USDGEN_ENABLE_CUDA
    const bool haveCuda =
        cudaLeg::Run(fx, sc, &cuda, &cudaUnavailable);
#else
    const bool haveCuda = false;
#endif

    // GPU setup/execution failures are failures even when the other leg is
    // unavailable. A parity proof needs both successfully executed backends.
    Check(haveVulkan || vulkanUnavailable, "Vulkan bring-up and execution succeed");
    Check(haveCuda || cudaUnavailable, "CUDA bring-up and execution succeed");
    if (g_failures) return 1;
    if (!haveVulkan && !haveCuda) {
        std::printf("testUsdGenCudaVulkanParity: both GPU backends unavailable\n");
        return 77;
    }

    if (haveVulkan)
        CheckLeg("vulkan", vulkan, expectedA, expectedB);
    else
        Check(vulkanUnavailable, "vulkan leg failed but a device exists");
#ifdef USDGEN_ENABLE_CUDA
    if (haveCuda)
        CheckLeg("cuda", cuda, expectedA, expectedB);
    else
        Check(cudaUnavailable, "cuda leg failed but a device exists");
#else
    std::printf("cuda leg not compiled (USDGEN_ENABLE_CUDA off)\n");
#endif

    if (haveVulkan && haveCuda) {
        float worst = 0.0f;
        bool nearA =
            WithinTolerance(vulkan.a, cuda.a, kCrossTolerance, &worst);
        std::printf("cuda vs vulkan scenario A: max diff %.6g (tol %.6g)\n",
                    worst, double(kCrossTolerance));
        Check(nearA, "cuda and vulkan agree on scenario A");
        bool nearB =
            WithinTolerance(vulkan.b, cuda.b, kCrossTolerance, &worst);
        std::printf("cuda vs vulkan scenario B: max diff %.6g (tol %.6g)\n",
                    worst, double(kCrossTolerance));
        Check(nearB, "cuda and vulkan agree on scenario B");
    } else {
        std::printf("cross-backend comparison skipped (%s only)\n",
                    haveVulkan ? "vulkan" : "cuda");
    }

    std::printf("testUsdGenCudaVulkanParity: %s\n",
                g_failures ? "FAILED" : (haveVulkan && haveCuda ? "PASS" : "SKIP (cross-backend comparison unavailable)"));
    return g_failures ? 1 : (haveVulkan && haveCuda ? 0 : 77);
}
