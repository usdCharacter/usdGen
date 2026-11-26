// testUsdGenVulkanMapVkSample — parity proof for the Vulkan image-map port
// (vulkan/mapVkImage.h + shaders/mapVkSample.comp) against the CUDA twin
// (gpu/imageSampler.cu, CudaImage::Sample).
//
// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Bitwise parity with gpu/imageSampler.cu (--fmad=false), using an explicit
// host model compiled with -ffp-contract=off. All CUDA comparisons are direct
// equality; unsupported hardware skips, while initialization and execution
// failures fail the test. The same contract applies to every Vulkan vendor.
#include "usdGenVulkan/vulkan/mapVkImage.h"
#include "vulkanNativeFixture.h"

#ifdef USDGEN_ENABLE_CUDA
#include "usdGenSurface/cuda/imageSampler.h"
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace usdGen;
using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "MapVk check failed: %s (%d)\n", #x, __LINE__); return 1; } } while (false)

// Test-only queue proof. Production Candidate::Poll never waits or reads back
// geometry. This fence is submitted after the candidate on its confined queue.
static bool Prove(std::shared_ptr<NativeOwner> const& native) {
    if (vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    return vkQueueSubmit(native->queue, 1, &submit, native->fence) == VK_SUCCESS &&
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) == VK_SUCCESS;
}

static bool Download(std::shared_ptr<NativeOwner> const& native,
    std::shared_ptr<const ChargedBuffer> const& source, void* values, size_t bytes) {
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto staging = ChargedBuffer::Create(source->context(), bi,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!staging) return false;
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = native->commands;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer command = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(native->device, &ai, &command) != VK_SUCCESS) return false;
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS) return false;
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(command, source->buffer(), staging->buffer(), 1, &region);
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(command) != VK_SUCCESS ||
        vkResetFences(native->device, 1, &native->fence) != VK_SUCCESS)
        return false;
    struct Keep {
        std::shared_ptr<NativeOwner> native;
        std::shared_ptr<const ChargedBuffer> source;
    };
    auto keep = std::make_shared<Keep>(Keep{native, source});
    if (staging->MarkSubmitted(native->fence, keep) != VK_SUCCESS) return false;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    if (vkQueueSubmit(native->queue, 1, &submit, native->fence) != VK_SUCCESS ||
        vkWaitForFences(native->device, 1, &native->fence, VK_TRUE, 10000000000ull) != VK_SUCCESS ||
        staging->PollComplete() != VK_SUCCESS)
        return false;
    void* mapped = nullptr;
    if (vkMapMemory(native->device, staging->memory(), 0, bytes, 0, &mapped) != VK_SUCCESS)
        return false;
    std::memcpy(values, mapped, bytes);
    vkUnmapMemory(native->device, staging->memory());
    vkFreeCommandBuffers(native->device, native->commands, 1, &command);
    return true;
}

static bool MemEqual(std::vector<float> const& a, std::vector<float> const& b) {
    return a.size() == b.size() && (a.empty() ||
        std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

static bool ExpectEqual(char const* tag, std::vector<float> const& pairs,
    std::vector<float> const& actual, std::vector<float> const& expected) {
    if (MemEqual(actual, expected)) return true;
    std::printf("MapVk MISMATCH %s: sizes %zu vs %zu\n", tag, actual.size(), expected.size());
    size_t const n = std::min(actual.size(), expected.size());
    for (size_t i = 0; i < n; ++i) {
        uint32_t abits = 0, ebits = 0;
        std::memcpy(&abits, &actual[i], 4);
        std::memcpy(&ebits, &expected[i], 4);
        if (abits == ebits) continue;
        uint32_t ubits = 0, vbits = 0;
        std::memcpy(&ubits, &pairs[2 * i], 4);
        std::memcpy(&vbits, &pairs[2 * i + 1], 4);
        std::printf("  [%s] i=%zu uv=(%a [%08x], %a [%08x]) actual=%a [%08x] expected=%a [%08x]\n",
            tag, i, pairs[2 * i], ubits, pairs[2 * i + 1], vbits, actual[i], abits,
            expected[i], ebits);
        return false;
    }
    return false;
}

// Independently evaluate the CUDA operation order with separate roundings.
// -ffp-contract=off is required so host compilation cannot fuse the model.
static float MapVkModelWrap(float value, UsdGenImageWrap mode, bool* outside) {
    *outside = false;
    if (!std::isfinite(value)) { *outside = true; return 0.0f; }
    switch (mode) {
    case UsdGenImageWrap::Clamp: return std::fmin(1.0f, std::fmax(0.0f, value));
    case UsdGenImageWrap::Repeat: return value - std::floor(value);
    case UsdGenImageWrap::Mirror: {
        float folded = -2.0f * std::floor(value * 0.5f) + value;
        return folded <= 1.0f ? folded : 2.0f - folded;
    }
    case UsdGenImageWrap::Black:
        if (value < 0.0f || value > 1.0f) *outside = true;
        return value;
    }
    *outside = true;
    return 0.0f;
}

static int MapVkModelIndex(int value, int extent, UsdGenImageWrap mode, bool* outside) {
    if (value >= 0 && value < extent) return value;
    switch (mode) {
    case UsdGenImageWrap::Clamp: return std::max(0, std::min(extent - 1, value));
    case UsdGenImageWrap::Repeat: {
        int result = value % extent;
        return result < 0 ? result + extent : result;
    }
    case UsdGenImageWrap::Mirror: {
        if (value < 0 || value > extent) { *outside = true; return 0; }
        return value < extent ? value : extent - 1;
    }
    case UsdGenImageWrap::Black: *outside = true; return 0;
    }
    *outside = true;
    return 0;
}

static float MapVkDecomposedSample(UsdGenImagePayload const& image, float u, float v,
    UsdGenImageSampleOptions const& options) {
    bool uo = false, vo = false;
    u = MapVkModelWrap(u, options.wrap, &uo);
    v = MapVkModelWrap(v, options.wrap, &vo);
    float raw = options.defaultValue;
    if (!uo && !vo) {
        float x = u * float(image.Width() - 1u);
        float y = v * float(image.Height() - 1u);
        auto texel = [&](int tx, int ty, bool* out) {
            bool xo = false, yo = false;
            int ix = MapVkModelIndex(tx, int(image.Width()), options.wrap, &xo);
            int iy = MapVkModelIndex(ty, int(image.Height()), options.wrap, &yo);
            if (xo || yo) { *out = true; return options.defaultValue; }
            uint32_t row = image.Orientation() == UsdGenImageRowOrientation::TopDown
                ? image.Height() - 1u - uint32_t(iy)
                : uint32_t(iy);
            size_t base = (size_t(row) * image.Width() + size_t(ix)) * image.Channels();
            if (options.channel == UsdGenImageChannel::Luminance) {
                float const* t = image.Data() + base;
                float luma = t[0] * 0.2126f + t[1] * 0.7152f;
                return luma + t[2] * 0.0722f;
            }
            return image.Data()[base + uint32_t(options.channel)];
        };
        bool outside = false;
        if (options.filter == UsdGenImageFilter::Nearest) {
            raw = texel(int(std::floor(x + 0.5f)), int(std::floor(y + 0.5f)), &outside);
        } else {
            int x0 = int(std::floor(x)), y0 = int(std::floor(y));
            int x1 = x0 + 1, y1 = y0 + 1;
            float fx = x - float(x0), fy = y - float(y0);
            float a = texel(x0, y0, &outside), b = texel(x1, y0, &outside);
            float c = texel(x0, y1, &outside), d = texel(x1, y1, &outside);
            float ab = a + (b - a) * fx;
            float cd = c + (d - c) * fx;
            raw = ab + (cd - ab) * fy;
        }
    }
    float value = raw * options.scale + options.offset;
    if (options.clampOutput) value = std::fmin(options.outputMax, std::fmax(options.outputMin, value));
    return value;
}

int main(int argc, char** argv) {
    bool quarantine = argc == 3 && std::string(argv[2]) == "--quarantine";
    CHECK(argc == 2 || quarantine);
    std::ifstream shader(argv[1], std::ios::binary);
    CHECK(shader);
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader)), {});
    CHECK(!bytes.empty() && bytes.size() % sizeof(uint32_t) == 0);
    std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
    std::memcpy(code.data(), bytes.data(), bytes.size());

    bool unavailable = false;
    auto native = CreateNative(&unavailable);
    if (unavailable) return 77;
    CHECK(native);

    std::weak_ptr<NativeOwner> weakNative = native;
    DeviceContext::CreateInfo info;
    info.instance = native->instance;
    info.physicalDevice = native->physical;
    info.device = native->device;
    info.computeQueue = native->queue;
    info.computeQueueFamily = native->family;
    info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 7101;
    info.nativeLifetime = native;
    info.resources = {size_t{512} << 20, 0};
    auto context = DeviceContext::Create(info);
    CHECK(context);
    info.nativeLifetime.reset();
    auto pool = context->resources();
    auto baseline = pool->Snapshot();

    VkResult status = VK_SUCCESS;
    CHECK(!MapVkSamplePipeline::Create(context, {0x07230203u}, &status));
    CHECK(status == VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline = MapVkSamplePipeline::Create(context, code, &status);
    CHECK(pipeline && status == VK_SUCCESS);

    auto uploadUv = [&](std::vector<float> const& pairs) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = pairs.size() * sizeof(float);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto buffer = ChargedBuffer::Create(context, bi,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            UsdGenExecutionResourceKind::Active);
        if (!buffer) return buffer;
        void* mapped = nullptr;
        if (vkMapMemory(native->device, buffer->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS)
            return std::shared_ptr<ChargedBuffer>{};
        std::memcpy(mapped, pairs.data(), bi.size);
        vkUnmapMemory(native->device, buffer->memory());
        return buffer;
    };

    auto run = [&](std::shared_ptr<MapVkImage> const& image, std::vector<float> const& pairs,
                   UsdGenImageSampleOptions const& options, std::vector<float>* out) {
        uint32_t const count = uint32_t(pairs.size() / 2);
        auto uv = uploadUv(pairs);
        if (!uv) return false;
        VkResult r = VK_SUCCESS;
        auto candidate = pipeline->Begin(image->View(), uv, count, options, &r);
        if (!candidate || r != VK_SUCCESS) return false;
        if (candidate->output()) return false;
        if (!Prove(native)) return false;
        uint32_t semantic = UINT32_MAX;
        if (candidate->Poll(&semantic) != VK_SUCCESS || semantic != 0) return false;
        if (!candidate->succeeded()) return false;
        auto output = candidate->output();
        if (!output || candidate->count() != count ||
            candidate->uvOwner() != uv ||
            candidate->imageOwner().texels != image->View().texels)
            return false;
        out->assign(count, 0);
        return Download(native, output, out->data(), out->size() * sizeof(float));
    };

    auto imageB = ImagePayload::Create(3, 2, 4,
        std::vector<float>{
            0.1f, -2.5f, 7.25f, 100.0f, 3.5f, 0.0f, -0.0f, 1e6f, 9.75f, 4.25f, -8.5f, 0.5f,
            11.0f, 12.5f, 13.75f, -14.25f, 15.5f, -16.75f, 17.125f, 18.0f, -19.5f, 20.25f, 21.5f, -22.75f},
        UsdGenImageRowOrientation::BottomUp);
    CHECK(imageB);
    auto vkB = MapVkImage::Create(context, imageB, &status);
    CHECK(vkB && status == VK_SUCCESS);
    float const qnan = std::numeric_limits<float>::quiet_NaN();
    float const inf = std::numeric_limits<float>::infinity();
    std::vector<float> pairsB{0, 0, 1, 1, 0.33f, 0.66f, -1.5f, 2.25f, 3.0f, -4.0f, 2.0f, 2.0f,
        qnan, 0.5f, 0.5f, qnan, inf, 0.25f, 0.25f, -inf, -0.0f, 1.0f, 0.9999999f, 0.0000001f};

    auto expect = [&](UsdGenImagePayload const& payload, std::vector<float> const& pairs,
                      UsdGenImageSampleOptions const& options) {
        std::vector<float> expected(pairs.size() / 2);
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = MapVkDecomposedSample(payload, pairs[2 * i], pairs[2 * i + 1], options);
        return expected;
    };

    // Case A: exact mirror of tests/testUsdGenCudaImageSampler.cu (2x2 RGBA,
    // top-down; the same 4 UVs; every wrap x filter; luminance channel).
    auto imageA = ImagePayload::Create(2, 2, 4,
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
        UsdGenImageRowOrientation::TopDown);
    CHECK(imageA);
    auto vkA = MapVkImage::Create(context, imageA, &status);
    CHECK(vkA && status == VK_SUCCESS && vkA->Width() == 2 && vkA->Height() == 2 &&
        vkA->Channels() == 4);
    std::vector<float> pairsA{0, 0, .5f, .5f, 1, 1, -0.1f, .5f};
    for (UsdGenImageWrap wrap :
        {UsdGenImageWrap::Clamp, UsdGenImageWrap::Repeat, UsdGenImageWrap::Mirror, UsdGenImageWrap::Black}) {
        for (UsdGenImageFilter filter : {UsdGenImageFilter::Nearest, UsdGenImageFilter::Bilinear}) {
            UsdGenImageSampleOptions options;
            options.filter = filter;
            options.channel = UsdGenImageChannel::Luminance;
            options.wrap = wrap;
            options.defaultValue = 3;
            options.clampOutput = false;
            std::vector<float> actual;
            CHECK(run(vkA, pairsA, options, &actual));
            CHECK(ExpectEqual("A", pairsA, actual, expect(*imageA, pairsA, options)));
        }
    }
    std::printf("MapVk: case A (CUDA-test mirror, 8 combos) exact\n");

    // Case B: channel sweep, scale/offset/clamp, bottom-up non-square image,
    // hostile UVs (out-of-range, integers, NaN, infinities). Image, device
    // copy, and UVs were created above for the rounding probe and are reused.
    for (UsdGenImageChannel channel : {UsdGenImageChannel::R, UsdGenImageChannel::G,
             UsdGenImageChannel::B, UsdGenImageChannel::A, UsdGenImageChannel::Luminance}) {
        for (UsdGenImageWrap wrap : {UsdGenImageWrap::Clamp, UsdGenImageWrap::Repeat,
                 UsdGenImageWrap::Mirror, UsdGenImageWrap::Black}) {
            for (UsdGenImageFilter filter : {UsdGenImageFilter::Nearest, UsdGenImageFilter::Bilinear}) {
                UsdGenImageSampleOptions plain, scaled;
                plain.filter = scaled.filter = filter;
                plain.channel = scaled.channel = channel;
                plain.wrap = scaled.wrap = wrap;
                plain.defaultValue = scaled.defaultValue = -7.5f;
                plain.clampOutput = false;
                scaled.scale = 2.5f;
                scaled.offset = -1.25f;
                scaled.clampOutput = true;
                scaled.outputMin = -3.0f;
                scaled.outputMax = 9.0f;
                std::vector<float> a, b;
                CHECK(run(vkB, pairsB, plain, &a));
                CHECK(ExpectEqual("B/plain", pairsB, a, expect(*imageB, pairsB, plain)));
                CHECK(run(vkB, pairsB, scaled, &b));
                CHECK(ExpectEqual("B/scaled", pairsB, b, expect(*imageB, pairsB, scaled)));
            }
        }
    }
    std::printf("MapVk: case B (5 channels x 4 wraps x 2 filters x 2 epilogues) exact\n");

    // Case C: 1- and 3-channel images plus a 257-count multi-group batch.
    auto imageC1 = ImagePayload::Create(4, 3, 1,
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        UsdGenImageRowOrientation::TopDown);
    auto imageC3 = ImagePayload::Create(2, 3, 3,
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
        UsdGenImageRowOrientation::BottomUp);
    CHECK(imageC1 && imageC3);
    auto vkC1 = MapVkImage::Create(context, imageC1, &status);
    auto vkC3 = MapVkImage::Create(context, imageC3, &status);
    CHECK(vkC1 && vkC3 && status == VK_SUCCESS);
    std::vector<float> pairsC;
    for (uint32_t i = 0; i < 257; ++i) {
        pairsC.push_back(float(i % 17) / 8.0f - 0.5f);
        pairsC.push_back(float(i % 13) / 6.0f - 0.25f);
    }
    for (auto entry : {std::make_pair(imageC1, vkC1), std::make_pair(imageC3, vkC3)}) {
        for (UsdGenImageWrap wrap : {UsdGenImageWrap::Clamp, UsdGenImageWrap::Repeat,
                 UsdGenImageWrap::Mirror, UsdGenImageWrap::Black}) {
            for (UsdGenImageFilter filter : {UsdGenImageFilter::Nearest, UsdGenImageFilter::Bilinear}) {
                UsdGenImageSampleOptions options;
                options.filter = filter;
                options.wrap = wrap;
                options.channel = entry.first->Channels() == 1
                    ? UsdGenImageChannel::R
                    : UsdGenImageChannel::Luminance;
                options.defaultValue = 11;
                options.clampOutput = false;
                std::vector<float> actual;
                CHECK(run(entry.second, pairsC, options, &actual));
                CHECK(ExpectEqual("C", pairsC, actual, expect(*entry.first, pairsC, options)));
            }
        }
    }
    std::printf("MapVk: case C (1/3-channel, 257-count batches) exact\n");

    bool cudaCompared = false;
#ifdef USDGEN_ENABLE_CUDA
    // Case D: direct bitwise CUDA comparison on the same inputs.
    {
        int devices = 0;
        cudaError_t const deviceStatus = cudaGetDeviceCount(&devices);
        bool const haveDevice = deviceStatus == cudaSuccess && devices > 0;
        CHECK(deviceStatus == cudaSuccess || deviceStatus == cudaErrorNoDevice ||
            deviceStatus == cudaErrorInsufficientDriver);
        gpu::CudaImage cudaA, cudaB;
        if (haveDevice) {
            CHECK(cudaSetDevice(0) == cudaSuccess);
            CHECK(gpu::ConfigureCudaExecutionResources(0, {512u << 20, 1024}));
            CHECK(cudaA.Upload(imageA) == cudaSuccess);
            CHECK(cudaB.Upload(imageB) == cudaSuccess);
        }
        bool const twinReady = haveDevice;
        if (twinReady) {
            // Every allocation, launch, and download failure fails this proof.
            auto cudaSample = [&](gpu::CudaImage const& cuda, std::vector<float> const& pairs,
                                  UsdGenImageSampleOptions const& options,
                                  std::vector<float>* out) {
                size_t const count = pairs.size() / 2;
                gpu::DeviceBuffer<float2> uvDevice;
                gpu::DeviceBuffer<float> output;
                cudaError_t status = uvDevice.reset(count);
                if (status == cudaSuccess) status = output.reset(count);
                std::vector<float2> uv(count);
                for (size_t i = 0; i < count; ++i) uv[i] = {pairs[2 * i], pairs[2 * i + 1]};
                if (status == cudaSuccess)
                    status = cudaMemcpy(uvDevice.data(), uv.data(), count * sizeof(float2),
                        cudaMemcpyHostToDevice);
                if (status == cudaSuccess) status = cuda.Sample(uvDevice.view(), output.view(), options);
                if (status == cudaSuccess) status = cudaDeviceSynchronize();
                if (status == cudaSuccess) {
                    out->assign(count, 0);
                    status = cudaMemcpy(out->data(), output.data(), count * sizeof(float),
                        cudaMemcpyDeviceToHost);
                }
                return status;
            };
            size_t compared = 0;
            for (UsdGenImageWrap wrap : {UsdGenImageWrap::Clamp, UsdGenImageWrap::Repeat,
                     UsdGenImageWrap::Mirror, UsdGenImageWrap::Black}) {
                for (UsdGenImageFilter filter :
                    {UsdGenImageFilter::Nearest, UsdGenImageFilter::Bilinear}) {
                    UsdGenImageSampleOptions options;
                    options.filter = filter;
                    options.channel = UsdGenImageChannel::Luminance;
                    options.wrap = wrap;
                    options.defaultValue = 3;
                    options.clampOutput = false;
                    std::vector<float> vk, cu;
                    CHECK(run(vkA, pairsA, options, &vk));
                    CHECK(cudaSample(cudaA, pairsA, options, &cu) == cudaSuccess);
                    CHECK(ExpectEqual("D/cu-model-A", pairsA, cu,
                        expect(*imageA, pairsA, options)));
                    CHECK(ExpectEqual("D/vk-cu-A", pairsA, vk, cu));
                    compared += pairsA.size() / 2;
                    UsdGenImageSampleOptions scaled = options;
                    scaled.scale = -1.5f;
                    scaled.offset = 4.25f;
                    scaled.clampOutput = true;
                    scaled.outputMin = 0.0f;
                    scaled.outputMax = 10.0f;
                    CHECK(run(vkB, pairsB, scaled, &vk));
                    CHECK(cudaSample(cudaB, pairsB, scaled, &cu) == cudaSuccess);
                    CHECK(ExpectEqual("D/cu-model-B", pairsB, cu,
                        expect(*imageB, pairsB, scaled)));
                    CHECK(ExpectEqual("D/vk-cu-B", pairsB, vk, cu));
                    compared += pairsB.size() / 2;
                }
            }
            std::printf("MapVk: case D (CUDA twin, %zu samples) bitwise equal\n", compared);
            CHECK(compared > 0);
            cudaCompared = true;
        } else {
            std::printf("MapVk: case D skipped (CUDA device unavailable)\n");
        }
    }
#else
    std::printf("MapVk: case D skipped (CUDA off)\n");
#endif

    // Case E: grid-stride tail past the historical 65,535-block launch cap,
    // mirroring the CUDA test's large-batch leg.
    {
        constexpr size_t kLarge = size_t(65535) * 256u + 1u;
        std::vector<float> zeros(2 * kLarge, 0.0f);
        UsdGenImageSampleOptions options;
        options.filter = UsdGenImageFilter::Nearest;
        options.channel = UsdGenImageChannel::R;
        options.clampOutput = false;
        std::vector<float> actual;
        CHECK(run(vkA, zeros, options, &actual));
        float const expected = ImageSampler::Sample(*imageA, 0, 0, options);
        CHECK(actual.size() == kLarge && actual.front() == expected &&
            actual.back() == expected && expected == 9.0f);
        std::printf("MapVk: case E (tail of %zu) exact\n", kLarge);
    }

    // Admission: invalid options, shape mismatches, foreign contexts, and
    // the empty short-circuit all fail closed without submitting.
    {
        auto uv = uploadUv(pairsA);
        CHECK(uv);
        auto before = pool->Snapshot();
        UsdGenImageSampleOptions bad = {};
        bad.channel = UsdGenImageChannel::Luminance;
        CHECK(!pipeline->Begin(vkC1->View(), uv, 4, bad, &status));
        bad = {};
        bad.channel = UsdGenImageChannel::A;
        CHECK(!pipeline->Begin(vkC1->View(), uv, 4, bad, &status));
        bad = {};
        bad.clampOutput = true;
        bad.outputMin = 2;
        bad.outputMax = 1;
        CHECK(!pipeline->Begin(vkA->View(), uv, 4, bad, &status));
        bad = {};
        bad.scale = qnan;
        CHECK(!pipeline->Begin(vkA->View(), uv, 4, bad, &status));
        bad = {};
        CHECK(!pipeline->Begin(vkA->View(), uv, 5, bad, &status));
        CHECK(!pipeline->Begin({}, uv, 4, bad, &status));
        CHECK(!pipeline->Begin(vkA->View(), {}, 4, bad, &status));
        {
            auto empty = pipeline->Begin(vkA->View(), {}, 0, bad, &status);
            uint32_t semantic = UINT32_MAX;
            CHECK(empty && status == VK_SUCCESS && empty->Poll(&semantic) == VK_SUCCESS &&
                semantic == 0 && empty->succeeded() && !empty->output());
        }
        {
            auto otherInfo = info;
            otherInfo.nativeLifetime = native;
            auto otherContext = DeviceContext::Create(otherInfo);
            CHECK(otherContext && otherContext != context);
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = pairsA.size() * sizeof(float);
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            auto other = ChargedBuffer::Create(otherContext, bi,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UsdGenExecutionResourceKind::Active);
            CHECK(other && !pipeline->Begin(vkA->View(), other, 4, bad, &status));
            bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            auto wrongUsage = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UsdGenExecutionResourceKind::Active);
            CHECK(wrongUsage && !pipeline->Begin(vkA->View(), wrongUsage, 4, bad, &status));
        }
        bool admitted = false;
        CHECK(!pipeline->Begin(vkA->View(), uv, 4, bad, &status,
            [&] { admitted = true; return false; }));
        CHECK(admitted && status == VK_ERROR_OUT_OF_DEVICE_MEMORY);
        CHECK(!pipeline->Begin(vkA->View(), uv, 4, bad, &status, []() -> bool { throw 1; }));
        CHECK(status == VK_ERROR_UNKNOWN);
        CHECK(pool->Snapshot().usedBytes == before.usedBytes &&
            pool->Snapshot().byKind == before.byKind);
        std::printf("MapVk: admission checks exact\n");
    }

    if (quarantine) {
        auto uv = uploadUv(pairsA);
        CHECK(uv);
        UsdGenImageSampleOptions options;
        auto candidate = pipeline->Begin(vkA->View(), uv, 4, options, &status);
        CHECK(candidate && status == VK_SUCCESS);
        // Images outside the candidate graph are freed; the snapshot covers
        // only the quarantined graph (candidate + its uv + its image).
        vkB.reset();
        vkC1.reset();
        vkC3.reset();
        auto charged = pool->Snapshot();
        CHECK(charged.usedBytes > baseline.usedBytes);
        std::weak_ptr<const ChargedBuffer> weakUv = uv;
        candidate->Quarantine();
        candidate.reset();
        uv.reset();
        vkA.reset();
        pipeline.reset();
        context.reset();
        CHECK(!weakUv.expired());
        CHECK(Prove(native));
        native.reset();
        CHECK(!weakNative.expired() && !weakUv.expired());
        CHECK(pool->Snapshot().usedBytes == charged.usedBytes &&
            pool->Snapshot().byKind == charged.byKind);
        return cudaCompared ? 0 : 77;
    }

    vkA.reset();
    vkB.reset();
    vkC1.reset();
    vkC3.reset();
    pipeline.reset();
    context.reset();
    native.reset();
    CHECK(weakNative.expired());
    CHECK(pool->Snapshot().usedBytes == baseline.usedBytes &&
        pool->Snapshot().byKind == baseline.byKind);
    std::puts(cudaCompared ? "MapVk: PASS (CUDA arithmetic contract)" :
        "MapVk: SKIP (CUDA comparison unavailable)");
    return cudaCompared ? 0 : 77;
}
