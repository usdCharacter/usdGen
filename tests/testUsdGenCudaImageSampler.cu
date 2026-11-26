#include "usdGen/gpu/imageSampler.h"
#include "usdGenCuda/backendId.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(value) do { if (!(value)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #value); return 1; } } while (false)

int main()
{
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(ConfigureCudaExecutionResources(device, {512u << 20, 1024}));
    auto image = ImagePayload::Create(2, 2, 4,
        std::vector<float>{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
        UsdGenImageRowOrientation::TopDown);
    CHECK(image);
    std::vector<float2> uv{{0,0}, {.5f,.5f}, {1,1}, {-0.1f,.5f}};
    DeviceBuffer<float2> uvDevice;
    DeviceBuffer<float> output;
    cudaError_t uvStatus = uvDevice.reset(uv.size());
    cudaError_t outStatus = output.reset(uv.size());
    if (uvStatus != cudaSuccess || outStatus != cudaSuccess)
        std::fprintf(stderr, "reset errors %s %s\n", cudaGetErrorString(uvStatus), cudaGetErrorString(outStatus));
    CHECK(uvStatus == cudaSuccess && outStatus == cudaSuccess);
    CHECK(cudaMemcpy(uvDevice.data(), uv.data(), uv.size()*sizeof(float2), cudaMemcpyHostToDevice) == cudaSuccess);
    auto pool = FindUsdGenExecutionResourcePool({usdGen::UsdGenCudaBackendId(), device});
    CHECK(pool);
    size_t const baseline = pool->Snapshot().usedBytes;
    {
        CudaImage gpu;
        CHECK(gpu.Upload(image) == cudaSuccess);
        for (UsdGenImageWrap wrap : {UsdGenImageWrap::Clamp, UsdGenImageWrap::Repeat,
                                     UsdGenImageWrap::Mirror, UsdGenImageWrap::Black}) {
            for (UsdGenImageFilter filter : {UsdGenImageFilter::Nearest, UsdGenImageFilter::Bilinear}) {
                UsdGenImageSampleOptions options;
                options.filter = filter; options.channel = UsdGenImageChannel::Luminance;
                options.wrap = wrap; options.defaultValue = 3; options.clampOutput = false;
                CHECK(gpu.Sample(uvDevice.view(), output.view(), options) == cudaSuccess);
                CHECK(cudaDeviceSynchronize() == cudaSuccess);
                std::vector<float> actual(uv.size());
                CHECK(cudaMemcpy(actual.data(), output.data(), actual.size()*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
                for (size_t i = 0; i < uv.size(); ++i) {
                    float expected = ImageSampler::Sample(*image, uv[i].x, uv[i].y, options);
                    CHECK(std::memcmp(&actual[i], &expected, sizeof(float)) == 0);
                }
            }
        }
        CHECK(pool->Snapshot().usedBytes >= baseline + image->TexelCount()*sizeof(float));
    }
    CHECK(pool->Snapshot().usedBytes == baseline);

    // Same-size uploads on independent streams must use a fresh allocation
    // while the prior upload/use is in flight. This also exercises retention
    // of both pageable host payloads until their events complete.
    cudaStream_t streamA = nullptr, streamB = nullptr;
    CHECK(cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking) == cudaSuccess);
    constexpr uint32_t kExtent = 2048;
    size_t const largeCount = size_t(kExtent) * kExtent;
    std::vector<float> texelsA(largeCount, 1.0f), texelsB(largeCount, 2.0f);
    auto imageA = ImagePayload::Create(kExtent, kExtent, 1, std::move(texelsA),
                                       UsdGenImageRowOrientation::BottomUp);
    auto imageB = ImagePayload::Create(kExtent, kExtent, 1, std::move(texelsB),
                                       UsdGenImageRowOrientation::BottomUp);
    CHECK(imageA && imageB);
    {
        CudaImage refreshed;
        CHECK(refreshed.Upload(imageA, streamA) == cudaSuccess);
        CHECK(refreshed.Upload(imageB, streamB) == cudaSuccess);
        UsdGenImageSampleOptions sampleOptions;
        sampleOptions.filter = UsdGenImageFilter::Nearest;
        sampleOptions.clampOutput = false;
        std::vector<float2> oneUv{{0.5f, 0.5f}};
        DeviceBuffer<float2> oneUvDevice;
        DeviceBuffer<float> oneOutput;
        CHECK(oneUvDevice.reset(1) == cudaSuccess && oneOutput.reset(1) == cudaSuccess);
        CHECK(cudaMemcpy(oneUvDevice.data(), oneUv.data(), sizeof(float2),
                         cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(refreshed.Sample(oneUvDevice.view(), oneOutput.view(), sampleOptions, streamB) == cudaSuccess);
        CHECK(cudaStreamSynchronize(streamB) == cudaSuccess);
        float actual = 0.0f;
        CHECK(cudaMemcpy(&actual, oneOutput.data(), sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(actual == 2.0f);
        CHECK(cudaStreamSynchronize(streamA) == cudaSuccess);
    }
    CHECK(pool->Snapshot().usedBytes == baseline);
    CHECK(cudaStreamDestroy(streamA) == cudaSuccess);
    CHECK(cudaStreamDestroy(streamB) == cudaSuccess);

    // Exercise the grid-stride tail beyond the historical 65,535-block
    // launch cap. The final element must be written, not left at its sentinel.
    constexpr size_t kLargeSamples = size_t(65535) * 256u + 1u;
    {
        DeviceBuffer<float2> largeUv;
        DeviceBuffer<float> largeOutput;
        CHECK(largeUv.reset(kLargeSamples) == cudaSuccess);
        CHECK(largeOutput.reset(kLargeSamples) == cudaSuccess);
        CHECK(cudaMemset(largeUv.data(), 0, kLargeSamples * sizeof(float2)) == cudaSuccess);
        CHECK(cudaMemset(largeOutput.data(), 0xcd, kLargeSamples * sizeof(float)) == cudaSuccess);
        CudaImage largeBatch;
        CHECK(largeBatch.Upload(image) == cudaSuccess);
        UsdGenImageSampleOptions largeOptions;
        largeOptions.filter = UsdGenImageFilter::Nearest;
        largeOptions.clampOutput = false;
        CHECK(largeBatch.Sample(largeUv.view(), largeOutput.view(), largeOptions) == cudaSuccess);
        CHECK(cudaDeviceSynchronize() == cudaSuccess);
        float tail = 0.0f;
        CHECK(cudaMemcpy(&tail, largeOutput.data() + (kLargeSamples - 1), sizeof(float),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK(tail == 9.0f);
    }
    CHECK(pool->Snapshot().usedBytes == baseline);
    return 0;
}
