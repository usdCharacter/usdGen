#include "imageSampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace usdGen::gpu {
namespace {

struct KernelImage {
    float const* texels;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint8_t orientation;
};

__device__ float Wrap(float value, UsdGenImageWrap mode, bool* outside)
{
    *outside = false;
    if (!isfinite(value)) { *outside = true; return 0.0f; }
    switch (mode) {
    case UsdGenImageWrap::Clamp: return fminf(1.0f, fmaxf(0.0f, value));
    case UsdGenImageWrap::Repeat: return value - floorf(value);
    case UsdGenImageWrap::Mirror: {
        float result = value - 2.0f * floorf(value * 0.5f);
        return result <= 1.0f ? result : 2.0f - result;
    }
    case UsdGenImageWrap::Black:
        if (value < 0.0f || value > 1.0f) *outside = true;
        return value;
    }
    *outside = true; return 0.0f;
}

__device__ int Index(int value, int extent, UsdGenImageWrap mode, bool* outside)
{
    if (value >= 0 && value < extent) return value;
    switch (mode) {
    case UsdGenImageWrap::Clamp: return max(0, min(extent - 1, value));
    case UsdGenImageWrap::Repeat: {
        int result = value % extent;
        return result < 0 ? result + extent : result;
    }
    case UsdGenImageWrap::Mirror: {
        long long const period = static_cast<long long>(extent) * 2ll;
        long long result = static_cast<long long>(value) % period;
        if (result < 0) result += period;
        return static_cast<int>(result < extent ? result : period - 1 - result);
    }
    case UsdGenImageWrap::Black: *outside = true; return 0;
    }
    *outside = true; return 0;
}

__device__ float Texel(KernelImage image, int x, int y,
                       UsdGenImageSampleOptions options, bool* outside)
{
    bool xo = false, yo = false;
    int const ix = Index(x, static_cast<int>(image.width), options.wrap, &xo);
    int const iy = Index(y, static_cast<int>(image.height), options.wrap, &yo);
    if (xo || yo) { *outside = true; return options.defaultValue; }
    uint32_t const row = image.orientation ==
        static_cast<uint8_t>(UsdGenImageRowOrientation::TopDown)
        ? image.height - 1u - static_cast<uint32_t>(iy) : static_cast<uint32_t>(iy);
    size_t const base = (size_t(row) * image.width + static_cast<size_t>(ix)) * image.channels;
    if (options.channel == UsdGenImageChannel::Luminance)
        return image.texels[base] * 0.2126f + image.texels[base + 1] * 0.7152f +
               image.texels[base + 2] * 0.0722f;
    return image.texels[base + static_cast<uint32_t>(options.channel)];
}

__device__ float RawSample(KernelImage image, float u, float v,
                           UsdGenImageSampleOptions options)
{
    bool uo = false, vo = false;
    u = Wrap(u, options.wrap, &uo); v = Wrap(v, options.wrap, &vo);
    if (uo || vo) return options.defaultValue;
    float const x = u * static_cast<float>(image.width - 1u);
    float const y = v * static_cast<float>(image.height - 1u);
    bool outside = false;
    if (options.filter == UsdGenImageFilter::Nearest)
        return Texel(image, static_cast<int>(floorf(x + 0.5f)),
                     static_cast<int>(floorf(y + 0.5f)), options, &outside);
    int const x0 = static_cast<int>(floorf(x));
    int const y0 = static_cast<int>(floorf(y));
    int const x1 = x0 + 1, y1 = y0 + 1;
    float const fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
    float const a = Texel(image, x0, y0, options, &outside);
    float const b = Texel(image, x1, y0, options, &outside);
    float const c = Texel(image, x0, y1, options, &outside);
    float const d = Texel(image, x1, y1, options, &outside);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) -
                                 (a + (b - a) * fx)) * fy;
}

__global__ void SampleKernel(KernelImage image, float2 const* uv, float* output,
                             size_t count, UsdGenImageSampleOptions options)
{
    size_t const stride = size_t(gridDim.x) * blockDim.x;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;) {
        float value = RawSample(image, uv[index].x, uv[index].y, options);
        value = value * options.scale + options.offset;
        if (options.clampOutput) value = fminf(options.outputMax, fmaxf(options.outputMin, value));
        output[index] = value;
        if (index > count - stride) break;
        index += stride;
    }
}

unsigned Blocks(size_t count)
{
    size_t const blocks = count / 256u + (count % 256u != 0);
    return static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(blocks, 65535)));
}

} // namespace

cudaError_t CudaImage::Upload(std::shared_ptr<const UsdGenImagePayload> const& payload,
                              cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation,
                              UsdGenExecutionResourceKind kind)
{
    if (!payload) return cudaErrorInvalidValue;
    return UploadImpl(payload, stream, reservation, kind);
}

cudaError_t CudaImage::Upload(UsdGenImagePayload const& payload, cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation,
                              UsdGenExecutionResourceKind kind)
{
    if (!payload.IsValid()) return cudaErrorInvalidValue;
    // A non-owning shared_ptr would not prove pageable source lifetime. Copy
    // into a fresh immutable payload so the asynchronous transfer is safe.
    auto owned = UsdGenImagePayload::Create(payload.Width(), payload.Height(), payload.Channels(),
        payload.Data(), payload.TexelCount(), payload.Orientation(), nullptr);
    if (!owned) return cudaErrorInvalidValue;
    return UploadImpl(std::move(owned), stream, reservation, kind);
}

cudaError_t CudaImage::UploadImpl(std::shared_ptr<const UsdGenImagePayload> payload,
                                  cudaStream_t stream,
                                  UsdGenExecutionMemoryReservation* reservation,
                                  UsdGenExecutionResourceKind kind)
{
    if (!payload || !payload->IsValid()) return cudaErrorInvalidValue;
    if (unproven_) return cudaErrorUnknown;
    cudaError_t status = ReclaimRetired();
    if (status != cudaSuccess) return status;
    std::unique_ptr<HostOwner> nextHost;
    try {
        nextHost = std::make_unique<HostOwner>(std::move(payload));
        if (image_.data()) retired_.reserve(retired_.size() + 1);
    } catch (...) {
        return cudaErrorMemoryAllocation;
    }

    // Always build a private candidate. Reusing image_ would race an earlier
    // upload or sample on another stream and would drop its pageable source
    // before the source's completion event was proven.
    DeviceBuffer<float> nextImage;
    status = nextImage.reset(nextHost->payload->TexelCount(), reservation, kind);
    if (status != cudaSuccess) return status;
    status = cudaMemcpyAsync(nextImage.data(), nextHost->payload->Data(),
                             nextHost->payload->TexelCount() * sizeof(float),
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
        nextImage.quarantine();
        (void)nextHost.release();
        Quarantine();
        return status;
    }
    status = nextImage.recordUse(stream);
    if (status != cudaSuccess) {
        nextImage.quarantine();
        (void)nextHost.release();
        Quarantine();
        return status;
    }

    if (image_.data()) {
        Retired retired;
        retired.host = std::move(host_);
        retired.image = std::move(image_);
        retired_.push_back(std::move(retired));
    }
    image_ = std::move(nextImage);
    host_ = std::move(nextHost);
    unproven_ = false;
    width_ = host_->payload->Width(); height_ = host_->payload->Height();
    channels_ = host_->payload->Channels();
    orientation_ = host_->payload->Orientation();
    return cudaSuccess;
}

cudaError_t CudaImage::ReclaimRetired() noexcept
{
    for (auto it = retired_.begin(); it != retired_.end();) {
        cudaError_t const status = it->image.queryUse();
        if (status == cudaSuccess) it = retired_.erase(it);
        else if (status == cudaErrorNotReady) ++it;
        else {
            Quarantine();
            return status;
        }
    }
    return cudaSuccess;
}

cudaError_t CudaImage::Sample(DeviceView<const float2> uv, DeviceView<float> output,
                              UsdGenImageSampleOptions const& options, cudaStream_t stream) const
{
    if (!IsValid() || output.size != uv.size || (!uv.data && uv.size) ||
        (!output.data && output.size)) return cudaErrorInvalidValue;
    if (!ValidateUsdGenImageSampleOptions(*host_->payload, options, nullptr))
        return cudaErrorInvalidValue;
    if (!uv.size) return cudaSuccess;
    cudaError_t status = const_cast<CudaImage*>(this)->ReclaimRetired();
    if (status != cudaSuccess) return status;
    status = image_.waitOn(stream);
    if (status != cudaSuccess) {
        const_cast<CudaImage*>(this)->Quarantine();
        return status;
    }
    KernelImage image{image_.data(), width_, height_, channels_,
                      static_cast<uint8_t>(orientation_)};
    SampleKernel<<<Blocks(uv.size), 256, 0, stream>>>(image, uv.data, output.data, uv.size, options);
    status = cudaGetLastError();
    // A launch-status failure cannot prove that this invocation did not begin
    // reading the image (for example a pre-existing sticky error can be
    // observed here). Preserve the image owner conservatively until context
    // teardown instead of allowing its destructor to free borrowed storage.
    if (status != cudaSuccess) {
        const_cast<CudaImage*>(this)->Quarantine();
        return status;
    }
    status = const_cast<DeviceBuffer<float>&>(image_).recordUse(stream);
    if (status != cudaSuccess) const_cast<CudaImage*>(this)->Quarantine();
    return status;
}

CudaImage::~CudaImage()
{
    if (unproven_) {
        Quarantine();
        return;
    }
    // DeviceBuffer destruction normally releases storage before the paired
    // HostOwner member. Prove every last-use event first; if CUDA can no
    // longer provide that proof, preserve both sides via quarantine.
    if (image_.synchronizeUse() != cudaSuccess) {
        Quarantine();
        return;
    }
    for (Retired& retired : retired_) {
        if (retired.image.synchronizeUse() != cudaSuccess) {
            Quarantine();
            return;
        }
    }
}

} // namespace usdGen::gpu
