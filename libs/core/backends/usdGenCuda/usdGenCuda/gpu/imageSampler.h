// CUDA-owned image upload and scalar sampling.  This is deliberately a
// low-level borrowed-view primitive; graph/session wiring belongs elsewhere.
#ifndef USDGEN_GPU_IMAGE_SAMPLER_H
#define USDGEN_GPU_IMAGE_SAMPLER_H

#include "deviceBuffer.h"
#include "usdGen/imagePayload.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace usdGen::gpu {

struct CudaImageView {
    DeviceView<const float> texels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    UsdGenImageRowOrientation orientation = UsdGenImageRowOrientation::TopDown;
};

class CudaImage final {
public:
    CudaImage() = default;
    ~CudaImage();
    CudaImage(CudaImage const&) = delete;
    CudaImage& operator=(CudaImage const&) = delete;
    // Native work and pageable source owners form one retirement transaction;
    // moving members independently could release a source before its event.
    CudaImage(CudaImage&&) = delete;
    CudaImage& operator=(CudaImage&&) = delete;

    // The immutable host payload is retained until its upload is proven
    // complete (or until destruction), making pageable-host cudaMemcpyAsync
    // source lifetime safe across overlapping uploads.
    cudaError_t Upload(std::shared_ptr<const UsdGenImagePayload> const& payload,
                       cudaStream_t stream = nullptr,
                       UsdGenExecutionMemoryReservation* reservation = nullptr,
                       UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Active);
    cudaError_t Upload(UsdGenImagePayload const& payload, cudaStream_t stream = nullptr,
                       UsdGenExecutionMemoryReservation* reservation = nullptr,
                       UsdGenExecutionResourceKind kind = UsdGenExecutionResourceKind::Active);

    // `uv` and `output` are borrowed. Their owner must record the same stream
    // on success and quarantine them on any post-submission failure; this
    // object applies that proof-or-quarantine rule to its owned image only.
    cudaError_t Sample(DeviceView<const float2> uv, DeviceView<float> output,
                       UsdGenImageSampleOptions const& options,
                       cudaStream_t stream = nullptr) const;
    cudaError_t Sample(DeviceView<float2> uv, DeviceView<float> output,
                       UsdGenImageSampleOptions const& options,
                       cudaStream_t stream = nullptr) const {
        return Sample(DeviceView<const float2>{uv.data, uv.size}, output, options, stream);
    }
    cudaError_t Sample(float2 const* uv, size_t count, float* output,
                       UsdGenImageSampleOptions const& options,
                       cudaStream_t stream = nullptr) const {
        return Sample({uv, count}, {output, count}, options, stream);
    }

    CudaImageView View() const noexcept {
        return {image_.view(), width_, height_, channels_, orientation_};
    }
    bool IsValid() const noexcept { return image_.data() != nullptr && !unproven_; }
    bool HasUnprovenWork() const noexcept { return unproven_; }
    uint32_t Width() const noexcept { return width_; }
    uint32_t Height() const noexcept { return height_; }
    uint32_t Channels() const noexcept { return channels_; }
    void Reclassify(UsdGenExecutionResourceKind kind) noexcept { image_.Reclassify(kind); }

    // Call only when completion/free cannot be proven (for example after a
    // lost CUDA context). The permit is intentionally retained conservatively.
    void Quarantine() noexcept {
        image_.quarantine();
        for (Retired& retired : retired_) {
            retired.image.quarantine();
            (void)retired.host.release();
        }
        // The keeper nodes were allocated before submission. Leaking them is
        // the allocation-free failure path that retains pageable CUDA copy
        // sources until process/context teardown.
        (void)host_.release();
        unproven_ = true; width_ = height_ = channels_ = 0;
    }

private:
    struct HostOwner {
        explicit HostOwner(std::shared_ptr<const UsdGenImagePayload> value)
            : payload(std::move(value)) {}
        std::shared_ptr<const UsdGenImagePayload> payload;
    };
    struct Retired {
        // Declare the host owner before the device owner so destruction frees
        // device storage while its pageable source remains alive.
        std::unique_ptr<HostOwner> host;
        DeviceBuffer<float> image;
    };

    cudaError_t ReclaimRetired() noexcept;
    cudaError_t UploadImpl(std::shared_ptr<const UsdGenImagePayload> payload,
                           cudaStream_t stream,
                           UsdGenExecutionMemoryReservation* reservation,
                           UsdGenExecutionResourceKind kind);
    std::vector<Retired> retired_;
    std::unique_ptr<HostOwner> host_;
    DeviceBuffer<float> image_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t channels_ = 0;
    UsdGenImageRowOrientation orientation_ = UsdGenImageRowOrientation::TopDown;
    bool unproven_ = false;
};

using DeviceImage = CudaImage;

} // namespace usdGen::gpu

#endif // USDGEN_GPU_IMAGE_SAMPLER_H
