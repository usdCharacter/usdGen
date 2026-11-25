// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_MAP_VK_IMAGE_H
#define USDGEN_VULKAN_MAP_VK_IMAGE_H

// Vulkan twin of gpu/imageSampler.h (CudaImage::Upload/Sample): host upload
// of an immutable UsdGenImagePayload into a charged texel buffer plus a
// compute dispatch that samples it with the exact CUDA scalar semantics.
// Bitwise parity with UsdGenImageSampler::Sample is the contract (the CUDA
// test compares with memcmp), so filtering stays manual in mapVkSample.comp
// with explicit fma() at the five sites the oracle build fuses; Vulkan
// hardware samplers cannot provide that bit-exactness.

#include "chargedBuffer.h"
#include "usdGen/imagePayload.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

struct MapVkImageView {
    std::shared_ptr<const ChargedBuffer> texels;
    std::shared_ptr<const UsdGenImagePayload> payload;
};

// Queue-owner confined. Upload copies immutable host texels into a charged
// device buffer; the payload is retained so options validation and texel
// bytes stay alive for every later sample.
class MapVkImage final {
public:
    static std::shared_ptr<MapVkImage> Create(std::shared_ptr<DeviceContext> const&,
        std::shared_ptr<const UsdGenImagePayload> const&, VkResult* = nullptr);
    ~MapVkImage() = default;
    MapVkImage(MapVkImage const&) = delete;
    MapVkImage& operator=(MapVkImage const&) = delete;

    MapVkImageView View() const noexcept;
    std::shared_ptr<DeviceContext> const& context() const noexcept { return context_; }
    uint32_t Width() const noexcept;
    uint32_t Height() const noexcept;
    uint32_t Channels() const noexcept;

private:
    MapVkImage(std::shared_ptr<DeviceContext>, std::shared_ptr<const UsdGenImagePayload>,
        std::shared_ptr<ChargedBuffer>);
    std::shared_ptr<DeviceContext> context_;
    std::shared_ptr<const UsdGenImagePayload> payload_;
    std::shared_ptr<ChargedBuffer> texels_;
};

// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion.
class MapVkSamplePipeline final : public std::enable_shared_from_this<MapVkSamplePipeline> {
public:
    class Candidate;
    // Owner-side admission checkpoint immediately before the first native
    // submit. It runs at most once only for a fully recorded nonempty
    // candidate; invalid input, allocation/recording failures, and count==0
    // never invoke it. True precedes pending state and every native submit.
    // False or an exception rejects without arming or submitting the candidate.
    using BeforeSubmit = std::function<bool()>;
    // SPIR-V must be the trusted, externally spirv-val-validated
    // mapVkSample.comp module with the documented four-binding/56-byte
    // push-constant ABI.
    static std::shared_ptr<MapVkSamplePipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~MapVkSamplePipeline();
    // Exact native identity is available before candidate allocation/submit.
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // `uv` holds count float2 texel coordinates (8 bytes each). Options are
    // validated against the image exactly like CudaImage::Sample (including
    // the luminance channel rule); count==0 short-circuits without dispatch.
    std::unique_ptr<Candidate> Begin(MapVkImageView image,
        std::shared_ptr<const ChargedBuffer> uv, uint32_t count,
        UsdGenImageSampleOptions const& options, VkResult* result = nullptr,
        BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit MapVkSamplePipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginInternal(MapVkImageView,
        std::shared_ptr<const ChargedBuffer>, uint32_t,
        UsdGenImageSampleOptions const&, VkResult*, BeforeSubmit);
    std::shared_ptr<Native> native_;
};

class MapVkSamplePipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained; semanticStatus must additionally
    // be zero before output is exposed. VK_NOT_READY preserves the candidate.
    VkResult Poll(uint32_t* semanticStatus = nullptr);
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> uvOwner() const noexcept;
    MapVkImageView imageOwner() const noexcept;
    uint32_t count() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class MapVkSamplePipeline;
};

} // namespace usdGen::vulkan

#endif
