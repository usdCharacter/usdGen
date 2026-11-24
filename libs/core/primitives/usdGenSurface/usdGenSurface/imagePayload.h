// Backend-neutral immutable image storage and deterministic scalar sampling.
// This header intentionally has no USD, Hio, or CUDA dependency.
#ifndef USDGEN_IMAGE_PAYLOAD_H
#define USDGEN_IMAGE_PAYLOAD_H

#include "usdGen/api/export.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

enum class UsdGenImageRowOrientation : uint8_t {
    // A payload row is ordered from the top of the image to the bottom.
    TopDown,
    // A payload row is ordered from the bottom of the image to the top.
    BottomUp
};

enum class UsdGenImageFilter : uint8_t { Nearest, Bilinear };
enum class UsdGenImageWrap : uint8_t { Clamp, Repeat, Mirror, Black };
enum class UsdGenImageChannel : uint8_t { R, G, B, A, Luminance };

struct UsdGenImageSampleOptions {
    UsdGenImageFilter filter = UsdGenImageFilter::Bilinear;
    UsdGenImageWrap wrap = UsdGenImageWrap::Clamp;
    UsdGenImageChannel channel = UsdGenImageChannel::R;
    float scale = 1.0f;
    float offset = 0.0f;
    bool clampOutput = true;
    float outputMin = 0.0f;
    float outputMax = 1.0f;
    float defaultValue = 0.0f;
};

// The source texels are immutable float values (linear or otherwise; the
// payload does not impose a color transform). Keeping this value immutable
// makes it safe to share one decoded image across capture workers
// and across a CPU sampler and a CUDA upload.
class USDGEN_CORE_API UsdGenImagePayload final {
public:
    UsdGenImagePayload() = default;

    // Copies texels into immutable storage. `texels` is row-major and has
    // width*height*channels entries. Row zero's interpretation is explicit
    // through `orientation`; UV (0,0) is the bottom-left corner.
    static std::shared_ptr<const UsdGenImagePayload> Create(
        uint32_t width, uint32_t height, uint32_t channels,
        std::vector<float> texels,
        UsdGenImageRowOrientation orientation = UsdGenImageRowOrientation::TopDown,
        std::string* error = nullptr);

    static std::shared_ptr<const UsdGenImagePayload> Create(
        uint32_t width, uint32_t height, uint32_t channels,
        float const* texels, size_t texelCount,
        UsdGenImageRowOrientation orientation = UsdGenImageRowOrientation::TopDown,
        std::string* error = nullptr);

    bool IsValid() const noexcept { return static_cast<bool>(storage_); }
    uint32_t Width() const noexcept { return width_; }
    uint32_t Height() const noexcept { return height_; }
    uint32_t Channels() const noexcept { return channels_; }
    UsdGenImageRowOrientation Orientation() const noexcept { return orientation_; }
    size_t TexelCount() const noexcept { return storage_ ? storage_->size() : 0; }
    float const* Data() const noexcept { return storage_ ? storage_->data() : nullptr; }
    std::shared_ptr<const std::vector<float>> Storage() const noexcept { return storage_; }

private:
    UsdGenImagePayload(uint32_t width, uint32_t height, uint32_t channels,
                       UsdGenImageRowOrientation orientation,
                       std::shared_ptr<const std::vector<float>> storage) noexcept;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t channels_ = 0;
    UsdGenImageRowOrientation orientation_ = UsdGenImageRowOrientation::TopDown;
    std::shared_ptr<const std::vector<float>> storage_;
};

USDGEN_CORE_API bool ValidateUsdGenImageSampleOptions(
    UsdGenImagePayload const&, UsdGenImageSampleOptions const&, std::string* error = nullptr) noexcept;

class USDGEN_CORE_API UsdGenImageSampler final {
public:
    static float Sample(UsdGenImagePayload const&, float u, float v,
                        UsdGenImageSampleOptions const& = {}) noexcept;
    static void Sample(UsdGenImagePayload const&, float const* uv, size_t count,
                       float* output,
                       UsdGenImageSampleOptions const& = {}) noexcept;
};

// Short aliases are kept for backend adapters and callers that do not use the
// longer schema-oriented names.
using ImagePayload = UsdGenImagePayload;
using ImageSampleOptions = UsdGenImageSampleOptions;
using ImageSampler = UsdGenImageSampler;

} // namespace usdGen

#endif // USDGEN_IMAGE_PAYLOAD_H
