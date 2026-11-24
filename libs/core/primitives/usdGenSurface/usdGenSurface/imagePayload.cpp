#include "imagePayload.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace usdGen {
namespace {

void Error(std::string* error, char const* message)
{
    if (error) *error = message;
}

bool Finite(float value) { return std::isfinite(value); }

float Wrap(float value, UsdGenImageWrap mode, bool* outside)
{
    *outside = false;
    if (!Finite(value)) { *outside = true; return 0.0f; }
    switch (mode) {
    case UsdGenImageWrap::Clamp: return std::max(0.0f, std::min(1.0f, value));
    case UsdGenImageWrap::Repeat: {
        float result = value - std::floor(value);
        return result == 1.0f ? 0.0f : result;
    }
    case UsdGenImageWrap::Mirror: {
        float result = value - 2.0f * std::floor(value * 0.5f);
        return result <= 1.0f ? result : 2.0f - result;
    }
    case UsdGenImageWrap::Black:
        if (value < 0.0f || value > 1.0f) *outside = true;
        return value;
    }
    *outside = true;
    return 0.0f;
}

int Index(int value, int extent, UsdGenImageWrap mode, bool* outside)
{
    if (value >= 0 && value < extent) return value;
    switch (mode) {
    case UsdGenImageWrap::Clamp: return std::max(0, std::min(extent - 1, value));
    case UsdGenImageWrap::Repeat: {
        int result = value % extent;
        return result < 0 ? result + extent : result;
    }
    case UsdGenImageWrap::Mirror: {
        int64_t const period = int64_t(extent) * 2;
        int64_t result = int64_t(value) % period;
        if (result < 0) result += period;
        return static_cast<int>(result < extent ? result : period - 1 - result);
    }
    case UsdGenImageWrap::Black: *outside = true; return 0;
    }
    *outside = true;
    return 0;
}

float Texel(UsdGenImagePayload const& image, int x, int y,
            UsdGenImageSampleOptions const& options, bool* outside)
{
    int const width = static_cast<int>(image.Width());
    int const height = static_cast<int>(image.Height());
    bool xOutside = false, yOutside = false;
    int const ix = Index(x, width, options.wrap, &xOutside);
    int const iy = Index(y, height, options.wrap, &yOutside);
    if (xOutside || yOutside) { *outside = true; return options.defaultValue; }
    uint32_t channel = static_cast<uint32_t>(options.channel);
    uint32_t const channels = image.Channels();
    uint32_t const row = image.Orientation() == UsdGenImageRowOrientation::TopDown
        ? image.Height() - 1u - static_cast<uint32_t>(iy) : static_cast<uint32_t>(iy);
    size_t const base = (static_cast<size_t>(row) * image.Width() +
                         static_cast<size_t>(ix)) * channels;
    if (options.channel == UsdGenImageChannel::Luminance) {
        return image.Data()[base] * 0.2126f + image.Data()[base + 1] * 0.7152f +
               image.Data()[base + 2] * 0.0722f;
    }
    return image.Data()[base + channel];
}

float RawSample(UsdGenImagePayload const& image, float u, float v,
                UsdGenImageSampleOptions const& options)
{
    bool uOutside = false, vOutside = false;
    u = Wrap(u, options.wrap, &uOutside);
    v = Wrap(v, options.wrap, &vOutside);
    if (uOutside || vOutside) return options.defaultValue;
    float x = u * static_cast<float>(image.Width() - 1u);
    float y = v * static_cast<float>(image.Height() - 1u);
    if (options.filter == UsdGenImageFilter::Nearest) {
        bool outside = false;
        return Texel(image, static_cast<int>(std::floor(x + 0.5f)),
                     static_cast<int>(std::floor(y + 0.5f)), options, &outside);
    }
    int const x0 = static_cast<int>(std::floor(x));
    int const y0 = static_cast<int>(std::floor(y));
    int const x1 = x0 + 1;
    int const y1 = y0 + 1;
    float const fx = x - static_cast<float>(x0);
    float const fy = y - static_cast<float>(y0);
    bool outside = false;
    float const a = Texel(image, x0, y0, options, &outside);
    float const b = Texel(image, x1, y0, options, &outside);
    float const c = Texel(image, x0, y1, options, &outside);
    float const d = Texel(image, x1, y1, options, &outside);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) -
                                 (a + (b - a) * fx)) * fy;
}

} // namespace

UsdGenImagePayload::UsdGenImagePayload(
    uint32_t width, uint32_t height, uint32_t channels,
    UsdGenImageRowOrientation orientation,
    std::shared_ptr<const std::vector<float>> storage) noexcept
    : width_(width), height_(height), channels_(channels),
      orientation_(orientation), storage_(std::move(storage)) {}

std::shared_ptr<const UsdGenImagePayload> UsdGenImagePayload::Create(
    uint32_t width, uint32_t height, uint32_t channels, std::vector<float> texels,
    UsdGenImageRowOrientation orientation, std::string* error)
{
    if (!width || !height || !channels || channels > 4 ||
        width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        Error(error, "image dimensions must be nonzero and channels must be in [1,4]");
        return {};
    }
    if (height > std::numeric_limits<size_t>::max() / width ||
        size_t(width) * height > std::numeric_limits<size_t>::max() / channels ||
        texels.size() != size_t(width) * height * channels) {
        Error(error, "image texel count does not match dimensions");
        return {};
    }
    for (float value : texels) {
        if (!Finite(value)) {
            Error(error, "image texels must be finite");
            return {};
        }
    }
    auto storage = std::make_shared<const std::vector<float>>(std::move(texels));
    return std::shared_ptr<const UsdGenImagePayload>(new UsdGenImagePayload(
        width, height, channels, orientation, std::move(storage)));
}

std::shared_ptr<const UsdGenImagePayload> UsdGenImagePayload::Create(
    uint32_t width, uint32_t height, uint32_t channels, float const* texels,
    size_t texelCount, UsdGenImageRowOrientation orientation, std::string* error)
{
    if (!texels) {
        Error(error, "null image texel pointer");
        return {};
    }
    if (!width || !height || !channels || channels > 4 ||
        width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        height > std::numeric_limits<size_t>::max() / width ||
        size_t(width) * height > std::numeric_limits<size_t>::max() / channels) {
        Error(error, "image dimensions are invalid or overflow host indexing");
        return {};
    }
    size_t const required = size_t(width) * height * channels;
    if (texelCount != required) {
        Error(error, "image texel count does not match dimensions");
        return {};
    }
    return Create(width, height, channels,
                  std::vector<float>(texels, texels + texelCount), orientation, error);
}

bool ValidateUsdGenImageSampleOptions(
    UsdGenImagePayload const& image, UsdGenImageSampleOptions const& options,
    std::string* error) noexcept
{
    if (!image.IsValid()) { Error(error, "invalid image payload"); return false; }
    uint32_t const channel = static_cast<uint32_t>(options.channel);
    if (options.channel == UsdGenImageChannel::Luminance) {
        if (image.Channels() < 3) { Error(error, "luminance requires at least 3 channels"); return false; }
    } else if (channel >= image.Channels()) {
        Error(error, "requested image channel is not present"); return false;
    }
    if (!Finite(options.scale) || !Finite(options.offset) || !Finite(options.defaultValue) ||
        !Finite(options.outputMin) || !Finite(options.outputMax) ||
        (options.clampOutput && options.outputMin > options.outputMax)) {
        Error(error, "image sample options contain non-finite or reversed limits");
        return false;
    }
    return true;
}

float UsdGenImageSampler::Sample(UsdGenImagePayload const& image, float u, float v,
                                 UsdGenImageSampleOptions const& options) noexcept
{
    if (!ValidateUsdGenImageSampleOptions(image, options, nullptr)) return options.defaultValue;
    float value = RawSample(image, u, v, options);
    value = value * options.scale + options.offset;
    if (options.clampOutput) value = std::max(options.outputMin, std::min(options.outputMax, value));
    return value;
}

void UsdGenImageSampler::Sample(UsdGenImagePayload const& image, float const* uv,
                                size_t count, float* output,
                                UsdGenImageSampleOptions const& options) noexcept
{
    if (!output || (!uv && count)) return;
    if (count > std::numeric_limits<size_t>::max() / 2u) return;
    for (size_t i = 0; i != count; ++i)
        output[i] = Sample(image, uv[2 * i], uv[2 * i + 1], options);
}

} // namespace usdGen
