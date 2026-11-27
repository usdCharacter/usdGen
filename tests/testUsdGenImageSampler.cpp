#include "usdGenSurface/imagePayload.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace usdGen;

namespace {
int failures = 0;
void Check(bool value, char const* text) {
    if (!value) { ++failures; std::fprintf(stderr, "FAIL: %s\n", text); }
}
bool Near(float a, float b) { return std::fabs(a - b) < 1e-6f; }
}

int main()
{
    // Top-down source: UV bottom-left sees the last source row.
    std::string error;
    auto image = ImagePayload::Create(2, 2, 4,
        std::vector<float>{1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16},
        UsdGenImageRowOrientation::TopDown, &error);
    Check(image && error.empty(), "valid immutable payload creates");
    Check(image->Data()[0] == 1 && image->TexelCount() == 16, "payload data is available read-only");
    auto sharedImage = image;
    auto sharedStorage = image->Storage();
    Check(sharedImage->Storage() == sharedStorage &&
              sharedImage->Data() == image->Data(),
          "immutable image payload copies share COW storage");

    ImageSampleOptions options;
    options.filter = UsdGenImageFilter::Nearest;
    options.channel = UsdGenImageChannel::R;
    options.clampOutput = false;
    Check(Near(ImageSampler::Sample(*image, 0, 0, options), 9), "explicit top-down orientation");
    Check(Near(ImageSampler::Sample(*image, 1, 1, options), 5), "nearest upper corner");

    options.filter = UsdGenImageFilter::Bilinear;
    options.channel = UsdGenImageChannel::Luminance;
    Check(Near(ImageSampler::Sample(*image, .5f, .5f, options),
               (1+5+9+13)*.25f*.2126f + (2+6+10+14)*.25f*.7152f +
               (3+7+11+15)*.25f*.0722f), "bilinear luminance parity");

    options.channel = UsdGenImageChannel::R;
    options.wrap = UsdGenImageWrap::Black;
    options.defaultValue = 7.0f;
    options.clampOutput = false;
    Check(Near(ImageSampler::Sample(*image, -0.1f, .5f, options), 7), "black wrap default");
    options.wrap = UsdGenImageWrap::Repeat;
    Check(Near(ImageSampler::Sample(*image, 1.0f, 0.0f, options), 9), "repeat wraps one to zero");
    options.wrap = UsdGenImageWrap::Mirror;
    Check(Near(ImageSampler::Sample(*image, -0.0f, 0.0f, options), 9), "mirror zero");

    options.filter = UsdGenImageFilter::Nearest;
    options.scale = 2; options.offset = -20; options.clampOutput = true;
    options.outputMin = 0; options.outputMax = 1;
    Check(Near(ImageSampler::Sample(*image, 0, 0, options), 0), "scale offset output clamp");
    options.clampOutput = false;
    Check(Near(ImageSampler::Sample(*image, 0, 0, options), -2), "output clamp can be disabled");

    auto badDims = ImagePayload::Create(0, 2, 1, std::vector<float>{},
                                        UsdGenImageRowOrientation::BottomUp, &error);
    Check(!badDims && !error.empty(), "invalid dimensions rejected");
    auto badChannels = ImagePayload::Create(2, 2, 1, std::vector<float>{1,2,3,4},
                                            UsdGenImageRowOrientation::BottomUp);
    options.channel = UsdGenImageChannel::Luminance;
    Check(!ValidateUsdGenImageSampleOptions(*badChannels, options, &error) &&
              error.find("luminance") != std::string::npos,
          "luminance validation names missing channels");
    options.channel = UsdGenImageChannel::R;
    options.outputMin = 2; options.outputMax = 1;
    options.clampOutput = true;
    Check(!ValidateUsdGenImageSampleOptions(*badChannels, options, nullptr), "reversed output limits rejected");

    // The implementation uses signed indexing internally; reject payloads
    // which cannot be represented rather than accepting them and narrowing.
    auto tooWide = ImagePayload::Create(
        static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u, 1, 1,
        std::vector<float>{1}, UsdGenImageRowOrientation::BottomUp, &error);
    Check(!tooWide, "dimensions beyond signed indexing range rejected");

    float one = 1.0f;
    auto badPointerCount = ImagePayload::Create(1, 1, 1, &one,
        std::numeric_limits<size_t>::max(), UsdGenImageRowOrientation::BottomUp, &error);
    Check(!badPointerCount, "pointer texel-count overflow rejected before copying");

    std::vector<float> untouched(1, 42.0f);
    ImageSampler::Sample(*image, nullptr, 0, untouched.data(), options);
    Check(untouched[0] == 42.0f, "empty batch accepts null UV input");
    return failures ? 1 : 0;
}
