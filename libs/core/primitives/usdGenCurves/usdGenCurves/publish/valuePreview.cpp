// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/publish/valuePreview.h"
#include "usdGen/registries/token.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

// Made at load, not on first use: no function-local static to guard.
TfToken const kColorRgb = UsdGenInternToken("rgb"), kColorIds = UsdGenInternToken("ids"),
    kColorGray = UsdGenInternToken("gray"), kColorViridis = UsdGenInternToken("viridis");
TfToken const kShadingFlat = UsdGenInternToken("flat");

double Clamp01(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }

float SrgbToLinear(double c)
{
    c = Clamp01(c);
    return float(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4));
}

GfVec3f SrgbToLinear(GfVec3f const &c)
{
    return GfVec3f(SrgbToLinear(c[0]), SrgbToLinear(c[1]), SrgbToLinear(c[2]));
}

// Eleven evenly spaced sRGB stops (matplotlib's viridis and inferno).
using Stops = unsigned char const[11][3];
Stops kViridis = {
    {0x44, 0x01, 0x54}, {0x48, 0x24, 0x75}, {0x41, 0x44, 0x87}, {0x35, 0x5f, 0x8d},
    {0x2a, 0x78, 0x8e}, {0x21, 0x91, 0x8c}, {0x22, 0xa8, 0x84}, {0x44, 0xbf, 0x70},
    {0x7a, 0xd1, 0x51}, {0xbd, 0xdf, 0x26}, {0xfd, 0xe7, 0x25}};
Stops kInferno = {
    {0x00, 0x00, 0x04}, {0x16, 0x0b, 0x39}, {0x42, 0x0a, 0x68}, {0x6a, 0x17, 0x6e},
    {0x93, 0x26, 0x67}, {0xbc, 0x37, 0x54}, {0xdd, 0x51, 0x3a}, {0xf3, 0x78, 0x19},
    {0xfc, 0xa5, 0x0a}, {0xf6, 0xd7, 0x46}, {0xfc, 0xff, 0xa4}};

GfVec3f Ramp(Stops const &stops, double t)
{
    double const x = Clamp01(t) * 10.0;
    int const i = std::min(9, int(x));
    double const f = x - i;
    GfVec3f out;
    for (int k = 0; k < 3; ++k)
        out[k] = float((stops[i][k] * (1.0 - f) + stops[i + 1][k] * f) / 255.0);
    return out;
}

// One colour per distinct value, keyed in 1/1024 steps like opUtil::RegionKey
// so strands an operator treats as one region or clump share a colour.
GfVec3f IdColor(double v)
{
    double const scaled = std::max(-9.0e18, std::min(9.0e18, v * 1024.0));
    uint64_t x = uint64_t(std::llround(scaled)) + 0x9e3779b97f4a7c15ULL;  // splitmix64
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    double const h = double(x & 0xffff) / 65536.0 * 6.0;
    double const s = 0.55 + 0.4 * double((x >> 16) & 0xff) / 255.0;
    double const value = 0.7 + 0.3 * double((x >> 24) & 0xff) / 255.0;
    double const f = h - std::floor(h);
    float const p = float(value * (1.0 - s));
    float const q = float(value * (1.0 - s * f));
    float const t = float(value * (1.0 - s * (1.0 - f)));
    float const w = float(value);
    switch (int(h) % 6) {
    case 0: return GfVec3f(w, t, p);
    case 1: return GfVec3f(q, w, p);
    case 2: return GfVec3f(p, w, t);
    case 3: return GfVec3f(p, q, w);
    case 4: return GfVec3f(t, p, w);
    default: return GfVec3f(w, p, q);
    }
}

} // namespace

GfVec3f UsdGenPreviewColor(TfToken const &colorMap, GfVec2f const &range,
                           double const *values, unsigned components)
{
    TfToken const &rgb = kColorRgb, &ids = kColorIds, &gray = kColorGray, &viridis = kColorViridis;
    if (!values || components == 0) return UsdGenPreviewMissingColor();
    for (unsigned k = 0; k < components; ++k)
        if (!std::isfinite(values[k])) return GfVec3f(1.0f, 0.0f, 1.0f);
    double const lo = range[0], hi = range[1];
    auto normalise = [&](double v) {
        if (hi == lo) return v >= hi ? 1.0 : 0.0;
        return Clamp01((v - lo) / (hi - lo));
    };
    if (colorMap == rgb) {
        GfVec3f out(0.0f);
        for (unsigned k = 0; k < 3; ++k)
            out[k] = float(normalise(components == 1 ? values[0]
                                     : k < components ? values[k] : lo));
        return out;
    }
    double v = values[0];
    if (components > 1) {
        double sum = 0.0;
        for (unsigned k = 0; k < components; ++k) sum += values[k] * values[k];
        v = std::sqrt(sum);
    }
    if (colorMap == ids) return SrgbToLinear(IdColor(v));
    double const t = normalise(v);
    if (colorMap == gray) return SrgbToLinear(GfVec3f(float(t)));
    return SrgbToLinear(Ramp(colorMap == viridis ? kViridis : kInferno, t));
}

GfVec3f UsdGenPreviewMissingColor()
{
    return GfVec3f(0.02f, 0.02f, 0.02f);
}

UsdGenPublishRole UsdGenPreviewMaterialRole(TfToken const &shading)
{
    return shading == kShadingFlat ? UsdGenPublishRole::PreviewMaterialFlat
                                   : UsdGenPublishRole::PreviewMaterial;
}

} // namespace usdGen
