// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenUsd/presentation.h"

#include <cstddef>
#include <cstring>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenUsd {

namespace {

struct Fnv {
    uint64_t h = 1469598103934665603ULL;
    void Bytes(void const *data, size_t n)
    {
        auto const *p = static_cast<unsigned char const *>(data);
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    void U64(uint64_t v) { Bytes(&v, sizeof(v)); }
    void F32(float v)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        Bytes(&bits, sizeof(bits));
    }
    void Vec3(GfVec3f const &v) { F32(v[0]); F32(v[1]); F32(v[2]); }
    // Length first: adjacent texts cannot alias.
    void Text(TfToken const &t)
    {
        std::string const &s = t.GetString();
        U64(s.size());
        Bytes(s.data(), s.size());
    }
};

}  // namespace

uint64_t PresentationDigest(UsdGenPresentation const &p)
{
    Fnv f;
    usdGen::UsdGenLookDesc const &look = p.look;
    f.Vec3(look.rootColor);
    f.Vec3(look.tipColor);
    f.U64(look.rampColors.size());
    for (GfVec3f const &c : look.rampColors) f.Vec3(c);
    f.U64(look.rampPositions.size());
    for (float x : look.rampPositions) f.F32(x);
    f.Text(look.rampInterpolation);
    f.F32(look.rampExponent);
    f.Text(look.bakeMode);
    f.Text(look.bakeTarget);
    f.Text(look.bakePrimvar);
    f.F32(look.hueJitter);
    f.F32(look.valueJitter);
    f.U64(uint64_t(int64_t(look.jitterSeed)));
    f.Text(p.purpose);
    f.Text(p.visibility);
    f.U64(p.material.value);
    f.Text(p.basis);
    f.Text(p.previewColorMap);
    f.F32(p.previewRange[0]);
    f.F32(p.previewRange[1]);
    f.Text(p.previewShading);
    f.U64(p.description.value);
    f.U64(p.dependency.value);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) {
            double const v = p.xformMatrix[r][c];
            f.Bytes(&v, sizeof(v));
        }
    return f.h;
}

}  // namespace usdGenUsd
