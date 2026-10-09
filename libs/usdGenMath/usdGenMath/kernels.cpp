// usdGenMath — deterministic float kernel implementations (review M-3).
//
// Compiled with -ffp-contract=off (root CMakeLists: usdGenMath is the only
// target with the flag), so no FMA contraction can make these results depend
// on translation unit, optimization path, or thread count. Keep every
// function out-of-line and free of caller-visible inlining decisions.
#include "usdGenMath/usdGenMath/kernels.h"

#include <cmath>

namespace usdGen {

float UsdGenEvalLut257(const float *lut, float t)
{
    float u = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    u *= 256.0f;
    float const fj = u < 255.0f ? std::floor(u) : 255.0f;
    const int j = int(fj);
    const float a = u - fj;
    return lut[j] * (1.0f - a) + lut[j + 1] * a;
}

std::uint64_t UsdGenMortonInterleave(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    // Scatter each 21-bit axis value so one zero bit separates adjacent bits
    // (magic-mask form: identical bits to the 21-iteration loop at ~5x the
    // throughput; the masks are proven by exhaustive comparison over all
    // 2^21 inputs in testUsdGenMorton).
    auto split3 = [](std::uint32_t v) -> std::uint64_t {
        std::uint64_t s = v & 0x1FFFFFull;
        s = (s | (s << 32)) & 0x001F00000000FFFFull;
        s = (s | (s << 16)) & 0x001F0000FF0000FFull;
        s = (s | (s << 8)) & 0x100F00F00F00F00Full;
        s = (s | (s << 4)) & 0x10C30C30C30C30C3ull;
        s = (s | (s << 2)) & 0x1249249249249249ull;
        return s;
    };
    return split3(x) | (split3(y) << 1) | (split3(z) << 2);
}

std::uint64_t UsdGenMortonKey3(float x, float y, float z, float cellScale)
{
    static constexpr int kMax = (1 << 20) - 1;  // +2^20-1, bias window [-2^20, 2^20-1]
    auto quant = [cellScale](float v) -> std::uint32_t {
        double q = std::floor(double(v) * double(cellScale));
        if (q < double(-(1 << 20))) q = double(-(1 << 20));
        if (q > double(kMax)) q = double(kMax);
        return std::uint32_t(std::int64_t(q) + (1 << 20));  // -> [0, 2^21)
    };
    return UsdGenMortonInterleave(quant(x), quant(y), quant(z));
}

float UsdGenTriangleArea(
    float ax, float ay, float az,
    float bx, float by, float bz,
    float cx, float cy, float cz)
{
    const float ux = bx - ax, uy = by - ay, uz = bz - az;
    const float vx = cx - ax, vy = cy - ay, vz = cz - az;
    const float nx = uy * vz - uz * vy;
    const float ny = uz * vx - ux * vz;
    const float nz = ux * vy - uy * vx;
    return 0.5f * std::sqrt(nx * nx + ny * ny + nz * nz);
}

float UsdGenPolygonRestArea(
    const float *px, const float *py, const float *pz,
    std::size_t p0, std::size_t n)
{
    if (n < 3) return 0.0f;
    const float x0 = px[p0], y0 = py[p0], z0 = pz[p0];
    double area = 0.0;  // accumulate in double: stable, no catastrophic loss
    for (std::size_t i = 1; i + 1 < n; ++i) {
        const float ux = px[p0 + i] - x0, uy = py[p0 + i] - y0, uz = pz[p0 + i] - z0;
        const float vx = px[p0 + i + 1] - x0, vy = py[p0 + i + 1] - y0, vz = pz[p0 + i + 1] - z0;
        const float nx = uy * vz - uz * vy;
        const float ny = uz * vx - ux * vz;
        const float nz = ux * vy - uy * vx;
        area += 0.5 * std::sqrt(double(nx) * nx + double(ny) * ny + double(nz) * nz);
    }
    return float(area);
}

float UsdGenPolylineRestLength(
    const float *px, const float *py, const float *pz,
    std::size_t base, std::size_t cv)
{
    double len = 0.0;
    for (std::size_t i = 0; i + 1 < cv; ++i) {
        const float dx = px[base + i + 1] - px[base + i];
        const float dy = py[base + i + 1] - py[base + i];
        const float dz = pz[base + i + 1] - pz[base + i];
        len += std::sqrt(double(dx) * dx + double(dy) * dy + double(dz) * dz);
    }
    return float(len);
}

}  // namespace usdGen
