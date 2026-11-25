#ifndef USDGEN_GPU_SEEXPR_NOISE_CUH
#define USDGEN_GPU_SEEXPR_NOISE_CUH
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "usdGenCurves/ops/noise_tables_gen.cuh"

// Shared device implementation of SeExpr's float noise path. Compile with
// --fmad=false to preserve the CPU oracle's float accumulation order.
namespace usdgen_noise {

// Quintic s-curve (SeExpr2::s_curve: double s_curve(double), result assigned
// back to float alphas) — polynomial in double, unsuffixed constants.
__device__ __host__ inline float SCurve(float x)
{
    const double t = x;
    return (float)(t * t * t * (t * (t * 6 - 15) + 10));
}

// Perlin-style lattice hash (SeExpr2::hashReduceChar<3>), 3D scalar form.
__device__ inline unsigned char HashReduceChar(int x, int y, int z)
{
    uint32_t seed = 0;
    static const uint32_t M = 1664525;
    static const uint32_t C = 1013904223;
    const int index[3] = {x, y, z};
    for (int k = 0; k < 3; k++) seed = seed * M + (uint32_t)index[k] + C;
    seed ^= (seed >> 11);
    seed ^= (seed << 7) & 0x9d2c5680UL;
    seed ^= (seed << 15) & 0xefc60000UL;
    seed ^= (seed >> 18);
    return (unsigned char)((((seed & 0xff0000) >> 4) + (seed & 0xff)) & 0xff);
}

// 3D Perlin noise (SeExpr2::noiseHelper<3, float, false>), float path.
__device__ inline float Noise3(float x, float y, float z)
{
    const float X[3] = {x, y, z};
    int index[3];
    float weights[2][3];
    for (int k = 0; k < 3; k++) {
        const float f = floorf(X[k]);
        index[k] = (int)f;
        weights[0][k] = X[k] - f;
        weights[1][k] = weights[0][k] - 1;  // dist to cell with index one above
    }
    float vals[8];
    for (int dummy = 0; dummy < 8; dummy++) {
        int latticeIndex[3];
        int offset[3];
        for (int k = 0; k < 3; k++) {
            offset[k] = ((dummy & (1 << k)) != 0);
            latticeIndex[k] = index[k] + offset[k];
        }
        const int lookup = HashReduceChar(latticeIndex[0], latticeIndex[1], latticeIndex[2]);
        float val = 0;
        for (int k = 0; k < 3; k++) {
            // CPU: `double grad = g[lookup][k]; val += grad * weight;` with
            // float val — double product added in double, one float rounding
            // per accumulation step.
            val = (float)((double)val + (double)G[lookup][k] * (double)weights[offset[k]][k]);
        }
        vals[dummy] = val;
    }
    float alphas[3];
    for (int k = 0; k < 3; k++) alphas[k] = SCurve(weights[0][k]);
    // Multilinear (trilinear) interpolation.
    for (int newd = 2; newd >= 0; newd--) {
        const int newnum = 1 << newd;
        const int k = (3 - newd - 1);
        const float alpha = alphas[k];
        const float beta = 1.0f - alphas[k];
        for (int dummy = 0; dummy < newnum; dummy++) {
            const int index = dummy * (1 << (3 - newd));
            const int otherIndex = index + (1 << k);
            vals[index] = beta * vals[index] + alpha * vals[otherIndex];
        }
    }
    return vals[0];
}

// FBM (SeExpr2::FBM<3, 1, false, float>), non-turbulent.
__device__ inline float Fbm3(float x, float y, float z, int octaves, float lacunarity, float gain)
{
    float P[3];
    P[0] = x;
    P[1] = y;
    P[2] = z;
    float out = 0;
    float scale = 1;
    int octave = 0;
    while (1) {
        out += Noise3(P[0], P[1], P[2]) * scale;
        if (++octave >= octaves) break;
        scale *= gain;
        P[0] = P[0] * lacunarity + 1234.0f;
        P[1] = P[1] * lacunarity + 1234.0f;
        P[2] = P[2] * lacunarity + 1234.0f;
    }
    return out;
}

}  // namespace usdgen_noise
#endif
