// usdGen — CUDA path for UsdGenNoiseOp::Capture (02 §2.7.1, 04 §2.9).
//
// One kernel pins the same fBm field the CPU loop in noise.cpp computes:
//   base[c]   = rootRest[c]*corr + (1-corr)*h_c      (precomputed on the host)
//   perCurve  = 2*fbm3(base[c]) - 1
//   field[i]  = 2*fbm3(base[c] + (0, 0, t[i]*freq)) - 1
// The device math mirrors thirdparty/seexpr/SeExpr2/Noise.cpp exactly:
// hashReduceChar<3>, NOISE_TABLES<3>::g (exact double copy; double grad ×
// double weight with float accumulation rounding as on CPU), the quintic
// s-curve, and the FBM<3,1,false,float> operation order
// (compiled with --fmad=false).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

#include "usdGenCuda/gpu/seexprNoise.cuh"


// One thread per curve. base[c*3+k] is the rest-pinned sample position,
// cvT[c*nCV+i] the CV t-value (hairT or the i/(cvCount-1) ramp).
__global__ void FbmKernel(
    const float * __restrict__ base,
    const float * __restrict__ cvT,
    float freq,
    int octaves,
    float lac,
    float gain,
    int nCurves,
    int nCV,
    float * __restrict__ field,
    float * __restrict__ perCurve)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCurves) return;
    const float bx = base[c * 3 + 0];
    const float by = base[c * 3 + 1];
    const float bz = base[c * 3 + 2];
    perCurve[c] = 2.0f * usdgen_noise::Fbm3(bx, by, bz, octaves, lac, gain) - 1.0f;
    for (int i = 0; i < nCV; i++) {
        const float t = cvT[c * nCV + i];
        field[c * nCV + i] = 2.0f * usdgen_noise::Fbm3(bx, by, bz + t * freq, octaves, lac, gain)
                             - 1.0f;
    }
}

namespace usdGen {

namespace {
// Persistent plumbing, constant-initialized at namespace scope.
cudaStream_t stream = nullptr;
float *dBase = nullptr;   // nCurves*3 floats
float *dT = nullptr;      // n floats
float *dField = nullptr;  // n floats
float *dPcv = nullptr;    // nCurves floats
float *hIn = nullptr;     // pinned: base | cvT
float *hOut = nullptr;    // pinned: field | perCurve
size_t capBase = 0, capT = 0, capField = 0, capPcv = 0;
size_t capIn = 0, capOut = 0;
}  // namespace

// GPU path for UsdGenNoiseOp::Capture. Returns false (CPU fallback) if the
// device, any transfer, or the launch fails.
//
// Persistent plumbing (CUDA1b): stream, device scratch, and pinned staging
// live at namespace scope (above), created once and grown on demand. Capture runs
// sequentially on the scheduler commit thread (topo order), so no locking
// is needed. Per call: memcpy into pinned staging, two async H2D copies,
// the kernel, two async D2H copies, stream sync, memcpy out.
bool UsdGenNoiseCaptureGPU(const float *base,
                           const float *cvT,
                           float frequency,
                           int octaves,
                           float lacunarity,
                           float gain,
                           int nCurves,
                           int nCV,
                           float *field,
                           float *perCurve)
{
    if (nCurves <= 0 || nCV <= 0) return false;
    const size_t n = (size_t)nCurves * (size_t)nCV;
    const size_t nBase = (size_t)nCurves * 3u;


    if (stream == nullptr) {
        int dev = -1;
        if (cudaGetDevice(&dev) != cudaSuccess || dev < 0) return false;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            return false;
    }

    auto grow = [](float **slot, size_t &cap, size_t need, bool pinned) {
        if (need <= cap) return true;
        if (pinned) cudaFreeHost(*slot); else cudaFree(*slot);
        cudaError_t e =
            pinned ? cudaMallocHost(slot, need * sizeof(float))
                   : cudaMalloc(slot, need * sizeof(float));
        if (e != cudaSuccess) {
            *slot = nullptr;
            cap = 0;
            (void)cudaGetLastError();
            return false;
        }
        cap = need;
        return true;
    };
    bool ok = grow(&dBase, capBase, nBase, false) &&
              grow(&dT, capT, n, false) &&
              grow(&dField, capField, n, false) &&
              grow(&dPcv, capPcv, (size_t)nCurves, false) &&
              grow(&hIn, capIn, nBase + n, true) &&
              grow(&hOut, capOut, n + (size_t)nCurves, true);
    if (ok) {
        std::memcpy(hIn, base, nBase * sizeof(float));
        std::memcpy(hIn + nBase, cvT, n * sizeof(float));
        ok =
            cudaMemcpyAsync(dBase, hIn, nBase * sizeof(float), cudaMemcpyHostToDevice, stream) ==
                cudaSuccess &&
            cudaMemcpyAsync(dT, hIn + nBase, n * sizeof(float), cudaMemcpyHostToDevice, stream) ==
                cudaSuccess;
    }
    if (ok) {
        const int block = 128;
        const int grid = (nCurves + block - 1) / block;
        FbmKernel<<<grid, block, 0, stream>>>(dBase, dT, frequency, octaves, lacunarity, gain,
                                              nCurves, nCV, dField, dPcv);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpyAsync(hOut, dField, n * sizeof(float), cudaMemcpyDeviceToHost, stream) ==
                 cudaSuccess &&
             cudaMemcpyAsync(hOut + n, dPcv, (size_t)nCurves * sizeof(float),
                             cudaMemcpyDeviceToHost, stream) == cudaSuccess;
    }
    if (ok) ok = cudaStreamSynchronize(stream) == cudaSuccess;
    if (ok) {
        std::memcpy(field, hOut, n * sizeof(float));
        std::memcpy(perCurve, hOut + n, (size_t)nCurves * sizeof(float));
    }
    if (!ok) (void)cudaGetLastError();  // clear the sticky error for the CPU fallback
    return ok;
}

}  // namespace usdGen
