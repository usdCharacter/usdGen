// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesCuda's registration: the CUDA kernel rows of the built-in curve
// ops (the capability matrix reads them in every build) and the curves' CUDA
// device program. Loaded after usdGenCuda (usdGenExtensions.txt), whose
// backend it names, and after usdGenCurves, whose op ids the rows attach to.
#include "usdGenCurves/cuda/export.h"
#include "usdGenCurves/cuda/cudaStages.h"
#include "usdGenCurves/cuda/program.h"
#include "usdGenCurves/ops/kernelOps.h"
#include "usdGenCurves/ops/noiseGpuHook.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

using namespace usdGen;

#if defined(USDGEN_USE_GPU_NOISE)
namespace usdGen {
bool UsdGenNoiseCaptureGPU(float const *base, float const *cvT, float frequency, int octaves,
                           float lacunarity, float gain, int nCurves, int nCV, float *field,
                           float *perCurve);   // noise_gpu.cu
}  // namespace usdGen
#endif

extern "C" USDGEN_CURVES_CUDA_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    r.RegisterDeviceProgram(UsdGenFindBackend("cuda"), UsdGenCurveSetType(),
                            &UsdGenCurvesCudaProgram());
    UsdGenCurvesKernelOps const &ops = UsdGenCurvesBuiltinKernelOps();
    UsdGenCudaKernelOps cuda;
    cuda.scatter = ops.scatter;
    cuda.grow = ops.grow;
    cuda.noise = ops.noise;
    cuda.length = ops.length;
    cuda.width = ops.width;
    cuda.widthBlend = ops.widthBlend;
    cuda.curveSource = ops.curveSource;
    cuda.deform = ops.deform;
    cuda.referenceSource = ops.referenceSource;
    UsdGenRegisterCurvesCuda(r, cuda);
#if defined(USDGEN_USE_GPU_NOISE)
    UsdGenSetNoiseCaptureGpu(&UsdGenNoiseCaptureGPU);
#endif
}
