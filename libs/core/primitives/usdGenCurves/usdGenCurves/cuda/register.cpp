// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesCuda's registration: the CUDA kernel rows of the built-in curve
// ops (the capability matrix reads them in every build) and the curves' CUDA
// device program. Loaded after usdGenCuda (USDGEN_DEFAULT_EXTENSIONS), whose
// backend it names, and after usdGenCurves, whose op ids the rows attach to.
#include "usdGenCurves/cuda/export.h"
#include "usdGenCurves/cuda/cudaStages.h"
#include "usdGenCurves/cuda/program.h"
#include "usdGenCurves/ops/kernelOps.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

using namespace usdGen;

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
}
