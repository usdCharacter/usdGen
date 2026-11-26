// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesCuda's registration: the curves' CUDA device program. Loaded
// after usdGenCuda (USDGEN_DEFAULT_EXTENSIONS), whose backend it names.
#include "usdGenCurves/cuda/export.h"
#include "usdGenCurves/cuda/program.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

using namespace usdGen;

extern "C" USDGEN_CURVES_CUDA_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    r.RegisterDeviceProgram(UsdGenFindBackend("cuda"), UsdGenCurveSetType(),
                            &UsdGenCurvesCudaProgram());
}
