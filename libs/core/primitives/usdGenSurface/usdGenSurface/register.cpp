// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenSurface's registration (step 6.4). Interns the subdivision token
// table; Scatter, RootSet, the surface and map input kinds and the map types
// still register from usdGenCurves (ops/registerCurves.cpp) so their ids and
// the input-kind run order stay as they were before the split.
#include "usdGen/registries/registrar.h"
#include "usdGenSurface/export.h"
#include "usdGenSurface/limitSurface.h"
#include "usdGenSurface/maps/ptexMap.h"

using namespace usdGen;

extern "C" USDGEN_SURFACE_API void usdGenRegister_usdGenSurface(UsdGenRegistrar &r);
extern "C" USDGEN_SURFACE_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    usdGenRegister_usdGenSurface(r);
}

extern "C" void usdGenRegister_usdGenSurface(UsdGenRegistrar &)
{
    (void)UsdGenSurfaceTokens();
    UsdGenPtexTexture::CreateCache();
}
