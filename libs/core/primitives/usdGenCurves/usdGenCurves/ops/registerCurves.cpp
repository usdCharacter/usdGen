// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurves' registration: built-in ops, map types, input kinds and
// kernels. The op registry loads the primitives (Surface, Expr, Curves) and
// calls each usdGenRegister; registration order is id order, so this file
// keeps registering the Surface and Expr items it registered before the
// split (their ids and the input-kind run order stay unchanged).
#include "usdGen/registries/registrar.h"
#include "usdGenCurves/export.h"
#include "usdGenCurves/ops/kernelOps.h"
#include "usdGenCurves/publish/curvePublisher.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGenSurface/types/rootSet.h"

#include "usdGenSurface/ops/scatter.h"
#include "usdGenCurves/ops/grow.h"
#include "usdGenCurves/ops/noise.h"
#include "usdGenCurves/ops/length.h"
#include "usdGenCurves/ops/width.h"
#include "usdGenCurves/ops/widthBlend.h"
#include "usdGenCurves/ops/curveSource.h"
#include "usdGenCurves/ops/deform.h"
#include "usdGenCurves/ops/referenceSource.h"
#include "usdGenCurves/ops/clump.h"
#include "usdGenCurves/ops/guideInterpolate.h"
#include "usdGenCurves/ops/curl.h"
#include "usdGenCurves/ops/bend.h"
#include "usdGenCurves/ops/wave.h"
#include "usdGenCurves/ops/part.h"
#include "usdGenCurves/ops/direction.h"
#include "usdGenCurves/ops/smooth.h"
#include "usdGenCurves/ops/resample.h"
#include "usdGenCurves/ops/scale.h"
#include "usdGenCurves/ops/straighten.h"
#include "usdGenCurves/ops/displace.h"
#include "usdGenCurves/ops/sculptLayer.h"
#include "usdGenCurves/ops/wind.h"
#include "usdGenExpr/ops/exprOp.h"
#include "usdGenCurves/ops/freeze.h"
#include "usdGenCurves/ops/collide.h"
#include "usdGenCurves/ops/braid.h"
#include "usdGenCurves/ops/twist.h"
#include "usdGenCurves/ops/trim.h"
#include "usdGenCurves/ops/frizz.h"
#include "usdGenCurves/ops/tieClump.h"
#include "usdGenCurves/ops/cardSource.h"

using namespace usdGen;

namespace {

// Written by usdGenRegister (before the freeze); read after it.
UsdGenCurvesKernelOps gKernelOps;

void RegisterMapTypes(UsdGenRegistrar &r)
{
    UsdGenTypeDef d;
    d.name = "UsdGenPtexMap";
    d.traits = kUsdGenTypeMap | kUsdGenTypeMapPtex | kUsdGenTypeMapExprSampleable;
    r.RegisterType(d);
    d.name = "UsdGenPaintMap";
    d.traits = kUsdGenTypeMap | kUsdGenTypeMapPaintSnapshot | kUsdGenTypeMapExprSampleable;
    r.RegisterType(d);
    d.name = "UsdGenImageMap";
    d.traits = kUsdGenTypeMap | kUsdGenTypeMapImage;
    r.RegisterType(d);
    d.name = "UsdGenExprMap";
    d.traits = kUsdGenTypeMap;
    r.RegisterType(d);
}

}  // namespace

UsdGenCurvesKernelOps const &usdGen::UsdGenCurvesBuiltinKernelOps() noexcept
{
    return gKernelOps;
}

extern "C" USDGEN_CURVES_API void usdGenRegister_usdGenCurves(UsdGenRegistrar &r);
extern "C" USDGEN_CURVES_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    usdGenRegister_usdGenCurves(r);
}

extern "C" void usdGenRegister_usdGenCurves(UsdGenRegistrar &r)
{
    UsdGenRegisterCurveSetType(r);   // first: op defs name its type
    UsdGenRegisterCurvePublisher(r);
    UsdGenCurvesKernelOps &k = gKernelOps;
    k.scatter = UsdGenRegisterScatterOp(r);
    k.grow = UsdGenRegisterGrowOp(r);
    k.noise = UsdGenRegisterNoiseOp(r);
    k.length = UsdGenRegisterLengthOp(r);
    k.width = UsdGenRegisterWidthOp(r);
    k.widthBlend = UsdGenRegisterWidthBlendOp(r);
    k.curveSource = UsdGenRegisterCurveSourceOp(r);
    k.deform = UsdGenRegisterDeformOp(r);
    k.referenceSource = UsdGenRegisterReferenceSourceOp(r);
    UsdGenRegisterClumpOp(r);
    UsdGenRegisterGuideInterpolateOp(r);
    UsdGenRegisterCurlOp(r);
    UsdGenRegisterBendOp(r);
    UsdGenRegisterWaveOp(r);
    UsdGenRegisterPartOp(r);
    UsdGenRegisterDirectionOp(r);
    UsdGenRegisterSmoothOp(r);
    k.resample = UsdGenRegisterResampleOp(r);
    UsdGenRegisterScaleOp(r);
    UsdGenRegisterStraightenOp(r);
    UsdGenRegisterDisplaceOp(r);
    UsdGenRegisterSculptLayerOp(r);
    UsdGenRegisterWindOp(r);
    UsdGenRegisterExprOpOp(r);
    UsdGenRegisterFreezeOp(r);
    UsdGenRegisterCollideOp(r);
    UsdGenRegisterBraidOp(r);
    UsdGenRegisterTwistOp(r);
    UsdGenRegisterTrimOp(r);
    UsdGenRegisterFrizzOp(r);
    UsdGenRegisterTieClumpOp(r);
    UsdGenRegisterCardSourceOp(r);
    RegisterMapTypes(r);
    // Input kinds after every older type, so their ids keep their values;
    // a spec's inputs run in this (kind) order.
    UsdGenRegisterCurveSetInputKind(r);
    UsdGenRegisterSurfaceInputKind(r);
    UsdGenRegisterMapInputKind(r);
    UsdGenRegisterExprInputKinds(r);
    UsdGenRegisterRootSetType(r);   // last: older ids keep their values
    // The CUDA and Vulkan kernel rows register from usdGenCuda and
    // usdGenVulkan, which load after this library.
}
