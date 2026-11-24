// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Built-in curve ops and map types (the future usdGenCurves, step 6). The op
// registry's constructor calls this; registration order is id order.
#include "usdGen/registries/registrar.h"
#include "usdGen/cudaStages.h"

#include "usdGen/ops/scatter.h"
#include "usdGen/ops/grow.h"
#include "usdGen/ops/noise.h"
#include "usdGen/ops/length.h"
#include "usdGen/ops/width.h"
#include "usdGen/ops/widthBlend.h"
#include "usdGen/ops/curveSource.h"
#include "usdGen/ops/deform.h"
#include "usdGen/ops/referenceSource.h"
#include "usdGen/ops/clump.h"
#include "usdGen/ops/guideInterpolate.h"
#include "usdGen/ops/curl.h"
#include "usdGen/ops/bend.h"
#include "usdGen/ops/wave.h"
#include "usdGen/ops/part.h"
#include "usdGen/ops/direction.h"
#include "usdGen/ops/smooth.h"
#include "usdGen/ops/resample.h"
#include "usdGen/ops/scale.h"
#include "usdGen/ops/straighten.h"
#include "usdGen/ops/displace.h"
#include "usdGen/ops/sculptLayer.h"
#include "usdGen/ops/wind.h"
#include "usdGen/ops/exprOp.h"
#include "usdGen/ops/freeze.h"
#include "usdGen/ops/collide.h"
#include "usdGen/ops/braid.h"
#include "usdGen/ops/twist.h"
#include "usdGen/ops/trim.h"
#include "usdGen/ops/frizz.h"
#include "usdGen/ops/tieClump.h"
#include "usdGen/ops/cardSource.h"
#include "usdGen/vulkan/vkStages.h"

using namespace usdGen;

namespace {

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

extern "C" void usdGenRegister_usdGenCurves(UsdGenRegistrar &r)
{
    UsdGenVulkanKernelOps k;
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
    UsdGenCudaKernelOps cuda;
    cuda.scatter = k.scatter;
    cuda.grow = k.grow;
    cuda.noise = k.noise;
    cuda.length = k.length;
    cuda.width = k.width;
    cuda.widthBlend = k.widthBlend;
    cuda.curveSource = k.curveSource;
    cuda.deform = k.deform;
    cuda.referenceSource = k.referenceSource;
    UsdGenRegisterCurvesCuda(r, cuda);
    UsdGenRegisterVulkanKernels(r, k);
}
