// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenCurvesVulkan's registration: the curves' Vulkan device program, so
// far its SPIR-V table (the runtime still builds the pipelines). Loaded
// after usdGenVulkan (USDGEN_DEFAULT_EXTENSIONS), whose backend it names.
#include "usdGenCurves/vulkan/export.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGenVulkan/compile/program.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/registrar.h"

using namespace usdGen;

namespace {

// width.spv first: it locates the shader directory.
constexpr vulkan::ShaderDecl kShaders[] = {
    {"width", "width.spv"},
    {"lengthScale", "lengthScale.spv"},
    {"lengthSet", "lengthSet.spv"},
    {"lengthCutExtend", "lengthCutExtend.spv"},
    {"lengthReparam", "lengthReparam.spv"},
    {"lengthMinimum", "lengthMinimum.spv"},
    {"lengthLiteralV1", "lengthLiteralV1.spv"},
    {"lengthEnvelopeV1", "lengthEnvelopeV1.spv"},
    {"lengthDeviceV1", "lengthDeviceV1.spv"},
    {"widthBlend", "widthBlend.spv"},
    {"nonWidthCompare", "nonWidthCompare.spv"},
    {"lengthCompaction", "lengthCompaction.spv"},
    {"noiseValidate", "noiseValidate.spv"},
    {"noiseGenerate", "noiseGenerate.spv"},
    {"resampleVk", "resampleVk.spv"},
    {"exprVkContext", "exprVkContext.spv"},
    {"exprVkEvaluate", "exprVkEvaluate.spv"},
    {"exprVkWidth", "exprVkWidth.spv"},
    {"exprVkPack", "exprVkPack.spv"},
    {"curveGrow", "curveGrow.spv"},
    {"picktileVkTiles", "picktileVkTiles.spv"},
    {"picktileVkBounds", "picktileVkBounds.spv"},
    {"rbfVkExtent", "rbfVkExtent.spv"},
    {"rbfVkGram", "rbfVkGram.spv"},
    {"rbfVkBuildMatrix", "rbfVkBuildMatrix.spv"},
    {"rbfVkLu", "rbfVkLu.spv"},
    {"rbfVkRhs", "rbfVkRhs.spv"},
    {"rbfVkTriSolve", "rbfVkTriSolve.spv"},
    {"rbfVkEvaluate", "rbfVkEvaluate.spv"},
    {"rbfVkApply", "rbfVkApply.spv"},
    {"rbfVkVerify", "rbfVkVerify.spv"},
    {"rbfVkFill", "rbfVkFill.spv"},
    {"rbfVkEvaluateCached", "rbfVkEvaluateCached.spv"},
};

// Constant-initialized; the anchor is this object, so the shader directory
// is resolved next to this module.
vulkan::ProgramDef const gProgram{UsdGenSpan<const vulkan::ShaderDecl>(kShaders), &gProgram,
                                  nullptr};

}  // namespace

extern "C" USDGEN_CURVES_VULKAN_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    r.RegisterDeviceProgram(UsdGenFindBackend("vulkan"), UsdGenCurveSetType(), &gProgram);
}
