// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenExpr's registration: the CPU connected-parameter evaluator. The op
// registry loads usdGenExpr and calls usdGenRegister (step 6.4).
#include "usdGen/api/ext/registrar.h"
#include "usdGenExpr/cpuParameters.h"
#include "usdGenExpr/export.h"
#include "usdGenExpr/exprValidate.h"

#include <memory>

using namespace usdGen;

namespace {

std::unique_ptr<UsdGenParamEvaluatorState> CreateCpuParameters()
{
    return std::make_unique<UsdGenCpuParameters>();
}

}  // namespace

extern "C" USDGEN_EXPR_API void usdGenRegister_usdGenExpr(UsdGenRegistrar &r);
extern "C" USDGEN_EXPR_API void usdGenRegister(UsdGenRegistrar &r);

extern "C" void usdGenRegister(UsdGenRegistrar &r)
{
    usdGenRegister_usdGenExpr(r);
}

extern "C" void usdGenRegister_usdGenExpr(UsdGenRegistrar &r)
{
    UsdGenParamEvaluatorDef def;
    def.create = &CreateCpuParameters;
    def.validateInputs = &UsdGenValidateExpressionInputs;
    def.validateBindings = &UsdGenValidateExpressionBindingSources;
    def.bindingDependencies = &UsdGenExpressionDependencies;
    r.RegisterParamEvaluator(def);
}
