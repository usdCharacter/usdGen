// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenExpr's registration: the CPU connected-parameter evaluator. Linked
// into usdGen and called by the registry build until the split (step 6.4).
#include "usdGen/registries/registrar.h"
#include "usdGenExpr/cpuParameters.h"

#include <memory>

using namespace usdGen;

namespace {

std::unique_ptr<UsdGenParamEvaluatorState> CreateCpuParameters()
{
    return std::make_unique<UsdGenCpuParameters>();
}

}  // namespace

extern "C" void usdGenRegister_usdGenExpr(UsdGenRegistrar &r)
{
    UsdGenParamEvaluatorDef def;
    def.create = &CreateCpuParameters;
    r.RegisterParamEvaluator(def);
}
