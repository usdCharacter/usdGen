// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Extension registration. Every library that adds ops, types or kernels
// exports
//
//     extern "C" void usdGenRegister(usdGen::UsdGenRegistrar &);
//
// (or usdGenRegister_<lib> when linked statically). Registries freeze on the
// first compile; after that every lookup reads immutable state.
#pragma once
#include "usdGen/export.h"
#include "usdGen/registries/backendRegistry.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/typeRegistry.h"
#include <string>

namespace usdGen {

class UsdGenRegistrar;
USDGEN_CORE_API bool UsdGenRegisterStatic(void (*fn)(UsdGenRegistrar &));

class USDGEN_CORE_API UsdGenRegistrar {
public:
    /// Invalid on a duplicate name, a missing create, more than
    /// kUsdGenMaxParamsPerOp params, a duplicate param or enumerant, or after freeze.
    UsdGenOpId RegisterOp(UsdGenOpDef const &);
    UsdGenTypeId RegisterType(UsdGenTypeDef const &);
    bool RegisterKernel(UsdGenOpId, UsdGenBackendId, UsdGenKernelDef const &);
private:
    friend bool UsdGenRegisterStatic(void (*)(UsdGenRegistrar &));
    friend class UsdGenOpRegistry;
    explicit UsdGenRegistrar(UsdGenOpRegistry &ops) noexcept : _ops(&ops) {}
    UsdGenOpRegistry *_ops;
};

using UsdGenRegisterFn = void (*)(UsdGenRegistrar &);
USDGEN_CORE_API bool UsdGenLoadExtension(std::string const &path, std::string *error);
USDGEN_CORE_API bool UsdGenRegisterStatic(UsdGenRegisterFn fn);
USDGEN_CORE_API void UsdGenFreezeRegistries();   // idempotent; the first Compile calls it

}  // namespace usdGen

/// Built-in ops, map types and kernels: ops/registerCurves.cpp.
extern "C" USDGEN_CORE_API void usdGenRegister_usdGenCurves(usdGen::UsdGenRegistrar &);
