// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Extension registration. Every library that adds ops, types or kernels
// exports
//
//     extern "C" void usdGenRegister(usdGen::UsdGenRegistrar &);
//
// (and usdGenRegister_<lib> for a static link). Registries freeze on the
// first compile; after that every lookup reads immutable state.
#pragma once
#include "usdGen/export.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/backendRegistry.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/paramEvaluator.h"
#include "usdGen/registries/publisher.h"
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
    /// Invalid on a duplicate name, a null createProvider, every handle
    /// already used, or after freeze. Handles are dense from 1.
    UsdGenBackendId RegisterBackend(UsdGenBackendDef const &);
    /// One per (backend, primitive type); false on a duplicate, an unknown
    /// handle or type, or after freeze. The pointer must outlive the registry.
    bool RegisterDeviceProgram(UsdGenBackendId, UsdGenTypeId, void const *programDef);
    /// One per engine; false on a second one, a missing create, or after freeze.
    bool RegisterParamEvaluator(UsdGenParamEvaluatorDef const &);
    /// One per primitive type; false on a second one for the type, an
    /// unregistered type, a missing build, or after freeze.
    bool RegisterPublisher(UsdGenPublisherDef const &);
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

/// Builds the registries: loads the extensions named by USDGEN_EXTENSIONS
/// (paths, ';' on Windows, ':' elsewhere), else the default extensions CMake
/// names (USDGEN_DEFAULT_EXTENSIONS) from usdGen's own directory, and
/// calls each usdGenRegister in that order. Idempotent and lock-free; every registry lookup calls it
/// implicitly. False with 'error' when an extension failed to load. Freezing
/// stays with the first compile, so extensions and tests can register after it.
USDGEN_CORE_API bool UsdGenEngineEnsureInitialized(std::string *error = nullptr);

}  // namespace usdGen
