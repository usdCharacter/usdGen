// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — extension registration. Registration is the only extension
// mechanism: every library that adds operators exports
//
//     extern "C" void usdGenRegister(usdGen::UsdGenRegistrar &);
//
// (or usdGenRegister_<lib> when linked statically). The loader calls it once,
// then freezes the registries; after Freeze() every lookup is a read of
// immutable state, so compile and exec take no lock.
#ifndef USDGEN_REGISTRAR_H
#define USDGEN_REGISTRAR_H

#include "usdGen/export.h"
#include "usdGen/opRegistry.h"

#include <string>

namespace usdGen {

class UsdGenRegistrar
{
public:
    explicit UsdGenRegistrar(UsdGenOpRegistry &ops) : _ops(ops) {}

    /// Add an operator kernel. False on a duplicate type or after Freeze().
    bool RegisterOp(TfToken const &type, UsdGenOpRegistry::Factory factory)
    {
        return _ops.Register(type, std::move(factory));
    }

private:
    UsdGenOpRegistry &_ops;
};

/// Signature of a library's registration entry point.
using UsdGenRegisterFn = void (*)(UsdGenRegistrar &);

/// Load a shared library that exports `usdGenRegister`, call it, and keep the
/// library loaded for the life of the process (plugins never unload). Fails
/// after Freeze(). On failure `error` says why.
USDGEN_CORE_API bool UsdGenLoadExtension(std::string const &path, std::string *error);

/// Call a statically linked library's entry point (usdGenRegister_<lib>).
USDGEN_CORE_API bool UsdGenRegisterStatic(UsdGenRegisterFn fn);

/// Freeze every registry. Idempotent; later registration fails.
USDGEN_CORE_API void UsdGenFreezeRegistries();

}  // namespace usdGen

/// The built-in curve operators' entry point (the future usdGenCurves).
extern "C" USDGEN_CORE_API void usdGenRegister_usdGenCurves(usdGen::UsdGenRegistrar &);

#endif  // USDGEN_REGISTRAR_H
