// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registrar.h"

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace usdGen {

bool UsdGenRegisterStatic(UsdGenRegisterFn fn)
{
    UsdGenOpRegistry &ops = UsdGenOpRegistry::Get();
    if (!fn || ops.IsFrozen()) return false;
    UsdGenRegistrar registrar(ops);
    fn(registrar);
    return true;
}

bool UsdGenLoadExtension(std::string const &path, std::string *error)
{
    auto fail = [error](std::string message) {
        if (error) *error = std::move(message);
        return false;
    };
    if (UsdGenOpRegistry::Get().IsFrozen())
        return fail("registries are frozen; load extensions before the first compile");
#if defined(_WIN32)
    HMODULE module = LoadLibraryA(path.c_str());
    if (!module) return fail("cannot load '" + path + "'");
    auto fn = reinterpret_cast<UsdGenRegisterFn>(GetProcAddress(module, "usdGenRegister"));
#else
    void *module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) return fail(std::string("cannot load '") + path + "': " + dlerror());
    auto fn = reinterpret_cast<UsdGenRegisterFn>(dlsym(module, "usdGenRegister"));
#endif
    // The module stays loaded: registered factories point into it.
    if (!fn) return fail("'" + path + "' exports no usdGenRegister");
    return UsdGenRegisterStatic(fn);
}

void UsdGenFreezeExecutionBackendRegistry() noexcept;  // executionBackend.h

void UsdGenFreezeRegistries()
{
    UsdGenOpRegistry::Get().Freeze();
    UsdGenFreezeExecutionBackendRegistry();
}

}  // namespace usdGen
