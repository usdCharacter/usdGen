// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// testUsdGenRegistrarLoad (step 6.4): links usdGen only. "load": with
// USDGEN_EXTENSIONS empty the registries start bare, and UsdGenLoadExtension
// loads usdGenSurface, usdGenExpr and usdGenCurves and runs their
// usdGenRegister. "defaults": the registry build loads the three from
// usdGen's directory. Neither registers a curve publisher: the curves' Hydra
// adapter (usdGenHydraCurves) does, and only before the registries freeze.

#include "utils/testToken.h"
#include "usdGen/api/engine.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/publisher.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/typeRegistry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (ok) return;
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
}

bool HasOp(char const *name)
{
    return UsdGenOpRegistry::Get().Find(TestToken(name)) != UsdGenOpId::Invalid;
}

UsdGenTypeId TypeOf(char const *name)
{
    return UsdGenTypeRegistry::Get().Find(TestToken(name));
}

void CheckPrimitives(std::string const &mode)
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    Check(HasOp("UsdGenScatter") && HasOp("UsdGenNoise") && HasOp("UsdGenExprOp"),
          mode + ": built-in ops registered");
    Check(ops.Find(TestToken("UsdGenScatter")) == UsdGenOpId(0), mode + ": Scatter keeps op id 0");
    Check(ops.ParamEvaluator().create != nullptr, mode + ": usdGenExpr's parameter evaluator");
    UsdGenTypeId const curves = TypeOf("UsdGenCurveSet");
    Check(curves != UsdGenTypeId::Invalid, mode + ": UsdGenCurveSet type");
    Check(UsdGenIndex(curves) == 0, mode + ": UsdGenCurveSet keeps type id 0");
    Check(TypeOf("UsdGenRootSet") != UsdGenTypeId::Invalid, mode + ": UsdGenRootSet type");
    Check(TypeOf("UsdGenSurfaceInput") != UsdGenTypeId::Invalid, mode + ": surface input kind");
    Check(UsdGenFindPublisher(curves) == nullptr,
          mode + ": the primitive registers no publisher (its Hydra adapter does)");
}

int RunLoad()
{
#if defined(_WIN32)
    _putenv_s("USDGEN_EXTENSIONS", ";");
#else
    setenv("USDGEN_EXTENSIONS", ":", 1);
#endif
    std::string error;
    Check(UsdGenEngine::Create().ok(), "bare engine initializes");
    Check(UsdGenOpRegistry::Get().Count() == 0, "no extension, no ops");
    Check(UsdGenOpRegistry::Get().ParamEvaluator().create == nullptr, "no extension, no evaluator");

    Check(!UsdGenLoadExtension(std::string(USDGEN_TEST_SURFACE) + ".missing", &error) &&
              !error.empty(), "a missing file fails with a message");
    error.clear();
    Check(!UsdGenLoadExtension(USDGEN_TEST_NOT_EXTENSION, &error) &&
              error.find("usdGenRegister") != std::string::npos,
          "a library without usdGenRegister fails: " + error);

    for (char const *path : {USDGEN_TEST_SURFACE, USDGEN_TEST_EXPR, USDGEN_TEST_CURVES}) {
        error.clear();
        Check(UsdGenLoadExtension(path, &error), std::string("load ") + path + ": " + error);
    }
    CheckPrimitives("load");
    error.clear();
    Check(UsdGenLoadExtension(USDGEN_TEST_HYDRA_CURVES, &error),
          std::string("load the curves' Hydra adapter: ") + error);
    Check(UsdGenFindPublisher(TypeOf("UsdGenCurveSet")) != nullptr,
          "load: the Hydra adapter registers the curve publisher");

    UsdGenFreezeRegistries();
    error.clear();
    Check(!UsdGenLoadExtension(USDGEN_TEST_CURVES, &error) &&
              error.find("frozen") != std::string::npos, "no load after the freeze");
    return 0;
}

int RunDefaults()
{
    std::string error;
    UsdGenStatus const created = UsdGenEngine::Create();
    Check(created.ok(), "default extensions load from usdGen's directory: " +
                            std::string(created.message()));
    CheckPrimitives("defaults");
    Check(UsdGenFindBackend(TestToken("cpu")) == kUsdGenBackendCpu &&
              UsdGenGetBackend(kUsdGenBackendCpu) == nullptr,
          "defaults: the CPU is built in");
    Check(UsdGenFindBackend(TestToken("noSuchBackend")) == UsdGenBackendId::Invalid,
          "defaults: an unregistered name finds no backend");
    // A host that registers its publisher after the first compile is too late:
    // the type publishes nothing (the cooker says so once per session).
    UsdGenFreezeRegistries();
    error.clear();
    Check(!UsdGenLoadExtension(USDGEN_TEST_HYDRA_CURVES, &error) &&
              error.find("frozen") != std::string::npos,
          "defaults: the Hydra adapter cannot register after the freeze");
    Check(UsdGenFindPublisher(TypeOf("UsdGenCurveSet")) == nullptr,
          "defaults: no curve publisher after a late registration");
#if defined(USDGEN_TEST_VULKAN_BACKEND)
    UsdGenBackendId const vk = UsdGenFindBackend(TestToken("vulkan"));
    UsdGenBackendDef const *def = UsdGenGetBackend(vk);
    Check(vk != UsdGenBackendId::Invalid && def && def->createProvider &&
              std::strcmp(UsdGenBackendName(vk), "vulkan") == 0,
          "defaults: usdGenVulkan loads from usdGen's directory and registers its backend");
#endif
    return 0;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string const mode = argc > 1 ? argv[1] : "load";
    if (mode == "load") RunLoad();
    else if (mode == "defaults") RunDefaults();
    else {
        std::printf("usage: testUsdGenRegistrarLoad [load|defaults]\n");
        return 2;
    }
    if (g_failures) {
        std::printf("testUsdGenRegistrarLoad %s: %d failure(s)\n", mode.c_str(), g_failures);
        return 1;
    }
    std::printf("testUsdGenRegistrarLoad %s: OK\n", mode.c_str());
    return 0;
}
