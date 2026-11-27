// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenHydraCurves/register.h"
#include "usdGenHydraCurves/curveColumns.h"
#include "usdGenHydraCurves/publisher.h"
#include "usdGen/registries/registrar.h"

#include <atomic>
#include <thread>

namespace usdGenHydra {

namespace {

// 0: not registered, 1: a thread is registering, 2: registered.
std::atomic<int> gState{0};

}  // namespace

bool UsdGenHydraCurvesEnsureRegistered(std::string *error)
{
    using namespace usdGen;
    for (;;) {
        int state = gState.load(std::memory_order_acquire);
        if (state == 2) return true;
        if (state == 0 && gState.compare_exchange_weak(state, 1, std::memory_order_acq_rel))
            break;
        std::this_thread::yield();   // another thread registers (microseconds)
    }
    std::string reason;
    (void)UsdGenEngineEnsureInitialized();   // the primitives register first
    if (!UsdGenCurvePublisherRegistered()) {
        if (UsdGenOpRegistry::Get().IsFrozen())
            reason = "the registries are frozen: register usdGenHydraCurves before the first compile";
        else if (!UsdGenRegisterStatic(&usdGenRegister_usdGenHydraCurves))
            reason = "usdGenHydraCurves could not register";
        else if (!UsdGenCurvePublisherRegistered() && reason.empty())
            reason = "no curve publisher after registration (UsdGenCurveSet missing or taken)";
    }
    bool const ok = reason.empty();
    if (!ok && error) *error = reason;
    gState.store(ok ? 2 : 0, std::memory_order_release);
    return ok;
}

}  // namespace usdGenHydra

// Also loadable as an extension (USDGEN_EXTENSIONS, UsdGenLoadExtension).
extern "C" USDGEN_HYDRA_CURVES_API void usdGenRegister(usdGen::UsdGenRegistrar &r);

extern "C" void usdGenRegister(usdGen::UsdGenRegistrar &r)
{
    usdGenRegister_usdGenHydraCurves(r);
}

extern "C" void usdGenRegister_usdGenHydraCurves(usdGen::UsdGenRegistrar &r)
{
    char const *reason = nullptr;
    if (!usdGen::UsdGenHydraCurvesResolveNames(&reason)) return;
    usdGen::UsdGenRegisterCurvePublisher(r);
}
