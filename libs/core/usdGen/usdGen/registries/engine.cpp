// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/engine.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/typeRegistry.h"

#include <atomic>

namespace usdGen {

void UsdGenCreateCacheDomainTable();   // executionCache.cpp

namespace {
std::atomic<UsdGenEngine *> gEngine{nullptr};
}  // namespace

UsdGenStatus UsdGenEngine::Create(UsdGenEngineDesc const &desc)
{
    if (desc.report) UsdGenSetReportSink(desc.report, desc.reportUser);
    if (UsdGenEngine *engine = gEngine.load(std::memory_order_acquire)) return engine->_status;
    UsdGenOpRegistry const &ops = UsdGenBuildOpRegistry(desc.manifest);
    UsdGenCreateCacheDomainTable();
    UsdGenEngine *fresh = new UsdGenEngine;
    if (!ops.InitError().empty())
        fresh->_status = UsdGenStatus::Error(UsdGenStatusCode::Failed, ops.InitError());
    // Never freed; a thread that loses the race discards its own copy.
    UsdGenEngine *winner = nullptr;
    if (!gEngine.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
        delete fresh;
        return winner->_status;
    }
    return fresh->_status;
}

UsdGenEngine &UsdGenEngine::Get()
{
    if (UsdGenEngine *engine = gEngine.load(std::memory_order_acquire)) return *engine;
    (void)Create();
    return *gEngine.load(std::memory_order_acquire);
}

UsdGenOpId UsdGenEngine::FindOp(UsdGenToken name) const noexcept
{
    return UsdGenOpRegistry::Get().Find(name);
}

UsdGenTypeId UsdGenEngine::FindType(UsdGenToken name) const noexcept
{
    return UsdGenTypeRegistry::Get().Find(name);
}

}  // namespace usdGen
