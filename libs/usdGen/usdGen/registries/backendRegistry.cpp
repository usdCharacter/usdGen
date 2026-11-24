// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/backendRegistry.h"
#include "usdGen/registries/opRegistry.h"

namespace usdGen {

UsdGenBackendRegistry &UsdGenMutableBackendRegistry() noexcept;

namespace {
std::atomic<UsdGenBackendRegistry *> gBackendRegistry{nullptr};
}  // namespace

UsdGenBackendRegistry::UsdGenBackendRegistry()
{
    for (auto &row : _order)
        for (UsdGenOpId &id : row) id = UsdGenOpId::Invalid;
}

UsdGenBackendRegistry &UsdGenMutableBackendRegistry() noexcept
{
    if (UsdGenBackendRegistry *r = gBackendRegistry.load(std::memory_order_acquire)) return *r;
    auto *fresh = new UsdGenBackendRegistry;
    UsdGenBackendRegistry *winner = nullptr;
    if (gBackendRegistry.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

UsdGenBackendRegistry const &UsdGenBackendRegistry::Get() noexcept
{
    (void)UsdGenOpRegistry::Get();
    return UsdGenMutableBackendRegistry();
}

UsdGenKernelDef const *UsdGenBackendRegistry::Find(UsdGenOpId op, UsdGenBackendId backend) const noexcept
{
    uint32_t const b = UsdGenIndex(backend), o = UsdGenIndex(op);
    if (b >= kUsdGenMaxBackends || o >= kUsdGenMaxOps || !_present[b][o]) return nullptr;
    return &_kernels[b][o];
}

TfSpan<const UsdGenOpId> UsdGenBackendRegistry::Ops(UsdGenBackendId backend) const noexcept
{
    uint32_t const b = UsdGenIndex(backend);
    if (b >= kUsdGenMaxBackends) return {};
    return TfSpan<const UsdGenOpId>(_order[b], _orderCount[b]);
}

}  // namespace usdGen
