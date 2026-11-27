// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/typeRegistry.h"
#include "usdGen/registries/opRegistry.h"

namespace usdGen {

UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;

namespace {
std::atomic<UsdGenTypeRegistry *> gTypeRegistry{nullptr};
UsdGenToken const kNoTypeName;
UsdGenTypeDef const kNoTypeDef;
}  // namespace

UsdGenTypeRegistry::UsdGenTypeRegistry() = default;

// Built-ins register from the op registry's constructor, so this one is
// created first without it; Get() makes sure the built-ins are in.
UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept
{
    if (UsdGenTypeRegistry *r = gTypeRegistry.load(std::memory_order_acquire)) return *r;
    auto *fresh = new UsdGenTypeRegistry;
    UsdGenTypeRegistry *winner = nullptr;
    if (gTypeRegistry.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                              std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

UsdGenTypeRegistry const &UsdGenTypeRegistry::Get() noexcept
{
    (void)UsdGenOpRegistry::Get();
    return UsdGenMutableTypeRegistry();
}

UsdGenTypeId UsdGenTypeRegistry::Find(UsdGenToken const &name) const noexcept
{
    if (name.IsEmpty()) return UsdGenTypeId::Invalid;
    for (uint32_t i = 0; i < _count; ++i)
        if (_names[i] == name) return UsdGenTypeId(i);
    return UsdGenTypeId::Invalid;
}

UsdGenTypeDef const &UsdGenTypeRegistry::Def(UsdGenTypeId id) const noexcept
{
    return UsdGenIndex(id) < _count ? _defs[UsdGenIndex(id)] : kNoTypeDef;
}

UsdGenTypeId UsdGenTypeRegistry::DefaultPrimitive() const noexcept
{
    for (uint32_t i = 0; i < _count; ++i)
        if ((_defs[i].traits & kUsdGenTypeDefaultPrimitive) && _defs[i].layout) return UsdGenTypeId(i);
    return UsdGenTypeId::Invalid;
}

UsdGenToken const &UsdGenTypeRegistry::Name(UsdGenTypeId id) const noexcept
{
    return UsdGenIndex(id) < _count ? _names[UsdGenIndex(id)] : kNoTypeName;
}

}  // namespace usdGen
