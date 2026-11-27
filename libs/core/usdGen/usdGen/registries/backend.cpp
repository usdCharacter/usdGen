// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/backend.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/typeRegistry.h"

#include <atomic>
#include <cstring>

namespace usdGen {

UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;   // typeRegistry.cpp
bool UsdGenRegistering() noexcept;                          // registrar.cpp

namespace {

// A slot is claimed by CAS (free -> writing), filled, then published (ready).
// Written by registration only (before the freeze); read after it.
enum : uint32_t { kFree = 0, kWriting = 1, kReady = 2 };
std::atomic<uint32_t> gState[kUsdGenMaxBackends] = {};
UsdGenBackendDef gDefs[kUsdGenMaxBackends];
std::atomic<void const *> gPrograms[kUsdGenMaxBackends][kUsdGenMaxTypes] = {};

// Builds the registries unless this thread is building them (a library's
// usdGenRegister may look a backend up).
void EnsureRegistries() noexcept
{
    if (!UsdGenRegistering()) (void)UsdGenOpRegistry::Get();
}

bool Ready(uint32_t slot) noexcept
{
    return slot < kUsdGenMaxBackends && gState[slot].load(std::memory_order_acquire) == kReady;
}

uint32_t FindSlot(char const *name) noexcept
{
    for (uint32_t i = 1; i < kUsdGenMaxBackends; ++i)
        if (Ready(i) && std::strcmp(gDefs[i].name, name) == 0) return i;
    return kUsdGenMaxBackends;
}

bool Claim(uint32_t slot) noexcept
{
    uint32_t expected = kFree;
    return gState[slot].compare_exchange_strong(expected, kWriting, std::memory_order_acq_rel);
}

}  // namespace

UsdGenBackendId UsdGenRegistrar::RegisterBackend(UsdGenBackendDef const &def)
{
    if (_ops->IsFrozen() || !def.name || !*def.name || !def.createProvider ||
        std::strcmp(def.name, "cpu") == 0 || std::strcmp(def.name, "cpuReference") == 0 ||
        FindSlot(def.name) != kUsdGenMaxBackends)
        return UsdGenBackendId::Invalid;
    // Dense from 1 in load order (usdGenExtensions.txt).
    uint32_t slot = kUsdGenMaxBackends;
    for (uint32_t i = 1; i < kUsdGenMaxBackends && slot == kUsdGenMaxBackends; ++i)
        if (Claim(i)) slot = i;
    if (slot == kUsdGenMaxBackends) return UsdGenBackendId::Invalid;
    gDefs[slot] = def;
    gState[slot].store(kReady, std::memory_order_release);
    return UsdGenBackendId(slot);
}

bool UsdGenRegistrar::RegisterDeviceProgram(UsdGenBackendId backend, UsdGenTypeId type,
                                            void const *programDef)
{
    uint32_t const b = UsdGenIndex(backend), t = UsdGenIndex(type);
    if (_ops->IsFrozen() || !programDef || b == 0 || !Ready(b) ||
        t >= UsdGenMutableTypeRegistry().Count() || t >= kUsdGenMaxTypes)
        return false;
    void const *expected = nullptr;
    return gPrograms[b][t].compare_exchange_strong(expected, programDef,
                                                   std::memory_order_acq_rel);
}

UsdGenBackendId UsdGenFindBackend(UsdGenToken const &name) noexcept
{
    return name.IsEmpty() ? UsdGenBackendId::Invalid : UsdGenFindBackend(name.GetText());
}

UsdGenBackendId UsdGenFindBackend(char const *text) noexcept
{
    if (!text || !*text) return UsdGenBackendId::Invalid;
    if (std::strcmp(text, "cpu") == 0 || std::strcmp(text, "cpuReference") == 0)
        return kUsdGenBackendCpu;
    EnsureRegistries();
    uint32_t const slot = FindSlot(text);
    return slot < kUsdGenMaxBackends ? UsdGenBackendId(slot) : UsdGenBackendId::Invalid;
}

UsdGenBackendDef const *UsdGenGetBackend(UsdGenBackendId backend) noexcept
{
    uint32_t const b = UsdGenIndex(backend);
    if (b == 0 || b >= kUsdGenMaxBackends) return nullptr;
    EnsureRegistries();
    return Ready(b) ? &gDefs[b] : nullptr;
}

char const *UsdGenBackendName(UsdGenBackendId backend) noexcept
{
    if (backend == kUsdGenBackendCpu) return "cpu";
    UsdGenBackendDef const *def = UsdGenGetBackend(backend);
    return def ? def->name : "invalid";
}

void const *UsdGenFindDeviceProgram(UsdGenBackendId backend, UsdGenTypeId type) noexcept
{
    uint32_t const b = UsdGenIndex(backend), t = UsdGenIndex(type);
    if (b == 0 || b >= kUsdGenMaxBackends || t >= kUsdGenMaxTypes) return nullptr;
    EnsureRegistries();
    return gPrograms[b][t].load(std::memory_order_acquire);
}

}  // namespace usdGen
