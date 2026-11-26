#include "usdGen/executionBackend.h"
#include "usdGen/registries/backend.h"

#include "usdGen/op.h"

#include <atomic>
#include <cstring>
#include <string>
#include <utility>

namespace usdGen {

namespace {

// Frozen adapter registry: one slot per backend handle, claimed once by CAS
// (the first registration wins) before UsdGenFreezeRegistries; reads are
// plain acquire loads. Entries live for the process. Constant-initialized and
// trivially destructible, so nothing runs at DLL detach.
struct AdapterEntry {
    std::shared_ptr<const UsdGenExecutionBackendAdapter> adapter;
};
std::atomic<AdapterEntry *> gAdapters[kUsdGenMaxBackends] = {};
std::atomic<bool> gAdaptersFrozen{false};

// A device backend's slot; kUsdGenMaxBackends for the CPU and Invalid.
size_t AdapterSlot(UsdGenBackendId backend) noexcept {
    uint32_t const index = UsdGenIndex(backend);
    return index > 0 && index < kUsdGenMaxBackends ? index : kUsdGenMaxBackends;
}

void SetReason(std::string *reason, char const *message) noexcept {
    if (!reason) return;
    try { *reason = message; } catch (...) {}
}

// The contract of a registered backend, from its def.
UsdGenExecutionBackendContract RegisteredContract(UsdGenBackendId backend,
                                                  UsdGenBackendDef const &def) noexcept {
    bool available = true;
    if (def.available) {
        try { available = def.available(nullptr); } catch (...) { available = false; }
    }
    return {backend, def.name,
            available ? "" : def.unavailableReason ? def.unavailableReason
                           : "the registered backend cannot run in this process",
            def.capabilityVersion,
            available ? UsdGenExecutionBackendAvailability::Available
                      : UsdGenExecutionBackendAvailability::Unavailable};
}

UsdGenExecutionBackendContract InvalidContract() noexcept {
    return {UsdGenBackendId::Invalid, "invalid", "invalid execution backend", 0,
            UsdGenExecutionBackendAvailability::Invalid};
}

} // namespace

std::shared_ptr<const UsdGenExecutionPlanHandle>
UsdGenExecutionPlanHandle::Create(
    UsdGenBackendId backend,
    std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata,
    std::shared_ptr<const void> payload, std::string *reason) {
    if (backend == UsdGenBackendId::Invalid) {
        SetReason(reason, "execution plan handle has an invalid backend");
        return {};
    }
    if (!metadata) {
        SetReason(reason, "execution plan handle has no metadata");
        return {};
    }
    if (metadata->Backend() != UsdGenBackendName(backend)) {
        SetReason(reason, "execution plan metadata backend does not match handle backend");
        return {};
    }
    try {
        return std::shared_ptr<const UsdGenExecutionPlanHandle>(
            new UsdGenExecutionPlanHandle(backend, std::move(metadata),
                                           std::move(payload)));
    } catch (...) {
        SetReason(reason, "execution plan handle allocation failed");
        return {};
    }
}

bool RegisterUsdGenExecutionBackendAdapter(
    std::shared_ptr<const UsdGenExecutionBackendAdapter> adapter,
    std::string *reason) noexcept {
    if (!adapter) {
        SetReason(reason, "cannot register a null execution backend adapter");
        return false;
    }
    UsdGenExecutionBackendContract const contract = adapter->Contract();
    size_t const index = AdapterSlot(contract.backend);
    if (index == kUsdGenMaxBackends ||
        !contract.name || !*contract.name ||
        contract.availability == UsdGenExecutionBackendAvailability::Invalid) {
        SetReason(reason, "execution backend adapter contract is invalid");
        return false;
    }
    // A registered handle keeps its def's name.
    if (UsdGenBackendDef const *def = UsdGenGetBackend(contract.backend)) {
        if (std::strcmp(def->name, contract.name) != 0) {
            SetReason(reason, "execution backend adapter contract name is not canonical");
            return false;
        }
    }
    if (gAdaptersFrozen.load(std::memory_order_acquire)) {
        SetReason(reason, "execution backend registry is frozen");
        return false;
    }
    auto &slot = gAdapters[index];
    if (slot.load(std::memory_order_acquire)) {
        SetReason(reason, "execution backend adapter is already registered");
        return false;
    }
    try {
        auto *entry = new AdapterEntry{std::move(adapter)};
        AdapterEntry *expected = nullptr;
        if (!slot.compare_exchange_strong(expected, entry, std::memory_order_acq_rel)) {
            delete entry;
            SetReason(reason, "execution backend adapter is already registered");
            return false;
        }
        return true;
    } catch (...) {
        SetReason(reason, "execution backend adapter registration failed");
        return false;
    }
}

void UsdGenFreezeExecutionBackendRegistry() noexcept {
    gAdaptersFrozen.store(true, std::memory_order_release);
}

std::shared_ptr<const UsdGenExecutionBackendAdapter>
FindUsdGenExecutionBackendAdapter(UsdGenBackendId backend) noexcept {
    size_t const index = AdapterSlot(backend);
    if (index == kUsdGenMaxBackends) return {};
    AdapterEntry const *entry = gAdapters[index].load(std::memory_order_acquire);
    return entry ? entry->adapter : nullptr;
}

UsdGenExecutionBackendContract GetUsdGenExecutionBackendContract(
    UsdGenBackendId backend) noexcept {
    if (auto adapter = FindUsdGenExecutionBackendAdapter(backend))
        return adapter->Contract();
    if (backend == kUsdGenBackendCpu)
        return {backend, "cpu", "", 1, UsdGenExecutionBackendAvailability::Available};
    if (UsdGenBackendDef const *def = UsdGenGetBackend(backend))
        return RegisteredContract(backend, *def);
    return InvalidContract();
}

bool ValidateUsdGenExecutionBackend(UsdGenBackendId backend,
                                    UsdGenDiagnostics *diagnostics) {
    auto const contract = GetUsdGenExecutionBackendContract(backend);
    if (contract.Available()) return true;
    if (diagnostics) {
        if (!contract.IsValid()) {
            diagnostics->Error("invalid execution backend '" +
                               std::string(UsdGenBackendName(backend)) +
                               "'; refusing implicit CPU fallback");
        } else {
            UsdGenBackendDef const *def = UsdGenGetBackend(backend);
            std::string label = def && def->label ? def->label : contract.name;
            diagnostics->Error(std::move(label) +
                               " execution backend factory is unavailable: " +
                               contract.unavailabilityReason);
        }
    }
    return false;
}

UsdGenExecutionContext MakeUsdGenExecutionContext(
    UsdGenBackendId backend, uint32_t capabilityVersion,
    int32_t deviceIndex, uint64_t deviceGeneration,
    UsdGenContext evaluationContext) noexcept {
    auto const contract = GetUsdGenExecutionBackendContract(backend);
    UsdGenExecutionContext context;
    context.backend = contract.backend;
    context.capabilityVersion = capabilityVersion;
    context.deviceIndex = deviceIndex;
    context.deviceGeneration = deviceGeneration;
    context.evaluationContext = evaluationContext;
    return context;
}

} // namespace usdGen
