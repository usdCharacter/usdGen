#include "usdGen/executionBackend.h"
#include "usdGen/registries/engineTokens.h"

#include "usdGen/op.h"

#include <atomic>
#include <cstring>
#include <string>
#include <utility>

namespace usdGen {

// Defined by executionBackendCudaProvider.cpp /
// executionBackendVulkanProvider.cpp. Keeping these tiny providers separate
// means the neutral backend contract does not include a concrete adapter
// header or know its capability-matrix type.
uint32_t UsdGenCudaCapabilityVersionForBackendContract() noexcept;
uint32_t UsdGenVulkanCapabilityVersionForBackendContract() noexcept;

namespace {

// Frozen backend registry: one slot per backend, claimed once by CAS (the
// first registration wins) before UsdGenFreezeRegistries; reads are plain
// acquire loads. Entries live for the process. Constant-initialized and
// trivially destructible, so nothing runs at DLL detach.
struct AdapterEntry {
    std::shared_ptr<const UsdGenExecutionBackendAdapter> adapter;
};
constexpr size_t kBackendSlots = 8;
std::atomic<AdapterEntry *> gAdapters[kBackendSlots] = {};
std::atomic<bool> gAdaptersFrozen{false};

size_t BackendSlot(UsdGenExecutionBackend backend) noexcept {
    return static_cast<size_t>(backend) % kBackendSlots;
}

void SetReason(std::string *reason, char const *message) noexcept {
    if (!reason) return;
    try { *reason = message; } catch (...) {}
}

UsdGenExecutionBackendContract InvalidContract() noexcept {
    return {UsdGenExecutionBackend::Invalid, UsdGenDeviceBackend::Unknown,
            "invalid", "invalid execution backend", 0,
            UsdGenExecutionBackendAvailability::Invalid};
}

} // namespace

std::shared_ptr<const UsdGenExecutionPlanHandle>
UsdGenExecutionPlanHandle::Create(
    UsdGenExecutionBackend backend,
    std::shared_ptr<const UsdGenExecutionPlanMetadata> metadata,
    std::shared_ptr<const void> payload, std::string *reason) {
    if (backend == UsdGenExecutionBackend::Invalid) {
        SetReason(reason, "execution plan handle has an invalid backend");
        return {};
    }
    if (!metadata) {
        SetReason(reason, "execution plan handle has no metadata");
        return {};
    }
    if (metadata->Backend() != UsdGenExecutionBackendName(backend)) {
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
    if (contract.backend == UsdGenExecutionBackend::Invalid ||
        contract.deviceBackend == UsdGenDeviceBackend::Unknown ||
        !contract.name || !*contract.name ||
        contract.availability == UsdGenExecutionBackendAvailability::Invalid) {
        SetReason(reason, "execution backend adapter contract is invalid");
        return false;
    }
    if (std::strcmp(UsdGenExecutionBackendName(contract.backend), contract.name) != 0) {
        SetReason(reason, "execution backend adapter contract name is not canonical");
        return false;
    }
    if (gAdaptersFrozen.load(std::memory_order_acquire)) {
        SetReason(reason, "execution backend registry is frozen");
        return false;
    }
    auto &slot = gAdapters[BackendSlot(contract.backend)];
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
FindUsdGenExecutionBackendAdapter(UsdGenExecutionBackend backend) noexcept {
    if (backend == UsdGenExecutionBackend::Invalid) return {};
    AdapterEntry const *entry = gAdapters[BackendSlot(backend)].load(std::memory_order_acquire);
    return entry ? entry->adapter : nullptr;
}

const char *UsdGenExecutionBackendName(UsdGenExecutionBackend backend) noexcept {
    switch (backend) {
    case UsdGenExecutionBackend::CpuReference: return "cpu";
    case UsdGenExecutionBackend::Cuda: return "cuda";
    case UsdGenExecutionBackend::Metal: return "metal";
    case UsdGenExecutionBackend::Vulkan: return "vulkan";
    case UsdGenExecutionBackend::Invalid: break;
    }
    return "invalid";
}

UsdGenExecutionBackend ParseUsdGenExecutionBackend(TfToken const &token) noexcept {
    if (token == UsdGenTokens().cpu || token == UsdGenTokens().cpuReference)
        return UsdGenExecutionBackend::CpuReference;
    if (token == UsdGenTokens().cuda) return UsdGenExecutionBackend::Cuda;
    if (token == UsdGenTokens().metal) return UsdGenExecutionBackend::Metal;
    if (token == UsdGenTokens().vulkan) return UsdGenExecutionBackend::Vulkan;
    return UsdGenExecutionBackend::Invalid;
}

UsdGenExecutionBackendContract GetUsdGenExecutionBackendContract(
    UsdGenExecutionBackend backend) noexcept {
    if (auto adapter = FindUsdGenExecutionBackendAdapter(backend))
        return adapter->Contract();
    switch (backend) {
    case UsdGenExecutionBackend::CpuReference:
        return {backend, UsdGenDeviceBackend::CpuReference, "cpu", "",
                1, UsdGenExecutionBackendAvailability::Available};
    case UsdGenExecutionBackend::Cuda:
#ifdef USDGEN_ENABLE_CUDA
        return {backend, UsdGenDeviceBackend::Cuda, "cuda", "",
                UsdGenCudaCapabilityVersionForBackendContract(),
                UsdGenExecutionBackendAvailability::Available};
#else
        return {backend, UsdGenDeviceBackend::Cuda, "cuda",
                "CUDA support was not enabled in this build",
                UsdGenCudaCapabilityVersionForBackendContract(),
                UsdGenExecutionBackendAvailability::Unavailable};
#endif
    case UsdGenExecutionBackend::Metal:
        return {backend, UsdGenDeviceBackend::Metal, "metal",
                "no Metal execution factory is linked", 0,
                UsdGenExecutionBackendAvailability::Unavailable};
    case UsdGenExecutionBackend::Vulkan:
#ifdef USDGEN_ENABLE_VULKAN_RUNTIME
        return {backend, UsdGenDeviceBackend::Vulkan, "vulkan", "",
                UsdGenVulkanCapabilityVersionForBackendContract(),
                UsdGenExecutionBackendAvailability::Available};
#else
        return {backend, UsdGenDeviceBackend::Vulkan, "vulkan",
                "Vulkan support was not enabled in this build",
                UsdGenVulkanCapabilityVersionForBackendContract(),
                UsdGenExecutionBackendAvailability::Unavailable};
#endif
    case UsdGenExecutionBackend::Invalid: break;
    }
    return InvalidContract();
}

bool ValidateUsdGenExecutionBackend(UsdGenExecutionBackend backend,
                                    UsdGenDiagnostics *diagnostics) {
    auto const contract = GetUsdGenExecutionBackendContract(backend);
    if (contract.Available()) return true;
    if (diagnostics) {
        if (!contract.IsValid()) {
            diagnostics->Error("invalid execution backend '" +
                               std::string(UsdGenExecutionBackendName(backend)) +
                               "'; refusing implicit CPU fallback");
        } else {
            std::string label = contract.name;
            if (backend == UsdGenExecutionBackend::Cuda) label = "CUDA";
            else if (backend == UsdGenExecutionBackend::Metal) label = "Metal";
            else if (backend == UsdGenExecutionBackend::Vulkan) label = "Vulkan";
            diagnostics->Error(std::move(label) +
                               " execution backend factory is unavailable: " +
                               contract.unavailabilityReason);
        }
    }
    return false;
}

UsdGenExecutionContext MakeUsdGenExecutionContext(
    UsdGenExecutionBackend backend, uint32_t capabilityVersion,
    int32_t deviceIndex, uint64_t deviceGeneration,
    UsdGenContext evaluationContext) noexcept {
    auto const contract = GetUsdGenExecutionBackendContract(backend);
    UsdGenExecutionContext context;
    context.backend = contract.deviceBackend;
    context.capabilityVersion = capabilityVersion;
    context.deviceIndex = deviceIndex;
    context.deviceGeneration = deviceGeneration;
    context.evaluationContext = evaluationContext;
    return context;
}

} // namespace usdGen
