// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Per-(op, backend) kernel rows. Backends register them; lanes look them up.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace usdGen {

struct UsdGenParamView;
class UsdGenBackendRegistry;
USDGEN_API void UsdGenFreezeRegistries();   // registrar.h
USDGEN_API UsdGenBackendRegistry &UsdGenMutableBackendRegistry() noexcept;   // registrar-only

struct UsdGenKernelDef {
    uint16_t stage = 0;                  // the backend's own stage enum
    uint32_t capabilityVersion = 0;      // executionPlan.h:46
    int minimumAlgorithmVersion = 0, maximumAlgorithmVersion = 0;
    uint32_t flags = 0;                  // UsdGenExecutionCapabilityFlag, executionPlan.h:26-39
    UsdGenSpan<const UsdGenParamSlot> paramMap;   // backend param index -> op slot
    bool (*profileLut)(UsdGenParamView const &, float *lut, size_t size) = nullptr;
};

class USDGEN_API UsdGenBackendRegistry {
public:
    static UsdGenBackendRegistry const &Get() noexcept;
    /// null: this backend cannot run the op; callers fail closed with a diagnostic.
    UsdGenKernelDef const *Find(UsdGenOpId, UsdGenBackendId) const noexcept;
    /// Ops with a kernel on the backend, in registration order (capability rows).
    UsdGenSpan<const UsdGenOpId> Ops(UsdGenBackendId) const noexcept;
private:
    friend class UsdGenRegistrar;
    friend void UsdGenFreezeRegistries();
    friend UsdGenBackendRegistry &UsdGenMutableBackendRegistry() noexcept;
    UsdGenBackendRegistry();
    UsdGenKernelDef _kernels[kUsdGenMaxBackends][kUsdGenMaxOps];
    bool _present[kUsdGenMaxBackends][kUsdGenMaxOps] = {};
    UsdGenOpId _order[kUsdGenMaxBackends][kUsdGenMaxOps];
    uint32_t _orderCount[kUsdGenMaxBackends] = {};
    std::atomic<bool> _frozen{false};
};

}  // namespace usdGen
