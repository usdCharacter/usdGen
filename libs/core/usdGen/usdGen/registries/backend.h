// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Execution backends by handle. The engine knows a backend only through the
// def its library registers from usdGenRegister; CPU is built in.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/token.h"
#include "usdGen/registries/handles.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace usdGen {

struct UsdGenGraphSpec;
struct UsdGenDiagnostics;
class UsdGenSessionDeviceProvider;

constexpr UsdGenBackendId kUsdGenBackendCpu = UsdGenBackendId(0);

enum UsdGenBackendFlag : uint32_t {
    /// Results carry the revisions the request supplied; a cache hit
    /// republishes with RepublishPreservingRevisions.
    kUsdGenBackendAuthoritativeRevisions = 1u << 0,
    /// A result may report a poisoned workspace; the cooker then invalidates
    /// the cache domain.
    kUsdGenBackendPoisonable = 1u << 1,
    /// Publishing needs a device-aware consumer.
    kUsdGenBackendNeedsDevicePublication = 1u << 2,
};

struct UsdGenBackendProviderDesc {
    int callerDevice = -1;                   // captureCallerDevice at the Session API
};

struct UsdGenBackendDef {
    char const *name = nullptr;              // the usdGen:executionBackend token
    char const *label = nullptr;             // diagnostics spelling; null: name
    uint32_t capabilityVersion = 0;          // part of the execution-cache context
    uint32_t flags = 0;                      // UsdGenBackendFlag
    /// False with a reason when this process cannot run the backend. Null: true.
    bool (*available)(std::string *reason) = nullptr;
    /// Compile-time admission of a graph.
    bool (*validate)(UsdGenGraphSpec const &, UsdGenDiagnostics *) = nullptr;
    /// The caller's device at the Session API boundary, -2 on failure. Null: -1.
    int (*captureCallerDevice)() noexcept = nullptr;
    /// One provider per Session executor. Runs on a scheduler task, never on
    /// the cook lane: device creation blocks.
    std::shared_ptr<UsdGenSessionDeviceProvider> (*createProvider)(
        UsdGenBackendProviderDesc const &, std::string *reason) = nullptr;
};

/// kUsdGenBackendCpu for "cpu" and "cpuReference", the handle of a registered
/// name, else Invalid. Compile and lowering phase only.
USDGEN_CORE_API UsdGenBackendId UsdGenFindBackend(UsdGenToken const &name) noexcept;
/// Null for kUsdGenBackendCpu and Invalid.
USDGEN_CORE_API UsdGenBackendDef const *UsdGenGetBackend(UsdGenBackendId) noexcept;
/// "cpu", a registered name, or "invalid".
USDGEN_CORE_API char const *UsdGenBackendName(UsdGenBackendId) noexcept;
/// The opaque program def registered for (backend, primitive type); null if none.
USDGEN_CORE_API void const *UsdGenFindDeviceProgram(UsdGenBackendId, UsdGenTypeId) noexcept;

/// Directory of the module holding 'anchor' (any object or function of it),
/// with a trailing separator, NUL-terminated in buffer. Its length; 0 when
/// unknown or longer than capacity - 1.
USDGEN_CORE_API size_t UsdGenModuleDirectory(void const *anchor, char *buffer,
                                             size_t capacity) noexcept;

}  // namespace usdGen
