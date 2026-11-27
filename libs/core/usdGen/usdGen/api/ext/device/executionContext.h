// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The input versions and device context a backend execution request carries
// (the execution cache keys on them).
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/types.h"

#include <cstdint>
#include <vector>

namespace usdGen {

struct UsdGenGraphSpec;

struct UsdGenExecutionInputGeneration {
    UsdGenNodeKey identity;
    uint64_t generation = 0;

    bool operator==(UsdGenExecutionInputGeneration const& rhs) const noexcept {
        return identity == rhs.identity && generation == rhs.generation;
    }
    bool operator!=(UsdGenExecutionInputGeneration const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

/// Canonical, immutable-after-publication input version tuple.  Source and
/// reference entries are input-set identities; map and surface entries are
/// their corresponding descriptor identities.  Relationship-only references
/// with no payload generation are retained at generation zero so a changed
/// target still changes the plan/cache identity.
struct USDGEN_API UsdGenExecutionInputVersions {
    std::vector<UsdGenExecutionInputGeneration> sources;
    std::vector<UsdGenExecutionInputGeneration> references;
    std::vector<UsdGenExecutionInputGeneration> maps;
    std::vector<UsdGenExecutionInputGeneration> surfaces;
    // False means two descriptor records named the same identity with
    // contradictory nonzero generations. Such a tuple must not be admitted
    // to an execution cache, even though it remains inspectable for
    // diagnostics.
    bool valid = true;

    static UsdGenExecutionInputVersions FromDescription(
        UsdGenGraphSpec const& desc);
    /// The same, refilling *out in place (its vectors keep their capacity).
    static void FromDescription(UsdGenGraphSpec const& desc,
                                UsdGenExecutionInputVersions* out);

    bool operator==(UsdGenExecutionInputVersions const& rhs) const noexcept {
        return valid == rhs.valid && sources == rhs.sources &&
            references == rhs.references && maps == rhs.maps &&
            surfaces == rhs.surfaces;
    }
    bool operator!=(UsdGenExecutionInputVersions const& rhs) const noexcept {
        return !(*this == rhs);
    }

    /// Stable process-independent FNV-1a digest over canonical path and
    /// generation terms.  Equality remains authoritative; this is for hash
    /// tables and cheap cache prefiltering only.
    uint64_t Hash() const noexcept;
    bool IsValid() const noexcept { return valid; }

    // Dirty routing updates are intentionally explicit and return whether an
    // existing identity was found. Unknown targets can still be represented
    // by callers by appending a generation-zero entry before publication.
    bool UpdateInput(UsdGenNodeKey identity, uint64_t generation) noexcept;
    bool UpdateMap(UsdGenNodeKey identity, uint64_t generation) noexcept;
    bool UpdateSurface(UsdGenNodeKey identity, uint64_t generation) noexcept;
};

/// Backend/device/capability context is deliberately separate from the
/// semantic input tuple. A CPU cache entry can therefore be compared without
/// pretending it is portable across incompatible device contexts.
struct UsdGenExecutionContext {
    UsdGenBackendId backend = UsdGenBackendId::Invalid;
    uint32_t capabilityVersion = 0;
    int32_t deviceIndex = -1;
    // Compatibility/context epoch for the selected device runtime. This is
    // not the published geometry generation and should only change when the
    // adapter's compatible execution context changes.
    uint64_t deviceGeneration = 0;
    // Interactive and render evaluation may select different kernels,
    // defaults, or quality policy even on the same backend/device.
    UsdGenContext evaluationContext = UsdGenContext::Interactive;

    bool operator==(UsdGenExecutionContext const& rhs) const noexcept {
        return backend == rhs.backend &&
            capabilityVersion == rhs.capabilityVersion &&
            deviceIndex == rhs.deviceIndex &&
            deviceGeneration == rhs.deviceGeneration &&
            evaluationContext == rhs.evaluationContext;
    }
    bool operator!=(UsdGenExecutionContext const& rhs) const noexcept {
        return !(*this == rhs);
    }
};

}  // namespace usdGen
