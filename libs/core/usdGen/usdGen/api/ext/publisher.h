// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Publication hook, one per primitive type: builds the type-erased
// publication hosts read. Interim until step 9 replaces it with PublishDesc.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace usdGen {

struct UsdGenGraphSpec;
struct UsdGenReferenceSet;
struct UsdGenDiagnostics;
class UsdGenGraph;
class UsdGenWorkDispatcher;

class UsdGenPublication : public UsdGenRefCounted, public UsdGenPooled
{
public:
    explicit UsdGenPublication(UsdGenTypeId t, uint8_t k = 0) noexcept : type(t), kind(k) {}
    UsdGenTypeId const type;             // hosts check it before downcasting
    uint8_t const kind;                  // publisher-defined (publication, report, ...)
    uint32_t groups = 0;                 // published groups (stats, progress gating)
};

/// Per-cook-owner state a publisher keeps across cooks (caches, warnings).
/// Owned by the cooker; only `build` sees it, on the cook lane.
class UsdGenPublisherState : public UsdGenPooled
{
public:
    virtual ~UsdGenPublisherState() = default;
};

/// A group built while the terminal still ran (progress); `build` reuses it.
struct UsdGenPublishedGroup
{
    uint16_t group = 0;
    UsdGenPublication const *publication = nullptr;
};

struct UsdGenPublishContext
{
    UsdGenGraphSpec const *spec = nullptr;
    UsdGenGraph const *graph = nullptr;  // the compiled graph that produced `terminal`
    UsdGenElementSet const *terminal = nullptr;
    UsdGenSpan<const UsdGenChunkRange> chunks;
    UsdGenSpan<const UsdGenChunkGroup> groups;
    UsdGenSpan<const UsdGenReferenceSet *const> references;   // guide publication
    UsdGenSpan<const UsdGenPublishedGroup> streamed;          // sorted by group, first wins
    UsdGenPublication const *previous = nullptr;
    UsdGenPublisherState *state = nullptr;
    double frame = 0.0;
    bool topologyChanged = false;        // every group rebuilds
    bool rebuildAll = false;             // `previous` is not this graph's last run
    bool pointsOnly = false;             // only points moved since `previous` was built
    UsdGenWorkDispatcher *dispatcher = nullptr;
    UsdGenDiagnostics *diag = nullptr;
    /// Optional: each group `build` builds (not carried over, not streamed), in
    /// build order, as a one-group publication.
    void (*emit)(void *user, UsdGenRef<const UsdGenPublication> group) = nullptr;
    void *emitUser = nullptr;
    /// Optional: phase marks for the cook trace (rebuilt SIZE_MAX: no counts).
    void (*phase)(void *user, char const *name, size_t rebuilt, size_t total) = nullptr;
    void *phaseUser = nullptr;
};

struct UsdGenPublisherDef
{
    UsdGenTypeId primitive = UsdGenTypeId::Invalid;
    UsdGenRef<const UsdGenPublication> (*build)(UsdGenPublishContext const &) = nullptr;
    /// One group of ctx.groups, from any worker; ctx.state is null.
    UsdGenRef<const UsdGenPublication> (*buildGroup)(UsdGenPublishContext const &,
                                                     uint16_t group) = nullptr;
    UsdGenRef<const UsdGenPublication> (*diff)(UsdGenPublication const *prev,
                                               UsdGenPublication const &next) = nullptr;
    /// Cache charge; 0 when it cannot be computed.
    size_t (*bytes)(UsdGenPublication const &) = nullptr;
    std::unique_ptr<UsdGenPublisherState> (*createState)() = nullptr;
};

/// The publisher registered for a primitive type; null when none.
USDGEN_CORE_API UsdGenPublisherDef const *UsdGenFindPublisher(UsdGenTypeId primitive) noexcept;

}  // namespace usdGen
