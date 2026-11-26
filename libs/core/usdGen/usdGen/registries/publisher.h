// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Publication hook, one per primitive type: builds the type-erased
// publication hosts read. Interim until step 9 replaces it with PublishDesc.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/span.h"
#include "usdGen/exec/epoch.h"
#include "usdGen/exec/pool.h"

#include <cstddef>
#include <cstdint>

namespace usdGen {

struct UsdGenGraphSpec;
struct UsdGenReferenceSet;
struct UsdGenDiagnostics;
class UsdGenWorkDispatcher;

class UsdGenPublication : public UsdGenRefCounted, public UsdGenPooled
{
public:
    explicit UsdGenPublication(UsdGenTypeId t) noexcept : type(t) {}
    UsdGenTypeId const type;             // hosts check it before downcasting
};

struct UsdGenPublishContext
{
    UsdGenGraphSpec const *spec = nullptr;
    UsdGenElementSet const *terminal = nullptr;
    UsdGenSpan<const UsdGenChunkRange> chunks;
    UsdGenSpan<const UsdGenChunkGroup> groups;
    UsdGenSpan<const UsdGenReferenceSet *const> references;   // guide publication
    UsdGenPublication const *previous = nullptr;
    double frame = 0.0;
    UsdGenWorkDispatcher *dispatcher = nullptr;
    UsdGenDiagnostics *diag = nullptr;
};

struct UsdGenPublisherDef
{
    UsdGenTypeId primitive = UsdGenTypeId::Invalid;
    UsdGenRef<const UsdGenPublication> (*build)(UsdGenPublishContext const &) = nullptr;
    UsdGenRef<const UsdGenPublication> (*buildGroup)(UsdGenPublishContext const &,
                                                     uint16_t group) = nullptr;
    UsdGenRef<const UsdGenPublication> (*diff)(UsdGenPublication const *prev,
                                               UsdGenPublication const &next) = nullptr;
    size_t (*bytes)(UsdGenPublication const &) = nullptr;
};

/// The publisher registered for a primitive type; null when none.
USDGEN_CORE_API UsdGenPublisherDef const *UsdGenFindPublisher(UsdGenTypeId primitive) noexcept;

}  // namespace usdGen
