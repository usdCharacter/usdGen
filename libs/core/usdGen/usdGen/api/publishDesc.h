// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// How a primitive's output should be published, as API data. Read by the
// Hydra adapters; the engine stores it with the type and never reads it.
#pragma once
#include "usdGen/api/elementSet.h"
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"
#include <cstdint>

namespace usdGen {

struct UsdGenPublishContext;      // registries/publisher.h
struct UsdGenDiagnostics;
struct UsdGenDeformBinding;       // api/deformBinding.h

/// Lines: one-dimensional prims (Hydra basis splines).
enum class UsdGenPublishKind : uint8_t { None, Lines, Points, Mesh, Instances };

/// Values for a viewport value preview (usdGen:preview:*); the host maps them to colours.
struct UsdGenPreviewValues {
    bool active = false;
    uint8_t level = 1;                     // 1 per element, 2 per item
    uint8_t components = 1;                // 3 for the rgb and ids maps
    UsdGenSpan<const double> values;       // valid until the hook state is next used
    /// Per element or item: 0 where the source has no value. Empty: every
    /// entry of `values` is present (none when `values` is empty too).
    UsdGenSpan<const uint8_t> present;
    uint64_t digest = 0;                   // 0 exactly when inactive
};

/// State the primitive's hooks keep across cooks; the publisher's state owns it.
class USDGEN_API UsdGenPublishHookState : public UsdGenPooled {
public:
    virtual ~UsdGenPublishHookState();
};

struct UsdGenPublishDesc {
    UsdGenPublishKind kind = UsdGenPublishKind::None;
    // Lines: the host's type, basis and wrap words.
    UsdGenToken type, basis, wrap;
    UsdGenToken materialClass;              // "hair"
    /// Optional hooks, called from the publisher's build on the cook lane.
    UsdGenPublishHookState *(*createState)() = nullptr;
    /// False: no values (treat as inactive).
    bool (*previewValues)(UsdGenPublishContext const &, UsdGenPublishHookState *,
                          UsdGenPreviewValues *out, UsdGenDiagnostics *) = nullptr;
    /// The index-th surface input whose triangles occlude the elements (one
    /// per mesh, in input order); an invalid key past the last.
    UsdGenNodeKey (*occluder)(UsdGenPublishContext const &, uint32_t index) = nullptr;
    /// Renderer playback binding, one group per ctx.groups entry (allocates:
    /// capture time only). False: none.
    bool (*deformBinding)(UsdGenPublishContext const &,
                          UsdGenRef<const UsdGenDeformBinding> *out) = nullptr;
};

}  // namespace usdGen
