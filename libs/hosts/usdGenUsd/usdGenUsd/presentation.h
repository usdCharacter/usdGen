// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd — a description's presentation (look, purpose, visibility,
// material, basis, preview colours): host data the engine never reads
// (step 9.3). Lower() keeps it in the spec's host data and puts its digest in
// UsdGenGraphSpec::hostDigest, so a presentation edit misses the execution
// cache. The Hydra adapters read it through Presentation() (lower.h).
#ifndef USDGENUSD_PRESENTATION_H
#define USDGENUSD_PRESENTATION_H

#include "usdGen/api/nodeKey.h"
#include "usdGen/api/primRef.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/staticData.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/types.h"

#include <cstdint>

namespace usdGenUsd {

/// A token spelled `Text`: a TfStaticData, zero-initialized and built on
/// first use by CAS (no guarded function static, no static destructor).
template <char const *Text>
struct UsdGenTextTokenFactory {
    static PXR_NS::TfToken *New() { return new PXR_NS::TfToken(Text); }
};
template <char const *Text>
using UsdGenTextToken = PXR_NS::TfStaticData<PXR_NS::TfToken, UsdGenTextTokenFactory<Text>>;

inline constexpr char kPresentationCatmullRomText[] = "catmullRom";
inline constexpr char kPresentationDisplayColorText[] = "displayColor";
inline constexpr char kPresentationBsplineText[] = "bspline";
inline constexpr char kPresentationHeatText[] = "heat";
inline constexpr char kPresentationLitText[] = "lit";
inline constexpr char kPresentationAllPurposeText[] = "allPurpose";
inline UsdGenTextToken<kPresentationCatmullRomText> const s_presentationCatmullRom{};
inline UsdGenTextToken<kPresentationDisplayColorText> const s_presentationDisplayColor{};
inline UsdGenTextToken<kPresentationBsplineText> const s_presentationBspline{};
inline UsdGenTextToken<kPresentationHeatText> const s_presentationHeat{};
inline UsdGenTextToken<kPresentationLitText> const s_presentationLit{};
inline UsdGenTextToken<kPresentationAllPurposeText> const s_presentationAllPurpose{};

}  // namespace usdGenUsd

namespace usdGen {

/// The description's look block (02-schema.md §2.12 defaults; 07 §1.2 bake
/// order): base colour, the root->tip ramp and the per-element jitter.
struct UsdGenLookDesc
{
    PXR_NS::GfVec3f  rootColor{0.035f, 0.018f, 0.008f};
    PXR_NS::GfVec3f  tipColor{0.210f, 0.115f, 0.045f};
    PXR_NS::VtVec3fArray rampColors;      // usdGen:look:colorRamp:colors (empty == two-colour mix)
    PXR_NS::VtFloatArray rampPositions;   // usdGen:look:colorRamp:positions
    PXR_NS::TfToken  rampInterpolation = *usdGenUsd::s_presentationCatmullRom;
    float    rampExponent = 1.0f;         // usdGen:look:rampExponent
    PXR_NS::TfToken  bakeMode;                                             // host-filled (the schema's fallback)
    PXR_NS::TfToken  bakeTarget = *usdGenUsd::s_presentationDisplayColor;  // displayColor | primvar | none
    PXR_NS::TfToken  bakePrimvar = *usdGenUsd::s_presentationDisplayColor;
    float    hueJitter = 0.0f;
    float    valueJitter = 0.0f;
    int      jitterSeed = 0;
};

}  // namespace usdGen

namespace usdGenUsd {

/// What the Hydra adapters publish a description's output with. Keys name
/// prims through the same host data's path table.
struct UsdGenPresentation
{
    usdGen::UsdGenLookDesc look;
    PXR_NS::TfToken purpose;                  // inherited by hand to every tile (C2)
    PXR_NS::TfToken visibility;               // inherited by hand to every tile (C2)
    usdGen::UsdGenNodeKey material;           // the description's bound Material (C2)
    PXR_NS::TfToken basis = *s_presentationBspline;   // the published basis (C2)
    PXR_NS::TfToken previewColorMap = *s_presentationHeat;   // heat | viridis | gray | ids | rgb
    PXR_NS::GfVec2f previewRange{0.0f, 1.0f};
    PXR_NS::TfToken previewShading = *s_presentationLit;     // lit | flat
    // Device publication scalars (DevicePresentation, lower.h).
    usdGen::UsdGenNodeKey description;
    usdGen::UsdGenNodeKey dependency;         // the source's one root-binding surface
    PXR_NS::GfMatrix4d xformMatrix{1.0};
};

/// FNV-1a over every field above: UsdGenGraphSpec::hostDigest.
uint64_t PresentationDigest(UsdGenPresentation const &presentation);

/// The scalars a renderer needs to publish a device generation, from the
/// accepted spec's presentation (never a later scene read).
struct UsdGenDevicePresentationMetadata
{
    usdGen::UsdGenNodeKey description;   // the host composes the render namespace under it
    PXR_NS::GfMatrix4d xformMatrix{1.0};
    PXR_NS::TfToken purpose;
    PXR_NS::TfToken visibility;
    usdGen::UsdGenPrimRef material;
    PXR_NS::TfToken materialPurpose = *s_presentationAllPurpose;
    int refineLevel = 2;
    usdGen::UsdGenPrimRef origin;
    usdGen::UsdGenPrimRef dependency;
};

}  // namespace usdGenUsd

#endif  // USDGENUSD_PRESENTATION_H
