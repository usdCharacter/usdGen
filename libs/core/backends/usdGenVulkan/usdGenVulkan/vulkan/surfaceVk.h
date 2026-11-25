// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// surfaceVk.h — Phase 2 (p2-surface): non-identity surface binding for Deform.
//
// The Vulkan Deform lane freezes its RBF fit at plan compile: host-side
// farthest-point sampling plus rest/posed sample pairs and per-curve root
// targets, all consumed by deformEvaluate/deformApply at execution. The
// admitted identity configs keep their existing host-gather path in
// executionPlan.cpp untouched; this module serves only moved source/surface
// transforms, which that path rejects.
//
// Space policy: ordinary C3 points stay source-local, and every point
// operator consumes source-local data (curveRootCapture.cpp converts
// rest-surface frames through the surface-to-source relative affine for the
// same reason). The samples and root targets compiled here are therefore
// mapped surface-local -> source-local by
// surface.worldMatrix * source.worldMatrix^-1, so the frozen fit, the
// pipeline input CVs, and the root targets share one space. A CUDA twin
// built by baking the same transforms into the points (identity matrices)
// computes the identical fit in world space; rigid-equivariance of the
// cubic RBF fit makes the two outputs agree up to float rounding.

#pragma once

#include "usdGen/graphDesc.h"
#include "usdGen/vulkan/executionPlan.h"

namespace usdGen::vulkan {

// Compiles Deform sample/root controls for arbitrary finite affine
// invertible source/surface transforms in source-local space. `budget` is
// the already-parsed rbfSamples literal ([4,100]); the already-parsed
// lockRoots/mask/groomEnvelope members of `out` are preserved, only
// sampleCount/restSamples/posedSamples/rootTargets are written. Returns
// false with a "Vulkan source-width plan: ..." diagnostic on rejection,
// using the same messages as ValidateDeform where the check coincides.
bool CompileSurfaceVkDeform(UsdGenCurveSetDesc const& source,
                            UsdGenSurfaceDesc const& surface, int budget,
                            VulkanDeformControls* out,
                            UsdGenDiagnostics* diagnostics);

} // namespace usdGen::vulkan
