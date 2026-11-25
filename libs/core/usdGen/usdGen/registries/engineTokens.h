// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Engine vocabulary that names no op, type or primitive.
#pragma once
#include "usdGen/export.h"
#include "pxr/base/tf/token.h"

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {

struct UsdGenEngineTokens {
    // plane rate / scalar (UsdGenPlane, types.h)
    TfToken vertex, uniform, constant, float_, int_;
    // base node parameters (compiler.cpp:779-801) and the "usdGen:" prefix
    TfToken input, surface, seed, enabled, guides, curves, mask, usdGenPrefix;
    // C3 curve-set vocabulary (graphDesc.h:224-235, cudaExecution.cpp:8376-8398)
    TfToken hair, cubic, bspline, catmullRom, pinned, nonperiodic, vertexWidths;
    // publication vocabulary; step 9 moves it (sessionCooker.cpp:678-2882,
    // generationStore.cpp:163-180, tile publication, graphDesc.h:124)
    TfToken renderNamespace /* __usdGenRender */, allPurpose, none, displayColor,
        hairT, hairId, st, bakeColor, velocities, scalpShadow, materialScalpShadow,
        result;
    // backend names (executionBackend.cpp:141-148)
    TfToken cpu, cpuReference, cuda, metal, vulkan;
    // S4-3 (Core-B): varying rate; desc defaults (graphDesc.h surface, look, preview)
    TfToken varying, rightHanded, edgeAndCorner, cornersPlus1, catmullClark, perCurve,
        heat, primitive, lit;
};

/// Valid from the first registry access for the life of the process.
USDGEN_CORE_API UsdGenEngineTokens const &UsdGenTokens() noexcept;

}  // namespace usdGen
