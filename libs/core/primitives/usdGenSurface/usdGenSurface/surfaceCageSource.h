// Descriptor-only bridge from captured sparse surface cages to transient C3.
#ifndef USDGEN_SURFACE_CAGE_SOURCE_H
#define USDGEN_SURFACE_CAGE_SOURCE_H

#include "usdGen/graphDesc.h"

#include <cstdint>
#include <string>

namespace usdGen {

/// Conservative dense allocation bound for a captured surface cage.  It
/// validates the same descriptor, sparse rails, owner payload, categorical
/// map relationship, and rest surface as the builder, but never opens or
/// samples the Ptex map.  The actual output can be smaller because owner-map
/// clipping and triangle containment reject roots.  `curves` and `cvs` are
/// left unchanged on failure.
bool UsdGenEstimateSurfaceCageCurveSet(
    UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
    UsdGenCurveSetDesc const &sparse, uint64_t *curves, uint64_t *cvs,
    std::string *error = nullptr);

/// Turns one immutable sparse surface-cage descriptor into a normal dense C3
/// curve set.  The caller chooses the CurveSource mode; this bridge performs
/// no stage reads and no CUDA work.  It requires one `usdGen:regionMap` Ptex
/// binding, a matching expectMapGeneration, one rest surface, and a uniform
/// int `tubeId` primitive plane on the sparse rails. Optional uniform
/// `regionId` and `hierarchyLevel` planes are preserved per leaf owner. The
/// returned C3 is explicitly world-space so scatter density matches Fill
/// under a transformed scalp. It leaves `out` unchanged on every error.
bool UsdGenBuildSurfaceCageCurveSet(
    UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
    UsdGenCurveSetDesc const &sparse, UsdGenCurveSetDesc *out,
    std::string *error = nullptr);

} // namespace usdGen

#endif // USDGEN_SURFACE_CAGE_SOURCE_H
