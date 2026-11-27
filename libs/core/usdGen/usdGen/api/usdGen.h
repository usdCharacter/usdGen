// usdGen — engine core public header.
//
// M0: minimal surface. The full graph / operator / element-set model
// (UsdGenGraphSpec, UsdGenOp, chunk/tile layout, digests, dirty router) lands
// in M1 (03-execution-engine.md). This header exists so the `usdGen` target has
// a stable public identity and a version symbol, and so `usdGenImaging` /
// `usdGenSchema` can link against the target.
#ifndef USDGEN_USDGEN_H
#define USDGEN_USDGEN_H

#include "usdGen/api/export.h"

#include <string>

namespace usdGen {

/// Monotonic version string for the engine core.
///
/// Canonical M0 version API (sol S-8): exactly one declaration in the
/// installed surface, returning a static C string. libusdGen.so is built with
/// hidden visibility and --exclude-libs,ALL, so the symbol is exported
/// only through the USDGEN_CORE_API macro (see usdGen/export.h).
USDGEN_CORE_API char const *GetVersionString();

/// Chunk size: the engine's dirty/parallel unit is 512 elements (S23).
constexpr int kDefaultChunkSize = 512;
/// Default publication group target (02-schema.md §2.15).
constexpr int kDefaultTileTarget = 64;

}  // namespace usdGen

#endif  // USDGEN_USDGEN_H
