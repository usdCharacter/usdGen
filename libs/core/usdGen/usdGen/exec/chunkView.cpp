// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Compile check for chunkView.h. Emits nothing: the legacy curve types of
// the same names (curveBuffer.h) stay in other TUs until 5.3a/5.4.
#include "usdGen/exec/chunkView.h"

#include <type_traits>

namespace usdGen {
static_assert(std::is_standard_layout<UsdGenChunkView>::value, "chunk view is plain data");
static_assert(std::is_trivially_copyable<UsdGenChunkExtent>::value, "chunk extent is plain data");
}  // namespace usdGen
