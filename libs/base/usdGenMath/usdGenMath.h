// usdGenMath - pure-math kernels (no USD, no Hydra, no I/O). Gate B-1.
#ifndef USDGEN_MATH_H
#define USDGEN_MATH_H

#include "usdGenMath/math.h"


namespace usdGenMath {

/// Monotonic version string for the math library (stage-free; gate B-1).
/// Renamed from usdGen::GetVersionString (sol S-8): the canonical core
/// declaration lives in usdGen/usdGen.h; the math lib keeps its own name.
char const *GetVersionString();

}  // namespace usdGenMath

#endif  // USDGEN_MATH_H
