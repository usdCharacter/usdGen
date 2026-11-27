// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Registers the Hydra adapters' publishers before a test's first compile
// (registries freeze then). Header-only: a test that calls it links
// usdGenHydraCurves; usdGenTestUtils itself stays Hydra-free.
#ifndef USDGEN_TESTUTILS_HYDRA_PUBLISHERS_H
#define USDGEN_TESTUTILS_HYDRA_PUBLISHERS_H

#include "usdGenHydraCurves/register.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace usdGenTest {

/// Call first in main. Exits 1 with the reason when registration is too late.
inline void EnsureHydraPublishers()
{
    std::string error;
    if (!usdGenHydra::UsdGenHydraCurvesEnsureRegistered(&error)) {
        std::fprintf(stderr, "EnsureHydraPublishers: %s\n", error.c_str());
        std::exit(1);
    }
}

}  // namespace usdGenTest

#endif  // USDGEN_TESTUTILS_HYDRA_PUBLISHERS_H
