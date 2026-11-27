// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// fastCores.h — big.LITTLE placement policy for usdGen-owned threads.
//
// Threads born spread over every allowed core stick where born, so on
// big.LITTLE silicon some workers run 2-3x slower on little cores while the
// span waits for them. Owners that pin their own threads ask here for the
// max-frequency cores. Everything fails closed: unreadable topology,
// homogeneous frequencies, too few fast cores, or USDGEN_NO_FAST_CORE_PIN=1
// all leave placement to the OS. Non-Linux builds always decline.

#pragma once

#include "usdGen/api/export.h"

#include <vector>

namespace usdGen {

// The process-allowed max-frequency core count where the topology
// qualifies (heterogeneous, readable, not disabled), else 0. Sizing a
// worker set to this count engages pinning by construction.
USDGEN_INTERNAL_API int FastCoreCount();

// The ordered process-allowed max-frequency CPU ids where the topology
// qualifies and the fast count covers `workers`, else empty (declines
// rather than oversubscribing).
USDGEN_INTERNAL_API std::vector<int> FastCoreList(int workers);

} // namespace usdGen
