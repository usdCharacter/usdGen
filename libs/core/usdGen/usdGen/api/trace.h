// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Trace scopes and counters. Alias phase (step 11.1): Tf's TRACE_* macros;
// the trace swap (11.8) routes them through a host-installed sink.
// Scope names stay literals, so bin/trace_playback.ps1 filters still match.
#pragma once
#include "pxr/base/trace/trace.h"

#define USDGEN_TRACE_SCOPE(name) TRACE_SCOPE(name)
#define USDGEN_TRACE_FUNCTION() TRACE_FUNCTION()
#define USDGEN_TRACE_COUNTER(name, delta) TRACE_COUNTER_DELTA(name, delta)
