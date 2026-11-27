// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Test interop with pxr types. Math (step 11.4): Gf values convert to the
// usdGenMath types implicitly (same bytes); usdGenUsd::ToHost converts back
// and usdGenUsd::ToEngine / ToEngineValue convert arrays and VtValues the
// engine reads. Hosts never include this header.
#pragma once
#include "usdGenUsd/convert/math.h"

#include "pxr/base/vt/types.h"

using usdGenUsd::ToEngine;
using usdGenUsd::ToEngineValue;
using usdGenUsd::ToHost;
using usdGenUsd::ToHostValue;
