// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// A published generation: immutable once published (03 §6.1/§6.2).
#pragma once
#include "usdGen/api/epoch.h"
#include "usdGen/api/hostData.h"
#include "usdGen/api/ext/publisher.h"

#include <cstdint>
#include <memory>

namespace usdGen {

class UsdGenDeviceGeneration;

struct UsdGenGeneration
{
    int64_t id = -1;
    double frame = 0.0;
    // What the terminal type's publisher built (UsdGenPublisherDef); null on
    // the device lane and when no publisher is registered.
    UsdGenRef<const UsdGenPublication> publication;
    // Mutually exclusive with host geometry. Device-aware consumers acquire
    // leases from this immutable payload; never cast its pointers to VtArray.
    std::shared_ptr<const UsdGenDeviceGeneration> device;
    // The cooked spec's opaque host data (UsdGenGraphSpec::host): the host
    // names the keys above through it and keeps the presentation it publishes
    // them with (a device generation's included: usdGenUsd DevicePresentation).
    UsdGenHostDataRef host;
};
using UsdGenGenerationConstPtr = std::shared_ptr<const UsdGenGeneration>;

}  // namespace usdGen
