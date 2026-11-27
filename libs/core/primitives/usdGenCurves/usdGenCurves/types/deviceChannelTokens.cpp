// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/types/deviceChannelTokens.h"

#include "usdGen/registries/token.h"

#include <atomic>

namespace usdGen {
namespace {

std::atomic<UsdGenCurveChannelTokens const *> gChannels{nullptr};

UsdGenCurveChannelTokens *Make()
{
    UsdGenCompileTokenScope scope;
    auto *t = new UsdGenCurveChannelTokens;
    t->points = UsdGenInternToken("points");
    t->rest = UsdGenInternToken("rest");
    t->width = UsdGenInternToken("width");
    t->widths = UsdGenInternToken("widths");
    t->hairT = UsdGenInternToken("hairT");
    t->curveOffsets = UsdGenInternToken("curveOffsets");
    t->curveId = UsdGenInternToken("curveId");
    t->stableIds = UsdGenInternToken("stableIds");
    t->skinprim = UsdGenInternToken("skinprim");
    t->skinprimuv = UsdGenInternToken("skinprimuv");
    t->rootPrim = UsdGenInternToken("rootPrim");
    t->rootUV = UsdGenInternToken("rootUV");
    t->sourceRootT = UsdGenInternToken("sourceRootT");
    t->sourceRootB = UsdGenInternToken("sourceRootB");
    t->sourceRootN = UsdGenInternToken("sourceRootN");
    t->restPoints = UsdGenInternToken("restPoints");
    t->sourceChunks = UsdGenInternToken("sourceChunks");
    return t;
}

}  // namespace

UsdGenCurveChannelTokens const &UsdGenCurveChannels() noexcept
{
    if (UsdGenCurveChannelTokens const *t = gChannels.load(std::memory_order_acquire)) return *t;
    UsdGenCurveChannelTokens *fresh = Make();
    UsdGenCurveChannelTokens const *winner = nullptr;
    if (gChannels.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                          std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

}  // namespace usdGen
