// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenCurves/types/curveTokens.h"
#include "usdGen/registries/token.h"

#include <atomic>

namespace usdGen {

namespace {

std::atomic<UsdGenCurveTokenTable *> gCurveTokens{nullptr};

UsdGenCurveTokenTable *MakeCurveTokens()
{
    auto *t = new UsdGenCurveTokenTable;
    auto i = [](char const *text) { return UsdGenInternToken(text); };
    t->hair = i("hair");
    t->linear = i("linear");
    t->cubic = i("cubic");
    t->bspline = i("bspline");
    t->catmullRom = i("catmullRom");
    t->pinned = i("pinned");
    t->nonperiodic = i("nonperiodic");
    t->periodic = i("periodic");
    t->vertexWidths = i("vertex");
    t->useRest = i("useRest");
    t->usdGenUseRest = i("usdGen:useRest");
    t->resampleTo = i("resampleTo");
    t->idSource = i("idSource");
    t->staleAction = i("staleAction");
    t->rebind = i("rebind");
    t->expectEpoch = i("expectEpoch");
    t->primvar = i("primvar");
    t->index = i("index");
    t->warn = i("warn");
    t->ignore = i("ignore");
    t->block = i("block");
    t->never = i("never");
    t->onError = i("onError");
    t->always = i("always");
    t->sourceColor = i("usdGenSourceColor");
    t->strandSeed = i("strandSeed");
    t->strandIndex = i("strandIndex");
    t->tubeId = i("tubeId");
    t->regionId = i("regionId");
    t->hierarchyLevel = i("hierarchyLevel");
    t->cudaScatterGrow = i("CudaScatterGrow");
    t->cudaPublication = i("CudaPublication");
    t->previewDestination = i("usdGenPreview");
    t->groom = i("groom");
    t->point = i("point");
    t->map = i("map");
    t->mapChannel = i("map:channel");
    t->mapFirstChannel = i("map:firstChannel");
    t->mapChannelCount = i("map:channelCount");
    t->previewChannels[0] = i("previewR");
    t->previewChannels[1] = i("previewG");
    t->previewChannels[2] = i("previewB");
    t->rgbChannels[0] = i("r");
    t->rgbChannels[1] = i("g");
    t->rgbChannels[2] = i("b");
    return t;
}

}  // namespace

// CAS-published like the engine tokens; never freed (read until exit).
UsdGenCurveTokenTable const &UsdGenCurveTokens() noexcept
{
    if (UsdGenCurveTokenTable *t = gCurveTokens.load(std::memory_order_acquire)) return *t;
    UsdGenCurveTokenTable *fresh = MakeCurveTokens();
    UsdGenCurveTokenTable *winner = nullptr;
    if (gCurveTokens.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                             std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

}  // namespace usdGen
