// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/token.h"

#include <atomic>

namespace usdGen {

namespace {

std::atomic<UsdGenEngineTokens *> gEngineTokens{nullptr};

UsdGenEngineTokens *MakeEngineTokens()
{
    auto *t = new UsdGenEngineTokens;
    auto i = [](char const *text) { return UsdGenInternToken(text); };
    t->vertex = i("vertex");
    t->uniform = i("uniform");
    t->constant = i("constant");
    t->float_ = i("float");
    t->int_ = i("int");
    t->input = i("input");
    t->surface = i("surface");
    t->seed = i("seed");
    t->enabled = i("enabled");
    t->guides = i("guides");
    t->mask = i("mask");
    t->usdGenPrefix = i("usdGen:");
    t->hair = i("hair");
    t->cubic = i("cubic");
    t->bspline = i("bspline");
    t->catmullRom = i("catmullRom");
    t->pinned = i("pinned");
    t->nonperiodic = i("nonperiodic");
    t->vertexWidths = i("vertex");
    t->allPurpose = i("allPurpose");
    t->none = i("none");
    t->displayColor = i("displayColor");
    t->hairT = i("hairT");
    t->hairId = i("hairId");
    t->st = i("st");
    t->bakeColor = i("bakeColor");
    t->velocities = i("velocities");
    t->scalpShadow = i("scalpShadow");
    t->materialScalpShadow = i("material_scalpShadow");
    t->result = i("result");
    t->varying = i("varying");
    t->rightHanded = i("rightHanded");
    t->edgeAndCorner = i("edgeAndCorner");
    t->cornersPlus1 = i("cornersPlus1");
    t->catmullClark = i("catmullClark");
    t->heat = i("heat");
    t->primitive = i("primitive");
    t->lit = i("lit");
    return t;
}

}  // namespace

// CAS-published like the registries; never freed (read until exit).
UsdGenEngineTokens const &UsdGenTokens() noexcept
{
    if (UsdGenEngineTokens *t = gEngineTokens.load(std::memory_order_acquire)) return *t;
    UsdGenEngineTokens *fresh = MakeEngineTokens();
    UsdGenEngineTokens *winner = nullptr;
    if (gEngineTokens.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                              std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

}  // namespace usdGen
