// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/tokenScope.h"

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
    t->result = i("result");
    t->varying = i("varying");
    t->none = i("none");
    t->rightHanded = i("rightHanded");
    t->edgeAndCorner = i("edgeAndCorner");
    t->cornersPlus1 = i("cornersPlus1");
    t->catmullClark = i("catmullClark");
    t->primitive = i("primitive");
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
