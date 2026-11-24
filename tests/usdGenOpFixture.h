// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Hand-driven op tests: the registered record, and a view whose parameters are
// resolved to the record's slots (what the compiler hands an op).
#pragma once
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"

namespace usdGenTest {

/// The registered record of `type`, registering the built-ins first.
inline usdGen::UsdGenOpRecord const &OpRecord(char const *type)
{
    usdGen::usdGenRegisterM1Operators();
    usdGen::UsdGenOpRegistry const &registry = usdGen::UsdGenOpRegistry::Get();
    return registry.Record(registry.Find(PXR_NS::TfToken(type)));
}

/// A view over `node` resolved against `record`. Set() again after the
/// node's parameter vector reallocates; in-place value edits need nothing.
class OpView {
public:
    OpView(usdGen::UsdGenOpRecord const &record, usdGen::UsdGenNodeDesc const *node,
           usdGen::UsdGenGraphDesc const *desc = nullptr)
    {
        _view.desc = desc;
        _view.op = &record;
        _view.slots = _slots;
        Set(node);
    }
    OpView(OpView const &) = delete;
    OpView &operator=(OpView const &) = delete;
    void Set(usdGen::UsdGenNodeDesc const *node)
    {
        _view.node = node;
        if (node) usdGen::UsdGenResolveParams(*_view.op, *node, _slots);
    }
    usdGen::UsdGenParamView const &View() const { return _view; }
    usdGen::UsdGenParamView &View() { return _view; }

private:
    usdGen::UsdGenResolvedParam _slots[usdGen::kUsdGenMaxParamsPerOp];
    usdGen::UsdGenParamView _view;
};

}  // namespace usdGenTest
