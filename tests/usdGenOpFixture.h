// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Hand-driven op tests: the registered record, and a view whose parameters are
// resolved to the record's slots (what the compiler hands an op).
#pragma once
#include "utils/pxrInterop.h"
#include "usdGen/registries/op.h"
#include "usdGenUsd/lower.h"
#include "usdGen/registries/opRegistry.h"

#include <string>
#include <vector>

namespace usdGenTest {

/// The registered record of `type`, registering the built-ins first.
inline usdGen::UsdGenOpRecord const &OpRecord(char const *type)
{
    usdGen::UsdGenOpRegistry const &registry = usdGen::UsdGenOpRegistry::Get();
    return registry.Record(registry.Find(PXR_NS::TfToken(type)));
}

/// A view over `node` resolved against `record`: lowered copies of the
/// host node and graph, with each resolved slot pointing back into the host
/// node's params, so in-place value edits show as they always did. Set()
/// again after the node's parameter vector reallocates.
class OpView {
public:
    OpView(usdGen::UsdGenOpRecord const &record, usdGen::UsdGenNodeDesc const *node,
           usdGen::UsdGenGraphDesc const *desc = nullptr)
        : _srcDesc(desc)
    {
        _view.op = &record;
        _view.slots = _slots;
        Set(node);
    }
    OpView(OpView const &) = delete;
    OpView &operator=(OpView const &) = delete;
    void Set(usdGen::UsdGenNodeDesc const *node)
    {
        if (_srcDesc) _spec = usdGenUsd::Lower(*_srcDesc);
        _view.desc = _srcDesc ? &_spec : nullptr;
        _view.node = nullptr;
        if (!node) return;
        _node = usdGenUsd::Lower(*node);
        _view.node = &_node;
        usdGen::UsdGenResolveParams(*_view.op, _node, _slots);
        for (usdGen::UsdGenResolvedParam &slot : _slots)
            if (slot.authored)
                slot.authored = &node->params[size_t(slot.authored - _node.params.data())];
    }
    usdGen::UsdGenParamView const &View() const { return _view; }
    usdGen::UsdGenParamView &View() { return _view; }

private:
    usdGen::UsdGenGraphDesc const *_srcDesc = nullptr;
    usdGen::UsdGenOpNode _node;
    usdGen::UsdGenGraphSpec _spec;
    usdGen::UsdGenResolvedParam _slots[usdGen::kUsdGenMaxParamsPerOp];
    usdGen::UsdGenParamView _view;
};

/// By-name literal reads of a node's authored params (exact spelling, the
/// first wins: the compiler's slot rule), for host-side test oracles.
class NodeParams {
public:
    explicit NodeParams(usdGen::UsdGenNodeDesc const &node) : _node(&node) {}
    PXR_NS::VtValue GetVtValue(PXR_NS::TfToken const &name, PXR_NS::VtValue const &fallback) const
    {
        for (usdGen::UsdGenParamValue const &p : _node->params)
            if (p.name == name) return p.value;
        return fallback;
    }
    double GetDouble(PXR_NS::TfToken const &name, double fallback) const
    {
        PXR_NS::VtValue const v = GetVtValue(name, PXR_NS::VtValue());
        if (v.IsHolding<double>()) return v.UncheckedGet<double>();
        if (v.IsHolding<float>()) return v.UncheckedGet<float>();
        return fallback;
    }
    int GetInt(PXR_NS::TfToken const &name, int fallback) const
    {
        PXR_NS::VtValue const v = GetVtValue(name, PXR_NS::VtValue());
        if (v.IsHolding<int>()) return v.UncheckedGet<int>();
        if (v.IsHolding<uint32_t>()) return static_cast<int>(v.UncheckedGet<uint32_t>());
        return fallback;
    }
    bool GetBool(PXR_NS::TfToken const &name, bool fallback) const
    {
        PXR_NS::VtValue const v = GetVtValue(name, PXR_NS::VtValue());
        return v.IsHolding<bool>() ? v.UncheckedGet<bool>() : fallback;
    }
    PXR_NS::TfToken GetToken(PXR_NS::TfToken const &name, PXR_NS::TfToken const &fallback) const
    {
        PXR_NS::VtValue const v = GetVtValue(name, PXR_NS::VtValue());
        if (v.IsHolding<PXR_NS::TfToken>()) return v.UncheckedGet<PXR_NS::TfToken>();
        if (v.IsHolding<std::string>()) return PXR_NS::TfToken(v.UncheckedGet<std::string>());
        return fallback;
    }

private:
    usdGen::UsdGenNodeDesc const *_node;
};

/// The record's routed parameter names, split as the routing snapshot splits
/// them: value == true gives the Value-class decls, false the rest.
inline std::vector<PXR_NS::TfToken> RoutedParams(usdGen::UsdGenOpRecord const &record, bool value)
{
    std::vector<PXR_NS::TfToken> out;
    for (usdGen::UsdGenParamSlot s = 0; s < record.ParamCount(); ++s)
        if (record.Def().params[s].routed &&
            (record.ParamClass(s) == usdGen::UsdGenParamClass::Value) == value)
            out.push_back(record.ParamName(s));
    return out;
}

/// The output planes the compiler binds for a node of this op with `params`:
/// the emittedPlanes hook (indexed run at its default) or the static planes.
inline std::vector<PXR_NS::TfToken> OutputPlaneNames(
    usdGen::UsdGenOpRecord const &record,
    std::vector<usdGen::UsdGenParamValue> const &params = {})
{
    std::vector<PXR_NS::TfToken> out;
    if (auto const emit = record.Def().hooks.emittedPlanes) {
        struct Sink final : usdGen::UsdGenPlaneSink {
            std::vector<PXR_NS::TfToken> *names;
            void Add(std::string const &name, usdGen::UsdGenPlaneRate, usdGen::UsdGenPlaneScalar,
                     uint8_t) override { names->push_back(PXR_NS::TfToken(name)); }
        } sink;
        sink.names = &out;
        usdGen::UsdGenNodeDesc node;
        node.params = params;
        usdGen::UsdGenGraphDesc const desc;
        OpView view(record, &node, &desc);
        emit(view.View(), -1, sink);
        return out;
    }
    for (usdGen::UsdGenChannelDesc const &plane : record.OutputPlanes()) out.push_back(plane.name);
    return out;
}

}  // namespace usdGenTest
