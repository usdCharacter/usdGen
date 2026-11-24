// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen - base for ops registered by def. Until op.h carries the record
// (S4-8) this holds it and answers the static questions from it, so the def is
// the one place an op declares its parameters, planes and traits.
#ifndef USDGEN_OPS_RECORDED_OP_H
#define USDGEN_OPS_RECORDED_OP_H

#include "usdGen/op.h"

namespace usdGen {

class UsdGenRecordedOp : public UsdGenOp
{
public:
    explicit UsdGenRecordedOp(UsdGenOpRecord const &record)
        : _record(&record)
    {
        for (UsdGenParamSlot s = 0; s < record.ParamCount(); ++s)
            (record.ParamClass(s) == UsdGenParamClass::Value ? _value : _topo)
                .push_back(record.ParamName(s));
        for (UsdGenPlane const &p : record.InputPlanes()) _inputs.push_back(p.name);
        for (UsdGenPlane const &p : record.OutputPlanes()) _outputs.push_back(p.name);
    }

    UsdGenOpRecord const &Record() const noexcept { return *_record; }

    TfToken Type() const final { return _record->Name(); }
    UsdGenTopoFx TopologyEffect() const final { return _record->Def().topoFx; }
    UsdGenRole Role() const final { return _record->Def().role; }
    using UsdGenOp::ReadsTime;
    bool ReadsTime() const final { return _record->Has(kUsdGenOpReadsTime); }
    bool SamplesFrameInputs() const final { return _record->Has(kUsdGenOpSamplesFrameInputs); }
    bool IsGenerator() const final { return _record->Has(kUsdGenOpGenerator); }
    bool CaptureDigestComplete() const final { return _record->Has(kUsdGenOpCaptureDigestComplete); }
    size_t GeometryInputArity() const final { return _record->Def().geometryInputArity; }
    TfSpan<const TfToken> TopologyParameters() const final
    {
        return TfSpan<const TfToken>(_topo.data(), _topo.size());
    }
    TfSpan<const TfToken> ValueParameters() const final
    {
        return TfSpan<const TfToken>(_value.data(), _value.size());
    }
    TfSpan<const TfToken> ReferenceInputs() const final { return _record->ReferenceInputs(); }
    TfSpan<const TfToken> OutputPrimvars() const final
    {
        return TfSpan<const TfToken>(_outputs.data(), _outputs.size());
    }
    TfSpan<const TfToken> InputPrimvars() const final
    {
        return TfSpan<const TfToken>(_inputs.data(), _inputs.size());
    }

private:
    UsdGenOpRecord const *_record;
    TfTokenVector _topo, _value, _inputs, _outputs;
};

/// A copy of the slot's authored value, empty when unauthored.
inline VtValue UsdGenAuthoredValue(UsdGenParamView const &p, UsdGenParamSlot s)
{
    UsdGenResolvedParam const *r = p.Slot(s);
    return r && r->authored ? r->authored->value : VtValue();
}

}  // namespace usdGen

#endif  // USDGEN_OPS_RECORDED_OP_H
