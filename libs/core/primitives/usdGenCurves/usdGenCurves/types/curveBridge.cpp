// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Bridge (5.3a; 5.4 deletes): buffer-backed set views.
#include "usdGenCurves/types/curveBridge.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGen/registries/engineTokens.h"

#include <algorithm>

namespace usdGen {

namespace {

UsdGenChannelDesc PlaneDesc(UsdGenPlane const &p, bool cv, UsdGenEngineTokens const &tk) noexcept
{
    UsdGenChannelDesc d;
    d.name = p.name;
    d.scalar = p.type == tk.int_ ? UsdGenScalar::I32 : UsdGenScalar::F32;
    d.arity = std::max<uint8_t>(p.arity, 1);
    d.level = cv ? 2 : p.interpolation == tk.constant ? 0 : 1;
    d.flags = d.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
    return d;
}

bool SameDesc(UsdGenChannelDesc const &a, UsdGenChannelDesc const &b) noexcept
{
    return a.name == b.name && a.scalar == b.scalar && a.arity == b.arity &&
           a.level == b.level && a.flags == b.flags;
}

template <class A>
void Put(UsdGenPoolVector<void const *> &columns, UsdGenPoolVector<uint32_t> &values,
         UsdGenSlot slot, A const &a, uint32_t arity) noexcept
{
    columns[slot] = a.empty() ? nullptr : static_cast<void const *>(a.cdata());
    values[slot] = uint32_t(a.size() * arity);
}

}  // namespace

void UsdGenCurveBufferSetView::Bind(UsdGenCurveBuffer const &b)
{
    UsdGenEngineTokens const &tk = UsdGenTokens();
    size_t const planes = b.extraCurve.size() + b.extraCv.size();
    bool same = _table && _dynamic.size() == planes;
    for (size_t k = 0; same && k < b.extraCurve.size(); ++k)
        same = SameDesc(_dynamic[k], PlaneDesc(b.extraCurve[k], false, tk));
    for (size_t k = 0; same && k < b.extraCv.size(); ++k)
        same = SameDesc(_dynamic[b.extraCurve.size() + k], PlaneDesc(b.extraCv[k], true, tk));
    if (!same) {
        // Layout change (topology time): a new table, slots by name.
        _dynamic.clear();
        for (UsdGenPlane const &p : b.extraCurve) _dynamic.push_back(PlaneDesc(p, false, tk));
        for (UsdGenPlane const &p : b.extraCv) _dynamic.push_back(PlaneDesc(p, true, tk));
        _table = UsdGenChannelTable::Make(UsdGenCurveSetLayout(),
            UsdGenSpan<const UsdGenChannelDesc>(_dynamic.data(), _dynamic.size()));
        _planeSlots.clear();
        for (UsdGenChannelDesc const &d : _dynamic)
            _planeSlots.push_back(_table ? _table->Find(d.name) : kUsdGenNoSlot);
    }
    size_t const slots = _table ? _table->Size() : 0;
    _columns.assign(slots, nullptr);
    _values.assign(slots, 0u);
    if (slots >= kUsdGenCurveStaticSlots) {
        Put(_columns, _values, kUsdGenCurvePx, b.px, 1);
        Put(_columns, _values, kUsdGenCurvePy, b.py, 1);
        Put(_columns, _values, kUsdGenCurvePz, b.pz, 1);
        Put(_columns, _values, kUsdGenCurveRest, b.rest, 3);
        Put(_columns, _values, kUsdGenCurveWidth, b.width, 1);
        Put(_columns, _values, kUsdGenCurveHairT, b.hairT, 1);
        Put(_columns, _values, kUsdGenCurveId, b.curveId, 1);
        Put(_columns, _values, kUsdGenCurveRootPrim, b.rootPrim, 1);
        Put(_columns, _values, kUsdGenCurveRootUV, b.rootUV, 2);
        Put(_columns, _values, kUsdGenCurveRootT, b.rootT, 3);
        Put(_columns, _values, kUsdGenCurveRootN, b.rootN, 3);
        Put(_columns, _values, kUsdGenCurveRootB, b.rootB, 3);
        size_t k = 0;
        for (int cv = 0; cv < 2; ++cv) {
            for (UsdGenPlane const &p : cv ? b.extraCv : b.extraCurve) {
                UsdGenSlot const s = _planeSlots[k++];
                if (s == kUsdGenNoSlot || s >= slots) continue;
                if (p.type == tk.int_) Put(_columns, _values, s, p.i, 1);
                else Put(_columns, _values, s, p.f, 1);
            }
        }
    }
    _view = UsdGenElementSetView{};
    _view.channels = _table.Get();
    _view.count[1] = b.totalCurves;
    _view.count[2] = b.totalCvs;
    _view.offsets[2] = b.cvOffsets.empty() ? nullptr
        : reinterpret_cast<uint32_t const *>(b.cvOffsets.cdata());
    _view.columns = _columns.data();
    _view.values = _values.data();
    _view.topologyVersion = b.topologyVersion;
    _view.valueVersion = b.valueVersion;
    _view.legacy = &b;
}

}  // namespace usdGen
