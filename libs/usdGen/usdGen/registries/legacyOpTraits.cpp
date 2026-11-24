// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The one place op names live between S4-1 and S4-8; S4-8 deletes it.
// IsDigestParam/IsTopologyParam are compiler.cpp's tables, verbatim.
#include "usdGen/registries/legacyOpTraits.h"
#include "usdGen/op.h"

#include <deque>
#include <string>
#include <vector>

namespace usdGen {

namespace {

/// §6.1: a term of the node's Merkle structural digest (compiler.cpp IsDigestParam).
bool IsDigestParam(std::string const &type, std::string const &param)
{
    if (type == "UsdGenGrow") return param == "segments" || param == "direction";
    if (type == "UsdGenLength")
        return param == "length:method" || param == "rebuild" || param == "length:mode";
    if (type == "UsdGenWidth") return param == "replace";
    if (type == "UsdGenClump") return param == "clump:level" || param == "clump:levels";
    if (type == "UsdGenCurl" || type == "UsdGenBend") return param == "axisMode";
    if (type == "UsdGenPart") return param == "part:curves";
    if (type == "UsdGenDirection") return param == "mode" || param == "direction:source";
    if (type == "UsdGenSmooth") return param == "mode";
    if (type == "UsdGenDisplace") return param == "mode" || param == "displace:map";
    if (type == "UsdGenResample") return param == "distribution";
    if (type == "UsdGenSculptLayer") return param == "sculpt:space";
    if (type == "UsdGenExprOp")
        return param == "mode" || param == "expr:maps" || param == "expr:returnType";
    if (type == "UsdGenFreeze") return param == "frozen:mode" || param == "frozen:curves";
    if (type == "UsdGenCollide") return param == "resolveType";
    return false;
}

/// §6.2: topology-class rows (compiler.cpp IsTopologyParam).
bool IsTopologyParam(std::string const &type, std::string const &param)
{
    if (type == "UsdGenLength") return param == "length:mode" || param == "cullThreshold";
    return false;
}

/// Traits that were op-name checks in the engine (design section 1, item 7).
uint32_t NamedTraits(std::string const &type)
{
    uint32_t traits = 0;
    if (type == "UsdGenGrow" || type == "UsdGenResample" ||
        type == "UsdGenGuideInterpolate" || type == "UsdGenFreeze")
        traits |= kUsdGenOpCaptureOwnsRestAndPlanes;          // scheduler.cpp:386-392, 440-446
    if (type == "UsdGenDeform")
        traits |= kUsdGenOpRecordsChunkExtents | kUsdGenOpPosesSpace;   // scheduler.cpp:1031, 1076
    if (type == "UsdGenLength") traits |= kUsdGenOpFollowsInputUntilOwning;   // graph.cpp:438
    if (type == "UsdGenClump") traits |= kUsdGenOpFixedOutputPlaneLayout;     // scheduler.cpp:228-236
    return traits;
}

bool IsScatter(std::string const &type) { return type == "UsdGenScatter"; }

/// Scatter roots carry limit-surface UVs (cpuParameters.cpp:353-356).
bool ScatterEmitsLimitSurfaceUv(UsdGenParamView const &view)
{
    if (!view.node) return false;
    for (UsdGenParamValue const &p : view.node->params)
        if (p.name.GetString() == "subdivisionLevel" && p.value.IsHolding<int>())
            return p.value.UncheckedGet<int>() > 0;
    return false;
}

constexpr UsdGenMapSlotDecl kLegacyMapSlots[] = {    // compiler.cpp:428-431
    {"usdGen:map"}, {"usdGen:regionMap"}, {"usdGen:densityMap"}, {"usdGen:lengthMap"}};

struct LegacyOp {
    UsdGenOpDef def;
    std::function<std::unique_ptr<UsdGenOp>()> factory;
    std::string name, displayName;
    std::deque<std::string> strings;     // stable c_str() for the decls
    std::vector<UsdGenParamDecl> params;
    std::vector<char const *> references;
};

std::unique_ptr<UsdGenOp> CreateLegacy(UsdGenOpRecord const &, void *context)
{
    return static_cast<LegacyOp *>(context)->factory();
}

}  // namespace

UsdGenOpDef const *UsdGenMakeLegacyOpDef(TfToken const &type,
                                         std::function<std::unique_ptr<UsdGenOp>()> factory)
{
    if (type.IsEmpty() || !factory) return nullptr;
    std::unique_ptr<UsdGenOp> probe = factory();
    if (!probe) return nullptr;
    auto legacy = std::make_unique<LegacyOp>();
    LegacyOp &l = *legacy;
    l.factory = std::move(factory);
    l.name = type.GetString();
    l.displayName = l.name.compare(0, 6, "UsdGen") == 0 ? l.name.substr(6) : l.name;
    auto keep = [&l](TfToken const &t) {
        l.strings.push_back(t.GetString());
        return l.strings.back().c_str();
    };
    for (TfToken const &p : probe->TopologyParameters()) {
        std::string const &n = p.GetString();
        UsdGenParamClass const cls = IsDigestParam(l.name, n) ? UsdGenParamClass::Structural
            : IsTopologyParam(l.name, n) ? UsdGenParamClass::Topology
                                         : UsdGenParamClass::Capture;
        l.params.push_back(UsdGenParamDecl{keep(p), cls});
    }
    for (TfToken const &p : probe->ValueParameters())
        l.params.push_back(UsdGenParamDecl{keep(p), UsdGenParamClass::Value});
    for (TfToken const &r : probe->ReferenceInputs()) l.references.push_back(keep(r));

    UsdGenOpDef &d = l.def;
    d.name = l.name.c_str();
    d.displayName = l.displayName.c_str();
    d.create = &CreateLegacy;
    d.createContext = &l;
    d.topoFx = probe->TopologyEffect();
    d.role = probe->Role();
    d.geometryInputArity = uint8_t(probe->GeometryInputArity());
    d.traits = NamedTraits(l.name);
    if (probe->IsGenerator()) d.traits |= kUsdGenOpGenerator;
    if (probe->ReadsTime()) d.traits |= kUsdGenOpReadsTime;
    if (probe->SamplesFrameInputs()) d.traits |= kUsdGenOpSamplesFrameInputs;
    if (probe->GeometryInputsOrdered()) d.traits |= kUsdGenOpGeometryInputsOrdered;
    if (probe->CaptureDigestComplete()) d.traits |= kUsdGenOpCaptureDigestComplete;
    if (probe->RemapsVertexPlanes()) d.traits |= kUsdGenOpRemapsVertexPlanes;
    d.params = TfSpan<const UsdGenParamDecl>(l.params.data(), l.params.size());
    d.referenceInputs = TfSpan<char const *const>(l.references.data(), l.references.size());
    d.mapSlots = UsdGenSpan(kLegacyMapSlots);
    if (IsScatter(l.name)) d.hooks.emitsLimitSurfaceUv = &ScatterEmitsLimitSurfaceUv;
    return &legacy.release()->def;
}

void UsdGenDiscardLegacyOpDef(UsdGenOpDef const *def)
{
    if (def) delete static_cast<LegacyOp *>(def->createContext);
}

}  // namespace usdGen
