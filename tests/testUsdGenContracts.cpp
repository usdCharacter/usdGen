// testUsdGenContracts — gate SI-* registry contract (docs/freezes/C1.md).
//
// C1 is FROZEN at M1: the usdGen prim type names, property names, value
// types, and fallback defaults pinned in docs/freezes/C1.md (authoritative;
// supersedes the abstract/API summary in docs/m1/interfaces.md §7) must match
// the live UsdSchemaRegistry / UsdPrimDefinition as loaded from
// libs/hosts/usdGenUsd/schema/resources/generatedSchema.usda.
//
// Uniform encoding note: the usda `uniform` qualifier surfaces as the
// attribute's SdfVariability (SdfVariabilityUniform). In USD 26.08 neither
// the value-type token ("int", "token", …) nor a "uniform" metadata key
// carries it on the loaded definition — verified live. (02 §2.2/§2.3, C1.md
// Uniform column.)
//
// Any FAIL line here is a C1 break: rename/retyping/default change requires
// a new contract version, not an M1 edit (02 §8.1/§8.5). No skip path: this
// test always asserts.

#include "utils/pxrInterop.h"
#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/types.h"
#include "pxr/usd/usd/primDefinition.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/usd/usd/schemaRegistry.h"

#include "usdGen/compiler.h"
#include "usdGenUsd/lower.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGenImaging/usdGenDirtyRouter.h"

#include <cstdio>
#include <memory>
#include <sstream>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int g_failures = 0;

void Check(bool ok, std::string const &what)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    } else {
        std::printf("ok:   %s\n", what.c_str());
    }
}

// Render an attribute's fallback default as a stable string for comparison:
// floats via %g (1.0 -> "1", 0.01 -> "0.01"), ints via ostream, bool ->
// "1"/"0", string/token -> contents. |ok| reports whether any fallback was
// readable at all (so "missing default" fails, distinct from "wrong value").
std::string FallbackToString(UsdPrimDefinition::Attribute const &attr,
                             bool *ok)
{
    float f;
    double d;
    std::int64_t i;
    int n;
    bool b;
    std::string s;
    TfToken t;
    char buf[64];
    if (attr.GetFallbackValue(&f)) {
        std::snprintf(buf, sizeof buf, "%g", static_cast<double>(f));
        *ok = true; return buf;
    }
    if (attr.GetFallbackValue(&d)) {
        std::snprintf(buf, sizeof buf, "%g", d);
        *ok = true; return buf;
    }
    if (attr.GetFallbackValue(&i)) {
        std::ostringstream os; os << i;
        *ok = true; return os.str();
    }
    if (attr.GetFallbackValue(&n)) {
        std::ostringstream os; os << n;
        *ok = true; return os.str();
    }
    if (attr.GetFallbackValue(&b)) {
        *ok = true; return b ? "1" : "0";
    }
    if (attr.GetFallbackValue(&s)) { *ok = true; return s; }
    if (attr.GetFallbackValue(&t)) { *ok = true; return t.GetString(); }
    GfVec2f v2;
    if (attr.GetFallbackValue(&v2)) {
        std::snprintf(buf, sizeof buf, "(%g, %g)",
                      static_cast<double>(v2[0]), static_cast<double>(v2[1]));
        *ok = true; return buf;
    }
    GfVec3f v3;
    if (attr.GetFallbackValue(&v3)) {
        std::snprintf(buf, sizeof buf, "(%g, %g, %g)",
                      static_cast<double>(v3[0]), static_cast<double>(v3[1]),
                      static_cast<double>(v3[2]));
        *ok = true; return buf;
    }
    *ok = false;
    return std::string();
}

// Assert |name| is an attribute of |pd| with value-type token |typeName| and
// fallback default stringifying to |defaultText|.
void CheckAttr(UsdPrimDefinition const *pd,
               std::string const &name,
               std::string const &typeName,
               std::string const &defaultText,
               bool uniformExpected = false)
{
    const TfToken prop(name);
    UsdPrimDefinition::Attribute attr = pd->GetAttributeDefinition(prop);
    if (!attr) {
        Check(false, name + ": not an attribute of prim definition");
        return;
    }
    Check(attr.GetTypeNameToken().GetString() == typeName,
          name + ": value-type token is \"" +
              attr.GetTypeNameToken().GetString() + "\", contract pins \"" +
              typeName + "\"");
    // Uniformity == SdfVariability (verified live: the metadata-key probe
    // returns nothing; GetVariability() is the 26.08 accessor).
    const SdfVariability var = attr.GetVariability();
    Check(var == (uniformExpected ? SdfVariabilityUniform
                                  : SdfVariabilityVarying),
          name + std::string(uniformExpected ? ": expected uniform, but "
                                            : ": expected non-uniform, but ") +
              "variability=" +
              (var == SdfVariabilityUniform ? "uniform" : "varying"));
    bool haveFallback = false;
    std::string got = FallbackToString(attr, &haveFallback);
    Check(haveFallback && got == defaultText,
          name + ": fallback default is " +
              (haveFallback ? "\"" + got + "\"" : "<none>") +
              ", contract pins \"" + defaultText + "\"");
}

// Assert |name| is a relationship of |pd| (and not also an attribute).
void CheckRel(UsdPrimDefinition const *pd, std::string const &name)
{
    const TfToken prop(name);
    UsdPrimDefinition::Relationship rel = pd->GetRelationshipDefinition(prop);
    Check(static_cast<bool>(rel), name + ": not a relationship");
    if (static_cast<bool>(pd->GetAttributeDefinition(prop))) {
        Check(false, name + ": defined as attribute as well as relationship");
    }
}

void CheckConcrete(std::string const &typeName)
{
    Check(UsdSchemaRegistry::IsConcrete(TfToken(typeName)),
          typeName + ": not registered concrete");
}

void CheckAbstract(std::string const &typeName)
{
    Check(UsdSchemaRegistry::IsAbstract(TfToken(typeName)),
          typeName + ": not registered abstract");
}

void CheckSingleApplyAPI(std::string const &typeName)
{
    const TfToken tok(typeName);
    Check(UsdSchemaRegistry::IsAppliedAPISchema(tok) &&
              !UsdSchemaRegistry::IsMultipleApplyAPISchema(tok),
          typeName + ": not a single-apply API schema");
}

} // namespace

int main()
{
    // C1 §1 — frozen prim type names (36 concrete + 5 abstract + 4 API).
    const char *kConcrete[] = {
        "UsdGenGroom", "UsdGenDescription", "UsdGenGuideSet", "UsdGenScatter",
        "UsdGenGrow", "UsdGenGuideInterpolate", "UsdGenReferenceSource",
        "UsdGenCurveSource",
        "UsdGenClump", "UsdGenNoise", "UsdGenLength", "UsdGenWidth",
        "UsdGenDirection", "UsdGenSmooth", "UsdGenResample", "UsdGenScale",
        "UsdGenCurl", "UsdGenBend", "UsdGenStraighten", "UsdGenDisplace",
        "UsdGenWave", "UsdGenPart", "UsdGenExprOp", "UsdGenSculptLayer",
        "UsdGenDeform", "UsdGenCollide", "UsdGenWind", "UsdGenFreeze",
        "UsdGenInstance", "UsdGenImageMap", "UsdGenPtexMap", "UsdGenExprMap",
        "UsdGenPaintMap", "UsdGenNoiseMap", "UsdGenCombineMap",
        "UsdGenGuideProximityMap",
    };
    for (const char *name : kConcrete)
        CheckConcrete(name);

    for (const char *name :
         { "UsdGenOperator", "UsdGenGenerator", "UsdGenStyler",
           "UsdGenDeformer", "UsdGenMap" })
        CheckAbstract(name);

    for (const char *name : { "UsdGenLookAPI",
                              "UsdGenRestAPI", "UsdGenCurveAPI" })
        CheckSingleApplyAPI(name);

    // C1 §2 — UsdGenGroom property rows.
    if (UsdPrimDefinition const *groom = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenGroom"))) {
        CheckAttr(groom, "usdGen:execution:backend", "token", "cuda",
                  /*uniform*/ true);
        CheckAttr(groom, "usdGen:sessionId", "string", "", /*uniform*/ true);
    } else {
        Check(false, "UsdGenGroom: no concrete prim definition (schema not loaded?)");
    }

    // C1 §2 — UsdGenDescription property rows.
    if (UsdPrimDefinition const *desc = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenDescription"))) {
        CheckRel(desc, "usdGen:surface");
        CheckRel(desc, "usdGen:guides");
        CheckAttr(desc, "usdGen:tileTarget", "int", "64", /*uniform*/ true);
        CheckAttr(desc, "usdGen:curve:basis", "token", "bspline", /*uniform*/ true);
        CheckAttr(desc, "usdGen:width:default", "float", "0.01");
    } else {
        Check(false, "UsdGenDescription: no concrete prim definition (schema not loaded?)");
    }

    // C1 §2 — UsdGenOperator (abstract) property rows. Every operator type
    // inherits exactly these two.
    if (UsdPrimDefinition const *op = UsdSchemaRegistry::GetInstance()
            .FindAbstractPrimDefinition(TfToken("UsdGenOperator"))) {
        CheckAttr(op, "usdGen:enabled", "bool", "1");
        CheckAttr(op, "usdGen:seed", "int", "0", /*uniform*/ true);
    } else {
        Check(false, "UsdGenOperator: no abstract prim definition (schema not loaded?)");
    }

    // C1 §2 — the operator envelope. usdGen:mask replaced UsdGenMaskAPI and
    // is declared on the two abstract groups that have upstream curves; a
    // generator must NOT inherit it.
    for (char const *group : { "UsdGenStyler", "UsdGenDeformer" }) {
        if (UsdPrimDefinition const *abstractDef = UsdSchemaRegistry::GetInstance()
                .FindAbstractPrimDefinition(TfToken(group))) {
            CheckAttr(abstractDef, "usdGen:mask", "float", "1");
        } else {
            Check(false, std::string(group) + ": no abstract prim definition");
        }
    }
    if (UsdPrimDefinition const *generator = UsdSchemaRegistry::GetInstance()
            .FindAbstractPrimDefinition(TfToken("UsdGenGenerator"))) {
        Check(!static_cast<bool>(
                  generator->GetAttributeDefinition(TfToken("usdGen:mask"))),
              "UsdGenGenerator: generators declare no usdGen:mask");
    } else {
        Check(false, "UsdGenGenerator: no abstract prim definition");
    }
    if (UsdPrimDefinition const *noise = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenNoise"))) {
        CheckAttr(noise, "usdGen:mask", "float", "1");
    } else {
        Check(false, "UsdGenNoise: no concrete prim definition");
    }
    if (UsdPrimDefinition const *scatterDef = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenScatter"))) {
        Check(!static_cast<bool>(
                  scatterDef->GetAttributeDefinition(TfToken("usdGen:mask"))),
              "UsdGenScatter: a generator inherits no usdGen:mask");
    }

    // C1 §2 — property rows of the operator types that have kernels.
    if (UsdPrimDefinition const *scatter = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenScatter"))) {
        CheckAttr(scatter, "usdGen:density", "float", "100");
        CheckAttr(scatter, "usdGen:flip", "bool", "0");
    } else {
        Check(false, "UsdGenScatter: no concrete prim definition");
    }
    if (UsdPrimDefinition const *grow = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenGrow"))) {
        CheckAttr(grow, "usdGen:segments", "int", "8");
        CheckAttr(grow, "usdGen:length", "float", "1");
        CheckAttr(grow, "usdGen:lengthRandom", "float2", "(1, 1)");
        CheckAttr(grow, "usdGen:direction", "token", "surfaceNormal", /*uniform*/ true);
        CheckAttr(grow, "usdGen:directionVector", "vector3f", "(0, 1, 0)");
        CheckAttr(grow, "usdGen:lift", "float", "0");
    } else {
        Check(false, "UsdGenGrow: no concrete prim definition");
    }
    if (UsdPrimDefinition const *referenceSource = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenReferenceSource"))) {
        CheckRel(referenceSource, "usdGen:reference");
    } else {
        Check(false, "UsdGenReferenceSource: no concrete prim definition");
    }
    if (UsdPrimDefinition const *source = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenCurveSource"))) {
        CheckRel(source, "usdGen:curves");
        CheckAttr(source, "usdGen:useRest", "bool", "1");
        CheckAttr(source, "usdGen:idSource", "token", "primvar", /*uniform*/ true);
        CheckAttr(source, "usdGen:expectEpoch", "string", "", /*uniform*/ true);
        CheckAttr(source, "usdGen:staleAction", "token", "warn", /*uniform*/ true);
        CheckAttr(source, "usdGen:resampleTo", "int", "0", /*uniform*/ true);
        CheckAttr(source, "usdGen:rebind", "token", "onError", /*uniform*/ true);
    } else {
        Check(false, "UsdGenCurveSource: no concrete prim definition");
    }
    if (UsdPrimDefinition const *width = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenWidth"))) {
        CheckAttr(width, "usdGen:width", "float", "0.01");
        CheckAttr(width, "usdGen:width:interpolation", "token", "catmullRom",
                  /*uniform*/ true);
        CheckAttr(width, "usdGen:replace", "bool", "1");
    } else {
        Check(false, "UsdGenWidth: no concrete prim definition");
    }
    if (UsdPrimDefinition const *length = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenLength"))) {
        CheckAttr(length, "usdGen:length:mode", "token", "scale", /*uniform*/ true);
        CheckAttr(length, "usdGen:length:value", "float", "1");
        CheckAttr(length, "usdGen:length:random", "float2", "(1, 1)");
        CheckAttr(length, "usdGen:length:method", "token", "scale", /*uniform*/ true);
        CheckAttr(length, "usdGen:rebuild", "token", "keepParam", /*uniform*/ true);
        CheckAttr(length, "usdGen:minRemainingLength", "float", "0");
        CheckAttr(length, "usdGen:cullThreshold", "float", "0");
    } else {
        Check(false, "UsdGenLength: no concrete prim definition");
    }
    if (UsdPrimDefinition const *deform = UsdSchemaRegistry::GetInstance()
            .FindConcretePrimDefinition(TfToken("UsdGenDeform"))) {
        CheckAttr(deform, "usdGen:lockRoots", "bool", "1");
        CheckAttr(deform, "usdGen:rbfSamples", "int", "100");
    } else {
        Check(false, "UsdGenDeform: no concrete prim definition");
    }

    // The imaging router must be able to retain its pure routing input while
    // the mutable compiled graph is replaced.  This is deliberately kept in
    // the contract test because it also guards the public graph boundary:
    // snapshots may not contain pointers into nodes, operators or Desc().
    usdGen::UsdGenGraphDesc routingDesc;
    routingDesc.description = SdfPath("/routing");
    routingDesc.terminal = SdfPath("/routing/source");
    usdGen::UsdGenNodeDesc routingNode;
    routingNode.path = SdfPath("/routing/source");
    routingNode.type = TfToken("UsdGenCurveSource");
    routingNode.params.push_back({TfToken("resampleTo"), VtValue(4), false});
    routingDesc.nodes.push_back(std::move(routingNode));
    usdGen::UsdGenCompiler routingCompiler;
    std::shared_ptr<const usdGen::UsdGenGraphRoutingSnapshot> retained;
    size_t retainedEntryCount = 0;
    usdGenImaging::UsdGenDirtyRouter router;
    {
        usdGen::UsdGenGraph routingGraph;
        auto routingResult = usdGenUsd::Compile(routingCompiler, routingDesc, &routingGraph);
        Check(routingResult.ok, "routing snapshot graph compiles");
        if (!routingResult.ok) retained.reset();
        else retained = routingGraph.RoutingSnapshot();
        if (retained) {
            Check(retained && retained->description == usdGenUsd::KeyOf(routingDesc.description) &&
                  retained->nodes.size() == 1 &&
                  retained->nodes.front().key == usdGenUsd::KeyOf(routingDesc.nodes.front().path) &&
                  !retained->nodes.front().paramRouting.empty(),
                  "routing snapshot owns description/node/classification data");
            router.Rebuild(*retained);
            retainedEntryCount = router.EntryCount();

            // Recompile into the same graph before destroying it.  The retained
            // snapshot must remain usable after both operations.
            routingDesc.description = SdfPath("/replacement");
            routingDesc.terminal = SdfPath("/replacement/source");
            routingDesc.nodes.front().path = SdfPath("/replacement/source");
            Check(usdGenUsd::Compile(routingCompiler, routingDesc, &routingGraph).ok,
                  "mutable graph can be recompiled after snapshot capture");
        }
    }
    if (retained) {
        router.Rebuild(*retained);
        Check(router.EntryCount() == retainedEntryCount,
              "router rebuild from retained snapshot survives graph destruction");
        usdGen::UsdGenPendingDirty routed;
        router.Route({{SdfPath("/routing/source"), HdDataSourceLocatorSet(
            HdDataSourceLocator(TfToken("usdGen")).Append(TfToken("resampleTo")))}},
            &routed);
        Check(!routed.structural && routed.nodeBits.size() == 1 &&
                  routed.nodeBits.begin()->first == retained->nodes.front().id &&
                  routed.nodeBits.begin()->second == usdGen::UsdGenDirtyCapture,
              "retained router still routes the original source path and dirty class");
    }

    // Default-time rest normal data is a live authored binding input, not a
    // frame-value deformation. It therefore re-captures every consuming
    // surface-bound node; current primvars:normals stays intentionally out
    // of this route until a separately specified current-normal feature.
    usdGenUsd::PathRecorder routingKeys;   // the hand-built snapshot's path table
    usdGen::UsdGenGraphRoutingSnapshot surfaceRouting;
    surfaceRouting.description = routingKeys(SdfPath("/routing"));
    surfaceRouting.terminal = 37;
    surfaceRouting.surfaceKeys = {routingKeys(SdfPath("/routing/surface"))};
    surfaceRouting.nodes.push_back(usdGen::UsdGenGraphRoutingNode{
        37, routingKeys(SdfPath("/routing/source")), TfToken("UsdGenCurveSource"),
        {}, {}, {}, 0, true, {}, {}});
    surfaceRouting.host = routingKeys.Table();
    router.Rebuild(surfaceRouting);
    auto const checkRestNormalRoute = [&](TfToken const &leaf,
                                          char const *what) {
        usdGen::UsdGenPendingDirty routed;
        router.Route({{SdfPath("/routing/surface"), HdDataSourceLocatorSet(
            HdDataSourceLocator(TfToken("usdGen")).Append(TfToken("rest")).Append(leaf))}},
            &routed);
        Check(!routed.structural && routed.surfaceBits.empty() &&
                  routed.nodeBits.size() == 1 &&
                  routed.nodeBits[37] == usdGen::UsdGenDirtyCapture,
              what);
    };
    checkRestNormalRoute(TfToken("normals"),
                         "rest normals dirty re-captures surface consumer");
    checkRestNormalRoute(TfToken("normalsInterpolation"),
                         "rest normal interpolation dirty re-captures surface consumer");
    {
        usdGen::UsdGenPendingDirty routed;
        router.Route({{SdfPath("/routing/surface"), HdDataSourceLocatorSet(
            HdDataSourceLocator(TfToken("primvars")).Append(TfToken("normals")).
                Append(TfToken("primvarValue")))}}, &routed);
        Check(!routed.structural && routed.nodeBits.empty() &&
                  routed.surfaceBits.empty(),
              "current-frame primvars:normals does not re-capture rest binding");
    }

    if (g_failures != 0) {
        std::printf("%d contract FAILURES — C1 break (docs/freezes/C1.md)\n",
                    g_failures);
        return 1;
    }
    std::printf("C1 registry contract holds\n");
    return 0;
}
