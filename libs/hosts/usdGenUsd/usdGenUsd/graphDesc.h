// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGenUsd — UsdGenGraphDesc, the path-based graph a host builder produces
// (step 7.2: moved out of the engine, shape unchanged). Lower() (lower.h)
// turns it into the engine's UsdGenGraphSpec. Payloads without an identity
// are the engine's own types (usdGen/api/graphSpec.h).
#ifndef USDGENUSD_GRAPH_DESC_H
#define USDGENUSD_GRAPH_DESC_H

#include "usdGen/api/graphSpec.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenSurface/surfaceInput.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"

#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// One input:<name> relationship of a UsdGenExpression prim. 'targets' is the
/// authored (forwarded) target list; the builder resolves it into gprims in
/// UsdGenGraphDesc::geometries (a non-gprim target contributes its descendant
/// gprims, in namespace order) and map prims in UsdGenGraphDesc::maps.
struct UsdGenExpressionInputDesc
{
    TfToken       name;         // "guideCurves" for input:guideCurves
    SdfPathVector targets;
    SdfPathVector geometries;   // resolved gprim paths, in order
    SdfPathVector maps;         // resolved UsdGenMap prim paths, in order
};

struct UsdGenExpressionDesc
{
    SdfPath path;
    std::string source;
    std::vector<UsdGenExpressionOutputDesc> outputs;
    std::vector<UsdGenExpressionInputDesc> inputs;
};

struct UsdGenGeometryDesc : UsdGenGeometryData
{
    SdfPath path;   // a GeomSubset input: the subset
};

struct UsdGenExpressionBinding {
    // 'output' empty means the connection named the expression PRIM rather
    // than one of its outputs; UsdGenFindExpressionOutput resolves it.
    SdfPath expression; TfToken output = UsdGenTokens().result; TfToken nativeType = UsdGenTokens().float_; TfToken destination;
    expr::ValueShape destinationShape; expr::Domain domain = expr::Domain::Groom;
    VtValue literal;
};

/// The authored relationship slot which consumes a map (UsdGenMapBinding).
struct UsdGenMapBindingDesc
{
    SdfPath  map;
    TfToken  relationship;
};

struct UsdGenNodeDesc
{
    SdfPath                      path;         // the operator prim's scene path (identity + Kahn tie-break)
    TfToken                      type;         // the operator's schema type name
    TfToken                      mode;         // usdGen:mode, when the type has one
    bool                         enabled = true;
    int                          seed = 0;
    SdfPathVector                inputs;       // compiler-owned hierarchy dependency edges
    SdfPathVector                references;   // guide sets, clump centres, card roots
    SdfPathVector                curves;       // usdGen:guides / usdGen:curves / usdGen:frozen:curves
                                               //   -> indices into UsdGenGraphDesc::curveSets
    SdfPathVector                surfaces;     // the Description's usdGen:surface targets (ADR R15)
    // Legacy untyped map paths. Keep this populated by builders and accept
    // it from old direct clients; mapBindings is the canonical typed form.
    SdfPathVector                maps;
    std::vector<UsdGenMapBindingDesc> mapBindings;
    std::vector<UsdGenParamValue> params;      // EVERY mapped locator of this prim (S14 pull-all)
    std::vector<UsdGenExpressionBinding> expressionBindings;
    std::vector<UsdGenRampDesc>   ramps;
    UsdGenOpId                   opId = UsdGenOpId::Invalid;   // engine-filled at intake
};

/// One authored C3 BasisCurves input: a UsdGenGuideSet child, a freeze, an
/// import, a sim cache. The only way curve data reaches the engine (R23).
struct UsdGenCurveSetDesc : UsdGenCurveSetData
{
    SdfPath path;
};

struct UsdGenSurfaceDesc : UsdGenSurfaceData
{
    SdfPath path;   // the bound target: a Mesh or a GeomSubset
};

/// The Mesh prim a surface's geometry and primvars live on: the target
/// itself, or a GeomSubset's parent (a subset is always a Mesh's child).
inline SdfPath
UsdGenSurfaceMeshPath(UsdGenSurfaceDesc const &surface)
{
    return surface.isSubset ? surface.path.GetParentPath() : surface.path;
}

struct UsdGenMapDesc : UsdGenMapData
{
    SdfPath path;
    SdfPath paintSurface;   // UsdGenPaintMap: the surface carrying the primvar
};

/// usdGen:preview:* on the description (UsdGenPreviewSpec). 'source' is the
/// first authored target, unresolved: a UsdGenExpression prim, a
/// UsdGenPtexMap prim (the builders add it to 'maps') or an operator
/// attribute path.
struct UsdGenPreviewDesc
{
    SdfPath  source;                    // empty == no preview
    TfToken  colorMap = UsdGenTokens().heat;          // heat | viridis | gray | ids | rgb
    GfVec2f  range{0.0f, 1.0f};
    TfToken  evaluation = UsdGenTokens().primitive;   // for an expression prim source
    TfToken  shading = UsdGenTokens().lit;            // lit | flat

    bool Active() const { return !source.IsEmpty(); }
};

/// The engine's look block with the schema fallbacks the host fills
/// (bakeMode). publishPaths.cpp.
UsdGenLookDesc UsdGenSchemaLook();

struct UsdGenGraphDesc
{
    SdfPath                        description;   // the UsdGenDescription prim
    SdfPath                        terminal;      // hierarchy-derived final operator
    std::vector<UsdGenNodeDesc>    nodes;         // composed execution order from the builder
    std::vector<UsdGenCurveSetDesc> curveSets;    // every C3 BasisCurves the graph names (R23)
    std::vector<UsdGenSurfaceDesc> surfaces;
    std::vector<UsdGenMapDesc>     maps;
    std::vector<UsdGenExpressionDesc> expressions;
    std::vector<UsdGenGeometryDesc> geometries;   // every gprim an expression input names
    UsdGenLookDesc                 look = UsdGenSchemaLook();
    UsdGenPreviewDesc              preview;
    GfMatrix4d                     xformMatrix{1.0}; // description world matrix (post-flattening, S4)
    TfToken                        purpose;        // inherited by hand to every tile (C2)
    TfToken                        visibility;     // inherited by hand to every tile (C2)
    SdfPath                        materialPath;   // the description's bound Material (C2)
    float    defaultWidth = 0.01f;                 // usdGen:width:default
    int      tileTarget = 64;                      // uniform int usdGen:tileTarget
    TfToken  curveBasis = UsdGenTokens().bspline;      // usdGen:curve:basis (C2)
    double   time = 0.0;
    double   timeCodesPerSecond = 24.0; // USD default; expressions expose seconds
    UsdGenBackendId executionBackend = kUsdGenBackendCpu;   // UsdGenFindBackend at lowering
    std::vector<std::string> validationErrors;
};

}  // namespace usdGen

#endif  // USDGENUSD_GRAPH_DESC_H
