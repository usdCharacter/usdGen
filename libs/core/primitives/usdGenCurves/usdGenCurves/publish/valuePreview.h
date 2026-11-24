// usdGen engine — viewport value preview (usdGen:preview:*, 07 §7.3).
//
// A description whose usdGen:preview:source targets something publishes that
// value as its displayColor instead of the look, so an artist can see what an
// expression, a Ptex map or an operator's parameter does to every strand.
// There are three kinds of source:
//
//   * a UsdGenExpression prim, evaluated here over the published (terminal)
//     strands in the usdGen:preview:evaluation domain;
//   * a UsdGenPtexMap prim, read at every strand root exactly as ptex() reads
//     it, through a private one-line `ptex("map")` expression;
//   * an operator attribute: the values that operator was cooked with (its
//     UsdGenCpuParameters over its INPUT strands, matched to the published
//     strands by curve id), or its authored value when nothing is connected.
//
// A preview never fails a cook. Whatever it cannot show is reported as a
// warning, and the strands read the "no value" colour instead.
#ifndef USDGEN_VALUE_PREVIEW_H
#define USDGEN_VALUE_PREVIEW_H

#include "usdGen/cpuParameters.h"
#include "usdGen/curveBuffer.h"
#include "usdGen/graphDesc.h"

#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"

#include <cstdint>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenGraph;

/// The preview colours of one cook, indexed like the terminal buffer.
struct UsdGenPreviewColors
{
    bool active = false;
    bool perCv = false;            // one colour per CV, else one per curve
    std::vector<GfVec3f> colors;   // linear
    uint64_t digest = 0;           // 0 exactly when inactive
};

/// Linear colour of one previewed value. The heat, viridis and gray maps and
/// the ids palette are defined in sRGB and converted, so a mid-range value
/// looks mid-range on screen; "rgb" shows the normalised components as a
/// linear colour, so an expression that computes a colour shows that colour.
GfVec3f UsdGenPreviewColor(TfToken const &colorMap, GfVec2f const &range,
                           double const *values, unsigned components);

/// The colour of a strand the source has no value for.
GfVec3f UsdGenPreviewMissingColor();

/// <description>/__usdGenRender/material_preview (lit) or
/// material_preview_flat: the synthetic material a previewing description's
/// tiles bind (served by UsdGenGroomSceneIndex).
SdfPath UsdGenPreviewMaterialPath(SdfPath const &description, TfToken const &shading);

class UsdGenValuePreview
{
public:
    /// Resolves desc.preview against the compiled graph and colours
    /// `terminal`. Returns an inactive result when nothing is previewed. A new
    /// warning is appended to `warnings`; a repeat of the last one is not.
    UsdGenPreviewColors Build(UsdGenGraphDesc const &desc, UsdGenGraph const &graph,
                              UsdGenCurveBuffer const &terminal, double frame,
                              std::vector<std::string> *warnings);

private:
    struct Values;
    bool FromExpression(UsdGenGraphDesc const &desc, UsdGenCurveBuffer const &terminal,
                        double frame, Values *values, std::string *warning);
    bool FromMap(UsdGenGraphDesc const &desc, UsdGenCurveBuffer const &terminal,
                 double frame, Values *values, std::string *warning);
    bool FromAttribute(UsdGenGraphDesc const &desc, UsdGenGraph const &graph,
                       Values *values, std::string *warning);
    bool Evaluate(UsdGenGraphDesc const &desc, UsdGenNodeDesc const &node,
                  UsdGenCurveBuffer const &terminal, double frame, std::string *warning);

    UsdGenCpuParameters _parameters;   // expression and map previews
    UsdGenGraphDesc _mapDesc;          // the private ptex("map") expression's world
    std::vector<double> _scratch;
    std::string _lastWarning;
};

} // namespace usdGen

#endif
