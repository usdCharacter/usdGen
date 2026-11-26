// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Viewport value preview evaluation (usdGen:preview:*, 07 §7.3): the values
// a description previews, per published strand or CV. Hosts colour them.
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
// warning, and those strands carry no value.
#pragma once

#include "usdGenExpr/cpuParameters.h"
#include "usdGenCurves/types/curveSet.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/publishDesc.h"

#include <cstdint>
#include <string>
#include <vector>

namespace usdGen {

class UsdGenGraph;
struct UsdGenDiagnostics;

class UsdGenValuePreview
{
public:
    /// Resolves desc.preview against the compiled graph and evaluates it over
    /// `terminal`: one value per curve (level 1) or per CV (level 2). Inactive
    /// when nothing is previewed. A new warning goes to `diag`; a repeat of
    /// the last one does not. `out` reads this object's buffers.
    void Build(UsdGenGraphSpec const &desc, UsdGenGraph const &graph,
               UsdGenCurveSetView const &terminal, double frame,
               UsdGenPreviewValues *out, UsdGenDiagnostics *diag);

private:
    struct Values;
    bool FromExpression(UsdGenGraphSpec const &desc, UsdGenCurveSetView const &terminal,
                        double frame, Values *values, std::string *warning);
    bool FromMap(UsdGenGraphSpec const &desc, UsdGenCurveSetView const &terminal,
                 double frame, Values *values, std::string *warning);
    bool FromAttribute(UsdGenGraphSpec const &desc, UsdGenGraph const &graph,
                       Values *values, std::string *warning);
    bool Evaluate(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                  UsdGenCurveSetView const &terminal, double frame, std::string *warning);

    UsdGenCpuParameters _parameters;   // expression and map previews
    UsdGenGraphSpec _mapDesc;          // the private ptex("map") expression's world
    std::vector<double> _scratch;
    std::vector<double> _values;       // Build's output, per published element
    std::vector<uint8_t> _present;
    std::string _lastWarning;
};

} // namespace usdGen
