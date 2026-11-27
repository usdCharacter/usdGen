// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Compile checks and dependencies of connected parameters (moved from the
// engine's compiler, step 6.5): the evaluator's UsdGenParamEvaluatorDef hooks.
#include "usdGenExpr/exprValidate.h"
#include "usdGen/api/token.h"
#include "usdGenExpr/exprInput.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGen/api/pool.h"

#include <functional>
#include <set>
#include <string>

namespace usdGen {

namespace {
/// Pool-backed like the compiler's: a recompile's checks allocate no heap.
template <class T>
using PSet = std::set<T, std::less<T>, UsdGenPoolAllocator<T>>;
}  // namespace

bool UsdGenValidateExpressionInputs(UsdGenGraphSpec const &desc, std::vector<std::string> *errors)
{
    bool ok = true;
    PSet<UsdGenNodeKey> expressionPaths;
    PSet<UsdGenNodeKey> geometryPaths, mapPaths;
    for (UsdGenInputRecord const &g : UsdGenGeometryInputs(desc)) {
        if (!g.key.IsValid() || !geometryPaths.insert(g.key).second) {
            errors->push_back("invalid or duplicate expression geometry " + desc.Label(g.key));
            ok = false;
        }
    }
    for (UsdGenInputRecord const &m : UsdGenMapInputs(desc)) mapPaths.insert(m.key);
    for (UsdGenInputRecord const &record : UsdGenExpressionInputs(desc)) {
        UsdGenExpressionInputView const e = UsdGenExpressionInputView::Of(record);
        if (!e.key.IsValid() || e.source.empty() || !expressionPaths.insert(e.key).second) {
            errors->push_back("expression has missing path or source"); ok = false;
        }
        PSet<UsdGenToken> inputNames;
        for (auto const &input : e.inputs) {
            if (input.name.IsEmpty() || !inputNames.insert(input.name).second) {
                errors->push_back("expression " + desc.Label(e.key) +
                    " has an empty or duplicate input:" + input.name.GetString());
                ok = false;
            }
            for (UsdGenNodeKey const g : input.geometries)
                if (!geometryPaths.count(g)) {
                    errors->push_back("expression " + desc.Label(e.key) + " input:" +
                        input.name.GetString() + " names unresolved geometry " + desc.Label(g));
                    ok = false;
                }
            for (UsdGenNodeKey const m : input.maps)
                if (!mapPaths.count(m)) {
                    errors->push_back("expression " + desc.Label(e.key) + " input:" +
                        input.name.GetString() + " names unresolved map " + desc.Label(m));
                    ok = false;
                }
        }
        PSet<UsdGenToken> seen;
        for (auto const &o : e.outputs)
            if (o.name.IsEmpty() || o.nativeType.IsEmpty() || !seen.insert(o.name).second) { errors->push_back("expression " + desc.Label(e.key) + " has invalid or duplicate output"); ok = false; }
    }
    return ok;
}

bool UsdGenValidateExpressionBindingSources(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                                            std::vector<std::string> *errors)
{
    bool ok = true;
    PSet<UsdGenToken> destinations;
    for (auto const &b : node.expressionBindings) {
        UsdGenInputRecord const *ei = UsdGenFindExpressionInput(desc, b.expression);
        if (!ei) { errors->push_back("expression binding references missing expression " + desc.Label(b.expression)); ok = false; continue; }
        // An empty output is a connection to the expression PRIM; it
        // resolves to outputs:result, or to a single declared output.
        UsdGenExpressionOutputDesc const *oi =
            UsdGenFindExpressionOutput(UsdGenExpressionInputView::Of(*ei), b.output);
        if (!oi) {
            errors->push_back(b.output.IsEmpty()
                ? "expression binding connects to prim " + desc.Label(b.expression) +
                      " which declares no outputs:result and no single outputs:* attribute"
                : "expression binding references missing or ambiguous output " + b.output.GetString());
            ok = false; continue;
        }
        if (!destinations.insert(b.destination).second) { errors->push_back("duplicate expression consumer " + desc.Label(node.key) + "." + b.destination.GetString()); ok = false; }
        if (!(b.domain == expr::Domain::Groom || b.domain == expr::Domain::Primitive || b.domain == expr::Domain::Point)) { errors->push_back("invalid expression evaluation domain"); ok = false; }
        if (b.destination.IsEmpty() || b.nativeType.IsEmpty()) { errors->push_back("expression binding has empty destination or native type"); ok = false; }
        // Hydra transports native property names; direct descriptor
        // clients may use operator-local names. Strip only our namespace.
        std::string destination = b.destination.GetString();
        if (destination.compare(0, 7, "usdGen:") == 0)
            destination.erase(0, 7);
        if ((destination == "enabled" || destination == "seed" ||
             destination == "segments" || destination == "cvCount") &&
            b.domain != expr::Domain::Groom) {
            errors->push_back("topology/control expression must evaluate at groom domain"); ok = false;
        }
        auto const &a = oi->shape; auto const &d = b.destinationShape;
        if (b.nativeType != oi->nativeType || a.scalar == expr::ScalarType::Invalid || d.scalar == expr::ScalarType::Invalid || a.scalar != d.scalar || a.elementCount != d.elementCount || a.components != d.components || a.rows != d.rows || a.columns != d.columns || a.isArray != d.isArray) {
            errors->push_back("expression output type/shape does not match destination " + b.destination.GetString()); ok = false;
        }
    }
    return ok;
}

void UsdGenExpressionDependencies(UsdGenGraphSpec const &desc, UsdGenNodeKey source, void *context,
                                  void (*add)(void *context, UsdGenNodeKey))
{
    for (UsdGenInputRecord const &record : UsdGenExpressionInputs(desc)) {
        if (record.key != source) continue;
        UsdGenExpressionInputView const expression = UsdGenExpressionInputView::Of(record);
        for (UsdGenExpressionSlot const &input : expression.inputs) {
            for (UsdGenNodeKey const key : input.targets) add(context, key);
            for (UsdGenNodeKey const key : input.geometries) add(context, key);
            for (UsdGenNodeKey const key : input.maps) add(context, key);
        }
    }
}

}  // namespace usdGen
