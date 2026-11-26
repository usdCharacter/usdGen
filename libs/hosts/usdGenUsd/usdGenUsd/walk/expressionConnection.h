// usdGen imaging — shared resolution of an authored expression connection.
//
// A connectable usdGen parameter may name either the expression output
//     float usdGen:mask.connect = </…/Expressions/styleMask.outputs:result>
// or the expression PRIM
//     float usdGen:mask.connect = </…/Expressions/styleMask>
// An attribute -> prim connection is valid USD, so both builders resolve it
// the same way and the descriptor carries the same binding either way:
// outputs:result when the expression declares it, otherwise its single
// outputs:* attribute.  Anything else yields an empty output token and the
// compiler owns the fail-closed diagnostic naming the prim
// (UsdGenFindExpressionOutput, usdGen/graphDesc.h).
#ifndef USDGEN_IMAGING_EXPRESSION_CONNECTION_H
#define USDGEN_IMAGING_EXPRESSION_CONNECTION_H

#include "pxr/base/tf/token.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/usd/usd/attribute.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"

#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// The `outputs:` suffix `connection` names, resolved against the stage
/// `consumer` lives on.  Empty when the connection is malformed, when the
/// named property is not an output, or when a prim-path connection is
/// ambiguous — the compiler refuses the descriptor in every such case.
inline TfToken
UsdGenResolveExpressionConnectionOutput(UsdPrim const &consumer,
                                        SdfPath const &connection)
{
    if (connection.IsPropertyPath()) {
        std::string const property = connection.GetNameToken().GetString();
        return property.rfind("outputs:", 0) == 0
            ? TfToken(property.substr(8)) : TfToken();
    }
    if (!connection.IsPrimPath() || !consumer) return TfToken();
    UsdStageWeakPtr const stage = consumer.GetStage();
    if (!stage) return TfToken();
    UsdPrim const expression = stage->GetPrimAtPath(connection);
    if (!expression) return TfToken();
    if (expression.GetAttribute(TfToken("outputs:result")))
        return TfToken("result");
    TfToken only;
    for (UsdAttribute const &attribute : expression.GetAttributes()) {
        std::string const name = attribute.GetName().GetString();
        if (name.rfind("outputs:", 0) != 0) continue;
        if (!only.IsEmpty()) return TfToken();   // ambiguous: several outputs
        only = TfToken(name.substr(8));
    }
    return only;
}

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_EXPRESSION_CONNECTION_H
