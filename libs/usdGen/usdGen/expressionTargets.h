// usdGen engine — which operator parameters may be driven by an expression.
//
// ONE rule for every lane: forwards to UsdGenValidateExpressionBindings over
// the op record's parameter decls (registries/opRegistry.h). The compiler,
// the CUDA admission and the Vulkan plan all call it.
//
// The rules are the shape/type/domain admission only. Whether the *value* is
// in range is a runtime question each kernel answers for itself.
#ifndef USDGEN_EXPRESSION_TARGETS_H
#define USDGEN_EXPRESSION_TARGETS_H

#include "usdGen/graphDesc.h"

#include "pxr/base/tf/token.h"

#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Validates every binding on `node` against its operator's allowlist.
/// Appends one message per rejection to `errors` (never null-checked away)
/// and returns false if anything was rejected.
bool UsdGenValidateExpressionTargets(UsdGenNodeDesc const &node,
                                     std::vector<std::string> *errors);

} // namespace usdGen

#endif
