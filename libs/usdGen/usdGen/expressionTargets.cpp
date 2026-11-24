#include "usdGen/expressionTargets.h"

#include "usdGen/registries/opRegistry.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

bool UsdGenValidateExpressionTargets(UsdGenNodeDesc const &node,
                                     std::vector<std::string> *errors)
{
    if (node.expressionBindings.empty()) return true;
    // Compile phase: opId when intake resolved it for this type, else a
    // pointer-hash probe (never interns).
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpId id = node.opId;
    if (id == UsdGenOpId::Invalid || UsdGenIndex(id) >= ops.Count() ||
        ops.Record(id).Name() != node.type)
        id = ops.Find(node.type);
    if (id == UsdGenOpId::Invalid) {
        if (errors)
            errors->push_back("operator " + node.type.GetString() + " at " +
                              node.path.GetString() +
                              " does not accept connected (expression) parameters");
        return false;
    }
    return UsdGenValidateExpressionBindings(ops.Record(id), node, errors);
}

} // namespace usdGen
