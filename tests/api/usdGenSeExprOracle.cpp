#include "usdGenSeExprOracle.h"

#include <algorithm>
#include <memory>
#include <sstream>

// Keep this translation unit on the same private namespace as the real
// frontend.  The build must define both aliases when compiling it:
// SeExpr2=UsdGenSeExprFrontend and
// SeExprInternal2=UsdGenSeExprFrontendInternal.
#include <Expression.h>

namespace usdGenTest {
namespace {

std::string CanonicalName(std::string name)
{
    if (!name.empty() && name.front() == '$') name.erase(name.begin());
    return name;
}

class BoundVariable final : public UsdGenSeExprFrontend::ExprVarRef {
public:
    explicit BoundVariable(const std::vector<double>& values)
        : ExprVarRef(UsdGenSeExprFrontend::ExprType().FP(
              static_cast<int>(values.size())).Varying()), _values(values) {}

    void eval(double* result) override {
        std::copy(_values.begin(), _values.end(), result);
    }
    void eval(const char** result) override { *result = nullptr; }

private:
    std::vector<double> _values;
};

class OracleExpression final : public UsdGenSeExprFrontend::Expression {
public:
    OracleExpression(const std::string& source, int requested,
                     const std::vector<SeExprOracleBinding>& bindings)
        : Expression(source, UsdGenSeExprFrontend::ExprType().FP(requested),
                     Expression::UseInterpreter), _bindings(bindings) {}

    UsdGenSeExprFrontend::ExprVarRef* resolveVar(
        const std::string& name) const override {
        const std::string wanted = CanonicalName(name);
        for (const auto& binding : _bindings) {
            if (CanonicalName(binding.name) != wanted) continue;
            if (binding.values.empty()) return nullptr;
            _variables.emplace_back(new BoundVariable(binding.values));
            return _variables.back().get();
        }
        return nullptr;
    }

private:
    const std::vector<SeExprOracleBinding>& _bindings;
    mutable std::vector<std::unique_ptr<BoundVariable>> _variables;
};

std::string Errors(const OracleExpression& expression)
{
    std::ostringstream out;
    if (!expression.parseError().empty()) out << expression.parseError();
    for (const auto& error : expression.getErrors()) {
        if (out.tellp() > 0) out << "; ";
        out << error.error << " [" << error.startPos << "," << error.endPos
            << "]";
    }
    return out.str();
}

} // namespace

SeExprOracleResult EvaluateSeExpr(
    const std::string& source, uint32_t requestedComponents,
    const std::vector<SeExprOracleBinding>& bindings)
{
    SeExprOracleResult result;
    if (requestedComponents == 0 || requestedComponents > 4) {
        result.diagnostic = "requested SeExpr dimension must be in [1,4]";
        return result;
    }

    OracleExpression expression(source, static_cast<int>(requestedComponents),
                                bindings);
    if (!expression.isValid()) {
        result.diagnostic = Errors(expression);
        if (result.diagnostic.empty()) result.diagnostic = "SeExpr validation failed";
        return result;
    }

    const auto& type = expression.returnType();
    if (type.type() != UsdGenSeExprFrontend::ExprType::tFP || type.dim() < 1 ||
        type.dim() > 4) {
        result.diagnostic = "SeExpr result is not a supported FP scalar/vector";
        return result;
    }
    result.outputComponents = static_cast<uint32_t>(type.dim());
    const double* values = expression.evalFP();
    if (!values) {
        result.diagnostic = "SeExpr interpreter returned no result";
        result.outputComponents = 0;
        return result;
    }
    // Expression::evalFP() returns the desired width: upstream inserts a
    // Promote op when a scalar AST is requested as a vector.  Keep the AST
    // dimension above for shape assertions, but copy the requested width.
    result.valueComponents = requestedComponents;
    result.values.assign(values, values + result.valueComponents);
    result.ok = true;
    return result;
}

} // namespace usdGenTest
