#ifndef USDGEN_TEST_SEEXPR_ORACLE_H
#define USDGEN_TEST_SEEXPR_ORACLE_H

#include <cstdint>
#include <string>
#include <vector>

namespace usdGenTest {

// A test binding is deliberately a value vector, rather than a production
// variable handle.  Its length is the SeExpr FP dimension (one is scalar).
struct SeExprOracleBinding {
    std::string name;             // accepts either "P" or "$P"
    std::vector<double> values;
};

struct SeExprOracleResult {
    bool ok = false;
    uint32_t outputComponents = 0; // the parsed expression's actual dimension
    uint32_t valueComponents = 0;   // number of values returned (requested width)
    std::vector<double> values;
    std::string diagnostic;
};

// Evaluate with the vendored SeExpr2 interpreter.  This is intentionally a
// test oracle only: it has no fallback evaluator and rejects parse/type
// errors instead of returning a fabricated value.
SeExprOracleResult EvaluateSeExpr(
    const std::string& source,
    uint32_t requestedComponents,
    const std::vector<SeExprOracleBinding>& bindings);

} // namespace usdGenTest

#endif
