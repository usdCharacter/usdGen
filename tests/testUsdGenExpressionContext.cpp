#include "usdGenExpr/expressions/context.h"
#include <cstdio>
#include <cstring>

using namespace usdGen::expr;
static int failures = 0;
static void Check(bool v, char const *s) { if (!v) { ++failures; std::printf("FAIL: %s\n",s); } else std::printf("ok: %s\n",s); }

int main()
{
    auto const &r = Registry::Get();
    for (Domain d : {Domain::Groom, Domain::Primitive, Domain::Point}) {
        for (char const *name : {"$value", "$frame", "$time", "$index", "$count", "$seed", "$descId"})
            Check(r.Validate(name, d), "core variable available in every domain");
    }
    Check(r.Validate("$pointIndex", Domain::Point), "point variable available at point domain");
    Check(!r.Validate("$pointIndex", Domain::Groom), "point variable rejected at groom domain");
    Check(!r.Validate("$doesNotExist", Domain::Point), "unknown variable rejected");

    uint32_t a = DescriptionId("/Char/Groom/hair");
    Check(a == DescriptionId("/Char/Groom/hair"), "description id is stable");
    Check(a != DescriptionId("/Char/Groom/other"), "description id changes with path");

    float vec[3] = {1,2,3};
    ValueView v{vec, 1, 3, 1, 1, 0, ScalarType::Float32};
    ValueShape good{ScalarType::Float32, 1, 3, 1, 1};
    ValueShape bad{ScalarType::Float32, 1, 2, 1, 1};
    Check(ValidateValueShape(v, good), "vector shape validates exactly");
    Check(!ValidateValueShape(v, bad), "vector component mismatch rejected");
    float matrix[16] = {};
    ValueView m{matrix, 1, 1, 4, 4, 0, ScalarType::Float32};
    Check(ValidateValueShape(m, {ScalarType::Float32,1,1,4,4}), "matrix row-major shape retained");
    Check(!ValidateValueShape(m, {ScalarType::Float32,2,1,4,4}), "array shape mismatch rejected");
    std::printf("testUsdGenExpressionContext: %s\n", failures ? "FAILED" : "PASS");
    return failures ? 1 : 0;
}
