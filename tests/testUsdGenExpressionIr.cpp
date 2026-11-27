#include "usdGenExpr/expressions/frontend.h"
#include <cstdio>
int main() {
    auto r = usdGen::expr::Frontend::Compile("$value * (1 - 0.95 * $t)", {usdGen::expr::Domain::Point, usdGen::expr::ScalarType::Float32, 1});
    if (!r.ok || r.program.IR().instructions.empty()) { std::printf("FAIL: IR lowering\n"); return 1; }
    bool loads=false, mul=false; for (auto const &i:r.program.IR().instructions) { loads |= i.op==usdGen::expr::IROp::LoadVariable; mul |= i.op==usdGen::expr::IROp::Mul; }
    if (!loads || !mul) { std::printf("FAIL: expected load/mul IR\n"); return 1; }
    auto vec = usdGen::expr::Frontend::Compile("[1,2,3]", {usdGen::expr::Domain::Point,usdGen::expr::ScalarType::Float32,3});
    if (!vec.ok || vec.program.IR().outputCount != 3 || vec.program.IR().valueComponents != 3) { std::printf("FAIL: vector outputs\n"); return 1; }
    auto pv = usdGen::expr::Frontend::Compile("$P * $t", {usdGen::expr::Domain::Point,usdGen::expr::ScalarType::Float32,3});
    if (!pv.ok) { std::printf("FAIL: vector variable arithmetic\n"); return 1; }
    auto fn = usdGen::expr::Frontend::Compile("clamp(abs($value), 0, 1)", {usdGen::expr::Domain::Point,usdGen::expr::ScalarType::Float32,1});
    if (!fn.ok) { std::printf("FAIL: supported function lowering\n"); return 1; }
    auto vf = usdGen::expr::Frontend::Compile("$value * 2", {usdGen::expr::Domain::Point,usdGen::expr::ScalarType::Float32,3});
    if (!vf.ok || vf.program.IR().outputCount != 3) { std::printf("FAIL: vector value broadcast\n"); return 1; }
    auto vt = usdGen::expr::Frontend::Compile("$frame > 0 ? [1,2,3] : [$value[0],$value[1],$value[2]]", {usdGen::expr::Domain::Point,usdGen::expr::ScalarType::Float32,3});
    if (!vt.ok || !vt.program.IR().hasLazyBranches) { std::printf("FAIL: vector ternary components\n"); return 1; }
    auto bad = usdGen::expr::Frontend::Compile("printf(\"%v\", $value)", {usdGen::expr::Domain::Point, usdGen::expr::ScalarType::Float32, 1});
    if (bad.ok) { std::printf("FAIL: unsafe function accepted\n"); return 1; }
    std::printf("testUsdGenExpressionIr: PASS\n"); return 0;
}
