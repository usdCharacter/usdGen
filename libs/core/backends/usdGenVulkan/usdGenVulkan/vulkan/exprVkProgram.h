// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan expression programs: host-side compile, pack and split.
//
// This is the Vulkan counterpart of cudaParameters.cpp's Compile: it resolves
// each UsdGenExpressionBinding against UsdGenGraphDesc::expressions, decodes
// the $value literal, compiles the SeExpr source through expr::Frontend, and
// refuses sampler programs, exactly as the CUDA lane does. It then packs the
// IR into the exprVkEvaluate.comp program-buffer encoding and splits programs
// at transcendental instructions (which Vulkan cannot execute in f64; see
// exprVkEvaluate.comp): exact prefixes run on device, each transcendental
// step runs here via expressions/exprMath.h and <cmath> — the same calls the
// CPU reference lane makes — and the results are re-injected through field
// table slots >= 33.
//
// The combination is bit-identical with the CPU lane for every program:
// exact prefixes are IEEE-754 correctly rounded on both, and host steps run
// identical code on bit-identical inputs.
#ifndef USDGEN_VULKAN_EXPR_VK_PROGRAM_H
#define USDGEN_VULKAN_EXPR_VK_PROGRAM_H

#include "usdGen/expressions/ir.h"
#include "usdGen/graphDesc.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace usdGen::vulkan {

// Device encoding shared with exprVkEvaluate.comp. Every numeric below is
// static_asserted against the C++ headers in exprVkProgram.cpp.
struct ExprVkEncoding {
    static constexpr unsigned kHeaderBytes = 96;
    static constexpr unsigned kInstructionBytes = 24;
    static constexpr unsigned kTableEntryBytes = 16;
    static constexpr unsigned kBaseVariables = 33;
    // First field-table slot for host-step injections (one per
    // transcendental step, in program order).
    static constexpr unsigned kFirstInjectionSlot = 33;
    // Synthetic outType (not an expr::ScalarType): bit-preserving f64
    // register dump without Store validation, for intermediate segments.
    static constexpr unsigned kRaw64OutType = 9;
};

enum class ExprVkCompileStatus { Ok, InvalidArgument, CompileError, UnsupportedType };

struct ExprVkCompiledBinding {
    UsdGenExpressionBinding binding;
    expr::IRProgram ir;
    std::vector<double> literal;
};

// Mirrors CudaParameterProgram::Compile for one binding (empty/duplicate
// detection of destinations is the node-level call below): missing
// expression/output, domain, shape, native type/shape match, exact finite
// literal, SeExpr compilation, and the sampler refusal. allowValue/allowTime
// mirror the UsdGenExprOp capture refusals ($value/$frame/$time); parameter
// bindings pass true/true.
ExprVkCompileStatus ExprVkCompileBinding(
    UsdGenGraphDesc const& graph, UsdGenExpressionBinding const& binding,
    bool allowValue, bool allowTime, ExprVkCompiledBinding* out,
    std::vector<std::string>* diagnostics = nullptr);

// Compiles every binding on node (duplicate canonical destinations refuse).
ExprVkCompileStatus ExprVkCompileNodeBindings(
    UsdGenGraphDesc const& graph, UsdGenNodeDesc const& node,
    bool allowValue, bool allowTime,
    std::vector<ExprVkCompiledBinding>* out,
    std::vector<std::string>* diagnostics = nullptr);

// True when the instruction needs the host (a transcendental IROp, or a Call
// to bias/contrast/gaussstep/remap/angle/rotate/up/midhsi, which reach one;
// remap/midhsi are split conservatively since the interp argument is runtime).
bool ExprVkIsHostInstruction(expr::IRInstruction const& instruction);
// Positions of the host-evaluated instructions, in program order.
std::vector<size_t> ExprVkHostSteps(expr::IRProgram const& program);

// Evaluates one host step over count elements. inputs[k] points at count
// doubles holding the step's k-th device-read input register (1 for a unary
// IROp, 2 for a binary one, the whole consecutive argument block for a Call).
// Uses expr::ExprCall and <cmath> exactly as the CPU lane does. Returns false
// only for a non-host instruction (a caller bug).
bool ExprVkEvaluateHostStep(expr::IRInstruction const& instruction,
                            double const* const* inputs, size_t count,
                            double* outputs);

// One field-table slot's device descriptor (16 bytes: u64 offsetDoubles,
// u32 count, u32 domain|components<<8; offset UINT64_MAX = unmaterialized).
struct ExprVkFieldSlot {
    uint64_t offsetDoubles = UINT64_MAX;
    uint32_t count = 0;
    expr::Domain domain = expr::Domain::None;
    unsigned components = 0;
};

// Packed field layout shared by the context builder and the evaluator.
struct ExprVkFieldLayout {
    std::vector<uint64_t> offsets; // doubles; UINT64_MAX = unmaterialized
    std::vector<uint32_t> counts;
    std::vector<expr::Domain> domains;
    std::vector<unsigned> components;
    uint64_t totalDoubles = 0;
    unsigned varCount = ExprVkEncoding::kBaseVariables;
};

// Lays out one domain-specialized evaluation: the CUDA BuildImpl variable
// set (groom carries only the literal), one $value literal region (groom,
// count 1, literalComponents), and injectionCount host-step regions (eval
// domain, count n, 1 component). n is 1/curves/points by domain.
ExprVkFieldLayout ExprVkLayoutFields(expr::Domain domain, size_t n,
                                     unsigned literalComponents,
                                     unsigned injectionCount, bool hasRest,
                                     bool hasWidths, bool hasRootUV,
                                     bool hasRootN, bool hasRootT,
                                     bool hasRootB, bool hasRootPrim);

// Segment outputs: cooked (final) or raw (intermediate register dump).
struct ExprVkSegmentOutputs {
    bool raw = false;
    // Cooked: resolved output registers (size regCount, 1..4) and type.
    // Raw: consecutive registers [rawBase, rawBase + regCount), 1..256.
    uint16_t regs[4] = {};
    unsigned regCount = 0;
    expr::ScalarType type = expr::ScalarType::Float32;
    uint16_t rawBase = 0;
};

struct ExprVkPackOptions {
    expr::Domain domain = expr::Domain::Groom;
    double frame = 0.0;
    double time = 0.0;
    int32_t seed = 0;
    uint32_t descId = 0;
    uint32_t count = 0;
    uint32_t primitiveCount = 0;
};

// Packs one program-buffer image: 96-byte header, instructions
// [0, endExclusive) with rewrite applied (original index -> injection slot,
// emitted as LoadVariable), the field table, and the noise tables uploaded
// from expressions/exprNoiseTables.h. Returns false (with a diagnostic)
// when any size overflows 32 bits.
bool ExprVkPackProgram(expr::IRProgram const& program, size_t endExclusive,
                       std::map<size_t, unsigned> const& rewriteSlot,
                       ExprVkSegmentOutputs const& outputs,
                       ExprVkPackOptions const& options,
                       std::vector<ExprVkFieldSlot> const& table,
                       std::vector<uint32_t>* words,
                       std::string* diagnostic = nullptr);

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_PROGRAM_H
