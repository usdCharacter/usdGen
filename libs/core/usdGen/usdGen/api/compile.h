// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Compiling a UsdGenGraphSpec, and the read side of the compiled graph a
// publish context hands out. The graph itself stays private (compile/graph.h).
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/span.h"
#include "usdGen/api/types.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

class UsdGenGraph;
/// Opaque here; the compiler's output.
using UsdGenCompiledGraph = UsdGenGraph;
class UsdGenElementSet;
class UsdGenExecutionPlanHandle;
class UsdGenOp;
class UsdGenOpRecord;
class UsdGenParamEvaluatorState;
struct UsdGenDiagnostics;
struct UsdGenGraphSpec;
struct UsdGenOpNode;
struct UsdGenParamView;

/// Extra named planes one node binds. Plane pointer tables are stack-local
/// in the chunk evaluator, so a fixed bound keeps Evaluate() allocation free.
constexpr uint32_t kUsdGenMaxExtraPlaneSlots = 16;

struct UsdGenCompileResult
{
    bool                      ok = false;
    UsdGenEpoch               structuralDigest {};  // graph-level digest (03 §3.2)
    UsdGenPoolVector<UsdGenNodeId> rebuilt;    // nodes whose UsdGenOp was recreated
    UsdGenPoolVector<UsdGenNodeId> reordered;  // nodes whose topological index moved
    UsdGenPoolVector<UsdGenNodeId> valueEdited; // kept nodes whose value-class params moved
    std::vector<std::string>  errors;        // cycle, terminal ambiguity, unknown op, ...
    std::vector<std::string>  warnings;      // non-fatal authoring diagnostics
    /// False only when a failed Recompile had already moved state out of
    /// *out (a broken invariant: every failure is preflighted). The caller
    /// must then discard the graph.
    bool                      graphIntact = true;
};

/// A backend's device plan for a spec (kUsdGenBackendReusesPlan).
using UsdGenDevicePlanCompiler = std::function<
    std::shared_ptr<const UsdGenExecutionPlanHandle>(
        UsdGenGraphSpec const&, UsdGenDiagnostics*)>;

/// Compile phase (allocates). ok=false leaves *out untouched; Recompile
/// rebuilds only the nodes whose digest moved.
USDGEN_CORE_API UsdGenCompileResult UsdGenCompile(UsdGenGraphSpec const &spec,
                                                  UsdGenCompiledGraph *out);
USDGEN_CORE_API UsdGenCompileResult UsdGenRecompile(UsdGenGraphSpec const &spec,
                                                    UsdGenCompiledGraph *out);
/// UsdGenCompile plus the device plan, compiled after the graph. Both
/// outputs stay untouched on failure.
USDGEN_CORE_API UsdGenCompileResult UsdGenCompileWithDevicePlan(
    UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out,
    UsdGenDevicePlanCompiler const &device,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan);
/// The same for an explicitly injected device provider.
USDGEN_CORE_API UsdGenCompileResult UsdGenCompileInjectedDevice(
    UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out,
    UsdGenDevicePlanCompiler const &device,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan);

/// One compiled node, read only. Pointers live as long as the graph's
/// current compilation.
struct UsdGenCompiledNodeView
{
    UsdGenNodeId id = kUsdGenInvalidNode;
    UsdGenOpId opId = UsdGenOpId::Invalid;
    UsdGenOpNode const *desc = nullptr;
    UsdGenOpRecord const *record = nullptr;
    UsdGenOp const *op = nullptr;
    UsdGenSpan<const UsdGenNodeId> inputs;     // every usdGen:input target
    UsdGenParamView const *paramView = nullptr;
    UsdGenParamEvaluatorState const *expressions = nullptr;
    UsdGenSurfaceId surface = 0;
    bool hasOutputPlanes = false;
};

USDGEN_CORE_API int UsdGenCompiledNodeCount(UsdGenCompiledGraph const &graph) noexcept;
USDGEN_CORE_API UsdGenNodeId UsdGenCompiledTerminal(UsdGenCompiledGraph const &graph) noexcept;
USDGEN_CORE_API UsdGenCompiledNodeView UsdGenCompiledNodeAt(UsdGenCompiledGraph const &graph,
                                                          UsdGenNodeId id);
USDGEN_CORE_API UsdGenElementSet const &UsdGenCompiledNodeOutput(
    UsdGenCompiledGraph const &graph, UsdGenNodeId id);
/// kUsdGenInvalidNode when no node has 'key'.
USDGEN_CORE_API UsdGenNodeId UsdGenCompiledNodeIdForKey(UsdGenCompiledGraph const &graph,
                                                        UsdGenNodeKey key);
/// The spec the graph was compiled from.
USDGEN_CORE_API UsdGenGraphSpec const &UsdGenCompiledSpec(
    UsdGenCompiledGraph const &graph) noexcept;

/// Host dirtying between cooks.
USDGEN_CORE_API void UsdGenCompiledDirtyGeometry(UsdGenCompiledGraph &graph, UsdGenNodeKey prim);
USDGEN_CORE_API void UsdGenCompiledDirtyInput(UsdGenCompiledGraph &graph, UsdGenNodeKey input,
                                              uint64_t generation = UINT64_MAX);
USDGEN_CORE_API void UsdGenCompiledDirtyMap(UsdGenCompiledGraph &graph, UsdGenNodeKey map,
                                            uint64_t generation = UINT64_MAX);

}  // namespace usdGen
