// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// api/compile.h over the private compiler and graph.
#include "usdGen/api/compile.h"
#include "usdGen/compile/compiler.h"
#include "usdGen/compile/graph.h"

namespace usdGen {

UsdGenCompileResult UsdGenCompile(UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out)
{
    return UsdGenCompiler().Compile(spec, out);
}

UsdGenCompileResult UsdGenRecompile(UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out)
{
    return UsdGenCompiler().Recompile(spec, out);
}

UsdGenCompileResult UsdGenCompileWithDevicePlan(
    UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out,
    UsdGenDevicePlanCompiler const &device,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan)
{
    return UsdGenCompiler().CompileWithDevicePlan(spec, out, device, plan);
}

UsdGenCompileResult UsdGenCompileInjectedDevice(
    UsdGenGraphSpec const &spec, UsdGenCompiledGraph *out,
    UsdGenDevicePlanCompiler const &device,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan)
{
    return UsdGenCompiler().CompileInjectedDevice(spec, out, device, plan);
}

int UsdGenCompiledNodeCount(UsdGenCompiledGraph const &graph) noexcept
{
    return graph.NodeCount();
}

UsdGenNodeId UsdGenCompiledTerminal(UsdGenCompiledGraph const &graph) noexcept
{
    return graph.TerminalNodeId();
}

UsdGenCompiledNodeView UsdGenCompiledNodeAt(UsdGenCompiledGraph const &graph, UsdGenNodeId id)
{
    UsdGenCompiledNode const &node = graph.Node(id);
    UsdGenCompiledNodeView view;
    view.id = node.id;
    view.opId = node.opId;
    view.desc = node.desc;
    view.record = node.record;
    view.op = node.op.get();
    view.inputs = UsdGenSpan<const UsdGenNodeId>(node.inputs.data(), node.inputs.size());
    view.paramView = &node.paramView;
    view.expressions = node.expressions.get();
    view.surface = node.surface;
    view.hasOutputPlanes = !node.outputPrimvars.empty();
    return view;
}

UsdGenElementSet const &UsdGenCompiledNodeOutput(UsdGenCompiledGraph const &graph,
                                                 UsdGenNodeId id)
{
    return graph.NodeOutput(id);
}

UsdGenNodeId UsdGenCompiledNodeIdForKey(UsdGenCompiledGraph const &graph, UsdGenNodeKey key)
{
    return graph.NodeIdForKey(key);
}

UsdGenGraphSpec const &UsdGenCompiledSpec(UsdGenCompiledGraph const &graph) noexcept
{
    return graph.Spec();
}

void UsdGenCompiledDirtyGeometry(UsdGenCompiledGraph &graph, UsdGenNodeKey prim)
{
    graph.DirtyGeometry(prim);
}

void UsdGenCompiledDirtyInput(UsdGenCompiledGraph &graph, UsdGenNodeKey input,
                              uint64_t generation)
{
    graph.DirtyInput(input, generation);
}

void UsdGenCompiledDirtyMap(UsdGenCompiledGraph &graph, UsdGenNodeKey map, uint64_t generation)
{
    graph.DirtyMap(map, generation);
}

}  // namespace usdGen
