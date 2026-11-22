// usdGen engine — compiler (03-execution-engine.md §3).
//
// Compile() walks UsdGenGraphDesc::nodes and: edges from the composed
// hierarchy order; Kahn sort with namespace tie-break; cycle detection
// (compile error naming the offending pair); dense node ids in topological
// order; reference-lane ordering (§1.5); OutputPrimvars
// slot binding (§1.2); Merkle structural digests (§3.3); tile arithmetic
// (R21); dirty routing table rebuild data (§5.1).
#ifndef USDGEN_COMPILER_H
#define USDGEN_COMPILER_H

#include "usdGen/graph.h"
#include "usdGen/graphDesc.h"
#include "usdGen/types.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenExecutionPlanHandle;
struct UsdGenDiagnostics;

struct UsdGenCompileResult
{
    bool                      ok = false;
    UsdGenEpoch               structuralDigest {};  // graph-level digest (03 §3.2)
    std::vector<UsdGenNodeId> rebuilt;       // nodes whose UsdGenOp was recreated
    std::vector<UsdGenNodeId> reordered;     // nodes whose topological index moved
    std::vector<std::string>  errors;        // cycle, terminal ambiguity, unknown op, ...
    std::vector<std::string>  warnings;      // non-fatal authoring diagnostics
    /// False only when a failed Recompile had already moved state out of
    /// *out (a broken invariant: every failure is preflighted). The caller
    /// must then discard the graph.
    bool                      graphIntact = true;
};

class UsdGenCompiler
{
public:
    /// Fills *out (moves the compiled state). ok=false leaves *out untouched.
    /// Recompile is transactional too: validation runs before any node moves.
    UsdGenCompileResult Compile(UsdGenGraphDesc const &desc, UsdGenGraph *out);

    /// Private integration boundary for an explicitly injected device provider.
    /// Runs the normal structural/routing compilation, but obtains the native
    /// plan from this exact descriptor without changing global availability.
    /// No source capture or CPU geometry evaluation occurs. Both outputs stay
    /// untouched on failure. Currently restricted to the Vulkan provider route.
    using DevicePlanCompiler = std::function<
        std::shared_ptr<const UsdGenExecutionPlanHandle>(
            UsdGenGraphDesc const&, UsdGenDiagnostics*)>;
    UsdGenCompileResult CompileInjectedDevice(UsdGenGraphDesc const&,
        UsdGenGraph*, DevicePlanCompiler const&,
        std::shared_ptr<const UsdGenExecutionPlanHandle>* plan);

    /// Incremental: rebuild only the sub-graph whose Merkle digest moved
    /// (03 §3.5). One node appended to a 200-node groom rebuilds exactly one
    /// node (gate E-6).
    UsdGenCompileResult Recompile(UsdGenGraphDesc const &newDesc, UsdGenGraph *out);

private:
    friend class UsdGenGraph;  // fills the compiled structure in place
    /// Shared pipeline for Compile() / Recompile(); `reuse` (non-null on the
    /// incremental path) supplies captures/buffers for digest-stable nodes.
    static void _Build(UsdGenGraphDesc const &desc, UsdGenGraph *out,
                       UsdGenGraph const *reuse, UsdGenCompileResult &result);
};

}  // namespace usdGen

#endif  // USDGEN_COMPILER_H
