// usdGen engine — compiler (03-execution-engine.md §3).
//
// Compile() walks UsdGenGraphSpec::nodes and: edges from the composed
// hierarchy order; Kahn sort with namespace tie-break; cycle detection
// (compile error naming the offending pair); dense node ids in topological
// order; reference-lane ordering (§1.5); OutputPrimvars
// slot binding (§1.2); Merkle structural digests (§3.3); tile arithmetic
// (R21); dirty routing table rebuild data (§5.1).
#ifndef USDGEN_COMPILER_H
#define USDGEN_COMPILER_H

#include "usdGen/api/export.h"
#include "usdGen/api/compile.h"
#include "usdGen/compile/graph.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/types.h"

#include <string>
#include <functional>
#include <memory>
#include <vector>


namespace usdGen {

class UsdGenExecutionPlanHandle;
struct UsdGenDiagnostics;

class USDGEN_INTERNAL_API UsdGenCompiler
{
public:
    /// Fills *out (moves the compiled state). ok=false leaves *out untouched.
    /// Recompile is transactional too: validation runs before any node moves.
    UsdGenCompileResult Compile(UsdGenGraphSpec const &desc, UsdGenGraph *out);

    /// Private integration boundary for an explicitly injected device provider.
    /// Runs the normal structural/routing compilation, but obtains the native
    /// plan from this exact descriptor without changing global availability.
    /// No source capture or CPU geometry evaluation occurs. Both outputs stay
    /// untouched on failure. Used by the injected-provider route.
    using DevicePlanCompiler = UsdGenDevicePlanCompiler;
    UsdGenCompileResult CompileInjectedDevice(UsdGenGraphSpec const&,
        UsdGenGraph*, DevicePlanCompiler const&,
        std::shared_ptr<const UsdGenExecutionPlanHandle>* plan);
    /// Compile() plus the device plan of a kUsdGenBackendReusesPlan backend,
    /// compiled after the graph. Both outputs stay untouched on failure.
    UsdGenCompileResult CompileWithDevicePlan(UsdGenGraphSpec const&,
        UsdGenGraph*, DevicePlanCompiler const&,
        std::shared_ptr<const UsdGenExecutionPlanHandle>* plan);

    /// Incremental: rebuild only the sub-graph whose Merkle digest moved
    /// (03 §3.5). One node appended to a 200-node groom rebuilds exactly one
    /// node (gate E-6).
    UsdGenCompileResult Recompile(UsdGenGraphSpec const &newDesc, UsdGenGraph *out);

private:
    friend class UsdGenGraph;  // fills the compiled structure in place
    /// Shared pipeline for Compile() / Recompile(); `reuse` (non-null on the
    /// incremental path) supplies captures/buffers for digest-stable nodes.
    static void _Build(UsdGenGraphSpec const &desc, UsdGenGraph *out,
                       UsdGenGraph const *reuse, UsdGenCompileResult &result);
};

}  // namespace usdGen

#endif  // USDGEN_COMPILER_H
