#ifndef USDGEN_EXPRESSIONS_FRONTEND_H
#define USDGEN_EXPRESSIONS_FRONTEND_H
#include "usdGen/expressions/context.h"
#include "usdGen/expressions/ir.h"
#include <memory>
#include <string>
#include <vector>
namespace usdGen::expr {
// SeExpr2 performs numeric math in FP; destination integer/bool conversion is
// deferred to the GPU boundary. int64/uint64 results are exact only through 2^53.
struct FrontendOptions {
    Domain domain = Domain::Groom;
    ScalarType destination = ScalarType::Float32;
    uint32_t components = 1;
    /// Compiles a geoSampler() element expression: $Q/$Qdist are admitted and
    /// sampler calls (geoSampler, ptex) are not.
    bool samplerElement = false;
};
class Program {
public:
    Program(); ~Program(); Program(Program&&) noexcept; Program& operator=(Program&&) noexcept;
    Program(Program const&) = delete; Program& operator=(Program const&) = delete;
    bool Valid() const noexcept; std::string const &Source() const noexcept;
    IRProgram const &IR() const noexcept;
private:
    struct Impl; std::unique_ptr<Impl> _impl; explicit Program(std::unique_ptr<Impl>); friend class Frontend;
};
struct CompileResult { bool ok=false; Program program; std::vector<std::string> diagnostics; };
/// One entry of the single source of truth for "what can I call in a usdGen
/// expression". Walk() rejects every function absent from this list, and
/// Lower() implements exactly the ones it contains, so the editor, the docs
/// and the compiler cannot disagree.
struct FunctionInfo {
    std::string name;
    /// The arity a completion should insert: the fixed arity for a fixed
    /// function, the minimum for a variadic one. Kept first and unchanged in
    /// meaning because the tools ABI reads it.
    uint32_t arity = 0;
    uint32_t minArity = 0;
    /// 0 means "no upper bound".
    uint32_t maxArity = 0;
    bool variadic = false;
    std::string signature;   // e.g. "clamp(x, lo, hi)"
    std::string doc;         // one line
    /// math | noise | vector | color | curve | control | sampling. The editor's
    /// function browser groups on this.
    std::string category;
    /// 1 for a scalar result, 3 for a vector/colour result.
    uint32_t components = 1;
};
/// One entry of the variable table, with everything a browser needs to present
/// it. Generated from expressions/context.cpp's registry, which is the only
/// place a variable is declared.
struct VariableDoc {
    std::string name;
    std::string type;        // "float32", "int32", ... or "" for the polymorphic $value
    uint32_t components = 1; // 0 for $value, whose shape belongs to the consumer
    std::string domains;     // comma-separated subset of groom,primitive,point
    std::string doc;
};
class Frontend {
public:
    static CompileResult Compile(std::string const&, FrontendOptions const& = {});
    /// Sorted by name. Stable across both execution lanes.
    static std::vector<FunctionInfo> SupportedFunctions();
    /// In registry order. Every entry carries a non-empty doc and domain list.
    static std::vector<VariableDoc> VariableDocs();
};
}
#endif
