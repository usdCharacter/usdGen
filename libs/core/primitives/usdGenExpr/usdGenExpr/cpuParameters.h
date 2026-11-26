// usdGen engine — connected-parameter (expression) evaluation for the CPU
// reference lane.
//
// This is the host counterpart of cudaParameters.{h,cpp}: one compiled program
// per node, one geometry context per evaluation domain, and one decoded result
// array per connected destination. It exists so a Description that runs on the
// CPU lane — the only lane that currently reaches usdview — honours
// `usdGen:width.connect`, `usdGen:mask.connect` and the rest instead of
// refusing them.
//
// Parity with the CUDA lane: the IR interpreter and the geometry fields are
// shared code (expressions/irExec.h, expressions/cpuEvaluator.{h,cpp}), the
// admissible destinations/domains per operator are the op record's
// parameter decls (UsdGenValidateExpressionBindings), and the per-element index rule below is the
// one gpu::ReadScalar/ReadBool/ReadInt implement: groom broadcasts, primitive
// indexes by curve, point indexes by CV.
#ifndef USDGEN_CPU_PARAMETERS_H
#define USDGEN_CPU_PARAMETERS_H

#include "usdGenCurves/types/curveSet.h"
#include "usdGenSurface/maps/mapInput.h"
#include "usdGenExpr/exprInput.h"
#include "usdGen/expressions/cpuEvaluator.h"
#include "usdGen/expressions/ir.h"
#include "usdGen/expressions/samplers.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/registries/handles.h"
#include "usdGen/registries/paramEvaluator.h"
#include "usdGen/registries/typeRegistry.h"

#include "pxr/base/tf/token.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenPtexTexture;
class UsdGenOpRecord;

/// The node's op record: the intake-resolved id, else a registry probe by
/// type (descs built outside the compiler). Null for an unregistered type.
/// Never interns.
UsdGenOpRecord const *UsdGenNodeRecord(UsdGenOpNode const &node) noexcept;

/// True when the node's op has the emitsLimitSurfaceUv hook and it fires
/// for the node's authored params.
bool UsdGenNodeEmitsLimitSurfaceUv(UsdGenGraphSpec const &desc, UsdGenOpNode const &node);

/// Parses a map prim's usdGen:map:* params once (the map types'
/// parseOptions). Unauthored or mistyped params keep their defaults.
bool UsdGenParseMapOptions(UsdGenMapInputView const &map, UsdGenMapOptions *out,
                           std::string *error);

/// One connected destination's evaluated values, decoded from the typed
/// destination buffer into doubles. `count` is 1 for the groom domain, the
/// curve count for primitive and the CV count for point; `level` is the
/// domain's (0, 1, 2) and `values` views `storage`.
struct UsdGenExpressionValue : UsdGenEvaluatedParam
{
    UsdGenParamSlot slot = kUsdGenNoParam;   // the op's param slot; none for undeclared names
    expr::Domain domain = expr::Domain::Groom;
    expr::ScalarType type = expr::ScalarType::Invalid;
    std::vector<double> storage;         // count * components
};

/// Per-node connected-parameter state, owned by the compiled node. Compilation
/// is cached against a digest of the bindings and their expression sources, so
/// editing `usdGen:expr:source` or re-pointing a connection recompiles on the
/// next evaluation without any graph-level cooperation.
class UsdGenCpuParameters final : public UsdGenParamEvaluatorState
{
public:
    /// Compiles (if needed) and evaluates every binding on `node` over
    /// `geometry`. Returns false and fills `diagnostics` on any failure,
    /// leaving the previously published values cleared so no operator reads a
    /// stale connected value. `changed` reports whether the published values
    /// differ from the previous evaluation (the caller uses it to force a
    /// re-capture and a chunk sweep).
    bool Evaluate(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                  UsdGenCurveSetView const &geometry, double frame, double time,
                  uint32_t seed, bool *changed,
                  std::vector<std::string> *diagnostics);
    /// The engine's entry: the same over a curve-set element view.
    bool Evaluate(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                  UsdGenElementSetView const &geometry, double frame, double time,
                  uint32_t seed, bool *changed, UsdGenDiagnostics *diag) override;

    /// Clears the published values (a node with no bindings, or a disabled
    /// one). Reports whether anything was published before.
    bool Clear() noexcept override;

    /// By name, with or without `usdGen:`. Never interns.
    UsdGenExpressionValue const *Find(TfToken const &destination) const noexcept override;
    /// By the op's param slot, O(1).
    UsdGenExpressionValue const *FindSlot(UsdGenParamSlot slot) const noexcept override
    {
        uint16_t const v = slot < kUsdGenMaxParamsPerOp ? slotToValue_[slot] : 0;
        return v && v <= values_.size() ? &values_[v - 1u] : nullptr;
    }
    bool Empty() const noexcept { return values_.empty(); }
    /// Identity of the currently published values; folded into the node's
    /// capture identity so a source edit re-captures.
    uint64_t Digest() const noexcept override { return digest_; }

    /// Non-fatal diagnostics of the last evaluation (a map that could not be
    /// read and fell back to its usdGen:map:default). Cleared by the call.
    std::vector<std::string> TakeWarnings();
    void TakeWarnings(UsdGenDiagnostics *diag) override;

private:
    /// The resolved data behind one IROp::Sample slot of one binding.
    struct SamplerState {
        expr::IRSampler spec;
        UsdGenNodeKey expression;
        // Geometry
        std::vector<UsdGenNodeKey> geometries;
        std::unique_ptr<expr::GeometrySampler> geometry;
        bool readsTime = false;          // the element expression reads $frame/$time
        double builtFrame = 0.0, builtTime = 0.0;
        bool built = false;
        // Ptex / Paint
        UsdGenNodeKey map;
        bool paint = false;            // UsdGenPaintMap: read paintValues, no texture
        std::shared_ptr<const UsdGenPtexTexture> texture;
        int channel = 0;                 // index into the sampled window; -1 = luminance
        double scale = 1.0, offset = 0.0, fallback = 0.0;
        double clampLo = 0.0, clampHi = 1.0;
        std::vector<double> values;      // one per strand of the last evaluation
        size_t reportedMisses = 0;       // roots off the map, as last reported
    };
    struct Item {
        UsdGenParamBinding binding;
        UsdGenParamSlot slot = kUsdGenNoParam;
        TfToken canonical;               // the destination without `usdGen:`
        expr::IRProgram ir;
        std::vector<double> literal;
        std::vector<std::unique_ptr<SamplerState>> samplers;
        expr::CpuExpressionSamplers table;
    };
    bool ResolveSamplers(UsdGenGraphSpec const &desc, UsdGenExpressionInputView const &expression,
                         Item *item, std::vector<std::string> *diagnostics);
    bool PrepareSamplers(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                         UsdGenCurveSetView const &geometry, expr::Context const &controls,
                         Item *item, std::vector<std::string> *diagnostics);
    bool Compile(UsdGenGraphSpec const &desc, UsdGenOpNode const &node,
                 std::vector<std::string> *diagnostics);

    uint64_t sourceDigest_ = 0;      // identity of the compiled bindings/sources
    bool compiled_ = false;
    std::vector<Item> items_;
    std::vector<UsdGenExpressionValue> values_;
    UsdGenOpRecord const *record_ = nullptr;   // resolves names to slots
    uint16_t slotToValue_[kUsdGenMaxParamsPerOp] = {};   // values_ index + 1; 0 none
    void IndexSlots() noexcept;
    // Scratch that the borrowed geometry view points into; reused per frame so
    // evaluation allocates nothing after the first cook of a given topology.
    std::vector<uint32_t> offsets_;
    std::vector<uint64_t> ids_;
    std::vector<std::string> warnings_;
    uint64_t digest_ = 0;
};

} // namespace usdGen

#endif
