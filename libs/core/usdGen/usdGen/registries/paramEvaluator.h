// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Connected parameters without naming expressions: the engine holds one
// evaluator state per compiled node; the registered evaluator fills it.
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include "usdGen/exec/pool.h"
#include "usdGen/registries/handles.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace usdGen {

struct UsdGenGraphSpec;
struct UsdGenOpNode;
struct UsdGenElementSetView;
struct UsdGenDiagnostics;

/// A parameter that may be a literal or a connected value. Value() is the one
/// indexing rule (gpu::ReadScalar/ReadBool/ReadInt match it): level 0
/// broadcasts, level 1 indexes by absolute element, level 2 by absolute item.
struct UsdGenParamField
{
    double literal = 0.0;
    double const *values = nullptr;
    size_t count = 0;
    uint32_t components = 1;
    uint8_t level = 0;
    bool connected = false;

    double Value(size_t element, size_t item, uint32_t component = 0) const
    {
        if (!connected || !values || component >= components) return literal;
        size_t const index = level == 1 ? element : level >= 2 ? item : size_t(0);
        return index >= count ? literal : values[index * components + component];
    }
    /// True when every element reads the same value (hoistable).
    bool Uniform() const { return !connected || level == 0; }
};

/// One connected destination's evaluated values: count * components doubles,
/// count 1 at level 0, the element count at 1, the item count at 2.
struct UsdGenEvaluatedParam
{
    UsdGenToken destination;             // canonical: no `usdGen:` prefix
    uint8_t level = 0;
    uint32_t components = 1;
    size_t count = 0;
    UsdGenSpan<const double> values;
};

/// Per compiled node; owned by the node. Evaluate runs once per cook over
/// the node's input, before its capture identity is taken.
class UsdGenParamEvaluatorState : public UsdGenPooled
{
public:
    virtual ~UsdGenParamEvaluatorState() = default;
    /// False with errors on diag on failure; the published values are then
    /// cleared. *changed: the published values differ from the last call.
    virtual bool Evaluate(UsdGenGraphSpec const &, UsdGenOpNode const &,
                          UsdGenElementSetView const &geometry, double frame, double time,
                          uint32_t seed, bool *changed, UsdGenDiagnostics *diag) = 0;
    /// Drops the published values; true when any were published.
    virtual bool Clear() noexcept = 0;
    /// By name, with or without `usdGen:`. Never interns.
    virtual UsdGenEvaluatedParam const *Find(UsdGenToken const &destination) const noexcept = 0;
    /// By the node's param slot, O(1).
    virtual UsdGenEvaluatedParam const *FindSlot(UsdGenParamSlot slot) const noexcept = 0;
    /// Identity of the published values (folded into the capture identity).
    virtual uint64_t Digest() const noexcept = 0;
    /// Moves the last evaluation's non-fatal warnings onto diag.
    virtual void TakeWarnings(UsdGenDiagnostics *diag) = 0;
};

struct UsdGenParamEvaluatorDef
{
    std::unique_ptr<UsdGenParamEvaluatorState> (*create)() = nullptr;
    /// Optional compile checks of the evaluator's inputs in the spec; false
    /// appends to 'errors'.
    bool (*validateInputs)(UsdGenGraphSpec const &, std::vector<std::string> *errors) = nullptr;
    /// Optional compile checks of one node's connected parameters, after
    /// the record's admission checks.
    bool (*validateBindings)(UsdGenGraphSpec const &, UsdGenOpNode const &,
                             std::vector<std::string> *errors) = nullptr;
    /// Optional: keys whose edits re-evaluate a node through one binding's
    /// source, in order.
    void (*bindingDependencies)(UsdGenGraphSpec const &, UsdGenNodeKey source, void *context,
                                void (*add)(void *context, UsdGenNodeKey)) = nullptr;
};

/// A fresh state from the registered evaluator; null when none is
/// registered. Compile phase.
USDGEN_CORE_API std::unique_ptr<UsdGenParamEvaluatorState> UsdGenCreateParamEvaluatorState();

}  // namespace usdGen
