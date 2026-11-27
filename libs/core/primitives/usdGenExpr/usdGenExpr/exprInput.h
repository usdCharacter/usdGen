// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The Expression and Geometry input kinds (step 7.3a): an expression
// program and the gprims its samplers read, as a host captures them, their
// record layouts and InputDescs, the packers and the read views.
#pragma once
#include "usdGen/api/graphSpec.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"
#include "usdGen/api/cowArray.h"
#include "usdGenMath/math.h"
#include "usdGen/api/value.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGenExpr/export.h"
#include "usdGenExpr/expressions/context.h"


#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace usdGen {

class UsdGenRegistrar;

struct UsdGenExpressionOutputDesc { UsdGenToken name = UsdGenTokens().result; UsdGenToken nativeType = UsdGenTokens().float_; expr::ValueShape shape; };

/// One input:<name> relationship of an expression: the external data
/// geoSampler("<name>", ...) and ptex("<name>") read. 'targets' is the
/// authored (forwarded) target list; 'geometries' the gprims it resolves to,
/// in namespace order; 'maps' the map prims, in order.
struct UsdGenExpressionSlot
{
    UsdGenToken name;                            // "guideCurves" for input:guideCurves
    std::vector<UsdGenNodeKey> targets;
    std::vector<UsdGenNodeKey> geometries;
    std::vector<UsdGenNodeKey> maps;
};

USDGEN_EXPR_API bool operator==(UsdGenExpressionOutputDesc const &, UsdGenExpressionOutputDesc const &) noexcept;
USDGEN_EXPR_API bool operator==(UsdGenExpressionSlot const &, UsdGenExpressionSlot const &) noexcept;
inline bool operator!=(UsdGenExpressionOutputDesc const &a, UsdGenExpressionOutputDesc const &b) noexcept { return !(a == b); }
inline bool operator!=(UsdGenExpressionSlot const &a, UsdGenExpressionSlot const &b) noexcept { return !(a == b); }

/// The record's outputs and slots values: Boxed payloads of usdGenExpr
/// (allocated and freed here).
USDGEN_EXPR_API UsdGenValue UsdGenExpressionOutputsValue(std::vector<UsdGenExpressionOutputDesc> const &outputs);
USDGEN_EXPR_API UsdGenValue UsdGenExpressionSlotsValue(std::vector<UsdGenExpressionSlot> const &slots);
/// The payload a value holds, or null.
USDGEN_EXPR_API std::vector<UsdGenExpressionOutputDesc> const *UsdGenExpressionOutputsOf(UsdGenValue const &v) noexcept;
USDGEN_EXPR_API std::vector<UsdGenExpressionSlot> const *UsdGenExpressionSlotsOf(UsdGenValue const &v) noexcept;

/// An expression program as a host captures it; travels as a
/// UsdGenExpressionInput record (values: source, outputs, slots).
struct UsdGenExpressionData
{
    std::string source;
    std::vector<UsdGenExpressionOutputDesc> outputs;
    std::vector<UsdGenExpressionSlot> inputs;
};

/// The output a binding names on 'expression' (a host desc or an engine
/// input). A named output must exist. An EMPTY name is a prim connection: it
/// resolves to "result" when the expression declares it, otherwise to its
/// single declared output. Returns nullptr when nothing matches, or when a
/// prim connection is ambiguous.
template <class Expression>
inline UsdGenExpressionOutputDesc const *
UsdGenFindExpressionOutput(Expression const &expression, UsdGenToken const &output)
{
    if (!output.IsEmpty()) {
        for (UsdGenExpressionOutputDesc const &o : expression.outputs)
            if (o.name == output) return &o;
        return nullptr;
    }
    for (UsdGenExpressionOutputDesc const &o : expression.outputs)
        if (o.name == UsdGenTokens().result) return &o;
    return expression.outputs.size() == 1 ? &expression.outputs.front() : nullptr;
}

/// A gprim an expression samples (geoSampler), by value. Points are in the
/// prim's object space; worldMatrix places them.
enum class UsdGenGeometryKind : uint8_t { Mesh, Curves, Points };

/// The payload of a sampled gprim as a host captures it; travels as a
/// UsdGenGeometryInput record.
struct UsdGenGeometryData
{
    UsdGenGeometryKind kind = UsdGenGeometryKind::Points;
    UsdGenIntArray         counts;       // faceVertexCounts | curveVertexCounts | empty
    UsdGenIntArray         indices;      // faceVertexIndices (mesh)
    UsdGenVec3fArray       points;       // at the graph time
    UsdGenVec3fArray       rest;         // Default-time rest when the prim has one, else empty
    UsdGenVec3fArray       normals;      // points: optional per-point normals
    UsdGenCowArray<uint64_t>  ids;          // curves: usdGen:curveId; points: ids (optional)
    // A GeomSubset input (R15): the arrays above are its parent Mesh's, and
    // only subsetFaces (sorted parent face ids) and their points are sampled.
    // isSubset keeps an empty subset empty.
    UsdGenIntArray         subsetFaces;
    bool               isSubset = false;
    UsdGenMatrix4d         worldMatrix{1.0};
    /// Content identity of everything above, computed by the builder. Any
    /// edit of the prim's data changes it, so a consumer's capture identity
    /// can fold it instead of rehashing arrays.
    uint64_t           generation = 0;
};

enum UsdGenExpressionInputValue : uint8_t {
    kUsdGenExpressionValueSource,    // std::string
    kUsdGenExpressionValueOutputs,   // std::vector<UsdGenExpressionOutputDesc>
    kUsdGenExpressionValueInputs,    // std::vector<UsdGenExpressionSlot>
    kUsdGenExpressionValues
};

enum UsdGenGeometryInputSlot : UsdGenSlot {
    kUsdGenGeometryInputCounts,        // I32
    kUsdGenGeometryInputIndices,       // I32
    kUsdGenGeometryInputPoints,        // F32x3
    kUsdGenGeometryInputRest,          // F32x3
    kUsdGenGeometryInputNormals,       // F32x3
    kUsdGenGeometryInputIds,           // U64
    kUsdGenGeometryInputSubsetFaces,   // I32
    kUsdGenGeometryInputSlots
};

enum UsdGenGeometryInputValue : uint8_t {
    kUsdGenGeometryValueKind,          // int (UsdGenGeometryKind)
    kUsdGenGeometryValueIsSubset,      // bool
    kUsdGenGeometryValueWorldMatrix,   // UsdGenMatrix4d
    kUsdGenGeometryValues
};

USDGEN_EXPR_API UsdGenTypeId UsdGenExpressionInputKind() noexcept;
USDGEN_EXPR_API UsdGenTypeId UsdGenGeometryInputKind() noexcept;
/// Registers "UsdGenExpressionInput" and "UsdGenGeometryInput".
USDGEN_EXPR_API void UsdGenRegisterExprInputKinds(UsdGenRegistrar &);

/// Read view of an expression input. Non-owning: valid while its record (or
/// payload) lives. No allocation.
struct UsdGenExpressionInputView
{
    UsdGenExpressionInputView() = default;
    /// The view of a record (Of(record)); implicit so a record passes where a view is read.
    UsdGenExpressionInputView(UsdGenInputRecord const &record) noexcept : UsdGenExpressionInputView(Of(record)) {}
    USDGEN_EXPR_API static UsdGenExpressionInputView Of(UsdGenInputRecord const &) noexcept;
    USDGEN_EXPR_API static UsdGenExpressionInputView Of(UsdGenExpressionData const &,
                                                        UsdGenNodeKey key = {}) noexcept;
    UsdGenNodeKey key;
    std::string_view source;
    UsdGenSpan<const UsdGenExpressionOutputDesc> outputs;
    UsdGenSpan<const UsdGenExpressionSlot> inputs;
    /// Keep a record's values alive (shared, no copy): a view may outlive its spec.
    UsdGenValue sourceHolder, outputsHolder, inputsHolder;
};

/// Read view of a sampled gprim with the payload's field names.
struct UsdGenGeometryInputView
{
    UsdGenGeometryInputView() = default;
    /// The view of a record (Of(record)); implicit so a record passes where a view is read.
    UsdGenGeometryInputView(UsdGenInputRecord const &record) noexcept : UsdGenGeometryInputView(Of(record)) {}
    USDGEN_EXPR_API static UsdGenGeometryInputView Of(UsdGenInputRecord const &) noexcept;
    USDGEN_EXPR_API static UsdGenGeometryInputView Of(UsdGenGeometryData const &,
                                                      UsdGenNodeKey key = {}) noexcept;
    UsdGenNodeKey key;
    UsdGenGeometryKind kind = UsdGenGeometryKind::Points;
    UsdGenSpan<const int> counts, indices;
    UsdGenSpan<const UsdGenVec3f> points, rest, normals;
    UsdGenSpan<const uint64_t> ids;
    UsdGenSpan<const int> subsetFaces;
    bool isSubset = false;
    UsdGenMatrix4d worldMatrix{1.0};
    uint64_t generation = 0;
    UsdGenRef<const UsdGenElementSet> data;   // keeps a record's columns alive
};

USDGEN_EXPR_API UsdGenSpan<const UsdGenInputRecord> UsdGenExpressionInputs(UsdGenGraphSpec const &) noexcept;
USDGEN_EXPR_API UsdGenSpan<const UsdGenInputRecord> UsdGenGeometryInputs(UsdGenGraphSpec const &) noexcept;
USDGEN_EXPR_API UsdGenInputRecord const *UsdGenFindExpressionInput(UsdGenGraphSpec const &, UsdGenNodeKey) noexcept;
USDGEN_EXPR_API UsdGenInputRecord const *UsdGenFindGeometryInput(UsdGenGraphSpec const &, UsdGenNodeKey) noexcept;

/// Packers. Compile phase: allocate.
USDGEN_EXPR_API UsdGenInputRecord UsdGenExpressionRecord(UsdGenExpressionData const &data,
                                                         UsdGenNodeKey key, UsdGenText const &label);
USDGEN_EXPR_API UsdGenInputRecord UsdGenGeometryRecord(UsdGenGeometryData const &data,
                                                       UsdGenNodeKey key, UsdGenText const &label,
                                                       UsdGenInputColumnCache *cache = nullptr);

}  // namespace usdGen
