// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// usdGen engine — UsdGenGraphSpec, the engine's pure-value input (step 7.2).
// Every identity is a host-assigned UsdGenNodeKey; the host (usdGenUsd)
// lowers its path-based UsdGenGraphDesc into this. Path text survives only as
// labels, for diagnostics.
#ifndef USDGEN_API_GRAPH_SPEC_H
#define USDGEN_API_GRAPH_SPEC_H

#include "usdGen/api/elementSet.h"
#include "usdGen/api/inputDesc.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/span.h"
#include "usdGen/compile/types.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/handles.h"
#include "usdGen/registries/typeRegistry.h"
#include "usdGen/api/valueShape.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/vt/array.h"
#include "pxr/base/vt/value.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// One authored usdGen:* property, resolved. 'name' is the property name with
/// the 'usdGen:' prefix stripped (C1 registry, docs/freezes/C1.md).
struct UsdGenParamValue
{
    TfToken name;                // "clump:size", "width:knots", "surface"
    VtValue value;               // scalar, token, array, or resolved asset path
    bool    animated = false;    // authored with a .spline / time samples
};

// Invalid is deliberately distinct from CpuReference: a present but
// malformed/unknown backend request must fail closed instead of silently
// selecting the legacy CPU implementation. Values are stable: host-side graph
// snapshots carry them.
enum class UsdGenExecutionBackend : uint8_t {
    CpuReference = 0,
    Cuda = 1,
    Invalid = 2,
    Metal = 3,
    Vulkan = 4
};

/// One connected parameter (was UsdGenExpressionBinding). 'output' empty means
/// the connection named the expression PRIM rather than one of its outputs;
/// UsdGenFindExpressionOutput resolves it.
struct UsdGenParamBinding {
    UsdGenNodeKey expression; TfToken output = UsdGenTokens().result; TfToken nativeType = UsdGenTokens().float_; TfToken destination;
    expr::ValueShape destinationShape; expr::Domain domain = expr::Domain::Groom;
    VtValue literal;
};

/// S11 ramp encodings, already resolved by the adapter (02-schema.md §2.17).
struct UsdGenRampDesc
{
    VtVec2fArray knots;          // scalar ramp: (position, value), sorted by x
    VtFloatArray positions;     // colour ramp
    VtVec3fArray colors;        // colour ramp
    TfToken      interpolation; // linear | catmullRom | bspline | constant (default catmullRom, R11)
};

/// The authored relationship slot which consumes a map. usdGen:map is the
/// only map relationship; the binding keeps its exact authored name so a
/// diagnostic can quote it. Empty is valid for clients which only populate
/// UsdGenOpNode::maps.
struct UsdGenMapBinding
{
    UsdGenNodeKey map;
    TfToken       relationship;
};

/// One operator (was UsdGenNodeDesc). Builder order is the Kahn tie-break.
struct UsdGenOpNode
{
    UsdGenNodeKey                key;          // the operator prim (identity)
    TfToken                      type;         // the operator's schema type name
    TfToken                      mode;         // usdGen:mode, when the type has one
    bool                         enabled = true;
    int                          seed = 0;
    std::vector<UsdGenNodeKey>   inputs;       // compiler-owned hierarchy dependency edges
    std::vector<UsdGenNodeKey>   references;   // reference sets the node reads by name
    std::vector<UsdGenNodeKey>   inputSets;    // input records the node reads as element sets
    std::vector<UsdGenNodeKey>   surfaces;     // the Description's usdGen:surface targets (ADR R15)
    // Untyped map keys. Builders populate both; mapBindings is the canonical form.
    std::vector<UsdGenNodeKey>   maps;
    std::vector<UsdGenMapBinding> mapBindings;
    std::vector<UsdGenParamValue> params;      // EVERY mapped locator of this prim (S14 pull-all)
    std::vector<UsdGenParamBinding> expressionBindings;
    std::vector<UsdGenRampDesc>   ramps;
    UsdGenOpId                   opId = UsdGenOpId::Invalid;   // engine-filled at intake (UsdGenResolveDescIds)
};

/// Compiler-resolved map value. It contains no stage or backend handle: the
/// identity is a deterministic digest of the fully resolved input and can be
/// retained safely by a CPU capture (or rejected by a backend which has no
/// transport for it).
struct UsdGenResolvedMapValue
{
    UsdGenNodeKey key;
    TfToken     type;
    std::string resolvedAssetPath;
    uint64_t    textureGeneration = 0;
    uint64_t    identity = 0;
    UsdGenTypeId typeId = UsdGenTypeId::Invalid;
    UsdGenMapOptions options;            // the map type's parseOptions, run once at compile
};

/// One external input of a registered kind (an element set, a surface, a
/// map, an expression program, ...): free-length columns per its kind's
/// registered layout, and values in its InputDesc's attribute order.
/// Primitives read it through their own input views.
struct UsdGenInputRecord
{
    UsdGenNodeKey key;
    UsdGenNodeKey owner;            // GeomSubset: its Mesh; paint map: its surface
    UsdGenTypeId  kind = UsdGenTypeId::Invalid;
    TfToken       label;            // the host path text; diagnostics only
    UsdGenRef<const UsdGenElementSet> data;
    std::vector<UsdGenParamValue> values;
    uint64_t      generation = 0;   // content identity the host bumps on any edit

    template <class T>
    UsdGenSpan<const T> Column(UsdGenSlot slot) const noexcept
    {
        return data ? data->Column<T>(slot) : UsdGenSpan<const T>();
    }
    VtValue const *Value(size_t index) const noexcept
    {
        return index < values.size() ? &values[index].value : nullptr;
    }
    template <class T>
    T const &Get(size_t index, T const &fallback) const noexcept
    {
        VtValue const *v = Value(index);
        return v && v->IsHolding<T>() ? v->UncheckedGet<T>() : fallback;
    }
};

/// Host-owned memo of the columns a host packed. A column whose source
/// VtArray is the same buffer as at the last packing of that input is shared
/// from the previous record instead of copied, so an unchanged input keeps
/// its column addresses across lowerings and identity-keyed caches hit. It
/// holds each source array (a held buffer is immutable, so the same address
/// is the same bytes) and the set its column lives in. Not thread-safe; one
/// per host builder.
class UsdGenInputColumnCache
{
public:
    USDGEN_CORE_API UsdGenInputColumnCache();
    USDGEN_CORE_API ~UsdGenInputColumnCache();
    /// Starts a lowering: forgets columns unused for 'age' lowerings.
    USDGEN_CORE_API void Advance(uint32_t age = 4);
    USDGEN_CORE_API void Clear() noexcept;
    size_t Size() const noexcept { return _entries.size(); }

private:
    friend class UsdGenInputRecordWriter;
    struct Entry {
        UsdGenNodeKey key;
        UsdGenTypeId kind = UsdGenTypeId::Invalid;
        UsdGenSlot slot = kUsdGenNoSlot;
        void const *data = nullptr;
        size_t bytes = 0;
        VtValue source;                            // holds the source buffer
        UsdGenRef<const UsdGenElementSet> set;     // holds the column
        uint64_t used = 0;
    };
    std::unordered_map<uint64_t, Entry> _entries;
    uint64_t _epoch = 0;
};

/// Packs one input record of a registered kind. Columns are staged (the
/// caller's memory must outlive Finish) or shared from another record's data.
/// Compile phase: allocates.
class UsdGenInputRecordWriter
{
public:
    USDGEN_CORE_API explicit UsdGenInputRecordWriter(UsdGenTypeId kind);
    UsdGenInputRecordWriter(UsdGenInputRecordWriter const &) = delete;
    UsdGenInputRecordWriter &operator=(UsdGenInputRecordWriter const &) = delete;
    /// False for an unregistered kind or one without a layout and InputDesc.
    bool Valid() const noexcept { return _layout && _desc; }
    template <class T>
    void Column(UsdGenSlot slot, T const *data, size_t count)
    {
        _Stage(slot, data, count * sizeof(T));
    }
    /// A host array: with a column cache (UseCache), an unchanged buffer
    /// shares the previous record's column. Empty leaves the column absent.
    template <class T>
    void Array(UsdGenSlot slot, VtArray<T> const &array)
    {
        Column(slot, array.cdata(), array.size());
        if (_cache && !array.empty() && slot < _sources.size()) _sources[slot] = VtValue(array);
    }
    template <class T>
    void Array(UsdGenSlot slot, UsdGenSpan<T> array)
    {
        Column(slot, array.cdata(), array.size());
    }
    /// Remember and reuse this input's columns across packings.
    void UseCache(UsdGenInputColumnCache *cache, UsdGenNodeKey key) noexcept
    {
        _cache = cache;
        _cacheKey = key;
    }
    /// Shares 'from''s column (same scalar and arity) without copying.
    USDGEN_CORE_API void Share(UsdGenSlot slot, UsdGenElementSet const &from, UsdGenSlot fromSlot);
    /// The value at 'index' (the InputDesc's attribute order).
    USDGEN_CORE_API void Value(size_t index, VtValue value);
    /// Values appended after the InputDesc's attributes (map parameters).
    USDGEN_CORE_API void Extra(UsdGenParamValue value);
    USDGEN_CORE_API UsdGenInputRecord Finish(UsdGenNodeKey key, UsdGenNodeKey owner,
                                             TfToken const &label, uint64_t generation);

private:
    struct Staged {
        void const *data = nullptr;
        size_t bytes = 0;
        UsdGenElementSet const *from = nullptr;
        UsdGenSlot fromSlot = kUsdGenNoSlot;
    };
    USDGEN_CORE_API void _Stage(UsdGenSlot slot, void const *data, size_t bytes);
    UsdGenTypeId _kind;
    UsdGenElementLayout const *_layout = nullptr;
    UsdGenInputDesc const *_desc = nullptr;
    std::vector<Staged> _staged;
    std::vector<VtValue> _sources;   // UseCache: the host arrays staged per slot
    std::vector<UsdGenParamValue> _values;
    UsdGenInputColumnCache *_cache = nullptr;
    UsdGenNodeKey _cacheKey;
};

/// The description's look block (02-schema.md §2.12 defaults; 07 §1.2 bake
/// order): base colour, the root->tip ramp and the per-element jitter.
struct UsdGenLookDesc
{
    GfVec3f  rootColor{0.035f, 0.018f, 0.008f};
    GfVec3f  tipColor{0.210f, 0.115f, 0.045f};
    VtVec3fArray rampColors;          // usdGen:look:colorRamp:colors (empty == two-colour mix)
    VtFloatArray rampPositions;       // usdGen:look:colorRamp:positions
    TfToken  rampInterpolation = UsdGenTokens().catmullRom;
    float    rampExponent = 1.0f;     // usdGen:look:rampExponent
    TfToken  bakeMode;                                 // host-filled (the schema's fallback)
    TfToken  bakeTarget = UsdGenTokens().displayColor;     // displayColor | primvar | none
    TfToken  bakePrimvar = UsdGenTokens().displayColor;
    float    hueJitter = 0.0f;
    float    valueJitter = 0.0f;
    int      jitterSeed = 0;
};

/// usdGen:preview:* on the description: a viewport-only colour override that
/// shows a value on the published geometry (07 §7.3). 'source' is the first authored
/// target, resolved by the host to a prim and an attribute: a UsdGenExpression
/// prim, a UsdGenPtexMap prim (the builders add it to 'maps') or an operator
/// attribute ('attribute' non-empty). The session cooker resolves it against
/// the compiled graph.
struct UsdGenPreviewSpec
{
    UsdGenNodeKey source;                         // invalid == no preview
    TfToken  attribute;                           // the operator attribute; empty for a prim
    TfToken  colorMap = UsdGenTokens().heat;          // heat | viridis | gray | ids | rgb
    GfVec2f  range{0.0f, 1.0f};
    TfToken  evaluation = UsdGenTokens().primitive;   // for an expression prim source
    TfToken  shading = UsdGenTokens().lit;            // lit | flat

    bool Active() const { return source.IsValid(); }
};

/// The text a key stands for (the host's prim path), for diagnostics only.
/// A derived key's text is its parent's text, '/', then 'text'.
struct UsdGenNodeLabel
{
    UsdGenNodeKey key;
    UsdGenNodeKey parent;     // valid for a derived key
    TfToken       text;
};

/// FNV-1a64 of a path text: the key a host gives a path that collides with
/// nothing (usdGenUsd NodeKeyMap); 1 for a zero hash.
UsdGenNodeKey UsdGenNodeKeyOfText(std::string const &text) noexcept;

struct UsdGenGraphSpec
{
    UsdGenNodeKey                  description;   // the UsdGenDescription prim
    UsdGenNodeKey                  terminal;      // hierarchy-derived final operator
    std::vector<UsdGenOpNode>      nodes;         // composed execution order from the builder
    /// Every external input (element sets, surfaces, maps, expressions, the
    /// gprims expressions sample), grouped by ascending kind; builder order
    /// within a kind (a surface's id is its index in its run).
    std::vector<UsdGenInputRecord> inputs;
    UsdGenLookDesc                 look;
    UsdGenPreviewSpec              preview;
    GfMatrix4d                     xformMatrix{1.0}; // description world matrix (post-flattening, S4)
    TfToken                        purpose;        // inherited by hand to every tile (C2)
    TfToken                        visibility;     // inherited by hand to every tile (C2)
    UsdGenNodeKey                  material;       // the description's bound Material (C2)
    float    defaultWidth = 0.01f;                 // usdGen:width:default
    int      tileTarget = 64;                      // uniform int usdGen:tileTarget
    TfToken  basis = UsdGenTokens().bspline;           // the published basis (C2)
    double   time = 0.0;
    double   timeCodesPerSecond = 24.0; // USD default; expressions expose seconds
    UsdGenExecutionBackend executionBackend = UsdGenExecutionBackend::CpuReference;
    std::vector<std::string> validationErrors;
    /// Sorted by key: the text of every key above.
    std::vector<UsdGenNodeLabel> labels;
    /// Opaque host data (usdGenUsd: this spec's key -> path table). The engine
    /// copies it into the generations and routing snapshots it makes from this
    /// spec and never reads it.
    std::shared_ptr<const void> host;

    /// The run of 'kind' in 'inputs' (empty when absent).
    UsdGenSpan<const UsdGenInputRecord> Inputs(UsdGenTypeId kind) const noexcept;
    UsdGenSpan<UsdGenInputRecord> MutableInputs(UsdGenTypeId kind) noexcept;
    /// The record of 'kind' with 'key', else null (linear within the run).
    UsdGenInputRecord const *FindInput(UsdGenTypeId kind, UsdGenNodeKey key) const noexcept;
    /// Stable sort of 'inputs' by kind (builders append kinds in any order).
    void SortInputs();
    /// The text of 'key' (its prim path), else empty. Allocates: diagnostics only.
    std::string Label(UsdGenNodeKey key) const;
    /// The host label token of 'key' without allocating; empty for a
    /// derived or unknown key.
    TfToken const &LabelToken(UsdGenNodeKey key) const noexcept;
    /// The child name of a derived key (DeriveKey); empty for a host key.
    TfToken const &DerivedName(UsdGenNodeKey key) const noexcept;
    /// The description's path text (expr::DescriptionId hashes it).
    char const *DescriptionText() const noexcept { return LabelToken(description).GetText(); }
    /// The key of a synthetic child of 'parent' (an engine-made node or
    /// input), keyed like the host keys the path <parent>/<child>, and its
    /// label. Compile phase: allocates.
    UsdGenNodeKey DeriveKey(UsdGenNodeKey parent, TfToken const &child);
    /// Label(a) < Label(b): the order of the host's paths (prim paths order as
    /// their text).
    bool LabelLess(UsdGenNodeKey a, UsdGenNodeKey b) const;
};

/// spec->Label(key), or empty without a spec.
inline std::string UsdGenLabel(UsdGenGraphSpec const *spec, UsdGenNodeKey key)
{
    return spec ? spec->Label(key) : std::string();
}

}  // namespace usdGen

#endif  // USDGEN_API_GRAPH_SPEC_H
