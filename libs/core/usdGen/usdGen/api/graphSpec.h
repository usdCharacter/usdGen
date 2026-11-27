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
#include "usdGen/api/types.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/ext/typeRegistry.h"
#include "usdGen/api/valueShape.h"
#include "usdGenMath/math.h"
#include "usdGenMath/math.h"
#include "usdGenMath/math.h"
#include "usdGen/api/text.h"
#include "usdGen/api/token.h"
#include "usdGen/api/cowArray.h"
#include "usdGen/api/value.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace usdGen {

/// One authored usdGen:* property, resolved. 'name' is the property name with
/// the 'usdGen:' prefix stripped (C1 registry, docs/freezes/C1.md).
struct UsdGenParamValue
{
    UsdGenToken name;                // "clump:size", "width:knots", "surface"
    UsdGenValue value;               // scalar, token, array, or resolved asset path
    bool    animated = false;    // authored with a .spline / time samples
};

/// One connected parameter (was UsdGenExpressionBinding). 'output' empty means
/// the connection named the expression PRIM rather than one of its outputs;
/// UsdGenFindExpressionOutput resolves it.
struct UsdGenParamBinding {
    UsdGenNodeKey expression; UsdGenToken output = UsdGenTokens().result; UsdGenToken nativeType = UsdGenTokens().float_; UsdGenToken destination;
    expr::ValueShape destinationShape; expr::Domain domain = expr::Domain::Groom;
    UsdGenValue literal;
};

/// S11 ramp encodings, already resolved by the adapter (02-schema.md §2.17).
struct UsdGenRampDesc
{
    UsdGenVec2fArray knots;          // scalar ramp: (position, value), sorted by x
    UsdGenFloatArray positions;     // colour ramp
    UsdGenVec3fArray colors;        // colour ramp
    UsdGenToken      interpolation; // the host's ramp interpolation token (R11)
};

/// The authored relationship slot which consumes a map. usdGen:map is the
/// only map relationship; the binding keeps its exact authored name so a
/// diagnostic can quote it. Empty is valid for clients which only populate
/// UsdGenOpNode::maps.
struct UsdGenMapBinding
{
    UsdGenNodeKey map;
    UsdGenToken       relationship;
};

/// One operator (was UsdGenNodeDesc). Builder order is the Kahn tie-break.
struct UsdGenOpNode
{
    UsdGenNodeKey                key;          // the operator prim (identity)
    UsdGenToken                      type;         // the operator's schema type name
    UsdGenToken                      mode;         // usdGen:mode, when the type has one
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
    UsdGenToken     type;
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
    UsdGenText        label;            // the host path text; diagnostics only
    UsdGenRef<const UsdGenElementSet> data;
    std::vector<UsdGenParamValue> values;
    uint64_t      generation = 0;   // content identity the host bumps on any edit

    template <class T>
    UsdGenSpan<const T> Column(UsdGenSlot slot) const noexcept
    {
        return data ? data->Column<T>(slot) : UsdGenSpan<const T>();
    }
    UsdGenValue const *Value(size_t index) const noexcept
    {
        return index < values.size() ? &values[index].value : nullptr;
    }
    template <class T>
    T const &Get(size_t index, T const &fallback) const noexcept
    {
        UsdGenValue const *v = Value(index);
        return v && v->IsHolding<T>() ? v->UncheckedGet<T>() : fallback;
    }
};

/// Host-owned memo of the columns a host packed. A column whose source
/// array is the same buffer as at the last packing of that input is shared
/// from the previous record instead of copied, so an unchanged input keeps
/// its column addresses across lowerings and identity-keyed caches hit. It
/// pins each source array's arena (an adopted host buffer or a pool arena;
/// a pinned buffer is immutable, so the same address is the same bytes) and
/// the set its column lives in. Not thread-safe; one
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
        UsdGenRef<const UsdGenArena> source;       // pins the source buffer
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
    void Array(UsdGenSlot slot, UsdGenCowArray<T> const &array)
    {
        Column(slot, array.cdata(), array.size());
        if (_cache && !array.empty() && slot < _sources.size()) _sources[slot] = array.ArenaRef();
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
    USDGEN_CORE_API void Value(size_t index, UsdGenValue value);
    /// Values appended after the InputDesc's attributes (map parameters).
    USDGEN_CORE_API void Extra(UsdGenParamValue value);
    USDGEN_CORE_API UsdGenInputRecord Finish(UsdGenNodeKey key, UsdGenNodeKey owner,
                                             UsdGenText const &label, uint64_t generation);

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
    std::vector<UsdGenRef<const UsdGenArena>> _sources;   // UseCache: the arenas staged per slot
    std::vector<UsdGenParamValue> _values;
    UsdGenInputColumnCache *_cache = nullptr;
    UsdGenNodeKey _cacheKey;
};

/// usdGen:preview:* on the description: the value a viewport preview shows on
/// the published geometry (07 §7.3); how it is coloured is host data.
/// 'source' is the first authored target, resolved by the host to a prim and
/// an attribute: a UsdGenExpression prim, a UsdGenPtexMap prim (the builders
/// add it to 'maps') or an operator attribute ('attribute' non-empty). The
/// publish desc's previewValues hook resolves it against the compiled graph.
struct UsdGenPreviewSpec
{
    UsdGenNodeKey source;                         // invalid == no preview
    UsdGenToken  attribute;                           // the operator attribute; empty for a prim
    UsdGenToken  evaluation = UsdGenTokens().primitive;   // for an expression prim source
    uint8_t  mapChannels = 1;                     // a map source: 3 reads its first three channels

    bool Active() const { return source.IsValid(); }
};

/// The text a key stands for (the host's prim path), for diagnostics only.
/// A derived key's text is its parent's text, '/', then 'text'.
struct UsdGenNodeLabel
{
    UsdGenNodeKey key;
    UsdGenNodeKey parent;     // valid for a derived key
    UsdGenText    text;
    UsdGenToken   name;       // a derived key: its child name
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
    UsdGenPreviewSpec              preview;
    UsdGenMatrix4d                     xformMatrix{1.0}; // description world matrix (post-flattening, S4)
    float    defaultWidth = 0.01f;                 // usdGen:width:default
    int      tileTarget = 64;                      // uniform int usdGen:tileTarget
    double   time = 0.0;
    double   timeCodesPerSecond = 24.0; // USD default; expressions expose seconds
    /// kUsdGenBackendCpu, a registered backend (registries/backend.h), or
    /// Invalid: a present but unknown request fails closed, never the CPU.
    UsdGenBackendId executionBackend = kUsdGenBackendCpu;
    std::vector<std::string> validationErrors;
    /// Sorted by key: the text of every key above.
    std::vector<UsdGenNodeLabel> labels;
    /// Opaque host data (usdGenUsd: this spec's key -> path table and its
    /// presentation). The engine copies it into the generations and routing
    /// snapshots it makes from this spec and never reads it.
    std::shared_ptr<const void> host;
    /// The host's digest of what its host data adds to a cook's output (its
    /// presentation); the execution cache key mixes it.
    uint64_t hostDigest = 0;

    /// The run of 'kind' in 'inputs' (empty when absent).
    UsdGenSpan<const UsdGenInputRecord> Inputs(UsdGenTypeId kind) const noexcept;
    UsdGenSpan<UsdGenInputRecord> MutableInputs(UsdGenTypeId kind) noexcept;
    /// The record of 'kind' with 'key', else null (linear within the run).
    UsdGenInputRecord const *FindInput(UsdGenTypeId kind, UsdGenNodeKey key) const noexcept;
    /// Stable sort of 'inputs' by kind (builders append kinds in any order).
    void SortInputs();
    /// The text of 'key' (its prim path), else empty. Allocates: diagnostics only.
    std::string Label(UsdGenNodeKey key) const;
    /// The host label text of 'key' without allocating; empty for a
    /// derived or unknown key.
    UsdGenText const &LabelText(UsdGenNodeKey key) const noexcept;
    /// The child name of a derived key (DeriveKey); empty for a host key.
    UsdGenToken const &DerivedName(UsdGenNodeKey key) const noexcept;
    /// The description's path text (expr::DescriptionId hashes it).
    char const *DescriptionText() const noexcept { return LabelText(description).c_str(); }
    /// The key of a synthetic child of 'parent' (an engine-made node or
    /// input), keyed like the host keys the path <parent>/<child>, and its
    /// label. Compile phase: allocates.
    UsdGenNodeKey DeriveKey(UsdGenNodeKey parent, UsdGenToken const &child);
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
