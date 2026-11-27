// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Input kinds a host reads into input records (step 7). Declared in 7.2;
// registered and consumed from 7.3a, when typed input payloads become
// records.
#ifndef USDGEN_API_INPUT_DESC_H
#define USDGEN_API_INPUT_DESC_H

#include "usdGen/api/export.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/api/ids.h"
#include "usdGen/api/nodeKey.h"
#include "usdGen/api/span.h"
#include "usdGen/api/token.h"
#include "usdGen/api/epoch.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace usdGen {

struct UsdGenInputRecord;
struct UsdGenGraphSpec;
struct UsdGenResolvedMapValue;
enum class UsdGenLane : uint8_t;

enum UsdGenInputSlotFlag : uint8_t {
    kUsdGenInputRequired = 1,
    kUsdGenInputPrimvar = 2,   // read the <attribute> primvar, flattening indexed
    kUsdGenInputCounts = 4,    // integer counts that become Offsets(level)
};

/// One attribute a host reads into one channel of the input record.
struct UsdGenInputSlot {
    UsdGenToken attribute;     // the authored attribute name, e.g. "points"
    UsdGenSlot channel = kUsdGenNoSlot;  // kUsdGenNoSlot for kUsdGenInputCounts
    uint8_t level = 0;         // kUsdGenInputCounts: the level these counts define
    uint8_t flags = 0;
};

/// What the engine does with a kind's records beyond handing them to ops.
enum class UsdGenInputRole : uint8_t {
    None,
    Surface,   // node 'surfaces' targets: compiler-assigned ids, Default-space variants, routing
    Map,       // node 'maps' targets: resolved map values
};

/// Two-lane content digest an input kind writes one record into (the
/// execution cache key). Arrays mix their length now and their bytes in
/// parallel after the walk. Equality-only, never persisted.
class USDGEN_API UsdGenInputDigest {
public:
    virtual void Mix(int lane, uint64_t value) = 0;
    virtual void MixText(int lane, std::string_view text) = 0;
    /// 'count' elements of 'elementBytes' each.
    virtual void MixArray(int lane, void const *data, size_t count, size_t elementBytes) = 0;
    template <class T>
    void MixSpan(int lane, UsdGenSpan<const T> values) { MixArray(lane, values.data(), values.size(), sizeof(T)); }
protected:
    ~UsdGenInputDigest() = default;
};

/// Registered per input kind. A host reads 'schemaType' prims per 'slots',
/// time-sampled, into a UsdGenInputRecord; it never links the primitive.
/// Record columns are free-length (level 0, values = the array's length *
/// arity), so a malformed input reaches the compiler's checks unchanged.
struct UsdGenInputDesc {
    UsdGenTypeId kind = UsdGenTypeId::Invalid;   // registered type with the record layout
    UsdGenToken schemaType;                       // the prim schema type, e.g. "Mesh"
    UsdGenSpan<const UsdGenInputSlot> slots;
    UsdGenSpan<const UsdGenToken> attributes;     // values, in record.values order
    UsdGenToken dynamicPrimvarPrefix;             // authored planes; empty = none
    /// Optional engine-side derivation run at compile (a reference set). True with a null 'out' when the record has none; false sets 'error'.
    bool (*materialize)(UsdGenInputRecord const &, UsdGenRef<const UsdGenElementSet> *out,
                        std::string *error) = nullptr;
    /// Optional intake fix-up run when a spec enters the compiler (maps:
    /// their type handle). Compile phase.
    void (*intake)(UsdGenInputRecord &) = nullptr;

    UsdGenInputRole role = UsdGenInputRole::None;
    /// Diagnostics noun for a record ("surface"); null: "input".
    char const *displayName = nullptr;
    /// Compile-time checks of one record; false appends to 'errors'.
    bool (*validate)(UsdGenGraphSpec const &, UsdGenInputRecord const &,
                     std::vector<std::string> *errors) = nullptr;
    /// The record's content for the execution cache key; 'layout': only the
    /// terms that size an execution (counts), on lane 0.
    void (*digest)(UsdGenInputRecord const &, UsdGenInputDigest &, bool layout) = nullptr;
    /// Records with a lane are versioned as sources (Chunked) or references.
    bool (*lane)(UsdGenInputRecord const &, UsdGenLane *out) = nullptr;
    /// Channel names the record authors, in authored order (indexed-channel
    /// reservations). Valid while the record lives.
    UsdGenSpan<const UsdGenToken> (*authoredChannels)(UsdGenInputRecord const &) = nullptr;
    /// Surface role: the compiler-assigned id and its setter, the prim whose
    /// edits route to the record, and the Default-space variant (false: none)
    /// with the content key that decides when to rebuild it.
    uint32_t (*surfaceId)(UsdGenInputRecord const &) = nullptr;
    void (*setSurfaceId)(UsdGenInputRecord &, uint32_t) = nullptr;
    UsdGenNodeKey (*routingKey)(UsdGenInputRecord const &) = nullptr;
    bool (*restVariant)(UsdGenInputRecord const &posed, UsdGenInputRecord *out) = nullptr;
    void (*restKey)(UsdGenInputRecord const &, uint64_t key[3]) = nullptr;
    /// Map role: the compiler-resolved value (key, type, path, generation,
    /// identity), and the surface primvar a paint snapshot reads (false: none).
    void (*resolveMap)(UsdGenInputRecord const &, UsdGenResolvedMapValue *out) = nullptr;
    bool (*paintSource)(UsdGenInputRecord const &, UsdGenNodeKey *surface,
                        UsdGenToken *primvar) = nullptr;
};

/// The InputDesc registered for 'kind', else null.
USDGEN_API UsdGenInputDesc const *UsdGenFindInputDesc(UsdGenTypeId kind) noexcept;
/// The first registered kind with 'role', else Invalid. Compile phase.
USDGEN_API UsdGenTypeId UsdGenInputKindWithRole(UsdGenInputRole role) noexcept;

}  // namespace usdGen

#endif  // USDGEN_API_INPUT_DESC_H
