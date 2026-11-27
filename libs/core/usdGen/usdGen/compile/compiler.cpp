// usdGen engine — compiler implementation (03-execution-engine.md §3).
//
// Compile() walks UsdGenGraphSpec::nodes and compiler-owned dependency edges
// derived from the composed hierarchy by the descriptor builder; Kahn sort
// with namespace tie-break; cycle detection (compile error
// naming the offending pair); dense node ids in topological order;
// reference-lane ordering (§1.5); output/input plane
// slot binding (§1.2); Merkle structural digests (§3.3); tile arithmetic
// (R21); dirty routing table rebuild data (§5.1).
//
// Recompile() is the incremental path (gate E-6): every node's structural
// digest is recomputed; a node whose digest is unchanged (same type,
// inputs, topology-class params and relationship targets) keeps its op,
// capture and buffer from the previous graph. Exactly one appended node therefore rebuilds exactly one node.
#include "usdGen/compile/compiler.h"

#include "usdGen/api/engineTokens.h"
#include "usdGen/api/ext/opRegistry.h"
#include "usdGen/api/ext/tokenScope.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/backend.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/backendRegistry.h"
#include "usdGen/api/types.h"
#include "usdGen/api/trace.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string_view>
#include <limits>
#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
namespace usdGen {
namespace {

// Compile temporaries recycle through the engine pools, so a warm recompile
// makes no heap call.
template <class T>
using PVec = UsdGenPoolVector<T>;
template <class T>
using PSet = std::set<T, std::less<T>, UsdGenPoolAllocator<T>>;
template <class K, class V>
using PMap = std::map<K, V, std::less<K>, UsdGenPoolAllocator<std::pair<const K, V>>>;
template <class K, class V, class H>
using PHashMap = std::unordered_map<K, V, H, std::equal_to<K>,
                                    UsdGenPoolAllocator<std::pair<const K, V>>>;

int _PopCount64(uint64_t value) noexcept
{
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(value));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value) { value &= value - 1; ++count; }
    return count;
#endif
}

int _CountTrailingZeros64(uint64_t value) noexcept
{
    // All callers pass a nonzero bitmap word.
#if defined(_MSC_VER)
    unsigned long index = 0;
    _BitScanForward64(&index, value);
    return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int count = 0;
    while ((value & 1u) == 0) { value >>= 1; ++count; }
    return count;
#endif
}

/// Each input kind's own record checks (UsdGenInputDesc::validate), record order.
bool ValidateInputRecords(UsdGenGraphSpec const &desc, UsdGenCompileResult &result)
{
    bool ok = true;
    for (UsdGenInputRecord const &record : desc.inputs) {
        UsdGenInputDesc const *input = UsdGenFindInputDesc(record.kind);
        if (input && input->validate && !input->validate(desc, record, &result.errors)) ok = false;
    }
    return ok;
}

bool BindExtraPlaneSlots(UsdGenGraphSpec const &desc, UsdGenOpRecord const &record,
                         UsdGenCompiledNode *node, UsdGenCompileResult *result)
{
    auto bindNames = [&](UsdGenSpan<const UsdGenToken> names, std::vector<UsdGenToken> *slots,
                         char const *direction) {
        if (names.size() > kUsdGenMaxExtraPlaneSlots) {
            result->errors.push_back(std::string("UsdGenCompiler: operator '") +
                desc.Label(node->desc->key) + "' declares too many " +
                direction + " primvar slots");
            return false;
        }
        slots->clear();
        slots->reserve(names.size());
        for (UsdGenToken const &name : names) {
            // At most kUsdGenMaxExtraPlaneSlots names: a scan, no set.
            if (name.IsEmpty() ||
                std::find(slots->begin(), slots->end(), name) != slots->end()) {
                result->errors.push_back(std::string("UsdGenCompiler: operator '") +
                    desc.Label(node->desc->key) + "' declares an empty or duplicate " +
                    direction + " primvar slot");
                return false;
            }
            slots->push_back(name);
        }
        return true;
    };
    // The record's static planes.
    auto bind = [&](UsdGenSpan<const UsdGenChannelDesc> planes, std::vector<UsdGenToken> *slots,
                    char const *direction) {
        if (planes.size() > kUsdGenMaxExtraPlaneSlots) {
            result->errors.push_back(std::string("UsdGenCompiler: operator '") +
                desc.Label(node->desc->key) + "' declares too many " +
                direction + " primvar slots");
            return false;
        }
        UsdGenToken names[kUsdGenMaxExtraPlaneSlots];
        for (size_t i = 0; i < planes.size(); ++i) names[i] = planes[i].name;
        return bindNames(UsdGenSpan<const UsdGenToken>(names, planes.size()), slots, direction);
    };
    bool outputs = false;
    if (auto const emit = record.Def().hooks.emittedPlanes) {
        // Parameter-dependent names (4a): interned here, compile scope.
        struct Sink final : UsdGenPlaneSink {
            UsdGenToken names[kUsdGenMaxExtraPlaneSlots + 1];
            size_t count = 0;
            void Add(std::string const &name, UsdGenPlaneRate, UsdGenPlaneScalar,
                     uint8_t) override
            {
                if (count < kUsdGenMaxExtraPlaneSlots + 1) names[count] = UsdGenInternToken(name);
                ++count;
            }
        } sink;
        emit(node->paramView, node->indexedPlaneBase, sink);
        // One past the limit is enough for bindNames to reject.
        size_t const kept = std::min(sink.count, size_t(kUsdGenMaxExtraPlaneSlots) + 1);
        outputs = bindNames(UsdGenSpan<const UsdGenToken>(sink.names, kept), &node->outputPrimvars,
                            "output");
    } else {
        outputs = bind(record.OutputPlanes(), &node->outputPrimvars, "output");
    }
    return outputs && bind(record.InputPlanes(), &node->inputPrimvars, "input");
}

/// Same slot names in the same order, without allocating.
bool SameSlots(UsdGenSpan<const UsdGenChannelDesc> planes, std::vector<UsdGenToken> const &slots)
{
    if (planes.size() != slots.size()) return false;
    for (size_t i = 0; i < slots.size(); ++i)
        if (planes[i].name != slots[i]) return false;
    return true;
}
/// The planes BindExtraPlaneSlots would bind are the ones already bound.
/// emittedPlanes reads structural params and the indexed run only, so on a
/// value-only edit with an unchanged run its planes are the bound ones.
bool SamePlaneSlots(UsdGenOpRecord const &record, UsdGenCompiledNode const &node)
{
    bool const out = record.Def().hooks.emittedPlanes ||
        SameSlots(record.OutputPlanes(), node.outputPrimvars);
    return out && SameSlots(record.InputPlanes(), node.inputPrimvars);
}

bool ValidateExpressionBindings(UsdGenGraphSpec const &desc,
                                UsdGenCompileResult &result,
                                bool injectedDevice = false)
{
    bool ok = true;
    ok = ValidateInputRecords(desc, result) && ok;
    if (!desc.validationErrors.empty()) {
        result.errors.insert(result.errors.end(), desc.validationErrors.begin(), desc.validationErrors.end());
        ok = false;
    }
    UsdGenDiagnostics backendDiagnostics;
    if (!injectedDevice &&
        !ValidateUsdGenExecutionBackend(desc.executionBackend, &backendDiagnostics)) {
        result.errors.insert(result.errors.end(), backendDiagnostics.errors.begin(),
                             backendDiagnostics.errors.end());
        ok = false;
    }
    // A registered backend's compile-time admission of the graph.
    UsdGenBackendDef const *backendDef = ok
        ? UsdGenGetBackend(desc.executionBackend) : nullptr;
    if (backendDef && backendDef->validate) {
        UsdGenDiagnostics diagnostics;
        if (!backendDef->validate(desc, &diagnostics)) {
            result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
            ok = false;
        }
    }
    UsdGenParamEvaluatorDef const &evaluator = UsdGenOpRegistry::Get().ParamEvaluator();
    if (evaluator.validateInputs && !evaluator.validateInputs(desc, &result.errors)) ok = false;
    for (auto const &node : desc.nodes) {
        // Both lanes run an expression evaluator, so a connected parameter is
        // no longer backend-gated. What each op admits is its record's decls.
        if (!UsdGenValidateExpressionBindings(node, &desc, &result.errors)) ok = false;
        if (evaluator.validateBindings && !evaluator.validateBindings(desc, node, &result.errors))
            ok = false;
    }
    return ok;
}

// FNV-1a 64-bit.
uint64_t Fnv1a(uint64_t h, void const *data, size_t n)
{
    auto const *p = static_cast<unsigned char const *>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

uint64_t Fnv1aI64(uint64_t h, int64_t v)
{
    uint64_t u = static_cast<uint64_t>(v);
    return Fnv1a(h, &u, sizeof(u));
}

uint64_t FloatBits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
uint64_t Fnv1aCstr(uint64_t h, char const *s)
{
    for (auto const *p = reinterpret_cast<unsigned char const *>(s); *p; ++p) {
        h ^= *p; h *= 0x100000001b3ULL;
    }
    return h;
}
uint64_t Fnv1aText(uint64_t h, std::string_view s)
{
    for (char c : s) {   // as Fnv1aCstr: stops at a NUL
        if (!c) break;
        h ^= static_cast<unsigned char>(c); h *= 0x100000001b3ULL;
    }
    return h;
}
uint64_t Fnv1aToken(uint64_t h, UsdGenToken const &t)
{
    // GetText(): the interned c-string; GetString() would allocate a
    // std::string per call on the E-6 hot path.
    return Fnv1aI64(Fnv1aCstr(h, t.GetText()),
                    static_cast<int64_t>(t.Hash()));
}

// mapBindings is the canonical transport.  The old maps array remains a
// source-compatible input surface, but when both forms are present they must
// describe the same ordered target sequence.
template <class Bindings>
bool EffectiveMapBindings(UsdGenOpNode const &node, Bindings *out,
                          std::string *error)
{
    out->clear();
    if (node.mapBindings.empty()) {
        out->reserve(node.maps.size());
        for (UsdGenNodeKey const path : node.maps)
            out->push_back({path, UsdGenToken()});
        return true;
    }
    if (!node.maps.empty()) {
        if (node.maps.size() != node.mapBindings.size()) {
            *error = "legacy map paths and typed map bindings disagree";
            return false;
        }
        for (size_t i = 0; i < node.maps.size(); ++i) {
            if (node.maps[i] != node.mapBindings[i].map) {
                *error = "legacy map paths and typed map bindings disagree";
                return false;
            }
        }
    }
    out->assign(node.mapBindings.begin(), node.mapBindings.end());
    return true;
}

bool ValidateMapBinding(UsdGenMapBinding const &binding,
                        UsdGenOpRecord const &record, std::string *error)
{
    if (!binding.map.IsValid()) {
        *error = "binding has an empty map target";
        return false;
    }
    // Empty relationship is deliberately accepted for direct typed clients
    // and maps-only compatibility descriptors. `usdGen:regionMap` is the
    // categorical Ptex owner-map slot of one source op's surface cage; the
    // operator admission layer enforces that it is used only there.
    if (binding.relationship.IsEmpty()) return true;
    // The op's declared map slots (UsdGenOpDef::mapSlots); the operators
    // enforce their own map types.
    UsdGenSpan<const UsdGenToken> const slots = record.MapRelationships();
    if (std::find(slots.begin(), slots.end(), binding.relationship) == slots.end()) {
        *error = "relationship '" + binding.relationship.GetString() +
            "' is not a map relationship";
        return false;
    }
    return true;
}

/// 128-bit digest = two independent FNV lanes over the term list
/// (ADR §4.2.1). `enabled` and `seed` are NOT digest terms; only the
/// topology-class parameter values (02 §6) enter, together with type, mode,
/// sorted input paths, sorted reference/map/surface relationship targets,
/// and the children's digests.
UsdGenEpoch ComputeNodeDigest(
    UsdGenOpNode const &nd,
    uint64_t typeSeed,
    UsdGenOpRecord const *record,
    UsdGenSpan<const UsdGenNodeKey> inputPaths,
    UsdGenSpan<const UsdGenNodeKey> refPaths,
    UsdGenSpan<const UsdGenMapBinding> mapBindings,
    UsdGenSpan<const std::pair<UsdGenNodeKey, UsdGenEpoch>> childDigests)
{
    uint64_t h0 = 1469598103934665603ULL;
    uint64_t h1 = 1469598103934665603ULL;
    auto mix = [&h0, &h1](uint64_t v) {
        h0 = Fnv1aI64(h0, static_cast<int64_t>(v));
        h1 = Fnv1aI64(h1, static_cast<int64_t>(~v));
    };

    mix(0xa5a5a5a5ULL ^ typeSeed);
    mix(0x00020002ULL ^ Fnv1aToken(0, nd.mode));
    for (UsdGenNodeKey const p : inputPaths) mix(0x00050005ULL ^ p.value);
    for (UsdGenNodeKey const p : refPaths)   mix(0x00060006ULL ^ p.value);
    for (UsdGenMapBinding const &binding : mapBindings) {
        mix(0x00060007ULL ^ binding.map.value);
        mix(0x00060009ULL ^ Fnv1aToken(0, binding.relationship));
    }
    // Digest terms: authored params named exactly like a Structural slot.
    UsdGenSpan<const UsdGenParamSlot> const digestSlots =
        record ? record->DigestSlots() : UsdGenSpan<const UsdGenParamSlot>();
    for (auto const &param : nd.params) {
        for (UsdGenParamSlot const s : digestSlots) {
            UsdGenToken const &t = record->ParamName(s);
            if (param.name == t) {
                mix(0x00070007ULL ^ Fnv1aToken(0, t));
                mix(0x00080008ULL ^ param.value.GetHash());
                break;
            }
        }
    }
    for (UsdGenNodeKey const p : inputPaths) {
        for (auto const &kv : childDigests)
            if (kv.first == p) { mix(kv.second[0] ^ (kv.second[1] << 1)); break; }
    }
    return {h0, h1};
}

/// FNV-1a of a node's type name: the digest's type term.
uint64_t TypeSeed(UsdGenToken const &type) { return Fnv1aCstr(0, type.GetText()); }
/// The graph-level digest's type term: FNV-1a of the pseudo type
/// UsdGenTerminal, a constant so the compiler names no type.
constexpr uint64_t kTerminalTypeSeed = 0x48b05df1120bb972ULL;

/// The slot declared with exactly this spelling, or kUsdGenNoParam.
UsdGenParamSlot ExactSlot(UsdGenOpRecord const &record, UsdGenToken const &name)
{
    UsdGenParamSlot const s = record.FindParam(name);
    return s != kUsdGenNoParam && record.ParamName(s) == name ? s : kUsdGenNoParam;
}

/// Value-class digest (skip-signature term, 03 §3.2): enabled, then every
/// authored param except routed non-Value decls and an undeclared seed.
/// Unknown params feed it: value-dirty is the safe superset.
uint64_t ParamValueDigest(UsdGenOpNode const &nd, UsdGenOpRecord const &record)
{
    UsdGenToken const &seed = UsdGenTokens().seed;
    uint64_t vh = 1469598103934665603ULL;
    auto feed = [&vh](uint64_t v) { vh ^= v; vh *= 0x100000001b3ULL; };
    feed(static_cast<uint64_t>(nd.enabled));
    for (auto const &param : nd.params) {
        UsdGenParamSlot const s = ExactSlot(record, param.name);
        bool inValue = false, inTopo = false;
        if (s != kUsdGenNoParam && record.Def().params[s].routed) {
            inValue = record.ParamClass(s) == UsdGenParamClass::Value;
            inTopo = !inValue;
        }
        if (inValue || (!inTopo && param.name != seed)) feed(param.value.GetHash());
    }
    return vh;
}

/// The decimal suffix of `name` after `prefix` (no sign, no leading zero).
bool IndexedSuffix(std::string const &name, char const *prefix, int *index)
{
    size_t const len = std::strlen(prefix);
    if (name.compare(0, len, prefix) != 0) return false;
    if (name.size() == len || (name.size() > len + 1 && name[len] == '0')) return false;
    int value = 0;
    for (size_t i = len; i < name.size(); ++i) {
        char const digit = name[i];
        if (digit < '0' || digit > '9' ||
            value > (std::numeric_limits<int>::max() - (digit - '0')) / 10)
            return false;
        value = value * 10 + (digit - '0');
    }
    *index = value;
    return true;
}

/// Indexed plane runs (4a) for nodes whose op has hooks.indexedPlanes, one
/// namespace per UsdGenOpDef::indexedPlaneFamily, in compiled topological
/// order. Authored input channels <family><n> reserve n (imports); a
/// pinned run takes its indices, an auto run the first free consecutive
/// ones. Disabled nodes reserve too, so a toggle renumbers nothing.
bool AllocateIndexedPlanes(UsdGenGraphSpec const &desc, PVec<int> const &order,
                           PVec<UsdGenOpRecord const *> const &descRecord,
                           PVec<int> *bases, PVec<int> *counts,
                           UsdGenCompileResult &result)
{
    auto familyOf = [&](int di) -> char const * {
        UsdGenOpRecord const *record = descRecord[size_t(di)];
        if (!record || !record->Def().hooks.indexedPlanes) return nullptr;
        char const *family = record->Def().indexedPlaneFamily;
        return family ? family : "";
    };
    for (size_t first = 0; first < order.size(); ++first) {
        char const *const family = familyOf(order[first]);
        if (!family) continue;
        bool seen = false;   // families are few: rescan instead of a set
        for (size_t k = 0; !seen && k < first; ++k) {
            char const *const other = familyOf(order[k]);
            seen = other && std::strcmp(other, family) == 0;
        }
        if (seen) continue;
        PMap<int, UsdGenNodeKey> owner;
        if (*family) {
            for (UsdGenInputRecord const &record : desc.inputs) {
                UsdGenInputDesc const *input = UsdGenFindInputDesc(record.kind);
                if (!input || !input->authoredChannels) continue;
                for (UsdGenToken const &name : input->authoredChannels(record)) {
                    int index = 0;
                    if (IndexedSuffix(name.GetString(), family, &index))
                        owner.emplace(index, record.key);
                }
            }
        }
        int nextAuto = 0;
        for (size_t k = first; k < order.size(); ++k) {
            int const di = order[k];
            char const *const f = familyOf(di);
            if (!f || std::strcmp(f, family) != 0) continue;
            UsdGenOpNode const &nd = desc.nodes[size_t(di)];
            UsdGenOpRecord const &record = *descRecord[size_t(di)];
            UsdGenIndexedPlaneRequest request;
            {
                UsdGenStackParams const params(desc, nd, record);
                request = record.Def().hooks.indexedPlanes(params.View());
            }
            if (request.count == 0) continue;
            int const count = request.count, pinned = request.pinnedBase;
            char const *const display = record.Def().displayName ? record.Def().displayName
                                                                 : record.Name().GetText();
            if (count < 0 || pinned < -1 ||
                (pinned >= 0 && pinned > std::numeric_limits<int>::max() - count) ||
                nextAuto > std::numeric_limits<int>::max() - count) {
                result.errors.push_back(std::string("UsdGenCompiler: invalid ") + display +
                                        " level range on '" + desc.Label(nd.key) + "'");
                return false;
            }
            int base = pinned >= 0 ? pinned : nextAuto;
            if (pinned < 0) {
                // First free consecutive run; runs stay contiguous around imports.
                for (;;) {
                    if (base > std::numeric_limits<int>::max() - count) {
                        result.errors.push_back(std::string("UsdGenCompiler: no free ") +
                            display + " level range on '" + desc.Label(nd.key) + "'");
                        return false;
                    }
                    bool occupied = false;
                    for (int offset = 0; offset < count; ++offset)
                        occupied = occupied || owner.count(base + offset) != 0;
                    if (!occupied) break;
                    ++base;
                }
            }
            (*bases)[size_t(di)] = base;
            (*counts)[size_t(di)] = count;
            nextAuto = std::max(nextAuto, base + count);
            for (int offset = 0; offset < count; ++offset) {
                int const index = base + offset;
                auto const inserted = owner.emplace(index, nd.key);
                if (!inserted.second) {
                    result.errors.push_back("UsdGenCompiler: " + std::string(family) +
                        std::to_string(index) + " is emitted by both '" +
                        desc.Label(inserted.first->second) + "' and '" +
                        desc.Label(nd.key) + "'");
                    return false;
                }
            }
        }
    }
    return true;
}

/// Points the node's view at its desc entry and re-resolves its slots;
/// every compile path calls it, reused nodes included.
void BindParams(UsdGenCompiledNode &node, UsdGenGraphSpec const *desc,
                UsdGenOpNode const *nd, UsdGenOpRecord const *record,
                int indexedBase, int indexedCount)
{
    node.opId = record ? record->Id() : UsdGenOpId::Invalid;
    node.record = record;
    node.paramView.desc = desc;
    node.paramView.node = nd;
    node.paramView.op = record;
    if (!node.expressions) node.expressions = UsdGenCreateParamEvaluatorState();
    node.paramView.expressions = node.expressions.get();
    if (record) {
        node.params.resize(record->ParamCount());
        UsdGenResolveParams(*record, *nd, node.params.data());
    } else {
        node.params.clear();
    }
    node.paramView.slots = record ? node.params.data() : nullptr;
    node.indexedPlaneBase = indexedBase;
    node.indexedPlaneCount = static_cast<uint8_t>(indexedCount);
    auto const emitsUv = record ? record->Def().hooks.emitsLimitSurfaceUv : nullptr;
    node.limitSurfaceUv = emitsUv && emitsUv(node.paramView);
    auto const remaps = record ? record->Def().hooks.remapsVertexPlanes : nullptr;
    node.remapsVertexPlanes = record &&
        (record->Has(kUsdGenOpRemapsVertexPlanes) || (remaps && remaps(node.paramView)));
}

/// True when `b` differs from `a` only in the values of routed Value-class
/// parameters of `record`. Such an edit keeps the compiled node: its
/// structure, slots and capture inputs are unchanged. Expression-bound
/// nodes never qualify (their program identity is not digested yet).
bool ValueOnlyEdit(UsdGenOpNode const &a, UsdGenOpNode const &b,
                   UsdGenOpRecord const &record)
{
    if (!a.expressionBindings.empty() || !b.expressionBindings.empty()) return false;
    if (a.key != b.key || a.type != b.type || a.mode != b.mode ||
        a.enabled != b.enabled || a.seed != b.seed || a.inputs != b.inputs ||
        a.references != b.references || a.inputSets != b.inputSets ||
        a.surfaces != b.surfaces || a.maps != b.maps ||
        a.mapBindings.size() != b.mapBindings.size() ||
        a.params.size() != b.params.size() || a.ramps.size() != b.ramps.size())
        return false;
    for (size_t i = 0; i < a.mapBindings.size(); ++i) {
        if (a.mapBindings[i].map != b.mapBindings[i].map ||
            a.mapBindings[i].relationship != b.mapBindings[i].relationship)
            return false;
    }
    for (size_t i = 0; i < a.ramps.size(); ++i) {
        if (a.ramps[i].knots != b.ramps[i].knots ||
            a.ramps[i].positions != b.ramps[i].positions ||
            a.ramps[i].colors != b.ramps[i].colors ||
            a.ramps[i].interpolation != b.ramps[i].interpolation)
            return false;
    }
    bool changed = false;
    for (size_t i = 0; i < a.params.size(); ++i) {
        UsdGenParamValue const &pa = a.params[i], &pb = b.params[i];
        if (pa.name != pb.name || pa.animated != pb.animated) return false;
        if (pa.value == pb.value) continue;
        UsdGenParamSlot const s = ExactSlot(record, pa.name);
        if (s == kUsdGenNoParam || !record.Def().params[s].routed ||
            record.ParamClass(s) != UsdGenParamClass::Value)
            return false;
        changed = true;
    }
    return changed;
}

}  // namespace

// ---------------------------------------------------------------------------

UsdGenCompileResult UsdGenCompiler::Compile(UsdGenGraphSpec const &desc, UsdGenGraph *out)
{
    USDGEN_TRACE_FUNCTION();
    UsdGenCompileResult result;
    if (!ValidateExpressionBindings(desc, result)) return result;
    UsdGenGraph candidate;
    _Build(desc, &candidate, /*reuse=*/nullptr, result);
    if (result.errors.empty()) {
        *out = std::move(candidate);
        out->_inputVersions =
            UsdGenExecutionInputVersions::FromDescription(out->Spec());
        result.ok = true;
        // Fresh compile: every node re-captures; all chunks value-dirty so
        // the first commit evaluates the whole chain.
        for (auto &np : out->_nodes) {
            np->captureNeeded = true;
            std::fill(np->chunkDirty.begin(), np->chunkDirty.end(),
                      UsdGenDirtyParameter);
        }
    }
    return result;
}

UsdGenCompileResult UsdGenCompiler::CompileWithDevicePlan(
    UsdGenGraphSpec const &desc, UsdGenGraph *out, DevicePlanCompiler const &compile,
    std::shared_ptr<const UsdGenExecutionPlanHandle> *plan)
{
    USDGEN_TRACE_FUNCTION();
    UsdGenCompileResult result;
    if (!out || !plan || !compile) {
        result.errors.push_back("device plan compilation requires a provider and valid outputs");
        return result;
    }
    if (!ValidateExpressionBindings(desc, result)) return result;
    UsdGenGraph candidate;
    _Build(desc, &candidate, /*reuse=*/nullptr, result);
    std::shared_ptr<const UsdGenExecutionPlanHandle> nativePlan;
    if (result.errors.empty()) {
        UsdGenDiagnostics diagnostics;
        nativePlan = compile(desc, &diagnostics);
        result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
        if (!nativePlan && result.errors.empty())
            result.errors.push_back("device plan compilation failed");
    }
    if (result.errors.empty()) {
        *out = std::move(candidate);
        out->_inputVersions =
            UsdGenExecutionInputVersions::FromDescription(out->Spec());
        result.ok = true;
        for (auto &np : out->_nodes) {
            np->captureNeeded = true;
            std::fill(np->chunkDirty.begin(), np->chunkDirty.end(),
                      UsdGenDirtyParameter);
        }
        *plan = std::move(nativePlan);
    }
    return result;
}

UsdGenCompileResult UsdGenCompiler::CompileInjectedDevice(
    UsdGenGraphSpec const& desc, UsdGenGraph* out,
    DevicePlanCompiler const& compile,
    std::shared_ptr<const UsdGenExecutionPlanHandle>* plan)
{
    UsdGenCompileResult result;
    if (!out || !plan || !compile ||
        desc.executionBackend == kUsdGenBackendCpu ||
        desc.executionBackend == UsdGenBackendId::Invalid) {
        result.errors.push_back("injected device compilation requires a device provider and valid outputs");
        return result;
    }
    // Injection bypasses only the process-wide availability lookup. All
    // authoring, space, expression, factory and graph checks remain in force.
    if (!ValidateExpressionBindings(desc, result, true)) return result;
    UsdGenDiagnostics diagnostics;
    auto nativePlan = compile(desc, &diagnostics);
    result.errors.insert(result.errors.end(), diagnostics.errors.begin(), diagnostics.errors.end());
    result.warnings.insert(result.warnings.end(), diagnostics.warnings.begin(), diagnostics.warnings.end());
    if (!nativePlan || nativePlan->Backend() != desc.executionBackend ||
        !nativePlan->Metadata() || !nativePlan->Payload()) {
        result.errors.push_back("injected device provider returned no matching immutable plan");
    }
    if (!result.errors.empty()) return result;
    UsdGenGraph candidate;
    _Build(desc, &candidate, nullptr, result);
    if (!result.errors.empty()) return result;
    candidate._inputVersions = UsdGenExecutionInputVersions::FromDescription(candidate.Spec());
    for (auto& node : candidate._nodes) {
        node->captureNeeded = true;
        std::fill(node->chunkDirty.begin(), node->chunkDirty.end(), UsdGenDirtyParameter);
    }
    *out = std::move(candidate);
    *plan = std::move(nativePlan);
    result.ok = true;
    return result;
}

UsdGenCompileResult UsdGenCompiler::Recompile(UsdGenGraphSpec const &newDesc, UsdGenGraph *out)
{
    // A plan that lives with the graph is immutable per descriptor. Compile
    // transactionally so a failed edit cannot invalidate the last usable
    // program or generation.
    if (UsdGenBackendDef const *def =
            UsdGenGetBackend(newDesc.executionBackend))
        if (def->flags & kUsdGenBackendReusesPlan)
            return Compile(newDesc, out);
    UsdGenCompileResult result;
    if (!ValidateExpressionBindings(newDesc, result)) return result;
    _Build(newDesc, out, out, result);
    if (result.errors.empty()) {
        result.ok = true;
        result.graphIntact = true;
        UsdGenExecutionInputVersions::FromDescription(out->Spec(), &out->_inputVersions);
        // Rebuilt nodes re-capture; digest-stable nodes keep their runtime
        // state (capture, buffer, chunk + dirty bytes) moved over by
        // _Build. The runtime partition is re-established by the first
        // commit once a generator installs its topology — a recompile
        // cannot know the live element count of an unpopulated output, so it
        // must not call Repartition() here (that would wipe the moved
        // chunk/dirty state and capture flags).
        for (auto &np : out->_nodes) np->captureNeeded = false;
        for (UsdGenNodeId id : result.rebuilt) out->Node(id).captureNeeded = true;
        // A kept value-edited node evaluates again, and so does everything
        // downstream of it. Its capture stays unless the op cannot prove
        // (CaptureDigestComplete) that the digest sees every parameter.
        for (UsdGenNodeId id : result.valueEdited) {
            UsdGenCompiledNode &node = out->Node(id);
            if (!node.record->Has(kUsdGenOpCaptureDigestComplete)) node.captureNeeded = true;
            out->MarkNode(id, UsdGenDirtyParameter);
        }
    }
    return result;
}

void UsdGenCompiler::_Build(
    UsdGenGraphSpec const &desc,
    UsdGenGraph *out,
    UsdGenGraph const *reuse,
    UsdGenCompileResult &result)
{
    // The first compile freezes every registry (idempotent).
    UsdGenFreezeRegistries();
    UsdGenCompileTokenScope tokenScope;
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();

    if (desc.nodes.empty()) {
        result.errors.push_back("UsdGenCompiler: graph has no operator nodes");
        return;
    }
    if (!desc.terminal.IsValid()) {
        result.errors.push_back("UsdGenCompiler: graph has no terminal operator");
        return;
    }

    // desc index by path (namespace order == desc.nodes' order, S26).
    // Sorted vector, not unordered_map (E-6): one allocation + sort instead
    // of ~200 per-insert map-node mallocs; binary search per lookup.
    // Ordered by key value (a lookup table; the order means nothing).
    PVec<std::pair<UsdGenNodeKey, int>> descIdxByPath;
    descIdxByPath.reserve(desc.nodes.size());
    for (size_t i = 0; i < desc.nodes.size(); ++i) {
        descIdxByPath.emplace_back(desc.nodes[i].key, static_cast<int>(i));
    }
    std::sort(descIdxByPath.begin(), descIdxByPath.end(),
              [](auto const &a, auto const &b) { return a.first < b.first; });
    auto findDescIdx = [&](UsdGenNodeKey p) {
        auto it = std::lower_bound(
            descIdxByPath.begin(), descIdxByPath.end(), p,
            [](std::pair<UsdGenNodeKey, int> const &e, UsdGenNodeKey v) { return e.first < v; });
        return (it != descIdxByPath.end() && !(p < it->first)) ? it
            : descIdxByPath.end();
    };
    auto termIt = findDescIdx(desc.terminal);
    if (termIt == descIdxByPath.end()) {
        result.errors.push_back(
            std::string("UsdGenCompiler: terminal '") +
            desc.Label(desc.terminal) + "' is not an operator prim in the graph");
        return;
    }

    int const n = static_cast<int>(desc.nodes.size());
    // Edge resolution, fused single pass (E-6): every input's desc index is
    // looked up ONCE into flat arrays shared by the counters, the CSR fill
    // and Kahn — not re-searched per pass.
    PVec<int> edgeOff(size_t(n) + 1, 0);
    for (int i = 0; i < n; ++i)
        edgeOff[size_t(i) + 1] = edgeOff[size_t(i)] + int(desc.nodes[i].inputs.size());
    PVec<int> edgeIds;
    edgeIds.assign(size_t(edgeOff[size_t(n)]), -1);
    PVec<int> inDeg(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int k = edgeOff[size_t(i)]; k < edgeOff[size_t(i) + 1]; ++k) {
            UsdGenNodeKey const in = desc.nodes[i].inputs[size_t(k - edgeOff[size_t(i)])];
            auto it = findDescIdx(in);
            if (it == descIdxByPath.end()) {
                result.errors.push_back(
                    std::string("UsdGenCompiler: usdGen:input target '") + desc.Label(in) +
                    "' of '" + desc.Label(desc.nodes[i].key) + "' is not in the graph");
                return;
            }
            edgeIds[size_t(k)] = it->second;
            ++inDeg[i];
        }
    }
    // Consumers in CSR form (E-6): one allocation instead of n per-node
    // vectors. consumersOf[i] = [begin,end) into consumerIds.
    PVec<int> consumersOf(size_t(n) + 1, 0);
    for (int id : edgeIds) ++consumersOf[size_t(id) + 1];
    for (int i = 0; i < n; ++i) consumersOf[size_t(i) + 1] += consumersOf[size_t(i)];
    PVec<int> consumerIds;
    consumerIds.resize(size_t(consumersOf[size_t(n)]));
    {
        PVec<int> cursor(consumersOf.begin(), consumersOf.begin() + n);
        for (int i = 0; i < n; ++i)
            for (int k = edgeOff[size_t(i)]; k < edgeOff[size_t(i) + 1]; ++k)
                consumerIds[size_t(cursor[edgeIds[size_t(k)]]++)] = i;
    }

    // Kahn with namespace-order tie-break: among the ready nodes, always
    // pick the lowest desc index (02 §3.2 dense ids). Min-heap, not std::set:
    // identical order, zero per-node allocations (E-6: 200 tree-node mallocs).
    PVec<int> order;
    order.reserve(n);
    {
        PVec<int> ready;
        ready.reserve(n);
        for (int i = 0; i < n; ++i) if (inDeg[i] == 0) ready.push_back(i);
        std::make_heap(ready.begin(), ready.end(), std::greater<int>());
        while (!ready.empty()) {
            std::pop_heap(ready.begin(), ready.end(), std::greater<int>());
            int const i = ready.back();
            ready.pop_back();
            order.push_back(i);
            for (int k = consumersOf[size_t(i)]; k < consumersOf[size_t(i) + 1]; ++k) {
                int const j = consumerIds[size_t(k)];
                if (--inDeg[j] == 0) {
                    ready.push_back(j);
                    std::push_heap(ready.begin(), ready.end(), std::greater<int>());
                }
            }
        }
    }
    if (static_cast<int>(order.size()) != n) {
        for (int i = 0; i < n; ++i) {
            if (inDeg[i] > 0) {
                result.errors.push_back(
                    std::string("UsdGenCompiler: input cycle involving '") +
                    desc.Label(desc.nodes[i].key) + "'");
                break;
            }
        }
        return;
    }
    // Dense node ids: topo position == id.
    PVec<int> topoOfDesc(n, -1);
    for (int pos = 0; pos < n; ++pos) topoOfDesc[order[pos]] = pos;
    // Op record per desc entry (null: unregistered, rejected below).
    PVec<UsdGenOpRecord const *> descRecord(size_t(n), nullptr);
    for (int i = 0; i < n; ++i) {
        UsdGenOpId const id = ops.Find(desc.nodes[size_t(i)].type);
        if (id != UsdGenOpId::Invalid) descRecord[size_t(i)] = &ops.Record(id);
    }
    PVec<int> indexedBases(size_t(n), -1);
    PVec<int> indexedCounts(size_t(n), 0);
    if (!AllocateIndexedPlanes(desc, order, descRecord, &indexedBases, &indexedCounts, result))
        return;

    UsdGenNodeId const terminalId =
        static_cast<UsdGenNodeId>(topoOfDesc[termIt->second]);

    // Strict descendants as dense-id bitmaps propagated in REVERSE
    // topological order — O(edges * words) word-ORs. This replaces a
    // per-node std::set BFS whose O(n^2) allocations were the dominant
    // term of the E-6 recompile budget on a 200-node chain.
    // M-4: the DIRECT consumer is a descendant too.
    // Rows are written straight into each node's own descendants vector
    // (fillDescendants below), so a recompile reuses its capacity.
    int const words = (n + 63) / 64;
    PVec<uint64_t> bits(size_t(n) * words, 0);
    // Counts per row first, so every row allocates at most once.
    PVec<int> rowCount(size_t(n), 0);
    {
        for (int pos = n - 1; pos >= 0; --pos) {
            int const di = order[pos];
            uint64_t *dst = bits.data() + size_t(pos) * words;
            for (int k = consumersOf[size_t(di)]; k < consumersOf[size_t(di) + 1]; ++k) {
                int const c = consumerIds[size_t(k)];
                int const cp = topoOfDesc[c];   // cp > pos (topological)
                uint64_t const *src = bits.data() + size_t(cp) * words;
                // OR + popcount in one row pass (E-6): the count feeds the
                // exact reserve below, so rows never regrow.
                int add = 0;
                for (int w = 0; w < words; ++w) {
                    uint64_t const before = dst[w];
                    uint64_t const after = before | src[w];
                    dst[w] = after;
                    add += _PopCount64(after & ~before);
                }
                uint64_t const bit = 1ull << (cp & 63);
                if (!(dst[cp >> 6] & bit)) { dst[cp >> 6] |= bit; ++add; }
                rowCount[size_t(pos)] += add;
            }
        }
    }
    auto fillDescendants = [&](int pos, std::vector<UsdGenNodeId> *v) {
        uint64_t const *row = bits.data() + size_t(pos) * words;
        v->clear();
        v->reserve(size_t(rowCount[size_t(pos)]));
        for (int w = 0; w < words; ++w) {
            uint64_t b = row[w];
            while (b) {
                v->push_back(static_cast<UsdGenNodeId>(
                    w * 64 + _CountTrailingZeros64(b)));
                b &= b - 1;   // ascending bit order == dense-id order
            }
        }
    };
    // Reuse matching (E-6): index the previous graph's nodes by path WITHOUT
    // copying anything (no Clone, no buffer/vector copies). Stable nodes are
    // moved whole — op, capture, buffer, chunks, dirty bytes, epochs — into
    // the new graph; only their desc pointers and digests refresh. This
    // replaces the OldNode snapshot whose per-node Clone + deep copies were
    // the dominant term of the chain-200 append budget.
    // Validate factories and their geometry dataflow contracts before moving
    // any retained runtime state. A failed incremental compile must leave the
    // old graph intact, just like Compile. In particular, no executor may
    // silently select the first edge of an unsupported fan-in.
    struct StaticOperatorContract {
        size_t geometryInputArity = 0;
        size_t referenceInputArity = 0;
        UsdGenLane lane = UsdGenLane::Chunked;
    };
    PVec<StaticOperatorContract> operatorContracts(desc.nodes.size());
    for (size_t nodeIndex = 0; nodeIndex != desc.nodes.size(); ++nodeIndex) {
        UsdGenOpNode const& node = desc.nodes[nodeIndex];
        UsdGenOpRecord const *const record = descRecord[nodeIndex];
        if (!record || !record->Def().create) {
            result.errors.push_back("UsdGenCompiler: no kernel registered for '" +
                node.type.GetString() + "' (prim " + desc.Label(node.key) + ")");
            return;
        }
        StaticOperatorContract& contract = operatorContracts[nodeIndex];
        contract.geometryInputArity = record->Def().geometryInputArity;
        contract.referenceInputArity = record->ReferenceInputs().size();
        contract.lane = record->Def().lane;
        if (node.inputs.size() != contract.geometryInputArity) {
            std::ostringstream message;
            message << "UsdGenCompiler: operator '" << desc.Label(node.key)
                    << "' (type '" << node.type.GetText() << "') requires exactly "
                    << contract.geometryInputArity << " geometry input"
                    << (contract.geometryInputArity == 1 ? "" : "s") << "; found "
                    << node.inputs.size();
            result.errors.push_back(message.str());
            return;
        }
        // WidthBlend is the first admitted binary value fan-in and therefore
        // has a deliberately closed descriptor ABI.  Validate it before any
        // incremental state moves; disabled operators bypass runtime Bind(),
        // so this contract cannot live only in the operator implementation.
        if (auto const validate = record->Def().hooks.validateNode) {
            UsdGenStackParams const params(desc, node, *record);
            std::string error;
            if (!validate(desc, node, params.View(), &error)) {
                result.errors.push_back(std::move(error));
                return;
            }
        }
    }
    // Typed ports: every geometry edge carries a type its consumer accepts
    // (undeclared: the default primitive), or a type registered as converting
    // one-to-one to the default into a consumer that takes the default.
    // Declared types must be registered.
    {
        UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
        auto typeText = [](UsdGenToken const &t) {
            return t.IsEmpty() ? std::string("the default primitive") : t.GetString();
        };
        for (int i = 0; i < n; ++i) {
            UsdGenOpRecord const &consumer = *descRecord[size_t(i)];
            UsdGenToken const &declared = consumer.OutputType();
            if (!declared.IsEmpty() && types.Find(declared) == UsdGenTypeId::Invalid) {
                result.errors.push_back("UsdGenCompiler: operator '" + desc.Label(desc.nodes[i].key) +
                    "' (type '" + desc.nodes[i].type.GetString() + "') produces unregistered type '" +
                    declared.GetString() + "'");
                return;
            }
            for (int k = edgeOff[size_t(i)]; k < edgeOff[size_t(i) + 1]; ++k) {
                int const producer = edgeIds[size_t(k)];
                UsdGenToken const &produced = descRecord[size_t(producer)]->OutputType();
                if (consumer.AcceptsInput(produced)) continue;
                if (!produced.IsEmpty() && consumer.AcceptsInput(UsdGenToken()) &&
                    types.Has(types.Find(produced), kUsdGenTypeConvertsToDefault))
                    continue;
                std::string accepted;
                for (UsdGenToken const &t : consumer.InputTypes())
                    accepted += (accepted.empty() ? "" : " or ") + typeText(t);
                if (accepted.empty()) accepted = typeText(UsdGenToken());
                result.errors.push_back("UsdGenCompiler: operator '" + desc.Label(desc.nodes[i].key) +
                    "' (type '" + desc.nodes[i].type.GetString() + "') cannot take input '" +
                    desc.Label(desc.nodes[size_t(producer)].key) + "': it produces " + typeText(produced) +
                    "; this input accepts " + accepted);
                return;
            }
        }
    }

    // Fully validate and materialize external descriptor values BEFORE an
    // incremental build moves the old graph's nodes/descriptor.  A malformed
    // reference/map edit must leave the prior graph usable.
    std::vector<UsdGenResolvedReferenceValue> resolvedReferenceValues;
    std::vector<UsdGenResolvedMapValue> resolvedMapValues;
    PHashMap<UsdGenNodeKey, uint32_t, UsdGenNodeKeyHash> referenceByPath;
    PHashMap<UsdGenNodeKey, uint32_t, UsdGenNodeKeyHash> mapByPath;
    // Inputs whose kind materializes a reference set, through the kind's
    // InputDesc hook.
    UsdGenTypeRegistry const &inputTypes = UsdGenTypeRegistry::Get();
    for (UsdGenInputRecord const &record : desc.inputs) {
        if (record.kind == UsdGenTypeId::Invalid) continue;
        UsdGenInputDesc const *input = inputTypes.Def(record.kind).input;
        if (!input || !input->materialize) continue;
        if (referenceByPath.count(record.key)) {
            UsdGenRef<const UsdGenElementSet> probe;
            std::string ignored;
            if (input->materialize(record, &probe, &ignored) && !probe) continue;
            result.errors.push_back(std::string("UsdGenCompiler: duplicate reference ") +
                                    (input->displayName ? input->displayName : "input") + " '" +
                                    desc.Label(record.key) + "'");
            return;
        }
        UsdGenRef<const UsdGenElementSet> set;
        std::string error;
        if (!input->materialize(record, &set, &error)) {
            result.errors.push_back("UsdGenCompiler: " + error);
            return;
        }
        if (!set) continue;   // nothing to materialize for this record
        auto shared = std::make_shared<UsdGenReferenceSet>();
        shared->generation = record.generation;
        shared->set = std::move(set);
        UsdGenResolvedReferenceValue value;
        value.key = record.key;
        value.generation = record.generation;
        value.identity = Fnv1aI64(Fnv1aI64(1469598103934665603ULL,
                                           static_cast<int64_t>(record.key.value)),
                                  static_cast<int64_t>(record.generation));
        value.value = std::move(shared);
        referenceByPath.emplace(record.key,
                                static_cast<uint32_t>(resolvedReferenceValues.size()));
        resolvedReferenceValues.push_back(std::move(value));
    }
    UsdGenTypeId const mapKind = UsdGenInputKindWithRole(UsdGenInputRole::Map);
    UsdGenInputDesc const *const mapInput = UsdGenFindInputDesc(mapKind);
    UsdGenSpan<const UsdGenInputRecord> const mapRecords =
        mapInput ? desc.Inputs(mapKind) : UsdGenSpan<const UsdGenInputRecord>();
    for (UsdGenInputRecord const &mapRecord : mapRecords) {
        if (!mapRecord.key.IsValid() || mapByPath.count(mapRecord.key)) {
            result.errors.push_back("UsdGenCompiler: invalid or duplicate map descriptor '" +
                                    desc.Label(mapRecord.key) + "'");
            return;
        }
        UsdGenResolvedMapValue value;
        value.key = mapRecord.key;
        if (mapInput->resolveMap) mapInput->resolveMap(mapRecord, &value);
        mapByPath.emplace(mapRecord.key, static_cast<uint32_t>(resolvedMapValues.size()));
        resolvedMapValues.push_back(std::move(value));
    }
    for (size_t nodeIndex = 0; nodeIndex != desc.nodes.size(); ++nodeIndex) {
        UsdGenOpNode const& node = desc.nodes[nodeIndex];
        StaticOperatorContract const& contract = operatorContracts[nodeIndex];
        bool const hasExternalInputs = !node.references.empty() ||
            !node.inputSets.empty() || !node.maps.empty() ||
            !node.mapBindings.empty();
        if (!hasExternalInputs) {
            // A reference-role producer cannot silently omit its declared
            // slots.  This is the only empty-input rule; every nonempty
            // reference, input set, or map path continues through the complete
            // duplicate/target/typed-binding validation below.
            if (contract.lane == UsdGenLane::Reference &&
                contract.referenceInputArity != 0) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' has 0 resolved reference values but its operator declares " +
                    std::to_string(contract.referenceInputArity) + " reference slots");
                return;
            }
            continue;
        }
        PSet<UsdGenNodeKey> references;
        auto addReference = [&](UsdGenNodeKey path) {
            if (!references.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' declares duplicate reference value '" + desc.Label(path) + "'");
                return false;
            }
            if (!referenceByPath.count(path)) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' references unresolved reference value '" + desc.Label(path) + "'");
                return false;
            }
            return true;
        };
        for (UsdGenNodeKey const path : node.references)
            if (!addReference(path)) return;
        for (UsdGenNodeKey const path : node.inputSets)
            if (referenceByPath.count(path) && !addReference(path)) return;
        // Preserve the reference-free incremental fast path: factory
        // existence and the complete version-resolved static contract were
        // already checked above.  Do not allocate a probe per node here:
        // role/reference-slot metadata came from the registration-time
        // probe, exactly like geometry input arity.
        if (!references.empty()) {
            if (references.size() != contract.referenceInputArity) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' has " + std::to_string(references.size()) +
                    " resolved reference values but its operator declares " +
                    std::to_string(contract.referenceInputArity) + " reference slots");
                return;
            }
        } else if (contract.lane == UsdGenLane::Reference &&
                   contract.referenceInputArity != 0) {
            result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                "' has 0 resolved reference values but its operator declares " +
                std::to_string(contract.referenceInputArity) + " reference slots");
            return;
        }
        PVec<UsdGenMapBinding> bindings;
        std::string bindingError;
        if (!EffectiveMapBindings(node, &bindings, &bindingError)) {
            result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                "' has ambiguous map bindings: " + bindingError);
            return;
        }
        PSet<UsdGenNodeKey> maps;
        for (UsdGenMapBinding const &binding : bindings) {
            UsdGenNodeKey const path = binding.map;
            if (!ValidateMapBinding(binding, *descRecord[nodeIndex], &bindingError)) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' has ambiguous map binding: " + bindingError);
                return;
            }
            if (!maps.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' declares duplicate map binding '" + desc.Label(path) + "'");
                return;
            }
            if (!mapByPath.count(path)) {
                result.errors.push_back("UsdGenCompiler: node '" + desc.Label(node.key) +
                    "' references unresolved map value '" + desc.Label(path) + "'");
                return;
            }
        }
    }
    // Desc copy elision (E-6): node entries byte-identical to the previous
    // graph's copy move instead of copying — a 201-entry copy is ~600 small
    // allocs (vectors + UsdGenValue holders per param). Scalars always copy.
    // Entry identity is by path; equality is field-wise (no allocs on hit).
    // Correctness: nodes read their desc entry read-only through desc/paramView.
    auto sameNodeDesc = [](UsdGenOpNode const &a, UsdGenOpNode const &b) {
        // Until expression program identity participates in incremental
        // digests, copy connected descriptors instead of reusing stale source
        // bindings/literals. This is conservative, not a semantic fallback.
        if (!a.expressionBindings.empty() || !b.expressionBindings.empty()) return false;
        if (a.key != b.key || a.type != b.type || a.mode != b.mode ||
            a.enabled != b.enabled ||
            a.seed != b.seed || a.inputs != b.inputs ||
            a.references != b.references || a.inputSets != b.inputSets ||
            a.surfaces != b.surfaces || a.maps != b.maps ||
            a.mapBindings.size() != b.mapBindings.size() ||
            a.params.size() != b.params.size() || a.ramps.size() != b.ramps.size())
            return false;
        for (size_t i = 0; i < a.mapBindings.size(); ++i) {
            if (a.mapBindings[i].map != b.mapBindings[i].map ||
                a.mapBindings[i].relationship != b.mapBindings[i].relationship)
                return false;
        }
        for (size_t i = 0; i < a.params.size(); ++i) {
            if (a.params[i].name != b.params[i].name ||
                a.params[i].animated != b.params[i].animated ||
                !(a.params[i].value == b.params[i].value))
                return false;
        }
        for (size_t i = 0; i < a.ramps.size(); ++i) {
            if (a.ramps[i].knots != b.ramps[i].knots ||
                a.ramps[i].positions != b.ramps[i].positions ||
                a.ramps[i].colors != b.ramps[i].colors ||
                a.ramps[i].interpolation != b.ramps[i].interpolation)
                return false;
        }
        return true;
    };
    // Transactional recompile: a rebuilt operator binds its extra plane
    // slots from its own Configure(), so probe every node that will rebuild
    // (no byte-identical entry in the previous descriptor) before any old
    // node or descriptor entry moves. Reused nodes keep slots that already
    // bound. A failure here leaves *out untouched.
    if (reuse == out && out->_desc) {
        UsdGenGraphSpec const &prev = *out->_desc;
        for (int di = 0; di < n; ++di) {
            UsdGenOpNode const &nd = desc.nodes[size_t(di)];
            UsdGenOpNode const *old = nullptr;
            if (size_t(di) < prev.nodes.size() && prev.nodes[size_t(di)].key == nd.key)
                old = &prev.nodes[size_t(di)];
            for (size_t j = 0; !old && j < prev.nodes.size(); ++j)
                if (prev.nodes[j].key == nd.key) old = &prev.nodes[j];
            bool const kept = old && sameNodeDesc(*old, nd);
            if (kept) continue;
            UsdGenOpRecord const &record = *descRecord[size_t(di)];
            std::unique_ptr<UsdGenOp> probe = ops.Create(record.Id());
            if (!probe) {
                result.errors.push_back("UsdGenCompiler: no kernel registered for '" +
                    nd.type.GetString() + "' (prim " + desc.Label(nd.key) + ")");
                return;
            }
            UsdGenCompiledNode scratch;
            scratch.desc = &nd;
            BindParams(scratch, &desc, &nd, &record, indexedBases[size_t(di)],
                       indexedCounts[size_t(di)]);
            probe->Configure({scratch.paramView, scratch.indexedPlaneBase});
            if (!BindExtraPlaneSlots(desc, record, &scratch, &result)) return;
        }
    }
    // The previous graph's nodes, in the graph's retained vector so its
    // capacity survives; whatever is left is destroyed when _Build returns.
    std::vector<std::unique_ptr<UsdGenCompiledNode>> &oldNodes = out->_retiredNodes;
    struct RetireOnExit {
        std::vector<std::unique_ptr<UsdGenCompiledNode>> &nodes;
        ~RetireOnExit() { for (auto &np : nodes) np.reset(); nodes.clear(); }
    } retireOnExit{oldNodes};
    oldNodes.clear();
    // descIdx → node index over the OLD graph (E-6): the merge and the node
    // loop map by position, with a linear path scan only for reorder misses
    // (rare) — no index vector, no sort, no per-lookup searches on the hit
    // key. UsdGenNodeKey::operator== is an integer compare.
    PVec<size_t> nodeByDesc;
    if (reuse && reuse != out) {
        // Defensive: Recompile always passes out as reuse; a foreign graph
        // is indexed read-only (no moves) — nodes rebuild fully below.
    } else if (reuse) {
        oldNodes.swap(out->_nodes);
        result.graphIntact = false;   // restored to true on success
    }
    if (!oldNodes.empty() && out->_desc) {
        nodeByDesc.assign(out->_desc->nodes.size(), SIZE_MAX);
        for (size_t i = 0; i < oldNodes.size(); ++i)
            if (oldNodes[i] && oldNodes[i]->descIdx >= 0 &&
                size_t(oldNodes[i]->descIdx) < nodeByDesc.size())
                nodeByDesc[size_t(oldNodes[i]->descIdx)] = i;
    }
    UsdGenGraphSpec *oldDescPtr = (reuse && out->_desc) ? out->_desc.get() : nullptr;
    PVec<char> descChanged(desc.nodes.size(), 1);
    // oldNodeForNewDesc[di] = old node index, or -1 (recorded by the merge,
    // so the node loop needs no per-node searches on the hit path).
    PVec<int> oldNodeForNewDesc(desc.nodes.size(), -1);
    // oldNodeAtPath[di] = the path-matched old node index even when the desc
    // entry changed (-1 when the path is new). A rebuilt node continues that
    // node's buffer stamps; see the rebuild branch of the node loop.
    PVec<int> oldNodeAtPath(desc.nodes.size(), -1);
    // entries copy from the input. Entry ORDER follows the input (S26).
    // Identity is positional (nodeByDesc) with a linear-scan fallback.
    static_assert(sizeof(UsdGenGraphSpec) ==
                      232 + sizeof(std::vector<UsdGenInputRecord>) + sizeof(UsdGenPreviewSpec) +
                      sizeof(std::vector<UsdGenNodeLabel>) + sizeof(std::shared_ptr<const void>),
        "UsdGenGraphSpec changed size: update the Recompile shell merge below");
    {
        // Double-buffered: the descriptor from two compiles back is
        // overwritten field by field, so its vectors' capacity is reused.
        std::unique_ptr<UsdGenGraphSpec> fresh = std::move(out->_spareDesc);
        if (!fresh) fresh = std::make_unique<UsdGenGraphSpec>();
        fresh->description = desc.description;
        fresh->terminal = desc.terminal;
        fresh->inputs = desc.inputs;
        fresh->validationErrors = desc.validationErrors;
        fresh->executionBackend = desc.executionBackend;
        fresh->preview = desc.preview;
        fresh->xformMatrix = desc.xformMatrix;
        fresh->labels = desc.labels;
        fresh->host = desc.host;
        fresh->hostDigest = desc.hostDigest;
        fresh->defaultWidth = desc.defaultWidth;
        fresh->tileTarget = desc.tileTarget;
        fresh->time = desc.time;
        fresh->timeCodesPerSecond = desc.timeCodesPerSecond;
        fresh->nodes.resize(desc.nodes.size());
        // Per-input-order entry change flags for digest-change propagation
        // below (plan §3.5: only changed nodes and their descendants pay).
        if (oldDescPtr) {
            size_t const oldSize = oldDescPtr->nodes.size();
            for (size_t i = 0; i < desc.nodes.size(); ++i) {
                // Positional pre-check (E-6): appends/param-edits keep
                // namespace order — one integer key == instead of
                // a binary search on the hit path.
                size_t o = oldSize;  // sentinel: linear-scan fallback below
                size_t on = oldSize; // old NODE index for oldNodeForNewDesc
                if (i < oldSize && oldDescPtr->nodes[i].key == desc.nodes[i].key) {
                    o = i;
                    if (i < nodeByDesc.size() && nodeByDesc[i] != SIZE_MAX) on = nodeByDesc[i];
                } else {
                    // Reorder miss (rare): linear path scan with integer
                    // key compares; node index rides nodeByDesc.
                    for (size_t j = 0; j < oldSize; ++j) {
                        if (oldDescPtr->nodes[j].key == desc.nodes[i].key) {
                            o = j;
                            if (j < nodeByDesc.size() && nodeByDesc[j] != SIZE_MAX)
                                on = nodeByDesc[j];
                            break;
                        }
                    }
                }
                if (on < oldSize) oldNodeAtPath[i] = int(on);
                if (o < oldSize && sameNodeDesc(oldDescPtr->nodes[o], desc.nodes[i])) {
                    // Swap, not move: the old entry keeps the stale
                    // storage for the next compile to overwrite.
                    std::swap(fresh->nodes[i], oldDescPtr->nodes[o]);
                    descChanged[i] = 0;
                    if (on < oldSize) oldNodeForNewDesc[i] = int(on);
                } else {
                    fresh->nodes[i] = desc.nodes[i];
                }
            }
        } else {
            fresh->nodes = desc.nodes;
        }
        out->_spareDesc = std::move(out->_desc);
        out->_desc = std::move(fresh);
        UsdGenResolveDescIds(out->_desc.get());
        // RestSpec() keys its cache on the descriptor's address, which the
        // double buffer reuses: drop the cache with the descriptor.
        out->_restSource = nullptr;
    }

    // External values were fully validated before any reusable graph state
    // moved.  Transfer the immutable candidates only after the new descriptor
    // shell has been installed.
    out->_referenceValues = std::move(resolvedReferenceValues);
    out->_mapValues = std::move(resolvedMapValues);
    out->_nodes.clear();
    out->_nodes.resize(n);
    out->_nodeByKey.clear();
    out->_terminal = terminalId;
    out->_chunkSize = kUsdGenDefaultChunkSize;   // USDGEN_CHUNK_SIZE (S23)
    out->_tileTarget = desc.tileTarget;
    PVec<UsdGenEpoch> digests(n);
    PVec<char> digestChanged(n, 0);
    for (int pos = 0; pos < n; ++pos) {
        int const di = order[pos];
        UsdGenOpNode const &nd = desc.nodes[di];
        UsdGenOpRecord const *const record = descRecord[size_t(di)];
        int const indexedBase = indexedBases[size_t(di)];
        int const indexedCount = indexedCounts[size_t(di)];
        UsdGenOpNode const *const newNd = &out->_desc->nodes[di];
        // A node whose indexed run moved binds different plane names.
        bool const indexedSlotsChanged = oldNodeForNewDesc[di] >= 0 &&
            oldNodes[size_t(oldNodeForNewDesc[di])] &&
            (oldNodes[size_t(oldNodeForNewDesc[di])]->indexedPlaneBase != indexedBase ||
             oldNodes[size_t(oldNodeForNewDesc[di])]->indexedPlaneCount != indexedCount);
        // Change propagation (plan §3.5): a node whose desc entry is
        // byte-identical AND whose inputs' digests all survived keeps its
        // old digest verbatim — no FNV, no path vectors, no GetHash. Only
        // changed nodes and downstream consumers recompute.
        // Inputs come pre-resolved from the fused edge pass (E-6): no
        // per-node binary searches for the change check or the refill.
        int const e0 = edgeOff[size_t(di)], e1 = edgeOff[size_t(di) + 1];
        bool inputChanged = false;
        for (int k = e0; !inputChanged && k < e1; ++k)
            inputChanged = digestChanged[topoOfDesc[edgeIds[size_t(k)]]] != 0;
        if (!oldNodes.empty() && !descChanged[di] && !inputChanged &&
            !indexedSlotsChanged &&
            oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode &oldS = *oldNodes[size_t(oldNodeForNewDesc[di])];
            if (oldS.record == record) {
                // Verbatim reuse: refresh desc-owned fields and rebuild the
                // dense-id edge list in retained capacity (no alloc when the
                // fan-in is stable). sameNodeDesc() above already proved all
                // value-class inputs, enabled state and ramps are
                // unchanged, so retain the existing value digest instead of
                // rescanning every operator parameter on every reused node.
                auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                moved->id = static_cast<UsdGenNodeId>(pos);
                moved->desc = &out->_desc->nodes[di];
                moved->descIdx = di;
                moved->enabled = nd.enabled;
                BindParams(*moved, out->_desc.get(), newNd, record, indexedBase, indexedCount);
                moved->inputs.clear();
                for (int k = e0; k < e1; ++k)
                    moved->inputs.push_back(static_cast<UsdGenNodeId>(topoOfDesc[edgeIds[size_t(k)]]));
                // Preserve authored edge order.  Unary operators do not give
                // that order semantic meaning, but ordered fan-in operators
                // (a binary blend) use inputs[0] as left and inputs[1] as
                // right and must retain it through recompile.
                moved->input = moved->inputs.empty() ? kUsdGenInvalidNode : moved->inputs.front();
                fillDescendants(pos, &moved->descendants);
                digests[pos] = moved->structuralDigest;
                out->_nodeByKey[nd.key] = moved->id;
                out->_nodes[pos] = std::move(moved);
                continue;
            }
        }
        // Value-only edit: the path-matched old node's entry changed only in
        // value-class parameter values, so its structure, slots, chunks,
        // capture and buffer all stay valid. Keep it whole (no new op, no
        // re-capture unless its capture digest moves); Recompile marks its
        // chunks and descendants value-dirty. The op is re-configured from
        // the new entry and must bind the same slots and planes.
        if (!oldNodes.empty() && descChanged[di] && !inputChanged &&
            oldNodeForNewDesc[di] < 0 && oldNodeAtPath[di] >= 0 &&
            size_t(oldNodeAtPath[di]) < oldNodes.size() &&
            oldNodes[size_t(oldNodeAtPath[di])]) {
            UsdGenCompiledNode &old = *oldNodes[size_t(oldNodeAtPath[di])];
            if (old.op && old.desc && old.record == record &&
                old.countChanges == record->Def().countChanges &&
                old.lane == record->Def().lane &&
                ValueOnlyEdit(*old.desc, nd, *record)) {
                uint32_t const planes = old.op->PlanesTouched();
                bool const sameRun = old.indexedPlaneBase == indexedBase &&
                    old.indexedPlaneCount == indexedCount;
                // Re-resolved against the new entry before Configure reads it.
                BindParams(old, out->_desc.get(), newNd, record, indexedBase, indexedCount);
                old.op->Configure({old.paramView, old.indexedPlaneBase});
                if (planes == old.op->PlanesTouched() && sameRun &&
                    SamePlaneSlots(*record, old)) {
                    auto moved = std::move(oldNodes[size_t(oldNodeAtPath[di])]);
                    moved->id = static_cast<UsdGenNodeId>(pos);
                    moved->desc = &out->_desc->nodes[di];
                    moved->descIdx = di;
                    moved->enabled = nd.enabled;
                    moved->inputs.clear();
                    for (int k = e0; k < e1; ++k)
                        moved->inputs.push_back(static_cast<UsdGenNodeId>(topoOfDesc[edgeIds[size_t(k)]]));
                    moved->input = moved->inputs.empty() ? kUsdGenInvalidNode : moved->inputs.front();
                    fillDescendants(pos, &moved->descendants);
                    // the value digest moves with the edited values
                    moved->paramValueDigest = ParamValueDigest(nd, *record);
                    digests[pos] = moved->structuralDigest;
                    digestChanged[pos] = 0;
                    result.valueEdited.push_back(moved->id);
                    out->_nodeByKey[nd.key] = moved->id;
                    out->_nodes[pos] = std::move(moved);
                    continue;
                }
            }
        }
        // E-6 fast path: a path-matched old node of the same type and
        // version reuses its OWN op for the stability check — no op Create,
        // no vector building, no routing copy. On stability the old node
        // moves whole with only desc pointers/digests refreshed; the
        // expensive per-node allocs (node, inputs, refs, routing table) and
        // their matching destructions never happen.
        if (!oldNodes.empty() && oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode &old0 = *oldNodes[size_t(oldNodeForNewDesc[di])];
            if (old0.record == record) {
                PVec<UsdGenNodeKey> inputPaths0(nd.inputs.begin(), nd.inputs.end());
                if (!record->Has(kUsdGenOpGeometryInputsOrdered))
                    std::sort(inputPaths0.begin(), inputPaths0.end());
                PVec<UsdGenNodeKey> refPaths0(nd.references.begin(), nd.references.end());
                PVec<UsdGenMapBinding> bindings0;
                std::string bindingError0;
                if (!EffectiveMapBindings(nd, &bindings0, &bindingError0)) {
                    result.errors.push_back("UsdGenCompiler: node '" + desc.Label(nd.key) +
                        "' has ambiguous map bindings: " + bindingError0);
                    return;
                }
                for (UsdGenMapBinding const &binding : bindings0)
                    refPaths0.push_back(binding.map);
                refPaths0.insert(refPaths0.end(), nd.surfaces.begin(), nd.surfaces.end());
                std::sort(refPaths0.begin(), refPaths0.end());
                PVec<std::pair<UsdGenNodeKey, UsdGenEpoch>> childDigests0;
                for (UsdGenNodeKey const p : inputPaths0) {
                    auto const it = findDescIdx(p);
                    if (it != descIdxByPath.end())
                        childDigests0.emplace_back(p, digests[topoOfDesc[it->second]]);
                }
                UsdGenEpoch const dg0 = ComputeNodeDigest(
                    nd, TypeSeed(nd.type), record,
                    inputPaths0, refPaths0, bindings0, childDigests0);
                if (dg0 == old0.structuralDigest && !indexedSlotsChanged &&
                    old0.countChanges == record->Def().countChanges &&
                    old0.lane == record->Def().lane) {
                    // Stable: move whole, refresh desc-owned fields only.
                    // inputs/descendants-aside everything rides along:
                    // op, capture, buffer, chunks, dirty bytes, epochs,
                    // lastParamDigest (scheduler skip state survives).
                    auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                    moved->id = static_cast<UsdGenNodeId>(pos);
                    moved->desc = &out->_desc->nodes[di];
                    moved->descIdx = di;
                    moved->structuralDigest = dg0;
                    moved->enabled = nd.enabled;
                    BindParams(*moved, out->_desc.get(), newNd, record, indexedBase,
                               indexedCount);
                    fillDescendants(pos, &moved->descendants);
                    // value digest refreshes (cheap, no allocs).
                    moved->paramValueDigest = ParamValueDigest(nd, *record);
                    digests[pos] = dg0;
                    digestChanged[pos] = 0;
                    out->_nodeByKey[nd.key] = moved->id;
                    out->_nodes[pos] = std::move(moved);
                    continue;
                }
            }
        }

        auto node = std::make_unique<UsdGenCompiledNode>();
        // The path-matched old node lends the storage of everything rebuilt
        // below; a stable node takes it back when it moves over.
        int const lendFrom = oldNodeForNewDesc[di] >= 0 ? oldNodeForNewDesc[di]
                                                        : oldNodeAtPath[di];
        auto lend = [](auto &to, auto &from) { to.swap(from); to.clear(); };
        if (lendFrom >= 0 && size_t(lendFrom) < oldNodes.size() &&
            oldNodes[size_t(lendFrom)]) {
            UsdGenCompiledNode &old = *oldNodes[size_t(lendFrom)];
            lend(node->inputs, old.inputs);
            lend(node->inputRefs, old.inputRefs);
            lend(node->mapRefs, old.mapRefs);
            lend(node->mapBindingRefs, old.mapBindingRefs);
            lend(node->outputPrimvars, old.outputPrimvars);
            lend(node->inputPrimvars, old.inputPrimvars);
            lend(node->paramRouting, old.paramRouting);
            lend(node->descendants, old.descendants);
            lend(node->params, old.params);
        }
        node->id = static_cast<UsdGenNodeId>(pos);
        node->desc = &out->_desc->nodes[di];
        node->descIdx = di;

        std::unique_ptr<UsdGenOp> op = record ? ops.Create(record->Id()) : nullptr;
        if (!op) {
            result.errors.push_back(
                "UsdGenCompiler: no kernel registered for '" +
                nd.type.GetString() + "' (prim " + desc.Label(nd.key) + ")");
            return;
        }
        node->op = std::move(op);
        // Parameter view and slots over the graph's desc copy. Set before
        // the planes are bound: an operator may name its planes from its
        // parameters.
        BindParams(*node, out->_desc.get(), newNd, record, indexedBase, indexedCount);
        node->op->Configure({node->paramView, node->indexedPlaneBase});
        if (!BindExtraPlaneSlots(desc, *record, node.get(), &result)) return;

        node->countChanges = record->Def().countChanges;
        node->lane = record->Def().lane;
        node->enabled = nd.enabled;

        // Input edges (dense ids).
        for (UsdGenNodeKey const in : nd.inputs) {
            auto const it = findDescIdx(in);
            if (it != descIdxByPath.end())
                node->inputs.push_back(static_cast<UsdGenNodeId>(topoOfDesc[it->second]));
        }
        // Keep dense IDs in the same authored order as UsdGenOpNode::inputs;
        // WidthBlend's ordered operands depend on this correspondence.
        node->input = node->inputs.empty() ? kUsdGenInvalidNode : node->inputs.front();

        // Relationship targets.
        node->inputRefs = nd.inputSets;
        PVec<UsdGenMapBinding> bindings;
        std::string bindingError;
        if (!EffectiveMapBindings(nd, &bindings, &bindingError)) {
            result.errors.push_back("UsdGenCompiler: node '" + desc.Label(nd.key) +
                "' has ambiguous map bindings: " + bindingError);
            return;
        }
        node->mapBindingRefs.assign(bindings.begin(), bindings.end());
        node->mapRefs.clear();
        node->mapRefs.reserve(bindings.size());
        for (UsdGenMapBinding const &binding : bindings)
            node->mapRefs.push_back(binding.map);
        if (!nd.surfaces.empty()) {
            node->hasSurface = true;
            node->surface = 0;   // filled below when the surface inputs are indexed
        }

        // Strict descendants, ascending dense-id order (bitmap pass above).
        fillDescendants(pos, &node->descendants);

        // Structural digest (ADR §4.2.1 term list).
        PVec<UsdGenNodeKey> inputPaths(nd.inputs.begin(), nd.inputs.end());
        if (!record->Has(kUsdGenOpGeometryInputsOrdered))
            std::sort(inputPaths.begin(), inputPaths.end());
        PVec<UsdGenNodeKey> refPaths(nd.references.begin(), nd.references.end());
        for (UsdGenMapBinding const &binding : bindings)
            refPaths.push_back(binding.map);
        refPaths.insert(refPaths.end(), nd.surfaces.begin(), nd.surfaces.end());
        std::sort(refPaths.begin(), refPaths.end());
        PVec<std::pair<UsdGenNodeKey, UsdGenEpoch>> childDigests;
        for (UsdGenNodeKey const p : inputPaths) {
            auto const it = findDescIdx(p);
            if (it != descIdxByPath.end())
                childDigests.emplace_back(p, digests[topoOfDesc[it->second]]);
        }
        // Structural digest: only §6.1-class (recompile) parameter values
        // are digest terms — capture- and topology-class edits must NOT
        // move the digest (03 §3.3/§3.6). The digest slots and the routing
        // table are the record's, built once at registration.
        digests[pos] = ComputeNodeDigest(
            nd, TypeSeed(nd.type), record,
            inputPaths, refPaths, bindings, childDigests);
        node->structuralDigest = digests[pos];

        // Value-class digest (skip-signature term, 03 §3.2): recompile keeps
        // this stable unless a value-class param or the enabled flag moved.
        // Unknown (unclassified) params feed it too — a value-class edit is
        // the safe superset for an unlisted property.
        node->paramValueDigest = ParamValueDigest(nd, *record);

        // Parameter routing table (02 §6): every routed parameter of this
        // type gets exactly one entry, copied from the record.
        UsdGenSpan<const std::pair<UsdGenToken, uint32_t>> const routing = record->Routing();
        node->paramRouting.assign(routing.begin(), routing.end());

        // E-6 reuse: a digest-STABLE node moves whole — op, capture, buffer,
        // chunks, dirty bytes, epochs — out of the previous graph. Only the
        // desc pointers and digests refresh. A null capture (never-run node)
        // is NOT a rebuild: the capture installs lazily on the next commit.
        if (!oldNodes.empty() && oldNodeForNewDesc[di] >= 0) {
            UsdGenCompiledNode const &oldRef = *oldNodes[size_t(oldNodeForNewDesc[di])];
            bool const stable = !indexedSlotsChanged &&
                oldRef.structuralDigest == node->structuralDigest &&
                oldRef.record == node->record &&
                oldRef.countChanges == node->countChanges &&
                oldRef.lane == node->lane;
            if (stable) {
                // Move the whole node; refresh only what the new desc
                // owns. Op, capture, buffer, chunks, dirty bytes,
                // capture epoch, topologySeq and eval signature ride
                // along untouched — including lastParamDigest, so the
                // scheduler's skip logic survives the recompile.
                auto moved = std::move(oldNodes[size_t(oldNodeForNewDesc[di])]);
                moved->id = node->id;
                moved->desc = node->desc;
                moved->descIdx = node->descIdx;
                moved->structuralDigest = node->structuralDigest;
                moved->paramValueDigest = node->paramValueDigest;
                moved->enabled = node->enabled;
                moved->input = node->input;
                moved->inputs = std::move(node->inputs);
                moved->descendants = std::move(node->descendants);
                moved->inputRefs = std::move(node->inputRefs);
                moved->mapRefs = std::move(node->mapRefs);
                moved->mapBindingRefs = std::move(node->mapBindingRefs);
                moved->hasSurface = node->hasSurface;
                moved->surface = node->surface;
                moved->params.swap(node->params);
                BindParams(*moved, out->_desc.get(), newNd, record, indexedBase,
                           indexedCount);
                moved->paramRouting = std::move(node->paramRouting);
                moved->outputPrimvars = std::move(node->outputPrimvars);
                moved->inputPrimvars = std::move(node->inputPrimvars);
                node = std::move(moved);
                digestChanged[pos] = 0;
            } else {
                result.rebuilt.push_back(node->id);
                digestChanged[pos] = 1;
            }
        } else {
            result.rebuilt.push_back(node->id);
            digestChanged[pos] = 1;
        }

        // A rebuilt node REPLACES its path-matched predecessor: its buffer
        // stamps continue the old sequence instead of restarting at zero.
        // Downstream captures (Clump, Curl, Expr, ...) key their validity on
        // the upstream (topologyVersion, valueVersion); a fresh node counting
        // 0 -> 1 again in one cook reproduces the very stamps they recorded,
        // so they keep serving the old geometry. Every expression-bound node
        // rebuilds on every recompile (sameNodeDesc), so a length-paint
        // repaint rebuilt Grow and the groom stayed unchanged until a
        // density edit changed the element count under the whole chain.
        // A rebuilt node also takes its predecessor's partition storage; the
        // first commit repartitions into it.
        if (lendFrom >= 0 && size_t(lendFrom) < oldNodes.size() &&
            oldNodes[size_t(lendFrom)] && node->chunks.empty()) {
            UsdGenCompiledNode &old = *oldNodes[size_t(lendFrom)];
            lend(node->chunks, old.chunks);
            lend(node->chunkDirty, old.chunkDirty);
            lend(node->chunkCaptured, old.chunkCaptured);
        }
        if (node->topologyVersion == 0 && node->valueVersion == 0 &&
            oldNodeAtPath[di] >= 0 && size_t(oldNodeAtPath[di]) < oldNodes.size() &&
            oldNodes[size_t(oldNodeAtPath[di])]) {
            UsdGenCompiledNode const &prior = *oldNodes[size_t(oldNodeAtPath[di])];
            node->topologySeq = prior.topologySeq;
            node->topologyVersion = prior.topologyVersion;
            node->valueVersion = prior.valueVersion;
        }
        out->_nodeByKey[nd.key] = node->id;
        out->_nodes[pos] = std::move(node);
    }

    // Bind external descriptor values after all node instances have been
    // created.  `guides` paths live in inputRefs in both builders; generic
    // `references` paths are accepted too.  Preserve authored order: duplicate
    // paths were rejected during the transactional preflight above.
    for (auto const &nodePtr : out->_nodes) {
        UsdGenCompiledNode &node = *nodePtr;
        node.referenceValues.clear();
        node.mapValues.clear();
        // Everything a connected expression samples (geoSampler/ptex inputs):
        // an edit there re-evaluates this node, like an input set or map edit.
        node.geometryRefs.clear();
        UsdGenParamEvaluatorDef const &evaluator = UsdGenOpRegistry::Get().ParamEvaluator();
        if (evaluator.bindingDependencies) {
            auto add = [](void *context, UsdGenNodeKey path) {
                std::vector<UsdGenNodeKey> &refs = *static_cast<std::vector<UsdGenNodeKey> *>(context);
                if (std::find(refs.begin(), refs.end(), path) == refs.end()) refs.push_back(path);
            };
            for (UsdGenParamBinding const &binding : node.desc->expressionBindings)
                evaluator.bindingDependencies(*out->_desc, binding.expression, &node.geometryRefs, add);
        }
        if (node.desc->references.empty() && node.inputRefs.empty() &&
            node.mapBindingRefs.empty()) {
            // Static empty-slot validation completed before any old runtime
            // state moved.  Avoid constructing empty ordered sets and making
            // a virtual ReferenceInputs() call for the common no-external-
            // inputs case; stale resolved handles were cleared above.
            continue;
        }
        PSet<UsdGenNodeKey> seenReferences;
        auto addReference = [&](UsdGenNodeKey path) {
            if (!seenReferences.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    desc.Label(node.desc->key) + "' declares duplicate reference value '" +
                    desc.Label(path) + "'");
                return false;
            }
            auto const it = referenceByPath.find(path);
            if (it == referenceByPath.end()) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    desc.Label(node.desc->key) + "' references unresolved reference value '" +
                    desc.Label(path) + "'");
                return false;
            }
            node.referenceValues.push_back(it->second);
            return true;
        };
        for (UsdGenNodeKey const path : node.desc->references)
            if (!addReference(path)) return;
        for (UsdGenNodeKey const path : node.inputRefs)
            if (referenceByPath.count(path) && !addReference(path)) return;

        UsdGenSpan<const UsdGenToken> const declaredReferences = node.record->ReferenceInputs();
        if ((!node.referenceValues.empty() || node.lane == UsdGenLane::Reference) &&
            node.referenceValues.size() != declaredReferences.size()) {
            result.errors.push_back("UsdGenCompiler: node '" +
                desc.Label(node.desc->key) + "' has " +
                std::to_string(node.referenceValues.size()) +
                " resolved reference values but its operator declares " +
                std::to_string(declaredReferences.size()) + " reference slots");
            return;
        }
        PSet<UsdGenNodeKey> seenMaps;
        for (UsdGenMapBinding const &binding : node.mapBindingRefs) {
            UsdGenNodeKey const path = binding.map;
            if (!seenMaps.insert(path).second) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    desc.Label(node.desc->key) + "' declares duplicate map binding '" +
                    desc.Label(path) + "'");
                return;
            }
            auto const it = mapByPath.find(path);
            if (it == mapByPath.end()) {
                result.errors.push_back("UsdGenCompiler: node '" +
                    desc.Label(node.desc->key) + "' references unresolved map value '" +
                    desc.Label(path) + "'");
                return;
            }
            node.mapValues.push_back(it->second);
        }
    }

    // Surface id assignment: index the surface inputs; bind node->surface.
    PHashMap<UsdGenNodeKey, UsdGenSurfaceId, UsdGenNodeKeyHash> surfaceIdByPath;
    UsdGenTypeId const surfaceKind = UsdGenInputKindWithRole(UsdGenInputRole::Surface);
    UsdGenInputDesc const *const surfaceInput = UsdGenFindInputDesc(surfaceKind);
    UsdGenSpan<UsdGenInputRecord> const surfaceRecords =
        surfaceInput ? out->_desc->MutableInputs(surfaceKind) : UsdGenSpan<UsdGenInputRecord>();
    for (size_t s = 0; s < surfaceRecords.size(); ++s) {
        surfaceIdByPath[surfaceRecords[s].key] = static_cast<UsdGenSurfaceId>(s);
        if (surfaceInput->setSurfaceId)
            surfaceInput->setSurfaceId(surfaceRecords[s], static_cast<uint32_t>(s));
    }
    for (auto &np : out->_nodes) {
        if (!np->hasSurface) continue;
        auto const it = surfaceIdByPath.find(np->desc->surfaces.front());
        np->surface = (it != surfaceIdByPath.end()) ? it->second : 0;
    }
    std::pair<UsdGenNodeKey, UsdGenEpoch> const terminalDigest{desc.terminal, digests[terminalId]};
    result.structuralDigest = ComputeNodeDigest(
        out->_desc->nodes[termIt->second], kTerminalTypeSeed, nullptr,
        UsdGenSpan<const UsdGenNodeKey>(&desc.terminal, 1), {}, {},
        UsdGenSpan<const std::pair<UsdGenNodeKey, UsdGenEpoch>>(&terminalDigest, 1));
}


void UsdGenDiagnostics::Error(std::string_view msg) { errors.emplace_back(msg); }
void UsdGenDiagnostics::Warn(std::string_view msg) { warnings.emplace_back(msg); }

void UsdGenResolveParams(UsdGenOpRecord const &rec, UsdGenOpNode const &node,
                         UsdGenResolvedParam *out)
{
    uint16_t const count = rec.ParamCount();
    for (uint16_t s = 0; s < count; ++s) out[s] = UsdGenResolvedParam();
    for (UsdGenParamValue const &p : node.params) {
        UsdGenParamSlot const s = ExactSlot(rec, p.name);
        if (s == kUsdGenNoParam || out[s].authored) continue;
        UsdGenResolvedParam &r = out[s];
        r.authored = &p;
        if (p.value.IsHolding<UsdGenToken>())
            r.token = p.value.UncheckedGet<UsdGenToken>();
        else if (p.value.IsHolding<std::string>())
            r.token = UsdGenInternToken(p.value.UncheckedGet<std::string>());
        else
            continue;
        r.enumIndex = rec.EnumIndex(s, r.token);
    }
}

}  // namespace usdGen
