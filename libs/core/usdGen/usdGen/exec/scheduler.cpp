// usdGen engine — scheduler implementation (03-execution-engine.md §5.3/§5.4).
//
// Run() executes engine-side commit steps in deterministic dependency-ready
// frontiers. Each frontier captures and repartitions serially, prepares all
// COW payloads serially, then sweeps its flattened node/chunk work in parallel:
//   - reference lane (03 §1.5): reference-role nodes run first, with
//     immutable resolved CurveSet/map values supplied to their capture;
//     this CPU slice does not implement chaining from a reference node's
//     produced geometry into another reference node;
//   - re-capture when the capture epoch moved (03 §3.4); generators install
//     their captured buffer and the graph re-partitions on topology change;
//   - per-frontier chunk-evaluate sweep over the scheduler's worker pool
//     (I8 — never pxr work::, which would serialise under
//     PXR_WORK_THREAD_LIMIT); chunk skip is governed by the dirty bytes plus
//     the evaluation signature (param value digest + upstream valueVersion);
//   - tile interleave: extents + dirty flags over dirty tiles only.
//
// I8: every parallel region runs over the worker pool, never pxr work::.
#include "usdGen/scheduler.h"
#include "usdGen/registries/engineTokens.h"

#include "usdGen/debugCodes.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/registries/opRegistry.h"
#include "usdGen/fastCores.h"
#include "usdGen/types.h"
#include "usdGen/api/writeSet.h"
#include "usdGen/exec/epoch.h"
#include "usdGenCurves/types/curveSet.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/getenv.h"
#include "pxr/base/trace/trace.h"
#include "pxr/base/vt/array.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

int ResolveThreadLimit(int requested)
{
    if (requested > 0) return requested;
    const std::string env = TfGetenv("USDGEN_THREAD_LIMIT");  // 26.08: std::string
    if (!env.empty()) {
        const int parsed = std::atoi(env.c_str());
        if (parsed > 0) return parsed;
    }
    // 03 §5.3: the topology-aware default. The measured 8-thread knee
    // (EV-001/EV-008) is the floor; on heterogeneous Linux with more fast
    // cores than that, the arena spans every fast core — pinning engages
    // by construction (workers == fast cores), and compute-bound passes
    // (the fp64 RBF kernel scales linearly to the fast count) gain while
    // bandwidth-bound passes keep at least their old width. Undetectable,
    // homogeneous, or disabled topologies keep 8, and
    // `USDGEN_THREAD_LIMIT` above wins and skips all of this.
    return std::max(8, FastCoreCount());
}


/// The input of a node with none, and the output of a node not run yet.
UsdGenElementSet const *OutputOrEmpty(UsdGenGraph const &graph, UsdGenCompiledNode const &node)
{
    return node.output ? node.output.Get() : &graph.EmptyOutput();
}

bool HasPrefix(std::string const &s, char const *prefix)
{
    return s.rfind(prefix, 0) == 0;
}

/// Legacy first-producer layouts by name, for ops still registered through
/// the legacy adapter (no declared output planes). Text compares only; S4-4
/// (outputPlanes) and 4a-1 (emittedPlanes) delete it.
UsdGenPlane LegacyOutputPlane(TfToken const &name)
{
    std::string const &s = name.GetString();
    auto const &tk = UsdGenTokens();
    UsdGenPlane plane;
    plane.name = name;
    plane.interpolation = tk.uniform;
    plane.type = tk.float_;
    if (s == "guideIndex" || s == "partId" || HasPrefix(s, "clumpId_") ||
        HasPrefix(s, "clumpCenterId_"))
        plane.type = tk.int_;
    if (HasPrefix(s, "clumpCenter_")) plane.arity = 3;
    if (HasPrefix(s, "clumpCenterId_")) plane.arity = 2;
    // Whole-strand identities are ints; the braid knot mask is per CV.
    if (s == "primaryGuide" || s == "nearestGuide" || s == "strandSeed" ||
        s == "strandIndex" || s == "mirrorSign")
        plane.type = tk.int_;
    if (HasPrefix(s, "clumpWeight_") || s == "braidKnotMask") plane.interpolation = tk.vertex;
    if (s == "guideIndex" || s == "guideWeight") plane.arity = 3;
    return plane;
}

/// The emitting op's declared layout for `name`, if it declares one.
UsdGenPlane const *DeclaredOutputPlane(UsdGenOpRecord const *record, TfToken const &name)
{
    if (!record) return nullptr;
    for (UsdGenPlane const &plane : record->OutputPlanes())
        if (plane.name == name) return &plane;
    return nullptr;
}

/// A first producer's layout (R24): the op's declaration, else the legacy
/// table. A later writer takes the upstream plane's layout instead.
UsdGenPlane DefaultOutputPlane(UsdGenOpRecord const *record, TfToken const &name)
{
    if (UsdGenPlane const *declared = DeclaredOutputPlane(record, name)) {
        UsdGenPlane plane;
        plane.name = name;
        plane.interpolation = declared->interpolation;
        plane.type = declared->type;
        plane.arity = declared->arity;
        return plane;
    }
    return LegacyOutputPlane(name);
}

/// A plane whose layout the emitting op fixes regardless of upstream.
bool IsFixedLayoutPlane(UsdGenOpRecord const *record, TfToken const &name)
{
    if (!record || !record->Has(kUsdGenOpFixedOutputPlaneLayout)) return false;
    if (DeclaredOutputPlane(record, name)) return true;
    std::string const &s = name.GetString();   // legacy clump families
    return HasPrefix(s, "clumpId_") || HasPrefix(s, "clumpCenter_") ||
           HasPrefix(s, "clumpCenterId_") || HasPrefix(s, "clumpWeight_");
}

/// A plane descriptor as a channel (the CurveSet converter's mapping).
UsdGenChannelDesc PlaneDesc(UsdGenPlane const &p)
{
    auto const &tk = UsdGenTokens();
    UsdGenChannelDesc d;
    d.name = p.name;
    d.arity = p.arity;
    d.level = p.interpolation == tk.vertex ? 2 : p.interpolation == tk.constant ? 0 : 1;
    d.scalar = p.type == tk.int_ ? UsdGenScalar::I32 : UsdGenScalar::F32;
    d.flags = p.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
    return d;
}

/// A dynamic (named, non-static) channel of 'table', or kUsdGenNoSlot.
UsdGenSlot DynamicSlot(UsdGenChannelTable const *table, TfToken const &name)
{
    if (!table) return kUsdGenNoSlot;
    UsdGenSlot const s = table->Find(name);
    return s != kUsdGenNoSlot && s >= table->StaticCount() ? s : kUsdGenNoSlot;
}

void const *ColumnOf(UsdGenElementSet const *set, UsdGenSlot s)
{
    return set && s != kUsdGenNoSlot && set->Has(s) ? set->View().columns[s] : nullptr;
}

uint32_t ValuesOf(UsdGenElementSet const *set, UsdGenSlot s)
{
    return set && s != kUsdGenNoSlot && set->Has(s) ? set->View().values[s] : 0u;
}

/// What one slot of a node's next output holds: nothing, a column shared
/// from another set, or a column of the node's own delta arena (zero-filled,
/// or keeping 'from''s column, which is where it already is when the arena
/// is rewritten in place).
struct ColumnPlan
{
    enum Kind : uint8_t { Absent, Share, Own };
    Kind kind = Absent;
    UsdGenElementSet const *from = nullptr;
    UsdGenSlot fromSlot = kUsdGenNoSlot;
};

ColumnPlan PlanShare(UsdGenElementSet const *set, UsdGenSlot s)
{
    ColumnPlan c;
    if (set && s != kUsdGenNoSlot && set->Has(s)) {
        c.kind = ColumnPlan::Share;
        c.from = set;
        c.fromSlot = s;
    }
    return c;
}

ColumnPlan PlanOwn(UsdGenElementSet const *keep, UsdGenSlot s)
{
    ColumnPlan c;
    c.kind = ColumnPlan::Own;
    if (keep && s != kUsdGenNoSlot && keep->Has(s)) {
        c.from = keep;
        c.fromSlot = s;
    }
    return c;
}

/// The prior output's column as it is: still the node's own when it lives in
/// the prior delta arena, else the alias it was.
ColumnPlan PlanKeep(UsdGenElementSet const *prior, UsdGenSlot s)
{
    if (!prior || s == kUsdGenNoSlot || !prior->Has(s)) return {};
    return prior->OwnsColumn(s) ? PlanOwn(prior, s) : PlanShare(prior, s);
}

/// Per-run scratch of the output preparation (commit thread; keeps capacity).
struct OutputScratch
{
    std::vector<ColumnPlan> plans;
    std::vector<UsdGenSlot> own;
    std::vector<UsdGenChannelDesc> added;
    std::vector<std::pair<TfToken, ColumnPlan>> named;   // emitted, then remapped
};

void Mix(uint64_t *h, uint64_t v)
{
    *h ^= v;
    *h *= 0x100000001b3ULL;
}

/// Seals the node's output from 'plans' over 'table', counts and offsets
/// from 'topology'. Own columns are written in one delta arena, recycled in
/// place from the prior output when its layout still fits.
void SealOutput(UsdGenCompiledNode &node, UsdGenElementSet const &topology,
                UsdGenRef<const UsdGenChannelTable> const &table, OutputScratch &S,
                uint64_t topologyVersion)
{
    UsdGenWriteSet ws = UsdGenWriteSet::Derive(UsdGenRef<const UsdGenElementSet>(&topology), table);
    S.own.clear();
    UsdGenChannelTable const &t = *table;
    for (UsdGenSlot s = 0; s < t.Size(); ++s) {
        ColumnPlan const &c = S.plans[s];
        if (c.kind == ColumnPlan::Absent) ws.Drop(s);
        else if (c.kind == ColumnPlan::Share) {
            // Derive already shares the topology set's column of the same name.
            if (c.from != &topology || c.from->Channels()[c.fromSlot].name != t[s].name ||
                !UsdGenSameLayout(c.from->Channels()[c.fromSlot], t[s]))
                ws.Share(s, *c.from, c.fromSlot);
        } else if (ws.ValueCount(s)) S.own.push_back(s);
        else ws.Drop(s);   // an empty column is an absent one
    }
    if (node.output) ws.RecycleInPlace(node.output);
    ws.Plan(UsdGenSpan<const UsdGenSlot>(S.own.data(), S.own.size()));
    for (UsdGenSlot s : S.own) {
        ColumnPlan const &c = S.plans[s];
        UsdGenSpan<std::byte> const dst = ws.Write<std::byte>(s);
        if (dst.empty()) continue;
        void const *src = ColumnOf(c.from, c.fromSlot);
        size_t const srcBytes = c.from ? size_t(ValuesOf(c.from, c.fromSlot)) *
            UsdGenScalarBytes(c.from->Channels()[c.fromSlot].scalar) : 0u;
        if (src == dst.data()) continue;   // kept in place
        if (src && srcBytes == dst.size()) std::memcpy(dst.data(), src, dst.size());
        else std::memset(dst.data(), 0, dst.size());
    }
    ws.SetVersionsExact(topologyVersion, node.valueVersion);
    node.topologyVersion = topologyVersion;
    node.output = std::move(ws).Seal();
}

/// Installs an owning capture's output as the node's output: the capture's
/// columns, plus the input's per-curve channels the capture leaves out.
void InstallCaptureOutput(UsdGenCompiledNode &node, UsdGenElementSet const *up)
{
    UsdGenElementSet const &cap = *node.captureOutput;
    // valueVersion is monotone per node: the capture's own buffer counts
    // from zero, and adopting that stamp would let a downstream capture
    // keyed on the old stamp match.
    node.valueVersion = std::max(cap.ValueVersion(), node.valueVersion);
    node.topologyVersion = ++node.topologySeq;
    UsdGenWriteSet ws = UsdGenWriteSet::Derive(node.captureOutput, {});
    if (up) {
        for (UsdGenSlot s : {UsdGenSlot(kUsdGenCurveId), UsdGenSlot(kUsdGenCurveRootPrim),
                             UsdGenSlot(kUsdGenCurveRootUV), UsdGenSlot(kUsdGenCurveRootT),
                             UsdGenSlot(kUsdGenCurveRootN), UsdGenSlot(kUsdGenCurveRootB)})
            if (!cap.Has(s) && up->Has(s)) ws.Share(s, *up, s);
    }
    ws.Plan({});
    ws.SetVersionsExact(node.topologyVersion, node.valueVersion);
    node.output = std::move(ws).Seal();
}

/// The input's set under this node's own value version (a disabled node).
void PassThroughOutput(UsdGenCompiledNode &node, UsdGenElementSet const &up)
{
    UsdGenWriteSet ws = UsdGenWriteSet::Derive(UsdGenRef<const UsdGenElementSet>(&up), {});
    ws.Plan({});
    ws.SetVersionsExact(up.TopologyVersion(), node.valueVersion);
    node.topologyVersion = up.TopologyVersion();
    node.output = std::move(ws).Seal();
}

/// Commit-thread preparation of one node's output before its evaluate
/// sweep (S24). The topology follows the input (or the owning capture);
/// per-curve channels and rest are aliases of the input's (re-aliased every
/// commit, so a stale shared column never outlives its source); a channel
/// the op WRITES is the node's own storage at full size, kept from the last
/// run (sparse sweeps leave clean chunks as they were) unless it no longer
/// fits or still aliases the input, when it starts zero-filled; a channel
/// the op passes through aliases the input's.
void PrepareNodeForEval(UsdGenGraph const &graph, UsdGenCompiledNode &node, UsdGenOp const &op,
                        UsdGenElementSet const *up, OutputScratch &S)
{
    bool const hasUp = up != nullptr;
    bool const generator = node.record->Has(kUsdGenOpGenerator);
    UsdGenElementSet const *prior = OutputOrEmpty(graph, node);
    bool const owns = hasUp && !generator && node.capture && node.captureOutput;

    // Topology FIRST: the CV count of the sizing below is the node's
    // EFFECTIVE count.
    UsdGenElementSet const *topology = prior;
    uint64_t topologyVersion = node.topologyVersion;
    if (hasUp && !generator) {
        if (!owns) {
            topology = up;
            topologyVersion = up->TopologyVersion();
        } else if (prior->Count(1) == 0) {
            // An owning capture that authored no topology (Deform) inherits it.
            topology = up;
            topologyVersion = up->TopologyVersion();
        } else if (prior->Offsets(2).empty() && !up->Offsets(2).empty() &&
                   prior->Count(1) == up->Count(1) && prior->Count(2) == up->Count(2)) {
            // FIX (P1, ragged-at-Deform): an owning capture with the input's
            // totals and no offsets of its own keeps the input's ragged layout.
            topology = up;
        }
    }

    uint32_t const nCurves = topology->Count(1);
    uint32_t const cvp = node.chunks.empty() ? 0 : node.chunks[0].stride[1];
    UsdGenSpan<const uint32_t> const offs = topology->Offsets(2);
    uint32_t const nCvs = (cvp == 0 && !offs.empty() && size_t(nCurves) < offs.size())
        ? offs[nCurves] : nCurves * cvp;

    // The channel table: the input's, plus the planes this op emits (and,
    // for a remapper, owns) -- unless a generator or an owning capture that
    // transformed the planes already fixed them.
    bool const captureOwnsPlanes = owns && node.record->Has(kUsdGenOpCaptureOwnsRestAndPlanes);
    bool const extrasFromInput = hasUp && !generator && !captureOwnsPlanes;
    UsdGenRef<const UsdGenChannelTable> table(&prior->Channels());
    S.named.clear();
    S.added.clear();
    if (extrasFromInput) {
        UsdGenChannelTable const *inT = &up->Channels();
        UsdGenChannelTable const *oldT = &prior->Channels();
        for (TfToken const &name : node.outputPrimvars) {
            UsdGenSlot const source = DynamicSlot(inT, name);
            UsdGenSlot const old = DynamicSlot(oldT, name);
            bool const nativeClump = IsFixedLayoutPlane(node.record, name);
            UsdGenChannelDesc d;
            if (nativeClump || (source == kUsdGenNoSlot && old == kUsdGenNoSlot))
                d = PlaneDesc(DefaultOutputPlane(node.record, name));
            else
                d = source != kUsdGenNoSlot ? (*inT)[source] : (*oldT)[old];
            size_t const values = size_t(d.level >= 2 ? topology->Count(2) : topology->Count(1)) *
                std::max<uint8_t>(d.arity, 1);
            // Keep only a same-layout private old output; one that aliases
            // the input's column (or no longer fits) starts fresh.
            bool const keepOld = old != kUsdGenNoSlot && UsdGenSameLayout((*oldT)[old], d) &&
                ColumnOf(prior, old) != ColumnOf(up, source) &&
                ValuesOf(prior, old) == values;
            S.added.push_back(d);
            S.named.emplace_back(name, PlanOwn(keepOld ? prior : nullptr, old));
        }
        if (node.remapsVertexPlanes) {
            for (UsdGenSlot source : inT->DynamicRun(2)) {
                UsdGenChannelDesc const &d = (*inT)[source];
                UsdGenSlot const old = DynamicSlot(oldT, d.name);
                bool const reusable = old != kUsdGenNoSlot && (*oldT)[old].level >= 2 &&
                    UsdGenSameLayout((*oldT)[old], d) &&
                    ColumnOf(prior, old) != ColumnOf(up, source) &&
                    ValuesOf(prior, old) == size_t(topology->Count(2)) * std::max<uint8_t>(d.arity, 1);
                S.added.push_back(d);
                S.named.emplace_back(d.name, PlanOwn(reusable ? prior : nullptr, old));
            }
        }
        uint64_t key = 0xcbf29ce484222325ULL;
        Mix(&key, inT->Digest());
        Mix(&key, oldT->Digest());
        for (UsdGenChannelDesc const &d : S.added) {
            Mix(&key, d.name.Hash());
            Mix(&key, uint64_t(d.level) | uint64_t(d.arity) << 8 | uint64_t(d.scalar) << 16);
        }
        if (node.channels && node.channelsInput == key) {
            table = node.channels;
        } else {
            table = inT->With(UsdGenSpan<const UsdGenChannelDesc>(S.added.data(), S.added.size()));
            if (node.channels && node.channels->Digest() == table->Digest()) table = node.channels;
            node.channels = table;
            node.channelsInput = key;
        }
    }

    UsdGenChannelTable const &t = *table;
    S.plans.assign(t.Size(), ColumnPlan{});

    // Built-in CV planes: a plane the OP WRITES is this node's own storage
    // (sharing would make the kernel write the input); a plane the op
    // passes through aliases the input's.
    uint32_t const planes = op.PlanesTouched();
    auto prep = [&](UsdGenSlot s, bool write) {
        uint32_t const have = ValuesOf(prior, s);
        if (write) {
            bool const fresh = have != nCvs ||
                (hasUp && have && ColumnOf(prior, s) == ColumnOf(up, s));
            S.plans[s] = PlanOwn(fresh ? nullptr : prior, s);
        } else if (hasUp && ValuesOf(up, s) == nCvs) {
            S.plans[s] = PlanShare(up, s);
        } else if (!hasUp && generator) {
            // Source generators own the presence of optional planes (Scatter
            // has no width; inventing one would change Grow's fallback).
            S.plans[s] = PlanKeep(prior, s);
        } else if (have != nCvs) {
            S.plans[s] = PlanOwn(nullptr, s);
        } else {
            S.plans[s] = PlanKeep(prior, s);
        }
    };
    prep(kUsdGenCurvePx, planes & UsdGenOp::kPlanePoints);
    prep(kUsdGenCurvePy, planes & UsdGenOp::kPlanePoints);
    prep(kUsdGenCurvePz, planes & UsdGenOp::kPlanePoints);
    prep(kUsdGenCurveWidth, planes & UsdGenOp::kPlaneWidths);
    prep(kUsdGenCurveHairT, planes & UsdGenOp::kPlaneHairT);

    // Rest is an immutable C3 transport channel: every topology-preserving
    // operator aliases the input's. Captures that repartition CVs (Grow,
    // Resample, GuideInterpolate, frozen Freeze) author their own.
    bool const captureAuthorsRest = owns && node.record->Has(kUsdGenOpCaptureOwnsRestAndPlanes);
    S.plans[kUsdGenCurveRest] = hasUp && !generator && !captureAuthorsRest
        ? PlanShare(up, kUsdGenCurveRest) : PlanKeep(prior, kUsdGenCurveRest);

    // Per-curve channels: aliases of the input (an owning capture keeps the
    // ones it authored and inherits the rest).
    for (UsdGenSlot s : {UsdGenSlot(kUsdGenCurveId), UsdGenSlot(kUsdGenCurveRootPrim),
                         UsdGenSlot(kUsdGenCurveRootUV), UsdGenSlot(kUsdGenCurveRootT),
                         UsdGenSlot(kUsdGenCurveRootN), UsdGenSlot(kUsdGenCurveRootB)}) {
        if (!hasUp || generator) S.plans[s] = PlanKeep(prior, s);
        else if (owns && prior->Has(s)) S.plans[s] = PlanKeep(prior, s);
        else S.plans[s] = PlanShare(up, s);
    }

    // Named planes: emitted and remapped ones are the node's own; the rest
    // alias the input's (or stay the capture's).
    for (UsdGenSlot s = t.StaticCount(); s < t.Size(); ++s) {
        TfToken const &name = t[s].name;
        if (!extrasFromInput) {
            S.plans[s] = PlanKeep(prior, DynamicSlot(&prior->Channels(), name));
            continue;
        }
        ColumnPlan const *own = nullptr;
        for (auto const &entry : S.named)
            if (entry.first == name) own = &entry.second;   // the last entry wins
        S.plans[s] = own ? *own : PlanShare(up, DynamicSlot(&up->Channels(), name));
    }
    SealOutput(node, *topology, table, S, topologyVersion);
}

UsdGenEpoch WithExternalValueIdentity(UsdGenEpoch digest,
                                      UsdGenCompiledNode const &node,
                                      UsdGenGraph const &graph)
{
    auto mix = [&digest](uint64_t value) {
        digest[0] ^= value;
        digest[0] *= 0x100000001b3ULL;
        digest[1] ^= ~value;
        digest[1] *= 0x100000001b3ULL;
    };
    TfSpan<const UsdGenResolvedReferenceValue> const references = graph.ReferenceValues();
    for (uint32_t index : node.referenceValues) {
        if (index >= references.size()) { mix(UINT64_MAX); continue; }
        mix(references[index].identity);
        mix(references[index].curveGeneration);
    }
    TfSpan<const UsdGenResolvedMapValue> const maps = graph.MapValues();
    for (uint32_t index : node.mapValues) {
        if (index >= maps.size()) { mix(UINT64_MAX - 1); continue; }
        mix(maps[index].identity);
        mix(maps[index].textureGeneration);
    }
    // Connected parameters are a capture input on this lane exactly as they
    // are on the CUDA lane: editing usdGen:expr:source, re-pointing a
    // connection or moving the geometry the expression samples changes this
    // digest, which re-captures the node and dirties its chunks.
    mix(node.expressions.Digest());
    return digest;
}
/// Group dirty bits: the roles a run wrote (UsdGenChunkGroup::dirtyRoles).
constexpr uint32_t kPositionDirty = 1u << uint32_t(UsdGenChannelRole::Position);
constexpr uint32_t kWidthDirty = 1u << uint32_t(UsdGenChannelRole::Width);
/// An extent slot no writer filled (range.live).
constexpr uint32_t kExtentUnwritten = 0xffffffffu;

struct NodeSweepPayload
{
    UsdGenCompiledNode *node;
    UsdGenGraph const *graph;
    UsdGenEvalContext ctx;          // node-local copy (params/seed already set)
    UsdGenOp const *op;
    UsdGenCompiledNode const *up;  // null when the node has no input
    UsdGenCompiledNode const *up2; // ordered second input for binary kernels
    bool evalAll;
    uint32_t planes;
    std::atomic<uint32_t> failureCode{0};
    bool deferCompletion = false;
    std::vector<GfVec3d> pointScratch;
    std::vector<double> scalarScratch;
    std::vector<uint8_t> didEval;  // one byte per chunk: a bit-packed
                                   // vector<bool> makes 64 workers share one
                                   // word — cache-line ping-pong (03 §5.4
                                   // post-sweep bookkeeping lives on the
                                   // commit thread; per-chunk flags must not
                                   // false-share).
    // Invoked after this chunk's writes have completed. Installed only on the
    // terminal node of an opt-in progressive CPU cook.
    std::function<void(size_t)> onChunkCompleted;
    // Per-chunk extent slots for the recording points-writer (one entry per
    // chunk of the gate-time partition), or null for every other node.
    // SweepChunk hands slot [index] to the kernel; slots are disjoint per
    // (node, chunk), so concurrent sweeps never share one.
    UsdGenChunkExtent *extentBase = nullptr;
    size_t extentCap = 0;
    // What the chunk view is built from: the node's sealed output and its
    // inputs, viewed on the commit thread; the per-chunk slot tables are
    // chunk-major and keep their capacity.
    UsdGenElementSetView outView{}, inView{}, in2View{};
    UsdGenElementSetView const *outSet = nullptr, *inSet = nullptr, *in2Set = nullptr;
    std::array<UsdGenSlot, kUsdGenMaxExtraPlaneSlots> emitted{}, emittedIn{}, consumed{};
    uint16_t emittedCount = 0, consumedCount = 0, remappedCount = 0;
    UsdGenSlot const *remapped = nullptr;
    std::vector<void *> outTables;
    std::vector<void const *> inTables, in2Tables;
    uint32_t outSlots = 0, inSlots = 0, in2Slots = 0;
    size_t tableChunks = 0;
    bool pointsIn = false;   // input points go to point writers
    bool widthIn = false;    // and input px alone to width writers
};

/// Chunk-local bases of every slot of 'set': level-2 channels at cvBase,
/// the others at curveBase (the old view's rule, constant channels included).
void FillSlotTable(UsdGenElementSetView const &set, uint32_t slots, size_t curveBase,
                   size_t cvBase, void const **table)
{
    for (uint32_t s = 0; s < slots; ++s) {
        void const *column = set.columns[s];
        if (!column) { table[s] = nullptr; continue; }
        UsdGenChannelDesc const &d = (*set.channels)[s];
        size_t const element = d.level >= 2 ? cvBase : curveBase;
        table[s] = static_cast<char const *>(column) +
                   element * d.arity * UsdGenScalarBytes(d.scalar);
    }
}


// A prepared node job owns the small vectors whose addresses are published in
// its EvalContext.  Preparation happens on the commit thread; arena workers
// only consume the immutable context and write this node's disjoint chunks.
std::string NodeLabel(UsdGenCompiledNode const &node)
{
    return node.desc ? node.desc->path.GetName() : std::string("?");
}

struct NodeExecution
{
    std::vector<UsdGenElementSetView> inputSets;   // views of the input nodes' outputs
    std::vector<UsdGenReferenceSet const *> referenceSets;
    std::vector<UsdGenResolvedReferenceValue const *> resolvedReferences;
    std::vector<UsdGenResolvedMapValue const *> resolvedMaps;
    NodeSweepPayload sweep{};
    char const *recaptureReason = nullptr;   // why reCaptured, for USDGEN_SCHEDULE
    bool reCaptured = false;
    bool expressionsChanged = false;
    bool shouldSweep = false;
    bool sweptDirect = false;   // capture wrote the outputs; sweep skipped
    bool evalAll = false;
    bool anyChunkDirty = false;
    std::chrono::steady_clock::time_point captureStart;
    std::chrono::steady_clock::time_point captureEnd;
    std::chrono::steady_clock::time_point evaluationStart;

    // Back to a fresh job, keeping every vector's capacity.
    void Reset()
    {
        inputSets.clear();
        referenceSets.clear();
        resolvedReferences.clear();
        resolvedMaps.clear();
        NodeSweepPayload &s = sweep;
        s.node = nullptr;
        s.graph = nullptr;
        s.ctx = UsdGenEvalContext{};
        s.op = nullptr;
        s.up = s.up2 = nullptr;
        s.evalAll = false;
        s.planes = 0;
        s.failureCode.store(0, std::memory_order_relaxed);
        s.deferCompletion = false;
        s.pointScratch.clear();
        s.scalarScratch.clear();
        s.didEval.clear();
        s.onChunkCompleted = nullptr;
        s.extentBase = nullptr;
        s.extentCap = 0;
        s.outSet = s.inSet = s.in2Set = nullptr;
        s.outView = s.inView = s.in2View = UsdGenElementSetView{};
        s.emittedCount = s.consumedCount = s.remappedCount = 0;
        s.remapped = nullptr;
        s.outSlots = s.inSlots = s.in2Slots = 0;
        s.tableChunks = 0;
        s.pointsIn = s.widthIn = false;
        recaptureReason = nullptr;
        reCaptured = expressionsChanged = shouldSweep = sweptDirect = false;
        evalAll = anyChunkDirty = false;
        captureStart = captureEnd = evaluationStart = {};
    }
};

struct ChunkExecution
{
    NodeSweepPayload *payload = nullptr;
    size_t chunk = 0;
};

// Everything the terminal node's per-chunk completion needs, so the
// std::function holds one pointer and stays in its small buffer.
struct TerminalTiles
{
    UsdGenGraph const *graph = nullptr;
    UsdGenCompiledNode const *node = nullptr;
    std::atomic<uint32_t> *remaining = nullptr;
    int nTiles = 0;
    UsdGenScheduler::TileCompleted const *completed = nullptr;
};

}  // namespace

struct UsdGenRunScratch
{
    std::vector<UsdGenChunkExtent> extents;
    std::vector<char> tileTouched, poseSpace, completed;
    std::vector<int> frontier;
    std::vector<std::unique_ptr<NodeExecution>> jobs;   // grows, never shrinks
    std::vector<ChunkExecution> work;
    std::unique_ptr<std::atomic<uint32_t>[]> terminalRemaining;
    size_t terminalCapacity = 0;
    TerminalTiles terminal;
    std::vector<UsdGenNodeRunStats> nodeStats;
    OutputScratch output;
    std::atomic<bool> running{false};
};

namespace {

void SweepChunk(size_t index, void *payload);

void SweepPreparedChunk(size_t index, void *payload)
{
    auto const &work = static_cast<ChunkExecution *>(payload)[index];
    SweepChunk(work.chunk, work.payload);
}

void SweepChunk(size_t index, void *payload)
{
    NodeSweepPayload &pl = *static_cast<NodeSweepPayload *>(payload);
    UsdGenCompiledNode &node = *pl.node;
    UsdGenChunkRange const &cd = node.chunks[index];
    if (cd.count[0] == 0) return;
    bool const chunkDirty =
        node.chunkDirty.empty()
            ? true   // no dirty bookkeeping for this partition: evaluate (never skip blindly)
            : (node.chunkDirty[index] & UsdGenDirtyParameter) != 0;
    if (!pl.evalAll && !chunkDirty) return;

    // Ragged offsets: count[1] + 1 entries, absolute; null on the uniform path.
    uint32_t const *offs = pl.outView.offsets[2];
    size_t const nOffs = offs ? size_t(pl.outView.count[1]) + 1 : 0;
    uint32_t const *upOffs = pl.inView.offsets[2];
    size_t const nUpOffs = upOffs ? size_t(pl.inView.count[1]) + 1 : 0;
    UsdGenChunkRange const *upC = (pl.up && pl.up->chunks.size() == node.chunks.size())
        ? &pl.up->chunks[index] : nullptr;
    UsdGenChunkRange const *upC2 = (pl.up2 && pl.up2->chunks.size() == node.chunks.size())
        ? &pl.up2->chunks[index] : nullptr;

    uint32_t nCvs = cd.count[0] * cd.stride[1];
    if (cd.stride[1] == 0 && nOffs) {
        // Ragged span [firstCurve, firstCurve+curveCount) from topology offsets.
        size_t const f = cd.first[0];
        if (f + cd.count[0] < nOffs)
            nCvs = uint32_t(offs[f + cd.count[0]] - offs[f]);
    }
    size_t const base = cd.first[1];
    size_t const upBase = upC ? upC->first[1] : 0;

    UsdGenChunkRange range = cd;
    range.count[1] = nCvs;
    UsdGenChunkView chunk{};
    chunk.range = &range;
    // Ragged offsets stay absolute, pointer at the chunk's first curve; null
    // on the uniform fast path.
    if (cd.stride[1] == 0 && nOffs && size_t(cd.first[0]) < nOffs)
        chunk.offsets[2] = offs + cd.first[0];
    if (upC && upC->stride[1] == 0 && nUpOffs && size_t(upC->first[0]) < nUpOffs)
        chunk.inOffsets[2] = upOffs + upC->first[0];
    chunk.inFirst[1] = upC ? upC->first[0] : cd.first[0];
    chunk.inFirst[2] = static_cast<uint32_t>(upBase);
    chunk.inStride[2] = upC ? upC->stride[1] : 0;
    if (pl.outSet && index < pl.tableChunks) {
        void **out = pl.outTables.data() + index * pl.outSlots;
        FillSlotTable(*pl.outSet, pl.outSlots, cd.first[0], base,
                      const_cast<void const **>(out));
        chunk.out = out;
        chunk.channels = pl.outSet->channels;
    }
    if (pl.inSet && index < pl.tableChunks) {
        void const **in = pl.inTables.data() + index * pl.inSlots;
        FillSlotTable(*pl.inSet, pl.inSlots, chunk.inFirst[1], upBase, in);
        // Input points go to point writers; input px alone to width writers.
        if (!pl.pointsIn && pl.inSlots > kUsdGenCurvePz) {
            if (!pl.widthIn) in[kUsdGenCurvePx] = nullptr;
            in[kUsdGenCurvePy] = in[kUsdGenCurvePz] = nullptr;
        }
        chunk.in = in;
        chunk.inChannels = pl.inSet->channels;
    }
    if (pl.in2Set && upC2 && index < pl.tableChunks) {
        void const **in2 = pl.in2Tables.data() + index * pl.in2Slots;
        FillSlotTable(*pl.in2Set, pl.in2Slots, upC2->first[0], upC2->first[1], in2);
        chunk.in2 = in2;
    }
    chunk.emitted = pl.emitted.data();
    chunk.emittedIn = pl.emittedIn.data();
    chunk.consumed = pl.consumed.data();
    chunk.remapped = pl.remapped;
    chunk.emittedCount = pl.emittedCount;
    chunk.consumedCount = pl.consumedCount;
    chunk.remappedCount = pl.remappedCount;
    chunk.pointScratch = pl.pointScratch.empty() ? nullptr : pl.pointScratch.data() + base;
    chunk.scalarScratch = pl.scalarScratch.empty() ? nullptr : pl.scalarScratch.data() + base;
    // The recording writer's slot, when the gate armed one and the chunk
    // still fits the gate-time partition (a mid-run repartition grows the
    // partition past the scratch; the interleave size check falls back).
    chunk.extentSlot = (pl.extentBase && index < pl.extentCap) ? pl.extentBase + index : nullptr;
    // Capture-payload views: the magnitude LUT comes from the node's capture.
    chunk.rampLut = node.capture ? node.capture->RampLut() : nullptr;

    pl.op->Evaluate(pl.ctx, *node.capture, chunk);

    pl.didEval[index] = 1;
    if (pl.onChunkCompleted && !pl.deferCompletion) pl.onChunkCompleted(index);
}

// ---------------------------------------------------------------------------
// Tile-interleave body (03 §5.4 step 5): per-tile extents + dirty flags.
// Tiles are disjoint memory; the pass parallelises in the private arena.
// ---------------------------------------------------------------------------

struct InterleavePayload
{
    UsdGenGraph *graph;
    UsdGenCompiledNode const *tn;
    UsdGenCurveSetView term;
    std::vector<char> *touched;
    bool widthsFlag;
    // Fused extents: per-chunk records from the recording sweep, consumed
    // in chunk order. Null unless every validation check passed.
    UsdGenChunkExtent const *extents = nullptr;
};

void InterleaveTile(size_t index, void *payload)
{
    InterleavePayload &pl = *static_cast<InterleavePayload *>(payload);
    UsdGenChunkGroup &tv =
        const_cast<UsdGenChunkGroup &>(pl.graph->Groups()[index]);
    // Tile dirty flags are per-run observations (03 §5.4 step 5), not sticky
    // state: clear first so a run that evaluates no chunk of this tile reports
    // "nothing changed" instead of replaying the previous run's flag.
    tv.dirtyRoles = 0;
    if (!(*pl.touched)[index]) return;

    // Terminal node owns the chunk partition the tiles index into.
    if (pl.tn->chunks.size() < tv.firstChunk + tv.chunkCount) return;

    UsdGenCurveSetView const &term = pl.term;
    GfRange3f extent;
    uint64_t liveCurves = 0, liveCvs = 0;
    float const *tpx = term.px.empty() ? nullptr : term.px.cdata();
    float const *tpy = term.py.empty() ? nullptr : term.py.cdata();
    float const *tpz = term.pz.empty() ? nullptr : term.pz.cdata();
    size_t const nPx = term.px.size();
    bool const ragged = !term.cvOffsets.empty();
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkRange const &cd = pl.tn->chunks[tv.firstChunk + i];
        if (pl.extents) {
            // Fused path: the counts below are the point path's statements
            // verbatim (same inputs, same sums); the extent unions the
            // recorded per-chunk extents in chunk order, which reproduces
            // the sequential pass bitwise for NaN-free points (the gate
            // admits only deform-recorded runs, and deform outputs cannot
            // carry NaN — see the contract on UsdGenChunkView::extentSlot).
            if (ragged && cd.stride[1] == 0) {
                for (uint32_t c = 0; c < cd.live; ++c) {
                    size_t const g = size_t(cd.first[0]) + c;
                    if (g + 1 >= term.cvOffsets.size()) break;
                    uint32_t const p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                    uint32_t const len =
                        static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
                    ++liveCurves;
                    liveCvs += len;
                }
            } else {
                liveCurves += cd.live;
                liveCvs += uint64_t(cd.live) * cd.stride[1];
            }
            UsdGenChunkExtent const &e = pl.extents[tv.firstChunk + i];
            extent.UnionWith(GfRange3f(GfVec3f(e.min[0], e.min[1], e.min[2]),
                                       GfVec3f(e.max[0], e.max[1], e.max[2])));
            continue;
        }
        // Ragged path (03 §1.3): per-curve CV spans come from cvOffsets, not
        // liveCount*cvCount (cvCount == 0 on the ragged path). g indexes the
        // absolute curve so cvOffsets[g] is valid.
        if (ragged && cd.stride[1] == 0) {
            for (uint32_t c = 0; c < cd.live; ++c) {
                size_t const g = size_t(cd.first[0]) + c;
                if (g + 1 >= term.cvOffsets.size()) break;
                uint32_t const p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                uint32_t const len =
                    static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
                ++liveCurves;
                liveCvs += len;
                if (!tpx || !tpy || !tpz) continue;
                for (uint32_t v = 0; v < len; ++v) {
                    size_t const p = size_t(p0) + v;
                    if (p >= nPx) break;
                    extent.ExtendBy(GfVec3f(tpx[p], tpy[p], tpz[p]));
                }
            }
            continue;
        }
        liveCurves += cd.live;
        liveCvs += uint64_t(cd.live) * cd.stride[1];
        if (!tpx || !tpy || !tpz) continue;
        for (uint32_t c = 0; c < cd.live; ++c) {
            size_t const o = size_t(cd.first[1]) + size_t(c) * cd.stride[1];
            for (uint32_t v = 0; v < cd.stride[1]; ++v) {
                size_t const p = o + v;
                if (p >= nPx) break;
                extent.ExtendBy(GfVec3f(tpx[p], tpy[p], tpz[p]));
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        tv.extentMin[k] = extent.GetMin()[k];
        tv.extentMax[k] = extent.GetMax()[k];
    }
    tv.live[0] = static_cast<uint32_t>(liveCurves);
    tv.live[1] = static_cast<uint32_t>(liveCvs);
    tv.dirtyRoles = kPositionDirty | (pl.widthsFlag ? kWidthDirty : 0u);
}

}  // namespace

UsdGenScheduler::UsdGenScheduler(int threadLimit)
    : _threadLimit(ResolveThreadLimit(threadLimit)),
      _pool(_threadLimit)
{
}

UsdGenScheduler::~UsdGenScheduler() = default;

int UsdGenScheduler::ThreadLimit() const noexcept { return _threadLimit; }

void UsdGenWorkDispatcher::ParallelFor(
    size_t count, void (*body)(size_t, void *), void *payload, size_t claimChunk)
{
    // Every parallel region runs over the scheduler's worker pool (never
    // pxr work::, which honours the process-global PXR_WORK_THREAD_LIMIT
    // and would serialise under PXR_WORK_THREAD_LIMIT=1, 03 §5.3 caveat).
    // The pool replaced the private TBB arena's parallel_for here: same
    // body/count contract, without the arena's ~0.2ms per-region wakeup.
    if (!_pool) {
        for (size_t i = 0; i < count; ++i) body(i, payload);
        return;
    }
    _pool->ParallelFor(count, body, payload, claimChunk);
}

UsdGenWorkDispatcher UsdGenScheduler::MakeWorkDispatcher() const
{
    return UsdGenWorkDispatcher(const_cast<UsdGenWorkerPool *>(&_pool));
}

int CalibrateThreads()
{
    // M1: the one-shot sweep of 03 §5.3 is deferred; the 8-thread knee
    // (EV-001/EV-008) is the documented default.
    return 8;
}

UsdGenRunResult UsdGenScheduler::Run(
    UsdGenGraph &graph,
    UsdGenEvalContext const &evalCtx,
    uint64_t generationRequested,
    TileCompleted tileCompleted,
    Superseded const &superseded)
{
    TF_UNUSED(generationRequested);
    if (!_scratch) _scratch = std::make_unique<UsdGenRunScratch>();
    if (_scratch->running.exchange(true, std::memory_order_acquire)) {
        TF_CODING_ERROR("UsdGenScheduler::Run is not reentrant");
        UsdGenRunResult refused;
        refused.diagnostics.Error("UsdGen: scheduler is already running");
        return refused;
    }
    _scratch->nodeStats.clear();
    UsdGenCookTokenCheck const tokenCheck;
    UsdGenRunResult result = _Run(graph, evalCtx, tileCompleted, superseded);
    UsdGenCookInternFails(tokenCheck, &result.diagnostics);
    result.nodeStats = TfSpan<const UsdGenNodeRunStats>(_scratch->nodeStats);
    _scratch->running.store(false, std::memory_order_release);
    // Cook-lane drain: what this cook retired is freed by the cook, not left
    // to an idle worker (a no-op on a pinned thread or with nothing due).
    UsdGenEpochDrain();
    return result;
}

UsdGenRunResult UsdGenScheduler::_Run(
    UsdGenGraph &graph,
    UsdGenEvalContext const &evalCtx,
    TileCompleted const &tileCompleted,
    Superseded const &superseded)
{
    TRACE_FUNCTION();
    UsdGenRunScratch &S = *_scratch;

    UsdGenRunResult result;
    if (graph.NodeCount() == 0) return result;

    // Fused tile extents: when the deepest points-writer on the terminal
    // chain is a deform, the deform sweep records each chunk's extent over
    // the floats it stores and the interleave unions the slots instead of
    // re-reading every point (deform capture success pins the outputs
    // NaN-free, and NaN-free chunked union reproduces the sequential pass
    // bitwise). Upstream writers do not matter: the deform consumes their
    // outputs, so its stores are the terminal points. Anything else — a
    // deeper non-deform writer, a partial sweep, unaliased planes, a
    // mid-run repartition — falls back to the point pass, so the union
    // only runs when every check against the current buffers passes. A
    // deform that does not record pays one predictable branch per CV, so
    // unselected graphs keep their old timing.
    int extentWriter = -1;
    size_t extentChunks = 0;
    {
        bool writerSeen = false;
        UsdGenNodeId id = graph.TerminalNodeId();
        for (int guard = 0;
             id != kUsdGenInvalidNode && id < UsdGenNodeId(graph.NodeCount()) &&
             guard <= graph.NodeCount();
             ++guard) {
            UsdGenCompiledNode const &n = graph.Node(id);
            if (!writerSeen && n.op &&
                (n.op->PlanesTouched() & UsdGenOp::kPlanePoints) != 0) {
                writerSeen = true;
                if (n.record && n.record->Has(kUsdGenOpRecordsChunkExtents))
                    extentWriter = int(id);
            }
            id = n.input;
        }
        if (extentWriter < 0) {
            extentWriter = -1;
        } else {
            extentChunks = graph.Node(graph.TerminalNodeId()).chunks.size();
            if (extentChunks == 0) extentWriter = -1;
        }
    }
    // Slots start unwritten (live matches no chunk), so the validation
    // below rejects any slot no writer filled.
    std::vector<UsdGenChunkExtent> &extentScratch = S.extents;
    extentScratch.clear();
    if (extentWriter >= 0) {
        UsdGenChunkExtent unwritten{};
        unwritten.range.live = kExtentUnwritten;
        extentScratch.resize(extentChunks, unwritten);
    }
    bool extentSwept = false;

    UsdGenDiagnostics aggregated;
    auto dispatcher = MakeWorkDispatcher();

    int nTiles = graph.NumGroups();
    std::vector<char> &tileTouched = S.tileTouched;
    tileTouched.assign(size_t(nTiles), 0);
    bool transactionalRun=false;
    for (int pos=0;pos<graph.NodeCount();++pos) {
        UsdGenCompiledNode const &node=graph.Node(pos);
        // Configure is structural, while effective enabled can be driven by
        // a value expression later in preparation. Conservatively transact
        // any graph containing a fallible remapper.
        transactionalRun |= node.op && node.remapsVertexPlanes;
    }

    // Operators evaluate in Default space until a deform: a deform, and every
    // node reading one's output, sees the posed surfaces (graph.Desc()); every
    // other node sees their Default-time rest (graph.RestDesc()), so a posed
    // surface re-captures only the deform and what follows it. Nodes are in
    // topological order; the loop repeats only if an input ever follows its
    // reader.
    std::vector<char> &poseSpace = S.poseSpace;
    poseSpace.assign(static_cast<size_t>(graph.NodeCount()), 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (int pos = 0; pos < graph.NodeCount(); ++pos) {
            if (poseSpace[size_t(pos)]) continue;
            UsdGenCompiledNode const &node = graph.Node(pos);
            bool posed = node.op && node.record && node.record->Has(kUsdGenOpPosesSpace);
            for (UsdGenNodeId input : node.inputs)
                posed = posed || (input < UsdGenNodeId(graph.NodeCount()) && poseSpace[input]);
            if (posed) { poseSpace[size_t(pos)] = 1; changed = true; }
        }
    }
    auto spaceDesc = [&](int pos) -> UsdGenGraphDesc const & {
        return poseSpace[size_t(pos)] ? graph.Desc() : graph.RestDesc();
    };

    // Each frontier is dependency-ready and therefore may share one flattened
    // chunk sweep. Capture, COW preparation, repartition and bookkeeping stay
    // on this commit thread; arena workers only execute SweepChunk.
    std::vector<char> &completed = S.completed;
    completed.assign(static_cast<size_t>(graph.NodeCount()), 0);
    auto prepareNode = [&](int pos, NodeExecution *job) -> bool {
        UsdGenCompiledNode &node = graph.Node(pos);
        UsdGenOp &op = *node.op;
        TRACE_SCOPE("usdGen prepare");   // a dynamic name interns a token per node
        UsdGenDiagnostics nodeDiag;
        bool const hasUp = !node.inputs.empty();
        UsdGenElementSet const *up = hasUp
            ? OutputOrEmpty(graph, graph.Node(node.inputs.front())) : nullptr;
        job->inputSets.reserve(node.inputs.size());
        for (UsdGenNodeId input : node.inputs) {
            if (input >= static_cast<UsdGenNodeId>(graph.NodeCount())) {
                nodeDiag.Error("UsdGen: invalid compiled geometry input on '" +
                               node.desc->path.GetString() + "'");
                break;
            }
            job->inputSets.push_back(OutputOrEmpty(graph, graph.Node(input))->View());
        }
        UsdGenOpInputs inputs;
        inputs.sets = job->inputSets.empty() ? nullptr : job->inputSets.data();
        inputs.count = static_cast<uint32_t>(job->inputSets.size());
        job->captureStart = std::chrono::steady_clock::now();

        TfSpan<const UsdGenResolvedReferenceValue> const graphReferences =
            graph.ReferenceValues();
        TfSpan<const UsdGenResolvedMapValue> const graphMaps = graph.MapValues();
        job->referenceSets.reserve(node.referenceValues.size());
        job->resolvedReferences.reserve(node.referenceValues.size());
        for (uint32_t value : node.referenceValues) {
            if (value >= graphReferences.size() || !graphReferences[value].value) {
                nodeDiag.Error("UsdGen: missing compiled reference value on '" +
                               node.desc->path.GetString() + "'");
                break;
            }
            job->resolvedReferences.push_back(&graphReferences[value]);
            job->referenceSets.push_back(graphReferences[value].value.get());
        }
        job->resolvedMaps.reserve(node.mapValues.size());
        for (uint32_t value : node.mapValues) {
            if (value >= graphMaps.size()) {
                nodeDiag.Error("UsdGen: missing compiled map value on '" +
                               node.desc->path.GetString() + "'");
                break;
            }
            job->resolvedMaps.push_back(&graphMaps[value]);
        }

        UsdGenCaptureContext cctx;
        cctx.desc = &spaceDesc(pos);
        cctx.params = &node.paramView;
        cctx.references = job->referenceSets.empty() ? nullptr
            : job->referenceSets.data();
        cctx.resolvedReferences = job->resolvedReferences.empty() ? nullptr
            : job->resolvedReferences.data();
        cctx.referenceCount = static_cast<uint32_t>(job->resolvedReferences.size());
        cctx.maps = job->resolvedMaps.empty() ? nullptr : job->resolvedMaps.data();
        cctx.mapCount = static_cast<uint32_t>(job->resolvedMaps.size());
        cctx.mapBindings = node.mapBindingRefs.empty() ? nullptr
            : node.mapBindingRefs.data();
        cctx.mapBindingCount = static_cast<uint32_t>(node.mapBindingRefs.size());
        cctx.surface = node.hasSurface ? node.surface : 0;
        cctx.seed = node.desc ? static_cast<uint32_t>(node.desc->seed) : 0;
        cctx.upstreamGeneration = hasUp ? up->TopologyVersion() : 0;
        cctx.inputs = inputs;
        cctx.dispatcher = &dispatcher;
        cctx.diag = &nodeDiag;
        // Capture-direct output: the node's previous-run point planes for
        // an op that writes its sweep's result during capture, plus the
        // chunk partition and this run's fused slots for extent recording
        // (slots for the armed writer only). Captures verify counts,
        // aliasing, and nesting before writing anything, and the
        // interleave revalidates recorded spans, so anything unexpected
        // fails closed into the sweep path. Progressive cooks keep the
        // sweep (per-chunk completion), so they offer no direct planes.
        if (!tileCompleted) {
            // Offered only while the previous output's point columns are
            // the node's own (its delta arena): the arena preparation below
            // keeps them in place, or copies them when it moves.
            UsdGenElementSet const *prior = node.output.Get();
            uint32_t const cvs = prior ? prior->Count(2) : 0u;
            bool const outSized = cvs > 0 &&
                prior->OwnsColumn(kUsdGenCurvePx) && prior->OwnsColumn(kUsdGenCurvePy) &&
                prior->OwnsColumn(kUsdGenCurvePz) &&
                prior->Column<float>(kUsdGenCurvePx).size() == cvs &&
                prior->Column<float>(kUsdGenCurvePy).size() == cvs &&
                prior->Column<float>(kUsdGenCurvePz).size() == cvs;
            auto column = [&](UsdGenSlot s) {
                return outSized ? const_cast<float *>(prior->Column<float>(s).data()) : nullptr;
            };
            cctx.outPx = column(kUsdGenCurvePx);
            cctx.outPy = column(kUsdGenCurvePy);
            cctx.outPz = column(kUsdGenCurvePz);
            cctx.outPlaneCvs = outSized ? size_t(cvs) : 0;
            cctx.chunks = node.chunks.empty() ? nullptr : node.chunks.data();
            cctx.chunkCount = node.chunks.size();
            if (pos == extentWriter && extentWriter >= 0) {
                cctx.chunkExtentSlots = extentScratch.data();
                cctx.chunkExtentCount = extentChunks;
            }
        }

        // Connected parameters are evaluated ONCE per cook, here, over this
        // node's INPUT geometry and before its capture identity is taken --
        // the same ordering CudaParameterEvaluator has on the device lane.
        {
            double const rate = graph.Desc().timeCodesPerSecond;
            double const seconds = std::isfinite(rate) && rate > 0.0
                ? evalCtx.time / rate : evalCtx.time;
            std::vector<std::string> expressionErrors;
            if (!node.expressions.Evaluate(spaceDesc(pos), *node.desc,
                                           UsdGenCurveSetView::Of(up ? up->View() : UsdGenElementSetView{}),
                                           evalCtx.time, seconds, cctx.seed,
                                           &job->expressionsChanged,
                                           &expressionErrors))
                for (auto const &message : expressionErrors)
                    nodeDiag.Error("UsdGen: " + message);
            for (auto const &message : node.expressions.TakeWarnings())
                nodeDiag.Warn("UsdGen: " + node.desc->path.GetString() + ": " + message);
            // A groom-domain usdGen:enabled drives the node's own gate. With
            // nothing connected the authored value stands.
            double enabled = 0.0;
            node.enabled = node.paramView.GroomExpressionValue(node.record->EnabledSlot(), &enabled)
                ? enabled != 0.0 : node.desc->enabled;
        }

        // 02 §6.3: a disabled operator contributes nothing of its own. A
        // disabled GENERATOR publishes an empty curve set, so the description
        // publishes no curves; every other disabled operator passes its
        // upstream through as a CoW alias in prepareEvaluation below.
        if (!node.enabled) {
            job->reCaptured = false;
            node.captureNeeded = false;
            if (node.record->Has(kUsdGenOpGenerator) && node.output &&
                (node.output->Count(1) != 0 || node.output->Count(2) != 0)) {
                node.valueVersion = 0;
                node.topologyVersion = ++node.topologySeq;
                UsdGenWriteSet ws = UsdGenWriteSet::Derive(
                    UsdGenRef<const UsdGenElementSet>(&graph.EmptyOutput()), {});
                ws.Plan({});
                ws.SetVersionsExact(node.topologyVersion, 0);
                node.output = std::move(ws).Seal();
                node.captureOutput.Reset();
                node.capture.reset();
                if (graph.Repartition(0, 0)) result.topologyChanged = true;
            }
            for (auto const &e : nodeDiag.errors) aggregated.Error(e);
            for (auto const &w : nodeDiag.warnings) aggregated.Warn(w);
            if (aggregated.HasErrors()) return false;
            job->captureEnd = std::chrono::steady_clock::now();
            job->evalAll = false;
            return true;
        }

        if (op.ReadsTime(node.paramView) &&
            (!node.hasEvaluationTime || node.lastEvaluationTime != evalCtx.time ||
             node.lastTimeCodesPerSecond != graph.Desc().timeCodesPerSecond))
            graph.MarkNode(static_cast<UsdGenNodeId>(pos), UsdGenDirtyParameter);
        node.hasEvaluationTime = true;
        node.lastEvaluationTime = evalCtx.time;
        node.lastTimeCodesPerSecond = graph.Desc().timeCodesPerSecond;

        UsdGenEpoch const captureIdentity =
            WithExternalValueIdentity(op.CaptureDigest(cctx), node, graph);
        // First reason wins; the trace reports it.
        job->recaptureReason =
            !node.capture ? "no capture yet" :
            node.captureNeeded ? "compiler requested" :
            captureIdentity != node.captureEpoch ? "capture digest moved" :
            !node.capture->ValidForTopology(inputs.Primary()) ? "input changed" : nullptr;
        job->reCaptured = job->recaptureReason != nullptr;
        if (job->reCaptured) {
            TRACE_SCOPE("usdGen capture");
            TRACE_COUNTER_DELTA("usdGen nodes captured", 1);
            cctx.previousCapture = node.capture.get();
            // Recycled through retirement: the spare is the capture replaced
            // one re-capture ago, so the newer generation no longer shares
            // its arrays and Recycle() keeps their storage.
            std::unique_ptr<UsdGenCapture> cap;
            if (node.spareCapture && node.spareCapture->Recycle())
                cap = std::move(node.spareCapture);
            else {
                node.spareCapture.reset();
                cap = op.CreateCapture();
            }
            if (cap && op.Capture(cctx, inputs, cap.get(), &nodeDiag)) {
                // An owning capture's output, converted once per capture.
                node.captureOutput = cap->Output();
                if (node.captureOutput)
                    InstallCaptureOutput(node, up);
                std::unique_ptr<UsdGenCapture> replaced = std::move(node.capture);
                node.capture = std::move(cap);
                if (replaced && replaced->Recyclable())
                    node.spareCapture = std::move(replaced);
            } else if (cap) {
                nodeDiag.Error(std::string("UsdGen: capture failed on '") +
                               node.desc->path.GetText() + "'");
            }
            node.captureEpoch = captureIdentity;
            node.captureNeeded = false;

            // Generators and CV-repartitioning stylers (Resample) publish a new
            // CV layout from Capture; the chunk plan must follow it. CurveCount
            // stylers (Length) do not own a buffer and keep the upstream plan.
            // Repartition unconditionally: its own keep-check is the single
            // comparator (a pre-check on chunk COUNT misses 20->10 curves at
            // 1 chunk, leaving stale spans that sweep out of bounds), and it
            // early-outs with no writes and no topology change when the layout
            // already matches. No totalCurves > 0 gate either: a generator
            // re-capturing to ZERO curves must still repartition, or its stale
            // chunk plan sweeps dead chunks against empty buffers.
            if (node.record->Has(kUsdGenOpGenerator) ||
                node.topoFx == UsdGenTopoFx::CvCount) {
                UsdGenElementSet const &out = graph.NodeOutput(node.id);
                int const total = static_cast<int>(out.Count(1));
                int const cvp = int(out.Count(2) / std::max<uint32_t>(1, out.Count(1)));
                if (graph.Repartition(uint32_t(total), uint32_t(cvp)))
                    result.topologyChanged = true;
            } else if (job->reCaptured && node.chunks.empty() && hasUp) {
                UsdGenCompiledNode const &upN = graph.Node(node.input);
                uint32_t const upCurves = graph.NodeOutput(node.input).Count(1);
                if (!upN.chunks.empty() && upCurves > 0 &&
                    graph.Repartition(upCurves, upN.chunks[0].stride[1]))
                    result.topologyChanged = true;
            }
        }
        for (auto const &e : nodeDiag.errors) aggregated.Error(e);
        for (auto const &w : nodeDiag.warnings) aggregated.Warn(w);
        if (aggregated.HasErrors()) return false;

        if (job->reCaptured && node.capture) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      static_cast<UsdGenChunkDirtyBits>(UsdGenDirtyParameter));
            for (UsdGenNodeId d : node.descendants) {
                if (d >= static_cast<UsdGenNodeId>(graph.NodeCount())) continue;
                std::fill(graph.Node(d).chunkDirty.begin(),
                          graph.Node(d).chunkDirty.end(),
                          static_cast<UsdGenChunkDirtyBits>(UsdGenDirtyParameter));
            }
        }
        job->captureEnd = std::chrono::steady_clock::now();
        // Native time readers move without any authored parameter dirty.
        // Route only their output and descendants; static upstream captures
        // remain reusable during playback.
        if (node.enabled && node.record->Has(kUsdGenOpReadsTime) &&
            (node.lastEvalTime != evalCtx.time ||
             node.lastEvalRate != graph.Desc().timeCodesPerSecond))
            graph.DirtyValue(node.id);
        job->evalAll = node.paramValueDigest != node.lastParamDigest ||
                       job->expressionsChanged;
        node.lastParamDigest = node.paramValueDigest;
        node.lastExpressionDigest = node.expressions.Digest();
        for (UsdGenChunkDirtyBits b : node.chunkDirty)
            job->anyChunkDirty |= (b & UsdGenDirtyParameter) != 0;

        return true;
    };

    // This is deliberately separate from capture.  Capture may call
    // graph.Repartition(), which updates every node's chunks; no node may
    // publish a prepared chunk payload until all captures in this frontier
    // have completed.
    auto prepareEvaluation = [&](int pos, NodeExecution *job) -> bool {
        UsdGenCompiledNode &node = graph.Node(pos);
        UsdGenOp &op = *node.op;
        bool const hasUp = !node.inputs.empty();
        UsdGenElementSet const *up = hasUp
            ? OutputOrEmpty(graph, graph.Node(node.inputs.front())) : nullptr;

        if (!node.enabled && node.record->Has(kUsdGenOpGenerator)) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }
        if (!node.enabled && hasUp) {
            PassThroughOutput(node, *up);
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }
        if ((!job->anyChunkDirty && !job->evalAll) || !node.capture) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }

        PrepareNodeForEval(graph, node, op, up, S.output);
        NodeSweepPayload &pl = job->sweep;
        pl.node = &node;
        pl.graph = &graph;
        pl.ctx = evalCtx;
        pl.ctx.params = &node.paramView;
        pl.ctx.desc = &spaceDesc(pos);
        pl.ctx.references = job->referenceSets.empty() ? nullptr
            : job->referenceSets.data();
        pl.ctx.resolvedReferences = job->resolvedReferences.empty() ? nullptr
            : job->resolvedReferences.data();
        pl.ctx.referenceCount = static_cast<uint32_t>(job->resolvedReferences.size());
        pl.ctx.maps = job->resolvedMaps.empty() ? nullptr : job->resolvedMaps.data();
        pl.ctx.mapCount = static_cast<uint32_t>(job->resolvedMaps.size());
        pl.ctx.mapBindings = node.mapBindingRefs.empty() ? nullptr
            : node.mapBindingRefs.data();
        pl.ctx.mapBindingCount = static_cast<uint32_t>(node.mapBindingRefs.size());
        pl.ctx.seed = node.desc ? static_cast<uint32_t>(node.desc->seed) : 0;
        pl.failureCode.store(0, std::memory_order_relaxed);
        pl.ctx.failureCode = &pl.failureCode;
        pl.op = &op;
        pl.up = hasUp ? &graph.Node(node.input) : nullptr;
        pl.up2 = node.inputs.size() > 1 ? &graph.Node(node.inputs[1]) : nullptr;
        pl.evalAll = job->evalAll;
        pl.planes = op.PlanesTouched();
        // Fallible remappers cannot progressively publish a tile before the
        // frontier joins and validates the shared failure flag.
        pl.deferCompletion = node.remapsVertexPlanes;
        if (pl.deferCompletion) {
            pl.pointScratch.resize(node.output->Count(2));
            pl.scalarScratch.resize(node.output->Count(2));
        }
        pl.didEval.assign(node.chunks.size(), 0);
        if (pos == extentWriter && extentWriter >= 0) {
            pl.extentBase = extentScratch.data();
            pl.extentCap = extentChunks;
        }
        // The sets and slots the chunk view reads, bound after every
        // commit-thread detach above.
        {
            pl.outView = node.output->View();
            pl.outSet = &pl.outView;
            UsdGenChannelTable const *outTable = pl.outSet->channels;
            UsdGenChannelTable const *inTable = nullptr;
            if (pl.up) {
                pl.inView = OutputOrEmpty(graph, graph.Node(node.input))->View();
                pl.inSet = &pl.inView;
                inTable = pl.inSet->channels;
            }
            if (pl.up2) {
                pl.in2View = OutputOrEmpty(graph, graph.Node(node.inputs[1]))->View();
                pl.in2Set = &pl.in2View;
            }
            // Named-plane slots, resolved once per run so the sweep does no
            // name lookup: output, its input predecessor of the same layout
            // (none otherwise), input.
            pl.emittedCount = static_cast<uint16_t>(
                std::min<size_t>(node.outputPrimvars.size(), kUsdGenMaxExtraPlaneSlots));
            for (uint16_t k = 0; k < pl.emittedCount; ++k) {
                UsdGenSlot const out = DynamicSlot(outTable, node.outputPrimvars[k]);
                UsdGenSlot const in = out == kUsdGenNoSlot ? kUsdGenNoSlot
                    : DynamicSlot(inTable, node.outputPrimvars[k]);
                pl.emitted[k] = out;
                pl.emittedIn[k] = in != kUsdGenNoSlot &&
                    UsdGenSameLayout((*outTable)[out], (*inTable)[in]) ? in : kUsdGenNoSlot;
            }
            pl.consumedCount = static_cast<uint16_t>(
                std::min<size_t>(node.inputPrimvars.size(), kUsdGenMaxExtraPlaneSlots));
            for (uint16_t k = 0; k < pl.consumedCount; ++k)
                pl.consumed[k] = DynamicSlot(inTable, node.inputPrimvars[k]);
            if (node.remapsVertexPlanes && outTable) {
                UsdGenSpan<const UsdGenSlot> const run = outTable->DynamicRun(2);
                pl.remapped = run.data();
                pl.remappedCount = static_cast<uint16_t>(run.size());
            }
            pl.outSlots = outTable ? outTable->Size() : 0;
            pl.inSlots = inTable ? inTable->Size() : 0;
            pl.in2Slots = pl.in2Set && pl.in2Set->channels ? pl.in2Set->channels->Size() : 0;
            pl.tableChunks = node.chunks.size();
            pl.outTables.resize(pl.tableChunks * pl.outSlots);
            pl.inTables.resize(pl.tableChunks * pl.inSlots);
            pl.in2Tables.resize(pl.tableChunks * pl.in2Slots);
            pl.pointsIn = (pl.planes & UsdGenOp::kPlanePoints) != 0;
            pl.widthIn = (pl.planes & UsdGenOp::kPlaneWidths) != 0;
        }
        // Capture-direct: this run's capture wrote the output planes, so
        // no sweep to run. Freshness comes from the re-capture flag, never
        // from the (possibly reused) capture alone. Preparation above
        // still runs (topology inherit, pass-through planes); the
        // post-frontier loop publishes what a full sweep would have.
        if (job->reCaptured && node.capture && node.capture->WroteDirectOutput()) {
            job->sweptDirect = true;
            job->shouldSweep = false;
            return true;
        }
        job->shouldSweep = true;
        return true;
    };

    for (int lane = 0; lane < 2; ++lane) {
        bool const referenceLane = lane == 0;
        int remaining = 0;
        for (int pos = 0; pos < graph.NodeCount(); ++pos)
            remaining += ((graph.Node(pos).role == UsdGenRole::Reference) ==
                          referenceLane);
        while (remaining > 0) {
            if (superseded && superseded()) {
                result.superseded = true;
                result.diagnostics = std::move(aggregated);
                return result;
            }
            std::vector<int> &frontier = S.frontier;
            frontier.clear();
            for (int pos = 0; pos < graph.NodeCount(); ++pos) {
                UsdGenCompiledNode const &node = graph.Node(pos);
                if (completed[size_t(pos)] ||
                    ((node.role == UsdGenRole::Reference) != referenceLane))
                    continue;
                bool ready = true;
                for (UsdGenNodeId input : node.inputs) {
                    if (input >= static_cast<UsdGenNodeId>(graph.NodeCount()) ||
                        !completed[size_t(input)]) {
                        ready = false;
                        break;
                    }
                }
                if (ready) frontier.push_back(pos);
            }
            if (frontier.empty()) {
                aggregated.Error(referenceLane
                    ? "UsdGen: reference dependency frontier stalled"
                    : "UsdGen: geometry dependency frontier stalled");
                result.diagnostics = std::move(aggregated);
                return result;
            }

            while (S.jobs.size() < frontier.size())
                S.jobs.push_back(std::make_unique<NodeExecution>());
            TfSpan<const std::unique_ptr<NodeExecution>> const jobs(
                S.jobs.data(), frontier.size());
            for (size_t j = 0; j < frontier.size(); ++j) {
                jobs[j]->Reset();
                if (!prepareNode(frontier[j], jobs[j].get())) {
                    result.diagnostics = std::move(aggregated);
                    return result;
                }
            }
            // Capture can repartition the graph. Keep completion counters and
            // the eventual interleave sweep aligned with the new tile plan.
            if (graph.NumGroups() != nTiles) {
                nTiles = graph.NumGroups();
                tileTouched.resize(nTiles, 1);
            }

            // Prepare every node only after every capture/repartition above
            // has settled the graph-wide chunk layout.
            for (size_t j = 0; j < frontier.size(); ++j) {
                if (!prepareEvaluation(frontier[j], jobs[j].get())) {
                    result.diagnostics = std::move(aggregated);
                    return result;
                }
            }

            std::vector<ChunkExecution> &work = S.work;
            work.clear();
            size_t sweepJobs = 0;
            bool frontierFallible = false;
            for (auto const &job : jobs) {
                if (!job->shouldSweep) continue;
                ++sweepJobs;
                frontierFallible |= job->sweep.deferCompletion;
                for (size_t c = 0; c < job->sweep.node->chunks.size(); ++c) {
                    if (job->evalAll ||
                        (job->sweep.node->chunkDirty[c] & UsdGenDirtyParameter))
                        work.push_back({&job->sweep, c});
                }
            }
            if (frontierFallible)
                for (auto const &job : jobs)
                    if (job->shouldSweep) job->sweep.deferCompletion=true;
            // The terminal output has no downstream writer. Its chunk spans
            // are disjoint, so completion of the last selected chunk in a
            // tile makes that tile safe to copy while other tiles still run.
            // The counters live through ParallelFor's synchronous join.
            std::atomic<uint32_t> *terminalRemaining = nullptr;
            if (tileCompleted && !transactionalRun && nTiles > 0) {
                for (auto const &job : jobs) {
                    if (!job->shouldSweep ||
                        job->sweep.node->id != graph.TerminalNodeId()) continue;
                    if (S.terminalCapacity < size_t(nTiles)) {
                        S.terminalRemaining =
                            std::make_unique<std::atomic<uint32_t>[]>(size_t(nTiles));
                        S.terminalCapacity = size_t(nTiles);
                    }
                    terminalRemaining = S.terminalRemaining.get();
                    for (int t = 0; t < nTiles; ++t)
                        terminalRemaining[t].store(0, std::memory_order_relaxed);
                    for (ChunkExecution const &item : work) {
                        if (item.payload != &job->sweep) continue;
                        UsdGenTileId const tile = job->sweep.node->chunks[item.chunk].group;
                        if (tile < static_cast<UsdGenTileId>(nTiles))
                            terminalRemaining[tile].fetch_add(1, std::memory_order_relaxed);
                    }
                    S.terminal = TerminalTiles{&graph, job->sweep.node,
                                               terminalRemaining, nTiles,
                                               &tileCompleted};
                    job->sweep.onChunkCompleted = [t = &S.terminal](size_t chunk) {
                        UsdGenTileId const tile = t->node->chunks[chunk].group;
                        if (tile >= static_cast<UsdGenTileId>(t->nTiles)) return;
                        if (t->remaining[tile].fetch_sub(1, std::memory_order_acq_rel) == 1)
                            (*t->completed)(t->graph->Groups()[tile], *t->node->output);
                    };
                    break;
                }
            }
            if (!work.empty()) {
                TRACE_SCOPE("usdGen chunk sweep");
                TRACE_COUNTER_DELTA("usdGen chunks evaluated", double(work.size()));
                for (auto const &job : jobs)
                    if (job->shouldSweep)
                        job->evaluationStart = std::chrono::steady_clock::now();
                // Preserve the small sparse-node fast path. A frontier with
                // multiple ready nodes intentionally goes through one
                // flattened dispatch, even when each node has only a handful
                // of dirty chunks, so independent branches can overlap.
                if (sweepJobs == 1 && work.size() <= 4) {
                    for (ChunkExecution const &item : work)
                        SweepChunk(item.chunk, item.payload);
                } else {
                    dispatcher.ParallelFor(work.size(), SweepPreparedChunk,
                                           work.data());
                }
            }

            for (auto const &job : jobs) {
                uint32_t const failure = job->sweep.failureCode.load(
                    std::memory_order_acquire);
                if (!failure) continue;
                std::string message = "UsdGen: operator evaluation failed";
                if (failure == 1)
                    message = "UsdGenCollide: closed-surface depth query failed";
                else if (failure == 2)
                    message = "UsdGenCollide: cut depth search exceeded its deterministic resource budget";
                aggregated.Error(std::move(message));
                result.diagnostics = std::move(aggregated);
                return result;
            }

            for (auto const &job : jobs) {
                if (!job->sweep.deferCompletion ||
                    !job->sweep.onChunkCompleted) continue;
                for (size_t c=0;c<job->sweep.didEval.size();++c)
                    if (job->sweep.didEval[c]) job->sweep.onChunkCompleted(c);
            }

            for (size_t j = 0; j < jobs.size(); ++j) {
                NodeExecution &job = *jobs[j];
                UsdGenCompiledNode &node = graph.Node(frontier[j]);
                if (int(frontier[j]) == extentWriter && job.shouldSweep) {
                    extentSwept = true;
                    for (uint8_t b : job.sweep.didEval)
                        if (!b) extentSwept = false;
                }
                if (int(frontier[j]) == extentWriter && job.sweptDirect &&
                    node.capture && node.capture->RecordedChunkExtents())
                    extentSwept = true;
                if (job.sweptDirect) {
                    // A capture-direct node wrote every output and recorded
                    // (or, gateless, skipped) its extents: publish exactly
                    // what a full sweep would have (fresh values, touched
                    // tiles, cleared dirt, stats with no eval leg).
                    UsdGenWriteSet::Restamp(*node.output, ++node.valueVersion);
                    for (size_t c = 0; c < node.chunks.size(); ++c)
                        if (node.chunks[c].group < tileTouched.size())
                            tileTouched[node.chunks[c].group] = 1;
                    std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                              UsdGenDirtyNone);
                    auto const end = std::chrono::steady_clock::now();
                    UsdGenNodeRunStats st;
                    st.id = static_cast<UsdGenNodeId>(frontier[j]);
                    st.captureMs = std::chrono::duration<double, std::milli>(
                        job.captureEnd - job.captureStart).count();
                    st.evalMs = 0.0;
                    st.chunksEvaluated = node.chunks.size();
                    S.nodeStats.push_back(st);
                }
                if (job.shouldSweep) {
                    bool wrote = job.reCaptured;
                    for (uint8_t b : job.sweep.didEval)
                        if (b) { wrote = true; break; }
                    if (wrote) {
                        UsdGenWriteSet::Restamp(*node.output, ++node.valueVersion);
                        for (size_t c = 0; c < node.chunks.size(); ++c)
                            if (job.sweep.didEval[c] &&
                                node.chunks[c].group < tileTouched.size())
                                tileTouched[node.chunks[c].group] = 1;
                    }
                    std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                              UsdGenDirtyNone);
                    auto const end = std::chrono::steady_clock::now();
                    UsdGenNodeRunStats st;
                    st.id = static_cast<UsdGenNodeId>(frontier[j]);
                    st.captureMs = std::chrono::duration<double, std::milli>(
                        job.captureEnd - job.captureStart).count();
                    st.evalMs = std::chrono::duration<double, std::milli>(
                        end - job.evaluationStart).count();
                    for (uint8_t b : job.sweep.didEval) if (b) ++st.chunksEvaluated;
                    S.nodeStats.push_back(st);
                }
                if (TfDebug::IsEnabled(USDGEN_SCHEDULE)) {
                    size_t evaluated = 0;
                    for (uint8_t b : job.sweep.didEval) evaluated += b != 0;
                    double const captureMs = std::chrono::duration<double, std::milli>(
                        job.captureEnd - job.captureStart).count();
                    double const evalMs = job.shouldSweep && evaluated
                        ? std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - job.evaluationStart).count()
                        : 0.0;
                    TfDebug::Helper().Msg(
                        "usdGen schedule  %-12s %-18s %-22s prepare %8.2f ms  eval %4zu/%-4zu chunks %8.2f ms%s\n",
                        NodeLabel(node).c_str(),
                        node.desc ? node.desc->type.GetText() : "?",
                        job.reCaptured ? job.recaptureReason
                            : node.enabled ? "capture reused" : "disabled",
                        captureMs, evaluated, node.chunks.size(), evalMs,
                        job.evalAll ? "  (all chunks: values moved)" : "");
                }
                completed[size_t(frontier[j])] = 1;
                if (node.record->Has(kUsdGenOpReadsTime) && node.enabled) {
                    node.lastEvalTime = evalCtx.time;
                    node.lastEvalRate = graph.Desc().timeCodesPerSecond;
                }
                --remaining;
            }
        }
    }
    UsdGenElementSet const &termSet = graph.Output();
    UsdGenCurveSetView const term = UsdGenCurveSetView::Of(termSet);
    UsdGenCompiledNode const &tn = graph.Node(graph.TerminalNodeId());
    UsdGenOp const *termOp = tn.op.get();

    if (result.topologyChanged)
        std::fill(tileTouched.begin(), tileTouched.end(), 1);
    // Validate the fused extent slots against the current buffers. Every
    // condition re-checks the present moment: the sweep recorded spans as
    // they were, so anything that moved since (a repartition, a plane
    // detach, a differently-sized terminal partition) fails closed into
    // the point pass.
    UsdGenChunkExtent const *fusedExtents = nullptr;
    if (extentWriter >= 0 && extentSwept) {
        UsdGenCompiledNode const &wn = graph.Node(UsdGenNodeId(extentWriter));
        UsdGenCurveSetView const written = UsdGenCurveSetView::Of(*OutputOrEmpty(graph, wn));
        auto sameBacking = [](auto const &a, auto const &b) {
            return a.size() == b.size() && (a.empty() || a.cdata() == b.cdata());
        };
        bool ok = tn.chunks.size() == extentChunks &&
                  wn.chunks.size() == extentChunks &&
                  sameBacking(term.px, written.px) &&
                  sameBacking(term.py, written.py) &&
                  sameBacking(term.pz, written.pz) &&
                  sameBacking(term.cvOffsets, written.cvOffsets);
        for (size_t i = 0; ok && i < extentChunks; ++i) {
            UsdGenChunkRange const &tc = tn.chunks[i];
            UsdGenChunkRange const &wc = wn.chunks[i];
            UsdGenChunkRange const &s = extentScratch[i].range;
            ok = tc.live == tc.count[0] && wc.live == wc.count[0] &&
                 s.first[0] == wc.first[0] && s.count[0] == wc.count[0] &&
                 s.live == wc.live && s.first[1] == wc.first[1] &&
                 s.stride[1] == wc.stride[1] && wc.first[0] == tc.first[0] &&
                 wc.count[0] == tc.count[0] && wc.live == tc.live &&
                 wc.first[1] == tc.first[1] && wc.stride[1] == tc.stride[1];
        }
        if (ok) fusedExtents = extentScratch.data();
    }
    if (nTiles > 0) {
        TRACE_SCOPE("usdGen interleave tiles");
        InterleavePayload ip;
        ip.graph = &graph;
        ip.tn = &tn;
        ip.term = term;
        ip.touched = &tileTouched;
        ip.widthsFlag = termOp &&
            (termOp->PlanesTouched() & UsdGenOp::kPlaneWidths) != 0;
        ip.extents = fusedExtents;
        dispatcher.ParallelFor(size_t(nTiles), InterleaveTile, &ip);
    }

    if (tileCompleted && transactionalRun)
        for (int tile=0;tile<nTiles;++tile)
            if (tileTouched[size_t(tile)])
                tileCompleted(graph.Groups()[tile], termSet);

    result.terminalOutput = &graph.Output();
    result.groups = graph.Groups();
    result.diagnostics = std::move(aggregated);
    return result;
}

}  // namespace usdGen
