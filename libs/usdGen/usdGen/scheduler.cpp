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

#include "usdGen/debugCodes.h"
#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/tbbFastCores.h"
#include "usdGen/types.h"

#include "pxr/pxr.h"
#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/tf/getenv.h"
#include "pxr/base/trace/trace.h"
#include "pxr/base/vt/array.h"

#include "tbb/task_arena.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

namespace {

// The one op whose Evaluate records per-chunk extents (see the gate in
// Run): a file static, interned at load, compared by pointer below.
TfToken const sDeformType{"UsdGenDeform"};

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


UsdGenCurveBuffer const &EmptyBuffer()
{
    static UsdGenCurveBuffer const empty;
    return empty;
}

/// Per-curve arrays a styler needs but does not compute (inherited from the
/// upstream buffer; VtArray assignment is a cheap CoW share).
void InheritPerCurve(UsdGenCurveBuffer &dst, UsdGenCurveBuffer const &src)
{
    if (dst.curveId.empty() && !src.curveId.empty()) dst.curveId = src.curveId;
    if (dst.rootPrim.empty() && !src.rootPrim.empty()) dst.rootPrim = src.rootPrim;
    if (dst.rootUV.empty() && !src.rootUV.empty()) dst.rootUV = src.rootUV;
    if (dst.rootT.empty() && !src.rootT.empty()) dst.rootT = src.rootT;
    if (dst.rootN.empty() && !src.rootN.empty()) dst.rootN = src.rootN;
    if (dst.rootB.empty() && !src.rootB.empty()) dst.rootB = src.rootB;
}

/// A non-owning styler's per-curve channels are immutable aliases of its
/// current input. Refresh them on every preparation: emptiness is not an
/// ownership bit, and a reused node may otherwise retain aliases from an old
/// upstream topology/stable-ID generation.
void AliasPerCurve(UsdGenCurveBuffer &dst, UsdGenCurveBuffer const &src)
{
    dst.curveId = src.curveId;
    dst.rootPrim = src.rootPrim;
    dst.rootUV = src.rootUV;
    dst.rootT = src.rootT;
    dst.rootN = src.rootN;
    dst.rootB = src.rootB;
}

void AliasExtraPlanes(UsdGenCurveBuffer &dst, UsdGenCurveBuffer const &src)
{
    // Copying the vectors copies only plane descriptors and VtArray handles.
    // The arrays themselves remain immutable aliases until an output slot is
    // explicitly materialized below.
    dst.extraCv = src.extraCv;
    dst.extraCurve = src.extraCurve;
}

UsdGenPlane const *FindPlane(std::vector<UsdGenPlane> const &planes,
                             TfToken const &name)
{
    auto const it = std::lower_bound(planes.begin(), planes.end(), name,
        [](UsdGenPlane const &plane, TfToken const &needle) {
            return plane.name < needle;
        });
    return it != planes.end() && it->name == name ? &*it : nullptr;
}

struct PlaneLookup
{
    UsdGenPlane const *plane = nullptr;
    bool cv = false;
};

PlaneLookup FindExtraPlane(UsdGenCurveBuffer const &buffer, TfToken const &name)
{
    if (UsdGenPlane const *plane = FindPlane(buffer.extraCv, name))
        return {plane, true};
    if (UsdGenPlane const *plane = FindPlane(buffer.extraCurve, name))
        return {plane, false};
    return {};
}

bool SamePlaneLayout(UsdGenPlane const &a, UsdGenPlane const &b)
{
    return a.name == b.name && a.interpolation == b.interpolation &&
        a.type == b.type && a.arity == b.arity;
}

UsdGenPlane DefaultOutputPlane(TfToken const &name)
{
    // R24's fixed declarations cover first producers.  A later writer takes
    // its layout from the upstream plane instead, which also supports vertex
    // planes created by a generator/capture without teaching the scheduler a
    // second metadata vocabulary.
    UsdGenPlane plane;
    plane.name = name;
    plane.interpolation = TfToken("uniform");
    plane.type = TfToken("float");
    if (name == TfToken("guideIndex") || name == TfToken("partId") ||
        name.GetString().rfind("clumpId_", 0) == 0 ||
        name.GetString().rfind("clumpCenterId_", 0) == 0)
        plane.type = TfToken("int");
    if (name.GetString().rfind("clumpCenter_", 0) == 0)
        plane.arity = 3;
    if (name.GetString().rfind("clumpCenterId_", 0) == 0)
        plane.arity = 2;
    // Shared whole-strand vocabulary (ops/wholeStrand.h): per-strand guide
    // and seed identities are ints; the braid knot mask is a per-CV float.
    if (name == TfToken("primaryGuide") || name == TfToken("nearestGuide") ||
        name == TfToken("strandSeed") || name == TfToken("strandIndex") ||
        name == TfToken("mirrorSign"))
        plane.type = TfToken("int");
    if (name.GetString().rfind("clumpWeight_", 0) == 0 ||
        name == TfToken("braidKnotMask"))
        plane.interpolation = TfToken("vertex");
    if (name == TfToken("guideIndex") || name == TfToken("guideWeight"))
        plane.arity = 3;
    return plane;
}

bool IsNativeClumpPlane(TfToken const &name)
{
    std::string const &s = name.GetString();
    return s.rfind("clumpId_", 0) == 0 ||
           s.rfind("clumpCenter_", 0) == 0 ||
           s.rfind("clumpCenterId_", 0) == 0 ||
           s.rfind("clumpWeight_", 0) == 0;
}

void InsertOrAssignPlane(std::vector<UsdGenPlane> *planes, UsdGenPlane plane)
{
    auto it = std::lower_bound(planes->begin(), planes->end(), plane.name,
        [](UsdGenPlane const &candidate, TfToken const &needle) {
            return candidate.name < needle;
        });
    if (it != planes->end() && it->name == plane.name)
        *it = std::move(plane);
    else
        planes->insert(it, std::move(plane));
}

bool PlaneAliases(UsdGenPlane const &a, UsdGenPlane const *b)
{
    if (!b) return false;
    return a.type == TfToken("int") ? a.i.cdata() == b->i.cdata()
                                    : a.f.cdata() == b->f.cdata();
}

void PrepareExtraPlanes(UsdGenCompiledNode &node,
                        UsdGenCurveBuffer const &upstream, bool hasUpstream)
{
    if (!hasUpstream || node.op->IsGenerator()) return;

    UsdGenCurveBuffer &buffer = node.buffer;
    // Preserve prior private output arrays for sparse sweeps, but overwrite
    // every pass-through descriptor with the CURRENT source first.  That is
    // what prevents a retained node from keeping VtArray aliases from an old
    // topology generation.
    std::vector<UsdGenPlane> const oldCv = buffer.extraCv;
    std::vector<UsdGenPlane> const oldCurve = buffer.extraCurve;
    AliasExtraPlanes(buffer, upstream);

    for (TfToken const &name : node.outputPrimvars) {
        PlaneLookup const source = FindExtraPlane(upstream, name);
        PlaneLookup old;
        if (UsdGenPlane const *cvPlane = FindPlane(oldCv, name)) old = {cvPlane, true};
        else if (UsdGenPlane const *curvePlane = FindPlane(oldCurve, name)) old = {curvePlane, false};

        PlaneLookup const layout = source.plane ? source : old;
        bool const nativeClump = node.type == TfToken("UsdGenClump") &&
                                 IsNativeClumpPlane(name);
        UsdGenPlane plane = nativeClump ? DefaultOutputPlane(name) :
            (layout.plane ? *layout.plane : DefaultOutputPlane(name));
        // A first producer's domain comes from its default layout (vertex
        // planes such as braidKnotMask are sized per CV).
        bool const cv = nativeClump ? plane.interpolation == TfToken("vertex") :
            (layout.plane ? layout.cv
                          : plane.interpolation == TfToken("vertex"));
        size_t const values = size_t(cv ? buffer.totalCvs : buffer.totalCurves) *
            std::max<uint8_t>(plane.arity, 1);

        // Retain only a same-layout private old output.  If it aliases the
        // upstream payload (or its topology-sized storage no longer fits), a
        // fresh VtArray is required before the chunk kernel receives a raw
        // writable pointer.
        bool const keepOld = old.plane && old.cv == cv &&
            SamePlaneLayout(*old.plane, plane) && !PlaneAliases(*old.plane, source.plane) &&
            (plane.type == TfToken("int") ? old.plane->i.size() == values
                                          : old.plane->f.size() == values);
        if (keepOld) plane = *old.plane;
        else if (plane.type == TfToken("int")) {
            plane.i = VtIntArray(values, 0);
            plane.f.clear();
        } else {
            plane.f = VtFloatArray(values, 0.0f);
            plane.i.clear();
        }
        if (nativeClump) {
            auto &other = cv ? buffer.extraCurve : buffer.extraCv;
            auto it = std::lower_bound(other.begin(), other.end(), name,
                [](UsdGenPlane const &p, TfToken const &n) { return p.name < n; });
            if (it != other.end() && it->name == name) other.erase(it);
        }
        InsertOrAssignPlane(cv ? &buffer.extraCv : &buffer.extraCurve,
                            std::move(plane));
    }
    if (node.op->RemapsVertexPlanes()) {
        for (UsdGenPlane const &source : upstream.extraCv) {
            UsdGenPlane plane = source;
            UsdGenPlane const *prior = FindPlane(oldCv, source.name);
            size_t const values = size_t(buffer.totalCvs) *
                std::max<uint8_t>(source.arity, 1);
            bool const reusable = prior && SamePlaneLayout(*prior, source) &&
                !PlaneAliases(*prior, &source) &&
                (source.type == TfToken("int") ? prior->i.size() == values
                                                : prior->f.size() == values);
            if (reusable) {
                plane = *prior;
                // Detach on the commit thread while the retained descriptor
                // is still a second owner. Workers only receive raw pointers.
                if (source.type == TfToken("int")) (void)plane.i.data();
                else (void)plane.f.data();
            }
            else if (source.type == TfToken("int")) {
                plane.i = VtIntArray(values, 0); plane.f.clear();
            } else {
                plane.f = VtFloatArray(values, 0.0f); plane.i.clear();
            }
            InsertOrAssignPlane(&buffer.extraCv, std::move(plane));
        }
    }
}

/// Commit-thread preparation of one node's output buffer before its
/// evaluate sweep: topology fields and per-curve arrays are inherited from
/// the upstream node; CV planes the operator does NOT touch are CoW-aliased
/// from upstream (re-aliased every commit, so a stale shared array can
/// never outlive its source); planes it DOES touch must exist at full
/// buffer size (chunk-local pointers are sliced from these).
void PrepareNodeForEval(
    UsdGenCompiledNode &node, UsdGenOp &op,
    UsdGenCurveBuffer const &upBuf, bool hasUp)
{
    UsdGenCurveBuffer &buf = node.buffer;
    bool ownsBuffer = false;

    // Inherit the topology FIRST: the CV count of the sizing below must be
    // the node's EFFECTIVE curve count, not its pre-inheritance 0 (an
    // un-populated styler buffer would otherwise size every touched plane
    // to zero and hand the kernel null plane pointers).
    if (hasUp && !op.IsGenerator()) {
        ownsBuffer = node.capture && node.capture->OwnsBuffer();
        if (!ownsBuffer) {
            buf.totalCurves = upBuf.totalCurves;
            buf.totalCvs = upBuf.totalCvs;
            buf.topologyVersion = upBuf.topologyVersion;
            AliasPerCurve(buf, upBuf);
        } else {
            if (buf.totalCurves == 0) {
                buf.totalCurves = upBuf.totalCurves;
                buf.totalCvs = upBuf.totalCvs;
                buf.topologyVersion = upBuf.topologyVersion;
            }
            InheritPerCurve(buf, upBuf);
        }
        if (!ownsBuffer) {
            // Topology metadata refresh/clear on pass-through preps only; owning
            // captures (e.g. Deform) keep their authored cvOffsets.
            buf.cvOffsets = upBuf.cvOffsets;
        } else if (buf.cvOffsets.empty() && !upBuf.cvOffsets.empty() &&
                   buf.totalCurves == upBuf.totalCurves &&
                   buf.totalCvs == upBuf.totalCvs) {
            // FIX (P1, ragged-at-Deform): an owning capture that authored NO
            // topology of its own (Deform — the totals just above are the
            // inherited upstream ones) must not drop a ragged input layout:
            // ragged totals with EMPTY offsets make Repartition derive a
            // uniform cvp (totalCvs/totalCurves == 12/3 == 4) and sweep
            // ragged spans as uniform. Carry the input cvOffsets verbatim,
            // mirroring the pass-through carry; the totals equality pins
            // "no topology authored" so a re-topologised owner keeps its own.
            buf.cvOffsets = upBuf.cvOffsets;
        }
        // Rest is an immutable C3 transport channel, separate from current
        // points.  Every topology-preserving operator aliases it on each
        // preparation so a reused node cannot retain a prior generation's
        // owner.  Topology producers that repartition CVs (Grow, Resample,
        // GuideInterpolate)
        // explicitly materialize a new rest layout during Capture instead.
        bool const captureAuthorsRest = ownsBuffer &&
            (op.Type() == TfToken("UsdGenGrow") ||
             op.Type() == TfToken("UsdGenResample") ||
             op.Type() == TfToken("UsdGenGuideInterpolate") ||
             // Frozen Freeze owns its buffer (OwnsBuffer()==!live) and keeps
             // the snapshotted rest; live Freeze re-aliases like a styler.
             op.Type() == TfToken("UsdGenFreeze"));
        if (!captureAuthorsRest)
            buf.rest = upBuf.rest;
    }

    uint32_t const nCurves = buf.totalCurves;
    uint32_t const cvp = node.chunks.empty() ? 0 : node.chunks[0].cvCount;
    uint32_t const nCvs = (cvp == 0 && !buf.cvOffsets.empty() &&
                           size_t(nCurves) < buf.cvOffsets.size())
        ? uint32_t(buf.cvOffsets[nCurves])   // ragged: total CVs from offsets
        : nCurves * cvp;

    // Plane policy (03 §1.2/§8.5, S24): a plane the OP WRITES must be THIS
    // node's own storage (VtArray CoW sharing would make the kernel's write
    // detach-and-memcpy the whole buffer once per run — the measured cost of
    // every sparse commit); a plane the op passes through is a read-only
    // alias of the upstream plane, refreshed only when the identity differs.
    // Kernels write through the raw pointer SweepChunk slices from
    // cdata()+const_cast. Fallible remappers detach retained storage once on
    // this commit thread before workers receive those pointers.
    uint32_t const planes = op.PlanesTouched();
    auto prep = [&](bool write, VtFloatArray &a, VtFloatArray const &up) {
        if (write) {
            if (a.size() != nCvs || (hasUp && a.cdata() == up.cdata()))
                a = VtFloatArray(nCvs, 0.0f);   // fresh private storage
        } else if (hasUp && up.size() == nCvs) {
            if (a.cdata() != up.cdata()) a = up;  // CoW share; read-only
        } else if (!hasUp && op.IsGenerator()) {
            // Source generators own the presence semantics of optional
            // planes.  In particular Scatter has no input width plane;
            // materializing a zero array here would make downstream Grow
            // resample an invented authored width instead of its description
            // fallback.  Capture-created Source planes remain untouched.
        } else if (a.size() != nCvs) {
            a = VtFloatArray(nCvs, 0.0f);
        }
    };
    prep(planes & UsdGenOp::kPlanePoints, buf.px, upBuf.px);
    prep(planes & UsdGenOp::kPlanePoints, buf.py, upBuf.py);
    prep(planes & UsdGenOp::kPlanePoints, buf.pz, upBuf.pz);
    prep(planes & UsdGenOp::kPlaneWidths, buf.width, upBuf.width);
    prep(planes & UsdGenOp::kPlaneHairT, buf.hairT, upBuf.hairT);
    // A CV-repartitioning capture has already transformed inherited named
    // planes or authored guide-derived replacements for its new CV layout.
    // Do not replace those private owners with the old upstream descriptors
    // during generic pass-through preparation.
    bool const captureOwnsTransformedPlanes = hasUp && ownsBuffer &&
        (op.Type() == TfToken("UsdGenGrow") ||
         op.Type() == TfToken("UsdGenResample") ||
         op.Type() == TfToken("UsdGenGuideInterpolate") ||
         // Frozen Freeze: the snapshot owns the transformed extras (same
         // ownsBuffer gate as above); live Freeze prepares pass-through.
         op.Type() == TfToken("UsdGenFreeze"));
    if (!captureOwnsTransformedPlanes)
        PrepareExtraPlanes(node, upBuf, hasUp);
}

/// Detach writable built-in planes on the commit thread for fallible
/// remappers. Their node buffer can still share a retained published snapshot
/// even after it no longer aliases the immediate upstream buffer.
void EnsureWritablePlanes(UsdGenCompiledNode &node, UsdGenOp &op)
{
    if (!op.RemapsVertexPlanes()) return;
    uint32_t const planes=op.PlanesTouched();
    if (planes & UsdGenOp::kPlanePoints) {
        (void)node.buffer.px.data();
        (void)node.buffer.py.data();
        (void)node.buffer.pz.data();
    }
    if (planes & UsdGenOp::kPlaneWidths) (void)node.buffer.width.data();
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
};

// A prepared node job owns the small vectors whose addresses are published in
// its EvalContext.  Preparation happens on the commit thread; arena workers
// only consume the immutable context and write this node's disjoint chunks.
std::string NodeLabel(UsdGenCompiledNode const &node)
{
    return node.desc ? node.desc->path.GetName() : std::string("?");
}

struct NodeExecution
{
    std::vector<UsdGenCurveBuffer const *> upstreamInputs;
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
};

struct ChunkExecution
{
    NodeSweepPayload *payload = nullptr;
    size_t chunk = 0;
};

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
    UsdGenChunkDesc const &cd = node.chunks[index];
    if (cd.curveCount == 0) return;
    bool const chunkDirty =
        node.chunkDirty.empty()
            ? true   // no dirty bookkeeping for this partition: evaluate (never skip blindly)
            : (node.chunkDirty[index] & UsdGenDirtyParameter) != 0;
    if (!pl.evalAll && !chunkDirty) return;

    UsdGenCurveBuffer &buf = node.buffer;
    UsdGenCurveBuffer const &upBuf = pl.up ? pl.up->buffer : EmptyBuffer();
    UsdGenChunkDesc const *upC = (pl.up && pl.up->chunks.size() == node.chunks.size())
        ? &pl.up->chunks[index] : nullptr;
    UsdGenCurveBuffer const &upBuf2 = pl.up2 ? pl.up2->buffer : EmptyBuffer();
    UsdGenChunkDesc const *upC2 = (pl.up2 && pl.up2->chunks.size() == node.chunks.size())
        ? &pl.up2->chunks[index] : nullptr;

    uint32_t nCvs = cd.curveCount * cd.cvCount;
    if (cd.cvCount == 0 && !buf.cvOffsets.empty()) {
        // Ragged span [firstCurve, firstCurve+curveCount) from topology offsets.
        size_t const f = cd.firstCurve;
        if (f + cd.curveCount < buf.cvOffsets.size())
            nCvs = uint32_t(buf.cvOffsets[f + cd.curveCount] - buf.cvOffsets[f]);
    }
    size_t const base = cd.firstCv;
    size_t const upBase = upC ? upC->firstCv : 0;

    UsdGenChunkView view{};
    view.desc = &cd;
    view.curveCount = cd.curveCount;
    view.cvCount = cd.cvCount;
    // The recording writer's slot, when the gate armed one and the chunk
    // still fits the gate-time partition (a mid-run repartition grows the
    // partition past the scratch; the interleave size check falls back).
    view.extentSlot = (pl.extentBase && index < pl.extentCap)
        ? pl.extentBase + index : nullptr;
    view.inCvCount = upC ? upC->cvCount : 0;
    view.inFirstCv = static_cast<uint32_t>(upBase);
    view.inCvOffsets = (upC && upC->cvCount == 0 && !upBuf.cvOffsets.empty() &&
                        size_t(upC->firstCurve) < upBuf.cvOffsets.size())
        ? upBuf.cvOffsets.cdata() + upC->firstCurve : nullptr;
    view.outCount = static_cast<uint32_t>(node.outputPrimvars.size());
    view.inCount = static_cast<uint32_t>(node.inputPrimvars.size());
    view.outF = nullptr;
    view.outI = nullptr;
    view.inF = nullptr;
    view.inI = nullptr;
    // Chunk-local BY BINDING (zero copy): absolute offsets; planes below slice at
    // base=cd.firstCv, so the op-facing plane-relative CV index is
    //   view->cvOffsets[c] - int(desc->firstCv) + i
    // (ops MUST apply the -firstCv subtraction; uniform fast path stays null).
    view.cvOffsets = (cd.cvCount == 0 && !buf.cvOffsets.empty() &&
                      size_t(cd.firstCurve) < buf.cvOffsets.size())
        ? buf.cvOffsets.cdata() + cd.firstCurve : nullptr;

    auto outPlane = [&](VtFloatArray &a) -> float * {
        // cdata(), NOT data(): detach is hoisted to the commit thread
        // (EnsureOwnedPlanes). data() here would CoW-copy the full buffer
        // once per chunk while a pass-through alias holds a reference.
        return a.empty() ? nullptr : const_cast<float *>(a.cdata()) + base;
    };
    auto inPlane = [&](VtFloatArray const &a) -> float const * {
        return a.empty() ? nullptr : a.cdata() + upBase;
    };

    if (pl.planes & UsdGenOp::kPlanePoints) {
        view.px = outPlane(buf.px);
        view.py = outPlane(buf.py);
        view.pz = outPlane(buf.pz);
        view.inPx = inPlane(upBuf.px);
        view.inPy = inPlane(upBuf.py);
        view.inPz = inPlane(upBuf.pz);
    } else {
        view.px = buf.px.empty() ? nullptr : const_cast<float *>(buf.px.cdata()) + base;
        view.py = buf.py.empty() ? nullptr : const_cast<float *>(buf.py.cdata()) + base;
        view.pz = buf.pz.empty() ? nullptr : const_cast<float *>(buf.pz.cdata()) + base;
    }
    view.inWidth = inPlane(upBuf.width);   // upstream-width READ port (03 §1.2)
    view.inWidth2 = upC2 && !upBuf2.width.empty()
        ? upBuf2.width.cdata() + upC2->firstCv : nullptr;
    if (pl.planes & UsdGenOp::kPlaneWidths) {
        view.width = outPlane(buf.width);
        if (!view.inPx) view.inPx = inPlane(upBuf.px);
    } else {
        view.width = buf.width.empty() ? nullptr
            : const_cast<float *>(buf.width.cdata()) + base;
    }
    if (pl.planes & UsdGenOp::kPlaneHairT) {
        view.hairT = outPlane(buf.hairT);
    } else {
        view.hairT = buf.hairT.empty() ? nullptr
            : const_cast<float *>(buf.hairT.cdata()) + base;
    }

    // Extra planes use fixed stack tables: Evaluate() remains allocation-free
    // even when an operator exposes several typed slots.  Slots are resolved
    // from the freshly prepared buffer, rather than cached pointers, so a
    // retained node cannot write a VtArray from a stale topology revision.
    std::array<float *, kUsdGenMaxExtraPlaneSlots> outF{};
    std::array<int *, kUsdGenMaxExtraPlaneSlots> outI{};
    std::array<float const *, kUsdGenMaxExtraPlaneSlots> inF{};
    std::array<int const *, kUsdGenMaxExtraPlaneSlots> inI{};
    std::array<float const *, kUsdGenMaxExtraPlaneSlots> outputInF{};
    std::array<int const *, kUsdGenMaxExtraPlaneSlots> outputInI{};
    std::array<size_t, kUsdGenMaxExtraPlaneSlots> outputValues{};
    auto offsetFor = [&](bool cv, bool upstreamPlane, uint8_t arity) {
        size_t const element = cv
            ? (upstreamPlane ? upBase : base)
            : (upstreamPlane && upC ? upC->firstCurve : cd.firstCurve);
        return element * std::max<uint8_t>(arity, 1);
    };
    for (uint32_t slot = 0; slot < view.outCount; ++slot) {
        PlaneLookup const output = FindExtraPlane(buf, node.outputPrimvars[slot]);
        if (!output.plane) continue;  // compiler/preparation is fail-closed to a null port
        size_t const offset = offsetFor(output.cv, false, output.plane->arity);
        outputValues[slot] = size_t(output.cv ? nCvs : cd.curveCount) *
            std::max<uint8_t>(output.plane->arity, 1);
        if (output.plane->type == TfToken("int")) {
            outI[slot] = output.plane->i.empty() ? nullptr
                : const_cast<int *>(output.plane->i.cdata()) + offset;
        } else {
            outF[slot] = output.plane->f.empty() ? nullptr
                : const_cast<float *>(output.plane->f.cdata()) + offset;
        }
        // Envelope endpoints operate on the logical predecessor of the same
        // output name, not on InputPrimvars slot position (the two lists may
        // legitimately differ).
        PlaneLookup const predecessor = FindExtraPlane(upBuf, node.outputPrimvars[slot]);
        if (!predecessor.plane || predecessor.cv != output.cv ||
            !SamePlaneLayout(*predecessor.plane, *output.plane)) continue;
        size_t const inOffset = offsetFor(predecessor.cv, true, predecessor.plane->arity);
        if (predecessor.plane->type == TfToken("int"))
            outputInI[slot] = predecessor.plane->i.empty() ? nullptr
                : predecessor.plane->i.cdata() + inOffset;
        else
            outputInF[slot] = predecessor.plane->f.empty() ? nullptr
                : predecessor.plane->f.cdata() + inOffset;
    }
    for (uint32_t slot = 0; slot < view.inCount; ++slot) {
        PlaneLookup const input = FindExtraPlane(upBuf, node.inputPrimvars[slot]);
        if (!input.plane) continue;
        size_t const offset = offsetFor(input.cv, true, input.plane->arity);
        if (input.plane->type == TfToken("int"))
            inI[slot] = input.plane->i.empty() ? nullptr : input.plane->i.cdata() + offset;
        else
            inF[slot] = input.plane->f.empty() ? nullptr : input.plane->f.cdata() + offset;
    }
    view.outF = view.outCount ? outF.data() : nullptr;
    view.outI = view.outCount ? outI.data() : nullptr;
    view.inF = view.inCount ? inF.data() : nullptr;
    view.inI = view.inCount ? inI.data() : nullptr;
    view.extraCv = pl.op->RemapsVertexPlanes() && !buf.extraCv.empty()
        ? buf.extraCv.data() : nullptr;
    view.inExtraCv = pl.op->RemapsVertexPlanes() && !upBuf.extraCv.empty()
        ? upBuf.extraCv.data() : nullptr;
    view.extraCvCount = pl.op->RemapsVertexPlanes()
        ? static_cast<uint32_t>(std::min(buf.extraCv.size(), upBuf.extraCv.size()))
        : 0;
    view.pointScratch = pl.pointScratch.empty() ? nullptr
        : pl.pointScratch.data()+base;
    view.scalarScratch = pl.scalarScratch.empty() ? nullptr
        : pl.scalarScratch.data()+base;

    // Per-curve arrays (read-only views into the node's own buffer).
    size_t const baseCurve = cd.firstCurve;
    view.curveId = buf.curveId.empty() ? nullptr
        : buf.curveId.cdata() + baseCurve;
    view.rootPrim = buf.rootPrim.empty() ? nullptr
        : buf.rootPrim.cdata() + baseCurve;
    view.rootUV = buf.rootUV.empty() ? nullptr
        : buf.rootUV.cdata() + baseCurve;
    view.rootT = buf.rootT.empty() ? nullptr
        : const_cast<GfVec3f *>(buf.rootT.cdata()) + baseCurve;
    view.rootN = buf.rootN.empty() ? nullptr
        : const_cast<GfVec3f *>(buf.rootN.cdata()) + baseCurve;
    view.rootB = buf.rootB.empty() ? nullptr
        : const_cast<GfVec3f *>(buf.rootB.cdata()) + baseCurve;

    // Capture-payload views: the magnitude LUT comes from the node's capture.
    view.rampLut = nullptr;
    if (node.capture) {
        if (auto p = dynamic_cast<UsdGenCapturePayload const *>(node.capture.get()))
            view.rampLut = p->rampLut.empty() ? nullptr : p->rampLut.cdata();
    }

    pl.op->Evaluate(pl.ctx, *node.capture, &view);

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
    UsdGenCurveBuffer const *term;
    std::vector<char> *touched;
    bool widthsFlag;
    // Fused extents: per-chunk records from the recording sweep, consumed
    // in chunk order. Null unless every validation check passed.
    UsdGenChunkExtent const *extents = nullptr;
};

void InterleaveTile(size_t index, void *payload)
{
    InterleavePayload &pl = *static_cast<InterleavePayload *>(payload);
    UsdGenTileView &tv =
        const_cast<UsdGenTileView &>(pl.graph->Tiles()[index]);
    // Tile dirty flags are per-run observations (03 §5.4 step 5), not sticky
    // state: clear first so a run that evaluates no chunk of this tile reports
    // "nothing changed" instead of replaying the previous run's flag.
    tv.pointsDirty = false;
    tv.widthsDirty = false;
    if (!(*pl.touched)[index]) return;

    // Terminal node owns the chunk partition the tiles index into.
    if (pl.tn->chunks.size() < tv.firstChunk + tv.chunkCount) return;

    UsdGenCurveBuffer const &term = *pl.term;
    GfRange3f extent;
    uint64_t liveCurves = 0, liveCvs = 0;
    float const *tpx = term.px.empty() ? nullptr : term.px.cdata();
    float const *tpy = term.py.empty() ? nullptr : term.py.cdata();
    float const *tpz = term.pz.empty() ? nullptr : term.pz.cdata();
    size_t const nPx = term.px.size();
    bool const ragged = !term.cvOffsets.empty();
    for (uint32_t i = 0; i < tv.chunkCount; ++i) {
        UsdGenChunkDesc const &cd = pl.tn->chunks[tv.firstChunk + i];
        if (pl.extents) {
            // Fused path: the counts below are the point path's statements
            // verbatim (same inputs, same sums); the extent unions the
            // recorded per-chunk extents in chunk order, which reproduces
            // the sequential pass bitwise for NaN-free points (the gate
            // admits only deform-recorded runs, and deform outputs cannot
            // carry NaN — see the contract on UsdGenChunkView::extentSlot).
            if (ragged && cd.cvCount == 0) {
                for (uint32_t c = 0; c < cd.liveCount; ++c) {
                    size_t const g = size_t(cd.firstCurve) + c;
                    if (g + 1 >= term.cvOffsets.size()) break;
                    uint32_t const p0 = static_cast<uint32_t>(term.cvOffsets[g]);
                    uint32_t const len =
                        static_cast<uint32_t>(term.cvOffsets[g + 1]) - p0;
                    ++liveCurves;
                    liveCvs += len;
                }
            } else {
                liveCurves += cd.liveCount;
                liveCvs += uint64_t(cd.liveCount) * cd.cvCount;
            }
            extent.UnionWith(pl.extents[tv.firstChunk + i].extent);
            continue;
        }
        // Ragged path (03 §1.3): per-curve CV spans come from cvOffsets, not
        // liveCount*cvCount (cvCount == 0 on the ragged path). g indexes the
        // absolute curve so cvOffsets[g] is valid.
        if (ragged && cd.cvCount == 0) {
            for (uint32_t c = 0; c < cd.liveCount; ++c) {
                size_t const g = size_t(cd.firstCurve) + c;
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
        liveCurves += cd.liveCount;
        liveCvs += uint64_t(cd.liveCount) * cd.cvCount;
        if (!tpx || !tpy || !tpz) continue;
        for (uint32_t c = 0; c < cd.liveCount; ++c) {
            size_t const o = size_t(cd.firstCv) + size_t(c) * cd.cvCount;
            for (uint32_t v = 0; v < cd.cvCount; ++v) {
                size_t const p = o + v;
                if (p >= nPx) break;
                extent.ExtendBy(GfVec3f(tpx[p], tpy[p], tpz[p]));
            }
        }
    }
    tv.extent = extent;
    tv.totalLiveCurves = static_cast<uint32_t>(liveCurves);
    tv.totalLiveCvs = static_cast<uint32_t>(liveCvs);
    tv.pointsDirty = true;
    tv.widthsDirty = pl.widthsFlag;
}

}  // namespace

UsdGenScheduler::UsdGenScheduler(int threadLimit)
    : _arena(ResolveThreadLimit(threadLimit)),
      _threadLimit(ResolveThreadLimit(threadLimit)),
      _pool(_threadLimit)
{
    _affinityObserver = ObserveFastCores(_arena, _threadLimit);
}

UsdGenScheduler::~UsdGenScheduler() = default;

int UsdGenScheduler::ThreadLimit() const noexcept { return _threadLimit; }

void UsdGenWorkDispatcher::ParallelFor(
    size_t count, void (*body)(size_t, void *), void *payload, size_t claimChunk)
{
    // Every parallel region runs over the scheduler's worker pool (never
    // pxr work::, which honours the process-global PXR_WORK_THREAD_LIMIT
    // and would serialise under PXR_WORK_THREAD_LIMIT=1, 03 §5.3 caveat).
    // The pool replaced the private arena's tbb::parallel_for here: same
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
    TileCompleted tileCompleted)
{
    TF_UNUSED(generationRequested);  // supersession is checked by the session
    TRACE_FUNCTION();

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
                if (n.op->Type() == sDeformType) extentWriter = int(id);
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
    std::vector<UsdGenChunkExtent> extentScratch;
    if (extentWriter >= 0) extentScratch.resize(extentChunks);
    bool extentSwept = false;

    UsdGenDiagnostics aggregated;
    auto dispatcher = MakeWorkDispatcher();
    UsdGenCurveBuffer const emptyBuf;

    int nTiles = graph.NumTiles();
    std::vector<char> tileTouched(nTiles, 0);
    bool transactionalRun=false;
    for (int pos=0;pos<graph.NodeCount();++pos) {
        UsdGenCompiledNode const &node=graph.Node(pos);
        // Configure is structural, while effective enabled can be driven by
        // a value expression later in preparation. Conservatively transact
        // any graph containing a fallible remapper.
        transactionalRun |= node.op && node.op->RemapsVertexPlanes();
    }

    // Operators evaluate in Default space until a deform: a deform, and every
    // node reading one's output, sees the posed surfaces (graph.Desc()); every
    // other node sees their Default-time rest (graph.RestDesc()), so a posed
    // surface re-captures only the deform and what follows it. Nodes are in
    // topological order; the loop repeats only if an input ever follows its
    // reader.
    std::vector<char> poseSpace(static_cast<size_t>(graph.NodeCount()), 0);
    for (bool changed = true; changed;) {
        changed = false;
        for (int pos = 0; pos < graph.NodeCount(); ++pos) {
            if (poseSpace[size_t(pos)]) continue;
            UsdGenCompiledNode const &node = graph.Node(pos);
            bool posed = node.op && node.op->Type() == sDeformType;
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
    std::vector<char> completed(static_cast<size_t>(graph.NodeCount()), 0);
    auto prepareNode = [&](int pos, NodeExecution *job) -> bool {
        UsdGenCompiledNode &node = graph.Node(pos);
        UsdGenOp &op = *node.op;
        TRACE_SCOPE_DYNAMIC("usdGen prepare " + NodeLabel(node));
        UsdGenDiagnostics nodeDiag;
        bool const hasUp = !node.inputs.empty();
        UsdGenCurveBuffer const &upBuf = hasUp
            ? graph.Node(node.inputs.front()).buffer : emptyBuf;
        job->upstreamInputs.reserve(node.inputs.size());
        for (UsdGenNodeId input : node.inputs) {
            if (input >= static_cast<UsdGenNodeId>(graph.NodeCount())) {
                nodeDiag.Error("UsdGen: invalid compiled geometry input on '" +
                               node.desc->path.GetString() + "'");
                break;
            }
            job->upstreamInputs.push_back(&graph.Node(input).buffer);
        }
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
        cctx.upstreamGeneration = hasUp ? upBuf.topologyVersion : 0;
        cctx.upstreams = job->upstreamInputs.empty() ? nullptr
            : job->upstreamInputs.data();
        cctx.upstreamCount = static_cast<uint32_t>(job->upstreamInputs.size());
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
            UsdGenCurveBuffer const &nbuf = node.buffer;
            bool const outSized = nbuf.totalCvs > 0 &&
                nbuf.px.size() == nbuf.totalCvs &&
                nbuf.py.size() == nbuf.totalCvs &&
                nbuf.pz.size() == nbuf.totalCvs;
            cctx.outPx = outSized ? const_cast<float *>(nbuf.px.cdata()) : nullptr;
            cctx.outPy = outSized ? const_cast<float *>(nbuf.py.cdata()) : nullptr;
            cctx.outPz = outSized ? const_cast<float *>(nbuf.pz.cdata()) : nullptr;
            cctx.outPlaneCvs = outSized ? size_t(nbuf.totalCvs) : 0;
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
            if (!node.expressions.Evaluate(spaceDesc(pos), *node.desc, upBuf,
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
            node.enabled = node.paramView.GroomExpressionValue(TfToken("enabled"), &enabled)
                ? enabled != 0.0 : node.desc->enabled;
        }

        // 02 §6.3: a disabled operator contributes nothing of its own. A
        // disabled GENERATOR publishes an empty curve set, so the description
        // publishes no curves; every other disabled operator passes its
        // upstream through as a CoW alias in prepareEvaluation below.
        if (!node.enabled) {
            job->reCaptured = false;
            node.captureNeeded = false;
            if (op.IsGenerator() &&
                (node.buffer.totalCurves != 0 || node.buffer.totalCvs != 0)) {
                node.buffer = UsdGenCurveBuffer();
                node.buffer.topologyVersion = ++node.topologySeq;
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
            !node.capture->ValidForTopology(upBuf) ? "input changed" : nullptr;
        job->reCaptured = job->recaptureReason != nullptr;
        if (job->reCaptured) {
            TRACE_SCOPE_DYNAMIC("usdGen capture " + NodeLabel(node));
            TRACE_COUNTER_DELTA("usdGen nodes captured", 1);
            cctx.previousCapture = node.capture.get();
            auto cap = op.CreateCapture();
            if (cap && op.Capture(cctx, upBuf, cap.get(), &nodeDiag)) {
                if (cap->OwnsBuffer()) {
                    // valueVersion is monotone per node: the capture's own
                    // buffer counts from zero, and adopting that stamp would
                    // let a downstream capture keyed on the old stamp match.
                    uint64_t const priorValueVersion = node.buffer.valueVersion;
                    node.buffer = cap->Buffer();
                    node.buffer.valueVersion =
                        std::max(node.buffer.valueVersion, priorValueVersion);
                    node.buffer.topologyVersion = ++node.topologySeq;
                    if (hasUp) InheritPerCurve(node.buffer, upBuf);
                }
                node.capture = std::move(cap);
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
            if (node.op->IsGenerator() ||
                node.topoFx == UsdGenTopoFx::CvCount) {
                int const total = static_cast<int>(node.buffer.totalCurves);
                int const cvp = int(node.buffer.totalCvs /
                                    std::max<uint32_t>(1, node.buffer.totalCurves));
                if (graph.Repartition(total, cvp))
                    result.topologyChanged = true;
            } else if (job->reCaptured && node.chunks.empty() && hasUp) {
                UsdGenCompiledNode const &upN = graph.Node(node.input);
                if (!upN.chunks.empty() && upN.buffer.totalCurves > 0 &&
                    graph.Repartition(static_cast<int>(upN.buffer.totalCurves),
                                      static_cast<int>(upN.chunks[0].cvCount)))
                    result.topologyChanged = true;
            }
        }
        for (auto const &e : nodeDiag.errors) aggregated.Error(e);
        for (auto const &w : nodeDiag.warnings) aggregated.Warn(w);
        if (aggregated.HasErrors()) return false;

        if (job->reCaptured && node.capture) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      static_cast<uint8_t>(UsdGenDirtyParameter));
            for (UsdGenNodeId d : node.descendants) {
                if (d >= static_cast<UsdGenNodeId>(graph.NodeCount())) continue;
                std::fill(graph.Node(d).chunkDirty.begin(),
                          graph.Node(d).chunkDirty.end(),
                          static_cast<uint8_t>(UsdGenDirtyParameter));
            }
        }
        job->captureEnd = std::chrono::steady_clock::now();
        // Native time readers move without any authored parameter dirty.
        // Route only their output and descendants; static upstream captures
        // remain reusable during playback.
        if (node.enabled && op.ReadsTime() &&
            (node.lastEvalTime != evalCtx.time ||
             node.lastEvalRate != graph.Desc().timeCodesPerSecond))
            graph.DirtyParameter(node.id, TfToken("__usdGenCookTime"));
        job->evalAll = node.paramValueDigest != node.lastParamDigest ||
                       job->expressionsChanged;
        node.lastParamDigest = node.paramValueDigest;
        node.lastExpressionDigest = node.expressions.Digest();
        for (uint8_t b : node.chunkDirty)
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
        UsdGenCurveBuffer const &upBuf = hasUp
            ? graph.Node(node.inputs.front()).buffer : emptyBuf;

        if (!node.enabled && op.IsGenerator()) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }
        if (!node.enabled && hasUp) {
            UsdGenCurveBuffer &buf = node.buffer;
            buf.px = upBuf.px; buf.py = upBuf.py; buf.pz = upBuf.pz;
            buf.rest = upBuf.rest;
            buf.width = upBuf.width; buf.hairT = upBuf.hairT;
            AliasPerCurve(buf, upBuf);
            AliasExtraPlanes(buf, upBuf);
            buf.totalCurves = upBuf.totalCurves;
            buf.totalCvs = upBuf.totalCvs;
            buf.topologyVersion = upBuf.topologyVersion;
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }
        if ((!job->anyChunkDirty && !job->evalAll) || !node.capture) {
            std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                      UsdGenDirtyNone);
            return true;
        }

        PrepareNodeForEval(node, op, upBuf, hasUp);
        EnsureWritablePlanes(node, op);
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
        pl.deferCompletion = op.RemapsVertexPlanes();
        if (pl.deferCompletion) {
            pl.pointScratch.resize(node.buffer.totalCvs);
            pl.scalarScratch.resize(node.buffer.totalCvs);
        }
        pl.didEval.assign(node.chunks.size(), 0);
        if (pos == extentWriter && extentWriter >= 0) {
            pl.extentBase = extentScratch.data();
            pl.extentCap = extentChunks;
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
            std::vector<int> frontier;
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

            std::vector<std::unique_ptr<NodeExecution>> jobs;
            jobs.reserve(frontier.size());
            for (int pos : frontier) {
                jobs.push_back(std::make_unique<NodeExecution>());
                if (!prepareNode(pos, jobs.back().get())) {
                    result.diagnostics = std::move(aggregated);
                    return result;
                }
            }
            // Capture can repartition the graph. Keep completion counters and
            // the eventual interleave sweep aligned with the new tile plan.
            if (graph.NumTiles() != nTiles) {
                nTiles = graph.NumTiles();
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

            std::vector<ChunkExecution> work;
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
            std::unique_ptr<std::atomic<uint32_t>[]> terminalRemaining;
            if (tileCompleted && !transactionalRun && nTiles > 0) {
                for (auto const &job : jobs) {
                    if (!job->shouldSweep ||
                        job->sweep.node->id != graph.TerminalNodeId()) continue;
                    terminalRemaining = std::make_unique<std::atomic<uint32_t>[]>(nTiles);
                    for (int t = 0; t < nTiles; ++t)
                        terminalRemaining[t].store(0, std::memory_order_relaxed);
                    for (ChunkExecution const &item : work) {
                        if (item.payload != &job->sweep) continue;
                        UsdGenTileId const tile = job->sweep.node->chunks[item.chunk].tile;
                        if (tile < static_cast<UsdGenTileId>(nTiles))
                            terminalRemaining[tile].fetch_add(1, std::memory_order_relaxed);
                    }
                    job->sweep.onChunkCompleted = [&, node=job->sweep.node](size_t chunk) {
                        UsdGenTileId const tile = node->chunks[chunk].tile;
                        if (tile >= static_cast<UsdGenTileId>(nTiles)) return;
                        if (terminalRemaining[tile].fetch_sub(1, std::memory_order_acq_rel) == 1)
                            tileCompleted(graph.Tiles()[tile], node->buffer);
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
                    node.buffer.valueVersion += 1;
                    for (size_t c = 0; c < node.chunks.size(); ++c)
                        if (node.chunks[c].tile < tileTouched.size())
                            tileTouched[node.chunks[c].tile] = 1;
                    std::fill(node.chunkDirty.begin(), node.chunkDirty.end(),
                              UsdGenDirtyNone);
                    auto const end = std::chrono::steady_clock::now();
                    UsdGenNodeRunStats st;
                    st.id = static_cast<UsdGenNodeId>(frontier[j]);
                    st.captureMs = std::chrono::duration<double, std::milli>(
                        job.captureEnd - job.captureStart).count();
                    st.evalMs = 0.0;
                    st.chunksEvaluated = node.chunks.size();
                    result.nodeStats.push_back(st);
                }
                if (job.shouldSweep) {
                    bool wrote = job.reCaptured;
                    for (uint8_t b : job.sweep.didEval)
                        if (b) { wrote = true; break; }
                    if (wrote) {
                        node.buffer.valueVersion += 1;
                        for (size_t c = 0; c < node.chunks.size(); ++c)
                            if (job.sweep.didEval[c] &&
                                node.chunks[c].tile < tileTouched.size())
                                tileTouched[node.chunks[c].tile] = 1;
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
                    result.nodeStats.push_back(st);
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
                if (node.op->ReadsTime() && node.enabled) {
                    node.lastEvalTime = evalCtx.time;
                    node.lastEvalRate = graph.Desc().timeCodesPerSecond;
                }
                --remaining;
            }
        }
    }
    UsdGenCurveBuffer const &term = graph.Output();
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
        auto sameBacking = [](auto const &a, auto const &b) {
            return a.size() == b.size() && (a.empty() || a.cdata() == b.cdata());
        };
        bool ok = tn.chunks.size() == extentChunks &&
                  wn.chunks.size() == extentChunks &&
                  sameBacking(term.px, wn.buffer.px) &&
                  sameBacking(term.py, wn.buffer.py) &&
                  sameBacking(term.pz, wn.buffer.pz) &&
                  sameBacking(term.cvOffsets, wn.buffer.cvOffsets);
        for (size_t i = 0; ok && i < extentChunks; ++i) {
            UsdGenChunkDesc const &tc = tn.chunks[i];
            UsdGenChunkDesc const &wc = wn.chunks[i];
            UsdGenChunkExtent const &s = extentScratch[i];
            ok = tc.liveCount == tc.curveCount && wc.liveCount == wc.curveCount &&
                 s.firstCurve == wc.firstCurve && s.curveCount == wc.curveCount &&
                 s.liveCount == wc.liveCount && s.firstCv == wc.firstCv &&
                 s.cvCount == wc.cvCount && wc.firstCurve == tc.firstCurve &&
                 wc.curveCount == tc.curveCount && wc.liveCount == tc.liveCount &&
                 wc.firstCv == tc.firstCv && wc.cvCount == tc.cvCount;
        }
        if (ok) fusedExtents = extentScratch.data();
    }
    if (nTiles > 0) {
        TRACE_SCOPE("usdGen interleave tiles");
        InterleavePayload ip;
        ip.graph = &graph;
        ip.tn = &tn;
        ip.term = &term;
        ip.touched = &tileTouched;
        ip.widthsFlag = termOp &&
            (termOp->PlanesTouched() & UsdGenOp::kPlaneWidths) != 0;
        ip.extents = fusedExtents;
        dispatcher.ParallelFor(size_t(nTiles), InterleaveTile, &ip);
    }

    if (tileCompleted && transactionalRun)
        for (int tile=0;tile<nTiles;++tile)
            if (tileTouched[size_t(tile)])
                tileCompleted(graph.Tiles()[tile],term);

    result.terminalOutput = &graph.Output();
    result.tiles = graph.Tiles();
    result.diagnostics = std::move(aggregated);
    return result;
}

}  // namespace usdGen
